"""NIfTI import and export (spec §3, §4, §7).

The one thing this converter must not get wrong is the **handedness**.  NIfTI's
affine is RAS+; MEDH5 grids declare their own ``coord_system`` and default to
LPS, the DICOM convention.  Converting between them is a sign flip on the first
two world axes --- applied to the affine, never to the voxels.

So the choice is made explicitly and recorded: ``coord_system="LPS"`` (the
default) converts and says so in the report; ``coord_system="RAS"`` keeps
NIfTI's own frame and stores that on the grid.  Either is correct; silently
picking one and not writing it down is how a segmentation ends up mirrored
against the image it was drawn on, which no unit test catches because both
volumes flip together.

Axis order is likewise stated rather than assumed.  NIfTI arrays are
``(x, y, z)`` fastest-first; MEDH5 spatial axes are trailing and, for a 3-D
volume written by a medical tool, conventionally ``(z, y, x)``.  ``transpose``
controls it, defaults to reordering, and the resulting axis names are written
onto the grid so a reader never has to infer them.

The converter is the engine's (``medh5::convert::nifti``), so this module,
the command line and the Rust crate read and write a file the same way, and
none of them needs nibabel: the header codec, gzip and every rule are
2.1.0's, held to what 2.1.0 made of a recorded corpus
(``tests/data/nifti-parity``).
"""

from __future__ import annotations

import importlib
import os
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.errors import MEDH5ValidationError
from medh5.geometry import COORD_SYSTEMS, MM_PER_UNIT
from medh5.io.report import ConversionReport

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.labels import LabelSet

RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
"""World-axis sign flip. ``LPS = RAS_TO_LPS @ RAS``, and it is its own inverse."""

RAS_TO_LPS_2D = np.diag([-1.0, -1.0, 1.0])
"""The same flip for a 2-D grid (§3.6).

A 2-D grid's two world axes are the first two of the frame it was reduced
from --- an import keeps a plane only when it lies in the x-y plane at
z = 0 --- and those are exactly the two axes RAS↔LPS flips.
"""

NIFTI_SUFFIXES: tuple[str, ...] = tuple(_core.NIFTI_SUFFIXES)
"""The names an export writes a NIfTI file under, and the only names whose
sidecar and ``.bval`` this module finds beside them."""


def require_nibabel() -> Any:
    """Import nibabel, or raise an ``ImportError`` saying how to install it.

    No converter needs it since 2.2.0, when NIfTI moved into the engine, and
    the ``nifti`` extra no longer installs it; it is kept for code that used
    it to reach nibabel itself.
    """
    try:
        return importlib.import_module("nibabel")
    except ImportError as exc:
        raise ImportError(
            "nibabel is not installed, and medh5 no longer needs it for NIfTI; "
            "install it with: pip install nibabel"
        ) from exc


def convert_world(
    affine: npt.ArrayLike, *, source: str, target: str
) -> npt.NDArray[np.float64]:
    """Re-express a 3-D index→world affine between RAS and LPS."""
    for name in (source, target):
        if name not in COORD_SYSTEMS:
            raise MEDH5ValidationError(
                f"unknown coordinate system {name!r}; expected one of "
                f"{list(COORD_SYSTEMS)}"
            )
    matrix = np.asarray(affine, dtype=np.float64)
    if source == target:
        return matrix
    if matrix.shape == (3, 3):
        # A 2-D grid, which §3.6 gives an S = 2 affine.  Both importers accept
        # 2-D input and both exporters called this, so refusing here meant no
        # 2-D sample could be written back out at all --- a radiograph could go
        # in and never come out.
        return np.asarray(RAS_TO_LPS_2D @ matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise MEDH5ValidationError(
            f"RAS↔LPS conversion is defined for 2-D and 3-D affines; got {matrix.shape}"
        )
    return np.asarray(RAS_TO_LPS @ matrix, dtype=np.float64)


def write_nifti(
    array: npt.NDArray[Any],
    affine: npt.ArrayLike,
    path: str | os.PathLike[str],
    *,
    rescale: tuple[float, float] | None = None,
    units: str = "mm",
    time_values: Sequence[float] | None = None,
    time_units: str | None = None,
    leading: str | None = None,
    channels: Mapping[str, Any] | None = None,
    report: ConversionReport | None = None,
) -> Path:
    """Write one NIfTI file, recording *rescale* in the header (§4.2).

    Every export goes through here, because the alternative is what shipped:
    ``image.read()`` returns **stored** values and the exporters wrote them
    with no ``scl_slope``/``scl_inter``, so a CT imported from DICOM with
    intercept −1024 left every voxel 1024 HU too high in the exported file and
    nothing in it said so.  NIfTI has the two fields for exactly this, and
    every reader --- nibabel, SimpleITK, and therefore nnU-Net --- applies them
    on load, so writing them makes the stored volume mean what it meant here.

    The header says what the numbers are in, too.  The exports wrote no units,
    so a grid in metres read back as millimetres and every frame series as one
    second apart from time 0.  *affine* is in the grid's *units*: lengths are
    written in millimetres (a grid in ``m`` or ``um`` scaled to them), and an
    uncalibrated ``px`` grid says so with NIfTI's "unknown".  A fourth axis
    with *time_values* evenly spaced becomes ``pixdim[4]`` and ``toffset``;
    uneven frames, which one temporal step cannot state, are written to a BIDS
    sidecar's ``VolumeTiming`` beside the file --- which :func:`read_nifti`
    reads back --- and the header's step is left at 0, which says nothing.

    A fourth axis that is a *channel* axis (*leading*, §3.6) is said to be one
    with what its channels mean taken from
    *channels* --- the image's per-channel acquisition values.  Written as a
    plain fourth axis it read back as time, with invented frame times.

    The file owns the statements about its volumes beside it --- the
    sidecar's per-volume fields and a ``.bval`` --- and an overwrite replaces
    them: the new image is serialised first, the old statements are withdrawn
    before it replaces the old image, and the new ones written after, so no
    reader ever pairs the image with another export's timeline, and an export
    that fails before its image is whole changes nothing.

    *path* names a ``.nii`` or ``.nii.gz`` file, and anything else is refused
    before a file is written.
    """
    written = _core.nifti_write(
        np.asarray(array),
        _matrix(affine),
        os.fspath(path),
        report=report,
        rescale=None if rescale is None else (float(rescale[0]), float(rescale[1])),
        units=units,
        time_values=None if time_values is None else [float(v) for v in time_values],
        time_units=time_units,
        leading=leading,
        channels=dict(channels) if channels else None,
    )
    return Path(written)


def _matrix(affine: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """*affine* as a 2-D float64 array, whose shape the engine checks."""
    matrix = np.asarray(affine, dtype=np.float64)
    return matrix if matrix.ndim == 2 else matrix.reshape(1, -1)


def _geometry(fields: dict[str, Any]) -> dict[str, Any]:
    """The engine's geometry JSON as 2.1.0's ``read_nifti`` returned it."""
    fields["notes"] = [tuple(note) for note in fields["notes"]]
    fields["shape"] = tuple(fields["shape"])
    fields["axis_order"] = tuple(fields["axis_order"])
    if fields["rescale"] is not None:
        fields["rescale"] = tuple(fields["rescale"])
    return fields


def _array(buffer: bytearray, dtype: Any, shape: Sequence[int]) -> npt.NDArray[Any]:
    return np.frombuffer(buffer, dtype=np.dtype(dtype)).reshape(tuple(shape))


def read_nifti(
    path: str | os.PathLike[str],
    *,
    coord_system: str = "LPS",
    transpose: bool = True,
    fourth_axis: str = "auto",
    assume_geometry: bool = False,
    plane_as_volume: bool = False,
) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """One NIfTI file as ``(array, geometry)`` in MEDH5 conventions.

    *fourth_axis* decides what a 4-D series' extra axis is --- ``"time"`` for
    cine, DCE and 4-D CT, ``"channel"`` for multi-b-value DWI and multi-echo
    (§3.6).  ``"auto"`` reads the answer out of the file where it can and
    reports a guess where it cannot.

    *assume_geometry* allows a file that declares **no** spatial mapping
    (``sform_code == qform_code == 0``) to be imported anyway, taking the
    fallback grid from ``pixdim``.  It is off by default because that fallback is an
    invented grid, and a grid this library invented is indistinguishable
    downstream from one a scanner measured (§3.3).

    A 2-D file is a 2-D grid (§3.6), which only a plane at z = 0 in the
    world's x-y plane can be; any other is refused.  *plane_as_volume* reads
    every 2-D file as ``(x, y, 1)`` instead --- the one-slice 3-D grid §3.6
    makes of a plane that lies elsewhere --- keeping the full affine, whose
    third column becomes the third axis: it must be non-zero and orthogonal
    to the first two, since no normal or thickness is invented.  The
    geometry's ``plane_as_volume`` then records the plane's ``z`` and
    ``normal``, and is ``None`` for a file that was not reshaped.

    The returned geometry carries ``notes``: what the reader had to decide from
    ambiguous headers, for the caller to put in its conversion report.
    """
    buffer, dtype, shape, geometry = _core.nifti_read(
        os.fspath(path),
        coord_system=coord_system,
        transpose=transpose,
        fourth_axis=fourth_axis,
        assume_geometry=assume_geometry,
        plane_as_volume=plane_as_volume,
    )
    return _array(buffer, dtype, shape), _geometry(geometry)


def _read_stored(
    path: str | os.PathLike[str],
) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """One file as it is stored: the voxels in the header's ``(x, y, z, ...)``
    shape, unscaled, and the header's codes, best affine (RAS, in the file's
    unit), spatial unit and scaling --- for a reader with conventions of its
    own (ITK displacement fields)."""
    buffer, dtype, shape, fields = _core.nifti_read_stored(os.fspath(path))
    data = np.frombuffer(buffer, dtype=np.dtype(dtype)).reshape(tuple(shape), order="F")
    return data, fields


def _pairs(paths: Mapping[str, str | os.PathLike[str]] | None) -> list[tuple[str, str]]:
    return [(name, os.fspath(p)) for name, p in (paths or {}).items()]


def from_nifti(
    images: Mapping[str, str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    masks: Mapping[str, str | os.PathLike[str]] | None = None,
    sample_id: str | None = None,
    subject_id: str | None = None,
    modalities: Mapping[str, str] | None = None,
    coord_system: str = "LPS",
    transpose: bool = True,
    fourth_axis: str = "auto",
    value_units: Mapping[str, str] | None = None,
    codec: str = "balanced",
    annotated_classes: Sequence[str] | str = "all_given",
    assume_geometry: bool = False,
    report: ConversionReport | None = None,
    plane_as_volume: bool = False,
    label_set: LabelSet | str | os.PathLike[str] | None = None,
) -> ConversionReport:
    """Write one sample from a set of co-registered NIfTI volumes.

    Every volume must share one grid.  A disagreement is refused rather than
    resampled: resampling changes the data, and a converter that does it
    silently produces a file whose labels no longer sit on the voxels they were
    drawn on.  Resample first, deliberately, with a tool that records it.

    *plane_as_volume* imports every 2-D image and mask as a one-slice 3-D
    grid, ``(x, y, 1)``, where it lies (:func:`read_nifti`); each is a
    ``plane_as_volume`` decision in the report.

    Without *label_set*, the masks' classes are minted from their names, in
    sorted order, so a class's id depends on which masks a case has.
    *label_set* --- a :class:`~medh5.labels.LabelSet`, or a JSON file
    :func:`medh5.labels.load_file` reads --- fixes them: each mask is the
    class of the key its name makes (or of its name), the set is the
    sample's, and a mask it has no class for is refused (E402).

    A class *annotated_classes* names that no mask supplies was examined and
    found absent (§11.3): minted after the masks' classes, with a
    ``label_set`` decision, or, with *label_set*, one the set must hold
    (E402).
    """
    from medh5.labels import LabelSet, load_file

    log = report or ConversionReport(converter="from-nifti")
    given = (
        label_set
        if label_set is None or isinstance(label_set, LabelSet)
        else load_file(label_set)
    )
    _core.nifti_from(
        _pairs(images),
        os.fspath(out),
        report=log,
        masks=_pairs(masks),
        sample_id=sample_id,
        subject_id=subject_id,
        modalities=list((modalities or {}).items()),
        coord_system=coord_system,
        transpose=transpose,
        fourth_axis=fourth_axis,
        value_units=list((value_units or {}).items()),
        codec=codec,
        annotated_classes=annotated_classes,
        assume_geometry=assume_geometry,
        plane_as_volume=plane_as_volume,
        label_set=given,
    )
    return log


SECONDS_PER_UNIT = {"s": 1.0, "ms": 1e-3}


def _lengths(geometry: Mapping[str, Any], key: str) -> npt.NDArray[np.float64]:
    """*key* (``spacing`` or ``origin``) in millimetres where the unit has a size."""
    values = np.asarray(geometry[key], dtype=np.float64)
    return values * MM_PER_UNIT.get(str(geometry.get("units") or "mm"), 1.0)


def _geometry_mismatch(
    first: Mapping[str, Any], other: Mapping[str, Any]
) -> str | None:
    """The first of units, spacing, origin and direction two geometries
    disagree on, or ``None``.

    Lengths are compared in millimetres, so one grid stated in metres and in
    millimetres agrees with itself; an uncalibrated (``px``) grid agrees only
    with another.  Comparing the raw numbers took a grid in metres for one
    1000 times smaller, and a mask with no origin check anywhere it pleased.
    """
    pixels = [str(g.get("units") or "mm") == "px" for g in (first, other)]
    if pixels[0] != pixels[1]:
        return "units"
    for key in ("spacing", "origin", "direction"):
        if key == "direction":
            ours = np.asarray(first[key], dtype=np.float64)
            theirs = np.asarray(other[key], dtype=np.float64)
        else:
            ours, theirs = _lengths(first, key), _lengths(other, key)
        if ours.shape != theirs.shape or not np.allclose(ours, theirs, atol=1e-4):
            return key
    return None


def _seconds(geometry: Mapping[str, Any]) -> npt.NDArray[np.float64] | None:
    values = geometry.get("time_values")
    if values is None:
        return None
    unit = str(geometry.get("time_units") or "s")
    return np.asarray(values, dtype=np.float64) * SECONDS_PER_UNIT.get(unit, 1.0)


def _same_grid(
    first: dict[str, Any], other: dict[str, Any], name: str, log: ConversionReport
) -> dict[str, Any]:
    """Refuse volumes that do not share a grid (spec §3.2).

    Uncoded, deliberately.  §15.2's table describes conditions found *in a MEDH5
    file*, and these are two NIfTI volumes that are not one yet: the shape
    mismatch is not `E202` (an image disagreeing with its grid --- neither of
    these is a grid), and the geometry mismatch is not `E101` (a reference to a
    grid that does not exist --- nothing here is referenced). Reporting them
    under those codes told anything branching on the code an untrue story about
    what had gone wrong.

    A grid is its timing as much as its space (§3.2), so frame times two
    volumes both *state* must agree; only space was compared, and the first
    file's times stood for every other's.  Where one states them and the other
    does not, the stated ones are the grid's.
    """
    if tuple(first["shape"]) != tuple(other["shape"]):
        raise MEDH5ValidationError(
            f"{name!r} has shape {other['shape']}, but the first volume has "
            f"{first['shape']}; resample before converting"
        )
    key = _geometry_mismatch(first, other)
    if key is not None:
        raise MEDH5ValidationError(
            f"{name!r} disagrees with the first volume on {key}; resample "
            "before converting rather than letting a converter do it silently"
        )
    ours, theirs = _seconds(first), _seconds(other)
    if ours is None or theirs is None or np.allclose(ours, theirs, atol=1e-6):
        return first
    if first.get("time_measured") and other.get("time_measured"):
        raise MEDH5ValidationError(
            f"{name!r} states frame times {other['time_values']} "
            f"{other['time_units']}, the first volume {first['time_values']} "
            f"{first['time_units']}; one grid has one set of `time_values` "
            "(§3.2), so convert them separately"
        )
    if not other.get("time_measured"):
        return first
    log.decision(
        "time_values",
        f"{name!r} states its frame times and the first volume does not, so "
        "the grid takes them",
        {"from": name, "time_units": other["time_units"]},
    )
    return {
        **first,
        "time_values": other["time_values"],
        "time_units": other["time_units"],
        "time_measured": True,
    }


def to_nifti(
    path: str | os.PathLike[str],
    image_id: str,
    out: str | os.PathLike[str],
    *,
    physical: bool = True,
    annotation: str | None = None,
    class_key: int | str | None = None,
    report: ConversionReport | None = None,
) -> Path:
    """Export one image, or one class of one annotation, as a NIfTI file.

    The affine is converted back to RAS+ and the axes back to ``(x, y, z)``, so
    the file opens in FSL, ITK-SNAP or 3D Slicer at the location it came from.

    ``physical=False`` writes the **stored** values, and then the image's
    rescale is written into the header, so the numbers a conforming reader
    gets are the physical ones either way.  A label volume has no rescale and
    is written as it is.

    An annotation is written on **its own** grid, whatever *image_id* is on:
    its voxels are that grid's, and the image's affine put a PET-grid mask at
    the CT's origin.  *report*, when given, collects what the header could not
    say as the grid does (see :func:`write_nifti`).

    The whole annotation is its labelmap, where an ignored voxel (§7.7) is the
    annotation's ``ignore_id``.  One class (*class_key*) is a 0/1 mask, which
    has no value for "ignored", and its 0 says "examined, absent": an
    annotation with an ignore region is refused rather than exported as one,
    since the mask would turn every ignored voxel into a negative.  For the
    same reason the class must be one the annotation examined
    (``annotated_class_ids``): a class nobody looked for went out as an empty
    mask, a verified negative to whoever imported it back.

    *out* is a ``.nii`` or ``.nii.gz`` file, checked before anything is read.
    """
    log = ConversionReport(converter="to-nifti")
    try:
        written = _core.nifti_to(
            os.fspath(path),
            image_id,
            os.fspath(out),
            report=log,
            physical=physical,
            annotation=annotation,
            class_key=class_key,
        )
    finally:
        # Flattening an overlap is the labelmap's warning, as it was when
        # this export called `labelmap()`; the rest is the report's.
        kept = []
        for note in log.notes:
            if note.kind == "overlap" and note.severity == "warning":
                warnings.warn(note.message, stacklevel=2)
            else:
                kept.append(note)
        if report is not None:
            report.notes.extend(kept)
    return Path(written)


def for_export(
    grid: Any, data: npt.NDArray[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[np.float64]]:
    """An array and affine in MEDH5 conventions, as NIfTI wants them.

    MEDH5 leads with time and trails with ``(z, y, x)``; NIfTI is the other way
    round, so the spatial axes are reversed and the affine's columns with them.
    A 2-D plane is *not* reversed --- ``read_nifti`` does not transpose one
    either, so reversing here would mirror every radiograph --- and its 3x3
    affine is embedded in the 4x4 a NIfTI file carries (§3.6).

    Shared by both exporters, because both had their own copy of the reversal
    and only one of them was ever fixed.
    """
    exported, affine = _core.nifti_for_export(grid, np.asarray(data))
    return exported, np.asarray(affine, dtype=np.float64)


def import_seg_nifti(
    path: str | os.PathLike[str],
    masks: Mapping[str, str | os.PathLike[str]],
    *,
    ann_id: str = "seg",
    grid: str | None = None,
    coord_system: str | None = None,
    transpose: bool = True,
    annotated_classes: Sequence[str] | str = "all_given",
    assume_aligned: bool = False,
    assume_geometry: bool = False,
    report: ConversionReport | None = None,
    plane_as_volume: bool = False,
) -> ConversionReport:
    """Add NIfTI masks to an existing sample, checking they sit on its grid.

    A mask must match the grid in shape *and* geometry --- spacing, origin and
    direction, lengths compared in millimetres.  Shape alone let a mask
    translated 10 cm, rotated or mirrored onto voxels nobody drew on, with a
    warning at most.  A mismatch is refused unless *assume_aligned*, which
    takes the voxels as the grid's anyway and records that as a guess: for a
    header known to be wrong, never for a mask from another grid, which needs
    resampling (by a tool that records it).  The comparison is in the grid's
    coordinate system; *coord_system* may only repeat it.

    *assume_geometry* and the report's guesses are :func:`from_nifti`'s: a
    mask declaring no spatial mapping is refused unless the fallback is
    taken on purpose, and what the reader assumed is in the report.
    *plane_as_volume* reads a 2-D mask as ``(x, y, 1)``, so it can sit on
    the one-slice 3-D grid a plane off z = 0 was imported as.

    A class *annotated_classes* names that no mask supplies was examined and
    found absent (§11.3): minted after the masks' classes when the sample has
    no label set, else one its label set must hold (E402).
    """
    log = report or ConversionReport(converter="import-seg-nifti")
    _core.nifti_import_seg(
        os.fspath(path),
        _pairs(masks),
        report=log,
        ann_id=ann_id,
        grid=grid or None,
        coord_system=coord_system,
        transpose=transpose,
        annotated_classes=annotated_classes,
        assume_aligned=assume_aligned,
        assume_geometry=assume_geometry,
        plane_as_volume=plane_as_volume,
    )
    return log


__all__ = [
    "COORD_SYSTEMS",
    "RAS_TO_LPS",
    "convert_world",
    "from_nifti",
    "import_seg_nifti",
    "read_nifti",
    "require_nibabel",
    "to_nifti",
]
