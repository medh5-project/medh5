"""DICOM import (spec §3, §4, §12).

A DICOM series is a pile of slices that *claims* to be a volume.  Turning it
into a grid means answering three questions the files do not answer directly,
and this converter answers each one by measurement rather than by assumption:

**Slice order.**  ``InstanceNumber`` is unreliable --- it is a display hint and
routinely disagrees with geometry.  Slices are ordered by their projection onto
the normal of ``ImageOrientationPatient``, which is the only ordering the
geometry itself defines.

**Spacing along the normal.**  ``SliceThickness`` is the thickness of the *slab*,
not the distance between slice origins; a series with 5 mm slices at 2.5 mm
increments is common and using the thickness would halve the volume's extent.
The z spacing is measured from the slice positions, and a series with a slice
farther than a tenth of a slice from where the regular stack puts it is refused
rather than averaged --- an irregular stack is not a grid, and pretending
otherwise puts every label at the wrong depth.

**Values.**  ``RescaleSlope``/``RescaleIntercept`` are stored, not applied
(§4.2), so ``read()`` returns what the file stored and ``read(physical=True)``
returns the rescaled values --- HU, for a CT whose ``RescaleType`` names no
other quantity.  A converter that bakes the rescale in makes the two indistinguishable
afterwards.  The one exception is asked for: a series whose slices each state
their own rescale (ordinary in PET) has no one modality LUT to store, and
``per_slice_rescale="apply"`` stores its physical values instead, saying so.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5._optional import require
from medh5.curation import ID_SOURCE
from medh5.errors import MEDH5ValidationError
from medh5.io._common import sanitize_ascii_stem
from medh5.io._frames import (
    _first_item,
    _group_value,
    _groups,
    _orientation,
    _pixel_spacing,
    _position,
)
from medh5.io.grouping import (
    FALLBACK_PREFIX,
    Occasion,
    SubjectGroup,
    group_by_subject,
    note_instance_ids,
    output_name,
)
from medh5.io.report import ConversionReport

SPACING_TOLERANCE = 0.1
"""How far a slice may sit from where the regular stack puts it, as a fraction
of the slice spacing, and never less than 0.01 mm.

Measured per slice against the whole stack, not per gap: position rounding is
per slice and does not accumulate, where drift does.  Each gap was held to 1 %
of the median gap, so 401 slices whose first 200 gaps were 0.9 % long and the
rest as short drifted 4.5 mm --- 1.8 slices --- by the middle and passed,
while positions only rounded to two decimals were refused.  A tenth of a
slice keeps every voxel well inside the slab it was acquired in, and the
0.01 mm floor is twice the rounding of a position written to two decimals.
The DICOM SEG reader holds the planes it stacks to the same bound."""

_POSITION_FLOOR_MM = 0.01
"""The least slice-position tolerance, in millimetres
(:data:`SPACING_TOLERANCE`)."""


_PER_SLICE_RESCALES = ("refuse", "apply")
"""What :func:`from_dicom` does with a series whose slices state different
rescales: refuse it, or apply each slice's and store physical values."""


def _per_slice_rescale(mode: str) -> str:
    if mode not in _PER_SLICE_RESCALES:
        raise ValueError(
            f"per_slice_rescale must be one of {', '.join(_PER_SLICE_RESCALES)}, "
            f"not {mode!r}"
        )
    return mode


def _position_tolerance(spacing: float) -> float:
    """How far, in millimetres, a slice may sit from where a regular stack of
    *spacing* millimetres puts it: the one bound both DICOM readers apply."""
    return max(SPACING_TOLERANCE * abs(float(spacing)), _POSITION_FLOOR_MM)


STORED_TAGS = (
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "KVP",
    "XRayTubeCurrent",
    "ExposureTime",
    "ConvolutionKernel",
    "SliceThickness",
    "RepetitionTime",
    "EchoTime",
    "FlipAngle",
    "MagneticFieldStrength",
    "SequenceName",
    "ContrastBolusAgent",
    "PatientPosition",
    "BodyPartExamined",
    "PhotometricInterpretation",
)
"""Acquisition parameters copied to ``/meta → acquisition`` (§4.5).

An explicit list, because §11.4 forbids copying DICOM tags wholesale: a bulk
copy carries names, dates and identifiers into a file that claims to be
de-identified.
"""


def require_pydicom() -> Any:
    """Import pydicom, or raise an ``ImportError`` naming the extra to install."""
    return require("pydicom", extra="dicom", purpose="DICOM conversion")


def _dcmread(path: str | os.PathLike[str], **options: Any) -> Any:
    """``pydicom.dcmread``, refusing by name a file that is not DICOM.

    pydicom's own error names neither the file nor the reader, and is no
    ``MEDH5Error``: a caller catching every input problem as one missed it,
    and the command line printed a traceback.
    """
    require_pydicom()
    from pydicom import dcmread
    from pydicom.errors import InvalidDicomError

    try:
        return dcmread(os.fspath(path), **options)
    except InvalidDicomError as exc:
        raise MEDH5ValidationError(
            f"{os.fspath(path)} is not a DICOM file: {exc}"
        ) from None


@dataclass(slots=True)
class Series:
    """One DICOM series, resolved into a volume and its geometry."""

    series_uid: str
    study_uid: str
    modality: str
    patient_id: str | None
    study_date: str | None
    paths: list[str] = field(default_factory=list)
    description: str = ""
    frame_uid: str | None = None
    birth_date: str | None = None
    """``PatientBirthDate``: compared across the studies of one PatientID, and
    never written to the file (§11.4)."""
    sex: str | None = None
    """``PatientSex``, compared the same way."""
    frames: list[tuple[str, int]] = field(default_factory=list)
    """For a multi-frame object, each slice as ``(file, frame index)``, the
    index 0-based within its file, in the order the object holds them; empty
    for a classic series, a file per slice."""
    stack: str | None = None
    """Which stack of its multi-frame object this series is, where the object
    holds several --- its ``StackID``, and the values of any other dimension
    that tells the stacks apart --- and ``None`` otherwise."""

    @property
    def n_slices(self) -> int:
        return len(self.frames) if self.frames else len(self.paths)

    def to_json(self) -> dict[str, Any]:
        return {
            "series_uid": self.series_uid,
            "study_uid": self.study_uid,
            "modality": self.modality,
            "patient_id": self.patient_id,
            "study_date": self.study_date,
            "slices": self.n_slices,
            "description": self.description,
            "frame_uid": self.frame_uid,
        } | ({"stack": self.stack} if self.stack is not None else {})

    def __repr__(self) -> str:
        return (
            f"Series({self.modality}, {self.n_slices} slices, "
            f"{self.description or self.series_uid[:16]})"
        )


_ENHANCED_SOP_CLASSES = (
    "EnhancedCTImageStorage",
    "EnhancedMRImageStorage",
    "EnhancedPETImageStorage",
    "LegacyConvertedEnhancedCTImageStorage",
    "LegacyConvertedEnhancedMRImageStorage",
    "LegacyConvertedEnhancedPETImageStorage",
)
"""Multi-frame image objects whose geometry lives in per-frame functional groups.

An enhanced object carries every slice as a frame, and each frame's
`ImagePositionPatient` in its functional groups rather than at the top level
(PS3.3 C.7.6.16).  Its frames are split into stacks, and each stack is
imported as the classic series it would have been.
"""


_UNSUPPORTED_SOP_CLASSES = ("EnhancedUSVolumeStorage",)
"""Enhanced objects a user would expect to be images that this importer does
not read: an ultrasound volume places its frames through a volume transform,
not a plane position per frame.  Skipped with a warning naming them."""


_STACK_ID = 0x00209056
"""``StackID``: the stack a frame belongs to (C.7.6.16.2.2)."""


_ACQUISITIONS = frozenset(
    {
        0x00209128,  # TemporalPositionIndex
        0x0020930D,  # TemporalPositionTimeOffset
        0x00209153,  # NominalCardiacTriggerDelayTime
        0x00209154,  # NominalCardiacTriggerTimePriorToRPeak
        0x00209241,  # NominalPercentageOfCardiacPhase
        0x00189236,  # CardiacCyclePosition
        0x00209255,  # NominalRespiratoryTriggerDelayTime
        0x00189214,  # RespiratoryCyclePosition
        0x00189082,  # EffectiveEchoTime
        0x00189087,  # DiffusionBValue
        0x00189089,  # DiffusionGradientOrientation
        0x00181314,  # FlipAngle
        0x00180080,  # RepetitionTime
        0x00189079,  # InversionTimes
        0x00189090,  # VelocityEncodingDirection
        0x00189257,  # ASLContext
        0x00089208,  # ComplexImageComponent
        0x00189344,  # ContrastBolusAgentPhase
    }
)
"""Dimensions each of whose values is one acquisition of the planes --- a
time point, a cardiac or respiratory phase, an echo, a diffusion weighting,
a flip angle or repetition time, a flow encoding, a labelling, a complex
component, a contrast phase --- so they tell stacks apart however many
planes a value holds (C.7.6.16.2.2)."""


_ACQUISITION_GROUPS = frozenset(
    {
        0x00209310,  # TemporalPositionSequence
        0x00189118,  # CardiacSynchronizationSequence
        0x00209253,  # RespiratorySynchronizationSequence
        0x00189114,  # MREchoSequence
        0x00189117,  # MRDiffusionSequence
        0x00189112,  # MRTimingAndRelatedParametersSequence
        0x00189115,  # MRModifierSequence
        0x00189197,  # MRVelocityEncodingSequence
        0x00189251,  # MRArterialSpinLabelingSequence
        0x00189152,  # MRMetaboliteMapSequence
        0x00189042,  # MRReceiveCoilSequence
        0x00189226,  # MRImageFrameTypeSequence
        0x00189329,  # CTImageFrameTypeSequence
        0x00189751,  # PETFrameTypeSequence
        0x00189341,  # ContrastBolusUsageSequence
    }
)
"""Functional groups whose every attribute times, weights, encodes or types
one acquisition, and is the same for all of its planes: a dimension pointing
into one is an acquisition whatever attribute it names, so an attribute
:data:`_ACQUISITIONS` does not list still splits. Groups that can vary plane
by plane within one acquisition --- frame content, plane position, a CT's
modulated exposure --- are left to the positions to judge."""


_POSITIONAL = frozenset({0x00209057, 0x00200032})
"""Dimensions that order frames within a stack --- ``InStackPositionNumber``
and ``ImagePositionPatient`` --- rather than tell stacks apart."""


_NON_IMAGE_SOP_CLASSES = (
    "RTDoseStorage",
    "RTStructureSetStorage",
    "RTPlanStorage",
    "RTIonPlanStorage",
    "SegmentationStorage",
    "SpatialRegistrationStorage",
    "DeformableSpatialRegistrationStorage",
    "RawDataStorage",
)
"""Objects that are not images, whatever patient position they carry.

An RT Dose grid carries `ImagePositionPatient` and sits beside the CT of every
radiotherapy export.  It was collected as a series, stacked to four
dimensions, and its refusal aborted the import of the whole tree.  Its values
are dose (scaled by `DoseGridScaling`), not image values; the structure sets,
segmentations and registrations have converters or no voxels of their own.
"""


def _sop_classes(pydicom: Any, names: Sequence[str]) -> frozenset[str]:
    return frozenset(
        str(getattr(pydicom.uid, name)) for name in names if hasattr(pydicom.uid, name)
    )


def _enhanced_sop_classes(pydicom: Any) -> frozenset[str]:
    return _sop_classes(pydicom, _ENHANCED_SOP_CLASSES)


def _frames(header: Any) -> int:
    """``NumberOfFrames``, 1 where it is absent or unreadable."""
    try:
        return int(getattr(header, "NumberOfFrames", 1) or 1)
    except (TypeError, ValueError):
        return 1


def scan_dicom(
    root: str | os.PathLike[str], *, report: ConversionReport | None = None
) -> list[Series]:
    """Walk a directory tree and collect its image series.

    Objects this importer cannot read as a series are skipped, and the ones a
    user would expect to be images --- enhanced and other multi-frame objects
    --- are reported on *report* as warnings, so an import that found nothing
    says why.  Objects that are not images but carry a patient position (an
    RT Dose grid) are skipped with a note: they are the ordinary company of
    the images, not a failure to read them.
    """
    pydicom = require_pydicom()
    enhanced = _enhanced_sop_classes(pydicom)
    unsupported_classes = _sop_classes(pydicom, _UNSUPPORTED_SOP_CLASSES)
    non_image = _sop_classes(pydicom, _NON_IMAGE_SOP_CLASSES)
    found: dict[str, Series] = {}
    objects: dict[str, list[tuple[str, Any]]] = {}
    unsupported: dict[str, list[str]] = {}
    multiframe: dict[str, list[str]] = {}
    others: dict[str, list[str]] = {}
    for path in sorted(Path(os.fspath(root)).rglob("*")):
        if not path.is_file():
            continue
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        except Exception:
            continue
        sop_class = str(getattr(header, "SOPClassUID", ""))
        if sop_class in unsupported_classes:
            unsupported.setdefault(sop_class, []).append(str(path))
            continue
        if sop_class in enhanced:
            if getattr(header, "SeriesInstanceUID", None) is not None:
                # A concatenation is one object split across instances
                # (C.7.6.16.2.2.4): its parts are joined by their UID, and
                # any other object by its own, so a copy of it at a second
                # path joins it (and is refused) rather than becoming a
                # second acquisition.
                key = (
                    _text(getattr(header, "ConcatenationUID", None))
                    or _text(getattr(header, "SOPInstanceUID", None))
                    or str(path)
                )
                objects.setdefault(key, []).append((str(path), header))
            continue
        uid = getattr(header, "SeriesInstanceUID", None)
        if uid is None or not hasattr(header, "ImagePositionPatient"):
            continue
        if sop_class in non_image:
            others.setdefault(sop_class, []).append(str(path))
            continue
        if _frames(header) > 1:
            # Its frames are not slices of a classic series, each a file of
            # its own: stacked as one, it became a 4-D grid nobody declared.
            multiframe.setdefault(sop_class, []).append(str(path))
            continue
        series = found.get(uid)
        if series is None:
            series = Series(
                series_uid=str(uid),
                study_uid=str(getattr(header, "StudyInstanceUID", "")),
                modality=str(getattr(header, "Modality", "OT")),
                patient_id=_text(getattr(header, "PatientID", None)),
                study_date=_text(getattr(header, "StudyDate", None)),
                description=_text(getattr(header, "SeriesDescription", "")) or "",
                frame_uid=_text(getattr(header, "FrameOfReferenceUID", None)),
                birth_date=_text(getattr(header, "PatientBirthDate", None)),
                sex=_text(getattr(header, "PatientSex", None)),
            )
            found[uid] = series
        series.paths.append(str(path))
    stacks = [stack for parts in objects.values() for stack in _object_series(parts)]
    if report is not None:
        if unsupported:
            report.warn(
                "unsupported",
                f"{_count(unsupported)} enhanced multi-frame instance(s) were "
                f"skipped ({', '.join(_sop_name(s) for s in sorted(unsupported))}): "
                "this importer reads classic series and enhanced CT, MR and PET "
                "objects, whose frames each state a plane position; these place "
                "their frames otherwise",
                _skipped(unsupported),
            )
        if multiframe:
            report.warn(
                "unsupported",
                f"{_count(multiframe)} multi-frame instance(s) were skipped: this "
                "importer reads classic series, a file per slice, and enhanced "
                "CT, MR and PET objects, and these multi-frame objects' frames "
                "are neither",
                _skipped(multiframe),
            )
        if others:
            report.add(
                "non_image",
                f"{_count(others)} object(s) that are not images were skipped "
                f"({', '.join(_sop_name(sop) for sop in sorted(others))}): they "
                "carry a patient position, but no image values to import --- an "
                "RT Dose grid holds dose",
                "info",
                _skipped(others),
            )
    # Stable: the stacks of one object keep the order the object holds them in.
    return sorted([*found.values(), *stacks], key=lambda s: (s.study_uid, s.series_uid))


def _object_series(parts: Sequence[tuple[str, Any]]) -> list[Series]:
    """One enhanced object's frames as series, a stack each.

    *parts* are the object's instances --- one, or a concatenation's ---
    as ``(path, header)``, joined in ``InConcatenationNumber`` order.  Frames
    are told apart by ``StackID`` and by the dimensions of the object's
    ``DimensionIndexSequence`` whose values each span several planes (a
    temporal position, an echo: :func:`_separating`); a stack is ordered
    later, by geometry.
    """
    parts = sorted(parts, key=lambda part: _number(part[1], "InConcatenationNumber"))
    header = parts[0][1]
    items = list(getattr(header, "DimensionIndexSequence", []))
    pointers = [int(getattr(item, "DimensionIndexPointer", 0) or 0) for item in items]
    groups = [int(getattr(item, "FunctionalGroupPointer", 0) or 0) for item in items]
    dimensions = [
        index
        for index, pointer in enumerate(pointers)
        if pointer not in (_POSITIONAL | {_STACK_ID})
    ]
    acquisitions = {
        d
        for d, index in enumerate(dimensions)
        if pointers[index] in _ACQUISITIONS
        or groups[index] in _ACQUISITION_GROUPS
        # A pointer may name the functional group itself (C.7.6.17.1).
        or pointers[index] in _ACQUISITION_GROUPS
    }
    found: list[tuple[str, int, tuple[Any, ...], tuple[float, ...] | None]] = []
    for path, part in parts:
        shared, per_frame = _groups(part)
        for index in range(_frames(part)):
            group = per_frame[index] if index < len(per_frame) else None
            found.append(
                (path, index, _stack_key(group, dimensions), _placed(group, shared))
            )
    by_id: dict[Any, list[int]] = {}
    for n, (_, _, key, _) in enumerate(found):
        by_id.setdefault(key[0], []).append(n)
    stacks: dict[tuple[Any, ...], list[tuple[str, int]]] = {}
    for members in by_id.values():
        keep = _separating(
            [found[n][2][1:] for n in members],
            [found[n][3] for n in members],
            acquisitions,
        )
        for n in members:
            path, index, key, _ = found[n]
            stack = (key[0], *(key[1 + d] for d in keep))
            stacks.setdefault(stack, []).append((path, index))
    labels = [
        "-".join(str(v) for v in key if v is not None) or str(n)
        for n, key in enumerate(stacks, start=1)
    ]
    if len(set(labels)) != len(labels):
        labels = [str(n) for n in range(1, len(labels) + 1)]
    return [
        Series(
            series_uid=str(header.SeriesInstanceUID),
            study_uid=str(getattr(header, "StudyInstanceUID", "")),
            modality=str(getattr(header, "Modality", "OT")),
            patient_id=_text(getattr(header, "PatientID", None)),
            study_date=_text(getattr(header, "StudyDate", None)),
            description=_text(getattr(header, "SeriesDescription", "")) or "",
            frame_uid=_text(getattr(header, "FrameOfReferenceUID", None)),
            birth_date=_text(getattr(header, "PatientBirthDate", None)),
            sex=_text(getattr(header, "PatientSex", None)),
            # Every part, though a stack's frames may lie in some of them:
            # a concatenation is checked whole (``_check_concatenation``).
            paths=[path for path, _ in parts],
            frames=frames,
            stack=label if len(stacks) > 1 else None,
        )
        for label, frames in zip(labels, stacks.values(), strict=True)
    ]


def _placed(group: Any, shared: Any) -> tuple[float, ...] | None:
    """A frame's ``ImagePositionPatient``, rounded to a micrometre, or
    ``None`` where it states none (refused later, when the frame is read)."""
    for holder in (group, shared):
        plane = _first_item(holder, "PlanePositionSequence")
        position = getattr(plane, "ImagePositionPatient", None) if plane else None
        if position is not None:
            try:
                return tuple(round(float(v), 3) for v in position)
            except (TypeError, ValueError):
                return None
    return None


def _separating(
    values: Sequence[tuple[Any, ...]],
    positions: Sequence[tuple[float, ...] | None],
    acquisitions: Collection[int] = (),
) -> list[int]:
    """Which of the dimensions in *values* (one tuple per frame of a
    ``StackID``) tell its stacks apart.

    The *acquisitions* --- dimensions whose pointer names one acquisition
    of the planes (:data:`_ACQUISITIONS`), or whose functional group does
    (:data:`_ACQUISITION_GROUPS`), named by the pointer itself or by
    ``FunctionalGroupPointer`` --- always do, a single-plane time series
    too.  Any other dimension does only where the planes repeat: when,
    within one acquisition, it tells apart frames at the same position, and
    a value of it is shared by frames at different positions.  So a
    per-frame or per-plane number (``FrameAcquisitionNumber``) splits
    nothing, and nor does a value a plane-varying attribute repeats --- a
    tube current alternating between two values --- where no position
    repeats.  Frames at one position that no dimension kept so far tells
    apart are told apart by the one other dimension that does, if only one
    does: a single-plane time series indexed by ``FrameReferenceDateTime``,
    a frame's time, which may as well be each slice's own in one volume and
    so names no acquisition by itself.  Where two would do it, neither is
    chosen, and the frames are refused as coincident.  With a position
    unknown, every dimension is kept, as before positions were read.
    """
    count = len(values[0]) if values else 0
    if any(position is None for position in positions):
        return list(range(count))
    chosen = sorted(d for d in acquisitions if d < count)
    # Frames at one position within one acquisition: the repeats any other
    # dimension must tell apart before it may split.
    at: dict[Any, list[int]] = {}
    for n, (value, position) in enumerate(zip(values, positions, strict=True)):
        at.setdefault((tuple(value[d] for d in chosen), position), []).append(n)
    repeats = [frames for frames in at.values() if len(frames) > 1]
    keep = []
    for d in range(count):
        if d in acquisitions:
            keep.append(d)
            continue
        places: dict[Any, set[Any]] = {}
        for value, position in zip(values, positions, strict=True):
            places.setdefault(value[d], set()).add(position)
        spans = any(len(where) > 1 for where in places.values())
        tells = any(len({values[n][d] for n in frames}) > 1 for frames in repeats)
        if spans and tells:
            keep.append(d)
    for frames in repeats:
        if any(len({values[n][d] for n in frames}) > 1 for d in keep):
            continue
        telling = [d for d in range(count) if len({values[n][d] for n in frames}) > 1]
        if len(telling) == 1:
            keep.append(telling[0])
    return sorted(set(keep))


def _stack_key(group: Any, dimensions: Sequence[int]) -> tuple[Any, ...]:
    """What tells a frame's stack from the others: its ``StackID`` and its
    index values in *dimensions*."""
    content = _first_item(group, "FrameContentSequence")
    stack = _text(getattr(content, "StackID", None)) if content is not None else None
    values = getattr(content, "DimensionIndexValues", None) if content else None
    if values is None:
        values = []
    elif isinstance(values, (int, str)):
        values = [values]
    values = list(values)
    return (stack, *(values[i] if i < len(values) else None for i in dimensions))


def _number(header: Any, keyword: str) -> int:
    """An integer attribute of *header*, 0 where it is absent or unreadable."""
    try:
        return int(getattr(header, keyword, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _ct_rescale_type(slice_: Any, *, enhanced: bool) -> str:
    """The ``RescaleType`` of a CT slice or frame, read where it is omitted.

    A classic CT omits it only when it is HU (C.8.2.1, Type 1C).  An enhanced
    CT must state it, and it must be HU only for an original, non-localizer,
    single-energy frame (C.8.15.3.10): omitted there, HU is what it had to
    be; omitted anywhere else, it is unknown, read as ``US`` (no units)
    rather than as a unit the values may not have.
    """
    stated = _text(getattr(slice_, "RescaleType", None))
    if stated is not None:
        return stated
    if not enhanced:
        return "HU"
    # The frame's own type, else the object's (a legacy-converted object
    # need not carry the frame type group).
    stated_kind = getattr(slice_, "FrameType", None) or getattr(
        slice_, "ImageType", None
    )
    kind = [str(v).upper() for v in (stated_kind or [])]
    multienergy = _text(getattr(slice_, "MultienergyCTAcquisition", None))
    original = bool(kind) and kind[0] == "ORIGINAL"
    localizer = len(kind) > 2 and kind[2] == "LOCALIZER"
    if original and not localizer and (multienergy or "").upper() != "YES":
        return "HU"
    return "US"


def _values(modality: str, rescale_type: str | None) -> dict[str, Any]:
    """An image's ``value_type`` and ``value_units`` from its modality and
    ``RescaleType``.

    A CT's rescale gives HU unless ``RescaleType`` names another quantity
    (C.8.15.3.10): a derived or multi-energy CT may hold mg/ml or Z_eff, and
    ``US`` (unspecified) names none.  Other modalities' values are intensity.
    """
    if modality != "CT":
        return {"value_type": "intensity", "value_units": None}
    if rescale_type == "US":
        return {"value_type": "intensity", "value_units": None}
    return {"value_type": "quantitative", "value_units": rescale_type or "HU"}


def _count(skipped: dict[str, list[str]]) -> int:
    return sum(len(paths) for paths in skipped.values())


def _skipped(skipped: dict[str, list[str]]) -> dict[str, Any]:
    """The detail of a note on skipped files: how many per SOP class, and the
    first twenty files."""
    return {
        "sop_classes": {sop: len(paths) for sop, paths in sorted(skipped.items())},
        "files": sorted(p for paths in skipped.values() for p in paths)[:20],
    }


def _sop_name(sop_class: str) -> str:
    """A SOP class UID by its name, where pydicom knows it."""
    from pydicom.uid import UID

    return str(UID(sop_class).name)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def read_series(
    series: Series,
    *,
    report: ConversionReport | None = None,
    per_slice_rescale: str = "refuse",
) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """One series as ``(volume, geometry)``, ordered and measured (§3.3).

    Every check reads the headers alone, and the pixels are decoded last, one
    slice at a time into the volume.  A transfer syntax no installed decoder
    reads, or a slice that does not decode, is refused by name --- as a
    ``MEDH5ValidationError``, not the decoder's own exception, which named no
    file and reached the command line as a traceback.

    The volume holds the stored values and ``geometry["rescale"]`` the
    series' one ``(slope, intercept)``.  A series whose slices state
    different ones is refused, unless *per_slice_rescale* is ``"apply"``:
    then the volume holds ``stored × slope + intercept`` of each slice,
    computed in float64 and returned as float32, ``geometry["rescale"]`` is
    ``None``, and *report* records the slopes and their slices.
    """
    planned = _plan_series(series, report, _per_slice_rescale(per_slice_rescale))
    volume = _decode(planned, series)
    planned.geometry["shape"] = tuple(int(v) for v in volume.shape)
    return volume, planned.geometry


@dataclass(slots=True)
class _Planned:
    """A series checked header by header: its files and instances in
    geometric order, and the geometry its volume will have."""

    paths: list[str]
    uids: list[str]
    geometry: dict[str, Any]
    samples: int
    rescales: list[tuple[float, float]] | None = None
    """Each slice's ``(slope, intercept)``, when they differ and are applied
    as the slices are decoded; ``None`` when the series has one."""
    frames: list[int | None] = field(default_factory=list)
    """Each slice's frame in its file, for a multi-frame object; empty for a
    classic series, whose files are its slices."""


def _plan_series(
    series: Series, report: ConversionReport | None, per_slice_rescale: str = "refuse"
) -> _Planned:
    """Read and check one series' headers --- every refusal but a slice that
    will not decode --- without decoding a pixel."""
    if not series.paths:
        raise MEDH5ValidationError(f"series {series.series_uid} has no files")
    if series.frames:
        slices = _frame_headers(series)
        sources: list[tuple[str, int | None]] = list(series.frames)
    else:
        slices = [_dcmread(p, stop_before_pixels=True) for p in series.paths]
        sources = [(p, None) for p in series.paths]
    orientation = np.asarray(
        _agreed(
            slices,
            lambda s: [float(v) for v in s.ImageOrientationPatient],
            "ImageOrientationPatient",
            series,
            expect=6,
        ),
        dtype=np.float64,
    )
    row, column = orientation[:3], orientation[3:]
    normal = np.cross(row, column)
    positions = np.asarray(
        [[float(v) for v in s.ImagePositionPatient] for s in slices], dtype=np.float64
    )
    # Geometry, not InstanceNumber: the projection onto the slice normal is the
    # only ordering the files themselves define.
    order = np.argsort(positions @ normal, kind="stable")
    slices = [slices[i] for i in order]
    paths = [sources[i][0] for i in order]
    frames = [sources[i][1] for i in order] if series.frames else []
    positions = positions[order]
    uids = [str(getattr(s, "SOPInstanceUID", "?")) for s in slices]

    z_spacing, gaps = _slice_spacing(
        positions,
        normal,
        series,
        report,
        _scalar(getattr(slices[0], "SliceThickness", None)),
        uids=uids,
    )
    if series.frames:
        _check_stack_order(slices, series)
    pixel_spacing = _agreed(
        slices,
        # No default: a series whose slices all omit `PixelSpacing` used to fall
        # through to 1 mm, which every slice then agreed on, so the stack was
        # written with an in-plane size the source never stated and nothing
        # recorded the assumption. Files reaching here carry
        # `ImagePositionPatient` (`scan_dicom` skips those that do not), so they
        # are cross-sectional images, for which `PixelSpacing` is mandatory ---
        # its absence is a broken series, not a series to guess about.
        lambda s: [float(v) for v in s.PixelSpacing],
        "PixelSpacing",
        series,
        expect=2,
    )
    _check_stacked(positions, normal, pixel_spacing, series, report)
    # One matrix, too: slices of different sizes are no volume, and reached
    # `np.stack` as a `ValueError` naming nothing.
    rows, columns = _agreed(
        slices,
        lambda s: [int(s.Rows), int(s.Columns)],
        "Rows/Columns",
        series,
        expect=2,
    )
    samples = _agreed(
        slices,
        lambda s: int(getattr(s, "SamplesPerPixel", 1)),
        "SamplesPerPixel",
        series,
    )
    _require_decoder(slices, series)
    photometric = _text(getattr(slices[0], "PhotometricInterpretation", None))
    if photometric == "MONOCHROME1" and report is not None:
        # Stored as written (§4.2): inverting would make the file disagree
        # with its source, and this importer never rewrites values.  But a
        # model trained on intensity needs to know that here low is bright.
        report.warn(
            "photometric",
            f"series {series.series_uid}: PhotometricInterpretation is "
            "MONOCHROME1, so lower stored values are brighter; values are stored "
            "as written, and a model trained on intensity should invert them",
            {"series_uid": series.series_uid},
        )
    rescale, rescales = _series_rescale(slices, uids, series, report, per_slice_rescale)
    # What the rescale gives: one quantity for the volume, so one type.  Only
    # a CT's is read (:func:`_values`), and one omitted is read as what DICOM
    # makes it (:func:`_ct_rescale_type`) before the slices are compared.
    rescale_type = (
        _agreed(
            slices,
            lambda s: _ct_rescale_type(s, enhanced=bool(series.frames)),
            "RescaleType",
            series,
        )
        if series.modality == "CT"
        else None
    )
    direction = np.stack([normal, column, row], axis=1)
    geometry = {
        "shape": (len(slices), int(rows), int(columns)),
        "spacing": [z_spacing, pixel_spacing[0], pixel_spacing[1]],
        "origin": [float(v) for v in positions[0]],
        "direction": [[float(v) for v in r] for r in direction],
        "coord_system": "LPS",
        "units": "mm",
        "frame_uid": series.frame_uid,
        "rescale": rescale,
        "rescale_type": rescale_type,
        "acquisition": _acquisition(slices, series, report),
        "gaps": gaps,
    }
    return _Planned(
        paths=paths,
        uids=uids,
        geometry=geometry,
        samples=int(samples),
        rescales=rescales,
        frames=frames,
    )


def _acquisition(
    slices: Sequence[Any], series: Series, report: ConversionReport | None
) -> dict[str, Any]:
    """The :data:`STORED_TAGS` every slice of *series* agrees on (§4.5).

    One value per volume is a claim about every slice, so an attribute the
    slices state differently --- tube-current modulation's per-slice current,
    an enhanced MR's per-frame echo time with no echo dimension to split on
    --- is left out, with a warning, rather than recorded as the first
    slice's.  It is not refused: the voxels do not depend on it.
    """
    out: dict[str, Any] = {}
    varying = []
    for tag in STORED_TAGS:
        values = [_scalar(getattr(s, tag, None)) for s in slices]
        if all(value is None for value in values):
            continue
        # Stated by some slices and not others is a disagreement too,
        # whichever slice sorts first.
        if any(_differs(values[0], value) for value in values[1:]):
            varying.append(tag)
            continue
        out[tag] = values[0]
    if varying and report is not None:
        report.warn(
            "acquisition",
            f"series {series.series_uid}: its slices state different values of "
            f"{', '.join(varying)}, so the image records none of them",
            {"series_uid": series.series_uid, "tags": varying},
        )
    return out


def _frame_headers(series: Series) -> list[Any]:
    """Each frame of a multi-frame series, read as the header of a classic
    slice: what a classic slice states at its top level, a frame states in
    its functional groups (:func:`~medh5.io._frames._group_value`), so the
    checks a classic series passes are the checks its frames pass.

    A frame is named in a refusal as ``frame N of instance UID``, *N*
    1-based as DICOM numbers frames.
    """
    headers = {path: _dcmread(path, stop_before_pixels=True) for path in series.paths}
    _check_concatenation(series, list(headers.values()))
    groups = {path: _groups(header) for path, header in headers.items()}
    for path, header in headers.items():
        _check_frame_count(series, header, len(groups[path][1]))
    out = []
    for path, index in series.frames:
        dataset = headers[path]
        shared, per_frame = groups[path]
        group = per_frame[index] if index < len(per_frame) else None
        sop = str(getattr(dataset, "SOPInstanceUID", "?"))
        name = f"frame {index + 1} of instance {sop}"
        attributes: dict[str, Any] = {
            "SOPInstanceUID": name,
            "file_meta": getattr(dataset, "file_meta", None),
            "ImagePositionPatient": list(_position(group, shared, name)),
            "ImageOrientationPatient": list(_orientation(group, shared, name)),
            "PixelSpacing": list(_pixel_spacing(group, shared, name)),
        }
        for keyword in (
            "Rows",
            "Columns",
            "SamplesPerPixel",
            "RescaleSlope",
            "RescaleIntercept",
            "RescaleType",
            "FrameType",
            "ImageType",
            "MultienergyCTAcquisition",
            *STORED_TAGS,
        ):
            value = _group_value(group, shared, dataset, keyword)
            if value is not None:
                attributes[keyword] = value
        content = _first_item(group, "FrameContentSequence")
        number = getattr(content, "InStackPositionNumber", None) if content else None
        if number is not None:
            attributes["InStackPositionNumber"] = int(number)
        out.append(SimpleNamespace(**attributes))
    return out


def _check_frame_count(series: Series, header: Any, items: int) -> None:
    """Refuse an instance whose ``NumberOfFrames`` is absent or disagrees
    with its per-frame functional groups, one item per frame (C.7.6.16.1):
    either count would leave frames unread, or read frames that are not."""
    stated = _number(header, "NumberOfFrames")
    if stated != items:
        sop = str(getattr(header, "SOPInstanceUID", "?"))
        said = (
            f"states NumberOfFrames {stated}" if stated else "states no NumberOfFrames"
        )
        raise MEDH5ValidationError(
            f"series {series.series_uid}: instance {sop} {said}, but holds "
            f"{items} per-frame functional group item(s), one per frame; which "
            "frames it holds cannot be told"
        )


def _check_not_a_part(series: Series, header: Any) -> None:
    """Refuse an instance with no ``ConcatenationUID`` whose concatenation
    markers say it is a part of a larger object: read alone, its frames
    would be imported as the whole.  Markers that describe a whole object
    (part 1 of 1, from frame 0) are harmless and pass."""
    number = _number(header, "InConcatenationNumber")
    total = _number(header, "InConcatenationTotalNumber")
    offset = _number(header, "ConcatenationFrameOffsetNumber")
    if number > 1 or total > 1 or offset > 0:
        sop = str(getattr(header, "SOPInstanceUID", "?"))
        raise MEDH5ValidationError(
            f"series {series.series_uid}: instance {sop} states no "
            f"ConcatenationUID, but is part {number or '?'} of "
            f"{total or 'an unstated number'} from frame offset {offset}; the "
            "concatenation it belongs to cannot be joined"
        )


def _check_concatenation(series: Series, headers: Sequence[Any]) -> None:
    """Refuse a concatenation some of whose instances are missing or repeated:
    its frames are one object's, and a part of them is not a volume."""
    uid = _text(getattr(headers[0], "ConcatenationUID", None))
    if uid is None:
        if len(headers) > 1:
            sop = _text(getattr(headers[0], "SOPInstanceUID", None))
            raise MEDH5ValidationError(
                f"series {series.series_uid}: instance {sop} is in "
                f"{len(headers)} files ({', '.join(series.paths)}); its frames "
                "would be read twice, so remove the duplicate files"
            )
        _check_not_a_part(series, headers[0])
        return
    # Optional, and any part may state it: every stated total must agree.
    totals = sorted({_number(h, "InConcatenationTotalNumber") for h in headers} - {0})
    if len(totals) > 1:
        raise MEDH5ValidationError(
            f"series {series.series_uid}: concatenation {uid} states its total "
            f"as {totals} in different instances; which instances are missing "
            "cannot be told"
        )
    total = totals[0] if totals else 0
    numbers = sorted(_number(h, "InConcatenationNumber") for h in headers)
    if len(set(numbers)) != len(numbers):
        raise MEDH5ValidationError(
            f"series {series.series_uid}: concatenation {uid} holds instance "
            f"number(s) {sorted({n for n in numbers if numbers.count(n) > 1})} "
            "more than once; remove the duplicate files"
        )
    # InConcatenationTotalNumber is optional; without it, the numbers must
    # still run from 1 without a gap, and only a missing last part goes unseen.
    expected = list(range(1, (total or len(numbers)) + 1))
    if numbers != expected:
        found = (
            f"{len(numbers)} of its {total} instances"
            if total
            else f"{len(numbers)} instance(s), with no total stated"
        )
        raise MEDH5ValidationError(
            f"series {series.series_uid}: concatenation {uid} was found as "
            f"{found}, numbered {numbers}. The frames of one object are split "
            "across them, and the missing ones' are not here to place; import "
            "the complete concatenation"
        )
    # Each part's ConcatenationFrameOffsetNumber is where its first frame sits
    # in the object, which the frames of the parts before it fix: so the
    # parts' frames follow on with none between.  Every part that states one
    # is held to it (it is required, but not always set); what one part omits
    # leaves the others' checked, since each follows from the frames alone.
    ordered = sorted(headers, key=lambda h: _number(h, "InConcatenationNumber"))
    expected = [0]
    for header in ordered[:-1]:
        expected.append(expected[-1] + _frames(header))
    stated = [getattr(h, "ConcatenationFrameOffsetNumber", None) for h in ordered]
    if any(
        s is not None and int(s) != e for s, e in zip(stated, expected, strict=True)
    ):
        offsets = [None if s is None else int(s) for s in stated]
        raise MEDH5ValidationError(
            f"series {series.series_uid}: concatenation {uid} states its "
            f"parts' frame offsets as {offsets}, but their frames put them "
            f"at {expected}: frames between the parts are missing, or the "
            "offsets are wrong, and either way the frames cannot be placed"
        )


def _check_stack_order(slices: Sequence[Any], series: Series) -> None:
    """Refuse a stack whose ``InStackPositionNumber`` its positions contradict,
    or that skips a number.

    *slices* are in geometric order.  The number is an ordinal from 1 that
    orders a frame within its stack (C.7.6.16.2.2.4), so along the slice
    normal it runs 1 to N one way or the other: the frame *i* places must
    be numbered ``i + 1``, or ``N - i``.  One that is not is a broken object
    --- a plane missing, or a numbering its geometry contradicts --- and
    re-sorting it by either would be a guess.  Frames that state no number
    are left out, and the numbers the others state are still held to that.
    """
    stated = [getattr(s, "InStackPositionNumber", None) for s in slices]
    known = [(i, int(n)) for i, n in enumerate(stated) if n is not None]
    count = len(slices)
    if all(n == i + 1 for i, n in known) or all(n == count - i for i, n in known):
        return
    numbers = [n for _, n in known]
    missing = sorted(set(range(1, count + 1)) - set(numbers))
    if len(known) == count and len(set(numbers)) == count:
        raise MEDH5ValidationError(
            f"series {series.series_uid}: its frames' InStackPositionNumber are "
            f"{sorted(numbers)}, which count a stack from 1 without a gap, so "
            f"the frame(s) numbered {missing} are missing; a stack with a plane "
            "missing is not a volume"
        )
    beyond = sorted({n for n in numbers if not 1 <= n <= count})
    if beyond:
        raise MEDH5ValidationError(
            f"series {series.series_uid}: its frames' InStackPositionNumber "
            f"include {beyond}, beyond the {count} frames of its stack, which "
            "the number counts from 1; frames of the stack are missing, and a "
            "stack with a plane missing is not a volume"
        )
    raise MEDH5ValidationError(
        f"series {series.series_uid}: in the order their positions put them "
        f"along the slice normal, its frames' InStackPositionNumber run "
        f"{[None if n is None else int(n) for n in stated]}, not one way. A "
        "stack whose numbering its geometry contradicts is a broken object, and "
        "re-sorting it by either would be a guess"
    )


def _series_rescale(
    slices: Sequence[Any],
    uids: Sequence[str],
    series: Series,
    report: ConversionReport | None,
    mode: str,
) -> tuple[tuple[float, float] | None, list[tuple[float, float]] | None]:
    """The series' one ``(slope, intercept)``, or under ``"apply"`` each
    slice's, when they differ: ``(rescale, None)`` or ``(None, rescales)``.

    *slices* are in geometric order, so a position in *rescales* is the
    slice's index along the grid's first axis.  A slice stating no rescale
    among slices that do is refused whatever *mode* says: its LUT is not
    stated, and taking it as the identity would be a guess.
    """
    what = "RescaleSlope/Intercept"
    values = [_extracted(s, _rescale, what, series) for s in slices]
    first = values[0]
    differing = [k for k, v in enumerate(values) if _differs(first, v)]
    if not differing:
        return first, None
    if None in values:
        k = values.index(None)
        raise MEDH5ValidationError(
            f"series {series.series_uid}: instance {uids[k]} states no {what}, "
            f"but other slices of the series do. Its modality LUT is not "
            f"stated, and taking it as the identity would be a guess; restore "
            f"the tags, or split the series."
        )
    if mode != "apply":
        k = differing[0]
        raise MEDH5ValidationError(
            f"series {series.series_uid}: its slices state "
            f"{len(_distinct(values))} different {what} pairs (instance "
            f"{uids[k]} states {values[k]!r}, the first slice {first!r}). An "
            f"image stores one modality LUT (§4.2), and storing the first "
            f"slice's for all of them would misscale the rest without saying "
            f'so. Import it with per_slice_rescale="apply" '
            f"(--per-slice-rescale apply) to apply each slice's rescale and "
            f"store physical values, or split the series."
        )
    rescales = [(float(v[0]), float(v[1])) for v in values if v is not None]
    if report is not None:
        groups = _distinct(rescales)
        report.decision(
            "per_slice_rescale",
            f"series {series.series_uid}: its {len(rescales)} slices state "
            f"{len(groups)} different {what} pairs, so stored × slope + "
            "intercept was applied slice by slice in float64 and stored as "
            "float32, with no rescale attributes (§4.2)",
            {
                "series_uid": series.series_uid,
                "rescales": [
                    {"slope": slope, "intercept": intercept, "slices": at}
                    for (slope, intercept), at in groups.items()
                ],
            },
        )
    return None, rescales


def _distinct(values: Sequence[Any]) -> dict[Any, list[int]]:
    """Each distinct value of *values*, in order of first appearance, with the
    positions holding it."""
    out: dict[Any, list[int]] = {}
    for k, value in enumerate(values):
        out.setdefault(value, []).append(k)
    return out


def _require_decoder(slices: Sequence[Any], series: Series) -> None:
    """Refuse a series stored in a transfer syntax no installed decoder reads,
    before any of it is decoded."""
    from pydicom.pixels import get_decoder
    from pydicom.uid import UID

    syntaxes = {
        str(syntax)
        for s in slices
        if (syntax := getattr(getattr(s, "file_meta", None), "TransferSyntaxUID", None))
    }
    for syntax in sorted(syntaxes):
        try:
            available = bool(get_decoder(UID(syntax)).is_available)
        except (NotImplementedError, ValueError):
            available = False
        if not available:
            raise MEDH5ValidationError(
                f"series {series.series_uid} is stored as {UID(syntax).name} "
                f"({syntax}), which no installed pixel-data decoder reads; "
                "install a pydicom decoder plugin for it, or decompress the "
                "series first"
            )


def _decode(planned: _Planned, series: Series) -> npt.NDArray[Any]:
    """The planned slices' pixels, decoded one at a time into one volume.

    Reading every slice's pixel data and stacking the decoded arrays held the
    series three times over before the first check ran.  A multi-frame
    object's frames are decoded one at a time too, never its whole pixel
    array.  A slice that does not decode, or decodes to another shape than
    its header states, is refused by name.
    """
    expected = tuple(planned.geometry["shape"][1:]) + (
        (planned.samples,) if planned.samples != 1 else ()
    )
    rescales = planned.rescales
    volume: npt.NDArray[Any] | None = None
    for k, plane in _planes(planned, series):
        path, uid = planned.paths[k], planned.uids[k]
        if plane.shape != expected:
            raise MEDH5ValidationError(
                f"series {series.series_uid}: instance {uid} ({path}) decodes to a "
                f"{plane.shape} array, not the {expected} its header states"
            )
        if rescales is not None:
            # Physical values, slice by slice: float64 arithmetic, float32 kept.
            slope, intercept = rescales[k]
            plane = (plane.astype(np.float64) * slope + intercept).astype(np.float32)
        if volume is None:
            volume = np.empty((len(planned.paths), *plane.shape), dtype=plane.dtype)
        elif plane.dtype != volume.dtype:
            # What `np.stack` did: the type every slice fits in.
            volume = volume.astype(np.result_type(volume.dtype, plane.dtype))
        volume[k] = plane
    assert volume is not None  # a plan has at least one file
    return volume


def _planes(
    planned: _Planned, series: Series
) -> Iterable[tuple[int, npt.NDArray[Any]]]:
    """``(slice index, decoded plane)`` for every planned slice, each once.

    A classic slice is its file's pixel array; a frame is decoded alone, with
    the frames of each file read in file order.
    """
    if not planned.frames:
        for k, path in enumerate(planned.paths):
            read = partial(getattr, _dcmread(path), "pixel_array")
            yield k, _decoded(read, planned, k, series)
        return
    from pydicom.pixels import iter_pixels

    files: dict[str, list[tuple[int, int]]] = {}
    for k, (path, frame) in enumerate(zip(planned.paths, planned.frames, strict=True)):
        assert frame is not None
        files.setdefault(path, []).append((frame, k))
    for path, entries in files.items():
        entries.sort()
        planes = iter(iter_pixels(path, indices=[frame for frame, _ in entries]))
        for _, k in entries:
            yield k, _decoded(partial(next, planes), planned, k, series)


def _decoded(
    read: Callable[[], Any], planned: _Planned, k: int, series: Series
) -> npt.NDArray[Any]:
    """``read()`` as an array, or a refusal naming slice *k*."""
    try:
        return np.asarray(read())
    except MemoryError:  # pragma: no cover - not a property of the file
        raise
    except Exception as exc:  # a decoder raises whatever it raises
        raise MEDH5ValidationError(
            f"series {series.series_uid}: instance {planned.uids[k]} "
            f"({planned.paths[k]}) could not be decoded "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def _slice_spacing(
    positions: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
    series: Series,
    report: ConversionReport | None,
    thickness: Any = None,
    *,
    uids: Sequence[str] | None = None,
) -> tuple[float, list[float]]:
    """Spacing measured over the stack, refusing one that is not a grid.

    *positions* are in geometric order.  Every slice is held to its place in
    a regular stack, ``origin + k * spacing * normal``, within
    :data:`SPACING_TOLERANCE` of a slice.  Each gap was held to 1 % of the
    median gap instead, which bounded no slice: per-gap errors under 1 %
    accumulate, and positions rounded to two decimals, which accumulate
    nothing, were refused.

    The spacing is the median gap where it places every slice --- a regular
    stack's gaps agree on it, and it is what earlier versions wrote --- and
    otherwise the least-squares step over every slice, from the first slice
    origin, where the grid starts: the median of gaps rounded to 0.62 and
    0.63 mm is one of them, and builds a millimetre of error over 200 slices
    that a fit to every position does not.  A stack neither places is
    refused, a gap of more than one and a half steps named as missing
    slices.  *uids* name the slices, in the same order, in a refusal.
    """
    if positions.shape[0] == 1:
        # No neighbour to measure an increment against.  SliceThickness is the
        # slab rather than the increment, which is why it is not trusted for a
        # stack --- but for a lone slice it is the only physical extent the
        # source states, and inventing 1 mm over a declared 5 mm gives the grid
        # a wrong extent that later resampling and export propagate.
        declared = _positive(thickness)
        if declared is None:
            if report is not None:
                report.guess(
                    "slice_spacing",
                    "a single-slice series carries no increment to measure and "
                    "declared no usable SliceThickness; 1 mm was assumed",
                    {"spacing": 1.0, "thickness": thickness, "slices": 1},
                )
            return 1.0, []
        if report is not None:
            report.decision(
                "slice_spacing",
                "a single-slice series has no neighbouring position to measure "
                f"against, so its declared SliceThickness ({declared:.4g} mm) "
                "was used as the through-plane spacing",
                {"spacing": declared, "thickness": declared, "slices": 1},
            )
        return declared, []
    n = positions.shape[0]
    names = list(uids) if uids is not None else [f"#{k}" for k in range(n)]
    projected = positions @ normal
    gaps = np.diff(projected)
    coincident = np.flatnonzero(np.abs(gaps) < _POSITION_FLOOR_MM)
    if coincident.size:
        k = int(coincident[0])
        raise MEDH5ValidationError(
            f"series {series.series_uid} has slices at coincident positions: "
            f"instances {names[k]} and {names[k + 1]} both lie "
            f"{projected[k]:.4g} mm along the slice normal ({coincident.size} "
            "such pair(s)). A series is one volume with one slice per position; "
            "remove the duplicate files, or split the series by phase or echo"
        )
    steps = np.arange(n, dtype=np.float64)
    along = projected - projected[0]
    fits = {
        "median gap": float(np.median(gaps)),
        "least squares": float(steps @ along / (steps @ steps)),
    }
    for estimate in fits:
        spacing = fits[estimate]
        tolerance = _position_tolerance(spacing)
        residual = along - steps * spacing
        worst = int(np.argmax(np.abs(residual)))
        error = float(abs(residual[worst]))
        if error <= tolerance:
            break
    else:
        raise _irregular(series, projected, names, spacing, worst, error, tolerance)
    if report is not None:
        how = (
            "the median gap between slice origins"
            if estimate == "median gap"
            else "the least-squares step over every slice origin from the first"
        )
        report.decision(
            "slice_spacing",
            f"z spacing {spacing:.4g} mm was measured as {how}, not taken from "
            "SliceThickness (which is the slab, not the increment); every slice "
            f"lies within {error:.3g} mm of where that regular stack puts it "
            f"(tolerance {tolerance:.3g} mm: a tenth of a slice, and at least "
            "0.01 mm)",
            {
                "spacing": spacing,
                "thickness": thickness,
                "slices": n,
                "estimate": estimate,
                "max_through_plane_error_mm": error,
                "tolerance_mm": tolerance,
            },
        )
    return spacing, [float(g) for g in gaps]


def _irregular(
    series: Series,
    projected: npt.NDArray[np.float64],
    names: Sequence[str],
    spacing: float,
    worst: int,
    error: float,
    tolerance: float,
) -> MEDH5ValidationError:
    """The refusal of a stack that is not a grid: missing slices where every
    gap is a whole number of the usual step, an irregular stack otherwise."""
    gaps = np.diff(projected)
    step = float(np.median(gaps))
    whole = np.round(gaps / step)
    if (
        np.all(whole >= 1)
        and np.any(whole >= 2)
        and np.all(np.abs(gaps - whole * step) <= _position_tolerance(step))
    ):
        k = int(np.flatnonzero(whole >= 2)[0])
        return MEDH5ValidationError(
            f"series {series.series_uid} is missing {int(whole.sum()) - len(gaps)} "
            f"slice(s): the gap from instance {names[k]} to instance {names[k + 1]} "
            f"is {gaps[k]:.4g} mm along the slice normal, {whole[k]:.0f} times the "
            f"{step:.4g} mm between the others. A stack with a hole is not a grid; "
            "import the complete series, or the parts either side of the gap as "
            "separate series"
        )
    return MEDH5ValidationError(
        f"series {series.series_uid} has irregular slice gaps: the regular stack "
        f"that best fits its positions --- slices {spacing:.4g} mm apart from the "
        f"first --- leaves the slice at index {worst} of {len(projected)} "
        f"(instance {names[worst]}) {error:.3g} mm off, beyond the "
        f"{tolerance:.3g} mm tolerance (a tenth of a slice, and at least 0.01 mm). "
        "An irregular stack is not a grid --- resample it deliberately, or "
        "convert the slices as separate images"
    )


def _check_stacked(
    positions: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
    pixel_spacing: Sequence[float],
    series: Series,
    report: ConversionReport | None,
) -> None:
    """Refuse a stack whose slice origins do not run along the slice normal.

    The grid this importer writes is orthonormal: slice *k* sits at ``origin +
    k * spacing * normal``.  A tilted gantry --- routine in head CT --- shifts
    each slice *within* its plane as well, so the stack is sheared, and writing
    it as an orthonormal grid places the far slices centimetres from where the
    scanner put them, with nothing in the file to say so.  Converters refuse
    rather than resample (§3), so the tilt is measured and reported, and
    de-tilting is left to a tool that records what it did.

    The measure is each slice's distance from the stack's axis through the
    first origin, against half the finest in-plane pixel: within it, every
    voxel of the grid is less than half a voxel from where the scanner put it.
    Measured per slice rather than per step, because position rounding in the
    files is per slice and does not accumulate, where a tilt does.
    """
    if positions.shape[0] < 2:
        return
    relative = positions - positions[0]
    along = relative @ normal
    in_plane = relative - along[:, None] * normal
    offsets = np.linalg.norm(in_plane, axis=1)
    worst = float(offsets.max())
    tolerance = 0.5 * float(min(pixel_spacing))
    if worst > tolerance:
        # The shear as a slope, fitted over the whole stack: in-plane offset per
        # millimetre along the normal is the tangent of the tilt.
        slope = (in_plane.T @ along) / float(along @ along)
        angle = float(np.degrees(np.arctan(np.linalg.norm(slope))))
        raise MEDH5ValidationError(
            f"series {series.series_uid} is sheared: its slice origins drift "
            f"within the slice plane as the stack advances --- a gantry tilt of "
            f"about {angle:.1f} degrees --- and the worst slice is {worst:.3g} mm "
            f"off the stack's axis (tolerance {tolerance:.3g} mm, half a pixel). "
            "An orthonormal grid would misplace every slice after the first; "
            "de-tilt (resample) the series with a tool that records what it "
            "did, then import the result"
        )
    if report is not None:
        report.decision(
            "slice_alignment",
            "slice origins lie along the slice normal, so the stack is not "
            f"sheared: the worst slice is {worst:.3g} mm off the stack's axis, "
            f"within half a pixel ({tolerance:.3g} mm)",
            {"in_plane_offset_mm": worst, "tolerance_mm": tolerance},
        )


def _agreed(
    slices: Sequence[Any],
    extract: Callable[[Any], Any],
    what: str,
    series: Series,
    *,
    expect: int | None = None,
) -> Any:
    """The value every slice agrees on, or a refusal.

    A series is one volume with one geometry and one modality LUT, so each of
    these is read once --- but reading it from ``slices[0]`` *without* checking
    is what makes that assumption dangerous.  A stack whose slices disagree then
    silently takes the first slice's answer for all of them: a stack with one
    rotated slice places that slice's voxels somewhere the scanner never put
    them, which is not visible in the output, so it is refused here rather
    than averaged or assumed.  (A per-slice rescale, ordinary in PET, has its
    own check, :func:`_series_rescale`, which can apply it instead.)
    """
    values = [_extracted(s, extract, what, series, expect) for s in slices]
    first = values[0]
    for slice_, value in zip(slices[1:], values[1:], strict=True):
        if _differs(first, value):
            # Named by SOPInstanceUID, not by position: these checks run either
            # side of the geometric sort, so an index would mean two things.
            uid = getattr(slice_, "SOPInstanceUID", "?")
            raise MEDH5ValidationError(
                f"series {series.series_uid}: instance {uid} declares "
                f"{what} {value!r}, but the first slice declares {first!r}. "
                f"A series is "
                f"one volume with one geometry and one modality LUT; storing the "
                f"first slice's value for all of them would misplace or misscale "
                f"the rest without saying so. Split the series, or resample it "
                f"deliberately with a tool that records what it did."
            )
    return first


def _extracted(
    slice_: Any,
    extract: Callable[[Any], Any],
    what: str,
    series: Series,
    expect: int | None = None,
) -> Any:
    """*what*, read off one slice, or a refusal naming that slice.

    ``scan_dicom`` groups files by series without requiring any of these tags,
    so a slice reaches here missing one outright, or carrying a value that will
    not parse.  Letting that surface as the raw ``AttributeError`` gives a CLI
    traceback where the refusal below would have named the instance --- and an
    exception a caller cannot catch as ``MEDH5Error`` like every other input
    problem this converter reports.
    """
    try:
        value = extract(slice_)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        uid = getattr(slice_, "SOPInstanceUID", "?")
        raise MEDH5ValidationError(
            f"series {series.series_uid}: instance {uid} has no usable {what} "
            f"({type(exc).__name__}: {exc}). Every slice of a series has to "
            f"declare it, because the volume is written with one value for all "
            f"of them."
        ) from exc
    # Cardinality, not just parseability.  A tag can extract perfectly well and
    # still be the wrong length, and every slice can carry the same wrong
    # length --- so the agreement check above sees no disagreement and passes it
    # through.  Downstream that is either a raw exception (a five-value
    # `ImageOrientationPatient` reaches `np.cross`) or, worse, silence (a
    # three-value `PixelSpacing` is read as its first two elements and the third
    # is discarded, giving the grid an in-plane size nobody wrote down).
    if expect is not None and len(value) != expect:
        uid = getattr(slice_, "SOPInstanceUID", "?")
        raise MEDH5ValidationError(
            f"series {series.series_uid}: instance {uid} declares {what} with "
            f"{len(value)} value(s), but it is defined as {expect}. A tag of the "
            f"wrong length is a malformed series, not one to read the first "
            f"{expect} values out of."
        )
    return value


def _differs(first: Any, other: Any) -> bool:
    if (first is None) != (other is None):
        return True
    if first is None:
        return False
    if isinstance(first, str) or isinstance(other, str):
        return bool(first != other)
    return not np.allclose(
        np.asarray(first, dtype=np.float64),
        np.asarray(other, dtype=np.float64),
        atol=1e-6,
    )


def _rescale(dataset: Any) -> tuple[float, float] | None:
    slope = _scalar(getattr(dataset, "RescaleSlope", None))
    intercept = _scalar(getattr(dataset, "RescaleIntercept", None))
    if slope is None and intercept is None:
        return None
    return float(slope if slope is not None else 1.0), float(
        intercept if intercept is not None else 0.0
    )


def _positive(value: Any) -> float | None:
    """*value* as a usable physical length, or ``None`` if it is not one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def from_dicom(
    root: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    group_by: str = "subject",
    modalities: Sequence[str] | None = None,
    series_uids: Sequence[str] | None = None,
    codec: str = "balanced",
    report: ConversionReport | None = None,
    per_slice_rescale: str = "refuse",
) -> ConversionReport:
    """Convert a DICOM tree into one sample per subject (or per study).

    *out* is a directory --- each sample named for its subject inside it ---
    when more than one sample results, when it is a directory already, or when
    it ends in a path separator (``out/``); otherwise it is the file the one
    sample is written to.

    A series this importer refuses --- a localizer in three orientations, a
    sheared or irregular stack, a transfer syntax no installed decoder reads
    --- is left out with a warning naming it, and the series beside it are
    imported: one such series aborted the whole tree.  A subject left with no
    series gets no file; when no series at all can be imported, the refusal
    is raised.

    A series whose slices state different ``RescaleSlope``/``RescaleIntercept``
    (ordinary in PET) is refused, since an image stores one modality LUT
    (§4.2).  *per_slice_rescale* ``"apply"`` imports it instead, as
    ``stored × slope + intercept`` of each slice in float32 with no rescale
    attributes, and records the slopes and their slices.  A series with one
    rescale is written the same either way, and SUV is never computed.
    """
    mode = _per_slice_rescale(per_slice_rescale)
    log = report or ConversionReport(converter="from-dicom")
    log.source = os.fspath(root)
    series = scan_dicom(root, report=log)
    if series_uids is not None:
        wanted = set(series_uids)
        series = [s for s in series if s.series_uid in wanted]
    if modalities is not None:
        allowed = {m.upper() for m in modalities}
        series = [s for s in series if s.modality.upper() in allowed]
    if not series:
        raise MEDH5ValidationError(f"no DICOM image series found under {root}")

    studies: dict[str, list[Series]] = {}
    for entry in series:
        studies.setdefault(entry.study_uid, []).append(entry)
    occasions = [
        Occasion(
            key=study_uid,
            subject_id=_consistent_patient(entries, log),
            date=next((e.study_date for e in entries if e.study_date), None),
            payload=entries,
            demographics=_demographics(entries),
        )
        for study_uid, entries in sorted(studies.items())
    ]
    groups = group_by_subject(occasions, mode=group_by, report=log)
    target = Path(os.fspath(out))
    named = os.fspath(out)
    directory = (
        len(groups) > 1
        or target.is_dir()
        or named.endswith(os.sep)
        or (os.altsep is not None and named.endswith(os.altsep))
    )
    if directory:
        target.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    refused: list[MEDH5ValidationError] = []
    written = 0
    for group in groups:
        path = (
            target / f"{output_name(group, used, safe=_safe)}.medh5"
            if directory
            else target
        )
        written += _write_subject(
            group, path, codec=codec, log=log, refused=refused, per_slice_rescale=mode
        )
    if not written:
        if len(refused) == 1:
            raise refused[0]
        raise MEDH5ValidationError(
            f"none of the {len(refused)} series under {root} could be imported: "
            + "; ".join(str(exc) for exc in refused)
        )
    return log


def _consistent_patient(entries: Sequence[Series], log: ConversionReport) -> str | None:
    """One PatientID per study, or none --- never a majority vote."""
    ids = {e.patient_id for e in entries if e.patient_id}
    if len(ids) == 1:
        return next(iter(ids))
    if len(ids) > 1:
        log.warn(
            "identity",
            f"study {entries[0].study_uid} carries {len(ids)} different "
            "PatientIDs; it was left ungrouped rather than merged on a guess",
            {"patient_ids": sorted(ids)},
        )
    return None


def _demographics(entries: Sequence[Series]) -> dict[str, str]:
    """What one study states about the person, for :func:`group_by_subject`.

    Every stated value is kept, joined, so a study whose own series disagree
    contradicts every other study rather than passing for one of them.
    """
    out: dict[str, str] = {}
    for name, values in (
        ("PatientBirthDate", {e.birth_date for e in entries if e.birth_date}),
        ("PatientSex", {e.sex for e in entries if e.sex}),
    ):
        if values:
            out[name] = "|".join(sorted(values))
    return out


def _safe(text: str) -> str:
    """*text* as a sample id or a file name stem: ASCII identifier characters.

    A ``PatientID`` of ``Müller^Hans`` aborted the import (E003); see
    :func:`~medh5.io._common.sanitize_ascii_stem`.
    """
    return sanitize_ascii_stem(text)


def _series_names(entries: Sequence[Series], tp: str) -> list[tuple[Series, str, str]]:
    """``(series, grid id, image id)`` for every series of one visit.

    The name is the modality and the visit --- ``CT_tp0`` on grid ``ct_tp0``
    --- and, where a visit holds several series of one modality, a 1-based
    position in ``SeriesInstanceUID`` order: ``MR_1_tp0``, ``MR_2_tp0``.  A T1
    and a T2 in one study are ordinary, and naming both ``MR_tp0`` made the
    second one a refusal (``grid 'mr_tp0' is already declared``) rather than an
    import.  Single-series modalities keep the short form so nothing already
    written changes its name.
    """
    counts = Counter(entry.modality for entry in entries)
    position: dict[str, int] = {}
    out: list[tuple[Series, str, str]] = []
    for entry in entries:
        base = _safe(entry.modality)
        if counts[entry.modality] > 1:
            position[entry.modality] = position.get(entry.modality, 0) + 1
            base = f"{base}_{position[entry.modality]}"
        out.append((entry, f"{base.lower()}_{tp}", f"{base}_{tp}"))
    return out


class _NothingWritten(Exception):
    """Every planned series of a subject failed to decode: abandon its file."""


def _write_subject(
    group: SubjectGroup,
    path: Path,
    *,
    codec: str,
    log: ConversionReport,
    refused: list[MEDH5ValidationError],
    per_slice_rescale: str = "refuse",
) -> bool:
    """Write one subject's sample to *path*; whether it was written.

    Every series is planned --- its headers read and checked --- before the
    file is created, and one that is refused is left out with a warning and
    its refusal added to *refused*; a subject with no series left gets no
    file.  Each timepoint is declared after its images, so its
    ``series_uids`` names exactly the series written.
    """
    import medh5

    planned = [
        [
            (entry, plan)
            for entry in occasion.payload
            if (plan := _plan(entry, log, refused, per_slice_rescale))
        ]
        for occasion in group.occasions
    ]
    if not any(planned):
        _no_sample(group, path, log)
        return False
    applied = [e.series_uid for o in planned for e, p in o if p.rescales is not None]
    note_instance_ids(group, log)
    days = group.days_from_baseline()
    # The ids are the PatientID --- usually the medical record number --- or,
    # where there was none, the StudyInstanceUID.  Either is an identifier, and
    # saying which is what lets `scrub` report it without guessing from shape.
    source = (
        "dicom:StudyInstanceUID"
        if group.subject_id.startswith(f"{FALLBACK_PREFIX}:")
        else "dicom:PatientID"
    )
    try:
        with medh5.create(
            path,
            sample_id=_safe(group.subject_id),
            subject_id=group.subject_id,
            codec=codec,
        ) as writer:
            writer.identity(**{ID_SOURCE: {"sample_id": source, "subject_id": source}})
            tool = writer.software("medh5", medh5.__version__)
            written = 0
            units: set[str] = set()
            for index, occasion in enumerate(group.occasions):
                written += _write_occasion(
                    writer,
                    tool,
                    group,
                    occasion,
                    index,
                    days[index],
                    planned[index],
                    log,
                    refused,
                    units,
                )
            if not written:
                raise _NothingWritten
            # What the rescale gives is each written image's value_units: HU
            # for a CT unless its RescaleType names another quantity.
            log.decision(
                "value_scale",
                "RescaleSlope/Intercept were stored, not applied; "
                "read(physical=True) returns the rescaled values, in each image's "
                f"value_units ({', '.join(sorted(units))}), and read() returns stored "
                "values (§4.2)"
                + (
                    f"; except in series {', '.join(applied)}, whose per-slice "
                    "rescale was applied (per_slice_rescale)"
                    if applied
                    else ""
                ),
                {"subject": group.subject_id}
                | ({"per_slice_rescale_applied": applied} if applied else {}),
            )
            # Information, not a warning: a warning fails the conversion, and
            # this one was on every conversion, so `convert from-dicom` exited 1
            # on every success.  The validator reports the file (W903).
            log.add(
                "deidentification",
                "no de-identification record was written: this converter does not "
                "de-identify, and a file without the record must be treated as "
                "potentially identifying (§11.4, W903)",
                "info",
                {"subject": group.subject_id},
            )
    except _NothingWritten:
        _no_sample(group, path, log)
        return False
    log.outputs.append(str(path))
    return True


def _plan(
    entry: Series,
    log: ConversionReport,
    refused: list[MEDH5ValidationError],
    per_slice_rescale: str = "refuse",
) -> _Planned | None:
    """*entry* checked for import, or ``None`` when it is refused --- with the
    refusal recorded as a warning and kept in *refused*."""
    try:
        planned = _plan_series(entry, log, per_slice_rescale)
        if planned.samples != 1:
            raise MEDH5ValidationError(
                f"series {entry.series_uid} has {planned.samples} samples per pixel; "
                "this importer reads greyscale series, one value per voxel"
            )
    except MEDH5ValidationError as exc:
        _refuse(entry, exc, log, refused)
        return None
    return planned


def _refuse(
    entry: Series,
    exc: MEDH5ValidationError,
    log: ConversionReport,
    refused: list[MEDH5ValidationError],
) -> None:
    refused.append(exc)
    log.warn(
        "series",
        f"{entry.modality} series {entry.series_uid} "
        f"({entry.description or 'no description'}, {_extent(entry)}) "
        f"was left out: {exc}",
        {
            "series_uid": entry.series_uid,
            "modality": entry.modality,
            "files": len(entry.paths),
        }
        | ({"frames": entry.n_slices} if entry.frames else {})
        | ({"stack": entry.stack} if entry.stack is not None else {}),
    )


def _extent(entry: Series) -> str:
    """How much of the source a series is, for a message."""
    if entry.frames:
        stack = f" (stack {entry.stack})" if entry.stack is not None else ""
        return f"{entry.n_slices} frame(s) in {len(entry.paths)} file(s){stack}"
    return f"{entry.n_slices} file(s)"


def _no_sample(group: SubjectGroup, path: Path, log: ConversionReport) -> None:
    log.warn(
        "subject",
        f"no series of subject {group.subject_id!r} could be imported, so no "
        f"sample was written to {path}",
        {"subject": group.subject_id, "output": str(path)},
    )


def _write_occasion(
    writer: Any,
    tool: Any,
    group: SubjectGroup,
    occasion: Occasion,
    index: int,
    days: Any,
    planned: Sequence[tuple[Series, _Planned]],
    log: ConversionReport,
    refused: list[MEDH5ValidationError],
    units: set[str],
) -> int:
    """Write one visit's series and then its timepoint; how many were written.

    The ``value_units`` of each image written are added to *units*."""
    tp = f"tp{index}"
    written: dict[str, Series] = {}
    # By position, not by SeriesInstanceUID: the stacks of one enhanced object
    # share it.
    names = _series_names([e for e, _ in planned], tp)
    for (entry, plan), (_, grid_id, image_id) in zip(planned, names, strict=True):
        geometry = plan.geometry
        try:
            volume = _decode(plan, entry)
        except MEDH5ValidationError as exc:
            _refuse(entry, exc, log, refused)
            continue
        params: dict[str, Any] = {"slices": entry.n_slices, "modality": entry.modality}
        if entry.stack is not None:
            params["stack"] = entry.stack
        if plan.rescales is not None:
            params["per_slice_rescale"] = "applied"
        activity = writer.activity(
            "import",
            agent=tool,
            tool="medh5 convert from-dicom",
            inputs=[f"dicom:{entry.series_uid}"],
            params=params,
        )
        writer.add_grid(
            grid_id,
            shape=geometry["shape"],
            spacing=geometry["spacing"],
            origin=geometry["origin"],
            direction=geometry["direction"],
            coord_system=geometry["coord_system"],
            units=geometry["units"],
            timepoint=tp,
            frame_uid=geometry["frame_uid"],
        )
        rescale = geometry["rescale"]
        values = _values(entry.modality, geometry["rescale_type"])
        writer.add_image(
            image_id,
            volume,
            grid=grid_id,
            modality=entry.modality,
            **values,
            rescale_slope=None if rescale is None else rescale[0],
            rescale_intercept=None if rescale is None else rescale[1],
            prov=activity,
        )
        if geometry["acquisition"]:
            writer.acquisition(image_id, **geometry["acquisition"])
        written[image_id] = entry
        units.add(values["value_units"] or "no units")
    writer.add_timepoint(
        tp,
        index=index,
        date=_iso(occasion.date),
        days_from_baseline=days,
        study_uid=occasion.key,
        # Keyed by image id, which is what §3.7 says the key is.  Keyed by
        # modality, two series of one modality collapsed to one entry.
        series_uids={image_id: entry.series_uid for image_id, entry in written.items()},
    )
    repeated = sorted(
        modality
        for modality, n in Counter(e.modality for e, _ in planned).items()
        if n > 1
    )
    if repeated:
        log.decision(
            "image_ids",
            f"{group.subject_id} {tp}: more than one series of "
            f"{', '.join(repeated)} in one study; images are numbered in "
            "SeriesInstanceUID order",
            {
                "timepoint": tp,
                "images": {
                    image_id: entry.series_uid
                    for image_id, entry in written.items()
                    if entry.modality in repeated
                },
            },
        )
    return len(written)


def _iso(value: str | None) -> str | None:
    """A DICOM ``YYYYMMDD`` date as ISO ``YYYY-MM-DD``."""
    if not value:
        return None
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def select_series(
    series: Iterable[Series], *, thinnest: bool = True
) -> dict[str, Series]:
    """One series per modality --- the thinnest, or the first seen.

    "Thinnest" means most slices for the same anatomy, which is the usual proxy
    for the reconstruction a reader wants when a study holds several.
    """
    best: dict[str, Series] = {}
    for entry in series:
        current = best.get(entry.modality)
        if current is None or (thinnest and entry.n_slices > current.n_slices):
            best[entry.modality] = entry
    return best


__all__ = [
    "SPACING_TOLERANCE",
    "STORED_TAGS",
    "Series",
    "from_dicom",
    "read_series",
    "require_pydicom",
    "scan_dicom",
    "select_series",
]
