"""DICOM Segmentation import and export (spec §7, §5).

A DICOM SEG stores one binary frame per (segment, source slice), each carrying
its own position, and the frames need not be complete or ordered.  Reading one
therefore means *placing* frames into a volume by their geometry rather than
reshaping them by their order — the same rule as §3.3 everywhere else.

Two things are preserved that a naive conversion loses:

* **Segment identity.**  ``SegmentLabel`` becomes the label-set ``key``, and the
  coded concept (``SegmentedPropertyTypeCodeSequence``) becomes an ontology
  binding when it is present.  Nothing is invented: a SEG with no coded concept
  yields a class with no binding, and W912 says so, which is better than a wrong
  SNOMED code that no validator can detect.
* **Overlap.**  ``SegmentationType`` ``FRACTIONAL`` and overlapping segments are
  both representable — the first as ``probmap``, the second by whichever §7
  encoding the overlap graph calls for. A converter that flattened them into one
  labelmap would silently drop the tumour inside the organ.

Writing needs `highdicom`, which is the reference implementation of the
Segmentation IOD; hand-rolling the per-frame functional groups is how invalid
SEGs get published.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5._optional import require
from medh5.annotations.voxel import TRANSCODABLE
from medh5.errors import MEDH5ValidationError
from medh5.io import _codes
from medh5.io._codes import same_concept
from medh5.io._common import examined_classes, sanitize_key, save_dicom
from medh5.io._frames import _direction, _first_item, _frame_placement, _groups

# Where it was defined, and is still imported from.
from medh5.io._frames import _lattice as _lattice
from medh5.io._patient import (
    _require_same_frame,
    _salt_hint,
    frames_agree,
    patient_mapping,
)
from medh5.io.report import ConversionReport

BINARY = "BINARY"
FRACTIONAL = "FRACTIONAL"

# A DICOM Long String (LO) holds at most 64 characters: `SeriesDescription`.
_LO_LENGTH = 64


def require_highdicom() -> Any:
    """Import highdicom, or raise an ``ImportError`` naming the extra to install."""
    return require("highdicom", extra="dicomseg", purpose="writing DICOM SEG")


SLICE_TOLERANCE = 0.25
"""How far off a slice centre a frame may sit and still be that slice.

A quarter of a slice: comfortably inside the rounding and floating-point error
of an ``ImagePositionPatient``, and far enough from half a slice that a frame
lying *between* two slices is refused rather than assigned to one.
"""


LATTICE_TOLERANCE = 1e-3
"""Unused, and kept so that importing it still works: the planes
:func:`read_dicom_seg` stacks are held to the bound a DICOM series is held to
(:data:`medh5.io.dicom.SPACING_TOLERANCE`, per plane), where each gap was held
to this many millimetres."""


def read_dicom_seg_frames(
    path: str | os.PathLike[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every frame of a SEG, each with the segment and the position it carries.

    The lower-level half of :func:`read_dicom_seg`, and what an import wants.
    A SEG stores one frame per (segment, source slice) and **need not store
    the empty ones**: ``omit_empty_frames=True`` is highdicom's default and
    the common form in the wild, so a file segmenting two slices of an
    eight-slice series has two frames --- and a volume assembled from the
    frames present is a two-slice volume whose through-plane spacing is the
    distance between those two slices.  That is not a smaller reconstruction
    of the study, it is the study with its gaps closed up, and comparing its
    shape to the grid refused an ordinary file for a reason that was not true.
    Frames carry their own ``ImagePositionPatient``, so :func:`place_frames`
    puts each one where it belongs instead.
    """
    from medh5.io.dicom import _dcmread

    dataset = _dcmread(path)
    if getattr(dataset, "Modality", None) != "SEG":
        raise MEDH5ValidationError(
            f"{path} declares Modality={getattr(dataset, 'Modality', None)!r}, "
            "not 'SEG'"
        )
    kind = str(getattr(dataset, "SegmentationType", BINARY))
    if kind not in (BINARY, FRACTIONAL):
        # A LABELMAP frame holds every segment, by pixel value, and names
        # none: read as BINARY, each frame became segment 1.
        raise MEDH5ValidationError(
            f"{path} is a {kind} segmentation; this importer reads BINARY and "
            "FRACTIONAL ones, whose frames each hold one segment"
        )
    segments = _segments(dataset)
    pixels = np.asarray(dataset.pixel_array)
    if pixels.ndim == 2:
        pixels = pixels[None]
    rows, columns = int(dataset.Rows), int(dataset.Columns)
    placement = _frame_placement(dataset, pixels.shape[0])
    numbers = _frame_segments(dataset, pixels.shape[0])
    fractional = kind == FRACTIONAL
    scale = float(getattr(dataset, "MaximumFractionalValue", 255) or 255)

    frames = [
        {
            "index": index,
            "segment": numbers[index],
            "position": placement["positions"][index],
            "orientation": placement["orientations"][index],
            "pixel_spacing": placement["pixel_spacings"][index],
            "plane": placement["indices"][index],
            "data": (
                pixels[index].astype(np.float32) / scale
                if fractional
                else pixels[index].astype(bool)
            ),
        }
        for index in range(pixels.shape[0])
    ]
    geometry: dict[str, Any] = {
        "shape": (max(placement["indices"]) + 1, rows, columns),
        "rows": rows,
        "columns": columns,
        "spacing": placement["spacing"],
        "origin": placement["origin"],
        "direction": placement["direction"],
        "coord_system": "LPS",
        "units": "mm",
        "frame_uid": _first(dataset, "FrameOfReferenceUID"),
        "segments": segments,
        "fractional": fractional,
        "source_series": placement["source_series"],
        "planes": placement["planes"],
        "regular": placement["regular"],
        "spacing_source": placement["spacing_source"],
        "frames": len(frames),
        "scale": scale,
    }
    return frames, geometry


def place_frames(
    frames: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    grid: Any,
    *,
    frame_salt: str | None = None,
) -> dict[int, npt.NDArray[Any]]:
    """Put each frame on the slice of *grid* that its position names (§3.3).

    Shape agreement becomes a question about rows and columns, which a SEG and
    its source series must share, and never about depth, which a SEG with
    omitted frames does not have.  A frame whose position is not a slice of
    this grid is refused **by name** rather than force-fitted: it is either a
    different reconstruction or a different series, and both are answers the
    caller has to act on.

    Before any of that, the frames of reference: a SEG drawn in one is not
    placed on a grid in another, whatever their numbers (§3.4).
    *frame_salt* compares the SEG's frame as ``medh5 scrub`` pseudonymised the
    grid's (see :func:`frames_agree`).
    """
    _require_same_frame(geometry.get("frame_uid"), grid, frame_salt)
    shape = tuple(grid.spatial_shape)
    if len(shape) != 3:
        raise MEDH5ValidationError(
            f"grid {grid.grid_id!r} has {len(shape)} spatial axes; a DICOM SEG "
            "is placed into a 3-D grid",
            code="E405",
        )
    if shape[1:] != (int(geometry["rows"]), int(geometry["columns"])):
        raise MEDH5ValidationError(
            f"the SEG's frames are {geometry['rows']}x{geometry['columns']} but "
            f"grid {grid.grid_id!r} is {shape[1]}x{shape[2]} in plane; it was "
            "drawn on a different reconstruction",
            code="E405",
        )
    signs, scale = _patient_mapping(grid)
    _check_plane(geometry, grid, signs, scale)
    for frame in frames:
        # A frame read from a file carries its own plane; every one is checked,
        # not only the file's first.
        if frame.get("orientation") is not None:
            own = {
                "direction": _direction(np.asarray(frame["orientation"], np.float64)),
                "spacing": [
                    geometry["spacing"][0],
                    *(float(v) for v in frame["pixel_spacing"]),
                ],
            }
            whose = f"SEG frame {frame['index']}'s"
            _check_plane(own, grid, signs, scale, whose=whose)
    volumes = _volumes(geometry, shape)
    for frame in frames:
        segment = int(frame["segment"])
        if segment not in volumes:
            continue
        world = signs * np.asarray(frame["position"], dtype=np.float64) / scale
        index = np.asarray(grid.world_to_index(world), dtype=np.float64).reshape(-1)
        position = float(index[0])
        nearest = int(np.round(position))
        if abs(position - nearest) > SLICE_TOLERANCE or not 0 <= nearest < shape[0]:
            raise MEDH5ValidationError(
                f"SEG frame {frame['index']} (segment {segment}) sits at index "
                f"{position:.3f} along grid {grid.grid_id!r}'s first axis, which "
                "is not one of its slices; the segmentation was drawn on a "
                "different reconstruction",
                code="E405",
            )
        # Its first pixel must be the slice's first voxel, too: rows and
        # columns that match in number and not in place shifted every label.
        if np.any(np.abs(index[1:3]) > SLICE_TOLERANCE):
            raise MEDH5ValidationError(
                f"SEG frame {frame['index']} (segment {segment}) starts at "
                f"in-plane index ({index[1]:.3f}, {index[2]:.3f}) of grid "
                f"{grid.grid_id!r}, not at (0, 0); the segmentation was drawn on a "
                "shifted or different reconstruction",
                code="E405",
            )
        volumes[segment][nearest] = frame["data"]
    return volumes


def _patient_mapping(grid: Any) -> tuple[npt.NDArray[np.float64], float]:
    """:func:`~medh5.io._patient.patient_mapping` for a SEG's frames."""
    return patient_mapping(grid, "a DICOM SEG's frames")


def _check_plane(
    geometry: Mapping[str, Any],
    grid: Any,
    signs: npt.NDArray[np.float64],
    scale: float,
    *,
    whose: str = "the SEG's",
) -> None:
    """The SEG's rows and columns run the way the grid's do, at its spacing.

    Rows and columns were compared by count alone, so a SEG with the source's
    matrix size but another orientation or pixel spacing --- another
    reconstruction --- had its frames laid on this grid as they were.
    """
    seg = signs[:, None] * np.asarray(geometry["direction"], dtype=np.float64)
    ours = np.asarray(grid.direction, dtype=np.float64)
    if not np.allclose(seg[:, 1:], ours[:, 1:], atol=1e-4):
        raise MEDH5ValidationError(
            f"{whose} rows and columns run along {seg[:, 1:].T.round(4).tolist()}, "
            f"grid {grid.grid_id!r}'s along {ours[:, 1:].T.round(4).tolist()}; it "
            "was drawn on a different reconstruction",
            code="E405",
        )
    pixel = np.asarray(geometry["spacing"][1:], dtype=np.float64)
    voxel = np.asarray(grid.spacing[1:], dtype=np.float64) * scale
    if not np.allclose(pixel, voxel, rtol=1e-4, atol=1e-4):
        raise MEDH5ValidationError(
            f"{whose} pixel spacing is {pixel.tolist()} mm, grid "
            f"{grid.grid_id!r}'s {voxel.tolist()} mm; it was drawn on a "
            "different reconstruction",
            code="E405",
        )


def read_dicom_seg(
    path: str | os.PathLike[str],
    *,
    report: ConversionReport | None = None,
) -> tuple[dict[int, npt.NDArray[Any]], dict[str, Any]]:
    """A SEG file as ``{segment number: volume}`` plus its geometry and segments.

    The volume spans the planes the file carries, which is what a reader with
    no other information can honestly build --- on the regular stack they sit
    on, so a plane the file omits between two it carries is an empty plane
    and every plane is where ``geometry`` says (see :func:`_lattice`).  A file
    whose planes no regular stack holds is refused.  A file of one plane has
    no step to measure: its ``SpacingBetweenSlices``, else its
    ``SliceThickness``, is the through-plane spacing, and one stating neither
    is refused --- 1 mm was assumed, without a note.  An import into an
    existing sample goes through :func:`read_dicom_seg_frames` and
    :func:`place_frames` instead, because the grid it is going onto is what
    knows how many slices the study has.
    """
    frames, geometry = read_dicom_seg_frames(path)
    reconstruct = (
        "; read its frames with read_dicom_seg_frames and place them on the grid "
        "they were drawn on with place_frames"
    )
    if not geometry["regular"]:
        planes = [float(p) for p in geometry["planes"]]
        raise MEDH5ValidationError(
            f"the SEG's planes lie at {planes} mm along their normal, and no "
            f"regular stack places them all{reconstruct}"
        )
    if not np.isfinite(geometry["spacing"][0]):
        raise MEDH5ValidationError(
            "the SEG carries one plane and states neither SpacingBetweenSlices "
            "nor SliceThickness, so its volume has no through-plane extent, and "
            f"none is invented{reconstruct}"
        )
    if report is not None and geometry["spacing_source"] == "SliceThickness":
        report.decision(
            "slice_spacing",
            "the SEG carries one plane, with no neighbour to measure a step "
            "against, so its declared SliceThickness "
            f"({geometry['spacing'][0]:.4g} mm) was used as the through-plane "
            "spacing",
            {"spacing": geometry["spacing"][0], "planes": 1},
        )
    volumes = _volumes(geometry, tuple(geometry["shape"]))
    for frame in frames:
        if frame["segment"] in volumes:
            volumes[frame["segment"]][frame["plane"]] = frame["data"]
    if report is not None:
        _seg_notes(report, geometry)
    return volumes, geometry


def _volumes(
    geometry: Mapping[str, Any], shape: tuple[int, ...]
) -> dict[int, npt.NDArray[Any]]:
    """An empty volume of *shape* per segment: float32 for a FRACTIONAL SEG,
    boolean for a BINARY one."""
    dtype = np.float32 if geometry["fractional"] else bool
    return {number: np.zeros(shape, dtype=dtype) for number in geometry["segments"]}


def _seg_notes(report: ConversionReport, geometry: Mapping[str, Any]) -> None:
    """The decisions every SEG read records: segments placed by geometry, and
    a FRACTIONAL SEG read as a probability map."""
    segments = geometry["segments"]
    report.decision(
        "segments",
        f"{len(segments)} segment(s) were read by frame geometry, not by "
        "frame order; a SEG may store its frames in any order",
        {"segments": {k: v["label"] for k, v in segments.items()}},
    )
    if geometry["fractional"]:
        report.decision(
            "fractional",
            "SegmentationType is FRACTIONAL, so the segments became a "
            "`probmap` rather than being thresholded",
            {"scale": float(geometry.get("scale", 255))},
        )


def _frame_segments(dataset: Any, n_frames: int) -> list[int]:
    """The segment each frame holds, by frame."""
    shared, per_frame = _groups(dataset)
    return [
        _segment_number(
            per_frame[index] if index < len(per_frame) else None, shared, index
        )
        for index in range(n_frames)
    ]


def _segment_number(group: Any, shared: Any, index: int) -> int:
    """The segment a frame holds, from its own functional groups or the
    shared ones; a frame that names none is refused, not read as segment 1."""
    for holder in (group, shared):
        identification = _first_item(holder, "SegmentIdentificationSequence")
        number = getattr(identification, "ReferencedSegmentNumber", None)
        if number is not None:
            return int(number)
    raise MEDH5ValidationError(
        f"SEG frame {index} names no segment: no SegmentIdentificationSequence "
        "with a ReferencedSegmentNumber, in its own functional groups or the "
        "shared ones"
    )


def _segments(dataset: Any) -> dict[int, dict[str, Any]]:
    """Segment number → label, algorithm, and the coded property type,
    category and type modifier, where given --- each as the file states it,
    designator included."""
    out: dict[int, dict[str, Any]] = {}
    for item in getattr(dataset, "SegmentSequence", []):
        number = int(item.SegmentNumber)
        out[number] = {
            "label": str(getattr(item, "SegmentLabel", f"segment_{number}")),
            "algorithm": str(getattr(item, "SegmentAlgorithmType", "")),
            "code": _coded(item, "SegmentedPropertyTypeCodeSequence"),
            "category": _coded(item, "SegmentedPropertyCategoryCodeSequence"),
            "modifier": _coded(item, "SegmentedPropertyTypeModifierCodeSequence"),
        }
    return out


def _coded(item: Any, sequence_name: str) -> dict[str, str] | None:
    """The first coded concept of *item*'s *sequence_name*, or ``None``."""
    sequence = getattr(item, sequence_name, None)
    if not sequence:
        return None
    entry = sequence[0]
    code = {
        "scheme": str(getattr(entry, "CodingSchemeDesignator", "")),
        "code": str(getattr(entry, "CodeValue", "")),
        "meaning": str(getattr(entry, "CodeMeaning", "")),
    }
    return code if code["scheme"] and code["code"] else None


def _spelled(code: Mapping[str, str] | None) -> str:
    """A coded concept as the file states it: ``SCT 7771000``, or ``none``."""
    return f"{code['scheme']} {code['code']}" if code else "none"


def _same(first: Mapping[str, str] | None, second: Mapping[str, str] | None) -> bool:
    """Whether two coded concepts of a file, or their absence, are one."""
    if first is None or second is None:
        return first is second
    return same_concept(
        (first["scheme"], first["code"]), (second["scheme"], second["code"])
    )


def _binding(code: Mapping[str, str]) -> Any:
    """A file's coded concept as an ontology binding (§5.2): the system as
    the label set spells it where the table knows the designator, and the
    designator the file states where it does not."""
    from medh5.labels import OntologyCode

    return OntologyCode(
        system=_codes.system(code["scheme"]) or code["scheme"],
        code=code["code"],
        name=code["meaning"] or None,
    )


def _first(dataset: Any, name: str) -> str | None:
    value = getattr(dataset, name, None)
    return None if value is None else str(value)


def from_dicom_seg(
    path: str | os.PathLike[str],
    sample: str | os.PathLike[str],
    *,
    ann_id: str = "seg",
    grid: str | None = None,
    annotated_classes: Sequence[str] | str = "all_given",
    frame_salt: str | None = None,
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Add a DICOM SEG's segments to an existing sample (§7).

    The SEG is matched to a grid by frame of reference where it declares one,
    and its shape is checked against that grid: a SEG drawn on a different
    reconstruction is refused rather than force-fitted.  A grid --- matched or
    named with *grid* --- whose frame of reference is known and is not the
    SEG's is refused before the sample is touched: equal geometry in two
    frames is no correspondence (§3.4).  Where the SEG or the grid states no
    frame, the placement rests on geometry and the report records a guess.
    *frame_salt* is the salt a ``medh5 scrub`` pseudonymised the sample's
    frames with (:func:`frames_agree`).

    Segments that make one class --- matched by SegmentLabel to an existing
    label set, or minted, by the key their labels make --- are combined into
    it: their union, or for ``FRACTIONAL`` their maximum, recorded as a
    decision.

    A class *annotated_classes* names that no segment supplies was examined
    and found absent (§11.3): minted after the segments' classes when the
    sample has no label set, else one its label set must hold (E402).
    """
    import medh5
    from medh5.labels import LabelSet

    log = report or ConversionReport(converter="from-dicom-seg")
    log.source = os.fspath(path)
    frames, geometry = read_dicom_seg_frames(path)

    with medh5.open(sample) as opened:
        grid_id = grid or _match_grid(opened, geometry, log, frame_salt)
        target = opened.grids[grid_id]
        existing = opened.label_set
    # Placed by position, not reshaped by count: a SEG that omits its empty
    # frames covers only the slices it labels, and the grid is what says how
    # many slices the study has (§3.3).
    volumes = place_frames(frames, geometry, target, frame_salt=frame_salt)
    if frames_agree(geometry.get("frame_uid"), target.frame_uid, frame_salt) is None:
        log.guess(
            "frame",
            f"{'the SEG' if not geometry.get('frame_uid') else f'grid {grid_id!r}'} "
            "states no frame of reference, so the frames could not be compared; "
            "the SEG was placed by its geometry alone",
            {"grid": grid_id, "seg_frame": geometry.get("frame_uid")},
        )
    log.decision(
        "frames",
        f"{len(frames)} frame(s) were placed on grid {grid_id!r} by "
        "ImagePositionPatient; a SEG may omit its empty frames and store the "
        "rest in any order",
        {"frames": len(frames), "slices": int(target.spatial_shape[0])},
    )
    _seg_notes(log, geometry)

    segments = geometry["segments"]
    label_set = existing
    if label_set is not None:
        # DICOM segment numbers are positional --- a SEG numbers its segments
        # 1..N whatever the classes mean --- so an existing label set is matched
        # by SegmentLabel, never by number.  Matching on the number would import
        # "lesion" as whatever class happens to hold id 2.
        mapping = _match_segments(segments, label_set)
        log.decision(
            "segment_mapping",
            "segments were matched to the sample's label set by SegmentLabel; "
            "DICOM segment numbers are positional and carry no identity",
            {"mapping": {str(k): v for k, v in mapping.items()}},
        )
    if label_set is None:
        classes, mapping = _mint(segments, log)
        label_set = LabelSet("dicom-seg", version="1.0.0", classes=classes)
        bound = sum(1 for c in classes if c.codes)
        log.decision(
            "label_set",
            f"{len(classes)} class(es) were minted from SegmentLabel; "
            f"{bound} carried a coded concept and got an ontology binding",
            {"bound": bound, "total": len(classes)},
        )
        if bound < len(classes):
            log.guess(
                "ontology",
                "segments without a coded concept were left unbound rather than "
                "given a guessed code: a wrong binding is undetectable (W912)",
                {"unbound": [c.key for c in classes if not c.codes]},
            )
    volumes = _by_class(volumes, mapping, segments, label_set, log, geometry)
    label_set, examined = examined_classes(
        annotated_classes,
        label_set,
        mint=existing is None,
        log=log,
        supplied="the SEG's segments",
    )

    with medh5.amend(sample) as writer:
        if existing is None:
            writer.label_set(label_set)
        tool = writer.software("medh5", medh5.__version__)
        activity = writer.activity(
            "import",
            agent=tool,
            tool="medh5 convert from-dicom-seg",
            inputs=[f"dicom:{_first_or(geometry, 'source_series')}"],
        )
        payload = {number: volumes[number] for number in sorted(volumes)}
        if geometry["fractional"]:
            kind, _ = writer.add_segmentation(
                ann_id,
                grid=grid_id,
                probabilities=payload,
                annotated_classes=examined,
                prov=activity,
            )
        else:
            kind, stats = writer.add_segmentation(
                ann_id,
                grid=grid_id,
                masks=payload,
                annotated_classes=examined,
                prov=activity,
            )
            if stats is not None and stats.edges:
                log.decision(
                    "overlap",
                    f"{len(stats.edges)} segment pair(s) overlap; the encoding "
                    f"chosen ({kind!r}) represents them without flattening",
                    {"overlapping_pairs": len(stats.edges)},
                )
        log.decision("encoding", f"segments were stored as {kind!r}", {"kind": kind})
    log.outputs.append(os.fspath(sample))
    return log


def _mint(
    segments: Mapping[int, dict[str, Any]],
    log: ConversionReport | None = None,
) -> tuple[list[Any], dict[int, int]]:
    """Classes minted from SegmentLabel, and segment number → class id.

    Segments whose labels make one key are one class, its id the first one's
    number: one "Lesion" segment per lesion is common, and minted apart, two
    classes keyed ``lesion`` were refused (E302).  Segments of one key that
    code different concepts are refused: one class cannot be both.  Nor can
    it be two sides: segments of one key whose type modifiers differ (Left
    and Right Kidney, both labelled "Kidney") are refused, where they became
    one class.

    Each class keeps what its first segment states besides the type: the
    property category as ``properties["dicom_category"]``, and as
    ``category`` where the concept stands for one category only (see
    :data:`medh5.io._codes.CATEGORY_OF`); a laterality modifier as
    ``laterality``.  *log* records both, and any modifier that is not a
    laterality, which no field holds.
    """
    from medh5.labels import LabelClass

    first: dict[str, int] = {}
    coded: dict[str, int] = {}
    mapping: dict[int, int] = {}

    def both(a: int, b: int) -> str:
        key = _key(segments[b]["label"])
        return (
            f"segments {a} and {b} make one class {key!r} by SegmentLabel "
            f"({segments[a]['label']!r}, {segments[b]['label']!r})"
        )

    for number, info in sorted(segments.items()):
        key = _key(info["label"])
        lead = first.setdefault(key, number)
        if info["code"] is not None:
            bound = coded.setdefault(key, number)
            if not _same(segments[bound]["code"], info["code"]):
                raise MEDH5ValidationError(
                    f"{both(bound, number)} but code different concepts "
                    f"({_spelled(segments[bound]['code'])}, "
                    f"{_spelled(info['code'])}): one class cannot be both. "
                    "Import it into a sample whose label set holds both"
                )
        if not _same(segments[lead]["modifier"], info["modifier"]):
            raise MEDH5ValidationError(
                f"{both(lead, number)} but carry different type modifiers "
                f"({_spelled(segments[lead]['modifier'])}, "
                f"{_spelled(info['modifier'])}): one class cannot be both. "
                "Give the segments their own labels, or import it into a "
                "sample whose label set holds both"
            )
        mapping[number] = lead

    classes: list[Any] = []
    categories: dict[str, str] = {}
    sides: dict[str, str] = {}
    other: dict[str, str] = {}
    for key, number in first.items():
        info = segments[number]
        code = segments[coded[key]]["code"] if key in coded else None
        properties: dict[str, Any] = {}
        category = None
        if info["category"] is not None:
            properties["dicom_category"] = _binding(info["category"]).to_json()
            category = _codes.CATEGORY_OF.get(_normal(info["category"]))
            if category is not None:
                categories[key] = category
        laterality = None
        if info["modifier"] is not None:
            laterality = _codes.LATERALITY_OF.get(_normal(info["modifier"]))
            if laterality is None:
                other[key] = _spelled(info["modifier"])
            else:
                sides[key] = laterality
        classes.append(
            LabelClass(
                number,
                key,
                info["label"],
                category=category,
                codes=(_binding(code),) if code else (),
                laterality=laterality,
                properties=properties,
            )
        )
    if log is not None and (categories or sides):
        log.decision(
            "concepts",
            f"{len(categories)} minted class(es) took a category from the "
            f"SEG's property category and {len(sides)} a laterality from its "
            "type modifier; every class keeps its property category as "
            "properties['dicom_category']",
            {"categories": categories, "laterality": sides},
        )
    if log is not None and other:
        log.decision(
            "modifier",
            f"{len(other)} segment(s) carry a type modifier that is not a "
            "laterality, which no field of a class holds; it was not carried",
            {"modifiers": other},
        )
    return classes, mapping


def _normal(code: Mapping[str, str]) -> tuple[str, str]:
    """A file's coded concept as ``(designator, code)``, its designator
    normalised through the table."""
    return (
        _codes.designator(code["scheme"]) or code["scheme"],
        code["code"],
    )


def _by_class(
    volumes: Mapping[int, npt.NDArray[Any]],
    mapping: Mapping[int, int],
    segments: Mapping[int, dict[str, Any]],
    label_set: Any,
    log: ConversionReport,
    geometry: Mapping[str, Any],
) -> dict[int, npt.NDArray[Any]]:
    """Segment volumes as class volumes: the segments of one class combined.

    A class mask says where the class is, not which segment drew it, so the
    segments of one class are their union --- their maximum, for fractional
    ones.  Keyed by class, each segment replaced the one before: a SEG with
    one "Lesion" segment per lesion kept only the last lesion's voxels.
    """
    combine = np.maximum if geometry["fractional"] else np.logical_or
    out: dict[int, npt.NDArray[Any]] = {}
    members: dict[int, list[int]] = {}
    for number in sorted(volumes):
        class_id = mapping[number]
        members.setdefault(class_id, []).append(number)
        volume = volumes[number]
        out[class_id] = (
            volume if class_id not in out else combine(out[class_id], volume)
        )
    groups = {c: numbers for c, numbers in members.items() if len(numbers) > 1}
    if groups:
        how = "maximum" if geometry["fractional"] else "union"
        log.decision(
            "segment_union",
            f"{sum(len(g) for g in groups.values())} segments make "
            f"{len(groups)} class(es) and were combined, each class as its "
            f"segments' {how}: a class records where it is, not which segment "
            "drew it",
            {
                "classes": {
                    label_set[class_id].key: [
                        {"segment": n, "label": segments[n]["label"]} for n in numbers
                    ]
                    for class_id, numbers in groups.items()
                },
            },
        )
    return out


def _match_segments(
    segments: Mapping[int, dict[str, Any]], label_set: Any
) -> dict[int, int]:
    """Segment number → class id, resolved through the label set by name."""
    mapping: dict[int, int] = {}
    unmatched: list[str] = []
    for number, info in sorted(segments.items()):
        label = info["label"]
        found = label_set.get(_key(label)) or label_set.get(label)
        if found is None:
            unmatched.append(label)
            continue
        mapping[number] = found.id
    if unmatched:
        raise MEDH5ValidationError(
            f"the sample's label set has no class for segment(s) {unmatched}; add "
            "them, or import into a sample with no label set to mint them",
            code="E402",
        )
    return mapping


def _first_or(geometry: Mapping[str, Any], key: str) -> str:
    return str(geometry.get(key) or "unknown")


def _key(label: str) -> str:
    """A SegmentLabel as a label-set key (§5.2)."""
    return sanitize_key(label, fallback="segment")


def _in_plane_match(grid: Any, geometry: Mapping[str, Any]) -> bool:
    """Whether a grid's rows and columns are the SEG's.

    Depth is deliberately not compared: a SEG that omits its empty frames has
    fewer planes than the series it was drawn on, and matching on the full
    shape rejected exactly the grid it belonged to.
    """
    shape = tuple(grid.spatial_shape)
    return len(shape) == 3 and shape[1:] == (
        int(geometry["rows"]),
        int(geometry["columns"]),
    )


def _match_grid(
    sample: Any,
    geometry: Mapping[str, Any],
    log: ConversionReport,
    salt: str | None = None,
) -> str:
    """Pick the grid the SEG was drawn on, by frame of reference then by shape.

    The shape is a fallback only among grids whose frame cannot be compared
    with the SEG's: one in another known frame was matched on its rows and
    columns alone.
    """
    frame = geometry.get("frame_uid")
    grids = list(sample.grids.values())
    if frame:
        matches = [g for g in grids if frames_agree(frame, g.frame_uid, salt)]
        if len(matches) == 1:
            return str(matches[0].grid_id)
        if len(matches) > 1:
            same = [g for g in matches if _in_plane_match(g, geometry)]
            if len(same) == 1:
                return str(same[0].grid_id)
    candidates = [
        g
        for g in grids
        if _in_plane_match(g, geometry)
        and frames_agree(frame, g.frame_uid, salt) is None
    ]
    if len(candidates) == 1:
        log.guess(
            "grid",
            f"the SEG names frame {frame!r}, which no grid declares; it was "
            f"matched to grid {candidates[0].grid_id!r}, which states no frame, "
            "on shape alone",
            {"grid": candidates[0].grid_id},
        )
        return str(candidates[0].grid_id)
    elsewhere = [
        str(g.grid_id)
        for g in grids
        if _in_plane_match(g, geometry)
        and frames_agree(frame, g.frame_uid, salt) is False
    ]
    raise MEDH5ValidationError(
        f"cannot tell which grid the SEG belongs to: frame {frame!r} matches no "
        f"grid and {len(candidates)} grid(s) without a frame share its rows and "
        "columns"
        + (
            f"; {elsewhere} share them in another frame of reference, which is "
            "no correspondence"
            if elsewhere
            else ""
        )
        + ". Pass grid= explicitly rather than letting the converter guess.",
        code="E101",
    )


def to_dicom_seg(
    sample: str | os.PathLike[str],
    ann_id: str,
    source_images: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    classes: Sequence[int | str] | None = None,
    series_description: str = "medh5 segmentation",
    frame_salt: str | None = None,
    report: ConversionReport | None = None,
    segmentation_type: str = "binary",
    fractional_type: str = "PROBABILITY",
) -> Path:
    """Export one voxel annotation as a DICOM SEG against its source slices.

    A SEG holds one segment per class, so the annotation is a ``labelmap``,
    ``layers``, ``bitmask``, ``instances`` or ``probmap``; any other kind is
    refused (E401) before anything is read --- a ``mask`` too, having no
    classes.

    *segmentation_type* ``"binary"`` writes a ``BINARY`` SEG from
    ``dense()``, which thresholds a ``probmap`` at its stored ``threshold``
    (recorded as a ``threshold`` decision).  ``"fractional"`` writes a
    ``probmap``'s probabilities as a ``FRACTIONAL`` SEG of *fractional_type*
    (``PROBABILITY`` or ``OCCUPANCY``), quantised to 1/255 --- within 1/510
    of each value --- and any other kind is refused; the report states the
    error, how many voxels the quantisation moved across the stored
    threshold, and that ``threshold`` and ``normalized`` are not carried.

    Each segment carries its class's concepts: the property type is the
    class's first code whose system :mod:`medh5.io._codes` designates, the
    property category ``properties["dicom_category"]`` or the one its
    ``category`` stands for, a ``laterality`` the type modifier, and the
    algorithm type the annotation's provenance says (``predict`` →
    ``AUTOMATIC``, naming its software).  DICOM requires a type, a category
    and an algorithm type, so where the sample states none the generic one
    is written --- SCT 91723000 "Anatomical Structure", ``MANUAL`` --- and
    the report records a guess naming each class it was written for.  A
    designator is never invented.

    *source_images* are the DICOM files the segmentation was drawn on: a SEG is
    defined by reference to them, and one written without them is orphaned in
    every PACS that receives it.  They must be the annotation grid's slices,
    in its frame of reference where the grid states one (E414 otherwise;
    *frame_salt* compares a frame ``medh5 scrub`` pseudonymised, as
    :func:`frames_agree` does) and on its lattice (E405 otherwise).

    An annotation with an ignore region (§7.7) is refused: a segment's 0
    says a voxel was examined and found absent, and it has no value for one
    nobody examined.  For the same reason a segment is
    written only for a class the annotation examined (``annotated_class_ids``):
    by default the others are left out, and *classes* naming one is refused.
    Every class went out, and one nobody looked for became an empty segment ---
    a verified negative to whoever read it back.

    *series_description* is the ``SeriesDescription``, a DICOM LO: one longer
    than 64 characters is refused before anything is read.
    """
    import medh5

    hd = require_highdicom()
    from medh5.io.dicom import _dcmread

    if len(series_description) > _LO_LENGTH:
        raise MEDH5ValidationError(
            f"series_description={series_description!r} "
            f"(--series-description) is {len(series_description)} characters, "
            "and the SeriesDescription it is written as is a DICOM LO of at "
            f"most {_LO_LENGTH}: shorten it"
        )
    log = report or ConversionReport(converter="to-dicom-seg")
    log.source = os.fspath(sample)
    fractional = _segmentation_type(segmentation_type, fractional_type)

    with medh5.open(sample) as opened:
        annotation = opened.annotations[ann_id]
        _require_segments(annotation, ann_id)
        if fractional and annotation.kind != "probmap":
            raise MEDH5ValidationError(
                f"annotation {ann_id!r} is a {annotation.kind!r}, which holds no "
                "fractional values; a FRACTIONAL SEG is written from a 'probmap'. "
                "Export it with segmentation_type='binary' "
                "(`--segmentation-type binary`, the default)",
                code="E401",
            )
        datasets = [_dcmread(p) for p in source_images]
        if not datasets:
            raise MEDH5ValidationError("to_dicom_seg needs the source DICOM images")
        normal = _source_normal(datasets[0])
        datasets.sort(key=lambda d: float(np.dot(_ipp(d), normal)))
        slices = _source_slices(annotation.grid, datasets, log, frame_salt)
        ids = _examined(annotation, ann_id, classes, log)
        ignored = int(np.count_nonzero(opened.ignore_region(ann_id)))
        if ignored:
            raise MEDH5ValidationError(
                f"annotation {ann_id!r} ignores {ignored} voxel(s) (§7.7), "
                "which a SEG cannot mark --- a segment's 0 would say "
                "they were examined and found absent; export it to NIfTI "
                "instead (`to_nifti` with annotation=, `medh5 convert "
                "to-nifti --annotation`), whose labelmap writes them as its "
                f"ignore id {annotation.ignore_id}"
            )
        contains = np.asarray(annotation.dense(list(ids)), dtype=bool)
        if fractional:
            values = np.asarray(annotation.probabilities(list(ids)), np.float32)
            _quantisation(values, contains, float(annotation.threshold), log)
            options: dict[str, Any] = {
                "segmentation_type": hd.seg.SegmentationTypeValues.FRACTIONAL,
                "fractional_type": fractional_type,
                "max_fractional_value": MAX_FRACTIONAL_VALUE,
            }
        else:
            values = contains
            if annotation.kind == "probmap":
                log.decision(
                    "threshold",
                    f"the probmap was thresholded at {annotation.threshold:g}, "
                    "its stored threshold, into a BINARY SEG; its probabilities "
                    "are not written (segmentation_type='fractional' writes "
                    "them)",
                    {"threshold": float(annotation.threshold)},
                )
            options = {"segmentation_type": hd.seg.SegmentationTypeValues.BINARY}
        algorithm_type, algorithm = _algorithm(hd, opened, annotation, ann_id, log)
        descriptions = _descriptions(
            hd, opened.label_set, annotation, ids, algorithm_type, algorithm, log
        )
        # highdicom wants (frames, rows, columns, segments), a frame per
        # source image in the order given; `dense` returns (segments, z, y, x).
        pixels = np.transpose(values, (1, 2, 3, 0))[slices]
        segmentation = hd.seg.Segmentation(
            source_images=datasets,
            pixel_array=pixels,
            segment_descriptions=descriptions,
            series_instance_uid=hd.UID(),
            series_number=1,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer="medh5",
            manufacturer_model_name="medh5",
            software_versions=medh5.__version__,
            device_serial_number="0",
            series_description=series_description,
            omit_empty_frames=False,
            **options,
        )
    target = Path(os.fspath(out))
    save_dicom(segmentation, target)
    log.decision(
        "segments",
        f"{len(ids)} class(es) were written as separate segments, so overlap "
        "survives the export",
        {"classes": [str(i) for i in ids]},
    )
    log.outputs.append(str(target))
    return target


SEGMENTATION_TYPES = ("binary", "fractional")
"""What :func:`to_dicom_seg`'s *segmentation_type* takes."""

FRACTIONAL_TYPES = ("PROBABILITY", "OCCUPANCY")
"""What :func:`to_dicom_seg`'s *fractional_type* takes (DICOM's
``SegmentationFractionalType``)."""

MAX_FRACTIONAL_VALUE = 255
"""The stored value of a probability of 1 in a FRACTIONAL SEG this package
writes: values are quantised to 1/255, so within 1/510 of what was given."""


def _segmentation_type(segmentation_type: str, fractional_type: str) -> bool:
    """Whether the export is FRACTIONAL; an unknown type is refused."""
    if segmentation_type not in SEGMENTATION_TYPES:
        raise MEDH5ValidationError(
            f"segmentation_type is {segmentation_type!r}; it is one of "
            f"{', '.join(map(repr, SEGMENTATION_TYPES))}"
        )
    if fractional_type not in FRACTIONAL_TYPES:
        raise MEDH5ValidationError(
            f"fractional_type is {fractional_type!r}; it is one of "
            f"{', '.join(map(repr, FRACTIONAL_TYPES))}"
        )
    return segmentation_type == "fractional"


def _quantisation(
    values: npt.NDArray[np.float32],
    contains: npt.NDArray[np.bool_],
    threshold: float,
    log: ConversionReport,
) -> None:
    """Record what writing *values* at 1/255 does to them.

    The stored values are ``round(p * 255)``, as highdicom writes them, so
    each comes back within 1/510.  A voxel whose class the annotation
    *contains* (``p >= threshold``, decided at its stored precision) may come
    back on the other side of that threshold; how many do is the measure of
    what a reader who thresholds the SEG at the same value would see change.
    """
    stored = np.around(values * float(MAX_FRACTIONAL_VALUE))
    back = stored.astype(np.float64) / MAX_FRACTIONAL_VALUE
    error = float(np.max(np.abs(back - values))) if values.size else 0.0
    changed = int(np.count_nonzero((back >= threshold) != contains))
    log.decision(
        "fractional",
        "the probabilities were written as a FRACTIONAL SEG, quantised to "
        f"1/{MAX_FRACTIONAL_VALUE}: each value is within 1/"
        f"{2 * MAX_FRACTIONAL_VALUE} of the stored one (at most {error:.3g} "
        f"here), and {changed} voxel(s) moved across the stored threshold "
        f"({threshold:g}). A SEG has no field for the probmap's threshold or "
        "normalized flag, so neither is carried",
        {
            "bound": 1 / (2 * MAX_FRACTIONAL_VALUE),
            "max_error": error,
            "changed_side": changed,
            "threshold": threshold,
            "max_fractional_value": MAX_FRACTIONAL_VALUE,
        },
    )


def _algorithm(
    hd: Any, sample: Any, annotation: Any, ann_id: str, log: ConversionReport
) -> tuple[Any, Any]:
    """The SEG's algorithm type and identification, from the annotation's
    provenance (§11.1).

    A ``predict`` activity whose agent is software of a stated version is
    ``AUTOMATIC``, identified by that software, family DCM 123110
    "Artificial Intelligence"; ``annotate`` and ``review`` are ``MANUAL``.
    Anything else is ``MANUAL`` too --- DICOM requires a value --- and
    recorded as a guess.
    """
    manual = hd.seg.SegmentAlgorithmTypeValues.MANUAL
    provenance = sample.document.provenance
    prov = annotation.prov
    activity = (
        provenance.activity(prov) if prov and provenance.has_activity(prov) else None
    )

    def generic(why: str) -> tuple[Any, Any]:
        log.guess(
            "algorithm_type",
            f"{why}; SegmentAlgorithmType MANUAL was written, the generic "
            "value, because DICOM requires one",
            {"algorithm_type": "MANUAL", "prov": prov},
        )
        return manual, None

    if activity is None:
        return generic(
            f"annotation {ann_id!r} has no prov"
            if not prov
            else f"annotation {ann_id!r}'s prov {prov!r} names no activity"
        )
    if activity.type == "predict":
        agent = (
            provenance.agent(activity.agent)
            if activity.agent and any(a.id == activity.agent for a in provenance.agents)
            else None
        )
        if agent is None or agent.type != "software" or not agent.version:
            return generic(
                f"annotation {ann_id!r} was predicted (activity "
                f"{activity.id!r}), but the activity names no software agent "
                "with a version, which an AUTOMATIC segment's algorithm "
                "identification requires"
            )
        family = _codes.ARTIFICIAL_INTELLIGENCE
        identification = hd.AlgorithmIdentificationSequence(
            name=agent.name,
            version=agent.version,
            family=hd.sr.CodedConcept(*family),
        )
        log.decision(
            "algorithm_type",
            f"annotation {ann_id!r} was predicted by {agent.name} "
            f"{agent.version} (activity {activity.id!r}), so its segments are "
            "AUTOMATIC, identified by that software",
            {
                "algorithm_type": "AUTOMATIC",
                "activity": activity.id,
                "name": agent.name,
                "version": agent.version,
            },
        )
        return hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC, identification
    if activity.type in ("annotate", "review"):
        log.decision(
            "algorithm_type",
            f"annotation {ann_id!r} comes from an {activity.type!r} activity "
            f"({activity.id!r}), so its segments are MANUAL",
            {"algorithm_type": "MANUAL", "activity": activity.id},
        )
        return manual, None
    return generic(
        f"annotation {ann_id!r} comes from a {activity.type!r} activity "
        f"({activity.id!r}), which says neither that a person drew it nor "
        "that a model predicted it"
    )


def _descriptions(
    hd: Any,
    label_set: Any,
    annotation: Any,
    ids: Sequence[int],
    algorithm_type: Any,
    algorithm: Any,
    log: ConversionReport,
) -> list[Any]:
    """One segment description per class, carrying its concepts (§5.2).

    The property type is the class's first code whose system the table
    designates; the category its ``properties["dicom_category"]``, else the
    concept its ``category`` stands for; its ``laterality`` the type modifier
    (0062,0011), which highdicom takes no parameter for.  A type or category
    the class does not state is the generic concept, as a guess naming each
    class.
    """
    generic = hd.sr.CodedConcept(*_codes.GENERIC)
    untyped: list[str] = []
    unknown: dict[str, list[str]] = {}
    passed_over: dict[str, list[str]] = {}
    meanings: list[str] = []
    uncategorised: dict[str, str | None] = {}
    lateral: dict[str, str] = {}
    descriptions = []
    for number, class_id in enumerate(ids, start=1):
        key = annotation.class_key(class_id)
        entry = label_set[class_id] if label_set is not None else None
        codes = tuple(entry.codes) if entry is not None else ()
        named = [c for c in codes if _codes.designator(c.system)]
        strange = [str(c.system) for c in codes if not _codes.designator(c.system)]
        if named and entry is not None:
            code = named[0]
            if code is not codes[0]:
                passed_over[key] = strange
            if not code.name:
                meanings.append(key)
            kind = hd.sr.CodedConcept(
                str(code.code),
                str(_codes.designator(code.system)),
                str(code.name or entry.name),
            )
        else:
            untyped.append(key)
            if strange:
                unknown[key] = strange
            kind = generic
        category = _category(hd, entry)
        if category is None:
            uncategorised[key] = None if entry is None else entry.category
            category = generic
        description = hd.seg.SegmentDescription(
            segment_number=number,
            segment_label=key,
            segmented_property_category=category,
            segmented_property_type=kind,
            algorithm_type=algorithm_type,
            algorithm_identification=algorithm,
        )
        side = None if entry is None else entry.laterality
        if side in _codes.LATERALITY:
            description.SegmentedPropertyTypeModifierCodeSequence = [
                hd.sr.CodedConcept(*_codes.LATERALITY[side])
            ]
            lateral[key] = side
        descriptions.append(description)
    if untyped:
        log.guess(
            "property_type",
            f"{len(untyped)} class(es) state no code a DICOM designator names "
            f"({', '.join(untyped)}), so their segments' property type is the "
            "generic SCT 91723000 'Anatomical Structure'; DICOM requires one, "
            "and a designator is never invented"
            + (f" (systems the table does not know: {unknown})" if unknown else ""),
            {"classes": untyped, "unknown_systems": unknown},
        )
    if passed_over:
        log.decision(
            "property_type_system",
            f"{len(passed_over)} class(es) took their property type from a "
            "later code, their first naming a system no DICOM designator "
            "names",
            {"passed_over": passed_over},
        )
    if meanings:
        log.decision(
            "code_meaning",
            f"{len(meanings)} class(es)' code states no name, so the class's "
            "display name is its CodeMeaning",
            {"classes": meanings},
        )
    if uncategorised:
        log.guess(
            "property_category",
            f"{len(uncategorised)} class(es) state no category a SEG property "
            f"category stands for ({', '.join(uncategorised)}), so their "
            "segments' category is the generic SCT 91723000 'Anatomical "
            "Structure'; DICOM requires one",
            {"classes": list(uncategorised), "categories": uncategorised},
        )
    if lateral:
        log.decision(
            "laterality",
            f"{len(lateral)} class(es)' laterality was written as the "
            "segment's property type modifier",
            {"laterality": lateral},
        )
    return descriptions


def _category(hd: Any, entry: Any) -> Any:
    """A class's SEG property category, or ``None`` where it states none: its
    ``properties["dicom_category"]`` (an import's), else the concept its
    ``category`` stands for (:data:`medh5.io._codes.CATEGORIES`)."""
    if entry is None:
        return None
    stated = entry.properties.get("dicom_category")
    if isinstance(stated, Mapping):
        scheme = _codes.designator(str(stated.get("system") or ""))
        code, meaning = stated.get("code"), stated.get("name")
        if scheme and code and meaning:
            return hd.sr.CodedConcept(str(code), scheme, str(meaning))
    concept = _codes.CATEGORIES.get(str(entry.category or ""))
    return None if concept is None else hd.sr.CodedConcept(*concept)


def _examined(
    annotation: Any,
    ann_id: str,
    classes: Sequence[int | str] | None,
    log: ConversionReport,
) -> tuple[int, ...]:
    """The classes a SEG export writes: *classes*, or every class examined.

    A segment's 0 says "examined, absent" (§11.3), which a class outside
    ``annotated_class_ids`` never was: by default it is left out, as a
    decision, and named in *classes* it is refused.
    """
    looked = set(annotation.annotated_class_ids)
    if classes is not None:
        ids = tuple(annotation.resolve_classes(classes))
        never = [annotation.class_key(c) for c in ids if c not in looked]
        if never:
            raise MEDH5ValidationError(
                f"annotation {ann_id!r} never examined {never} (they are not in "
                "its annotated_class_ids), and a segment's 0 would say they were "
                "examined and found absent; export the classes it examined"
            )
        return ids
    every = tuple(annotation.resolve_classes(None))
    ids = tuple(c for c in every if c in looked)
    left = [annotation.class_key(c) for c in every if c not in looked]
    if not ids:
        raise MEDH5ValidationError(
            f"annotation {ann_id!r} examined none of its classes "
            "(annotated_class_ids is empty), so a SEG of it would assert "
            "absences nobody observed"
        )
    if left:
        log.decision(
            "unexamined",
            f"{len(left)} class(es) the annotation never examined were left out "
            f"({', '.join(left)}): a segment's 0 says examined and absent",
            {"left_out": left},
        )
    return ids


#: The kinds a SEG export writes: those with classes, one segment each ---
#: the kinds the engine transcodes between.
_SEGMENT_KINDS: tuple[str, ...] = TRANSCODABLE


def _require_segments(annotation: Any, ann_id: str) -> None:
    """Refuse an annotation a SEG cannot hold (E401).

    A SEG holds one segment per class.  A geometric annotation has no voxels
    to write and a ``mask`` no class to write them as (§6.3); the export met
    them as an ``AttributeError`` and an ``IndexError`` out of highdicom.
    """
    if annotation.kind in _SEGMENT_KINDS:
        return
    what = (
        "a 'mask', which has no classes (§6.3)"
        if annotation.kind == "mask"
        else f"{annotation.kind!r}, not a voxel annotation"
    )
    kinds = ", ".join(repr(kind) for kind in _SEGMENT_KINDS[:-1])
    raise MEDH5ValidationError(
        f"annotation {ann_id!r} is {what}; a SEG holds one segment per class "
        f"of a {kinds} or {_SEGMENT_KINDS[-1]!r} annotation",
        code="E401",
    )


def _source_slices(
    grid: Any,
    datasets: Sequence[Any],
    log: ConversionReport,
    salt: str | None = None,
) -> list[int]:
    """The slice of *grid* each source image is, in the images' order.

    A SEG is written against its source images: highdicom gives frame ``k``
    the geometry of image ``k``, so the annotation's grid must be theirs ---
    one frame of reference, rows and columns running the same way at the same
    spacing, and each image one of its slices whose first pixel is the
    slice's first voxel.  The export compared nothing, so an annotation on a
    grid turned in plane, or shifted 10 mm in the same frame, was written
    onto the source geometry, centimetres from where it was drawn.  A grid
    that is not the images' is refused; nothing is resampled.  Another frame
    of reference is E414, as on import; *salt* compares the images' frames
    as ``medh5 scrub`` pseudonymised the grid's.
    """
    signs, scale = _patient_mapping(grid)
    shape = tuple(grid.spatial_shape)
    if len(shape) != 3 or len(datasets) != shape[0]:
        raise MEDH5ValidationError(
            f"grid {grid.grid_id!r} is {shape} and {len(datasets)} source "
            "image(s) were given: a SEG is written a frame per source image, "
            "so they must be the grid's slices, one each",
            code="E405",
        )
    frames = sorted(
        {str(getattr(d, "FrameOfReferenceUID", "") or "") for d in datasets}
    )
    if grid.frame_uid and not all(
        frames_agree(f, grid.frame_uid, salt) for f in frames
    ):
        hint = _salt_hint(grid.frame_uid, salt, "The grid's", "the images' frame")
        raise MEDH5ValidationError(
            f"grid {grid.grid_id!r} is in frame {grid.frame_uid!r} and the source "
            f"images in {[f or None for f in frames]}; a segmentation is not "
            f"placed across frames of reference.{hint}",
            code="E414",
        )
    if not grid.frame_uid:
        log.guess(
            "frame_of_reference",
            f"grid {grid.grid_id!r} declares no frame_uid, so the source images "
            "were matched to it by position alone",
            {"grid": grid.grid_id},
        )
    order: list[int] = []
    for k, dataset in enumerate(datasets):
        whose = f"source image {k}'s"
        if (int(dataset.Rows), int(dataset.Columns)) != shape[1:]:
            raise MEDH5ValidationError(
                f"{whose} matrix is {int(dataset.Rows)}x{int(dataset.Columns)} and "
                f"grid {grid.grid_id!r}'s {shape[1]}x{shape[2]}; it is another "
                "reconstruction",
                code="E405",
            )
        orientation = np.asarray(
            [float(v) for v in dataset.ImageOrientationPatient], dtype=np.float64
        )
        own = {
            "direction": _direction(orientation),
            "spacing": [0.0, *(float(v) for v in dataset.PixelSpacing)],
        }
        _check_plane(own, grid, signs, scale, whose=whose)
        world = signs * _ipp(dataset) / scale
        index = np.asarray(grid.world_to_index(world), dtype=np.float64).reshape(-1)
        nearest = int(np.round(index[0]))
        if (
            abs(index[0] - nearest) > SLICE_TOLERANCE
            or not 0 <= nearest < shape[0]
            or np.any(np.abs(index[1:3]) > SLICE_TOLERANCE)
        ):
            raise MEDH5ValidationError(
                f"source image {k} starts at index "
                f"({index[0]:.3f}, {index[1]:.3f}, {index[2]:.3f}) of grid "
                f"{grid.grid_id!r}, which is not the first voxel of one of its "
                "slices; the annotation was drawn on another grid",
                code="E405",
            )
        order.append(nearest)
    if sorted(order) != list(range(shape[0])):
        raise MEDH5ValidationError(
            f"the source images fall on slices {order} of grid {grid.grid_id!r}; "
            "a SEG needs each slice once",
            code="E405",
        )
    return order


def _source_normal(dataset: Any) -> npt.NDArray[np.float64]:
    orientation = np.asarray(
        [float(v) for v in dataset.ImageOrientationPatient], dtype=np.float64
    )
    return np.cross(orientation[:3], orientation[3:])


def _ipp(dataset: Any) -> npt.NDArray[np.float64]:
    return np.asarray(
        [float(v) for v in dataset.ImagePositionPatient], dtype=np.float64
    )


__all__ = [
    "BINARY",
    "FRACTIONAL",
    "FRACTIONAL_TYPES",
    "MAX_FRACTIONAL_VALUE",
    "SEGMENTATION_TYPES",
    "SLICE_TOLERANCE",
    "frames_agree",
    "from_dicom_seg",
    "place_frames",
    "read_dicom_seg",
    "read_dicom_seg_frames",
    "require_highdicom",
    "to_dicom_seg",
]
