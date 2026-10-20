"""Where a multi-frame DICOM object's frames lie: the functional groups.

A multi-frame object --- a Segmentation, an enhanced CT, MR or PET image ---
places each frame through functional groups that may sit in the shared item
or in the frame's own (PS3.3 C.7.6.16).  These read them: each frame's
position, orientation and pixel spacing, any other attribute a group holds,
and the regular stack the frames' planes sit on.  They were
:mod:`medh5.io.dicom_seg`'s own; one copy serves every reader that places
frames, and what only a SEG has --- the segment a frame holds --- stays with
the SEG reader.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError


def _frame_placement(dataset: Any, n_frames: int) -> dict[str, Any]:
    """Where each frame belongs, from the per-frame functional groups.

    A functional group may sit in the shared item or in each frame's own
    (PS3.3 C.7.6.16), so every frame's orientation and pixel spacing are read
    where they are --- and the frames must agree on them, being one volume's
    slices.  The first orientation found and the shared spacing stood for
    every frame: a frame turned in plane was placed as the first one was, and
    spacing kept in the frames read as 1 mm.
    """
    shared, per_frame = _groups(dataset)
    positions: list[npt.NDArray[np.float64]] = []
    orientations: list[npt.NDArray[np.float64]] = []
    pixels: list[npt.NDArray[np.float64]] = []
    for index in range(n_frames):
        group = per_frame[index] if index < len(per_frame) else None
        frame = f"SEG frame {index}"
        positions.append(_position(group, shared, frame))
        orientations.append(_orientation(group, shared, frame))
        pixels.append(_pixel_spacing(group, shared, frame))
    _congruent(orientations, pixels)
    orientation = orientations[0]
    normal = np.cross(orientation[:3], orientation[3:])
    projected = [float(np.dot(p, normal)) for p in positions]
    planes = sorted(set(np.round(projected, 4)))
    first = per_frame[0] if per_frame else None
    between = _between(first, shared)
    regular = _lattice(planes, between)
    # An import places every frame by its own position (`place_frames`), so a
    # stack no regular lattice holds is still placeable; only a volume of the
    # file's own (`read_dicom_seg`) needs one.
    step, lattice = regular or (float("nan"), list(range(len(planes))))
    source: str | None = "SpacingBetweenSlices" if between else "positions"
    if regular is None:
        source = None
    elif len(planes) == 1 and not between:
        # One plane has no neighbour to measure a step against: the slab it
        # states is its only extent, as for a lone DICOM slice; stating none,
        # it has none (`read_dicom_seg` refuses it), where 1 mm was assumed.
        step = _thickness(first, shared)
        source = "SliceThickness" if np.isfinite(step) else None
    indices = [lattice[planes.index(round(v, 4))] for v in projected]
    spacing = [step, float(pixels[0][0]), float(pixels[0][1])]
    origin_index = indices.index(0) if 0 in indices else 0
    return {
        "planes": planes,
        "regular": regular is not None,
        "spacing_source": source,
        "indices": indices,
        # Each frame's own `ImagePositionPatient`, kept rather than collapsed
        # into a plane index: placing a frame on the grid it is being imported
        # onto needs the position, and the index is only meaningful among the
        # planes this file happens to carry.
        "positions": positions,
        "orientations": orientations,
        "pixel_spacings": pixels,
        "spacing": spacing,
        "origin": [float(v) for v in positions[origin_index]],
        "direction": _direction(orientation),
        "source_series": _source_series(dataset),
    }


def _direction(orientation: npt.NDArray[np.float64]) -> list[list[float]]:
    """A frame's `ImageOrientationPatient` as a (slice, row, column) direction."""
    normal = np.cross(orientation[:3], orientation[3:])
    return [
        [float(v) for v in row]
        for row in np.stack([normal, orientation[3:], orientation[:3]], axis=1)
    ]


def _groups(dataset: Any) -> tuple[Any, list[Any]]:
    """An object's shared functional group item and its per-frame items."""
    shared = _first_item(dataset, "SharedFunctionalGroupsSequence")
    return shared, list(getattr(dataset, "PerFrameFunctionalGroupsSequence", []))


def _numbers(
    value: Any, count: int, frame: str, keyword: str
) -> npt.NDArray[np.float64]:
    """*value*, which *frame* states as *keyword*, as *count* finite numbers,
    or a refusal naming the frame: a component that is not a number would
    otherwise escape as a ``ValueError`` and stop every series in the tree."""
    try:
        numbers = np.asarray([float(v) for v in value], dtype=np.float64)
    except (TypeError, ValueError):
        numbers = None
    if numbers is None or numbers.shape != (count,) or not np.isfinite(numbers).all():
        raise MEDH5ValidationError(
            f"{frame} states {keyword} as {value!r}, which is not {count} "
            "finite numbers, so it cannot be placed"
        )
    return numbers


def _orientation(group: Any, shared: Any, frame: str) -> npt.NDArray[np.float64]:
    for holder in (group, shared):
        plane = _first_item(holder, "PlaneOrientationSequence")
        if plane is not None and hasattr(plane, "ImageOrientationPatient"):
            return _numbers(
                plane.ImageOrientationPatient, 6, frame, "ImageOrientationPatient"
            )
    raise MEDH5ValidationError(
        f"{frame} carries no ImageOrientationPatient, in its own functional "
        "groups or the shared ones, so it cannot be placed"
    )


def _position(group: Any, shared: Any, frame: str) -> npt.NDArray[np.float64]:
    for holder in (group, shared):
        plane = _first_item(holder, "PlanePositionSequence")
        if plane is not None and hasattr(plane, "ImagePositionPatient"):
            return _numbers(
                plane.ImagePositionPatient, 3, frame, "ImagePositionPatient"
            )
    raise MEDH5ValidationError(
        f"{frame} carries no ImagePositionPatient, in its own functional groups "
        "or the shared ones, so it cannot be placed"
    )


def _pixel_spacing(group: Any, shared: Any, frame: str) -> npt.NDArray[np.float64]:
    for holder in (group, shared):
        measures = _first_item(holder, "PixelMeasuresSequence")
        if measures is not None and getattr(measures, "PixelSpacing", None):
            return _numbers(measures.PixelSpacing, 2, frame, "PixelSpacing")
    raise MEDH5ValidationError(
        f"{frame} carries no PixelSpacing, in its own functional groups or the "
        "shared ones; where its pixels lie is not stated, and it is refused "
        "rather than read as 1 mm"
    )


def _between(group: Any, shared: Any) -> float | None:
    for holder in (group, shared):
        measures = _first_item(holder, "PixelMeasuresSequence")
        if measures is not None and getattr(measures, "SpacingBetweenSlices", None):
            try:
                between = float(measures.SpacingBetweenSlices)
            except (TypeError, ValueError):
                return None
            return between if np.isfinite(between) else None
    return None


def _thickness(group: Any, shared: Any) -> float:
    """The ``SliceThickness`` a frame's pixel measures state, NaN for none."""
    for holder in (group, shared):
        measures = _first_item(holder, "PixelMeasuresSequence")
        value = getattr(measures, "SliceThickness", None) if measures else None
        try:
            thickness = abs(float(value)) if value is not None else 0.0
        except (TypeError, ValueError):
            thickness = 0.0
        if np.isfinite(thickness) and thickness > 0:
            return thickness
    return float("nan")


def _congruent(
    orientations: Sequence[npt.NDArray[np.float64]],
    pixels: Sequence[npt.NDArray[np.float64]],
) -> None:
    """Every frame oriented and spaced as the first: one volume's slices.

    About the SEG alone, so no §15.2 code: the codes describe a sample, and
    this refusal comes before there is one.
    """
    pairs = zip(orientations, pixels, strict=True)
    for index, (orientation, pixel) in enumerate(pairs):
        if not np.allclose(orientation, orientations[0], atol=1e-4):
            raise MEDH5ValidationError(
                f"SEG frame {index} is oriented {orientation.round(4).tolist()} and "
                f"frame 0 {orientations[0].round(4).tolist()}: frames turned "
                "against each other are not one volume's slices"
            )
        if not np.allclose(pixel, pixels[0], rtol=1e-4, atol=1e-6):
            raise MEDH5ValidationError(
                f"SEG frame {index} has pixel spacing {pixel.tolist()} mm and frame "
                f"0 {pixels[0].tolist()} mm: frames of different spacing are not "
                "one volume's slices"
            )


def _lattice(
    planes: Sequence[float], between: float | None
) -> tuple[float, list[int]] | None:
    """The regular stack the planes a SEG carries sit on: its step, and each
    plane's index in it; ``None`` when no regular stack holds them.

    A SEG may omit its empty frames, so the planes it carries need not be
    adjacent.  The stack is the one whose step --- the stated
    ``SpacingBetweenSlices``, else the smallest gap --- divides every gap a
    whole number of times, the omitted planes empty.  The median gap stood
    for the step: planes at 10, 12.5 and 17.5 mm read as a stack 3.75 mm
    apart, the middle one 1.25 mm from where it was drawn.

    Every plane must lie within the bound a DICOM series is held to --- a
    tenth of a step, and at least 0.01 mm --- of where the stack puts it,
    measured per plane, not per gap: gaps each within a fixed bound drift.
    Where the smallest gap does not place every plane, the least-squares step
    from the first plane does, or nothing does; a stated step is kept or
    refused.  Each gap was held to 10⁻³ mm, which refused planes of a series
    whose positions were written to two decimals.  One plane has no step to
    measure: the stated one, or NaN.
    """
    from medh5.io.dicom import _position_tolerance

    if len(planes) < 2:
        return (abs(between) if between else float("nan")), [0] * len(planes)
    positions = np.asarray(planes, dtype=np.float64)
    gaps = np.diff(positions)
    along = positions - positions[0]
    step = abs(float(between)) if between else float(gaps.min())
    whole = np.round(gaps / step)
    if np.any(whole < 1):
        return None
    indices = np.concatenate([[0.0], np.cumsum(whole)])
    fits = [step] if between else [step, float(indices @ along / (indices @ indices))]
    for step in fits:
        if np.max(np.abs(along - indices * step)) <= _position_tolerance(step):
            return step, indices.astype(int).tolist()
    return None


_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "EchoTime": ("EffectiveEchoTime",),
    "ExposureTime": ("ExposureTimeInms",),
    "XRayTubeCurrent": ("XRayTubeCurrentInmA",),
}
"""The names an enhanced object's functional groups give an attribute a
classic slice states at its top level: an enhanced MR's echo time is the
``EffectiveEchoTime`` of its ``MREchoSequence``."""


def _group_value(group: Any, shared: Any, dataset: Any, keyword: str) -> Any:
    """*keyword* as a frame states it: in an item of its own functional groups,
    else of the shared ones, else at the object's top level; ``None`` where
    none does.

    A functional group is a sequence of one item holding the attributes it
    groups, so the attributes of a group are one item down
    (``PixelMeasuresSequence`` → ``SliceThickness``); a legacy-converted
    object keeps the classic attributes no group defines in its
    ``Unassigned…ConvertedAttributesSequence``, one item down too.
    """
    names = (keyword, *_GROUP_ALIASES.get(keyword, ()))
    for holder in (group, shared):
        if holder is None:
            continue
        for element in holder:
            if element.VR != "SQ":
                continue
            for item in element.value:
                for name in names:
                    value = getattr(item, name, None)
                    if value is not None and value != "":
                        return value
    value = getattr(dataset, keyword, None)
    return None if value == "" else value


def _first_item(holder: Any, name: str) -> Any:
    sequence = getattr(holder, name, None) if holder is not None else None
    return sequence[0] if sequence else None


def _source_series(dataset: Any) -> str | None:
    for item in getattr(dataset, "ReferencedSeriesSequence", []):
        uid = getattr(item, "SeriesInstanceUID", None)
        if uid:
            return str(uid)
    return None


__all__: list[str] = []
