"""The annotation header and the read contract every kind shares (spec §6).

An annotation is one coherent unit of ground truth.  The classes here are
views over the format engine's reader: every answer --- coverage, class
resolution, dense planes, counts, boxes --- is computed by the engine from the
file, so the command line, this package and the Rust SDK agree by
construction.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.geometry import Grid, box_to_slices
from medh5.labels import LabelClass, LabelSet

VOXEL_KINDS: tuple[str, ...] = _core.VOXEL_KINDS
GEOMETRIC_KINDS: tuple[str, ...] = _core.GEOMETRIC_KINDS
ANNOTATION_KINDS: tuple[str, ...] = _core.ANNOTATION_KINDS
RESERVED_KINDS: tuple[str, ...] = _core.RESERVED_KINDS
"""Kinds a 1.0 writer must not write (spec §16)."""

TASKS: tuple[str, ...] = _core.TASKS
DEFAULT_TASK_FOR_KIND: dict[str, str] = {
    kind: str(_core.default_task_for_kind(kind)) for kind in ANNOTATION_KINDS
}
SPEC_ANNOTATION_ATTRS: tuple[str, ...] = _core.SPEC_ANNOTATION_ATTRS

AnnotationHeader = _core.AnnotationHeader
"""The fixed attribute header every annotation carries (spec §6.2)."""

AnnotationPayload = _core.AnnotationPayload
"""Datasets and kind-specific attributes for one encoded annotation.

What every encoder returns and the writer stores; ``data`` is the ``data``
dataset, ``class_ids`` the encoding order (§6.2), ``stacked_axes`` the leading
axes of ``data`` that get chunk extent 1 (§14.1).
"""


def instance_id_dtype(ids: Iterable[int]) -> Any:
    """``uint32`` unless an id needs the wider form (spec §7.4, §8.2)."""
    return _core.instance_id_dtype(list(ids)).type


@dataclass(frozen=True, slots=True, eq=False)
class Instance:
    """One physical object: a box, a class, an id and optionally a cropped mask.

    ``instance_id`` is **sample-scoped**: the same lesion observed at several
    timepoints reuses its id, so lesion tracking is a join on this field rather
    than an additional structure.

    Two instances are equal when every field is: the box and mask by value.
    An instance holds arrays, so it is not hashable.
    """

    index: int
    instance_id: int
    class_id: int
    box: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_] | None = None
    score: float | None = None

    @property
    def slices(self) -> tuple[slice, ...]:
        return box_to_slices(self.box)

    @property
    def voxel_count(self) -> int:
        if self.mask is None:
            # Python's integers: a box from one end of the int64 range to
            # the other overflowed NumPy's product.
            return math.prod(s.stop - s.start for s in self.slices)
        return int(self.mask.sum())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instance):
            return NotImplemented
        if (self.index, self.instance_id, self.class_id, self.score) != (
            other.index,
            other.instance_id,
            other.class_id,
            other.score,
        ):
            return False
        if not np.array_equal(self.box, other.box):
            return False
        if self.mask is None or other.mask is None:
            return self.mask is None and other.mask is None
        return bool(np.array_equal(self.mask, other.mask))

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"Instance(id={self.instance_id}, class={self.class_id}, "
            f"box={self.box.tolist()})"
        )


def _instance(row: tuple[Any, ...]) -> Instance:
    index, instance_id, class_id, box, mask, score = row
    return Instance(
        index=int(index),
        instance_id=int(instance_id),
        class_id=int(class_id),
        box=np.asarray(box, dtype=np.float32),
        mask=mask,
        score=score,
    )


class Annotation:
    """What every annotation answers, whatever its kind."""

    __slots__ = ("_handle", "ann_id")

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.ann_id: str = handle.ann_id

    @property
    def header(self) -> Any:
        """The §6.2 header, as stored."""
        return self._handle.header

    @property
    def group(self) -> Any:
        """The stored group (a read-only :class:`~medh5.nodes.Group`), for
        inspecting what the reader does not model."""
        return self._handle.group

    @property
    def kind(self) -> str:
        return str(self._handle.kind)

    @property
    def task(self) -> str:
        return str(self._handle.task)

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(self._handle.class_ids)

    @property
    def annotated_class_ids(self) -> tuple[int, ...]:
        """What was *looked for* (§11.3): a class here and absent from the data
        is a usable negative; a class not here was never examined."""
        return tuple(self._handle.annotated_class_ids)

    @property
    def closure(self) -> str:
        return str(self._handle.closure)

    @property
    def ignore_id(self) -> int:
        return int(self._handle.ignore_id)

    @property
    def prov(self) -> str | None:
        found: str | None = self._handle.prov
        return found

    @property
    def quality_key(self) -> str | None:
        found: str | None = self._handle.quality_key
        return found

    @property
    def label_set(self) -> LabelSet | None:
        found: LabelSet | None = self._handle.label_set
        return found

    @property
    def grid_id(self) -> str | None:
        found: str | None = self._handle.grid_id
        return found

    @property
    def grid(self) -> Grid:
        found: Grid = self._handle.grid
        return found

    @property
    def timepoints(self) -> tuple[str, ...]:
        """The timepoints this annotation covers: declared, or its grid's."""
        return tuple(self._handle.timepoints)

    def is_annotated(self, class_key: int | str) -> bool:
        """Whether *class_key* was examined --- not whether it is present."""
        return bool(self._handle.is_annotated(class_key))

    @property
    def is_fully_covered(self) -> bool:
        return bool(self._handle.is_fully_covered)

    @property
    def has_ignore_region(self) -> bool:
        return bool(self._handle.has_ignore_region)

    def resolve_class(self, class_key: int | str) -> int:
        return int(self._handle.resolve_class(class_key))

    def resolve_classes(self, keys: Sequence[int | str] | None) -> tuple[int, ...]:
        return tuple(self._handle.resolve_classes(keys))

    @property
    def classes(self) -> tuple[LabelClass, ...]:
        return tuple(self._handle.classes)

    @property
    def annotated_classes(self) -> tuple[LabelClass, ...]:
        return tuple(self._handle.annotated_classes)

    def class_key(self, class_id: int) -> str:
        return str(self._handle.class_key(int(class_id)))

    def summary(self) -> dict[str, Any]:
        found: dict[str, Any] = self._handle.summary()
        return found

    def __repr__(self) -> str:
        return str(self._handle.__repr__())


class VoxelAnnotation(Annotation):
    """The uniform read contract of the five voxel encodings (spec §7)."""

    __slots__ = ()

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(self._handle.spatial_shape)

    def dense(
        self,
        classes: Sequence[int | str] | None = None,
        roi: Sequence[slice] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """``(C, *roi)`` boolean planes, one per class, whatever the encoding.

        *roi* slices take no step: a step other than 1 raises ``ValueError``.
        """
        found: npt.NDArray[np.bool_] = self._handle.dense(classes, roi)
        return found

    def contains(self, class_key: int | str, voxel: Sequence[int]) -> bool:
        return bool(self._handle.contains(class_key, [int(v) for v in voxel]))

    def labelmap(
        self,
        roi: Sequence[slice] | None = None,
        priority: Sequence[int | str] | None = None,
        *,
        dtype: npt.DTypeLike = np.uint16,
    ) -> npt.NDArray[Any]:
        """Flatten to one integer volume, breaking overlap ties **explicitly**.

        *priority* is ordered highest-precedence first; classes it omits are
        painted first, in ``class_ids`` order.  Flattening an overlapping
        annotation is lossy, and which class survives is the caller's decision,
        not the format's --- so when voxels are actually lost and the caller
        expressed no preference, this warns rather than picking silently.

        The §7.7 ignore region reads as ``ignore_id`` under every encoding,
        whether it is held in band or in the sibling mask ``ignore_mask``
        names: ``0`` means verified absent, and an ignored voxel is not.

        *dtype* must hold every class id, and ``ignore_id`` when a voxel is
        ignored; a narrower one raises ``ValueError`` rather than saturating
        two values into one, and ``bool`` is refused (``dense()`` returns
        boolean planes).  *roi* slices take no step: a step other than 1
        raises ``ValueError``.
        """
        volume, _, _, warning = self._handle.labelmap(roi, priority, dtype)
        if warning is not None:
            warnings.warn(warning, stacklevel=2)
        found: npt.NDArray[Any] = volume
        return found

    def voxel_counts(
        self, classes: Sequence[int | str] | None = None
    ) -> dict[int, int]:
        found: dict[int, int] = self._handle.voxel_counts(classes)
        return found

    def class_bboxes(
        self, classes: Sequence[int | str] | None = None
    ) -> dict[int, npt.NDArray[np.float32] | None]:
        """Each class's box at voxel edges, ``None`` for an absent class."""
        found: dict[int, npt.NDArray[np.float32] | None] = self._handle.class_bboxes(
            classes
        )
        return found

    def instances(self) -> Iterator[Instance]:
        """Iterate objects, where the encoding carries object identity."""
        for row in self._handle.instances():
            yield _instance(row)


__all__ = [
    "ANNOTATION_KINDS",
    "DEFAULT_TASK_FOR_KIND",
    "GEOMETRIC_KINDS",
    "RESERVED_KINDS",
    "SPEC_ANNOTATION_ATTRS",
    "TASKS",
    "VOXEL_KINDS",
    "Annotation",
    "AnnotationHeader",
    "AnnotationPayload",
    "Instance",
    "VoxelAnnotation",
    "instance_id_dtype",
]
