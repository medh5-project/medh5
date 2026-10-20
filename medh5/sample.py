"""``Sample`` and ``SampleWriter`` --- where the pieces become a file.

A sample is **one subject at one or more timepoints**.  Reading is lazy and
timepoint-aware; writing is a builder that validates as it goes and commits
atomically (spec §14.4).

A :class:`Sample` is a read-only view over the format engine's reader.  It is
memoised the way the file is immutable to it: ``amend`` is copy-on-write and
replaces the inode, so an open sample never sees an edit --- reopen to read
one.

The write model: ``create`` writes to a sibling temporary file and atomically
replaces the target on ``commit``, so a reader never sees a half-written
sample and a crash leaves the previous file intact.  ``amend`` is
copy-on-write: it builds a new file from the old one and replaces it, because
HDF5 does not reclaim space on delete.  On POSIX anything holding the old file
open keeps reading the old inode; Windows cannot replace an open file, so
close every reader of it before committing an amendment.  Every ``add_*``
validates as it goes, and ``commit`` refuses to write a file the validator
would reject: refused before it changed anything, it leaves the writer open
to mend and commit again.  The builder is the format engine's
``SampleWriter``::

    with medh5.create("case.medh5", sample_id="case-001") as w:
        w.add_grid("ct", shape=(64, 64, 32), spacing=(0.8, 0.8, 2.0))
        w.add_image("CT", volume, grid="ct", modality="CT")
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations import Annotation, open_annotation
from medh5.curation import Cohort, Identity, Timeline, Timepoint, Tracking
from medh5.document import SampleDocument
from medh5.errors import MEDH5ValidationError
from medh5.geometry import Grid
from medh5.image import Image
from medh5.labels import LabelSet
from medh5.storage import SamplingIndex
from medh5.transforms import Transform, wrap_transform

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.clinical import Clinical

FORMAT_VERSION: str = _core.FORMAT_VERSION
PROFILES: tuple[str, ...] = _core.PROFILES

ROOT_DIGEST_ATTRS: tuple[str, ...] = _core.ROOT_DIGEST_ATTRS
"""Root attributes covered by ``content_id``.

``created`` and ``generator`` are deliberately excluded: two byte-identical
samples written an hour apart must share a ``content_id``, or it is not a
content address and cannot be used as a cache or dedup key (spec §13.2).
"""

MANAGED_ROOT_ATTRS: tuple[str, ...] = _core.MANAGED_ROOT_ATTRS
"""Root attributes ``commit`` writes itself; an amend does not copy them."""

STANDARD_GROUPS: tuple[str, ...] = _core.STANDARD_GROUPS

SampleWriter = _core.SampleWriter
create = _core.create
amend = _core.amend


# --------------------------------------------------------------------------
# Collections
# --------------------------------------------------------------------------


def _left_out(lookup: Callable[[str], Any] | None, key: object) -> None:
    """Raise the engine's W913 when *key* names an object a later minor's
    file holds that this engine's projection leaves out (1.1 §2.2); return
    when the key is simply absent."""
    if lookup is not None and isinstance(key, str):
        with contextlib.suppress(KeyError):
            lookup(key)


class _Collection(Mapping[str, Any]):
    """A read-only mapping with a helpful ``KeyError``.

    In a later minor's file it holds what this engine's projection reads
    (1.1 §2.2): an object the projection leaves out is not in it, and looking
    one up by name raises the engine's W913 error, which says why.
    """

    __slots__ = ("_items", "_lookup", "_what")

    def __init__(
        self,
        what: str,
        items: Mapping[str, Any],
        lookup: Callable[[str], Any] | None = None,
    ) -> None:
        self._what = what
        self._items = dict(items)
        self._lookup = lookup

    def __getitem__(self, key: str) -> Any:
        try:
            return self._items[key]
        except KeyError:
            _left_out(self._lookup, key)
            raise KeyError(
                f"no {self._what} {key!r}; available: {sorted(self._items)}"
            ) from None

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def get(self, key: str, default: Any = None) -> Any:
        return self._items.get(key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"<{self._what}s: {sorted(self._items)}>"


class _Listed(dict[str, Any]):
    """A ``dict`` of what a later minor's file lists: what this engine's
    projection reads (1.1 §2.2).  Looking up an object the projection leaves
    out raises the engine's W913 error, which says why; any other missing key
    is the ``KeyError`` a ``dict`` raises.  A copy is a plain ``dict``.
    """

    __slots__ = ("_lookup",)

    def __init__(self, items: Mapping[str, Any], lookup: Callable[[str], Any]) -> None:
        super().__init__(items)
        self._lookup = lookup

    def __missing__(self, key: str) -> Any:
        _left_out(self._lookup, key)
        raise KeyError(key)

    def __reduce__(self) -> tuple[Any, ...]:
        return (dict, (dict(self),))


class ImageCollection(_Collection):
    """The sample's images, with timepoint- and modality-aware views."""

    __slots__ = ()

    def by_timepoint(self, timepoint_id: str) -> tuple[str, ...]:
        return tuple(
            name
            for name, image in self._items.items()
            if image.timepoint == timepoint_id
        )

    def by_modality(self, modality: str) -> tuple[str, ...]:
        return tuple(
            name for name, image in self._items.items() if image.modality == modality
        )

    def on_grid(self, grid_id: str) -> tuple[str, ...]:
        return tuple(
            name for name, image in self._items.items() if image.grid_id == grid_id
        )


class AnnotationCollection(_Collection):
    """The sample's annotations, with task- and timepoint-aware views."""

    __slots__ = ()

    def by_task(self, task: str) -> tuple[Annotation, ...]:
        return tuple(a for a in self._items.values() if a.task == task)

    def by_kind(self, kind: str) -> tuple[Annotation, ...]:
        return tuple(a for a in self._items.values() if a.kind == kind)

    def by_timepoint(self, timepoint_id: str) -> tuple[Annotation, ...]:
        return tuple(a for a in self._items.values() if timepoint_id in a.timepoints)

    def spanning(self) -> tuple[Annotation, ...]:
        """Annotations covering more than one timepoint --- change labels."""
        return tuple(a for a in self._items.values() if len(a.timepoints) > 1)


class TimepointView:
    """Everything in the sample that belongs to one timepoint."""

    __slots__ = ("_sample", "timepoint")

    def __init__(self, sample: Sample, timepoint: Timepoint) -> None:
        self._sample = sample
        self.timepoint = timepoint

    @property
    def id(self) -> str:
        return str(self.timepoint.id)

    @property
    def grids(self) -> dict[str, Grid]:
        return {
            gid: g
            for gid, g in self._sample.grids.items()
            if g.timepoint == self.timepoint.id
        }

    @property
    def images(self) -> ImageCollection:
        names = self._sample.images.by_timepoint(self.timepoint.id)
        return ImageCollection("image", {n: self._sample.images[n] for n in names})

    @property
    def annotations(self) -> AnnotationCollection:
        found = self._sample.annotations.by_timepoint(self.timepoint.id)
        return AnnotationCollection("annotation", {a.ann_id: a for a in found})

    def __repr__(self) -> str:
        return (
            f"TimepointView({self.timepoint.id!r}, "
            f"{len(self.images)} images, {len(self.annotations)} annotations)"
        )


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


class Sample:
    """A read-only view of one sample root (a file, or a collection member).

    A file of a later minor version is read as this engine's projection
    (``support == "projection"``, 1.1 §2.2): an object holding a value only
    that minor defines --- a grid with an axis kind, an annotation or
    transform of a kind this engine does not know --- is left out of
    :attr:`grids`, :attr:`images`, :attr:`annotations`, :attr:`transforms` and
    :attr:`index`, with what depends on it, and looking it up by name raises
    :class:`~medh5.errors.MEDH5ValidationError` coded W913, which says why.
    """

    __slots__ = (
        "_annotations",
        "_clinical",
        "_document",
        "_grids",
        "_handle",
        "_images",
        "_index",
        "_resolved",
        "_transforms",
        "path",
    )

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.path: str | None = handle.path
        # Built on first use and kept.  Two threads reading one at once may
        # both build it: each gets a whole value and one is kept --- duplicate
        # work, never a torn read --- so they take no lock.
        self._document: SampleDocument | None = None
        self._grids: dict[str, Grid] | None = None
        self._images: ImageCollection | None = None
        self._annotations: AnnotationCollection | None = None
        self._transforms: _Collection | None = None
        self._index: dict[str, SamplingIndex] | None = None
        self._resolved: dict[tuple[str, str], Transform | None] = {}
        self._clinical: tuple[Clinical | None] | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> Sample:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the file, and everything read from it.  Safe to call twice.

        A collection member owns no file: closing one releases the member
        alone, and the shard stays open until its collection is closed."""
        self._handle.close()

    @property
    def is_open(self) -> bool:
        """Whether the sample can still be read: ``False`` once it is closed,
        and, for a collection member, once its collection is closed."""
        return bool(self._handle.is_open)

    def __repr__(self) -> str:
        return str(self._handle.repr())

    @property
    def root(self) -> Any:
        """The sample's root group, for inspection (attributes, members)."""
        return self._handle.root

    # -- document ----------------------------------------------------------

    @property
    def document(self) -> SampleDocument:
        """The sample document, ``/meta`` (§2.4).

        A later minor's --- a 1.2 file's --- is read as its supported
        projection (1.1 §2.2): members only that minor defines are set aside,
        and an unknown value is read as its vocabulary's ``other``, or as
        absent where it has none (in a member the schema requires, that leaves
        the document unreadable: ``W913``, the code the validator reports the
        value under).
        """
        if self._document is None:
            self._document = self._handle.document()
        return self._document

    @property
    def identity(self) -> Identity:
        found: Identity = self.document.identity
        return found

    @property
    def cohort(self) -> Cohort:
        found: Cohort = self.document.cohort
        return found

    @property
    def timepoints(self) -> Timeline:
        found: Timeline = self.document.timepoints
        return found

    @property
    def label_set(self) -> LabelSet | None:
        found: LabelSet | None = self.document.label_set
        return found

    @property
    def version(self) -> str:
        return str(self._handle.version)

    @property
    def kind(self) -> str:
        return str(self._handle.kind)

    @property
    def profiles(self) -> frozenset[str]:
        return frozenset(self._handle.profiles)

    @property
    def content_id(self) -> str | None:
        found: str | None = self._handle.content_id
        return found

    @property
    def support(self) -> str:
        """``"full"``, or ``"projection"`` for a higher minor version: read as
        what this engine knows, validated with W913, and never amended
        (1.1 §2.2)."""
        return str(self._handle.support)

    @property
    def clinical(self) -> Clinical | None:
        """The ``clinical`` profile's records (format 1.1), or ``None`` when
        the sample does not declare the profile."""
        if self._clinical is None:
            from medh5.clinical import Clinical

            handle = self._handle.clinical()
            self._clinical = (None if handle is None else Clinical(handle),)
        return self._clinical[0]

    def document_text(self, document_id: str) -> str:
        """One clinical document's text, read on its own: the document
        table's offsets and that document's bytes --- not the events, the links
        or any other document (format 1.1 §6).  ``KeyError`` for no such
        document, or a sample without the profile."""
        return str(self._handle.document_text(document_id))

    # -- objects -----------------------------------------------------------

    def _later(self, lookup: Callable[[str], Any]) -> Callable[[str], Any] | None:
        """The engine's lookup, for a mapping of a later minor's file: a key
        its projection leaves out raises the W913 that says why.  A sample
        declaring no version --- a collection member that lost it, read as
        the oldest (1.1 §8) --- is no later minor's."""
        try:
            support = self._handle.support
        except MEDH5ValidationError:
            return None
        return lookup if support == "projection" else None

    def _listed(
        self, items: Mapping[str, Any], lookup: Callable[[str], Any]
    ) -> dict[str, Any]:
        """*items* as a ``dict``; in a later minor's file, one whose lookup of
        a key its projection leaves out raises the W913 that says why."""
        later = self._later(lookup)
        return dict(items) if later is None else _Listed(items, later)

    @property
    def grids(self) -> dict[str, Grid]:
        """The sample's grids, with §3.7's implicit timepoint resolved.

        A grid MUST name its timepoint only when the sample declares more than
        one (§3.7 rule 2); with exactly one declared, the grid belongs to that
        one, and every timepoint-aware reader takes its answer from here.
        """
        if self._grids is None:
            self._grids = self._listed(self._handle.grids(), self._handle.grid)
        return self._grids

    @property
    def reference_grid(self) -> Grid:
        """``grids/ref`` when present, else the grid of the first image (§3.2)."""
        found: Grid = self._handle.reference_grid()
        return found

    @property
    def images(self) -> ImageCollection:
        if self._images is None:
            self._images = ImageCollection(
                "image",
                {
                    name: Image(self._handle.image(name))
                    for name in sorted(self._handle.image_ids())
                },
                self._later(self._handle.image),
            )
        return self._images

    @property
    def annotations(self) -> AnnotationCollection:
        if self._annotations is None:
            self._annotations = AnnotationCollection(
                "annotation",
                {
                    name: open_annotation(self._handle.annotation(name))
                    for name in sorted(self._handle.annotation_ids())
                },
                self._later(self._handle.annotation),
            )
        return self._annotations

    def ignore_region(
        self, ann_id: str, roi: Sequence[slice] | None = None
    ) -> npt.NDArray[np.bool_]:
        """The §7.7 ignore region of a voxel annotation, under any encoding.

        ``labelmap`` and ``layers`` may keep it in band; every encoding may
        instead name a sibling `mask` annotation in ``header.ignore_mask``.
        This reads both.  All ``False`` where the annotation declares no region.
        """
        found: npt.NDArray[np.bool_] = self._handle.ignore_region(ann_id, roi)
        return found

    def valid_region(
        self, image_id: str, roi: Sequence[slice] | None = None
    ) -> npt.NDArray[np.bool_]:
        """Where an image holds data (§4.4): its ``valid_mask``, or everywhere.

        *roi* covers the spatial axes and is resolved as an annotation's
        window is --- a negative bound counts from the end, and every bound is
        clipped to its axis --- and a slice with a step other than 1 is
        refused (``ValueError``)."""
        found: npt.NDArray[np.bool_] = self._handle.valid_region(image_id, roi)
        return found

    @property
    def index(self) -> dict[str, SamplingIndex]:
        if self._index is None:
            self._index = self._listed(
                {name: self._handle.index(name) for name in self._handle.index_ids()},
                self._handle.index,
            )
        return self._index

    @property
    def fresh_indices(self) -> frozenset[str]:
        """Index entries whose ``source_digest`` still matches their source
        (§13.3).  A stale entry is ignored, not trusted."""
        return frozenset(self._handle.fresh_indices())

    @property
    def transforms(self) -> _Collection:
        if self._transforms is None:
            self._transforms = _Collection(
                "transform",
                {
                    name: wrap_transform(self._handle.transform(name))
                    for name in self._handle.transform_ids()
                },
                self._later(self._handle.transform),
            )
        return self._transforms

    def transform_between(
        self,
        source: str,
        target: str,
        *,
        by: Literal["timepoint", "grid", "frame"] | None = None,
    ) -> Transform | None:
        """The transform relating two timepoints, grids or frames (spec §10).

        *by* says how both keys are read; by default a key is read as a
        timepoint first, then as a grid, then as a frame uid, and one that is
        none of these raises ``KeyError``.

        A visit has an answer only when its framed grids share one frame of
        reference.  When either key names a visit holding several (a CT and a
        PET on frames of their own), no one transform relates the two, and
        this raises :class:`~medh5.errors.MEDH5ValidationError` naming each
        visit's grids and frames: ask about two grids (``by="grid"``) or two
        frames (:meth:`resolve_frames`) instead.

        Returns ``None`` when the two already share a frame, when a side names
        no frame at all, or when no transform relates them: geometry is never
        invented, so ``None`` alone does not say the two are aligned.
        """
        found = self._handle.transform_between(source, target, by)
        return None if found is None else wrap_transform(found)

    def _frames_for(self, key: str) -> tuple[str, ...]:
        """The frame uids a key names, read as :meth:`transform_between` reads
        it: a timepoint (every grid of the visit), then a grid, then a frame."""
        return tuple(self._handle.frames_for(key))

    def resolve_frames(self, from_frame: str, to_frame: str) -> Transform | None:
        """The transform relating two frame uids, resolved once per handle.

        A paired dataset asks the same question for every item; the answer
        (and the object carrying it) is the same each time.
        """
        key = (from_frame, to_frame)
        if key not in self._resolved:
            found = self._handle.resolve_frames(from_frame, to_frame)
            self._resolved[key] = None if found is None else wrap_transform(found)
        return self._resolved[key]

    # -- timepoints --------------------------------------------------------

    def at(self, timepoint: str | int) -> TimepointView:
        """A timepoint-scoped view of the whole sample."""
        return TimepointView(self, self.timepoints[timepoint])

    @property
    def is_longitudinal(self) -> bool:
        return bool(self.timepoints.is_longitudinal)

    def tracks(
        self, class_key: int | str | None = None, *, measure: bool = True
    ) -> Tracking:
        """Join ``instance_id`` across timepoints --- the tracking operation (§7.4).

        Whether an absence means *resolved* or *unexamined* depends on whether
        the object was looked for there: ``annotated_class_ids`` (§11.3), the
        ignore region (§7.7) and where the object lies, mapped from the visits
        that saw it.  ``Tracking.reason_at`` says which answered.
        """
        found: Tracking = self._handle.tracks(class_key, measure=measure)
        return found

    # -- integrity ---------------------------------------------------------

    def attr_name_map(self) -> dict[str, tuple[str, ...]]:
        """Object path -> the attribute names ``content_id`` covers (§13.2)."""
        found: dict[str, tuple[str, ...]] = self._handle.attr_name_map()
        return found

    def verify(self, partial: Sequence[str] | None = None) -> Any:
        from medh5.integrity import VerifyResult

        return VerifyResult(
            **self._handle.verify(None if partial is None else list(partial))
        )

    def compute_content_id(self) -> str:
        return str(self._handle.compute_content_id())

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        found: dict[str, Any] = self._handle.summary()
        return found


FRAME_ATTRS: tuple[tuple[str, tuple[str, ...]], ...] = _core.FRAME_ATTRS
"""Where frame-of-reference uids are named: ``(group, attribute names)``."""


def annotation_id(reference: str) -> str:
    """An annotation id from a reference that may be written as a path.

    §6.2 says ``derived_from`` holds annotation ids; the RTSTRUCT importer
    wrote ``annotations/<id>`` for a release, so both spellings name one thing.
    """
    return str(_core.annotation_id(reference))


def open_sample(path: str | os.PathLike[str], mode: str = "r") -> Sample:
    """Open a ``.medh5`` sample file, read-only.

    ``mode`` exists for the callers that spelled out ``"r"``; edits go through
    :func:`amend`, which is copy-on-write.
    """
    if mode != "r":
        raise MEDH5ValidationError(
            f"open() is read-only; use create() or amend() to write, not {mode!r}"
        )
    return Sample(_core.open_sample(os.fspath(path)))


__all__ = [
    "FORMAT_VERSION",
    "FRAME_ATTRS",
    "MANAGED_ROOT_ATTRS",
    "PROFILES",
    "ROOT_DIGEST_ATTRS",
    "STANDARD_GROUPS",
    "AnnotationCollection",
    "ImageCollection",
    "Sample",
    "SampleWriter",
    "TimepointView",
    "amend",
    "annotation_id",
    "create",
    "open_sample",
]
