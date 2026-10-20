"""PyTorch datasets over MEDH5 samples.

Three datasets, one contract: every item of :class:`VolumeDataset`,
:class:`PatchDataset` and :class:`GridPatchDataset` is a ``dict`` with
``images``, ``valid``, ``label``, ``ignore`` and ``meta``.  Keeping the shape
uniform is what lets a training script swap whole-volume for patch-based
sampling by changing one constructor.  :class:`PairedPatchDataset` reads two
visits: its ``images`` and ``valid`` are keyed by timepoint, and its only
label is a change label (see its docstring).

The file's contracts travel with the arrays, because a loss can only honour
what it is handed:

* ``ignore[ann_id]`` --- ``True`` where §7.7 says a voxel must not be scored:
  the annotation's ignore region under whichever encoding stores it, and any
  padding added to reach the patch size.  ``label_format="labelmap"`` also
  writes ``65535`` there.
* ``valid[image_id]`` --- ``True`` where the image holds data (§4.4): its
  ``valid_mask`` where it declares one, and never the padding.
* ``meta["annotated"][ann_id]`` --- one ``bool`` per label channel: whether the
  class was *examined* (§11.3).  A ``0`` in an unexamined class is not a
  negative.

Two things here are load-bearing:

* **Reads are single-call.**  Every array comes out of one ``dataset[roi]`` and
  never ``dataset[...][roi]``, because the latter materialises the whole volume
  first --- 40× slower for a 64³ ROI out of a 160³ bitplane (§14.5).  The API
  offers no way to write the slow form.
* **Files stay open per worker, keyed by PID.**  See :mod:`medh5.torch.handles`.

Labels come back in one of three formats.  ``onehot`` is ``(C, *patch)`` float
planes in the caller's class order; ``labelmap`` is a single ``int64`` plane of
class ids, ties broken by that same order and every class not asked for
background; ``instances`` returns the objects of the classes asked for that
overlap the patch.  The class order is always the one the caller asked for,
never the file's storage order, so channel *i* means the same structure across
a cohort whose files were written by different tools.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.base import VoxelAnnotation
from medh5.annotations.voxel import InstancesAnnotation
from medh5.errors import MEDH5ValidationError
from medh5.labels import IGNORE_ID
from medh5.sample import Sample, open_sample
from medh5.sampling import (
    PairReport,
    Patch,
    PatchSampler,
    Seed,
    TimepointPair,
    TimepointPairSampler,
    grid_patches,
)
from medh5.torch._compat import dataset_base, require_torch, to_tensor
from medh5.torch.handles import CACHE

LABEL_FORMATS = ("onehot", "labelmap", "instances", "none")
ALIGNMENTS = ("none", "transform")

PathLike = str | os.PathLike[str]


def _check_label_format(name: str) -> str:
    if name not in LABEL_FORMATS:
        raise MEDH5ValidationError(
            f"unknown label_format {name!r}; expected one of {list(LABEL_FORMATS)}"
        )
    return name


_DatasetBase = dataset_base()


class _Epoch:
    """The epoch, in a tensor every ``DataLoader`` worker shares.

    Patches are seeded with ``(seed, epoch, index)``.  An ``int`` attribute set
    in the main process never reached a worker started with
    ``persistent_workers=True`` --- the worker kept the copy of the dataset it
    was started with --- so every epoch drew epoch 0's patches, and random
    crops, the main source of variety in patch-based 3-D training, became one
    fixed draw.  A shared-memory tensor is the same storage in every worker,
    under ``fork`` and under ``spawn``, so ``set_epoch`` is seen at the next
    ``__getitem__`` wherever that runs.
    """

    __slots__ = ("_value",)

    def __init__(self) -> None:
        import torch

        self._value = torch.zeros((), dtype=torch.int64).share_memory_()

    def get(self) -> int:
        return int(self._value.item())

    def set(self, epoch: int) -> None:
        self._value.fill_(int(epoch))


class _Base(_DatasetBase):  # type: ignore[misc,valid-type]
    """Shared reading logic, over ``torch.utils.data.Dataset`` when it exists."""

    def __init__(
        self,
        paths: Sequence[PathLike],
        *,
        images: Sequence[str] | None = None,
        annotations: Mapping[str, Sequence[int | str]] | None = None,
        label_format: str = "onehot",
        physical: bool = True,
        dtype: npt.DTypeLike = np.float32,
        timepoint: str | None = None,
    ) -> None:
        require_torch()
        self.paths = [str(p) for p in paths]
        if not self.paths:
            raise MEDH5ValidationError("a dataset needs at least one file")
        self.images = None if images is None else tuple(images)
        self.annotations = (
            {} if annotations is None else {k: tuple(v) for k, v in annotations.items()}
        )
        self.label_format = _check_label_format(label_format)
        self.physical = bool(physical)
        self.dtype = np.dtype(dtype)
        self.timepoint = timepoint

    # -- reading -----------------------------------------------------------

    def _lease(self, path: str) -> AbstractContextManager[Sample]:
        """The cached handle, held against eviction while an item is read."""
        return CACHE.lease(path)

    @staticmethod
    @contextmanager
    def _scan(path: str) -> Iterator[Sample]:
        """A short-lived open, for reading metadata while building a plan.

        Deliberately not ``open_cached``.  ``CACHE`` is process-global and
        shared by every dataset in the worker, so walking a 10 000-file cohort
        through its 32-entry LRU evicts whatever training was about to reuse and
        leaves it holding the tail of the cohort instead --- entries chosen by
        the order the plan was built in rather than by what is being read.  The
        pass itself is unavoidable: ``__len__`` is not knowable without one
        metadata read per file.
        """
        sample = open_sample(path)
        try:
            yield sample
        finally:
            sample.close()

    def _image_ids(self, sample: Sample) -> tuple[str, ...]:
        if self.images is not None:
            missing = [i for i in self.images if i not in sample.images]
            if missing:
                raise MEDH5ValidationError(
                    f"{sample.path}: no image(s) {missing}; present: "
                    f"{sorted(sample.images)}"
                )
            if self.timepoint is not None:
                # `timepoint=` was ignored beside `images=`: another visit's
                # images were read without a word.
                at = sample.at(self.timepoint).images
                elsewhere = [i for i in self.images if i not in at]
                if elsewhere:
                    raise MEDH5ValidationError(
                        f"{sample.path}: image(s) {elsewhere} are not at "
                        f"timepoint {self.timepoint!r}, whose images are "
                        f"{sorted(at)}"
                    )
            return self.images
        if self.timepoint is not None:
            return tuple(sorted(sample.at(self.timepoint).images))
        return tuple(sorted(sample.images))

    def _images_grid(self, sample: Sample) -> str | None:
        """The one grid the images this dataset reads share --- what a window
        is drawn, and a cover planned, in --- or ``None`` when it reads none.

        Not the reference grid: a dataset scoped to a later visit, by
        ``timepoint=`` or ``images=``, drew every window on the first visit's
        grid and refused every item it read.
        """
        names = self._image_ids(sample)
        if not names:
            return None
        return self._single_grid(
            sample, {f"image {n!r}": sample.images[n].grid_id for n in names}
        )

    @staticmethod
    def _single_grid(sample: Sample, members: Mapping[str, str]) -> str:
        """The one grid *members* share, or a refusal to guess across grids.

        A patch is a window in a single grid's index space.  Reading a second
        grid with those same slices assumes both index the same anatomy at the
        same spacing, which nothing in the format guarantees, and a smaller grid
        silently returns a truncated array because the padding was computed for
        the first.  Converters refuse rather than resample (§3); so does the
        loader --- narrowing the selection or resampling is the caller's call
        to make deliberately.
        """
        grids = sorted(set(members.values()))
        if len(grids) > 1:
            listed = ", ".join(f"{k} on {v!r}" for k, v in sorted(members.items()))
            raise MEDH5ValidationError(
                f"{sample.path}: a patch is a window in one grid, but this "
                f"selection spans {len(grids)} of them ({listed}); restrict "
                "`images=`/`annotations=` to one grid, scope the dataset with "
                "`timepoint=`, or resample the volumes onto a common grid first "
                "(`medh5 resample PATH --image ID --onto GRID`, which carries "
                "the annotations along)"
            )
        return grids[0]

    def _check_patch_grid(self, sample: Sample, patch: Patch) -> None:
        """Everything a patch window is read out of must sit on the window's grid.

        The window itself is a member of the comparison, not just the objects
        being read.  Checking only the objects passes whenever they happen to
        agree with each other --- a single image on grid B read with a window
        drawn on annotation grid A looks like one grid and is silently
        misregistered, and truncated wherever B is the smaller of the two.
        """
        members = {
            f"image {name!r}": sample.images[name].grid_id
            for name in self._image_ids(sample)
        }
        if patch.grid_id is not None:
            members["the patch window"] = patch.grid_id
        if self.label_format != "none":
            for name in self.annotations:
                if name not in sample.annotations:
                    continue
                grid_id = sample.annotations[name].grid_id
                if grid_id is not None:
                    members[f"annotation {name!r}"] = grid_id
        if members:
            self._single_grid(sample, members)

    def _read_images(
        self, sample: Sample, patch: Patch | None
    ) -> dict[str, npt.NDArray[Any]]:
        roi = None if patch is None else patch.slices
        out: dict[str, npt.NDArray[Any]] = {}
        for image_id in self._image_ids(sample):
            image = sample.images[image_id]
            array = image.read(roi, physical=self.physical, dtype=self.dtype)
            out[image_id] = array if patch is None else patch.apply_padding(array)
        return out

    def _read_valid(
        self, sample: Sample, image_ids: Sequence[str], patch: Patch | None
    ) -> dict[str, npt.NDArray[np.bool_]]:
        """Each image's valid region over the window; padding is never valid."""
        roi = None if patch is None else patch.slices
        out: dict[str, npt.NDArray[np.bool_]] = {}
        for image_id in image_ids:
            region = sample.valid_region(image_id, roi)
            out[image_id] = (
                region if patch is None else patch.apply_padding(region, value=False)
            )
        return out

    def _read_labels(
        self, sample: Sample, patch: Patch | None
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """``(labels, ignore, annotated)`` for every requested annotation.

        The ignore region is read through :meth:`Sample.ignore_region`, which
        knows both places §7.7 lets an encoding keep it --- in band, and in a
        sibling mask --- and goes beside the labels as ``ignore``: ``dense()``
        planes have no value for it.  A ``labelmap`` item carries it as well,
        as ``IGNORE_ID`` (65535): ``labelmap()`` writes the annotation's own
        ``ignore_id`` there, which a file may declare as another value, and
        one ``ignore_index=65535`` then serves every file.
        """
        if self.label_format == "none" or not self.annotations:
            return {}, {}, {}
        roi = None if patch is None else list(patch.slices)
        labels: dict[str, Any] = {}
        ignore: dict[str, Any] = {}
        annotated: dict[str, Any] = {}
        for ann_id, classes in self.annotations.items():
            if ann_id not in sample.annotations:
                raise MEDH5ValidationError(
                    f"{sample.path}: no annotation {ann_id!r}; present: "
                    f"{sorted(sample.annotations)}"
                )
            ann = sample.annotations[ann_id]
            if not isinstance(ann, VoxelAnnotation):
                # Boxes, points and the like have no voxels to read, in any
                # format: each one failed on a missing method, by its name.
                raise MEDH5ValidationError(
                    f"{sample.path}: annotation {ann_id!r} is {ann.kind!r}, and "
                    f"label_format={self.label_format!r} reads voxel annotations "
                    "(segmentations) only; leave it out of `annotations=`"
                )
            wanted = list(classes) if classes else None
            examined = set(ann.annotated_class_ids)
            requested = ann.resolve_classes(wanted)
            annotated[ann_id] = np.asarray(
                [c in examined for c in requested], dtype=bool
            )
            region = sample.ignore_region(ann_id, roi)
            ignore[ann_id] = (
                region if patch is None else patch.apply_padding(region, value=True)
            )
            if self.label_format == "instances":
                chosen = None if wanted is None else requested
                labels[ann_id] = self._instances_in(ann, patch, chosen)
                continue
            if self.label_format == "labelmap":
                # `int64`, the target dtype a classification loss takes: the
                # `uint16` plane was refused by `CrossEntropyLoss`.
                array = np.asarray(
                    ann.labelmap(roi=roi, priority=wanted), dtype=np.int64
                )
                if wanted is not None:
                    # `priority` orders the requested classes on top, and the
                    # rest are painted beneath them: what was not asked for is
                    # background, as in every other label format.
                    array[~np.isin(array, [*requested, IGNORE_ID])] = 0
                # As 65535, whatever `ignore_id` the file declares.
                array[region] = IGNORE_ID
                pad: float = IGNORE_ID
            else:
                array = np.asarray(ann.dense(wanted, roi=roi), dtype=self.dtype)
                pad = 0.0
            labels[ann_id] = (
                array if patch is None else patch.apply_padding(array, value=pad)
            )
        return labels, ignore, annotated

    @staticmethod
    def _instances_in(
        ann: Any, patch: Patch | None, classes: Sequence[int] | None = None
    ) -> list[dict[str, Any]]:
        """Objects of *classes* (all, when ``None``) overlapping the patch, with
        boxes in patch coordinates.

        A box ``[a, b]`` sits at voxel edges --- it is the slice
        ``a+0.5 : b+0.5`` --- so it overlaps the voxels the patch read,
        ``start .. stop - 1``, when ``b > start - 0.5`` and ``a < stop - 0.5``.
        Patch coordinates count from the *padded* patch's first voxel, so a
        patch reaching past the volume shifts every box by the padding before
        it: subtracting the start alone left boxes ``pad_before`` voxels from
        the anatomy they bound.  Only the requested classes are returned, as
        every other label format does.
        """
        wanted = None if classes is None else set(classes)
        objects = []
        for instance_id, class_id, stored, score in _object_rows(ann):
            if wanted is not None and class_id not in wanted:
                continue
            box = np.asarray(stored, dtype=np.float64)
            if patch is not None:
                start = np.asarray([s.start for s in patch.slices], dtype=np.float64)
                stop = np.asarray([s.stop for s in patch.slices], dtype=np.float64)
                before = np.asarray([b for b, _ in patch.padding], dtype=np.float64)
                if np.any(box[:, 1] <= start - 0.5) or np.any(box[:, 0] >= stop - 0.5):
                    continue
                box = box - start[:, None] + before[:, None]
            objects.append(
                {
                    "instance_id": instance_id,
                    "class_id": class_id,
                    "box": box.astype(np.float32),
                    "score": score,
                }
            )
        return objects

    def _meta(self, sample: Sample, patch: Patch | None) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "path": sample.path,
            "sample_id": sample.identity.sample_id,
            "subject_id": sample.identity.subject_id,
        }
        if patch is not None:
            meta["patch"] = patch.to_json()
        return meta

    def _item(
        self,
        sample: Sample,
        patch: Patch | None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if patch is not None:
            self._check_patch_grid(sample, patch)
        images = self._read_images(sample, patch)
        valid = self._read_valid(sample, list(images), patch)
        item: dict[str, Any] = {
            "images": {k: to_tensor(v) for k, v in images.items()},
            "valid": {k: to_tensor(v) for k, v in valid.items()},
            "meta": {**self._meta(sample, patch), **(extra or {})},
        }
        labels, ignore, annotated = self._read_labels(sample, patch)
        if labels:
            item["label"] = {
                k: (v if isinstance(v, list) else to_tensor(v))
                for k, v in labels.items()
            }
        if ignore:
            item["ignore"] = {k: to_tensor(v) for k, v in ignore.items()}
        if annotated:
            item["meta"]["annotated"] = {k: to_tensor(v) for k, v in annotated.items()}
        return item


class VolumeDataset(_Base):
    """One item per file: the whole volume, eagerly read.

    For validation and inference on volumes that fit in memory.  Training on
    3-D volumes almost always wants :class:`PatchDataset` instead.
    """

    def __len__(self) -> int:
        return len(self.paths)

    def file_groups(self) -> list[range]:
        """Item indices per file, for :class:`~medh5.torch.FileGroupedSampler`."""
        return [range(i, i + 1) for i in range(len(self.paths))]

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self._lease(self.paths[index]) as sample:
            return self._item(sample, None)


class PatchDataset(_Base):
    """``samples_per_volume`` patches from each file, drawn by a sampler.

    Length is ``len(paths) * samples_per_volume`` so that an epoch is a
    well-defined number of steps; *which* patches those are is redrawn every
    epoch, seeded per item so a run is reproducible and workers do not
    duplicate each other's draws.
    """

    def __init__(
        self,
        paths: Sequence[PathLike],
        sampler: PatchSampler,
        *,
        samples_per_volume: int = 1,
        annotation: str | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(paths, **kwargs)
        self.sampler = sampler
        self.samples_per_volume = int(samples_per_volume)
        if self.samples_per_volume < 1:
            raise MEDH5ValidationError("samples_per_volume must be at least 1")
        self.annotation = annotation or (
            next(iter(self.annotations)) if self.annotations else None
        )
        self.seed = int(seed)
        self._epoch = _Epoch()

    @property
    def epoch(self) -> int:
        return self._epoch.get()

    @epoch.setter
    def epoch(self, epoch: int) -> None:
        self._epoch.set(epoch)

    def set_epoch(self, epoch: int) -> None:
        """Redraw patches from the next item on --- call it once per epoch.

        Seen by every ``DataLoader`` worker, persistent ones included; see
        :class:`_Epoch`.  A loop that never calls it draws the same patches
        every epoch, by design: the draw is a function of ``(seed, epoch,
        index)``, which is what makes a run reproducible.
        """
        self._epoch.set(epoch)

    def __len__(self) -> int:
        return len(self.paths) * self.samples_per_volume

    def file_groups(self) -> list[range]:
        """Item indices per file, for :class:`~medh5.torch.FileGroupedSampler`."""
        n = self.samples_per_volume
        return [range(f * n, (f + 1) * n) for f in range(len(self.paths))]

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index // self.samples_per_volume]
        with self._lease(path) as sample:
            patch = self.sampler.draw(
                sample,
                self.annotation,
                (self.seed, self.epoch, index),
                grid=self._images_grid(sample),
            )
            return self._item(sample, patch)


class GridPatchDataset(_Base):
    """Deterministic sliding-window cover of every file --- the inference path.

    The cover is planned on the grid the images read share (the reference
    grid when none is read), so images on more than one grid are refused
    here rather than at the first item.

    Each item's ``meta["cover"]`` is ``{"file": i, "index": k, "of": n}``: the
    *k*-th of the *n* patches covering ``paths[i]``, which is what
    :class:`~medh5.torch.GridStitcher` counts a file's completion by.
    """

    def __init__(
        self,
        paths: Sequence[PathLike],
        patch_size: int | Sequence[int],
        *,
        overlap: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(paths, **kwargs)
        self.patch_size = patch_size
        self.overlap = int(overlap)
        self._plan: list[tuple[str, Patch]] = []
        self._cover: list[tuple[int, int, int]] = []
        # Per file: the spatial shape and grid its cover is planned on, and
        # where its patches start in the plan --- what a stitcher reads.
        self._volumes: list[tuple[tuple[int, ...], str]] = []
        self._starts: list[int] = []
        for i, path in enumerate(self.paths):
            with self._scan(path) as sample:
                planned = sample.grids[
                    self._images_grid(sample) or sample.reference_grid.grid_id
                ]
                shape, grid_id = planned.spatial_shape, planned.grid_id
            patches = grid_patches(
                shape, patch_size, overlap=self.overlap, grid_id=grid_id
            )
            self._volumes.append((tuple(int(n) for n in shape), str(grid_id)))
            self._starts.append(len(self._plan))
            self._plan.extend((path, patch) for patch in patches)
            self._cover.extend((i, k, len(patches)) for k in range(len(patches)))

    def __len__(self) -> int:
        return len(self._plan)

    def file_groups(self) -> list[list[int]]:
        """Item indices per file, for :class:`~medh5.torch.FileGroupedSampler`."""
        groups: dict[str, list[int]] = {}
        for index, (path, _) in enumerate(self._plan):
            groups.setdefault(path, []).append(index)
        return list(groups.values())

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, patch = self._plan[index]
        file, k, n = self._cover[index]
        with self._lease(path) as sample:
            cover = {"file": file, "index": k, "of": n}
            return self._item(sample, patch, {"cover": cover})


class PairedPatchDataset(_Base):
    """Corresponding patches from two visits of one subject (§3.7, §10).

    ``align="transform"`` maps the patch centre through the transform relating
    the two frames, so the two patches cover the same anatomy.  ``align="none"``
    reads the same index window from both, which is what a model that learns
    its own alignment wants.

    A cross-sectional file contributes no pairs.  The count is reported through
    :attr:`report` rather than absorbed: a dataset that silently drops most of
    its files looks exactly like one that is training normally.

    An item's ``images`` and ``valid`` are keyed by timepoint, then image; its
    ``label``, where there is one, is the change label spanning the pair
    (*label* names it).  Of the reading arguments it takes ``images``,
    ``physical`` and ``dtype``: the pair chooses the visits, and *annotation*
    the foreground windows are drawn from, so ``timepoint=``,
    ``annotations=`` and ``label_format=`` are refused rather than taken and
    ignored.
    """

    def __init__(
        self,
        paths: Sequence[PathLike],
        sampler: PatchSampler,
        *,
        pair_sampler: TimepointPairSampler | None = None,
        align: str = "transform",
        samples_per_pair: int = 1,
        annotation: str | None = None,
        label: str | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        unread = sorted(set(kwargs) & {"annotations", "label_format", "timepoint"})
        if unread:
            raise MEDH5ValidationError(
                f"PairedPatchDataset does not read {unread}: a pair's visits are "
                "its own, `annotation=` names what its windows are drawn from, "
                "and `label=` the change label an item carries"
            )
        super().__init__(paths, **kwargs)
        if align not in ALIGNMENTS:
            raise MEDH5ValidationError(
                f"unknown align {align!r}; expected one of {list(ALIGNMENTS)}"
            )
        self.sampler = sampler
        self.pair_sampler = pair_sampler or TimepointPairSampler()
        self.align = align
        self.samples_per_pair = int(samples_per_pair)
        if self.samples_per_pair < 1:
            raise MEDH5ValidationError("samples_per_pair must be at least 1")
        self.annotation = annotation
        self.label = label
        self.seed = int(seed)
        self._epoch = _Epoch()
        self.report = PairReport()
        self._plan: list[tuple[str, TimepointPair]] = []
        for path in self.paths:
            with self._scan(path) as sample:
                pairs = self.pair_sampler.pairs(sample)
            self.report.files += 1
            if not pairs:
                self.report.add_skip(path)
                continue
            self._plan.extend((path, pair) for pair in pairs)
        self.report.pairs = len(self._plan)

    @property
    def epoch(self) -> int:
        return self._epoch.get()

    @epoch.setter
    def epoch(self, epoch: int) -> None:
        self._epoch.set(epoch)

    def set_epoch(self, epoch: int) -> None:
        """Redraw pairs' patches from the next item on; see :class:`_Epoch`."""
        self._epoch.set(epoch)

    def __len__(self) -> int:
        return len(self._plan) * self.samples_per_pair

    def file_groups(self) -> list[list[int]]:
        """Item indices per file, for :class:`~medh5.torch.FileGroupedSampler`."""
        n = self.samples_per_pair
        groups: dict[str, list[int]] = {}
        for position, (path, _) in enumerate(self._plan):
            groups.setdefault(path, []).extend(range(position * n, (position + 1) * n))
        return list(groups.values())

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, _ = self._plan[index // self.samples_per_pair]
        with self._lease(path) as sample:
            return self._paired_item(sample, index)

    def _paired_item(self, sample: Sample, index: int) -> dict[str, Any]:
        _, pair = self._plan[index // self.samples_per_pair]
        first = self._patch_for(sample, pair.first, (self.seed, self.epoch, index))
        second = self._corresponding(sample, pair, first)
        images = {
            pair.first: self._read_at(sample, pair.first, first),
            pair.second: self._read_at(sample, pair.second, second),
        }
        windows = {pair.first: first, pair.second: second}
        item = {
            "images": {
                tp: {k: to_tensor(v) for k, v in arrays.items()}
                for tp, arrays in images.items()
            },
            "valid": {
                tp: {
                    k: to_tensor(v)
                    for k, v in self._read_valid(
                        sample, list(arrays), windows[tp]
                    ).items()
                }
                for tp, arrays in images.items()
            },
            "meta": {
                **self._meta(sample, first),
                "pair": [pair.first, pair.second],
                "interval_days": pair.interval_days,
                "aligned": self.align,
                "patches": {
                    pair.first: first.to_json(),
                    pair.second: second.to_json(),
                },
            },
        }
        label_id = self.label or pair.label
        if label_id and label_id in sample.annotations:
            ann = sample.annotations[label_id]
            if ann.kind != "classification":
                raise MEDH5ValidationError(
                    f"{sample.path}: label={label_id!r} is a {ann.kind!r} "
                    "annotation; a pair's label is a classification of the "
                    "change between its two visits"
                )
            # Only the pair it spans: one pair's change was served as the
            # target of every other pair of the file.
            if tuple(ann.timepoints) == (pair.first, pair.second):
                item["label"] = {label_id: dict(ann.labels)}
                item["meta"]["label_annotation"] = label_id
        return item

    # -- internals ---------------------------------------------------------

    def _annotation_at(
        self, sample: Sample, timepoint: str, grid: str | None = None
    ) -> str | None:
        """The annotation to draw *timepoint*'s foreground from, if there is one.

        Auto-selection is confined to *grid* --- the space the images being read
        live in --- so a visit holding both a CT and a PET grid picks the
        annotation belonging to the modality that was asked for rather than
        whichever annotation sorts first.

        A named ``annotation`` lies on one visit's grid, and is drawn from at
        that visit only; elsewhere the visit's own is selected, as without
        one.  Taken everywhere, it was refused for every pair not starting at
        its visit.  A name the file does not hold is still the sampler's to
        refuse.
        """
        named = self.annotation
        if named is not None:
            if named not in sample.annotations:
                return named
            if grid is None or sample.annotations[named].grid_id == grid:
                return named
        for name in sorted(sample.at(timepoint).annotations):
            annotation = sample.annotations[name]
            if annotation.kind == "classification":
                continue
            if grid is not None and annotation.grid_id != grid:
                continue
            return name
        return None

    def _images_at(self, sample: Sample, timepoint: str) -> list[str]:
        """The images this dataset reads at *timepoint*."""
        view = sample.at(timepoint)
        if self.images is None:
            return sorted(view.images)
        return [name for name in self.images if name in view.images]

    def _window_grid_at(self, sample: Sample, timepoint: str) -> str:
        """The grid every window for *timepoint* is measured in.

        Taken from the data this dataset will actually read at that visit
        rather than from whichever grid sorts first: a visit may legitimately
        hold a CT grid and a PET grid, and pinning the window to the
        alphabetical winner refused the PET pair instead of sampling it.  Both
        ends of a pair go through here, so the window drawn in the first visit
        and the one derived for the second agree about what they are measuring.
        """
        names = self._images_at(sample, timepoint)
        if not names:
            return str(self._grid_at(sample, timepoint).grid_id)
        return self._single_grid(
            sample, {f"image {n!r}": sample.images[n].grid_id for n in names}
        )

    def _patch_for(self, sample: Sample, timepoint: str, rng: Seed) -> Patch:
        """A window in *timepoint*'s own grid, annotated or not.

        The grid is named rather than left to the sampler.  `_annotation_at`
        answers `None` for a visit with no voxel annotation, and `None` on its
        own tells the sampler to go and find one anywhere in the sample --- so a
        pair whose follow-up is annotated and whose baseline is not drew the
        baseline window in the follow-up's grid.  It comes from the images
        being read rather than from the visit's first grid, so a visit holding
        more than one modality is sampled in the one that was asked for.
        """
        grid = self._window_grid_at(sample, timepoint)
        return self.sampler.draw(
            sample, self._annotation_at(sample, timepoint, grid), rng, grid=grid
        )

    def _corresponding(
        self, sample: Sample, pair: TimepointPair, patch: Patch
    ) -> Patch:
        """The window in the second visit covering the same anatomy."""
        target = sample.grids[self._window_grid_at(sample, pair.second)]
        shape = target.spatial_shape
        # The *requested* size, not the clipped extent.  Where the first visit
        # is smaller than the patch its window is short and gets padded back up
        # to `patch.shape`; asking the second visit for the clipped extent would
        # return a full, unpadded array of that smaller size and leave the pair
        # with tensors of different shapes.
        size = patch.shape
        if self.align == "none":
            from medh5.sampling import window_around

            slices, pad = window_around(patch.center, size, shape)
            return Patch(
                slices=slices,
                pad=pad,
                center=patch.center,
                strategy="paired",
                grid_id=target.grid_id,
            )
        center = self._map_center(sample, pair, patch)
        from medh5.sampling import window_around

        slices, pad = window_around(center, size, shape)
        return Patch(
            slices=slices,
            pad=pad,
            center=center,
            strategy="paired",
            grid_id=target.grid_id,
        )

    def _grid_at(self, sample: Sample, timepoint: str) -> Any:
        view = sample.at(timepoint)
        names = sorted(view.grids)
        if not names:
            raise MEDH5ValidationError(
                f"{sample.path}: timepoint {timepoint!r} has no grid"
            )
        return sample.grids[names[0]]

    def _map_center(
        self, sample: Sample, pair: TimepointPair, patch: Patch
    ) -> tuple[int, ...]:
        """Move the centre through the transform relating the two frames (§10.2)."""
        source = sample.grids[patch.grid_id or self._window_grid_at(sample, pair.first)]
        target = sample.grids[self._window_grid_at(sample, pair.second)]
        world = source.index_to_world(np.asarray([patch.center], dtype=np.float64))
        # Resolve between the two *frames*, not the two timepoints and not by
        # name.  A visit may hold a CT grid and a PET grid on different frames,
        # and a timepoint-level question searches every frame of one visit
        # against every frame of the other and returns the first path it finds
        # --- so a CT registration answered "yes" for a PET pair with no
        # registration of its own, and its displacement was then applied to PET
        # coordinates.  Nothing raised; the patches came back the right shape
        # from the wrong place.
        #
        # Asking `transform_between` for the two *grid ids* is not enough to
        # close that.  Grid ids and timepoint ids are separate namespaces (§2.3
        # scopes uniqueness to the group), so a grid may legitimately be named
        # `tp0`, and `Sample._frames_for` reads a key as a timepoint before it
        # reads it as a grid --- which puts the whole visit's frames back in
        # play and restores the bug for exactly the files most likely to hit it.
        # Here the grids are already in hand, so resolve frame to frame and let
        # no name be interpreted at all.
        transform = None
        if source.frame_uid and target.frame_uid:
            transform = sample.resolve_frames(source.frame_uid, target.frame_uid)
        one_frame = source.frame_uid is not None and (
            source.frame_uid == target.frame_uid
        )
        if transform is None and not one_frame:
            # A `None` here has two possible meanings: the grids already share
            # a frame (nothing to apply), or no path exists between them.  Only
            # the first makes the coordinates comparable, and §3.3 says what
            # "share" means: the same declared frame, not the same *absence*
            # of one.  Comparing the two `frame_uid`s let two frame-less grids
            # --- every longitudinal sample the NIfTI and nnU-Net importers
            # write --- through as registered (`None == None`), feeding one
            # visit's world coordinates into the other's grid and returning
            # paired patches from different anatomy, which trains quietly.
            raise MEDH5ValidationError(
                f"{sample.path}: align='transform' needs a transform relating "
                f"grid {source.grid_id!r} at {pair.first!r} (frame "
                f"{source.frame_uid!r}) to grid {target.grid_id!r} at "
                f"{pair.second!r} (frame {target.frame_uid!r}), and the file has "
                "none"
                + (
                    " --- and grids without a frame of reference are never "
                    "comparable, whatever their coordinates say (§3.3)"
                    if source.frame_uid is None or target.frame_uid is None
                    else ""
                )
                + "; register those grids, or use align='none' to read the same "
                "index window from both"
            )
        if transform is not None:
            _require_units(transform, source, target)
            world = transform.transform_points(world)
        else:
            # One frame, but each grid's numbers are in its own units (§3.5):
            # a point in millimetres read on a grid in metres landed a
            # thousandfold off.  Two conventions are refused here (E414):
            # §3.3 rule 4 compares world coordinates in one `coord_system`
            # only.
            world = source.world_into(target, world)
        index = target.world_to_index(world)[0]
        return tuple(int(round(float(v))) for v in index)

    def _read_at(
        self, sample: Sample, timepoint: str, patch: Patch
    ) -> dict[str, npt.NDArray[Any]]:
        wanted = self._images_at(sample, timepoint)
        if wanted:
            members = {f"image {n!r}": sample.images[n].grid_id for n in wanted}
            if patch.grid_id is not None:
                members["the patch window"] = patch.grid_id
            self._single_grid(sample, members)
        out: dict[str, npt.NDArray[Any]] = {}
        for image_id in wanted:
            array = sample.images[image_id].read(
                patch.slices, physical=self.physical, dtype=self.dtype
            )
            out[image_id] = patch.apply_padding(array)
        return out


def _object_rows(ann: Any) -> Iterator[tuple[int, int, Any, Any]]:
    """``(instance_id, class_id, box, score)`` of every object of *ann*.

    From the object columns where the encoding keeps them: ``instances()``
    also decodes every object's mask, for every patch, and an item carries
    none of them.
    """
    if isinstance(ann, InstancesAnnotation):
        scores = ann.scores
        rows = zip(
            ann.instance_ids.tolist(),
            ann.object_class_ids.tolist(),
            np.asarray(ann.boxes, dtype=np.float32),
            strict=True,
        )
        for i, (instance_id, class_id, box) in enumerate(rows):
            yield (
                instance_id,
                class_id,
                box,
                None if scores is None else float(scores[i]),
            )
        return
    for obj in ann.instances():
        yield obj.instance_id, obj.class_id, obj.box, obj.score


def _require_units(transform: Any, source: Any, target: Any) -> None:
    """Refuse a transform whose coordinates are not both grids' (§10.1, E506).

    A transform maps points in its own ``units``, and 1.0 defines no transform
    across units.  One in millimetres between two metre grids moved every
    point a thousand times as far as it was meant to; nothing is converted
    here --- the file is wrong, and says so.
    """
    wrong = [grid for grid in (source, target) if grid.units != transform.units]
    if wrong:
        raise MEDH5ValidationError(
            f"transform {transform.transform_id!r} maps coordinates in "
            f"{transform.units!r}, and "
            + ", ".join(f"grid {grid.grid_id!r} is in {grid.units!r}" for grid in wrong)
            + ": a transform applies only to grids in its units (§10.1)",
            code="E506",
        )


__all__ = [
    "ALIGNMENTS",
    "LABEL_FORMATS",
    "GridPatchDataset",
    "PairedPatchDataset",
    "PatchDataset",
    "VolumeDataset",
]
