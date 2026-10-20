"""MONAI's own reader and loader for ``.medh5`` files.

Private, and imported by :mod:`medh5.monai` only when one of its names is
asked for: both classes subclass MONAI's (``ImageReader``, ``MapTransform``),
so defining them imports MONAI, and ``import medh5.monai`` must not.  Where
MONAI is not installed they are defined over ``object`` and refuse to be
built, naming the extra --- as the torch datasets are without torch --- so
the names, and their signatures, are the same in every environment.

:class:`MedH5Reader` plugs into ``LoadImage``/``LoadImaged``: it reads one
image --- or, with ``annotation=``, one voxel annotation as an ``int64`` label
volume --- of each file it is given, with the affine and metadata
:func:`~medh5.monai.to_metatensor` builds.  :class:`LoadMedH5d` is
:func:`~medh5.monai.to_dict` as a dictionary transform.  Both read through
the per-process handle cache of :mod:`medh5.torch.handles`, so a worker opens
a file once however many items read it, and never reads through a handle
inherited across ``fork``.

A collection member is named ``path::key``, as a split names it
(:meth:`~medh5.dataset.split.Split.paths`); the name is split only after a
``.medh5c`` suffix, so ``::`` anywhere else is part of the path.
"""

from __future__ import annotations

import os
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.errors import MEDH5ValidationError
from medh5.monai import _label_volume, meta_dict, require_monai, to_dict
from medh5.sample import Sample
from medh5.torch.handles import CACHE


def _bases() -> tuple[type, type]:
    """MONAI's ``ImageReader`` and ``MapTransform``, or ``object`` twice."""
    try:
        from monai.data import ImageReader
        from monai.transforms import MapTransform
    except ImportError:  # pragma: no cover - monai is an optional extra
        return object, object
    return ImageReader, MapTransform


_ReaderBase, _TransformBase = _bases()

SUFFIXES: tuple[str, ...] = _core.MANIFEST_SUFFIXES
_MEMBER = ".medh5c::"

Reference = tuple[str, str | None]
"""``(path, sample_key)``: a standalone file is ``(path, None)``."""


def split_reference(name: Any, sample_key: str | None = None) -> Reference:
    """``path::key`` as ``(path, key)``; any other name as ``(path,
    sample_key)``.  Only a ``.medh5c`` suffix ends a path before ``::``."""
    text = os.fspath(name)
    at = text.rfind(_MEMBER)
    if at < 0:
        return text, sample_key
    end = at + len(_MEMBER) - 2
    return text[:end], text[end + 2 :]


def _names(data: Any) -> list[Any]:
    if isinstance(data, (str, os.PathLike)):
        return [data]
    return list(data)


def _where(reference: Reference) -> str:
    path, key = reference
    return path if key is None else f"{path}::{key}"


class MedH5Reader(_ReaderBase):  # type: ignore[misc,valid-type]
    """A MONAI ``ImageReader`` for ``.medh5`` samples and collection members.

    ``image`` names the image to read; ``None`` reads the sample's only image,
    and refuses a sample holding several, naming them.  ``annotation`` reads a
    voxel annotation instead, as :func:`~medh5.monai.to_dict` does: ``int64``
    class ids, ``65535`` in the §7.7 ignore region, the affine of the
    annotation's own grid (pass ``dtype=None`` to ``LoadImage``, which
    otherwise casts to ``float32``).  ``sample_key`` names the member of a
    ``.medh5c`` given without ``::key``.  ``physical``, ``space``, ``level`` and
    ``dtype`` are :func:`~medh5.monai.to_metatensor`'s.

    Several files given to one ``LoadImage`` call stack as channels, only
    when their volumes share one grid (shape, affine and space); their
    ``original_channel_dim`` is then ``0``.

    :meth:`read` returns references only; :meth:`get_data` reads each through
    the handle cache, leased for the length of the read.
    """

    def __init__(
        self,
        image: str | None = None,
        *,
        annotation: str | None = None,
        sample_key: str | None = None,
        physical: bool = True,
        space: str | None = None,
        level: int = 0,
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        require_monai()
        super().__init__()
        if image is not None and annotation is not None:
            raise MEDH5ValidationError(
                f"a MedH5Reader reads an image or an annotation, not both "
                f"(image={image!r}, annotation={annotation!r}): use one reader "
                "for each"
            )
        self.image = image
        self.annotation = annotation
        self.sample_key = sample_key
        self.physical = bool(physical)
        self.space = space
        self.level = int(level)
        self.dtype = dtype

    def verify_suffix(self, filename: Any) -> bool:
        """Whether every name is a ``.medh5``, a ``.medh5c`` or ``path::key``."""
        return all(
            split_reference(name)[0].lower().endswith(SUFFIXES)
            for name in _names(filename)
        )

    def read(self, data: Any, **kwargs: Any) -> list[Reference]:
        """``(path, sample_key)`` of each name; nothing is opened here."""
        return [split_reference(name, self.sample_key) for name in _names(data)]

    def get_data(self, img: Any) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        """The array and metadata of every reference :meth:`read` returned,
        stacked as channels when there are several."""
        references: list[Reference] = (
            [img] if isinstance(img, tuple) and len(img) == 2 else list(img)
        )
        if not references:
            raise MEDH5ValidationError("MedH5Reader.get_data was given no file")
        read: list[tuple[npt.NDArray[Any], dict[str, Any]]] = []
        for reference in references:
            path, key = reference
            if key is None and path.lower().endswith(".medh5c"):
                raise MEDH5ValidationError(
                    f"{path} is a collection: name its member as "
                    f"'{path}::<key>', or give the reader sample_key="
                )
            with CACHE.lease(path, key) as sample:
                read.append(self._one(sample, reference))
        if len(read) == 1:
            return read[0]
        return _stacked(references, read)

    def _one(
        self, sample: Sample, reference: Reference
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        if self.annotation is not None:
            planes, meta = _label_volume(
                sample, self.annotation, space=self.space, reader="MedH5Reader"
            )
            return planes, meta
        image_id = self.image
        if image_id is None:
            names = sorted(sample.images)
            if len(names) != 1:
                raise MEDH5ValidationError(
                    f"{_where(reference)} holds {len(names)} images "
                    f"({', '.join(map(repr, names)) or 'none'}): name the one to "
                    "read, MedH5Reader(image=...)"
                )
            image_id = names[0]
        image = sample.images[image_id]
        source = image.level(self.level) if self.level else image
        array = np.ascontiguousarray(
            source.read(None, physical=self.physical, dtype=self.dtype)
        )
        return array, meta_dict(sample, image_id, space=self.space, level=self.level)


def _stacked(
    references: Sequence[Reference],
    read: Sequence[tuple[npt.NDArray[Any], dict[str, Any]]],
) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """Several volumes as the channels of one, when they share one grid."""
    first, meta = read[0]
    for reference, (array, other) in zip(references[1:], read[1:], strict=True):
        same = (
            tuple(other["spatial_shape"]) == tuple(meta["spatial_shape"])
            and other["space"] == meta["space"]
            and other["original_channel_dim"] == meta["original_channel_dim"]
            and array.shape == first.shape
            and np.allclose(other["affine"], meta["affine"])
        )
        if not same:
            raise MEDH5ValidationError(
                f"{_where(references[0])} and {_where(reference)} are not on one "
                "grid (shape, affine and space), so they cannot be stacked as "
                "channels: load them as separate keys, or resample one onto the "
                "other's grid first (`medh5 resample PATH --image ID --onto "
                "GRID`) if a model needs both stacked"
            )
    arrays = [array for array, _ in read]
    out = dict(meta)
    channel = meta["original_channel_dim"]
    if channel == "no_channel":
        out["original_channel_dim"] = 0
        return np.stack(arrays, axis=0), out
    return np.concatenate(arrays, axis=int(channel)), out


class LoadMedH5d(_TransformBase):  # type: ignore[misc,valid-type]
    """:func:`~medh5.monai.to_dict` as a MONAI dictionary transform.

    The item's ``keys`` entry names a sample (a path, or ``path::key`` for a
    collection member); the transform adds :func:`~medh5.monai.to_dict`'s
    keys --- one ``MetaTensor`` per image and per annotation, by id --- read
    from one lease of the cached handle.  ``images=None`` reads every image.
    One input key only: a second would write the same output keys again.
    """

    def __init__(
        self,
        keys: str | Sequence[str] = "path",
        *,
        images: Sequence[str] | None = None,
        annotations: Sequence[str] = (),
        space: str | None = None,
        physical: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        require_monai()
        super().__init__(keys, allow_missing_keys)
        if len(self.keys) != 1:
            raise MEDH5ValidationError(
                f"LoadMedH5d reads one sample per item, from one key; given "
                f"{list(self.keys)}, the samples' images and annotations would "
                "overwrite each other: use one LoadMedH5d per key"
            )
        self.images = None if images is None else tuple(images)
        self.annotations = tuple(annotations)
        self.space = space
        self.physical = bool(physical)

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        out = dict(data)
        for key in self.key_iterator(out):
            path, member = split_reference(out[key])
            with CACHE.lease(path, member) as sample:
                out.update(
                    to_dict(
                        sample,
                        self.images,
                        self.annotations,
                        space=self.space,
                        physical=self.physical,
                    )
                )
        return out


__all__ = ["LoadMedH5d", "MedH5Reader"]
