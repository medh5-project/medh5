"""Registration transforms (spec §10).

A transform maps points from ``from_frame`` to ``to_frame``: ``x_M = T(x_F)``,
the ITK convention, with no attribute to switch it.  The classes here are views
over the format engine's transform model, which evaluates every kind:
``identity`` and ``affine`` (§10.3), ``displacement`` (§10.4), ``bspline``
and ``composite`` (§10.5).

Resolution walks the frame graph, not transform names; it inverts a step only
where the inverse can be evaluated, refuses an ambiguous pair, and returns
``None`` --- never an invented transform --- when no path exists.

The field numerics are the engine's too --- linear and cubic interpolation match
SciPy's ``map_coordinates`` between a field's outermost samples, and take the
outermost sample's value in the half-voxel margin beyond them (§10.4), and the
Jacobian matches NumPy's gradient --- so every frontend evaluates a stored field
identically (§10.4--§10.6).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.geometry import Grid

TRANSFORM_KINDS: tuple[str, ...] = _core.TRANSFORM_KINDS
VECTOR_SPACES: tuple[str, ...] = _core.VECTOR_SPACES
INTERPOLATIONS: tuple[str, ...] = _core.INTERPOLATIONS
EXTRAPOLATIONS: tuple[str, ...] = _core.EXTRAPOLATIONS
SPEC_TRANSFORM_ATTRS: tuple[str, ...] = _core.SPEC_TRANSFORM_ATTRS
LAST_ROW_TOL: float = _core.LAST_ROW_TOL
FLOAT16_SAFE_VOXELS: float = _core.FLOAT16_SAFE_VOXELS
SUPPORTED_ORDERS: tuple[int, ...] = _core.SUPPORTED_ORDERS
DEFAULT_ORDER: int = _core.DEFAULT_ORDER
RESAMPLE_INTERPOLATIONS: tuple[str, ...] = _core.RESAMPLE_INTERPOLATIONS

TransformHeader = _core.TransformHeader
"""The attribute header every transform carries (spec §10.1)."""

check_transform_id = _core.check_transform_id

# -- encoders: what ``SampleWriter.add_transform`` stores -----------------------

encode_identity = _core.encode_identity
encode_affine = _core.encode_affine
encode_bspline = _core.encode_bspline
encode_composite = _core.encode_composite


def encode_displacement(
    field: npt.ArrayLike,
    *,
    field_grid: str,
    vector_space: str = "world",
    interpolation: str = "linear",
    extrapolation: str = "zero",
    dtype: npt.DTypeLike = np.float32,
) -> Any:
    """Pack a ``(S, *spatial)`` displacement field (§10.4), cast to *dtype*."""
    return _core.encode_displacement(
        field,
        field_grid=field_grid,
        vector_space=vector_space,
        interpolation=interpolation,
        extrapolation=extrapolation,
        dtype=dtype,
    )


# -- field numerics (§10.4--§10.6) ----------------------------------------------

basis = _core.basis
inside_extent = _core.inside_extent
refuse_outside = _core.refuse_outside
linear_sample = _core.linear_sample
cubic_sample = _core.cubic_sample
sample_field = _core.sample_field
linear_part = _core.linear_part
to_world_vectors = _core.to_world_vectors
jacobian_determinant = _core.jacobian_determinant
folding_fraction = _core.folding_fraction
target_registration_error = _core.target_registration_error

# -- the read contract every kind shares (§10.1) ---------------------------------


class Transform:
    """What every transform answers, whatever its kind."""

    __slots__ = ("_handle", "transform_id")

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.transform_id: str = handle.transform_id

    @property
    def header(self) -> Any:
        return self._handle.header

    @property
    def group(self) -> Any:
        """The stored group (a read-only :class:`~medh5.nodes.Group`)."""
        return self._handle.group

    @property
    def kind(self) -> str:
        return str(self._handle.kind)

    @property
    def from_frame(self) -> str:
        return str(self._handle.from_frame)

    @property
    def to_frame(self) -> str:
        return str(self._handle.to_frame)

    @property
    def units(self) -> str:
        return str(self._handle.units)

    @property
    def prov(self) -> str | None:
        found: str | None = self._handle.prov
        return found

    @property
    def metrics_key(self) -> str | None:
        found: str | None = self._handle.metrics_key
        return found

    @property
    def is_invertible(self) -> bool:
        """Whether the mapping is *claimed* invertible: declared
        (``invertible``, ``inverse_id``), analytic, or true of every step.
        Whether an inverse can be evaluated is
        :meth:`InverseTransform.can_invert`."""
        return bool(self._handle.is_invertible)

    @property
    def timepoints(self) -> tuple[str, ...]:
        return tuple(self._handle.timepoints)

    def grid_in(self, frame: str) -> Grid | None:
        """A grid of this transform's file declared in *frame*, if any."""
        found: Grid | None = self._handle.grid_in(frame)
        return found

    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map ``(..., S)`` world points from ``from_frame`` to ``to_frame``."""
        found: npt.NDArray[np.float64] = self._handle.transform_points(points)
        return found

    def inverse(self) -> Transform | None:
        """The inverse, or ``None`` when none can be evaluated.

        The stored inverse (``inverse_id``) when the file carries one.  A
        :class:`ChainTransform`, or a composite without a usable
        ``inverse_id``, inverts as the chain of its steps' inverses in reverse
        order, when every step can be inverted (an affine's or identity's
        analytic inverse, or a stored one).  :attr:`is_invertible` is the
        claim; :meth:`InverseTransform.can_invert` says whether an inverse can
        be evaluated.
        """
        found = self._handle.inverse()
        return None if found is None else wrap_transform(found)

    def summary(self) -> dict[str, Any]:
        found: dict[str, Any] = self._handle.summary()
        return found

    def __repr__(self) -> str:
        return str(self._handle.__repr__())


# -- the kinds --------------------------------------------------------------------


class IdentityTransform(Transform):
    """Two frames that coincide, said explicitly."""

    __slots__ = ()


class AffineTransform(Transform):
    """A homogeneous ``(S+1, S+1)`` matrix in world coordinates."""

    __slots__ = ()

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.matrix
        return found

    @property
    def n_spatial(self) -> int:
        return int(self._handle.n_spatial)

    def inverse_matrix(self) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.inverse_matrix()
        return found

    def inverse_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.inverse_points(points)
        return found

    @property
    def jacobian_determinant_value(self) -> float:
        return float(self._handle.jacobian_determinant_value)


class DisplacementTransform(Transform):
    """``T(x) = x + u(x)`` with ``u`` sampled on ``field_grid``."""

    __slots__ = ()

    @property
    def field(self) -> Any:
        """The stored field dataset (read regions with :meth:`read_field`)."""
        return self._handle.field

    @property
    def field_grid_id(self) -> str:
        return str(self._handle.field_grid_id)

    @property
    def field_grid(self) -> Grid:
        found: Grid = self._handle.field_grid
        return found

    @property
    def vector_space(self) -> str:
        return str(self._handle.vector_space)

    @property
    def interpolation(self) -> str:
        return str(self._handle.interpolation)

    @property
    def extrapolation(self) -> str:
        return str(self._handle.extrapolation)

    def read_field(
        self, roi: Sequence[slice] | None = None, component: int | None = None
    ) -> npt.NDArray[Any]:
        found: npt.NDArray[Any] = self._handle.read_field(roi, component)
        return found

    def displacement_at(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """World displacement vectors at ``(..., S)`` world points."""
        found: npt.NDArray[np.float64] = self._handle.displacement_at(points)
        return found

    def sample_indices(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """The stored field interpolated at ``(N, S)`` continuous field indices.

        A paired dataset asks for one displacement per training item, and
        reading the *entire* field for each turned a deformable registration
        into a full-volume decompress per item.  Linear interpolation needs the
        two lattice points either side of each query along each axis, so the
        read is the bounding window of the points, padded by one --- kilobytes
        of a 512³ field.  The result equals sampling the whole field
        (:func:`sample_field`).  Cubic interpolation reads the whole field: its
        spline coefficients are global.
        """
        found: npt.NDArray[np.float64] = self._handle.sample_indices(indices)
        return found

    def jacobian_determinant(
        self, roi: Sequence[slice] | None = None
    ) -> npt.NDArray[np.float64]:
        """``det(I + du/dx)`` per voxel of the field grid, or of *roi*.

        *roi* takes what :meth:`read_field` takes: one slice per spatial axis,
        whole axes after the last one given, NumPy's negative and out-of-range
        bounds, and a positive step.  The result is the whole field's
        determinant at the voxels *roi* selects,
        ``jacobian_determinant(roi) == jacobian_determinant()[roi]``: the
        differences reach one voxel past the region and are one-sided only at
        the field's own edges, so a strided region keeps a fold narrower than
        its step and an axis of one voxel is accepted.  Only the voxels the
        region needs are read.  The field itself needs at least 2 voxels on
        every axis.
        """
        found: npt.NDArray[np.float64] = self._handle.jacobian_determinant(roi)
        return found

    def folding_fraction(self, roi: Sequence[slice] | None = None) -> float:
        """The fraction of voxels where the map folds (``det J <= 0``), over
        the field or the voxels *roi* selects (as :meth:`jacobian_determinant`
        takes it)."""
        return float(self._handle.folding_fraction(roi))

    @property
    def max_magnitude(self) -> float:
        return float(self._handle.max_magnitude)


class BSplineTransform(Transform):
    """A free-form deformation evaluated with a uniform B-spline basis."""

    __slots__ = ()

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.control_points
        return found

    @property
    def cp_grid_id(self) -> str:
        return str(self._handle.cp_grid_id)

    @property
    def cp_grid(self) -> Grid:
        found: Grid = self._handle.cp_grid
        return found

    @property
    def order(self) -> int:
        return int(self._handle.order)

    @property
    def vector_space(self) -> str:
        return str(self._handle.vector_space)

    def displacement_at(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.displacement_at(points)
        return found

    def to_displacement_field(self, grid: Grid) -> npt.NDArray[np.float32]:
        """The equivalent dense ``(S, *spatial)`` field on *grid*."""
        found: npt.NDArray[np.float32] = self._handle.to_displacement_field(grid)
        return found


class CompositeTransform(Transform):
    """Components applied in order; each one's ``to_frame`` is the next's
    ``from_frame``."""

    __slots__ = ()

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(self._handle.component_ids)

    def components(self) -> tuple[Transform, ...]:
        return tuple(wrap_transform(h) for h in self._handle.components())

    def check_chain(self) -> list[str]:
        """Every break in the frame chain (empty when it is whole)."""
        return list(self._handle.check_chain())


class InverseTransform(Transform):
    """The inverse of another transform, evaluated (not stored)."""

    __slots__ = ()

    def __init__(self, inner: Transform) -> None:
        super().__init__(_core.TransformHandle.inverse_of(inner._handle))

    @staticmethod
    def can_invert(inner: Transform) -> bool:
        """Whether *inner*'s inverse can be evaluated, not merely declared."""
        return bool(_core.TransformHandle.can_invert(inner._handle))


class ChainTransform(Transform):
    """Several transforms applied in order --- a resolved path.

    Each step must start in the frame the step before it ends in, as a
    composite's components must; steps whose frames do not meet, or that are
    in different units, are refused (E501).
    """

    __slots__ = ()

    def __init__(self, chain: Sequence[Transform]) -> None:
        super().__init__(_core.TransformHandle.chain([t._handle for t in chain]))

    @property
    def steps(self) -> tuple[Transform, ...]:
        return tuple(wrap_transform(h) for h in self._handle.steps or ())


_READERS: dict[str, type[Transform]] = {
    cls.__name__: cls
    for cls in (
        IdentityTransform,
        AffineTransform,
        DisplacementTransform,
        BSplineTransform,
        CompositeTransform,
        InverseTransform,
        ChainTransform,
    )
}
"""Engine class name -> facade class."""


def wrap_transform(handle: Any) -> Transform:
    """The facade class for an engine transform handle."""
    cls = _READERS.get(handle.class_name, Transform)
    made: Transform = cls.__new__(cls)
    Transform.__init__(made, handle)
    return made


# -- resolution (§10.1) -------------------------------------------------------------


def frame_graph(transforms: Mapping[str, Transform]) -> dict[str, list[str]]:
    """Frame -> frames one hop away, as :func:`resolve_between` would walk
    them: a reverse edge exists only where the inverse can be evaluated, and
    a transform and the stored inverse its ``inverse_id`` names are one edge
    each way (§10.1)."""
    found: dict[str, list[str]] = _core.frame_graph(transforms)
    return found


def resolve_between(
    transforms: Mapping[str, Transform], from_frame: str, to_frame: str
) -> Transform | None:
    """The transform relating two frame uids, or ``None`` when none does."""
    found = _core.resolve_between(transforms, from_frame, to_frame)
    return None if found is None else wrap_transform(found)


def frames_of_timepoint(grids: Mapping[str, Grid], timepoint: str) -> tuple[str, ...]:
    """The frames of every grid of one timepoint, in grid order."""
    return tuple(_core.frames_of_timepoint(grids, timepoint))


# -- resampling, a recorded derivation (§10.2, §4.4, §11.1) -------------------------


@dataclass(frozen=True, slots=True)
class ResampledImage:
    """One image a resample wrote."""

    source: str
    """The image resampled."""
    image_id: str
    """The image written."""
    dtype: str
    """The dtype written."""
    valid_mask: str | None = None
    """Its ``valid_mask`` (§4.4), written when any voxel holds no data."""
    filled_voxels: int = 0
    """Target voxels whose source point lay outside the source, which took
    the fill value (spatial voxels, counted once whatever the channels)."""


@dataclass(frozen=True, slots=True)
class ResampledAnnotation:
    """One annotation a resample carried onto the target."""

    source: str
    """The annotation carried."""
    annotation_id: str
    """The annotation written, which names *source* in ``derived_from``."""
    kind: str
    ignore_mask: str | None = None
    """The ignore mask the written annotation names (§7.7): its own, carried,
    or a new one over the target voxels outside the source."""
    note: str | None = None
    """What the carry changed beyond the coordinates, when anything: a
    normalized probability map with voxels outside the source is written
    with ``normalized`` false, for one."""


@dataclass(frozen=True, slots=True)
class ResampleReport:
    """What :func:`resample` wrote."""

    path: str
    """The file written: the source, amended, or ``out``."""
    activity_id: str
    """The ``resample`` activity every output names in its ``prov``."""
    grid_id: str
    """The target grid."""
    new_grid: bool = False
    """Whether the resample wrote the target grid."""
    images: tuple[ResampledImage, ...] = ()
    transforms: dict[str, list[str]] = field(default_factory=dict)
    """The transform ids on each path, by source grid, for the source grids
    reached through a transform; an inverted step is ``T⁻¹``.  A grid sharing
    the target's frame (§3.3 rule 4) is mapped by the affines alone."""
    annotations: tuple[ResampledAnnotation, ...] = ()
    """The annotations carried onto the target, in the sample's order."""
    not_carried: tuple[str, ...] = ()
    """The annotations on the source grids left where they are: all of them
    with ``annotations=False``."""
    written: tuple[str, ...] = ()
    """Every object written, as a path in the sample (``images/CT_on_ref``)."""
    filled_voxels: int = 0
    """The sum of every image's ``filled_voxels``."""
    content_id: str | None = None
    """The written file's ``content_id``."""

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> ResampleReport:
        return cls(
            path=str(doc["path"]),
            activity_id=str(doc["activity_id"]),
            grid_id=str(doc["grid_id"]),
            new_grid=bool(doc.get("new_grid", False)),
            images=tuple(
                ResampledImage(
                    source=str(i["source"]),
                    image_id=str(i["image_id"]),
                    dtype=str(i["dtype"]),
                    valid_mask=i.get("valid_mask"),
                    filled_voxels=int(i.get("filled_voxels", 0)),
                )
                for i in doc.get("images") or ()
            ),
            transforms={
                str(k): [str(t) for t in v]
                for k, v in (doc.get("transforms") or {}).items()
            },
            annotations=tuple(
                ResampledAnnotation(
                    source=str(a["source"]),
                    annotation_id=str(a["annotation_id"]),
                    kind=str(a["kind"]),
                    ignore_mask=a.get("ignore_mask"),
                    note=a.get("note"),
                )
                for a in doc.get("annotations") or ()
            ),
            not_carried=tuple(str(a) for a in doc.get("not_carried") or ()),
            written=tuple(str(w) for w in doc.get("written") or ()),
            filled_voxels=int(doc.get("filled_voxels", 0)),
            content_id=doc.get("content_id"),
        )


def resample(
    path: str | os.PathLike[str],
    image_ids: str | Sequence[str],
    *,
    onto: str | None = None,
    spacing: Sequence[float] | None = None,
    grid_id: str | None = None,
    suffix: str | None = None,
    interpolation: str = "linear",
    fill: float = 0,
    dtype: npt.DTypeLike | None = None,
    out: str | os.PathLike[str] | None = None,
    annotations: bool = True,
) -> ResampleReport:
    """Resample images of the sample at *path* onto another grid, and record it.

    Give exactly one target: *onto*, an existing grid of the sample, or
    *spacing* (one value per spatial axis, in stored order) with *grid_id*, a
    new grid in the source's frame that covers the same extent edge to edge.
    Grids sharing a frame of reference are mapped by their affines; any
    other pair through the sample's transforms (§10.2), and a pair no
    transform relates is refused --- alignment is never assumed.

    Each image is written beside its source as ``<id><suffix>`` (``suffix``
    defaults to ``_on_<target grid>``), interpolated ``linear`` (default),
    ``nearest`` or ``cubic`` on its stored values, so its rescale carries
    over.  ``nearest`` keeps the dtype; ``linear`` and ``cubic`` write
    ``float32`` unless *dtype* is given, rounded half to even and clipped.
    A voxel whose source point lies outside the source takes *fill*, and the
    image's ``valid_mask`` (§4.4) marks the voxels holding data; a non-finite
    *fill* is recorded as Python spells it (``"nan"``, ``"inf"``).  One
    ``resample`` activity records the inputs, the target, the transform path
    and the interpolation, and every object written names it.

    The annotations on the source grids follow the images, each written as
    ``<id><suffix>`` naming its source in ``derived_from``, with its classes
    and coverage unchanged: voxel encodings nearest in their own encoding,
    ``instances`` object by object under their own ids, a probability map
    linearly, points, keypoints and meshes exactly.  A box is carried only
    where the mapping keeps the axes, and a contour only where its plane
    lands on a grid plane; anything that cannot be carried exactly refuses
    the resample.  Target voxels outside the source become an ignore region
    (§7.7) of every class-bearing annotation carried.  ``annotations=False``
    moves the images alone, and the report lists what stayed behind in
    ``not_carried``.

    ``out=None`` amends *path* (copy-on-write); otherwise the result is a new
    file, and an existing *out* is refused.
    """
    ids = [image_ids] if isinstance(image_ids, str) else list(image_ids)
    return ResampleReport.from_json(
        _core.resample(
            os.fspath(path),
            ids,
            onto=onto,
            spacing=None if spacing is None else [float(v) for v in spacing],
            grid_id=grid_id,
            suffix=suffix,
            interpolation=interpolation,
            fill=float(fill),
            dtype=dtype,
            out=None if out is None else os.fspath(out),
            annotations=bool(annotations),
        )
    )


__all__ = [
    "DEFAULT_ORDER",
    "EXTRAPOLATIONS",
    "FLOAT16_SAFE_VOXELS",
    "INTERPOLATIONS",
    "LAST_ROW_TOL",
    "RESAMPLE_INTERPOLATIONS",
    "SPEC_TRANSFORM_ATTRS",
    "SUPPORTED_ORDERS",
    "TRANSFORM_KINDS",
    "VECTOR_SPACES",
    "AffineTransform",
    "BSplineTransform",
    "ChainTransform",
    "CompositeTransform",
    "DisplacementTransform",
    "IdentityTransform",
    "InverseTransform",
    "ResampleReport",
    "ResampledAnnotation",
    "ResampledImage",
    "Transform",
    "TransformHeader",
    "basis",
    "check_transform_id",
    "cubic_sample",
    "encode_affine",
    "encode_bspline",
    "encode_composite",
    "encode_displacement",
    "encode_identity",
    "folding_fraction",
    "frame_graph",
    "frames_of_timepoint",
    "inside_extent",
    "jacobian_determinant",
    "linear_part",
    "linear_sample",
    "refuse_outside",
    "resample",
    "resolve_between",
    "sample_field",
    "target_registration_error",
    "to_world_vectors",
    "wrap_transform",
]
