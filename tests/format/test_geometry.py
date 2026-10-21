"""Geometry: the affine, the box convention and grid validation (spec §3, §8.1)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from medh5.errors import MEDH5ValidationError
from medh5.geometry import (
    Grid,
    Pyramid,
    apply_affine_to_box,
    box_corners,
    box_to_slices,
    build_affine,
    check_orthonormal,
    check_pyramid,
    decompose_affine,
    derive_level_grid,
    index_to_world,
    is_proper_rotation,
    pyramid_factors,
    slices_to_box,
    world_to_index,
)


def rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def grid(**kwargs):
    defaults = dict(
        grid_id="ct",
        shape=(16, 24, 24),
        axis_names=("z", "y", "x"),
        axis_kinds=("spatial", "spatial", "spatial"),
        spacing=(1.5, 0.8, 0.8),
        origin=(-12.0, -9.6, -9.6),
        direction=np.eye(3),
    )
    defaults.update(kwargs)
    return Grid(**defaults)


class TestAffine:
    def test_S3_3_index_to_world_round_trip(self):
        """§3.3: index -> world -> index is the identity for any oblique grid."""
        rng = np.random.default_rng(0)
        for angle in (0.0, 0.3, 1.1, -2.0):
            affine = build_affine((1.5, 0.8, 0.8), (-12.0, -9.6, -9.6), rotation(angle))
            points = rng.uniform(-5, 20, size=(64, 3))
            back = world_to_index(affine, index_to_world(affine, points))
            assert np.allclose(back, points, atol=1e-9)

    def test_S3_3_integer_index_is_voxel_centre(self):
        """§3.3: integer index k is the *centre* of voxel k, so index 0 is origin."""
        affine = build_affine((2.0, 1.0, 1.0), (10.0, 20.0, 30.0), np.eye(3))
        assert np.allclose(index_to_world(affine, np.zeros(3)), [10.0, 20.0, 30.0])
        assert np.allclose(
            index_to_world(affine, np.array([1.0, 0, 0])), [12.0, 20, 30]
        )

    def test_decompose_inverts_build(self):
        direction = rotation(0.7)
        affine = build_affine((1.5, 0.8, 0.4), (1.0, 2.0, 3.0), direction)
        spacing, origin, recovered = decompose_affine(affine)
        assert np.allclose(spacing, (1.5, 0.8, 0.4))
        assert np.allclose(origin, (1.0, 2.0, 3.0))
        assert np.allclose(recovered, direction)

    def test_S3_2_orthonormality_is_enforced(self):
        """§3.2: `direction` MUST be orthonormal to 1e-4 (E102)."""
        skewed = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(MEDH5ValidationError) as exc:
            check_orthonormal(skewed)
        assert exc.value.code == "E102"
        check_orthonormal(rotation(0.9))

    def test_proper_rotation_rejects_reflection(self):
        reflect = np.diag([1.0, 1.0, -1.0])
        assert not is_proper_rotation(reflect)
        assert is_proper_rotation(rotation(0.4))


class TestBoxConvention:
    def test_S8_1_box_slice_round_trip(self):
        """§8.1: numpy slice a:b <-> lo = a-0.5, hi = b-0.5, exactly."""
        for start, stop in ((0, 1), (12, 40), (3, 3), (7, 100)):
            box = slices_to_box([slice(start, stop)])
            assert box[0].tolist() == [start - 0.5, stop - 0.5]
            assert box_to_slices(box) == (slice(start, stop),)

    def test_S8_1_integer_edge_boxes_keep_their_extent(self):
        """§8.1: half-up rounding, so extent survives a box on integer edges.

        Every other box test starts from ``slices_to_box``, which emits
        half-integer coordinates -- ``lo + 0.5`` is then an exact integer and no
        rounding tie is ever reached.  A box that came from a world->index
        conversion, an even-factor resample or a pyramid level change is
        integer-valued and hits the tie on every axis.  Under half-to-even
        rounding a one-voxel box alternated between empty and two voxels wide
        with the parity of its position.
        """
        for lo in (-0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            box = np.array([[lo, lo + 1.0]] * 3)
            slices = box_to_slices(box)
            voxels = int(np.prod([s.stop - s.start for s in slices]))
            assert voxels == 1, f"box [{lo}, {lo + 1}] covered {voxels} voxels"

    def test_S8_1_extent_is_preserved_for_every_integer_box(self):
        for lo in range(0, 6):
            for width in range(1, 5):
                box = np.array([[float(lo), float(lo + width)]])
                (s,) = box_to_slices(box)
                assert s.stop - s.start == width

    @pytest.mark.parametrize(
        "bounds", [(np.nan, 4.0), (0.0, np.nan), (-np.inf, 4.0), (0.0, np.inf)]
    )
    def test_S8_1_a_bound_that_is_not_finite_is_E406(self, bounds):
        """``lo > hi`` is false for a NaN: a NaN bound became voxel 0 and an
        infinite one a slice to the end of the integers."""
        box = np.array([[0.0, 1.0], bounds])
        for shape in (None, (4, 10)):
            with pytest.raises(MEDH5ValidationError) as exc:
                box_to_slices(box, shape)
            assert exc.value.code == "E406"

    def test_S8_1_an_extent_past_the_integers_clips_nothing(self):
        """An extent past ``2**63 - 1`` wrapped negative, and clamping a
        slice to it panicked."""
        assert box_to_slices(np.array([[-0.5, 3.5]]), (2**63,)) == (slice(0, 4),)

    def test_S8_1_extent_is_voxel_count(self):
        box = slices_to_box([slice(12, 40), slice(0, 3)])
        extents = box[:, 1] - box[:, 0]
        assert extents.tolist() == [28.0, 3.0]

    def test_S8_1_lo_gt_hi_is_rejected(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            box_to_slices(np.array([[5.0, 1.0]]))
        assert exc.value.code == "E406"

    def test_box_to_slices_clips_to_shape(self):
        assert box_to_slices(np.array([[-4.5, 99.5]]), (10,)) == (slice(0, 10),)

    @pytest.mark.parametrize(
        "bad",
        [
            slice(0, None),
            slice(None, None),
            slice(-3, 5),
            slice(2, -1),
            slice(0, 10, 2),
            slice(5, 3),
        ],
    )
    def test_S8_1_a_box_is_made_only_from_a_bounded_block(self, bad):
        """An open stop raised a bare ``TypeError``; a negative bound, counted
        from an end only the array knows, and a step made a box of other
        voxels."""
        with pytest.raises(ValueError, match="0 <= start <= stop") as exc:
            slices_to_box([slice(0, 4), bad])
        assert "axis 1" in str(exc.value)
        # An open start is 0 on any axis, and a step of 1 is no step.
        box = slices_to_box([slice(None, 4), slice(2, 2), slice(0, 4, 1)])
        assert box.tolist() == [[-0.5, 3.5], [1.5, 1.5], [-0.5, 3.5]]

    def test_box_corners_are_odometer_ordered(self):
        corners = box_corners(np.array([[0.0, 1.0], [10.0, 20.0]]))
        assert corners.tolist() == [[0, 10], [0, 20], [1, 10], [1, 20]]

    def test_oblique_box_uses_corners_not_endpoints(self):
        """An axis-aligned index box is an oriented box in world space."""
        affine = build_affine((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), rotation(np.pi / 4))
        box = np.array([[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]])
        bounds = apply_affine_to_box(affine, box)
        naive_lo = index_to_world(affine, box[:, 0])
        assert np.any(bounds[:, 0] < naive_lo - 1e-9)


class TestMisshapenArguments:
    """An affine that is not ``(S+1, S+1)``, or a
    box clipped to a shape of another rank, indexed past what it was given
    and raised ``PanicException`` --- a ``BaseException``, which ``except
    Exception`` does not catch.  Each is a ``ValueError`` now."""

    @pytest.mark.parametrize(
        "affine",
        [
            np.vstack([np.eye(3), np.zeros((1, 3))]),  # 4x3, not degenerate
            np.zeros((0, 0)),
            np.ones((1, 1)),
        ],
        ids=["4x3", "empty", "1x1"],
    )
    def test_S3_3_an_affine_that_is_not_square_is_a_value_error(self, affine):
        for call in (
            lambda: decompose_affine(affine),
            lambda: index_to_world(affine, np.zeros((2, 3))),
            lambda: world_to_index(affine, np.zeros((2, 3))),
            lambda: apply_affine_to_box(affine, np.zeros((3, 2))),
        ):
            with pytest.raises(ValueError) as info:
                call()
            assert type(info.value).__name__ != "PanicException"

    def test_S8_1_a_box_and_a_shape_of_another_rank_are_refused(self):
        box = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="one extent per axis|extents"):
            box_to_slices(box, (4, 4))
        with pytest.raises(ValueError):
            box_to_slices(box[:2], (4, 4, 4))
        with pytest.raises(ValueError):
            apply_affine_to_box(np.eye(4), box[:2])
        with pytest.raises(ValueError):
            box_corners(np.zeros((64, 2)))
        # The controls answer as before.
        assert box_to_slices(box, (4, 4, 4)) == (slice(0, 1),) * 3
        affine = build_affine((2.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3))
        np.testing.assert_array_equal(
            index_to_world(affine, np.array([[1.0, 2.0, 3.0]])), [[2.0, 2.0, 3.0]]
        )
        spacing, _, _ = decompose_affine(affine)
        np.testing.assert_array_equal(spacing, [2.0, 1.0, 1.0])

    def test_S8_1_every_box_helper_refuses_what_box_to_slices_refuses(self):
        """Only ``box_to_slices`` checked a box's shape: ``box_corners`` and
        ``apply_affine_to_box`` flattened a ``(2, 3)`` min/max-corner box and
        answered for three axes of made-up bounds."""
        corners = np.array([[0.0, 0.0, 0.0], [9.0, 19.0, 29.0]])
        helpers = (
            box_to_slices,
            box_corners,
            lambda b: apply_affine_to_box(np.eye(4), b),
        )
        for bad in (corners, corners.ravel(), corners.T[..., None]):
            for helper in helpers:
                with pytest.raises(ValueError, match=r"shape \(S, 2\)"):
                    helper(bad)
        box = corners.T
        assert box_to_slices(box) == (slice(0, 9), slice(0, 19), slice(0, 29))
        np.testing.assert_array_equal(apply_affine_to_box(np.eye(4), box), box)
        found = box_corners(box)
        np.testing.assert_array_equal(found.min(axis=0), corners[0])
        np.testing.assert_array_equal(found.max(axis=0), corners[1])


def test_mm_per_unit_is_the_engines():
    """One table of millimetres per unit (§3.2), the engine's: the converters
    and the MONAI metadata read it, where ``medh5.io.nifti`` kept its own."""
    import medh5.io.nifti
    from medh5 import _core
    from medh5.geometry import KNOWN_UNITS, MM_PER_UNIT

    assert MM_PER_UNIT is _core.MM_PER_UNIT
    assert medh5.io.nifti.MM_PER_UNIT is MM_PER_UNIT
    assert MM_PER_UNIT == {"mm": 1.0, "um": 1e-3, "m": 1000.0}
    assert set(MM_PER_UNIT) == set(KNOWN_UNITS) - {"px"}


class TestGrid:
    def test_S3_1_spatial_axes_must_be_trailing(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            grid(
                axis_kinds=("spatial", "spatial", "channel"), axis_names=("y", "x", "c")
            )
        assert exc.value.code == "E103"

    def test_S3_1_at_most_one_time_and_channel_axis(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            Grid(
                grid_id="g",
                shape=(2, 2, 8, 8),
                axis_names=("t", "u", "y", "x"),
                axis_kinds=("time", "time", "spatial", "spatial"),
                spacing=(1.0, 1.0),
                origin=(0.0, 0.0),
                direction=np.eye(2),
            )
        assert exc.value.code == "E110"

    def test_S3_1_a_grid_does_not_freeze_the_array_it_was_handed(self):
        """`direction` is immutable *on the Grid*, not in the caller's scope.

        `asarray` returns the caller's own array when the dtype and layout
        already match, so marking it read-only in place reached back out of the
        constructor: the caller's next write raised, and every other Grid built
        from that array in the same scope was frozen along with it.
        """
        direction = np.eye(3)
        built = grid(direction=direction)

        direction[0, 0] = -1.0  # the caller still owns their array
        assert built.direction[0, 0] == 1.0, "and the Grid kept its own copy"
        with pytest.raises(ValueError, match="read-only"):
            built.direction[0, 0] = 5.0

    def test_S3_2_spacing_must_be_positive(self):
        with pytest.raises(MEDH5ValidationError) as exc:
            grid(spacing=(1.5, 0.0, 0.8))
        assert exc.value.code == "E104"

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_S3_2_spacing_and_origin_are_finite(self, value):
        """An infinite spacing or a NaN origin was accepted, and put every
        voxel nowhere."""
        with pytest.raises(MEDH5ValidationError) as exc:
            grid(spacing=(1.5, abs(value), 0.8))
        assert exc.value.code == "E104"
        with pytest.raises(MEDH5ValidationError, match="must be finite") as exc:
            grid(origin=(-12.0, value, -9.6))
        assert exc.value.code == "E109"

    def test_S3_2_a_shape_whose_voxels_cannot_be_counted_is_refused(self):
        """Its voxel count wrapped: ``n_voxels`` of this grid was 0."""
        with pytest.raises(MEDH5ValidationError, match="64-bit count") as exc:
            grid(shape=(2**62, 2**62, 2))
        assert exc.value.code == "E109"
        assert grid(shape=(2**20, 2**20, 2**20)).n_voxels == 2**60

    def test_S3_3_extent_spans_half_voxel_outside(self):
        g = grid()
        assert g.extent[:, 0].tolist() == [-0.5, -0.5, -0.5]
        assert g.extent[:, 1].tolist() == [15.5, 23.5, 23.5]

    def test_S3_3_comparable_only_with_shared_frame(self):
        a = grid(grid_id="a", frame_uid="f1")
        b = grid(
            grid_id="b", shape=(8, 12, 12), spacing=(3.0, 1.6, 1.6), frame_uid="f1"
        )
        c = grid(grid_id="c", frame_uid="f2")
        assert a.comparable_with(b)
        assert not a.comparable_with(c)
        assert not a.comparable_with(grid(grid_id="d"))

    def test_S3_3_a_plane_is_not_comparable_with_a_volume(self):
        # `direction` is S×S (§3.2): a 2-D grid's world is a plane of its own,
        # whatever frame it names, and a box on it is not a volume's.
        volume = grid(grid_id="v", frame_uid="f1")
        plane = grid(
            grid_id="p",
            shape=(24, 24),
            axis_names=("y", "x"),
            axis_kinds=("spatial", "spatial"),
            spacing=(0.8, 0.8),
            origin=(-9.6, -9.6),
            direction=np.eye(2),
            frame_uid="f1",
        )
        assert not volume.comparable_with(plane)
        assert not plane.comparable_with(volume)
        assert plane.comparable_with(dataclasses.replace(plane, grid_id="q"))

    def test_time_values_must_match_the_time_axis(self):
        with pytest.raises(MEDH5ValidationError):
            Grid(
                grid_id="g",
                shape=(4, 8, 8),
                axis_names=("t", "y", "x"),
                axis_kinds=("time", "spatial", "spatial"),
                spacing=(1.0, 1.0),
                origin=(0.0, 0.0),
                direction=np.eye(2),
                time_values=(0.0, 1.0),
            )

    def test_grids_hash_and_compare_by_value(self):
        assert grid() == grid()
        assert len({grid(), grid()}) == 1
        assert grid() != grid(spacing=(1.0, 1.0, 1.0))

    def test_S3_2_a_signed_zero_is_the_same_grid(self):
        """Equal grids hash alike: ``-0.0 == 0.0``, so an origin or a direction
        computed with a negated zero names the same grid, and a set or a dict
        keyed by grids holds it once."""
        negated = np.eye(3)
        negated[0, 1] = -0.0
        a = grid(origin=(0.0, 0.0, 0.0))
        b = grid(origin=(-0.0, 0.0, -0.0), direction=negated)
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_is_congruent_tolerates_float_noise(self):
        assert grid().is_congruent(grid(spacing=(1.5 + 1e-12, 0.8, 0.8)))


class TestMultiscale:
    def test_S4_3_half_voxel_shift(self):
        """§4.3: origin' = origin + direction @ (spacing * (f-1)/2)."""
        base = grid(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
        level1 = derive_level_grid(base, (2, 2, 2), "l1")
        assert level1.spacing == (2.0, 2.0, 2.0)
        assert np.allclose(level1.origin, (0.5, 0.5, 0.5))
        assert check_pyramid(base, [base, level1], [[1, 1, 1], [2, 2, 2]]) == []

    def test_S4_3_missing_shift_is_reported(self):
        base = grid(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
        broken = derive_level_grid(base, (2, 2, 2), "l1")
        naive = Grid(
            grid_id="l1",
            shape=broken.shape,
            axis_names=broken.axis_names,
            axis_kinds=broken.axis_kinds,
            spacing=broken.spacing,
            origin=(0.0, 0.0, 0.0),
            direction=broken.direction,
        )
        problems = check_pyramid(base, [base, naive], [[1, 1, 1], [2, 2, 2]])
        assert problems and all("origin" in p for p in problems)

    def test_pyramid_factors_recovers_the_declaration(self):
        base = grid(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
        level1 = derive_level_grid(base, (2, 4, 4), "l1")
        assert np.allclose(
            pyramid_factors(base, [base, level1]), [[1, 1, 1], [2, 4, 4]]
        )

    @pytest.mark.parametrize("factor", [np.nan, np.inf, 0.0, -2.0])
    def test_S4_3_a_factor_that_is_not_finite_and_positive_is_E105(self, factor):
        """``v <= 0`` is false for a NaN: the declaration took a NaN factor
        the validator refuses."""
        with pytest.raises(MEDH5ValidationError, match="finite") as exc:
            Pyramid(
                levels=2,
                downsample_factors=np.array([[1.0, 1.0, 1.0], [2.0, factor, 2.0]]),
                downsample_method="mean",
                grid_levels=("l0", "l1"),
            )
        assert exc.value.code == "E105"

    def test_pyramid_rejects_level0_downsampling(self):
        with pytest.raises(MEDH5ValidationError):
            Pyramid(
                levels=1,
                downsample_factors=np.array([[2.0, 2.0, 2.0]]),
                downsample_method="mean",
                grid_levels=("l0",),
            )


class TestPyramidChecks:
    """§4.3: a level's geometry is derived from level 0, and checked against it."""

    def _base(self):
        return Grid(
            "l0",
            (16, 16, 16),
            ("z", "y", "x"),
            ("spatial",) * 3,
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
            np.eye(3),
        )

    def test_S4_3_a_transposed_level_direction_is_caught(self):
        """The expected origin is computed from the *base* direction.

        So a level whose axes are permuted passed both the spacing and the
        origin check, and `direction` was never compared at all -- exactly the
        failure the module docstring describes, where predictions made at level
        2 are mapped back to level 0 and land transposed.
        """
        base = self._base()
        level = derive_level_grid(base, (2, 2, 2), "l1")
        permuted = dataclasses.replace(
            level,
            direction=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        )
        assert check_pyramid(base, [base, level], [[1, 1, 1], [2, 2, 2]]) == []
        problems = check_pyramid(base, [base, permuted], [[1, 1, 1], [2, 2, 2]])
        assert any("direction" in p for p in problems)

    @pytest.mark.parametrize(
        ("spacing", "origin", "direction"),
        [
            ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), np.eye(3)),
            # Whole-body CT, 1600 voxels from zero: the tolerance scaled with
            # |origin| and swallowed the missing shift.
            ((0.5, 0.5, 0.5), (-800.0, 0.0, 0.0), np.eye(3)),
            # A µm stage, its axes permuted and anisotropic: a world axis was
            # weighed by another axis's spacing.
            (
                (0.25, 2.0, 1.0),
                (5e4, -2e4, 3e4),
                np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            ),
        ],
    )
    def test_S4_3_the_origin_is_held_to_a_level_voxel_wherever_it_lies(
        self, spacing, origin, direction
    ):
        """A level's origin is accepted within 1e-3 of the level's voxel of
        the derived one along each of its axes, and nowhere else --- however
        far from zero the grid sits."""
        base = dataclasses.replace(
            self._base(), spacing=spacing, origin=origin, direction=direction
        )
        level = derive_level_grid(base, (2, 2, 2), "l1")
        factors = [[1, 1, 1], [2, 2, 2]]
        assert check_pyramid(base, [base, level], factors) == []
        unshifted = dataclasses.replace(level, origin=base.origin)
        problems = check_pyramid(base, [base, unshifted], factors)
        assert len(problems) == 1 and "half-voxel shift missing" in problems[0]
        for axis in range(3):
            for voxels, ok in ((0.5e-3, True), (-0.5e-3, True), (2e-3, False)):
                offset = np.zeros(3)
                offset[axis] = voxels
                nudged = dataclasses.replace(
                    level, origin=tuple(level.index_to_world(offset))
                )
                found = check_pyramid(base, [base, nudged], factors)
                assert (found == []) is ok, (axis, voxels, found)

    def test_S4_3_a_level_extent_that_is_not_a_resampling_is_caught(self):
        base = self._base()
        level = derive_level_grid(base, (2, 2, 2), "l1")
        stunted = dataclasses.replace(level, shape=(3, 3, 3))
        problems = check_pyramid(base, [base, stunted], [[1, 1, 1], [2, 2, 2]])
        assert any("resampling" in p for p in problems)

    def test_S4_3_both_rounding_conventions_are_accepted(self):
        """Only the spacing/origin relation is normative, so 15/2 may be 7 or 8."""
        odd = Grid(
            "l0",
            (15, 15, 15),
            ("z", "y", "x"),
            ("spatial",) * 3,
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
            np.eye(3),
        )
        ceiled = derive_level_grid(odd, (2, 2, 2), "l1")
        floored = dataclasses.replace(ceiled, shape=(7, 7, 7))
        assert check_pyramid(odd, [odd, ceiled], [[1, 1, 1], [2, 2, 2]]) == []
        assert check_pyramid(odd, [odd, floored], [[1, 1, 1], [2, 2, 2]]) == []
