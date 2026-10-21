"""An annotation reads back as written, or is refused with a code (spec §6--§9).

Every reader of a per-object annotation indexes its columns by row, and every
voxel read is a window of the annotation's grid.  A file whose columns disagree
--- planted here with ``h5py``, the way a damaged or foreign file arrives ---
is refused when it is opened or read, with the code the validator reports for
the same file, and the validator reports it as a diagnostic rather than
abandoning the file.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.annotations.base import AnnotationPayload, Instance
from medh5.annotations.geometric import Polygon
from medh5.annotations.voxel import (
    InstanceInput,
    encode_instances,
    payload_to_masks,
    transcode_payload,
)
from medh5.errors import MEDH5ValidationError
from medh5.labels import LabelClass, LabelSet
from medh5.validate import validate_file

SHAPE = (8, 12, 12)


def _cube(lo: int, hi: int) -> np.ndarray:
    mask = np.zeros(SHAPE, bool)
    mask[lo:hi, lo:hi, lo:hi] = True
    return mask


def _objects(path: Path) -> Path:
    """One annotation of each per-object kind, two objects apiece."""
    with medh5.create(path, sample_id="s", codec="portable") as w:
        w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), frame_uid="pseudo:f")
        w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        w.add_segmentation(
            "inst",
            grid="g",
            instances=[
                InstanceInput(class_id=1, instance_id=7, mask=_cube(1, 3)),
                InstanceInput(class_id=2, instance_id=8, mask=_cube(4, 7)),
            ],
        )
        w.add_boxes(
            "bx",
            np.array([[[0, 2], [0, 2], [0, 2]], [[3, 5], [3, 5], [3, 5]]], float),
            class_ids=[1, 2],
            scores=[0.5, 0.6],
            grid="g",
            space="index",
        )
        w.add_obb(
            "ob",
            np.zeros((2, 3)),
            np.ones((2, 3)),
            np.stack([np.eye(3)] * 2),
            class_ids=[1, 2],
            grid="g",
            space="index",
        )
        w.add_contours(
            "ct",
            [
                Polygon(np.zeros((3, 3)), 1, (0, 1)),
                Polygon(np.ones((4, 3)), 2, (0, 2)),
            ],
            grid="g",
        )
        w.add_classification("cl", {1: 1.0, 2: 0.0}, scope="sample")
    return path


@pytest.fixture(scope="module")
def objects(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _objects(tmp_path_factory.mktemp("objects") / "objects.medh5")


def _replace(path: Path, dataset: str, data: Any) -> None:
    """Swap a stored dataset for *data*, keeping its attributes."""
    with h5py.File(path, "r+") as f:
        attrs = dict(f[dataset].attrs)
        del f[dataset]
        f[dataset] = data
        f[dataset].attrs.update(attrs)


def _read_everything(annotation: Any) -> None:
    """Every read a per-object annotation answers."""
    kind = annotation.kind
    if kind == "instances":
        annotation.dense()
        list(annotation.instances())
        annotation.crop(0)
    elif kind == "boxes":
        list(annotation)
        annotation.as_slices()
    elif kind == "obb":
        annotation.corners()
        annotation.as_aabb()
    elif kind == "contours":
        list(annotation.polygons())
    elif kind == "classification":
        list(annotation.assertions())
        annotation.by_scope_id()


def _codes(path: Path) -> list[str]:
    return sorted({d.code for d in validate_file(path).errors})


class TestColumnsAgree:
    """§7.4, §8 and §9: every per-object column has one row per object."""

    def test_the_fixture_reads_and_validates(self, objects: Path) -> None:
        assert _codes(objects) == []
        with medh5.open(objects) as sample:
            for name in ("inst", "bx", "ob", "ct", "cl"):
                _read_everything(sample.annotations[name])

    @pytest.mark.parametrize(
        ("name", "dataset", "data"),
        [
            ("bx", "class_ids", np.array([1], np.uint16)),
            ("bx", "scores", np.array([0.5], np.float32)),
            ("bx", "boxes", np.zeros((2, 3, 3), np.float32)),
            ("bx", "boxes", np.float32(3.0)),
            ("inst", "class_ids", np.array([1, 2, 2], np.uint16)),
            ("inst", "instance_ids", np.array([7], np.uint32)),
            ("inst", "mask_shapes", np.array([[2, 2, 2]], np.int32)),
            ("inst", "mask_offsets", np.array([0, 1], np.int64)),
            ("ob", "sizes", np.ones((1, 3), np.float32)),
            ("ob", "rotations", np.eye(2, dtype=np.float32)[None]),
            ("ct", "contour_class_ids", np.array([1], np.uint16)),
            ("cl", "values", np.array([1.0], np.float32)),
        ],
        ids=lambda v: v if isinstance(v, str) else None,
    )
    def test_S7_4_S8_S9_a_column_out_of_step_is_E405_to_reader_and_validator(
        self, objects: Path, tmp_path: Path, name: str, dataset: str, data: Any
    ) -> None:
        """Read past its end, a short column panicked the reader; the validator
        passed most of these files, and abandoned the rest with E001."""
        path = tmp_path / "planted.medh5"
        shutil.copyfile(objects, path)
        _replace(path, f"annotations/{name}/{dataset}", data)
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError) as exc:
            _read_everything(sample.annotations[name])
        assert exc.value.code == "E405"
        assert _codes(path) == ["E405"]

    def test_S8_1_index_boxes_have_the_grid_rank(
        self, objects: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "planted.medh5"
        shutil.copyfile(objects, path)
        _replace(path, "annotations/bx/boxes", np.zeros((2, 2, 2), np.float32))
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError) as exc:
            _read_everything(sample.annotations["bx"])
        assert exc.value.code == "E405"
        assert _codes(path) == ["E405"]

    def test_S7_4_a_column_out_of_step_does_not_stop_the_validator(
        self, objects: Path, tmp_path: Path
    ) -> None:
        """The instances rule raised on the mismatch and the driver reported one
        E001 for the file, dropping every diagnostic of the annotations after."""
        path = tmp_path / "planted.medh5"
        shutil.copyfile(objects, path)
        _replace(path, "annotations/inst/instance_ids", np.arange(6, dtype=np.uint32))
        _replace(path, "annotations/cl/values", np.array([np.nan, 0.0], np.float32))
        report = validate_file(path)
        found = {(d.code, d.location) for d in report.errors}
        assert ("E405", "/annotations/inst/instance_ids") in found
        assert ("E404", "/annotations/cl") in found
        assert "E001" not in {code for code, _ in found}

    def test_S8_2_slice_index_beside_boxes_of_the_wrong_shape_is_one_E405(
        self, objects: Path, tmp_path: Path
    ) -> None:
        """The plane check read a box's fourth axis out of a three-entry grid
        shape, and the validator panicked."""
        path = tmp_path / "planted.medh5"
        shutil.copyfile(objects, path)
        _replace(path, "annotations/bx/boxes", np.zeros((2, 3, 3), np.float32))
        with h5py.File(path, "r+") as f:
            f["annotations/bx"].create_dataset(
                "slice_index", data=np.array([0, 0], np.int32)
            )
        found = {(d.code, d.location) for d in validate_file(path).errors}
        assert found == {("E405", "/annotations/bx/boxes")}

    def test_S9_a_value_that_is_not_a_probability_is_E404(
        self, objects: Path, tmp_path: Path
    ) -> None:
        """NaN fails every comparison, so a min/max range test let it through."""
        path = tmp_path / "planted.medh5"
        shutil.copyfile(objects, path)
        _replace(path, "annotations/cl/values", np.array([np.nan, 0.0], np.float32))
        assert _codes(path) == ["E404"]


class TestInstanceMasks:
    """§7.4: an object's mask is cropped to its box, and no larger than its grid."""

    def test_S7_4_a_mask_of_another_extent_than_its_box_is_E405(
        self, objects: Path, tmp_path: Path
    ) -> None:
        path = tmp_path / "planted.medh5"
        shutil.copyfile(objects, path)
        _replace(
            path,
            "annotations/inst/mask_shapes",
            np.array([[2, 2, 4], [3, 3, 3]], np.int32),
        )
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError) as exc:
            sample.annotations["inst"].dense()
        assert exc.value.code == "E405"
        assert _codes(path) == ["E405"]

    def test_S7_4_a_box_past_the_integers_counts_exactly(self) -> None:
        """A box-only object's voxels were NumPy's int64 product of its
        extents: an axis from one end of the integers to the other is
        ``2**64 - 1`` voxels, which overflowed it."""
        box = np.array([[-3e38, 3e38]] * 3, np.float32)
        wide = Instance(index=0, instance_id=1, class_id=1, box=box)
        assert wide.voxel_count == (2**64 - 1) ** 3

    def test_S7_4_a_mask_larger_than_the_grid_is_refused_before_it_is_allocated(
        self, objects: Path, tmp_path: Path
    ) -> None:
        """A planted shape of 2**50 voxels aborted the process allocating it."""
        path = tmp_path / "planted.medh5"
        shutil.copyfile(objects, path)
        _replace(
            path,
            "annotations/inst/mask_shapes",
            np.array([[1 << 20, 1 << 20, 1 << 10], [3, 3, 3]], np.int64),
        )
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError) as exc:
            sample.annotations["inst"].crop(0)
        assert exc.value.code == "E405"
        assert "E405" in _codes(path)

    # The fixture's two masks pack into 1 byte (2x2x2) and 4 bytes (3x3x3):
    # offsets [0, 1, 5] over 5 bytes.
    OFFSETS = {
        "falling": [0, 5, 1],
        "past the data": [0, 1, 4096],
        "a span too long": [0, 3, 5],
        "a span too short": [0, 1, 3],
    }

    @pytest.mark.parametrize("offsets", OFFSETS.values(), ids=OFFSETS.keys())
    def test_S7_4_offsets_that_do_not_delimit_each_packed_mask_are_E408(
        self, objects: Path, tmp_path: Path, offsets: list[int]
    ) -> None:
        """Falling offsets panicked slicing the data; a span short of its mask
        was padded with `false`, and the object read back as part of itself."""
        path = tmp_path / "planted.medh5"
        shutil.copyfile(objects, path)
        _replace(path, "annotations/inst/mask_offsets", np.array(offsets, np.uint64))
        with medh5.open(path) as sample:
            annotation = sample.annotations["inst"]
            for read in (annotation.dense, lambda: annotation.crop(1)):
                with pytest.raises(MEDH5ValidationError) as exc:
                    read()
                assert exc.value.code == "E408"
        for level in ("structural", "semantic"):
            found = {
                (d.code, d.location) for d in validate_file(path, level=level).errors
            }
            assert found == {("E408", "/annotations/inst/mask_offsets")}, level

    @pytest.mark.parametrize("offsets", OFFSETS.values(), ids=OFFSETS.keys())
    def test_S7_4_a_payload_is_held_to_the_same_offsets(
        self, offsets: list[int]
    ) -> None:
        """`payload_to_masks`, and the transcode built on it, decode a payload
        a converter assembled exactly as a reader decodes a file."""
        stored = encode_instances(
            [
                InstanceInput(class_id=1, instance_id=7, mask=_cube(1, 3)),
                InstanceInput(class_id=2, instance_id=8, mask=_cube(4, 7)),
            ],
            SHAPE,
        )
        assert stored.datasets["mask_offsets"].tolist() == [0, 1, 5]
        datasets = dict(stored.datasets)
        datasets["mask_offsets"] = np.array(offsets, np.uint64)
        planted = AnnotationPayload(
            "instances",
            datasets=datasets,
            attrs=dict(stored.attrs),
            class_ids=stored.class_ids,
        )
        for decode in (
            lambda: payload_to_masks(planted, spatial_shape=SHAPE),
            lambda: transcode_payload(
                planted, "layers", spatial_shape=SHAPE, drop_identity=True
            ),
        ):
            with pytest.raises(MEDH5ValidationError) as exc:
                decode()
            assert exc.value.code == "E408"

    def test_S7_4_the_writer_refuses_a_crop_that_is_not_its_box(self) -> None:
        """A crop larger than its box was stored and read back truncated; a
        smaller one failed only at `dense()`, and the validator passed both."""
        box = np.array([[0.5, 3.5]] * 3, np.float32)  # voxels 1..3: 3 a side
        for crop in (
            np.ones((4, 3, 3), bool),
            np.ones((2, 3, 3), bool),
            np.ones((3, 3), bool),
        ):
            with pytest.raises(MEDH5ValidationError) as exc:
                encode_instances(
                    [InstanceInput(class_id=1, instance_id=1, box=box, crop=crop)],
                    SHAPE,
                )
            assert exc.value.code == "E405"
        stored = encode_instances(
            [
                InstanceInput(
                    class_id=1, instance_id=1, box=box, crop=np.ones((3, 3, 3), bool)
                )
            ],
            SHAPE,
        )
        assert stored.datasets["mask_shapes"].tolist() == [[3, 3, 3]]


class TestCoverage:
    """§6.2: `class_ids` holds every class the annotation can express."""

    def test_S6_2_an_objects_class_is_declared_beside_the_classes_examined(
        self, tmp_path: Path
    ) -> None:
        """Naming the classes examined replaced the objects' own: class 2 went
        missing from `class_ids` while `contains()` still answered for it."""
        path = tmp_path / "inst.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "inst",
                grid="g",
                instances=[InstanceInput(class_id=2, instance_id=7, mask=_cube(1, 3))],
                annotated_classes=[1],
            )
        with medh5.open(path) as sample:
            annotation = sample.annotations["inst"]
            assert annotation.class_ids == (1, 2)
            assert annotation.annotated_class_ids == (1,)
            assert annotation.contains(2, (1, 1, 1))
        payload = encode_instances(
            [InstanceInput(class_id=2, instance_id=7, mask=_cube(1, 3))],
            SHAPE,
            class_ids=[1],
        )
        assert payload.class_ids == (1, 2)

    @pytest.mark.parametrize(
        ("name", "column"),
        [("inst", "class_ids"), ("bx", "class_ids"), ("ct", "contour_class_ids")],
    )
    def test_S6_2_an_object_of_an_undeclared_class_is_E404(
        self, objects: Path, tmp_path: Path, name: str, column: str
    ) -> None:
        path = tmp_path / "planted.medh5"
        shutil.copyfile(objects, path)
        with h5py.File(path, "r+") as f:
            f[f"annotations/{name}"].attrs["class_ids"] = np.array([1], np.uint16)
            f[f"annotations/{name}"].attrs["annotated_class_ids"] = np.array(
                [1], np.uint16
            )
        found = {(d.code, d.location) for d in validate_file(path).errors}
        assert found == {("E404", f"/annotations/{name}/{column}")}


def _ignored(path: Path, encoding: str, ignore: np.ndarray) -> None:
    """Classes 1 and 300 in one annotation of *encoding*, with an ignore region."""
    labels = LabelSet(
        "v",
        version="1.0.0",
        classes=[LabelClass(1, "a", "A"), LabelClass(300, "b", "B")],
    )
    first = np.zeros(SHAPE, bool)
    first[0:2] = True
    second = np.zeros(SHAPE, bool)
    second[3:4] = True
    with medh5.create(path, sample_id="s", codec="portable") as w:
        w.label_set(labels)
        w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
        w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        common: dict[str, Any] = {
            "grid": "g",
            "annotated_classes": [1, 300],
            "ignore": ignore,
        }
        if encoding == "instances":
            w.add_segmentation(
                "seg",
                instances=[
                    InstanceInput(class_id=1, instance_id=1, mask=first),
                    InstanceInput(class_id=300, instance_id=2, mask=second),
                ],
                **common,
            )
        elif encoding == "probmap":
            w.add_segmentation(
                "seg",
                probabilities={1: first.astype(float), 300: second.astype(float)},
                **common,
            )
        else:
            w.add_segmentation(
                "seg", masks={1: first, 300: second}, encoding=encoding, **common
            )


ENCODINGS = ("labelmap", "layers", "bitmask", "instances", "probmap")


class TestFlatten:
    """§7.7: `labelmap()` keeps the ignore region, in a dtype that holds it."""

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_S7_7_an_ignored_voxel_reads_as_the_ignore_id_under_every_encoding(
        self, tmp_path: Path, encoding: str
    ) -> None:
        """Only the `labelmap` kind kept it: the other four flattened an ignored
        voxel to 0, which means verified absent."""
        ignore = np.zeros(SHAPE, bool)
        ignore[6] = True
        path = tmp_path / f"{encoding}.medh5"
        _ignored(path, encoding, ignore)
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            assert annotation.kind == encoding
            volume = annotation.labelmap()
            assert np.array_equal(volume == annotation.ignore_id, ignore)
            assert sorted(np.unique(volume).tolist()) == [0, 1, 300, 65535]
            window = annotation.labelmap(roi=(slice(5, 8), slice(0, 4), slice(2, 6)))
            assert np.array_equal(window, volume[5:8, 0:4, 2:6])
            assert np.array_equal(
                volume == annotation.ignore_id, sample.ignore_region("seg")
            )

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_S7_7_a_dtype_that_cannot_hold_a_value_is_refused(
        self, tmp_path: Path, encoding: str
    ) -> None:
        """Saturating, `uint8` read class 300 and the ignore id both as 255."""
        ignore = np.zeros(SHAPE, bool)
        ignore[6] = True
        path = tmp_path / f"{encoding}.medh5"
        _ignored(path, encoding, ignore)
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            for dtype in (np.uint8, np.int8, bool):
                with pytest.raises(ValueError):
                    annotation.labelmap(dtype=dtype)
            assert annotation.labelmap(dtype=np.int32).max() == 65535

    def test_S7_7_a_narrow_dtype_is_accepted_when_every_value_fits(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "small.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "seg",
                grid="g",
                masks={1: _cube(0, 2), 2: _cube(1, 4)},
                encoding="layers",
            )
        with medh5.open(path) as sample, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            annotation = sample.annotations["seg"]
            wide = annotation.labelmap()
            for dtype in (np.uint8, np.int8, np.float16):
                narrow = annotation.labelmap(dtype=dtype)
                assert narrow.dtype == dtype
                assert np.array_equal(narrow.astype(np.uint16), wide)


class TestRegionOfInterest:
    """§7.6: a region of interest is a window of the grid, read whole."""

    def test_S7_6_a_step_is_refused_rather_than_dropped(self, tmp_path: Path) -> None:
        """`slice(0, 4, 2)` read four planes where `Image.read` reads two."""
        ignore = np.zeros(SHAPE, bool)
        ignore[6] = True
        path = tmp_path / "layers.medh5"
        _ignored(path, "layers", ignore)
        stepped = (slice(0, 4, 2), slice(None), slice(None))
        with medh5.open(path) as sample:
            annotation = sample.annotations["seg"]
            for read in (
                lambda: annotation.dense(roi=stepped),
                lambda: annotation.labelmap(roi=stepped),
                lambda: annotation.ignore_mask(roi=stepped),
                lambda: sample.ignore_region("seg", roi=stepped),
            ):
                with pytest.raises(ValueError, match="step 2"):
                    read()
            unit = (slice(0, 4, 1), slice(None), slice(None))
            assert annotation.dense(roi=unit).shape == (2, 4, 12, 12)
            assert sample.images["CT"].read(stepped).shape == (2, 12, 12)


class TestVoxelShape:
    """§7: a voxel annotation covers its grid exactly."""

    @pytest.mark.parametrize("encoding", ["labelmap", "layers", "bitmask", "probmap"])
    def test_S7_a_volume_of_another_shape_than_its_grid_is_E405(
        self, tmp_path: Path, encoding: str
    ) -> None:
        """Every read is a window of the grid; the reader panicked broadcasting
        the stored volume into one."""
        path = tmp_path / f"{encoding}.medh5"
        _ignored(path, encoding, np.zeros(SHAPE, bool))
        with h5py.File(path, "r+") as f:
            f["grids/g"].attrs["shape"] = np.array([8, 12, 13])
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError) as exc:
            sample.annotations["seg"].dense()
        assert exc.value.code == "E405"
        assert "E405" in _codes(path)

    def test_S7_5_a_probmap_has_one_plane_per_class(self, tmp_path: Path) -> None:
        path = tmp_path / "probmap.medh5"
        _ignored(path, "probmap", np.zeros(SHAPE, bool))
        with h5py.File(path, "r+") as f:
            f["annotations/seg"].attrs["class_ids"] = np.array([1, 2, 300], np.uint16)
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError) as exc:
            sample.annotations["seg"].dense()
        assert exc.value.code == "E405"
        assert "E405" in _codes(path)


def _segmentation(
    path: Path,
    encoding: str,
    masks: dict[int, np.ndarray],
    *,
    label_set: LabelSet | None = None,
    **kwargs: Any,
) -> Path:
    """One `seg` annotation of *masks* under *encoding*, written by the writer."""
    with medh5.create(path, sample_id="s", codec="portable") as w:
        if label_set is not None:
            w.label_set(label_set)
        w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
        w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
        common: dict[str, Any] = {
            "grid": "g",
            "annotated_classes": sorted(masks),
            **kwargs,
        }
        if encoding == "instances":
            objects = [
                InstanceInput(class_id=c, instance_id=i + 1, mask=m)
                for i, (c, m) in enumerate(masks.items())
            ]
            w.add_segmentation("seg", instances=objects, **common)
        elif encoding == "probmap":
            soft = {c: m.astype(float) for c, m in masks.items()}
            w.add_segmentation("seg", probabilities=soft, **common)
        else:
            w.add_segmentation("seg", masks=masks, encoding=encoding, **common)
    return path


HIERARCHY = LabelSet(
    "closure-v1",
    classes=[
        LabelClass(1, "kidney", "Kidney"),
        LabelClass(2, "left_kidney", "Left kidney", parents=[1]),
        LabelClass(3, "spleen", "Spleen"),
    ],
)


class TestImplicitVoxelClosure:
    """§5.4: writers SHOULD use `explicit` for voxel annotations; one
    declaring `implicit` is W918, since every read of an ancestor then
    includes its descendants' voxels."""

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_S5_4_a_voxel_annotation_declaring_implicit_is_W918(
        self, tmp_path, encoding
    ):
        masks = {1: _cube(0, 2), 2: _cube(3, 5)}
        implicit = _segmentation(
            tmp_path / "i.medh5",
            encoding,
            masks,
            label_set=HIERARCHY,
            closure="implicit",
        )
        found = [d for d in validate_file(implicit).diagnostics if d.code == "W918"]
        assert [d.location for d in found] == ["/annotations/seg"]
        assert "W918" not in validate_file(implicit, level="structural").codes
        explicit = _segmentation(
            tmp_path / "e.medh5", encoding, masks, label_set=HIERARCHY
        )
        assert "W918" not in validate_file(explicit).codes

    def test_S5_4_an_implicit_classification_is_not_W918(self, tmp_path):
        path = tmp_path / "c.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.label_set(HIERARCHY)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.add_classification("dx", {"left_kidney": 1.0}, closure="implicit")
        assert "W918" not in validate_file(path).codes


class TestClosure:
    """§5.4: under `implicit` a class holds wherever a descendant does, in
    every encoding; under `explicit` nothing is inferred.  The closure was
    stored and never applied, so `dense("kidney")` missed the left kidney."""

    @staticmethod
    def _masks() -> dict[int, np.ndarray]:
        return {1: _cube(0, 2), 2: _cube(3, 5), 3: _cube(6, 8)}

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_S5_4_implicit_closure_reaches_every_voxel_reader(self, tmp_path, encoding):
        masks = self._masks()
        path = _segmentation(
            tmp_path / "c.medh5",
            encoding,
            masks,
            label_set=HIERARCHY,
            closure="implicit",
        )
        kidney = masks[1] | masks[2]
        with medh5.open(path) as sample:
            ann = sample.annotations["seg"]
            assert np.array_equal(ann.dense(["kidney"])[0], kidney)
            assert np.array_equal(ann.dense(["left_kidney"])[0], masks[2])
            assert ann.contains("kidney", (4, 4, 4))
            assert ann.voxel_counts()[1] == int(kidney.sum())
            found = np.nonzero(kidney)
            expected = [[v.min() - 0.5, v.max() + 0.5] for v in found]
            assert np.array_equal(ann.class_bboxes()[1], expected)
            # A labelmap holds what is stored, one class per voxel.
            assert ann.labelmap()[4, 4, 4] == 2

    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_S5_4_explicit_closure_infers_nothing(self, tmp_path, encoding):
        masks = self._masks()
        path = _segmentation(tmp_path / "e.medh5", encoding, masks, label_set=HIERARCHY)
        with medh5.open(path) as sample:
            ann = sample.annotations["seg"]
            assert np.array_equal(ann.dense(["kidney"])[0], masks[1])
            assert not ann.contains("kidney", (4, 4, 4))
            assert ann.voxel_counts()[1] == int(masks[1].sum())

    def _classification(self, path: Path, closure: str) -> Path:
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.label_set(HIERARCHY)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.add_classification(
                "dx",
                [("left_kidney", 1.0, 7), ("spleen", 1.0, 8)],
                scope="instance",
                annotated_classes="all",
                closure=closure,
            )
        return path

    def test_S9_an_implicit_ancestor_of_an_asserted_class_is_positive(self, tmp_path):
        path = self._classification(tmp_path / "dx.medh5", "implicit")
        with medh5.open(path) as sample:
            dx = sample.annotations["dx"]
            assert dx.state("kidney", scope_id=7) == "positive"
            # Per scope unit: nothing in unit 8 implies a kidney.
            assert dx.state("kidney", scope_id=8) == "negative"
            # The record stays literal.
            assert dx.value("kidney", scope_id=7) is None
            assert 1 not in dx.asserted_class_ids.tolist()

    def test_S9_explicit_closure_keeps_the_ancestor_negative(self, tmp_path):
        path = self._classification(tmp_path / "dx.medh5", "explicit")
        with medh5.open(path) as sample:
            assert sample.annotations["dx"].state("kidney", scope_id=7) == "negative"


class TestReservedIds:
    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_S6_2_background_is_no_class(self, tmp_path, encoding):
        """`dense([0])` answered every unlabelled voxel under `labelmap` and
        nothing under the other encodings."""
        path = _segmentation(tmp_path / "z.medh5", encoding, {1: _cube(0, 2)})
        with medh5.open(path) as sample:
            ann = sample.annotations["seg"]
            for read in (lambda: ann.dense([0]), lambda: ann.contains(0, (0, 0, 0))):
                with pytest.raises(MEDH5ValidationError) as exc:
                    read()
                assert exc.value.code == "E404"

    @pytest.mark.parametrize("planted", [0, 70000, 1])
    def test_S6_2_a_stored_ignore_id_is_range_checked_at_open(self, tmp_path, planted):
        path = _segmentation(tmp_path / "i.medh5", "labelmap", {1: _cube(0, 2)})
        with h5py.File(path, "r+") as f:
            f["annotations/seg"].attrs["ignore_id"] = planted
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError) as exc:
            sample.annotations["seg"].dense()
        assert exc.value.code == "E404"

    def test_S7_2_a_layer_table_short_of_the_planes_is_refused_at_open(self, tmp_path):
        masks = {1: _cube(0, 4), 2: _cube(2, 6)}
        path = _segmentation(tmp_path / "l.medh5", "layers", masks)
        with h5py.File(path, "r+") as f:
            table = f["annotations/seg/layer_class_ids"][...]
            assert table.shape[0] == 2
        _replace(path, "annotations/seg/layer_class_ids", table[:1])
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError) as exc:
            sample.annotations["seg"].dense()
        assert exc.value.code == "E405"

    def test_S7_3_a_repeated_bit_class_id_is_refused_at_open(self, tmp_path):
        masks = {1: _cube(0, 4), 2: _cube(2, 6)}
        path = _segmentation(tmp_path / "b.medh5", "bitmask", masks)
        _replace(path, "annotations/seg/bit_class_ids", np.array([1, 1], np.uint16))
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError) as exc:
            sample.annotations["seg"].dense()
        assert exc.value.code == "E404"

    def test_S7_7_an_empty_sibling_mask_is_no_ignore_region(self, tmp_path):
        """`has_ignore_region` said yes where `ignore_region` found nothing."""
        ignore = _cube(0, 3)
        masks = {1: _cube(0, 2)}  # overlaps: the region goes to a sibling mask
        path = _segmentation(tmp_path / "s.medh5", "bitmask", masks, ignore=ignore)
        with medh5.open(path) as sample:
            sibling = sample.annotations["seg"].header.ignore_mask
            assert sibling and sample.annotations["seg"].has_ignore_region
        with h5py.File(path, "r+") as f:
            f[f"annotations/{sibling}/data"][...] = 0
        with medh5.open(path) as sample:
            ann = sample.annotations["seg"]
            assert not ann.has_ignore_region
            assert not sample.ignore_region("seg").any()
