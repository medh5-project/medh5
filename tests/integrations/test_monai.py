"""The MONAI bridge (``medh5.monai``): what MONAI receives, and the geometry and
metadata computed without it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5ValidationError
from medh5.labels import LabelClass, LabelSet
from tests.helpers import SHAPE
from tests.kits import Hierarchy


def _frameless(path: Path, *, seed: int = 0, images: tuple[str, ...] = ("CT",)) -> Path:
    """A sample as the NIfTI and nnU-Net importers write one: no frame of
    reference, no value units, a labelmap with an ignore band."""
    rng = np.random.default_rng(seed)
    organs = np.zeros(Hierarchy.SHAPE, dtype=np.uint16)
    organs[2:6, 2:6, 2:6] = 1
    ignore = np.zeros(Hierarchy.SHAPE, dtype=bool)
    ignore[-1] = True
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.label_set(
            LabelSet("v", version="1.0.0", classes=[LabelClass(1, "liver", "Liver")])
        )
        w.add_grid("g", shape=Hierarchy.SHAPE, spacing=(2.0, 1.0, 1.0))
        for image_id in images:
            w.add_image(
                image_id,
                rng.integers(0, 100, Hierarchy.SHAPE).astype(np.int16),
                grid="g",
                modality="CT",
            )
        w.add_segmentation(
            "organs",
            grid="g",
            masks={1: organs == 1},
            encoding="labelmap",
            ignore=ignore,
        )
    return path


class TestMonaiLabelDtype:
    """`to_dict` labels are `int64`: the full id range, and the ignore id intact."""

    def test_S5_3_ids_beyond_int16_and_the_ignore_id_survive(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("monai")
        from medh5.monai import to_dict

        wide = LabelClass(40000, "wide", "A class beyond int16")
        path = tmp_path / "wide.medh5"
        ignore = np.zeros(Hierarchy.SHAPE, dtype=bool)
        ignore[-1] = True
        with medh5.create(path, sample_id="wide") as w:
            w.label_set(Hierarchy.label_set(wide))
            w.add_grid(
                "g", shape=Hierarchy.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image("CT", Hierarchy.image(), grid="g", modality="CT")
            w.add_segmentation(
                "organs",
                grid="g",
                masks={40000: Hierarchy.mask()},
                encoding="labelmap",
                ignore=ignore,
            )
        with medh5.open(path) as sample:
            item = to_dict(sample, ["CT"], ["organs"])
        label = item["organs"]
        assert str(label.dtype) == "torch.int64"
        values = set(np.unique(label.numpy()).tolist())
        assert 40000 in values
        assert 65535 in values
        assert -1 not in values


class TestMonai:
    def test_available_reports_whether_monai_imports(self):
        """Both branches are real; neither may be excluded from coverage.

        This carried a blanket `pragma: no cover`, including on the `return
        True` that runs wherever the extra is installed -- the same shape as
        the untested-because-skipped problem the MONAI CI job was added for.
        """
        from medh5.monai import available

        pytest.importorskip("monai")
        assert available() is True

    def test_an_annotation_carries_its_own_grids_affine_not_an_images(self, tmp_path):
        """`to_dict` binds each label tensor to the grid it was drawn on.

        A sample holding CT and PET on different grids is the case this format
        exists for.  Taking the affine from the first *image* handed the
        annotation geometry belonging to something else --- here a 70 mm origin
        shift and a 2x spacing error --- so `Spacingd` resampled the label
        against the wrong frame and the ground truth moved.
        """
        pytest.importorskip("monai")
        from medh5.monai import affine_for, to_dict

        ct = np.zeros((16, 32, 32), np.int16)
        pet = np.zeros((8, 16, 16), np.int16)
        seg = np.zeros((8, 16, 16), bool)
        seg[2:6, 4:12, 4:12] = True
        path = tmp_path / "twogrid.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(
                LabelSet(
                    "v", version="1.0.0", classes=[LabelClass(1, "tumour", "Tumour")]
                )
            )
            w.add_timepoint("tp0")
            w.add_grid(
                "gct",
                shape=ct.shape,
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                timepoint="tp0",
            )
            w.add_grid(
                "gpet",
                shape=pet.shape,
                spacing=(2.0, 2.0, 2.0),
                origin=(50.0, 60.0, 70.0),
                timepoint="tp0",
            )
            w.add_image("CT", ct, grid="gct", modality="CT")
            w.add_image("PET", pet, grid="gpet", modality="PT")
            w.add_segmentation(
                "tumour",
                grid="gpet",
                masks={"tumour": seg},
                annotated_classes=["tumour"],
            )

        with medh5.open(path) as sample:
            item = to_dict(sample, images=["CT", "PET"], annotations=["tumour"])
            got = np.asarray(item["tumour"].affine)
            assert np.allclose(got, affine_for(sample, "PET"))
            assert not np.allclose(got, affine_for(sample, "CT"))
            assert item["tumour"].meta["medh5"]["grid_id"] == "gpet"
            # No image requested must not mean an invented identity affine.
            only = to_dict(sample, images=[], annotations=["tumour"])
            assert np.allclose(
                np.asarray(only["tumour"].affine), affine_for(sample, "PET")
            )

    def test_the_affine_is_the_grid_affine(self, indexed_cohort):
        from medh5.monai import affine_for, meta_dict

        with medh5.open(indexed_cohort[0]) as sample:
            affine = affine_for(sample, "CT_tp0")
            assert np.allclose(affine, sample.grids["ct_tp0"].affine)
            meta = meta_dict(sample, "CT_tp0")
            assert meta["space"] == "LPS"
            assert meta["medh5"]["modality"] == "CT"
            assert list(meta["spatial_shape"]) == list(SHAPE)

    def test_S3_1_LPS_to_RAS_flips_only_the_first_two_world_axes(self, indexed_cohort):
        from medh5.monai import affine_for, convert_affine

        with medh5.open(indexed_cohort[0]) as sample:
            lps = affine_for(sample, "CT_tp0")
            ras = affine_for(sample, "CT_tp0", space="RAS")
        assert np.allclose(ras[:2], -lps[:2])
        assert np.allclose(ras[2:], lps[2:])
        assert np.allclose(convert_affine(lps, source="LPS", target="LPS"), lps)

    def test_a_conversion_it_cannot_justify_is_refused(self):
        from medh5.monai import convert_affine

        with pytest.raises(MEDH5ValidationError, match="3-D"):
            convert_affine(np.eye(3), source="LPS", target="RAS")
        with pytest.raises(MEDH5ValidationError, match="world convention"):
            convert_affine(np.eye(4), source="LPS", target="talairach")

    def test_an_roi_moves_the_origin(self, indexed_cohort):
        from medh5.monai import _shift_origin, affine_for

        with medh5.open(indexed_cohort[0]) as sample:
            grid = sample.grids["ct_tp0"]
            affine = affine_for(sample, "CT_tp0")
            roi = (slice(2, 6, 1), slice(4, 8, 1), slice(0, 4, 1))
            shifted = _shift_origin(affine, roi)
        assert np.allclose(shifted[:3, 3], grid.index_to_world([[2, 4, 0]])[0])
        assert np.allclose(shifted[:3, :3], affine[:3, :3])

    def test_S3_3_a_negative_or_strided_roi_lands_where_its_voxels_are(self, tmp_path):
        """A negative start counted from the first voxel, and a step was
        dropped: the crop's affine pointed at the wrong end of the volume, at
        the full spacing."""
        pytest.importorskip("monai")
        from medh5.monai import to_metatensor

        path = tmp_path / "roi.medh5"
        shape = (6, 7, 8)
        with medh5.create(path, sample_id="roi") as w:
            w.add_grid("g", shape=shape, spacing=(2.0, 1.0, 0.5), timepoint="tp0")
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
        roi = [slice(-3, None), slice(None, None, 2), slice(1, 8, 3)]
        with medh5.open(path) as sample:
            grid = sample.grids["g"]
            tensor = to_metatensor(sample, "CT", roi=roi, space="LPS")
            affine = np.asarray(tensor.affine, dtype=np.float64)
            expected = [
                [3 + i, 2 * j, 1 + 3 * k]
                for i in range(3)
                for j in range(4)
                for k in range(3)
            ]
            local = [[i, j, k] for i in range(3) for j in range(4) for k in range(3)]
            world = grid.index_to_world(expected)
        assert tuple(tensor.shape) == (3, 4, 3)
        assert np.allclose(
            (affine[:3, :3] @ np.asarray(local).T).T + affine[:3, 3], world
        )
        assert tensor.meta["medh5"]["roi"] == [[3, 6, 1], [0, 7, 2], [1, 8, 3]]

    def test_S3_6_a_cropped_channel_image_keeps_its_channels(self, tmp_path):
        """`spatial_shape` was the crop's whole shape, channel axis included,
        and SaveImage wrote a two-channel crop as one.  It is the image's
        spatial shape, beside the
        `original_affine` it goes with, as after a MONAI crop: a resampling
        writer puts the crop back where it came from."""
        pytest.importorskip("monai")
        nib = pytest.importorskip("nibabel")
        from monai.transforms import SaveImage

        from medh5.monai import to_metatensor

        path = tmp_path / "mr.medh5"
        volume = np.zeros((2, 4, 5, 6), np.float32)
        volume[0, 1:3, 1:4, 2:5] = 1.0
        volume[1, 1:3, 1:4, 2:5] = 2.0
        with medh5.create(path, sample_id="mr") as w:
            w.add_grid(
                "g",
                shape=(2, 4, 5, 6),
                spacing=(2.0, 1.0, 1.0),
                axis_names=("c", "z", "y", "x"),
                axis_kinds=("channel", "spatial", "spatial", "spatial"),
                timepoint="tp0",
            )
            w.add_image("MR", volume, grid="g", modality="MR")
        roi = (slice(1, 3), slice(1, 4), slice(2, 5))
        with medh5.open(path) as sample:
            tensor = to_metatensor(sample, "MR", roi=roi)
        assert tuple(tensor.shape) == (2, 2, 3, 3)
        assert list(tensor.meta["spatial_shape"]) == [4, 5, 6]
        written = {}
        for resample in (False, True):
            out = tmp_path / f"resample-{resample}"
            SaveImage(
                output_dir=out,
                output_ext=".nii.gz",
                resample=resample,
                separate_folder=False,
                print_log=False,
                padding_mode="zeros",
            )(tensor)
            (saved,) = out.glob("*.nii.gz")
            # Named after the sample it came from (`filename_or_obj`), not
            # numbered.
            assert saved.name.startswith("mr")
            written[resample] = np.asarray(nib.load(str(saved)).dataobj)
        crop = written[False]
        assert crop.shape == (2, 3, 3, 2)
        assert crop[..., 0].max() == 1.0 and crop[..., 1].max() == 2.0
        assert np.count_nonzero(crop.reshape(-1, 2), axis=0).tolist() == [18, 18]
        # Resampled: the image's extent, both channels.  MONAI's channel-last
        # interpolation decides the values, as it does after its own crop.
        back = written[True]
        assert back.shape == (4, 5, 6, 2)
        assert np.count_nonzero(back.reshape(-1, 2), axis=0).all()

    def test_an_index_in_the_roi_is_refused(self, tmp_path):
        pytest.importorskip("monai")
        from medh5.monai import to_metatensor

        path = tmp_path / "roi.medh5"
        with medh5.create(path, sample_id="roi") as w:
            w.add_grid("g", shape=(6, 7, 8), spacing=(2.0, 1.0, 0.5), timepoint="tp0")
            w.add_image("CT", np.zeros((6, 7, 8), np.int16), grid="g", modality="CT")
        with (
            medh5.open(path) as sample,
            pytest.raises(MEDH5ValidationError, match="slice"),
        ):
            to_metatensor(sample, "CT", roi=[slice(0, 2), 3, slice(None)])

    def test_S4_3_a_MetaTensor_reads_the_level_its_affine_describes(self, tmp_path):
        """Level-0 voxels under a level-1 affine misplace every saved prediction.

        `meta_dict` already selects the requested level's grid and affine, so the
        array has to come from that level too; nothing about the MetaTensor
        would say the two disagree.
        """
        pytest.importorskip("monai")
        from medh5.geometry import derive_level_grid
        from medh5.monai import to_metatensor

        shape = (16, 32, 32)
        path = tmp_path / "pyr.medh5"
        with medh5.create(path, codec="portable") as w:
            base = w.add_grid(
                "l0", shape=shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
            )
            level1 = derive_level_grid(base, (2, 2, 2), "l1")
            w.add_grid(
                "l1",
                shape=level1.shape,
                spacing=level1.spacing,
                origin=level1.origin,
                direction=level1.direction,
            )
            w.add_pyramid(
                "CT",
                [
                    np.zeros(shape, dtype=np.int16),
                    np.full((8, 16, 16), 7, dtype=np.int16),
                ],
                grid_levels=["l0", "l1"],
                modality="CT",
            )

        with medh5.open(path) as sample:
            tensor = to_metatensor(sample, "CT", level=1)
            assert tuple(tensor.shape) == (8, 16, 16), "the level-1 array, not level 0"
            assert float(np.asarray(tensor).flat[0]) == 7.0
            assert list(tensor.meta["spatial_shape"]) == [8, 16, 16]
            assert np.allclose(
                np.asarray(tensor.meta["affine"]),
                sample.images["CT"].level(1).grid.affine,
            )

    def test_S8_a_box_annotation_is_refused_by_name(self, tmp_path):
        """`labelmap()` ran before the voxel check, so a box failed on the
        missing method instead of naming the annotation."""
        pytest.importorskip("monai")
        from medh5.monai import to_dict

        path = tmp_path / "boxes.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            box = np.asarray([[[0.5, 2.5], [0.5, 2.5], [0.5, 2.5]]], np.float32)
            w.add_boxes("det", box, [1], grid="g")
        with medh5.open(path) as sample:
            with pytest.raises(MEDH5ValidationError, match="'det' is 'boxes'"):
                to_dict(sample, ["CT"], ["det"])
            assert set(to_dict(sample, ["CT"])) == {"CT"}

    def test_a_sample_without_frame_uid_batches(self, tmp_path):
        """MONAI's collate refused the `None` of a missing `frame_uid`, so no
        NIfTI or nnU-Net import could be batched (D30)."""
        pytest.importorskip("monai")
        from monai.data import list_data_collate

        from medh5.monai import to_dict

        items = []
        for i in range(2):
            path = _frameless(tmp_path / f"case{i}.medh5", seed=i)
            with medh5.open(path) as sample:
                items.append(to_dict(sample, ["CT"], ["organs"]))
        assert items[0]["CT"].meta["medh5"]["frame_uid"] == "none"
        batch = list_data_collate(items)
        assert tuple(batch["CT"].shape) == (2, *Hierarchy.SHAPE)
        assert batch["CT"].meta["medh5"]["frame_uid"] == ["none", "none"]
        assert batch["organs"].meta["medh5"]["sample_key"] == ["none", "none"]

    def test_medh5_metadata_is_json_safe(self, indexed_cohort):
        from medh5.monai import meta_dict

        with medh5.open(indexed_cohort[0]) as sample:
            json.dumps(meta_dict(sample, "CT_tp0")["medh5"])

    def test_the_metadata_names_its_file(self, indexed_cohort):
        from medh5.monai import _annotation_meta, meta_dict

        with medh5.open(indexed_cohort[0]) as sample:
            assert meta_dict(sample, "CT_tp0")["filename_or_obj"] == sample.path
            ann = sample.annotations["organs_tp0"]
            assert _annotation_meta(sample, ann)["filename_or_obj"] == sample.path


class TestWithoutMonai:
    """The parts of the adapter that do not need MONAI installed."""

    def test_importing_medh5_monai_imports_no_monai(self, tmp_path):
        """The reader subclasses MONAI's, so it lives in a module of its own,
        loaded when it is asked for."""
        import subprocess
        import sys

        script = (
            "import sys; import medh5.monai as m; "
            "assert 'monai' not in sys.modules, 'imported monai'; "
            "assert 'MedH5Reader' in dir(m) and 'LoadMedH5d' in m.__all__; "
            "print('clean')"
        )
        # Away from the checkout, whose `medh5/` would shadow the installed one.
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            cwd=tmp_path,
        )
        assert "clean" in result.stdout

    def test_the_reader_names_the_extra(self, tmp_path):
        import subprocess
        import sys

        script = (
            "import sys\n"
            "class Refuse:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'monai':\n"
            "            raise ImportError('no monai here', name=name)\n"
            "sys.meta_path.insert(0, Refuse())\n"
            "import medh5.monai as m\n"
            "for build in (lambda: m.MedH5Reader('CT'), lambda: m.LoadMedH5d()):\n"
            "    try:\n"
            "        build()\n"
            "    except ImportError as error:\n"
            "        print(error)\n"
            "try:\n"
            "    m.NoSuchName\n"
            "except AttributeError as error:\n"
            "    print(error)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            cwd=tmp_path,
        )
        assert result.stdout.count("pip install 'medh5[monai]'") == 2
        assert "has no attribute 'NoSuchName'" in result.stdout

    def test_from_metatensor_maps_none_back(self, tmp_path):
        """`"none"` is how MONAI's metadata spells an absent value; a writer
        reading it back gets `None`, not a frame named "none"."""
        from medh5.monai import from_metatensor, meta_dict

        path = _frameless(tmp_path / "f.medh5")
        with medh5.open(path) as sample:
            meta = meta_dict(sample, "CT")
        assert meta["medh5"]["frame_uid"] == "none"
        assert meta["medh5"]["value_units"] == "none"
        assert meta["medh5"]["sample_key"] == "none"

        class Duck:
            def __array__(self, dtype=None, copy=None):
                return np.zeros(Hierarchy.SHAPE, dtype=dtype)

        duck = Duck()
        duck.meta = meta  # type: ignore[attr-defined]
        recovered = from_metatensor(duck)[1]["medh5"]
        assert recovered["frame_uid"] is None
        assert recovered["value_units"] is None
        assert recovered["sample_key"] is None
        assert recovered["modality"] == "CT"

    def test_from_metatensor_recovers_the_geometry(self):
        from medh5.monai import from_metatensor

        class Duck:
            """Anything with `.meta` and array semantics --- MONAI's own shape."""

            def __init__(self, array, meta):
                self._array = array
                self.meta = meta

            def __array__(self, dtype=None, copy=None):
                return np.asarray(self._array, dtype=dtype)

        affine = np.diag([2.0, 1.0, 1.0, 1.0])
        affine[:3, 3] = [5.0, 6.0, 7.0]
        array, geometry = from_metatensor(
            Duck(np.zeros((4, 4, 4)), {"affine": affine, "space": "RAS"})
        )
        assert array.shape == (4, 4, 4)
        assert geometry["spacing"] == [2.0, 1.0, 1.0]
        assert geometry["origin"] == [5.0, 6.0, 7.0]
        assert geometry["coord_system"] == "RAS"

    def test_S3_3_a_tensor_without_geometry_is_refused(self):
        """An identity affine and `LPS` stood in for a tensor's missing ones,
        and a `(C, Z, Y, X)` array came back with four spacings."""
        from medh5.monai import from_metatensor

        with pytest.raises(MEDH5ValidationError, match="no affine and no space"):
            from_metatensor(np.zeros((2, 4, 4, 4)))

        class Duck:
            def __init__(self, meta):
                self.meta = meta

            def __array__(self, dtype=None, copy=None):
                return np.zeros((4, 4, 4), dtype=dtype)

        with pytest.raises(MEDH5ValidationError, match="no space"):
            from_metatensor(Duck({"affine": np.eye(4)}))

    @pytest.mark.parametrize(("units", "scale"), [("m", 1000.0), ("mm", 1.0)])
    def test_S3_5_monai_receives_the_affine_in_millimetres(
        self, tmp_path: Path, units: str, scale: float
    ):
        """MONAI reads an affine in millimetres: in the grid's own units, a
        metre grid's 2 mm voxels said 2 µm."""
        from medh5.monai import affine_for, from_metatensor, meta_dict

        path = tmp_path / "units.medh5"
        spacing = np.array([2.0, 1.0, 0.5]) / scale
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_grid(
                "g",
                shape=(4, 5, 6),
                spacing=spacing,
                origin=(0.1, 0.2, 0.3),
                units=units,
            )
            w.add_image("CT", np.zeros((4, 5, 6), np.int16), grid="g", modality="CT")
        with medh5.open(path) as sample:
            grid = sample.grids["g"]
            affine = affine_for(sample, "CT")
            meta = meta_dict(sample, "CT")
        assert np.allclose(np.diag(affine)[:3], [2.0, 1.0, 0.5])
        assert np.allclose(affine[:3, 3], np.array([0.1, 0.2, 0.3]) * scale)
        assert np.allclose(affine[:3, :3], grid.affine[:3, :3] * scale)
        assert np.allclose(meta["affine"], affine)
        assert meta["medh5"]["units"] == units

        class Duck:
            def __array__(self, dtype=None, copy=None):
                return np.zeros((4, 5, 6), dtype=dtype)

        duck = Duck()
        duck.meta = meta  # type: ignore[attr-defined]
        _, geometry = from_metatensor(duck)
        assert np.allclose(geometry["spacing"], [2.0, 1.0, 0.5])
        assert geometry["units"] == "mm"

    def test_S3_5_a_pixel_grid_stays_in_pixels(self, tmp_path: Path):
        """`px` has no size to scale, and the metadata says which it is."""
        from medh5.monai import affine_for, from_metatensor, meta_dict

        path = tmp_path / "px.medh5"
        with medh5.create(path, sample_id="s", codec="portable") as w:
            w.add_grid("g", shape=(4, 5, 6), spacing=(1.0, 1.0, 1.0), units="px")
            w.add_image("CT", np.zeros((4, 5, 6), np.int16), grid="g", modality="CT")
        with medh5.open(path) as sample:
            assert np.allclose(affine_for(sample, "CT"), sample.grids["g"].affine)
            meta = meta_dict(sample, "CT")

        class Duck:
            def __array__(self, dtype=None, copy=None):
                return np.zeros((4, 5, 6), dtype=dtype)

        duck = Duck()
        duck.meta = meta  # type: ignore[attr-defined]
        assert from_metatensor(duck)[1]["units"] == "px"

    def test_S3_6_original_channel_dim_follows_the_grid(self, tmp_path: Path):
        """`no_channel` on an RGB grid made `EnsureChannelFirst` add a second axis."""
        from medh5.monai import meta_dict

        path = tmp_path / "rgb.medh5"
        with medh5.create(path, sample_id="s", subject_id="s", codec="portable") as w:
            w.add_grid(
                "rgb",
                shape=(3, 4, 5, 6),
                spacing=(1, 1, 1),
                axis_kinds=("channel", "spatial", "spatial", "spatial"),
            )
            w.add_image(
                "RGB",
                np.zeros((3, 4, 5, 6), np.uint8),
                grid="rgb",
                modality="OT",
                channel_names=["r", "g", "b"],
            )
            w.add_grid("g", shape=(4, 5, 6), spacing=(1, 1, 1))
            w.add_image("CT", np.zeros((4, 5, 6), np.int16), grid="g", modality="CT")
        with medh5.open(path) as s:
            assert meta_dict(s, "RGB")["original_channel_dim"] == 0
            assert meta_dict(s, "CT")["original_channel_dim"] == "no_channel"


def _opens(item: dict) -> dict:
    """A worker's handle-cache opens, beside the item it read (module level, so
    a `spawn` worker can unpickle it)."""
    import os

    from medh5.torch.handles import CACHE

    return {**item, "opens": CACHE.opens, "pid": os.getpid()}


class TestMonaiReader:
    """`MedH5Reader`: `LoadImage` opens a `.medh5` as MONAI's own readers do."""

    def test_reader_affine_matches_to_metatensor(self, indexed_cohort):
        pytest.importorskip("monai")
        from monai.transforms import LoadImage

        from medh5.monai import MedH5Reader, to_metatensor

        loaded = LoadImage(reader=MedH5Reader("CT_tp0"), image_only=True)(
            indexed_cohort[0]
        )
        with medh5.open(indexed_cohort[0]) as sample:
            expected = to_metatensor(sample, "CT_tp0")
        assert np.allclose(np.asarray(loaded.affine), np.asarray(expected.affine))
        assert np.array_equal(np.asarray(loaded), np.asarray(expected))
        assert loaded.meta["medh5"]["image_id"] == "CT_tp0"
        assert loaded.meta["space"] == "LPS"

    def test_label_is_int64_with_65535_ignore(self, tmp_path):
        pytest.importorskip("monai")
        from monai.transforms import LoadImage

        from medh5.monai import MedH5Reader

        path = _frameless(tmp_path / "f.medh5")
        label = LoadImage(
            reader=MedH5Reader(annotation="organs"), image_only=True, dtype=None
        )(path)
        assert str(label.dtype) == "torch.int64"
        assert set(np.unique(np.asarray(label)).tolist()) == {0, 1, 65535}
        assert np.all(np.asarray(label)[-1] == 65535)
        assert label.meta["medh5"]["annotation_id"] == "organs"
        with pytest.raises(MEDH5ValidationError, match="an image or an annotation"):
            MedH5Reader("CT", annotation="organs")

    def test_an_unnamed_image_is_refused_when_there_are_several(self, tmp_path):
        pytest.importorskip("monai")
        from monai.transforms import LoadImage

        from medh5.monai import MedH5Reader

        one = _frameless(tmp_path / "one.medh5")
        two = _frameless(tmp_path / "two.medh5", images=("CT", "PET"))
        assert tuple(LoadImage(reader=MedH5Reader(), image_only=True)(one).shape) == (
            Hierarchy.SHAPE
        )
        with pytest.raises(MEDH5ValidationError, match="'CT', 'PET'.*image="):
            LoadImage(reader=MedH5Reader(), image_only=True)(two)

    def test_loadimaged_batches_through_list_data_collate(self, tmp_path):
        pytest.importorskip("monai")
        from monai.data import list_data_collate
        from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

        from medh5.monai import MedH5Reader

        paths = [_frameless(tmp_path / f"c{i}.medh5", seed=i) for i in range(3)]
        load = Compose(
            [
                LoadImaged("image", reader=MedH5Reader("CT")),
                LoadImaged(
                    "label", reader=MedH5Reader(annotation="organs"), dtype=None
                ),
                EnsureChannelFirstd(["image", "label"]),
            ]
        )
        items = [load({"image": str(p), "label": str(p)}) for p in paths]
        batch = list_data_collate(items)
        assert tuple(batch["image"].shape) == (3, 1, *Hierarchy.SHAPE)
        assert tuple(batch["label"].shape) == (3, 1, *Hierarchy.SHAPE)
        assert batch["image"].meta["medh5"]["frame_uid"] == ["none"] * 3

    def test_several_files_stack_as_channels_only_on_one_grid(self, tmp_path):
        pytest.importorskip("monai")
        from monai.transforms import LoadImage

        from medh5.monai import MedH5Reader

        a = _frameless(tmp_path / "a.medh5", seed=1)
        b = _frameless(tmp_path / "b.medh5", seed=2)
        stacked = LoadImage(reader=MedH5Reader("CT"), image_only=True)([a, b])
        assert tuple(stacked.shape) == (2, *Hierarchy.SHAPE)
        assert stacked.meta["original_channel_dim"] == 0
        other = tmp_path / "other.medh5"
        with medh5.create(other, sample_id="other", codec="portable") as w:
            w.add_grid("g", shape=Hierarchy.SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT", np.zeros(Hierarchy.SHAPE, np.int16), grid="g", modality="CT"
            )
        with pytest.raises(MEDH5ValidationError, match="not on one grid") as caught:
            LoadImage(reader=MedH5Reader("CT"), image_only=True)([a, other])
        assert "medh5 resample" in str(caught.value)

    def test_a_collection_member_by_path_key(self, tmp_path):
        pytest.importorskip("monai")
        from monai.transforms import LoadImage

        from medh5.collection import pack
        from medh5.monai import MedH5Reader

        a = _frameless(tmp_path / "a.medh5", seed=1)
        b = _frameless(tmp_path / "b.medh5", seed=2)
        shard = pack([a, b], tmp_path / "shard.medh5c")
        reader = MedH5Reader("CT")
        assert reader.verify_suffix(f"{shard}::b")
        assert reader.verify_suffix(shard)
        assert not reader.verify_suffix(tmp_path / "x.nii.gz")
        member = LoadImage(reader=reader, image_only=True)(f"{shard}::b")
        with medh5.open(b) as sample:
            expected = sample.images["CT"].read(physical=True, dtype=np.float32)
        assert np.array_equal(np.asarray(member), expected)
        assert member.meta["medh5"]["sample_key"] == "b"
        keyed = LoadImage(reader=MedH5Reader("CT", sample_key="a"), image_only=True)
        assert keyed(shard).meta["medh5"]["sample_key"] == "a"
        with pytest.raises(MEDH5ValidationError, match="is a collection"):
            LoadImage(reader=reader, image_only=True)(shard)

    def test_workers_reuse_handles(self, tmp_path):
        """Each worker opens a file once, however many items read it."""
        pytest.importorskip("monai")
        from monai.data import DataLoader, Dataset
        from monai.transforms import Compose, LoadImaged

        from medh5.monai import MedH5Reader
        from medh5.torch import CACHE, worker_init_fn

        CACHE.opens = 0  # what a forked worker starts counting from
        paths = [_frameless(tmp_path / f"c{i}.medh5", seed=i) for i in range(2)]
        data = [{"image": str(p)} for p in paths * 6]
        transform = Compose([LoadImaged("image", reader=MedH5Reader("CT")), _opens])
        loader = DataLoader(
            Dataset(data, transform),
            batch_size=2,
            num_workers=2,
            worker_init_fn=worker_init_fn,
        )
        opens: dict[int, int] = {}
        for batch in loader:
            for pid, n in zip(batch["pid"], batch["opens"], strict=True):
                opens[int(pid)] = max(opens.get(int(pid), 0), int(n))
        assert opens and all(n <= len(paths) for n in opens.values()), opens


class TestMonaiLoadMedH5d:
    """`LoadMedH5d`: `to_dict` as a dictionary transform."""

    def test_one_lease_per_item(self, tmp_path, monkeypatch):
        pytest.importorskip("monai")
        from medh5.monai import LoadMedH5d, to_dict
        from medh5.torch.handles import HandleCache

        path = _frameless(tmp_path / "f.medh5")
        leases: list[tuple] = []
        lease = HandleCache.lease

        def counted(self, *args, **kwargs):
            leases.append(args)
            return lease(self, *args, **kwargs)

        monkeypatch.setattr(HandleCache, "lease", counted)
        item = LoadMedH5d(images=["CT"], annotations=["organs"])({"path": str(path)})
        assert len(leases) == 1
        assert set(item) == {"path", "CT", "organs"}
        with medh5.open(path) as sample:
            expected = to_dict(sample, ["CT"], ["organs"])
        assert np.array_equal(
            np.asarray(item["organs"]), np.asarray(expected["organs"])
        )
        with pytest.raises(MEDH5ValidationError, match="one key"):
            LoadMedH5d(["a", "b"])

    def test_spacingd_and_orientationd_accept_the_output(self, tmp_path):
        pytest.importorskip("monai")
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            Orientationd,
            Spacingd,
        )

        from medh5.monai import LoadMedH5d

        path = _frameless(tmp_path / "f.medh5")
        keys = ["CT", "organs"]
        item = Compose(
            [
                LoadMedH5d(images=["CT"], annotations=["organs"]),
                EnsureChannelFirstd(keys),
                Orientationd(keys, axcodes="RAS"),
                Spacingd(keys, pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")),
            ]
        )({"path": str(path)})
        assert item["CT"].shape[1:] == item["organs"].shape[1:]
        assert np.allclose(
            np.abs(np.diag(np.asarray(item["CT"].affine))[:3]), [2.0, 2.0, 2.0]
        )
