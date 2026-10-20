"""Agreement between two raters' annotations (spec §11.2, §11.3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.curation.agreement import (
    box_iou,
    compare,
    compare_instances,
    compare_voxel,
    dice,
    iou,
)
from medh5.errors import MEDH5ValidationError
from medh5.labels import LabelClass, LabelSet, OntologyCode
from medh5.validate import validate_file
from tests.helpers import SHAPE, block, lesion, write_raters
from tests.kits import Framed


class TestMeasuredAgreement:
    """Agreement scores what was measured, and its record is storable."""

    def _raters(self, path: Path) -> Path:
        empty = np.empty((0, 3, 2), np.float32)
        with Framed.writer(path) as w:
            w.add_boxes("r1", empty, [], grid="g", annotated_classes=["c1"])
            w.add_boxes("r2", empty, [], grid="g", annotated_classes=["c1"])
            w.add_boxes(
                "r3",
                np.array([Framed.box(0, 2), Framed.box(4, 6)], np.float32),
                ["c1", "c2"],
                grid="g",
                annotated_classes=["c1", "c2"],
            )
            w.add_boxes(
                "r4",
                np.array([Framed.box(0, 2)], np.float32),
                ["c1"],
                grid="g",
                annotated_classes=["c1"],
            )
        return path

    def test_S11_2_two_raters_who_found_nothing_are_undefined_not_zero(
        self, tmp_path: Path
    ):
        """Object F1 was 0.0 --- and so was the record --- when both found nothing."""
        from medh5.curation.agreement import compare_instances

        with medh5.open(self._raters(tmp_path / "a.medh5")) as s:
            result = compare_instances(s.annotations["r1"], s.annotations["r2"])
            assert result.value is None
            assert result.mean_iou is None
            with pytest.raises(MEDH5ValidationError, match="nothing was comparable"):
                result.to_record()

    def test_S11_3_a_class_one_rater_never_examined_is_skipped(self, tmp_path: Path):
        """Identical work scored 0.667 when the other rater never looked for c2."""
        from medh5.curation.agreement import compare_instances

        with medh5.open(self._raters(tmp_path / "a.medh5")) as s:
            result = compare_instances(s.annotations["r3"], s.annotations["r4"])
            assert result.value == pytest.approx(1.0)
            assert result.only_in_a == ()
            assert result.skipped == ("c2 (not examined by both)",)

    def test_S8_1_index_boxes_on_different_grids_are_refused(self, tmp_path: Path):
        """Boxes on a 4 mm and a 1 mm grid were compared in raw index units."""
        from medh5.curation.agreement import compare_instances

        path = tmp_path / "grids.medh5"
        with Framed.writer(path) as w:
            w.add_grid("fine", shape=(16, 32, 32), spacing=(1.0, 0.5, 0.5))
            w.add_boxes("a", np.array([Framed.box(0, 2)], np.float32), ["c1"], grid="g")
            w.add_boxes(
                "b", np.array([Framed.box(0, 2)], np.float32), ["c1"], grid="fine"
            )
        with medh5.open(path) as s, pytest.raises(MEDH5ValidationError) as caught:
            compare_instances(s.annotations["a"], s.annotations["b"])
        assert caught.value.code == "E101"

    def test_S8_1_boxes_in_different_spaces_are_refused(self, tmp_path: Path):
        from medh5.curation.agreement import compare_instances

        path = tmp_path / "spaces.medh5"
        with Framed.writer(path) as w:
            w.add_boxes("a", np.array([Framed.box(0, 2)], np.float32), ["c1"], grid="g")
            w.add_boxes(
                "b",
                np.array([Framed.box(0, 4)], np.float32),
                ["c1"],
                grid="g",
                space="world",
                frame_uid="1.2.3.4",
            )
        with medh5.open(path) as s, pytest.raises(MEDH5ValidationError) as caught:
            compare_instances(s.annotations["a"], s.annotations["b"])
        assert caught.value.code == "E414"

    def test_S3_3_world_boxes_compare_within_one_frame(self, tmp_path: Path):
        """Two gridless world annotations passed a grid test as `None == None`."""
        from medh5.curation.agreement import compare_instances

        path = tmp_path / "frames.medh5"
        box = np.array([Framed.box(0, 2)], np.float32)
        with Framed.writer(path) as w:
            w.add_boxes("a", box, ["c1"], space="world", frame_uid="F")
            w.add_boxes("b", box, ["c1"], space="world", frame_uid="M")
            w.add_boxes("c", box, ["c1"], space="world", frame_uid="F")
        with medh5.open(path) as s:
            same = compare_instances(s.annotations["a"], s.annotations["c"])
            assert same.value == pytest.approx(1.0)
            with pytest.raises(MEDH5ValidationError) as caught:
                compare_instances(s.annotations["a"], s.annotations["b"])
        assert caught.value.code == "E414"

    def test_compare_refuses_an_argument_it_cannot_use(self, tmp_path: Path):
        from medh5.curation.agreement import compare

        path = self._raters(tmp_path / "a.medh5")
        mask = np.zeros(Framed.SHAPE, bool)
        mask[2:5, 2:5, 2:5] = True
        with medh5.amend(path) as w:
            w.add_segmentation("s1", grid="g", masks={1: mask})
            w.add_segmentation("s2", grid="g", masks={1: mask})
        with medh5.open(path) as s:
            assert compare(s.annotations["r3"], s.annotations["r4"]).value == 1.0
            assert compare(s.annotations["s1"], s.annotations["s2"]).metric == "dice"
            with pytest.raises(MEDH5ValidationError, match="scores voxels"):
                compare(s.annotations["r3"], s.annotations["r4"], metric="iou")
            with pytest.raises(MEDH5ValidationError, match="matches objects"):
                compare(s.annotations["s1"], s.annotations["s2"], threshold=0.5)
            with pytest.raises(MEDH5ValidationError, match="common kind"):
                compare(s.annotations["r3"], s.annotations["s1"])

    def test_medh5_agree_on_two_boxes_annotations(self, tmp_path: Path, capsys):
        """It exited with `'BoxesAnnotation' object has no attribute 'dense'`."""
        from medh5.cli import main

        path = self._raters(tmp_path / "a.medh5")
        assert main(["agree", str(path), "r3", "r4"]) == 0
        out = capsys.readouterr().out
        assert "object_f1 = 1.0000" in out and "not scored: c2" in out
        assert main(["agree", str(path), "r1", "r2", "--record"]) == 1
        assert "nothing was comparable" in capsys.readouterr().err

    def test_S11_2_voxel_agreement_with_nothing_comparable_is_undefined(
        self, tmp_path: Path
    ):
        from medh5.curation.agreement import compare_voxel

        path = tmp_path / "vox.medh5"
        mask = np.zeros(Framed.SHAPE, bool)
        mask[2:5, 2:5, 2:5] = True
        with Framed.writer(path) as w:
            w.add_segmentation("a", grid="g", masks={1: mask}, annotated_classes=[1])
            w.add_segmentation("b", grid="g", masks={2: mask}, annotated_classes=[2])
        with medh5.open(path) as s:
            result = compare_voxel(s.annotations["a"], s.annotations["b"])
            assert result.value is None
            with pytest.raises(MEDH5ValidationError):
                result.to_record()

    def test_S11_2_an_undefined_voxel_comparison_is_reported_not_refused(
        self, tmp_path: Path, capsys
    ):
        """The report went through `to_record()`, so it refused as the record does."""
        from medh5.cli import main
        from medh5.curation.agreement import compare_voxel

        path = tmp_path / "undefined.medh5"
        mask = np.zeros(Framed.SHAPE, bool)
        mask[2:5, 2:5, 2:5] = True
        with Framed.writer(path) as w:
            w.add_segmentation("a", grid="g", masks={1: mask}, annotated_classes=[1])
            w.add_segmentation("b", grid="g", masks={2: mask}, annotated_classes=[2])
        with medh5.open(path) as s:
            report = compare_voxel(s.annotations["a"], s.annotations["b"]).to_json()
        assert report["value"] is None and report["compared"] == 0
        assert main(["agree", str(path), "a", "b"]) == 0
        assert "undefined (nothing comparable)" in capsys.readouterr().out
        assert main(["agree", str(path), "a", "b", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["value"] is None
        assert main(["agree", str(path), "a", "b", "--record"]) == 1
        assert "nothing was comparable" in capsys.readouterr().err

    def test_S11_2_an_agreement_record_can_be_stored_in_its_file(self, tmp_path: Path):
        """Keyed by class name, and by `mean_iou`, no record passed the schema."""
        from medh5.curation.agreement import compare_instances, compare_voxel

        path = self._raters(tmp_path / "a.medh5")
        mask = np.zeros(Framed.SHAPE, bool)
        mask[2:5, 2:5, 2:5] = True
        with medh5.amend(path) as w:
            w.add_segmentation("s1", grid="g", masks={1: mask})
            w.add_segmentation("s2", grid="g", masks={1: mask})
        with medh5.open(path) as s:
            voxel = compare_voxel(s.annotations["s1"], s.annotations["s2"]).to_record()
            boxes = compare_instances(
                s.annotations["r3"], s.annotations["r4"]
            ).to_record()
        assert voxel.per_class == {"1": 1.0}
        assert boxes.per_class == {}
        with medh5.amend(path) as w:
            w.set_quality(
                "q", status="reviewed", agreement=[voxel.to_json(), boxes.to_json()]
            )
        assert "E005" not in validate_file(path).codes


class TestAgreement:
    def test_dice_and_iou_are_undefined_on_two_empty_masks(self):
        empty = np.zeros((4, 4), dtype=bool)
        assert dice(empty, empty) is None
        assert iou(empty, empty) is None
        full = np.ones((4, 4), dtype=bool)
        assert dice(full, full) == 1.0
        assert iou(full, empty) == 0.0

    def test_box_iou(self):
        a = np.array([[0.0, 2.0], [0.0, 2.0]])
        assert box_iou(a, a) == pytest.approx(1.0)
        far = np.array([[9.0, 10.0], [9.0, 10.0]])
        assert box_iou(a, far) == 0.0

    def test_S11_2_per_class_dice_between_two_raters(self, tmp_path, label_set):
        path = tmp_path / "raters.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation(
                "r1",
                grid="g",
                masks={1: block(SHAPE, (2, 2, 2), 8), 2: block(SHAPE, (8, 8, 8), 4)},
                annotated_classes=[1, 2, 3],
            )
            w.add_segmentation(
                "r2",
                grid="g",
                masks={1: block(SHAPE, (2, 2, 2), 8), 2: block(SHAPE, (8, 8, 8), 2)},
                annotated_classes=[1, 2],
            )
        with medh5.open(path) as sample:
            result = compare_voxel(sample.annotations["r1"], sample.annotations["r2"])
            assert result.per_class["liver"] == pytest.approx(1.0)
            assert 0.0 < result.per_class["spleen"] < 1.0
            assert result.value == pytest.approx(
                np.mean(list(result.per_class.values()))
            )
            record = result.to_record()
            assert record.metric == "dice"
            assert record.against == "annotations/r2"
            # lesion: r1 examined it, r2 did not --- not a disagreement (§11.3).
            assert result.skipped == ("lesion (not examined by both)",)
            assert "lesion" not in result.per_class

            by_iou = compare_voxel(
                sample.annotations["r1"], sample.annotations["r2"], metric="iou"
            )
            assert by_iou.metric == "iou"
            assert by_iou.value <= result.value
            assert compare(sample.annotations["r1"], sample.annotations["r2"]).metric

    def test_an_unknown_metric_is_refused(self, sample_path):
        with (
            medh5.open(sample_path) as sample,
            pytest.raises(MEDH5ValidationError),
        ):
            compare_voxel(
                sample.annotations["organs_tp0"],
                sample.annotations["organs_tp0"],
                metric="vibes",
            )

    def test_annotations_on_different_grids_are_refused(self, series):
        with (
            medh5.open(series) as sample,
            pytest.raises(MEDH5ValidationError) as exc,
        ):
            compare_voxel(sample.annotations["les_tp0"], sample.annotations["les_tp1"])
        assert exc.value.code == "E101"
        assert "medh5 resample" in str(exc.value)

    def test_instances_on_two_visits_grids_are_refused(self, series):
        # Index boxes on two grids count different voxels; comparing them
        # number for number measured nothing.
        with (
            medh5.open(series) as sample,
            pytest.raises(MEDH5ValidationError) as exc,
        ):
            compare_instances(
                sample.annotations["les_tp0"], sample.annotations["les_tp1"]
            )
        assert exc.value.code == "E101"
        assert "medh5 resample" in str(exc.value)

    def test_instances_match_on_declared_ids_first(self, tmp_path, label_set):
        path = write_raters(tmp_path / "raters.medh5", label_set)
        with medh5.open(path) as sample:
            result = compare_instances(
                sample.annotations["r1"], sample.annotations["r2"]
            )
            assert result.matched_by == "instance_id"
            assert [m[0] for m in result.matched] == [0]
            assert result.only_in_a == (1,)
            assert result.only_in_b == (1,)
            assert 0.0 < result.value < 1.0
            assert result.to_record().metric == "object_f1"
            assert result.to_json()["matched_by"] == "instance_id"

    def test_class_mismatches_are_reported_on_matched_objects(
        self, tmp_path, label_set
    ):
        path = write_raters(
            tmp_path / "mismatch.medh5",
            label_set,
            second=[InstanceInput(class_id=1, instance_id=7, mask=lesion(8, 8, 8, 2))],
            annotated_second=[1, 3],
        )
        with medh5.open(path) as sample:
            result = compare_instances(
                sample.annotations["r1"], sample.annotations["r2"]
            )
            assert result.class_mismatches == ((7, 3, 1),)
            # r1 never looked for class 1, so it is skipped, not scored --- but
            # the matched pair counts, because both raters found the object.
            assert result.matched and result.skipped

    def test_iou_matching_is_the_fallback_without_shared_ids(self, tmp_path, label_set):
        path = write_raters(
            tmp_path / "noshare.medh5",
            label_set,
            second=[InstanceInput(class_id=3, instance_id=99, mask=lesion(8, 8, 8, 2))],
        )
        with medh5.open(path) as sample:
            result = compare_instances(
                sample.annotations["r1"], sample.annotations["r2"]
            )
            assert result.matched_by == "iou"
            assert result.matched and result.matched[0][2] == pytest.approx(1.0)
            assert result.mean_iou == pytest.approx(1.0)

    @pytest.mark.parametrize("threshold", [float("nan"), -1.0, 0.0, 1.5, float("inf")])
    def test_an_iou_threshold_lies_in_0_to_1(self, tmp_path, label_set, threshold):
        """As ``medh5 agree --threshold`` holds it.  The Python API took any
        float: NaN matched no object, and 0 or less every pair, objects that
        do not touch among them, each scored as agreement."""
        path = write_raters(tmp_path / "raters.medh5", label_set)
        with medh5.open(path) as sample:
            a, b = sample.annotations["r1"], sample.annotations["r2"]
            for measure in (compare_instances, compare):
                with pytest.raises(MEDH5ValidationError, match=r"in \(0, 1\]"):
                    measure(a, b, threshold=threshold)
            assert compare_instances(a, b, threshold=1.0).threshold == 1.0

    def test_no_matches_scores_zero(self, tmp_path, label_set):
        path = write_raters(
            tmp_path / "nomatch.medh5",
            label_set,
            second=[
                InstanceInput(class_id=3, instance_id=99, mask=lesion(2, 21, 21, 1))
            ],
        )
        with medh5.open(path) as sample:
            result = compare_instances(
                sample.annotations["r1"], sample.annotations["r2"]
            )
            assert result.value == 0.0
            # A mean over no matched pairs is undefined, not zero overlap.
            assert result.mean_iou is None


class TestIgnoredVoxels:
    """§7.7: a voxel either rater declared unexamined is evidence neither way."""

    @staticmethod
    def _slab(z0: int, z1: int) -> np.ndarray:
        mask = np.zeros(Framed.SHAPE, bool)
        mask[z0:z1, 2:5, 2:5] = True
        return mask

    @staticmethod
    def _ignore(z0: int, z1: int) -> np.ndarray:
        region = np.zeros(Framed.SHAPE, bool)
        region[z0:z1] = True
        return region

    @pytest.mark.parametrize("encoding", ["labelmap", "bitmask"])
    def test_S7_7_what_one_rater_never_examined_is_not_scored(
        self, tmp_path: Path, encoding: str
    ):
        """Identical work scored 0.75: rater b's finding where rater a declared
        the slab unexamined counted against a.  In band (`labelmap`) and as the
        sibling `mask` its `ignore_mask` names (`bitmask`) alike."""
        path = tmp_path / "ignore.medh5"
        with Framed.writer(path) as w:
            w.add_segmentation(
                "a",
                grid="g",
                masks={1: self._slab(2, 5)},
                annotated_classes=[1],
                ignore=self._ignore(6, 8),
                encoding=encoding,
            )
            w.add_segmentation(
                "b",
                grid="g",
                masks={1: self._slab(2, 5) | self._slab(6, 8)},
                annotated_classes=[1],
            )
        with medh5.open(path) as s:
            a, b = s.annotations["a"], s.annotations["b"]
            assert compare_voxel(a, b).value == pytest.approx(1.0)
            assert compare_voxel(b, a).value == pytest.approx(1.0)
            assert compare_voxel(a, b, metric="iou").value == pytest.approx(1.0)

    def test_S7_7_nothing_outside_the_ignore_regions_is_nothing_comparable(
        self, tmp_path: Path
    ):
        """Each rater found the class only where the other never looked."""
        path = tmp_path / "disjoint.medh5"
        with Framed.writer(path) as w:
            w.add_segmentation(
                "a",
                grid="g",
                masks={1: self._slab(2, 5)},
                annotated_classes=[1],
                ignore=self._ignore(6, 8),
            )
            w.add_segmentation(
                "b",
                grid="g",
                masks={1: self._slab(6, 8)},
                annotated_classes=[1],
                ignore=self._ignore(2, 5),
            )
        with medh5.open(path) as s:
            result = compare_voxel(s.annotations["a"], s.annotations["b"])
            assert result.value is None
            assert result.skipped == ("c1 (empty in both outside the ignore region)",)


class TestAcrossSamples:
    """A grid id is its sample's own name: two
    samples each calling a grid `g` were compared voxel for voxel and box for
    box though one was 100 mm away or in an unrelated frame, scoring 1.0, and
    grids of two shapes, each with an ignore region, panicked the merge."""

    @staticmethod
    def _sample(
        path: Path,
        *,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        frame: str = "1.2.3.4",
        shape: tuple[int, int, int] = Framed.SHAPE,
        ignore: np.ndarray | None = None,
    ) -> Path:
        mask = np.zeros(shape, bool)
        mask[2:4, 2:5, 2:5] = True
        with medh5.create(path, sample_id=path.stem, codec="portable") as w:
            w.add_grid(
                "g",
                shape=shape,
                spacing=(2.0, 1.0, 1.0),
                origin=origin,
                frame_uid=frame,
            )
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            w.label_set(Framed.label_set())
            w.add_segmentation(
                "seg", grid="g", masks={1: mask}, annotated_classes=[1], ignore=ignore
            )
            if shape == Framed.SHAPE:
                box = np.array([Framed.box(2, 4)], np.float32)
                w.add_boxes("boxes", box, ["c1"], grid="g")
                w.add_boxes("world", box, ["c1"], space="world", frame_uid=frame)
        return path

    def test_S3_3_a_grid_of_the_same_name_elsewhere_is_another_grid(
        self, tmp_path: Path
    ):
        here = self._sample(tmp_path / "here.medh5")
        moved = self._sample(tmp_path / "moved.medh5", origin=(0.0, 0.0, 100.0))
        other = self._sample(tmp_path / "other.medh5", frame="9.9.9")
        with (
            medh5.open(here) as h,
            medh5.open(moved) as m,
            medh5.open(other) as o,
        ):
            for there in (m, o):
                for name, compare_of in (
                    ("seg", compare_voxel),
                    ("boxes", compare_instances),
                ):
                    with pytest.raises(
                        MEDH5ValidationError, match="two samples"
                    ) as caught:
                        compare_of(h.annotations[name], there.annotations[name])
                    assert caught.value.code == "E101"
            with pytest.raises(MEDH5ValidationError) as caught:
                compare_instances(h.annotations["world"], o.annotations["world"])
            assert caught.value.code == "E414"

    def test_S3_3_grids_of_two_shapes_are_refused_not_merged(self, tmp_path: Path):
        def ignored(shape: tuple[int, int, int]) -> np.ndarray:
            region = np.zeros(shape, bool)
            region[0] = True
            return region

        small = self._sample(
            tmp_path / "s.medh5", shape=(4, 4, 4), ignore=ignored((4, 4, 4))
        )
        large = self._sample(
            tmp_path / "l.medh5", shape=(5, 5, 5), ignore=ignored((5, 5, 5))
        )
        with (
            medh5.open(small) as s,
            medh5.open(large) as big,
            pytest.raises(MEDH5ValidationError) as caught,
        ):
            compare_voxel(s.annotations["seg"], big.annotations["seg"])
        assert caught.value.code == "E101"

    def test_S3_3_one_lattice_in_one_frame_compares_across_samples(
        self, tmp_path: Path
    ):
        """Two raters' files of one scan: their grids share a frame and are
        one lattice, whatever they are called."""
        a = self._sample(tmp_path / "a.medh5")
        b = self._sample(tmp_path / "b.medh5")
        with medh5.open(a) as sa, medh5.open(b) as sb:
            assert (
                compare_voxel(sa.annotations["seg"], sb.annotations["seg"]).value == 1.0
            )
            for name in ("boxes", "world"):
                found = compare_instances(sa.annotations[name], sb.annotations[name])
                assert found.value == pytest.approx(1.0)
        # One file, opened twice, is one sample's grids again.
        with medh5.open(a) as first, medh5.open(a) as second:
            same = compare_voxel(first.annotations["seg"], second.annotations["seg"])
            assert same.value == 1.0
            # ... and one sample, so its `against` is the other's own path.
            assert same.to_record().against == "annotations/seg"

    def test_S11_2_a_cross_sample_record_needs_against(self, tmp_path: Path):
        """A record's `against` is a path of the recording sample: across two
        samples it named the other file's annotation as though it were here."""
        a = self._sample(tmp_path / "a.medh5")
        b = self._sample(tmp_path / "b.medh5")
        with medh5.open(a) as sa, medh5.open(b) as sb:
            voxel = compare_voxel(sa.annotations["seg"], sb.annotations["seg"])
            boxes = compare_instances(sa.annotations["boxes"], sb.annotations["boxes"])
            for result in (voxel, boxes):
                with pytest.raises(MEDH5ValidationError, match="against"):
                    result.to_record()
                for path in ("/", "images/CT", "annotations/", "annotations/a/b"):
                    with pytest.raises(MEDH5ValidationError, match="against"):
                        result.to_record(against=path)
                record = result.to_record(against="/annotations/reference")
                assert record.against == "annotations/reference"
        # Within one sample, `against` is the other annotation's own path.
        with medh5.open(a) as s:
            one = compare_voxel(s.annotations["seg"], s.annotations["seg"])
            assert one.to_record().against == "annotations/seg"

    def test_S2_2_two_members_of_one_collection_are_two_samples(self, tmp_path: Path):
        """One file holds every member of a collection, and each member is a
        sample of its own: one file is not one sample."""
        from medh5.collection import open_collection, pack

        a = self._sample(tmp_path / "a.medh5")
        b = self._sample(tmp_path / "b.medh5")
        shard = pack([a, b], tmp_path / "pair.medh5c")
        with open_collection(shard) as collection:
            sa, sb = collection["a"], collection["b"]
            across = compare_voxel(sa.annotations["seg"], sb.annotations["seg"])
            assert across.against is None
            with pytest.raises(MEDH5ValidationError, match="against"):
                across.to_record()
            one = compare_voxel(
                sa.annotations["seg"], collection["a"].annotations["seg"]
            )
            assert one.to_record().against == "annotations/seg"


class TestWorldUnits:
    """World coordinates are numbers in their grid's units (§3.5) and are
    compared in one convention only (§3.3 rule 4).  Read as they were, one
    frame's boxes in metres against millimetres scored an F1 of 0
    where they were the same boxes, and in LPS against RAS an IoU of 1 where
    they were disjoint."""

    @staticmethod
    def _raters(path: Path) -> Path:
        box = np.array([Framed.box(2, 6)], np.float32)
        with Framed.writer(path) as w:
            for grid, options in (
                ("m", {"spacing": (0.002, 0.001, 0.001), "units": "m"}),
                ("ras", {"spacing": (2.0, 1.0, 1.0), "coord_system": "RAS"}),
            ):
                w.add_grid(grid, shape=Framed.SHAPE, frame_uid="1.2.3.4", **options)
            w.add_boxes("mm", box, ["c1"], grid="g", space="world")
            w.add_boxes("metres", box / 1000, ["c1"], grid="m", space="world")
            # The same numbers in metres: 2 to 6 m, nowhere near 2 to 6 mm.
            w.add_boxes("far", box, ["c1"], grid="m", space="world")
            # In RAS the same numbers are elsewhere, and the same place is
            # other numbers: neither is compared.
            w.add_boxes("ras", box, ["c1"], grid="ras", space="world")
            flipped = np.array([[[-6, -2], [-6, -2], [2, 6]]], np.float32)
            w.add_boxes("ras_same", flipped, ["c1"], grid="ras", space="world")
        return path

    def test_S3_5_one_frames_boxes_compare_in_one_unit(self, tmp_path: Path):
        with medh5.open(self._raters(tmp_path / "units.medh5")) as s:
            mm, metres, far = (s.annotations[k] for k in ("mm", "metres", "far"))
            assert compare_instances(mm, metres).value == pytest.approx(1.0)
            assert compare_instances(metres, mm).value == pytest.approx(1.0)
            assert compare_instances(mm, far).value == 0.0
            assert compare_instances(far, mm).value == 0.0
            assert compare(mm, metres).value == pytest.approx(1.0)

    def test_S3_3_rule_4_two_conventions_are_refused(self, tmp_path: Path):
        with medh5.open(self._raters(tmp_path / "conventions.medh5")) as s:
            mm = s.annotations["mm"]
            for other in ("ras", "ras_same"):
                for pair in ((mm, s.annotations[other]), (s.annotations[other], mm)):
                    with pytest.raises(
                        MEDH5ValidationError, match="coord_system"
                    ) as caught:
                        compare_instances(*pair)
                    assert caught.value.code == "E414"
            # One grid's boxes still compare as they are.
            assert compare_instances(
                s.annotations["ras"], s.annotations["ras"]
            ).value == pytest.approx(1.0)


class TestClassMeaningsAcrossFiles:
    """A class id is a number within its own label set: compared across two
    files, "liver" in one was scored against "spleen" in the other at a Dice
    of 1.  An id the two label sets give two meanings is refused with C102,
    the code ``dataset check`` gives the same clash in a cohort."""

    @staticmethod
    def _rater(path: Path, classes: dict[int, str] | None, found: int) -> Path:
        """Class *found* drawn at one place, as a mask and as a box."""
        mask = np.zeros(Framed.SHAPE, bool)
        mask[2:4, 2:5, 2:5] = True
        with medh5.create(path, sample_id=path.stem, codec="portable") as w:
            w.add_grid(
                "g", shape=Framed.SHAPE, spacing=(2.0, 1.0, 1.0), frame_uid="1.2.3.4"
            )
            w.add_image("CT", np.zeros(Framed.SHAPE, np.int16), grid="g", modality="CT")
            if classes is not None:
                named = [LabelClass(i, k, k.title()) for i, k in classes.items()]
                w.label_set(LabelSet("organs", version="1.0.0", classes=named))
            w.add_segmentation(
                "seg", grid="g", masks={found: mask}, annotated_classes=[found]
            )
            w.add_boxes(
                "boxes",
                np.array([Framed.box(2, 4)], np.float32),
                [found],
                grid="g",
                annotated_classes=[found],
            )
        return path

    def test_C102_an_id_meaning_two_classes_is_refused_not_scored(self, tmp_path: Path):
        a = self._rater(tmp_path / "a.medh5", {1: "liver", 2: "spleen"}, 1)
        b = self._rater(tmp_path / "b.medh5", {1: "spleen", 2: "liver"}, 1)
        with medh5.open(a) as sa, medh5.open(b) as sb:
            for name, compare_of in (
                ("seg", compare_voxel),
                ("boxes", compare_instances),
                ("seg", compare),
                ("boxes", compare),
            ):
                with pytest.raises(MEDH5ValidationError) as caught:
                    compare_of(sa.annotations[name], sb.annotations[name])
                assert caught.value.code == "C102"
                message = str(caught.value)
                assert "1 is 'liver' in the first and 'spleen' in the second" in message
                assert "a.medh5" in message and "b.medh5" in message

    def test_C102_a_class_name_is_one_id_on_both_sides(self, tmp_path: Path):
        """'liver' is 1 in one file and 2 in the other: by name, the two sides
        name two ids, and agreement compares one."""
        a = self._rater(tmp_path / "a.medh5", {1: "liver", 2: "spleen"}, 1)
        b = self._rater(tmp_path / "b.medh5", {1: "spleen", 2: "liver"}, 2)
        with medh5.open(a) as sa, medh5.open(b) as sb:
            for name, compare_of in (
                ("seg", compare_voxel),
                ("boxes", compare_instances),
            ):
                with pytest.raises(MEDH5ValidationError) as caught:
                    compare_of(
                        sa.annotations[name], sb.annotations[name], classes=["liver"]
                    )
                assert caught.value.code == "C102"
                assert "'liver' is id 1" in str(caught.value)
            # By id, each side has examined a class the other has not: an
            # id only `b` uses is named by `b`'s label set.
            result = compare_voxel(sa.annotations["seg"], sb.annotations["seg"])
            assert result.value is None
            assert result.skipped == ("liver (not examined by both)",)

    def test_C102_label_sets_that_agree_on_the_class_compare_it(self, tmp_path: Path):
        """A later version of the vocabulary that adds a class: the ids both
        files use mean the same, and are scored."""
        a = self._rater(tmp_path / "a.medh5", {1: "liver", 2: "spleen"}, 1)
        b = self._rater(tmp_path / "b.medh5", {1: "liver", 2: "spleen", 3: "kidney"}, 1)
        with medh5.open(a) as sa, medh5.open(b) as sb:
            seg = (sa.annotations["seg"], sb.annotations["seg"])
            assert compare_voxel(*seg).value == 1.0
            assert compare_voxel(*seg, classes=["liver"]).per_class == {"liver": 1.0}
            found = compare_instances(sa.annotations["boxes"], sb.annotations["boxes"])
            assert found.value == pytest.approx(1.0)
        # A class one vocabulary lacks is one the other side never examined.
        c = self._rater(tmp_path / "c.medh5", {1: "liver"}, 1)
        with medh5.open(a) as sa, medh5.open(c) as sc:
            seg = (sa.annotations["seg"], sc.annotations["seg"])
            assert compare_voxel(*seg).value == 1.0
            spleen = compare_voxel(*seg, classes=["spleen"])
            assert spleen.value is None
            assert spleen.skipped == ("spleen (not examined by both)",)

    def test_without_a_label_set_an_id_is_compared_as_a_number(self, tmp_path: Path):
        a = self._rater(tmp_path / "a.medh5", None, 1)
        b = self._rater(tmp_path / "b.medh5", None, 1)
        labelled = self._rater(tmp_path / "c.medh5", {1: "liver"}, 1)
        with medh5.open(a) as sa, medh5.open(b) as sb, medh5.open(labelled) as sc:
            for other in (sb, sc):
                seg = compare_voxel(sa.annotations["seg"], other.annotations["seg"])
                assert seg.value == 1.0
                found = compare_instances(
                    sa.annotations["boxes"], other.annotations["boxes"]
                )
                assert found.value == pytest.approx(1.0)


class TestClassesAcrossVocabularies:
    """§11.2: two raters' files often number one vocabulary differently.
    Compared by id, "liver" in one was scored against "spleen" in the other,
    or refused with C102; a class map says which class of ``b`` each class
    of ``a`` is scored against."""

    @staticmethod
    def _rater(
        path: Path,
        classes: list[LabelClass] | None,
        found: int,
        *,
        reference: bool = False,
    ) -> Path:
        """Class *found* drawn at one place, as a mask and as a box."""
        mask = np.zeros(Framed.SHAPE, bool)
        mask[2:4, 2:5, 2:5] = True
        with medh5.create(path, sample_id=path.stem, codec="portable") as w:
            w.add_grid(
                "g", shape=Framed.SHAPE, spacing=(2.0, 1.0, 1.0), frame_uid="1.2.3.4"
            )
            w.add_image("CT", np.zeros(Framed.SHAPE, np.int16), grid="g", modality="CT")
            if classes is not None:
                labels = LabelSet("organs", version="1.0.0", classes=classes)
                w.label_set(
                    labels.as_ref("https://example.org/organs") if reference else labels
                )
            w.add_segmentation(
                "seg", grid="g", masks={found: mask}, annotated_classes=[found]
            )
            w.add_boxes(
                "boxes",
                np.array([Framed.box(2, 4)], np.float32),
                [found],
                grid="g",
                annotated_classes=[found],
            )
        return path

    @staticmethod
    def _organs(*keys: str) -> list[LabelClass]:
        return [LabelClass(i, k, k.title()) for i, k in enumerate(keys, start=1)]

    def test_S11_2_key_mapping_scores_liver_against_liver_across_vocabularies(
        self, tmp_path: Path
    ):
        a = self._rater(tmp_path / "a.medh5", self._organs("liver", "spleen"), 1)
        b = self._rater(tmp_path / "b.medh5", self._organs("spleen", "liver"), 2)
        with medh5.open(a) as sa, medh5.open(b) as sb:
            seg = (sa.annotations["seg"], sb.annotations["seg"])
            with pytest.raises(MEDH5ValidationError) as caught:
                compare_voxel(*seg, classes=["liver"])
            assert caught.value.code == "C102"
            voxel = compare_voxel(*seg, map_classes="key")
            assert voxel.per_class == {"liver": 1.0}
            assert voxel.b_class_ids == {"liver": 2}
            assert voxel.skipped == ()
            boxes = (sa.annotations["boxes"], sb.annotations["boxes"])
            found = compare_instances(*boxes, map_classes="key")
            assert found.value == pytest.approx(1.0)
            assert found.b_class_ids == {"liver": 2}
            kind = compare(*boxes, map_classes="key")
            assert kind.value == pytest.approx(1.0)
            # The same pairing, stated: a's liver is b's class 2.
            explicit = compare_voxel(*seg, map_classes={"liver": 2})
            assert explicit.per_class == {"liver": 1.0}

    def test_S11_2_a_key_missing_from_b_is_skipped_not_zero(self, tmp_path: Path):
        a = self._rater(tmp_path / "a.medh5", self._organs("liver", "spleen"), 2)
        b = self._rater(tmp_path / "b.medh5", self._organs("liver"), 1)
        with medh5.open(a) as sa, medh5.open(b) as sb:
            result = compare_voxel(
                sa.annotations["seg"], sb.annotations["seg"], map_classes="key"
            )
            assert result.value is None
            assert result.per_class == {}
            assert len(result.skipped) == 1
            assert result.skipped[0].startswith("spleen (not in annotation 'seg'")
            assert "b.medh5" in result.skipped[0]

    def test_S11_2_one_key_with_two_codes_is_C102(self, tmp_path: Path):
        def liver(code: str) -> list[LabelClass]:
            return [
                LabelClass(
                    1, "liver", "Liver", codes=(OntologyCode("SNOMED-CT", code),)
                )
            ]

        a = self._rater(tmp_path / "a.medh5", liver("10200004"), 1)
        b = self._rater(tmp_path / "b.medh5", liver("181268008"), 1)
        same = self._rater(tmp_path / "c.medh5", liver("10200004"), 1)
        with medh5.open(a) as sa, medh5.open(b) as sb, medh5.open(same) as sc:
            with pytest.raises(MEDH5ValidationError) as caught:
                compare_voxel(
                    sa.annotations["seg"], sb.annotations["seg"], map_classes="key"
                )
            assert caught.value.code == "C102"
            assert "'liver'" in str(caught.value)
            agreed = compare_voxel(
                sa.annotations["seg"], sc.annotations["seg"], map_classes="key"
            )
            assert agreed.value == 1.0

    def test_S11_2_codes_are_compared_both_ways(self, tmp_path: Path):
        """A key whose codes in one ontology differ on the two sides is C102
        whichever side holds the extra code: the check read only b's codes
        against a's, so swapping the annotations changed the answer."""
        snomed = ("10200004", "181268008")
        both = [
            LabelClass(
                1,
                "liver",
                "Liver",
                codes=tuple(OntologyCode("SNOMED-CT", c) for c in snomed),
            )
        ]
        one = [
            LabelClass(
                1, "liver", "Liver", codes=(OntologyCode("SNOMED-CT", snomed[0]),)
            )
        ]
        a = self._rater(tmp_path / "a.medh5", both, 1)
        b = self._rater(tmp_path / "b.medh5", one, 1)
        with medh5.open(a) as sa, medh5.open(b) as sb:
            seg = (sa.annotations["seg"], sb.annotations["seg"])
            for x, y in (seg, seg[::-1]):
                with pytest.raises(MEDH5ValidationError) as caught:
                    compare_voxel(x, y, map_classes="key")
                assert caught.value.code == "C102"

    def test_S11_2_an_explicit_map_must_be_injective(self, tmp_path: Path):
        a = self._rater(tmp_path / "a.medh5", self._organs("liver", "spleen"), 1)
        b = self._rater(tmp_path / "b.medh5", self._organs("organ"), 1)
        with (
            medh5.open(a) as sa,
            medh5.open(b) as sb,
            pytest.raises(MEDH5ValidationError, match="one to one"),
        ):
            compare_voxel(
                sa.annotations["seg"],
                sb.annotations["seg"],
                map_classes={"liver": 1, "spleen": 1},
            )

    def test_S11_2_a_label_set_by_reference_refuses_key_mapping(self, tmp_path: Path):
        organs = self._organs("liver", "spleen")
        a = self._rater(tmp_path / "a.medh5", organs, 1)
        b = self._rater(tmp_path / "b.medh5", organs, 1, reference=True)
        with medh5.open(a) as sa, medh5.open(b) as sb:
            seg = (sa.annotations["seg"], sb.annotations["seg"])
            with pytest.raises(MEDH5ValidationError, match="label set by reference"):
                compare_voxel(*seg, map_classes="key")
            # By id, or by an explicit map, the reference is not read.
            assert compare_voxel(*seg).value == 1.0
            assert compare_voxel(*seg, map_classes={1: 1}).value == 1.0

    def test_S11_2_skipped_names_each_class_once(self, tmp_path: Path):
        """An id each side names differently was listed once per side."""
        a = self._rater(tmp_path / "a.medh5", self._organs("liver", "spleen"), 1)
        b = self._rater(tmp_path / "b.medh5", self._organs("spleen", "liver"), 2)
        with medh5.open(a) as sa, medh5.open(b) as sb:
            for name in ("seg", "boxes"):
                result = compare(sa.annotations[name], sb.annotations[name])
                assert len(result.skipped) == len(set(result.skipped))


class TestInstanceIdsAreSampleScoped:
    """§7.4: an `instance_id` names an object within its own sample.  Two
    raters' files that each number from 1 matched disjoint lesions by id,
    at an F1 of 1 and a mean IoU of 0."""

    @staticmethod
    def _rater(path: Path, places: list[tuple[int, int]], ids: list[int]) -> Path:
        with Framed.writer(path) as w:
            w.add_boxes(
                "boxes",
                np.array([Framed.box(lo, hi) for lo, hi in places], np.float32),
                ["c1"] * len(places),
                grid="g",
                instance_ids=ids,
            )
        return path

    def test_S7_4_two_files_objects_match_by_iou(self, tmp_path: Path):
        a = self._rater(tmp_path / "a.medh5", [(0, 1), (2, 3)], [1, 2])
        b = self._rater(tmp_path / "b.medh5", [(4, 5), (6, 7)], [1, 2])
        with medh5.open(a) as sa, medh5.open(b) as sb:
            result = compare_instances(sa.annotations["boxes"], sb.annotations["boxes"])
            assert result.matched_by == "iou"
            assert result.value == 0.0
            assert result.mean_iou is None
            assert (result.only_in_a, result.only_in_b) == ((0, 1), (0, 1))
        # The same objects, numbered apart: matched by where they are.
        c = self._rater(tmp_path / "c.medh5", [(0, 1), (2, 3)], [1, 3])
        with medh5.open(a) as sa, medh5.open(c) as sc:
            result = compare_instances(sa.annotations["boxes"], sc.annotations["boxes"])
            assert result.matched_by == "iou"
            assert result.value == pytest.approx(1.0)
            assert [pair[:2] for pair in result.matched] == [(0, 0), (1, 1)]

    def test_S7_4_within_one_sample_an_id_is_the_object(self, tmp_path: Path):
        """Within one sample ids state the correspondence, and an id-matched
        pair is paired whatever its IoU: its report gives no threshold."""
        path = tmp_path / "one.medh5"
        with Framed.writer(path) as w:
            for name, places, ids in (
                ("r1", [(0, 1), (2, 3)], [1, 2]),
                ("r2", [(0, 1), (2, 3)], [1, 3]),
            ):
                w.add_boxes(
                    name,
                    np.array([Framed.box(lo, hi) for lo, hi in places], np.float32),
                    ["c1", "c1"],
                    grid="g",
                    instance_ids=ids,
                )
        with medh5.open(path) as s:
            result = compare_instances(s.annotations["r1"], s.annotations["r2"])
            assert result.matched_by == "instance_id"
            assert result.value == pytest.approx(0.5)
            assert result.to_json()["threshold"] is None
        # One file opened twice is still one sample, whose ids are one set.
        with medh5.open(path) as first, medh5.open(path) as second:
            again = compare_instances(first.annotations["r1"], second.annotations["r2"])
            assert again.matched_by == "instance_id"
            assert again.value == pytest.approx(0.5)
            assert again.to_record().against == "annotations/r2"

    def test_the_command_line_reports_no_threshold_for_an_id_match(
        self, tmp_path: Path, capsys
    ):
        from medh5.cli import main

        path = tmp_path / "one.medh5"
        with Framed.writer(path) as w:
            for name in ("r1", "r2"):
                w.add_boxes(
                    name,
                    np.array([Framed.box(0, 2)], np.float32),
                    ["c1"],
                    grid="g",
                    instance_ids=[1],
                )
        assert main(["agree", str(path), "r1", "r2", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["matched_by"] == "instance_id" and report["threshold"] is None

    def test_S7_4_match_by_instance_id_refuses_two_samples(self, tmp_path: Path):
        a = self._rater(tmp_path / "a.medh5", [(0, 1)], [1])
        b = self._rater(tmp_path / "b.medh5", [(0, 1)], [1])
        with medh5.open(a) as sa, medh5.open(b) as sb:
            boxes = (sa.annotations["boxes"], sb.annotations["boxes"])
            with pytest.raises(MEDH5ValidationError, match="its own sample"):
                compare_instances(*boxes, match_by="instance_id")
            assert compare_instances(*boxes, match_by="iou").matched_by == "iou"
            with pytest.raises(ValueError, match="match_by"):
                compare_instances(*boxes, match_by="name")

    def test_S7_4_match_by_iou_ignores_shared_ids(self, tmp_path: Path):
        """Within one sample the ids pair r1's second object with nothing;
        matched by where they are, both objects pair."""
        path = tmp_path / "one.medh5"
        with Framed.writer(path) as w:
            for name, ids in (("r1", [1, 2]), ("r2", [1, 3])):
                w.add_boxes(
                    name,
                    np.array([Framed.box(0, 1), Framed.box(2, 3)], np.float32),
                    ["c1", "c1"],
                    grid="g",
                    instance_ids=ids,
                )
        with medh5.open(path) as s:
            boxes = (s.annotations["r1"], s.annotations["r2"])
            assert compare_instances(*boxes).value == pytest.approx(0.5)
            by_id = compare_instances(*boxes, match_by="instance_id")
            assert by_id.matched_by == "instance_id"
            assert by_id.value == pytest.approx(0.5)
            by_place = compare_instances(*boxes, match_by="iou")
            assert by_place.matched_by == "iou"
            assert by_place.value == pytest.approx(1.0)
            assert by_place.to_json()["threshold"] == pytest.approx(0.5)

    def test_S11_2_the_command_line_maps_classes_and_pairs_by_place(
        self, tmp_path: Path, capsys
    ):
        from medh5.cli import main

        path = tmp_path / "one.medh5"
        with Framed.writer(path) as w:
            for name, label, ids in (("r1", "c1", [1, 2]), ("r2", "c2", [1, 3])):
                w.add_boxes(
                    name,
                    np.array([Framed.box(0, 1), Framed.box(2, 3)], np.float32),
                    [label, label],
                    grid="g",
                    instance_ids=ids,
                )
        args = ["agree", str(path), "r1", "r2", "--json", "--map-class", "c1=c2"]
        assert main(args) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["matched_by"] == "instance_id"
        assert report["value"] == pytest.approx(0.5)
        assert main([*args, "--match-by", "iou"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["matched_by"] == "iou"
        assert report["value"] == pytest.approx(1.0)
        # Unmapped, c1 and c2 are two classes, each examined by one rater.
        assert main(["agree", str(path), "r1", "r2", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["class_mismatches"] == [[1, 1, 2]]
        assert report["skipped"] == [
            "c1 (not examined by both)",
            "c2 (not examined by both)",
        ]


class TestBoxesOfOneSpace:
    """§8.1: IoU is a ratio of two volumes of one space.  A 2-D box against
    a 3-D one scored about 10, as an area over a volume, and matched; a box of
    no axes scored 1; a NaN, infinite or inverted bound scored as a box."""

    def test_S8_1_box_iou_refuses_what_is_not_two_boxes_of_one_space(self):
        plane = np.array([[0.0, 1.0], [0.0, 1.0]])
        volume = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 0.1]])
        for pair in ((plane, volume), (volume, plane)):
            with pytest.raises(ValueError, match="not in one space"):
                box_iou(*pair)
        with pytest.raises(ValueError, match="at least one axis"):
            box_iou(np.zeros((0, 2)), np.zeros((0, 2)))
        for bad in ([np.nan, 1.0], [0.0, np.inf], [-np.inf, 0.0], [1.0, 0.0]):
            broken = np.array([bad, [0.0, 1.0]])
            for pair in ((broken, plane), (plane, broken)):
                with pytest.raises(MEDH5ValidationError) as caught:
                    box_iou(*pair)
                assert caught.value.code == "E406"
        assert box_iou(volume, volume) == pytest.approx(1.0)
        flat = np.array([[0.0, 0.0], [0.0, 1.0]])
        assert box_iou(flat, flat) == 0.0

    def test_E414_boxes_of_two_and_three_axes_are_not_compared(self, tmp_path: Path):
        path = tmp_path / "mixed.medh5"
        with medh5.create(path, sample_id="mixed", codec="portable") as w:
            w.add_grid("g2", shape=(4, 4), spacing=(1.0, 1.0), frame_uid="1.2.3.4")
            w.add_grid(
                "g3", shape=(4, 4, 4), spacing=(1.0, 1.0, 1.0), frame_uid="1.2.3.4"
            )
            w.add_image("XR", np.zeros((4, 4), np.int16), grid="g2", modality="DX")
            w.add_image("CT", np.zeros((4, 4, 4), np.int16), grid="g3", modality="CT")
            w.label_set(Framed.label_set())
            w.add_boxes(
                "plane",
                np.array([[[0, 1], [0, 1]]], np.float32),
                ["c1"],
                grid="g2",
                space="world",
            )
            w.add_boxes(
                "volume",
                np.array([[[0, 1], [0, 1], [0, 0.1]]], np.float32),
                ["c1"],
                grid="g3",
                space="world",
            )
        with medh5.open(path) as s:
            plane, volume = s.annotations["plane"], s.annotations["volume"]
            assert not s.grids["g2"].comparable_with(s.grids["g3"])
            for pair, axes in (
                ((plane, volume), "2 and 3"),
                ((volume, plane), "3 and 2"),
            ):
                for compare_of in (compare_instances, compare):
                    with pytest.raises(MEDH5ValidationError) as caught:
                        compare_of(*pair)
                    assert caught.value.code == "E414"
                    assert f"boxes of {axes} spatial axes" in str(caught.value)

    def test_E406_an_inverted_box_in_a_file_is_named_not_scored(self, tmp_path: Path):
        """A box the writer would refuse, planted: refused by the comparison,
        which names the object, rather than scored as the box of its
        inversion."""
        import h5py

        path = tmp_path / "planted.medh5"
        with Framed.writer(path) as w:
            for name in ("r1", "r2"):
                w.add_boxes(
                    name, np.array([Framed.box(0, 2)], np.float32), ["c1"], grid="g"
                )
        with h5py.File(path, "r+") as f:
            boxes = f["annotations/r2/boxes"]
            planted = boxes[...]
            planted[0, :2] = planted[0, :2, ::-1]
            boxes[...] = planted
        with medh5.open(path) as s, pytest.raises(MEDH5ValidationError) as caught:
            compare_instances(s.annotations["r1"], s.annotations["r2"])
        assert caught.value.code == "E406"
        assert "annotation 'r2': object 0" in str(caught.value)


class TestVoxelCountsReadTogether:
    """Every class is counted from one read of each side, and agrees with the
    per-class Dice of the masks themselves."""

    def test_S11_2_many_classes_score_as_their_masks_do(self, tmp_path: Path):
        rng = np.random.default_rng(7)
        classes = 12
        first = rng.integers(0, classes + 1, Framed.SHAPE)
        second = np.where(
            rng.random(Framed.SHAPE) < 0.3,
            rng.integers(0, classes + 1, Framed.SHAPE),
            first,
        )
        ignore = np.zeros(Framed.SHAPE, bool)
        ignore[:, :, :3] = True
        path = tmp_path / "many.medh5"
        with Framed.writer(path) as w:
            w.label_set(Framed.label_set(classes))
            for name, labels, encoding, ignored in (
                ("a", first, "labelmap", ignore),
                ("b", second, "layers", None),
            ):
                w.add_segmentation(
                    name,
                    grid="g",
                    masks={c: labels == c for c in range(1, classes + 1)},
                    encoding=encoding,
                    ignore=ignored,
                )
        with medh5.open(path) as s:
            a, b = s.annotations["a"], s.annotations["b"]
            dice_of = compare_voxel(a, b)
            iou_of = compare_voxel(b, a, metric="iou")
        kept = ~ignore
        for c in range(1, classes + 1):
            x, y = (first == c) & kept, (second == c) & kept
            both = int((x & y).sum())
            assert dice_of.per_class[f"c{c}"] == pytest.approx(
                2 * both / (x.sum() + y.sum())
            )
            assert iou_of.per_class[f"c{c}"] == pytest.approx(both / (x | y).sum())
