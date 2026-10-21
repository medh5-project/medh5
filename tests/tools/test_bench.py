"""``medh5 bench``: the measurements, and that each measures what it names."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

import medh5
from medh5.validate import validate_file
from tests.helpers import encode_attr


class TestBench:
    def test_the_metrics_run_and_are_reported(self, indexed_cohort):
        from medh5.bench import TARGETS, benchmark_file, report

        measurements = benchmark_file(indexed_cohort[0], patch=8, repeats=2)
        names = {m.name for m in measurements}
        # The many-class row is measured on the 63-class sample `medh5 bench`
        # writes for it, not on a file passed in.
        assert set(TARGETS) - {"foreground_sample_many_ms"} <= names
        assert all(m.value >= 0 for m in measurements)
        missed = [m.name for m in measurements if not m.ok]
        verdict = report(measurements).splitlines()[-1]
        assert verdict == (
            f"{len(missed)} metric(s) below target: {', '.join(missed)}"
            if missed
            else "all targets met"
        )

    def test_a_measurement_knows_whether_it_met_its_target(self):
        from medh5.bench import Measurement, report

        assert Measurement("x", 1.0, target=2.0).ok
        assert not Measurement("x", 3.0, target=2.0).ok
        assert Measurement("x", 3.0).ok
        assert "!" in str(Measurement("x", 3.0, target=2.0))
        assert Measurement("x", 1.0).to_json()["ok"] is True
        met = [Measurement("a", 1.0, target=2.0), Measurement("b", 9.0)]
        assert report(met).splitlines()[-1] == "all targets met"
        missed = [*met, Measurement("c", 3.0, target=2.0)]
        assert report(missed).splitlines()[-1] == "1 metric(s) below target: c"

    def test_S14_3_the_draw_row_says_what_the_sampler_used(self, indexed_cohort):
        """``used_index`` is the sampler's own answer.  The row reported an
        index entry's existence, so a stale entry, which the sampler scans
        past, was reported as used."""
        from medh5.bench import benchmark_file

        def used(path: Path) -> object:
            rows = {m.name: m for m in benchmark_file(path, patch=8, repeats=1)}
            detail = rows["foreground_sample_ms"].detail or {}
            return detail["used_index"]

        path = indexed_cohort[0]
        assert used(path) is True
        with h5py.File(path, "r+") as handle:
            handle["index/organs_tp0"].attrs["source_digest"] = encode_attr(
                "sha256:" + "0" * 64
            )
        assert used(path) is False

    def test_timed_returns_a_median(self):
        from medh5.bench import timed
        from medh5.errors import MEDH5ValidationError

        assert timed(lambda: None, repeats=3, warmup=1) >= 0.0
        assert timed(lambda: None, repeats=1, warmup=0) >= 0.0
        with pytest.raises(MEDH5ValidationError, match="^warmup must be"):
            timed(lambda: None, warmup=-1)

    @pytest.mark.parametrize(
        ("patch", "repeats", "refused"),
        [(8, 0, "repeats"), (0, 2, "patch"), (8, -1, "repeats"), (-8, 2, "patch")],
    )
    def test_S14_a_run_that_measures_nothing_is_refused(
        self, indexed_cohort, patch, repeats, refused
    ):
        """The command line refused `--repeats 0`, and the Python API ran it:
        no run has no median, so the row was NaN, which the report could not
        print and `benchmark_file` could not read back (`TypeError`).  A
        patch of side 0 timed reads of nothing.  A negative count raised
        PyO3's `OverflowError`; it is refused as 0 is."""
        from medh5.bench import benchmark_file, many_class_measurement, timed
        from medh5.errors import MEDH5ValidationError

        for measure in (benchmark_file, many_class_measurement):
            with pytest.raises(MEDH5ValidationError, match=f"^{refused} must be"):
                measure(indexed_cohort[0], patch=patch, repeats=repeats)
        if repeats <= 0:
            ran = []
            with pytest.raises(MEDH5ValidationError, match="^repeats must be"):
                timed(lambda: ran.append(1), repeats=repeats)
            assert ran == []

    def test_the_throughput_run_measures_steady_state(self, indexed_cohort):
        pytest.importorskip("torch")  # the run goes through the real DataLoader
        from medh5.bench import throughput

        measured = throughput(
            indexed_cohort, patch=8, batches=2, batch_size=1, workers=0
        )
        assert measured.unit == "patches/s"
        assert measured.value > 0
        assert measured.detail["workers"] == 0

    def test_bench_measures_a_paired_centre(self, tmp_path: Path):
        from medh5.bench import benchmark_file, synthetic_pair

        pair = synthetic_pair(tmp_path, shape=(6, 8, 8))
        names = [m.name for m in benchmark_file(pair, patch=4, repeats=2)]
        assert "paired_center_ms" in names

    def test_the_bench_has_a_many_class_row(self, tmp_path: Path):
        from medh5.bench import (
            MANY_CLASSES,
            TARGETS,
            many_class_measurement,
            synthetic_sample,
        )

        path = synthetic_sample(
            tmp_path, shape=(16, 32, 32), classes=MANY_CLASSES, name="many.medh5"
        )
        measured = many_class_measurement(path, patch=8, repeats=3)
        assert measured.name == "foreground_sample_many_ms"
        assert measured.target == TARGETS["foreground_sample_many_ms"][0]
        assert measured.detail == {"classes": MANY_CLASSES, "used_index": True}

    def test_bench_json_is_only_the_document(self, indexed_cohort, capsys):
        """Progress lines on stdout ahead of the JSON made `--json` unparsable.

        Measured on a file: without one, `bench` writes its synthetic samples
        at their published size, which made this the slowest test of the
        suite.  The command line's own test of it
        (`s14_bench_json_is_only_the_document`) runs that path, progress lines
        and all, on samples a fraction of the size."""
        from medh5.cli import main

        path = str(indexed_cohort[0])
        argv = ["bench", path, "--no-throughput", "--json", "--repeats", "1"]
        code = main([*argv, "--patch", "8"])
        out, err = capsys.readouterr()
        payload = json.loads(out)
        assert payload["path"] == path
        names = {m["name"] for m in payload["measurements"]}
        assert {"foreground_sample_ms", "meta_read_ms"} <= names
        assert code == (0 if payload["ok"] else 1)

    def test_bench_measures_throughput_through_the_dataloader(
        self, indexed_cohort, capsys
    ):
        """The command line hands the throughput run to the package it was
        started from, which runs the real DataLoader: the row is measured,
        not skipped."""
        pytest.importorskip("torch")
        from medh5.cli import main

        code = main(
            [
                "bench",
                str(indexed_cohort[0]),
                "--json",
                "--repeats",
                "1",
                "--patch",
                "8",
            ]
        )
        out, err = capsys.readouterr()
        assert "skipping throughput" not in err
        payload = json.loads(out)
        rows = {m["name"]: m for m in payload["measurements"]}
        assert rows["patch_throughput"]["unit"] == "patches/s"
        assert rows["patch_throughput"]["value"] > 0
        assert rows["patch_throughput"]["target"] is None
        assert code == (0 if payload["ok"] else 1)


class TestSyntheticBench:
    def test_the_synthetic_sample_is_valid_and_indexed(self, tmp_path):
        from medh5.bench import synthetic_sample

        path = synthetic_sample(
            tmp_path, shape=(8, 16, 16), classes=2, codec="portable"
        )
        assert not validate_file(path).errors
        with medh5.open(path) as sample:
            assert "training" in sample.profiles
            assert len(sample.annotations["organs"].class_ids) == 2


class TestThreadBench:
    """``docs/examples/bench_threads.py``, run small: every workload at every
    thread count, and the per-call row."""

    def test_the_thread_bench_runs_every_workload(self, tmp_path):
        import importlib.util
        import sys

        script = Path(__file__).parents[2] / "docs" / "examples" / "bench_threads.py"
        spec = importlib.util.spec_from_file_location("bench_threads", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["bench_threads"] = module
        spec.loader.exec_module(module)
        out = tmp_path / "out.json"
        small = "--files 2 --shape 8 16 16 --patch 4 --reads 6 --opens 3 --calls 2"
        results = module.main(
            [*small.split(), "--repeats", "1", "--threads", "1", "2"]
            + ["--out", str(tmp_path / "samples"), "--json", str(out)]
        )
        assert json.loads(out.read_text())["conditions"]["files"] == 2
        for workload in ("image", "portable", "labels", "open"):
            rows = results[workload]["threads"]
            assert set(rows) == {"1", "2"}
            assert rows["1"]["vs_first"] == 1.0
            assert all(row["seconds"] > 0 for row in rows.values())
        assert set(results["calls_us"]) == {"read_one_voxel", "contains"}

    @pytest.mark.parametrize(
        "flags",
        [
            "--shape 8 16 16 --patch 9",
            "--patch 0",
            "--threads 1 0",
            "--reads 0",
            "--files 4 --threads 1 3",
        ],
    )
    def test_a_run_that_would_time_something_else_is_refused(self, flags, capsys):
        import importlib.util

        script = Path(__file__).parents[2] / "docs" / "examples" / "bench_threads.py"
        spec = importlib.util.spec_from_file_location("bench_threads_flags", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with pytest.raises(SystemExit):
            module.main(flags.split())
        assert "--" in capsys.readouterr().err
