"""The task-and-cache contract (``docs/spec/task-cache-1.md``), through Python and
the command line.

Fixtures are written by the public writer (``tests.kits.History``); the
companion files by the public task and cache APIs.  Test names cite the
contract's clause.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5 import _core
from medh5.cache import (
    CacheEntry,
    CacheWriter,
    FeatureCache,
    HashingTextEncoder,
    build_document_cache,
    event_entry_id,
    fitted_on,
    open_cache,
    validate_cache,
)
from medh5.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, main
from medh5.clinical import DAY, HOUR, Clock, Document, Event, Link
from medh5.collection import pack
from medh5.errors import MEDH5ValidationError
from medh5.task import (
    CODES,
    Finding,
    Packed,
    RepinReport,
    Slot,
    SourceRef,
    Target,
    TaskManifest,
    preflight,
    schema_text,
)
from medh5.validate import validate_file
from tests.kits import History

TARGET = Target(
    "progression",
    "1",
    History.RECIST,
    "overall_response",
    positive=("PD",),
    negative=("SD", "PR", "CR"),
    horizon_us=180 * DAY,
    min_follow_up_us=60 * DAY,
)


def cohort(root: Path, responses: dict[str, str] | None = None) -> dict[str, Path]:
    responses = responses or {"P-01": "PD", "P-02": "SD", "P-03": "SD"}
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for subject, response in responses.items():
        paths[subject] = root / f"{subject}.medh5"
        History.write(paths[subject], subject_id=subject, response=response)
    return paths


def task_for(
    root: Path,
    paths: dict[str, Path],
    refs: dict[str, SourceRef] | None = None,
    **options: Any,
) -> TaskManifest:
    """A task over ``paths``, each pinned now --- or, for a subject in
    ``refs``, referenced as given (a pin some earlier tool took)."""
    refs = refs or {}
    task = TaskManifest.new(
        "progression",
        "1",
        identity_namespace="site",
        slots=[Slot("ct", "CT", required=True, patch=(4, 8, 8), classes=(3,))],
        target=TARGET,
        split=("fold-0", ["train", "val"]),
        base=root,
        **options,
    )
    for i, (subject, path) in enumerate(sorted(paths.items())):
        task.add_subject(
            subject,
            [refs.get(subject) or SourceRef.pin(path, uri=path.name)],
            partition="val" if i == len(paths) - 1 else "train",
        )
        task.add_row(f"{subject}@24h", subject, 24 * HOUR)
        task.add_row(f"{subject}@d95", subject, 95 * DAY)
    return task


@pytest.fixture
def setup(tmp_path: Path) -> tuple[TaskManifest, dict[str, Path]]:
    paths = cohort(tmp_path / "cohort")
    return task_for(tmp_path / "cohort", paths), paths


def found(findings: Any) -> list[str]:
    return sorted({f.code for f in findings})


def damage(group: Any, name: str) -> None:
    """Re-store ``group[name]`` chunked and compressed --- the same values, so
    the same digest and ``content_id`` --- and damage its chunk: its bytes no
    longer decompress, and nothing stored says so."""
    values = group[name][...]
    attrs = dict(group[name].attrs)
    del group[name]
    dataset = group.create_dataset(
        name, data=values, chunks=values.shape, compression="gzip"
    )
    dataset.attrs.update(attrs)
    dataset.id.write_direct_chunk((0,) * values.ndim, b"\x00" * 16)


def instance_task(root: Path, paths: dict[str, Path]) -> TaskManifest:
    """``task_for``, its slot centred on the first eligible instance."""
    doc = task_for(root, paths).to_json()
    doc["slots"][0]["roi"] = "eligible_instances"
    return TaskManifest(doc, base=root)


class TestManifest:
    def test_S3_1_build_save_load(self, setup, tmp_path: Path):
        task, _ = setup
        path = task.save(tmp_path / "cohort" / "t.task.json")
        loaded = TaskManifest.load(path)
        assert loaded.declared_fingerprint == task.manifest_fingerprint
        assert loaded.to_json()["subjects"] == task.to_json()["subjects"]
        assert loaded.task_id == "progression" and loaded.task_version == "1"
        assert loaded.identity_namespace == "site"
        assert loaded.policy.selection == "strict_prospective"
        assert loaded.slots[0] == Slot("ct", "CT", True, (4, 8, 8), "center", (3,))
        assert loaded.target == TARGET
        assert loaded.split == ("fold-0", ("train", "val"))
        assert loaded.training_partition == "train"
        assert [s.subject_id for s in loaded.subjects] == ["P-01", "P-02", "P-03"]
        assert len(loaded.rows) == 6 and loaded.partition_of("P-03") == "val"
        assert loaded.validate() == []
        assert "TaskManifest('progression'" in repr(loaded)
        assert '"medh5.task/1"' in schema_text()
        assert set(CODES) >= {"T101", "T302", "T406"}

    def test_S3_1_a_save_replaces_the_manifest_whole(
        self, setup, tmp_path: Path, capsys
    ):
        """A manifest was written in place, so a write cut short --- a full
        disk, a crash --- truncated it; `task reconcile` writes over its input.
        Python's save and the engine's now each replace the file whole."""
        task, _ = setup
        path = task.save(tmp_path / "cohort" / "t.task.json")
        for write in (
            lambda: task.save(path),
            lambda: main(["task", "reconcile", str(path)]),
        ):
            before = path.stat().st_ino
            write()
            assert path.stat().st_ino != before, "a new file, not the old rewritten"
            assert TaskManifest.load(path).validate() == []
        capsys.readouterr()
        assert [p.name for p in path.parent.glob(".*")] == []

    @pytest.mark.parametrize("alias", ["exact", "relative", "symlink", "hardlink"])
    def test_S3_3_save_never_writes_over_a_source(
        self, setup, tmp_path: Path, alias: str
    ):
        """Only the command line refused an output that is a source:
        ``TaskManifest.load(m).reconcile().save("ct.medh5")`` replaced a
        patient's sample with JSON.  Both now refuse, by any path or link."""
        task, paths = setup
        source = paths["P-02"]
        if alias == "exact":
            out = source
        elif alias == "relative":
            (tmp_path / "sub").mkdir()
            out = tmp_path / "sub" / ".." / "cohort" / source.name
        else:
            out = tmp_path / "alias.medh5"
            try:
                (out.symlink_to if alias == "symlink" else out.hardlink_to)(source)
            except OSError as exc:  # pragma: no cover - no links on this file system
                pytest.skip(str(exc))
        before = source.read_bytes()
        loaded = TaskManifest.load(task.save(tmp_path / "cohort" / "t.task.json"))
        for unsaved in (task, loaded):
            with pytest.raises(
                MEDH5ValidationError, match="a source of subject 'P-02'"
            ):
                unsaved.save(out)
        assert source.read_bytes() == before

    def test_S3_3_save_without_a_base_guards_the_working_directory(
        self, tmp_path: Path, monkeypatch
    ):
        """Without a base, a source resolves against the working directory;
        the guard looked only beside the output, so ``save("data/p.medh5")``
        replaced the source ``data/p.medh5`` it pinned."""
        (tmp_path / "data").mkdir()
        source = tmp_path / "data" / "p.medh5"
        History.write(source, subject_id="P-01")
        monkeypatch.chdir(tmp_path)
        task = TaskManifest.new("t", "1", identity_namespace="n")
        task.add_subject("P-01", [SourceRef.pin(source, uri="data/p.medh5")])
        before = source.read_bytes()
        with pytest.raises(MEDH5ValidationError, match="a source of subject 'P-01'"):
            task.save(Path("data") / "p.medh5")
        assert source.read_bytes() == before

    def test_S3_3_both_frontends_reconcile_alike(self, tmp_path: Path):
        """The reconcile loop was written twice: the binding kept the declared
        fingerprint the command line dropped.  Both are now the engine's."""
        root = tmp_path / "f"
        root.mkdir()
        a, b = root / "a.medh5", root / "b.medh5"
        History.write(a, subject_id="P-01")
        History.write(b, subject_id="P-01")
        task = TaskManifest.new("t", "1", identity_namespace="n", base=root)
        task.add_subject(
            "P-01", [SourceRef.pin(a, uri="a.medh5"), SourceRef.pin(b, uri="b.medh5")]
        )
        task.add_row("r", "P-01", 24 * HOUR)
        manifest = task.save(root / "t.task.json")
        assert (
            main(["task", "reconcile", str(manifest), "-o", str(root / "c.json")]) == 0
        )
        python = TaskManifest.load(manifest).reconcile()
        assert len(python.subjects[0].reconciled) == 6
        assert python.validate() == [], "no stale declared fingerprint (T103)"
        python.save(root / "p.json")
        cli = json.loads((root / "c.json").read_text(encoding="utf-8"))
        assert json.loads((root / "p.json").read_text(encoding="utf-8")) == cli
        assert (root / "p.json").read_bytes() == (root / "c.json").read_bytes()

    def test_S3_1_both_frontends_save_one_byte_form(self, setup, tmp_path: Path):
        """The engine wrote ``"M\\u00fcller"``, Python ``"Müller"``: one
        document, two files.  Python now writes the engine's bytes."""
        task, _ = setup
        doc = task.to_json()
        doc["task"]["description"] = "Müller's cohort, 東京"
        doc["identity_namespace"] = "hôpital"
        root = tmp_path / "cohort"
        python = TaskManifest(doc, base=root).save(root / "p.json")
        native = root / "n.json"
        assert main(["task", "reconcile", str(python), "-o", str(native)]) == 0
        assert python.read_bytes() == native.read_bytes()
        assert TaskManifest.load(python).identity_namespace == "hôpital"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX groups")
    def test_S3_1_a_saved_manifest_keeps_its_group(self, setup, tmp_path: Path):
        """A save replaces the file through the engine (1.0 §14.4), which
        carries the replaced file's group: a manifest shared with a project
        group stays that group's."""
        task, _ = setup
        egid = os.getegid()
        others = [g for g in os.getgroups() if g != egid]
        group = others[0] if others else (65534 if os.geteuid() == 0 else None)
        if group is None:  # pragma: no cover - GitHub's runner user has one
            pytest.skip("no group but the primary one")
        path = task.save(tmp_path / "cohort" / "shared.task.json")
        os.chown(path, -1, group)
        path.chmod(0o640)
        task.save(path)
        assert (path.stat().st_gid, stat.S_IMODE(path.stat().st_mode)) == (group, 0o640)

    @pytest.mark.parametrize("spelling", ["NaN", "Infinity", "-Infinity"])
    def test_S3_1_a_manifest_holding_nan_is_not_json(
        self, setup, tmp_path: Path, capsys, spelling: str
    ):
        """Python's ``json.loads`` read ``NaN``; the engine refused it.  JSON
        has none, so both refuse: T101."""
        task, _ = setup
        path = task.save(tmp_path / "cohort" / "t.task.json")
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace('"version": "1"', f'"x": {spelling}, "version": "1"', 1),
            encoding="utf-8",
        )
        with pytest.raises(MEDH5ValidationError) as caught:
            TaskManifest.load(path)
        assert caught.value.code == "T101"
        capsys.readouterr()
        assert main(["task", "validate", str(path)]) == EXIT_ERROR
        assert "T101" in capsys.readouterr().out

    def test_S3_2_fingerprints_follow_the_manifest(self, setup):
        """Computed once until the manifest changes --- each access parsed the
        whole manifest, a row's twice --- and the same values."""
        task, _ = setup
        doc = task.to_json()
        assert task.row_fingerprint("P-02@d95") == _core.task_row_fingerprint(
            doc, "P-02@d95"
        )
        first = task.manifest_fingerprint
        task.add_row("P-01@48h", "P-01", 48 * HOUR)
        assert task.manifest_fingerprint != first
        assert task.row_fingerprint("P-01@48h") == _core.task_row_fingerprint(
            task.to_json(), "P-01@48h"
        )
        assert task.subjects_digest("train") == _core.task_subjects_digest(doc, "train")
        with pytest.raises(KeyError):
            task.row_fingerprint("nope")

    def test_S3_3_default_source_ids_are_valid_and_tell_sites_apart(
        self, setup, tmp_path: Path
    ):
        """A reference built without an id had `""`, which fails the schema,
        and `pin` named one by its file stem alone, so two sites' `P-01.medh5`
        collided (T204).  `validate` raised the schema's T101; it reports it."""
        _, paths = setup
        refs = []
        for site in ("site-a", "site-b"):
            copy = tmp_path / site / "P-01.medh5"
            copy.parent.mkdir()
            shutil.copyfile(paths["P-01"], copy)
            refs.append(SourceRef.pin(copy, uri=f"{site}/P-01.medh5"))
        assert [r.source_id for r in refs] == ["site-a_P-01", "site-b_P-01"]
        bare = SourceRef("P-01.medh5", refs[0].content_id, sample_key="m")
        assert bare.source_id == "P-01:m"
        task = TaskManifest.new("t", "1", identity_namespace="n")
        task.add_subject("P-01", [SourceRef("", refs[0].content_id)])
        (finding,) = task.validate()
        assert finding.code == "T101"

    def test_S2_colliding_default_ids_are_widened(self):
        """``a/site-a/P-01.medh5`` and ``b/site-a/P-01.medh5`` both derived
        ``site-a_P-01``: T204, from ids nobody wrote."""
        task = TaskManifest.new("t", "1", identity_namespace="n")
        pin = "sha256:" + "0" * 64
        refs = [SourceRef(f"{d}/site-a/P-01.medh5", pin) for d in ("a", "b")]
        task.add_subject("P-09", refs)
        ids = [s.source_id for s in task.subjects[-1].sources]
        assert ids == ["site-a_P-01", "b_site-a_P-01"]
        assert task.validate() == []

    def test_S2_default_ids_widen_then_number(self, setup):
        task, _ = setup
        pin = task.subjects[0].sources[0].content_id
        refs = [SourceRef(f"site-a/P-01.{ext}", pin) for ext in ("medh5", "h5")]
        task.add_subject("P-09", refs[:1], partition="train")
        task.add_subject("P-10", refs[1:], partition="train")
        ids = [s.sources[0].source_id for s in task.subjects[-2:]]
        assert ids == ["site-a_P-01", "site-a_P-01@2"]
        # The cut at 128 characters collides too, and is numbered within it.
        long = [SourceRef(f"{'n' * 200}.{e}", pin) for e in ("medh5", "h5")]
        task.add_subject("P-11", long, partition="train")
        assert [s.source_id for s in task.subjects[-1].sources] == [
            "n" * 128,
            "n" * 126 + "@2",
        ]

    def test_S2_every_widened_id_is_taken(self):
        """A widened id was not added to those taken: the third source
        sharing a derived id was widened to the second's, T204 again."""
        task = TaskManifest.new("t", "1", identity_namespace="n")
        pin = "sha256:" + "0" * 64
        exts = ("medh5", "h5", "hdf5")
        task.add_subject("P-09", [SourceRef(f"site-a/P-01.{e}", pin) for e in exts])
        ids = [s.source_id for s in task.subjects[-1].sources]
        assert ids == ["site-a_P-01", "site-a_P-01@2", "site-a_P-01@3"]
        assert task.validate() == []

    def test_S2_an_explicit_colliding_id_is_still_T204(self, setup):
        task, _ = setup
        pin = task.subjects[0].sources[0].content_id
        taken = task.subjects[0].sources[0].source_id
        task.add_subject(
            "P-09",
            [SourceRef("elsewhere.medh5", pin, source_id=taken)],
            partition="train",
        )
        assert "T204" in found(task.validate())

    @pytest.mark.parametrize(
        ("uri", "key", "expected"),
        [
            ("P-01.medh5", None, "P-01"),
            ("site-a/P-01.medh5", None, "site-a_P-01"),
            ("data/site-a/P-01.medh5", None, "site-a_P-01"),
            ("../data/P-01.medh5", None, "data_P-01"),
            ("../P-01.medh5", None, ".._P-01"),
            ("/abs/P 01.medh5", None, "abs_P_01"),
            ("file:///data/site%20b/P-01.medh5", None, "site_b_P-01"),
            ("file:///C:/data/P-01.medh5", None, "data_P-01"),
            ("s3://b/x.medh5", None, "b_x"),
            ("..\\x\\y.medh5c", None, ".._x_y"),
            ("shard.medh5c", "P-01", "shard:P-01"),
            ("n" * 200, None, "n" * 128),
            ("", "k", ":k"),
        ],
    )
    def test_S2_uncolliding_default_ids_are_unchanged(
        self, uri: str, key: str | None, expected: str
    ):
        """What 2.0.2 derived on POSIX, now on every platform: a backslash is
        no separator, so one script's ids, and fingerprints, are one."""
        assert SourceRef(uri, "sha256:" + "0" * 64, key).source_id == expected

    def test_S3_2_the_task_fingerprint_is_the_definition(self, setup):
        task, _ = setup
        spelled = task.to_json()
        # Defaults written out are the same task as defaults left out.
        terse = {k: v for k, v in spelled.items() if k != "policy"}
        assert TaskManifest(terse).task_fingerprint == task.task_fingerprint
        moved = dict(spelled, rows=spelled["rows"][:2])
        assert TaskManifest(moved).task_fingerprint == task.task_fingerprint
        assert TaskManifest(moved).manifest_fingerprint != task.manifest_fingerprint
        other = dict(spelled, policy={"selection": "latest_provable"})
        assert TaskManifest(other).task_fingerprint != task.task_fingerprint
        assert task.row_fingerprint("P-01@24h") != task.row_fingerprint("P-01@d95")
        assert task.subjects_digest("train") != task.subjects_digest("val")

    def test_S3_2_a_wrong_declared_fingerprint_is_T103(self, setup):
        task, _ = setup
        doc = dict(task.to_json(), fingerprint="sha256:" + "0" * 64)
        assert found(TaskManifest(doc).validate()) == ["T103"]

    def test_S3_3_splits_are_by_subject(self, setup):
        task, paths = setup
        doc = task.to_json()
        doc["subjects"][0]["partition"] = "test"
        doc["subjects"][1].pop("partition")
        doc["subjects"].append(dict(doc["subjects"][2]))
        doc["rows"].append({"row_id": "x", "subject_id": "P-99", "cutoff_us": 0})
        doc["rows"].append(dict(doc["rows"][0]))
        codes = found(TaskManifest(doc).validate())
        assert {"T201", "T202", "T204"} <= set(codes)

    def test_S3_3_two_subjects_never_pin_one_sample(self, setup, tmp_path: Path):
        task, paths = setup
        copy = tmp_path / "cohort" / "copy.medh5"
        shutil.copyfile(paths["P-01"], copy)
        task.add_subject(
            "P-04", [SourceRef.pin(copy, uri=copy.name)], partition="train"
        )
        assert "T203" in found(task.validate())

    def test_S3_4_an_inconsistent_definition_is_T102(self, tmp_path: Path):
        bad = Target(
            "t",
            "1",
            "s",
            "c",
            positive=("PD",),
            negative=("PD",),
            horizon_us=1,
            min_follow_up_us=1,
        )
        task = TaskManifest.new(
            "t",
            "1",
            identity_namespace="n",
            slots=[Slot("ct", "CT"), Slot("ct", "MR")],
            target=bad,
        )
        assert found(task.validate()) == ["T102"]
        with pytest.raises(MEDH5ValidationError) as caught:
            TaskManifest.new(
                "t", "1", identity_namespace="n", policy={"keep": "middle"}
            )
        assert caught.value.code in {"T101", "T102"}

    def test_S5_a_target_names_its_horizon(self):
        """`horizon_us` has no default: it defaulted to 0, which the schema
        refuses, so a target built without it failed every manifest (T101)."""
        with pytest.raises(TypeError, match="horizon_us"):
            Target("t", "1", "s", "c", ("PD",))
        target = Target(
            "t", "1", "s", "c", ("PD",), ("SD",), horizon_us=DAY, min_follow_up_us=DAY
        )
        assert Target.from_json(target.to_json()) == target

    def test_S5_a_target_with_negatives_states_its_follow_up(self, setup, tmp_path):
        """The default follow-up, the horizon, leaves a negative only an
        observation at exactly ``c + h``, so nearly every row without a
        positive is censored.  It cannot change under ``/1``, so a target
        built without it warns, and a manifest written without it is noted."""
        with pytest.warns(DeprecationWarning, match="min_follow_up_us: state it"):
            bare = Target("t", "1", "s", "c", ("PD",), ("SD",), horizon_us=DAY)
        stated = Target(
            "t", "1", "s", "c", ("PD",), ("SD",), horizon_us=DAY, min_follow_up_us=1
        )
        positives_only = Target("t", "1", "s", "c", ("PD",), horizon_us=DAY)
        assert Target.from_json(bare.to_json()) == bare, "a document is not warned"
        built = TaskManifest.new("t", "1", identity_namespace="n", target=bare)
        assert [str(n) for n in built.notes] == [
            f"note /target/min_follow_up_us: {built.notes[0].message}"
        ]
        assert "nearly every row" in built.notes[0].message
        assert built.validate() == [], "a note is no finding"
        for target in (stated, positives_only, None):
            task = TaskManifest.new("t", "1", identity_namespace="n", target=target)
            assert task.notes == ()
        task, _ = setup
        doc = task.to_json()
        del doc["target"]["min_follow_up_us"]
        path = tmp_path / "cohort" / "noted.task.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        loaded = TaskManifest.load(path)
        assert [n.location for n in loaded.notes] == ["/target/min_follow_up_us"]
        assert [n.location for n in loaded.reconcile().notes] == [
            "/target/min_follow_up_us"
        ], "the document as written, after a reconcile too"
        assert TaskManifest(loaded.to_json()).notes == (), "the normal form states it"

    def test_S3_2_the_default_follow_up_is_in_the_fingerprint(self, setup):
        """Leaving ``min_follow_up_us`` out is the task that states the
        horizon, and not the task that states half of it: the default is in
        every task fingerprint, which is why ``/1`` cannot move it."""
        task, _ = setup
        doc = task.to_json()
        horizon = doc["target"]["horizon_us"]
        fingerprints = []
        for follow_up in (None, horizon, horizon // 2):
            if follow_up is None:
                del doc["target"]["min_follow_up_us"]
            else:
                doc["target"]["min_follow_up_us"] = follow_up
            fingerprints.append(TaskManifest(doc).task_fingerprint)
        omitted, explicit, half = fingerprints
        assert omitted == explicit != half

    def test_S3_1_python_defaults_are_the_engines(self):
        """``SelectionPolicy()``, ``Slot()`` and ``Target()`` keep their
        Python defaults: each is what the engine fills into a document that
        states only what is required."""
        from medh5.clinical import SelectionPolicy

        bare = {
            "schema": "medh5.task/1",
            "task": {"id": "t", "version": "1"},
            "identity_namespace": "n",
            "slots": [{"name": "ct", "modality": "CT"}],
            "target": {
                "id": "t",
                "version": "1",
                "event": {"code_system": "s", "code": "c"},
                "positive": ["PD"],
                "horizon_us": DAY,
            },
            "subjects": [],
            "rows": [],
        }
        normal = TaskManifest(bare).to_json()
        assert normal["policy"] == SelectionPolicy().to_json()
        assert normal["slots"] == [Slot("ct", "CT").to_json()]
        target = Target("t", "1", "s", "c", ("PD",), horizon_us=DAY).to_json()
        assert normal["target"] == {**target, "min_follow_up_us": DAY}

    def test_S5_a_target_without_negatives_is_the_default(self, setup):
        """`negative` defaults to none: a target that leaves it out is the task
        that writes `[]`.  The schema required it, so the manifest failed
        (T101)."""
        task, _ = setup
        doc = task.to_json()
        doc["target"]["negative"] = []
        explicit = TaskManifest(doc)
        del doc["target"]["negative"]
        implicit = TaskManifest(doc)
        assert implicit.validate() == []
        assert implicit.target is not None and implicit.target.negative == ()
        assert implicit.task_fingerprint == explicit.task_fingerprint

    def test_S3_1_not_json_is_T101(self, tmp_path: Path):
        path = tmp_path / "broken.json"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(MEDH5ValidationError) as caught:
            TaskManifest.load(path)
        assert caught.value.code == "T101"
        with pytest.raises(MEDH5ValidationError) as caught:
            TaskManifest({"schema": "medh5.task/1"})
        assert caught.value.code == "T101"

    def test_a_manifest_pickles(self, setup):
        task, _ = setup
        back = pickle.loads(pickle.dumps(task))
        assert back.manifest_fingerprint == task.manifest_fingerprint
        assert back.base == task.base


class TestVocabularies:
    def test_constants_are_the_engines(self):
        """The preflight vocabularies were spelled in the binding while the
        engine wrote literals, and five more in Python: each is now the
        engine's table, bound as is."""
        import medh5.cache
        import medh5.clinical
        import medh5.task

        for module, name, core in [
            (medh5.task, "ROW_STATUSES", "ROW_STATUSES"),
            (medh5.task, "SELECTION_STATUSES", "SELECTION_STATUSES"),
            (medh5.task, "TARGET_STATUSES", "TARGET_STATUSES"),
            (medh5.task, "ROIS", "ROIS"),
            (medh5.task, "ROI", "TASK_ROI"),
            (medh5.task, "CENSORING", "TASK_CENSORING"),
            (medh5.cache, "LEVELS", "CACHE_LEVELS"),
            (medh5.cache, "SUFFIX", "CACHE_SUFFIX"),
            (medh5.clinical, "ASSESSMENT_SYSTEM", "ASSESSMENT_SYSTEM"),
            (medh5.clinical, "LESION_PRESENCE", "LESION_PRESENCE"),
        ]:
            assert getattr(module, name) is getattr(_core, core), name
        # Two ROI vocabularies: what a slot may ask for, and what a fill reports.
        assert set(medh5.task.ROI) < set(medh5.task.ROIS)
        assert "center_fallback" in medh5.task.ROIS

    @pytest.mark.parametrize(
        ("name", "now"), [("STATUSES", "ROW_STATUSES"), ("LABELS", "TARGET_STATUSES")]
    )
    def test_the_old_names_warn(self, name: str, now: str):
        """``task.STATUSES`` and ``clinical.STATUSES`` named two vocabularies."""
        import medh5.task

        assert name not in medh5.task.__all__
        with pytest.warns(DeprecationWarning, match=f"use {now}"):
            old = getattr(medh5.task, name)
        assert old is getattr(medh5.task, now)


class TestConceptToken:
    def test_tokens_escape_their_delimiter_and_keep_the_plain_ones(self):
        from medh5.task import concept_token

        assert concept_token("observation", "http://loinc.org", "2160-0") == (
            "observation|http://loinc.org|2160-0"
        )
        assert concept_token("note", None, "x") == concept_token("note", "s", None)
        assert concept_token("note", None, None) == "note|"
        pairs = [
            (("alpha|beta", "gamma"), ("alpha", "beta|gamma")),
            (("a\\", "|b"), ("a\\|", "b")),
            (("a\\|", "b"), ("a", "\\|b")),
        ]
        for one, other in pairs:
            assert concept_token("k", *one) != concept_token("k", *other)
        assert concept_token("k", "é|", "ü") == "k|é\\||ü"


class TestIntegers:
    """JSON Schema's `integer` admits `3.0`, and an
    element that did not read as an integer was dropped --- `classes: [3.0]`
    read no class and `patch: [4.0, 8, 8]` a 2-D patch.  Refused instead."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("classes", [3.0]),
            ("classes", [3, 2.5]),
            ("patch", [4.0, 8, 8]),
            ("patch", [4, 8, 1e30]),
        ],
    )
    def test_S3_1_a_number_that_is_not_an_integer_is_refused(
        self, setup, field: str, value: list[Any]
    ):
        task, _ = setup
        doc = task.to_json()
        doc["slots"][0][field] = value
        with pytest.raises(MEDH5ValidationError) as caught:
            TaskManifest(doc, base=task.base)
        assert caught.value.code == "T101"
        # The schema refuses a fraction; the parser, what the schema admits.
        assert "integer" in str(caught.value) and field in str(caught.value)

    def test_S3_5_integers_and_a_non_positive_patch(self, setup):
        task, _ = setup
        doc = task.to_json()
        doc["slots"][0].update(classes=[3, 7], patch=[4, 8, 8])
        (slot,) = TaskManifest(doc, base=task.base).slots
        assert slot.classes == (3, 7) and slot.patch == (4, 8, 8)
        doc["slots"][0]["patch"] = [0, 8, 8]
        with pytest.raises(MEDH5ValidationError) as caught:
            TaskManifest(doc, base=task.base)
        assert caught.value.code == "T101"

    @pytest.mark.parametrize("value", [3.0, 3.5, "3"])
    def test_a_manifest_entry_refuses_a_class_id_that_is_not_an_integer(
        self, tmp_path: Path, value: Any
    ):
        """Through Python and through the engine alike."""
        from medh5.dataset import Manifest, scan

        cohort(tmp_path / "cohort")
        manifest, failures = scan(tmp_path / "cohort")
        assert not failures
        doc = manifest.to_json()
        assert doc["entries"][0]["class_ids"] == [3]
        doc["entries"][0]["class_ids"] = [value]
        # One reader: the engine's words, "not an integer" or "not a number".
        with pytest.raises(ValueError, match="which is not"):
            Manifest.from_json(doc)
        saved = tmp_path / "manifest.json"
        saved.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError, match="which is not"):
            Manifest.load(saved)


def _without(doc: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """``doc`` less every member that holds its default."""
    return {k: v for k, v in doc.items() if k not in defaults or v != defaults[k]}


class TestDigestsFromTheText:
    """Every digest the contract compares is defined to the byte (§3.1--§3.3,
    §7.1).  Computed from those definitions alone --- `json`, `hashlib`, `h5py`
    --- each is the engine's; were it not, a second implementation would find
    T103, T305, T404, T405 or T401 in what the engine wrote."""

    POLICY = {  # §3.4
        "selection": "strict_prospective",
        "order_by": "effective",
        "context_us": None,
        "context_boundary": "closed",
        "uncertainty": "contained",
        "kinds": None,
        "plans": False,
        "static": True,
        "max_events": None,
        "keep": "latest",
        "ties": "keep_group",
    }
    SLOT = {"required": False, "patch": None, "roi": "center", "classes": []}  # §3.5
    TARGET = {"negative": [], "censoring": "censor", "exclude_prevalent": True}  # §5

    @staticmethod
    def fingerprint(value: Any) -> str:
        """§3.2: sha256 over the canonical JSON (1.0 §5.1) of ``value``."""
        text = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def normal_form(cls, doc: dict[str, Any]) -> dict[str, Any]:
        """§3.1's normal form of a manifest."""
        task = {"id": doc["task"]["id"], "version": doc["task"]["version"]}
        if "description" in doc["task"]:
            task["description"] = doc["task"]["description"]
        target = doc.get("target")
        if target is not None:
            target = {
                **cls.TARGET,
                "min_follow_up_us": target["horizon_us"],
                **target,
            }
        out = {
            "schema": doc["schema"],
            "task": task,
            "identity_namespace": doc["identity_namespace"],
            "policy": {**cls.POLICY, **doc.get("policy", {})},
            "slots": [{**cls.SLOT, **slot} for slot in doc.get("slots", [])],
            "target": target,
            "subjects": [
                {
                    "clock_id": None,
                    "partition": None,
                    "reconciled": [],
                    **subject,
                    "sources": [
                        {"sample_key": None, "local_subject_id": None, **source}
                        for source in subject["sources"]
                    ],
                }
                for subject in doc["subjects"]
            ],
            "rows": doc["rows"],
        }
        for optional in ("split", "fingerprint"):
            if optional in doc:
                out[optional] = doc[optional]
        return out

    def test_S3_1_the_normal_form_fills_what_the_text_says(self, setup):
        task, _ = setup
        full = task.to_json()
        terse = {k: v for k, v in full.items() if k != "policy"}
        terse["slots"] = [_without(slot, self.SLOT) for slot in full["slots"]]
        terse["target"] = _without(full["target"], self.TARGET)
        terse["subjects"] = [
            {
                **_without(s, {"clock_id": None, "partition": None, "reconciled": []}),
                "sources": [
                    _without(r, {"sample_key": None, "local_subject_id": None})
                    for r in s["sources"]
                ],
            }
            for s in full["subjects"]
        ]
        assert terse != full and self.normal_form(terse) == full
        assert TaskManifest(terse).to_json() == full
        described = dict(terse, task={**terse["task"], "description": "progression"})
        assert TaskManifest(described).to_json() == self.normal_form(described)

    def test_S3_2_every_fingerprint_reproduces_from_the_text(self, setup):
        task, _ = setup
        full = task.to_json()
        definition = {
            "schema": full["schema"],
            "task": {"id": full["task"]["id"], "version": full["task"]["version"]},
            "identity_namespace": full["identity_namespace"],
            "policy": full["policy"],
            "slots": full["slots"],
            "target": full["target"],
        }
        assert task.task_fingerprint == self.fingerprint(definition)
        assert task.manifest_fingerprint == self.fingerprint(full)
        described = dict(full, task={**full["task"], "description": "progression"})
        assert TaskManifest(described).task_fingerprint == task.task_fingerprint
        assert TaskManifest(described).manifest_fingerprint == self.fingerprint(
            described
        )
        namespace = full["identity_namespace"]
        subjects = {s["subject_id"]: s for s in full["subjects"]}
        for row in full["rows"]:
            pins = sorted(
                r["content_id"] for r in subjects[row["subject_id"]]["sources"]
            )
            assert task.row_fingerprint(row["row_id"]) == self.fingerprint(
                {
                    "task": task.task_fingerprint,
                    "namespace": namespace,
                    "subject_id": row["subject_id"],
                    "sources": pins,
                    "cutoff_us": row["cutoff_us"],
                }
            )
        for partition in ("train", "val"):
            members = sorted(
                [namespace, s["subject_id"]]
                for s in full["subjects"]
                if s["partition"] == partition
            )
            assert task.subjects_digest(partition) == self.fingerprint(members)

    def test_S3_3_a_reconciliation_digest_is_the_event_version_spelled_out(
        self, tmp_path: Path
    ):
        root = tmp_path / "f"
        root.mkdir()
        a, b = root / "a.medh5", root / "b.medh5"
        History.write(a, subject_id="P-01")
        History.write(b, subject_id="P-01")
        task = TaskManifest.new("t", "1", identity_namespace="n", base=root)
        task.add_subject(
            "P-01", [SourceRef.pin(a, uri="a.medh5"), SourceRef.pin(b, uri="b.medh5")]
        )
        task.reconcile()
        with medh5.open(a) as sample:
            clinical = sample.clinical
            assert clinical is not None
            events = {e.event_id: e.to_json() for e in clinical.records().events}
        records = task.to_json()["subjects"][0]["reconciled"]
        assert len(records) == len(events)
        for record in records:
            event = events[record["event_id"]]
            assert len(event) == 19 and None in event.values()
            assert record["digest"] == self.fingerprint(event)
            assert record["sources"] == ["a", "b"]

    def test_S7_1_the_manifest_digest_is_over_the_stored_bytes(
        self, setup, tmp_path: Path
    ):
        task, _ = setup
        path = tmp_path / "cohort" / "docs.medh5cache"
        build_document_cache(task, path, HashingTextEncoder(dim=4))
        with h5py.File(path, "r") as f:
            stored = f["manifest"][()]
            declared = f.attrs["manifest_digest"]
        declared = declared.decode() if isinstance(declared, bytes) else declared
        assert declared == "sha256:" + hashlib.sha256(stored).hexdigest()
        manifest = json.loads(stored)
        assert stored.decode("utf-8") == json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        # `selection` names the policy a cache read under: none, for features
        # of event versions, which are free of any cutoff.
        assert manifest["selection"] is None
        rows = tmp_path / "cohort" / "rows.medh5cache"
        with CacheWriter(
            rows,
            level="patient",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
            task=task,
        ):
            pass
        with h5py.File(rows, "r") as f:
            assert json.loads(f["manifest"][()])["selection"] == "strict_prospective"
        assert validate_cache(rows, task=task).ok


def hostile_task(root: Path) -> tuple[TaskManifest, str]:
    """task-cache-1 Appendix A's probe: subject ``S-01``'s first reference
    pins one version but names a file that is another, holding a secret
    identity, clock, lab value and document; its second, pinned correctly,
    holds the same event id at another value."""
    root.mkdir(parents=True, exist_ok=True)

    def lab(value: float) -> Event:
        return Event(
            "ev-secret", "ev-secret", "observation", "point", "final",
            effective_start_us=-48 * HOUR, available_us=-47 * HOUR,
            code_system="http://loinc.org", code="2160-0", value_num=value,
            unit="mg/dL",
        )  # fmt: skip

    secret = History.write(
        root / "secret.medh5",
        subject_id="MRN-0042-SECRET",
        clock=Clock.relative("clk-secret", "baseline CT acquisition"),
        events=[
            lab(42.0),
            Event(
                "ev-doc-secret",
                "ev-doc-secret",
                "document",
                "point",
                "final",
                effective_start_us=-HOUR,
                available_us=-HOUR,
            ),
        ],
        documents=[Document("doc-secret", "The secret note.")],
        links=[
            Link.between(
                ("event", "ev-doc-secret"), "describes", ("document", "doc-secret")
            )
        ],
    )
    History.write(
        root / "good.medh5",
        subject_id="S-01",
        clock=Clock.relative("clk-01", "baseline CT acquisition"),
        events=[lab(7.0)],
    )
    good = SourceRef.pin(root / "good.medh5", uri="good.medh5", source_id="s-good")
    hostile = SourceRef("secret.medh5", good.content_id, source_id="s-hostile")
    task = TaskManifest.new("hostile", "1", identity_namespace="site", base=root)
    task.add_subject("S-01", [hostile, good])
    task.add_row("S-01@0h", "S-01", 0)
    return task, secret


def outside_task(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """A manifest in ``tmp_path/tasks`` whose sources are in ``../cohort``:
    outside its directory, as most manifests' are."""
    paths = cohort(tmp_path / "cohort")
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    refs = {s: SourceRef.pin(p, uri=f"../cohort/{p.name}") for s, p in paths.items()}
    task = task_for(tmp_path / "cohort", paths, refs=refs)
    return task.save(tasks / "t.task.json"), paths


class TestConfinement:
    """Contract §2: whoever runs a manifest chooses what may be opened."""

    def test_S2_confine_refuses_a_path_outside_the_root(self, tmp_path: Path):
        manifest, _ = outside_task(tmp_path)
        task = TaskManifest.load(manifest)
        report = task.preflight(confine=True)
        assert found(report.findings) == ["T301"]
        assert all("outside the directory" in f.message for f in report.findings)
        # Absent or present, one message: confinement answers nothing about
        # a path it refuses.
        doc = task.to_json()
        doc["subjects"][0]["sources"][0]["uri"] = "../cohort/absent.medh5"
        doc.pop("fingerprint")
        absent = TaskManifest(doc, base=manifest.parent).preflight(confine=True)
        first = [f.message for f in absent.findings][0]
        present = [f.message for f in report.findings][0]
        assert first.replace("absent", "P-01") == present
        # A root wide enough admits them; so does lifting it.
        assert task.preflight(confine=tmp_path).ok
        assert task.preflight(confine=False).ok
        source = task.subjects[0].sources[0]
        with pytest.raises(MEDH5ValidationError, match="outside the directory"):
            source.open(manifest.parent, root=manifest.parent)
        inside = source.resolve(manifest.parent, root=tmp_path)
        assert inside == (tmp_path / "cohort" / "P-01.medh5").resolve()
        with pytest.raises(MEDH5ValidationError) as refused:
            task.reconcile(confine=True)
        assert refused.value.code == "T301"

    def test_S2_an_outside_source_warns_until_3_0(self, tmp_path: Path):
        manifest, paths = outside_task(tmp_path)
        task = TaskManifest.load(manifest)
        with pytest.warns(FutureWarning, match="^medh5 3.0 confines") as caught:
            assert task.preflight().ok
        assert caught[0].filename == __file__
        with pytest.warns(FutureWarning, match="^medh5 3.0 confines"):
            preflight(manifest)
        # Said which way, it is silent; so is a manifest built in memory.
        assert task.preflight(confine=False).ok
        assert task_for(tmp_path / "cohort", paths).preflight().ok
        inside = task_for(tmp_path / "cohort", paths).save(
            tmp_path / "cohort" / "t.json"
        )
        assert TaskManifest.load(inside).preflight().ok


class TestSources:
    def test_S2_a_pin_holds_until_the_sample_changes(self, setup):
        task, paths = setup
        source = task.subjects[0].sources[0]
        assert source.check(task.base) == []
        with medh5.amend(paths["P-01"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        assert found(source.check(task.base)) == ["T302"]
        repinned = SourceRef.pin(paths["P-01"], uri=source.uri)
        assert repinned.content_id != source.content_id
        assert repinned.check(task.base) == []

    def test_S2_a_repin_keeps_the_view_of_a_post_cutoff_addition(self, setup):
        """An old pin fails against a changed source; once repinned on purpose,
        an addition definitely after a row's cutoff leaves its view unchanged."""
        task, paths = setup
        before = task.preflight().row("P-01@24h")
        with medh5.amend(paths["P-01"]) as w:
            w.add_event(
                Event(
                    "late_lab",
                    "late_lab",
                    "observation",
                    "point",
                    "final",
                    effective_start_us=300 * DAY,
                    available_us=300 * DAY,
                    code_system="local",
                    code="x",
                    value_text="high",
                )
            )
        assert found(task.preflight().findings) == ["T302"]
        doc = task.to_json()
        source = doc["subjects"][0]["sources"][0]
        source["content_id"] = SourceRef.pin(paths["P-01"]).content_id
        repinned = TaskManifest(doc, base=task.base)
        after = repinned.preflight().row("P-01@24h")
        assert repinned.preflight().ok
        assert after.events == before.events and after.slots == before.slots
        assert after.target == before.target
        assert after.fingerprint != before.fingerprint  # it pins other bytes

    def test_S2_a_hostile_manifest_reads_nothing_it_does_not_pin(
        self, tmp_path: Path, capsys
    ):
        """A source off its pin was read on: its subject id (T303), its clock
        (T304), its events (T305) and, under ``--json``, its whole event
        table were reported, and T302 printed the content_id it stores
        (task-cache-1 Appendix A)."""
        task, stored = hostile_task(tmp_path / "probe")
        pre = task.preflight()
        assert [(f.code, f.location) for f in pre.findings] == [("T302", "s-hostile")]
        manifest = task.save(tmp_path / "probe" / "t.task.json")
        assert main(["task", "preflight", str(manifest), "--json"]) == EXIT_ERROR
        as_json = capsys.readouterr().out
        assert main(["task", "preflight", str(manifest)]) == EXIT_ERROR
        as_text = capsys.readouterr().out
        reported = "\n".join([repr(pre.findings), as_json, as_text])
        for secret in ("MRN-0042-SECRET", "clk-secret", "doc-secret", "42.0", stored):
            assert secret not in reported, secret
        assert "ev-secret" in as_json and "7.0" in as_json  # the pinned source
        with pytest.raises(MEDH5ValidationError) as refused:
            task.reconcile()
        assert stored not in str(refused.value)

    def test_S2_edited_clinical_bytes_fail_under_an_unchanged_root(self, setup):
        task, paths = setup
        with h5py.File(paths["P-02"], "r+") as f:
            data = f["clinical/documents/text/data"]
            data[1] = data[1] ^ 1
        source = task.subjects[1].sources[0]
        assert found(source.check(task.base)) == ["T302"]

    def test_S2_a_dataset_the_root_does_not_cover_breaks_the_pin(self, setup):
        """`content_id` is a root over *stored* digests, so a dataset added
        without one changes no root: a planted column passed the shallow and
        the deep check and a fresh preflight while the rows read its values
        (1.1 §8, E818).  Every dataset of an attested object must carry its
        digest (task-cache-1 §2, step 2)."""
        task, paths = setup
        with h5py.File(paths["P-02"], "r+") as f:
            events = f["clinical/events"]
            n = events["available_lo_us"].shape[0]
            for name in ("effective_end_lo_us", "effective_end_hi_us"):
                events.create_dataset(name, data=np.zeros(n, dtype="<i8"))
                events["valid"].create_dataset(name, data=np.zeros(n, dtype="u1"))
        with h5py.File(paths["P-03"], "r+") as f:
            f["images"].create_dataset("extra", data=np.zeros(4, dtype="u1"))
        for i in (1, 2):
            source = task.subjects[i].sources[0]
            assert found(source.check(task.base)) == ["T302"]
            assert found(source.check(task.base, deep=True)) == ["T302"]
        assert "no line of the root binds" in str(
            task.subjects[1].sources[0].check(task.base)[0]
        )
        report = task.preflight()
        assert found(report.findings) == ["T302"]
        assert {r.status for r in report.rows if r.subject_id != "P-01"} == {"error"}
        assert "E818" in validate_file(paths["P-02"], level="integrity").codes

    @pytest.mark.parametrize("alias", ["aaa", "zzz"])
    def test_S2_a_root_alias_does_not_hide_an_undigested_column(self, setup, alias):
        """A walk of the root visits an object once, at its first path, and the
        pin decided what an object belongs to by that path: an undigested
        column linked first at the root, before `clinical` in name order, was
        no clinical dataset, and the shallow and deep checks and a preflight
        passed it while the rows read it.  Membership
        is reached through each attested group, whatever else links to it."""
        task, paths = setup
        with h5py.File(paths["P-02"], "r+") as f:
            events = f["clinical/events"]
            n = events["available_lo_us"].shape[0]
            for name in ("effective_end_lo_us", "effective_end_hi_us"):
                events.create_dataset(name, data=np.zeros(n, dtype="<i8"))
                events["valid"].create_dataset(name, data=np.zeros(n, dtype="u1"))
                f[f"{alias}_{name}"] = events[name]
                f[f"{alias}_valid_{name}"] = events[f"valid/{name}"]
        source = task.subjects[1].sources[0]
        for deep in (False, True):
            (finding,) = source.check(task.base, deep=deep)
            assert finding.code == "T302"
            assert "clinical/events/valid/effective_end_lo_us" in str(finding)
        report = task.preflight()
        assert found(report.findings) == ["T302"]
        assert {r.status for r in report.rows if r.subject_id == "P-02"} == {"error"}

    def test_S2_a_later_alias_of_a_clinical_column_breaks_no_pin(self, setup):
        """The control.  A later alias of a digested column changes no line,
        so it changes no root and breaks no pin."""
        task, paths = setup
        source = task.subjects[1].sources[0]
        with h5py.File(paths["P-02"], "r+") as f:
            f["zzz_value"] = f["clinical/events/value_num"]
        assert source.check(task.base) == []
        assert source.check(task.base, deep=True) == []
        assert "E704" not in validate_file(paths["P-02"], level="integrity").codes

    @staticmethod
    def _seal(path: Path, *names: str) -> str:
        """Digest the datasets at ``names`` where the walk now first reaches
        them, and the root over them: what a crafted file carries."""
        from medh5.integrity import dataset_digest

        digests = {}
        with medh5.open(path) as sample:
            for name in names:
                digests[name] = dataset_digest(sample.root[name], name)
        with h5py.File(path, "r+") as f:
            for name, digest in digests.items():
                f[name].attrs["digest"] = digest
        with medh5.open(path) as sample:
            content_id = sample.compute_content_id()
        with h5py.File(path, "r+") as f:
            f.attrs["content_id"] = content_id
        return content_id

    @pytest.mark.parametrize("link", ["hard", "soft"])
    def test_S2_a_clinical_column_relinked_after_its_pin_is_refused(
        self, tmp_path: Path, link: str
    ):
        """A file crafted before its pin --- an
        alias sorting before `clinical` holding the column's line, and a decoy
        with a line of its own, both digested and sealed --- passed every check
        by object: relinking the column to the decoy changed no line, so the
        root recomputed and the pin, preflight and the validator held while
        the rows read the decoy.  A line binds bytes to the path it names: the
        file cannot be pinned, and a pin an earlier release took of it is
        refused (T302), by preflight too, and the validator reports E704."""
        from medh5.errors import MEDH5IntegrityError

        paths = cohort(tmp_path / "cohort")
        path = paths["P-02"]
        with h5py.File(path, "r+") as f:
            f["aaa_value"] = f["clinical/events/value_num"]
            values = f["clinical/events/value_num"][...]
            valid = f["clinical/events/valid/value_num"][...].astype(bool)
            f.create_dataset("aaa_decoy", data=np.where(valid, values + 1000.0, values))
        sealed = self._seal(path, "aaa_value", "aaa_decoy")
        with pytest.raises(MEDH5IntegrityError, match="cannot be pinned"):
            SourceRef.pin(path, uri=path.name)
        earlier = SourceRef(path.name, sealed, source_id="P-02")
        with h5py.File(path, "r+") as f:
            del f["clinical/events/value_num"]
            if link == "hard":
                f["clinical/events/value_num"] = f["aaa_decoy"]
            else:
                f["clinical/events/value_num"] = h5py.SoftLink("/aaa_decoy")
        with medh5.open(path) as sample:
            # The premise: no line changed, so the root still recomputes.
            assert sample.compute_content_id() == sealed
            assert not sample.verify().ok
            assert "clinical/events/value_num" in sample.verify().unattested
        base = tmp_path / "cohort"
        for deep in (False, True):
            (finding,) = earlier.check(base, deep=deep)
            assert finding.code == "T302"
            assert "clinical/events/value_num" in str(finding)
        task = task_for(base, paths, refs={"P-02": earlier})
        report = task.preflight()
        assert found(report.findings) == ["T302"]
        assert {r.status for r in report.rows if r.subject_id == "P-02"} == {"error"}
        diagnostics = validate_file(path, level="integrity").diagnostics
        assert ("E704", "/clinical/events/value_num") in {
            (d.code, d.location) for d in diagnostics
        }

    def test_S2_a_column_soft_linked_elsewhere_is_outside_the_pin(self, setup):
        """Readers follow a soft link as HDF5 does; the walk that decides what
        a pin covers followed hard links only.  Optional columns linked softly
        to storage under `index/`, which the root excludes, were read by every
        selection while the pin, the deep check and a preflight passed.
        Attestation walks what readers read."""
        task, paths = setup
        with h5py.File(paths["P-02"], "r+") as f:
            events = f["clinical/events"]
            n = events["available_lo_us"].shape[0]
            store = f.require_group("index").create_group("x")
            for name in ("effective_end_lo_us", "effective_end_hi_us"):
                store.create_dataset(name, data=np.full(n, 518, dtype="<i8"))
                store.create_dataset(f"valid_{name}", data=np.ones(n, dtype="u1"))
                events[name] = h5py.SoftLink(f"/index/x/{name}")
                events["valid"][name] = h5py.SoftLink(f"/index/x/valid_{name}")
        source = task.subjects[1].sources[0]
        for deep in (False, True):
            (finding,) = source.check(task.base, deep=deep)
            assert finding.code == "T302"
            assert "clinical/events/effective_end_lo_us" in str(finding)
        # 518 us ends some events before they start: the tables are invalid
        # too (T306), and that is the second finding, not the first.
        assert "T302" in found(task.preflight().findings)
        assert "E818" in validate_file(paths["P-02"], level="integrity").codes

    @staticmethod
    def _registered(path: Path) -> Path:
        shape = (6, 8, 8)
        matrix = np.eye(4)
        matrix[0, 3] = 2.0
        with medh5.create(path, codec="portable") as w:
            for tp, frame in (("tp0", "F0"), ("tp1", "F1")):
                w.add_timepoint(tp, days_from_baseline=0 if tp == "tp0" else 92)
                w.add_grid(
                    f"ct_{tp}", shape=shape, spacing=(1.5, 0.8, 0.8), timepoint=tp,
                    frame_uid=frame,
                )  # fmt: skip
                w.add_image(
                    f"CT_{tp}",
                    np.zeros(shape, np.int16),
                    grid=f"ct_{tp}",
                    modality="CT",
                )
            w.add_transform(
                "t", kind="affine", from_frame="F0", to_frame="F1", matrix=matrix,
                from_grid="ct_tp0", to_grid="ct_tp1", invertible=True,
            )  # fmt: skip
        return path

    @staticmethod
    def _translate(path: Path, by: float) -> None:
        with h5py.File(path, "r+") as f:
            matrix = f["transforms/t/matrix"]
            values = matrix[...]
            values[0, 3] = by
            matrix[...] = values

    def test_S2_a_transform_aliased_under_index_is_outside_the_pin(
        self, tmp_path: Path
    ):
        """An object is listed once, under the first path that reaches it, and
        `index/` is outside the root: a transform also linked under `index/`
        before its sample was restamped and pinned was in no dataset line, and
        its translation moved from 2 to 102 mm under a matching root and an
        unchanged pin.  Covered is by identity."""
        from medh5.errors import MEDH5IntegrityError

        path = self._registered(tmp_path / "reg.medh5")
        with h5py.File(path, "r+") as f:
            f.require_group("index")["alias"] = f["transforms/t/matrix"]
        with medh5.open(path) as sample:
            content_id = sample.compute_content_id()
        with h5py.File(path, "r+") as f:
            f.attrs["content_id"] = content_id
        with pytest.raises(MEDH5IntegrityError, match="transforms/t/matrix"):
            SourceRef.pin(path, uri=path.name)
        pin = SourceRef(path.name, content_id)  # as an earlier release pinned it
        for deep in (False, True):
            (finding,) = pin.check(tmp_path, deep=deep)
            assert finding.code == "T302" and "transforms/t/matrix" in str(finding)
        self._translate(path, 102.0)
        assert found(pin.check(tmp_path, deep=True)) == ["T302"]
        report = validate_file(path, level="integrity")
        assert "/transforms/t/matrix" in {
            d.location for d in report.diagnostics if d.code == "E702"
        }

    def test_S2_an_unaliased_transform_is_held_by_its_digest(self, tmp_path: Path):
        """The control: without the alias the pin holds, and an edit fails the
        deep check and the validator."""
        path = self._registered(tmp_path / "reg.medh5")
        pin = SourceRef.pin(path, uri=path.name)
        assert pin.check(tmp_path) == [] and pin.check(tmp_path, deep=True) == []
        self._translate(path, 102.0)
        assert found(pin.check(tmp_path, deep=True)) == ["T302"]
        assert "E701" in validate_file(path, level="integrity").codes

    def test_S2_a_file_uri_names_the_path_it_was_made_from(self, setup, tmp_path: Path):
        """`Path.as_uri()` percent-escapes a space and may name `localhost`;
        stripping `file://` kept the escapes, and made `file://localhost/x`
        the relative `localhost/x`."""
        _, paths = setup
        spaced = tmp_path / "with space" / "P 01.medh5"
        spaced.parent.mkdir()
        shutil.copyfile(paths["P-01"], spaced)
        ref = SourceRef.pin(spaced, uri=spaced.as_uri())
        assert ref.resolve() == spaced and ref.check() == []
        assert ref.source_id == "with_space_P_01"
        if spaced.as_uri().startswith("file:///"):  # POSIX: no drive
            local = SourceRef(
                "file://localhost" + spaced.as_uri()[len("file://") :], ref.content_id
            )
            assert local.resolve(tmp_path / "elsewhere") == spaced
            assert local.check() == []

    def test_S2_a_missing_source_is_T301(self, setup, tmp_path: Path):
        task, paths = setup
        paths["P-03"].unlink()
        report = task.preflight()
        assert "T301" in found(report.findings)
        assert {r.status for r in report.rows if r.subject_id == "P-03"} == {"error"}

    def test_S2_members_are_named_by_shard_and_key(self, setup, tmp_path: Path):
        task, paths = setup
        shard = pack([paths["P-01"]], tmp_path / "cohort" / "s.medh5c", keys=["P-01"])
        member = SourceRef.pin(shard, sample_key="P-01", uri=shard.name)
        assert member.locator == "s.medh5c::P-01"
        assert member.content_id == task.subjects[0].sources[0].content_id
        with member.open(task.base) as sample:
            assert sample.identity.subject_id == "P-01"
        with pytest.raises(MEDH5ValidationError, match="collection"):
            SourceRef(shard.name, member.content_id).open(task.base)
        assert SourceRef.from_json(member.to_json()) == member
        assert member.resolve() == Path(shard.name)


class TestPreflight:
    def test_S4_rows_admit_what_was_available(self, setup):
        task, _ = setup
        report = preflight(task)
        assert report.ok
        assert report.counts == {"eligible": 5, "excluded": 1}
        early = report.row("P-02@24h")
        assert [e.event_id for e in early.events] == ["lab0", "ct0", "rep_v1"]
        assert early.slots["ct"].image_id == "CT_tp0"
        assert early.slots["ct"].annotations == ()  # nothing attested them
        assert early.slots["ct"].label_annotations == ("lesions_tp0",)
        late = report.row("P-02@d95")
        assert late.slots["ct"].image_id == "CT_tp1"
        assert late.selection is not None and late.selection.certified
        assert report.eligible("val") == (
            report.row("P-03@24h"),
            report.row("P-03@d95"),
        )
        with pytest.raises(KeyError):
            report.row("nope")

    def test_S6_a_slot_is_supervised_by_annotations_of_its_classes(self, setup):
        """`label_annotations` listed every voxel annotation on the grid, while
        the loader skips one that neither labels nor examined a class the slot
        asks for: the row view and the batch it builds disagreed."""
        task, _ = setup
        doc = task.to_json()
        doc["slots"][0]["classes"] = [7]
        other = TaskManifest(doc, base=task.base)
        for asked, found in ((task, ("lesions_tp0",)), (other, ())):
            fill = asked.preflight().row("P-02@24h").slots["ct"]
            assert fill.image_id == "CT_tp0" and fill.label_annotations == found

    def test_S5_targets(self, setup):
        task, _ = setup
        report = task.preflight()
        assert report.row("P-01@24h").target.status == "positive"
        assert report.row("P-01@24h").target.value == 1.0
        assert report.row("P-02@24h").target.status == "negative"
        prevalent = report.row("P-01@d95")
        assert (prevalent.status, prevalent.reasons) == (
            "excluded",
            ("prevalent_target",),
        )
        censored = report.row("P-02@d95")
        assert censored.target.status == "censored" and not censored.target.observed
        assert censored.status == "eligible"

    @staticmethod
    def _response(event_id: str, status: str, day: int, value: str = "PD") -> Event:
        return Event(
            event_id,
            event_id,
            "assessment",
            "point",
            status,
            effective_start_us=day * DAY,
            available_us=day * DAY + HOUR,
            code_system=History.RECIST,
            code="overall_response",
            value_text=value,
        )

    def _row(
        self, tmp_path: Path, response: str, *events: Event, row: str = "P-01@24h"
    ) -> Any:
        root = tmp_path / "o"
        root.mkdir(parents=True)
        path = root / "P-01.medh5"
        History.write(path, response=response, events=list(events))
        return task_for(root, {"P-01": path}).preflight().row(row)

    def _target(self, tmp_path: Path, response: str, *events: Event) -> Any:
        return self._row(tmp_path, response, *events).target

    def test_S5_a_cancelled_outcome_is_neither_positive_nor_negative(
        self, tmp_path: Path
    ):
        """A cancelled progression assessment at day 100 did not happen: the
        response observed at day 90 still labels the row.  It labelled the row
        positive."""
        target = self._target(tmp_path, "SD", self._response("pd", "cancelled", 100))
        assert (target.status, target.value) == ("negative", 0.0)

    def test_S5_a_planned_outcome_censors_its_window(self, tmp_path: Path):
        target = self._target(tmp_path, "SD", self._response("pd", "planned", 100))
        assert target.status == "censored" and not target.observed
        assert target.reason is not None and "planned" in target.reason

    def test_S5_a_planned_negative_observes_nothing(self, tmp_path: Path):
        """`NE` is neither outcome nor absence, so the only negative is a plan."""
        target = self._target(
            tmp_path, "NE", self._response("sd", "planned", 100, value="SD")
        )
        assert target.status == "censored" and not target.observed

    @pytest.mark.parametrize("status", ["planned", "cancelled"])
    def test_S5_an_outcome_that_did_not_occur_is_not_prevalent(
        self, tmp_path: Path, status: str
    ):
        """A progression planned or cancelled at day 50 had not occurred by day
        95: the row is not excluded as prevalent (as a final one is), and with
        nothing observed in its window it is censored."""
        row = self._row(
            tmp_path, "SD", self._response("pd", status, 50), row="P-01@d95"
        )
        assert (row.status, row.target.status) == ("eligible", "censored")
        final = self._row(
            tmp_path / "final", "SD", self._response("pd", "final", 50), row="P-01@d95"
        )
        assert (final.status, final.reasons) == ("excluded", ("prevalent_target",))

    @pytest.mark.parametrize(
        ("status", "label"), [("final", "negative"), ("cancelled", "censored")]
    )
    def test_S5_a_cancelled_negative_observes_nothing(
        self, tmp_path: Path, status: str, label: str
    ):
        """`NE` at day 90 is neither outcome nor absence, so the only negative
        is the stable disease at day 170 --- unless it was cancelled."""
        target = self._target(
            tmp_path, "NE", self._response("sd", status, 170, value="SD")
        )
        assert target.status == label
        assert target.observed is (label == "negative")

    def test_S5_censored_rows_can_be_excluded(self, tmp_path: Path):
        paths = cohort(tmp_path / "c")
        doc = task_for(tmp_path / "c", paths).to_json()
        doc["target"]["censoring"] = "exclude"
        report = TaskManifest(doc, base=tmp_path / "c").preflight()
        assert report.row("P-02@d95").reasons == ("censored",)

    def test_S4_an_uncertain_revision_is_uncertifiable(self, tmp_path: Path):
        root = tmp_path / "u"
        root.mkdir()
        path = root / "P-01.medh5"
        History.write(
            path,
            events=[
                Event(
                    "rep_v3",
                    "rep",
                    "document",
                    "point",
                    "amended",
                    effective_start_us=0,
                )
            ],
            documents=[],
            links=[
                Link.between(("event", "rep_v3"), "supersedes", ("event", "rep_v2"))
            ],
        )
        report = task_for(root, {"P-01": path}).preflight()
        row = report.row("P-01@24h")
        assert row.status == "uncertifiable"
        assert row.reasons == ("uncertain_revision:rep",)

    def test_S4_a_missing_required_slot_excludes(self, tmp_path: Path):
        paths = cohort(tmp_path / "m", {"P-01": "SD"})
        doc = task_for(tmp_path / "m", paths).to_json()
        doc["slots"][0]["modality"] = "MR"
        report = TaskManifest(doc, base=tmp_path / "m").preflight()
        assert report.row("P-01@24h").reasons == ("missing_required_slot:ct",)

    def test_S3_3_fragments_reconcile(self, tmp_path: Path):
        root = tmp_path / "f"
        root.mkdir()
        a, b = root / "a.medh5", root / "b.medh5"
        History.write(a, subject_id="P-01")
        History.write(b, subject_id="P-01")  # the same history, exported twice
        task = TaskManifest.new(
            "t", "1", identity_namespace="n", target=TARGET, base=root
        )
        task.add_subject(
            "P-01", [SourceRef.pin(a, uri="a.medh5"), SourceRef.pin(b, uri="b.medh5")]
        )
        task.add_row("r", "P-01", 24 * HOUR)
        assert found(task.preflight().findings) == ["T305"]
        task.reconcile()
        assert len(task.subjects[0].reconciled) == 6
        assert task.preflight().ok

    def test_S3_3_identity_and_clocks(self, tmp_path: Path):
        root = tmp_path / "i"
        root.mkdir()
        a, b = root / "a.medh5", root / "b.medh5"
        History.write(a, subject_id="P-01")
        History.write(b, subject_id="P-02", clock=Clock.relative("other", "elsewhere"))
        task = TaskManifest.new("t", "1", identity_namespace="n", base=root)
        # A recorded cross-site mapping is legitimate: site B calls the subject
        # P-02, and the manifest says so.  A recorded id the sample does not
        # carry is not.
        pinned_b = SourceRef.pin(b, uri="b.medh5")
        assert pinned_b.local_subject_id == "P-02"
        task.add_subject("P-01", [SourceRef.pin(a, uri="a.medh5"), pinned_b])
        # Two clocks (T304), and the shared events not yet reconciled (T305).
        assert found(task.preflight().findings) == ["T304", "T305"]
        pinned_a = SourceRef.pin(a, uri="a.medh5")
        wrong = SourceRef(
            "a.medh5", pinned_a.content_id, source_id="x", local_subject_id="P-9"
        )
        other = TaskManifest.new("t", "1", identity_namespace="n", base=root)
        other.add_subject("P-01", [wrong])
        assert "T303" in found(other.preflight().findings)

    def test_S3_3_a_source_without_a_mapping_is_its_subjects(self, tmp_path: Path):
        """A source that records no ``local_subject_id`` is checked against the
        subject it is listed under: another patient's file was accepted."""
        paths = cohort(tmp_path / "c", {"P-01": "PD", "P-05": "SD"})

        def listed(subject: str, path: Path) -> TaskManifest:
            pinned = SourceRef.pin(path, uri=path.name)
            unmapped = SourceRef(pinned.uri, pinned.content_id, source_id="s")
            task = TaskManifest.new("t", "1", identity_namespace="n", base=path.parent)
            task.add_subject(subject, [unmapped])
            return task

        report = listed("P-01", paths["P-05"]).preflight()
        (finding,) = report.findings
        assert finding.code == "T303" and "'P-05'" in finding.message
        assert listed("P-01", paths["P-01"]).preflight().ok

    @staticmethod
    def _observation(event_id: str, record_id: str, available_us: int) -> Event:
        return Event(
            event_id,
            record_id,
            "observation",
            "point",
            "final",
            effective_start_us=-10 * HOUR,
            available_us=available_us,
            code_system="http://loinc.org",
            code="1-1",
            value_num=1.0,
        )

    def test_S3_3_fragments_that_contradict_are_their_subjects_finding(
        self, tmp_path: Path
    ):
        """Each fragment's revision chain was sound, but merged v0 was
        superseded twice: the whole preflight raised E816, and the cohort's
        other subject got no report."""
        root = tmp_path / "c7"
        root.mkdir()
        obs = self._observation
        for name, newer in (("f1", "v1"), ("f2", "v2")):
            History.write(
                root / f"{name}.medh5",
                subject_id="P-01",
                events=[obs("v0", "r", -9 * HOUR), obs(newer, "r", -8 * HOUR)],
                links=[Link.between(("event", newer), "supersedes", ("event", "v0"))],
            )
        History.write(root / "good.medh5", subject_id="P-02")
        task = TaskManifest.new("t", "1", identity_namespace="n", base=root)
        task.add_subject(
            "P-01",
            [
                SourceRef.pin(root / "f1.medh5", uri="f1.medh5"),
                SourceRef.pin(root / "f2.medh5", uri="f2.medh5"),
            ],
        )
        task.add_subject("P-02", [SourceRef.pin(root / "good.medh5", uri="good.medh5")])
        task.add_row("bad", "P-01", 24 * HOUR)
        task.add_row("good", "P-02", 24 * HOUR)
        task.reconcile()
        report = task.preflight()
        assert found(report.findings) == ["T305"]
        assert "contradict" in str(report.findings[0])
        assert report.row("bad").status == "error"
        assert report.row("good").status == "eligible"

    def test_S3_5_one_row_identity_reads_one_input(self, tmp_path: Path):
        """Two fragments of one subject hold the same reconciled imaging event,
        each describing its own CT --- every voxel 11 in one, 93 in the other.
        The slot kept the first fragment the manifest listed while the row
        fingerprint sorts the pins, so reordering two sources changed the
        input under one row identity.  A tie goes to
        the smallest pinned `content_id`, whatever the order."""
        root = tmp_path / "b04"
        root.mkdir()
        refs = {}
        for name, value in (("a", 11), ("b", 93)):
            History.write(root / f"{name}.medh5", subject_id="P-01", fill=value)
            refs[name] = SourceRef.pin(root / f"{name}.medh5", uri=f"{name}.medh5")
        canonical = min(refs.values(), key=lambda r: r.content_id)

        def read(*names: str) -> tuple[str, float, str]:
            task = TaskManifest.new(
                "t", "1", identity_namespace="n", slots=[Slot("ct", "CT")], base=root
            )
            task.add_subject("P-01", [refs[n] for n in names])
            task.add_row("r", "P-01", 24 * HOUR)
            report = task.reconcile().preflight()
            assert report.ok, report.findings
            row = report.row("r")
            fill = row.slots["ct"]
            assert fill.fragment is not None and fill.image_id == "CT_tp0"
            source = row.sources[fill.fragment]
            with source.open(root) as sample:
                value = float(np.mean(sample.images[fill.image_id].read()))
            return row.fingerprint, value, source.content_id

        forward, backward = read("a", "b"), read("b", "a")
        assert forward == backward
        assert forward[2] == canonical.content_id
        assert forward[1] == (11.0 if canonical is refs["a"] else 93.0)

    def test_S3_3_a_reconciled_document_event_owns_one_text(self, tmp_path: Path):
        """The event JSON agreed while each fragment's event owned another
        text --- "NO metastasis" and "Metastasis CONFIRMED" --- and the merged
        version owned both.  A copy under another id is the same text."""
        root = tmp_path / "c13"
        root.mkdir()
        report_event = Event(
            "rep_x",
            "rep_x",
            "document",
            "point",
            "final",
            effective_start_us=0,
            available_us=2 * HOUR,
        )

        def fragment(name: str, document_id: str, text: str) -> SourceRef:
            path = root / f"{name}.medh5"
            History.write(
                path,
                subject_id="P-01",
                events=[report_event],
                documents=[Document(document_id, text)],
                links=[
                    Link.between(
                        ("event", "rep_x"), "describes", ("document", document_id)
                    )
                ],
            )
            return SourceRef.pin(path, uri=path.name, source_id=name)

        def task(*sources: SourceRef) -> TaskManifest:
            t = TaskManifest.new("t", "1", identity_namespace="n", base=root)
            t.add_subject("P-01", list(sources))
            t.add_row("r", "P-01", 24 * HOUR)
            return t.reconcile()

        contradicting = task(
            fragment("A", "doc_a", "NO metastasis."),
            fragment("B", "doc_b", "Metastasis CONFIRMED."),
        )
        report = contradicting.preflight()
        assert found(report.findings) == ["T305"]
        assert "owns another text" in str(report.findings[0])

        aliased = task(
            fragment("C", "doc_c", "NO metastasis."),
            fragment("D", "doc_d", "NO metastasis."),
        )
        report = aliased.preflight()
        assert report.ok
        row = report.row("r")
        owned = row.subject.owned_documents(row.subject.events.index("rep_x"))
        assert len(owned) == 1, owned

        # The record names who holds the event; a record of no shared event,
        # or a second record of one, is not a reconciliation.
        doc = aliased.to_json()
        records = doc["subjects"][0]["reconciled"]
        bogus = [dict(r, sources=["nowhere-1", "nowhere-2"]) for r in records]
        stale = [*records, dict(records[0], event_id="no_such_event")]
        twice = [*records, records[0]]
        for reconciled, why in (
            (bogus, "held by"),
            (stale, "no two"),
            (twice, "twice"),
        ):
            doc["subjects"][0]["reconciled"] = reconciled
            findings = TaskManifest(doc, base=root).preflight().findings
            assert "T305" in found(findings), why
            assert any(why in str(f) for f in findings), (why, findings)


class TestRepin:
    def test_S2_repin_pins_what_is_there(self, setup):
        """Re-pinning was by hand (``SourceRef.pin`` per source, then
        reconcile): ``repin`` pins each source to the whole version it names
        now, and names the rows it stales."""
        task, paths = setup
        assert task.repin() == RepinReport((), ("P-01", "P-02", "P-03"), False, ())
        before = task.subjects[1].sources[0].content_id
        with medh5.amend(paths["P-02"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        assert not task.preflight().ok
        report = task.repin()
        assert [(r.source_id, r.before) for r in report.changed] == [("P-02", before)]
        after = report.changed[0].after
        assert after == task.subjects[1].sources[0].content_id != before
        assert report.rows == ("P-02@24h", "P-02@d95")
        assert task.preflight().ok and task.validate() == []
        assert task.repin().changed == ()

    def test_S2_repin_dry_run_writes_nothing(self, setup, tmp_path: Path, capsys):
        task, paths = setup
        manifest = task.save(tmp_path / "cohort" / "t.task.json")
        with medh5.amend(paths["P-01"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        before = manifest.read_bytes()
        loaded = TaskManifest.load(manifest)
        report = loaded.repin(dry_run=True)
        assert [r.source_id for r in report.changed] == ["P-01"]
        assert loaded.to_json() == TaskManifest.load(manifest).to_json()
        capsys.readouterr()
        assert main(["task", "repin", str(manifest), "--dry-run", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["dry_run"] is True
        assert manifest.read_bytes() == before
        assert main(["task", "repin", str(manifest), "--source", "P-02"]) == 0
        assert "repinned 0, 1 unchanged" in capsys.readouterr().out
        assert manifest.read_bytes() == before
        assert main(["task", "repin", str(manifest)]) == EXIT_OK
        out = capsys.readouterr().out
        assert "repinned P-01: " in out and "2 row fingerprint(s) change" in out
        assert TaskManifest.load(manifest).preflight().ok

    def test_S2_repin_refuses_another_subject(self, setup):
        task, paths = setup
        pinned = task.to_json()
        History.write(paths["P-02"], subject_id="P-99", response="SD")
        with pytest.raises(MEDH5ValidationError) as caught:
            task.repin()
        assert caught.value.code == "T303"
        assert task.to_json() == pinned, "refused, nothing changes"

    def test_S2_repin_refuses_a_corrupt_source(self, setup):
        task, paths = setup
        with medh5.amend(paths["P-02"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        with h5py.File(paths["P-02"], "a") as f:
            damage(f["clinical/events/event_id"], "data")
        pinned = task.to_json()
        with pytest.raises(MEDH5ValidationError) as caught:
            task.repin()
        assert caught.value.code == "T302"
        assert task.to_json() == pinned
        with pytest.raises(MEDH5ValidationError):
            task.repin(sources=["nope"])


class TestInstanceCentres:
    """§3.5 and §4: an ``eligible_instances`` slot centres on its first eligible
    instance, in voxels of the image's own grid, from the object table alone."""

    @pytest.mark.parametrize("space", ["index", "world"])
    def test_S3_5_a_crop_centres_on_its_instance_in_grid_voxels(
        self, tmp_path: Path, space: str
    ):
        """A box stored in world space is carried onto the image's grid before
        its centre is taken: its millimetres were read as voxel indices, and
        the crop landed at (4, 2, 3), every voxel of it valid."""
        root = tmp_path / "w"
        root.mkdir()
        path = root / "P-01.medh5"
        History.write(path, lesion=False)
        edges = np.array([[1.5, 2.5, 3.5], [2.5, 3.5, 4.5]])  # voxel (2, 3, 4)
        if space == "world":
            with medh5.open(path) as sample:
                world = sample.grids["ct_tp0"].index_to_world(edges)
            edges = np.stack([world.min(axis=0), world.max(axis=0)])
        with medh5.amend(path) as w:
            boxes = np.ascontiguousarray(edges.T[None])  # (1, 3, 2)
            w.add_boxes(
                "found", boxes, [3], grid="ct_tp0", space=space, instance_ids=[7]
            )
            w.add_event(
                Event(
                    "det0",
                    "det0",
                    "procedure",
                    "point",
                    "completed",
                    effective_start_us=2 * HOUR,
                    available_us=2 * HOUR,
                )
            )
            w.add_link(
                Link.between(
                    ("event", "ct0"),
                    "describes",
                    ("annotation", "found"),
                    asserted_by="det0",
                )
            )
        report = instance_task(root, {"P-01": path}).preflight()
        assert report.ok, report.findings
        fill = report.row("P-01@24h").slots["ct"]
        assert (fill.roi, fill.center) == ("eligible_instances", (2, 3, 4))

    def test_S3_5_a_self_asserted_mask_centres_the_crop(self, tmp_path: Path):
        """The procedure that drew the mask asserts it from itself, as both
        shipped examples do.  Of unknown time, it is never selected under
        effective ordering, and the asserter had to be selected when it was
        also the endpoint: the crop fell back to the image's centre."""
        root = tmp_path / "w"
        root.mkdir()
        path = root / "P-01.medh5"
        History.write(
            path,
            events=[
                Event(
                    "seg0",
                    "seg0",
                    "procedure",
                    "unknown",
                    "completed",
                    available_us=2 * HOUR,
                )
            ],
            links=[
                Link.between(
                    ("event", "seg0"),
                    "describes",
                    ("annotation", "lesions_tp0"),
                    asserted_by="seg0",
                )
            ],
        )
        report = instance_task(root, {"P-01": path}).preflight()
        assert report.ok, report.findings
        fill = report.row("P-01@24h").slots["ct"]
        assert (fill.roi, fill.center) == ("eligible_instances", (3, 2, 7))
        assert fill.annotations == ("lesions_tp0",)

    def test_S4_preflight_reads_no_mask(self, tmp_path: Path):
        """A crop is centred from the object table: a mask whose chunk no
        longer decompresses fails no preflight.  Every mask was decoded, and
        one damaged mask failed the whole cohort's preflight."""
        paths = cohort(tmp_path / "c")
        with h5py.File(paths["P-03"], "r+") as f:
            damage(f["annotations/lesions_tp0"], "mask_data")
        task = instance_task(tmp_path / "c", paths)
        report = task.preflight()
        assert report.ok, report.findings
        assert report.counts == {"eligible": 5, "excluded": 1}
        # Its bytes are checked where every dataset's bytes are read.
        assert found(task.preflight(deep=True).findings) == ["T302"]

    def test_S4_an_unreadable_object_table_is_its_subjects_finding(
        self, tmp_path: Path
    ):
        paths = cohort(tmp_path / "c")
        with h5py.File(paths["P-03"], "r+") as f:
            damage(f["annotations/lesions_tp0"], "boxes")
        report = instance_task(tmp_path / "c", paths).preflight()
        (finding,) = report.findings
        assert finding.code == "T306" and "CT_tp0" in finding.message
        assert {r.status for r in report.rows if r.subject_id == "P-03"} == {"error"}
        assert report.row("P-02@24h").status == "eligible"


class TestPreflightAtScale:
    """§4 at cohort scale: rows index their subject's history rather than
    copying it, a damaged source is a finding, and the result pickles small."""

    def many_cutoffs(self, tmp_path: Path, rows: int) -> TaskManifest:
        paths = cohort(tmp_path / f"c{rows}", {"P-01": "PD"})
        task = TaskManifest.new(
            "progression",
            "1",
            identity_namespace="site",
            slots=[Slot("ct", "CT")],
            target=TARGET,
            base=tmp_path / f"c{rows}",
        )
        task.add_subject("P-01", [SourceRef.pin(paths["P-01"], uri="P-01.medh5")])
        for k in range(rows):
            task.add_row(f"r{k}", "P-01", (k + 1) * DAY)
        return task

    def test_S4_rows_index_one_shared_history(self, tmp_path: Path):
        report = self.many_cutoffs(tmp_path, 40).preflight()
        assert report.ok, report.findings
        (subject,) = report.subjects
        assert len(subject.events) == len(History.events())
        for row in report.rows:
            assert row.subject is subject
            assert row.events == tuple(subject.events[int(i)] for i in row.selected)
            assert row.event_fragments == (0,) * len(row.selected)
            selection = row.selection
            assert selection is not None
            assert [e.event_id for e in selection.events] == [
                e.event_id for e in row.events
            ]
        late = report.row("r39")
        assert [e.event_id for e in late.events] == ["lab0", "ct0", "rep_v2"]
        assert subject.owned_documents(subject.events.index("rep_v2")) == [
            (0, "rep_text_v2")
        ]
        # Each event version crosses once however many rows admit it: `lab0`
        # is its own record, so its id is in the pickle twice (event id and
        # record id), not once per row.
        blob = pickle.dumps(report)
        assert blob.count(b"lab0") == 2
        again = pickle.loads(blob)
        assert again.counts == report.counts
        assert again.row("r39").events == late.events
        assert again.row("r39").selection == late.selection

    def test_S4_a_damaged_source_is_a_finding_not_a_failure(self, tmp_path: Path):
        from medh5.clinical import Document

        path = tmp_path / "notes.medh5"
        long = Document("note_text", "a long note " * 20_000)  # chunked, compressed
        History.write(
            path,
            events=[
                Event(
                    "note",
                    "note",
                    "document",
                    "point",
                    "final",
                    effective_start_us=DAY,
                    available_us=DAY + HOUR,
                )
            ],
            documents=[long],
            links=[
                Link.between(("event", "note"), "describes", ("document", "note_text"))
            ],
        )
        task = TaskManifest.new("t", "1", identity_namespace="site", base=tmp_path)
        task.add_subject("P-01", [SourceRef.pin(path, uri=path.name)])
        task.add_row("r", "P-01", 2 * DAY)
        with h5py.File(path, "r+") as f:
            data = f["clinical/documents/text/data"]
            assert data.chunks is not None
            # A chunk that no longer decompresses, under unchanged digests:
            # verifying it and checking its text both fail to read it.
            data.id.write_direct_chunk((0,), b"\x00" * 16)
        report = task.preflight()
        assert not report.ok
        # The bytes fail their digests, so the source is read no further: no
        # T306 for the table it would have read (task-cache-1 Appendix A).
        assert {f.code for f in report.findings} == {"T302"}
        assert report.row("r").status == "error"

    def test_S4_packed_columns_carry_64_bit_offsets(self, tmp_path: Path):
        """A packed column's offsets are ``int64``, as documented.  They were
        ``int32``, and a subject whose text passed 2 GiB panicked rather than
        crossing."""
        report = self.many_cutoffs(tmp_path, 2).preflight()
        (subject,) = report.subjects
        assert subject.events.column("event_id").offsets.dtype == np.int64
        assert report.row("r0").slots["ct"].annotations == ()

    def test_a_packed_column_indexes_as_a_sequence(self):
        """`column[-1]` sliced `offsets[-1]:offsets[0]` and returned `""`."""
        column = Packed(b"abcdef", np.array([0, 3, 6], np.int64))
        assert (column[-1], column[-2], column[1]) == ("def", "abc", "def")
        for outside in (2, -3):
            with pytest.raises(IndexError):
                column[outside]

    def test_S4_the_manifest_findings_leave_no_history_read(self, tmp_path: Path):
        task = self.many_cutoffs(tmp_path, 2)
        task.add_row("ghost", "P-99", DAY)
        report = task.preflight()
        assert {f.code for f in report.findings} == {"T201"}
        assert report.subjects == ()
        assert all(r.subject is None and r.events == () for r in report.rows)
        assert report.row("r0").reasons[0].startswith("manifest_invalid")


class TestCaches:
    def test_S7_event_level_caches(self, setup, tmp_path: Path):
        task, _ = setup
        path = tmp_path / "cohort" / "docs.medh5cache"
        encoder = HashingTextEncoder(dim=8)
        report = build_document_cache(task, path, encoder)
        assert report.ok and report.entries == 6
        source = task.subjects[0].sources[0]
        with open_cache(path) as cache:
            assert cache.level == "event" and len(cache) == 6
            feature = cache.event_feature(source.content_id, "rep_v1")
            assert feature is not None and feature.shape == (8,)
            assert cache.event_feature(source.content_id, "lab0") is None
            entry_id = event_entry_id(source.content_id, "rep_v1")
            assert np.array_equal(cache.get(entry_id), feature)
            assert cache.header["encoder"]["name"] == HashingTextEncoder.NAME
            assert cache.row_feature("P-01@24h") is None
            assert "FeatureCache(" in repr(cache)
        assert np.array_equal(encoder.encode("a b"), encoder.encode("A, b!"))

    def test_S8_stale_corrupt_and_inadmissible_are_told_apart(
        self, setup, tmp_path: Path
    ):
        task, paths = setup
        report = task.preflight()
        path = tmp_path / "cohort" / "rows.medh5cache"
        record = fitted_on(task)
        assert record["partition"] == "train"
        with CacheWriter(
            path,
            level="patient",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
            task=task,
            fitted_on=record,
        ) as w:
            for row in report.eligible("train"):
                w.add(
                    row.row_id,
                    np.zeros(2, np.float32),
                    sources=list(row.sources),
                    row_id=row.row_id,
                    row_fingerprint=row.fingerprint,
                    cutoff_us=row.cutoff_us,
                    event_versions=[e.event_id for e in row.events],
                )
        assert validate_cache(path, task=task).ok

        # Inadmissible: an entry encoding a version its row cannot have read.
        leaky = tmp_path / "cohort" / "leaky.medh5cache"
        with CacheWriter(
            leaky,
            level="patient",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
            task=task,
        ) as w:
            row = report.row("P-02@24h")
            w.add(
                row.row_id,
                np.zeros(2, np.float32),
                sources=list(row.sources),
                row_id=row.row_id,
                row_fingerprint=row.fingerprint,
                cutoff_us=row.cutoff_us,
                event_versions=["lab0", "ct0", "rep_v2"],
            )
        assert found(validate_cache(leaky, task=task).findings) == ["T406"]

        # Another task.
        other = TaskManifest(
            dict(task.to_json(), policy={"context_us": DAY}), base=task.base
        )
        assert "T404" in found(validate_cache(path, task=other).findings)

        # Corrupt: the cache's own bytes.
        with h5py.File(path, "r+") as f:
            f["entries"][sorted(f["entries"])[0]][0] = 5.0
        corrupt = validate_cache(path)
        assert corrupt.corrupt and not corrupt.stale

        # Stale: a source changed.
        shutil.copyfile(paths["P-01"], tmp_path / "keep.medh5")
        with medh5.amend(paths["P-01"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        stale = validate_cache(path)
        assert stale.stale and "T403" in found(stale.findings)

    def test_S8_an_entry_of_a_row_in_error_is_stale_not_inadmissible(
        self, setup, tmp_path: Path
    ):
        """A row in error admits nothing that is known: each of its entries was
        T406, inadmissible, on top of the stale source that put it there."""
        task, paths = setup
        path = tmp_path / "cohort" / "rows.medh5cache"
        with CacheWriter(
            path,
            level="patient",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
            task=task,
        ) as w:
            entry = w.add_row(task.preflight().row("P-02@24h"), np.zeros(2, np.float32))
        assert entry.event_versions == ("lab0", "ct0", "rep_v1")
        assert validate_cache(path, task=task).ok
        with medh5.amend(paths["P-02"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        findings = validate_cache(path, task=task).findings
        assert [(f.code, f.location) for f in findings] == [
            ("T403", "P-02@24h"),
            ("T404", "P-02@24h"),
        ]
        assert "in error" in findings[1].message

    def test_S7_1_a_cache_kept_away_from_its_task_validates_against_it(
        self, setup, tmp_path: Path, capsys
    ):
        """Entries copy the task's references, relative to the manifest's
        directory, and were resolved against the cache's: a cache kept
        anywhere else was all stale (T403), however often it was rebuilt."""
        task, _ = setup
        path = tmp_path / "elsewhere" / "docs.medh5cache"
        path.parent.mkdir()
        assert build_document_cache(task, path, HashingTextEncoder(dim=4)).ok
        assert validate_cache(path, task=task).ok
        assert validate_cache(path).stale, "without a task: the cache's directory"
        assert validate_cache(path, base=task.base).ok
        manifest = task.save(tmp_path / "cohort" / "t.task.json")
        args = ["cache", "validate", str(path), "--task", str(manifest)]
        assert main(args) == EXIT_OK
        capsys.readouterr()

    def test_S7_1_the_writer_refuses_what_its_reader_rejects(
        self, setup, tmp_path: Path
    ):
        """Entries were never held to the schema: the writer committed caches
        its own reader refused (T401).  And a cache holds one feature per event
        version (or row): a second was served in the first's place."""
        task, _ = setup
        source = task.subjects[0].sources[0]
        path = tmp_path / "cohort" / "x.medh5cache"
        with CacheWriter(
            path,
            level="event",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [1]},
        ) as w:
            for bad in (
                SourceRef("", source.content_id),
                SourceRef(source.uri, "not-a-digest"),
            ):
                with pytest.raises(MEDH5ValidationError) as caught:
                    w.add(
                        "e1", np.ones(1, np.float32), sources=[bad], event_id="rep_v1"
                    )
                assert caught.value.code == "T401"
            w.add_event(source, "rep_v1", np.ones(1, np.float32))
            with pytest.raises(MEDH5ValidationError, match="one feature per event"):
                w.add("e2", np.ones(1, np.float32), sources=[source], event_id="rep_v1")
        assert validate_cache(path).ok

    def test_S7_1_two_entries_for_one_row_are_corrupt(self, setup, tmp_path: Path):
        """Of two entries for one row, a lookup served the last, and the other's
        checksum was never compared: a wrong one hid behind its twin."""
        task, _ = setup
        path = tmp_path / "cohort" / "rows.medh5cache"
        with CacheWriter(
            path,
            level="patient",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
            task=task,
        ) as w:
            w.add_row(task.preflight().row("P-01@24h"), np.zeros(2, np.float32))

        def twin(doc: dict[str, Any]) -> None:
            first = doc["entries"][0]
            doc["entries"].append(
                dict(first, entry_id="twin", digest="sha256:" + "f" * 64)
            )

        self._rewrite_manifest(path, twin)
        report = validate_cache(path, task=task)
        assert report.corrupt == ("manifest",) and found(report.findings) == ["T401"]
        with pytest.raises(MEDH5ValidationError, match="two entries for row"):
            FeatureCache.open(path)

    def test_S7_a_document_cache_encodes_only_the_pinned_versions(
        self, setup, tmp_path: Path
    ):
        """A source changed since its pin had its new text encoded under the
        old version's entries --- which, the file restored, validated clean."""
        task, paths = setup
        with medh5.amend(paths["P-02"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        path = tmp_path / "cohort" / "docs.medh5cache"
        with pytest.raises(MEDH5ValidationError) as caught:
            build_document_cache(task, path, HashingTextEncoder(dim=4))
        assert caught.value.code == "T302" and "P-02.medh5" in str(caught.value)
        assert not path.exists()

    def test_a_writer_finished_in_its_block_is_not_finished_again(
        self, setup, tmp_path: Path
    ):
        """`commit()` or `abort()` inside `with` raised "this cache writer has
        finished" as the block closed."""
        task, _ = setup
        source = task.subjects[0].sources[0]
        header: dict[str, Any] = {
            "level": "event",
            "encoder": {"name": "fixture", "revision": "1"},
            "output": {"dtype": "float32", "shape": [1]},
        }
        committed = tmp_path / "cohort" / "c.medh5cache"
        with CacheWriter(committed, **header) as w:
            w.add_event(source, "rep_v1", np.ones(1, np.float32))
            digest = w.commit()
        assert validate_cache(committed).manifest_digest == digest
        aborted = tmp_path / "cohort" / "a.medh5cache"
        with CacheWriter(aborted, **header) as w:
            w.add_event(source, "rep_v1", np.ones(1, np.float32))
            w.abort()
        assert not aborted.exists()

    def test_S8_a_path_naming_no_cache_is_refused_not_corrupt(
        self, tmp_path: Path, capsys
    ):
        """A mistyped path was "1 corrupt" (T401): the cache's own bytes
        wrong.  It is refused as opening it is; a file that is not a cache is
        still corrupt."""
        missing = tmp_path / "nope.medh5cache"
        with pytest.raises(medh5.MEDH5FileError, match="no such file"):
            validate_cache(missing)
        assert main(["cache", "validate", str(missing)]) == EXIT_ERROR
        assert "no such file" in capsys.readouterr().err
        garbage = tmp_path / "garbage.medh5cache"
        garbage.write_bytes(b"not a cache")
        assert validate_cache(garbage).corrupt == ("manifest",)

    def test_S7_2_a_cutoff_that_is_not_an_integer_is_refused_as_such(
        self, setup, tmp_path: Path
    ):
        """The writer read a float cutoff as none, and refused the entry for
        pinning no cutoff (T404); it is the cutoff that is wrong."""
        task, _ = setup
        row = task.preflight().row("P-01@24h")
        writer = CacheWriter(
            tmp_path / "cohort" / "rows.medh5cache",
            level="patient",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
            task=task,
        )
        with pytest.raises(MEDH5ValidationError, match="cutoff_us") as caught:
            writer.add(
                row.row_id,
                np.zeros(2, np.float32),
                sources=list(row.sources),
                row_id=row.row_id,
                row_fingerprint=row.fingerprint,
                cutoff_us=float(row.cutoff_us),  # type: ignore[arg-type]
                event_versions=[e.event_id for e in row.events],
            )
        writer.abort()
        assert caught.value.code == "T401"

    def test_S8_a_row_entrys_versions_are_compared_as_a_set(
        self, setup, tmp_path: Path
    ):
        """The contract fixes no order for an entry's `event_versions`: the
        versions its row admits, listed in any order, are admissible, and only
        one missing or extra is T406.  Any order but the engine's was T406."""
        task, _ = setup
        row = task.preflight().row("P-02@24h")
        admitted = [e.event_id for e in row.events]
        assert admitted == ["lab0", "ct0", "rep_v1"]

        def findings(name: str, versions: list[str]) -> list[str]:
            path = tmp_path / "cohort" / f"{name}.medh5cache"
            with CacheWriter(
                path,
                level="patient",
                encoder={"name": "fixture", "revision": "1"},
                output={"dtype": "float32", "shape": [2]},
                task=task,
            ) as w:
                w.add(
                    row.row_id,
                    np.zeros(2, np.float32),
                    sources=list(row.sources),
                    row_id=row.row_id,
                    row_fingerprint=row.fingerprint,
                    cutoff_us=row.cutoff_us,
                    event_versions=versions,
                )
            return found(validate_cache(path, task=task).findings)

        assert findings("sorted", sorted(admitted)) == []
        assert findings("reversed", admitted[::-1]) == []
        assert findings("missing", admitted[:2]) == ["T406"]
        assert findings("extra", [*admitted, "rep_v2"]) == ["T406"]

    def test_S8_an_entry_its_source_cannot_verify_is_T407(self, setup, tmp_path: Path):
        """An event-level entry is checked against its source.  A source whose
        clinical tables are malformed but sealed passes its pin, which compares
        digests only; its entries were accepted unchecked --- one naming an
        event the source does not have among them."""
        _, paths = setup
        path = paths["P-01"]
        column = "clinical/events/value_num"
        with h5py.File(path, "r+") as f:
            values = f[column][...]
            attrs = dict(f[column].attrs)
            del f[column]
            f.create_dataset(column, data=values.astype(np.float32))
            f[column].attrs.update(attrs)
        TestSources._seal(path, column)
        source = SourceRef.pin(path, uri=path.name)
        cache = tmp_path / "cohort" / "ghost.medh5cache"
        with CacheWriter(
            cache,
            level="event",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
        ) as w:
            w.add_event(source, "ghost", np.zeros(2, np.float32), document_id="nothing")
        report = validate_cache(cache)
        (finding,) = report.findings
        assert finding.code == "T407" and "cannot be read" in finding.message
        assert not report.ok

    def test_S8_a_manifest_that_fails_its_checksum_is_T401(self, setup, tmp_path: Path):
        task, _ = setup
        path = tmp_path / "cohort" / "docs.medh5cache"
        build_document_cache(task, path, HashingTextEncoder(dim=4))
        with h5py.File(path, "r+") as f:
            f.attrs["manifest_digest"] = "sha256:" + "0" * 64
        report = validate_cache(path)
        assert report.corrupt == ("manifest",) and found(report.findings) == ["T401"]
        with pytest.raises(MEDH5ValidationError) as caught:
            FeatureCache.open(path)
        assert caught.value.code == "T401"

    def test_S7_1_a_moved_cache_validates_through_its_task(self, setup, tmp_path: Path):
        """An entry copies its task's URI.  Once the samples moved and the
        task's URIs followed them, every entry was stale (T403) though no
        content changed: a validator given the task now opens each entry
        source through the task's reference pinning it."""
        task, paths = setup
        cache = tmp_path / "cohort" / "docs.medh5cache"
        assert build_document_cache(task, cache, HashingTextEncoder(dim=4)).ok
        moved = tmp_path / "moved"
        moved.mkdir()
        for path in paths.values():
            path.rename(moved / path.name)
        doc = task.to_json()
        for subject in doc["subjects"]:
            for source in subject["sources"]:
                source["uri"] = f"../moved/{source['uri']}"
        repointed = TaskManifest(doc, base=tmp_path / "cohort")
        report = validate_cache(cache, task=repointed, check_rows=False, confine=False)
        assert report.ok, report.findings
        alone = validate_cache(cache)
        assert len(alone.stale) == alone.entries == 6, "without the task: by URI"

    def test_S7_1_an_entry_without_sources_is_T401(self, tmp_path: Path):
        """The schema requires a source, and a writer refuses an entry that
        would fail its schema (T401): it was refused as stale (T403)."""
        with CacheWriter(
            tmp_path / "x.medh5cache",
            level="event",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
        ) as writer:
            with pytest.raises(MEDH5ValidationError) as caught:
                writer.add("x", np.zeros(2, np.float32), sources=[], event_id="e")
            assert caught.value.code == "T401"
            writer.abort()

    def test_S7_1_an_entry_reads_as_the_engine_reads_it(self, setup):
        """``CacheEntry.from_json`` read leniently what the engine refuses: a
        float cutoff, a source list that is not a list."""
        task, _ = setup
        source = task.subjects[0].sources[0]
        doc = {
            "entry_id": "e",
            "sources": [
                {"uri": source.uri, "sample_key": None, "content_id": source.content_id}
            ],
            "event_id": None,
            "document_id": None,
            "row_id": "P-01@24h",
            "cutoff_us": 24 * HOUR,
            "event_versions": ["rep_v1"],
            "row_fingerprint": "sha256:" + "0" * 64,
            "digest": "sha256:" + "1" * 64,
        }
        entry = CacheEntry.from_json(doc)
        assert entry.sources == (SourceRef(source.uri, source.content_id),)
        assert _core.cache_entry_json(doc) == doc, "round-trips unchanged"
        for member, wrong in [("cutoff_us", 1.5), ("sources", "ct.medh5")]:
            with pytest.raises(MEDH5ValidationError) as caught:
                CacheEntry.from_json({**doc, member: wrong})
            assert caught.value.code == "T401", member

    def test_S8_a_report_says_what_the_engine_says(self):
        """``ok``, ``stale`` and ``corrupt`` re-computed the engine's verdict."""
        from medh5.cache import CacheReport

        report = CacheReport(
            "c.medh5cache",
            3,
            (
                Finding("T403", "b", "gone"),
                Finding("T403", "a", "gone"),
                Finding("T402", "manifest", "bytes"),
                Finding("T406", "c", "inadmissible"),
            ),
        )
        assert not report.ok
        assert report.stale == ("a", "b") and report.corrupt == ("manifest",)
        assert report.to_json()["findings"][0]["summary"] == CODES["T403"]
        assert CacheReport("c", 0, ()).ok

    def test_S7_entries_match_the_declared_output(self, tmp_path: Path, setup):
        task, _ = setup
        source = task.subjects[0].sources[0]
        path = tmp_path / "x.medh5cache"
        writer = CacheWriter(
            path,
            level="event",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
        )
        with pytest.raises(MEDH5ValidationError) as caught:
            writer.add_event(source, "rep_v1", np.zeros(3, np.float32))
        assert caught.value.code == "T402"
        writer.abort()
        assert not path.exists()
        with pytest.raises(MEDH5ValidationError) as caught:
            CacheWriter(
                path,
                level="patient",
                encoder={"name": "f", "revision": "1"},
                output={"dtype": "float32", "shape": [2]},
            )
        assert caught.value.code == "T404"

    def test_S7_learned_preprocessing_needs_a_split(self):
        task = TaskManifest.new("t", "1", identity_namespace="n")
        with pytest.raises(MEDH5ValidationError) as caught:
            fitted_on(task)
        assert caught.value.code == "T405"

    def test_a_closed_cache_says_so(self, setup, tmp_path: Path):
        task, _ = setup
        path = tmp_path / "cohort" / "d.medh5cache"
        build_document_cache(task, path, HashingTextEncoder(dim=4))
        cache = FeatureCache.open(path)
        cache.close()
        with pytest.raises(medh5.MEDH5FileError):
            cache.path  # noqa: B018
        cache.close()
        abandoned = FeatureCache.open(path)
        abandoned.abandon()

    def test_S8_a_row_feature_holds_for_the_row_it_was_built_from(self, tmp_path: Path):
        """An honest patient-level feature --- the mean of the CT the row's slot
        reads --- built while the subject had one fragment still validated
        after a second fragment of the same visit was added and the slot read
        that one instead: the cutoff and the admitted versions were the same,
        the image and so the feature were not.  The
        entry pins the fingerprint of the row it was built from; a relocated
        copy of unchanged content is the same row."""
        root = tmp_path / "b04"
        root.mkdir()
        refs = {}
        for name, value in (("a", 11), ("b", 93)):
            History.write(root / f"{name}.medh5", subject_id="P-01", fill=value)
            refs[name] = SourceRef.pin(root / f"{name}.medh5", uri=f"{name}.medh5")
        # With both listed the slot reads the smaller pin; the feature is
        # first built from the other.
        later = min(refs, key=lambda n: refs[n].content_id)
        first = "b" if later == "a" else "a"

        def task_of(*sources: SourceRef) -> TaskManifest:
            task = TaskManifest.new(
                "t", "1", identity_namespace="n", slots=[Slot("ct", "CT")], base=root
            )
            task.add_subject("P-01", list(sources))
            task.add_row("r", "P-01", 24 * HOUR)
            return task.reconcile()

        def honest(task: TaskManifest, path: Path) -> float:
            row = task.preflight().row("r")
            fill = row.slots["ct"]
            assert fill.fragment is not None and fill.image_id is not None
            read = row.sources[fill.fragment]
            with read.open(root) as sample:
                mean = float(np.mean(sample.images[fill.image_id].read()))
            with CacheWriter(
                path,
                level="patient",
                encoder={"name": "mean-ct", "revision": "1"},
                output={"dtype": "float32", "shape": [1]},
                task=task,
            ) as w:
                w.add_row(row, np.full(1, mean, np.float32), sources=[read])
            return mean

        alone = task_of(refs[first])
        cache = root / "rows.medh5cache"
        assert honest(alone, cache) == (11.0 if first == "a" else 93.0)
        assert validate_cache(cache, task=alone).ok
        both = task_of(refs[first], refs[later])
        assert honest(both, root / "now.medh5cache") != honest(alone, cache)
        findings = validate_cache(cache, task=both).findings
        assert found(findings) == ["T404"]
        assert "built for row 'r'" in str(findings[0])
        # The same content elsewhere is the same row.
        moved = root / "moved"
        moved.mkdir()
        shutil.copyfile(root / f"{first}.medh5", moved / "copy.medh5")
        relocated = task_of(SourceRef.pin(moved / "copy.medh5", uri="moved/copy.medh5"))
        assert relocated.preflight().row("r").fingerprint == (
            alone.preflight().row("r").fingerprint
        )
        assert validate_cache(cache, task=relocated).ok

    def test_S7_2_an_event_feature_names_no_row(self, setup, tmp_path: Path):
        """An event-level entry declaring a later row, its cutoff and its
        history validated against that row, while a lookup by event version
        --- which `documents=` makes --- served it to every row.  An event
        feature names no row: the writer refuses
        one, and a manifest declaring one fails its schema."""
        task, _ = setup
        later = task.preflight().row("P-01@d95")
        source = task.subjects[0].sources[0]
        path = tmp_path / "cohort" / "events.medh5cache"
        header: dict[str, Any] = {
            "level": "event",
            "encoder": {"name": "fixture", "revision": "1"},
            "output": {"dtype": "float32", "shape": [1]},
        }
        row_fields: dict[str, Any] = {
            "row_id": later.row_id,
            "cutoff_us": later.cutoff_us,
            "event_versions": [e.event_id for e in later.events],
        }
        with (
            pytest.raises(MEDH5ValidationError, match="no row") as caught,
            CacheWriter(path, **header) as w,
        ):
            w.add(
                "e1",
                np.ones(1, np.float32),
                sources=[source],
                event_id="rep_v1",
                **row_fields,
            )
        assert caught.value.code == "T404" and not path.exists()
        with CacheWriter(path, **header) as w:
            w.add_event(source, "rep_v1", np.ones(1, np.float32))
        assert validate_cache(path, task=task).ok  # stateless reuse
        self._rewrite_manifest(path, lambda doc: doc["entries"][0].update(row_fields))
        report = validate_cache(path, task=task)
        assert found(report.findings) == ["T401"] and report.corrupt == ("manifest",)

    @staticmethod
    def _rewrite_manifest(path: Path, change: Any) -> None:
        """Edit a cache's manifest and re-checksum it, as a writer would."""
        import hashlib

        from medh5._core import canonical_json

        with h5py.File(path, "r+") as f:
            doc = json.loads(f["manifest"][()].decode())
            change(doc)
            text = canonical_json(doc)
            del f["manifest"]
            f.create_dataset(
                "manifest", data=text.decode(), dtype=h5py.string_dtype("utf-8")
            )
            f.attrs["manifest_digest"] = "sha256:" + hashlib.sha256(text).hexdigest()

    def test_S8_a_row_feature_reads_only_its_subjects_sources(
        self, setup, tmp_path: Path
    ):
        """Event ids are local to a sample, so two patients' rows admit the same
        ids at one cutoff: an entry for one row pinned to the other patient's
        source validated against the task, and served that patient's feature."""
        task, paths = setup
        report = task.preflight()
        mine, theirs = report.row("P-01@24h"), report.row("P-02@24h")
        assert [e.event_id for e in mine.events] == [e.event_id for e in theirs.events]
        header = {
            "level": "patient",
            "encoder": {"name": "fixture", "revision": "1"},
            "output": {"dtype": "float32", "shape": [2]},
            "task": task,
        }

        def write(path: Path, sources: list[SourceRef]) -> Path:
            with CacheWriter(path, **header) as w:
                w.add(
                    mine.row_id,
                    np.zeros(2, np.float32),
                    sources=sources,
                    row_id=mine.row_id,
                    row_fingerprint=mine.fingerprint,
                    cutoff_us=mine.cutoff_us,
                    event_versions=[e.event_id for e in mine.events],
                )
            return path

        crossed = write(
            tmp_path / "cohort" / "crossed.medh5cache", list(theirs.sources)
        )
        findings = validate_cache(crossed, task=task).findings
        assert found(findings) == ["T404"]
        assert "does not pin" in str(findings[0])
        # A relocated copy of the row's own source is the same version.
        moved = tmp_path / "elsewhere" / "P-01.medh5"
        moved.parent.mkdir()
        shutil.copyfile(paths["P-01"], moved)
        relocated = SourceRef.pin(moved, uri=str(moved))
        assert relocated.content_id == mine.sources[0].content_id
        own = write(tmp_path / "cohort" / "own.medh5cache", [relocated])
        assert validate_cache(own, task=task).ok

    def test_S7_1_an_entry_is_the_layout_its_cache_declares(
        self, setup, tmp_path: Path
    ):
        """The checksum vouches for the bytes, not for the declaration: a header
        relabelled `float64[999]` validated and served `float32[2]`."""
        task, _ = setup
        source = task.subjects[0].sources[0]
        path = tmp_path / "cohort" / "events.medh5cache"
        with CacheWriter(
            path,
            level="event",
            encoder={"name": "fixture", "revision": "1"},
            output={"dtype": "float32", "shape": [2]},
        ) as w:
            w.add_event(source, "rep_v1", np.ones(2, np.float32))
        for output in (
            {"dtype": "float64", "shape": [2]},
            {"dtype": "float32", "shape": [3]},
        ):
            self._rewrite_manifest(path, lambda doc, o=output: doc.update(output=o))
            report = validate_cache(path)
            assert found(report.findings) == ["T402"] and report.corrupt
            with (
                FeatureCache.open(path) as cache,
                pytest.raises(MEDH5ValidationError) as caught,
            ):
                cache.event_feature(source.content_id, "rep_v1")
            assert caught.value.code == "T402"

    def test_S7_2_a_cache_holds_what_its_level_requires(self, setup, tmp_path: Path):
        """The writer required a patient cache to name its task and every entry
        its row, cutoff and selection, and nothing read it back: a cache
        relabelled `patient` with none of it validated against the task."""
        task, _ = setup
        source = task.subjects[0].sources[0]
        header = {
            "encoder": {"name": "fixture", "revision": "1"},
            "output": {"dtype": "float32", "shape": [1]},
        }
        path = tmp_path / "cohort" / "relabelled.medh5cache"
        with CacheWriter(path, level="event", **header) as w:
            w.add_event(source, "rep_v1", np.ones(1, np.float32))
        self._rewrite_manifest(path, lambda doc: doc.update(level="patient"))
        assert found(validate_cache(path, task=task).findings) == ["T401"]
        writer = CacheWriter(
            tmp_path / "cohort" / "x.medh5cache", level="event", **header
        )
        with pytest.raises(MEDH5ValidationError) as caught:
            writer.add("x", np.ones(1, np.float32), sources=[source])
        writer.abort()
        assert caught.value.code == "T404", "an event-level entry names its event"
        # Fitted on another split with the same training subjects is still
        # another split: the record names it.
        foreign = tmp_path / "cohort" / "foreign.medh5cache"
        record = dict(fitted_on(task), set_id="another-split")
        with CacheWriter(foreign, level="event", fitted_on=record, **header) as w:
            w.add_event(source, "rep_v1", np.ones(1, np.float32))
        findings = validate_cache(foreign, task=task).findings
        assert found(findings) == ["T405"] and "set_id" in str(findings[0])

    @pytest.mark.parametrize("cutoff", [2**63, -(2**63) - 1, 3600000000.0])
    def test_S3_1_a_cutoff_is_a_64_bit_integer(self, setup, cutoff):
        """JSON Schema's `integer` admits `3600000000.0`, which serialisers
        write for integers, and values past int64; both read as cutoff 0 and
        validated --- moving the row, and what it may read."""
        task, _ = setup
        doc = task.to_json()
        bad = dict(doc, rows=[dict(doc["rows"][0], cutoff_us=cutoff)])
        with pytest.raises(MEDH5ValidationError) as caught:
            TaskManifest(bad, base=task.base)
        assert caught.value.code == "T101"

    @pytest.mark.parametrize("cutoff", [2**63 - 1, -(2**63)])
    def test_S3_1_both_int64_endpoints_are_cutoffs(self, setup, cutoff):
        task, _ = setup
        doc = task.to_json()
        fine = TaskManifest(dict(doc, rows=[dict(doc["rows"][0], cutoff_us=cutoff)]))
        assert fine.validate() == [] and fine.rows[0].cutoff_us == cutoff

    def test_findings_print_their_code(self):
        finding = Finding("T302", "P-01", "changed")
        assert str(finding) == "T302 P-01: changed"
        assert "pinned" in finding.summary

    def test_S9_the_code_table_is_the_engines(self):
        """§9 was a marker the site replaced: a reader of the text --- on GitHub,
        in the sdist --- found no table.  It is written out, and held to the
        engine's ``CODES`` here."""
        spec = Path(__file__).resolve().parents[2] / "docs/spec/task-cache-1.md"
        text = spec.read_text(encoding="utf-8")
        section = text.split("## 9. Finding codes", 1)[1].split("\n## ", 1)[0]
        rows = re.findall(
            r'^\| <a id="(t\d{3})"></a>`(T\d{3})` \| (.*) \|$', section, re.M
        )
        assert {code: meaning for _, code, meaning in rows} == CODES
        assert all(anchor == code.lower() for anchor, code, _ in rows)

    def test_S9_T404_says_what_it_is_found_for(self):
        """§6 and §8 find T404 for another task's cache, another manifest's
        preflight, another row or row version, and a cache that changed after
        it was validated; a cache read under another selection is T406."""
        summary = Finding("T404", "cache", "x").summary
        for use in ("task", "manifest", "preflight", "row version", "validated"):
            assert use in summary
        assert "selection" not in summary


class TestCommandLine:
    def test_task_and_cache_commands(self, setup, tmp_path: Path, capsys):
        task, paths = setup
        manifest = task.save(tmp_path / "cohort" / "t.task.json")
        assert main(["task", "validate", str(manifest)]) == EXIT_OK
        assert "OK" in capsys.readouterr().out
        assert main(["task", "validate", str(manifest), "--json"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["ok"] is True
        assert main(["task", "preflight", str(manifest)]) == EXIT_OK
        out = capsys.readouterr().out
        assert "eligible" in out and "prevalent_target" in out
        assert main(["task", "preflight", str(manifest), "--json", "--deep"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["ok"] is True
        reconciled = tmp_path / "cohort" / "r.task.json"
        assert main(["task", "reconcile", str(manifest), "--out", str(reconciled)]) == 0
        assert "recorded 0" in capsys.readouterr().out
        cache = tmp_path / "cohort" / "d.medh5cache"
        build_document_cache(task, cache, HashingTextEncoder(dim=4))
        assert (
            main(["cache", "validate", str(cache), "--task", str(manifest)]) == EXIT_OK
        )
        assert "OK" in capsys.readouterr().out
        with medh5.amend(paths["P-02"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        assert main(["cache", "validate", str(cache), "--json"]) == EXIT_ERROR
        assert json.loads(capsys.readouterr().out)["stale"]
        assert main(["task", "preflight", str(manifest)]) == EXIT_ERROR
        assert "T302" in capsys.readouterr().out
        broken = tmp_path / "broken.json"
        broken.write_text("[]", encoding="utf-8")
        assert main(["task", "validate", str(broken)]) == EXIT_ERROR
        assert "T101" in capsys.readouterr().out

    @pytest.mark.parametrize("alias", ["exact", "relative", "symlink", "hardlink"])
    def test_S3_3_reconcile_never_writes_over_a_source(
        self, setup, tmp_path: Path, capsys, alias: str
    ):
        """``task reconcile -o`` saved the manifest over whatever ``--out``
        named, a source sample too, with exit status 0 (F11). A source, by
        any path or link, is refused before anything is written."""
        task, paths = setup
        manifest = task.save(tmp_path / "cohort" / "t.task.json")
        source = paths["P-02"]
        if alias == "exact":
            out = source
        elif alias == "relative":
            (tmp_path / "sub").mkdir()
            out = tmp_path / "sub" / ".." / "cohort" / source.name
        else:
            out = tmp_path / "alias.medh5"
            try:
                (out.symlink_to if alias == "symlink" else out.hardlink_to)(source)
            except OSError as exc:  # pragma: no cover - no links on this file system
                pytest.skip(str(exc))
        before = source.read_bytes(), manifest.read_bytes()
        capsys.readouterr()
        assert main(["task", "reconcile", str(manifest), "-o", str(out)]) == EXIT_ERROR
        assert "a source of subject 'P-02'" in capsys.readouterr().err
        assert (source.read_bytes(), manifest.read_bytes()) == before
        elsewhere = tmp_path / "r.task.json"
        assert main(["task", "reconcile", str(manifest), "-o", str(elsewhere)]) == 0
        assert TaskManifest.load(elsewhere).validate() == []

    def test_S3_1_a_manifest_that_cannot_be_saved_is_named(
        self, setup, tmp_path: Path, capsys
    ):
        """The atomic save said only "No such file or directory (os error 2)"
        where the command line names the file, as for every other output."""
        task, _ = setup
        manifest = task.save(tmp_path / "cohort" / "t.task.json")
        out = tmp_path / "nodir" / "r.task.json"
        capsys.readouterr()
        assert main(["task", "reconcile", str(manifest), "-o", str(out)]) == EXIT_ERROR
        assert repr(str(out)) in capsys.readouterr().err

    def test_S3_3_reconcile_never_writes_over_a_collection_it_reads(
        self, setup, tmp_path: Path, capsys
    ):
        _, paths = setup
        root = tmp_path / "cohort"
        shard = pack([paths["P-01"]], root / "s.medh5c", keys=["P-01"])
        member = SourceRef.pin(shard, sample_key="P-01", uri=shard.name)
        manifest = task_for(root, paths, refs={"P-01": member}).save(root / "t.json")
        before = shard.read_bytes()
        capsys.readouterr()
        argv = ["task", "reconcile", str(manifest), "-o", str(shard)]
        assert main(argv) == EXIT_ERROR
        assert "'s.medh5c::P-01', a source" in capsys.readouterr().err
        assert shard.read_bytes() == before

    def test_preflight_does_not_depend_on_the_worker_count(
        self, setup, tmp_path: Path, capsys, monkeypatch
    ):
        """``task preflight --workers N`` reads the subjects in up to N
        processes and prints what one process prints, a stale pin (T302)
        included; a subject whose worker dies stops the command there."""
        task, paths = setup
        manifest = task.save(tmp_path / "cohort" / "t.task.json")
        with medh5.amend(paths["P-02"]) as w:
            w.add_event(
                Event("late", "late", "other", "static", "final", available_us=0)
            )
        capsys.readouterr()
        for extra in ([], ["--json"], ["--json", "--deep"]):
            seen = set()
            for workers in ([], ["--workers", "1"], ["--workers", "3"]):
                code = main(["task", "preflight", str(manifest), *extra, *workers])
                printed = capsys.readouterr()
                seen.add((code, printed.out, printed.err))
            assert len(seen) == 1, sorted(seen)
            ((code, out, _),) = seen
            assert code == EXIT_ERROR
            assert "T302" in out
        monkeypatch.setenv("MEDH5_TEST_WORKERS", "exit:P-03")
        argv = ["task", "preflight", str(manifest), "--workers", "2"]
        assert main(argv) == EXIT_ERROR
        assert "the worker reading it stopped" in capsys.readouterr().err

    def test_task_validate_notes_a_defaulted_follow_up(
        self, setup, tmp_path: Path, capsys
    ):
        """``task validate`` and ``task preflight`` print the note after the
        findings, and ``--json`` carries it; neither ``ok`` nor the exit
        status changes."""
        task, _ = setup
        doc = task.to_json()
        del doc["target"]["min_follow_up_us"]
        path = tmp_path / "cohort" / "noted.task.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        capsys.readouterr()
        note = "note /target/min_follow_up_us: "
        for command in ("validate", "preflight"):
            assert main(["task", command, str(path)]) == EXIT_OK
            out = capsys.readouterr().out
            assert ": OK (" in out and f"  {note}" in out
            assert main(["task", command, str(path), "--json"]) == EXIT_OK
            report = json.loads(capsys.readouterr().out)
            assert report["ok"] is True
            expected = TaskManifest.load(path).notes[0]
            assert report["notes"] == [
                {"location": expected.location, "message": expected.message}
            ]
        stated = task.save(tmp_path / "cohort" / "stated.task.json")
        assert main(["task", "validate", str(stated), "--json"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["notes"] == []

    def test_cutoff_hours_rounds_as_hours_does(self, setup, capsys):
        """1.25e-09 hours is 4.5 us: ``hours`` gives 4 (half to even), and the
        command line gave 5 (half away from zero)."""
        from medh5.clinical import hours

        _, paths = setup
        argv = ["clinical", "select", str(paths["P-01"]), "--json"]
        assert main([*argv, "--cutoff-hours=1.25e-09"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["cutoff_us"] == hours(1.25e-09) == 4

    def test_cache_ls_and_info(self, setup, tmp_path: Path, capsys):
        task, _ = setup
        cache = tmp_path / "cohort" / "docs.medh5cache"
        build_document_cache(task, cache, HashingTextEncoder(dim=4))
        source = task.subjects[0].sources[0]
        capsys.readouterr()
        assert main(["cache", "info", str(cache), "--json"]) == EXIT_OK
        info = json.loads(capsys.readouterr().out)
        assert (info["level"], info["entries"], info["sources"]) == ("event", 6, 3)
        assert info["encoder"]["name"] == HashingTextEncoder.NAME
        assert info["output"]["shape"] == [4]
        assert main(["cache", "info", str(cache)]) == EXIT_OK
        out = capsys.readouterr().out
        assert "event-level, 6 entries over 3 source versions" in out
        assert main(["cache", "ls", str(cache), "--json"]) == EXIT_OK
        assert len(json.loads(capsys.readouterr().out)) == 6
        argv = ["cache", "ls", str(cache), "--source", source.content_id, "--json"]
        assert main(argv) == EXIT_OK
        listed = json.loads(capsys.readouterr().out)
        assert len(listed) == 2
        assert {e["sources"][0]["content_id"] for e in listed} == {source.content_id}
        with open_cache(cache) as opened:
            assert [CacheEntry.from_json(e) for e in listed] == [
                e
                for e in opened.entries
                if e.sources[0].content_id == source.content_id
            ], "the entries the reader holds"
        assert main(["cache", "ls", str(cache)]) == EXIT_OK
        out = capsys.readouterr().out
        assert "event document" in out and "rep_v1" in out
        assert main(["cache", "ls", str(cache), "--row", "P-01@24h", "--json"]) == 0
        assert json.loads(capsys.readouterr().out) == []

    def test_cache_validate_task_base(self, setup, tmp_path: Path, capsys):
        """``--task-base`` locates the task's sources when the manifest is
        kept away from them, as ``task preflight --base`` does."""
        task, _ = setup
        cohort = tmp_path / "cohort"
        cache = cohort / "docs.medh5cache"
        build_document_cache(task, cache, HashingTextEncoder(dim=4))
        elsewhere = tmp_path / "tasks"
        elsewhere.mkdir()
        manifest = task.save(elsewhere / "t.task.json")
        argv = ["cache", "validate", str(cache), "--task", str(manifest), "--json"]
        capsys.readouterr()
        assert main(argv) == EXIT_ERROR, "the sources are not beside the manifest"
        capsys.readouterr()
        assert main([*argv, "--task-base", str(cohort)]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["ok"] is True
        assert main(["cache", "validate", str(cache), "--task-base", str(cohort)]) == (
            EXIT_USAGE
        ), "--task-base without --task"

    def test_cutoff_hours_takes_a_negative_exponent_after_equals(self, setup, capsys):
        """``--cutoff-hours -5.5e-07`` reads as a flag; the documented
        ``--cutoff-hours=-5.5e-07`` is the cutoff."""
        from medh5.clinical import hours

        _, paths = setup
        argv = ["clinical", "select", str(paths["P-01"]), "--json"]
        assert main([*argv, "--cutoff-hours=-5.5e-07"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["cutoff_us"] == hours(-5.5e-07)
        assert main([*argv, "--cutoff-hours", "-2"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["cutoff_us"] == hours(-2)

    def test_task_preflight_source_root_and_any_source_path(
        self, tmp_path: Path, capsys
    ):
        manifest, _ = outside_task(tmp_path)
        argv = ["task", "preflight", str(manifest)]
        assert main(argv) == EXIT_OK
        assert "note: 3 source(s) lie outside" in capsys.readouterr().err
        assert main([*argv, "--any-source-path"]) == EXIT_OK
        assert capsys.readouterr().err == ""
        assert main([*argv, "--source-root", str(tmp_path)]) == EXIT_OK
        assert capsys.readouterr().err == ""
        assert main([*argv, "--source-root", str(manifest.parent), "--json"]) == (
            EXIT_ERROR
        )
        report = json.loads(capsys.readouterr().out)
        assert {f["code"] for f in report["findings"]} == {"T301"}
        both = [*argv, "--source-root", str(tmp_path), "--any-source-path"]
        assert main(both) == EXIT_USAGE
        capsys.readouterr()
        reconcile = ["task", "reconcile", str(manifest)]
        assert main([*reconcile, "--source-root", str(manifest.parent)]) == (EXIT_ERROR)
        assert "outside the directory" in capsys.readouterr().err

    def test_clinical_commands(self, setup, tmp_path: Path, capsys):
        _, paths = setup
        path = str(paths["P-01"])
        assert main(["clinical", "show", path]) == EXIT_OK
        out = capsys.readouterr().out
        assert "rep_v1" in out and "clock" in out
        assert main(["clinical", "show", path, "--json"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out)["events"] == 6
        assert main(["clinical", "select", path, "--cutoff-hours", "24"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "certified" in out and "rep_v2" not in out
        assert (
            main(["clinical", "select", path, "--cutoff-us", "0", "--json"]) == EXIT_OK
        )
        assert json.loads(capsys.readouterr().out)["status"] == "certified"
        assert main(["clinical", "select", path]) == EXIT_USAGE
        capsys.readouterr()
        records = tmp_path / "records.json"
        assert main(["clinical", "export", path, "--out", str(records)]) == EXIT_OK
        capsys.readouterr()
        stripped = tmp_path / "imaging.medh5"
        assert main(["clinical", "strip", path, "--out", str(stripped)]) == EXIT_OK
        assert "removed 6 events" in capsys.readouterr().out
        back = tmp_path / "back.medh5"
        args = ["clinical", "augment", str(stripped), str(records), "--out", str(back)]
        assert main(args) == EXIT_OK
        assert "1.0 -> 1.1" in capsys.readouterr().out
        with medh5.open(back) as s:
            assert s.clinical is not None and len(s.clinical.events) == 6
        assert main(["clinical", "show", str(stripped)]) == EXIT_ERROR
        assert "does not declare the clinical profile" in capsys.readouterr().err


def placed(task: TaskManifest) -> list[tuple[str, str | None]]:
    return [(s.subject_id, s.partition) for s in task.subjects]


class TestFromCohort:
    """A task drawn from a cohort manifest and its split, or its claims."""

    @pytest.fixture
    def scanned(self, tmp_path: Path) -> tuple[Any, Path]:
        from medh5.dataset import scan

        root = tmp_path / "cohort"
        cohort(root, {"P-01": "PD", "P-02": "SD", "P-03": "SD", "P-04": "PR"})
        manifest, failed = scan(root)
        assert failed == ()
        return manifest, root

    @staticmethod
    def split_of(manifest: Any, places: dict[str, tuple[str, int | None]]) -> Any:
        from medh5.dataset import Assignment, Split

        return Split(
            "s1",
            manifest.sha256(),
            [
                Assignment(e.subject_id, *places[e.subject_id], entries=(e.path,))
                for e in manifest
            ],
            group_by="subject_id",
            k_folds=2 if any(f is not None for _, f in places.values()) else None,
        )

    def build(self, manifest: Any, split: Any, root: Path, **kw: Any) -> TaskManifest:
        return TaskManifest.from_cohort(
            manifest,
            split,
            task_id="progression",
            version="1",
            identity_namespace="site",
            base=root,
            **kw,
        )

    def test_S3_3_partitions_come_from_the_split(self, scanned):
        manifest, root = scanned
        places = {
            "P-01": ("train", None),
            "P-02": ("train", None),
            "P-03": ("val", None),
            "P-04": ("test", None),
        }
        task = self.build(
            manifest,
            self.split_of(manifest, places),
            root,
            slots=[Slot("ct", "CT", required=True, patch=(4, 8, 8), classes=(3,))],
            target=TARGET,
            description="from a cohort",
        )
        assert placed(task) == [(s, p) for s, (p, _) in places.items()]
        assert task.split == ("s1", ("train", "val", "test"))
        assert [s.sources[0].uri for s in task.subjects] == [
            f"{s}.medh5" for s in places
        ]
        assert task.rows == () and task.validate() == []
        assert task.target == TARGET
        assert task.to_json()["task"]["description"] == "from a cohort"
        task.add_row("P-01@24h", "P-01", 24 * HOUR)
        assert task.preflight().ok

    def test_S3_3_partitions_come_from_this_manifests_claims(self, scanned):
        from medh5.dataset import make_splits, scan, write_claims

        manifest, root = scanned
        split = make_splits(
            manifest, set_id="claimed", group_by="subject_id", algorithm="medh5.split/2"
        )
        write_claims(split, manifest)
        claimed, _ = scan(root)
        task = self.build(claimed, "claimed", root)
        assert placed(task) == [(e.subject_id, split.partition_of(e)) for e in claimed]
        assert task.split is not None and task.split[0] == "claimed"
        # The pins are of the files as claimed, not as first scanned.
        assert [s.sources[0].content_id for s in task.subjects] == [
            e.content_id for e in claimed
        ]

    def test_S3_3_a_claim_of_another_manifest_is_refused(self, scanned):
        from medh5.dataset import make_splits, scan, write_claims

        manifest, root = scanned
        write_claims(
            make_splits(manifest, set_id="claimed", algorithm="medh5.split/2"),
            manifest,
        )
        claimed, _ = scan(root)
        fewer = claimed.filter(lambda e: e.subject_id != "P-04")
        with pytest.raises(MEDH5ValidationError, match="against another manifest"):
            self.build(fewer, "claimed", root)
        with pytest.raises(MEDH5ValidationError, match="has no partition"):
            self.build(claimed, "another-set", root)

    def test_S3_3_a_split_of_another_cohort_is_refused(self, scanned):
        manifest, root = scanned
        places = {e.subject_id: ("train", None) for e in manifest}
        split = self.split_of(manifest, places)
        fewer = manifest.filter(lambda e: e.subject_id != "P-04")
        with pytest.raises(MEDH5ValidationError, match="another cohort manifest"):
            self.build(fewer, split, root)

    def test_S3_3_a_subject_in_two_partitions_is_refused(self, scanned):
        from medh5.dataset import Assignment, scan

        _, root = scanned
        History.write(root / "P-01-b.medh5", subject_id="P-01", response="PD")
        manifest, _ = scan(root)
        split = self.split_of(
            manifest, {e.subject_id: ("train", None) for e in manifest}
        )
        split.group_by = "sample_id"
        split.assignments = [
            Assignment(
                e.sample_id,
                "test" if e.path.endswith("-b.medh5") else "train",
                entries=(e.path,),
            )
            for e in manifest
        ]
        with pytest.raises(MEDH5ValidationError, match="in two partitions"):
            self.build(manifest, split, root)
        for a in split.assignments:
            a.partition = "unassigned" if a.entries[0].endswith("-b.medh5") else "train"
        with pytest.raises(MEDH5ValidationError, match="unassigned"):
            self.build(manifest, split, root)

    def test_S3_3_fold_n_is_val(self, scanned):
        manifest, root = scanned
        places = {
            "P-01": ("holdout", 0),
            "P-02": ("holdout", 1),
            "P-03": ("holdout", 0),
            "P-04": ("holdout", 1),
        }
        split = self.split_of(manifest, places)
        with pytest.raises(MEDH5ValidationError, match="needs a fold"):
            self.build(manifest, split, root)
        task = self.build(manifest, split, root, fold=1)
        assert placed(task) == [
            ("P-01", "train"),
            ("P-02", "val"),
            ("P-03", "train"),
            ("P-04", "val"),
        ]
        with pytest.raises(MEDH5ValidationError, match="not a fold"):
            self.build(manifest, split, root, fold=2)

    def test_S3_3_the_training_partition_comes_first(self, scanned):
        manifest, root = scanned
        places = {
            "P-01": ("train", None),
            "P-02": ("val", None),
            "P-03": ("test", None),
            "P-04": ("test", None),
        }
        split = self.split_of(manifest, places)
        task = self.build(manifest, split, root, training_partition="val")
        assert task.split == ("s1", ("val", "train", "test"))
        absolute = TaskManifest.from_cohort(
            manifest, split, task_id="t", version="1", identity_namespace="ns"
        )
        assert all(Path(s.sources[0].uri).is_absolute() for s in absolute.subjects)
        assert absolute.preflight().ok


def test_task_new_from_a_split(tmp_path: Path, capsys):
    """``medh5 task new``: a task drawn from a cohort and its split, its
    definition from a file and its rows from a CSV."""
    from medh5.dataset import make_splits, scan

    root = tmp_path / "cohort"
    cohort(root)
    manifest, _ = scan(root)
    manifest.save(root / "cohort.json")
    split = make_splits(
        manifest,
        set_id="s1",
        group_by="subject_id",
        k_folds=3,
        algorithm="medh5.split/2",
    )
    (root / "split.json").write_text(json.dumps(split.to_json()))
    definition = TaskManifest.new(
        "progression",
        "1",
        identity_namespace="site",
        slots=[Slot("ct", "CT", required=True, patch=(4, 8, 8), classes=(3,))],
        target=TARGET,
    ).to_json()
    del definition["target"]["min_follow_up_us"]
    (root / "definition.json").write_text(json.dumps(definition))
    (root / "rows.csv").write_text(
        "row_id,subject_id,cutoff_us\nP-01@24h,P-01,86400000000\n"
    )
    out = root / "t.task.json"
    args = [
        "task", "new", str(out),
        "--cohort", str(root / "cohort.json"),
        "--split", str(root / "split.json"),
        "--definition", str(root / "definition.json"),
        "--rows", str(root / "rows.csv"),
    ]  # fmt: skip
    assert main([*args, "--fold", "0"]) == EXIT_ERROR
    assert "min-follow-up-us" in capsys.readouterr().err and not out.exists()
    assert main([*args]) == EXIT_ERROR
    assert "needs a fold" in capsys.readouterr().err
    follow = ["--min-follow-up-us", str(60 * DAY), "--fold", "0", "--json"]
    assert main([*args, *follow]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    task = TaskManifest.load(out)
    assert report["task_fingerprint"] == task.task_fingerprint
    assert report["rows"] == 1 and sum(report["subjects"].values()) == 3
    assert task.split is not None and task.split[1][0] == "train"
    assert [s.sources[0].uri for s in task.subjects] == [
        "P-01.medh5",
        "P-02.medh5",
        "P-03.medh5",
    ]
    assert task.target is not None and task.target.min_follow_up_us == 60 * DAY
    assert task.validate() == [] and task.preflight().ok
    (root / "rows.csv").write_text("row_id,subject_id,cutoff_us\nr,P-99,0\n")
    assert main([*args, *follow]) == EXIT_ERROR
    assert "T201" in capsys.readouterr().err
