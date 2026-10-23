"""MEDS import (1.1 §5, §10): one subject's rows as clinical records, with
nothing invented --- no zone, no origin, no unit, no availability, no code."""

from __future__ import annotations

import datetime
import json
import os
import sys
import zoneinfo
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from medh5.clinical import DAY, HOUR, Clock, day_bounds, utc_us
from medh5.errors import MEDH5Error, MEDH5ValidationError
from medh5.io.meds import augment_from_meds, read_meds
from medh5.sample import open_sample
from medh5.validate import validate_file
from tests.helpers import unrooted
from tests.kits import Meds

NEW_YORK = "America/New_York"


def _utc(**options):
    return {"subject_id": Meds.SUBJECT, "clock": Clock.utc("ehr"), **options}


def _by_code(records):
    return {e.code: e for e in records.events}


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return Meds.write(tmp_path / "meds")


def test_S10_meds_naive_times_need_a_zone(dataset):
    with pytest.raises(MEDH5ValidationError, match="no time zone.*give timezone"):
        read_meds(dataset, **_utc())
    records, report = read_meds(dataset, **_utc(timezone=NEW_YORK))
    lab = _by_code(records)["LOINC/2160-0"]
    expected = utc_us(
        datetime.datetime(2020, 3, 1, 8, 15, tzinfo=zoneinfo.ZoneInfo(NEW_YORK))
    )
    assert lab.temporal_type == "point"
    assert lab.effective_start_us == (expected, expected)
    assert _by_code(records)["GENDER//F"].temporal_type == "static"
    assert not report.guesses


def test_S10_a_reading_that_falls_twice_is_bounded_by_both(tmp_path):
    rows = [
        {
            "subject_id": Meds.SUBJECT,
            "time": datetime.datetime(2020, 11, 1, 1, 30),
            "code": "X",
            "numeric_value": None,
            "text_value": None,
        }
    ]
    records, _ = read_meds(Meds.write(tmp_path, rows), **_utc(timezone=NEW_YORK))
    lo, hi = records.events[0].effective_start_us
    assert hi - lo == HOUR
    rows[0]["time"] = datetime.datetime(2020, 3, 8, 2, 30)
    with pytest.raises(MEDH5ValidationError, match="never occurred"):
        read_meds(Meds.write(tmp_path / "gap", rows), **_utc(timezone=NEW_YORK))


def test_S10_a_relative_clock_needs_an_origin(dataset):
    clock = Clock.relative("ehr", "admission, 2020-03-01 00:00 local")
    with pytest.raises(MEDH5ValidationError, match="invents no origin"):
        read_meds(dataset, subject_id=Meds.SUBJECT, clock=clock)
    records, report = read_meds(
        dataset, subject_id=Meds.SUBJECT, clock=clock, origin="2020-03-01T00:00:00"
    )
    lab = _by_code(records)["LOINC/2160-0"]
    assert lab.effective_start_us == (8 * HOUR + 15 * 60 * 10**6,) * 2
    assert report.of_kind("time")  # counted on one wall clock, and said so
    with pytest.raises(MEDH5ValidationError, match="origin is for a relative clock"):
        read_meds(dataset, **_utc(timezone=NEW_YORK, origin="2020-03-01T00:00:00"))


def test_S5_2_meds_availability_stays_unknown(tmp_path, dataset):
    records, report = read_meds(dataset, **_utc(timezone=NEW_YORK))
    assert all(e.available_us is None for e in records.events)
    (note,) = report.of_kind("availability")
    assert note.severity == "warning" and "never its inputs" in note.message
    known = [datetime.datetime(2020, 3, 6, 9, 0)] * len(Meds.rows())
    declared = Meds.write(
        tmp_path / "declared",
        extra={"reported": (__import__("pyarrow").timestamp("us"), known)},
    )
    records, report = read_meds(
        declared, **_utc(timezone=NEW_YORK, columns={"available": "reported"})
    )
    when = utc_us(datetime.datetime(2020, 3, 6, 9, tzinfo=zoneinfo.ZoneInfo(NEW_YORK)))
    assert {e.available_us for e in records.events} == {(when, when)}
    assert not report.of_kind("availability")


def test_S5_2_a_day_spans_all_of_its_instants(dataset):
    records, _ = read_meds(dataset, **_utc(day_precision=["DX"], timezone=NEW_YORK))
    dx = _by_code(records)["DX//I10"]
    assert dx.effective_start_us == day_bounds(datetime.date(2020, 3, 1), NEW_YORK)
    relative = Clock.relative("ehr", "admission")
    records, _ = read_meds(
        dataset,
        subject_id=Meds.SUBJECT,
        clock=relative,
        origin="2020-03-01T00:00:00",
        day_precision=["DX"],
    )
    assert _by_code(records)["DX//I10"].effective_start_us == (0, DAY - 1)


def test_S5_1_no_unit_is_invented(tmp_path, dataset):
    records, report = read_meds(dataset, **_utc(timezone=NEW_YORK))
    lab = _by_code(records)["LOINC/2160-0"]
    assert lab.value_num == 1.1  # the float32 MEDS stores, written shortest
    assert lab.unit is None and report.of_kind("unit")
    sample = Meds.sample(tmp_path / "s.medh5")
    augment_from_meds(sample, dataset, **_utc(timezone=NEW_YORK))
    assert "W914" in validate_file(sample).codes
    units = ["mg/dL"] * len(Meds.rows())
    declared = Meds.write(
        tmp_path / "units", extra={"unit": (__import__("pyarrow").string(), units)}
    )
    records, _ = read_meds(
        declared, **_utc(timezone=NEW_YORK, columns={"unit": "unit"})
    )
    found = _by_code(records)
    assert found["LOINC/2160-0"].unit == "mg/dL"
    assert found["GENDER//F"].unit is None  # a unit belongs to a value


def test_S5_1_both_values_are_refused(tmp_path):
    rows = Meds.rows()
    rows[2]["text_value"] = "high"
    root = Meds.write(tmp_path, rows)
    with pytest.raises(MEDH5ValidationError, match="both_values"):
        read_meds(root, **_utc(timezone=NEW_YORK))
    for keep, value in (("numeric", (1.1, None)), ("text", (None, "high"))):
        records, report = read_meds(root, **_utc(timezone=NEW_YORK, both_values=keep))
        lab = _by_code(records)["LOINC/2160-0"]
        assert (lab.value_num, lab.value_text) == value
        assert report.of_kind("value")[0].severity == "decision"


def test_S5_1_a_code_system_is_declared_never_guessed(tmp_path, dataset):
    records, _ = read_meds(
        dataset,
        **_utc(
            timezone=NEW_YORK,
            code_systems={"LOINC": "http://loinc.org"},
            kinds={"LOINC": "observation", "DX": "diagnosis"},
        ),
    )
    found = {(e.code_system, e.code): e.kind for e in records.events}
    assert found[("http://loinc.org", "2160-0")] == "observation"
    assert found[("meds:demo-ehr", "LAB//50912")] == "other"
    assert found[("meds:demo-ehr", "DX//I10")] == "diagnosis"
    unnamed = Meds.write(tmp_path / "unnamed", name=None)
    with pytest.raises(MEDH5ValidationError, match="give code_system"):
        read_meds(unnamed, **_utc(timezone=NEW_YORK))
    records, _ = read_meds(unnamed, **_utc(timezone=NEW_YORK, code_system="local"))
    assert {e.code_system for e in records.events} == {"local"}
    with pytest.raises(MEDH5ValidationError, match="event kind"):
        read_meds(dataset, **_utc(timezone=NEW_YORK, kinds={"LAB": "lab"}))


def test_S10_birth_and_text_are_opt_in(dataset):
    options = _utc(timezone=NEW_YORK, document_codes=["NOTE//discharge"])
    records, report = read_meds(dataset, **options)
    assert "MEDS_BIRTH" not in _by_code(records)
    note = _by_code(records)["NOTE//discharge"]
    assert note.kind == "document" and note.value_text is None
    assert not records.documents and not records.links
    assert report.of_kind("excluded") and report.of_kind("text")
    records, report = read_meds(
        dataset, **{**options, "exclude_codes": (), "texts": True}
    )
    assert "MEDS_BIRTH" in _by_code(records)
    (document,) = records.documents
    assert document.text == "Discharged home."
    (link,) = records.links
    assert (link.source_id, link.relation, link.target_id) == (
        _by_code(records)["NOTE//discharge"].event_id,
        "describes",
        document.document_id,
    )
    assert not report.of_kind("text")


def test_S10_ids_survive_reextraction(tmp_path, dataset):
    first, _ = read_meds(dataset, **_utc(timezone=NEW_YORK))
    rows = Meds.rows()[::-1]
    again = Meds.write(tmp_path / "again", rows, shards=3)
    second, _ = read_meds(again, **_utc(timezone=NEW_YORK))
    assert {e.event_id for e in first.events} == {e.event_id for e in second.events}
    assert all(e.event_id.startswith("meds.") for e in first.events)
    assert all(e.event_id == e.record_id for e in first.events)
    doubled = Meds.write(tmp_path / "doubled", [*Meds.rows(), Meds.rows()[2]])
    records, report = read_meds(doubled, **_utc(timezone=NEW_YORK))
    ids = sorted(e.event_id for e in records.events)
    assert len(set(ids)) == len(ids) and any(i.endswith(".2") for i in ids)
    assert report.of_kind("id")


def test_S10_a_declared_column_is_part_of_the_id(tmp_path):
    pa = pytest.importorskip("pyarrow")
    lab = Meds.rows()[2]
    rows, units = [lab, lab], ["mg/dL", "umol/L"]
    known = [datetime.datetime(2020, 3, 2), datetime.datetime(2020, 3, 9)]

    def read(name, order):
        root = Meds.write(
            tmp_path / name,
            [rows[i] for i in order],
            extra={
                "unit": (pa.string(), [units[i] for i in order]),
                "reported": (pa.timestamp("us"), [known[i] for i in order]),
            },
        )
        columns = {"unit": "unit", "available": "reported"}
        records, _ = read_meds(root, **_utc(timezone=NEW_YORK, columns=columns))
        return {e.event_id: (e.unit, e.available_us) for e in records.events}

    first, again = read("first", [0, 1]), read("again", [1, 0])
    assert first == again and len(first) == 2
    assert not any(i.endswith(".2") for i in first)


def test_S10_a_reimport_is_refused(tmp_path, dataset):
    sample = Meds.sample(tmp_path / "s.medh5")
    options = _utc(timezone=NEW_YORK)
    result, report = augment_from_meds(sample, dataset, **options)
    assert result["version_after"] == "1.1"
    assert report.outputs == [str(sample)]
    with open_sample(sample) as s:
        before = s.content_id
        assert len(s.clinical.events) == 5
    with pytest.raises(MEDH5Error, match="meds\\."):
        augment_from_meds(sample, dataset, **options)
    with open_sample(sample) as s:
        assert s.content_id == before


def test_S10_the_subject_is_the_callers_mapping(tmp_path, dataset):
    with pytest.raises(MEDH5ValidationError, match="no row of MEDS subject 7"):
        read_meds(dataset, **{**_utc(timezone=NEW_YORK), "subject_id": 7})
    _, report = read_meds(dataset, **_utc(timezone=NEW_YORK))
    (note,) = report.of_kind("subject")
    assert note.severity == "decision" and note.detail["meds_subject_id"] == 1001
    old = Meds.write(tmp_path / "old", rename={"subject_id": "patient_id"})
    with pytest.raises(MEDH5ValidationError, match="patient_id"):
        read_meds(old, **_utc(timezone=NEW_YORK))
    records, _ = read_meds(
        old, **_utc(timezone=NEW_YORK, columns={"subject_id": "patient_id"})
    )
    assert len(records.events) == 5
    shard, _ = read_meds(dataset / "data" / "0.parquet", **_utc(timezone=NEW_YORK))
    assert {e.code_system for e in shard.events} == {"meds:demo-ehr"}


def test_clinical_augment_from_meds(tmp_path, dataset, capsys):
    """`medh5 clinical augment --from-meds`: the import, its report kept as
    JSON, and exit 1 for the availability warning MEDS always earns."""
    import json

    from medh5.cli import EXIT_ERROR, EXIT_USAGE, main

    sample = Meds.sample(tmp_path / "s.medh5")
    out, report = tmp_path / "out.medh5", tmp_path / "report.json"
    argv = [
        "clinical", "augment", str(sample), "--from-meds", str(dataset),
        "--meds-subject", "1001", "--clock-id", "ehr", "--clock", "utc",
        "--timezone", NEW_YORK, "--code-system", "LOINC=http://loinc.org",
        "--kind", "LOINC=observation", "--day-precision", "DX",
        "--document-code", "NOTE//discharge", "--with-text",
        "--report", str(report), "-o", str(out),
    ]  # fmt: skip
    assert main(argv) == EXIT_ERROR
    printed = capsys.readouterr().out
    assert "MEDH5 1.0 -> 1.1; 5 events, 1 documents, 1 links" in printed
    assert "WARNING  availability" in printed
    kept = json.loads(report.read_text(encoding="utf-8"))
    assert kept["converter"] == "from-meds" and kept["outputs"] == [str(out)]
    with open_sample(out) as s:
        assert s.clinical.clock == Clock.utc("ehr")
        found = {e.code_system for e in s.clinical.events}
        assert "http://loinc.org" in found
    # Again, onto the sample's own clock: the same rows are refused.
    again = ["clinical", "augment", str(out), "--from-meds", str(dataset),
             "--meds-subject", "1001", "--timezone", NEW_YORK, "--json"]  # fmt: skip
    assert main(again) == EXIT_ERROR
    assert "already exists" in capsys.readouterr().err
    assert main([*argv[:3], "--from-meds", str(dataset)]) == EXIT_USAGE
    capsys.readouterr()
    plain = Meds.sample(tmp_path / "plain.medh5")
    assert main(["clinical", "augment", str(plain), "--from-meds", str(dataset),
                 "--meds-subject", "1001"]) == EXIT_ERROR  # fmt: skip
    assert "--clock-id and --clock" in capsys.readouterr().err


# -- a cohort, in one pass (W22-09) ------------------------------------------


COHORT = {
    "clock": Clock.utc("ehr"),
    "timezone": NEW_YORK,
    "code_systems": {"LOINC": "http://loinc.org"},
    "document_codes": ["NOTE//discharge"],
    "texts": True,
}
"""The options every cohort test imports with."""


def _cohort_rows(subjects, rows=3):
    """The kit's rows of subject 1001, again as each of *subjects*, with
    *rows* more creatinines of their own."""
    import datetime as dt

    found = []
    for subject in subjects:
        for row in Meds.rows():
            if row["subject_id"] == Meds.SUBJECT:
                found.append({**row, "subject_id": subject})
        for i in range(rows):
            found.append(
                {
                    "subject_id": subject,
                    "time": dt.datetime(2021, 1, 1) + dt.timedelta(hours=i),
                    "code": "LOINC/2160-0",
                    "numeric_value": float(subject % 7 + i),
                    "text_value": None,
                }
            )
    return found


def _content(path):
    with open_sample(path) as sample:
        return sample.content_id, sample.clinical.records().to_json()


def _notes(report):
    found = report.to_json()
    del found["outputs"]
    return found


def test_a_cohort_import_equals_each_subjects_import(tmp_path):
    """Each subject's sample is the one a per-sample import writes: same
    content, same records, same report --- over shards that split subjects
    and interleave their rows."""
    from medh5.io.meds import augment_cohort_from_meds

    subjects = (11, 12, 13)
    dataset = Meds.write(tmp_path / "meds", _cohort_rows(subjects), shards=2)
    samples = {s: Meds.sample(tmp_path / f"s{s}.medh5") for s in subjects}
    cohort = augment_cohort_from_meds(
        dataset, samples, out_dir=tmp_path / "cohort", **COHORT
    )
    assert [s.subject for s in cohort.imported] == list(subjects)
    assert not cohort.refused
    for found in cohort.subjects:
        alone = tmp_path / f"alone{found.subject}.medh5"
        result, report = augment_from_meds(
            samples[found.subject],
            dataset,
            subject_id=found.subject,
            out=alone,
            **COHORT,
        )
        written = tmp_path / "cohort" / f"s{found.subject}.medh5"
        assert found.augment is not None and found.report is not None
        assert found.augment["path"] == str(written)
        assert found.report.outputs == [str(written)]
        assert _content(written) == _content(alone)
        assert _notes(found.report) == _notes(report)
        assert {k: v for k, v in found.augment.items() if k != "path"} == {
            k: v for k, v in result.items() if k != "path"
        }
    # The sources are left as they were, and an import in place is one too.
    with open_sample(samples[11]) as sample:
        assert sample.clinical is None
    cohort = augment_cohort_from_meds(dataset, {11: samples[11]}, **COHORT)
    assert _content(samples[11]) == _content(tmp_path / "alone11.medh5")
    assert cohort.imported[0].augment["path"] == str(samples[11])


def test_a_cohort_import_reads_the_dataset_once(tmp_path, monkeypatch):
    """One dataset, read twice however many subjects: its subject column,
    then the mapped rows --- where per-sample imports scan once each."""
    import pyarrow.dataset as ds

    from medh5.io.meds import augment_cohort_from_meds

    opened, scans = [], []

    class Counting:
        def __init__(self, dataset):
            self._dataset = dataset

        def __getattr__(self, name):
            found = getattr(self._dataset, name)
            if name in ("to_table", "to_batches", "scanner", "get_fragments"):
                scans.append(name)
            return found

    real = ds.dataset

    def dataset(*args, **kwargs):
        opened.append(args)
        return Counting(real(*args, **kwargs))

    monkeypatch.setattr(ds, "dataset", dataset)
    subjects = tuple(range(20, 30))
    source = Meds.write(tmp_path / "meds", _cohort_rows(subjects), shards=3)
    samples = {s: Meds.sample(tmp_path / f"s{s}.medh5") for s in subjects}
    cohort = augment_cohort_from_meds(
        source, samples, out_dir=tmp_path / "out", **COHORT
    )
    assert len(cohort.imported) == len(subjects)
    assert len(opened) == 1 and scans == ["get_fragments", "get_fragments"]
    opened.clear()
    scans.clear()
    for s in subjects[:3]:
        augment_from_meds(
            samples[s], source, subject_id=s, out=tmp_path / f"a{s}.medh5", **COHORT
        )
    assert len(opened) == 3 and scans == ["to_table"] * 3


def test_a_refused_subject_leaves_the_others(tmp_path):
    """A subject with no rows, a sample with no clock to import onto, and
    rows already imported are each that subject's refusal; the others are
    written, and the refused leave no file."""
    from medh5.io.meds import augment_cohort_from_meds

    dataset = Meds.write(tmp_path / "meds", _cohort_rows((1, 2, 3, 4)))
    samples = {s: Meds.sample(tmp_path / f"s{s}.medh5") for s in (1, 2, 3, 4, 5)}
    done = tmp_path / "done.medh5"
    augment_from_meds(samples[3], dataset, subject_id=3, out=done, **COHORT)
    samples[3] = done
    on_own = {k: v for k, v in COHORT.items() if k != "clock"}
    # Subject 4's sample has a clock of its own; 1 and 2 have none.
    mixed = augment_cohort_from_meds(
        dataset, {1: samples[1], 4: done}, out_dir=tmp_path / "own", **on_own
    )
    assert [s.subject for s in mixed.refused] == [1]
    assert "no clinical clock" in mixed.refused[0].error
    assert "--clock-id" in mixed.refused[0].error
    (four,) = mixed.imported
    assert four.subject == 4 and four.augment["events"] == 8
    cohort = augment_cohort_from_meds(
        dataset, samples, out_dir=tmp_path / "out", **COHORT
    )
    assert [s.subject for s in cohort.imported] == [1, 2, 4]
    refused = {s.subject: s.error for s in cohort.refused}
    assert set(refused) == {3, 5}
    assert "already" in refused[3]
    assert refused[5] == f"{dataset} holds no row of MEDS subject 5"
    assert all(s.augment is None and s.report is None for s in cohort.refused)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "s1.medh5",
        "s2.medh5",
        "s4.medh5",
    ]
    assert not cohort.ok
    shown = cohort.to_json()
    assert shown["imported"] == 3 and shown["refused"] == 2
    assert shown["subjects"][4] == {
        "subject": 5,
        "sample": str(samples[5]),
        "augment": None,
        "report": None,
        "error": refused[5],
    }
    assert "REFUSED  5 ->" in cohort.format()


def test_a_cohort_import_counts_the_subjects_it_does_not_read(tmp_path):
    from medh5.io.meds import augment_cohort_from_meds

    dataset = Meds.write(tmp_path / "meds", _cohort_rows((1, 2, 3), rows=2))
    cohort = augment_cohort_from_meds(
        dataset, {2: Meds.sample(tmp_path / "s.medh5")}, **COHORT
    )
    (note,) = cohort.report.of_kind("subject")
    assert note.severity == "info"
    assert note.detail == {"subjects": 2, "rows": 16}
    assert "2 subject(s)" in str(cohort)


def test_a_cohort_mapping_is_checked_before_anything_is_read(tmp_path):
    from medh5.io.meds import augment_cohort_from_meds

    dataset = Meds.write(tmp_path / "meds")
    sample = Meds.sample(tmp_path / "s.medh5")
    with pytest.raises(MEDH5ValidationError, match="not a MEDS subject_id"):
        augment_cohort_from_meds(dataset, {"1001": sample}, **COHORT)
    with pytest.raises(MEDH5ValidationError, match="names no subject"):
        augment_cohort_from_meds(dataset, {}, **COHORT)
    with pytest.raises(MEDH5ValidationError, match="status"):
        augment_cohort_from_meds(dataset, {1001: sample}, status="x", **COHORT)
    with pytest.raises(MEDH5ValidationError, match="no MEDS shard"):
        (tmp_path / "empty").mkdir()
        augment_cohort_from_meds(tmp_path / "empty", {1001: sample}, **COHORT)
    with open_sample(sample) as found:
        assert found.clinical is None


def test_a_cohort_never_writes_two_subjects_to_one_sample(tmp_path):
    """A sample holds one subject's history: a mapping that would write two
    subjects to one file --- the same sample, a link to it, two samples of
    one name into `out_dir` (or of names told apart only by case), or one
    subject over another's source --- is refused before anything is
    written."""
    from medh5.io.meds import augment_cohort_from_meds

    dataset = Meds.write(tmp_path / "meds", _cohort_rows((1, 2)))
    sample = Meds.sample(tmp_path / "s.medh5")
    link = tmp_path / "link.medh5"
    os.link(sample, link)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = Meds.sample(tmp_path / "a" / "s.medh5")
    b = Meds.sample(tmp_path / "b" / "s.medh5")
    out = tmp_path / "out"
    for mapping, out_dir in (
        ({1: sample, 2: sample}, None),
        ({1: sample, 2: link}, None),
        ({1: sample, 2: tmp_path / "sub" / ".." / "s.medh5"}, None),
        ({1: a, 2: b}, out),
        ({1: a, 2: out / "s.medh5"}, out),
        # Not yet written, so maybe onto a file system blind to case.
        ({1: a, 2: tmp_path / "b" / "S.medh5"}, out),
    ):
        with pytest.raises(MEDH5ValidationError, match="both be written to"):
            augment_cohort_from_meds(dataset, mapping, out_dir=out_dir, **COHORT)
    assert not out.exists()
    # Nor into the dataset, whose files it reads, nor over a shard read alone.
    with pytest.raises(MEDH5ValidationError, match="inside .*which this import"):
        augment_cohort_from_meds(dataset, {1: a}, out_dir=dataset / "data", **COHORT)
    (shard,) = (dataset / "data").glob("*.parquet")
    (tmp_path / "src").mkdir()
    named = Meds.sample(tmp_path / "src" / shard.name)
    for mapping, out_dir in (({1: named}, shard.parent), ({1: shard}, None)):
        with pytest.raises(MEDH5ValidationError, match="inside or over"):
            augment_cohort_from_meds(shard, mapping, out_dir=out_dir, **COHORT)
    for path in (sample, a, b):
        with open_sample(path) as found:
            assert found.clinical is None
    # Into the samples' own directory, each subject is written to itself.
    other = Meds.sample(tmp_path / "t.medh5")
    cohort = augment_cohort_from_meds(
        dataset, {1: a, 2: other}, out_dir=tmp_path / "a", **COHORT
    )
    assert not cohort.refused


def test_the_subjects_a_cohort_does_not_read_are_counted_on_disk(tmp_path):
    """The subjects outside the mapping are counted, not held: read a row
    group of 1,000 at a time, 60,000 of them peak where 2,000 do, where a set
    of them is some 3 MB more."""
    import datetime as dt
    import tracemalloc

    from medh5.io.meds import augment_cohort_from_meds

    def dataset(name, n):
        rows = [
            {
                "subject_id": s,
                "time": dt.datetime(2021, 1, 1),
                "code": "LOINC/2160-0",
                "numeric_value": 1.0,
                "text_value": None,
            }
            for s in range(1, n + 1)
        ]
        return Meds.write(tmp_path / name, rows, row_group_size=1_000)

    def peak(root):
        mapping = {1: Meds.sample(tmp_path / f"{root.name}.medh5")}
        tracemalloc.start()
        try:
            cohort = augment_cohort_from_meds(root, mapping, **COHORT)
            top = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert not cohort.refused, cohort.format()
        (others,) = cohort.report.of_kind("subject")
        return others.detail["subjects"], top / 2**20  # MB

    (few, low), (many, high) = (
        peak(dataset("few", 2_000)),
        peak(dataset("many", 60_000)),
    )
    assert (few, many) == (1_999, 59_999)
    assert high - low < 1, (low, high)


@pytest.mark.skipif(sys.platform == "win32", reason="peak RSS is POSIX's")
def test_memory_is_bounded_by_one_subject(tmp_path):
    """A dataset in MEDS order, a subject per row group, is read a row group
    and a subject at a time: importing 24 subjects peaks where importing 12
    does, in resident memory and in Arrow's pool, where holding every
    subject's rows would be some 50 MB more."""
    import datetime as dt
    import json
    import subprocess

    rows, subjects = 1000, list(range(1, 25))
    data = [
        {
            "subject_id": s,
            "time": dt.datetime(2020, 1, 1) + dt.timedelta(minutes=i),
            "code": f"NOTE//{i}",
            "numeric_value": None,
            "text_value": f"{s}-{i} " * 600,
        }
        for s in subjects
        for i in range(rows)
    ]
    Meds.write(tmp_path / "meds", data, row_group_size=rows)
    del data
    for s in subjects:
        Meds.sample(tmp_path / f"s{s}.medh5")
    script = """
import json, resource, sys
from pathlib import Path
import pyarrow as pa
from medh5.clinical import Clock
from medh5.io.meds import augment_cohort_from_meds
root, n = Path(sys.argv[1]), int(sys.argv[2])
mapping = {s: root / f"s{s}.medh5" for s in range(1, n + 1)}
cohort = augment_cohort_from_meds(
    root / "meds", mapping, out_dir=root / f"out{n}", clock=Clock.utc("ehr"),
    timezone="UTC",
)
assert len(cohort.imported) == n, cohort.format()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss *= 1 if sys.platform == "darwin" else 1024
print(json.dumps([rss, pa.default_memory_pool().max_memory()]))
"""

    def peaks(n):
        done = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), str(n)],
            capture_output=True,
            text=True,
            check=True,
        )
        return [b / 2**20 for b in json.loads(done.stdout)]  # MB

    (rss_half, arrow_half), (rss_all, arrow_all) = peaks(12), peaks(24)
    assert rss_all - rss_half < 25, (rss_half, rss_all)
    assert arrow_all - arrow_half < 15, (arrow_half, arrow_all)


def test_clinical_augment_cohort_from_meds(tmp_path, capsys):
    """`medh5 clinical augment-cohort --from-meds`: a mapping CSV, the
    cohort's report kept as JSON, and exit 1 for the availability warning
    MEDS always earns."""
    import json

    from medh5.cli import EXIT_ERROR, EXIT_USAGE, main

    dataset = Meds.write(tmp_path / "meds", _cohort_rows((1, 2, 3)))
    for s in (1, 2):
        Meds.sample(tmp_path / f"s{s}.medh5")
    mapping = tmp_path / "map.csv"
    mapping.write_text(
        f"subject,sample\n1,s1.medh5\n2,{tmp_path / 's2.medh5'}\n9,s9.medh5\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    argv = [
        "clinical", "augment-cohort", "--from-meds", str(dataset),
        "--mapping", str(mapping), "--clock-id", "ehr", "--clock", "utc",
        "--timezone", NEW_YORK, "--code-system", "LOINC=http://loinc.org",
        "--kind", "LOINC=observation", "--day-precision", "DX",
        "--document-code", "NOTE//discharge", "--with-text",
        "--out-dir", str(tmp_path / "out"), "--report", str(report),
    ]  # fmt: skip
    assert main(argv) == EXIT_ERROR
    printed = capsys.readouterr().out
    assert "2 subject(s) imported, 1 refused" in printed
    assert "REFUSED  9 ->" in printed and "WARNING  availability" in printed
    kept = json.loads(report.read_text(encoding="utf-8"))
    assert kept["converter"] == "from-meds" and kept["imported"] == 2
    assert [s["subject"] for s in kept["subjects"]] == [1, 2, 9]
    with open_sample(tmp_path / "out" / "s2.medh5") as s:
        assert "http://loinc.org" in {e.code_system for e in s.clinical.events}
    # A subject_id that is not one, before anything is read.
    mapping.write_text("subject,sample\nP1,s1.medh5\n", encoding="utf-8")
    assert main([*argv, "--json"]) == EXIT_ERROR
    assert "'P1' is not a MEDS subject_id" in capsys.readouterr().err
    # A report inside the dataset it reads is refused.
    inside = [*argv[:-1], str(dataset / "report.json")]
    assert main(inside) == EXIT_ERROR
    assert "inside the MEDS dataset" in capsys.readouterr().err
    # So is a report that would replace a sample it reads or writes.
    mapping.write_text("subject,sample\n1,s1.medh5\n", encoding="utf-8")
    for named, what in (
        (tmp_path / "out" / "s1.medh5", "which this command writes"),
        (tmp_path / "s1.medh5", "which this command reads"),
    ):
        assert main([*argv[:-1], str(named)]) == EXIT_ERROR
        assert f"subject 1's sample, {what}" in capsys.readouterr().err
    in_place = [*argv[:-4], "--report", str(tmp_path / "s1.medh5")]
    assert main(in_place) == EXIT_ERROR
    assert "which this command amends" in capsys.readouterr().err
    # A sample not yet written, named in another case: one file on macOS.
    fresh = tmp_path / "fresh"
    cased = [*argv[:-4], "--out-dir", str(fresh), "--report", str(fresh / "S1.medh5")]
    assert main(cased) == EXIT_ERROR
    assert "which this command writes" in capsys.readouterr().err
    assert not fresh.exists()
    # Nor is a sample written inside the dataset, among the files it reads.
    shards = dataset / "data"
    within = [*argv[:-4], "--out-dir", str(shards), "--report", str(report)]
    assert main(within) == EXIT_ERROR
    assert "is inside the MEDS dataset this command reads" in capsys.readouterr().err
    assert not (shards / "s1.medh5").exists()
    # Nor is the report the directory the samples are written into.
    fresh_out = [*argv[:-4], "--out-dir", str(fresh), "--report", str(fresh)]
    assert main(fresh_out) == EXIT_ERROR
    assert "the directory this command writes the samples" in capsys.readouterr().err
    assert not fresh.exists()
    with open_sample(tmp_path / "s1.medh5") as s:
        assert s.clinical is None
    # One source, and a mapping.
    assert main(argv[:4]) == EXIT_USAGE
    capsys.readouterr()


def _cohort_as_run(cohort, out):
    """A cohort's report as printed, its output folder named ``OUT``."""
    return unrooted(json.dumps(cohort.to_json(), sort_keys=True), out)


def test_a_cohort_import_does_not_depend_on_the_worker_count(tmp_path, monkeypatch):
    """``clinical augment-cohort --workers N`` imports each subject in one of
    up to N processes, and writes and reports what one process does: the
    refused in their place among the imported, over shards splitting
    subjects.  A subject whose worker dies is that subject's refusal."""
    from medh5.io.meds import _cohort_from_meds

    dataset = Meds.write(tmp_path / "meds", _cohort_rows((1, 2, 4, 6)), shards=2)
    samples = {s: Meds.sample(tmp_path / f"s{s}.medh5") for s in (1, 2, 4, 5, 6)}
    seen = set()
    for workers in (1, 3):
        out = tmp_path / f"out{workers}"
        cohort = _cohort_from_meds(
            dataset, samples, out_dir=out, workers=workers, **COHORT
        )
        written = tuple(_content(p)[0] for p in sorted(out.iterdir()))
        seen.add((_cohort_as_run(cohort, out), written))
    assert len(seen) == 1
    ((_, written),) = seen
    assert len(written) == 4

    monkeypatch.setenv("MEDH5_TEST_WORKERS", "exit:2")
    cohort = _cohort_from_meds(
        dataset, samples, out_dir=tmp_path / "died", workers=2, **COHORT
    )
    assert [s.subject for s in cohort.imported] == [1, 4, 6]
    assert [s.subject for s in cohort.refused] == [2, 5]
    assert "the worker importing it stopped" in cohort.refused[0].error
