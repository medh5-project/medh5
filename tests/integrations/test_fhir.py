"""FHIR R4 import (1.1 §5, §10): precision becomes bounds, ``issued`` is
availability, versions supersede, and nothing FHIR leaves out is invented."""

from __future__ import annotations

import datetime
import json
import zoneinfo
from pathlib import Path

import pytest

from medh5.clinical import HOUR, Clock, day_bounds, utc_us
from medh5.errors import MEDH5Error, MEDH5ValidationError
from medh5.io.fhir import augment_from_fhir, read_fhir
from medh5.sample import open_sample
from medh5.validate import validate_file
from tests.kits import Fhir

UTC = datetime.timezone.utc


def _zones() -> bool:
    try:
        zoneinfo.ZoneInfo("America/New_York")
    except zoneinfo.ZoneInfoNotFoundError:  # pragma: no cover - Windows, no tzdata
        return False
    return True


NAMED_ZONES = pytest.mark.skipif(
    not _zones(), reason="no IANA time-zone data here (Windows needs tzdata)"
)
"""On the tests that name a zone: Linux and macOS have the data, and run them."""


def _read(resources=None, **options):
    options = {"subject": Fhir.SUBJECT, "clock": Clock.utc("ehr"), **options}
    return read_fhir(Fhir.bundle(resources), **options)


def _events(records):
    return {e.event_id: e for e in records.events}


def _at(*parts, offset=0):
    zone = datetime.timezone(datetime.timedelta(hours=offset))
    return utc_us(datetime.datetime(*parts, tzinfo=zone))


@NAMED_ZONES
def test_S5_2_fhir_precision_becomes_bounds():
    records, report = _read()
    found = _events(records)
    t = _at(2020, 3, 1, 8, 15, offset=-5)
    assert found["obs.creat.v1"].effective_start_us == (t, t)
    # A date with no known zone spans every instant it covers anywhere.
    assert found["obs.k"].effective_start_us == day_bounds(datetime.date(2020, 3, 1))
    a1c = found["obs.a1c"]
    assert a1c.temporal_type == "interval" and a1c.effective_end_us is None
    lo, hi = a1c.effective_start_us
    assert lo == _at(2020, 2, 1) - 14 * HOUR and hi == _at(2020, 3, 1) + 12 * HOUR - 1
    assert found["obs.smoke"].temporal_type == "unknown"
    assert any("effectiveTiming" in n.message for n in report.warnings)
    records, _ = _read(timezone="America/New_York")
    assert _events(records)["obs.k"].effective_start_us == day_bounds(
        datetime.date(2020, 3, 1), "America/New_York"
    )
    year = [{**Fhir.resources()[3], "effectiveDateTime": "2020"}]
    records, _ = _read(year)
    assert records.events[0].effective_start_us == (
        _at(2020, 1, 1) - 14 * HOUR,
        _at(2021, 1, 1) + 12 * HOUR - 1,
    )


@NAMED_ZONES
def test_S10_a_time_without_an_offset_needs_a_zone():
    naive = [{**Fhir.resources()[3], "effectiveDateTime": "2020-03-01T08:00:00"}]
    with pytest.raises(MEDH5ValidationError, match="give timezone"):
        _read(naive)
    records, _ = _read(naive, timezone="Europe/Paris")
    assert records.events[0].effective_start_us == (_at(2020, 3, 1, 7),) * 2
    clock = Clock.relative("ehr", "admission")
    with pytest.raises(MEDH5ValidationError, match="invents no origin"):
        _read(clock=clock)
    records, _ = _read(clock=clock, origin="2020-03-01T00:00:00-05:00")
    assert (
        _events(records)["obs.creat.v1"].effective_start_us
        == (8 * HOUR + 15 * 60 * 10**6,) * 2
    )


def test_S5_2_issued_is_availability():
    records, report = _read()
    found = _events(records)
    issued = _at(2020, 3, 1, 9, 15, offset=-5)
    assert found["obs.creat.v1"].available_us == (issued, issued)
    assert found["obs.k"].available_us is None
    (note,) = report.of_kind("availability")
    assert "never its inputs" in note.message


def test_S5_1_comparators_are_kept():
    records, _ = _read()
    found = _events(records)
    k = found["obs.k"]
    assert (k.value_comparator, k.value_num, k.unit) == ("lt", 5.0, "mmol/L")
    creat = found["obs.creat.v2"]
    assert (creat.value_comparator, creat.value_num, creat.unit) == (None, 1.2, "mg/dL")
    assert found["obs.bp.c0"].unit == "mm[Hg]"  # the UCUM code, not the display
    assert found["obs.bp.c1"].unit == "mmHg"  # no UCUM code: the unit string
    ad = [
        {
            **Fhir.resources()[3],
            "valueQuantity": {"value": 5, "comparator": "ad", "unit": "mmol/L"},
        },
        Fhir.resources()[1],
    ]
    records, report = _read(ad)
    assert "obs.k" not in _events(records)
    assert any("'ad'" in n.message for n in report.warnings)


def test_S5_1_data_absent_reason_is_missing_reason():
    a1c = _events(_read()[0])["obs.a1c"]
    assert a1c.missing_reason == "not-performed"
    assert a1c.value_num is None and a1c.value_text is None


def test_S5_2_versions_supersede():
    records, report = _read()
    found = _events(records)
    assert {found[e].record_id for e in ("obs.creat.v1", "obs.creat.v2")} == {
        "obs.creat"
    }
    assert found["obs.creat.v1"].status == "preliminary"
    assert found["obs.creat.v2"].status == "amended"
    assert any(n.severity == "decision" for n in report.of_kind("status"))
    supersedes = [
        (link.source_id, link.target_id)
        for link in records.links
        if link.relation == "supersedes"
    ]
    assert supersedes == [("obs.creat.v2", "obs.creat.v1")]
    # Unorderable versions are refused, not ordered by guess.
    first, second = Fhir.resources()[1:3]
    first = {**first, "meta": {"versionId": "a"}}
    second = {**second, "meta": {"versionId": "b"}}
    with pytest.raises(MEDH5ValidationError, match="have no order"):
        _read([first, second])
    stamped = [
        {**first, "meta": {"versionId": "a", "lastUpdated": "2020-03-02T00:00:00Z"}},
        {**second, "meta": {"versionId": "b", "lastUpdated": "2020-03-01T00:00:00Z"}},
    ]
    records, _ = _read(stamped)
    assert [(link.source_id, link.target_id) for link in records.links] == [
        ("obs.creat.va", "obs.creat.vb")
    ]
    # "1" and "01" are two versionIds with one number: not an order either.
    same = [
        {**first, "meta": {"versionId": "1"}},
        {**second, "meta": {"versionId": "01"}},
    ]
    with pytest.raises(MEDH5ValidationError, match="have no order"):
        _read(same)


def test_S10_components_are_events_of_their_own():
    records, report = _read()
    found = _events(records)
    panel, systolic = found["obs.bp"], found["obs.bp.c0"]
    assert (systolic.code, systolic.value_num) == ("8480-6", 120.0)
    assert systolic.effective_start_us == panel.effective_start_us
    assert systolic.available_us == panel.available_us
    assert systolic.record_id == "obs.bp.c0"
    (dropped,) = report.of_kind("dropped")
    assert dropped.detail == {"interpretation": 1, "referenceRange": 1}


def test_S10_a_parent_value_left_out_keeps_its_components():
    panel = next(r for r in Fhir.resources() if r.get("id") == "bp")
    ranged = {**panel, "valueRange": {"low": {"value": 1}, "high": {"value": 2}}}
    others = [r for r in Fhir.resources() if r.get("id") != "bp"]
    records, report = _read([*others, ranged])
    found = _events(records)
    assert "obs.bp" not in found
    assert found["obs.bp.c0"].value_num == 120.0
    assert found["obs.bp.c1"].value_num == 80.0
    assert any("valueRange" in n.message for n in report.of_kind("value"))


def test_S10_registered_is_skipped():
    records, report = _read()
    found = _events(records)
    assert "obs.ordered" not in found
    assert any("registered" in n.message for n in report.of_kind("status"))
    # No code is guessed from text, and other subjects stay out.
    assert "obs.free" not in found and "obs.other" not in found
    assert any("no coding" in n.message for n in report.warnings)
    assert report.of_kind("subject")[0].severity == "decision"
    assert found["obs.smoke"].value_text == "8517006"
    with pytest.raises(MEDH5ValidationError, match="no Observation"):
        _read(subject="Patient/p9")
    # A subject that is not a Reference is refused, not read as no subject.
    malformed = {**Fhir.resources()[1], "subject": "Patient/p1"}
    with pytest.raises(MEDH5ValidationError, match="not a FHIR Reference"):
        _read([malformed])


def test_S10_images_link_only_through_the_map(tmp_path):
    records, report = _read()
    found = _events(records)
    assert found["dr.ct-report"].kind == "document"
    assert not records.documents and not records.links[1:]
    notes = {n.message for n in report.of_kind("link")} | {
        n.message for n in report.of_kind("text")
    }
    assert any("image_studies" in m for m in notes)
    assert any("texts=True" in m for m in notes)
    records, _ = _read(texts=True, image_studies={Fhir.STUDY: "CT_tp0"})
    (document,) = records.documents
    assert document.text == Fhir.TEXT
    described = {
        (link.target_type, link.target_id)
        for link in records.links
        if link.relation == "describes"
    }
    assert described == {("document", "dr.ct-report"), ("image", "CT_tp0")}
    sample = Fhir.sample(tmp_path / "s.medh5")
    bundle = Fhir.write(tmp_path / "bundle.json")
    result, report = augment_from_fhir(
        sample,
        bundle,
        subject=Fhir.SUBJECT,
        clock=Clock.utc("ehr"),
        texts=True,
        image_studies={Fhir.STUDY: "CT_tp0"},
    )
    assert result["version_after"] == "1.1" and report.outputs == [str(sample)]
    assert validate_file(sample).ok
    with open_sample(sample) as s:
        assert s.clinical.text("dr.ct-report") == Fhir.TEXT
    with pytest.raises(MEDH5Error):
        augment_from_fhir(sample, bundle, subject=Fhir.SUBJECT, clock=Clock.utc("ehr"))


def test_S10_a_report_holds_no_guess(tmp_path: Path):
    _, report = read_fhir(
        Fhir.write(tmp_path / "b.json"), subject=Fhir.SUBJECT, clock=Clock.utc("c")
    )
    assert report.converter == "from-fhir" and not report.guesses


def test_clinical_augment_from_fhir(tmp_path, capsys):
    """`medh5 clinical augment --from-fhir`, with `--json`; and the MEDS
    options it does not take."""
    import json

    from medh5.cli import EXIT_ERROR, EXIT_OK, main

    sample = Fhir.sample(tmp_path / "s.medh5")
    bundle = Fhir.write(tmp_path / "bundle.json")
    argv = [
        "clinical", "augment", str(sample), "--from-fhir", str(bundle),
        "--fhir-subject", Fhir.SUBJECT, "--clock-id", "ehr", "--clock", "utc",
    ]  # fmt: skip
    assert main([*argv, "--both-values", "text"]) == EXIT_ERROR
    assert "--both-values: MEDS options" in capsys.readouterr().err
    # Warnings (a Timing, a code in text only) exit 1, the file written.
    assert main([*argv, "--with-text", "--json"]) == EXIT_ERROR
    printed = json.loads(capsys.readouterr().out)
    assert printed["augment"]["version_after"] == "1.1"
    assert printed["augment"]["documents"] == 1
    assert printed["report"]["converter"] == "from-fhir"
    clean = [r for i, r in enumerate(Fhir.resources()) if i in (1, 2, 3)]
    other = Fhir.sample(tmp_path / "t.medh5")
    relative = [
        "clinical", "augment", str(other),
        "--from-fhir", str(Fhir.write(tmp_path / "clean.json", clean)),
        "--fhir-subject", Fhir.SUBJECT, "--clock-id", "ehr", "--clock", "relative",
        "--origin", "2020-03-01T00:00:00-05:00",
    ]  # fmt: skip
    assert main(relative) == EXIT_OK
    assert "0 warning(s)" in capsys.readouterr().out
    with open_sample(other) as s:
        assert s.clinical.clock.reference == "relative"


# -- a bulk export into a cohort (W22-09) -------------------------------------


BULK = {"clock": Clock.utc("ehr"), "texts": True, "timezone": datetime.timezone.utc}
"""The options every bulk test imports with: a zone that needs no IANA data,
so they run on Windows without tzdata too."""


def _export(tmp_path, *subjects):
    """A bulk export of the kit's resources (``Patient/p1``'s, one of
    ``Patient/p2``, a ``Patient``) and *subjects*' copies of ``p1``'s, the
    subjects' resources interleaved line by line."""
    sources = [Fhir.resources(), *(Fhir.of(s) for s in subjects)]
    resources = []
    for i in range(max(map(len, sources))):
        resources.extend(source[i] for source in sources if i < len(source))
    return Fhir.write_bulk(tmp_path / "export", resources), resources


def _of(resources, subject):
    """*subject*'s Observations and DiagnosticReports, in the order a bulk
    export holds them: by file name, then by line."""
    mine = [
        r
        for r in resources
        if (r.get("subject") or {}).get("reference") == subject
        and r["resourceType"] in ("Observation", "DiagnosticReport")
    ]
    return sorted(mine, key=lambda r: r["resourceType"])


def _content(path):
    with open_sample(path) as sample:
        return sample.content_id, sample.clinical.records().to_json()


def test_a_bulk_export_import_equals_each_subjects_bundle_import(tmp_path):
    from medh5.io.fhir import augment_cohort_from_fhir

    export, resources = _export(tmp_path, "Patient/p3")
    mapping = {
        "Patient/p1": Fhir.sample(tmp_path / "one.medh5"),
        "Patient/p3": Fhir.sample(tmp_path / "three.medh5"),
    }
    cohort = augment_cohort_from_fhir(export, mapping, out_dir=tmp_path / "out", **BULK)
    assert [s.subject for s in cohort.imported] == list(mapping)
    for found in cohort.subjects:
        bundle = Fhir.write(
            tmp_path / f"{found.subject[-2:]}.json", _of(resources, found.subject)
        )
        alone = tmp_path / f"alone-{found.subject[-2:]}.medh5"
        result, report = augment_from_fhir(
            mapping[found.subject], bundle, subject=found.subject, out=alone, **BULK
        )
        written = tmp_path / "out" / mapping[found.subject].name
        assert _content(written) == _content(alone)
        assert found.report.source == str(export)
        assert found.report.outputs == [str(written)]
        assert found.report.notes == report.notes
        assert found.augment["events"] == result["events"]
    # Two versions of one resource are still versions of one record.
    with open_sample(tmp_path / "out" / "three.medh5") as sample:
        events = sample.clinical.records().events
        assert sum(e.record_id == "obs.creat" for e in events) == 2


def test_a_resource_for_an_unmapped_subject_is_counted(tmp_path):
    from medh5.io.fhir import augment_cohort_from_fhir

    export, _ = _export(tmp_path, "Patient/p3", "Patient/p4")
    sample = Fhir.sample(tmp_path / "s.medh5")
    cohort = augment_cohort_from_fhir(export, {"Patient/p3": sample}, **BULK)
    assert [s.subject for s in cohort.imported] == ["Patient/p3"]
    (subjects,) = cohort.report.of_kind("subject")
    # p1's nine and p4's nine, and p2's one: three subjects not read.
    assert subjects.severity == "info"
    assert subjects.detail == {"subjects": 3, "resources": 19}
    (patient,) = cohort.report.of_kind("resource")
    assert patient.detail == {"resourceType": "Patient", "resources": 1}
    assert "19 Observation(s) and DiagnosticReport(s)" in str(cohort)


def test_a_bulk_index_is_kept_out_of_memory(tmp_path):
    """Where a bulk export's lines are, and which subjects it holds that the
    mapping does not name, is kept on disk: indexing 24 subjects' 48,000
    lines peaks where indexing 12 subjects' does, and 30,000 subjects outside
    the mapping add nothing, where an index in memory holds some 2.5 MB more
    in each case.  The samples have no clock and none is given, so every
    subject is refused before its lines are read, and the peak is the
    index's."""
    import json
    import tracemalloc

    from medh5.io.fhir import augment_cohort_from_fhir

    def obs(subject, i):
        return json.dumps({
            "resourceType": "Observation", "id": f"o{i}", "status": "final",
            "subject": {"reference": subject},
            "code": {"coding": [{"system": Fhir.LOINC, "code": "2160-0"}]},
            "effectiveDateTime": "2020-03-01T08:15:00Z",
        }) + "\n"  # fmt: skip

    subjects = [f"Patient/s{s}" for s in range(24)]
    narrow, wide = tmp_path / "narrow", tmp_path / "wide"
    for root in (narrow, wide):
        root.mkdir()
        with (root / "Observation.ndjson").open("w", encoding="utf-8") as out:
            out.writelines(obs(s, i) for i in range(2000) for s in subjects)
            if root is wide:
                out.writelines(obs(f"Patient/u{u}", 0) for u in range(30_000))
    samples = {s: Fhir.sample(tmp_path / f"{s[8:]}.medh5") for s in subjects}

    def peak(export, n):
        mapping = {s: samples[s] for s in subjects[:n]}
        tracemalloc.start()
        try:
            cohort = augment_cohort_from_fhir(export, mapping)
            top = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert len(cohort.refused) == n
        assert "no clinical clock" in cohort.refused[0].error
        return top / 2**20  # MB

    half, whole = peak(narrow, 12), peak(narrow, 24)
    assert whole - half < 0.5, (half, whole)
    assert peak(wide, 24) - whole < 0.5, whole


def test_a_bulk_import_reads_its_options_once_for_every_subject(tmp_path):
    """`prefer_systems` given as a generator serves every subject, not only
    the first, as each subject's own import would be given it."""
    from medh5.io.fhir import augment_cohort_from_fhir

    def obs(subject):
        return {
            "resourceType": "Observation", "id": "creat", "status": "final",
            "subject": {"reference": subject},
            "code": {"coding": [
                {"system": "http://example.org/local", "code": "CR"},
                {"system": Fhir.LOINC, "code": "2160-0"},
            ]},
            "effectiveDateTime": "2020-03-01T08:15:00Z",
        }  # fmt: skip

    subjects = ["Patient/a", "Patient/b"]
    export = Fhir.write_bulk(tmp_path / "export", [obs(s) for s in subjects])
    mapping = {s: Fhir.sample(tmp_path / f"{s[-1]}.medh5") for s in subjects}
    cohort = augment_cohort_from_fhir(
        export, mapping, prefer_systems=(s for s in [Fhir.LOINC]), **BULK
    )
    assert [s.subject for s in cohort.imported] == subjects
    for sample in mapping.values():
        with open_sample(sample) as found:
            (event,) = found.clinical.records().events
            assert (event.code_system, event.code) == (Fhir.LOINC, "2160-0")


def test_a_bulk_import_writes_nothing_inside_its_export(tmp_path):
    """A sample named like a file of the export, written into the export,
    would replace what the import reads: refused before anything is."""
    from medh5.io.fhir import augment_cohort_from_fhir

    export, _ = _export(tmp_path)
    sample = Fhir.sample(tmp_path / "Observation.ndjson")
    before = (export / "Observation.ndjson").read_bytes()
    for out_dir in (export, export / "sub"):
        with pytest.raises(MEDH5ValidationError, match="inside .*which this import"):
            augment_cohort_from_fhir(
                export, {"Patient/p1": sample}, out_dir=out_dir, **BULK
            )
    assert (export / "Observation.ndjson").read_bytes() == before
    assert not (export / "sub").exists()


def test_a_verbose_cohort_shows_each_subjects_notes(tmp_path):
    from medh5.io.fhir import augment_cohort_from_fhir

    export, _ = _export(tmp_path)
    sample = Fhir.sample(tmp_path / "one.medh5")
    cohort = augment_cohort_from_fhir(export, {"Patient/p1": sample}, **BULK)
    (subject,) = cohort.imported
    verbose = subject.report.format(verbose=True)
    assert verbose != subject.report.format()
    shown = cohort.format(verbose=True)
    assert all("    " + line in shown for line in verbose.splitlines())


def test_a_bulk_subject_that_refuses_leaves_the_others(tmp_path):
    from medh5.io.fhir import augment_cohort_from_fhir

    export, _ = _export(tmp_path)
    mapping = {
        "Patient/p9": Fhir.sample(tmp_path / "nine.medh5"),
        "Patient/p1": Fhir.sample(tmp_path / "one.medh5"),
    }
    cohort = augment_cohort_from_fhir(export, mapping, **BULK)
    (refused,) = cohort.refused
    assert refused.subject == "Patient/p9"
    assert "no Observation or DiagnosticReport of Patient/p9" in refused.error
    assert [s.subject for s in cohort.imported] == ["Patient/p1"]
    with open_sample(mapping["Patient/p9"]) as nine:
        assert nine.clinical is None
    # The sample's own clock, when none is given: p1's sample has one now.
    again = augment_cohort_from_fhir(
        export,
        {"Patient/p2": mapping["Patient/p1"]},
        **{k: v for k, v in BULK.items() if k != "clock"},
    )
    assert again.imported[0].augment["events"] == 1


def test_a_bulk_export_is_read_as_ndjson(tmp_path):
    from medh5.io.fhir import augment_cohort_from_fhir

    sample = Fhir.sample(tmp_path / "s.medh5")
    with pytest.raises(MEDH5ValidationError, match="not a FHIR bulk export"):
        augment_cohort_from_fhir(tmp_path / "none", {"Patient/p1": sample}, **BULK)
    (tmp_path / "empty").mkdir()
    with pytest.raises(MEDH5ValidationError, match=r"no FHIR bulk file"):
        augment_cohort_from_fhir(tmp_path / "empty", {"Patient/p1": sample}, **BULK)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "Observation.ndjson").write_text(
        '{"resourceType": "Observation"}\n\n{', encoding="utf-8"
    )
    with pytest.raises(MEDH5ValidationError, match="line 3 of .* is not JSON"):
        augment_cohort_from_fhir(broken, {"Patient/p1": sample}, **BULK)
    (broken / "Observation.ndjson").write_text(
        '{"resourceType": "Observation"}\n'
        '{"resourceType": "Observation", "subject": "Patient/p1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        MEDH5ValidationError,
        match="Observation on line 2 of .* has a subject that is not a FHIR Reference",
    ):
        augment_cohort_from_fhir(broken, {"Patient/p1": sample}, **BULK)
    (broken / "Observation.ndjson").write_text("[1]\n", encoding="utf-8")
    with pytest.raises(
        MEDH5ValidationError, match="line 1 of .* is not a FHIR resource"
    ):
        augment_cohort_from_fhir(broken, {"Patient/p1": sample}, **BULK)
    with open_sample(sample) as found:
        assert found.clinical is None


def test_clinical_augment_cohort_from_fhir_bulk(tmp_path, capsys):
    """`medh5 clinical augment-cohort --from-fhir-bulk`, a mapping CSV whose
    paths are its own, and the MEDS options it does not take."""
    import json

    from medh5.cli import EXIT_ERROR, EXIT_OK, main

    export, _ = _export(tmp_path, "Patient/p3")
    Fhir.sample(tmp_path / "three.medh5")
    mapping = tmp_path / "map.csv"
    mapping.write_text("subject,sample\nPatient/p3,three.medh5\n", encoding="utf-8")
    argv = [
        "clinical", "augment-cohort", "--from-fhir-bulk", str(export),
        "--mapping", str(mapping), "--clock-id", "ehr", "--clock", "utc",
        "--out-dir", str(tmp_path / "out"),
    ]  # fmt: skip
    assert main([*argv, "--kind", "LOINC=lab"]) == EXIT_ERROR
    assert "--kind: MEDS options" in capsys.readouterr().err
    # The kit's warnings (a Timing, a code in text only) exit 1.
    assert main([*argv, "--json"]) == EXIT_ERROR
    printed = json.loads(capsys.readouterr().out)
    assert printed["converter"] == "from-fhir" and printed["imported"] == 1
    (subject,) = printed["subjects"]
    assert subject["subject"] == "Patient/p3"
    assert subject["augment"]["path"] == str(tmp_path / "out" / "three.medh5")
    # Again into the same directory: refused, as an import over a file is.
    assert main(argv) == EXIT_ERROR
    assert "REFUSED  Patient/p3" in capsys.readouterr().out
    # A sample named like a file of the export is not written over it.
    Fhir.sample(tmp_path / "Observation.ndjson")
    mapping.write_text(
        "subject,sample\nPatient/p3,Observation.ndjson\n", encoding="utf-8"
    )
    before = (export / "Observation.ndjson").read_bytes()
    assert main([*argv[:-1], str(export)]) == EXIT_ERROR
    assert "inside the FHIR bulk export this command reads" in capsys.readouterr().err
    assert (export / "Observation.ndjson").read_bytes() == before
    mapping.write_text("patient,sample\nPatient/p3,three.medh5\n", encoding="utf-8")
    assert main(argv) == EXIT_ERROR
    assert "header subject,sample" in capsys.readouterr().err
    mapping.write_text("subject,sample\nPatient/p3\n", encoding="utf-8")
    assert main(argv) == EXIT_ERROR
    assert "line 2 of" in capsys.readouterr().err
    mapping.write_text(
        "subject,sample\nPatient/p3,a\n\nPatient/p3,b\n", encoding="utf-8"
    )
    assert main(argv) == EXIT_ERROR
    assert "line 4 of" in capsys.readouterr().err
    # No warning, no refusal: exit 0, the text form.
    clean = Fhir.write_bulk(tmp_path / "clean", Fhir.resources()[1:4])
    Fhir.sample(tmp_path / "one.medh5")
    mapping.write_text("subject,sample\nPatient/p1,one.medh5\n", encoding="utf-8")
    assert main([*argv[:2], "--from-fhir-bulk", str(clean), *argv[4:]]) == EXIT_OK
    assert "1 subject(s) imported, 0 refused" in capsys.readouterr().out


def test_a_bulk_import_does_not_depend_on_the_worker_count(tmp_path, monkeypatch):
    """``clinical augment-cohort --workers N`` imports each subject in one of
    up to N processes, and writes and reports what one process does.  A
    subject whose worker dies is that subject's refusal."""
    from medh5.io.fhir import _cohort_from_fhir

    export, _ = _export(tmp_path, "Patient/p3", "Patient/p4")
    mapping = {
        f"Patient/p{i}": Fhir.sample(tmp_path / f"s{i}.medh5") for i in (1, 3, 4, 5)
    }
    seen = set()
    for workers in (1, 3):
        out = tmp_path / f"out{workers}"
        cohort = _cohort_from_fhir(
            export, mapping, out_dir=out, workers=workers, **BULK
        )
        shown = json.dumps(cohort.to_json(), sort_keys=True).replace(str(out), "OUT")
        written = tuple(_content(p)[0] for p in sorted(out.iterdir()))
        seen.add((shown, written))
    assert len(seen) == 1
    ((_, written),) = seen
    assert len(written) == 3

    monkeypatch.setenv("MEDH5_TEST_WORKERS", "exit:p3")
    cohort = _cohort_from_fhir(
        export, mapping, out_dir=tmp_path / "died", workers=2, **BULK
    )
    assert [s.subject for s in cohort.imported] == ["Patient/p1", "Patient/p4"]
    assert [s.subject for s in cohort.refused] == ["Patient/p3", "Patient/p5"]
    assert "the worker importing it stopped" in cohort.refused[0].error
