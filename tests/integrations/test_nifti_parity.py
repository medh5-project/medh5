"""The NIfTI parity corpus: what 2.1.0's converter did, and that it still does.

``tests/data/nifti-parity`` holds small NIfTI inputs, the cases run on them
(``cases.json``) and what medh5 2.1.0 made of each (``records/``), recorded
by ``record.py`` under the released 2.1.0.  2.2.0 moves the converter into
the engine (decision E3), and the contract is parity with these records:
the same objects with the same digests, the same ``/meta`` but for the
version its software agent records, the same report but for the paths it
names, the same refusals, and exports with the same header fields and voxel
bytes.

The records are compared through the harness that made them, so a case runs
here exactly as it ran under 2.1.0.  A difference the contract allows that
is not one of the two masks is listed in ``ALLOWED`` by case, with the
change that made it.
"""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("nibabel")
pytest.importorskip("h5py")

CORPUS = Path(__file__).resolve().parent.parent / "data" / "nifti-parity"
TABLE = json.loads((CORPUS / "cases.json").read_text(encoding="utf-8"))
CASES = {case["id"]: case for case in TABLE["cases"]}

ALLOWED: dict[str, tuple[tuple[str, ...], str]] = {}
"""Case id → the record's keys that may differ from 2.1.0's beyond the two
masks, and the change that made them differ.  Empty: the engine's converter
gives 2.1.0's record for every case."""

RENAMED = {"HeaderDataError": "ValueError"}
"""The one exception type E3 lets change: nibabel's ``HeaderDataError``,
which the engine raises as the ``ValueError`` it means (a type NIfTI cannot
hold), with nibabel's text."""


def _renamed(expected: dict[str, Any], got: dict[str, Any]) -> None:
    error, fresh = expected.get("error"), got.get("error")
    if (
        error is not None
        and fresh is not None
        and RENAMED.get(error["type"]) == fresh["type"]
    ):
        error["type"] = fresh["type"]


def _harness() -> ModuleType:
    """``record.py``, the harness the records were made with."""
    spec = importlib.util.spec_from_file_location(
        "nifti_parity_record", CORPUS / "record.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(case_id: str) -> dict[str, Any]:
    path = CORPUS / "records" / f"{case_id}.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _slashes(value: Any) -> Any:
    """Windows paths in *value* spelled as the records spell them: a path
    in a message is its ``repr`` there, with every separator doubled."""
    if isinstance(value, dict):
        return {k: _slashes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_slashes(v) for v in value]
    if isinstance(value, str):
        return re.sub(
            r"\{(corpus|out)\}[^'\"\s,]*",
            lambda m: m.group(0).replace("\\\\", "/").replace("\\", "/"),
            value,
        )
    return value


def _respelled(value: Any, out: Path) -> Any:
    """*value* with the corpus and the case's directory named as tokens however
    a message spells them.  ``record.py`` replaces each as ``str()`` spells it;
    on Windows a message may spell one in a ``repr`` (each separator doubled)
    or with ``/``, as a reader of paths does.  Elsewhere this changes nothing.
    """
    if isinstance(value, dict):
        return {k: _respelled(v, out) for k, v in value.items()}
    if isinstance(value, list):
        return [_respelled(v, out) for v in value]
    if isinstance(value, str):
        for base, token in ((out, "{out}"), (CORPUS, "{corpus}")):
            for spelling in (str(base).replace("\\", "\\\\"), base.as_posix()):
                value = value.replace(spelling, token)
        return value
    return value


def masked(record: dict[str, Any]) -> dict[str, Any]:
    """*record* without what parity does not compare (E3).

    The two masks: the version ``/meta``'s software agent records, and the
    paths the report names.  The rest is not the converter's output: the
    versions the record was made with, the root's ``generator`` (the writer's
    name and version) and ``content_id``, which covers ``/meta`` and so the
    version (1.0 §13.2); the objects' digests stand for the content.
    """
    out = {
        k: v
        for k, v in record.items()
        if k not in ("id", "generator", "nibabel", "numpy")
    }
    report = out.get("report")
    if report is not None:
        out["report"] = {
            k: v for k, v in report.items() if k not in ("source", "outputs")
        }
    sample = out.get("sample")
    if sample is not None:
        root = {
            k: v
            for k, v in sample["root"].items()
            if k not in ("generator", "content_id")
        }
        meta = json.loads(json.dumps(sample["meta"]))
        for agent in meta.get("provenance", {}).get("agents", []):
            if agent.get("type") == "software" and agent.get("name") == "medh5":
                agent["version"] = "*"
        out["sample"] = {**sample, "root": root, "meta": meta}
    return _slashes(out)


class TestTheCorpus:
    """The records are complete, and they are 2.1.0's."""

    def test_every_case_has_a_record_made_by_2_1_0(self) -> None:
        recorded = {p.stem for p in (CORPUS / "records").glob("*.json")}
        assert recorded == set(CASES), "one record per case, and no other"
        for case_id in CASES:
            record = _record(case_id)
            assert record["id"] == case_id
            assert record["generator"] == "medh5 2.1.0", case_id
            assert (record["nibabel"], record["numpy"]) == ("5.4.2", "2.5.4"), case_id
            assert record["outcome"] in ("ok", "refused"), case_id
            sample = record.get("sample")
            if sample is not None:
                assert sample["root"]["generator"] == "medh5 2.1.0", case_id

    def test_every_input_is_used(self) -> None:
        """An input no case reads is weight with no test behind it."""
        named: set[str] = set()
        for case in TABLE["cases"]:
            for key in ("images", "masks"):
                named.update(path for _, path in case.get(key, ()))
            for key in ("path", "sample"):
                if key in case:
                    named.add(case[key])
            named.update(p for p in case.get("beside", {}).values() if p)
            if "label_set" in case.get("options", {}):
                named.add(case["options"]["label_set"])
        # A pair's image, and a sidecar, are read through the file they
        # belong to.
        companions = {".img", ".json", ".bval"}
        inputs = {
            p.relative_to(CORPUS).as_posix()
            for p in (CORPUS / "inputs").rglob("*")
            if p.is_file()
        }
        unused = {
            p
            for p in inputs - named
            if not (Path(p).suffix in companions and Path(p).parent.name == "inputs")
        }
        assert not unused
        missing = {p for p in named - inputs if not p.endswith("missing.nii")}
        assert not missing

    def test_the_cases_cover_every_call(self) -> None:
        calls = {case["call"] for case in TABLE["cases"]}
        assert calls == {"from_nifti", "import_seg_nifti", "to_nifti", "read_nifti"}
        outcomes = {_record(c)["outcome"] for c in CASES}
        assert outcomes == {"ok", "refused"}


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    return _harness()


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_E3_the_converter_matches_2_1_0(case_id: str, harness: ModuleType) -> None:
    """Every case gives what 2.1.0 gave, under E3's two masks."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch).resolve() / "out"
        out.mkdir()
        fresh = _respelled(harness.outcome(CASES[case_id], out), out)
    expected, got = masked(_record(case_id)), masked(fresh)
    _renamed(expected, got)
    for key in ALLOWED.get(case_id, ((), ""))[0]:  # pragma: no cover - none yet
        expected.pop(key, None)
        got.pop(key, None)
    assert got == expected
