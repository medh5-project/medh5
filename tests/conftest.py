"""Fixtures for the whole suite; the builders behind them are in ``helpers``."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from medh5.labels import LabelClass, LabelSet
from tests.helpers import SHAPE, block, write_sample, write_series


@pytest.fixture
def label_set() -> LabelSet:
    return LabelSet(
        "test-v1",
        version="1.0.0",
        classes=[
            LabelClass(1, "liver", "Liver", category="organ"),
            LabelClass(2, "spleen", "Spleen", category="organ"),
            LabelClass(3, "lesion", "Lesion", parents=[1], category="lesion"),
            LabelClass(4, "vessel", "Vessel", category="vessel"),
        ],
    )


@pytest.fixture
def masks() -> dict[int, Any]:
    """Three classes where 1 and 3 overlap and 2 does not touch either."""
    return {
        1: block(SHAPE, (2, 2, 2), 8),
        2: block(SHAPE, (2, 14, 2), 6),
        3: block(SHAPE, (4, 4, 4), 3),
    }


@pytest.fixture
def sample_path(tmp_path: Path, label_set: LabelSet, masks: dict[int, Any]) -> Path:
    return write_sample(tmp_path / "case.medh5", label_set=label_set, masks=masks)


@pytest.fixture
def longitudinal_path(
    tmp_path: Path, label_set: LabelSet, masks: dict[int, Any]
) -> Path:
    return write_sample(
        tmp_path / "long.medh5",
        label_set=label_set,
        masks=masks,
        timepoints=("tp0", "tp1"),
        index=True,
    )


@pytest.fixture
def indexed_cohort(tmp_path: Path, label_set, masks) -> list[Path]:
    return [
        write_sample(
            tmp_path / f"case{i}.medh5",
            label_set=label_set,
            masks=masks,
            sample_id=f"case{i}",
            index=True,
        )
        for i in range(3)
    ]


@pytest.fixture
def series(tmp_path: Path, label_set: LabelSet) -> Path:
    """Two visits of one subject, lesions tracked across them."""
    return write_series(tmp_path / "series.medh5", label_set)


# -- pytest-run-parallel ------------------------------------------------------
#
# The free-threaded CI job runs `tests/format` and `tests/tools` again with each
# test's body on several threads at once.  What such a run finds should be
# medh5's, so a test that races the other threads over something of its own
# runs on one thread:
#
# - a file a fixture wrote.  The plugin gives each thread its own `tmp_path`,
#   but a fixture is set up once and shared, and a test that plants a defect
#   in its file, or amends it, races the others over the bytes;
# - a process-wide thing it changes or counts: a module global it patches, a
#   registry name, the engine's read counters;
# - a process it starts, which inherits whatever files the other threads have
#   open (`h5py` opens them inheritable) and keeps their locks;
# - the modules that start threads of their own, or build the whole corpus.

_TEMPORARY = {"tmp_path", "tmpdir", "tmp_path_factory", "tmpdir_factory"}

_ONE_THREAD = {
    "test_threads.py": "starts threads of its own, and times them",
    "test_conformance.py": "builds the whole corpus in each test",
    "test_storage.py::TestOneReplace": "patches nibabel.save, a module global",
    "test_labels.py::TestRegistry::test_register_and_unregister": (
        "registers and drops one name in the process-wide registry"
    ),
}

_IN_SOURCE = {
    "_read_stats": "counts the reads of the whole process",
    "mock.patch": "patches a module global",
    "subprocess": "starts a process, which inherits the other threads' files",
    "--workers": "starts processes, which inherit the other threads' files",
}


def _shared_files(item: pytest.Item) -> list[str]:
    """The fixtures *item* requests that wrote files for it."""
    defs = item._fixtureinfo.name2fixturedefs

    def wrote(name: str, seen: frozenset[str]) -> bool:
        return name not in seen and any(
            arg in _TEMPORARY or wrote(arg, seen | {name})
            for definition in defs.get(name, ())
            for arg in definition.argnames
        )

    return [
        n for n in item.fixturenames if n not in _TEMPORARY and wrote(n, frozenset())
    ]


def _one_thread(item: pytest.Item) -> str | None:
    """Why *item* runs on one thread, or ``None``."""
    module = item.path.name
    names = [module, *(f"{module}::{part}" for part in _qualified(item))]
    for name in names:
        if name in _ONE_THREAD:
            return _ONE_THREAD[name]
    try:
        source = inspect.getsource(item.function)
    except (AttributeError, OSError, TypeError):  # pragma: no cover - not a function
        source = ""
    for needle, reason in _IN_SOURCE.items():
        if needle in source:
            return reason
    shared = _shared_files(item)
    if shared:
        return f"shares the files of {', '.join(sorted(shared))} between threads"
    return None


def _qualified(item: pytest.Item) -> list[str]:
    """``Class`` and ``Class::test`` for a method, ``test`` for a function."""
    parts = item.nodeid.split("::")[1:]
    parts[-1] = parts[-1].split("[")[0]
    return ["::".join(parts[: i + 1]) for i in range(len(parts))]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.pluginmanager.has_plugin("run-parallel"):
        return
    for item in items:
        reason = _one_thread(item)
        if reason is not None:
            item.add_marker(pytest.mark.thread_unsafe(reason=reason))
