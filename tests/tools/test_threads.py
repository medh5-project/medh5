"""One thread setting: how many threads one call may use for its work
outside HDF5 (``medh5.set_threads``, ``MEDH5_THREADS``, ``--threads``)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import medh5


def _python(code: str, threads: str | None) -> subprocess.CompletedProcess[str]:
    """``code`` in a fresh interpreter, with ``MEDH5_THREADS`` as given."""
    env = {k: v for k, v in os.environ.items() if k != "MEDH5_THREADS"}
    if threads is not None:
        env["MEDH5_THREADS"] = threads
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def restore():
    """Leave the process's count as the test found it."""
    before = medh5.get_threads()
    yield
    medh5.set_threads(before)


def test_the_count_is_what_was_set(restore) -> None:
    medh5.set_threads(3)
    assert medh5.get_threads() == 3
    medh5.set_threads(1)
    assert medh5.get_threads() == 1


@pytest.mark.parametrize("n", [0, -1])
def test_a_count_below_one_is_refused(restore, n: int) -> None:
    medh5.set_threads(2)
    with pytest.raises(ValueError, match="at least 1"):
        medh5.set_threads(n)
    assert medh5.get_threads() == 2


def test_the_default_is_one() -> None:
    ran = _python("import medh5; print(medh5.get_threads())", None)
    assert ran.stdout.strip() == "1", ran.stderr


def test_the_environment_is_read_once_at_the_first_call_that_asks() -> None:
    ran = _python(
        """
        import os, medh5
        os.environ["MEDH5_THREADS"] = "5"   # before the first ask: read
        first = medh5.get_threads()
        os.environ["MEDH5_THREADS"] = "6"   # after it: not read again
        print(first, medh5.get_threads())
        """,
        "3",
    )
    assert ran.stdout.split() == ["5", "5"], ran.stderr


def test_set_threads_comes_before_the_environment() -> None:
    ran = _python("import medh5; medh5.set_threads(2); print(medh5.get_threads())", "8")
    assert ran.stdout.strip() == "2", ran.stderr


@pytest.mark.parametrize("value", ["0", "-2", "four", "", "1.5"])
def test_an_environment_that_is_not_a_count_is_refused_naming_it(value: str) -> None:
    ran = _python(
        """
        import medh5
        try:
            medh5.get_threads()
        except ValueError as e:
            print(e)
        medh5.set_threads(2)                # a set count needs no environment
        print(medh5.get_threads())
        """,
        value,
    )
    lines = ran.stdout.splitlines()
    assert lines[0] == f"MEDH5_THREADS must be a positive integer, not {value!r}"
    assert lines[1] == "2"


def test_a_dataloader_worker_uses_one_thread(restore) -> None:
    pytest.importorskip("torch")
    from medh5.torch import worker_init_fn

    medh5.set_threads(4)
    worker_init_fn(0)
    assert medh5.get_threads() == 1
