"""``--workers N`` for the commands the Python package runs: units across processes.

The native command line runs the units of its batch commands in worker
processes (``crates/medh5-cli/src/workers.rs``).  The commands it hands to
this package --- ``convert to-nnunet`` and ``clinical augment-cohort`` ---
run theirs with :func:`run`, on the same rules:

- each unit's result comes back in the order the units were given, whatever
  order the workers finish them in, so the command's output is one
  process's;
- an exception a unit raises is raised again at its turn and stops the run
  there, as it would in one process;
- a worker that dies fails only the unit it held, as the command words that
  (``died``); what no worker is left to take runs here, as does everything
  when the system refuses to start one;
- a worker is started fresh (``spawn``), never forked: a child must not
  inherit an open HDF5 file.

Units are taken from their iterable only when a worker is free for one, so
a command that reads each unit's input as it hands it out holds as many as
there are workers; and a worker is started only for a unit no started one is
free for, so no more start than there are units.
"""

from __future__ import annotations

import contextlib
import itertools
import multiprocessing
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from multiprocessing.connection import Connection, wait
from pathlib import PurePath
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run(
    fn: Callable[[T], R],
    units: Iterable[tuple[str, T]],
    workers: int,
    died: Callable[[str, T, str], R],
) -> Iterator[R]:
    """``fn(task)`` for each ``(label, task)`` of *units*, on up to *workers*
    processes; the results in the order of *units*.

    *fn* and each task cross to a worker by pickle, so *fn* is a function a
    module defines.  A unit whose worker dies yields ``died(label, task,
    status)``, *status* the worker's exit (``exit status: 1``, ``signal:
    9``), or raises what it raises, at the unit's turn.  With one worker, or
    none, every unit runs here, in order.
    """
    if workers <= 1:
        for label, task in units:
            yield _unit(fn, label, task)
        return
    pending = iter(units)
    context = multiprocessing.get_context("spawn")
    pool: list[_Worker] = []
    try:
        done: dict[int, tuple[bool, Any]] = {}
        given = 0
        out = 0
        exhausted = False
        while True:
            # A worker is started only for a unit no started one is free
            # for: never more than there are units.
            while not exhausted:
                idle = next((w for w in pool if w.alive and w.held is None), None)
                if idle is None and len(pool) >= workers:
                    break
                unit = next(pending, None)
                if unit is None:
                    exhausted = True
                    break
                if idle is None:
                    try:
                        idle = _Worker(context, fn)
                    except OSError as exc:
                        # The system refuses another process (a limit on
                        # processes or memory): the ones started go on,
                        # and with none, the rest runs here.
                        if not pool:
                            print(
                                f"medh5: cannot start workers ({exc}); "
                                "running in one process",
                                file=sys.stderr,
                            )
                        workers = len(pool)
                        pending = itertools.chain([unit], pending)
                        break
                    pool.append(idle)
                idle.give(given, *unit)
                given += 1
            while out in done:
                ok, value = done.pop(out)
                out += 1
                if not ok:
                    raise value
                yield value
            busy = [w for w in pool if w.held is not None]
            if not busy:
                if exhausted:
                    return
                # No worker is left: the rest runs here, in order.
                for label, task in pending:
                    yield _unit(fn, label, task)
                return
            ready = wait([w.connection for w in busy])
            for worker in busy:
                if worker.connection in ready:
                    index, ok, value = worker.answer(died)
                    done[index] = (ok, value)
    finally:
        for worker in pool:
            worker.close()


def _unit(fn: Callable[[T], R], label: str, task: T) -> R:
    """``fn(task)``, but for the test suite's ``MEDH5_TEST_WORKERS=error:``."""
    if _test_hook("error", label):
        raise RuntimeError(f"{label}: stopped here, as MEDH5_TEST_WORKERS asks")
    return fn(task)


def _test_hook(what: str, label: str) -> bool:
    """For the test suite only, as ``workers.rs`` reads it:
    ``MEDH5_TEST_WORKERS=exit:NAME`` ends the worker handed the unit named
    ``NAME`` as a crash would; ``error:NAME`` makes that unit raise."""
    hook = os.environ.get("MEDH5_TEST_WORKERS", "")
    kind, _, name = hook.partition(":")
    return kind == what and PurePath(label).name == name


class _Worker:
    """One worker process and the unit it holds."""

    def __init__(self, context: Any, fn: Callable[[Any], Any]) -> None:
        self.connection, theirs = context.Pipe()
        self.process = context.Process(target=_serve, args=(theirs, fn), daemon=True)
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            raise
        finally:
            theirs.close()
        self.alive = True
        self.held: tuple[int, str, Any] | None = None

    def give(self, index: int, label: str, task: Any) -> None:
        self.held = (index, label, task)
        # One that has ended says so when its answer is read.
        with contextlib.suppress(OSError, EOFError):
            self.connection.send((index, label, task))

    def answer(self, died: Callable[[str, Any, str], Any]) -> tuple[int, bool, Any]:
        assert self.held is not None
        index, label, task = self.held
        self.held = None
        try:
            got, ok, value = self.connection.recv()
        except (OSError, EOFError):
            self.alive = False
            self.process.join()
            try:
                return index, True, died(label, task, _status(self.process.exitcode))
            except Exception as exc:
                return index, False, exc
        return got, ok, value

    def close(self) -> None:
        """End it once it has finished the unit it holds, as a worker of the
        native command line ends: a unit is never cut off part way."""
        if self.alive:
            with contextlib.suppress(OSError, EOFError):
                if self.held is not None:
                    self.connection.recv()
                self.connection.send(None)
        self.connection.close()
        self.process.join()


def _status(code: int | None) -> str:
    """A worker's end as the native command line words a process's."""
    if code is not None and code < 0:
        return f"signal: {-code}"
    return f"exit status: {code}"


def _serve(connection: Connection, fn: Callable[[Any], Any]) -> None:
    """A worker: each unit it is sent, answered with its result or the
    exception it raised, until it is sent ``None``."""
    while True:
        try:
            unit = connection.recv()
        except (OSError, EOFError):
            return
        if unit is None:
            return
        index, label, task = unit
        if _test_hook("exit", label):
            os._exit(101)
        try:
            answer = (index, True, _unit(fn, label, task))
        except Exception as exc:
            answer = (index, False, exc)
        try:
            connection.send(answer)
        except Exception as exc:
            failure = RuntimeError(f"{type(exc).__name__}: {exc}")
            connection.send((index, False, failure))


__all__: list[str] = []
