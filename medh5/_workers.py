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
- units that share a file run one after another, in input order: two
  workers never touch one file at once;
- a worker that dies fails only the unit it held, as the command words that
  (``died``); what no worker is left to take runs here, as does everything
  when the system refuses to start one;
- a worker is started fresh (``spawn``), never forked: a child must not
  inherit an open HDF5 file;
- each worker is given this run's thread count (:func:`medh5.get_threads`,
  which ``--threads`` sets for the run), as the native pool hands its
  workers ``MEDH5_THREADS``.

Units are taken from their iterable only when a worker is free for one, so
a command that reads each unit's input as it hands it out holds as many as
there are workers; and a worker is started only for a unit no started one is
free for, so no more start than there are units.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from multiprocessing.connection import Connection, wait
from pathlib import PurePath
from typing import Any, TypeVar

from medh5._core import get_threads, set_threads

T = TypeVar("T")
R = TypeVar("R")


def run(
    fn: Callable[[T], R],
    units: Iterable[tuple[str, T, Sequence[str | os.PathLike[str]]]],
    workers: int,
    died: Callable[[str, T, str], R],
) -> Iterator[R]:
    """``fn(task)`` for each ``(label, task, files)`` of *units*, on up to
    *workers* processes; the results in the order of *units*.

    *fn* and each task cross to a worker by pickle, so *fn* is a function a
    module defines.  *files* are those the unit reads or writes: a unit that
    shares one with a unit still running waits for it, so two never touch a
    file at once and a file's units run in input order, as in one process.
    A unit whose worker dies yields ``died(label, task, status)``, *status*
    the worker's exit (``exit status: 1``, ``signal: 9``), or raises what it
    raises, at the unit's turn.  With one worker, or none, every unit runs
    here, in order.
    """
    if workers <= 1:
        for label, task, _ in units:
            yield _unit(fn, label, task)
        return
    pending = iter(units)
    context = multiprocessing.get_context("spawn")
    pool: list[_Worker] = []
    threads: int | None = None
    # A unit taken from *units* and not yet handed out: its files are
    # another running unit's, or no worker could be started for it.
    waiting: tuple[str, T, frozenset[str]] | None = None
    try:
        done: dict[int, tuple[bool, Any]] = {}
        given = 0
        out = 0
        exhausted = False
        while True:
            # A worker is started only for a unit no started one is free
            # for: never more than there are units.
            while waiting is not None or not exhausted:
                idle = next((w for w in pool if w.alive and w.held is None), None)
                if idle is None and len(pool) >= workers:
                    break
                if waiting is None:
                    unit = next(pending, None)
                    if unit is None:
                        exhausted = True
                        break
                    label, task, files = unit
                    waiting = (label, task, frozenset(map(_file_key, files)))
                if any(waiting[2] & w.keys for w in pool if w.held is not None):
                    break
                if idle is None:
                    if threads is None:
                        threads = get_threads()
                    try:
                        idle = _Worker(context, fn, threads)
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
                        break
                    pool.append(idle)
                idle.give(given, *waiting)
                waiting = None
                given += 1
            while out in done:
                ok, value = done.pop(out)
                out += 1
                if not ok:
                    raise value
                yield value
            busy = [w for w in pool if w.held is not None]
            if not busy:
                if exhausted and waiting is None:
                    return
                # No worker is left: the rest runs here, in order.
                if waiting is not None:
                    yield _unit(fn, waiting[0], waiting[1])
                for label, task, _ in pending:
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


def _file_key(path: str | os.PathLike[str]) -> str:
    """A file by what it is: its device and inode when it exists, so a
    second name or a hard link is the same file; otherwise its path, links
    resolved, case folded --- over-joining two units costs only order."""
    try:
        status = os.stat(path)
    except OSError:
        return "path:" + os.path.normcase(os.path.realpath(path)).casefold()
    return f"file:{status.st_dev}:{status.st_ino}"


def _unit(fn: Callable[[T], R], label: str, task: T) -> R:
    """``fn(task)``, but for the test suite's ``MEDH5_TEST_WORKERS=error:``
    and ``threads:``."""
    if _test_hook("error", label):
        raise RuntimeError(f"{label}: stopped here, as MEDH5_TEST_WORKERS asks")
    if _test_hook("threads", label):
        threads = get_threads()
        raise RuntimeError(f"{label}: {threads} threads, as MEDH5_TEST_WORKERS asks")
    return fn(task)


def _test_hook(what: str, label: str) -> bool:
    """For the test suite only, as ``workers.rs`` reads it:
    ``MEDH5_TEST_WORKERS=exit:NAME`` ends the worker handed the unit named
    ``NAME`` as a crash would; ``error:NAME`` makes that unit raise;
    ``threads:NAME`` makes it raise an error that says the thread count the
    process was given."""
    hook = os.environ.get("MEDH5_TEST_WORKERS", "")
    kind, _, name = hook.partition(":")
    return kind == what and PurePath(label).name == name


class _Worker:
    """One worker process and the unit it holds."""

    def __init__(self, context: Any, fn: Callable[[Any], Any], threads: int) -> None:
        self.connection, theirs = context.Pipe()
        self.process = context.Process(
            target=_serve, args=(theirs, fn, threads), daemon=True
        )
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            raise
        finally:
            theirs.close()
        self.alive = True
        self.held: tuple[int, str, Any] | None = None
        self.keys: frozenset[str] = frozenset()

    def give(self, index: int, label: str, task: Any, keys: frozenset[str]) -> None:
        self.held = (index, label, task)
        self.keys = keys
        # One that has ended says so when its answer is read.
        with contextlib.suppress(OSError, EOFError):
            self.connection.send((index, label, task))

    def answer(self, died: Callable[[str, Any, str], Any]) -> tuple[int, bool, Any]:
        assert self.held is not None
        index, label, task = self.held
        self.held = None
        self.keys = frozenset()
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


def _serve(connection: Connection, fn: Callable[[Any], Any], threads: int) -> None:
    """A worker, given the run's thread count: each unit it is sent,
    answered with its result or the exception it raised, until it is sent
    ``None``."""
    set_threads(threads)
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
