"""Per-worker file handles (spec §14.4).

An open HDF5 file **must not** cross a ``fork``.  The HDF5 library keeps
state behind the file descriptor, and a child that inherits and then uses a
parent's handle produces corrupt reads that look like data errors, not
concurrency errors --- which is why they are usually diagnosed as a broken
dataset months later.  The cache is therefore keyed by PID and abandoned
(never closed --- the descriptors belong to the parent) as soon as it notices
it is running somewhere else.  Abandoning is explicit: each inherited sample
pins its engine handle so that nothing in this process closes the file, and is
kept alive for the life of the process, so no destructor of what it holds runs
either.

Reopening per ``__getitem__`` would be correct but costs an open per patch;
with a 1000-file dataset and 8 workers that is the dominant cost in the
dataloader.  An LRU of open samples per worker removes it while keeping the
number of descriptors bounded.

**A handle is a sample, not a file.**  The cache is keyed by ``(path,
sample_key)``: a ``.medh5`` is ``(path, None)``, and each member of a
``.medh5c`` collection is its own entry --- opened, leased, evicted and
abandoned across a fork exactly like a standalone file, so a task whose
sources are collection members reads them through the same rules.

**Threads share the cache, so it is locked, and a handle in use is never
closed.**  A thread-based loader calls ``__getitem__`` from several threads of
one process; without a lock one thread's eviction could close the ``Sample``
another was reading --- and §14.4 already requires external locking to share a
handle across threads.  :meth:`HandleCache.lease` holds a handle against
eviction for the length of an item, and the datasets read through it; a
cache full of leased handles grows past ``maxsize`` rather than closing one,
and shrinks back as they are released.

**A cached handle can outlive its file.**  ``amend`` is copy-on-write: it
writes a new file and renames it over the old, and a handle opened before
keeps reading the old inode.  A lease given the ``content_id`` it must read
(:meth:`HandleCache.lease`) therefore checks the handle's, and the file's
identity (device, inode, size, modification time) against the one recorded
when the handle was opened --- one ``stat`` per lease: a stale handle is
reopened (for that lease alone while another is still reading the old
handle), and a sample that is still another version is refused (T302) ---
what a task's rows pin is what they read, warm cache or cold.  A lease in
progress finishes on the bytes it opened.

**A fork is noticed in the child before any thread runs there.**  An
``os.register_at_fork`` hook abandons every cache's handles and replaces its
lock as the child starts, while it has one thread.  The PID check each call
makes stays as the fallback, under a lock of its own: two threads of a child
that both noticed the fork once replaced the cache's lock one after the other,
and then held different locks over the same handles.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

from medh5.sample import Sample, open_sample

DEFAULT_MAXSIZE = 32

Key = tuple[str, str | None]
"""``(path, sample_key)``: a standalone file is ``(path, None)``."""

Identity = tuple[int, int, int, int] | None
"""``(st_dev, st_ino, st_size, st_mtime_ns)`` of a path; ``None`` unknown."""


def _key(path: str | os.PathLike[str], member: str | None) -> Key:
    return (str(Path(path)), member)


def _open(key: Key) -> Sample:
    path, member = key
    if member is None:
        return open_sample(path)
    from medh5.collection import open_any

    found = open_any(path, key=member)
    if not isinstance(found, Sample):  # pragma: no cover - open_any with a key
        found.close()
        raise TypeError(f"{path}::{member} is not a sample")
    return found


def _identity(path: str) -> Identity:
    try:
        found = os.stat(path)
    except OSError:
        return None
    return (found.st_dev, found.st_ino, found.st_size, found.st_mtime_ns)


def _open_identified(key: Key) -> tuple[Sample, Identity]:
    """A handle for *key* and the identity of the file it holds.

    A file replaced while it was being opened leaves which one the handle
    holds unknown, so it is opened again; ``None`` after three tries, which
    the next pinned lease takes for a replacement.
    """
    for _ in range(3):
        before = _identity(key[0])
        sample = _open(key)
        if _identity(key[0]) == before and before is not None:
            return sample, before
        sample.close()
    return _open(key), None


_ABANDONED: list[Sample] = []
"""Samples inherited across a fork: pinned, never closed, never collected."""


def _abandon(samples: Iterator[Sample] | list[Sample]) -> None:
    for sample in samples:
        with contextlib.suppress(Exception):  # pragma: no cover - best effort
            sample._handle.abandon()
        _ABANDONED.append(sample)


_CACHES: weakref.WeakSet[HandleCache] = weakref.WeakSet()
"""Every cache, for the fork hook to adopt in the child."""

_ADOPTING = threading.Lock()
"""Held by the PID check that adopts a cache in a child the hook missed."""


def _after_fork_in_child() -> None:
    """Adopt every cache in a new child, before a second thread exists: its
    handles are the parent's (abandoned), and a lock another of the parent's
    threads held at the fork would be held forever."""
    global _ADOPTING
    _ADOPTING = threading.Lock()
    for cache in list(_CACHES):
        cache._adopt()


if hasattr(os, "register_at_fork"):  # POSIX; Windows has no fork
    os.register_at_fork(after_in_child=_after_fork_in_child)


class HandleCache:
    """PID-scoped, thread-safe LRU cache of open samples."""

    __slots__ = (
        "__weakref__",
        "_identities",
        "_items",
        "_lock",
        "_owner_pid",
        "_pins",
        "maxsize",
        "opens",
    )

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE) -> None:
        self.maxsize = int(maxsize)
        self._items: OrderedDict[Key, Sample] = OrderedDict()
        self._pins: dict[Key, int] = {}
        self._identities: dict[Key, Identity] = {}
        self._lock = threading.Lock()
        self._owner_pid = os.getpid()
        self.opens = 0
        _CACHES.add(self)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def owner_pid(self) -> int:
        return self._owner_pid

    def _ensure_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            # A child the fork hook did not reach: adopt it once, whichever
            # thread notices first.
            with _ADOPTING:
                if os.getpid() != self._owner_pid:
                    self._adopt()

    def _adopt(self) -> None:
        """Take the cache over in a forked child.  Abandon, never close: the
        descriptors are the parent's.  The lock is replaced too --- one held
        by another of the parent's threads at the fork is held forever in the
        child."""
        _abandon(list(self._items.values()))
        self._lock = threading.Lock()
        self._items = OrderedDict()
        self._pins = {}
        self._identities = {}
        self._owner_pid = os.getpid()

    def _acquire(self, key: Key, *, pin: bool) -> Sample:
        """The handle for *key*, opened if need be; the caller holds the lock."""
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
        else:
            cached, self._identities[key] = _open_identified(key)
            self.opens += 1
            self._items[key] = cached
        if pin:
            self._pins[key] = self._pins.get(key, 0) + 1
        return cached

    def _overflow(self, keep: Key | None = None) -> list[Sample]:
        """Evict least-recently-used idle handles past ``maxsize``; under the lock.

        *keep* is the handle being handed out: evicting it would return a
        closed file to the caller that asked for it.
        """
        evicted: list[Sample] = []
        for key in list(self._items):
            if len(self._items) <= self.maxsize:
                break
            if key == keep or self._pins.get(key):
                continue
            evicted.append(self._items.pop(key))
            self._identities.pop(key, None)
        return evicted

    @staticmethod
    def _close(samples: list[Sample]) -> None:
        for sample in samples:
            with contextlib.suppress(Exception):  # pragma: no cover - best effort
                sample.close()

    def get(
        self, path: str | os.PathLike[str], sample_key: str | None = None
    ) -> Sample:
        """An open handle, not held against eviction; see :meth:`lease`.

        ``sample_key`` names a member of a ``.medh5c`` collection.
        """
        self._ensure_owner()
        key = _key(path, sample_key)
        with self._lock:
            sample = self._acquire(key, pin=False)
            evicted = self._overflow(keep=key)
        self._close(evicted)
        return sample

    @contextlib.contextmanager
    def lease(
        self,
        path: str | os.PathLike[str],
        sample_key: str | None = None,
        *,
        content_id: str | None = None,
    ) -> Iterator[Sample]:
        """An open handle that no eviction closes until the block ends.

        With ``content_id``, the handle is that version of the sample: a
        cached handle whose file was replaced since it was opened --- another
        version, or the file's identity changed --- is reopened, in the
        cache, or, while another lease is still reading the old handle, for
        this lease alone; and a sample that is another version all the same
        is refused with ``MEDH5ValidationError`` (T302).  A warm handle of the
        pinned version was served from a replaced file, while a cold cache
        refused the same row.
        """
        self._ensure_owner()
        key = _key(path, sample_key)
        stale: list[Sample] = []
        private: list[Sample] = []
        with self._lock:
            sample = self._acquire(key, pin=True)
            if content_id is not None and (
                sample.content_id != content_id
                or _identity(key[0]) != self._identities.get(key)
                or self._identities.get(key) is None
            ):
                try:
                    sample, identity = _open_identified(key)
                except BaseException:
                    self._unpin(key)
                    raise
                self.opens += 1
                if self._pins.get(key) == 1:  # no other lease is reading it
                    stale.append(self._items.pop(key))
                    self._items[key] = sample
                    self._identities[key] = identity
                else:
                    private.append(sample)
            evicted = self._overflow(keep=key)
        self._close(stale + evicted)
        try:
            if content_id is not None and sample.content_id != content_id:
                from medh5.errors import MEDH5ValidationError

                where = path if sample_key is None else f"{path}::{sample_key}"
                raise MEDH5ValidationError(
                    f"{where} is now {sample.content_id}; the task pins {content_id} "
                    "--- the source changed after its preflight: re-run it",
                    "T302",
                )
            yield sample
        finally:
            with self._lock:
                self._unpin(key)
                evicted = self._overflow()
            self._close(private + evicted)

    def _unpin(self, key: Key) -> None:
        """Release one lease of *key*; under the lock."""
        remaining = self._pins.get(key, 0) - 1
        if remaining > 0:
            self._pins[key] = remaining
        else:
            self._pins.pop(key, None)

    def resize(self, maxsize: int) -> None:
        """Set ``maxsize``, closing idle handles past it now rather than later."""
        self._ensure_owner()
        with self._lock:
            self.maxsize = int(maxsize)
            evicted = self._overflow()
        self._close(evicted)

    def clear(self) -> None:
        """Drop every handle: abandon what was inherited across a fork, and
        close what this process opened --- but a handle a lease is reading,
        which its lease's end leaves to the LRU.

        Abandoning in the owning process too pinned every handle open for the
        life of the process: each call leaked them all.
        """
        self._ensure_owner()
        with self._lock:
            idle = [key for key in self._items if not self._pins.get(key)]
            samples = [self._items.pop(key) for key in idle]
            for key in idle:
                self._identities.pop(key, None)
        self._close(samples)

    def close_all(self) -> None:
        """Close what *this* process opened, and abandon anything inherited.

        The PID check is not optional here.  A forked child that exits through
        normal interpreter shutdown runs the ``atexit`` hook below, and without
        this it would call into HDF5 to close descriptors belonging to its
        parent --- the one thing the module docstring says must never happen.
        ``worker_init_fn`` prevents it only for callers who remember to pass it,
        and a plain ``os.fork()`` never had the chance.
        """
        self._ensure_owner()
        with self._lock:
            samples = list(reversed(self._items.values()))
            self._items = OrderedDict()
            self._pins = {}
            self._identities = {}
        self._close(samples)


CACHE = HandleCache()


@atexit.register
def _close_cache() -> None:  # pragma: no cover - process exit hook
    CACHE.close_all()


def open_cached(path: str | os.PathLike[str], sample_key: str | None = None) -> Sample:
    """Open a sample --- or a collection member --- through the per-worker cache."""
    return CACHE.get(path, sample_key)


def worker_init_fn(worker_id: int) -> None:
    """``DataLoader(worker_init_fn=...)``: drop handles inherited across ``fork``
    (in a process that opened them, close them; see :meth:`HandleCache.clear`)."""
    CACHE.clear()


def set_cache_size(maxsize: int) -> None:
    """Resize this process's cache: how many files each worker keeps open.

    Lower it when file descriptors are scarce; raise it toward the number of
    files a worker cycles through, so a shuffled epoch does not re-open them
    (see :class:`~medh5.torch.FileGroupedSampler` for the other way round).
    Call it in ``worker_init_fn`` to size each worker's cache.
    """
    if int(maxsize) < 1:
        raise ValueError("the handle cache needs room for at least one file")
    CACHE.resize(int(maxsize))


__all__ = [
    "CACHE",
    "DEFAULT_MAXSIZE",
    "HandleCache",
    "open_cached",
    "set_cache_size",
    "worker_init_fn",
]
