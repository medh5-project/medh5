"""What the clinical importers share: source times put on a sample's clock.

A source time is a wall-clock reading (no zone), an instant (an offset or a
zone was stored with it), or a calendar period known to the year, month or
day.  Each becomes inclusive bounds on the subject clock (1.1 §5.2), and
nothing is assumed on the way: a wall-clock time needs a zone to name an
instant, a relative clock needs its origin, a period spans every instant it
may cover, a reading that falls twice (a daylight-saving fall-back) is bounded
by both, and one that never occurred (the spring-forward gap) is refused.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import re
import sqlite3
import unicodedata
import zoneinfo
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from medh5.clinical import DAY, HOUR, Bounds, ClinicalRecords, Clock, augment, utc_us
from medh5.errors import MEDH5Error, MEDH5ValidationError
from medh5.io.report import ConversionReport, SubjectImport

_NAIVE_EPOCH = datetime.datetime(1970, 1, 1)
_UTC = datetime.timezone.utc
_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")


def valid_id(value: str) -> bool:
    """Whether *value* is a 1.0 id (`[A-Za-z0-9_.-]{1,128}`)."""
    return _ID.fullmatch(value) is not None


def wall_us(moment: datetime.datetime) -> int:
    """A naive ``datetime`` as microseconds since 1970-01-01 on its own
    wall clock."""
    delta = moment - _NAIVE_EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def wall_datetime(us: int) -> datetime.datetime:
    """The naive ``datetime`` *us* wall-clock microseconds name."""
    return _NAIVE_EPOCH + datetime.timedelta(microseconds=us)


def zone_of(value: str | datetime.tzinfo | None) -> datetime.tzinfo | None:
    if value is None or isinstance(value, datetime.tzinfo):
        return value
    try:
        return zoneinfo.ZoneInfo(value)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        raise MEDH5ValidationError(
            f"timezone {value!r} is not an IANA time zone this Python knows "
            "(on Windows, install tzdata)"
        ) from None


def parse_moment(value: str | datetime.datetime, what: str) -> datetime.datetime:
    """An ISO 8601 date-time, or a ``datetime`` as given."""
    if isinstance(value, datetime.datetime):
        return value
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise MEDH5ValidationError(
            f"{what} {value!r} is not an ISO 8601 date-time"
        ) from None


class Timeline:
    """Puts source times on *clock*.

    ``timezone`` reads wall-clock times; ``origin`` is the instant (or the
    wall-clock reading) a relative clock counts from.  On a ``utc`` or
    ``shifted_utc`` clock a wall-clock time needs ``timezone``; on a relative
    clock with neither a zone nor an aware origin, wall-clock times are
    counted from a wall-clock origin as read (:attr:`uniform`).
    """

    def __init__(
        self,
        clock: Clock,
        *,
        timezone: str | datetime.tzinfo | None,
        origin: str | datetime.datetime | None,
        converter: str,
    ) -> None:
        if clock.reference not in ("utc", "shifted_utc", "relative"):
            raise MEDH5ValidationError(
                f"clock {clock.id!r} has reference {clock.reference!r}; "
                "expected utc, shifted_utc or relative"
            )
        self.clock = clock
        self.zone = zone_of(timezone)
        self.relative = clock.reference == "relative"
        self.uniform = False
        """Wall-clock times counted from a wall-clock origin, no zone known."""
        self._origin_us = 0
        self._origin_wall: int | None = None
        if not self.relative:
            if origin is not None:
                raise MEDH5ValidationError(
                    f"origin is for a relative clock; clock {clock.id!r} is "
                    f"{clock.reference}, so times are placed as they are"
                )
            return
        if origin is None:
            raise MEDH5ValidationError(
                f"clock {clock.id!r} is relative, and {converter} invents no "
                "origin: give origin, the instant the clock's zero is"
            )
        moment = parse_moment(origin, "origin")
        if moment.tzinfo is not None and moment.utcoffset() is not None:
            self._origin_us = utc_us(moment)
        elif self.zone is not None:
            lo, hi = self._resolve(wall_us(moment))
            if lo != hi:
                raise MEDH5ValidationError(
                    f"origin {moment.isoformat()} occurs twice in "
                    f"{self.zone}: give it with its offset"
                )
            self._origin_us = lo
        else:
            self._origin_wall = wall_us(moment)
            self.uniform = True

    def _resolve(self, wall: int) -> Bounds:
        """The instants (UTC) the wall-clock reading *wall* names in the zone."""
        assert self.zone is not None
        moment = wall_datetime(wall)
        found: set[int] = set()
        for fold in (0, 1):
            instant = utc_us(moment.replace(tzinfo=self.zone, fold=fold))
            back = (
                datetime.datetime(1970, 1, 1, tzinfo=_UTC)
                + datetime.timedelta(microseconds=instant)
            ).astimezone(self.zone)
            if back.replace(tzinfo=None) == moment:
                found.add(instant)
        if not found:
            raise MEDH5ValidationError(
                f"{moment.isoformat()} never occurred in {self.zone} (a "
                "daylight-saving gap): the source time names no instant"
            )
        return min(found), max(found)

    def _boundary(self, wall: int) -> int:
        """The instant a period boundary at wall-clock *wall* begins."""
        assert self.zone is not None
        return utc_us(wall_datetime(wall).replace(tzinfo=self.zone))

    def _shift(self, bounds: Bounds) -> Bounds:
        return bounds[0] - self._origin_us, bounds[1] - self._origin_us

    def instant(self, utc: int) -> Bounds:
        """An instant, as microseconds since the Unix epoch."""
        if self._origin_wall is not None:
            raise MEDH5ValidationError(
                "the source gives an instant and the origin is a wall-clock "
                "reading: give timezone, or the origin with its offset"
            )
        return self._shift((utc, utc))

    def wall(self, wall: int) -> Bounds:
        """A wall-clock reading, as microseconds since 1970 on that clock."""
        if self._origin_wall is not None:
            return wall - self._origin_wall, wall - self._origin_wall
        if self.zone is None:
            raise MEDH5ValidationError(
                f"{wall_datetime(wall).isoformat()} has no time zone, so it "
                f"names no instant on {self.clock.reference} clock "
                f"{self.clock.id!r}: give timezone"
            )
        return self._shift(self._resolve(wall))

    def period(self, start: int, end: int) -> Bounds:
        """Every instant of the wall-clock period ``[start, end)``.

        With no zone known, the period spans every instant it covers
        somewhere: from its start at UTC+14 to its end at UTC-12."""
        if self._origin_wall is not None:
            return start - self._origin_wall, end - 1 - self._origin_wall
        if self.zone is None:
            return self._shift((start - 14 * HOUR, end - 1 + 12 * HOUR))
        return self._shift((self._boundary(start), self._boundary(end) - 1))

    def day(self, day: datetime.date) -> Bounds:
        """Every instant of a calendar *day*."""
        start = wall_us(datetime.datetime.combine(day, datetime.time()))
        return self.period(start, start + DAY)


def option_map(
    value: Mapping[str, Any] | None, what: str, allowed: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """A mapping option as a ``dict``; keys outside *allowed* are refused."""
    found = dict(value or {})
    if allowed is not None:
        unknown = sorted(set(found) - set(allowed))
        if unknown:
            raise MEDH5ValidationError(
                f"{what} has no key {', '.join(map(repr, unknown))}; it takes "
                f"{', '.join(allowed)}"
            )
    return found


def cohort_targets(
    mapping: Mapping[Any, str | os.PathLike[str]],
    out_dir: str | os.PathLike[str] | None,
    source: str | os.PathLike[str],
) -> list[tuple[Any, Path, Path | None]]:
    """Each mapped subject, the sample it augments, and where the result is
    written: the sample itself, or a file of its name in *out_dir*.  Nothing
    is written inside *source*, the dataset, shard or export the import
    reads, nor over it."""
    if not mapping:
        raise MEDH5ValidationError("the mapping names no subject to import")
    root = Path(os.fspath(source))
    out = None if out_dir is None else Path(os.fspath(out_dir))
    if out is not None and _inside(out, root):
        raise MEDH5ValidationError(
            f"out_dir {out} is inside {root}, which this import reads: a "
            "sample written there could replace a file it reads, so write the "
            "samples outside it"
        )
    targets = []
    for subject, sample in mapping.items():
        path = Path(os.fspath(sample))
        targets.append((subject, path, None if out is None else out / path.name))
    # A sample holds one subject: two subjects written to one file, or one
    # written over another's source, would put two histories in it.  Checked
    # before anything is written.
    sources = {file_identity(path): subject for subject, path, _ in targets}
    written: dict[Any, Any] = {}
    read = file_identity(root)
    for subject, path, target in targets:
        where = path if target is None else target
        key = file_identity(where)
        if key == read or _inside(where, root):
            raise MEDH5ValidationError(
                f"subject {subject!r} would be written to {where}, inside or "
                f"over {root}, which this import reads: write the samples "
                "outside it"
            )
        other = written.get(key, sources.get(key, subject))
        if other != subject:
            raise MEDH5ValidationError(
                f"subjects {other!r} and {subject!r} would both be written to "
                f"{where}: a sample holds one subject's history, so map each "
                "subject to a sample of its own"
            )
        written[key] = subject
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    return targets


def _inside(path: Path, tree: Path) -> bool:
    """Whether *path* is *tree* or lies inside it, once both are resolved."""
    try:
        return path.resolve().is_relative_to(tree.resolve())
    except (OSError, RuntimeError):  # RuntimeError: a link loop
        return False


def file_identity(path: Path) -> Any:
    """What tells two paths to one file apart: the file itself, by device
    and inode, where it exists (a link, a hard link, ``sub/../a``), else its
    directory and its name, case-folded and NFC-normalised.  A file not yet
    written may land on a file system that does not tell ``A.medh5`` from
    ``a.medh5`` (macOS's, Windows'), so two such names are one file here."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):  # RuntimeError: a link loop
        resolved = path.absolute()
    with contextlib.suppress(OSError):
        found = resolved.stat()
        return (found.st_dev, found.st_ino)
    parent: Any = resolved.parent
    with contextlib.suppress(OSError):
        found = parent.stat()
        parent = (found.st_dev, found.st_ino)
    return (parent, unicodedata.normalize("NFC", resolved.name).casefold())


class Tally:
    """How many distinct keys were seen, and how many items of them: kept in
    a temporary database on disk (SQLite's, removed when it closes), so the
    subjects of a dataset that a mapping does not name cost a bounded buffer
    in memory, never a set of them.  ``None`` is a key of its own."""

    BUFFER = 1_000
    """Keys gathered before they are written to the database."""

    def __init__(self) -> None:
        self._db = sqlite3.connect("")
        self._db.executescript(
            "PRAGMA cache_size = -2048; PRAGMA temp_store = FILE;"
            "CREATE TABLE tally (key, n INTEGER);"
        )
        self._buffer: Counter[Any] = Counter()

    def __enter__(self) -> Tally:
        return self

    def __exit__(self, *_: object) -> None:
        self._db.close()

    def add(self, key: Any, n: int = 1) -> None:
        self._buffer[key] += n
        if len(self._buffer) >= self.BUFFER:
            self._flush()

    def _flush(self) -> None:
        self._db.executemany("INSERT INTO tally VALUES (?, ?)", self._buffer.items())
        self._buffer.clear()

    def counts(self) -> tuple[int, int]:
        """The distinct keys, and the items of all of them."""
        self._flush()
        keys, items = self._db.execute(
            "SELECT COUNT(DISTINCT key) + COALESCE(MAX(key IS NULL), 0),"
            " COALESCE(SUM(n), 0) FROM tally"
        ).fetchone()
        return int(keys), int(items)


def sample_clock(path: Path) -> Clock:
    """The clinical clock of the sample at *path*: what a cohort import puts
    a subject's times on when no clock is given."""
    from medh5.sample import open_sample

    with open_sample(path) as found:
        clinical = found.clinical
        if clinical is None:
            raise MEDH5ValidationError(
                f"{path} has no clinical clock to import onto: give clock "
                "(--clock-id and --clock)"
            )
        clock: Clock = clinical.clock
        return clock


def import_subject(
    subject: Any,
    sample: Path,
    out: Path | None,
    clock: Clock | None,
    read: Callable[[Clock], tuple[ClinicalRecords, ConversionReport]],
) -> SubjectImport:
    """Read one subject's records onto *clock* (else the sample's own) and
    add them to *sample*, as the per-sample importer does; a refusal is the
    subject's result, never the cohort's."""
    try:
        records, report = read(clock if clock is not None else sample_clock(sample))
        result = augment(sample, records, out=out)
    except (MEDH5Error, OSError) as refusal:
        return SubjectImport(subject, os.fspath(sample), error=str(refusal))
    report.outputs.append(str(result["path"]))
    return SubjectImport(subject, os.fspath(sample), result, report)


def subject_died(label: str, task: tuple[Any, ...], status: str) -> SubjectImport:
    """A subject whose worker stopped (``augment-cohort --workers``) is
    refused, as a subject that cannot be imported is: the others are still
    written.  Whether its sample was written is not known."""
    subject, sample = task[0], task[1]
    return SubjectImport(
        subject, os.fspath(sample), error=f"the worker importing it stopped ({status})"
    )


__all__: list[str] = []
