"""MEDS import: one subject's rows of a MEDS dataset as clinical records (1.1 §10).

A `MEDS <https://github.com/Medical-Event-Data-Standard/meds>`_ dataset is a
directory of Parquet shards under ``data/``, one row per measurement:

.. code-block:: text

    subject_id  time                 code            numeric_value  text_value
    1001        2020-03-01 08:15:00  LOINC/2160-0    1.1
    1001        null                 GENDER//F

with ``metadata/dataset.json`` naming the dataset.  :func:`read_meds` reads the
rows of one ``subject_id`` (MEDS 0.3; a dataset older than 0.3 names that
column ``patient_id``: pass ``columns={"subject_id": "patient_id"}``) and maps
them onto events, inventing nothing MEDS does not say:

* **Who.** The caller says which MEDS subject a sample is; the mapping is
  recorded as a ``decision`` note.
* **When.** A null ``time`` is a ``static`` event; any other a ``point``.
  MEDS times carry no zone: on a ``utc`` or ``shifted_utc`` clock they need
  ``timezone``, and a relative clock needs ``origin``.  Codes under a
  ``day_precision`` prefix are known to the day, so they span it.
* **Known when.** MEDS records no availability.  It stays null unless a
  column holds it (``columns={"available": ...}``); such events can be a
  task's targets, never its inputs, until it is known (1.1 §5.2).
* **What.** ``code`` under a declared prefix maps to a code system
  (``code_systems={"LOINC": "http://loinc.org"}``: ``LOINC/2160-0`` is that
  system's ``2160-0``); any other stays verbatim in the dataset's own system,
  ``meds:<dataset_name>``.  No standard code is guessed.
* **Values.** ``numeric_value`` and ``text_value`` are ``value_num`` and
  ``value_text``; a row holding both is refused unless ``both_values`` says
  which to keep.  A unit comes only from a declared column: MEDS has none.
* **Ids.** ``event_id = record_id = "meds." +`` 24 hex digits of the SHA-256
  of the row's canonical JSON (with the declared ``unit`` and ``available``
  columns), so a re-extraction keeps them, and importing the same rows again
  is refused as repeated ids.
* **Left out unless asked.** ``MEDS_BIRTH`` rows (a birth instant
  identifies), and free text: rows of ``document_codes`` become ``document``
  events, owning their ``text_value`` only with ``texts=True``.

Reading needs ``pyarrow`` (``pip install 'medh5[meds]'``).  The dataset is
scanned once per call.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from medh5 import _core
from medh5._optional import require
from medh5.clinical import (
    EVENT_KINDS,
    STATUSES,
    Bounds,
    ClinicalRecords,
    Clock,
    Document,
    Event,
    Link,
    augment,
)
from medh5.errors import MEDH5ValidationError
from medh5.io import _clinical
from medh5.io._clinical import Timeline, option_map, wall_datetime, zone_of
from medh5.io.report import CohortImport, ConversionReport, SubjectImport

BIRTH = "MEDS_BIRTH"
"""The MEDS code of a subject's birth."""

COLUMNS: tuple[str, ...] = (
    "subject_id",
    "time",
    "code",
    "numeric_value",
    "text_value",
    "unit",
    "available",
)
"""The columns ``columns=`` may rename: MEDS's own five, and the two MEDS
lacks, which are read only when declared."""

BOTH_VALUES: tuple[str, ...] = ("refuse", "numeric", "text")


def _pyarrow() -> Any:
    return require("pyarrow", extra="meds", purpose="reading MEDS")


def _dataset_name(root: Path) -> str | None:
    meta = root / "metadata" / "dataset.json"
    if not meta.is_file():
        return None
    try:
        name = json.loads(meta.read_text(encoding="utf-8")).get("dataset_name")
    except (ValueError, AttributeError):
        raise MEDH5ValidationError(f"{meta} is not a MEDS dataset.json") from None
    return str(name) if name else None


def _shards(source: Path) -> tuple[Path, list[Path]]:
    """The dataset root and its Parquet shards."""
    if source.is_file():
        root = source.parent.parent if source.parent.name == "data" else source.parent
        return root, [source]
    if not source.is_dir():
        raise MEDH5ValidationError(f"{source} is not a MEDS dataset or shard")
    data = source / "data"
    found = sorted((data if data.is_dir() else source).rglob("*.parquet"))
    if not found:
        raise MEDH5ValidationError(f"{source} holds no MEDS shard (data/*.parquet)")
    return source, found


def _prefixed(code: str, prefixes: Iterable[str]) -> str | None:
    """The longest of *prefixes* *code* lies under (``LOINC`` holds
    ``LOINC/2160-0`` and ``LAB//50912``), or ``None``."""
    best = None
    for prefix in prefixes:
        under = code == prefix or code.startswith(prefix + "/")
        if under and (best is None or len(prefix) > len(best)):
            best = prefix
    return best


def _times(pa: Any, column: Any, name: str) -> tuple[list[int | None], str | None]:
    """A timestamp column as microseconds (wall-clock when it has no zone,
    UTC when it has one), and its zone."""
    kind = column.type
    if not pa.types.is_timestamp(kind):
        raise MEDH5ValidationError(f"MEDS column {name!r} is {kind}, not a timestamp")
    try:
        micros = column.cast(pa.timestamp("us", kind.tz)).cast(pa.int64())
    except pa.ArrowInvalid:
        raise MEDH5ValidationError(
            f"MEDS column {name!r} holds times finer than a microsecond, which a "
            "clinical clock cannot place"
        ) from None
    return micros.to_pylist(), kind.tz


def _float32(value: float) -> float:
    """The shortest decimal that reads back as the float32 *value*."""
    import numpy as np

    single = np.float32(value)
    return (
        float(np.format_float_positional(single, unique=True, trim="-"))
        if np.isfinite(single)
        else float(value)
    )


def _row_id(row: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_core.canonical_json(dict(row))).hexdigest()
    return "meds." + digest[:24]


def read_meds(
    source: str | os.PathLike[str],
    *,
    subject_id: int,
    clock: Clock,
    timezone: str | datetime.tzinfo | None = None,
    origin: str | datetime.datetime | None = None,
    code_system: str | None = None,
    code_systems: Mapping[str, str] | None = None,
    kinds: Mapping[str, str] | None = None,
    status: str = "unknown",
    day_precision: Iterable[str] = (),
    document_codes: Iterable[str] = (),
    exclude_codes: Iterable[str] = (BIRTH,),
    texts: bool = False,
    both_values: str = "refuse",
    columns: Mapping[str, str] | None = None,
) -> tuple[ClinicalRecords, ConversionReport]:
    """The rows of MEDS subject *subject_id* in *source*, as clinical records
    on *clock*, and what the conversion decided.

    *source* is a MEDS dataset's root (its ``data/`` shards are read) or one
    shard.  ``code_systems`` and ``kinds`` map code prefixes to a code system
    and an event kind (``other`` otherwise); ``day_precision`` names prefixes
    whose times are known to the day; ``status`` is every event's
    (``unknown`` unless the dataset is known to hold final results).
    ``columns`` renames MEDS's columns and declares ``unit`` and ``available``
    ones.  Refused, as ``MEDH5ValidationError``: no rows for the subject, a
    time with no zone on a UTC clock, a relative clock without ``origin``, a
    row holding both values under ``both_values="refuse"``.
    """
    pa = _pyarrow()
    import pyarrow.dataset as ds

    report = ConversionReport(source=os.fspath(source), converter="from-meds")
    rules = _Rules.of(
        code_systems=code_systems,
        kinds=kinds,
        status=status,
        day_precision=day_precision,
        document_codes=document_codes,
        exclude_codes=exclude_codes,
        texts=texts,
        both_values=both_values,
        columns=columns,
    )
    timeline = Timeline(clock, timezone=timezone, origin=origin, converter="from-meds")
    scan = _Scan.of(source, rules, code_system)
    table = scan.dataset.to_table(
        columns=scan.wanted,
        filter=ds.field(rules.names["subject_id"]) == int(subject_id),
    )
    records = _records(pa, rules, scan, int(subject_id), table, timeline, report)
    return records, report


@dataclass(frozen=True, slots=True)
class _Rules:
    """``read_meds``'s options, checked: how every subject's rows are read."""

    names: dict[str, str]
    systems: dict[str, str]
    kinds: dict[str, str]
    status: str
    both_values: str
    day_prefixes: tuple[str, ...]
    documents_of: frozenset[str]
    excluded: frozenset[str]
    texts: bool

    @classmethod
    def of(
        cls,
        *,
        code_systems: Mapping[str, str] | None,
        kinds: Mapping[str, str] | None,
        status: str,
        day_precision: Iterable[str],
        document_codes: Iterable[str],
        exclude_codes: Iterable[str],
        texts: bool,
        both_values: str,
        columns: Mapping[str, str] | None,
    ) -> _Rules:
        names = {c: c for c in COLUMNS[:5]}
        names.update(option_map(columns, "columns", COLUMNS))
        kinds_by = option_map(kinds, "kinds")
        for prefix, kind in kinds_by.items():
            if kind not in EVENT_KINDS:
                raise MEDH5ValidationError(
                    f"kinds[{prefix!r}] is {kind!r}; an event kind is one of "
                    f"{', '.join(EVENT_KINDS)}"
                )
        if status not in STATUSES:
            raise MEDH5ValidationError(
                f"status {status!r} is not one of {', '.join(STATUSES)}"
            )
        if both_values not in BOTH_VALUES:
            raise MEDH5ValidationError(
                f"both_values {both_values!r} is not one of {', '.join(BOTH_VALUES)}"
            )
        return cls(
            names=names,
            systems=option_map(code_systems, "code_systems"),
            kinds=kinds_by,
            status=status,
            both_values=both_values,
            day_prefixes=tuple(day_precision),
            documents_of=frozenset(document_codes),
            excluded=frozenset(exclude_codes),
            texts=texts,
        )


@dataclass(frozen=True, slots=True)
class _Scan:
    """The dataset a read scans: its columns and the codes' own system."""

    source: str
    dataset: Any
    present: set[str]
    wanted: list[str]
    own_system: str

    @classmethod
    def of(
        cls, source: str | os.PathLike[str], rules: _Rules, code_system: str | None
    ) -> _Scan:
        import pyarrow.dataset as ds

        names = rules.names
        root, shards = _shards(Path(os.fspath(source)))
        if code_system is None:
            name = _dataset_name(root)
            if name is None:
                raise MEDH5ValidationError(
                    f"{root} has no metadata/dataset.json naming the dataset, so its "
                    "codes have no system: give code_system"
                )
            code_system = f"meds:{name}"
        if not code_system:
            raise MEDH5ValidationError("code_system cannot be empty")
        dataset = ds.dataset([os.fspath(s) for s in shards], format="parquet")
        present = set(dataset.schema.names)
        if names["subject_id"] not in present:
            hint = (
                ": a dataset older than MEDS 0.3 names it patient_id --- pass "
                'columns={"subject_id": "patient_id"}'
                if "patient_id" in present
                else ""
            )
            raise MEDH5ValidationError(
                f"{source} has no column {names['subject_id']!r}{hint}"
            )
        for key in ("time", "code", *(k for k in ("unit", "available") if k in names)):
            if names[key] not in present:
                raise MEDH5ValidationError(f"{source} has no column {names[key]!r}")
        wanted = [names[k] for k in names if names[k] in present]
        return cls(os.fspath(source), dataset, present, wanted, code_system)


def _records(
    pa: Any,
    rules: _Rules,
    scan: _Scan,
    subject_id: int,
    table: Any,
    timeline: Timeline,
    report: ConversionReport,
) -> ClinicalRecords:
    """One subject's rows, *table*, as clinical records on *timeline*'s clock,
    noting in *report* what the mapping decided.  Every import reads a
    subject's rows here, so a cohort's subject is mapped as one alone is."""
    source = scan.source
    names, present = rules.names, scan.present
    systems, kinds_by = rules.systems, rules.kinds
    status, both_values, texts = rules.status, rules.both_values, rules.texts
    day_prefixes, documents_of, excluded = (
        rules.day_prefixes,
        rules.documents_of,
        rules.excluded,
    )
    own_system = scan.own_system
    if table.num_rows == 0:
        raise MEDH5ValidationError(
            f"{source} holds no row of MEDS subject {subject_id}"
        )
    report.decision(
        "subject",
        f"MEDS subject {subject_id} is this sample's subject: the mapping is "
        "the caller's, and MEDS's id is not written",
        {"meds_subject_id": subject_id, "rows": table.num_rows},
    )

    times, time_zone = _times(pa, table.column(names["time"]), names["time"])
    available: list[int | None] = [None] * table.num_rows
    available_zone = None
    if "available" in names:
        available, available_zone = _times(
            pa, table.column(names["available"]), names["available"]
        )

    def column(key: str) -> list[Any]:
        if key in names and names[key] in present:
            values: list[Any] = table.column(names[key]).to_pylist()
            return values
        return [None] * table.num_rows

    codes = column("code")
    numbers = column("numeric_value")
    if names["numeric_value"] in present and pa.types.is_float32(
        table.column(names["numeric_value"]).type
    ):
        # MEDS stores values as float32; each is written as the shortest
        # decimal that is that float32 (1.1, not 1.100000023841858), which
        # names the same stored value.
        numbers = [None if v is None else _float32(v) for v in numbers]
    words = column("text_value")
    units = column("unit") if "unit" in names else [None] * table.num_rows

    def place(value: int, zone: str | None) -> Bounds:
        return timeline.instant(value) if zone else timeline.wall(value)

    def whole_day(value: int, zone: str | None) -> Bounds:
        moment = wall_datetime(value)
        if zone:
            local = timeline.zone or zone_of(zone)
            moment = (
                datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
                + datetime.timedelta(microseconds=value)
            ).astimezone(local)
        return timeline.day(moment.date())

    events: list[Event] = []
    documents: list[Document] = []
    links: list[Link] = []
    seen: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for i in range(table.num_rows):
        code = codes[i]
        if code is None or code == "":
            counts["no_code"] += 1
            continue
        if code in excluded:
            skipped[code] += 1
            continue
        number = numbers[i]
        if number is not None and math.isnan(number):
            counts["nan"] += 1
            number = None
        word = words[i]
        identity = {
            "subject_id": subject_id,
            "time": times[i],
            "code": code,
            "numeric_value": number,
            "text_value": word,
        }
        # A declared extension column changes the event it is read into, so
        # it is part of the row's identity: otherwise two rows that differ
        # only there would take their ids in scan order.
        if "unit" in names:
            identity["unit"] = units[i]
        if "available" in names:
            identity["available"] = available[i]
        if number is not None and word is not None:
            if both_values == "refuse":
                raise MEDH5ValidationError(
                    f"a row of code {code!r} holds both a numeric_value "
                    f"({number}) and a text_value, and an event holds one "
                    "value (1.1 §5.1): pass both_values='numeric' or 'text' "
                    "to say which to keep"
                )
            counts["both"] += 1
            if both_values == "numeric":
                word = None
            else:
                number = None
        base = _row_id(identity)
        seen[base] += 1
        event_id = base if seen[base] == 1 else f"{base}.{seen[base]}"
        if seen[base] > 1:
            counts["repeat"] += 1

        under = _prefixed(code, systems)
        system, local = own_system, code
        if under is not None and code != under:
            system, local = systems[under], code[len(under) :].lstrip("/")
            if not local:
                system, local = own_system, code
        document = code in documents_of
        kind_prefix = _prefixed(code, kinds_by)
        kind = (
            "document"
            if document
            else (kinds_by[kind_prefix] if kind_prefix is not None else "other")
        )
        when = times[i]
        if when is None:
            temporal, start = "static", None
        elif _prefixed(code, day_prefixes) is not None:
            temporal, start = "point", whole_day(when, time_zone)
            counts["day"] += 1
        else:
            temporal, start = "point", place(when, time_zone)
        known = available[i]
        avail = None if known is None else place(known, available_zone)
        unit = units[i] if number is not None else None
        if units[i] is not None and number is None:
            counts["unit_without_value"] += 1
        if number is not None and unit is None:
            counts["no_unit"] += 1
        value_text = word
        if document:
            value_text = None
            if word is not None and texts:
                documents.append(Document(event_id, word))
                links.append(
                    Link.between(
                        ("event", event_id), "describes", ("document", event_id)
                    )
                )
            elif word is not None:
                counts["text_left_out"] += 1
            number = unit = None
        events.append(
            Event(
                event_id,
                event_id,
                kind,
                temporal,
                status,
                effective_start_us=start,
                available_us=avail,
                code_system=system,
                code=local,
                value_num=None if number is None else float(number),
                unit=unit,
                value_text=value_text,
            )
        )
    if not events:
        why = [f"{c}: {n}" for c, n in sorted(skipped.items())]
        if counts["no_code"]:
            why.append(f"without a code: {counts['no_code']}")
        raise MEDH5ValidationError(
            f"every row of MEDS subject {subject_id} was left out ({'; '.join(why)})"
        )
    _notes(report, skipped, counts, timeline, names, texts, len(events))
    records = ClinicalRecords.from_json(
        ClinicalRecords(
            timeline.clock, tuple(events), tuple(documents), tuple(links)
        ).to_json()
    )
    return records


def _notes(
    report: ConversionReport,
    skipped: Counter[str],
    counts: Counter[str],
    timeline: Timeline,
    names: Mapping[str, str],
    texts: bool,
    n_events: int,
) -> None:
    for code, n in sorted(skipped.items()):
        detail = {"code": code, "rows": n}
        if code == BIRTH:
            report.add(
                "excluded",
                f"{n} {BIRTH} row(s) left out: a birth instant identifies the "
                "subject (pass exclude_codes=() to keep it)",
                detail=detail,
            )
        else:
            report.add(
                "excluded", f"{n} row(s) of code {code!r} left out", detail=detail
            )
    if counts["no_code"]:
        report.warn(
            "code",
            f"{counts['no_code']} row(s) without a code left out",
            {"rows": counts["no_code"]},
        )
    if counts["nan"]:
        report.warn(
            "value",
            f"{counts['nan']} numeric_value(s) were NaN and are read as absent",
            {"rows": counts["nan"]},
        )
    if counts["both"]:
        report.decision(
            "value",
            f"{counts['both']} row(s) held both values; the one both_values "
            "names was kept",
            {"rows": counts["both"]},
        )
    if counts["repeat"]:
        report.decision(
            "id",
            f"{counts['repeat']} row(s) repeat an earlier row exactly; each "
            "repeat is its own event (.2, .3, ...)",
            {"rows": counts["repeat"]},
        )
    if counts["day"]:
        report.decision(
            "time",
            f"{counts['day']} time(s) known to the day span the whole day",
            {"rows": counts["day"]},
        )
    if timeline.uniform:
        report.decision(
            "time",
            "MEDS times have no zone and neither has the origin: times are "
            "counted from the origin as read, on one wall clock",
        )
    if "available" not in names:
        report.warn(
            "availability",
            f"MEDS records no availability, so all {n_events} event(s) have "
            "none: they can be a task's targets, never its inputs, until it is "
            "known (declare a column with columns={'available': ...})",
            {"events": n_events},
        )
    if counts["no_unit"]:
        report.add(
            "unit",
            f"{counts['no_unit']} numeric value(s) have no unit, and none is "
            "invented (W914): MEDS has no unit column; declare one with "
            "columns={'unit': ...}",
            detail={"values": counts["no_unit"]},
        )
    if counts["unit_without_value"]:
        report.add(
            "unit",
            f"{counts['unit_without_value']} unit(s) on rows with no numeric "
            "value left out: a unit belongs to a value_num",
            detail={"rows": counts["unit_without_value"]},
        )
    if counts["text_left_out"]:
        report.add(
            "text",
            f"{counts['text_left_out']} document text(s) left out: free text "
            "is imported only with texts=True",
            detail={"documents": counts["text_left_out"]},
        )


def augment_from_meds(
    path: str | os.PathLike[str],
    source: str | os.PathLike[str],
    *,
    out: str | os.PathLike[str] | None = None,
    **options: Any,
) -> tuple[dict[str, Any], ConversionReport]:
    """:func:`read_meds` *source* and add the records to the sample at *path*
    (:func:`medh5.clinical.augment`), in place or into *out*: the augment
    report and the conversion's."""
    records, report = read_meds(source, **options)
    result = augment(path, records, out=out)
    report.outputs.append(str(result["path"]))
    return result, report


def augment_cohort_from_meds(
    dataset: str | os.PathLike[str],
    mapping: Mapping[int, str | os.PathLike[str]],
    *,
    out_dir: str | os.PathLike[str] | None = None,
    clock: Clock | None = None,
    timezone: str | datetime.tzinfo | None = None,
    origin: str | datetime.datetime | None = None,
    code_system: str | None = None,
    code_systems: Mapping[str, str] | None = None,
    kinds: Mapping[str, str] | None = None,
    status: str = "unknown",
    day_precision: Iterable[str] = (),
    document_codes: Iterable[str] = (),
    exclude_codes: Iterable[str] = (BIRTH,),
    texts: bool = False,
    both_values: str = "refuse",
    columns: Mapping[str, str] | None = None,
) -> CohortImport:
    """Import every subject *mapping* names from one MEDS *dataset*, each into
    its sample, in one pass over the shards.

    *mapping* maps a MEDS ``subject_id`` to the sample its rows augment, in
    place or, with *out_dir*, into a file of the sample's name there.  Each
    subject's rows are mapped exactly as :func:`read_meds` maps them for that
    subject alone, onto *clock* or, when it is ``None``, onto each sample's
    own clinical clock, so a subject's sample is byte for byte the one
    :func:`augment_from_meds` writes: the batch changes the cost, never the
    answer.  The other options are :func:`read_meds`'s.

    The dataset is opened once and read twice, a Parquet row group at a
    time: its subject column, to count each mapped subject's rows, then the
    mapped subjects' rows, skipping the row groups whose statistics hold
    none.  A subject is imported as soon as its last row is read, so a
    dataset in MEDS order (sorted by subject) holds one row group and one
    subject's rows in memory at a time.
    Subjects the mapping does not name are counted in an ``info`` note and
    never read.  A subject that refuses --- no rows, a sample without a
    clock, ids already present --- is reported in the result, and the others
    are still written.
    """
    return _cohort_from_meds(
        dataset,
        mapping,
        out_dir=out_dir,
        clock=clock,
        timezone=timezone,
        origin=origin,
        code_system=code_system,
        code_systems=code_systems,
        kinds=kinds,
        status=status,
        day_precision=day_precision,
        document_codes=document_codes,
        exclude_codes=exclude_codes,
        texts=texts,
        both_values=both_values,
        columns=columns,
    )


def _cohort_from_meds(
    dataset: str | os.PathLike[str],
    mapping: Mapping[int, str | os.PathLike[str]],
    *,
    out_dir: str | os.PathLike[str] | None = None,
    clock: Clock | None = None,
    timezone: str | datetime.tzinfo | None = None,
    origin: str | datetime.datetime | None = None,
    code_system: str | None = None,
    code_systems: Mapping[str, str] | None = None,
    kinds: Mapping[str, str] | None = None,
    status: str = "unknown",
    day_precision: Iterable[str] = (),
    document_codes: Iterable[str] = (),
    exclude_codes: Iterable[str] = (BIRTH,),
    texts: bool = False,
    both_values: str = "refuse",
    columns: Mapping[str, str] | None = None,
    workers: int = 1,
) -> CohortImport:
    """:func:`augment_cohort_from_meds`, its subjects imported on *workers*
    processes: what ``medh5 clinical augment-cohort --workers`` runs.  The
    dataset is scanned here, once, and each subject's rows handed out as
    soon as they are all read; a worker reads the subject's sample and
    writes its result."""
    from medh5 import _workers

    pa = _pyarrow()
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    for key in mapping:
        if isinstance(key, bool) or not isinstance(key, int):
            raise MEDH5ValidationError(
                f"the mapping's key {key!r} is not a MEDS subject_id (an integer)"
            )
    targets = _clinical.cohort_targets(mapping, out_dir, dataset)
    rules = _Rules.of(
        code_systems=code_systems,
        kinds=kinds,
        status=status,
        day_precision=day_precision,
        document_codes=document_codes,
        exclude_codes=exclude_codes,
        texts=texts,
        both_values=both_values,
        columns=columns,
    )
    scan = _Scan.of(dataset, rules, code_system)
    cohort = CohortImport(
        source=scan.source,
        converter="from-meds",
        report=ConversionReport(source=scan.source, converter="from-meds"),
    )
    column = rules.names["subject_id"]

    # The subject column alone, to know when each subject's rows are all read.
    # The subjects it does not read are counted on disk, not in a set: a
    # dataset may hold millions of them.
    expected: Counter[int] = Counter()
    with _clinical.Tally() as others:
        for group in _row_groups(scan, [column], None):
            found = pc.value_counts(group.column(0))
            values = found.field("values").to_pylist()
            counts = found.field("counts").to_pylist()
            for value, n in zip(values, counts, strict=True):
                if value in mapping:
                    expected[value] += n
                else:
                    others.add(value, n)
        other_subjects, other_rows = others.counts()
    if other_subjects:
        cohort.report.add(
            "subject",
            f"{other_subjects} subject(s) of the dataset are not in the mapping, "
            f"and their {other_rows} row(s) were not read",
            detail={"subjects": other_subjects, "rows": other_rows},
        )

    places = {subject: (sample, out) for subject, sample, out in targets}
    results: dict[int, SubjectImport] = {}
    # What a worker's read needs of the scan: not the open dataset.
    scanned = replace(scan, dataset=None)

    def units() -> Iterator[tuple[str, tuple[Any, ...], list[Path]]]:
        pending: dict[int, list[Any]] = {}
        read_rows: Counter[int] = Counter()
        if not expected:
            return
        for group in _row_groups(
            scan, scan.wanted, ds.field(column).isin(sorted(expected))
        ):
            ids = group.column(column)
            for subject in pc.unique(ids).to_pylist():
                rows = group.filter(pc.equal(ids, subject))
                pending.setdefault(subject, []).append(rows)
                read_rows[subject] += rows.num_rows
                if read_rows[subject] == expected[subject]:
                    sample, out = places[subject]
                    table = pa.concat_tables(pending.pop(subject))
                    task = (
                        subject,
                        sample,
                        out,
                        clock,
                        table,
                        rules,
                        scanned,
                        timezone,
                        origin,
                    )
                    yield str(subject), task, [sample, *([out] if out else [])]

    for result in _workers.run(
        _import_subject, units(), workers, _clinical.subject_died
    ):
        results[cast(int, result.subject)] = result
    for subject, sample, out in targets:
        if subject not in results:
            # No row of it: refused as read_meds refuses it.
            def none(on: Clock, subject: int = subject) -> Any:
                raise MEDH5ValidationError(
                    f"{scan.source} holds no row of MEDS subject {subject}"
                )

            results[subject] = _clinical.import_subject(
                subject, sample, out, clock, none
            )
        cohort.subjects.append(results[subject])
    return cohort


def _import_subject(task: tuple[Any, ...]) -> SubjectImport:
    """One subject of :func:`_cohort_from_meds`, its rows as read."""
    subject, sample, out, clock, table, rules, scan, timezone, origin = task
    pa = _pyarrow()

    def read(on: Clock) -> tuple[ClinicalRecords, ConversionReport]:
        report = ConversionReport(source=scan.source, converter="from-meds")
        timeline = Timeline(on, timezone=timezone, origin=origin, converter="from-meds")
        records = _records(pa, rules, scan, subject, table, timeline, report)
        return records, report

    return _clinical.import_subject(subject, sample, out, clock, read)


def _row_groups(scan: _Scan, columns: list[str], where: Any) -> Iterable[Any]:
    """The dataset's rows a Parquet row group at a time, in the order a
    filtered ``to_table`` returns them, each group read only when reached:
    *where* skips the groups whose statistics rule them out."""
    schema = scan.dataset.schema
    for fragment in scan.dataset.get_fragments(filter=where):
        for group in fragment.split_by_row_group(where, schema=schema):
            yield group.to_table(
                schema=schema, columns=columns, filter=where, use_threads=False
            )


__all__ = [
    "BIRTH",
    "BOTH_VALUES",
    "COLUMNS",
    "augment_cohort_from_meds",
    "augment_from_meds",
    "read_meds",
]
