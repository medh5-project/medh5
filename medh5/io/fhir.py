"""FHIR R4 import: one patient's Observations and DiagnosticReports as clinical
records (1.1 §5, §10).

FHIR says more about time than most sources: when a result held
(``effective[x]``, to the precision it was recorded), when it was issued
(``issued``, its availability), which version it is (``meta.versionId``) and
why an expected value is missing (``dataAbsentReason``).  :func:`read_fhir`
keeps each of those and invents none of what FHIR leaves out:

* **Ids.** ``record_id`` is ``obs.<id>`` or ``dr.<id>``.  Several versions of
  one resource in a Bundle are versions of one record: ``event_id`` adds
  ``.v<versionId>``, and each version ``supersedes`` the one before.
* **Status.** ``preliminary``, ``final``, ``cancelled`` and ``unknown`` are
  kept; ``partial`` is ``preliminary``; ``amended``, ``corrected`` and
  ``appended`` are ``amended``; ``entered-in-error`` is ``entered_in_error``;
  a ``registered`` resource holds no result yet and is left out.
* **Time.** A date known to the year, month or day spans every instant of
  it; a date with no known zone spans every instant it covers anywhere
  (1.1 §5.2).  A ``Period`` is an ``interval``, with an open end left null;
  a ``Timing``, or no effective time, is ``unknown``.  ``issued`` is the
  availability, and null when absent.
* **Values.** A ``Quantity`` is ``value_num`` with its UCUM code (else its
  unit string) and its comparator (``<5`` is ``lt 5``); a string, a boolean
  or a concept's chosen code is ``value_text``; ``dataAbsentReason`` is
  ``missing_reason``, with no value.  Each ``component`` is an event of its
  own, sharing its resource's times and status.
* **Codes.** The first coding in ``prefer_systems`` order, else the first,
  names what was measured.  A resource with no coding is left out: no
  standard code is guessed from ``code.text``.
* **Left out unless asked.** A report's text (``texts=True``), and the
  imaging studies a report describes, which link to the sample's images only
  through ``image_studies``.

What could only be kept by a later format minor --- ``interpretation``,
``referenceRange``, a report's ``result`` references --- is counted in the
report.  Standard library only: a Bundle, a single resource or a list of
resources, as JSON.
"""

from __future__ import annotations

import base64
import binascii
import datetime
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from medh5.clinical import (
    Bounds,
    ClinicalRecords,
    Clock,
    Document,
    Event,
    Link,
    augment,
    utc_us,
)
from medh5.errors import MEDH5ValidationError
from medh5.io import _clinical
from medh5.io._clinical import Timeline, option_map, valid_id, wall_us
from medh5.io.report import CohortImport, ConversionReport

UCUM = "http://unitsofmeasure.org"

STATUSES: dict[str, str] = {
    "preliminary": "preliminary",
    "final": "final",
    "cancelled": "cancelled",
    "unknown": "unknown",
    "partial": "preliminary",
    "amended": "amended",
    "corrected": "amended",
    "appended": "amended",
    "entered-in-error": "entered_in_error",
}
"""FHIR R4 Observation and DiagnosticReport status -> event status
(``registered`` is left out)."""

COMPARATORS: dict[str, str] = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
"""``Quantity.comparator`` -> ``value_comparator`` (``ad`` has none)."""

_KINDS = {"Observation": ("obs", "observation"), "DiagnosticReport": ("dr", "document")}

_DATE_TIME = re.compile(
    r"(?P<year>\d{4})(?:-(?P<month>\d{2})(?:-(?P<day>\d{2})"
    r"(?:T(?P<time>\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)"
    r"(?P<offset>Z|[+-]\d{2}:\d{2})?)?)?)?"
)


def _load(source: Any) -> list[Any]:
    """The resources in *source*: a path to JSON, or JSON already read."""
    if isinstance(source, (str, os.PathLike)):
        path = Path(os.fspath(source))
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise MEDH5ValidationError(f"{path} is not JSON: {exc}") from None
    else:
        doc = source
    if isinstance(doc, Mapping) and doc.get("resourceType") == "Bundle":
        return [e.get("resource") for e in doc.get("entry", []) if e.get("resource")]
    if isinstance(doc, Mapping):
        return [doc]
    if isinstance(doc, Sequence) and not isinstance(doc, (str, bytes)):
        return list(doc)
    raise MEDH5ValidationError(
        "a FHIR source is a Bundle, a resource or a list of resources"
    )


class _Times:
    """FHIR date, dateTime and instant values, on the sample's clock."""

    def __init__(self, timeline: Timeline) -> None:
        self.timeline = timeline

    def bounds(self, value: str, what: str) -> Bounds:
        found = _DATE_TIME.fullmatch(value)
        if found is None:
            raise MEDH5ValidationError(f"{what} {value!r} is not a FHIR date-time")
        try:
            return self._bounds(found)
        except MEDH5ValidationError:
            raise
        except ValueError:
            raise MEDH5ValidationError(f"{what} {value!r} is not a date") from None

    def _bounds(self, found: re.Match[str]) -> Bounds:
        year = int(found["year"])
        month, day = found["month"], found["day"]
        if found["time"] is None:
            if day is not None:
                return self.timeline.day(datetime.date(year, int(month), int(day)))
            if month is not None:
                first = datetime.date(year, int(month), 1)
                following = (
                    datetime.date(year + 1, 1, 1)
                    if first.month == 12
                    else datetime.date(year, first.month + 1, 1)
                )
            else:
                first = datetime.date(year, 1, 1)
                following = datetime.date(year + 1, 1, 1)
            return self.timeline.period(_midnight(first), _midnight(following))
        clock, _, fraction = found["time"].partition(".")
        hour, minute, *second = (int(p) for p in clock.split(":"))
        if len(fraction) > 6 and int(fraction[6:]):
            raise ValueError("finer than a microsecond")
        moment = datetime.datetime(
            year,
            int(month),
            int(day),
            hour,
            minute,
            second[0] if second else 0,
            int(fraction[:6].ljust(6, "0")) if fraction else 0,
        )
        if found["offset"] is None:
            return self.timeline.wall(wall_us(moment))
        sign, hours, minutes = 1, 0, 0
        if found["offset"] != "Z":
            sign = -1 if found["offset"][0] == "-" else 1
            hours, minutes = (int(p) for p in found["offset"][1:].split(":"))
        offset = datetime.timedelta(hours=hours, minutes=minutes) * sign
        aware = moment.replace(tzinfo=datetime.timezone(offset))
        return self.timeline.instant(utc_us(aware))


def _midnight(day: datetime.date) -> int:
    return wall_us(datetime.datetime.combine(day, datetime.time()))


def _coding(concept: Any, prefer: Sequence[str]) -> Mapping[str, Any] | None:
    """The coding of *concept* naming a system and a code: the first in
    *prefer*'s order, else the first."""
    codings = [
        c
        for c in (concept or {}).get("coding", [])
        if isinstance(c, Mapping) and c.get("system") and c.get("code")
    ]
    for system in prefer:
        for c in codings:
            if c["system"] == system:
                return c
    return codings[0] if codings else None


class _Reader:
    def __init__(
        self,
        report: ConversionReport,
        times: _Times,
        prefer: Sequence[str],
        texts: bool,
        image_studies: Mapping[str, str],
    ) -> None:
        self.report = report
        self.times = times
        self.prefer = prefer
        self.texts = texts
        self.image_studies = image_studies
        self.counts: Counter[str] = Counter()
        self.events: list[Event] = []
        self.documents: list[Document] = []
        self.links: list[Link] = []

    def warn(self, what: str, message: str) -> None:
        self.report.warn(what, message)

    def dropped(self, item: Mapping[str, Any]) -> None:
        """Count what *item* holds that no event column can."""
        for name in ("interpretation", "referenceRange", "note", "method"):
            if item.get(name):
                self.counts[f"dropped:{name}"] += 1

    def effective(self, resource: Mapping[str, Any], name: str) -> dict[str, Any]:
        """``temporal_type`` and bounds from ``effective[x]``."""
        for key in ("effectiveDateTime", "effectiveInstant"):
            if key in resource:
                start = self.times.bounds(str(resource[key]), f"{name} {key}")
                return {"temporal_type": "point", "effective_start_us": start}
        if "effectivePeriod" in resource:
            period = resource["effectivePeriod"] or {}
            if not period.get("start"):
                self.warn(
                    "time",
                    f"{name}: effectivePeriod has no start, so its time is unknown",
                )
                return {"temporal_type": "unknown"}
            end = period.get("end")
            return {
                "temporal_type": "interval",
                "effective_start_us": self.times.bounds(
                    str(period["start"]), f"{name} effectivePeriod.start"
                ),
                "effective_end_us": (
                    None
                    if not end
                    else self.times.bounds(str(end), f"{name} effectivePeriod.end")
                ),
            }
        if "effectiveTiming" in resource:
            self.warn(
                "time",
                f"{name}: effectiveTiming is a schedule, not a time; the time "
                "is unknown",
            )
        else:
            self.counts["no_time"] += 1
        return {"temporal_type": "unknown"}

    def value(self, item: Mapping[str, Any], name: str) -> dict[str, Any] | None:
        """The value columns of an Observation or a component; ``None`` when
        its value cannot be kept, and the item is left out."""
        absent = item.get("dataAbsentReason")
        keys = [k for k in item if k.startswith("value")]
        if not keys:
            if absent is None:
                return {}
            reason = _coding(absent, ())
            if reason is None:
                self.warn(
                    "value",
                    f"{name}: its dataAbsentReason has no coded reason, so the "
                    "reason is not kept",
                )
                return {}
            return {"missing_reason": str(reason["code"])}
        (key,) = keys
        value = item[key]
        if key == "valueQuantity":
            if not isinstance(value, Mapping) or value.get("value") is None:
                self.warn("value", f"{name}: a Quantity with no value is left out")
                return None
            comparator = value.get("comparator")
            if comparator is not None and comparator not in COMPARATORS:
                self.warn(
                    "value",
                    f"{name}: comparator {comparator!r} has no event form and is "
                    "left out",
                )
                return None
            unit = (
                value.get("code")
                if value.get("system") == UCUM and value.get("code")
                else value.get("unit")
            )
            return {
                "value_num": float(value["value"]),
                "value_comparator": None
                if comparator is None
                else COMPARATORS[comparator],
                "unit": str(unit) if unit else None,
            }
        if key == "valueInteger":
            return {"value_num": float(value)}
        if key == "valueString":
            return {"value_text": str(value)}
        if key == "valueBoolean":
            return {"value_text": "true" if value else "false"}
        if key == "valueCodeableConcept":
            coding = _coding(value, self.prefer)
            if coding is None:
                self.warn(
                    "value", f"{name}: a concept value with no coding is left out"
                )
                return None
            self.counts["concept"] += 1
            return {"value_text": str(coding["code"])}
        self.warn("value", f"{name}: {key} has no event form and is left out")
        return None

    def read(self, kind: str, versions: list[Mapping[str, Any]]) -> None:
        """Every version of one resource, oldest first."""
        prefix, event_kind = _KINDS[kind]
        rid = versions[0]["id"]
        record = f"{prefix}.{rid}"
        name = f"{kind}/{rid}"
        chains: dict[str, list[str]] = defaultdict(list)
        components: set[int] = set()
        for resource in versions:
            version = (resource.get("meta") or {}).get("versionId")
            suffix = f".v{version}" if len(versions) > 1 else ""
            label = f"{name}/_history/{version}" if suffix else name
            status = STATUSES.get(str(resource.get("status")))
            if resource.get("status") == "registered":
                self.counts["registered"] += 1
                continue
            if status is None:
                self.warn(
                    "status",
                    f"{label}: status {resource.get('status')!r} is not FHIR "
                    "R4's; left out",
                )
                continue
            if resource.get("status") in ("corrected", "appended"):
                self.counts[f"status:{resource['status']}"] += 1
            coding = _coding(resource.get("code"), self.prefer)
            if coding is None:
                self.warn(
                    "code",
                    f"{label}: its code has no coding with a system; left out "
                    "(no code is guessed from its text)",
                )
                continue
            shared: dict[str, Any] = {
                "kind": event_kind,
                "status": status,
                **self.effective(resource, label),
            }
            issued = resource.get("issued")
            shared["available_us"] = (
                None
                if not issued
                else self.times.bounds(str(issued), f"{label} issued")
            )
            if not issued:
                self.counts["no_issued"] += 1
            encounter = (resource.get("encounter") or {}).get("reference")
            shared["encounter_id"] = str(encounter) if encounter else None
            self.dropped(resource)
            event_id = record + suffix
            if kind == "DiagnosticReport":
                self.add(event_id, record, coding, shared, {}, chains)
                self.report_parts(resource, event_id, label)
                continue
            # A parent value with no event form leaves the parent out, and
            # not its components: each is an event of its own.
            found = self.value(resource, label)
            if found is not None:
                self.add(event_id, record, coding, shared, found, chains)
            parts = resource.get("component") or []
            if components and len(parts) != len(components):
                self.warn(
                    "version",
                    f"{label} has {len(parts)} component(s) where an earlier "
                    f"version had {len(components)}",
                )
            for i, part in enumerate(parts):
                components.add(i)
                part_label = f"{label} component[{i}]"
                part_coding = _coding(part.get("code"), self.prefer)
                if part_coding is None:
                    self.warn(
                        "code",
                        f"{part_label}: its code has no coding with a system; left out",
                    )
                    continue
                self.dropped(part)
                part_values = self.value(part, part_label)
                if part_values is None:
                    continue
                self.add(
                    f"{record}.c{i}{suffix}",
                    f"{record}.c{i}",
                    part_coding,
                    shared,
                    part_values,
                    chains,
                )
        for ids in chains.values():
            for older, newer in zip(ids, ids[1:], strict=False):
                self.links.append(
                    Link.between(("event", newer), "supersedes", ("event", older))
                )

    def add(
        self,
        event_id: str,
        record: str,
        coding: Mapping[str, Any],
        shared: Mapping[str, Any],
        values: Mapping[str, Any],
        chains: dict[str, list[str]],
    ) -> None:
        for identifier in (event_id, record):
            if not valid_id(identifier):
                raise MEDH5ValidationError(
                    f"{identifier!r} is not an event id ([A-Za-z0-9_.-], at most "
                    "128): the FHIR id or versionId it is made from cannot be kept"
                )
        version = coding.get("version")
        self.events.append(
            Event(
                event_id,
                record,
                code_system=str(coding["system"]),
                code=str(coding["code"]),
                code_version=str(version) if version else None,
                **shared,
                **values,
            )
        )
        chains[record].append(event_id)

    def report_parts(
        self, resource: Mapping[str, Any], event_id: str, label: str
    ) -> None:
        """A DiagnosticReport's text and imaging studies."""
        text = None
        for form in resource.get("presentedForm") or []:
            if str(form.get("contentType", "")).split(";")[0].strip() == "text/plain":
                try:
                    text = base64.b64decode(form.get("data", ""), validate=True).decode(
                        "utf-8"
                    )
                except (binascii.Error, UnicodeDecodeError):
                    raise MEDH5ValidationError(
                        f"{label}: its text/plain presentedForm is not base64 UTF-8"
                    ) from None
                break
        if text is None and resource.get("conclusion"):
            text = str(resource["conclusion"])
        if text is not None:
            if self.texts:
                self.documents.append(Document(event_id, text))
                self.links.append(
                    Link.between(
                        ("event", event_id), "describes", ("document", event_id)
                    )
                )
            else:
                self.counts["text_left_out"] += 1
        for study in resource.get("imagingStudy") or []:
            reference = (study or {}).get("reference")
            image = self.image_studies.get(str(reference))
            if image is None:
                self.counts["study_unlinked"] += 1
                continue
            self.links.append(
                Link.between(("event", event_id), "describes", ("image", image))
            )
        if resource.get("result"):
            self.counts["result"] += len(resource["result"])


def _order(
    kind: str, rid: str, versions: list[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """The versions of one resource, oldest first; refused when they cannot
    be told apart or ordered."""
    if len(versions) == 1:
        return versions
    name = f"{kind}/{rid}"
    by_version: dict[str, Mapping[str, Any]] = {}
    for resource in versions:
        version = (resource.get("meta") or {}).get("versionId")
        if not version:
            raise MEDH5ValidationError(
                f"{name} appears {len(versions)} times, and a version has no "
                "meta.versionId: they cannot be told apart"
            )
        held = by_version.setdefault(str(version), resource)
        if held is not resource and held != resource:
            raise MEDH5ValidationError(
                f"{name} has two different resources as version {version}"
            )
    unique = list(by_version.items())
    numbers = {int(v) for v, _ in unique if v.isascii() and v.isdigit()}
    if len(numbers) == len(unique):
        # Distinct integers only: "1" and "01" are two versionIds with one
        # number, and Bundle order is not a revision order.
        return [r for _, r in sorted(unique, key=lambda item: int(item[0]))]
    stamps = [(r.get("meta") or {}).get("lastUpdated") for _, r in unique]
    if all(stamps):
        times = _Times(
            Timeline(Clock.utc("order"), timezone=None, origin=None, converter="")
        )
        try:
            moments = [times.bounds(str(s), f"{name} meta.lastUpdated") for s in stamps]
        except MEDH5ValidationError:
            moments = []
        if (
            moments
            and all(lo == hi for lo, hi in moments)
            and len(set(moments)) == len(moments)
        ):
            return [
                r
                for _, (_, r) in sorted(
                    zip(moments, unique, strict=True), key=lambda p: p[0]
                )
            ]
    raise MEDH5ValidationError(
        f"{name}'s versions {', '.join(v for v, _ in unique)} have no order "
        "(numeric versionIds or distinct meta.lastUpdated): a revision order "
        "is not invented"
    )


def read_fhir(
    source: str | os.PathLike[str] | Mapping[str, Any] | Sequence[Any],
    *,
    subject: str,
    clock: Clock,
    timezone: str | datetime.tzinfo | None = None,
    origin: str | datetime.datetime | None = None,
    texts: bool = False,
    prefer_systems: Iterable[str] = (),
    image_studies: Mapping[str, str] | None = None,
) -> tuple[ClinicalRecords, ConversionReport]:
    """The Observations and DiagnosticReports of *subject* (a reference, such
    as ``Patient/123``) in *source*, as clinical records on *clock*, and what
    the conversion decided.

    ``timezone`` reads a time recorded without an offset, and a relative
    clock needs ``origin``.  ``image_studies`` maps an ``ImagingStudy``
    reference to the id of the sample's image it is.  Refused, as
    ``MEDH5ValidationError``: no resource for the subject, a time without an
    offset and no ``timezone``, a relative clock without ``origin``, versions
    of a resource that cannot be ordered.
    """
    report = ConversionReport(
        source=os.fspath(source)
        if isinstance(source, (str, os.PathLike))
        else "<json>",
        converter="from-fhir",
    )
    timeline = Timeline(clock, timezone=timezone, origin=origin, converter="from-fhir")
    reader = _Reader(
        report,
        _Times(timeline),
        tuple(prefer_systems),
        texts,
        option_map(image_studies, "image_studies"),
    )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    others: Counter[str] = Counter()
    elsewhere = 0
    for resource in _load(source):
        if not isinstance(resource, Mapping):
            raise MEDH5ValidationError("a FHIR resource is a JSON object")
        kind = str(resource.get("resourceType"))
        if kind not in _KINDS:
            others[kind] += 1
            continue
        if _subject_of(resource, f"a {kind}") != subject:
            elsewhere += 1
            continue
        if not resource.get("id"):
            report.warn("id", f"a {kind} with no id is left out: it names no record")
            continue
        grouped[(kind, str(resource["id"]))].append(resource)
    if not grouped:
        raise MEDH5ValidationError(
            f"the FHIR source holds no Observation or DiagnosticReport of {subject}"
        )
    report.decision(
        "subject",
        f"{subject} is this sample's subject: the mapping is the caller's, and "
        "the reference is not written",
        {"subject": subject, "resources": sum(len(v) for v in grouped.values())},
    )
    for (kind, rid), versions in grouped.items():
        reader.read(kind, _order(kind, rid, versions))
    if not reader.events:
        raise MEDH5ValidationError(
            f"every Observation and DiagnosticReport of {subject} was left out "
            "(see the warnings)"
        )
    _notes(report, reader, others, elsewhere, timeline)
    records = ClinicalRecords.from_json(
        ClinicalRecords(
            clock, tuple(reader.events), tuple(reader.documents), tuple(reader.links)
        ).to_json()
    )
    return records, report


def _notes(
    report: ConversionReport,
    reader: _Reader,
    others: Counter[str],
    elsewhere: int,
    timeline: Timeline,
) -> None:
    counts = reader.counts
    if others:
        listed = ", ".join(f"{k}: {n}" for k, n in sorted(others.items()))
        report.add(
            "resource",
            f"resources of other types left out ({listed})",
            detail=dict(others),
        )
    if elsewhere:
        report.add(
            "subject",
            f"{elsewhere} Observation(s) and DiagnosticReport(s) of other "
            "subjects left out",
            detail={"resources": elsewhere},
        )
    if counts["registered"]:
        report.add(
            "status",
            f"{counts['registered']} registered resource(s) left out: they hold "
            "no result yet",
            detail={"resources": counts["registered"]},
        )
    amended = {k.split(":")[1]: n for k, n in counts.items() if k.startswith("status:")}
    if amended:
        report.decision(
            "status",
            "FHIR's corrected and appended are the event status amended "
            f"({', '.join(f'{k}: {n}' for k, n in sorted(amended.items()))})",
            amended,
        )
    if counts["no_time"]:
        report.add(
            "time",
            f"{counts['no_time']} resource(s) give no effective time; it is unknown",
            detail={"resources": counts["no_time"]},
        )
    if counts["no_issued"]:
        report.add(
            "availability",
            f"{counts['no_issued']} resource(s) have no issued time, so their "
            "availability stays unknown: a task's targets, never its inputs",
            detail={"resources": counts["no_issued"]},
        )
    if timeline.uniform:
        report.decision(
            "time",
            "times without an offset and an origin without one are counted on "
            "one wall clock",
        )
    if counts["concept"]:
        report.decision(
            "value",
            f"{counts['concept']} concept value(s) kept as the chosen coding's "
            "code; its system is not kept",
            {"values": counts["concept"]},
        )
    dropped = {
        k.split(":")[1]: n for k, n in counts.items() if k.startswith("dropped:")
    }
    if dropped:
        report.add(
            "dropped",
            "elements with no event form left out --- only a later format "
            "minor could keep them: "
            + ", ".join(f"{k} ({n})" for k, n in sorted(dropped.items())),
            detail=dropped,
        )
    if counts["text_left_out"]:
        report.add(
            "text",
            f"{counts['text_left_out']} report text(s) left out: free text is "
            "imported only with texts=True",
            detail={"documents": counts["text_left_out"]},
        )
    if counts["study_unlinked"]:
        report.add(
            "link",
            f"{counts['study_unlinked']} imaging study reference(s) not linked: "
            "a study links to an image only through image_studies",
            detail={"references": counts["study_unlinked"]},
        )
    if counts["result"]:
        report.add(
            "link",
            f"{counts['result']} report result reference(s) not linked: no "
            "relation of this profile says a report holds a result",
            detail={"references": counts["result"]},
        )


def augment_from_fhir(
    path: str | os.PathLike[str],
    source: str | os.PathLike[str] | Mapping[str, Any] | Sequence[Any],
    *,
    out: str | os.PathLike[str] | None = None,
    **options: Any,
) -> tuple[dict[str, Any], ConversionReport]:
    """:func:`read_fhir` *source* and add the records to the sample at *path*
    (:func:`medh5.clinical.augment`), in place or into *out*: the augment
    report and the conversion's."""
    records, report = read_fhir(source, **options)
    result = augment(path, records, out=out)
    report.outputs.append(str(result["path"]))
    return result, report


def _subject_of(resource: Mapping[str, Any], what: str) -> Any:
    """The reference of *resource*'s ``subject``, or ``None`` without one; a
    ``subject`` that is not a Reference (an object) is refused."""
    found = resource.get("subject")
    if not found:
        return None
    if not isinstance(found, Mapping):
        raise MEDH5ValidationError(
            f"{what} has a subject that is not a FHIR Reference: {found!r}"
        )
    return found.get("reference")


def _bulk_files(export_dir: Path) -> list[Path]:
    """The NDJSON files of a FHIR Bulk Data export, in name order."""
    if not export_dir.is_dir():
        raise MEDH5ValidationError(f"{export_dir} is not a FHIR bulk export directory")
    found = sorted(export_dir.glob("*.ndjson"))
    if not found:
        raise MEDH5ValidationError(f"{export_dir} holds no FHIR bulk file (*.ndjson)")
    return found


def augment_cohort_from_fhir(
    export_dir: str | os.PathLike[str],
    mapping: Mapping[str, str | os.PathLike[str]],
    *,
    out_dir: str | os.PathLike[str] | None = None,
    clock: Clock | None = None,
    timezone: str | datetime.tzinfo | None = None,
    origin: str | datetime.datetime | None = None,
    texts: bool = False,
    prefer_systems: Iterable[str] = (),
    image_studies: Mapping[str, str] | None = None,
) -> CohortImport:
    """Import every subject *mapping* names from a FHIR Bulk Data export, each
    into its sample.

    *export_dir* holds the export's NDJSON files (``Observation.ndjson``,
    ``DiagnosticReport.ndjson``, …: one resource per line, many patients);
    *mapping* maps a subject reference (``Patient/123``) to the sample its
    resources augment, in place or, with *out_dir*, into a file of the
    sample's name there.  Each subject's Observations and DiagnosticReports,
    in file and line order, are mapped exactly as :func:`read_fhir` maps a
    Bundle of them, onto *clock* or, when it is ``None``, onto each sample's
    own clinical clock: the batch changes the cost, never the answer.  The
    other options are :func:`read_fhir`'s.

    The files are streamed twice: once to find where each mapped subject's
    resources are (a line's offset, not the resource), then once per subject
    to read them, so one subject's resources are in memory at a time.  The
    index of where they are, and of the subjects not read, is kept in a
    temporary database on disk, so memory does not grow with the export.
    Resources of subjects the mapping does not name, and resource types the
    profile does not read, are counted in ``info`` notes.  A subject that
    refuses is reported in the result, and the others are still written.
    """
    return _cohort_from_fhir(
        export_dir,
        mapping,
        out_dir=out_dir,
        clock=clock,
        timezone=timezone,
        origin=origin,
        texts=texts,
        prefer_systems=prefer_systems,
        image_studies=image_studies,
    )


def _cohort_from_fhir(
    export_dir: str | os.PathLike[str],
    mapping: Mapping[str, str | os.PathLike[str]],
    *,
    out_dir: str | os.PathLike[str] | None = None,
    clock: Clock | None = None,
    timezone: str | datetime.tzinfo | None = None,
    origin: str | datetime.datetime | None = None,
    texts: bool = False,
    prefer_systems: Iterable[str] = (),
    image_studies: Mapping[str, str] | None = None,
    workers: int = 1,
) -> CohortImport:
    """:func:`augment_cohort_from_fhir`, its subjects imported on *workers*
    processes: what ``medh5 clinical augment-cohort --workers`` runs.  The
    export is indexed here, once, and each subject's resources read here as
    a worker is free for them; a worker reads the subject's sample and
    writes its result."""
    from medh5 import _workers

    root = Path(os.fspath(export_dir))
    files = _bulk_files(root)
    targets = _clinical.cohort_targets(mapping, out_dir, root)
    # One tuple for every subject: a generator would serve only the first.
    prefer = tuple(prefer_systems)
    source = os.fspath(export_dir)
    cohort = CohortImport(
        source=source,
        converter="from-fhir",
        report=ConversionReport(source=source, converter="from-fhir"),
    )
    options = (timezone, origin, texts, prefer, image_studies, source)
    with _BulkIndex(mapping) as index:
        for n, path in enumerate(files):
            index.scan(n, path)
        index.notes(cohort.report)

        def units() -> Iterator[tuple[str, tuple[Any, ...]]]:
            for subject, sample, out in targets:
                try:
                    found: Any = list(_bulk_read(files, index.lines(subject)))
                except Exception as exc:  # raised in the subject's read
                    found = exc
                yield str(subject), (subject, sample, out, clock, found, options)

        cohort.subjects.extend(
            _workers.run(_import_subject, units(), workers, _clinical.subject_died)
        )
    return cohort


def _import_subject(task: tuple[Any, ...]) -> Any:
    """One subject of :func:`_cohort_from_fhir`, its resources as read."""
    subject, sample, out, clock, found, options = task
    timezone, origin, texts, prefer, image_studies, source = options

    def read(on: Clock) -> tuple[ClinicalRecords, ConversionReport]:
        if isinstance(found, BaseException):
            raise found
        records, report = read_fhir(
            found,
            subject=subject,
            clock=on,
            timezone=timezone,
            origin=origin,
            texts=texts,
            prefer_systems=prefer,
            image_studies=image_studies,
        )
        report.source = source
        return records, report

    return _clinical.import_subject(subject, sample, out, clock, read)


class _BulkIndex:
    """Where each mapped subject's lines are in a bulk export, and how many
    resources of each subject the mapping does not name: in a temporary
    database on disk (SQLite's, removed when it closes), so that what is held
    in memory is the mapping and a bounded buffer, never a row per resource
    or per subject of the export."""

    BUFFER = 1_000
    """Rows gathered before they are written to the database."""

    def __init__(self, subjects: Iterable[str]) -> None:
        self._subjects = {subject: n for n, subject in enumerate(subjects)}
        self._db = sqlite3.connect("")
        self._db.executescript(
            "PRAGMA cache_size = -2048; PRAGMA temp_store = FILE;"
            "CREATE TABLE line (subject INTEGER, file INTEGER, start INTEGER,"
            " length INTEGER);"
        )
        self._lines: list[tuple[int, int, int, int]] = []
        self._elsewhere = _clinical.Tally()
        self._others: Counter[str] = Counter()
        self._indexed = False

    def __enter__(self) -> _BulkIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self._db.close()
        self._elsewhere.__exit__()

    def scan(self, number: int, path: Path) -> None:
        """Index the lines of *path*, the export's file *number*."""
        with path.open("rb") as stream:
            offset = 0
            for line_number, line in enumerate(stream, 1):
                start, offset = offset, offset + len(line)
                if not line.strip():
                    continue
                try:
                    resource = json.loads(line)
                except ValueError as exc:
                    raise MEDH5ValidationError(
                        f"line {line_number} of {path} is not JSON: {exc}"
                    ) from None
                if not isinstance(resource, Mapping):
                    raise MEDH5ValidationError(
                        f"line {line_number} of {path} is not a FHIR resource"
                    )
                kind = str(resource.get("resourceType"))
                if kind not in _KINDS:
                    self._others[kind] += 1
                    continue
                reference = _subject_of(
                    resource, f"the {kind} on line {line_number} of {path}"
                )
                subject = (
                    self._subjects.get(reference)
                    if isinstance(reference, str)
                    else None
                )
                if subject is not None:
                    self._lines.append((subject, number, start, len(line)))
                    if len(self._lines) >= self.BUFFER:
                        self._flush()
                else:
                    self._elsewhere.add(str(reference))

    def _flush(self) -> None:
        self._db.executemany("INSERT INTO line VALUES (?, ?, ?, ?)", self._lines)
        self._lines.clear()

    def notes(self, report: ConversionReport) -> None:
        """Count, in *report*, what the export holds that is not read."""
        _bulk_notes(report, self._others, *self._elsewhere.counts())

    def lines(self, subject: str) -> Iterator[tuple[int, int, int]]:
        """*subject*'s lines, as (file, offset, length), in file and line
        order."""
        if not self._indexed:
            self._flush()
            self._db.execute("CREATE INDEX line_subject ON line (subject)")
            self._indexed = True
        return self._db.execute(
            "SELECT file, start, length FROM line WHERE subject = ? ORDER BY rowid",
            (self._subjects[subject],),
        )


def _bulk_read(
    files: Sequence[Path], where: Iterable[tuple[int, int, int]]
) -> Iterable[Any]:
    """The resources at *where*: (file, offset, length) of each line."""
    streams: dict[int, Any] = {}
    try:
        for index, start, length in where:
            stream = streams.get(index)
            if stream is None:
                stream = streams[index] = files[index].open("rb")
            stream.seek(start)
            yield json.loads(stream.read(length))
    finally:
        for stream in streams.values():
            stream.close()


def _bulk_notes(
    report: ConversionReport, others: Counter[str], subjects: int, resources: int
) -> None:
    if resources:
        report.add(
            "subject",
            f"{resources} Observation(s) and DiagnosticReport(s) "
            f"of {subjects} subject(s) not in the mapping were not read",
            detail={"subjects": subjects, "resources": resources},
        )
    for kind, n in sorted(others.items()):
        report.add(
            "resource",
            f"{n} {kind} resource(s) left out: this profile reads Observations "
            "and DiagnosticReports",
            detail={"resourceType": kind, "resources": n},
        )


__all__ = [
    "COMPARATORS",
    "STATUSES",
    "UCUM",
    "augment_cohort_from_fhir",
    "augment_from_fhir",
    "read_fhir",
]
