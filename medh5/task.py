"""Task manifests (``medh5.task/1``): which rows, at which cutoffs, see what.

A ``.medh5`` file is the patient source.  A *task* is a separate, versioned
JSON document (``docs/spec/task-cache-1.md``) naming

- the **sources** of each subject --- files or collection members, each pinned
  to the ``content_id`` it was built from (a URI says where bytes are; the pin
  says which bytes count);
- the **partition** of each subject, assigned before any row exists, so no
  window puts one patient on both sides of a split;
- the **rows**: a subject at a cutoff;
- the **selection policy** deciding what a row may read (strict prospective by
  default), the **modality slots** a row's images fill, and the **target**.

:func:`preflight` opens every source once, checks every pin, reconciles the
subject's fragments, and says per row whether it is *eligible*,
*uncertifiable*, *excluded* or in *error* --- and why --- before a voxel or a
report is read.  The engine does all of it; this module is the Python face.

.. code-block:: python

    from medh5.clinical import DAY
    from medh5.task import Slot, SourceRef, TaskManifest

    task = TaskManifest.new("progression", "1", identity_namespace="site-a",
                            slots=[Slot("ct", "CT", required=True, patch=(32, 32, 16))],
                            split=("fold-0", ["train", "val"]))
    task.add_subject("P-01", [SourceRef.pin("p01.medh5")], partition="train")
    task.add_row("P-01@d30", "P-01", cutoff_us=30 * DAY)
    report = task.preflight()
    report.counts            # {"eligible": 1}
"""

from __future__ import annotations

import copy
import os
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5._deprecation import module_getattr
from medh5.clinical import (
    SELECTION_POLICIES,
    Event,
    Link,
    SelectedEvent,
    Selection,
    SelectionPolicy,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.dataset import Manifest, Split
    from medh5.sample import Sample

SCHEMA: str = _core.TASK_SCHEMA
"""The manifest's own version."""

CODES: dict[str, str] = dict(_core.COMPANION_CODES)
"""The companion contract's finding codes (T1xx manifest, T2xx identity and
splits, T3xx sources, T4xx caches) and what each means."""

ROI: tuple[str, ...] = _core.TASK_ROI
"""What a slot may request its region be centred on (contract §3.5)."""
CENSORING: tuple[str, ...] = _core.TASK_CENSORING
"""What a target does with a row it cannot label (contract §5)."""

PathLike = str | os.PathLike[str]

Confine = bool | str | os.PathLike[str] | None
"""Where a runner lets a task's sources be opened from (contract §2):
``True`` confines them to the base relative URIs resolve against, a path to
that directory, ``False`` opens wherever the URIs point.  ``None``, 2.1's
default, opens wherever they point, and warns (``FutureWarning``) when a
manifest loaded from a file names a source outside its directory: 3.0
confines to it by default."""


def _root(confine: Confine, base: PathLike | None) -> str | None:
    """The directory ``confine`` confines sources to, or ``None``."""
    if confine is None or confine is False:
        return None
    if confine is True:
        return os.fspath(base if base is not None else ".")
    return os.fspath(confine)


def schema_text() -> str:
    """The ``medh5.task/1`` JSON Schema."""
    return str(_core.task_schema_text())


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing wrong with a task, its sources, or a cache."""

    code: str
    location: str
    message: str

    @property
    def summary(self) -> str:
        return CODES.get(self.code, "")

    def __str__(self) -> str:
        return f"{self.code} {self.location}: {self.message}"

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Finding:
        return cls(str(doc["code"]), str(doc["location"]), str(doc["message"]))


@dataclass(frozen=True, slots=True)
class Note:
    """Something a manifest should state and does not.  Unlike a
    :class:`Finding`, a note never makes a manifest fail."""

    location: str
    message: str

    def __str__(self) -> str:
        return f"note {self.location}: {self.message}"

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Note:
        return cls(str(doc["location"]), str(doc["message"]))


@dataclass(frozen=True, slots=True)
class Repinned:
    """One source pinned to a new version."""

    source_id: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class RepinReport:
    """What :meth:`TaskManifest.repin` did, or would do."""

    changed: tuple[Repinned, ...]
    """The sources pinned to a new version."""
    unchanged: tuple[str, ...]
    """The sources already pinned to what they name."""
    reconciled: bool
    """Whether a changed subject's reconciled events changed."""
    rows: tuple[str, ...]
    """The rows whose fingerprints changed: their patient-level cache
    entries are stale (T403), as they should be."""

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> RepinReport:
        return cls(
            tuple(
                Repinned(str(r["source_id"]), str(r["before"]), str(r["after"]))
                for r in doc["changed"]
            ),
            tuple(str(u) for u in doc["unchanged"]),
            bool(doc["reconciled"]),
            tuple(str(r) for r in doc["rows"]),
        )


def _findings(found: Iterable[Mapping[str, Any]]) -> tuple[Finding, ...]:
    return tuple(Finding.from_json(f) for f in found)


@dataclass(frozen=True, slots=True)
class SourceRef:
    """One sample --- a file, or a member of a ``.medh5c`` --- at a pinned version.

    ``uri`` is a path (absolute, or relative to the manifest's directory;
    ``file://`` is accepted).  ``content_id`` is the pin: the reference counts
    only while the sample's content address is this one.  ``local_subject_id``
    is the sample's own ``identity.subject_id`` (:meth:`pin` records it);
    preflight holds the sample to it, or, when it is ``None``, to the id of
    the subject the source is listed under (T303).
    """

    uri: str
    content_id: str
    sample_key: str | None = None
    source_id: str = ""
    """Default: the file's stem, after the directory holding it when the URI
    names one (``site-a_P-01``), and ``:<key>`` for a member; the engine's
    derivation, the same on every platform.  :meth:`TaskManifest.add_subject`
    widens a derived id another source already holds: by the directories
    above it, then ``@2``, ``@3``."""
    local_subject_id: str | None = None
    _derived: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Derived once, here: an empty id failed the schema (T101), and the
        # stem alone gave site-a/P-01.medh5 and site-b/P-01.medh5 one (T204).
        if not self.source_id:
            object.__setattr__(
                self,
                "source_id",
                _core.source_default_id(self.uri, self.sample_key),
            )
            object.__setattr__(self, "_derived", True)

    @classmethod
    def pin(
        cls,
        path: PathLike,
        *,
        sample_key: str | None = None,
        source_id: str | None = None,
        uri: str | None = None,
    ) -> SourceRef:
        """A reference pinned to what the sample at ``path`` is now.

        ``uri`` is what the manifest records (default: ``path`` as given) ---
        pass a relative one to keep a manifest movable with its data.
        """
        return cls.from_json(
            _core.source_pin(os.fspath(path), sample_key, source_id, uri)
        )

    @property
    def locator(self) -> str:
        """``uri`` or ``uri::sample_key``."""
        return self.uri if self.sample_key is None else f"{self.uri}::{self.sample_key}"

    def resolve(
        self, base: PathLike | None = None, *, root: PathLike | None = None
    ) -> Path:
        """The file the URI names, relative to ``base`` when it is relative.

        A ``file:`` URI is decoded as the engine decodes it --- its host empty
        or ``localhost``, percent-escapes decoded, a Windows drive kept ---
        so what :meth:`pathlib.Path.as_uri` writes names the path it was made
        from.  With ``root``, the path's ``.`` and ``..`` are resolved as
        text, and one outside ``root`` is refused (T301) without being
        examined (contract §2).
        """
        return Path(
            _core.source_resolve(
                self.uri,
                None if base is None else os.fspath(base),
                None if root is None else os.fspath(root),
            )
        )

    def open(
        self, base: PathLike | None = None, *, root: PathLike | None = None
    ) -> Sample:
        """Open the sample (the pin is not checked; see :meth:`check`); with
        ``root``, only inside it (:meth:`resolve`)."""
        from medh5.collection import open_any
        from medh5.errors import MEDH5ValidationError
        from medh5.sample import Sample

        found = open_any(self.resolve(base, root=root), key=self.sample_key)
        if not isinstance(found, Sample):
            found.close()
            raise MEDH5ValidationError(
                f"{self.uri!r} is a collection; a source names one of its members "
                "with `sample_key`"
            )
        return found

    def check(
        self,
        base: PathLike | None = None,
        *,
        deep: bool = False,
        root: PathLike | None = None,
    ) -> list[Finding]:
        """Whether the sample is still the pinned version: its stored
        ``content_id`` is the pin, the root recomputes to it, and the clinical
        datasets' bytes match their digests (every dataset's with ``deep``).
        With ``root``, a path outside it is T301 (:meth:`resolve`)."""
        found = _core.source_check(
            self.to_json(),
            None if base is None else os.fspath(base),
            deep,
            None if root is None else os.fspath(root),
        )
        return list(_findings(found))

    def to_json(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "uri": self.uri,
            "sample_key": self.sample_key,
            "content_id": self.content_id,
            "local_subject_id": self.local_subject_id,
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> SourceRef:
        return cls(
            str(doc["uri"]),
            str(doc["content_id"]),
            doc.get("sample_key"),
            str(doc.get("source_id") or ""),
            doc.get("local_subject_id"),
        )


@dataclass(frozen=True, slots=True)
class Slot:
    """A stable modality slot, filled per row by the newest *eligible* image of
    its modality --- never by a filename (contract §3.5).

    ``patch`` is the window read around the slot's centre, in voxels of the
    image's own grid (``None`` reads the whole volume); ``roi`` centres it on
    the grid (``center``) or on the first eligible instance
    (``eligible_instances``).  ``classes`` are read as labels from the voxel
    annotations on the image's grid, eligible or not: supervision, which ---
    like the target --- may come from after the cutoff and never enters an
    input.
    """

    name: str
    modality: str
    required: bool = False
    patch: tuple[int, ...] | None = None
    roi: str = "center"
    classes: tuple[int, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modality": self.modality,
            "required": self.required,
            "patch": None if self.patch is None else list(self.patch),
            "roi": self.roi,
            "classes": list(self.classes),
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Slot:
        patch = doc.get("patch")
        return cls(
            str(doc["name"]),
            str(doc["modality"]),
            bool(doc.get("required", False)),
            None if patch is None else tuple(int(v) for v in patch),
            str(doc.get("roi", "center")),
            tuple(int(c) for c in doc.get("classes", ())),
        )


@dataclass(frozen=True, slots=True)
class Target:
    """What a row is labelled with (contract §5): an event concept whose final
    ``value_text`` falls in ``positive`` or ``negative`` inside
    ``(cutoff, cutoff + horizon_us]``, read from the full history.

    ``horizon_us`` and the members after it are keyword-only, and
    ``horizon_us`` has no default: a window of no length labels nothing, and
    the schema requires at least one microsecond (T101).  A negative needs an
    observation inside the window at least ``min_follow_up_us`` after the
    cutoff.  The default, the horizon, leaves a negative only an observation
    at exactly ``cutoff + horizon_us``, so nearly every row without a positive
    is censored: give ``min_follow_up_us`` whenever negatives matter.  Leaving
    it out with ``negative`` set warns (``DeprecationWarning``): 3.0 refuses
    such a target.  The default stays what it is, since it is part of every
    task fingerprint; a manifest read from a file gets a note instead
    (:attr:`TaskManifest.notes`).  A row
    it cannot label is censored --- kept with its target unobserved --- or,
    with ``censoring="exclude"``, excluded; a row whose outcome had already
    occurred is excluded as prevalent unless ``exclude_prevalent=False``.
    """

    id: str
    version: str
    code_system: str
    code: str
    positive: tuple[str, ...]
    negative: tuple[str, ...] = ()
    _: KW_ONLY
    horizon_us: int
    min_follow_up_us: int | None = None
    censoring: str = "censor"
    exclude_prevalent: bool = True
    kind: str | None = None

    def __post_init__(self) -> None:
        if self.negative and self.min_follow_up_us is None:
            from medh5._deprecation import warn

            warn(
                "medh5 3.0 refuses a target with negatives and no "
                "min_follow_up_us: state it",
                stacklevel=4,
            )

    def to_json(self) -> dict[str, Any]:
        event: dict[str, Any] = {"code_system": self.code_system, "code": self.code}
        if self.kind is not None:
            event["kind"] = self.kind
        out: dict[str, Any] = {
            "id": self.id,
            "version": self.version,
            "event": event,
            "positive": list(self.positive),
            "negative": list(self.negative),
            "horizon_us": int(self.horizon_us),
            "censoring": self.censoring,
            "exclude_prevalent": self.exclude_prevalent,
        }
        if self.min_follow_up_us is not None:
            out["min_follow_up_us"] = int(self.min_follow_up_us)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Target:
        """A target as a document states it.  A missing ``min_follow_up_us``
        is not warned about here: the manifest notes it."""
        event = doc["event"]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "medh5 3.0 refuses a target", DeprecationWarning
            )
            return cls._from_json(doc, event)

    @classmethod
    def _from_json(cls, doc: Mapping[str, Any], event: Mapping[str, Any]) -> Target:
        return cls(
            str(doc["id"]),
            str(doc["version"]),
            str(event["code_system"]),
            str(event["code"]),
            tuple(doc.get("positive", ())),
            tuple(doc.get("negative", ())),
            horizon_us=int(doc["horizon_us"]),
            min_follow_up_us=doc.get("min_follow_up_us"),
            censoring=str(doc.get("censoring", "censor")),
            exclude_prevalent=bool(doc.get("exclude_prevalent", True)),
            kind=event.get("kind"),
        )


@dataclass(frozen=True, slots=True)
class Reconciled:
    """An event version several fragments hold, and the digest they share."""

    event_id: str
    digest: str
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Subject:
    """One subject: its fragments, its clock and its partition."""

    subject_id: str
    sources: tuple[SourceRef, ...]
    partition: str | None = None
    clock_id: str | None = None
    reconciled: tuple[Reconciled, ...] = ()

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Subject:
        return cls(
            str(doc["subject_id"]),
            tuple(SourceRef.from_json(s) for s in doc.get("sources", ())),
            doc.get("partition"),
            doc.get("clock_id"),
            tuple(
                Reconciled(str(r["event_id"]), str(r["digest"]), tuple(r["sources"]))
                for r in doc.get("reconciled", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class Row:
    """One example: a subject at a cutoff."""

    row_id: str
    subject_id: str
    cutoff_us: int


def _policy_json(policy: SelectionPolicy | Mapping[str, Any] | str | None) -> Any:
    if policy is None:
        return None
    if isinstance(policy, SelectionPolicy):
        return policy.to_json()
    if isinstance(policy, str):
        return {"selection": policy}
    return dict(policy)


class TaskManifest:
    """A ``medh5.task/1`` manifest: build it, check it, save it, preflight it.

    The document is held as JSON and normalised by the engine, which fills
    every default --- so two spellings of one task share a
    :attr:`task_fingerprint`.  ``base`` resolves relative source URIs: the
    manifest's directory when it was loaded from a file.
    """

    __slots__ = ("_doc", "_ids", "_loaded", "_normal", "base")

    def __init__(self, doc: Mapping[str, Any], *, base: PathLike | None = None) -> None:
        self._doc: dict[str, Any] = copy.deepcopy(dict(doc))
        # The normal form, made on first use.  Two threads asking at once may
        # both make it; each is whole and one is kept.  Building the task
        # (`_changed`) is one thread's work.
        self._normal: dict[str, Any] | None = None
        self._ids: dict[str, Any] = {}
        self._loaded = False
        self.base: Path | None = None if base is None else Path(base)
        self._normalised()

    # -- construction ------------------------------------------------------

    @classmethod
    def new(
        cls,
        task_id: str,
        version: str,
        *,
        identity_namespace: str,
        policy: SelectionPolicy | Mapping[str, Any] | str | None = None,
        slots: Sequence[Slot] = (),
        target: Target | None = None,
        split: tuple[str, Sequence[str]] | None = None,
        description: str | None = None,
        base: PathLike | None = None,
    ) -> TaskManifest:
        """An empty manifest.  ``split`` is ``(set_id, partitions)``, the
        training partition first: learned preprocessing is fitted on it."""
        task: dict[str, Any] = {"id": task_id, "version": str(version)}
        if description is not None:
            task["description"] = description
        doc: dict[str, Any] = {
            "schema": SCHEMA,
            "task": task,
            "identity_namespace": identity_namespace,
            "slots": [s.to_json() for s in slots],
            "target": None if target is None else target.to_json(),
            "subjects": [],
            "rows": [],
        }
        found = _policy_json(policy)
        if found is not None:
            doc["policy"] = found
        if split is not None:
            doc["split"] = {"set_id": split[0], "partitions": list(split[1])}
        return cls(doc, base=base)

    @classmethod
    def from_cohort(
        cls,
        cohort: Manifest,
        split: Split | str,
        *,
        task_id: str,
        version: str,
        identity_namespace: str,
        base: PathLike | None = None,
        fold: int | None = None,
        training_partition: str = "train",
        policy: SelectionPolicy | Mapping[str, Any] | str | None = None,
        slots: Sequence[Slot] = (),
        target: Target | None = None,
        description: str | None = None,
    ) -> TaskManifest:
        """A task over a cohort: one subject per cohort ``subject_id``, its
        entries as its sources, each pinned to what its file holds now.

        ``split`` places the subjects: a :class:`~medh5.dataset.Split` of
        this cohort (a k-fold one needs ``fold``, which becomes ``val`` and
        every other fold ``train``), or the ``set_id`` of the claims its
        samples carry.  Refused: a split or a claim made from another
        manifest, an entry the split does not place or places
        ``unassigned``, a subject placed in two partitions, and an entry
        whose recorded ``content_id`` is not its file's.

        Source URIs are relative to ``base`` --- the directory the manifest
        will be saved in --- where they lie beneath it, else absolute.  The
        split is ``(set_id, partitions)`` with ``training_partition`` first;
        fragments of one subject are reconciled; there are no rows yet
        (:meth:`add_row`).
        """
        found: dict[str, Any] = _core.task_from_cohort(
            cohort._doc(),
            task_id,
            str(version),
            identity_namespace,
            split=None if isinstance(split, str) else split._doc(),
            set_id=split if isinstance(split, str) else None,
            fold=fold,
            base=None if base is None else os.fspath(base),
            training_partition=training_partition,
        )
        if description is not None:
            found["task"]["description"] = description
        found["slots"] = [s.to_json() for s in slots]
        found["target"] = None if target is None else target.to_json()
        chosen = _policy_json(policy)
        if chosen is not None:
            found["policy"] = chosen
        return cls(found, base=base)

    @classmethod
    def load(cls, path: PathLike) -> TaskManifest:
        """Read a manifest; relative source URIs resolve against its directory.

        T101 for a file that is not JSON --- ``NaN`` and the infinities
        included, which JSON does not have.
        """
        path = Path(path)
        out = cls(_core.task_load_document(path), base=path.parent)
        out._loaded = True
        return out

    def save(self, path: PathLike) -> Path:
        """Write the manifest, pretty-printed, with its fingerprint: the bytes
        ``medh5 task`` writes.

        Through a temporary file renamed over *path*: a write cut short --- a
        full disk, a crash --- leaves the previous manifest whole.  A *path*
        that is one of the manifest's sources, by any name or link, is
        refused: saving there would replace a sample with JSON.
        """
        path = Path(path)
        _core.task_save(self._doc, path, self.base)
        return path

    def _normalised(self) -> dict[str, Any]:
        if self._normal is None:
            self._normal = _core.task_normalize(self._doc)
        return self._normal

    def _changed(self) -> None:
        self._normal = None
        self._ids = {}
        self._doc.pop("fingerprint", None)

    def _identity(self, key: str) -> Any:
        """A fingerprint (or ``rows``, every row's), computed once until the
        manifest changes: each was a parse of the whole manifest per access."""
        if key not in self._ids:
            if key == "rows":
                self._ids[key] = dict(_core.task_row_fingerprints(self._doc))
            else:
                self._ids.update(_core.task_fingerprints(self._doc))
        return self._ids[key]

    def add_subject(
        self,
        subject_id: str,
        sources: Sequence[SourceRef],
        *,
        partition: str | None = None,
        clock_id: str | None = None,
    ) -> TaskManifest:
        """Declare a subject, its sources and its partition.

        A source whose id was derived (none was given) and that another
        source already holds is given a wider one: by the directories above
        it, then ``@2``, ``@3``; the first holder keeps its id.  An id given
        explicitly is kept, so one that collides is still T204."""
        taken = [
            str(s["source_id"])
            for subject in self._doc.get("subjects", ())
            for s in subject.get("sources", ())
            if s.get("source_id")
        ]
        taken += [s.source_id for s in sources if not s._derived]
        placed = []
        for source in sources:
            derived = source._derived
            if derived and source.source_id in taken:
                wider = _core.source_default_id(
                    source.uri, source.sample_key, tuple(taken)
                )
                source = replace(source, source_id=wider)
            if derived:
                taken.append(source.source_id)
            placed.append(source)
        entry: dict[str, Any] = {
            "subject_id": subject_id,
            "sources": [s.to_json() for s in placed],
        }
        if partition is not None:
            entry["partition"] = partition
        if clock_id is not None:
            entry["clock_id"] = clock_id
        self._doc.setdefault("subjects", []).append(entry)
        self._changed()
        return self

    def add_row(self, row_id: str, subject_id: str, cutoff_us: int) -> TaskManifest:
        """Add one example: ``subject_id`` at ``cutoff_us`` on its clock."""
        self._doc.setdefault("rows", []).append(
            {"row_id": row_id, "subject_id": subject_id, "cutoff_us": int(cutoff_us)}
        )
        self._changed()
        return self

    def reconcile(
        self, base: PathLike | None = None, *, confine: Confine = None
    ) -> TaskManifest:
        """Record, per subject, the event versions several fragments hold and
        the digest they agree on --- what preflight checks duplicates against.
        ``confine`` says where sources may be opened (:data:`Confine`).  A
        declared fingerprint is dropped, as ``medh5 task reconcile`` drops
        it: the manifest changed, and :meth:`save` declares the new one."""
        found = _core.task_reconcile(
            self._doc, self._base(base), self._confined(base, confine)
        )
        # The document as written, so its notes still read as written.
        self._doc["subjects"] = found["subjects"]
        self._doc.pop("fingerprint", None)
        self._changed()
        return self

    def repin(
        self,
        base: PathLike | None = None,
        *,
        sources: Iterable[str] | None = None,
        dry_run: bool = False,
        confine: Confine = None,
    ) -> RepinReport:
        """Pin each source (or those whose ids ``sources`` names) to the
        sample it names now.

        Only a whole version is pinned: its stored root recomputes and its
        digests hold.  A source that does not open is refused (T301), one
        whose integrity fails (T302), another subject (T303), another clock
        (T304) or one this engine cannot certify (T306); refused, nothing
        changes.  Changed subjects are reconciled again and the declared
        fingerprint dropped.  With ``dry_run`` the manifest is left as it
        is and the report says what would change.  ``confine`` says where
        sources may be opened (:data:`Confine`).
        """
        found = _core.task_repin(
            self._doc,
            self._base(base),
            self._confined(base, confine),
            None if sources is None else [str(s) for s in sources],
        )
        report = RepinReport.from_json(found["report"])
        if report.changed and not dry_run:
            # The document as written, its subjects the engine's.
            self._doc["subjects"] = found["manifest"]["subjects"]
            self._doc.pop("fingerprint", None)
            self._changed()
        return report

    # -- reading -----------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """The normalised manifest (every default filled), as a ``dict``."""
        return copy.deepcopy(self._normalised())

    @property
    def task_id(self) -> str:
        return str(self._normalised()["task"]["id"])

    @property
    def task_version(self) -> str:
        return str(self._normalised()["task"]["version"])

    @property
    def identity_namespace(self) -> str:
        return str(self._normalised()["identity_namespace"])

    @property
    def policy(self) -> SelectionPolicy:
        return SelectionPolicy.from_json(self._normalised()["policy"])

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(Slot.from_json(s) for s in self._normalised()["slots"])

    @property
    def target(self) -> Target | None:
        found = self._normalised().get("target")
        return None if found is None else Target.from_json(found)

    @property
    def split(self) -> tuple[str, tuple[str, ...]] | None:
        found = self._normalised().get("split")
        if found is None:
            return None
        return str(found["set_id"]), tuple(found["partitions"])

    @property
    def training_partition(self) -> str | None:
        """The split's first partition: what learned preprocessing fits on."""
        split = self.split
        return None if split is None or not split[1] else split[1][0]

    @property
    def subjects(self) -> tuple[Subject, ...]:
        return tuple(Subject.from_json(s) for s in self._normalised()["subjects"])

    @property
    def rows(self) -> tuple[Row, ...]:
        return tuple(
            Row(str(r["row_id"]), str(r["subject_id"]), int(r["cutoff_us"]))
            for r in self._normalised()["rows"]
        )

    def partition_of(self, subject_id: str) -> str | None:
        for s in self._normalised()["subjects"]:
            if s["subject_id"] == subject_id:
                found: str | None = s.get("partition")
                return found
        return None

    # -- identity ----------------------------------------------------------

    @property
    def task_fingerprint(self) -> str:
        """Identifies the task: the digest of its canonical definition."""
        return str(self._identity("task"))

    @property
    def manifest_fingerprint(self) -> str:
        """Identifies the whole manifest, rows and pins included."""
        return str(self._identity("manifest"))

    @property
    def declared_fingerprint(self) -> str | None:
        found: str | None = self._normalised().get("fingerprint")
        return found

    def row_fingerprint(self, row_id: str) -> str:
        """Identifies one example: task, subject, pinned versions, cutoff."""
        found = self._identity("rows").get(row_id)
        if found is None:  # the engine's KeyError
            return str(_core.task_row_fingerprint(self._doc, row_id))
        return str(found)

    def subjects_digest(self, partition: str) -> str:
        """Who is in a partition --- what learned preprocessing records."""
        key = f"subjects:{partition}"
        if key not in self._ids:
            self._ids[key] = _core.task_subjects_digest(self._doc, partition)
        return str(self._ids[key])

    # -- checking ----------------------------------------------------------

    def validate(self) -> list[Finding]:
        """Everything wrong with the manifest that opening no file can find ---
        a document that fails its schema is the one finding T101, as
        ``medh5 task validate`` reports it."""
        from medh5.errors import MEDH5ValidationError

        try:
            return list(_findings(_core.task_validate(self._doc)))
        except MEDH5ValidationError as exc:
            if exc.code is None:
                raise
            return [Finding(exc.code, "/", exc.message)]

    @property
    def notes(self) -> tuple[Note, ...]:
        """What the manifest should state and does not, as it was written or
        built: a target with negatives and no ``min_follow_up_us`` (contract
        §5).  Notes never make :meth:`validate` or a preflight fail."""
        return tuple(Note.from_json(n) for n in _core.task_notes(self._doc))

    def _base(self, base: PathLike | None) -> str | None:
        found = base if base is not None else self.base
        return None if found is None else os.fspath(found)

    def _confined(self, base: PathLike | None, confine: Confine) -> str | None:
        """The root ``confine`` names (:data:`Confine`), warning under the
        default when a loaded manifest names a source outside its directory."""
        found = self._base(base)
        if confine is None and self._loaded:
            outside = _core.task_sources_outside(self._doc, found)
            if outside:
                from medh5._deprecation import warn

                more = f" and {len(outside) - 1} other(s)" if len(outside) > 1 else ""
                warn(
                    "medh5 3.0 confines a task's sources to its directory: pass "
                    f"confine=False to keep reading {outside[0]!r}{more}, or "
                    "confine=<a directory> to confine them there",
                    category=FutureWarning,
                    stacklevel=4,
                )
        return _root(confine, found)

    def preflight(
        self,
        base: PathLike | None = None,
        *,
        deep: bool = False,
        confine: Confine = None,
    ) -> Preflight:
        """Open and check every source, and build every row's view.
        ``confine`` says where sources may be opened (:data:`Confine`)."""
        return Preflight.from_core(
            _core.task_preflight(
                self._doc, self._base(base), deep, self._confined(base, confine)
            )
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "doc": self._doc,
            "base": None if self.base is None else os.fspath(self.base),
            "loaded": self._loaded,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self._doc = state["doc"]
        self._normal = None
        self._ids = {}
        self._loaded = bool(state.get("loaded", False))
        self.base = None if state["base"] is None else Path(state["base"])

    def __repr__(self) -> str:
        n = self._normalised()
        return (
            f"TaskManifest({self.task_id!r} v{self.task_version}, "
            f"{len(n['subjects'])} subjects, {len(n['rows'])} rows)"
        )


# --------------------------------------------------------------------------
# Preflight: what a task admits, per row
# --------------------------------------------------------------------------
#
# A preflight crosses from the engine as columns (``_core.task_preflight``):
# each subject's merged history once, and per row only indices into it.  A
# row, an event or a selection becomes a Python object when it is asked for,
# so a cohort of millions of selected events is a few NumPy arrays, which
# also pickle compactly into ``DataLoader`` workers.

# The engine's tables, in the order the columns code them: a word the
# engine writes is always one of these (the binding refuses one that is not).
ROW_STATUSES: tuple[str, ...] = _core.ROW_STATUSES
SELECTION_STATUSES: tuple[str, ...] = _core.SELECTION_STATUSES
TARGET_STATUSES: tuple[str, ...] = _core.TARGET_STATUSES
ROIS: tuple[str, ...] = _core.ROIS
"""How a slot's region was centred: :data:`ROI`, and ``center_fallback``
when a slot asked for an eligible instance and the row has none."""


class Packed:
    """A packed UTF-8 column: one buffer, ``int64`` offsets, and a validity
    mask (``None`` when every cell is valid) --- the clinical tables' own
    encoding (1.1 §4).  Cell ``i`` is decoded when it is read."""

    __slots__ = ("data", "offsets", "valid")

    def __init__(
        self,
        data: bytes,
        offsets: npt.NDArray[np.int64],
        valid: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        self.data = data
        self.offsets = offsets
        self.valid = valid

    @classmethod
    def from_core(cls, value: Sequence[Any]) -> Packed:
        data, offsets, valid = value
        return cls(
            bytes(data),
            np.asarray(offsets),
            None if valid is None else np.asarray(valid),
        )

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> str | None:
        n = len(self)
        if not -n <= i < n:
            raise IndexError(f"cell {i} of {n}")
        i %= n
        if self.valid is not None and not self.valid[i]:
            return None
        return self.data[int(self.offsets[i]) : int(self.offsets[i + 1])].decode(
            "utf-8"
        )

    def __iter__(self) -> Iterator[str | None]:
        return (self[i] for i in range(len(self)))

    def tolist(self) -> list[str | None]:
        return list(self)

    def __getstate__(self) -> tuple[Any, ...]:
        return (self.data, self.offsets, self.valid)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self.data, self.offsets, self.valid = state


def _packed(value: Any) -> Packed | None:
    """A packed column from the engine; ``None`` for one that is all null."""
    return None if value is None else Packed.from_core(value)


def concept_token(kind: str, code_system: str | None, code: str | None) -> str:
    """An event's concept token: ``kind|code_system|code``, or ``kind|`` when
    it is uncoded.

    A ``\\`` or ``|`` inside the system or the code is escaped with a
    backslash, so two concepts never share a token: unescaped, system
    ``alpha|beta`` with code ``gamma`` and system ``alpha`` with code
    ``beta|gamma`` would be one input.  A token without either character
    reads as it always did, so a vocabulary fitted before still holds.
    """
    if code_system is None or code is None:
        return f"{kind}|"
    return f"{kind}|{_escaped(code_system)}|{_escaped(code)}"


def _escaped(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _cell(column: Packed | None, i: int) -> str | None:
    """Cell ``i`` of a packed column that may be all null."""
    return None if column is None else column[i]


def _columns(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: Packed.from_core(v) if isinstance(v, (tuple, list)) else v
        for k, v in doc.items()
    }


def _bounds_at(
    values: npt.NDArray[np.int64], known: npt.NDArray[np.bool_], i: int
) -> tuple[int, int] | None:
    return (int(values[i, 0]), int(values[i, 1])) if known[i] else None


class EventTable:
    """One subject's merged event versions, as columns.

    Indexing builds an :class:`~medh5.clinical.Event`; the numeric columns
    (``column(name)``) serve batches without building any: ``kind_code``,
    ``temporal_type_code``, ``status_code`` and ``value_comparator_code``
    index the vocabularies of :mod:`medh5.clinical` (``-1`` for null);
    ``effective_start``, ``effective_end`` and ``available`` are ``(n, 2)``
    inclusive bounds with an ``*_known`` mask; ``value_num`` has
    ``value_num_valid``; ``fragment`` names the source a version was read from.
    """

    __slots__ = ("_c", "_ids", "_concepts")

    def __init__(self, columns: Mapping[str, Any]) -> None:
        self._c = _columns(columns)
        # Lookups made on first use: two threads may both make one, each
        # whole, and one is kept --- duplicate work, so no lock.
        self._ids: dict[str, int] | None = None
        self._concepts: tuple[tuple[str, ...], npt.NDArray[np.int32]] | None = None

    def __len__(self) -> int:
        return int(self._c["n"])

    def column(self, name: str) -> Any:
        return self._c[name]

    def __getitem__(self, i: int) -> Event:
        c = self._c
        n = len(self)
        if not -n <= i < n:
            raise IndexError(f"event {i} of {n}")
        i %= n

        def text(name: str) -> str | None:
            return _cell(c[name], i)

        return Event(
            str(text("event_id")),
            str(text("record_id")),
            str(text("kind")),
            str(text("temporal_type")),
            str(text("status")),
            effective_start_us=_bounds_at(
                c["effective_start"], c["effective_start_known"], i
            ),
            effective_end_us=_bounds_at(
                c["effective_end"], c["effective_end_known"], i
            ),
            available_us=_bounds_at(c["available"], c["available_known"], i),
            timepoint_id=text("timepoint_id"),
            encounter_id=text("encounter_id"),
            code_system=text("code_system"),
            code=text("code"),
            code_version=text("code_version"),
            value_num=float(c["value_num"][i]) if c["value_num_valid"][i] else None,
            value_comparator=text("value_comparator"),
            unit=text("unit"),
            value_text=text("value_text"),
            missing_reason=text("missing_reason"),
            prov=text("prov"),
        )

    def __iter__(self) -> Iterator[Event]:
        return (self[i] for i in range(len(self)))

    def index(self, event_id: str) -> int:
        """The position of an event version, by id."""
        if self._ids is None:
            self._ids = {
                e: i for i, e in enumerate(self._c["event_id"]) if e is not None
            }
        try:
            return self._ids[event_id]
        except KeyError:
            raise KeyError(f"no event version {event_id!r}") from None

    def concepts(self) -> tuple[tuple[str, ...], npt.NDArray[np.int32]]:
        """The subject's concept tokens (``kind|code_system|code``, or
        ``kind|`` uncoded), and each version's index into them."""
        if self._concepts is None:
            kinds, systems, codes = (
                self._c["kind"],
                self._c["code_system"],
                self._c["code"],
            )
            tokens: dict[str, int] = {}
            index = np.empty(len(self), dtype=np.int32)
            for i in range(len(self)):
                token = concept_token(str(kinds[i]), _cell(systems, i), _cell(codes, i))
                index[i] = tokens.setdefault(token, len(tokens))
            self._concepts = (tuple(tokens), index)
        return self._concepts

    def __getstate__(self) -> dict[str, Any]:
        return self._c

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._c = state
        self._ids = None
        self._concepts = None

    def __repr__(self) -> str:
        return f"EventTable({len(self)} event versions)"


class LinkTable:
    """One subject's links, every fragment's, as columns; indexing builds a
    :class:`~medh5.clinical.Link`, and ``fragment(i)`` says whose it is."""

    __slots__ = ("_c",)

    def __init__(self, columns: Mapping[str, Any]) -> None:
        self._c = _columns(columns)

    def __len__(self) -> int:
        return int(self._c["n"])

    def column(self, name: str) -> Any:
        return self._c[name]

    def fragment(self, i: int) -> int:
        return int(self._c["fragment"][i])

    def __getitem__(self, i: int) -> Link:
        c = self._c
        n = len(self)
        if not -n <= i < n:
            raise IndexError(f"link {i} of {n}")
        i %= n
        span = _bounds_at(c["source_span"], c["source_span_valid"], i)
        return Link(
            str(c["source_type"][i]),
            str(c["source_id"][i]),
            str(c["relation"][i]),
            str(c["target_type"][i]),
            str(c["target_id"][i]),
            source_span=span,
            target_annotation_id=_cell(c["target_annotation_id"], i),
            asserted_by_event_id=_cell(c["asserted_by_event_id"], i),
        )

    def __iter__(self) -> Iterator[Link]:
        return (self[i] for i in range(len(self)))

    def __getstate__(self) -> dict[str, Any]:
        return self._c

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._c = state

    def __repr__(self) -> str:
        return f"LinkTable({len(self)} links)"


@dataclass(frozen=True, slots=True, eq=False)
class SubjectHistory:
    """One subject's history, merged across its fragments: what every row of
    the subject indexes.

    ``documents`` lists, per ``document`` version, the documents it owns
    (``(event, fragment, document_id)``, 1.1 §6); ``payloads`` is the table a
    row's admitted payloads index.
    """

    subject_id: str
    partition: str | None
    sources: tuple[SourceRef, ...]
    events: EventTable
    links: LinkTable
    document_events: npt.NDArray[np.int32]
    document_fragments: npt.NDArray[np.int32]
    document_ids: Packed
    payload_fragments: npt.NDArray[np.int32]
    payload_kinds: Packed
    payload_ids: Packed

    @classmethod
    def from_core(cls, doc: Mapping[str, Any]) -> SubjectHistory:
        documents, payloads = doc["documents"], doc["payloads"]
        return cls(
            str(doc["subject_id"]),
            doc.get("partition"),
            tuple(SourceRef.from_json(s) for s in doc["sources"]),
            EventTable(doc["events"]),
            LinkTable(doc["links"]),
            np.asarray(documents["event"]),
            np.asarray(documents["fragment"]),
            Packed.from_core(documents["document_id"]),
            np.asarray(payloads["fragment"]),
            Packed.from_core(payloads["kind"]),
            Packed.from_core(payloads["id"]),
        )

    def owned_documents(self, event: int) -> list[tuple[int, str]]:
        """``(fragment, document_id)`` of the documents version ``event`` owns."""
        lo, hi = np.searchsorted(self.document_events, [event, event + 1])
        return [
            (int(self.document_fragments[k]), str(self.document_ids[k]))
            for k in range(int(lo), int(hi))
        ]

    def payload(self, k: int) -> tuple[int, str, str]:
        return (
            int(self.payload_fragments[k]),
            str(self.payload_kinds[k]),
            str(self.payload_ids[k]),
        )


@dataclass(frozen=True, slots=True)
class SlotFill:
    """How one slot was filled for one row --- or that it was not."""

    slot: str
    available: bool
    fragment: int | None
    image_id: str | None
    grid_id: str | None
    event_id: str | None
    center: tuple[int, ...] | None
    roi: str
    annotations: tuple[str, ...]
    """Eligible voxel annotations on the image's grid: admissible as inputs."""
    label_annotations: tuple[str, ...] = ()
    """Every voxel annotation on the grid that labels or examined one of the
    slot's classes: supervision, which --- like the target --- may come from
    after the cutoff and never enters an input."""


@dataclass(frozen=True, slots=True)
class TargetLabel:
    """A row's target: ``positive`` (1.0), ``negative`` (0.0), or unobserved
    (``censored``, ``prevalent``, ``none``) with the reason."""

    status: str
    value: float | None
    event_id: str | None
    reason: str | None

    @property
    def observed(self) -> bool:
        return self.value is not None


def _csr(offsets: npt.NDArray[np.int64], i: int) -> slice:
    return slice(int(offsets[i]), int(offsets[i + 1]))


class RowView:
    """Everything a task admits for one row, and nothing else.

    A view over the preflight's columns: ``events``, ``selection`` and
    ``slots`` are built when asked for.  The arrays a batch reads need no
    objects at all: ``selected`` (the admitted versions, indices into
    ``subject.events``, in input order), ``order`` and ``order_known`` (each
    one's ordering bounds), ``tie_group`` and ``plan``.
    """

    __slots__ = ("_pre", "_i")

    def __init__(self, pre: Preflight, i: int) -> None:
        self._pre = pre
        self._i = i

    def _col(self, name: str) -> Any:
        return self._pre._rows[name]

    @property
    def row_id(self) -> str:
        return str(self._col("row_id")[self._i])

    @property
    def subject_id(self) -> str:
        return str(self._col("subject_id")[self._i])

    @property
    def partition(self) -> str | None:
        return _cell(self._col("partition"), self._i)

    @property
    def cutoff_us(self) -> int:
        return int(self._col("cutoff_us")[self._i])

    @property
    def fingerprint(self) -> str:
        return str(self._col("fingerprint")[self._i])

    @property
    def status(self) -> str:
        return ROW_STATUSES[int(self._col("status")[self._i])]

    @property
    def reasons(self) -> tuple[str, ...]:
        found: tuple[str, ...] = self._col("reasons")[self._i]
        return found

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"

    @property
    def subject_index(self) -> int | None:
        """The subject's position in :attr:`Preflight.subjects`."""
        k = int(self._col("subject")[self._i])
        return None if k < 0 else k

    @property
    def subject(self) -> SubjectHistory | None:
        """The subject's merged history; ``None`` when the manifest has
        findings and no source was read."""
        k = self.subject_index
        return None if k is None else self._pre.subjects[k]

    @property
    def sources(self) -> tuple[SourceRef, ...]:
        subject = self.subject
        return () if subject is None else subject.sources

    # -- the selection, as arrays -----------------------------------------

    def _events(self) -> slice:
        return _csr(self._col("events")["offsets"], self._i)

    @property
    def selected(self) -> npt.NDArray[np.int32]:
        """The admitted versions, as indices into ``subject.events``, in
        input order (clinical order, tie groups adjacent)."""
        found: npt.NDArray[np.int32] = self._col("events")["index"][self._events()]
        return found

    @property
    def order(self) -> npt.NDArray[np.int64]:
        """``(n, 2)``: the bounds each admitted version is ordered by."""
        found: npt.NDArray[np.int64] = self._col("events")["order"][self._events()]
        return found

    @property
    def order_known(self) -> npt.NDArray[np.bool_]:
        """Whether a version has an ordering time (a static one has none)."""
        found: npt.NDArray[np.bool_] = self._col("events")["order_known"][
            self._events()
        ]
        return found

    @property
    def tie_group(self) -> npt.NDArray[np.int32]:
        """Versions sharing a group have overlapping ordering times: their
        relative order is unknown."""
        found: npt.NDArray[np.int32] = self._col("events")["tie_group"][self._events()]
        return found

    @property
    def plan(self) -> npt.NDArray[np.bool_]:
        """Admitted as a plan: planned, or not started by the cutoff."""
        found: npt.NDArray[np.bool_] = self._col("events")["plan"][self._events()]
        return found

    # -- as objects -------------------------------------------------------------

    @property
    def events(self) -> tuple[Event, ...]:
        """The selected event versions, in input order."""
        subject = self.subject
        if subject is None:
            return ()
        return tuple(subject.events[int(i)] for i in self.selected)

    @property
    def event_fragments(self) -> tuple[int, ...]:
        """For each selected version, the source (index into ``sources``) it
        was read from."""
        subject = self.subject
        if subject is None:
            return ()
        fragments = subject.events.column("fragment")
        return tuple(int(fragments[i]) for i in self.selected)

    @property
    def selection(self) -> Selection | None:
        """What the cutoff admits, as :class:`~medh5.clinical.Selection`."""
        rows, i = self._pre._rows, self._i
        subject = self.subject
        if not rows["selection"][i] or subject is None:
            return None
        events = subject.events
        ids, records = events.column("event_id"), events.column("record_id")
        kinds = events.column("kind")
        selected = tuple(
            SelectedEvent(
                str(ids[k]),
                str(records[k]),
                str(kinds[k]),
                (int(o[0]), int(o[1])) if known else None,
                int(group),
                bool(plan),
            )
            for k, o, known, group, plan in zip(
                self.selected,
                self.order,
                self.order_known,
                self.tie_group,
                self.plan,
                strict=True,
            )
        )
        excluded = {
            key: int(n)
            for key, n in zip(rows["excluded_keys"], rows["excluded"][i], strict=True)
            if n
        }
        payloads = rows["payloads"]["index"][_csr(rows["payloads"]["offsets"], i)]
        return Selection(
            self.cutoff_us,
            SELECTION_POLICIES[int(rows["policy"][i])],
            SELECTION_STATUSES[int(rows["selection_status"][i])],
            selected,
            tuple(
                int(k)
                for k in rows["links"]["index"][_csr(rows["links"]["offsets"], i)]
            ),
            frozenset(subject.payload(int(k)) for k in payloads),
            tuple(rows["uncertain_records"][i]),
            excluded,
        )

    @property
    def slots(self) -> dict[str, SlotFill]:
        out: dict[str, SlotFill] = {}
        i = self._i
        if not self._col("slotted")[i]:
            return out
        for slot in self._col("slots"):
            fragment = int(slot["fragment"][i])
            ndim = int(slot["center_ndim"][i])
            image_id = _cell(slot["image_id"], i)
            out[slot["name"]] = SlotFill(
                slot["name"],
                image_id is not None,
                None if fragment < 0 else fragment,
                image_id,
                _cell(slot["grid_id"], i),
                _cell(slot["event_id"], i),
                None if ndim < 0 else tuple(int(v) for v in slot["center"][i, :ndim]),
                ROIS[int(slot["roi"][i])] if slot["roi"][i] >= 0 else "center",
                _ids_at(slot["annotations"], i),
                _ids_at(slot["label_annotations"], i),
            )
        return out

    @property
    def target(self) -> TargetLabel:
        t, i = self._col("target"), self._i
        value = float(t["value"][i])
        return TargetLabel(
            TARGET_STATUSES[int(t["status"][i])],
            None if value != value else value,
            _cell(t["event_id"], i),
            _cell(t["reason"], i),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RowView):
            return NotImplemented
        mine = (
            self.row_id,
            self.fingerprint,
            self.status,
            self.reasons,
            self.events,
            self.slots,
            self.target,
        )
        theirs = (
            other.row_id,
            other.fingerprint,
            other.status,
            other.reasons,
            other.events,
            other.slots,
            other.target,
        )
        return mine == theirs

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"RowView({self.row_id!r}, {self.status}, {len(self.selected)} events)"


def _ids_at(column: Mapping[str, Any], i: int) -> tuple[str, ...]:
    ids = column["ids"]
    if not isinstance(ids, Packed):
        ids = Packed.from_core(ids)
    span = _csr(np.asarray(column["offsets"]), i)
    return tuple(str(ids[k]) for k in range(span.start, span.stop))


class Preflight:
    """A task's preflight: manifest and source findings, every subject's
    merged history, and every row's view.

    Any finding makes the task unfit to train on (:attr:`ok` is ``False``);
    rows whose sources are wrong are in ``error``, with the reason.  ``rows``
    is a sequence of :class:`RowView`, built as they are read.
    """

    __slots__ = (
        "_by_id",
        "_rows",
        "_views",
        "findings",
        "manifest_fingerprint",
        "subjects",
        "task_fingerprint",
    )

    def __init__(
        self,
        task_fingerprint: str,
        manifest_fingerprint: str,
        findings: tuple[Finding, ...],
        subjects: tuple[SubjectHistory, ...],
        rows: Mapping[str, Any],
    ) -> None:
        self.task_fingerprint = task_fingerprint
        self.manifest_fingerprint = manifest_fingerprint
        self.findings = findings
        self.subjects = subjects
        self._rows = dict(rows)
        # Made on first use: two threads may both make one, each whole, and
        # one is kept --- duplicate work, so no lock.
        self._by_id: dict[str, int] | None = None
        self._views: tuple[RowView, ...] | None = None

    @classmethod
    def from_core(cls, doc: Mapping[str, Any]) -> Preflight:
        """From ``_core.task_preflight``'s columns."""
        rows = dict(doc["rows"])
        for key in ("row_id", "subject_id", "partition", "fingerprint"):
            rows[key] = _packed(rows[key])
        target = dict(rows["target"])
        for key in ("event_id", "reason"):
            target[key] = _packed(target[key])
        rows["target"] = target
        slots = []
        for slot in rows["slots"]:
            slot = dict(slot)
            for key in ("image_id", "grid_id", "event_id"):
                slot[key] = _packed(slot[key])
            for key in ("annotations", "label_annotations"):
                slot[key] = {
                    "offsets": np.asarray(slot[key]["offsets"]),
                    "ids": Packed.from_core(slot[key]["ids"]),
                }
            slots.append(slot)
        rows["slots"] = slots
        rows["excluded_keys"] = tuple(rows["excluded_keys"])
        return cls(
            str(doc["task_fingerprint"]),
            str(doc["manifest_fingerprint"]),
            _findings(doc.get("findings", ())),
            tuple(SubjectHistory.from_core(s) for s in doc["subjects"]),
            rows,
        )

    @property
    def rows(self) -> tuple[RowView, ...]:
        if self._views is None:
            self._views = tuple(RowView(self, i) for i in range(int(self._rows["n"])))
        return self._views

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def counts(self) -> dict[str, int]:
        codes = np.bincount(
            np.asarray(self._rows["status"], dtype=np.int64),
            minlength=len(ROW_STATUSES),
        )
        return {ROW_STATUSES[k]: int(n) for k, n in enumerate(codes) if n}

    def row(self, row_id: str) -> RowView:
        if self._by_id is None:
            self._by_id = {str(r): i for i, r in enumerate(self._rows["row_id"])}
        try:
            return RowView(self, self._by_id[row_id])
        except KeyError:
            raise KeyError(f"no row {row_id!r}") from None

    def eligible(self, partition: str | None = None) -> tuple[RowView, ...]:
        """The eligible rows, of one partition when given."""
        return tuple(
            r
            for r in self.rows
            if r.eligible and (partition is None or r.partition == partition)
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "task_fingerprint": self.task_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "findings": self.findings,
            "subjects": self.subjects,
            "rows": self._rows,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.task_fingerprint = state["task_fingerprint"]
        self.manifest_fingerprint = state["manifest_fingerprint"]
        self.findings = state["findings"]
        self.subjects = state["subjects"]
        self._rows = dict(state["rows"])
        self._by_id = None
        self._views = None

    def __repr__(self) -> str:
        return f"Preflight({int(self._rows['n'])} rows, {self.counts}, ok={self.ok})"


def preflight(
    task: TaskManifest | PathLike,
    base: PathLike | None = None,
    *,
    deep: bool = False,
    confine: Confine = None,
) -> Preflight:
    """Preflight a task (a manifest, or the path of one); ``confine`` says
    where sources may be opened (:data:`Confine`)."""
    manifest = task if isinstance(task, TaskManifest) else TaskManifest.load(task)
    return manifest.preflight(base, deep=deep, confine=confine)


__all__ = [
    "CENSORING",
    "CODES",
    "ROI",
    "ROIS",
    "ROW_STATUSES",
    "SCHEMA",
    "SELECTION_STATUSES",
    "TARGET_STATUSES",
    "Confine",
    "EventTable",
    "Finding",
    "LinkTable",
    "Note",
    "Packed",
    "Preflight",
    "Reconciled",
    "RepinReport",
    "Repinned",
    "Row",
    "RowView",
    "SlotFill",
    "Slot",
    "SourceRef",
    "Subject",
    "SubjectHistory",
    "Target",
    "TargetLabel",
    "TaskManifest",
    "concept_token",
    "preflight",
    "schema_text",
]

_DEPRECATED = {
    "STATUSES": (
        ROW_STATUSES,
        "medh5 3.0 removes medh5.task.STATUSES: use ROW_STATUSES",
    ),
    "LABELS": (
        TARGET_STATUSES,
        "medh5 3.0 removes medh5.task.LABELS: use TARGET_STATUSES",
    ),
}
__getattr__ = module_getattr(__name__, _DEPRECATED)
