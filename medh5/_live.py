"""Live views of a sample document's members.

``SampleDocument.extra``, ``.acquisition`` and ``.quality`` are dicts a caller
edits in place --- ``w.document.extra["tool"] = {...}`` reaches the file --- as
1.x's dataclass fields were.  The document lives in the format engine, so each
is a ``dict`` that writes every edit through to it: a member set, replaced or
deleted, and a value nested in a member changed in place.

The records a document hands out are members too.  ``w.document.identity``,
``w.activity(...)``, a timepoint, a quality record, a label set's class: each
keeps its dicts (``extra``, ``params``, ``series_uids``, an agreement's
``per_class``, a class's ``properties``), and an edit to one rewrites the
record as a whole in the document.

What a view holds is what the document held when it was read, and again after
every edit made through it.  A value set is converted as ``SampleWriter.extra``
converts one, and the view holds the conversion, not the object given.  A
nested change rewrites its member as a whole, and only while the document's
copy is still what the view read: once that member has changed another way
(``SampleWriter.acquisition``, or another view), the change is refused rather
than written over it, and the member's containers refuse every further edit
--- read the member again and edit that.

A copy of a view (``copy.copy``, ``dict(view)``, ``pickle``) is a plain
``dict``, detached from the document; a copy of a record is a record no
document handed out.

Threads may share a view.  A change nested in a member, the rewrite it makes
(comparing the member's token and replacing it), an edit through a view
with its reread of the members the edit left, and a view's reread of the
document each happen in one step, under one lock: two threads editing one
member each write their edit over the other's, never over a token the other
has already spent, and never rewrite a container the other is changing.  A
value, a key or an index given is converted before the lock is taken, and an
item to remove is found before it, so no caller's code runs under it.
"""

from __future__ import annotations

import copy
import operator
import os
import threading
from collections.abc import Iterable, Mapping
from typing import Any, SupportsIndex

__all__ = ["Members"]

_LOCK = threading.RLock()
"""Held while a container nested in a member changes and the member is
rewritten (`_Anchor.rewrite`), while a view edits the document and takes
the members the edit left, and while a view rereads the document
(`Members._sync`).  Reentrant, so nested use on one thread never blocks
itself."""


def _after_fork_in_child() -> None:
    """A lock of the child's own: one another of the parent's threads held at
    the fork would be held forever.  The engine looks the lock up here on
    each use, so it takes this one too."""
    global _LOCK
    _LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):  # POSIX; Windows has no fork
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _key(key: Any) -> Any:
    """*key* as the document keeps it, made before the lock: a subclass of
    ``str``, ``int`` or ``float`` (whose ``__hash__``, ``__eq__`` and
    ``__str__`` are the caller's) as the built-in value it holds, an object
    with ``__index__`` as its ``int``; any other key is refused here, as the
    rewrite would refuse it, before anything changes."""
    if key is None or type(key) in (str, int, float, bool):
        return key
    if isinstance(key, str):
        return str.__str__(key)
    if isinstance(key, int):
        return int.__int__(key)
    if isinstance(key, float):
        return float.__float__(key)
    if hasattr(type(key), "__index__"):
        return operator.index(key)
    raise TypeError(
        f"keys must be str, int, float, bool or None, not {type(key).__name__}"
    )


_ABSENT = object()


def _found(key: Any) -> Any:
    """*key* for a lookup, as `_key` makes it; ``_ABSENT`` for a key no
    member can hold (an unhashable one still raises, as a dict's lookup
    would)."""
    try:
        return _key(key)
    except TypeError:
        hash(key)
        return _ABSENT


def _index(index: Any) -> Any:
    """*index* as a list takes it, converted before the lock: an object's
    ``__index__``, and a slice's bounds', are the caller's code."""
    if isinstance(index, slice):
        bounds = (index.start, index.stop, index.step)
        return slice(*(None if b is None else operator.index(b) for b in bounds))
    return operator.index(index)


def _unchanged(items: list[Any], before: list[Any]) -> bool:
    """Whether *items* still holds what *before* copied of it, item by item."""
    return len(items) == len(before) and all(
        a is b for a, b in zip(items, before, strict=True)
    )


class _Anchor:
    """One member's containers, and the member as the document held it when
    they were read (its canonical JSON).  The member is a key of *field*, or
    with no *key*, the field itself (``identity``); its root is the value read
    --- a record, or plain JSON."""

    __slots__ = ("document", "field", "key", "root", "stale", "token")

    def __init__(self, document: Any, field: str, key: str | None, token: str) -> None:
        self.document = document
        self.field = field
        self.key = key
        self.token = token
        self.root: Any = None
        self.stale = False

    @property
    def member(self) -> str:
        """The member, as a caller names it."""
        if self.key is None:
            return f"document.{self.field}"
        if self.field == "provenance.activities":
            return f"document.provenance.activity({self.key!r})"
        return f"document.{self.field}[{self.key!r}]"

    def check(self) -> None:
        if self.stale:
            raise RuntimeError(
                f"`{self.member}` was read before an edit the document "
                "refused; read it again and edit that"
            )

    def rewrite(self) -> None:
        """Write the member back after a change nested in it."""
        with _LOCK:
            self.check()
            try:
                self.token = self.document._replace_member(
                    self.field, self.key, self.token, self.root
                )
            except BaseException:
                # The change is made here and not in the document: these
                # containers no longer hold what the document holds.
                self.stale = True
                raise


def _refresh(view: dict[str, Any], held: dict[str, Any]) -> None:
    """Make *view* hold *held*, in its order, a key at a time: a thread
    reading the view without the lock (``view[key]``, ``len``, iteration)
    sees each key as it was or as it is, never a view cleared to be refilled.
    A key the document moved to the end (deleted and added again) moves
    there; the keys before it stay where they are."""
    for key in [k for k in dict.keys(view) if k not in held]:
        dict.__delitem__(view, key)
    order = list(held)
    kept = 0
    for key in dict.keys(view):
        if kept < len(order) and order[kept] == key:
            kept += 1
    for at, key in enumerate(order):
        value = held[key]
        if at >= kept:
            dict.pop(view, key, None)
            dict.__setitem__(view, key, value)
        elif dict.__getitem__(view, key) is not value:
            dict.__setitem__(view, key, value)


class Members(dict[str, Any]):
    """One free-form member mapping of a document (``extra``, ``acquisition``
    or ``quality``), writing every edit through to the document."""

    __slots__ = ("_anchors", "_document", "_field")

    _document: Any
    _field: str
    _anchors: dict[str, _Anchor]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Built directly --- by `dataclasses.asdict`, say --- a view is a
        # plain dict, with no document to write through to.
        super().__init__(*args, **kwargs)
        self._document = None
        self._field = ""
        self._anchors = {}

    @classmethod
    def _of(
        cls, document: Any, field: str, members: list[Any] | None = None
    ) -> Members:
        """The view of *field* of *document*, holding *members* when an edit
        handed them over (as `_sync` takes them)."""
        view = cls()
        view._document = document
        view._field = field
        view._sync(members)
        return view

    def _sync(self, members: list[Any] | None = None) -> None:
        """Hold what the document holds, keeping the containers of a member
        that has not changed (a caller may be holding them).  An edit hands
        over *members* as it left them, read under its own guard: a view
        brought up to date after an edit the document kept is never refused,
        as a second read could be."""
        with _LOCK:
            if members is None:
                members = self._document._members(self._field)
            held: dict[str, Any] = {}
            anchors: dict[str, _Anchor] = {}
            for key, value, token in members:
                anchor = self._anchors.get(key)
                if anchor is not None and anchor.token == token and not anchor.stale:
                    held[key] = anchor.root
                else:
                    anchor = _Anchor(self._document, self._field, key, token)
                    anchor.root = held[key] = _adopted(value, anchor)
                anchors[key] = anchor
            _refresh(self, held)
            self._anchors = anchors

    # -- mutation ----------------------------------------------------------------

    def _converted(self, value: Any) -> Any:
        """*value* as a member holds it, made before the lock, so the caller's
        code runs outside it (a quality record is the engine's own).  An edit
        and the reread of the members it left are one step under the lock, so
        two threads' rereads land in the order of their edits."""
        return value if self._field == "quality" else self._document._plain(value)

    def __setitem__(self, key: str, value: Any) -> None:
        if self._document is None:
            super().__setitem__(key, value)
            return
        value = self._converted(value)
        with _LOCK:
            self._sync(self._document._set_member(self._field, key, value))

    def __delitem__(self, key: str) -> None:
        if self._document is None:
            super().__delitem__(key)
            return
        with _LOCK:
            self._sync(self._document._delete_member(self._field, key))

    def __ior__(self, other: Any) -> Members:  # type: ignore[override,misc]
        self.update(other)
        return self

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        if self._document is None:
            super().update(other, **kwargs)
            return
        # A value refused partway leaves those before it written.
        given: list[tuple[Any, Any]] = []
        refused: BaseException | None = None
        for key, value in dict(other, **kwargs).items():
            try:
                given.append((key, self._converted(value)))
            except BaseException as e:
                refused = e
                break
        with _LOCK:
            members = None
            try:
                for key, value in given:
                    members = self._document._set_member(self._field, key, value)
            finally:
                if members is not None:
                    self._sync(members)
        if refused is not None:
            raise refused

    def setdefault(self, key: str, default: Any = None) -> Any:
        if self._document is None:
            return super().setdefault(key, default)
        # A default is converted only when the key is missing, and before
        # the lock, so the caller's code runs outside it (a quality record is
        # the engine's own); the look and the write are one step, so one
        # default is the member.
        key = _key(key)
        with _LOCK:
            self._sync()
            if key in self:
                return self[key]
        value = self._converted(default)
        with _LOCK:
            self._sync()
            if key not in self:
                self[key] = value
            return self[key]

    def pop(self, key: str, *default: Any) -> Any:
        if self._document is None:
            return super().pop(key, *default)
        found = _found(key)
        with _LOCK:
            self._sync()
            if found is _ABSENT or found not in self:
                if default:
                    return default[0]
                raise KeyError(key)
            # Detached: the member it was is gone from the document.
            value = copy.deepcopy(self[found])
            del self[found]
            return value

    def popitem(self) -> tuple[str, Any]:
        if self._document is None:
            return super().popitem()
        with _LOCK:
            self._sync()
            if not self:
                raise KeyError("popitem(): dictionary is empty")
            key = next(reversed(self))
            return key, self.pop(key)

    def clear(self) -> None:
        if self._document is None:
            super().clear()
            return
        with _LOCK:
            self._sync(self._document._clear_members(self._field))

    def __reduce__(self) -> tuple[Any, ...]:
        return (dict, (dict(self),))


class _NestedDict(dict[str, Any]):
    """A JSON object inside a member: an edit rewrites the member."""

    __slots__ = ("_anchor",)

    _anchor: _Anchor | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._anchor = None

    def _adopt(self, value: Any) -> Any:
        if self._anchor is None:
            return value
        self._anchor.check()
        return _nested(self._anchor.document._plain(value), self._anchor)

    def _changed(self) -> None:
        if self._anchor is not None:
            self._anchor.rewrite()

    def _check(self) -> None:
        if self._anchor is not None:
            self._anchor.check()

    def __setitem__(self, key: str, value: Any) -> None:
        key, value = _key(key), self._adopt(value)
        with _LOCK:
            self._check()
            super().__setitem__(key, value)
            self._changed()

    def __delitem__(self, key: str) -> None:
        found = _found(key)
        if found is _ABSENT:
            raise KeyError(key)
        with _LOCK:
            self._check()
            super().__delitem__(found)
            self._changed()

    def __ior__(self, other: Any) -> _NestedDict:  # type: ignore[override,misc]
        self.update(other)
        return self

    def update(self, other: Any = (), /, **kwargs: Any) -> None:
        given = dict(other, **kwargs)
        changes = {_key(key): self._adopt(value) for key, value in given.items()}
        with _LOCK:
            self._check()
            super().update(changes)
            self._changed()

    def setdefault(self, key: str, default: Any = None) -> Any:
        # A default is converted only when the key is missing, as a dict
        # never reads one it does not keep; outside the lock, as `Members`.
        key = _key(key)
        with _LOCK:
            if key in self:
                return self[key]
        value = self._adopt(default)
        with _LOCK:
            if key not in self:
                self._check()
                super().__setitem__(key, value)
                self._changed()
            return self[key]

    def pop(self, key: str, *default: Any) -> Any:
        found = _found(key)
        with _LOCK:
            if found is _ABSENT or found not in self:
                if default:
                    return default[0]
                raise KeyError(key)
            self._check()
            value = super().pop(found)
            self._changed()
        return copy.deepcopy(value)

    def popitem(self) -> tuple[str, Any]:
        with _LOCK:
            self._check()
            key, value = super().popitem()
            self._changed()
        return key, copy.deepcopy(value)

    def clear(self) -> None:
        with _LOCK:
            self._check()
            super().clear()
            self._changed()

    def __reduce__(self) -> tuple[Any, ...]:
        return (dict, (dict(self),))


class _NestedList(list[Any]):
    """A JSON array inside a member: an edit rewrites the member."""

    __slots__ = ("_anchor",)

    _anchor: _Anchor | None

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._anchor = None

    def _adopt(self, value: Any) -> Any:
        if self._anchor is None:
            return value
        self._anchor.check()
        return _nested(self._anchor.document._plain(value), self._anchor)

    def _check(self) -> None:
        if self._anchor is not None:
            self._anchor.check()

    def _changed(self) -> None:
        if self._anchor is not None:
            self._anchor.rewrite()

    def __setitem__(self, index: Any, value: Any) -> None:
        # Refused before anything changes, even an empty slice's values.
        self._check()
        index = _index(index)
        if isinstance(index, slice):
            value = [self._adopt(v) for v in value]
        else:
            value = self._adopt(value)
        with _LOCK:
            self._check()
            super().__setitem__(index, value)
            self._changed()

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        index = _index(index)
        with _LOCK:
            self._check()
            super().__delitem__(index)
            self._changed()

    def __iadd__(self, other: Iterable[Any]) -> _NestedList:  # type: ignore[misc]
        self.extend(other)
        return self

    def __imul__(self, count: SupportsIndex) -> _NestedList:
        count = operator.index(count)
        with _LOCK:
            self._check()
            super().__imul__(count)
            self._changed()
        return self

    def append(self, value: Any) -> None:
        value = self._adopt(value)
        with _LOCK:
            self._check()
            super().append(value)
            self._changed()

    def extend(self, values: Iterable[Any]) -> None:
        adopted = [self._adopt(v) for v in values]
        with _LOCK:
            self._check()
            super().extend(adopted)
            self._changed()

    def insert(self, index: SupportsIndex, value: Any) -> None:
        index, value = operator.index(index), self._adopt(value)
        with _LOCK:
            self._check()
            super().insert(index, value)
            self._changed()

    def pop(self, index: SupportsIndex = -1) -> Any:
        index = operator.index(index)
        with _LOCK:
            self._check()
            value = super().pop(index)
            self._changed()
        return copy.deepcopy(value)

    def _read(self) -> tuple[list[Any], str | None]:
        """A copy of the list, and the member's token as it was then: any edit
        to the member, a change to one of this list's items included, spends
        the token."""
        with _LOCK:
            return list(self), self._anchor.token if self._anchor else None

    def _still(self, before: list[Any], token: str | None) -> bool:
        """Whether the list and its items are as `_read` read them."""
        return _unchanged(self, before) and (
            self._anchor is None or self._anchor.token == token
        )

    def remove(self, value: Any) -> None:
        self._check()
        # Found in a copy before the lock (`==` is the caller's), and removed
        # only if no other edit came in meanwhile --- to the list or to an
        # item of it; if one did, found again.
        while True:
            before, token = self._read()
            found = (
                i for i, item in enumerate(before) if item is value or item == value
            )
            at = next(found, None)
            if at is None:
                raise ValueError("list.remove(x): x not in list")
            with _LOCK:
                self._check()
                if self._still(before, token):
                    super().__delitem__(at)
                    self._changed()
                    return

    def clear(self) -> None:
        with _LOCK:
            self._check()
            super().clear()
            self._changed()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        self._check()
        # A sort that fails partway leaves a list part-sorted, which the next
        # edit would write through: sort a copy, and keep it only if it ends.
        # The copy is sorted before the lock (`key` is the caller's), and
        # kept only if no other edit came in meanwhile, to the list or to an
        # item it was sorted by.
        before, token = self._read()
        ordered = sorted(before, key=key, reverse=reverse)
        with _LOCK:
            self._check()
            if not self._still(before, token):
                raise RuntimeError(
                    f"`{self._anchor.member if self._anchor else 'list'}` "
                    "changed on another thread while it was sorted; sort it again"
                )
            super().__setitem__(slice(None), ordered)
            self._changed()

    def reverse(self) -> None:
        with _LOCK:
            self._check()
            super().reverse()
            self._changed()

    def __reduce__(self) -> tuple[Any, ...]:
        return (list, (list(self),))


def _adopted(value: Any, anchor: _Anchor) -> Any:
    """*value*, read as *anchor*'s member, writing its edits back through
    *anchor*: a record (a quality record) anchored, plain JSON as
    containers."""
    if hasattr(value, "_anchor_to"):
        value._anchor_to(anchor)
        return value
    return _nested(value, anchor)


def _nested(value: Any, anchor: _Anchor) -> Any:
    """*value* (plain JSON) with its objects and arrays as containers of
    *anchor*'s member."""
    nested: _NestedDict | _NestedList
    if isinstance(value, Mapping):
        nested = _NestedDict({k: _nested(v, anchor) for k, v in value.items()})
    elif isinstance(value, list):
        nested = _NestedList([_nested(v, anchor) for v in value])
    else:
        return value
    nested._anchor = anchor
    return nested
