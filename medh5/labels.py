"""The label space: label sets and the vocabulary registry (spec §5).

Annotations reference classes by ``uint16`` id and never by name, so a label set
is the only thing standing between an integer and a diagnosis.  Two properties
matter more than the data model:

* The hierarchy is a **DAG, not a tree**.  ``left_kidney`` is a ``kidney`` and is
  part of the urinary system; forcing that into a tree loses one of the two.
* ``closure`` is declared per annotation, never inferred.  A reader that helpfully
  adds ``liver`` because ``liver_segment_iv`` is present has invented ground
  truth, so the spec forbids it unless ``closure = "implicit"`` says otherwise.

The classes are the engine's (``medh5._core``): construction validates §5.1-§5.3
and ``to_json`` is the canonical serialization, in every language binding alike.

Three vocabularies ship with the engine, chosen because they cover the shapes a
label set can take rather than because they are exhaustive: one class
(``binary-foreground``), a small hierarchy with overlap-capable sub-regions
(``brats-subregions``), and a flat multi-organ set (``amos22-organs``).

``brats-subregions`` uses the BraTS 2023 numbering (enhancing tumour is 3) and
adds a synthetic parent, ``tumor``, as id 4.  BraTS releases up to 2021 label
enhancing tumour 4: relabel their masks 4 to 3 before writing them with this
vocabulary, or their enhancing tumour lands on the parent.

**No ontology codes are bundled.**  A wrong SNOMED-CT or FMA binding is a silent
data-integrity defect that propagates into every file written with the
vocabulary, and it is not detectable by any validator.  Bindings are the
curator's to add --- :class:`OntologyCode` exists for exactly that --- and the
validator's W912 says so when they are missing.
"""

from __future__ import annotations

import copy
import os
import threading
from collections.abc import Sequence
from typing import Any

from medh5 import _core
from medh5._core import (
    BACKGROUND_ID,
    CLOSURES,
    FORMS,
    IGNORE_ID,
    INLINE_REQUIRED_BELOW,
    MAX_CLASS_ID,
    LabelClass,
    LabelSet,
    OntologyCode,
    Relation,
    Skeleton,
    canonical_json,
    check_class_id,
    from_keys,
)

# -- the vocabulary registry ---------------------------------------------------

_REGISTERED: dict[str, LabelSet] = {}
"""The objects callers registered, so ``load`` hands back the same one."""

_REGISTRY = threading.Lock()
"""Held across both halves of a registration, the engine's and
``_REGISTERED``, so threads registering and loading one name never see one
half without the other."""


def _after_fork_in_child() -> None:
    """A registry lock of the child's own: one another of the parent's
    threads held at the fork would be held forever."""
    global _REGISTRY
    _REGISTRY = threading.Lock()


if hasattr(os, "register_at_fork"):  # POSIX; Windows has no fork
    os.register_at_fork(after_in_child=_after_fork_in_child)


def available() -> tuple[str, ...]:
    """Every vocabulary name :func:`load` accepts, bundled and registered."""
    return tuple(_core.registry_available())


def load(name: str) -> LabelSet:
    """A bundled or registered vocabulary by name; E305 for an unknown one."""
    with _REGISTRY:
        registered = _REGISTERED.get(name)
        if registered is not None:
            return registered
        return _core.registry_load(name)


def load_file(path: str | os.PathLike[str]) -> LabelSet:
    """A vocabulary from a JSON file on disk."""
    return _core.registry_load_file(path)


def register(name: str, label_set: LabelSet) -> LabelSet:
    """Make a vocabulary loadable by name for the rest of the process."""
    # Taken as it stands before the lock: reading it runs the caller's code
    # (a value in a class's `properties`), which may load or register too.
    current = copy.copy(label_set) if isinstance(label_set, LabelSet) else label_set
    with _REGISTRY:
        _core.registry_register(name, current)
        _REGISTERED[name] = label_set
    return label_set


def unregister(name: str) -> None:
    """Drop a registered vocabulary; bundled ones cannot be removed."""
    with _REGISTRY:
        _REGISTERED.pop(name, None)
        _core.registry_unregister(name)


def describe() -> dict[str, dict[str, Any]]:
    """Name -> ``{id, version, classes, sha256}``."""
    return dict(_core.registry_describe())


# -- across files --------------------------------------------------------------


def check(paths: Sequence[str | os.PathLike[str]]) -> dict[str, Any]:
    """Whether sample files share one vocabulary, as ``medh5 labels check --json``.

    Files are grouped by their label set's digest, the rule ``dataset check``
    applies for ``C101`` and ``C103``: an empty label set is a vocabulary of its
    own, told apart by its id.  The result holds ``ok`` (every file read, one
    vocabulary), ``vocabularies`` (the files of each, by ``id@version#`` and 16
    hex digits of its digest, or ``<none>``), ``unreadable`` (path to reason)
    and ``files`` (an entry for every path, in order).
    """
    return dict(_core.labels_check([os.fspath(p) for p in paths]))


__all__ = [
    "BACKGROUND_ID",
    "CLOSURES",
    "FORMS",
    "IGNORE_ID",
    "INLINE_REQUIRED_BELOW",
    "MAX_CLASS_ID",
    "LabelClass",
    "LabelSet",
    "OntologyCode",
    "Relation",
    "Skeleton",
    "available",
    "canonical_json",
    "check",
    "check_class_id",
    "describe",
    "from_keys",
    "load",
    "load_file",
    "register",
    "unregister",
]
