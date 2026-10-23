"""medh5 --- a self-describing HDF5 container for one medical imaging sample.

A **sample** is one subject at one or more timepoints, with every image,
annotation, transform and curation record about them in a single file --- and,
from format 1.1, the clinical history around them (``medh5.clinical``).  See
``docs/spec/medh5-1.0.md`` and ``docs/spec/medh5-1.1.md`` for the normative
format specification; tasks and feature caches over samples are the companion
contract of ``docs/spec/task-cache-1.md`` (``medh5.task``, ``medh5.cache``).

.. code-block:: python

    import medh5

    with medh5.open("case_0001.medh5") as s:
        s.identity.subject_id
        s.at("tp1").images["CT_tp1"].read(physical=True)
        s.annotations["organs"].dense(["liver", "spleen"])

0.x files are read by ``medh5 migrate`` and nothing else: the package carries a
reader for the old layout, not an implementation of it.

The format engine is Rust (the ``medh5`` crate); this package is its Python
face.  ``__version__`` is the engine's --- the Cargo workspace version, stamped
on the wheel, into every file's ``generator`` and into every manifest --- and
``__format_version__`` the newest format version it writes.  A sample is
written at the lowest version its content needs: 1.0 for imaging alone, 1.1
with the clinical profile.  The collection, sampling, task and cache tools load
on first use.
"""

from __future__ import annotations

from typing import Any

from medh5._core import __format_version__, __version__, get_threads, set_threads
from medh5.annotations import Annotation, Instance, VoxelAnnotation
from medh5.curation import (
    Activity,
    Agent,
    Agreement,
    Cohort,
    Deidentification,
    Identity,
    Issue,
    Observation,
    Provenance,
    QualityRecord,
    SplitClaim,
    Timeline,
    Timepoint,
    Track,
    Tracking,
)
from medh5.document import SampleDocument
from medh5.errors import (
    CODES,
    MEDH5Error,
    MEDH5FileError,
    MEDH5FileNotFoundError,
    MEDH5IntegrityError,
    MEDH5IOError,
    MEDH5IsADirectoryError,
    MEDH5PermissionError,
    MEDH5SchemaError,
    MEDH5ValidationError,
    MEDH5VersionError,
)
from medh5.geometry import Grid
from medh5.image import Image
from medh5.labels import LabelClass, LabelSet
from medh5.sample import (
    FORMAT_VERSION,
    PROFILES,
    Sample,
    SampleWriter,
    amend,
    create,
    open_sample,
)

open = open_sample

_LAZY: dict[str, tuple[str, str]] = {
    "Collection": ("medh5.collection", "Collection"),
    "open_collection": ("medh5.collection", "open_collection"),
    "pack": ("medh5.collection", "pack"),
    "unpack": ("medh5.collection", "unpack"),
    "compare_annotations": ("medh5.curation.agreement", "compare"),
    "SplitAudit": ("medh5.curation.splits", "SplitAudit"),
    "audit_splits": ("medh5.curation.splits", "audit_splits"),
    "Patch": ("medh5.sampling", "Patch"),
    "PatchSampler": ("medh5.sampling", "PatchSampler"),
    "TimepointPair": ("medh5.sampling", "TimepointPair"),
    "TimepointPairSampler": ("medh5.sampling", "TimepointPairSampler"),
    "grid_patches": ("medh5.sampling", "grid_patches"),
    "TaskManifest": ("medh5.task", "TaskManifest"),
    "FeatureCache": ("medh5.cache", "FeatureCache"),
    "validate_cache": ("medh5.cache", "validate_cache"),
}


def __getattr__(name: str) -> Any:
    """Load the tool modules on first use (PEP 562)."""
    try:
        module, attribute = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'medh5' has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "CODES",
    "FORMAT_VERSION",
    "PROFILES",
    "Activity",
    "Agent",
    "Agreement",
    "Annotation",
    "Cohort",
    "Collection",
    "Deidentification",
    "FeatureCache",
    "Grid",
    "Identity",
    "Image",
    "Instance",
    "Issue",
    "LabelClass",
    "LabelSet",
    "MEDH5Error",
    "MEDH5FileError",
    "MEDH5FileNotFoundError",
    "MEDH5IOError",
    "MEDH5IntegrityError",
    "MEDH5IsADirectoryError",
    "MEDH5PermissionError",
    "MEDH5SchemaError",
    "MEDH5ValidationError",
    "MEDH5VersionError",
    "Observation",
    "Patch",
    "PatchSampler",
    "Provenance",
    "QualityRecord",
    "Sample",
    "SampleDocument",
    "SampleWriter",
    "SplitAudit",
    "SplitClaim",
    "TaskManifest",
    "Timeline",
    "Timepoint",
    "TimepointPair",
    "TimepointPairSampler",
    "Track",
    "Tracking",
    "VoxelAnnotation",
    "__format_version__",
    "__version__",
    "amend",
    "audit_splits",
    "compare_annotations",
    "create",
    "get_threads",
    "grid_patches",
    "open",
    "open_collection",
    "open_sample",
    "pack",
    "set_threads",
    "unpack",
    "validate_cache",
]
