"""Type stubs for the compiled format engine (``medh5._core``).

The extension is the Rust engine's Python face; the modules of the
``medh5`` package re-export it under the names 1.x users know.  Keep this
file in step with ``crates/medh5-python``: ``tests/project/test_typing.py``
checks every name and parameter against the built module.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Literal, final

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, disjoint_base

from medh5.annotations.base import VoxelAnnotation
from medh5.annotations.geometric import Polygon, Polyline
from medh5.annotations.voxel import InstanceInput

__all__ = [
    "ACTIVITY_FIELDS",
    "ACTIVITY_TYPES",
    "AGENT_FIELDS",
    "AGENT_TYPES",
    "AGREEMENT_DEFAULT_IOU",
    "AGREEMENT_OBJECT_KINDS",
    "ANNOTATION_KINDS",
    "ASSESSMENT_SYSTEM",
    "ATTESTED_GROUPS",
    "AXIS_KINDS",
    "Activity",
    "Agent",
    "Agreement",
    "AnnotationHandle",
    "AnnotationHeader",
    "AnnotationPayload",
    "Attrs",
    "BACKGROUND_ID",
    "BENCH_MANY_CLASSES",
    "BENCH_TARGETS",
    "BITS_PER_PLANE",
    "BLOSC2_FILTER_ID",
    "BLOSC_FILTER_ID",
    "BUILTIN_FILTER_IDS",
    "BULK_MIN_BYTES",
    "CACHE_LEVELS",
    "CACHE_SAFETY",
    "CACHE_SCHEMA",
    "CACHE_SUFFIX",
    "CHECK_CODES",
    "CHECK_CODE_SEVERITY",
    "CHECK_SEVERITIES",
    "CLINICAL_PROFILE",
    "CLINICAL_SCHEMA",
    "CLOCK_REFERENCES",
    "CLOSURES",
    "CODE_DOMAINS",
    "COLLECTION_SUFFIX",
    "COMPANION_CODES",
    "COMPARATORS",
    "COMPRESS_MIN_BYTES",
    "CONFORMANCE_CHECKSUMS",
    "CONFORMANCE_SCHEMA",
    "CONFORMANCE_SEED",
    "CONTOUR_ROLES",
    "COORD_SYSTEMS",
    "CacheWriterHandle",
    "ClinicalHandle",
    "Cohort",
    "CollectionHandle",
    "CostModel",
    "DATE_REGEX",
    "DEFAULT_ALGO",
    "DEFAULT_L3_BYTES",
    "DEFAULT_MAX_COORDS",
    "DEFAULT_OCCUPANCY_FACTOR",
    "DEFAULT_ORDER",
    "DEFAULT_PATCH",
    "DEFAULT_PROFILE",
    "DEFAULT_THRESHOLD",
    "DIGEST_ALGOS",
    "DOWNSAMPLE_METHODS",
    "Dataset",
    "Deidentification",
    "ENDPOINT_TYPES",
    "EVENT_KINDS",
    "EXIT_ERROR",
    "EXIT_OK",
    "EXIT_USAGE",
    "EXTRAPOLATIONS",
    "FLOAT16_SAFE_VOXELS",
    "FORMAT_VERSION",
    "FORMS",
    "FRAME_ATTRS",
    "FeatureCacheHandle",
    "GEOMETRIC_KINDS",
    "GEOMETRY_RTOL",
    "GROUPABLE",
    "Grid",
    "Group",
    "ID_SOURCE",
    "IGNORE_ID",
    "INLINE_REQUIRED_BELOW",
    "INTERPOLATIONS",
    "IN_BAND_IGNORE_KINDS",
    "IO_FALLBACK_PREFIX",
    "IO_SEVERITIES",
    "ISSUE_SEVERITY",
    "Identity",
    "ImageHandle",
    "IndexPayload",
    "Issue",
    "KNOWN_UNITS",
    "LABEL_SAFE_METHODS",
    "LAST_ROW_TOL",
    "LATERALITY_VALUES",
    "LEGACY_BOX_SHIFT",
    "LEGACY_SCHEMA_VERSION",
    "LESION_PRESENCE",
    "LESION_VALUES",
    "LEVELS",
    "LOCALIZED_BBOX_FRACTION",
    "LabelClass",
    "LabelSet",
    "MANAGED_ROOT_ATTRS",
    "MANIFEST_SCHEMA",
    "MANIFEST_SUFFIXES",
    "MAX_CHUNK_BYTES",
    "MAX_CLASS_ID",
    "META_DATASET",
    "MIN_CHUNK_BYTES",
    "MM_PER_UNIT",
    "NIFTI_SUFFIXES",
    "ORTHONORMAL_TOL",
    "OVERSHOOT_LIMIT",
    "Observation",
    "OntologyCode",
    "OverlapStats",
    "PARTITIONS",
    "PRESENT",
    "PROFILES",
    "PSEUDONYM_SOURCE",
    "PatchDraws",
    "PatchSamplerHandle",
    "Provenance",
    "Pyramid",
    "QUALITY_FIELDS",
    "QUALITY_STATUS",
    "QualityRecord",
    "REASONS",
    "RELATIONS",
    "RESAMPLE_INTERPOLATIONS",
    "RESERVED_KINDS",
    "RESOLVED",
    "RFC3339_REGEX",
    "ROIS",
    "ROOT_DIGEST_ATTRS",
    "ROTATION_TOL",
    "ROW_STATUSES",
    "Relation",
    "SAMPLES_GROUP",
    "SAMPLING_PAIR_MODES",
    "SAMPLING_STRATEGIES",
    "SCOPES",
    "SCRUB_AGE_LIMIT",
    "SCRUB_DATE_KEYS",
    "SCRUB_FREE_TEXT",
    "SCRUB_IDENTIFYING_KEYS",
    "SCRUB_IDENTITY_RULES",
    "SCRUB_INTERNAL_REFERENCES",
    "SCRUB_MAX_DEPTH",
    "SCRUB_NOT_CHECKED",
    "SCRUB_PATH_REMOVED",
    "SCRUB_PROFILES",
    "SCRUB_PSEUDONYM_PREFIX",
    "SCRUB_QUASI_IDENTIFYING_KEYS",
    "SCRUB_STRICT_RULES",
    "SCRUB_UID_KEYS",
    "SCRUB_UNFIXABLE_LOCATIONS",
    "SELECTION_POLICIES",
    "SELECTION_STATUSES",
    "SEX_VALUES",
    "SLAB_BYTES",
    "SPACES",
    "SPARSE_FILL",
    "SPEC_ANNOTATION_ATTRS",
    "SPEC_GRID_ATTRS",
    "SPEC_IMAGE_ATTRS",
    "SPEC_TRANSFORM_ATTRS",
    "STANDARD_GROUPS",
    "STATES",
    "STATUSES",
    "STREAM_BYTES",
    "SUPPORTED_ORDERS",
    "SampleDocument",
    "SampleHandle",
    "SampleWriter",
    "SamplingIndex",
    "Skeleton",
    "SplitClaim",
    "StagedFileHandle",
    "TARGET_STATUSES",
    "TASKS",
    "TASK_CENSORING",
    "TASK_ROI",
    "TASK_SCHEMA",
    "TEMPORAL_TYPES",
    "TIMEPOINT_FIELDS",
    "TIME_UNITS",
    "TRANSCODABLE",
    "TRANSFORM_KINDS",
    "Timeline",
    "Timepoint",
    "Track",
    "Tracking",
    "TransformHandle",
    "TransformHeader",
    "UNEXAMINED",
    "VALUE_TYPES",
    "VECTOR_SPACES",
    "VISIBILITY",
    "VOXEL_KINDS",
    "__format_version__",
    "__version__",
    "_read_stats",
    "affine_summary",
    "agreement_box_iou",
    "agreement_compare",
    "agreement_compare_instances",
    "agreement_compare_voxel",
    "agreement_dice",
    "agreement_instance_json",
    "agreement_instance_mean_iou",
    "agreement_instance_record",
    "agreement_instance_value",
    "agreement_iou",
    "agreement_voxel_json",
    "agreement_voxel_record",
    "agreement_voxel_value",
    "amend",
    "analyse",
    "annotation_id",
    "annotation_to_masks",
    "apply_affine_to_box",
    "array_digest",
    "assertion_rows",
    "attrs_digest",
    "audit_anatomy_units",
    "audit_conflict_json",
    "audit_conflict_line",
    "audit_counts",
    "audit_json",
    "audit_leak_json",
    "audit_leak_line",
    "audit_membership_json",
    "audit_ok",
    "audit_partitions",
    "audit_set_ids",
    "audit_splits",
    "baseline_day_clock",
    "basis",
    "bench_benchmark_file",
    "bench_line",
    "bench_many_class_measurement",
    "bench_measurement_json",
    "bench_report",
    "bench_synthetic_many_class_sample",
    "bench_synthetic_pair",
    "bench_synthetic_sample",
    "bench_timed",
    "box_corners",
    "box_to_slices",
    "build_affine",
    "build_index",
    "cache_create",
    "cache_entry_json",
    "cache_event_entry_id",
    "cache_fitted_on",
    "cache_fitted_on_mismatches",
    "cache_open",
    "cache_owned_documents",
    "cache_report_json",
    "cache_schema_text",
    "cache_validate",
    "canonical_attrs",
    "canonical_json",
    "carries_instance_ids",
    "check_class_id",
    "check_contour_role",
    "check_orthonormal",
    "check_pyramid",
    "check_roundtrip",
    "check_scope",
    "check_slice_index",
    "check_space",
    "check_timestamp",
    "check_transform_id",
    "check_value_type",
    "chunk_report",
    "cli_command_tree",
    "cli_lookup_message",
    "cli_main",
    "cli_sample_file_refusal",
    "clinical_augment",
    "clinical_day_containing",
    "clinical_records",
    "clinical_round_micros",
    "clinical_schema_text",
    "clinical_select",
    "clinical_strip",
    "code_lookup",
    "codec_profiles",
    "codes_for_domain",
    "codes_table",
    "collect_digests",
    "compute_content_id",
    "conformance_build_case",
    "conformance_build_corpus",
    "conformance_case_json",
    "conformance_cases",
    "conformance_check_checksums",
    "conformance_load_manifest",
    "conformance_publish",
    "conformance_result_json",
    "conformance_run_corpus",
    "conformance_score",
    "conformance_score_companion",
    "conformance_summary",
    "contains_at",
    "cost_model",
    "create",
    "cubic_sample",
    "dataset_check",
    "dataset_check_format",
    "dataset_check_json",
    "dataset_class_stats_json",
    "dataset_compute_stats",
    "dataset_counts",
    "dataset_default_ratios",
    "dataset_digest",
    "dataset_entries_for",
    "dataset_entry_field",
    "dataset_entry_json",
    "dataset_find",
    "dataset_finding_json",
    "dataset_finding_line",
    "dataset_group_keys",
    "dataset_layout",
    "dataset_make_splits",
    "dataset_manifest_json",
    "dataset_manifest_legacy",
    "dataset_manifest_load",
    "dataset_manifest_save",
    "dataset_manifest_sha256",
    "dataset_manifest_stale",
    "dataset_moments_from_json",
    "dataset_moments_json",
    "dataset_moments_merge",
    "dataset_moments_std",
    "dataset_moments_update",
    "dataset_scan",
    "dataset_split_balance",
    "dataset_split_counts",
    "dataset_split_empty_folds",
    "dataset_split_fold_of",
    "dataset_split_json",
    "dataset_split_leaks",
    "dataset_split_load",
    "dataset_split_moved_from",
    "dataset_split_partition_of",
    "dataset_split_paths",
    "dataset_split_underfilled",
    "dataset_stats_class_weights",
    "dataset_stats_for",
    "dataset_stats_from_json",
    "dataset_stats_json",
    "dataset_stats_merge",
    "dataset_stats_normalization",
    "dataset_write_claims",
    "decompose_affine",
    "default_key",
    "default_task_for_kind",
    "derive_level_grid",
    "describe_filters",
    "detect_l3_bytes",
    "diagnose",
    "dice_agreement",
    "digest_bytes",
    "encode_affine",
    "encode_bitmask",
    "encode_boxes",
    "encode_bspline",
    "encode_classification",
    "encode_composite",
    "encode_contours",
    "encode_displacement",
    "encode_identity",
    "encode_instances",
    "encode_keypoints",
    "encode_labelmap",
    "encode_layers",
    "encode_mask",
    "encode_masks",
    "encode_mesh",
    "encode_obb",
    "encode_points",
    "encode_probmap",
    "encode_voxels",
    "extract",
    "field_chunks",
    "fit_chunks",
    "fix",
    "fix_paths",
    "folding_fraction",
    "frame_graph",
    "frames_of_timepoint",
    "from_keys",
    "greedy_colour",
    "grid_chunks",
    "group_digest",
    "imaging_events_from_timepoints",
    "index_to_world",
    "inside_extent",
    "instance_id_dtype",
    "instances_from_masks",
    "integrity_diagnosis_json",
    "integrity_repair_json",
    "integrity_verify_json",
    "io_contradictions",
    "io_days_from_baseline",
    "io_group_by_subject",
    "io_note_instance_ids",
    "io_note_json",
    "io_note_line",
    "io_output_name",
    "io_report_format",
    "io_report_json",
    "io_report_ok",
    "io_sanitize_key",
    "io_sanitize_stem",
    "is_bulk",
    "is_collection",
    "is_orthonormal",
    "is_probability",
    "is_proper_rotation",
    "jacobian_determinant",
    "label_dtype_size",
    "labels_check",
    "layers_from_colouring",
    "legacy_build_label_set",
    "legacy_is",
    "legacy_load_sidecar",
    "legacy_migrate",
    "legacy_migrate_paths",
    "legacy_read_meta",
    "legacy_read_sample",
    "legacy_write_sidecar",
    "linear_part",
    "linear_sample",
    "lossless_as_int16",
    "masks_equal",
    "new_document",
    "nifti_for_export",
    "nifti_from",
    "nifti_import_seg",
    "nifti_read",
    "nifti_read_stored",
    "nifti_to",
    "nifti_write",
    "nifti_write_field",
    "normalize_masks",
    "occupancy",
    "open_any",
    "open_collection",
    "open_sample",
    "optimize_chunks",
    "pack",
    "parse_digest",
    "payload_to_masks",
    "polygon_depths",
    "profile_family",
    "pyramid_factors",
    "quality_from_json",
    "quality_to_json",
    "raw_chunks",
    "read_document",
    "read_document_text",
    "read_grid",
    "read_grids",
    "read_indices",
    "read_itk_h5",
    "recompress",
    "recompress_paths",
    "refuse_outside",
    "registry_available",
    "registry_describe",
    "registry_load",
    "registry_load_file",
    "registry_register",
    "registry_unregister",
    "relative_path",
    "resample",
    "resolve_between",
    "resolve_profile_name",
    "rules_for",
    "same_file",
    "sample_field",
    "sampling_check_pair_mode",
    "sampling_coerce_patch_size",
    "sampling_grid_patches",
    "sampling_pairs",
    "sampling_window_around",
    "schema",
    "schema_text",
    "scrub_apply",
    "scrub_check",
    "scrub_finding",
    "scrub_finding_json",
    "scrub_finding_line",
    "scrub_pseudonymise",
    "scrub_report_format",
    "scrub_report_json",
    "scrub_report_ok",
    "scrub_report_open_identity",
    "scrub_scan",
    "scrub_scan_document",
    "select_encoding",
    "slices_to_box",
    "source_check",
    "source_default_id",
    "source_pin",
    "source_resolve",
    "spatial_chunk_for",
    "splits_from_json",
    "stage_file",
    "stale_index_entries",
    "storage_dtype",
    "storage_recompress_json",
    "storage_recompress_line",
    "subtrees_identical",
    "target_registration_error",
    "task_fingerprints",
    "task_from_cohort",
    "task_load_document",
    "task_normalize",
    "task_notes",
    "task_preflight",
    "task_reconcile",
    "task_repin",
    "task_row_fingerprint",
    "task_row_fingerprints",
    "task_save",
    "task_schema_text",
    "task_sources_outside",
    "task_subjects_digest",
    "task_validate",
    "to_world_vectors",
    "transcode",
    "transcode_payload",
    "unpack",
    "validate_against_schema",
    "validate_file",
    "validate_merge",
    "validate_paths",
    "validate_report_format",
    "validate_report_json",
    "validate_report_positions",
    "validate_root",
    "validate_sample_key",
    "verify_object",
    "verify_root",
    "voxel_volume",
    "world_to_index",
    "write_atomic",
    "write_itk_h5",
    "written_version",
]

ACTIVITY_FIELDS: frozenset[str]
ACTIVITY_TYPES: tuple[str, ...]
AGENT_FIELDS: frozenset[str]
AGENT_TYPES: tuple[str, ...]
AGREEMENT_DEFAULT_IOU: float
AGREEMENT_OBJECT_KINDS: tuple[str, ...]
ANNOTATION_KINDS: tuple[str, ...]
ASSESSMENT_SYSTEM: str
ATTESTED_GROUPS: tuple[str, ...]
AXIS_KINDS: tuple[str, ...]
BACKGROUND_ID: int
BENCH_MANY_CLASSES: int
BENCH_TARGETS: dict[str, tuple[Any, ...]]
BITS_PER_PLANE: int
BLOSC2_FILTER_ID: int
BLOSC_FILTER_ID: int
BUILTIN_FILTER_IDS: tuple[int, ...]
BULK_MIN_BYTES: int
CACHE_LEVELS: tuple[str, ...]
CACHE_SAFETY: float
CACHE_SCHEMA: str
CACHE_SUFFIX: str
CHECK_CODES: dict[str, str]
CHECK_CODE_SEVERITY: dict[str, str]
CHECK_SEVERITIES: tuple[str, ...]
CLINICAL_PROFILE: str
CLINICAL_SCHEMA: str
CLOCK_REFERENCES: tuple[str, ...]
CLOSURES: tuple[str, ...]
CODE_DOMAINS: tuple[str, ...]
COLLECTION_SUFFIX: str
COMPANION_CODES: dict[str, str]
COMPARATORS: tuple[str, ...]
COMPRESS_MIN_BYTES: int
CONFORMANCE_CHECKSUMS: str
CONFORMANCE_SCHEMA: str
CONFORMANCE_SEED: int
CONTOUR_ROLES: tuple[str, ...]
COORD_SYSTEMS: tuple[str, ...]
DATE_REGEX: str
DEFAULT_ALGO: str
DEFAULT_L3_BYTES: int
DEFAULT_MAX_COORDS: int
DEFAULT_OCCUPANCY_FACTOR: int
DEFAULT_ORDER: int
DEFAULT_PATCH: int
DEFAULT_PROFILE: str
DEFAULT_THRESHOLD: float
DIGEST_ALGOS: tuple[str, ...]
DOWNSAMPLE_METHODS: tuple[str, ...]
ENDPOINT_TYPES: tuple[str, ...]
EVENT_KINDS: tuple[str, ...]
EXIT_ERROR: int
EXIT_OK: int
EXIT_USAGE: int
EXTRAPOLATIONS: tuple[str, ...]
FLOAT16_SAFE_VOXELS: float
FORMAT_VERSION: str
FORMS: tuple[str, ...]
FRAME_ATTRS: tuple[tuple[str, tuple[str, ...]], ...]
GEOMETRIC_KINDS: tuple[str, ...]
GEOMETRY_RTOL: float
GROUPABLE: tuple[str, ...]
ID_SOURCE: str
IGNORE_ID: int
INLINE_REQUIRED_BELOW: int
INTERPOLATIONS: tuple[str, ...]
IN_BAND_IGNORE_KINDS: tuple[str, ...]
IO_FALLBACK_PREFIX: str
IO_SEVERITIES: tuple[str, ...]
ISSUE_SEVERITY: tuple[str, ...]
KNOWN_UNITS: tuple[str, ...]
MM_PER_UNIT: dict[str, float]
LABEL_SAFE_METHODS: tuple[str, ...]
LAST_ROW_TOL: float
LATERALITY_VALUES: tuple[str, ...]
LEGACY_BOX_SHIFT: float
LEGACY_SCHEMA_VERSION: str
LESION_PRESENCE: str
LESION_VALUES: tuple[str, ...]
LEVELS: tuple[Literal["structural", "semantic", "integrity", "strict"], ...]
LOCALIZED_BBOX_FRACTION: float
MANAGED_ROOT_ATTRS: tuple[str, ...]
MANIFEST_SCHEMA: str
MANIFEST_SUFFIXES: tuple[str, ...]
MAX_CHUNK_BYTES: float
MAX_CLASS_ID: int
META_DATASET: str
MIN_CHUNK_BYTES: float
NIFTI_SUFFIXES: tuple[str, ...]
ORTHONORMAL_TOL: float
OVERSHOOT_LIMIT: float
PARTITIONS: tuple[str, ...]
PRESENT: str
PROFILES: tuple[str, ...]
PSEUDONYM_SOURCE: str
QUALITY_FIELDS: frozenset[str]
QUALITY_STATUS: tuple[str, ...]
REASONS: tuple[str, ...]
RELATIONS: tuple[str, ...]
RESAMPLE_INTERPOLATIONS: tuple[str, ...]
RESERVED_KINDS: tuple[str, ...]
RESOLVED: str
RFC3339_REGEX: str
ROIS: tuple[str, ...]
ROOT_DIGEST_ATTRS: tuple[str, ...]
ROTATION_TOL: float
ROW_STATUSES: tuple[str, ...]
SAMPLES_GROUP: str
SAMPLING_PAIR_MODES: tuple[str, ...]
SAMPLING_STRATEGIES: tuple[str, ...]
SCOPES: tuple[str, ...]
SCRUB_AGE_LIMIT: float
SCRUB_DATE_KEYS: frozenset[str]
SCRUB_FREE_TEXT: int
SCRUB_IDENTIFYING_KEYS: frozenset[str]
SCRUB_IDENTITY_RULES: tuple[str, ...]
SCRUB_INTERNAL_REFERENCES: tuple[str, ...]
SCRUB_MAX_DEPTH: int
SCRUB_NOT_CHECKED: tuple[str, ...]
SCRUB_PATH_REMOVED: str
SCRUB_PROFILES: tuple[str, ...]
SCRUB_PSEUDONYM_PREFIX: str
SCRUB_QUASI_IDENTIFYING_KEYS: frozenset[str]
SCRUB_STRICT_RULES: tuple[str, ...]
SCRUB_UID_KEYS: frozenset[str]
SCRUB_UNFIXABLE_LOCATIONS: tuple[str, ...]
SELECTION_POLICIES: tuple[str, ...]
SELECTION_STATUSES: tuple[str, ...]
SEX_VALUES: tuple[str, ...]
SLAB_BYTES: int
SPACES: tuple[str, ...]
SPARSE_FILL: float
SPEC_ANNOTATION_ATTRS: tuple[str, ...]
SPEC_GRID_ATTRS: tuple[str, ...]
SPEC_IMAGE_ATTRS: tuple[str, ...]
SPEC_TRANSFORM_ATTRS: tuple[str, ...]
STANDARD_GROUPS: tuple[str, ...]
STATES: tuple[str, ...]
STATUSES: tuple[str, ...]
STREAM_BYTES: int
SUPPORTED_ORDERS: tuple[int, ...]
TARGET_STATUSES: tuple[str, ...]
TASKS: tuple[str, ...]
TASK_CENSORING: tuple[str, ...]
TASK_ROI: tuple[str, ...]
TASK_SCHEMA: str
TEMPORAL_TYPES: tuple[str, ...]
TIMEPOINT_FIELDS: frozenset[str]
TIME_UNITS: tuple[str, ...]
TRANSCODABLE: tuple[str, ...]
TRANSFORM_KINDS: tuple[str, ...]
UNEXAMINED: str
VALUE_TYPES: tuple[str, ...]
VECTOR_SPACES: tuple[str, ...]
VISIBILITY: dict[int, str]
VOXEL_KINDS: tuple[str, ...]
__format_version__: str
__version__: str

def _read_stats() -> dict[str, int]:
    """The engine's dataset reads by kind (`whole`, `window`, `hyperslab`,
    `rows`, `strings`) since the last call, and the counts set back to zero.

    Private, for the tests that hold a read path to its number of reads: the
    reads are the engine's, so a count of `h5py` calls sees none of them.
    """
    ...

def affine_summary(affine: npt.ArrayLike) -> dict[str, Any]: ...
def agreement_box_iou(a: Any, b: Any) -> float:
    """IoU of two `(S, 2)` boxes in the same space.  Refused for boxes of
    different numbers of axes or of none (`ValueError`), and for a bound that
    is not finite or a `lo` above its `hi` (E406).
    """
    ...

def agreement_compare(
    a: Any,
    b: Any,
    *,
    metric: str | None = None,
    threshold: float | None = None,
    classes: Any = None,
    map_classes: Any = None,
    match_by: str = "auto",
) -> tuple[str, dict[Any, Any]]:
    """The comparison two annotations' kinds support:
    `("voxel" | "instance", fields)`.
    """
    ...

def agreement_compare_instances(
    a: Any,
    b: Any,
    *,
    threshold: float = ...,
    classes: Any = None,
    map_classes: Any = None,
    match_by: str = "auto",
) -> dict[Any, Any]:
    """Object-level agreement between two annotations: the fields of an
    `InstanceAgreement`.
    """
    ...

def agreement_compare_voxel(
    a: Any, b: Any, *, metric: str = ..., classes: Any = None, map_classes: Any = None
) -> dict[Any, Any]:
    """Per-class Dice or IoU between two voxel annotations: the fields of a
    `VoxelAgreement`.
    """
    ...

def agreement_dice(a: Any, b: Any) -> float | None:
    """Sørensen--Dice, or `None` when both masks are empty."""
    ...

def agreement_instance_json(agreement: Any) -> Any: ...
def agreement_instance_mean_iou(agreement: Any) -> float | None: ...
def agreement_instance_record(
    agreement: Any, against: str | None = None
) -> Agreement: ...
def agreement_instance_value(agreement: Any) -> float | None: ...
def agreement_iou(a: Any, b: Any) -> float | None:
    """Intersection over union, or `None` when both masks are empty."""
    ...

def agreement_voxel_json(agreement: Any) -> Any: ...
def agreement_voxel_record(agreement: Any, against: str | None = None) -> Agreement: ...
def agreement_voxel_value(agreement: Any) -> float | None: ...
def amend(path: str | os.PathLike[str], *, codec: str | None = None) -> SampleWriter:
    """Copy-on-write amend: build a new file from the old and replace it.

    On POSIX, anything holding the old file open keeps reading the old inode.
    Windows cannot replace an open file: close every reader of it before the
    commit, whose rename otherwise fails and leaves the file as it was.
    """
    ...

def analyse(
    masks: Any, spatial_shape: tuple[int, ...] | None = None
) -> OverlapStats: ...
def annotation_id(reference: str) -> str: ...
def annotation_to_masks(
    annotation: VoxelAnnotation, classes: Sequence[int | str] | None = None
) -> dict[int, npt.NDArray[np.bool_]]:
    """Every class of a voxel annotation as a full mask."""
    ...

def apply_affine_to_box(affine: npt.ArrayLike, box: Any) -> npt.NDArray[np.float64]: ...
def array_digest(
    path: str, array: np.ndarray[Any, np.dtype[Any]], algo: str = "sha256"
) -> str: ...
def assertion_rows(labels: Any) -> tuple[Any, ...]:
    """`(classes, values, scope_ids, schemes, scheme_values)` from a mapping or
    assertion rows (§9).
    """
    ...

def attrs_digest(obj: Any, names: Iterable[str], algo: str = "sha256") -> str: ...
def audit_anatomy_units(pairs: Any) -> dict[Any, Any]:
    """Grouping key -> every grouping key sharing anatomy with it, sorted."""
    ...

def audit_conflict_json(conflict: Any) -> Any: ...
def audit_conflict_line(conflict: Any) -> str: ...
def audit_counts(audit: Any) -> Any: ...
def audit_json(audit: Any) -> Any: ...
def audit_leak_json(leak: Any) -> Any: ...
def audit_leak_line(leak: Any) -> str: ...
def audit_membership_json(membership: Any) -> Any: ...
def audit_ok(audit: Any) -> bool: ...
def audit_partitions(audit: Any, set_id: str) -> dict[Any, Any]:
    """`partition -> sample ids` for one split."""
    ...

def audit_set_ids(audit: Any) -> tuple[Any, ...]: ...
def audit_splits(paths: Sequence[str | os.PathLike[str]]) -> dict[Any, Any]:
    """Read every file's claims and cross-check them (§12.3): the audit's fields."""
    ...

def baseline_day_clock(clock_id: str) -> dict[str, Any]:
    """The clock `imaging_events_from_timepoints` measures on."""
    ...

def basis(order: int, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The B-spline basis weights at `t`: `(order + 1, *t.shape)`."""
    ...

def bench_benchmark_file(
    path: str | os.PathLike[str],
    *,
    annotation: str | None = None,
    patch: int = 64,
    repeats: int = 20,
) -> list[Any]: ...
def bench_line(measurement: Any) -> str:
    """One measurement's report line (`str(measurement)`), given as a
    `Measurement` or its JSON form.
    """
    ...

def bench_many_class_measurement(
    path: str | os.PathLike[str], *, patch: int = 64, repeats: int = 20
) -> Any: ...
def bench_measurement_json(measurement: Any) -> Any:
    """A measurement as `bench --json` prints it: a NaN as `null`."""
    ...

def bench_report(measurements: Any) -> str:
    """The text report of measurements, given as `Measurement`s or their JSON
    form.
    """
    ...

def bench_synthetic_many_class_sample(
    directory: str | os.PathLike[str], *, classes: int = ...
) -> str: ...
def bench_synthetic_pair(
    directory: str | os.PathLike[str],
    *,
    shape: Sequence[int] = ...,
    codec: str = ...,
    seed: int = 20260815,
) -> str: ...
def bench_synthetic_sample(
    directory: Any,
    *,
    shape: Any = ...,
    classes: Any = 8,
    codec: Any = ...,
    index: Any = True,
    seed: Any = 20260815,
    name: Any = ...,
) -> Any: ...
def bench_timed(fn: Any, *, repeats: int = 20, warmup: int = 3) -> float:
    """Median milliseconds per call of a Python callable."""
    ...

def box_corners(box: Any) -> npt.NDArray[np.float64]: ...
def box_to_slices(
    box: Any, shape: Sequence[int] | None = None
) -> tuple[slice, ...]: ...
def build_affine(
    spacing: npt.ArrayLike, origin: npt.ArrayLike, direction: npt.ArrayLike
) -> npt.NDArray[np.float64]: ...
def build_index(
    annotation: VoxelAnnotation,
    *,
    classes: Sequence[int | str] | None = None,
    max_coords: int = ...,
    occupancy: int | None = ...,
    seed: int = 0,
    source_digest: str | None = None,
) -> IndexPayload:
    """Compute the sampling index of one voxel annotation (not written)."""
    ...

def cache_create(
    path: str | os.PathLike[str], header_doc: Mapping[str, Any]
) -> CacheWriterHandle:
    """Start writing a feature cache (`medh5.cache/1`)."""
    ...

def cache_event_entry_id(content_id: str, event_id: str) -> str: ...
def cache_fitted_on(doc: Any, partition: str) -> dict[str, Any]: ...
def cache_fitted_on_mismatches(doc: Any, fitted: Mapping[str, Any]) -> list[str]:
    """Why a `fitted_on` record is not this task's training partition (T405)."""
    ...

def cache_open(path: str | os.PathLike[str]) -> FeatureCacheHandle: ...
def cache_schema_text() -> str: ...
def cache_entry_json(doc: Any) -> dict[str, Any]:
    """An entry as the engine reads it, written back."""
    ...

def cache_report_json(report: Any) -> dict[str, Any]:
    """A `CacheReport` dataclass's engine JSON."""
    ...

def cache_owned_documents(
    source: Any,
    base: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, list[str]]:
    """Each event version of a source and the documents it owns."""
    ...

def cache_validate(
    path: str | os.PathLike[str],
    base: str | os.PathLike[str] | None = None,
    task: Any = None,
    task_base: str | os.PathLike[str] | None = None,
    check_rows: bool = True,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate a cache; with a task, also against that task (and, with
    ``check_rows``, against what its preflight admits)."""
    ...

def canonical_attrs(obj: Any, names: Iterable[str]) -> str:
    """Canonical JSON over an object's named attributes (§13.2)."""
    ...

def canonical_json(doc: Mapping[str, Any]) -> bytes: ...
def carries_instance_ids(annotation: Any) -> bool:
    """Whether an annotation carries object identity (`instance_ids`) to join on."""
    ...

def check_class_id(class_id: Any) -> int: ...
def check_contour_role(role: str) -> str:
    """A contour role, or E411."""
    ...

def polygon_depths(polygons: Any) -> list[int]:
    """Each of one class's polygons in one plane, its depth: how many of the
    others enclose it (§8.6).  Each is `(V, 2)` in-plane coordinates."""
    ...

def check_orthonormal(
    direction: npt.ArrayLike, tol: float = ..., *, what: str = "direction"
) -> npt.NDArray[np.float64]: ...
def check_pyramid(
    base: Grid, levels: Sequence[Grid], factors: npt.ArrayLike, *, rtol: float = ...
) -> list[str]: ...
def check_roundtrip(
    payload: AnnotationPayload,
    to_kind: str,
    *,
    spatial_shape: tuple[int, ...] | None = None,
) -> bool: ...
def check_scope(scope: str) -> str: ...
def check_slice_index(
    planes: Any,
    n_boxes: int,
    *,
    boxes: npt.NDArray[np.float64] | None = None,
    shape: Sequence[int] | None = None,
) -> str | None: ...
def check_space(space: str) -> str: ...
def check_timestamp(value: str, *, where: Any) -> str: ...
def check_transform_id(transform_id: str) -> str: ...
def check_value_type(value_type: str) -> str: ...
def chunk_report(
    shape: Sequence[int], chunks: Sequence[int], itemsize: int
) -> dict[str, Any]: ...
def cli_command_tree() -> Any:
    """The grammar as data: `{"options", "positionals", "commands"}`."""
    ...

def cli_lookup_message(text: str) -> str:
    """What the command line prints for a failed lookup whose message is `text`:
    the message when it is a sentence, the key named otherwise.
    """
    ...

def cli_main(argv: Sequence[str], host: Any) -> int:
    """Run the command line on `argv` (without the program name); the exit code."""
    ...

def cli_sample_file_refusal(path: str) -> str | None:
    """Why `path` is not worked on as one sample file, as every native command
    words it (a collection, with the way to its members); `None` otherwise.
    """
    ...

def clinical_augment(
    path: str | os.PathLike[str],
    records: Any,
    out: str | os.PathLike[str] | None = None,
) -> dict[str, Any]: ...
def clinical_records(records: Any) -> dict[str, Any]:
    """Check a logical-record bundle against its schema and parse it back."""
    ...

def clinical_day_containing(t: int) -> tuple[int, int]:
    """The clock day holding `t`, as inclusive `(lo, hi)` bounds."""
    ...

def clinical_round_micros(value: float, unit_us: int) -> int:
    """`value` units of `unit_us` microseconds, rounded half to even."""
    ...

def clinical_schema_text() -> str: ...
def clinical_select(
    events: Sequence[Any], links: Sequence[Any], cutoff_us: int, policy: Any = None
) -> dict[str, Any]:
    """Select from records not read from a file: `events` and `links` as dicts."""
    ...

def clinical_strip(
    path: str | os.PathLike[str], out: str | os.PathLike[str]
) -> dict[str, Any]: ...
def code_lookup(name: str) -> str:
    """A code's name, or the engine's `KeyError` for one the table lacks."""
    ...

def codec_profiles() -> list[Any]:
    """The codec profiles, as plain data (`medh5.storage` builds its
    `CodecProfile` values from this).
    """
    ...

def codes_for_domain(domain: str) -> list[str]:
    """The codes of one domain, in table order."""
    ...

def codes_table() -> str:
    """The normative diagnostic code table (§15.2), as JSON text."""
    ...

def collect_digests(root: Group, skip: Sequence[str] = ...) -> dict[str, str]:
    """The `digest` of every dataset under a sample root that carries one."""
    ...

def compute_content_id(
    root: Group,
    attr_names: Mapping[str, Sequence[str]] | None = None,
    *,
    algo: str | None = None,
) -> str:
    """The Merkle root over stored digests, `meta` and canonical attributes."""
    ...

def conformance_build_case(name: str, path: str | os.PathLike[str]) -> None:
    """Write one corpus case's file to `path`."""
    ...

def conformance_build_corpus(
    outdir: str | os.PathLike[str], *, names: Sequence[str] | None = None
) -> str: ...
def conformance_case_json(case: Any) -> Any:
    """A case's entry in `expected.json`."""
    ...

def conformance_cases() -> list[Any]:
    """Every case, as its manifest record, in corpus order."""
    ...

def conformance_check_checksums(root: str | os.PathLike[str]) -> list[str]: ...
def conformance_load_manifest(root: str | os.PathLike[str]) -> Any: ...
def conformance_publish(
    outdir: str | os.PathLike[str], *, names: Sequence[str] | None = None
) -> str: ...
def conformance_result_json(result: Any) -> Any:
    """A result as `conformance run --json` prints it."""
    ...

def conformance_run_corpus(
    outdir: str | os.PathLike[str], *, names: Sequence[str] | None = None
) -> list[Any]: ...
def conformance_score(
    root: str | os.PathLike[str],
    submitted: Any,
    *,
    format_version: str | None = None,
    codes_revision: int | None = None,
) -> list[Any]: ...
def conformance_score_companion(
    root: str | os.PathLike[str], submitted: Any
) -> list[Any]: ...
def conformance_summary(results: Any) -> Any:
    """The `--json` summary of a run or a score."""
    ...

def contains_at(values: npt.NDArray[Any], threshold: float) -> npt.NDArray[np.bool_]:
    """`values >= threshold`, decided in the stored precision (§7.5)."""
    ...

def cost_model(stats: OverlapStats, *, ignore: bool = False) -> CostModel: ...
def create(
    path: str | os.PathLike[str],
    *,
    sample_id: str | None = None,
    subject_id: str | None = None,
    codec: str = "balanced",
    profiles: Sequence[str] | None = None,
) -> SampleWriter:
    """Create a new sample; use as a context manager, or call `commit()`.

    Without `sample_id`, the sample id is the file name up to its first dot,
    and `subject_id` defaults to the sample id.  Where that is not the file's
    stem (`case.v2.medh5`) it warns (`FutureWarning`): medh5 3.0 takes the
    stem.  A name with nothing before its first dot is refused.
    """
    ...

def cubic_sample(
    field: npt.NDArray[Any],
    coords: npt.NDArray[np.float64],
    *,
    extrapolation: str = "zero",
) -> npt.NDArray[np.float64]: ...
def dataset_check(
    manifest: Any, *, set_id: str | None = None, deep: bool = False
) -> Any: ...
def dataset_check_format(report: Any) -> str: ...
def dataset_check_json(report: Any) -> Any: ...
def dataset_class_stats_json(s: Any) -> Any:
    """`ClassStats.to_json`: the engine's JSON for one class."""
    ...

def dataset_compute_stats(
    paths: Sequence[str | os.PathLike[str]],
    *,
    images: Any = None,
    annotations: Any = None,
    workers: int = 1,
    sample_stride: int = 1,
    physical: bool = True,
) -> dict[Any, Any]: ...
def dataset_counts(entries: Any, by: str) -> dict[Any, Any]: ...
def dataset_default_ratios() -> dict[Any, Any]: ...
def dataset_digest(dataset: Any, path: str | None = None, algo: str = "sha256") -> str:
    """The digest of a stored dataset over its decompressed content (§13.1)."""
    ...

def dataset_entries_for(path: str | os.PathLike[str]) -> list[Any]: ...
def dataset_entry_field(doc: Any, dotted: str) -> Any:
    """A field by its dotted name (`cohort.site_id`), refused when not a field."""
    ...

def dataset_entry_json(doc: Any) -> Any:
    """An entry's JSON as the engine writes it (key order, omitted `None`s)."""
    ...

def dataset_find(
    root: str | os.PathLike[str], *, suffixes: Sequence[str] | None = None
) -> list[str]: ...
def dataset_finding_json(finding: Any) -> Any: ...
def dataset_finding_line(finding: Any) -> str: ...
def dataset_group_keys(entries: Any, by: str) -> list[str]:
    """The grouping key of each entry, in order (`str(entry.field(by))`)."""
    ...

def dataset_layout(
    shape: Sequence[int],
    itemsize: int,
    *,
    profile: str | None = None,
    role: str = "image",
    chunks: Sequence[int] | None = None,
) -> dict[Any, Any]:
    """The layout one dataset gets under a profile: `{}` when stored contiguous,
    else `{"chunks": ..., "codec": ...}`.
    """
    ...

def dataset_make_splits(
    manifest: Any,
    *,
    set_id: str = ...,
    group_by: str = ...,
    stratify_by: str | None = None,
    ratios: Any = None,
    k_folds: int | None = None,
    seed: int = 0,
    algorithm: str | None = None,
    previous: Any = None,
) -> tuple[Any, int | None]:
    """`(split JSON, groups medh5.split/2 places elsewhere)`: the count only when
    `algorithm` is left out and there is no `previous` split, else `None`.
    """
    ...

def dataset_manifest_json(doc: Any) -> Any:
    """The full manifest JSON (with its `sha256`)."""
    ...

def dataset_manifest_legacy(doc: Any) -> bool:
    """Whether a manifest is legacy (no `schema`); an unknown schema is refused."""
    ...

def dataset_manifest_load(
    path: str | os.PathLike[str], *, base: str | os.PathLike[str] | None = None
) -> tuple[Any, bool]:
    """`(manifest JSON, legacy and read otherwise than 3.0 will read it)`."""
    ...

def dataset_manifest_save(
    doc: Any, path: str | os.PathLike[str], *, paths: str = "keep"
) -> str:
    """Written with paths as `paths` says: `"keep"`, `"relative"` or `"absolute"`."""
    ...

def dataset_manifest_sha256(doc: Any) -> str: ...
def dataset_manifest_stale(doc: Any) -> list[str]: ...
def dataset_moments_from_json(doc: Any) -> tuple[int, float, float, float, float, int]:
    """`Moments.from_json`: the state the JSON form describes."""
    ...

def dataset_moments_json(m: Any) -> Any: ...
def dataset_moments_merge(
    m: Any, other: Any
) -> tuple[int, float, float, float, float, int]:
    """`Moments.merge` (Chan--Golub--LeVeque): the merged state."""
    ...

def dataset_moments_std(m: Any) -> float: ...
def dataset_moments_update(
    m: Any, values: Any
) -> tuple[int, float, float, float, float, int]:
    """`Moments.update`: the state after folding in `values`."""
    ...

def dataset_scan(
    root: str | os.PathLike[str],
    *,
    suffixes: Sequence[str] | None = None,
    on_error: str = "warn",
) -> tuple[Any, ...]:
    """`(manifest JSON, failures)`; `on_error="raise"` re-raises the first."""
    ...

def dataset_split_balance(doc: Any) -> dict[Any, Any]: ...
def dataset_split_counts(doc: Any) -> dict[Any, Any]: ...
def dataset_split_empty_folds(doc: Any) -> list[int]: ...
def dataset_split_fold_of(doc: Any, entry: Any) -> int | None: ...
def dataset_split_json(doc: Any) -> Any: ...
def dataset_split_leaks(doc: Any) -> list[str]: ...
def dataset_split_load(path: str | os.PathLike[str]) -> Any: ...
def dataset_split_moved_from(doc: Any, other: Any) -> dict[str, tuple[str, str]]:
    """The groups `doc` places elsewhere than `other` does: `{group: (was, now)}`."""
    ...

def dataset_split_partition_of(doc: Any, entry: Any) -> str | None: ...
def dataset_split_paths(doc: Any, partition: str) -> list[str]: ...
def dataset_split_underfilled(doc: Any) -> list[str]: ...
def dataset_stats_class_weights(
    s: Any, *, scheme: str = "inverse_frequency"
) -> dict[Any, Any]: ...
def dataset_stats_for(
    path: str | os.PathLike[str],
    *,
    images: Any = None,
    annotations: Any = None,
    sample_stride: int = 1,
    physical: bool = True,
) -> dict[Any, Any]: ...
def dataset_stats_from_json(doc: Any) -> dict[Any, Any]:
    """`DatasetStats.from_json`: the state the JSON form describes."""
    ...

def dataset_stats_json(s: Any) -> Any: ...
def dataset_stats_merge(s: Any, other: Any) -> dict[Any, Any]: ...
def dataset_stats_normalization(s: Any, image_key: str) -> tuple[float, float]: ...
def dataset_write_claims(
    split: Any,
    manifest: Any,
    *,
    assigned_by: str | None = None,
    fold: int | None = None,
) -> list[str]: ...
def decompose_affine(
    affine: npt.ArrayLike,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]: ...
def default_key(path: str | os.PathLike[str]) -> str: ...
def default_task_for_kind(kind: str) -> str | None: ...
def derive_level_grid(
    base: Grid,
    factors: Sequence[float],
    grid_id: str,
    *,
    shape: Sequence[int] | None = None,
) -> Grid: ...
def describe_filters(dataset: Any) -> str:
    """A dataset's actual HDF5 filter pipeline, e.g. `blosc2:zstd:3+shuffle`."""
    ...

def detect_l3_bytes() -> int: ...
def diagnose(path: str | os.PathLike[str]) -> Any: ...
def dice_agreement(
    per_class: Mapping[int, float], against: str | None = None
) -> Agreement: ...
def digest_bytes(payload: bytes, algo: str = "sha256") -> str: ...
def encode_affine(matrix: npt.ArrayLike) -> AnnotationPayload: ...
def encode_bitmask(
    masks: Any, spatial_shape: tuple[int, ...] | None = None
) -> AnnotationPayload: ...
def encode_boxes(
    boxes: npt.ArrayLike,
    class_ids: Sequence[int],
    *,
    instance_ids: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
    attributes: Sequence[Mapping[str, Any]] | None = None,
    slice_index: Sequence[int] | None = None,
) -> AnnotationPayload: ...
def encode_bspline(
    control_points: npt.ArrayLike,
    *,
    cp_grid: str,
    order: int = ...,
    vector_space: str = "world",
) -> AnnotationPayload: ...
def encode_classification(
    labels: Any,
    *,
    scope: str = "sample",
    multilabel: bool = True,
    scope_ids: Sequence[int] | None = None,
    schemes: Sequence[str] | None = None,
    scheme_values: Sequence[str] | None = None,
) -> AnnotationPayload: ...
def encode_composite(components: Sequence[str]) -> AnnotationPayload: ...
def encode_contours(
    polygons: Sequence[Polygon], *, ndim: int | None = None
) -> AnnotationPayload: ...
def encode_displacement(
    field: Any,
    *,
    field_grid: str,
    vector_space: str = "world",
    interpolation: str = "linear",
    extrapolation: str = "zero",
    dtype: Any = None,
) -> AnnotationPayload: ...
def encode_identity() -> AnnotationPayload: ...
def encode_instances(
    objects: Any,
    spatial_shape: Any = None,
    *,
    store_masks: bool = True,
    class_ids: Any = None,
) -> AnnotationPayload: ...
def encode_keypoints(
    points: npt.ArrayLike,
    keypoint_class_ids: Sequence[int],
    class_ids: Sequence[int],
    *,
    visibility: npt.ArrayLike | None = None,
    instance_ids: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
    skeleton: str | None = None,
) -> AnnotationPayload: ...
def encode_labelmap(
    masks: Any,
    spatial_shape: tuple[int, ...] | None = None,
    *,
    ignore: npt.NDArray[np.bool_] | None = None,
    ignore_id: int = ...,
) -> AnnotationPayload:
    """Pack mutually exclusive class masks into one integer volume (§7.1).
    `ignore` is painted as `ignore_id`; a region that overlaps a class (E404:
    §7.7 puts it in a sibling `mask`), a region of another shape (E405), or an
    `ignore_id` outside 1..65535 or equal to a class id (E404) is refused.
    """
    ...

def encode_layers(
    masks: Any,
    spatial_shape: tuple[int, ...] | None = None,
    *,
    colouring: dict[int, int] | None = None,
    ignore: npt.NDArray[np.bool_] | None = None,
    ignore_id: int = ...,
) -> AnnotationPayload:
    """Colour the overlap graph and pack each colour into one labelmap (§7.2);
    `ignore` and `ignore_id` are checked and painted as `encode_labelmap`
    does.
    """
    ...

def encode_mask(mask: npt.NDArray[Any]) -> AnnotationPayload: ...
def encode_masks(
    masks: Mapping[int, npt.NDArray[np.bool_]],
    kind: str,
    spatial_shape: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> AnnotationPayload: ...
def encode_mesh(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    *,
    normals: npt.ArrayLike | None = None,
    vertex_class_ids: Sequence[int] | None = None,
    mesh_offsets: Sequence[int] | None = None,
    mesh_class_ids: Sequence[int] | None = None,
) -> AnnotationPayload: ...
def encode_obb(
    centers: npt.ArrayLike,
    sizes: npt.ArrayLike,
    rotations: npt.ArrayLike,
    class_ids: Sequence[int],
    *,
    instance_ids: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
    attributes: Sequence[Mapping[str, Any]] | None = None,
) -> AnnotationPayload: ...
def encode_points(
    points: npt.ArrayLike,
    *,
    class_ids: Sequence[int] | None = None,
    instance_ids: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    correspondence: str | None = None,
) -> AnnotationPayload: ...
def encode_probmap(
    probabilities: Any,
    spatial_shape: Any = None,
    *,
    dtype: Any = None,
    normalized: bool = False,
    threshold: float | None = None,
) -> AnnotationPayload: ...
def encode_voxels(
    masks: Any,
    spatial_shape: Any = None,
    *,
    encoding: str = "auto",
    ignore: Any = None,
    **kwargs: dict[Any, Any],
) -> tuple[Any, ...]:
    """Encode class masks, choosing the encoding by measurement when asked to:
    `(payload, stats)`.  An ignore region rides in band only under
    `labelmap`/`layers`, and only where it meets no class; anything else is
    refused (E404), because one payload cannot carry the §7.7 sibling mask.
    """
    ...

def extract(
    path: str | os.PathLike[str], key: str, out: str | os.PathLike[str]
) -> str: ...
def field_chunks(
    grid: Grid, shape: Sequence[int], itemsize: int
) -> tuple[int, ...] | None: ...
def fit_chunks(
    proposed: Sequence[int], shape: Sequence[int]
) -> tuple[int, ...] | None: ...
def fix(
    path: str | os.PathLike[str],
    *,
    rebuild_index: bool = False,
    rewrite_digests: bool = False,
    reason: str | None = None,
    performed_by: str | None = None,
    max_coords: int | None = None,
) -> Any: ...
def fix_paths(
    paths: Sequence[str | os.PathLike[str]],
    *,
    rebuild_index: bool = False,
    rewrite_digests: bool = False,
    reason: str | None = None,
    performed_by: str | None = None,
    max_coords: int | None = None,
) -> list[Any]: ...
def folding_fraction(determinants: npt.NDArray[np.float64]) -> float: ...
def frame_graph(transforms: Any) -> dict[Any, Any]:
    """Frame -> frames one hop away, as the resolver would walk them."""
    ...

def frames_of_timepoint(grids: Any, timepoint: str) -> tuple[Any, ...]:
    """The frames of every grid of one timepoint, in grid order."""
    ...

def from_keys(
    keys: Sequence[str], *, id: str, version: str = "1.0.0", start: int = 1
) -> LabelSet: ...
def greedy_colour(
    class_ids: Sequence[int], edges: frozenset[tuple[int, int]] | set[tuple[int, int]]
) -> dict[int, int]: ...
def grid_chunks(grid: Grid, itemsize: int, *, leading: int = 0) -> tuple[int, ...]: ...
def group_digest(
    group: Group, algo: str = "sha256", *, root: Group | None = None
) -> str:
    """The digest of an object group: its datasets and canonical attributes."""
    ...

def imaging_events_from_timepoints(path: str | os.PathLike[str]) -> list[Any]:
    """`(events, links, notes)`: imaging events from `days_from_baseline`."""
    ...

def index_to_world(
    affine: npt.ArrayLike, indices: npt.ArrayLike
) -> npt.NDArray[np.float64]: ...
def inside_extent(
    spatial: npt.ArrayLike, points: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]: ...
def instance_id_dtype(ids: Any) -> Any:
    """`uint32` unless an id needs the wider form (§7.4, §8.2)."""
    ...

def instances_from_masks(masks: Any, *, start_id: int = 1) -> list[Any]:
    """One object per class mask, ids from `start_id`: the field values of
    `InstanceInput`s, as `(class_id, instance_id, mask)` triples.  An id past
    the largest `uint64` is refused (E404).
    """
    ...

def integrity_diagnosis_json(diagnosis: Any) -> Any:
    """A diagnosis as `diagnose --json` prints it."""
    ...

def integrity_repair_json(repair: Any) -> Any:
    """A repair as `fix --json` prints it."""
    ...

def integrity_verify_json(result: Any) -> Any:
    """A verification's summary: its verdict, and what failed."""
    ...

def io_contradictions(occasions: Any) -> dict[Any, Any]:
    """Subject keys the sources contradict -> the facts that disagree."""
    ...

def io_days_from_baseline(dates: Sequence[str | None]) -> list[int | None]:
    """Days from the first visit for each date (`None` where one is missing)."""
    ...

def io_group_by_subject(
    occasions: Any, *, mode: str = "subject"
) -> tuple[list[Any], list[Any]]:
    """Group occasions into subjects: `(groups, notes)`, a group being
    `(subject_id, positions, ordered_by)` and the notes what grouping recorded.
    """
    ...

def io_note_instance_ids(subject_id: str, occasions: int) -> list[Any]:
    """What a merged group records about instance ids (§7.4): note dicts."""
    ...

def io_note_json(note: Any) -> Any: ...
def io_note_line(note: Any) -> str: ...
def io_output_name(
    subject_id: str, keys: Sequence[str], used: set[Any], safe: Any = None
) -> str:
    """A unique filename stem for one group; adds it to `used`."""
    ...

def io_report_format(report: Any, *, verbose: bool = False) -> str: ...
def io_report_json(report: Any) -> Any: ...
def io_report_ok(report: Any) -> bool: ...
def io_sanitize_key(name: Any, *, fallback: str = "class") -> str:
    """A label-set `key` from free text (§5.2): `^[a-z0-9][a-z0-9_]*$`."""
    ...

def io_sanitize_stem(text: Any, *, limit: int = 200) -> str:
    """A filename stem from free text: identifier characters only, truncated."""
    ...

def is_bulk(dataset: Any) -> bool:
    """Whether a dataset is large enough for the W902 warning."""
    ...

def is_collection(obj: Any) -> bool:
    """Whether a root declares itself a collection: a `Group`, a `Sample`, a
    collection, or a path.
    """
    ...

def is_orthonormal(direction: npt.ArrayLike, tol: float = ...) -> bool: ...
def is_probability(array: Any) -> bool:
    """Whether every value lies in `[0, 1]` (and there is at least one)."""
    ...

def is_proper_rotation(matrix: npt.ArrayLike, tol: float = ...) -> bool: ...
def jacobian_determinant(
    field: npt.NDArray[Any], grid: Grid, *, vector_space: str = "world"
) -> npt.NDArray[np.float64]: ...
def labels_check(paths: Sequence[str | os.PathLike[str]]) -> dict[str, Any]: ...
def label_dtype_size(class_ids: Sequence[int], *, ignore: bool = False) -> int: ...
def layers_from_colouring(
    colouring: Mapping[int, int],
) -> tuple[tuple[int, ...], ...]: ...
def legacy_build_label_set(paths: Any) -> tuple[LabelSet, list[Any]]:
    """One label set over a cohort: `(label_set, notes)`."""
    ...

def legacy_is(path: str | os.PathLike[str]) -> bool: ...
def legacy_load_sidecar(path: str | os.PathLike[str]) -> LabelSet: ...
def legacy_migrate(
    path: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    label_set: Any = None,
    codec: str = ...,
    report: Any = None,
) -> dict[Any, Any]:
    """Migrate one 0.x file; the report's fields (`report` carried into it)."""
    ...

def legacy_migrate_paths(
    paths: Any,
    outdir: str | os.PathLike[str],
    *,
    group_by: str = ...,
    subject_key: str | None = None,
    label_set: Any = None,
    codec: str = ...,
) -> dict[Any, Any]:
    """Migrate a cohort, minting one label set for all of it; the report's fields."""
    ...

def legacy_read_meta(path: str | os.PathLike[str]) -> dict[Any, Any]:
    """A 0.x file's metadata, as `LegacyMeta` fields."""
    ...

def legacy_read_sample(path: str | os.PathLike[str]) -> dict[Any, Any]:
    """A whole 0.x file, as `LegacySample` fields (`meta` as `LegacyMeta` fields)."""
    ...

def legacy_write_sidecar(label_set: Any, path: str | os.PathLike[str]) -> str: ...
def linear_part(grid: Grid) -> npt.NDArray[np.float64]: ...
def linear_sample(
    field: npt.NDArray[Any],
    coords: npt.NDArray[np.float64],
    *,
    extrapolation: str = "zero",
) -> npt.NDArray[np.float64]: ...
def lossless_as_int16(array: Any) -> bool:
    """Whether a float array would survive `int16` storage unchanged (W907)."""
    ...

def masks_equal(
    a: Mapping[int, npt.NDArray[np.bool_]], b: Mapping[int, npt.NDArray[np.bool_]]
) -> bool: ...
def new_document(
    sample_id: str,
    subject_id: str | None = None,
    *,
    timepoints: Timeline | Sequence[str] | None = None,
    **identity_fields: Any,
) -> SampleDocument: ...
def nifti_for_export(
    grid: Grid, data: Any
) -> tuple[npt.NDArray[Any], npt.NDArray[np.float64]]:
    """An array and affine in MEDH5 conventions as NIfTI wants them."""
    ...

def nifti_from(
    images: list[tuple[str, str]],
    out: str | os.PathLike[str],
    *,
    report: Any,
    masks: list[tuple[str, str]] = ...,
    sample_id: str | None = None,
    subject_id: str | None = None,
    modalities: list[tuple[str, str]] = ...,
    coord_system: str = ...,
    transpose: bool = ...,
    fourth_axis: str = ...,
    value_units: list[tuple[str, str]] = ...,
    codec: str = ...,
    annotated_classes: Any = None,
    assume_geometry: bool = ...,
    plane_as_volume: bool = ...,
    label_set: Any = None,
) -> None:
    """Write one sample from NIfTI volumes; the caller's report is updated."""
    ...

def nifti_import_seg(
    path: str | os.PathLike[str],
    masks: list[tuple[str, str]],
    *,
    report: Any,
    ann_id: str = ...,
    grid: str | None = None,
    coord_system: str | None = None,
    transpose: bool = ...,
    annotated_classes: Any = None,
    assume_aligned: bool = ...,
    assume_geometry: bool = ...,
    plane_as_volume: bool = ...,
) -> None:
    """Add NIfTI masks to a sample; the caller's report is updated."""
    ...

def nifti_read(
    path: str | os.PathLike[str],
    *,
    coord_system: str = ...,
    transpose: bool = ...,
    fourth_axis: str = ...,
    assume_geometry: bool = ...,
    plane_as_volume: bool = ...,
) -> tuple[bytearray, Any, list[int], dict[str, Any]]:
    """One file as `(bytes, dtype, shape, geometry)`: the voxels in C order in
    the returned shape and the file's byte order, and the geometry as JSON."""
    ...

def nifti_read_stored(
    path: str | os.PathLike[str],
) -> tuple[bytearray, Any, list[int], dict[str, Any]]:
    """One file as it is stored (`engine::read_stored`): `(bytes, dtype, shape,
    fields)`, the voxels first axis fastest in the header's shape."""
    ...

def nifti_to(
    path: str | os.PathLike[str],
    image_id: str,
    out: str | os.PathLike[str],
    *,
    report: Any,
    physical: bool = ...,
    annotation: str | None = None,
    class_key: Any = None,
) -> str:
    """Export one image, or one annotation or class of it; the file written."""
    ...

def nifti_write(
    array: Any,
    affine: npt.NDArray[np.float64],
    path: str | os.PathLike[str],
    *,
    report: Any = None,
    rescale: tuple[float, float] | None = None,
    units: str = ...,
    time_values: list[float] | None = None,
    time_units: str | None = None,
    leading: str | None = None,
    channels: Any = None,
) -> str:
    """Write one array as a NIfTI file; the file written."""
    ...

def nifti_write_field(
    array: Any, affine: npt.NDArray[np.float64], path: str | os.PathLike[str]
) -> None:
    """An ITK displacement field, `(x, y, z, 1, 3)` under the DISPVECT intent."""
    ...

def normalize_masks(masks: Any, spatial_shape: Any = None) -> tuple[Any, ...]: ...
def occupancy(mask: Any, factor: int) -> npt.NDArray[np.bool_]:
    """The occupancy map of a mask: one bit per `factor`-cube of voxels
    (`factor` at least 1)."""
    ...

def open_any(path: str | os.PathLike[str], *, key: str | None = None) -> Any:
    """A file whatever its kind: a sample handle, or a collection handle (a
    member's sample handle when `key` is given).
    """
    ...

def open_collection(path: str | os.PathLike[str]) -> CollectionHandle: ...
def open_sample(path: str | os.PathLike[str]) -> SampleHandle:
    """Open a `.medh5` sample read-only."""
    ...

def optimize_chunks(
    shape: Sequence[int],
    axis_kinds: Sequence[str],
    patch: Sequence[int] | int | None = None,
    *,
    itemsize: int = 4,
    l3_bytes: int | None = None,
    leading: int = 0,
) -> tuple[int, ...]: ...
def pack(
    sources: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    keys: Any = None,
) -> str: ...
def parse_digest(value: str) -> tuple[str, str]: ...
def payload_to_masks(
    payload: AnnotationPayload,
    *,
    spatial_shape: tuple[int, ...] | None = None,
    threshold: float | None = None,
) -> dict[int, npt.NDArray[np.bool_]]: ...
def profile_family(path: str | os.PathLike[str]) -> str:
    """`portable` when every dataset of the file needs only HDF5's own filters,
    else `balanced` --- what an amend of the file defaults to.
    """
    ...

def pyramid_factors(base: Grid, levels: Sequence[Grid]) -> npt.NDArray[np.float64]: ...
def quality_from_json(doc: Mapping[str, Any] | None) -> dict[str, QualityRecord]: ...
def quality_to_json(records: Mapping[str, QualityRecord]) -> dict[str, Any]: ...
def raw_chunks(dataset: Any) -> list[bytes]: ...
def read_document(root: Group) -> SampleDocument:
    """The sample document under a sample root, parsed: a later minor's as its
    supported projection (1.1 §2.2).
    """
    ...

def read_document_text(root: Group) -> str:
    """The raw `/meta` text under a sample root (`sample.root`)."""
    ...

def read_grid(group: Group, grid_id: str | None = None) -> Grid:
    """One grid group read back (`sample.root["grids/ct"]`)."""
    ...

def read_grids(root: Group) -> dict[str, Grid]:
    """Every grid under a sample root, by id."""
    ...

def read_indices(root: Group) -> dict[str, SamplingIndex]:
    """Every stored index entry under a sample root (`Sample.root`, or the
    `Sample`), by annotation id.
    """
    ...

def recompress(
    path: str | os.PathLike[str],
    profile: str,
    *,
    out: str | os.PathLike[str] | None = None,
    rechunk: bool = False,
) -> Any: ...
def recompress_paths(
    paths: Sequence[str | os.PathLike[str]], profile: str, *, rechunk: bool = False
) -> list[Any]: ...
def refuse_outside(inside: npt.NDArray[np.bool_]) -> None: ...
def registry_available() -> tuple[Any, ...]: ...
def registry_describe() -> dict[Any, Any]: ...
def registry_load(name: str) -> LabelSet: ...
def registry_load_file(path: str | os.PathLike[str]) -> LabelSet: ...
def registry_register(name: str, label_set: Any) -> LabelSet: ...
def registry_unregister(name: str) -> None: ...
def relative_path(node: Any, root: Group | None = None) -> str:
    """An object's path relative to a sample root (the file root by default)."""
    ...

def resample(
    path: str | os.PathLike[str],
    image_ids: list[str],
    *,
    onto: str | None = None,
    spacing: list[float] | None = None,
    grid_id: str | None = None,
    suffix: str | None = None,
    interpolation: str = "linear",
    fill: float = 0.0,
    dtype: Any | None = None,
    out: str | os.PathLike[str] | None = None,
) -> Any:
    """Resample images of a sample onto another grid, recorded as one `resample`
    activity; the report as plain data (`medh5.transforms.resample` builds its
    `ResampleReport` from it)."""
    ...

def resolve_between(
    transforms: Any, from_frame: str, to_frame: str
) -> TransformHandle | None:
    """The transform relating two frames, or `None` when no path exists."""
    ...

def resolve_profile_name(name: str | None = None) -> str:
    """A profile name, checked (`None` is the default, `balanced`)."""
    ...

def rules_for(level: str) -> list[str]:
    """Every rule run at `level`, in order."""
    ...

def same_file(a: str | os.PathLike[str], b: str | os.PathLike[str]) -> bool:
    """Whether two paths name one existing file, through `..`, a symbolic or a
    hard link.
    """
    ...

def sample_field(
    field: npt.NDArray[Any],
    coords: npt.NDArray[np.float64],
    *,
    interpolation: str = "linear",
    extrapolation: str = "zero",
) -> npt.NDArray[np.float64]: ...
def sampling_check_pair_mode(mode: str) -> None:
    """Refuse an unknown pair mode."""
    ...

def sampling_coerce_patch_size(patch_size: Any, ndim: int) -> tuple[Any, ...]:
    """Broadcast a patch size across `ndim` spatial axes."""
    ...

def sampling_grid_patches(
    shape: Sequence[int],
    patch_size: Any,
    *,
    overlap: int = 0,
    grid_id: str | None = None,
) -> list[Any]:
    """The sliding-window cover of a volume: a list of `Patch` field dicts."""
    ...

def sampling_pairs(mode: str, sample: Any) -> list[Any]:
    """The visit pairs of a sample in `mode`: `(first, second, interval_days,
    label)` each.
    """
    ...

def sampling_window_around(
    center: Sequence[int], patch: Sequence[int], shape: Sequence[int]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """`(slices, pad)` covering `patch` voxels around `center`."""
    ...

def schema() -> Any: ...
def schema_text() -> str: ...
def scrub_apply(
    path: str | os.PathLike[str],
    *,
    profile: str = ...,
    salt: str = ...,
    date_shift_days: int | None = None,
    date_shift_range: tuple[int, int] | None = None,
    performed_by: str | None = None,
    pseudonymise_ids: bool = False,
    withhold_shift: bool | None = None,
    out: str | os.PathLike[str] | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    attest_text: str | None = None,
    redact_text: bool = False,
    skip_refused: bool = False,
) -> dict[Any, Any]:
    """Act on the actionable findings and write the §11.4 attestation.  The
    report's fields.
    """
    ...

def scrub_check(
    apply_changes: bool,
    *,
    profile: str = ...,
    salt: str = ...,
    date_shift_days: int | None = None,
    date_shift_range: tuple[int, int] | None = None,
    performed_by: str | None = None,
    pseudonymise_ids: bool = False,
    withhold_shift: bool | None = None,
    out: str | os.PathLike[str] | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    attest_text: str | None = None,
    redact_text: bool = False,
    skip_refused: bool = False,
) -> None:
    """Refuse what these options cannot do, before any file is read
    (`ScrubOptions::check`).
    """
    ...

def scrub_finding(
    rule: str,
    where: Any,
    detail: str,
    value: str | None = None,
    *,
    actionable: bool = False,
    fixable: bool = True,
) -> dict[Any, Any]:
    """A finding as `ScrubReport.add` records it: the value previewed, and
    actionable only where `--apply` may act.
    """
    ...

def scrub_finding_json(finding: Any) -> Any: ...
def scrub_finding_line(finding: Any) -> str: ...
def scrub_pseudonymise(uid: str, salt: str = "") -> str:
    """A stable pseudonym for a UID: same input, same output, everywhere."""
    ...

def scrub_report_format(report: Any) -> str: ...
def scrub_report_json(report: Any) -> Any: ...
def scrub_report_ok(report: Any) -> bool: ...
def scrub_report_open_identity(report: Any) -> list[int]:
    """Positions of the findings on what the sample is *called*, in `remaining`
    once applied and in `findings` before.
    """
    ...

def scrub_scan(path: str | os.PathLike[str], *, profile: str = ...) -> dict[Any, Any]:
    """Find identifiers in one file; changes nothing.  The report's fields."""
    ...

def scrub_scan_document(document: Any, report: Any) -> dict[Any, Any]:
    """Every rule over one sample document, added to `report`: its fields after."""
    ...

def select_encoding(
    masks: Any = None,
    spatial_shape: tuple[int, ...] | None = None,
    *,
    stats: OverlapStats | None = None,
    soft: bool = False,
    prefer: str | None = None,
    ignore: bool = False,
) -> tuple[str, OverlapStats]:
    """Choose an encoding by measurement: `(kind, stats)`."""
    ...

def slices_to_box(slices: Sequence[slice]) -> npt.NDArray[np.float32]: ...
def source_check(
    source: Any,
    base: str | os.PathLike[str] | None = None,
    deep: bool = False,
    root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """The findings of checking a reference against its sample now."""
    ...

def source_default_id(
    uri: str, sample_key: str | None = None, taken: Sequence[str] = ...
) -> str:
    """The id a source reference gets when none is given, widened past the ids
    in `taken` (`SourceRef::default_id_among`).
    """
    ...

def source_pin(
    path: str | os.PathLike[str],
    sample_key: str | None = None,
    source_id: str | None = None,
    uri: str | None = None,
) -> dict[str, Any]:
    """A source reference pinned to what the sample is now."""
    ...

def source_resolve(
    uri: str,
    base: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> Path:
    """The file a source URI names, relative to `base` when it is relative: a
    `file:` URI decoded as the engine decodes it.
    """
    ...

def spatial_chunk_for(
    spatial_shape: Sequence[int],
    patch: Sequence[int] | int | None = None,
    *,
    itemsize: int = 4,
    l3_bytes: int | None = None,
) -> tuple[int, ...]: ...
def splits_from_json(
    docs: Sequence[Mapping[str, Any]] | None,
) -> tuple[SplitClaim, ...]: ...
def stage_file(path: str | os.PathLike[str]) -> StagedFileHandle:
    """Stage a replacement of `path`: a new, empty temporary beside the file it
    names, keeping its suffix.
    """
    ...

def stale_index_entries(root: Group) -> tuple[str, ...]: ...
def storage_dtype(planes: Any, threshold: float, requested: Any = None) -> Any: ...
def storage_recompress_json(result: Any) -> Any:
    """A re-encoding's result as `recompress --json` prints it."""
    ...

def storage_recompress_line(result: Any) -> str:
    """A re-encoding's result in one line (`str(result)`)."""
    ...

def subtrees_identical(a: Group, b: Group) -> tuple[str, ...]: ...
def target_registration_error(
    transform: Any,
    fixed_points: npt.ArrayLike,
    moving_points: npt.ArrayLike,
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """TRE `‖T(p_F) − p_M‖` over world landmarks with matching row order
    (§10.6): `{mean, median, max, n}`.
    """
    ...

def task_fingerprints(doc: Any) -> dict[str, str]:
    """`{"task": ..., "manifest": ...}`."""
    ...

def task_load_document(path: str | os.PathLike[str]) -> Any:
    """A manifest file's document, unparsed: T101 for anything not JSON."""
    ...

def task_notes(doc: Any) -> list[dict[str, Any]]:
    """What the manifest document should state and does not, as written."""
    ...

def task_normalize(doc: Any) -> dict[str, Any]:
    """Parse, check the schema, and return the normalised manifest."""
    ...

def task_preflight(
    doc: Any,
    base: str | os.PathLike[str] | None = None,
    deep: bool = False,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """The preflight of a task, as columns (`crate::preflight`)."""
    ...

def task_reconcile(
    doc: Any,
    base: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """The manifest with every subject's duplicated events recorded."""
    ...

def task_from_cohort(
    cohort: Any,
    task_id: str,
    version: str,
    identity_namespace: str,
    *,
    split: Any | None = None,
    set_id: str | None = None,
    fold: int | None = None,
    base: str | os.PathLike[str] | None = None,
    training_partition: str | None = None,
) -> dict[str, Any]:
    """A task over a cohort (`TaskManifest::from_cohort`), partitioned by
    `split` (a split's JSON) or by the claims the samples carry for `set_id`:
    the manifest's JSON.
    """
    ...

def task_repin(
    doc: Any,
    base: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    sources: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The manifest with its sources repinned, and what changed."""
    ...

def task_row_fingerprint(doc: Any, row_id: str) -> str: ...
def task_save(
    doc: Any,
    path: str | os.PathLike[str],
    base: str | os.PathLike[str] | None = None,
) -> None:
    """Write the manifest as the engine writes it, refusing an output that is
    one of its sources.
    """
    ...

def task_sources_outside(
    doc: Any, base: str | os.PathLike[str] | None = None
) -> list[str]:
    """The URIs of the sources that confining to `base` would refuse, in
    manifest order; no file is examined.  Empty when `base` cannot be
    canonicalized: its sources fail to open whatever confinement does.
    """
    ...

def task_row_fingerprints(doc: Any) -> dict[str, str]:
    """Every row's fingerprint by row id --- the first row of an id, as
    `task_row_fingerprint` finds it --- from one parse of the manifest: a
    parse per row made fingerprinting a task's rows quadratic.
    """
    ...

def task_schema_text() -> str: ...
def task_subjects_digest(doc: Any, partition: str) -> str: ...
def task_validate(doc: Any) -> list[dict[str, Any]]:
    """Everything wrong with a manifest that opening no file can find."""
    ...

def to_world_vectors(
    vectors: npt.NDArray[Any], grid: Grid, vector_space: str
) -> npt.NDArray[np.float64]: ...
def transcode(
    annotation: VoxelAnnotation, to_kind: str, *, drop_identity: bool = False
) -> AnnotationPayload:
    """Convert an open voxel annotation to another encoding (§7.6).
    A `probmap` is thresholded at its `threshold` (§7.5): another encoding
    keeps the voxels at or above it as the class, and none of the
    probabilities.
    """
    ...

def transcode_payload(
    payload: AnnotationPayload,
    to_kind: str,
    *,
    spatial_shape: tuple[int, ...] | None = None,
    threshold: float | None = None,
    drop_identity: bool = False,
    **kwargs: Any,
) -> AnnotationPayload:
    """Re-encode a payload; the same payload comes back when it already has
    `to_kind`.
    """
    ...

def unpack(
    path: str | os.PathLike[str],
    outdir: str | os.PathLike[str],
    *,
    keys: Any = None,
    suffix: str = ...,
) -> list[str]: ...
def validate_against_schema(doc: Any) -> list[str]: ...
def validate_file(
    path: str | os.PathLike[str],
    *,
    level: str = "semantic",
    profiles: Any = None,
    codes_revision: int | None = None,
) -> Any: ...
def validate_merge(reports: Any, path: str = "<multiple>") -> Any:
    """Several reports as one, each location prefixed with its file: the
    merged report's JSON.
    """
    ...

def validate_paths(
    paths: Sequence[str | os.PathLike[str]],
    *,
    level: str = "semantic",
    profiles: Any = None,
    codes_revision: int | None = None,
) -> list[Any]: ...
def validate_report_format(report: Any, *, verbose: bool = False) -> str:
    """A report as `validate` prints it."""
    ...

def validate_report_json(report: Any) -> Any:
    """A report as `validate --json` prints it."""
    ...

def validate_report_positions(report: Any) -> tuple[list[int], list[int], list[int]]:
    """The positions in `report.diagnostics` of its errors, its warnings and
    the warnings it promotes, at its level.
    """
    ...

def validate_root(
    sample: Any,
    *,
    path: str | None = None,
    level: str = "semantic",
    profiles: Any = None,
    errors_only: bool = False,
) -> Any:
    """Validate an open sample (`Sample`'s handle), with `errors_only` skipping
    the warning-only checks that read bulk data.
    """
    ...

def validate_sample_key(name: str) -> str:
    """A collection member key (§2.2), or E003."""
    ...

def verify_object(root: Group, path: str) -> bool: ...
def verify_root(
    root: Any,
    attr_names: Any = None,
    *,
    partial: Sequence[str] | None = None,
    check_content_id: bool = True,
) -> Any: ...
def voxel_volume(spacing: Sequence[float]) -> float: ...
def world_to_index(
    affine: npt.ArrayLike, points: npt.ArrayLike
) -> npt.NDArray[np.float64]: ...
def read_itk_h5(
    path: str | os.PathLike[str],
) -> list[tuple[str, list[float], list[float]]]:
    """The transforms of an ITK HDF5 transform file, in its order, each as
    `(type, parameters, fixed parameters)`.
    """

def write_itk_h5(
    path: str | os.PathLike[str],
    entries: Sequence[tuple[str, Sequence[float], Sequence[float]]],
) -> None:
    """Write `entries`, each `(type, parameters, fixed parameters)`, as the ITK
    HDF5 transform file `path`, replacing it atomically.
    """

def write_atomic(path: str | os.PathLike[str], data: str | bytes) -> None:
    """`data` as `path`'s whole content: a `str` as UTF-8, its `\\n` kept on
    every platform, `bytes` as given.
    """
    ...

def written_version(
    source: str | None, profiles: Sequence[str] = ..., *, kinds: Sequence[str] = ...
) -> str:
    """The `medh5_version` a commit writes: the lowest version `profiles` and
    annotation `kinds` need (`1.0`; `1.1` with `clinical`; `1.2` with a
    `polylines` annotation), and never below `source`, the version an amended
    file already declared (1.1 §2.3).  An amendment that removes the clinical
    profile (`drop_clinical`), or the last `polylines` annotation, passes no
    `source`: what remains is written at the lowest version it needs (1.1
    §2.1, 1.2 §2.1).
    """
    ...

@final
@dataclass(frozen=True)
class Activity:
    id: str
    type: str
    agent: str | None = ...
    started: str | None = ...
    ended: str | None = ...
    tool: str | None = ...
    inputs: tuple[str, ...] = ...
    outputs: tuple[str, ...] = ...
    params: dict[str, Any] = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Activity: ...
    def to_json(self) -> dict[str, Any]:
        """The record as it stands, as JSON."""
        ...

@final
@dataclass(frozen=True)
class Agent:
    id: str
    type: str
    name: str
    role: str | None = ...
    version: str | None = ...
    qualification: str | None = ...
    organization: str | None = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Agent: ...
    def to_json(self) -> dict[str, Any]:
        """The record as it stands, as JSON."""
        ...

@final
@dataclass(frozen=True)
class Agreement:
    metric: str
    value: float
    against: str | None = ...
    per_class: dict[str, float] = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Agreement: ...
    def to_json(self) -> dict[str, Any]:
        """The record as it stands, as JSON."""
        ...

@final
class AnnotationHandle:
    @property
    def ann_id(self) -> str: ...
    @property
    def annotated_class_ids(self) -> tuple[Any, ...]: ...
    @property
    def annotated_classes(self) -> tuple[Any, ...]: ...
    def as_slices(self, grid: Any = None) -> list[Any]: ...
    def as_world(self, grid: Any = None) -> Any: ...
    @property
    def asserted_class_ids(self) -> Any: ...
    @property
    def assertion_values(self) -> Any: ...
    def assertions(self) -> list[Any]:
        """Every assertion as `(class_id, value, scope_id, scheme, scheme_value)`."""
        ...
    @property
    def attributes(self) -> Any: ...
    @property
    def bit_class_ids(self) -> Any: ...
    @property
    def box_ndim(self) -> int: ...
    @property
    def boxes(self) -> Any: ...
    def by_plane(self) -> dict[Any, Any]: ...
    def by_scope_id(self) -> dict[Any, Any]:
        """`{scope_id: [assertion tuples]}`."""
        ...
    def class_bboxes(self, classes: Any = None) -> dict[Any, Any]: ...
    @property
    def class_ids(self) -> tuple[Any, ...]: ...
    def class_key(self, class_id: int) -> str: ...
    @property
    def class_name(self) -> str: ...
    @property
    def classes(self) -> tuple[Any, ...]: ...
    def classes_at(self, voxel: Sequence[int]) -> tuple[Any, ...]: ...
    @property
    def closure(self) -> str: ...
    @property
    def compared_timepoints(self) -> tuple[Any, ...]: ...
    def contains(self, class_key_: Any, voxel: Sequence[int]) -> bool: ...
    @property
    def contour_offsets(self) -> Any: ...
    @property
    def contour_planes(self) -> Any: ...
    @property
    def contour_roles(self) -> tuple[Any, ...]: ...
    @property
    def correspondence(self) -> str | None: ...
    def crop(self, index: int) -> Any: ...
    def dataset(self, name: str) -> Dataset:
        """A stored dataset of the annotation group."""
        ...
    def dense(self, classes: Any = None, roi: Any = None) -> Any: ...
    @property
    def frame_uid(self) -> str | None: ...
    @property
    def grid(self) -> Grid: ...
    @property
    def grid_id(self) -> str | None: ...
    @property
    def group(self) -> Group:
        """The annotation's stored group, for inspection."""
        ...
    def has_dataset(self, name: str) -> bool: ...
    @property
    def has_ignore_region(self) -> bool: ...
    @property
    def has_masks(self) -> bool: ...
    @property
    def header(self) -> AnnotationHeader: ...
    @property
    def ignore_id(self) -> int: ...
    def ignore_mask(self, roi: Any = None) -> Any: ...
    def instance(self, instance_id: int) -> tuple[Any, ...]: ...
    @property
    def instance_ids(self) -> Any: ...
    def instances(self) -> list[Any]:
        """Every instance as `(index, instance_id, class_id, box, mask, score)`."""
        ...
    def is_annotated(self, class_key_: Any) -> bool: ...
    @property
    def is_change_label(self) -> bool: ...
    @property
    def is_fully_covered(self) -> bool: ...
    @property
    def kind(self) -> str: ...
    @property
    def label_set(self) -> LabelSet | None: ...
    def labelled(self) -> Any: ...
    def labelmap(
        self, roi: Any = None, priority: Any = None, dtype: Any = None
    ) -> tuple[Any, ...]:
        """`(volume, overwritten, order, warning)`."""
        ...
    @property
    def labels(self) -> dict[Any, Any]: ...
    @property
    def layer_class_ids(self) -> Any: ...
    def layer_classes(self) -> tuple[Any, ...]: ...
    @property
    def layer_of(self) -> dict[Any, Any]: ...
    def mesh_bounds(self) -> Any: ...
    @property
    def multilabel(self) -> bool: ...
    @property
    def n_items(self) -> int: ...
    @property
    def n_layers(self) -> int: ...
    @property
    def n_objects(self) -> int: ...
    @property
    def n_planes(self) -> int: ...
    @property
    def n_spatial(self) -> int: ...
    @property
    def n_submeshes(self) -> int: ...
    def named_points(self) -> dict[Any, Any]: ...
    @property
    def normalized(self) -> bool: ...
    def obb_as_aabb(self) -> Any: ...
    def obb_corners(self) -> Any: ...
    def obb_volumes(self) -> Any: ...
    @property
    def object_class_ids(self) -> Any: ...
    @property
    def point_names(self) -> tuple[Any, ...] | None: ...
    def polygon(self, index: int) -> Any: ...
    def polygons(self) -> list[Any]:
        """Every polygon as `(vertices, class_id, (axis, index), role)`."""
        ...
    def polyline(self, index: int) -> Any: ...
    @property
    def polyline_offsets(self) -> Any: ...
    @property
    def polyline_planes(self) -> Any: ...
    def polylines(self) -> list[Any]:
        """Every polyline as `(vertices, class_id, (axis, index))` (1.2 §3)."""
        ...
    @property
    def position_of(self) -> dict[Any, Any]: ...
    @property
    def positives(self) -> tuple[Any, ...]: ...
    def probabilities(self, classes: Any = None, roi: Any = None) -> Any: ...
    @property
    def prov(self) -> str | None: ...
    @property
    def quality_key(self) -> str | None: ...
    def read_layer(self, layer: int, roi: Any = None) -> Any: ...
    def read_mask(self, roi: Any = None) -> Any: ...
    def resolve_class(self, key: Any) -> int: ...
    def resolve_classes(self, keys: Any = None) -> tuple[Any, ...]: ...
    def scheme(self, name: str) -> str | None: ...
    @property
    def scheme_values(self) -> tuple[Any, ...] | None: ...
    @property
    def schemes(self) -> tuple[Any, ...] | None: ...
    @property
    def scope(self) -> str: ...
    @property
    def scope_ids(self) -> Any: ...
    @property
    def scores(self) -> Any: ...
    def skeleton(self) -> Skeleton | None: ...
    @property
    def skeleton_id(self) -> str | None: ...
    @property
    def slice_index(self) -> Any: ...
    @property
    def space(self) -> str: ...
    @property
    def spatial_shape(self) -> tuple[Any, ...]: ...
    def state(self, class_key_: Any, *, scope_id: int | None = None) -> str: ...
    def summary(self) -> Any: ...
    @property
    def task(self) -> str: ...
    @property
    def threshold(self) -> float: ...
    @property
    def timepoints(self) -> tuple[Any, ...]: ...
    def to_index(self, coords: Any, *, grid: Any = None) -> Any: ...
    def to_world(self, coords: Any, *, grid: Any = None) -> Any: ...
    def tracking(self) -> dict[Any, Any]: ...
    def value(
        self, class_key_: Any, *, scope_id: int | None = None
    ) -> float | None: ...
    @property
    def visibility(self) -> Any: ...
    def voxel_counts(self, classes: Any = None) -> dict[Any, Any]: ...
    def world_corners(self, grid: Any = None) -> Any: ...

@final
@dataclass(frozen=True)
class AnnotationHeader:
    """The fixed attribute header every annotation carries (spec §6.2)."""

    kind: str
    task: str
    grid: str | None = ...
    timepoints: tuple[str, ...] | None = ...
    space: str | None = ...
    frame_uid: str | None = ...
    class_ids: tuple[int, ...] = ...
    annotated_class_ids: tuple[int, ...] = ...
    closure: str = ...
    ignore_id: int = ...
    ignore_mask: str | None = ...
    prov: str | None = ...
    quality: str | None = ...
    derived_from: tuple[str, ...] = ...
    extra: Mapping[str, Any] = ...
    def attrs(self) -> dict[str, Any]:
        """The attributes this header writes."""
        ...
    @classmethod
    def read(cls, group: Group) -> AnnotationHeader:
        """The header an annotation group carries (`sample.root["annotations/x"]`)."""
        ...

@final
@dataclass
class AnnotationPayload:
    """Datasets and kind-specific attributes for one encoded annotation.

    A mutable record, as 1.x's dataclass was: `datasets` and `attrs` are the
    payload's own dicts, so an array edited in place, or a dataset or an
    attribute added, is what the writer and the transcoder then read.  They
    are handed out on first access --- the arrays move to NumPy uncopied ---
    and read back when the payload is used.
    """

    kind: str
    datasets: dict[str, npt.NDArray[Any]] = ...
    attrs: dict[str, Any] = ...
    stacked_axes: int = ...
    class_ids: tuple[int, ...] = ...
    @property
    def data(self) -> npt.NDArray[Any]:
        """The `data` dataset."""
        ...
    def describe(self) -> dict[str, Any]:
        """The kind, and each dataset's shape and dtype, for reports."""
        ...
    @property
    def nbytes(self) -> int:
        """Total bytes of every dataset."""
        ...

@final
class Attrs:
    """An object's attributes, as a mutable mapping.

    Writes go straight to the file, so they only succeed on a file open for
    writing --- `SampleWriter.handle`, in practice.
    """
    def __contains__(self, key: Any, /) -> bool:
        """Return bool(key in self)."""
        ...
    def __delitem__(self, key: Any, /) -> None:
        """Delete self[key]."""
        ...
    def __getitem__(self, key: Any, /) -> Any:
        """Return self[key]."""
        ...
    def __iter__(self) -> Iterator[str]:
        """Implement iter(self)."""
        ...
    def __len__(self) -> int: ...
    def __setitem__(self, key: Any, value: Any, /) -> None:
        """Set self[key] to value."""
        ...
    def get(self, name: str, default: Any = None) -> Any: ...
    def items(self) -> list[tuple[str, Any]]: ...
    def keys(self) -> list[str]: ...
    def to_dict(self) -> dict[str, Any]:
        """The attributes as a plain `dict`."""
        ...
    def values(self) -> list[Any]: ...

@final
class CacheWriterHandle:
    """Writes a feature cache; `commit()` moves it into place."""
    def abort(self) -> None:
        """Discard the half-written cache."""
        ...
    def add(self, entry: Any, values: Any) -> dict[str, Any]:
        """Add one feature: `entry` as a dict (its `digest` is computed)."""
        ...
    def commit(self) -> str:
        """Write the manifest and its checksum; returns the checksum."""
        ...

@final
class ClinicalHandle:
    """One sample's clinical profile, read."""
    def descriptor(self) -> dict[str, Any]: ...
    def document(self, document_id: str) -> dict[str, Any]: ...
    def documents(self) -> list[dict[str, Any]]:
        """Document metadata (no text): id, media type, language, source type,
        and the text's length in UTF-8 bytes.
        """
        ...
    def events(self) -> list[dict[str, Any]]: ...
    def links(self) -> list[dict[str, Any]]: ...
    @property
    def projection(self) -> bool: ...
    def records(self) -> dict[str, Any]: ...
    def select(self, cutoff_us: int, policy: Any = None) -> dict[str, Any]: ...
    def summary(self) -> dict[str, Any]: ...
    def text(self, document_id: str) -> str:
        """One document's text, read from the file now."""
        ...

@final
@dataclass(frozen=True)
class Cohort:
    dataset_id: str | None = ...
    site_id: str | None = ...
    scanner_id: str | None = ...
    group_id: str | None = ...
    acquisition_protocol: str | None = ...
    extra: dict[str, Any] = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | None) -> Cohort: ...
    def grouping_key(self, subject_id: str) -> str:
        """The key subjects are grouped by when splitting (§12.2)."""
        ...
    def to_json(self) -> dict[str, Any]:
        """The record as it stands, as JSON."""
        ...

@final
class CollectionHandle:
    """One open collection shard.  Members are samples sharing its file."""
    def __len__(self) -> int: ...
    def close(self) -> None: ...
    def contains(self, key: str) -> bool: ...
    def get(self, key: str) -> SampleHandle:
        """One member, as a sample handle."""
        ...
    @property
    def group(self) -> Group:
        """The `samples` group."""
        ...
    @property
    def is_open(self) -> bool: ...
    def keys(self) -> list[str]: ...
    @property
    def kind(self) -> str: ...
    @property
    def path(self) -> str | None: ...
    def repr(self) -> str: ...
    @property
    def root(self) -> Group:
        """The root group."""
        ...
    def subject_ids(self) -> dict[Any, Any]:
        """`{sample_key: subject_id}`."""
        ...
    def summary(self) -> Any: ...
    @property
    def version(self) -> str: ...

@final
@dataclass(frozen=True, init=False)
class CostModel:
    """Raw (pre-compression) bytes per encoding."""

    labelmap: int | None
    layers: int
    bitmask: int
    instances: int
    probmap: int
    detail: dict[str, Any]
    def best(self) -> str:
        """The cheapest encoding (`labelmap` only when it can hold the masks)."""
        ...
    def to_json(self) -> dict[str, Any]: ...

@final
class Dataset:
    """A stored dataset: shape, dtype, chunks, filters, attributes and `ds[...]`."""
    def __array__(self, dtype: Any = None, copy: Any = None) -> npt.NDArray[Any]: ...
    def __getitem__(self, key: Any, /) -> Any:
        """Return self[key]."""
        ...
    def __len__(self) -> int: ...
    @property
    def attrs(self) -> Attrs: ...
    @property
    def chunks(self) -> tuple[int, ...] | None: ...
    @property
    def dtype(self) -> np.dtype[Any]: ...
    @property
    def filters(self) -> list[tuple[int, tuple[int, ...]]]:
        """The filter pipeline, as `(filter id, client data)` pairs."""
        ...
    @property
    def name(self) -> str: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    def read(self) -> npt.NDArray[Any]:
        """The whole dataset as an array: a numeric dataset as its dtype, a
        string dataset as an object array of `str`."""
        ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def size(self) -> int: ...
    @property
    def storage_size(self) -> int:
        """Bytes on disk (after compression)."""
        ...

@final
@dataclass(frozen=True)
class Deidentification:
    method: str
    profile: str | None = ...
    date_shift_days: int | None = ...
    id_mapping: str | None = ...
    performed_by: str | None = ...
    date: str | None = ...
    burned_in_annotation_checked: bool | None = ...
    extra: dict[str, Any] = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | None) -> Deidentification | None: ...
    def to_json(self) -> dict[str, Any]:
        """The record as it stands, as JSON."""
        ...

@final
class FeatureCacheHandle:
    """An open feature cache; `close()` releases the file."""
    def abandon(self) -> None:
        """Forget the handle without closing it: what a forked child does with
        its parent's (§14.4) --- the descriptor is the parent's to close.
        """
        ...
    def close(self) -> None: ...
    def entries(self) -> list[dict[str, Any]]: ...
    def entry_count(self) -> int:
        """How many entries the cache holds, without building them."""
        ...
    def event_entry(self, content_id: str, event_id: str) -> dict[str, Any] | None: ...
    def get(self, entry_id: str) -> npt.NDArray[Any]:
        """An entry's payload, its checksum verified."""
        ...
    def header(self) -> dict[str, Any]: ...
    @property
    def is_open(self) -> bool: ...
    @property
    def manifest_digest(self) -> str: ...
    @property
    def path(self) -> str: ...
    def row_entry(self, row_id: str) -> dict[str, Any] | None: ...

@final
@dataclass(frozen=True)
class Grid:
    grid_id: str
    shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    axis_kinds: tuple[str, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: npt.NDArray[np.float64]
    coord_system: str = ...
    units: str = ...
    timepoint: str | None = ...
    frame_uid: str | None = ...
    time_values: tuple[float, ...] | None = ...
    time_units: str | None = ...
    chunk_hint: tuple[int, ...] | None = ...
    patch_hint: tuple[int, ...] | None = ...
    extra: Mapping[str, Any] = ...
    @property
    def affine(self) -> npt.NDArray[np.float64]: ...
    def attrs(self) -> dict[str, Any]: ...
    @property
    def channel_axis(self) -> int | None: ...
    def check(self) -> None: ...
    def comparable_with(self, other: Grid) -> bool: ...
    @property
    def extent(self) -> npt.NDArray[np.float64]: ...
    def index_to_world(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]: ...
    def is_congruent(self, other: Grid, tol: float = 1e-06) -> bool: ...
    @property
    def n_spatial(self) -> int: ...
    @property
    def n_voxels(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def physical_size(self) -> tuple[float, ...]: ...
    @property
    def spatial_axes(self) -> tuple[int, ...]: ...
    @property
    def spatial_names(self) -> tuple[str, ...]: ...
    @property
    def spatial_shape(self) -> tuple[int, ...]: ...
    def summary(self) -> dict[str, Any]: ...
    @property
    def time_axis(self) -> int | None: ...
    def world_into(self, other: Grid, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """`(..., S)` world points of this grid, in `other`'s units (E414 across
        `coord_system`s, or from `px`).
        """
        ...
    def world_to_index(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]: ...

@final
class Group:
    """A stored group: members by name (or path), and attributes."""
    def __contains__(self, key: Any, /) -> bool:
        """Return bool(key in self)."""
        ...
    def __delitem__(self, key: Any, /) -> None:
        """Delete self[key]."""
        ...
    def __getitem__(self, key: Any, /) -> Group | Dataset:
        """Return self[key]."""
        ...
    def __iter__(self) -> Iterator[str]:
        """Implement iter(self)."""
        ...
    def __len__(self) -> int: ...
    def __setitem__(self, key: Any, value: Any, /) -> Any:
        """Set self[key] to value."""
        ...
    @property
    def attrs(self) -> Attrs: ...
    def get(self, path: str, default: Any = None) -> Any: ...
    def items(self) -> list[tuple[str, Group | Dataset]]: ...
    def keys(self) -> list[str]: ...
    @property
    def name(self) -> str: ...
    def values(self) -> list[Group | Dataset]: ...

@final
@dataclass(frozen=True)
class Identity:
    sample_id: str
    subject_id: str
    sex: str | None = ...
    laterality: str | None = ...
    bodypart: str | None = ...
    extra: dict[str, Any] = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Identity: ...
    def to_json(self) -> dict[str, Any]:
        """The record as it stands, as JSON."""
        ...

@final
class ImageHandle:
    @property
    def attrs(self) -> dict[Any, Any]: ...
    @property
    def channel_names(self) -> tuple[Any, ...] | None: ...
    @property
    def chunks(self) -> tuple[Any, ...] | None: ...
    @property
    def dataset(self) -> Dataset: ...
    @property
    def digest(self) -> str | None: ...
    @property
    def dtype(self) -> Any: ...
    @property
    def grid(self) -> Grid: ...
    @property
    def grid_id(self) -> str: ...
    @property
    def image_id(self) -> str: ...
    @property
    def is_multiscale(self) -> bool: ...
    @property
    def is_rescaled(self) -> bool: ...
    def level(self, index: int) -> ImageHandle: ...
    @property
    def level_index(self) -> int: ...
    @property
    def levels(self) -> int: ...
    @property
    def modality(self) -> str: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def prov(self) -> str | None: ...
    @property
    def pyramid(self) -> Pyramid | None: ...
    def read(
        self, roi: Any = None, *, physical: bool = False, dtype: Any = None
    ) -> Any: ...
    @property
    def rescale(self) -> tuple[float, float]: ...
    @property
    def shape(self) -> tuple[Any, ...]: ...
    def summary(self) -> Any: ...
    @property
    def timepoint(self) -> str | None: ...
    @property
    def valid_mask(self) -> str | None: ...
    @property
    def value_type(self) -> str: ...
    @property
    def value_units(self) -> str | None: ...
    @property
    def window(self) -> tuple[Any, ...] | None: ...

@final
@dataclass(frozen=True, init=False)
class IndexPayload:
    """The datasets of one annotation's index entry (§14.3)."""

    ann_id: str
    class_ids: npt.NDArray[np.uint16]
    voxel_counts: npt.NDArray[np.int64]
    class_bboxes: npt.NDArray[np.float32]
    fg_coords: dict[int, npt.NDArray[np.int32]]
    occupancy: npt.NDArray[np.bool_] | None
    source_digest: str | None
    max_coords: int
    seed: int
    stats: dict[str, Any]

@final
@dataclass(frozen=True)
class Issue:
    code: str
    severity: str = ...
    class_ids: tuple[int, ...] = ...
    note: str | None = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Issue: ...
    def to_json(self) -> dict[str, Any]:
        """The record as it stands, as JSON."""
        ...

@final
@dataclass(frozen=True)
class LabelClass:
    """One semantic class (§5.2).

    `properties` is the class's own dict, as 1.x's dataclass field was: made
    on its first read, the same object on every read after, and read back
    whenever the class is used --- `to_json`, `==`, `is_lesion`, a label set
    or a writer given it.  A class a document's label set hands out writes an
    edit to it through to the document.
    """

    id: int
    key: str
    name: str
    parents: tuple[int, ...] = ...
    category: str | None = ...
    color: tuple[int, int, int, int] | None = ...
    codes: tuple[OntologyCode, ...] = ...
    laterality: str | None = ...
    properties: dict[str, Any] = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> LabelClass: ...
    @property
    def is_lesion(self) -> bool:
        """Whether the class marks a lesion: `properties["is_lesion"]`, else its
        category.
        """
        ...
    def to_json(self) -> dict[str, Any]:
        """The class as it stands, as JSON."""
        ...

@final
class LabelSet:
    """A controlled vocabulary (§5.1).

    The classes are the set's own objects, as 1.x kept them --- those it was
    given, and from the first read those it hands out (`ls[2] is
    ls["liver"]`) --- so an edit to one's `properties` is the set's, digest
    included, and for a document's label set, the document's.
    """
    def __new__(
        cls,
        id: str,
        classes: Sequence[LabelClass] | None = None,
        *,
        version: str = ...,
        relations: Sequence[Relation] | None = None,
        skeletons: Sequence[Skeleton] | None = None,
        form: str = ...,
        uri: str | None = None,
        sha256: str | None = None,
    ) -> Self: ...
    def __contains__(self, key: Any, /) -> bool:
        """Return bool(key in self)."""
        ...
    def __getitem__(self, key: int | str, /) -> LabelClass:
        """Return self[key]."""
        ...
    def __iter__(self) -> Iterator[LabelClass]:
        """Implement iter(self)."""
        ...
    def __len__(self) -> int: ...
    def ancestors(self, key: int | str) -> tuple[int, ...]: ...
    def as_ref(self, uri: str) -> LabelSet: ...
    def check(self) -> None: ...
    @property
    def classes(self) -> tuple[LabelClass, ...]: ...
    def close(self, ids: Iterable[int], closure: str) -> tuple[int, ...]: ...
    def colors(self) -> dict[int, tuple[int, int, int, int]]: ...
    def content_doc(self) -> dict[str, Any]: ...
    def descendants(self, key: int | str) -> tuple[int, ...]: ...
    def digest(self, algo: str = "sha256") -> str: ...
    @property
    def form(self) -> str: ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | None) -> LabelSet | None: ...
    def get(self, key: int | str) -> LabelClass | None: ...
    @property
    def id(self) -> str: ...
    @property
    def ids(self) -> tuple[int, ...]: ...
    def ids_for(self, keys: Iterable[int | str]) -> tuple[int, ...]: ...
    @property
    def keys(self) -> tuple[str, ...]: ...
    def missing(self, ids: Iterable[int]) -> tuple[int, ...]: ...
    @property
    def relations(self) -> tuple[Relation, ...]: ...
    def relations_of(
        self, key: int | str, predicate: str | None = None
    ) -> tuple[Relation, ...]: ...
    def resolve(self, keys: Iterable[int | str]) -> tuple[LabelClass, ...]: ...
    @property
    def sha256(self) -> str: ...
    def skeleton(self, skeleton_id: str) -> Skeleton: ...
    @property
    def skeletons(self) -> tuple[Skeleton, ...]: ...
    def subset(
        self, keys: Iterable[int | str], *, id: str | None = None
    ) -> LabelSet: ...
    def to_json(self, *, form: str | None = None) -> dict[str, Any]:
        """The vocabulary as it stands, as JSON."""
        ...
    @property
    def uri(self) -> str | None: ...
    @property
    def version(self) -> str: ...

@final
@dataclass(frozen=True, init=False)
class Observation:
    """One object seen once: a row of one annotation."""

    timepoint: str
    annotation: str
    index: int
    instance_id: int
    class_id: int
    box: npt.NDArray[np.float32]
    voxel_count: int | None
    volume: float | None
    units: str | None
    score: float | None
    grid: str | None
    @property
    def centroid(self) -> npt.NDArray[np.float64]: ...
    @property
    def extent(self) -> npt.NDArray[np.float64]: ...
    @property
    def kind(self) -> str:
        """The annotation's kind: `instances` (a mask), `boxes`, `obb`, or
        `points` or `polylines` (presence only).  Not one of the 1.x
        dataclass's fields, so its shape and `repr` are unchanged.
        """
        ...
    def to_json(self) -> dict[str, Any]: ...
    @property
    def volume_mm(self) -> float | None:
        """`volume` in millimetres to the power of the box's axes, or `None`
        where it is unmeasured or its grid is uncalibrated.
        """
        ...

@final
@dataclass(frozen=True)
class OntologyCode:
    system: str
    code: str
    name: str | None = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> OntologyCode: ...
    def to_json(self) -> dict[str, Any]: ...

@final
@dataclass(frozen=True, init=False)
class OverlapStats:
    """Measured properties of a set of class masks, and what they imply (§7.6)."""

    class_ids: tuple[int, ...]
    spatial_shape: tuple[int, ...]
    counts: dict[int, int]
    edges: frozenset[tuple[int, int]]
    colouring: dict[int, int]
    localized: frozenset[int]
    n_labelled_voxels: int
    @property
    def depth(self) -> float: ...
    @property
    def fill(self) -> float: ...
    @property
    def is_edgeless(self) -> bool: ...
    @property
    def mean_degree(self) -> float: ...
    @property
    def n_classes(self) -> int: ...
    @property
    def n_layers(self) -> int: ...
    @property
    def n_planes(self) -> int: ...
    @property
    def n_voxels(self) -> int: ...
    def summary(self) -> dict[str, Any]: ...
    @property
    def total_foreground(self) -> int: ...

@final
class PatchDraws:
    """`n` patch draws from one generator, drawn one per `next()`: nothing is
    held but the generator and the count, so a long run costs one patch at a
    time.
    """
    def __iter__(self) -> Self: ...
    def __length_hint__(self) -> int:
        """The draws still to come."""
        ...
    def __next__(self) -> dict[str, Any]:
        """The next draw: the fields of a `Patch`."""
        ...

@final
class PatchSamplerHandle:
    """The engine half of `PatchSampler`: its configuration, validated, and the
    draws.
    """
    def __new__(
        cls,
        patch_size: Any,
        *,
        strategy: str = ...,
        foreground_prob: float = 0.5,
        foreground_classes: Any = None,
        class_weights: Any = None,
    ) -> Self: ...
    def annotation(
        self, sample: Any, annotation: str | None = None, grid: str | None = None
    ) -> str | None:
        """The annotation a draw takes foreground from, auto-selected if `None`."""
        ...
    def draw(
        self,
        sample: Any,
        annotation: str | None = None,
        rng: int | Sequence[int] | np.random.Generator | None = None,
        *,
        grid: str | None = None,
    ) -> dict[str, Any]:
        """One draw: the fields of a `Patch`."""
        ...
    def draws(
        self,
        sample: Any,
        annotation: str | None = None,
        n: int = 1,
        rng: int | Sequence[int] | np.random.Generator | None = None,
        *,
        grid: str | None = None,
    ) -> list[dict[str, Any]]:
        """`n` draws from one generator: the fields of each `Patch`."""
        ...
    def iter_draws(
        self,
        sample: Any,
        annotation: str | None = None,
        n: int = 1,
        rng: int | Sequence[int] | np.random.Generator | None = None,
        *,
        grid: str | None = None,
    ) -> PatchDraws:
        """The draws of [`draws`](Self::draws), one per `next()`.  The annotation
        and the grid are checked here, before the first patch.
        """
        ...
    def pick_class(
        self,
        counts: Mapping[int, int],
        rng: int | Sequence[int] | np.random.Generator | None = None,
    ) -> int | None:
        """A class to sample from, weighted as configured; `None` when no class
        has foreground.
        """
        ...
    def repr(self) -> str: ...
    def window_grid(
        self, sample: Any, annotation: str | None = None, grid: str | None = None
    ) -> str:
        """The grid a draw is measured in."""
        ...

@final
class Provenance:
    """Who did what (§11.1): agents and the activities they performed.

    The activities are the graph's own objects, as 1.x kept them --- those it
    was given, and from the first read those it hands out --- so an edit to
    one's `params` is the graph's.  A document's provenance makes every change
    to the document as well.
    """
    def __new__(
        cls,
        agents: Sequence[Agent] | None = None,
        activities: Sequence[Activity] | None = None,
    ) -> Self: ...
    def __bool__(self) -> bool: ...
    def __eq__(self, other: object, /) -> bool: ...
    __hash__: ClassVar[None]  # type: ignore[assignment]
    def __iter__(self) -> Iterator[Activity]:
        """Implement iter(self)."""
        ...
    def __len__(self) -> int: ...
    @property
    def activities(self) -> tuple[Activity, ...]: ...
    def activities_by_type(self, activity_type: str) -> tuple[Activity, ...]: ...
    def activity(self, activity_id: str) -> Activity: ...
    def add_activity(self, activity: Activity, *, replace: bool = False) -> Activity:
        """Add an activity; an id already in the graph is refused unless
        `replace`.  The activity given is kept, as 1.x kept it; a document's
        provenance keeps, and returns, the document's.
        """
        ...
    def add_agent(self, agent: Agent, *, replace: bool = False) -> Agent:
        """Add an agent; an id already in the graph is refused unless `replace`."""
        ...
    def agent(self, agent_id: str) -> Agent: ...
    @property
    def agents(self) -> tuple[Agent, ...]: ...
    def dangling_agent_refs(self) -> tuple[tuple[str, str], ...]: ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | None) -> Provenance: ...
    def has_activity(self, activity_id: str) -> bool: ...
    def has_agent(self, agent_id: str) -> bool: ...
    def produced_by(self, object_path: str) -> tuple[Activity, ...]: ...
    def to_json(self) -> dict[str, Any]:
        """The graph as it stands, as JSON."""
        ...

@final
@dataclass(frozen=True)
class Pyramid:
    levels: int
    downsample_factors: npt.NDArray[np.float64]
    downsample_method: str
    grid_levels: tuple[str, ...]
    def attrs(self) -> dict[str, Any]: ...

@final
@dataclass(frozen=True)
class QualityRecord:
    status: str
    confidence: float | None = ...
    reviewed_by: tuple[str, ...] = ...
    agreement: tuple[Agreement, ...] = ...
    issues: tuple[Issue, ...] = ...
    edit_effort_s: float | None = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> QualityRecord: ...
    @property
    def is_usable(self) -> bool:
        """Whether the record may be trained on: not rejected or deprecated."""
        ...
    def to_json(self) -> dict[str, Any]:
        """The record as it stands, as JSON."""
        ...

@final
@dataclass(frozen=True)
class Relation:
    subject: int
    predicate: str
    object: int
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Relation: ...
    def to_json(self) -> dict[str, Any]: ...

@final
@dataclass
class SampleDocument:
    """The whole document, typed.  Fields read and write the engine's types."""

    identity: Identity
    timepoints: Timeline
    cohort: Cohort = ...
    label_set: LabelSet | None = ...
    provenance: Provenance = ...
    quality: dict[str, QualityRecord] = ...
    splits: tuple[SplitClaim, ...] = ...
    acquisition: dict[str, dict[str, Any]] = ...
    deidentification: Deidentification | None = ...
    extra: dict[str, Any] = ...
    def check_schema(self) -> list[str]: ...
    def dumps(self, *, indent: int | None = None) -> str: ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> SampleDocument: ...
    @property
    def group_id(self) -> str: ...
    @classmethod
    def loads(cls, payload: str | bytes) -> SampleDocument:
        """Parse a document from its JSON text, or from that text's UTF-8 bytes.

        Bytes that are not UTF-8 are refused (§2.4), not decoded with
        replacement characters standing in for what was there."""
        ...
    def quality_of(self, key: str | None) -> QualityRecord | None: ...
    @property
    def subject_id(self) -> str: ...
    def summary(self) -> dict[str, Any]: ...
    def to_json(self) -> dict[str, Any]: ...

@final
class SampleHandle:
    """One open sample root.  `close()` releases the file handle."""
    def abandon(self) -> bool:
        """Make sure this process never closes the file: for a process that
        inherited the handle across `fork`, where closing would call into HDF5
        on descriptors that belong to the parent.  One reference to the engine
        sample is leaked, so neither `close()` nor collection releases it.
        Returns whether there was an open handle to pin.
        """
        ...
    def annotation(self, ann_id: str) -> AnnotationHandle: ...
    def annotation_ids(self) -> list[str]: ...
    def annotations_at(self, timepoint: str) -> list[str]: ...
    def attr_name_map(self) -> dict[Any, Any]: ...
    def clinical(self) -> ClinicalHandle | None:
        """The clinical profile's records, when the sample declares it (1.1)."""
        ...
    def close(self) -> None:
        """Close the file and everything read from it, as 1.x's sample did: an
        image, annotation or group still held becomes invalid rather than
        keeping the file open and locked.  A collection member owns no file:
        closing one releases the member alone, and the shard --- its other
        members, and what was read from this one --- stays open until the
        collection is closed.
        """
        ...
    def compute_content_id(self) -> str: ...
    @property
    def content_id(self) -> str | None: ...
    def document(self) -> SampleDocument:
        """The sample document (§2), as read --- a later minor's as its supported
        projection (1.1 §2.2): a copy, so editing it changes nothing in the
        file.
        """
        ...
    def document_text(self, document_id: str) -> str:
        """One clinical document's text, read on its own: the document table's
        offsets and that document's bytes --- not the events, the links or any
        other document's text (1.1 §6)."""
        ...
    def frames_for(self, key: str) -> list[str]: ...
    def fresh_indices(self) -> list[str]: ...
    def grid(self, grid_id: str) -> Grid:
        """One grid: `KeyError`, or W913 for one a later minor's file holds that
        this engine's projection leaves out (1.1 §2.2).
        """
        ...
    def grids(self) -> dict[Any, Any]:
        """`{grid_id: Grid}`, with §3.7's implicit timepoint resolved."""
        ...
    def ignore_region(self, ann_id: str, roi: Any = None) -> Any: ...
    def image(self, image_id: str) -> ImageHandle: ...
    def image_ids(self) -> list[str]: ...
    def images_at(self, timepoint: str) -> list[str]: ...
    def index(self, ann_id: str) -> SamplingIndex: ...
    def index_ids(self) -> list[str]: ...
    @property
    def is_open(self) -> bool:
        """Whether the sample can be read: not once closed, and not, for a
        collection member, once its collection is closed."""
        ...
    @property
    def kind(self) -> str: ...
    @property
    def path(self) -> str | None: ...
    @property
    def profiles(self) -> list[str]: ...
    def reference_grid(self) -> Grid: ...
    def repr(self) -> str: ...
    def resolve_frames(
        self, from_frame: str, to_frame: str
    ) -> TransformHandle | None: ...
    @property
    def root(self) -> Group:
        """The sample's root group (read-only use)."""
        ...
    def summary(self) -> Any: ...
    @property
    def support(self) -> str:
        """`full` or `projection` (a higher minor, read as what this engine knows)."""
        ...
    def tracks(self, class_key: Any = None, *, measure: bool = True) -> Tracking: ...
    def transform(self, transform_id: str) -> TransformHandle: ...
    def transform_between(
        self, source: str, target: str, by: str | None = None
    ) -> TransformHandle | None: ...
    def transform_ids(self) -> list[str]: ...
    def valid_region(self, image_id: str, roi: Any = None) -> Any: ...
    def verify(self, partial: Sequence[str] | None = None) -> Any: ...
    @property
    def version(self) -> str: ...

@disjoint_base
class SampleWriter:
    """Builder for one sample.  Every `add_*` validates immediately; `commit`
    validates the whole and atomically replaces the target (§14.4).
    """
    def __new__(
        cls,
        path: str | os.PathLike[str],
        *,
        sample_id: str | None = None,
        subject_id: str | None = None,
        codec: str = "balanced",
        profiles: Sequence[str] | None = None,
    ) -> Self: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        _exc: BaseException | None = None,
        _tb: TracebackType | None = None,
    ) -> bool | None: ...
    def abort(self) -> None:
        """Discard the in-progress file, leaving any existing one untouched."""
        ...
    def acquisition(self, image_id: str, **params: Any) -> dict[str, Any]:
        """Merge acquisition parameters for one image; returns them all, as the
        document's dict: an edit to it reaches the file.
        """
        ...
    def activity(
        self,
        activity_type: str,
        *,
        agent: Agent | str | None = None,
        activity_id: str | None = None,
        **fields: Any,
    ) -> Activity:
        """Record an activity; ids are `act_<type>_<n>` unless given."""
        ...
    def add_boxes(
        self,
        ann_id: str,
        boxes: npt.ArrayLike,
        class_ids: Sequence[int | str],
        *,
        grid: str | None = None,
        space: str = "index",
        frame_uid: str | None = None,
        instance_ids: Sequence[int] | None = None,
        scores: Sequence[float] | None = None,
        attributes: Sequence[Mapping[str, Any]] | None = None,
        slice_index: Sequence[int] | None = None,
        annotated_classes: str | Sequence[int | str] | None = None,
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] | None = None,
        task: str = "detection",
        codec: str | None = None,
    ) -> Group:
        """Axis-aligned boxes, `(N, S, 2)` in `[lo, hi]` at voxel edges (§8.2)."""
        ...
    def add_classification(
        self,
        ann_id: str,
        labels: Mapping[int | str, float] | Sequence[Sequence[Any]],
        *,
        scope: str = "sample",
        multilabel: bool = True,
        scope_ids: Sequence[int] | None = None,
        schemes: Sequence[str] | None = None,
        scheme_values: Sequence[str] | None = None,
        grid: str | None = None,
        annotated_classes: str | Sequence[int | str] | None = None,
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] | None = None,
        codec: str | None = None,
    ) -> Group:
        """A classification annotation (§9).  A change label is `scope="sample"`
        with explicit `timepoints`.  *labels* is a mapping `class -> value` or
        rows `(class, value[, scope_id[, scheme, scheme_value]])`.
        """
        ...
    def add_contours(
        self,
        ann_id: str,
        polygons: Sequence[Polygon],
        *,
        grid: str | None = None,
        space: str = "index",
        frame_uid: str | None = None,
        annotated_classes: str | Sequence[int | str] | None = None,
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] | None = None,
        task: str = "segmentation",
        codec: str | None = None,
    ) -> Group:
        """Planar polygons (§8.6) --- the RTSTRUCT-shaped annotation."""
        ...
    def add_document(self, document: Any = None, **fields: Any) -> dict[str, Any]:
        """Add one source document (1.1 §6)."""
        ...
    def add_event(self, event: Any = None, **fields: Any) -> dict[str, Any]:
        """Add one event version (1.1 §5): an `Event`, a dict, or keywords."""
        ...
    def add_grid(
        self,
        grid_id: str,
        *,
        shape: Sequence[int],
        spacing: Sequence[float],
        origin: Sequence[float] | None = None,
        direction: npt.ArrayLike | None = None,
        axis_names: Sequence[str] | None = None,
        axis_kinds: Sequence[str] | None = None,
        coord_system: str = "LPS",
        units: str = "mm",
        timepoint: str | None = None,
        frame_uid: str | None = None,
        patch_hint: Sequence[int] | None = None,
        chunk_hint: Sequence[int] | None = None,
        time_values: Sequence[float] | None = None,
        time_units: str | None = None,
    ) -> Grid:
        """Declare a grid.  Geometry lives here and nowhere else."""
        ...
    def add_image(
        self,
        image_id: str,
        data: npt.ArrayLike,
        *,
        grid: str,
        modality: str,
        value_type: str = "intensity",
        value_units: str | None = None,
        channel_names: Sequence[str] | None = None,
        rescale_slope: float | None = None,
        rescale_intercept: float | None = None,
        window_center: Sequence[float] | None = None,
        window_width: Sequence[float] | None = None,
        valid_mask: str | None = None,
        prov: Activity | str | None = None,
        codec: str | None = None,
    ) -> Dataset:
        """Write one image, chunked for its grid's patch hint."""
        ...
    def add_keypoints(
        self,
        ann_id: str,
        points: npt.ArrayLike,
        keypoint_classes: Sequence[int | str],
        class_ids: Sequence[int | str],
        *,
        grid: str | None = None,
        space: str = "index",
        frame_uid: str | None = None,
        visibility: npt.ArrayLike | None = None,
        instance_ids: Sequence[int] | None = None,
        scores: Sequence[float] | None = None,
        skeleton: str | None = None,
        annotated_classes: str | Sequence[int | str] | None = None,
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] | None = None,
        task: str = "detection",
        codec: str | None = None,
    ) -> Group:
        """`(N, K, S)` keypoints with per-slot classes (§8.4)."""
        ...
    def add_link(self, link: Any = None, **fields: Any) -> dict[str, Any]:
        """Add one typed link (1.1 §7)."""
        ...
    def add_mask(
        self,
        ann_id: str,
        mask: npt.NDArray[Any],
        *,
        grid: str,
        task: str = "other",
        prov: Activity | str | None = None,
        codec: str | None = None,
    ) -> Group:
        """Write a boolean `mask` annotation (FOV, ignore region).  A
        floating-point mask holding a value other than 0 and 1 is refused
        (E411) rather than read as set wherever it is non-zero.
        """
        ...
    def add_mesh(
        self,
        ann_id: str,
        vertices: npt.ArrayLike,
        faces: npt.ArrayLike,
        *,
        grid: str | None = None,
        space: str = "world",
        frame_uid: str | None = None,
        normals: npt.ArrayLike | None = None,
        vertex_class_ids: Sequence[int | str] | None = None,
        mesh_offsets: Sequence[int] | None = None,
        mesh_class_ids: Sequence[int | str] | None = None,
        annotated_classes: str | Sequence[int | str] | None = None,
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] | None = None,
        task: str = "segmentation",
        codec: str | None = None,
    ) -> Group:
        """A triangle surface mesh (§8.7); `space` defaults to `world`."""
        ...
    def add_obb(
        self,
        ann_id: str,
        centers: npt.ArrayLike,
        sizes: npt.ArrayLike,
        rotations: npt.ArrayLike,
        class_ids: Sequence[int | str],
        *,
        grid: str | None = None,
        space: str = "index",
        frame_uid: str | None = None,
        instance_ids: Sequence[int] | None = None,
        scores: Sequence[float] | None = None,
        attributes: Sequence[Mapping[str, Any]] | None = None,
        annotated_classes: str | Sequence[int | str] | None = None,
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] | None = None,
        task: str = "detection",
        codec: str | None = None,
    ) -> Group:
        """Oriented boxes: centre, full edge lengths, rotation (§8.3)."""
        ...
    def add_points(
        self,
        ann_id: str,
        points: npt.ArrayLike,
        *,
        grid: str | None = None,
        space: str = "index",
        frame_uid: str | None = None,
        class_ids: Sequence[int | str] | None = None,
        instance_ids: Sequence[int] | None = None,
        names: Sequence[str] | None = None,
        weights: Sequence[float] | None = None,
        correspondence: str | None = None,
        annotated_classes: str | Sequence[int | str] | None = None,
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] | None = None,
        task: str = "detection",
        codec: str | None = None,
    ) -> Group:
        """A point set: landmarks, seeds, or half a correspondence (§8.5)."""
        ...
    def add_polylines(
        self,
        ann_id: str,
        polylines: Sequence[Polyline],
        *,
        grid: str | None = None,
        space: str = "index",
        frame_uid: str | None = None,
        instance_ids: Sequence[int] | None = None,
        annotated_classes: str | Sequence[int | str] | None = None,
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] | None = None,
        task: str = "other",
        codec: str | None = None,
    ) -> Group:
        """Open curves (1.2 §3): centrelines, catheter and applicator paths,
        RTSTRUCT's open contours.  A sample holding one is written at format
        1.2.
        """
        ...
    def add_pyramid(
        self,
        image_id: str,
        levels: Sequence[npt.ArrayLike],
        *,
        grid_levels: Sequence[str],
        modality: str,
        value_type: str = "intensity",
        downsample_method: str = "mean",
        value_units: str | None = None,
        rescale_slope: float | None = None,
        rescale_intercept: float | None = None,
        prov: Activity | str | None = None,
        codec: str | None = None,
    ) -> Group:
        """Write a multiscale image (§4.3); level geometry is checked here."""
        ...
    def add_records(self, records: Any) -> None:
        """Add a logical-record bundle: `clinical`, `events`, `documents`, `links`."""
        ...
    def add_segmentation(
        self,
        ann_id: str,
        *,
        grid: str,
        masks: Mapping[Any, npt.NDArray[Any]] | None = None,
        probabilities: Mapping[Any, npt.NDArray[Any]] | None = None,
        instances: Sequence[InstanceInput] | None = None,
        labelmap: npt.NDArray[np.integer[Any]] | None = None,
        label_values: Mapping[int, int | str | Sequence[int | str]] | None = None,
        ignore_value: int | None = None,
        encoding: str = "auto",
        threshold: float | None = None,
        annotated_classes: str | Sequence[int | str] | None = None,
        closure: str = "explicit",
        ignore: npt.NDArray[np.bool_] | None = None,
        ignore_mask: str | None = None,
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] | None = None,
        task: str = "segmentation",
        codec: str | None = None,
    ) -> tuple[str, OverlapStats | None]:
        """Write a voxel annotation, choosing the encoding by measurement.

        Returns the chosen `kind` and the overlap statistics behind the choice.
        `encoding` is `"auto"` or one of `TRANSCODABLE`: a `mask` has no
        classes, so it is refused (E404), as a transcode to it is.  Masks
        stored as `instances` are one object per class, each minted an id
        above every id the sample holds (§7.4).

        `labelmap` is the masks as one integer volume, borrowed rather than
        copied: each nonzero value the class of that id, or the classes
        `label_values` gives it, and `ignore_value` the ignore region.  What it
        writes is what the masks it implies write, byte for byte.
        """
        ...
    def add_timepoint(self, timepoint_id: str, **fields: Any) -> Timepoint:
        """Declare a timepoint; the first explicit one replaces the implicit `tp0`."""
        ...
    def add_transform(
        self,
        transform_id: str,
        *,
        kind: str,
        from_frame: str,
        to_frame: str,
        matrix: npt.ArrayLike | None = None,
        field: npt.ArrayLike | None = None,
        control_points: npt.ArrayLike | None = None,
        components: Sequence[str] | None = None,
        field_grid: str | None = None,
        cp_grid: str | None = None,
        vector_space: str = "world",
        interpolation: str = "linear",
        extrapolation: str = "zero",
        order: int = 3,
        units: str | None = None,
        from_grid: str | None = None,
        to_grid: str | None = None,
        invertible: bool | None = None,
        inverse_id: str | None = None,
        metrics: str | Mapping[str, Any] | None = None,
        prov: Activity | str | None = None,
        codec: str | None = None,
        field_dtype: str = "float32",
    ) -> Group:
        """A transform mapping points from `from_frame` to `to_frame`:
        `x_M = T(x_F)`, the ITK convention.  `units` defaults to those of the
        grids in the two frames (§10.1).

        `field_dtype` stores a displacement field as `"float32"` (the
        default), `"float16"` (refused when a finite value would overflow), or
        `"auto"`: float16 when the largest displacement is below 64 voxels of
        the field's grid and within float16's range (§10.4).  Any but
        `"float32"` is refused for another kind.
        """
        ...
    def build_index(
        self,
        ann_ids: Sequence[str] | None = None,
        *,
        max_coords: int = ...,
        occupancy: int | None = ...,
        seed: int = 0,
    ) -> tuple[str, ...]:
        """Build sampling indices (§14.3); every non-mask voxel annotation when
        `ann_ids` is `None`.  `occupancy=None` writes no occupancy planes.
        """
        ...
    def clinical(self) -> dict[str, Any] | None:
        """The clinical records so far (an amended file's once loaded), or `None`."""
        ...
    @property
    def closed(self) -> bool:
        """Whether the writer has committed or been aborted.  A commit refused
        before it changed anything leaves it open.
        """
        ...
    @property
    def codec(self) -> str:
        """The codec profile datasets default to."""
        ...
    def cohort(self, **fields: Any) -> Cohort: ...
    def commit(self, *, digests: bool = True) -> str | None:
        """Validate, write `/meta`, stamp digests and atomically replace.

        `digests=True` recomputes every digest, and refuses (E701) to restamp
        one whose bytes the amended file already failed; `digests=False`
        stamps only datasets that carry none (or one in another algorithm than
        the file's).  An amended file keeps its `digest_algo`.  A commit
        refused before it changes anything --- no image (E201), a reference
        that does not resolve, a schema error (E005) --- leaves the writer
        open, to mend and commit again; one that fails after it began writing
        aborts it.  Returns the `content_id`, or `None` when already
        committed; committing an aborted writer raises.
        """
        ...
    def deidentification(self, **fields: Any) -> Deidentification: ...
    @property
    def document(self) -> SampleDocument:
        """The sample document being built.  Live: assigning one of its fields
        edits the writer's document, and `commit` validates the result.
        """
        ...
    @document.setter
    def document(self, value: SampleDocument) -> None: ...
    def drop_clinical(self) -> None:
        """Remove the clinical profile: the imaging projection (1.1 §10), written
        at the lowest version what remains needs.  A sample with no clinical
        profile to remove keeps the version it declared (1.1 §2.1).
        """
        ...
    def extra(self, namespace: str, value: Any) -> None:
        """Set a namespaced extension member of the document."""
        ...
    def frame_uids(self) -> dict[str, tuple[str, ...]]:
        """Every frame UID in the file so far, and the attributes naming it."""
        ...
    @property
    def grids(self) -> dict[str, Grid]:
        """Grids declared so far."""
        ...
    @property
    def handle(self) -> Group:
        """The file being built, for tools that rewrite what the builder does not
        model.  `commit` stamps what is written through it, and refuses
        (E701) to restamp a dataset the amended file already failed.
        """
        ...
    @property
    def has_clinical(self) -> bool: ...
    def identity(self, **fields: Any) -> Identity:
        """Merge fields into the identity (`sample_id`, `subject_id`, ...)."""
        ...
    def infer_profiles(self) -> frozenset[str]:
        """Profiles this sample actually satisfies, unioned with declared ones."""
        ...
    def label_set(self, label_set: LabelSet) -> LabelSet:
        """Set the label set, as it stands; the document's is returned, so an
        edit to a class's `properties` through it reaches the file.
        """
        ...
    def organization(self, name: str, **fields: Any) -> Agent: ...
    @property
    def path(self) -> str:
        """The target path."""
        ...
    def person(
        self, name: str, agent_id: str | None = None, **fields: Any
    ) -> Agent: ...
    def remap_frame_uids(self, mapping: Mapping[str, str]) -> tuple[str, ...]:
        """Rename frames of reference everywhere they are named (§3.4)."""
        ...
    def remove_annotation(self, ann_id: str) -> None:
        """Drop an annotation, and any index entry derived from it."""
        ...
    def remove_grid(self, grid_id: str) -> None:
        """Drop a grid; what still names it (E101, E503), and the last grid
        going (E111), are checked at commit.
        """
        ...
    def remove_image(self, image_id: str) -> None:
        """Drop an image, its `acquisition` entry and its `series_uids`
        mentions; what still names it, and the last image going (E201), are
        checked at commit.
        """
        ...
    def remove_transform(self, transform_id: str) -> None:
        """Drop a transform; what still names it is checked at commit."""
        ...
    def set_clock(self, clock: Any = None, **fields: Any) -> dict[str, Any]:
        """Declare the subject clock, starting the `clinical` profile (1.1 §3)."""
        ...
    def set_image_attrs(
        self,
        image_id: str,
        *,
        modality: str = ...,
        value_type: str = ...,
        value_units: str | None = ...,
        channel_names: Sequence[str] | None = ...,
        rescale_slope: float | None = ...,
        rescale_intercept: float | None = ...,
        window_center: Sequence[float] | None = ...,
        window_width: Sequence[float] | None = ...,
        valid_mask: str | None = ...,
        prov: Activity | str | None = ...,
    ) -> None:
        """Change an image's attributes without rewriting its samples (§4.1):
        `None` removes one.  Checked as `add_image` checks them; `modality`
        and `value_type` are required and cannot be removed (E205).
        """
        ...
    def set_quality(self, key: str, **fields: Any) -> QualityRecord:
        """Create or replace a quality record (status defaults to `draft`)."""
        ...
    def software(
        self, name: str, version: str | None = None, **fields: Any
    ) -> Agent: ...
    def split(self, **fields: Any) -> SplitClaim:
        """Record a split claim, replacing any earlier claim for the same set."""
        ...
    def transcode_annotation(
        self,
        ann_id: str,
        to_kind: str,
        *,
        codec: str | None = None,
        drop_identity: bool = False,
    ) -> str:
        """Re-encode a voxel annotation in place, preserving its header (§7.6).
        A `probmap` is thresholded at its `threshold` (§7.5): another encoding
        keeps the voxels at or above it as the class, and none of the
        probabilities.
        """
        ...

@final
class SamplingIndex:
    @property
    def ann_id(self) -> str: ...
    def bbox(self, class_id: int) -> npt.NDArray[np.float32] | None: ...
    @property
    def class_ids(self) -> tuple[int, ...]: ...
    def class_weights(self, mode: str = "inverse_frequency") -> dict[int, float]: ...
    def coords(self, class_id: int) -> npt.NDArray[np.int32]: ...
    @property
    def group(self) -> Group:
        """The `index/<ann_id>` group, as 1.x exposed it."""
        ...
    def has_class(self, class_id: int) -> bool: ...
    @property
    def has_occupancy(self) -> bool: ...
    def is_current(self, annotation_digest: str) -> bool:
        """Whether this entry was built from the annotation with this digest.

        Compute `annotation_digest` under the algorithm the entry's
        `source_digest` names, its prefix (§13.3) ---
        `group_digest(group, algo, root=...)` --- which need not be the file's
        `digest_algo`.
        """
        ...
    @property
    def max_coords(self) -> int: ...
    def occupancy_plane(self, position: int) -> npt.NDArray[np.bool_] | None: ...
    def sample_foreground(
        self,
        class_id: int,
        n: int = 1,
        rng: int | Sequence[int] | np.random.Generator | None = None,
    ) -> npt.NDArray[np.int32]:
        """Draw `n` foreground voxel coordinates of a class; `rng` seeds the draw
        (an int, ints, or a `numpy.random.Generator`; fresh entropy when `None`).
        """
        ...
    @property
    def source_digest(self) -> str | None: ...
    def summary(self) -> dict[str, Any]: ...
    @property
    def voxel_counts(self) -> dict[int, int]: ...

@final
@dataclass(frozen=True)
class Skeleton:
    id: str
    keypoints: tuple[int, ...]
    edges: tuple[tuple[int, int], ...] = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Skeleton: ...
    def to_json(self) -> dict[str, Any]: ...

@final
class StagedFileHandle:
    """A replacement of a file being written by someone else: write `path`,
    then `commit()`.  As a context manager it commits when the block ends
    without an exception, and aborts otherwise.
    """
    @property
    def path(self) -> Path:
        """The temporary, which the foreign writer writes."""
        ...
    @property
    def target(self) -> Path:
        """The file the commit replaces: the target, its links followed."""
        ...
    def abort(self) -> None:
        """Remove the temporary, leaving the target as it was.  Aborting a
        committed or aborted file does nothing.
        """
        ...
    def commit(self) -> None:
        """Rename the written temporary over the target."""
        ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        _exc: BaseException | None = None,
        _tb: TracebackType | None = None,
    ) -> bool: ...

@final
@dataclass(frozen=True)
class SplitClaim:
    set_id: str
    partition: str
    fold: int | None = ...
    assigned_by: str | None = ...
    assigned_at: str | None = ...
    manifest_sha256: str | None = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> SplitClaim: ...
    def to_json(self) -> dict[str, Any]:
        """The record as it stands, as JSON."""
        ...

@final
class Timeline:
    """The sample's timepoints, in acquisition order; indexable by position or id.

    The timepoints are the timeline's own objects, as 1.x kept them --- those
    it was given, and from the first read those it hands out --- so an edit to
    one's `series_uids` is the timeline's, and for a document's timeline, the
    document's.
    """
    def __new__(cls, timepoints: Sequence[Timepoint]) -> Self: ...
    def __contains__(self, key: Any, /) -> bool:
        """Return bool(key in self)."""
        ...
    def __eq__(self, other: object, /) -> bool: ...
    def __getitem__(self, key: Any, /) -> Any:
        """Return self[key]."""
        ...
    __hash__: ClassVar[None]  # type: ignore[assignment]
    def __iter__(self) -> Iterator[Timepoint]:
        """Implement iter(self)."""
        ...
    def __len__(self) -> int: ...
    @property
    def baseline(self) -> Timepoint: ...
    def check(self) -> None: ...
    def count(self, value: Any) -> int:
        """How many timepoints equal `value` (`Sequence.count`)."""
        ...
    @classmethod
    def from_json(cls, docs: Sequence[Mapping[str, Any]]) -> Timeline: ...
    @property
    def ids(self) -> tuple[str, ...]: ...
    def index(self, value: Any, start: int = 0, stop: int | None = None) -> int:
        """The position of the first timepoint equal to `value` (`Sequence.index`)."""
        ...
    def interval_days(self, a: str, b: str) -> float | None: ...
    @property
    def is_longitudinal(self) -> bool: ...
    def require(self, timepoint_id: str, *, where: Any = "") -> Timepoint: ...
    @classmethod
    def single(cls, timepoint_id: str = "tp0", **kwargs: Any) -> Timeline: ...
    def to_json(self) -> list[dict[str, Any]]:
        """The timeline as it stands, as JSON."""
        ...

@final
@dataclass(frozen=True)
class Timepoint:
    id: str
    index: int
    label: str | None = ...
    date: str | None = ...
    days_from_baseline: float | None = ...
    study_uid: str | None = ...
    series_uids: dict[str, str] = ...
    subject_age_years: float | None = ...
    description: str | None = ...
    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Timepoint: ...
    def to_json(self) -> dict[str, Any]:
        """The record as it stands, as JSON."""
        ...

@final
@dataclass(frozen=True, init=False)
class Track:
    """Every observation of one physical object, ordered by timepoint."""

    instance_id: int
    class_ids: tuple[int, ...]
    observations: tuple[Observation, ...]
    class_key: str | None
    def __iter__(self) -> Iterator[Observation]:
        """Implement iter(self)."""
        ...
    def __len__(self) -> int: ...
    def at(self, timepoint: str) -> Observation | None:
        """The observation that answers for `timepoint` --- the one
        `measurements_at` finds --- or `None` where the object was not seen.
        Refused (`MEDH5ValidationError`) where it finds two, as two raters'
        masks of one visit: which one is meant is the caller's to choose.
        `volume`, `volumes` and `relative_change` answer by the same rule.
        """
        ...
    @property
    def class_id(self) -> int: ...
    @property
    def has_class_conflict(self) -> bool: ...
    def measurements_at(self, timepoint: str) -> tuple[Observation, ...]:
        """The observations that measure the object at `timepoint`: its masks
        (`instances`), or --- where no mask saw it --- its boxes, axis-aligned
        or oriented.  A point marks the object and measures nothing, so it
        answers only where nothing else saw it.
        """
        ...
    def observations_at(self, timepoint: str) -> tuple[Observation, ...]:
        """Every observation at `timepoint`: one per annotation that saw the
        object there --- several where raters annotated the same visit, or a
        detection sits beside a segmentation.
        """
        ...
    def relative_change(self, first: str, second: str) -> float | None:
        """`(v2 - v1) / v1` between the volumes at `first` and `second`, compared
        in millimetres; `None` where either is unmeasured, or the two share no
        physical measure (an uncalibrated grid beside a calibrated one, an
        area beside a volume).
        """
        ...
    @property
    def timepoints(self) -> tuple[str, ...]: ...
    def to_json(self) -> dict[str, Any]: ...
    def volume(self, timepoint: str) -> float | None:
        """The volume of the observation `at` finds, or `None`; refused where
        `at` is.
        """
        ...
    @property
    def volumes(self) -> dict[str, float | None]:
        """`{timepoint: volume}` for every timepoint the object was seen at,
        under the rule of `at`.
        """
        ...

@final
class Tracking:
    """`{instance_id: Track}` with the coverage needed to tell *resolved* from
    *unexamined*, and the reason for every absence.
    """
    def __contains__(self, key: Any, /) -> bool:
        """Return bool(key in self)."""
        ...
    def __getitem__(self, key: Any, /) -> Track:
        """Return self[key]."""
        ...
    def __iter__(self) -> Iterator[int]:
        """Implement iter(self)."""
        ...
    def __len__(self) -> int: ...
    def class_conflicts(self) -> dict[int, tuple[int, ...]]: ...
    @property
    def coverage(self) -> dict[Any, Any]:
        """`{timepoint: frozenset of class ids}`: the classes the annotators at
        each timepoint committed to finding (§11.3), of the annotations there
        that ignore no part of their grid (§7.7).  Class-level: it does not
        say where anyone looked, which `reason_at` does.
        """
        ...
    def get(self, instance_id: Any, default: Any = None) -> Any: ...
    def is_new(self, instance_id: int, *, located: bool = False) -> bool:
        """Absent-then-present: `resolved` at the first timepoint (looked for
        and not seen) and `present` at a later one.  With `located`, only
        where the object was located at the first (not by class alone).
        """
        ...
    def is_persistent(self, instance_id: int) -> bool: ...
    def is_resolved(self, instance_id: int, *, located: bool = False) -> bool:
        """Present-then-gone: `present` at the first timepoint and `resolved` at
        the last, whose annotators looked for it.  With `located`, only where
        the object was located at the last.
        """
        ...
    def items(self) -> tuple[tuple[int, Track], ...]: ...
    def keys(self) -> tuple[int, ...]: ...
    def reason_at(self, instance_id: int, timepoint: str) -> str | None:
        """Why the object is absent at `timepoint` (`REASONS`): `located` or
        `class_examined` where it is `resolved`, `outside_fov`, `ignored`,
        `unrelated`, `no_identity` or `not_examined` where it is
        `unexamined`; `None` where it was seen.
        """
        ...
    def reasons(self, instance_id: int) -> dict[str, str | None]:
        """`{timepoint: reason_at(instance_id, timepoint)}` over every
        timepoint, in order.
        """
        ...
    @property
    def skipped(self) -> dict[str, str]:
        """`{annotation: why}` for the annotations that carry `instance_ids`
        and that no track joined.
        """
        ...
    def state_at(self, instance_id: int, timepoint: str) -> str:
        """`present`, `resolved` or `unexamined`.  `resolved` is claimed only
        where the object was looked for there (`reason_at` says how); anywhere
        else an unseen object is `unexamined`, and its absence measures
        nothing.  `KeyError` for an id no track has.
        """
        ...
    def states(self, instance_id: int) -> dict[str, str]:
        """`{timepoint: state_at(instance_id, timepoint)}` over every timepoint,
        in order.
        """
        ...
    def summary(self) -> dict[str, Any]: ...
    @property
    def timepoints(self) -> tuple[str, ...]: ...
    def to_json(self) -> dict[str, Any]: ...
    def unexamined(self) -> dict[str, tuple[int, ...]]:
        """`{timepoint: instance ids}` of the objects that are `unexamined`
        there: not seen, and of a class nobody there examined without
        ignoring part of the volume.
        """
        ...
    def values(self) -> tuple[Track, ...]: ...

@final
class TransformHandle:
    @staticmethod
    def can_invert(inner: TransformHandle) -> bool: ...
    @staticmethod
    def chain(steps: Sequence[TransformHandle]) -> TransformHandle: ...
    def check_chain(self) -> list[str]: ...
    @property
    def class_name(self) -> str: ...
    @property
    def component_ids(self) -> tuple[Any, ...]: ...
    def components(self) -> list[TransformHandle]: ...
    @property
    def control_points(self) -> Any: ...
    @property
    def cp_grid(self) -> Grid: ...
    @property
    def cp_grid_id(self) -> str: ...
    def displacement_at(self, points: Any) -> Any: ...
    @property
    def extrapolation(self) -> str: ...
    @property
    def field(self) -> Dataset: ...
    @property
    def field_grid(self) -> Grid: ...
    @property
    def field_grid_id(self) -> str: ...
    def folding_fraction(self, roi: Any = None) -> float: ...
    @property
    def from_frame(self) -> str: ...
    def grid_in(self, frame: str) -> Grid | None: ...
    @property
    def group(self) -> Group:
        """The transform's stored group, for inspection."""
        ...
    @property
    def header(self) -> TransformHeader: ...
    @property
    def interpolation(self) -> str: ...
    def inverse(self) -> TransformHandle | None: ...
    def inverse_matrix(self) -> Any: ...
    @staticmethod
    def inverse_of(inner: TransformHandle) -> TransformHandle: ...
    def inverse_points(self, points: Any) -> Any: ...
    @property
    def is_invertible(self) -> bool: ...
    def jacobian_determinant(self, roi: Any = None) -> Any: ...
    @property
    def jacobian_determinant_value(self) -> float: ...
    @property
    def kind(self) -> str: ...
    @property
    def matrix(self) -> Any: ...
    @property
    def max_magnitude(self) -> float: ...
    @property
    def metrics_key(self) -> str | None: ...
    @property
    def n_spatial(self) -> int: ...
    @property
    def order(self) -> int: ...
    @property
    def prov(self) -> str | None: ...
    def read_field(self, roi: Any = None, component: int | None = None) -> Any: ...
    def sample_indices(self, indices: Any) -> Any:
        """The stored field interpolated at continuous field indices, `(N, S)`."""
        ...
    @property
    def steps(self) -> list[TransformHandle] | None: ...
    def summary(self) -> Any: ...
    @property
    def timepoints(self) -> tuple[Any, ...]: ...
    def to_displacement_field(self, grid: Any) -> Any: ...
    @property
    def to_frame(self) -> str: ...
    @property
    def transform_id(self) -> str: ...
    def transform_points(self, points: Any) -> Any: ...
    @property
    def units(self) -> str: ...
    @property
    def vector_space(self) -> str: ...

@final
@dataclass(frozen=True)
class TransformHeader:
    """The attribute header every transform carries (spec §10.1)."""

    kind: str
    from_frame: str
    to_frame: str
    units: str = ...
    from_grid: str | None = ...
    to_grid: str | None = ...
    invertible: bool | None = ...
    inverse_id: str | None = ...
    prov: str | None = ...
    metrics: str | None = ...
    extra: Mapping[str, Any] = ...
    def attrs(self) -> dict[str, Any]: ...
    @classmethod
    def read(cls, group: Group) -> TransformHeader:
        """The header a transform group carries (`sample.root["transforms/x"]`)."""
        ...
