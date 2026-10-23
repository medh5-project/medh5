//! The 0.x on-disk layout, read-only, and its migration to 1.0 (Appendix B).
//!
//! ```text
//! /images/<name>   one dataset per modality, all the same shape
//! /seg/<name>      one boolean dataset per mask name
//! /bboxes          (n, ndim, 2) integers, slice-like [min, max)
//! /bbox_scores     (n,) floats            (optional)
//! /bbox_labels     (n,) strings           (optional)
//! root attrs       schema_version, label, label_name, extra (JSON), ...
//!                  (and shape, up to 0.5)
//! /images attrs    shape, spacing, origin, direction (flattened), axis_labels,
//!                  coord_system, patch_size
//! ```
//!
//! The denormalised 0.x flags (`has_seg`, `seg_names`, `image_names`) are
//! ignored in favour of what the file actually contains.  Four migration
//! steps are not mechanical and each is reported: the voxel encoding, the
//! half-voxel box shift, the minted label set, and the grouping.  So is the
//! geometry a file does not state, which is never invented: without a spacing
//! the grid is uncalibrated (`px`, §3.5).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use ndarray::ArrayD;
use serde_json::{json, Map, Value};

use super::grouping::{
    group_by_subject, is_fallback, note_instance_ids, output_name, sanitize_key, sanitize_stem, Occasion, SubjectGroup,
    FALLBACK_PREFIX,
};
use super::report::ConversionReport;
use crate::annotations::Assertions;
use crate::array::NdArray;
use crate::curation::provenance::is_timestamp;
use crate::curation::quality::QUALITY_STATUS;
use crate::h5::attrs::{self, AttrValue};
use crate::h5::file::same_file;
use crate::h5::{data, ops};
use crate::json::repr_str;
use crate::labels::{ClassKey, LabelClass, LabelSet, MAX_CLASS_ID};
use crate::sample::{
    create, Annotated, AnnotationOptions, GridOptions, ImageOptions, ObjectFields, Placement, QualityArg,
    SegmentationOptions, SegmentationSource,
};
use crate::{Error, Result, VERSION};

/// The only 0.x schema version that ever shipped.
pub const SCHEMA_VERSION: &str = "1";
/// 0.x `[min, max)` integer boxes sit at voxel edges once shifted (§8.1).
pub const BOX_SHIFT: f64 = -0.5;

/// A 0.x sample-level label.
#[derive(Debug, Clone, PartialEq)]
pub enum LegacyLabel {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
}

impl LegacyLabel {
    /// Python's `str()`.
    pub fn text(&self) -> String {
        match self {
            LegacyLabel::Int(i) => i.to_string(),
            LegacyLabel::Float(f) => crate::json::py_float(*f),
            LegacyLabel::Bool(b) => if *b { "True" } else { "False" }.into(),
            LegacyLabel::Text(t) => t.clone(),
        }
    }
}

/// 0.x geometry: one grid shared by every image in the file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LegacySpatial {
    pub spacing: Option<Vec<f64>>,
    pub origin: Option<Vec<f64>>,
    pub direction: Option<Vec<Vec<f64>>>,
    pub axis_labels: Option<Vec<String>>,
    pub coord_system: Option<String>,
}

/// 0.x metadata, as far as a migration needs it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LegacyMeta {
    pub spatial: LegacySpatial,
    pub shape: Option<Vec<i64>>,
    pub image_names: Vec<String>,
    pub seg_names: Vec<String>,
    pub label: Option<LegacyLabel>,
    pub label_name: Option<String>,
    pub patch_size: Option<Vec<i64>>,
    pub extra: Map<String, Value>,
    pub schema_version: String,
}

/// A whole 0.x file in memory.
#[derive(Debug, Clone)]
pub struct LegacySample {
    pub images: IndexMap<String, NdArray>,
    pub seg: IndexMap<String, ArrayD<bool>>,
    pub bboxes: Option<ArrayD<f64>>,
    pub bbox_scores: Option<Vec<f64>>,
    pub bbox_labels: Option<Vec<String>>,
    pub meta: LegacyMeta,
}

fn open_legacy(path: &Path) -> Result<hdf5::File> {
    crate::h5::init();
    let text = path.to_string_lossy().into_owned();
    let handle = crate::h5::file::marked(|| hdf5::File::open(path))
        .map_err(|e| Error::File(format!("cannot open {} as a 0.x file: {e}", repr_str(&text))))?;
    crate::h5::ops::check_self_contained(&handle, Some(path))?;
    if ops::exists(&handle, "meta") {
        return Err(Error::Schema(format!("{} is a 1.0 file (it has `/meta`), not a 0.x file", repr_str(&text))));
    }
    if ops::child_group(&handle, "images").is_none() {
        return Err(Error::Schema(format!("{} has no 0.x `/images` group", repr_str(&text))));
    }
    let version = attrs::get_str(&handle, "schema_version")?.unwrap_or_else(|| SCHEMA_VERSION.into());
    if version != SCHEMA_VERSION {
        return Err(Error::Schema(format!(
            "{} declares 0.x schema version {}; this reader understands {}",
            repr_str(&text),
            repr_str(&version),
            repr_str(SCHEMA_VERSION)
        )));
    }
    Ok(handle)
}

/// Whether `path` is a readable 0.x file.
pub fn is_legacy(path: &Path) -> bool {
    open_legacy(path).is_ok()
}

fn read_meta_from(handle: &hdf5::File) -> Result<LegacyMeta> {
    let mut meta = LegacyMeta::default();
    let group = handle.group("images")?;
    meta.image_names = ops::members(&group)?;
    let spatial = &mut meta.spatial;
    spatial.spacing = attrs::get_f64s(&group, "spacing")?;
    spatial.origin = attrs::get_f64s(&group, "origin")?;
    spatial.axis_labels = attrs::get_strs(&group, "axis_labels")?;
    spatial.coord_system = attrs::get_str(&group, "coord_system")?;
    // 0.x up to 0.5 wrote `shape` on the root; 0.6 moved it to `/images`.
    meta.shape = match attrs::get_i64s(&group, "shape")? {
        Some(shape) => Some(shape),
        None => attrs::get_i64s(handle, "shape")?,
    };
    meta.patch_size = attrs::get_i64s(&group, "patch_size")?;
    if let (Some(raw), Some(first)) = (attrs::get_f64s(&group, "direction")?, meta.image_names.first()) {
        let ndim = group.dataset(first)?.ndim();
        if raw.len() != ndim * ndim {
            return Err(Error::Schema(format!(
                "0.x `direction` has {} element(s), not {} for a {ndim}-D volume",
                raw.len(),
                ndim * ndim
            )));
        }
        meta.spatial.direction = Some(raw.chunks(ndim.max(1)).map(<[f64]>::to_vec).collect());
    }
    meta.label = match attrs::read(handle, "label")? {
        None => None,
        Some(AttrValue::Int(i)) => Some(LegacyLabel::Int(i)),
        Some(AttrValue::Float(f)) => Some(LegacyLabel::Float(f)),
        Some(AttrValue::Bool(b)) => Some(LegacyLabel::Bool(b)),
        Some(other) => {
            let text = other.as_str().unwrap_or_else(|| attrs::stringify_value(&other));
            Some(match text.trim().parse::<i64>() {
                Ok(i) => LegacyLabel::Int(i),
                Err(_) => LegacyLabel::Text(text),
            })
        }
    };
    meta.label_name = attrs::get_str(handle, "label_name")?;
    if let Some(text) = attrs::get_str(handle, "extra")? {
        let parsed =
            crate::json::loads(&text).map_err(|e| Error::Schema(format!("0.x attribute 'extra' is not JSON: {e}")))?;
        match parsed {
            Value::Object(m) => meta.extra = m,
            Value::Null => {}
            // Kept under the attribute's own name, so it reaches
            // `extra.legacy` as every key of an object does; it was dropped.
            other => {
                meta.extra.insert("extra".into(), other);
            }
        }
    }
    meta.schema_version = attrs::get_str(handle, "schema_version")?.unwrap_or_else(|| SCHEMA_VERSION.into());
    meta.seg_names = match ops::child_group(handle, "seg") {
        Some(g) => ops::members(&g)?,
        None => Vec::new(),
    };
    Ok(meta)
}

/// Read 0.x metadata without touching the arrays.
pub fn read_meta(path: &Path) -> Result<LegacyMeta> {
    read_meta_from(&open_legacy(path)?)
}

/// The `bbox_labels` of a 0.x file, without reading anything else.
pub fn read_bbox_labels(path: &Path) -> Result<Option<Vec<String>>> {
    let handle = open_legacy(path)?;
    match ops::child_dataset(&handle, "bbox_labels") {
        None => Ok(None),
        Some(ds) => Ok(Some(data::read_strings(&ds)?)),
    }
}

/// Read a whole 0.x file: images, masks, boxes and metadata.
pub fn read_sample(path: &Path) -> Result<LegacySample> {
    let handle = open_legacy(path)?;
    let group = handle.group("images")?;
    let mut images = IndexMap::new();
    for name in ops::members(&group)? {
        images.insert(name.clone(), data::read(&group.dataset(&name)?)?);
    }
    let mut seg = IndexMap::new();
    if let Some(g) = ops::child_group(&handle, "seg") {
        for name in ops::members(&g)? {
            seg.insert(name.clone(), data::read(&g.dataset(&name)?)?.nonzero_mask());
        }
    }
    let bboxes = match ops::child_dataset(&handle, "bboxes") {
        Some(ds) => Some(data::read(&ds)?.to_f64()),
        None => None,
    };
    let bbox_scores = match ops::child_dataset(&handle, "bbox_scores") {
        Some(ds) => Some(data::read(&ds)?.to_f64().iter().copied().collect()),
        None => None,
    };
    let bbox_labels = match ops::child_dataset(&handle, "bbox_labels") {
        Some(ds) => Some(data::read_strings(&ds)?),
        None => None,
    };
    let meta = read_meta_from(&handle)?;
    Ok(LegacySample { images, seg, bboxes, bbox_scores, bbox_labels, meta })
}

fn key(name: &str) -> String {
    sanitize_key(name, "class")
}

/// The class a 0.x sample label names: its `label_name`, or the label's own
/// text when it has none (Appendix B maps both to a classification).
fn label_class(meta: &LegacyMeta) -> Option<String> {
    match (&meta.label_name, &meta.label) {
        (Some(name), _) if !name.is_empty() => Some(name.clone()),
        (_, Some(label)) => Some(label.text()),
        _ => None,
    }
}

/// A sample id from a subject id: §2.3's identifier rule, lowercased.
pub fn sample_key(subject_id: &str) -> String {
    let stem = sanitize_stem(subject_id.trim(), 128);
    if stem.is_empty() {
        "sample".to_string()
    } else {
        stem.to_lowercase()
    }
}

/// Mint one label set covering a whole cohort's mask names and box labels.
///
/// Cohort-wide rather than per file, so `liver` has one id everywhere; ids an
/// `extra.nnunetv2.labels` mapping already fixed are reused.  A class is a
/// key, so names that sanitize to one key (`liver`, `Liver`) are one class,
/// and the report names them; two files' mappings that disagree keep the
/// first one's id, as does an id two keys ask for, and each is a warning.
/// Keyed by name instead, `liver` and `Liver` were two classes with one key,
/// and the label set a curator is asked to review was refused (E302).  A
/// file that cannot be read is an `unreadable` warning, and the label set is
/// minted from the rest.
pub fn build_label_set(paths: &[&Path], report: Option<&mut ConversionReport>) -> Result<LabelSet> {
    mint_label_set(paths, report, true)
}

/// [`build_label_set`]; `unreadable` says whether the report hears of a file
/// that could not be read, which [`migrate_paths`] reports itself.
fn mint_label_set(paths: &[&Path], mut report: Option<&mut ConversionReport>, unreadable: bool) -> Result<LabelSet> {
    let mut names: IndexMap<String, Vec<String>> = IndexMap::new();
    let mut add = |name: &str| {
        let spellings = names.entry(key(name)).or_default();
        if !spellings.iter().any(|n| n == name) {
            spellings.push(name.to_string());
        }
    };
    // key -> (id, the file whose mapping gave it)
    let mut reused: IndexMap<String, (i64, String)> = IndexMap::new();
    let mut conflicts: Vec<(String, String, i64, String, i64)> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    for path in paths {
        let source = path.to_string_lossy().into_owned();
        let meta = match read_meta(path) {
            Ok(meta) => meta,
            Err(e) => {
                // Skipped in silence, its classes were missing from the label
                // set a curator reviews, with exit status 0.
                skipped.push((source, e.to_string()));
                continue;
            }
        };
        for name in &meta.seg_names {
            add(name);
        }
        if let Some(Value::Object(labels)) = meta.extra.get("nnunetv2").and_then(|n| n.get("labels")) {
            for (label, value) in labels {
                let Some(v) = value.as_i64().filter(|v| (1..=MAX_CLASS_ID).contains(v) && value.is_i64()) else {
                    continue;
                };
                match reused.get(&key(label)) {
                    None => {
                        reused.insert(key(label), (v, source.clone()));
                    }
                    Some((kept, first)) if *kept != v => {
                        conflicts.push((key(label), first.clone(), *kept, source.clone(), v));
                    }
                    Some(_) => {}
                }
            }
        }
        let has_boxes = open_legacy(path).map(|h| ops::exists(&h, "bboxes")).unwrap_or(false);
        match read_bbox_labels(path) {
            Ok(Some(labels)) => {
                for label in labels {
                    add(&label);
                }
            }
            // Unlabelled 0.x boxes migrate as class `object`, which the label
            // set must therefore declare.
            Ok(None) if has_boxes => add("object"),
            _ => {}
        }
        // The sample label is a class too (Appendix B); minted last, so the
        // ids of a cohort's mask and box classes do not move.
        if let Some(name) = label_class(&meta) {
            add(&name);
        }
    }
    // An id goes to the first key whose mapping asks for it; a later key
    // asking for it is minted another.
    let mut owner: IndexMap<i64, String> = IndexMap::new();
    let mut shared: Vec<(String, String, i64)> = Vec::new();
    for (k, (id, _)) in &reused {
        if !names.contains_key(k) {
            continue;
        }
        match owner.get(id) {
            Some(first) => shared.push((k.clone(), first.clone(), *id)),
            None => {
                owner.insert(*id, k.clone());
            }
        }
    }
    let mut classes = Vec::new();
    let mut used: BTreeSet<i64> = reused.values().map(|(id, _)| *id).collect();
    let mut next_id = 1;
    for (k, spellings) in &names {
        let class_id = match reused.get(k).filter(|(id, _)| owner.get(id) == Some(k)) {
            Some((id, _)) => *id,
            None => {
                while used.contains(&next_id) {
                    next_id += 1;
                }
                used.insert(next_id);
                next_id
            }
        };
        classes.push(LabelClass::new(class_id, k.clone(), spellings[0].clone())?);
    }
    if let Some(log) = report.as_deref_mut() {
        if unreadable {
            for (source, e) in &skipped {
                log.warn(
                    "unreadable",
                    format!("{source}: {e}; none of its classes are in the label set"),
                    json!({"path": source}),
                );
            }
        }
        for (k, spellings) in names.iter().filter(|(_, s)| s.len() > 1) {
            log.guess(
                "label_set",
                format!(
                    "names {} all became class key {}, so they are one class; edit the label set if they are not",
                    crate::json::repr_list(spellings),
                    repr_str(k)
                ),
                json!({"key": k, "names": spellings}),
            );
        }
        for (k, first, kept, source, other) in &conflicts {
            log.warn(
                "label_set",
                format!(
                    "{source}: extra.nnunetv2.labels gives {} id {other}, and {first}'s gives it {kept}; the label set \
                     keeps {kept}",
                    repr_str(k)
                ),
                json!({"key": k, "kept": kept, "kept_from": first, "ignored": other, "source": source}),
            );
        }
        for (k, first, id) in &shared {
            let minted = classes.iter().find(|c| &c.key == k).map(|c| c.id).unwrap_or_default();
            log.warn(
                "label_set",
                format!(
                    "extra.nnunetv2.labels gives id {id} to {} and to {}; {} keeps it, and {} is id {minted}",
                    repr_str(first),
                    repr_str(k),
                    repr_str(first),
                    repr_str(k)
                ),
                json!({"id": id, "kept_by": first, "key": k, "minted": minted}),
            );
        }
    }
    if let Some(log) = report {
        let ids: Map<String, Value> = classes.iter().map(|c| (c.key.clone(), json!(c.id))).collect();
        let mut reused_keys: Vec<&String> = reused.keys().collect();
        reused_keys.sort();
        log.decision(
            "label_set",
            format!(
                "{} class(es) were minted across {} file(s); {}",
                classes.len(),
                paths.len(),
                if reused.is_empty() {
                    "no existing id mapping was found, so ids are sequential".to_string()
                } else {
                    format!("{} id(s) came from an existing extra.nnunetv2.labels mapping", reused.len())
                }
            ),
            json!({"ids": ids, "reused": reused_keys}),
        );
    }
    LabelSet::new("migrated", classes, "1.0.0", Vec::new(), Vec::new(), "inline", None, None)
}

/// Migrate one 0.x file into one 1.0 sample.
///
/// The sample is named as [`migrate_paths`] names a study it gives no
/// subject: `sample_id` from the file's stem, `subject_id` `study:<that>`.
/// The stem was the subject id, which inferred identity from a file name.
pub fn migrate(
    path: &Path,
    out: &Path,
    label_set: Option<&LabelSet>,
    codec: &str,
    report: Option<ConversionReport>,
) -> Result<ConversionReport> {
    if same_file(path, out) {
        return Err(Error::invalid(format!(
            "{} is the 0.x file being migrated; write the sample to a new path",
            repr_str(&out.to_string_lossy())
        )));
    }
    let mut log = report.unwrap_or_else(|| ConversionReport::new("migrate", ""));
    log.source = path.to_string_lossy().into_owned();
    let labels = match label_set {
        Some(l) => l.clone(),
        None => build_label_set(&[path], Some(&mut log))?,
    };
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let name = sample_key(&stem);
    let group = SubjectGroup {
        subject_id: format!("{FALLBACK_PREFIX}:{name}"),
        occasions: vec![Occasion { key: log.source.clone(), payload: 0, ..Default::default() }],
        ordered_by: "given".into(),
    };
    write_group(&group, &name, &[path.to_path_buf()], out, &labels, codec, &mut log)?;
    Ok(log)
}

fn date_of(meta: &LegacyMeta) -> Option<String> {
    for k in ["study_date", "date", "acquisition_date"] {
        if let Some(v) = meta.extra.get(k) {
            let truthy = match v {
                Value::Null => false,
                Value::Bool(b) => *b,
                Value::String(s) => !s.is_empty(),
                Value::Number(n) => n.as_f64() != Some(0.0),
                Value::Array(a) => !a.is_empty(),
                Value::Object(o) => !o.is_empty(),
            };
            if truthy {
                return Some(crate::json::py_str(v));
            }
        }
    }
    None
}

fn subject_key_of(meta: &LegacyMeta, path_key: Option<&str>) -> Option<String> {
    let path_key = path_key.filter(|k| !k.is_empty())?;
    let mut node = json!({"extra": meta.extra});
    for part in path_key.split('.') {
        node = node.as_object()?.get(part)?.clone();
    }
    match &node {
        Value::Null => None,
        Value::String(s) if s.is_empty() => None,
        other => Some(crate::json::py_str(other)),
    }
}

/// Migrate a cohort, minting one label set for all of it.
///
/// A file that is not 0.x is reported (`unreadable`), and so is one whose
/// content cannot be written (`failed`, with nothing written for the sample
/// it belongs to); the rest of the cohort is migrated.
pub fn migrate_paths(
    paths: &[&Path],
    outdir: &Path,
    group_by: &str,
    subject_key: Option<&str>,
    label_set: Option<&LabelSet>,
    codec: &str,
) -> Result<ConversionReport> {
    let MigrationPlan { mut log, labels, groups } = plan_migration(paths, outdir, group_by, subject_key, label_set)?;
    for group in &groups {
        let written = migrate_group(group, &labels, codec);
        log.outputs.extend(written.outputs);
        log.notes.extend(written.notes);
    }
    Ok(log)
}

/// What [`migrate_paths`] decides before it writes a sample: the report so
/// far (the label set it minted, the files it could not read, the grouping),
/// the label set, and each sample to write, in the order it writes them.
/// Public for the frontends: `medh5 migrate --workers` writes the samples
/// in several processes, each with [`migrate_group`].
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    pub log: ConversionReport,
    pub labels: LabelSet,
    pub groups: Vec<MigrationGroup>,
}

/// One sample of a [`MigrationPlan`]: its subject's files, as grouped, and
/// where it goes.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct MigrationGroup {
    /// The occasions in timepoint order; each `payload` indexes `sources`.
    pub group: SubjectGroup,
    pub sources: Vec<PathBuf>,
    /// The sample's id, unique in the run, and its file's stem.
    pub name: String,
    pub target: PathBuf,
    /// The input `target` is, when it is one (`-o .`): nothing is written.
    pub clash: Option<PathBuf>,
}

/// [`migrate_paths`] up to its first write.
#[doc(hidden)]
pub fn plan_migration(
    paths: &[&Path],
    outdir: &Path,
    group_by: &str,
    subject_key: Option<&str>,
    label_set: Option<&LabelSet>,
) -> Result<MigrationPlan> {
    std::fs::create_dir_all(outdir)?;
    let mut log = ConversionReport::new("migrate", &format!("{} file(s)", paths.len()));
    let labels = match label_set {
        Some(l) => l.clone(),
        // A file it cannot read is reported below, once.
        None => mint_label_set(paths, Some(&mut log), false)?,
    };
    let mut occasions = Vec::new();
    let mut sources = Vec::new();
    for path in paths {
        let text = path.to_string_lossy().into_owned();
        let meta = match read_meta(path) {
            Ok(m) => m,
            Err(e) => {
                log.warn("unreadable", format!("{text}: {e}"), json!({"path": text}));
                continue;
            }
        };
        let mtime = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64());
        occasions.push(Occasion {
            key: text,
            subject_id: subject_key_of(&meta, subject_key),
            date: date_of(&meta),
            order_hint: mtime,
            demographics: IndexMap::new(),
            payload: sources.len(),
        });
        sources.push(path.to_path_buf());
    }
    if group_by == "subject" && subject_key.is_none() {
        log.warn(
            "grouping",
            "--group-by subject needs --subject-key: a 0.x file has no subject field of its own, and identity is never \
             inferred from filenames",
            json!({}),
        );
    }
    let mut groups = Vec::new();
    let mut used = BTreeSet::new();
    for mut group in group_by_subject(occasions, group_by, Some(&mut log))? {
        // The output's name is the sample's id, unique in the run: a
        // subject's studies, or keys differing only in case, shared one.
        let name = output_name(&group, &mut used, &|s: &str| sample_key(s));
        // A study with no subject was `study:<its path>`, which put a local
        // path --- often an MRN or a name --- in the identity, and made
        // `content_id` depend on where the file lay.
        if is_fallback(&group) {
            group.subject_id = format!("{FALLBACK_PREFIX}:{name}");
        }
        let target = outdir.join(format!("{name}.medh5"));
        // `-o .` wrote each sample over its 0.x original.
        let clash = paths.iter().find(|p| same_file(p, &target)).map(|p| p.to_path_buf());
        let mut own = Vec::new();
        for occasion in &mut group.occasions {
            own.push(sources[occasion.payload].clone());
            occasion.payload = own.len() - 1;
        }
        groups.push(MigrationGroup { group, sources: own, name, target, clash });
    }
    Ok(MigrationPlan { log, labels, groups })
}

/// Write one sample of a [`MigrationPlan`]: what [`migrate_paths`] adds to
/// its report for it, its notes and its output.  A sample that cannot be
/// written is a `failed` note, never an error.
#[doc(hidden)]
pub fn migrate_group(planned: &MigrationGroup, labels: &LabelSet, codec: &str) -> ConversionReport {
    let mut log = ConversionReport::new("migrate", "");
    let MigrationGroup { group, sources, name, target, clash } = planned;
    let inputs: Vec<&str> = group.occasions.iter().map(|o| o.key.as_str()).collect();
    if let Some(input) = clash {
        log.warn(
            "failed",
            format!(
                "{}: the output {} is the input {}; nothing was written for it",
                inputs.join(", "),
                repr_str(&target.to_string_lossy()),
                repr_str(&input.to_string_lossy())
            ),
            json!({"inputs": inputs, "output": target.to_string_lossy()}),
        );
        return log;
    }
    // A file whose content cannot be migrated is reported, and the rest of
    // the cohort is: the first such file used to end the migration, with the
    // files after it never written.  What the group's notes described was
    // never written either, so they go.
    if let Err(e) = write_group(group, name, sources, target, labels, codec, &mut log) {
        log.notes.clear();
        log.warn(
            "failed",
            format!("{}: {e}; nothing was written for it", inputs.join(", ")),
            json!({"inputs": inputs, "output": target.to_string_lossy()}),
        );
    }
    log
}

fn write_group(
    group: &SubjectGroup,
    sample_id: &str,
    sources: &[PathBuf],
    target: &Path,
    label_set: &LabelSet,
    codec: &str,
    log: &mut ConversionReport,
) -> Result<()> {
    // An order by mtime is reported where the grouping chose it.
    note_instance_ids(group, log);
    let days = group.days_from_baseline();
    let timepoints = group.timepoint_ids();
    let mut writer = create(target, Some(sample_id), Some(&group.subject_id), codec, &[])?;
    let result = (|| -> Result<()> {
        writer.label_set(label_set.clone());
        let tool = writer.software("medh5", Some(VERSION), Map::new())?;
        for (index, (timepoint, d)) in timepoints.iter().zip(&days).enumerate() {
            let mut fields = Map::new();
            fields.insert("index".into(), json!(index));
            fields.insert("days_from_baseline".into(), json!(d));
            writer.add_timepoint(timepoint, fields)?;
        }
        let single = group.occasions.len() == 1;
        let mut extra = LegacyExtra::default();
        for (occasion, timepoint) in group.occasions.iter().zip(&timepoints) {
            let source = &sources[occasion.payload];
            let sample = read_sample(source)?;
            migrate_one(&mut writer, &sample, &source.to_string_lossy(), timepoint, label_set, &tool.id, log, single)?;
            extra.add(&sample.meta.extra, timepoint, &source.to_string_lossy(), log);
        }
        extra.write(&mut writer);
        writer.commit(true)?;
        Ok(())
    })();
    if let Err(e) = result {
        writer.abort();
        return Err(e);
    }
    log.outputs.push(target.to_string_lossy().into_owned());
    Ok(())
}

/// The 0.x `extra` of a subject's files, as `/meta` keeps it (Appendix B).
///
/// `extra.nnunetv2` is preserved verbatim at `/meta → extra.nnunetv2`, where
/// the nnU-Net exporter reads a stashed `dataset.json`; the other keys are
/// kept under `extra.legacy`, one entry per timepoint.  Every file's whole
/// `extra` went to `extra.legacy`, so the stash was never where it is read,
/// and in a grouped migration each file's replaced the one before.  A later
/// file whose `nnunetv2` differs from the first's
/// keeps it in its own entry, and the report says so: one sample has one
/// `dataset.json`.
#[derive(Default)]
struct LegacyExtra {
    nnunetv2: Option<(Value, String)>,
    legacy: Map<String, Value>,
}

impl LegacyExtra {
    fn add(&mut self, extra: &Map<String, Value>, timepoint: &str, source: &str, log: &mut ConversionReport) {
        let mut rest = extra.clone();
        if let Some(stash) = rest.remove("nnunetv2") {
            match &self.nnunetv2 {
                None => self.nnunetv2 = Some((stash, source.to_string())),
                Some((kept, _)) if *kept == stash => {}
                Some((_, first)) => {
                    log.warn(
                        "nnunetv2",
                        format!(
                            "{source}: extra.nnunetv2 differs from {first}'s, which the sample keeps as its \
                             extra.nnunetv2; this one stays in extra.legacy.{timepoint}.nnunetv2"
                        ),
                        json!({"source": source, "kept": first, "timepoint": timepoint}),
                    );
                    rest.insert("nnunetv2".into(), stash);
                }
            }
        }
        if !rest.is_empty() {
            self.legacy.insert(timepoint.to_string(), Value::Object(rest));
        }
    }

    fn write(self, writer: &mut crate::sample::SampleWriter) {
        if let Some((stash, _)) = self.nnunetv2 {
            writer.extra("nnunetv2", stash);
        }
        if !self.legacy.is_empty() {
            writer.extra("legacy", Value::Object(self.legacy));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn migrate_one(
    writer: &mut crate::sample::SampleWriter,
    sample: &LegacySample,
    source: &str,
    timepoint: &str,
    label_set: &LabelSet,
    tool: &str,
    log: &mut ConversionReport,
    single: bool,
) -> Result<()> {
    let meta = &sample.meta;
    let suffix = if single { String::new() } else { format!("_{timepoint}") };
    let mut fields = Map::new();
    fields.insert("tool".into(), json!("medh5 migrate"));
    // The file's name, not its path, for the reason the identity is.
    let file = Path::new(source).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    fields.insert("inputs".into(), json!([format!("medh5-0.x:{file}")]));
    let activity = writer.activity("import", Some(tool), None, fields)?;
    let grid_id = format!("ref{suffix}");
    let spatial = &meta.spatial;
    let (_, first) = sample.images.first().ok_or_else(|| Error::Schema(format!("{source}: 0.x file has no images")))?;
    let shape: Vec<i64> = first.shape().iter().map(|v| *v as i64).collect();
    let direction = match &spatial.direction {
        Some(rows) => {
            Some(ndarray::Array2::from_shape_vec((rows.len(), rows.len()), rows.iter().flatten().copied().collect())?)
        }
        None => None,
    };
    // A 0.x file need not state its geometry.  Without a spacing the grid is
    // uncalibrated (§3.5) rather than one of millimetre voxels nobody
    // measured, and a coord_system it does not state is an assumption.
    let (spacing, units) = match &spatial.spacing {
        Some(spacing) => (spacing.clone(), None),
        None => {
            log.decision(
                "geometry",
                format!(
                    "{source}: the file states no spacing, so grid {} is uncalibrated --- units 'px' (§3.5), one per \
                     voxel --- rather than given a size nobody measured",
                    repr_str(&grid_id)
                ),
                json!({"source": source, "grid": grid_id, "units": "px"}),
            );
            (vec![1.0; shape.len()], Some("px".to_string()))
        }
    };
    let coord_system = match &spatial.coord_system {
        Some(coord_system) => coord_system.clone(),
        None => {
            log.guess(
                "geometry",
                format!(
                    "{source}: the file states no coord_system, and grid {} was written as 'LPS', which nothing in it \
                     confirms",
                    repr_str(&grid_id)
                ),
                json!({"source": source, "grid": grid_id, "coord_system": "LPS"}),
            );
            "LPS".to_string()
        }
    };
    writer.add_grid(
        &grid_id,
        &shape,
        &spacing,
        GridOptions {
            origin: spatial.origin.clone(),
            direction,
            axis_names: spatial.axis_labels.clone(),
            coord_system: Some(coord_system),
            units,
            timepoint: Some(timepoint.into()),
            patch_hint: meta.patch_size.clone(),
            ..Default::default()
        },
    )?;
    let mut names: Vec<&String> = sample.images.keys().collect();
    names.sort();
    for name in names {
        writer.add_image(
            &format!("{name}{suffix}"),
            &sample.images[name],
            &grid_id,
            "OT",
            ImageOptions { prov: Some(activity.id.clone()), ..Default::default() },
        )?;
    }
    // The review is of what the file held: each annotation migrated from it
    // carries the review's quality record, and the review names them.
    let review = match meta.extra.get("review") {
        Some(Value::Object(review)) => Some(LegacyReview::read(writer, review, tool, log, source)?),
        _ => None,
    };
    let quality = review.as_ref().map(|r| QualityArg::Record(r.quality.clone()));
    let mut written: Vec<String> = Vec::new();
    if !sample.seg.is_empty() {
        let mut seg_names: Vec<&String> = sample.seg.keys().collect();
        seg_names.sort();
        let mut masks: Vec<(ClassKey, ArrayD<bool>)> = Vec::new();
        let mut annotated = Vec::new();
        for name in &seg_names {
            let class = ClassKey::Id(label_set.lookup(&ClassKey::Key(key(name)))?.id);
            let mask = &sample.seg[*name];
            // Names of one class key are one class, so their masks are its
            // voxels together; passed apart, the second replaced the first.
            match masks.iter_mut().find(|(k, _)| *k == class) {
                Some((_, kept)) if kept.shape() == mask.shape() => kept.zip_mut_with(mask, |a, b| *a |= *b),
                Some(_) => {
                    return Err(Error::coded(
                        "E405",
                        format!("{source}: the masks of class {} differ in shape", repr_str(&key(name))),
                    ))
                }
                None => {
                    masks.push((class.clone(), mask.clone()));
                    annotated.push(class);
                }
            }
        }
        if masks.len() < seg_names.len() {
            log.decision(
                "label_set",
                format!(
                    "{source}: {} masks are {} classes; the masks of names that became one class key are stored as \
                     their union",
                    seg_names.len(),
                    masks.len()
                ),
                json!({"source": source, "masks": seg_names, "classes": masks.len()}),
            );
        }
        let ann_id = format!("seg{suffix}");
        let (kind, stats) = writer.add_segmentation(
            &ann_id,
            &grid_id,
            SegmentationSource::Masks(masks),
            SegmentationOptions {
                common: AnnotationOptions {
                    annotated_classes: Annotated::Classes(annotated),
                    prov: Some(activity.id.clone()),
                    quality: quality.clone(),
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        written.push(ann_id);
        log.decision(
            "encoding",
            format!("{source}: {} mask(s) were measured and stored as {}", seg_names.len(), repr_str(&kind)),
            json!({"source": source, "kind": kind, "overlapping_pairs": stats.map(|s| s.edges.len()).unwrap_or(0)}),
        );
        log.guess(
            "coverage",
            format!(
                "{source}: annotated_class_ids was set to the migrated mask names --- the only defensible inference. Widen \
                 or narrow it if the curator knows which classes were actually searched for (§11.3)"
            ),
            json!({"source": source, "classes": seg_names}),
        );
    }
    if let Some(boxes) = sample.bboxes.as_ref().filter(|b| b.shape().first().copied().unwrap_or(0) > 0) {
        let shifted = boxes.mapv(|v| v + BOX_SHIFT);
        let n = shifted.shape()[0];
        let labels = sample.bbox_labels.clone().unwrap_or_else(|| vec!["object".into(); n]);
        let class_ids: Vec<ClassKey> = labels
            .iter()
            .map(|l| Ok(ClassKey::Id(label_set.lookup(&ClassKey::Key(key(l)))?.id)))
            .collect::<Result<_>>()?;
        let stored = shifted.mapv(|v| f64::from(v as f32));
        let ann_id = format!("boxes{suffix}");
        writer.add_boxes(
            &ann_id,
            &stored,
            &class_ids,
            ObjectFields { scores: sample.bbox_scores.clone(), ..Default::default() },
            None,
            Placement { grid: Some(grid_id.clone()), space: Some("index".into()), frame_uid: None },
            AnnotationOptions {
                task: Some("detection".into()),
                prov: Some(activity.id.clone()),
                quality: quality.clone(),
                ..Default::default()
            },
        )?;
        written.push(ann_id);
        log.decision(
            "box_convention",
            format!(
                "{source}: {n} box(es) were shifted by -0.5 on every axis --- 0.x stored slice-like [min, max) integers, \
                 1.0 stores voxel edges, and the numbers differ by half a voxel (§8.1)"
            ),
            json!({"source": source, "boxes": n, "shift": BOX_SHIFT}),
        );
    }
    if let Some(name) = label_class(meta) {
        let class = key(&name);
        let raw = match &meta.label {
            None => Value::Null,
            Some(LegacyLabel::Int(i)) => json!(i),
            Some(LegacyLabel::Float(f)) => crate::json::num(*f),
            Some(LegacyLabel::Bool(b)) => json!(b),
            Some(LegacyLabel::Text(t)) => json!(t),
        };
        match label_set.get(&ClassKey::Key(class.clone())) {
            Some(entry) => {
                let ann_id = format!("label{suffix}");
                writer.add_classification(
                    &ann_id,
                    Assertions { class_ids: vec![entry.id], values: vec![1.0], ..Default::default() },
                    "sample",
                    true,
                    None,
                    AnnotationOptions {
                        timepoints: Some(vec![timepoint.into()]),
                        prov: Some(activity.id.clone()),
                        quality: quality.clone(),
                        ..Default::default()
                    },
                )?;
                let detail = json!({"source": source, "label": raw, "label_name": meta.label_name, "class": class,
                    "annotation": ann_id});
                let shown = match &meta.label {
                    Some(LegacyLabel::Text(t)) => repr_str(t),
                    Some(other) => other.text(),
                    None => String::new(),
                };
                let named = meta.label_name.as_deref().is_some_and(|n| !n.is_empty());
                match (&meta.label, named) {
                    (Some(_), true) => log.decision(
                        "label",
                        format!(
                            "{source}: sample label {shown} ({}) became classification {} of class {}",
                            repr_str(&name),
                            repr_str(&ann_id),
                            repr_str(&class)
                        ),
                        detail,
                    ),
                    (Some(_), false) => log.guess(
                        "label",
                        format!(
                            "{source}: sample label {shown} has no label_name, so its text became class {} of \
                             classification {}; rename the class if the value means something else, such as a 0 \
                             for absent",
                            repr_str(&class),
                            repr_str(&ann_id)
                        ),
                        detail,
                    ),
                    (None, _) => log.guess(
                        "label",
                        format!(
                            "{source}: label_name {} has no label, and classification {} asserts it as class {}",
                            repr_str(&name),
                            repr_str(&ann_id),
                            repr_str(&class)
                        ),
                        detail,
                    ),
                };
                written.push(ann_id);
            }
            None => {
                log.warn(
                    "label",
                    format!("{source}: sample label {} is not in the label set and was not migrated", repr_str(&name)),
                    json!({"source": source, "label": name}),
                );
            }
        }
    }
    if let Some(review) = review {
        review.write(writer, &written, log, source)?;
    }
    Ok(())
}

/// An activity time from a 0.x review time: an RFC 3339 timestamp (0.x wrote
/// `datetime.isoformat()`, with its `+00:00` offset) or a bare `YYYY-MM-DD`.
fn timestamp(value: &Value) -> Option<String> {
    let text = crate::json::py_str(value).trim().to_string();
    if is_timestamp(&text) {
        return Some(text);
    }
    let midnight = format!("{text}T00:00:00Z");
    (text.len() == 10 && is_timestamp(&midnight)).then_some(midnight)
}

/// A 0.x `extra.review`: who reviewed the file, and their verdict.
///
/// The 0.x API (`medh5.review`, 0.3 to 0.6) wrote `status` (`pending`,
/// `reviewed`, `flagged` or `rejected`), `annotator`, `timestamp`, `notes`
/// and `history`.  Only `reviewer`/`by` and `date`/`reviewed_at` were read,
/// so the activity of every record the API wrote had no agent and the file
/// failed to migrate (E005), and a status outside 1.0's became `reviewed`.
/// `pending` and `flagged` now map to `submitted`, a flag stays an issue, and
/// each such mapping is a guess; with no reviewer named, the import tool is
/// the agent.  The record rides on each annotation migrated from the file,
/// and the activity names those annotations.  The dict itself stays in the
/// file's `extra.legacy` entry, verbatim, with whatever else it held
/// (Appendix B).
struct LegacyReview {
    agent: String,
    verdict: Option<String>,
    status: String,
    ended: Option<String>,
    quality: Map<String, Value>,
}

impl LegacyReview {
    fn read(
        writer: &mut crate::sample::SampleWriter,
        review: &Map<String, Value>,
        tool: &str,
        log: &mut ConversionReport,
        source: &str,
    ) -> Result<LegacyReview> {
        let given = |k: &str| review.get(k).filter(|v| !v.is_null() && v.as_str() != Some(""));
        let verdict = given("status").map(crate::json::py_str);
        // 0.x's own default status was `pending`.
        let stated = verdict.as_deref().unwrap_or("pending").to_lowercase();
        let (status, why) = match stated.as_str() {
            s if QUALITY_STATUS.contains(&s) => (s.to_string(), None),
            "pending" if verdict.is_none() => {
                ("submitted".to_string(), Some("the record states no status, and 0.x's default was 'pending'"))
            }
            "pending" => ("submitted".to_string(), Some("0.x 'pending' means nobody had reviewed it yet")),
            "flagged" => (
                "submitted".to_string(),
                Some("0.x 'flagged' neither accepts nor rejects; the flag is kept as an issue"),
            ),
            _ => ("submitted".to_string(), Some("it is neither a 0.x nor a 1.0 review status")),
        };
        if let Some(why) = why {
            log.guess(
                "review",
                format!(
                    "{source}: review status {} became quality status {}: {why}",
                    repr_str(verdict.as_deref().unwrap_or("(none)")),
                    repr_str(&status)
                ),
                json!({"source": source, "verdict": verdict, "status": status}),
            );
        }
        let reviewer = given("reviewer").or_else(|| given("by")).or_else(|| given("annotator"));
        let person = match reviewer {
            Some(r) => Some(writer.person(&crate::json::py_str(r), None, Map::new())?.id),
            None => None,
        };
        let agent = match &person {
            Some(id) => id.clone(),
            None => {
                log.guess(
                    "review",
                    format!(
                        "{source}: extra.review names no reviewer, so its `review` activity is attributed to the import \
                         tool, which recorded it"
                    ),
                    json!({"source": source, "agent": tool}),
                );
                tool.to_string()
            }
        };
        let mut quality = Map::new();
        quality.insert("status".into(), json!(status));
        // A pending record names whoever set it, not a reviewer.
        let reviewed: Vec<&String> = person.iter().filter(|_| stated != "pending").collect();
        quality.insert("reviewed_by".into(), json!(reviewed));
        if stated == "flagged" {
            quality.insert("issues".into(), json!([{"code": "flagged", "severity": "warning"}]));
        }
        let stamp = given("timestamp").or_else(|| given("date")).or_else(|| given("reviewed_at"));
        let ended = stamp.and_then(timestamp);
        if let (Some(raw), None) = (stamp, &ended) {
            log.decision(
                "review",
                format!(
                    "{source}: review time {} is neither an RFC 3339 timestamp nor a date, so the `review` activity \
                     states no end; extra.legacy keeps it",
                    repr_str(&crate::json::py_str(raw))
                ),
                json!({"source": source, "time": raw}),
            );
        }
        Ok(LegacyReview { agent, verdict, status, ended, quality })
    }

    /// The `review` activity, whose outputs are the annotations reviewed.
    fn write(
        self,
        writer: &mut crate::sample::SampleWriter,
        annotations: &[String],
        log: &mut ConversionReport,
        source: &str,
    ) -> Result<()> {
        let mut fields = Map::new();
        if let Some(t) = &self.ended {
            fields.insert("ended".into(), json!(t));
        }
        if let Some(verdict) = &self.verdict {
            fields.insert("params".into(), json!({"verdict": verdict}));
        }
        let outputs: Vec<String> = annotations.iter().map(|a| format!("annotations/{a}")).collect();
        fields.insert("outputs".into(), json!(outputs));
        writer.activity("review", Some(&self.agent), None, fields)?;
        log.decision(
            "review",
            format!(
                "{source}: extra.review became a `review` activity and the quality record of {} --- what the file \
                 held; 0.x kept review state in an ad-hoc dict that could not say what produced the data being \
                 reviewed (§11.1)",
                if annotations.is_empty() { "no annotation".to_string() } else { crate::json::repr_list(annotations) }
            ),
            json!({"source": source, "status": self.status, "verdict": self.verdict, "annotations": annotations}),
        );
        Ok(())
    }
}

/// Write the minted label set for review before a cohort-wide migration.
///
/// Atomically: a write cut short --- a full disk, a crash --- left a
/// truncated label set where the one under review was.
pub fn write_sidecar(label_set: &LabelSet, path: &Path) -> Result<PathBuf> {
    let text = crate::json::dumps(&label_set.to_json(None), crate::json::Style::PYTHON.with_indent(Some(2)));
    crate::h5::file::write_atomic(path, format!("{text}\n").as_bytes())?;
    Ok(path.to_path_buf())
}

/// Read a label-set sidecar back.
pub fn load_sidecar(path: &Path) -> Result<LabelSet> {
    let text = std::fs::read_to_string(path)?;
    let doc = crate::json::loads(&text)?;
    LabelSet::from_json(Some(&doc))?
        .ok_or_else(|| Error::Value(format!("{} holds no label set", path.to_string_lossy())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appendix_b_review_times_are_0x_isoformat_or_a_date() {
        // 0.x wrote `datetime.now(timezone.utc).isoformat()`.
        let iso = json!("2026-01-01T10:00:00.123456+00:00");
        assert_eq!(timestamp(&iso).as_deref(), Some("2026-01-01T10:00:00.123456+00:00"));
        assert_eq!(timestamp(&json!("2026-03-01")).as_deref(), Some("2026-03-01T00:00:00Z"));
        // A naive time states no zone, and a near-date is not a date.
        assert_eq!(timestamp(&json!("2026-01-01T10:00:00")), None);
        assert_eq!(timestamp(&json!("2026-1x-01")), None);
        assert_eq!(timestamp(&json!("é")), None);
    }
}
