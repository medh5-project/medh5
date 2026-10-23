//! [`SampleWriter`]: where the pieces become a file.
//!
//! Writing is a builder that validates as it goes and commits atomically
//! (§14.4): create writes a sibling temporary file and renames it over the
//! target, so a reader never sees a half-written sample and a crash leaves the
//! previous file intact.  Amend is copy-on-write, because HDF5 does not
//! reclaim space on delete.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use ndarray::Array2;
use serde_json::{Map, Value};

use super::reader::{annotation_id, attr_name_map_of, frame_references, require_major, FRAME_ATTRS, PROFILES};
use crate::array::NdArray;
use crate::clinical::ClinicalRecords;
use crate::curation::identity::{Cohort, Deidentification, Identity, SplitClaim, ID_SOURCE};
use crate::curation::provenance::{Activity, Agent};
use crate::curation::quality::QualityRecord;
use crate::curation::timeline::{Timeline, Timepoint};
use crate::document::{SampleDocument, META_DATASET};
use crate::geometry::grid::{
    default_axis_kinds, default_axis_names, read_grids, write_grid, Grid, KNOWN_UNITS, TIME_UNITS,
};
use crate::geometry::multiscale::{check_pyramid, pyramid_factors, Pyramid};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::data;
use crate::h5::file::{open_read, AtomicFile, Snapshot};
use crate::h5::ops;
use crate::ids::validate_id;
use crate::integrity::{collect_digests, compute_content_id, failing_digests, stamp_digests};
use crate::json::{repr_int_tuple, repr_list, repr_str};
use crate::labels::{ClassKey, LabelSet};
use crate::storage::chunking::grid_chunks;
use crate::storage::codecs::{dataset_layout, profile_family, resolve_profile, Role};
use crate::{Error, Result, VERSION};

/// Root attributes `commit` writes itself, so an amend must not copy them.
pub const MANAGED_ROOT_ATTRS: [&str; 7] =
    ["medh5_version", "medh5_kind", "medh5_profiles", "created", "generator", "digest_algo", "content_id"];

/// The groups the writer manages; anything else is copied through on amend.
pub const STANDARD_GROUPS: [&str; 6] = ["meta", "grids", "images", "annotations", "transforms", "index"];

/// `YYYY-mm-ddTHH:MM:SSZ`, now, in UTC.
pub fn utcnow() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}

/// Days since 1970-01-01 to a proleptic Gregorian date.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// The `medh5_version` a commit writes: the lowest version the declared
/// profiles need (1.0, or 1.1 with `clinical`), never below `source`, what
/// an amended file already declared --- a no-op amend does not downgrade
/// (1.1 §2.3).  An amendment that removes the clinical profile is the
/// exception (1.1 §2.1): `drop_clinical`, or an amendment leaving the group
/// behind, passes no `source`, so the imaging projection is written at the
/// lowest version what remains needs.
pub fn written_version(source: Option<&str>, profiles: &[String]) -> String {
    let none: [&str; 0] = [];
    written_version_for_content(source, profiles, &none)
}

/// [`written_version`], for content declaring `profiles` and holding
/// annotations of `kinds`: 1.2 once it holds a `polylines` annotation (1.2
/// §2), and never below `source`.  An amendment that removes the last
/// `polylines` annotation passes no `source` (1.2 §2.1), as one removing the
/// clinical profile does: what remains is written at the lowest version it
/// needs.
pub fn written_version_for_content<S: AsRef<str>, K: AsRef<str>>(
    source: Option<&str>,
    profiles: &[S],
    kinds: &[K],
) -> String {
    let needed = crate::version::required_for_content(profiles, kinds);
    match source {
        Some(v) if crate::version::support(v) == crate::version::Support::Full => {
            crate::version::later(v, needed).to_string()
        }
        _ => needed.to_string(),
    }
}

/// Where a writer's `clinical/` group came from (1.1 §3, §10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClinicalSource {
    /// None: the sample has no `clinical` group.
    #[default]
    Absent,
    /// The profile's own content, declared and recognised, copied through
    /// untouched until something changes it.
    Inherited,
    /// A `clinical` group that is not the profile's --- a 1.0 file's own
    /// extension.  Copied through untouched, and never reinterpreted.
    Foreign,
}

/// Which classes an annotation claims were looked for (§11.3).
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Annotated {
    /// Only the classes supplied.
    #[default]
    AllGiven,
    /// The whole label set.
    All,
    /// Exactly these.
    Classes(Vec<ClassKey>),
}

/// A quality reference: an existing record's key, or a record to create.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityArg {
    Key(String),
    Record(Map<String, Value>),
}

/// Options for [`SampleWriter::add_image`].
#[derive(Debug, Clone, Default)]
pub struct ImageOptions {
    pub value_type: Option<String>,
    pub value_units: Option<String>,
    pub channel_names: Option<Vec<String>>,
    pub rescale_slope: Option<f64>,
    pub rescale_intercept: Option<f64>,
    pub window_center: Option<Vec<f64>>,
    pub window_width: Option<Vec<f64>>,
    pub valid_mask: Option<String>,
    pub prov: Option<String>,
    pub codec: Option<String>,
}

/// The attributes [`SampleWriter::set_image_attrs`] changes.  For each
/// field, `None` leaves the attribute as it is, `Some(None)` removes it and
/// `Some(Some(value))` sets it.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct ImageEdit {
    pub modality: Option<Option<String>>,
    pub value_type: Option<Option<String>>,
    pub value_units: Option<Option<String>>,
    pub channel_names: Option<Option<Vec<String>>>,
    pub rescale_slope: Option<Option<f64>>,
    pub rescale_intercept: Option<Option<f64>>,
    pub window_center: Option<Option<Vec<f64>>>,
    pub window_width: Option<Option<Vec<f64>>>,
    pub valid_mask: Option<Option<String>>,
    pub prov: Option<Option<String>>,
}

/// Options for [`SampleWriter::add_grid`].
#[derive(Debug, Clone, Default)]
pub struct GridOptions {
    pub origin: Option<Vec<f64>>,
    pub direction: Option<Array2<f64>>,
    pub axis_names: Option<Vec<String>>,
    pub axis_kinds: Option<Vec<String>>,
    pub coord_system: Option<String>,
    pub units: Option<String>,
    pub timepoint: Option<String>,
    pub frame_uid: Option<String>,
    pub patch_hint: Option<Vec<i64>>,
    pub chunk_hint: Option<Vec<i64>>,
    pub time_values: Option<Vec<f64>>,
    pub time_units: Option<String>,
}

/// Where a writer is in its life.  A commit refused before it changes
/// anything leaves the writer [`State::Open`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Being built: `add_*` and `commit` work.
    Open,
    /// The file is written; a second `commit` has nothing to do.
    Committed,
    /// Discarded --- by `abort`, or by a commit that failed once it had begun
    /// changing the file --- and the target left as it was.
    Aborted,
}

/// Builder for one sample.  Every `add_*` validates immediately.
pub struct SampleWriter {
    pub path: PathBuf,
    /// The codec profile name datasets default to.
    pub codec: String,
    file: Option<AtomicFile>,
    state: State,
    /// The algorithm the commit digests with: an amended file's own, so an
    /// amendment never mixes two (§13.1).
    digest_algo: String,
    /// The `created` the commit writes in place of now: the conformance
    /// corpus pins it, so a suite built twice is the same bytes, and a strict
    /// scrub cuts it to its year.
    pub(crate) created: Option<String>,
    /// Every path but `meta` at which the amended file held its `/meta` by a
    /// hard link.  Until the commit re-links each to the document it writes,
    /// the file being built holds a copy of the old document there.
    pub(crate) meta_aliases: Vec<String>,
    pub(crate) grids: IndexMap<String, Grid>,
    pub(crate) images: IndexMap<String, (String, String)>,
    pub(crate) image_multiscale: HashMap<String, bool>,
    pub(crate) annotation_kinds: IndexMap<String, String>,
    pub(crate) transform_frames: IndexMap<String, (String, String)>,
    declared_profiles: BTreeSet<String>,
    default_timeline: bool,
    source_version: Option<String>,
    /// The kinds of the annotations the amended file held: an amendment that
    /// removes the last of a kind a later minor added lowers the version
    /// (1.2 §2.1).
    source_kinds: BTreeSet<String>,
    pub(crate) document: SampleDocument,
    /// The clinical records, once written to or changed in this writer.
    pub(crate) clinical: Option<ClinicalRecords>,
    /// Something was removed from the file being built, so its bytes are
    /// still in it: commit rewrites the file before it replaces anything.
    pub(crate) compact: bool,
    pub(crate) clinical_source: ClinicalSource,
    /// Event and document ids already in `clinical`, so a duplicate is
    /// refused without scanning every row.
    pub(crate) clinical_ids: (std::collections::HashSet<String>, std::collections::HashSet<String>),
    /// The kind of every event version the amended file already held: those
    /// versions are immutable, so no payload may be attached to one.
    pub(crate) inherited_events: std::collections::HashMap<String, String>,
    /// The file an amendment started from, which a digest the commit finds
    /// changed is checked against.
    pub(crate) source: Option<PathBuf>,
    /// What the amendment saw of its source when it read it: a commit
    /// refuses a source that changed since (§14.4).
    seen: Option<Snapshot>,
}

impl std::fmt::Debug for SampleWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SampleWriter").field("path", &self.path).field("codec", &self.codec).finish()
    }
}

/// Create a new sample.  Call [`SampleWriter::commit`] to write it.
pub fn create(
    path: &Path,
    sample_id: Option<&str>,
    subject_id: Option<&str>,
    codec: &str,
    profiles: &[String],
) -> Result<SampleWriter> {
    SampleWriter::new(path, sample_id, subject_id, codec, profiles, None)
}

/// What an amend copies from: the source's root, and the top-level members
/// it leaves behind.
pub type Source<'a> = (&'a hdf5::Group, &'a [&'a str]);

/// Copy-on-write amend: build a new file from the old and replace it (§14.4).
///
/// Unknown objects are copied through untouched; known profiles are re-derived
/// from what the amended file holds, and a recognised `clinical` profile is
/// carried with its group.  A file this engine cannot preserve --- another
/// major, a higher minor, a profile it does not implement --- is refused before
/// anything is written (1.1 §2.3).  `codec` defaults to the family the file was
/// written in, and the commit digests in the algorithm the file declares.
///
/// The commit renames the new file over the old.  On POSIX a reader holding
/// the old file open keeps reading it; Windows cannot replace a file that is
/// open, so there every reader of it must be closed before the commit, whose
/// rename otherwise fails and leaves the file as it was.
pub fn amend(path: &Path, codec: Option<&str>) -> Result<SampleWriter> {
    amend_into(path, path, codec, &[])
}

/// [`amend`], writing the result to `out` and leaving the source's top-level
/// members named in `without` behind.  What is left behind is never copied,
/// so no byte of it reaches `out` --- where copying the file and deleting
/// the group left every deleted byte in it.  The source is only read.
///
/// A source whose attestation no longer holds is refused before anything is
/// written ([`refuse_unattestable`]): a `content_id` that does not recompute
/// (E702), a digest or a `digest_algo` this engine cannot read (E703).  A
/// source that changes after it was read is refused at the commit (§14.4).
pub fn amend_into(path: &Path, out: &Path, codec: Option<&str>, without: &[&str]) -> Result<SampleWriter> {
    amend_into_with(path, out, codec, without, false)
}

/// [`amend`] for a deliberate re-attestation --- `medh5 fix
/// --rewrite-digests --reason`, which records it: the one amendment of a
/// source whose `content_id` no longer recomputes, or whose digests this
/// engine cannot read, which it restamps in the default algorithm.
pub(crate) fn amend_reattesting(path: &Path, codec: Option<&str>) -> Result<SampleWriter> {
    amend_into_with(path, path, codec, &[], true)
}

fn amend_into_with(
    path: &Path,
    out: &Path,
    codec: Option<&str>,
    without: &[&str],
    reattest: bool,
) -> Result<SampleWriter> {
    let source = open_read(path)?;
    let seen = Snapshot::of(&source, path, "amendment")?;
    let version = require_major(&source, path)?;
    let declared = attrs::get_strs(&source, "medh5_profiles")?.unwrap_or_default();
    crate::version::require_amendable(&version, &declared, &repr_str(&path.to_string_lossy()))?;
    let kind = attrs::get_str(&source, "medh5_kind")?.unwrap_or_else(|| "sample".into());
    if kind != "sample" {
        return Err(Error::File(format!(
            "{} is a {}; amend works on one sample --- extract the member with `medh5 unpack`, amend it, and pack again",
            repr_str(&path.to_string_lossy()),
            repr_str(&kind)
        )));
    }
    let root = source.as_group()?;
    ops::refuse_references(&root, "an amendment")?;
    if !reattest {
        refuse_unattestable(&root, &repr_str(&path.to_string_lossy()))?;
    }
    let chosen = match codec {
        Some(c) => c.to_string(),
        None => profile_family(&root)?.to_string(),
    };
    let mut writer = SampleWriter::new(out, None, None, &chosen, &[], Some((&root, without)))?;
    writer.source = Some(path.to_path_buf());
    if let Some(file) = writer.file.as_mut() {
        file.expect_unchanged(path.to_path_buf(), seen.clone());
    }
    writer.seen = Some(seen);
    Ok(writer)
}

/// The repair that re-attests a file deliberately, as a refusal names it.
const REATTEST: &str = "re-attest it deliberately with `medh5 fix --rewrite-digests --reason ...`, which records the \
                        reason in the file's provenance";

/// Refuse, before anything is written, to amend a sample whose attestation a
/// commit would carry past a break (§14.4): one declaring a `digest_algo`
/// outside §2.1, or holding a digest that does not parse (E703), or whose
/// declared `content_id` does not recompute from what it stores (E702).  A
/// commit recomputes the `content_id` from whatever is stored, so a no-op
/// amendment of a file edited outside medh5 made it validate, with no record
/// of the edit; and an algorithm it could not compute gave way to `sha256`.
/// Only attributes and `/meta` are read, never bulk data.  `shown` names the
/// file in the refusal.
pub(crate) fn refuse_unattestable(root: &hdf5::Group, shown: &str) -> Result<()> {
    if let Some(algo) = unknown_algo(root)? {
        return Err(Error::coded(
            "E703",
            format!(
                "refusing to amend {shown}: it declares digest_algo {}, which §2.1 does not permit (sha256, sha512 \
                 and blake2b), so a commit could neither keep that algorithm nor check the digests made with it; \
                 {REATTEST}, which restamps it in sha256",
                repr_str(&algo)
            ),
        ));
    }
    let stored = collect_digests(root, &["index"])?;
    let unreadable: Vec<String> = stored
        .iter()
        .filter(|(_, value)| !crate::digest::is_canonical_digest(value))
        .map(|(path, _)| path.clone())
        .collect();
    if !unreadable.is_empty() {
        return Err(unreadable_refused(shown, &unreadable));
    }
    // Without `/meta` there is no root to recompute: the document's own
    // read refuses the file, as it did.
    if ops::child_dataset(root, META_DATASET).is_none() {
        return Ok(());
    }
    let declared = match attrs::read(root, "content_id")? {
        None => return Ok(()),
        Some(AttrValue::Str(s)) => s,
        Some(other) => attrs::stringify_value(&other),
    };
    if !crate::digest::is_canonical_digest(&declared) {
        return Err(Error::coded(
            "E703",
            format!(
                "refusing to amend {shown}: its `content_id` {} is not a digest as §13.1 spells it, so nothing \
                 attests what it stores; {REATTEST}",
                repr_str(&declared)
            ),
        ));
    }
    let algo = crate::integrity::digest::root_algo(root)?;
    let computed = compute_content_id(root, &attr_name_map_of(root)?, &algo, Some(&stored))?;
    if computed == declared {
        return Ok(());
    }
    Err(Error::coded(
        "E702",
        format!(
            "refusing to amend {shown}: its `content_id` {declared} does not recompute from what it stores \
             ({computed}) --- it was changed outside medh5 --- and a commit would re-attest the change with no \
             record of it; restore the file, or {REATTEST}"
        ),
    ))
}

/// E703 for amending the file `shown` names, whose datasets `paths` carry a
/// digest that does not parse.
fn unreadable_refused(shown: &str, paths: &[String]) -> Error {
    let one = paths.len() == 1;
    let listed = paths.iter().take(5).map(|p| format!("/{p}")).collect::<Vec<_>>().join(", ");
    let more = if paths.len() > 5 { format!(" (and {} more)", paths.len() - 5) } else { String::new() };
    Error::coded(
        "E703",
        format!(
            "refusing to amend {shown}: the digest{} of {listed}{more} {} not parse, so a commit would restamp bytes \
             nobody verified; restore {}, or {REATTEST}",
            if one { "" } else { "s" },
            if one { "does" } else { "do" },
            if one { "it" } else { "them" },
        ),
    )
}

/// The `digest_algo` a root declares when it is outside §2.1.
pub(crate) fn unknown_algo(root: &hdf5::Group) -> Result<Option<String>> {
    let algo = match attrs::read(root, "digest_algo")? {
        None => return Ok(None),
        Some(AttrValue::Str(s)) => s,
        Some(other) => attrs::stringify_value(&other),
    };
    Ok((!crate::digest::DIGEST_ALGOS.contains(&algo.as_str())).then_some(algo))
}

/// Refuse, by reading alone, a sample whose amendment would be refused: one
/// [`refuse_unattestable`] refuses (E702, E703), or a dataset whose bytes
/// fail the digest it carries, which the commit would restamp (E701).  A tool that amends many files checks each before it writes
/// any, so a refusal is met before the first file changes rather than at the
/// file it falls on.  `shown` names the file in the refusal.
pub fn refuse_failed_digests(root: &hdf5::Group, shown: &str) -> Result<()> {
    refuse_unattestable(root, shown)?;
    // What a commit restamps: every dataset that has a digest to compute,
    // but `meta` and the derived `index/` (§13.3), as `stamp_digests` walks.
    let mut restamped = Vec::new();
    for (name, ds) in ops::datasets(root)? {
        let derived = name == META_DATASET || name.split('/').next() == Some("index");
        if !derived && attrs::has(&ds, "digest") && crate::integrity::digest::digestible(&ds)? {
            restamped.push(name);
        }
    }
    let failed = failing_digests(root, &restamped)?;
    if failed.is_empty() {
        return Ok(());
    }
    Err(restamp_refused(shown, &failed))
}

/// E701 for amending the file `shown` names, whose datasets `failed` do not
/// match their digests.
fn restamp_refused(shown: &str, failed: &[String]) -> Error {
    let one = failed.len() == 1;
    let listed = failed.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
    let more = if failed.len() > 5 { format!(" (and {} more)", failed.len() - 5) } else { String::new() };
    Error::coded(
        "E701",
        format!(
            "refusing to amend {shown}: {} {} not match {} digest{} ({listed}{more}), and a commit would restamp {} \
             --- re-attesting bytes nobody verified, with no record that they changed. Restore the content, or \
             restamp it deliberately with `medh5 fix --rewrite-digests --reason ...`, which records the reason in \
             the file's provenance",
            if one { "a dataset" } else { "datasets" },
            if one { "does" } else { "do" },
            if one { "its" } else { "their" },
            if one { "" } else { "s" },
            if one { "it" } else { "them" },
        ),
    )
}

/// Stamp the digests under `root` in `algo`, as [`stamp_digests`] does ---
/// with `only_missing`, also restamping a digest kept in another algorithm,
/// so the file never carries two.  Returns every digest, by path.
fn stamp_in(root: &hdf5::Group, algo: &str, only_missing: bool) -> Result<IndexMap<String, String>> {
    let mut stamped = stamp_digests(root, algo, &["index"], only_missing)?;
    if only_missing {
        for (path, value) in stamped.iter_mut() {
            // A malformed digest is left as it was, to fail as it did.
            if crate::digest::parse_digest(value).is_ok_and(|(found, _)| found != algo) {
                let ds = root.dataset(path)?;
                if crate::integrity::digest::digestible(&ds)? {
                    let fresh = crate::integrity::digest::dataset_digest(&ds, path, algo)?;
                    attrs::write(&ds, "digest", &AttrValue::Str(fresh.clone()))?;
                    *value = fresh;
                }
            }
        }
    }
    Ok(stamped)
}

/// Every path but `meta` itself at which `root` holds its `/meta` by a hard
/// link, outside the top-level members `without` names.  Each group is
/// listed once however many paths reach it, so a link to `/meta` in it is
/// found under the group's first path, whatever the link is called.
fn meta_aliases(root: &hdf5::Group, without: &[&str]) -> Result<Vec<String>> {
    let Some(meta) = ops::object_id_by_name(root, META_DATASET) else { return Ok(Vec::new()) };
    let mut found = Vec::new();
    let mut scan = |prefix: &str, group: &hdf5::Group| -> Result<()> {
        for name in ops::members(group)? {
            let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
            if path != META_DATASET
                && ops::link_kind(group, &name) == Some(ops::LinkKind::Hard)
                && ops::object_id_by_name(group, &name) == Some(meta)
            {
                found.push(path);
            }
        }
        Ok(())
    };
    scan("", root)?;
    ops::visit(root, &mut |path, node| {
        if let ops::Node::Group(group) = node {
            scan(path, group)?;
        }
        Ok(true)
    })?;
    found.retain(|path| !without.contains(&path.split('/').next().unwrap_or_default()));
    Ok(found)
}

/// The id a file gets when none is given: what 2.x takes, and what 3.0 will.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultIds {
    /// The file name up to its first dot: the 2.x `sample_id` and pack key.
    pub current: String,
    /// The file stem, the name less its last suffix: 3.0's.
    pub stem: String,
}

/// The default `sample_id` and pack key of `path` (the file name up to its
/// first dot), and the stem 3.0 takes instead.  A name with nothing before
/// its first dot gives no id, and is refused naming the file.
#[doc(hidden)]
pub fn default_ids(path: &Path) -> Result<DefaultIds> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let current = name.split('.').next().unwrap_or("").to_string();
    if current.is_empty() {
        return Err(Error::invalid(format!(
            "{} gives no sample id: its name has nothing before its first dot; pass sample_id (and subject_id)",
            repr_str(&path.to_string_lossy())
        )));
    }
    let stem = path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(DefaultIds { current, stem })
}

impl SampleWriter {
    /// A writer for `path`; `source` copies an existing sample in (amend),
    /// but for the top-level members it names.
    pub fn new(
        path: &Path,
        sample_id: Option<&str>,
        subject_id: Option<&str>,
        codec: &str,
        profiles: &[String],
        source: Option<Source<'_>>,
    ) -> Result<SampleWriter> {
        let codec = resolve_profile(Some(codec))?.name.to_string();
        let file = AtomicFile::create(path)?;
        // An amendment takes its identity from the source; a placeholder
        // stands until it does.
        let default_id = match (sample_id, &source) {
            (Some(id), _) => id.to_string(),
            (None, Some(_)) => "amended".to_string(),
            (None, None) => default_ids(path)?.current,
        };
        let identity =
            Identity::new(default_id.clone(), subject_id.map(str::to_string).unwrap_or_else(|| default_id.clone()));
        let mut writer = SampleWriter {
            path: path.to_path_buf(),
            codec,
            file: Some(file),
            state: State::Open,
            digest_algo: crate::digest::DEFAULT_ALGO.to_string(),
            created: None,
            meta_aliases: Vec::new(),
            grids: IndexMap::new(),
            images: IndexMap::new(),
            image_multiscale: HashMap::new(),
            annotation_kinds: IndexMap::new(),
            transform_frames: IndexMap::new(),
            declared_profiles: profiles.iter().cloned().collect(),
            default_timeline: true,
            source_version: None,
            source_kinds: BTreeSet::new(),
            document: SampleDocument::new(identity?, Timeline::single("tp0")?),
            clinical: None,
            compact: false,
            clinical_source: ClinicalSource::Absent,
            clinical_ids: Default::default(),
            inherited_events: Default::default(),
            source: None,
            seen: None,
        };
        let setup = (|| -> Result<()> {
            for name in ["grids", "images", "annotations"] {
                ops::new_group(&writer.root()?, name)?;
            }
            if let Some((src, without)) = source {
                writer.inherit(src, without)?;
            }
            Ok(())
        })();
        if let Err(e) = setup {
            writer.abort();
            return Err(e);
        }
        Ok(writer)
    }

    /// The file being built.
    pub fn handle(&self) -> Result<&hdf5::File> {
        match &self.file {
            Some(f) => Ok(f.handle()),
            None => Err(Error::invalid("this writer has already committed or aborted")),
        }
    }

    /// The root group of the file being built.
    pub fn root(&self) -> Result<hdf5::Group> {
        Ok(self.handle()?.as_group()?)
    }

    fn group(&self, name: &str) -> Result<hdf5::Group> {
        Ok(self.root()?.group(name)?)
    }

    /// `written`, or --- when it failed --- nothing: the object `name` the
    /// failed write began under `parent` is unlinked, so a corrected retry can
    /// write it and a commit never finds it half-written, and the commit
    /// rewrites the file, so none of its bytes stay behind either.  The
    /// write's own error is the one reported.
    pub(crate) fn or_unlink<T>(&mut self, parent: &hdf5::Group, name: &str, written: Result<T>) -> Result<T> {
        if written.is_err() && ops::exists(parent, name) {
            ops::unlink(parent, name).ok();
            self.compact = true;
        }
        written
    }

    /// Discard the in-progress file, leaving any existing one untouched.  A
    /// writer that has committed stays committed.
    pub fn abort(&mut self) {
        if let Some(f) = self.file.take() {
            f.abort();
        }
        if self.state == State::Open {
            self.state = State::Aborted;
        }
    }

    /// Whether the writer has committed or been aborted.  A commit refused
    /// before it changed anything leaves it open.
    pub fn is_closed(&self) -> bool {
        self.state != State::Open
    }

    /// The algorithm the commit digests with: the amended file's, or the
    /// default for a new one.  A tool restamping before the commit stamps in
    /// it, or the commit finds every digest changed.
    pub(crate) fn digest_algo(&self) -> &str {
        &self.digest_algo
    }

    /// Digest a new file in `algo`, an algorithm of §2.1: what the corpus
    /// builds a sha512 file with.  The reference writer offers no such
    /// option.
    pub(crate) fn set_digest_algo(&mut self, algo: &str) {
        self.digest_algo = algo.to_string();
    }

    pub fn document(&self) -> &SampleDocument {
        &self.document
    }

    pub fn document_mut(&mut self) -> &mut SampleDocument {
        &mut self.document
    }

    /// Replace the whole sample document; `commit` validates it as usual.
    pub fn set_document(&mut self, document: SampleDocument) {
        self.document = document;
    }

    /// Remove the `clinical` profile --- its group, its records and its
    /// declaration --- for the imaging projection (1.1 §10).  The sample is
    /// then written in the lowest version its remaining profiles need, and is
    /// a different sample with a different `content_id`.  A sample with no
    /// clinical profile to remove keeps the version it declared: only an
    /// amendment that removes the profile lowers it (1.1 §2.1).
    ///
    /// HDF5 does not reclaim an unlinked group's space, so the commit
    /// rewrites the file from its root before it replaces anything: unlinked
    /// alone, every record stayed in the file's bytes.
    ///
    /// The event versions the amended file held stay held: one added again
    /// in this amendment must be what it was, attesting what it did (1.1
    /// §7.3).  Dropped and added again with a payload it never had, a
    /// version became an input with that payload from its old availability.
    pub fn drop_clinical(&mut self) -> Result<()> {
        let root = self.root()?;
        let present = ops::exists(&root, crate::clinical::GROUP);
        if present {
            self.compact = true;
        }
        if present || self.clinical.is_some() || self.declared_profiles.contains(crate::clinical::PROFILE) {
            self.source_version = None;
        }
        if self.clinical_source == ClinicalSource::Inherited && self.inherited_events.is_empty() {
            // Tables that do not read hold no version anything can be held to.
            if let Ok(Some(held)) = crate::clinical::Clinical::open(&root, false) {
                self.inherited_events = held.events.iter().map(|e| (e.event_id.clone(), e.kind.clone())).collect();
            }
        }
        ops::unlink(&root, crate::clinical::GROUP)?;
        self.clinical = None;
        self.clinical_ids = Default::default();
        self.clinical_source = ClinicalSource::Absent;
        self.declared_profiles.remove(crate::clinical::PROFILE);
        Ok(())
    }

    fn inherit(&mut self, source: &hdf5::Group, without: &[&str]) -> Result<()> {
        stage!("amend.inherit");
        let leaves_clinical = without.contains(&crate::clinical::GROUP);
        // Without its clinical group the sample is written in the lowest
        // version what is left needs, as `drop_clinical` writes it.
        self.source_version = if leaves_clinical { None } else { attrs::get_str(source, "medh5_version")? };
        self.document = super::reader::read_document(source)?;
        self.default_timeline = false;
        // Known profiles are re-derived from the amended content, so a claim
        // an edit no longer justifies is dropped rather than kept false; an
        // unknown one was refused before the amend began (1.1 §2.3), and a
        // recognised `clinical` group carries its profile with it.
        let declared = attrs::get_strs(source, "medh5_profiles")?.unwrap_or_default();
        if leaves_clinical && declared.iter().any(|p| p == crate::clinical::PROFILE) {
            // Left behind, the versions the source held stay held: one added
            // again in this amendment must be what it was (1.1 §7.3), as
            // after `drop_clinical`.  Tables that do not read hold no
            // version anything can be held to.
            if let Ok(Some(held)) = crate::clinical::Clinical::open(source, false) {
                self.inherited_events = held.events.iter().map(|e| (e.event_id.clone(), e.kind.clone())).collect();
            }
        }
        if ops::exists(source, crate::clinical::GROUP) && !leaves_clinical {
            self.clinical_source =
                if declared.iter().any(|p| p == crate::clinical::PROFILE) && crate::clinical::recognised(source) {
                    ClinicalSource::Inherited
                } else {
                    ClinicalSource::Foreign
                };
        }
        // The file's own algorithm: restamped in another, every digest and
        // the `content_id` changed, and the digests a commit left alone were
        // kept under a root declaring the other.  One this engine cannot
        // compute (E703) gives way to the default.
        if let Some(algo) = attrs::get_str(source, "digest_algo").ok().flatten() {
            if crate::digest::DIGEST_ALGOS.contains(&algo.as_str()) {
                self.digest_algo = algo;
            }
        }
        let root = self.root()?;
        // The source's graph comes across whole, as one copy: a hard link
        // between two of its groups stays one object, where copying group by
        // group made it two --- a new dataset line, so a no-op amend changed
        // the `content_id`.  `meta` is written at
        // commit; the groups a new sample starts with give way to the
        // source's.
        for name in ["grids", "images", "annotations"] {
            ops::unlink(&root, name)?;
        }
        let mut skip: Vec<&str> = vec![META_DATASET];
        skip.extend(without);
        crate::h5::graph::GraphCopy::new(source, &root).members(source, &root, &skip, &mut crate::h5::graph::Raw)?;
        // Another hard link to `/meta` is the document under another name,
        // and the copy brought the old document across under it: written
        // anew at commit, `/meta` became another object, and the alias kept
        // every identifier a scrub had replaced.  The commit re-links each
        // alias to the new document, and rewrites the file so the old one's
        // bytes go with it.  A soft link names `/meta` by path, and follows.
        self.meta_aliases = meta_aliases(source, without)?;
        if !self.meta_aliases.is_empty() {
            self.compact = true;
        }
        for name in ["grids", "images", "annotations"] {
            if !ops::exists(&root, name) {
                ops::new_group(&root, name)?;
            }
        }
        self.grids = read_grids(&root)?.into_iter().collect();
        if let Some(images) = ops::child_group(&root, "images") {
            for name in ops::members(&images)? {
                let multiscale = ops::is_group(&images, &name);
                let (grid, modality) = if multiscale {
                    let g = images.group(&name)?;
                    (attrs::get_str(&g, "grid")?, attrs::get_str(&g, "modality")?)
                } else {
                    let d = images.dataset(&name)?;
                    (attrs::get_str(&d, "grid")?, attrs::get_str(&d, "modality")?)
                };
                self.images.insert(name.clone(), (grid.unwrap_or_default(), modality.unwrap_or_else(|| "OT".into())));
                self.image_multiscale.insert(name, multiscale);
            }
        }
        if let Some(anns) = ops::child_group(&root, "annotations") {
            for name in ops::members(&anns)? {
                if let Some(g) = ops::child_group(&anns, &name) {
                    self.annotation_kinds.insert(name, attrs::get_str(&g, "kind")?.unwrap_or_else(|| "mask".into()));
                }
            }
        }
        self.source_kinds = self.annotation_kinds.values().cloned().collect();
        if let Some(ts) = ops::child_group(&root, "transforms") {
            for name in ops::members(&ts)? {
                if let Some(g) = ops::child_group(&ts, &name) {
                    let frames = (
                        attrs::get_str(&g, "from_frame")?.unwrap_or_default(),
                        attrs::get_str(&g, "to_frame")?.unwrap_or_default(),
                    );
                    self.transform_frames.insert(name, frames);
                }
            }
        }
        for name in attrs::names(source)? {
            if MANAGED_ROOT_ATTRS.contains(&name.as_str()) {
                continue;
            }
            attrs::copy_raw(source, &root, &name)?;
        }
        Ok(())
    }

    // -- document ---------------------------------------------------------------

    /// Merge fields into the identity.  Changing an id drops what `id_source`
    /// recorded about it, unless the fields record a new source.
    pub fn identity(&mut self, fields: Map<String, Value>) -> Result<Identity> {
        let current = match self.document.identity.to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        let mut merged = current.clone();
        merged.extend(fields.clone());
        if !fields.contains_key(ID_SOURCE) {
            if let Some(Value::Object(source)) = merged.get(ID_SOURCE).cloned() {
                let kept: Map<String, Value> = source
                    .into_iter()
                    .filter(|(name, _)| match fields.get(name) {
                        None => true,
                        Some(v) => Some(v) == current.get(name),
                    })
                    .collect();
                if kept.is_empty() {
                    merged.remove(ID_SOURCE);
                } else {
                    merged.insert(ID_SOURCE.into(), Value::Object(kept));
                }
            }
        }
        self.document.identity = Identity::from_json(&Value::Object(merged))?;
        Ok(self.document.identity.clone())
    }

    /// Merge fields into the cohort record.
    pub fn cohort(&mut self, fields: Map<String, Value>) -> Result<Cohort> {
        let mut merged = match self.document.cohort.to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        merged.extend(fields);
        self.document.cohort = Cohort::from_json(Some(&Value::Object(merged)))?;
        Ok(self.document.cohort.clone())
    }

    /// Declare a timepoint; the first explicit one replaces the implicit `tp0`.
    pub fn add_timepoint(&mut self, timepoint_id: &str, mut fields: Map<String, Value>) -> Result<Timepoint> {
        let existing: Vec<Timepoint> =
            if self.default_timeline { Vec::new() } else { self.document.timepoints.points().to_vec() };
        let index = fields.remove("index").unwrap_or_else(|| Value::from(existing.len()));
        let mut doc = Map::new();
        doc.insert("id".into(), Value::String(timepoint_id.into()));
        doc.insert("index".into(), index);
        doc.extend(fields);
        let tp = Timepoint::from_json(&Value::Object(doc))?;
        let mut all = existing;
        all.push(tp.clone());
        self.document.timepoints = Timeline::new(all)?;
        self.default_timeline = false;
        Ok(tp)
    }

    pub fn label_set(&mut self, label_set: LabelSet) -> LabelSet {
        self.document.label_set = Some(label_set.clone());
        label_set
    }

    /// Declare an agent; an explicit id must be free, an automatic one is
    /// `<type initial><n>`, skipping ids already taken.
    pub fn agent(
        &mut self,
        agent_type: &str,
        name: &str,
        agent_id: Option<&str>,
        fields: Map<String, Value>,
    ) -> Result<Agent> {
        let prov = &self.document.provenance;
        let id = match agent_id {
            Some(id) => id.to_string(),
            None => {
                let initial = agent_type.chars().next().unwrap_or('a');
                let mut n = prov.n_agents() + 1;
                while prov.has_agent(&format!("{initial}{n}")) {
                    n += 1;
                }
                format!("{initial}{n}")
            }
        };
        let mut doc = Map::new();
        doc.insert("id".into(), Value::String(id));
        doc.insert("type".into(), Value::String(agent_type.into()));
        doc.insert("name".into(), Value::String(name.into()));
        for (k, v) in fields {
            if !v.is_null() {
                doc.insert(k, v);
            }
        }
        let agent = Agent::from_json(&Value::Object(doc))?;
        self.document.provenance.add_agent(agent, false)
    }

    pub fn person(&mut self, name: &str, agent_id: Option<&str>, fields: Map<String, Value>) -> Result<Agent> {
        self.agent("person", name, agent_id, fields)
    }

    pub fn software(&mut self, name: &str, version: Option<&str>, mut fields: Map<String, Value>) -> Result<Agent> {
        if let Some(v) = version {
            fields.insert("version".into(), Value::String(v.into()));
        }
        self.agent("software", name, None, fields)
    }

    pub fn organization(&mut self, name: &str, fields: Map<String, Value>) -> Result<Agent> {
        self.agent("organization", name, None, fields)
    }

    /// Record an activity; ids are `act_<type>_<n>`, skipping taken ones.
    pub fn activity(
        &mut self,
        activity_type: &str,
        agent: Option<&str>,
        activity_id: Option<&str>,
        fields: Map<String, Value>,
    ) -> Result<Activity> {
        let prov = &self.document.provenance;
        let id = match activity_id {
            Some(id) => id.to_string(),
            None => {
                let mut n = prov.n_activities() + 1;
                while prov.has_activity(&format!("act_{activity_type}_{n}")) {
                    n += 1;
                }
                format!("act_{activity_type}_{n}")
            }
        };
        let mut doc = Map::new();
        doc.insert("id".into(), Value::String(id));
        doc.insert("type".into(), Value::String(activity_type.into()));
        if let Some(a) = agent {
            doc.insert("agent".into(), Value::String(a.into()));
        }
        for (k, v) in fields {
            if !v.is_null() {
                doc.insert(k, v);
            }
        }
        let activity = Activity::from_json(&Value::Object(doc))?;
        self.document.provenance.add_activity(activity, false)
    }

    /// Create or replace a quality record (status defaults to `draft`).
    pub fn set_quality(&mut self, key: &str, fields: Map<String, Value>) -> Result<QualityRecord> {
        let mut doc = Map::new();
        doc.insert("status".into(), Value::String("draft".into()));
        doc.extend(fields);
        let record = QualityRecord::from_json(&Value::Object(doc))?;
        self.document.quality.insert(key.to_string(), record.clone());
        Ok(record)
    }

    /// Record a split claim, replacing any earlier claim for the same set.
    pub fn split(&mut self, fields: Map<String, Value>) -> Result<SplitClaim> {
        let claim = SplitClaim::from_json(&Value::Object(fields))?;
        self.document.splits.retain(|s| s.set_id != claim.set_id);
        self.document.splits.push(claim.clone());
        Ok(claim)
    }

    /// Set the de-identification record (it needs at least `method`).
    pub fn deidentification(&mut self, fields: Map<String, Value>) -> Result<Deidentification> {
        let record = Deidentification::from_json(Some(&Value::Object(fields)))?
            .ok_or_else(|| Error::invalid("a de-identification record needs at least its `method` (§11.4)"))?;
        self.document.deidentification = Some(record.clone());
        Ok(record)
    }

    /// Merge acquisition parameters for one image.
    pub fn acquisition(&mut self, image_id: &str, params: Map<String, Value>) -> Value {
        let entry = self.document.acquisition.entry(image_id.to_string()).or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(m) = entry {
            m.extend(params);
        }
        entry.clone()
    }

    /// Set a namespaced extension member of the document.
    pub fn extra(&mut self, namespace: &str, value: Value) {
        self.document.extra.insert(namespace.to_string(), value);
    }

    // -- grids ------------------------------------------------------------------

    /// Declare a grid.  Geometry lives here and nowhere else.
    pub fn add_grid(&mut self, grid_id: &str, shape: &[i64], spacing: &[f64], options: GridOptions) -> Result<Grid> {
        validate_id(grid_id, "grid id")?;
        if self.grids.contains_key(grid_id) {
            return Err(Error::invalid(format!("grid {} is already declared", repr_str(grid_id))));
        }
        let units = options.units.clone().unwrap_or_else(|| "mm".into());
        if !KNOWN_UNITS.contains(&units.as_str()) {
            return Err(Error::coded(
                "E109",
                format!(
                    "grid {}: units {} is not one of {} (§3.2); convert spacing and origin rather than declaring another unit",
                    repr_str(grid_id),
                    repr_str(&units),
                    repr_list(&KNOWN_UNITS)
                ),
            ));
        }
        if let Some(tu) = &options.time_units {
            if !TIME_UNITS.contains(&tu.as_str()) {
                return Err(Error::coded(
                    "E109",
                    format!(
                        "grid {}: time_units {} is not one of {} (§3.2)",
                        repr_str(grid_id),
                        repr_str(tu),
                        repr_list(&TIME_UNITS)
                    ),
                ));
            }
        }
        let axis_kinds = match options.axis_kinds {
            Some(k) => k,
            None => default_axis_kinds(shape.len())?,
        };
        let axis_names = options.axis_names.unwrap_or_else(|| default_axis_names(&axis_kinds));
        if axis_kinds.iter().any(|k| k == "time") && options.time_values.is_none() {
            return Err(Error::coded(
                "E109",
                format!(
                    "grid {} declares a time axis but no time_values; §3.2 requires one acquisition time per frame",
                    repr_str(grid_id)
                ),
            ));
        }
        let n_spatial = axis_kinds.iter().filter(|k| *k == "spatial").count();
        let mut grid = Grid::new(
            grid_id,
            shape.to_vec(),
            axis_names,
            axis_kinds,
            spacing.to_vec(),
            options.origin.unwrap_or_else(|| vec![0.0; n_spatial]),
            options.direction.unwrap_or_else(|| Array2::eye(n_spatial)),
            options.coord_system.unwrap_or_else(|| "LPS".into()),
            units,
        );
        if let Ok(g) = &mut grid {
            g.timepoint = options.timepoint;
            g.frame_uid = options.frame_uid;
            g.time_values = options.time_values;
            g.time_units = options.time_units;
            g.patch_hint = options.patch_hint;
            g.chunk_hint = options.chunk_hint;
            g.check()?;
        }
        let grid = grid?;
        write_grid(&self.group("grids")?, &grid)?;
        self.grids.insert(grid_id.to_string(), grid.clone());
        Ok(grid)
    }

    /// Grids declared so far.
    pub fn grids(&self) -> &IndexMap<String, Grid> {
        &self.grids
    }

    /// A declared grid, or E101.
    pub fn grid_ref(&self, grid_id: &str) -> Result<&Grid> {
        self.grids.get(grid_id).ok_or_else(|| {
            Error::coded(
                "E101",
                format!("grid {} is not declared; declare it before referencing it", repr_str(grid_id)),
            )
        })
    }

    /// Rename frames of reference everywhere they are named (§3.4): on
    /// grids, world-space annotations and transforms at once.  Returns the
    /// attributes rewritten.
    ///
    /// HDF5 leaves an overwritten attribute's old value in the file's free
    /// space, so after a rename the commit rewrites the file from its root:
    /// rewritten in place alone, every old UID stayed in the committed bytes.
    pub fn remap_frame_uids(&mut self, mapping: &HashMap<String, String>) -> Result<Vec<String>> {
        let root = self.root()?;
        let mut changed = Vec::new();
        for (group, names) in FRAME_ATTRS {
            let Some(node) = ops::child_group(&root, group) else { continue };
            for name in ops::members(&node)? {
                let Some(obj) = ops::child_group(&node, &name) else { continue };
                for attr in names {
                    let current = attrs::get_str(&obj, attr)?.unwrap_or_default();
                    match mapping.get(&current) {
                        Some(replacement) if replacement != &current => {
                            self.compact = true;
                            attrs::write(&obj, attr, &AttrValue::Str(replacement.clone()))?;
                            changed.push(format!("{group}.{name}.{attr}"));
                        }
                        _ => {}
                    }
                }
            }
        }
        if !changed.is_empty() {
            self.grids = read_grids(&root)?.into_iter().collect();
            let transforms = ops::child_group(&root, "transforms");
            for (name, frames) in self.transform_frames.iter_mut() {
                if let Some(g) = transforms.as_ref().and_then(|t| ops::child_group(t, name)) {
                    *frames = (
                        attrs::get_str(&g, "from_frame")?.unwrap_or_default(),
                        attrs::get_str(&g, "to_frame")?.unwrap_or_default(),
                    );
                }
            }
        }
        Ok(changed)
    }

    /// Every frame UID in the file so far, and the attributes naming it.
    pub fn frame_uids(&self) -> Result<IndexMap<String, Vec<String>>> {
        frame_references(&self.root()?)
    }

    // -- images -------------------------------------------------------------------

    pub(crate) fn chunks_for(&self, grid: &Grid, itemsize: usize, leading: usize) -> Result<Vec<usize>> {
        grid_chunks(grid, itemsize, leading)
    }

    fn profile(&self, codec: Option<&str>) -> Result<crate::storage::codecs::CodecProfile> {
        resolve_profile(Some(codec.unwrap_or(&self.codec)))
    }

    /// Write one image, chunked for its grid's patch hint.
    pub fn add_image(
        &mut self,
        image_id: &str,
        data: &NdArray,
        grid: &str,
        modality: &str,
        options: ImageOptions,
    ) -> Result<hdf5::Dataset> {
        let (target, value_type) = self.check_image(image_id, grid, &options)?;
        let shape: Vec<i64> = data.shape().iter().map(|v| *v as i64).collect();
        if shape != target.shape {
            return Err(Error::coded(
                "E202",
                format!(
                    "image {} has shape {}; grid {} declares {}",
                    repr_str(image_id),
                    repr_int_tuple(&data.shape()),
                    repr_str(grid),
                    repr_int_tuple(&target.shape)
                ),
            ));
        }
        let chunks = self.chunks_for(&target, data.dtype().itemsize(), 0)?;
        let layout = dataset_layout(
            &data.shape(),
            data.dtype().itemsize(),
            &self.profile(options.codec.as_deref())?,
            Role::Image,
            Some(chunks),
        );
        let parent = self.group("images")?;
        let node = data::create(&parent, image_id, data, &layout)?;
        self.finish_image(&parent, node, image_id, grid, modality, value_type, options)
    }

    /// An image of `dtype` on `grid`, created empty for the caller to fill
    /// region by region ([`data::write_region`]), with its chunk shape: what
    /// a derivation too large to hold whole writes ([`crate::transforms::resample`]).
    /// The commit digests what the regions wrote.
    pub(crate) fn create_image(
        &mut self,
        image_id: &str,
        dtype: crate::array::DType,
        grid: &str,
        modality: &str,
        options: ImageOptions,
    ) -> Result<(hdf5::Dataset, Option<Vec<usize>>)> {
        let (target, value_type) = self.check_image(image_id, grid, &options)?;
        let shape = target.shape_usize();
        let chunks = self.chunks_for(&target, dtype.itemsize(), 0)?;
        let layout = dataset_layout(
            &shape,
            dtype.itemsize(),
            &self.profile(options.codec.as_deref())?,
            Role::Image,
            Some(chunks),
        );
        let parent = self.group("images")?;
        let node = data::create_empty(&parent, image_id, dtype, &shape, &layout)?;
        let chunks = layout.chunks.clone();
        Ok((self.finish_image(&parent, node, image_id, grid, modality, value_type, options)?, chunks))
    }

    /// Refuse an image the sample cannot take before anything is written:
    /// its grid and `value_type`, and the grid's channel names.
    fn check_image(&self, image_id: &str, grid: &str, options: &ImageOptions) -> Result<(Grid, String)> {
        validate_id(image_id, "image id")?;
        if self.images.contains_key(image_id) {
            return Err(Error::invalid(format!("image {} already exists", repr_str(image_id))));
        }
        let value_type = options.value_type.clone().unwrap_or_else(|| "intensity".into());
        super::image::check_value_type(&value_type)?;
        let target = self.grid_ref(grid)?.clone();
        if let Some(names) = &options.channel_names {
            let ok = match target.channel_axis() {
                Some(axis) => names.len() as i64 == target.shape[axis],
                None => false,
            };
            if !ok {
                return Err(Error::coded(
                    "E204",
                    format!("image {}: channel_names does not match the channel axis", repr_str(image_id)),
                ));
            }
        }
        Ok((target, value_type))
    }

    /// Write an image dataset's attributes and register it.
    #[allow(clippy::too_many_arguments)]
    fn finish_image(
        &mut self,
        parent: &hdf5::Group,
        node: hdf5::Dataset,
        image_id: &str,
        grid: &str,
        modality: &str,
        value_type: String,
        options: ImageOptions,
    ) -> Result<hdf5::Dataset> {
        let mut values: Vec<(&str, Option<AttrValue>)> = vec![
            ("grid", Some(AttrValue::Str(grid.into()))),
            ("modality", Some(AttrValue::Str(modality.into()))),
            ("value_type", Some(AttrValue::Str(value_type))),
            ("value_units", options.value_units.map(AttrValue::Str)),
            ("channel_names", options.channel_names.filter(|v| !v.is_empty()).map(|v| AttrValue::strs(&v))),
            ("rescale_slope", options.rescale_slope.map(AttrValue::Float)),
            ("rescale_intercept", options.rescale_intercept.map(AttrValue::Float)),
            ("window_center", options.window_center.filter(|v| !v.is_empty()).map(|v| AttrValue::floats(&v))),
            ("window_width", options.window_width.filter(|v| !v.is_empty()).map(|v| AttrValue::floats(&v))),
            ("valid_mask", options.valid_mask.map(AttrValue::Str)),
            ("prov", options.prov.map(AttrValue::Str)),
        ];
        values.retain(|(_, v)| v.is_some());
        self.or_unlink(parent, image_id, attrs::write_all(&node, &values))?;
        self.images.insert(image_id.to_string(), (grid.to_string(), modality.to_string()));
        self.image_multiscale.insert(image_id.to_string(), false);
        Ok(node)
    }

    /// Drop an image, its `/meta` `acquisition` entry and its `series_uids`
    /// mention in every timepoint.  What still names it --- a `valid_mask`,
    /// an annotation's `derived_from` --- and the last image going (E201) are
    /// checked at commit, so removals and additions may come in any order.
    pub fn remove_image(&mut self, image_id: &str) -> Result<()> {
        let node = self.group("images")?;
        if !self.images.contains_key(image_id) || !ops::exists(&node, image_id) {
            return Err(Error::invalid(format!("no image {} to remove", repr_str(image_id))));
        }
        ops::unlink(&node, image_id)?;
        // Its bytes stay in the file being built until commit rewrites it.
        self.compact = true;
        self.images.shift_remove(image_id);
        self.image_multiscale.remove(image_id);
        self.document.acquisition.remove(image_id);
        if self.document.timepoints.iter().any(|t| t.series_uids.contains_key(image_id)) {
            let points = self
                .document
                .timepoints
                .iter()
                .cloned()
                .map(|mut t| {
                    t.series_uids.remove(image_id);
                    t
                })
                .collect();
            self.document.timepoints = Timeline::new(points)?;
        }
        Ok(())
    }

    /// Drop a grid.  What still names it --- an image, an annotation, a
    /// transform's field (E101, E503) --- and the last grid going (E111) are
    /// checked at commit.
    pub fn remove_grid(&mut self, grid_id: &str) -> Result<()> {
        let node = self.group("grids")?;
        if !self.grids.contains_key(grid_id) || !ops::exists(&node, grid_id) {
            return Err(Error::invalid(format!("no grid {} to remove", repr_str(grid_id))));
        }
        ops::unlink(&node, grid_id)?;
        self.compact = true;
        self.grids.shift_remove(grid_id);
        Ok(())
    }

    /// Change an image's attributes without rewriting its samples (§4.1).
    ///
    /// The values are checked as [`add_image`](Self::add_image) checks them:
    /// an unknown `value_type` (E203), `channel_names` that do not match the
    /// grid's channel extent (E204), a rescale that is not finite, and a
    /// required attribute removed (E205).  `grid`, `digest` and the pyramid
    /// attributes are not attributes it edits.
    pub fn set_image_attrs(&mut self, image_id: &str, edit: ImageEdit) -> Result<()> {
        let images = self.group("images")?;
        let Some((grid_id, _)) = self.images.get(image_id).cloned().filter(|_| ops::exists(&images, image_id)) else {
            return Err(Error::invalid(format!("no image {} to edit", repr_str(image_id))));
        };
        let at = |name: &str| format!("image {}: {name}", repr_str(image_id));
        for (name, removed) in
            [("modality", matches!(edit.modality, Some(None))), ("value_type", matches!(edit.value_type, Some(None)))]
        {
            if removed {
                return Err(Error::coded("E205", format!("{} is required (§4.1) and cannot be removed", at(name))));
            }
        }
        if let Some(Some(value_type)) = &edit.value_type {
            super::image::check_value_type(value_type)?;
        }
        if let Some(Some(names)) = &edit.channel_names {
            let grid = self.grid_ref(&grid_id)?;
            if !grid.channel_axis().is_some_and(|axis| names.len() as i64 == grid.shape[axis]) {
                return Err(Error::coded("E204", format!("{} does not match the channel axis", at("channel_names"))));
            }
        }
        for (name, value) in [("rescale_slope", edit.rescale_slope), ("rescale_intercept", edit.rescale_intercept)] {
            if let Some(Some(v)) = value {
                if !v.is_finite() {
                    return Err(Error::invalid(format!("{} must be finite, not {v}", at(name))));
                }
            }
        }
        let strs = |v: Option<Option<String>>| v.map(|v| v.map(AttrValue::Str));
        let floats = |v: Option<Option<Vec<f64>>>| v.map(|v| v.map(|v| AttrValue::floats(&v)));
        let changes: Vec<(&str, Option<Option<AttrValue>>)> = vec![
            ("modality", strs(edit.modality.clone())),
            ("value_type", strs(edit.value_type)),
            ("value_units", strs(edit.value_units)),
            ("channel_names", edit.channel_names.map(|v| v.map(|v| AttrValue::strs(&v)))),
            ("rescale_slope", edit.rescale_slope.map(|v| v.map(AttrValue::Float))),
            ("rescale_intercept", edit.rescale_intercept.map(|v| v.map(AttrValue::Float))),
            ("window_center", floats(edit.window_center)),
            ("window_width", floats(edit.window_width)),
            ("valid_mask", strs(edit.valid_mask)),
            ("prov", strs(edit.prov)),
        ];
        let node: hdf5::Location = if ops::is_group(&images, image_id) {
            (*images.group(image_id)?).clone()
        } else {
            (**images.dataset(image_id)?).clone()
        };
        for (name, change) in changes {
            match change {
                None => {}
                Some(None) => {
                    if attrs::has(&node, name) {
                        attrs::delete(&node, name)?;
                    }
                }
                Some(Some(value)) => attrs::write(&node, name, &value)?,
            }
        }
        // An overwritten attribute's old value stays in the file's free
        // space until the commit rewrites it.
        self.compact = true;
        if let Some(Some(modality)) = edit.modality {
            self.images.insert(image_id.to_string(), (grid_id, modality));
        }
        Ok(())
    }

    /// Write a multiscale image (§4.3); the level geometry is checked here.
    #[allow(clippy::too_many_arguments)]
    pub fn add_pyramid(
        &mut self,
        image_id: &str,
        levels: &[NdArray],
        grid_levels: &[String],
        modality: &str,
        downsample_method: &str,
        options: ImageOptions,
    ) -> Result<hdf5::Group> {
        validate_id(image_id, "image id")?;
        if self.images.contains_key(image_id) {
            return Err(Error::invalid(format!("image {} already exists", repr_str(image_id))));
        }
        let value_type = options.value_type.clone().unwrap_or_else(|| "intensity".into());
        super::image::check_value_type(&value_type)?;
        if levels.is_empty() {
            return Err(Error::coded(
                "E105",
                format!("image {}: a pyramid needs at least one level", repr_str(image_id)),
            ));
        }
        if levels.len() != grid_levels.len() {
            return Err(Error::coded(
                "E105",
                format!("image {}: {} arrays for {} grid levels", repr_str(image_id), levels.len(), grid_levels.len()),
            ));
        }
        let grids: Vec<Grid> = grid_levels.iter().map(|g| self.grid_ref(g).cloned()).collect::<Result<_>>()?;
        let refs: Vec<&Grid> = grids.iter().collect();
        let factors = pyramid_factors(&grids[0], &refs);
        let problems = check_pyramid(&grids[0], &refs, &factors, crate::geometry::multiscale::GEOMETRY_RTOL)?;
        if !problems.is_empty() {
            return Err(Error::coded(
                "E105",
                format!("image {}: inconsistent pyramid geometry: {}", repr_str(image_id), problems.join("; ")),
            ));
        }
        // Every check comes before the first write, so a refused pyramid
        // leaves nothing a corrected retry would collide with.
        let profile = self.profile(options.codec.as_deref())?;
        let mut layouts = Vec::with_capacity(levels.len());
        for (level, (array, target)) in levels.iter().zip(&grids).enumerate() {
            let shape: Vec<i64> = array.shape().iter().map(|v| *v as i64).collect();
            if shape != target.shape {
                return Err(Error::coded(
                    "E202",
                    format!(
                        "image {} level {level} has shape {}; grid {} declares {}",
                        repr_str(image_id),
                        repr_int_tuple(&array.shape()),
                        repr_str(&target.grid_id),
                        repr_int_tuple(&target.shape)
                    ),
                ));
            }
            let chunks = self.chunks_for(target, array.dtype().itemsize(), 0)?;
            layouts.push(dataset_layout(&array.shape(), array.dtype().itemsize(), &profile, Role::Image, Some(chunks)));
        }
        let pyramid = Pyramid::new(levels.len(), factors, downsample_method, grid_levels.to_vec())?;
        let mut values: Vec<(String, AttrValue)> = vec![
            ("grid".into(), AttrValue::Str(grid_levels[0].clone())),
            ("modality".into(), AttrValue::Str(modality.into())),
            ("value_type".into(), AttrValue::Str(value_type)),
        ];
        if let Some(v) = options.value_units {
            values.push(("value_units".into(), AttrValue::Str(v)));
        }
        if let Some(v) = options.rescale_slope {
            values.push(("rescale_slope".into(), AttrValue::Float(v)));
        }
        if let Some(v) = options.rescale_intercept {
            values.push(("rescale_intercept".into(), AttrValue::Float(v)));
        }
        if let Some(v) = options.prov {
            values.push(("prov".into(), AttrValue::Str(v)));
        }
        values.extend(pyramid.attrs());
        let parent = self.group("images")?;
        let group = ops::new_group(&parent, image_id)?;
        let written = levels
            .iter()
            .zip(&layouts)
            .enumerate()
            .try_for_each(|(level, (array, layout))| data::create(&group, &level.to_string(), array, layout).map(drop))
            .and_then(|()| values.iter().try_for_each(|(k, v)| attrs::write(&group, k, v)));
        self.or_unlink(&parent, image_id, written)?;
        self.images.insert(image_id.to_string(), (grid_levels[0].clone(), modality.to_string()));
        self.image_multiscale.insert(image_id.to_string(), true);
        Ok(group)
    }

    // -- commit -------------------------------------------------------------------

    /// Profiles this sample actually satisfies, unioned with declared ones.
    pub fn infer_profiles(&self) -> Result<BTreeSet<String>> {
        let kinds: BTreeSet<&str> = self.annotation_kinds.values().map(String::as_str).collect();
        let mut found: BTreeSet<String> = BTreeSet::from(["core".to_string()]);
        let doc = &self.document;
        let has_labels = doc.label_set.is_some();
        if ["labelmap", "layers", "bitmask", "instances", "probmap"].iter().any(|k| kinds.contains(k)) && has_labels {
            found.insert("seg".into());
        }
        if has_labels && self.has_task("detection")? {
            found.insert("det".into());
        }
        if kinds.contains("classification") && has_labels {
            found.insert("cls".into());
        }
        // `reg` and `curation` by the predicates the validator holds them to
        // (E009), so a commit never refuses a profile the writer declared.
        if crate::validate::rules::relates_grids(&self.root()?)? {
            found.insert("reg".into());
        }
        if crate::validate::rules::records_activity(doc) && !self.annotation_kinds.is_empty() {
            let anns = self.group("annotations")?;
            let mut all = true;
            for name in self.annotation_kinds.keys() {
                let g = anns.group(name)?;
                if !attrs::has(&g, "quality") {
                    all = false;
                    break;
                }
            }
            if all {
                found.insert("curation".into());
            }
        }
        if !self.images.is_empty() && self.images.keys().all(|i| self.image_multiscale.get(i).copied().unwrap_or(false))
        {
            found.insert("multiscale".into());
        }
        if let Some(index) = ops::child_group(&self.root()?, "index") {
            if !ops::members(&index)?.is_empty() {
                found.insert("training".into());
            }
        }
        if doc.timepoints.is_longitudinal()
            && self.grids.values().all(|g| g.timepoint.as_deref().is_some_and(|t| !t.is_empty()))
        {
            found.insert("longitudinal".into());
        }
        if self.clinical.is_some() || self.clinical_source == ClinicalSource::Inherited {
            found.insert(crate::clinical::PROFILE.into());
        }
        let mut unknown: Vec<String> =
            self.declared_profiles.union(&found).filter(|p| !PROFILES.contains(&p.as_str())).cloned().collect();
        unknown.sort();
        if !unknown.is_empty() {
            return Err(Error::coded("E007", format!("unknown profile(s) {}", repr_list(&unknown))));
        }
        Ok(found.union(&self.declared_profiles).cloned().collect())
    }

    fn has_task(&self, task: &str) -> Result<bool> {
        let anns = self.group("annotations")?;
        for name in ops::members(&anns)? {
            // A curve serves no profile, whatever task it states (1.2 §3).
            if self.annotation_kinds.get(&name).is_some_and(|k| k == "polylines") {
                continue;
            }
            if let Some(g) = ops::child_group(&anns, &name) {
                if attrs::get_str(&g, "task")?.as_deref() == Some(task) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn check_references(&self) -> Result<()> {
        let doc = &self.document;
        let declared: BTreeSet<String> = doc.timepoints.ids().into_iter().collect();
        let multi = declared.len() > 1;
        for grid in self.grids.values() {
            match &grid.timepoint {
                None => {
                    if multi {
                        return Err(Error::coded(
                            "E106",
                            format!(
                                "grid {} has no `timepoint`, but the sample declares {} timepoints",
                                repr_str(&grid.grid_id),
                                declared.len()
                            ),
                        ));
                    }
                }
                Some(tp) if !declared.contains(tp) => {
                    return Err(Error::coded(
                        "E107",
                        format!("grid {} names undeclared timepoint {}", repr_str(&grid.grid_id), repr_str(tp)),
                    ))
                }
                _ => {}
            }
        }
        let anns = self.group("annotations")?;
        for name in self.annotation_kinds.keys() {
            let node = anns.group(name)?;
            if let Some(ls) = &doc.label_set {
                if ls.form == "inline" {
                    if let Some(ids) = attrs::get_i64s(&node, "class_ids")? {
                        let missing = ls.missing(ids);
                        if !missing.is_empty() {
                            return Err(Error::coded(
                                "E402",
                                format!(
                                    "annotation {} uses class ids {} that are not in label set {}",
                                    repr_str(name),
                                    crate::json::repr_int_list(&missing),
                                    repr_str(&ls.id)
                                ),
                            ));
                        }
                    }
                }
            }
            if let Some(prov) = attrs::get_str(&node, "prov")? {
                if !doc.provenance.has_activity(&prov) {
                    return Err(Error::coded(
                        "E601",
                        format!(
                            "annotation {} names activity {}, which is not in the provenance graph",
                            repr_str(name),
                            repr_str(&prov)
                        ),
                    ));
                }
            }
            if let Some(q) = attrs::get_str(&node, "quality")? {
                if !doc.quality.contains_key(&q) {
                    return Err(Error::coded(
                        "E602",
                        format!(
                            "annotation {} names quality record {}, which does not exist",
                            repr_str(name),
                            repr_str(&q)
                        ),
                    ));
                }
            }
            if let Some(tps) = attrs::get_strs(&node, "timepoints")? {
                for tp in tps {
                    if !declared.contains(&tp) {
                        return Err(Error::coded(
                            "E409",
                            format!("annotation {} names undeclared timepoint {}", repr_str(name), repr_str(&tp)),
                        ));
                    }
                }
            }
            if let Some(mask) = attrs::get_str(&node, "ignore_mask")? {
                self.check_mask_reference(&format!("annotation {}", repr_str(name)), "ignore_mask", &mask, &node)?;
            }
            if let Some(derived) = attrs::get_strs(&node, "derived_from")? {
                for reference in derived {
                    if !self.annotation_kinds.contains_key(annotation_id(&reference)) {
                        return Err(Error::coded(
                            "E413",
                            format!(
                                "annotation {}: derived_from names {}, which does not exist",
                                repr_str(name),
                                repr_str(&reference)
                            ),
                        ));
                    }
                }
            }
        }
        let images = self.group("images")?;
        for image_id in self.images.keys() {
            let loc: hdf5::Location = if ops::is_group(&images, image_id) {
                (*images.group(image_id)?).clone()
            } else {
                let d = images.dataset(image_id)?;
                (**d).clone()
            };
            if let Some(prov) = attrs::get_str(&loc, "prov")? {
                if !doc.provenance.has_activity(&prov) {
                    return Err(Error::coded(
                        "E601",
                        format!(
                            "image {} names activity {}, which is not in the provenance graph",
                            repr_str(image_id),
                            repr_str(&prov)
                        ),
                    ));
                }
            }
            if let Some(mask) = attrs::get_str(&loc, "valid_mask")? {
                self.check_mask_reference(&format!("image {}", repr_str(image_id)), "valid_mask", &mask, &loc)?;
            }
        }
        if !self.transform_frames.is_empty() {
            let ts = self.group("transforms")?;
            for name in self.transform_frames.keys() {
                let g = ts.group(name)?;
                if let Some(prov) = attrs::get_str(&g, "prov")? {
                    if !doc.provenance.has_activity(&prov) {
                        return Err(Error::coded(
                            "E601",
                            format!(
                                "transform {} names activity {}, which is not in the provenance graph",
                                repr_str(name),
                                repr_str(&prov)
                            ),
                        ));
                    }
                }
                if let Some(m) = attrs::get_str(&g, "metrics")? {
                    if !doc.quality.contains_key(&m) {
                        return Err(Error::coded(
                            "E602",
                            format!(
                                "transform {} names quality record {}, which does not exist",
                                repr_str(name),
                                repr_str(&m)
                            ),
                        ));
                    }
                }
                // A transform declared before the grids of its frames was
                // checked against none of them, and one the file already held
                // against none at all (§10.1, E506).  An amendment mends the
                // latter with `remove_transform` and `add_transform`.
                let (from_frame, to_frame) = &self.transform_frames[name];
                let units = attrs::get_str(&g, "units")?.unwrap_or_else(|| "mm".into());
                let mut named: Vec<String> = Vec::new();
                for key in ["from_grid", "to_grid"] {
                    named.extend(attrs::get_str(&g, key)?);
                }
                let grids = self.grids.values().filter(|grid| {
                    grid.frame_uid.as_deref().is_some_and(|f| f == from_frame || f == to_frame)
                        || named.contains(&grid.grid_id)
                });
                let wrong = crate::transforms::model::units_disagreeing(grids.clone(), &units);
                if !wrong.is_empty() {
                    return Err(Error::coded(
                        "E506",
                        format!(
                            "transform {} is in {}, and {}: a transform maps coordinates in one unit, and none maps \
                             between two (§10.1)",
                            repr_str(name),
                            repr_str(&units),
                            wrong.join(", ")
                        ),
                    ));
                }
                // So, for the same reason, is an affine's dimension (§10.3, E504).
                crate::transforms::model::check_affine_rank(name, &g, grids)?;
            }
        }
        let dangling = doc.provenance.dangling_agent_refs();
        if !dangling.is_empty() {
            let pairs: Vec<String> = dangling.iter().map(|(a, g)| format!("{a} -> {g}")).collect();
            return Err(Error::coded("E605", format!("activities name undeclared agents: {}", pairs.join(", "))));
        }
        Ok(())
    }

    fn check_mask_reference(&self, owner: &str, attr: &str, target: &str, node: &hdf5::Location) -> Result<()> {
        let Some(kind) = self.annotation_kinds.get(target) else {
            return Err(Error::coded(
                "E413",
                format!("{owner}: {attr} names annotation {}, which does not exist", repr_str(target)),
            ));
        };
        if kind != "mask" {
            return Err(Error::coded(
                "E413",
                format!(
                    "{owner}: {attr} names {}, whose kind is {}; §4.4 and §7.7 require a `mask` annotation",
                    repr_str(target),
                    repr_str(kind)
                ),
            ));
        }
        let mine = attrs::get_str(node, "grid")?;
        let target_group = self.group("annotations")?.group(target)?;
        let theirs = attrs::get_str(&target_group, "grid")?;
        if let (Some(m), Some(t)) = (mine, theirs) {
            if m != t {
                return Err(Error::coded(
                    "E413",
                    format!(
                        "{owner}: {attr} names {} on grid {}, but this object is on grid {}",
                        repr_str(target),
                        repr_str(&t),
                        repr_str(&m)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Validate, write `/meta`, stamp digests and atomically replace.
    ///
    /// `digests = true` recomputes every dataset's digest.  An amendment never
    /// restamps a digest its source already failed: a dataset whose bytes did
    /// not match its digest in the file being amended is evidence --- of an
    /// edit made behind the library's back, or of damage --- so the commit is
    /// refused (E701) and nothing is written.  Restoring the bytes, or
    /// restamping them deliberately with `medh5 fix --rewrite-digests`, which
    /// records a reason in the file, are the ways on.  A dataset this
    /// amendment rewrote is stamped afresh.
    ///
    /// `digests = false` stamps only the datasets that carry no digest yet,
    /// and leaves every other as it was --- but for one in another algorithm
    /// than the file's, restamped in the file's and checked as any changed
    /// digest is, so a file never carries two.  `content_id` is always
    /// computed.
    ///
    /// A commit refused before it changes anything --- a sample without an
    /// image (E201), a reference that does not resolve, a document its schema
    /// rejects (E005), an event version the amended file held, changed ---
    /// leaves the writer open: mend what was refused and commit again.  One
    /// that fails once it has begun writing aborts the writer, discarding the
    /// file being built and leaving the target as it was; so does one whose
    /// source changed after the amendment read it (§14.4).  Returns the
    /// `content_id`, or `None` when the writer had already committed; an
    /// aborted writer has nothing to commit, and says so.
    pub fn commit(&mut self, digests: bool) -> Result<Option<String>> {
        stage!("commit");
        match self.state {
            State::Committed => return Ok(None),
            State::Aborted => {
                return Err(Error::invalid(
                    "this writer was aborted --- by abort(), or by a commit that failed once it had begun writing \
                     --- so there is nothing to commit, and the target is as it was; build the sample again",
                ))
            }
            State::Open => {}
        }
        // Before the source is read again: what it holds now may not be what
        // this amendment inherited, and no commit of this writer could be
        // anything but a lost update (§14.4).
        if let (Some(source), Some(seen)) = (&self.source, &self.seen) {
            if let Err(e) = seen.check(source) {
                self.abort();
                return Err(e);
            }
        }
        // Refused here, nothing has changed: the writer stays open.
        let profiles = self.check_commit()?;
        let result = self.commit_inner(digests, &profiles);
        if result.is_err() {
            self.abort();
        }
        result.map(Some)
    }

    /// The `medh5_version` the commit writes ([`written_version_for_content`]).
    ///
    /// The amended file's version is kept unless the amendment removed the
    /// last annotation of a kind that version added (1.2 §2.1): the file was
    /// at that version for the kind, and without it what remains is written
    /// at the lowest version it needs, as after removing the clinical profile.
    fn version_written(&self, profiles: &[String]) -> String {
        let kinds: Vec<&str> = self.annotation_kinds.values().map(String::as_str).collect();
        let removed = self.source_kinds.iter().any(|k| {
            crate::annotations::header::kind_since(k).is_some_and(|since| since != crate::version::BASE_VERSION)
                && !kinds.contains(&k.as_str())
        });
        let source = self.source_version.as_deref().filter(|_| !removed);
        written_version_for_content(source, profiles, &kinds)
    }

    /// What a commit refuses before it changes anything, and the profiles it
    /// declares.
    fn check_commit(&self) -> Result<Vec<String>> {
        if self.images.is_empty() {
            return Err(Error::coded("E201", "a sample must contain at least one image"));
        }
        self.check_references()?;
        let errors = self.document.check_schema();
        if !errors.is_empty() {
            return Err(Error::coded(
                "E005",
                format!(
                    "sample document fails its JSON Schema: {}",
                    errors.iter().take(5).cloned().collect::<Vec<_>>().join("; ")
                ),
            ));
        }
        self.check_held()?;
        Ok(self.infer_profiles()?.into_iter().collect())
    }

    fn commit_inner(&mut self, digests: bool, profiles: &[String]) -> Result<String> {
        let root = self.root()?;
        ops::unlink(&root, META_DATASET)?;
        data::create_scalar_string(&root, META_DATASET, &self.document.dumps())?;
        for alias in &self.meta_aliases {
            // Gone if this amendment removed it, or the group holding it.
            if ops::exists(&root, alias) {
                ops::unlink(&root, alias)?;
                root.link_hard(META_DATASET, alias)?;
            }
        }
        self.write_clinical()?;
        let version = self.version_written(profiles);
        let algo = self.digest_algo.clone();
        attrs::write(&root, "medh5_version", &AttrValue::Str(version))?;
        attrs::write(&root, "medh5_kind", &AttrValue::Str("sample".into()))?;
        attrs::write(&root, "medh5_profiles", &crate::annotations::header::list_attr(profiles))?;
        let created = self.created.clone().unwrap_or_else(utcnow);
        attrs::write(&root, "created", &AttrValue::Str(created))?;
        attrs::write(&root, "generator", &AttrValue::Str(format!("medh5 {VERSION}")))?;
        attrs::write(&root, "digest_algo", &AttrValue::Str(algo.clone()))?;
        let stored = collect_digests(&root, &["index"])?;
        let stamped = {
            stage!("commit.digests");
            stamp_in(&root, &algo, !digests)?
        };
        {
            stage!("commit.restamp_check");
            self.refuse_restamping(&stored, &stamped)?;
        }
        let content_id = compute_content_id(&root, &attr_name_map_of(&root)?, &algo, None)?;
        attrs::write(&root, "content_id", &AttrValue::Str(content_id.clone()))?;
        {
            stage!("commit.validate");
            self.check_valid()?;
        }
        drop(root);
        stage!("commit.replace");
        let file = self.file.take().ok_or_else(|| Error::invalid("this writer has already committed or aborted"))?;
        if self.compact {
            // Only what the root reaches comes across; the copy keeps the
            // graph, so every digest and the `content_id` are unchanged.
            file.rebuild(crate::collection::copy_root)?.commit()?;
        } else {
            file.commit()?;
        }
        self.state = State::Committed;
        Ok(content_id)
    }

    /// Refuse to restamp a digest the amended file already failed (E701).
    ///
    /// A digest the stamp changed belongs to bytes that no longer match it:
    /// bytes this amendment rewrote (a scrub de-identifying a string dataset),
    /// or bytes that did not match it in the file being amended either.  The
    /// source tells the two apart.  A restamp of the second kind erased the
    /// only record that the bytes had changed.
    fn refuse_restamping(&self, stored: &IndexMap<String, String>, stamped: &IndexMap<String, String>) -> Result<()> {
        let changed: Vec<String> = stamped
            .iter()
            .filter(|(path, value)| stored.get(*path).is_some_and(|before| before != *value))
            .map(|(path, _)| path.clone())
            .collect();
        let Some(source) = self.source.as_deref().filter(|_| !changed.is_empty()) else {
            return Ok(());
        };
        let file = open_read(source)?;
        let root = file.as_group()?;
        let failed = failing_digests(&root, &changed)?;
        // A digest the source holds that does not parse is no record of
        // anything a restamp could keep (E703), not a mismatch.
        let unreadable = failed.iter().any(|path| {
            ops::resolve_dataset(&root, path)
                .and_then(|ds| attrs::get_str(&ds, "digest").ok().flatten())
                .is_some_and(|value| !crate::digest::is_canonical_digest(&value))
        });
        drop(root);
        drop(file);
        if failed.is_empty() {
            return Ok(());
        }
        let shown = repr_str(&source.to_string_lossy());
        if unreadable {
            return Err(unreadable_refused(&shown, &failed));
        }
        Err(restamp_refused(&shown, &failed))
    }

    /// Refuse to write a file the validator would reject (§15): the
    /// validator's structural and semantic *error* rules over the finished
    /// temporary file are the one definition of a valid file.
    fn check_valid(&mut self) -> Result<()> {
        let root = self.root()?;
        let report = crate::validate::validate_root(&root, Some(&self.path), "semantic", true)?;
        let errors: Vec<_> = report.diagnostics.iter().filter(|d| d.severity == "error").collect();
        if errors.is_empty() {
            return Ok(());
        }
        let listed: Vec<String> =
            errors.iter().take(5).map(|d| format!("{} {}: {}", d.code, d.location, d.message)).collect();
        let more = if errors.len() > 5 { format!(" (and {} more)", errors.len() - 5) } else { String::new() };
        Err(Error::coded(
            &errors[0].code,
            format!(
                "refusing to write {}: the file would fail validation: {}{more}",
                repr_str(&self.path.to_string_lossy()),
                listed.join("; ")
            ),
        ))
    }
}

impl Drop for SampleWriter {
    fn drop(&mut self) {
        // A writer dropped without `commit` is an abort: the temporary
        // sibling goes, the target is untouched.
        if let Some(f) = self.file.take() {
            f.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrity::verify_root;

    /// One image on one grid, and a fixed-length string dataset outside
    /// every group the specification defines.
    fn sample(path: &Path) {
        sample_of(path, None);
    }

    /// [`sample`], of the subject `subject_id` names.
    fn sample_of(path: &Path, subject_id: Option<&str>) {
        let mut writer = create(path, Some("s"), subject_id, "portable", &[]).unwrap();
        writer.add_grid("g", &[4, 6, 8], &[1.0, 1.0, 1.0], GridOptions::default()).unwrap();
        let ct =
            NdArray::from_vec(&[4, 6, 8], (0..192).map(f64::from).collect()).unwrap().astype(crate::array::DType::I16);
        writer.add_image("CT", &ct, "g", "CT", ImageOptions::default()).unwrap();
        let notes = writer.root().unwrap().create_group("x_site").unwrap();
        let ds = notes.new_dataset::<hdf5::types::FixedAscii<8>>().shape([2]).create("notes").unwrap();
        data::write_fixed_strings(&ds, &["Jane Doe".into(), "ok".into()], 8, true).unwrap();
        writer.commit(true).unwrap();
    }

    fn mismatched(path: &Path) -> Vec<String> {
        let file = open_read(path).unwrap();
        let root = file.as_group().unwrap();
        verify_root(&root, Some(&attr_name_map_of(&root).unwrap()), None, true).unwrap().mismatched
    }

    /// An amendment restamps what it rewrote and refuses to restamp what its
    /// source already failed: every amend recomputed every digest, so a
    /// no-op amend of an edited file made it verify, with no record of the
    /// edit.
    #[test]
    fn s13_1_an_amendment_never_restamps_a_digest_its_source_failed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        sample(&path);

        // Rewritten in the session, as a scrub rewrites a string: stamped.
        let mut writer = amend(&path, None).unwrap();
        let notes = writer.root().unwrap().dataset("x_site/notes").unwrap();
        data::write_fixed_strings(&notes, &["redacted".into(), "ok".into()], 8, true).unwrap();
        drop(notes);
        writer.commit(true).unwrap();
        assert!(mismatched(&path).is_empty());

        // Edited behind the library's back: refused, and nothing written.
        {
            let file = crate::h5::file::marked(|| hdf5::File::open_rw(&path)).unwrap();
            let ct = file.dataset("images/CT").unwrap();
            ct.write_slice(&ndarray::arr3(&[[[999i16]]]), (0..1, 0..1, 0..1)).unwrap();
        }
        let before = std::fs::read(&path).unwrap();
        let err = amend(&path, None).unwrap().commit(true).unwrap_err();
        assert_eq!(err.code(), Some("E701"), "{err}");
        assert!(err.to_string().contains("images/CT") && err.to_string().contains("--rewrite-digests"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        // Stamping only what carries no digest restamps nothing it inherited.
        amend(&path, None).unwrap().commit(false).unwrap();
        assert_eq!(mismatched(&path), vec!["images/CT".to_string()]);
    }

    /// Read alone, a sample is refused as its amendment's commit refuses it,
    /// so a tool amending many files meets the refusal before writing any.
    #[test]
    fn s13_1_reading_refuses_the_amendment_its_commit_would() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        sample(&path);
        let shown = repr_str(&path.to_string_lossy());
        let read = || refuse_failed_digests(&open_read(&path).unwrap().as_group().unwrap(), &shown);
        read().unwrap();
        {
            let file = crate::h5::file::marked(|| hdf5::File::open_rw(&path)).unwrap();
            let ct = file.dataset("images/CT").unwrap();
            ct.write_slice(&ndarray::arr3(&[[[999i16]]]), (0..1, 0..1, 0..1)).unwrap();
        }
        let refused = read().unwrap_err();
        let committed = amend(&path, None).unwrap().commit(true).unwrap_err();
        assert_eq!((refused.code(), refused.to_string()), (committed.code(), committed.to_string()));
        assert!(!refused.to_string().contains("  "), "{refused}");
    }

    /// A commit refused before it changed anything leaves the writer open, to
    /// mend and commit again, and a second commit of a committed writer has
    /// nothing to do.  A refused commit killed the writer, and the next one
    /// returned `None`, as if it had written the file it never wrote.
    #[test]
    fn s14_4_a_commit_refused_before_writing_leaves_the_writer_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        let mut writer = create(&path, Some("s"), None, "portable", &[]).unwrap();
        writer.add_grid("g", &[2, 3, 4], &[1.0, 1.0, 1.0], GridOptions::default()).unwrap();
        let refused = writer.commit(true).unwrap_err();
        assert_eq!(refused.code(), Some("E201"), "{refused}");
        assert!(!writer.is_closed() && !path.exists());
        // A reference that does not resolve is refused alike, and mended.
        let ct = NdArray::from_vec(&[2, 3, 4], vec![0i16; 24]).unwrap();
        let options = ImageOptions { prov: Some("act_import_9".into()), ..Default::default() };
        writer.add_image("CT", &ct, "g", "CT", options).unwrap();
        assert_eq!(writer.commit(true).unwrap_err().code(), Some("E601"));
        assert!(!writer.is_closed() && !path.exists());
        let agent = writer.software("importer", None, Map::new()).unwrap();
        writer.activity("import", Some(&agent.id), Some("act_import_9"), Map::new()).unwrap();
        let content_id = writer.commit(true).unwrap();
        assert!(content_id.is_some() && path.exists());
        assert!(writer.is_closed());
        assert_eq!(writer.commit(true).unwrap(), None);
    }

    /// An aborted writer has nothing to commit, and says so --- aborted by
    /// its caller, or by a commit refused once it had begun writing --- and
    /// leaves the target as it was, with no temporary file beside it.
    #[test]
    fn s14_4_an_aborted_writer_refuses_to_commit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        let mut writer = create(&path, Some("s"), None, "portable", &[]).unwrap();
        writer.abort();
        assert!(writer.is_closed());
        let refused = writer.commit(true).unwrap_err();
        assert!(refused.to_string().contains("aborted"), "{refused}");
        assert!(!path.exists());

        sample(&path);
        {
            let file = crate::h5::file::marked(|| hdf5::File::open_rw(&path)).unwrap();
            let ct = file.dataset("images/CT").unwrap();
            ct.write_slice(&ndarray::arr3(&[[[999i16]]]), (0..1, 0..1, 0..1)).unwrap();
        }
        let before = std::fs::read(&path).unwrap();
        let mut writer = amend(&path, None).unwrap();
        // Refused by the stamp, after the commit had begun writing.
        assert_eq!(writer.commit(true).unwrap_err().code(), Some("E701"));
        assert!(writer.is_closed());
        let refused = writer.commit(true).unwrap_err();
        assert!(refused.to_string().contains("aborted"), "{refused}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    /// Every digest under `path`, and its `content_id`, restamped in `algo`,
    /// as a writer digesting in it would have written them.
    fn restamped_in(path: &Path, algo: &str) {
        let file = crate::h5::file::marked(|| hdf5::File::open_rw(path)).unwrap();
        let root = file.as_group().unwrap();
        attrs::write(&root, "digest_algo", &AttrValue::Str(algo.into())).unwrap();
        stamp_digests(&root, algo, &["index"], false).unwrap();
        let content_id = compute_content_id(&root, &attr_name_map_of(&root).unwrap(), algo, None).unwrap();
        attrs::write(&root, "content_id", &AttrValue::Str(content_id)).unwrap();
    }

    fn digests_and_address(path: &Path) -> (Option<String>, IndexMap<String, String>, Option<String>) {
        let file = open_read(path).unwrap();
        let root = file.as_group().unwrap();
        let algo = attrs::get_str(&root, "digest_algo").unwrap();
        (algo, collect_digests(&root, &["index"]).unwrap(), attrs::get_str(&root, "content_id").unwrap())
    }

    /// An amendment digests in the algorithm the file declares.  It
    /// restamped every digest in sha256 under a root it relabelled --- a new
    /// `content_id` for an unchanged sample --- and with `digests = false`
    /// kept the old digests under a root declaring sha256.  A digest kept in
    /// another algorithm than the file's is restamped in the file's.
    #[test]
    fn s13_1_an_amendment_keeps_the_files_digest_algorithm() {
        for algo in ["sha512", "blake2b"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("s.medh5");
            sample(&path);
            restamped_in(&path, algo);
            let (_, _, address) = digests_and_address(&path);
            for digests in [false, true] {
                amend(&path, None).unwrap().commit(digests).unwrap();
                let (declared, stored, after) = digests_and_address(&path);
                assert_eq!(declared.as_deref(), Some(algo));
                assert!(stored.values().all(|d| d.starts_with(&format!("{algo}:"))), "{algo}: {stored:?}");
                assert_eq!(after, address, "{algo}: a no-op amendment moved the address");
                assert!(mismatched(&path).is_empty());
            }
            // A third-party file carrying one digest in another algorithm,
            // its `content_id` computed over it.
            {
                let file = crate::h5::file::marked(|| hdf5::File::open_rw(&path)).unwrap();
                let notes = file.dataset("x_site/notes").unwrap();
                let other = crate::integrity::digest::dataset_digest(&notes, "x_site/notes", "sha256").unwrap();
                attrs::write(&notes, "digest", &AttrValue::Str(other)).unwrap();
                let root = file.as_group().unwrap();
                let content_id = compute_content_id(&root, &attr_name_map_of(&root).unwrap(), algo, None).unwrap();
                attrs::write(&root, "content_id", &AttrValue::Str(content_id)).unwrap();
            }
            amend(&path, None).unwrap().commit(false).unwrap();
            let (_, stored, after) = digests_and_address(&path);
            assert!(stored.values().all(|d| d.starts_with(&format!("{algo}:"))), "{algo}: {stored:?}");
            assert_eq!(after, address);
            assert!(mismatched(&path).is_empty());
        }
    }

    /// A hard link to `/meta` names the document the commit writes, under
    /// any name and wherever it sorts.  Copied by path, an alias kept the
    /// old document as an object of its own: an identifier the amendment
    /// replaced stayed in the file under it, and in its bytes.
    #[test]
    fn s2_4_a_hard_link_to_meta_names_the_document_the_commit_writes() {
        for alias in ["x_acme/meta", "a_first/meta", "meta_copy"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("s.medh5");
            sample_of(&path, Some("MRN-000042"));
            {
                let file = crate::h5::file::marked(|| hdf5::File::open_rw(&path)).unwrap();
                if let Some((group, _)) = alias.split_once('/') {
                    file.create_group(group).unwrap();
                }
                file.link_hard("meta", alias).unwrap();
            }
            let mut writer = amend(&path, None).unwrap();
            assert_eq!(writer.meta_aliases, vec![alias.to_string()]);
            let mut fields = Map::new();
            fields.insert("subject_id".into(), Value::String("pseudo-1".into()));
            writer.identity(fields).unwrap();
            writer.commit(true).unwrap();
            {
                let file = open_read(&path).unwrap();
                let root = file.as_group().unwrap();
                assert!(ops::object_id_by_name(&root, alias).is_some());
                assert_eq!(ops::object_id_by_name(&root, alias), ops::object_id_by_name(&root, "meta"), "{alias}");
            }
            let bytes = std::fs::read(&path).unwrap();
            assert!(!bytes.windows(10).any(|w| w == b"MRN-000042"), "{alias}: the old document is in the file");
            assert!(mismatched(&path).is_empty());
        }
    }

    /// Every file in `dir`.
    fn listed(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> =
            std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        names
    }

    /// `edit` applied to `path` behind the library's back.
    fn edited(path: &Path, edit: impl FnOnce(&hdf5::Group)) {
        let file = crate::h5::file::marked(|| hdf5::File::open_rw(path)).unwrap();
        edit(&file.as_group().unwrap());
    }

    /// An amendment refuses a file whose `content_id` no longer recomputes
    /// (E702) before it writes anything, naming the repair, and the repair
    /// re-attests it.  A commit recomputed the `content_id` from whatever
    /// was stored, so a no-op amendment of a file edited outside medh5 made
    /// it validate, with no record of the edit.
    #[test]
    fn s14_4_a_stale_root_is_refused_before_anything_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        sample(&path);
        edited(&path, |root| {
            attrs::write(&root.dataset("images/CT").unwrap(), "modality", &AttrValue::Str("MR".into())).unwrap()
        });
        let before = std::fs::read(&path).unwrap();
        let shown = repr_str(&path.to_string_lossy());
        let refused = amend(&path, None).unwrap_err();
        assert_eq!(refused.code(), Some("E702"), "{refused}");
        assert!(refused.to_string().contains("--rewrite-digests --reason"), "{refused}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(listed(dir.path()), vec!["s.medh5".to_string()]);
        // Read alone, as a tool amending many files checks each.
        let read = refuse_failed_digests(&open_read(&path).unwrap().as_group().unwrap(), &shown).unwrap_err();
        assert_eq!((read.code(), read.to_string()), (refused.code(), refused.to_string()));
        // Written elsewhere, the source is refused alike.
        let out = dir.path().join("out.medh5");
        assert_eq!(amend_into(&path, &out, None, &[]).unwrap_err().code(), Some("E702"));
        assert!(!out.exists());
        // The repair is the one amendment that re-attests it.
        let mut writer = amend_reattesting(&path, None).unwrap();
        writer.commit(true).unwrap();
        let file = open_read(&path).unwrap();
        let root = file.as_group().unwrap();
        assert_eq!(
            verify_root(&root, Some(&attr_name_map_of(&root).unwrap()), None, true).unwrap().content_id_ok(),
            Some(true)
        );
    }

    /// An amendment refuses a file declaring a `digest_algo` outside §2.1, or
    /// holding a digest that does not parse (E703), before it writes anything,
    /// whatever its commit would stamp.  The unknown algorithm gave way to
    /// `sha256`: a root declaring sha256 over digests made with another, or
    /// a relabelled root, validating clean with no record.
    #[test]
    fn s2_1_an_unknown_digest_algorithm_is_not_amended() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        sample(&path);
        edited(&path, |root| attrs::write(root, "digest_algo", &AttrValue::Str("blake3".into())).unwrap());
        let before = std::fs::read(&path).unwrap();
        for _digests in [false, true] {
            let refused = amend(&path, None).unwrap_err();
            assert_eq!(refused.code(), Some("E703"), "{refused}");
            assert!(refused.to_string().contains("'blake3'"), "{refused}");
            assert!(refused.to_string().contains("--rewrite-digests --reason"), "{refused}");
            assert_eq!(std::fs::read(&path).unwrap(), before);
        }
        // Every digest in an algorithm nobody can compute, under a known root.
        sample(&path);
        edited(&path, |root| {
            for (_, ds) in ops::datasets(root).unwrap() {
                if attrs::has(&ds, "digest") {
                    attrs::write(&ds, "digest", &AttrValue::Str("blake3:00ff".into())).unwrap();
                }
            }
        });
        let before = std::fs::read(&path).unwrap();
        let refused = amend(&path, None).unwrap_err();
        assert_eq!(refused.code(), Some("E703"), "{refused}");
        assert!(refused.to_string().contains("/images/CT"), "{refused}");
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(listed(dir.path()), vec!["s.medh5".to_string()]);
    }

    /// A commit refuses a source another writer replaced, or changed in
    /// place, after this amendment read it --- at the commit's start, before
    /// it reads the source again, and as the last thing before its rename ---
    /// and leaves the other writer's file as it is.  Both renamed over the
    /// file, and the first one's work was lost without an error.
    #[test]
    fn s14_4_a_source_replaced_before_the_rename_is_not_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        sample(&path);
        let shown = repr_str(&path.to_string_lossy());
        let refusal = format!("{shown} changed after this amendment read it; nothing was written: amend it again");

        // Another amendment committed between this one's read and its commit.
        let mut first = amend(&path, None).unwrap();
        let mut second = amend(&path, None).unwrap();
        second.extra("x_test", Value::String("second".into()));
        second.commit(true).unwrap();
        let committed = std::fs::read(&path).unwrap();
        first.extra("x_test", Value::String("first".into()));
        let err = first.commit(true).unwrap_err();
        assert!(matches!(err, Error::File(_)) && err.to_string() == refusal, "{err:?}");
        assert!(first.is_closed());
        assert_eq!(std::fs::read(&path).unwrap(), committed);
        assert_eq!(listed(dir.path()), vec!["s.medh5".to_string()]);

        // Replaced, or changed in place, after the commit's own checks.
        for in_place in [false, true] {
            let mut writer = amend(&path, None).unwrap();
            crate::h5::file::before_rename(Some(Box::new(move |target: &Path| {
                let other = if in_place { target.to_path_buf() } else { target.with_extension("other") };
                if !in_place {
                    std::fs::copy(target, &other).unwrap();
                }
                edited(&other, |root| attrs::write(root, "x_note", &AttrValue::Str("another writer".into())).unwrap());
                if !in_place {
                    std::fs::rename(&other, target).unwrap();
                }
            })));
            let err = writer.commit(true).unwrap_err();
            crate::h5::file::before_rename(None);
            assert!(matches!(err, Error::File(_)) && err.to_string() == refusal, "{err:?}");
            let file = open_read(&path).unwrap();
            assert_eq!(attrs::get_str(&file, "x_note").unwrap().as_deref(), Some("another writer"));
            drop(file);
            assert_eq!(listed(dir.path()), vec!["s.medh5".to_string()]);
            // The writer that refused is done; the file amends again.
            edited(&path, |root| attrs::delete(root, "x_note").unwrap());
        }
        amend(&path, None).unwrap().commit(true).unwrap();
    }
}
