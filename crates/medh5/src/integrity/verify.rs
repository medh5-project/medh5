//! Verification of digests, `content_id` and derived-index currency (§13).

use std::collections::HashMap;

use indexmap::IndexMap;
use serde_json::{json, Value};

use super::digest::{
    collect_digests, compute_content_id, dataset_digest, dataset_digest_inspected, digestible, group_digest, root_algo,
    AttrNameMap, Inspect, STREAM_BYTES,
};
use super::workers::{Inspector, Job};
use crate::array::NdArray;
use crate::digest::parse_canonical_digest;
use crate::h5::attrs::{self, AttrValue};
use crate::h5::ops::{self, LinkKind, Node, NodeKind};
use crate::json::repr_str;
use crate::{Error, Result};

/// Where every dataset is part of an object a `content_id` speaks for.
pub const ATTESTED_GROUPS: [&str; 4] = ["grids", "images", "annotations", "transforms"];

/// How a path through an attested group fails to be the path the root's
/// lines bind (§13.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Binding {
    /// A dataset at its own path that carries no digest: no line lists it.
    Undigested,
    /// A dataset carrying a digest that no line lists: its object is reached
    /// first through `index/`, which the root excludes, or only through a soft
    /// link (E702).
    Unlisted,
    /// A path that is not its object's own (E704): an alias sorting before
    /// it, a second link, a soft link, a group reached a second time, or an
    /// external link.  `first` is the object's own path --- for a digested
    /// dataset, the path its line names --- when the root reaches it by hard
    /// links at all.
    Elsewhere { first: Option<String> },
}

/// A path through an attested group whose bytes no line of the root binds to
/// it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unbound {
    /// Relative to the sample root.
    pub path: String,
    pub binding: Binding,
}

/// The groups whose every path a `content_id` speaks for: the four of 1.0,
/// and `clinical/` where the profile is declared (1.1 §8, E818); in a 1.0
/// file a group of that name is somebody's extension.
fn attested_groups(root: &hdf5::Group) -> Result<Vec<&'static str>> {
    let clinical =
        attrs::get_strs(root, "medh5_profiles")?.unwrap_or_default().iter().any(|p| p == crate::clinical::PROFILE);
    Ok(ATTESTED_GROUPS.into_iter().chain(clinical.then_some(crate::clinical::GROUP)).collect())
}

/// Every path through an attested group that is not the path the root's
/// lines bind to its bytes (§13.2), sorted.
///
/// A dataset has one line, at its first path: the walk of the root visits
/// each object once, through hard links, in byte order.  Covered is therefore
/// a property of a *path*: an attested path is covered only when it is the
/// path its object's line names.  Judging by identity alone passed a file in
/// which a second path to a covered object --- an alias sorting earlier, a
/// soft link --- held the attested name: relinking that name to another
/// covered object changed no line, so the root recomputed, the pin and
/// `verify` held, and a reader read other bytes.
///
/// Every link under an attested group is judged, soft links followed as
/// readers follow them.  A group is entered only at its own first path, so
/// a cycle or an alias of a group is one finding, not a walk without end.
pub fn unbound(root: &hdf5::Group) -> Result<Vec<Unbound>> {
    let mut first: HashMap<ops::ObjectId, String> = HashMap::new();
    ops::visit(root, &mut |path, node| {
        let id = match node {
            Node::Group(g) => ops::object_id_of(g),
            Node::Dataset(d) => ops::object_id_of(d),
        };
        if let Some(id) = id {
            first.insert(id, path.to_string());
        }
        Ok(true)
    })?;
    let lines = collect_digests(root, &["index"])?;
    let mut out = Vec::new();
    for name in attested_groups(root)? {
        if ops::link_kind(root, name).is_none() {
            continue;
        }
        judge(root, name, name, 0, &first, &lines, &mut out)?;
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// Judge the link `name` in `group`, reached at `path`, and what it leads to.
fn judge(
    group: &hdf5::Group,
    name: &str,
    path: &str,
    depth: usize,
    first: &HashMap<ops::ObjectId, String>,
    lines: &IndexMap<String, String>,
    out: &mut Vec<Unbound>,
) -> Result<()> {
    if depth > ops::MAX_DEPTH {
        return Err(Error::File(format!(
            "{} is nested more than {} groups deep; it is refused rather than followed",
            repr_str(path),
            ops::MAX_DEPTH
        )));
    }
    let elsewhere = |first: Option<&String>| Unbound {
        path: path.to_string(),
        binding: Binding::Elsewhere { first: first.cloned() },
    };
    match ops::link_kind(group, name) {
        Some(LinkKind::Hard | LinkKind::Soft) => {}
        // Bytes in another file, or behind a link HDF5 does not resolve
        // itself: nothing of this root.
        Some(_) => {
            out.push(elsewhere(None));
            return Ok(());
        }
        None => return Ok(()),
    }
    // A soft link to nothing names no object.
    let Some(id) = ops::object_id_by_name(group, name) else { return Ok(()) };
    let own = first.get(&id);
    match ops::node_kind(group, name) {
        Some(NodeKind::Group) => {
            if own.map(String::as_str) != Some(path) {
                out.push(elsewhere(own));
                return Ok(());
            }
            let child = group.group(name)?;
            for member in ops::members(&child)? {
                judge(&child, &member, &format!("{path}/{member}"), depth + 1, first, lines, out)?;
            }
        }
        Some(NodeKind::Dataset) => {
            if lines.contains_key(path) {
                return Ok(());
            }
            let binding = if own.map(String::as_str) == Some(path) {
                Binding::Undigested
            } else if own.is_some_and(|p| lines.contains_key(p)) {
                Binding::Elsewhere { first: own.cloned() }
            } else if attrs::has(&*group.dataset(name)?, "digest") {
                Binding::Unlisted
            } else {
                Binding::Elsewhere { first: own.cloned() }
            };
            out.push(Unbound { path: path.to_string(), binding });
        }
        _ => {}
    }
    Ok(())
}

/// The paths of [`unbound`]: what a declared `content_id` does not cover.
/// The root is a Merkle root over *stored* digests at the paths their lines
/// name (§13.2), so such a path is content no address covers there: a pin, a
/// cache entry and `verify` would all pass over it.
///
/// A dataset without a digest is not covered; nor is one whose object is
/// listed under no line --- reached first through `index/`, which the root
/// excludes, or only through a soft link --- nor one listed under another
/// path.
pub fn unattested(root: &hdf5::Group) -> Result<Vec<String>> {
    Ok(unbound(root)?.into_iter().map(|u| u.path).collect())
}

/// Outcome of a verification pass.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VerifyResult {
    pub checked: Vec<String>,
    pub mismatched: Vec<String>,
    pub undigested: Vec<String>,
    pub malformed: Vec<String>,
    pub content_id_declared: Option<String>,
    pub content_id_computed: Option<String>,
    pub stale_index: Vec<String>,
    /// Paths inside an object a declared `content_id` covers that no line of
    /// it binds there: [`unattested`].
    pub unattested: Vec<String>,
    /// Datasets carrying a digest their type cannot have (§13.1): data of
    /// variable length other than strings, or references.  Not compared,
    /// and not counted as a mismatch: the claim can be neither checked nor
    /// refuted.
    pub unverifiable: Vec<String>,
}

impl VerifyResult {
    /// Whether the root `content_id` matches, or `None` when unjudged.
    pub fn content_id_ok(&self) -> Option<bool> {
        match (&self.content_id_declared, &self.content_id_computed) {
            (Some(a), Some(b)) => Some(a == b),
            _ => None,
        }
    }

    pub fn ok(&self) -> bool {
        self.mismatched.is_empty()
            && self.malformed.is_empty()
            && self.unattested.is_empty()
            && self.content_id_ok() != Some(false)
    }

    pub fn summary(&self) -> Value {
        json!({
            "ok": self.ok(),
            "checked": self.checked.len(),
            "mismatched": self.mismatched,
            "undigested": self.undigested,
            "unattested": self.unattested,
            "malformed": self.malformed,
            "content_id_ok": self.content_id_ok(),
            "stale_index": self.stale_index,
            "unverifiable": self.unverifiable,
        })
    }
}

/// Recompute one dataset's digest and compare it with the stored value.
pub fn verify_object(root: &hdf5::Group, path: &str) -> Result<bool> {
    let ds = root.dataset(path)?;
    let Some(stored) = attrs::get_str(&ds, "digest")? else {
        return Ok(false);
    };
    let (algo, _) = parse_canonical_digest(&stored)?;
    Ok(dataset_digest(&ds, path, &algo)? == stored)
}

/// The paths among `paths` whose dataset under `root` carries a digest that
/// is malformed or does not match its bytes --- what [`verify_root`] reports
/// as malformed or mismatched.  A path naming no dataset, or one carrying no
/// digest, is not among them.
pub fn failing_digests(root: &hdf5::Group, paths: &[String]) -> Result<Vec<String>> {
    // Each path with its stored digest, and whether it is hashed: one that
    // does not parse fails as it is.
    // An attribute that cannot be read stops the pass where it is reached.
    let mut found = Vec::new();
    let mut jobs = Vec::new();
    let mut stopped = None;
    for path in paths {
        let Some(ds) = ops::resolve_dataset(root, path) else { continue };
        let stored = match attrs::get_str(&ds, "digest") {
            Ok(Some(stored)) => stored,
            Ok(None) => continue,
            Err(e) => {
                stopped = Some(e);
                break;
            }
        };
        let hashed = match parse_canonical_digest(&stored) {
            Ok((algo, _)) => {
                jobs.push(Job::new(ds, path, &algo));
                true
            }
            Err(_) => false,
        };
        found.push((path, stored, hashed));
    }
    let mut digests = super::workers::digests(jobs)?.into_iter();
    let mut out = Vec::new();
    for (path, stored, hashed) in found {
        let holds = hashed && digests.next().expect("a digest for every path hashed")? == stored;
        if !holds {
            out.push(path.clone());
        }
    }
    match stopped {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// Index entries whose `source_digest` no longer matches their source (§13.3).
///
/// The current digest is computed under the algorithm the entry names, as a
/// dataset's is (§13.1): a sha512 file's sha512 index is current.  An entry
/// whose `source_digest` is malformed, or names an algorithm outside §2.1, is
/// stale.  A stale entry is **not** a file error: readers ignore it and
/// rebuild.
pub fn stale_index_entries(root: &hdf5::Group) -> Result<Vec<String>> {
    let Some(index) = ops::child_group(root, "index") else {
        return Ok(Vec::new());
    };
    let mut stale = Vec::new();
    for name in ops::members(&index)? {
        let declared = match ops::node_kind(&index, &name) {
            Some(NodeKind::Group) => {
                let g = index.group(&name)?;
                attrs::get_str(&g, "source_digest")?
            }
            Some(NodeKind::Dataset) => {
                let d = index.dataset(&name)?;
                attrs::get_str(&d, "source_digest")?
            }
            _ => None,
        };
        let Some((algo, declared)) = declared.and_then(|d| parse_canonical_digest(&d).ok().map(|(algo, _)| (algo, d)))
        else {
            stale.push(name);
            continue;
        };
        let source_path = format!("annotations/{name}");
        let current = match ops::node_kind(root, &source_path) {
            None => {
                stale.push(name);
                continue;
            }
            Some(NodeKind::Group) => Some(group_digest(&root.group(&source_path)?, root, &algo)?),
            Some(NodeKind::Dataset) => {
                let d = root.dataset(&source_path)?;
                attrs::get_str(&d, "digest")?.or(Some(String::new()))
            }
            Some(_) => None,
        };
        if current.as_deref() != Some(declared.as_str()) {
            stale.push(name);
        }
    }
    Ok(stale)
}

/// Verify a sample root; `partial` restricts the pass to named objects.
pub fn verify_root(
    root: &hdf5::Group,
    attr_names: Option<&AttrNameMap>,
    partial: Option<&[String]>,
    check_content_id: bool,
) -> Result<VerifyResult> {
    verify(root, attr_names, partial, check_content_id, Inspection::None, None)
}

/// [`verify_root`], handing every block of every digested dataset it reads to
/// `inspect` with the dataset's path (relative to `root`): what a check of
/// stored values needs, at the cost of no second read.  The blocks are each
/// dataset in C order, about [`STREAM_BYTES`] at a time
/// ([`dataset_digest_inspected`]).
pub fn verify_root_inspecting(
    root: &hdf5::Group,
    attr_names: Option<&AttrNameMap>,
    partial: Option<&[String]>,
    check_content_id: bool,
    inspect: &mut dyn FnMut(&str, &NdArray) -> Result<()>,
) -> Result<VerifyResult> {
    verify(root, attr_names, partial, check_content_id, Inspection::Shared(inspect), None)
}

/// What a target of a verification is, before the digests are compared.
enum Step {
    /// Its type has no digest.
    Undigestible,
    /// Hashed here, or failed before hashing.
    Digest(Result<String>),
    /// Hashed on the workers: the next of their digests.
    Hashed,
}

/// What [`verify_root_inspecting`] hands each block it reads to, with the
/// dataset's path.
pub type InspectPath<'a> = &'a mut dyn FnMut(&str, &NdArray) -> Result<()>;

/// What a reporting pass could not read: the path relative to the root (`""`
/// for the root's `content_id`), and why.
pub(crate) type Unread = Vec<(String, Error)>;

/// [`verify_root_inspecting`] over the whole root, for the validator: an
/// object that cannot be read --- a dataset whose chunk no longer decodes, a
/// `content_id` that cannot be recomputed --- is returned beside the result
/// with what stopped it, and the pass goes on to the next.  Stopping at the
/// first unreadable chunk hid every other E701 and E702 in the file, and the
/// name of the dataset that stopped it.
///
/// `inspectors` holds a check per dataset path, each handed that dataset's
/// blocks on whichever worker hashes it; the one a dataset read here used is
/// taken out.
pub(crate) fn verify_root_reporting<'a>(
    root: &hdf5::Group,
    attr_names: Option<&AttrNameMap>,
    check_content_id: bool,
    inspectors: &mut HashMap<String, Inspector<'a>>,
) -> Result<(VerifyResult, Unread)> {
    let mut unread = Vec::new();
    let result = verify(root, attr_names, None, check_content_id, Inspection::PerPath(inspectors), Some(&mut unread))?;
    Ok((result, unread))
}

/// How a verification hands the blocks it reads to a check.
enum Inspection<'i, 'a> {
    None,
    /// One check for every dataset, called on this thread in dataset order:
    /// the datasets are hashed one after another.
    Shared(InspectPath<'i>),
    /// A check per path, on the workers.
    PerPath(&'i mut HashMap<String, Inspector<'a>>),
}

/// `step`, or --- when `unread` collects failures --- its value, a failure
/// recorded under `path` as `None`.
fn tolerated<T>(unread: &mut Option<&mut Unread>, path: &str, step: Result<T>) -> Result<Option<T>> {
    match (step, unread) {
        (Ok(v), _) => Ok(Some(v)),
        (Err(e), Some(unread)) => {
            unread.push((path.to_string(), e));
            Ok(None)
        }
        (Err(e), None) => Err(e),
    }
}

fn verify(
    root: &hdf5::Group,
    attr_names: Option<&AttrNameMap>,
    partial: Option<&[String]>,
    check_content_id: bool,
    mut inspection: Inspection<'_, '_>,
    mut unread: Option<&mut Unread>,
) -> Result<VerifyResult> {
    stage!("verify");
    let stored = collect_digests(root, &["index"])?;
    let targets: Vec<String> = match partial {
        Some(p) => p.to_vec(),
        None => {
            let mut keys: Vec<String> = stored.keys().cloned().collect();
            keys.sort();
            keys
        }
    };
    let mut result = VerifyResult::default();
    // Each target's stored digest and the dataset to hash, or why there is
    // none; then the digests, on the workers but with a shared check.
    let mut found = Vec::new();
    let mut jobs = Vec::new();
    for path in targets {
        let Some(value) = stored.get(&path) else {
            found.push((path, None));
            continue;
        };
        let Ok((algo, _)) = parse_canonical_digest(value) else {
            found.push((path, None));
            continue;
        };
        let ds = (|| -> Result<Option<hdf5::Dataset>> {
            let ds = root.dataset(&path)?;
            Ok(digestible(&ds)?.then_some(ds))
        })();
        let step = match ds {
            Ok(Some(ds)) => match &mut inspection {
                Inspection::Shared(inspect) => {
                    let mut each = |block: &NdArray| inspect(&path, block);
                    let each: Inspect<'_> = &mut each;
                    Step::Digest(dataset_digest_inspected(&ds, &path, &algo, STREAM_BYTES, Some(each)))
                }
                other => {
                    let mut job = Job::new(ds, &path, &algo);
                    if let Inspection::PerPath(inspectors) = other {
                        job.inspect = inspectors.remove(&path);
                    }
                    jobs.push(job);
                    Step::Hashed
                }
            },
            Ok(None) => Step::Undigestible,
            Err(e) => Step::Digest(Err(e)),
        };
        found.push((path, Some((value, step))));
    }
    let mut hashed = super::workers::digests(jobs)?.into_iter();
    for (path, step) in found {
        let Some((value, step)) = step else {
            result.malformed.push(path);
            continue;
        };
        // One dataset that cannot be read is that dataset's failure, not the
        // pass's, where the caller collects them.
        let digest = match step {
            Step::Undigestible => Ok(None),
            Step::Digest(digest) => digest.map(Some),
            Step::Hashed => hashed.next().expect("a digest for every dataset hashed").map(Some),
        };
        // Named, where nothing collects it: HDF5's message does not say which
        // dataset's chunk failed to decode.
        let digest = match (digest, unread.is_some()) {
            (Err(e), false) => {
                let message = format!("{path}: {}", e.message());
                Err(super::restated(e, message))
            }
            (digest, _) => digest,
        };
        match tolerated(&mut unread, &path, digest)? {
            None => {}
            Some(None) => result.unverifiable.push(path),
            Some(Some(computed)) => {
                result.checked.push(path.clone());
                if &computed != value {
                    result.mismatched.push(path);
                }
            }
        }
    }
    drop(hashed);
    let mut undigested = Vec::new();
    ops::visit(root, &mut |name, node| {
        if let Node::Dataset(ds) = node {
            if name != "meta" && !name.starts_with("index/") && !attrs::has(ds, "digest") {
                undigested.push(name.to_string());
            }
        }
        Ok(true)
    })?;
    undigested.sort();
    let declared = match attrs::read(root, "content_id")? {
        Some(AttrValue::Str(s)) => Some(s),
        Some(other) => Some(attrs::stringify_value(&other)),
        None => None,
    };
    if check_content_id && partial.is_none() {
        if let Some(names) = attr_names {
            let computed = root_algo(root).and_then(|algo| compute_content_id(root, names, &algo, Some(&stored)));
            result.content_id_computed = tolerated(&mut unread, "", computed)?;
        }
    }
    if declared.is_some() && partial.is_none() {
        result.unattested = tolerated(&mut unread, "", unattested(root))?.unwrap_or_default();
    }
    result.undigested = undigested;
    result.content_id_declared = declared;
    result.stale_index = tolerated(&mut unread, "index", stale_index_entries(root))?.unwrap_or_default();
    Ok(result)
}

/// Paths at which two object trees differ, byte for byte: structure,
/// attributes and *stored* chunk bytes.  Empty means a pure copy.
///
/// The trees are compared as graphs, the way a copy keeps them.  Each pair of
/// objects is compared once, so a hard link back to an ancestor is a cycle to
/// stop at rather than a recursion without end (a self-cycle overflowed the
/// stack); a second link to an object must pair with
/// a second link to its counterpart, so an alias and a duplicate of the same
/// bytes still differ; and a soft link is compared by the path it holds,
/// relative to its tree's root, not followed.
pub fn subtrees_identical(a: &hdf5::Group, b: &hdf5::Group) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut pairs = Pairs { roots: (a.name(), b.name()), ..Pairs::default() };
    pairs.pair(ops::object_id_of(a), ops::object_id_of(b));
    compare_groups(a, b, "", 0, &mut pairs, &mut out)?;
    out.sort();
    Ok(out)
}

/// The objects of two trees paired so far, both ways.
#[derive(Default)]
struct Pairs {
    a_to_b: std::collections::HashMap<ops::ObjectId, ops::ObjectId>,
    b_to_a: std::collections::HashMap<ops::ObjectId, ops::ObjectId>,
    /// The names of the two roots, which soft-link targets are relative to.
    roots: (String, String),
}

enum Paired {
    /// Not met before: compare them.
    New,
    /// Met before, as each other's counterparts: compared already.
    Again,
    /// One was met before with another counterpart: the link structure differs.
    Mismatched,
}

impl Pairs {
    fn pair(&mut self, a: Option<ops::ObjectId>, b: Option<ops::ObjectId>) -> Paired {
        let (Some(a), Some(b)) = (a, b) else { return Paired::New };
        match (self.a_to_b.get(&a), self.b_to_a.get(&b)) {
            (None, None) => {
                self.a_to_b.insert(a, b);
                self.b_to_a.insert(b, a);
                Paired::New
            }
            (Some(x), Some(y)) if *x == b && *y == a => Paired::Again,
            _ => Paired::Mismatched,
        }
    }
}

/// A soft link's target as a path inside the tree rooted at `root`: an
/// absolute target under the root loses the root's prefix, so a sample's
/// links compare equal in a file of their own and in a collection.
fn relative_target(root: &str, target: &str) -> String {
    let root = root.trim_end_matches('/');
    if root.is_empty() || !target.starts_with('/') {
        return target.to_string();
    }
    if target == root {
        return "/".to_string();
    }
    match target.strip_prefix(root) {
        Some(rest) if rest.starts_with('/') => rest.to_string(),
        _ => target.to_string(),
    }
}

fn compare_attrs(a: &hdf5::Location, b: &hdf5::Location, prefix: &str, out: &mut Vec<String>) -> Result<()> {
    let label = if prefix.is_empty() { "/" } else { prefix };
    let names_a = attrs::names(a)?;
    let names_b = attrs::names(b)?;
    let mut only: Vec<&String> = names_a
        .iter()
        .filter(|n| !names_b.contains(n))
        .chain(names_b.iter().filter(|n| !names_a.contains(n)))
        .collect();
    only.sort();
    for key in only {
        out.push(format!("{label}@{key}: present in only one tree"));
    }
    let mut both: Vec<&String> = names_a.iter().filter(|n| names_b.contains(n)).collect();
    both.sort();
    for key in both {
        if attrs::read(a, key)? != attrs::read(b, key)? {
            out.push(format!("{label}@{key}: differs"));
        }
    }
    Ok(())
}

fn compare_groups(
    a: &hdf5::Group,
    b: &hdf5::Group,
    prefix: &str,
    depth: usize,
    pairs: &mut Pairs,
    out: &mut Vec<String>,
) -> Result<()> {
    if depth > ops::MAX_DEPTH {
        return Err(Error::File(format!(
            "{} is nested more than {} groups deep; it is refused rather than followed",
            repr_str(&a.name()),
            ops::MAX_DEPTH
        )));
    }
    compare_attrs(a, b, prefix, out)?;
    let names_a = ops::members(a)?;
    let names_b = ops::members(b)?;
    for name in names_a.iter().filter(|n| !names_b.contains(n)).chain(names_b.iter().filter(|n| !names_a.contains(n))) {
        out.push(format!("{prefix}/{name}: present in only one tree"));
    }
    for name in names_a.iter().filter(|n| names_b.contains(n)) {
        let path = format!("{prefix}/{name}");
        let (link_a, link_b) = (ops::link_kind(a, name), ops::link_kind(b, name));
        if link_a == Some(LinkKind::Soft) || link_b == Some(LinkKind::Soft) {
            if link_a != link_b {
                out.push(format!("{path}: {} vs {}", link_name(link_a), link_name(link_b)));
            } else if relative_target(&pairs.roots.0, &ops::soft_link_target(a, name)?)
                != relative_target(&pairs.roots.1, &ops::soft_link_target(b, name)?)
            {
                out.push(format!("{path}: soft link targets differ"));
            }
            continue;
        }
        match pairs.pair(ops::object_id_by_name(a, name), ops::object_id_by_name(b, name)) {
            Paired::Again => continue,
            Paired::Mismatched => {
                out.push(format!("{path}: links to objects the other tree links to differently"));
                continue;
            }
            Paired::New => {}
        }
        match (ops::node_kind(a, name), ops::node_kind(b, name)) {
            (Some(NodeKind::Group), Some(NodeKind::Group)) => {
                compare_groups(&a.group(name)?, &b.group(name)?, &path, depth + 1, pairs, out)?
            }
            (Some(NodeKind::Dataset), Some(NodeKind::Dataset)) => {
                let (da, db) = (a.dataset(name)?, b.dataset(name)?);
                compare_attrs(&da, &db, &path, out)?;
                if da.shape() != db.shape() || crate::h5::data::kind(&da)? != crate::h5::data::kind(&db)? {
                    out.push(format!("{path}: shape/dtype differ"));
                } else if !data_equal(&da, &db)? {
                    out.push(format!("{path}: stored bytes differ"));
                }
            }
            (ka, kb) => out.push(format!("{path}: {} vs {}", kind_name(ka), kind_name(kb))),
        }
    }
    Ok(())
}

fn link_name(kind: Option<LinkKind>) -> &'static str {
    match kind {
        Some(LinkKind::Hard) => "hard link",
        Some(LinkKind::Soft) => "soft link",
        Some(LinkKind::External) => "external link",
        _ => "other link",
    }
}

fn kind_name(kind: Option<NodeKind>) -> &'static str {
    match kind {
        Some(NodeKind::Group) => "Group",
        Some(NodeKind::Dataset) => "Dataset",
        _ => "Other",
    }
}

/// Every stored chunk of a dataset, exactly as it sits on disk.
pub fn raw_chunks(ds: &hdf5::Dataset) -> Result<Vec<Vec<u8>>> {
    if crate::h5::data::chunks(ds).is_none() {
        return Ok(Vec::new());
    }
    ops::stored_chunks(ds)?.iter().map(|offset| Ok(ops::read_raw_chunk(ds, offset)?.1)).collect()
}

fn data_equal(a: &hdf5::Dataset, b: &hdf5::Dataset) -> Result<bool> {
    let chunks_a = raw_chunks(a)?;
    let chunks_b = raw_chunks(b)?;
    if !chunks_a.is_empty() || !chunks_b.is_empty() {
        return Ok(chunks_a == chunks_b);
    }
    if crate::h5::data::chunks(a) != crate::h5::data::chunks(b) {
        return Ok(false);
    }
    match crate::h5::data::kind(a)? {
        // As stored: decoded as text, different invalid bytes compared equal.
        crate::h5::data::Kind::Strings => {
            Ok(crate::h5::data::read_stored_strings(a)? == crate::h5::data::read_stored_strings(b)?)
        }
        crate::h5::data::Kind::Other(_) => {
            match (crate::h5::data::stored_element_size(a)?, crate::h5::data::stored_element_size(b)?) {
                (Some(size), Some(other)) if size == other => {
                    let rows = a.shape().first().copied().unwrap_or(1);
                    Ok(crate::h5::data::read_stored_rows(a, size, 0, rows)?
                        == crate::h5::data::read_stored_rows(b, size, 0, rows)?)
                }
                (Some(_), Some(_)) => Ok(false),
                _ => Err(Error::Type(format!(
                    "{} holds data of variable length, which has no bytes of its own to compare",
                    a.name()
                ))),
            }
        }
        _ => Ok(crate::h5::data::read(a)? == crate::h5::data::read(b)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file whose root holds `images/CT` and the given extra links.
    fn tree(path: &std::path::Path, extra: impl FnOnce(&hdf5::File)) -> hdf5::File {
        let file = crate::h5::file::create_truncate(path).unwrap();
        let images = file.create_group("images").unwrap();
        images.new_dataset::<u8>().shape([4]).create("CT").unwrap().write(&[1u8, 2, 3, 4]).unwrap();
        extra(&file);
        file
    }

    /// A hard link back to an ancestor is a cycle, compared once ---
    /// the comparison of a self-cycle with itself recursed until the stack
    /// overflowed.
    #[test]
    fn s14_4_a_cycle_is_compared_once() {
        let dir = tempfile::tempdir().unwrap();
        let cyclic = |file: &hdf5::File| {
            file.link_hard("/", "loop").unwrap();
            file.group("images").unwrap().link_hard("/images", "self").unwrap();
        };
        let a = tree(&dir.path().join("a.h5"), cyclic);
        let b = tree(&dir.path().join("b.h5"), cyclic);
        assert!(subtrees_identical(&a.as_group().unwrap(), &a.as_group().unwrap()).unwrap().is_empty());
        assert!(subtrees_identical(&a.as_group().unwrap(), &b.as_group().unwrap()).unwrap().is_empty());
        // A cycle against a tree without one differs, and says where.
        let c = tree(&dir.path().join("c.h5"), |file| file.link_hard("/", "loop").unwrap());
        let found = subtrees_identical(&a.as_group().unwrap(), &c.as_group().unwrap()).unwrap();
        assert!(found.iter().any(|d| d.starts_with("/images/self")), "{found:?}");
    }

    /// An alias and a duplicate of the same bytes are different graphs.
    #[test]
    fn s13_2_an_alias_is_not_a_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        let alias = tree(&dir.path().join("alias.h5"), |file| file.link_hard("/images/CT", "zzz").unwrap());
        let copy = tree(&dir.path().join("copy.h5"), |file| {
            file.new_dataset::<u8>().shape([4]).create("zzz").unwrap().write(&[1u8, 2, 3, 4]).unwrap();
        });
        let found = subtrees_identical(&alias.as_group().unwrap(), &copy.as_group().unwrap()).unwrap();
        assert_eq!(found, vec!["/zzz: links to objects the other tree links to differently".to_string()]);
        let again = tree(&dir.path().join("again.h5"), |file| file.link_hard("/images/CT", "zzz").unwrap());
        assert!(subtrees_identical(&alias.as_group().unwrap(), &again.as_group().unwrap()).unwrap().is_empty());
    }

    /// A soft link is compared by the path it holds, whether it closes a
    /// cycle, dangles or names a dataset, and never followed.
    #[test]
    fn s2_2_soft_links_compare_by_target() {
        let dir = tempfile::tempdir().unwrap();
        let soft = |target: &'static str| {
            move |file: &hdf5::File| {
                file.link_soft("/", "up").unwrap();
                file.link_soft("/nowhere", "dangling").unwrap();
                file.link_soft(target, "ct").unwrap();
            }
        };
        let a = tree(&dir.path().join("a.h5"), soft("/images/CT"));
        let b = tree(&dir.path().join("b.h5"), soft("/images/CT"));
        assert!(subtrees_identical(&a.as_group().unwrap(), &b.as_group().unwrap()).unwrap().is_empty());
        let c = tree(&dir.path().join("c.h5"), soft("images/CT"));
        let found = subtrees_identical(&a.as_group().unwrap(), &c.as_group().unwrap()).unwrap();
        assert_eq!(found, vec!["/ct: soft link targets differ".to_string()]);
        // A soft link against a hard link to the same object is a difference.
        let d = tree(&dir.path().join("d.h5"), |file| {
            file.link_soft("/", "up").unwrap();
            file.link_soft("/nowhere", "dangling").unwrap();
            file.link_hard("/images/CT", "ct").unwrap();
        });
        let found = subtrees_identical(&a.as_group().unwrap(), &d.as_group().unwrap()).unwrap();
        assert_eq!(found, vec!["/ct: soft link vs hard link".to_string()]);
        // Under another root, a target under that root is the same path.
        assert_eq!(relative_target("/samples/k", "/samples/k/images/CT"), "/images/CT");
        assert_eq!(relative_target("/samples/k", "/samples/k"), "/");
        assert_eq!(relative_target("/samples/k", "/samples/kk/x"), "/samples/kk/x");
        assert_eq!(relative_target("/", "/images/CT"), "/images/CT");
    }

    /// [`tree`] with a second image, every dataset digested where the walk
    /// first reaches it, after `extra` has linked what it links.
    fn stamped(path: &std::path::Path, extra: impl FnOnce(&hdf5::File)) -> hdf5::File {
        let file = tree(path, |file| {
            let images = file.group("images").unwrap();
            images.new_dataset::<u8>().shape([4]).create("MR").unwrap().write(&[5u8, 6, 7, 8]).unwrap();
            extra(file);
        });
        super::super::digest::stamp_digests(&file.as_group().unwrap(), "sha256", &["index"], false).unwrap();
        file
    }

    fn bindings(file: &hdf5::File) -> Vec<(String, Binding)> {
        unbound(&file.as_group().unwrap()).unwrap().into_iter().map(|u| (u.path, u.binding)).collect()
    }

    /// A line binds bytes to the path it names.  An alias sorting before
    /// `images/` holds the image's line, so `images/CT` is a name no line
    /// binds --- it could be relinked to other covered bytes, changing no line.
    #[test]
    fn s13_2_an_earlier_alias_leaves_the_name_unbound() {
        let dir = tempfile::tempdir().unwrap();
        let file = stamped(&dir.path().join("a.h5"), |file| file.link_hard("/images/CT", "aaa").unwrap());
        assert_eq!(bindings(&file), vec![("images/CT".into(), Binding::Elsewhere { first: Some("aaa".into()) })]);
        // A later alias, outside the attested groups, leaves every name bound.
        let file = stamped(&dir.path().join("z.h5"), |file| file.link_hard("/images/CT", "zzz").unwrap());
        assert_eq!(bindings(&file), vec![]);
    }

    /// Inside an attested group every link is judged --- a second hard
    /// link, a soft link, an external link --- and an undigested dataset at
    /// its own path is told apart from all of them.
    #[test]
    fn s13_2_every_link_through_an_attested_group_is_judged() {
        let dir = tempfile::tempdir().unwrap();
        let other = dir.path().join("other.h5");
        tree(&other, |_| {});
        let file = stamped(&dir.path().join("links.h5"), |file| {
            let images = file.group("images").unwrap();
            images.link_hard("/images/CT", "CT_second").unwrap();
            images.link_soft("/images/MR", "MR_soft").unwrap();
            images.link_soft("/nowhere", "dangling").unwrap();
            images.link_external(&other.to_string_lossy(), "/images/CT", "far").unwrap();
        });
        file.group("images").unwrap().new_dataset::<u8>().shape([1]).create("unstamped").unwrap();
        assert_eq!(
            bindings(&file),
            vec![
                ("images/CT_second".into(), Binding::Elsewhere { first: Some("images/CT".into()) }),
                ("images/MR_soft".into(), Binding::Elsewhere { first: Some("images/MR".into()) }),
                ("images/far".into(), Binding::Elsewhere { first: None }),
                ("images/unstamped".into(), Binding::Undigested),
            ]
        );
    }

    /// A group reached a second time --- a cycle, or an alias of a group
    /// --- is one finding, and is not walked again.
    #[test]
    fn s13_2_a_group_reached_twice_is_one_finding() {
        let dir = tempfile::tempdir().unwrap();
        let file = stamped(&dir.path().join("cycle.h5"), |file| {
            file.group("images").unwrap().link_hard("/images", "self").unwrap();
            file.create_group("grids").unwrap().link_soft("/images", "soft").unwrap();
        });
        assert_eq!(
            bindings(&file),
            vec![
                ("grids/soft".into(), Binding::Elsewhere { first: Some("images".into()) }),
                ("images/self".into(), Binding::Elsewhere { first: Some("images".into()) }),
            ]
        );
    }

    /// A digest on an object no line lists --- reached first through
    /// `index/`, which sorts before `transforms/` --- stays E702: a claim the
    /// root does not make.
    #[test]
    fn s13_2_a_digest_no_line_lists_is_unlisted() {
        let dir = tempfile::tempdir().unwrap();
        let file = stamped(&dir.path().join("index.h5"), |file| {
            let t = file.create_group("transforms").unwrap().create_group("t").unwrap();
            t.new_dataset::<f64>().shape([2]).create("matrix").unwrap().write(&[1.0, 2.0]).unwrap();
        });
        file.create_group("index").unwrap().link_hard("/transforms/t/matrix", "alias").unwrap();
        assert_eq!(bindings(&file), vec![("transforms/t/matrix".into(), Binding::Unlisted)]);
    }

    /// An index entry is read under the algorithm its `source_digest` names,
    /// whatever the file's: a sha256 and a sha512 entry over the same
    /// annotation are both current, and one naming an algorithm outside §2.1,
    /// or spelt otherwise than §13.1 spells a digest, is stale.
    #[test]
    fn s13_3_a_source_digest_is_read_under_the_algorithm_it_names() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("index.h5")).unwrap();
        let root = file.as_group().unwrap();
        let annotations = file.create_group("annotations").unwrap();
        let index = file.create_group("index").unwrap();
        let entry = |name: &str, pin: &dyn Fn(&hdf5::Group) -> String| {
            let source = annotations.create_group(name).unwrap();
            source.new_dataset::<u8>().shape([4]).create("data").unwrap().write(&[0u8, 1, 1, 0]).unwrap();
            let g = index.create_group(name).unwrap();
            attrs::write(&g, "source_digest", &AttrValue::Str(pin(&source))).unwrap();
        };
        let under = |g: &hdf5::Group, algo: &str| group_digest(g, &root, algo).unwrap();
        let hex = |g: &hdf5::Group| under(g, "sha256").split_once(':').unwrap().1.to_string();
        entry("a256", &|g| under(g, "sha256"));
        entry("a512", &|g| under(g, "sha512"));
        entry("md5", &|g| format!("md5:{}", &hex(g)[..32]));
        entry("upper", &|g| format!("sha256:{}", hex(g).to_uppercase()));
        entry("wrong", &|g| format!("sha512:{0}{0}", hex(g)));
        assert_eq!(stale_index_entries(&root).unwrap(), ["md5", "upper", "wrong"]);
    }
}
