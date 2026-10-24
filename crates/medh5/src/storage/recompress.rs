//! Re-encoding a file's bulk data under a different codec profile (§14.2).
//!
//! Safe at any point because **digests cover decompressed content** (§13.1):
//! recompression changes every stored byte and no digest, so `content_id`
//! before and after are identical.  The output is read back and verified
//! against the digests it carries.  Chunking is preserved unless `rechunk`
//! asks for the writer's own derivation.

use std::path::Path;

use indexmap::IndexMap;
use serde_json::{json, Value};

use super::chunking::{field_chunks, fit_chunks, grid_chunks};
use super::codecs::{
    builtin_only, dataset_layout, describe_filters, needs_plugin, profile_names, resolve_profile, CodecProfile, Role,
};
use crate::array::{Index, Slice};
use crate::geometry::grid::{read_grids, Grid};
use crate::h5::attrs;
use crate::h5::chunks::Chunks;
use crate::h5::data::{self, Kind};
use crate::h5::file::{atomic_rewrite, open_read};
use crate::h5::graph::{copy_stored, CopyHooks, GraphCopy};
use crate::h5::{ops, relative_name};
use crate::integrity::verify_root;
use crate::json::repr_str;
use crate::sample::reader::{attr_name_map_of, require_major};
use crate::validate::SAMPLES_GROUP;
use crate::{Error, Result};

/// What one file's re-encoding did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecompressResult {
    pub path: String,
    pub profile: String,
    pub datasets: usize,
    pub bytes_before: u64,
    pub bytes_after: u64,
    pub content_id: Option<String>,
    pub content_id_preserved: bool,
    /// Whether the *output* verifies: every object digest, and the root.
    pub verified: bool,
    pub mismatched: Vec<String>,
    pub unattested: Vec<String>,
    /// `(path, codec before, codec after)` for each dataset re-encoded.
    pub changed: Vec<(String, String, String)>,
    /// Datasets copied as stored rather than re-encoded: their type is one a
    /// rebuild from a plain number type would lose --- an enumeration's
    /// names, a committed (named) type, another byte order.
    pub kept: Vec<String>,
    /// What the check of the output could not read, and why.  The output has
    /// replaced the input by then, so a failure there is reported here, with
    /// `verified` false, rather than raised as if nothing had been written.
    pub unverified: Vec<String>,
}

impl RecompressResult {
    pub fn ratio(&self) -> f64 {
        if self.bytes_before == 0 {
            1.0
        } else {
            self.bytes_after as f64 / self.bytes_before as f64
        }
    }

    pub fn ok(&self) -> bool {
        self.verified && self.content_id_preserved
    }

    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "profile": self.profile,
            "datasets": self.datasets,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "ratio": crate::json::num(self.ratio()),
            "content_id": self.content_id,
            "content_id_preserved": self.content_id_preserved,
            "verified": self.verified,
            "mismatched": self.mismatched,
            "unattested": self.unattested,
            "ok": self.ok(),
            "changed": self.changed.iter().map(|(a, b, c)| json!([a, b, c])).collect::<Vec<_>>(),
            "kept": self.kept,
            "unverified": self.unverified,
        })
    }

    /// Python's `str()`.
    pub fn line(&self) -> String {
        format!(
            "{}: {}, {} datasets, {} -> {} bytes ({:.2}\u{d7})",
            self.path,
            self.profile,
            self.datasets,
            self.bytes_before,
            self.bytes_after,
            self.ratio()
        )
    }
}

/// Rewrite `path` with every bulk dataset re-encoded under `profile`.
///
/// Copy-on-write like every other mutation (§14.4).
pub fn recompress(path: &Path, profile: &str, out: Option<&Path>, rechunk: bool) -> Result<RecompressResult> {
    check_profile(profile)?;
    let codec = resolve_profile(Some(profile))?;
    refuse_projection(path)?;
    let target = out.unwrap_or(path);
    let mut result =
        RecompressResult { path: target.to_string_lossy().into_owned(), profile: profile.into(), ..Default::default() };
    result.bytes_before = std::fs::metadata(path)?.len();
    let threads = crate::parallel::threads()?;
    let before = atomic_rewrite(path, Some(target), |src, dst| {
        stage!("recompress.rewrite");
        require_major(src, path)?;
        let before = attrs::get_str(src, "content_id")?;
        let src_root = src.as_group()?;
        ops::refuse_references(&src_root, "recompression")?;
        let dst_root = dst.as_group()?;
        let mut hooks =
            Recode { codec: &codec, rechunk, threads, result: &mut result, layouts: Vec::new(), roots: Vec::new() };
        GraphCopy::new(&src_root, &dst_root).members(&src_root, &dst_root, &[], &mut hooks)?;
        for key in attrs::names(&src_root)? {
            attrs::copy_raw(&src_root, &dst_root, &key)?;
        }
        Ok(before)
    })?;
    result.bytes_after = std::fs::metadata(target)?.len();
    // The output has replaced the input: from here on, what cannot be read is
    // a finding of the result, never an error that reads as "nothing changed".
    let (after, all_ok, ids_ok) = match verify_output(target, &mut result) {
        Ok(found) => found,
        Err(e) => {
            result.unverified.push(format!("{}: {e}", target.to_string_lossy()));
            (None, false, false)
        }
    };
    result.content_id = after.clone();
    result.verified = all_ok;
    result.content_id_preserved = before == after && ids_ok;
    Ok(result)
}

/// Refuse a profile name that is not one of the four (§14.2).
fn check_profile(profile: &str) -> Result<()> {
    if profile_names().contains(&profile) {
        return Ok(());
    }
    let mut names = profile_names();
    names.sort_unstable();
    Err(Error::invalid(format!(
        "unknown codec profile {}; expected one of {}",
        repr_str(profile),
        crate::json::repr_list(&names)
    )))
}

/// Verify every sample root of the written file: its `content_id`, whether
/// every root verifies, and whether every root's `content_id` recomputes.  A
/// root that cannot be read is recorded in `unverified`.
fn verify_output(target: &Path, result: &mut RecompressResult) -> Result<(Option<String>, bool, bool)> {
    stage!("recompress.verify");
    let check = open_read(target)?;
    let after = attrs::get_str(&check, "content_id")?;
    let (mut all_ok, mut ids_ok) = (true, true);
    for (prefix, root) in sample_roots(&check.as_group()?)? {
        let verification = match attr_name_map_of(&root).and_then(|names| verify_root(&root, Some(&names), None, true))
        {
            Ok(v) => v,
            Err(e) => {
                result.unverified.push(format!("{}: {e}", if prefix.is_empty() { "/" } else { prefix.as_str() }));
                (all_ok, ids_ok) = (false, false);
                continue;
            }
        };
        for name in verification.mismatched.iter().chain(&verification.malformed) {
            result.mismatched.push(format!("{prefix}{name}"));
        }
        for name in &verification.unattested {
            result.unattested.push(format!("{prefix}{name}"));
        }
        all_ok &= verification.ok();
        ids_ok &= verification.content_id_ok() != Some(false);
    }
    Ok((after, all_ok, ids_ok))
}

/// Refuse a file holding a version this engine reads only as a projection:
/// recompression replaces the file and then verifies it under the integrity
/// contract this engine implements, which a later minor may have extended
/// (1.1 §2.1) --- so the check could not vouch for what it rewrote.
fn refuse_projection(path: &Path) -> Result<()> {
    let source = open_read(path)?;
    require_major(&source, path)?;
    for (prefix, root) in sample_roots(&source.as_group()?)? {
        let version = attrs::get_str(&root, "medh5_version")?.unwrap_or_default();
        if crate::version::is_projection(&version) {
            return Err(Error::Version(format!(
                "{}{} is MEDH5 {version}; this engine implements up to {} and cannot verify a recompression of it \
                 (1.1 §2.1)",
                repr_str(&path.to_string_lossy()),
                if prefix.is_empty() {
                    String::new()
                } else {
                    format!(" member {}", repr_str(prefix.trim_end_matches('/')))
                },
                crate::FORMAT_VERSION
            )));
        }
    }
    Ok(())
}

/// `(path prefix, sample root)` for a sample, or each collection member.
pub fn sample_roots(root: &hdf5::Group) -> Result<Vec<(String, hdf5::Group)>> {
    let kind = attrs::get_str(root, "medh5_kind")?.unwrap_or_else(|| "sample".into());
    if kind != "collection" {
        return Ok(vec![(String::new(), root.clone())]);
    }
    let members = root.group(SAMPLES_GROUP)?;
    let mut out = Vec::new();
    for key in ops::members(&members)? {
        if let Some(g) = ops::child_group(&members, &key) {
            out.push((format!("{SAMPLES_GROUP}/{key}/"), g));
        }
    }
    Ok(out)
}

/// One sample root's grids, and the chunks the writer derives from them.
struct Layout {
    root: hdf5::Group,
    name: String,
    grids: Option<IndexMap<String, Grid>>,
}

impl Layout {
    fn new(root: &hdf5::Group) -> Layout {
        Layout { root: root.clone(), name: format!("{}/", root.name().trim_end_matches('/')), grids: None }
    }

    fn grid(&mut self, grid_id: Option<String>) -> Option<Grid> {
        if self.grids.is_none() {
            self.grids = Some(read_grids(&self.root).map(|g| g.into_iter().collect()).unwrap_or_default());
        }
        grid_id.and_then(|g| self.grids.as_ref().and_then(|m| m.get(&g).cloned()))
    }

    fn chunks_for(&mut self, ds: &hdf5::Dataset, parent: &hdf5::Group) -> Result<Option<Vec<usize>>> {
        let name = ds.name();
        let rest = name.strip_prefix(&self.name).unwrap_or(&name).to_string();
        let parts: Vec<&str> = rest.split('/').collect();
        let shape = ds.shape();
        let itemsize = match data::kind(ds)? {
            Kind::Numeric(d) => d.itemsize(),
            _ => return Ok(None),
        };
        let section = parts[0];
        if section == "images" && parts.len() == 2 {
            return Ok(match self.grid(attrs::get_str(ds, "grid")?) {
                None => None,
                Some(g) => fit_chunks(&grid_chunks(&g, itemsize, 0)?, &shape),
            });
        }
        if section == "images" && parts.len() == 3 && parts[2].chars().all(|c| c.is_ascii_digit()) {
            let levels = attrs::get_strs(parent, "grid_levels")?;
            let level: usize = parts[2].parse().unwrap_or(usize::MAX);
            let Some(levels) = levels.filter(|l| level < l.len()) else { return Ok(None) };
            return Ok(match self.grid(Some(levels[level].clone())) {
                None => None,
                Some(g) => fit_chunks(&grid_chunks(&g, itemsize, 0)?, &shape),
            });
        }
        if section == "annotations" && parts.len() == 3 && parts[2] == "data" {
            let Some(g) = self.grid(attrs::get_str(parent, "grid")?) else { return Ok(None) };
            if shape.len() < g.n_spatial() {
                return Ok(None);
            }
            return Ok(fit_chunks(&grid_chunks(&g, itemsize, shape.len() - g.n_spatial())?, &shape));
        }
        if section == "transforms" && parts.len() == 3 && parts[2] == "field" {
            return match self.grid(attrs::get_str(parent, "field_grid")?) {
                None => Ok(None),
                Some(g) => field_chunks(&g, &shape, itemsize),
            };
        }
        if section == "index" && parts.last() == Some(&"occupancy") && shape.len() > 1 {
            let mut c = vec![1];
            c.extend(&shape[1..]);
            return Ok(Some(c));
        }
        Ok(None)
    }
}

/// The datasets of a recompression: each re-encoded under the profile, the
/// graph around them kept by [`GraphCopy`] --- an alias stays an alias, a
/// soft link a link, a cycle a cycle.  A sample root's grids decide its
/// chunks when `rechunk` asks for the writer's derivation.
struct Recode<'a> {
    codec: &'a CodecProfile,
    rechunk: bool,
    /// How many threads compress each dataset's chunks.
    threads: usize,
    result: &'a mut RecompressResult,
    /// One entry per group entered: the layout of the sample root it is,
    /// when it is one.
    layouts: Vec<Option<Layout>>,
    /// One entry per group entered: its name, when it is a sample root.
    roots: Vec<Option<String>>,
}

impl CopyHooks for Recode<'_> {
    fn enter(&mut self, src: &hdf5::Group) -> Result<()> {
        let root = ops::child_group(src, "grids").is_some();
        self.roots.push(root.then(|| src.name()));
        self.layouts.push((self.rechunk && root).then(|| Layout::new(src)));
        Ok(())
    }

    fn leave(&mut self, _src: &hdf5::Group) {
        self.layouts.pop();
        self.roots.pop();
    }

    fn dataset(&mut self, src: &hdf5::Group, name: &str, dst: &hdf5::Group) -> Result<()> {
        // The innermost sample root, and its layout, as the walk nests them.
        let root = self.roots.iter().rev().find_map(Option::as_deref).unwrap_or("/");
        let layout = self.layouts.iter_mut().rev().find_map(Option::as_mut);
        copy_dataset(src, name, dst, self.codec, self.rechunk, self.threads, self.result, layout, root)
    }
}

/// The codec a dataset takes, by the section of its sample root it lies in
/// (`relative` is its path from that root): what `annotations/`,
/// `transforms/` and `index/` hold is coded as labels, the rest as images.
/// A collection member named `annotations` holds images like any other.
fn role(relative: &str) -> Role {
    match relative.split('/').next() {
        Some("annotations" | "transforms" | "index") => Role::Label,
        _ => Role::Image,
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_dataset(
    parent: &hdf5::Group,
    name: &str,
    dst: &hdf5::Group,
    codec: &CodecProfile,
    rechunk: bool,
    threads: usize,
    result: &mut RecompressResult,
    layout: Option<&mut Layout>,
    root: &str,
) -> Result<()> {
    let ds = parent.dataset(name)?;
    let dtype = match data::kind(&ds)? {
        Kind::Numeric(d) => d,
        _ => return copy_stored(parent, name, dst),
    };
    // A rebuild writes `dtype`'s own type.  A stored type that is not that
    // one --- an enumeration with its names, a committed type with its
    // attributes, another byte order --- is copied as it is, not rebuilt
    // into a plain number type and the rest silently lost.
    if !data::stored_as_written(&ds, dtype)? {
        result.kept.push(ds.name());
        return copy_stored(parent, name, dst);
    }
    let chunks = if rechunk {
        match layout {
            Some(l) => l.chunks_for(&ds, parent)?,
            None => None,
        }
    } else {
        data::chunks(&ds)
    };
    let shape = ds.shape();
    let new_layout = dataset_layout(&shape, dtype.itemsize(), codec, role(&relative_name(&ds.name(), root)), chunks);
    // Below the compression threshold: copied through as stored, exactly
    // what a fresh write would leave contiguous --- unless another writer
    // compressed it with a plugin and the profile is one that needs none,
    // which a copy left needing one.  It is rewritten contiguous.
    let left_needing_plugin =
        || -> Result<bool> { Ok(builtin_only(codec) && !shape.contains(&0) && needs_plugin(&ds)?) };
    if new_layout.chunks.is_none() && !left_needing_plugin()? {
        return copy_stored(parent, name, dst);
    }
    let before = describe_filters(&ds)?;
    let out = data::create_empty(dst, name, dtype, &shape, &new_layout)?;
    // Blosc2 compresses whole chunks on workers, outside HDF5's lock, from
    // slabs cut on the new chunks' boundaries; anything else is written
    // through HDF5, which compresses under its lock.  A row of chunks over
    // the slab budget is read and written a band of its chunk columns at a
    // time; one chunk column over it is written through HDF5, since a slab
    // of whole chunks would read more than the budget at once.
    let budget = slab_budget();
    #[cfg(test)]
    let plane = plane_bytes(&shape, dtype.itemsize());
    let direct = Chunks::of(&out, dtype)
        .and_then(|w| Some((band(w.chunk(), &shape, dtype.itemsize(), budget)?, w)))
        .filter(|_| !through_hdf5());
    #[cfg(test)]
    DIRECT.with(|n| n.set(n.get() + usize::from(direct.is_some())));
    let rows = direct.as_ref().map_or(1, |(_, w)| w.chunk()[0]);
    for (start, stop) in slabs(&shape, dtype.itemsize(), budget, rows) {
        let rows = Slice::new(start as i64, stop as i64);
        match &direct {
            Some((columns, writer)) if shape.len() > 1 && *columns < shape[1] => {
                for at in (0..shape[1]).step_by(*columns) {
                    let end = (at + columns).min(shape[1]);
                    #[cfg(test)]
                    MOST_READ.set(MOST_READ.get().max((stop - start) * (end - at) * plane / shape[1]));
                    let cols = Slice::new(at as i64, end as i64);
                    let block = data::read_region(&ds, &[Index::Slice(rows), Index::Slice(cols)])?;
                    let mut origin = vec![0usize; shape.len()];
                    (origin[0], origin[1]) = (start, at);
                    writer.write_block(&block, &origin, threads)?;
                }
            }
            direct => {
                #[cfg(test)]
                MOST_READ.set(MOST_READ.get().max((stop - start) * plane));
                let block = data::read_region(&ds, &[Index::Slice(rows)])?;
                match direct {
                    Some((_, writer)) => writer.write_rows(&block, start, threads)?,
                    None => {
                        let mut starts = vec![0usize; shape.len()];
                        starts[0] = start;
                        data::write_region(&out, &block, &starts)?;
                    }
                }
            }
        }
    }
    for key in attrs::names(&ds)? {
        attrs::copy_raw(&ds, &out, &key)?;
    }
    let after = describe_filters(&out)?;
    result.datasets += 1;
    if before != after {
        result.changed.push((ds.name(), before, after));
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// Every dataset written through HDF5, for the tests that a direct write
    /// stores what HDF5's would.
    static THROUGH_HDF5: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The datasets written a chunk at a time.
    static DIRECT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The slab budget, when a test sets one (0: the default).
    static BUDGET: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The most bytes one slab read.
    static MOST_READ: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn through_hdf5() -> bool {
    #[cfg(test)]
    return THROUGH_HDF5.get();
    #[cfg(not(test))]
    false
}

/// The bytes one slab reads at most, unless one plane is more.
fn slab_budget() -> usize {
    #[cfg(test)]
    if BUDGET.get() > 0 {
        return BUDGET.get();
    }
    64 << 20
}

/// The columns (indices of the second axis) a slab of whole chunks may
/// take at once under `budget`: all of them when a row of chunks is within
/// it, else as many whole chunk columns as are, and `None` when not even one
/// chunk column is.
fn band(chunk: &[usize], shape: &[usize], itemsize: usize, budget: usize) -> Option<usize> {
    let row = chunk[0].saturating_mul(plane_bytes(shape, itemsize));
    if row <= budget {
        return Some(shape.get(1).copied().unwrap_or(0));
    }
    if shape.len() < 2 {
        return None;
    }
    let column = chunk[0].saturating_mul(plane_bytes(&shape[1..], itemsize));
    let columns = (budget / column) / chunk[1] * chunk[1];
    (columns > 0).then_some(columns)
}

/// The bytes of one index of a dataset's first axis.
fn plane_bytes(shape: &[usize], itemsize: usize) -> usize {
    (shape.get(1..).unwrap_or(&[]).iter().product::<usize>().max(1) * itemsize).max(1)
}

/// Row windows covering a dataset, each under `budget` bytes, and each a
/// whole number of `rows` (a chunk's first extent) but the last.
fn slabs(shape: &[usize], itemsize: usize, budget: usize, rows: usize) -> Vec<(usize, usize)> {
    if shape.is_empty() || shape[0] == 0 {
        return vec![(0, shape.first().copied().unwrap_or(0))];
    }
    let per_plane = plane_bytes(shape, itemsize);
    let rows = rows.max(1);
    let step = ((budget / per_plane) / rows * rows).clamp(rows, shape[0].max(rows));
    let mut out = Vec::new();
    let mut start = 0;
    while start < shape[0] {
        let stop = (start + step).min(shape[0]);
        out.push((start, stop));
        start = stop;
    }
    out
}

/// Recompress many files: every one of them, a file that cannot be
/// recompressed stopping no other.
///
/// When any could not be, the error --- the first failure's kind --- names
/// each file that failed and why, and how many of the others were
/// recompressed, as `fix_paths` reports a repair.  Stopping at the first
/// failure left the files after it as they were, and lost the results of
/// those before it, already replaced.  One file fails as [`recompress`] does.
pub fn recompress_paths(paths: &[&Path], profile: &str, rechunk: bool) -> Result<Vec<RecompressResult>> {
    check_profile(profile)?;
    let mut done = Vec::new();
    let mut failed: Vec<(String, Error)> = Vec::new();
    for path in paths {
        match recompress(path, profile, None, rechunk) {
            Ok(result) => done.push(result),
            Err(e) => failed.push((path.to_string_lossy().into_owned(), e)),
        }
    }
    if failed.is_empty() {
        return Ok(done);
    }
    if paths.len() == 1 {
        return Err(failed.remove(0).1);
    }
    let listed: Vec<String> = failed.iter().map(|(path, e)| format!("{}: {}", repr_str(path), e.message())).collect();
    let message = format!(
        "{} of {} files could not be recompressed, and {} were: {}",
        failed.len(),
        paths.len(),
        done.len(),
        listed.join("; ")
    );
    Err(crate::integrity::restated(failed.remove(0).1, message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{DType, NdArray};
    use crate::sample::{create, GridOptions, ImageOptions};

    /// A sample of two images, one of noise, and a mask, under
    /// `profile`, with edge chunks on every axis.
    fn sample(path: &Path, profile: &str) {
        let mut writer = create(path, Some("s"), None, profile, &[]).unwrap();
        let shape = [37usize, 70, 83];
        writer.add_grid("g", &[37, 70, 83], &[1.0, 1.0, 1.0], GridOptions::default()).unwrap();
        let n: usize = shape.iter().product();
        let ct = NdArray::from_vec(&shape, (0..n).map(|i| ((i / 7) % 300) as f64 - 100.0).collect()).unwrap();
        writer.add_image("CT", &ct.astype(DType::I16), "g", "CT", ImageOptions::default()).unwrap();
        let mut rng = crate::rng::Rng::new(5);
        let pet = NdArray::from_vec(&shape, (0..n).map(|_| 9.0 * rng.random()).collect()).unwrap();
        writer.add_image("PET", &pet.astype(DType::F32), "g", "PT", ImageOptions::default()).unwrap();
        let mask =
            ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&shape), |i| (i[0] / 9 + i[1] / 20 + i[2] / 30) % 3 == 0);
        writer.add_mask("fov", mask, "g", "other", None, None).unwrap();
        writer.commit(true).unwrap();
    }

    /// A recompression stores, at one thread and at four, exactly the chunks
    /// HDF5's own write of the same values stores --- every chunk's bytes and
    /// filter mask, under each profile from each --- and the same
    /// `content_id`.
    #[test]
    fn s14_2_a_recompression_stores_hdf5s_chunks_at_any_thread_count() {
        let _serial = crate::parallel::tests::serial();
        let dir = tempfile::tempdir().unwrap();
        for from in ["portable", "training"] {
            let source = dir.path().join(format!("{from}.medh5"));
            sample(&source, from);
            for profile in ["training", "balanced", "archive", "portable"] {
                for rechunk in [false, true] {
                    let mut outputs = Vec::new();
                    for (threads, hdf5) in [(1, true), (1, false), (4, false)] {
                        let out = dir.path().join(format!("{from}-{profile}-{rechunk}-{threads}-{hdf5}.medh5"));
                        let _threads = crate::parallel::for_this_run(threads).unwrap();
                        THROUGH_HDF5.set(hdf5);
                        DIRECT.set(0);
                        let result = recompress(&source, profile, Some(&out), rechunk);
                        THROUGH_HDF5.set(false);
                        let result = result.unwrap();
                        assert!(result.ok(), "{result:?}");
                        // Both images, unless HDF5 writes them; the mask
                        // is a `bool`, an enumeration HDF5 maps.
                        let direct = if hdf5 || profile == "portable" { 0 } else { 2 };
                        assert_eq!(DIRECT.get(), direct, "{from} to {profile}");
                        outputs.push((result.content_id.clone(), crate::h5::chunks::tests::stored_chunks(&out)));
                    }
                    let (reference, rest) = outputs.split_first().unwrap();
                    assert!(!reference.1.is_empty());
                    for other in rest {
                        assert_eq!(other.0, reference.0, "{from} to {profile}");
                        assert_eq!(other.1, reference.1, "{from} to {profile}, rechunk {rechunk}");
                    }
                }
            }
        }
    }

    /// A dataset whose one chunk column is over the slab budget is written
    /// through HDF5, a budget at a time: a slab of whole chunks would read
    /// the column whole.  The file is the one a direct write gives.
    #[test]
    fn s14_2_a_chunk_column_over_the_slab_budget_is_written_through_hdf5() {
        let _serial = crate::parallel::tests::serial();
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("s.medh5");
        sample(&source, "portable");
        // Four planes of the f32 PET, less than one chunk column of 37 planes.
        let budget = 4 * 70 * 83 * 4;
        let mut ids = Vec::new();
        for (n, set) in [(0, false), (1, true)] {
            let out = dir.path().join(format!("{n}.medh5"));
            let _threads = crate::parallel::for_this_run(4).unwrap();
            BUDGET.set(if set { budget } else { 0 });
            DIRECT.set(0);
            MOST_READ.set(0);
            let result = recompress(&source, "archive", Some(&out), false);
            BUDGET.set(0);
            let result = result.unwrap();
            assert!(result.ok(), "{result:?}");
            if set {
                assert_eq!(DIRECT.get(), 0);
                assert!(MOST_READ.get() <= budget, "{} > {budget}", MOST_READ.get());
            } else {
                assert_eq!(DIRECT.get(), 2);
            }
            ids.push(result.content_id);
        }
        assert_eq!(ids[0], ids[1]);
    }

    /// A row of chunks over the slab budget, whose chunk columns are within
    /// it, is written directly a band of columns at a time, never reading
    /// more than the budget, and stores the chunks a whole-row write does.
    #[test]
    fn s14_2_a_chunk_row_over_the_slab_budget_is_written_a_band_at_a_time() {
        let _serial = crate::parallel::tests::serial();
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("s.medh5");
        let shape = [20usize, 400, 150];
        let mut writer = create(&source, Some("s"), None, "portable", &[]).unwrap();
        writer.add_grid("g", &[20, 400, 150], &[1.0, 1.0, 1.0], GridOptions::default()).unwrap();
        let n: usize = shape.iter().product();
        let mut rng = crate::rng::Rng::new(7);
        let pet = NdArray::from_vec(&shape, (0..n).map(|_| 9.0 * rng.random()).collect()).unwrap();
        writer.add_image("PET", &pet.astype(DType::F32), "g", "PT", ImageOptions::default()).unwrap();
        writer.commit(true).unwrap();
        let whole = dir.path().join("whole.medh5");
        let _threads = crate::parallel::for_this_run(4).unwrap();
        let reference = recompress(&source, "archive", Some(&whole), false).unwrap();
        let chunk = {
            let file = crate::h5::file::marked(|| hdf5::File::open(&whole)).unwrap();
            Chunks::of(&file.dataset("images/PET").unwrap(), DType::F32).unwrap().chunk().to_vec()
        };
        assert!(chunk[1] * 2 < shape[1], "{chunk:?}: the test needs several chunk columns");
        // Two chunk columns: under a row, over one column.
        let budget = chunk[0] * 2 * chunk[1] * shape[2] * 4;
        let out = dir.path().join("bands.medh5");
        BUDGET.set(budget);
        DIRECT.set(0);
        MOST_READ.set(0);
        let result = recompress(&source, "archive", Some(&out), false);
        BUDGET.set(0);
        let result = result.unwrap();
        assert!(result.ok(), "{result:?}");
        assert_eq!(DIRECT.get(), 1);
        assert!(MOST_READ.get() <= budget, "{} > {budget}", MOST_READ.get());
        assert_eq!(result.content_id, reference.content_id);
        let stored = crate::h5::chunks::tests::stored_chunks;
        assert_eq!(stored(&out), stored(&whole));
    }

    /// What the check of a recompressed file cannot read is reported in the
    /// result: the file has been replaced by then, and an error raised there
    /// read as a recompression that wrote nothing.
    #[test]
    fn s14_4_what_the_output_check_cannot_read_is_reported_not_raised() {
        use crate::h5sys::{h5d, h5p};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        let mut writer = create(&path, Some("s"), None, "portable", &[]).unwrap();
        writer.add_grid("g", &[4, 6, 8], &[1.0, 1.0, 1.0], GridOptions::default()).unwrap();
        let ct = NdArray::from_vec(&[4, 6, 8], (0..192).map(f64::from).collect()).unwrap().astype(DType::I16);
        writer.add_image("CT", &ct, "g", "CT", ImageOptions::default()).unwrap();
        writer.commit(true).unwrap();
        // An extension dataset of a compound type, which recompression copies
        // as stored, with a chunk whose deflate stream is damaged.
        {
            #[derive(hdf5::H5Type, Clone, Copy)]
            #[repr(C)]
            struct Pair {
                a: i32,
                b: f32,
            }
            let file = crate::h5::file::marked(|| hdf5::File::open_rw(&path)).unwrap();
            let ds = file
                .create_group("x_acme")
                .unwrap()
                .new_dataset::<Pair>()
                .shape([4])
                .chunk([4])
                .deflate(4)
                .create("table")
                .unwrap();
            let garbage = [0xA5u8; 16];
            let offset = [0u64];
            let status = crate::h5::locked(|| unsafe {
                h5d::H5Dwrite_chunk(
                    ds.id(),
                    h5p::H5P_DEFAULT,
                    0,
                    offset.as_ptr(),
                    garbage.len(),
                    garbage.as_ptr().cast(),
                )
            });
            assert!(status >= 0);
            attrs::write(&ds, "digest", &attrs::AttrValue::Str(format!("sha256:{}", "0".repeat(64)))).unwrap();
        }
        let result = recompress(&path, "archive", None, false).unwrap();
        assert!(!result.verified && !result.ok());
        assert_eq!(result.unverified.len(), 1, "{:?}", result.unverified);
        assert!(result.to_json()["unverified"].is_array());
        // The file was replaced, and the rest of it reads.
        let sample = crate::sample::open_sample(&path).unwrap();
        assert_eq!(sample.image("CT").unwrap().read(None, false, None).unwrap(), ct);
    }
}
