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
    let before = atomic_rewrite(path, Some(target), |src, dst| {
        stage!("recompress.rewrite");
        require_major(src, path)?;
        let before = attrs::get_str(src, "content_id")?;
        let src_root = src.as_group()?;
        ops::refuse_references(&src_root, "recompression")?;
        let dst_root = dst.as_group()?;
        let mut hooks = Recode { codec: &codec, rechunk, result: &mut result, layouts: Vec::new(), roots: Vec::new() };
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
        copy_dataset(src, name, dst, self.codec, self.rechunk, self.result, layout, root)
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
    for (start, stop) in slabs(&shape, dtype.itemsize(), 64 << 20) {
        let block = data::read_region(&ds, &[Index::Slice(Slice::new(start as i64, stop as i64))])?;
        let mut starts = vec![0usize; shape.len()];
        starts[0] = start;
        data::write_region(&out, &block, &starts)?;
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

/// Row windows covering a dataset, each under `budget` bytes.
fn slabs(shape: &[usize], itemsize: usize, budget: usize) -> Vec<(usize, usize)> {
    if shape.is_empty() || shape[0] == 0 {
        return vec![(0, shape.first().copied().unwrap_or(0))];
    }
    let trailing: usize = shape[1..].iter().product::<usize>().max(1);
    let per_plane = (trailing * itemsize).max(1);
    let step = (budget / per_plane).clamp(1, shape[0]);
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
