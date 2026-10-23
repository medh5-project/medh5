//! §13 --- digests, `content_id` and index currency (E7xx).

use std::collections::{BTreeMap, HashMap};

use super::{grid_spatial, loc, str_attr, sub_dataset, Context};
use crate::annotations::encode::normalization_tolerance;
use crate::array::{DType, NdArray};
use crate::digest::{is_canonical_digest, parse_digest, DEFAULT_ALGO};
use crate::h5::attrs;
use crate::h5::data::{self, Kind};
use crate::h5::ops::{self, Node};
use crate::integrity::verify::{verify_root_reporting, ATTESTED_GROUPS};
use crate::integrity::workers::{self, Inspector, Job};
use crate::integrity::{unbound, Binding};
use crate::json::{py_float, repr_int_tuple, repr_str};
use crate::validate::Diagnostic;
use crate::Result;

/// Integrity: every object digest and `content_id` (§13.1--§13.2:
/// E701--E704, W901, W913 for a projection's), the sampling index's currency
/// (§13.3, W905), and the stored values of probability maps as the digest
/// pass reads them (§7.5: E404, E411).
///
/// An object that cannot be read --- a dataset whose chunk no longer
/// decodes --- is E001 where it is, and every other object is still checked.
/// Every finding is reported whatever else the file holds: a file in which
/// no digest parses had its E703, its `content_id` mismatch and its stale
/// index dropped, and passed, while `verify` failed it.
pub fn check_integrity(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let root = ctx.root.clone();
    let mut algo_known = true;
    if let Some(algo) = str_attr(&root, "digest_algo")? {
        if parse_digest(&format!("{algo}:00")).is_err() {
            algo_known = false;
            out.push(ctx.err(
                "E703",
                "/",
                format!("unsupported digest_algo {}; §2.1 permits sha256, sha512 and blake2b", repr_str(&algo)),
            ));
        }
    }
    // The stored probabilities are checked as the digest pass reads them.
    let mut probmaps = probmaps(ctx, &mut out)?;
    let (result, unread) = {
        let mut inspectors: HashMap<String, Inspector<'_>> =
            probmaps.iter_mut().map(|(path, check)| (path.clone(), feeding(check))).collect();
        verify_root_reporting(&root, ctx.attr_names.as_ref(), algo_known, &mut inspectors)?
    };
    for (path, e) in &unread {
        out.push(match (path.is_empty(), e.code()) {
            (true, None) => {
                ctx.err("E001", "/", format!("the root's `content_id` could not be checked: {}", e.python_line()))
            }
            (true, Some(_)) => ctx.failed("/", e),
            (false, _) => ctx.failed(format!("/{path}"), e),
        });
    }
    let unreadable: Vec<&str> = unread.iter().map(|(path, _)| path.as_str()).collect();
    // Undigested, so the pass above did not read them: read here, on the
    // workers, and each failure reported in the order of the maps.
    let mut failures: Vec<(String, Option<crate::Error>)> = Vec::new();
    let mut jobs = Vec::new();
    for (path, check) in probmaps.iter_mut().filter(|(p, c)| !c.read && !unreadable.contains(&p.as_str())) {
        match root.dataset(path) {
            Ok(ds) => {
                let mut job = Job::new(ds, path, DEFAULT_ALGO);
                job.inspect = Some(feeding(check));
                jobs.push(job);
                failures.push((path.clone(), None));
            }
            Err(e) => failures.push((path.clone(), Some(e.into()))),
        }
    }
    let mut read = workers::digests(jobs)?.into_iter();
    for (path, failed) in failures {
        let failed = match failed {
            Some(e) => Some(e),
            None => read.next().expect("a result for every map read").err(),
        };
        if let Some(e) = failed {
            out.push(ctx.failed(format!("/{path}"), &e));
        }
    }
    drop(read);
    for (path, check) in &probmaps {
        out.extend(check.findings(ctx, path));
    }
    // W901 says no dataset carries a digest: one whose digest does not parse,
    // or whose type has none, carries one.
    let digested = !(result.checked.is_empty() && result.malformed.is_empty() && result.unverifiable.is_empty());
    let no_digests = !result.undigested.is_empty() && !digested && !unread.iter().any(|(path, _)| !path.is_empty());
    if no_digests {
        out.push(ctx.err("W901", "/", "no dataset carries a `digest` attribute"));
    }
    for path in &result.malformed {
        out.push(ctx.err("E703", format!("/{path}"), "malformed digest string"));
    }
    for path in &result.mismatched {
        out.push(ctx.err("E701", format!("/{path}"), "digest does not match the stored data"));
    }
    if result.content_id_ok() == Some(false) && ctx.projection {
        // A later minor may cover root attributes this engine does not know:
        // the root's integrity is unsupported here, not refuted (1.1 §2.2).
        out.push(ctx.err(
            "W913",
            "/",
            format!(
                "`content_id` does not recompute under the rules of MEDH5 {}: the root's integrity is unsupported in \
                 this projection, not verified (per-dataset digests were checked)",
                crate::FORMAT_VERSION
            ),
        ));
    } else if result.content_id_declared.as_deref().is_some_and(|d| !is_canonical_digest(d)) {
        // A spelling, not a mismatch (§13.1): reported as tampering, it sent
        // people looking for corruption that was not there.
        out.push(ctx.err(
            "E703",
            "/",
            format!(
                "malformed `content_id` {}: §13.1 spells a digest `<algo>:` and lowercase hex of the algorithm's \
                 length",
                repr_str(result.content_id_declared.as_deref().unwrap_or_default())
            ),
        ));
    } else if result.content_id_ok() == Some(false) {
        out.push(ctx.err(
            "E702",
            "/",
            format!(
                "`content_id` {} does not match the computed {}",
                result.content_id_declared.clone().unwrap_or_default(),
                result.content_id_computed.clone().unwrap_or_default()
            ),
        ));
    }
    if result.content_id_declared.is_some() && !unreadable.contains(&"") {
        // A line binds a dataset's bytes to the path it names (§13.2).  A
        // dataset carrying a digest that no line lists claims an attestation
        // it does not have: its object is reached first through `index/`,
        // which the root excludes, or only through a soft link, and its bytes
        // changed under a matching root and an unchanged pin.  A path that is
        // not its object's own is covered nowhere: relinked to other covered
        // bytes, it changed no line, and a reader read them under a matching
        // root and an unchanged pin.
        for found in unbound(&root)? {
            let at = format!("/{}", found.path);
            match found.binding {
                // A SHOULD (§13.1), so a warning; `clinical/` makes it a MUST
                // (1.1 §8, E818), and a file with no digests at all is W901.
                Binding::Undigested => {
                    let group = found.path.split('/').next().unwrap_or_default();
                    if !no_digests && ATTESTED_GROUPS.contains(&group) {
                        out.push(ctx.err(
                            "W915",
                            at,
                            "carries no `digest`, inside an object `content_id` covers: the root does not attest its \
                             bytes, so an edit to them changes no line (§13.1, §13.2)",
                        ));
                    }
                }
                Binding::Unlisted => out.push(ctx.err(
                    "E702",
                    at,
                    "carries a digest no line of `content_id` covers: its object is reached first through `index/`, \
                     which the root excludes, or only through a soft link, so the root does not speak for its bytes",
                )),
                Binding::Elsewhere { first: Some(own) } => out.push(ctx.err(
                    "E704",
                    at,
                    format!(
                        "is not its object's own path, {}: `content_id` binds the object's bytes there, not here, so \
                         this name could lead to other bytes under an unchanged root",
                        repr_str(&format!("/{own}"))
                    ),
                )),
                Binding::Elsewhere { first: None } => out.push(ctx.err(
                    "E704",
                    at,
                    "leads out of the sample --- a soft link the root's walk does not reach, or an external link --- \
                     so no line of `content_id` binds what is read here",
                )),
            }
        }
    }
    for name in &result.stale_index {
        out.push(ctx.err(
            "W905",
            format!("/index/{name}"),
            "index `source_digest` does not match its annotation; readers must ignore this entry and rebuild it",
        ));
    }
    Ok(out)
}

/// What a `probmap`'s stored values must be (§7.5): numbers in [0, 1] and,
/// where it is `normalized`, summing to 1 over its classes at every voxel ---
/// checked on the bytes the integrity pass reads anyway.  Only the threshold
/// was checked, so a map of NaN, or one declared normalised that was not,
/// validated clean.
///
/// The sums are checked only on a map whose shape is one --- a row per class
/// it declares, on its grid's spatial shape, which the semantic level holds it
/// to (E405) --- and held one map at a time: allocated, fallibly, when its
/// first block is read, and judged and freed after its last.  A shape of
/// `(0, 2**62)` allocated a sum per voxel before anything was checked, and
/// the validator panicked.
///
/// A block is whatever the digest pass reads, about [`STREAM_BYTES`] in C
/// order: each value is placed by how many came before it, so no block need
/// hold a class's plane whole.
struct ProbmapCheck {
    class_ids: Vec<i64>,
    spatial: Vec<usize>,
    dtype: DType,
    classes: usize,
    /// Voxels per class where the sums are checked: declared normalized, on a
    /// sound shape.
    summed: Option<usize>,
    /// The values read so far, in C order: where the next block starts.
    seen: usize,
    sums: Vec<f64>,
    bad: Option<(usize, usize, f64)>,
    /// The first voxel whose sum is off, and the sum.
    off: Option<(usize, f64)>,
    /// Why the sums could not be held, if they could not.
    unsummed: Option<String>,
    read: bool,
}

/// Every probability map's check, by its data's path.  A map whose header
/// cannot be read is E001 where it is, and the others are still checked.
fn probmaps(ctx: &Context, out: &mut Vec<Diagnostic>) -> Result<BTreeMap<String, ProbmapCheck>> {
    let mut checks = BTreeMap::new();
    let grids = ops::child_group(&ctx.root, "grids");
    for (name, group) in ctx.children("annotations")? {
        match probmap(&group, grids.as_ref()) {
            Ok(Some(check)) => {
                checks.insert(format!("annotations/{name}/data"), check);
            }
            Ok(None) => {}
            Err(e) => out.push(ctx.failed(format!("/annotations/{name}"), &e)),
        }
    }
    Ok(checks)
}

/// The check of one annotation's stored values, if it is a probability map
/// with numeric data.
fn probmap(group: &Node, grids: Option<&hdf5::Group>) -> Result<Option<ProbmapCheck>> {
    if str_attr(loc(group), "kind")?.as_deref() != Some("probmap") {
        return Ok(None);
    }
    let Some(ds) = sub_dataset(group, "data") else { return Ok(None) };
    let Kind::Numeric(dtype) = data::kind(&ds)? else { return Ok(None) };
    let shape = ds.shape();
    let Some((&classes, spatial)) = shape.split_first() else { return Ok(None) };
    let class_ids = attrs::get_i64s(loc(group), "class_ids")?.unwrap_or_default();
    // A grid that cannot be read is the semantic level's finding; here it
    // only means the map is not summed.  A grid is a group: a dataset of its
    // name is none.
    let on_grid = match (str_attr(loc(group), "grid").ok().flatten(), grids) {
        (Some(grid), Some(grids)) if ops::is_group(grids, &grid) => grid_spatial(grids, &grid)
            .is_ok_and(|g| g.iter().map(|v| usize::try_from(*v).ok()).eq(spatial.iter().map(|v| Some(*v)))),
        _ => false,
    };
    let sound = classes > 0 && classes == class_ids.len() && !spatial.is_empty() && on_grid;
    let normalized = attrs::get_bool(loc(group), "normalized")?.unwrap_or(false);
    Ok(Some(ProbmapCheck {
        class_ids,
        spatial: spatial.to_vec(),
        dtype,
        classes,
        summed: if normalized && sound { spatial.iter().try_fold(1usize, |n, e| n.checked_mul(*e)) } else { None },
        seen: 0,
        sums: Vec::new(),
        bad: None,
        off: None,
        unsummed: None,
        read: false,
    }))
}

/// A map's check as the digest pass hands it each block.
fn feeding(check: &mut ProbmapCheck) -> Inspector<'_> {
    Box::new(move |block| {
        check.feed(block);
        Ok(())
    })
}

impl ProbmapCheck {
    /// The next block of the map, in C order, of any size.
    fn feed(&mut self, block: &NdArray) {
        self.read = true;
        // The values of one class: a voxel's place is its offset in a plane.
        let plane = self.spatial.iter().try_fold(1usize, |n, e| n.checked_mul(*e)).unwrap_or(usize::MAX).max(1);
        if self.seen == 0 && self.unsummed.is_none() {
            if let Some(n) = self.summed {
                if let Err(e) = self.sums.try_reserve_exact(n) {
                    self.unsummed = Some(format!("the sums of its {n} voxels do not fit in memory ({e})"));
                    self.summed = None;
                } else {
                    self.sums.resize(n, 0.0);
                }
            }
        }
        let values = block.to_f64();
        for (k, v) in values.iter().enumerate() {
            let at = self.seen + k;
            if self.bad.is_none() && !(0.0..=1.0).contains(v) {
                self.bad = Some((at / plane, at % plane, *v));
            }
            if self.summed.is_some() {
                self.sums[at % plane] += *v;
            }
        }
        self.seen += values.len();
        if self.seen / plane >= self.classes && self.summed.is_some() {
            // The last value: judged now, so the next map's sums are not held
            // beside these.
            let (dtype, classes) = (self.dtype, self.classes);
            self.off = self
                .sums
                .iter()
                .position(|s| (*s - 1.0).abs() > normalization_tolerance(dtype, classes, *s))
                .map(|voxel| (voxel, self.sums[voxel]));
            self.sums = Vec::new();
        }
    }

    fn voxel(&self, flat: usize) -> String {
        let mut index = vec![0; self.spatial.len()];
        let mut rest = flat;
        for (axis, extent) in self.spatial.iter().enumerate().rev() {
            let extent = (*extent).max(1);
            index[axis] = rest % extent;
            rest /= extent;
        }
        repr_int_tuple(&index)
    }

    fn findings(&self, ctx: &Context, path: &str) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        if let Some((row, voxel, value)) = self.bad {
            let class = self.class_ids.get(row).map_or_else(|| format!("row {row}"), |c| format!("class {c}"));
            out.push(ctx.err(
                "E411",
                format!("/{path}"),
                format!(
                    "a stored probability is {} ({class}, voxel {}); §7.5 stores values in [0, 1]",
                    py_float(value),
                    self.voxel(voxel)
                ),
            ));
        } else if let Some((voxel, sum)) = self.off {
            out.push(ctx.err(
                "E404",
                format!("/{path}"),
                format!(
                    "`normalized` is true, but the classes sum to {} at voxel {}; §7.5 has them sum to 1 at every \
                     voxel",
                    py_float(sum),
                    self.voxel(voxel)
                ),
            ));
        }
        if let Some(why) = &self.unsummed {
            out.push(ctx.err("E001", format!("/{path}"), format!("`normalized` could not be checked: {why}")));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A normalized map of `classes` planes over `spatial`, before any read.
    fn unread(classes: usize, spatial: &[usize]) -> ProbmapCheck {
        ProbmapCheck {
            class_ids: (1..=classes as i64).collect(),
            spatial: spatial.to_vec(),
            dtype: DType::F32,
            classes,
            summed: Some(spatial.iter().product()),
            seen: 0,
            sums: Vec::new(),
            bad: None,
            off: None,
            unsummed: None,
            read: false,
        }
    }

    /// The stored values are judged from blocks of any size, in C order: the
    /// check asked the digest pass for whole rows of the leading axis, so the
    /// integrity level read every dataset a row at a time --- a class's whole
    /// plane of a probability map, one channel of a channel-first image.
    #[test]
    fn s7_5_a_probability_map_is_judged_from_blocks_of_any_size() {
        // Two classes on a 2 x 3 grid: class 1 holds 1.5 at voxel (0, 1), and
        // the classes sum to 0.75 at voxel (1, 2).
        let first = [0.25f32, 1.5, 0.5, 0.5, 0.5, 0.5];
        let second = [0.75f32, -0.5, 0.5, 0.5, 0.5, 0.25];
        let flat: Vec<f32> = first.iter().chain(&second).copied().collect();
        for size in 1..=flat.len() {
            let mut check = unread(2, &[2, 3]);
            for block in flat.chunks(size) {
                check.feed(&NdArray::from_vec(&[block.len()], block.to_vec()).unwrap());
            }
            assert_eq!(check.bad, Some((0, 1, 1.5)), "blocks of {size}");
            assert_eq!(check.off, Some((5, 0.75)), "blocks of {size}");
            assert_eq!(check.voxel(5), "(1, 2)");
            assert!(check.sums.is_empty(), "freed once judged");
        }
    }
}
