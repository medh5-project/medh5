//! Per-object digests and the Merkle `content_id` (spec §13).
//!
//! A digest covers one object's **decompressed** content, so recompressing a
//! file does not invalidate it.  The root `content_id` is a Merkle root over
//! the sorted per-object digests: incremental, partially verifiable, a content
//! address, and local --- a mismatch names the object that changed.
//!
//! Digest input for a dataset (§13.1):
//!
//! ```text
//! H( object_path || 0x00 || dtype_str || 0x00 || shape_csv
//!    || 0x00 || C-order little-endian bytes )
//! ```
//!
//! A string dataset hashes as `|O`, each string's stored bytes followed by
//! `0x00`; a dataset of another fixed-size type --- a compound, an array, an
//! opaque type --- as `|V<size>` over its elements' stored bytes.  Data of
//! variable length other than strings, and references, have no canonical
//! byte stream, so no digest: they are never stamped, and one found on them
//! cannot be verified.

use indexmap::IndexMap;

use crate::array::{Index, NdArray, Slice};
use crate::digest::{Hasher, DEFAULT_ALGO};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::data::{self, Kind};
use crate::h5::ops;
use crate::{Error, Result};

/// Slab size for streaming a large dataset through the hash.
pub const STREAM_BYTES: usize = 32 * 1024 * 1024;

/// The little-endian bytes a block is hashed in, a piece at a time.
pub(super) const LE_PIECE_BYTES: usize = 1 << 20;

pub(super) fn feed_header(hasher: &mut Hasher, path: &str, dtype_str: &str, shape: &[usize]) {
    hasher.update(path.as_bytes());
    hasher.update(b"\x00");
    hasher.update(dtype_str.as_bytes());
    hasher.update(b"\x00");
    let csv: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    hasher.update(csv.join(",").as_bytes());
    hasher.update(b"\x00");
}

/// Digest of an in-memory array under the §13.1 canonical byte stream.
pub fn array_digest(path: &str, array: &NdArray, algo: &str) -> Result<String> {
    let mut hasher = Hasher::new(algo)?;
    feed_header(&mut hasher, path, array.dtype().numpy_str(), &array.shape());
    hasher.update(&array.le_bytes());
    Ok(hasher.finish())
}

/// Digest of in-memory strings, as a variable-length string dataset hashes.
pub fn strings_digest(path: &str, values: &[String], shape: &[usize], algo: &str) -> Result<String> {
    let mut hasher = Hasher::new(algo)?;
    feed_header(&mut hasher, path, "|O", shape);
    for value in values {
        hasher.update(value.as_bytes());
        hasher.update(b"\x00");
    }
    Ok(hasher.finish())
}

/// Whether a dataset's type has a digest (§13.1): numbers, strings, and
/// every other type of fixed-size elements.  Data of variable length other
/// than strings, and references, have no bytes of their own to hash.
pub fn digestible(ds: &hdf5::Dataset) -> Result<bool> {
    Ok(match data::kind(ds)? {
        Kind::Numeric(_) | Kind::Strings => true,
        Kind::Other(_) => data::stored_element_size(ds)?.is_some(),
    })
}

/// Digest an HDF5 dataset, streaming so a large volume never lands in RAM.
///
/// String datasets hash each string's stored bytes followed by `0x00`,
/// because a variable-length dataset has no meaningful raw byte stream; a
/// type [`digestible`] refuses is a `TypeError`.
pub fn dataset_digest(ds: &hdf5::Dataset, path: &str, algo: &str) -> Result<String> {
    dataset_digest_streamed(ds, path, algo, STREAM_BYTES)
}

/// [`dataset_digest`] reading about `stream_bytes` at a time: the slab size
/// changes the reads, never the digest.
pub fn dataset_digest_streamed(ds: &hdf5::Dataset, path: &str, algo: &str, stream_bytes: usize) -> Result<String> {
    dataset_digest_inspected(ds, path, algo, stream_bytes, None)
}

/// What [`dataset_digest_inspected`] hands each block it reads to.
pub type Inspect<'a> = &'a mut dyn FnMut(&NdArray) -> Result<()>;

/// [`dataset_digest_streamed`], handing every block of a numeric dataset it
/// reads to `inspect`, in C order, so a check of the stored values costs no
/// second read.
///
/// A block is about `stream_bytes` however the dataset is shaped: when one
/// row of the leading axis is larger, the leading axes are read an index at a
/// time and the first axis whose rows fit is sliced.  A channel-first image,
/// a layered annotation or a displacement field has a short leading axis, and
/// a row at a time read and copied one 256 MiB row whole --- as the value
/// checks still asked for, until they placed each value by its offset.  A
/// row no buffer can hold is refused, as any read is.
pub fn dataset_digest_inspected(
    ds: &hdf5::Dataset,
    path: &str,
    algo: &str,
    stream_bytes: usize,
    mut inspect: Option<Inspect<'_>>,
) -> Result<String> {
    let mut hasher = Hasher::new(algo)?;
    let shape = ds.shape();
    match data::kind(ds)? {
        Kind::Strings => {
            feed_header(&mut hasher, path, "|O", &shape);
            // A slab at a time: holding every string of the dataset at once
            // let a few kilobytes of file take hundreds of megabytes.  Each
            // string is hashed as stored: decoded as text, different invalid
            // bytes hashed alike, and an edit to them went unseen.
            data::for_each_string_slab(ds, stream_bytes, &mut |values| {
                for value in values {
                    hasher.update(&value);
                    hasher.update(b"\x00");
                }
                Ok(())
            })?;
            Ok(hasher.finish())
        }
        Kind::Other(t) => {
            let Some(size) = data::stored_element_size(ds)? else {
                return Err(Error::Type(format!(
                    "{} holds {t}, data of variable length or references, which have no digest (§13.1)",
                    ds.name()
                )));
            };
            // NumPy's string for a fixed-size type it has no number for, as
            // 1.x hashed one through h5py.
            feed_header(&mut hasher, path, &format!("|V{size}"), &shape);
            if shape.contains(&0) {
                return Ok(hasher.finish());
            }
            let rows = shape.first().copied().unwrap_or(1);
            let row_bytes = shape.iter().skip(1).try_fold(size, |n, e| n.checked_mul(*e)).unwrap_or(usize::MAX);
            let step = (stream_bytes / row_bytes.max(1)).clamp(1, rows);
            let chunk = data::chunks(ds).and_then(|c| c.first().copied());
            let slabs = Slabs::new(step, chunk, row_bytes, stream_bytes);
            let mut start = 0;
            while start < rows {
                let stop = slabs.end(start, rows);
                hasher.update(&data::read_stored_rows(ds, size, start, stop - start)?);
                start = stop;
            }
            Ok(hasher.finish())
        }
        Kind::Numeric(dtype) => {
            feed_header(&mut hasher, path, dtype.numpy_str(), &shape);
            // Empty by any extent of 0, whatever the others: a product of the
            // extents can overflow first.
            if shape.contains(&0) {
                return Ok(hasher.finish());
            }
            if shape.is_empty() {
                let all = data::read(ds)?;
                if let Some(f) = inspect.as_mut() {
                    f(&all)?;
                }
                hasher.update(&all.le_bytes());
                return Ok(hasher.finish());
            }
            // A row too large to hold is refused, not read a slice at a time.
            data::ensure_allocatable(ds, &shape[1..], dtype.itemsize())?;
            for_each_block(ds, &shape, dtype.itemsize(), stream_bytes, &mut |block| {
                if let Some(f) = inspect.as_mut() {
                    f(&block)?;
                }
                // A piece at a time: a copy of the block beside it doubled
                // what a digest holds.
                block.for_each_le_piece(LE_PIECE_BYTES, &mut |bytes| hasher.update(bytes));
                Ok(())
            })?;
            Ok(hasher.finish())
        }
    }
}

/// The bytes of one index along `axis` of an array of `shape`.
pub(super) fn row_bytes(shape: &[usize], axis: usize, itemsize: usize) -> usize {
    shape[axis + 1..].iter().try_fold(itemsize, |n, e| n.checked_mul(*e)).unwrap_or(usize::MAX).max(1)
}

/// How one axis of a chunked dataset is cut into slabs: `step` rows each,
/// at chunk boundaries, so no chunk is decompressed for two slabs.
///
/// The step is the budget's, rounded down to whole bands of chunks along the
/// axis.  A band larger than the budget is read whole when it is at most
/// twice the budget, so each of its chunks is still decompressed once; past
/// that it is cut evenly, and no slab crosses into the next band.  Slabs
/// that straddled bands had every chunk of a band decompressed for each
/// slab touching it --- 2 to 4 times over, as HDF5's chunk cache holds far
/// less than a band.
pub(super) struct Slabs {
    step: usize,
    /// The band no slab crosses, where a band is cut into several.
    band: Option<usize>,
}

impl Slabs {
    /// `step` rows of `row` bytes fit the budget; `chunk` is the chunk extent
    /// along the axis, `None` for a dataset that is not chunked.
    pub(super) fn new(step: usize, chunk: Option<usize>, row: usize, budget: usize) -> Slabs {
        match chunk {
            Some(c) if c > 1 && step >= c => Slabs { step: step / c * c, band: None },
            Some(c) if c > 1 && c.saturating_mul(row) <= budget.saturating_mul(2) => Slabs { step: c, band: None },
            Some(c) if c > 1 => Slabs { step: c.div_ceil(c.div_ceil(step.max(1))), band: Some(c) },
            _ => Slabs { step: step.max(1), band: None },
        }
    }

    /// Where the slab starting at row `start` of `n` ends.
    pub(super) fn end(&self, start: usize, n: usize) -> usize {
        let stop = start.saturating_add(self.step).min(n);
        match self.band {
            Some(c) => stop.min((start / c).saturating_add(1).saturating_mul(c)),
            None => stop,
        }
    }
}

/// Read a numeric dataset of `shape` in C order, a block at a time, each
/// about `budget` bytes: whole rows of the leading axis when one fits, else
/// the leading axes an index at a time and the first axis whose rows fit
/// sliced --- in slabs aligned to its chunks ([`Slabs`]).
fn for_each_block(
    ds: &hdf5::Dataset,
    shape: &[usize],
    itemsize: usize,
    budget: usize,
    each: &mut dyn FnMut(NdArray) -> Result<()>,
) -> Result<()> {
    let mut axis = 0;
    while axis + 1 < shape.len() && row_bytes(shape, axis, itemsize) > budget {
        axis += 1;
    }
    let row = row_bytes(shape, axis, itemsize);
    let chunk = data::chunks(ds).and_then(|c| c.get(axis).copied());
    let slabs = Slabs::new((budget / row).max(1), chunk, row, budget);
    let mut outer = vec![0usize; axis];
    loop {
        let mut start = 0;
        while start < shape[axis] {
            let stop = slabs.end(start, shape[axis]);
            let mut index: Vec<Index> = outer.iter().map(|i| Index::At(*i as i64)).collect();
            index.push(Index::Slice(Slice::new(start as i64, stop as i64)));
            each(data::read_region(ds, &index)?)?;
            start = stop;
        }
        // The next index of the axes read one at a time, the last fastest.
        let mut at = axis;
        loop {
            if at == 0 {
                return Ok(());
            }
            at -= 1;
            outer[at] += 1;
            if outer[at] < shape[at] {
                break;
            }
            outer[at] = 0;
        }
    }
}

/// An object's path relative to a sample root.
///
/// Digests hash the object path, so it must be *sample-root relative*: a
/// sample extracted from a collection digests identically standalone (§2.2).
pub fn relative_path(name: &str, root: &hdf5::Group) -> String {
    crate::h5::relative_name(name, &root.name())
}

/// Canonical JSON over the attributes of `names` --- an object's covered
/// attributes (§13.2) --- that it carries: sorted keys, arrays as nested
/// lists, floats in shortest round-trip form.
pub fn canonical_attrs(obj: &hdf5::Location, names: &[&str]) -> Result<String> {
    let present = attrs::names(obj)?;
    let mut wanted: Vec<&str> = names.iter().copied().filter(|n| present.iter().any(|p| p == n)).collect();
    wanted.sort_unstable();
    wanted.dedup();
    // Canonical JSON (§5.1) written from the values themselves, so a NaN or
    // infinity is spelled as 1.x hashed it rather than lost in a JSON value.
    let mut out = String::from("{");
    for (i, name) in wanted.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        crate::json::write_string(&mut out, name, false);
        out.push(':');
        attrs::read(obj, name)?.unwrap_or(AttrValue::Unsupported(String::new())).write_canonical(&mut out);
    }
    out.push('}');
    Ok(out)
}

/// Digest of an object's canonical attributes.
pub fn attrs_digest(obj: &hdf5::Location, names: &[&str], algo: &str) -> Result<String> {
    let mut hasher = Hasher::new(algo)?;
    hasher.update(canonical_attrs(obj, names)?.as_bytes());
    Ok(hasher.finish())
}

fn top_level(path: &str) -> &str {
    path.split('/').next().unwrap_or(path)
}

/// Write a `digest` attribute on every dataset under `root`.
///
/// `index/` is skipped (it carries `source_digest` instead, §13.3), and so is
/// a dataset whose type has no digest ([`digestible`]).  `only_missing`
/// digests just the datasets that carry none and returns the stored value
/// for the rest.
///
/// The datasets are hashed on the thread setting's workers
/// ([`crate::parallel::threads`]), and the attributes written here, in the
/// order of the walk: the file is the one a sequential pass writes.
pub fn stamp_digests(
    root: &hdf5::Group,
    algo: &str,
    skip: &[&str],
    only_missing: bool,
) -> Result<IndexMap<String, String>> {
    // Each dataset, with the digest it keeps, or `None` for one to hash; and
    // what stopped the walk, which stops the pass where the walk reached it.
    let mut walk: Vec<(String, hdf5::Dataset, Option<String>)> = Vec::new();
    let mut stopped = None;
    for (name, ds) in ops::datasets(root)? {
        if name == "meta" || skip.contains(&top_level(&name)) {
            continue;
        }
        let step = (|| -> Result<Option<Option<String>>> {
            if only_missing {
                if let Some(stored) = attrs::get_str(&ds, "digest")? {
                    return Ok(Some(Some(stored)));
                }
            }
            Ok(digestible(&ds)?.then_some(None))
        })();
        match step {
            Ok(Some(kept)) => walk.push((name, ds, kept)),
            Ok(None) => {}
            Err(e) => {
                stopped = Some(e);
                break;
            }
        }
    }
    let jobs = walk
        .iter()
        .filter(|(_, _, kept)| kept.is_none())
        .map(|(name, ds, _)| super::workers::Job::new(ds.clone(), name, algo))
        .collect();
    let mut hashed = super::workers::digests(jobs)?.into_iter();
    let mut digests = IndexMap::new();
    for (name, ds, kept) in walk {
        let value = match kept {
            Some(stored) => stored,
            None => {
                let value = hashed.next().expect("a digest for every dataset hashed")?;
                attrs::write(&ds, "digest", &AttrValue::Str(value.clone()))?;
                value
            }
        };
        digests.insert(name, value);
    }
    match stopped {
        Some(e) => Err(e),
        None => Ok(digests),
    }
}

/// The `digest` attribute of every dataset that carries one.
pub fn collect_digests(root: &hdf5::Group, skip: &[&str]) -> Result<IndexMap<String, String>> {
    let mut out = IndexMap::new();
    for (name, ds) in ops::datasets(root)? {
        if skip.contains(&top_level(&name)) {
            continue;
        }
        if let Some(value) = attrs::get_str(&ds, "digest")? {
            out.insert(name, value);
        }
    }
    Ok(out)
}

/// A digest over every dataset in a group --- an annotation's identity.
///
/// A mini-Merkle over the group's own datasets, sorted by name; datasets not
/// stamped yet are digested on the fly with the path [`stamp_digests`] would
/// use, so an index built before the commit is not stale right after it.
pub fn group_digest(group: &hdf5::Group, root: &hdf5::Group, algo: &str) -> Result<String> {
    let mut lines = String::new();
    for name in ops::members(group)? {
        let Some(ds) = ops::child_dataset(group, &name) else { continue };
        let value = match attrs::get_str(&ds, "digest")? {
            Some(v) => v,
            // Never stamped, so not part of the group's identity either.
            None if !digestible(&ds)? => continue,
            None => dataset_digest(&ds, &relative_path(&ds.name(), root), algo)?,
        };
        lines.push_str(&format!("{name}\t{value}\n"));
    }
    let mut hasher = Hasher::new(algo)?;
    hasher.update(lines.as_bytes());
    Ok(hasher.finish())
}

/// Object path -> the attribute names `content_id` covers (§13.2).
pub type AttrNameMap = IndexMap<String, Vec<&'static str>>;

/// The Merkle root over dataset digests, `meta` and canonical attributes.
pub fn compute_content_id(
    root: &hdf5::Group,
    attr_names: &AttrNameMap,
    algo: &str,
    digests: Option<&IndexMap<String, String>>,
) -> Result<String> {
    Hasher::new(algo)?;
    let owned;
    let digests = match digests {
        Some(d) => d,
        None => {
            owned = collect_digests(root, &["index"])?;
            &owned
        }
    };
    let mut dataset_lines: Vec<String> = digests.iter().map(|(p, v)| format!("{p}\t{v}\n")).collect();
    dataset_lines.sort();
    let meta = ops::child_dataset(root, "meta")
        .ok_or_else(|| Error::Key("\"Unable to synchronously open object (object 'meta' doesn't exist)\"".into()))?;
    // The document as stored: decoded as text, bytes that are not UTF-8
    // became U+FFFD, and different stored documents shared a root.  Valid
    // UTF-8 hashes as it always did.
    let meta_bytes = match data::read_scalar_string(&meta) {
        Ok(text) if !text.contains('\u{FFFD}') => text.into_bytes(),
        Ok(_) => data::read_stored_strings(&meta)?.into_iter().next().unwrap_or_default(),
        Err(e) => return Err(e),
    };
    let mut meta_hasher = Hasher::new(algo)?;
    meta_hasher.update(&meta_bytes);
    let meta_hash = meta_hasher.hexdigest();
    let mut attr_lines = Vec::new();
    for (path, names) in attr_names {
        if path.is_empty() {
            attr_lines.push(format!("@\t{}\n", attrs_digest(root, names, algo)?));
            continue;
        }
        let digest = match ops::node_kind(root, path) {
            Some(ops::NodeKind::Group) => {
                let g = root.group(path)?;
                attrs_digest(&g, names, algo)?
            }
            Some(ops::NodeKind::Dataset) => {
                let d = root.dataset(path)?;
                attrs_digest(&d, names, algo)?
            }
            _ => continue,
        };
        attr_lines.push(format!("@{path}\t{digest}\n"));
    }
    attr_lines.sort();
    let payload = format!("{}meta\t{meta_hash}\n{}", dataset_lines.concat(), attr_lines.concat());
    let mut root_hasher = Hasher::new(algo)?;
    root_hasher.update(payload.as_bytes());
    Ok(root_hasher.finish())
}

/// The digest algorithm a root declares, or the default.
pub fn root_algo(root: &hdf5::Group) -> Result<String> {
    Ok(attrs::get_str(root, "digest_algo")?.unwrap_or_else(|| DEFAULT_ALGO.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No block is larger than the budget however short the leading axis ---
    /// a row at a time read one channel of a channel-first image whole --- and
    /// the blocks are the array in C order, so the digest is the same for any
    /// budget.  A check of the stored values gets the same bounded blocks: it
    /// asked for whole rows, and the integrity level held a row of every
    /// dataset at once.
    #[test]
    fn s13_1_a_short_leading_axis_is_read_in_bounded_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("s.h5"))).unwrap();
        let data = ndarray::Array4::<f32>::from_shape_fn((2, 3, 16, 20), |(c, z, y, x)| {
            (c * 1000 + z * 100 + y * 20 + x) as f32
        });
        let ds = file.new_dataset_builder().with_data(&data).create("d").unwrap();
        let whole = array_digest("d", &NdArray::from(data.clone().into_dyn()), "sha256").unwrap();
        let shape = ds.shape();
        for budget in [4, 100, 1280, 4000, 1 << 20] {
            let mut blocks: Vec<NdArray> = Vec::new();
            for_each_block(&ds, &shape, 4, budget, &mut |block| {
                blocks.push(block);
                Ok(())
            })
            .unwrap();
            assert!(blocks.iter().all(|b| b.len() * 4 <= budget.max(4)), "budget {budget}");
            let flat: Vec<f32> =
                blocks.iter().flat_map(|b| b.cast::<f32>().iter().copied().collect::<Vec<_>>()).collect();
            assert_eq!(flat, data.iter().copied().collect::<Vec<_>>(), "budget {budget}");
            assert_eq!(dataset_digest_streamed(&ds, "d", "sha256", budget).unwrap(), whole, "budget {budget}");
        }
        let mut inspected: Vec<NdArray> = Vec::new();
        let each: Inspect<'_> = &mut |block: &NdArray| {
            inspected.push(block.clone());
            Ok(())
        };
        assert_eq!(dataset_digest_inspected(&ds, "d", "sha256", 100, Some(each)).unwrap(), whole);
        let shapes: Vec<Vec<usize>> = inspected.iter().map(NdArray::shape).collect();
        assert!(inspected.iter().all(|b| b.len() * 4 <= 100), "{shapes:?}");
        let flat: Vec<f32> =
            inspected.iter().flat_map(|b| b.cast::<f32>().iter().copied().collect::<Vec<_>>()).collect();
        assert_eq!(flat, data.iter().copied().collect::<Vec<_>>());
    }

    /// Slabs start and end on chunk boundaries: no slab ends inside a band
    /// of chunks the next one starts in, so no chunk is decompressed for two
    /// slabs, and none holds more than twice the budget.  The digest is the
    /// array's whatever the budget.
    #[test]
    fn s13_1_slabs_are_aligned_to_the_chunks_and_the_digest_is_unchanged() {
        for n in [1, 10, 16, 17] {
            for c in [1, 3, 4, 8] {
                for step in 1..=20 {
                    let slabs = Slabs::new(step, Some(c), 1, step);
                    let mut start = 0;
                    while start < n {
                        let stop = slabs.end(start, n);
                        assert!(stop > start && stop - start <= 2 * step, "n {n}, chunk {c}, step {step}");
                        let within = start / c == (stop - 1) / c;
                        let whole_bands = start % c == 0 && (stop % c == 0 || stop == n);
                        assert!(within || whole_bands, "n {n}, chunk {c}, step {step}: {start}..{stop}");
                        if c <= step {
                            assert!(whole_bands, "n {n}, chunk {c}, step {step}: {start}..{stop}");
                        }
                        start = stop;
                    }
                }
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("c.h5"))).unwrap();
        let data = ndarray::Array3::<f32>::from_shape_fn((10, 6, 8), |(z, y, x)| (z * 100 + y * 10 + x) as f32);
        let ds = file.new_dataset_builder().with_data(&data).chunk((4, 3, 8)).create("d").unwrap();
        let whole = array_digest("d", &NdArray::from(data.clone().into_dyn()), "sha256").unwrap();
        let row = 6 * 8 * 4;
        for budget in [4, 100, row, 3 * row, 4 * row, 5 * row, 7 * row, 9 * row, 1 << 20] {
            let mut rows = Vec::new();
            let mut flat: Vec<f32> = Vec::new();
            for_each_block(&ds, &[10, 6, 8], 4, budget, &mut |block| {
                if block.ndim() == 3 {
                    rows.push(block.shape()[0]);
                }
                flat.extend(block.cast::<f32>().iter().copied());
                Ok(())
            })
            .unwrap();
            assert_eq!(flat, data.iter().copied().collect::<Vec<_>>(), "budget {budget}");
            assert_eq!(dataset_digest_streamed(&ds, "d", "sha256", budget).unwrap(), whole, "budget {budget}");
            if budget >= 4 * row {
                // Whole bands of 4 rows, the last one short.
                let mut ends = rows.iter().scan(0, |at, n| {
                    *at += n;
                    Some(*at)
                });
                assert!(ends.all(|end| end % 4 == 0 || end == 10), "budget {budget}: {rows:?}");
            }
        }
    }

    /// A string hashes as stored: decoded as text, invalid UTF-8 became U+FFFD
    /// before hashing, so different stored bytes shared a digest.  A
    /// fixed-length string hashes without its padding, as the same text
    /// stored variable-length does.
    #[test]
    fn s13_1_a_string_hashes_its_stored_bytes() {
        use crate::h5sys::{h5d, h5p, h5s, h5t};
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("s.h5"))).unwrap();
        // Variable-length strings of any bytes, written through the C API:
        // the typed writers refuse invalid UTF-8.
        let vlen = |name: &str, values: &[&[u8]]| {
            let owned: Vec<std::ffi::CString> = values.iter().map(|v| std::ffi::CString::new(*v).unwrap()).collect();
            let pointers: Vec<*const std::os::raw::c_char> = owned.iter().map(|c| c.as_ptr()).collect();
            let cname = std::ffi::CString::new(name).unwrap();
            crate::h5::locked(|| unsafe {
                let tid = h5t::H5Tcopy(*h5t::H5T_C_S1);
                h5t::H5Tset_size(tid, h5t::H5T_VARIABLE);
                h5t::H5Tset_cset(tid, h5t::H5T_cset_t::H5T_CSET_UTF8);
                let dims = [values.len() as u64];
                let space = h5s::H5Screate_simple(1, dims.as_ptr(), std::ptr::null());
                let ds = h5d::H5Dcreate2(
                    file.id(),
                    cname.as_ptr(),
                    tid,
                    space,
                    h5p::H5P_DEFAULT,
                    h5p::H5P_DEFAULT,
                    h5p::H5P_DEFAULT,
                );
                assert!(ds >= 0);
                let status =
                    h5d::H5Dwrite(ds, tid, h5s::H5S_ALL, h5s::H5S_ALL, h5p::H5P_DEFAULT, pointers.as_ptr().cast());
                assert!(status >= 0);
                h5d::H5Dclose(ds);
                h5s::H5Sclose(space);
                h5t::H5Tclose(tid);
            });
            file.dataset(name).unwrap()
        };
        let e9 = vlen("e9", &[b"caf\xe9", b"ok"]);
        let e8 = vlen("e8", &[b"caf\xe8", b"ok"]);
        assert_ne!(dataset_digest(&e9, "x", "sha256").unwrap(), dataset_digest(&e8, "x", "sha256").unwrap());
        let mut hasher = Hasher::new("sha256").unwrap();
        feed_header(&mut hasher, "x", "|O", &[2]);
        hasher.update(b"caf\xe9\x00ok\x00");
        assert_eq!(dataset_digest(&e9, "x", "sha256").unwrap(), hasher.finish());

        let fixed = file.new_dataset::<hdf5::types::FixedAscii<6>>().shape([2]).create("fixed").unwrap();
        crate::h5::data::write_fixed_strings(&fixed, &["caf".into(), "ok".into()], 6, true).unwrap();
        let text = vlen("text", &[b"caf", b"ok"]);
        assert_eq!(dataset_digest(&fixed, "x", "sha256").unwrap(), dataset_digest(&text, "x", "sha256").unwrap());
        assert_eq!(
            dataset_digest(&fixed, "x", "sha256").unwrap(),
            strings_digest("x", &["caf".into(), "ok".into()], &[2], "sha256").unwrap()
        );
    }

    /// A compound hashes as 1.x hashed it through `h5py` --- NumPy's `|V<n>`
    /// over its stored elements --- where it was refused, so a file holding
    /// one could be neither amended nor verified.  Variable-length data other
    /// than strings has no digest.
    #[test]
    fn s13_1_a_compound_hashes_as_void_and_a_sequence_has_no_digest() {
        #[derive(hdf5::H5Type, Clone, Copy)]
        #[repr(C)]
        struct Row {
            id: i32,
            value: f32,
        }
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("c.h5"))).unwrap();
        let rows = [Row { id: 1, value: 0.5 }, Row { id: 2, value: -1.0 }];
        assert_eq!(std::mem::size_of::<Row>(), 8);
        let table = file.new_dataset_builder().with_data(&rows).create("table").unwrap();
        assert!(digestible(&table).unwrap());
        let size = std::mem::size_of::<Row>();
        let mut hasher = Hasher::new("sha256").unwrap();
        feed_header(&mut hasher, "x_acme/table", &format!("|V{size}"), &[2]);
        // SAFETY: two 4-byte fields, no padding: every byte is initialised,
        // and HDF5 stores the rows in this native layout.
        hasher.update(unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), size * 2) });
        let expected = hasher.finish();
        assert_eq!(dataset_digest(&table, "x_acme/table", "sha256").unwrap(), expected);
        assert_eq!(dataset_digest_streamed(&table, "x_acme/table", "sha256", 1).unwrap(), expected);

        let sequences = file.new_dataset::<hdf5::types::VarLenArray<u8>>().shape([2]).create("sequences").unwrap();
        assert!(!digestible(&sequences).unwrap());
        assert!(matches!(dataset_digest(&sequences, "x", "sha256"), Err(Error::Type(_))));
        let stamped = stamp_digests(&file.as_group().unwrap(), "sha256", &["index"], false).unwrap();
        assert!(stamped.contains_key("table") && !stamped.contains_key("sequences"), "{stamped:?}");
    }

    #[test]
    fn s13_1_streaming_matches_the_whole_array() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("s.h5"))).unwrap();
        let data = ndarray::Array2::<i16>::from_shape_fn((64, 64), |(i, j)| (i * 64 + j) as i16);
        let ds = file.new_dataset_builder().with_data(&data).create("d").unwrap();
        let whole = array_digest("d", &NdArray::from(data.into_dyn()), "sha256").unwrap();
        // 128 bytes is one row of 64 int16: one row per read.
        assert_eq!(dataset_digest_streamed(&ds, "d", "sha256", 128).unwrap(), whole);
        assert_eq!(dataset_digest_streamed(&ds, "d", "sha256", 1).unwrap(), whole);
        assert_eq!(dataset_digest(&ds, "d", "sha256").unwrap(), whole);
    }
}
