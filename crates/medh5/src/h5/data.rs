//! Datasets of any element type: create, read whole, read a region.
//!
//! Regions are read **in one call** (spec §14.5): `d[(k, *roi)]`, never
//! `d[k][roi]`, which materialises the whole sub-array first.

use std::sync::atomic::{AtomicU64, Ordering};

use hdf5::filters::Filter;
use hdf5::types::{TypeDescriptor as TD, VarLenAscii, VarLenUnicode};
use hdf5::{Hyperslab, SliceOrIndex};
use ndarray::{ArrayD, IxDyn};

use super::attrs::{bytes_to_array, numeric_dtype, read_fixed_strings, text};
use crate::array::{DType, Index, NdArray};
use crate::{with_array, with_dtype, Error, Result};

/// How a new dataset is laid out on disk.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Layout {
    /// Chunk shape; `None` stores the dataset contiguous.
    pub chunks: Option<Vec<usize>>,
    /// The filter pipeline, applied in order (chunked datasets only).
    pub filters: Vec<Filter>,
}

impl Layout {
    /// Contiguous, unfiltered.
    pub fn contiguous() -> Layout {
        Layout::default()
    }
}

// -- read accounting ------------------------------------------------------------

/// A read of a dataset's data, as [`read_stats`] counts it.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Read {
    /// A whole dataset ([`read`]).
    Whole,
    /// A region through the Blosc2 blocks it covers (the `window` module).
    Window,
    /// A region through HDF5's filter pipeline.
    Hyperslab,
    /// Rows as their stored bytes, for a digest or a comparison.
    Rows,
    /// Strings: a scalar, or rows of an array.
    Strings,
}

impl Read {
    /// Every kind, in the order [`read_stats`] reports them.
    pub const ALL: [Read; 5] = [Read::Whole, Read::Window, Read::Hyperslab, Read::Rows, Read::Strings];

    /// The kind's name.
    pub fn name(self) -> &'static str {
        match self {
            Read::Whole => "whole",
            Read::Window => "window",
            Read::Hyperslab => "hyperslab",
            Read::Rows => "rows",
            Read::Strings => "strings",
        }
    }
}

/// The reads of each kind since [`read_stats`] last took them.
///
/// A test holds a read path to its number of reads --- 1.x drew a patch in
/// 190 --- and a count kept in Python cannot see the engine's reads, so the
/// engine counts its own: one relaxed increment per read, nothing more.
static READS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

fn counted(read: Read) {
    READS[read as usize].fetch_add(1, Ordering::Relaxed);
}

/// The reads of each kind ([`Read::ALL`]) counted since the last call, in
/// every thread, and the counts set back to zero.  For tests: not an API.
#[doc(hidden)]
pub fn read_stats() -> [(Read, u64); 5] {
    Read::ALL.map(|read| (read, READS[read as usize].swap(0, Ordering::Relaxed)))
}

/// What a dataset stores.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// Numbers or booleans.
    Numeric(DType),
    /// Strings (variable- or fixed-length).
    Strings,
    /// Something else (compound, reference, ...).
    Other(String),
}

/// The element kind of a dataset.
pub fn kind(ds: &hdf5::Dataset) -> Result<Kind> {
    super::alive(ds)?;
    let td = ds.dtype()?.to_descriptor();
    Ok(match td {
        Ok(TD::VarLenUnicode | TD::VarLenAscii | TD::FixedAscii(_) | TD::FixedUnicode(_)) => Kind::Strings,
        Ok(td) => match numeric_dtype(&td) {
            Some(d) => Kind::Numeric(d),
            None => Kind::Other(format!("{td}")),
        },
        Err(e) => Kind::Other(e.to_string()),
    })
}

/// The numeric dtype of a dataset, or an error for strings and others.
pub fn dtype(ds: &hdf5::Dataset) -> Result<DType> {
    super::alive(ds)?;
    match kind(ds)? {
        Kind::Numeric(d) => Ok(d),
        Kind::Strings => Err(Error::Type(format!("{} holds strings, not numbers", ds.name()))),
        Kind::Other(t) => Err(Error::Type(format!("{} has an unsupported datatype {t}", ds.name()))),
    }
}

/// Whether `ds` is stored in exactly the type a fresh write of `dtype`
/// creates: that type, not committed.  An enumeration's names, a committed
/// type and another byte order are what a rebuild from `dtype` would lose.
pub fn stored_as_written(ds: &hdf5::Dataset, dtype: DType) -> Result<bool> {
    use crate::h5sys::{h5d, h5t};
    let committed = super::locked(|| unsafe {
        let tid = h5d::H5Dget_type(ds.id());
        let committed = tid >= 0 && h5t::H5Tcommitted(tid) > 0;
        if tid >= 0 {
            h5t::H5Tclose(tid);
        }
        committed
    });
    if committed {
        return Ok(false);
    }
    let stored = ds.dtype()?;
    Ok(with_dtype!(dtype, T => stored == hdf5::Datatype::from_type::<T>()?))
}

/// Whether a type is, or holds, variable-length data or a reference: what has
/// no bytes of its own in the dataset that stores it.  The caller holds the
/// HDF5 lock.
fn holds_variable(tid: crate::h5sys::h5i::hid_t) -> bool {
    use crate::h5sys::h5t::{self, H5T_class_t as Class};
    // SAFETY: `tid` is a live type id; each member or base type opened is
    // closed.
    unsafe {
        match h5t::H5Tget_class(tid) {
            Class::H5T_VLEN | Class::H5T_REFERENCE => true,
            Class::H5T_STRING => h5t::H5Tis_variable_str(tid) > 0,
            Class::H5T_COMPOUND => (0..h5t::H5Tget_nmembers(tid).max(0) as u32).any(|i| {
                let member = h5t::H5Tget_member_type(tid, i);
                let found = member < 0 || holds_variable(member);
                if member >= 0 {
                    h5t::H5Tclose(member);
                }
                found
            }),
            Class::H5T_ARRAY => {
                let base = h5t::H5Tget_super(tid);
                let found = base < 0 || holds_variable(base);
                if base >= 0 {
                    h5t::H5Tclose(base);
                }
                found
            }
            _ => false,
        }
    }
}

/// The element size, in bytes, of a dataset whose stored type has fixed-size
/// elements holding no variable-length data or reference --- a compound, an
/// array or an opaque type; `None` for a type that holds either.
pub fn stored_element_size(ds: &hdf5::Dataset) -> Result<Option<usize>> {
    use crate::h5sys::{h5d, h5t};
    super::alive(ds)?;
    super::locked(|| {
        // SAFETY: a live dataset; the type id is closed before returning.
        let tid = unsafe { h5d::H5Dget_type(ds.id()) };
        if tid < 0 {
            return Err(Error::Io(format!("could not read the type of {}", ds.name())));
        }
        let size = if holds_variable(tid) { None } else { Some(unsafe { h5t::H5Tget_size(tid) }) };
        unsafe { h5t::H5Tclose(tid) };
        Ok(size.filter(|n| *n > 0))
    })
}

/// Rows `[start, start + count)` of a dataset of fixed-size elements, along
/// its first axis with every other axis whole (the one element of a scalar),
/// as their stored bytes: read in the dataset's own type, so nothing is
/// converted.  `size` is the element size ([`stored_element_size`]).
pub(crate) fn read_stored_rows(ds: &hdf5::Dataset, size: usize, start: usize, count: usize) -> Result<Vec<u8>> {
    phase!(Read);
    use crate::h5sys::{h5d, h5p, h5s, h5t};
    let shape = ds.shape();
    let mut first = vec![0u64; shape.len()];
    let mut counts: Vec<u64> = shape.iter().map(|n| *n as u64).collect();
    if let Some(axis) = counts.first_mut() {
        first[0] = start as u64;
        *axis = count as u64;
    }
    let elements = counts.iter().try_fold(1usize, |n, c| n.checked_mul(usize::try_from(*c).ok()?));
    let mut buf = Vec::new();
    let bytes = elements
        .and_then(|n| n.checked_mul(size))
        .filter(|bytes| holdable(&[*bytes], 1) && buf.try_reserve_exact(*bytes).is_ok());
    let Some(bytes) = bytes else {
        return Err(Error::Io(format!(
            "{} cannot be read: rows {start}..{} are more than this process can hold",
            ds.name(),
            start + count
        )));
    };
    buf.resize(bytes, 0);
    super::locked(|| unsafe {
        let ftype = h5d::H5Dget_type(ds.id());
        let fspace = h5d::H5Dget_space(ds.id());
        let mut status = if ftype < 0 || fspace < 0 { -1 } else { 0 };
        let mspace = if shape.is_empty() {
            h5s::H5Screate(h5s::H5S_class_t::H5S_SCALAR)
        } else {
            if status >= 0 {
                status = h5s::H5Sselect_hyperslab(
                    fspace,
                    h5s::H5S_seloper_t::H5S_SELECT_SET,
                    first.as_ptr(),
                    std::ptr::null(),
                    counts.as_ptr(),
                    std::ptr::null(),
                );
            }
            h5s::H5Screate_simple(counts.len() as i32, counts.as_ptr(), std::ptr::null())
        };
        if status >= 0 && bytes > 0 {
            counted(Read::Rows);
            status = h5d::H5Dread(ds.id(), ftype, mspace, fspace, h5p::H5P_DEFAULT, buf.as_mut_ptr().cast());
        }
        h5s::H5Sclose(mspace);
        if fspace >= 0 {
            h5s::H5Sclose(fspace);
        }
        if ftype >= 0 {
            h5t::H5Tclose(ftype);
        }
        if status < 0 {
            return Err(Error::Io(format!("could not read {}", ds.name())));
        }
        Ok(buf)
    })
}

/// Whether a dataset holds strings.
pub fn is_strings(ds: &hdf5::Dataset) -> bool {
    matches!(kind(ds), Ok(Kind::Strings))
}

/// Whether a dataset is stored as an HDF5 enumeration other than the boolean
/// one: `h5py`'s `FALSE = 0, TRUE = 1` enum is read as a boolean (§2.5).
fn is_plain_enum(ds: &hdf5::Dataset) -> bool {
    matches!(ds.dtype().and_then(|t| t.to_descriptor()), Ok(TD::Enum(_)))
}

/// Reads at least this large reserve their buffer fallibly first.
const PROBE_BYTES: usize = 64 * 1024 * 1024;

/// The machine's physical memory, where the platform states it.
fn physical_memory() -> Option<usize> {
    #[cfg(unix)]
    {
        // SAFETY: `sysconf` reads a system constant and has no preconditions.
        let (pages, size) = unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
        if pages > 0 && size > 0 {
            return usize::try_from(pages).ok()?.checked_mul(usize::try_from(size).ok()?);
        }
    }
    None
}

/// Refuse a read of `extents` elements of `itemsize` bytes that no buffer
/// can hold, rather than let the allocation fail.
///
/// A file can declare a dataset of any extent and store none of it --- an
/// unallocated chunk reads as the fill value --- so a 10 KiB file asked the
/// allocator for 2**63 bytes, which panicked (`capacity overflow`), or for
/// 9 TB, which aborted the process.  The size is
/// computed with checked arithmetic, a buffer larger than the machine's
/// memory is refused outright --- where the system overcommits (macOS, say)
/// reserving one succeeds, and filling it is what kills the process --- and
/// a large one is reserved fallibly, so such a read is an error the caller
/// reports.
pub(crate) fn ensure_allocatable(ds: &hdf5::Dataset, extents: &[usize], itemsize: usize) -> Result<()> {
    ensure_holdable(&ds.name(), extents, itemsize)
}

/// [`ensure_allocatable`] for anything HDF5 reads into a buffer, named
/// `what` --- an attribute, say.
pub(crate) fn ensure_holdable(what: &str, extents: &[usize], itemsize: usize) -> Result<()> {
    if holdable(extents, itemsize) {
        return Ok(());
    }
    Err(Error::Io(format!(
        "{what} cannot be read: {} elements of {itemsize} bytes are more than this process can hold",
        extents.iter().map(usize::to_string).collect::<Vec<_>>().join(" x ")
    )))
}

/// Whether a buffer of `extents` elements of `itemsize` bytes can be held.
pub(crate) fn holdable(extents: &[usize], itemsize: usize) -> bool {
    if extents.contains(&0) {
        return true;
    }
    let bytes = extents.iter().try_fold(itemsize, |n, e| n.checked_mul(*e)).filter(|b| *b <= isize::MAX as usize);
    match bytes {
        Some(b) if b >= PROBE_BYTES => {
            physical_memory().is_none_or(|memory| b <= memory) && Vec::<u8>::new().try_reserve_exact(b).is_ok()
        }
        Some(_) => true,
        None => false,
    }
}

/// Read a whole numeric dataset.
pub fn read(ds: &hdf5::Dataset) -> Result<NdArray> {
    phase!(Read);
    super::alive(ds)?;
    let dtype = dtype(ds)?;
    ensure_allocatable(ds, &ds.shape(), dtype.itemsize())?;
    counted(Read::Whole);
    if is_plain_enum(ds) {
        return read_enum(ds, dtype);
    }
    Ok(with_dtype!(dtype, T => NdArray::from(ds.read_dyn::<T>()?)))
}

fn read_enum(ds: &hdf5::Dataset, dtype: DType) -> Result<NdArray> {
    phase!(Read);
    use crate::h5sys::{h5d, h5p, h5s, h5t};
    let shape = ds.shape();
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut buf = vec![0u8; n * dtype.itemsize()];
    super::locked(|| unsafe {
        let ftype = h5d::H5Dget_type(ds.id());
        let mtype = h5t::H5Tget_native_type(ftype, h5t::H5T_direction_t::H5T_DIR_ASCEND);
        let status =
            h5d::H5Dread(ds.id(), mtype, h5s::H5S_ALL, h5s::H5S_ALL, h5p::H5P_DEFAULT, buf.as_mut_ptr().cast());
        h5t::H5Tclose(mtype);
        h5t::H5Tclose(ftype);
        if status < 0 {
            return Err(Error::Io(format!("could not read {}", ds.name())));
        }
        Ok(())
    })?;
    bytes_to_array(&buf, dtype, &shape)
}

/// Read a hyperslab of an enum dataset as its base integer (or `bool`).
fn read_enum_region(
    ds: &hdf5::Dataset,
    dtype: DType,
    axes: &[(usize, usize, usize, bool)],
    out_shape: &[usize],
) -> Result<NdArray> {
    phase!(Read);
    use crate::h5sys::{h5d, h5p, h5s, h5t};
    let start: Vec<u64> = axes.iter().map(|a| a.0 as u64).collect();
    let count: Vec<u64> = axes.iter().map(|a| a.1 as u64).collect();
    let stride: Vec<u64> = axes.iter().map(|a| a.2 as u64).collect();
    let n: usize = count.iter().product::<u64>() as usize;
    let mut buf = vec![0u8; n.max(1) * dtype.itemsize()];
    super::locked(|| unsafe {
        let ftype = h5d::H5Dget_type(ds.id());
        let mtype = h5t::H5Tget_native_type(ftype, h5t::H5T_direction_t::H5T_DIR_ASCEND);
        let fspace = h5d::H5Dget_space(ds.id());
        let mut status = h5s::H5Sselect_hyperslab(
            fspace,
            h5s::H5S_seloper_t::H5S_SELECT_SET,
            start.as_ptr(),
            stride.as_ptr(),
            count.as_ptr(),
            std::ptr::null(),
        );
        let mspace = h5s::H5Screate_simple(count.len() as i32, count.as_ptr(), std::ptr::null());
        if status >= 0 {
            status = h5d::H5Dread(ds.id(), mtype, mspace, fspace, h5p::H5P_DEFAULT, buf.as_mut_ptr().cast());
        }
        h5s::H5Sclose(mspace);
        h5s::H5Sclose(fspace);
        h5t::H5Tclose(mtype);
        h5t::H5Tclose(ftype);
        if status < 0 {
            return Err(Error::Io(format!("could not read a region of {}", ds.name())));
        }
        Ok(())
    })?;
    let shape: Vec<usize> = count.iter().map(|c| *c as usize).collect();
    bytes_to_array(&buf, dtype, &shape)?.reshape(out_shape)
}

/// Resolve a selection against a shape: per-axis `(start, count, step)` and
/// whether the axis is kept in the output.
pub(crate) fn resolve(shape: &[usize], index: &[Index]) -> Result<Vec<(usize, usize, usize, bool)>> {
    if index.len() > shape.len() {
        return Err(Error::Index(format!(
            "too many indices: {} for a {}-dimensional dataset",
            index.len(),
            shape.len()
        )));
    }
    let mut out = Vec::with_capacity(shape.len());
    for (axis, n) in shape.iter().enumerate() {
        match index.get(axis).copied().unwrap_or(Index::Slice(crate::array::Slice::full())) {
            Index::Slice(s) => {
                let (start, stop, step) = s.resolve(*n)?;
                out.push((start, (stop - start).div_ceil(step), step, true));
            }
            Index::At(i) => {
                let k = if i < 0 { i + *n as i64 } else { i };
                if k < 0 || k >= *n as i64 {
                    return Err(Error::Index(format!("index {i} is out of range for axis {axis} with size {n}")));
                }
                out.push((k as usize, 1, 1, false));
            }
        }
    }
    Ok(out)
}

/// Read a region of a numeric dataset in one call.
pub fn read_region(ds: &hdf5::Dataset, index: &[Index]) -> Result<NdArray> {
    super::alive(ds)?;
    let dtype = dtype(ds)?;
    let shape = ds.shape();
    let axes = resolve(&shape, index)?;
    let out_shape: Vec<usize> = axes.iter().filter(|a| a.3).map(|a| a.1).collect();
    ensure_allocatable(ds, &axes.iter().map(|a| a.1).collect::<Vec<_>>(), dtype.itemsize())?;
    if axes.iter().all(|(start, _count, step, kept)| *kept && *start == 0 && *step == 1)
        && axes.iter().zip(&shape).all(|(a, n)| a.1 == *n)
    {
        return read(ds);
    }
    if axes.iter().any(|a| a.1 == 0) {
        return Ok(NdArray::zeros(dtype, &out_shape));
    }
    if is_plain_enum(ds) {
        counted(Read::Hyperslab);
        return read_enum_region(ds, dtype, &axes, &out_shape);
    }
    if let Some(window) = super::window::read_window(ds, dtype, &axes, &out_shape)? {
        counted(Read::Window);
        return Ok(window);
    }
    counted(Read::Hyperslab);
    read_hyperslab(ds, dtype, &axes, &out_shape)
}

/// A region as [`read_region_stored`] reads it.
pub(crate) enum Region {
    /// Its values, read as [`read_region`] reads them.
    Values(NdArray),
    /// The stored chunks [`read_region`] would decompress itself, for a
    /// worker to ([`super::window::Frames::decode`]); the axes read it
    /// through HDF5 ([`read_hyperslab`]) where that cannot.
    Stored(super::window::Frames, Vec<(usize, usize, usize, bool)>),
}

/// [`read_region`], but for a region it would read through the Blosc2 blocks
/// it covers: that one's stored chunks are returned as they are, so that the
/// blocks can be decompressed on another thread.
pub(crate) fn read_region_stored(ds: &hdf5::Dataset, index: &[Index]) -> Result<Region> {
    super::alive(ds)?;
    let dtype = dtype(ds)?;
    let shape = ds.shape();
    let axes = resolve(&shape, index)?;
    let out_shape: Vec<usize> = axes.iter().filter(|a| a.3).map(|a| a.1).collect();
    ensure_allocatable(ds, &axes.iter().map(|a| a.1).collect::<Vec<_>>(), dtype.itemsize())?;
    let enumeration = is_plain_enum(ds);
    if !enumeration && axes.iter().all(|a| a.1 > 0) {
        if let Some(frames) = super::window::read_frames(ds, dtype, &axes, &out_shape) {
            counted(Read::Window);
            return Ok(Region::Stored(frames, axes));
        }
    }
    // Where the blocks are not to be had, as `read_region` reads it.
    if axes.iter().all(|(start, _count, step, kept)| *kept && *start == 0 && *step == 1)
        && axes.iter().zip(&shape).all(|(a, n)| a.1 == *n)
    {
        return Ok(Region::Values(read(ds)?));
    }
    if axes.iter().any(|a| a.1 == 0) {
        return Ok(Region::Values(NdArray::zeros(dtype, &out_shape)));
    }
    if enumeration {
        counted(Read::Hyperslab);
        return Ok(Region::Values(read_enum_region(ds, dtype, &axes, &out_shape)?));
    }
    counted(Read::Hyperslab);
    Ok(Region::Values(read_hyperslab(ds, dtype, &axes, &out_shape)?))
}

/// The values of a [`Region::Stored`]'s axes, read through HDF5.
pub(crate) fn read_axes(ds: &hdf5::Dataset, axes: &[(usize, usize, usize, bool)]) -> Result<NdArray> {
    let out_shape: Vec<usize> = axes.iter().filter(|a| a.3).map(|a| a.1).collect();
    read_hyperslab(ds, dtype(ds)?, axes, &out_shape)
}

/// [`read_region`] through HDF5's filter pipeline: whole chunks, decompressed
/// under HDF5's lock.
pub(crate) fn read_hyperslab(
    ds: &hdf5::Dataset,
    dtype: DType,
    axes: &[(usize, usize, usize, bool)],
    out_shape: &[usize],
) -> Result<NdArray> {
    phase!(Read);
    let selection: Vec<SliceOrIndex> = axes
        .iter()
        .map(|(start, count, step, kept)| {
            if *kept {
                SliceOrIndex::SliceCount { start: *start, step: *step, count: *count, block: 1 }
            } else {
                SliceOrIndex::Index(*start)
            }
        })
        .collect();
    let hyper = Hyperslab::from(selection);
    Ok(with_dtype!(dtype, T => {
        let a: ArrayD<T> = ds.read_slice::<T, _, IxDyn>(hyper)?;
        NdArray::from(a.into_shape_with_order(IxDyn(out_shape))?)
    }))
}

/// Slice an in-memory array with the same semantics as [`read_region`].
pub fn slice_array(array: &NdArray, index: &[Index]) -> Result<NdArray> {
    let shape = array.shape();
    let axes = resolve(&shape, index)?;
    with_array!(array, a => {
        let mut view = a.view();
        // Apply slices first (keeping every axis), then drop indexed axes from
        // the highest down so positions stay valid.
        for (axis, (start, count, step, _)) in axes.iter().enumerate() {
            let stop = start + count.saturating_sub(1) * step + usize::from(*count > 0);
            let stop = if *count == 0 { *start } else { stop };
            view.slice_axis_inplace(
                ndarray::Axis(axis),
                ndarray::Slice::new(*start as isize, Some(stop as isize), *step as isize),
            );
        }
        let mut owned = view.to_owned();
        for (axis, (_, _, _, kept)) in axes.iter().enumerate().rev() {
            if !kept {
                owned = owned.index_axis_move(ndarray::Axis(axis), 0);
            }
        }
        Ok(NdArray::from(owned))
    })
}

/// What one decoded string costs beyond its text: the `String` itself, and
/// what the allocator spends on the smallest allocation it makes for the text.
pub(crate) const STRING_COST: usize = std::mem::size_of::<String>() + 16;

/// The stored width of a fixed-length string type; `None` for a
/// variable-length one, and an error for anything else.
fn string_width(ds: &hdf5::Dataset, td: &TD) -> Result<Option<usize>> {
    match td {
        TD::VarLenUnicode | TD::VarLenAscii => Ok(None),
        TD::FixedAscii(width) | TD::FixedUnicode(width) => Ok(Some((*width).max(1))),
        other => Err(Error::Type(format!("{} holds {other}, not strings", ds.name()))),
    }
}

/// Bytes of text reading rows `[start, start + count)` of a
/// variable-length string dataset allocates, measured before anything is
/// (`H5Dvlen_get_buf_size`).  Many elements may name one stored string, so
/// the text a read allocates is not bounded by the file's size.
///
/// A scalar holds one string, which the file stores: it is not measured
/// (HDF5 iterates a scalar selection without coordinates, which the
/// measurement dereferences).
fn vlen_text_bytes(ds: &hdf5::Dataset, rows: Option<(usize, usize)>) -> Result<usize> {
    use crate::h5sys::{h5d, h5s, h5t};
    let shape = ds.shape();
    if shape.is_empty() {
        return Ok(0);
    }
    super::locked(|| unsafe {
        let stored = h5d::H5Dget_type(ds.id());
        // The measurement reads as a read would, into memory's type.
        let ftype = h5t::H5Tget_native_type(stored, h5t::H5T_direction_t::H5T_DIR_DEFAULT);
        h5t::H5Tclose(stored);
        if ftype < 0 {
            return Err(Error::Io(format!("could not measure the strings of {}", ds.name())));
        }
        let fspace = h5d::H5Dget_space(ds.id());
        let mut status = 0;
        if let Some((start, count)) = rows {
            let mut first = vec![0u64; shape.len()];
            let mut counts: Vec<u64> = shape.iter().map(|n| *n as u64).collect();
            first[0] = start as u64;
            counts[0] = count as u64;
            status = h5s::H5Sselect_hyperslab(
                fspace,
                h5s::H5S_seloper_t::H5S_SELECT_SET,
                first.as_ptr(),
                std::ptr::null(),
                counts.as_ptr(),
                std::ptr::null(),
            );
        }
        let mut size: u64 = 0;
        if status >= 0 {
            status = h5d::H5Dvlen_get_buf_size(ds.id(), ftype, fspace, &mut size);
        }
        h5s::H5Sclose(fspace);
        h5t::H5Tclose(ftype);
        if status < 0 {
            return Err(Error::Io(format!("could not measure the strings of {}", ds.name())));
        }
        Ok(usize::try_from(size).unwrap_or(usize::MAX))
    })
}

/// Refuse a variable-length string that a contiguous dataset states is
/// longer than its file, before HDF5 reads any.
///
/// Each stored element states its string's length, up to 4 GiB, and names
/// where the string is.  The length is the file's to state, and a string the
/// file stores is no longer than the file, so a longer one is refused before
/// anything is reserved for it, as a stored chunk is ([`super::window`]).
///
/// The elements are read from the file, as stored, only where that is
/// exact: a contiguous dataset whose storage HDF5 has placed, in a file
/// opened for reading through the default driver.  A file this process is
/// writing may hold its newest bytes in HDF5's cache, and is not checked;
/// nor are chunked or compact datasets, whose elements HDF5 alone reads.
/// They are read through HDF5's own descriptor, not by name: the name may
/// since have been given to another file (`amend` replaces a file under its
/// open handles), or to none, while HDF5 reads on from the file it opened.
fn check_stored_lengths(ds: &hdf5::Dataset, n: usize) -> Result<()> {
    use crate::h5sys::{h5d, h5f, h5p};

    // Elements read per pass: 64 KiB of them, at most.
    const SLAB: usize = 4096;
    let Ok(file) = ds.file() else { return Ok(()) };
    if n == 0 {
        return Ok(());
    }
    let Some(stored) = super::file::descriptor(&file).and_then(Opened::borrow) else { return Ok(()) };
    let placed = super::locked(|| unsafe {
        let mut intent = 0u32;
        if h5f::H5Fget_intent(file.id(), &mut intent) < 0 || intent & h5f::H5F_ACC_RDWR != 0 {
            return None;
        }
        let offset = h5d::H5Dget_offset(ds.id());
        if offset == u64::MAX {
            return None;
        }
        let plist = h5f::H5Fget_create_plist(file.id());
        if plist < 0 {
            return None;
        }
        let (mut address, mut length) = (0usize, 0usize);
        let sized = h5p::H5Pget_sizes(plist, &mut address, &mut length);
        h5p::H5Pclose(plist);
        // An element is the string's length, then its heap object: the
        // collection's address and the object's index.
        (sized >= 0).then_some((offset, 4 + address + 4))
    });
    let Some((mut offset, element)) = placed else { return Ok(()) };
    let Ok(file_bytes) = stored.0.metadata().map(|m| m.len()) else { return Ok(()) };
    let mut slab = vec![0u8; SLAB.min(n) * element];
    let mut left = n;
    while left > 0 {
        let count = SLAB.min(left);
        let bytes = &mut slab[..count * element];
        // Storage the file stops short of is HDF5's to answer for, as it
        // answers for the rest of a damaged dataset.
        if stored.read_at(bytes, offset).is_err() {
            return Ok(());
        }
        offset += bytes.len() as u64;
        for stated in bytes.chunks_exact(element) {
            let length = u64::from(u32::from_le_bytes([stated[0], stated[1], stated[2], stated[3]]));
            if length > file_bytes {
                return Err(Error::Io(format!(
                    "{} cannot be read: it states a string of {length} bytes, and the file holds {file_bytes}",
                    ds.name()
                )));
            }
        }
        left -= count;
    }
    Ok(())
}

/// The file HDF5 opened, borrowed through its descriptor for reads that leave
/// HDF5's position in it as it was.  HDF5 owns the descriptor and closes it.
struct Opened(std::mem::ManuallyDrop<std::fs::File>);

impl Opened {
    #[cfg(unix)]
    fn borrow(fd: std::os::raw::c_int) -> Option<Self> {
        use std::os::unix::io::FromRawFd;
        // SAFETY: `fd` is HDF5's open descriptor, which outlives the borrow
        // (the dataset holds its file open), and is never closed here.
        Some(Self(std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(fd) })))
    }

    #[cfg(windows)]
    fn borrow(fd: std::os::raw::c_int) -> Option<Self> {
        use std::os::windows::io::FromRawHandle;
        let handle = super::file::os_handle(fd)?;
        // SAFETY: as on Unix; the handle is the one behind HDF5's descriptor.
        Some(Self(std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_handle(handle) })))
    }

    #[cfg(not(any(unix, windows)))]
    fn borrow(_fd: std::os::raw::c_int) -> Option<Self> {
        None
    }

    /// Fill `bytes` from `offset`.  A positioned read leaves the descriptor's
    /// offset alone.
    #[cfg(unix)]
    fn read_at(&self, bytes: &mut [u8], offset: u64) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.0.read_exact_at(bytes, offset)
    }

    /// Fill `bytes` from `offset`.  Windows has no read that leaves the file
    /// pointer where it was, and HDF5's `sec2` there skips its seek when it
    /// believes the pointer is where it left it; so, under HDF5's lock, the
    /// pointer is put back.
    #[cfg(windows)]
    fn read_at(&self, bytes: &mut [u8], offset: u64) -> std::io::Result<()> {
        use std::io::{Read, Seek, SeekFrom};
        super::locked(|| {
            let mut file: &std::fs::File = &self.0;
            let was = file.stream_position()?;
            let read = file.seek(SeekFrom::Start(offset)).and_then(|_| file.read_exact(bytes));
            file.seek(SeekFrom::Start(was))?;
            read
        })
    }

    #[cfg(not(any(unix, windows)))]
    fn read_at(&self, _bytes: &mut [u8], _offset: u64) -> std::io::Result<()> {
        Err(std::io::ErrorKind::Unsupported.into())
    }
}

/// The memory reading `n` strings costs once decoded: one `String` each and
/// its text, held twice while it is decoded --- as HDF5 read it and as text.
/// That is `text` bytes for variable-length strings (and a pointer each), and
/// the width for fixed-length ones.
fn decoded_bytes(n: usize, width: Option<usize>, text: usize) -> Option<usize> {
    let per = STRING_COST.checked_add(width.map_or(std::mem::size_of::<VarLenUnicode>(), |w| w.saturating_mul(2)))?;
    n.checked_mul(per)?.checked_add(text.checked_mul(2)?)
}

/// Read rows `[start, start + count)` of a string dataset along its first
/// axis, every other axis whole, flattened in C order: each string's stored
/// bytes, a fixed-length one's without its NUL padding.  The caller has
/// sized the read.
fn read_string_rows(ds: &hdf5::Dataset, td: &TD, start: usize, count: usize) -> Result<Vec<Vec<u8>>> {
    phase!(Read);
    let shape = ds.shape();
    let selection: Vec<SliceOrIndex> = shape
        .iter()
        .enumerate()
        .map(|(axis, n)| {
            let (first, many) = if axis == 0 { (start, count) } else { (0, *n) };
            SliceOrIndex::SliceCount { start: first, step: 1, count: many, block: 1 }
        })
        .collect();
    let hyper = Hyperslab::from(selection);
    let mut out = Vec::new();
    let n: usize = count.saturating_mul(shape[1..].iter().product::<usize>());
    out.try_reserve_exact(n)
        .map_err(|_| Error::Io(format!("{n} strings of {} are more than this process can hold", ds.name())))?;
    counted(Read::Strings);
    match td {
        TD::VarLenUnicode => {
            out.extend(ds.read_slice::<VarLenUnicode, _, IxDyn>(hyper)?.iter().map(|s| s.as_bytes().to_vec()))
        }
        TD::VarLenAscii => {
            out.extend(ds.read_slice::<VarLenAscii, _, IxDyn>(hyper)?.iter().map(|s| s.as_bytes().to_vec()))
        }
        _ => out.extend(super::attrs::read_fixed_string_rows(ds, &shape, start, count)?),
    }
    Ok(out)
}

/// Read a string dataset slab by slab along its first axis, handing each
/// slab's strings to `each` in C order, so that about `budget` bytes of
/// strings are held at a time --- what a digest needs, where it held every
/// string of the dataset at once.  Each string is its stored bytes, a
/// fixed-length one's without its NUL padding: a digest covers what is
/// stored, not text decoded from it (§13.1).
///
/// A string read costs an allocation beyond its bytes, so a few kilobytes
/// of one-byte strings, compressed, held hundreds of megabytes once read.
/// A slab is sized by what it reads to: variable-length text is measured
/// before it is read, and a slab that measures over the budget is halved.
/// A single row that no buffer can hold is refused, as any read is
/// ([`ensure_allocatable`]).
pub fn for_each_string_slab(
    ds: &hdf5::Dataset,
    budget: usize,
    each: &mut dyn FnMut(Vec<Vec<u8>>) -> Result<()>,
) -> Result<()> {
    super::alive(ds)?;
    let td = ds.dtype()?.to_descriptor()?;
    let width = string_width(ds, &td)?;
    let shape = ds.shape();
    if shape.is_empty() {
        return each(read_stored_strings(ds)?);
    }
    if shape.contains(&0) {
        return Ok(());
    }
    // A declared extent no machine holds is refused before it is measured,
    // as a whole read refuses it: a slab at a time, an 11 KiB file declaring
    // 2**58 strings was a read without end.
    ensure_allocatable(ds, &shape, width.unwrap_or(std::mem::size_of::<VarLenUnicode>()))?;
    if width.is_none() {
        check_stored_lengths(ds, shape.iter().product())?;
    }
    let row: usize = shape[1..].iter().try_fold(1usize, |n, e| n.checked_mul(*e)).unwrap_or(usize::MAX);
    let per_row = decoded_bytes(row, width, 0).unwrap_or(usize::MAX).max(1);
    let mut step = (budget / per_row).clamp(1, shape[0]);
    let mut start = 0;
    while start < shape[0] {
        let count = step.min(shape[0] - start);
        let text = match width {
            None => vlen_text_bytes(ds, Some((start, count)))?,
            Some(_) => 0,
        };
        let need = decoded_bytes(count.saturating_mul(row), width, text);
        if count > 1 && need.is_none_or(|b| b > budget) {
            step = count / 2;
            continue;
        }
        if !need.is_some_and(|b| holdable(&[b], 1)) {
            return Err(Error::Io(format!(
                "{} cannot be read: row {start} decodes to more than this process can hold",
                ds.name()
            )));
        }
        each(read_string_rows(ds, &td, start, count)?)?;
        start += count;
    }
    Ok(())
}

/// Read a string dataset (any shape) as text, flattened in C order; invalid
/// UTF-8 is replaced rather than trusted.
pub fn read_strings(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    Ok(read_stored_strings(ds)?.into_iter().map(text).collect())
}

/// Read a string dataset (any shape), flattened in C order: each string's
/// stored bytes, a fixed-length one's without its NUL padding.
///
/// Sized by what it reads to, before anything is read: a fixed-length string
/// type declares any width and a dataspace any extent, and an 11 KiB file of
/// 2**59 16-byte strings panicked the validator on a capacity overflow, where
/// a 1 GiB scalar string aborted a capped process.  Each string read costs an
/// allocation beyond its stored bytes, and variable-length text is measured
/// first, since many elements can name one stored string.
pub fn read_stored_strings(ds: &hdf5::Dataset) -> Result<Vec<Vec<u8>>> {
    super::alive(ds)?;
    let td = ds.dtype()?.to_descriptor()?;
    let width = string_width(ds, &td)?;
    let extents = if ds.is_scalar() { vec![1] } else { ds.shape() };
    let n = extents.iter().try_fold(1usize, |n, e| n.checked_mul(*e));
    // The element count is checked against the stored width first, as it
    // always was, so that a declared extent no machine holds is refused
    // before it is measured.
    ensure_allocatable(ds, &extents, width.unwrap_or(std::mem::size_of::<VarLenUnicode>()))?;
    if let (None, Some(n)) = (width, n) {
        check_stored_lengths(ds, n)?;
    }
    let text = match width {
        None if n.is_some_and(|n| n > 0) => vlen_text_bytes(ds, None)?,
        _ => 0,
    };
    if !n.and_then(|n| decoded_bytes(n, width, text)).is_some_and(|b| holdable(&[b], 1)) {
        return Err(Error::Io(format!(
            "{} cannot be read: {} strings decode to more than this process can hold",
            ds.name(),
            extents.iter().map(usize::to_string).collect::<Vec<_>>().join(" x ")
        )));
    }
    if ds.is_scalar() {
        counted(Read::Strings);
    }
    Ok(match td {
        TD::VarLenUnicode if ds.is_scalar() => {
            ds.read_raw::<VarLenUnicode>()?.iter().map(|s| s.as_bytes().to_vec()).collect()
        }
        TD::VarLenAscii if ds.is_scalar() => {
            ds.read_raw::<VarLenAscii>()?.iter().map(|s| s.as_bytes().to_vec()).collect()
        }
        TD::FixedAscii(_) | TD::FixedUnicode(_) if ds.is_scalar() => read_fixed_strings(ds.id(), 1, false)?,
        _ if extents.contains(&0) => Vec::new(),
        _ => read_string_rows(ds, &td, 0, extents[0])?,
    })
}

/// Read a scalar string dataset (`/meta`, a table's descriptor, a cache's
/// manifest).
///
/// One string, so a dataset of any other shape is refused rather than read
/// whole and cut to its first element.
pub fn read_scalar_string(ds: &hdf5::Dataset) -> Result<String> {
    phase!(Read);
    if !ds.is_scalar() {
        return Err(Error::Value(format!(
            "{} is a dataset of shape {}, not one string",
            ds.name(),
            crate::json::repr_int_tuple(&ds.shape())
        )));
    }
    let mut values = read_strings(ds)?;
    if values.is_empty() {
        return Err(Error::Value(format!("{} holds no string", ds.name())));
    }
    Ok(values.swap_remove(0))
}

/// Create a numeric dataset holding `data`.
pub fn create(group: &hdf5::Group, name: &str, data: &NdArray, layout: &Layout) -> Result<hdf5::Dataset> {
    phase!(Write);
    let chunks = layout.chunks.clone();
    with_array!(data, a => {
        let mut builder = group.new_dataset_builder().obj_track_times(false).with_data(a.view());
        if let Some(c) = &chunks {
            builder = builder.chunk(c.clone()).set_filters(&layout.filters);
        }
        Ok(builder.create(name)?)
    })
}

/// Create an empty numeric dataset to fill region by region.
pub fn create_empty(
    group: &hdf5::Group,
    name: &str,
    dtype: DType,
    shape: &[usize],
    layout: &Layout,
) -> Result<hdf5::Dataset> {
    phase!(Metadata);
    let chunks = layout.chunks.clone();
    with_dtype!(dtype, T => {
        let mut builder = group.new_dataset::<T>().obj_track_times(false).shape(shape.to_vec());
        if let Some(c) = &chunks {
            builder = builder.chunk(c.clone()).set_filters(&layout.filters);
        }
        Ok(builder.create(name)?)
    })
}

/// Write `data` into a region of an existing dataset (slices only).
pub fn write_region(ds: &hdf5::Dataset, data: &NdArray, starts: &[usize]) -> Result<()> {
    phase!(Write, data.nbytes());
    let shape = data.shape();
    let selection: Vec<SliceOrIndex> = starts
        .iter()
        .zip(&shape)
        .map(|(s, n)| SliceOrIndex::SliceCount { start: *s, step: 1, count: *n, block: 1 })
        .collect();
    if shape.contains(&0) {
        return Ok(());
    }
    let hyper = Hyperslab::from(selection);
    with_array!(data, a => ds.write_slice(a.view(), hyper)?);
    Ok(())
}

/// Create a 1-D variable-length UTF-8 string dataset.
pub fn create_strings(group: &hdf5::Group, name: &str, values: &[String]) -> Result<hdf5::Dataset> {
    phase!(Write);
    let parsed: Vec<VarLenUnicode> = values
        .iter()
        .map(|s| s.parse::<VarLenUnicode>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| Error::Value(format!("{e}")))?;
    let arr = ndarray::Array1::from(parsed);
    Ok(group.new_dataset_builder().obj_track_times(false).with_data(arr.view()).create(name)?)
}

/// Create a scalar variable-length UTF-8 string dataset.
pub fn create_scalar_string(group: &hdf5::Group, name: &str, value: &str) -> Result<hdf5::Dataset> {
    phase!(Write);
    let parsed: VarLenUnicode = value.parse().map_err(|e| Error::Value(format!("{e}")))?;
    let ds = group.new_dataset::<VarLenUnicode>().obj_track_times(false).shape(()).create(name)?;
    ds.write_scalar(&parsed)?;
    Ok(ds)
}

/// The dataset's chunk shape, or `None` when contiguous.
pub fn chunks(ds: &hdf5::Dataset) -> Option<Vec<usize>> {
    ds.chunk()
}

/// The dataset's filter pipeline, as `(id, client data)` pairs, every
/// client value included.
pub fn filters(ds: &hdf5::Dataset) -> Result<Vec<(i32, Vec<u32>)>> {
    use crate::h5sys::{h5d, h5p};
    super::locked(|| unsafe {
        let plist = h5d::H5Dget_create_plist(ds.id());
        if plist < 0 {
            return Err(Error::Io(format!("could not read the creation properties of {}", ds.name())));
        }
        // `nelmts` goes in as the buffer's capacity and comes back as the
        // filter's full count, which may be larger: then ask again with room
        // for all of them.
        let filter = |i: u32, values: &mut Vec<u32>| -> Option<(i32, usize)> {
            let (mut flags, mut config, mut nelmts) = (0u32, 0u32, values.len());
            let id = h5p::H5Pget_filter2(
                plist,
                i,
                &mut flags,
                &mut nelmts,
                values.as_mut_ptr(),
                0,
                std::ptr::null_mut(),
                &mut config,
            );
            (id >= 0).then_some((id, nelmts))
        };
        let n = h5p::H5Pget_nfilters(plist);
        let mut out = Vec::new();
        for i in 0..n.max(0) as u32 {
            let mut values = vec![0u32; 32];
            let Some((id, mut count)) = filter(i, &mut values) else {
                h5p::H5Pclose(plist);
                return Err(Error::Io(format!("could not read filter {i} of {}", ds.name())));
            };
            if count > values.len() {
                values = vec![0u32; count];
                let Some((_, again)) = filter(i, &mut values) else {
                    h5p::H5Pclose(plist);
                    return Err(Error::Io(format!("could not read filter {i} of {}", ds.name())));
                };
                count = again;
            }
            values.truncate(count);
            out.push((id, values));
        }
        h5p::H5Pclose(plist);
        Ok(out)
    })
}

/// The number of bytes the dataset occupies when decompressed.
///
/// The shape is the file's to state, and its product with the element size
/// need not fit a machine word: such a size is an error, where it wrapped
/// round (or, in a debug build, panicked).
pub fn nbytes(ds: &hdf5::Dataset) -> Result<usize> {
    let itemsize = match kind(ds)? {
        Kind::Numeric(d) => d.itemsize(),
        // h5py reads variable-length strings as an object array: one
        // pointer per element, as NumPy counts it.
        Kind::Strings if matches!(ds.dtype()?.to_descriptor(), Ok(TD::VarLenUnicode | TD::VarLenAscii)) => 8,
        _ => ds.dtype()?.size(),
    };
    let shape = ds.shape();
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(itemsize, |n, e| n.checked_mul(*e)).ok_or_else(|| {
        Error::Io(format!(
            "{} holds {} elements of {itemsize} bytes, more bytes than this process can count",
            ds.name(),
            shape.iter().map(usize::to_string).collect::<Vec<_>>().join(" x ")
        ))
    })
}

/// A fixed-length string type: NUL-padded, `width` bytes, ASCII or UTF-8 ---
/// what h5py creates for NumPy's `S` dtype.
fn fixed_string_type(width: usize, ascii: bool) -> Result<crate::h5sys::h5i::hid_t> {
    use crate::h5sys::h5t;
    // SAFETY: plain HDF5 type construction; the caller closes the type.
    let tid = unsafe { h5t::H5Tcopy(*h5t::H5T_C_S1) };
    if tid < 0 {
        return Err(Error::Io("could not create a string type".into()));
    }
    let cset = if ascii { h5t::H5T_cset_t::H5T_CSET_ASCII } else { h5t::H5T_cset_t::H5T_CSET_UTF8 };
    // SAFETY: `tid` is a valid, writable string type.
    let ok = unsafe {
        h5t::H5Tset_size(tid, width.max(1)) >= 0
            && h5t::H5Tset_strpad(tid, h5t::H5T_str_t::H5T_STR_NULLPAD) >= 0
            && h5t::H5Tset_cset(tid, cset) >= 0
    };
    if !ok {
        // SAFETY: `tid` is valid.
        unsafe { h5t::H5Tclose(tid) };
        return Err(Error::Io("could not configure a string type".into()));
    }
    Ok(tid)
}

/// What [`write_fixed_strings`] and [`recreate_fixed_strings`] hand HDF5 must
/// be what it reads: both write through `H5S_ALL`, which takes one
/// `width`-byte element from the buffer for every element of the dataset,
/// whatever length the buffer has.  A shorter buffer is a read past its end.
fn check_fixed_strings(ds: &hdf5::Dataset, values: &[String], width: usize) -> Result<()> {
    super::alive(ds)?;
    if !is_strings(ds) {
        return Err(Error::Type(format!("{} does not hold strings", ds.name())));
    }
    if width == 0 {
        return Err(Error::Value("a fixed-length string is at least one byte wide".into()));
    }
    // Counted as `read_strings` counts them: a scalar holds one.
    let n = ds.size().max(usize::from(ds.is_scalar()));
    if values.len() != n {
        return Err(Error::Value(format!(
            "{} holds {n} string(s) and {} were given: every element is rewritten",
            ds.name(),
            values.len()
        )));
    }
    if n.checked_mul(width).is_none() {
        return Err(Error::Value(format!("{n} strings of {width} bytes do not fit in memory")));
    }
    Ok(())
}

fn fixed_bytes(values: &[String], width: usize) -> Vec<u8> {
    let mut buf = vec![0u8; values.len() * width.max(1)];
    for (i, value) in values.iter().enumerate() {
        let bytes = value.as_bytes();
        let n = bytes.len().min(width);
        buf[i * width..i * width + n].copy_from_slice(&bytes[..n]);
    }
    buf
}

/// Overwrite a fixed-length string dataset whose width fits `values`, one
/// value per element of the dataset.
pub fn write_fixed_strings(ds: &hdf5::Dataset, values: &[String], width: usize, ascii: bool) -> Result<()> {
    phase!(Write);
    use crate::h5sys::{h5d, h5p, h5s, h5t};
    check_fixed_strings(ds, values, width)?;
    let buf = fixed_bytes(values, width);
    super::locked(|| -> Result<()> {
        let mtype = fixed_string_type(width, ascii)?;
        // SAFETY: `buf` holds one `width`-byte element per selected value.
        let status =
            unsafe { h5d::H5Dwrite(ds.id(), mtype, h5s::H5S_ALL, h5s::H5S_ALL, h5p::H5P_DEFAULT, buf.as_ptr().cast()) };
        // SAFETY: `mtype` was created above.
        unsafe { h5t::H5Tclose(mtype) };
        if status < 0 {
            return Err(Error::Io(format!("could not rewrite {}", ds.name())));
        }
        Ok(())
    })
}

/// Recreate a fixed-length string dataset wider, keeping its creation
/// property list (filters, chunking, fill value) and its attributes.
///
/// Built under a temporary name and swapped into place ([`swap_in`]), so a
/// pipeline HDF5 cannot rebuild for the new type, or a failure at any later
/// step, leaves the original where it was and no temporary behind.  One value
/// per element of the dataset, as [`write_fixed_strings`].
pub fn recreate_fixed_strings(ds: &hdf5::Dataset, values: &[String], width: usize, ascii: bool) -> Result<()> {
    phase!(Write);
    use crate::h5sys::{h5d, h5i, h5p, h5s, h5t};
    use std::ffi::CString;
    check_fixed_strings(ds, values, width)?;
    let full = ds.name();
    let (parent_path, name) = match full.rsplit_once('/') {
        Some((p, n)) => (if p.is_empty() { "/".to_string() } else { p.to_string() }, n.to_string()),
        None => ("/".to_string(), full.clone()),
    };
    let file = ds.file()?;
    let parent = file.group(&parent_path)?;
    let suffix: String = {
        let mut rng = crate::rng::Rng::from_entropy();
        format!("{:08x}", rng.next_u32())
    };
    let staging = format!("{name}.scrub-{suffix}");
    let chunks = ds.chunk();
    let buf = fixed_bytes(values, width);
    let created = super::locked(|| -> Result<h5i::hid_t> {
        let ftype = fixed_string_type(width, ascii)?;
        // SAFETY: valid ids from the open dataset; every id is closed below.
        unsafe {
            let dcpl = h5d::H5Dget_create_plist(ds.id());
            if let Some(c) = &chunks {
                let dims: Vec<u64> = c.iter().map(|v| *v as u64).collect();
                h5p::H5Pset_chunk(dcpl, dims.len() as i32, dims.as_ptr());
            }
            let space = h5d::H5Dget_space(ds.id());
            let cname = CString::new(staging.as_str()).map_err(|e| Error::Value(e.to_string()))?;
            let id =
                h5d::H5Dcreate2(parent.id(), cname.as_ptr(), ftype, space, h5p::H5P_DEFAULT, dcpl, h5p::H5P_DEFAULT);
            h5s::H5Sclose(space);
            h5p::H5Pclose(dcpl);
            if id < 0 {
                h5t::H5Tclose(ftype);
                return Err(Error::Io(format!("could not recreate {full} with a wider string type")));
            }
            let status = h5d::H5Dwrite(id, ftype, h5s::H5S_ALL, h5s::H5S_ALL, h5p::H5P_DEFAULT, buf.as_ptr().cast());
            h5t::H5Tclose(ftype);
            if status < 0 {
                h5d::H5Dclose(id);
                return Err(Error::Io(format!("could not write the widened {full}")));
            }
            Ok(id)
        }
    });
    let created = match created {
        Ok(id) => id,
        Err(err) => {
            // Linked when it was created, whatever happened after.
            let _ = super::ops::unlink(&parent, &staging);
            return Err(err);
        }
    };
    // SAFETY: `created` is a dataset id we own; closing it releases it.
    super::locked(|| unsafe { h5d::H5Dclose(created) });
    let copied = (|| -> Result<()> {
        let replacement = parent.dataset(&staging)?;
        for key in super::attrs::names(ds)? {
            super::attrs::copy_raw(ds, &replacement, &key)?;
        }
        Ok(())
    })();
    if let Err(err) = copied {
        let _ = super::ops::unlink(&parent, &staging);
        return Err(err);
    }
    swap_in(&parent, &name, &staging, &format!("{name}.orig-{suffix}"), &move_link)
}

/// Rename the link `from` in `group` to `to`.
fn move_link(group: &hdf5::Group, from: &str, to: &str) -> Result<()> {
    use crate::h5sys::{h5l, h5p};
    use std::ffi::CString;
    let source = CString::new(from).map_err(|e| Error::Value(e.to_string()))?;
    let target = CString::new(to).map_err(|e| Error::Value(e.to_string()))?;
    // SAFETY: both names are valid C strings in the same live group.
    let status = super::locked(|| unsafe {
        h5l::H5Lmove(group.id(), source.as_ptr(), group.id(), target.as_ptr(), h5p::H5P_DEFAULT, h5p::H5P_DEFAULT)
    });
    if status < 0 {
        return Err(Error::Io(format!(
            "could not move {} to {} in {}",
            crate::json::repr_str(from),
            crate::json::repr_str(to),
            group.name()
        )));
    }
    Ok(())
}

/// Put the dataset linked as `staging` in `parent` in place of `name`.
///
/// The original is moved aside first and unlinked last, so a failure at any
/// step leaves it where it was and `staging` gone; it was unlinked first, and
/// a move that then failed lost it.  `mv` renames a link ([`move_link`]).
fn swap_in(
    parent: &hdf5::Group,
    name: &str,
    staging: &str,
    aside: &str,
    mv: &dyn Fn(&hdf5::Group, &str, &str) -> Result<()>,
) -> Result<()> {
    if let Err(err) = mv(parent, name, aside) {
        let _ = super::ops::unlink(parent, staging);
        return Err(err);
    }
    if let Err(err) = mv(parent, staging, name) {
        let _ = super::ops::unlink(parent, staging);
        return match mv(parent, aside, name) {
            Ok(()) => Err(err),
            Err(_) => Err(Error::Io(format!(
                "{err}; the original of {} could not be put back and is left as {}",
                crate::json::repr_str(name),
                crate::json::repr_str(aside)
            ))),
        };
    }
    super::ops::unlink(parent, aside)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `H5Pget_filter2` fills at most the capacity it is given and returns
    /// the full count: a filter with more than 32 client values lost the
    /// rest.
    #[test]
    fn filters_returns_every_client_value() {
        use crate::h5sys::{h5d, h5p, h5s, h5t, h5z};
        super::super::init();
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("f.h5")).unwrap();
        let values: Vec<u32> = (0..40).map(|v| v * 3 + 1).collect();
        let dims = [4u64, 4];
        let id = super::super::locked(|| unsafe {
            let dcpl = h5p::H5Pcreate(*h5p::H5P_CLS_DATASET_CREATE);
            h5p::H5Pset_chunk(dcpl, 2, dims.as_ptr());
            // An optional filter no library registers, as h5py records one it
            // has not got.
            h5p::H5Pset_filter(dcpl, 40000, h5z::H5Z_FLAG_OPTIONAL, values.len(), values.as_ptr());
            let space = h5s::H5Screate_simple(2, dims.as_ptr(), std::ptr::null());
            let id = h5d::H5Dcreate2(
                file.id(),
                c"x".as_ptr(),
                *h5t::H5T_NATIVE_UCHAR,
                space,
                h5p::H5P_DEFAULT,
                dcpl,
                h5p::H5P_DEFAULT,
            );
            h5s::H5Sclose(space);
            h5p::H5Pclose(dcpl);
            h5d::H5Dclose(id)
        });
        assert!(id >= 0);
        let found = filters(&file.dataset("x").unwrap()).unwrap();
        assert_eq!(found, vec![(40000, values)]);
    }

    fn fixed(n: usize) -> (tempfile::TempDir, hdf5::File, hdf5::Dataset) {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("s.h5"))).unwrap();
        let ds = file.new_dataset::<hdf5::types::FixedAscii<8>>().shape([n]).create("s").unwrap();
        (dir, file, ds)
    }

    /// Both writers hand HDF5 a buffer through `H5S_ALL`, which reads one
    /// element per element of the dataset whatever the buffer holds.  Fewer
    /// values read past the buffer (none at all was a segfault); more were
    /// silently dropped.
    #[test]
    fn s2_5_fixed_strings_take_one_value_per_element() {
        let (_dir, _file, ds) = fixed(3);
        let values: Vec<String> = ["a", "bb", "ccc", "d"].iter().map(|s| s.to_string()).collect();
        for wrong in [&values[..0], &values[..1], &values[..]] {
            assert!(write_fixed_strings(&ds, wrong, 8, true).is_err(), "{} values", wrong.len());
            assert!(recreate_fixed_strings(&ds, wrong, 16, true).is_err(), "{} values", wrong.len());
        }
        assert!(write_fixed_strings(&ds, &values[..3], 0, true).is_err(), "a zero width");
        write_fixed_strings(&ds, &values[..3], 8, true).unwrap();
        assert_eq!(read_strings(&ds).unwrap(), values[..3]);
    }

    /// The element count is checked against a dataset of strings only.
    #[test]
    fn s2_5_only_strings_are_rewritten_as_strings() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("n.h5"))).unwrap();
        let ds = file.new_dataset::<u8>().shape([1]).create("n").unwrap();
        let err = write_fixed_strings(&ds, &["x".to_string()], 8, true).unwrap_err();
        assert!(err.to_string().contains("does not hold strings"), "{err}");
    }

    /// A read is refused before anything is reserved when it overflows or
    /// exceeds the machine's memory: where the system overcommits, a
    /// reservation of terabytes succeeds and filling it killed the process
    /// (on macOS).
    #[test]
    fn s14_5_a_read_no_machine_can_hold_is_refused_before_it_is_reserved() {
        assert!(holdable(&[4, 4], 8));
        assert!(holdable(&[0, usize::MAX], 8), "empty, whatever the other extents");
        assert!(!holdable(&[usize::MAX, 2], 8), "overflows");
        assert!(!holdable(&[1 << 62], 2), "beyond isize::MAX");
        if let Some(memory) = physical_memory() {
            assert!(!holdable(&[memory, 2], 1), "twice the machine's memory");
        }
    }

    /// A string read is sized as a numeric one.  Its buffer was
    /// `vec![0; width * n]`, unchecked: 2**59 strings of 16 bytes, declared
    /// and never stored, panicked on a capacity overflow.
    #[test]
    fn s14_5_a_string_read_no_machine_can_hold_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("big.h5"))).unwrap();
        let fixed = file
            .new_dataset::<hdf5::types::FixedAscii<16>>()
            .shape([1usize << 59])
            .chunk([1024])
            .create("fixed")
            .unwrap();
        let vlen = file
            .new_dataset::<hdf5::types::VarLenUnicode>()
            .shape([1usize << 61])
            .chunk([1024])
            .create("vlen")
            .unwrap();
        for ds in [&fixed, &vlen] {
            let err = read_strings(ds).unwrap_err();
            assert!(err.to_string().contains("more than this process can hold"), "{err}");
            // A slab at a time too: refused before the first, not read
            // without end.
            let err = for_each_string_slab(ds, 1 << 20, &mut |_| Ok(())).unwrap_err();
            assert!(err.to_string().contains("more than this process can hold"), "{err}");
        }
        let small = file.new_dataset::<hdf5::types::FixedAscii<8>>().shape([2]).create("small").unwrap();
        let values = ["a".to_string(), "bc".to_string()];
        write_fixed_strings(&small, &values, 8, true).unwrap();
        assert_eq!(read_strings(&small).unwrap(), values);
        // One string is a scalar: a vector is refused, not cut to its first.
        let err = read_scalar_string(&small).unwrap_err();
        assert!(err.to_string().contains("not one string"), "{err}");
    }

    /// A dataset's size is the file's to state, and one past what a machine
    /// word counts is an error: it wrapped round (or panicked, in a debug
    /// build).
    #[test]
    fn s14_5_a_size_past_a_machine_word_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("n.h5"))).unwrap();
        let huge = file.new_dataset::<u64>().shape([1usize << 61, 4]).chunk([1, 4]).create("huge").unwrap();
        let err = nbytes(&huge).unwrap_err();
        assert!(err.to_string().contains("more bytes than this process can count"), "{err}");
        let empty = file.new_dataset::<u64>().shape([0usize, 1 << 62]).chunk([1, 1]).create("empty").unwrap();
        assert_eq!(nbytes(&empty).unwrap(), 0);
        let small = file.new_dataset::<u16>().shape([3usize, 5]).create("small").unwrap();
        assert_eq!(nbytes(&small).unwrap(), 30);
    }

    /// A widened dataset is swapped in with the original moved aside first
    /// and unlinked last: a step that fails leaves the original under its
    /// name, holding what it held, and nothing temporary behind.  The
    /// original was unlinked first, and a move that then failed lost it.
    #[test]
    fn s2_5_a_failed_swap_leaves_the_original_in_place() {
        let (_dir, file, ds) = fixed(2);
        let before: Vec<String> = vec!["a".into(), "b".into()];
        write_fixed_strings(&ds, &before, 8, true).unwrap();
        let group = file.as_group().unwrap();
        let fails_at = |n: usize| {
            let calls = std::cell::Cell::new(0);
            move |group: &hdf5::Group, from: &str, to: &str| {
                calls.set(calls.get() + 1);
                if calls.get() == n {
                    return Err(Error::Io(format!("move {n} refused")));
                }
                move_link(group, from, to)
            }
        };
        for n in [1, 2] {
            create_strings(&group, "s.scrub-x", &["c".into(), "d".into()]).unwrap();
            let err = swap_in(&group, "s", "s.scrub-x", "s.orig-x", &fails_at(n)).unwrap_err();
            assert!(err.to_string().contains(&format!("move {n} refused")), "{err}");
            assert_eq!(members(&group), vec!["s".to_string()], "after a failed move {n}");
            assert_eq!(read_strings(&file.dataset("s").unwrap()).unwrap(), before);
        }
        // Widened for real: the values and the attributes, under the name.
        crate::h5::attrs::write(&ds, "note", &crate::h5::attrs::AttrValue::Str("kept".into())).unwrap();
        let wider: Vec<String> = vec!["a much longer value".into(), "b".into()];
        recreate_fixed_strings(&ds, &wider, 32, true).unwrap();
        assert_eq!(members(&group), vec!["s".to_string()]);
        let now = file.dataset("s").unwrap();
        assert_eq!(read_strings(&now).unwrap(), wider);
        assert_eq!(crate::h5::attrs::get_str(&now, "note").unwrap().as_deref(), Some("kept"));
    }

    fn members(group: &hdf5::Group) -> Vec<String> {
        crate::h5::ops::members(group).unwrap()
    }

    /// A string read is held a slab at a time, each slab within its budget by
    /// what it reads to, and the slabs are the whole dataset in C order.  A
    /// dataset of one-byte strings read to a `String` each, all at once: a
    /// 17 KB file took 687 MB to digest.
    #[test]
    fn s13_1_strings_are_digested_a_bounded_slab_at_a_time() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("s.h5"))).unwrap();
        let n = 20_000usize;
        let fixed = file.new_dataset::<hdf5::types::FixedAscii<1>>().shape([n / 4, 4]).create("fixed").unwrap();
        let values: Vec<String> = (0..n).map(|i| ((b'a' + (i % 26) as u8) as char).to_string()).collect();
        write_fixed_strings(&fixed, &values, 1, true).unwrap();
        let long: Vec<String> = (0..64).map(|i| format!("{i:04}").repeat(1000)).collect();
        let vlen = create_strings(&file.as_group().unwrap(), "vlen", &long).unwrap();
        for (ds, budget, whole) in [(&fixed, 64 * 1024, &values), (&vlen, 20_000, &long)] {
            let mut slabs = Vec::new();
            for_each_string_slab(ds, budget, &mut |slab| {
                slabs.push(slab);
                Ok(())
            })
            .unwrap();
            assert!(slabs.len() > 1, "{}: one slab of {}", ds.name(), slabs[0].len());
            for slab in &slabs {
                let text: usize = slab.iter().map(Vec::len).sum();
                let width = if ds.name() == "/fixed" { Some(1) } else { None };
                let held = decoded_bytes(slab.len(), width, if width.is_some() { 0 } else { text }).unwrap();
                assert!(held <= budget || slab.len() == ds.shape()[1..].iter().product::<usize>(), "{held} > {budget}");
            }
            let stored: Vec<Vec<u8>> = whole.iter().map(|s| s.as_bytes().to_vec()).collect();
            assert_eq!(slabs.concat(), stored);
            assert_eq!(&read_strings(ds).unwrap(), whole);
            let streamed = crate::integrity::digest::dataset_digest_streamed(ds, "x", "sha256", budget).unwrap();
            let at_once = crate::integrity::digest::strings_digest("x", whole, &ds.shape(), "sha256").unwrap();
            assert_eq!(streamed, at_once);
        }
        // A scalar string reads whole, unmeasured: it holds one string.
        let scalar = create_scalar_string(&file.as_group().unwrap(), "scalar", "{\"a\": 1}").unwrap();
        assert_eq!(read_strings(&scalar).unwrap(), vec!["{\"a\": 1}".to_string()]);
        assert_eq!(vlen_text_bytes(&scalar, None).unwrap(), 0);
        assert!(vlen_text_bytes(&vlen, None).unwrap() >= 64 * 4000);
    }

    /// A whole read is budgeted by what it decodes to, not by its stored
    /// bytes alone: many elements may name one stored string.
    #[test]
    fn s14_5_a_whole_read_counts_what_it_decodes_to() {
        assert_eq!(decoded_bytes(10, Some(1), 0), Some(10 * (STRING_COST + 2)));
        assert_eq!(decoded_bytes(10, None, 100), Some(10 * (STRING_COST + 8) + 200));
        assert_eq!(decoded_bytes(usize::MAX, Some(1), 0), None);
        assert_eq!(decoded_bytes(1, None, usize::MAX), None);
    }

    /// A string's stored element states its length, which no string the
    /// file stores exceeds the file's size by: a longer one is refused before
    /// HDF5 reads, in a whole read and a slab at a time, from the file HDF5
    /// opened whatever its name now names.
    #[test]
    fn s14_5_a_string_stated_longer_than_its_file_is_refused_unread() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long.h5");
        let values: Vec<String> = ["a", "bc", "def"].iter().map(|s| s.to_string()).collect();
        let offsets = {
            let file = crate::h5::file::marked(|| hdf5::File::create(&path)).unwrap();
            let one = create_scalar_string(&file, "one", "{}").unwrap();
            let rows = create_strings(&file, "rows", &values).unwrap();
            create_strings(&file, "kept", &values).unwrap();
            let offset =
                |ds: &hdf5::Dataset| super::super::locked(|| unsafe { crate::h5sys::h5d::H5Dget_offset(ds.id()) });
            (offset(&one), offset(&rows))
        };
        // The length of the scalar's string, and of the vector's last.
        let stated = 0xFFFF_FF00u32.to_le_bytes();
        let written = std::fs::read(&path).unwrap();
        let mut bytes = written.clone();
        for at in [offsets.0 as usize, offsets.1 as usize + 2 * 16] {
            bytes[at..at + 4].copy_from_slice(&stated);
        }
        std::fs::write(&path, &bytes).unwrap();

        let file = crate::h5::file::open_read(&path).unwrap();
        // The name is given to the file as written while the crafted one is
        // open, as `amend` replaces a file: HDF5 reads on from the file it
        // opened, and so does the check.
        #[cfg(unix)]
        {
            let sound = dir.path().join("sound.h5");
            std::fs::write(&sound, &written).unwrap();
            std::fs::rename(&sound, &path).unwrap();
        }
        let refused = |err: Error| assert!(err.to_string().contains("states a string of 4294967040 bytes"), "{err}");
        refused(read_stored_strings(&file.dataset("one").unwrap()).unwrap_err());
        let rows = file.dataset("rows").unwrap();
        refused(read_stored_strings(&rows).unwrap_err());
        refused(for_each_string_slab(&rows, 1 << 20, &mut |_| Ok(())).unwrap_err());
        assert_eq!(read_strings(&file.dataset("kept").unwrap()).unwrap(), values);
    }
}
