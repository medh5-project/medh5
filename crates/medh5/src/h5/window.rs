//! Windowed reads that decompress only the blocks a window covers.
//!
//! The Blosc2 profiles store each chunk as a B2ND frame (spec §14.2): a grid
//! of independently compressed blocks.  HDF5's filter pipeline returns whole
//! chunks, so a 96³ patch over 1 MiB chunks decompresses up to eighteen of
//! them to keep 1.7 MiB.  [`read_window`] reads each stored chunk as it is and
//! has Blosc2 decompress only the window's part --- outside HDF5's global
//! lock, which the filter pipeline holds while it decompresses.
//!
//! The result is HDF5's bytes or nothing.  Another codec or a second filter,
//! a stored type HDF5 would convert on the way out, a chunk never written or
//! stored with its filter skipped, a frame that is not B2ND, a strided
//! selection: each makes [`read_window`] answer `None`, and the caller reads
//! through HDF5.

use ndarray::{ArrayD, IxDyn};

use crate::array::{DType, NdArray};
use crate::h5sys::{h5d, h5p};
use crate::{with_dtype, Result};

/// The chunk shape of a chunked dataset whose one filter is Blosc2, when the
/// filter's values describe that chunk, and the chunk's size in bytes.
///
/// HDF5's filter refuses a chunk its values do not describe --- a chunk size
/// in the filter's values (`cd_values[3]`) smaller than the chunk aborted the
/// process until the filter checked it --- so a window is read here only when
/// the filter would read the whole chunk, and HDF5 answers the rest with its
/// error.
fn blosc2_chunk(ds: &hdf5::Dataset, rank: usize, itemsize: usize) -> Option<(Vec<usize>, u64)> {
    super::locked(|| unsafe {
        // A contiguous dataset has an address; asking is cheaper than copying
        // the creation properties, and small datasets (index pools) are all
        // contiguous.
        if h5d::H5Dget_offset(ds.id()) != u64::MAX {
            return None;
        }
        let plist = h5d::H5Dget_create_plist(ds.id());
        if plist < 0 {
            return None;
        }
        let found = (|| {
            if h5p::H5Pget_layout(plist) != h5d::H5D_layout_t::H5D_CHUNKED || h5p::H5Pget_nfilters(plist) != 1 {
                return None;
            }
            let mut values = [0u32; 16];
            let (mut flags, mut n_values, mut config) = (0u32, values.len(), 0u32);
            let id = h5p::H5Pget_filter2(
                plist,
                0,
                &mut flags,
                &mut n_values,
                values.as_mut_ptr(),
                0,
                std::ptr::null_mut(),
                &mut config,
            );
            if id < 0 || id as u32 != medh5_sys::BLOSC2_FILTER_ID || !(4..=values.len()).contains(&n_values) {
                return None;
            }
            let mut dims = vec![0u64; rank];
            if h5p::H5Pget_chunk(plist, rank as i32, dims.as_mut_ptr()) != rank as i32 {
                return None;
            }
            // Type size, chunk bytes and, when present, the chunk's rank and
            // shape (the layout `blosc2_filter.c` documents).
            let bytes = dims.iter().try_fold(itemsize as u64, |n, d| n.checked_mul(*d))?;
            let shape_ok = n_values < 8
                || (values[7] as usize == rank
                    && n_values >= 8 + rank
                    && dims.iter().zip(&values[8..8 + rank]).all(|(d, v)| *d == u64::from(*v)));
            if values[2] as usize != itemsize || u64::from(values[3]) != bytes || !shape_ok {
                return None;
            }
            Some((dims.into_iter().map(|d| d as usize).collect(), bytes))
        })();
        h5p::H5Pclose(plist);
        found
    })
}

/// The largest stored chunk a window reads: twice the chunk's bytes and 64
/// KiB more --- far more than a Blosc2 frame adds to what it holds, its
/// header, trailer, B2ND metadata and block offsets, however badly the data
/// compresses --- and never more than the file holding it.
fn frame_ceiling(chunk_bytes: u64, file_bytes: u64) -> u64 {
    chunk_bytes.saturating_mul(2).saturating_add(64 * 1024).min(file_bytes)
}

/// Read the stored bytes of the chunk at `origin` into `frame`; their length,
/// or `None` for a chunk never written, stored with a filter skipped, or
/// stored at a size no frame of a chunk of `chunk_bytes` can have.
///
/// The size is the file's to state, up to 0xFFFFFFFF, so it is checked
/// before anything is reserved: a buffer sized by it alone took a crafted
/// 140 KB file to 4 GB of memory, or aborted the process.
///
/// `file_bytes` is the file's size as last measured, kept across the chunks
/// of one window: asking HDF5 for it opened a file identifier, under the
/// lock, for every chunk.
fn stored_chunk(
    ds: &hdf5::Dataset,
    origin: &[u64],
    chunk_bytes: u64,
    file_bytes: &mut Option<u64>,
    frame: &mut Vec<u8>,
) -> Option<usize> {
    phase!(Read);
    super::locked(|| unsafe {
        let (mut mask, mut addr, mut size) = (0u32, 0u64, 0u64);
        if h5d::H5Dget_chunk_info_by_coord(ds.id(), origin.as_ptr(), &mut mask, &mut addr, &mut size) < 0
            || addr == u64::MAX
            || size == 0
        {
            return None;
        }
        // The file's size once the chunk is in it: asking for the chunk
        // writes it out of HDF5's cache, in a file still being written.  So
        // a chunk stated past the size last measured has it measured again,
        // and is turned away only if it is past that too.
        if file_bytes.is_none_or(|bytes| size > frame_ceiling(chunk_bytes, bytes)) {
            *file_bytes = Some(ds.file().ok()?.size());
        }
        if size > frame_ceiling(chunk_bytes, (*file_bytes)?) {
            return None;
        }
        let size = usize::try_from(size).ok()?;
        if frame.len() < size {
            frame.try_reserve_exact(size - frame.len()).ok()?;
            frame.resize(size, 0);
        }
        let (mut filters, mut len) = (0u32, frame.len());
        if h5d::H5Dread_chunk(
            ds.id(),
            h5p::H5P_DEFAULT,
            origin.as_ptr(),
            &mut filters,
            frame.as_mut_ptr().cast(),
            &mut len,
        ) < 0
            || filters != 0
            || len > frame.len()
        {
            return None;
        }
        Some(len)
    })
}

/// Read the window `axes` --- [`super::data::read_region`]'s per-axis
/// `(start, count, step, kept)` --- by decompressing only the blocks it
/// covers; `None` when that cannot give exactly what HDF5 would.  One stored
/// chunk is held at a time, beside the window's values.
pub(crate) fn read_window(
    ds: &hdf5::Dataset,
    dtype: DType,
    axes: &[(usize, usize, usize, bool)],
    out_shape: &[usize],
) -> Result<Option<NdArray>> {
    let Some(window) = Window::new(ds, dtype, axes, out_shape) else {
        return Ok(None);
    };
    let mut file_bytes = None;
    let (mut frame, mut part) = (Vec::new(), Vec::new());
    window.values(|itemsize, out| {
        window.origins().all(|origin| {
            let offset: Vec<u64> = origin.iter().map(|o| *o as u64).collect();
            let Some(len) = stored_chunk(ds, &offset, window.chunk_bytes, &mut file_bytes, &mut frame) else {
                return false;
            };
            window.put(&origin, &frame[..len], itemsize, &mut part, out)
        })
    })
}

/// The stored chunks of the window `axes`, as [`read_window`] reads them,
/// held to be decompressed later: `None` when the window cannot be read so,
/// and HDF5 must read it, or when its frames would hold more than
/// `max_stored` bytes --- a thin window across many chunks holds far more
/// stored bytes than values.
pub(crate) fn read_frames(
    ds: &hdf5::Dataset,
    dtype: DType,
    axes: &[(usize, usize, usize, bool)],
    out_shape: &[usize],
    max_stored: usize,
) -> Option<Frames> {
    let window = Window::new(ds, dtype, axes, out_shape)?;
    let mut file_bytes = None;
    let (mut stored, mut held) = (Vec::new(), 0usize);
    for origin in window.origins() {
        let offset: Vec<u64> = origin.iter().map(|o| *o as u64).collect();
        let mut frame = Vec::new();
        let len = stored_chunk(ds, &offset, window.chunk_bytes, &mut file_bytes, &mut frame)?;
        held = held.saturating_add(holding(len, origin.len()));
        if held > max_stored {
            return None;
        }
        frame.truncate(len);
        stored.push((origin, frame));
    }
    Some(Frames { window, stored })
}

/// What [`read_frames`] counts against its cap for one chunk: its stored
/// bytes and their bookkeeping, so that many small chunks count too.
fn holding(frame_len: usize, rank: usize) -> usize {
    frame_len + std::mem::size_of::<(Vec<usize>, Vec<u8>)>() + rank * std::mem::size_of::<usize>()
}

/// The stored chunks a window covers, read under the lock: what
/// [`Frames::decode`] turns into the window's values with no HDF5 call, on
/// any thread.  A digest's workers decompress a slab so
/// (`integrity::workers`).
pub(crate) struct Frames {
    window: Window,
    /// Each touched chunk's origin and stored bytes, last axis fastest.
    stored: Vec<(Vec<usize>, Vec<u8>)>,
}

impl Frames {
    /// Whether every frame is one [`Frames::decode`] reads, but for blocks
    /// that do not decompress: B2ND, of the chunk's rank, shape and item
    /// size.  Parses each frame's metadata, decompressing nothing.
    pub(crate) fn fit(&self) -> bool {
        let chunk_shape: Vec<i64> = self.window.chunk.iter().map(|c| *c as i64).collect();
        self.stored.iter().all(|(_, frame)| {
            // SAFETY: the pointers span the lengths passed with them.
            unsafe {
                medh5_sys::medh5_b2nd_frame_fits(
                    frame.as_ptr(),
                    frame.len() as i64,
                    chunk_shape.len() as i8,
                    chunk_shape.as_ptr(),
                    self.window.dtype.itemsize() as i32,
                ) == 0
            }
        })
    }

    /// The window's values, decompressed from the frames; `None` when a
    /// frame cannot be read so (one that is not B2ND), and HDF5 must read it.
    /// Makes no HDF5 call.
    pub(crate) fn decode(&self) -> Result<Option<NdArray>> {
        let mut part = Vec::new();
        self.window.values(|itemsize, out| {
            self.stored.iter().all(|(origin, frame)| self.window.put(origin, frame, itemsize, &mut part, out))
        })
    }
}

/// A window of a dataset whose chunks are Blosc2 frames, and the chunks it
/// touches.
struct Window {
    dtype: DType,
    chunk: Vec<usize>,
    /// What each chunk holds decompressed ([`stored_chunk`]).
    chunk_bytes: u64,
    start: Vec<usize>,
    count: Vec<usize>,
    out_shape: Vec<usize>,
}

impl Window {
    /// The window `axes`, or `None` when its blocks cannot give exactly what
    /// HDF5 would.
    fn new(
        ds: &hdf5::Dataset,
        dtype: DType,
        axes: &[(usize, usize, usize, bool)],
        out_shape: &[usize],
    ) -> Option<Window> {
        // B2ND frames are rank 2 and up; a `bool` is an enumeration HDF5 maps.
        if dtype == DType::Bool || axes.len() < 2 || axes.iter().any(|(_, count, step, _)| *step != 1 || *count == 0) {
            return None;
        }
        let (chunk, chunk_bytes) = blosc2_chunk(ds, axes.len(), dtype.itemsize())?;
        // Another byte order, say: HDF5 would convert, and the stored bytes
        // are not the type's.
        let native = with_dtype!(dtype, T => hdf5::Datatype::from_type::<T>());
        match (ds.dtype(), native) {
            (Ok(stored), Ok(native)) if stored == native => {}
            _ => return None,
        }
        Some(Window {
            dtype,
            chunk,
            chunk_bytes,
            start: axes.iter().map(|a| a.0).collect(),
            count: axes.iter().map(|a| a.1).collect(),
            out_shape: out_shape.to_vec(),
        })
    }

    /// The origin of each chunk the window touches, last axis fastest, one
    /// at a time: a window over many small chunks has many.
    fn origins(&self) -> Origins<'_> {
        let rank = self.chunk.len();
        let first: Vec<usize> = (0..rank).map(|a| self.start[a] / self.chunk[a]).collect();
        let last: Vec<usize> = (0..rank).map(|a| (self.start[a] + self.count[a] - 1) / self.chunk[a]).collect();
        Origins { chunk: &self.chunk, at: Some(first.clone()), first, last }
    }

    /// The window's values, `fill` writing their bytes, C-ordered, given the
    /// item size; `None` when it cannot.
    fn values(&self, fill: impl FnOnce(usize, &mut [u8]) -> bool) -> Result<Option<NdArray>> {
        Ok(with_dtype!(self.dtype, T => {
            let mut values = vec![T::default(); self.count.iter().product()];
            // SAFETY: `T` is a number (`bool` was turned away), for which
            // every bit pattern is a value, and the slice spans `values`
            // exactly.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(values.as_slice()))
            };
            if !fill(std::mem::size_of::<T>(), bytes) {
                return Ok(None);
            }
            Some(NdArray::from(ArrayD::from_shape_vec(IxDyn(&self.out_shape), values)?))
        }))
    }

    /// Decompress the window's part of the chunk at `origin`, stored as
    /// `frame`, into `out` (through `part`); `false` when the frame cannot be
    /// read so.
    fn put(&self, origin: &[usize], frame: &[u8], itemsize: usize, part: &mut Vec<u8>, out: &mut [u8]) -> bool {
        let (chunk, start, count) = (&self.chunk, &self.start, &self.count);
        let rank = chunk.len();
        let chunk_shape: Vec<i64> = chunk.iter().map(|c| *c as i64).collect();
        let lo: Vec<i64> = (0..rank).map(|a| (start[a].max(origin[a]) - origin[a]) as i64).collect();
        let hi: Vec<i64> =
            (0..rank).map(|a| ((start[a] + count[a]).min(origin[a] + chunk[a]) - origin[a]) as i64).collect();
        let shape: Vec<usize> = lo.iter().zip(&hi).map(|(l, h)| (h - l) as usize).collect();
        let n = shape.iter().product::<usize>() * itemsize;
        if part.len() < n {
            part.resize(n, 0u8);
        }
        phase!(Window, n);
        // SAFETY: every pointer spans the length passed with it; the helper
        // reads `frame` and writes at most `n` bytes of `part`.
        let rc = unsafe {
            medh5_sys::medh5_b2nd_read_slice(
                frame.as_ptr(),
                frame.len() as i64,
                rank as i8,
                chunk_shape.as_ptr(),
                itemsize as i32,
                lo.as_ptr(),
                hi.as_ptr(),
                part.as_mut_ptr().cast(),
                n as i64,
            )
        };
        if rc != 0 {
            return false;
        }
        let corner: Vec<usize> = (0..rank).map(|a| origin[a] + lo[a] as usize - start[a]).collect();
        place(&part[..n], &shape, out, count, &corner, itemsize);
        true
    }
}

/// [`Window::origins`].
struct Origins<'w> {
    chunk: &'w [usize],
    first: Vec<usize>,
    last: Vec<usize>,
    /// The next chunk's index along each axis; `None` past the last.
    at: Option<Vec<usize>>,
}

impl Iterator for Origins<'_> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        let at = self.at.as_mut()?;
        let origin = at.iter().zip(self.chunk).map(|(i, c)| i * c).collect();
        // The next chunk, last axis fastest.
        let mut axis = at.len();
        loop {
            if axis == 0 {
                self.at = None;
                break;
            }
            axis -= 1;
            if at[axis] < self.last[axis] {
                at[axis] += 1;
                at[axis + 1..].copy_from_slice(&self.first[axis + 1..]);
                break;
            }
        }
        Some(origin)
    }
}

/// Copy the C-ordered `block` of `shape` into `out` --- C-ordered, of shape
/// `count` --- with its first element at `corner`.
fn place(block: &[u8], shape: &[usize], out: &mut [u8], count: &[usize], corner: &[usize], itemsize: usize) {
    let rank = shape.len();
    let row = shape[rank - 1] * itemsize;
    let mut index = vec![0usize; rank - 1];
    for source in block.chunks_exact(row) {
        let mut flat = 0usize;
        for a in 0..rank - 1 {
            flat = flat * count[a] + corner[a] + index[a];
        }
        let at = (flat * count[rank - 1] + corner[rank - 1]) * itemsize;
        out[at..at + row].copy_from_slice(source);
        for a in (0..rank - 1).rev() {
            index[a] += 1;
            if index[a] < shape[a] {
                break;
            }
            index[a] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{Index, Slice};
    use crate::h5::data::{self, Layout};
    use crate::rng::Rng;
    use crate::storage::codecs::{resolve_profile, Role};

    /// Runs and noise, so blocks compress unevenly.
    fn volume(shape: &[usize], dtype: DType, rng: &mut Rng) -> NdArray {
        let n: usize = shape.iter().product();
        let values: Vec<f64> = (0..n)
            .map(|i| if (i / 97) % 3 == 0 { ((i / 97) % 50) as f64 } else { rng.integer(0, 120).unwrap() as f64 })
            .collect();
        NdArray::from_vec(shape, values).unwrap().astype(dtype)
    }

    /// A random window: some axes sliced anywhere (edge chunks included),
    /// some indexed, as an annotation layer or a timepoint is.
    fn window(shape: &[usize], rng: &mut Rng) -> Vec<Index> {
        shape
            .iter()
            .map(|n| {
                let n = *n as i64;
                if rng.integer(0, 5).unwrap() == 0 {
                    Index::At(rng.integer(-n, n).unwrap())
                } else {
                    let start = rng.integer(0, n).unwrap();
                    Index::Slice(Slice::new(start, rng.integer(start + 1, n + 1).unwrap()))
                }
            })
            .collect()
    }

    fn through_hdf5(ds: &hdf5::Dataset, index: &[Index]) -> NdArray {
        let axes = data::resolve(&ds.shape(), index).unwrap();
        let out: Vec<usize> = axes.iter().filter(|a| a.3).map(|a| a.1).collect();
        data::read_hyperslab(ds, data::dtype(ds).unwrap(), &axes, &out).unwrap()
    }

    fn window_only(ds: &hdf5::Dataset, index: &[Index]) -> Option<NdArray> {
        let axes = data::resolve(&ds.shape(), index).unwrap();
        let out: Vec<usize> = axes.iter().filter(|a| a.3).map(|a| a.1).collect();
        read_window(ds, data::dtype(ds).unwrap(), &axes, &out).unwrap()
    }

    #[test]
    fn s14_2_a_window_reads_hdf5s_bytes_from_every_blosc2_profile() {
        crate::h5::init();
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("w.h5")).unwrap();
        let mut rng = Rng::new(14);
        // (profile, role, dtype, shape, chunks)
        type Case = (&'static str, Role, DType, &'static [usize], &'static [usize]);
        let cases: [Case; 6] = [
            ("training", Role::Image, DType::I16, &[37, 50, 61], &[16, 32, 24]),
            ("balanced", Role::Image, DType::F32, &[20, 33, 47], &[8, 16, 16]),
            ("balanced", Role::Label, DType::U8, &[3, 30, 40, 50], &[1, 16, 16, 16]),
            ("archive", Role::Label, DType::U16, &[25, 31, 29], &[25, 8, 29]),
            ("archive", Role::Image, DType::F64, &[2, 17, 19, 23], &[1, 17, 8, 8]),
            ("training", Role::Label, DType::I32, &[64, 70], &[32, 32]),
        ];
        for (k, (profile, role, dtype, shape, chunks)) in cases.into_iter().enumerate() {
            let layout = Layout {
                chunks: Some(chunks.to_vec()),
                filters: resolve_profile(Some(profile)).unwrap().codec(role).filters(),
            };
            let ds = data::create(&file, &format!("d{k}"), &volume(shape, dtype, &mut rng), &layout).unwrap();
            for _ in 0..60 {
                let index = window(shape, &mut rng);
                let fast = window_only(&ds, &index);
                assert!(fast.is_some(), "{profile} {dtype:?} {index:?}: the window path declined");
                assert_eq!(fast.unwrap(), through_hdf5(&ds, &index), "{profile} {dtype:?} {index:?}");
                assert_eq!(data::read_region(&ds, &index).unwrap(), through_hdf5(&ds, &index));
            }
        }
    }

    /// The chunks a window touches are walked one at a time, last axis
    /// fastest: 2**26 of them cost nothing until each is asked for.
    #[test]
    fn s14_2_a_window_walks_its_chunks_one_at_a_time() {
        let window = |start: Vec<usize>, count: Vec<usize>, chunk: Vec<usize>| Window {
            dtype: DType::U8,
            chunk,
            chunk_bytes: 1,
            out_shape: count.clone(),
            start,
            count,
        };
        let many = window(vec![0, 0], vec![8192, 8192], vec![1, 1]);
        let first: Vec<Vec<usize>> = many.origins().take(3).collect();
        assert_eq!(first, [vec![0, 0], vec![0, 1], vec![0, 2]]);
        let few = window(vec![3, 9], vec![10, 8], vec![4, 8]);
        let all: Vec<Vec<usize>> = few.origins().collect();
        assert_eq!(
            all,
            [vec![0, 8], vec![0, 16], vec![4, 8], vec![4, 16], vec![8, 8], vec![8, 16], vec![12, 8], vec![12, 16]]
        );
    }

    /// Frames held for a worker decode to the window's bytes, and are not
    /// held past their cap: a thin window across many chunks holds the
    /// chunks' stored bytes, many times its values, and HDF5 reads it.
    #[test]
    fn s14_2_held_frames_decode_to_the_window_and_stop_at_their_cap() {
        crate::h5::init();
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("w.h5")).unwrap();
        let mut rng = Rng::new(15);
        let layout = Layout {
            chunks: Some(vec![8, 16, 16]),
            filters: resolve_profile(Some("training")).unwrap().codec(Role::Image).filters(),
        };
        let ds = data::create(&file, "d", &volume(&[24, 64, 64], DType::I16, &mut rng), &layout).unwrap();
        let thin = [Index::Slice(Slice::new(0, 24)), Index::Slice(Slice::new(0, 64)), Index::At(5)];
        let axes = data::resolve(&ds.shape(), &thin).unwrap();
        let out: Vec<usize> = axes.iter().filter(|a| a.3).map(|a| a.1).collect();
        let frames = read_frames(&ds, DType::I16, &axes, &out, usize::MAX).expect("Blosc2 frames");
        let held: usize = frames.stored.iter().map(|(origin, f)| holding(f.len(), origin.len())).sum();
        assert_eq!(frames.stored.len(), 3 * 4, "every chunk of the plane x = 5");
        assert!(frames.fit());
        assert_eq!(frames.decode().unwrap().unwrap(), through_hdf5(&ds, &thin));
        let values = 24 * 64 * 2;
        assert!(held > values, "the frames ({held} B) hold more than the window ({values} B)");
        assert!(read_frames(&ds, DType::I16, &axes, &out, held - 1).is_none());
        assert!(read_frames(&ds, DType::I16, &axes, &out, held).is_some());
        match data::read_region_stored(&ds, &thin, values).unwrap() {
            data::Region::Values(read) => assert_eq!(read, through_hdf5(&ds, &thin)),
            data::Region::Stored(..) => panic!("held past the cap"),
        }
    }

    /// The file is the one to state a stored chunk's size, up to 0xFFFFFFFF:
    /// a size no frame of the chunk can have, and the file cannot hold, is
    /// refused before a byte is reserved --- by the window and by a raw read.
    ///
    /// The file is written in HDF5's earliest format, whose chunk index is a
    /// version-1 B-tree: each key holds a chunk's stored size, and the node
    /// carries no checksum, so the size can be rewritten in place.
    #[test]
    fn s14_2_a_stored_chunk_size_beyond_the_file_is_refused_before_it_is_reserved() {
        crate::h5::init();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crafted.h5");
        let shape = [16usize, 16, 16];
        {
            let file = crate::h5::file::marked(|| hdf5::File::create(&path)).unwrap();
            let layout = Layout {
                chunks: Some(shape.to_vec()),
                filters: resolve_profile(Some("training")).unwrap().codec(Role::Image).filters(),
            };
            data::create(&file, "d", &volume(&shape, DType::I16, &mut Rng::new(3)), &layout).unwrap();
        }
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes.windows(4).position(|w| w == b"TREE").expect("a version-1 B-tree");
        assert_eq!(bytes[at + 4], 1, "a node of raw-data chunks");
        // Signature, type, level, entries used, two sibling addresses: then
        // the first key, whose first field is the chunk's stored size.
        bytes[at + 24..at + 28].copy_from_slice(&0xF000_0000u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let file = crate::h5::file::marked(|| hdf5::File::open(&path)).unwrap();
        let ds = file.dataset("d").unwrap();
        let window =
            [Index::Slice(Slice::new(2, 10)), Index::Slice(Slice::new(2, 10)), Index::Slice(Slice::new(2, 10))];
        assert!(window_only(&ds, &window).is_none(), "the window declines the chunk");
        assert!(data::read_region(&ds, &window).is_err(), "and HDF5 refuses it");
        let err = crate::h5::ops::read_raw_chunk(&ds, &[0, 0, 0]).unwrap_err();
        assert!(err.to_string().contains("more than the file's"), "{err}");
        assert_eq!(frame_ceiling(8192, 1 << 40), 2 * 8192 + 64 * 1024);
        assert_eq!(frame_ceiling(8192, 4000), 4000, "never past the file");
    }

    #[test]
    fn s14_2_what_the_window_cannot_read_exactly_goes_through_hdf5() {
        crate::h5::init();
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("f.h5")).unwrap();
        let mut rng = Rng::new(15);
        let shape = [24usize, 40, 40];
        let data_ = volume(&shape, DType::I16, &mut rng);
        let blosc2 = resolve_profile(Some("balanced")).unwrap().codec(Role::Image).filters();
        let gzip = resolve_profile(Some("portable")).unwrap().codec(Role::Image).filters();
        let inner = [Index::Slice(Slice::new(3, 21)), Index::Slice(Slice::new(5, 33)), Index::Slice(Slice::new(0, 17))];

        // Another codec.
        let portable =
            data::create(&file, "portable", &data_, &Layout { chunks: Some(vec![8, 16, 16]), filters: gzip }).unwrap();
        assert!(window_only(&portable, &inner).is_none());
        assert_eq!(data::read_region(&portable, &inner).unwrap(), through_hdf5(&portable, &inner));

        // A strided selection.
        let ds =
            data::create(&file, "blosc2", &data_, &Layout { chunks: Some(vec![8, 16, 16]), filters: blosc2.clone() })
                .unwrap();
        let strided = [Index::Slice(Slice { start: Some(1), stop: Some(20), step: Some(3) })];
        assert!(window_only(&ds, &strided).is_none());
        assert_eq!(data::read_region(&ds, &strided).unwrap(), through_hdf5(&ds, &strided));

        // Chunks never written read as the fill value, through HDF5.
        let sparse = data::create_empty(
            &file,
            "sparse",
            DType::I16,
            &shape,
            &Layout { chunks: Some(vec![8, 16, 16]), filters: blosc2 },
        )
        .unwrap();
        let block = volume(&[8, 16, 16], DType::I16, &mut rng);
        data::write_region(&sparse, &block, &[8, 16, 0]).unwrap();
        assert!(window_only(&sparse, &inner).is_none());
        let read = data::read_region(&sparse, &inner).unwrap();
        assert_eq!(read, through_hdf5(&sparse, &inner));
        assert!(read.cast::<i16>().iter().any(|v| *v != 0) && read.cast::<i16>().iter().any(|v| *v == 0));
        let written =
            [Index::Slice(Slice::new(9, 15)), Index::Slice(Slice::new(17, 30)), Index::Slice(Slice::new(2, 9))];
        assert_eq!(window_only(&sparse, &written).unwrap(), through_hdf5(&sparse, &written));
    }
}
