//! Chunk writes that compress outside HDF5's lock: the mirror of the window
//! reader ([`super::window`]).
//!
//! HDF5 runs a dataset's filters inside `H5Dwrite`, under its global lock, so
//! a write compresses one chunk at a time whatever else is running.  For a
//! dataset whose one filter is Blosc2, [`Chunks`] compresses each whole chunk
//! with the filter's own compression (`medh5_blosc2_compress`, the vendored
//! filter's patch 7) and the dataset's filter values, which needs no lock and
//! runs on workers; the caller stores each frame with `H5Dwrite_chunk`, in
//! chunk order.  The same code compresses as when HDF5 runs the filter, so
//! the stored bytes are the filter's.
//!
//! The result is HDF5's bytes or HDF5's write.  Another codec or a second
//! filter, a stored type HDF5 would convert on the way in, a `bool` (an
//! enumeration HDF5 maps): [`Chunks::of`] answers `None`, and the caller
//! writes through HDF5.  A chunk the filter's compression refuses is written
//! through HDF5 too, which fails it, or stores it unfiltered, as the filter's
//! flags say.

use std::os::raw::{c_int, c_void};

use crate::array::{DType, NdArray};
use crate::h5sys::{h5d, h5p};
use crate::{with_array, with_dtype, Error, Result};

unsafe extern "C" {
    /// The HDF5-Blosc2 filter's compression of one chunk (`blosc2_filter.c`,
    /// patch 7): its frame's size, the frame in `*out` for the caller to
    /// `free`, or a negative code.  It calls no HDF5 function and starts no
    /// Blosc2 thread when `one_thread` is set; C-Blosc2 must have started
    /// (`medh5_sys::register_blosc2_filter`).
    fn medh5_blosc2_compress(
        cd_nelmts: usize,
        cd_values: *const u32,
        input: *const c_void,
        nbytes: usize,
        one_thread: c_int,
        out: *mut *mut c_void,
    ) -> i64;
}

/// One chunk's frame, as the filter's compression allocated it.
pub(crate) struct Frame {
    data: *mut c_void,
    len: usize,
}

// SAFETY: the frame is a heap buffer nothing else points into.
unsafe impl Send for Frame {}

impl Frame {
    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: `data` holds `len` bytes, written by the compression.
        unsafe { std::slice::from_raw_parts(self.data.cast::<u8>(), self.len) }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        // SAFETY: allocated by the C library's `malloc`, freed once.
        unsafe { libc::free(self.data) }
    }
}

/// What a worker hands back for one chunk: its frame, or its values when the
/// compression refused them, for HDF5 to write.
pub(crate) enum Compressed {
    Frame(Frame),
    Refused(Vec<u8>),
}

/// How one chunked dataset's chunks are compressed: plain values, so workers
/// share it.
pub(crate) struct Codec {
    values: Vec<u32>,
    chunk: Vec<usize>,
    shape: Vec<usize>,
    itemsize: usize,
}

impl Codec {
    /// The bytes of one whole chunk.
    fn chunk_bytes(&self) -> usize {
        self.chunk.iter().product::<usize>() * self.itemsize
    }

    /// Compress one whole chunk's values, C order, as the filter would.
    pub(crate) fn compress(&self, chunk: Vec<u8>) -> Compressed {
        debug_assert_eq!(chunk.len(), self.chunk_bytes());
        let mut out: *mut c_void = std::ptr::null_mut();
        // SAFETY: the values and the chunk span the lengths passed with
        // them; the frame is the caller's to free, which `Frame` does.
        let len = unsafe {
            medh5_blosc2_compress(
                self.values.len(),
                self.values.as_ptr(),
                chunk.as_ptr().cast(),
                chunk.len(),
                1,
                &mut out,
            )
        };
        if len <= 0 || out.is_null() {
            return Compressed::Refused(chunk);
        }
        Compressed::Frame(Frame { data: out, len: len as usize })
    }

    /// The chunk at `origin` cut from `slab` --- the C-ordered bytes of the
    /// dataset's values from `slab_start` over `slab_shape` --- and padded
    /// with zeros past the dataset's extent, as HDF5 hands an edge chunk to
    /// the filter.  The slab must cover the chunk's part inside the extent.
    pub(crate) fn cut(&self, slab: &[u8], slab_shape: &[usize], slab_start: &[usize], origin: &[usize]) -> Vec<u8> {
        let rank = self.chunk.len();
        let mut out = vec![0u8; self.chunk_bytes()];
        let extent: Vec<usize> = (0..rank).map(|a| self.chunk[a].min(self.shape[a] - origin[a])).collect();
        if extent.contains(&0) {
            return out;
        }
        let row = extent[rank - 1] * self.itemsize;
        let mut index = vec![0usize; rank];
        loop {
            // The row at `index` (last axis 0) of the chunk, in the chunk and
            // in the slab.
            let (mut at_chunk, mut at_slab) = (0usize, 0usize);
            for a in 0..rank {
                at_chunk = at_chunk * self.chunk[a] + index[a];
                at_slab = at_slab * slab_shape[a] + (origin[a] + index[a] - slab_start[a]);
            }
            let (at_chunk, at_slab) = (at_chunk * self.itemsize, at_slab * self.itemsize);
            out[at_chunk..at_chunk + row].copy_from_slice(&slab[at_slab..at_slab + row]);
            let mut axis = rank - 1;
            loop {
                if axis == 0 {
                    return out;
                }
                axis -= 1;
                index[axis] += 1;
                if index[axis] < extent[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
    }
}

/// A dataset written a whole chunk at a time ([module](self)).
pub(crate) struct Chunks {
    ds: hdf5::Dataset,
    dtype: DType,
    codec: Codec,
}

impl Chunks {
    /// The writer of `ds`, holding `dtype`'s values, when its one filter is
    /// Blosc2 and its stored type is `dtype`'s own; `None` otherwise.
    pub(crate) fn of(ds: &hdf5::Dataset, dtype: DType) -> Option<Chunks> {
        if dtype == DType::Bool || ds.ndim() == 0 {
            return None;
        }
        let same_type = with_dtype!(dtype, T => matches!(
            (ds.dtype(), hdf5::Datatype::from_type::<T>()),
            (Ok(stored), Ok(native)) if stored == native
        ));
        if !same_type {
            return None;
        }
        let rank = ds.ndim();
        let (values, chunk) = super::locked(|| unsafe {
            let plist = h5d::H5Dget_create_plist(ds.id());
            if plist < 0 {
                return None;
            }
            let found = (|| {
                if h5p::H5Pget_layout(plist) != h5d::H5D_layout_t::H5D_CHUNKED || h5p::H5Pget_nfilters(plist) != 1 {
                    return None;
                }
                let mut values = [0u32; 32];
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
                if id < 0 || id as u32 != medh5_sys::BLOSC2_FILTER_ID || n_values > values.len() {
                    return None;
                }
                let mut dims = vec![0u64; rank];
                if h5p::H5Pget_chunk(plist, rank as i32, dims.as_mut_ptr()) != rank as i32 {
                    return None;
                }
                Some((values[..n_values].to_vec(), dims.into_iter().map(|d| d as usize).collect::<Vec<_>>()))
            })();
            h5p::H5Pclose(plist);
            found
        })?;
        let codec = Codec { values, chunk, shape: ds.shape(), itemsize: dtype.itemsize() };
        // The filter's values describe a chunk of this type, as `set_local`
        // derived them: anything else is HDF5's to refuse.
        if codec.values.len() < 4
            || codec.values[2] as usize != codec.itemsize
            || codec.values[3] as usize != codec.chunk_bytes()
        {
            return None;
        }
        Some(Chunks { ds: ds.clone(), dtype, codec })
    }

    /// The dataset's chunk shape.
    pub(crate) fn chunk(&self) -> &[usize] {
        &self.codec.chunk
    }

    /// How its chunks are compressed, for workers to share.
    pub(crate) fn codec(&self) -> &Codec {
        &self.codec
    }

    /// The origins of the chunks whose first coordinate lies in `rows`, in C
    /// order: the chunks a slab of those rows covers.
    pub(crate) fn origins(&self, rows: std::ops::Range<usize>) -> Vec<Vec<usize>> {
        let (chunk, shape) = (&self.codec.chunk, &self.codec.shape);
        let first = rows.start.div_ceil(chunk[0]) * chunk[0];
        let mut lead: Vec<usize> = (first..rows.end.min(shape[0])).step_by(chunk[0]).collect();
        let mut out: Vec<Vec<usize>> = lead.drain(..).map(|r| vec![r]).collect();
        for a in 1..shape.len() {
            out = out
                .into_iter()
                .flat_map(|o| {
                    (0..shape[a]).step_by(chunk[a]).map(move |c| {
                        let mut o = o.clone();
                        o.push(c);
                        o
                    })
                })
                .collect();
        }
        out
    }

    /// Store one chunk: its frame with `H5Dwrite_chunk`, or the values the
    /// compression refused through HDF5's own write.
    pub(crate) fn store(&self, origin: &[usize], chunk: Compressed) -> Result<()> {
        match chunk {
            Compressed::Frame(frame) => {
                let offset: Vec<u64> = origin.iter().map(|o| *o as u64).collect();
                let bytes = frame.bytes();
                let status = super::locked(|| unsafe {
                    h5d::H5Dwrite_chunk(
                        self.ds.id(),
                        h5p::H5P_DEFAULT,
                        0,
                        offset.as_ptr(),
                        bytes.len(),
                        bytes.as_ptr().cast(),
                    )
                });
                if status < 0 {
                    return Err(Error::Io(format!("could not write the chunk at {origin:?} of {}", self.ds.name())));
                }
                Ok(())
            }
            Compressed::Refused(values) => self.write_through_hdf5(origin, &values),
        }
    }

    /// The chunk at `origin`, from its whole chunk's values, written as
    /// HDF5 writes it.
    fn write_through_hdf5(&self, origin: &[usize], values: &[u8]) -> Result<()> {
        let (chunk, shape) = (&self.codec.chunk, &self.codec.shape);
        let extent: Vec<usize> = (0..chunk.len()).map(|a| chunk[a].min(shape[a] - origin[a])).collect();
        let inside =
            Codec { values: Vec::new(), chunk: extent.clone(), shape: chunk.clone(), itemsize: self.codec.itemsize };
        // The chunk's part inside the extent, as a slab of its own.
        let part = inside.cut(values, chunk, &vec![0; chunk.len()], &vec![0; chunk.len()]);
        let array = with_dtype!(self.dtype, T => {
            let n = part.len() / std::mem::size_of::<T>();
            let mut typed = vec![T::default(); n];
            // SAFETY: `T` is a number (`bool` is turned away in `of`), for
            // which every bit pattern is a value; the lengths agree.
            unsafe { std::ptr::copy_nonoverlapping(part.as_ptr(), typed.as_mut_ptr().cast::<u8>(), part.len()) };
            NdArray::from(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&extent), typed)?)
        });
        super::data::write_region(&self.ds, &array, origin)
    }

    /// Write `slab` --- the dataset's rows from `start`, whole along every
    /// other axis --- chunk by chunk, compressed on up to `threads` workers.
    /// The slab starts on a chunk boundary and ends on one or at the extent.
    pub(crate) fn write_rows(&self, slab: &NdArray, start: usize, threads: usize) -> Result<()> {
        let slab_shape = slab.shape();
        let rows = start..start + slab_shape[0];
        let mut slab_start = vec![0usize; slab_shape.len()];
        slab_start[0] = start;
        let bytes = native_bytes(slab);
        let origins = self.origins(rows);
        let codec = &self.codec;
        crate::parallel::ordered(
            threads,
            origins.iter(),
            |origin| codec.compress(codec.cut(&bytes, &slab_shape, &slab_start, origin)),
            {
                let mut each = origins.iter();
                move |chunk| self.store(each.next().expect("one origin per chunk"), chunk)
            },
        )
    }
}

/// The C-ordered bytes of `array`'s values, as they lie in memory.
pub(crate) fn native_bytes(array: &NdArray) -> std::borrow::Cow<'_, [u8]> {
    with_array!(array, a => {
        match a.as_slice() {
            // SAFETY: a number's bytes, every one of them initialised.
            Some(values) => std::borrow::Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
            }),
            None => {
                let standard = a.as_standard_layout();
                let values = standard.as_slice().expect("standard layout");
                // SAFETY: as above.
                std::borrow::Cow::Owned(unsafe {
                    std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
                }
                .to_vec())
            }
        }
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::array::{Index, Slice};
    use crate::h5::data::{self, Layout};
    use crate::rng::Rng;
    use crate::storage::codecs::{resolve_profile, Role};

    /// Every chunked dataset of a file, by path: each stored chunk's offset,
    /// filter mask and bytes, in the order HDF5 indexes them.
    pub(crate) fn stored_chunks(
        path: &std::path::Path,
    ) -> std::collections::BTreeMap<String, Vec<(Vec<u64>, u32, Vec<u8>)>> {
        use crate::h5sys::{h5d, h5s};
        fn walk(group: &hdf5::Group, out: &mut std::collections::BTreeMap<String, Vec<(Vec<u64>, u32, Vec<u8>)>>) {
            for name in group.member_names().unwrap() {
                if let Ok(ds) = group.dataset(&name) {
                    if ds.chunk().is_none() {
                        continue;
                    }
                    let rank = ds.ndim();
                    let chunks = crate::h5::locked(|| unsafe {
                        let space = h5d::H5Dget_space(ds.id());
                        let mut n = 0u64;
                        h5d::H5Dget_num_chunks(ds.id(), space, &mut n);
                        let mut found = Vec::new();
                        for k in 0..n {
                            let mut offset = vec![0u64; rank];
                            let (mut mask, mut addr, mut size) = (0u32, 0u64, 0u64);
                            h5d::H5Dget_chunk_info(
                                ds.id(),
                                space,
                                k,
                                offset.as_mut_ptr(),
                                &mut mask,
                                &mut addr,
                                &mut size,
                            );
                            found.push(offset);
                        }
                        h5s::H5Sclose(space);
                        found
                    });
                    let read = chunks
                        .into_iter()
                        .map(|offset| {
                            let (mask, bytes) = crate::h5::ops::read_raw_chunk(&ds, &offset).unwrap();
                            (offset, mask, bytes)
                        })
                        .collect();
                    out.insert(ds.name(), read);
                } else if let Ok(child) = group.group(&name) {
                    walk(&child, out);
                }
            }
        }
        let file = crate::h5::file::open_read(path).unwrap();
        let mut out = std::collections::BTreeMap::new();
        walk(&file.as_group().unwrap(), &mut out);
        out
    }

    /// Runs and noise, so blocks compress unevenly.
    fn volume(shape: &[usize], dtype: DType, rng: &mut Rng) -> NdArray {
        let n: usize = shape.iter().product();
        let values: Vec<f64> = (0..n)
            .map(|i| if (i / 97) % 3 == 0 { ((i / 97) % 50) as f64 } else { rng.integer(0, 120).unwrap() as f64 })
            .collect();
        NdArray::from_vec(shape, values).unwrap().astype(dtype)
    }

    /// Every chunk's stored bytes and filter mask, in C order of origins.
    fn stored(ds: &hdf5::Dataset, writer: &Chunks) -> Vec<(Vec<usize>, u32, Vec<u8>)> {
        writer
            .origins(0..ds.shape()[0])
            .into_iter()
            .map(|origin| {
                let offset: Vec<u64> = origin.iter().map(|o| *o as u64).collect();
                let (mask, bytes) = crate::h5::ops::read_raw_chunk(ds, &offset).unwrap();
                (origin, mask, bytes)
            })
            .collect()
    }

    type Case = (&'static str, Role, DType, &'static [usize], &'static [usize]);

    const CASES: [Case; 8] = [
        ("training", Role::Image, DType::I16, &[37, 50, 61], &[16, 32, 24]),
        ("balanced", Role::Image, DType::F32, &[20, 33, 47], &[8, 16, 16]),
        ("balanced", Role::Label, DType::U8, &[3, 30, 40, 50], &[1, 16, 16, 16]),
        ("archive", Role::Label, DType::U16, &[25, 31, 29], &[25, 8, 29]),
        ("archive", Role::Image, DType::F64, &[2, 17, 19, 23], &[1, 17, 8, 8]),
        ("training", Role::Label, DType::I32, &[64, 70], &[32, 32]),
        // Rank 1 is plain Blosc2, not B2ND.
        ("training", Role::Image, DType::U32, &[1000], &[384]),
        ("archive", Role::Image, DType::I64, &[300], &[300]),
    ];

    /// Each chunk stored directly is the chunk HDF5's write stores --- its
    /// bytes and its filter mask, edge chunks included --- at one thread and
    /// at four, and reads back through HDF5 and through the window alike.
    #[test]
    fn s14_2_a_chunk_written_directly_is_the_chunk_hdf5_writes() {
        crate::h5::init();
        let _serial = crate::parallel::tests::serial();
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("c.h5")).unwrap();
        let mut rng = Rng::new(12);
        for (k, (profile, role, dtype, shape, chunks)) in CASES.into_iter().enumerate() {
            let values = volume(shape, dtype, &mut rng);
            let layout = Layout {
                chunks: Some(chunks.to_vec()),
                filters: resolve_profile(Some(profile)).unwrap().codec(role).filters(),
            };
            let through = data::create(&file, &format!("hdf5-{k}"), &values, &layout).unwrap();
            let expected = stored(&through, &Chunks::of(&through, dtype).expect("a Blosc2 dataset"));
            for threads in [1, 4] {
                let ds = data::create_empty(&file, &format!("direct-{k}-{threads}"), dtype, shape, &layout).unwrap();
                let writer = Chunks::of(&ds, dtype).expect("a Blosc2 dataset");
                // Slabs of two chunk rows, and the rest.
                let step = 2 * chunks[0];
                for start in (0..shape[0]).step_by(step) {
                    let stop = (start + step).min(shape[0]);
                    let slab =
                        data::read_region(&through, &[Index::Slice(Slice::new(start as i64, stop as i64))]).unwrap();
                    writer.write_rows(&slab, start, threads).unwrap();
                }
                assert_eq!(stored(&ds, &writer), expected, "{profile} {dtype:?} {shape:?} at {threads} threads");
                assert_eq!(data::read(&ds).unwrap(), values);
                if shape.len() > 1 {
                    let inner: Vec<Index> =
                        shape.iter().map(|n| Index::Slice(Slice::new((*n > 2) as i64, *n as i64))).collect();
                    let axes = data::resolve(shape, &inner).unwrap();
                    let out: Vec<usize> = axes.iter().map(|a| a.1).collect();
                    let window = crate::h5::window::read_window(&ds, dtype, &axes, &out).unwrap();
                    assert_eq!(window.expect("the window reads it"), data::read_region(&through, &inner).unwrap());
                }
            }
        }
    }

    /// With `BLOSC_NTHREADS=8`, a write on four threads --- the process's
    /// first Blosc2 compression --- starts four threads and no more: Blosc2
    /// starts none of its own inside a worker.  It stores what one thread
    /// stores.  HDF5's write still spreads each chunk's blocks over Blosc2's
    /// threads, through the threads callback.  In a process of its own, whose
    /// environment is set before Blosc2 reads it.
    #[cfg(target_os = "linux")]
    #[test]
    fn s14_2_blosc2_starts_no_thread_of_its_own_in_a_worker() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        const CHILD: &str = "MEDH5_TEST_BLOSC_THREADS_CHILD";
        let threads = || std::fs::read_dir("/proc/self/task").unwrap().count();
        // The most threads `f` had running at once beside those before it.
        let started = |f: &mut dyn FnMut()| {
            let (done, most) = (AtomicBool::new(false), AtomicUsize::new(0));
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    while !done.load(Ordering::SeqCst) {
                        most.fetch_max(threads(), Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_micros(100));
                    }
                });
                // The watcher is running: every thread from here is `f`'s.
                while most.load(Ordering::SeqCst) == 0 {
                    std::thread::yield_now();
                }
                let before = threads();
                f();
                done.store(true, Ordering::SeqCst);
                most.load(Ordering::SeqCst).saturating_sub(before)
            })
        };
        if std::env::var_os(CHILD).is_some() {
            crate::h5::init();
            let dir = tempfile::tempdir().unwrap();
            let file = crate::h5::file::create_truncate(&dir.path().join("t.h5")).unwrap();
            // Chunks of several Blosc2 blocks each.
            let shape = [128usize, 192, 192];
            let values = volume(&shape, DType::I16, &mut Rng::new(8));
            let layout = Layout {
                chunks: Some(vec![64, 192, 192]),
                filters: resolve_profile(Some("balanced")).unwrap().codec(Role::Image).filters(),
            };
            let mut found = Vec::new();
            for (k, workers) in [4, 1].into_iter().enumerate() {
                let ds = data::create_empty(&file, &format!("d{k}"), DType::I16, &shape, &layout).unwrap();
                let writer = Chunks::of(&ds, DType::I16).unwrap();
                let n = started(&mut || writer.write_rows(&values, 0, workers).unwrap());
                assert!(n <= workers, "{n} threads started by a write on {workers}");
                found.push(stored(&ds, &writer));
            }
            assert_eq!(found[0], found[1]);
            let mut through = None;
            let n = started(&mut || through = Some(data::create(&file, "hdf5", &values, &layout).unwrap()));
            let through = through.unwrap();
            assert!(n > 0, "HDF5's write shares a chunk's blocks out");
            // Its threads store each chunk's blocks in the order they finish
            // them, so its bytes are its own; the values are the same.
            assert_eq!(data::read(&through).unwrap(), values);
            assert_eq!(data::read(&file.dataset("d0").unwrap()).unwrap(), values);
            return;
        }
        let name = "h5::chunks::tests::s14_2_blosc2_starts_no_thread_of_its_own_in_a_worker";
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", name, "--test-threads=1"])
            .env(CHILD, "1")
            .env("BLOSC_NTHREADS", "8")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let run = crate::h5::spawn(&mut command).unwrap().wait_with_output().unwrap();
        let out = String::from_utf8_lossy(&run.stdout);
        assert!(run.status.success() && out.contains("1 passed"), "{out}{}", String::from_utf8_lossy(&run.stderr));
    }

    /// What the writer cannot store as HDF5 would is left to HDF5.
    #[test]
    fn s14_2_what_a_chunk_write_cannot_store_exactly_goes_through_hdf5() {
        crate::h5::init();
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("f.h5")).unwrap();
        let shape = [24usize, 40, 40];
        let chunks = Some(vec![8, 16, 16]);
        let blosc2 = resolve_profile(Some("balanced")).unwrap().codec(Role::Image).filters();
        let gzip = resolve_profile(Some("portable")).unwrap().codec(Role::Image).filters();
        let make =
            |name: &str, dtype: DType, layout: Layout| data::create_empty(&file, name, dtype, &shape, &layout).unwrap();
        let portable = make("portable", DType::I16, Layout { chunks: chunks.clone(), filters: gzip });
        assert!(Chunks::of(&portable, DType::I16).is_none(), "another codec");
        let contiguous = make("contiguous", DType::I16, Layout { chunks: None, filters: Vec::new() });
        assert!(Chunks::of(&contiguous, DType::I16).is_none(), "no chunks");
        let blosc = make("blosc2", DType::I16, Layout { chunks: chunks.clone(), filters: blosc2.clone() });
        assert!(Chunks::of(&blosc, DType::I16).is_some());
        assert!(Chunks::of(&blosc, DType::U16).is_none(), "a type HDF5 would convert");
        let mask = make("mask", DType::Bool, Layout { chunks, filters: blosc2 });
        assert!(Chunks::of(&mask, DType::Bool).is_none(), "an enumeration");
    }

    /// The stored chunks of each case through HDF5's own write, hashed in C
    /// order of origins: what 2.2.2's filter stored, before its compression
    /// became the chunk writer's too.
    const PINNED: [&str; 8] = [
        "4c26671a50e7882798ec1b5fdb1f57f0726217c7a7c8841b73760e9428dcdd14",
        "56d995af6a5dd6c90edfcc9d7911c254de8bb72f58a05fef8a006e5065242988",
        "55560d8c215e99ebedf961b7bdcdb1c5b047c1a97328eee0e6b50f03bbf87cd7",
        "dbd3bbfdc723d37b876aff10cda7ced5d6a76cf044c0a01acd10f40e476e9f61",
        "353133886b76502f5293a42ff2254325e4f919662930348bc224f99ca3e17fea",
        "a1bcd2cbdbfd81f038fee7543e42fc3b73b75647c4b9826edb5f0625627f5178",
        "dc02d22c3562dc3e0a0d3a733e4fbf9c1999f7174be8426e2503e8f11a5c6dd2",
        "9939121d8478c6ec21405dcf8fe4fca8ef846d4d23dc0a10b5d7c80ceb2fd32b",
    ];

    #[test]
    fn s14_2_the_filter_stores_the_chunks_2_2_2_stored() {
        crate::h5::init();
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("p.h5")).unwrap();
        let mut rng = Rng::new(12);
        let mut found = Vec::new();
        for (k, (profile, role, dtype, shape, chunks)) in CASES.into_iter().enumerate() {
            let values = volume(shape, dtype, &mut rng);
            let layout = Layout {
                chunks: Some(chunks.to_vec()),
                filters: resolve_profile(Some(profile)).unwrap().codec(role).filters(),
            };
            let ds = data::create(&file, &format!("d{k}"), &values, &layout).unwrap();
            let mut digest = crate::digest::Hasher::new("sha256").unwrap();
            for (_, mask, bytes) in stored(&ds, &Chunks::of(&ds, dtype).unwrap()) {
                digest.update(&mask.to_le_bytes());
                digest.update(&bytes);
            }
            found.push(digest.hexdigest());
        }
        assert_eq!(found, PINNED);
    }
}
