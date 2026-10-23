//! Digests of many datasets at once, on the thread setting's workers
//! ([`crate::parallel::threads`]).
//!
//! A digest is one hash over its dataset's §13.1 stream, so a dataset is
//! hashed by one worker at a time, in order; several datasets are hashed at
//! once.  The calling thread makes every HDF5 call: it reads each slab as
//! [`super::digest::dataset_digest_inspected`] cuts it --- the stored chunks
//! of a slab the Blosc2 blocks can give, else the slab's values through
//! HDF5 --- and the workers decompress the chunks, place them in C order and
//! hash.  Each digest, and every block an inspector is handed, is the one
//! the sequential pass gives: the slabs are the same, and a slab's values are
//! HDF5's ([`crate::h5::window`]'s rule).

use std::collections::VecDeque;

use super::digest::{dataset_digest_inspected, feed_header, row_bytes, Inspect, Slabs, LE_PIECE_BYTES, STREAM_BYTES};
use crate::array::{Index, NdArray, Slice};
use crate::digest::Hasher;
use crate::h5::data::{self, Kind, Region};
use crate::h5::window::Frames;
use crate::{Error, Result};

/// A check of a dataset's stored values, handed each block of it the digest
/// pass reads, in C order, on whichever worker hashes it.
pub(crate) type Inspector<'a> = Box<dyn FnMut(&NdArray) -> Result<()> + Send + 'a>;

/// One dataset to digest.
pub(crate) struct Job<'a> {
    pub(crate) ds: hdf5::Dataset,
    /// The path the digest names (relative to the sample root).
    pub(crate) path: String,
    pub(crate) algo: String,
    pub(crate) inspect: Option<Inspector<'a>>,
}

impl<'a> Job<'a> {
    pub(crate) fn new(ds: hdf5::Dataset, path: &str, algo: &str) -> Job<'a> {
        Job { ds, path: path.to_string(), algo: algo.to_string(), inspect: None }
    }
}

/// Each job's digest, as [`dataset_digest_inspected`] gives it, in the order
/// of `jobs`: on the thread setting's workers, or one after another on this
/// thread when it is 1.  `Err` only when the setting cannot be read.
pub(crate) fn digests(jobs: Vec<Job<'_>>) -> Result<Vec<Result<String>>> {
    Ok(digests_on(crate::parallel::threads()?, jobs))
}

/// [`digests`] on `threads` workers.
pub(crate) fn digests_on(threads: usize, jobs: Vec<Job<'_>>) -> Vec<Result<String>> {
    digests_in_slabs(threads, STREAM_BYTES, jobs)
}

/// [`digests_on`], reading about `budget` bytes at a time, as
/// [`dataset_digest_inspected`]'s `stream_bytes`: the slabs change the reads
/// and the blocks an inspector is handed, never a digest.
fn digests_in_slabs(threads: usize, budget: usize, mut jobs: Vec<Job<'_>>) -> Vec<Result<String>> {
    if threads <= 1 || jobs.is_empty() {
        return jobs.iter_mut().map(|job| sequential(job, budget)).collect();
    }
    // The header is hashed, and what to read worked out, here.
    let mut states = Vec::with_capacity(jobs.len());
    let mut plans = Vec::with_capacity(jobs.len());
    for job in &mut jobs {
        let inspect = job.inspect.take();
        match plan(job, budget) {
            Ok((hasher, plan)) => {
                states.push(Stream { hasher: Some(hasher), inspect });
                plans.push(Ok(plan));
            }
            Err(e) => {
                states.push(Stream { hasher: None, inspect });
                plans.push(Err(e));
            }
        }
    }
    let results = crate::parallel::streams(
        threads,
        states,
        |feed| feed_all(feed, threads, budget, &jobs, plans),
        decode,
        consume,
    );
    jobs.iter_mut()
        .zip(results)
        .map(|(job, (stream, failed))| match failed {
            None => Ok(stream.hasher.expect("a stream that did not fail has its hasher").finish()),
            Some(Failure::Error(e)) => Err(e),
            Some(Failure::Undecoded(axes)) => {
                // The blocks did not decompress: HDF5 reads the slab, and
                // fails as the sequential pass does.  HDF5 reading what
                // Blosc2 could not is not expected; if it does, the dataset
                // is digested again, here, as the sequential pass would.
                data::read_axes(&job.ds, &axes)?;
                job.inspect = stream.inspect;
                sequential(job, budget)
            }
        })
        .collect()
}

/// A job digested on this thread, as before there were workers.
fn sequential(job: &mut Job<'_>, budget: usize) -> Result<String> {
    let inspect: Option<Inspect<'_>> = match job.inspect.as_mut() {
        Some(f) => Some(&mut **f),
        None => None,
    };
    dataset_digest_inspected(&job.ds, &job.path, &job.algo, budget, inspect)
}

/// A dataset's hash and its inspector, which one worker at a time holds.
struct Stream<'a> {
    /// `None` for a dataset that failed before its first byte.
    hasher: Option<Hasher>,
    inspect: Option<Inspector<'a>>,
}

/// What one slab is when the calling thread has read it.
enum Item {
    /// Stored chunks a worker decompresses; the axes, should it not.
    Frames(Frames, Vec<(usize, usize, usize, bool)>),
    /// Values HDF5 read.
    Values(NdArray),
    /// Bytes hashed as they are: strings, other fixed-size types.
    Bytes(Vec<u8>),
}

/// What one slab is when a worker has decompressed it.
enum Decoded {
    Values(NdArray),
    Bytes(Vec<u8>),
}

/// Why a dataset has no digest from the workers.
enum Failure {
    Error(Error),
    /// The blocks of a slab did not decompress: the slab's axes, which
    /// HDF5 reads instead.
    Undecoded(Vec<(usize, usize, usize, bool)>),
}

/// What the calling thread reads of a dataset.
enum Plan {
    /// Nothing past the header: an empty dataset.
    Nothing,
    /// A dataset of no axes, read whole.
    Scalar,
    Strings,
    /// Rows of a fixed-size type that is not a number, `size` bytes each,
    /// in slabs.
    Stored {
        size: usize,
        slabs: Slabs,
        rows: usize,
    },
    /// A numeric dataset, a slab at a time.
    Blocks(Cursor),
}

/// The hasher with the dataset's header in it, and what to read, as
/// [`dataset_digest_inspected`] reads it.
fn plan(job: &Job<'_>, budget: usize) -> Result<(Hasher, Plan)> {
    let ds = &job.ds;
    let mut hasher = Hasher::new(&job.algo)?;
    let shape = ds.shape();
    let plan = match data::kind(ds)? {
        Kind::Strings => {
            feed_header(&mut hasher, &job.path, "|O", &shape);
            Plan::Strings
        }
        Kind::Other(t) => {
            let Some(size) = data::stored_element_size(ds)? else {
                return Err(Error::Type(format!(
                    "{} holds {t}, data of variable length or references, which have no digest (§13.1)",
                    ds.name()
                )));
            };
            feed_header(&mut hasher, &job.path, &format!("|V{size}"), &shape);
            if shape.contains(&0) {
                Plan::Nothing
            } else {
                let rows = shape.first().copied().unwrap_or(1);
                let row = shape.iter().skip(1).try_fold(size, |n, e| n.checked_mul(*e)).unwrap_or(usize::MAX);
                let step = (budget / row.max(1)).clamp(1, rows);
                let chunk = data::chunks(ds).and_then(|c| c.first().copied());
                Plan::Stored { size, slabs: Slabs::new(step, chunk, row, budget), rows }
            }
        }
        Kind::Numeric(dtype) => {
            feed_header(&mut hasher, &job.path, dtype.numpy_str(), &shape);
            if shape.contains(&0) {
                Plan::Nothing
            } else if shape.is_empty() {
                Plan::Scalar
            } else {
                data::ensure_allocatable(ds, &shape[1..], dtype.itemsize())?;
                Plan::Blocks(Cursor::new(ds, &shape, dtype.itemsize(), budget))
            }
        }
    };
    Ok((hasher, plan))
}

/// The slabs of a numeric dataset, one at a time, as
/// [`super::digest::dataset_digest_inspected`]'s `for_each_block` cuts them.
struct Cursor {
    shape: Vec<usize>,
    axis: usize,
    slabs: Slabs,
    /// The index of the axes before `axis`, read one at a time.
    outer: Vec<usize>,
    /// The next slab's first row along `axis`.
    start: usize,
    done: bool,
}

impl Cursor {
    fn new(ds: &hdf5::Dataset, shape: &[usize], itemsize: usize, budget: usize) -> Cursor {
        let mut axis = 0;
        while axis + 1 < shape.len() && row_bytes(shape, axis, itemsize) > budget {
            axis += 1;
        }
        let row = row_bytes(shape, axis, itemsize);
        let chunk = data::chunks(ds).and_then(|c| c.get(axis).copied());
        let slabs = Slabs::new((budget / row).max(1), chunk, row, budget);
        Cursor { shape: shape.to_vec(), axis, slabs, outer: vec![0; axis], start: 0, done: false }
    }

    /// The next slab's index, or `None` past the last.
    fn next(&mut self) -> Option<Vec<Index>> {
        if self.done {
            return None;
        }
        let n = self.shape[self.axis];
        let stop = self.slabs.end(self.start, n);
        let mut index: Vec<Index> = self.outer.iter().map(|i| Index::At(*i as i64)).collect();
        index.push(Index::Slice(Slice::new(self.start as i64, stop as i64)));
        self.start = stop;
        if self.start >= n {
            self.start = 0;
            // The next index of the axes read one at a time, the last fastest.
            let mut at = self.axis;
            loop {
                if at == 0 {
                    self.done = true;
                    break;
                }
                at -= 1;
                self.outer[at] += 1;
                if self.outer[at] < self.shape[at] {
                    break;
                }
                self.outer[at] = 0;
            }
        }
        Some(index)
    }
}

/// Read every dataset and hand each slab to its stream: up to `threads`
/// numeric datasets at once, a slab of each in turn, so that as many are
/// hashed at once; the others whole, at their turn.
fn feed_all(
    feed: &crate::parallel::Feed<'_, Stream<'_>, Item, Decoded, Failure>,
    threads: usize,
    budget: usize,
    jobs: &[Job<'_>],
    plans: Vec<Result<Plan>>,
) {
    let mut waiting: VecDeque<(usize, Result<Plan>)> = plans.into_iter().enumerate().collect();
    let mut open: Vec<(usize, Cursor)> = Vec::new();
    loop {
        while open.len() < threads {
            let Some((s, plan)) = waiting.pop_front() else { break };
            match plan {
                Err(e) => feed.fail(s, Failure::Error(e)),
                Ok(Plan::Blocks(cursor)) => open.push((s, cursor)),
                Ok(plan) => {
                    if let Err(e) = feed_whole(feed, s, &jobs[s].ds, plan, budget) {
                        feed.fail(s, Failure::Error(e));
                    }
                }
            }
        }
        if open.is_empty() {
            return;
        }
        open.retain_mut(|(s, cursor)| {
            if feed.failed(*s) {
                return false;
            }
            let Some(index) = cursor.next() else { return false };
            match read_slab(&jobs[*s].ds, &index, budget) {
                Ok(item) => {
                    feed.push(*s, item);
                    true
                }
                Err(e) => {
                    feed.fail(*s, Failure::Error(e));
                    false
                }
            }
        });
    }
}

/// A dataset that is not read a numeric slab at a time, fed whole.
fn feed_whole(
    feed: &crate::parallel::Feed<'_, Stream<'_>, Item, Decoded, Failure>,
    s: usize,
    ds: &hdf5::Dataset,
    plan: Plan,
    budget: usize,
) -> Result<()> {
    match plan {
        Plan::Nothing | Plan::Blocks(_) => {}
        Plan::Scalar => feed.push(s, Item::Values(data::read(ds)?)),
        Plan::Strings => data::for_each_string_slab(ds, budget, &mut |values| {
            let mut bytes = Vec::with_capacity(values.iter().map(|v| v.len() + 1).sum());
            for value in values {
                bytes.extend_from_slice(&value);
                bytes.push(0);
            }
            feed.push(s, Item::Bytes(bytes));
            Ok(())
        })?,
        Plan::Stored { size, slabs, rows } => {
            let mut start = 0;
            while start < rows && !feed.failed(s) {
                let stop = slabs.end(start, rows);
                feed.push(s, Item::Bytes(data::read_stored_rows(ds, size, start, stop - start)?));
                start = stop;
            }
        }
    }
    Ok(())
}

/// One slab: its stored chunks, where a worker can decompress them all and
/// they hold no more than `budget` bytes, else its values, read here.
fn read_slab(ds: &hdf5::Dataset, index: &[Index], budget: usize) -> Result<Item> {
    Ok(match data::read_region_stored(ds, index, budget)? {
        Region::Values(values) => Item::Values(values),
        Region::Stored(frames, axes) if frames.fit() => Item::Frames(frames, axes),
        Region::Stored(_, axes) => Item::Values(data::read_axes(ds, &axes)?),
    })
}

/// A worker's first step: the slab's values out of its stored chunks.
fn decode(item: Item) -> std::result::Result<Decoded, Failure> {
    match item {
        Item::Frames(frames, axes) => match frames.decode() {
            Ok(Some(values)) => Ok(Decoded::Values(values)),
            Ok(None) => Err(Failure::Undecoded(axes)),
            Err(e) => Err(Failure::Error(e)),
        },
        Item::Values(values) => Ok(Decoded::Values(values)),
        Item::Bytes(bytes) => Ok(Decoded::Bytes(bytes)),
    }
}

/// A worker's second step, one stream at a time and in its order: inspect
/// the slab and hash it.
fn consume(stream: &mut Stream<'_>, decoded: Decoded) -> std::result::Result<(), Failure> {
    let hasher = stream.hasher.as_mut().expect("a stream fed slabs has its hasher");
    match decoded {
        Decoded::Values(values) => {
            if let Some(inspect) = stream.inspect.as_mut() {
                inspect(&values).map_err(Failure::Error)?;
            }
            values.for_each_le_piece(LE_PIECE_BYTES, &mut |bytes| hasher.update(bytes));
        }
        Decoded::Bytes(bytes) => hasher.update(&bytes),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::DType;
    use crate::h5::data::Layout;
    use crate::h5sys::{h5d, h5p, h5s, h5t};
    use crate::rng::Rng;
    use crate::storage::codecs::{resolve_profile, Role};

    /// Runs and noise, so chunks and blocks compress unevenly.
    fn volume(shape: &[usize], dtype: DType, rng: &mut Rng) -> NdArray {
        let n: usize = shape.iter().product();
        let values: Vec<f64> = (0..n)
            .map(|i| if (i / 31) % 3 == 0 { ((i / 31) % 50) as f64 } else { rng.integer(0, 120).unwrap() as f64 })
            .collect();
        NdArray::from_vec(shape, values).unwrap().astype(dtype)
    }

    fn layout(profile: &str, chunks: &[usize]) -> Layout {
        Layout {
            chunks: Some(chunks.to_vec()),
            filters: resolve_profile(Some(profile)).unwrap().codec(Role::Image).filters(),
        }
    }

    /// Store `bytes` at the chunk `origin` with the filter mask `mask`: 0 as
    /// if the filters had written them, 1 as stored with the first skipped.
    fn plant_chunk(ds: &hdf5::Dataset, origin: &[u64], mask: u32, bytes: &[u8]) {
        let status = crate::h5::locked(|| unsafe {
            h5d::H5Dwrite_chunk(ds.id(), h5p::H5P_DEFAULT, mask, origin.as_ptr(), bytes.len(), bytes.as_ptr().cast())
        });
        assert!(status >= 0);
    }

    /// A dataset stored big-endian, chunked: HDF5 converts it on the way
    /// out, so its stored bytes are not the values §13.1 hashes.
    fn big_endian(file: &hdf5::File, name: &str, values: &[i32], dims: &[u64; 2]) -> hdf5::Dataset {
        let cname = std::ffi::CString::new(name).unwrap();
        crate::h5::locked(|| unsafe {
            let dcpl = h5p::H5Pcreate(*h5p::H5P_CLS_DATASET_CREATE);
            let chunk = [4u64, 3];
            h5p::H5Pset_chunk(dcpl, 2, chunk.as_ptr());
            let space = h5s::H5Screate_simple(2, dims.as_ptr(), std::ptr::null());
            let id = h5d::H5Dcreate2(
                file.id(),
                cname.as_ptr(),
                *h5t::H5T_STD_I32BE,
                space,
                h5p::H5P_DEFAULT,
                dcpl,
                h5p::H5P_DEFAULT,
            );
            assert!(id >= 0);
            let written = h5d::H5Dwrite(
                id,
                *h5t::H5T_NATIVE_INT,
                h5s::H5S_ALL,
                h5s::H5S_ALL,
                h5p::H5P_DEFAULT,
                values.as_ptr().cast(),
            );
            assert!(written >= 0);
            h5d::H5Dclose(id);
            h5s::H5Sclose(space);
            h5p::H5Pclose(dcpl);
        });
        file.dataset(name).unwrap()
    }

    /// Every kind of dataset the digest pass reads, each in a way its slabs
    /// fall on and across chunk edges.
    fn corpus(file: &hdf5::File) -> Vec<(String, hdf5::Dataset)> {
        let mut rng = Rng::new(13);
        let mut out = Vec::new();
        let mut add = |name: String, ds: hdf5::Dataset| out.push((name, ds));
        let dtypes = [
            DType::U8,
            DType::I8,
            DType::U16,
            DType::I16,
            DType::U32,
            DType::I32,
            DType::U64,
            DType::I64,
            DType::F32,
            DType::F64,
            DType::Bool,
        ];
        for profile in ["training", "balanced", "archive", "portable"] {
            for dtype in dtypes {
                let name = format!("{profile}_{dtype:?}");
                // Chunks that split every axis, with short ones at the edges.
                let data = volume(&[13, 17, 19], dtype, &mut rng);
                add(name.clone(), data::create(file, &name, &data, &layout(profile, &[5, 8, 7])).unwrap());
            }
            let name = format!("{profile}_4d");
            let data = volume(&[3, 9, 10, 11], DType::F32, &mut rng);
            add(name.clone(), data::create(file, &name, &data, &layout(profile, &[1, 4, 5, 6])).unwrap());
            let name = format!("{profile}_1d");
            let data = volume(&[300], DType::I16, &mut rng);
            add(name.clone(), data::create(file, &name, &data, &layout(profile, &[64])).unwrap());
        }
        let data = volume(&[11, 13], DType::F64, &mut rng);
        add("contiguous".into(), data::create(file, "contiguous", &data, &Layout::default()).unwrap());
        let scalar = NdArray::from_vec(&[], vec![2.5f64]).unwrap();
        add("scalar".into(), data::create(file, "scalar", &scalar, &Layout::default()).unwrap());
        let empty = NdArray::zeros(DType::U8, &[0, 4]);
        add("empty".into(), data::create(file, "empty", &empty, &Layout::default()).unwrap());
        // Chunks never written read as the fill value.
        let sparse =
            data::create_empty(file, "sparse", DType::I16, &[24, 40, 40], &layout("balanced", &[8, 16, 16])).unwrap();
        data::write_region(&sparse, &volume(&[8, 16, 16], DType::I16, &mut rng), &[8, 16, 0]).unwrap();
        add("sparse".into(), sparse);
        // A chunk stored with the Blosc2 filter skipped.
        let data = volume(&[12, 20, 20], DType::U16, &mut rng);
        let skipped = data::create(file, "skipped", &data, &layout("training", &[6, 10, 10])).unwrap();
        let raw = volume(&[6, 10, 10], DType::U16, &mut rng);
        plant_chunk(&skipped, &[6, 10, 0], 1, &raw.le_bytes());
        add("skipped".into(), skipped);
        let values: Vec<i32> = (0..12 * 10).map(|i| i * 7919 - 400).collect();
        add("big_endian".into(), big_endian(file, "big_endian", &values, &[12, 10]));
        let strings: Vec<String> = (0..50).map(|i| format!("value {i} {}", "x".repeat(i % 7))).collect();
        add("strings".into(), data::create_strings(file, "strings", &strings).unwrap());
        #[derive(hdf5::H5Type, Clone, Copy)]
        #[repr(C)]
        struct Row {
            id: i32,
            value: f32,
        }
        let rows: Vec<Row> = (0..40).map(|i| Row { id: i, value: i as f32 / 3.0 }).collect();
        let compound = file.new_dataset::<Row>().shape([40]).chunk([16]).create("compound").unwrap();
        compound.write(&rows).unwrap();
        add("compound".into(), compound);
        out
    }

    fn jobs(datasets: &[(String, hdf5::Dataset)]) -> Vec<Job<'static>> {
        datasets.iter().map(|(name, ds)| Job::new(ds.clone(), name, "sha256")).collect()
    }

    fn shown(results: Vec<Result<String>>) -> Vec<std::result::Result<String, String>> {
        results.into_iter().map(|r| r.map_err(|e| e.to_string())).collect()
    }

    #[test]
    fn s13_1_digests_on_workers_are_the_sequential_ones() {
        let _serial = crate::parallel::tests::serial();
        crate::h5::init();
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("d.h5")).unwrap();
        let datasets = corpus(&file);
        for budget in [700, 4096, 1 << 20] {
            let expected = shown(digests_in_slabs(1, budget, jobs(&datasets)));
            for (k, (name, ds)) in datasets.iter().enumerate() {
                let one = dataset_digest_inspected(ds, name, "sha256", STREAM_BYTES, None).map_err(|e| e.to_string());
                assert_eq!(expected[k], one, "{name}: the slabs changed a digest");
            }
            for threads in [2, 4] {
                assert_eq!(
                    shown(digests_in_slabs(threads, budget, jobs(&datasets))),
                    expected,
                    "{threads} threads, {budget}"
                );
            }
        }
    }

    /// An inspector is handed the blocks the sequential pass hands it, in
    /// its order, whichever worker hashes them.
    #[test]
    fn s13_1_an_inspector_sees_the_sequential_blocks() {
        let _serial = crate::parallel::tests::serial();
        crate::h5::init();
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("i.h5")).unwrap();
        let datasets: Vec<_> = corpus(&file).into_iter().filter(|(name, _)| name != "strings").collect();
        let seen = |threads: usize| -> Vec<Vec<NdArray>> {
            let mut blocks: Vec<Vec<NdArray>> = vec![Vec::new(); datasets.len()];
            let jobs: Vec<Job<'_>> = datasets
                .iter()
                .zip(blocks.iter_mut())
                .map(|((name, ds), into)| {
                    let mut job = Job::new(ds.clone(), name, "sha256");
                    job.inspect = Some(Box::new(move |block: &NdArray| {
                        into.push(block.clone());
                        Ok(())
                    }));
                    job
                })
                .collect();
            assert!(digests_in_slabs(threads, 4096, jobs).iter().all(Result::is_ok));
            blocks
        };
        assert_eq!(seen(4), seen(1));
    }

    /// A chunk whose blocks do not decompress fails its dataset with HDF5's
    /// error, as the sequential pass does, and only its dataset.
    #[test]
    fn s13_1_an_undecodable_chunk_fails_as_it_does_in_sequence() {
        let _serial = crate::parallel::tests::serial();
        crate::h5::init();
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("c.h5")).unwrap();
        let mut rng = Rng::new(5);
        let good = data::create(
            &file,
            "good",
            &volume(&[12, 20, 20], DType::I16, &mut rng),
            &layout("training", &[6, 10, 10]),
        )
        .unwrap();
        let bad =
            data::create(&file, "bad", &volume(&[12, 20, 20], DType::I16, &mut rng), &layout("training", &[6, 10, 10]))
                .unwrap();
        // A B2ND frame whose blocks are damaged: its metadata parses, and
        // its blocks do not decompress.  Found by damaging 16 bytes at a time.
        let stored = |ds: &hdf5::Dataset, at: i64| match data::read_region_stored(
            ds,
            &[Index::Slice(Slice::new(at, at + 6))],
            usize::MAX,
        )
        .unwrap()
        {
            Region::Stored(frames, _) => Some(frames),
            Region::Values(_) => None,
        };
        let (_, original) = crate::h5::ops::read_raw_chunk(&bad, &[6, 0, 0]).unwrap();
        let damaged = (0..original.len()).step_by(16).find_map(|at| {
            let mut frame = original.clone();
            for b in &mut frame[at..(at + 16).min(original.len())] {
                *b ^= 0x5a;
            }
            plant_chunk(&bad, &[6, 0, 0], 0, &frame);
            let frames = stored(&bad, 6).expect("a slab of stored chunks");
            (frames.fit() && frames.decode().unwrap().is_none()).then_some(frame)
        });
        assert!(damaged.is_some(), "no damage both parses and fails to decompress");
        // Not a frame at all.
        let worse = data::create(
            &file,
            "worse",
            &volume(&[12, 20, 20], DType::I16, &mut rng),
            &layout("training", &[6, 10, 10]),
        )
        .unwrap();
        plant_chunk(&worse, &[0, 10, 0], 0, &[7u8; 64]);
        // An intact frame is read; one that is not a frame is not handed to
        // a worker.
        let intact = stored(&good, 6).expect("a slab of stored chunks");
        assert!(intact.fit() && intact.decode().unwrap().is_some());
        assert!(!stored(&worse, 0).expect("a slab of stored chunks").fit());
        let datasets = vec![("bad".to_string(), bad), ("good".into(), good), ("worse".into(), worse)];
        let expected = shown(digests_in_slabs(1, 4096, jobs(&datasets)));
        assert!(expected[0].is_err() && expected[1].is_ok() && expected[2].is_err(), "{expected:?}");
        assert_eq!(shown(digests_in_slabs(4, 4096, jobs(&datasets))), expected);
    }
}
