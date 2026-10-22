//! Hostile files that are not conformance cases: regression fixtures for the
//! denial-of-service bugs 2.0.2 fixed, which `.github/scripts/damaged.py
//! --structural` holds every binary to report within a time and a memory
//! bound.
//!
//! Each is a valid sample with one structure planted in it --- a chain of
//! groups deeper than [`MAX_DEPTH`], a dataset naming a filter nobody
//! registered, a Zstandard frame that decodes to far more than its chunk, a
//! Blosc2 chunk that claims more than its dataset's chunk --- written by the
//! engine's own writer and the HDF5 calls its unit tests make, or one
//! structure of HDF5's own edited in the bytes it wrote: a global heap object
//! smaller than its header, which HDF5 2.2.0 decoded without end.  Random damage
//! (`damaged.py`'s other mode) rarely produces these shapes.  They are not in
//! the corpus: the limits they cross are this implementation's, not the
//! specification's.

use std::path::{Path, PathBuf};

use hdf5::filters::Filter;

use super::build::{base, mutate, SHAPE};
use crate::array::{DType, NdArray};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::data::{self, Layout};
use crate::h5::ops::{self, MAX_DEPTH};
use crate::h5sys::{h5d, h5p, h5s, h5t, h5z};
use crate::{Error, Result};

/// Where each fixture plants its structure.
const GROUP: &str = "x_robustness";

/// A digest no dataset has, so that verifying the dataset reads it.
fn any_digest() -> AttrValue {
    AttrValue::Str(format!("sha256:{}", "0".repeat(64)))
}

/// What plants one fixture's structure in a valid sample.
type Plant = fn(&hdf5::Group) -> Result<()>;

/// What edits the bytes HDF5 wrote, once the file is closed.
type Edit = fn(&mut [u8]) -> Result<()>;

/// Write each fixture into `dir`, and return their paths.
#[doc(hidden)]
pub fn write(dir: &Path) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let fixtures: [(&str, Plant, Option<Edit>); 5] = [
        ("deep-groups", deep_groups, None),
        ("unregistered-filter", unregistered_filter, None),
        ("zstd-frame-larger-than-its-chunk", zstd_bomb, None),
        ("blosc2-chunk-larger-than-its-dataset-chunk", blosc2_overclaim, None),
        ("global-heap-object-smaller-than-its-header", heap_strings, Some(empty_free_space)),
    ];
    let mut out = Vec::new();
    for (name, plant, edit) in fixtures {
        let path = dir.join(format!("{name}.medh5"));
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        base(&path, &SHAPE)?;
        mutate(&path, plant)?;
        if let Some(edit) = edit {
            let mut bytes = std::fs::read(&path)?;
            edit(&mut bytes)?;
            std::fs::write(&path, bytes)?;
        }
        out.push(path);
    }
    Ok(out)
}

/// A chain of groups 4,096 deep: HDF5's own walk of it recursed without
/// limit, and 30,000 overflowed the stack.  Opening it refuses it past
/// [`MAX_DEPTH`].
fn deep_groups(root: &hdf5::Group) -> Result<()> {
    let mut group = ops::new_group(root, GROUP)?;
    for _ in 0..64 * MAX_DEPTH {
        group = ops::new_group(&group, "a")?;
    }
    Ok(())
}

/// One chunked `uint8` dataset of `len` bytes with `filters`, in the planted
/// group, carrying a digest: verifying it reads it.
fn planted(root: &hdf5::Group, len: usize, filters: Vec<Filter>) -> Result<hdf5::Dataset> {
    let group = ops::new_group(root, GROUP)?;
    let layout = Layout { chunks: Some(vec![len]), filters };
    let ds = data::create_empty(&group, "data", DType::U8, &[len], &layout)?;
    attrs::write(&ds, "digest", &any_digest())?;
    Ok(ds)
}

/// Store `bytes` as the dataset's first chunk, as if its filters had written
/// them.
fn write_raw_chunk(ds: &hdf5::Dataset, bytes: &[u8]) -> Result<()> {
    let offset = [0u64];
    // SAFETY: under HDF5's lock, for a dataset open for the call; `bytes`
    // is valid for its length.
    let status = crate::h5::locked(|| unsafe {
        h5d::H5Dwrite_chunk(ds.id(), h5p::H5P_DEFAULT, 0, offset.as_ptr(), bytes.len(), bytes.as_ptr().cast())
    });
    if status < 0 {
        return Err(Error::Io(format!("could not write the chunk of {}", ds.name())));
    }
    Ok(())
}

/// A dataset naming filter 40000, which nobody registers --- optional, as
/// h5py names a filter it has not got.  HDF5 searched its plugin directories
/// for a library to run; the engine turns that off, and the read fails.
fn unregistered_filter(root: &hdf5::Group) -> Result<()> {
    let group = ops::new_group(root, GROUP)?;
    let dims = [4096u64];
    // SAFETY: under HDF5's lock; every identifier made here is closed here.
    let id = crate::h5::locked(|| unsafe {
        let dcpl = h5p::H5Pcreate(*h5p::H5P_CLS_DATASET_CREATE);
        h5p::H5Pset_chunk(dcpl, 1, dims.as_ptr());
        h5p::H5Pset_filter(dcpl, 40000, h5z::H5Z_FLAG_OPTIONAL, 0, std::ptr::null());
        let space = h5s::H5Screate_simple(1, dims.as_ptr(), std::ptr::null());
        let id = h5d::H5Dcreate2(
            group.id(),
            c"data".as_ptr(),
            *h5t::H5T_NATIVE_UCHAR,
            space,
            h5p::H5P_DEFAULT,
            dcpl,
            h5p::H5P_DEFAULT,
        );
        h5s::H5Sclose(space);
        h5p::H5Pclose(dcpl);
        if id >= 0 {
            h5d::H5Dclose(id);
        }
        id
    });
    if id < 0 {
        return Err(Error::Io("could not create a dataset naming filter 40000".into()));
    }
    let ds = group.dataset("data")?;
    attrs::write(&ds, "digest", &any_digest())?;
    write_raw_chunk(&ds, &[0x5A; 4096])
}

/// A Zstandard frame of 16,384 run-length blocks of 128 KiB --- 64 KiB that
/// decode to 2 GiB --- with no content size in its header, for a 64 KiB
/// chunk.  The filter grew its buffer, doubling, until the frame ended.
fn zstd_bomb(root: &hdf5::Group) -> Result<()> {
    const BLOCK: u32 = 128 * 1024;
    const BLOCKS: usize = 16_384;
    let ds = planted(root, 64 * 1024, vec![Filter::user(medh5_sys::ZSTD_FILTER_ID as _, &[3])])?;
    // The magic number, then a frame header with no content size and a
    // 128 KiB window; each block is a three-byte header and its one byte.
    let mut frame = 0xFD2F_B528u32.to_le_bytes().to_vec();
    frame.extend([0x00, 0x38]);
    for i in 0..BLOCKS {
        let header = u32::from(i + 1 == BLOCKS) | (1 << 1) | (BLOCK << 3);
        frame.extend(&header.to_le_bytes()[..3]);
        frame.push(0x41);
    }
    write_raw_chunk(&ds, &frame)
}

/// A Blosc2 chunk of 4 MiB of zeros --- a few hundred bytes --- stored as the
/// one chunk of a 4 KiB dataset.  The filter trusted the chunk's header for
/// the size it allocated and decompressed.
fn blosc2_overclaim(root: &hdf5::Group) -> Result<()> {
    let blosc2 = || vec![Filter::user(medh5_sys::BLOSC2_FILTER_ID as _, &[0, 0, 0, 0, 5, 1, 0])];
    let large = 4 << 20;
    let scratch = ops::new_group(root, "x_scratch")?;
    let zeros = NdArray::from_vec(&[large], vec![0f64; large])?.astype(DType::U8);
    let source = data::create(&scratch, "zeros", &zeros, &Layout { chunks: Some(vec![large]), filters: blosc2() })?;
    let (_, chunk) = ops::read_raw_chunk(&source, &[0])?;
    drop(source);
    drop(scratch);
    root.unlink("x_scratch")?;
    let ds = planted(root, 4096, blosc2())?;
    write_raw_chunk(&ds, &chunk)
}

/// A string attribute on the planted group: one more object in the file's
/// global heap, so that a collection has free space after its objects.
fn heap_strings(root: &hdf5::Group) -> Result<()> {
    let group = ops::new_group(root, GROUP)?;
    attrs::write(&group, "note", &AttrValue::Str("a string in the global heap".into()))
}

/// The length of a global heap collection's header, and of each object's:
/// the files the engine writes state lengths in 8 bytes.
const HEAP_HEADER: usize = 16;

/// Every global heap collection whose objects end in a free-space object
/// gets that object's size set to 0.  HDF5 2.2.0 moved on by the size each
/// object states, so the walk stayed where it was, and reading any string in
/// the collection --- the file's own attributes among them --- never ended.
/// A collection is edited only when its objects lead exactly to its end, so
/// bytes that merely look like a signature are left alone.
fn empty_free_space(bytes: &mut [u8]) -> Result<()> {
    let u64_at = |bytes: &[u8], at: usize| -> Option<u64> {
        Some(u64::from_le_bytes(bytes.get(at..at.checked_add(8)?)?.try_into().ok()?))
    };
    let mut edited = 0;
    let mut at = 0;
    while let Some(found) = bytes.get(at..).and_then(|rest| rest.windows(4).position(|w| w == b"GCOL")) {
        let start = at + found;
        at = start + 4;
        let Some(size) = u64_at(bytes, start + 8).and_then(|size| usize::try_from(size).ok()) else { continue };
        let Some(end) = start.checked_add(size).filter(|&end| end <= bytes.len()) else { continue };
        if bytes[start + 4] != 1 || size < 4096 {
            continue;
        }
        let mut object = start + HEAP_HEADER;
        let mut free = None;
        while object + HEAP_HEADER <= end {
            let index = u16::from_le_bytes([bytes[object], bytes[object + 1]]);
            let Some(stated) = u64_at(bytes, object + 8).and_then(|n| usize::try_from(n).ok()) else { break };
            // An object's header, then its data padded to 8 bytes; the free
            // space states its whole size, header included.
            let need = match index {
                0 => {
                    free = Some(object);
                    stated
                }
                _ => {
                    stated.checked_add(7).map(|n| n & !7).and_then(|n| n.checked_add(HEAP_HEADER)).unwrap_or(usize::MAX)
                }
            };
            if need == 0 {
                break;
            }
            object = object.saturating_add(need);
        }
        if let (Some(free), true) = (free, object == end) {
            bytes[free + 8..free + 16].fill(0);
            edited += 1;
        }
    }
    if edited == 0 {
        return Err(Error::Io("no global heap collection with free space to edit".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    use serde_json::Value;

    /// The environment variable naming the file [`validate_in_a_child`]
    /// validates, and the prefix of the line its report is printed on.
    const CHILD: &str = "MEDH5_ROBUSTNESS_VALIDATE";
    const REPORT: &str = "medh5-robustness-report: ";

    /// Run by [`bounded`], in a process of its own: strict validation of the
    /// file [`CHILD`] names, its report printed as one line of JSON.  With no
    /// file named, it does nothing.
    #[test]
    fn validate_in_a_child() {
        let Some(path) = std::env::var_os(CHILD) else { return };
        let report = crate::validate::validate_file(Path::new(&path), "strict", None).unwrap();
        println!("{REPORT}{}", report.to_json());
    }

    /// The error diagnostics of strict validation of `path`, run in a child
    /// process this test binary starts, or `None` if it has not returned
    /// within `limit`, when the child is killed.  A fixture that hangs ---
    /// the global heap's, on an HDF5 without its patch --- holds HDF5's lock
    /// while it does, so a thread of this process would leave every other
    /// test that reads a file waiting with it.  The child is started through
    /// [`crate::h5::spawn`]: a plain start could catch another test's file
    /// between its open and its close-on-exec mark, and the child, which
    /// runs for seconds, would hold that file's lock until it exits.
    fn bounded(path: &Path, limit: Duration) -> Option<Vec<Value>> {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "conformance::robustness::tests::validate_in_a_child", "--nocapture"])
            .env(CHILD, path)
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = crate::h5::spawn(&mut command).unwrap();
        let mut stdout = child.stdout.take().unwrap();
        let reader = std::thread::spawn(move || {
            let mut out = String::new();
            stdout.read_to_string(&mut out).map(|_| out)
        });
        let deadline = Instant::now() + limit;
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if Instant::now() > deadline {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        let out = reader.join().unwrap().unwrap();
        assert!(status.success(), "{}: the child failed: {out}", path.display());
        let line = out.lines().find_map(|l| l.strip_prefix(REPORT)).expect("the child prints its report");
        let report: Value = serde_json::from_str(line).unwrap();
        let errors = report["diagnostics"].as_array().unwrap().iter().filter(|d| d["severity"] == "error");
        Some(errors.cloned().collect())
    }

    /// Each fixture is validated to a report with an error, never a panic,
    /// within 30 seconds, and the report names what was planted: the file is
    /// refused at open, or the planted dataset is the one that could not be
    /// read.  Each is [`bounded`], so one that hangs fails here.
    #[test]
    fn every_robustness_fixture_validates_to_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let paths = write(dir.path()).unwrap();
        assert_eq!(paths.len(), 5);
        for path in paths {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let errors = bounded(&path, Duration::from_secs(30)).unwrap_or_else(|| panic!("{name} hung"));
            let names = |d: &Value, key: &str| d[key].as_str().is_some_and(|s| s.contains(GROUP));
            assert!(
                errors.iter().any(|d| d["code"] == "E001" || names(d, "location") || names(d, "message")),
                "{name}: {errors:?}"
            );
        }
    }

    /// HDF5 2.2.0 never returned from the first string it read in such a
    /// file; the patched HDF5 refuses the heap, and the file with it, as
    /// unreadable and nothing else.
    #[test]
    fn a_global_heap_object_smaller_than_its_header_is_refused_in_bounded_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("heap.medh5");
        base(&path, &SHAPE).unwrap();
        mutate(&path, heap_strings).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        empty_free_space(&mut bytes).unwrap();
        std::fs::write(&path, bytes).unwrap();
        let errors = bounded(&path, Duration::from_secs(60)).expect("validation returns");
        let codes: Vec<_> = errors.iter().map(|d| d["code"].as_str().unwrap()).collect();
        assert_eq!(codes, ["E001"], "{errors:?}");
    }

    /// Only a collection whose objects lead exactly to its end is edited, and
    /// a file with none is an error rather than a fixture that plants nothing.
    #[test]
    fn the_heap_edit_touches_only_a_whole_collection() {
        let mut collection = b"GCOL\x01\0\0\0".to_vec();
        collection.extend(4096u64.to_le_bytes());
        // One object of 5 bytes, padded to 8, then the free space: the rest.
        collection.extend([1, 0, 1, 0, 0, 0, 0, 0]);
        collection.extend(5u64.to_le_bytes());
        collection.extend([0; 8]);
        let rest = 4096 - collection.len() as u64;
        collection.extend([0, 0, 0, 0, 0, 0, 0, 0]);
        collection.extend(rest.to_le_bytes());
        collection.resize(4096, 0);
        let mut file = b"GCOL, but not a collection".to_vec();
        file.extend(&collection);
        let free = file.len() - 4096 + 40 + 8;
        empty_free_space(&mut file).unwrap();
        assert_eq!(file[free..free + 8], [0; 8]);
        let mut truncated = collection[..4000].to_vec();
        assert!(empty_free_space(&mut truncated).is_err());
        assert_eq!(truncated, collection[..4000]);
    }
}
