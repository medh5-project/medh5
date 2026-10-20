//! Native dependencies of the medh5 format engine.
//!
//! This crate links, statically, into whatever depends on it:
//!
//! * **HDF5**, built from source by `hdf5-metno-sys` (re-exported as
//!   [`hdf5_sys`]);
//! * **C-Blosc2** 3.3.5, vendored, with LZ4, Zstd and zlib from their -sys
//!   crates;
//! * the **HDF5-Blosc2 filter** (HDF Group filter id 32026), upstream's own C
//!   source, so chunks are written and read exactly as `hdf5plugin` writes and
//!   reads them;
//! * the **HDF5 Zstandard filter** (id 32015) in the chunk format `hdf5plugin`
//!   uses, for datasets other tools wrote into a file's extension groups;
//! * [`medh5_b2nd_read_slice`], which decompresses only the part of a stored
//!   chunk a window covers.
//!
//! Nothing here interprets a MEDH5 file; the `medh5` crate does.  The entry
//! points beyond the raw bindings are [`register_blosc2_filter`] and
//! [`register_zstd_filter`], which must run before a dataset using either
//! filter is created or read.  Both filters are compiled in, so nothing here
//! needs HDF5's dynamically loaded plugins; the `medh5` crate turns their
//! loading off unless the user asks for it.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

pub use hdf5_metno_sys as hdf5_sys;

mod zstd_filter;
pub use zstd_filter::ZSTD_FILTER_ID;

// The build script's MSVC check, included here for its tests.
#[cfg(test)]
mod ndebug;

// The codec libraries are linked for their symbols alone; naming the crates
// keeps them in the link even though no Rust code here calls them.
use libz_sys as _;
use lz4_sys as _;
use zstd_sys as _;

/// The HDF Group's registered id for the Blosc2 filter.
pub const BLOSC2_FILTER_ID: u32 = 32026;

/// The vendored C-Blosc2 version: upstream's release, byte for byte.
///
/// 3.3.5 carries upstream's fixes to BloscLZ's bounds checks: in 3.3.2 a
/// match length accumulated past `i32::MAX` before it was checked, so a chunk
/// of a few megabytes could wrap it and write past the block (c-blosc2
/// 3.3.3 and 3.3.5).  Any Blosc2 chunk a file carries
/// reaches that decoder, whatever codec this crate writes with.
pub const BLOSC2_VERSION: &str = "3.3.5";

unsafe extern "C" {
    /// Registers the Blosc2 filter with the HDF5 library (upstream's
    /// `register_blosc2`).  Returns 1 when the filter is available.
    ///
    /// `version` and `date`, when both are non-null, receive `strdup`ed
    /// strings the caller must `free`.
    pub fn register_blosc2(version: *mut *mut c_char, date: *mut *mut c_char) -> c_int;

    fn blosc2_init();

    /// Decompress the slice `[start, stop)` of one stored chunk --- a B2ND
    /// frame of `frame_len` bytes --- into `out`, in C order, touching only
    /// the blocks the slice covers.
    ///
    /// Returns 0 on success; 1 when the frame is not a B2ND frame of rank
    /// `ndim`, shape `chunk_shape` and item size `typesize`, or the slice does
    /// not fit in it or in `out_len` bytes (read the chunk through HDF5
    /// instead); a negative Blosc2 error code when the blocks do not
    /// decompress.  `frame` is not modified.
    ///
    /// It touches no HDF5 state, so it needs no HDF5 lock, and each call
    /// builds its own Blosc2 contexts.
    pub fn medh5_b2nd_read_slice(
        frame: *const u8,
        frame_len: i64,
        ndim: i8,
        chunk_shape: *const i64,
        typesize: i32,
        start: *const i64,
        stop: *const i64,
        out: *mut c_void,
        out_len: i64,
    ) -> c_int;
}

/// Start C-Blosc2, once per process.  `blosc2_init` is not safe to enter
/// from two threads at once: a call that finds the library not yet started
/// registers the codecs afresh and initialises the global context's mutex,
/// under any other thread doing the same.  Later calls (the filter makes one
/// per chunk) return at once.
fn start_blosc2() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    // SAFETY: C-Blosc2's own initialisation, entered by one thread only.
    STARTED.call_once(|| unsafe { blosc2_init() });
}

/// Register the HDF5-Blosc2 filter with the process's HDF5 library.
///
/// Idempotent and cheap after the first call.  **The caller must hold HDF5's
/// global lock** (`hdf5::sync::sync`) --- the `medh5` crate does --- because the
/// HDF5 C library is not re-entrant.  Returns `false` when HDF5 refused the
/// registration.
///
/// # Safety
///
/// Calls into the HDF5 C library; see above for the locking requirement.
pub unsafe fn register_blosc2_filter() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    // 0 = not yet, 1 = registered, 2 = refused.
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Acquire) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    start_blosc2();
    let ok = unsafe { register_blosc2(std::ptr::null_mut(), std::ptr::null_mut()) } >= 0
        && unsafe { hdf5_sys::h5z::H5Zfilter_avail(BLOSC2_FILTER_ID as _) } > 0;
    STATE.store(if ok { 1 } else { 2 }, Ordering::Release);
    ok
}

/// Register the HDF5 Zstandard filter (id 32015) with the process's HDF5
/// library.
///
/// Idempotent; the same locking requirement as [`register_blosc2_filter`].
/// Returns `false` when HDF5 refused the registration.
///
/// # Safety
///
/// Calls into the HDF5 C library; the caller must hold HDF5's global lock.
pub unsafe fn register_zstd_filter() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Acquire) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    let ok = unsafe { zstd_filter::register() } && unsafe { hdf5_sys::h5z::H5Zfilter_avail(ZSTD_FILTER_ID as _) } > 0;
    STATE.store(if ok { 1 } else { 2 }, Ordering::Release);
    ok
}

/// HDF5 here is not built thread-safe, and the tests of a crate run in
/// parallel: a test that calls into HDF5 holds this.
#[cfg(test)]
static HDF5_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The Blosc2 filter run as HDF5 runs it --- `data` in a buffer of `buf_size`
/// bytes, or of `data`'s own size if larger --- forward or `reverse`, with the
/// filter values `values`: what it hands back, if anything.  What the tests
/// and the fuzz target (`fuzz/`) filter through; the caller holds HDF5's lock
/// where other threads may call into HDF5.
#[cfg(any(test, fuzzing))]
fn run_blosc2_filter(reverse: bool, values: &[std::os::raw::c_uint], data: &[u8], buf_size: usize) -> Option<Vec<u8>> {
    use std::os::raw::c_uint;
    unsafe extern "C" {
        fn blosc2_filter_function(
            flags: c_uint,
            cd_nelmts: usize,
            cd_values: *const c_uint,
            nbytes: usize,
            buf_size: *mut usize,
            buf: *mut *mut c_void,
        ) -> usize;
        fn malloc(size: usize) -> *mut c_void;
        fn free(ptr: *mut c_void);
    }
    // The filter starts C-Blosc2 itself; started here first, its call returns
    // at once whichever thread makes it.
    start_blosc2();
    let flags = if reverse { hdf5_sys::h5z::H5Z_FLAG_REVERSE } else { 0 };
    let mut size = buf_size.max(data.len()).max(1);
    // SAFETY: the buffer is allocated as the filter frees it, and holds
    // `data`; the filter frees or keeps it, and what it holds after is freed
    // here.
    unsafe {
        let mut buf = malloc(size);
        assert!(!buf.is_null());
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf.cast::<u8>(), data.len());
        let n = blosc2_filter_function(flags, values.len(), values.as_ptr(), data.len(), &mut size, &mut buf);
        let out = (n > 0).then(|| std::slice::from_raw_parts(buf.cast::<u8>(), n).to_vec());
        free(buf);
        out
    }
}

/// The filters run backwards on bytes a fuzzer chose, as HDF5 runs them
/// (`fuzz/`, through `cargo fuzz`, which sets `cfg(fuzzing)`).  Not part of
/// the crate's API: nothing else compiles it.
#[cfg(fuzzing)]
#[doc(hidden)]
pub mod fuzzing {
    /// The Zstandard filter (32015) decoding `frame` for a chunk buffer of
    /// `buf_size` bytes.
    pub fn zstd_decode(frame: &[u8], buf_size: usize) -> Option<Vec<u8>> {
        cleared(|| crate::zstd_filter::run_backwards(frame, buf_size))
    }

    /// The Blosc2 filter (32026) decoding `chunk` with the filter values
    /// `values`, for a chunk buffer of `buf_size` bytes.
    pub fn blosc2_decode(values: &[u32], chunk: &[u8], buf_size: usize) -> Option<Vec<u8>> {
        super::start_blosc2();
        cleared(|| super::run_blosc2_filter(true, values, chunk, buf_size))
    }

    /// A filter that fails pushes its error onto HDF5's error stack, which
    /// HDF5 clears as the next API call enters.  Called directly, nothing
    /// would: once the stack is full, each push's copies of its names leak.
    fn cleared<T>(decode: impl FnOnce() -> T) -> T {
        let out = decode();
        // SAFETY: clears the calling thread's default error stack.
        unsafe { crate::hdf5_sys::h5e::H5Eclear2(crate::hdf5_sys::h5e::H5E_DEFAULT) };
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::os::raw::c_uint;

    /// C-Blosc2's `blosc2_storage`: null pointers take its defaults.
    #[repr(C)]
    struct Blosc2Storage {
        contiguous: bool,
        urlpath: *mut c_char,
        cparams: *mut c_void,
        dparams: *mut c_void,
        io: *mut c_void,
    }

    /// C-Blosc2's `blosc2_codec`; the callbacks are never called here.
    #[repr(C)]
    struct Blosc2Codec {
        compcode: u8,
        compname: *const c_char,
        complib: u8,
        version: u8,
        encoder: Option<unsafe extern "C" fn()>,
        decoder: Option<unsafe extern "C" fn()>,
    }

    unsafe extern "C" {
        fn blosc2_get_version_string() -> *const c_char;
        fn blosc2_register_codec(codec: *mut Blosc2Codec) -> c_int;
        fn blosc2_compname_to_compcode(compname: *const c_char) -> c_int;
        fn blosc1_set_compressor(compname: *const c_char) -> c_int;
        fn blosc2_compress(
            clevel: c_int,
            doshuffle: c_int,
            typesize: i32,
            src: *const c_void,
            srcsize: i32,
            dest: *mut c_void,
            destsize: i32,
        ) -> c_int;
        fn blosc2_decompress(src: *const c_void, srcsize: i32, dest: *mut c_void, destsize: i32) -> c_int;
        fn blosclz_decompress(input: *const c_void, length: c_int, output: *mut c_void, maxout: c_int) -> c_int;
        fn blosc2_schunk_new(storage: *mut Blosc2Storage) -> *mut c_void;
        fn blosc2_schunk_to_buffer(schunk: *mut c_void, cframe: *mut *mut u8, needs_free: *mut bool) -> i64;
        fn blosc2_schunk_free(schunk: *mut c_void) -> c_int;
    }

    #[test]
    fn the_stated_version_is_the_vendored_one() {
        // SAFETY: a static string the library owns.
        let found = unsafe { CStr::from_ptr(blosc2_get_version_string()) };
        assert_eq!(found.to_str().unwrap(), BLOSC2_VERSION);
    }

    /// Two literal bytes, then a long match whose length continues in
    /// `0xFF` bytes, `continued` of them --- enough to carry a 32-bit
    /// accumulator past `i32::MAX` --- then its last length byte and its
    /// distance.
    fn overlong_match(continued: usize) -> Vec<u8> {
        let mut stream = vec![1, b'A', b'B', 0xE0];
        stream.resize(stream.len() + continued, 0xFF);
        stream.extend([0x00, 0x01]);
        stream
    }

    #[test]
    fn a_blosclz_match_length_past_i32_max_is_refused() {
        // 255 x 8,421,506 > i32::MAX: 3.3.2 added each byte before checking
        // and wrapped (undefined behaviour; in practice a negative length,
        // which then passed the bounds checks).  Refused before the addition.
        let stream = overlong_match(8_421_506);
        let mut out = [0u8; 64];
        // SAFETY: both buffers are valid for the lengths given.
        let n =
            unsafe { blosclz_decompress(stream.as_ptr().cast(), stream.len() as c_int, out.as_mut_ptr().cast(), 64) };
        assert_eq!(n, 0, "the decoder must refuse the stream");
    }

    #[test]
    fn a_blosc2_chunk_with_an_overlong_match_is_refused() {
        // Upstream's own reproduction (tests/test_blosclz_bounds.c): a Blosc2
        // chunk claiming one 64 KiB BloscLZ block whose stream is a match
        // longer than the block.
        let (blocksize, cbytes) = (65_536_i32, 100_000_usize);
        let mut chunk = vec![0u8; cbytes];
        chunk[0] = 2; // format version
        chunk[1] = 4; // Blosc2 format version
        chunk[3] = 1; // typesize
        chunk[4..8].copy_from_slice(&blocksize.to_le_bytes()); // nbytes
        chunk[8..12].copy_from_slice(&blocksize.to_le_bytes()); // blocksize
        chunk[12..16].copy_from_slice(&(cbytes as i32).to_le_bytes()); // cbytes
        let header = 32; // BLOSC_EXTENDED_HEADER_LENGTH; codec 0 is BloscLZ
        chunk[header..header + 3].copy_from_slice(&[0, b'A', 0xE0]);
        chunk[header + 3..cbytes - 2].fill(0xFF);
        chunk[cbytes - 2..].copy_from_slice(&[0x01, 0x00]);
        let mut out = vec![0u8; blocksize as usize];
        // SAFETY: both buffers are valid for the lengths given.
        let n = unsafe {
            start_blosc2();
            blosc2_decompress(chunk.as_ptr().cast(), cbytes as i32, out.as_mut_ptr().cast(), blocksize)
        };
        assert!(n < 0, "the chunk must be refused, not decompressed to {n} bytes");
    }

    #[test]
    fn valid_blosclz_chunks_still_round_trip() {
        let values: Vec<u8> = (0..1_048_576_u32).map(|i| ((i / 7) % 251) as u8).collect();
        let mut chunk = vec![0u8; values.len() + 32];
        let mut back = vec![0u8; values.len()];
        // SAFETY: every buffer is valid for the length given; the global
        // compressor is set to the one this test is about.
        let (written, read) = unsafe {
            start_blosc2();
            assert!(blosc1_set_compressor(c"blosclz".as_ptr()) >= 0);
            let written = blosc2_compress(
                5,
                1,
                1,
                values.as_ptr().cast(),
                values.len() as i32,
                chunk.as_mut_ptr().cast(),
                chunk.len() as i32,
            );
            assert!(written > 0, "compression failed: {written}");
            let read = blosc2_decompress(chunk.as_ptr().cast(), written, back.as_mut_ptr().cast(), back.len() as i32);
            (written, read)
        };
        assert!((written as usize) < values.len(), "BloscLZ compressed nothing");
        assert_eq!(read as usize, values.len());
        assert_eq!(back, values);
    }

    /// Fill the stack the next call will use with a byte that is not zero,
    /// so that what a callee leaves uninitialised is not zero by luck.
    #[inline(never)]
    fn dirty_stack() {
        let mut junk = [0xA5u8; 64 * 1024];
        std::hint::black_box(&mut junk);
    }

    /// [`run_blosc2_filter`] on a stack left dirty.
    fn blosc2_filter(reverse: bool, values: &[c_uint], data: &[u8], buf_size: usize) -> Option<Vec<u8>> {
        let _held = HDF5_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        dirty_stack();
        run_blosc2_filter(reverse, values, data, buf_size)
    }

    /// Filter values: revision, block size, type size, chunk bytes, level,
    /// shuffle, BloscLZ, and for B2ND the chunk's rank and shape.
    fn filter_values(chunk: usize, shape: Option<[c_uint; 2]>) -> Vec<c_uint> {
        let mut values = vec![2, 0, 1, chunk as c_uint, 5, 1, 0];
        if let Some(shape) = shape {
            values.push(2);
            values.extend(shape);
        }
        values
    }

    /// A chunk that decompresses to more than its dataset's chunk is refused
    /// before anything is allocated for it --- 4 MiB of zeros compress to a
    /// few hundred bytes, and both a chunk header and the filter values can
    /// state up to gigabytes --- and a chunk of the dataset still decodes, in
    /// plain Blosc2 (which trusted the header's size) and in B2ND (which
    /// trusted the filter values).
    #[test]
    fn s14_2_a_blosc2_chunk_larger_than_its_dataset_chunk_is_refused() {
        let zeros = vec![0u8; 4 << 20];
        for shape in [None, Some([1024, 4096])] {
            let values = filter_values(zeros.len(), shape);
            let frame = blosc2_filter(false, &values, &zeros, zeros.len()).expect("compressed");
            assert!(frame.len() < 64 * 1024, "{} bytes", frame.len());
            let small = filter_values(4096, shape.map(|_| [64, 64]));
            assert_eq!(blosc2_filter(true, &small, &frame, 4096), None, "shape: {shape:?}");
            // The filter values of the dataset the chunk belongs to, with a
            // smaller buffer than the chunk: B2ND checks them against it.
            if shape.is_some() {
                assert_eq!(blosc2_filter(true, &values, &frame, 4096), None);
            }
            assert_eq!(blosc2_filter(true, &values, &frame, zeros.len()).as_deref(), Some(&zeros[..]));
        }
    }

    /// A super-chunk of no chunks --- the frame C-Blosc2 itself writes for
    /// one --- is refused.  Asked for its chunk 0, C-Blosc2 left the filter's
    /// chunk pointer unset, and the filter then freed whatever the stack held
    /// there.
    #[test]
    fn s14_2_a_blosc2_frame_of_no_chunks_is_refused() {
        // SAFETY: a super-chunk of default parameters in memory; its frame is
        // copied out before it is freed.
        let frame = unsafe {
            start_blosc2();
            use std::ptr::null_mut;
            let mut storage = Blosc2Storage {
                contiguous: true,
                urlpath: null_mut(),
                cparams: null_mut(),
                dparams: null_mut(),
                io: null_mut(),
            };
            let schunk = blosc2_schunk_new(&mut storage);
            assert!(!schunk.is_null());
            let (mut cframe, mut needs_free) = (std::ptr::null_mut(), false);
            let len = blosc2_schunk_to_buffer(schunk, &mut cframe, &mut needs_free);
            assert!(len > 0 && !needs_free);
            let frame = std::slice::from_raw_parts(cframe, len as usize).to_vec();
            blosc2_schunk_free(schunk);
            frame
        };
        assert_eq!(blosc2_filter(true, &filter_values(4096, None), &frame, 4096), None);
    }

    /// A damaged chunk leaves C-Blosc2 as it was.  The filter's failure path
    /// called `blosc2_destroy()`, which frees the library's global context
    /// and destroys the mutexes and thread pools a window read on another
    /// thread may be using; the next `blosc2_init()` then started the library
    /// afresh, forgetting the codecs registered with it among the rest.
    #[test]
    fn s14_2_a_damaged_blosc2_chunk_leaves_the_library_as_it_was() {
        let name = c"medh5-test-codec";
        let mut codec = Blosc2Codec {
            compcode: 250,
            compname: name.as_ptr(),
            complib: 250,
            version: 1,
            encoder: None,
            decoder: None,
        };
        // SAFETY: the codec is copied into C-Blosc2's registry, and its name
        // is a static string; nothing compresses with it.
        unsafe {
            start_blosc2();
            assert!(blosc2_register_codec(&mut codec) >= 0);
            assert_eq!(blosc2_compname_to_compcode(name.as_ptr()), 250);
        }
        // Not a frame at all.
        assert_eq!(blosc2_filter(true, &filter_values(4096, None), &[0x5A; 256], 4096), None);
        // SAFETY: as above.
        unsafe {
            blosc2_init();
            assert_eq!(blosc2_compname_to_compcode(name.as_ptr()), 250);
        }
    }
}
