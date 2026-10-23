//! Where a command's time goes, in a build made to say so.
//!
//! Compiled only with `--cfg medh5_phase_timings` in `RUSTFLAGS`, which no
//! profile sets: it is not a Cargo feature and adds nothing to the API.  In
//! every other build `phase!` and `stage!` expand to nothing.
//!
//! - A **phase** is exclusive: the time a thread spends in it, less the time
//!   in phases entered inside it.  So the Blosc2 filter, which HDF5 runs inside
//!   a read or a write, is counted as decompression or compression and not
//!   again as the read or the write.  Each phase is either under HDF5's lock
//!   (an HDF5 call, and the filters inside it) or outside it.
//! - A **stage** is inclusive: the whole of a named step of a command
//!   (`commit.digests`), phases and all.
//!
//! The process reports at exit: a table on standard error, or, where
//! `MEDH5_PHASE_TIMINGS` names a file, one JSON object appended to it.
//! What no phase claims is reported as `other`: Rust work outside every
//! phase, and the HDF5 calls made outside the wrappers here (small metadata
//! calls through the `hdf5` crate).

/// Count the rest of the enclosing block as phase `$phase` (a
/// `timings::Phase` variant), with `$bytes` bytes, in a timing build.
macro_rules! phase {
    ($phase:ident) => {
        #[cfg(medh5_phase_timings)]
        let _phase = $crate::timings::Span::enter($crate::timings::Phase::$phase, 0);
    };
    ($phase:ident, $bytes:expr) => {
        #[cfg(medh5_phase_timings)]
        let _phase = $crate::timings::Span::enter($crate::timings::Phase::$phase, ($bytes) as u64);
    };
}

/// Count the rest of the enclosing block as stage `$name`, in a timing build.
macro_rules! stage {
    ($name:literal) => {
        #[cfg(medh5_phase_timings)]
        let _stage = $crate::timings::Stage::enter($name);
    };
}

#[cfg(medh5_phase_timings)]
pub(crate) use imp::*;

#[cfg(medh5_phase_timings)]
mod imp {
    use std::cell::RefCell;
    use std::fmt::Write as _;
    use std::os::raw::{c_int, c_uint, c_void};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, Once, OnceLock};
    use std::time::Instant;

    use indexmap::IndexMap;
    use medh5_sys::hdf5_sys::h5::herr_t;
    use medh5_sys::hdf5_sys::h5i::hid_t;

    /// What a thread is doing.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Phase {
        // Under HDF5's lock.
        /// Opening, creating, flushing and closing files.
        Open,
        /// Attributes, links, groups, and making datasets.
        Metadata,
        /// Copying objects between files (an amendment's inherited objects).
        Copy,
        /// Reading a dataset, less the Blosc2 filter.
        Read,
        /// Writing a dataset, less the Blosc2 filter.
        Write,
        /// The Blosc2 filter decompressing, inside a read.
        Decompress,
        /// The Blosc2 filter compressing, inside a write.
        Compress,
        // Outside it.
        /// Blosc2 blocks the window reader decompresses itself.
        Window,
        /// Target voxels to source coordinates (`resample`).
        Map,
        /// Sampling the source at them, and converting the values.
        Interpolate,
        /// A carried annotation's own work (`resample`).
        Carry,
        /// The digest's hash.
        Hash,
        /// A NIfTI file's stream: reading and gunzipping it.
        Inflate,
        /// A NIfTI file's bytes to values.
        Decode,
    }

    const PHASES: [Phase; 14] = [
        Phase::Open,
        Phase::Metadata,
        Phase::Copy,
        Phase::Read,
        Phase::Write,
        Phase::Decompress,
        Phase::Compress,
        Phase::Window,
        Phase::Map,
        Phase::Interpolate,
        Phase::Carry,
        Phase::Hash,
        Phase::Inflate,
        Phase::Decode,
    ];

    impl Phase {
        fn name(self) -> &'static str {
            match self {
                Phase::Open => "open",
                Phase::Metadata => "metadata",
                Phase::Copy => "copy",
                Phase::Read => "read",
                Phase::Write => "write",
                Phase::Decompress => "decompress",
                Phase::Compress => "compress",
                Phase::Window => "window",
                Phase::Map => "map",
                Phase::Interpolate => "interpolate",
                Phase::Carry => "carry",
                Phase::Hash => "hash",
                Phase::Inflate => "inflate",
                Phase::Decode => "decode",
            }
        }

        /// Whether the phase runs inside an HDF5 call.
        fn locked(self) -> bool {
            matches!(
                self,
                Phase::Open
                    | Phase::Metadata
                    | Phase::Copy
                    | Phase::Read
                    | Phase::Write
                    | Phase::Decompress
                    | Phase::Compress
            )
        }
    }

    /// Nanoseconds, entries and bytes of each phase, over every thread.
    static NANOS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
    static CALLS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];
    static BYTES: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

    /// One stage's totals: nanoseconds, entries, and the nanoseconds of each
    /// phase while it ran.
    #[derive(Clone, Default)]
    struct Totals {
        nanos: u64,
        calls: u64,
        phases: [u64; 14],
    }

    /// Each stage's totals, in the order first entered.
    static STAGES: Mutex<Option<IndexMap<&'static str, Totals>>> = Mutex::new(None);

    /// When the first phase or stage was entered.
    static START: OnceLock<Instant> = OnceLock::new();

    thread_local! {
        /// The phases this thread is inside, innermost last, each with the
        /// moment it last resumed.
        static STACK: RefCell<Vec<(Phase, Instant)>> = const { RefCell::new(Vec::new()) };
    }

    fn add(phase: Phase, since: Instant, now: Instant) {
        NANOS[phase as usize].fetch_add(now.duration_since(since).as_nanos() as u64, Ordering::Relaxed);
    }

    fn started() {
        static REPORT: Once = Once::new();
        START.get_or_init(Instant::now);
        REPORT.call_once(|| {
            unsafe extern "C" {
                fn atexit(callback: extern "C" fn()) -> c_int;
            }
            // SAFETY: registers a function of no arguments the C runtime
            // calls once, at exit.
            unsafe { atexit(report_at_exit) };
        });
    }

    /// One phase entered, until dropped.
    pub(crate) struct Span(Phase);

    impl Span {
        pub(crate) fn enter(phase: Phase, bytes: u64) -> Span {
            started();
            let now = Instant::now();
            STACK.with(|stack| {
                let mut stack = stack.borrow_mut();
                if let Some((outer, since)) = stack.last_mut() {
                    add(*outer, *since, now);
                    *since = now;
                }
                stack.push((phase, now));
            });
            CALLS[phase as usize].fetch_add(1, Ordering::Relaxed);
            BYTES[phase as usize].fetch_add(bytes, Ordering::Relaxed);
            Span(phase)
        }

        pub(crate) fn add_bytes(&self, bytes: u64) {
            BYTES[self.0 as usize].fetch_add(bytes, Ordering::Relaxed);
        }
    }

    impl Drop for Span {
        fn drop(&mut self) {
            let now = Instant::now();
            STACK.with(|stack| {
                let mut stack = stack.borrow_mut();
                if let Some((phase, since)) = stack.pop() {
                    add(phase, since, now);
                }
                if let Some((_, since)) = stack.last_mut() {
                    *since = now;
                }
            });
        }
    }

    /// Every phase's nanoseconds so far.
    fn phase_nanos() -> [u64; 14] {
        std::array::from_fn(|i| NANOS[i].load(Ordering::Relaxed))
    }

    /// One stage entered, until dropped, with the phases' counts as it began.
    /// A stage begins and ends outside every phase, where those counts are
    /// whole; the phases of other threads meanwhile count towards it too.
    pub(crate) struct Stage(&'static str, Instant, [u64; 14]);

    impl Stage {
        pub(crate) fn enter(name: &'static str) -> Stage {
            started();
            Stage(name, Instant::now(), phase_nanos())
        }
    }

    impl Drop for Stage {
        fn drop(&mut self) {
            let nanos = self.1.elapsed().as_nanos() as u64;
            let now = phase_nanos();
            let mut stages = STAGES.lock().unwrap_or_else(|e| e.into_inner());
            let entry = stages.get_or_insert_with(IndexMap::new).entry(self.0).or_default();
            entry.nanos += nanos;
            entry.calls += 1;
            for (i, phase) in entry.phases.iter_mut().enumerate() {
                *phase += now[i].saturating_sub(self.2[i]);
            }
        }
    }

    unsafe extern "C" {
        fn blosc2_filter_function(
            flags: c_uint,
            cd_nelmts: usize,
            cd_values: *const c_uint,
            nbytes: usize,
            buf_size: *mut usize,
            buf: *mut *mut c_void,
        ) -> usize;
        fn blosc2_set_local(dcpl: hid_t, dtype: hid_t, space: hid_t) -> herr_t;
    }

    /// The vendored filter's `set_local`, as HDF5's class takes it.
    extern "C" fn set_local(dcpl: hid_t, dtype: hid_t, space: hid_t) -> herr_t {
        // SAFETY: HDF5 calls this, under its lock, as it would the filter's.
        unsafe { blosc2_set_local(dcpl, dtype, space) }
    }

    /// The vendored filter, timed as decompression or compression.
    unsafe extern "C" fn timed_blosc2(
        flags: c_uint,
        cd_nelmts: usize,
        cd_values: *const c_uint,
        nbytes: usize,
        buf_size: *mut usize,
        buf: *mut *mut c_void,
    ) -> usize {
        let reverse = flags & medh5_sys::hdf5_sys::h5z::H5Z_FLAG_REVERSE != 0;
        let span =
            if reverse { Span::enter(Phase::Decompress, 0) } else { Span::enter(Phase::Compress, nbytes as u64) };
        // SAFETY: HDF5's arguments, passed on as it passes them.
        let n = unsafe { blosc2_filter_function(flags, cd_nelmts, cd_values, nbytes, buf_size, buf) };
        if reverse {
            span.add_bytes(n as u64);
        }
        n
    }

    /// Register the Blosc2 filter again, timed.  HDF5 replaces a class
    /// registered under the same id.  The caller holds HDF5's lock.
    pub(crate) fn time_filters() {
        use medh5_sys::hdf5_sys::h5z;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let class = h5z::H5Z_class2_t {
                version: h5z::H5Z_CLASS_T_VERS as c_int,
                id: medh5_sys::BLOSC2_FILTER_ID as h5z::H5Z_filter_t,
                encoder_present: 1,
                decoder_present: 1,
                name: c"blosc2".as_ptr(),
                can_apply: None,
                set_local: Some(set_local),
                filter: Some(timed_blosc2),
            };
            // SAFETY: HDF5 copies the class; the caller holds its lock.
            unsafe { h5z::H5Zregister((&class as *const h5z::H5Z_class2_t).cast()) };
        });
    }

    /// What was counted, as one JSON object.
    fn json(wall: f64) -> serde_json::Value {
        let mut phases = serde_json::Map::new();
        let mut named = 0.0;
        for phase in PHASES {
            let i = phase as usize;
            let seconds = NANOS[i].load(Ordering::Relaxed) as f64 / 1e9;
            named += seconds;
            phases.insert(
                phase.name().into(),
                serde_json::json!({
                    "seconds": seconds,
                    "calls": CALLS[i].load(Ordering::Relaxed),
                    "bytes": BYTES[i].load(Ordering::Relaxed),
                    "locked": phase.locked(),
                }),
            );
        }
        let stages = STAGES.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default();
        let stages: serde_json::Map<String, serde_json::Value> = stages
            .into_iter()
            .map(|(name, totals)| {
                let phases: serde_json::Map<String, serde_json::Value> = PHASES
                    .iter()
                    .map(|p| (p.name().into(), serde_json::json!(totals.phases[*p as usize] as f64 / 1e9)))
                    .collect();
                let seconds = totals.nanos as f64 / 1e9;
                (name.into(), serde_json::json!({"seconds": seconds, "calls": totals.calls, "phases": phases}))
            })
            .collect();
        serde_json::json!({
            "wall": wall,
            "other": (wall - named).max(0.0),
            "phases": phases,
            "stages": stages,
        })
    }

    /// What was counted, as a table.
    fn table(wall: f64) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "medh5 phase timings: {wall:.3} s from the first phase");
        let mut named = 0.0;
        for phase in PHASES {
            let i = phase as usize;
            let seconds = NANOS[i].load(Ordering::Relaxed) as f64 / 1e9;
            named += seconds;
            if CALLS[i].load(Ordering::Relaxed) == 0 {
                continue;
            }
            let bytes = BYTES[i].load(Ordering::Relaxed);
            let rate = if bytes > 0 && seconds > 0.0 {
                format!("{:>9.1} MiB/s", bytes as f64 / seconds / (1 << 20) as f64)
            } else {
                String::new()
            };
            let _ = writeln!(
                out,
                "  {:<12} {:<7} {seconds:>9.3} s {:>9} calls{rate}",
                phase.name(),
                if phase.locked() { "locked" } else { "outside" },
                CALLS[i].load(Ordering::Relaxed),
            );
        }
        let _ = writeln!(out, "  {:<12} {:<7} {:>9.3} s", "other", "", (wall - named).max(0.0));
        let stages = STAGES.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default();
        for (name, totals) in stages {
            let seconds = totals.nanos as f64 / 1e9;
            let _ = writeln!(out, "  stage {name:<26} {seconds:>9.3} s {:>9} calls", totals.calls);
        }
        out
    }

    extern "C" fn report_at_exit() {
        let wall = START.get().map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
        match std::env::var_os("MEDH5_PHASE_TIMINGS").filter(|v| !v.is_empty()) {
            Some(path) => {
                use std::io::Write as _;
                let line = format!("{}\n", json(wall));
                let file = std::fs::OpenOptions::new().create(true).append(true).open(&path);
                if let Err(e) = file.and_then(|mut f| f.write_all(line.as_bytes())) {
                    eprintln!("medh5 phase timings: could not write {}: {e}", path.to_string_lossy());
                }
            }
            None => eprint!("{}", table(wall)),
        }
    }

    // Run alone (`timings` as the filter): the counts are the process's, and
    // another test's reads would add to them.
    #[cfg(test)]
    mod tests {
        use super::*;
        use std::time::Duration;

        fn seconds(phase: Phase) -> f64 {
            NANOS[phase as usize].load(Ordering::Relaxed) as f64 / 1e9
        }

        /// A phase inside another counts once: the outer one stops while the
        /// inner one runs, and goes on after it.
        #[test]
        fn timings_a_nested_phase_is_not_counted_twice() {
            let (outer, inner) = (seconds(Phase::Inflate), seconds(Phase::Decode));
            {
                let _outer = Span::enter(Phase::Inflate, 0);
                std::thread::sleep(Duration::from_millis(20));
                {
                    let _inner = Span::enter(Phase::Decode, 7);
                    std::thread::sleep(Duration::from_millis(200));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            let (outer, inner) = (seconds(Phase::Inflate) - outer, seconds(Phase::Decode) - inner);
            // Counted twice, the outer phase would hold the inner one's 0.2 s too.
            assert!((0.04..0.2).contains(&outer), "outer {outer}");
            assert!(inner >= 0.2, "inner {inner}");
            assert!(BYTES[Phase::Decode as usize].load(Ordering::Relaxed) >= 7);
        }

        /// The Blosc2 filter is timed inside the write and the read that run
        /// it, and what it stores reads back.
        #[test]
        fn timings_the_blosc2_filter_is_counted_as_compression() {
            use crate::array::NdArray;
            use crate::h5::data::{self, Layout};
            use crate::storage::codecs::{resolve_profile, Role};
            crate::h5::init();
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("t.h5");
            let file = crate::h5::file::create_truncate(&path).unwrap();
            let values = NdArray::from_vec(&[32, 64], (0..2048).map(|i| (i % 37) as f64).collect()).unwrap();
            let layout = Layout {
                chunks: Some(vec![16, 32]),
                filters: resolve_profile(Some("training")).unwrap().codec(Role::Image).filters(),
            };
            let before = |phase: Phase| BYTES[phase as usize].load(Ordering::Relaxed);
            let (compressed, decompressed) = (before(Phase::Compress), before(Phase::Decompress));
            // HDF5 compresses a chunk as it leaves the chunk cache, so as
            // the file closes, and reads it back from a file opened anew.
            drop(data::create(&file, "d", &values, &layout).unwrap());
            drop(file);
            assert_eq!(before(Phase::Compress) - compressed, 2048 * 8);
            let file = crate::h5::file::open_read(&path).unwrap();
            assert_eq!(data::read(&file.dataset("d").unwrap()).unwrap(), values);
            assert_eq!(before(Phase::Decompress) - decompressed, 2048 * 8);
        }

        /// A stage holds the phases that ran inside it.
        #[test]
        fn timings_a_stage_counts_the_phases_inside_it() {
            {
                let _stage = Stage::enter("timings.test");
                let _phase = Span::enter(Phase::Inflate, 0);
                std::thread::sleep(Duration::from_millis(30));
            }
            let report = json(1.0);
            let stage = &report["stages"]["timings.test"];
            assert!(stage["seconds"].as_f64().unwrap() >= 0.03);
            assert!(stage["phases"]["inflate"].as_f64().unwrap() >= 0.03);
        }

        /// The report names every phase, and what none of them claims.
        #[test]
        fn timings_the_report_names_every_phase() {
            let report = json(1.0);
            for phase in PHASES {
                assert_eq!(report["phases"][phase.name()]["locked"], phase.locked(), "{phase:?}");
            }
            assert!(report["other"].as_f64().unwrap() >= 0.0);
            assert!(table(1.0).starts_with("medh5 phase timings"));
        }
    }
}
