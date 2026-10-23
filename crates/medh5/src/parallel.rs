//! How many threads one call may use for its work outside HDF5.
//!
//! HDF5 runs one call at a time, whatever the number of threads that ask, so
//! the threads a call starts do the work that needs no HDF5: compressing a
//! chunk, hashing bytes already read.  The setting is one count for the
//! process.  1, the default, runs every call on its caller's thread, as
//! before there was a setting, and starts no thread.
//!
//! The count is, in order: what [`set_threads`] set; `MEDH5_THREADS` in the
//! environment, read once, at the first call that asks; 1.  The command line's
//! `--threads N` sets it for the one command.  A value that is not a positive
//! integer is refused, never guessed: there is no automatic count.
//!
//! A call that uses threads starts them inside the call and joins them before
//! it returns, so no thread outlives a call and there is no pool.  Only the
//! calling thread talks to HDF5: workers are given bytes and give bytes back,
//! and the caller makes every HDF5 call in a fixed order, so a file's layout
//! does not depend on which worker finished first.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Condvar, Mutex, OnceLock};

use crate::{Error, Result};

/// The environment variable the count is read from.
const VARIABLE: &str = "MEDH5_THREADS";

/// What [`set_threads`] set; 0 while nothing has.
static SET: AtomicUsize = AtomicUsize::new(0);

/// `MEDH5_THREADS`, as read at the first call that asked: `None` when unset.
static ENVIRONMENT: OnceLock<Result<Option<usize>, String>> = OnceLock::new();

/// Let one call use up to `n` threads for its work outside HDF5, from now on
/// and in every thread of the process.  `0` is refused.
pub fn set_threads(n: usize) -> Result<()> {
    if n == 0 {
        return Err(Error::Value("the thread count must be at least 1, not 0".into()));
    }
    SET.store(n, Ordering::SeqCst);
    Ok(())
}

/// How many threads one call may use for its work outside HDF5: what
/// [`set_threads`] set, else `MEDH5_THREADS`, else 1.  `Err` when it falls to
/// `MEDH5_THREADS` and that is not a positive integer.
pub fn threads() -> Result<usize> {
    match SET.load(Ordering::SeqCst) {
        0 => match ENVIRONMENT.get_or_init(|| from_environment(std::env::var_os(VARIABLE))) {
            Ok(n) => Ok(n.unwrap_or(1)),
            Err(why) => Err(Error::Value(why.clone())),
        },
        n => Ok(n),
    }
}

/// The count for one command line run, `--threads N`: set until the guard is
/// dropped, and then what was set before it.  Not API: the command line's.
#[doc(hidden)]
pub fn for_this_run(n: usize) -> Result<Restore> {
    if n == 0 {
        return Err(Error::Value("the thread count must be at least 1, not 0".into()));
    }
    Ok(Restore(SET.swap(n, Ordering::SeqCst)))
}

/// What [`for_this_run`] gives back when it is dropped.
#[doc(hidden)]
pub struct Restore(usize);

impl Drop for Restore {
    fn drop(&mut self) {
        SET.store(self.0, Ordering::SeqCst);
    }
}

/// `MEDH5_THREADS`'s value as a count, or why it is not one.
fn from_environment(value: Option<std::ffi::OsString>) -> Result<Option<usize>, String> {
    let Some(value) = value else { return Ok(None) };
    match value.to_str().map(str::trim).and_then(|v| v.parse::<usize>().ok()) {
        Some(n) if n > 0 => Ok(Some(n)),
        _ => Err(format!(
            "{VARIABLE} must be a positive integer, not {}",
            crate::json::repr_str(&value.to_string_lossy())
        )),
    }
}

/// The parallel sections: a call holds a shared hold from before its first
/// worker starts until its last has joined, and a `fork` takes it exclusively
/// (`h5::fork`), so a child never inherits an allocator's or C-Blosc2's lock
/// held by a worker it does not have.
pub(crate) static SECTIONS: parking_lot::RwLock<()> = parking_lot::RwLock::new(());

thread_local! {
    /// Whether this thread is in a parallel section already: an inner one
    /// runs under the outer one's hold.
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

/// Workers started, for the test that one thread starts none.
#[cfg(test)]
static STARTED: AtomicUsize = AtomicUsize::new(0);

/// `work` on each of `items`, on up to `threads` workers, and each result
/// handed to `sink` on the calling thread, in the order of `items`.
///
/// With one thread, or inside another parallel section, everything runs here,
/// one item after another, and no thread is started and no hold is taken.
/// Otherwise at most twice `threads` items are in flight, so memory is
/// bounded however many there are.  `work` must make no HDF5 call; `sink`
/// is where they go.  The first `Err` from `sink` stops the workers and is
/// returned.  The caller must not hold HDF5's lock: a `fork` waiting for the
/// parallel sections waits for this call, which may wait for HDF5's lock.
#[cfg_attr(not(test), allow(dead_code))] // no path uses threads yet
pub(crate) fn ordered<T, R, E>(
    threads: usize,
    items: impl Iterator<Item = T> + Send,
    work: impl Fn(T) -> R + Sync,
    mut sink: impl FnMut(R) -> Result<(), E>,
) -> Result<(), E>
where
    T: Send,
    R: Send,
{
    if threads <= 1 || INSIDE.get() {
        return items.map(work).try_for_each(sink);
    }
    debug_assert!(!crate::h5sys::LOCK.is_owned_by_current_thread(), "a parallel section entered holding HDF5's lock");
    let _hold = SECTIONS.read();
    INSIDE.set(true);
    let _inside = Inside;
    let window = 2 * threads;
    let source = Mutex::new(items.enumerate());
    // Items a worker may still take before the caller has sunk more, and
    // whether to stop.
    let flow = (Mutex::new((window, false)), Condvar::new());
    let (tx, rx) = mpsc::channel::<(usize, R)>();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (tx, source, flow, work) = (tx.clone(), &source, &flow, &work);
            #[cfg(test)]
            STARTED.fetch_add(1, Ordering::SeqCst);
            scope.spawn(move || {
                // A worker that panics stops the others, which would
                // otherwise wait for the caller to sink the item it lost.
                let _stop = Stop(flow);
                // A section inside the work runs on this worker.
                INSIDE.set(true);
                loop {
                    {
                        let (state, ready) = flow;
                        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
                        while state.0 == 0 && !state.1 {
                            state = ready.wait(state).unwrap_or_else(|e| e.into_inner());
                        }
                        if state.1 {
                            return;
                        }
                        state.0 -= 1;
                    }
                    let Some((index, item)) = source.lock().unwrap_or_else(|e| e.into_inner()).next() else {
                        return;
                    };
                    if tx.send((index, work(item))).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        let mut waiting = BTreeMap::new();
        let mut next = 0;
        let mut outcome = Ok(());
        for (index, result) in rx.iter() {
            waiting.insert(index, result);
            while let Some(result) = waiting.remove(&next) {
                next += 1;
                if let Err(e) = sink(result) {
                    outcome = Err(e);
                    break;
                }
                let (state, ready) = &flow;
                state.lock().unwrap_or_else(|e| e.into_inner()).0 += 1;
                ready.notify_one();
            }
            if outcome.is_err() {
                break;
            }
        }
        Stop(&flow).stop();
        outcome
    })
}

/// Marks the end of this thread's parallel section.
struct Inside;

impl Drop for Inside {
    fn drop(&mut self) {
        INSIDE.set(false);
    }
}

/// Stops every worker of a call when dropped.
struct Stop<'a>(&'a (Mutex<(usize, bool)>, Condvar));

impl Stop<'_> {
    fn stop(&self) {
        let (state, ready) = self.0;
        state.lock().unwrap_or_else(|e| e.into_inner()).1 = true;
        ready.notify_all();
    }
}

impl Drop for Stop<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.stop();
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::time::Duration;

    /// The tests that set the count, count workers or hold the sections,
    /// here and in `h5::fork`, one at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    pub(crate) fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn the_count_is_what_was_set_and_zero_is_refused() {
        let _serial = serial();
        let before = SET.load(Ordering::SeqCst);
        set_threads(3).unwrap();
        assert_eq!(threads().unwrap(), 3);
        assert!(matches!(set_threads(0), Err(Error::Value(_))));
        assert_eq!(threads().unwrap(), 3, "a refused count changed the setting");
        SET.store(before, Ordering::SeqCst);
    }

    #[test]
    fn a_run_sets_the_count_until_it_ends() {
        let _serial = serial();
        let before = SET.load(Ordering::SeqCst);
        set_threads(2).unwrap();
        {
            let _run = for_this_run(5).unwrap();
            assert_eq!(threads().unwrap(), 5);
        }
        assert_eq!(threads().unwrap(), 2);
        assert!(for_this_run(0).is_err());
        SET.store(before, Ordering::SeqCst);
    }

    #[test]
    fn the_environment_is_a_positive_integer_or_refused() {
        let os = |s: &str| Some(std::ffi::OsString::from(s));
        assert_eq!(from_environment(None), Ok(None));
        assert_eq!(from_environment(os("4")), Ok(Some(4)));
        assert_eq!(from_environment(os(" 2 ")), Ok(Some(2)));
        for bad in ["0", "-1", "four", "", "1.5", "auto"] {
            let why = from_environment(os(bad)).unwrap_err();
            assert!(why.starts_with("MEDH5_THREADS must be a positive integer"), "{bad}: {why}");
        }
    }

    #[test]
    fn results_reach_the_sink_in_order_at_every_count() {
        let _serial = serial();
        for threads in [1, 2, 4] {
            let mut seen = Vec::new();
            ordered(
                threads,
                0..200u64,
                |i| {
                    // Later items finish first, now and then.
                    std::thread::sleep(Duration::from_micros((200 - i) % 7 * 50));
                    i * i
                },
                |r| {
                    seen.push(r);
                    Ok::<_, ()>(())
                },
            )
            .unwrap();
            assert_eq!(seen, (0..200u64).map(|i| i * i).collect::<Vec<_>>(), "{threads} threads");
        }
    }

    #[test]
    fn one_thread_starts_no_thread_and_takes_no_hold() {
        let _serial = serial();
        let before = STARTED.load(Ordering::SeqCst);
        let caller = std::thread::current().id();
        ordered(
            1,
            0..10,
            |i| {
                assert_eq!(std::thread::current().id(), caller);
                // Another thread finds the sections free: a fork would not wait.
                assert!(std::thread::spawn(|| SECTIONS.try_write().is_some()).join().unwrap());
                i
            },
            |_| Ok::<_, ()>(()),
        )
        .unwrap();
        assert_eq!(STARTED.load(Ordering::SeqCst), before, "one thread started a worker");
        ordered(4, 0..10, |i| i, |_| Ok::<_, ()>(())).unwrap();
        assert_eq!(STARTED.load(Ordering::SeqCst), before + 4);
    }

    #[test]
    fn a_section_inside_a_section_runs_on_the_caller() {
        let _serial = serial();
        let before = STARTED.load(Ordering::SeqCst);
        ordered(2, 0..3, |i| i, |_| ordered(4, 0..3, |i| i, |_| Ok::<_, ()>(()))).unwrap();
        assert_eq!(STARTED.load(Ordering::SeqCst), before + 2);
    }

    #[test]
    fn a_failing_sink_stops_the_workers() {
        let _serial = serial();
        let done = AtomicUsize::new(0);
        let mut sunk = 0;
        let outcome = ordered(
            4,
            0..10_000,
            |i| {
                done.fetch_add(1, Ordering::SeqCst);
                i
            },
            |i| {
                sunk += 1;
                if i == 10 {
                    Err("stop")
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(outcome, Err("stop"));
        assert_eq!(sunk, 11);
        // The window bounds what ran past the failure.
        assert!(done.load(Ordering::SeqCst) <= 11 + 2 * 4, "{} items ran", done.load(Ordering::SeqCst));
    }

    #[test]
    fn a_worker_that_panics_ends_the_call_with_its_panic() {
        let _serial = serial();
        let caught = std::panic::catch_unwind(|| {
            ordered(
                4,
                0..100,
                |i| {
                    assert_ne!(i, 7, "the seventh item");
                    i
                },
                |_| Ok::<_, ()>(()),
            )
        });
        assert!(caught.is_err());
        assert!(!INSIDE.get(), "the section outlived the call");
        assert!(SECTIONS.try_write().is_some(), "the hold outlived the call");
    }
}
