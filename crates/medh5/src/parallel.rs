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

/// The count of the newest command line run still going, over [`SET`]; 0
/// while none is.  Written only with [`RUNS`] held.
static RUN: AtomicUsize = AtomicUsize::new(0);

/// Each command line run still going, by the number its guard holds, and
/// its count, oldest first.  A `fork` holds it (`h5::fork`): the command
/// line runs detached from Python, so a thread of Python's could fork while
/// another holds it.
pub(crate) static RUNS: Mutex<(u64, Vec<(u64, usize)>)> = Mutex::new((0, Vec::new()));

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
    if let n @ 1.. = RUN.load(Ordering::SeqCst) {
        return Ok(n);
    }
    match SET.load(Ordering::SeqCst) {
        0 => match ENVIRONMENT.get_or_init(|| from_environment(std::env::var_os(VARIABLE))) {
            Ok(n) => Ok(n.unwrap_or(1)),
            Err(why) => Err(Error::Value(why.clone())),
        },
        n => Ok(n),
    }
}

/// The count for one command line run, `--threads N`, over what
/// [`set_threads`] set, until the guard is dropped.  Runs that overlap (the
/// command line called from two threads) each hold their own: the newest
/// still going is the count, and a [`set_threads`] made during a run takes
/// effect when the last ends.  Not API: the command line's.
#[doc(hidden)]
pub fn for_this_run(n: usize) -> Result<Restore> {
    if n == 0 {
        return Err(Error::Value("the thread count must be at least 1, not 0".into()));
    }
    let mut runs = RUNS.lock().unwrap_or_else(|e| e.into_inner());
    runs.0 += 1;
    let id = runs.0;
    runs.1.push((id, n));
    RUN.store(n, Ordering::SeqCst);
    Ok(Restore(id))
}

/// What [`for_this_run`] gives back: dropped, its run ends.
#[doc(hidden)]
pub struct Restore(u64);

impl Drop for Restore {
    fn drop(&mut self) {
        let mut runs = RUNS.lock().unwrap_or_else(|e| e.into_inner());
        runs.1.retain(|&(id, _)| id != self.0);
        RUN.store(runs.1.last().map_or(0, |&(_, n)| n), Ordering::SeqCst);
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
pub(crate) static SECTIONS: Sections = Sections(AtomicUsize::new(0));

/// A lock no thread parks on: one count, its top bit the exclusive hold and
/// the rest the shared ones, and a thread that finds it taken waits by
/// yielding, then sleeping.  parking_lot's `RwLock` queues the threads that
/// wait for it in a table its own locks guard, and a thread queueing for the
/// sections while a `fork` holds them could hold one of those at the fork:
/// the child, letting the sections go, would wait for it forever.  This one
/// is let go with one atomic write, on either side of the fork.
pub(crate) struct Sections(AtomicUsize);

/// The exclusive hold's bit of [`Sections`].
const EXCLUSIVE: usize = 1 << (usize::BITS - 1);

impl Sections {
    /// A shared hold, once no exclusive one is held or claimed.
    pub(crate) fn read(&'static self) -> Shared {
        let mut wait = Wait(0);
        loop {
            let held = self.0.load(Ordering::Relaxed);
            if held & EXCLUSIVE == 0
                && self.0.compare_exchange_weak(held, held + 1, Ordering::Acquire, Ordering::Relaxed).is_ok()
            {
                return Shared(self);
            }
            wait.snooze();
        }
    }

    /// The exclusive hold, if nothing is held.
    pub(crate) fn try_write(&'static self) -> Option<Exclusive> {
        self.0.compare_exchange(0, EXCLUSIVE, Ordering::Acquire, Ordering::Relaxed).ok().map(|_| Exclusive(self))
    }

    /// The exclusive hold: claimed first, so that no section starts, and held
    /// once those running have ended.
    pub(crate) fn write(&'static self) -> Exclusive {
        let mut wait = Wait(0);
        while self.0.fetch_or(EXCLUSIVE, Ordering::Acquire) & EXCLUSIVE != 0 {
            wait.snooze();
        }
        let held = Exclusive(self);
        let mut wait = Wait(0);
        while self.0.load(Ordering::Acquire) != EXCLUSIVE {
            wait.snooze();
        }
        held
    }
}

/// A shared hold of the [`Sections`], until dropped.
#[must_use]
pub(crate) struct Shared(&'static Sections);

impl Drop for Shared {
    fn drop(&mut self) {
        self.0 .0.fetch_sub(1, Ordering::Release);
    }
}

/// The exclusive hold of the [`Sections`], until dropped.
#[must_use]
pub(crate) struct Exclusive(&'static Sections);

impl Drop for Exclusive {
    fn drop(&mut self) {
        self.0 .0.fetch_and(!EXCLUSIVE, Ordering::Release);
    }
}

/// Waiting for the [`Sections`]: yielding at first, then sleeping, up to a
/// millisecond at a time.  A section waits only while a `fork` is made, and
/// a `fork` only for the sections running to end.
struct Wait(u32);

impl Wait {
    fn snooze(&mut self) {
        if self.0 < 64 {
            std::thread::yield_now();
        } else {
            std::thread::sleep(std::time::Duration::from_micros(10 << (self.0 - 64).min(6)));
        }
        self.0 = self.0.saturating_add(1);
    }
}

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
        let mut workers = Joined(Vec::with_capacity(threads));
        // A `sink` that panics stops the workers too: they would otherwise
        // wait for it to sink more, and the scope for them.  Dropped before
        // `workers`, so they are stopped before they are joined.
        let _stop = Stop(&flow);
        for _ in 0..threads {
            let (tx, source, flow, work) = (tx.clone(), &source, &flow, &work);
            #[cfg(test)]
            STARTED.fetch_add(1, Ordering::SeqCst);
            workers.0.push(scope.spawn(move || {
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
            }));
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
        workers.join();
        outcome
    })
}

/// A call's workers, each joined when dropped: its thread has ended, not
/// only its closure, as the scope alone would wait for.  A thread's last
/// steps take a lock of the standard library's that every thread's first
/// steps take too (its stack overflow handler's), so a `fork` made as the
/// parallel section ends would leave it held in the child, and the child's
/// first thread would wait for it forever.
struct Joined<'scope>(Vec<std::thread::ScopedJoinHandle<'scope, ()>>);

impl Joined<'_> {
    /// Join every worker, then go on with the first one's panic, if one
    /// panicked: a worker joined is one the scope no longer reports.
    fn join(mut self) {
        let mut panicked = None;
        for worker in self.0.drain(..) {
            if let Err(panic) = worker.join() {
                panicked.get_or_insert(panic);
            }
        }
        if let Some(panic) = panicked {
            std::panic::resume_unwind(panic);
        }
    }
}

impl Drop for Joined<'_> {
    /// Joins what [`Joined::join`] did not, as the call unwinds.
    fn drop(&mut self) {
        for worker in self.0.drain(..) {
            let _ = worker.join();
        }
    }
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

    /// Runs that overlap end in any order and leave each other's count,
    /// and what was set, as they found them; a count set during a run is
    /// the count once the runs end.
    #[test]
    fn overlapping_runs_and_a_setting_made_during_one_leave_each_other_be() {
        let _serial = serial();
        let before = SET.load(Ordering::SeqCst);
        set_threads(2).unwrap();
        let three = for_this_run(3).unwrap();
        let four = for_this_run(4).unwrap();
        assert_eq!(threads().unwrap(), 4);
        drop(three);
        assert_eq!(threads().unwrap(), 4, "the run still going keeps its count");
        drop(four);
        assert_eq!(threads().unwrap(), 2);
        let run = for_this_run(3).unwrap();
        set_threads(5).unwrap();
        assert_eq!(threads().unwrap(), 3, "--threads comes before set_threads");
        drop(run);
        assert_eq!(threads().unwrap(), 5);
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

    /// A `sink` that panics with more items to come ends the call with its
    /// panic: the workers, out of items they may take, are stopped rather
    /// than left waiting for it, and the scope for them.
    #[test]
    fn a_sink_that_panics_ends_the_call_with_its_panic() {
        let _serial = serial();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let ended = std::panic::catch_unwind(|| {
                ordered(2, 0..1000, |i| i, |i| if i == 3 { panic!("sink") } else { Ok::<(), ()>(()) })
            });
            tx.send(ended.is_err()).unwrap();
        });
        assert_eq!(rx.recv_timeout(Duration::from_secs(20)), Ok(true), "the call did not end");
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
        // The worker's own panic, not the scope's word that one panicked.
        let panic = caught.unwrap_err();
        let message = panic.downcast_ref::<String>().map(String::as_str).unwrap_or_default();
        assert!(message.contains("the seventh item"), "{message:?}");
        assert!(!INSIDE.get(), "the section outlived the call");
        assert!(SECTIONS.try_write().is_some(), "the hold outlived the call");
    }
}
