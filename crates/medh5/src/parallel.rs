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

/// In a forked child, end every run: the child is in none of the parent's.
/// A run on another thread would never end there, that thread being gone;
/// and the forking thread's own run is the parent's command, whose children
/// (a loader's workers) set their own count, which a run would override.
/// A child the command line starts itself is told its count
/// (`MEDH5_THREADS`).
pub(crate) fn end_runs_in_child() {
    RUNS.lock().unwrap_or_else(|e| e.into_inner()).1.clear();
    RUN.store(0, Ordering::SeqCst);
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

/// Sequential streams worked on in parallel: each stream's items, fed by
/// the caller, are decoded on any worker, several at once, and consumed by
/// one worker at a time, in the order they were fed, into the stream's own
/// state --- a digest's hasher, which must see its dataset's bytes in order.
///
/// `feed` runs on the calling thread and makes every HDF5 call: it hands
/// each item to its stream with [`Feed::push`] (or a failure to read one
/// with [`Feed::fail`]), and waits while twice `threads` items are in
/// flight.  `decode` and `consume` run on the workers and make none.  A
/// stream's first failure, in the order its items were fed, is its result,
/// and its later items are dropped; the other streams go on.  Returns each
/// stream's state, with its failure if it has one, in the order of `states`.
///
/// With one thread, or inside another parallel section, everything runs
/// here: each item is decoded and consumed as it is fed, and no thread is
/// started and no hold is taken.  The caller must not hold HDF5's lock, as
/// for [`ordered`].
pub(crate) fn streams<S, P, D, E>(
    threads: usize,
    states: Vec<S>,
    feed: impl FnOnce(&Feed<'_, S, P, D, E>),
    decode: impl Fn(P) -> Result<D, E> + Sync,
    consume: impl Fn(&mut S, D) -> Result<(), E> + Sync,
) -> Vec<(S, Option<E>)>
where
    S: Send,
    P: Send,
    D: Send,
    E: Send,
{
    let n = states.len();
    let board = Board {
        queues: (0..n).map(|_| std::collections::VecDeque::new()).collect(),
        states: states.into_iter().map(Some).collect(),
        failed: (0..n).map(|_| None).collect(),
        in_flight: 0,
        next: 0,
        fed: false,
        stop: false,
    };
    if threads <= 1 || INSIDE.get() {
        let here =
            Feed { board: Mutex::new(board), ready: Condvar::new(), window: 0, inline: Some((&decode, &consume)) };
        feed(&here);
        return here.board.into_inner().unwrap_or_else(|e| e.into_inner()).results();
    }
    debug_assert!(!crate::h5sys::LOCK.is_owned_by_current_thread(), "a parallel section entered holding HDF5's lock");
    let _hold = SECTIONS.read();
    INSIDE.set(true);
    let _inside = Inside;
    let shared = Feed { board: Mutex::new(board), ready: Condvar::new(), window: 2 * threads, inline: None };
    std::thread::scope(|scope| {
        let mut workers = Joined(Vec::with_capacity(threads));
        let feed_side = &shared;
        let _stop = StopBoard(feed_side);
        for _ in 0..threads {
            let (decode, consume) = (&decode, &consume);
            #[cfg(test)]
            STARTED.fetch_add(1, Ordering::SeqCst);
            workers.0.push(scope.spawn(move || {
                let _stop = StopBoard(feed_side);
                INSIDE.set(true);
                feed_side.work(decode, consume);
            }));
        }
        feed(feed_side);
        {
            let mut board = feed_side.lock();
            board.fed = true;
            feed_side.ready.notify_all();
        }
        workers.join();
    });
    shared.board.into_inner().unwrap_or_else(|e| e.into_inner()).results()
}

/// What [`streams`]' `feed` hands its items to.
pub(crate) struct Feed<'f, S, P, D, E> {
    board: Mutex<Board<S, P, D, E>>,
    ready: Condvar,
    /// How many items may be in flight; 0 when everything runs here.
    window: usize,
    /// The decode and consume steps, where everything runs here.
    inline: Option<Steps<'f, S, P, D, E>>,
}

/// A [`streams`] call's two steps, where everything runs on its caller.
type Steps<'f, S, P, D, E> = (Decode<'f, P, D, E>, Consume<'f, S, D, E>);

/// A [`streams`] call's first step.
type Decode<'f, P, D, E> = &'f (dyn Fn(P) -> Result<D, E> + Sync);

/// A [`streams`] call's second step.
type Consume<'f, S, D, E> = &'f (dyn Fn(&mut S, D) -> Result<(), E> + Sync);

/// A stream's items not yet consumed, each with the number it was fed as.
type Queue<P, D, E> = std::collections::VecDeque<(usize, Slot<P, D, E>)>;

/// The state [`streams`]' workers and its caller share.
struct Board<S, P, D, E> {
    /// Each stream's items not yet consumed, in the order they were fed.
    queues: Vec<Queue<P, D, E>>,
    /// Each stream's state; `None` while a worker consumes into it.
    states: Vec<Option<S>>,
    /// Each stream's failure, once it has one.
    failed: Vec<Option<E>>,
    /// Items fed and not yet consumed or dropped.
    in_flight: usize,
    /// The number the next item fed is given.
    next: usize,
    /// Whether `feed` has returned.
    fed: bool,
    /// Whether a worker or the caller panicked.
    stop: bool,
}

/// One item of a stream, as it goes.
enum Slot<P, D, E> {
    Fed(P),
    Decoding,
    Decoded(D),
    Failed(E),
}

impl<S, P, D, E> Board<S, P, D, E> {
    /// Stream `s` has failed with `e`: what it still holds is dropped, but
    /// for items being decoded, which their workers drop.
    fn fail(&mut self, s: usize, e: E) {
        self.failed[s].get_or_insert(e);
        let before = self.queues[s].len();
        self.queues[s].retain(|(_, slot)| matches!(slot, Slot::Decoding));
        self.in_flight -= before - self.queues[s].len();
    }

    fn results(self) -> Vec<(S, Option<E>)> {
        let states =
            self.states.into_iter().map(|s| s.expect("every stream's state is back once its workers have joined"));
        states.zip(self.failed).collect()
    }
}

impl<S, P, D, E> Feed<'_, S, P, D, E> {
    fn lock(&self) -> std::sync::MutexGuard<'_, Board<S, P, D, E>> {
        self.board.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Hand `item` to stream `s`, after what was fed to it before; dropped
    /// when the stream has failed.
    pub(crate) fn push(&self, s: usize, item: P) {
        if let Some((decode, consume)) = self.inline {
            let mut board = self.lock();
            let board = &mut *board;
            if board.failed[s].is_some() {
                return;
            }
            let state = board.states[s].as_mut().expect("a stream's state is here while it runs here");
            if let Err(e) = decode(item).and_then(|d| consume(state, d)) {
                board.failed[s] = Some(e);
            }
            return;
        }
        self.enqueue(s, Slot::Fed(item));
    }

    /// Stream `s` could not be fed its next item: `e` is its failure, unless
    /// an item fed before fails first.
    pub(crate) fn fail(&self, s: usize, e: E) {
        if self.inline.is_some() {
            self.lock().failed[s].get_or_insert(e);
            return;
        }
        self.enqueue(s, Slot::Failed(e));
    }

    /// Whether stream `s` has failed, or the call is stopping, so that
    /// nothing more need be read for it.
    pub(crate) fn failed(&self, s: usize) -> bool {
        let board = self.lock();
        board.stop || board.failed[s].is_some()
    }

    fn enqueue(&self, s: usize, slot: Slot<P, D, E>) {
        let mut board = self.lock();
        while board.in_flight >= self.window && !board.stop {
            board = self.ready.wait(board).unwrap_or_else(|e| e.into_inner());
        }
        if board.stop || board.failed[s].is_some() {
            return;
        }
        let seq = board.next;
        board.next += 1;
        board.queues[s].push_back((seq, slot));
        board.in_flight += 1;
        self.ready.notify_all();
    }

    /// A worker's loop: consume what is decoded, in each stream's order,
    /// else decode what was fed, until everything fed is consumed.
    fn work(&self, decode: Decode<'_, P, D, E>, consume: Consume<'_, S, D, E>) {
        let mut board = self.lock();
        loop {
            if board.stop {
                return;
            }
            let next = (0..board.queues.len()).find(|&s| {
                board.states[s].is_some()
                    && board.failed[s].is_none()
                    && matches!(board.queues[s].front(), Some((_, Slot::Decoded(_) | Slot::Failed(_))))
            });
            if let Some(s) = next {
                match board.queues[s].pop_front() {
                    Some((_, Slot::Failed(e))) => {
                        board.in_flight -= 1;
                        board.fail(s, e);
                    }
                    Some((_, Slot::Decoded(d))) => {
                        let mut state = board.states[s].take().expect("checked above");
                        drop(board);
                        let done = consume(&mut state, d);
                        board = self.lock();
                        board.states[s] = Some(state);
                        board.in_flight -= 1;
                        if let Err(e) = done {
                            board.fail(s, e);
                        }
                    }
                    _ => unreachable!("the front was checked above"),
                }
                self.ready.notify_all();
                continue;
            }
            let fed =
                board.queues.iter().enumerate().find_map(|(s, queue)| {
                    queue.iter().position(|(_, slot)| matches!(slot, Slot::Fed(_))).map(|i| (s, i))
                });
            if let Some((s, i)) = fed {
                let seq = board.queues[s][i].0;
                let Slot::Fed(item) = std::mem::replace(&mut board.queues[s][i].1, Slot::Decoding) else {
                    unreachable!("found as fed")
                };
                drop(board);
                let decoded = decode(item);
                board = self.lock();
                // Found again by its number: items before it may have been
                // consumed meanwhile, or the stream failed.
                if let Some(at) = board.queues[s].iter().position(|(n, _)| *n == seq) {
                    if board.failed[s].is_some() {
                        board.queues[s].remove(at);
                        board.in_flight -= 1;
                    } else {
                        board.queues[s][at].1 = match decoded {
                            Ok(d) => Slot::Decoded(d),
                            Err(e) => Slot::Failed(e),
                        };
                    }
                }
                self.ready.notify_all();
                continue;
            }
            if board.fed && board.in_flight == 0 {
                return;
            }
            board = self.ready.wait(board).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Stops every worker of a [`streams`] call, and the caller, when dropped
/// during a panic.
struct StopBoard<'a, 'f, S, P, D, E>(&'a Feed<'f, S, P, D, E>);

impl<S, P, D, E> Drop for StopBoard<'_, '_, S, P, D, E> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.lock().stop = true;
            self.0.ready.notify_all();
        }
    }
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
    /// What a stream test feeds: `n` items to each of `streams` streams,
    /// interleaved, each stream's state the items it consumed, in order;
    /// and the most items in flight at once, where nothing fails.
    #[allow(clippy::type_complexity)]
    fn feed_streams(
        threads: usize,
        streams_: usize,
        n: usize,
        fail: impl Fn(usize, usize) -> Option<&'static str> + Sync,
    ) -> (Vec<(Vec<usize>, Option<String>)>, usize) {
        let in_flight = AtomicUsize::new(0);
        let most = AtomicUsize::new(0);
        let out = streams(
            threads,
            vec![Vec::new(); streams_],
            |feed| {
                for i in 0..n {
                    for s in 0..streams_ {
                        let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                        most.fetch_max(now, Ordering::SeqCst);
                        feed.push(s, (s, i));
                    }
                }
            },
            |(s, i): (usize, usize)| {
                // Later items decode first, now and then.
                std::thread::sleep(Duration::from_micros(((n - i) % 5 * 40) as u64));
                match fail(s, i) {
                    Some(why) if why.starts_with("decode") => {
                        in_flight.fetch_sub(1, Ordering::SeqCst);
                        Err(format!("{why} {s}:{i}"))
                    }
                    _ => Ok((s, i)),
                }
            },
            |state: &mut Vec<usize>, (s, i): (usize, usize)| {
                in_flight.fetch_sub(1, Ordering::SeqCst);
                match fail(s, i) {
                    Some(why) if why.starts_with("consume") => Err(format!("{why} {s}:{i}")),
                    _ => {
                        state.push(i);
                        Ok(())
                    }
                }
            },
        );
        (out, most.load(Ordering::SeqCst))
    }

    #[test]
    fn each_stream_consumes_its_items_in_order_at_every_count() {
        let _serial = serial();
        for threads in [1, 2, 4] {
            let (out, most) = feed_streams(threads, 5, 60, |_, _| None);
            if threads > 1 {
                // The caller counts an item before pushing it, so one more.
                assert!(most <= 2 * threads + 1, "{most} in flight at {threads} threads");
            }
            for (state, failed) in out {
                assert_eq!(state, (0..60).collect::<Vec<_>>(), "{threads} threads");
                assert!(failed.is_none());
            }
        }
    }

    /// A stream's failure is the first in the order its items were fed,
    /// whichever worker met which first; its later items are dropped, and
    /// the other streams go on.
    #[test]
    fn a_stream_fails_at_its_first_failure_and_alone() {
        let _serial = serial();
        let fail = |s: usize, i: usize| match (s, i) {
            (1, 20) => Some("consume"),
            (1, 30) => Some("decode"),
            (3, 7) => Some("decode"),
            (3, 50) => Some("consume"),
            _ => None,
        };
        for threads in [1, 4] {
            let (out, _) = feed_streams(threads, 4, 60, fail);
            assert_eq!(out[1].1.as_deref(), Some("consume 1:20"), "{threads} threads");
            assert_eq!(out[1].0, (0..20).collect::<Vec<_>>());
            assert_eq!(out[3].1.as_deref(), Some("decode 3:7"));
            assert_eq!(out[3].0, (0..7).collect::<Vec<_>>());
            for s in [0, 2] {
                assert_eq!(out[s].0, (0..60).collect::<Vec<_>>());
                assert!(out[s].1.is_none());
            }
        }
    }

    #[test]
    fn one_thread_streams_on_the_caller() {
        let _serial = serial();
        let before = STARTED.load(Ordering::SeqCst);
        let caller = std::thread::current().id();
        let out = streams(
            1,
            vec![0usize; 2],
            |feed| (0..10).for_each(|i| feed.push(i % 2, i)),
            |i: usize| {
                assert_eq!(std::thread::current().id(), caller);
                Ok::<_, ()>(i)
            },
            |sum: &mut usize, i: usize| {
                *sum += i;
                Ok(())
            },
        );
        assert_eq!(out, vec![(20, None), (25, None)]);
        assert_eq!(STARTED.load(Ordering::SeqCst), before, "one thread started a worker");
    }

    #[test]
    fn a_stream_worker_that_panics_ends_the_call_with_its_panic() {
        let _serial = serial();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let caught = std::panic::catch_unwind(|| {
                streams(
                    4,
                    vec![(); 3],
                    |feed| (0..1000).for_each(|i| feed.push(i % 3, i)),
                    |i: usize| {
                        assert_ne!(i, 77, "the item 77");
                        Ok::<_, ()>(i)
                    },
                    |_: &mut (), _| Ok(()),
                )
            });
            let message = caught.unwrap_err().downcast_ref::<String>().cloned().unwrap_or_default();
            tx.send((message, INSIDE.get(), SECTIONS.try_write().is_some())).unwrap();
        });
        let (message, inside, free) = rx.recv_timeout(Duration::from_secs(20)).expect("the call did not end");
        assert!(message.contains("the item 77"), "{message:?}");
        assert!(!inside && free, "the section outlived the call");
    }
}
