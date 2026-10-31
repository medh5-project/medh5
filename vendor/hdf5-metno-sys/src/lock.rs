//! medh5: HDF5's lock on Unix (PATCHES.md, change 1).
//!
//! A lock that behaves as parking_lot's does, with its queue of waiters in one
//! word of its own. parking_lot keeps the threads waiting for any of its locks
//! in a table shared across the process, and a thread holds a lock of the
//! table's for a moment as it starts to wait. A `fork` in that moment leaves
//! the child that lock held by a thread it does not have, and letting go of a
//! lock with waiters needs it, so the child waits forever. parking_lot has no
//! way to hold or reset the table across a `fork`. This lock's queue is two
//! counts in [`RawLock`]'s word, so a child puts it right with one write
//! ([`RawLock::forget_waiters`]).
//!
//! A thread that finds the lock held spins for a moment, then takes a number
//! and sleeps until it is called. Letting go of the lock calls the next number,
//! so threads wake in the order they came: threads asleep on one address wake
//! in an order of the kernel's, which on macOS is not that one. A running
//! thread may take the lock ahead of the thread called, so a thread making
//! call after call does not stop to wake another each time; but every
//! [`PATIENCE`] the lock goes to the thread called, so none starves.
//!
//! The word alone says who has been called and who has the lock, in the one
//! write that makes the call, so a thread held up for any number of calls still
//! reads what happened to it there. A place to sleep only wakes a thread to look.

use std::cell::Cell;
use std::hint::spin_loop;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::thread::yield_now;
use std::time::{Duration, Instant};

use lock_api::{GuardNoSend, RawMutex};

/// The lock is held.
const LOCKED: u64 = 1;
/// The lock is held for the last number called, whose thread has yet to look.
/// The call sets it, and that thread clears it.
const HANDED: u64 = 1 << 1;
/// One thread in the queue.
const ONE_QUEUED: u64 = 1 << 2;
/// How many threads are in the queue, in the 15 bits above [`HANDED`].
const QUEUED: u64 = 0x7fff << 2;
/// Where the next number to call starts, in the top 47 bits. The queue is the
/// numbers from it on, as many as [`QUEUED`] counts. Calls cost a system call
/// each, so a thread can be held up for years of them before the numbers come
/// round to its own.
const NEXT: u32 = 17;

/// How often the lock goes to the thread called, rather than to whichever
/// takes it first. parking_lot's does every half a millisecond on average.
const PATIENCE: Duration = Duration::from_micros(500);

/// How many times a thread looks at a held lock before it sleeps, as
/// parking_lot does: spinning three times, each twice as long, then yielding.
const SPINS: u32 = 10;

/// How many places there are to sleep, one for each number in turn. Threads
/// whose numbers share a place wake together, and those not called sleep on.
const SLOTS: usize = 64;

/// The next number to call in `state`.
fn next(state: u64) -> u64 {
    state >> NEXT
}

/// How many numbers `to` is past `from`, in the numbers' width: they wrap.
fn numbers_past(from: u64, to: u64) -> u64 {
    to.wrapping_sub(from) << NEXT >> NEXT
}

/// The number after the last in the queue in `state`.
fn past_the_queue(state: u64) -> u64 {
    next(state) + (state & QUEUED) / ONE_QUEUED
}

/// Whether `number` is still to be called in `state`.
fn in_the_queue(state: u64, number: u64) -> bool {
    numbers_past(next(state), number) < (state & QUEUED) / ONE_QUEUED
}

/// Whether `state` holds the lock for `number`, which it has called.
fn handed_to(state: u64, number: u64) -> bool {
    state & HANDED != 0 && numbers_past(number, next(state)) == 1
}

/// The lock itself, without the reentrancy `lock_api` adds.
pub struct RawLock {
    /// [`LOCKED`], [`HANDED`], [`QUEUED`] and the number at [`NEXT`].
    state: AtomicU64,
    /// Where each number sleeps, counting the calls made there. A call changes
    /// the count only to wake the threads asleep on it: what it called is in
    /// `state`.
    slots: [AtomicU32; SLOTS],
    /// When the lock next goes to the thread called.
    hand_over_at: Cell<Option<Instant>>,
}

// SAFETY: `hand_over_at`, the one field not `Sync`, is used only by the thread
// that holds the lock, and taking the lock orders that use.
unsafe impl Sync for RawLock {}

// SAFETY: the word's `LOCKED` bit is taken only by a thread that finds it
// clear, atomically, and cleared only by the thread that holds the lock, which
// may instead leave it set for the one thread it calls with the lock.
unsafe impl RawMutex for RawLock {
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self =
        Self { state: AtomicU64::new(0), slots: [const { AtomicU32::new(0) }; SLOTS], hand_over_at: Cell::new(None) };

    type GuardMarker = GuardNoSend;

    fn lock(&self) {
        if !self.try_lock() {
            self.lock_contended();
        }
    }

    fn try_lock(&self) -> bool {
        self.state.fetch_or(LOCKED, Ordering::Acquire) & LOCKED == 0
    }

    unsafe fn unlock(&self) {
        let mut state = self.state.load(Ordering::Relaxed);
        while state & QUEUED == 0 {
            match self.state.compare_exchange_weak(state, state & !LOCKED, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => return,
                Err(now) => state = now,
            }
        }
        self.call_next(state);
    }

    fn is_locked(&self) -> bool {
        self.state.load(Ordering::Relaxed) & LOCKED != 0
    }
}

impl RawLock {
    /// The slot where `number` sleeps.
    fn slot(&self, number: u64) -> &AtomicU32 {
        &self.slots[number as usize % SLOTS]
    }

    #[cold]
    fn lock_contended(&self) {
        let mut spins = 0;
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            if state & LOCKED == 0 {
                match self.state.compare_exchange_weak(state, state | LOCKED, Ordering::Acquire, Ordering::Relaxed) {
                    Ok(_) => return,
                    Err(now) => state = now,
                }
                continue;
            }
            if state & QUEUED == 0 && spins < SPINS {
                if spins < 3 {
                    (0..2 << spins).for_each(|_| spin_loop());
                } else {
                    yield_now();
                }
                spins += 1;
                state = self.state.load(Ordering::Relaxed);
                continue;
            }
            // Past 32767 threads asleep, the next waits for room awake.
            if state & QUEUED == QUEUED {
                yield_now();
                state = self.state.load(Ordering::Relaxed);
                continue;
            }
            // Only with the lock held, so the thread that lets go calls it.
            if let Err(now) =
                self.state.compare_exchange_weak(state, state + ONE_QUEUED, Ordering::Relaxed, Ordering::Relaxed)
            {
                state = now;
                continue;
            }
            if self.sleep_until_called(past_the_queue(state)) {
                return;
            }
            spins = 0;
            state = self.state.load(Ordering::Relaxed);
        }
    }

    /// Whether the call for `number`, this thread's, came with the lock.
    fn sleep_until_called(&self, number: u64) -> bool {
        let slot = self.slot(number);
        loop {
            // The slot first: a call moves the word, then the slot, so this
            // thread either sees the call in the word or sleeps on a slot that
            // the call has yet to change.
            let calls = slot.load(Ordering::Acquire);
            let state = self.state.load(Ordering::Acquire);
            if !in_the_queue(state, number) {
                let with_the_lock = handed_to(state, number);
                if with_the_lock {
                    self.state.fetch_and(!HANDED, Ordering::Relaxed);
                }
                return with_the_lock;
            }
            wait(slot, calls);
        }
    }

    /// Let go of the lock with threads in the queue, which `state` shows, and
    /// call the first: every [`PATIENCE`], with the lock.
    #[cold]
    fn call_next(&self, mut state: u64) {
        let now = Instant::now();
        let with_the_lock = self.hand_over_at.get().is_none_or(|at| at <= now);
        if with_the_lock {
            self.hand_over_at.set(Some(now + PATIENCE));
        }
        // Only the thread holding the lock moves the number on, and threads
        // only join the queue, so that it is not empty.
        let number = next(state);
        let called = |state: u64| {
            let held = if with_the_lock { state | HANDED } else { state & !LOCKED };
            held.wrapping_add(1 << NEXT) - ONE_QUEUED
        };
        while let Err(now) =
            self.state.compare_exchange_weak(state, called(state), Ordering::Release, Ordering::Relaxed)
        {
            state = now;
        }
        let slot = self.slot(number);
        slot.fetch_add(1, Ordering::Release);
        wake_all(slot);
    }

    /// Forget the threads waiting for the lock, which a child of `fork` does
    /// not have: the numbers they hold will never be called, and letting go of
    /// the lock calls nobody.
    ///
    /// # Safety
    ///
    /// The caller is the one thread of a process just forked, and holds the
    /// lock.
    pub unsafe fn forget_waiters(&self) {
        let state = self.state.load(Ordering::Relaxed);
        self.state.store(past_the_queue(state) << NEXT | LOCKED, Ordering::Relaxed);
    }
}

/// Sleep while `word` holds `expected`. Returning does not say that it
/// changed: the caller looks again.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn wait(word: &AtomicU32, expected: u32) {
    let operation = libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG;
    let forever = std::ptr::null::<libc::timespec>();
    // SAFETY: the word outlives the call, and the kernel only reads it.
    unsafe { libc::syscall(libc::SYS_futex, word.as_ptr(), operation, expected, forever) };
}

/// Wake every thread sleeping in [`wait`] on `word`.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn wake_all(word: &AtomicU32) {
    let operation = libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG;
    // SAFETY: the kernel uses the word's address, and does not touch it.
    unsafe { libc::syscall(libc::SYS_futex, word.as_ptr(), operation, i32::MAX) };
}

/// macOS's counterpart of a futex, which `std::atomic::wait` in libc++ uses
/// too. Its header is private; the public `os_sync_wait_on_address` came with
/// macOS 14.4, and the wheels run on macOS 11.
#[cfg(target_vendor = "apple")]
mod ulock {
    use std::ffi::{c_int, c_void};

    /// `UL_COMPARE_AND_WAIT | ULF_NO_ERRNO`, from xnu's `sys/ulock.h`: a
    /// 32-bit word, private to the process.
    pub const WORD: u32 = 1 | 0x0100_0000;
    /// `ULF_WAKE_ALL`.
    pub const ALL: u32 = 0x100;

    unsafe extern "C" {
        pub fn __ulock_wait(operation: u32, address: *mut c_void, value: u64, timeout: u32) -> c_int;
        pub fn __ulock_wake(operation: u32, address: *mut c_void, value: u64) -> c_int;
    }
}

/// Sleep while `word` holds `expected`. Returning does not say that it
/// changed: the caller looks again.
#[cfg(target_vendor = "apple")]
fn wait(word: &AtomicU32, expected: u32) {
    // SAFETY: the word outlives the call, the kernel only reads it, and a
    // timeout of 0 is none.
    unsafe { ulock::__ulock_wait(ulock::WORD, word.as_ptr().cast(), expected.into(), 0) };
}

/// Wake every thread sleeping in [`wait`] on `word`.
#[cfg(target_vendor = "apple")]
fn wake_all(word: &AtomicU32) {
    let operation = ulock::WORD | ulock::ALL;
    // SAFETY: the kernel uses the word's address, and does not touch it.
    unsafe { ulock::__ulock_wake(operation, word.as_ptr().cast(), 0) };
}

/// Sleep briefly, on a system with no wait on an address medh5 knows of.
#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
fn wait(word: &AtomicU32, expected: u32) {
    if word.load(Ordering::Relaxed) == expected {
        std::thread::sleep(Duration::from_micros(50));
    }
}

/// Nothing: a thread sleeping in [`wait`] wakes by itself.
#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
fn wake_all(_word: &AtomicU32) {}

// `crates/medh5-sys/tests/lock.rs` runs these: this crate is not in the workspace.
#[cfg(test)]
mod tests {
    use std::sync::{mpsc, Arc, Mutex};

    use super::*;

    /// The number a thread takes on joining the queue of a held lock.
    fn join(lock: &RawLock) -> u64 {
        let state = lock.state.fetch_add(ONE_QUEUED, Ordering::Relaxed);
        assert_ne!(state & LOCKED, 0);
        past_the_queue(state)
    }

    /// Calls from now on come without the lock, as most do.
    fn no_hand_over(lock: &RawLock) {
        lock.hand_over_at.set(Some(Instant::now() + Duration::from_secs(3600)));
    }

    /// Let go of a lock this thread holds.
    fn let_go(lock: &RawLock) {
        // SAFETY: the caller holds the lock.
        unsafe { lock.unlock() };
    }

    /// `calls` numbers go by, one thread after another calling the next.
    fn calls_go_by(lock: &RawLock, calls: u32) {
        for _ in 0..calls {
            assert!(lock.try_lock());
            let number = join(lock);
            let_go(lock);
            assert!(!lock.sleep_until_called(number));
        }
    }

    /// A thread can be off the processor for thousands of calls on a loaded
    /// machine, between being called and looking. However many went by, it
    /// finds that it was called, and did not get the lock.
    #[test]
    fn a_thread_held_up_for_any_number_of_calls_finds_it_was_called() {
        for later in [1, 64, 32_768, 65_536, 100_000] {
            let lock = Arc::new(RawLock::INIT);
            assert!(lock.try_lock());
            let held_up = join(&lock);
            no_hand_over(&lock);
            let_go(&lock);
            calls_go_by(&lock, later);
            let (answer, answered) = mpsc::channel();
            let waiter = Arc::clone(&lock);
            std::thread::spawn(move || answer.send(waiter.sleep_until_called(held_up)));
            let found = answered.recv_timeout(Duration::from_secs(60));
            assert_eq!(found, Ok(false), "held up for {later} calls");
        }
    }

    /// A number 65,536 on from a thread's own, which a smaller field would
    /// make the same number, does not give that thread a lock handed to the
    /// other: two threads would hold it.
    #[test]
    fn a_thread_held_up_does_not_take_a_lock_handed_to_a_later_one() {
        let lock = RawLock::INIT;
        assert!(lock.try_lock());
        let held_up = join(&lock);
        no_hand_over(&lock);
        let_go(&lock);
        calls_go_by(&lock, 65_535);
        assert!(lock.try_lock());
        let later = join(&lock);
        lock.hand_over_at.set(None);
        let_go(&lock);
        assert!(!lock.sleep_until_called(held_up), "two threads were given the lock");
        assert!(lock.sleep_until_called(later));
        let_go(&lock);
    }

    /// A lock handed over stays held until the thread called lets go, and
    /// nothing is left of the hand-over.
    #[test]
    fn a_lock_handed_over_is_held_for_the_thread_called() {
        let lock = RawLock::INIT;
        assert!(lock.try_lock());
        let number = join(&lock);
        let_go(&lock);
        assert!(!lock.try_lock());
        assert!(lock.sleep_until_called(number));
        assert_eq!(lock.state.load(Ordering::Relaxed) & (HANDED | QUEUED), 0);
        assert!(lock.is_locked());
        let_go(&lock);
        assert!(lock.try_lock());
        let_go(&lock);
    }

    /// More threads than slots, so that some share one: those not called
    /// sleep on, and all are called in the order they came.
    #[test]
    fn threads_sharing_a_slot_are_called_in_the_order_they_came() {
        let threads = 2 * SLOTS + 1;
        let lock = Arc::new(RawLock::INIT);
        let order = Arc::new(Mutex::new(Vec::new()));
        assert!(lock.try_lock());
        let waiting: Vec<_> = (0..threads)
            .map(|i| {
                let (held, order) = (Arc::clone(&lock), Arc::clone(&order));
                let thread = std::thread::spawn(move || {
                    held.lock();
                    order.lock().unwrap().push(i);
                    let_go(&held);
                });
                while (lock.state.load(Ordering::Relaxed) & QUEUED) / ONE_QUEUED <= i as u64 {
                    yield_now();
                }
                thread
            })
            .collect();
        let_go(&lock);
        waiting.into_iter().for_each(|t| t.join().unwrap());
        assert_eq!(*order.lock().unwrap(), (0..threads).collect::<Vec<_>>());
    }

    /// What a child of `fork` does: the queue is gone, letting go calls
    /// nobody, and the numbers go on past those of the threads it forgot.
    #[test]
    fn a_lock_that_forgets_its_waiters_calls_none_of_them() {
        let lock = RawLock::INIT;
        assert!(lock.try_lock());
        let forgotten = [join(&lock), join(&lock), join(&lock)];
        // SAFETY: no other thread uses this lock, and this one holds it.
        unsafe { lock.forget_waiters() };
        assert_eq!(lock.state.load(Ordering::Relaxed) & QUEUED, 0);
        let_go(&lock);
        assert!(!lock.is_locked());
        assert!(lock.try_lock());
        let number = join(&lock);
        assert!(!forgotten.contains(&number));
        let_go(&lock);
        assert!(lock.sleep_until_called(number));
        let_go(&lock);
    }

    /// The numbers wrap, and everything that compares them survives it.
    #[test]
    fn numbers_past_the_top_of_their_field_go_on() {
        let lock = RawLock::INIT;
        let last = u64::MAX >> NEXT;
        lock.state.store(last << NEXT, Ordering::Relaxed);
        no_hand_over(&lock);
        calls_go_by(&lock, 3);
        assert_eq!(next(lock.state.load(Ordering::Relaxed)), 2);
        lock.state.store(last << NEXT, Ordering::Relaxed);
        assert!(lock.try_lock());
        let (first, second) = (join(&lock), join(&lock));
        lock.hand_over_at.set(None);
        let_go(&lock);
        assert!(lock.sleep_until_called(first));
        no_hand_over(&lock);
        let_go(&lock);
        assert!(!lock.sleep_until_called(second));
        assert!(!lock.is_locked());
    }

    /// A call made just as its thread goes to sleep, round after round. The
    /// call has to change what the thread sleeps on: otherwise the kernel can
    /// put the thread to sleep after the wake, and nothing wakes it again.
    #[test]
    fn a_thread_going_to_sleep_does_not_sleep_through_its_call() {
        const NOTHING: u64 = u64::MAX;
        const DONE: u64 = u64::MAX - 1;
        let lock = Arc::new(RawLock::INIT);
        let number = Arc::new(AtomicU64::new(NOTHING));
        let (answer, answered) = mpsc::channel();
        let (held, given) = (Arc::clone(&lock), Arc::clone(&number));
        let waiter = std::thread::spawn(move || loop {
            match given.swap(NOTHING, Ordering::Acquire) {
                NOTHING => spin_loop(),
                DONE => return,
                mine => {
                    held.sleep_until_called(mine);
                    answer.send(()).unwrap();
                }
            }
        });
        no_hand_over(&lock);
        for round in 0..100_000 {
            assert!(lock.try_lock());
            number.store(join(&lock), Ordering::Release);
            (0..round % 256).for_each(|_| spin_loop());
            let_go(&lock);
            answered.recv_timeout(Duration::from_secs(60)).expect("it slept through its call");
        }
        number.store(DONE, Ordering::Release);
        waiter.join().unwrap();
    }
}
