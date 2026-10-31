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

use std::cell::Cell;
use std::hint::spin_loop;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread::yield_now;
use std::time::{Duration, Instant};

use lock_api::{GuardNoSend, RawMutex};

/// The lock is held.
const LOCKED: u32 = 1;
/// One thread in the queue.
const ONE_QUEUED: u32 = 1 << 1;
/// How many threads are in the queue, in the 15 bits above [`LOCKED`].
const QUEUED: u32 = 0x7fff << 1;
/// Where the next number to call starts, in the top 16 bits. The queue is the
/// numbers from it on, as many as [`QUEUED`] counts.
const NEXT: u32 = 16;

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
fn next(state: u32) -> u16 {
    (state >> NEXT) as u16
}

/// The number after the last in the queue in `state`.
fn past_the_queue(state: u32) -> u16 {
    next(state).wrapping_add(((state & QUEUED) / ONE_QUEUED) as u16)
}

/// What a slot holds once `number` is called: the number, counted from 1, as a
/// slot starts at 0, and in the low bit whether the lock comes with the call.
fn called(number: u16, with_the_lock: bool) -> u32 {
    u32::from(number.wrapping_add(1)) << 1 | u32::from(with_the_lock)
}

/// Whether a slot that holds `call` has reached `number`, or a later number
/// that shares the slot. The numbers wrap, and those in use span less than half
/// of them.
fn reached(call: u32, number: u16) -> bool {
    ((call >> 1) as u16).wrapping_sub(number.wrapping_add(1)) as i16 >= 0
}

/// The lock itself, without the reentrancy `lock_api` adds.
pub struct RawLock {
    /// [`LOCKED`], [`QUEUED`] and the number at [`NEXT`].
    state: AtomicU32,
    /// Where each number sleeps, holding the last number called there.
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
    const INIT: Self = Self {
        state: AtomicU32::new(0),
        slots: [const { AtomicU32::new(0) }; SLOTS],
        hand_over_at: Cell::new(None),
    };

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
            match self.state.compare_exchange_weak(
                state,
                state & !LOCKED,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
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
    fn slot(&self, number: u16) -> &AtomicU32 {
        &self.slots[usize::from(number) % SLOTS]
    }

    #[cold]
    fn lock_contended(&self) {
        let mut spins = 0;
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            if state & LOCKED == 0 {
                match self.state.compare_exchange_weak(
                    state,
                    state | LOCKED,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
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
            if let Err(now) = self.state.compare_exchange_weak(
                state,
                state + ONE_QUEUED,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
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
    fn sleep_until_called(&self, number: u16) -> bool {
        let slot = self.slot(number);
        loop {
            let call = slot.load(Ordering::Acquire);
            if reached(call, number) {
                return call == called(number, true);
            }
            wait(slot, call);
        }
    }

    /// Let go of the lock with threads in the queue, which `state` shows, and
    /// call the first: every [`PATIENCE`], with the lock.
    #[cold]
    fn call_next(&self, mut state: u32) {
        let now = Instant::now();
        let with_the_lock = self.hand_over_at.get().is_none_or(|at| at <= now);
        if with_the_lock {
            self.hand_over_at.set(Some(now + PATIENCE));
        }
        // Only the thread holding the lock moves the number on, and threads
        // only join the queue, so that it is not empty.
        let number = next(state);
        let keep = if with_the_lock { !0 } else { !LOCKED };
        while let Err(now) = self.state.compare_exchange_weak(
            state,
            (state & keep).wrapping_add(1 << NEXT) - ONE_QUEUED,
            Ordering::Release,
            Ordering::Relaxed,
        ) {
            state = now;
        }
        // Having let go of the lock, this thread may stop here for as long as
        // later threads take to call a later number in this slot.
        let slot = self.slot(number);
        let mut held = slot.load(Ordering::Relaxed);
        while !reached(held, number) {
            match slot.compare_exchange_weak(
                held,
                called(number, with_the_lock),
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(now) => held = now,
            }
        }
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
        let number = u32::from(past_the_queue(state));
        self.state.store(number << NEXT | LOCKED, Ordering::Relaxed);
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
