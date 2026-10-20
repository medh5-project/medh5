//! The engine's locks a thread may hold while detached from its frontend,
//! held across a `fork`.
//!
//! A forked child has one thread, the one that forked: a lock another thread
//! held at the fork stays held in the child, and the child's first use of it
//! hangs.  A frontend whose callers fork (Python's `os.fork`) [`install`]s
//! handlers that take each of these locks as `fork` begins and let it go on
//! both sides as it returns, so each is free in the parent at the fork, and so
//! in the child.  They run inside `fork` itself, before and after any hook of
//! the frontend's: no code of the frontend's runs while they are held, so a
//! thread of its own cannot meet one held while it holds what the forking
//! thread needs (with the GIL, the interpreter).  A lock taken only by a
//! thread attached to the frontend needs none of this: the frontend's own lock
//! (the GIL, or a free-threaded interpreter stopping the world for the fork)
//! keeps every such thread out of it at the fork.

use std::any::Any;
use std::cell::RefCell;
use std::sync::{Mutex, TryLockError};

thread_local! {
    /// What [`prepare`] took, until [`release`]: on the thread that forks,
    /// which is the child's one thread too.
    static HELD: RefCell<Vec<Box<dyn Any>>> = const { RefCell::new(Vec::new()) };
}

/// One lock: taken if it is free, and waited for until it is.
struct Lock {
    take: fn() -> Option<Box<dyn Any>>,
    wait: fn(),
}

/// HDF5's lock, which a thread holds through every HDF5 call, and the files
/// checked self-contained, which an open takes.
const LOCKS: [Lock; 2] = [Lock { take: take_hdf5, wait: wait_hdf5 }, Lock { take: take_checked, wait: wait_checked }];

fn take_hdf5() -> Option<Box<dyn Any>> {
    crate::h5sys::LOCK.try_lock().map(|held| Box::new(held) as Box<dyn Any>)
}

fn wait_hdf5() {
    drop(crate::h5sys::LOCK.lock());
}

fn take_checked() -> Option<Box<dyn Any>> {
    free(super::ops::checked_files())
}

fn wait_checked() {
    drop(super::ops::checked_files().lock());
}

/// `lock`, if it is free.  A poisoned lock is taken as it stands, as its
/// users take it.
fn free<T: 'static>(lock: &'static Mutex<T>) -> Option<Box<dyn Any>> {
    match lock.try_lock() {
        Ok(held) => Some(Box::new(held)),
        Err(TryLockError::Poisoned(held)) => Some(Box::new(held.into_inner())),
        Err(TryLockError::WouldBlock) => None,
    }
}

/// Bracket every `fork` of this process with [`prepare`], [`release`] and
/// [`release_in_child`] (`pthread_atfork`); once, however often it is called.
#[cfg(unix)]
pub fn install() {
    extern "C" fn before() {
        prepare();
    }
    extern "C" fn parent() {
        release();
    }
    extern "C" fn child() {
        release_in_child();
    }
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        // SAFETY: the handlers take and let go of the engine's own locks,
        // on the forking thread, and never unwind.
        unsafe { libc::pthread_atfork(Some(before), Some(parent), Some(child)) };
    });
}

/// Take every lock, for a `fork` this thread is about to make; they are held
/// until [`release`].
///
/// None is waited for holding another: finding one taken, this lets go of
/// those it holds, waits until it is free, and starts again.  So no thread
/// holding one of them waits for this one.
fn prepare() {
    'all: loop {
        let mut held = Vec::with_capacity(LOCKS.len());
        for lock in &LOCKS {
            match (lock.take)() {
                Some(guard) => held.push(guard),
                None => {
                    drop(held);
                    (lock.wait)();
                    continue 'all;
                }
            }
        }
        HELD.with(|h| *h.borrow_mut() = held);
        return;
    }
}

/// Let go of what [`prepare`] took, in the parent after the fork.
fn release() {
    HELD.with(|h| h.borrow_mut().clear());
}

/// Let go of what [`prepare`] took, in the child, whose one thread is the one
/// that took it.
///
/// HDF5's lock is parking_lot's, whose queue of the threads waiting for it
/// the child inherits: the parent's threads that met the lock held for the
/// fork.  Let go with them queued, it could be handed to one --- parking_lot
/// hands a lock over, now and then, to be fair --- which the child does not
/// have, and the child's first HDF5 call would wait for it forever.  So they
/// are taken off the queue first.
fn release_in_child() {
    // SAFETY: only the raw lock's address is used, the key parking_lot
    // queues the lock's waiters under.
    let key = unsafe { crate::h5sys::LOCK.raw() } as *const _ as usize;
    // SAFETY: wakes the threads queued under that key, none of which this
    // process has; the lock they waited for is held, by this thread.
    unsafe { parking_lot_core::unpark_all(key, parking_lot_core::DEFAULT_UNPARK_TOKEN) };
    release();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;

    /// Whether another thread finds each lock free.  Held by this thread,
    /// none is, whatever other tests hold.
    fn free_elsewhere() -> Vec<bool> {
        std::thread::spawn(|| LOCKS.iter().map(|l| (l.take)().is_some()).collect()).join().unwrap()
    }

    /// `prepare` waits for a lock another thread holds, then holds every
    /// lock until `release`.
    #[test]
    fn prepare_waits_for_a_held_lock_and_holds_every_lock_until_release() {
        static RELEASED: AtomicBool = AtomicBool::new(false);
        let (held_tx, held_rx) = mpsc::channel();
        let holder = std::thread::spawn(move || {
            let held = super::super::ops::checked_files().lock();
            held_tx.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            RELEASED.store(true, Ordering::SeqCst);
            drop(held);
        });
        held_rx.recv().unwrap();
        prepare();
        assert!(RELEASED.load(Ordering::SeqCst), "prepare took a lock another thread held");
        holder.join().unwrap();
        assert_eq!(free_elsewhere(), vec![false, false], "a lock was free between prepare and release");
        release();
        // Another thread takes each, once whoever else had it lets it go.
        std::thread::spawn(|| LOCKS.iter().for_each(|l| (l.wait)())).join().unwrap();
    }
}
