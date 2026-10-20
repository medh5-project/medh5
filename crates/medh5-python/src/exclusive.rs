//! One thread at a time, refused rather than raced (E4).
//!
//! The extension's mutable objects --- a writer, a document, a payload, a
//! staged file --- are frozen pyclasses holding their state in an
//! [`Exclusive`].  A call takes its lock without waiting: readers share it,
//! and a call that changes the object holds it alone.  A call that finds it
//! taken, by another thread or by the same thread re-entering through a
//! callback, is refused with a `MEDH5Error` ("a SampleWriter is used by one
//! thread at a time").  Waiting instead could deadlock: a thread blocked on
//! the lock while attached to the interpreter never reaches the pause a
//! free-threaded collection stops every thread for, and the holder may be
//! waiting for that pause.  Refusing never blocks.
//!
//! The lock is taken after a call's arguments are converted, so a caller's
//! own code (an `__array__`, an `__index__`, an `__fspath__`) never runs
//! while it is held; what runs under it is the engine and the extension's
//! conversions of its own values.

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};

use pyo3::prelude::*;

use crate::errors::{BindError, R};

/// State one thread at a time may change, and any number may read.
pub struct Exclusive<T> {
    /// What the refusal names: `"a SampleWriter"`.
    what: &'static str,
    cell: RwLock<T>,
}

impl<T> Exclusive<T> {
    pub const fn new(what: &'static str, value: T) -> Self {
        Exclusive { what, cell: RwLock::new(value) }
    }

    /// The state, to read, alongside other readers.
    pub fn read(&self) -> R<RwLockReadGuard<'_, T>> {
        match self.cell.try_read() {
            Ok(guard) => Ok(guard),
            // A panic under the lock was raised to Python as it happened; the
            // state is what it left, as a `PyRefMut` left it before.
            Err(TryLockError::Poisoned(poisoned)) => Ok(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => Err(busy(self.what)),
        }
    }

    /// The state, to change, alone.
    pub fn write(&self) -> R<RwLockWriteGuard<'_, T>> {
        match self.cell.try_write() {
            Ok(guard) => Ok(guard),
            Err(TryLockError::Poisoned(poisoned)) => Ok(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => Err(busy(self.what)),
        }
    }

    /// The state to read, or `None` while a call holds it: for the garbage
    /// collector's traversal, which must not fail or wait.  What it skips it
    /// counts as reachable, so a skipped object is kept, never freed early.
    pub fn peek(&self) -> Option<RwLockReadGuard<'_, T>> {
        match self.cell.try_read() {
            Ok(guard) => Some(guard),
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// The state to change, or `None` while a call holds it: for `__clear__`.
    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, T>> {
        match self.cell.try_write() {
            Ok(guard) => Some(guard),
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }
}

/// The refusal of a call that found `what` in use.
pub fn busy(what: &str) -> BindError {
    BindError::Py(Python::attach(|py| {
        crate::errors::medh5_error(py, format!("{what} is used by one thread at a time, and this one is in use"))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_writer_is_refused_and_readers_share() {
        let cell = Exclusive::new("a thing", 1);
        {
            let a = cell.read().ok().expect("free");
            let b = cell.read().ok().expect("readers share");
            assert_eq!(*a + *b, 2);
            assert!(cell.try_write().is_none(), "a reader keeps a writer out");
        }
        let held = cell.write().ok().expect("free");
        assert!(cell.peek().is_none(), "a writer keeps everyone out");
        drop(held);
        assert!(cell.peek().is_some());
    }

    #[test]
    fn a_poisoned_lock_hands_out_what_the_panic_left() {
        let cell = std::sync::Arc::new(Exclusive::new("a thing", 1));
        let other = cell.clone();
        let _ = std::thread::spawn(move || {
            let mut held = other.write().ok().expect("free");
            *held = 2;
            panic!("under the lock");
        })
        .join();
        assert_eq!(*cell.read().ok().expect("recovered"), 2);
    }
}
