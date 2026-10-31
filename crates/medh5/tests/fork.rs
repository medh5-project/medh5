//! A child forked while other threads use HDF5 can use it too
//! (`medh5::h5::fork`).  A test binary of its own: a forked child shares the
//! open files of every other test running at the fork, and their locks.
#![cfg(unix)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Threads in the middle of an HDF5 call at the fork, and waiting, queued,
/// for HDF5's lock the fork holds: the child takes the lock, in every one of
/// forty forks.
#[test]
fn a_child_forked_while_threads_use_hdf5_takes_hdf5s_lock() {
    medh5::h5::fork::install();
    let stop = Arc::new(AtomicBool::new(false));
    let busy: Vec<_> = (0..2)
        .map(|_| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    medh5::hdf5::sync::sync(|| std::thread::sleep(Duration::from_micros(50)));
                }
            })
        })
        .collect();
    let mut hung = 0;
    for _ in 0..40 {
        std::thread::sleep(Duration::from_millis(1));
        // SAFETY: the child takes HDF5's lock and exits.
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            // SAFETY: an alarm ends a child that waits for the lock forever,
            // late enough not to take one a loaded machine slows down for it.
            unsafe { libc::alarm(60) };
            medh5::hdf5::sync::sync(|| ());
            unsafe { libc::_exit(0) };
        }
        let mut status = 0;
        // SAFETY: waits for the child just forked.
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
        hung += usize::from(status != 0);
        // A hang takes that alarm to show: one says what all would.
        if hung > 0 {
            break;
        }
    }
    stop.store(true, Ordering::SeqCst);
    busy.into_iter().for_each(|t| t.join().unwrap());
    assert_eq!(hung, 0, "a child found HDF5's lock held");
}
