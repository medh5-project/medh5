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

/// HDF5's lock, which a thread holds through every HDF5 call; the files
/// checked self-contained, which an open takes; the parallel sections, which
/// a call holds while its workers run (`crate::parallel`); and the command
/// line's runs, which `--threads` takes as a run begins and ends.
const LOCKS: [Lock; 4] = [
    Lock { take: take_hdf5, wait: wait_hdf5 },
    Lock { take: take_checked, wait: wait_checked },
    Lock { take: take_sections, wait: wait_sections },
    Lock { take: take_runs, wait: wait_runs },
];

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

/// Exclusively: a call's workers, which the child would not have, may hold
/// an allocator's lock or C-Blosc2's while it runs.  A call holding the
/// sections may be waiting for HDF5's lock, so this waits holding nothing;
/// no call enters a section holding HDF5's lock.
fn take_sections() -> Option<Box<dyn Any>> {
    crate::parallel::SECTIONS.try_write().map(|held| Box::new(held) as Box<dyn Any>)
}

fn wait_sections() {
    drop(crate::parallel::SECTIONS.write());
}

fn take_runs() -> Option<Box<dyn Any>> {
    free(&crate::parallel::RUNS)
}

fn wait_runs() {
    drop(crate::parallel::RUNS.lock());
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
/// are taken off the queue first.  The parallel sections queue no thread
/// (`crate::parallel::Sections`), so letting them go touches no queue.
///
/// The command line's runs end here: the child is in none of the parent's
/// (`crate::parallel::end_runs_in_child`).
fn release_in_child() {
    // SAFETY: only the raw lock's address is used, the key parking_lot
    // queues its waiters under.
    let hdf5 = unsafe { crate::h5sys::LOCK.raw() } as *const _ as usize;
    // SAFETY: wakes the threads queued under that key, none of which this
    // process has; the lock they waited for is held, by this thread.
    unsafe { parking_lot_core::unpark_all(hdf5, parking_lot_core::DEFAULT_UNPARK_TOKEN) };
    release();
    crate::parallel::end_runs_in_child();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;

    /// `fork`, the child closing every descriptor past stderr before
    /// anything else.  A file another test has open at the fork is the
    /// child's too, and HDF5 locks a file it opens: held open by the child,
    /// the file would stay locked after that test closed it, and its next
    /// open would be refused.
    ///
    /// # Safety
    ///
    /// As `libc::fork`'s: the child runs only what is safe after a fork.
    #[cfg(unix)]
    pub(crate) unsafe fn fork() -> libc::pid_t {
        // SAFETY: the caller's child runs only what is safe after a fork.
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            close_inherited();
        }
        pid
    }

    /// Close every descriptor past stderr, in a child just forked.
    #[cfg(unix)]
    fn close_inherited() {
        #[cfg(target_os = "linux")]
        // SAFETY: closes this process's descriptors from 3 up; none is used again.
        if unsafe { libc::syscall(libc::SYS_close_range, 3, libc::c_uint::MAX, 0) } == 0 {
            return;
        }
        // SAFETY: as above, one at a time, up to the process's limit.
        let limit = unsafe { libc::sysconf(libc::_SC_OPEN_MAX) };
        for fd in 3..limit.clamp(256, 1 << 16) as libc::c_int {
            unsafe { libc::close(fd) };
        }
    }

    /// The status of `pid`, a child just forked; one that has not exited
    /// in 20 seconds is killed and fails the test, rather than hanging it.
    #[cfg(unix)]
    pub(crate) fn exit_status(pid: libc::pid_t) -> libc::c_int {
        let mut status = 0;
        for _ in 0..2000 {
            // SAFETY: polls the child just forked.
            match unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) } {
                0 => std::thread::sleep(std::time::Duration::from_millis(10)),
                done => {
                    assert_eq!(done, pid);
                    return status;
                }
            }
        }
        // SAFETY: ends, and reaps, the child that hung.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
            libc::waitpid(pid, &mut status, 0);
        }
        panic!("a forked child had not exited after 20 s");
    }

    /// Whether another thread finds each lock free.  Held by this thread,
    /// none is, whatever other tests hold.
    fn free_elsewhere() -> Vec<bool> {
        std::thread::spawn(|| LOCKS.iter().map(|l| (l.take)().is_some()).collect()).join().unwrap()
    }

    /// `prepare` waits for a lock another thread holds, then holds every
    /// lock until `release`.
    #[test]
    fn prepare_waits_for_a_held_lock_and_holds_every_lock_until_release() {
        // It holds the parallel sections, which the tests of calls find free.
        let _serial = crate::parallel::tests::serial();
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
        assert_eq!(free_elsewhere(), vec![false; LOCKS.len()], "a lock was free between prepare and release");
        release();
        // Another thread takes each, once whoever else had it lets it go.
        std::thread::spawn(|| LOCKS.iter().for_each(|l| (l.wait)())).join().unwrap();
    }

    /// A child forked while threads run threaded calls, waiting for the
    /// parallel sections the fork holds and queued for them, runs a threaded
    /// call to the end, in every one of forty forks.
    #[cfg(unix)]
    #[test]
    fn a_child_forked_while_calls_use_threads_runs_one_to_the_end() {
        let _serial = crate::parallel::tests::serial();
        install();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let busy: Vec<_> = (0..2)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    while !stop.load(Ordering::SeqCst) {
                        crate::parallel::ordered(2, 0..4, |i| i, |_| Ok::<_, ()>(())).unwrap();
                    }
                })
            })
            .collect();
        let mut failed = Vec::new();
        for _ in 0..40 {
            std::thread::sleep(std::time::Duration::from_millis(1));
            // SAFETY: the child runs a threaded call, which makes no HDF5
            // call, and exits.
            let pid = unsafe { fork() };
            if pid == 0 {
                // SAFETY: an alarm ends a child that waits forever.
                unsafe { libc::alarm(5) };
                let sum = std::sync::atomic::AtomicUsize::new(0);
                let ran = crate::parallel::ordered(
                    2,
                    0..8usize,
                    |i| i,
                    |i| {
                        sum.fetch_add(i, Ordering::SeqCst);
                        Ok::<_, ()>(())
                    },
                );
                let code = if ran.is_ok() && sum.load(Ordering::SeqCst) == 28 { 0 } else { 1 };
                unsafe { libc::_exit(code) };
            }
            match exit_status(pid) {
                0 => {}
                status if libc::WIFSIGNALED(status) => failed.push(format!("signal {}", libc::WTERMSIG(status))),
                status => failed.push(format!("exit {}", libc::WEXITSTATUS(status))),
            }
        }
        stop.store(true, Ordering::SeqCst);
        busy.into_iter().for_each(|t| t.join().unwrap());
        assert_eq!(failed, Vec::<String>::new(), "a child could not run a threaded call");
    }

    /// A child forked while another thread begins and ends command line
    /// runs begins one of its own: the fork found the runs free.
    #[cfg(unix)]
    #[test]
    fn a_child_forked_while_runs_begin_and_end_begins_one() {
        let _serial = crate::parallel::tests::serial();
        install();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let busy = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    drop(crate::parallel::for_this_run(3).unwrap());
                }
            })
        };
        let mut failed = 0;
        for _ in 0..40 {
            std::thread::sleep(std::time::Duration::from_millis(1));
            // SAFETY: the child begins and ends a run, and exits.
            let pid = unsafe { fork() };
            if pid == 0 {
                let run = crate::parallel::for_this_run(2).unwrap();
                let code = i32::from(crate::parallel::threads().ok() != Some(2));
                drop(run);
                unsafe { libc::_exit(code) };
            }
            failed += usize::from(exit_status(pid) != 0);
        }
        stop.store(true, Ordering::SeqCst);
        busy.join().unwrap();
        assert_eq!(failed, 0, "a child could not begin a run");
    }

    /// A forked child is in none of the parent's command line runs: not
    /// another thread's, which nothing would end there, nor the forking
    /// thread's, whose count would override what the child sets (a loader's
    /// worker sets one thread).  The runs go on in the parent.
    #[cfg(unix)]
    #[test]
    fn a_forked_child_is_in_none_of_the_parents_runs() {
        let _serial = crate::parallel::tests::serial();
        install();
        let before = crate::parallel::threads().unwrap();
        let (begun, end) = (mpsc::channel(), mpsc::channel::<()>());
        let other = std::thread::spawn(move || {
            let run = crate::parallel::for_this_run(before + 2).unwrap();
            begun.0.send(()).unwrap();
            end.1.recv().unwrap();
            drop(run);
        });
        begun.1.recv().unwrap();
        let mut statuses = Vec::new();
        for own in [None, Some(before + 1)] {
            let run = own.map(|n| crate::parallel::for_this_run(n).unwrap());
            // SAFETY: the child reads and sets the count, begins and ends a
            // run, and exits.
            let pid = unsafe { fork() };
            if pid == 0 {
                let first = crate::parallel::threads().ok() == Some(before);
                let next = crate::parallel::for_this_run(before + 3).unwrap();
                let during = crate::parallel::threads().ok() == Some(before + 3);
                drop(next);
                let after = crate::parallel::threads().ok() == Some(before);
                crate::parallel::set_threads(1).unwrap();
                let set = crate::parallel::threads().ok() == Some(1);
                unsafe { libc::_exit(i32::from(!(first && during && after && set))) };
            }
            statuses.push(exit_status(pid));
            drop(run);
            assert_eq!(crate::parallel::threads().unwrap(), before + 2, "the parent's run went on");
        }
        end.0.send(()).unwrap();
        other.join().unwrap();
        assert_eq!(crate::parallel::threads().unwrap(), before);
        assert_eq!(statuses, [0, 0], "a child was in a run of its parent's");
    }

    /// A fork from another thread while its workers compress chunks waits
    /// until they have joined, and leaves a child that opens, writes and
    /// reads files, with threads and without.
    ///
    /// The other thread compresses and stores nothing: a thread that meets
    /// HDF5's lock held for the fork can leave parking_lot's table held in
    /// the child (#106), which is not what this test is about.
    #[cfg(unix)]
    #[test]
    fn a_child_forked_during_threaded_compression_opens_writes_and_reads() {
        use crate::array::{DType, NdArray};
        use crate::h5::chunks::{native_bytes, Chunks};
        use crate::h5::data::{self, Layout};
        use crate::storage::codecs::{resolve_profile, Role};
        let _serial = crate::parallel::tests::serial();
        install();
        crate::h5::init();
        let dir = tempfile::tempdir().unwrap();
        let shape = [48usize, 64, 64];
        let values = NdArray::from_vec(&shape, (0..48 * 64 * 64).map(|i| (i % 1000) as f64).collect())
            .unwrap()
            .astype(DType::I16);
        let layout = Layout {
            chunks: Some(vec![8, 32, 32]),
            filters: resolve_profile(Some("archive")).unwrap().codec(Role::Image).filters(),
        };
        let write = |path: &std::path::Path, threads: usize| -> crate::Result<()> {
            let file = crate::h5::file::create_truncate(path)?;
            let ds = data::create_empty(&file, "d", DType::I16, &shape, &layout)?;
            Chunks::of(&ds, DType::I16).expect("a Blosc2 dataset").write_rows(&values, 0, threads)
        };
        let read = |path: &std::path::Path| data::read(&crate::h5::file::open_read(path)?.dataset("d")?);
        // The chunks of `values`, as a write would cut and compress them.
        let source = crate::h5::file::create_truncate(&dir.path().join("source.h5")).unwrap();
        let chunks = Chunks::of(&data::create_empty(&source, "d", DType::I16, &shape, &layout).unwrap(), DType::I16)
            .expect("a Blosc2 dataset");
        let (codec, bytes, origins) = (chunks.codec(), native_bytes(&values), chunks.origins(0..shape[0]));
        let stop = AtomicBool::new(false);
        let failed = std::thread::scope(|scope| {
            scope.spawn(|| {
                while !stop.load(Ordering::SeqCst) {
                    crate::parallel::ordered(
                        4,
                        origins.iter(),
                        |origin| codec.compress(codec.cut(&bytes, &shape, &[0, 0, 0], origin)),
                        |chunk| {
                            drop(chunk);
                            Ok::<_, ()>(())
                        },
                    )
                    .unwrap();
                }
            });
            let mut failed = Vec::new();
            for k in 0..20 {
                std::thread::sleep(std::time::Duration::from_millis(3));
                let path = dir.path().join(format!("child-{k}.h5"));
                // SAFETY: the child writes and reads files of its own and exits.
                let pid = unsafe { fork() };
                if pid == 0 {
                    // SAFETY: an alarm ends a child that waits forever.
                    unsafe { libc::alarm(10) };
                    let failed =
                        [2, 1].into_iter().find_map(|threads| match write(&path, threads).and_then(|()| read(&path)) {
                            Ok(read) if read == values => None,
                            Ok(_) => Some(format!("{threads} thread(s): read back other values\n")),
                            Err(e) => Some(format!("{threads} thread(s): {e}\n")),
                        });
                    if let Some(why) = &failed {
                        // SAFETY: writes the reason to stderr, taking no lock
                        // a parent's thread may hold.
                        unsafe { libc::write(2, why.as_ptr().cast(), why.len()) };
                    }
                    let ok = failed.is_none();
                    // SAFETY: the child ends without running the parent's exit handlers.
                    unsafe { libc::_exit(if ok { 0 } else { 1 }) };
                }
                match exit_status(pid) {
                    0 => {}
                    status if libc::WIFSIGNALED(status) => failed.push(format!("signal {}", libc::WTERMSIG(status))),
                    status => failed.push(format!("exit {}", libc::WEXITSTATUS(status))),
                }
            }
            stop.store(true, Ordering::SeqCst);
            failed
        });
        assert_eq!(failed, Vec::<String>::new(), "a child could not write and read");
    }

    /// A fork made while a call holds the parallel sections and waits for
    /// HDF5's lock waits for the call to end: every result was sunk first.
    #[cfg(unix)]
    #[test]
    fn a_fork_waits_for_a_calls_workers() {
        use std::sync::atomic::AtomicUsize;
        static SUNK: AtomicUsize = AtomicUsize::new(0);
        let _serial = crate::parallel::tests::serial();
        install();
        let hdf5 = crate::h5sys::LOCK.lock();
        let call = std::thread::spawn(|| {
            crate::parallel::ordered(
                2,
                0..4,
                |i| i,
                |_| {
                    crate::h5::locked(|| ());
                    SUNK.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, ()>(())
                },
            )
            .unwrap();
        });
        // The call is in its section, its sink waiting for HDF5's lock.
        while crate::parallel::SECTIONS.try_write().is_some() {
            std::thread::yield_now();
        }
        let fork = std::thread::spawn(|| {
            // SAFETY: the child exits at once.
            let pid = unsafe { fork() };
            if pid == 0 {
                unsafe { libc::_exit(0) };
            }
            let sunk = SUNK.load(Ordering::SeqCst);
            (sunk, exit_status(pid))
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        drop(hdf5);
        call.join().unwrap();
        assert_eq!(fork.join().unwrap(), (4, 0), "the fork did not wait for the call");
    }
}
