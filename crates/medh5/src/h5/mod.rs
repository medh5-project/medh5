//! HDF5 plumbing: attribute codecs, dynamic-dtype datasets, atomic create,
//! copy-on-write and the low-level operations the high-level API lacks.
//!
//! Everything here is about *how* values reach HDF5, never about what they
//! mean.  Spec §2.5 fixes the attribute encoding; spec §14.4 fixes the write
//! model (atomic create, copy-on-write amend, unknown-object preservation).

pub mod attrs;
pub mod data;
pub mod file;
#[doc(hidden)]
pub mod fork;
pub mod graph;
pub mod ops;
mod window;

pub use attrs::AttrValue;
pub use hdf5::{Dataset, File, Group, Location};

/// Prepare the HDF5 library for MEDH5: register the Blosc2 and Zstandard
/// filters, and turn HDF5's dynamically loaded plugins off unless the user
/// asked for them (`MEDH5_ALLOW_PLUGINS=1`, or `HDF5_PLUGIN_PATH` naming the
/// directories to load them from).
///
/// Idempotent and cheap after the first call; every entry point that opens or
/// creates a file calls it, so a caller never has to.
pub fn init() {
    static PLUGINS: std::sync::Once = std::sync::Once::new();
    hdf5::sync::sync(|| {
        // SAFETY: the HDF5 global lock is held for the duration.  A refused
        // registration is not fatal: reading a dataset that needs the filter
        // then fails with HDF5's own error, which names the missing filter.
        unsafe {
            medh5_sys::register_blosc2_filter();
            medh5_sys::register_zstd_filter();
        }
        PLUGINS.call_once(|| {
            let path = std::env::var_os(plugins::SEARCH_PATH);
            let allow = std::env::var_os(plugins::ALLOW);
            plugins::settle(path.as_deref(), allow.as_deref());
        });
    });
}

/// HDF5's dynamically loaded plugins: off, unless asked for.
///
/// A dataset names its filters by number, and HDF5 loads one it has not
/// registered from the shared libraries in its plugin search path --- built
/// in, a directory of the machine that built the library and then
/// `/usr/local/hdf5/lib/plugin` (on Windows `%ALLUSERSPROFILE%\hdf5\lib\plugin`,
/// which any local user can create).  So a file naming a filter nobody had
/// registered ran whatever library had been planted there, as whoever opened
/// the file.  MEDH5 needs none: Blosc2 and Zstandard are compiled in, and
/// HDF5 has gzip, shuffle and Fletcher32 of its own.  Reading a
/// dataset whose filter is not registered is an error, HDF5's "filter plugins
/// disabled".
///
/// Asked for --- [`SEARCH_PATH`] naming directories, or [`ALLOW`] set and not
/// `0` --- plugins load as HDF5 loads them: from `HDF5_PLUGIN_PATH`, or,
/// with [`ALLOW`] alone, from HDF5's standard directory above, never from the
/// directory the library was built in.
pub(crate) mod plugins {
    use std::ffi::{CStr, OsStr};
    use std::os::raw::{c_int, c_uint};

    use crate::h5sys::h5pl;

    /// HDF5's variable naming the directories plugins are loaded from.
    pub(crate) const SEARCH_PATH: &str = "HDF5_PLUGIN_PATH";

    /// Set (to anything but `0`) to load HDF5's plugins from its standard
    /// directory when [`SEARCH_PATH`] names none.
    pub(crate) const ALLOW: &str = "MEDH5_ALLOW_PLUGINS";

    /// Where HDF5 looks for plugins by default, less the build directory it
    /// puts first.  HDF5 expands the variable on Windows.
    const STANDARD: &CStr =
        if cfg!(windows) { c"%ALLUSERSPROFILE%\\hdf5\\lib\\plugin" } else { c"/usr/local/hdf5/lib/plugin" };

    // As HDF5 declares them: `hdf5_sys` gives the first a pointer for the
    // mask it takes by value.
    unsafe extern "C" {
        fn H5PLset_loading_state(plugin_control_mask: c_uint) -> c_int;
        #[cfg(test)]
        fn H5PLget_loading_state(plugin_control_mask: *mut c_uint) -> c_int;
    }

    /// Whether the user asked for plugins: `HDF5_PLUGIN_PATH` naming
    /// directories, or `MEDH5_ALLOW_PLUGINS` set to anything but `0`.
    pub(crate) fn wanted(search_path: Option<&OsStr>, allow: Option<&OsStr>) -> bool {
        search_path.is_some_and(|p| !p.is_empty()) || allow.is_some_and(|a| !a.is_empty() && a != "0")
    }

    /// Put HDF5's plugin loading in the state the environment asks for.  The
    /// caller holds the HDF5 lock.
    pub(super) fn settle(search_path: Option<&OsStr>, allow: Option<&OsStr>) {
        if search_path.is_some_and(|p| !p.is_empty()) {
            // HDF5 read the directories from the variable itself.
            return;
        }
        let wanted = wanted(search_path, allow);
        // SAFETY: HDF5's own plugin controls, under its lock.
        unsafe {
            if !wanted {
                H5PLset_loading_state(0);
            }
            // The built-in search path goes either way: the build directory
            // always, and the standard one unless plugins were asked for.
            let mut n: c_uint = 0;
            if h5pl::H5PLsize(&mut n) >= 0 {
                for _ in 0..n {
                    h5pl::H5PLremove(0);
                }
            }
            if wanted {
                h5pl::H5PLappend(STANDARD.as_ptr());
            }
        }
    }

    /// HDF5's plugin-loading mask (0: none load) and the directories it
    /// would search.
    #[cfg(test)]
    pub(crate) fn state() -> (u32, Vec<String>) {
        super::locked(|| {
            let mut mask: c_uint = 0;
            let mut n: c_uint = 0;
            // SAFETY: HDF5's own plugin queries, under its lock, into buffers
            // of the sizes passed.
            unsafe {
                H5PLget_loading_state(&mut mask);
                h5pl::H5PLsize(&mut n);
                let paths = (0..n)
                    .filter_map(|i| {
                        let len = h5pl::H5PLget(i, std::ptr::null_mut(), 0);
                        let mut buf = vec![0u8; usize::try_from(len).ok()? + 1];
                        h5pl::H5PLget(i, buf.as_mut_ptr().cast(), buf.len());
                        CStr::from_bytes_until_nul(&buf).ok().map(|s| s.to_string_lossy().into_owned())
                    })
                    .collect();
                (mask, paths)
            }
        })
    }
}

/// What an object of a closed file reports.
pub const CLOSED: &str = "the file this object was read from has been closed";

/// Refuse an object whose file has been closed.
///
/// HDF5 answers questions about a closed object as if it were empty --- no
/// attribute, no member, shape `()` --- which reads as a malformed file rather
/// than a closed one, so the readers ask this first.
pub fn alive(obj: &hdf5::Location) -> crate::Result<()> {
    if obj.is_valid() {
        Ok(())
    } else {
        Err(crate::Error::File(CLOSED.into()))
    }
}

/// Run raw HDF5 C calls under the library's global lock.
///
/// A thread that would wait for the lock --- another thread holds it --- waits
/// through the frontend's [`on_wait`] hook, which the Python extension uses to
/// detach from the interpreter for the wait and the calls: a thread blocked
/// while attached holds up a free-threaded interpreter's stop-the-world pause
/// for as long as the holder's call runs.  A thread that already holds the
/// lock, or finds it free, never waits and runs `f` where it is: the hook
/// never lets go of the interpreter while this thread holds HDF5's lock, so
/// no thread ever waits for the interpreter while holding it.
pub(crate) fn locked<T>(f: impl FnOnce() -> T) -> T {
    // The library starts once, under the lock, before any thread holds the
    // lock here: `sync` waits for that start, and a thread holding the lock
    // while another starts the library would wait for it forever.
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| hdf5::sync::sync(|| ()));
    match crate::h5sys::LOCK.try_lock() {
        // Free, or already this thread's: nothing to wait for.
        Some(_held) => hdf5::sync::sync(f),
        None => match WAIT.get() {
            Some(wait) => {
                let (mut f, mut out) = (Some(f), None);
                wait(&mut || out = f.take().map(|f| hdf5::sync::sync(f)));
                out.expect("the wait hook runs what it is given")
            }
            None => hdf5::sync::sync(f),
        },
    }
}

static WAIT: std::sync::OnceLock<fn(&mut dyn FnMut())> = std::sync::OnceLock::new();

/// Set how a thread waits for HDF5's lock that another thread holds: `wait`
/// is given the wait and the calls to run, and runs them once, on this thread
/// (the Python extension, detached from the interpreter).  The first hook set
/// is kept; whether this one was.
#[doc(hidden)]
pub fn on_wait(wait: fn(&mut dyn FnMut())) -> bool {
    WAIT.set(wait).is_ok()
}

/// Start `command` while no file is between HDF5's open and the mark that
/// keeps it out of children ([`file::close_on_exec`]): under HDF5's lock,
/// which every open through medh5 holds until its file is marked, and which
/// every close takes too.  Only the start is held; the child runs, and is
/// waited on, outside it.
#[doc(hidden)]
pub fn spawn(command: &mut std::process::Command) -> std::io::Result<std::process::Child> {
    locked(|| {
        let mut child = command.spawn()?;
        settle(&mut child);
        Ok(child)
    })
}

/// Wait, briefly, until `child` has closed the descriptors it was not to
/// inherit.  Linux resumes the parent of a `vfork`-style spawn as soon as
/// `execve` replaces the child's memory, and closes the close-on-exec
/// descriptors only after: for those microseconds the child shares every open
/// file, and the lock HDF5 holds through it, so a file closed and opened again
/// meanwhile fails "unable to lock file".  Held under HDF5's lock, the wait
/// keeps every close and open out of that window.  A child that ends, or
/// takes longer than a second, is not waited for further.
#[cfg(target_os = "linux")]
fn settle(child: &mut std::process::Child) {
    use std::os::unix::io::RawFd;
    use std::path::PathBuf;

    let entries = |dir: &str| -> Vec<(RawFd, PathBuf)> {
        let Ok(read) = std::fs::read_dir(dir) else { return Vec::new() };
        read.flatten()
            .filter_map(|e| Some((e.file_name().to_str()?.parse().ok()?, std::fs::read_link(e.path()).ok()?)))
            .collect()
    };
    // SAFETY: `fcntl(F_GETFD)` only reads a descriptor's flags.
    let ours: Vec<(RawFd, PathBuf)> = entries("/proc/self/fd")
        .into_iter()
        .filter(|(fd, _)| unsafe { libc::fcntl(*fd, libc::F_GETFD) } & libc::FD_CLOEXEC != 0)
        .collect();
    let theirs = format!("/proc/{}/fd", child.id());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while std::time::Instant::now() < deadline && !matches!(child.try_wait(), Ok(Some(_)) | Err(_)) {
        if !entries(&theirs).iter().any(|entry| ours.contains(entry)) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_micros(50));
    }
}

/// Elsewhere a spawn is one call (`posix_spawn` on macOS), or nothing is
/// inherited (Windows).
#[cfg(not(target_os = "linux"))]
fn settle(_: &mut std::process::Child) {}

/// An object's path relative to a sample root (`/samples/x/images/CT` with
/// root `/samples/x` gives `images/CT`).
pub fn relative_name(name: &str, root: &str) -> String {
    let base = if root == "/" { "/".to_string() } else { format!("{}/", root.trim_end_matches('/')) };
    match name.strip_prefix(&base) {
        Some(rest) => rest.to_string(),
        None => name.trim_start_matches('/').to_string(),
    }
}

/// The last component of an HDF5 path.
pub fn basename(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

/// A chunked dataset of `dims` with no chunk written, as a file can state
/// one of any extent: for tests of extents past a reader's arithmetic, which
/// the high-level builder cannot make (it multiplies them).
#[cfg(test)]
pub(crate) fn unwritten(group: &hdf5::Group, name: &str, dtype: crate::array::DType, dims: &[u64]) -> hdf5::Dataset {
    use crate::array::DType;
    use crate::h5sys::{h5d, h5p, h5s, h5t};
    let mut chunk = vec![1u64; dims.len()];
    if let (Some(last), Some(n)) = (chunk.last_mut(), dims.last()) {
        *last = (*n).min(1024);
    }
    let cname = std::ffi::CString::new(name).unwrap();
    // SAFETY: under HDF5's lock; every identifier made here is closed here.
    let id = locked(|| unsafe {
        let tid = match dtype {
            DType::U8 => *h5t::H5T_NATIVE_UINT8,
            DType::U16 => *h5t::H5T_NATIVE_UINT16,
            DType::I16 => *h5t::H5T_NATIVE_INT16,
            DType::F32 => *h5t::H5T_NATIVE_FLOAT,
            DType::F64 => *h5t::H5T_NATIVE_DOUBLE,
            other => panic!("no unwritten {other:?} dataset"),
        };
        let dcpl = h5p::H5Pcreate(*h5p::H5P_CLS_DATASET_CREATE);
        h5p::H5Pset_chunk(dcpl, chunk.len() as i32, chunk.as_ptr());
        let space = h5s::H5Screate_simple(dims.len() as i32, dims.as_ptr(), std::ptr::null());
        let id = h5d::H5Dcreate2(group.id(), cname.as_ptr(), tid, space, h5p::H5P_DEFAULT, dcpl, h5p::H5P_DEFAULT);
        h5s::H5Sclose(space);
        h5p::H5Pclose(dcpl);
        if id >= 0 {
            h5d::H5Dclose(id);
        }
        id
    });
    assert!(id >= 0, "HDF5 refused an unwritten dataset of {dims:?}");
    group.dataset(name).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    /// The threads that have waited for HDF5's lock through the hook.
    static WAITERS: std::sync::Mutex<Vec<std::thread::ThreadId>> = std::sync::Mutex::new(Vec::new());

    /// The hook the tests set: records each thread that waits, and refuses a
    /// wait by a thread already holding the lock --- the Python extension's
    /// would let go of the interpreter there, and a thread holding HDF5's
    /// lock must never wait to take the interpreter back.
    fn recorded(run: &mut dyn FnMut()) {
        assert!(!crate::h5sys::LOCK.is_owned_by_current_thread(), "a thread holding HDF5's lock waited for it");
        WAITERS.lock().unwrap().push(std::thread::current().id());
        run();
    }

    fn waits_of(thread: std::thread::ThreadId) -> usize {
        WAITERS.lock().unwrap().iter().filter(|&&t| t == thread).count()
    }

    /// A thread waits for HDF5's lock through the hook only when another
    /// thread holds it; one that holds it already never waits again.
    #[test]
    fn locked_waits_through_the_hook_only_for_another_threads_lock() {
        on_wait(recorded);
        let me = std::thread::current().id();
        // Another test may hold the lock as this begins, so the outer call
        // may wait; the inner, made holding it, never does.
        locked(|| {
            let before = waits_of(me);
            locked(|| ());
            assert_eq!(waits_of(me), before, "a thread holding the lock waited for it again");
        });
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let holder = std::thread::spawn(move || {
            locked(|| {
                held_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })
        });
        held_rx.recv().unwrap();
        let waiter = std::thread::spawn(|| locked(|| ()));
        let id = waiter.thread().id();
        // Released only once the waiter is waiting, through the hook.
        while waits_of(id) == 0 {
            std::thread::yield_now();
        }
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        waiter.join().unwrap();
        assert_eq!(waits_of(id), 1);
    }

    /// Set in the child process of the test that sets the environment HDF5
    /// starts in.
    const CHILD: &str = "MEDH5_TEST_PLUGINS_CHILD";

    /// Off unless asked for: the loading mask is 0 and no directory is left
    /// to search.  A file naming a filter nobody registered made HDF5 load
    /// and run every library in its built-in search path, the build machine's
    /// directory among them.
    #[test]
    fn s14_2_dynamically_loaded_filters_are_off_unless_asked_for() {
        let set = |value: &'static str| Some(OsStr::new(value));
        assert!(!plugins::wanted(None, None));
        assert!(!plugins::wanted(set(""), set("")));
        assert!(!plugins::wanted(None, set("0")));
        assert!(plugins::wanted(None, set("1")));
        assert!(plugins::wanted(set("/opt/plugins"), None));
        init();
        let path = std::env::var_os(plugins::SEARCH_PATH);
        let allow = std::env::var_os(plugins::ALLOW);
        if !plugins::wanted(path.as_deref(), allow.as_deref()) {
            assert_eq!(plugins::state(), (0, Vec::new()));
        }
    }

    /// Asked for with `MEDH5_ALLOW_PLUGINS` alone, plugins load from HDF5's
    /// standard directory, never from the one the library was built in ---
    /// checked in a process of its own, whose environment is set before HDF5
    /// starts.
    #[test]
    fn s14_2_plugins_asked_for_load_from_the_standard_directory_only() {
        let standard = if cfg!(windows) {
            format!("{}\\hdf5\\lib\\plugin", std::env::var("ALLUSERSPROFILE").unwrap_or_default())
        } else {
            "/usr/local/hdf5/lib/plugin".to_string()
        };
        if std::env::var_os(CHILD).is_some() {
            init();
            let (mask, paths) = plugins::state();
            assert_ne!(mask, 0);
            assert_eq!(paths, vec![standard]);
            return;
        }
        let name = "h5::tests::s14_2_plugins_asked_for_load_from_the_standard_directory_only";
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", name, "--test-threads=1"])
            .env(CHILD, "1")
            .env(plugins::ALLOW, "1")
            .env_remove(plugins::SEARCH_PATH)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        // Through `spawn`, so the child catches no other test's file mid-open.
        let run = super::spawn(&mut command).unwrap().wait_with_output().unwrap();
        let out = String::from_utf8_lossy(&run.stdout);
        assert!(run.status.success() && out.contains("1 passed"), "{out}{}", String::from_utf8_lossy(&run.stderr));
    }

    /// A dataset whose filter is not registered is a read error, not a
    /// search for a library to run.
    #[test]
    fn s14_2_a_filter_nobody_registered_is_a_read_error() {
        use crate::h5sys::{h5d, h5p, h5s, h5t, h5z};
        init();
        let dir = tempfile::tempdir().unwrap();
        let file = file::create_truncate(&dir.path().join("unknown.h5")).unwrap();
        // A chunked dataset naming filter 40000 --- optional, as h5py names
        // a filter it has not got --- and one chunk stored as if the filter
        // had been applied to it.
        let (dims, bytes) = ([4u64, 4], [7u8; 16]);
        let id = locked(|| unsafe {
            let dcpl = h5p::H5Pcreate(*h5p::H5P_CLS_DATASET_CREATE);
            h5p::H5Pset_chunk(dcpl, 2, dims.as_ptr());
            h5p::H5Pset_filter(dcpl, 40000, h5z::H5Z_FLAG_OPTIONAL, 0, std::ptr::null());
            let space = h5s::H5Screate_simple(2, dims.as_ptr(), std::ptr::null());
            let id = h5d::H5Dcreate2(
                file.id(),
                c"x".as_ptr(),
                *h5t::H5T_NATIVE_UCHAR,
                space,
                h5p::H5P_DEFAULT,
                dcpl,
                h5p::H5P_DEFAULT,
            );
            h5s::H5Sclose(space);
            h5p::H5Pclose(dcpl);
            assert!(id >= 0);
            let origin = [0u64, 0];
            assert!(
                h5d::H5Dwrite_chunk(id, h5p::H5P_DEFAULT, 0, origin.as_ptr(), bytes.len(), bytes.as_ptr().cast()) >= 0
            );
            h5d::H5Dclose(id)
        });
        assert!(id >= 0);
        let err = data::read(&file.dataset("x").unwrap()).unwrap_err();
        assert!(matches!(err, crate::Error::Io(_)), "{err:?}");
        let path = std::env::var_os(plugins::SEARCH_PATH);
        let allow = std::env::var_os(plugins::ALLOW);
        if !plugins::wanted(path.as_deref(), allow.as_deref()) {
            // HDF5 looked for no library: its loader was off.
            assert!(err.to_string().ends_with("filter plugins disabled"), "{err}");
        }
    }
}
