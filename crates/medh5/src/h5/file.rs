//! Opening files, and the atomic write model of spec §14.4.
//!
//! **Create** is atomic: write to a sibling temporary file, fsync it,
//! rename it over the target, fsync the directory.  A reader never observes a
//! partially written file and a crash leaves the previous file intact.  Every
//! copy-on-write command --- amend, `scrub --apply`, `fix`, `recompress` ---
//! goes through [`AtomicFile`], and when the target exists its permission bits
//! are carried onto the replacement: these are the commands most likely to be
//! pointed at data whose mode *is* its access control.  A target that is a
//! symbolic link is followed: the file it names is replaced, beside itself,
//! and the link kept.

use std::fs;
use std::path::{Path, PathBuf};

use hdf5::file::LibraryVersion;

use super::ops::check_self_contained;
use crate::json::repr_str;
use crate::{Error, Result};

/// Open an HDF5 file read-only, mapping failures onto `MEDH5FileError`.
///
/// Every file is also checked to be self-contained before it is handed back
/// (see [`check_self_contained`]), because this is the one door that readers,
/// validators and every copy-on-write path open files through.
pub fn open_read(path: &Path) -> Result<hdf5::File> {
    super::init();
    let failed = |why: &dyn std::fmt::Display| {
        Error::File(format!("failed to open {}: {why}", repr_str(&path.to_string_lossy())))
    };
    // Asked of the system first: HDF5's report of a missing file is its error
    // stack, which carries the time of the call, and a report that differs
    // from run to run is no report to compare (`validate --json`).
    match fs::metadata(path) {
        Ok(meta) if meta.is_dir() => return Err(failed(&"it is a directory, not a file")),
        // A named pipe or a device would block or stream inside `H5Fopen`:
        // a sample is a regular file (task-cache-1 Appendix A).
        Ok(meta) if !meta.is_file() => return Err(failed(&"it is not a regular file")),
        Ok(_) => {}
        Err(err) => return Err(failed(&os_reason(&err))),
    }
    let handle = marked(|| hdf5::File::open(path)).map_err(|e| failed(&e))?;
    check_self_contained(&handle, Some(path))?;
    Ok(handle)
}

/// Open a file through `open` and mark its descriptor, in one hold of HDF5's
/// lock: a child [`spawn`](super::spawn)ed by another thread starts before the
/// open or after the mark, never between them.
pub(crate) fn marked(open: impl FnOnce() -> hdf5::Result<hdf5::File>) -> hdf5::Result<hdf5::File> {
    phase!(Open);
    super::locked(|| {
        let file = open()?;
        close_on_exec(&file);
        Ok(file)
    })
}

/// Open an existing file for reading and writing, its descriptor marked as
/// [`open_read`]'s is.
pub(crate) fn open_rw(path: &Path) -> Result<hdf5::File> {
    super::init();
    Ok(marked(|| hdf5::File::open_rw(path))?)
}

/// The C runtime descriptor HDF5 opened `file` through, when its driver is
/// `sec2`, the default: another driver hands back no descriptor.
pub(crate) fn descriptor(file: &hdf5::File) -> Option<std::os::raw::c_int> {
    use hdf5::file::FileDriver;

    use crate::h5sys::{h5f, h5p};

    if !matches!(file.access_plist().and_then(|fapl| fapl.get_driver()), Ok(FileDriver::Sec2)) {
        return None;
    }
    super::locked(|| {
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: under HDF5's lock, for a file that stays open for the
        // call; `sec2` hands back a pointer to the descriptor it holds.
        unsafe {
            if h5f::H5Fget_vfd_handle(file.id(), h5p::H5P_DEFAULT, &mut handle) < 0 || handle.is_null() {
                return None;
            }
            Some(*(handle as *const std::os::raw::c_int))
        }
    })
}

/// The Windows handle behind a C runtime descriptor.  HDF5 and medh5 are one
/// module on one C runtime (HDF5 is built with CMake's default dynamic
/// runtime, and Rust links the same UCRT), so HDF5's descriptor is valid here.
#[cfg(windows)]
pub(crate) fn os_handle(fd: std::os::raw::c_int) -> Option<windows_sys::Win32::Foundation::HANDLE> {
    extern "C" {
        fn _get_osfhandle(fd: std::os::raw::c_int) -> isize;
    }
    // SAFETY: `_get_osfhandle` validates the descriptor and returns -1 (or
    // -2, for one with no stream) for one that names nothing.
    let os = unsafe { _get_osfhandle(fd) };
    (os != -1 && os != -2).then_some(os as windows_sys::Win32::Foundation::HANDLE)
}

/// Keep the descriptor HDF5 opened `file` through out of every program this
/// process starts.
///
/// HDF5 opens a file without `O_CLOEXEC`, and on Windows its handle is
/// inheritable, so a child started while it was open inherited it: it could
/// write a file opened for writing, and it held the lock HDF5 takes through it
/// after the file was closed, so the next open of the file failed "unable to
/// lock file" --- and on Windows every rename over it "Access is denied", so
/// every amendment --- for as long as the child ran.  Only the `sec2` driver,
/// the default, hands back a descriptor; another is left as it is.
///
/// What stays open: a child started by *other* code, not through
/// [`spawn`](super::spawn), on another thread between HDF5's open and this
/// mark; and a `fork` without `exec`, whose child shares every descriptor
/// (1.0 §14.4: handles are not inherited across `fork`).
pub(crate) fn close_on_exec(file: &hdf5::File) {
    let Some(fd) = descriptor(file) else { return };
    #[cfg(unix)]
    // SAFETY: `fd` is HDF5's open descriptor for `file`.
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFD);
        if flags >= 0 {
            libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC);
        }
    }
    #[cfg(windows)]
    if let Some(handle) = os_handle(fd) {
        use windows_sys::Win32::Foundation::{SetHandleInformation, HANDLE_FLAG_INHERIT};
        // SAFETY: `handle` is the open handle behind HDF5's descriptor.
        unsafe { SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0) };
    }
    #[cfg(not(any(unix, windows)))]
    let _ = fd;
}

/// What the system said went wrong, without the error number: "no such file
/// or directory" alike on every platform, and the system's own words
/// otherwise.
fn os_reason(err: &std::io::Error) -> String {
    if err.kind() == std::io::ErrorKind::NotFound {
        return "no such file or directory".into();
    }
    let full = err.to_string();
    match err.raw_os_error() {
        Some(code) => full.strip_suffix(&format!(" (os error {code})")).unwrap_or(&full).to_string(),
        None => full,
    }
}

/// Create (truncate) an HDF5 file with the format's library-version bounds.
///
/// The lower bound is HDF5 1.10, the oldest container the specification
/// promises to be readable by; the upper bound is the library's latest, so
/// features newer than 1.10 are used only where an object needs them.  The
/// root group is untimed, so the same content is the same bytes.
pub fn create_truncate(path: &Path) -> Result<hdf5::File> {
    super::init();
    marked(|| {
        hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            // The root group untimed, as `ops::new_group` makes every other.
            .with_fcpl(|p| p.obj_track_times(false))
            .create(path)
    })
    .map_err(|e| Error::File(format!("failed to create {}: {}", repr_str(&path.to_string_lossy()), e)))
}

/// The most links a chain is followed through, as Linux bounds one.
const MAX_LINKS: usize = 40;

/// The file a replace of `target` must replace: where its symbolic links
/// lead, so the link stays a link and the file it names is the one rewritten.
///
/// The temporary file was made beside the link and renamed over it, which
/// turned the link into an independent copy and left the file it named
/// untouched --- for `scrub --apply`, the identifying original, while the user
/// believed it scrubbed.  Only the last component is followed: a directory
/// reached through a link is the directory the rename happens in either way.
/// A link that leads nowhere names the file to create, as `open` takes it.
fn resolve_links(target: &Path) -> Result<PathBuf> {
    let mut path = target.to_path_buf();
    for _ in 0..MAX_LINKS {
        match fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                if !may_follow(&path, &meta) {
                    return Err(Error::File(format!(
                        "{} is a symbolic link another user owns, in a directory anyone can write to; it is not \
                         followed, as the system would not follow it to open the file",
                        repr_str(&path.to_string_lossy())
                    )));
                }
                let leads_to = fs::read_link(&path).map_err(|e| crate::error::os_error(&e, &path))?;
                // `join` keeps an absolute target as it is.
                path = match path.parent() {
                    Some(dir) => dir.join(leads_to),
                    None => leads_to,
                };
            }
            _ => return Ok(path),
        }
    }
    Err(Error::File(format!(
        "{} leads through more than {MAX_LINKS} symbolic links",
        repr_str(&target.to_string_lossy())
    )))
}

/// Whether a write may follow the link `link`: not one in a sticky directory
/// anyone can write to (`/tmp`) that neither the user nor the directory's owner
/// owns --- the rule the system applies to `open` (Linux's
/// `fs.protected_symlinks`), which a replace that resolves the link itself
/// would otherwise step around.
#[cfg(unix)]
fn may_follow(link: &Path, meta: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    let dir = link.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let Ok(dir_meta) = fs::metadata(dir) else { return true };
    // SAFETY: `geteuid` has no preconditions.
    let user = unsafe { libc::geteuid() };
    dir_meta.mode() & 0o1002 != 0o1002 || meta.uid() == user || meta.uid() == dir_meta.uid()
}

#[cfg(not(unix))]
fn may_follow(_link: &Path, _meta: &fs::Metadata) -> bool {
    true
}

/// A random part for a temporary name: a name taken is passed over, so it
/// need only make a collision unlikely.
fn unique() -> u32 {
    #[cfg(test)]
    if let Some(next) = NEXT_UNIQUE.with(|queue| queue.borrow_mut().pop()) {
        return next;
    }
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut h = RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.finish() as u32
}

/// How a temporary beside its target is named.
#[derive(Debug, Clone, Copy)]
enum Naming {
    /// `.{name}.tmp-{pid}-{hex}`: an HDF5 file or a document.  A cohort walk
    /// skips only `._*`, and would scan a temporary ending in `.medh5` as a
    /// sample.
    Hidden,
    /// `.{pid}-{hex}-{name}`: the target's suffix kept, for a foreign writer
    /// that picks a format by it (nibabel).
    Suffixed,
}

/// A sibling name for `target`'s temporary, under `naming`.
fn temporary_name_as(target: &Path, naming: Naming) -> PathBuf {
    let name = target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let (pid, unique) = (std::process::id(), unique());
    target.with_file_name(match naming {
        Naming::Hidden => format!(".{name}.tmp-{pid}-{unique:08x}"),
        Naming::Suffixed => format!(".{pid}-{unique:08x}-{name}"),
    })
}

/// The most names a staged file tries before it gives up: each one taken is
/// another writer's temporary, or something planted where one would go.
const NAME_TRIES: usize = 64;

/// The one way the engine replaces a file (§14.4): a temporary created
/// exclusively beside the file a path names --- its symbolic links followed,
/// so a link is kept and the file it names rewritten --- flushed, given that
/// file's mode, renamed over it, and the directory flushed.  A link or a file
/// already at a temporary's name is passed over for another name, never
/// written through.  Dropped without a commit, the temporary goes and the
/// target is untouched.
#[derive(Debug)]
struct Staged {
    /// The path asked for, which an error names.
    given: PathBuf,
    /// The file the commit replaces: `given`, its links followed.
    target: PathBuf,
    /// The temporary; empty once committed or discarded.
    tmp: PathBuf,
    /// The target's mode, carried onto the replacement; `None` for a new file.
    mode: Option<u32>,
    /// The target's owner and group, carried onto the replacement on POSIX;
    /// `None` for a new file, and off POSIX.
    owner: Option<(u32, u32)>,
}

impl Staged {
    fn create(given: &Path, naming: Naming) -> Result<Staged> {
        let target = resolve_links(given)?;
        let mode = existing_mode(&target);
        let owner = existing_owner(&target);
        let named = |e: std::io::Error| crate::error::os_error(&e, given);
        for _ in 0..NAME_TRIES {
            let tmp = temporary_name_as(&target, naming);
            match precreate(&tmp, mode) {
                Ok(()) => return Ok(Staged { given: given.to_path_buf(), target, tmp, mode, owner }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(named(e)),
            }
        }
        Err(Error::File(format!(
            "could not create a temporary file beside {}: {NAME_TRIES} names were taken",
            repr_str(&given.to_string_lossy())
        )))
    }

    /// Flush the closed temporary, give it the target's mode and rename it
    /// over the target.  `before_replace` runs last, and its refusal leaves
    /// the target as it was.  Whatever happens, the temporary is gone after.
    fn commit(&mut self, before_replace: impl FnOnce() -> Result<()>) -> Result<()> {
        let named = |e: std::io::Error| crate::error::os_error(&e, &self.given);
        let result = (|| -> Result<()> {
            // Flushed before the mode is set: a read-only file refuses the
            // writable handle Windows flushes through.
            fsync_path(&self.tmp).map_err(named)?;
            if let Some(mode) = self.mode {
                // The group before the mode: a change of owner clears the
                // set-id bits.
                let mode = match self.owner {
                    Some(owner) => carry_owner(&self.tmp, owner, mode),
                    None => mode,
                };
                set_mode(&self.tmp, mode);
            }
            #[cfg(test)]
            BEFORE_RENAME.with(|hook| {
                if let Some(hook) = hook.borrow_mut().as_mut() {
                    hook(&self.target)
                }
            });
            before_replace()?;
            replace(&self.tmp, &self.target).map_err(named)?;
            if let Some(parent) = self.target.parent() {
                fsync_dir(if parent.as_os_str().is_empty() { Path::new(".") } else { parent });
            }
            Ok(())
        })();
        if result.is_err() {
            remove_temporary(&self.tmp);
        }
        // Nothing is left for a drop to clean up.
        self.tmp = PathBuf::new();
        result
    }

    fn discard(&mut self) {
        if !self.tmp.as_os_str().is_empty() {
            remove_temporary(&self.tmp);
        }
        self.tmp = PathBuf::new();
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        self.discard();
    }
}

/// A file a foreign writer writes whole, then renamed over its target as the
/// engine renames its own (§14.4): a symbolic link at the target is kept and
/// the file it names replaced, that file's mode kept, and the data flushed.
/// The temporary keeps the target's suffix, for a writer that picks a format
/// by it.  Dropped without a commit, it is aborted.
#[doc(hidden)]
#[derive(Debug)]
pub struct StagedFile {
    staged: Staged,
}

impl StagedFile {
    /// Stage a replacement of `target`: a new, empty temporary beside the
    /// file it names, created exclusively.
    pub fn create(target: &Path) -> Result<StagedFile> {
        Ok(StagedFile { staged: Staged::create(target, Naming::Suffixed)? })
    }

    /// The temporary, which the foreign writer writes and closes.
    pub fn path(&self) -> &Path {
        &self.staged.tmp
    }

    /// The file the commit replaces: the target, its links followed.
    pub fn target(&self) -> &Path {
        &self.staged.target
    }

    /// Rename the written temporary over the target.
    pub fn commit(mut self) -> Result<()> {
        self.staged.commit(|| Ok(()))
    }

    /// Remove the temporary, leaving the target as it was.
    pub fn abort(mut self) {
        self.staged.discard();
    }
}

/// The permission bits of `target`, or `None` when it does not exist.
fn existing_mode(target: &Path) -> Option<u32> {
    let meta = fs::metadata(target).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        Some(meta.permissions().mode() & 0o7777)
    }
    #[cfg(not(unix))]
    {
        Some(if meta.permissions().readonly() { 0o444 } else { 0o666 })
    }
}

/// The owner and group of `target`, or `None` when it does not exist or the
/// system has no such thing.
fn existing_owner(target: &Path) -> Option<(u32, u32)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = fs::metadata(target).ok()?;
        Some((meta.uid(), meta.gid()))
    }
    #[cfg(not(unix))]
    {
        let _ = target;
        None
    }
}

/// Give the replacement `tmp` the replaced file's group --- and its owner,
/// when run as root --- and return the mode to give it.
///
/// A replacement belongs to whoever wrote it, in their primary group, so a
/// `0o640` file of a shared project group became readable by the writer's
/// group and unreadable by the project's.  Where the group cannot be given
/// (the writer is not in it), the mode returned grants the group only what it
/// grants everyone else ([`narrowed`]), which never widens access.  Access
/// control lists and extended attributes are not carried.
#[cfg(unix)]
fn carry_owner(tmp: &Path, (uid, gid): (u32, u32), mode: u32) -> u32 {
    use std::os::unix::fs::MetadataExt;
    let Ok(meta) = fs::symlink_metadata(tmp) else { return narrowed(mode) };
    // SAFETY: `geteuid` reads the process's effective user and cannot fail.
    let root = unsafe { libc::geteuid() } == 0;
    let uid = (root && meta.uid() != uid).then_some(uid);
    if uid.is_none() && meta.gid() == gid {
        return mode;
    }
    match std::os::unix::fs::lchown(tmp, uid, Some(gid)) {
        Ok(()) => mode,
        Err(_) => narrowed(mode),
    }
}

#[cfg(not(unix))]
fn carry_owner(_tmp: &Path, _owner: (u32, u32), mode: u32) -> u32 {
    mode
}

/// `mode` with its group bits replaced by its other bits: what a file in the
/// wrong group may grant that group without granting it more than anyone.
#[cfg(unix)]
fn narrowed(mode: u32) -> u32 {
    mode & !0o070 | (mode & 0o007) << 3
}

/// Create the temporary file before any data, readable by its owner only when
/// it will replace an existing file.
fn precreate(tmp: &Path, mode: Option<u32>) -> std::io::Result<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(if mode.is_none() { 0o666 } else { 0o600 });
    }
    let _ = mode;
    options.open(tmp)?;
    Ok(())
}

fn set_mode(path: &Path, mode: u32) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, fs::Permissions::from_mode(mode));
    }
    #[cfg(not(unix))]
    {
        if let Ok(meta) = fs::metadata(path) {
            let mut perms = meta.permissions();
            perms.set_readonly(mode & 0o222 == 0);
            let _ = fs::set_permissions(path, perms);
        }
    }
}

/// Flush a closed file to stable storage before it is renamed into place.
///
/// Windows flushes only through a handle that may write, so this runs before
/// the file is made read-only.
fn fsync_path(path: &Path) -> std::io::Result<()> {
    let file = fs::OpenOptions::new().read(true).write(cfg!(windows)).open(path)?;
    file.sync_all()?;
    Ok(())
}

/// Rename `from` over `to`.
///
/// Windows refuses to replace a read-only file, so a read-only target is made
/// writable for the rename --- the replacement carries the mode back --- and
/// read-only again if the rename fails.
fn replace(from: &Path, to: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        if let Ok(meta) = fs::metadata(to) {
            let kept = meta.permissions();
            if kept.readonly() {
                let mut writable = kept.clone();
                // Windows only, where this clears the read-only attribute.
                #[allow(clippy::permissions_set_readonly_false)]
                writable.set_readonly(false);
                fs::set_permissions(to, writable)?;
                let renamed = fs::rename(from, to);
                if renamed.is_err() {
                    let _ = fs::set_permissions(to, kept);
                }
                return renamed;
            }
        }
    }
    fs::rename(from, to)
}

/// Remove a temporary file, which may already be read-only: Windows does not
/// delete a read-only file.
fn remove_temporary(path: &Path) {
    #[cfg(windows)]
    {
        if let Ok(meta) = fs::metadata(path) {
            let mut writable = meta.permissions();
            if writable.readonly() {
                // Windows only, where this clears the read-only attribute.
                #[allow(clippy::permissions_set_readonly_false)]
                writable.set_readonly(false);
                let _ = fs::set_permissions(path, writable);
            }
        }
    }
    let _ = fs::remove_file(path);
}

fn fsync_dir(dir: &Path) {
    #[cfg(unix)]
    {
        if let Ok(handle) = fs::File::open(dir) {
            let _ = handle.sync_all();
        }
    }
    let _ = dir;
}

/// Whether two paths name one existing file: the same path, or one reached
/// through `..`, a symbolic link or a hard link.  A path naming nothing names
/// no file.
///
/// The file's identity is compared, not its name: the device and inode on
/// POSIX, the volume serial number and file index on Windows.  A canonical
/// path resolves symbolic links, but a hard link is a second name with nothing
/// to resolve, so comparing canonical paths let a command on Windows write
/// over the file it read.
#[doc(hidden)]
pub fn same_file(a: &Path, b: &Path) -> bool {
    matches!((identity(a), identity(b)), (Some(x), Some(y)) if x == y)
}

/// A key for the file `path` names, equal for two paths exactly when
/// [`same_file`] holds: to group many paths by file without comparing every
/// pair.  `None` for a path naming no file.
#[doc(hidden)]
pub fn file_key(path: &Path) -> Option<String> {
    identity(path).map(|id| format!("{id:?}"))
}

#[cfg(unix)]
fn identity(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let meta = fs::metadata(path).ok()?;
    Some((meta.dev(), meta.ino()))
}

#[cfg(windows)]
fn identity(path: &Path) -> Option<(u32, u64)> {
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION, FILE_FLAG_BACKUP_SEMANTICS,
    };
    // As `fs::metadata` opens it: asking for no access, so a file another
    // process holds open or locks still answers, and a directory opens too.
    let file = fs::OpenOptions::new().access_mode(0).custom_flags(FILE_FLAG_BACKUP_SEMANTICS).open(path).ok()?;
    let mut info = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: the handle is `file`'s, open for the call, and `info` is a
    // writable structure of the type the call fills.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
        return None;
    }
    Some((info.dwVolumeSerialNumber, u64::from(info.nFileIndexHigh) << 32 | u64::from(info.nFileIndexLow)))
}

#[cfg(not(any(unix, windows)))]
fn identity(path: &Path) -> Option<PathBuf> {
    fs::canonicalize(path).ok()
}

/// Close `file` and every object opened through it --- what h5py's
/// `File.close()` does.
///
/// HDF5 keeps a file open while any object in it is, so a dataset or group a
/// caller still holds would otherwise keep a written file open and unflushed
/// under a name that has already been renamed away, and keep a read file
/// locked against the next writer.  Those objects become invalid instead:
/// using one fails, and dropping one does nothing.  `flush` first, so a
/// writer learns of a failed write rather than losing it in the close.
///
/// A close that fails is an error too --- an object whose cached data could
/// not be written, the file's own last write --- so a commit never renames a
/// file whose close failed.
pub fn close_everything(file: hdf5::File, flush: bool) -> Result<()> {
    phase!(Open);
    use crate::h5sys::h5f::{self, H5F_scope_t};
    let fid = file.id();
    // Released below, so the handle's own drop must not release it again.
    std::mem::forget(file);
    super::locked(|| unsafe {
        let flushed = !flush || h5f::H5Fflush(fid, H5F_scope_t::H5F_SCOPE_LOCAL) >= 0;
        // Read now: the calls below clear HDF5's record of the failure.
        if !flushed {
            let why = super::ops::hdf5_error_text();
            release_all(fid);
            return Err(Error::Io(format!("could not flush the file: {why}")));
        }
        match release_all(fid) {
            None => Ok(()),
            Some(why) => Err(Error::Io(format!("could not close the file: {why}"))),
        }
    })
}

/// Release every identifier open through the file `fid`, and `fid` last;
/// what HDF5 said about the first release that failed, if one did.  The
/// caller holds the HDF5 lock.
///
/// # Safety
///
/// `fid` is a file identifier the caller owns.
unsafe fn release_all(fid: crate::h5sys::h5i::hid_t) -> Option<String> {
    use crate::h5sys::h5f;
    use crate::h5sys::h5i;
    let mut failed = None;
    // A failed release leaves the identifier open, so each is released until
    // it is gone or HDF5 refuses.
    let mut release = |id: h5i::hid_t| unsafe {
        while h5i::H5Iis_valid(id) > 0 {
            let left = h5i::H5Idec_ref(id);
            if left < 0 && failed.is_none() {
                failed = Some(super::ops::hdf5_error_text());
            }
            if left <= 0 {
                break;
            }
        }
    };
    let opened = |types: std::os::raw::c_uint| unsafe {
        let count = h5f::H5Fget_obj_count(fid, types);
        if count <= 0 {
            return Vec::new();
        }
        let mut ids = vec![0 as h5i::hid_t; count as usize];
        let found = h5f::H5Fget_obj_ids(fid, types, ids.len(), ids.as_mut_ptr());
        ids.truncate(found.max(0) as usize);
        ids
    };
    for id in opened(
        h5f::H5F_OBJ_LOCAL | h5f::H5F_OBJ_DATASET | h5f::H5F_OBJ_GROUP | h5f::H5F_OBJ_DATATYPE | h5f::H5F_OBJ_ATTR,
    ) {
        release(id);
    }
    // The file identifiers last: this one, and any handed out for it.
    for id in opened(h5f::H5F_OBJ_LOCAL | h5f::H5F_OBJ_FILE) {
        release(id);
    }
    release(fid);
    #[cfg(test)]
    if CLOSE_FAILS.with(|fails| fails.replace(false)) {
        failed.get_or_insert_with(|| "H5Fclose(): the close failed (a test made it)".into());
    }
    failed
}

/// What a test runs on the target just before a commit's rename.
#[cfg(test)]
pub(crate) type RenameHook = Box<dyn FnMut(&Path)>;

/// The HDF5 file identifiers open on `path` in this process.
#[cfg(test)]
pub(crate) fn open_on(path: &Path) -> usize {
    use crate::h5sys::h5f;
    use crate::h5sys::h5i::hid_t;
    let want = path.to_string_lossy().into_owned();
    super::locked(|| unsafe {
        let all = h5f::H5F_OBJ_ALL as hid_t;
        let count = h5f::H5Fget_obj_count(all, h5f::H5F_OBJ_FILE);
        let mut ids = vec![0 as hid_t; count.max(0) as usize];
        let found = h5f::H5Fget_obj_ids(all, h5f::H5F_OBJ_FILE, ids.len(), ids.as_mut_ptr());
        ids.iter()
            .take(found.max(0) as usize)
            .filter(|&&id| {
                let mut buf = vec![0u8; 4096];
                let n = h5f::H5Fget_name(id, buf.as_mut_ptr().cast(), buf.len());
                n > 0 && buf[..n as usize] == *want.as_bytes()
            })
            .count()
    })
}

/// The HDF5 identifiers of every kind --- files, groups, datasets,
/// attributes --- open on `path` in this process.  A group outlives the file
/// identifier it was opened through, and keeps the file open, so a count of
/// file identifiers alone misses it.
#[cfg(test)]
pub(crate) fn held_on(path: &Path) -> usize {
    use crate::h5sys::h5f;
    use crate::h5sys::h5i::hid_t;
    let want = path.to_string_lossy().into_owned();
    super::locked(|| unsafe {
        let all = h5f::H5F_OBJ_ALL as hid_t;
        let count = h5f::H5Fget_obj_count(all, h5f::H5F_OBJ_ALL);
        let mut ids = vec![0 as hid_t; count.max(0) as usize];
        let found = h5f::H5Fget_obj_ids(all, h5f::H5F_OBJ_ALL, ids.len(), ids.as_mut_ptr());
        ids.iter()
            .take(found.max(0) as usize)
            .filter(|&&id| {
                let mut buf = vec![0u8; 4096];
                let n = h5f::H5Fget_name(id, buf.as_mut_ptr().cast(), buf.len());
                n > 0 && buf[..n as usize] == *want.as_bytes()
            })
            .count()
    })
}

/// Run `hook` on the target just before every commit's rename on this
/// thread, or no longer with `None`.
#[cfg(test)]
pub(crate) fn before_rename(hook: Option<RenameHook>) {
    BEFORE_RENAME.with(|slot| *slot.borrow_mut() = hook);
}

#[cfg(test)]
thread_local! {
    static BEFORE_RENAME: std::cell::RefCell<Option<RenameHook>> = const { std::cell::RefCell::new(None) };
    /// The random parts the next temporary names take, last first: what a
    /// test plants something at.
    static NEXT_UNIQUE: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Set by a test to have the next close report the failure HDF5 reports
    /// of a close it could not finish.
    static CLOSE_FAILS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// What a writer saw of the file it read: its identity and the
/// `content_id` it declared.  A commit that would replace the file, or that
/// reads it again, refuses one that changed since (§14.4): two amendments,
/// or an amendment and a `repack`, both renamed over the file, and the first
/// one's work was lost without an error.
///
/// This detects a lost update; it does not exclude one.  The last check and
/// the rename are two calls, and no portable rename compares before it
/// replaces, so two commits that both pass the last check can still both
/// rename.  Closing that window needs the exclusive access §14.4 asks of
/// writers.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Snapshot {
    /// On POSIX the identity of the file HDF5 read through (device, inode,
    /// size, modification and change times); on Windows its size and
    /// last-write time --- not its creation time, which Windows carries over
    /// when a file is replaced by a rename.
    identity: Option<Identity>,
    content_id: Option<String>,
    /// What read it, as the refusal names it: "amendment", "rewrite".
    reader: &'static str,
}

/// A file's identity, as [`Snapshot`] compares it.
#[cfg(unix)]
type Identity = super::ops::FileIdentity;
#[cfg(not(unix))]
type Identity = (u64, u64);

#[cfg(unix)]
fn identity_of(meta: &fs::Metadata) -> Option<Identity> {
    super::ops::identity(meta)
}

#[cfg(windows)]
fn identity_of(meta: &fs::Metadata) -> Option<Identity> {
    use std::os::windows::fs::MetadataExt;
    Some((meta.file_size(), meta.last_write_time()))
}

#[cfg(not(any(unix, windows)))]
fn identity_of(meta: &fs::Metadata) -> Option<Identity> {
    let modified = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some((meta.len(), modified.as_nanos() as u64))
}

/// The `content_id` the root of `file` declares, as text.
fn declared_content_id(file: &hdf5::File) -> Result<Option<String>> {
    Ok(match super::attrs::read(file, "content_id")? {
        None => None,
        Some(super::attrs::AttrValue::Str(s)) => Some(s),
        Some(other) => Some(super::attrs::stringify_value(&other)),
    })
}

impl Snapshot {
    /// What `reader` sees of `file`, open read-only from `path`.
    pub(crate) fn of(file: &hdf5::File, path: &Path, reader: &'static str) -> Result<Snapshot> {
        #[cfg(unix)]
        let identity = super::ops::opened_identity(file, Some(path)).or_else(|| identity_of(&fs::metadata(path).ok()?));
        #[cfg(not(unix))]
        let identity = fs::metadata(path).ok().and_then(|meta| identity_of(&meta));
        Ok(Snapshot { identity, content_id: declared_content_id(file)?, reader })
    }

    /// Refuse `path` if it is no longer the file this snapshot saw (§14.4).
    /// The file is opened, to read its `content_id`, and closed again before
    /// this returns: Windows cannot replace a file that is open.
    pub(crate) fn check(&self, path: &Path) -> Result<()> {
        let changed = || {
            Error::File(format!(
                "{} changed after this {} read it; nothing was written: {} it again",
                repr_str(&path.to_string_lossy()),
                self.reader,
                if self.reader == "amendment" { "amend" } else { "run" },
            ))
        };
        let Some(identity) = fs::metadata(path).ok().and_then(|meta| identity_of(&meta)) else {
            return Err(changed());
        };
        if self.identity.is_some_and(|seen| seen != identity) {
            return Err(changed());
        }
        let file = open_read(path).map_err(|_| changed())?;
        let declared = declared_content_id(&file);
        close_everything(file, false)?;
        if declared? != self.content_id {
            return Err(changed());
        }
        Ok(())
    }
}

/// An HDF5 file being written to a temporary sibling of its target.
///
/// [`commit`](AtomicFile::commit) closes it and atomically replaces the
/// target; dropping it without committing removes the temporary file and leaves
/// the target untouched.
pub struct AtomicFile {
    staged: Staged,
    handle: Option<hdf5::File>,
    /// A file the commit refuses to replace the target over if it changed
    /// since it was read: the source of an amendment or a rewrite.
    unchanged: Option<(PathBuf, Snapshot)>,
}

impl AtomicFile {
    /// Start writing `target`: through its symbolic links, if it is one
    /// (`resolve_links`), so the file it names is the one replaced.
    pub fn create(target: &Path) -> Result<AtomicFile> {
        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let staged = Staged::create(target, Naming::Hidden)?;
        // A failed create drops `staged`, and the temporary with it.
        let handle = create_truncate(&staged.tmp)?;
        Ok(AtomicFile { staged, handle: Some(handle), unchanged: None })
    }

    /// The file being written.
    pub fn handle(&self) -> &hdf5::File {
        self.handle.as_ref().expect("an AtomicFile is open until committed")
    }

    /// The path that will be replaced: the file the target given named,
    /// its symbolic links followed.
    pub fn target(&self) -> &Path {
        &self.staged.target
    }

    /// The temporary path being written.
    pub fn temporary(&self) -> &Path {
        &self.staged.tmp
    }

    /// Have [`commit`](AtomicFile::commit) refuse, just before it replaces
    /// the target, if `source` is no longer the file `seen` saw (§14.4).
    pub(crate) fn expect_unchanged(&mut self, source: PathBuf, seen: Snapshot) {
        self.unchanged = Some((source, seen));
    }

    /// Close and atomically move the file into place.
    ///
    /// Everything opened through the file is closed with it, so nothing a
    /// caller kept can hold the written file open across the rename.  A
    /// source named by [`expect_unchanged`](AtomicFile::expect_unchanged)
    /// that changed is refused as the last thing before the rename, and the
    /// target is left as it was.
    pub fn commit(mut self) -> Result<()> {
        if let Some(handle) = self.handle.take() {
            close_everything(handle, true)?;
        }
        let unchanged = self.unchanged.take();
        self.staged.commit(|| match &unchanged {
            Some((source, seen)) => seen.check(source),
            None => Ok(()),
        })
    }

    /// Rewrite the file being built into a fresh temporary file, `copy`
    /// taking the old root to the new one, and carry on with that: what the
    /// copy does not reach is gone from the bytes.  HDF5 never reclaims an
    /// unlinked object's space, so a group removed from a file stays in it
    /// until the file is rewritten.
    pub fn rebuild(mut self, copy: impl FnOnce(&hdf5::Group, &hdf5::Group) -> Result<()>) -> Result<AtomicFile> {
        if let Some(handle) = self.handle.take() {
            close_everything(handle, true)?;
        }
        let mut fresh = Staged::create(&self.staged.given, Naming::Hidden)?;
        // The target the first temporary was staged for, whatever its links
        // lead to now.
        fresh.target = self.staged.target.clone();
        fresh.mode = self.staged.mode;
        let built = (|| -> Result<hdf5::File> {
            let src = open_read(&self.staged.tmp)?;
            let dst = create_truncate(&fresh.tmp)?;
            copy(&src.as_group()?, &dst.as_group()?)?;
            close_everything(src, false)?;
            Ok(dst)
        })();
        // On failure `fresh` drops, and its temporary with it.
        let handle = built?;
        self.staged.discard();
        self.staged = fresh;
        self.handle = Some(handle);
        Ok(self)
    }

    /// Discard the in-progress file, leaving any existing one untouched.
    pub fn abort(mut self) {
        self.discard();
    }

    fn discard(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = close_everything(handle, false);
        }
        self.staged.discard();
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        self.discard();
    }
}

/// Rebuild a file from an existing one, atomically.
///
/// The source is opened read-only and handed to `build` with the new file; the
/// source handle is closed **before** the replace, which matters on Windows,
/// where a file that is still open cannot be replaced.  `target` defaults to
/// `source` --- the rewrite-in-place case.
pub fn atomic_rewrite<T>(
    source: &Path,
    target: Option<&Path>,
    build: impl FnOnce(&hdf5::File, &hdf5::File) -> Result<T>,
) -> Result<T> {
    let dst_path = target.unwrap_or(source);
    let src = open_read(source)?;
    let seen = Snapshot::of(&src, source, "rewrite")?;
    let mut out = AtomicFile::create(dst_path)?;
    out.expect_unchanged(source.to_path_buf(), seen);
    let value = build(&src, out.handle())?;
    // Everything the build opened in the source, closed with it: a source
    // still open cannot be replaced on Windows.
    close_everything(src, false)?;
    out.commit()?;
    Ok(value)
}

/// Write `bytes` as `target` the way [`AtomicFile`] writes a file: to a
/// temporary sibling, flushed, its mode the target's, renamed into place
/// through the target's symbolic links.  A write cut short leaves the
/// previous file whole --- what a text file written over its input needs.
/// The bytes are written as given: a text's `\n` stays `\n` on every
/// platform.
#[doc(hidden)]
pub fn write_atomic(target: &Path, bytes: &[u8]) -> Result<()> {
    let mut staged = Staged::create(target, Naming::Hidden)?;
    // An I/O error names the file asked for, as the engine words one; the
    // temporary's name would tell the caller nothing.
    fs::write(&staged.tmp, bytes).map_err(|e| crate::error::os_error(&e, target))?;
    staged.commit(|| Ok(()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a caller kept from a closed file is invalid, not a reason the file
    /// stays open: the file reopens for writing at once.
    #[test]
    fn s14_4_close_everything_releases_what_was_opened_through_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.h5");
        {
            let file = create_truncate(&path).unwrap();
            file.new_dataset::<i32>().shape([3]).create("d").unwrap().write(&[1, 2, 3]).unwrap();
        }
        let file = open_read(&path).unwrap();
        let kept = file.dataset("d").unwrap();
        let group = file.as_group().unwrap();
        close_everything(file, false).unwrap();
        assert!(!kept.is_valid() && !group.is_valid());
        assert!(super::super::alive(&kept).is_err());
        let again = crate::h5::file::marked(|| hdf5::File::open_rw(&path)).unwrap();
        assert_eq!(again.dataset("d").unwrap().read_raw::<i32>().unwrap(), vec![1, 2, 3]);
        drop((kept, group));
    }

    /// A small file holding `value`, written through [`AtomicFile`].
    fn write_value(target: &Path, value: i32) -> Result<()> {
        let out = AtomicFile::create(target)?;
        out.handle().new_dataset::<i32>().shape([1]).create("v")?.write(&[value])?;
        out.commit()
    }

    fn read_value(path: &Path) -> i32 {
        open_read(path).unwrap().dataset("v").unwrap().read_raw::<i32>().unwrap()[0]
    }

    /// Nothing temporary is left in `dir`.
    fn no_temporaries(dir: &Path) -> bool {
        fs::read_dir(dir).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().contains(".tmp-"))
    }

    /// A missing file is reported in the system's words, the same on every
    /// run: HDF5's error stack carries the time of the call, which made
    /// `validate --json` differ from run to run.
    #[test]
    fn s14_4_a_missing_file_is_reported_the_same_on_every_run() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.medh5");
        let first = open_read(&missing).unwrap_err().to_string();
        let second = open_read(&missing).unwrap_err().to_string();
        assert_eq!(first, second);
        assert_eq!(
            first,
            format!("failed to open {}: no such file or directory", repr_str(&missing.to_string_lossy()))
        );
        let err = open_read(dir.path()).unwrap_err();
        assert!(matches!(err, Error::File(_)));
        assert!(err.to_string().ends_with(": it is a directory, not a file"), "{err}");
    }

    /// A named pipe given as a sample is refused before HDF5 opens it: the
    /// open blocked inside `H5Fopen` until something wrote to the pipe, so a
    /// task manifest naming one hung `task preflight`.
    #[cfg(unix)]
    #[test]
    fn s2_a_fifo_is_refused_before_hdf5_opens_it() {
        use std::os::unix::ffi::OsStrExt;
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("pipe.medh5");
        let name = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        let (tx, rx) = std::sync::mpsc::channel();
        let opening = fifo.clone();
        std::thread::spawn(move || tx.send(open_read(&opening).map(drop).map_err(|e| e.to_string())).unwrap());
        let err = rx.recv_timeout(std::time::Duration::from_secs(10)).expect("the open blocked").unwrap_err();
        assert!(err.ends_with(": it is not a regular file"), "{err}");
    }

    /// A replace through a symbolic link replaces the file the link names
    /// and keeps the link, whether the link is absolute or relative, or one
    /// of a chain: the link was replaced by an independent copy, and the file
    /// it named left as it was --- for `scrub --apply`, the identifying
    /// original.
    #[cfg(unix)]
    #[test]
    fn s14_4_a_replace_through_a_symlink_keeps_the_link() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        let farm = dir.path().join("farm");
        fs::create_dir_all(&data).unwrap();
        fs::create_dir_all(&farm).unwrap();
        let real = data.join("real.medh5");
        write_value(&real, 1).unwrap();
        let absolute = farm.join("absolute.medh5");
        symlink(&real, &absolute).unwrap();
        let relative = farm.join("relative.medh5");
        symlink("../data/real.medh5", &relative).unwrap();
        let chained = farm.join("chained.medh5");
        symlink("relative.medh5", &chained).unwrap();
        for (value, link) in [(2, &absolute), (3, &relative), (4, &chained)] {
            let inode = fs::metadata(&real).unwrap();
            write_value(link, value).unwrap();
            assert!(fs::symlink_metadata(link).unwrap().file_type().is_symlink(), "{link:?} is no longer a link");
            assert_eq!(read_value(&real), value, "{link:?}");
            assert_eq!(read_value(link), value, "{link:?}");
            use std::os::unix::fs::MetadataExt;
            assert_ne!(fs::metadata(&real).unwrap().ino(), inode.ino(), "replaced, not written in place");
            assert!(no_temporaries(&data) && no_temporaries(&farm));
        }
        // The replacement takes the real file's mode, as it always did.
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&real, fs::Permissions::from_mode(0o640)).unwrap();
        write_value(&chained, 5).unwrap();
        assert_eq!(fs::metadata(&real).unwrap().permissions().mode() & 0o777, 0o640);
    }

    /// A link that leads nowhere names the file to create, as `open` takes
    /// it; a loop of links is refused, not followed for ever.
    #[cfg(unix)]
    #[test]
    fn s14_4_a_dangling_link_is_created_through_and_a_loop_refused() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let link = dir.path().join("link.medh5");
        symlink("made.medh5", &link).unwrap();
        write_value(&link, 7).unwrap();
        assert!(fs::symlink_metadata(&link).unwrap().file_type().is_symlink());
        assert_eq!(read_value(&dir.path().join("made.medh5")), 7);
        let (a, b) = (dir.path().join("a.medh5"), dir.path().join("b.medh5"));
        symlink("b.medh5", &a).unwrap();
        symlink("a.medh5", &b).unwrap();
        let err = write_value(&a, 8).unwrap_err();
        assert!(err.to_string().contains("more than 40 symbolic links"), "{err}");
        assert!(no_temporaries(dir.path()));
    }

    /// A link in a sticky directory anyone can write to is followed only
    /// when the user or the directory's owner owns it, as the system follows
    /// it to open a file there; otherwise a link planted in `/tmp` could
    /// point a write at any file of the user's.
    #[cfg(unix)]
    #[test]
    fn s14_4_a_link_another_user_planted_in_a_shared_directory_is_not_followed() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("shared");
        fs::create_dir(&shared).unwrap();
        fs::set_permissions(&shared, fs::Permissions::from_mode(0o1777)).unwrap();
        let mine = dir.path().join("mine.medh5");
        write_value(&mine, 1).unwrap();
        let link = shared.join("out.medh5");
        symlink(&mine, &link).unwrap();
        write_value(&link, 2).unwrap();
        assert_eq!(read_value(&mine), 2, "the user's own link is followed");
        // Only root can give a link to another user.
        if unsafe { libc::geteuid() } == 0 {
            let path = std::ffi::CString::new(link.as_os_str().as_encoded_bytes()).unwrap();
            assert_eq!(unsafe { libc::lchown(path.as_ptr(), 4242, 4242) }, 0);
            let err = write_value(&link, 3).unwrap_err();
            assert!(err.to_string().contains("another user owns"), "{err}");
            assert_eq!(read_value(&mine), 2);
            assert!(no_temporaries(&shared) && no_temporaries(dir.path()));
        }
    }

    /// A read-only file is replaced, and stays read-only.  The temporary
    /// file was made read-only and *then* opened for writing to flush it,
    /// which Windows refuses, and Windows does not rename over a read-only
    /// file.
    #[test]
    fn s14_4_a_read_only_file_is_replaced_and_stays_read_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ro.medh5");
        write_value(&path, 1).unwrap();
        let mut perms = fs::metadata(&path).unwrap().permissions();
        perms.set_readonly(true);
        fs::set_permissions(&path, perms).unwrap();
        write_value(&path, 2).unwrap();
        assert_eq!(read_value(&path), 2);
        assert!(fs::metadata(&path).unwrap().permissions().readonly());
        assert!(no_temporaries(dir.path()));
        set_mode(&path, 0o644);
    }

    /// A group this process may give a file besides the one its new files
    /// get: a supplementary group, or, as root, any other.
    #[cfg(unix)]
    fn another_group() -> Option<u32> {
        // SAFETY: `getegid`, `geteuid` and `getgroups` read the process's
        // credentials; the buffer holds as many groups as the first call
        // counted.
        let (egid, euid) = unsafe { (libc::getegid(), libc::geteuid()) };
        let count = unsafe { libc::getgroups(0, std::ptr::null_mut()) };
        let mut groups = vec![0; usize::try_from(count).unwrap_or(0)];
        let count = unsafe { libc::getgroups(count, groups.as_mut_ptr()) };
        groups.truncate(usize::try_from(count).unwrap_or(0));
        let nobody = if egid == 65534 { 65533 } else { 65534 };
        groups.into_iter().find(|g| *g != egid).or((euid == 0).then_some(nobody))
    }

    /// The replacement belonged to the writer's primary group, so a `0o640`
    /// file of a shared project group became readable by the wrong group
    /// and unreadable by the right one.
    #[cfg(unix)]
    #[test]
    fn s14_4_an_amend_keeps_the_files_group() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        // GitHub's runner user has a supplementary group; a user with none
        // has no other group to keep.
        let Some(group) = another_group() else { return };
        let dir = tempfile::tempdir().unwrap();
        let (sample, report) = (dir.path().join("shared.medh5"), dir.path().join("shared.json"));
        write_value(&sample, 1).unwrap();
        write_atomic(&report, b"1").unwrap();
        for path in [&sample, &report] {
            std::os::unix::fs::chown(path, None, Some(group)).unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o640)).unwrap();
        }
        write_value(&sample, 2).unwrap();
        write_atomic(&report, b"2").unwrap();
        for path in [&sample, &report] {
            let meta = fs::metadata(path).unwrap();
            assert_eq!((meta.gid(), meta.mode() & 0o7777), (group, 0o640), "{path:?}");
        }
        assert_eq!((read_value(&sample), fs::read(&report).unwrap()), (2, b"2".to_vec()));
        assert!(no_temporaries(dir.path()));
    }

    /// A group the writer is not in cannot be given: the replacement grants
    /// its own group only what it grants everyone, never more than before.
    #[cfg(unix)]
    #[test]
    fn s14_4_a_group_not_given_is_granted_what_others_are() {
        assert_eq!(narrowed(0o640), 0o600);
        assert_eq!(narrowed(0o664), 0o644);
        assert_eq!(narrowed(0o2770), 0o2700);
        for mode in 0..0o10000 {
            let n = narrowed(mode);
            assert_eq!(n & !0o070, mode & !0o070, "{mode:o}");
            assert_eq!(n >> 3 & 0o7, mode & 0o7, "{mode:o}");
        }
        // A replacement that cannot be looked at is narrowed too.
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(carry_owner(&dir.path().join("gone"), (0, 0), 0o660), 0o600);
    }

    /// A close that fails stops the commit before the rename: the target is
    /// left as it was and the temporary file removed.  The final close's
    /// result was ignored, so a file whose close failed could be renamed
    /// into place.
    #[test]
    fn s14_4_a_failed_close_is_not_renamed_into_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.medh5");
        write_value(&path, 1).unwrap();
        CLOSE_FAILS.with(|fails| fails.set(true));
        let err = write_value(&path, 2).unwrap_err();
        assert!(err.to_string().starts_with("could not close the file: "), "{err}");
        assert_eq!(read_value(&path), 1);
        assert!(no_temporaries(dir.path()));
        CLOSE_FAILS.with(|fails| fails.set(true));
        let file = create_truncate(&dir.path().join("d.h5")).unwrap();
        assert!(close_everything(file, false).is_err());
    }

    #[test]
    fn s14_4_temporary_names_are_unique_within_a_process() {
        let names: std::collections::HashSet<PathBuf> =
            (0..50).map(|_| temporary_name_as(Path::new("x.medh5"), Naming::Hidden)).collect();
        assert_eq!(names.len(), 50);
        assert!(names.iter().all(|n| n.to_string_lossy().starts_with(".x.medh5.tmp-")));
    }

    /// `fsync_path` opens a writable descriptor on Windows, where a C runtime
    /// in text mode treats a trailing 0x1A as an end-of-file mark and strips
    /// it: the 1.x Windows job found four samples in eleven hundred one byte
    /// short.
    #[test]
    fn fsync_leaves_a_trailing_control_z_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ctrlz.bin");
        let payload: Vec<u8> = (0..=255u8).chain([0x1a]).collect();
        fs::write(&path, &payload).unwrap();
        fsync_path(&path).unwrap();
        assert_eq!(fs::read(&path).unwrap(), payload);
    }

    /// A child that runs until it is killed.  Started through
    /// [`super::super::spawn`], as every child of these tests is: a plain
    /// start on one test's thread could catch another test's file between its
    /// open and its mark, and fail that test --- the window `spawn` closes.
    fn long_lived_child() -> std::process::Child {
        let mut command = if cfg!(windows) {
            let mut c = std::process::Command::new("ping");
            c.args(["-n", "30", "127.0.0.1"]);
            c
        } else {
            let mut c = std::process::Command::new("sleep");
            c.arg("30");
            c
        };
        super::super::spawn(command.stdout(std::process::Stdio::null())).unwrap()
    }

    /// A child started while a file was open kept HDF5's descriptor, and the
    /// lock HDF5 holds through it: the file could not be opened again until
    /// the child exited.  The command line's `bench` test met it when a
    /// Python another test started outlived the sample being timed.  On
    /// Windows the handle was inheritable too, and a rename over the file
    /// --- every amendment's last step --- failed "Access is denied" while
    /// the child ran.
    #[test]
    fn s14_4_a_child_process_inherits_no_open_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.h5");
        let spare = dir.path().join("spare.h5");
        create_truncate(&path).unwrap();
        for door in [create_truncate, open_read, open_rw] {
            let file = door(&path).unwrap();
            let mut child = long_lived_child();
            drop(file);
            // For writing: an exclusive lock, which any lock the child kept refuses.
            let again = create_truncate(&path).map(drop);
            fs::write(&spare, b"x").unwrap();
            let renamed = fs::rename(&spare, &path);
            child.kill().unwrap();
            child.wait().unwrap();
            again.unwrap();
            renamed.unwrap();
            create_truncate(&path).unwrap();
        }
    }

    /// The descriptor HDF5 hands back is the file's own, so marking it marks
    /// the file: on Windows, the C runtime's descriptor names the handle HDF5
    /// opened, which holds only if HDF5 and medh5 share one C runtime.
    #[test]
    fn s14_4_the_descriptor_is_the_files_own() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.h5");
        let file = create_truncate(&path).unwrap();
        let fd = descriptor(&file).expect("sec2 hands back a descriptor");
        #[cfg(unix)]
        {
            let mut stat: libc::stat = unsafe { std::mem::zeroed() };
            assert_eq!(unsafe { libc::fstat(fd, &mut stat) }, 0);
            assert_eq!(Some((stat.st_dev as u64, stat.st_ino as u64)), identity(&path));
        }
        #[cfg(windows)]
        {
            use windows_sys::Win32::Storage::FileSystem::{GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION};
            let handle = os_handle(fd).expect("the descriptor names a handle");
            let mut info = BY_HANDLE_FILE_INFORMATION::default();
            assert_ne!(unsafe { GetFileInformationByHandle(handle, &mut info) }, 0);
            let own = (info.dwVolumeSerialNumber, u64::from(info.nFileIndexHigh) << 32 | u64::from(info.nFileIndexLow));
            assert_eq!(Some(own), identity(&path));
        }
        #[cfg(not(any(unix, windows)))]
        let _ = fd;
    }

    /// A child medh5 starts waits while another thread is between an open
    /// and its mark: while HDF5's lock is held, [`super::super::spawn`] does
    /// not return, and once it is released, it does.
    #[test]
    fn s14_4_a_child_starts_only_between_opens() {
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let holder = std::thread::spawn(move || {
            super::super::locked(|| {
                held_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            })
        });
        held_rx.recv().unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let starter = std::thread::spawn(move || {
            let mut command = if cfg!(windows) {
                let mut c = std::process::Command::new("cmd");
                c.args(["/C", "exit 0"]);
                c
            } else {
                std::process::Command::new("true")
            };
            let child = super::super::spawn(&mut command);
            started_tx.send(()).unwrap();
            child.unwrap().wait().unwrap();
        });
        let waited = started_rx.recv_timeout(std::time::Duration::from_millis(200));
        assert!(waited.is_err(), "a child started while HDF5's lock was held");
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        started_rx.recv_timeout(std::time::Duration::from_secs(10)).expect("the child starts once the lock is free");
        starter.join().unwrap();
    }

    /// Off POSIX the 2.0.1 checks compared canonical paths, which a hard link
    /// gets past: on Windows `task reconcile -o` wrote the manifest over its
    /// source and `clinical export -o` over the sample it read.
    #[test]
    fn s14_4_same_file_knows_every_name_of_one_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.medh5");
        let twin = dir.path().join("b.medh5");
        fs::write(&file, b"same bytes").unwrap();
        fs::write(&twin, b"same bytes").unwrap();
        let hard = dir.path().join("hard.medh5");
        fs::hard_link(&file, &hard).unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        let dotted = dir.path().join("sub").join("..").join("a.medh5");
        let missing = dir.path().join("missing.medh5");

        assert!(same_file(&file, &file));
        assert!(same_file(&file, &hard));
        assert!(same_file(&dotted, &file));
        assert!(!same_file(&file, &twin), "equal bytes are not one file");
        assert!(!same_file(&file, &missing));
        assert!(!same_file(&missing, &missing), "a path naming nothing names no file");
        #[cfg(unix)]
        {
            let link = dir.path().join("link.medh5");
            std::os::unix::fs::symlink(&file, &link).unwrap();
            assert!(same_file(&link, &hard));
        }
    }

    /// The source of a rewrite in place is closed before the rename.
    ///
    /// `with open_h5(...) as src, atomic_h5(...) as dst:` exits right to left,
    /// so 1.x once replaced the file while its source was still open.  POSIX
    /// allows that, so it passed on Linux and macOS; Windows refuses to replace
    /// an open file, so every rewrite in place --- a repack, `recompress`
    /// without `out` --- failed there.  This asserts the ordering directly
    /// rather than relying on a platform to notice.
    #[test]
    fn s14_4_the_source_is_closed_before_the_replace() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rw.h5");
        {
            let file = create_truncate(&path).unwrap();
            file.new_dataset::<i32>().shape([3]).create("d").unwrap().write(&[1, 2, 3]).unwrap();
        }
        let seen = std::rc::Rc::new(std::cell::Cell::new(None));
        let record = seen.clone();
        BEFORE_RENAME.with(|hook| *hook.borrow_mut() = Some(Box::new(move |target| record.set(Some(open_on(target))))));
        let kept = std::cell::RefCell::new(None);
        atomic_rewrite(&path, None, |src, dst| {
            assert_eq!(open_on(&path), 1, "the source is open while the build runs");
            let data = src.dataset("d")?;
            dst.new_dataset::<i32>().shape([3]).create("d")?.write(&data.read_raw::<i32>()?)?;
            // A view the build kept must not hold the source open either.
            *kept.borrow_mut() = Some(data);
            Ok(())
        })
        .unwrap();
        BEFORE_RENAME.with(|hook| *hook.borrow_mut() = None);
        assert_eq!(seen.get(), Some(0), "the source was still open at the rename");
        assert!(!kept.into_inner().unwrap().is_valid());
        let again = open_read(&path).unwrap();
        assert_eq!(again.dataset("d").unwrap().read_raw::<i32>().unwrap(), vec![1, 2, 3]);
    }

    /// A copy of `path` holding `value`, renamed over it: what another
    /// writer's commit does.
    fn replaced_by_another(path: &Path, value: i32) {
        let other = path.with_extension("other");
        fs::copy(path, &other).unwrap();
        open_rw(&other).unwrap().dataset("v").unwrap().write(&[value]).unwrap();
        fs::rename(&other, path).unwrap();
    }

    /// A rewrite in place refuses a source that another writer replaced, or
    /// changed in place, after it was read: renamed over it, the other
    /// writer's work was lost without an error.  The check is the last thing
    /// before the rename, and leaves no temporary file behind.
    #[test]
    fn s14_4_a_repack_refuses_a_source_that_changed() {
        for in_place in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("rw.h5");
            write_value(&path, 1).unwrap();
            before_rename(Some(Box::new(move |target: &Path| {
                if in_place {
                    open_rw(target).unwrap().dataset("v").unwrap().write(&[2]).unwrap();
                } else {
                    replaced_by_another(target, 2);
                }
            })));
            let rewritten = atomic_rewrite(&path, None, |src, dst| {
                let value = src.dataset("v")?.read_raw::<i32>()?;
                dst.new_dataset::<i32>().shape([1]).create("v")?.write(&value)?;
                Ok(())
            });
            before_rename(None);
            let err = rewritten.unwrap_err();
            assert!(matches!(err, Error::File(_)), "{err:?}");
            let shown = repr_str(&path.to_string_lossy());
            assert_eq!(
                err.to_string(),
                format!("{shown} changed after this rewrite read it; nothing was written: run it again")
            );
            assert_eq!(read_value(&path), 2, "in place: {in_place}");
            assert!(no_temporaries(dir.path()));
        }
    }

    /// An unchanged source is replaced as before.
    #[test]
    fn s14_4_a_rewrite_of_an_unchanged_source_commits() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rw.h5");
        write_value(&path, 1).unwrap();
        atomic_rewrite(&path, None, |src, dst| {
            let value = src.dataset("v")?.read_raw::<i32>()?;
            dst.new_dataset::<i32>().shape([1]).create("v")?.write(&[value[0] + 1])?;
            Ok(())
        })
        .unwrap();
        assert_eq!(read_value(&path), 2);
    }

    /// A link, or a file, already at a temporary's name is passed over for
    /// another name, never written through; and a staged file keeps its
    /// target's suffix, for a writer that picks a format by it.
    #[cfg(unix)]
    #[test]
    fn s14_4_a_link_at_the_next_temporary_name_is_passed_over() {
        let dir = tempfile::tempdir().unwrap();
        let other = dir.path().join("other.txt");
        fs::write(&other, b"someone else's").unwrap();
        let target = dir.path().join("image.nii.gz");
        let planted = dir.path().join(format!(".{}-00000000-image.nii.gz", std::process::id()));
        std::os::unix::fs::symlink(&other, &planted).unwrap();
        NEXT_UNIQUE.with(|queue| *queue.borrow_mut() = vec![1, 0]);
        let staged = StagedFile::create(&target).unwrap();
        NEXT_UNIQUE.with(|queue| queue.borrow_mut().clear());
        assert_eq!(staged.path(), dir.path().join(format!(".{}-00000001-image.nii.gz", std::process::id())));
        assert!(staged.path().to_string_lossy().ends_with(".nii.gz"));
        fs::write(staged.path(), b"the image").unwrap();
        staged.commit().unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"the image");
        assert_eq!(fs::read(&other).unwrap(), b"someone else's");
    }

    /// Every way the engine writes a file keeps a link at the target, and
    /// replaces the file it names, with that file's mode; a document's bytes
    /// are written as given.
    #[cfg(unix)]
    #[test]
    fn s14_4_every_replace_keeps_the_link_and_the_mode() {
        use std::os::unix::fs::PermissionsExt;
        type Write = fn(&Path) -> Result<()>;
        let writes: [(&str, Write); 3] = [
            ("write_atomic", |t| write_atomic(t, b"one\ntwo\n")),
            ("staged", |t| {
                let staged = StagedFile::create(t)?;
                fs::write(staged.path(), b"one\ntwo\n")?;
                staged.commit()
            }),
            ("hdf5", |t| write_value(t, 7)),
        ];
        for (name, write) in writes {
            let dir = tempfile::tempdir().unwrap();
            let file = dir.path().join("file");
            fs::write(&file, b"old").unwrap();
            fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
            let link = dir.path().join("link");
            std::os::unix::fs::symlink("file", &link).unwrap();
            write(&link).unwrap();
            assert!(fs::symlink_metadata(&link).unwrap().file_type().is_symlink(), "{name}");
            assert_eq!(fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600, "{name}");
            if name != "hdf5" {
                assert_eq!(fs::read(&file).unwrap(), b"one\ntwo\n", "{name}");
            } else {
                assert_eq!(read_value(&file), 7);
            }
            let mut left: Vec<String> = fs::read_dir(dir.path())
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            left.sort();
            assert_eq!(left, ["file", "link"], "{name}");
        }
    }

    /// A staged file dropped or aborted leaves its target as it was, and no
    /// temporary behind.
    #[test]
    fn s14_4_an_aborted_staged_file_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("r.json");
        fs::write(&target, b"before").unwrap();
        let staged = StagedFile::create(&target).unwrap();
        fs::write(staged.path(), b"after").unwrap();
        staged.abort();
        drop(StagedFile::create(&target).unwrap());
        assert_eq!(fs::read(&target).unwrap(), b"before");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
}
