//! Low-level operations: traversal, links, and the self-containment check.
//! Copying lives in [`super::graph`], which keeps the links a copy crosses.
//!
//! A MEDH5 file holds its own bytes (§2).  Three HDF5 features read bytes from
//! elsewhere --- a dataset whose raw data lives in *external storage*, a
//! *virtual dataset* mapping other files, and an *external link* --- and a tool
//! that follows one copies bytes it was never given: `recompress` of a crafted
//! file once wrote the contents of a local private key into its output.  Every
//! file is checked on open and refused if it carries any of them.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use crate::h5sys::{h5, h5a, h5d, h5i, h5l, h5o, h5p, h5t};
use crate::json::repr_str;
use crate::{Error, Result};

/// What kind of object a name in a group refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Group,
    Dataset,
    Other,
}

/// What a link is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    Hard,
    Soft,
    External,
    Other,
}

fn cstring(name: &str) -> Result<CString> {
    CString::new(name).map_err(|_| Error::Value(format!("name {} contains a NUL byte", repr_str(name))))
}

/// The kind of link `name` is in `group`, or `None` when it does not exist.
pub fn link_kind(group: &hdf5::Group, name: &str) -> Option<LinkKind> {
    phase!(Metadata);
    let cname = cstring(name).ok()?;
    super::locked(|| unsafe {
        if h5l::H5Lexists(group.id(), cname.as_ptr(), h5p::H5P_DEFAULT) <= 0 {
            return None;
        }
        let mut info: h5l::H5L_info2_t = std::mem::zeroed();
        if h5l::H5Lget_info2(group.id(), cname.as_ptr(), &mut info, h5p::H5P_DEFAULT) < 0 {
            return None;
        }
        Some(match raw_link_type(&info) {
            LINK_HARD => LinkKind::Hard,
            LINK_SOFT => LinkKind::Soft,
            LINK_EXTERNAL => LinkKind::External,
            _ => LinkKind::Other,
        })
    })
}

/// What `name` in `group` is, following soft links; `None` when it cannot be
/// resolved.
pub fn node_kind(group: &hdf5::Group, name: &str) -> Option<NodeKind> {
    phase!(Metadata);
    match link_kind(group, name)? {
        LinkKind::External => return Some(NodeKind::Other),
        LinkKind::Other => return Some(NodeKind::Other),
        _ => {}
    }
    match group.loc_type_by_name(name) {
        Ok(hdf5::LocationType::Group) => Some(NodeKind::Group),
        Ok(hdf5::LocationType::Dataset) => Some(NodeKind::Dataset),
        Ok(_) => Some(NodeKind::Other),
        Err(_) => None,
    }
}

/// Whether `name` exists in `group` as a link.
pub fn exists(group: &hdf5::Group, name: &str) -> bool {
    link_kind(group, name).is_some()
}

/// Whether `name` in `group` is a group.
pub fn is_group(group: &hdf5::Group, name: &str) -> bool {
    node_kind(group, name) == Some(NodeKind::Group)
}

/// Whether `name` in `group` is a dataset.
pub fn is_dataset(group: &hdf5::Group, name: &str) -> bool {
    node_kind(group, name) == Some(NodeKind::Dataset)
}

/// Create the group `name` in `parent`, untimed.
///
/// HDF5 stamps a group with the time it was created and changed unless told
/// not to, so two writes of the same content differ in bytes; a dataset is
/// already written untimed (`h5::data`).  The times are not format content
/// (no digest covers them), and without them a corpus built twice is the
/// same bytes.  Every group the engine creates goes through here.
pub fn new_group(parent: &hdf5::Group, name: &str) -> Result<hdf5::Group> {
    phase!(Metadata);
    Ok(parent.create_group_builder().obj_track_times(false).create(name)?)
}

/// The names in a group, sorted (HDF5's name order).
pub fn members(group: &hdf5::Group) -> Result<Vec<String>> {
    phase!(Metadata);
    super::alive(group)?;
    let mut names = group.member_names()?;
    names.sort();
    Ok(names)
}

/// A child group, if `name` is one.
pub fn child_group(group: &hdf5::Group, name: &str) -> Option<hdf5::Group> {
    if is_group(group, name) {
        group.group(name).ok()
    } else {
        None
    }
}

/// A child dataset, if `name` is one.
pub fn child_dataset(group: &hdf5::Group, name: &str) -> Option<hdf5::Dataset> {
    if is_dataset(group, name) {
        group.dataset(name).ok()
    } else {
        None
    }
}

/// The dataset at a `/`-separated path below `group`, if there is one.
pub fn resolve_dataset(group: &hdf5::Group, path: &str) -> Option<hdf5::Dataset> {
    let (parent, leaf) = match path.trim_matches('/').rsplit_once('/') {
        Some((parent, leaf)) => (resolve_group(group, parent)?, leaf),
        None => (group.clone(), path.trim_matches('/')),
    };
    child_dataset(&parent, leaf)
}

/// The group at a `/`-separated path below `group`, if there is one.
fn resolve_group(group: &hdf5::Group, path: &str) -> Option<hdf5::Group> {
    path.split('/').filter(|p| !p.is_empty()).try_fold(group.clone(), |at, name| child_group(&at, name))
}

/// One object reached by [`visit`].
pub enum Node {
    Group(hdf5::Group),
    Dataset(hdf5::Dataset),
}

/// An object's identity within its file: two paths to one object share it.
pub type ObjectId = (std::os::raw::c_ulong, [u8; 16]);

// `H5O_token_t` is `repr(C)` around one `[u8; H5O_MAX_TOKEN_SIZE]`.
const _: () = assert!(std::mem::size_of::<h5o::H5O_token_t>() == 16);

fn object_id(info: &h5o::H5O_info2_t) -> ObjectId {
    // SAFETY: a plain-bytes struct of exactly 16 bytes (asserted above).
    let token = unsafe { std::mem::transmute::<h5o::H5O_token_t, [u8; 16]>(info.token) };
    (info.fileno, token)
}

/// The identity of the object `name` in `group` names, without opening it.
pub fn object_id_by_name(group: &hdf5::Group, name: &str) -> Option<ObjectId> {
    let cname = cstring(name).ok()?;
    super::locked(|| unsafe {
        let mut info: h5o::H5O_info2_t = std::mem::zeroed();
        if h5o::H5Oget_info_by_name3(group.id(), cname.as_ptr(), &mut info, h5o::H5O_INFO_BASIC, h5p::H5P_DEFAULT) < 0 {
            return None;
        }
        Some(object_id(&info))
    })
}

/// The identity of an open object.
pub fn object_id_of(obj: &hdf5::Location) -> Option<ObjectId> {
    super::locked(|| unsafe {
        let mut info: h5o::H5O_info2_t = std::mem::zeroed();
        if h5o::H5Oget_info3(obj.id(), &mut info, h5o::H5O_INFO_BASIC) < 0 {
            return None;
        }
        Some(object_id(&info))
    })
}

/// How deep [`visit`] goes.  A sample is a handful of levels deep; a group
/// nested past this is refused rather than followed, so no file can exhaust
/// the stack, whatever its links.
pub const MAX_DEPTH: usize = 64;

/// Recursively visit every object under `group` through hard links, depth
/// first in name order --- h5py's `visititems`.  Paths are relative to
/// `group`.  `f` returns `false` to stop.
///
/// Each object is visited **once**, at the first path that reaches it, as
/// `H5Ovisit` does: a hard link back to an ancestor is a cycle, not an
/// infinite tree, and a second link to an object is an alias of it, not a
/// second object (1.x digested an alias once, under its first path).
pub fn visit(group: &hdf5::Group, f: &mut dyn FnMut(&str, &Node) -> Result<bool>) -> Result<()> {
    visit_links(group, false, f)
}

/// [`visit`], following soft links as well: what a reader reaches by name
/// under `group`, each object once, at the first path that reaches it.  A
/// soft link that resolves to nothing is passed by.
///
/// Readers follow a soft link as HDF5 does, so a column of `clinical/` linked
/// softly to storage under `index/` was read by every selection while the
/// walk of hard links that decides what a pin covers never saw it.
/// Attestation walks what readers read.
pub fn visit_resolving(group: &hdf5::Group, f: &mut dyn FnMut(&str, &Node) -> Result<bool>) -> Result<()> {
    visit_links(group, true, f)
}

fn visit_links(group: &hdf5::Group, soft: bool, f: &mut dyn FnMut(&str, &Node) -> Result<bool>) -> Result<()> {
    phase!(Metadata);
    fn walk(
        group: &hdf5::Group,
        prefix: &str,
        depth: usize,
        soft: bool,
        seen: &mut HashSet<ObjectId>,
        f: &mut dyn FnMut(&str, &Node) -> Result<bool>,
    ) -> Result<bool> {
        if depth > MAX_DEPTH {
            return Err(Error::File(format!(
                "{} is nested more than {MAX_DEPTH} groups deep; it is refused rather than followed",
                repr_str(&group.name())
            )));
        }
        for name in members(group)? {
            let kind = link_kind(group, &name);
            if kind != Some(LinkKind::Hard) && !(soft && kind == Some(LinkKind::Soft)) {
                continue;
            }
            match object_id_by_name(group, &name) {
                Some(id) if !seen.insert(id) => continue,
                // A soft link to nothing names no object.
                None if kind == Some(LinkKind::Soft) => continue,
                _ => {}
            }
            let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
            match node_kind(group, &name) {
                Some(NodeKind::Group) => {
                    let child = group.group(&name)?;
                    if !f(&path, &Node::Group(child.clone()))? {
                        return Ok(false);
                    }
                    if !walk(&child, &path, depth + 1, soft, seen, f)? {
                        return Ok(false);
                    }
                }
                Some(NodeKind::Dataset) => {
                    let ds = group.dataset(&name)?;
                    if !f(&path, &Node::Dataset(ds))? {
                        return Ok(false);
                    }
                }
                _ => {}
            }
        }
        Ok(true)
    }
    // The starting group itself: a link back to it is a cycle too.
    let mut seen: HashSet<ObjectId> = object_id_of(group).into_iter().collect();
    walk(group, "", 0, soft, &mut seen, f)?;
    Ok(())
}

/// Every dataset under `group`, with its path relative to `group`.
pub fn datasets(group: &hdf5::Group) -> Result<Vec<(String, hdf5::Dataset)>> {
    let mut out = Vec::new();
    visit(group, &mut |path, node| {
        if let Node::Dataset(ds) = node {
            out.push((path.to_string(), ds.clone()));
        }
        Ok(true)
    })?;
    Ok(out)
}

/// Every dataset a reader reaches by name under `group`, soft links
/// followed ([`visit_resolving`]), with its path relative to `group`.
pub fn datasets_resolving(group: &hdf5::Group) -> Result<Vec<(String, hdf5::Dataset)>> {
    let mut out = Vec::new();
    visit_resolving(group, &mut |path, node| {
        if let Node::Dataset(ds) = node {
            out.push((path.to_string(), ds.clone()));
        }
        Ok(true)
    })?;
    Ok(out)
}

/// The path a soft link holds, as stored --- not what it resolves to.
pub fn soft_link_target(group: &hdf5::Group, name: &str) -> Result<String> {
    let cname = cstring(name)?;
    super::locked(|| unsafe {
        let mut info: h5l::H5L_info2_t = std::mem::zeroed();
        if h5l::H5Lget_info2(group.id(), cname.as_ptr(), &mut info, h5p::H5P_DEFAULT) < 0
            || !matches!(info.type_, h5l::H5L_type_t::H5L_TYPE_SOFT)
        {
            return Err(Error::Value(format!("{} is not a soft link", repr_str(name))));
        }
        let size = *info.u.val_size();
        let mut buf = vec![0u8; size.max(1)];
        if h5l::H5Lget_val(group.id(), cname.as_ptr(), buf.as_mut_ptr().cast(), buf.len(), h5p::H5P_DEFAULT) < 0 {
            return Err(Error::Io(format!("could not read the soft link {}", repr_str(name))));
        }
        let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
    })
}

/// Delete a link if present.
pub fn unlink(group: &hdf5::Group, name: &str) -> Result<()> {
    phase!(Metadata);
    if exists(group, name) {
        group.unlink(name)?;
    }
    Ok(())
}

/// The text of the most recent HDF5 error on this thread: the failed call
/// and what HDF5 says went wrong, as every other HDF5 error reads.  Taken
/// right after the call that failed, since the next call clears it.
pub fn hdf5_error_text() -> String {
    match hdf5::Error::query() {
        Ok(error) => error.to_string(),
        Err(_) => "HDF5 gave no reason".to_string(),
    }
}

// -- self-containment ---------------------------------------------------------

/// A link's type as HDF5 stores it: `H5L_type_t` names the built-in types
/// only, and a file may carry a user-defined one (65 to 255), which is no
/// value of that enum.
type RawLinkType = std::os::raw::c_int;

const LINK_HARD: RawLinkType = h5l::H5L_type_t::H5L_TYPE_HARD as RawLinkType;
const LINK_SOFT: RawLinkType = h5l::H5L_type_t::H5L_TYPE_SOFT as RawLinkType;
const LINK_EXTERNAL: RawLinkType = h5l::H5L_type_t::H5L_TYPE_EXTERNAL as RawLinkType;

/// The type of the link `info` describes.
///
/// # Safety
///
/// `info` points at a link description HDF5 filled in.
unsafe fn raw_link_type(info: *const h5l::H5L_info2_t) -> RawLinkType {
    // SAFETY: read as the C `int` it is, never as the Rust enum.
    unsafe { std::ptr::addr_of!((*info).type_).cast::<RawLinkType>().read() }
}

/// Every link of one group, by name, with its type: what
/// [`outside_references`] lists of each group it walks.
unsafe extern "C" fn collect_link(
    _group: h5i::hid_t,
    name: *const c_char,
    info: *const h5l::H5L_info2_t,
    data: *mut c_void,
) -> h5::herr_t {
    // SAFETY: HDF5 passes valid pointers for the duration of the callback,
    // and `data` is the list `links_of` handed it.
    unsafe {
        let links = &mut *(data as *mut Vec<(CString, RawLinkType)>);
        links.push((CStr::from_ptr(name).to_owned(), raw_link_type(info)));
    }
    0
}

/// The links of the open group `group`, in name order; `None` when HDF5
/// cannot list them.  The caller holds the HDF5 lock.
fn links_of(group: h5i::hid_t) -> Option<Vec<(CString, RawLinkType)>> {
    let mut links: Vec<(CString, RawLinkType)> = Vec::new();
    // SAFETY: a live group id, and a callback that only appends to `links`.
    let status = unsafe {
        h5l::H5Literate2(
            group,
            crate::h5sys::h5::H5_index_t::H5_INDEX_NAME,
            crate::h5sys::h5::H5_iter_order_t::H5_ITER_INC,
            std::ptr::null_mut(),
            Some(collect_link),
            (&mut links as *mut Vec<(CString, RawLinkType)>).cast(),
        )
    };
    (status >= 0).then_some(links)
}

/// What reads a dataset's bytes from outside the file, if anything does.  The
/// caller holds the HDF5 lock.
fn reads_outside(parent: h5i::hid_t, name: &CStr) -> Option<&'static str> {
    // SAFETY: a live group id and a NUL-terminated name; every id opened is
    // closed.
    unsafe {
        let ds = h5d::H5Dopen2(parent, name.as_ptr(), h5p::H5P_DEFAULT);
        if ds < 0 {
            // An object whose header cannot be read cannot have its data
            // read either, so it cannot leak anything.
            return None;
        }
        let mut found = None;
        let plist = h5d::H5Dget_create_plist(ds);
        if plist >= 0 {
            if h5p::H5Pget_layout(plist) == h5d::H5D_layout_t::H5D_VIRTUAL {
                found = Some("a virtual dataset");
            } else if h5p::H5Pget_external_count(plist) > 0 {
                found = Some("external raw-data storage");
            }
            h5p::H5Pclose(plist);
        }
        h5d::H5Dclose(ds);
        found
    }
}

/// One group of [`outside_references`]' walk: open, with its links listed and
/// the next one to look at.  Dropping it closes the group.
struct Frame {
    group: h5i::hid_t,
    path: String,
    links: Vec<(CString, RawLinkType)>,
    next: usize,
}

impl Drop for Frame {
    fn drop(&mut self) {
        // SAFETY: the frame owns the group id it was opened with.
        super::locked(|| unsafe { crate::h5sys::h5g::H5Gclose(self.group) });
    }
}

/// Whether a datatype is, or holds, an HDF5 reference: an object or region
/// reference, alone or as a member of a compound, array or variable-length
/// type.  The caller holds the HDF5 lock.
fn holds_reference(tid: h5i::hid_t) -> bool {
    use h5t::H5T_class_t as Class;
    unsafe {
        match h5t::H5Tget_class(tid) {
            Class::H5T_REFERENCE => true,
            Class::H5T_COMPOUND => {
                let members = h5t::H5Tget_nmembers(tid).max(0) as u32;
                (0..members).any(|i| {
                    let member = h5t::H5Tget_member_type(tid, i);
                    if member < 0 {
                        return false;
                    }
                    let found = holds_reference(member);
                    h5t::H5Tclose(member);
                    found
                })
            }
            Class::H5T_ARRAY | Class::H5T_VLEN => {
                let base = h5t::H5Tget_super(tid);
                if base < 0 {
                    return false;
                }
                let found = holds_reference(base);
                h5t::H5Tclose(base);
                found
            }
            _ => false,
        }
    }
}

/// Every dataset and attribute under `root` whose type holds an HDF5
/// reference: `/path` for a dataset, `/path@name` for an attribute.
///
/// Attributes are read on every kind of object that carries them: groups,
/// datasets and committed (named) datatypes.  [`visit`] walks groups and
/// datasets, so a reference in an attribute of a committed datatype was
/// missed, and a rewrite that copies the type nulled it.  A committed
/// datatype need not keep a name: one whose link was
/// removed lives on as the type of the datasets and attributes that use it,
/// so each dataset's and attribute's type is opened too when it is a
/// committed one --- `/path (type)@name` --- each type once, by identity.
pub fn reference_carriers(root: &hdf5::Group) -> Result<Vec<String>> {
    /// The committed datatype `tid` is, opened as an object; `None` when it is
    /// transient.  Takes ownership of `tid`.
    fn committed(tid: h5i::hid_t) -> Option<hdf5::CommittedDatatype> {
        if tid < 0 {
            return None;
        }
        // SAFETY: `tid` is a type id HDF5 just handed out, owned here: it is
        // either wrapped (and closed when the wrapper drops) or closed.
        let is_committed = super::locked(|| unsafe { h5t::H5Tcommitted(tid) }) > 0;
        if !is_committed {
            super::locked(|| unsafe { h5t::H5Tclose(tid) });
            return None;
        }
        unsafe { hdf5::from_id::<hdf5::CommittedDatatype>(tid) }.ok()
    }
    /// The attributes of the committed datatype `tid` is, when it is one not
    /// inspected yet.
    fn type_of(tid: h5i::hid_t, path: &str, types: &mut HashSet<ObjectId>, found: &mut Vec<String>) -> Result<()> {
        let Some(named) = committed(tid) else { return Ok(()) };
        if object_id_of(&named).is_some_and(|id| !types.insert(id)) {
            return Ok(());
        }
        attributes(&named, &format!("{path} (type)"), types, found)
    }
    fn attributes(
        obj: &hdf5::Location,
        path: &str,
        types: &mut HashSet<ObjectId>,
        found: &mut Vec<String>,
    ) -> Result<()> {
        for name in obj.attr_names()? {
            let attr = obj.attr(&name)?;
            let dtype = attr.dtype()?;
            if super::locked(|| holds_reference(dtype.id())) {
                found.push(format!("{path}@{name}"));
            }
            // SAFETY: a live attribute; the type id returned is owned by `type_of`.
            let tid = super::locked(|| unsafe { h5a::H5Aget_type(attr.id()) });
            type_of(tid, &format!("{path}@{name}"), types, found)?;
        }
        Ok(())
    }
    // The committed datatypes a group links to, each once.
    fn named_types(
        group: &hdf5::Group,
        prefix: &str,
        types: &mut HashSet<ObjectId>,
        found: &mut Vec<String>,
    ) -> Result<()> {
        for name in members(group)? {
            if link_kind(group, &name) != Some(LinkKind::Hard)
                || group.loc_type_by_name(&name).ok() != Some(hdf5::LocationType::NamedDatatype)
            {
                continue;
            }
            if object_id_by_name(group, &name).is_some_and(|id| !types.insert(id)) {
                continue;
            }
            let path = if prefix.is_empty() { format!("/{name}") } else { format!("/{prefix}/{name}") };
            let named = group.committed_datatype(&name)?;
            attributes(&named, &path, types, found)?;
        }
        Ok(())
    }
    let mut found = Vec::new();
    let mut types = HashSet::new();
    attributes(root, "/", &mut types, &mut found)?;
    named_types(root, "", &mut types, &mut found)?;
    visit(root, &mut |name, node| {
        let path = format!("/{name}");
        match node {
            Node::Dataset(ds) => {
                let dtype = ds.dtype()?;
                if super::locked(|| holds_reference(dtype.id())) {
                    found.push(path.clone());
                }
                attributes(ds, &path, &mut types, &mut found)?;
                // SAFETY: a live dataset; the type id returned is owned by `type_of`.
                let tid = super::locked(|| unsafe { h5d::H5Dget_type(ds.id()) });
                type_of(tid, &path, &mut types, &mut found)?;
            }
            Node::Group(g) => {
                attributes(g, &path, &mut types, &mut found)?;
                named_types(g, name, &mut types, &mut found)?;
            }
        }
        Ok(true)
    })?;
    Ok(found)
}

/// Refuse to copy a file that holds HDF5 references into a new one.
///
/// A reference is an address in the file that holds it.  MEDH5 stores none,
/// so any is another tool's content, which amend, recompress, pack and repack
/// carry over without understanding: copied, an object reference pointed
/// wherever its address happened to land in the new file, and a no-op amend
/// nulled the ones in an extension group --- while every digest still
/// verified.
pub fn refuse_references(root: &hdf5::Group, action: &str) -> Result<()> {
    let found = reference_carriers(root)?;
    if found.is_empty() {
        return Ok(());
    }
    let named = found.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
    let more = if found.len() > 5 { format!(" (and {} more)", found.len() - 5) } else { String::new() };
    Err(Error::File(format!(
        "{} holds HDF5 references ({named}{more}). A reference is an address in the file that holds it, and \
         {action} writes a new file, where it would point at whatever lands at that address; MEDH5 stores no \
         references and does not rewrite another tool's, so the file is refused --- remove them, or rewrite it \
         with the tool that wrote them",
        repr_str(&root.filename())
    )))
}

/// Objects that read bytes from outside the file: `(path, what)` pairs.
///
/// The file's own walk, not HDF5's `H5Lvisit`, which recurses without limit
/// and resolved each link from the root: a file of 30,000 nested groups
/// overflowed the stack on open (SIGSEGV), and 10,000 took a minute and
/// 326 MB.  This one keeps its own stack, lists each group's links once
/// (`H5Literate2`) and looks each object up by name in its own group.  It goes
/// as [`visit`] goes --- hard links, depth first in name order, each object
/// once --- and refuses a group past [`MAX_DEPTH`] as `visit` does, so a file
/// this check passes is one `visit` can walk.
pub fn outside_references(handle: &hdf5::File) -> Result<Vec<(String, String)>> {
    let unwalkable = || {
        Error::File(format!("{} could not be walked to check that it is self-contained", repr_str(&handle.filename())))
    };
    super::locked(|| {
        let root = cstring("/")?;
        // SAFETY: a live file id; the group id is owned by its frame.
        let group = unsafe { crate::h5sys::h5g::H5Gopen2(handle.id(), root.as_ptr(), h5p::H5P_DEFAULT) };
        if group < 0 {
            return Err(unwalkable());
        }
        let mut frame = Frame { group, path: String::new(), links: Vec::new(), next: 0 };
        frame.links = links_of(group).ok_or_else(unwalkable)?;
        // The root itself: a link back to it is a cycle too.
        let mut seen: HashSet<ObjectId> = HashSet::new();
        let mut info: h5o::H5O_info2_t = unsafe { std::mem::zeroed() };
        // SAFETY: a live group id and a zeroed struct HDF5 fills in.
        if unsafe { h5o::H5Oget_info3(group, &mut info, h5o::H5O_INFO_BASIC) } >= 0 {
            seen.insert(object_id(&info));
        }
        let mut stack = vec![frame];
        let mut found = Vec::new();
        while let Some(top) = stack.last_mut() {
            let Some(link) = top.links.get_mut(top.next) else {
                stack.pop();
                continue;
            };
            let (name, kind) = std::mem::take(link);
            top.next += 1;
            let parent = top.group;
            let shown = name.to_string_lossy();
            let path = if top.path.is_empty() { shown.into_owned() } else { format!("{}/{shown}", top.path) };
            if kind == LINK_EXTERNAL {
                found.push((path, "an external link".to_string()));
                continue;
            }
            if kind != LINK_HARD {
                continue;
            }
            // SAFETY: a live group id, a NUL-terminated name, and a struct
            // HDF5 fills in.
            if unsafe {
                h5o::H5Oget_info_by_name3(parent, name.as_ptr(), &mut info, h5o::H5O_INFO_BASIC, h5p::H5P_DEFAULT)
            } < 0
            {
                // As `H5Lvisit` did: an object that cannot be looked up
                // stops the walk.
                return Err(unwalkable());
            }
            if !seen.insert(object_id(&info)) {
                continue;
            }
            match info.type_ {
                h5o::H5O_type_t::H5O_TYPE_DATASET => {
                    if let Some(what) = reads_outside(parent, &name) {
                        found.push((path, what.to_string()));
                    }
                }
                h5o::H5O_type_t::H5O_TYPE_GROUP => {
                    // The root's frame is at depth 0, so this group is at the
                    // depth the stack is long.
                    if stack.len() > MAX_DEPTH {
                        return Err(Error::File(format!(
                            "{} in {} is nested more than {MAX_DEPTH} groups deep; it is refused rather than \
                             followed",
                            repr_str(&format!("/{path}")),
                            repr_str(&handle.filename())
                        )));
                    }
                    // SAFETY: as above; the id is owned by its frame.
                    let group = unsafe { crate::h5sys::h5g::H5Gopen2(parent, name.as_ptr(), h5p::H5P_DEFAULT) };
                    if group < 0 {
                        return Err(unwalkable());
                    }
                    let mut frame = Frame { group, path, links: Vec::new(), next: 0 };
                    frame.links = links_of(group).ok_or_else(unwalkable)?;
                    stack.push(frame);
                }
                _ => {}
            }
        }
        Ok(found)
    })
}

/// The identity of a file a handle had open: `(volume, file id, size,
/// modification time, change time)` --- on POSIX `(st_dev, st_ino, st_size,
/// st_mtime_ns, st_ctime_ns)`, on Windows the volume serial number, the
/// file's id, its end of file, and its last-write and change times (in
/// 100-nanosecond ticks).
pub(crate) type FileIdentity = (u64, u128, u64, i128, i128);

/// The files checked already.  A panic while it is held leaves the map whole
/// (an insert or a clear), so a poisoned lock is taken as it stands.
pub(super) fn checked_files() -> &'static Mutex<HashMap<FileIdentity, ()>> {
    static CHECKED: std::sync::OnceLock<Mutex<HashMap<FileIdentity, ()>>> = std::sync::OnceLock::new();
    CHECKED.get_or_init(|| Mutex::new(HashMap::new()))
}

const CHECKED_LIMIT: usize = 65_536;

#[cfg(unix)]
pub(crate) fn identity(meta: &std::fs::Metadata) -> Option<FileIdentity> {
    use std::os::unix::fs::MetadataExt;
    Some((
        meta.dev(),
        u128::from(meta.ino()),
        meta.size(),
        meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128,
        meta.ctime() as i128 * 1_000_000_000 + meta.ctime_nsec() as i128,
    ))
}

/// The identity of the file a Windows handle has open.
///
/// The volume serial number and 128-bit file id come from `FileIdInfo`;
/// where a file system refuses it (FAT, some SMB shares), from
/// `GetFileInformationByHandle`'s 32-bit serial and 64-bit file index, so a
/// file is keyed one way for as long as its volume is.
///
/// What is left: a program can set a file's change time on Windows, which
/// POSIX's ctime cannot be, so an edit in place that restores the size and
/// both times, within this process's lifetime, skips the re-check.  That is
/// a deliberate forgery by a process that could write the file anyway, and
/// still more than 1.x's file index alone told.
#[cfg(windows)]
fn windows_identity(handle: windows_sys::Win32::Foundation::HANDLE) -> Option<FileIdentity> {
    use std::mem::{size_of, MaybeUninit};
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::FileSystem::{
        FileBasicInfo, FileIdInfo, FileStandardInfo, GetFileInformationByHandle, GetFileInformationByHandleEx,
        BY_HANDLE_FILE_INFORMATION, FILE_BASIC_INFO, FILE_ID_INFO, FILE_INFO_BY_HANDLE_CLASS, FILE_STANDARD_INFO,
    };

    /// One class of information about the file, or `None` when refused.
    ///
    /// # Safety
    /// `handle` is open, and `T` is the structure `class` fills.
    unsafe fn info<T>(handle: HANDLE, class: FILE_INFO_BY_HANDLE_CLASS) -> Option<T> {
        let mut out = MaybeUninit::<T>::zeroed();
        // SAFETY: as the caller promises; the buffer is a `T`, of its size.
        let done =
            unsafe { GetFileInformationByHandleEx(handle, class, out.as_mut_ptr().cast(), size_of::<T>() as u32) };
        // SAFETY: the call succeeded, so it filled the structure.
        (done != 0).then(|| unsafe { out.assume_init() })
    }

    // SAFETY: `handle` is the open handle behind HDF5's descriptor, and each
    // structure is the one its class fills.
    let basic: FILE_BASIC_INFO = unsafe { info(handle, FileBasicInfo)? };
    // SAFETY: as above.
    let standard: FILE_STANDARD_INFO = unsafe { info(handle, FileStandardInfo)? };
    // SAFETY: as above.
    let (volume, file) = match unsafe { info::<FILE_ID_INFO>(handle, FileIdInfo) } {
        Some(id) => (id.VolumeSerialNumber, u128::from_le_bytes(id.FileId.Identifier)),
        None => {
            let mut by = MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::zeroed();
            // SAFETY: as above; the call fills the structure it is given.
            if unsafe { GetFileInformationByHandle(handle, by.as_mut_ptr()) } == 0 {
                return None;
            }
            // SAFETY: the call succeeded, so it filled the structure.
            let by = unsafe { by.assume_init() };
            let index = u128::from(by.nFileIndexHigh) << 32 | u128::from(by.nFileIndexLow);
            (u64::from(by.dwVolumeSerialNumber), index)
        }
    };
    Some((
        volume,
        file,
        u64::try_from(standard.EndOfFile).ok()?,
        i128::from(basic.LastWriteTime),
        i128::from(basic.ChangeTime),
    ))
}

/// Whether no other handle may write the file `path` names, and it is still
/// the file `id` identifies.
///
/// Windows does not promise a file's times are current until every handle
/// that wrote it has closed, so a file rewritten to its old length through a
/// handle still open could keep its old identity.  Opening it for reading
/// while refusing to share writing fails while any handle may write it; a
/// file that is being written is checked, not remembered.
#[cfg(windows)]
fn unshared(path: &Path, id: &FileIdentity) -> bool {
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ;
    let Ok(probe) = std::fs::OpenOptions::new().read(true).share_mode(FILE_SHARE_READ).open(path) else {
        return false;
    };
    windows_identity(probe.as_raw_handle()).is_some_and(|p| (p.0, p.1) == (id.0, id.1))
}

/// The identity of the file `handle` holds open --- not whatever `path` names
/// now.
///
/// Between an open and a stat by name, another process can atomically replace
/// the path; recording the replacement as checked would let it skip the check
/// on its next open, and the next `recompress` would copy whatever it points
/// at.  So the identity comes from the descriptor HDF5 read through: the
/// file's volume and id, its size, and its modification and change times.
///
/// Size and modification time alone do not tell two files apart --- an
/// archive extracts many with one mtime, and the time is the author's to set
/// --- so keying on them would let a crafted file of a checked file's length
/// and time skip the check.  A file with no identity (another driver, another
/// system, and on Windows a file another handle may be writing) is checked on
/// every open rather than trusted on a guess.
pub(crate) fn opened_identity(handle: &hdf5::File, path: Option<&Path>) -> Option<FileIdentity> {
    #[cfg(not(windows))]
    let _ = path;
    let fd = super::file::descriptor(handle)?;
    #[cfg(unix)]
    {
        use std::os::unix::io::FromRawFd;
        // Borrowed for one `fstat`: HDF5 owns the descriptor and closes it.
        // SAFETY: `fd` is HDF5's open descriptor, which outlives the call.
        let file = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(fd) });
        identity(&file.metadata().ok()?)
    }
    #[cfg(windows)]
    {
        let id = windows_identity(super::file::os_handle(fd)?)?;
        path.is_some_and(|p| unshared(p, &id)).then_some(id)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = fd;
        None
    }
}

/// Refuse a file that reads bytes from outside itself (`MEDH5FileError`).
///
/// Memoised per file identity, so re-opening an unchanged file costs one
/// `stat` --- a training loop re-opens the same few thousand files for a run.
pub fn check_self_contained(handle: &hdf5::File, path: Option<&Path>) -> Result<()> {
    let key = opened_identity(handle, path);
    if let Some(k) = key {
        if checked_files().lock().unwrap_or_else(PoisonError::into_inner).contains_key(&k) {
            return Ok(());
        }
    }
    let found = outside_references(handle)?;
    if !found.is_empty() {
        let named: Vec<String> = found.iter().take(5).map(|(n, w)| format!("/{n} is {w}")).collect();
        let more = if found.len() > 5 { format!(" (and {} more)", found.len() - 5) } else { String::new() };
        let where_ = path.map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| handle.filename());
        return Err(Error::File(format!(
            "{} is not self-contained: {}{more}. A MEDH5 file holds its own bytes (§2); following \
             these would read files on this machine that the file's author chose, so the file is refused",
            repr_str(&where_),
            named.join("; ")
        )));
    }
    if let Some(k) = key {
        let mut map = checked_files().lock().unwrap_or_else(PoisonError::into_inner);
        if map.len() >= CHECKED_LIMIT {
            map.clear();
        }
        map.insert(k, ());
    }
    Ok(())
}

/// Read one stored (still compressed) chunk by its logical offset.
///
/// Returns the filter mask and the bytes exactly as stored.  `offset` has one
/// coordinate per axis: HDF5 reads that many from it, whatever its length.
///
/// The chunk's size is the file's to state, up to 0xFFFFFFFF: one larger than
/// the file holding it is refused before anything is reserved, and the rest
/// is reserved fallibly --- a buffer sized by the stated size alone aborted
/// the process on a crafted 140 KB file.
pub fn read_raw_chunk(ds: &hdf5::Dataset, offset: &[u64]) -> Result<(u32, Vec<u8>)> {
    phase!(Read);
    super::alive(ds)?;
    if offset.len() != ds.ndim() {
        return Err(Error::Value(format!(
            "a chunk offset names one coordinate per axis: {} has {} axes, the offset {offset:?} has {}",
            ds.name(),
            ds.ndim(),
            offset.len()
        )));
    }
    if ds.chunk().is_none() {
        return Err(Error::Value(format!("{} is not chunked, so it stores no chunks", ds.name())));
    }
    let file = ds.file()?;
    super::locked(|| unsafe {
        let mut mask: u32 = 0;
        let mut addr: u64 = 0;
        let mut nbytes: u64 = 0;
        if h5d::H5Dget_chunk_info_by_coord(ds.id(), offset.as_ptr(), &mut mask, &mut addr, &mut nbytes) < 0 {
            return Err(Error::Io(format!("no stored chunk at {offset:?} in {}", ds.name())));
        }
        // Measured once the chunk is in the file: asking for it writes it
        // out of HDF5's cache, in a file still being written.
        let file_bytes = file.size();
        let mut buf = Vec::new();
        let reserved = usize::try_from(nbytes)
            .ok()
            .filter(|n| nbytes <= file_bytes && super::data::holdable(&[*n], 1) && buf.try_reserve_exact(*n).is_ok());
        let Some(nbytes) = reserved else {
            return Err(Error::Io(format!(
                "the chunk at {offset:?} in {} declares {nbytes} stored bytes, more than the file's {file_bytes}",
                ds.name()
            )));
        };
        buf.resize(nbytes, 0);
        let mut filter_mask: u32 = 0;
        let mut size = nbytes;
        if h5d::H5Dread_chunk(
            ds.id(),
            h5p::H5P_DEFAULT,
            offset.as_ptr(),
            &mut filter_mask,
            buf.as_mut_ptr().cast(),
            &mut size,
        ) < 0
        {
            return Err(Error::Io(format!("could not read the chunk at {offset:?} in {}", ds.name())));
        }
        buf.truncate(size);
        Ok((filter_mask, buf))
    })
}

/// What [`stored_chunks`]' walk collects: each chunk's logical offset.
struct ChunkOffsets {
    rank: usize,
    found: Vec<Vec<u64>>,
}

unsafe extern "C" fn chunk_offset(
    offset: *const u64,
    _filter_mask: std::os::raw::c_uint,
    _addr: u64,
    _size: u64,
    data: *mut c_void,
) -> std::os::raw::c_int {
    // SAFETY: HDF5 passes the dataset's rank of coordinates, valid for the
    // duration of the callback, and the state handed to `H5Dchunk_iter`.
    unsafe {
        let state = &mut *(data as *mut ChunkOffsets);
        state.found.push(std::slice::from_raw_parts(offset, state.rank).to_vec());
    }
    0
}

/// The logical offset of every chunk actually stored for a chunked dataset,
/// in the order the chunk index holds them.
///
/// One walk of the index (`H5Dchunk_iter`): asking for the chunks one by one
/// by ordinal walks the index from its start each time, so 10 000 chunks took
/// 13 seconds.
pub fn stored_chunks(ds: &hdf5::Dataset) -> Result<Vec<Vec<u64>>> {
    phase!(Metadata);
    super::alive(ds)?;
    let mut state = ChunkOffsets { rank: ds.ndim(), found: Vec::new() };
    let status = super::locked(|| unsafe {
        h5d::H5Dchunk_iter(ds.id(), h5p::H5P_DEFAULT, Some(chunk_offset), (&mut state as *mut ChunkOffsets).cast())
    });
    if status < 0 {
        return Err(Error::Io(format!("could not list the chunks of {}", ds.name())));
    }
    Ok(state.found)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(path: &Path) {
        let file = crate::h5::file::create_truncate(path).unwrap();
        file.new_dataset::<u8>().shape([4]).create("data").unwrap().write(&[1u8, 2, 3, 4]).unwrap();
    }

    /// A file carrying an external link: one of the three ways a file reads
    /// bytes from outside itself.
    fn linked(path: &Path, other: &Path) {
        plain(path);
        let file = crate::h5::file::marked(|| hdf5::File::open_rw(path)).unwrap();
        file.link_external(&other.to_string_lossy(), "/data", "x_link").unwrap();
    }

    /// A path replaced between the open and the check must not inherit it.
    ///
    /// The memo was keyed by a stat of the *path*, taken after the open: a
    /// replacement landing in between was recorded as checked while the handle
    /// scanned the old file, and its next open skipped the check.
    #[cfg(unix)]
    #[test]
    fn s2_the_check_is_remembered_for_the_file_it_read() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.medh5");
        let bad = dir.path().join("bad.medh5");
        plain(&target);
        linked(&bad, &target);
        let handle = crate::h5::file::marked(|| hdf5::File::open(&target)).unwrap();
        std::fs::rename(&bad, &target).unwrap(); // the path now names another file
        check_self_contained(&handle, Some(&target)).unwrap(); // the file read is clean
        drop(handle);
        let err = crate::h5::file::open_read(&target).unwrap_err();
        assert!(err.to_string().contains("not self-contained"), "{err}");
        assert!(err.to_string().contains("external link"), "{err}");
    }

    /// A file whose groups are nested `depth` deep below the root:
    /// `x_deep/a/a/...`.
    fn nested(path: &Path, depth: usize) {
        let file = crate::h5::file::create_truncate(path).unwrap();
        let mut group = file.create_group("x_deep").unwrap();
        for _ in 1..depth {
            group = group.create_group("a").unwrap();
        }
    }

    /// The check on open is a walk of medh5's own, refused past
    /// [`MAX_DEPTH`] as [`visit`] refuses it: HDF5's `H5Lvisit` recursed
    /// without limit and resolved every link from the root, so 30,000 nested
    /// groups overflowed the stack (SIGSEGV) and 10,000 took a minute.
    #[test]
    fn s2_a_chain_of_groups_past_the_depth_limit_is_refused_quickly() {
        let dir = tempfile::tempdir().unwrap();
        let deep = dir.path().join("deep.medh5");
        nested(&deep, 20_000);
        let started = std::time::Instant::now();
        let err = crate::h5::file::open_read(&deep).unwrap_err();
        let elapsed = started.elapsed();
        assert!(matches!(err, Error::File(_)), "{err:?}");
        assert!(err.message().contains("nested more than 64 groups deep"), "{err}");
        assert!(elapsed < std::time::Duration::from_secs(5), "{elapsed:?} to refuse it");
    }

    /// Where the check refuses a file is where [`visit`] refuses to walk
    /// it, so a file that opens is one every walk of it can finish.
    #[test]
    fn s2_the_check_and_the_walk_refuse_the_same_depth() {
        let dir = tempfile::tempdir().unwrap();
        for (depth, opens) in [(MAX_DEPTH, true), (MAX_DEPTH + 1, false)] {
            let path = dir.path().join(format!("d{depth}.medh5"));
            nested(&path, depth);
            let raw = crate::h5::file::marked(|| hdf5::File::open(&path)).unwrap();
            let walked = visit(&raw.as_group().unwrap(), &mut |_, _| Ok(true));
            assert_eq!(walked.is_ok(), opens, "{depth}: {walked:?}");
            drop(raw);
            let opened = crate::h5::file::open_read(&path);
            assert_eq!(opened.is_ok(), opens, "{depth}: {:?}", opened.err());
        }
    }

    /// Each group is walked once, so a hard link back to an ancestor ends
    /// the walk rather than looping, and what the file reads from outside is
    /// found however deep and however often it is linked.
    #[test]
    fn s2_the_walk_ends_on_a_cycle_and_finds_what_reads_outside() {
        let dir = tempfile::tempdir().unwrap();
        let other = dir.path().join("other.medh5");
        plain(&other);
        let path = dir.path().join("cycle.medh5");
        plain(&path);
        {
            let file = crate::h5::file::marked(|| hdf5::File::open_rw(&path)).unwrap();
            let inner = file.create_group("x_a").unwrap().create_group("b").unwrap();
            inner.link_hard("/", "up").unwrap();
            inner.link_external(&other.to_string_lossy(), "/data", "out").unwrap();
            file.link_hard("x_a", "x_again").unwrap();
        }
        let found = outside_references(&crate::h5::file::marked(|| hdf5::File::open(&path)).unwrap()).unwrap();
        assert_eq!(found, vec![("x_a/b/out".to_string(), "an external link".to_string())]);
        let clean = dir.path().join("clean.medh5");
        plain(&clean);
        {
            let file = crate::h5::file::marked(|| hdf5::File::open_rw(&clean)).unwrap();
            file.create_group("x_a").unwrap().link_hard("/", "up").unwrap();
        }
        crate::h5::file::open_read(&clean).unwrap();
    }

    /// HDF5 reads one coordinate per axis from the offset, so an offset
    /// of the wrong length read past it (an empty one was a segfault).
    #[test]
    fn s2_2_a_chunk_offset_has_one_coordinate_per_axis() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("c.h5"))).unwrap();
        let ds = file.new_dataset::<u8>().chunk((2, 2)).shape((4, 4)).create("c").unwrap();
        ds.write(&ndarray::Array2::<u8>::ones((4, 4))).unwrap();
        for wrong in [&[][..], &[0][..], &[0, 0, 0][..]] {
            assert!(read_raw_chunk(&ds, wrong).is_err(), "{wrong:?}");
        }
        assert_eq!(read_raw_chunk(&ds, &[0, 2]).unwrap().1, vec![1u8; 4]);
        let flat = file.new_dataset::<u8>().shape([4]).create("flat").unwrap();
        assert!(read_raw_chunk(&flat, &[0]).is_err(), "a contiguous dataset stores no chunks");
    }

    /// Every stored chunk is listed, and none never written, in one walk of
    /// the chunk index: asked for one by one, each ordinal walked the index
    /// from its start, and 10 000 chunks took 13 seconds.
    #[test]
    fn s14_1_the_stored_chunks_are_listed_in_one_walk() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("c.h5")).unwrap();
        let ds = file.new_dataset::<u8>().chunk((1, 1)).shape((100, 100)).create("c").unwrap();
        ds.write(&ndarray::Array2::<u8>::ones((100, 100))).unwrap();
        let sparse = file.new_dataset::<u8>().chunk((2, 2)).shape((6, 6)).create("sparse").unwrap();
        sparse.write_slice(&ndarray::Array2::<u8>::ones((2, 2)), (2..4, 4..6)).unwrap();
        let started = std::time::Instant::now();
        let all = stored_chunks(&ds).unwrap();
        let elapsed = started.elapsed();
        assert_eq!(all.len(), 10_000);
        let distinct: HashSet<Vec<u64>> = all.into_iter().collect();
        assert_eq!(distinct.len(), 10_000);
        assert!(distinct.contains(&vec![99, 0]) && distinct.contains(&vec![0, 99]));
        assert!(elapsed < std::time::Duration::from_secs(3), "{elapsed:?} to list 10 000 chunks");
        assert_eq!(stored_chunks(&sparse).unwrap(), vec![vec![2, 4]]);
    }

    /// A failed call's text names the call and HDF5's reason for refusing it,
    /// where it used to read "HDF5 error" whatever had happened.
    #[test]
    fn the_text_of_an_hdf5_error_names_the_failed_call() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("e.h5")).unwrap();
        let name = cstring("absent").unwrap();
        let text = super::super::locked(|| {
            let id = unsafe { h5d::H5Dopen2(file.id(), name.as_ptr(), h5p::H5P_DEFAULT) };
            assert!(id < 0);
            hdf5_error_text()
        });
        assert!(text.contains("H5Dopen2"), "{text}");
    }

    /// An unchanged file is checked once; the memo is what keeps re-opens cheap.
    #[test]
    fn s2_an_unchanged_file_is_remembered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clean.medh5");
        plain(&path);
        let handle = crate::h5::file::open_read(&path).unwrap();
        let key = opened_identity(&handle, Some(&path)).expect("a sec2 file has an identity");
        assert!(checked_files().lock().unwrap().contains_key(&key));
    }

    /// Windows does not promise a file's times are current while a handle
    /// that may write it is open, so such a file is checked on every open.
    #[cfg(windows)]
    #[test]
    fn s2_a_file_another_handle_may_write_is_not_remembered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clean.medh5");
        plain(&path);
        let writer = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        let handle = crate::h5::file::open_read(&path).unwrap();
        assert!(opened_identity(&handle, Some(&path)).is_none());
        drop(writer);
        assert!(opened_identity(&handle, Some(&path)).is_some());
    }

    /// A file rewritten in place to its old length, its modification time
    /// put back, is a new identity: the change time moved, which a write
    /// cannot help.  So it is checked again rather than remembered.
    #[test]
    fn s2_a_file_changed_in_place_is_checked_again() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clean.medh5");
        plain(&path);
        let before = {
            let handle = crate::h5::file::open_read(&path).unwrap();
            opened_identity(&handle, Some(&path)).expect("a sec2 file has an identity")
        };
        let bytes = std::fs::read(&path).unwrap();
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        // Past the clock's tick, so the change time can move.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_modified(modified).unwrap();
        drop(file);
        let handle = crate::h5::file::open_read(&path).unwrap();
        let after = opened_identity(&handle, Some(&path)).unwrap();
        assert_eq!((after.0, after.1, after.2, after.3), (before.0, before.1, before.2, before.3));
        assert_ne!(after.4, before.4, "the change time moved");
        assert!(checked_files().lock().unwrap().contains_key(&after), "checked under its new identity");
    }
}
