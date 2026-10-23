//! Copying an object graph into another file, links and all.
//!
//! HDF5's `H5Ocopy` keeps the links *inside* one call: two hard links to one
//! dataset under a copied group stay two links to one copy.  It shares nothing
//! across calls, so a sample copied member by member --- as pack, unpack,
//! extract, repack and amend did --- turned a hard link between two members
//! into a second dataset, and `H5O_COPY_EXPAND_SOFT_LINK_FLAG` turned a soft
//! link into one too.  Each is a new dataset line under `content_id` (§13.2):
//! packed and unpacked, a sample with an alias failed `verify`, and a no-op
//! amend changed its address.
//!
//! [`GraphCopy`] walks the source itself, with one map of what it has copied
//! for the whole walk:
//!
//! * a hard link to an object already copied becomes a hard link to its copy,
//!   so an alias stays an alias and a link back to an ancestor stays a cycle;
//! * a soft link stays a soft link, its absolute target rebased from the
//!   source's sample root to the destination's;
//! * groups are created and their attributes copied as stored;
//! * a dataset goes through [`CopyHooks::dataset`] --- `H5Ocopy` of the one
//!   dataset in [`Raw`], so chunks move as stored bytes, or re-encoded by
//!   recompress;
//! * a committed datatype is copied with HDF5's merge of committed types, so
//!   datasets that share one share its copy.
//!
//! Members are walked depth first in byte order of their names, the order of
//! the walk that gives every object its first path (§13.2), so each object is
//! copied at its first path and every later path links to it.

use std::collections::HashMap;
use std::ffi::CString;

use super::attrs;
use super::ops::{self, LinkKind, NodeKind, ObjectId};
use crate::h5sys::{h5o, h5p};
use crate::json::repr_str;
use crate::{Error, Result};

/// What a [`GraphCopy`] does with the datasets it reaches.
pub trait CopyHooks {
    /// Copy dataset `name` of `src` into `dst` under the same name.
    fn dataset(&mut self, src: &hdf5::Group, name: &str, dst: &hdf5::Group) -> Result<()>;

    /// The walk enters a group of the source (the root included).
    fn enter(&mut self, _src: &hdf5::Group) -> Result<()> {
        Ok(())
    }

    /// The walk leaves a group it entered.
    fn leave(&mut self, _src: &hdf5::Group) {}
}

/// Datasets move as stored: `H5Ocopy` of the one dataset, its chunks as raw
/// bytes, its attributes and its type as they are.
pub struct Raw;

impl CopyHooks for Raw {
    fn dataset(&mut self, src: &hdf5::Group, name: &str, dst: &hdf5::Group) -> Result<()> {
        copy_stored(src, name, dst)
    }
}

/// `H5Ocopy` one object as stored, committed datatypes merged with those
/// already in the destination --- so two datasets copied in two calls still
/// share the one type they shared --- and soft links inside it kept as links.
pub fn copy_stored(src: &hdf5::Group, name: &str, dst: &hdf5::Group) -> Result<()> {
    phase!(Copy);
    let cname = CString::new(name).map_err(|_| Error::Value(format!("name {} contains a NUL byte", repr_str(name))))?;
    super::locked(|| unsafe {
        let ocpypl = h5p::H5Pcreate(*crate::h5sys::h5p::H5P_CLS_OBJECT_COPY);
        h5p::H5Pset_copy_object(ocpypl, h5o::H5O_COPY_MERGE_COMMITTED_DTYPE_FLAG);
        let status = h5o::H5Ocopy(src.id(), cname.as_ptr(), dst.id(), cname.as_ptr(), ocpypl, h5p::H5P_DEFAULT);
        h5p::H5Pclose(ocpypl);
        if status < 0 {
            return Err(Error::Io(format!("could not copy {} into {}", repr_str(name), dst.name())));
        }
        Ok(())
    })
}

/// One copy of a graph: where each source object went.
pub struct GraphCopy {
    /// Source object -> the absolute path of its first copy.
    copied: HashMap<ObjectId, String>,
    /// The source's sample root, and the destination's: absolute soft-link
    /// targets under the one are rewritten under the other.
    src_root: String,
    dst_root: String,
}

impl GraphCopy {
    /// A copy of the graph under `src_root` into `dst_root`; the two roots
    /// are each other's, so a link back to the source root is a link to the
    /// destination root.
    pub fn new(src_root: &hdf5::Group, dst_root: &hdf5::Group) -> GraphCopy {
        let mut copy = GraphCopy { copied: HashMap::new(), src_root: src_root.name(), dst_root: dst_root.name() };
        copy.note(src_root, &dst_root.name());
        copy
    }

    fn note(&mut self, src: &hdf5::Location, dst_path: &str) {
        if let Some(id) = ops::object_id_of(src) {
            self.copied.insert(id, dst_path.to_string());
        }
    }

    /// Copy every member of `src` --- except those `skip` names --- into
    /// `dst`, which must not hold them yet.  `src`'s own attributes are the
    /// caller's to copy.
    pub fn members(
        &mut self,
        src: &hdf5::Group,
        dst: &hdf5::Group,
        skip: &[&str],
        hooks: &mut dyn CopyHooks,
    ) -> Result<()> {
        phase!(Copy);
        hooks.enter(src)?;
        let result = self.walk(src, dst, skip, 0, hooks);
        hooks.leave(src);
        result
    }

    fn walk(
        &mut self,
        src: &hdf5::Group,
        dst: &hdf5::Group,
        skip: &[&str],
        depth: usize,
        hooks: &mut dyn CopyHooks,
    ) -> Result<()> {
        if depth > ops::MAX_DEPTH {
            return Err(Error::File(format!(
                "{} is nested more than {} groups deep; it is refused rather than followed",
                repr_str(&src.name()),
                ops::MAX_DEPTH
            )));
        }
        let base = dst.name();
        let base = base.trim_end_matches('/');
        for name in ops::members(src)? {
            if skip.contains(&name.as_str()) {
                continue;
            }
            match ops::link_kind(src, &name) {
                Some(LinkKind::Soft) => {
                    let target = self.rebase(&ops::soft_link_target(src, &name)?, &src.name(), &name)?;
                    dst.link_soft(&target, &name)?;
                    continue;
                }
                Some(LinkKind::Hard) => {
                    if let Some(first) = ops::object_id_by_name(src, &name).and_then(|id| self.copied.get(&id)) {
                        dst.link_hard(first, &name)?;
                        continue;
                    }
                }
                // A file holding an external link is refused when it is
                // opened (§2); anything else is a link this walk does not
                // know how to keep.
                other => {
                    return Err(Error::File(format!(
                        "{}/{} is {}, which a copy cannot keep",
                        src.name().trim_end_matches('/'),
                        name,
                        match other {
                            Some(LinkKind::External) => "an external link",
                            _ => "a link of a kind HDF5 defines for another library",
                        }
                    )))
                }
            }
            let path = format!("{base}/{name}");
            match ops::node_kind(src, &name) {
                Some(NodeKind::Group) => {
                    let source = src.group(&name)?;
                    let child = ops::new_group(dst, &name)?;
                    for key in attrs::names(&source)? {
                        attrs::copy_raw(&source, &child, &key)?;
                    }
                    self.note(&source, &path);
                    hooks.enter(&source)?;
                    let walked = self.walk(&source, &child, &[], depth + 1, hooks);
                    hooks.leave(&source);
                    walked?;
                }
                Some(NodeKind::Dataset) => {
                    let id = ops::object_id_by_name(src, &name);
                    hooks.dataset(src, &name, dst)?;
                    if let Some(id) = id {
                        self.copied.insert(id, path);
                    }
                }
                // A committed datatype, as HDF5 copies one.
                _ => {
                    let id = ops::object_id_by_name(src, &name);
                    copy_stored(src, &name, dst)?;
                    if let Some(id) = id {
                        self.copied.insert(id, path);
                    }
                }
            }
        }
        Ok(())
    }

    /// The target an absolute soft link holds, carried from the source's
    /// sample root to the destination's.  A relative target is a path from
    /// the link's own group, the same in both.
    fn rebase(&self, target: &str, group: &str, name: &str) -> Result<String> {
        if !target.starts_with('/') {
            return Ok(target.to_string());
        }
        let src = self.src_root.trim_end_matches('/');
        let dst = self.dst_root.trim_end_matches('/');
        let rest = if src.is_empty() {
            Some(target)
        } else if target == src {
            Some("/")
        } else {
            target.strip_prefix(src).filter(|rest| rest.starts_with('/'))
        };
        let Some(rest) = rest else {
            return Err(Error::File(format!(
                "the soft link {}/{name} points at {}, outside its sample root {}; a copy of the sample cannot \
                 keep it pointing at the same object",
                group.trim_end_matches('/'),
                repr_str(target),
                repr_str(&self.src_root)
            )));
        };
        Ok(if dst.is_empty() {
            rest.to_string()
        } else if rest == "/" {
            dst.to_string()
        } else {
            format!("{dst}{rest}")
        })
    }
}

/// Copy every member of `src` but `skip`, and every attribute of `src`, into
/// `dst`, datasets as stored: what pack, unpack, extract and repack do.
pub fn copy_root_raw(src: &hdf5::Group, dst: &hdf5::Group, skip: &[&str]) -> Result<()> {
    phase!(Copy);
    GraphCopy::new(src, dst).members(src, dst, skip, &mut Raw)?;
    for key in attrs::names(src)? {
        attrs::copy_raw(src, dst, &key)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(path: &std::path::Path) -> hdf5::File {
        let file = crate::h5::file::create_truncate(path).unwrap();
        let images = file.create_group("images").unwrap();
        images.new_dataset::<u8>().shape([4]).create("CT").unwrap().write(&[1u8, 2, 3, 4]).unwrap();
        file.link_hard("/images/CT", "zzz_alias").unwrap();
        file.link_hard("/", "zzz_loop").unwrap();
        file.link_soft("/images/CT", "zzz_soft").unwrap();
        file.link_soft("CT", "images/relative").unwrap();
        file
    }

    /// A copy keeps the graph --- an alias, a cycle and soft links ---
    /// into a root of another name and back.
    #[test]
    fn s2_2_a_copy_keeps_aliases_cycles_and_soft_links() {
        let dir = tempfile::tempdir().unwrap();
        let src = source(&dir.path().join("src.h5"));
        let packed = crate::h5::file::create_truncate(&dir.path().join("packed.h5")).unwrap();
        let member = packed.create_group("samples").unwrap().create_group("k").unwrap();
        copy_root_raw(&src.as_group().unwrap(), &member, &[]).unwrap();
        let id = |group: &hdf5::Group, name: &str| ops::object_id_by_name(group, name).unwrap();
        assert_eq!(id(&member, "zzz_alias"), id(&member, "images/CT"));
        assert_eq!(id(&member, "zzz_loop"), ops::object_id_of(&member).unwrap());
        assert_eq!(ops::soft_link_target(&member, "zzz_soft").unwrap(), "/samples/k/images/CT");
        assert_eq!(ops::soft_link_target(&member.group("images").unwrap(), "relative").unwrap(), "CT");
        assert!(crate::integrity::subtrees_identical(&src.as_group().unwrap(), &member).unwrap().is_empty());
        // And back to a root of its own.
        let unpacked = crate::h5::file::create_truncate(&dir.path().join("unpacked.h5")).unwrap();
        let root = unpacked.as_group().unwrap();
        copy_root_raw(&member, &root, &[]).unwrap();
        assert_eq!(ops::soft_link_target(&root, "zzz_soft").unwrap(), "/images/CT");
        assert_eq!(id(&root, "zzz_alias"), id(&root, "images/CT"));
        assert!(crate::integrity::subtrees_identical(&src.as_group().unwrap(), &root).unwrap().is_empty());
    }

    /// A soft link out of the sample root it is copied from cannot be kept.
    #[test]
    fn s2_2_a_soft_link_out_of_its_sample_root_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("c.h5")).unwrap();
        let samples = file.create_group("samples").unwrap();
        let a = samples.create_group("a").unwrap();
        a.new_dataset::<u8>().shape([1]).create("x").unwrap();
        let b = samples.create_group("b").unwrap();
        b.link_soft("/samples/a/x", "elsewhere").unwrap();
        let out = crate::h5::file::create_truncate(&dir.path().join("out.h5")).unwrap();
        let err = copy_root_raw(&b, &out.as_group().unwrap(), &[]).unwrap_err();
        assert!(err.to_string().contains("outside its sample root"), "{err}");
    }

    /// Two datasets sharing one committed datatype share its copy.
    #[test]
    fn s2_2_datasets_sharing_a_committed_type_share_its_copy() {
        use crate::h5sys::{h5d, h5i, h5s, h5t};
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("t.h5")).unwrap();
        crate::h5::locked(|| unsafe {
            let tid = h5t::H5Tcopy(*h5t::H5T_NATIVE_INT32);
            let name = CString::new("types_t").unwrap();
            assert!(
                h5t::H5Tcommit2(file.id(), name.as_ptr(), tid, h5p::H5P_DEFAULT, h5p::H5P_DEFAULT, h5p::H5P_DEFAULT)
                    >= 0
            );
            let dims = [2u64];
            for ds in ["a", "b"] {
                let space = h5s::H5Screate_simple(1, dims.as_ptr(), std::ptr::null());
                let cname = CString::new(ds).unwrap();
                let id = h5d::H5Dcreate2(
                    file.id(),
                    cname.as_ptr(),
                    tid,
                    space,
                    h5p::H5P_DEFAULT,
                    h5p::H5P_DEFAULT,
                    h5p::H5P_DEFAULT,
                );
                assert!(id >= 0);
                h5d::H5Dclose(id);
                h5s::H5Sclose(space);
            }
            h5t::H5Tclose(tid);
        });
        let out = crate::h5::file::create_truncate(&dir.path().join("o.h5")).unwrap();
        copy_root_raw(&file.as_group().unwrap(), &out.as_group().unwrap(), &[]).unwrap();
        let type_of = |name: &str| {
            let ds = out.dataset(name).unwrap();
            crate::h5::locked(|| unsafe {
                let tid = h5d::H5Dget_type(ds.id());
                assert!(h5t::H5Tcommitted(tid) > 0, "{name} lost its committed type");
                let mut info: h5o::H5O_info2_t = std::mem::zeroed();
                h5o::H5Oget_info3(tid, &mut info, h5o::H5O_INFO_BASIC);
                h5i::H5Idec_ref(tid);
                std::mem::transmute::<h5o::H5O_token_t, [u8; 16]>(info.token)
            })
        };
        assert_eq!(type_of("a"), type_of("b"));
    }
}
