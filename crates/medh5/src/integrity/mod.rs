//! Content addressing and verification (spec §13).

pub mod digest;
pub mod repair;
pub mod verify;
pub(crate) mod workers;

pub use digest::{
    array_digest, attrs_digest, canonical_attrs, collect_digests, compute_content_id, dataset_digest, group_digest,
    relative_path, stamp_digests, AttrNameMap,
};
pub use verify::{
    failing_digests, stale_index_entries, subtrees_identical, unattested, unbound, verify_object, verify_root,
    verify_root_inspecting, Binding, Unbound, VerifyResult,
};

/// `error`, saying `message` instead, of the same kind and code.
pub(crate) fn restated(error: crate::Error, message: String) -> crate::Error {
    use crate::Error;
    match error {
        Error::File(_) => Error::File(message),
        Error::Version(_) => Error::Version(message),
        Error::Schema(_) => Error::Schema(message),
        Error::Validation { code, .. } => Error::Validation { message, code },
        Error::Integrity(_) => Error::Integrity(message),
        Error::Key(_) => Error::Key(message),
        Error::Value(_) => Error::Value(message),
        Error::Type(_) => Error::Type(message),
        Error::Index(_) => Error::Index(message),
        Error::Io(_) => Error::Io(message),
        Error::NotImplemented(_) => Error::NotImplemented(message),
        Error::Runtime(_) => Error::Runtime(message),
    }
}
