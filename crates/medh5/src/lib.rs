#![doc = include_str!("../README.md")]

// First, so its macros reach every module after it.
#[macro_use]
mod timings;

pub mod annotations;
pub mod array;
pub mod bench;
pub mod clinical;
pub mod codes;
pub mod collection;
pub mod companion;
pub mod conformance;
pub mod convert;
pub mod curation;
pub mod dataset;
pub mod digest;
pub mod document;
pub mod error;
pub mod geometry;
pub mod h5;
pub mod ids;
pub mod integrity;
pub mod json;
pub mod labels;
pub mod numeric;
pub mod pyval;
pub mod rng;
pub mod sample;
pub mod sampling;
pub mod storage;
pub mod transforms;
pub mod validate;
pub mod version;

pub use error::{Error, Result};
/// One HDF5 attribute value, as the documented `extra` and `attrs` fields hold
/// it (`Grid.extra`, `TransformHeader.extra`, `AnnotationHeader.extra`,
/// `AnnotationPayload.attrs`) and [`sample::Image::attr`] returns it.
pub use h5::attrs::AttrValue;
/// The HDF5 bindings the engine is built on, for frontends that need a handle.
pub use hdf5;
/// The array crate the engine's arrays convert to and from ([`array::NdArray`]).
pub use ndarray;

/// Raw HDF5 C bindings, for the operations the high-level API lacks.
pub(crate) use medh5_sys::hdf5_sys as h5sys;

/// The package version, stamped into every file's `generator` attribute.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The newest MEDH5 format version this engine implements.
///
/// It implements every version in [`version::format_versions`], and writes the
/// lowest one a sample's content needs: 1.0 for imaging, 1.1 once a sample
/// carries the `clinical` profile ([`version`]).
pub const FORMAT_VERSION: &str = version::LATEST_VERSION;
