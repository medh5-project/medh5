//! Registration transforms (spec §10).
//!
//! **A transform with `from_frame = F` and `to_frame = M` maps a point
//! expressed in F to the corresponding point in M: `x_M = T(x_F)`.**  That is
//! the ITK `TransformPoint` convention, and there is no attribute to select the
//! other one.

pub mod apply;
pub mod encode;
pub mod model;
pub mod resample;
pub mod resolve;

pub use apply::{
    folding_fraction, jacobian_determinant, linear_sample, sample_field, to_world_vectors, Tre, EXTRAPOLATIONS,
};
pub use encode::{encode_affine, encode_bspline, encode_composite, encode_displacement, encode_identity};
pub use model::{
    can_invert, stored_inverse, Body, Transform, TransformHeader, INTERPOLATIONS, SPEC_TRANSFORM_ATTRS,
    TRANSFORM_KINDS, VECTOR_SPACES,
};
pub use resample::{
    resample, ResamplePlan, ResampleTarget, Resampled, ResampledAnnotation, ResampledImage, RESAMPLE_INTERPOLATIONS,
};
pub use resolve::{frame_graph, frames_of_timepoint, resolve_between};
