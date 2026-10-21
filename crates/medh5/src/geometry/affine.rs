//! The index-world affine and the box/slice convention (spec §3.3, §8.1).
//!
//! Two conventions are fixed here and nowhere else, because a format that
//! leaves either implicit produces silent half-voxel errors that survive every
//! unit test and show up as a systematic bias in a trained model:
//!
//! **Index coordinates.**  Integer index *i* is the **centre** of voxel *i*.  A
//! volume of extent *n* spans the closed region `[-0.5, n - 0.5]`, and
//!
//! ```text
//! x_world = origin + direction @ (spacing * i)
//! ```
//!
//! **Box corners.**  Boxes are measured at voxel **edges**, so a box maps onto
//! a NumPy slice without rounding: slice `a:b` <-> `lo = a - 0.5, hi = b - 0.5`.

use ndarray::Array2;
use serde_json::{json, Value};

use super::linalg::{allclose, det, eye, inv};
use crate::json::format_g;
use crate::{Error, Result};

/// Tolerance on `direction.T @ direction == I` (spec §3.2).
pub const ORTHONORMAL_TOL: f64 = 1e-4;

/// Assemble the `(S+1, S+1)` index-to-world affine of spec §3.3.
pub fn build_affine(spacing: &[f64], origin: &[f64], direction: &Array2<f64>) -> Result<Array2<f64>> {
    let s = spacing.len();
    if origin.len() != s || direction.dim() != (s, s) {
        return Err(Error::coded(
            "E109",
            format!(
                "spacing ({s},), origin ({},) and direction {:?} must agree on the number of spatial axes",
                origin.len(),
                direction.dim()
            ),
        ));
    }
    let mut affine = eye(s + 1);
    for r in 0..s {
        for c in 0..s {
            affine[[r, c]] = direction[[r, c]] * spacing[c];
        }
        affine[[r, s]] = origin[r];
    }
    Ok(affine)
}

/// The number of spatial axes `S` of an `(S+1, S+1)` affine.
///
/// Anything else is refused --- a matrix that is not square, or smaller than
/// 2x2 --- because every function here indexes the affine by the size of its
/// rows: a 4x3 one read past its last column, and an empty one underflowed,
/// each a panic rather than an error.
pub fn spatial_axes(affine: &Array2<f64>) -> Result<usize> {
    let (rows, cols) = affine.dim();
    if rows != cols || rows < 2 {
        return Err(Error::Value(format!(
            "an index-to-world affine is (S+1, S+1) with at least one spatial axis; got shape ({rows}, {cols})"
        )));
    }
    Ok(rows - 1)
}

/// Refuse flat points that are not `s` coordinates each.
fn check_points(s: usize, flat: &[f64]) -> Result<()> {
    if flat.len() % s != 0 {
        return Err(Error::Value(format!(
            "points must have {s} coordinates each, one per spatial axis of the affine; got {} values",
            flat.len()
        )));
    }
    Ok(())
}

/// Split an affine back into `(spacing, origin, direction)`.
pub fn decompose_affine(affine: &Array2<f64>) -> Result<(Vec<f64>, Vec<f64>, Array2<f64>)> {
    let s = spatial_axes(affine)?;
    let mut spacing = vec![0.0; s];
    for (c, sp) in spacing.iter_mut().enumerate() {
        *sp = (0..s).map(|r| affine[[r, c]].powi(2)).sum::<f64>().sqrt();
    }
    if spacing.iter().any(|v| *v <= 0.0) {
        return Err(Error::coded("E104", "affine has a degenerate spatial axis"));
    }
    let mut direction = Array2::zeros((s, s));
    for r in 0..s {
        for c in 0..s {
            direction[[r, c]] = affine[[r, c]] / spacing[c];
        }
    }
    let origin = (0..s).map(|r| affine[[r, s]]).collect();
    Ok((spacing, origin, direction))
}

fn residual(direction: &Array2<f64>) -> Array2<f64> {
    let n = direction.nrows();
    direction.t().dot(direction) - eye(n)
}

/// Whether `direction` is orthonormal within `tol`.
pub fn is_orthonormal(direction: &Array2<f64>, tol: f64) -> bool {
    if direction.nrows() != direction.ncols() {
        return false;
    }
    let n = direction.nrows();
    let gram = direction.t().dot(direction);
    allclose(gram.as_standard_layout().as_slice().unwrap(), eye(n).as_slice().unwrap(), tol, 0.0)
}

/// Validate orthonormality, raising E102 when it fails.
pub fn check_orthonormal(direction: &Array2<f64>, tol: f64, what: &str) -> Result<()> {
    if !is_orthonormal(direction, tol) {
        let r = if direction.nrows() == direction.ncols() {
            residual(direction).iter().fold(0.0f64, |m, v| m.max(v.abs()))
        } else {
            f64::NAN
        };
        return Err(Error::coded(
            "E102",
            format!("{what} is not orthonormal to {} (max residual {})", format_g(tol, 6), format_g(r, 3)),
        ));
    }
    Ok(())
}

/// Whether `matrix` is orthonormal **and** has determinant +1.
pub fn is_proper_rotation(matrix: &Array2<f64>, tol: f64) -> bool {
    if !is_orthonormal(matrix, tol) {
        return false;
    }
    (det(matrix) - 1.0).abs() <= tol.max(1e-6)
}

/// Map continuous index coordinates (flat, `S` per point) to world.
pub fn index_to_world(affine: &Array2<f64>, indices: &[f64]) -> Result<Vec<f64>> {
    let s = spatial_axes(affine)?;
    check_points(s, indices)?;
    let mut out = Vec::with_capacity(indices.len());
    for p in indices.chunks(s) {
        for r in 0..s {
            // The dot product first, then the origin: NumPy's evaluation
            // order for `flat @ A[:S,:S].T + A[:S,S]`.
            let mut v = 0.0;
            for c in 0..s {
                v += affine[[r, c]] * p[c];
            }
            out.push(v + affine[[r, s]]);
        }
    }
    Ok(out)
}

/// Map world coordinates (flat, `S` per point) back to continuous index.
pub fn world_to_index(affine: &Array2<f64>, points: &[f64]) -> Result<Vec<f64>> {
    let s = spatial_axes(affine)?;
    check_points(s, points)?;
    let linear = affine.slice(ndarray::s![..s, ..s]).to_owned();
    let inverse = inv(&linear)?;
    let mut out = Vec::with_capacity(points.len());
    for p in points.chunks(s) {
        for r in 0..s {
            let mut v = 0.0;
            for c in 0..s {
                v += inverse[[r, c]] * (p[c] - affine[[c, s]]);
            }
            out.push(v);
        }
    }
    Ok(out)
}

/// One axis of a box converted to a slice: `(start, stop)`.
pub type SliceBounds = (i64, i64);

/// Convert one `(S, 2)` box (flat `[lo0, hi0, lo1, hi1, ...]`) to slices.
///
/// `start = floor(lo + 0.5)`, `stop = floor(hi + 0.5)` (spec §8.1) --- half
/// **up**, never half-to-even, or a box on integer edge coordinates rounds its
/// two ends in opposite directions.  With `shape`, one extent per axis of the
/// box, the result is clipped.
pub fn box_to_slices(bbox: &[f64], shape: Option<&[usize]>) -> Result<Vec<SliceBounds>> {
    if bbox.len() % 2 != 0 {
        return Err(Error::invalid(format!("box must have shape (S, 2), got ({},)", bbox.len())));
    }
    let s = bbox.len() / 2;
    if let Some(extent) = shape {
        if extent.len() != s {
            return Err(Error::Value(format!(
                "a box with {s} axes is clipped to a shape of {s} extents; got {}",
                crate::json::repr_int_tuple(extent)
            )));
        }
    }
    // `lo > hi` is false for a NaN, which became voxel 0 (`NaN as i64`), and
    // an infinite bound a slice to the end of the integers.
    if (0..s).any(|k| !(bbox[2 * k].is_finite() && bbox[2 * k + 1].is_finite() && bbox[2 * k] <= bbox[2 * k + 1])) {
        return Err(Error::coded("E406", "box has lo > hi, or a bound that is not finite, on at least one axis"));
    }
    let mut out = Vec::with_capacity(s);
    for k in 0..s {
        let mut start = (bbox[2 * k] + 0.5).floor() as i64;
        let mut stop = (bbox[2 * k + 1] + 0.5).floor() as i64;
        if let Some(extent) = shape {
            // An extent past `i64::MAX` wrapped negative, and `clamp` panics
            // on a range whose bounds are reversed.  No index reaches it.
            let n = i64::try_from(extent[k]).unwrap_or(i64::MAX);
            start = start.clamp(0, n);
            stop = stop.clamp(start, n);
        }
        out.push((start, stop));
    }
    Ok(out)
}

/// Convert slices to one `(S, 2)` box (flat), in continuous index coordinates.
pub fn slices_to_box(slices: &[(i64, i64)]) -> Vec<f32> {
    let mut out = Vec::with_capacity(slices.len() * 2);
    for (start, stop) in slices {
        out.push(*start as f32 - 0.5);
        out.push(*stop as f32 - 0.5);
    }
    out
}

/// The most axes a box's corners are enumerated for: `2**S` of them.
pub const MAX_CORNER_AXES: usize = 16;

/// The `2**S` corners of an axis-aligned box (flat `(S, 2)`), odometer order.
pub fn box_corners(bbox: &[f64]) -> Result<Vec<Vec<f64>>> {
    if bbox.len() % 2 != 0 {
        return Err(Error::invalid(format!("box must have shape (S, 2), got ({},)", bbox.len())));
    }
    let s = bbox.len() / 2;
    // `1 << S` overflowed past 63 axes, and long before that the corners
    // outgrow any memory.
    if s > MAX_CORNER_AXES {
        return Err(Error::Value(format!(
            "a box with {s} axes has 2**{s} corners; at most {MAX_CORNER_AXES} axes are enumerated"
        )));
    }
    let mut out = Vec::with_capacity(1 << s);
    for n in 0..(1usize << s) {
        let mut corner = Vec::with_capacity(s);
        for k in 0..s {
            // Odometer: the first axis varies slowest.
            let bit = (n >> (s - 1 - k)) & 1;
            corner.push(bbox[2 * k + bit]);
        }
        out.push(corner);
    }
    Ok(out)
}

/// Axis-aligned bounds `(S, 2)` (flat) of a box after an affine.
pub fn apply_affine_to_box(affine: &Array2<f64>, bbox: &[f64]) -> Result<Vec<f64>> {
    let s = spatial_axes(affine)?;
    if bbox.len() != 2 * s {
        return Err(Error::Value(format!(
            "a box under an affine with {s} spatial axes has shape ({s}, 2); got {} values",
            bbox.len()
        )));
    }
    let corners: Vec<f64> = box_corners(bbox)?.into_iter().flatten().collect();
    let world = index_to_world(affine, &corners)?;
    let mut out = Vec::with_capacity(2 * s);
    for k in 0..s {
        let values = world.iter().skip(k).step_by(s);
        let lo = values.clone().fold(f64::INFINITY, |a, b| a.min(*b));
        let hi = values.fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        out.push(lo);
        out.push(hi);
    }
    Ok(out)
}

/// Volume of one voxel in the grid's units^S.
pub fn voxel_volume(spacing: &[f64]) -> f64 {
    spacing.iter().product()
}

/// Human-facing summary of an affine.
pub fn affine_summary(affine: &Array2<f64>) -> Result<Value> {
    let (spacing, origin, direction) = decompose_affine(affine)?;
    Ok(json!({
        "spacing": spacing,
        "origin": origin,
        "direction": direction.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
        "determinant": det(&direction),
        "orthonormal": is_orthonormal(&direction, ORTHONORMAL_TOL),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_roundtrip() {
        let dir = eye(3);
        let a = build_affine(&[1.5, 0.8, 0.8], &[-12.0, -9.6, -9.6], &dir).unwrap();
        let w = index_to_world(&a, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(w, vec![-10.5, -8.0, -7.199999999999999]);
        let back = world_to_index(&a, &w).unwrap();
        assert!(allclose(&back, &[1.0, 2.0, 3.0], 1e-12, 0.0));
    }

    #[test]
    fn boxes_round_half_up() {
        // [11.5, 39.5] <-> slice(12, 40), extent 28 (Appendix C.2).
        assert_eq!(box_to_slices(&[11.5, 39.5], None).unwrap(), vec![(12, 40)]);
        // One voxel on integer edges: half-to-even would give an empty slice.
        assert_eq!(box_to_slices(&[1.0, 2.0], None).unwrap(), vec![(1, 2)]);
        assert_eq!(box_to_slices(&[2.0, 3.0], None).unwrap(), vec![(2, 3)]);
        assert_eq!(slices_to_box(&[(12, 40)]), vec![11.5, 39.5]);
        assert_eq!(box_to_slices(&[2.0, 1.0], None).unwrap_err().code(), Some("E406"));
    }

    /// `lo > hi` is false for a NaN: a NaN bound became voxel 0 and an
    /// infinite one a slice to the end of the integers, clipped or not.
    #[test]
    fn s8_1_a_bound_that_is_not_finite_is_e406() {
        let (nan, inf) = (f64::NAN, f64::INFINITY);
        for (lo, hi) in [(nan, 4.0), (0.0, nan), (nan, nan), (-inf, 4.0), (0.0, inf), (-inf, inf), (inf, inf)] {
            let bbox = [0.0, 1.0, lo, hi];
            for shape in [None, Some(&[4, 10][..])] {
                assert_eq!(box_to_slices(&bbox, shape).unwrap_err().code(), Some("E406"), "{bbox:?} {shape:?}");
            }
        }
        assert_eq!(box_to_slices(&[0.0, 1.0, -0.5, 9.5], Some(&[4, 10])).unwrap(), vec![(0, 1), (0, 10)]);
    }

    /// An extent past `i64::MAX` wrapped negative, and clamping a slice to
    /// it panicked on reversed bounds.  No index reaches it: it clips nothing.
    #[test]
    fn s8_1_an_extent_past_the_integers_clips_without_a_panic() {
        let wide = [1usize << 63, usize::MAX];
        assert_eq!(box_to_slices(&[-0.5, 3.5, 2.0, 1e300], Some(&wide)).unwrap(), vec![(0, 4), (2, i64::MAX)]);
        assert_eq!(box_to_slices(&[-1e300, 1e300], Some(&[usize::MAX])).unwrap(), vec![(0, i64::MAX)]);
    }

    #[test]
    fn corners_in_odometer_order() {
        let c = box_corners(&[0.0, 1.0, 10.0, 11.0]).unwrap();
        assert_eq!(c, vec![vec![0.0, 10.0], vec![0.0, 11.0], vec![1.0, 10.0], vec![1.0, 11.0]]);
    }

    /// A matrix that is not `(S+1, S+1)`, points that are not `S`
    /// coordinates each, and a shape that is not one extent per axis are
    /// errors.  Each indexed past what it was given and panicked.
    #[test]
    fn s3_3_misshapen_arguments_are_errors_not_panics() {
        let tall = Array2::from_shape_fn((4, 3), |(r, c)| if r == c { 2.0 } else { 0.0 });
        let empty = Array2::<f64>::zeros((0, 0));
        let one = Array2::<f64>::ones((1, 1));
        for bad in [&tall, &empty, &one] {
            assert!(matches!(spatial_axes(bad), Err(Error::Value(_))), "{:?}", bad.dim());
            assert!(decompose_affine(bad).is_err());
            assert!(index_to_world(bad, &[1.0, 2.0, 3.0]).is_err());
            assert!(world_to_index(bad, &[1.0, 2.0, 3.0]).is_err());
            assert!(apply_affine_to_box(bad, &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).is_err());
        }
        let a = build_affine(&[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], &eye(3)).unwrap();
        assert!(matches!(index_to_world(&a, &[1.0, 2.0]), Err(Error::Value(_))));
        assert!(matches!(world_to_index(&a, &[1.0, 2.0, 3.0, 4.0]), Err(Error::Value(_))));
        assert!(apply_affine_to_box(&a, &[0.0, 1.0, 0.0, 1.0]).is_err());
        // A 3-D box clipped to a 2-D shape, or a 2-D one to a 3-D shape.
        assert!(matches!(box_to_slices(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0], Some(&[4, 4])), Err(Error::Value(_))));
        assert!(matches!(box_to_slices(&[0.0, 1.0, 0.0, 1.0], Some(&[4, 4, 4])), Err(Error::Value(_))));
        assert!(box_corners(&[0.0; 3]).is_err());
        assert!(matches!(box_corners(&[0.0; 2 * 64]), Err(Error::Value(_))));
        // The controls still answer.
        assert_eq!(index_to_world(&a, &[1.0, 2.0, 3.0]).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(box_to_slices(&[0.0, 1.0, 0.0, 1.0], Some(&[4, 4])).unwrap(), vec![(0, 1), (0, 1)]);
        assert_eq!(
            apply_affine_to_box(&a, &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unwrap(),
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        );
        assert_eq!(decompose_affine(&a).unwrap().0, vec![1.0, 1.0, 1.0]);
    }
}
