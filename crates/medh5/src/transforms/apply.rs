//! Evaluating transforms: interpolation, Jacobians and registration error (§10).
//!
//! Interpolation is implemented here rather than delegated, because the two
//! things a reader needs from a displacement field --- its value between
//! samples and the determinant of its Jacobian --- must agree with each other.
//! Cubic sampling reproduces SciPy's `map_coordinates(order=3)` --- the same
//! prefilter, boundary handling and accumulation order --- so a file declaring
//! `interpolation = "cubic"` evaluates identically from every frontend.  At the
//! field's edge it follows §10.4 instead, as linear sampling does: within the
//! extent's half-voxel margin a point takes the outermost sample's value, where
//! SciPy's constant mode gives zero.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use ndarray::{Array2, ArrayD, Axis, IxDyn};

use crate::geometry::grid::Grid;
use crate::geometry::linalg::{det, inv};
use crate::json::{repr_int_tuple, repr_str};
use crate::{Error, Result};

/// The `extrapolation` values of a displacement field (§10.4).
pub const EXTRAPOLATIONS: [&str; 3] = ["zero", "nearest", "error"];

/// `2**52`: from here on every `f64` is an integer.  A continuous coordinate
/// clamped to it keeps its interpolation weights, and the integer indices
/// computed from it stay far from overflow.
pub const EXACT_INTEGER: f64 = 4_503_599_627_370_496.0;

fn check_extrapolation(extrapolation: &str) -> Result<()> {
    if !EXTRAPOLATIONS.contains(&extrapolation) {
        return Err(Error::invalid(format!("unknown extrapolation {}", repr_str(extrapolation))));
    }
    Ok(())
}

/// Refuse `points` that are not `(N, S)` for a lattice of `spatial` extent:
/// one coordinate per spatial axis.  The samplers indexed a point by axis, so
/// one with fewer coordinates panicked --- a `PanicException`, which `except
/// Exception` does not catch --- and one with more had the rest ignored, where
/// 1.x's SciPy refused the shape.
pub fn check_points(spatial: &[usize], points: &Array2<f64>) -> Result<()> {
    if points.ncols() != spatial.len() {
        return Err(Error::Value(format!(
            "points have {} coordinate(s) and the field {} spatial axes; each point needs one per axis",
            points.ncols(),
            spatial.len()
        )));
    }
    Ok(())
}

/// Which `(N, S)` points lie within a lattice of `spatial` extent, edges included.
pub fn inside_extent(spatial: &[usize], points: &Array2<f64>) -> Vec<bool> {
    points.outer_iter().map(|p| p.iter().zip(spatial).all(|(v, n)| *v >= -0.5 && *v <= *n as f64 - 0.5)).collect()
}

/// `extrapolation = "error"` is a refusal, not a quieter fill value.
pub fn refuse_outside(inside: &[bool]) -> Result<()> {
    let outside = inside.iter().filter(|v| !**v).count();
    if outside == 0 {
        return Ok(());
    }
    Err(Error::invalid(format!("{outside} point(s) fall outside the field and extrapolation='error'")))
}

/// `(N, S)` view of `(..., S)` coordinates: `dim` of them on the last axis.
///
/// Only the last axis says what a point is.  Regrouping the values by their
/// count instead read three 2-D points as two 3-D ones, and a 3-D affine
/// applied to them returned coordinates mixed across points, with no error.
pub fn as_points(coords: &ArrayD<f64>, dim: usize) -> Result<Array2<f64>> {
    if dim == 0 || coords.shape().last() != Some(&dim) {
        return Err(Error::Value(format!(
            "points must have {dim} coordinates on their last axis, one per spatial axis, got shape {}",
            repr_int_tuple(coords.shape())
        )));
    }
    let flat: Vec<f64> = coords.iter().copied().collect();
    Ok(Array2::from_shape_vec((flat.len() / dim, dim), flat)?)
}

/// Refuse a field that is not `(C, *spatial)` with a sample on every axis.
/// The samplers index it by those axes, so a 0-d field or an empty axis
/// panicked.
pub fn check_field(field: &ArrayD<f64>) -> Result<()> {
    if field.ndim() < 2 || field.shape()[1..].contains(&0) {
        return Err(Error::Value(format!(
            "a field is (C, *spatial) with at least one sample per spatial axis, got shape {}",
            repr_int_tuple(field.shape())
        )));
    }
    Ok(())
}

/// Multilinear interpolation of `(C, *spatial)` data at continuous indices.
///
/// `coords` is `(N, S)` in the field's continuous index coordinates.  Points
/// outside follow `extrapolation`: `zero`, `nearest` (clamp) or `error`.
pub fn linear_sample(field: &ArrayD<f64>, coords: &Array2<f64>, extrapolation: &str) -> Result<Array2<f64>> {
    phase!(Interpolate);
    check_field(field)?;
    let components = field.shape()[0];
    let spatial: Vec<usize> = field.shape()[1..].to_vec();
    let dim = spatial.len();
    check_extrapolation(extrapolation)?;
    check_points(&spatial, coords)?;
    let inside = inside_extent(&spatial, coords);
    if extrapolation == "error" {
        refuse_outside(&inside)?;
    }
    let n = coords.nrows();
    let mut out = Array2::<f64>::zeros((n, components));
    // Each axis's two taps, as offsets into the C-order values, and their
    // weights are worked out once a point; each corner then sums its axes'
    // weights in axis order, as the whole computation did a corner.
    let mut taps = vec![[0usize; 2]; dim];
    let mut weights = vec![[0f64; 2]; dim];
    let field = field.as_standard_layout();
    let values = field.as_slice().expect("standard layout");
    let plane: usize = spatial.iter().product();
    let strides = c_strides(&spatial);
    for (row, point) in coords.outer_iter().enumerate() {
        for axis in 0..dim {
            (taps[axis], weights[axis]) = linear_axis(point[axis], spatial[axis], strides[axis]);
        }
        add_corners(out.row_mut(row), &taps, &weights, values, plane);
    }
    if extrapolation == "zero" {
        for (row, ok) in inside.iter().enumerate() {
            if !ok {
                out.row_mut(row).fill(0.0);
            }
        }
    }
    Ok(out)
}

/// [`linear_sample`] under `nearest` at the C-order product of `axes`: the
/// points of a block whose every voxel takes its coordinate on axis `k`
/// from `axes[k]`, `(N, S)` with `N` the product of the axes' lengths.  Each
/// axis's taps and weights are worked out once a coordinate, not once a
/// point; every sum is [`linear_sample`]'s, term by term.
pub(crate) fn linear_sample_axes(field: &ArrayD<f64>, axes: &[Vec<f64>]) -> Result<Array2<f64>> {
    phase!(Interpolate);
    check_field(field)?;
    let components = field.shape()[0];
    let spatial: Vec<usize> = field.shape()[1..].to_vec();
    let dim = spatial.len();
    check_axes(&spatial, axes)?;
    let field = field.as_standard_layout();
    let values = field.as_slice().expect("standard layout");
    let plane: usize = spatial.iter().product();
    let strides = c_strides(&spatial);
    let found: Vec<Vec<([usize; 2], [f64; 2])>> = axes
        .iter()
        .enumerate()
        .map(|(axis, coords)| coords.iter().map(|v| linear_axis(*v, spatial[axis], strides[axis])).collect())
        .collect();
    let lengths: Vec<usize> = axes.iter().map(Vec::len).collect();
    let n: usize = lengths.iter().product();
    let mut out = Array2::<f64>::zeros((n, components));
    let mut taps = vec![[0usize; 2]; dim];
    let mut weights = vec![[0f64; 2]; dim];
    let mut at = vec![0usize; dim];
    for row in 0..n {
        for axis in 0..dim {
            (taps[axis], weights[axis]) = found[axis][at[axis]];
        }
        add_corners(out.row_mut(row), &taps, &weights, values, plane);
        count_up(&mut at, &lengths);
    }
    Ok(out)
}

/// One axis of a multilinear sample at `coordinate`, clamped to the axis's
/// `extent` samples: its two taps, as offsets `stride` apart a sample into
/// the C-order values, and their weights.
fn linear_axis(coordinate: f64, extent: usize, stride: usize) -> ([usize; 2], [f64; 2]) {
    let extent = extent as i64;
    let clamped = coordinate.max(0.0).min(extent as f64 - 1.0);
    // `clamped` is at least 0, where truncating is the floor.
    let b = (clamped as i64).min((extent - 2).max(0));
    let frac = clamped - b as f64;
    let mut taps = [0usize; 2];
    for (offset, tap) in taps.iter_mut().enumerate() {
        *tap = (b + offset as i64).min(extent - 1).max(0) as usize * stride;
    }
    (taps, [1.0 - frac, frac])
}

/// Add each corner's term to `sums`, one a component: the corner's weight is
/// its axes' weights multiplied in axis order, as the whole computation did a
/// corner.
fn add_corners(
    mut sums: ndarray::ArrayViewMut1<f64>,
    taps: &[[usize; 2]],
    weights: &[[f64; 2]],
    values: &[f64],
    plane: usize,
) {
    for corner in 0..(1usize << taps.len()) {
        let mut weight = weights[0][corner & 1];
        let mut at = taps[0][corner & 1];
        for axis in 1..taps.len() {
            let bit = (corner >> axis) & 1;
            weight *= weights[axis][bit];
            at += taps[axis][bit];
        }
        for (c, sum) in sums.iter_mut().enumerate() {
            *sum += weight * values[c * plane + at];
        }
    }
}

/// The C-order strides, in elements, of an array of `shape`.
pub(crate) fn c_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

/// The next of the C-order positions in a product of axes `lengths` long.
pub(crate) fn count_up(at: &mut [usize], lengths: &[usize]) {
    for k in (0..at.len()).rev() {
        at[k] += 1;
        if at[k] < lengths[k] {
            return;
        }
        at[k] = 0;
    }
}

/// Refuse `axes` that are not one list of coordinates per spatial axis.
fn check_axes(spatial: &[usize], axes: &[Vec<f64>]) -> Result<()> {
    if axes.len() != spatial.len() {
        return Err(Error::Value(format!(
            "points have {} coordinate(s) and the field {} spatial axes; each point needs one per axis",
            axes.len(),
            spatial.len()
        )));
    }
    Ok(())
}

// -- cubic: SciPy's `map_coordinates(order=3)` -------------------------------------

/// The cubic B-spline filter pole, `sqrt(3) - 2`.
const CUBIC_POLE: f64 = -0.2679491924311227;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Boundary {
    /// `mode="constant"`: mirror-initialised prefilter, no interpolation outside.
    Constant,
    /// `mode="nearest"`: edge-padded by 12, reflect-initialised, clamped.
    Nearest,
}

fn init_causal_mirror(c: &mut [f64], z: f64) {
    let n = c.len();
    let mut z_i = z;
    let z_n_1 = z.powf((n - 1) as f64);
    c[0] += z_n_1 * c[n - 1];
    for i in 1..n - 1 {
        c[0] += z_i * (c[i] + z_n_1 * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] /= 1.0 - z_n_1 * z_n_1;
}

fn init_anticausal_mirror(c: &mut [f64], z: f64) {
    let n = c.len();
    c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1.0);
}

fn init_causal_reflect(c: &mut [f64], z: f64) {
    let n = c.len();
    let mut z_i = z;
    let z_n = z.powf(n as f64);
    let mut sum = c[0] + z_n * c[n - 1];
    for i in 1..n {
        sum += z_i * (c[i] + z_n * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] += sum * z / (1.0 - z_n * z_n);
}

fn init_anticausal_reflect(c: &mut [f64], z: f64) {
    let n = c.len();
    c[n - 1] *= z / (z - 1.0);
}

fn spline_filter_line(c: &mut [f64], boundary: Boundary) {
    let n = c.len();
    if n <= 1 {
        return;
    }
    let z = CUBIC_POLE;
    let gain = (1.0 - z) * (1.0 - 1.0 / z);
    for v in c.iter_mut() {
        *v *= gain;
    }
    match boundary {
        Boundary::Constant => init_causal_mirror(c, z),
        Boundary::Nearest => init_causal_reflect(c, z),
    }
    for i in 1..n {
        c[i] += z * c[i - 1];
    }
    match boundary {
        Boundary::Constant => init_anticausal_mirror(c, z),
        Boundary::Nearest => init_anticausal_reflect(c, z),
    }
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

/// Prefilter every axis in turn (`scipy.ndimage.spline_filter`).
fn spline_filter(data: &mut ArrayD<f64>, boundary: Boundary) {
    for axis in 0..data.ndim() {
        for mut lane in data.lanes_mut(Axis(axis)) {
            let mut line: Vec<f64> = lane.iter().copied().collect();
            spline_filter_line(&mut line, boundary);
            for (dst, src) in lane.iter_mut().zip(line) {
                *dst = src;
            }
        }
    }
}

/// `np.pad(data, 12, mode="edge")`.
fn pad_edge(data: &ArrayD<f64>, pad: usize) -> ArrayD<f64> {
    let shape: Vec<usize> = data.shape().iter().map(|n| n + 2 * pad).collect();
    let src_shape = data.shape().to_vec();
    // The index is moved in place: collecting it into a `Vec` allocated once
    // per voxel.
    ArrayD::from_shape_fn(IxDyn(&shape), |mut idx| {
        for (k, n) in src_shape.iter().enumerate() {
            idx[k] = (idx[k] as i64 - pad as i64).clamp(0, *n as i64 - 1) as usize;
        }
        data[idx]
    })
}

fn cubic_weights(cc: f64) -> [f64; 4] {
    let x = cc - cc.floor();
    let y = x;
    let z = 1.0 - x;
    let mut w = [0.0; 4];
    w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
    w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
    w[0] = z * z * z / 6.0;
    w[3] = 1.0;
    for i in 0..3 {
        w[3] -= w[i];
    }
    w
}

fn mirror_index(idx: i64, len: i64) -> i64 {
    if len <= 1 {
        return 0;
    }
    let sz2 = 2 * len - 2;
    if idx < 0 {
        let v = sz2 * (-idx / sz2) + idx;
        if v <= 1 - len {
            v + sz2
        } else {
            -v
        }
    } else if idx > len - 1 {
        let mut v = idx - sz2 * (idx / sz2);
        if v >= len {
            v = sz2 - v;
        }
        v
    } else {
        idx
    }
}

/// Interpolate one prefiltered component at `(N, S)` coordinates.
///
/// Each point's footprint is 4 taps an axis: the taps' offsets into the
/// C-order coefficients are worked out once an axis, and the 4**S terms
/// summed in C order (last axis fastest), each the coefficient times its
/// axes' weights in axis order --- the arithmetic, in its order, of reading
/// every term through a full index.
fn spline_interpolate(coeffs: &ArrayD<f64>, coords: &Array2<f64>, boundary: Boundary, npad: usize) -> Vec<f64> {
    let dim = coeffs.ndim();
    let dims: Vec<i64> = coeffs.shape().iter().map(|n| *n as i64).collect();
    let coeffs = coeffs.as_standard_layout();
    let values = coeffs.as_slice().expect("standard layout");
    let strides = c_strides(coeffs.shape());
    let mut out = Vec::with_capacity(coords.nrows());
    let mut weights = vec![[0.0f64; 4]; dim];
    let mut taps = vec![[0usize; 4]; dim];
    let mut digits = vec![0usize; dim];
    for point in coords.outer_iter() {
        let mut constant = false;
        for axis in 0..dim {
            match cubic_axis(point[axis], dims[axis], strides[axis], boundary, npad) {
                Some(found) => (taps[axis], weights[axis]) = found,
                None => {
                    constant = true;
                    break;
                }
            }
        }
        out.push(if constant { 0.0 } else { spline_sum(values, &taps, &weights, &mut digits) });
    }
    out
}

/// One axis's four taps, as offsets into the C-order coefficients, and their
/// weights.
type CubicAxis = ([usize; 4], [f64; 4]);

/// One axis of a cubic sample at `coordinate`: its four taps, as offsets
/// `stride` apart a sample into the C-order coefficients of an axis `extent`
/// long (mirrored or clamped past its ends), and their weights; `None` where
/// the constant boundary gives the point no value.
fn cubic_axis(coordinate: f64, extent: i64, stride: usize, boundary: Boundary, npad: usize) -> Option<CubicAxis> {
    let mut cc = coordinate + npad as f64;
    if boundary == Boundary::Constant && (cc < 0.0 || cc > (extent - 1) as f64) {
        cc = -1.0;
    }
    if !(cc > -1.0 || boundary == Boundary::Nearest) {
        return None;
    }
    let start = cc.floor() as i64 - 1;
    let mut taps = [0usize; 4];
    for (digit, tap) in taps.iter_mut().enumerate() {
        let i = start + digit as i64;
        let mapped = if i < 0 || i >= extent {
            match boundary {
                Boundary::Constant => mirror_index(i, extent),
                Boundary::Nearest => i.clamp(0, extent - 1),
            }
        } else {
            i
        };
        *tap = mapped as usize * stride;
    }
    Some((taps, cubic_weights(cc)))
}

/// A point's 4**S terms summed in C order (last axis fastest), each the
/// coefficient times its axes' weights in axis order.
fn spline_sum(values: &[f64], taps: &[[usize; 4]], weights: &[[f64; 4]], digits: &mut [usize]) -> f64 {
    let dim = taps.len();
    let mut t = 0.0;
    if dim == 3 {
        let ([ta, tb, tc], [wa, wb, wc]) = ([taps[0], taps[1], taps[2]], [weights[0], weights[1], weights[2]]);
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    t += values[ta[a] + tb[b] + tc[c]] * wa[a] * wb[b] * wc[c];
                }
            }
        }
    } else {
        digits.fill(0);
        for _ in 0..4usize.pow(dim as u32) {
            let mut coeff = values[(0..dim).map(|axis| taps[axis][digits[axis]]).sum::<usize>()];
            for axis in 0..dim {
                coeff *= weights[axis][digits[axis]];
            }
            t += coeff;
            for axis in (0..dim).rev() {
                digits[axis] += 1;
                if digits[axis] < 4 {
                    break;
                }
                digits[axis] = 0;
            }
        }
    }
    t
}

/// Cubic interpolation of `(C, *spatial)` data at `(N, S)` continuous indices,
/// as `scipy.ndimage.map_coordinates(order=3)` computes it between the
/// outermost samples.
///
/// The field covers its voxel extent, `[-0.5, n - 0.5]` per axis (§10.4).
/// Under `zero` and `error` a point in the half-voxel margin between the
/// outermost sample and the extent's edge takes that sample's value, as
/// [`linear_sample`] gives it; SciPy's constant mode is zero beyond the
/// outermost samples, so such a point --- one `error` had admitted --- came
/// out with no displacement.  `nearest` interpolates the
/// edge-extended field, as SciPy does.
pub fn cubic_sample(field: &ArrayD<f64>, coords: &Array2<f64>, extrapolation: &str) -> Result<Array2<f64>> {
    check_field(field)?;
    check_extrapolation(extrapolation)?;
    let spatial: Vec<usize> = field.shape()[1..].to_vec();
    check_points(&spatial, coords)?;
    if extrapolation == "error" {
        refuse_outside(&inside_extent(&spatial, coords))?;
    }
    CubicCoefficients::new(field, extrapolation)?.sample(coords)
}

/// A field's cubic B-spline coefficients: each component prefiltered once,
/// under the boundary its extrapolation takes, for any number of
/// evaluations.  Prefiltering is a pass over the whole field per component,
/// and it was made again for every evaluation --- once per training item.
pub(crate) struct CubicCoefficients {
    spatial: Vec<usize>,
    extrapolation: String,
    boundary: Boundary,
    npad: usize,
    planes: Vec<ArrayD<f64>>,
}

impl std::fmt::Debug for CubicCoefficients {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CubicCoefficients")
            .field("spatial", &self.spatial)
            .field("extrapolation", &self.extrapolation)
            .field("components", &self.planes.len())
            .finish()
    }
}

impl CubicCoefficients {
    /// Prefilter every component of a `(C, *spatial)` field.
    pub(crate) fn new(field: &ArrayD<f64>, extrapolation: &str) -> Result<CubicCoefficients> {
        phase!(Interpolate);
        check_field(field)?;
        check_extrapolation(extrapolation)?;
        let boundary = if extrapolation == "nearest" { Boundary::Nearest } else { Boundary::Constant };
        let npad = if boundary == Boundary::Nearest { 12 } else { 0 };
        let planes = field
            .outer_iter()
            .map(|plane| {
                let mut coeffs = if npad > 0 { pad_edge(&plane.to_owned(), npad) } else { plane.to_owned() };
                spline_filter(&mut coeffs, boundary);
                coeffs
            })
            .collect();
        Ok(CubicCoefficients {
            spatial: field.shape()[1..].to_vec(),
            extrapolation: extrapolation.to_string(),
            boundary,
            npad,
            planes,
        })
    }

    /// Interpolate at `(N, S)` continuous indices, as [`cubic_sample`] does.
    pub(crate) fn sample(&self, coords: &Array2<f64>) -> Result<Array2<f64>> {
        let all: Vec<usize> = (0..self.planes.len()).collect();
        self.sample_components(coords, &all)
    }

    /// Interpolate only the listed components, in that order: `(N, len)`.
    pub(crate) fn sample_components(&self, coords: &Array2<f64>, components: &[usize]) -> Result<Array2<f64>> {
        phase!(Interpolate);
        let spatial = &self.spatial;
        check_points(spatial, coords)?;
        let inside = inside_extent(spatial, coords);
        if self.extrapolation == "error" {
            // SciPy has no raising mode; the domain check happens here, against
            // the bounds `linear_sample` uses, rather than folding into
            // constant-zero.
            refuse_outside(&inside)?;
        }
        // Clamped to the outermost samples, every point is interpolated;
        // `zero` then zeroes the points beyond the extent, as `linear_sample`
        // does.  `nearest` interpolates the edge-extended field everywhere,
        // and its points are clamped only to `EXACT_INTEGER`, where the
        // weights stop changing --- unclamped, a coordinate of 1e300
        // overflowed the indices.
        let clamped = Array2::from_shape_fn(coords.dim(), |(row, axis)| match self.boundary {
            Boundary::Constant => coords[[row, axis]].clamp(0.0, (spatial[axis] as f64 - 1.0).max(0.0)),
            Boundary::Nearest => coords[[row, axis]].clamp(-EXACT_INTEGER, EXACT_INTEGER),
        });
        let mut out = Array2::<f64>::zeros((clamped.nrows(), components.len()));
        for (c, component) in components.iter().enumerate() {
            let coeffs = &self.planes[*component];
            for (row, v) in spline_interpolate(coeffs, &clamped, self.boundary, self.npad).into_iter().enumerate() {
                out[[row, c]] = if self.extrapolation == "zero" && !inside[row] { 0.0 } else { v };
            }
        }
        Ok(out)
    }

    /// [`CubicCoefficients::sample_components`] at the C-order product of
    /// `axes`, as [`linear_sample_axes`] takes its points: each axis's taps
    /// and weights are worked out once a coordinate, and every sum is
    /// [`spline_interpolate`]'s, term by term.
    pub(crate) fn sample_components_axes(&self, axes: &[Vec<f64>], components: &[usize]) -> Result<Array2<f64>> {
        phase!(Interpolate);
        let spatial = &self.spatial;
        check_axes(spatial, axes)?;
        let lengths: Vec<usize> = axes.iter().map(Vec::len).collect();
        let n: usize = lengths.iter().product();
        let dim = spatial.len();
        // Whether each coordinate lies within its axis's extent, edges
        // included: a point is inside where all of its coordinates are.
        let within: Vec<Vec<bool>> = axes
            .iter()
            .zip(spatial)
            .map(|(coords, m)| coords.iter().map(|v| *v >= -0.5 && *v <= *m as f64 - 0.5).collect())
            .collect();
        let mut inside = Vec::with_capacity(n);
        let mut at = vec![0usize; dim];
        for _ in 0..n {
            inside.push((0..dim).all(|axis| within[axis][at[axis]]));
            count_up(&mut at, &lengths);
        }
        if self.extrapolation == "error" {
            refuse_outside(&inside)?;
        }
        let mut out = Array2::<f64>::zeros((n, components.len()));
        let Some(first) = self.planes.first() else { return Ok(out) };
        let dims: Vec<i64> = first.shape().iter().map(|n| *n as i64).collect();
        let strides = c_strides(first.shape());
        let found: Vec<Vec<Option<CubicAxis>>> = axes
            .iter()
            .enumerate()
            .map(|(axis, coords)| {
                coords
                    .iter()
                    .map(|v| {
                        let clamped = match self.boundary {
                            Boundary::Constant => v.clamp(0.0, (spatial[axis] as f64 - 1.0).max(0.0)),
                            Boundary::Nearest => v.clamp(-EXACT_INTEGER, EXACT_INTEGER),
                        };
                        cubic_axis(clamped, dims[axis], strides[axis], self.boundary, self.npad)
                    })
                    .collect()
            })
            .collect();
        let mut taps = vec![[0usize; 4]; dim];
        let mut weights = vec![[0.0f64; 4]; dim];
        let mut digits = vec![0usize; dim];
        for (c, component) in components.iter().enumerate() {
            let coeffs = self.planes[*component].as_standard_layout();
            let values = coeffs.as_slice().expect("standard layout");
            at.fill(0);
            for row in 0..n {
                let mut constant = false;
                for axis in 0..dim {
                    match found[axis][at[axis]] {
                        Some(tap) => (taps[axis], weights[axis]) = tap,
                        None => {
                            constant = true;
                            break;
                        }
                    }
                }
                let v = if constant { 0.0 } else { spline_sum(values, &taps, &weights, &mut digits) };
                out[[row, c]] = if self.extrapolation == "zero" && !inside[row] { 0.0 } else { v };
                count_up(&mut at, &lengths);
            }
        }
        Ok(out)
    }

    /// The bytes the coefficients take.
    fn nbytes(&self) -> usize {
        self.planes.iter().map(|plane| plane.len() * std::mem::size_of::<f64>()).sum()
    }
}

/// The most bytes of cubic coefficients transform handles keep between
/// evaluations, across the process.  A training worker keeps up to 32 samples
/// open (`medh5.torch`), each with its resolved transform, and the
/// coefficients of one image-sized field take ~175 MB: kept without a bound,
/// they could hold gigabytes per worker, where 2.0.1 held them only while it
/// evaluated.
pub(crate) const CUBIC_CACHE_BYTES: usize = 256 * 1024 * 1024;

static CUBIC_BUDGET: CoefficientBudget = CoefficientBudget::new(CUBIC_CACHE_BYTES);

/// Bytes of kept coefficients, counted against a limit.
pub(crate) struct CoefficientBudget {
    held: AtomicUsize,
    limit: usize,
}

impl CoefficientBudget {
    pub(crate) const fn new(limit: usize) -> CoefficientBudget {
        CoefficientBudget { held: AtomicUsize::new(0), limit }
    }

    /// Count `bytes` as held, if they fit within the limit.
    ///
    /// A compare-exchange loop: `fetch_update` is deprecated from Rust 1.99
    /// (renamed `try_update`), and `try_update` is newer than `rust-version`.
    fn reserve(&self, bytes: usize) -> bool {
        let mut held = self.held.load(Ordering::Relaxed);
        loop {
            let Some(total) = held.checked_add(bytes).filter(|total| *total <= self.limit) else {
                return false;
            };
            match self.held.compare_exchange_weak(held, total, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return true,
                Err(now) => held = now,
            }
        }
    }
}

/// Coefficients a cache keeps, counted against its budget until they drop.
struct Kept {
    coefficients: CubicCoefficients,
    bytes: usize,
    budget: &'static CoefficientBudget,
}

impl Drop for Kept {
    fn drop(&mut self) {
        self.budget.held.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// A transform handle's cubic coefficients: made at its first evaluation,
/// and kept for the next while the process-wide budget has room for them.
pub(crate) struct CubicCache {
    slot: OnceLock<Kept>,
    budget: &'static CoefficientBudget,
}

impl Default for CubicCache {
    fn default() -> CubicCache {
        CubicCache::within(&CUBIC_BUDGET)
    }
}

impl std::fmt::Debug for CubicCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CubicCache").field("kept_bytes", &self.slot.get().map(|kept| kept.bytes)).finish()
    }
}

impl CubicCache {
    pub(crate) const fn within(budget: &'static CoefficientBudget) -> CubicCache {
        CubicCache { slot: OnceLock::new(), budget }
    }

    /// Interpolate at `(N, S)` continuous indices, as [`cubic_sample`] does,
    /// through the kept coefficients, or ones `make` makes when none are.
    pub(crate) fn sample(
        &self,
        coords: &Array2<f64>,
        make: impl FnOnce() -> Result<CubicCoefficients>,
    ) -> Result<Array2<f64>> {
        if let Some(kept) = self.slot.get() {
            return kept.coefficients.sample(coords);
        }
        match self.keep(make()?) {
            Ok(kept) => kept.sample(coords),
            // Past the budget they serve this evaluation alone, and the next
            // makes them again, as every evaluation did before 2.0.2.
            Err(made) => made.sample(coords),
        }
    }

    /// Keep `made` if the budget has room for it, else hand it back.
    fn keep(&self, made: CubicCoefficients) -> std::result::Result<&CubicCoefficients, CubicCoefficients> {
        let bytes = made.nbytes();
        if !self.budget.reserve(bytes) {
            return Err(made);
        }
        if let Err(lost) = self.slot.set(Kept { coefficients: made, bytes, budget: self.budget }) {
            // Another thread filled the slot first, with the same
            // coefficients: this copy returns its bytes as it drops.
            drop(lost);
        }
        Ok(&self.slot.get().expect("the slot is filled").coefficients)
    }
}

/// Interpolate a field with the declared interpolation.
pub fn sample_field(
    field: &ArrayD<f64>,
    coords: &Array2<f64>,
    interpolation: &str,
    extrapolation: &str,
) -> Result<Array2<f64>> {
    match interpolation {
        "linear" => linear_sample(field, coords, extrapolation),
        "cubic" => cubic_sample(field, coords, extrapolation),
        other => Err(Error::invalid(format!("unknown interpolation {}", repr_str(other)))),
    }
}

/// `direction @ diag(spacing)`: index displacement to world displacement.
pub fn linear_part(grid: &Grid) -> Array2<f64> {
    let s = grid.spacing.len();
    let mut out = Array2::zeros((s, s));
    for r in 0..s {
        for c in 0..s {
            out[[r, c]] = grid.direction[[r, c]] * grid.spacing[c];
        }
    }
    out
}

/// Convert `(N, S)` displacement components to world units.
pub fn to_world_vectors(values: &Array2<f64>, grid: &Grid, vector_space: &str) -> Result<Array2<f64>> {
    match vector_space {
        "world" => Ok(values.clone()),
        "index" => Ok(values.dot(&linear_part(grid).t())),
        other => Err(Error::invalid(format!("unknown vector_space {}", repr_str(other)))),
    }
}

/// `np.gradient(values, axis=axis)` with unit spacing and first-order edges.
pub fn gradient(values: &ArrayD<f64>, axis: usize) -> Result<ArrayD<f64>> {
    let n = values.shape()[axis];
    if n < 2 {
        return Err(Error::Value(
            "Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required."
                .into(),
        ));
    }
    let mut out = ArrayD::<f64>::zeros(values.raw_dim());
    for (src, mut dst) in values.lanes(Axis(axis)).into_iter().zip(out.lanes_mut(Axis(axis))) {
        for i in 1..n - 1 {
            dst[i] = (src[i + 1] - src[i - 1]) / 2.0;
        }
        dst[0] = src[1] - src[0];
        dst[n - 1] = src[n - 1] - src[n - 2];
    }
    Ok(out)
}

/// `det(I + du/dx)` per voxel of the field grid; values `<= 0` mark folding.
pub fn jacobian_determinant(field: &ArrayD<f64>, grid: &Grid, vector_space: &str) -> Result<ArrayD<f64>> {
    // Differentiated along every axis of the grid, a field of another rank
    // was indexed past its own axes.
    let dim = grid.n_spatial();
    if field.ndim() != dim + 1 || field.shape()[0] != dim {
        return Err(Error::coded(
            "E503",
            format!("a field of shape {} is not (S, *spatial) for a {dim}-D grid", repr_int_tuple(field.shape())),
        ));
    }
    let linear = linear_part(grid);
    let world = match vector_space {
        "world" => field.clone(),
        "index" => {
            let mut out = ArrayD::<f64>::zeros(field.raw_dim());
            for r in 0..dim {
                let mut lane = out.index_axis_mut(Axis(0), r);
                for c in 0..dim {
                    let l = linear[[r, c]];
                    lane.zip_mut_with(&field.index_axis(Axis(0), c), |o, v| *o += l * v);
                }
            }
            out
        }
        other => return Err(Error::invalid(format!("unknown vector_space {}", repr_str(other)))),
    };
    let inverse_linear = inv(&linear)?;
    let mut gradients = Vec::with_capacity(dim * dim);
    for c in 0..dim {
        let comp = world.index_axis(Axis(0), c).to_owned();
        for a in 0..dim {
            gradients.push(gradient(&comp, a)?);
        }
    }
    let spatial: Vec<usize> = field.shape()[1..].to_vec();
    let mut out = ArrayD::<f64>::zeros(IxDyn(&spatial));
    let mut jac = Array2::<f64>::zeros((dim, dim));
    for (flat, slot) in out.iter_mut().enumerate() {
        for c in 0..dim {
            for b in 0..dim {
                let mut v = 0.0;
                for a in 0..dim {
                    let g = gradients[c * dim + a].as_slice().map(|s| s[flat]).unwrap_or(0.0);
                    v += g * inverse_linear[[a, b]];
                }
                jac[[c, b]] = v + if c == b { 1.0 } else { 0.0 };
            }
        }
        *slot = det(&jac);
    }
    Ok(out)
}

/// Fraction of voxels where the transform folds (`det <= 0`).
pub fn folding_fraction(determinants: &ArrayD<f64>) -> f64 {
    if determinants.is_empty() {
        return 0.0;
    }
    determinants.iter().filter(|v| **v <= 0.0).count() as f64 / determinants.len() as f64
}

/// Target registration error summary (§10.6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tre {
    pub mean: f64,
    pub median: f64,
    pub max: f64,
    pub n: usize,
}

/// `np.median` of a slice.
pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// TRE `‖T(p_F) − p_M‖` from already-warped fixed points.
///
/// With no landmark pairs there is no error to summarise: `mean`, `median`
/// and `max` are NaN rather than a 0.0 that reads as a perfect registration.
/// Weights are finite, non-negative and not all zero --- a weighted mean is
/// meaningless otherwise, and zero weights reported a mean of 0.0 for any
/// error.
pub fn tre_from_warped(warped: &Array2<f64>, moving: &Array2<f64>, weights: Option<&[f64]>) -> Result<Tre> {
    if warped.dim() != moving.dim() {
        return Err(Error::invalid(format!(
            "landmark sets disagree: {:?} vs {:?}; §10.6 requires equal N and matching row order",
            warped.dim(),
            moving.dim()
        )));
    }
    let errors: Vec<f64> = warped
        .outer_iter()
        .zip(moving.outer_iter())
        .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt())
        .collect();
    // One weight per landmark: `zip` dropped the weights or landmarks past the
    // shorter list while the total summed every weight, so two landmarks with
    // one weight scored the first alone --- 0.0 for a 10 mm miss.
    let w: Vec<f64> = match weights {
        Some(w) if w.len() != errors.len() => {
            return Err(Error::invalid(format!(
                "{} weight(s) for {} landmark(s); TRE takes one weight per landmark (§10.6)",
                w.len(),
                errors.len()
            )));
        }
        Some(w) => w.to_vec(),
        None => vec![1.0; errors.len()],
    };
    let total: f64 = w.iter().sum();
    if !errors.is_empty() && (w.iter().any(|v| !v.is_finite() || *v < 0.0) || total <= 0.0) {
        return Err(Error::invalid(format!(
            "weights [{}] are not a weighting: each must be finite and non-negative, and their total positive (§10.6)",
            w.iter().map(|v| crate::json::py_float(*v)).collect::<Vec<_>>().join(", ")
        )));
    }
    if errors.is_empty() {
        return Ok(Tre { mean: f64::NAN, median: f64::NAN, max: f64::NAN, n: 0 });
    }
    let weighted: f64 = errors.iter().zip(&w).map(|(e, w)| e * w).sum();
    Ok(Tre {
        mean: weighted / total,
        median: median(&errors),
        max: errors.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        n: errors.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_extrapolation_modes() {
        let field = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0);
        let outside = Array2::from_shape_vec((1, 2), vec![-5.0, -5.0]).unwrap();
        assert_eq!(linear_sample(&field, &outside, "zero").unwrap()[[0, 0]], 0.0);
        assert_eq!(linear_sample(&field, &outside, "nearest").unwrap()[[0, 0]], 1.0);
        assert!(linear_sample(&field, &outside, "error").unwrap_err().to_string().contains("outside"));
        assert!(linear_sample(&field, &outside, "wing-it").is_err());
    }

    #[test]
    fn cubic_matches_scipy_map_coordinates() {
        // i = np.arange(20.0); field = (i * i * 0.37 + i / 3.0).reshape(1, 4, 5)
        let field = ArrayD::from_shape_fn(IxDyn(&[1, 4, 5]), |i| {
            let v = (i[1] * 5 + i[2]) as f64;
            v * v * 0.37 + v / 3.0
        });
        let coords = Array2::from_shape_vec((3, 2), vec![1.3, 2.7, 0.2, 0.1, 2.9, 3.95]).unwrap();
        let constant = cubic_sample(&field, &coords, "zero").unwrap();
        let nearest = cubic_sample(&field, &coords, "nearest").unwrap();
        // scipy.ndimage.map_coordinates(field[0], coords.T, order=3, mode=...)
        let want_constant = [32.707648506666686, 0.6104213523809539, 138.73197259642862];
        let want_nearest = [33.326160022592546, 1.16275434318072, 135.63402382870262];
        for i in 0..3 {
            assert_eq!(constant[[i, 0]], want_constant[i]);
            assert_eq!(nearest[[i, 0]], want_nearest[i]);
        }
        let ones = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0);
        let outside = Array2::from_shape_vec((1, 2), vec![-5.0, -5.0]).unwrap();
        assert_eq!(cubic_sample(&ones, &outside, "zero").unwrap()[[0, 0]], 0.0);
        assert!((cubic_sample(&ones, &outside, "nearest").unwrap()[[0, 0]] - 1.0).abs() < 1e-6);
        assert!(cubic_sample(&ones, &outside, "error").is_err());
    }

    #[test]
    fn s10_4_coefficients_past_the_budget_are_not_kept_and_answer_alike() {
        let field = ArrayD::from_shape_fn(IxDyn(&[2, 4, 5]), |i| {
            let v = (i[0] * 20 + i[1] * 5 + i[2]) as f64;
            v * v * 0.37 + v / 3.0
        });
        let coords = Array2::from_shape_vec((3, 2), vec![1.3, 2.7, -4.0, 0.1, 2.9, 3.95]).unwrap();
        let make = || CubicCoefficients::new(&field, "nearest");
        let alone = cubic_sample(&field, &coords, "nearest").unwrap();
        let bytes = make().unwrap().nbytes();
        // Room for one field's coefficients and half of another's.
        let budget: &'static CoefficientBudget = Box::leak(Box::new(CoefficientBudget::new(bytes + bytes / 2)));
        let held = || budget.held.load(Ordering::Relaxed);
        let first = CubicCache::within(budget);
        assert_eq!(first.sample(&coords, make).unwrap(), alone);
        assert!(first.slot.get().is_some());
        assert_eq!(held(), bytes);
        // Past what remains, they are made for each evaluation and kept for none.
        let second = CubicCache::within(budget);
        for _ in 0..2 {
            assert_eq!(second.sample(&coords, make).unwrap(), alone);
            assert!(second.slot.get().is_none());
            assert_eq!(held(), bytes);
        }
        // Dropped coefficients return their bytes, and the room is used again.
        drop(first);
        assert_eq!(held(), 0);
        assert_eq!(second.sample(&coords, make).unwrap(), alone);
        assert!(second.slot.get().is_some());
        assert_eq!(held(), bytes);
        drop(second);
        assert_eq!(held(), 0);
        // A copy that lost the race to fill a slot gives its bytes back.
        let racing: &'static CoefficientBudget = Box::leak(Box::new(CoefficientBudget::new(2 * bytes)));
        let raced = CubicCache::within(racing);
        for _ in 0..2 {
            assert!(raced.keep(make().unwrap()).is_ok());
            assert_eq!(racing.held.load(Ordering::Relaxed), bytes);
        }
        assert_eq!(raced.sample(&coords, || unreachable!("the coefficients are kept")).unwrap(), alone);
        drop(raced);
        assert_eq!(racing.held.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn s10_4_the_margin_takes_the_outermost_samples_value_as_linear_gives_it() {
        // Constant mode is zero beyond the outermost samples, so a point `error`
        // admitted --- half a voxel inside the extent --- had no displacement.
        let field = ArrayD::from_shape_fn(IxDyn(&[1, 3, 4]), |i| {
            let v = (i[1] * 4 + i[2]) as f64;
            v * v * 0.37 + v / 3.0
        });
        let margin = Array2::from_shape_vec((3, 2), vec![-0.5, 1.0, 2.0, 3.5, 2.4, -0.3]).unwrap();
        let samples = Array2::from_shape_vec((3, 2), vec![0.0, 1.0, 2.0, 3.0, 2.0, 0.0]).unwrap();
        for mode in ["zero", "error"] {
            let cubic = cubic_sample(&field, &margin, mode).unwrap();
            let at_samples = cubic_sample(&field, &samples, mode).unwrap();
            let linear = linear_sample(&field, &margin, mode).unwrap();
            for row in 0..3 {
                assert_eq!(cubic[[row, 0]], at_samples[[row, 0]], "{mode} row {row}");
                assert!((cubic[[row, 0]] - linear[[row, 0]]).abs() < 1e-9, "{mode} row {row}");
                assert!(cubic[[row, 0]] != 0.0, "{mode} row {row}");
            }
        }
        // Beyond the extent `zero` is still zero and `error` still refuses; the
        // samples themselves are unchanged.
        let beyond = Array2::from_shape_vec((1, 2), vec![-0.6, 1.0]).unwrap();
        assert_eq!(cubic_sample(&field, &beyond, "zero").unwrap()[[0, 0]], 0.0);
        assert!(cubic_sample(&field, &beyond, "error").is_err());
        let interior = Array2::from_shape_vec((1, 2), vec![1.3, 2.7]).unwrap();
        let before = cubic_sample(&field, &interior, "zero").unwrap()[[0, 0]];
        assert_eq!(cubic_sample(&field, &interior, "error").unwrap()[[0, 0]], before);
    }

    #[test]
    fn samplers_refuse_points_of_another_dimensionality() {
        // Fewer coordinates than axes panicked; more were ignored.
        let field = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0);
        for columns in [1, 3] {
            let points = Array2::zeros((2, columns));
            for mode in EXTRAPOLATIONS {
                let linear = linear_sample(&field, &points, mode).unwrap_err().to_string();
                let cubic = cubic_sample(&field, &points, mode).unwrap_err().to_string();
                for message in [linear, cubic] {
                    assert!(message.contains(&format!("{columns} coordinate(s)")), "{message}");
                }
            }
            assert!(check_points(&[4, 4], &points).is_err());
        }
        assert!(check_points(&[4, 4], &Array2::zeros((2, 2))).is_ok());
    }

    #[test]
    fn tre_takes_one_weight_per_landmark() {
        // Errors of 0 and 10: one weight scored the first landmark alone (0.0),
        // three deflated the mean to 3.3.
        let moving = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0]).unwrap();
        let warped = Array2::zeros((2, 3));
        for weights in [vec![1.0], vec![1.0, 1.0, 1.0]] {
            let message = tre_from_warped(&warped, &moving, Some(&weights)).unwrap_err().to_string();
            assert!(message.contains("one weight per landmark"), "{message}");
        }
        let tre = tre_from_warped(&warped, &moving, Some(&[3.0, 1.0])).unwrap();
        assert!((tre.mean - 2.5).abs() < 1e-12);
        assert!((tre_from_warped(&warped, &moving, None).unwrap().mean - 5.0).abs() < 1e-12);
    }

    #[test]
    fn s10_6_no_landmarks_is_no_tre_and_a_weighting_is_one() {
        let none = Array2::<f64>::zeros((0, 3));
        let tre = tre_from_warped(&none, &none, None).unwrap();
        assert!(tre.mean.is_nan() && tre.median.is_nan() && tre.max.is_nan());
        assert_eq!(tre.n, 0);
        // Errors of 0 and 10: zero weights scored 0.0, and a negative one
        // took the mean below every error.
        let moving = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0]).unwrap();
        let warped = Array2::zeros((2, 3));
        for weights in [[0.0, 0.0], [1.0, -0.5], [1.0, f64::NAN], [f64::INFINITY, 1.0]] {
            let message = tre_from_warped(&warped, &moving, Some(&weights)).unwrap_err().to_string();
            assert!(message.contains("not a weighting"), "{message}");
        }
        assert_eq!(tre_from_warped(&warped, &moving, Some(&[0.0, 1.0])).unwrap().mean, 10.0);
    }

    #[test]
    fn s10_points_are_grouped_by_their_last_axis() {
        // Six values are three 2-D points, not two 3-D ones.
        let points = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(as_points(&points, 3).unwrap_err().to_string().contains("got shape (3, 2)"));
        assert_eq!(as_points(&points, 2).unwrap().dim(), (3, 2));
        let one = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(as_points(&one, 3).unwrap().dim(), (1, 3));
        assert!(as_points(&ArrayD::from_elem(IxDyn(&[]), 1.0), 3).is_err());
    }

    #[test]
    fn s10_4_a_field_without_a_sample_on_every_axis_is_refused() {
        let points = Array2::zeros((1, 2));
        for shape in [&[][..], &[4], &[1, 0, 4]] {
            let field = ArrayD::zeros(IxDyn(shape));
            for mode in EXTRAPOLATIONS {
                assert!(linear_sample(&field, &points, mode).unwrap_err().to_string().contains("(C, *spatial)"));
                assert!(cubic_sample(&field, &points, mode).unwrap_err().to_string().contains("(C, *spatial)"));
            }
        }
        let grid = Grid::from_spec(crate::geometry::grid::GridSpec {
            grid_id: "g".into(),
            shape: vec![4, 4, 4],
            spacing: vec![1.0, 1.0, 1.0],
            ..Default::default()
        })
        .unwrap();
        for shape in [&[3, 4, 4][..], &[2, 4, 4, 4], &[3, 4, 4, 4, 4]] {
            let err = jacobian_determinant(&ArrayD::zeros(IxDyn(shape)), &grid, "world").unwrap_err();
            assert_eq!(err.code(), Some("E503"), "{shape:?}");
        }
    }

    #[test]
    fn s10_4_a_coordinate_of_any_size_samples_the_edge() {
        // `nearest` cubic sampling computed indices from the coordinate
        // itself, and 1e300 overflowed them.  Far enough out, every
        // coordinate takes the edge-extended field's value there.
        let field = ArrayD::from_shape_fn(IxDyn(&[1, 4, 5]), |i| (i[1] * 5 + i[2]) as f64);
        let far = |v: f64| Array2::from_shape_vec((1, 2), vec![v, 2.0]).unwrap();
        for (huge, near) in [(1e300, 1e3), (-1e300, -1e3), (f64::INFINITY, 1e3), (EXACT_INTEGER * 4.0, 1e3)] {
            let at = cubic_sample(&field, &far(huge), "nearest").unwrap()[[0, 0]];
            assert_eq!(at, cubic_sample(&field, &far(near), "nearest").unwrap()[[0, 0]], "{huge}");
        }
    }

    #[test]
    fn gradients_follow_numpy() {
        let v = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 4.0, 7.0]).unwrap();
        assert_eq!(gradient(&v, 0).unwrap().iter().copied().collect::<Vec<_>>(), vec![1.0, 1.5, 2.5, 3.0]);
    }
}
