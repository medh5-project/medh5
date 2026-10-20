//! `medh5.transforms`: encoders, field sampling and the frame resolver
//! (spec §10).
//!
//! Every transform maps points from `from_frame` to `to_frame` ---
//! `x_M = T(x_F)`, the ITK convention --- and the resolver returns `None`
//! rather than inventing a path between frames no transform relates.

use indexmap::IndexMap;
use ndarray::{Array2, ArrayD, IxDyn};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use medh5::array::DType;
use medh5::transforms::apply;
use medh5::transforms::encode as enc;
use medh5::transforms::model;
use medh5::transforms::resample as rs;
use medh5::transforms::resolve;

use crate::annotations::AnnotationPayload;
use crate::convert::{array_to_py, attr_to_py, dtype_arg, f64_array, json_to_py, py_to_attr, py_to_nd, strings};
use crate::errors::{path_error, R};
use crate::geometry::grid_arg;
use crate::reader::TransformHandle;

fn points2(obj: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    let a = f64_array(obj)?;
    let dim = a.shape().last().copied().unwrap_or(0);
    apply::as_points(&a, dim).map_err(|e| crate::errors::BindError::from(e).into())
}

// -- encoders ---------------------------------------------------------------------------

/// Return the payload of an identity transform, which stores nothing but its endpoints.
#[pyfunction]
fn encode_identity() -> AnnotationPayload {
    AnnotationPayload::wrap(enc::encode_identity())
}

/// Pack a homogeneous world-to-world `(S+1, S+1)` matrix as an `affine` transform payload (§10.3).
///
/// The last row must be `[0 ... 0 1]`, every entry finite and the linear part
/// non-singular (E504). Index-space affines are not storable: compose with the
/// grids' affines first.
#[pyfunction]
fn encode_affine(matrix: &Bound<'_, PyAny>) -> R<AnnotationPayload> {
    Ok(AnnotationPayload::wrap(enc::encode_affine(&f64_array(matrix)?)?))
}

#[pyfunction]
#[pyo3(signature = (field, *, field_grid, vector_space="world", interpolation="linear", extrapolation="zero", dtype=None))]
fn encode_displacement(
    py: Python<'_>,
    field: &Bound<'_, PyAny>,
    field_grid: &str,
    vector_space: &str,
    interpolation: &str,
    extrapolation: &str,
    dtype: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    let dtype = match dtype.filter(|d| !d.is_none()) {
        Some(d) => dtype_arg(d)?,
        None => DType::F32,
    };
    let field = py_to_nd(field)?;
    let payload = py.detach(move || {
        enc::encode_displacement(&field, field_grid, vector_space, interpolation, extrapolation, dtype)
    })?;
    Ok(AnnotationPayload::wrap(payload))
}

/// Pack `(S, *cp_shape)` B-spline control-point coefficients as a `bspline` payload (§10.5).
///
/// `cp_grid` names the grid of the control points; `order` is 1 or 3 and
/// `vector_space` `world` or `index` (E502 otherwise).
#[pyfunction]
#[pyo3(signature = (control_points, *, cp_grid, order=model::DEFAULT_ORDER, vector_space="world"))]
fn encode_bspline(
    control_points: &Bound<'_, PyAny>,
    cp_grid: &str,
    order: i64,
    vector_space: &str,
) -> R<AnnotationPayload> {
    Ok(AnnotationPayload::wrap(enc::encode_bspline(&f64_array(control_points)?, cp_grid, order, vector_space)?))
}

/// Declare a `composite` transform: two or more component transform ids, applied in order (§10.5).
///
/// `T(x) = T_n(... T_1(x))`: the first component is applied first. Fewer than
/// two components raise E501.
#[pyfunction]
fn encode_composite(components: &Bound<'_, PyAny>) -> R<AnnotationPayload> {
    Ok(AnnotationPayload::wrap(enc::encode_composite(&strings(components)?)?))
}

/// The B-spline basis weights at `t`: `(order + 1, *t.shape)`.
#[pyfunction]
fn basis<'py>(py: Python<'py>, order: i64, t: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    // The order sizes the output, so it is refused (E502) before anything is
    // allocated by it: `basis(2**31, 0.5)` asked for 16 GB and aborted.
    model::basis(order, 0.0)?;
    let ts = f64_array(t)?;
    let n = (order + 1) as usize;
    let mut shape = vec![n];
    shape.extend_from_slice(ts.shape());
    let mut out = ArrayD::<f64>::zeros(IxDyn(&shape));
    for (i, value) in ts.iter().enumerate() {
        let w = model::basis(order, *value)?;
        for (k, weight) in w.iter().enumerate() {
            out.as_slice_mut().expect("standard layout")[k * ts.len() + i] = *weight;
        }
    }
    Ok(array_to_py(py, out))
}

/// Return `transform_id` if it is a valid object id, else raise E003.
///
/// An id matches `[A-Za-z0-9_.-]{1,128}` and is not a reserved name (§2.3).
#[pyfunction]
fn check_transform_id(transform_id: &str) -> R<String> {
    Ok(model::check_transform_id(transform_id)?.to_string())
}

// -- sampling -----------------------------------------------------------------------------

/// Return a boolean array saying which points lie within a lattice's voxel extent, edges included.
///
/// `spatial` is the lattice's spatial shape and `points` `(..., S)` continuous
/// indices; the extent is `[-0.5, n - 0.5]` along an axis of `n` samples
/// (§10.4).
#[pyfunction]
fn inside_extent<'py>(py: Python<'py>, spatial: Vec<usize>, points: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let points = points2(points)?;
    let inside = py.detach(move || -> medh5::Result<Vec<bool>> {
        apply::check_points(&spatial, &points)?;
        Ok(apply::inside_extent(&spatial, &points))
    })?;
    Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[inside.len()]), inside)?))
}

/// Raise `MEDH5ValidationError` if any of the `inside` flags is false: `extrapolation='error'`.
#[pyfunction]
fn refuse_outside(py: Python<'_>, inside: &Bound<'_, PyAny>) -> R<()> {
    let flags = crate::convert::bool_array(inside)?;
    let large = flags.len() >= crate::convert::DETACH_ROWS;
    Ok(crate::convert::detach_if(py, large, move || {
        let flags: Vec<bool> = flags.iter().copied().collect();
        apply::refuse_outside(&flags)
    })?)
}

fn field_of(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<f64>> {
    f64_array(obj)
}

/// Interpolate `(C, *spatial)` data multilinearly at `(N, S)` continuous indices: an `(N, C)` array.
///
/// Beyond the outermost samples `extrapolation` decides: `zero`, `nearest`
/// (clamped) or `error`, which refuses a point outside the voxel extent
/// (§10.4).
#[pyfunction]
#[pyo3(signature = (field, coords, *, extrapolation="zero"))]
fn linear_sample<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    extrapolation: &str,
) -> R<Bound<'py, PyAny>> {
    let (field, coords) = (field_of(field)?, points2(coords)?);
    let sampled = py.detach(move || apply::linear_sample(&field, &coords, extrapolation))?;
    Ok(array_to_py(py, sampled.into_dyn()))
}

/// Interpolate `(C, *spatial)` data with cubic splines at `(N, S)` continuous indices: `(N, C)`.
///
/// Between the outermost samples it computes what
/// `scipy.ndimage.map_coordinates(order=3)` does. Under `zero` and `error`, a
/// point in the half-voxel margin takes the outermost sample's value; `nearest`
/// interpolates the edge-extended field (§10.4).
#[pyfunction]
#[pyo3(signature = (field, coords, *, extrapolation="zero"))]
fn cubic_sample<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    extrapolation: &str,
) -> R<Bound<'py, PyAny>> {
    let (field, coords) = (field_of(field)?, points2(coords)?);
    let sampled = py.detach(move || apply::cubic_sample(&field, &coords, extrapolation))?;
    Ok(array_to_py(py, sampled.into_dyn()))
}

/// Interpolate a field at `(N, S)` continuous indices with the declared `interpolation` (§10.4).
///
/// `interpolation` is `linear` or `cubic`, as `linear_sample` and
/// `cubic_sample` compute it.
#[pyfunction]
#[pyo3(signature = (field, coords, *, interpolation="linear", extrapolation="zero"))]
fn sample_field<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    coords: &Bound<'py, PyAny>,
    interpolation: &str,
    extrapolation: &str,
) -> R<Bound<'py, PyAny>> {
    let (field, coords) = (field_of(field)?, points2(coords)?);
    let sampled = py.detach(move || apply::sample_field(&field, &coords, interpolation, extrapolation))?;
    Ok(array_to_py(py, sampled.into_dyn()))
}

/// Return a grid's `direction @ diag(spacing)`: the `(S, S)` map from index to world displacement.
#[pyfunction]
fn linear_part<'py>(py: Python<'py>, grid: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(array_to_py(py, apply::linear_part(&grid_arg(grid)?).into_dyn()))
}

/// Convert `(N, S)` displacement vectors in `vector_space` on `grid` to world units (§10.4).
///
/// `world` vectors are returned as given; `index` vectors, in voxels along the
/// grid's axes, are multiplied by the grid's `linear_part`.
#[pyfunction]
fn to_world_vectors<'py>(
    py: Python<'py>,
    vectors: &Bound<'py, PyAny>,
    grid: &Bound<'py, PyAny>,
    vector_space: &str,
) -> R<Bound<'py, PyAny>> {
    let (vectors, grid) = (points2(vectors)?, grid_arg(grid)?);
    let world = py.detach(move || apply::to_world_vectors(&vectors, &grid, vector_space))?;
    Ok(array_to_py(py, world.into_dyn()))
}

/// Return `det(I + du/dx)` per voxel of a displacement field on `grid`; values `<= 0` mark folding.
///
/// `field` is `(S, *spatial)` on `grid` (E503 otherwise), its vectors in
/// `vector_space` (§10.4). The result has the grid's spatial shape.
#[pyfunction]
#[pyo3(signature = (field, grid, *, vector_space="world"))]
fn jacobian_determinant<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    grid: &Bound<'py, PyAny>,
    vector_space: &str,
) -> R<Bound<'py, PyAny>> {
    let (field, grid) = (field_of(field)?, grid_arg(grid)?);
    Ok(array_to_py(py, py.detach(move || apply::jacobian_determinant(&field, &grid, vector_space))?))
}

/// Return the fraction of Jacobian determinants that are `<= 0`: where the transform folds.
///
/// An empty array gives 0.0.
#[pyfunction]
fn folding_fraction(py: Python<'_>, determinants: &Bound<'_, PyAny>) -> R<f64> {
    let determinants = f64_array(determinants)?;
    Ok(py.detach(move || apply::folding_fraction(&determinants)))
}

/// TRE `‖T(p_F) − p_M‖` over world landmarks with matching row order
/// (§10.6): `{mean, median, max, n}`.
#[pyfunction]
#[pyo3(signature = (transform, fixed_points, moving_points, *, weights=None))]
fn target_registration_error<'py>(
    py: Python<'py>,
    transform: &Bound<'py, PyAny>,
    fixed_points: &Bound<'py, PyAny>,
    moving_points: &Bound<'py, PyAny>,
    weights: Option<Vec<f64>>,
) -> R<Bound<'py, PyDict>> {
    let fixed = f64_array(fixed_points)?;
    let moving = f64_array(moving_points)?;
    if fixed.shape() != moving.shape() {
        return Err(medh5::Error::invalid(format!(
            "landmark sets disagree: {} vs {}; §10.6 requires equal N and matching row order",
            medh5::json::repr_int_tuple(fixed.shape()),
            medh5::json::repr_int_tuple(moving.shape())
        ))
        .into());
    }
    let warped = transform.call_method1("transform_points", (array_to_py(py, fixed.clone()),))?;
    let warped = points2(&warped)?;
    let dim = moving.shape().last().copied().unwrap_or(0);
    let moving = apply::as_points(&moving, dim)?;
    let tre = apply::tre_from_warped(&warped, &moving, weights.as_deref())?;
    let out = PyDict::new(py);
    out.set_item("mean", tre.mean)?;
    out.set_item("median", tre.median)?;
    out.set_item("max", tre.max)?;
    out.set_item("n", tre.n)?;
    Ok(out)
}

// -- header and graph ------------------------------------------------------------------------

/// The attribute header every transform carries (spec §10.1).
#[pyclass(module = "medh5.transforms", name = "TransformHeader", skip_from_py_object, frozen)]
pub struct TransformHeader {
    pub inner: model::TransformHeader,
}

impl TransformHeader {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &[
        "kind",
        "from_frame",
        "to_frame",
        "units",
        "from_grid",
        "to_grid",
        "invertible",
        "inverse_id",
        "prov",
        "metrics",
        "extra",
    ];
}

#[pymethods]
impl TransformHeader {
    /// The header a transform group carries (`sample.root["transforms/x"]`).
    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>, group: &Bound<'_, PyAny>) -> R<Self> {
        let group = crate::integrity::group_of(group)?;
        let inner = crate::h5(cls.py(), || medh5::transforms::model::TransformHeader::read(&group))?;
        Ok(TransformHeader { inner })
    }

    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    /// `copy.replace(obj, **changes)`.
    #[pyo3(signature = (**changes))]
    fn __replace__(slf: &Bound<'_, Self>, changes: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Py<PyAny>> {
        crate::values::dataclass_replace(slf.as_any(), changes)
    }

    #[new]
    #[pyo3(signature = (kind, from_frame, to_frame, units="mm", from_grid=None, to_grid=None, invertible=None,
        inverse_id=None, prov=None, metrics=None, extra=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: &str,
        from_frame: &str,
        to_frame: &str,
        units: &str,
        from_grid: Option<String>,
        to_grid: Option<String>,
        invertible: Option<bool>,
        inverse_id: Option<String>,
        prov: Option<String>,
        metrics: Option<String>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let mut h = model::TransformHeader::new(kind, from_frame, to_frame)?;
        h.units = units.to_string();
        h.from_grid = from_grid;
        h.to_grid = to_grid;
        h.invertible = invertible;
        h.inverse_id = inverse_id;
        h.prov = prov;
        h.metrics = metrics;
        if let Some(e) = extra.filter(|e| !e.is_none()) {
            for item in e.call_method0("items")?.try_iter()? {
                let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
                h.extra.push((k, py_to_attr(&v)?));
            }
        }
        Ok(TransformHeader { inner: h })
    }
    #[getter]
    fn kind(&self) -> &str {
        &self.inner.kind
    }
    #[getter(from_frame)]
    fn source_frame(&self) -> &str {
        &self.inner.from_frame
    }
    #[getter]
    fn to_frame(&self) -> &str {
        &self.inner.to_frame
    }
    #[getter]
    fn units(&self) -> &str {
        &self.inner.units
    }
    #[getter(from_grid)]
    fn source_grid(&self) -> Option<&str> {
        self.inner.from_grid.as_deref()
    }
    #[getter]
    fn to_grid(&self) -> Option<&str> {
        self.inner.to_grid.as_deref()
    }
    #[getter]
    fn invertible(&self) -> Option<bool> {
        self.inner.invertible
    }
    #[getter]
    fn inverse_id(&self) -> Option<&str> {
        self.inner.inverse_id.as_deref()
    }
    #[getter]
    fn prov(&self) -> Option<&str> {
        self.inner.prov.as_deref()
    }
    #[getter]
    fn metrics(&self) -> Option<&str> {
        self.inner.metrics.as_deref()
    }
    #[getter]
    fn extra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.inner.extra {
            out.set_item(k, attr_to_py(py, v)?)?;
        }
        Ok(out)
    }
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.inner.attrs() {
            out.set_item(k, attr_to_py(py, &v)?)?;
        }
        Ok(out)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<TransformHeader>().map(|o| o.get().inner == self.inner).unwrap_or(false)
    }
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let names = [
            "kind",
            "from_frame",
            "to_frame",
            "units",
            "from_grid",
            "to_grid",
            "invertible",
            "inverse_id",
            "prov",
            "metrics",
            "extra",
        ];
        let fields = names.iter().map(|n| Ok((*n, slf.getattr(*n)?))).collect::<PyResult<Vec<_>>>()?;
        crate::values::dataclass_repr("TransformHeader", &fields)
    }
    /// Pickling and `copy`: the fields, as the constructor takes them.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_fields(slf.as_any(), None, Self::FIELDS)
    }
}

/// `{transform_id: engine transform}` from a mapping of facades or handles.
fn transforms_arg(obj: &Bound<'_, PyAny>) -> PyResult<IndexMap<String, model::Transform>> {
    let mut out = IndexMap::new();
    for item in obj.call_method0("items")?.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        out.insert(k, handle_of(&v)?.get().inner.as_ref().clone());
    }
    Ok(out)
}

pub fn handle_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, TransformHandle>> {
    match obj.cast::<TransformHandle>() {
        Ok(h) => Ok(h.clone()),
        Err(_) => Ok(obj.getattr("_handle")?.cast_into::<TransformHandle>()?),
    }
}

/// Frame -> frames one hop away, as the resolver would walk them.
#[pyfunction]
fn frame_graph<'py>(py: Python<'py>, transforms: &Bound<'py, PyAny>) -> R<Bound<'py, PyDict>> {
    let transforms = transforms_arg(transforms)?;
    let graph = crate::h5(py, || resolve::frame_graph(&transforms))?;
    let out = PyDict::new(py);
    for (k, v) in graph {
        out.set_item(k, PyList::new(py, v)?)?;
    }
    Ok(out)
}

/// The transform relating two frames, or `None` when no path exists.
#[pyfunction]
fn resolve_between(
    py: Python<'_>,
    transforms: &Bound<'_, PyAny>,
    from_frame: &str,
    to_frame: &str,
) -> R<Option<TransformHandle>> {
    let transforms = transforms_arg(transforms)?;
    let found = crate::h5(py, || resolve::resolve_between(&transforms, from_frame, to_frame))?;
    Ok(found.map(TransformHandle::wrap))
}

/// The frames of every grid of one timepoint, in grid order.
#[pyfunction]
fn frames_of_timepoint<'py>(py: Python<'py>, grids: &Bound<'py, PyAny>, timepoint: &str) -> R<Bound<'py, PyTuple>> {
    let mut map = IndexMap::new();
    for item in grids.call_method0("items")?.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        map.insert(k, grid_arg(&v)?);
    }
    Ok(PyTuple::new(py, resolve::frames_of_timepoint(&map, timepoint))?)
}

// -- resampling ---------------------------------------------------------------------------

/// Resample images of a sample onto another grid, recorded as one `resample`
/// activity; the report as plain data (`medh5.transforms.resample` builds its
/// `ResampleReport` from it).
#[pyfunction]
#[pyo3(signature = (
    path, image_ids, *, onto=None, spacing=None, grid_id=None, suffix=None, interpolation="linear", fill=0.0,
    dtype=None, out=None
))]
#[allow(clippy::too_many_arguments)]
fn resample<'py>(
    py: Python<'py>,
    path: std::path::PathBuf,
    image_ids: Vec<String>,
    onto: Option<String>,
    spacing: Option<Vec<f64>>,
    grid_id: Option<String>,
    suffix: Option<String>,
    interpolation: &str,
    fill: f64,
    dtype: Option<&Bound<'py, PyAny>>,
    out: Option<std::path::PathBuf>,
) -> R<Bound<'py, PyAny>> {
    let mut plan = rs::ResamplePlan::from_options(image_ids, onto, spacing, grid_id)?
        .with_interpolation(interpolation)
        .with_fill(fill);
    if let Some(suffix) = suffix {
        plan = plan.with_suffix(suffix);
    }
    if let Some(d) = dtype.filter(|d| !d.is_none()) {
        plan = plan.with_dtype(dtype_arg(d)?);
    }
    if let Some(out) = out {
        plan = plan.with_out(out);
    }
    let at = path.clone();
    let result = py.detach(move || rs::resample(&path, &plan)).map_err(|e| path_error(e, &at))?;
    Ok(json_to_py(py, &result.to_json())?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TransformHeader>()?;
    for f in [
        wrap_pyfunction!(encode_identity, m)?,
        wrap_pyfunction!(encode_affine, m)?,
        wrap_pyfunction!(encode_displacement, m)?,
        wrap_pyfunction!(encode_bspline, m)?,
        wrap_pyfunction!(encode_composite, m)?,
        wrap_pyfunction!(basis, m)?,
        wrap_pyfunction!(check_transform_id, m)?,
        wrap_pyfunction!(inside_extent, m)?,
        wrap_pyfunction!(refuse_outside, m)?,
        wrap_pyfunction!(linear_sample, m)?,
        wrap_pyfunction!(cubic_sample, m)?,
        wrap_pyfunction!(sample_field, m)?,
        wrap_pyfunction!(linear_part, m)?,
        wrap_pyfunction!(to_world_vectors, m)?,
        wrap_pyfunction!(jacobian_determinant, m)?,
        wrap_pyfunction!(folding_fraction, m)?,
        wrap_pyfunction!(target_registration_error, m)?,
        wrap_pyfunction!(frame_graph, m)?,
        wrap_pyfunction!(resolve_between, m)?,
        wrap_pyfunction!(frames_of_timepoint, m)?,
        wrap_pyfunction!(resample, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    m.add("TRANSFORM_KINDS", PyTuple::new(py, model::TRANSFORM_KINDS)?)?;
    m.add("VECTOR_SPACES", PyTuple::new(py, model::VECTOR_SPACES)?)?;
    m.add("INTERPOLATIONS", PyTuple::new(py, model::INTERPOLATIONS)?)?;
    m.add("EXTRAPOLATIONS", PyTuple::new(py, apply::EXTRAPOLATIONS)?)?;
    m.add("SPEC_TRANSFORM_ATTRS", PyTuple::new(py, model::SPEC_TRANSFORM_ATTRS)?)?;
    m.add("LAST_ROW_TOL", model::LAST_ROW_TOL)?;
    m.add("SUPPORTED_ORDERS", PyTuple::new(py, model::SUPPORTED_ORDERS)?)?;
    m.add("DEFAULT_ORDER", model::DEFAULT_ORDER)?;
    m.add("FLOAT16_SAFE_VOXELS", model::FLOAT16_SAFE_VOXELS)?;
    m.add("RESAMPLE_INTERPOLATIONS", PyTuple::new(py, rs::RESAMPLE_INTERPOLATIONS)?)?;
    Ok(())
}
