//! `medh5.geometry`: grids, affines, pyramids (spec §3, §4.3).

use std::hash::{Hash, Hasher};

use ndarray::{Array2, ArrayD, IxDyn};
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple};

use medh5::geometry::{affine, grid as engine_grid, multiscale, Grid as EngineGrid};

use crate::convert::{array_to_py, attr_to_py, f64_array, f64_vec, json_to_py, matrix as to_matrix, py_to_attr};
use crate::errors::R;

// -- dataclass compatibility -----------------------------------------------------------

/// `__dataclass_fields__` for a class whose constructor takes these names:
/// `dataclasses.replace`, `fields` and `asdict` then work on it.
pub fn dataclass_fields(py: Python<'_>, names: &[&str]) -> PyResult<Py<PyDict>> {
    let dataclasses = py.import("dataclasses")?;
    let marker = dataclasses.getattr("_FIELD")?;
    let out = PyDict::new(py);
    for name in names {
        let field = dataclasses.call_method0("field")?;
        field.setattr("name", *name)?;
        field.setattr("type", "object")?;
        field.setattr("_field_type", &marker)?;
        out.set_item(*name, field)?;
    }
    Ok(out.unbind())
}

// -- Grid ------------------------------------------------------------------------------

/// The lattice of a stored array: its shape, axes and index-to-world geometry (§3.2).
///
/// `spacing` and `origin` have one entry per spatial axis, in `units`, and
/// `direction` is `(S, S)`, column *k* the world direction of spatial axis *k*.
/// The constructor checks the grid and raises `MEDH5ValidationError` when it
/// breaks a rule of §3.2.
#[pyclass(module = "medh5.geometry", name = "Grid", skip_from_py_object, frozen)]
pub struct Grid(pub EngineGrid, PyOnceLock<Py<PyAny>>);

impl Grid {
    pub fn wrap(grid: EngineGrid) -> Self {
        Grid(grid, PyOnceLock::new())
    }

    /// A hash that agrees with [`Grid::same_key`]: floats compare with `==`,
    /// under which `-0.0` and `0.0` are one value, so they hash as one
    /// (`-0.0 + 0.0` is `0.0`).
    fn key_hash(&self) -> isize {
        let g = &self.0;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        g.grid_id.hash(&mut h);
        g.shape.hash(&mut h);
        g.axis_names.hash(&mut h);
        g.axis_kinds.hash(&mut h);
        for v in g.spacing.iter().chain(&g.origin).chain(g.direction.iter()) {
            (v + 0.0).to_bits().hash(&mut h);
        }
        g.coord_system.hash(&mut h);
        g.units.hash(&mut h);
        g.timepoint.hash(&mut h);
        g.frame_uid.hash(&mut h);
        if let Some(t) = &g.time_values {
            for v in t {
                (v + 0.0).to_bits().hash(&mut h);
            }
        }
        g.time_units.hash(&mut h);
        h.finish() as isize
    }

    fn same_key(&self, other: &EngineGrid) -> bool {
        let (a, b) = (&self.0, other);
        a.grid_id == b.grid_id
            && a.shape == b.shape
            && a.axis_names == b.axis_names
            && a.axis_kinds == b.axis_kinds
            && a.spacing == b.spacing
            && a.origin == b.origin
            && a.direction == b.direction
            && a.coord_system == b.coord_system
            && a.units == b.units
            && a.timepoint == b.timepoint
            && a.frame_uid == b.frame_uid
            && a.time_values == b.time_values
            && a.time_units == b.time_units
    }
}

/// A grid argument: the binding class only.
pub fn grid_arg(obj: &Bound<'_, PyAny>) -> PyResult<EngineGrid> {
    Ok(obj.cast::<Grid>()?.get().0.clone())
}

fn floats<'py>(py: Python<'py>, values: &[f64]) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, values)
}

fn readonly<'py>(array: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    array.getattr("flags")?.setattr("writeable", false)?;
    Ok(array)
}

const GRID_FIELDS: [&str; 16] = [
    "grid_id",
    "shape",
    "axis_names",
    "axis_kinds",
    "spacing",
    "origin",
    "direction",
    "coord_system",
    "units",
    "timepoint",
    "frame_uid",
    "time_values",
    "time_units",
    "chunk_hint",
    "patch_hint",
    "extra",
];

#[pymethods]
impl Grid {
    #[new]
    #[pyo3(signature = (grid_id, shape, axis_names, axis_kinds, spacing, origin, direction, coord_system="LPS", units="mm", timepoint=None, frame_uid=None, time_values=None, time_units=None, chunk_hint=None, patch_hint=None, extra=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        grid_id: String,
        shape: Vec<i64>,
        axis_names: Vec<String>,
        axis_kinds: Vec<String>,
        spacing: &Bound<'_, PyAny>,
        origin: &Bound<'_, PyAny>,
        direction: &Bound<'_, PyAny>,
        coord_system: &str,
        units: &str,
        timepoint: Option<String>,
        frame_uid: Option<String>,
        time_values: Option<&Bound<'_, PyAny>>,
        time_units: Option<String>,
        chunk_hint: Option<Vec<i64>>,
        patch_hint: Option<Vec<i64>>,
        extra: Option<&Bound<'_, PyDict>>,
    ) -> R<Self> {
        let direction_array = f64_array(direction)?;
        let direction = if direction_array.ndim() == 2 {
            direction_array.into_dimensionality::<ndarray::Ix2>().map_err(|e| medh5::Error::Value(e.to_string()))?
        } else {
            // Wrong-shaped directions are E109, raised by `check` below.
            let n = direction_array.len();
            Array2::from_shape_vec((1, n), direction_array.iter().copied().collect())
                .map_err(|e| medh5::Error::Value(e.to_string()))?
        };
        let mut extras = Vec::new();
        if let Some(extra) = extra {
            for (k, v) in crate::convert::dict_items(extra)? {
                extras.push((k.extract::<String>()?, py_to_attr(&v)?));
            }
        }
        let time_values = match time_values {
            Some(t) if !t.is_none() => Some(f64_vec(t)?),
            _ => None,
        };
        let grid = EngineGrid {
            grid_id,
            shape,
            axis_names,
            axis_kinds,
            spacing: f64_vec(spacing)?,
            origin: f64_vec(origin)?,
            direction,
            coord_system: coord_system.to_string(),
            units: units.to_string(),
            timepoint,
            frame_uid,
            time_values,
            time_units,
            chunk_hint,
            patch_hint,
            extra: extras,
        };
        grid.check()?;
        Ok(Grid::wrap(grid))
    }

    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<PyDict>> {
        dataclass_fields(py, &GRID_FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<PyTuple>> {
        crate::values::match_args(py, &GRID_FIELDS)
    }

    /// `copy.replace(grid, **changes)`.
    #[pyo3(signature = (**changes))]
    fn __replace__(slf: &Bound<'_, Self>, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
        crate::values::dataclass_replace(slf.as_any(), changes)
    }

    #[getter]
    fn grid_id(&self) -> &str {
        &self.0.grid_id
    }
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.shape)
    }
    #[getter]
    fn axis_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.axis_names)
    }
    #[getter]
    fn axis_kinds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.axis_kinds)
    }
    #[getter]
    fn spacing<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        floats(py, &self.0.spacing)
    }
    #[getter]
    fn origin<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        floats(py, &self.0.origin)
    }
    /// The direction matrix, read-only (it belongs to the grid).
    #[getter]
    fn direction<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let made = self.1.get_or_try_init(py, || {
            readonly(array_to_py(py, self.0.direction.clone().into_dyn())).map(Bound::unbind)
        })?;
        Ok(made.bind(py).clone())
    }
    #[getter]
    fn coord_system(&self) -> &str {
        &self.0.coord_system
    }
    #[getter]
    fn units(&self) -> &str {
        &self.0.units
    }
    #[getter]
    fn timepoint(&self) -> Option<&str> {
        self.0.timepoint.as_deref()
    }
    #[getter]
    fn frame_uid(&self) -> Option<&str> {
        self.0.frame_uid.as_deref()
    }
    #[getter]
    fn time_values<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.0.time_values.as_ref().map(|t| floats(py, t)).transpose()
    }
    #[getter]
    fn time_units(&self) -> Option<&str> {
        self.0.time_units.as_deref()
    }
    #[getter]
    fn chunk_hint<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.0.chunk_hint.as_ref().map(|t| PyTuple::new(py, t)).transpose()
    }
    #[getter]
    fn patch_hint<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.0.patch_hint.as_ref().map(|t| PyTuple::new(py, t)).transpose()
    }
    #[getter]
    fn extra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.0.extra {
            out.set_item(k, attr_to_py(py, v)?)?;
        }
        Ok(out)
    }
    fn check(&self) -> R<()> {
        Ok(self.0.check()?)
    }
    #[getter]
    fn ndim(&self) -> usize {
        self.0.ndim()
    }
    #[getter]
    fn spatial_axes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.spatial_axes())
    }
    #[getter]
    fn n_spatial(&self) -> usize {
        self.0.n_spatial()
    }
    #[getter]
    fn spatial_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.spatial_shape())
    }
    #[getter]
    fn spatial_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.spatial_names())
    }
    #[getter]
    fn channel_axis(&self) -> Option<usize> {
        self.0.channel_axis()
    }
    #[getter]
    fn time_axis(&self) -> Option<usize> {
        self.0.time_axis()
    }
    #[getter]
    fn n_voxels(&self) -> usize {
        self.0.n_voxels()
    }
    #[getter]
    fn affine<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array_to_py(py, self.0.affine().into_dyn())
    }
    fn index_to_world<'py>(&self, py: Python<'py>, indices: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        points_through(py, indices, self.0.n_spatial(), |flat| self.0.index_to_world(flat))
    }
    fn world_to_index<'py>(&self, py: Python<'py>, points: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        points_through(py, points, self.0.n_spatial(), |flat| self.0.world_to_index(flat))
    }
    /// `(..., S)` world points of this grid, in `other`'s units (E414 across
    /// `coord_system`s, or from `px`).
    fn world_into<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
        points: &Bound<'py, PyAny>,
    ) -> R<Bound<'py, PyAny>> {
        let other = grid_arg(other)?;
        points_through(py, points, self.0.n_spatial(), |flat| self.0.world_into(&other, flat))
    }
    #[getter]
    fn extent<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let s = self.0.n_spatial();
        Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[s, 2]), self.0.extent())?))
    }
    #[getter]
    fn physical_size<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        floats(py, &self.0.physical_size())
    }
    fn comparable_with(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.0.comparable_with(&grid_arg(other)?))
    }
    #[pyo3(signature = (other, tol=1e-6))]
    fn is_congruent(&self, other: &Bound<'_, PyAny>, tol: f64) -> PyResult<bool> {
        Ok(self.0.is_congruent(&grid_arg(other)?, tol))
    }
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp, py: Python<'_>) -> Py<PyAny> {
        let Ok(other) = other.cast::<Grid>() else {
            return py.NotImplemented();
        };
        let same = self.same_key(&other.get().0);
        match op {
            CompareOp::Eq => same.into_pyobject(py).map(|b| b.to_owned().into_any().unbind()).unwrap_or(py.None()),
            CompareOp::Ne => (!same).into_pyobject(py).map(|b| b.to_owned().into_any().unbind()).unwrap_or(py.None()),
            _ => py.NotImplemented(),
        }
    }
    fn __hash__(&self) -> isize {
        self.key_hash()
    }
    fn __repr__(&self) -> String {
        self.0.repr()
    }
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.0.attrs() {
            out.set_item(k, attr_to_py(py, &v)?)?;
        }
        Ok(out)
    }
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.summary())
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let values = GRID_FIELDS.iter().map(|n| slf.getattr(*n)).collect::<PyResult<Vec<_>>>()?;
        Ok((slf.get_type().into_any(), PyTuple::new(py, values)?))
    }
}

/// Run a flat `S`-per-point function over an array of points of shape
/// `(S,)` or `(N, S)` (or any `(..., S)`), keeping that shape.  It runs with
/// the GIL released: a point set can be a whole volume's voxel centres.
fn points_through<'py>(
    py: Python<'py>,
    points: &Bound<'py, PyAny>,
    s: usize,
    f: impl FnOnce(&[f64]) -> medh5::Result<Vec<f64>> + Send,
) -> R<Bound<'py, PyAny>> {
    let array = f64_array(points)?;
    let shape = array.shape().to_vec();
    if shape.last().copied() != Some(s) {
        return Err(medh5::Error::Value(format!(
            "points must have {s} coordinates on their last axis, got shape {}",
            medh5::json::repr_int_tuple(&shape)
        ))
        .into());
    }
    let out = py.detach(move || {
        let flat: Vec<f64> = array.iter().copied().collect();
        f(&flat)
    })?;
    Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&shape), out)?))
}

// -- Pyramid -----------------------------------------------------------------------------

/// The multiscale declaration of an image group: its levels, factors, method and grids (§4.3).
///
/// `downsample_factors` is `(levels, S)`, relative to level 0, and
/// `grid_levels` names one grid per level.
#[pyclass(module = "medh5.geometry", name = "Pyramid", skip_from_py_object, frozen)]
pub struct Pyramid(pub multiscale::Pyramid);

impl Pyramid {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &["levels", "downsample_factors", "downsample_method", "grid_levels"];
}

#[pymethods]
impl Pyramid {
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
    fn new(
        levels: usize,
        downsample_factors: &Bound<'_, PyAny>,
        downsample_method: String,
        grid_levels: Vec<String>,
    ) -> R<Self> {
        Ok(Pyramid(multiscale::Pyramid::new(levels, to_matrix(downsample_factors)?, downsample_method, grid_levels)?))
    }
    #[getter]
    fn levels(&self) -> usize {
        self.0.levels
    }
    #[getter]
    fn downsample_factors<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array_to_py(py, self.0.downsample_factors.clone().into_dyn())
    }
    #[getter]
    fn downsample_method(&self) -> &str {
        &self.0.downsample_method
    }
    #[getter]
    fn grid_levels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.grid_levels)
    }
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.0.attrs() {
            out.set_item(k, attr_to_py(py, &v)?)?;
        }
        Ok(out)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Pyramid>().map(|o| o.get().0 == self.0).unwrap_or(false)
    }
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let names = ["levels", "downsample_factors", "downsample_method", "grid_levels"];
        let fields = names.iter().map(|n| Ok((*n, slf.getattr(*n)?))).collect::<PyResult<Vec<_>>>()?;
        crate::values::dataclass_repr("Pyramid", &fields)
    }
    /// Pickling and `copy`: the fields, as the constructor takes them.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_fields(slf.as_any(), None, Self::FIELDS)
    }
}

// -- functions -------------------------------------------------------------------------------

/// Return the `(S+1, S+1)` index-to-world affine built from spacing, origin and direction (§3.3).
///
/// `A[:S, :S] = direction @ diag(spacing)` and `A[:S, S] = origin`. Raises
/// `MEDH5ValidationError` (E109) when the three disagree on the number of
/// spatial axes.
#[pyfunction]
fn build_affine<'py>(
    py: Python<'py>,
    spacing: &Bound<'py, PyAny>,
    origin: &Bound<'py, PyAny>,
    direction: &Bound<'py, PyAny>,
) -> R<Bound<'py, PyAny>> {
    let a = affine::build_affine(&f64_vec(spacing)?, &f64_vec(origin)?, &to_matrix(direction)?)?;
    Ok(array_to_py(py, a.into_dyn()))
}

/// Split an index-to-world affine back into `(spacing, origin, direction)` arrays (§3.3).
///
/// The spacing is the length of each column of the linear part; an axis of
/// length zero raises `MEDH5ValidationError` (E104), and a matrix that is not
/// `(S+1, S+1)` raises `ValueError`.
#[pyfunction]
fn decompose_affine<'py>(py: Python<'py>, affine: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    let (spacing, origin, direction) = affine::decompose_affine(&to_matrix(affine)?)?;
    Ok(PyTuple::new(
        py,
        [
            array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[spacing.len()]), spacing)?),
            array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[origin.len()]), origin)?),
            array_to_py(py, direction.into_dyn()),
        ],
    )?)
}

/// Whether `direction` is a square matrix with `direction.T @ direction == I` to within `tol`.
///
/// A matrix that is not square is `False`, never an error (§3.2).
#[pyfunction]
#[pyo3(signature = (direction, tol=affine::ORTHONORMAL_TOL))]
fn is_orthonormal(direction: &Bound<'_, PyAny>, tol: f64) -> PyResult<bool> {
    let d = f64_array(direction)?;
    if d.ndim() != 2 || d.shape()[0] != d.shape()[1] {
        return Ok(false);
    }
    Ok(affine::is_orthonormal(&d.into_dimensionality().expect("checked 2-D"), tol))
}

/// Return `direction` as a float array, raising E102 unless it is orthonormal to within `tol`.
///
/// `what` names the matrix in the message of the `MEDH5ValidationError` (§3.2).
#[pyfunction]
#[pyo3(signature = (direction, tol=affine::ORTHONORMAL_TOL, *, what="direction"))]
fn check_orthonormal<'py>(
    py: Python<'py>,
    direction: &Bound<'py, PyAny>,
    tol: f64,
    what: &str,
) -> R<Bound<'py, PyAny>> {
    let d = to_matrix(direction)?;
    affine::check_orthonormal(&d, tol, what)?;
    Ok(array_to_py(py, d.into_dyn()))
}

/// Whether `matrix` is orthonormal to within `tol` and has determinant +1.
///
/// A matrix that is not square is `False`. Oriented boxes need a proper
/// rotation (§8.3).
#[pyfunction]
#[pyo3(signature = (matrix, tol=affine::ORTHONORMAL_TOL))]
fn is_proper_rotation(matrix: &Bound<'_, PyAny>, tol: f64) -> PyResult<bool> {
    let d = f64_array(matrix)?;
    if d.ndim() != 2 || d.shape()[0] != d.shape()[1] {
        return Ok(false);
    }
    Ok(affine::is_proper_rotation(&d.into_dimensionality().expect("checked 2-D"), tol))
}

/// Map continuous index coordinates to world coordinates through an `(S+1, S+1)` affine (§3.3).
///
/// `indices` is `(..., S)`, an integer at a voxel centre; the result has the
/// same shape, in the grid's units. Raises `ValueError` when the last axis is
/// not `S` long.
#[pyfunction]
fn index_to_world<'py>(
    py: Python<'py>,
    affine: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
) -> R<Bound<'py, PyAny>> {
    let a = to_matrix(affine)?;
    let s = affine::spatial_axes(&a)?;
    points_through(py, indices, s, |flat| affine::index_to_world(&a, flat))
}

/// Map world coordinates back to continuous index coordinates through an `(S+1, S+1)` affine (§3.3).
///
/// `points` is `(..., S)` in the grid's units; the result has the same shape,
/// an integer at a voxel centre. Raises `ValueError` when the last axis is not
/// `S` long.
#[pyfunction]
fn world_to_index<'py>(
    py: Python<'py>,
    affine: &Bound<'py, PyAny>,
    points: &Bound<'py, PyAny>,
) -> R<Bound<'py, PyAny>> {
    let a = to_matrix(affine)?;
    let s = affine::spatial_axes(&a)?;
    points_through(py, points, s, |flat| affine::world_to_index(&a, flat))
}

/// A box argument, `(S, 2)`, flattened to `[lo0, hi0, lo1, hi1, ...]`.
///
/// Only `box_to_slices` checked the shape: `box_corners` and
/// `apply_affine_to_box` flattened whatever they were given, and read a
/// `(2, 3)` min/max-corner box as three axes of made-up bounds.
fn box_arg(r#box: &Bound<'_, PyAny>) -> R<Vec<f64>> {
    let b = f64_array(r#box)?;
    if b.ndim() != 2 || b.shape()[1] != 2 {
        return Err(medh5::Error::invalid(format!(
            "box must have shape (S, 2), got {}",
            medh5::json::repr_int_tuple(b.shape())
        ))
        .into());
    }
    Ok(b.iter().copied().collect())
}

/// Convert one `(S, 2)` box at voxel edges to a tuple of `slice(start, stop)`, one per axis (§8.1).
///
/// `start = floor(lo + 0.5)` and `stop = floor(hi + 0.5)`: half up, never half
/// to even. With `shape`, one extent per axis, the slices are clipped to it. A
/// box with `lo > hi` or a bound that is not finite raises
/// `MEDH5ValidationError` (E406).
#[pyfunction]
#[pyo3(signature = (r#box, shape=None))]
fn box_to_slices<'py>(py: Python<'py>, r#box: &Bound<'py, PyAny>, shape: Option<Vec<usize>>) -> R<Bound<'py, PyTuple>> {
    let found = affine::box_to_slices(&box_arg(r#box)?, shape.as_deref())?;
    let slices = found
        .into_iter()
        .map(|(start, stop)| pyo3::types::PySlice::new(py, start as isize, stop as isize, 1))
        .collect::<Vec<_>>();
    // `slice(a, b)` prints without a step: build them the way Python does.
    let builtins = py.import("builtins")?;
    let out = slices
        .iter()
        .map(|s| builtins.getattr("slice")?.call1((s.getattr("start")?, s.getattr("stop")?)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, out)?)
}

/// Convert `slice(start, stop)` objects to one `(S, 2)` `float32` box at voxel edges (§8.1).
///
/// Each axis becomes `[start - 0.5, stop - 0.5]`. A slice with an open stop, a
/// negative bound, `start > stop` or a step raises `ValueError`: only the array
/// knows what those select.
#[pyfunction]
fn slices_to_box<'py>(py: Python<'py>, slices: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let mut pairs = Vec::new();
    for (axis, item) in slices.try_iter()?.enumerate() {
        let item = item?;
        let start = item.getattr("start")?.extract::<Option<i64>>()?.unwrap_or(0);
        let stop = item.getattr("stop")?.extract::<Option<i64>>()?;
        let step = item.getattr("step")?.extract::<Option<i64>>()?;
        // A box is the voxel edges a slice starts and stops at.  An open
        // stop and a negative bound are counted from an end only the array
        // knows, and a step selects no block: an open stop raised a bare
        // `TypeError`, and the others made a box of other voxels.
        match stop {
            Some(stop) if 0 <= start && start <= stop && step.unwrap_or(1) == 1 => pairs.push((start, stop)),
            _ => {
                return Err(medh5::Error::Value(format!(
                    "a box is made from slice(start, stop) with 0 <= start <= stop and no step; axis {axis} is {}",
                    item.repr()?
                ))
                .into())
            }
        }
    }
    let flat = affine::slices_to_box(&pairs);
    Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[pairs.len(), 2]), flat)?))
}

/// Return the `2**S` corners of an `(S, 2)` axis-aligned box as a `(2**S, S)` array.
///
/// Corners are in odometer order, the first axis varying slowest. A box of
/// another shape raises `MEDH5ValidationError`, and one of more than 16 axes
/// raises `ValueError`.
#[pyfunction]
fn box_corners<'py>(py: Python<'py>, r#box: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let flat = box_arg(r#box)?;
    let s = flat.len() / 2;
    let corners: Vec<f64> = affine::box_corners(&flat)?.into_iter().flatten().collect();
    Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[1 << s, s]), corners)?))
}

/// Return the `(S, 2)` axis-aligned bounds of a box's corners mapped through an affine.
///
/// `box` is `(S, 2)` and `affine` `(S+1, S+1)`: an index-space box at voxel
/// edges and a grid's affine give the box's world bounds.
#[pyfunction]
fn apply_affine_to_box<'py>(
    py: Python<'py>,
    affine: &Bound<'py, PyAny>,
    r#box: &Bound<'py, PyAny>,
) -> R<Bound<'py, PyAny>> {
    let out = affine::apply_affine_to_box(&to_matrix(affine)?, &box_arg(r#box)?)?;
    let s = out.len() / 2;
    Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[s, 2]), out)?))
}

/// Return the volume of one voxel: the product of `spacing`, in the grid's units to the power `S`.
#[pyfunction]
fn voxel_volume(spacing: &Bound<'_, PyAny>) -> PyResult<f64> {
    Ok(affine::voxel_volume(&f64_vec(spacing)?))
}

/// Return a dict describing an affine: `spacing`, `origin`, `direction`, `determinant`, `orthonormal`.
///
/// The determinant is that of `direction`, and `orthonormal` is checked to
/// `ORTHONORMAL_TOL`. Raises as `decompose_affine` does.
#[pyfunction]
fn affine_summary<'py>(py: Python<'py>, affine: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &affine::affine_summary(&to_matrix(affine)?)?)?)
}

/// Derive the `Grid` of one pyramid level from level 0 and its per-axis downsample factors (§4.3).
///
/// `spacing' = spacing * f` and the origin moves by the half-voxel shift
/// `direction @ (spacing * (f - 1) / 2)`. `shape` defaults to `ceil(shape / f)`
/// on the spatial axes, which keeps the field of view covered. A `factors` of
/// the wrong length raises `MEDH5ValidationError` (E105).
#[pyfunction]
#[pyo3(signature = (base, factors, grid_id, *, shape=None))]
fn derive_level_grid(
    base: &Bound<'_, PyAny>,
    factors: &Bound<'_, PyAny>,
    grid_id: &str,
    shape: Option<Vec<i64>>,
) -> R<Grid> {
    Ok(Grid::wrap(multiscale::derive_level_grid(&grid_arg(base)?, &f64_vec(factors)?, grid_id, shape.as_deref())?))
}

/// Return a message for each pyramid level whose geometry breaks the derivation rule (§4.3).
///
/// `levels` are the level grids and `factors` the `(L, S)` downsample factors
/// relative to `base`, row by row. Spacing is held to `rtol` of the derived
/// spacing and the origin to `rtol` of the level's voxel. An empty list means
/// the pyramid conforms.
#[pyfunction]
#[pyo3(signature = (base, levels, factors, *, rtol=multiscale::GEOMETRY_RTOL))]
fn check_pyramid(
    base: &Bound<'_, PyAny>,
    levels: &Bound<'_, PyAny>,
    factors: &Bound<'_, PyAny>,
    rtol: f64,
) -> R<Vec<String>> {
    let grids = levels.try_iter()?.map(|g| grid_arg(&g?)).collect::<PyResult<Vec<_>>>()?;
    let refs: Vec<&EngineGrid> = grids.iter().collect();
    Ok(multiscale::check_pyramid(&grid_arg(base)?, &refs, &to_matrix(factors)?, rtol)?)
}

/// Recover the `(L, S)` downsample factors of pyramid levels from their spacings relative to `base`.
#[pyfunction]
fn pyramid_factors<'py>(
    py: Python<'py>,
    base: &Bound<'py, PyAny>,
    levels: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let grids = levels.try_iter()?.map(|g| grid_arg(&g?)).collect::<PyResult<Vec<_>>>()?;
    let refs: Vec<&EngineGrid> = grids.iter().collect();
    Ok(array_to_py(py, multiscale::pyramid_factors(&grid_arg(base)?, &refs).into_dyn()))
}

/// One grid group read back (`sample.root["grids/ct"]`).
#[pyfunction]
#[pyo3(signature = (group, grid_id=None))]
fn read_grid(py: Python<'_>, group: &Bound<'_, PyAny>, grid_id: Option<&str>) -> R<Grid> {
    let group = crate::integrity::group_of(group)?;
    Ok(Grid::wrap(crate::h5(py, || engine_grid::read_grid(&group, grid_id))?))
}

/// Every grid under a sample root, by id.
#[pyfunction]
fn read_grids<'py>(py: Python<'py>, root: &Bound<'py, PyAny>) -> R<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let root = crate::integrity::group_of(root)?;
    for (grid_id, grid) in crate::h5(py, || engine_grid::read_grids(&root))? {
        out.set_item(grid_id, Grid::wrap(grid))?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_function(wrap_pyfunction!(read_grid, m)?)?;
    m.add_function(wrap_pyfunction!(read_grids, m)?)?;
    m.add_class::<Grid>()?;
    m.add_class::<Pyramid>()?;
    for f in [
        wrap_pyfunction!(build_affine, m)?,
        wrap_pyfunction!(decompose_affine, m)?,
        wrap_pyfunction!(is_orthonormal, m)?,
        wrap_pyfunction!(check_orthonormal, m)?,
        wrap_pyfunction!(is_proper_rotation, m)?,
        wrap_pyfunction!(index_to_world, m)?,
        wrap_pyfunction!(world_to_index, m)?,
        wrap_pyfunction!(box_to_slices, m)?,
        wrap_pyfunction!(slices_to_box, m)?,
        wrap_pyfunction!(box_corners, m)?,
        wrap_pyfunction!(apply_affine_to_box, m)?,
        wrap_pyfunction!(voxel_volume, m)?,
        wrap_pyfunction!(affine_summary, m)?,
        wrap_pyfunction!(derive_level_grid, m)?,
        wrap_pyfunction!(check_pyramid, m)?,
        wrap_pyfunction!(pyramid_factors, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add("ORTHONORMAL_TOL", affine::ORTHONORMAL_TOL)?;
    m.add("AXIS_KINDS", PyTuple::new(py, engine_grid::AXIS_KINDS)?)?;
    m.add("KNOWN_UNITS", PyTuple::new(py, engine_grid::KNOWN_UNITS)?)?;
    m.add("COORD_SYSTEMS", PyTuple::new(py, engine_grid::COORD_SYSTEMS)?)?;
    m.add("TIME_UNITS", PyTuple::new(py, engine_grid::TIME_UNITS)?)?;
    // Millimetres per calibrated length unit (§3.2), `px` having none.
    let mm_per_unit = pyo3::types::PyDict::new(py);
    for unit in engine_grid::KNOWN_UNITS {
        if let Some(mm) = engine_grid::mm_per_unit(unit) {
            mm_per_unit.set_item(unit, mm)?;
        }
    }
    m.add("MM_PER_UNIT", mm_per_unit)?;
    m.add("SPEC_GRID_ATTRS", PyTuple::new(py, engine_grid::SPEC_GRID_ATTRS)?)?;
    m.add("DOWNSAMPLE_METHODS", PyTuple::new(py, multiscale::DOWNSAMPLE_METHODS)?)?;
    m.add("LABEL_SAFE_METHODS", PyTuple::new(py, multiscale::LABEL_SAFE_METHODS)?)?;
    m.add("GEOMETRY_RTOL", multiscale::GEOMETRY_RTOL)?;
    Ok(())
}
