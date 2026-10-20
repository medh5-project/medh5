//! `medh5.io.nifti`: the engine's NIfTI converter (2.2.0 decision E3).
//!
//! `medh5/io/nifti.py` is a facade over these: the reading, the import, the
//! mask import, the export and the writer are the engine's, so the command
//! line, the Rust crate and Python convert a file the same way.  Each call
//! that takes a report brings the caller's `ConversionReport` back up to
//! date whether it succeeds or not, since what a conversion noted before it
//! was refused is part of what it says.

use std::path::PathBuf;

use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyList, PyTuple};
use serde_json::{Map, Value};

use medh5::convert::nifti::{self as engine, Element, ExportOptions, ImportOptions, ReadOptions, SegOptions};
use medh5::convert::report::ConversionReport;

use crate::convert::{class_key, json_to_py, nd_to_py, py_to_json, py_to_nd};
use crate::errors::{errno_message, R};
use crate::geometry::grid_arg;
use crate::io::{report_arg, report_fields};
use crate::labels::label_set_arg;
use crate::writer::annotated_arg;

/// A refusal as 2.1.0's converter raised it: a failed read with no errno
/// (a file shorter than its header says) is the plain `OSError` nibabel
/// raised, not the `MEDH5IOError` an HDF5 failure is.
fn refused<T>(done: medh5::Result<T>) -> R<T> {
    match done {
        Err(medh5::Error::Io(message)) if errno_message(&message).is_none() => Err(PyOSError::new_err(message).into()),
        other => Ok(other?),
    }
}

/// Run `step` on the engine value of `report` without the GIL, then bring
/// the Python report up to date, refused or not.
fn reported<T: Send>(
    py: Python<'_>,
    report: &Bound<'_, PyAny>,
    step: impl FnOnce(&mut ConversionReport) -> medh5::Result<T> + Send,
) -> R<T> {
    let mut engine = report_arg(report)?;
    let (done, engine) = py.detach(move || {
        let done = step(&mut engine);
        (done, engine)
    });
    report.call_method1("_update", (report_fields(py, &engine)?,))?;
    refused(done)
}

/// What `numpy.dtype` makes of the voxels' type: the byte order and code,
/// or nibabel's record type for RGB.
fn dtype_spec<'py>(py: Python<'py>, voxels: &engine::Voxels) -> PyResult<Bound<'py, PyAny>> {
    let channels: &[&str] = match voxels.element {
        Element::Rgb24 => &["R", "G", "B"],
        Element::Rgba32 => &["R", "G", "B", "A"],
        _ => return Ok(voxels.numpy_descr().into_pyobject(py)?.into_any()),
    };
    let fields: Vec<Bound<'py, PyTuple>> =
        channels.iter().map(|c| PyTuple::new(py, [*c, "u1"])).collect::<PyResult<_>>()?;
    Ok(PyList::new(py, fields)?.into_any())
}

/// The voxels' bytes (a copy Python may write to), their type and shape.
fn voxels_to_py<'py>(
    py: Python<'py>,
    voxels: &engine::Voxels,
) -> PyResult<(Bound<'py, PyByteArray>, Bound<'py, PyAny>, Vec<usize>)> {
    Ok((PyByteArray::new(py, &voxels.bytes), dtype_spec(py, voxels)?, voxels.shape.clone()))
}

/// One file as `(bytes, dtype, shape, geometry)`: the voxels in C order in
/// the returned shape and the file's byte order, and the geometry as JSON.
#[pyfunction]
#[pyo3(signature = (path, *, coord_system="LPS".to_string(), transpose=true, fourth_axis="auto".to_string(),
    assume_geometry=false, plane_as_volume=false))]
#[allow(clippy::type_complexity)]
fn nifti_read<'py>(
    py: Python<'py>,
    path: PathBuf,
    coord_system: String,
    transpose: bool,
    fourth_axis: String,
    assume_geometry: bool,
    plane_as_volume: bool,
) -> R<(Bound<'py, PyByteArray>, Bound<'py, PyAny>, Vec<usize>, Bound<'py, PyAny>)> {
    let options = ReadOptions::default()
        .with_coord_system(coord_system)
        .with_transpose(transpose)
        .with_fourth_axis(fourth_axis)
        .with_assume_geometry(assume_geometry)
        .with_plane_as_volume(plane_as_volume);
    let read = refused(py.detach(move || engine::read_nifti(&path, &options)))?;
    let (bytes, dtype, shape) = voxels_to_py(py, &read.voxels)?;
    Ok((bytes, dtype, shape, json_to_py(py, &read.geometry.to_json())?))
}

/// One file as it is stored (`engine::read_stored`): `(bytes, dtype, shape,
/// fields)`, the voxels first axis fastest in the header's shape.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn nifti_read_stored<'py>(
    py: Python<'py>,
    path: PathBuf,
) -> R<(Bound<'py, PyByteArray>, Bound<'py, PyAny>, Vec<usize>, Bound<'py, PyAny>)> {
    let stored = refused(py.detach(move || engine::read_stored(&path)))?;
    let (bytes, dtype, shape) = voxels_to_py(py, &stored.voxels)?;
    let fields = serde_json::json!({
        "sform_code": stored.header.sform_code,
        "qform_code": stored.header.qform_code,
        "intent_code": stored.header.intent_code,
        "affine": stored.affine,
        "units": stored.units,
        "units_stated": stored.units_stated,
        "rescale": stored.rescale.map(|(s, i)| [s, i]),
    });
    Ok((bytes, dtype, shape, json_to_py(py, &fields)?))
}

/// Write one sample from NIfTI volumes; the caller's report is updated.
#[pyfunction]
#[pyo3(signature = (images, out, *, report, masks=Vec::new(), sample_id=None, subject_id=None,
    modalities=Vec::new(), coord_system="LPS".to_string(), transpose=true, fourth_axis="auto".to_string(),
    value_units=Vec::new(), codec="balanced".to_string(), annotated_classes=None, assume_geometry=false,
    plane_as_volume=false, label_set=None))]
#[allow(clippy::too_many_arguments)]
fn nifti_from(
    py: Python<'_>,
    images: Vec<(String, PathBuf)>,
    out: PathBuf,
    report: &Bound<'_, PyAny>,
    masks: Vec<(String, PathBuf)>,
    sample_id: Option<String>,
    subject_id: Option<String>,
    modalities: Vec<(String, String)>,
    coord_system: String,
    transpose: bool,
    fourth_axis: String,
    value_units: Vec<(String, String)>,
    codec: String,
    annotated_classes: Option<&Bound<'_, PyAny>>,
    assume_geometry: bool,
    plane_as_volume: bool,
    label_set: Option<&Bound<'_, PyAny>>,
) -> R<()> {
    let mut options = ImportOptions::default()
        .with_masks(masks)
        .with_modalities(modalities)
        .with_coord_system(coord_system)
        .with_transpose(transpose)
        .with_fourth_axis(fourth_axis)
        .with_value_units(value_units)
        .with_codec(codec)
        .with_annotated_classes(annotated_arg(annotated_classes)?)
        .with_assume_geometry(assume_geometry)
        .with_plane_as_volume(plane_as_volume);
    if let Some(id) = sample_id {
        options = options.with_sample_id(id);
    }
    if let Some(id) = subject_id {
        options = options.with_subject_id(id);
    }
    if let Some(set) = label_set.filter(|s| !s.is_none()) {
        options = options.with_label_set(label_set_arg(set)?);
    }
    reported(py, report, move |log| engine::from_nifti(&images, &out, &options, log))
}

/// Add NIfTI masks to a sample; the caller's report is updated.
#[pyfunction]
#[pyo3(signature = (path, masks, *, report, ann_id="seg".to_string(), grid=None, coord_system=None,
    transpose=true, annotated_classes=None, assume_aligned=false, assume_geometry=false, plane_as_volume=false))]
#[allow(clippy::too_many_arguments)]
fn nifti_import_seg(
    py: Python<'_>,
    path: PathBuf,
    masks: Vec<(String, PathBuf)>,
    report: &Bound<'_, PyAny>,
    ann_id: String,
    grid: Option<String>,
    coord_system: Option<String>,
    transpose: bool,
    annotated_classes: Option<&Bound<'_, PyAny>>,
    assume_aligned: bool,
    assume_geometry: bool,
    plane_as_volume: bool,
) -> R<()> {
    let mut options = SegOptions::default()
        .with_ann_id(ann_id)
        .with_transpose(transpose)
        .with_annotated_classes(annotated_arg(annotated_classes)?)
        .with_assume_aligned(assume_aligned)
        .with_assume_geometry(assume_geometry)
        .with_plane_as_volume(plane_as_volume);
    if let Some(grid) = grid {
        options = options.with_grid(grid);
    }
    if let Some(system) = coord_system {
        options = options.with_coord_system(system);
    }
    reported(py, report, move |log| engine::import_seg_nifti(&path, &masks, &options, log))
}

/// Export one image, or one annotation or class of it; the file written.
#[pyfunction]
#[pyo3(signature = (path, image_id, out, *, report, physical=true, annotation=None, class_key=None))]
#[allow(clippy::too_many_arguments)]
fn nifti_to(
    py: Python<'_>,
    path: PathBuf,
    image_id: String,
    out: PathBuf,
    report: &Bound<'_, PyAny>,
    physical: bool,
    annotation: Option<String>,
    class_key: Option<&Bound<'_, PyAny>>,
) -> R<String> {
    let mut options = ExportOptions::default().with_physical(physical);
    if let Some(ann) = annotation {
        options = options.with_annotation(ann);
    }
    if let Some(key) = class_key.filter(|k| !k.is_none()) {
        options = options.with_class_key(self::class_key(key)?);
    }
    let written = reported(py, report, move |log| engine::to_nifti(&path, &image_id, &out, &options, Some(log)))?;
    Ok(medh5::pyval::py_path(&written))
}

/// An affine argument: a 2-D float64 array.
fn affine_arg(affine: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    affine.as_array().to_owned()
}

/// Write one array as a NIfTI file; the file written.
#[pyfunction]
#[pyo3(signature = (array, affine, path, *, report=None, rescale=None, units="mm".to_string(), time_values=None,
    time_units=None, leading=None, channels=None))]
#[allow(clippy::too_many_arguments)]
fn nifti_write(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    affine: PyReadonlyArray2<'_, f64>,
    path: PathBuf,
    report: Option<&Bound<'_, PyAny>>,
    rescale: Option<(f64, f64)>,
    units: String,
    time_values: Option<Vec<f64>>,
    time_units: Option<String>,
    leading: Option<String>,
    channels: Option<&Bound<'_, PyAny>>,
) -> R<String> {
    let array = py_to_nd(array)?;
    let affine = affine_arg(affine);
    let channels = match channels.filter(|c| !c.is_none()).map(py_to_json).transpose()? {
        Some(Value::Object(m)) => m,
        _ => Map::new(),
    };
    let options = engine::WriteOptions::default()
        .with_rescale(rescale)
        .with_units(units)
        .with_time_values(time_values, time_units)
        .with_leading(leading)
        .with_channels(channels);
    let written = match report.filter(|r| !r.is_none()) {
        Some(report) => {
            reported(py, report, move |log| engine::write_nifti(&array, &affine, &path, &options, Some(log)))?
        }
        None => refused(py.detach(move || engine::write_nifti(&array, &affine, &path, &options, None)))?,
    };
    Ok(medh5::pyval::py_path(&written))
}

/// An ITK displacement field, `(x, y, z, 1, 3)` under the DISPVECT intent.
#[pyfunction]
fn nifti_write_field(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    affine: PyReadonlyArray2<'_, f64>,
    path: PathBuf,
) -> R<()> {
    let array = py_to_nd(array)?;
    let affine = affine_arg(affine);
    refused(py.detach(move || engine::write_field(&array, &affine, &path)))
}

/// An array and affine in MEDH5 conventions as NIfTI wants them.
#[pyfunction]
fn nifti_for_export<'py>(
    py: Python<'py>,
    grid: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
) -> R<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let grid = grid_arg(grid)?;
    let data = py_to_nd(data)?;
    let (data, affine) = engine::for_export(&grid, data)?;
    Ok((nd_to_py(py, data), numpy::PyArray::from_owned_array(py, affine).into_any()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(nifti_read, m)?,
        wrap_pyfunction!(nifti_read_stored, m)?,
        wrap_pyfunction!(nifti_from, m)?,
        wrap_pyfunction!(nifti_import_seg, m)?,
        wrap_pyfunction!(nifti_to, m)?,
        wrap_pyfunction!(nifti_write, m)?,
        wrap_pyfunction!(nifti_write_field, m)?,
        wrap_pyfunction!(nifti_for_export, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add("NIFTI_SUFFIXES", PyTuple::new(m.py(), engine::NIFTI_SUFFIXES)?)?;
    Ok(())
}
