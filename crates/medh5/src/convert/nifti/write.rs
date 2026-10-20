//! NIfTI export (spec §3, §4.2): one image, or one annotation, written as
//! the file 2.1.0 wrote through nibabel.
//!
//! The header is nibabel's `Nifti1Image(data, affine)` saved as it is: a
//! NIfTI-1 single file in little-endian order, the affine as the sform
//! (`aligned`) and again as a qform (code `unknown`) whose quaternion is
//! nibabel's ([`super::qform`]), `scl_slope`/`scl_inter` the rescale or the
//! identity, the voxels after the header's four-byte extension flag.  What
//! the header cannot say beside the voxels --- frame times that one step does
//! not state, per-channel acquisition values, b-values --- goes in a BIDS
//! sidecar and a `.bval` beside the file, which the file owns: an export
//! over an earlier one withdraws what that one said about its volumes.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flate2::{Compression, GzBuilder};
use indexmap::IndexMap;
use ndarray::{Array2, ArrayD, Axis, IxDyn};
use serde_json::{json, Map, Value};

use super::header::{Element, Header, Version, VoxOffset, NIFTI1_SINGLE_OFFSET};
use super::import::swap_suffix;
use super::qform::qform;
use crate::array::{DType, NdArray};
use crate::convert::report::ConversionReport;
use crate::geometry::affine::{build_affine, decompose_affine};
use crate::geometry::grid::{mm_per_unit, Grid};
use crate::json::{num, repr_str};
use crate::labels::ClassKey;
use crate::pyval::py_path;
use crate::{Error, Result};

/// The names an export writes a NIfTI file under, and the only names whose
/// sidecar and `.bval` are found beside them.
pub const NIFTI_SUFFIXES: [&str; 2] = [".nii.gz", ".nii"];

/// Sidecar fields stating one value per volume: what [`super::read_nifti`]
/// reads as the fourth axis's meaning, and so what an export owns beside
/// its file.
const VOLUME_FIELDS: [&str; 6] = ["VolumeTiming", "EchoTime", "EchoTimes", "EchoNumber", "InversionTime", "FlipAngle"];

/// A channel axis's per-channel acquisition values (§4.5): the DICOM
/// keyword they are stored under, and the BIDS field they are written to.
const CHANNEL_FIELDS: [(&str, &str); 4] = [
    ("EchoTime", "EchoTime"),
    ("EchoNumbers", "EchoNumber"),
    ("InversionTime", "InversionTime"),
    ("FlipAngle", "FlipAngle"),
];

/// Seconds per unit of a time axis.
fn seconds_per_unit(unit: &str) -> Option<f64> {
    match unit {
        "s" => Some(1.0),
        "ms" => Some(1e-3),
        _ => None,
    }
}

/// NIfTI's `intent_code` for a vector (NIFTI_INTENT_VECTOR).
const INTENT_VECTOR: i64 = 1007;

/// NIfTI's displacement-vector intent, an ITK field's.
const INTENT_DISPVECT: i64 = 1006;

/// The largest extent NIfTI-1's `dim` holds.
const DIM_MAX: usize = i16::MAX as usize;

/// What [`write_nifti`] states about the array beside its voxels.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct WriteOptions {
    /// `(slope, intercept)` written to `scl_slope`/`scl_inter`: the voxels
    /// are stored values, and these make them physical (§4.2).
    pub rescale: Option<(f64, f64)>,
    /// The affine's spatial unit (§3.2); lengths are written in millimetres,
    /// and a grid without a physical size (`px`) says NIfTI's "unknown".
    pub units: String,
    /// One time per frame of a fourth axis that is time.
    pub time_values: Option<Vec<f64>>,
    pub time_units: Option<String>,
    /// What the fourth axis is (`time` or `channel`), when there is one.
    pub leading: Option<String>,
    /// The image's acquisition values (§4.5), by DICOM keyword: a channel
    /// axis's per-channel values are taken from here.
    pub channels: Map<String, Value>,
}

impl Default for WriteOptions {
    fn default() -> Self {
        WriteOptions {
            rescale: None,
            units: "mm".into(),
            time_values: None,
            time_units: None,
            leading: None,
            channels: Map::new(),
        }
    }
}

impl WriteOptions {
    pub fn with_rescale(mut self, value: Option<(f64, f64)>) -> Self {
        self.rescale = value;
        self
    }

    pub fn with_units(mut self, value: impl Into<String>) -> Self {
        self.units = value.into();
        self
    }

    pub fn with_time_values(mut self, values: Option<Vec<f64>>, units: Option<String>) -> Self {
        self.time_values = values;
        self.time_units = units;
        self
    }

    pub fn with_leading(mut self, value: Option<String>) -> Self {
        self.leading = value;
        self
    }

    pub fn with_channels(mut self, value: Map<String, Value>) -> Self {
        self.channels = value;
        self
    }
}

/// What [`to_nifti`] exports.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ExportOptions {
    /// Write physical values; `false` writes the stored ones with the
    /// image's rescale in the header.
    pub physical: bool,
    /// Export this annotation, on its own grid, instead of the image.
    pub annotation: Option<String>,
    /// One class of the annotation, as a 0/1 mask; without it, the whole
    /// annotation's labelmap.
    pub class_key: Option<ClassKey>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        ExportOptions { physical: true, annotation: None, class_key: None }
    }
}

impl ExportOptions {
    pub fn with_physical(mut self, value: bool) -> Self {
        self.physical = value;
        self
    }

    pub fn with_annotation(mut self, value: impl Into<String>) -> Self {
        self.annotation = Some(value.into());
        self
    }

    pub fn with_class_key(mut self, value: ClassKey) -> Self {
        self.class_key = Some(value);
        self
    }
}

/// `path` as the NIfTI file an export writes, refused before anything is:
/// a directory is no file to write, and a name not ending in `.nii` or
/// `.nii.gz` names no NIfTI file.
pub fn nifti_target(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        return Err(Error::invalid(format!(
            "{} is a directory; name the .nii or .nii.gz file to write",
            pathlib_str(path)
        )));
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if !NIFTI_SUFFIXES.iter().any(|s| name.ends_with(s)) {
        return Err(Error::invalid(format!(
            "{} is not a NIfTI file name: an export writes one ending in .nii or .nii.gz",
            pathlib_str(path)
        )));
    }
    Ok(path.to_path_buf())
}

/// `str(pathlib.Path(path))`, as 2.1.0 named a target: on Windows, with
/// its own separator.
fn pathlib_str(path: &Path) -> String {
    let shown = py_path(path);
    if cfg!(windows) {
        shown.replace('/', "\\")
    } else {
        shown
    }
}

/// The file beside a NIfTI file sharing its stem and ending in `suffix`,
/// the stem in the name's own bytes (a name that is not UTF-8 keeps them).
pub fn sidecar_path(target: &Path, suffix: &str) -> PathBuf {
    if let Some(name) = target.file_name() {
        let shown = name.to_string_lossy();
        if let Some(extension) = NIFTI_SUFFIXES.iter().find(|e| shown.ends_with(*e)) {
            return target.with_file_name(swap_suffix(name, extension.len(), suffix));
        }
    }
    target.with_extension(suffix.trim_start_matches('.'))
}

/// An array and affine in MEDH5 conventions, as NIfTI wants them.
///
/// MEDH5 leads with time or channels and trails with `(z, y, x)`; NIfTI is
/// the other way round, so the spatial axes are reversed and the affine's
/// columns with them.  A 2-D plane is *not* reversed --- [`super::read_nifti`]
/// does not transpose one either, so reversing here would mirror every
/// radiograph --- and its 3x3 affine is embedded in the 4x4 a NIfTI file
/// carries (§3.6).  The array is returned as a view of the same memory with
/// its axes permuted: nothing is copied.
pub fn for_export(grid: &Grid, data: NdArray) -> Result<(NdArray, Array2<f64>)> {
    let affine = super::convert_world(&grid.affine(), &grid.coord_system, "RAS")?;
    let (mut spacing, origin, mut direction) = decompose_affine(&affine)?;
    let ndim = data.ndim();
    if spacing.len() == 2 && ndim > 2 {
        return Err(Error::invalid(format!(
            "grid {} is a 2-D plane with a {} axis, which no NIfTI layout this exporter writes can state; export \
             its planes one at a time",
            repr_str(&grid.grid_id),
            grid.axis_kinds.first().map_or("leading", String::as_str)
        )));
    }
    let data = if ndim >= 3 {
        let mut order = vec![ndim - 1, ndim - 2, ndim - 3];
        order.extend(0..ndim - 3);
        let index = [2, 1, 0];
        spacing = index.iter().map(|i| spacing[*i]).collect();
        direction = Array2::from_shape_fn((3, 3), |(r, c)| direction[[r, index[c]]]);
        crate::with_array!(data, a => NdArray::from(a.permuted_axes(IxDyn(&order))))
    } else {
        data
    };
    Ok((data, embed_plane(&build_affine(&spacing, &origin, &direction)?)?))
}

/// A 2-D grid's 3x3 affine as the 4x4 a NIfTI file carries: the third index
/// axis unit and the third world coordinate zero, the only embedding the
/// import reads back.  A 4x4 affine is returned as it is.
pub fn embed_plane(affine: &Array2<f64>) -> Result<Array2<f64>> {
    match affine.dim() {
        (4, 4) => Ok(affine.clone()),
        (3, 3) => {
            let mut out = Array2::eye(4);
            for r in 0..2 {
                for c in 0..2 {
                    out[[r, c]] = affine[[r, c]];
                }
                out[[r, 3]] = affine[[r, 2]];
            }
            Ok(out)
        }
        (rows, cols) => Err(Error::invalid(format!("embed_plane takes a 2-D grid's 3x3 affine; got ({rows}, {cols})"))),
    }
}

/// nibabel's refusal of 64-bit integers, which it makes before anything
/// else about an image.
fn refuse_wide(dtype: DType) -> Result<()> {
    if matches!(dtype, DType::I64 | DType::U64) {
        return Err(Error::Value(format!(
            "Image data has type {}, which may cause incompatibilities with other tools. To use this type, pass an \
             explicit header or dtype argument to Nifti1Image().",
            dtype.name()
        )));
    }
    Ok(())
}

/// The NIfTI element of a dtype, or nibabel's refusal of it.
fn element_of(dtype: DType) -> Result<Element> {
    refuse_wide(dtype)?;
    Ok(match dtype {
        DType::U8 => Element::U8,
        DType::I8 => Element::I8,
        DType::I16 => Element::I16,
        DType::U16 => Element::U16,
        DType::I32 => Element::I32,
        DType::U32 => Element::U32,
        DType::F32 => Element::F32,
        DType::F64 => Element::F64,
        DType::I64 | DType::U64 => unreachable!("refused above"),
        DType::Bool | DType::F16 => return Err(Error::Value(format!("data dtype \"{}\" not supported", dtype.name()))),
    })
}

/// The array's elements little-endian, first axis fastest: NIfTI's order.
fn fortran_bytes(array: &NdArray) -> Vec<u8> {
    macro_rules! bytes {
        ($a:expr) => {{
            let view = $a.view().reversed_axes();
            let mut out =
                Vec::with_capacity(view.len() * std::mem::size_of_val(&$a.first().copied().unwrap_or_default()));
            for v in view.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }};
    }
    match array {
        NdArray::Bool(a) => a.view().reversed_axes().iter().map(|v| u8::from(*v)).collect(),
        NdArray::I8(a) => bytes!(a),
        NdArray::U8(a) => bytes!(a),
        NdArray::I16(a) => bytes!(a),
        NdArray::U16(a) => bytes!(a),
        NdArray::I32(a) => bytes!(a),
        NdArray::U32(a) => bytes!(a),
        NdArray::I64(a) => bytes!(a),
        NdArray::U64(a) => bytes!(a),
        NdArray::F16(a) => bytes!(a),
        NdArray::F32(a) => bytes!(a),
        NdArray::F64(a) => bytes!(a),
    }
}

/// `shape (4, 3, 2) does not fit in dim datatype`, nibabel's refusal of a
/// shape NIfTI-1's `dim` cannot hold.
fn check_shape(dims: &[usize]) -> Result<()> {
    if dims.len() > 7 || dims.iter().any(|d| *d > DIM_MAX) {
        return Err(Error::Value(format!("shape {} does not fit in dim datatype", crate::json::repr_int_tuple(dims))));
    }
    Ok(())
}

/// The header `Nifti1Image(data, affine)` builds: the type, the shape, the
/// affine as the sform and as an `unknown` qform.
fn image_header(dims: &[usize], element: Element, affine: &[[f64; 4]; 4]) -> Result<Header> {
    check_shape(dims)?;
    let mut header = Header::new(Version::One);
    header.big_endian = false;
    header.datatype = element.code();
    header.bitpix = 8 * element.itemsize() as i64;
    header.dim = [1; 8];
    header.dim[0] = dims.len() as i64;
    for (k, d) in dims.iter().enumerate() {
        header.dim[k + 1] = *d as i64;
    }
    header.srow_x = affine[0];
    header.srow_y = affine[1];
    header.srow_z = affine[2];
    header.sform_code = 2;
    let q = qform(affine).ok_or_else(|| Error::Value(format!("Could not decompose affine:\n{}", numpy_str(affine))))?;
    header.qform_code = 0;
    header.pixdim[0] = q.qfac;
    header.pixdim[1..4].copy_from_slice(&q.zooms);
    [header.quatern_b, header.quatern_c, header.quatern_d] = q.quaternion;
    [header.qoffset_x, header.qoffset_y, header.qoffset_z] = q.offset;
    header.vox_offset = VoxOffset::Float(NIFTI1_SINGLE_OFFSET);
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;
    Ok(header)
}

/// A 4x4 float64 array as numpy's `str()` prints it (`precision=8`,
/// `floatmode='maxprec'`): positional, or scientific when the non-zero
/// magnitudes span more than 1e3 or reach 1e8 or fall below 1e-4.
fn numpy_str(m: &[[f64; 4]; 4]) -> String {
    let values: Vec<f64> = m.iter().flatten().copied().collect();
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let magnitudes = finite.iter().map(|v| v.abs()).filter(|v| *v != 0.0);
    let (max, min) = magnitudes.fold((None, None), |(hi, lo): (Option<f64>, Option<f64>), v| {
        (Some(hi.map_or(v, |h| h.max(v))), Some(lo.map_or(v, |l| l.min(v))))
    });
    let scientific = matches!((max, min), (Some(hi), Some(lo)) if hi >= 1e8 || lo < 1e-4 || hi / lo > 1e3);
    // Each finite value's integer part (sign included), fraction digits and
    // exponent, each at most 8 digits and with trailing zeros trimmed.
    let parts: Vec<(String, String, i32)> = finite.iter().map(|v| float_parts(*v, scientific)).collect();
    let mut pad_left = parts.iter().map(|p| p.0.len()).max().unwrap_or(0);
    let precision = parts.iter().map(|p| p.1.len()).max().unwrap_or(0);
    let exp_size = parts.iter().map(|p| p.2.unsigned_abs().to_string().len().max(2)).max().unwrap_or(0);
    let pad_right = if scientific { exp_size + 2 + precision } else { precision };
    if finite.len() != values.len() {
        let neginf = values.contains(&f64::NEG_INFINITY) as usize;
        pad_left = pad_left.max((3 + neginf).saturating_sub(pad_right + 1));
    }
    let cell = |v: f64| -> String {
        if !v.is_finite() {
            let word = if v.is_nan() {
                "nan"
            } else if v > 0.0 {
                "inf"
            } else {
                "-inf"
            };
            return format!("{word:>width$}", width = pad_left + pad_right + 1);
        }
        let (int, frac, exp) = float_parts(v, scientific);
        if scientific {
            let sign = if exp < 0 { '-' } else { '+' };
            format!("{int:>pad_left$}.{frac:0<precision$}e{sign}{:0>exp_size$}", exp.unsigned_abs())
        } else {
            format!("{int:>pad_left$}.{frac:<precision$}")
        }
    };
    let rows: Vec<String> =
        m.iter().map(|row| format!("[{}]", row.iter().map(|v| cell(*v)).collect::<Vec<_>>().join(" "))).collect();
    format!("[{}]", rows.join("\n "))
}

/// `v`'s integer part, fraction digits and exponent as numpy's dragon4
/// gives them: the shortest digits that read back as `v`, rounded to 8
/// fraction digits where there are more, trailing zeros dropped.
fn float_parts(v: f64, scientific: bool) -> (String, String, i32) {
    let trim = |s: String| -> (String, String) {
        let (int, frac) = s.split_once('.').map_or((s.clone(), String::new()), |(i, f)| (i.into(), f.into()));
        (int, frac.trim_end_matches('0').into())
    };
    if scientific {
        let shortest = format!("{v:e}");
        let (mantissa, exp) = shortest.split_once('e').expect("`{:e}` writes an exponent");
        let mantissa = match mantissa.split_once('.') {
            Some((_, frac)) if frac.len() > 8 => format!("{v:.8e}").split_once('e').unwrap().0.to_string(),
            _ => mantissa.to_string(),
        };
        let (int, frac) = trim(mantissa);
        (int, frac, exp.parse().expect("`{:e}` writes an integer exponent"))
    } else {
        let shortest = format!("{v}");
        let fixed = match shortest.split_once('.') {
            Some((_, frac)) if frac.len() > 8 => format!("{v:.8}"),
            _ => shortest,
        };
        let (int, frac) = trim(fixed);
        (int, frac, 0)
    }
}

/// nibabel's `set_slope_inter` checks.
fn slope_inter(slope: f64, inter: f64) -> Result<(f64, f64)> {
    if slope == 0.0 || slope.is_infinite() {
        return Err(Error::Value("Slope cannot be 0 or infinite".into()));
    }
    if inter.is_infinite() {
        return Err(Error::Value("Intercept cannot be infinite".into()));
    }
    if slope.is_nan() != inter.is_nan() {
        return Err(Error::Value("None or both of slope, inter should be nan".into()));
    }
    // Both NaN asks the writer to choose, and a writer that does not scale
    // chooses the identity.
    Ok(if slope.is_nan() { (1.0, 0.0) } else { (slope, inter) })
}

#[cfg(test)]
thread_local! {
    /// Where a test stops an export, as a full disk or a crash would.
    static FAULT: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

/// Fail here when a test asks to: the steps an export takes have no other
/// door through which to interrupt them.
fn fault(_at: &'static str) -> std::io::Result<()> {
    #[cfg(test)]
    if FAULT.with(|f| f.get()) == Some(_at) {
        return Err(std::io::Error::other("interrupted"));
    }
    Ok(())
}

/// The header, its extension flag and the voxels, as one NIfTI file at
/// `path` --- gzip-compressed when it ends in `.gz`.
fn save(path: &Path, header: &Header, voxels: &[u8]) -> Result<()> {
    let io = |e: std::io::Error| crate::error::os_error(&e, path);
    let body = |sink: &mut dyn Write| -> std::io::Result<()> {
        sink.write_all(&header.encode())?;
        fault("save")?;
        sink.write_all(&[0u8; 4])?;
        sink.write_all(voxels)
    };
    let file = BufWriter::new(fs::File::create(path).map_err(io)?);
    let mut file = if path.to_string_lossy().ends_with(".gz") {
        let mut gz = GzBuilder::new().mtime(0).write(file, Compression::new(1));
        body(&mut gz).map_err(io)?;
        // Finished here, not on drop, which would swallow a failed trailer.
        gz.finish().map_err(io)?
    } else {
        let mut file = file;
        body(&mut file).map_err(io)?;
        file
    };
    file.flush().map_err(io)
}

/// A displacement field as ITK reads one, `(x, y, z, 1, 3)` vectors under
/// NIfTI's DISPVECT intent (1006) with lengths in millimetres: what
/// nibabel's `Nifti1Image(data, affine)`, `set_intent(1006)` and
/// `set_xyzt_units(xyz="mm")` wrote.  `path` is replaced through a staged
/// file, once the image is whole.
pub fn write_field(array: &NdArray, affine: &Array2<f64>, path: &Path) -> Result<()> {
    refuse_wide(array.dtype())?;
    if affine.dim() != (4, 4) {
        return Err(Error::Value("Affine should be shape 4,4".into()));
    }
    let element = element_of(array.dtype())?;
    let rows: [[f64; 4]; 4] = std::array::from_fn(|r| std::array::from_fn(|c| affine[[r, c]]));
    let mut header = image_header(&array.shape(), element, &rows)?;
    header.intent_code = INTENT_DISPVECT;
    header.xyzt_units = 2;
    let voxels = fortran_bytes(array);
    let staged = crate::h5::file::StagedFile::create(path)?;
    if let Err(err) = save(staged.path(), &header, &voxels) {
        staged.abort();
        return Err(err);
    }
    staged.commit()
}

/// A file written whole beside its target, renamed over it.
fn replace_text(path: &Path, text: Option<&str>) -> Result<()> {
    match text {
        None => match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(crate::error::os_error(&e, path)),
        },
        Some(text) => crate::h5::file::write_atomic(path, text.as_bytes()),
    }
}

/// `json.dumps(fields, indent=2) + "\n"`.
fn sidecar_text(fields: &Map<String, Value>) -> String {
    crate::json::pretty(&Value::Object(fields.clone())) + "\n"
}

/// The fields of the sidecar beside the file, read before anything is
/// written; `None` when there is none, or one that is not a JSON object.
/// One that is not an object is left alone, and refused when this export
/// has fields to put in it --- refused before anything is withdrawn.
fn volume_statements(sidecar: &Path, needed: bool) -> Result<Option<Map<String, Value>>> {
    if !sidecar.exists() {
        return Ok(None);
    }
    let fields = fs::read_to_string(sidecar).ok().and_then(|text| match crate::json::loads_lenient(&text) {
        Ok((Value::Object(map), _)) => Some(map),
        _ => None,
    });
    match fields {
        Some(map) => Ok(Some(map)),
        None if needed => Err(Error::invalid(format!(
            "{} is not a JSON object, and this export writes its per-volume fields there; move it aside first",
            pathlib_str(sidecar)
        ))),
        None => Ok(None),
    }
}

/// Withdraw what was said beside `target` about the volumes it is about to
/// replace --- its `.bval`, and the sidecar's per-volume lists --- and
/// return the sidecar's other fields, kept.  A scalar `EchoTime` is the
/// scanner's, about every volume, and stays.
fn withdraw_volume_statements(
    target: &Path,
    sidecar: &Path,
    fields: Option<&Map<String, Value>>,
) -> Result<Map<String, Value>> {
    replace_text(&sidecar_path(target, ".bval"), None)?;
    let Some(fields) = fields else { return Ok(Map::new()) };
    let kept: Map<String, Value> = fields
        .iter()
        .filter(|(k, v)| !(VOLUME_FIELDS.contains(&k.as_str()) && v.is_array()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if kept.len() != fields.len() {
        let text = (!kept.is_empty()).then(|| sidecar_text(&kept));
        replace_text(sidecar, text.as_deref())?;
    }
    Ok(kept)
}

/// A channel axis as NIfTI states one (§3.6), and what its channels mean.
///
/// With a b-value per channel it is a diffusion series: four dimensions and
/// a `.bval` beside the file.  Otherwise the voxels are vectors: NIfTI-1
/// puts a vector's components on `dim[5]`, `dim[4]` of 1, under a vector
/// intent.  Per-channel echo and inversion times and flip angles go to the
/// sidecar as BIDS fields.
fn channel_layout(
    dims: &mut Vec<usize>,
    channels: &Map<String, Value>,
    report: Option<&mut ConversionReport>,
) -> (Option<i64>, IndexMap<String, Value>, Option<Vec<f64>>) {
    let frames = dims[3];
    let per_channel = |keyword: &str| -> Option<Vec<f64>> {
        let Some(Value::Array(values)) = channels.get(keyword) else { return None };
        if values.len() != frames {
            return None;
        }
        values.iter().map(|v| crate::pyval::to_float(v).ok()).collect()
    };
    let mut statements = IndexMap::new();
    for (keyword, field) in CHANNEL_FIELDS {
        if let Some(found) = per_channel(keyword) {
            statements.insert(field.to_string(), Value::Array(found.into_iter().map(num).collect()));
        }
    }
    let b_values = per_channel("b_values");
    let mut intent = None;
    if b_values.is_none() {
        dims.splice(3..4, [1, frames]);
        intent = Some(INTENT_VECTOR);
    }
    if let Some(report) = report {
        let mut fields: Vec<&String> = statements.keys().collect();
        fields.sort();
        let names: Vec<String> = fields.iter().map(|f| f.to_string()).collect();
        report.decision(
            "channels",
            format!(
                "the fourth axis is a channel axis of {frames}; it was written {}{}",
                if b_values.is_some() {
                    "with its b-values in a .bval beside the file"
                } else {
                    "as NIfTI's vector dimension (dim[5], vector intent)"
                },
                if statements.is_empty() {
                    String::new()
                } else {
                    format!(", and {} in the sidecar", crate::json::repr_list(&names))
                }
            ),
            json!({"channels": frames, "fields": names}),
        );
    }
    (intent, statements, b_values)
}

/// Put a time axis in `header`: its NIfTI unit code, and --- uneven
/// frames, which one step cannot state --- their times in seconds for the
/// sidecar.
fn write_timing(
    header: &mut Header,
    frames: usize,
    time_values: &[f64],
    time_units: Option<&str>,
) -> Result<(i64, Option<Vec<f64>>)> {
    let unit = time_units.filter(|u| seconds_per_unit(u).is_some()).unwrap_or("s");
    if time_values.len() != frames {
        return Err(Error::invalid(format!("{} frame times for a fourth axis of {frames} frames", time_values.len())));
    }
    let steps: Vec<f64> = time_values.windows(2).map(|w| w[1] - w[0]).collect();
    // `np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9)`
    let even =
        steps.iter().all(|s| *s > 0.0) && steps.iter().all(|s| (s - steps[0]).abs() <= 1e-9 + 1e-6 * steps[0].abs());
    if frames < 2 || even {
        if frames >= 2 {
            header.pixdim[4] = steps[0];
        }
        // A fourth axis of no frames has no first time to state.
        if let Some(first) = time_values.first() {
            header.toffset = *first;
        }
        return Ok((if unit == "ms" { 16 } else { 8 }, None));
    }
    header.pixdim[4] = 0.0;
    let scale = seconds_per_unit(unit).unwrap_or(1.0);
    Ok((0, Some(time_values.iter().map(|v| v * scale).collect())))
}

/// Write one NIfTI file, recording `rescale` in the header (§4.2).
///
/// `array` is in NIfTI's axis order, `(x, y, z, ...)` ([`for_export`]), and
/// `affine` its 4x4 index-to-world affine, RAS+, in the grid's units.
/// Lengths are written in millimetres --- a grid in `m` or `um` scaled to
/// them --- and an uncalibrated `px` grid says so with NIfTI's "unknown".  A
/// fourth axis with evenly spaced times becomes `pixdim[4]` and `toffset`;
/// uneven frames are written to a BIDS sidecar's `VolumeTiming` beside the
/// file and the header's step is left at 0.  A fourth axis that is a
/// channel axis is said to be one (see `channel_layout`).
///
/// The file owns the statements about its volumes beside it, and an
/// overwrite replaces them: the new image is written first, the old
/// statements are withdrawn before it replaces the old image, and the new
/// ones written after, so no reader pairs the image with another export's
/// timeline, and an export that fails before its image is whole changes
/// nothing.  The image itself is staged beside the target and renamed over
/// it (§14.4).
pub fn write_nifti(
    array: &NdArray,
    affine: &Array2<f64>,
    path: &Path,
    options: &WriteOptions,
    mut report: Option<&mut ConversionReport>,
) -> Result<PathBuf> {
    let target = nifti_target(path)?;
    let mut matrix = affine.clone();
    let scale = mm_per_unit(&options.units);
    if let Some(scale) = scale {
        for r in 0..matrix.nrows().min(3) {
            for c in 0..matrix.ncols() {
                matrix[[r, c]] *= scale;
            }
        }
    }
    let mut dims = array.shape();
    let mut statements: IndexMap<String, Value> = IndexMap::new();
    let mut b_values = None;
    let mut intent = None;
    let channel = options.leading.as_deref() == Some("channel");
    if dims.len() == 4 && channel {
        (intent, statements, b_values) = channel_layout(&mut dims, &options.channels, report.as_deref_mut());
    }
    refuse_wide(array.dtype())?;
    if matrix.dim() != (4, 4) {
        return Err(Error::Value("Affine should be shape 4,4".into()));
    }
    let element = element_of(array.dtype())?;
    let rows: [[f64; 4]; 4] = std::array::from_fn(|r| std::array::from_fn(|c| matrix[[r, c]]));
    let mut header = image_header(&dims, element, &rows)?;
    if let Some(code) = intent {
        header.intent_code = code;
    }
    if let Some((slope, inter)) = options.rescale {
        (header.scl_slope, header.scl_inter) = slope_inter(slope, inter)?;
    }
    if let (None, Some(report)) = (scale, report.as_deref_mut()) {
        report.decision(
            "units",
            format!(
                "the grid is in {}, which has no physical size; the file's spatial unit is NIfTI's 'unknown'",
                repr_str(&options.units)
            ),
            json!({"units": options.units}),
        );
    }
    let mut temporal = 0;
    if let (Some(times), 4, false) = (&options.time_values, array.ndim(), channel) {
        let (code, timing) = write_timing(&mut header, array.shape()[3], times, options.time_units.as_deref())?;
        temporal = code;
        if let Some(timing) = timing {
            statements.insert("VolumeTiming".into(), Value::Array(timing.into_iter().map(num).collect()));
        }
    }
    header.xyzt_units = if scale.is_some() { 2 } else { 0 } | temporal;
    let sidecar = sidecar_path(&target, ".json");
    let fields = volume_statements(&sidecar, !statements.is_empty())?;
    let voxels = fortran_bytes(array);
    let staged = crate::h5::file::StagedFile::create(&target)?;
    let written = save(staged.path(), &header, &voxels)
        .and_then(|()| withdraw_volume_statements(&target, &sidecar, fields.as_ref()))
        .and_then(|kept| fault("withdrawn").map(|()| kept).map_err(|e| crate::error::os_error(&e, &target)));
    let kept = match written {
        Ok(kept) => kept,
        Err(err) => {
            staged.abort();
            return Err(err);
        }
    };
    drop(voxels);
    staged.commit()?;
    if !statements.is_empty() {
        let mut all = kept;
        let frames = statements.get("VolumeTiming").and_then(Value::as_array).map(Vec::len);
        for (k, v) in statements {
            all.insert(k, v);
        }
        replace_text(&sidecar, Some(&sidecar_text(&all)))?;
        if let (Some(frames), Some(report)) = (frames, report) {
            let name = sidecar.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            report.decision(
                "time_values",
                format!(
                    "the frames are unevenly spaced, which one NIfTI temporal step cannot state; their times were \
                     written to {name} as VolumeTiming"
                ),
                json!({"sidecar": pathlib_str(&sidecar), "frames": frames}),
            );
        }
    }
    if let Some(b_values) = b_values {
        let text: Vec<String> = b_values.iter().map(|v| crate::json::format_g(*v, 6)).collect();
        replace_text(&sidecar_path(&target, ".bval"), Some(&(text.join(" ") + "\n")))?;
    }
    Ok(target)
}

/// Export one image, or one class of one annotation, as a NIfTI file.
///
/// The affine is converted back to RAS+ and the axes back to `(x, y, z)`,
/// so the file opens in FSL, ITK-SNAP or 3D Slicer where it came from.
/// `physical: false` writes the **stored** values with the image's rescale
/// in the header, so a conforming reader gets the physical ones either way.
///
/// An annotation is written on **its own** grid, whatever `image` is on.
/// The whole annotation is its labelmap, where an ignored voxel (§7.7) is
/// the annotation's `ignore_id`; an overlap the flattening loses is a
/// warning in the report.  One class is a 0/1 mask, which has no value for
/// "ignored" and whose 0 says "examined, absent": an annotation with an
/// ignore region is refused, and so is a class it never examined
/// (`annotated_class_ids`).
pub fn to_nifti(
    sample: &Path,
    image: &str,
    out: &Path,
    options: &ExportOptions,
    mut report: Option<&mut ConversionReport>,
) -> Result<PathBuf> {
    let target = nifti_target(out)?;
    let opened = crate::sample::reader::open_sample(sample)?;
    let source = opened.image(image)?;
    let mut grid = source.grid()?.clone();
    let mut rescale = None;
    let data = match &options.annotation {
        Some(ann_id) => {
            let ann = opened.annotation(ann_id)?;
            grid = ann.grid()?.clone();
            match &options.class_key {
                None => {
                    let (volume, overwritten, ordered) = ann.labelmap(None, None, DType::U16)?;
                    if overwritten > 0 {
                        if let Some(report) = report.as_deref_mut() {
                            report.warn(
                                "overlap",
                                ann.flatten_warning(overwritten, &ordered),
                                json!({"annotation": ann_id, "voxels": overwritten}),
                            );
                        }
                    }
                    volume
                }
                Some(key) => {
                    let class_id = ann.resolve_class(key)?;
                    if !ann.annotated_class_ids().contains(&class_id) {
                        return Err(Error::invalid(format!(
                            "annotation {} never examined {} (it is not in its annotated_class_ids), and a \
                             one-class mask's 0 would say it was examined and found absent; export a class it \
                             examined",
                            repr_str(ann_id),
                            repr_str(&ann.class_key(class_id))
                        )));
                    }
                    let planes = ann.dense(Some(&[ClassKey::Id(class_id)]), None)?;
                    let mask: ArrayD<u8> = planes.index_axis(Axis(0), 0).mapv(u8::from);
                    let ignored = opened.ignore_region(ann_id, None)?.iter().filter(|v| **v).count();
                    if ignored > 0 {
                        return Err(Error::invalid(format!(
                            "annotation {} ignores {ignored} voxel(s) (§7.7), which a one-class mask cannot mark --- \
                             its 0 would say they were examined and found absent; export the whole annotation \
                             instead (without a class), whose labelmap writes them as its ignore id {}",
                            repr_str(ann_id),
                            ann.ignore_id()
                        )));
                    }
                    NdArray::from(mask)
                }
            }
        }
        None => {
            let data = source.read(None, options.physical, None)?;
            if !options.physical {
                rescale = Some(source.rescale()?);
            }
            data
        }
    };
    let (data, affine) = for_export(&grid, data)?;
    let leading = (grid.axis_kinds.len() == 4).then(|| grid.axis_kinds[0].clone());
    let mut channels = Map::new();
    if options.annotation.is_none() && leading.as_deref() == Some("channel") {
        if let Some(Value::Object(values)) = opened.document()?.acquisition.get(image) {
            channels = values.iter().filter(|(_, v)| v.is_array()).map(|(k, v)| (k.clone(), v.clone())).collect();
        }
    }
    let write = WriteOptions {
        rescale,
        units: grid.units.clone(),
        time_values: grid.time_values.clone(),
        time_units: grid.time_units.clone(),
        leading,
        channels,
    };
    drop(opened);
    write_nifti(&data, &affine, &target, &write, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s3_1_nifti_bytes_run_first_axis_fastest() {
        let a = NdArray::from(ndarray::arr2(&[[1u8, 2, 3], [4, 5, 6]]).into_dyn());
        assert_eq!(fortran_bytes(&a), [1, 4, 2, 5, 3, 6]);
        // A permuted view's NIfTI order is its source's C order.
        let c = ndarray::Array3::from_shape_fn((2, 3, 4), |(t, y, x)| (t * 12 + y * 4 + x) as i16).into_dyn();
        let flat: Vec<u8> = c.iter().flat_map(|v| v.to_le_bytes()).collect();
        let permuted = NdArray::from(c.permuted_axes(IxDyn(&[2, 1, 0])));
        assert_eq!(fortran_bytes(&permuted), flat);
    }

    #[test]
    fn s4_2_nibabel_refuses_what_nifti_cannot_hold() {
        assert!(matches!(element_of(DType::I64), Err(Error::Value(m)) if m.starts_with("Image data has type int64")));
        assert!(matches!(element_of(DType::Bool), Err(Error::Value(m)) if m == "data dtype \"bool\" not supported"));
        assert!(matches!(check_shape(&[40000, 2, 2]), Err(Error::Value(m))
            if m == "shape (40000, 2, 2) does not fit in dim datatype"));
        assert!(check_shape(&[32767, 1]).is_ok());
        assert!(slope_inter(0.0, 1.0).is_err());
        assert_eq!(slope_inter(f64::NAN, f64::NAN).unwrap(), (1.0, 0.0));
    }

    #[test]
    fn s4_2_numpy_prints_an_affine_as_it_did() {
        // Each against numpy 2's `str()` of the same array.
        let d =
            |a: f64, b: f64, c: f64| [[a, 0.0, 0.0, 0.0], [0.0, b, 0.0, 0.0], [0.0, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let cases: [([[f64; 4]; 4], &str); 7] = [
            (d(1.0, 0.0, 1.0), "[[1. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 1. 0.]\n [0. 0. 0. 1.]]"),
            (
                [[0.5, 0.0, 0.0, -120.25], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 1500.0], [0.0, 0.0, 0.0, 1.0]],
                "[[ 5.0000e-01  0.0000e+00  0.0000e+00 -1.2025e+02]\n [ 0.0000e+00  0.0000e+00  0.0000e+00  0.0000e+00]\n \
                 [ 0.0000e+00  0.0000e+00  3.0000e+00  1.5000e+03]\n [ 0.0000e+00  0.0000e+00  0.0000e+00  1.0000e+00]]",
            ),
            (d(f64::NAN, 1.0, 1.0), "[[nan  0.  0.  0.]\n [ 0.  1.  0.  0.]\n [ 0.  0.  1.  0.]\n [ 0.  0.  0.  1.]]"),
            (
                d(f64::NEG_INFINITY, 1.5, 1.0),
                "[[-inf  0.   0.   0. ]\n [ 0.   1.5  0.   0. ]\n [ 0.   0.   1.   0. ]\n [ 0.   0.   0.   1. ]]",
            ),
            (
                [[1e-5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.5, -3.0], [0.0, 0.0, 0.0, 1.0]],
                "[[ 1.0e-05  0.0e+00  0.0e+00  0.0e+00]\n [ 0.0e+00  0.0e+00  0.0e+00  0.0e+00]\n \
                 [ 0.0e+00  0.0e+00  2.5e+00 -3.0e+00]\n [ 0.0e+00  0.0e+00  0.0e+00  1.0e+00]]",
            ),
            (
                [[-0.0, 0.0, 0.0, 0.1 + 0.2], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0 / 3.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                "[[-0.          0.          0.          0.3       ]\n [ 0.          0.          0.          0.        ]\n \
                 [ 0.          0.          0.33333333  0.        ]\n [ 0.          0.          0.          1.        ]]",
            ),
            (
                [[f64::NAN, 0.0, 0.0, 1e9], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.25, 0.0], [0.0, 0.0, 0.0, 1.0]],
                "[[     nan 0.00e+00 0.00e+00 1.00e+09]\n [0.00e+00 0.00e+00 0.00e+00 0.00e+00]\n \
                 [0.00e+00 0.00e+00 1.25e+00 0.00e+00]\n [0.00e+00 0.00e+00 0.00e+00 1.00e+00]]",
            ),
        ];
        for (affine, printed) in cases {
            assert_eq!(numpy_str(&affine), printed);
        }
    }

    #[test]
    fn s4_2_an_affine_with_no_rotation_is_refused_as_nibabel_refused_it() {
        // A zero column divides to NaN, on which numpy's SVD does not
        // converge: nibabel's `HeaderDataError`, the affine printed.
        let mut affine = eye();
        affine[[1, 1]] = 0.0;
        let out = tempfile::tempdir().unwrap();
        let err = write_nifti(&series(2, 0), &affine, &out.path().join("flat.nii"), &WriteOptions::default(), None);
        assert!(matches!(err, Err(Error::Value(m))
            if m == "Could not decompose affine:\n[[1. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 1. 0.]\n [0. 0. 0. 1.]]"));
        assert_eq!(fs::read_dir(out.path()).unwrap().count(), 0, "nothing written");
    }

    #[test]
    fn s3_6_a_plane_is_embedded_at_z_zero() {
        let plane = ndarray::arr2(&[[0.9, 0.0, -20.0], [0.0, 0.8, -10.0], [0.0, 0.0, 1.0]]);
        let out = embed_plane(&plane).unwrap();
        assert_eq!(out.row(0).to_vec(), [0.9, 0.0, 0.0, -20.0]);
        assert_eq!(out.row(2).to_vec(), [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn s3_2_frame_times_are_a_step_or_a_sidecar() {
        let mut header = Header::new(Version::One);
        assert_eq!(write_timing(&mut header, 3, &[10.0, 12.5, 15.0], Some("s")).unwrap(), (8, None));
        assert_eq!((header.pixdim[4], header.toffset), (2.5, 10.0));
        let (code, timing) = write_timing(&mut header, 3, &[0.0, 1000.0, 3000.0], Some("ms")).unwrap();
        assert_eq!((code, timing), (0, Some(vec![0.0, 1.0, 3.0])));
        assert_eq!(header.pixdim[4], 0.0);
        assert!(write_timing(&mut header, 2, &[0.0], None).is_err());
        // No frames: nothing to state, and no first time to read.
        let mut header = Header::new(Version::One);
        assert_eq!(write_timing(&mut header, 0, &[], Some("ms")).unwrap(), (16, None));
        assert_eq!((header.pixdim[4], header.toffset), (Header::new(Version::One).pixdim[4], 0.0));
        let out = tempfile::tempdir().unwrap();
        let options = WriteOptions { time_values: Some(Vec::new()), ..WriteOptions::default() };
        write_nifti(&series(0, 0), &eye(), &out.path().join("empty.nii"), &options, None).unwrap();
    }

    /// The files of `dir`, by name, with their bytes.
    fn files(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut out: Vec<(String, Vec<u8>)> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), fs::read(&p).unwrap()))
            .collect();
        out.sort();
        out
    }

    fn interrupted<T>(at: &'static str, step: impl FnOnce() -> Result<T>) -> Result<T> {
        FAULT.with(|f| f.set(Some(at)));
        let done = step();
        FAULT.with(|f| f.set(None));
        done
    }

    fn series(frames: usize, value: i16) -> NdArray {
        NdArray::from(ndarray::ArrayD::from_elem(IxDyn(&[2, 3, 4, frames]), value))
    }

    fn eye() -> Array2<f64> {
        Array2::eye(4)
    }

    #[test]
    fn s14_4_an_interrupted_save_leaves_the_previous_file() {
        // The image is written to a staged temporary beside it, renamed over
        // it only once whole: stopped half way, the old file stands.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("ct.nii.gz");
        let volume = |v: i16| NdArray::from(ndarray::ArrayD::from_elem(IxDyn(&[4, 3, 2]), v));
        write_nifti(&volume(0), &eye(), &out, &WriteOptions::default(), None).unwrap();
        let before = files(dir.path());
        let err = interrupted("save", || write_nifti(&volume(1), &eye(), &out, &WriteOptions::default(), None));
        assert!(err.unwrap_err().message().contains("interrupted"));
        assert_eq!(files(dir.path()), before, "the old file, and no temporary");
    }

    #[test]
    fn s3_2_an_export_that_cannot_write_its_image_changes_nothing() {
        // The old statements beside the file are withdrawn only once the new
        // image is whole: a save that fails leaves the old image its timing
        // and its b-values.
        let dir = tempfile::tempdir().unwrap();
        let uneven = WriteOptions::default().with_time_values(Some(vec![0.0, 1.0, 3.0]), Some("s".into()));
        let out = dir.path().join("dce.nii.gz");
        write_nifti(&series(3, 0), &eye(), &out, &uneven, None).unwrap();
        let mut channels = Map::new();
        channels.insert("b_values".into(), json!([0.0, 900.0]));
        let dwi = WriteOptions::default().with_leading(Some("channel".into())).with_channels(channels);
        write_nifti(&series(2, 0), &eye(), &dir.path().join("dwi.nii.gz"), &dwi, None).unwrap();
        let before = files(dir.path());
        assert_eq!(before.len(), 4, "each image with its timing or its b-values");
        let again = WriteOptions::default().with_time_values(Some(vec![0.0, 5.0, 6.0]), Some("s".into()));
        assert!(interrupted("save", || write_nifti(&series(3, 1), &eye(), &out, &again, None)).is_err());
        let dwi_out = dir.path().join("dwi.nii.gz");
        assert!(interrupted("save", || write_nifti(&series(2, 1), &eye(), &dwi_out, &dwi, None)).is_err());
        assert_eq!(files(dir.path()), before);
    }

    #[test]
    fn s3_2_an_interrupted_overwrite_never_reads_another_timeline() {
        // Withdrawn before the image is replaced: an export stopped between
        // the two leaves the old image with its timing unstated, never the
        // new image with the old timing or the old image with the new.
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("dce.nii.gz");
        let uneven = |times: Vec<f64>| WriteOptions::default().with_time_values(Some(times), Some("s".into()));
        write_nifti(&series(3, 0), &eye(), &out, &uneven(vec![0.0, 1.0, 3.0]), None).unwrap();
        let image = fs::read(&out).unwrap();
        let err =
            interrupted("withdrawn", || write_nifti(&series(3, 1), &eye(), &out, &uneven(vec![0.0, 5.0, 6.0]), None));
        assert!(err.is_err());
        assert_eq!(files(dir.path()), [("dce.nii.gz".to_string(), image)]);
    }

    #[test]
    fn s14_4_a_field_is_written_whole_or_not_at_all() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("warp.nii.gz");
        let field = NdArray::from(ndarray::ArrayD::from_elem(IxDyn(&[2, 3, 4, 1, 3]), 0.5f32));
        write_field(&field, &eye(), &out).unwrap();
        let mut raw = Vec::new();
        std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(fs::File::open(&out).unwrap()), &mut raw).unwrap();
        let header = Header::decode(&raw[..348], Version::One).unwrap();
        assert_eq!((header.intent_code, header.xyzt_units, header.datatype), (1006, 2, 16));
        assert_eq!(&header.dim[..6], &[5, 2, 3, 4, 1, 3]);
        let before = files(dir.path());
        assert!(interrupted("save", || write_field(&field, &eye(), &out)).is_err());
        assert_eq!(files(dir.path()), before);
    }

    #[test]
    fn s3_sidecars_share_the_stem() {
        assert_eq!(sidecar_path(Path::new("/d/x.nii.gz"), ".json"), Path::new("/d/x.json"));
        assert_eq!(sidecar_path(Path::new("/d/x.nii"), ".bval"), Path::new("/d/x.bval"));
        #[cfg(unix)]
        {
            // "caf\xe9": no UTF-8.  Its sidecar keeps the byte the reader
            // looks for, not the lossy "caf\u{fffd}".
            use std::ffi::OsStr;
            use std::os::unix::ffi::OsStrExt;
            let target = Path::new("/d").join(OsStr::from_bytes(b"caf\xe9.nii.gz"));
            assert_eq!(sidecar_path(&target, ".json"), Path::new("/d").join(OsStr::from_bytes(b"caf\xe9.json")));
        }
        assert!(nifti_target(Path::new("/d/x.img")).is_err());
    }
}
