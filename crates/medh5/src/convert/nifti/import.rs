//! [`read_nifti`] and [`from_nifti`]: one file in MEDH5 conventions, and a
//! sample from a set of co-registered files.
//!
//! Each step is 2.1.0's, in 2.1.0's order, because which of two problems a
//! file is refused for is part of the contract (2.2.0 decision E3).

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use ndarray::{Array2, ArrayD};
use serde_json::{json, Map, Value};

use super::header::Header;
use super::read::{self, Image, LoadError, Loaded};
use super::{
    convert_world, float_list, Geometry, GeometryNote, ImportOptions, NiftiRead, Plane, ReadOptions, SegOptions,
    Voxels, FOURTH_AXES,
};
use crate::convert::grouping::{sanitize_key, sanitize_stem};
use crate::convert::ConversionReport;
use crate::geometry::affine::{decompose_affine, ORTHONORMAL_TOL};
use crate::geometry::grid::mm_per_unit;
use crate::geometry::linalg::allclose;
use crate::json::{format_g, num, py_float, repr_int_list, repr_int_tuple, repr_list, repr_str};
use crate::labels::{ClassKey, LabelClass, LabelSet};
use crate::pyval::title_case;
use crate::sample::writer::{create, Annotated, GridOptions, ImageOptions, SampleWriter};
use crate::sample::writer_annotations::{AnnotationOptions, SegmentationOptions, SegmentationSource};
use crate::{Error, Result};

/// NIfTI intent codes whose extra dimension holds components, not frames:
/// vectors, matrices and the RGB(A) ones.
const CHANNEL_INTENTS: [i64; 7] = [1001, 1004, 1005, 1006, 1007, 2003, 2004];

/// NIfTI's time-series intent.
const TIME_SERIES_INTENT: i64 = 2001;

/// BIDS sidecar fields holding one value per volume that make the fourth
/// axis a channel axis: the field, the DICOM keyword `acquisition` stores
/// them under (§4.5), and a short channel label.
const PER_VOLUME_CHANNEL: [(&str, &str, &str); 5] = [
    ("EchoTime", "EchoTime", "TE"),
    ("EchoTimes", "EchoTime", "TE"),
    ("EchoNumber", "EchoNumbers", "echo"),
    ("InversionTime", "InversionTime", "TI"),
    ("FlipAngle", "FlipAngle", "FA"),
];

/// BIDS sidecar fields holding one acquisition time per volume.
const PER_VOLUME_TIME: [&str; 1] = ["VolumeTiming"];

/// How far from z = 0 a 2-D plane may lie, in millimetres.
const PLANE_TOL_MM: f64 = 1e-3;

/// The names a NIfTI file ends in, which its sidecars replace.
const NIFTI_SUFFIXES: [&str; 2] = [".nii.gz", ".nii"];

/// A path as the caller gave it, for messages.
fn shown(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// What `path` names, as a NIfTI image, or why it is none.
fn load(path: &Path) -> Result<Image> {
    let name = shown(path);
    let image = match read::load(path) {
        Ok(Loaded::Nifti(image)) => *image,
        Ok(Loaded::Other(class)) => {
            return Err(Error::invalid(format!("{name} is not a NIfTI file: nibabel reads it as {class}")))
        }
        Err(LoadError::NotFound) => {
            return Err(Error::Io(format!("[Errno 2] No such file or no access: {}", repr_str(&name))))
        }
        Err(LoadError::NotAnImage(why)) => return Err(Error::invalid(format!("{name} is not a NIfTI file: {why}"))),
        Err(LoadError::Broken(err)) => return Err(err),
    };
    // The affine a reader takes from the header is built as it loads: a
    // qform that is the best affine and is no rotation is refused then.
    let header = &image.header;
    if header.sform_code == 0 && header.qform_code != 0 {
        header.qform_quaternion()?;
    }
    Ok(image)
}

type Matrix4 = [[f64; 4]; 4];

/// What the header left ambiguous: no spatial mapping at all (refused
/// unless assumed), or an sform and a qform that disagree.
fn geometry_notes(header: &Header, name: &str, assume_geometry: bool) -> Result<Vec<GeometryNote>> {
    let mut notes = Vec::new();
    let (sform_code, qform_code) = (header.sform_code, header.qform_code);
    if sform_code == 0 && qform_code == 0 {
        if !assume_geometry {
            return Err(Error::invalid(format!(
                "{} declares no spatial mapping (sform_code = qform_code = 0), so it has no world geometry to \
                 import; nibabel's fallback affine is built from pixdim and is not a measurement. Pass \
                 assume_geometry=True (CLI: --assume-geometry) to accept that fallback deliberately, and it will \
                 be recorded as a guess.",
                repr_str(name)
            )));
        }
        notes.push(GeometryNote {
            kind: "geometry".into(),
            message: "the file declares no spatial mapping (sform_code = qform_code = 0); the grid was taken from \
                      pixdim and is assumed, not measured"
                .into(),
            detail: json!({"sform_code": 0, "qform_code": 0}),
        });
        return Ok(notes);
    }
    if sform_code != 0 && qform_code != 0 {
        let (sform, qform) = (header.sform(), own_qform(header, name)?);
        let a: Vec<f64> = sform.iter().flatten().copied().collect();
        let b: Vec<f64> = qform.iter().flatten().copied().collect();
        if !allclose(&a, &b, 1e-4, 0.0) {
            let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(f64::NEG_INFINITY, nan_max);
            notes.push(GeometryNote {
                kind: "geometry".into(),
                message: "sform and qform describe different geometry; the sform was used. A reader preferring \
                          the qform will place this volume elsewhere."
                    .into(),
                detail: json!({"sform_code": sform_code, "qform_code": qform_code, "max_abs_difference": num(worst)}),
            });
        }
    }
    Ok(notes)
}

/// `max` as NumPy takes it: a NaN anywhere is the answer.
fn nan_max(acc: f64, value: f64) -> f64 {
    if acc.is_nan() || value.is_nan() {
        f64::NAN
    } else {
        acc.max(value)
    }
}

/// The qform, in `f64`, as NIfTI-1 defines it: `a` is zero within the
/// stored precision of zero (and `(b, c, d)` normalised), and below it the
/// parameters are no rotation.
fn own_qform(header: &Header, name: &str) -> Result<Matrix4> {
    let (mut b, mut c, mut d) = (header.quatern_b, header.quatern_c, header.quatern_d);
    let w2 = 1.0 - (b * b + c * c + d * d);
    let a = if w2.abs() < f64::from(f32::EPSILON) {
        let norm = (b * b + c * c + d * d).sqrt();
        (b, c, d) = (b / norm, c / norm, d / norm);
        0.0
    } else if w2 < 0.0 {
        return Err(Error::invalid(format!(
            "{name}: the qform's quaternion parameters (b, c, d) = ({}, {}, {}) have b² + c² + d² > 1, so they \
             are no rotation",
            py_float(b),
            py_float(c),
            py_float(d)
        )));
    } else {
        w2.sqrt()
    };
    let pixdim = &header.pixdim[..4];
    if pixdim[1..].iter().any(|v| *v < 0.0) {
        return Err(Error::invalid(format!(
            "{name}: pixdim[1:4] = {} holds a negative voxel size, which the qform cannot scale by",
            float_list(&pixdim[1..])
        )));
    }
    let qfac = if pixdim[0] < 0.0 { -1.0 } else { 1.0 };
    let rotation = [
        [a * a + b * b - c * c - d * d, 2.0 * b * c - 2.0 * a * d, 2.0 * b * d + 2.0 * a * c],
        [2.0 * b * c + 2.0 * a * d, a * a + c * c - b * b - d * d, 2.0 * c * d - 2.0 * a * b],
        [2.0 * b * d - 2.0 * a * c, 2.0 * c * d + 2.0 * a * b, a * a + d * d - c * c - b * b],
    ];
    let scale = [pixdim[1], pixdim[2], qfac * pixdim[3]];
    let offset = [header.qoffset_x, header.qoffset_y, header.qoffset_z];
    let mut out = super::header::identity();
    for row in 0..3 {
        for column in 0..3 {
            out[row][column] = rotation[row][column] * scale[column];
        }
        out[row][3] = offset[row];
    }
    Ok(out)
}

/// The file's index-to-RAS affine: the sform where `sform_code` is set,
/// else the qform where `qform_code` is, else the fallback from `pixdim`.
fn best_affine(header: &Header, name: &str) -> Result<Matrix4> {
    if header.sform_code == 0 && header.qform_code != 0 {
        return own_qform(header, name);
    }
    Ok(if header.sform_code != 0 { header.sform() } else { header.base_affine() })
}

/// The spatial unit in the §3.5 vocabulary, and what the header said where
/// that is no unit of it.
fn units(header: &Header) -> (String, Option<String>) {
    let spatial = header.xyzt_units().map_or("unreadable", |(xyz, _)| xyz);
    match spatial {
        "meter" => ("m".into(), None),
        "mm" => ("mm".into(), None),
        "micron" => ("um".into(), None),
        other => ("mm".into(), Some(other.to_string())),
    }
}

/// `scl_slope`/`scl_inter` as a §4.2 rescale, or `None` when nothing scales.
fn rescale(image: &Image) -> Option<(f64, f64)> {
    match image.slope_inter {
        Some((slope, inter)) if !(slope == 1.0 && inter == 0.0) => Some((slope, inter)),
        _ => None,
    }
}

/// The temporal zoom and offset in the returned unit, the unit, and
/// whether the file states them: a positive zoom *and* a temporal unit.
fn temporal(header: &Header) -> (f64, f64, &'static str, bool) {
    let zooms = header.zooms();
    let step = zooms.get(3).copied().unwrap_or(0.0);
    let unit = header.xyzt_units().map_or("unknown", |(_, t)| t);
    let (units, scale) = match unit {
        "sec" => ("s", 1.0),
        "msec" => ("ms", 1.0),
        "usec" => ("ms", 1e-3),
        _ => return (0.0, 0.0, "s", false),
    };
    if step <= 0.0 {
        return (0.0, 0.0, "s", false);
    }
    (step * scale, header.toffset * scale, units, true)
}

/// Per-frame times, their unit, and whether they were read or assumed.
fn time_axis(header: &Header, frames: usize, stamps: Option<&[f64]>, on_dim4: bool) -> (Vec<f64>, String, bool) {
    if let Some(stamps) = stamps {
        return (stamps.to_vec(), "s".into(), true);
    }
    let (step, offset, units, stated) = if on_dim4 { temporal(header) } else { (0.0, 0.0, "s", false) };
    if !stated {
        return ((0..frames).map(|k| k as f64).collect(), "s".into(), false);
    }
    ((0..frames).map(|k| offset + k as f64 * step).collect(), units.into(), true)
}

/// The existing file beside a NIfTI sharing its stem and ending in
/// `suffix`; `None` when there is none, or `path` is no NIfTI name.
fn sidecar(path: &Path, suffix: &str) -> Option<PathBuf> {
    let name = path.file_name()?;
    let ending = NIFTI_SUFFIXES.iter().find(|s| name.to_string_lossy().ends_with(*s))?;
    let candidate = path.with_file_name(swap_suffix(name, ending.len(), suffix));
    candidate.exists().then_some(candidate)
}

/// `name` less its last `cut` bytes, an ASCII suffix, then `suffix`.  On Unix
/// the stem keeps the name's own bytes, so a name that is not UTF-8 finds its
/// own sidecars, as 2.1.0 did through `surrogateescape`.
pub(super) fn swap_suffix(name: &OsStr, cut: usize, suffix: &str) -> OsString {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};
        let bytes = name.as_bytes();
        OsString::from_vec([&bytes[..bytes.len() - cut], suffix.as_bytes()].concat())
    }
    #[cfg(not(unix))]
    {
        let name = name.to_string_lossy();
        OsString::from(format!("{}{suffix}", &name[..name.len() - cut]))
    }
}

/// A `.bval` file's b-values.
fn read_bvals(path: &Path) -> Result<Vec<f64>> {
    let bytes = std::fs::read(path).map_err(|e| crate::error::os_error(&e, path))?;
    let text = String::from_utf8(bytes).map_err(|e| {
        Error::Value(format!(
            "'utf-8' codec can't decode byte 0x{:02x} in position {}: invalid start byte",
            e.as_bytes()[e.utf8_error().valid_up_to()],
            e.utf8_error().valid_up_to()
        ))
    })?;
    text.split_whitespace().map(|token| crate::pyval::to_float(&Value::String(token.into()))).collect()
}

/// A BIDS sidecar as a mapping; one that is malformed or unreadable reads
/// as absent.
fn read_json(path: &Path) -> Map<String, Value> {
    let Ok(text) = std::fs::read_to_string(path) else { return Map::new() };
    match crate::json::loads_lenient(&text) {
        Ok((Value::Object(map), _)) => map,
        _ => Map::new(),
    }
}

/// The first of `fields` holding exactly `frames` numbers, and its values.
fn per_volume<'a>(
    sidecar: &Map<String, Value>,
    fields: impl IntoIterator<Item = &'a str>,
    frames: usize,
) -> Option<(String, Vec<f64>)> {
    for field in fields {
        let Some(Value::Array(items)) = sidecar.get(field) else { continue };
        if items.len() != frames {
            continue;
        }
        if let Ok(values) = items.iter().map(crate::pyval::to_float).collect::<Result<Vec<f64>>>() {
            return Some((field.to_string(), values));
        }
    }
    None
}

/// What the fourth axis is, by a BIDS JSON sidecar: `(kind, field, values)`.
fn bids_axis(path: &Path, frames: usize, requested: &str) -> Result<Option<(String, String, Vec<f64>)>> {
    let Some(beside) = sidecar(path, ".json") else { return Ok(None) };
    let fields = read_json(&beside);
    let mut channel = per_volume(&fields, PER_VOLUME_CHANNEL.iter().map(|f| f.0), frames);
    let mut timing = per_volume(&fields, PER_VOLUME_TIME, frames);
    if let (Some(c), Some(t)) = (&channel, &timing) {
        if requested == "auto" {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            return Err(Error::invalid(format!(
                "the sidecar beside {name} states both {} and {} per volume, so it says the fourth axis is a \
                 channel axis and a time axis at once (§3.6) --- pass `fourth_axis=` to settle it",
                c.0, t.0
            )));
        }
        // The caller settled it: keep only the evidence that agrees.
        if requested != "channel" {
            channel = None;
        }
        if requested != "time" {
            timing = None;
        }
    }
    Ok(match (channel, timing) {
        (Some((field, values)), _) => Some(("channel".into(), field, values)),
        (None, Some((field, values))) => Some(("time".into(), field, values)),
        (None, None) => None,
    })
}

/// Whether a 4-D file's extra axis is `time` or `channel` (§3.6), and
/// whether the source stated it.
fn fourth_axis_kind(
    header: &Header,
    path: &Path,
    requested: &str,
    bids: Option<&(String, String, Vec<f64>)>,
    on_dim4: bool,
) -> Result<(String, bool)> {
    if !FOURTH_AXES.contains(&requested) {
        return Err(Error::invalid(format!(
            "unknown fourth_axis {}; expected one of {}",
            repr_str(requested),
            repr_list(&FOURTH_AXES)
        )));
    }
    if requested != "auto" {
        return Ok((requested.into(), true));
    }
    if sidecar(path, ".bval").is_some() {
        return Ok(("channel".into(), true));
    }
    if let Some((kind, _, _)) = bids {
        return Ok((kind.clone(), true));
    }
    if CHANNEL_INTENTS.contains(&header.intent_code) {
        return Ok(("channel".into(), true));
    }
    if header.intent_code == TIME_SERIES_INTENT {
        return Ok(("time".into(), true));
    }
    if !on_dim4 {
        return Ok(("channel".into(), false));
    }
    Ok(("time".into(), temporal(header).3))
}

fn column(affine: &Array2<f64>, k: usize) -> [f64; 3] {
    [affine[[0, k]], affine[[1, k]], affine[[2, k]]]
}

fn norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Where a 2-D plane read as `(x, y, 1)` lies, or a refusal: its third
/// axis is the affine's third column, which must be non-zero and
/// orthogonal to the in-plane axes.
fn plane_volume(affine: &Array2<f64>, name: &str, units: &str) -> Result<Plane> {
    let third = column(affine, 2);
    let length = norm(&third);
    if !(length.is_finite() && length > 0.0) {
        return Err(Error::invalid(format!(
            "{name} is a 2-D plane whose affine's third column is {}: read as (x, y, 1), that column is the third \
             axis, and a zero one gives the slice no normal and no thickness, which are not invented; write them \
             into the file's affine, or import the plane without plane_as_volume",
            float_list(&third)
        )));
    }
    let normal = [third[0] / length, third[1] / length, third[2] / length];
    for k in 0..2 {
        let axis = column(affine, k);
        let size = norm(&axis);
        let dot = normal[0] * axis[0] + normal[1] * axis[1] + normal[2] * axis[2];
        if size > 0.0 && (dot / size).abs() > ORTHONORMAL_TOL {
            return Err(Error::invalid(format!(
                "{name} is a 2-D plane whose affine's third column {} is not orthogonal to its in-plane axes: read \
                 as (x, y, 1), the grid would be sheared, and no other normal is invented; correct the file's \
                 affine, or import the plane without plane_as_volume",
                float_list(&third)
            )));
        }
    }
    Ok(Plane { z: affine[[2, 3]], normal: normal.to_vec(), thickness: length, units: units.into() })
}

/// A 2-D plane's geometry on the first two world axes, or a refusal: only a
/// plane parallel to the x-y plane, at z = 0, has a 2-D grid.
fn reduce_plane(
    spacing: &[f64],
    origin: &[f64],
    direction: &Array2<f64>,
    units: &str,
    name: &str,
) -> Result<(Vec<f64>, Vec<f64>, Array2<f64>)> {
    let tilt = nan_max(direction[[2, 0]].abs(), direction[[2, 1]].abs());
    if tilt > ORTHONORMAL_TOL {
        return Err(Error::invalid(format!(
            "{name} is a 2-D plane not parallel to the x-y world plane, and §3.6 gives a 2-D grid a 2x2 direction \
             on the first two world axes --- no 2-D grid describes it; save it as (x, y, 1), or pass \
             plane_as_volume=True (--plane-as-volume), to import it as a single-slice 3-D volume"
        )));
    }
    let offset = origin[2] * mm_per_unit(units).unwrap_or(1.0);
    if offset.abs() > PLANE_TOL_MM {
        return Err(Error::invalid(format!(
            "{name} is a 2-D plane at world z = {} mm, and a 2-D grid (§3.6) has no z to keep it at: imported, it \
             would lie at z = 0, on one grid with planes from any other slice; save it as (x, y, 1), or pass \
             plane_as_volume=True (--plane-as-volume), to import it as a single-slice 3-D volume, which keeps its z",
            format_g(offset, 6)
        )));
    }
    let reduced = direction.slice(ndarray::s![..2, ..2]).to_owned();
    Ok((spacing[..2].to_vec(), origin[..2].to_vec(), reduced))
}

/// Fortran-order bytes of `shape`, reordered by `order` into C order.
fn c_order(bytes: Vec<u8>, item: usize, shape: &[usize], order: &[usize]) -> Vec<u8> {
    let n = shape.len();
    // An F-order array is its reversed-shape C-order array: a full reversal
    // moves no byte, and nor does any order over one axis that is not 1.
    if shape.contains(&0) {
        return Vec::new();
    }
    let reversed = order.iter().copied().eq((0..n).rev());
    let kept: Vec<usize> = order.iter().copied().filter(|k| shape[*k] != 1).collect();
    if reversed || kept.len() <= 1 || kept.windows(2).all(|w| w[0] > w[1]) {
        return bytes;
    }
    let mut stride = vec![0usize; n];
    let mut acc = 1usize;
    for (axis, extent) in shape.iter().enumerate() {
        stride[axis] = acc;
        acc *= extent;
    }
    let out_shape: Vec<usize> = order.iter().map(|k| shape[*k]).collect();
    let steps: Vec<usize> = order.iter().map(|k| stride[*k]).collect();
    let total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total * item);
    if total == 0 {
        return out;
    }
    let mut index = vec![0usize; n];
    loop {
        let at = index.iter().zip(&steps).map(|(i, s)| i * s).sum::<usize>() * item;
        out.extend_from_slice(&bytes[at..at + item]);
        let mut k = n;
        loop {
            if k == 0 {
                return out;
            }
            k -= 1;
            index[k] += 1;
            if index[k] < out_shape[k] {
                break;
            }
            index[k] = 0;
        }
    }
}

/// One NIfTI file as voxels and geometry in MEDH5 conventions (§3, §3.6).
///
/// The world frame is converted from NIfTI's RAS to `coord_system`, on the
/// affine only; `(x, y, z)` is reordered to `(z, y, x)` under `transpose`,
/// a fourth axis leading; trailing axes of extent 1 past the third are
/// dropped.  A 4-D series' extra axis is `time` or `channel`, read from a
/// `.bval` or BIDS JSON sidecar, the intent code or the temporal zoom, or
/// as `fourth_axis` says.  A file that declares no spatial mapping is
/// refused unless `assume_geometry`; a 2-D plane is a 2-D grid only at
/// z = 0 in the world's x-y plane, and every 2-D file a one-slice volume
/// under `plane_as_volume`.  What was decided from an ambiguous header is
/// in the geometry's `notes`.
pub fn read_nifti(path: &Path, options: &ReadOptions) -> Result<NiftiRead> {
    let name = shown(path);
    let image = load(path)?;
    let header = &image.header;
    let mut notes = geometry_notes(header, &name, options.assume_geometry)?;
    let (units, stated) = units(header);
    if let Some(stated) = stated {
        notes.push(GeometryNote {
            kind: "units".into(),
            message: format!(
                "the file states its spatial unit as {}, so its lengths were read as millimetres, as NIfTI readers \
                 commonly do",
                repr_str(&stated)
            ),
            detail: json!({ "xyz_units": stated }),
        });
    }
    let rescale = rescale(&image);
    let voxels = image.read_voxels()?;
    let mut shape = voxels.shape.clone();
    // Axes of extent 1 past the spatial block carry nothing.
    let squeezed: Vec<usize> = (3..shape.len()).rev().filter(|a| shape[*a] == 1).collect();
    // NIfTI's time is dim[4]; an axis kept past it is dim[5].
    let kept: Vec<usize> = (3..shape.len()).filter(|a| !squeezed.contains(a)).collect();
    let on_dim4 = kept == [3];
    for axis in &squeezed {
        shape.remove(*axis);
    }
    let affine = best_affine(header, &name)?;
    let affine = Array2::from_shape_fn((4, 4), |(r, c)| affine[r][c]);
    let affine = convert_world(&affine, "RAS", &options.coord_system)?;
    let mut plane = None;
    if options.plane_as_volume && shape.len() == 2 {
        plane = Some(plane_volume(&affine, &name, &units)?);
        shape.push(1);
    }
    let ndim = shape.len();
    let (mut spacing, mut origin, mut direction) = decompose_affine(&affine)?;
    let mut order: Vec<usize> = (0..ndim).collect();
    if options.transpose && ndim >= 3 {
        order = (3..ndim).chain([2, 1, 0]).collect();
        spacing = vec![spacing[2], spacing[1], spacing[0]];
        direction = Array2::from_shape_fn((3, 3), |(r, c)| direction[[r, 2 - c]]);
    }
    if ndim == 2 {
        (spacing, origin, direction) = reduce_plane(&spacing, &origin, &direction, &units, &name)?;
    }
    let mut geometry = Geometry {
        notes,
        spacing,
        origin,
        direction: direction.rows().into_iter().map(|r| r.to_vec()).collect(),
        coord_system: options.coord_system.clone(),
        shape: order.iter().map(|k| shape[*k]).collect(),
        axis_order: order.clone(),
        time_values: None,
        time_units: None,
        time_measured: false,
        leading_kind: "spatial".into(),
        leading_stated: false,
        b_values: None,
        channel_field: None,
        channel_values: None,
        stated_by: None,
        squeezed,
        units,
        dtype: String::new(),
        rescale,
        plane_as_volume: plane,
        descrip: Header::text(&header.descrip),
    };
    if ndim == 4 {
        // The fourth axis is the file's dim[4], whichever way it was ordered.
        let frames = shape[3];
        let bids = bids_axis(path, frames, &options.fourth_axis)?;
        let (kind, stated) = fourth_axis_kind(header, path, &options.fourth_axis, bids.as_ref(), on_dim4)?;
        geometry.stated_by = bids.as_ref().filter(|b| b.0 == kind).map(|b| b.1.clone());
        if kind == "time" {
            let stamps = bids.as_ref().filter(|b| b.0 == "time").map(|b| b.2.as_slice());
            let (times, units, measured) = time_axis(header, frames, stamps, on_dim4);
            geometry.time_values = Some(times);
            geometry.time_units = Some(units);
            geometry.time_measured = measured;
        } else {
            if let Some(bval) = sidecar(path, ".bval") {
                geometry.b_values = Some(read_bvals(&bval)?);
            }
            if let Some((_, field, values)) = bids.filter(|b| b.0 == "channel") {
                geometry.channel_field = Some(field);
                geometry.channel_values = Some(values);
            }
            geometry.time_measured = stated;
        }
        geometry.leading_kind = kind;
        geometry.leading_stated = geometry.time_measured;
    }
    let item = voxels.element.itemsize();
    // The geometry keeps a scalar's shape, `()`; its array, as 2.1.0's
    // `ascontiguousarray` returned it, is `(1,)`.
    let held = if geometry.shape.is_empty() { vec![1] } else { geometry.shape.clone() };
    let mut voxels = Voxels { shape: held, ..voxels };
    voxels.bytes = c_order(std::mem::take(&mut voxels.bytes), item, &shape, &order);
    geometry.dtype = voxels.numpy_str();
    Ok(NiftiRead { voxels, geometry })
}

/// The geometry notes of a read, as guesses in the report.
fn replay_notes(report: &mut ConversionReport, name: &str, geometry: &Geometry) {
    for note in &geometry.notes {
        report.guess(&note.kind, format!("{name}: {}", note.message), note.detail.clone());
    }
}

/// Record a plane read as `(x, y, 1)`; whether there was one.
fn note_plane(report: &mut ConversionReport, name: &str, geometry: &Geometry) -> bool {
    let Some(plane) = &geometry.plane_as_volume else { return false };
    let mut detail = Map::new();
    detail.insert("file".into(), json!(name));
    if let Value::Object(fields) = plane.to_json() {
        detail.extend(fields);
    }
    report.decision(
        "plane_as_volume",
        format!(
            "{name}: the 2-D plane was imported as a one-slice 3-D grid, (x, y, 1), at z = {} {units}, its third \
             axis the affine's third column (normal {}, {} {units} thick) (§3.6)",
            format_g(plane.z, 6),
            float_list(&plane.normal),
            format_g(plane.thickness, 6),
            units = plane.units,
        ),
        Value::Object(detail),
    );
    true
}

/// `spacing` or `origin` in millimetres, where the unit has a size.
fn lengths(geometry: &Geometry, values: &[f64]) -> Vec<f64> {
    let scale = mm_per_unit(&geometry.units).unwrap_or(1.0);
    values.iter().map(|v| v * scale).collect()
}

/// The first of units, spacing, origin and direction two geometries
/// disagree on.
fn geometry_mismatch(first: &Geometry, other: &Geometry) -> Option<&'static str> {
    if (first.units == "px") != (other.units == "px") {
        return Some("units");
    }
    let pairs = [
        ("spacing", lengths(first, &first.spacing), lengths(other, &other.spacing)),
        ("origin", lengths(first, &first.origin), lengths(other, &other.origin)),
        ("direction", first.direction.concat(), other.direction.concat()),
    ];
    for (key, ours, theirs) in pairs {
        let same_shape =
            key != "direction" || first.direction.iter().map(Vec::len).eq(other.direction.iter().map(Vec::len));
        if !same_shape || !allclose(&ours, &theirs, 1e-4, 1e-5) {
            return Some(key);
        }
    }
    None
}

/// The frame times in seconds.
fn seconds(geometry: &Geometry) -> Option<Vec<f64>> {
    let values = geometry.time_values.as_ref()?;
    let scale = match geometry.time_units.as_deref().unwrap_or("s") {
        "ms" => 1e-3,
        _ => 1.0,
    };
    Some(values.iter().map(|v| v * scale).collect())
}

/// Refuse a volume that does not share the first's grid (§3.2).  Frame
/// times both state must agree; where only `other` states them, they are
/// returned for the grid to take, and where `other` states the times the
/// grid assumed, the grid's own are returned: they are stated now, so a
/// later volume stating others is a conflict, not a correction.
fn same_grid(
    first: &Geometry,
    other: &Geometry,
    name: &str,
    report: &mut ConversionReport,
) -> Result<Option<(Vec<f64>, Option<String>)>> {
    if first.shape != other.shape {
        return Err(Error::invalid(format!(
            "{} has shape {}, but the first volume has {}; resample before converting",
            repr_str(name),
            repr_int_tuple(&other.shape),
            repr_int_tuple(&first.shape)
        )));
    }
    if let Some(key) = geometry_mismatch(first, other) {
        return Err(Error::invalid(format!(
            "{} disagrees with the first volume on {key}; resample before converting rather than letting a \
             converter do it silently",
            repr_str(name)
        )));
    }
    let (Some(ours), Some(theirs)) = (seconds(first), seconds(other)) else { return Ok(None) };
    if allclose(&ours, &theirs, 1e-6, 1e-5) {
        if other.time_measured && !first.time_measured {
            return Ok(Some((first.time_values.clone().unwrap_or_default(), first.time_units.clone())));
        }
        return Ok(None);
    }
    let times = |g: &Geometry| g.time_values.as_deref().map_or("None".into(), float_list);
    let unit = |g: &Geometry| g.time_units.clone().unwrap_or_else(|| "None".into());
    if first.time_measured && other.time_measured {
        return Err(Error::invalid(format!(
            "{} states frame times {} {}, the first volume {} {}; one grid has one set of `time_values` (§3.2), so \
             convert them separately",
            repr_str(name),
            times(other),
            unit(other),
            times(first),
            unit(first)
        )));
    }
    if !other.time_measured {
        return Ok(None);
    }
    report.decision(
        "time_values",
        format!("{} states its frame times and the first volume does not, so the grid takes them", repr_str(name)),
        json!({"from": name, "time_units": other.time_units}),
    );
    Ok(Some((other.time_values.clone().unwrap_or_default(), other.time_units.clone())))
}

/// Add a volume to the grid the first began: refused where it does not
/// share it, and the grid takes frame times it states.
fn join_grid(first: &mut Geometry, other: &Geometry, name: &str, report: &mut ConversionReport) -> Result<()> {
    if let Some((times, units)) = same_grid(first, other, name, report)? {
        first.time_values = Some(times);
        first.time_units = units;
        first.time_measured = true;
    }
    Ok(())
}

/// Every volume on one grid must agree what its non-spatial axis is.
fn same_axis_kind(per_image: &[(String, Geometry)]) -> Result<()> {
    let mut kinds: Vec<(&str, &str)> = per_image.iter().map(|(n, g)| (n.as_str(), g.leading_kind.as_str())).collect();
    let mut distinct: Vec<&str> = kinds.iter().map(|k| k.1).collect();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() > 1 {
        kinds.sort_unstable();
        let listed: Vec<String> = kinds.iter().map(|(n, k)| format!("{n} is {k}")).collect();
        return Err(Error::invalid(format!(
            "these volumes share a grid but disagree about their non-spatial axis ({}); one grid states one set of \
             `axis_kinds`, so convert them separately or pass `fourth_axis=` to settle it",
            listed.join(", ")
        )));
    }
    Ok(())
}

/// `(axis_names, axis_kinds)` for a grid of a converted array: reordered,
/// `(t, z, y, x)`; as NIfTI keeps it, `(x, y, z)`.  More than one axis past
/// the spatial three is refused rather than guessed at.
pub fn grid_axes(shape: &[usize], transposed: bool, leading_kind: &str) -> Result<(Vec<String>, Vec<String>)> {
    let n = shape.len();
    let extra = n as i64 - 3;
    let strings = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    if !transposed {
        if extra > 0 {
            return Err(Error::invalid(format!(
                "a {n}-D NIfTI keeps its time axis last, and §3.1 requires the spatial axes to be trailing and \
                 contiguous, so this array cannot be declared as a grid without reordering it --- drop \
                 `transpose=False` for 4-D input"
            )));
        }
        return Ok((strings(&["x", "y", "z"][..n]), vec!["spatial".into(); n]));
    }
    if extra <= 0 {
        let names = ["z", "y", "x"];
        // `names[-n:]`, which for no axes is all three.
        let names = if n == 0 { &names[..] } else { &names[3 - n..] };
        return Ok((strings(names), vec!["spatial".into(); n]));
    }
    if extra == 1 {
        let name = if leading_kind == "time" { "t" } else { "c" };
        return Ok((strings(&[name, "z", "y", "x"]), strings(&[leading_kind, "spatial", "spatial", "spatial"])));
    }
    Err(Error::invalid(format!(
        "a {n}-D NIfTI has {extra} axes beyond (x, y, z); only a single trailing time axis converts without a \
         decision about what the others mean --- convert the volumes separately, or declare the axes"
    )))
}

/// The class a mask named `name` is: by the key its name makes, else by the
/// name itself.
fn class_of<'a>(label_set: &'a LabelSet, name: &str) -> Option<&'a LabelClass> {
    label_set
        .get(&ClassKey::Key(sanitize_key(name, "class")))
        .or_else(|| label_set.get(&ClassKey::Key(name.to_string())))
}

/// A display name minted from a class key.
fn key_title(key: &str) -> String {
    title_case(&key.replace('_', " "))
}

/// Mask names, sorted, as classes with minted ids (Appendix B).
fn mint_label_set(names: &[&str], report: &mut ConversionReport) -> Result<LabelSet> {
    let mut keys: Vec<(String, &str)> = Vec::new();
    for name in names {
        let key = sanitize_key(name, "class");
        if let Some((_, earlier)) = keys.iter().find(|(k, _)| *k == key) {
            return Err(Error::invalid(format!(
                "masks {} and {} would both be class {}; a class key keeps lowercase letters, digits and \
                 underscores only, so rename one",
                repr_str(earlier),
                repr_str(name),
                repr_str(&key)
            )));
        }
        keys.push((key, name));
    }
    let mut classes = Vec::new();
    let mut ids = Map::new();
    for (i, (key, name)) in keys.iter().enumerate() {
        let id = i as i64 + 1;
        let shown = if key == name { key_title(key) } else { name.to_string() };
        classes.push(LabelClass::new(id, key.clone(), shown)?);
        ids.insert(key.clone(), json!(id));
    }
    report.decision(
        "label_set",
        "class ids were minted from the mask names in sorted order, and keys from the names as a class key allows \
         them; review the generated label set before applying it cohort-wide",
        json!({ "ids": ids }),
    );
    LabelSet::inline("converted", classes)
}

/// `label_set` and the classes looked for, a class named that the input
/// does not supply recorded as examined and found absent (§11.3): minted
/// after the supplied ids where the import mints its label set (`mint`),
/// refused (E402) where it was given.  Classes come back as their ids.
pub(crate) fn examined_classes(
    annotated: &Annotated,
    label_set: LabelSet,
    mint: bool,
    report: &mut ConversionReport,
    supplied: &str,
) -> Result<(LabelSet, Annotated)> {
    let Annotated::Classes(entries) = annotated else { return Ok((label_set, annotated.clone())) };
    fn found(set: &LabelSet, entry: &ClassKey) -> Option<i64> {
        match entry {
            ClassKey::Key(name) => {
                set.get(entry).or_else(|| set.get(&ClassKey::Key(sanitize_key(name, "class")))).map(|c| c.id)
            }
            ClassKey::Id(_) => set.get(entry).map(|c| c.id),
        }
    }
    let absent: Vec<&ClassKey> = entries.iter().filter(|e| found(&label_set, e).is_none()).collect();
    let mut label_set = label_set;
    if !absent.is_empty() && !mint {
        let named: Vec<String> = absent
            .iter()
            .map(|e| match e {
                ClassKey::Id(i) => i.to_string(),
                ClassKey::Key(k) => k.clone(),
            })
            .collect();
        return Err(Error::coded(
            "E402",
            format!(
                "annotated_classes names {}, which label set {} does not hold and {supplied} do not supply: a class \
                 examined and found absent must be a class of the label set --- add it there, or leave it out of \
                 annotated_classes (--annotated-classes)",
                repr_list(&named),
                repr_str(&label_set.id)
            ),
        ));
    }
    if !absent.is_empty() {
        let numbers: Vec<i64> = absent
            .iter()
            .filter_map(|e| match e {
                ClassKey::Id(i) => Some(*i),
                ClassKey::Key(_) => None,
            })
            .collect();
        if !numbers.is_empty() {
            return Err(Error::coded(
                "E402",
                format!(
                    "annotated_classes names class id(s) {}, which {supplied} do not supply; a class to mint is \
                     named by its key, not its id",
                    repr_int_list(&numbers)
                ),
            ));
        }
        let mut classes = label_set.classes().to_vec();
        let mut minted: Vec<(String, i64)> = Vec::new();
        let mut next = label_set.ids().into_iter().max().unwrap_or(0) + 1;
        for entry in &absent {
            let ClassKey::Key(name) = entry else { continue };
            let key = sanitize_key(name, "class");
            if minted.iter().any(|(k, _)| *k == key) {
                continue;
            }
            let shown = if key == *name { key_title(&key) } else { name.clone() };
            classes.push(LabelClass::new(next, key.clone(), shown)?);
            minted.push((key, next));
            next += 1;
        }
        label_set = LabelSet::new(
            label_set.id.clone(),
            classes,
            label_set.version.clone(),
            Vec::new(),
            Vec::new(),
            "inline",
            None,
            None,
        )?;
        let mut sorted: Vec<&str> = minted.iter().map(|(k, _)| k.as_str()).collect();
        sorted.sort_unstable();
        let detail: Map<String, Value> = minted.iter().map(|(k, id)| (k.clone(), json!(id))).collect();
        report.decision(
            "label_set",
            format!(
                "class(es) {} are named in annotated_classes and {supplied} do not supply them: each was examined \
                 and found absent, so it was minted after the supplied ids, with no voxel or shape of its own",
                repr_list(&sorted)
            ),
            json!({ "examined_absent": detail }),
        );
    }
    let mut ids: Vec<i64> = Vec::new();
    for entry in entries {
        let id = found(&label_set, entry).expect("every class is in the label set now");
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    Ok((label_set, Annotated::Classes(ids.into_iter().map(ClassKey::Id).collect())))
}

/// A sample id from a file name: NFKD-folded, then ASCII identifier
/// characters (§2.2).
fn sanitize_ascii_stem(text: &str, limit: usize) -> String {
    use unicode_normalization::char::canonical_combining_class;
    use unicode_normalization::UnicodeNormalization;
    let folded: String = text.nfkd().filter(|c| canonical_combining_class(*c) == 0).collect();
    sanitize_stem(&folded, limit).chars().map(|c| if c.is_ascii() { c } else { '_' }).collect()
}

/// `{v:g}`.
fn g(value: f64) -> String {
    format_g(value, 6)
}

/// Write one sample from a set of co-registered NIfTI volumes, `images` by
/// name, into `out`.
///
/// Every volume and mask must share one grid: a disagreement is refused,
/// never resampled.  The masks' classes come from `label_set`, or are
/// minted from their names in sorted order.  What was decided and guessed
/// is in `report`, whose `source` this sets.
pub fn from_nifti(
    images: &[(String, PathBuf)],
    out: &Path,
    options: &ImportOptions,
    report: &mut ConversionReport,
) -> Result<()> {
    let given = options.label_set.clone();
    report.source = images.iter().map(|(_, p)| shown(p)).collect::<Vec<_>>().join(", ");
    if images.is_empty() {
        return Err(Error::invalid("from_nifti needs at least one image"));
    }
    let read = options.read();
    let mut arrays: Vec<(String, Voxels)> = Vec::new();
    let mut per_image: Vec<(String, Geometry)> = Vec::new();
    let mut geometry: Option<Geometry> = None;
    let mut reshaped = false;
    for (name, path) in images {
        let NiftiRead { voxels, geometry: geo } = read_nifti(path, &read)?;
        replay_notes(report, name, &geo);
        reshaped = note_plane(report, name, &geo) || reshaped;
        match &mut geometry {
            None => geometry = Some(geo.clone()),
            Some(first) => join_grid(first, &geo, name, report)?,
        }
        per_image.push((name.clone(), geo));
        arrays.push((name.clone(), voxels));
    }
    let geometry = geometry.expect("at least one image");
    same_axis_kind(&per_image)?;

    let mask_read = ReadOptions { fourth_axis: "auto".into(), ..read.clone() };
    let mut masks: Vec<(String, ArrayD<bool>)> = Vec::new();
    for (name, path) in &options.masks {
        let NiftiRead { voxels, geometry: geo } = read_nifti(path, &mask_read)?;
        replay_notes(report, name, &geo);
        reshaped = note_plane(report, name, &geo) || reshaped;
        same_grid(&geometry, &geo, name, report)?;
        masks.push((name.clone(), voxels.nonzero(geo.rescale)?));
    }

    let coord_system = &options.coord_system;
    if coord_system != "RAS" {
        report.decision(
            "coord_system",
            format!(
                "NIfTI is RAS+; the grid was written in {coord_system} (sign flip on the first two world axes, \
                 applied to the affine only)"
            ),
            json!({"source": "RAS", "target": coord_system}),
        );
    }
    if !geometry.squeezed.is_empty() {
        report.decision(
            "axis_order",
            "trailing axes of extent 1 beyond (x, y, z) carry nothing and were dropped, so the grid describes the \
             data rather than the file",
            json!({ "axes": geometry.squeezed }),
        );
    }
    let leading = geometry.shape.first().copied().unwrap_or(0);
    match (geometry.leading_kind.as_str(), geometry.leading_stated) {
        ("channel", false) => {
            report.guess(
                "axis_kinds",
                "the series is on NIfTI's fifth dimension, which NIfTI-1 gives a voxel's components, and nothing \
                 says what they are; it was read as a `channel` axis (§3.6) --- pass `fourth_axis=` to say otherwise",
                json!({ "channels": leading }),
            );
        }
        ("channel", true) => {
            let b_values = geometry.b_values.as_deref().map_or(Value::Null, |v| v.iter().map(|x| num(*x)).collect());
            report.decision(
                "axis_kinds",
                "the fourth axis holds components rather than frames, so it is a `channel` axis (§3.6) --- \
                 b-values, a sidecar field or an intent code said so",
                json!({"channels": leading, "b_values": b_values, "stated_by": geometry.stated_by}),
            );
        }
        ("time", false) => {
            report.guess(
                "axis_kinds",
                "the source does not say whether the fourth axis is time or channel; it was read as time (§3.6) \
                 --- pass `fourth_axis=` to say otherwise",
                json!({ "frames": leading }),
            );
        }
        _ => {}
    }
    if let Some(times) = &geometry.time_values {
        let detail = json!({"time_units": geometry.time_units, "frames": times.len()});
        if geometry.time_measured {
            report.decision("time_values", "frame times were read from the NIfTI temporal zoom and toffset", detail);
        } else {
            report.guess(
                "time_values",
                "the NIfTI declares no temporal zoom, so frame indices were used as times; §3.2 requires \
                 `time_values` where there is a time axis",
                detail,
            );
        }
    }
    if options.transpose && geometry.shape.len() >= 3 {
        report.decision(
            "axis_order",
            "NIfTI (x, y, z) was reordered to (z, y, x), any trailing time axis moved in front of them, and the \
             spacing, direction and axis names follow",
            json!({ "order": geometry.axis_order }),
        );
    }

    // A sample id is ASCII (§2.2); the subject keeps the name whole.
    let file_stem = out.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let sample_id = options.sample_id.as_deref().filter(|s| !s.is_empty());
    let stem = sample_id.map_or_else(|| sanitize_ascii_stem(&file_stem, 255), str::to_string);
    let subject =
        options.subject_id.as_deref().filter(|s| !s.is_empty()).or(sample_id).unwrap_or(&file_stem).to_string();
    if stem != sample_id.unwrap_or(&file_stem) {
        report.decision(
            "identity",
            format!(
                "the file name {} is no sample id, which is ASCII letters, digits and '_.-' (§2.2), so the sample id \
                 is {}",
                repr_str(&file_stem),
                repr_str(&stem)
            ),
            json!({"sample_id": stem, "name": file_stem}),
        );
    }
    if options.subject_id.is_none() {
        report.guess(
            "identity",
            format!(
                "no subject_id was given and NIfTI states none, so the sample was written as a subject of its own \
                 (subject_id = {}); pass subject_id= (--subject-id) to file it under its subject",
                repr_str(&subject)
            ),
            json!({ "subject_id": subject }),
        );
    }

    let mut vocabulary = given.clone();
    let mut examined = options.annotated_classes.clone();
    if !masks.is_empty() {
        let set = match &given {
            None => {
                let mut names: Vec<&str> = masks.iter().map(|(n, _)| n.as_str()).collect();
                names.sort_unstable();
                mint_label_set(&names, report)?
            }
            Some(given) => {
                let mut unmatched: Vec<&str> =
                    masks.iter().map(|(n, _)| n.as_str()).filter(|n| class_of(given, n).is_none()).collect();
                if !unmatched.is_empty() {
                    unmatched.sort_unstable();
                    return Err(Error::coded(
                        "E402",
                        format!(
                            "label set {} has no class for mask(s) {}: a mask is the class of the key its name \
                             makes, or of its name --- add the class, or rename the mask",
                            repr_str(&given.id),
                            repr_list(&unmatched)
                        ),
                    ));
                }
                let mut names: Vec<&str> = masks.iter().map(|(n, _)| n.as_str()).collect();
                names.sort_unstable();
                let ids: Map<String, Value> =
                    names.iter().map(|n| (n.to_string(), json!(class_of(given, n).expect("matched").id))).collect();
                report.decision(
                    "label_set",
                    format!(
                        "the masks were matched to label set {} by the keys their names make; no class was minted",
                        repr_str(&given.id)
                    ),
                    json!({"label_set": given.id, "ids": ids}),
                );
                given.clone()
            }
        };
        let (set, classes) = examined_classes(&options.annotated_classes, set, given.is_none(), report, "the masks")?;
        vocabulary = Some(set);
        examined = classes;
    }

    let mut params = Map::new();
    params.insert("coord_system".into(), json!(coord_system));
    params.insert("transpose".into(), json!(options.transpose));
    if reshaped {
        params.insert("plane_as_volume".into(), json!(true));
    }
    let mut writer = create(out, Some(&stem), Some(&subject), &options.codec, &[])?;
    let written = write_sample(
        &mut writer,
        images,
        options,
        &geometry,
        &per_image,
        arrays,
        masks,
        vocabulary,
        examined,
        params,
        report,
    );
    match written.and_then(|()| writer.commit(true).map(drop)) {
        Ok(()) => {
            report.outputs.push(shown(out));
            Ok(())
        }
        Err(err) => {
            writer.abort();
            Err(err)
        }
    }
}

/// Everything [`from_nifti`] writes, into an open writer.
#[allow(clippy::too_many_arguments)]
fn write_sample(
    writer: &mut SampleWriter,
    images: &[(String, PathBuf)],
    options: &ImportOptions,
    geometry: &Geometry,
    per_image: &[(String, Geometry)],
    arrays: Vec<(String, Voxels)>,
    masks: Vec<(String, ArrayD<bool>)>,
    vocabulary: Option<LabelSet>,
    examined: Annotated,
    params: Map<String, Value>,
    report: &mut ConversionReport,
) -> Result<()> {
    let tool = writer.software("medh5", Some(crate::VERSION), Map::new())?;
    let mut fields = Map::new();
    fields.insert("tool".into(), json!("medh5 convert from-nifti"));
    // The files' names, not their paths: a local path often holds a
    // patient's name or MRN.
    let inputs: Vec<String> = images
        .iter()
        .map(|(_, p)| format!("nifti:{}", p.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()))
        .collect();
    fields.insert("inputs".into(), json!(inputs));
    fields.insert("params".into(), Value::Object(params));
    let activity = writer.activity("import", Some(&tool.id), None, fields)?;
    if let Some(set) = &vocabulary {
        writer.label_set(set.clone());
    }
    // Whether the axes moved, as the read recorded it: arrays below 3-D are
    // never reordered, whatever was asked.
    let order = &geometry.axis_order;
    let transposed = !order.iter().copied().eq(0..order.len());
    let (axis_names, axis_kinds) = grid_axes(&geometry.shape, transposed, &geometry.leading_kind)?;
    let shape: Vec<i64> = geometry.shape.iter().map(|v| *v as i64).collect();
    let n = geometry.direction.len();
    let direction = Array2::from_shape_fn((n, n), |(r, c)| geometry.direction[r][c]);
    writer.add_grid(
        "ref",
        &shape,
        &geometry.spacing,
        GridOptions {
            origin: Some(geometry.origin.clone()),
            direction: Some(direction),
            axis_names: Some(axis_names),
            axis_kinds: Some(axis_kinds),
            coord_system: Some(geometry.coord_system.clone()),
            units: Some(geometry.units.clone()),
            timepoint: Some("tp0".into()),
            time_values: geometry.time_values.clone(),
            time_units: geometry.time_units.clone(),
            ..GridOptions::default()
        },
    )?;
    let lookup = |pairs: &[(String, String)], name: &str| pairs.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone());
    let frames = geometry.shape.first().copied().unwrap_or(0);
    for ((name, voxels), (_, own)) in arrays.into_iter().zip(per_image) {
        let b_values = own.b_values.as_deref().filter(|v| !v.is_empty());
        let channel = match (&own.channel_field, own.channel_values.as_deref()) {
            (Some(field), Some(values)) if !field.is_empty() && !values.is_empty() => {
                PER_VOLUME_CHANNEL.iter().find(|f| f.0 == field).map(|f| (f, values))
            }
            _ => None,
        };
        let mut channel_names = None;
        if let Some(b) = b_values.filter(|b| b.len() == frames) {
            channel_names = Some(b.iter().map(|v| format!("b={}", g(*v))).collect::<Vec<_>>());
        } else if let Some((field, values)) = channel.filter(|(_, v)| v.len() == frames) {
            channel_names = Some(values.iter().map(|v| format!("{}={}", field.2, g(*v))).collect());
        }
        if let Some(b) = b_values {
            let mut params = Map::new();
            params.insert("b_values".into(), b.iter().map(|v| num(*v)).collect());
            writer.acquisition(&name, params);
        }
        if let Some((field, values)) = channel {
            let mut params = Map::new();
            params.insert(field.1.into(), values.iter().map(|v| num(*v)).collect());
            writer.acquisition(&name, params);
        }
        if let Some((slope, intercept)) = own.rescale {
            report.decision(
                "value_scale",
                format!(
                    "{name}: scl_slope/scl_inter were stored as the image's rescale, not applied; read(physical=True) \
                     applies them and read() returns what the file stores (§4.2)"
                ),
                json!({"image": name, "slope": num(slope), "intercept": num(intercept)}),
            );
        }
        let value_units = lookup(&options.value_units, &name).filter(|u| !u.is_empty());
        let modality = lookup(&options.modalities, &name).unwrap_or_else(|| "OT".into());
        let array = voxels.into_ndarray()?;
        writer.add_image(
            &name,
            &array,
            "ref",
            &modality,
            ImageOptions {
                value_type: Some(if value_units.is_some() { "quantitative" } else { "intensity" }.into()),
                value_units: lookup(&options.value_units, &name),
                channel_names,
                rescale_slope: own.rescale.map(|r| r.0),
                rescale_intercept: own.rescale.map(|r| r.1),
                prov: Some(activity.id.clone()),
                ..ImageOptions::default()
            },
        )?;
    }
    if let (false, Some(set)) = (masks.is_empty(), &vocabulary) {
        let mut names: Vec<String> = masks.iter().map(|(n, _)| n.clone()).collect();
        let mut resolved: Vec<(i64, ArrayD<bool>)> = Vec::new();
        for (name, mask) in masks {
            let id = class_of(set, &name).expect("every mask has a class").id;
            match resolved.iter_mut().find(|(k, _)| *k == id) {
                Some(slot) => slot.1 = mask,
                None => resolved.push((id, mask)),
            }
        }
        let count = resolved.len();
        let source = SegmentationSource::Masks(resolved.into_iter().map(|(id, m)| (ClassKey::Id(id), m)).collect());
        let (kind, stats) = writer.add_segmentation(
            "seg",
            "ref",
            source,
            SegmentationOptions {
                encoding: Some("auto".into()),
                common: AnnotationOptions {
                    annotated_classes: examined,
                    closure: Some("explicit".into()),
                    prov: Some(activity.id.clone()),
                    task: Some("segmentation".into()),
                    ..AnnotationOptions::default()
                },
                ..SegmentationOptions::default()
            },
        )?;
        report.decision(
            "encoding",
            format!("{count} mask(s) were measured and stored as {}", repr_str(&kind)),
            json!({"kind": kind, "overlapping": stats.map(|s| s.edges.len())}),
        );
        if options.annotated_classes == Annotated::AllGiven {
            names.sort_unstable();
            report.guess(
                "coverage",
                "annotated_class_ids was set to the masks supplied; NIfTI cannot say which classes were searched \
                 for and not found --- pass annotated_classes= (--annotated-classes) to say",
                json!({ "classes": names }),
            );
        }
    }
    Ok(())
}

/// One NIfTI file as it is stored, for a caller with conventions of its own
/// (ITK displacement fields): the voxels in NIfTI's order, first axis
/// fastest, in the shape the header gives; the affine a reader takes from
/// the header (RAS, in the file's unit); the spatial unit; and the scaling
/// the voxels are read through.  The refusals are [`read_nifti`]'s.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct StoredNifti {
    pub header: Header,
    pub voxels: Voxels,
    pub affine: [[f64; 4]; 4],
    /// The spatial unit in the §3.5 vocabulary (millimetres where the
    /// header states none of it).
    pub units: String,
    /// What the header said, where that is no unit of the vocabulary.
    pub units_stated: Option<String>,
    /// `scl_slope` and `scl_inter`, where they scale.
    pub rescale: Option<(f64, f64)>,
}

/// Read one NIfTI file as it is stored (see [`StoredNifti`]).
pub fn read_stored(path: &Path) -> Result<StoredNifti> {
    let image = load(path)?;
    let affine = best_affine(&image.header, &shown(path))?;
    let voxels = image.read_voxels()?;
    let (units, units_stated) = units(&image.header);
    let rescale = rescale(&image);
    Ok(StoredNifti { header: image.header, voxels, affine, units, units_stated, rescale })
}

/// Add NIfTI masks to an existing sample, checking they sit on its grid.
///
/// A mask must match the grid in shape *and* geometry (spacing, origin and
/// direction, lengths compared in millimetres); a mismatch is refused unless
/// `assume_aligned`, which takes the voxels as the grid's and records that
/// as a guess.  The comparison is in the grid's coordinate system, which
/// `coord_system` may only repeat.  A class `annotated_classes` names that no
/// mask supplies was examined and found absent (§11.3).
pub fn import_seg_nifti(
    path: &Path,
    masks: &[(String, PathBuf)],
    options: &SegOptions,
    report: &mut ConversionReport,
) -> Result<()> {
    report.source = masks.iter().map(|(_, p)| shown(p)).collect::<Vec<_>>().join(", ");
    let (grid_id, target, existing) = {
        let sample = crate::sample::reader::open_sample(path)?;
        let grid_id = match options.grid.as_deref().filter(|g| !g.is_empty()) {
            Some(g) => g.to_string(),
            None => sample.reference_grid()?.grid_id,
        };
        let target = sample.grids()?.get(&grid_id).cloned().ok_or_else(|| Error::Key(grid_id.clone()))?;
        (grid_id, target, sample.label_set()?.cloned())
    };
    let system = target.coord_system.clone();
    if let Some(cs) = options.coord_system.as_deref().filter(|cs| *cs != system) {
        return Err(Error::invalid(format!(
            "grid {} is in {system}, and masks are compared with it there; coord_system={} cannot apply",
            repr_str(&grid_id),
            repr_str(cs)
        )));
    }
    let read = ReadOptions {
        coord_system: system.clone(),
        transpose: options.transpose,
        assume_geometry: options.assume_geometry,
        plane_as_volume: options.plane_as_volume,
        ..ReadOptions::default()
    };
    let spatial = target.spatial_shape();
    let mut arrays: Vec<(String, ArrayD<bool>)> = Vec::new();
    let mut reshaped = false;
    for (name, mask_path) in masks {
        let NiftiRead { voxels, geometry: geo } = read_nifti(mask_path, &read)?;
        replay_notes(report, name, &geo);
        reshaped = note_plane(report, name, &geo) || reshaped;
        if geo.shape != spatial {
            return Err(Error::coded(
                "E405",
                format!(
                    "mask {} has shape {}, but grid {} is {}",
                    repr_str(name),
                    repr_int_tuple(&geo.shape),
                    repr_str(&grid_id),
                    repr_int_tuple(&spatial)
                ),
            ));
        }
        let mut reference = geo.clone();
        reference.spacing = target.spacing.clone();
        reference.origin = target.origin.clone();
        reference.direction = target.direction.rows().into_iter().map(|r| r.to_vec()).collect();
        reference.units = target.units.clone();
        if let Some(key) = geometry_mismatch(&reference, &geo) {
            if !options.assume_aligned {
                return Err(Error::invalid(format!(
                    "mask {} disagrees with grid {} on {key}: it was drawn on another grid, and taking its voxels \
                     as this one's would put the labels where nobody drew them --- resample it onto the grid first, \
                     or pass assume_aligned=True if its header is what is wrong",
                    repr_str(name),
                    repr_str(&grid_id)
                )));
            }
            report.guess(
                "geometry",
                format!(
                    "mask {} disagrees with grid {} on {key}; its voxels were taken as the grid's because \
                     assume_aligned=True",
                    repr_str(name),
                    repr_str(&grid_id)
                ),
                json!({
                    "mask": name,
                    "field": key,
                    "mask_value": geo.to_json()[key],
                    "grid_value": reference.to_json()[key],
                }),
            );
        }
        let mask = voxels.nonzero(geo.rescale)?;
        // A name given twice is one mask, the later, where the first stood.
        match arrays.iter_mut().find(|(n, _)| n == name) {
            Some(slot) => slot.1 = mask,
            None => arrays.push((name.clone(), mask)),
        }
    }

    let label_set = match &existing {
        Some(set) => set.clone(),
        None => {
            let mut names: Vec<&str> = arrays.iter().map(|(n, _)| n.as_str()).collect();
            names.sort_unstable();
            mint_label_set(&names, report)?
        }
    };
    let missing: Vec<&str> =
        arrays.iter().map(|(n, _)| n.as_str()).filter(|n| class_of(&label_set, n).is_none()).collect();
    if !missing.is_empty() {
        return Err(Error::coded(
            "E402",
            format!(
                "the sample's label set has no class(es) {}; add them or convert without an existing label set",
                repr_list(&missing)
            ),
        ));
    }
    let (label_set, examined) =
        examined_classes(&options.annotated_classes, label_set, existing.is_none(), report, "the masks")?;
    let mut resolved: Vec<(i64, ArrayD<bool>)> = Vec::new();
    for (name, mask) in arrays {
        let id = class_of(&label_set, &name).expect("every mask has a class").id;
        match resolved.iter_mut().find(|(k, _)| *k == id) {
            Some(slot) => slot.1 = mask,
            None => resolved.push((id, mask)),
        }
    }

    let mut writer = crate::sample::writer::amend(path, None)?;
    let written = (|| -> Result<String> {
        if existing.is_none() {
            writer.label_set(label_set);
        }
        let tool = writer.software("medh5", Some(crate::VERSION), Map::new())?;
        let mut fields = Map::new();
        fields.insert("tool".into(), json!("medh5.io.nifti.import_seg_nifti"));
        // Params only when a plane was reshaped: an import that reshaped
        // none records the activity it always did.
        if reshaped {
            fields.insert("params".into(), json!({"plane_as_volume": true}));
        }
        let activity = writer.activity("import", Some(&tool.id), None, fields)?;
        let source = SegmentationSource::Masks(resolved.into_iter().map(|(id, m)| (ClassKey::Id(id), m)).collect());
        let (kind, _) = writer.add_segmentation(
            &options.ann_id,
            &grid_id,
            source,
            SegmentationOptions {
                encoding: Some("auto".into()),
                common: AnnotationOptions {
                    annotated_classes: examined,
                    closure: Some("explicit".into()),
                    prov: Some(activity.id.clone()),
                    task: Some("segmentation".into()),
                    ..AnnotationOptions::default()
                },
                ..SegmentationOptions::default()
            },
        )?;
        Ok(kind)
    })();
    match written {
        Ok(kind) => {
            report.decision("encoding", format!("masks were stored as {}", repr_str(&kind)), json!({"kind": kind}));
            if let Err(err) = writer.commit(true) {
                writer.abort();
                return Err(err);
            }
            report.outputs.push(shown(path));
            Ok(())
        }
        Err(err) => {
            writer.abort();
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::header::{Version, VoxOffset};
    use super::*;

    #[test]
    fn s3_1_reordering_fortran_voxels_gives_c_order() {
        // (x, y) = (3, 2), Fortran order: x fastest.
        let bytes: Vec<u8> = (0..6).collect();
        // Kept as (x, y): row-major, so y fastest.
        assert_eq!(c_order(bytes.clone(), 1, &[3, 2], &[0, 1]), [0, 3, 1, 4, 2, 5]);
        // Reversed to (y, x): the bytes as they are.
        assert_eq!(c_order(bytes.clone(), 1, &[3, 2], &[1, 0]), bytes);
        // (x, y, z, t) to (t, z, y, x), two-byte items, a 1 among them.
        let shape = [2, 1, 3, 2];
        let bytes: Vec<u8> = (0..24).collect();
        assert_eq!(c_order(bytes.clone(), 2, &shape, &[3, 2, 1, 0]), bytes);
        let kept = c_order(bytes, 2, &shape, &[0, 1, 2, 3]);
        assert_eq!(&kept[..8], &[0, 1, 12, 13, 4, 5, 16, 17]);
    }

    #[test]
    fn s02_2_a_scalar_reads_as_one_element_on_no_axes() {
        // dim[0] = -1, big-endian: a shape of no dimensions.  The geometry
        // keeps `()`, the array is `(1,)`, as 2.1.0 returned them.
        let mut h = Header::new(Version::One);
        (h.dim[0], h.dim[1], h.datatype, h.bitpix, h.big_endian) = (-1, 4, 4, 16, true);
        h.vox_offset = VoxOffset::Float(352.0);
        let mut bytes = h.encode();
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 7, 0, 9]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scalar.nii");
        std::fs::write(&path, bytes).unwrap();
        let read = read_nifti(&path, &ReadOptions::default().with_assume_geometry(true)).unwrap();
        assert_eq!((read.geometry.shape, read.geometry.axis_order), (Vec::new(), Vec::new()));
        assert_eq!((read.voxels.shape, read.voxels.bytes), (vec![1], vec![0, 7]));
    }

    #[test]
    fn s4_5_sidecars_are_found_beside_their_image() {
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("dwi.nii.gz");
        assert_eq!(sidecar(&image, ".bval"), None);
        std::fs::write(dir.path().join("dwi.bval"), "0 1000\n").unwrap();
        assert_eq!(sidecar(&image, ".bval"), Some(dir.path().join("dwi.bval")));
        assert_eq!(sidecar(&dir.path().join("dwi.mha"), ".bval"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn s4_5_a_name_in_another_encoding_finds_its_own_sidecars() {
        // "caf\xe9": Latin-1, no UTF-8. Its sidecar is "caf\xe9.bval", not
        // the lossy "caf\u{fffd}.bval".
        use std::os::unix::ffi::OsStringExt;
        let dir = tempfile::tempdir().unwrap();
        let named =
            |suffix: &str| dir.path().join(OsString::from_vec([b"caf\xe9".as_slice(), suffix.as_bytes()].concat()));
        std::fs::write(named(".bval"), "0 1000\n").unwrap();
        std::fs::write(named(".json"), "{}").unwrap();
        assert_eq!(sidecar(&named(".nii.gz"), ".bval"), Some(named(".bval")));
        assert_eq!(sidecar(&named(".nii"), ".json"), Some(named(".json")));
    }

    fn series(times: &[f64], measured: bool) -> Geometry {
        Geometry {
            notes: Vec::new(),
            spacing: vec![1.0; 3],
            origin: vec![0.0; 3],
            direction: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
            coord_system: "LPS".into(),
            shape: vec![times.len(), 2, 2, 2],
            axis_order: vec![3, 2, 1, 0],
            time_values: Some(times.to_vec()),
            time_units: Some("s".into()),
            time_measured: measured,
            leading_kind: "time".into(),
            leading_stated: measured,
            b_values: None,
            channel_field: None,
            channel_values: None,
            stated_by: None,
            squeezed: Vec::new(),
            units: "mm".into(),
            dtype: "int16".into(),
            rescale: None,
            plane_as_volume: None,
            descrip: String::new(),
        }
    }

    #[test]
    fn s3_2_times_a_volume_states_stay_stated_when_the_grid_assumed_them() {
        // A assumes 0, 1, 2; B states 0, 1, 2; C states 0, 2, 4. B and C
        // disagree, so C is refused, not taken over B's stated times.
        let mut report = ConversionReport::new("nifti", "a.nii.gz");
        let mut grid = series(&[0.0, 1.0, 2.0], false);
        join_grid(&mut grid, &series(&[0.0, 1.0, 2.0], true), "B", &mut report).unwrap();
        assert!(grid.time_measured);
        assert_eq!(grid.time_values.as_deref(), Some(&[0.0, 1.0, 2.0][..]));
        let err = join_grid(&mut grid, &series(&[0.0, 2.0, 4.0], true), "C", &mut report).unwrap_err();
        assert!(err.message().contains("'C' states frame times"), "{err:?}");
        // Assumed times a later volume states otherwise are still taken.
        let mut grid = series(&[0.0, 1.0, 2.0], false);
        join_grid(&mut grid, &series(&[5.0, 7.0, 9.0], true), "B", &mut report).unwrap();
        assert_eq!(grid.time_values.as_deref(), Some(&[5.0, 7.0, 9.0][..]));
    }

    #[test]
    fn s2_2_a_sample_id_is_folded_to_ascii() {
        assert_eq!(sanitize_ascii_stem("Müller Hans", 255), "Muller_Hans");
        // Beyond Latin Extended-A, NFKD still finds the base letter.
        assert_eq!(sanitize_ascii_stem("ḿri ǹ", 255), "mri_n");
        assert_eq!(sanitize_ascii_stem("患者", 255), "__");
        assert_eq!(sanitize_ascii_stem("case_01", 255), "case_01");
    }

    #[test]
    fn s3_1_grid_axes_follow_the_order_the_read_left() {
        let axes = |shape: &[usize], transposed, kind| grid_axes(shape, transposed, kind).unwrap();
        assert_eq!(axes(&[2, 3, 4], true, "spatial").0, ["z", "y", "x"]);
        assert_eq!(axes(&[2, 3], false, "spatial").0, ["x", "y"]);
        assert_eq!(
            axes(&[5, 2, 3, 4], true, "channel"),
            (
                vec!["c".to_string(), "z".into(), "y".into(), "x".into()],
                vec!["channel".to_string(), "spatial".into(), "spatial".into(), "spatial".into()]
            )
        );
        assert!(grid_axes(&[5, 2, 3, 4], false, "time").unwrap_err().message().contains("drop `transpose=False`"));
        assert!(grid_axes(&[5, 5, 2, 3, 4], true, "time").unwrap_err().message().contains("2 axes beyond"));
    }
}
