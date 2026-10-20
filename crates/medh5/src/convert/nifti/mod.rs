//! NIfTI import (spec §3, §4, §7), in the engine.
//!
//! The one thing this converter must not get wrong is the **handedness**.
//! NIfTI's affine is RAS+; MEDH5 grids declare their own `coord_system` and
//! default to LPS, the DICOM convention.  Converting between them is a sign
//! flip on the first two world axes --- applied to the affine, never to the
//! voxels --- and the choice is made explicitly and recorded.  Axis order is
//! likewise stated rather than assumed: NIfTI arrays are `(x, y, z)`
//! fastest-first, MEDH5's spatial axes trail as `(z, y, x)`, and the axis
//! names are written onto the grid.
//!
//! This is the converter 2.1.0 shipped in Python over nibabel, ported rule
//! for rule: its contract is parity with what that converter made of every
//! input in `tests/data/nifti-parity` (2.2.0 decision E3) --- the same
//! objects with the same digests, the same report but for paths, the same
//! refusals with the same words.  [`header`] is the NIfTI-1/NIfTI-2 codec;
//! reading is in `read` (gzip inflated in pure Rust, never past what the
//! header implies); [`read_nifti`] and [`from_nifti`] are the import.

pub mod header;
mod import;
mod qform;
mod read;
mod write;

use ndarray::Array2;
use serde_json::{json, Value};

pub use header::{Element, Header, Version, VoxOffset};
pub use import::{from_nifti, grid_axes, import_seg_nifti, read_nifti, read_stored, StoredNifti};
pub use read::Voxels;
pub use write::{
    embed_plane, for_export, nifti_target, sidecar_path, to_nifti, write_field, write_nifti, ExportOptions,
    WriteOptions, NIFTI_SUFFIXES,
};

use crate::array::NdArray;
use crate::json::{num, repr_str};
use crate::labels::LabelSet;
use crate::sample::writer::Annotated;
use crate::{Error, Result};

/// The coordinate systems a world affine may be stated in.
pub const COORD_SYSTEMS: [&str; 2] = crate::geometry::grid::COORD_SYSTEMS;

/// What a 4-D series' fourth axis may be declared as.
pub const FOURTH_AXES: [&str; 3] = ["auto", "time", "channel"];

/// How [`read_nifti`] reads a file.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ReadOptions {
    /// The world frame the geometry is returned in: `LPS` (converted) or
    /// `RAS` (NIfTI's own).
    pub coord_system: String,
    /// Reorder `(x, y, z)` to `(z, y, x)`, any further axis leading.
    pub transpose: bool,
    /// What a 4-D series' extra axis is: `auto`, `time` or `channel`.
    pub fourth_axis: String,
    /// Accept a file that declares no spatial mapping, taking the fallback
    /// grid from `pixdim` as a guess.
    pub assume_geometry: bool,
    /// Read a 2-D plane as a one-slice volume `(x, y, 1)`.
    pub plane_as_volume: bool,
}

impl Default for ReadOptions {
    fn default() -> Self {
        ReadOptions {
            coord_system: "LPS".into(),
            transpose: true,
            fourth_axis: "auto".into(),
            assume_geometry: false,
            plane_as_volume: false,
        }
    }
}

impl ReadOptions {
    pub fn with_coord_system(mut self, value: impl Into<String>) -> Self {
        self.coord_system = value.into();
        self
    }

    pub fn with_transpose(mut self, value: bool) -> Self {
        self.transpose = value;
        self
    }

    pub fn with_fourth_axis(mut self, value: impl Into<String>) -> Self {
        self.fourth_axis = value.into();
        self
    }

    pub fn with_assume_geometry(mut self, value: bool) -> Self {
        self.assume_geometry = value;
        self
    }

    pub fn with_plane_as_volume(mut self, value: bool) -> Self {
        self.plane_as_volume = value;
        self
    }
}

/// How [`from_nifti`] writes a sample.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ImportOptions {
    /// Masks, by name, on the images' grid.
    pub masks: Vec<(String, std::path::PathBuf)>,
    /// The sample id; by default the output file's stem, made ASCII.
    pub sample_id: Option<String>,
    /// The subject id; by default the sample's.
    pub subject_id: Option<String>,
    /// Each image's modality, by name (default `OT`).
    pub modalities: Vec<(String, String)>,
    pub coord_system: String,
    pub transpose: bool,
    pub fourth_axis: String,
    /// Each image's value units, by name: a quantitative image.
    pub value_units: Vec<(String, String)>,
    /// The codec profile.
    pub codec: String,
    /// Which classes were looked for (§11.3).
    pub annotated_classes: Annotated,
    pub assume_geometry: bool,
    pub plane_as_volume: bool,
    /// The label set the masks' classes come from; minted from their names
    /// without one.
    pub label_set: Option<LabelSet>,
}

impl Default for ImportOptions {
    fn default() -> Self {
        let read = ReadOptions::default();
        ImportOptions {
            masks: Vec::new(),
            sample_id: None,
            subject_id: None,
            modalities: Vec::new(),
            coord_system: read.coord_system,
            transpose: read.transpose,
            fourth_axis: read.fourth_axis,
            value_units: Vec::new(),
            codec: "balanced".into(),
            annotated_classes: Annotated::AllGiven,
            assume_geometry: false,
            plane_as_volume: false,
            label_set: None,
        }
    }
}

impl ImportOptions {
    pub fn with_masks(mut self, masks: Vec<(String, std::path::PathBuf)>) -> Self {
        self.masks = masks;
        self
    }

    pub fn with_sample_id(mut self, value: impl Into<String>) -> Self {
        self.sample_id = Some(value.into());
        self
    }

    pub fn with_subject_id(mut self, value: impl Into<String>) -> Self {
        self.subject_id = Some(value.into());
        self
    }

    pub fn with_modalities(mut self, value: Vec<(String, String)>) -> Self {
        self.modalities = value;
        self
    }

    pub fn with_coord_system(mut self, value: impl Into<String>) -> Self {
        self.coord_system = value.into();
        self
    }

    pub fn with_transpose(mut self, value: bool) -> Self {
        self.transpose = value;
        self
    }

    pub fn with_fourth_axis(mut self, value: impl Into<String>) -> Self {
        self.fourth_axis = value.into();
        self
    }

    pub fn with_value_units(mut self, value: Vec<(String, String)>) -> Self {
        self.value_units = value;
        self
    }

    pub fn with_codec(mut self, value: impl Into<String>) -> Self {
        self.codec = value.into();
        self
    }

    pub fn with_annotated_classes(mut self, value: Annotated) -> Self {
        self.annotated_classes = value;
        self
    }

    pub fn with_assume_geometry(mut self, value: bool) -> Self {
        self.assume_geometry = value;
        self
    }

    pub fn with_plane_as_volume(mut self, value: bool) -> Self {
        self.plane_as_volume = value;
        self
    }

    pub fn with_label_set(mut self, value: LabelSet) -> Self {
        self.label_set = Some(value);
        self
    }

    fn read(&self) -> ReadOptions {
        ReadOptions {
            coord_system: self.coord_system.clone(),
            transpose: self.transpose,
            fourth_axis: self.fourth_axis.clone(),
            assume_geometry: self.assume_geometry,
            plane_as_volume: self.plane_as_volume,
        }
    }
}

/// How [`import_seg_nifti`] adds masks to a sample.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SegOptions {
    /// The annotation's id.
    pub ann_id: String,
    /// The grid the masks sit on; by default the reference grid.
    pub grid: Option<String>,
    /// May only repeat the grid's coordinate system.
    pub coord_system: Option<String>,
    pub transpose: bool,
    /// Which classes were looked for (§11.3).
    pub annotated_classes: Annotated,
    /// Take a mask's voxels as the grid's when their geometries disagree,
    /// recording that as a guess: for a header known to be wrong.
    pub assume_aligned: bool,
    pub assume_geometry: bool,
    pub plane_as_volume: bool,
}

impl Default for SegOptions {
    fn default() -> Self {
        SegOptions {
            ann_id: "seg".into(),
            grid: None,
            coord_system: None,
            transpose: true,
            annotated_classes: Annotated::AllGiven,
            assume_aligned: false,
            assume_geometry: false,
            plane_as_volume: false,
        }
    }
}

impl SegOptions {
    pub fn with_ann_id(mut self, value: impl Into<String>) -> Self {
        self.ann_id = value.into();
        self
    }

    pub fn with_grid(mut self, value: impl Into<String>) -> Self {
        self.grid = Some(value.into());
        self
    }

    pub fn with_coord_system(mut self, value: impl Into<String>) -> Self {
        self.coord_system = Some(value.into());
        self
    }

    pub fn with_transpose(mut self, value: bool) -> Self {
        self.transpose = value;
        self
    }

    pub fn with_annotated_classes(mut self, value: Annotated) -> Self {
        self.annotated_classes = value;
        self
    }

    pub fn with_assume_aligned(mut self, value: bool) -> Self {
        self.assume_aligned = value;
        self
    }

    pub fn with_assume_geometry(mut self, value: bool) -> Self {
        self.assume_geometry = value;
        self
    }

    pub fn with_plane_as_volume(mut self, value: bool) -> Self {
        self.plane_as_volume = value;
        self
    }
}

/// What the reader had to decide from an ambiguous header.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryNote {
    pub kind: String,
    pub message: String,
    pub detail: Value,
}

/// Where a 2-D plane read as `(x, y, 1)` lies.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Plane {
    /// World z of the plane's origin.
    pub z: f64,
    /// The third axis' direction: the affine's third column, normalised.
    pub normal: Vec<f64>,
    /// The third column's length.
    pub thickness: f64,
    pub units: String,
}

impl Plane {
    fn to_json(&self) -> Value {
        json!({
            "z": num(self.z),
            "normal": self.normal.iter().map(|v| num(*v)).collect::<Vec<_>>(),
            "thickness": num(self.thickness),
            "units": self.units,
        })
    }
}

/// One NIfTI file's geometry in MEDH5 conventions.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub notes: Vec<GeometryNote>,
    pub spacing: Vec<f64>,
    pub origin: Vec<f64>,
    /// Row-major, one row per world axis.
    pub direction: Vec<Vec<f64>>,
    pub coord_system: String,
    /// The array's shape, as returned.
    pub shape: Vec<usize>,
    /// Which axis of the file each returned axis is.
    pub axis_order: Vec<usize>,
    pub time_values: Option<Vec<f64>>,
    pub time_units: Option<String>,
    /// Whether the frame times were read, not assumed.
    pub time_measured: bool,
    /// `spatial`, `time` or `channel`: what the leading axis is.
    pub leading_kind: String,
    pub leading_stated: bool,
    pub b_values: Option<Vec<f64>>,
    /// The sidecar field naming the channels, and its values.
    pub channel_field: Option<String>,
    pub channel_values: Option<Vec<f64>>,
    /// The sidecar field that settled the fourth axis.
    pub stated_by: Option<String>,
    /// The file's axes of extent 1 dropped, highest first.
    pub squeezed: Vec<usize>,
    /// The spatial unit (§3.2).
    pub units: String,
    /// NumPy's name for the voxels' type (`str(dtype)`).
    pub dtype: String,
    /// `scl_slope`/`scl_inter` as a rescale (§4.2), not applied.
    pub rescale: Option<(f64, f64)>,
    pub plane_as_volume: Option<Plane>,
    /// The header's `descrip`.
    pub descrip: String,
}

fn floats(values: &[f64]) -> Value {
    Value::Array(values.iter().map(|v| num(*v)).collect())
}

impl Geometry {
    /// The geometry as 2.1.0's `read_nifti` returned it.
    pub fn to_json(&self) -> Value {
        let opt = |v: &Option<Vec<f64>>| v.as_deref().map_or(Value::Null, floats);
        json!({
            "notes": self.notes.iter().map(|n| json!([n.kind, n.message, n.detail])).collect::<Vec<_>>(),
            "spacing": floats(&self.spacing),
            "origin": floats(&self.origin),
            "direction": self.direction.iter().map(|r| floats(r)).collect::<Vec<_>>(),
            "coord_system": self.coord_system,
            "shape": self.shape,
            "axis_order": self.axis_order,
            "time_values": opt(&self.time_values),
            "time_units": self.time_units,
            "time_measured": self.time_measured,
            "leading_kind": self.leading_kind,
            "leading_stated": self.leading_stated,
            "b_values": opt(&self.b_values),
            "channel_field": self.channel_field,
            "channel_values": opt(&self.channel_values),
            "stated_by": self.stated_by,
            "squeezed": self.squeezed,
            "units": self.units,
            "dtype": self.dtype,
            "rescale": self.rescale.map_or(Value::Null, |(s, i)| json!([num(s), num(i)])),
            "plane_as_volume": self.plane_as_volume.as_ref().map_or(Value::Null, Plane::to_json),
            "header": {"descrip": self.descrip, "scl_slope": null, "scl_inter": null},
        })
    }
}

/// A NIfTI file read: its voxels, row-major (C order) in the returned
/// shape and the file's byte order, and its geometry.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct NiftiRead {
    pub voxels: Voxels,
    pub geometry: Geometry,
}

impl Voxels {
    /// Whether the voxels need swapping to be read on this machine.
    fn swapped(&self) -> bool {
        self.element.has_byte_order() && self.big_endian != header::NATIVE_BIG
    }

    /// NumPy's `str(dtype)`: the type's name, or its code with the byte
    /// order where that is not this machine's (`>i2`).
    pub fn numpy_str(&self) -> String {
        if self.swapped() {
            format!("{}{}", if self.big_endian { '>' } else { '<' }, self.element.numpy_code())
        } else {
            self.element.numpy_name().to_string()
        }
    }

    /// NumPy's `dtype.str`: byte order and code (`<i2`, `|u1`, `|V3`).
    pub fn numpy_descr(&self) -> String {
        let order = if !self.element.has_byte_order() {
            '|'
        } else if self.big_endian {
            '>'
        } else {
            '<'
        };
        format!("{order}{}", self.element.numpy_code())
    }

    /// Each `N`-byte unit in this machine's byte order, read from the stored
    /// bytes as it is asked for, not from a swapped copy of them all.
    fn units<const N: usize>(&self) -> impl Iterator<Item = [u8; N]> + '_ {
        let swap = self.swapped();
        self.bytes.chunks_exact(N).map(move |c| {
            let mut unit: [u8; N] = c.try_into().expect("a whole unit");
            if swap {
                unit.reverse();
            }
            unit
        })
    }

    /// The voxels as an engine array, or the refusal the writer gives a
    /// type it cannot store (complex, RGB). Consumed: the stored bytes are
    /// decoded straight into the array and freed, so an image is held
    /// twice at most, while it is decoded.
    pub fn into_ndarray(self) -> Result<NdArray> {
        use header::Element as E;
        let shape = ndarray::IxDyn(&self.shape);
        macro_rules! typed {
            ($t:ty, $variant:ident) => {{
                let values: Vec<$t> = self.units().map(<$t>::from_ne_bytes).collect();
                NdArray::$variant(ndarray::ArrayD::from_shape_vec(shape, values)?)
            }};
        }
        Ok(match self.element {
            E::U8 => NdArray::U8(ndarray::ArrayD::from_shape_vec(shape, self.bytes)?),
            E::I8 => typed!(i8, I8),
            E::I16 => typed!(i16, I16),
            E::U16 => typed!(u16, U16),
            E::I32 => typed!(i32, I32),
            E::U32 => typed!(u32, U32),
            E::I64 => typed!(i64, I64),
            E::U64 => typed!(u64, U64),
            E::F32 => typed!(f32, F32),
            E::F64 => typed!(f64, F64),
            E::C64 | E::C128 | E::Rgb24 | E::Rgba32 => {
                return Err(Error::Type(format!(
                    "arrays of dtype {} cannot be stored; use a boolean, integer or floating dtype",
                    self.numpy_str()
                )))
            }
        })
    }

    /// Each voxel's parts as numbers: one for a real voxel, the real and
    /// imaginary parts of a complex one.
    fn parts(&self) -> Box<dyn Iterator<Item = (f64, f64)> + '_> {
        use header::Element as E;
        let real = |v: f64| (v, 0.0);
        match self.element {
            E::U8 => Box::new(self.bytes.iter().map(move |b| real(f64::from(*b)))),
            E::I8 => Box::new(self.bytes.iter().map(move |b| real(f64::from(*b as i8)))),
            E::I16 => Box::new(self.units().map(move |u| real(f64::from(i16::from_ne_bytes(u))))),
            E::U16 => Box::new(self.units().map(move |u| real(f64::from(u16::from_ne_bytes(u))))),
            E::I32 => Box::new(self.units().map(move |u| real(f64::from(i32::from_ne_bytes(u))))),
            E::U32 => Box::new(self.units().map(move |u| real(f64::from(u32::from_ne_bytes(u))))),
            E::I64 => Box::new(self.units().map(move |u| real(i64::from_ne_bytes(u) as f64))),
            E::U64 => Box::new(self.units().map(move |u| real(u64::from_ne_bytes(u) as f64))),
            E::F32 => Box::new(self.units().map(move |u| real(f64::from(f32::from_ne_bytes(u))))),
            E::F64 => Box::new(self.units().map(move |u| real(f64::from_ne_bytes(u)))),
            // Each part is swapped on its own: the swap unit is a part.
            E::C64 => {
                let mut it = self.units().map(|u| f64::from(f32::from_ne_bytes(u)));
                Box::new(std::iter::from_fn(move || Some((it.next()?, it.next()?))))
            }
            E::C128 => {
                let mut it = self.units().map(f64::from_ne_bytes);
                Box::new(std::iter::from_fn(move || Some((it.next()?, it.next()?))))
            }
            E::Rgb24 | E::Rgba32 => Box::new(
                self.bytes
                    .chunks_exact(self.element.itemsize())
                    .map(move |c| real(f64::from(u8::from(c.iter().any(|b| *b != 0))))),
            ),
        }
    }

    /// The non-zero **physical** voxels: stored values, or under a rescale
    /// the values it gives (stored `1` under intercept `-1` is `0`). A
    /// complex voxel is non-zero when either part is, scaled or not.
    pub fn nonzero(&self, rescale: Option<(f64, f64)>) -> Result<ndarray::ArrayD<bool>> {
        let set: Vec<bool> = match rescale {
            None => self.parts().map(|(re, im)| re != 0.0 || im != 0.0).collect(),
            Some((slope, inter)) => {
                self.parts().map(|(re, im)| re * slope + inter != 0.0 || im * slope != 0.0).collect()
            }
        };
        Ok(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&self.shape), set)?)
    }
}

/// Re-express a 3-D (4x4) or 2-D (3x3) index-to-world affine between RAS
/// and LPS: a sign flip on the first two world axes.
pub fn convert_world(affine: &Array2<f64>, source: &str, target: &str) -> Result<Array2<f64>> {
    for name in [source, target] {
        if !COORD_SYSTEMS.contains(&name) {
            return Err(Error::invalid(format!(
                "unknown coordinate system {}; expected one of {}",
                repr_str(name),
                crate::json::repr_list(&COORD_SYSTEMS)
            )));
        }
    }
    if source == target {
        return Ok(affine.clone());
    }
    let (rows, cols) = affine.dim();
    if !((rows, cols) == (3, 3) || (rows, cols) == (4, 4)) {
        return Err(Error::invalid(format!(
            "RAS↔LPS conversion is defined for 2-D and 3-D affines; got ({rows}, {cols})"
        )));
    }
    // `diag(-1, -1, 1[, 1]) @ affine`, summed as a matrix product sums.
    let flip: Vec<f64> = (0..rows).map(|r| if r < 2 { -1.0 } else { 1.0 }).collect();
    let mut out = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for (k, sign) in flip.iter().enumerate() {
                let d = if k == r { *sign } else { 0.0 };
                acc += d * affine[[k, c]];
            }
            out[[r, c]] = acc;
        }
    }
    Ok(out)
}

/// `[1.0, -0.5]` as Python prints a list of floats.
pub(crate) fn float_list(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|v| crate::json::py_float(*v)).collect();
    format!("[{}]", items.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voxels(element: Element, big_endian: bool, bytes: Vec<u8>) -> Voxels {
        let n = bytes.len() / element.itemsize();
        Voxels { element, big_endian, shape: vec![n], bytes }
    }

    #[test]
    fn s02_5_voxels_decode_in_either_byte_order() {
        let values = [1i16, -2, 300];
        for big in [false, true] {
            let bytes = values.iter().flat_map(|v| if big { v.to_be_bytes() } else { v.to_le_bytes() }).collect();
            match voxels(Element::I16, big, bytes).into_ndarray().unwrap() {
                NdArray::I16(a) => assert_eq!(a.as_slice().unwrap(), &values),
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn s02_5_a_scaled_complex_mask_keeps_its_imaginary_voxels() {
        // (0, 0), (0, 1), (1, 0): under slope 2 the second is 0 + 2i, which
        // is a voxel; dropping the imaginary part lost it.
        let parts = [0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
        for big in [false, true] {
            let bytes = parts.iter().flat_map(|v| if big { v.to_be_bytes() } else { v.to_le_bytes() }).collect();
            let v = voxels(Element::C64, big, bytes);
            for rescale in [None, Some((2.0, 0.0))] {
                assert_eq!(v.nonzero(rescale).unwrap().as_slice().unwrap(), &[false, true, true], "{rescale:?}");
            }
            // An intercept moves only the real part.
            assert_eq!(v.nonzero(Some((1.0, -1.0))).unwrap().as_slice().unwrap(), &[true, true, false]);
        }
    }
}
