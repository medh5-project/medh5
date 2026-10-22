//! ITK's HDF5 transform files: the layout ITK's `HDF5TransformIO` reads and
//! writes, through the engine's HDF5 (medh5 does not depend on `h5py`).
//!
//! A file holds `/TransformGroup/<n>/` for `n = 0, 1, …`, each with a
//! `TransformType` string (`AffineTransform_double_3_3`) and, except for a
//! composite, `TransformParameters` and `TransformFixedParameters`.  A
//! composite is entry 0, its components the entries after it, in the order
//! the text format lists them.  What the entries mean --- which types are
//! read, how a composite composes --- is the Python converter's
//! (`medh5.io.itk`), as it is for the text format: this module moves the
//! entries in and out of the file and nothing else.

use std::path::Path;

use hdf5::types::VarLenAscii;
use ndarray::Array1;

use crate::h5::file::{open_read, AtomicFile};
use crate::h5::{data, ops};
use crate::json::repr_str;
use crate::{Error, Result};

/// The group every transform of the file is a member of.
const GROUP: &str = "TransformGroup";
const TYPE: &str = "TransformType";
const PARAMETERS: &str = "TransformParameters";
const FIXED: &str = "TransformFixedParameters";
/// The misspellings earlier ITK releases wrote, which its reader still falls
/// back to.
const LEGACY_PARAMETERS: &str = "TranformParameters";
const LEGACY_FIXED: &str = "TranformFixedParameters";

/// One transform of an ITK transform file, as the file lists it: ITK's type
/// name, its parameters and its fixed parameters.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ItkEntry {
    /// ITK's type name, `AffineTransform_double_3_3`.
    pub name: String,
    /// `TransformParameters`: none for a composite.
    pub parameters: Vec<f64>,
    /// `TransformFixedParameters`: none for a composite.
    pub fixed_parameters: Vec<f64>,
}

impl ItkEntry {
    /// An entry of type `name`.
    pub fn new(name: impl Into<String>, parameters: Vec<f64>, fixed_parameters: Vec<f64>) -> ItkEntry {
        ItkEntry { name: name.into(), parameters, fixed_parameters }
    }

    fn is_composite(&self) -> bool {
        self.name.starts_with("CompositeTransform")
    }
}

/// The entries of the ITK HDF5 transform file at `path`, in the order of
/// their numbers.
///
/// Refused: a file with no `/TransformGroup`, or none in it; members not
/// numbered `0` to `n - 1`; an entry without one `TransformType` string, or
/// whose parameters are not one-dimensional numbers; and parameters on a
/// composite, which ITK neither writes nor reads.  Parameters under the
/// misspelling earlier ITK releases wrote (`TranformParameters`) are read as ITK
/// reads them.
pub fn read_itk_h5(path: &Path) -> Result<Vec<ItkEntry>> {
    let shown = repr_str(&path.to_string_lossy());
    let refused = |why: String| Error::invalid(format!("{shown} is not an ITK transform file: {why}"));
    let file = open_read(path)?;
    let file_bytes = std::fs::metadata(path).map_or(u64::MAX, |meta| meta.len());
    let group = file.group(GROUP).map_err(|_| refused(format!("it has no /{GROUP}")))?;
    let mut numbered = Vec::new();
    for name in group.member_names()? {
        match name.parse::<usize>() {
            Ok(n) if n.to_string() == name => numbered.push(n),
            _ => return Err(refused(format!("/{GROUP} holds {}, which is not a transform's number", repr_str(&name)))),
        }
    }
    numbered.sort_unstable();
    if numbered.is_empty() {
        return Err(refused(format!("/{GROUP} holds no transform")));
    }
    if numbered.iter().enumerate().any(|(i, n)| i != *n) {
        return Err(refused(format!(
            "/{GROUP}'s transforms are numbered {numbered:?}, where ITK numbers them from 0 with none left out"
        )));
    }
    let mut entries = Vec::with_capacity(numbered.len());
    for n in numbered {
        let at = format!("/{GROUP}/{n}");
        let member = group.group(&n.to_string()).map_err(|_| refused(format!("{at} is not a group")))?;
        let kind = member.dataset(TYPE).map_err(|_| refused(format!("{at} has no {TYPE}")))?;
        if !data::is_strings(&kind) {
            return Err(refused(format!("{at}/{TYPE} is not a string")));
        }
        let mut names = data::read_strings(&kind)?;
        if names.len() != 1 {
            return Err(refused(format!("{at}/{TYPE} holds {} strings, not one", names.len())));
        }
        let numbers = |dataset: &str, legacy: &str| -> Result<Option<Vec<f64>>> {
            let (dataset, ds) = match member.dataset(dataset) {
                Ok(ds) => (dataset, ds),
                Err(_) => match member.dataset(legacy) {
                    Ok(ds) => (legacy, ds),
                    Err(_) => return Ok(None),
                },
            };
            if data::is_strings(&ds) || ds.ndim() > 1 {
                return Err(refused(format!("{at}/{dataset} is not a one-dimensional array of numbers")));
            }
            if !stored_in_full(&ds, file_bytes) {
                return Err(refused(format!(
                    "{at}/{dataset} states {} values and the file does not store them all",
                    ds.shape().first().copied().unwrap_or(1)
                )));
            }
            Ok(Some(into_f64(data::read(&ds)?)))
        };
        let entry =
            ItkEntry::new(names.swap_remove(0).trim(), numbers(PARAMETERS, LEGACY_PARAMETERS)?.unwrap_or_default(), {
                numbers(FIXED, LEGACY_FIXED)?.unwrap_or_default()
            });
        if entry.is_composite() && !(entry.parameters.is_empty() && entry.fixed_parameters.is_empty()) {
            return Err(refused(format!("{at} is a composite with parameters of its own, which ITK does not write")));
        }
        entries.push(entry);
    }
    Ok(entries)
}

/// Whether the file stores every value `ds` states.
///
/// A chunk HDF5 never wrote reads as the fill value, so a dataset can state
/// any number of values and store none: a 9.6 KiB file stated 240 million
/// parameters, and reading them took 11 GB.  ITK writes every parameter it
/// states, so an array is read only when its storage is allocated in full,
/// takes no more bytes than the file (chunks that share bytes would), and,
/// unfiltered, takes the bytes its values do.
fn stored_in_full(ds: &hdf5::Dataset, file_bytes: u64) -> bool {
    use crate::h5sys::h5d;

    let Some(values) = ds.shape().iter().try_fold(1u64, |n, e| n.checked_mul(*e as u64)) else {
        return false;
    };
    if values == 0 {
        return true;
    }
    let mut status = h5d::H5D_space_status_t::H5D_SPACE_STATUS_ERROR;
    // SAFETY: a live dataset id, and a status HDF5 writes.
    if crate::h5::locked(|| unsafe { h5d::H5Dget_space_status(ds.id(), &mut status) }) < 0
        || status != h5d::H5D_space_status_t::H5D_SPACE_STATUS_ALLOCATED
    {
        return false;
    }
    let stored = ds.storage_size();
    let bytes = values.saturating_mul(ds.dtype().map_or(u64::MAX, |t| t.size() as u64));
    stored <= file_bytes && (stored >= bytes || !ds.filters().is_empty())
}

/// An array's values as `f64`, without a copy when they are `f64` already.
fn into_f64(array: crate::array::NdArray) -> Vec<f64> {
    let flat = |a: ndarray::ArrayD<f64>| match a.as_slice() {
        Some(_) => a.into_raw_vec_and_offset().0,
        None => a.iter().copied().collect(),
    };
    match array {
        crate::array::NdArray::F64(a) => flat(a),
        other => flat(other.to_f64()),
    }
}

/// Write `entries` as the ITK HDF5 transform file `path`, replacing it
/// atomically (1.0 §14.4): numbered from 0, a composite with its type alone,
/// as ITK writes one, and every string as ITK's variable-length ASCII.
///
/// Refused: no entries, and a type name that is empty, not ASCII, or holds a
/// NUL, where HDF5 would end the stored string.
pub fn write_itk_h5(path: &Path, entries: &[ItkEntry]) -> Result<()> {
    if entries.is_empty() {
        return Err(Error::invalid("an ITK transform file holds at least one transform"));
    }
    if let Some(bad) = entries.iter().find(|e| !e.name.is_ascii() || e.name.is_empty() || e.name.contains('\0')) {
        return Err(Error::invalid(format!("{} is not an ITK transform type", repr_str(&bad.name))));
    }
    let staged = AtomicFile::create(path)?;
    let group = ops::new_group(staged.handle(), GROUP)?;
    for (n, entry) in entries.iter().enumerate() {
        let member = ops::new_group(&group, &n.to_string())?;
        let name = VarLenAscii::from_ascii(&entry.name)
            .map_err(|e| Error::invalid(format!("{}: {e}", repr_str(&entry.name))))?;
        member.new_dataset_builder().obj_track_times(false).with_data(&[name]).create(TYPE)?;
        if entry.is_composite() {
            continue;
        }
        for (dataset, values) in [(FIXED, &entry.fixed_parameters), (PARAMETERS, &entry.parameters)] {
            let array = Array1::from(values.clone());
            member.new_dataset_builder().obj_track_times(false).with_data(array.view()).create(dataset)?;
        }
    }
    staged.commit()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn affine() -> ItkEntry {
        ItkEntry::new(
            "AffineTransform_double_3_3",
            vec![1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.2, 1.0, 2.0, 3.0],
            vec![5.0, 6.0, 7.0],
        )
    }

    #[test]
    fn itk_h5_entries_round_trip_with_a_composite_bare() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let composite = ItkEntry::new("CompositeTransform_double_3_3", vec![], vec![]);
        let entries = vec![composite, affine(), ItkEntry::new("TranslationTransform_double_3_3", vec![1.0; 3], vec![])];
        write_itk_h5(&path, &entries).unwrap();
        assert_eq!(read_itk_h5(&path).unwrap(), entries);
        let file = crate::h5::file::marked(|| hdf5::File::open(&path)).unwrap();
        let bare = file.group("TransformGroup/0").unwrap();
        assert_eq!(bare.member_names().unwrap(), ["TransformType"]);
        let kind = bare.dataset("TransformType").unwrap();
        assert_eq!(kind.shape(), [1]);
        assert!(matches!(kind.dtype().unwrap().to_descriptor().unwrap(), hdf5::types::TypeDescriptor::VarLenAscii));
    }

    /// A file whose parameter dataset is chunked, its chunks written or not.
    fn chunked(path: &Path, values: usize, written: Option<&[f64]>, deflate: bool) {
        let file = crate::h5::file::marked(|| hdf5::File::create(path)).unwrap();
        let zero = file.create_group(GROUP).unwrap().create_group("0").unwrap();
        data::create_strings(&zero, TYPE, &["AffineTransform_double_3_3".into()]).unwrap();
        let mut builder = zero.new_dataset::<f64>().shape([values]).chunk([4]);
        if deflate {
            builder = builder.deflate(9);
        }
        let ds = builder.create(PARAMETERS).unwrap();
        if let Some(written) = written {
            ds.write_slice(written, ndarray::s![..written.len()]).unwrap();
        }
        zero.new_dataset_builder().with_data(Array1::from(vec![5.0, 6.0, 7.0]).view()).create(FIXED).unwrap();
    }

    /// An unwritten chunk reads as the fill value, so a dataset states values
    /// it does not store: 2**28 parameters (2 GiB) in a file of a few KiB were
    /// read, and held three times over.  Such an array is refused before
    /// anything is read; so is one with a chunk of its twelve left unwritten.
    #[test]
    fn itk_h5_parameters_the_file_does_not_store_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        chunked(&path, 1 << 28, None, false);
        let err = read_itk_h5(&path).unwrap_err().to_string();
        assert!(err.contains("states 268435456 values and the file does not store them all"), "{err}");
        let twelve = affine().parameters;
        chunked(&path, 12, Some(&twelve[..8]), false);
        let err = read_itk_h5(&path).unwrap_err().to_string();
        assert!(err.contains("/TransformGroup/0/TransformParameters states 12 values"), "{err}");
        for deflate in [false, true] {
            chunked(&path, 12, Some(&twelve), deflate);
            assert_eq!(read_itk_h5(&path).unwrap(), [affine()], "deflate {deflate}");
        }
    }

    #[test]
    fn itk_h5_the_legacy_misspelled_parameters_are_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        {
            let file = crate::h5::file::marked(|| hdf5::File::create(&path)).unwrap();
            let zero = file.create_group("TransformGroup").unwrap().create_group("0").unwrap();
            data::create_strings(&zero, "TransformType", &["AffineTransform_double_3_3".into()]).unwrap();
            let entry = affine();
            zero.new_dataset_builder()
                .with_data(Array1::from(entry.parameters).view())
                .create("TranformParameters")
                .unwrap();
            let fixed = Array1::from(entry.fixed_parameters);
            zero.new_dataset_builder().with_data(fixed.view()).create("TranformFixedParameters").unwrap();
        }
        assert_eq!(read_itk_h5(&path).unwrap(), [affine()]);
    }

    /// HDF5 ends a variable-length string at its first NUL, so a type name
    /// holding one would be read back cut short: it is refused, as a name
    /// that is not ASCII is.
    #[test]
    fn itk_h5_a_type_name_holding_a_nul_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        let cut = ItkEntry::new("AffineTransform_double_3_3\0x", affine().parameters, affine().fixed_parameters);
        let err = write_itk_h5(&path, &[cut]).unwrap_err().to_string();
        assert!(err.contains("is not an ITK transform type"), "{err}");
        assert!(!path.exists());
    }

    #[test]
    fn itk_h5_a_file_numbered_with_a_gap_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.h5");
        {
            let file = crate::h5::file::marked(|| hdf5::File::create(&path)).unwrap();
            let one = file.create_group("TransformGroup").unwrap().create_group("1").unwrap();
            data::create_strings(&one, "TransformType", &["IdentityTransform_double_3_3".into()]).unwrap();
        }
        let err = read_itk_h5(&path).unwrap_err().to_string();
        assert!(err.contains("numbered [1]"), "{err}");
        let other = dir.path().join("o.h5");
        crate::h5::file::marked(|| hdf5::File::create(&other)).unwrap().create_group("images").unwrap();
        let err = read_itk_h5(&other).unwrap_err().to_string();
        assert!(err.contains("has no /TransformGroup"), "{err}");
    }
}
