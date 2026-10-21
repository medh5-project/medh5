//! Any file, read as an ITK HDF5 transform file (`read_itk_h5`), as `medh5
//! from-itk-transform` and `medh5.io.itk` read one.  A refusal is fine; a
//! crash, a sanitizer's report, a timeout or running out of memory is a
//! finding.  Entries that are read are written and read again, and must come
//! back as they were: what the reader accepts, the writer stores exactly or
//! refuses.
#![no_main]

use std::path::PathBuf;

use libfuzzer_sys::fuzz_target;
use medh5::convert::{read_itk_h5, write_itk_h5, ItkEntry};

/// Where the input is written: memory, where `/dev/shm` exists.
fn scratch(what: &str) -> PathBuf {
    let name = format!("medh5-fuzz-itk-{what}-{}.h5", std::process::id());
    let shm = PathBuf::from("/dev/shm");
    if shm.is_dir() {
        shm.join(name)
    } else {
        std::env::temp_dir().join(name)
    }
}

/// The same entry, a NaN parameter equal to itself.
fn same(a: &ItkEntry, b: &ItkEntry) -> bool {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    a.name == b.name
        && bits(&a.parameters) == bits(&b.parameters)
        && bits(&a.fixed_parameters) == bits(&b.fixed_parameters)
}

fuzz_target!(|data: &[u8]| {
    let path = scratch("in");
    std::fs::write(&path, data).expect("the scratch file is writable");
    if let Ok(entries) = read_itk_h5(&path) {
        let again = scratch("out");
        if write_itk_h5(&again, &entries).is_ok() {
            let read = read_itk_h5(&again).expect("what the writer wrote is read");
            assert!(
                read.len() == entries.len() && read.iter().zip(&entries).all(|(a, b)| same(a, b)),
                "entries written and read again changed: {entries:?} became {read:?}"
            );
        }
        let _ = std::fs::remove_file(&again);
    }
    let _ = std::fs::remove_file(&path);
});
