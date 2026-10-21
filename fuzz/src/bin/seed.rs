//! Seed a target's corpus: `open_validate`'s with the conformance corpus's
//! small files, `itk_transform`'s with ITK transform files the engine writes.
//!
//!     cargo run --release --bin seed -- open_validate corpus/open_validate
//!     cargo run --release --bin seed -- itk_transform corpus/itk_transform
//!
//! Each conformance case the engine builds, of 256 KiB or less: the larger
//! ones slow every run more than they add.

use std::path::{Path, PathBuf};

use medh5::convert::{write_itk_h5, ItkEntry};

const LARGEST: u64 = 256 * 1024;

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(target), Some(out)) = (args.next(), args.next().map(PathBuf::from)) else {
        eprintln!("usage: seed TARGET DIR");
        std::process::exit(2);
    };
    std::fs::create_dir_all(&out).expect("the corpus directory is writable");
    let kept = match target.as_str() {
        "open_validate" => conformance(&out),
        "itk_transform" => itk(&out),
        _ => {
            eprintln!("seed: no seeds for {target}");
            std::process::exit(2);
        }
    };
    println!("{kept} seeds in {}", out.display());
}

fn conformance(out: &Path) -> usize {
    let built = std::env::temp_dir().join(format!("medh5-fuzz-seed-{}", std::process::id()));
    medh5::conformance::build_corpus(&built, None).expect("the corpus builds");
    let mut kept = 0;
    for entry in std::fs::read_dir(&built).expect("the built corpus is readable") {
        let path = entry.expect("a directory entry").path();
        let hdf5 = path.extension().is_some_and(|e| e == "medh5" || e == "medh5c");
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(u64::MAX);
        if hdf5 && size <= LARGEST {
            std::fs::copy(&path, out.join(path.file_name().expect("a file name"))).expect("the seed is written");
            kept += 1;
        }
    }
    let _ = std::fs::remove_dir_all(&built);
    kept
}

/// One file of each shape the reader takes: a single affine, a composite of
/// a translation and an affine, and a B-spline with its fixed parameters.
fn itk(out: &Path) -> usize {
    let affine = || {
        ItkEntry::new(
            "AffineTransform_double_3_3",
            vec![1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.2, 1.0, 2.0, 3.0],
            vec![5.0, 6.0, 7.0],
        )
    };
    let translation = ItkEntry::new("TranslationTransform_double_3_3", vec![1.0, -2.0, 0.5], vec![]);
    let composite = ItkEntry::new("CompositeTransform_double_3_3", vec![], vec![]);
    let mut fixed = vec![4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0];
    fixed.extend([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let bspline = ItkEntry::new("BSplineTransform_double_3_3", vec![0.25; 3 * 64], fixed);
    let files = [
        ("affine.h5", vec![affine()]),
        ("composite.h5", vec![composite, translation, affine()]),
        ("bspline.h5", vec![bspline]),
    ];
    for (name, entries) in &files {
        write_itk_h5(&out.join(name), entries).expect("the seed is written");
    }
    files.len()
}
