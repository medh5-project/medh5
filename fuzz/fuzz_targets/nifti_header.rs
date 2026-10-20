//! NIfTI files the input chose, read as the converter reads them: a header
//! of either version and byte order decoded, checked and re-encoded, and a
//! whole file --- `.nii`, `.nii.gz` inflated from bytes the input chose or
//! compressed here, or a `.hdr`/`.img` pair --- through `read_nifti`.  Any
//! refusal is fine; a crash, or a read or an inflation past what the header
//! implies, is a finding.
#![no_main]

use std::io::Write;
use std::path::PathBuf;

use arbitrary::Arbitrary;
use flate2::write::GzEncoder;
use flate2::Compression;
use libfuzzer_sys::fuzz_target;
use medh5::convert::nifti::{read_nifti, Header, ReadOptions, Version};

#[derive(Arbitrary, Debug)]
enum Layout {
    /// `.nii`, the bytes as given.
    Single,
    /// `.nii.gz`, the bytes compressed here.
    Gzipped,
    /// `.nii.gz`, the bytes as given: the inflater's input.
    RawGzip,
    /// `.hdr` holding the bytes, `.img` the image.
    Pair,
}

#[derive(Arbitrary, Debug)]
struct Input {
    layout: Layout,
    bytes: Vec<u8>,
    image: Vec<u8>,
    transpose: bool,
    plane_as_volume: bool,
    assume_geometry: bool,
    channel: bool,
}

fn scratch() -> PathBuf {
    let name = format!("medh5-fuzz-nifti-{}", std::process::id());
    let shm = PathBuf::from("/dev/shm");
    let dir = if shm.is_dir() { shm.join(name) } else { std::env::temp_dir().join(name) };
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A header decoded, checked and encoded is decoded to what encodes the
/// same bytes again.
fn round_trip(bytes: &[u8]) {
    for version in [Version::One, Version::Two] {
        let size = version.size();
        if bytes.len() < size {
            continue;
        }
        let Ok(mut header) = Header::decode(&bytes[..size], version) else { continue };
        let _ = header.check_fix();
        let once = header.encode();
        let again = Header::decode(&once, version).unwrap().encode();
        assert_eq!(once, again, "a decoded header encodes the same bytes twice");
    }
}

fuzz_target!(|input: Input| {
    round_trip(&input.bytes);
    let dir = scratch();
    let path = match input.layout {
        Layout::Single => {
            let path = dir.join("x.nii");
            std::fs::write(&path, &input.bytes).unwrap();
            path
        }
        Layout::Gzipped => {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(&input.bytes).unwrap();
            let path = dir.join("x.nii.gz");
            std::fs::write(&path, encoder.finish().unwrap()).unwrap();
            path
        }
        Layout::RawGzip => {
            let path = dir.join("x.nii.gz");
            std::fs::write(&path, &input.bytes).unwrap();
            path
        }
        Layout::Pair => {
            std::fs::write(dir.join("x.img"), &input.image).unwrap();
            let path = dir.join("x.hdr");
            std::fs::write(&path, &input.bytes).unwrap();
            path
        }
    };
    let options = ReadOptions::default()
        .with_transpose(input.transpose)
        .with_plane_as_volume(input.plane_as_volume)
        .with_assume_geometry(input.assume_geometry)
        .with_fourth_axis(if input.channel { "channel" } else { "auto" });
    if let Ok(read) = read_nifti(&path, &options) {
        // What was read is what the shape says: never more.
        let count: usize = read.voxels.shape.iter().product();
        assert_eq!(read.voxels.bytes.len(), count * read.voxels.element.itemsize());
        let _ = read.voxels.into_ndarray();
        let _ = read.geometry.to_json();
    }
    let _ = std::fs::remove_dir_all(&dir);
});
