//! Any file, validated at the strictest level: a report or a refusal is
//! fine; a crash, a sanitizer's report, a timeout or running out of memory is
//! a finding.
#![no_main]

use std::path::PathBuf;

use libfuzzer_sys::fuzz_target;

/// Where the input is written: memory, where `/dev/shm` exists.
fn scratch() -> PathBuf {
    let name = format!("medh5-fuzz-{}.medh5", std::process::id());
    let shm = PathBuf::from("/dev/shm");
    if shm.is_dir() {
        shm.join(name)
    } else {
        std::env::temp_dir().join(name)
    }
}

fuzz_target!(|data: &[u8]| {
    let path = scratch();
    std::fs::write(&path, data).expect("the scratch file is writable");
    let _ = medh5::validate::validate_file(&path, "strict", None);
    let _ = std::fs::remove_file(&path);
});
