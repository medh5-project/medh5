//! Compiles the vendored C-Blosc2 library and the HDF5-Blosc2 filter.
//!
//! C-Blosc2's own CMake build fetches LZ4, Zstd and zlib-ng from the network at
//! configure time, which no reproducible build can allow.  The library is small
//! and its build is regular, so it is compiled here with `cc`: the core sources,
//! the SIMD shuffle kernels each with the instruction-set flag it needs (the
//! dispatcher in `shuffle.c` picks one at run time, as upstream does), and the
//! codecs from the `lz4-sys`, `zstd-sys` and `libz-sys` crates.
//!
//! The filter is upstream's `blosc2_filter.c` --- the same source `hdf5plugin`
//! ships --- compiled against the headers of the HDF5 that `hdf5-metno-sys`
//! builds, so a chunk this library writes is what `hdf5plugin` would write, and
//! every chunk `hdf5plugin` writes is readable here.
//!
//! Everything lands in one static archive.  The core calls the kernels through
//! the dispatcher and the kernels call back into the core's generic fallbacks;
//! split across archives, that cycle depends on the order a single-pass linker
//! happens to see them in.

use std::env;
use std::path::{Path, PathBuf};

#[path = "src/ndebug.rs"]
mod ndebug;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest.join("vendor");
    let blosc_src = vendor.join("c-blosc2").join("blosc");
    let blosc_inc = vendor.join("c-blosc2").join("include");
    let filter_src = vendor.join("hdf5-blosc2");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor");
    println!("cargo:rerun-if-changed=src/b2nd_slice.c");

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let msvc = target_env == "msvc";

    // HDF5 is compiled by now (hdf5-metno-src builds it for hdf5-metno-sys,
    // whose build script runs before this one): stop an optimised build that
    // would link an HDF5 which kept its assertions (see `ndebug`).
    println!("cargo:rerun-if-env-changed={}", ndebug::OPT_OUT);
    let settings = env::var("DEP_HDF5_ROOT")
        .map(|root| Path::new(&root).join("lib").join("libhdf5.settings"))
        .map_err(|_| "hdf5-metno-sys did not say where HDF5 is installed (DEP_HDF5_ROOT)".to_string())
        .and_then(|path| {
            println!("cargo:rerun-if-changed={}", path.display());
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok((path.display().to_string(), text))
        });
    let refusal = ndebug::refusal(
        &env::var("TARGET").unwrap_or_default(),
        env::var("OPT_LEVEL").map_or(true, |level| level != "0"),
        settings.as_ref().map(|(place, text)| (place.as_str(), text.as_str())).map_err(String::as_str),
        env::var(ndebug::OPT_OUT).ok().as_deref(),
    );
    if let Some(message) = refusal {
        eprintln!("{message}");
        std::process::exit(1);
    }
    // `forget_lock_waiters`: the lock this repository's `hdf5-metno-sys` has
    // says so; the published crate's, parking_lot's, does not.
    println!("cargo:rerun-if-env-changed=DEP_HDF5_LOCK_IN_OWN_WORD");
    println!("cargo:rustc-check-cfg=cfg(hdf5_lock_in_own_word)");
    if env::var_os("DEP_HDF5_LOCK_IN_OWN_WORD").is_some() {
        println!("cargo:rustc-cfg=hdf5_lock_in_own_word");
    }

    let x86 = arch == "x86_64" || arch == "x86";
    let neon = arch == "aarch64" && !msvc;

    let codec_includes = codec_include_dirs();
    let new_build = || {
        let mut build = cc::Build::new();
        build
            .include(&blosc_inc)
            .include(&blosc_src)
            .define("BUILD_STATIC", None)
            .define("HAVE_ZLIB", "1")
            .define("HAVE_ZSTD", "1")
            .warnings(false)
            .opt_level(3)
            .pic(true);
        for dir in &codec_includes {
            build.include(dir);
        }
        if !msvc {
            build.flag_if_supported("-std=gnu99");
            build.flag_if_supported("-fvisibility=hidden");
        }
        build
    };

    // --- SIMD kernels, each compiled with its own instruction-set flag -----
    let mut objects: Vec<PathBuf> = Vec::new();
    let mut have_avx512 = false;
    if x86 {
        let mut sse2 = new_build();
        sse2.file(blosc_src.join("shuffle-sse2.c")).file(blosc_src.join("bitshuffle-sse2.c"));
        if !msvc {
            sse2.flag("-msse2");
        } else if arch == "x86" {
            sse2.flag("/arch:SSE2");
        }
        objects.extend(sse2.compile_intermediates());

        let mut avx2 = new_build();
        avx2.file(blosc_src.join("shuffle-avx2.c")).file(blosc_src.join("bitshuffle-avx2.c")).flag(if msvc {
            "/arch:AVX2"
        } else {
            "-mavx2"
        });
        objects.extend(avx2.compile_intermediates());

        let mut avx512 = new_build();
        have_avx512 = if msvc {
            avx512.flag("/arch:AVX512");
            true
        } else if avx512.is_flag_supported("-mavx512f").unwrap_or(false)
            && avx512.is_flag_supported("-mavx512bw").unwrap_or(false)
        {
            avx512.flag("-mavx512f").flag("-mavx512bw");
            true
        } else {
            false
        };
        if have_avx512 {
            avx512.file(blosc_src.join("bitshuffle-avx512.c"));
            objects.extend(avx512.compile_intermediates());
        }
    } else if neon {
        let mut kernels = new_build();
        kernels.file(blosc_src.join("shuffle-neon.c")).flag_if_supported("-flax-vector-conversions");
        objects.extend(kernels.compile_intermediates());
    }

    // --- the HDF5 filter, against the HDF5 that hdf5-metno-sys builds ------
    let hdf5_include =
        env::var("DEP_HDF5_INCLUDE").expect("hdf5-metno-sys did not report its include directory (DEP_HDF5_INCLUDE)");
    let mut filter = new_build();
    filter.include(&filter_src);
    for dir in hdf5_include.split([';', ',']).filter(|s| !s.is_empty()) {
        filter.include(dir);
    }
    filter.file(filter_src.join("blosc2_filter.c"));
    // Ours, not upstream's: the window reads that skip the filter pipeline.
    filter.file(manifest.join("src").join("b2nd_slice.c"));
    objects.extend(filter.compile_intermediates());

    // --- the portable core, the dispatcher, and everything above -----------
    let mut core = new_build();
    if x86 {
        core.define("SHUFFLE_SSE2_ENABLED", None);
        core.define("SHUFFLE_AVX2_ENABLED", None);
        if have_avx512 {
            core.define("SHUFFLE_AVX512_ENABLED", None);
        }
    } else if neon {
        core.define("SHUFFLE_NEON_ENABLED", None);
    }
    for file in [
        "blosc2.c",
        "blosclz.c",
        "fastcopy.c",
        "schunk.c",
        "frame.c",
        "stune.c",
        "delta.c",
        "shuffle.c",
        "shuffle-generic.c",
        "bitshuffle-generic.c",
        "trunc-prec.c",
        "timestamp.c",
        "sframe.c",
        "directories.c",
        "blosc2-stdio.c",
        "b2nd.c",
        "b2nd_utils.c",
    ] {
        core.file(blosc_src.join(file));
    }
    if target_os == "windows" {
        core.file(blosc_src.join("win32").join("threading.c"));
    }
    core.objects(&objects);
    core.compile("medh5_blosc2");

    if target_os == "linux" || target_os == "android" {
        println!("cargo:rustc-link-lib=m");
        if target_env == "musl" {
            // `__builtin_cpu_supports` lives in libgcc.
            println!("cargo:rustc-link-lib=gcc");
        }
    }
    println!("cargo:include={}", blosc_inc.display().to_string().replace('\\', "/"));
}

/// Include directories of the codec libraries, from their -sys crates.
fn codec_include_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for var in ["DEP_LZ4_INCLUDE", "DEP_ZSTD_INCLUDE", "DEP_Z_INCLUDE"] {
        if let Ok(value) = env::var(var) {
            for part in value.split([';', ',']).filter(|s| !s.is_empty()) {
                let path = Path::new(part);
                if path.exists() {
                    out.push(path.to_path_buf());
                }
            }
        }
    }
    out
}
