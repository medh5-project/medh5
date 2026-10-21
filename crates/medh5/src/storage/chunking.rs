//! Cache-aware chunk sizing (spec §14.1).
//!
//! HDF5 decompresses a whole chunk to serve any element of it, so the chunk is
//! the real unit of I/O.  Sizing it to the L3 cache keeps a patch read inside
//! cache after decompression; sizing it near the training patch keeps read
//! amplification low.  Those two pull in opposite directions; this resolves
//! them by starting at the patch, growing toward the cache budget, and stopping
//! before the chunk is much larger than the patch.

use std::sync::OnceLock;

use serde_json::{json, Value};

use crate::geometry::Grid;
use crate::json::repr_int_tuple;
use crate::{Error, Result};

/// Fallback L3 slice, ~1.375 MiB.
pub const DEFAULT_L3_BYTES: u64 = 1_441_792;
pub const MIN_CHUNK_BYTES: f64 = 512.0 * 1024.0;
pub const MAX_CHUNK_BYTES: f64 = 4.0 * 1024.0 * 1024.0;
pub const CACHE_SAFETY: f64 = 0.8;
/// Stop growing once the mean chunk/patch ratio exceeds this.
pub const OVERSHOOT_LIMIT: f64 = 1.5;
pub const DEFAULT_PATCH: usize = 64;

/// Best-effort detection of one CPU's share of L3, falling back to
/// [`DEFAULT_L3_BYTES`].
///
/// The cache is shared: the size the platform reports is the whole of it,
/// which every CPU sharing it divides --- 33 MiB among four, or among
/// sixteen --- so the size is divided by the number of CPUs that share it.
pub fn detect_l3_bytes() -> u64 {
    static L3: OnceLock<u64> = OnceLock::new();
    *L3.get_or_init(|| {
        let index3 = "/sys/devices/system/cpu/cpu0/cache/index3";
        if let Some(total) = std::fs::read_to_string(format!("{index3}/size")).ok().as_deref().and_then(cache_size) {
            let sharing = std::fs::read_to_string(format!("{index3}/shared_cpu_list")).ok();
            return per_sharing_cpu(total, sharing.as_deref().and_then(cpu_list_len));
        }
        #[cfg(target_os = "macos")]
        {
            if let Some(v) = macos_l3() {
                return v;
            }
        }
        DEFAULT_L3_BYTES
    })
}

/// sysfs's cache `size`: `33792K`, `32M` or bytes.
fn cache_size(raw: &str) -> Option<u64> {
    let raw = raw.trim();
    let (digits, unit) = match raw.strip_suffix('K') {
        Some(k) => (k, 1024),
        None => match raw.strip_suffix('M') {
            Some(m) => (m, 1024 * 1024),
            None => (raw, 1),
        },
    };
    digits.parse::<u64>().ok().map(|v| v * unit).filter(|v| *v > 0)
}

/// How many CPUs a sysfs CPU list names: `0-3`, `0-7,64-71`, `5`.
fn cpu_list_len(raw: &str) -> Option<u64> {
    let mut count = 0;
    for part in raw.trim().split(',') {
        count += match part.split_once('-') {
            Some((a, b)) => b.trim().parse::<u64>().ok()?.checked_sub(a.trim().parse::<u64>().ok()?)? + 1,
            None => {
                part.trim().parse::<u64>().ok()?;
                1
            }
        };
    }
    Some(count)
}

/// One CPU's share of a cache of `total` bytes that `sharing` CPUs share.
fn per_sharing_cpu(total: u64, sharing: Option<u64>) -> u64 {
    total / sharing.filter(|n| *n > 0).unwrap_or(1)
}

/// `hw.cacheconfig` counts the CPUs sharing each level of cache, memory
/// first: L3's count is the fourth.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn cacheconfig_l3_sharing(raw: &str) -> Option<u64> {
    raw.split_whitespace().nth(3)?.parse().ok()
}

#[cfg(target_os = "macos")]
fn macos_l3() -> Option<u64> {
    let sysctl = |key: &str| -> Option<String> {
        let mut command = std::process::Command::new("/usr/sbin/sysctl");
        command.args(["-n", key]).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
        let out = crate::h5::spawn(&mut command).ok()?.wait_with_output().ok()?;
        Some(String::from_utf8_lossy(&out.stdout).into_owned())
    };
    let total: u64 = sysctl("hw.l3cachesize")?.trim().parse().ok()?;
    if total == 0 {
        return None;
    }
    Some(per_sharing_cpu(total, sysctl("hw.cacheconfig").as_deref().and_then(cacheconfig_l3_sharing)))
}

fn budget(l3_bytes: Option<u64>) -> f64 {
    let target = l3_bytes.unwrap_or_else(detect_l3_bytes) as f64 * CACHE_SAFETY;
    target.clamp(MIN_CHUNK_BYTES, MAX_CHUNK_BYTES)
}

fn pow2_ceil(p: usize) -> usize {
    let exp = (p.max(1) as f64).log2().ceil().max(0.0) as u32;
    1usize << exp
}

/// A training patch hint: none, one size for every axis, or per axis.
#[derive(Debug, Clone, PartialEq)]
pub enum Patch {
    Default,
    Uniform(usize),
    PerAxis(Vec<i64>),
}

/// Chunk extents for the spatial axes alone.
pub fn spatial_chunk_for(
    spatial_shape: &[usize],
    patch: &Patch,
    itemsize: usize,
    l3_bytes: Option<u64>,
) -> Result<Vec<usize>> {
    if spatial_shape.is_empty() || spatial_shape.contains(&0) {
        return Err(Error::invalid(format!("spatial shape must be positive, got {}", repr_int_tuple(spatial_shape))));
    }
    let patch_t: Vec<usize> = match patch {
        Patch::Default => spatial_shape.iter().map(|s| DEFAULT_PATCH.min(*s)).collect(),
        Patch::Uniform(p) => spatial_shape.iter().map(|s| (*p).min(*s)).collect(),
        Patch::PerAxis(p) => {
            if p.len() != spatial_shape.len() {
                return Err(Error::invalid(format!(
                    "patch {} does not match spatial shape {}",
                    repr_int_tuple(p),
                    repr_int_tuple(spatial_shape)
                )));
            }
            p.iter().map(|v| (*v).max(0) as usize).collect()
        }
    };
    let patch_t: Vec<usize> = patch_t.iter().zip(spatial_shape).map(|(p, s)| (*p).min(*s).max(1)).collect();
    let mut chunk: Vec<usize> = patch_t.iter().zip(spatial_shape).map(|(p, s)| pow2_ceil(*p).min(*s)).collect();
    let budget = budget(l3_bytes);
    let nbytes = |c: &[usize]| c.iter().map(|v| *v as f64).product::<f64>() * itemsize as f64;
    let ratio = |c: &[usize], i: usize| c[i] as f64 / patch_t[i] as f64;
    // The patch rounded up to a power of two can be past the ceiling by
    // itself --- a 96³ float32 patch started at 8 MiB, a 160³ int16 one at
    // 32 MiB --- and growth only ever made a chunk larger.  Past it, the
    // chunk starts at the patch itself, halved on the axis furthest past
    // its patch until it is within the ceiling.
    if nbytes(&chunk) > MAX_CHUNK_BYTES {
        chunk = patch_t.clone();
        while nbytes(&chunk) > MAX_CHUNK_BYTES {
            let Some(axis) = (0..chunk.len()).filter(|i| chunk[*i] > 1).reduce(|best, i| {
                if ratio(&chunk, i) > ratio(&chunk, best) {
                    i
                } else {
                    best
                }
            }) else {
                break;
            };
            chunk[axis] = chunk[axis].div_ceil(2);
        }
    }
    while nbytes(&chunk) < budget {
        let growable: Vec<usize> = (0..chunk.len()).filter(|i| chunk[*i] < spatial_shape[*i]).collect();
        if growable.is_empty() {
            break;
        }
        // `min(growable, key=ratio)` keeps the first minimum.
        let mut axis = growable[0];
        for &i in &growable[1..] {
            if (chunk[i] as f64 / patch_t[i] as f64) < (chunk[axis] as f64 / patch_t[axis] as f64) {
                axis = i;
            }
        }
        let step = pow2_ceil(patch_t[axis]);
        let mut grown = chunk.clone();
        grown[axis] = (grown[axis] + step).min(spatial_shape[axis]);
        let mean: f64 =
            grown.iter().zip(&patch_t).map(|(c, p)| *c as f64 / *p as f64).sum::<f64>() / grown.len() as f64;
        if mean > OVERSHOOT_LIMIT || nbytes(&grown) > MAX_CHUNK_BYTES {
            break;
        }
        chunk = grown;
    }
    Ok(chunk.iter().zip(spatial_shape).map(|(c, s)| (*c).min(*s)).collect())
}

/// Full chunk shape for an array whose axes are described by `axis_kinds`.
///
/// Non-spatial axes get extent 1; `leading` prepends that many extra size-1
/// axes --- the `(1, *spatial_chunk)` requirement of stacked encodings.
pub fn optimize_chunks(
    shape: &[usize],
    axis_kinds: &[String],
    patch: &Patch,
    itemsize: usize,
    l3_bytes: Option<u64>,
    leading: usize,
) -> Result<Vec<usize>> {
    if axis_kinds.len() + leading != shape.len() {
        return Err(Error::invalid(format!(
            "axis_kinds {} (+{leading} leading) does not describe shape {}",
            crate::json::repr_list(axis_kinds),
            repr_int_tuple(shape)
        )));
    }
    let body = &shape[leading..];
    let spatial_idx: Vec<usize> =
        axis_kinds.iter().enumerate().filter(|(_, k)| *k == "spatial").map(|(i, _)| i).collect();
    let spatial_shape: Vec<usize> = spatial_idx.iter().map(|i| body[*i]).collect();
    let spatial_chunk = spatial_chunk_for(&spatial_shape, patch, itemsize, l3_bytes)?;
    let mut out = vec![1usize; body.len()];
    for (slot, i) in spatial_idx.iter().enumerate() {
        out[*i] = spatial_chunk[slot];
    }
    let mut full = vec![1usize; leading];
    full.extend(out);
    Ok(full)
}

/// The chunk shape the writer gives an array laid out on `grid` (§14.1).
///
/// A `chunk_hint` is a preference, held to the array: each entry, positive
/// (§3.2, E109), is clipped to its extent, as the computed chunks are ---
/// HDF5 refuses a chunk larger than a fixed-size dataset.  A stacked encoding
/// (`leading` planes) takes the hint's spatial chunk one plane deep, as it
/// takes a computed one: it fell back to the optimiser, so labels and the
/// images on their grid were chunked apart (§14.1).
pub fn grid_chunks(grid: &Grid, itemsize: usize, leading: usize) -> Result<Vec<usize>> {
    if let Some(hint) = &grid.chunk_hint {
        let clipped = hint.iter().zip(grid.shape_usize()).map(|(c, n)| (*c).clamp(1, n.max(1) as i64) as usize);
        if leading == 0 {
            return Ok(clipped.collect());
        }
        let mut out = vec![1usize; leading];
        out.extend(clipped.zip(&grid.axis_kinds).map(|(c, kind)| if kind == "spatial" { c } else { 1 }));
        return Ok(out);
    }
    let mut shape = vec![1usize; leading];
    shape.extend(grid.shape_usize());
    let patch = match &grid.patch_hint {
        Some(p) => Patch::PerAxis(p.clone()),
        None => Patch::Default,
    };
    optimize_chunks(&shape, &grid.axis_kinds, &patch, itemsize, None, leading)
}

/// `proposed`'s trailing entries clipped to `shape`, or `None`.
pub fn fit_chunks(proposed: &[usize], shape: &[usize]) -> Option<Vec<usize>> {
    let dims = shape.len();
    if proposed.len() < dims {
        return None;
    }
    let tail = &proposed[proposed.len() - dims..];
    Some(tail.iter().zip(shape).map(|(c, s)| (*c).min(*s)).collect())
}

/// Chunks for a displacement field on `grid`: one vector component per chunk.
pub fn field_chunks(grid: &Grid, shape: &[usize], itemsize: usize) -> Result<Option<Vec<usize>>> {
    if shape.len() != grid.n_spatial() + 1 {
        return Ok(None);
    }
    let all = grid_chunks(grid, itemsize, 0)?;
    let spatial = &all[all.len() - grid.n_spatial()..];
    let mut proposed = vec![1usize];
    proposed.extend_from_slice(spatial);
    Ok(fit_chunks(&proposed, shape))
}

/// Diagnostics for `medh5 info`: chunk bytes, count.
///
/// A count past 64 bits --- extents a file states, chunked finely --- is
/// `null`, as an unknown count is, rather than a wrapped one.
pub fn chunk_report(shape: &[usize], chunks: &[usize], itemsize: usize) -> Value {
    let chunk_bytes = chunks.iter().try_fold(itemsize, |n, c| n.checked_mul(*c));
    let n_chunks = shape.iter().zip(chunks).try_fold(1usize, |n, (s, c)| n.checked_mul(s.div_ceil((*c).max(1))));
    let mib = chunk_bytes.map(|b| crate::geometry::grid::round_to(b as f64 / 1024.0 / 1024.0, 3));
    json!({"chunks": chunks, "chunk_bytes": chunk_bytes, "chunk_mib": mib, "n_chunks": n_chunks})
}

// -- h5py's chunk guess, for datasets chunked without an explicit shape ----------

const CHUNK_BASE: f64 = 16.0 * 1024.0;
const CHUNK_MIN: f64 = 8.0 * 1024.0;
const CHUNK_MAX: f64 = 1024.0 * 1024.0;

/// h5py's `guess_chunk`: a power-of-two fraction of each axis, sized from the
/// dataset's total size.  Used where 1.x passed `chunks=True`, so a dataset
/// is chunked the way it always was.
pub fn guess_chunk(shape: &[usize], typesize: usize) -> Vec<usize> {
    let mut chunks: Vec<f64> = shape.iter().map(|x| if *x == 0 { 1024.0 } else { *x as f64 }).collect();
    let ndims = chunks.len();
    if ndims == 0 {
        return Vec::new();
    }
    let product = |c: &[f64]| c.iter().product::<f64>();
    let dset_size = product(&chunks) * typesize as f64;
    let target = (CHUNK_BASE * 2f64.powf((dset_size / (1024.0 * 1024.0)).log10())).clamp(CHUNK_MIN, CHUNK_MAX);
    let mut idx = 0usize;
    loop {
        let chunk_bytes = product(&chunks) * typesize as f64;
        if (chunk_bytes < target || (chunk_bytes - target).abs() / target < 0.5) && chunk_bytes < CHUNK_MAX {
            break;
        }
        if product(&chunks) == 1.0 {
            break;
        }
        let axis = idx % ndims;
        chunks[axis] = (chunks[axis] / 2.0).ceil();
        idx += 1;
    }
    chunks.iter().map(|c| *c as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spatial_chunks_match_reference() {
        // medh5 1.4.4 on a host whose L3 clamps the budget to 4 MiB.
        let kinds: Vec<String> = vec!["spatial".into(); 3];
        let c =
            optimize_chunks(&[16, 24, 24], &kinds, &Patch::PerAxis(vec![8, 8, 8]), 2, Some(272_629_760), 0).unwrap();
        assert_eq!(c, vec![16, 8, 8]);
    }

    /// A file states any extents: a count of `2**62` chunks on each of two
    /// axes wrapped to 0.  A count past 64 bits is `null`.
    #[test]
    fn s14_1_a_chunk_count_past_the_integers_is_null() {
        let report = chunk_report(&[1 << 62, 1 << 62], &[1, 1], 1);
        assert_eq!(report["n_chunks"], Value::Null);
        assert_eq!(report["chunk_bytes"], json!(1));
        let report = chunk_report(&[4, 4], &[1 << 62, 1 << 62], 8);
        assert_eq!((report["n_chunks"].clone(), report["chunk_bytes"].clone()), (json!(1), Value::Null));
        assert_eq!(report["chunk_mib"], Value::Null);
        assert_eq!(chunk_report(&[64, 64], &[32, 16], 2)["n_chunks"], json!(8));
    }

    /// The ceiling holds from the start: the patch rounded up to a power of
    /// two was the first chunk, however large, and growth only added to it.
    #[test]
    fn s14_1_no_computed_chunk_is_past_the_ceiling() {
        let bytes = |c: &[usize], itemsize: usize| c.iter().product::<usize>() * itemsize;
        for (patch, itemsize) in [(96, 4), (96, 8), (160, 2), (160, 4), (256, 4), (512, 1), (64, 16)] {
            for l3 in [None, Some(256 << 20), Some(1)] {
                let c = spatial_chunk_for(&[600, 600, 600], &Patch::Uniform(patch), itemsize, l3).unwrap();
                assert!(bytes(&c, itemsize) as f64 <= MAX_CHUNK_BYTES, "{patch} {itemsize} {l3:?}: {c:?}");
            }
        }
        // 96³ float32 started at 128³, 8 MiB, and is the patch, which fits;
        // 160³ int16 started at 256³, 32 MiB, and its patch is halved once.
        assert_eq!(spatial_chunk_for(&[600; 3], &Patch::Uniform(96), 4, None).unwrap(), vec![96, 96, 96]);
        assert_eq!(spatial_chunk_for(&[600; 3], &Patch::Uniform(160), 2, None).unwrap(), vec![80, 160, 160]);
        assert_eq!(spatial_chunk_for(&[600; 3], &Patch::Uniform(96), 8, None).unwrap(), vec![48, 96, 96]);
        // A patch within the ceiling is the chunk it was.
        assert_eq!(spatial_chunk_for(&[600; 3], &Patch::Uniform(96), 2, Some(1)).unwrap(), vec![128, 128, 128]);
        // A voxel past the ceiling is one voxel, not a loop.
        assert_eq!(spatial_chunk_for(&[8, 8], &Patch::Uniform(4), 8 << 20, None).unwrap(), vec![1, 1]);
    }

    /// L3 is divided among the CPUs that share it.
    #[test]
    fn s14_1_l3_is_read_per_sharing_cpu() {
        assert_eq!(cache_size("33792K\n"), Some(33792 * 1024));
        assert_eq!(cache_size("32M"), Some(32 << 20));
        assert_eq!(cache_size("1048576"), Some(1 << 20));
        assert_eq!(cache_size("0K"), None);
        assert_eq!(cache_size("lots"), None);
        assert_eq!(cpu_list_len("0-3\n"), Some(4));
        assert_eq!(cpu_list_len("0-7,64-71"), Some(16));
        assert_eq!(cpu_list_len("5"), Some(1));
        assert_eq!(cpu_list_len("3-1"), None);
        assert_eq!(cpu_list_len(""), None);
        assert_eq!(per_sharing_cpu(33792 * 1024, Some(4)), 8448 * 1024);
        assert_eq!(per_sharing_cpu(1 << 20, None), 1 << 20);
        assert_eq!(per_sharing_cpu(1 << 20, Some(0)), 1 << 20);
        assert_eq!(cacheconfig_l3_sharing("16 2 2 16 0 0 0 0 0 0\n"), Some(16));
        assert_eq!(cacheconfig_l3_sharing("8 1 1"), None);
    }

    #[test]
    fn h5py_guess_chunk() {
        // Reference values from h5py 3.16's `guess_chunk`.
        assert_eq!(guess_chunk(&[1000], 8), vec![1000]);
        assert_eq!(guess_chunk(&[100, 100, 100], 4), vec![13, 25, 25]);
        assert_eq!(guess_chunk(&[7, 3, 2], 2), vec![7, 3, 2]);
        assert_eq!(guess_chunk(&[5_000_000], 1), vec![39063]);
        assert_eq!(guess_chunk(&[3, 160, 160, 160], 2), vec![1, 20, 20, 40]);
        assert_eq!(guess_chunk(&[41213], 1), vec![10304]);
        assert_eq!(guess_chunk(&[200, 64, 64], 8), vec![25, 8, 16]);
    }
}
