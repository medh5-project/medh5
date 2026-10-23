# medh5

The MEDH5 format engine.  A `.medh5` file is **one subject at one or more
timepoints**, with every image, annotation, transform and curation record about
them, in one self-describing HDF5 container.  This crate is the canonical
implementation of formats 1.0, 1.1 and 1.2 ([the specification], [1.1] for
the optional clinical profile, and [1.2] for open curves): the data model, HDF5 I/O, validation, chunked
access, compression, geometry and transforms, annotations, provenance,
integrity — and, for 1.1, clinical events on a subject clock, strict
prospective selection, and the [task and cache contract] training builds on.

It has two siblings, and all three read and write the same bytes:

- the **Python package** (`pip install medh5`), a thin layer over this crate
  that adds the NumPy, PyTorch, MONAI and converter integrations;
- the **`medh5` command line** ([`medh5-cli`]), a native binary over this crate.

HDF5 and C-Blosc2 are built from source and linked statically (see
[`medh5-sys`]), so a dependent crate needs no HDF5 installation.  For an MSVC
target, set `CFLAGS_x86_64_pc_windows_msvc=-DNDEBUG` (or the variable for your
target): without it HDF5 keeps its assertions, and `medh5-sys` stops the build.

## Write and read a sample

```rust
use medh5::array::NdArray;
use medh5::sample::{create, open_sample, GridOptions, ImageOptions};

fn main() -> medh5::Result<()> {
    let path = std::env::temp_dir().join("medh5-readme-case_0001.medh5");

    // A writer validates every call and commits atomically: a reader never
    // sees half a file, and a crash leaves the previous one intact.
    let ct = NdArray::from_vec(&[8, 16, 16], vec![0i16; 8 * 16 * 16])?;
    let mut w = create(&path, Some("case_0001"), Some("DEMO-0001"), "balanced", &[])?;
    w.add_grid("ct", &[8, 16, 16], &[2.0, 0.8, 0.8], GridOptions::default())?;
    w.add_image("CT", &ct, "ct", "CT", ImageOptions::default())?;
    let content_id = w.commit(true)?; // the Merkle root over every stored digest

    let sample = open_sample(&path)?;
    assert_eq!(sample.identity()?.subject_id, "DEMO-0001");
    assert_eq!(sample.content_id()?, content_id);
    let voxels = sample.image("CT")?.read(None, false, None)?;
    assert_eq!(voxels.shape(), [8, 16, 16]);
    assert!(sample.verify(None)?.ok());

    // The validator the writer is held to (§15); `strict` would also ask for a
    // de-identification record (W903).
    let report = medh5::validate::validate_file(&path, "integrity", None)?;
    assert!(report.ok(), "{:?}", report.codes());

    std::fs::remove_file(&path)?;
    Ok(())
}
```

## Where things are

| Module | Specification |
|---|---|
| [`sample`] | §2, §4, §14.4 — reading (`Sample`) and writing (`SampleWriter`, `create`, `amend`) |
| [`document`] | §2.4 — the sample document, everything in `/meta` |
| [`ids`] | §2.2, §2.3 — identifier rules |
| [`geometry`] | §3 — grids, affines, frames, multiscale |
| [`labels`] | §5 — label sets, hierarchies and ontology codes |
| [`annotations`] | §6–§9 — the voxel and geometric encodings, headers, coverage |
| [`transforms`] | §10 — affine, displacement, B-spline and composite transforms |
| [`curation`] | §11, §12 — identity, provenance, quality, timelines, splits, scrubbing |
| [`integrity`], [`digest`] | §13 — per-object digests and `content_id`; the hash algorithms and digest strings |
| [`storage`] | §14 — codec profiles, chunking, the sampling index |
| [`validate`], [`codes`] | §15 — the validator and its diagnostic code table |
| [`collection`] | §2.2 — packing samples into a `.medh5c` shard |
| [`convert`] | Appendix B — the 0.x migration, and the `ConversionReport` every importer fills |
| [`array`] | `NdArray`, the array every reader returns and every writer takes |
| [`dataset`], [`sampling`] | cohort manifests, splits, statistics; patch sampling |
| [`bench`] | reproducing the performance targets |
| [`rng`] | the generator behind sampling, the same stream in every frontend |
| [`error`] | the one [`Error`] type, and the code each error carries |
| [`parallel`] | how many threads one call may use for its work outside HDF5: `set_threads`, `threads` (`MEDH5_THREADS`, else 1) |
| [`conformance`] | the conformance corpus third-party implementations run |
| [`version`] | 1.1 §2 — which versions are read fully, read as a projection, or refused |
| [`clinical`] | 1.1 §3–§10 — the clinical profile: records, columns, checks, selection, augmentation |
| [`companion`] | task-cache-1 — task manifests, source pins, preflight row views, feature caches |
| `h5`, `json`, `numeric`, `pyval` | plumbing, outside semver (below) |

`h5`, `json`, `numeric` and `pyval` are the engine's plumbing rather than the
format, and from 2.1 they are outside semver: a minor release may change them.
They are HDF5 access below the high-level API, the format's exact JSON
serialisations, NumPy-compatible reductions and the 1.x value coercions. The
documented way to the canonical JSON of §5.1 is [`labels::canonical_json`]; an attribute value is
[`AttrValue`], re-exported at the crate root.

The crate's public signatures take and return types from three other crates,
so they are public dependencies: **hdf5-metno 0.15** (re-exported as `hdf5`),
**ndarray 0.17** (re-exported as `ndarray`) and **IndexMap 2**. A bump of any of
them is a major medh5 release.

Errors are one [`Error`] type whose diagnostic code, when it has one, is the
code the validator reports for the same defect (§15.2, [1.1] §11.2).

[the specification]: https://medh5.readthedocs.io/en/stable/spec/medh5-1.0/
[1.1]: https://medh5.readthedocs.io/en/stable/spec/medh5-1.1/
[1.2]: https://medh5.readthedocs.io/en/stable/spec/medh5-1.2/
[task and cache contract]: https://medh5.readthedocs.io/en/stable/spec/task-cache-1/
[`version`]: https://docs.rs/medh5/latest/medh5/version/
[`clinical`]: https://docs.rs/medh5/latest/medh5/clinical/
[`companion`]: https://docs.rs/medh5/latest/medh5/companion/
[`medh5-cli`]: https://crates.io/crates/medh5-cli
[`medh5-sys`]: https://crates.io/crates/medh5-sys
[`sample`]: https://docs.rs/medh5/latest/medh5/sample/
[`document`]: https://docs.rs/medh5/latest/medh5/document/
[`ids`]: https://docs.rs/medh5/latest/medh5/ids/
[`digest`]: https://docs.rs/medh5/latest/medh5/digest/
[`convert`]: https://docs.rs/medh5/latest/medh5/convert/
[`array`]: https://docs.rs/medh5/latest/medh5/array/
[`bench`]: https://docs.rs/medh5/latest/medh5/bench/
[`geometry`]: https://docs.rs/medh5/latest/medh5/geometry/
[`annotations`]: https://docs.rs/medh5/latest/medh5/annotations/
[`transforms`]: https://docs.rs/medh5/latest/medh5/transforms/
[`labels`]: https://docs.rs/medh5/latest/medh5/labels/
[`curation`]: https://docs.rs/medh5/latest/medh5/curation/
[`integrity`]: https://docs.rs/medh5/latest/medh5/integrity/
[`storage`]: https://docs.rs/medh5/latest/medh5/storage/
[`validate`]: https://docs.rs/medh5/latest/medh5/validate/
[`codes`]: https://docs.rs/medh5/latest/medh5/codes/
[`collection`]: https://docs.rs/medh5/latest/medh5/collection/
[`dataset`]: https://docs.rs/medh5/latest/medh5/dataset/
[`sampling`]: https://docs.rs/medh5/latest/medh5/sampling/
[`conformance`]: https://docs.rs/medh5/latest/medh5/conformance/
[`parallel`]: https://docs.rs/medh5/latest/medh5/parallel/
[`Error`]: https://docs.rs/medh5/latest/medh5/enum.Error.html
