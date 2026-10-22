# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [2.2.1] — 2026-10-22

**A patch release, with no format change.** Fixes for hostile input and one
performance regression, and nothing else: the documented API is unchanged, the
format stays 1.2 and the diagnostic code table keeps its codes, and what 2.2.0
wrote, 2.2.1 writes with the same `content_id`. A crafted file can no longer
hang a read inside HDF5, which the wheels and binaries now build with a fix;
arithmetic on what a file states is exact, saturated or refused, never wrapped;
and the ITK transform reader is fuzzed with the rest. Opening samples from
several threads on the free-threaded build is back to 2.1.0's time. The
performance guide is measured on 2.2, and only the newest 2.2.x receives fixes.

### Security

- **A crafted file no longer hangs opening or validating it.** One damaged
  structure inside HDF5 made the library loop without end on the first read
  of a string, so `medh5 validate`, opening the file from Python, and every
  other read never returned. HDF5 is now built with a fix
  (`vendor/hdf5-metno-src/PATCHES.md`), and the file is reported as
  unreadable (`E001`). The wheels and the release binaries carry it; a build
  from crates.io does not until HDF5 releases the fix (`SECURITY.md`).

### Fixed

- **Arithmetic on what a file states no longer overflows.** A box from one
  end of the integers to the other, extents past 64 bits, a link span of
  `2**64 - 1` and counts summed past 64 bits panicked in a debug build and
  wrapped in a release one: the layers overlap check took four layers on a
  grid of `2**62` voxels for empty and passed them unread, a box-only
  object counted -1 voxels, and an index on a grid wider than `2**31` voxels
  wrapped its coordinates. Each is now exact, saturated where that is still
  the answer, or refused through the code that already covers the field.
- **An ITK B-spline whose lattice overflows is refused.** NumPy's product of
  `2**32` control points an axis wrapped to 0, and an empty parameter list
  matched it.
- **Writing an ITK HDF5 transform file refuses a type name holding a NUL.**
  HDF5 ends a stored string at its first NUL, so the file kept a shorter name
  than the one given, and read back as another transform type.

### Performance

- **Opening samples from several threads no longer slows down on the
  free-threaded build.** In 2.2.0, four threads opening samples on Python
  3.14t took 3.5 times as long as one thread, twice 2.1.0's time. Two
  changes fix it. A value read on first use (a grid, an image's metadata, a
  document) is now read under one take of HDF5's lock, and a thread no longer
  waits on the value while another thread waits for the lock. An open also
  takes the lock once for all of its calls. 256 opens from four threads now
  take 0.25 s instead of 0.52 s, on 2.1.0's level, and 0.19 s instead of
  0.26 s on the GIL build. One thread is unchanged.

### Documentation

- **Only the newest 2.2.x receives fixes.** `SECURITY.md` gains "Supported
  versions": the 2.0.x and 2.1.x lines are closed, and nothing is backported
  to them. The weekly advisory audit reads `main` and `release/2.2` in place
  of `release/2.0`.
- **The performance guide's numbers are 2.2.0's, measured beside 2.1.0's on
  one machine.** 2.2.0 is within the run-to-run spread of 2.1.0 on every
  `medh5 bench` row. A new example, `docs/examples/bench_threads.py`, times
  the same reads from one, two and four threads, and the guide gives its
  table. It found the free-threaded build's slow opens that this release
  fixes (Performance).

## [2.2.0] — 2026-10-20

**A minor release, and format 1.2.** 1.2 adds one annotation kind,
`polylines`, for open curves, and nothing else: a sample is written at 1.2
only when it holds one, a 1.1 reader reads it as a projection, and the
diagnostic code table is unchanged. What 2.1.0 wrote, 2.2.0 writes with the
same `content_id`, but for the imports a fix changed (Fixed). The Python and
Rust APIs are additive. macOS on Intel is no longer supported; Python 3.10
still is.

**One NIfTI converter for three frontends, and more of DICOM and ITK.** NIfTI
is read and written in the engine, held to a corpus recorded from 2.1.0, so
nibabel is no longer needed and the standalone binary converts NIfTI with no
Python. Enhanced and legacy-converted multi-frame CT, MR and PET import;
RTSTRUCT's open contours import and export as polylines; ITK's HDF5 transform
files and its B-spline transforms are read and written.

**Resampling on the record, de-identification and clinical import at scale.**
`medh5 resample` puts images and their annotations onto another grid as a
recorded `resample` derivation; converters and loaders still refuse to
resample, and now name the command. `medh5 scrub` scrubs a collection member
by member and redacts clinical text with its spans kept valid, and a whole
MEDS dataset or FHIR bulk export imports into a cohort in one pass.

**Free-threaded Python and more platforms.** The free-threaded CPython 3.14
wheel runs without the GIL, and a writer or document that a second thread uses
at once refuses it rather than race. Alpine and other musl Linux get wheels and
a static binary. A string stated longer than its file is refused before it is
read, and the C libraries are fuzzed under AddressSanitizer.

### Security

- **A string stated longer than its file is refused before it is read.** A
  variable-length string's stored length is the file's to state, up to
  4 GiB, and a small crafted file could make a read set aside that much
  memory before failing. A contiguous string dataset — `/meta` among them —
  is now checked first, and a length longer than the file is reported as
  unreadable (`E001`) at once.

### Added

- **`SampleWriter.add_polylines` writes open curves, and
  `PolylinesAnnotation` reads them.** Each `Polyline` is ordered vertices
  with a class, an optional grid plane (`plane=(axis, index)`) and, through
  `instance_ids=`, an instance; `polyline(i)`, `polylines()`, `offsets` and
  `planes` read them back, in index or world space. A polyline of fewer than
  two vertices is `E404`, offsets that do not delimit the vertices `E408`, and
  a polylines annotation satisfies neither `seg` nor `det`, whatever its
  `task`. A curve's instance joins its object's track as presence, as a
  point's does, and `W909` reads its class. In Rust,
  `annotations::Polyline`, `encode_polylines` and
  `SampleWriter::add_polylines`; `version::required_for_content` and
  `written_version_for_content` give the version a content needs.
- **Annotations follow a resample.** `medh5 resample` and
  `medh5.transforms.resample` carry the annotations on the source grids onto
  the target, each naming its source in `derived_from` and keeping its classes
  and coverage. Voxel encodings move nearest in their own encoding, instances
  object by object under their own ids (an object with no voxel left is
  refused, never dropped), probability maps linearly; points and keypoints
  map exactly, by the inverse of the path the image is sampled through, and
  meshes along an affine path. A box is carried only where the
  mapping keeps the axes and a contour only where its plane stays a grid
  plane; anything else is refused rather than approximated. Target voxels
  outside the source become an ignore region (§7.7). `--no-annotations`
  (`annotations=False`) moves the images alone, and the report lists what was
  carried (`ResampledAnnotation`) and what was not (`not_carried`).
- **Images resample onto another grid as a recorded derivation.**
  `medh5.transforms.resample(path, image_ids, onto=…)` and
  `medh5 resample PATH --image ID --onto GRID` map images onto an existing
  grid: through the affines when the two grids share a comparable frame, and
  through the sample's transforms otherwise (§10.2), evaluated at the target's
  voxels. `spacing=` with `grid_id=` (`--spacing`, `--grid-id`) makes a new
  grid in the image's frame that keeps its voxel edges. A pair no transform
  relates, a frame that cannot be compared (`W917`) and a grid naming no frame
  are refused rather than assumed aligned. Interpolation is `linear`,
  `nearest` or `cubic`; voxels outside the source take `fill` and the output's
  `valid_mask` marks the rest. One `resample` activity records the inputs, the
  target, the transform path and the interpolation, and every object written
  names it. The work goes a target chunk at a time, so memory follows the
  chunk, not the volume. The refusals that send you to resample now name the
  command: the torch datasets' patch across two grids, the MONAI reader's
  channel stack across two grids, and agreement between annotations on
  different grids (`E101`) each say `medh5 resample PATH --image ID --onto
  GRID`; the NIfTI import's volumes on two grids, its masks drawn on another
  grid, and the nnU-Net export's labels off the channels' grid name it too.
- **Format 1.2 adds one annotation kind, `polylines`: open curves.** Vessel
  centrelines, catheter and applicator paths, needle tracks and RTSTRUCT's
  open contours had no kind to go into: `contours` are closed polygons, and
  `points` are unordered. A sample is written at 1.2 only when it holds a
  polylines annotation, so imaging alone stays 1.0 and the clinical profile
  1.1; removing the last one lowers the version again. A 1.1 reader reads a
  1.2 file as its projection, the curves left out as `W913`, and its digests
  and `content_id` verify. No diagnostic code is added; `E408`'s summary names
  polyline offsets. The corpus has 175 cases: `shapes-polylines`,
  `E408-polyline-offsets` and `E404-polyline-one-vertex` are new, and the
  higher-minor case declares 1.3. The specification is
  `docs/spec/medh5-1.2.md`.
- **`medh5 scrub` scans and scrubs a collection, member by member.** A
  `.medh5c` is scanned into one report whose findings name their member
  (`KEY::where`); `--apply` writes a new collection at `--out`, or under
  its own name in `--out-dir`, one of which it requires, and only reads the
  source. Each member is scrubbed with the one salt, so
  pseudonyms, re-minted ids and a range's per-subject shift agree across
  members; under `--pseudonymise-ids` a member is keyed by its pseudonym. A
  member that refuses leaves nothing written, or with `--skip-refused`
  (`skip_refused=True`) is left out and listed.
- **`medh5 scrub --apply --redact-text` rewrites the names, dates, UIDs and
  sample ids it finds in clinical text.** Each becomes `[NAME]`, `[DATE]` or
  `[ID]` in a new document owned by a new version of its event, with every
  span into it moved through the edit; a span that overlaps a redacted range
  refuses the run (1.1 §6, §10). Python's `scrub.apply` takes
  `redact_text=True`. The redaction is recorded with the digests it wrote,
  and the new text still needs `--attest-text`.
- **`augment_cohort_from_fhir` imports a FHIR Bulk Data export into a
  cohort's samples.** The export's NDJSON files are streamed and grouped by
  subject reference, one subject in memory at a time (where each line is, is
  indexed in a temporary file), and each subject is mapped exactly as
  `read_fhir` maps a Bundle of its Observations and DiagnosticReports.
  Resources of unmapped subjects, and resource types the profile does not
  read, are counted. On the command line, `medh5 clinical augment-cohort
  --from-fhir-bulk DIR --mapping MAP.csv`.
- **`augment_cohort_from_meds` imports a whole MEDS cohort into its
  samples.** A mapping from `subject_id` to sample says where each subject's
  rows go, in place or into `out_dir`; each subject is mapped exactly as
  `augment_from_meds` maps it alone. A subject that refuses is listed with
  its reason and the others are written; subjects the mapping does not name
  are counted. On the command line, `medh5 clinical augment-cohort
  --from-meds DIR --mapping MAP.csv`, with `clinical augment`'s options.
- **ITK B-spline transforms are imported and exported.** A
  `BSplineTransform` of order 1 or 3, alone or applied first in a composite,
  is stored as 1.0 §10.5's `bspline` on a grid of its lattice, and maps every
  point of its domain as SimpleITK does; `to_itk_transform` writes a `bspline`
  back as one. Other orders and ITKv3's `BSplineDeformableTransform` are
  refused, each with its reason. B-splines were refused before.
- **ITK's HDF5 transform files are read and written.** `read_itk_transform`,
  `from_itk_transform` and `medh5 convert from-itk-transform` take an `.h5`
  (`.hdf5`, `.hdf`) file as ITK writes one, a composite flattened in reverse
  as for `.tfm`, and `to_itk_transform` and `write_itk_transform` write one;
  through the engine's HDF5, so the converters still need no `h5py`. They
  were refused before. Parameters under the misspelling earlier ITK
  releases wrote (`TranformParameters`) are read, as ITK reads them.
- **RTSTRUCT open curves import and export as polylines.**
  `from_rtstruct(open_curves="polylines")` (`--open-curves polylines`)
  imports `OPEN_PLANAR` and `OPEN_NONPLANAR` ROIs as a polylines annotation,
  `<id>_curves` unless `curves_id=` (`--curves-id`) names another, each
  `OPEN_PLANAR` curve with the grid plane it lies in; the report records that
  the sample is now written at format 1.2. `to_rtstruct(curves=...)`
  (`--curves ANN`) writes one back as `OPEN_PLANAR` or `OPEN_NONPLANAR`
  contours. `omit` stays the default, and its warning now names the option.
- **The engine writes NIfTI and adds NIfTI masks to a sample.**
  `medh5::convert::nifti` gains `to_nifti`, `write_nifti`, `for_export`,
  `write_field` and `import_seg_nifti`, 2.1.0's converter ported rule for
  rule and held to the same parity corpus as the reader: every one of its 348
  cases gives 2.1.0's objects, `/meta`, report and refusal.
- **The engine reads and imports NIfTI itself.** `medh5::convert::nifti`
  holds `read_nifti` and `from_nifti`, 2.1.0's converter ported rule for
  rule: NIfTI-1 and NIfTI-2 in either byte order, `.nii`, `.nii.gz` and
  `.hdr`/`.img`, with the same geometry, notes, objects and refusals, which
  a parity corpus recorded under 2.1.0 holds it to. gzip is inflated in Rust,
  never past the bytes the header implies, so a small file cannot make the
  reader decompress or allocate more; zlib's gzip file interface stays
  unlinked.
- **Enhanced multi-frame DICOM imports.** Enhanced and legacy-converted CT, MR
  and PET objects, skipped with a warning until now, are split into stacks —
  by `StackID` and the dimensions of `DimensionIndexSequence` that tell stacks
  apart, such as a time point, a cardiac phase or a flip angle — and each
  stack imports as the classic series it would have been: the same grid and
  the same values, placed by the frames' functional groups and decoded one
  frame at a time. Per-frame rescales follow `per_slice_rescale`, acquisition
  attributes come from the shared and per-frame groups, and a concatenation is
  joined into one object. A frame with no position, an irregular stack, a
  stack whose `InStackPositionNumber` its positions contradict or that skips a
  number, a concatenation missing a part, a part with no `ConcatenationUID`,
  an instance whose `NumberOfFrames` its per-frame groups contradict, one
  instance at two paths and a frame whose geometry is not numbers are
  refused with the reason. `Series` gains `frames` and `stack`.
- **The standalone `medh5` binary converts NIfTI itself.** `convert
  from-nifti`, `import-seg-nifti` and `to-nifti` run in the engine, with the
  checks, messages and reports the Python package gave them, so they work
  with no Python installed. The other converters are still handed to Python.
  An error the package let escape as a traceback, such as a header NIfTI
  cannot hold, is now `medh5: <why>` with exit code 1.
- **medh5 runs on free-threaded CPython without re-enabling the GIL.** The
  cp314t wheel declares the extension safe to run with no GIL, so
  `import medh5` leaves it off; 2.1.0 asked for it, and CPython turned it on
  with a warning. The test suite runs on 3.14t with the GIL off, many threads
  at a time.
- **Alpine and other musl Linux install a wheel.** x86_64 and aarch64 each get
  a `musllinux_1_2` abi3 wheel and a free-threaded 3.14 one, so medh5 no
  longer builds from source there. Each release also has a `medh5` binary
  for `x86_64-unknown-linux-musl` and `aarch64-unknown-linux-musl` that links
  musl statically and so runs on any Linux of its architecture.
- **The package's licence names Unicode-3.0.** `unicode-normalization` 0.1.26
  compiles in Unicode's data tables, under the Unicode License v3, so the
  wheels' `License-Expression`, the Homebrew formula and `cargo deny`'s
  allowed licences add `Unicode-3.0`, and `THIRD_PARTY_NOTICES` reproduces its
  text for that crate.

### Behaviour changes

- **A writer, document, payload, provenance, cache writer, staged file or
  patch draw used by a second thread at once raises `MEDH5Error`.** The
  second call is refused, says the object "is used by one thread at a time",
  and changes nothing; it raised `RuntimeError: Already borrowed` before.
  Calls that only read share the object: they are refused only while another
  thread is changing it.
- **NIfTI conversion no longer needs nibabel.** `medh5.io.nifti`, the ITK
  displacement fields of `medh5.io.itk` and the nnU-Net exporter read and
  write NIfTI through the engine, with their signatures unchanged. nibabel is
  neither installed nor imported; the `nifti` extra is kept, empty, so
  `pip install "medh5[nifti]"` still works, and `require_nibabel` says medh5
  no longer needs it rather than naming that extra.
- **A header NIfTI cannot hold is a `ValueError`.** It was nibabel's
  `HeaderDataError`, a `ValueError` subclass, with the same message.
- **A `.nii.bz2` or `.nii.zst` file is refused.** nibabel read them through
  Python's codecs; the engine inflates gzip only, and says to decompress the
  file first.
- **`convert to-nifti` keeps its overlap warning in the report.** Flattening
  overlapping classes into one labelmap was a Python `UserWarning` on stderr
  from the command line; it is now a warning in the report, as the other
  notes are. `to_nifti` in Python still warns, as it did.
- **macOS on Intel (x86_64) is no longer supported.** 2.2.0 publishes no
  macOS x86_64 wheel or binary, and the Homebrew formula refuses to install on
  an Intel Mac. On one, pip builds medh5 from its source package (Rust, a C
  compiler and CMake), which is no longer tested there; or stay on 2.1. Apple
  silicon is unchanged.

### Fixed

- **A box whose finite bounds lie past the integers no longer overflows the
  validator.** An `instances` object's extent was its box's two slice ends
  subtracted, and a box from one end of the `i64` range to the other
  overflowed: a build with overflow checks panicked (the `open_validate`
  fuzzer found it), and a release build compared the crop with a negative
  extent. The extent now saturates, and the crop is reported as `E405`.
- **A scaled complex-valued NIfTI mask keeps its imaginary voxels.** Under
  `scl_slope`, 2.1.0 tested only each voxel's real part, so a voxel such as
  `0+1i` was dropped from the mask; a voxel is now in the mask when either
  physical part is non-zero, as it already was unscaled.
- **Frame times one volume states bind the volumes after it.** When the
  first volume assumed its frame indices as times and a second stated those
  same times, a third stating other times replaced them, putting the second
  on a timeline it disagreed with; it is now refused, as two volumes stating
  different times always were.
- **`write_nifti` writes a fourth axis of no frames given its empty
  `time_values`.** 2.1.0 failed reading the first of no frame times
  (`IndexError`); there is no first time to state, so the header states none.
- **A CT's units follow its `RescaleType`.** Every CT was written as
  quantitative HU, though DICOM lets a derived or multi-energy CT rescale to
  another quantity. `value_units` is now the stated `RescaleType`. An omitted
  type is `HU` in a classic CT and in an original, non-localizer,
  single-energy enhanced frame, and unknown elsewhere. `US` or an unknown type
  gives `intensity` with no units, and a series whose slices state different
  types is refused. The `value_scale` decision names the units
  `read(physical=True)` returns, where it said HU for every image.
- **Acquisition attributes are recorded only where the slices agree.** A
  series recorded its first slice's `XRayTubeCurrent`, `ExposureTime` or
  `EchoTime` for the whole volume, even where the slices stated different
  ones. An attribute the slices disagree on is now left out, with a warning
  in the report.
- **A forked child takes over every handle cache before its first call.** An
  `os.register_at_fork` hook abandons the parent's handles and gives each
  cache a new lock as the child starts: two threads of a child that both
  noticed the fork replaced the cache's lock one after the other, and then
  held different locks over the same handles. Registering a vocabulary
  (`medh5.labels.register`) and editing a document's live views from several
  threads are locked too, and no code of the caller's runs under that lock
  (an index's `__index__`, an item's `__eq__`, a key's `__hash__`). A key
  JSON cannot spell is refused as it is given, leaving the member editable.
  `medh5.labels.describe()` no longer raises E305 for a vocabulary another
  thread unregisters meanwhile, and a document's label set and timeline are
  read with every class and timepoint as it stood at one moment.
- **A thread waiting for HDF5 on free-threaded CPython lets go of the
  interpreter.** Waiting attached for another thread's HDF5 call held up every
  stop-the-world pause, a garbage collection among them, until that call ended.
  A read or write through medh5 now runs detached throughout. Closing a sample
  no longer holds up a thread asking for it, and a lock left by a panic is taken
  as it stands instead of raising again. A payload's dicts, edited by another
  thread while it is written or described, no longer panic the read.
- **A forked child no longer hangs at its first HDF5 call.** A child forked
  while another thread was in the middle of a read inherited HDF5's lock held
  by a thread it does not have. A fork now waits for the call to end, and the
  child takes the lock over cleanly.

### Performance

- **A MEDS cohort is read once, not once per sample.** The dataset is opened
  once and read a Parquet row group at a time: its subject column, then the
  mapped subjects' rows. A subject is imported as soon as its last row is
  read, so a dataset in MEDS order holds one row group and one subject in
  memory at a time.

### Documentation

- **A guide to resampling.** "Resample onto another grid" covers when to
  resample and when not to, onto a grid or onto a spacing, what happens to the
  annotations, what the `resample` activity records, why the loaders never
  resample, and when MONAI's `Spacingd` is enough. The NIfTI refusals and the
  NIfTI import guide link to it.

## [2.1.0] — 2026-10-15

**A minor release, with no new format version.** Files are still written at
1.0 or 1.1, and what 2.0.3 wrote, 2.1.0 writes with the same `content_id`, but
for the imports a fix changed (Behaviour changes). The specification changes by
errata only, and the diagnostic code table gains a second revision, W915–W919:
`validate --level strict` can fail a file 2.0 passed, and `--codes-revision 1`
keeps 2.0's verdicts. The Python and Rust APIs are additive; what a caller can
notice is a silently wrong answer fixed, or a deprecation that warns until 3.0.

**Answers that were silently wrong, and integrity that cannot be talked past.**
An amendment refuses a file edited outside medh5 rather than attest the edit,
and refuses to replace one that changed after it read it; tracking no longer
calls a lesion resolved where a follow-up never imaged it; `transform_between`
no longer answers from whichever grid sorted first; two same-named nested
RTSTRUCT ROIs no longer import as a hole; and MONAI can batch medh5 items.

**Clinical data in and out, training at scale, converters that say what they
did.** Clinical histories import from MEDS and FHIR, and a clinical sample can
be de-identified; a task can be drawn from a cohort and its split, and
repinned. Training gains DDP sharding, sliding-window stitching, a MONAI reader
and per-class ignore masks, the batch commands gain `--workers`, and
free-threaded CPython 3.14 gets a wheel. The converters gain FRACTIONAL SEG,
codes and laterality in SEG, per-slice PET rescale, RTSTRUCT points, nnU-Net
test cases and ITK transforms. Every release artifact is attested, and the
crates are published through trusted publishing.

### Security

- **A name is no longer pseudonymised without a salt.** `--profile strict`
  replaced a person or organization agent's name with an unsalted hash, which
  anyone reverses by hashing candidate names; the run is now refused without
  `--salt`. An unsalted run that pseudonymised UIDs reports
  `unsalted_pseudonym`; medh5 3.0 requires a salt.
- **A tool's timestamps no longer date the visit.** Provenance `started` and
  `ended`, and the file's `created`, were passed over, though a scan run the
  day of the visit gives its date. `--profile strict` cuts them to their year
  (`tool_timestamp`), and `basic` reports them once the dates are shifted.
- **The amount of a date shift can be withheld (D14).** A record stating
  `date_shift_days` lets one known date restore every other.
  `--withhold-shift` (`withhold_shift=True`) leaves it out and records in the
  de-identification activity that the dates were shifted, with a salted
  commitment a re-run checks. `--profile strict` withholds by default; `basic`
  still records it and warns (`FutureWarning`) that medh5 3.0 will withhold
  it.
- **A task's sources can be confined to one directory.** Whoever wrote a
  manifest chose every path its runner opened, and `T301` telling an absent
  path from an unreadable one answered whether a path exists. `--source-root
  DIR` on `task preflight`, `task reconcile` and `cache validate`, and
  `confine=True` (the manifest's directory) or `confine=DIR` in Python, refuse
  a URI outside the directory, its `.` and `..` resolved as text, without
  examining the path (`T301`, naming the URI only). Lexically, so a symbolic
  link inside the directory still works; in Rust, `SourceAccess` and the
  `*_in` functions taking it.
- **The crates are published through crates.io's trusted publishing.** The
  release held a long-lived `CARGO_REGISTRY_TOKEN` and verified one crate at a
  time, so a failure verifying `medh5-cli` left `medh5-sys` and `medh5`
  published. Every crate is now verified first, with no token in existence,
  and the upload uses a short-lived token issued to the release workflow.
- **Every release artifact carries a build attestation.** The binaries'
  archives on the release page had only `SHA256SUMS`, which says what was
  uploaded, not who built it; each archive, wheel and sdist now has a signed
  statement that the release workflow built it from the tagged commit,
  checked with `gh attestation verify FILE --repo medh5-project/medh5`.

### Added

- **An image or a grid can be removed, and an image's attributes edited,
  without rewriting the sample.** `remove_image` and `remove_grid` join
  `remove_annotation` and `remove_transform`; removing an image drops its
  `acquisition` entry and its `series_uids` mentions too. `set_image_attrs`
  changes `modality`, `value_type`, `value_units`, `channel_names`, the
  rescale, the window, `valid_mask` or `prov` in place, `None` removing one,
  where correcting a modality used to mean writing the pixels again. Each is
  checked as `add_image` checks it, and the commit refuses what still names
  a removed object (E101, E503) or leaves no image (E201) or grid (E111).
- **`add_mask` returns the group it wrote**, as every other `add_*` does. It
  returned `None`.
- **A missing, unreadable or directory path raises the matching `OSError`
  subclass.** `MEDH5FileNotFoundError`, `MEDH5PermissionError` and
  `MEDH5IsADirectoryError` are each a `MEDH5FileError` and the built-in of
  their name, and every `MEDH5FileError` carries `errno` and `filename`, so
  `except FileNotFoundError` catches a missing sample. `str()` is unchanged.
- **An HDF5 read failure raises `MEDH5IOError`.** It is an `OSError`, as the
  plain `OSError` it replaces was, and not yet a `MEDH5Error`. An engine
  `OSError` that names an errno now carries it, so it is the subclass of that
  errno.
- **`PatchSampler.draws` and `iter_patches` take `grid=`, as `draw` does.** A
  run of draws can now be pinned to one grid, such as one visit's, whether or
  not an annotation lives there.
- **`medh5.geometry.COORD_SYSTEMS`, the coordinate systems the converters
  map.** It is `("LPS", "RAS")`, the engine's tuple, which the NIfTI, DICOM
  SEG, RTSTRUCT and MONAI layers now share instead of each spelling it.
- **A displacement field can be stored in `float16`.** `add_transform` takes
  `field_dtype=`: `"float32"` (the default, as before), `"float16"`, which
  refuses a component `float16` cannot hold, or `"auto"`, which stores
  `float16` when every displacement is finite, within `float16`'s range and
  below the 64 voxels §10.3 recommends it for, and `float32` otherwise. The
  field takes half the space.
- **The inverse of a chain or a composite is its steps' inverses, in
  reverse.** `inverse()` of a `ChainTransform`, or of a stored composite
  without a usable `inverse_id`, returns that chain when every step can be
  inverted, where it returned `None`, and `InverseTransform.can_invert` says
  so. `is_invertible` stays the file's claim, and the frame graph is unchanged.
- **Each conformance case states its format version, and a score can be
  bounded by version and by code revision.** `expected.json` gives each case
  a `format_version`, the latest version among the codes it expects, which is
  not always what its file declares. It also gives the suite's
  `codes_revision`. `medh5 conformance score --format-version 1.0` scores a
  1.0-only implementation on the cases addressed to 1.0.
  `--codes-revision 1` scores one written against the first revision of the
  code table without the codes later revisions allocated. Either way the
  cases left out are counted as skipped. The same bounds are available as
  `medh5.conformance.score(..., format_version=, codes_revision=)` and Rust's
  `score_with` with `ScoreOptions`. Unbounded, a third party's score fell
  with every case added by an erratum it was never told about.
- **The diagnostic code table is revisioned, and a validator can pin a
  revision.** Each code names the format version whose rules it reports
  (`since`) and the revision of the table that allocated it (`revision`);
  `Code` gains both, and `medh5.errors.CODES_REVISION` is the table's own.
  `medh5 validate --codes-revision N`, `validate_file(..., codes_revision=N)`
  and the engine's `validate_file_with` report only the codes revision `N`
  allocated, so a `--level strict` gate does not move when a later revision
  adds a warning (1.0 §15.1, §15.2, §16). The conformance suite's
  `codes.json` carries both members and the table's revision.
- **The conformance suite builds to the same bytes every time.** Two
  publishes a few minutes apart differed in every case file, with the same
  `content_id`s: each sample recorded when it was built, and every group
  HDF5's creation and modification times. The corpus now pins `created`, and
  the engine creates every group, the root included, without timestamps, as
  it already wrote datasets. A mirror can check a published suite by
  publishing it again and comparing `SHA256SUMS`. Any file the engine writes
  differs from another of the same content only in `created`; no digest or
  `content_id` changes.
- **Agreement compares two vocabularies.** `compare`, `compare_voxel` and
  `compare_instances` take `map_classes`: `"id"` (the default), `"key"` to
  score classes of one key across two inline label sets (a label set by
  reference is refused, and one key with two codes of one ontology is
  `C102`), or an explicit one-to-one map of `a`'s classes to `b`'s. A class
  `b` lacks is listed in `skipped` with its reason, never scored 0, and each
  result's `b_class_ids` names the class of `b` every class was scored
  against. `medh5 agree` takes `--map-classes id|key` and `--map-class A=B`.
  In Rust, `curation::agreement::compare_with` takes `CompareOptions` and
  returns a `Compared`.
- **Objects pair the way you ask.** `compare` and `compare_instances` take
  `match_by`: `"auto"` (the default), `"iou"` to pair by where the objects
  are even where both carry ids, or `"instance_id"`, which refuses two
  samples because an id names an object only within its own sample (§7.4).
  `medh5 agree` takes `--match-by auto|iou|instance-id`.
- **Tracking says why each absence is, or is not, a finding.**
  `Tracking.reason_at(instance_id, timepoint)` and `reasons(instance_id)`
  give one of `medh5.curation.REASONS`: `located` or `class_examined` where
  the object is `resolved`, `outside_fov`, `ignored`, `unrelated`,
  `no_identity` or `not_examined` where it is `unexamined`. `is_new` and
  `is_resolved` take `located=True` to count only located absences; the
  JSON of `Sample.tracks()` and `medh5 track --json` gives each track its
  `reasons`, and `medh5 track` prints a `reasons` column. In Rust,
  `curation::tracking::locate_tracks` returns a `LocatedTracking`.
- **`scrub --apply` de-identifies a clinical sample, as a new sample (D15).**
  The scan reads the clock, the ids and each document's text in turn
  (`clinical_clock`, `clinical_id`, `clinical_text`); `--apply` moves a `utc`
  clock's times by the one shift, makes it `shifted_utc`, and re-mints every
  id from the salt, links following, so a cutoff moved by the shift admits
  the same versions. Text is never rewritten: `--attest-text TOOL`
  (`attest_text=`) records who de-identified it, by the digests of its bytes,
  and lapses when they change. Unattested text, or a name, UID or record
  number in it, still refuses the apply.
- **`ScrubOptions` holds every scrub option, checked once for every door.**
  `ScrubOptions::check` refuses an apply-only option without `--apply`, a
  pseudonym or a range without a salt, and a shift of 0, in one message the
  command line and the Python API share; `apply_with` and `scrub_each` take
  it. `--pseudonymise-ids` is no longer tied to `--apply` by the grammar, so
  its refusal says why.
- **`--date-shift-range MIN:MAX` gives each subject its own shift,** taken
  from the salt and the subject id, the same for every file of one subject,
  and always withheld (`date_shift_range=(min, max)`).
- **`--out FILE` and `--out-dir DIR` scrub to a copy** and leave the source
  as it is (`out=`, `out_dir=`); a copy never replaces an existing file.
- **Each cohort check code has one severity, in `CHECK_CODE_SEVERITY`.**
  Python's `medh5.dataset.CHECK_CODE_SEVERITY` and Rust's `CHECK_SEVERITY`
  and `code_severity` give every `C` code's severity; `C401` is the one a
  flag changes, an error under `dataset check --deep`.
- **`ClassStats.examined_prevalence` is `None` for a class no sample
  examined.** Rust's `ClassStats::examined_prevalence` returns
  `Option<f64>`.
- **`medh5.labels.check(paths)` reports which files share a vocabulary.** It
  returns what `medh5 labels check --json` prints; Rust has
  `check_vocabularies`, `vocabulary_of` and `vocabulary_key`.
- **`medh5 dataset upgrade` rewrites a 2.0 manifest as `medh5.dataset/1`.**
  `--base` names the directory its paths are relative to; it refuses, writing
  nothing, when an entry's file is missing there. `Manifest.save` takes
  `paths="keep" | "relative" | "absolute"`, `Manifest.load` takes `base=`, and
  `medh5 dataset index` takes `--absolute`.
- **`medh5.split/2` deals a split so that adding a group moves few others.**
  `make_splits(..., algorithm="medh5.split/2")` (`--algorithm 2`) fixes each
  partition's total by largest remainder and each stratum's counts by
  controlled rounding, then fills them in hash-rank order: one group added to
  115 moves a median of 1 other where `medh5.split/1` moved 19, and strata of
  3, 5 and 7 groups at 70/15/15 get a test group. The split JSON records
  `algorithm`; one without it reads as `medh5.split/1`.
- **`make_splits(..., previous=…)` continues a split without moving any
  group.** New groups fill each stratum's shortfall by `medh5.split/2`, the
  JSON records the lineage under `previous`, and `Split.moved_from(other)`
  lists what two splits place differently (`--previous` on the command line).
- **A FHIR R4 importer.** `medh5.io.fhir.read_fhir(source, subject=...,
  clock=...)` reads one patient's Observations and DiagnosticReports from a
  Bundle as clinical records, and `augment_from_fhir` adds them to a sample.
  A date's precision becomes its bounds, `issued` the availability,
  `meta.versionId` a chain of revisions linked by `supersedes`, a comparator
  `lt`, `le`, `gt` or `ge`, and `dataAbsentReason` the `missing_reason`; each
  component is an event of its own. A resource with no coding is left out
  rather than coded from its text, and versions that cannot be ordered are
  refused. Report texts and imaging-study links are imported only when asked.
  Standard library only.
- **A MEDS importer.** `medh5.io.meds.read_meds(source, subject_id=...,
  clock=...)` reads one subject's rows of a MEDS 0.3 dataset as clinical
  records, and `augment_from_meds` adds them to a sample. It invents no zone,
  origin, unit, code or availability: a time without a zone on a UTC clock,
  a relative clock without `origin` and a row holding both values are
  refused, availability stays null unless a column holds it, and a unit comes
  only from a declared column. Ids are hashes of the rows, so a
  re-extraction keeps them and a second import is refused. `MEDS_BIRTH` and
  free text are left out unless asked. Needs `pip install "medh5[meds]"`
  (pyarrow 14 or later).
- **Every documented converter option has a flag on `medh5 convert`.**
  `--annotated-classes all_given|all|K1,K2` (from-nifti, from-dicom-seg,
  from-rtstruct, import-seg-nifti), `--codec PROFILE` (from-nifti, from-dicom,
  from-nnunet and `medh5 migrate`, which wrote `balanced` whatever it was
  asked), `--class` and `--series-description` (to-dicom-seg), `--label`
  (to-rtstruct), `--mask-id` (from-rtstruct, with `--rasterize`),
  `--coord-system` (from-nnunet), `--file-ending` (to-nnunet), `--value-units`
  (from-nifti), and `--report`/`--json` on to-nifti, which still prints the
  file it wrote unless `--json` asks for the report.
- **`medh5 convert import-seg-nifti` adds NIfTI masks to a sample.**
  `import_seg_nifti` had no command; it takes `--mask NAME=PATH` (repeatable),
  `--id`, `--grid`, `--assume-aligned`, `--assume-geometry`,
  `--plane-as-volume` and `--annotated-classes`, and amends the sample in
  place as `from-dicom-seg` does.
- **`from_nifti(label_set=)` fixes a cohort's class ids** (`--label-set
  FILE`). Masks are matched by key to a `LabelSet`, or a JSON file
  `medh5.labels.load_file` reads, instead of minted in sorted order, so a
  case without one organ no longer renumbers the others; a mask the set has
  no class for is refused (E402).
- **`from_dicom(per_slice_rescale="apply")` imports a PET series with a
  rescale per slice.** An image stores one modality LUT, so a series whose
  slices state different `RescaleSlope`/`RescaleIntercept` is refused, and
  the refusal now names the option (`--per-slice-rescale apply`). With it,
  each slice is stored as `stored × slope + intercept`, computed in float64,
  as float32 with no rescale; a `per_slice_rescale` decision lists the slopes
  and their slices, and the import activity says `"per_slice_rescale":
  "applied"`. SUV is never computed, a slice stating no rescale among slices
  that do is still refused, and a series with one rescale is written the
  same either way. `read_series` takes the same option.
- **ITK and ANTs transforms import and export.** `medh5.io.itk` reads and
  writes ITK `.tfm`/`.txt` and ANTs `.mat` affines (`read_itk_transform`,
  `write_itk_transform`), and `from_itk_transform` / `to_itk_transform`
  store one between two grids' frames, re-expressed from ITK's LPS
  millimetres in each grid's coordinate system and units, with an ANTs
  displacement field (`warp`, and an optional `inverse_warp`) as a
  `composite` through an intermediate frame. `vectors="lps"` or `"ras"` is
  required with a field, whose file does not say. The command line has
  `medh5 convert from-itk-transform` and `to-itk-transform`. B-spline
  transforms and `.h5` transform files are refused.
- **`add_segmentation(labelmap=...)` writes a segmentation from one integer
  volume.** Each nonzero value is the class of that id, or the classes
  `label_values=` gives it (a value may be in several, as an nnU-Net region's
  component is), and `ignore_value=` marks the ignore region. It writes what
  the masks the volume implies write, byte for byte, without building them;
  the volume is borrowed from NumPy, not copied. In Rust,
  `SampleWriter::add_segmentation_labelmap` and `LabelmapSource`.
- **`from_nifti(plane_as_volume=True)` imports a 2-D plane where it lies.**
  A 2-D NIfTI off z = 0, or not parallel to x-y, is refused, since a 2-D grid
  has no z; the refusals now name the option (`--plane-as-volume`). With it,
  every 2-D image and mask is read as `(x, y, 1)` with the file's whole
  affine — the import of the plane re-saved that way — and a
  `plane_as_volume` decision records its z and normal. A third affine column
  that is zero or not orthogonal to the plane is refused. `read_nifti` and
  `import_seg_nifti` take the same option.
- **`from_nnunetv2(include_test=True)` imports an nnU-Net dataset's test
  cases.** `imagesTs` cases are imported with labels from `labelsTs` where
  present, and each case records its folder as a split claim in set
  `nnunetv2:<dataset>` (`--include-test`).
- **`to_nnunetv2(split=SET_ID)` writes a split's test cases to `imagesTs`.**
  A case claiming `test` in that set goes to `imagesTs` and `labelsTs`, one
  claiming `train` or `val` to `imagesTr`; an unplaced or held-out case is
  refused (`--split`). The stray-file check now covers `labelsTs`.
- **`to_nnunetv2(replace=True)` replaces an existing dataset.** The new
  dataset is written over it in place, file by file, each file whole, with
  `dataset.json` last; the NIfTI files of the case folders that the export
  no longer writes are removed after it, and every other file is left where
  it is (`--replace`). The report lists the files removed and left. A crash
  part way leaves each file whole but the dataset a mix of the old and the
  new until the export is run again.
- **RTSTRUCT POINT ROIs import as a `points` annotation, and export again.**
  `from_rtstruct` writes each POINT contour as one row of `points_id`
  (`<ann_id>_points` by default; `--points-id`), in world coordinates of
  the contours' frame, of the ROI's class, with `task="other"`; it was left
  out with a warning. A structure set of points alone writes no contours.
  `to_rtstruct(points=…)` (`--points`) writes them back as POINT items
  interpreted as `MARKER`. Open curves are still left out, now with a
  warning naming `open_curves="omit"` (`--open-curves`), the only value
  until `polylines` arrive in 2.2.0.
- **`from_rtstruct(roi_classes=…)` puts an ROI in a class of your choosing.**
  `{ROINumber: key}`, `--roi NUMBER=KEY` on the command line, imports ROIs
  that share a name apart; the key is minted, or must be in the sample's
  label set (`E402`).
- **`to_dicom_seg` writes a `probmap` as a `FRACTIONAL` SEG.** Pass
  `segmentation_type="fractional"` (`--segmentation-type fractional`), and
  `fractional_type="PROBABILITY"` or `"OCCUPANCY"` (`--fractional-type`).
  Values are stored out of 255, within 1/510; the report gives the largest
  error and how many voxels moved across the stored threshold, and says that
  `threshold` and `normalized` are not carried. Any other kind is refused.
- **A SEG export carries each class's codes, category and laterality.** The
  property type is the class's first code whose system has a DICOM
  designator (one table, `SNOMED-CT` ↔ `SCT` and six more, read both ways);
  the category comes from `properties["dicom_category"]` or `category`; a
  `laterality` is the type modifier. Where the sample states none, the
  generic SCT 91723000 is written as before, now as a guess naming each class.
- **`--workers N` for `validate`, `verify`, `fix`, `index build` and
  `recompress`.** Up to *N* processes run the files, and the output — the
  text, the `--json` document, the exit code — is a one-process run's, byte
  for byte, in input order. A file named twice, or by a hard link, goes to one
  process; a process that dies fails only the file it held. The default, 1,
  runs in one process as before.
- **`medh5 clinical augment --from-meds DIR --meds-subject N` and
  `--from-fhir FILE --fhir-subject REF`** import one subject's history in
  place of a logical-record bundle, onto the clock `--clock-id` and `--clock`
  name or the sample's own, with the conversion report printed and kept by
  `--report`. The standalone binary hands them to the Python package, as it
  hands over `medh5 convert`.
- **`ClinicalTaskDataset(class_ignore=True)` gives each class its own ignore
  mask.** `class_ignore[slot]`, `(K, *patch)`, holds per class the padding and
  the ignore regions of the annotations that label or examined that class
  only (1.0 §7.7), so one rater's uncertain band no longer masks another
  rater's class from a per-class loss. `ignore[slot]` is unchanged, and is the
  union of the channels.
- **`FileGroupedSampler` shards an epoch across DDP ranks.** New
  `num_replicas`, `rank` and `drop_last` (defaulting to `torch.distributed`'s
  world size and rank) give each rank one contiguous share of the epoch's file
  order, the same length on every rank and every epoch, so a file is split
  between ranks only where a share ends; with one replica the order is
  2.0.2's. `ClinicalTaskDataset` gains `file_groups()`, its rows grouped by
  subject, so the sampler takes a clinical task too.
- **Patch predictions stitch back into volumes.** `GridStitcher` takes a
  `GridPatchDataset`'s predictions batch by batch, in any order, and returns
  each file's volume as soon as its last patch arrives; `finish()` refuses,
  naming each file still missing patches, rather than return a volume with
  holes. Overlaps are blended with Gaussian or constant weights.
  `medh5.sampling.PatchAggregator` and `blend_weights` do the arithmetic
  without torch, and each `GridPatchDataset` item's `meta["cover"]` says which
  of its file's patches it is.
- **`MedH5Reader` lets MONAI's `LoadImage` and `LoadImaged` open `.medh5`
  files.** It is a MONAI `ImageReader`: it reads one image — or, with
  `annotation=`, one voxel annotation as an `int64` label volume, `65535` in
  its ignore region — with the affine `to_metatensor` gives, takes a
  collection member as `path::key`, and stacks several files as channels only
  when they share one grid. `LoadMedH5d` is `to_dict` as a dictionary
  transform. Both read through the per-process handle cache, and
  `import medh5.monai` still imports no MONAI.
- **The conformance suite scores feature caches and task results.**
  `companion/caches/` publishes nine feature caches over the suite's clinical
  samples: a valid cache of each level, and one for each code from T401 to
  T407. Each comes with the findings validating it must report, and with its
  task where the finding is about the task. Until now no fixture exercised
  any cache code. `medh5 conformance score SUITE RESULTS --companion` scores
  what another implementation reported on every task and cache fixture, as
  do `medh5.conformance.score_companion` and Rust's
  `conformance::companion::score`. A report's findings must equal the
  fixture's, and its `ok` must agree with them. For a fixture that states its
  rows, both a row added and a row left out fail it, and each row's reasons
  and inputs are compared as well as its status, selection, slots and label.
  Before, the fixtures were checked only from inside this engine's test
  suite.
- **A task from a cohort.** `TaskManifest.from_cohort(cohort, split, ...)`
  and `medh5 task new OUT --cohort C.json --split S.json` draw a task's
  subjects, pinned sources and partitions from a cohort manifest and its
  split, or the claims its samples carry (`--claims SET_ID`), refusing a
  split or claim made from another manifest, an unplaced entry, a subject in
  two partitions and a stale recorded `content_id`
  (`TaskManifest::from_cohort` in Rust).
- **Clocks and days for clinical tasks.** `Clock.utc(id)` and
  `Clock.shifted_utc(id)` build the two common clocks; `utc_us(datetime)`
  turns an aware datetime into the microseconds a task states, and
  `day_bounds(day, tz=None)` and `clock_day(t_us)` give the inclusive bounds
  of a calendar day, the span every time zone's day can cover when no zone
  is given (`Bounds::day_containing` in Rust).
- **`medh5 cache info` and `medh5 cache ls`** show what a cache holds and
  each entry it records, by source version or row, without decoding
  anything.
- **A moved cache validates through its task.** With `--task`, an entry
  source the task pins is opened where the task's reference says, and
  `cache validate --task-base DIR` resolves the task's URIs (contract §7.1).
- **Repin a task** after its samples were changed on purpose:
  `TaskManifest.repin()` and `medh5 task repin` pin each source to the
  version its file holds now, refusing one that is not whole or no longer
  its subject's, and report the rows whose selection may have moved. T302
  names the way out.
- **Free-threaded CPython 3.14 installs a wheel.** Each platform gets a second
  wheel, `cp314t`, for the interpreter the abi3 wheel cannot load, so medh5
  installs there without a compiler. The module asks for the GIL, which
  CPython turns on as it imports it; running without it is planned for 2.2.
- **Each pull request records its change in `changes/`.** A fragment per pull
  request, written while the change is fresh, replaces the hand-edited
  [Unreleased] section; `.github/scripts/changelog.py` checks them, previews
  them on the documentation site and assembles them at release, and CI fails a
  pull request that adds none (`changes/README.md`).
- **`medh5::AttrValue` is re-exported at the crate root.** The documented
  `extra` and `attrs` fields of grids, transform and annotation headers and
  payloads hold it, and `Image::attr` returns it; it was reachable only as
  `medh5::h5::attrs::AttrValue`, which stays.
- **The engine's readers are fuzzed twice a week.** Four targets run under
  libFuzzer (`fuzz/`): any file validated at the strictest level, a clinical
  documents table's offsets and bytes read deferred and scanned, and the
  Zstandard and Blosc2 filters decoding chunks they did not write.

### Deprecated

- **A default `sample_id` or pack key that is not the file stem warns.**
  Without `sample_id`, a sample is named for its file name up to the first
  dot, and `pack` keys a member the same way, so `case.v2.medh5` and
  `case.v3.medh5` both became `case`. medh5 3.0 takes the stem (`case.v2`).
  Changing it now would silently rename samples, and split `P001.ct` and
  `P001.mr` into two subjects. So `create` and `pack` raise a
  `FutureWarning` wherever the two differ, and `medh5 pack` prints a note.
  Pass `sample_id` (and `subject_id`), or `keys`, to keep the name you want.
- **`medh5.storage.dataset_kwargs` is now `dataset_layout`.** Since 2.0 it has
  returned the writer's layout (`{"chunks": ..., "codec": ...}`), not `h5py`
  keywords, so the old name misled. The old name is the same function and
  warns; 3.0 removes it. For `h5py` keywords, pass
  `resolve_profile(p).codec(role).kwargs()` to `create_dataset`.
- **`MEDH5IOError` derives from `MEDH5FileError` in medh5 3.0.** An HDF5 read
  failure will then be a `MEDH5Error` too; catch `OSError` to catch it in both.
- **`annotations::header::{ANNOTATION_KINDS, GEOMETRIC_KINDS}` and
  `version::FORMAT_VERSIONS` are deprecated in the Rust crate.** Their lengths
  are part of their types, so a kind or a version added in a later minor
  would break every caller; use `annotations::kinds(version)`,
  `annotations::geometric_kinds(version)` and `version::format_versions()`,
  which answer with slices. Removed in 3.0; Python's `ANNOTATION_KINDS` and
  `GEOMETRIC_KINDS` are unchanged.
- **The class-level `Tracking::{state_at, states, is_new, is_resolved}`
  are deprecated in the Rust crate.** Under them an object outside a
  visit's field of view reads as resolved; use `locate_tracks`. 3.0 removes
  them.
- **`scrub_paths` and `STRICT_RULES` (Rust).** `scrub_each` returns every
  path's result, and `strict_rules()` lists the rules a later minor adds;
  both old names are removed in 3.0.
- **`ClassStats.prevalence` answers 0.0 for a class no sample examined until
  3.0.** It raises a `FutureWarning` there, and only there: medh5 3.0 returns
  `None`. Rust's `ClassStats::prevalence` is deprecated for
  `examined_prevalence`.
- **A manifest written before 2.1 is read against the working directory until
  3.0.** Where that reading and its own directory's differ, loading it raises
  a `FutureWarning` (the native command line prints `warning: …`); medh5 3.0
  refuses it unless `base=` is given. Upgrade it with `medh5 dataset upgrade`.
- **`make_splits` without `algorithm` deals by `medh5.split/1` until 3.0.** It
  raises a `FutureWarning` (the native `dataset split` prints `warning: …`)
  only where `medh5.split/2` would place some group differently; pass
  `algorithm="medh5.split/1"` (`--algorithm 1`) to keep such a split.
- **A manifest's sources outside its directory are refused by default in
  3.0.** Until then they open as before; Python warns (`FutureWarning`) when a
  manifest loaded from a file names one, and the command line says so on
  stderr. Pass `confine=False` or `--any-source-path` to keep reading them, or
  name the directory to confine to.
- **A target with negatives states its follow-up.** `min_follow_up_us`
  defaults to the horizon, which leaves a negative only an observation at
  exactly the cutoff plus the horizon, so nearly every row without a
  positive is censored. The default cannot change under `medh5.task/1`: it
  is in every task fingerprint. `Target` with `negative` set and no
  `min_follow_up_us` now raises a `DeprecationWarning` (3.0 refuses it); a
  manifest read from a file gets a note instead
  (`TaskManifest.notes`, `medh5::companion::task::notes_of`), which
  `medh5 task validate` and `task preflight` print after the findings and
  list under `notes` with `--json`. A note fails nothing.
- **`medh5.task.STATUSES` and `LABELS`** are `ROW_STATUSES` and
  `TARGET_STATUSES` by other names, and `task.STATUSES` shared its name with
  `clinical.STATUSES`, a different vocabulary: reaching either now warns,
  and 3.0 removes both. Every vocabulary of the task and cache modules is
  now the engine's table (`ROI`, `CENSORING`, `ROIS`, `cache.LEVELS`,
  `cache.SUFFIX`, `clinical.ASSESSMENT_SYSTEM`, `LESION_PRESENCE`), and a
  status the engine wrote outside its table is refused rather than read as
  the table's last word. `build_document_cache` files documents by the
  engine's ownership rule, the one `validate_cache` holds entries to.
- **Six engine functions and one type that nothing called are deprecated
  in the Rust crate:** `clinical::columns::columns_of` (an unknown table
  name answered with the link columns), `annotations::encode::array_data`
  (use `PayloadData::Array`), `annotations::select::bbox_volume`,
  `bench::targeted_metrics`, `storage::codecs::profiles_repr`,
  `geometry::linalg::matmul` (use `a.dot(b)`) and `sampling::PairReport`
  (Python's `medh5.sampling.PairReport` is the live one). Each note names
  its replacement; 3.0 removes them.
- **`Deidentification.extra` is deprecated, in the Rust crate and in
  Python.** The schema's `deidentification` object allows no member it does
  not define, so anything `extra` holds makes the document invalid. Reading
  it warns; 3.0 removes it.

### Behaviour changes

- **An amendment refuses to replace a file that changed after it read it.**
  Two amendments, or an amendment and a `repack`, both renamed over the file,
  and the first one's work was lost without an error. An amendment, and a
  `repack` or `recompress` in place, now refuses to commit when its source was
  replaced or edited in place since it was read (`MEDH5FileError`, "changed
  after this amendment read it; nothing was written"). This detects a lost
  update; it does not lock one out (§14.4).
- **Files medh5 writes keep a symbolic link at their path, and the file's
  mode.** A conversion report (`convert … --report`) and a NIfTI sidecar
  replaced a link at their path with a new file, wrote a `0o600` file back as
  `0o644`, failed on a read-only file on Windows, and wrote `\r\n` for every
  `\n` there. `clinical export -o`, `migrate --report`, `dataset split -o` and
  `dataset stats -o` wrote in place: through a link, and not atomically, so an
  interrupted run left half a file. They all now go through the engine's
  replace, as every amendment already did (§14.4): the link is kept and the file
  it names replaced, its mode is kept, a read-only file stays read-only, the
  bytes are written as given, and an error names the file rather than the
  temporary beside it.
- **An amendment refuses a file whose `content_id` no longer matches it, or
  whose digests it cannot read.** A file edited outside medh5 (`E702`), one
  holding a digest that does not parse, or one declaring a `digest_algo` §2.1
  does not permit (`E703`) is refused before anything is written, by every
  command that amends: `medh5.amend`, `fix` without `--rewrite-digests`,
  `scrub --apply`, split claims, `clinical augment` and `strip`, and `seg`.
  Previously the next scrub, split claim or annotation re-attested the edit,
  and an unknown algorithm was silently replaced by `sha256`. Repair the file
  with `medh5 fix --rewrite-digests --reason`, which restamps an unknown
  algorithm in `sha256` and records the change in the activity's parameters.
  `verify` and `diagnose` name that repair when they meet an unknown
  algorithm.
- **A region's Jacobian is the field's own.**
  `jacobian_determinant(roi)` differenced the region as if it were the whole
  field, one-sidedly at its own edges, so its edge values depended on where
  it was cut: two halves of a field could each report no folding where the
  field folds. It is now `jacobian_determinant()[roi]` exactly — the
  differences reach one voxel past the region — and `folding_fraction(roi)`
  is the fraction over those voxels. A slice with a step and an axis of one
  voxel, refused before, are accepted; the field itself still needs two
  voxels on every axis. A large region is read in slabs.
- **`transform_between` over two visits refuses when a visit holds several
  frames.** It answered from whichever grids sorted first, so renaming a grid
  turned "no registration" into a CT's registration for a PET pair. It now
  raises `MEDH5ValidationError` naming each visit's grids and frames; with one
  frame a visit, answers are unchanged. The new keyword `by="timepoint"`,
  `"grid"` or `"frame"` (Rust `transform_between_as` and `KeyKind`) says how
  the keys are read, so a grid named like a visit can be asked about.
- **`--level strict` fails a file that draws a revision-2 warning; pass
  `--codes-revision 1` to keep 2.0's verdicts.** Revision 2 of the code table
  allocates W915: a dataset without a `digest`, inside an object a declared
  `content_id` covers, is reported at the integrity level (§13.1). Until now
  `validate --level strict` passed a file that `verify` failed as unattested,
  so a CI gating on the validator missed an edit nothing could detect. `verify`
  is unchanged; the two now fail the same files.
  At the semantic level it allocates W916, a classification under
  `closure="implicit"` asserting a class 0.0 beside a positive `is_a`
  descendant, which `state()` answers positive (§5.4, §9); W917, two grids
  naming one `frame_uid` that §3.3 still keeps apart, such as a 2-D key image
  tagged with a CT's frame (§3.3); and W918, a voxel annotation declaring
  `closure="implicit"`, whose ancestor reads include every descendant's
  voxels (§5.4).
- **`W919` reports a contour role that disagrees with its depth, and `E404`
  one that is not 0 or 1.** §8.6 now says how a class's polygons in a plane
  combine: by nesting depth, with `contour_role` the depth's parity. Files
  2.0 imported from same-named nested RTSTRUCT ROIs draw `W919`; under
  `--level strict` they fail; pass `--codes-revision 1` to keep 2.0's
  verdicts.
- **An object outside a follow-up's field of view is no longer `resolved`
  there.** Tracking maps the object's boxes from the visits that saw it into
  the follow-up's grid, through the shared frame or affine transforms (§7.4,
  as corrected in Appendix C.1): outside the grid or its images' acquired
  voxels it is `unexamined` (`outside_fov`), and so is it inside an ignore
  region (`ignored`). An ignore region away from the object no longer makes
  the absence `unexamined`, and a baseline that never imaged the place finds
  no new lesion there. Where nothing relates the visits, the class-level rule
  stands (`class_examined`).
- **A scrub keeps a `burned_in_annotation_checked` someone recorded as
  true.** It wrote `false` over it, though it does not look at pixels.
- **The prevalence of a class no sample examined is `null` in the statistics
  JSON.** A saved 0.0 recorded a measurement that was never made: a class drawn
  where nobody looked for it read as "looked for everywhere, found nowhere".
  `medh5 dataset stats` prints `-` for it.
- **`medh5 labels check` groups files by their label set's digest, as
  `dataset check` does.** An empty label set is a vocabulary of its own, told
  apart by its id, where it printed `<none>` alongside the files that declare
  no label set; two such files with different ids now fail the check, as they
  raise `C101`.
- **A cohort manifest's paths are relative to the manifest.** Every save names
  `"schema": "medh5.dataset/1"` and writes a path relative to the manifest's
  directory when it lies beneath it, absolute otherwise; loading joins it back
  onto that directory. The saved text changes only where it was already
  unreadable from any directory but the scan's: a manifest loaded elsewhere
  listed every file as stale, `dataset check` reported `C402` for files that
  exist, and the statistics opened nothing. `sha256` is unchanged.
- **A time given in hours or days is rounded half to even everywhere.**
  `medh5 clinical select --cutoff-hours` and the imaging events
  `clinical augment` makes from `days_from_baseline` rounded half away from
  zero, while `medh5.clinical.hours` and `days` round half to even, so at an
  exact tie a cutoff from the command line was one microsecond later than the
  same cutoff from a script. The engine now rounds once
  (`medh5::clinical::round_micros`), half to even, as Python always has.
- **`to_rtstruct` refuses a `label` longer than 16 characters.** The
  `StructureSetLabel` is a DICOM SH: a longer label was written, and the file
  was not valid DICOM. Shorten it (`label=`, `--label`).
- **A NIfTI, SEG or RTSTRUCT export keeps a symbolic link at its path, and
  the file's mode.** `to_nifti` and `write_nifti` replaced a link at their
  path with a new file and wrote a `0o600` file back as `0o644`; they now
  replace the file the link names and keep its mode, as every file medh5
  writes does (§14.4). An nnU-Net export writes through a link at a case
  file inside the dataset, and refuses one naming a file outside it at any
  channel's file — only the first channel's and the labels' were checked.
- **A qform-only NIfTI imports to the same grid on every platform.** nibabel
  builds the qform in `np.longdouble`, 80-bit on x86-64 and 64-bit elsewhere,
  so direction and spacing — and the sample's `content_id` — differed in the
  last bit between platforms. `read_nifti` now computes the qform in float64
  from the quaternion parameters, as NIfTI-1 defines it: imports on x86-64
  can change in the last bit, and are the same everywhere.
- **An nnU-Net import that leaves test cases out warns.** `from_nnunetv2`
  imports `imagesTr` only, as before, and the `imagesTs` cases it leaves out
  are now a warning that counts them, so `medh5 convert from-nnunet` exits 1
  on a dataset with an `imagesTs`. Pass `include_test=True`
  (`--include-test`) to import them.
- **Same-named RTSTRUCT ROIs are one class, with a note, or refused.** ROIs
  whose names make one key are combined into one class, minting or not, and
  the report records a `merged_rois` decision naming them; minting used to
  fail with `E302`, and an existing label set merged them silently. Roles
  now come from nesting over the whole class: an import where one ROI's
  contour lies inside another's on a slice is refused, naming the ROIs and
  the slice, where 2.0 made the inner one a hole (an 11×11 "Bladder" holding
  a 4×4 one, 121 voxels drawn, imported as 105).
- **A SEG export's category follows the class.** A `lesion` class is SCT
  49755003 "Morphologically Abnormal Structure", a `device` SCT 260787004
  "Physical object" and a `region` SCT 309825002 "Spatial and Relational
  Concept", where every segment was "Anatomical Structure".
- **A predicted annotation exports as `AUTOMATIC`.** A `predict` activity by a
  software agent of stated version makes its segments `AUTOMATIC`, identified
  by that software (family DCM 123110 "Artificial Intelligence"); every
  segment was `MANUAL`. Without such provenance the export still writes
  `MANUAL`, and records a guess. A binary export of a `probmap` records the
  threshold it cut at.
- **A SEG import records category and laterality, and spells systems as §5.2
  does.** A minted class keeps the property category as
  `properties["dicom_category"]`, takes `category` where the concept names one
  (`lesion`, `device`, `region`) and `laterality` from the type modifier; a
  binding's `system` is `SNOMED-CT`, not `SCT`, for a designator the table
  knows. Segments of one label with different modifiers are refused, where
  Left and Right Kidney became one class.
- **MONAI can batch medh5 items.** `meta["medh5"]` carried `None` for a
  sample's absent `frame_uid`, `value_units`, `modality`, `timepoint` or
  `subject_id`, and MONAI's `list_data_collate` refuses a `None`, so no sample
  without a frame of reference — every NIfTI and nnU-Net import — could be
  batched. The MONAI metadata now spells an absent value `"none"`, MONAI's own
  `TraceKeys.NONE`, and `from_metatensor` reads it back as `None`: code testing
  `meta["medh5"]["frame_uid"] is None` now sees `"none"`. `meta["medh5"]` also
  names the collection member it came from, as `sample_key`.
- **`TaskManifest.save` and `load` are the engine's.** Python wrote a
  manifest with its non-ASCII text as is, the engine with it escaped, so one
  document saved as two files; Python now writes the engine's bytes, the
  same as `medh5 task reconcile`. `TaskManifest.load` refuses a manifest
  holding `NaN` or an infinity (T101), as the engine did: JSON has neither.
  `TaskManifest.save` refuses a path that is one of the manifest's sources,
  by any name or link, as `medh5 task reconcile --out` already did, where it
  replaced the sample with JSON. `TaskManifest.reconcile` drops the declared
  fingerprint, as the command line does: the manifest changed.
- **The repository and the Homebrew tap moved to the `medh5-project`
  organization.** Install with `brew install medh5-project/medh5/medh5`; the
  old `XwK-P` addresses redirect, so an existing clone or tap keeps working.

### Fixed

- **An amendment keeps the file's group.** A replaced file belonged to the
  amending user's primary group, so a `0o640` file of a shared project group
  became readable by the wrong group and unreadable by the right one. The
  replacement now takes the file's group (and, run as root, its owner) before
  its mode. Where you are not in the file's group, the replacement grants its
  own group only what it grants everyone, which never widens access. Every
  file the engine writes — samples, manifests, reports — does the same. Access
  control lists and extended attributes are not carried.
- **A file name with nothing before its first dot is refused by name.**
  `.hidden.medh5` failed "[E005] identity requires both sample_id and
  subject_id", naming neither the file nor the cause. It now names the file
  and says to pass `sample_id`.
- **`Image.read` takes an Ellipsis.** `read((..., slice(0, 2)))` raised
  `TypeError` where the dataset's own `[..., 0:2]` worked; one `...` now
  stands for the whole axes the other items leave, and two raise
  `IndexError`, as in NumPy.
- **Two `Instance`s compare by value.** `a == b` and `a in [b]` raised
  "truth value … ambiguous" on the box array; equality now compares every
  field, the box and mask by value. An instance is still unhashable.
- **A Python converter refuses a collection in the native commands' words.**
  It said "works on sample files" where every native command says "works on
  one sample file"; both now come from one function.
- **`medh5.document.schema()` returns a fresh dict on every call.** It was
  cached, so every caller shared one mutable dict, and one caller's edit was
  every later caller's schema.
- **`Measurement.to_json()` writes `None` for a NaN, as `medh5 bench --json`
  does, and `Measurement.from_json()` reads it back as NaN.** It wrote
  `NaN`, which JSON cannot carry. The result types —
  validation `Report` and `merge`, `conformance.summarize`, `Case`,
  `CaseResult`, `RecompressResult`, `Diagnosis`, `Repair`, `VerifyResult` and
  `Measurement` — now ask the engine for their verdicts, counts, text and
  JSON, so they say what the command line prints.
- **Reading contours no longer fails on a `contour_role` value.** The reader
  raised an index error on a role other than 0 or 1, against §16's rule that
  readers ignore values they do not know. A polygon's role is now its
  nesting depth's parity among its class's polygons in its plane, from
  which §8.6 now defines the region; a polygon out of any plane keeps the
  role stored for it.
- **A digest spelled in uppercase is malformed (E703), not a mismatch (E701).**
  A digest was read as Python's `int(hex, 16)` reads a number, so a sign,
  `0x`, `_`, whitespace, uppercase or short hex parsed, and an uppercase
  digest of unchanged bytes was reported as a mismatch and its root as
  tampered with. §13.1 now says `<hex>` is lowercase hex of the algorithm's
  length; anything else is E703, in `verify`'s and `diagnose`'s `malformed`,
  and `fix --rewrite-digests` restamps it. A `content_id` spelled otherwise is
  E703 rather than E702, and an amendment refuses it. `parse_digest` stays as
  lenient as it was; `medh5::digest::is_canonical_digest` decides.
- **A sampling index is read in the algorithm its digest names.** §13.3
  prefixed an index's `source_digest` with `<algo>:` and never said what
  hash it was; medh5 always recomputed in sha256, so a sha512 index another
  writer put in a sha512 file was stale (W905) and the file failed `strict`.
  The prefix now decides, as it does for a dataset's digest: `fresh_indices`,
  `verify` and the validator recompute under it, and an entry naming an
  algorithm outside §2.1, or spelled otherwise than §13.1 spells a digest, is
  stale. medh5 still pins its own indexes in sha256, so every index it wrote
  stays current.
- **Voxel annotations are not pyramided in 1.x.** §4.3 said a voxel
  annotation "MAY be pyramided identically, with `nearest` or `max`" and
  defined nothing more: not which voxel `nearest` keeps under an even factor,
  what `max` means over class ids, or how coverage and the ignore region
  downsample. Nothing wrote, read or validated one. §4.3 now says an
  annotation's datasets hold one level, and §16 reserves `levels`,
  `downsample_factors`, `downsample_method` and `grid_levels` on an
  annotation for a later minor. The writer refuses them (E412), so a
  transcode no longer carries one a third-party header holds.
- **1.1 §10 names MEDS's columns as MEDS does.** It gave this profile's
  column names (`value_num`, `value_text`) for MEDS's, so an importer written
  from it looked for columns no MEDS dataset has; it now maps `time`,
  `numeric_value` and `text_value` onto them (Appendix A).
- **A comparison across two samples no longer records the other file's path
  as its own.** Its `against` was `annotations/<the other annotation>`, which
  resolves inside the recording sample. It is now `None`, and `to_record()`
  refuses it unless `against=` names an annotation of the recording sample,
  `annotations/<id>`; in Rust, `to_record_against`. §11.2 now says whose
  annotation `against` names.
- **`skipped` names each class once.** A class each side calls by the same
  key was listed twice.
- **A date in a locale's order and a DICOM time are reported.**
  `31.01.2020`, `01/31/2020` and `093000` under a key naming a time passed
  every scan; they are now reported for review, never shifted, since which
  number is the day is a guess.
- **A refused file no longer loses the reports of the files scrubbed before
  it in Rust.** `scrub_paths` stopped at the first refusal, and the `uid_map`
  of each file already rewritten — the only key to its pseudonyms — went with
  it; `scrub_each` keeps them.
- **`Deidentification::shift_days` reads a recorded 0 as no shift,** as the
  scrub always had.
- **`Entry.from_json` reads an entry as the engine does.** It coerced a
  document its own way, so `"deidentified": "false"` made a de-identified
  member, a string size or mtime was kept, and `save` wrote what it coerced;
  `Manifest.load` was not affected.
- **An amendment holds what a clinical event version attests, wherever it
  writes.** Writing to another path read the output for what the source held,
  and passed; leaving the clinical group behind with `amend_into` held no
  version, so one added again could carry a new payload; and an attested
  payload was compared by its datasets' digests alone, so an edit to an
  attested image's `rescale_slope` passed. Each is now `E809` (1.1 §7.3).
- **A class named in `annotated_classes` that the input does not supply is
  recorded as examined and absent.** The NIfTI, DICOM SEG and RTSTRUCT
  importers raised a bare `KeyError` ("label set 'converted' has no class
  'tumor'"). An import that mints its classes now mints it after the
  supplied ones, with a `label_set` decision; one into a given label set
  that lacks it is refused as E402, naming the class.
- **`to-dicom-seg --series-description` longer than 64 characters is refused
  by name.** highdicom's `ValueError` for the DICOM LO reached the command
  line as a traceback.
- **An interrupted DICOM SEG or RTSTRUCT export leaves the previous file
  whole.** `save_as` wrote over the previous file in place, so a crash or a
  full disk during it left a truncated SEG or structure set. The file is now
  written to a temporary beside it and renamed over it once whole, through
  the engine's replace (§14.4); a failure leaves the previous file
  byte-identical and no temporary behind.
- **An interrupted nnU-Net export leaves the previous `dataset.json` whole.**
  It was written in place; it now goes through the engine's replace, and is
  the last file an export writes, so an export that stops before it keeps
  the old description. Its NIfTI files were already written whole; they now
  go through the same replace.
- **`ConceptVocabulary.save` writes the file whole.** It wrote through the
  file in place, so a reader could see half a vocabulary and a failed write
  lost the previous one; it now writes a temporary beside the file and renames
  it over. `ConceptVocabulary.digest` is unchanged, and its documentation now
  says why it is not §5.1's canonical JSON: feature caches record it.
- **A cache entry with no sources is T401.** The writer refused it as stale
  (T403), but the schema requires a source, and a writer refuses an entry
  that would fail its schema (T401).
- **`CacheEntry.from_json` reads an entry as the engine does**, refusing a
  member of the wrong type (T401) rather than reading it leniently; and
  `CacheReport.ok`, `stale` and `corrupt` are the engine's verdict
  (`CacheReport.to_json`).
- **`Selection.admits_any(kind, id)`** (Python and Rust) looks in every
  fragment: `admits` answers for one, fragment 0 unless told, so a payload
  of a subject's second fragment read as not admitted.
- **A negative cutoff with an exponent** is written
  `--cutoff-hours=-5.5e-07`: after a space it reads as a flag. The reference
  says so.
- **Default source ids no longer collide, and are the same on every
  platform.** A reference given no id took the file's stem after its
  directory, so `a/site-a/P-01.medh5` and `b/site-a/P-01.medh5`, or
  `P-01.medh5` and `P-01.h5`, were one id (`T204`), and Python split the URI
  with the platform's path rules, so one script's ids, and its fingerprints,
  differed on Windows. The engine derives the id now
  (`SourceRef::default_id`), splitting on `/` only, and
  `TaskManifest.add_subject` widens a derived id another source holds — by
  the directories above it, then `@2`, `@3` — leaving the first holder's as
  it was. An id given explicitly is never changed. Manifests that validated
  before keep their ids.
- **CI and the advisory audit run on the patch line.** CI ran on pushes and
  pull requests to `main` only, so a fix on `release/2.0` merged untested, and
  the weekly audit, which runs on the default branch, saw only `main`'s lock.
  Both now run on `release/**`, and the weekly audit checks each line.
- **A dataset's filter pipeline reports every client value of a filter.**
  `h5::data::filters` read at most 32, so a filter with more lost the rest,
  and it recorded an entry for a filter it could not read; it now reads them
  all, and a filter it cannot read is an error.
- **An unknown diagnostic code is quoted the same from Rust and Python.**
  `codes::code` quoted it as Rust does (`"X999"`) where `medh5.errors.code`
  quoted it as Python does (`'X999'`); `errors.code` now asks the engine.

### Performance

- **Windows remembers a file it has checked.** Opening a sample checks that it
  reads no bytes from outside itself, and POSIX remembered an unchanged file by
  its identity, so a training loop re-opening the same files paid for the check
  once. Windows had no identity to remember a file by and walked every file on
  every open. A file there is now known by its volume, its 128-bit file id, its
  size, and its last-write and change times, read from the handle that opened
  it; a file changed since it was checked is checked again, and so is one
  another program has open for writing, whose times Windows may not have updated
  yet.
- **More calls let other Python threads run while they work.** A large array
  argument (1 MiB or more) is copied into the engine's layout with the GIL
  released, and so are `curation.agreement.dice` and `iou`, which now count on
  the masks themselves, broadcasting a plane against a volume without copying
  it; `encode_contours` packs with the GIL released; `encode_boxes`,
  `encode_obb`, `encode_keypoints`, `encode_points`, the writer's `add_boxes`,
  `add_obb`, `add_keypoints` and `add_points`, an annotation's `to_world` and
  `to_index`, and `transforms.refuse_outside` release it from 4096 rows, and
  read class ids given as a NumPy integer array as an array rather than one
  Python int at a time. Letting go of the last handle to a dataset just
  written flushes its chunk cache with the GIL released too: 58 ms that held
  it for a 192 × 256 × 256 image. A `dice` of two 300 × 512 × 512 masks took
  0.88 s, every millisecond of it holding the GIL; it now takes 0.06 s and
  holds it for 1 ms. `normalize_masks` no longer copies a C-contiguous boolean
  mask: it returns the array it was given. Do not change an array while a call
  reads it.
- **`iter_patches` draws one patch per `next()`.** It drew all `n` patches
  before yielding the first, so `iter_patches(n=10_000)` held every patch at
  once. It yields exactly what `draws` returns for the same seed, and checks
  the annotation and the grid when it is called.
- **A composite prefilters a cubic component once.** A composite opened its
  components afresh at every evaluation, so a `cubic` displacement field
  inside one was read whole and prefiltered on every `transform_points` call
  — once per item of a paired loader. Components and named inverses now
  share the coefficients of the transform of that id, within the same
  process-wide budget.
- **The nnU-Net import no longer builds a mask per class.** A case's label
  volume is written through `labelmap=`: at 104 classes of 384³, the masks
  cost 11.2 GB of peak memory above the image write, and the label volume
  nothing.
- **Batch commands scale with processes.** One lock covers the HDF5 library
  in each process, so files checked in one process take turns; `--workers`
  runs them side by side: over 32 files on four cores, `verify` went from
  10.1 s to 2.7 s, `index build` from 17.4 s to 4.6 s and `recompress` from
  128 s to 45 s.
- **Validating an event-level cache opens each source once**, not once for
  its pin and again for what its versions own; `TaskManifest::rows_in`
  looks each subject up once rather than once per row.

### Documentation

- **A physical read is float32 only where a rescale applies.** The reference,
  the PyTorch page and the cohort and DICOM guides now say so, and that
  `dtype` gives one type across a cohort; `physical=True` itself is unchanged.
- **1.1 §10 describes de-identification as a derivation,** with its row in
  Appendix A.
- **1.0 §11.4 advises against recording the shift's amount** in a file shared
  beyond whoever holds the key, and 1.1 §3 says `/meta` may withhold it; a
  conformance case holds a shifted clock whose shift is withheld.
- **The cohort check codes page shows each code's severity.** It is rendered
  from the engine's table, with `C401`'s promotion under `--deep` beneath it.
- **`cargo install medh5-cli` can build the binary the releases ship.** A
  crate on crates.io carries no build profile, so the install built without
  link-time optimisation and kept the symbol table; the install page and the
  crate's README give the release profile as environment variables, in
  PowerShell and as `--config` options, and still prefer the release binaries.
- **The Rust crate states its public surface.** hdf5-metno 0.15, ndarray 0.17
  and IndexMap 2 are public dependencies, so a bump of any of them is a major
  medh5 release. `h5`, `json`, `numeric` and `pyval` are plumbing outside
  semver from 2.1; `rng` and `error` join the module table, which the crate's
  README and the reference page now keep equal.
- **An advisory against what an extra installs no longer holds a release.**
  medh5 neither ships nor pins PyTorch, MONAI, pydicom or the other extras'
  projects, so the release reports their advisories in its run's summary,
  for `SECURITY.md` to disposition; one against a compiled-in crate or NumPy
  still stops the release.
- **Every public Python name has a reference entry.** The new API index lists
  each name a public module exports, with its kind, its docstring's first
  sentence and a link to the section that documents it. It is generated from
  the package, so it cannot fall behind, and every public name now has a
  docstring.
- **Import from MEDS or FHIR**, a new guide: both mapping tables, the
  conversion report, the refusals, and what `medh5 scrub` does to an imported
  history and what it leaves to a text de-identification tool.
- **A tutorial from clinical history to a training batch.**
  `docs/tutorials/clinical-task.md` writes two subjects on a `utc` clock, each
  event with when it happened and when it became known, asks a task of them at
  a cutoff, shows a preflight excluding a row and why, and loads a batch. Its
  blocks build on each other, and the documentation tests now run such a page
  (marked `<!-- sequential -->`) top to bottom in one session.
- **A page on threads, processes and open files.**
  `docs/explanation/concurrency.md` gathers what was spread over five pages:
  one HDF5 lock per process and the Blosc2 window reads that decompress
  outside it, when threads help and when processes do, `fork` and `exec`,
  one thread per writer, `amend` replacing the file, and where free-threaded
  Python stands.
- **How-to guides for collections, review records, label sets, the
  conformance suite and DICOM SEG and RTSTRUCT imports.**
  `docs/guides/collections.md` packs samples into a `.medh5c`, reads and
  verifies its members and splits a cohort over them;
  `docs/guides/review-and-provenance.md` records who drew and reviewed a
  mask, its quality, and two raters' agreement, through `medh5.amend`;
  `docs/guides/label-sets.md` chooses a vocabulary, finds and fixes the
  `C101` and `C102` a cohort's files raise, and compares two raters'
  numberings with `map_classes`;
  `docs/guides/conformance.md` publishes, validates and scores the suite,
  bounded for an earlier implementation, and takes over the how-to the
  conformance page held; `docs/guides/import-seg-rtstruct.md` imports a
  SEG and a structure set onto a CT series, keeps same-named ROIs apart
  with `roi_classes=`, and shows what the importers refuse.
- **`SECURITY.md` says what opening a file you did not write guarantees.** No
  filter plugin is loaded unless asked for, nesting and decompression are
  bounded, a malformed file is refused rather than crashing the process, and
  a rewrite refuses a link in another user's sticky directory. It also says
  what is not guaranteed: a file within those bounds can still cost time and
  memory in proportion to what it states.
- **Every schema the engine embeds is on the site.** The clinical, task
  manifest and cache manifest schemas get a reference page each, rendered from
  the schema as the sample-document schema's page is, and all four are served
  under `reference/schemas/`. The sample-document schema stays at its old URL
  too. Each page says where its file is served; the `$id`s are unchanged.

## [2.0.3] — 2026-10-14

**A patch release, with no format change.** Four fixes found after 2.0.2, and
nothing else: the documented API is unchanged, and no `content_id`, fingerprint
or normal form changes. A task manifest no longer reports anything from a source
that is not the version it pins, and a pipe given as a source no longer hangs
the command reading it; on Windows a program started while a file is open no
longer keeps it locked, which made every amendment fail while it ran; a
procedure that asserts it drew a mask attests the mask under effective
ordering too;
and the DICOM extras declare the pydicom floor highdicom requires.

### Security

- **A task manifest reads and reports nothing its pins do not name, and a pipe
  given as a source is refused.** Once a source failed its pin (`T302`),
  `task preflight` kept reading the file: `T303` printed the subject id it
  holds, `T304` and `T305` compared its clock and events with the other
  sources', `--json` reported its whole event table, and `T302` itself printed
  the `content_id` the file stores. A manifest naming a file its author does
  not hold therefore reported that file's contents. A source off its pin now
  adds nothing to its subject and `T302` names only the pin, in the engine and
  in `build_document_cache`; `task reconcile` refuses such a source (`T302`)
  rather than reconciling its events. A named pipe or a device given as a
  source blocked `task preflight`, `task reconcile` and `cache validate` inside
  HDF5's open; every open now refuses what is not a regular file.
  task-cache-1 gains Appendix A (Errata), and `SECURITY.md` says what running a
  manifest someone else wrote can open.

### Fixed

- **On Windows, a program started while a file is open no longer keeps it open
  and locked.** HDF5's handle was inheritable, so a child process started while
  a sample was open kept it open and locked after medh5 closed it, and every
  rename over the file — the last step of every amendment, `scrub --apply`,
  `recompress` and `fix` — failed "Access is denied" while the child ran. 2.0.2
  fixed this on Linux and macOS; the handle is now kept out of children on
  Windows too. On every platform a file is opened and marked in one hold of
  HDF5's lock, and the programs medh5 starts itself (the Python commands of
  the standalone `medh5`, the interpreter probe) start only outside it, so none
  can catch a file between the two; on Linux, where a starting child shares
  every open file for an instant after the parent resumes, medh5 also waits
  for that instant to pass before another file is closed or opened.
- **The DICOM extras require pydicom 3.0.1.** highdicom 0.23 requires it, so
  the declared 3.0 floor was one no install of `medh5[dicomseg]` could resolve;
  `medh5[dicom]` and `medh5[dicomseg]` now say `pydicom>=3.0.1`, and CI installs
  exactly that floor.

### Behaviour changes

- **A procedure's assertion about its own mask attests the mask.** 2.0.2 made
  an asserting version need only be eligible at the cutoff, but when the link's
  event endpoint was the asserter itself — `seg0 describes lesions_tp0`, as both
  shipped examples phrase it — the asserter still had to be admitted, so under
  `order_by = effective` a segmentation procedure of unknown time never
  attested its mask, and `roi = eligible_instances` fell back to
  `center_fallback`. An endpoint that is the asserting version now needs only
  its eligibility (1.1 §7.3, §9.1 step 5; task-cache-1 §3.5). The conformance
  suite gains the task fixture `valid-self-asserted-mask`, whose row states the
  annotations its slot may read. No fingerprint changes, so a cache built
  before still validates: rebuild caches whose slots read self-asserted masks.

## [2.0.2] — 2026-10-13

**A patch release, with no format change.** What the writer stores keeps the
`content_id` 2.0.1 gave it, though a new file may be chunked differently; the
Python API gains only attributes and keyword arguments, and the Rust crate keeps
its 2.0.1 signatures, but for three result structs that gain a public field and
two key arrays that lose entries no key could match (Behaviour changes). The
converters, `migrate`, `clinical export -o` and `task reconcile -o` no longer
overwrite the files they read, and what gave a silently wrong answer — a
transform chain whose steps do not meet, a merged class, a fabricated negative,
an identifier a scrub left behind, a validation that passed a file `verify`
failed — is now refused or reported. A hostile sample can no longer crash the
process that opens it, make it load a library or exhaust its memory, nor can a
hostile feature cache make `FeatureCache.get` allocate without bound; the torch
loaders no longer serve wrong targets, string datasets read, and encoding,
transcoding, digests, index building and field sampling release the GIL.

### Security

- **A file can no longer crash the process that opens it by nesting groups.**
  Opening a file checked that it was self-contained with HDF5's recursive
  `H5Lvisit`, which has no depth limit: 30,000 nested groups crashed the process
  (SIGSEGV), and 10,000 took a minute. The check now walks the file itself, a
  group at a time, and a file nested more than 64 groups deep is refused at open
  (`E001`).
- **HDF5's filter plugins load only when asked for.** A dataset names its
  filters by number, and HDF5 loaded one it had not registered from the shared
  libraries in its built-in search path — on Windows
  `%ALLUSERSPROFILE%\hdf5\lib\plugin`, which any local user can create — as
  whoever opened the file. Blosc2 and Zstandard are compiled in, so plugins are
  now off, and a dataset whose filter is not built in fails to read with HDF5's
  "filter plugins disabled". `HDF5_PLUGIN_PATH` loads them as before, and
  `MEDH5_ALLOW_PLUGINS` (set, and not `0`) from HDF5's standard directory only,
  never the directory HDF5 was built in; both are read once per process.
- **A chunk can no longer decompress to far more than its dataset's chunk.** The
  Zstandard and Blosc2 filters allocated whatever size a chunk stated: 32 KiB of
  run-length blocks asked for a gigabyte, and a Blosc2 header can state 2 GB in
  a few bytes. Both now cap their output at twice the buffer HDF5 hands them
  plus 64 KiB, refusing a stated size over it before anything is allocated, and
  the Blosc2 filter checks that a chunk's header parses
  (`vendor/hdf5-blosc2/PATCHES.md`, patch 4).
- **A damaged Blosc2 chunk no longer tears down Blosc2's global state.** The
  filter's failure path called `blosc2_destroy()`, which frees the library's
  global context and destroys the mutexes a window read on another thread may be
  using, and a frame of no chunks made it free an uninitialised pointer. Both
  are fixed (patches 5 and 6), with tests that call the filter as HDF5 does.
- **A rewrite through a symbolic link rewrites the file the link names.**
  `amend`, `scrub --apply`, `recompress` and `fix` renamed their result over the
  link, which became an independent copy, and left the linked file as it was —
  after `scrub --apply`, the identifying original. The replacement is now
  written beside the real file and renamed over it, following a chain of links,
  and the link is kept (§14.4); a link in a sticky, world-writable directory
  that another user owns is refused, as Linux's `fs.protected_symlinks` refuses
  it.
- **A temporary file is no longer one another user can name in advance.**
  `to_nifti` and `to_nnunetv2` wrote each image and its sidecar to
  `.<pid>-<name>` beside the target and renamed it into place, so a link
  planted at that name, by anyone who could write the directory, was followed
  and the file it named overwritten. The temporary is now created exclusively,
  under a name with a random part, as are those of a converter's `--report` and
  `TaskManifest.save`.
- **`scrub` reads every name, and every copy of the document.** A second hard
  link's name, a soft link and its target, a committed datatype's name and
  attributes, and `/meta` under a second name or copied elsewhere all scanned
  clean. Each name and attribute is now scanned, a second name for `/meta` or a
  copy of the document is an identity finding that `--apply` removes, and what
  the scan cannot decode — strings nested in compound, array or variable-length
  types, and byte buffers or undecoded types outside the imaging groups — is
  reported `unreadable`, with every non-string dataset outside those groups
  named in `not_checked`. A file that scanned clean may not now.
- **An amendment keeps no copy of the identifiers it replaced.** It copied
  `/meta` by path, so a hard-link alias of it such as `x_acme/meta` kept the old
  document — the identifiers `scrub --pseudonymise-ids` had replaced — and
  `remap_frame_uids` left every replaced UID in the file's free space, where
  HDF5 keeps an overwritten value. The commit now re-links each alias to the new
  document and, after an alias or a remap, rewrites the file from its root.
- **`--pseudonymise-ids` replaces an old id wherever a string holds it.** A
  record number in a label, a note or an attribute outlived its pseudonym; every
  string in the file is now swept for each replaced id of four characters or
  more (a shorter one, such as `1` or `CT`, would be found in every number and
  modality).
- **Identifiers the scrub's tables could never match are found.** The key tables
  were stored unnormalised, so `nhs_number`, `birth_date`, `date_of_birth` and
  every `*_uid` key never matched, top-level `extra` and `acquisition` keys such
  as `PatientID` escaped the key rules, and a value that is wholly a date under
  an unlisted key, a UID inside text, the de-identification record and agents'
  `organization` were never examined. Each is now found, and shifted,
  pseudonymised or removed as its kind is.
- **A hostile feature cache can no longer make `FeatureCache.get` allocate up to
  physical memory.** An entry's stored dtype and shape are checked against the
  cache's declaration before its payload is read, and a mismatch is `T402`, with
  nothing read.
- **Advisories are checked weekly and before every release.** `cargo audit` over
  `Cargo.lock` and `pip-audit` over the runtime dependency and every extra but
  `dev` run every week, on a pull request that changes `Cargo.lock` or
  `pyproject.toml`, and before the release workflow publishes anything.
  `SECURITY.md` inventories the C libraries compiled in, which neither tool can
  see — HDF5 2.2.0, zlib 1.3.2, C-Blosc2 3.3.5, LZ4 1.10.0 and Zstandard 1.5.7 —
  with a disposition for every advisory against them.
- **CVE-2026-85091, the heap overflow in zlib's `gz_vacate`, does not affect
  medh5.** The bundled zlib 1.3.2 is in its range, but nothing medh5 compiles in
  uses zlib's gzip file interface and no `gz*` function is linked, so the
  overflow cannot be reached. `vex/medh5.openvex.json` says so to scanners, and
  zlib will be bumped once a release fixes it.
- **Release builds use pinned inputs.** Rust is pinned in `rust-toolchain.toml`
  (1.97.0); the build script pins maturin and cmake and checks `rustup-init`
  against a SHA-256 instead of piping a download to a shell, and the manylinux
  images are pinned by digest.

### Fixed

- **A rewrite that fails leaves the file as it was.** A commit whose final close
  failed renamed the file into place anyway; it now fails the commit. A
  fixed-length string dataset widened to fit a longer value, as `scrub --apply`
  may widen one, is swapped in so that a failure at any step leaves the original
  and no temporary behind.
- **A dataset whose size does not fit a machine word is an error.** Its byte
  count, the product of a shape the file states, wrapped round, or panicked in a
  debug build.
- **A read-only file can be amended on Windows, and stays read-only.** The
  temporary file was made read-only before it was opened to be flushed, and
  Windows does not rename over a read-only file; the order is fixed, and a test
  amends a read-only file on every platform.
- **A program started while a file is open no longer keeps the file locked.**
  HDF5 opens files without close-on-exec, so a child process started while a
  sample was open, by any thread, inherited the descriptor and the lock HDF5
  takes through it: once the sample was closed, opening it again failed
  "unable to lock file" until the child exited, and a child started during a
  write held a descriptor that could write the file. The engine now marks each
  descriptor close-on-exec as it opens the file, on Linux and macOS.
- **`W907` no longer advises `int16` for a float image holding NaN or
  infinity.** Only the finite values were tested, so an image with NaN outside
  its field of view was advised into a type that cannot hold it.
- **`repr()` of a closed `Sample` or `Image` says that it is closed.** It
  raised, and debuggers, loggers and test reports all ask for it.
- **A string dataset reads through `Dataset.read()` and `numpy.asarray(ds)`.**
  Both refused strings, though the reference documents them; a string dataset
  now reads as an object array of `str` in its stored shape, 0-d for a scalar,
  which `ds[...]` indexes as NumPy does (Behaviour changes).
- **A free-threaded Python turns the GIL back on when it imports
  `medh5._core`.** PyO3 declares a module free of the GIL unless told otherwise,
  so a build from source for free-threaded CPython ran the extension, which was
  never audited for that, without the GIL. The module now declares that it uses
  the GIL, so such an interpreter re-enables it at import, with its usual
  warning.
- **The document's live views stay true to the file after a failed edit.** An
  `update` refused partway hid the members it had written, a nested `sort()`
  that failed partway left the list part-sorted for the next edit to write, and
  a refused slice assignment on a stale list changed it before refusing.
- **Under `closure = "implicit"` the readers answer for a class's ancestors
  (§5.4, §9).** The closure was stored and never applied, so a classification
  asserting only `left_kidney`, with `kidney` examined, answered `kidney`
  negative — a false negative for exactly the pattern §9 recommends. `dense()`,
  `contains()`, voxel counts, bounding boxes, the index and
  `Classification.state()` now treat a class as present wherever a descendant
  is; what is stored (`labelmap()`, `value()`, `labels`) stays literal, and
  `explicit` is unchanged ([Annotations](docs/reference/annotations.md)).
- **`payload_to_masks` reports an inconsistent payload as an error.** It
  panicked with a `PanicException`, which `except Exception` does not catch; it
  raises `E405` or `E404`.
- **`has_ignore_region` agrees with `ignore_region`.** It answered `True` where
  the sibling ignore mask is empty, which ignores nothing.
- **`instances_from_masks` skips an empty mask.** It made an object of one,
  which `encode_instances` then refused; the class keeps its place in the
  numbering.
- **A transform chain whose steps do not meet is refused (`E501`).**
  `ChainTransform` and the engine's `Transform::chain` checked only units, so
  steps out of order were chained and points mapped along a path that does not
  exist. Each step must now start in the frame the step before it ends in, as a
  composite's components must.
- **Stacked annotations and displacement fields follow the grid's `chunk_hint`
  (§14.1).** The writer and `recompress --rechunk` chunked them apart from the
  images they label; each now takes the grid's spatial chunk, with 1 along every
  leading axis.
- **`recompress --profile portable` leaves nothing that needs a plugin, and
  `recompress_paths` tries every file.** A small dataset another writer stored
  under Blosc2 or Zstandard was copied as it was, and is now rewritten
  contiguous. One failure stopped a batch, and the files already rewritten went
  unreported; the error now names each failure, as `fix_paths` does.
- **`describe_filters` and `medh5 info` name Zstandard and unknown plugin
  filters.** They showed such a dataset as `chunked`, though a reader needs the
  plugin to open it; they now print `zstd:<level>` and `filter:<id>`.
- **`storage.dataset_kwargs` honours a `CodecProfile` of your own.** Only its
  name reached the engine, which refused it as unknown; the codec now comes from
  the profile itself.
- **Integrity and strict validation fail a file in which no digest parses.** An
  early return dropped the `E703`, `E702` and `W905` already found, so
  `medh5 validate --level strict` exited 0 on a file whose pixels had changed
  while `verify` failed it. The corpus gains `E703-every-digest-malformed`, and
  a test holds validation and `verify` to one answer on every digest case.
- **The validator sees a broken container at every level.** A `meta`, `grids`,
  `images` or grid of the wrong kind (`E004`, `E008`, `E109`), a member that
  leads nowhere (`E001`, and `E201` when no image resolves), and an image or
  pyramid level that is not numeric (`E001`) passed all four levels, strict
  included. Each is now reported, and five such files join the conformance
  corpus, which has 163 cases (it had 157).
- **One unreadable chunk no longer hides the rest of the integrity report.** It
  ended the pass, hiding every `E701` and `E702` after it and the name of the
  dataset; it is now `E001` at its own path and the pass goes on, and every rule
  reports an object it cannot read and checks the next.
- **The validator checks what it passed over.** A missing `task`, `closure` or
  `class_ids` (§6.2) is `E412` — a missing `closure` was read as `explicit` —
  and `/meta` and `clinical/meta` must be UTF-8 (`E004`, `E801`), where a lossy
  read put U+FFFD in place of a bad byte and the document parsed.
- **An atlas registration no longer silences `W911`, and an unparsed `/meta` no
  longer floods the report.** A transform with one end in a frame no grid is in
  counted as relating two timepoints, so one registration to an atlas hid `W911`
  for visits nothing relates; and with no document to resolve against, every
  reference to a timepoint or label was reported (`E409`, `E413`) beside
  `/meta`'s own finding.
- **`diagnose` and `fix` see a digest that does not parse.** `verify` failed a
  file holding one while `fix` found nothing to fix; `Diagnosis.malformed` now
  lists such digests, and `fix --rewrite-digests` restamps them. `fix_paths`
  attempts every file and raises one error naming each that failed, where it
  stopped at the first, and `fix --rebuild-index` keeps each entry's
  `max_coords`, `seed` and occupancy planes, where it rebuilt with the defaults.
- **The specification defines `content_id` and `label_set.sha256` to the byte.**
  Zero is written `0.0`, `blake2b` is unkeyed BLAKE2b-512, and the paths, the
  root line, the `/meta` line, the sort order and the form in which a label
  set's classes, relations and skeletons are digested are spelled out. Each is
  what the engine has always written, so no digest changes.
- **Format 1.0 and 1.1 agree on later minor versions.** A later minor may extend
  the attributes `content_id` covers, so a reader that cannot recompute one
  reports `W913`, and a reader must not refuse a file only because its minor is
  higher (1.0 §2.1 and §13.2, 1.1 §2.2).
- **The specification states the rules 2.0.2 enforces.** Implicit closure in the
  readers, the ignore region in tracking's coverage, numeric images, a 2-D
  grid's world axes, finite boxes, centres and affines, `E412` for a missing
  annotation attribute, a replace through a symbolic link, how a version is read
  (§2.1), a pyramid origin's tolerance (§4.3) and how a 0.x file migrates
  (Appendix B) are each written into the clause they correct, and §14.1
  recommends the chunk sizing the writer uses. 1.0 Appendix C records
  twenty-three corrections from the audit of 2.0.1, and 1.1 Appendix A four,
  among them that an asserting event need only be eligible (1.1 §7.3, §9.1).
- **task-cache-1 says what the reference does.** It defines `file:` URIs by RFC
  8089, where a patch is placed and how it is padded, the bounds that make a
  label positive or negative, the default `min_follow_up_us`, which censors
  nearly every row without a positive, the lease rule (`T302`), and the base and
  uniqueness of a cache's entry URIs (`T401`). §9 carries the code table inline.
- **`Tracking[key]` raises `KeyError` for any key no track has.** A key that is
  no instance id, such as `-1` or `"7"`, raised `OverflowError` or `TypeError`,
  which `get` and `in` already answered as absent.
- **A timepoint date added after a recorded shift is reported.** A file that
  records a date shift had its dates passed over by every later scan, so a date
  added since was never seen; a date whose `days_from_baseline` implies another
  baseline than the shifted dates do, by more than a day and a half, is now a
  finding.
- **A pseudonym under a UID key is no longer a finding.** Each was reported
  again, so a scrubbed file holding a UID could not scan clean.
- **Every name a scrub pseudonymises is recorded in `uid_map`.** Staff and
  organisation names a strict scrub replaced, and names replaced inside text,
  were left out of the map, the only key to the pseudonyms a file now holds.
- **Cohort statistics survive a NaN voxel.** One NaN or infinite value (PET,
  resampled data) made the moments NaN, and the minimum and maximum depended on
  the order of the files and could drop a finite extreme. Non-finite values are
  left out of the moments and counted (`Moments.nonfinite`), and the result no
  longer depends on file order.
- **`extract` and `unpack` never write a member over its own collection.**
  `extract(shard, key, out=shard)` replaced the whole collection with that one
  member and exited 0, through the same path or a relative or linked alias of
  it, and `unpack` did the same to a shard named for one of its members. Every
  destination is now checked against the collection before the first is written,
  in Python and in `medh5 unpack`.
- **`C201` counts samples, not claims.** A sample with two stale claims counted
  as two.
- **The cohort scan follows symlinked directories, matches `.MEDH5` in any case
  and skips AppleDouble `._*` files.** It passed over the first two and took the
  third for samples; each linked directory is walked once.
- **`clinical export -o` and `task reconcile -o` no longer destroy a sample.**
  `medh5 clinical export S -o S` wrote its JSON over the sample it read, and
  `task reconcile -o` its manifest over a sample the task names, each exiting 0.
  An output that is the file being exported or one of the task's sources — by
  the same path, through `..` or a link, or as the collection a member comes
  from — is now refused before anything is written. On Windows these checks,
  and those of `pack` and `migrate`, compared canonical paths, which a hard link
  gets past; each now compares the file's identity, as on POSIX.
- **`E805` blames the column whose row count is wrong.** A table's row count was
  its first column's, so a short `event_id` had every other column blamed and
  dropped; it is now the count most columns agree on, and the finding names the
  counts.
- **A DICOM SEG with several segments of one label imports all of them.** Each
  segment replaced the one before, so every lesion but the last was lost without
  a warning. Segments of one class are combined — their union, or their maximum
  in a `FRACTIONAL` SEG — and two segments with one label no longer fail `E302`
  when classes are minted.
- **No converter writes its report or output over a file it reads or writes.**
  `from-dicom-seg --report case.medh5` replaced the sample it had just amended
  with JSON and exited 0, and an output naming an input replaced the input.
  Every converter now refuses such a path before it runs, comparing paths
  resolved and as files, so `sub/../x`, a symbolic link and a hard link are each
  caught, and writes its report atomically.
- **A non-ASCII name no longer aborts an import or a migration.** `from_dicom`
  kept the Unicode letters of a `PatientID` in the sample id, `from_nifti` those
  of a file name and `migrate` those of a subject, which failed `E003`. Each is
  now folded to ASCII (`Müller Hans` becomes `Muller_Hans`), by the Python
  converters' one shared rule and by `migrate` alike, and `from_dicom`'s subject
  id keeps the original.
- **`to_rtstruct` declares the character set of its text, and colours each ROI
  as the label set does.** Text beyond ASCII went out as Latin-1 under the ASCII
  default, and every ROI was red; such text is now UTF-8 under `ISO_IR 192`, and
  each ROI takes its class's colour.
- **A slice that will not decode is refused by name.** A pixel decoder's failure
  was a raw traceback naming no file; it is a refusal naming the series and the
  file, and a series in a transfer syntax no installed decoder reads is refused
  before any of it is decoded.
- **`import_seg_nifti` reports what it guessed, and Analyze and MGH files are
  refused by name.** It dropped its reader's notes, did not take
  `assume_geometry`, and recorded a command that does not exist as its
  provenance tool; nibabel reads Analyze, MGH and MINC through the NIfTI call,
  and the first NIfTI field read raised a bare `ValueError`.
- **A review record written by the 0.x API migrates.** Its `annotator` and
  `timestamp` were never read, so every such file failed with `E005`, an
  activity with no agent, and a `pending` or `flagged` record became `reviewed`.
  Both now become `submitted`, a flag is kept as an issue and the report calls
  each mapping a guess, and a record naming no reviewer is attributed to the
  import tool.
- **A 0.x sample label, root `shape` and non-object `extra` survive migration.**
  The label was kept only when its name matched a mask or a box, and is now a
  class in the cohort's label set (Appendix B); the root `shape` of 0.x up to
  0.5 is read, and a non-object `extra` is kept as `extra.legacy.<tp>.extra`. A
  group that fails takes its report notes with it, where they described what was
  never written.
- **A non-ASCII date no longer crashes the grouping of a migration.**
  `parse_date` counted characters and then sliced bytes, which panicked inside a
  multibyte character.
- **Each migrated study keeps its own `sample_id`, and a subject named `Study-7`
  is named for itself.** A subject's studies, or keys differing only in case,
  shared one id; a sample's id is now its output file's name, unique in the run.
  A subject whose name began with `study` was named for its source file, as if
  it had no subject.
- **`migrate` never writes over a file it reads.** `-o .` wrote each sample over
  its 0.x original; a sample whose output is an input is now reported `failed`
  and not written, and `migrate()` raises. A `--report` or `--write-labels`
  naming an input, and a `--report` that is a directory or in a missing one, are
  refused before anything is written, and a report naming an output is not
  written.
- **`medh5 verify`'s text output names the datasets it could not check.** Each
  is an `UNCHECKED` line, as `--json` already listed them.
- **`index build` and `seg convert` leave a file they would not change
  untouched.** They rewrote it, copying every image and putting a new inode
  under every open handle. `index build` now keeps an entry that already holds
  what it would write, and `seg convert` to the stored encoding writes nothing,
  while a real conversion transcodes once and a kind it refuses is refused
  before the copy.
- **Command-line messages name flags, not Python arguments.** Where the engine
  speaks to the Python API — `rewrite_digests`, `rebuild_index`, `key`,
  `open_collection()` — the command line now says `--rewrite-digests`,
  `--rebuild-index`, `--key` or `medh5 unpack`.
- **`medh5.bench` refuses a run that measures nothing, and a negative count.**
  `repeats=0` gave rows of NaN and `patch=0` timed reads of nothing, in the
  engine, Python and the command line alike, and a negative `repeats`, `patch`
  or `warmup` raised PyO3's `OverflowError`; each is now a
  `MEDH5ValidationError`.
- **An unrelated annotation's ignore region no longer erases a clinical slot's
  supervision.** A task slot took the ignore region of every voxel annotation on
  its grid, so an organ annotation ignoring the whole grid removed every lesion
  positive from the loss mask. Only annotations that label or examined one of
  the slot's classes contribute to `ignore[slot]` now; across several classes it
  is still one conservative union.
- **`PairedPatchDataset` labels and samples each pair at its own visits.**
  `label=` served one pair's change label as the target of every pair, and
  `annotation=` failed on every pair that did not start at its visit, as the
  longitudinal guide's own example did. The label is attached only to the pair
  it spans, and the annotation is used only at its own visit.
- **A missing multichannel optional slot batches beside a present one.** The
  placeholder's channel count came from its own row, so rows with and without a
  multichannel image could not be collated, even with fixed patches; it now
  takes the channel count of the slot's images across the task, read once from
  grid metadata.
- **Warm and cold handle caches agree after a pinned source is replaced.** A
  warm lease kept serving the replaced file's old bytes while a cold one refused
  it (`T302`), so in one epoch one worker served the old version and another
  refused the row. Each new pinned lease re-checks its file with one `stat`: a
  replaced file is refused (`T302`), or reread when it holds the same version
  (task-cache-1 §6).
- **`HandleCache.clear()` and `worker_init_fn` close the handles their own
  process opened.** They abandoned every handle, as a forked child must, so in
  the process that had opened them each stayed open until exit.
- **A broken PyTorch install is reported with its cause, and MONAI metadata
  names the source file.** Every import error read as PyTorch not being
  installed. `filename_or_obj` is now set, so MONAI's `SaveImage` names its
  output after the sample.
- **A cache kept away from its task validates against it.** Relative source URIs
  resolved against the cache's directory, so a cache anywhere but beside its
  task had every entry stale (`T403`); with a task they now resolve against the
  task manifest's directory by default (`validate_cache`,
  `medh5 cache validate --base`).
- **An entry of a row whose preflight is in error is reported once, as `T404`.**
  It was `T406`, as if what the row admits were known, and the cutoff,
  fingerprint and pin checks ran on it as well.
- **`file:` URIs resolve as RFC 8089 defines them.** The prefix was only
  stripped, so `file://localhost/x` became the relative path `localhost/x`, a
  `%20` stayed in a name and `/C:/x` was joined onto the base; `localhost`,
  percent-escapes, Windows drive letters and UNC shares are now read.
- **A task manifest, a cohort manifest and a label-set sidecar are each saved
  whole.** `TaskManifest.save`, `medh5 task reconcile`, `Manifest.save` and
  `migrate --write-labels` wrote over the file in place, so a save cut short
  left it part-written; each now writes beside the file and renames over it. A
  failed save names the file, where `task reconcile -o` and
  `migrate --write-labels` said only "No such file or directory (os error 2)".
- **`with CacheWriter(...)` allows `commit()` or `abort()` inside the block.**
  It commits at the end only a writer still open, and aborts one whose block
  raised.
- **`Packed` columns index like Python sequences.** A negative index counts from
  the end, and one out of range raises `IndexError`, where `Packed[-1]` returned
  `""`.
- **A `SourceRef` built without an id gets one that tells files apart.** The
  default was the file's stem, so `a/case.medh5` and `b/case.medh5` shared one;
  it now names the parent directory too, in id characters and at most 128 of
  them.
- **A backport release no longer becomes GitHub's "Latest", or moves the
  Homebrew tap back.** The release script compares a tag with every final
  release, and an older one is published with `--latest=false` and leaves the
  tap alone.
- **A yanked crate version stops the publish step, saying what to do.** It read
  as unpublished, so a re-run met crates.io's refusal to take the version again,
  and `verify` called the crate missing.
- **Every commit merged to `main` is tested.** One concurrency group per branch
  let each new run cancel the one waiting; on `main` and on tags each commit now
  has a group of its own, and a pull request still cancels its older runs.
- **The specification's "HDF5 ≥ 1.10" and every declared dependency floor are
  tested.** An `hdf5-readers` job reads every valid corpus case, and samples
  under each Blosc2 profile, with h5py on HDF5 1.10, 1.12 and 1.14, value for
  value against the newest, and `test-min-deps` runs the suite with every
  declared floor installed.
- **A release page breaks no line of its notes mid-sentence.** GitHub prints
  each newline in a release's notes as a line break, and the notes are this
  file's sections, wrapped at 80 columns, so every release page broke each
  paragraph at every line. The notes now put each paragraph on one line and
  render as the section does; `python .github/scripts/release_notes.py vX.Y.Z`
  writes a published release's notes again, for `gh release edit`.

### Behaviour changes

- **Rust: three result structs gain a public field, and two key arrays lose
  entries that never matched.** `dataset::Moments` gains `nonfinite`,
  `dataset::ClassStats` `key` and `integrity::repair::Diagnosis` `malformed`, so
  a struct literal that names every field must add the new one.
  `curation::scrub::IDENTIFYING_KEYS` holds 76 keys (it held 78) and `DATE_KEYS`
  14 (it held 16): spellings with underscores, which no normalised key could
  equal, gave way to the normalised forms, so code that spells either array's
  length needs updating. Every public function keeps its 2.0.1 signature;
  `box_iou` and `box_iou_f64` answer NaN where no IoU exists.
- **A missing file or a directory is reported in fixed words.** The report was
  HDF5's error stack, which carries the time and an address, so
  `validate --json` differed from run to run; it is `E001`, ending "no such file
  or directory" or "it is a directory, not a file".
- **`add_segmentation(encoding="mask")` is refused (`E404`).** It merged every
  class into one volume written with empty `class_ids` and
  `annotated_class_ids`, and the file validated clean. `encoding` takes `"auto"`
  or one of `TRANSCODABLE`: the writer, `seg convert` and Python's
  `select_encoding` refuse any other preference (`E401` for a name no encoding
  has, `E404` for `mask`), and the Rust `select_encoding` returns a preference
  as given, as in 2.0.1.
- **A float mask holding a value other than 0 and 1 is refused (`E411`).**
  `add_mask`, `add_segmentation(masks=…, ignore=…)` and `InstanceInput` read 0.2
  as set; threshold the mask first, or pass a boolean array.
- **A commit refused before it changes anything leaves the writer open.** A
  refused commit — a sample with no image (`E201`), or a reference that does not
  resolve — aborted the writer, and a second `commit()` returned `None` as if
  the file had been written. The writer can now be mended and committed again,
  and committing an aborted one raises; in a `with` block a refused commit still
  aborts.
- **An amendment keeps the file's `digest_algo`.** A `sha512` or `blake2b` file
  was restamped in `sha256`, with a new `content_id`, or with `digests=False`
  left holding two algorithms; `fix --rewrite-digests` restamps in the file's
  algorithm too, where it failed such a file with `E701`.
- **`drop_clinical()` keeps the version of a sample without the clinical
  profile.** Only an amendment that removes the profile lowers the version (1.1
  §2.1); one with nothing to remove lowered it too.
- **A writer used from two threads at once refuses the second call
  (`RuntimeError: Already borrowed`).** `add_mask`, `add_contours`, `add_mesh`
  and `add_records` now release the GIL while they work, as `add_image` already
  did, so a second thread calling the same writer is refused instead of waiting
  for the first.
- **`SampleDocument.loads` refuses bytes that are not UTF-8 (`MEDH5SchemaError`,
  §2.4).** It decoded them with replacement characters standing in for what was
  there.
- **Indexing a string `Dataset` answers as NumPy does.** It indexed a flat list:
  `ds[:]` was a `list`, `ds[0]` of a 2-D dataset one element rather than its
  row, and a scalar gave its `str` for any key. `ds[:]` is now an object array
  of `str`, and on a scalar `ds[...]` is a 0-d array, `ds[()]` the `str` and any
  other key an `IndexError`.
- **`dense()` and `contains()` refuse class 0 (`E404`), as they refuse the
  ignore id.** Background is no class: a `labelmap` answered with every voxel no
  class holds, and every other encoding with none.
- **A coordinate, matrix entry or factor that is not finite is an error.** The
  encoders refuse a box or oriented-box coordinate that is not finite (`E406`),
  which passed, and mesh offsets that do not delimit their data (`E408`), which
  were found only at commit; an affine entry (`E504`) or a pyramid factor
  (`E105`) that is not finite is refused when written, and `box_to_slices`
  refuses such a bound (`E406`). The validator reports a non-finite box bound or
  `obb` centre (`E406`) or affine entry (`E504`) in a file, where a NaN centre
  validated clean at every level.
- **An annotation whose ignore id or class-id tables contradict its classes is
  refused when opened.** A stored `ignore_id` outside 1..65535 or equal to a
  class id, a `bit_class_ids` that is not one bit per class, and a
  `layer_class_ids` without one row per layer read as false verified negatives —
  an `ignore_id` of 0 marked all background ignored. Each is refused at open and
  reported by the validator (`E404`, `E405`).
- **`transcode_payload` refuses what the writer refuses.** It converted a dense
  payload to `instances` and dropped an in-band ignore region, both of which a
  write refuses, and `check_roundtrip` now compares the ignore regions as well
  as the class masks.
- **A geometric annotation's coordinate columns are read once, and read-only.**
  The same array comes back on every access, so writing into one raises; copy it
  to edit it. A score that is not finite reads as `None` on the objects an
  annotation yields.
- **The box helpers refuse a box or a block they would misread.** `box_corners`
  and `apply_affine_to_box` read a `(2, 3)` box of two corners as three made-up
  axes, and now hold a box to `(S, 2)`. `slices_to_box` refuses an open stop,
  which was a bare `TypeError`, and a negative bound, a step other than 1 or a
  stop before its start, each of which made a box of other voxels
  (`ValueError`).
- **`jacobian_determinant(roi=…)` and `folding_fraction(roi=…)` resolve a region
  as `read_field` does, and refuse one they cannot difference.** A negative stop
  was `E109`, a short ROI raised a points error, and a strided ROI was
  differenced as if contiguous: `::2` reported 100 % folding on a field that
  folds nowhere. A step other than 1, an axis of fewer than 2 voxels and more
  slices than spatial axes are a `ValueError`, and `cropped_grid` refuses a
  step.
- **A pyramid level's origin is held to a thousandth of its own voxel
  (`E105`).** It was held to 1e-3 of the origin's magnitude, so about 500 voxels
  from zero — whole-body CT, a microscope stage in µm — a missing half-voxel
  shift passed the writer and the validator.
- **A new file may be chunked differently, and no `content_id` changes.** The
  chunk budget is one CPU's share of the L3, not the whole cache the CPUs share,
  and no computed chunk passes 4 MiB, where a 96³ `float32` patch made 8 MiB
  chunks and a 160³ `int16` one 32 MiB. Digests cover decompressed content, so
  only the stored bytes change.
- **The conformance scorer compares severity.** A validator that reported every
  code as a warning, rejecting nothing, passed every case; an expected error
  reported as a warning now fails it ([Conformance](docs/spec/conformance.md)).
- **`rules_for` refuses a level it does not know.** It returned every rule for a
  misspelt level; Python raises `ValueError`, and the engine returns no rule.
- **`fix --rewrite-digests` refuses a file the validator rejects.** It restamped
  a structurally invalid file, a dangling `/images` say, which then passed
  `strict` with no code; the writer's commit gate now refuses it.
- **A file of another major is not read past its version, and a version is read
  one way everywhere.** Validation reports `E002` and stops, where this major's
  rules went on to judge a file that is not theirs, and a `medh5_version` that
  is not a string is `E001`. Whitespace and signs are no part of a version,
  which the support check trimmed while `open` refused it: before the major
  (`" 1.0"`, `"+1.0"`) they make another major (`E002`), and after it (`"1.0 "`)
  an unknown minor, a projection (`W913`) that `amend`, `recompress` and `pack`
  refuse.
- **A broken image is `E001`, not a `KeyError` or a `TypeError`.** An image
  member that links to nothing made every listing of a sample's images raise a
  bare `KeyError`, and an image stored as strings made `summary()`, `dtype` and
  `read()` raise `TypeError`. Both now raise `MEDH5ValidationError` with the
  validator's `E001`, and a name the sample does not have is still a `KeyError`.
- **Tracking no longer resolves an object an ignore region may hide.** An
  annotation with any ignored voxel no longer counts its classes as examined, so
  an object absent there is `unexamined`, not `resolved`, and a baseline that
  ignored the place no longer makes a lesion `new` (§7.4, §7.7, §11.3); the
  `medh5 track` legend says so. Where a review ignored only some slabs this says
  `unexamined` more often than it must, and never wrongly.
- **A class id with two meanings is refused, not merged.** Cross-file agreement
  scored `liver` in one file against `spleen` in another (a Dice of 1), and
  cohort statistics counted the two as one class in `class_weights`. Both now
  compare class keys and refuse a clash, naming the files (`C102` in agreement,
  as `dataset check` reports it); `ClassStats` carries each class's `key`,
  `medh5 dataset stats` prints it, and where neither side has a label set ids
  are compared as numbers.
- **Object agreement matches instance ids only within one sample.** Two raters
  who each number their lesions from 1 scored F1 = 1 for disjoint lesions;
  across files objects are matched by IoU, and a match by id reports its
  `threshold` as `null` in JSON.
- **Box IoU refuses boxes it cannot measure, and agreement a threshold outside
  (0, 1].** A 2-D box against a 3-D one scored about 10 and matched, and a box
  of no spatial axes scored 1; agreement and Python's `box_iou` refuse unequal
  or zero axis counts (`ValueError`) and a bound that is not finite or a `lo`
  above its `hi` (`E406`), and nothing is clamped. `compare` and
  `compare_instances` refuse an IoU threshold that is NaN, 0 or less, or above
  1, as `medh5 agree --threshold` does: NaN matched no object, and 0 or less
  every pair, objects that do not touch among them. `compare_instances` raises
  `E414` across dimensions, and `Grid.comparable_with` is false for grids with
  different numbers of spatial axes.
- **A subject in two folds of one split is a leak.** K-fold claims are `train`
  in every fold but one, so a subject in folds 0 and 3 read `train` twice and
  passed `medh5 splits`, `Split.leaks()` and `dataset check` (`C202`); each now
  reports it, naming the place with its fold, as `train (fold 0)`.
- **A date shift of 0, or one other than the recorded shift, is refused.** A
  shift of 0 was recorded as a shift, after which every later scan passed over
  the file's dates; a recorded 0 now reads as no shift.
- **`scrub` refuses options that only `--apply` acts on.** `--date-shift-days`,
  `--salt` and `--by` without `--apply` did nothing, and so did `scrub_paths`'s
  apply options without `apply_changes=True`. `scrub_paths` takes explicit
  keywords, so a misspelt `apply=True` is a `TypeError` where it ran a scan, and
  it processes every path, then raises the first refusal with `.reports` set
  (`None` for a refused path).
- **The cohort tools say when they did not see the whole cohort.**
  `medh5 dataset stats` printed the files it could not read among the numbers
  and exited 0; it names them on stderr and exits 1, with `--json` still one
  document, and `scan(on_error="warn")` now issues the warning it promised.
- **Saved statistics with a `null` moment are refused on reading.** A NaN moment
  was written as `null` and read back as a mean and standard deviation of 0, so
  normalisation silently became (0, 1).
- **`write_claims` refuses a manifest the split does not place.** Claims stamped
  with a split's digest were written for a manifest the split was not made from;
  a filtered manifest is still accepted.
- **Sampling refuses an annotation it cannot draw from.** A misspelt name drew
  `uniform` windows without a word, and `balanced` failed only when its coin
  chose foreground; an unknown name is refused before any draw, and so is a
  non-voxel annotation where a draw may need foreground. `PatchSampler` checks a
  setting changed after construction as its constructor does, and a
  `sample_stride` below 1 is refused.
- **Selecting from a later minor's projection raises `MEDH5VersionError`.**
  `Clinical.select` and `medh5 clinical select` reported `certified` for a 1.2
  file read as a projection, which a task's preflight refuses (`T306`); a later
  minor may define what changes the answer, so neither `certified` nor
  `provable` is this engine's to claim (1.1 §2.2).
- **An asserting event attests what was known at the cutoff, whatever a row
  reads (1.1 §7.3).** `asserted_by_event_id` counted only when the asserter
  passed the kinds, plan, unknown-time, window and limit filters, so under the
  default policy a segmentation of unknown time never attested its mask and
  `roi="eligible_instances"` fell back to the grid centre. An asserter now need
  only be eligible — available by the cutoff and current in its record — while
  the link's event and document endpoints must still be admitted, so more
  payloads are admitted.
- **Revision chains are checked whole (`E816`).** A version available before an
  older one could have been is caught across a version of unknown availability,
  where selection certified the newest version at a cutoff before its ancestor
  existed, and the writer's commit refuses it. A repeated `supersedes` link is
  one link, no longer refused as a branch and a merge, and a cycle no longer
  hides other records' problems.
- **An `Event` takes NumPy integers as times, and refuses a float or a bool
  (`E811`).** A NumPy integer was taken for a pair and failed to unpack, and a
  float or a bool in a time pair was truncated to an integer; a value that is
  neither an integer nor a pair raises a `TypeError` saying what a time is.
- **Every DICOM slice is held to where a regular stack puts it.** Each gap was
  held to 1 % of the spacing, which passed 4.5 mm of accumulated drift and
  refused positions rounded to two decimals; a slice must now lie within a tenth
  of the spacing, and at least 0.01 mm, of its place
  ([Import DICOM](docs/guides/import-dicom.md)). `SPACING_TOLERANCE` is that
  fraction, 0.1, and `read_dicom_seg` holds each plane to the same bound.
- **One unreadable series no longer aborts `from_dicom`.** A localizer or an RT
  Dose series refused the whole tree. A series that cannot be imported is left
  out with a `series` warning while the rest import, and the command exits 1; RT
  Dose is skipped with a note, multi-frame objects with a warning, and only a
  tree with nothing readable is refused.
- **A one-plane DICOM SEG is not given a 1 mm step.** Its
  `SpacingBetweenSlices`, else its `SliceThickness` (recorded as a decision), is
  the step, and a SEG stating neither is refused.
- **RTSTRUCT contours are read as their `ContourGeometricType` says.** Open
  curves were filled as areas, a flattened applicator marked its class examined
  with no voxels — a fabricated negative — and POINT ROIs vanished. POINT and
  open ROIs are left out with a warning, and crossing `CLOSEDPLANAR_XOR`
  contours, an ROI mixing types, and contours between or across slices or
  outside the grid are refused.
- **RTSTRUCT import uses each ROI's own frame of reference.** It applied the
  first frame the file listed to every ROI, so the result depended on the order
  of that list; ROIs in several frames are refused (`E414`) before the sample is
  opened.
- **`to_rtstruct` names each ROI for its class's display name, and warns where
  contours of one class cross.** It used the class key; a display name is used
  where every class has a distinct one that fits an `ROIName`, and the import
  resolves it. Crossing contours are a warning because medh5 means their union,
  and readers that combine an ROI's contours by exclusive-or cancel the overlap.
- **An export never turns an unexamined class into a verified negative.**
  `to_dicom_seg` wrote a segment for every class, and `to_nifti(class_key=)` a
  mask for any, so a class nobody examined came back examined and absent.
  `to_dicom_seg` writes only the classes the annotation examined, and both
  refuse a named class it did not examine.
- **A 2-D NIfTI plane off z = 0 is refused.** It was moved to z = 0, so two
  planes, or an image and a mask from different slices, could share one grid;
  save the plane as `(x, y, 1)` to import it as a one-slice volume.
- **A 5-D `(x, y, z, 1, n)` NIfTI is read as channels, not time.** Its timing
  came from the squeezed fourth axis, making a measured time series of it; an
  axis on `dim[5]` with nothing saying otherwise is a channel axis, recorded as
  a guess (pass `fourth_axis=` to say otherwise).
- **`to_nnunetv2` refuses a destination holding NIfTI files it would not
  write.** A re-export into one directory left removed cases, dropped channels,
  old file endings and a case moved to `imagesTs` still in `imagesTr`, leaking
  test into training, and `from_nnunetv2` brought the removed case back. An
  identical re-export is still allowed, and other files are left alone.
- **A converter's report is refused inside a tree it reads or writes.** A
  `--report` inside the `from-dicom` or `from-nnunet` input tree, or inside the
  nnU-Net dataset `to-nnunet` writes, is refused before the command runs.
- **Migration and NIfTI import write no local path into a sample.**
  `migrate_paths` gave a 0.x file naming no subject the `subject_id`
  `study:<its path>`, often an MRN or a name, and `migrate` took the file's stem
  for the subject; both now name such a sample for its stem, with the
  `subject_id` `study:<stem>`. Their import activity, and a NIfTI import's,
  names each input by its file name (`medh5-0.x:<name>`, `nifti:<name>`), not
  its path, so a sample's `content_id` no longer depends on where its source
  lay.
- **`migrate --write-labels` reports a file it cannot read, and exits 1.** It
  left that file's classes out of the label set and exited 0; it now warns
  (`unreadable`), and still writes the label set of the files it could read.
- **`medh5 fix` exits 0 only when nothing is left to fix, and `index build`
  exits 0 when there is nothing to index.** `fix` exited 0 with an index entry
  its `--rewrite-digests` left stale, or digests `--rebuild-index` could not
  touch, and now always prints its notes and counts malformed digests.
  `index build` exited 1, after rewriting the file, though an index is optional
  for every annotation (§14.3).
- **A file that failed is one shape of entry in batch `--json` output.** `fix`,
  `index build`, `labels check` and `splits` report it as
  `{path, ok: false, error}`, where `index build` gave `null` and `splits`
  printed `UNREADABLE` lines on stdout. `splits` names such files on stderr, in
  the command line's words, and no longer counts them as files without a claim.
- **The command line refuses a flag without its partner, and an IoU threshold
  outside (0, 1].** `fix --reason` and `--by` need `--rewrite-digests`, and
  `dataset split --fold` and `--assigned-by` need `--write-claims`, where each
  did nothing without it, and `agree --threshold` takes a number in (0, 1]. Each
  is a usage error (exit 2).
- **A bare `medh5` prints its help on stderr.** No command is a usage error, so
  its help goes where every usage error goes; `medh5 --help` still prints it on
  stdout.
- **The standalone binary hands commands only to a Python `medh5` of its own
  major version, and says why it cannot.** It handed 2.x commands to a 1.x
  package, whose command line has another grammar and whose converters write
  another format, and a broken install read as one not installed; it now prints
  Python's last line of error, or the version it found.
- **The torch loaders refuse, by name, what they misread.** A non-voxel
  annotation failed with `AttributeError` in every label format and in
  `medh5.monai.to_dict`, and `label=` naming an annotation that is not a
  classification, or `images=` not at `timepoint=`, were read silently. Each is
  now refused, naming the annotation or image.
- **A clinical slot whose images disagree in channel count is refused at
  construction (`T102`).** Where some rows leave the slot empty, no placeholder
  can match them all. `collate_clinical` raises `MEDH5ValidationError` (still a
  `ValueError`) for a shape mismatch, and says to give the slot a `patch` or use
  `batch_size=1`.
- **A cache manifest that lists an entry twice, or two entries for one event
  version or row, is refused (`T401`).** Duplicates could hide a wrong checksum.
  `CacheWriter` refuses them too, with entries that fail the cache schema, and a
  `cutoff_us` that is not an integer is refused (`T401`) where it was read as
  absent.
- **Validating a cache path that does not exist raises `MEDH5FileError`.** It
  was reported as a corrupt cache (`T401`), as an existing file that is no cache
  still is.
- **`build_document_cache` refuses a source whose content changed since its pin
  (`T302`).** It encoded the new content under the old pin; the writer is
  aborted, and no file is left.
- **Preflight's `label_annotations` lists only the annotations that label or
  examined one of the slot's classes.** These are the annotations the loader
  takes a slot's supervision from.
- **`TaskManifest.validate()` returns a schema failure as one `T101` finding.**
  It raised, where `medh5 task validate` reports the finding.
- **Saving a task manifest, a cohort manifest or a label-set sidecar replaces a
  read-only file, keeping its mode, as an amendment does.** Each was written in
  place, which a read-only file refused.
- **Release builds strip the symbol table.** It is about 5 MB of the 30 MB
  extension; loading the module and panic messages are unchanged, but a Rust
  backtrace loses its frame names.

### Performance

- **Window reads and attribute checks ask HDF5 less.** A windowed read queried
  the file's size once per chunk, under HDF5's lock, and now once per read;
  checking that an attribute exists listed every attribute name, and now asks
  HDF5 directly.
- **Writing `layers` measures the overlap graph once, and image reads copy
  less.** The writer analysed a `layers` annotation twice; `read(physical=True)`
  converts and rescales in one pass, and a read in the stored dtype no longer
  copies. Values are identical.
- **Encoding, transcoding, digests, index building, field sampling and point
  mapping release the GIL.** The annotation encoders and transcoding, the
  round-trip checks, the digests, the writer's `add_mask`, `add_contours`,
  `add_mesh` and `add_records`, the summaries, `tracks` and the index reads now
  let other Python threads run while they work, and so do `build_index` and
  `occupancy`, the displacement-field samplers, `encode_displacement`,
  `jacobian_determinant` and `folding_fraction`, `index_to_world` and
  `world_to_index`, clinical `records()`, the cache writer's `add` and `commit`,
  `dataset.find`, `Manifest.stale`, `lossless_as_int16`, `is_probability` and
  `read_document`. Results are identical; a writer used from two threads at once
  refuses the second call instead (Behaviour changes).
- **Label reads read each volume, layer or bit plane once.** `dense()` re-read a
  `labelmap` window for every class, and the index, `class_bboxes` and `analyse`
  walked each class over the whole volume: a 60-class `dense()` is about ten
  times faster, and `build_index`, `class_bboxes` and `analyse` 15 to 200 times.
  Repeated access to boxes, `polygons()` and `instance(id)` no longer re-read or
  decode every object, and every output is identical, `content_id` included.
- **A cubic displacement field is prefiltered once per transform handle, within
  a 256 MiB process-wide budget.** It was read whole and prefiltered at every
  evaluation, which in a paired loader meant once per item; past the budget,
  coefficients serve one evaluation and are made again, as before. Results are
  identical.
- **Digests decode each chunk once.** Their slabs did not follow chunk
  boundaries, so each chunk was decoded two to four times; the digests are
  identical.
- **Agreement and split audits scale with the data.** `compare_voxel` decoded
  the annotation once per class: at 40 classes over a million voxels, `layers`
  takes 0.38 s instead of 1.15 s and `bitmask` 0.38 s instead of 5.12 s. Finding
  anatomy units was quadratic, and 40,000 of them take 0.28 s instead of 2.43 s.
- **A scrubbed file is rewritten once.** `--apply` compacts the file at its
  commit instead of repacking it whole a second time.
- **Cohort statistics read chunk-sized slabs, and split lookups are cached.**
  Statistics read plane by plane, decompressing each chunk once per plane, and
  now give bit-identical results from whole chunks. `Split.partition_of` and
  `fold_of` rebuilt the split on every call and are cached, the cohort guide
  lists a partition with `split.paths()`, and pack and `dataset check` use hash
  lookups.
- **A clinical history is sorted without copying an event id per comparison.**
  The order is the same, and as stable.
- **A DICOM series is checked before any of it is decoded.** `read_series`
  checks every header first and decodes one slice at a time into the volume,
  where it held the decoded series about three times over before the first
  check.
- **The console script hands Python its output by line or by 64 KiB block.** It
  made one call per fragment; a stream Python flushes per line — stderr, one
  with `line_buffering` or `write_through`, a terminal — still gets a line at a
  time, and stdout is flushed before each write to stderr.
- **The torch `instances` format no longer decodes every object mask for each
  patch.** It reads the object columns instead, with the same output.
- **`CacheWriter.add_row`, task fingerprints and `len(FeatureCache)` stop
  rebuilding what they read.** `add_row` indexes event ids by position instead
  of building an `Event` per selected event, a `TaskManifest` computes its
  fingerprints once per change and every row's in one pass, and
  `len(FeatureCache)` counts entries without building them. Results are
  identical.

### Documentation

- **The front pages link format 1.1 and the task-and-cache contract.** The
  README and the documentation's home page link all three normative documents,
  1.0, 1.1 and task-cache-1, and a test holds them to it.
- **New samples are checked at `--level integrity`.** `--level strict` promotes
  `W903` and `W912`, which every fresh import carries, so the check the READMEs,
  the home page and the import, segmentation, detection and migration guides
  gave failed on every new sample; `strict` is kept for a dataset being
  published. The DICOM guide states the slice tolerance and the NIfTI guide that
  a 2-D plane lies at z = 0, and the converters reference says which RTSTRUCT
  contour types import.
- **The `medh5-sys` documentation no longer announces "three things" before a
  list of five.** It says what the crate links, and that the engine turns HDF5's
  dynamically loaded plugins off unless asked.
- **The writer's docstrings say what it does.** `Image.read(physical=True)`
  returns `float32` only for an image that declares a rescale and reads any
  other as stored, an amendment on Windows needs every reader of the file closed
  before it commits, and `written_version` names its `drop_clinical` exception.
- **The reference says what commits, closes and errors do.** A commit is an
  fsync, a rename and a directory fsync, following a link to the file it names,
  as the curation reference now says of an amendment; after a file is closed its
  views and reads raise while cached grids, headers and frames still answer; and
  a wrong argument raises the built-in error and a failed HDF5 read or write a
  plain `OSError`, as `MEDH5Error`'s docstring now says. The Python reference
  says that `c.select` refuses a later minor's projection, the curation
  reference gives the IoU threshold's range, the annotation kinds are counted
  rightly, and the Rust reference and the crate's README list every public
  module.
- **The annotations reference says what `closure` means to each reader.**
  `brats-subregions` uses the BraTS 2023 numbering, and its description now says
  so: in BraTS 2021 and earlier label 4 is the enhancing tumour, which the
  vocabulary numbers 3, and its own id 4 is a synthetic parent.
- **The segmentation guide and `seg convert --help` name both lossy
  conversions.** A `probmap` is thresholded, and instances converted to a dense
  encoding lose their ids, which takes `--drop-identity` (`drop_identity=True`).
- **The partial-labels guide agrees with its own example.** What it reads back
  matches what it wrote, and manifest coverage is keyed as `split.paths()` names
  a sample, `path::key` for a collection member.
- **`transform_between` between visits is described as it behaves.** It answers
  for the first pair of frames in grid-id order, so where visits share a frame
  the longitudinal and registration guides resolve between frames instead.
- **The performance guide and the storage reference state the chunk budget the
  writer uses.** It is 80 % of one CPU's share of the L3, and no computed chunk
  passes 4 MiB; the storage reference also says how a commit replaces a file
  through a link ([Storage](docs/reference/storage.md)). The performance guide
  reads `used_index` as a batch holds it, a list, and says that this changelog's
  2.0.0 figures came from an earlier 2.0 build.
- **`storage.dataset_kwargs` says that it changed meaning in 2.0.** 1.x returned
  `h5py` keywords and 2.0 returns `{"chunks", "codec"}` or `{}`; for `h5py`,
  pass `resolve_profile(p).codec(role).kwargs()`.
- **The conformance page says that severity is scored.** It also says that 1.1
  reclassifies some files a pure 1.0 validator reports as `E007` (`E009`,
  `E803`), and cites §2.2 for collections; the code table's summaries of `E004`,
  `E008`, `E109`, `E406`, `E504` and `E702` say what each now covers.
- **Tracking's states are documented with the ignore region.** `resolved` means
  examined at the visit, outside any ignore region, and not found, in the
  longitudinal guide and the Python reference as in `medh5 track`'s legend.
- **The de-identification guide states its coverage exactly.** Its suggested
  publication gate passed open identity findings, clinical samples and anything
  not examined, as not actionable; it blocks on them now
  ([De-identify](docs/guides/deidentify.md)). It also says how `scrub --apply`
  replaces a file: through a symbolic link it rewrites the file the link names,
  while the file's other hard links keep the original, identifiers and all, and
  must be removed.
- **The clinical guides run as written.** The setup command, amending a copy (an
  amendment in place broke the training guide's pin), selection policies with
  `context_us` and plans, and the encoder contract (`encode`, `dim`, `header()`)
  are corrected, with an encoder that runs, and the signatures they document are
  tested against the code.
- **The export guide and the design rationale no longer promise what does not
  exist.** The export guide promised lossless round trips to DICOM, which has no
  image exporter; it now lists what none writes — images to DICOM, boxes,
  keypoints, meshes, transforms and clinical data — what an existing nnU-Net
  destination does, and when an RTSTRUCT ROI is named for its class key rather
  than its display name (`roi_names`). The design rationale no longer promises a
  C interface or DICOM round trips.
- **The refusals page covers the converters' 2.0.2 rules, and the install page
  explains "filter plugins disabled".** The first adds the DICOM, RTSTRUCT and
  path refusals of this release, `E414` among them; the second says when
  `HDF5_PLUGIN_PATH` or `MEDH5_ALLOW_PLUGINS` is the way out
  ([Install](docs/install.md)).
- **The 0.x migration guide says how a 0.x file migrates.** It covers review
  records and sample labels, how samples are named, that no output replaces an
  input, that a file it cannot read is reported, and that a reviewed sidecar
  must keep its keys ([Migrate from 0.x](docs/guides/migrate-0x.md)).
- **The command-line reference covers what 2.0.2 changed.** Help and usage
  streams, long-option prefixes (kept from 1.x; spell options in full in
  scripts), the batch failure entry, rewrites through links, `verify`'s
  `UNCHECKED` lines, `fix`'s exit code and partner flags, the no-ops of
  `seg convert` and `index build`, `agree`'s threshold range, `bench`'s floors,
  `task reconcile --out`, and the scrub, cohort, tracking, converter-path and
  clinical rules are all there ([Command line](docs/reference/cli.md)). The
  command line's README names the Python the standalone binary hands commands
  to: `MEDH5_PYTHON`, else `python3`, or `python` on Windows.
- **The torch reference states the loaders' contracts.** It now covers voxel
  annotations only; the scope of a pair's `label=` and `annotation=`;
  `ignore[slot]` as a conservative union; a missing slot's placeholder and
  `patch=None`; that every new lease re-checks its source while the unpinned
  datasets can mix versions of a file across an amend; and `clear()`,
  `timepoint=` and `filename_or_obj` ([PyTorch](docs/reference/torch.md)). The
  longitudinal guide's `PairedPatchDataset` example runs as written.
- **`SECURITY.md` says how to report a vulnerability, and what is checked.**
  CONTRIBUTING's check chain matches CLAUDE.md's and runs everything CI gates,
  and its release runbook covers a yanked crate.
- **The documentation build is pinned, and an example whose gate closes fails
  the docs tests.** `docs/requirements.txt` pins pymdown-extensions and Markdown
  beside MkDocs and Material, since `--strict` checks the anchors they make, and
  the canonical URL is the one Read the Docs gives each version. The docs tests
  took any `SystemExit` for success, so an example that exited non-zero passed.

## [2.0.1] — 2026-10-11

**Documentation only.** The code is 2.0.0's; this release carries the new
install instructions and the corrected dashes to the pages PyPI, crates.io and
the documentation's `stable` version show.

### Added

- **An install page** ([Install](docs/install.md), after Home): one command for
  most people, then what each extra is for and what it installs, where a wheel
  exists and what building from source needs, the command line without Python,
  the Rust crate, the 1.x line, and what to do when something goes wrong.

### Changed

- **The README's install section starts from the task**: `pip install medh5`,
  a table of extras by what they connect to, then the binary and the crate.
  The platform details moved to the install page, which the documentation's
  home page and first tutorial now point to.

### Fixed

- **A dash renders as a dash.** The documentation, the READMEs and this
  changelog wrote over six hundred dashes as `---`, which GitHub, PyPI,
  crates.io and the site all print as three hyphens; they are `—` now, and a
  test keeps them so.

## [2.0.0] — 2026-10-11

**One format engine, three frontends.** The format is now implemented once, in
Rust: the [`medh5`](https://crates.io/crates/medh5) crate is the canonical
implementation of the specification — data model, HDF5 I/O, validation,
chunked access, compression, geometry and transforms, annotations, provenance
and integrity. The Python package is a layer over it (`medh5._core`, built with
PyO3), and the command line is a native binary over the same code. Python keeps
what is Python's: NumPy at the API, PyTorch and MONAI, and the NIfTI, DICOM,
RTSTRUCT and nnU-Net converters.

**Format 1.1: clinical context beside the images, and what was known when.**
The format gains one optional profile, `clinical`, and nothing else: a sample
may carry its subject's laboratory values, reports and their revisions,
diagnoses, medications and assessments on one subject clock, each event an
immutable version with when it happened *and* when it became available
([MEDH5 1.1](docs/spec/medh5-1.1.md)). A sample is still written at the lowest
version its content needs, so imaging-only data stays 1.0 and every 1.x reader
keeps reading it. Training gets a separately versioned companion contract —
task manifests and feature caches pinned to the sample versions they read
([task-cache-1](docs/spec/task-cache-1.md)) — and a dataset that builds
cutoff-aware multimodal batches from it.

**What a file is.** A sample is written at the lowest format version its
content needs. Imaging-only samples are still **format 1.0**: files written by
1.x and 2.0 are the same format, read by both, and a sample's `content_id` does
not depend on which wrote it. A sample carrying the clinical profile is
**format 1.1**; 1.x opens it and reads its imaging, and its validator reports
the profile it does not know (`E007`). The package version is a major one
because of the Python API at the HDF5 boundary — the package no longer hands
out `h5py` objects — and the behaviour changes below.

### Added

- **The Rust crates.** `medh5` (the engine), `medh5-sys` (HDF5 built from
  source and linked statically, C-Blosc2, and the Blosc2 and Zstandard HDF5
  filters — nothing to install) and `medh5-cli` (the `medh5` binary), for
  Rust 1.85.1 or later. Their version is the package's, from one place: the
  Cargo workspace.
- **The native command line**, from `cargo install --locked medh5-cli`, the
  binaries on each GitHub Release, or Homebrew. The `medh5` command `pip`
  installs runs the same Rust code, so both give the same output and exit
  codes. `migrate` is native; the converters (`convert …`) and the dataloader
  row of `medh5 bench` are the Python integrations, which the binary runs
  through a Python that has the package (`MEDH5_PYTHON`, else `python3`, or
  `python` on Windows).
- **Wheels** for Linux (x86_64 and aarch64, glibc 2.28 or later), macOS 11 or
  later (x86_64, arm64) and Windows (x64): one `abi3` wheel per platform serves
  CPython 3.10 and later. For MSVC targets `.cargo/config.toml` gives the C
  builds `-DNDEBUG`: cmake-rs drops CMake's release flags under the Visual
  Studio generator, and HDF5's assertions would abort the process on a damaged
  file instead of reporting it. A build that never reads that file —
  `cargo install --locked medh5-cli` from crates.io — is stopped by
  `medh5-sys` when the HDF5 it links was compiled without `NDEBUG` (read from
  HDF5's own `libhdf5.settings`), naming the variable to set — `-DNDEBUG`,
  which `cl` takes as it takes `/DNDEBUG` and no shell rewrites, where Git Bash
  turns `/DNDEBUG` into a path (`C:/Program Files/Git/DNDEBUG`) and CMake's
  compiler check fails; `MEDH5_SYS_SKIP_NDEBUG_CHECK=1` overrides. CI damages
  files for every platform's binary, and for such a build, and fails on an
  abort.
- **`THIRD_PARTY_NOTICES`**: the licences of everything the engine compiles in
  — HDF5, C-Blosc2 and its codecs, the HDF5-Blosc2 filter, every Rust crate
  and the Rust standard library — in the wheel's `.dist-info/licenses/`, the
  sdist and every binary archive; each crate carries `LICENSE`. The file is
  generated from `Cargo.lock` (`.github/scripts/third_party_notices.py`), and
  CI fails while it is stale. The wheels declare the licences of everything
  they compile in, where 1.x declared `MIT`, and so does the Homebrew formula:
  `MIT AND BSD-3-Clause AND BSD-2-Clause AND Zlib AND Apache-2.0 AND MIT-0`.
- **A release that fails partway can be finished** by re-running its failed
  jobs: each publishing step skips what is already published *and identical*
  to the run's build — on PyPI the wheels and the sdist, not the
  attestations Trusted Publishing writes beside them — publishes the rest,
  and stops on anything published from something else; a last job checks
  PyPI, crates.io, the release page and the tap. Each crate is verified from
  its package without `CARGO_REGISTRY_TOKEN` in the environment, where a
  dependency's build script could read it, and published once crates.io's
  index serves the crate before it. Release notes longer than a GitHub release
  takes (125,000 characters) are refused before anything is published, not
  once the packages are on PyPI and crates.io. In this repository a missing
  `CARGO_REGISTRY_TOKEN` or `HOMEBREW_TAP_TOKEN` fails its job instead of
  warning. `CONTRIBUTING.md` has the runbook, and what to set up before the
  first release to each registry. CI's one required check is `ci-ok`, which
  needs every other job.
- **The sdist** builds with maturin 1.9.3 or later, the first to write PEP
  639's `License-Expression` (CI builds it with exactly 1.9.3), and carries
  the workflows, release scripts, `mkdocs.yml` and docs hooks its own
  `tests/project` and the documentation read; CI runs those tests from the
  unpacked archive.
- **CI runs what ships, where it ships**: the test suite on the Linux aarch64
  and macOS x86_64 wheels too, `medh5.torch` on Windows, the engine's tests on
  Windows and macOS, `cargo package` for the three crates on every pull
  request, and each binary's hand-over of the converters to Python.
- **Types for the engine**: `medh5/_core.pyi`, checked against the built module
  by `stubtest` in the test suite, so `mypy --strict` users keep the 1.x
  signatures.
- `medh5.nodes` — `Group`, `Dataset` and `Attrs`, the package's views of
  stored objects (see below); `Sample.is_open`; `medh5.cli.command_tree()`, the
  command line's grammar as data; and an `h5py` extra (`h5py` and `hdf5plugin`)
  for reading a file with `h5py` directly.
- **The `clinical` profile** (`clinical/meta`, `events`, `documents`,
  `links`): a column layout of primitives — packed UTF-8, `int64` time
  bounds, `float64` values, validity masks — written, read, validated and
  digested by the engine. `SampleWriter.set_clock`, `add_event`,
  `add_document`, `add_link`, `add_records`, `drop_clinical`; `Sample.clinical`
  and `Sample.support`; the `medh5.clinical` module (`Clock`, `Event`,
  `Document`, `Link`, `ClinicalRecords`, `Selection`, `SelectionPolicy`,
  `select`, `augment`, `strip`, `imaging_events_from_timepoints`).
- **Strict prospective selection**, normative in 1.1 §9 and implemented once in
  the engine: only versions known available by the cutoff, the newest of each
  revision chain, *uncertifiable* when a later revision's availability is
  unknown, plans and unknown times excluded, tie groups kept whole, and only the
  payloads the selected versions attest. `latest_provable` is the one named
  alternative.
- **Task manifests** (`medh5.task/1`, `medh5.task`): sources pinned by
  `content_id` (files or collection members), subjects split before rows,
  modality slots, targets with explicit censoring, task / manifest / row
  fingerprints, and a preflight that reports every row as eligible,
  uncertifiable, excluded or in error, with the reason. Fragmented subjects are
  checked for identity, clock and duplicate agreement, and reconciled.
- **A preflight scales with the cohort, not with rows times history.**
  Subjects are taken one at a time (their files closed before the next's open),
  each subject's history is prepared for selection once and shared by its rows,
  each source is walked once and its clinical tables read once — validated,
  then used — and only events or documents that several fragments hold are
  digested or compared. The result crosses into Python as columns:
  `Preflight.subjects` holds each subject's merged history once
  (`SubjectHistory`, `EventTable`, `LinkTable`, packed strings), and a
  `RowView` is a view over arrays (`selected`, `order`, `tie_group`, `plan`)
  whose `events`, `selection` and `slots` are built when read; the columns are
  built as each subject is handed over (`preflight_each`), so the cohort is
  never held as records. On a 60-subject, two-site cohort of 720 rows
  (`bench_preflight.py`): 45–61 s and 5.2 GB peak → 3.4 s and 131 MB; the
  engine alone 10.0 s → about 4 s. At 200 subjects and 4 800 rows: 18.8 s and
  481 MB. The command line takes the subjects one at a time too
  (`write_preflight`, `preflight_each`): at 200 subjects its table went from
  18.6 s and 742 MiB to 13.5 s and 34 MiB, and `--json` — 1.6 GB of output,
  written a subject at a time — from 130 s and 12.3 GB to 70 s and 315 MiB.
  Validating a cache against a task keeps only what each row admits
  (`Admitted`).
- **Feature caches** (`medh5.cache/1`, `medh5.cache`): event- and
  patient-level features with a checksummed dependency manifest, validated so
  that *stale* (a source changed, T403) is told apart from *corrupt* (T401,
  T402: a manifest missing what its level requires, a payload that is not the
  declared dtype and shape) and from *inadmissible* (another task, row, cutoff
  or subject's sources, T404; fitted on another split, T405; encoding versions
  a row cannot have read, T406). A source pin also requires every dataset the
  sample's `content_id` speaks for to carry its digest (T302), so a column
  added without one is not read under an unchanged pin.
- **`medh5.torch.ClinicalTaskDataset`** and **`collate_clinical`**: one item per
  admitted row — slot windows on each image's own grid, the admitted event
  history, document features from an encoder or a cache, and the target —
  with padding, modality availability, field of view, annotation coverage and
  target observation kept as distinct masks. `ConceptVocabulary` is fitted on
  the training partition and records it. Both take only the manifest's own
  preflight (T404), and document features only from an event-level cache
  fitted on the training split. The vocabulary keeps each concept's text
  values (`value_index`: `female` and `male` are different inputs) and the one
  unit its statistics are in: a concept with training values in two units is
  refused at fit time, and a value in another unit is flagged (`unit`) rather
  than normalised with statistics it does not share.
- **Batches keep time's uncertainty.** Every time an input carries — an
  event's effective start and end, its availability, an image's acquisition
  — is given as its bounds, as ages before the cutoff (`start_age_h`,
  `end_age_h`, `available_age_h`, `(…, 2)`), each with a `*_known` mask; a
  day-precision time keeps its day. Events also carry `status`,
  `temporal_type` (a `static` fact is not an unknown time), `comparator`
  (`< 5` is not 5), `missing` (a result not done is not a negative),
  `tie_group` and `plan`. Images carry `image_time`. The rules are task-cache-1
  §6's.
- **Document text is read on demand.** Opening the profile, selecting, a task's
  preflight and the validator read the text column's offsets, never its bytes
  whole: a document's text is read — and checked as UTF-8 — when it is
  asked for (`Clinical.text`, and `Sample.document_text`, which reads neither
  the events nor any other document), and the validator and the pin check
  stream the buffer a bounded slab at a time.
- **Commands**: `medh5 clinical show | select | export | augment | strip`,
  `medh5 task validate | preflight | reconcile`, `medh5 cache validate` — in
  the native binary and the Python console script alike.
- **Diagnostic codes** `E011`, `E801`–`E819`, `W913`, `W914`, and 36
  conformance cases. The published suite also carries
  `medh5-clinical-1.schema.json` and the task and cache schemas.
- **Diagnostic codes `E506` and `E704`**, and the cache code `T407`: a
  transform whose `units` are not its grids' (§10.1), an object `content_id`
  covers reached at a path that is not its own (§13.2), and an event-level
  cache entry whose document its event's version does not own (task-cache-1
  §7.2). Four conformance cases come with them; the corpus is 157 cases.
- **Landmarks carry `instance_ids`** (`add_points(instance_ids=…)`, §8.5), as
  §10.6 asks of points that mark trackable objects, and tracking joins them.
  `Tracking.skipped` names the annotations with `instance_ids` that no track
  joined, and `Observation.volume_mm` is a volume in millimetres.
- **`SampleWriter.remove_transform`**, as `remove_annotation` is for
  annotations: an amendment had no way to drop a transform, so a file holding
  one the validator rejects could not be amended at all. What still names a
  removed transform — another's `inverse_id`, a composite's `components` —
  is checked at commit, and one added again under its name keeps them.
- **`RecompressResult.kept`** (`KEPT` in `medh5 recompress`): the datasets
  copied as stored because a fresh write of their dtype would not be their
  stored type — an enumeration, a committed datatype.
- **`RecompressResult.unverified`** (`UNCHECKED` in `medh5 recompress`,
  counted in its `FAILED`) and **`VerifyResult.unverifiable`**: what the check
  of a recompressed output could not read, reported with `verified` false
  where it was raised after the output had replaced the input; and the
  datasets carrying a digest their type cannot have (§13.1: variable-length
  data other than strings, references), neither compared nor counted as
  mismatched.
- **`to_nnunetv2(unlabeled=…)`** (`--unlabeled refuse|test`), **`frame_salt=`**
  (`--frame-salt`) on every DICOM SEG and RTSTRUCT conversion, and
  **`medh5.task.concept_token`**, below.
- **Runnable examples**: `clinical_longitudinal.py` (the worked example: write,
  reopen, validate, preflight, batch), `clinical_collection.py` (a subject split
  across two members of a shard, reconciled, loaded through workers),
  `clinical_cache.py` (every way a cache stops being usable), and the
  measurements `bench_clinical.py` and `bench_preflight.py` (a large
  multi-site cohort; [Runnable examples](docs/examples/index.md)).

### Behaviour changes

- **Views, not `h5py` objects.** `Sample.root`, `Image.dataset`,
  `Annotation.group`, `Transform.group`, `SamplingIndex.group`,
  `SampleWriter.handle` and the writer's `add_*` return values are
  `medh5.nodes` views. The operations a caller used still work — `ds[...]`,
  `numpy.asarray(ds)`, `shape`, `dtype`, `chunks`, `attrs`, `group["a/b"]`,
  `in`, `keys()` — but `h5py`-only API (`.id`, `create_dataset`,
  `visititems`, …) is gone, a string-list attribute reads as a `list` of `str`
  rather than a NumPy object array, and an array of a dataset is always a copy
  (`numpy.asarray(ds, copy=False)` raises `ValueError`, as `h5py`'s does). For
  anything else, open the file with `h5py` (`pip install "medh5[h5py]"`); it is
  plain HDF5.
- **Functions that took an `h5py` object take a view**:
  `medh5.integrity`'s digests and verification, `medh5.storage`'s
  `describe_filters`, `is_bulk` and `read_indices`, `read_document`,
  `read_grid`, `read_grids`, `collect_digests`, `AnnotationHeader.read` and
  `TransformHeader.read`.
- **Closing a sample closes what was read from it**, as `h5py`'s
  `File.close()` does: a view keeps the values it read while open — ids, an
  annotation's or transform's header — and anything that would read the file
  raises `MEDH5FileError` ("the file this object was read from has been
  closed"), instead of answering from a stale handle, keeping the file open and
  locked, or giving a wrong answer (a shape of `()`, a name of `''`) or a wrong
  error (`E410`, a bare `OSError`). A member got from `Collection[key]` does
  not own its shard: closing it (`with collection[key] as s:`) releases that
  member alone, where it closed the shard, and `Collection.close()` closes
  them all.
- **A region of interest is one slice per spatial axis, without a step** (the
  annotation readers — `dense`, `labelmap`, `ignore_mask`, `read_layer` —
  `Sample.ignore_region` and `valid_region`, `read_field`, …): an integer `v`
  was read as `slice(v, v+1)`, so `-1` read nothing, and a step was dropped,
  so `dense(roi=(slice(0, 4, 2), …))` returned four planes where `Image.read`
  returns two. An integer is a `TypeError` naming the slice that reads it
  (`slice(-1, None)`), a step other than 1 a `ValueError`. `valid_region`
  resolves a negative bound and clips to the grid as the annotation readers
  do, where it returned a region larger than the grid.
- **Casts saturate.** `Image.read(dtype=…)` and every engine conversion take a
  value the target cannot hold to the nearest one it can — 300 becomes 255
  as `uint8`, −1 becomes 0, and NaN 0 — where NumPy's `astype`, which 1.x
  used, wraps them to 44 and 255.
- **`labelmap(dtype=…)` refuses a dtype that cannot hold its values**
  (`ValueError`): a narrower one saturated, so class 300 and the ignore id both
  read as 255 in `uint8`. The dtype must hold every class id and, where a voxel
  is ignored, the ignore id; `bool` is refused (`dense()` gives boolean
  planes).
- **An annotation is checked when it is opened**: columns out of step, a mask
  larger than its grid or a voxel volume of the wrong shape raise when
  `Sample.annotations` opens it, as a malformed header always did, rather than
  when a reader reaches the bad row; one such annotation makes the whole
  mapping raise.
- **NumPy is the only runtime dependency.** `h5py`, `hdf5plugin` and
  `jsonschema` are no longer installed; HDF5, the filters, the JSON Schema
  validator and the interpolation SciPy did are compiled into the engine.
- **Off the wheels' platforms, installing builds the engine.** musl, an older
  glibc, Windows on ARM, PyPy and a free-threaded CPython get the sdist, which
  needs Rust 1.85.1 or later, a C compiler, CMake and network access to
  crates.io, where 1.x was pure Python. `pip install "medh5<2"` keeps the 1.x
  line, which writes format 1.0.
- **Random draws come from the engine.** Patch sampling, foreground draws and
  index subsampling use one small generator (PCG-64) in the engine, so a seed
  fixes the draws from Python, Rust and the command line alike. `rng=` takes
  an `int` from 0 to 2⁶⁴ − 1, a sequence of them — `PatchDataset` passes
  `(seed, epoch, index)` — or `None` for fresh entropy; a seed outside that
  range is a `ValueError`, where `2**64` drew `rng=0`'s patches. A
  `numpy.random.Generator` still works, and now supplies one seed (advancing)
  rather than being drawn from. The streams are not NumPy's, so a seeded 1.x
  pipeline picks different patches under 2.0, from the same distributions.
  No file depends on it: `index/` is outside every digest and `content_id`
  (§13.1), and its `seed` attribute still reproduces the subsample.
- **One module per area, not a package of re-exports.** `medh5.geometry`,
  `medh5.labels`, `medh5.integrity`, `medh5.storage`, `medh5.transforms`,
  `medh5.validate`, `medh5.conformance`, `medh5.cli` and
  `medh5.annotations.voxel` are modules where 1.x had packages of files that
  now only re-exported the engine; `medh5.curation`'s record classes
  (identity, provenance, quality, timeline, tracking) live in the package
  itself, `medh5.writer` is `medh5.sample`, and `medh5.io`'s private 0.x reader
  is `medh5.io.legacy`. Every public name kept its place one level up —
  `from medh5.geometry import Grid` and `from medh5.storage import recompress`
  work as before — while a deep import drops its last component:
  `from medh5.geometry.grid import Grid`,
  `from medh5.transforms.apply import jacobian_determinant` and
  `from medh5.labels.registry import load` become `medh5.geometry`,
  `medh5.transforms` and `medh5.labels`. Each class's `__module__` names its
  new home, which is where `pickle` looks for it.
- **`medh5.cli.main()` returns the exit code** for every outcome — `--help`
  and `--version` return 0 and a usage error 2 — instead of raising
  `SystemExit` for those three. Only the console script exits.
- **Every usage error exits 2**, as documented, where these exited 1: a
  command group without a subcommand (`medh5 seg`), `clinical select` with
  both cutoffs or neither, `recompress --out` with several inputs,
  `scrub --pseudonymise-ids` without `--apply` or `--salt`, and
  `fix --rewrite-digests` without `--reason`. So is an argument that is not
  UTF-8: a Latin-1 file name from a shell glob ended the binary in a panic
  (exit 101) and the console script in a traceback.
- **A count is held to its least value, not clamped to it**:
  `index build --max-coords` must be at least 1, and `build_index` refuses
  `max_coords=0` — a negative value or 0 wrote an index caching no
  coordinates, from which every foreground draw failed, and which the sampler
  now scans past. `--occupancy` must be at least 0 (0 builds no map),
  `dataset stats --stride` at least 1 and `--workers` at least 0, and
  `bench --patch` and `--repeats` at least 1; `dataset stats --image` or
  `--annotation` naming an id no sample has fails, where it printed an empty
  table.
- **A drifted conformance suite fails the score** (`medh5 conformance score`,
  exit 1, `"ok": false`): a score over files that are not the published ones
  is not a score. `--json` lists them under `changed`, where a warning was
  printed before the document. The published suite's README says the score
  fails, where it said it warns.
- **`E004` quotes the engine's JSON parser** ("key must be a string at line 1
  column 2") rather than Python's `json` module.
- **A NaN or infinity in the sample document is refused** with
  `MEDH5ValidationError`. JSON has neither, and §2.4 requires `/meta` to be
  JSON; 1.x wrote Python's `NaN` and `Infinity` tokens, which no JSON parser
  outside Python accepts. A file 1.x wrote that way still opens — the tokens
  read as `None` — and now validates with **E004**. An *attribute* may still
  hold one, and hashes exactly as 1.x hashed it.
- **A JSON value without one canonical spelling is refused.** A set or
  frozenset in a document was written in hash order, so `content_id` changed
  with `PYTHONHASHSEED`: it raises the `TypeError` `json.dumps` raises, as an
  iterator does. A `Decimal` NaN or infinity, which became `null`, and an
  integer outside −2⁶³ … 2⁶⁴ − 1, which became a rounded float, are
  `MEDH5ValidationError` (store such an integer as a string); every 64-bit
  integer is kept exactly.
- **A record or label set given to a writer is stored as it stands**
  (`w.document.identity = rec`, `w.label_set(ls)`): an edit to `rec` or `ls`
  afterwards reached the file in 1.x, and no longer does. Edit what the
  document hands out: `w.label_set(ls)` returns the document's set, not `ls`.
  An `extra` member named after a field of its record
  (`identity.extra["sex"]`), which 1.x wrote over the field, is refused.
- **`model` is not an agent type** (`E603` when the agent is made): the schema
  never allowed it, so such an agent was accepted and its document failed at
  commit (`E005`). Record a model as the `software` that runs it, as §11.1
  records nnU-Net.
- **A class id below 0 or above 65535 is `E306`** — in a label set, in the
  guard every encoder runs and in the writer's annotation header check
  (`encode_labelmap({70000: m})`, `annotated_classes=[-5]`) — as §15.2 has
  it, where it was `E303`, which is a reserved id (0 or 65535) given to a
  class.
- **`FORMAT_VERSION` is `"1.1"`**, the newest version the package writes. A
  sample without the clinical profile is still written as `1.0`.
- **A later minor is a projection.** A file of a minor version newer than the
  package implements opens, lists and validates as what it knows, with every
  unknown item reported as `W913` — an unrecomputable `content_id` too,
  rather than `E702` — and is never amended, recompressed or packed
  (`MEDH5VersionError`). An unknown value in its `/meta` (`sex: "X"`, a new
  activity `type`) is set aside or read as `other`, and a grid's unknown unit
  or axis kind is `W913` too: one such value stopped every document rule, so
  a planted `E107` went unreported and the file passed. An annotation's
  unknown `task`, `closure`, `space` or `scope`, and a transform's unknown
  `vector_space`, are `W913` to the validator as well, where they failed the
  file; at a version the package implements they stay `E412` and `E502`.
  `Sample.document` and `read_document` read `/meta` through the same
  projection, where they refused (`E005`) what the validator passed; a
  required member holding a value nothing known can stand in for (an agent's
  `type`) leaves the document unreadable, under `W913`, naming the member.
  1.0 §16's "accept a higher MINOR" is corrected accordingly (Appendix C.1).
- **A later minor's objects are listed as its projection.** A grid,
  annotation or transform holding a value only that minor defines — an axis
  kind; an annotation's kind, task, closure, space or scope; a transform's
  kind or a field's vector space — is left out of `Sample.grids`, `.images`,
  `.annotations`, `.transforms` and `.index`, and so is what depends on it: an
  image or annotation on a grid left out, a field sampled on one, a composite
  of a transform left out, an index entry of an annotation left out. Looking
  one up raises `MEDH5ValidationError` coded `W913`, naming the value, and
  `summary()["left_out"]` and a `left out` line in `medh5 info` name them all.
  One such object failed whole listings (`E110`, `E401`, `E412`, `E502`), and
  `summary()` and `medh5 info` with them, though the validator passed the
  file. A defect met before the value is still the error it was (a grid's
  shape, `E109`).
- **An unknown profile refuses an amendment** (`E007`) before anything is
  written, instead of being re-derived away.
- **`--profile` and `profiles=` add to the profiles a file declares**, where
  they replaced them: a clinical sample validated with `--profile seg` failed
  `E803` and skipped its clinical rules. An unknown profile name is refused
  (`ValueError`, or a usage error from `medh5 validate`) rather than passing
  the file, and `E009` no longer calls a profile only the caller requires
  "declared".
- **What a profile requires is checked** (`E009`, §1.3, Appendix C.1): `reg`
  a transform whose `from_frame` and `to_frame` are each one of the sample's
  grid `frame_uid`s, `curation` a provenance activity — agents alone do not
  count — beside `quality` on every annotation, and `multiscale` every image
  a §4.3 pyramid; `det` an annotation whose `task` is `detection`, of any
  kind, and `training` an index with an entry, a stale one being `W905`. §1.3
  said "resolvable endpoints" and "a provenance graph", which no validator
  could check. The writer infers `reg` and `curation` by the same tests.
  **A file 1.x wrote can fail `E009`**: it declared `reg` for any transform and
  `curation` for any provenance record; a no-op `medh5.amend(path)` re-derives
  the profiles and drops the claim.
- **A collection declares its newest member's version** (`E011` when lower).
- **The torch handle cache is keyed by `(path, sample_key)`**: `open_cached`
  and `HandleCache.get` / `lease` take an optional member key, so collection
  members are cached, leased and abandoned across a fork like files.
  `lease(..., content_id=...)` reads the version a task row pins: a handle
  cached from before its file was replaced is reopened — for that lease
  alone while another lease is still reading the old one, rather than refused
  — and a changed source refused (T302).
- **A `document` event owns at most one document** (`E815`, 1.1 §6, recorded
  in Appendix A): two texts under one event version would share its
  availability and revision chain, and an event-level feature would name
  neither — `build_document_cache` failed on such a file.
- **`medh5 task preflight --json` lists its members in the order they become
  known**: the fingerprints, `subjects`, `rows`, then `ok`, `counts` and
  `findings` (`Preflight::to_json` likewise). The content is unchanged.
- **`validate_cache` takes what a preflight admits** (`Admitted`, built a
  subject at a time by `Admitted::preflight` or from a `Preflight` by
  `Admitted::of`) instead of the whole preflight.
- **A damaged source is a finding, not a failed preflight**: clinical bytes
  that cannot be read fail the pin (T302) and the table check (T306), and an
  image or annotation that cannot be read is its subject's T306, with that
  subject's rows in error. Preflight reads object tables and no voxel:
  centring an `eligible_instances` crop decoded every mask, so one mask that no
  longer decompressed failed the whole cohort's preflight.
- **Only an imaging version fills a slot** (task-cache-1 §3.5): an image some
  other kind of event `describes` is not the slot's.
- **A row's fingerprint determines its inputs, and a row feature pins it**
  (task-cache-1 §3.3, §3.5, §7.2): a slot tie between fragments — one
  reconciled imaging version, each fragment's own image — went to the source
  the manifest listed first, and the merged version came from it, while the row
  fingerprint sorts the pins, so reordering two sources changed a row's input
  under one identity. Ties now go to the smallest pinned `content_id`, then
  image id. A patient-level cache entry names the fingerprint of the row it was
  built for (`row_fingerprint`, required by `medh5.cache/1`), and validation
  refuses an entry for another version of its row (`T404`): a source added to
  the subject changed what a slot read with the cutoff and the admitted
  versions unchanged, and the old feature still validated.
  `CacheWriter.add(row_fingerprint=…)`, and `CacheWriter.add_row(row, values)`
  pins a preflight row's id, fingerprint, cutoff and admitted versions. An
  event-level entry names no row, cutoff or versions (`T404` from the writer,
  `T401` in a manifest): a lookup by event version served one declaring a later
  row's history to every row.
- **A vocabulary is held to the cache's `fitted_on` comparison**, the split's
  `set_id` included (`T405`): one fitted under another split of the same
  subjects passed. `medh5.cache.fitted_on_mismatches` is that comparison.
- **A dataset reads the cache it validated** (`ClinicalTaskDataset`,
  task-cache-1 §8): validation reports the `manifest_digest` and `level` it
  read (`CacheReport.manifest_digest`, `.level`), and every later open of the
  cache — the first lazy read, a forked worker, an unpickled dataset — is
  held to them (`T404`). A cache rebuilt at the same path for another patient
  whose rows share row ids was read under the first one's verdict, pairing one
  patient's images with the other's features.
- **A row's ages are computed in floating point** (`ClinicalTaskDataset`): the
  difference of two `int64` microsecond times wrapped at the type's ends, so a
  584,000-year span read as a small negative age.
- **A pin covers a column however it is linked** (task-cache-1 §2): which
  datasets an object holds was decided by the path a walk of the root first
  reached them at, so an undigested clinical column linked first at the root
  — an alias sorting before `clinical` — was no clinical dataset, and the
  shallow and deep checks and a preflight passed it while the rows read it.
  Each attested group is walked on its own, and the shallow check re-reads
  every object reachable through `clinical/`. Coverage is by object, not by
  path: a dataset is covered only if it is one the root's digests list, so a
  column or a transform reached through a soft link, or through a second hard
  link from under `index/` — which no digest covers — fails the pin (T302)
  and is listed by `verify()` as unattested, where its value changed under a
  pin that the shallow and deep checks and a preflight all passed. The
  validator reports such a dataset's own digest, which no root line covers, as
  `E702` (§13.2, Appendix C.1), and the attestation walks follow soft links,
  as a reader does.
- `medh5.errors.Domain` includes `"clinical"`.
- **`recompress` keeps a file's link graph.** A soft link is copied as a soft
  link and a second hard link to an object as a link to its copy, where 1.x
  followed a soft link and copied its target again under the link's name. Every
  walk visits an object once, at its first path, as 1.x did; a group nested
  more than 64 levels deep is refused rather than followed, so no arrangement
  of links can exhaust the stack.
- **Where a file's identity cannot be read, every open is checked for outside
  references** (`MEDH5FileError`, §2). POSIX keys the memo by device and inode;
  off POSIX the engine had only size and modification time, which a crafted
  file can match, so nothing is remembered there (1.x keyed Windows on the
  volume serial and file index).
- **`scrub` does not de-identify the clinical profile, and says so.** Its text
  is packed UTF-8, which the sweep does not read, and a date shift moves
  `/meta`'s dates, not the clinical clock (1.1 §3, §6). A scan of a sample that
  carries the profile, or a `clinical` group it does not declare, reports it as
  not examined — never clean — and `--apply` refuses the file, before
  writing anything, rather than write an attestation the records would
  contradict; `medh5 clinical strip` writes the imaging projection to share
  instead.
- **A split ratio is a finite number of at least 0** (`MEDH5ValidationError`):
  1.x accepted a negative share and wrote it into the split.
- **A k-fold split has no more folds than groups** (`make_splits`):
  `--k-folds 1000000000` aborted the process allocating the empty folds.
  `ratios` given with `k_folds`, which were recorded unchecked, are refused,
  and a k-fold split records none. `leaks()` compares folds as well as
  partitions.

- **A NIfTI mask added to a sample must sit on its grid** (`import_seg_nifti`):
  spacing, origin and direction are compared with the grid's, lengths in
  millimetres, and a mask from another grid is refused. Shape alone admitted a
  mask translated, rotated, mirrored or rescaled onto voxels nobody drew on, with
  at most a warning about spacing. `assume_aligned=True` takes the voxels anyway
  and records a guess; `coord_system=` may only repeat the grid's.
- **NIfTI volumes share a grid in units and frame times too** (`from_nifti`,
  `from_nnunetv2`): lengths compare in millimetres, so the same grid stated in
  metres converts, and frame times two volumes both state must agree — the
  first file's times stood for every other's. A volume that states times lends
  them to one that does not.
- **NIfTI mask names become class keys a sample allows** (`from_nifti`,
  `import_seg_nifti`): `Liver` or `GTV-1` were taken as keys as given and
  failed the document schema at commit (`E005`). The key is the name as a key
  allows it (`gtv_1`), the display name the name as given, and two names that
  make one key are refused.
- **A conversion reports what it assumed**, as guesses, which leave the exit
  status alone: `identity` where no subject identity was established
  (`from_nifti` without `subject_id`, each case of `from_nnunetv2`), and
  `units` for a NIfTI file stating no spatial unit — `unknown`, the header's
  default and what an export of a `px` grid writes — which is read as
  millimetres, as it was without a word. `from_nnunetv2` reports that once per
  dataset, and the reader's sform and qform guesses, which it dropped.
- **A DICOM SEG must sit on the grid in plane, too** (`from_dicom_seg`): each
  frame must start at its slice's first voxel, and the SEG's row and column
  directions and pixel spacing must be the grid's (`E405` otherwise); patient
  positions are converted for an RAS grid or other length units rather than
  read as the grid's own, and a grid in pixels or another convention, where no
  patient position is a place, is `E414`. Every frame's orientation and pixel
  spacing are read where the file keeps them — its own functional groups or
  the shared ones — and every frame is checked: the first orientation found
  stood for all of them, so a frame turned in plane was placed as the first
  one, and spacing kept per frame read as 1 mm. Frames turned or spaced
  against each other are refused, with no code, being about the SEG alone, and
  a frame stating no pixel spacing is refused rather than read as 1 mm.
- **A DICOM SEG is not placed in a frame known to be another**
  (`from_dicom_seg`): a grid whose frame of reference is known and is not the
  SEG's is refused (`E414`), matched or named with `--grid`, before the sample
  is touched — an unrelated frame with identical geometry received every
  voxel. Where either side states no frame, the placement rests on geometry and
  is recorded as a guess. A sample `scrub` pseudonymised is compared through
  its pseudonyms (`--frame-salt`, `frame_salt=`), and `frames_agree` asks the
  question on its own.
- **`read_dicom_seg` puts a sparse SEG's planes where they were drawn**: on the
  regular stack of the stated `SpacingBetweenSlices`, else the smallest gap,
  with the planes the file omits empty. The median gap stood for the step, so
  planes at 10, 12.5 and 17.5 mm read as 3.75 mm apart and the middle one 1.25
  mm from its position. Planes no regular stack holds are refused.
- **`to_dicom_seg` writes an annotation only onto its own grid**: the source
  images must be the annotation grid's slices — in its frame of reference
  where the grid states one (`E414` otherwise, as on import; `frame_salt=`
  compares a frame `medh5 scrub` pseudonymised), rows and columns running the
  same way at the same spacing, each image a slice starting at its first voxel
  (`E405` otherwise) — and a grid in pixels or another convention is `E414`.
  Their geometry was given to every frame unchecked, so an annotation on a
  grid turned in plane, or shifted 10 mm in the same frame, was exported
  centimetres from where it was drawn. Images in either slice order are
  matched to their slices. A SEG holds a segment per class, so any kind but
  `labelmap`, `layers`, `bitmask`, `instances` and `probmap` is refused
  (`E401`) before anything is read, where boxes failed with an
  `AttributeError` traceback and a `mask`, which has no classes, with an
  `IndexError` from highdicom.
- **`from_nnunetv2` reads what the header's scaling means**: an image keeps
  `scl_slope`/`scl_inter` as its rescale (they were dropped, leaving a CT at its
  stored counts), and label ids are matched after the scale, which must leave
  integers. **`to_nnunetv2`** writes labels on their own grid and refuses labels
  that are not on the channels' grid; the reference grid's affine was written
  whatever grid they were on.
- **`to_nnunetv2` settles the dataset before it writes a file.** Every channel
  and the labels are one lattice in one frame, convention and unit: equal
  numbers in LPS and RAS, or in metres and millimetres, were exported as one
  space, 48 mm or a thousandfold apart. The label table is every case's
  classes, one name per id — the first case's stood for all, and a later
  case's lesion became background — and every case must have examined every
  class exported, where a class one case never looked for was written as its
  verified absence; `classes=` (`--class`) exports a subset all of them
  examined. A region-based dataset is painted in its `regions_class_order`,
  where only the classes with a value of their own were written and the
  components named only inside regions became background. A case whose label
  volume does not give back every exported label — two classes on one voxel
  — refuses the export before any case is written, where the cases before it
  were left written with no `dataset.json`. These refusals carry no §15.2
  code: the samples are valid, and what they fail is nnU-Net's layout.
- **A `probmap` is refused where it cannot mean what it was given** (`E411`):
  a probability that is not a number (the range check skipped NaN), and a map
  in which some value is within `float32` rounding of the threshold on the
  other side of it — §7.5 says a writer MUST keep every voxel's side, and the
  map was stored as `float32` anyway, moving that voxel across. A map written
  `normalized=True` whose classes do not sum to 1 is `E404`.
- **Stored probabilities are validated** at the integrity level, on the bytes
  the digest pass reads, in its bounded blocks (about 32 MiB): a value outside
  [0, 1] or not a number is `E411`, and a `normalized` map whose classes do
  not sum to 1 is `E404`. Only `threshold` was checked. "Sum to 1" holds to
  the rounding of the values stored (§7.5, Appendix C.1) — under 2·10⁻³ at
  any class count, where an allowance of the class count times the dtype's
  epsilon let a map of 1,024 `float16` classes whose every value had been
  zeroed validate. A probmap's planes are one per class id (`E405`
  otherwise), and a map is summed only on that shape: a declared shape of
  `(0, 2**62)` allocated a sum per voxel and panicked the validator.
- **A dataset too large to hold is reported, not a crash.** A file may declare
  a dataset of any extent and store none of it; reading one asked the
  allocator for, say, 2⁶³ bytes, which panicked, or 9 TB, which aborted the
  process — a 10 KiB file stopped a validator. Every read sizes its buffer
  with checked arithmetic and reserves a large one fallibly first
  (`OSError` from Python, `E001` in a report). Strings are sized the same way:
  a fixed-width string `/meta` declaring 2⁵⁹ elements panicked on capacity
  overflow, and a scalar one of 2³⁰ bytes aborted under a memory limit. A
  `/meta` that is not one scalar string is `E004` (§2.4); a vector was read
  whole and cut to its first element.
- **A `chunk_hint` is a chunk shape** (§3.2, Appendix C.1): an extent below 1
  is `E109` to the writer, the validator and the reader, where the writer took
  it for 1 and the validator passed it; a file holding one still opens, but
  its grids no longer read. One larger than its grid is clipped to it, where
  the write failed with `OSError`.
- **A clinical numeric column stored big-endian is `E805`** (1.1 §4 stores them
  little-endian; HDF5 converted on read, so it validated clean), as are a UTF-8
  column's big-endian `offsets` (1.1 Appendix A), and a member a
  UTF-8 column does not define is `W913` under a projection, as an unknown
  column is — it was `E804` at any version.
- **`amend`, `recompress` and `pack` refuse a file holding HDF5 references**
  (`MEDH5FileError`), before writing anything: a reference is an address in the
  file that holds it, so a copied one pointed wherever its address landed in the
  new file, and a no-op amend nulled those in an extension group, while every
  digest still verified. 1.0 §16 and Appendix C say so. An attribute of a
  committed (named) datatype counts, and so does one of a committed type no
  link names any more, found through the dataset or attribute whose type it
  is: the walk read groups and datasets only, and then named types by their
  links.
- **NIfTI exports state their units and timing** (`to_nifti`, `to_nnunetv2`):
  lengths in millimetres (a grid in `m` or `um` scaled to them, `px` as
  "unknown"), evenly spaced frames as `pixdim[4]` and `toffset`, and uneven ones
  in a BIDS sidecar's `VolumeTiming`, which `from_nifti` reads back; the header
  said nothing, so a metre grid read back as millimetres and every series as 1 s
  frames from 0. `to_nifti(annotation=…)` exports on the annotation's grid,
  where it took the named image's affine; it takes `report=` for the notes.
- **An export replaces what was said beside its file**: the per-volume fields
  of the sidecar (`VolumeTiming`, and per-volume echo and inversion times and
  flip angles) and a `.bval` are withdrawn once the new image is serialised,
  before it replaces the old one, and the new ones written after it, other
  fields kept. An even series written over an uneven one left its
  `VolumeTiming`, which the reader prefers to the header, so the new image read
  back with the old timeline; an interrupted export now leaves timing
  unmeasured, never another image's, and one that cannot write its image —
  a full disk — changes nothing beside it. A sidecar that is not a JSON
  object is refused before anything is withdrawn: the image's `.bval` was
  deleted first, and the refusal left the image it described without its
  b-values.
- **A NIfTI export is named `.nii` or `.nii.gz`** (`to_nifti`,
  `convert to-nifti`, `to_nnunetv2(file_ending=…)`): a directory or any other
  name, `.NII` and `.nii.bz2` included, is refused before the file is written,
  and by `to_nnunetv2` before it creates a folder. nibabel takes the format,
  and the name, from the suffix, so the temporary beside such a target was
  written under another name and left behind, or `out.img` became an image
  whose `.hdr` was the stray.
- **A channel axis is exported as one** (`to_nifti`): with a b-value per
  channel as four dimensions and a `.bval`, the layout diffusion tools read;
  otherwise as NIfTI's vector dimension (`dim[5]`, vector intent), with
  per-channel echo and inversion times and flip angles in the sidecar. Written
  as a plain fourth axis it read back as time, with invented frame times. A 2-D
  grid with a time or channel axis is refused, where it failed with an
  `IndexError`.
- **nnU-Net's ignore label is an ignore region, both ways**: `to_nnunetv2`
  writes an annotation's §7.7 ignore region as nnU-Net's `ignore` label — one
  above every other value, declared in `dataset.json` — where it wrote
  ignored voxels as background, verified negatives for every class; a region
  over a class is written as ignore, which nnU-Net neither trains nor scores.
  `from_nnunetv2` reads the label back as the region, where it made it a class
  called `ignore`.
- **A one-class NIfTI export and a DICOM SEG export refuse an annotation with
  an ignore region** (`to_nifti(class_key=…)`, `convert to-nifti --class`,
  `to_dicom_seg`, §7.7): a 0/1 mask and a `BINARY` segment have no value for
  "ignored", and their 0 says "examined, absent", so every ignored voxel was
  exported as a negative. Exported whole to NIfTI, the annotation's labelmap
  writes those voxels as its `ignore_id`.
- **A MONAI crop keeps the image's `spatial_shape`** (`to_metatensor(roi=…)`):
  it was the crop's whole shape, channel axis included, and `SaveImage` wrote a
  two-channel crop as one channel. As after a MONAI crop, `spatial_shape` and
  `original_affine` describe the image and `affine` the crop.
- **MONAI receives affines in millimetres** (`medh5.monai`): an affine in the
  grid's own units was read as millimetres, so a metre grid's voxels were a
  thousand times too small; a `px` grid stays in pixels. `from_metatensor`
  refuses a tensor without an affine or a space, where it substituted an
  identity affine in LPS.
- **A `labelmap` item holds the requested classes only, as `int64`**
  (`VolumeDataset`, `PatchDataset`, `GridPatchDataset`): every other class was
  painted beneath them while `meta["annotated"]` named only the requested
  ones, and a `uint16` plane is no `CrossEntropyLoss` target. The ignore value
  (65535) is kept.
- **A split names each member of a collection** (`Split.paths`, the `entries`
  of `medh5 dataset split -o`): `path::key`, the locator task sources use. It
  was the collection's path, so members placed in two partitions put the whole
  collection in both partitions' file lists. `leaks()` also reports an entry
  placed in two partitions, which a split loaded from a file can do.
- **`write_claims` checks everything before it writes**
  (`medh5 dataset split --write-claims`): a fold a k-fold split does not have is
  refused — it matched no assignment, so every file was written as `train` —
  and so are a `fold` given for a ratio split, a split that leaks, a sample
  inside a collection, and a file that amending would refuse (a later minor
  version, an unknown profile, HDF5 references, a digest it fails: `E701`) or
  that fails validation at the level the commit uses, each before the first
  file is amended, where the files before it had been. An I/O failure partway
  can still leave the files before it claimed.
- **`dataset check` compares what a class id means** (C102, C302): ids were
  grouped by `label_set_id@version`, so one id and version carrying two meanings
  passed with a C101 warning, and a version bump that kept every class was a
  C102 error. Manifests record each entry's `class_keys`, and C102 reports an
  id whose key differs between label sets; a manifest indexed before this
  reports every id two label sets share until it is re-indexed. Coverage
  reports `named_in` where it reported `present_in`, which counted the samples
  naming a class as those holding it, so C302 never fired: it reports a class
  named somewhere and examined nowhere.
- **Voxel agreement leaves out what either rater did not examine**
  (`compare_voxel`, `medh5 agree`): a voxel either annotation declares ignored
  (§7.7, in band or as the `mask` its `ignore_mask` names) counts neither for
  nor against agreement. It counted against the rater who declared it.
  Annotations of two samples are compared only on grids that share a declared
  frame and are one lattice (`E101`, §3.3 rule 4): a grid id is its sample's
  own name, and two samples each calling a grid `g` were scored 1.0 though 100
  mm apart, and grids of two shapes with ignore regions panicked the merge.
- **`dice_agreement({})` is refused**: a mean Dice over no class was 0.0,
  which says two annotations disagree everywhere when nobody compared them.
- **`dataset.scan(on_error=…)` takes `"warn"` or `"raise"`**: anything but
  `"raise"` was read as `"warn"`.
- **World coordinates are compared in one unit and one convention**
  (`compare_instances`, `Annotation.to_index` and `to_world`, the paired
  loader): one frame's numbers are in each grid's units (§3.5), so they are
  carried into one — the same boxes in metres and millimetres scored an F1 of
  0, and a point at 1 mm went to index 1000 on a grid of 0.001 m voxels —
  and grids in two `coord_system`s are refused (`E414`, §3.3 rule 4), where
  disjoint boxes in LPS and RAS scored an IoU of 1 and an LPS point was read on
  an RAS grid unflipped. `Grid.world_into(other, points)` is the conversion.
- **Tracking answers one observation per visit**, in every accessor
  (`Track.at`, `volume`, `volumes`, `relative_change`, `medh5 track`): the
  object's mask where an `instances` annotation saw it, else its box —
  axis-aligned or oriented — whose volume is only its bounding box's, else a
  point, which marks the object and measures nothing. Two masks of one visit
  — two raters — are refused; `at` and `volume` answered with the first
  rater and `volumes` with the last. `observations_at()` and
  `measurements_at()` list them, and `Observation.kind` says which annotation
  kind each came from, as `medh5 track --json` does, with `volume_mm`.
- **Tracking joins every kind that identifies objects.** An oriented box
  (`obb`) or a landmark carrying `instance_ids` made the whole join fail, and
  `medh5 track` with it: an oriented box joins on its enclosing box and
  measures its own volume, which its rotation leaves alone, and a point is
  presence. An annotation of another kind that carries `instance_ids` is
  named in `Tracking.skipped`, and the rest still join.
- **Instance ids minted from masks are unique within the sample** (§7.4):
  masks written with `masks=` and stored as `instances` were numbered from 1
  in every annotation, as in 1.x, so two visits' lesions centimetres apart
  were both object 1, and `tracks()` joined them. The writer mints above the
  largest `instance_id` the sample holds (its first such annotation still
  starts at 1) and keeps the ids given with `instances=`. Files already
  written keep theirs: nothing marks an id as minted. An id past the largest
  `uint64` is refused (`E404`), by `instances_from_masks` too, where it
  wrapped to 0.
- **Volume change is measured in millimetres** (`Track.relative_change`,
  `medh5 track`): each observation's volume is in its grid's units cubed, so a
  visit on a grid in metres beside one in millimetres read as a millionfold
  change. Visits with no measure in common — an uncalibrated (`px`) grid
  beside a calibrated one, an area beside a volume — have no change, `None`.
  `medh5 track` prints volumes in mm³.
- **A cubic displacement field takes the outermost sample's value in the
  half-voxel margin**, as a linear one does (§10.4): under `zero` and `error`, a
  point between the outermost sample and the extent's edge was zero — SciPy's
  constant mode, as in 1.x — so under `error` a point admitted was given no
  displacement. Values between the outermost samples are unchanged.
- **Field sampling and TRE check the shape of what they are given**:
  `linear_sample`, `cubic_sample`, `sample_field` and `inside_extent` refuse
  points without one coordinate per spatial axis (`ValueError`), and
  `target_registration_error` refuses weights that are not one per landmark
  (`MEDH5ValidationError`). The engine panicked on too few coordinates and
  ignored extra ones, and paired weights with landmarks by position, dropping
  the excess: two landmarks with one weight reported the first one's error
  alone. 1.x refused the coordinates, and broadcast a single weight. A
  transform's points (`transform_points`, …) carry their coordinates on their
  last axis: a flat array whose length was a multiple of S was regrouped into
  points, and is a `ValueError` unless it is one point (`[x, y, z]`).
  `target_registration_error` also refuses weights that are negative, not
  finite or all zero, and with no landmarks reports NaN for `mean`, `median`
  and `max` (`n = 0`), not a 0.0 that reads as a perfect registration.
- **`fix(rewrite_digests=True, rebuild_index=True)` leaves the index
  current**: the index entries to rebuild are found after the restamp. They were
  the diagnosis from before it, so an index the restamp made stale was left
  stale by the call asked to rebuild it. Restamping alone names the entries it
  leaves stale.
- **An amendment never restamps a digest its source failed** (`E701`, naming
  the paths, before anything is written): a no-op `amend` — or `scrub`,
  `augment`, split claims, `fix` without `--rewrite-digests`, which commit
  through it — made a file edited behind the library's back verify again,
  with no record that its bytes had changed. Restore the bytes, or run
  `medh5 fix --rewrite-digests --reason …`, which records why.
- **Each attested path is bound to its bytes** (§13.2, `E704`): every path
  through `grids/`, `images/`, `annotations/` and `transforms/` (and
  `clinical/` where the profile is declared), soft links followed, must be its
  object's own. Coverage was judged by object, so a file crafted with an alias
  sorting before `images/` — the image's line digested and sealed there —
  passed every check, and `images/CT` relinked after a pin to other covered
  bytes changed no line: `verify`, a task's source pin, a deep preflight and
  the validator all held while a reader read other bytes. `verify` and the pin
  check fail such a file, `SourceRef.pin` refuses to pin it, and the validator
  reports `E704`. The writer makes no links, so no file it writes changes, and
  no `content_id` does.
- **Copies are made of the graph, not its top-level members**: `pack`,
  `unpack`, extract, repack, `amend` and `recompress` copy a sample with one
  object map, so a hard link stays one object, a cycle a cycle and a soft link
  a link (an absolute target is rebased between sample roots, and one that
  leaves its sample root is refused). Each member was copied on its own: an
  alias became a second dataset — a new `content_id` line — so a packed
  aliased sample failed `verify` and a no-op amend changed its `content_id`.
  `pack` and `unpack` refuse a copy that would change a member's
  `content_id`.
- **`recompress` keeps a stored type it cannot rewrite exactly**: an
  enumeration lost its names, and a committed datatype its identity and
  attributes, while the result said ok. Such a dataset is copied as stored and
  listed in `RecompressResult.kept`; datasets sharing one committed type share
  its copy.
- **Removing clinical data removes its bytes** (`medh5 clinical strip`,
  `drop_clinical()`, `remove_annotation()`): the profile was unlinked, and HDF5
  never reclaims an unlinked object, so every record — report text included
  — stayed in the output's bytes. The output is rebuilt from what remains.
- **What an event version attests is fixed with it** (`E809`, 1.1 §7.3,
  Appendix A): an amendment cannot give a version the file already holds a
  payload it did not have — text linked to an inherited, text-less document
  event available at 1 h was owned by that version and admitted at 24 h,
  before it existed — nor add a link the version itself asserts, nor change
  a payload or transform it attests; and a held version dropped with the
  profile and added again is held to what it was. A new version superseding it
  carries the change. Payloads are compared by content, so
  `medh5 fix --rewrite-digests` over unchanged bytes still commits.
- **A cache entry must be owned by the version it is filed under** (`T407`):
  `validate_cache` checks every event-level entry's event and document against
  its source, and the training dataset checks the document before every read,
  also after a reopen or a pickle. A feature filed under a version but
  encoding a later revision's text was served to every row that admitted the
  version. An entry whose source's clinical tables cannot be read is `T407`
  too, where it was accepted unchecked.
- **The context window bounds plans too** (1.1 §9.1): a plan's effective time
  is bounded below by the window like any other event's; only the cutoff is
  not a plan's edge. A stale order from 100 days before a seven-day window was
  read beside the week.
- **A row without documents has the declared feature shape**, and collation
  pads by the populated rows and refuses rows that disagree. Its features were
  `(0, dim[0])`, so a batch collated in one order and raised in the other.
- **Concept tokens escape their separator** (`medh5.task.concept_token`):
  fields were joined with an unescaped `|`, so two different codes could be one
  input. `\` and `|` are escaped; a token without them is unchanged, so fitted
  vocabularies hold.
- **A task's target reads occurrences** (task-cache-1 §5): a `cancelled`
  outcome did not happen, and is neither a positive nor a negative; a
  `planned` one is not known to have happened, so it labels no row positive or
  prevalent, censors a window it may fall in, and as a negative observes
  nothing. Both were read as having occurred: a cancelled operation in the
  window labelled its row positive.
- **A target names its horizon**: `Target(horizon_us=…)` is required, and it
  and the members after it are keyword-only — its default of 0 always failed
  the schema (`T101`). `negative` may be left out (`[]`), and an unknown `roi`
  or `censoring` is `T101` alone, where it was `T102` as well.
- **Integers are integers** in task slots (`patch`, `classes`) and dataset
  manifests: JSON Schema's `integer` admits `3.0`, and a value that did not
  read as an integer was dropped. It is refused (`T101` in a task, a
  `ValueError` in a manifest), as is a non-positive patch.
- **A transform's `units` are its grids'** (§10.1, `E506`). The writer
  defaulted `units` to `mm` whatever the grids were in and wrote any `units`
  it was given, and nothing compared them with the grids: a transform in
  millimetres between two metre grids validated, and the paired loader, which
  applies a transform to its grids' coordinates as they are, moved every point
  a thousand times its displacement. A transform written without `units` takes
  its grids'; one whose `units` are not theirs — or between frames whose
  grids are in two units — is refused by the writer and reported by the
  validator, and the paired loader refuses to apply it to a grid in other
  units. **A file 1.x wrote with grids in metres and a transform left at `mm`
  is now invalid**: it still opens, and an amendment that leaves the transform
  as it is would write a file the validator rejects, so it is refused. One
  that removes the transform (`remove_transform`, above) and adds it again in
  its grids' units commits.
- **`as_world(grid)` answers in that grid's world**: stored world boxes came
  back as stored whatever grid was named, so a box at 3 mm asked for on a grid
  in metres read as 3 m. Another grid's world is computed through it —
  converting units, refusing another frame or convention (`E414`).
- **A foreground draw weighs every candidate class** (`class_weights`), from
  the index or from a scan alike: the scan that stands in for a missing or
  stale index picked classes uniformly, and with an index an empty weighted
  pool fell back to that scan, so a class weighted zero was drawn. An index is
  used only when it covers every candidate class the annotation can express
  — a `foreground_classes` entry it cannot express counts zero, where it sent
  every draw to the scan — and a draw with no candidate that has both
  foreground and weight is a uniform one. The scan counts every candidate in
  one pass and decodes only the class it drew. `PatchSampler` checks
  `class_weights` when it is built: an unknown weighting, or a weight that is
  negative or not finite, raises `MEDH5ValidationError` there, not at the
  first draw. A `mask` annotation yields no foreground draw. **Seeded draws
  without a current index, or for a class a file lacks, differ from earlier
  2.0 builds.**
- **nnU-Net exports are planned before the first file is written**
  (`to_nnunetv2`, `medh5 convert to-nnunet`): a case's `sample_id` and the
  dataset name must be file names — a sample key, not `.` or `..`, no two
  differing only in case — where an id like `../../victim` wrote outside the
  export. A case without the annotation is refused, or with
  `unlabeled="test"` written to `imagesTs` and left out of `numTraining`.
  **Label values are written `0..K`**, as nnU-Net requires, and `dataset.json`
  records the class id of each renumbered value (`medh5_class_ids`), which
  `from_nnunetv2` reads back; a stashed `dataset.json` that does not name a
  class some case carries is extended with it (a region-based one is refused),
  where the class was written as background. Without a stash, a CT image's
  channel is named `CT`, which nnU-Net normalises as CT; named for its image
  (`CT_tp0`), it was z-scored.
- **RTSTRUCT contours are patient coordinates, both ways** (`to_rtstruct`,
  `from_rtstruct`): millimetres, LPS. The export wrote a metre grid's 0.003 for
  3 mm and an RAS grid's x and y unflipped, and the import stored `ContourData`
  as the grid's world coordinates as they were: a square from an RAS grid came
  back mirrored off the grid, `--rasterize` silently skipping what fell
  outside, and one from a metre grid read its millimetres as metres. A grid in
  pixels or another convention is refused, and so are contours with no grid to
  give their units (`E414`). On export the source images must share one frame
  of reference, and the contours' must be it; they were filed under the first
  image's frame whatever the grid's. On import a grid in another known frame
  — matched, the sample's only one, or named with `grid=` — is refused
  (`E414`) before the sample is touched, and where either side states no frame
  the placement is a recorded guess. `frame_salt=` (`--frame-salt`) compares a
  frame `medh5 scrub` pseudonymised.
- **A migrated `extra.nnunetv2` is where the exporter reads it** (Appendix B):
  `/meta → extra.nnunetv2`, and every other 0.x `extra` key under
  `extra.legacy.<timepoint>`, one entry per file. The whole `extra` went to
  `extra.legacy`, so `to_nnunetv2` never found the stash, and a grouped
  migration kept only the last file's. Grouped files whose stashes differ keep
  the first, and the report warns.
- **A 0.x file without a spacing migrates to an uncalibrated grid**
  (`units="px"`, §3.5, a `geometry` decision in the report), not to
  millimetre voxels nobody measured, and a `coord_system` it does not state is
  LPS on a recorded guess.

### Removed

- The private `medh5._hdf5` and `medh5.document_fields` modules.
- `medh5::h5::ops::copy_object` and `copy_unknown` (Rust): they copied one
  member at a time, which is how an alias became a second object;
  `medh5::h5::graph` copies a sample as a graph.
- More of the Rust API: `medh5::validate::clinical_errors`
  (`clinical_checked` returns the same diagnostics),
  `integrity::digest::walk`, `integrity::repair::shown` and
  `h5::data::is_enum`. `medh5_cli::app` and `medh5_cli::common` are private:
  the crate's API is `run`, `run_os`, `run_with`, `utf8_args`,
  `command_tree`, `Host`, `top_level_message` and the exit codes.
- The argparse internals of `medh5.cli` (`build_parser`, `MODULES` and the
  per-command modules); `command_tree()` describes the grammar and `main()` runs
  it.
- `h5py`-level writers and internals with no meaning over the engine:
  `write_document`, `write_grid`, `stamp_digests`, `write_index`,
  `read_transforms`, `open_transform`, `component_ids`, `composite_cycle`,
  `require_major`, `validate.Context`, `validate_collection` (`validate_file`
  validates collections), `validate.rules`, `labels.registry.VOCAB_DIR` (the
  vocabularies are compiled in) and `voxel.payload`'s scan helpers. Writing goes
  through `SampleWriter`, which stamps digests and indices at commit; reading
  through `medh5.open`.
- The `schema` and `interp` extras, which installed `jsonschema` and SciPy for
  work the engine now does, the `itk` extra, whose SimpleITK nothing used, and
  `medh5.document.schema_available()`, which could only answer `True`.
  `pip install "medh5[schema]"` or `"medh5[itk]"` warns that the extra is
  unknown and installs the package.
- **The second and third copies of the JSON Schema.** `schemas/` and the
  package's `medh5/schemas/` held the bytes the engine embeds, kept equal by a
  test. The one file is `crates/medh5/data/medh5-sample-1.0.schema.json`: §2.4
  names it, the documentation site publishes it at
  `reference/medh5-sample-1.0.schema.json`, and `medh5 conformance publish`
  writes it into every suite. `medh5.document.SCHEMA_PATH` and `SCHEMA_NAME`
  went with the package's copy; `medh5.document.schema_text()` returns the
  schema.
- What the module consolidation left with nothing to do: `medh5.__about__`
  (`medh5.__version__` is the version), `medh5.annotations.base.readers()` and
  `medh5.transforms.transform_readers()` (`medh5.annotations.READERS`, with
  `open_annotation` beside it), `stored_inverse()` (`Transform.inverse()`),
  the 0.x reader's second names `read_sample` and `read_meta`
  (`medh5.io.legacy.read_legacy` and `legacy_meta`), and the private
  `medh5.cli._common` helpers `human_bytes`, `indent` and `table`, which only
  the test suite called, with the two engine bindings behind them.
- `VerifyResult.details`, which nothing filled.

### Fixed

- **A damaged file is an error, not a crash.** The vendored HDF5-Blosc2 filter
  asserted that its chunk-size value covered the chunk, so a file whose filter
  values were too small aborted the process; it checks them with
  overflow-checked arithmetic and returns HDF5's ordinary error
  (`vendor/hdf5-blosc2/PATCHES.md`). Comparing two trees with a link back to an
  ancestor recursed until the stack overflowed; it compares graphs. And the
  geometry functions (`decompose_affine`, `index_to_world`, `world_to_index`,
  `box_corners`, `box_to_slices`, …) raise `ValueError` on a malformed affine,
  points or box, where they panicked with a `PanicException` — a
  `BaseException` that `except Exception` does not catch.
- **Strings are read within a budget**: a few kilobytes of compressed one-byte
  strings decode to hundreds of megabytes, and digesting held every string at
  once. Digests stream strings a bounded slab at a time, whole reads and
  attributes are budgeted by what they decode to, and a declared extent no
  machine holds is refused before anything is read.
- **C-Blosc2 3.3.5**, vendored byte for byte from upstream (it was 3.3.2):
  BloscLZ added a match length's continuation bytes to a signed counter before
  checking it, so a run of eight million of them in a hostile chunk overflowed
  the counter — undefined behaviour in the decoder every Blosc2 read goes
  through. 3.3.5 checks first, with upstream's other bounds fixes since 3.3.2;
  tests feed the overflowing streams to the decoder.
- **A stored chunk is read within its file**: a crafted file declaring an
  oversized chunk made a windowed read or `integrity.raw_chunks` reserve
  gigabytes, or abort; such a chunk is read through HDF5, or refused
  (`OSError`). `raw_chunks` and `subtrees_identical` list a dataset's chunks in
  one pass, where 10 000 chunks took 13 s.
- **Strings and compound types digest as 1.x digested them** (§13.1,
  Appendix C.1): a string dataset over its stored bytes, a fixed-length one's
  without its NUL padding — the engine decoded strings as text, so two values
  differing only in invalid UTF-8 shared a digest — and a compound, array or
  opaque dataset as `|V<n>`, on which `amend`, `verify`, validation and
  `recompress` failed. No digest of a valid UTF-8 string changes.
- **The specification names bytes, not Python functions.** A second
  implementation found three clauses it could only satisfy by reading 1.x's
  code: booleans and strings (§2.5) were `np.bool_` and `h5py.string_dtype()`,
  canonical JSON (§5.1, §13.2) had floats "in `repr` form" and no separators,
  and a digest's `dtype_str` (§13.1) was "the NumPy dtype string", which says
  nothing for a string dataset. Each is now defined in HDF5 and JSON terms and
  recorded in [Appendix C.1](docs/spec/medh5-1.0.md#c1-reference-implementation).
  The definitions are what 1.x always wrote: no file and no digest changes.
- **An index's `seed` promises what it can.** §14.3 said `max_coords` and
  `seed` gave "reproducibility of the subsample", which across implementations
  needs a generator the specification never named. The implementation that
  drew a subsample reproduces it from them; another draws a different one,
  equally conforming (Appendix C.1).
- **NaN in `/meta` is reported.** 1.x's validator parsed `/meta` with Python's
  `json`, which accepts `NaN`, so a document that is not JSON passed.
- **The executable prototype agrees with the specification**
  (`docs/examples/reference_writer.py`, Appendix C.2). Its `content_id` hashed
  `index/` and every attribute, sorted every line as one group and spelled the
  `meta` line `sha256:`; its `source_digest` was the `data` dataset's digest, not
  §13.3's group digest; and its checks printed `[ok]` whatever they found — the
  engine reported E702 and W905 for the file it passed, as 1.x did. It now
  computes both as §13.2 and §13.3 define them, stamps no digest under
  `index/`, raises on a failed check, and checks the round trips the appendix
  listed and it never ran. CI validates what it writes with the wheel and the
  native binary.
- **§14.1 lets a small dataset stay contiguous** (Appendix C.1): below 64 KiB,
  or empty, a dataset **MAY** be stored contiguous and unfiltered, as every
  implementation since 1.0 has stored it; the clause said image and
  voxel-annotation datasets **MUST** be chunked, without exception. **§10.4**
  says where a displacement field ends — its voxel extent — and what its
  half-voxel margin holds, which linear and cubic interpolation answered
  differently.
- **§13.2 spells each line** (Appendix C.1): a dataset line carries the stored
  `<algo>:<hex>`, an attribute line `<algo>:` and the hex digest, the `meta`
  line the bare hex digest, and every member of `grids/`, `images/`,
  `annotations/` and `transforms/` has an attribute line. The pseudo-code wrote
  `H(...)` for both kinds and named no objects; the text now says what every
  implementation has written since 1.0, and no digest changes.
- **§13.2 lists the attributes `content_id` covers**, object by object
  (Appendix C.1). It hashed "spec-defined attributes", which include §8.5's
  `correspondence`, while every implementation since 1.0 has covered one fixed
  list per object that leaves it out — and names `scope_ids`, a §9 dataset,
  among the annotation attributes — so an implementation following the text
  computed another `content_id` for every file with landmark pairs. No
  `content_id` changes; `correspondence` is named as not covered.
- **Six more 1.0 clauses say what the implementations do** (Appendix C.1):
  validating a sample compares its own split claims for `W906`, and a split
  audit (`medh5 splits`) a cohort's (§12.3); a validator checks an extension
  dataset's digest and storage as it does every dataset's (§16); the root
  attribute table is a sample root's, a collection's `/` needing only
  `medh5_kind` and `medh5_version` (§2.1); `source_digest` is on each index
  entry, not on every object under `index/` (§13.3); a timestamp is RFC 3339
  and **SHOULD** be UTC, an offset having always passed `E604` (§2.1, §11.1);
  and amend copies through what it does not interpret only in a version the
  writer implements (§14.4).
- **The specification says what the corpus scores** (Appendix C.1, 1.1
  Appendix A), where a validator written from §15.2 failed cases the reference
  passes: `contour_offsets` and `mesh_offsets` end at the number of vertices
  or faces they index, and an `instances` object's span in `mask_data` is its
  packed crop (`E408`, which said only "not monotonic"); an `ignore_mask` or
  `valid_mask` names a `mask` on its object's grid (`E413`); a collection
  holds a `samples` group with a sample root (`E008`); and `integrity`, not
  `semantic`, makes the `probmap` value checks, in its full read (§15.1). 1.1
  §11.1 puts `E802`'s rule for a `shifted_utc` clock at `semantic` and lists
  `E704` among the integrity codes. The summaries of `E001`, `E003`, `E008`,
  `E104`, `E105`, `E108`, `E109`, `E306`, `E405`, `E408`, `E411`, `E412`,
  `E413`, `E502`, `E503` and `E504` — `CODES[…].summary`, the published
  `codes.json`, the reference page — add what the validator reports under
  them that they left out. The text changed, not the verdicts.
- **The corpus case `W907-float-storage` verifies**: it rewrote its image as
  `float32` and kept the `int16` digest, so a case meant to be valid with
  `W907` failed its digest (`E701`) and was marked `mutated`. It is stamped
  over the bytes it stores; its expected outcome is unchanged.
- **1.1 defines what it relied on**
  ([Appendix A](docs/spec/medh5-1.1.md#appendix-a--changes-from-the-reviewed-draft-and-why)):
  a *recognised* `clinical` group, and when an undeclared one is `E803` (§3);
  the identifier columns `E809` holds non-empty, `record_id` taking the 1.0 id
  syntax (§4, §5.1); an `instance` endpoint as one in some annotation's
  `instance_ids`, `E813` when none holds it (§7.1); an amendment that drops the
  clinical profile as the §10 projection, written at 1.0 (§2.1); no conversion
  between clocks — fragments join only on one clock (`T304`), and versions
  that contradict each other are refused (`T305`), not resolved (§3, §7.2);
  and an event of `unknown` time as a plan only when `planned`, as the engine
  always read it (§9.1).
- **task-cache-1's digests can be reproduced from its text**: §3.1–§3.3 and
  §7.1 define, to the byte, a manifest's normal form, what each fingerprint
  hashes, the reconciliation digest (an event version as the clinical schema
  spells it, all nineteen members) and a cache's `manifest_digest` (over the
  stored manifest's bytes), and §3.5 the order a slot's images are tried in and
  where an `eligible_instances` crop is centred. The engine computed each this
  way already. `T404` says what it is found for — a cache or preflight of
  another task, manifest, row or row version, or a cache changed since it was
  validated — where it said "another task, cutoff or selection", though a
  cache read under another selection is `T406`.
- **The §14.2 codec table timed a cache.** Its 64³ patch reads — 0.08 ms for
  every codec, gzip included — re-read one window through one open dataset,
  which HDF5's chunk cache answers after the first read, and the text concluded
  that patch reads are codec-insensitive. Re-measured with a new window per
  read, the codec shows: 1.0 ms for `training`'s lz4, 8.6 ms for `portable`'s
  gzip, and the table gains `balanced`, the default.
- **Every stated count is the real one**: the code table has 95 codes (pages
  said 93 and 71), the clinical profile 36 corpus cases, 8 valid
  and 28 invalid (they said 35 and 27), and the corpus *at least* one case per
  code, not one each, nor one per clause; tests now read every stated count. The
  published suite's README names a case it holds (`E102-non-orthonormal`),
  counts its codes against both tables, and says the scorer takes
  `medh5 validate --json`'s `path` and `diagnostics` as they are, where it
  said the command "emits a superset" of a submission; its manifest cites
  §5.4 for the hierarchy cycle. The level pages describe the payload scans as
  they run: the ignore-region and layer scans have had no 64 M cap since 1.4.1
  — they read a bounded slab at a time, so memory is bounded and the work is
  not — only the `int16` probe is capped, at 4 M values, and a `structural`
  pass scans for an ignore region only where the answer decides a finding.
- **The documentation's examples run as written, and say what the API
  does.** The docs test passed a block whose only error was an unbound name
  — thirty did, and ten ran nothing — and now supplies what the pages
  leave to their prose and fails a name nobody binds; five examples the writer
  refused are fixed. The first tutorial validates at the default level, where
  its file passes with two warnings, not at `strict`, where it fails, and the
  CLI check reads every block, catching a misspelled subcommand and a
  synopsis's bracketed flags. The annotation pages say that `class_ids` is
  what an annotation can express, that a box's class is in `object_class_ids`
  and that the ignore region is a region, not "class 65535"; the API
  reference and the longitudinal guide give `physical_size`,
  `transform_points` and track volumes in their grid's or transform's units,
  not millimetres; and §14.2's lz4hc L8 row is the 0.x default codec, not
  `balanced`.
- **The READMEs link the released documentation** (`/en/stable/`), where they
  linked `main`'s (`/en/latest/`) — on the PyPI and crates.io pages too.
- **A grid is finite and countable, and the validator checks it as the reader
  does** (§3.2, Appendix C.1): an infinite spacing is `E104`, a NaN or
  infinite origin `E109`, and a shape whose voxel count overflows 64 bits
  `E109` — `n_voxels` of a 2⁶² × 2⁶² × 2 grid was 0. The validator passed
  grids `open` refuses — `spacing`, `origin` or hints of the wrong length, a
  `direction` that is not S×S, spacing that is not a finite positive number, a
  non-positive extent — and gave a NaN `direction` a residual of 0; each is
  reported under the reader's code (`E102`, `E103`, `E104`, `E109`, `E110`),
  and `units` or `time_units` outside the §3.2 vocabularies is `E109`.
- **A pyramid is checked whole**: an empty `grid_levels` panicked the
  validator, and `add_pyramid` with no levels the writer, and nothing compared
  `levels` with `grid_levels`, required a dataset for every level, compared a
  level's shape with its grid or checked the shape of `downsample_factors`.
  Each is `E105` (a level's shape `E202`), and a refused pyramid leaves
  nothing behind, so a corrected retry commits.
- **A refused write leaves nothing behind** (`add_*`, `transcode_annotation`):
  an annotation's group was made before its codec was resolved or its
  datasets and attributes written, so an unknown codec, or a string HDF5
  cannot store, left half an object — the corrected call failed with
  "already exists", and the commit refused the file (`E412`) or kept the
  object without what had failed — as did `add_transform` and `add_grid`.
  A write that fails partway is undone, its bytes too; an inline quality
  record is filed once its object is written, where a refused duplicate
  replaced the record of the one already there; and a refused transcode keeps
  the annotation, which it had removed.
- **A rule that stops reports what stopped it**: any error a rule raised
  became one `E001` at `/`, so six instance ids for three class ids read as an
  unreadable file. The error's own code is reported (`E405` there), and
  `E001` only for what could not be read, at the object where the rule knows
  it.
- **One unknown value is one finding**: the annotations and transforms on a
  grid naming an unknown axis kind were measured against the axes the known
  kinds counted, adding `E405`, `E503` and `E504` to the grid's own `E110`,
  and an `inverse_id` naming a transform of an unknown kind, or without
  frames, added `E505` ("does not exist") to that transform's own `E502`. In
  a later minor's file, where the unknown value is only `W913`, the added
  errors failed the file.
- **An unreadable `/meta` no longer floods a clinical sample with `E813`**:
  every timepoint, activity and de-identification reference was checked
  against an empty document, and a task's preflight refused the subject for
  it. The record rules that need no document still run.
- **`integrity` reports `W908`**, as `semantic` and `strict` do, so each level
  includes the ones before it.
- **An empty `medh5_version` is `E001`**, not `E002`.
- **A malformed transform is an error, not a crash** (§10.3, §10.5, Appendix
  C.1): a `matrix`, `field` or `control_points` of the wrong shape panicked
  whatever read it — `is_invertible`, `summary()`, resolution,
  `medh5 info`. A matrix is `(S+1, S+1)` for two or three spatial axes
  (`E504`), a field `(S, *spatial)` on its field grid, and control points
  `(S, *cp_shape)` on the axes of `cp_grid` with at least `order + 1` on each
  (`E503`), which the validator checks as the writer always has.
  `linear_sample`, `cubic_sample` and `sample_field` refuse a field without a
  sample on every axis, and `basis` an order it cannot evaluate before sizing
  its output by it: `basis(2**31, 0.5)` aborted the process.
- **An affine moves points of its own dimension** (§10.3): a 3×3 matrix
  between two 3-D frames was written, validated, and applied to 3-D points by
  regrouping their coordinates in twos; the writer and the validator refuse it
  (`E504`).
- **Windowed displacement sampling is the whole field's at any coordinate**
  (`Transform.sample_indices`): the window is taken over the points clamped
  onto the lattice, where an infinite coordinate overflowed it or moved it off
  the samples a finite point needed, and points of the wrong width are
  refused.
- **A label hierarchy as deep as the id space loads**: the cycle check
  recursed once per parent link, and a chain through every id crashed
  whatever opened the file (SIGSEGV).
- **`frame_graph` draws a transform and its stored inverse once each way**
  (§10.1), as resolution walks them; it drew each edge twice.
- **Composite cycles are found in one walk**: every composite rebuilt its
  siblings and walked from itself, so a thousand chained composites took
  thirty seconds to validate.
- **An annotation whose datasets disagree is refused, not read past.**
  Readers indexed per-object columns by row and trusted the mask shapes: a
  column one row short panicked, a planted mask shape of 2⁵⁰ voxels aborted
  the process allocating it, falling `mask_offsets` panicked, and a span
  shorter than its crop read back with the missing voxels `false`. Every
  per-object column is held to one row per object, index coordinates to the
  grid's rank, a voxel volume to its grid's shape and a `probmap` to a plane
  per class (`E405`); an `instances` object to a mask of its box's extent, no
  larger than its grid (`E405`), and to offsets that delimit exactly its
  packed crop inside `mask_data` (`E408`). `payload_to_masks` and
  `transcode_payload` hold a payload to the same rules, and decode a box
  hanging off the grid as a reader does, where they refused it. The validator
  reports the same codes, where it passed most of these files, panicked on
  boxes of the wrong shape beside `slice_index`, and answered the rest with
  one `E001`. `encode_instances` refuses a crop that is not its box's extent,
  and `encode_classification` takes one value per class id (`E405`).
- **`class_ids` holds every class an annotation can express** (§6.2): an
  `instances` annotation written with `annotated_classes=`, or built with
  `encode_instances(class_ids=…)`, declared the classes examined in place of
  its objects', so an object's class was missing from the header while
  `contains()` still answered for it. The two are united, and the validator
  reports an object of a class the header does not declare (`E404`).
- **`labelmap()` keeps the ignore region under every encoding** (§7.7): only
  the `labelmap` kind read `ignore_id` at an ignored voxel; `layers`,
  `bitmask`, `instances` and `probmap`, and any region kept in a sibling mask,
  flattened it to 0, a verified absence — and so did
  `to_nifti(annotation=…)`, which exports `labelmap()`.
- **World coordinates are read in the frame they name** (§3.4): an annotation
  in `space="world"` whose `frame_uid` is not its grid's frame was written,
  validated and read through the grid's affine as the grid's own. The writer,
  the validator and a read on that grid refuse it (`E414`); a grid that states
  no frame still takes it.
- **Values are checked before they are narrowed**: keypoint visibility 256
  was stored as 0 and 258 as 2, and a face index past 2³¹ or a contour plane
  or `slice_index` past `int32` wrapped into range. Each is refused (`E411`,
  `E405`), and a NaN classification value is `E404` to the validator, as it
  was to the writer.
- **The exported encoders validate as the writer does**: `encode_labelmap`,
  `encode_layers` and `encode_voxels` panicked on an `ignore` region of
  another shape (`E405` now), and stored any `ignore_id` cast to `uint16` —
  70000 became 4464, −1 65535, and 1 class 1; it must be a `uint16` that is
  neither background nor a class (`E404`). An `ignore=` region no voxel
  carries no longer makes a `uint16` labelmap, which the writer refused as its
  own output (`E411`), and one over a class is refused (`E404`): §7.7 keeps
  such a region in a sibling `mask`, as the writer does, where
  `encode_labelmap` lost those voxels and `encode_layers` painted the ignore
  id into their other layers. A `layers` annotation holding one is refused a
  transcode to `labelmap`, which lost the region.
- **A polygon with no stored plane is out of plane**, `(-1, 0)`, `Polygon`'s
  own default, rather than `(-1, -1)`.
- **An array in any NumPy memory layout is written**: every writer entry
  point refused a Fortran-ordered, transposed, flipped or strided array —
  nibabel's `get_fdata()`, `x.T`, `np.flip(x, 0)` — with "input array is
  not in standard layout"; it is copied into C order, as 1.x did.
- **Edits to the document reach the file again**:
  `w.document.extra["tool"] = {…}`, the same for `acquisition` and `quality`,
  and an edit to a value nested in one, changed a fresh copy and were dropped.
  Each is a dict whose edits write through; an edit to a member changed since
  it was read raises `RuntimeError` rather than undo that change, and a copy
  (`copy.copy`, `dict(…)`, `pickle`) is a plain, detached dict.
- **A record's dicts are its own, and a document's records write through**:
  `Identity.extra`, `Cohort.extra`, `Deidentification.extra`,
  `Activity.params`, `Timepoint.series_uids`, `Agreement.per_class` and
  `LabelClass.properties` were rebuilt on every read, so an edit to one was
  lost. Each is kept from its first read, and `to_json`, `==`, a copy or a
  writer reads the record as edited; its hash leaves out what an edit can
  change. A record read from a writer's document — `w.document.identity`, a
  timepoint, a quality record, a class of `w.document.label_set`, or one
  `w.identity(…)` and its siblings return — writes each edit through to the
  document, and an edit through one whose entry has changed since it was read
  raises `RuntimeError`. `w.document.provenance.add_agent` and `add_activity`
  reach the document, where they added to a copy, and so does an edit to the
  dict `w.acquisition()` returns. `Provenance`, `Timeline` and `LabelSet`
  keep the records and classes they are given, as 1.x did, and
  `bool(Provenance([agent]))` is true again.
- **`AnnotationPayload` is a mutable record again**: `datasets` and `attrs`
  were rebuilt on every access, so `payload.attrs["k"] = v`, an array edited
  in place or a dataset added was lost. They are the payload's own dicts, its
  fields can be reassigned, and the writer, the transcoder and
  `payload_to_masks` read it as it stands.
- **Every class 1.x shipped as a dataclass pickles and copies**:
  `AnnotationHeader`, `TransformHeader`, `Pyramid`, `AnnotationPayload`,
  `OverlapStats`, `CostModel`, `IndexPayload`, `Observation`, `Track` and
  `Tracking` raised on `pickle.dumps` and `copy.copy`. `Attrs`, `Dataset`,
  `Group` and `SamplingIndex` name their public modules (`medh5.nodes`,
  `medh5.storage`) in `__module__`.
- **Signatures say what a call takes**: `help()` and `inspect.signature` show
  the curation records' fields (`Identity`, `Activity`, `QualityRecord`, …)
  where they showed `(*args, **kwargs)`, and too many arguments is a
  dataclass's `TypeError` ("takes from 3 to 7 positional arguments"); `Grid`,
  `LabelSet`, `Skeleton` and `AnnotationHeader` show their real defaults, not
  `...`.
- **Patch arithmetic refuses impossible shapes**: `coerce_patch_size`
  refuses more than 32 axes before allocating —
  `coerce_patch_size(4, 2**31)` aborted the process — and `window_around`
  and `grid_patches` a negative shape or patch, whose windows ran backwards
  and read nothing.
- **A grid with `-0.0` in it hashes like its `0.0` twin**, which it already
  compared equal to, so a set or dict of grids holds it once.
- **Reads that decompress release the GIL**: annotation, transform and region
  reads, `PatchSampler.draw`, the digests and `verify_root`.
- **`storage.occupancy(mask, 0)` is a `ValueError`**, where it panicked.
- **Errors name what failed**: a failed flush reports HDF5's own message, not
  "HDF5 error", and a missing grid's `KeyError` names the grids there are.
- **Detection boxes in a padded patch are where their voxels are.** A patch
  reaching past the volume is padded before as well as after, and
  `label_format="instances"` subtracted the patch start alone, leaving every box
  `pad_before` voxels from the anatomy; it now adds the padding, and tests
  overlap at voxel edges (a box ending at `start - 0.5` covers none of the
  patch). It also returns only the requested classes, as every other label
  format does — it returned every class.
- **A MONAI ROI's affine follows the read.** `to_metatensor(roi=…)` shifted the
  origin by `slice.start or 0`: a negative start counted from the first voxel
  rather than the last, and a step kept the full spacing. The ROI is resolved
  against the level's shape, its step composed into the affine, the resolved
  `[start, stop, step]` recorded in `meta["medh5"]["roi"]`, and an integer
  index — which drops an axis the affine keeps — refused.
- **Torch windows are drawn on the grid their images share** (`PatchDataset`,
  `GridPatchDataset`): they were placed on the reference grid, so a dataset
  scoped to a later visit (`timepoint="tp1"`) refused every item of a
  multi-visit sample. `PairedPatchDataset` refuses the `annotations=`,
  `label_format=` and `timepoint=` it ignored, and a `samples_per_pair` below
  1; `collate` takes the keys of every item, not the first's, and names a key
  some items lack.
- **nnU-Net imports keep every channel and every ignored voxel**
  (`from_nnunetv2`): channels were keyed by name, and nnU-Net names a channel
  for its normalisation, so a second `zscore` channel replaced the first; each
  is an image of its own (`zscore_0000`). The ignore label, one above the
  written values, was restored along with them and could become a class id. A
  label value `dataset.json` does not declare is refused, as nnU-Net refuses
  it, where it was background for every class.
- **`medh5 convert from-dicom` exits 0 on success**: the note that no
  de-identification record was written was a warning, on every conversion; it
  is information, and the validator reports the file (`W903`). An output path
  that is an existing directory, or ends in a separator, holds the samples
  even when there is only one, where it became a file of that name or an
  `OSError`; slices of two matrix sizes are refused by name.
- **A DICOM SEG is read only as what its frames hold**
  (`read_dicom_seg_frames`, `from_dicom_seg`): a `LABELMAP` SEG, whose frames
  hold every segment by pixel value, was read as `BINARY` with every frame
  segment 1, as was a frame naming no segment; both are refused. A segment
  named in the shared functional groups, where highdicom names a
  single-segment SEG's, is read from there.
- **`read_nifti(transpose=False)` counts a 4-D file's frames on its fourth
  axis**, not its first: a four-frame file had as many frame times as its x
  extent.
- **A converter's input errors are `medh5: <why>`, not tracebacks**
  (`medh5 convert`): a file that is not DICOM or NIfTI is refused as such,
  naming it, where pydicom's and nibabel's own errors escaped; any `OSError`
  is reported, exiting 1, and a missing NIfTI file is named as it was given,
  where nibabel spelled a Windows path `C:/…`; and a `--report` naming a
  directory is refused before the conversion, not after its output is written.
- **The DICOM extras need pydicom 3.0, and `dicomseg` highdicom 0.23**:
  writing an RTSTRUCT passes `enforce_file_format` to `save_as`, which pydicom
  3.0 added, so under the old floors (2.4 and 0.22) it raised `TypeError`.
- **0.x migration keeps a cohort coherent** (Appendix B): names that become
  one class key (`liver`, `Liver`) are one class, listed in the report —
  `--write-labels` refused the sidecar (`E302`) on the guide's own example —
  and their masks are stored as their union, where the second replaced the
  first. `extra.nnunetv2.labels` that disagree keep the first file's ids, with
  a warning, where the last silently won. A file that cannot be written is
  reported (`failed`) and the rest migrated, where the first ended the run.
  `extra.review` is the quality record of every annotation migrated from its
  file, and the `review` activity names them as outputs: the record was keyed
  `seg`, which no annotation was, even in a file of boxes. An mtime order is
  reported once per subject, not twice.
- **`scrub` finds and shifts every date and date-time, in ASCII digits.** The
  date rules matched any script's decimal digits and the shift sliced the
  match as ASCII, so a `StudyDate` in Devanagari digits made
  `scrub --apply --date-shift-days` panic. An ISO date needed a word boundary
  after its day, which a date-time never has: a date-time in free text went
  unreported, one under a date attribute was dropped, a timepoint's was
  deleted rather than shifted, and only a value's first date moved — while
  the attestation recorded the shift and the re-scan found nothing left. Every
  date in a value moves, a date-time keeps its time, DICOM `DT` values count,
  and an eight-digit code naming no day (SNOMED CT `52988006`) is no longer a
  date finding no `--apply` could clear. A timepoint `date` or a timestamp in
  another script's digits is `E604`.
- **Prevalence is at most 1** (`compute_stats`, `medh5 dataset stats`):
  presence was counted wherever a class's voxels were but examination only
  where the class was looked for, giving a prevalence of 3.0; it is the share
  of the samples that looked which found it.
- **The cohort tools take collection members**: `dataset stats` and
  `compute_stats` measure a member by its locator (`path::key`), findings
  name members, and C401, C402 and C501 count a shard once.
- **The documented cohort workflow runs**: `dataset stats --partition` after
  `--write-claims` told the caller to run the `--write-claims` they had just
  run; it says to re-index, as the guides now do, and the README's
  `migrate --group-by subject` has the `--subject-key` it needs.
- **`medh5 bench` reports what the sampler did**: `used_index` meant that an
  index entry existed, so a stale one was reported as used.
- **Output that cannot be written fails the command**: `clinical export`,
  `info --json` or `tree` onto a full disk exited 0, and now report
  `cannot write output` and exit 1. A reader that leaves early
  (`medh5 validate … | head`) no longer turns a failed check into success, and
  a character the stdout encoding cannot spell — Windows writes a redirected
  stdout in its ANSI code page — is escaped, where `tree` dropped its
  box-drawing lines.
- **Every `--json` output is one JSON document**: `labels show --json` on a
  sample without a label set prints `null`, and
  `migrate --write-labels --json` puts its review note on stderr.
- **A batch reports on every file**: `verify`, `fix`, `scrub`, `recompress`,
  `index build` and `labels check` stopped at the first file they could not
  process, printing nothing with `--json`, so `scrub --apply` lost the
  `uid_map` of the files it had pseudonymised. Each file gets an entry
  (`{"path", "ok": false, "error"}` for a failure), the batch goes on, and the
  command exits 1 at the end.
- **`verify` and `fix` say why they fail**: `verify` names a digest it cannot
  check on its `MALFORMED` line, `fix` the datasets no line of the
  `content_id` covers, and `fix --rebuild-index` the annotations that have no
  index, pointing to `medh5 index build`.
- **`seg stats` recommends only an encoding that keeps the annotation**: it
  recommended a dense encoding for instances, which drops every
  `instance_id`, and called that lossless.
- **`seg convert` says that it thresholds a `probmap`**: converting one keeps
  the voxels at or above its `threshold` and drops the probabilities (§7.5,
  §7.6), where the help, the command-line reference and the segmentation
  guide called every conversion lossless.
- **`clinical select --cutoff-hours` refuses NaN, infinity and a time off the
  subject clock**, which selected at 0 µs or at the clock's ends, and
  certified the result.
- **`conformance build|run|publish --case` refuses a name the corpus lacks**,
  and says how many cases it wrote: `--case nosuch` exited 0, and `run`
  reported "0/0 cases pass".
- **Errors name their file, in the command line's own words**: an I/O or
  parse error says `[Errno 2] No such file or directory: 'm.json'` or
  `'m.json' is not JSON: …`, where it said "No such file or directory (os
  error 2)" or "EOF while parsing a value". A collection handed to a
  converter or to a command without `--key`, or a sample to `ls` or `unpack`,
  is told what the file is and pointed to `medh5 unpack` or `medh5 info`, not
  to `open_collection()` or `medh5.open`, and `seg stats`, `agree` and the
  clinical commands no longer quote the engine's lookup sentence. A SEG or
  RTSTRUCT conversion refused over a frame `medh5 scrub` pseudonymised names
  `--frame-salt` beside `frame_salt=`, where it named the keyword alone, or
  nothing when exporting an RTSTRUCT.
- **`pack` never writes over one of its own sources**, which replaced the
  sample with the collection, and `unpack` refuses members whose names differ
  only in case, or a member named twice. `dataset index` refuses a root it
  cannot list, where it wrote an empty manifest, and it and `scan` report a
  directory they cannot list with the files that would not open — an
  `OSError` when strict — where they passed it over in silence; `tree` names
  the clinical group and a shard's members, which it called extension
  objects; and recompressing a shard keeps the image codec for the images of
  a member keyed `annotations`.
- **Ctrl-C ends the console script at once**: a command running in native
  code ran to completion and then printed a `KeyboardInterrupt` traceback.
- **The standalone binary hands a command to Python without the current
  directory on `sys.path`**: a `json.py` beside the files being converted was
  imported, and run, in place of the standard library's.
- **A world-space box centres a crop where it is** (task-cache-1 §3.5): its
  millimetres were read as voxel indices, so the crop landed elsewhere with
  every voxel marked valid. Boxes are carried into the grid's index space
  first, and an annotation whose boxes cannot be placed is passed over.
- **A source without `local_subject_id` is held to its subject's id**
  (`T303`): another patient's file listed under `P-01` passed preflight.
- **A row feature's versions are compared as a set**: listing them in another
  order was `T406`; only a missing or extra version is.
- **`ClinicalTaskDataset` checks `row_features` against the sources it
  reads**: for a task held without a base, a valid cache was refused (`T404`,
  `T406`).
- **`imaging_events_from_timepoints` keeps a visit's fractional day and its
  range**: 1.75 days became [1, 2) days, and is [1.75, 2.75); an interval the
  clock cannot hold wrapped, and makes no event, with a note; and an image id
  over 120 characters made an invalid event id, and keeps its first 103
  characters and 16 hex digits of a hash.
- **Revision chains are checked in linear time** — 16 000 versions of one
  record took 17 s to validate — and **preflight's string columns have
  64-bit offsets**: a subject whose text passed 2 GiB raised a
  `PanicException`.

### Performance

**A window decompresses only what it covers.** The Blosc2 profiles store each
chunk as a grid of independently compressed blocks, and HDF5's filter pipeline
decompresses whole chunks: a 96³ patch over the default 1 MiB chunks
decompressed up to eighteen of them. A windowed read now takes each stored
chunk as it is and has Blosc2 decompress only the blocks the window touches,
outside HDF5's global lock. Whatever that path cannot read exactly as HDF5
would — another codec, a byte order HDF5 converts, an unwritten chunk, a
strided selection — goes through HDF5 as before.

Measured on one machine and the same files, 1.4.4 → 2.0.0, with a new random
window for every read:

- a 64³ image patch 3.9 → 1.4 ms and its eight-class labels 9.3 → 3.5 ms; at
  96³, 6.9 → 3.3 ms and 15.0 → 12.1 ms;
- `open()` and the first 64³ patch 12–13 → 3–3.5 ms;
- four threads reading 64³ patches at once: 7.2 → 1.2 ms per read, wall clock;
- sliding-window inference, 96³ at 50 % overlap: 5.4 → 3.1 ms per window;
- a paired-visit centre, which reads the displacement field around one point,
  10.2 → 1.9 ms;
- the training loader (`balanced`, images only), in one process and with four
  workers on four cores: 172 → 376 and 440 → 600 patches/s at 64³, 110 → 189
  and 269 → 335 at 96³.

The one read it does not speed up is the same chunks again, through one open
dataset, while HDF5 still caches them: one 64³ window re-read twenty times
costs 0.66 ms a read where HDF5's cache answered in 0.13 ms. `medh5 bench`
timed exactly that, which made an image patch look like 0.13 ms; its patch
metrics now read a different seeded window every time.

Also on one machine and the same files: a metadata-only read 0.79 → 0.19 ms,
foreground centre sampling 0.088 → 0.034 ms (0.32 → 0.045 ms at 63 classes).
Choosing a patch window the way
`PatchDataset` does, from a seed per item, costs 4.5 µs instead of 37 µs for a
uniform draw, 44 µs instead of 106 µs for a foreground one and 25 µs instead of
76 µs for `balanced` (the median per draw over 3000 draws): 1.x built a NumPy
generator per item.

## [1.4.4] — 2026-10-02

Release tooling and `README.md` only: **no format change and no change to the
package's behaviour**. It exists, like 1.4.3 before it, because `README.md` is
the package's PyPI long description, and the badge links fixed below could
reach the project page only through a release.

### Added

- **GitHub Releases.** `release.yml` creates one for each tag once PyPI has the
  distributions: its notes are the version's section of this file, with links
  into the repository pinned to the tag, and it carries the same wheel and
  sdist. A tag whose version has no section here now fails before anything is
  published. Releases for 0.5.0 to 1.4.3 were created from this file.
- A Read the Docs badge in `README.md`.
- `README.md`'s Python examples run in the documentation tests, as every page
  under `docs/` already did. It is the page GitHub and PyPI show, and it was
  the only one whose examples nothing executed.

### Fixed

- **`README.md` on PyPI.** The License and Typed badges were relative links,
  which resolve on GitHub but 404 on the PyPI project page, where the README is
  the long description; both are absolute now.
- `README.md`'s "Write a sample" raised `NameError` when pasted; it now defines
  the arrays it writes. "The reasoning behind each" pointed at the
  documentation's front page rather than the design rationale.

## [1.4.3] — 2026-09-27

A documentation release. **No format change and no change to the package's
behaviour**: a review of every page against the code and the specification.
It exists, like 1.1.1 before it, because `README.md` is the package's PyPI
long description — the corrected install note and the new links below could
only reach the project page through a release.

### Removed

- **The `design/` records.** The 1.0 design proposal and implementation plan
  were pre-implementation working documents, unmaintained by their own account.
  The reasoning that still holds — the problems 0.x could not express, the
  alternatives weighed, the costs accepted, the non-goals — is now a maintained
  page, [Design rationale](docs/explanation/design-rationale.md), and the
  benchmark results the proposal carried are on
  [Runnable examples](docs/examples/index.md) beside the scripts that produce
  them. Old links to the proposal redirect to the new page; the plan remains in
  the repository's history. Docstrings and `medh5 bench --help` no longer cite
  the plan.

### Added

- `CONTRIBUTING.md`: setup, the gates, how the specification and the tested
  documentation are kept in step with the code, and the release process.
- Documentation for what existed but was undocumented: `from-nifti
  --fourth-axis` and 4-D series; the `grid` and `roi` classification scopes;
  `identity` transforms; `mask` annotations and `valid_mask`; the Python entry
  points of every converter; `s.index`, `s.fresh_indices`, `s.label_set`,
  `Timeline.interval_days`, `commit()`/`abort()`, `remap_frame_uids`,
  `audit_splits`, `pack`/`unpack`, `grid_patches`; the MONAI adapter's `space=`;
  `PairedPatchDataset`'s change labels and `samples_per_pair`.
- A test holding the conformance page's breakdown of the corpus — valid,
  invalid, samples, collections, mutated — to the corpus. Only the total was
  checked, and the breakdown had drifted.

### Fixed

- **Specification, editorially.** The heading for §5.4 (hierarchy and
  `closure`) was missing although three clauses cite it; §6.3's kind registry
  pointed every geometric kind one subsection early; §4.2 and §8.5 cited §10.3
  and §10.5 for §10.4 and §10.6; §2.4's table of `/meta` members omitted
  `quality`. No requirement changed.
- **Claims the code contradicts.** `dense([65535])` is refused with `E404`, not
  answered with an all-zero plane; contours and meshes do not satisfy the `det`
  profile; the profile table now states §1.3's requirements; a classification's
  `labels` are keyed by class key; `Cohort` has `acquisition_protocol`, not
  `protocol`; `transform_between` traverses a link backwards only where an
  inverse can be evaluated; `portable` shuffles before gzip; the core install
  includes `jsonschema`.
- **Stale numbers.** The conformance page's 39 valid cases, 111 samples and 71
  mutated cases are 41, 113 and 82; the landing page and the training tutorial
  quoted foreground-sampling timings from before 1.4.2.
- **The CLI reference** now matches the parser: `fix --by`, `track --key`,
  `index build` defaults, `dataset split --assigned-by`, `dataset stats
  --set-id` and `--stride`, `--report` and `--json` on the converters, and the
  default annotation ids. `medh5 index build --help` now says `--occupancy` is
  the occupancy map's block size, which is what it always was, not its side
  length.
- Page titles that disagreed with the navigation, links labelled with pages
  that no longer exist, and a tutorial that duplicated the next one.

## [1.4.2] — 2026-09-26

The second patch release under the third audit's plan, and the last of it: the
cohort, curation and registration tools, performance, and hygiene. **No format
change** — 1.4.2 reads and writes the files 1.0 does, and the conformance corpus
is still 117 cases. Most of what changes is a tool answering a question wrongly
without saying so: an agreement score for a comparison that measured nothing, a
split audit that missed one subject in two groups, a verified file whose
attested object had gained an undigested dataset, a box read on another grid
at the wrong size. After this release every one of the twelve "final 1.x"
properties holds.

### Behaviour changes

Read these before upgrading a pipeline. Each is a correction; each can change what existing code
produces or accepts:

- **Agreement scores only what was measured (L-29).** Object agreement now does what voxel
  agreement did: an unmatched object whose class the other annotation never examined is left out
  and listed under `skipped` — identical work scored 0.667 when the second rater had not been asked
  to look for a class — while a matched pair always counts. When neither side has anything to
  compare, `value` is **`None`** rather than 0, `mean_iou` is `None` when nothing matched, and
  `to_record()` refuses rather than writing a 0 into the file. `compare_instances` refuses two
  `index` annotations on different grids (`E101`), and boxes in different spaces or in different
  frames (`E414`): boxes on a 4 mm and a 1 mm grid were compared in raw index units. `compare()` picks the comparison the two kinds
  support and refuses an argument it cannot use; `medh5 agree` goes through it, so two `boxes`
  annotations no longer end in an `AttributeError`, and `--metric` or `--threshold` on the wrong
  kind is an error rather than ignored. **Comparisons across two visits' grids now raise.**

- **An agreement record can be stored in the file it describes (L-38).** `to_record()` keyed
  `per_class` by class name, and the object record put its mean IoU there as `mean_iou`; the schema
  admits only class ids, so every record `medh5` produced failed with `E005` when written. Records
  are keyed by class id now, and the object record carries no `per_class`.

- **Duplicate ids are refused (L-30).** A document that declares one agent or activity id twice is
  refused when read — the reader kept the last, and a no-op `amend` deleted the other from the file.
  The validator reports it through the parse (`E005`; a dedicated code is format 1.1). Two objects
  sharing an `instance_id` inside one `instances` annotation — which §7.4 forbids — are refused by
  the writer and reported by the validator as `E404`. **Such third-party files no longer open or
  validate.**

- **Split audits find a subject stored under two `group_id`s (L-33).** `medh5 splits`,
  `audit_splits` and the cohort check `C202` grouped by the grouping key alone, so one subject in
  train and in test under two keys reported `leaks = 0, ok = True` — the case `make_splits` refuses
  as `C204`. Leaks are now found over subjects and keys together (`anatomy_units`, union–find), and
  a `Leak` lists the keys and subjects it joined. **Audits can report leaks they used to miss.**

- **The frame graph (L-35, S-12).** A transform and the stored inverse its `inverse_id` names are
  one route: mutually declared inverses — what `E505` checks as "mutually consistent" — raised
  `E501` in both directions, and the registration guide told users not to declare them. They now
  resolve both ways. A composite that contains itself is `E501` in the validator and when evaluated,
  where it validated clean and recursed until `RecursionError`. `transform_between` raises
  **`KeyError`** for a key that is not a timepoint, a grid or a frame the file declares; a mistyped
  `"TP1"` returned `None`, the answer for "no registration exists".

- **`verify` fails a file whose attested object gained an undigested dataset (L-26).** `content_id`
  covers the digests present, so `instance_ids` added to a boxes annotation left the file verified
  and changed `tracks()`. In a file that declares a `content_id`, an undigested dataset inside a
  grid, image, annotation or transform is now listed in `VerifyResult.unattested` and makes `ok`
  false (`medh5 verify` and `recompress` print it as `UNSIGNED`; `fix` counts it as needing
  digests). The writer
  digests every dataset, so none of its files is affected.

- **Index coordinates belong to the annotation's own grid (L-32).** `as_slices(grid=)`,
  `to_index(grid=)` and `to_world(grid=)` on an `index`-space annotation read its coordinates as the
  other grid's: a box on a 4 mm grid asked for slices on a 1 mm grid of the same frame came back
  unchanged, covering a quarter of the anatomy. They now convert through world when the frames
  match and refuse (`E414`) otherwise; since §3.3 (1.4.1) a grid without a `frame_uid` relates to
  nothing, which now applies to `world`-space conversions onto another grid too. Boxes with
  `slice_index` are read on their own grid only.

- **RTSTRUCT rasterisation keeps an island inside a hole (Q-18).** Every outer contour on a slice
  was unioned and every enclosed contour subtracted, so an island drawn inside a hole was erased. A
  contour's role now follows its nesting depth, and each slice is filled by the even-odd rule the
  provenance record already named. **Rasterised masks gain those islands.**

- **`recompress --rechunk` chunks the way the writer does (L-28)** — from each grid's `patch_hint`
  or `chunk_hint`, `(1, *spatial)` for stacked encodings — where it let h5py guess and spanned the
  stacked axis the spec keeps at 1, which the validator reported as `W902`.

- **Sampling-index datasets are compressed (P-12)** with the file's label codec, and `occupancy` is
  chunked one class per chunk; it was uncompressed and contiguous. Files with an index get smaller.

- **The CLI reports a missing name as an error, not a traceback (Q-12).** `medh5 convert to-nifti
  FILE <unknown image>` printed a `KeyError` traceback; any lookup error is now `medh5: …` and exit 1.

- **`SampleWriter.deidentification()` with no fields is refused (Q-11).** An `assert` stood there,
  so under `python -O` the call wrote nothing and cleared an existing record.

### Added

- `medh5.torch.FileGroupedSampler` (P-11): shuffles files and yields each file's items together, so
  an epoch opens each file once per worker. At 100 small files and four items each, 304 opens
  became 100 and 0.77 ms per item became 0.30. Every dataset gains `file_groups()`, and the order
  follows the dataset's `set_epoch`.
- `medh5.torch.set_cache_size` is exported and documented, `HandleCache.resize` closes idle
  handles past the new size, and `HandleCache.lease(path)` holds a handle against eviction.
- `compare(a, b, *, metric=, threshold=, classes=)`; `compare_instances(..., classes=)`;
  `InstanceAgreement.skipped`; `VoxelAgreement.class_ids`; `OBJECT_KINDS`.
- `VerifyResult.unattested` and `Diagnosis.unattested`; `ATTESTED_GROUPS`.
- `Leak.groups`, `Leak.subjects` and `medh5.curation.splits.anatomy_units`.
- `SamplingIndex.has_class`; `medh5.storage.chunking.grid_chunks`, `fit_chunks` and
  `field_chunks` — the writer's chunk rule, shared with `recompress`.
- `medh5.transforms.resolve.stored_inverse` and `medh5.transforms.composite.composite_cycle`.
- `medh5 bench` gains `foreground_sample_many_ms`, a foreground draw on a 63-class sample held to
  the same 1 ms target (T-11); `medh5.bench.many_class_measurement` and
  `synthetic_many_class_sample`.

### Fixed

- **The indexed foreground draw is O(1) in the class count too (P-08).** `SamplingIndex` re-read its
  class table once per class it checked and twice per count it looked up: 190 HDF5 reads and 7.6 ms
  per draw at 63 classes. It reads them once per open, and a draw reads the one coordinate it
  picks: one read, 0.10 ms at 63 classes and 0.03 ms at eight (was 0.9). Draws are unchanged for
  a given seed.
- **Plane counts are bounded (P-09).** `layers` and `bitmask` counts read a whole plane before
  slabbing it — 256 MiB for a `uint16` layer at 512³, 1 GiB for a bit-plane. The slab reader now
  descends an axis when one row is over its budget, for every bounded scan.
- **A transcode carries attributes it does not know (Q-13).** `AnnotationHeader.read` dropped them,
  so re-encoding an annotation lost anything a later minor version had added, which `amend` carries
  everywhere else (§16).
- **The handle cache is thread-safe (Q-16).** Lookup and eviction are locked, and the datasets hold
  their handle for the length of an item, so a thread-based loader's eviction can no longer close a
  file another thread is reading.
- **`medh5.monai` names the channel axis (Q-17)**: `original_channel_dim` comes from the grid, where
  `"no_channel"` on an RGB image made `EnsureChannelFirst` add a second channel axis.
- **`medh5 bench --json` prints only JSON (Q-19).** Its progress lines went to stdout ahead of the
  document; they go to stderr.
- **`recompress` of a collection (L-39).** A `.medh5c` shard was rewritten and then raised a
  `KeyError` while verifying the output, which it read as a single sample. Each member is verified.

### Removed

- `medh5.geometry.grid.iter_spatial_slices` and `KNOWN_COORD_SYSTEMS`, which nothing used (Q-11);
  the duplicate `transforms.apply._refuse_outside`.

### Specification corrections (Appendix C, entry 21)

- **§10.1** — a transform and the stored inverse its `inverse_id` names are one route between two
  frames, traversable both ways; a resolver must not count them as two (S-12).

### Conformance, docs, tests and CI

- The `W909-instance-id-two-classes` case puts its conflicting id in two annotations — the
  sample-scoped form Appendix C's §7.4 entry already requires — because inside one annotation a
  shared id is now `E404`. Its expected codes are unchanged; the corpus is still 117 cases.
- A test holds every place the corpus size is written down — README, site, conformance page,
  Appendix C, CI comments — to `len(CASES)` (D-09). Stale pointers in `storage/codecs.py` and
  `.readthedocs.yaml` are corrected; the codec sentence in `CLAUDE.md` now says what each profile
  shuffles (D-10); the performance guide and the training reference say to set
  `HDF5_USE_FILE_LOCKING=FALSE` on NFS, Lustre and GPFS, which medh5 never does for you (D-11).
- `tests/v1/test_release_1_4_2.py` carries every reproduction above; run against 1.4.1, 54 of its 55
  tests fail, and the one that passes guards behaviour that must not change. It includes a
  persistent-worker loader under the `spawn` start method on every platform (T-10).
- CI actions are pinned by commit SHA, and Dependabot proposes their updates; a concurrency group
  cancels superseded pull-request runs; and `release.yml` runs the whole CI on the tagged commit
  before it builds and publishes (K-03).

### The final-1.x properties

With 1.4.2 all twelve hold. The five still open after 1.4.1:

- **2. No reader answers differently for a third party's file** — holds: L-35 (mutual
  `inverse_id`), after L-20 in 1.4.1.
- **5. Nothing materialises a volume to answer a header-shaped question** — holds: P-09.
- **7. Every attestation is re-checkable, and the exit code agrees** — holds: L-26, and L-38 makes
  the agreement record writable at all.
- **10. Every rewrite carries what it does not understand** — holds: Q-13 for annotation attributes.
- **12. Every file a tool writes passes the validator** — holds: L-28 for `recompress --rechunk`.

## [1.4.1] — 2026-09-25

A patch release under the maintenance contract: defect and security fixes, and **no format
change** — 1.4.1 reads and writes the files 1.0 does. A third full audit went after how the files
are *used*: shipped to a collaborator after a scrub, fed to a training loop, recompressed by a
curator, amended by a tool one version behind. It found twelve silent wrong answers and one
security defect, all fixed here. Four fixes needed a little new API, because the defect cannot be
fixed without it — loader item keys, classification rows, `drop_identity`, `--pseudonymise-ids` —
and each is listed under **Added**. `jsonschema` becomes a core dependency. The corpus grows from
116 to 117 cases, and Appendix C now lists twenty corrected clauses.

### Behaviour changes

Read these before upgrading a pipeline. Each is a correction; each can change what existing code
produces or accepts:

- **Security: files that read outside themselves are refused by every tool (F-22).** HDF5 lets a
  dataset keep its bytes in an arbitrary file on the reader's machine (external storage), and lets
  a file reach others through external links and virtual datasets. A crafted file whose extension
  dataset pointed at `~/.ssh/id_rsa` validated clean at `--level strict`, and `recompress --out`
  copied the key into the file a curator then shared. `open`, `amend`, `recompress`, `repack`,
  `pack`, `verify`, `validate` and the 0.x reader now walk the file first and refuse any such
  object by name, before reading a byte of it; `validate` reports E001. Copies no longer expand
  external data. The writer never produced these objects.

- **`scrub` examines the whole file and can exit 1 on files that used to pass (F-14, L-36).** It
  kept a list of places to look, and each audit found string-bearing places the list had missed —
  this time provenance `inputs`/`outputs`, the sample's own ids, timepoint text, per-object JSON,
  and every HDF5 attribute. It now walks every string in `/meta`, every object name, every attribute
  and every string dataset (JSON inside a string is decoded and examined as a mapping), skipping
  only fields named in the code with a reason, and the scan and `--apply` are one traversal, so
  "actionable" means what `--apply` will change. New rules: DICOM UIDs inside provenance references
  (`dicom:<uid>`) get the same pseudonym as the field they came from; filesystem paths keep only
  their file name, or nothing under `--profile strict`; ages over 90 and organization agents are
  reported and handled under `strict`; person names are recognised in any script (`Müller^Hans`).
  **The DICOM importer records where the ids came from** (`identity.id_source`), `scrub` reports
  a `sample_id`/`subject_id` copied from `PatientID`, and **a strict `--apply` exits non-zero while
  an id is still an identifier** — re-mint it, or pass `--pseudonymise-ids --salt ...`. A file
  named after such an id, or after a UID, is reported too. The attestation's profile string now
  says what was not examined. After a DICOM import and a strict apply with `--pseudonymise-ids`, no
  PatientID, UID or date bytes remain in the file.

- **Loader items carry the ignore region, validity and coverage (F-16).** Items gain
  `ignore[ann_id]` (the §7.7 region, wherever the encoding keeps it, *and* any padding),
  `valid[image_id]` (the image's `valid_mask`, never the padding) and `meta["annotated"][ann_id]`
  (one flag per label channel). `label_format="labelmap"` — and `medh5.monai.to_dict` — now write
  **65535 at ignored voxels under every encoding**, where `layers`, `bitmask`, `instances` and
  `probmap` gave 0, and at padding, which was 0 (background). The ignored voxels §7.7 exists to keep
  out of the loss went into it as negatives.

- **Persistent workers draw new patches every epoch (F-17).** The epoch was an `int` on the dataset;
  `DataLoader(persistent_workers=True)` — the documented setup — kept the copy it started with, so
  every epoch repeated epoch 0's patches. It now lives in shared memory and `set_epoch` reaches every
  worker, under `fork` and `spawn`. `ds.epoch` still reads and assigns as before.

- **Two grids without a `frame_uid` are not registered (F-18).** `PairedPatchDataset(align=
  "transform")` compared the two frames with `==`, and `None == None` let every unregistered
  longitudinal sample from the NIfTI and nnU-Net importers through as aligned. It now requires a
  shared frame (§3.3) or a transform; **such samples raise** and say to use `align="none"`.

- **A gantry-tilted DICOM stack is refused (F-19).** It imported as an orthonormal grid, 17 mm off at
  the last of twenty slices at 20°. The refusal names the tilt and the worst slice's offset; a stack
  within half a pixel of its axis imports as before, with a `slice_alignment` decision recorded.

- **`amend` refuses a future major version and keeps a later minor (F-20).** It restamped a 2.0 file
  `1.0`, and a 1.1 file `1.0` while carrying its 1.1 objects. `scrub --apply`, `fix` and
  `seg convert` go through `amend`.

- **Cohort class statistics come from voxel annotations, per sample (F-24).** A sample-level
  classification arrived as "examined, 0 voxels", and `class_weights()` floored it to 1 voxel —
  making a one-bit diagnosis the heaviest segmentation class. Classification, geometric and `mask`
  annotations no longer contribute; `present_in`/`examined_in` count samples, not annotations; and
  `class_weights()` leaves out classes with no voxels, with a warning naming them. **Weights and
  prevalence change for cohorts with non-voxel annotations.**

- **Contradictory demographics split a PatientID (F-25).** An anonymiser's constant `PatientID`
  merged two patients into one subject with two visits. Studies whose `PatientBirthDate` or
  `PatientSex` disagree now become one sample each, with a guess in the report naming both values.
  **Such imports produce more files.** Study-keyed files are named by their whole
  StudyInstanceUID, where `Path.stem` used to chop its last component.

- **`commit()` runs the validator (L-19–L-22).** The writer and the validator each kept their own
  list of rules and drifted apart. `commit()` now runs the validator's structural and semantic error
  rules over the finished file and refuses to write one it would reject (warnings are not raised;
  measured at about 2 ms on a 192×256×256 sample). A displacement field off its grid's lattice is
  refused at `add_transform` (E503); grid `units` outside `mm`, `um`, `m`, `px` and `time_units`
  outside `s`, `ms` are refused (E109); and a `form="ref"` label set is written, where the writer
  refused what the validator accepts. **Files the validator would reject are no longer written.**

- **`add_segmentation` refuses what it used to half-honour (L-23, L-24, L-37).** Passing more than
  one of `masks=`, `probabilities=`, `instances=`, or an `encoding=` the argument cannot use, is E404
  — `masks=` was silently dropped beside `probabilities=`. An ignore region that **overlaps a class**
  is stored as the sibling mask under every encoding, so it reads back whole: under `labelmap` 56 of
  64 voxels came back. **Such files gain a sibling mask instead of an in-band region.**
  `instances=[]` with `annotated_classes=` now writes the "examined, none found" annotation (§7.4)
  it used to refuse.

- **A transcode from `instances` to a dense encoding needs `drop_identity=True` (L-25).** Every
  voxel survived and every `instance_id` did not; `tracks()` then found nothing. With the flag (CLI
  `--drop-identity`) it proceeds and records a `transcode` activity saying identity was dropped.
  **Callers that relied on the silent drop must pass the flag.**

- **`amend` keeps a `portable` file portable (L-27).** New datasets were Blosc2, unreadable without
  `hdf5plugin`. `amend(codec=None)` now means the source's codec family.

- **`unpack` refuses a member name that is a path (L-31).** Names read from a shard were used as
  file names unchecked; `..\..\evil` wrote outside the output directory on Windows.

- **Intensity moments honour `valid_mask` (L-34),** and `total_voxels` counts each measured image's
  grid once. A CT padded outside its reconstruction circle reported a mean of −1180 where the tissue
  averages 40. **Normalisation statistics change for images that declare a valid mask.**

- **A probability map keeps the voxels on its threshold (F-15).** Stored as float16, a vote of 1 in
  3 became 0.33325 and fell below a threshold of 1/3 — at write time. Readers now compare at the
  stored precision, and the writer widens to float32 where float16 would change an answer, so such
  maps take twice the bytes. **Existing files read their boundary voxels as contained.**

- **The validator no longer invents E411 and W904 on large labelmaps (F-21).** It declined to look
  for an ignore voxel past 64M elements and then reported the `uint16` the ignore id requires as an
  error — `medh5 validate` failed the writer's own output at ordinary CT sizes.

- **Every encoder refuses a class id outside 1–65534 with E303 (Q-14)** before the `uint16` cast,
  where numpy 2 raised `OverflowError` and numpy 1.24 wrapped the id silently.

- **`jsonschema` is a core dependency (L-21).** Without it E005 went unchecked and `commit()` wrote
  documents a machine with it rejects. The `schema` extra still installs, and adds nothing.

### Added

Each is needed by a fix above, and none changes the format:

- `Sample.ignore_region(ann_id, roi=None)` and `Sample.valid_region(image_id, roi=None)` — the two
  reads the loaders use, public so a custom loader can make them.
- Loader item keys `ignore`, `valid` and `meta["annotated"]` (F-16).
- `add_classification` accepts rows `(class, value[, scope_id[, scheme, scheme_value]])`, so one
  class can be asserted per lesion, slice or visit (F-23). The mapping form is unchanged.
- `transcode(..., drop_identity=True)`, `transcode_annotation(..., drop_identity=True)` and
  `medh5 seg convert --drop-identity` (L-25).
- `scrub.apply(..., pseudonymise_ids=True)` and `medh5 scrub --pseudonymise-ids`, which require a
  salt (F-14); `identity.id_source`, written by the DICOM importer and by `--pseudonymise-ids`, and
  dropped by `SampleWriter.identity` when the id it describes changes.

### Fixed

- The temporary file an atomic write creates is readable by its owner only until the write is
  done, then gets the target's exact mode (Q-15). It was created with the umask's default,
  usually world-readable, for as long as a large amend of a `0o600` sample ran.
- `describe_filters` no longer names the compression twice (`gzip:4+gzip:4`).
- A commit refused inside a `with` block no longer leaves its temporary file behind.
- W908 builds the overlap graph slab by slab (P-10), where it decoded one volume per class and
  skipped annotations over 64M elements.
- `scrub` treated date keys in `identity.extra`, `cohort` and activity parameters as actionable in
  a file that already records a date shift, so a second `--apply` could never go green; and it
  dropped its own "dates were left alone" note from the report.

### Specification corrections (Appendix C, entries 17–20)

- **§3.3** — a grid without a `frame_uid` is comparable with nothing, another frame-less grid
  included.
- **§7.4** — an `instances` annotation may hold `N = 0` objects: the verified negative tracking
  depends on.
- **§7.5** — `contains` is decided at the stored precision, and a writer must store `data` in a
  dtype under which that matches the input.
- **§7.7** — an ignore region overlapping a class is stored as the sibling mask under every
  encoding. The clause also now names `instances`, as entry 16 already said it did.

### Conformance, tests and CI

- The corpus grows to **117 cases**, with `seg-instances-empty` (N = 0, §7.4).
- `tests/v1/test_release_1_4_1.py` carries every reproduction above, each named for its finding,
  including a strict apply over a sample with a person name planted in every string slot, and
  a no-op `amend` over every corpus file that must either refuse or write a valid file.
- Python 3.14 joins the CI matrix and the classifiers; the minimum-dependency job pins
  `jsonschema` 4.18.

### The final-1.x properties

The third audit added four properties to the eight "final 1.x" means. After this release:

- **9. A tool never reads outside the file it was handed, nor writes outside the directory it was
  handed** — holds: F-22 and L-31.
- **10. Every rewrite goes through one gate** that refuses an unknown major, never lowers the
  version, keeps the codec family, and carries what it does not understand — holds for the first
  three (F-20, L-27); unknown *annotation* attributes (Q-13) are carried by 1.4.2.
- **11. What a loader hands a model carries the file's contracts** — ignore region, validity,
  coverage, a frame check before alignment, fresh randomness each epoch — holds: F-16, F-17, F-18,
  F-24, L-34.
- **12. Every file a tool writes passes the validator at its default level, and the validator never
  reports "absent" because it declined to look** — holds for the writer (L-19–L-22, checked over
  the whole corpus) and the validator (F-21); `recompress --rechunk` (L-28) is 1.4.2's.

## [1.4.0] — 2026-09-04

**The final 1.x release.** The **format version is unchanged**: 1.4.0 reads and writes the files
1.0 does. What changes is what the *writer* does with what it is given. A second full audit went
after the inputs rather than the outputs — an ignore region under an encoding chosen by
measurement, a provenance id a caller picked, an export of a 2-D sample, a DICOM SEG written the
way `highdicom` writes them by default — and found that the package produced good files while
quietly dropping, overwriting or refusing several ordinary things asked of it. Those are the
subject of this release. The corpus grows from 115 to 116 cases and Appendix C now lists sixteen
corrected clauses.

After this, the package is in maintenance against format 1.0.

### Behaviour changes

Read these before upgrading a pipeline. Each is a correction; each can change what existing code
produces or accepts:

- **`add_segmentation(ignore=...)` now keeps the region under every encoding.** It was forwarded
  to the encoder only for `labelmap` and `layers`; under `bitmask`, `instances` and `probmap` the
  array was dropped without a word — no in-band value, no `ignore_mask` attribute,
  `has_ignore_region` reading `False`. With `encoding="auto"` the caller could not know which
  branch they were on, so the same call kept the region for one cohort and lost it for the next,
  and every ignored voxel became a *verified negative* for every annotated class with W904 unable
  to fire. The writer now stores the region where the chosen encoding can hold it: in band for
  `labelmap` and `layers`, and otherwise as a sibling `mask` annotation named `<ann_id>_ignore` —
  same grid, same `prov` — referenced by the header's `ignore_mask` (§7.7). **Files written with
  `ignore=` under those three encodings gain that extra annotation.** Passing both `ignore=` and
  `ignore_mask=` is now E404, and `encode_voxels`, which returns a single payload and so cannot
  create the sibling, refuses rather than dropping. `select_encoding` takes `ignore=` and costs the
  `uint16` widening an in-band ignore forces, so the encoding is chosen with the region in view.

- **Provenance ids are never silently overwritten.** `Provenance.add_agent` and `add_activity`
  assigned into a dict, so `person("Alice", agent_id="s2")` followed by `software("tool")` — whose
  automatic id is also `s2` — left one agent, the software, and every activity that named Alice
  named the tool. The file validated, because every reference resolved; it resolved to the wrong
  node. Both now refuse a duplicate id (`replace=True` for the one legitimate rewrite), and the
  writer's automatic ids skip taken ones. **Code that relied on overwriting now raises.**

- **`transcode_annotation(..., "mask")` is refused (E404).** A `mask` has no classes, so the
  conversion OR-ed every class into one volume and left `class_ids` and `annotated_class_ids` empty
  under `task="segmentation"` — the coverage contract gone, in a file that still validated.
  `TRANSCODABLE` is now the whole truth; build a deliberate union with `encode_mask` and
  `add_mask`. The CLI was already safe, since `seg convert --to` restricts its choices.

- **`scrub --apply` acts on everything it flags, re-scans what it wrote, and can exit 1.** It
  cleaned `extra` and `acquisition` only, then wrote a §11.4 record whose profile string reads
  "quasi-identifiers removed" and exited 0 — over files still carrying the patient name,
  institution and referring physician it had itself reported in `identity.extra` and `cohort`.
  Three places were never scanned at all: `activity.params` (where a converter naturally stores
  `OperatorsName`), `activity.tool`, and `quality.issues[].note`. All are scanned and cleaned now,
  along with agent roles and qualifications; under `--profile strict` a person agent's name becomes
  a stable pseudonym and flagged free text is removed. `apply` then re-runs the rules over the file
  it wrote, records `remaining` and `remaining_actionable` in the deidentify activity's params, and
  **the CLI exit code follows that re-scan**. `identity.sample_id` and `identity.subject_id` are
  reported and never rewritten — every manifest, split claim and cross-file join names the sample
  by them.

- **Exported NIfTI files carry their rescale.** `Image.read()` returns *stored* values, and both
  exporters wrote them with no `scl_slope`/`scl_inter`: a CT imported from DICOM with intercept
  −1024 left every voxel 1024 HU too high, in `to_nnunetv2`'s only path and in
  `to_nifti(physical=False)`, with nothing in `dataset.json` or the report to say so. Both now go
  through one `write_nifti`, which records the scale, so every conforming reader — nibabel,
  SimpleITK, and therefore nnU-Net — sees the physical values. **The numbers a downstream tool
  reads from those files change, which is the correction.**

- **2-D samples can be exported.** `convert_world` was defined for 4×4 affines only, so both
  exporters stopped on a 2-D grid's 3×3 with "RAS↔LPS conversion is defined for 3-D affines" —
  and the import → export round trip the converters page promises was broken for every 2-D sample,
  although both importers accept one. It handles a 2-D affine now, and `embed_plane` puts it back
  in the 4×4 a NIfTI file carries (§3.6).

- **A DICOM SEG that omits its empty frames imports.** `omit_empty_frames=True` is highdicom's
  default and the common form in the wild. The reader assembled a volume from the planes present —
  two slices, spaced by the distance between them — and the import compared *that* shape with the
  grid's and refused with E405 "drawn on a different reconstruction". Frames carry their own
  `ImagePositionPatient`, so they are now placed on the grid by position (§3.3), and shape
  agreement is a question about rows and columns. A frame that genuinely is not a slice of the
  target grid is still refused, now naming the frame. `read_dicom_seg_frames` and `place_frames`
  are the new public entry points; `read_dicom_seg` keeps its shape and meaning.

- **`recompress` verifies its output, and its exit code follows.** `content_id_preserved` compared
  the attribute the function had just copied with itself — true by construction, including on a
  file whose bytes were corrupted before the run, which reported "content_id yes", exited 0, and
  failed `verify()` immediately after. The re-encoded file is now read back and checked against the
  digests it carries. **It takes roughly twice as long, and can now fail on a file that was already
  broken.**

- **`commit(digests=False)` keeps `content_id`.** It skipped the root address entirely, so an
  amend that used the flag for speed turned an addressed file into an unaddressed one. It now
  stamps the datasets that carry no digest — on an amend, the ones just written — and always
  computes the root. `create(...).commit(digests=False)` therefore stamps everything, since nothing
  has a digest yet.

- **`Timepoint.extra`, `QualityRecord.extra` and `Activity.extra` are gone, and `Agent.extra` with
  them.** The schema closes all four objects, so anything placed there failed E005 at `commit()`:
  an extension point that could not reach a file. An unknown key is now E005 where it is parsed,
  with the field named. `Agent` gains the `organization` field the schema has always had, and
  `Timepoint` gains `description`.

- **`open_collection` takes no `mode`.** `Collection` has no mutating method, as `medh5.open`
  stopped taking one in 1.3.0.

- **`dataset check` C202 groups by the §12.2 grouping key**, as `medh5 splits` does, and names the
  subjects involved. A family or longitudinal group straddling two partitions was a LEAK in one
  tool and clean in the other, on the same files.

### Fixed

- `info`, `tree`, `verify`, `timeline` and `track` accept a `.medh5c` shard with `--key`; `info`
  and `tree` without one describe the shard. Six help strings promised collection support the
  commands did not have.
- `Sample.reference_grid` raises E111 on a file with no grids, where it raised a bare `IndexError`.
- `to_nnunetv2` lists in `report.outputs` only the files it wrote; it named a `labelsTr` path
  whether or not the sample had the annotation.
- `recompress` no longer copies the root attributes twice.
- The annotation reference gave `closure` as `"explicit" | "complete"`; the values are `explicit`
  and `implicit`. `seg convert` is documented with the five targets it accepts.

### Performance

- **`layers` and `bitmask` cache their class tables per open annotation**, as `instances` has since
  1.3.0. `dense()` asked the table once per class: 63 HDF5 reads of the same table for a 63-class
  annotation packed into four layers.
- **`storage.index._occupancy` is vectorised** — pad, reshape, `any` over the block axes — where it
  was a Python loop over every coarse block: about 80 ms per class at 256³ and 0.6 s at 512³, so
  `build_index` on a 200-class annotation spent minutes on a map measured in kilobytes.
- **`voxel_counts()` answers every class in one pass** where the encoding allows it: a slab
  `bincount` for `labelmap` and `layers`, a per-plane population count for `bitmask`. It decoded
  each class separately, and `build_index` and the unindexed path of `dataset stats` both go
  through it.

### Specification corrections (Appendix C, entry 16)

- **§7.7** names `instances` beside `bitmask` and `probmap` as an encoding whose ignore region must
  live in a separate `mask` annotation. It has no in-band value either, so the clause left a
  conforming writer with nowhere to put one.

### Conformance and tests

- The corpus grows to **116 cases**, with `seg-bitmask-ignore-mask`: §7.7's separate-mask form as
  the writer now emits it, partial coverage and W904 silent.
- `tests/v1/test_release_1_4_0.py` carries every reproduction above, each named for the finding it
  came from, and each failing on 1.3.0.
- `tests/v1/test_docs_python.py` **executes every Python example in the documentation** against a
  real sample. `test_docs_examples.py` checked the prose and the CLI flags but never ran a line of
  Python, and one of the examples it passed was the `ignore=` call this release exists to fix. A
  block whose only failure is a name it never binds is a fragment and is tolerated; anything else —
  a method that no longer exists, a signature that changed — fails. A block that cannot be executed
  says so on the page with `<!-- illustrative -->`.

## [1.3.0] — 2026-09-04

The last 1.x feature release. The **format version is unchanged**: 1.3.0 reads and writes files 1.0
does, and every file 1.2.x wrote validates under it unless it carried one of the defects the new
rules exist to catch. The corpus grows from 103 to 115 cases, so that every *clause* behind a code
has a case and not only every code. Five specification clauses are corrected (Appendix C now lists
fifteen), none of which changes what a conforming file looks like.

### Behaviour changes

Read these before upgrading a pipeline. Each is a correction; each can change what existing code
produces or accepts:

- **h5py 3.13 is the floor** (was 3.10). The new minimum-dependency job found that every `amend`
  — the copy-on-write path, so also `scrub --apply`, `fix`, `recompress` and `seg convert` —
  segfaults inside HDF5's `H5Ocopy` on the HDF5 builds h5py 3.10–3.12 bundle (1.12.2 and
  1.14.2). The trigger is an object with more than eight attributes, which the latest file
  format stores densely, when one of them is a variable-length string; every grid has nine,
  and the isolation is reproducible in ten lines of h5py without this package. h5py 3.13
  bundles HDF5 1.14.6, where the copy succeeds. `pip install medh5` resolves the floor; an
  environment pinned to an older h5py could not amend a file, whatever this package did.

- **New validator refusals.** A `labelmap` stored `uint16` whose ids fit `uint8` and carries no
  ignore voxel is **E411** (§7.1 has always said `uint8` MUST be used); a grid with a `time` axis and
  no `time_values` is **E109** (§3.2); a dangling `ignore_mask`, `derived_from` entry or image
  `valid_mask`, or an `ignore_mask` naming anything but a `mask` annotation, is **E413**; a transform
  whose `prov` or `metrics` names nothing is **E601**/**E602**; a `splits[].assigned_at` or
  `deidentification.date` that is not RFC 3339 is **E604**; and a `digest_algo` outside `sha256`,
  `sha512`, `blake2b` is **E703**. The writer refuses the same things at `commit()`.

- **`medh5.open(path, "r+")` is refused.** `Sample` has no mutating method and never had one;
  `amend` is the edit path. `open(path)` and `open(path, "r")` are unchanged.

- **`add_grid` refuses a `time` axis without `time_values`** (E109), as §3.2 requires and as the
  validator now checks.

- **Label-set keys minted by the converters are schema-valid.** `sanitize_key` replaces the four
  private copies of the key sanitiser, and it holds to the schema's `^[a-z0-9][a-z0-9_]*$`: an ROI
  named `GTV-1` becomes `gtv_1`, where the old copies minted `gtv-1` and the write then failed E005.
  Names that were already valid keys are unchanged.

- **`derived_from` carries annotation ids, not paths.** The RTSTRUCT importer wrote
  `annotations/<id>`; it writes `<id>` now, as §6.2 says. Readers and the validator accept both.

- **Imported RTSTRUCT polygons record their plane** as `(0, k)`, so `by_plane()` works on them;
  the `contour_plane` column, and with it the `content_id` of a freshly imported structure set,
  differs from 1.2.x's.

- **DICOM imports record `PhotometricInterpretation` in `acquisition`**, which changes the
  `content_id` of a freshly imported study; and `from_dicom` **warns** rather than stays silent when
  a tree holds enhanced multi-frame objects it cannot read, or a `MONOCHROME1` series. A warning makes
  `report.ok` false and the command exit non-zero.

- **`Entry.field` accepts manifest fields only.** `--group-by to_json` used to "work".

- **`compute_content_id` and `verify()` raise E703 for an unknown `digest_algo`** instead of a
  `ValueError` from `hashlib`.

- **`medh5.monai.to_dict`, `dataset stats` and the other 1.2.1 changes stand** — see that entry.

### Specification corrections (Appendix C, entries 11–15)

- **§2.1** `digest_algo` names `sha256`, `sha512`, `blake2b`, the algorithms every Python ships,
  rather than `blake3` and `xxh3-128`, which the reference implementation never carried and reported
  as E703 malformed. Any other value is E703, normatively.
- **§7.5** `threshold` is a spec-defined `probmap` attribute, default 0.5, covered by `content_id`.
  §7.6 already said transcoding was "lossless only under a declared threshold" and the reader honoured
  one; nothing defined it. `add_segmentation(probabilities=..., threshold=0.3)` writes it.
- **§3.4** The sentence requiring a *validator* to report an annotation compared with an image on
  another frame described a query, not a file. It is reader guidance now; the refusal lives in
  `PairedPatchDataset`.
- **§5.1** A collection carrying one label set at `/` with `uri = "medh5:/label_set"` is MAY, not
  SHOULD; the reference implementation neither writes nor resolves it.
- **§7.1, §3.2, §15.2** The `uint8` rule and the `time_values` requirement are validated; E603's
  summary reads "unknown agent or activity type", as the table always said.

### Fixed

- **JSON, Markdown and checksum files are read and written as UTF-8 on every platform.**
  Manifests, splits, reports, nnU-Net `dataset.json`, NIfTI sidecars and the published
  conformance suite used the platform default, which on Windows is cp1252 — a manifest with a
  non-ASCII subject id could not be read back there. The new Windows job found it in the docs
  tests before anyone found it in the package.
- **`create` and `amend` commit on Windows.** The `fsync` before the rename into place opened
  the finished file read-only, and Windows flushes only through a writable handle, so every
  write failed at commit with `EBADF`. The writable handle is opened in binary mode, because
  the C runtime's text mode strips a trailing 0x1A from a writable file on open, and one HDF5
  file in 256 ends with that byte. Nothing had run there before the Windows job existed.

### Fixed — performance

- **A displacement field was read whole on every evaluation.** `PairedPatchDataset(align="transform")`
  asks for one displacement per training item, and `displacement_at` answered by decompressing the
  entire field each time. Linear interpolation now reads the bounding window of the query points,
  padded by one voxel — kilobytes rather than gigabytes on a 512³ field — with the `zero`/`error`
  cases decided against the full extent and the `nearest` clamp landing where it always did, so the
  result is bit-identical to the whole-field read. Cubic interpolation still reads the field whole,
  because a spline's coefficients are global and a windowed answer would depend on its neighbours.
  `medh5 bench` gains a `paired_center_ms` row measuring it on a synthetic two-visit sample.
- **`instances` re-read its columns on every access**: `dense()` fetched `boxes` and `class_ids` once
  per class and `crop()` three datasets per object. The per-object columns are read once per open
  annotation; only `mask_data` is read on demand, one object's slice at a time.
- **`labelmap.has_ignore_region` and `mask.summary()` materialised the whole volume** to answer a
  header-shaped question; they scan in bounded slabs, as `layers` already did, through one helper.
- **`Sample.resolve_frames(a, b)`** resolves between two frame uids and memoises per handle;
  `transform_between` and the paired loader go through it, so a pair is resolved once, not once per
  item.

### Changed — structure

- **`SampleWriter` lives in `medh5/writer.py`.** `sample.py` was 2 400 lines holding the reader and
  the writer; every name is still importable from `medh5.sample` and from `medh5`.
- One `medh5/io/_common.py` (`sanitize_key`, `sanitize_stem`) replaces six private copies; one
  `medh5/_optional.py` says how to install every optional dependency; one `indent` in the CLI; one
  `EXTRAPOLATIONS`.
- `transforms.base.frame_graph` follows the resolver's `can_invert`, not the file's `is_invertible`
  claim, so the graph it draws is the one `resolve_between` walks.
- `PatchSampler` no longer auto-selects a `mask` annotation, which has no classes to draw from.
- Temporary files are named with a pid **and** a uuid, so two threads amending one path in one
  process cannot collide.
- `open_collection`, `open_any` and `pack` check the format major through the same
  `require_major` as `open`; a 2.0 shard used to open through the collection door.
- An empty polygon list is a valid `contours` annotation when `annotated_classes` names what was
  looked for, as an empty box set already was.
- Fifty `# noqa: PLR2004` comments guarded a rule family that was never selected; `RUF100` is on and
  every suppression that suppressed nothing is gone.

### Changed — tooling

- The package version is written once, in `medh5/__about__.py`; `pyproject.toml` declares it dynamic
  and the release workflow checks the tag against it. `license` is the SPDX string `MIT`.
- CI gains a **Windows** job and a **minimum-dependency** job (`numpy 1.24`, `h5py 3.13`,
  `hdf5plugin 4.1` on 3.10), asserts numpy 2 on the matrix, and prints skips.
- `.pre-commit-config.yaml` pins the ruff the `dev` extra pins. `.hypothesis/` is ignored. The
  README's static coverage badge is gone; the number lives in CI.

## [1.2.1] — 2026-09-04

A correctness patch from a full audit of 1.2.0. The **format version is unchanged**: 1.2.1
reads and writes exactly the files 1.0 does, and the conformance corpus is the same 103 cases.
Every fix below was reproduced against 1.2.0 before it was made, and each ships with a
regression test that fails on 1.2.0 (`tests/v1/test_regressions_1_2_1.py`).

### Behaviour changes

Read these before upgrading a pipeline. Each is a correction; each can change what existing
code produces or accepts:

- **`medh5 dataset stats` and `compute_stats` now measure physical values.** Each image's
  rescale is applied before the moments are taken, which is what the loaders read with
  `physical=True`. For any rescaled image — every CT the DICOM importer writes — the mean,
  standard deviation, minimum and maximum change, and so does `normalization()`. `--stored`
  (or `physical=False`) reproduces the 1.2.0 numbers, and the result records which convention
  it used under `physical`.

- **A single-timepoint sample whose grids omit `timepoint` now resolves to that timepoint.**
  `Sample.grids`, and everything downstream of it — `Image.timepoint`, `Annotation.timepoints`,
  `at()`, `tracks()`, `medh5 timeline`, `VolumeDataset(timepoint=...)` — used to answer as if
  such grids belonged to no visit. They answer for the one declared visit now. The file is not
  rewritten; the writer's view is still the stored attributes.

- **`medh5.monai.to_dict` returns `int64` label tensors**, not `int16`. Class ids above 32767
  no longer wrap, and the ignore id comes through as `65535` rather than `-1`.

- **`from_nifti` keeps a scaled file's stored dtype and records `scl_slope`/`scl_inter` as the
  image's rescale.** Physical values are unchanged; the stored dtype and `content_id` of such
  imports differ from 1.2.0's.

- **`from_dicom` keys `series_uids` by image id** (`CT_tp0`), which is what spec §3.7 says the
  key is, rather than by modality (`CT`).

- **Converter-written single-visit samples name `tp0` on their grid.** `from_nifti`,
  `from_nnunet` and `medh5 bench` now write the attribute the reader resolves anyway, so a
  third-party reader sees it too. The `content_id` of a freshly converted file therefore differs
  from one 1.2.0 wrote from the same input.

### Fixed — silent wrong answers

- **64-bit instance ids wrapped to 32 bits on read.** 1.1.0 widened the `instances` encoder to
  `uint64` so `2**32 + 7` would stop becoming 7; the reader then cast the column back to `uint32`
  and it became 7 again, in `instance_ids`, `instances()`, `tracking()` and every longitudinal
  join built on them. The §8 kinds had the mirror defect on the write side: `boxes`, `obb` and
  `keypoints` hard-cast `instance_ids` to `uint32` while their reader returned `uint64`. One
  `instance_id_dtype` helper now serves every kind, the reader widens and never narrows, and the
  round trip is tested end to end rather than on the payload dtype alone.

- **Cohort statistics were computed on stored values while the loaders read physical ones.**
  See **Behaviour changes**. A z-score built from `stats.normalization("CT")` normalised the
  stored counts, not the HU the model was trained on.

- **Single-timepoint files written by the converters were invisible to every timepoint-aware
  reader.** §3.7 makes the grid attribute optional when one timepoint is declared, and
  `from_nifti`, `from_nnunet`, `migrate` and `medh5 bench` all omitted it, so on their output
  `sample.at("tp0").images` was empty, `medh5 timeline` printed `-` in every column, and
  `sample.tracks()` reported every lesion as *unexamined* at the only visit there was, with the
  real coverage filed under an empty-string key. The reader resolves the implicit timepoint once,
  on `Sample.grids`, and `medh5 track` on such a file now says *present*.

- **`medh5.monai.to_dict` narrowed labels to `int16`.** See **Behaviour changes**.

- **`from_nifti` read `scl_slope`/`scl_inter` into the report and then ignored them**, storing
  nibabel's scaled `float64` volume as `float32` with no rescale attribute — three times the bytes
  for the same numbers, and W907 on the converter's own output. The stored dtype is kept and the
  scale recorded, as the DICOM importer already did with the modality LUT. Mask files carrying a
  scale are thresholded after it is applied.

### Fixed — hard failure on ordinary input

- **A DICOM study holding two series of one modality could not be imported.** Images were named
  `{Modality}_{tp}`, so a T1 and a T2 in one study stopped the whole import with
  `grid 'mr_tp0' is already declared`. A visit with several series of a modality now numbers them
  in `SeriesInstanceUID` order — `MR_1_tp0`, `MR_2_tp0` — records the assignment as a decision,
  and keys `series_uids` by image id. Single-series modalities keep their 1.2.0 names.

### Fixed — statements that did not match the code

- The README that `medh5 conformance publish` writes said the suite held "one collection"; it
  holds four. The count is now computed from the cases rather than typed.
- `medh5 verify` printed `content_id None` for a partial pass and `content_id False` for a
  mismatch. It says `not verified` and `MISMATCH`.
- The `medh5.torch` docstring called `worker_init_fn` "not optional"; the code, the README and
  the reference say recommended, and the docstring now agrees.
- `medh5/sample.py` carried the `content_id` root-attribute docstring under the wrong constant.

## [1.2.0] — 2026-09-02

A correctness release. The **format version is unchanged**: 1.2.0 reads and writes
exactly the files 1.0 does, and the conformance corpus is the same 103 cases. The
package takes a minor bump because three fixes change what existing code produces or
accepts — see **Behaviour changes** below before upgrading a pipeline.

The two headline fixes are both silent-wrong-answer bugs found by review of the
documentation restructure in 1.1.1: a paired patch aligned by a registration belonging
to another modality, and `dense()` answering for the reserved ignore id under one
encoding of six.

### Behaviour changes

Read these before upgrading a pipeline. Each is a correction, and each can change what
existing code produces or accepts:

- **`Patch.used_index` is three-state.** It no longer defaults to `True`, so
  `if not patch.used_index:` now matches uniform draws, which it did not before. See
  **Changed** below for what each value means.

- **`dense()` raises `E404` for the reserved ignore id** under every encoding, where it
  previously returned an all-zero plane — or, under `mask`, the whole volume. Code that
  followed the old documentation and read the ignore region with `dense([65535])` must
  switch to `ignore_mask()`, or to the `mask` annotation named by `header.ignore_mask`.

- **`PairedPatchDataset(align="transform")` now refuses pairs it used to accept.** The
  refusal itself is not new — it has raised for unrelatable frames since 1.0.0 — but it
  was reached by asking about the two *timepoints*, so a pair whose own grids had no
  transform still resolved through a registration belonging to another modality at the
  same visits, and paired silently. That question is now asked frame to frame, so those
  pairs reach the refusal instead. A multi-modality cohort that trained without
  complaint may now stop at the first such file; that file was contributing patches
  from mismatched anatomy. `align="none"` reads the same index window from both visits,
  as before, and is unaffected.

### Fixed

- **A paired patch could be aligned by another grid's registration.**
  `PairedPatchDataset._map_center` selected its source and target *grids* from the
  configured images but resolved the transform between the two *timepoints*, which
  searches every frame of one visit against every frame of the other and returns the
  first path it finds. A visit holding a CT grid and a PET grid on different frames,
  with only the CT pair registered, moved the PET pair by the CT registration:
  measured, a 10 mm CT shift displaced PET centres by 10 voxels. Nothing raised — the
  patches came back the right shape from the wrong place, which is the failure the
  guard beside it exists to prevent.

  Resolution is now directly between the two grids' `frame_uid`s, and the error names
  the grids rather than the visits. Asking `transform_between` for the two *grid ids*
  does not close this: grid ids and timepoint ids are separate namespaces — §2.3 scopes
  uniqueness to the group — so a grid may legitimately be named `tp0`, and
  `Sample._frames_for` matches a timepoint before a grid, which puts the whole visit's
  frames back in play for precisely the files most likely to be affected. That
  precedence is now documented on `transform_between`, and the preflight recipe in the
  longitudinal guide resolves on frames rather than grid ids.

- **`dense()` answered for the reserved ignore id instead of refusing.** `65535` is
  not a class (§5.2 says it **MUST NOT** appear in `classes`), so no encoding can
  return a plane for it — but every encoding could return an all-zero one, which is
  indistinguishable from a class examined and found absent. Documentation shipped
  `dense([65535])` as the way to read the ignore region on the strength of that shape.
  It now raises `E404` naming `ignore_mask()` and `header.ignore_mask`. The check is in
  `resolve_classes`, which every encoding's `dense` routes through.

  `mask` did not, initially. It overrides `dense` and discarded the argument, so five
  encodings refused and the sixth returned its whole volume — reading as "ignored
  everywhere", the worst of the available wrong answers. It now validates an explicitly
  supplied class and still ignores it for selection, since a `mask` has none (§4.4);
  argument-free `dense()` is unchanged. All six are covered by a parametrized test,
  which the first fix shipped without.

### Changed

- **`Patch.used_index` is three-state and no longer defaults to `True`.** A uniform
  draw consults no index, and reported `True` because that was the field's default —
  so anyone logging it to find out whether a cohort was indexed got `True` from every
  uniform patch of a `balanced` run. It is `True` when the index answered, `False` when
  the foreground was scanned, and **`None` when the question did not arise**. Code
  testing `if patch.used_index:` sees no change for foreground draws; code testing
  `if not patch.used_index:` will now match uniform draws, which it did not before.

### Added

- **Help text for every CLI argument.** 93 of 196 arguments carried none, so
  `medh5 convert to-nifti --help` documented one of its six. All 189 non-structural
  arguments have it now. Two documentation defects in this cycle — `--source ct/*.dcm`,
  which argparse rejects, and `--out cold/`, which raises `IsADirectoryError` — were
  written because `--help` could not settle the question.

## [1.1.1] — 2026-08-30

A documentation release. **No code changes**: the library, the format and every
diagnostic code are identical to 1.1.0. It exists because `README.md` is the package's
PyPI long description, so the corrected performance claim below could only reach the
project page through a release.

### Fixed — claims that were true only under a condition nobody stated

- **"Foreground sampling is O(1) in the volume" needed an index, and no page said so.**
  The claim holds only after `build_index()`, which `README.md` never mentioned and its
  own write example did not call. `medh5 bench` reports 0.9 ms because it builds an index
  on the sample it generates; a reader following the README got the unindexed path, which
  scans the labels — 30 ms on the benchmark's own 12.6 Mvox volume, 312 ms at 512³, growing
  with the volume while the indexed draw stays flat at 0.92 ms. `README.md`, `docs/index.md`
  and the `docs/training.md` performance table now state the precondition and give both
  numbers. `docs/training.md`'s prose already did.

- **`medh5 bench` prints "all targets met" while reporting a number below a documented
  target.** Four of the five rows in the performance table carry a target `bench` verifies;
  sustained throughput depends on worker count, so it is measured and printed without one.
  A default run reports ~330 patches/s against a table listing ≥ 400, then declares success —
  a statement about the four checked rows that reads as one about all five. The table now
  says which rows are checked and that the documented figure needs `--workers 4`.

- **"Reproduce it with `medh5 bench`" was attached to a two-part claim `bench` half
  reproduces.** The 0.x comparison figures cannot be rerun: there is no 0.x code left in the
  package. Scoped to what `bench` actually measures.

### Fixed — behaviour changed in 1.1.0 that the docs still described as it was

- **`docs/converters.md` documented none of 1.1.0's converter refusals.** The DICOM
  per-slice agreement checks on `ImageOrientationPatient`, `PixelSpacing` and
  `RescaleSlope`/`RescaleIntercept`, the refusal of a missing or wrong-length tag, and the
  reason those refusals carry no diagnostic code were all absent — as were nnU-Net's
  `_same_grid` check on channels and label volumes, and the `E402` export refusal. A reader
  hitting one of these had nothing to consult.

- **`slice_index` was undocumented outside the specification.** 1.1.0 added enforcement at
  three layers; `docs/annotations.md` did not mention the column at all. It now documents the
  per-box rule and the 2-D-box idiom it exists for.

### Fixed — statements that did not match the code

- `docs/annotations.md` said "Twelve kinds"; `ANNOTATION_KINDS` has thirteen. `mask` (§4.4)
  was the one missing, and `docs/python-api.md` omitted it from `ann.kind` as well.
- `docs/curation.md` listed nine activity types; `ACTIVITY_TYPES` has ten. `transcode` was
  missing. `docs/python-api.md` had all ten.
- `docs/conformance.md` described the published corpus as "samples, and one collection". It
  ships four collection cases out of 103.
- `docs/cohorts.md` listed `C204` in its cross-file-check table, under `medh5 dataset check`,
  which never emits it — it is raised by `make_splits` when a `--group-by` field cuts across
  a subject. `check` reports the realised leak as `C202`. Both are now described where they
  belong.
- A `docs/cohorts.md` example used `stats.images["CT"].mean, .std, ...` shorthand inside a
  `python` block. The attributes are real; the syntax was not. It was the only fenced
  `python` block in the docs that would not parse.
- The `medh5 fix` link from `docs/file-format.md` pointed at a mis-generated anchor.
- **1.1.0's own changelog attributed `E101`/`E202` to a refusal that deliberately carries no
  code.** `_same_grid` was made uncoded in the same release that gave nnU-Net import its
  `_same_grid` check, so the entry describing that change was wrong the day it shipped. It is
  corrected in place, and the codes are named in `docs/converters.md` only to say which
  conditions they are *not*. Worth recording because the sweep for this release reproduced the
  error before catching it: the first draft of the converter documentation copied the codes
  straight out of the changelog. A wrong code in prose propagates the way a wrong constant
  does.

### Verified — no change needed

Checked mechanically against the code rather than read for plausibility: 103 documented CLI
invocations parse against the real parser with every flag existing on the command it is shown
with; every documented import resolves and every `python` block parses; the writer's 24
documented methods, `Sample`'s 15 members and the `Grid`/`Image`/`Transform`/`Tracking`
surfaces all exist; the §15.2 table and `medh5.errors.CODES` agree on all 71 codes, and
`CHECK_CODES` matches `docs/cohorts.md` on all 12 including their wording; the conformance
counts (103 cases, 38 valid, 65 invalid, 71 mutated) are exact; the codec profile table, the
L3 fallback and the chunk bounds match `medh5/storage/`; every internal doc link and anchor
resolves; and the getting-started example reproduces its documented output exactly, down to
the voxel counts.

## [1.1.0] — 2026-08-30

A correctness release from a full review of the library. The **format version is unchanged**: 1.1.0
reads and writes exactly the files 1.0 does, and `__format_version__` stays `"1.0"`. The package
takes a **minor** bump rather than a patch because several fixes change what existing code produces
or accepts — see **Behaviour changes** below before upgrading a pipeline.

Two clauses of the specification were corrected (Appendix C now lists ten), and neither changes what
a conforming file looks like: one pins a rounding rule that was under-specified, the other writes
down an exclusion the reference implementation already applied.

### Fixed — `slice_index` shorter than its boxes dropped objects

- **A `slice_index` naming fewer planes than there are boxes silently lost every box past its end.**
  `slice_index` records the plane each 2-D box sits on (§8.2), so it is a per-box column like
  `class_ids` — but it is written after `_object_columns`, which validates the columns it builds and
  never sees this one. A short one was written without complaint, `medh5 validate` reported no error,
  and `as_slices()` skipped the boxes it did not reach, leaving each of them a degenerate
  zero-thickness slice that selects **no voxels at all**. Two boxes with `slice_index=[5]` returned
  one object and an empty region, with nothing anywhere saying an object had been dropped. The bounds
  guard that produced this was defensive — it existed to avoid an `IndexError` — and turned a crash
  into silent loss of ground truth. Now refused at all three layers: the writer rejects a mismatched
  `slice_index` (`E405`), `as_slices()` refuses rather than skipping so files written before the
  check are caught, and a new semantic rule reports it from `medh5 validate`.

- **A `slice_index` naming a plane outside the grid was clamped to its edge.** On an 8-slice grid,
  `-1` became slice 0 and `99` became slice 7 — so a box recorded against a plane the grid does not
  have was silently moved to a plane nobody drew on, written without complaint and validating clean.
  Clamping is the same class of mistake as the bounds guard above: it makes a wrong input look like a
  valid one. The plane is refused now, at all three layers.

- **A `slice_index` of shape `(N, K)` passed the length check and raised a raw `TypeError`.** The
  check read only the leading axis, so `[[3, 4], [6, 7]]` for two boxes wrote and validated clean,
  then `int(planes[i])` failed on a whole row. §8.2 gives `slice_index` shape `(N,)`; the full shape
  is checked now.

- **The rule is stated once.** The writer, `as_slices()` and the semantic validator each enforced
  "one entry per box" separately, which is exactly how `check_chain()` and the validator came to
  disagree about composite units. All three call `check_slice_index` now, and a test drives a file
  the writer never saw through the reader and the validator to hold them to the same answer.

- **Four more per-element columns had the same hole.** Chasing the cause rather than the report
  found that `slice_index` was not special: every column appended *after* `_object_columns` escaped
  its length loop, and only the two somebody happened to write a check for — `names` on points,
  `normals` on a mesh — were guarded. `class_ids` and `weights` on a point set, and
  `vertex_class_ids` and `mesh_class_ids` on a mesh, were not. A three-point annotation carrying one
  `class_id` wrote cleanly and validated clean, leaving two landmarks silently unlabelled. All six
  columns now route through one `_per_element` helper, so a new column cannot be added without the
  check — the previous arrangement guarded exactly the columns someone remembered to guard.

### Fixed — converters that assumed instead of checking

A second review pass covered the three subsystems the first one never reached: the DICOM family,
the cohort tooling, and the §14 storage claims. The converter findings share one shape — read an
attribute from element zero, assume the rest of the stack agrees, never check — and each produces a
file that looks entirely well-formed.

- **nnU-Net channels and label volumes were filed onto the first channel's grid, whatever their own
  geometry.** `from_nnunetv2` kept `geometry or geo`, so a second channel at a different spacing or
  origin — or a label volume resampled by some other tool — was written onto channel 0's grid with
  its voxels intact and its position silently wrong. Both now go through the same `_same_grid`
  refusal `from_nifti` has always used (§3.2, uncoded — see 1.1.1).

- **nnU-Net export wrote an all-background label volume for most real datasets.** `_labelmap_for`
  matched classes by their `dataset.json` name, but the import sanitises that name into the label-set
  key, so any dataset naming a class `"Tumour Core"` or `"GTV"` resolved nothing — and the failure
  fell into a bare `except: continue`. `dataset.json` was written listing every class, the label
  files were the right shape, and every voxel was 0. Classes are matched by id now (the import keeps
  nnU-Net's own integers precisely so they can be), and a class genuinely absent is refused (E402)
  rather than dropped. The round-trip test missed it because its fixture's labels — `edema`,
  `enhancing` — are already valid keys, so sanitising was a no-op.

- **A DICOM series took its modality LUT, orientation and pixel spacing from slice 0 alone.** A
  per-slice `RescaleSlope` — ordinary in PET — was collapsed to the first slice's, reporting a value
  1740 HU out on every slice it did not apply to; a single rotated slice was placed on the first
  slice's direction matrix. All three are now checked across the stack and refused, naming the
  offending SOPInstanceUID. These refusals carry **no diagnostic code**: §15.2's table describes
  conditions in a MEDH5 file, and a DICOM series is not one yet — no code in it means "these slices
  disagree", so borrowing one would have reported a modality-LUT problem as malformed `channel_names`.
  A slice missing one of these tags outright is refused the same way, rather than surfacing the raw
  `AttributeError` that `scan_dicom` (which does not require the tags) makes reachable — as is a tag
  of the wrong *length*, which extracts perfectly well and so passed the agreement check when every
  slice carried the same wrong length. A five-value `ImageOrientationPatient` then reached
  `np.cross`; a three-value `PixelSpacing` was silently read as its first two elements, giving the
  grid an in-plane size nobody wrote down.

- **A split could put one subject in two partitions.** `group_id` is declared per file and defaults
  to the subject, so two visits curated at different times can disagree about it — the subject then
  becomes two groups, dealt independently, and the same anatomy lands in train and val. `Split.leaks`
  cannot see this: each group really was assigned once, which is all it can know from the assignments.
  `make_splits` now refuses a grouping finer than the subject it is meant to contain (new cohort code
  `C204`), which is where the entries are still in hand. The existing coverage passed because its
  fixture gives every file of a subject the same `group_id`.

### Verified — no change needed

- **§14's storage claims hold as written.** Measured rather than assumed: every image and voxel
  annotation is chunked, stacked encodings use `(1, *spatial)` so one plane reads without the others,
  chunks land inside the 0.5–4 MiB target, and `portable` uses only native `shuffle`+`gzip` while the
  other profiles use Blosc2 as documented. Recompression across `training`/`archive`/`portable` left
  `content_id` and every voxel unchanged with `verify()` passing, and a forked child read correctly
  with the parent uncorrupted.

### Added

- **A property-based sweep over the annotation encoders.** Four findings in this release were one
  shape — a column or tag whose length or rank did not match the elements it described — and each was
  found by inspection, one encoder at a time, with the sibling columns beside it turning out to have
  the same hole. `tests/v1/test_properties.py` enumerates the mismatches instead, holding every
  geometric encoder to one contract (succeed with self-consistent columns, or raise `MEDH5Error` —
  never a bare `IndexError` or `TypeError`), round-tripping masks through `labelmap`, `layers` and
  `bitmask` including the empty, full and overlapping cases, and checking that every transcode pair
  either preserves the masks exactly or refuses. Checked for teeth rather than assumed: run against
  the tree at `6c8da0b` it independently reproduces the `slice_index`, `points.class_ids` and
  `mesh.vertex_class_ids` defects. Against the current tree it finds nothing further. `hypothesis`
  joins the `dev` extra.

- **A corpus smoke test over the whole public read surface.** The conformance corpus checked that
  each case reports its expected diagnostic codes but never called `summary()`, `verify()` or the
  grid/image/annotation/transform accessors on those files. Two contracts now hold across all 103
  cases: a valid case survives the full read surface, and *any* case — including the deliberately
  malformed ones — fails only through `MEDH5Error`, never an `AttributeError` or `KeyError` a caller
  cannot catch by the documented type.

### Fixed — silent loss of ground truth

- **A box on integer edge coordinates lost or gained voxels according to its parity.**
  `box_to_slices` rounded with `np.rint`, which rounds half to **even**, so `lo + 0.5` and
  `hi + 0.5` rounded in opposite directions: a one-voxel box at `[1.0, 2.0]` became an *empty*
  slice and one at `[2.0, 3.0]` became two voxels wide. Boxes built from `slices_to_box` are
  half-integer and never reach the tie, which is why the round-trip tests could not see it; boxes
  from a world→index conversion, an even-factor resample or a pyramid level change are
  integer-valued and reach it constantly. Five call sites depended on it, including
  `curation/tracking.py`, where a lesion's volume is `prod(stop − start) × voxel_volume` — so the
  same lesion measured 0 mm³ or 8× across visits, in the longitudinal measurement the format exists
  for. §8.1 now states `floor(x + 0.5)` explicitly.

- **`labelmap()` deleted overlapping voxels without saying so, on three exits.** `to_nifti(annotation=...)`,
  `VolumeDataset(label_format="labelmap")` and `medh5.monai.to_dict` all flatten on the way out, and
  each silently dropped the overlap region: with a lesion inside a liver, the liver came back 224
  voxels instead of 256. One integer volume cannot hold overlapping classes — which is why `layers`,
  `bitmask` and `probmap` exist (§7.0) — so `labelmap()` now warns, naming the count, unless the
  caller passed an explicit `priority`.

- **`annotated_classes="all"` was byte-identical to `"all_given"`.** `_resolve_annotated` intersected
  the label set back down to `class_ids`, so it could never claim a class that had no mask. Every
  class the annotator searched for and did not find was recorded as *never looked for* instead of
  *verified absent* — the one distinction the coverage contract exists to keep (§11.3) — and no
  validator fired, because W904 only warns when `annotated_class_ids` is a strict subset and this
  made the two equal. `"all"` now also works with `probabilities=`, which needed
  zero-valued planes injected for the absent classes the same way absent masks get an
  empty mask; without them the write failed `E403`, since `annotated_class_ids` named
  more than `class_ids` held.

- **Transcoding destroyed an in-band ignore region.** `labelmap`/`layers` carry ignore in the data;
  `bitmask` and `probmap` express it as a separate `mask` annotation (§7.7), which a
  payload-returning function cannot create. The region was simply dropped, turning "nobody examined
  these voxels" into "verified absent for every annotated class" — what §7.7 names as the most
  common cause of a silently mistrained segmentation model. Transcoding to an encoding that cannot
  hold it is now refused, with the two ways forward in the message. A non-default
  `ignore_id` is carried to the target encoder along with the mask — passing only the
  mask left the header naming a value the data did not contain, which put the region
  right back to reading as background. `LayersAnnotation` gained the `ignore_mask()`
  that `labelmap` already had: it could report `has_ignore_region` while offering no way
  to read the region, so a caller written as `getattr(a, "ignore_mask", None)` — the
  refusal above among them — concluded there was none.

- **Transcoding *to* `instances` merged every object of a class into one.** A dense encoding records
  which voxels belong to a class and never which object, so the conversion collapsed two lesions
  into a single object carrying a freshly minted `instance_id` that belonged to neither — in the
  field §7.4 makes the entire longitudinal join. Refused now, for the same reason
  `instances_from_masks` already refuses to *split* components.

- **The MONAI bridge gave annotations another image's geometry.** `to_dict` took the affine from the
  first requested image rather than from the grid the annotation was bound to. On a sample holding
  CT and PET on different grids — the case this format exists for — the label tensor arrived with
  PET's shape and CT's affine: 70 mm off in z, at half the true spacing, against a docstring
  promising the opposite. It also fabricated an identity affine when no image was requested. The
  affine now comes from the annotation's own grid, and `to_dict` has a test; it had none, and its
  body had never executed even in the dedicated MONAI CI job.

- **`class_ids` and the payload disagreed after transcoding to `probmap`.** `probmap` carries plane
  order only in the §6.2 `class_ids` attribute and its encoder always emits ascending, while
  `transcode_annotation` kept the source header verbatim. A file whose encoding order was not
  ascending — which §6.2 permits, and §7.3 makes explicit for `bitmask` via `bit_class_ids` — read
  back with each class's mask under a different class's name, per-class voxel counts unchanged and
  nothing for the validator to see. Not reachable through this writer, which normalises to
  ascending; reachable through a conforming third-party file.

- **A class examined and found empty vanished on the `instances` decode.** `payload_to_masks` keyed
  off the objects present rather than the declared `class_ids`, and `check_roundtrip` decodes
  through that same path — so the module's own losslessness check could not see the loss.

### Fixed — geometry, registration and sampling

- **An ambiguous frame path resolved silently.** Two registrations between the same pair of
  frames — which §10.1 permits, and which a rigid plus a deformable pair makes ordinary —
  left `transform_between` picking by dict iteration order, i.e. lexicographic transform id.
  Two affines disagreeing by 109 mm resolved to whichever was named first, with the
  validator silent. §10.2 exists because "ambiguity here is the leading cause of silently
  mirrored registration results", so resolution now refuses, names both candidates, and
  points at `sample.transforms` to select one. A single route resolves exactly as before.
  Routes that *converge* before the destination count as two: marking a frame seen on the
  first path to reach it discarded the second, so `A→B→D→T` and `A→C→D→T` arrived as one
  and the tie went undetected.

- **`check_pyramid` never compared a level's `direction` to level 0.** Spacing, origin,
  `coord_system`, `units` and `frame_uid` were checked; orientation was not — and because
  the *expected* origin is derived from the base's direction, a level with permuted axes
  passed both remaining checks. That is precisely the failure §4.3 exists to prevent: a
  model trained at level 2 has its predictions mapped back to level 0 and they land
  transposed. Level extent is now bounded too — one voxel either side of `n / f`, which
  admits both rounding conventions and still rejects a level that is not a resampling of
  its parent.

- **A transform's `units` were never checked against the chain it sits in.** §10.1 makes
  `units` a MUST — "coordinate units, matching the frames' grids" — but only frames were
  compared, so a composite chaining an `mm` leg to a `um` leg validated clean and applied a
  1000× error to half the transform. `check_chain` compares units, and a resolved
  `ChainTransform` refuses to assemble steps that disagree.

- **`strategy="uniform"` was not uniform.** Centres were drawn over every voxel and then
  clamped inward, so every centre in the leading half-patch collapsed onto window start 0 —
  measured at 3.6× the uniform share for the first window and 2.8× for the last, on a
  24-voxel axis with an 8-voxel patch, and worse as the patch grows. Border-heavy training
  data from a strategy named for the opposite. The centre is now drawn from the range that
  maps one-to-one onto valid window starts; clamping stays on the foreground path, where the
  centre is a voxel the caller specifically wants included.

- **`as_slices()` ignored `slice_index`**, so §8.2's canonical "2D box on slice k" — the
  common radiology annotation — selected **no voxels at all**: a degenerate axis converts to
  a zero-thickness slice. The named slice now gets one voxel of thickness. A box with real
  extent on every axis is left alone rather than reinterpreted.

- **The classification accessors contradicted each other on multi-assertion scopes.** §9
  makes several assertions per class ordinary — `scope_ids` is "per assertion", and
  `scope = "timepoint"` means one per visit — but `value()` returned the first matching row
  while `labels` kept the last, so `state()` answered "negative" for a class `positives`
  listed as positive, on one file. `value()` and `state()` take `scope_id=` to select, and
  every collapsing accessor refuses rather than picking when a class carries more than one
  assertion. Single-assertion files are unaffected — including in `summary()`, which keeps
  its flat shape there and reports per scope unit only where a class is asserted more than
  once, so `Sample.summary()` and `medh5 info` keep working on the very files this supports.

### Fixed — de-identification and access control

- **A scrubbed file still contained the DICOM UID it had pseudonymised.** `scrub --apply` runs
  inside the copy-on-write amend, which copies each object and *then* rewrites the attribute; HDF5
  never reclaims what it supersedes, so the released file carried the original
  `FrameOfReferenceUID` in freed space — recoverable with `strings` while every API read returned
  the pseudonym and the file attested `id_mapping: "external"`. A UID links back to the originating
  study in the source PACS. `--apply` now compacts the file before returning; digests and
  `content_id` are unaffected, since this rewrites storage and not content.

- **`id_mapping: "external"` was attested whenever a salt was given**, even when no UID matched and
  the mapping was empty. It is §11.4's strongest claim; it is now recorded only when something was
  actually mapped.

- **Every copy-on-write command widened file permissions.** `amend`, `scrub --apply`, `fix` and
  `recompress` replace the file, and the replacement was created under the process umask — so a
  `0o600` sample came back `0o644` and world-readable, in the commands most likely to be pointed at
  sensitive data on a shared filesystem. The mode of the file being replaced is now carried across.

### Fixed — tools that must not lie or crash

- **`medh5 validate` crashed on roughly a third of corrupted files.** Bytes damaged past the header
  raise out of h5py's traversal or its decompressor; the command exited with a traceback, printed
  nothing on stdout, and `--json` emitted no JSON at all. Failing to read an object is a finding
  about the file, not a crash of the tool: it is reported as `E001` now, per rule, so one unreadable
  object does not hide everything else. 120 randomly corrupted files, up to ten byte flips each, now
  all produce a valid report.

- **`--level strict` promoted warnings in the verdict but not in the counts**, so a report read
  `FAILED … (0 errors, 2 warnings)` and a CI job gating on `errors == 0` passed a file the same
  payload called not-ok. §15.1 defines strict as the other levels with warnings promoted, so at
  strict there are no warnings; each diagnostic keeps its measured `severity`, and the JSON gains a
  `promoted` count.

- **`amend` dropped unknown attributes on the sample root.** Grids, images, annotations and unknown
  groups already kept theirs; the root was the one level that did not, so a 1.0 tool amending a 1.1
  file silently discarded what 1.1 had added — against §16, which permits a minor version to add
  attributes and requires readers to ignore ones they do not recognise.

- **`from_nifti` invented geometry and reported no doubt.** A NIfTI with `sform_code == qform_code
  == 0` states that it carries *no* spatial mapping; nibabel still returns an affine rebuilt from
  `pixdim`, and importing it minted a world grid nobody measured, with the report saying "0
  guesses". It is refused now unless `assume_geometry=True` (CLI: `--assume-geometry`), which
  records it as a guess. An sform and qform that *disagree* — the signature of a file one tool
  updated and another did not — is likewise recorded rather than resolved in silence.

- **The exported payload encoders accepted reserved and out-of-range class ids** and wrapped them
  into the label dtype: `0` became background, `-1` became 255, `65535` became the ignore value,
  `70000` became 4464. The public writer already refused these; the encoders under it, which a
  third-party converter calls directly, now raise `E303` too.

- **`instance_id` was hard-cast to `uint32`**, so an id minted from a 64-bit key wrapped — `2³² + 7`
  became `7`, taking another object's identity. §7.4 permits `uint64`; the width now follows the
  data.

- **`encode_obb` raised a bare `ValueError`** on an empty collection where `boxes`, `mesh` and
  `instances` all raise a coded `E405`. An empty detection annotation is the verified negative the
  coverage contract records, not a degenerate input.

- **`medh5 conformance` with no subcommand** printed its usage to stdout; the other six group
  commands write to stderr.

- **`to_metatensor` emitted a spurious MONAI warning on every call**, passing the affine both
  positionally and inside `meta`. The affine was correct; the warning said otherwise.

### Behaviour changes

Read these before upgrading a pipeline. Each is a correction, and each can change what existing code
produces or accepts:

- **Converter refusals no longer carry a format diagnostic code.** `from_nifti`'s grid-disagreement
  refusals raised `E202` (shape) and `E101` (spacing/origin/direction), and the new DICOM per-slice
  checks initially borrowed `E102`/`E104`/`E204`. §15.2's table describes conditions found *in a
  MEDH5 file*, and a NIfTI volume or DICOM series is not one yet: none of those codes means "these
  inputs disagree", so a caller branching on the code was told an untrue story — a modality-LUT
  problem read as malformed `channel_names`, a grid disagreement as a dangling grid reference. These
  refusals are now uncoded — six sites in total, found one at a time across four review rounds: the
  irregular-stack and coincident-slice refusals in `io/dicom.py` (`E104`, which means *spacing not
  strictly positive* — an irregular stack's median spacing is positive, it is merely nonuniform), the
  two SEG placement refusals in `io/dicom_seg.py` (`E109`, a missing *grid* attribute, where a SEG is
  not a grid), the tilted 2-D plane in `io/nifti.py` (`E102`, a non-orthonormal `direction`, where
  that direction is perfectly orthonormal and simply cannot reduce to a 2×2), and the axis-kind
  disagreement (`E110`, an invalid `axis_kinds` in a file, where nothing here has one). Refusals that
  really do describe the sample being written or targeted keep their codes — a SEG naming a grid the
  sample lacks is genuinely `E101`, a class absent from its label set genuinely `E402` — and a test
  now pins the exhaustive list, so a new coded refusal in `medh5.io` has to be added deliberately.
  **Code branching on `exc.code` for a converter refusal must switch to the exception type**; the
  messages are unchanged and still name what disagreed.

- **A DICOM series that declares no `PixelSpacing` at all is now refused.** One slice omitting it was
  already caught, because it disagreed with the others; *every* slice omitting it meant they all
  agreed on a 1 mm default, and the stack was written with an in-plane size the source never stated.
  Files reaching this path carry `ImagePositionPatient`, so they are cross-sectional images for which
  `PixelSpacing` is mandatory — its absence is a broken series, not one to guess about. A series that
  previously converted with assumed 1 mm spacing will now refuse; that spacing was never the source's.

- Boxes on integer edge coordinates now yield the extent they describe. ROIs derived from
  `box_to_slices` — crops, instance decoding, `as_slices`, tracked lesion volumes — change where
  they were previously off.
- `annotated_classes="all"` now records the whole label set, so files written with it gain
  `class_ids` and `annotated_class_ids` entries (and the zero masks behind them).
- `from_nifti` refuses a NIfTI declaring no spatial mapping; pass `--assume-geometry` to keep the
  old behaviour, now reported as a guess.
- Transcoding refuses two conversions it used to perform silently: to an encoding that cannot hold
  an in-band ignore region, and from a dense encoding to `instances`.
- `labelmap()` warns when it flattens real overlap and no `priority` was given.
- The payload encoders raise `E303` for class ids the public writer already rejected.
- `validate --level strict` reports promoted warnings in `errors` rather than in `warnings`.
- `transform_between` raises rather than choosing when two equally short routes exist; select
  one by id from `sample.transforms`.
- `strategy="uniform"` places windows uniformly, so **the same seed now draws different
  patches**. The change removes a border bias; it does not make previous runs invalid, but it
  does mean a run is not bit-reproducible across this upgrade.
- The classification accessors (`value`, `state`, `labels`, `positives`) raise on a file that
  asserts one class more than once; pass `scope_id=` or read `assertions()`.
- `check_pyramid` — and therefore `E105` — now fires on a level whose `direction` or extent
  disagrees with level 0.

### Changed

- **`medh5 tree` gained `--json`.** It was the only inspection command without a machine-readable
  form, and naming each object with the spec clause that gives it its role is exactly what a cohort
  audit wants.
- **Python 3.13 is tested.** `requires-python` has always admitted it; the matrix and the classifiers
  stopped at 3.12.
- **The specification's executable prototype runs in CI.** Appendix C.2 publishes a table of its
  results, and nothing referenced it — unlike the conformance corpus beside it. It also writes its
  output to the working directory now, rather than next to its own source, so running it does not
  dirty a checkout.
- `medh5.monai.available()` is measured rather than excluded from coverage by a blanket pragma —
  the same shape as the untested-because-skipped problem the MONAI CI job was added to prevent.

### Removed

- `medh5.storage.index.index_attrs()`, a stub returning `{}` that nothing called. It was the
  unfinished half of putting `index/` attributes into `content_id`; §13.2 now states the exclusion
  instead.

### Specification

- **§8.1** pins the box↔slice rounding to `floor(x + 0.5)`. "round" was read as a language default,
  and both Python's `round` and NumPy's `rint` round half to even — under which the extent identity
  in the same clause does not hold.
- **§13.1, §13.2** state normatively that `index/` is excluded from object digests and from
  `content_id`. The reference implementation always skipped it, on the grounds that a derived cache
  should not change the address of the sample it derives from; the text did not say so, so a
  conforming implementation that stamped index digests would compute a different `content_id` for
  the same bytes — and `content_id` is only useful as a cross-implementation key if every
  implementation agrees on what it covers.

### Internal

- Test suite 924 → 969, coverage 93% → 93.5%. Two tests that could not fail were repaired:
  `json.dumps(..., default=str)` coerces anything, so two "is JSON-safe" assertions were vacuous.
  `test_the_format_version_is_not_the_package_version` asserted the package version starts with
  `"1.0"`, tying it to the format version in exactly the way its own docstring forbids; it only
  looked right while the package sat on 1.0.x.

## [1.0.1] — 2026-08-18

Fixes against MEDH5 format 1.0. The **format version is unchanged**: 1.0.1 reads and writes exactly
the files 1.0.0 does, and `__format_version__` stays `"1.0"`.

### Fixed

- **A multi-echo NIfTI is no longer imported as a time series** ([#9]). NIfTI-1 puts time in
  `dim[4]`, but §3.6 gives that axis a `time` row and a `channel` row, and a multi-echo series
  states neither an intent code nor a temporal unit — so it fell through to the guess and arrived
  with per-frame timings the converter had invented. `from_nifti` now reads the BIDS JSON sidecar,
  the same convention `.bval` already covers for DWI.

  Evidence has to be **per volume**: a scalar `EchoTime` is in every MRI sidecar ever written, so
  only a list whose length matches the frame count settles the axis. `EchoTime`, `EchoNumber`,
  `InversionTime` and `FlipAngle` name a channel axis; `VolumeTiming` names a time axis *and*
  supplies measured frame times, which beats a ramp rebuilt from `pixdim[4]` for sparse-sampled
  acquisitions. Echo times are recorded in `acquisition` under the DICOM keyword §4.5 asks for.
  A sidecar stating both kinds at once is refused, and `fourth_axis=` still overrides everything.

- **`medh5.__version__` can no longer drift from the published wheel.** The release workflow
  verified the tag against `pyproject.toml` alone, so a half-completed bump could publish a wheel
  that stamped the wrong version into every file's `generator` and every dataset manifest. Both the
  workflow and the test suite now check the two agree.

### Changed

- **The MONAI tests run in CI** ([#8]). They are guarded by `pytest.importorskip` and MONAI was in
  neither the `dev` extra nor the test matrix, so `to_metatensor(level=...)` shipped covered by a
  test that had never executed. A dedicated job installs the `monai` extra and asserts the import
  before running pytest — without that assertion a failed install leaves the tests skipping and the
  job green.

[#8]: https://github.com/XwK-P/medh5/issues/8
[#9]: https://github.com/XwK-P/medh5/issues/9

## [1.0.0] — MEDH5 format 1.0

A clean-slate reimplementation of the format. **Not backward compatible with 0.x**, by design: a
1.0 reader refuses a 0.x file rather than guessing, and a 0.x reader raises on the missing
`schema_version`. See [the specification](docs/spec/medh5-1.0.md) and
[the implementation plan](https://github.com/XwK-P/medh5/blob/11c4860/design/medh5-1.0-implementation-plan.md)
(a working document, kept in the repository's history).

### The model

A **sample is one subject at one or more timepoints**, each with one or more images. Longitudinal
work — change detection, response assessment, lesion tracking, follow-up registration — lives in one
file, and assigning whole files to train/val/test cannot leak a patient across partitions.

### Added

- `medh5.open` / `create` / `amend` and the `Sample` / `SampleWriter` API.
- **Geometry as a first-class object.** Grids carry the full index→world affine, a frame of
  reference and a timepoint; images and annotations inherit rather than repeat it. Multiscale
  pyramids validate the half-voxel origin shift that silently misplaces predictions when omitted.
- **Label sets** as DAGs with ontology bindings, explicit/implicit closure, canonical digests, and
  three bundled vocabularies (`binary-foreground`, `brats-subregions`, `amos22-organs`).
- **Five voxel encodings** — `labelmap`, `layers`, `bitmask`, `instances`, `probmap` — behind one
  read contract (`contains`, `dense`, `labelmap`, `instances`). The encoding is chosen by measuring
  the class overlap graph, and any pair transcodes losslessly, so it is a storage decision rather
  than a data-model decision.
- **The coverage contract.** `annotated_class_ids` records what the annotator committed to finding,
  so `0` reads as "verified absent" only where that is true. Partially-labelled cohorts become
  safely trainable instead of quietly mistrained.
- **Content addressing.** Per-object SHA-256 digests over decompressed content plus a Merkle
  `content_id`, so verification is incremental, partial and local, and recompression does not
  invalidate anything.
- **A sampling index** that answers foreground patch sampling in O(1) in volume size — 0.52 ms and
  48 KiB against 9.2 ms and O(volume) for the 0.x `argwhere` path.
- **A validator** with four levels and a stable diagnostic-code table, and a **103-case conformance
  corpus** with per-code expectations that a third-party implementation can run. Every code in the
  table has a case.
- **Every geometric annotation** (§8): axis-aligned boxes that convert to numpy slices without
  rounding, oriented boxes stored as rotation *matrices* (dimension-generic, no ordering convention,
  no double cover), keypoints with per-slot classes and visibility, landmark point sets with
  correspondence, planar contours for RTSTRUCT round-trips, and triangle surface meshes. Coordinates
  live in a declared `space` — a grid's continuous index coordinates or a named frame — and readers
  convert through the affine instead of assuming.
- **Classification** (§9), including the three-state semantics that make partial labels safe
  (positive / verified-negative / unknown), ordinal schemes stored verbatim rather than coerced to
  numbers, and **change labels**: an ordinary classification whose `timepoints` names the visits
  compared, so `["tp0","tp2"]` and `["tp1","tp2"]` are distinct assessments.
- **Registration** (§10): identity, affine, dense displacement fields, B-spline free-form
  deformations and composites. One direction convention, `x_M = T(x_F)`, with no attribute to
  reverse it — ambiguity there is the leading cause of silently mirrored results. Displacement
  fields store components on the leading axis so one component or one ROI reads without the rest,
  and report their Jacobian determinant and folding fraction. `transform_between` resolves through
  the **frame graph** rather than by name, uses inverses where they exist, and refuses to invent
  one for a dense field — approximating it would report an accuracy nobody measured. Target
  registration error is computed from paired landmark sets (§10.6).
- **Curation** (§11, §12): a two-node PROV graph of agents and activities that describes the
  dominant real workflow — a model pre-annotation corrected by a human — where a review-status field
  cannot; quality records whose status is current state, with history living in the graph; and
  `agreement` computed from the annotations themselves (per-class Dice/IoU, object-level F1), so a
  number in a file is reproducible from that file. Classes one side never examined are reported as
  *not scored* rather than scored zero, because a class nobody looked at is not a disagreement.
- **Longitudinal tracking joins** (§7.4): `Sample.tracks()` groups objects on `instance_id` across
  visits and reports per-visit volumes and growth. Absence resolves to **`resolved`** only where the
  class is in that visit's `annotated_class_ids`, and to **`unexamined`** otherwise — a growth curve
  that reads "not assessed" as volume zero reports a complete response that never happened. W909 is
  sample-scoped, so it catches the conflict that matters: one lesion classed differently at two
  visits, each annotation internally consistent and only the join wrong.
- **A cross-file split audit** (§12.3). A per-file validator cannot see either failure that matters:
  two files claiming one `set_id` against different manifests (W906), or one subject appearing in
  two partitions — the leakage that inflates every reported metric in medical AI. `medh5 splits`
  reports both, and keeps them separate because the remedies differ.
- **Collections** (§2.2): `.medh5c` shards for cohorts where one file per sample is an operational
  problem. `pack` and `unpack` move stored chunks rather than re-encoding them, so a round trip is
  byte-identical and every `content_id` survives it — a shard is a container for samples, never a
  second encoding of them. A packed sample root *is* a sample root: every reader, validator and
  loader works on it unchanged.
- **Loaders** (plan §2.3). `medh5.sampling` chooses patch windows and visit pairs and depends on no
  deep-learning framework, because where to read is a geometry question. Foreground sampling reads
  the cached coordinate subsample (§14.3) instead of scanning a mask — **0.90 ms and O(1) in volume
  size**, against 9.2 ms and O(volume) for the 0.x `argwhere` path — and a file without an index
  still works but says so on every patch it returns, since a silent 20× slowdown in a dataloader is
  indistinguishable from a slow disk. `medh5.torch` adds `VolumeDataset`, `PatchDataset`,
  `GridPatchDataset`, `PairedPatchDataset`, a collate that keeps ragged detection targets ragged,
  and a **PID-keyed handle cache** (§14.4) — an HDF5 handle inherited across `fork` returns corrupt
  reads that look like data errors, so the cache abandons its contents the moment it notices a new
  process.
- **Paired longitudinal sampling.** `align="transform"` maps the patch centre through the transform
  relating two visits, so both patches cover the same anatomy; `align="none"` returns unregistered
  pairs. A cross-sectional file contributes no pairs and is *counted*, not silently dropped.
- **A MONAI adapter**: `to_metatensor` hands MONAI the correct affine, labelled with its world
  convention rather than converted to RAS behind the caller's back. An ROI shifts the origin, so a
  cropped tensor still lands where its anatomy is.
- **`recompress`** (§14.2). Because digests cover *decompressed* content (§13.1), re-encoding
  changes every stored byte and no `content_id`: a cache keyed on it stays valid, and moving a
  cohort from `training` to `archive` is a storage decision rather than a data migration.
- **Faster multi-class label reads.** `dense()` on `layers` and `bitmask` now groups by **plane
  rather than by class**: a 200-class annotation packed into four layers costs four reads, not two
  hundred. A 64³ multi-class patch read is **4.0 ms** against the 117 ms the 0.x layout needed.
- **`medh5 bench`**, which reproduces every performance target in the plan on the reader's own
  hardware. All are met: sustained patch throughput measures 600–850 patches/s against a target of
  400.
- **Converters** for NIfTI, DICOM, DICOM SEG, RTSTRUCT and nnU-Net v2, plus `migrate` from 0.x.
  Each returns a report of what it *decided* and what it *guessed*, because the interesting part of
  an import is never that it succeeded:
  - **NIfTI**: RAS↔LPS is a sign flip on the affine, never on the voxels, and which convention was
    written is recorded rather than assumed. Volumes that disagree on a grid are refused instead of
    silently resampled.
  - **DICOM**: slices are ordered by their projection on the slice normal, not by `InstanceNumber`
    (a display hint that routinely disagrees with geometry); z spacing is *measured* between slice
    origins rather than taken from `SliceThickness`, which is the slab and not the increment; an
    irregular stack is refused, because it is not a grid. Rescale is stored, not applied (§4.2).
    Only a named list of acquisition tags is copied — §11.4 forbids bulk-copying DICOM into a file
    that claims to be de-identified.
  - **DICOM SEG**: frames are placed by geometry, so a SEG that stores them out of order still
    reads; overlapping segments and `FRACTIONAL` both survive, where flattening to a labelmap would
    drop the tumour inside the organ. Segments match an existing label set by `SegmentLabel`, since
    DICOM segment numbers are positional and carry no identity.
  - **RTSTRUCT**: contours are stored as contours (§8.6) and round-trip exactly. Rasterisation is
    opt-in and its rule is written into the provenance graph, because "does a boundary voxel count"
    is a decision that belongs in the record. A contour enclosed by another on the same slice is a
    hole, not a second region.
  - **nnU-Net v2**: the dataset's own integer ids are kept, so a model's predictions map back with
    no translation table; region labels become classes whose components are their children in the
    §5.1 DAG; `dataset.json` is stashed verbatim, so an export reproduces the dataset rather than
    reconstructing it.
  - **`migrate`**: applies Appendix B and reports each non-mechanical step — the encoding chosen,
    the ids minted (cohort-wide, into a reviewable sidecar), the **half-voxel box shift** from 0.x's
    `[min, max)` integers to voxel edges, and whether a timepoint order was read from dates or
    guessed from mtimes. Instance correspondence across merged files is never inferred.
  - **Grouping**: identity comes from a declared key and never from a filename, a date or an
    accession number. Where it cannot be established the converter falls back to one sample per
    study, names the inputs, and records the fallback.
- **A CLI**: `info`, `tree`, `validate`, `verify`, `timeline`, `track`, `labels`, `seg stats`,
  `seg convert`, `index build`, `pack`, `unpack`, `ls`, `prov`, `agree`, `splits`, `recompress`,
  `bench`, `convert`, `migrate`, `conformance`.

- **Cohort tools** (`medh5.dataset`, `medh5 dataset`). A manifest is a metadata-only scan cached as
  JSON, and it is the *authority* for splits (§12.3): its digest covers membership and grouping —
  which samples, grouped how — and deliberately not content, because writing a claim into a file
  changes the file, and a content-covering digest would make every claim stale the moment it was
  written. Splitting groups before it splits, never by file, and is deterministic given
  `(digest, seed, parameters)`. Groups are dealt by largest deficit against the target ratios
  rather than sliced by index — slicing six groups at 70/15/15 gives train all of them — and where
  an indivisible set of groups genuinely cannot meet the ratios, the split *says which partition got
  nothing* instead of leaving an empty test set to be discovered after the results are written up.
  Strata are interleaved with a rotating lead and dealt against one global tally, so stratifying
  balances the cohort instead of starving val and test. Statistics stream with an exact
  Chan-Golub-LeVeque merge weighted by voxel count, read class counts from the sampling index when
  it is current, and never count an unexamined class as a zero. `dataset check` asks the questions
  no single file can answer — one label set or several, a class id meaning two things, a claim from
  another manifest, a subject in two partitions, a class examined in a tenth of the cohort — under
  its own `C1xx` codes, because a file is not non-conforming because the cohort around it is.
- **`medh5 fix`**, which separates rebuilding a derived cache from restamping a claim. Rebuilding
  an index recomputes a cache from what it caches. Rewriting digests is *not* repair: a mismatch is
  evidence the bytes changed, and recomputing it destroys the evidence. So `--rewrite-digests`
  requires a reason, records an activity naming what it did not verify, and says so on stdout.
- **`medh5 scrub`** (§11.4): a de-identification sweep over the container that attests to exactly
  what it did. UIDs are pseudonymised rather than deleted, since a frame UID is how two files agree
  they share a frame of reference, and the pseudonym is stable so a cohort scrubbed file by file
  still joins. Only a *salted* run records `id_mapping: external` — an unsalted hash is recoverable
  by anyone holding the original UIDs. Dates shift rather than vanish so intervals survive, and
  running it twice does not shift them twice. It reads metadata and not pixels, so it writes
  `burned_in_annotation_checked: false` and lists what it did not check, rather than writing
  "de-identified".
- **A publishable conformance suite.** `medh5 conformance publish` writes a standalone directory —
  103 cases, `expected.json`, the §15.2 code table as data, the JSON Schema, `SHA256SUMS` and a
  README — so an implementer needs nothing installed to be measured. `medh5 conformance score`
  scores any validator, in any language, from `[{file, errors, warnings}, ...]`. `medh5 validate
  --json` emits a superset of that shape, so the reference implementation is scored through exactly
  the same door as everybody else, and a test asserts it.
- **Thirteen documentation pages** written against the 1.0 API, every snippet executed against a
  real sample rather than proofread.

### Changed

- **The 0.x implementation is gone.** 1.0 ships a *reader* for the old layout (~200 lines,
  documenting the format in full) so `medh5 migrate` still works, but not an implementation of it:
  shipping the old package inside the new one would let a curator keep writing the format they are
  migrating away from. The `medh5-0x` console script is removed.
- Eight specification clauses were corrected because implementing them showed the text was not
  implementable, or contradicted itself: `/meta` cannot be compressed, the label-set canonical
  serialization is now defined, `content_id` excludes `created`/`generator`, "the digest of an
  annotation" is defined for a multi-dataset group, the `det` profile requires a *detection-task*
  annotation rather than any §8 kind, §9's `class_ids` dataset and attribute are explicitly
  distinguished, `E010` was added because §2.2 stated a MUST with no code to report it, and W909 is
  stated to be sample-scoped. See Appendix C of the specification.
- `add_segmentation` now keeps every class named in `annotated_classes` expressible, encoding an
  empty one rather than dropping it. Without that, "searched for and not found" collapsed into
  "never looked for" — the distinction §11.3 exists to preserve.

- `writer.split()` now **replaces** a claim for the same `set_id` instead of appending one. Two
  claims for one set is precisely the W906 conflict §12.3 defines, so appending on a re-split
  manufactured the defect the validator exists to catch.
- `set_quality(issues=[...])` accepts constructed `Issue` and `Agreement` records as readily as
  JSON, instead of raising `TypeError: 'Issue' object is not subscriptable`.

### Fixed

- `medh5 dataset check` reported `C201` on the split it had just written, because the manifest
  digest covered `content_id` and `--write-claims` rewrites every file. The digest now covers
  membership and grouping; content drift is a separate question answered by `C401` and `--deep`.
- Stratified splitting put every group in `train` on small cohorts: each stratum was dealt against
  its own tally, and every small stratum rounds that way. Fixed by interleaving strata against one
  global tally, with the lead stratum rotating per round so the small partitions do not fill from
  the same stratum every time.

### Notes

- `content_id` is a Merkle digest over *stored object digests*, so editing a dataset without
  restamping it breaks that object's digest and leaves the root matching. `verify` therefore checks
  every object rather than only the root, and there is a test that says so.
- Reading a case at a level deeper than the conformance manifest declares is **not** safe: 71 of
  the invalid cases are built by editing a valid file, so an integrity pass adds a `content_id`
  mismatch the case never claimed. The published README says so; it originally said the opposite,
  and running it corrected that.

**COCO was dropped from 1.0.** It has no world geometry, no spacing and no frame of reference, so
importing one means inventing a grid and exporting one means discarding the geometry that makes a
medical annotation reproducible. Every other converter here is built on not telling that kind of
silent lie. A 2-D-native path can be added in a minor version — §3.6 already supports 2-D grids.

## [0.6.0]

Hardening pass driven by the napari-medh5 plugin integration report:
clearer single-open diagnostics, process-shared read handles that
obsolete downstream registry workarounds, tri-state checksum
verification so audit UIs can distinguish "no checksum" from "verified
good", and a handful of small ergonomic additions that several
downstream consumers had been re-implementing locally.

### Added

- **`medh5.open_shared(path)`**: ref-counted read-only context manager.
  Multiple callers in the same process (and across threads) share a
  single underlying `h5py.File`; the handle closes only after the last
  caller releases it. Keyed by `Path.resolve()` so symlinks share a
  handle. Replaces hand-rolled handle registries in lazy-read consumers
  (napari plugins, dashboards, viewers).
- **`medh5.VerifyResult`**: `StrEnum` with `OK`, `MISSING`, `MISMATCH`.
  `MEDH5File.verify()` now returns this enum so callers can distinguish
  "no checksum was ever stored" from "checksum verified successfully"
  — the two cases previously both returned `True`, making trustworthy
  audit UIs impossible to build.
- **`medh5.validate_bboxes(bboxes, sample_shape)`**: public clamping
  helper. Returns `(clamped, issues)` where issues is a list of
  `(index, axis, reason)` tuples describing every `"min<0"`,
  `"max>shape"`, or `"min>max"` adjustment applied. Shape mismatches
  raise `MEDH5ValidationError`.
- **`SpatialMeta.as_affine(ndim)`**: compose
  `direction · diag(spacing) + origin` into an `(ndim+1, ndim+1)`
  homogeneous matrix, or return `None` when the rotation is effectively
  identity so consumers can fall back to simpler `scale`+`translate`.
  Obsoletes ~30 lines of hand-rolled affine composition that every
  viewer-style consumer was writing.
- **`on_reopened` callback on `MEDH5File.update` / `update_meta` /
  `add_seg` / `set_review_status`**: fired with `path` only after the
  HDF5 write handle has closed successfully. Lets lazy-read consumers
  re-acquire handles or rebind cached views without reinventing an
  event system.
- **`ValidationIssue.location`**: optional `str | None` field
  (e.g. `"images/CT"`, `"seg/tumor"`, `"bboxes"`,
  `"extra.nnunetv2.labels"`). `_validate_open_file` populates it at
  every error/warning site so downstream UIs can highlight the offending
  dataset without re-parsing `message`. Non-breaking — `to_dict()`
  omits the key when it is `None`.
- **Subsystem `schema_version` stamping**: `set_review_status` stamps
  `extra["review"]["schema_version"] = 1`; the nnU-Net v2 converter
  stamps `extra["nnunetv2"]["schema_version"] = 1`. `read_meta` emits
  a `UserWarning` when a subsystem's stamp is newer than this library
  understands, so consumers can fail loudly on schema drift instead of
  silently mis-rendering.
- **Malformed-`extra` warnings**: `read_meta` validates the shape of
  well-known subsystems (`review.status` must be str;
  `nnunetv2.labels` must be `dict[str, int]`; subsystems must be
  dicts) and emits `UserWarning` on mismatches. The raw payload is
  preserved so consumers can still introspect.
- **Initial "pending" in review history**: `set_review_status` now
  always records the prior state (treating absent as `"pending"`), so
  the audit trail captures the sample's pre-review life from the very
  first call.
- **Clearer single-open diagnostics**: `MEDH5File.update` and
  `set_review_status` detect HDF5's "file is already open" /
  "unable to lock file" errors and raise
  `MEDH5FileError("'{path}' is already open in this process; close
  other MEDH5File handles before …")` with the original as `__cause__`,
  instead of passing the raw h5py message through. Docstrings document
  the exclusive-access requirement and point at `open_shared` for the
  cooperative read side.
- **`"Choosing the right read API"` section** in `docs/python-api.md`:
  table comparing `read()` / `read_meta()` / `MEDH5File(path)` context
  manager so consumers pick the right path the first time.
- **Tests**: expanded to 255 passing (92% coverage) — new
  `tests/test_shared.py`, `tests/test_validate.py`,
  `tests/test_bbox_validation.py`, `tests/test_meta.py`; the existing
  update/review/integrity/io/cli suites now exercise `VerifyResult`,
  `on_reopened`, initial-pending-in-history, and `location` field
  propagation.

### Changed

- **BREAKING**: `MEDH5File.verify(path)` now returns `VerifyResult`
  instead of `bool`. Callers that did `if MEDH5File.verify(p): ...` must
  switch to `if MEDH5File.verify(p) is VerifyResult.OK: ...` (or the
  looser `is not VerifyResult.MISMATCH` for the previous semantics).
  `verify_checksum(f)` in `medh5.integrity` returns the same enum.
  Per the project's pre-1.0 policy in CLAUDE.md, backward compatibility
  is not guaranteed.
- **`set_review_status` returns `ReviewStatus`** instead of `None`, so
  UIs can refresh without re-reading the file. Non-breaking for callers
  that ignored the return value.
- `MEDH5File.__init__` now closes the underlying h5py handle if any
  post-open assignment ever raised (belt-and-braces — impossible in
  practice today, but removes the last bare-open-without-with in the
  module).
- `medh5` CLI `audit` and `recompress --checksum` routes migrated to
  `VerifyResult`: audit still passes on `OK` or `MISSING` (checksums
  remain opt-in); `recompress --checksum` now requires `OK` after
  post-write verification (was "not False", which let `MISSING` slip
  through).

### Fixed

- `set_review_status` history no longer skips the sample's entire
  pre-review life — the first call now records the implicit initial
  `"pending"` state before overwriting it with the user's chosen
  status.
- `MEDH5File.update` no longer leaks the HDF5 write handle on
  post-open exceptions from `__init__` field assignments (defensive;
  no known trigger pre-fix).

## [0.5.0]

First PyPI release. Bundles the 0.4.0 work (never released) with a
dedicated release-hardening pass covering data-safety, PyTorch
multiprocessing, spatial-metadata validation, statistics numerics, CLI
exit codes, packaging, and adds the nnU-Net v2 dataset converter and a
post-review refactor round that split the CLI into a package and
consolidated duplicated helpers.

### Added

- **Atomic writes**: `MEDH5File.write()` now writes to a sibling temp file,
  `fsync`s, and `os.replace`s into place. An interrupted write (Ctrl-C,
  OOM, crash) can no longer leave a truncated `.medh5` file at the
  destination path. Any pre-existing file at the target path is preserved
  on failure.
- **Checksum verification before in-place updates**: `MEDH5File.update()`
  (and by extension `update_meta`, `add_seg`, `set_review_status`) now
  verifies any stored SHA-256 *before* mutating, so an externally
  corrupted file cannot silently have a fresh checksum baked in over top
  of the corruption. New `force=True` escape hatch for intentional
  repairs.
- **Fork/spawn-safe PyTorch handle cache**: `medh5.torch._HandleCache` is
  now PID-scoped — a forked worker observes the PID mismatch and resets
  to a cold cache instead of inheriting parent h5py state. Works
  transparently with `multiprocessing_context="spawn"` (default on macOS
  / Windows / Python 3.14+).
- **`medh5.torch.worker_init_fn`**: the supported `DataLoader(
  worker_init_fn=…)` helper for `num_workers > 0`. Documented in
  README.
- **`PatchSampler(include_bboxes=True)`**: opt-in bbox return from
  `PatchSampler.sample()`. Bboxes are translated into patch-local
  coordinates and filtered to the ones intersecting the patch;
  `bbox_scores` / `bbox_labels` are filtered consistently.
- **`RandomFlip` geometry sync**: flipping now negates the corresponding
  column of `meta.spatial.direction` (via `dataclasses.replace`, so the
  file's cached `SampleMeta` is not mutated) and mirrors any bboxes in
  the sample dict, keeping physical-space metadata consistent with the
  flipped voxel data.
- **`MEDH5File.is_valid(path)`**: thin convenience wrapper returning a
  plain `bool` for the common "is this file OK?" check (swallows
  `MEDH5ValidationError`).
- **Dimension checks in `SampleMeta.validate()`**: `direction` must be
  `ndim × ndim` and `axis_labels` length must equal `ndim`. A malformed
  `direction` attribute on read now raises `MEDH5SchemaError` instead of
  emitting a warning.
- **Numerically-stable parallel stats**: `compute_stats` now accumulates
  per-file `(n, mean, M2)` via Welford and merges with Chan's parallel
  algorithm. Large uint16 CT volumes no longer suffer catastrophic
  cancellation on variance.
- **CLI exit codes**: `medh5 <no args>` and unknown subcommands return
  exit code 2; runtime errors (`MEDH5Error`, `ValueError`, `ImportError`)
  return 1; success returns 0. Replaced the `if cmd == …` ladder with a
  typed dispatch table (`_TOP_HANDLERS`, `_SUB_DISPATCH`).
- **macOS CI job**: `test-macos` on `macos-latest` + Python 3.12
  exercises the `spawn` multiprocessing path that the Linux matrix does
  not cover.
- **Release-build CI job**: runs `python -m build`, `twine check dist/*`,
  inspects the wheel for `medh5/py.typed` + `LICENSE`, and uploads the
  dist/ artifact.
- **PyPI packaging metadata**: authors, project URLs (Homepage,
  Repository, Issues, Changelog), classifiers (Development Status ::
  4 - Beta, Topic :: Scientific/Engineering :: Medical Science Apps.,
  Typing :: Typed), `package-data = {medh5 = ["py.typed"]}`,
  `license = {file = "LICENSE"}`. `LICENSE` file (MIT, Puyang Wang,
  2026) added to the repo root and bundled in both wheel and sdist.
- **Tightened lower bounds**: `h5py >= 3.10`, `hdf5plugin >= 4.1`,
  `numpy >= 1.24`. No upper bounds.
- **nnU-Net v2 dataset converters** (`medh5.io.nnunetv2`): `from_nnunetv2()`
  converts a raw nnU-Net v2 dataset folder (`imagesTr/`, `labelsTr/`,
  optional `imagesTs/`, `dataset.json`) into a directory of per-case
  `.medh5` files, bundling every channel and splitting the integer label
  volume into one boolean mask per foreground class declared in
  `dataset.json`. `to_nnunetv2()` is the reverse: it emits a raw nnU-Net
  v2 layout from a directory of `.medh5` files. The parsed `dataset.json`
  payload is stashed in each file's `extra["nnunetv2"]` so export is
  lossless — channel order, label integer values, and optional fields
  (`overwrite_image_reader_writer`, `regions_class_order`, `name`) all
  round-trip. Region-based (list-valued) labels are rejected with a clear
  error. Requires the `nifti` extra. Lazy-imported from `medh5.io`.
- **CLI nnU-Net v2 subcommands**: `medh5 import nnunetv2 <src> -o <dst>`
  and `medh5 export nnunetv2 <src> -o <dst>` with `--no-test`,
  `--compression`, `--checksum`, `--dataset-name`, and `--file-ending`
  flags.
- **`MEDH5File.is_valid(strict=...)`**: `is_valid()` now forwards a
  `strict` kwarg to `ValidationReport.ok()`, so callers that want the
  one-call "did this file pass cleanly, warnings included?" check can
  get it without building a report object themselves.
- **Deterministic `stats.compute_stats` sampling**: per-file percentile
  sample seeds now derive from a stable BLAKE2b digest of the file path
  instead of Python's hash-randomized `hash()`, so percentile estimates
  are reproducible across runs and across Python invocations.
- **Tests**: expanded to 217 passing (91% coverage), including
  `test_dataloader_workers[spawn]`, `test_patch_dataloader_spawn`,
  `test_interrupted_write_*`, `test_update_verifies_checksum`,
  `test_include_bboxes_*`, `test_randomflip_direction_sync`,
  `test_compute_stats_parallel_matches_serial`, `test_is_valid_*`,
  CLI exit-code tests, end-to-end `medh5 import dicom` CLI coverage,
  `TestFromNnunetv2`/`TestToNnunetv2` happy-path and silent-data-loss
  guards, and a `medh5 import/export nnunetv2` CLI round-trip test.

### Changed

- `MEDH5File.read()` returns `sample.seg = None` when the `seg/` group
  exists but is empty, and `read_meta()` reports `has_seg = False` in
  the same case — previously both could be inconsistent with file
  state.
- Bounding-box datasets are only Blosc2-compressed when `n > 64`;
  tiny bbox arrays are written raw to avoid per-chunk filter overhead.
- **`MEDH5File.validate()` no longer takes `strict`**: strictness is
  applied on the returned `ValidationReport` via `report.ok(strict=...)`,
  keeping the report layer policy-free. The one-call `is_valid()`
  shortcut accepts `strict` as described above.
- **CLI split into `medh5.cli` package**: the 819-line flat
  `medh5/cli.py` is now a package grouped by command —
  `cli/inspect.py` (`info`/`validate`/`validate-all`/`audit`/`recompress`),
  `cli/dataset.py` (`index`/`split`/`stats`), `cli/convert.py`
  (`import`/`export` subgroups), `cli/review.py` (`review set`/`get`/
  `list`/`import-seg`), and `cli/_common.py` for shared helpers. Each
  submodule exposes `register(sub)` and `dispatch(cmd, args) -> int | None`;
  `cli/__init__.py::main()` composes them. Public surface
  (`medh5.cli:main`, `python -m medh5.cli`) is unchanged.
- **`.medh5` suffix helper consolidated**: the duplicate
  `_validate_suffix` / `_SUFFIX` pair in `core.py` and `review.py` was
  hoisted into `medh5.meta` and re-used from both modules.

### Fixed

- `MEDH5File.write()` no longer leaves partial output when interrupted
  mid-write (see "Atomic writes" above).
- `MEDH5File.update()` no longer silently re-hashes corrupted data
  (see "Checksum verification" above).
- `MEDH5PatchDataset` + `DataLoader(num_workers > 0)` no longer
  deadlocks under `fork` or crashes pickling under `spawn`.
- `RandomFlip` no longer silently desynchronizes `meta.spatial.direction`
  from the flipped voxel grid; downstream NIfTI export and
  physical-space metrics now see consistent geometry.
- `compute_stats(workers > 1)` no longer suffers precision loss on
  large integer volumes.
- `medh5 <no args>` now returns exit code 2 instead of 0, unbreaking
  shell automation like `medh5 validate … || exit 1`.
- nnU-Net v2 import no longer silently drops voxels whose integer label
  is not declared in `dataset.json` — `_split_label_volume` raises
  `MEDH5ValidationError` listing the offending values, and rejects
  float label volumes that contain genuinely non-integer voxels while
  still accepting integer-valued floats (`0.0`, `1.0`, …).
- nnU-Net v2 export no longer silently drops seg masks whose names are
  not declared in the nnU-Net label map when merging back to an integer
  label volume; it raises `MEDH5ValidationError` and asks the caller to
  update `extra["nnunetv2"]["labels"]` or remove the extra mask.
- nnU-Net v2 export no longer silently omits per-file image channels
  that disagree with the dataset-wide channel set resolved from the
  first file's metadata; channel mismatches raise
  `MEDH5ValidationError` with a clear missing/extra report.

## [0.4.0]

Bundled into 0.5.0 — never released on PyPI. Entries below describe
work landed under the 0.4 development branch.

### Added

- **Structured validation** (`ValidationReport`, `ValidationIssue`): `MEDH5File.validate()`
  returns a report with typed error/warning codes instead of plain strings.
  Supports `strict` mode where warnings are treated as failures. `ValidationReport`
  is exported from `medh5`.
- **Unified update API** (`MEDH5File.update()`): single entry point for in-place
  metadata, segmentation (add/replace/remove), and bounding-box mutations.
  Automatically resyncs `image_names`, `shape`, `has_seg`, `seg_names`, `has_bbox`
  from file state and recomputes checksums when present.
- **DICOM series selection**: `from_dicom()` now accepts `series_uid` to select
  a specific series when multiple exist. Without it, the largest series is chosen
  deterministically. Available series UIDs are recorded in `extra["dicom"]`.
- **DICOM geometry validation**: strict checks for consistent
  `ImageOrientationPatient`, `PixelSpacing`, and uniform slice spacing across
  the selected series. Multi-frame and non-grayscale DICOM are rejected with
  clear errors.
- **DICOM modality LUT**: `apply_modality_lut` parameter (default `True`) applies
  RescaleSlope/RescaleIntercept before writing via `pydicom.pixels`.
  Disable with `apply_modality_lut=False` or `--no-modality-lut` on the CLI.
- **SimpleITK resampling** for NIfTI imports: `from_nifti(resample_to=...)` resamples
  all images and masks onto a shared reference grid. Supports `"linear"`,
  `"nearest"`, and `"bspline"` interpolators. Masks always use nearest-neighbor.
- **`import_seg_nifti()`** (`medh5.io`): import a NIfTI segmentation mask into
  an existing `.medh5` file with optional resampling and replace semantics.
- **Expanded checksum coverage**: SHA-256 now covers segmentation masks, bounding
  boxes, and critical metadata attributes — not just image datasets. Review status
  updates also recompute the checksum when one is stored.
- **JSON output on CLI**: `--json` flag on `info`, `validate`, `stats`, and
  `review get` commands for machine-readable output.
- **CLI flags**: `--strict` on `validate`, `--fail-fast` on `validate-all`,
  `--resample-to`/`--interpolator` on `import nifti`, `--series-uid`/`--no-modality-lut`
  on `import dicom`, `--resample`/`--replace` on `review import-seg`.
- **Dataset record fields**: `DatasetRecord` now includes `shape`, `spacing`,
  `coord_system`, `patch_size`, and `review_status`.
- **Metadata validation**: `SampleMeta.validate()` now checks `patch_size`
  length and element types.
- **`meta.py` attribute lists**: `_ROOT_META_ATTRS` and `_IMAGE_META_ATTRS`
  tuples canonically define which HDF5 attributes belong to the schema.
  `write_meta()` clears stale attributes before writing.

### Changed

- `MEDH5File.update_meta()` now delegates to `MEDH5File.update()` internally.
- `MEDH5File.add_seg()` now delegates to `MEDH5File.update()` internally.
- `_validate_file()` in `cli.py` replaced by `MEDH5File.validate()`.
- DICOM `_read_series()` returns provenance metadata (selected UID, available
  UIDs, instance count, LUT application status).
- `from_dicom()` now raises on missing `ImageOrientationPatient`,
  `ImagePositionPatient`, or `PixelSpacing` instead of falling back to defaults.
- CLI `main()` wraps all command handlers in a top-level
  `except (ImportError, MEDH5Error, ValueError)` for consistent error reporting.
- **Tests**: expanded from 135 to 167 tests (90% coverage).

### Fixed

- `ValidationPayload` type alias was defined after `if __name__ == "__main____"`
  in `cli.py`, making it unreachable during normal imports. Moved to module top.
- `_build_info_payload()` opened the file twice (once via `MEDH5File` context
  manager, once via `get_review_status()`). Now extracts review status from
  `meta.extra` inline.
- `_validate_open_file()` loaded the entire `bboxes` dataset into memory just
  to check its shape. Now reads only HDF5 dataset metadata.
- Duplicate attribute-name tuples in `integrity.py` (`_HASHED_ROOT_ATTRS`,
  `_HASHED_IMAGE_ATTRS`) now reuse the canonical tuples from `meta.py`.

## [0.3.0]

### Added

- **NIfTI converter** (`medh5.io.nifti`): `from_nifti()` and `to_nifti()` for
  round-trip conversion between NIfTI and `.medh5`. Automatically extracts
  spacing, origin, direction, and coordinate system from the NIfTI affine.
  Requires optional `nibabel` dependency (`pip install medh5[nifti]`).
- **DICOM converter** (`medh5.io.dicom`): `from_dicom()` ingests a DICOM
  series directory into `.medh5`, extracting spatial metadata from standard
  tags and storing selected DICOM attributes under `extra["dicom"]`. Requires
  optional `pydicom` dependency (`pip install medh5[dicom]`).
- **Dataset manifest** (`medh5.dataset`): `Dataset.from_directory()` scans a
  directory tree for `.medh5` files and builds a lightweight manifest (no
  array reads). Supports `filter()`, `save()`/`load()` (JSON), and staleness
  detection via file mtime/size.
- **Dataset splitting** (`medh5.dataset.make_splits`): reproducible
  train/val/test splitting with stratification (`stratify_by`), patient-level
  grouping (`group_by` with dotted-path support into `extra`), and k-fold
  cross-validation.
- **Dataset statistics** (`medh5.stats.compute_stats`): streaming
  per-modality mean, std, min, max, and percentiles (p01/p99) using Welford
  merge across files. Supports foreground-restricted stats via a named
  segmentation mask, label distribution counts, shape histograms, and
  segmentation coverage fractions. Multi-process via `ProcessPoolExecutor`.
- **Patch sampler** (`medh5.sampling.PatchSampler`): lazy, chunk-aligned
  patch extraction with three strategies: `uniform`, `foreground` (biased
  toward a named seg mask), and `balanced` (alternating). Caches foreground
  voxel coordinates per file for efficiency.
- **Pure-numpy transforms** (`medh5.transforms`): `Compose`, `Clip`,
  `Normalize`, `ZScore`, and `RandomFlip`. No torch or PIL dependency.
- **Patch-based PyTorch dataset** (`medh5.torch.MEDH5PatchDataset`): uses
  `PatchSampler` for lazy patch reads instead of full-volume eager loads.
  Configurable `samples_per_volume` for virtual dataset length.
- **Per-worker file handle cache** (`medh5.torch._HandleCache`): LRU cache
  (default 32 handles) shared by both `MEDH5TorchDataset` and
  `MEDH5PatchDataset`. Each DataLoader worker gets its own cache (forked
  process). Eliminates redundant `h5py.File()` opens across epochs.
- **Review/QA workflow**: `MEDH5File.set_review_status()` and
  `MEDH5File.get_review_status()` for tracking annotation review state
  (`pending`/`reviewed`/`flagged`/`rejected`), annotator, timestamp, and
  notes. Prior states are appended to a `history` list. Stored under
  `extra["review"]` (no schema change). `ReviewStatus` dataclass exported
  from `medh5`.
- **Batch CLI commands**:
  - `medh5 validate-all <dir>` — parallel validation of all `.medh5` files.
  - `medh5 audit <dir>` — parallel SHA-256 checksum verification.
  - `medh5 recompress <dir|file> --compression <preset>` — rewrite files with
    a different compression preset. Supports `--out-dir` or atomic in-place
    rewrite via tempfile + rename. Optional `--checksum` flag.
- **Dataset CLI commands**:
  - `medh5 index <dir> -o manifest.json` — build a manifest.
  - `medh5 split <manifest> --ratios 0.7,0.15,0.15 -o splits/` — split with
    optional `--stratify`, `--group`, `--k-folds`, `--seed`.
  - `medh5 stats <dir|manifest> -o stats.json` — compute dataset statistics.
- **Import/export CLI commands**:
  - `medh5 import nifti --image <name> <path> -o out.medh5`
  - `medh5 import dicom <dir> -o out.medh5`
  - `medh5 export nifti <file> -o <dir>`
- **Review CLI commands**:
  - `medh5 review set <file> --status <status> --annotator <name>`
  - `medh5 review get <file>`
  - `medh5 review list <dir> --status <status>`
  - `medh5 review import-seg <file> --name <mask> --from <nifti>`
- **Optional dependency extras** in `pyproject.toml`: `nifti`, `dicom`, `itk`.
- **Tests**: expanded from 62 to 135 tests (91% coverage).

## [0.2.0]

### Breaking Changes

- **Multi-modality images**: The `image` parameter in `MEDH5File.write()` is
  replaced by `images: dict[str, np.ndarray]`.  Each key is a modality name
  (e.g. `"CT"`, `"MRI_T1"`, `"PET"`).  All arrays must share the same shape.
- **On-disk layout**: Image data is stored under an `images/` HDF5 group
  instead of a top-level `image` dataset.
- **`MEDH5Sample.image`** is replaced by `MEDH5Sample.images` (a dict).
- **Schema version** remains `"1"` for the current multi-image layout.
- `SampleMeta` gains `image_names: list[str]`.

### Added

- **Compression presets**: `compression="fast"`, `"balanced"`, or `"max"` as
  a shorthand for `cname`/`clevel` pairs.
- **Context-manager protocol**: `MEDH5File` is now instantiable and supports
  `with MEDH5File("file.medh5") as f:` for typed lazy access via `f.images`,
  `f.seg`, `f.meta`.
- **Custom exceptions**: `MEDH5Error`, `MEDH5ValidationError`,
  `MEDH5FileError`, `MEDH5SchemaError`.
- **Write-time validation**: seg shape vs image shape, bbox count vs
  scores/labels, bboxes shape, clevel range, empty images dict.
- **Schema version checking**: reading a file with a future schema version
  raises `MEDH5SchemaError`.
- **`MEDH5File.update_meta()`**: update label, label_name, or extra metadata
  without rewriting arrays.
- **`MEDH5File.add_seg()`**: add a segmentation mask to an existing file.
- **`MEDH5File.verify()`**: verify SHA-256 checksum of image data.
- **`checksum=True`** parameter on `write()` to store a SHA-256 digest.
- **CLI**: `medh5 info <file>` and `medh5 validate <file>` commands.
- **PyTorch integration**: `MEDH5TorchDataset` in `medh5.torch` (optional
  dependency via `pip install medh5[torch]`).
- **`__repr__`** for `MEDH5Sample` and `SampleMeta`.
- **`py.typed`** marker for downstream type checking.
- **Chunk optimizer**: named `_CHUNK_OVERSHOOT_LIMIT` constant, optional
  L3 cache auto-detection.
- **CI**: GitHub Actions workflow (lint, typecheck, test on Python 3.10-3.12).
- **Tooling**: ruff linting/formatting, pre-commit hooks.
- **Tests**: expanded from 12 to 62 tests with pytest-cov.

### Fixed

- Removed unused `from copy import deepcopy` import in `chunks.py`.
- Malformed `direction` attribute now emits a warning instead of crashing.

## [0.1.0]

Initial release with single-image `.medh5` format, HDF5 + Blosc2 compression,
segmentation masks, bounding boxes, labels, spatial metadata, and chunk
optimization.
