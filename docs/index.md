# medh5

One medical imaging **sample** — a subject, at one or more timepoints, with
every image, annotation, registration and curation record about them — in a
single self-describing HDF5 file.

```python
import medh5

# A subject with two visits: images, segmentation, registration, tracking.
with medh5.open("subject.medh5") as s:
    s.identity.subject_id                              # "BRATS-GLI-01234"
    s.at("tp1").images["CT_tp1"].read(physical=True)   # HU, not raw counts
    s.annotations["organs"].dense(["liver", "spleen"]) # any encoding, one API
    s.transform_between("tp0", "tp1")                  # resolved via frames
    s.tracks("lesion")                                 # lesions joined across visits
```

```bash
pip install medh5
```

That installs the Python package and the `medh5` command line. Extras for
PyTorch, MONAI, DICOM and NIfTI, the command line without Python, and the Rust
crate are on the [install page](install.md).

One format engine, written in Rust, with three frontends that read and write
the same bytes: the [Python package](reference/python-api.md), the
[Rust crate](reference/rust.md) and the [command line](reference/cli.md).
On a Mac, [medh5 Viewer](install.md#a-viewer-and-editor-for-the-mac) is built
on the same engine: a native app that shows everything a sample holds and edits
it.

## Where to start

**New to it?** [Write and read your first sample](tutorials/first-sample.md) —
twenty minutes, start to finish, then
[a training run](tutorials/first-training-run.md).

**Already have data?** You probably have DICOM or NIfTI, not this.
[Import from DICOM](guides/import-dicom.md) ·
[Import from NIfTI and nnU-Net](guides/import-nifti.md) ·
[Migrate from 0.x](guides/migrate-0x.md)

**Writing your own reader?** The specifications are normative:
[1.0](spec/medh5-1.0.md) for the format, [1.1](spec/medh5-1.1.md) for the
clinical profile and for how a reader opens a later minor version,
[1.2](spec/medh5-1.2.md) for open curves, and the
separately versioned [task and cache contract](spec/task-cache-1.md) for the
companions training builds on. The [conformance suite](spec/conformance.md) is
175 cases you can run against your implementation, and the
[diagnostic codes](reference/diagnostic-codes.md) are the stable contract
between the two.

Otherwise: [how-to guides](guides/index.md) for specific tasks,
[reference](reference/index.md) for what everything is.

## What the format is for

**One file per subject, not per scan.** A sample is a *subject*, and a subject
has visits. Longitudinal work — change detection, response assessment, lesion
tracking, follow-up registration — lives inside one file, which also means
assigning whole files to train and test cannot leak a patient between them.

**Geometry is stated once and never guessed.** Every array is bound to a
declared grid with spacing, origin and direction; a box is at voxel edges and a
voxel index is a voxel centre, both written down. Converting to NIfTI or DICOM
moves numbers between conventions explicitly, and
[refuses when it cannot](explanation/refusals.md).

**Absence is not silence.** `class_ids` says what an annotation can express;
`annotated_class_ids` says what was *looked for*. A class searched for and not
found is recorded as searched for and not found, which is a different training
signal from a class nobody examined. See
[partial labels and coverage](guides/partial-labels.md).

**Every claim is checkable.** Per-object SHA-256 over decompressed content and a
Merkle `content_id` that survives recompression; a validator with a
[stable diagnostic-code table](reference/diagnostic-codes.md); and a 175-case
conformance corpus, at least one case per code, that any implementation can run.

**What was known, when.** From format 1.1 a sample may carry the subject's
clinical history beside the images — labs, reports and their revisions,
diagnoses, medications, assessments — each with when it happened *and* when
it became known. A model trained at a cutoff reads only what was available
then: the preliminary report, not the amendment two days later; the baseline
scan, not the follow-up; a lesion mask only after someone drew it. Tasks and
feature caches are separately versioned companions, pinned to the exact sample
versions they read. See [clinical history](guides/clinical.md) and
[training on clinical tasks](guides/clinical-training.md).

**Reading a patch is fast.** A 64³ multi-class patch reads in ~2.3 ms, against
117 ms measured on 0.x, because chunks are sized for it and the sampling index
makes foreground sampling O(1) in the volume — 0.03 ms per draw at 12.6 Mvox and
at 512³ alike. The index is written by `build_index()` and is not automatic:
without one the same draw scans the labels and costs 210 ms and 2.4 s. See
[tune performance](guides/performance.md), and run `medh5 bench` on your own
hardware.

## The command line

```
$ medh5 info case_0001.medh5             # grids, images, annotations, coverage
$ medh5 validate case_0001.medh5 --level integrity
$ medh5 track case_0001.medh5            # per-lesion volumes across visits
$ medh5 dataset index studies/ -o cohort.json
$ medh5 dataset split cohort.json --group-by group_id --stratify-by site_id
$ medh5 convert from-dicom /studies out/ # one sample per patient, all visits
$ medh5 scrub out/*.medh5 --apply --date-shift-days -117
$ medh5 bench                            # reproduce the performance targets
$ medh5 clinical select case_0001.medh5 --cutoff-hours 24   # what was known then
$ medh5 task preflight progression.task.json                 # every row, in or out, and why
```

Every command is in the [CLI reference](reference/cli.md).

## Going deeper

- **[Runnable examples](examples/index.md)** — a complete two-timepoint sample
  written by following the specification literally, a clinical cohort trained
  on end to end, and the benchmark scripts behind every number quoted here.
- **[Sample document schema](reference/schema.md)** — every field of `/meta`,
  with the machine-readable schema itself.
- **[Design rationale](explanation/design-rationale.md)** — why the format is
  shaped the way it is: what 0.x could not express, the alternatives weighed,
  the costs accepted, why 2.0 is one engine with three frontends, how 1.1 keeps
  what was known when, and what the format deliberately is not.
- **[Changelog](changelog.md)** — every release, with the behaviour changes to
  read before upgrading a pipeline.
- **[Contributing](https://github.com/medh5-project/medh5/blob/main/CONTRIBUTING.md)** —
  the checks a change has to pass, and how the documentation is kept honest.

## Versioning

The **format** is versioned by `medh5_version`: [1.0](spec/medh5-1.0.md);
[1.1](spec/medh5-1.1.md), which adds the optional `clinical` profile and
nothing else; and [1.2](spec/medh5-1.2.md), which adds the `polylines`
annotation kind and nothing else. A sample is written at the lowest version its
content needs, so imaging-only data is still 1.0. A minor version may add optional objects,
profiles, encodings and diagnostic codes; it may not change what an existing
one means, and a reader opens a later minor as a projection — validated for
what it knows, never amended. See spec §16, 1.1 §2 and 1.2 §2.

The **package** follows semantic versioning from 1.0.0. `medh5.__version__` is
the package; `medh5.FORMAT_VERSION` is the newest format it writes.

0.x files are not readable by 1.0 and are not meant to be: `medh5 migrate`
converts them once, reporting every decision it took. See
[Migrate from 0.x](guides/migrate-0x.md).
