# Reference

What everything is, rather than how to do anything. For task-shaped answers, see
[the how-to guides](../guides/index.md).

## The package

| | |
|---|---|
| **[Python API](python-api.md)** | Reading, writing, amending. |
| **[API index](api-index.md)** | Every public Python name, its summary and where it is documented. |
| **[Rust crate](rust.md)** | The format engine itself, as a Rust library. |
| **[Command line](cli.md)** | Every `medh5` command. |
| **[PyTorch and MONAI](torch.md)** | Datasets, samplers, collation, the MetaTensor adapter. |

## The data

| | |
|---|---|
| **[Annotation kinds](annotations.md)** | Fourteen kinds: six voxel kinds (five encodings and `mask`), boxes, oriented boxes, keypoints, points, contours, polylines, meshes, classification. |
| **[Converters](converters.md)** | NIfTI, DICOM, DICOM SEG, RTSTRUCT, nnU-Net v2, ITK transforms, MEDS and FHIR R4, and 0.x migration. |
| **[Curation records](curation.md)** | Provenance, quality, agreement, identity, de-identification. |
| **[Storage](storage.md)** | On-disk layout, codec profiles, chunking, integrity, collections. |

## Tables

| | |
|---|---|
| **[Sample document schema](schema.md)** | Every field of `/meta`. |
| **[Clinical schema](clinical-schema.md)** | The `clinical/meta` descriptor and the clinical records. |
| **[Task manifest schema](task-schema.md)** | Every field of a `medh5.task/1` manifest. |
| **[Cache manifest schema](cache-schema.md)** | Every field of a `medh5.cache/1` manifest. |
| **[Diagnostic codes](diagnostic-codes.md)** | All 100 codes `medh5 validate` can report. |
| **[The cohort manifest](cohort-manifest.md)** | Every field of a `medh5.dataset/1` manifest, which `medh5 dataset index` writes. |
| **[Cohort check codes](cohort-checks.md)** | The `C1xx`–`C5xx` codes `medh5 dataset check` reports. |
| **[Task and cache codes](../spec/task-cache-1.md#9-finding-codes)** | The `T1xx`–`T4xx` codes `medh5 task` and `medh5 cache` report. |
| **[Profiles and validation levels](profiles-and-levels.md)** | What to check, and how much. |
| **[Runnable examples](../examples/index.md)** | A complete sample written by following the specification, plus the benchmark scripts. |

The [specification](../spec/medh5-1.0.md) — with [1.1](../spec/medh5-1.1.md)
for the clinical profile, [1.2](../spec/medh5-1.2.md) for open curves, and the
separately versioned
[task and cache contract](../spec/task-cache-1.md) — is the normative
statement of all of it. Where this section and the specification disagree, one
of them is a bug.
