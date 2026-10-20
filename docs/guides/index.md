# How-to guides

Answers to specific questions, assuming you already know roughly what a sample
is. If you do not, [the tutorial](../tutorials/first-sample.md) is twenty
minutes.

## Annotating

| | |
|---|---|
| **[Segmentation](segmentation.md)** | Write masks, let the encoding be chosen, read them back for a loss. |
| **[Detection and boxes](detection.md)** | Boxes at voxel edges, cropping without an off-by-one, 2-D on a slice. |
| **[Classification and change labels](classification.md)** | Choosing a scope; labels that span an interval. |
| **[Partial labels and coverage](partial-labels.md)** | Recording what you looked for, not just what you found. |
| **[Record review, provenance and agreement](review-and-provenance.md)** | Who drew a mask and who checked it, its quality, and how far two raters agree. |
| **[Choose and map a label set](label-sets.md)** | A vocabulary for a cohort, the clashes `dataset check` finds, relabelling, and raters who numbered differently. |

## Getting data in

| | |
|---|---|
| **[Import from DICOM](import-dicom.md)** | A directory of studies to one sample per patient, every visit in one file. |
| **[Import from NIfTI and nnU-Net](import-nifti.md)** | Volumes and masks as separate files; nnU-Net v2 datasets. |
| **[Import DICOM SEG and RTSTRUCT](import-seg-rtstruct.md)** | Segments and structure sets onto the series they were drawn on; same-named ROIs; what is refused. |
| **[Migrate from 0.x](migrate-0x.md)** | The one-way door, and the four decisions it cannot make for you. |

## Working across visits

| | |
|---|---|
| **[Longitudinal studies](longitudinal.md)** | Timepoints, tracking one lesion across visits, paired sampling. |
| **[Registration between visits](registration.md)** | Transforms, the frame graph, and what `None` means. |
| **[Resample onto another grid](resample.md)** | Images and their annotations onto a grid or a spacing, recorded; and why the loaders never do it for you. |

## Clinical history (format 1.1)

| | |
|---|---|
| **[Clinical history beside the images](clinical.md)** | Labs, reports and their revisions on one subject clock; what was known at a moment; adding a history to 1.0 files. |
| **[Train on clinical tasks](clinical-training.md)** | Task manifests, preflight, cutoff-aware batches with missing modalities, document features and caches that know when they are stale. |
| **[Import from MEDS or FHIR](clinical-import.md)** | One subject's MEDS rows or FHIR R4 resources as events, with nothing invented: no zone, unit, code or availability. |

## Working with a cohort

| | |
|---|---|
| **[Build and split a cohort](cohorts.md)** | Manifests, leakage-free splits, streaming statistics, cross-file checks. |
| **[Pack small samples into collections](collections.md)** | Many samples in one `.medh5c`: packing, members, splits over members, verifying them. |
| **[Check a file before training on it](validate.md)** | `validate` versus `verify`, choosing a level, reading a code. |
| **[Run the conformance suite](conformance.md)** | Publish the suite, validate every case at its level, score it, and bound the score for an earlier implementation. |

## Getting data out

| | |
|---|---|
| **[Export to other formats](export.md)** | NIfTI, DICOM SEG, RTSTRUCT, nnU-Net — and what each will not do. |
| **[De-identify and publish](deidentify.md)** | Find identifiers, apply the shift, record what was done — and what the tool cannot see. |

## Making it fast

| | |
|---|---|
| **[Tune performance](performance.md)** | The sampling index, chunk sizing, codec profiles, workers — in payoff order. |

Each guide ends by pointing at the reference page for the API it used.
