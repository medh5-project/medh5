# Import from DICOM

Turn a directory of studies into one sample per patient, with every visit in the
same file.

```bash
pip install "medh5[dicom]"
```

## 1. See what is there

```bash
medh5 convert from-dicom /studies out/ --group-by subject
```

That is the whole command for the common case. `--group-by subject` is the
default: it resolves patient identity across studies and writes one
multi-timepoint sample per patient, which is what makes a later split
subject-safe. `out/` is a directory, and each sample is written into it named
for its subject — also when the tree holds one patient; without the trailing
separator, a name that is not an existing directory is the file one sample is
written to.

To narrow it first:

```bash
medh5 convert from-dicom /studies out/ --modality CT
medh5 convert from-dicom /studies out/ --series 1.2.840...
```

## 2. Read the report — especially the guesses

```bash
medh5 convert from-dicom /studies out/ --report import.json
```

A conversion report separates what the converter **decided** from the data from
what it **guessed** because the source did not say. A guess is not a failure; it
is the thing to go back and check.

The guess that matters most here is **subject grouping**. When identity cannot
be established — usually because a de-identification pass randomised
`PatientID` — the import falls back to one sample per study, warns, and names
the affected inputs. It never infers identity from filenames, dates or accession
numbers.

The same fallback applies when the source **contradicts** its own key: studies
under one `PatientID` whose `PatientBirthDate` or `PatientSex` disagree are not
one person on the evidence — an anonymiser that writes a constant `ANONYMOUS`
makes this ordinary. Each such study becomes its own sample, and the report
records a guess (`identity`) with the values that disagreed. Only stated values
count: a study that omits a birth date contradicts nothing. The birth date and
sex are compared, never written to the file.

The sample's ids **are** the `PatientID` — usually the medical record number —
or, in the fallback, the `StudyInstanceUID`, and `identity.id_source` says
which. `medh5 scrub` reports them for that reason; see
[De-identify](deidentify.md#2-apply-with-a-salt) for replacing them with
pseudonyms before the files leave your control. A multi-subject import names its
files after the same ids, and `scrub` reports that too. A sample id is ASCII
(§2.3), so a `PatientID` of `Müller^Hans` names sample and file `Muller_Hans`,
and the subject id keeps it whole.

If that fallback happened, you have one file per *study*, not per subject, and
splitting by file is no longer subject-safe.

**`--group-by group_id` does not fix this**, although it looks as though it
should. The fallback gives each output a distinct synthetic subject
(`study:<StudyInstanceUID>`) and sets no `cohort.group_id`, and `group_id`
falls back to `subject_id` when unset — so grouping by it groups by values that
are already distinct, and two visits of one patient can still land on opposite
sides of the split. Silently: nothing warns.

Stamp the grouping you know, then re-scan:

```python
import medh5

for path, patient in known_identities.items():     # from your own records
    with medh5.amend(path) as w:
        w.cohort(group_id=patient)
```

Amending rewrites each file, so this is not free — but it is the only thing that
makes the split subject-safe.

```bash
medh5 dataset index out/ -o cohort.json   # re-scan, so the manifest sees it
medh5 dataset split cohort.json --group-by group_id --write-claims  # C204 if a group splits a subject
medh5 dataset index out/ -o cohort.json   # re-scan, so the manifest sees the claims
medh5 dataset check cohort.json           # C202 if the written claims disagree
```

`C204` comes from **`split`**, not from `check` — it is refused while the split
is being made, which is before anything is written. `check` reads what is on
disk: it catches a subject whose samples claim different partitions (`C202`),
not a grouping that was wrong to begin with.

Neither can recover an identity DICOM has lost. If the mapping you stamped above
was wrong, both commands will agree with it.

## 3. Expect refusals, and read them

The importer would rather stop than write a file whose geometry is invented. A
series is refused when its slices disagree about orientation, pixel spacing,
matrix size or rescale; when a slice is not where a regular stack puts it; when
a required tag is missing or malformed; when its pixels do not decode — a
compressed transfer syntax that no installed decoder reads, say; or when the
stack is **sheared** — the slice origins drift within the slice plane, as a
tilted gantry makes them in head CT. The grid the importer writes is
orthonormal, so a sheared stack would place the far slices centimetres from
where the scanner put them; the refusal names the tilt angle and the worst
slice's offset, and de-tilting is left to a tool that records what it did. The
tolerance is half a pixel, and a passing check is recorded as a
`slice_alignment` decision. The messages name the offending `SOPInstanceUID` or
series.

**A PET series with a rescale per slice is refused by default.** An image
stores one modality LUT, and many PET scanners write a `RescaleSlope` per
slice. The refusal names the option that imports it:

```bash
medh5 convert from-dicom /pet pet.medh5 --per-slice-rescale apply
```

Each slice is then stored as its physical values, `stored × slope +
intercept`, in float32 and with no rescale, so `read()` and
`read(physical=True)` agree; the report's `per_slice_rescale` decision lists
the slopes and the slices that state each. SUV is not computed. A series whose
slices share one rescale is written the same with or without the option, so it
is safe to pass for a whole tree.

**The slice-position tolerance is a physical bound.** The slice spacing is
measured from the slice positions, never read from `SliceThickness`, which is
the slab: it is the median gap between neighbouring slices or, where that does
not place every slice, the least-squares step over all of them. Every slice
must then lie within **a tenth of that spacing, and at least 0.01 mm**, of
where the regular stack puts it along the slice normal — 0.25 mm for 2.5 mm
slices. Positions written to two decimals pass, because rounding does not
accumulate; gaps that each look regular but drift, until the middle of a long
stack is millimetres off, do not. Nothing is resampled to make a stack fit.
The `slice_spacing` decision records the spacing, how it was estimated and the
worst slice's distance (`max_through_plane_error_mm`); a refusal names the
slice and how far off it is, a gap of whole steps as missing slices, and two
slices at one position as duplicates. The same bound holds the planes of a
DICOM SEG that `read_dicom_seg` stacks.

**One refused series does not stop the tree.** It is left out with a warning
(`series`) that names it and why, and the series beside it are imported; a
subject left with none gets no file, and a warning (`subject`) says so. Only
when no series at all can be imported is the import refused. Objects that are
not images — the RT Dose grid beside every radiotherapy export — are skipped
with a note, and multi-frame objects other than enhanced CT, MR and PET with
a warning. A warning fails the command's exit status, so a partial import is
never taken for a complete one.

**Enhanced objects are read frame by frame.** An Enhanced or legacy-converted
CT, MR or PET object holds every slice as a frame, with each frame's position,
orientation, pixel spacing and rescale in its functional groups. Its frames
are split into stacks — by `StackID` and by any dimension of its
`DimensionIndexSequence` that tells stacks apart, such as a temporal
position — and each stack is imported as the classic series it would have
been, under the same checks: placed by its frames' positions, refused when
they do not form a regular grid, and decoded one frame at a time. Two stacks
of one object become two images (`CT_1_tp0`, `CT_2_tp0`). A frame stating no
position is refused, never sorted by guess, and so is a stack whose
`InStackPositionNumber` its positions contradict or that skips a number.
Frames stating different rescales follow `--per-slice-rescale`, as slices
do. A concatenation (one object split across instances) is joined by its
`ConcatenationUID`, and refused when a part is missing. Enhanced ultrasound
volumes are still skipped with a warning.

One thing is assumed rather than refused, and the report records it as a guess
(`slice_spacing`): a single-slice series that states no `SliceThickness` has no
through-plane extent to measure, and is given 1 mm. No voxel moves with it —
a lone slice's centres do not depend on its thickness — but its extent does.

These refusals carry **no diagnostic code**, because a DICOM series is not a
MEDH5 file yet. [The reasoning for each is here](../explanation/refusals.md#dicom).

## 4. Check the result

```bash
medh5 info out/case_0001.medh5
medh5 timeline out/case_0001.medh5     # did the visits land as you expect?
medh5 validate out/*.medh5 --level integrity
```

`--level integrity` runs every check, every byte against its digest included,
and fails on errors. `--level strict` is the same with warnings promoted to
errors, so it fails every freshly imported file: the importer writes no
de-identification record (`W903`, below), and classes an import mints with no
ontology code, as an RTSTRUCT's are from its ROI names, raise `W912`. Strict is
the gate for a dataset you publish, once [de-identification](deidentify.md) and
the label set's codes have dealt with both.

A plane written as a 2-D grid, such as a key image, that names the CT's
`FrameOfReferenceUID` is reported `W917`. §3.3 compares no plane with a volume,
so the shared identifier places the plane nowhere. The identifier is kept, as
the source gave it, and the warning advises: give the plane its own frame, or
store it as a one-slice 3-D grid where it lies, which is how this importer
writes a single-slice series.

Images are named for their modality and visit — `CT_tp0` for the first
visit's CT — and values are stored as the scanner wrote them, with the
modality LUT recorded rather than applied:

```python
ct = next(i for i in s.images.values() if i.modality == "CT")
ct.read()                  # stored counts
ct.read(physical=True)     # HU
```

A physical read is float32 only where a rescale applies: a series the importer
stored without one reads in its stored type. Pass `dtype` for one type across
a cohort.

The report also says that the files are not de-identified: the importer copies
only the tags it names (§11.4), but it writes no de-identification record, and
`medh5 validate` reports a file without one (`W903`). See
[De-identify](deidentify.md).

## Related

- **[Converters](../reference/converters.md#dicom)** — options and Python entry points.
- **[What the converters refuse, and why](../explanation/refusals.md#dicom)** — every DICOM refusal.
- **[Build and split a cohort](cohorts.md)** — the next step for a directory of samples.
