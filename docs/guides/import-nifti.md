# Import from NIfTI and nnU-Net

You have volumes and masks as separate files. This turns them into one sample.
The package reads NIfTI itself, so nothing beyond `pip install medh5` is
needed, and the standalone `medh5` binary converts NIfTI without Python.

## Images and masks as separate files

```bash
medh5 convert from-nifti case.medh5 \
    --image CT=ct.nii.gz \
    --mask liver=liver.nii.gz --mask lesion=lesion.nii.gz
```

```python
from medh5.io.nifti import from_nifti

report = from_nifti({"CT": "ct.nii.gz"}, "case.medh5",
                    masks={"liver": "liver.nii.gz", "lesion": "lesion.nii.gz"})
report.ok
report.of_kind("encoding")     # which encoding was chosen, and why
```

Two conversions happen and both are recorded rather than assumed. NIfTI is
`(x, y, z)` and medh5 is `(z, y, x)`, so the array is transposed and the spacing
and direction permuted to match. NIfTI is RAS+ and medh5 defaults to LPS, so the
affine takes a sign flip — `diag(-1, -1, 1, 1)`, no resampling. Pass
`--coord-system RAS` to stay in RAS.

The round trip `from_nifti` → `to_nifti` reproduces the affine and the voxels.
A file with a scale (`scl_slope`/`scl_inter`) comes back as its physical values
by default, as floats; `to-nifti --stored` writes the stored voxels and the
scale, as the file had them.

**A 2-D file must lie in the x-y world plane at z = 0.** A 2-D grid (spec
§3.6) has two world axes, the first two, and no z: a plane at z = 50 would be
imported at z = 0, on one grid with a plane — or a mask — from any other
slice, so it is refused, and so is a plane not parallel to x-y. Save such an
image as `(x, y, 1)` instead: it imports as a one-slice volume, whose third
axis keeps its position. Or let the import do that:

```bash
medh5 convert from-nifti xr.medh5 --image XR=xr.nii.gz --plane-as-volume
```

`--plane-as-volume` reads every 2-D image and mask as `(x, y, 1)` with the
file's whole affine, so the result is the one you would get from re-saving
the plane yourself. Planes at z = 0 are read the same way, so a cohort whose
headers put some planes at z = 0 and some elsewhere still comes out on one
kind of grid. The affine's third column is the slice's normal and thickness;
a zero column, or one not orthogonal to the plane, is refused, since neither
is invented. Each plane is a `plane_as_volume` decision in the report,
naming its z and normal.

Mask names become class keys as a label set allows them — `--mask GTV-1=...`
is the class `gtv_1`, displayed as `GTV-1` — and two names that make one key
are refused. NIfTI names no subject, so give one with `--subject-id`; without
it the sample is a subject of its own, and the report says so (`identity`). A
file that states no spatial unit is read as millimetres, as a guess (`units`).

## When the files disagree

**Two files with different affines are refused.** They are not the same grid, and
the importer will not resample one onto the other to make the import work.
Resample deliberately, in your own code, and say so.

**A file declaring no geometry is refused.** `sform_code == qform_code == 0` is
NIfTI saying it has voxel indices and no spatial mapping. A reader can still
rebuild an affine from `pixdim`, as nibabel does, and importing that mints a
world grid nobody measured:

```bash
medh5 convert from-nifti case.medh5 --image CT=ct.nii.gz --assume-geometry
```

That takes the fallback on purpose, and the report records it as a guess.

[The reasoning for both](../explanation/refusals.md#nifti).

## 4-D series

NIfTI's fourth dimension does not say what it is. Cine, DCE and 4-D CT are
**time**; multi-b-value DWI and multi-echo series are **channels** (spec §3.6),
and reading a DWI gradient axis as time hands every volume an invented
acquisition time. The importer decides from what the file states — a `.bval`
sidecar, a BIDS sidecar, the header's intent code, a time unit — and records a
**guess** when none of them says:

```bash
medh5 convert from-nifti case.medh5 --image DWI=dwi.nii.gz --fourth-axis channel
```

`--fourth-axis time` or `channel` settles it; the default, `auto`, reads the
file. With a `.bval` sidecar the b-values go into `acquisition`, so a channel
axis still says which volume is which.

## nnU-Net v2 datasets

```bash
medh5 convert from-nnunet /Dataset001_Liver out/
```

Each case's channels and per-class masks become one sample. nnU-Net's class ids
are **kept**, so a model trained against the original dataset still means the
same thing, and region labels become label-set DAG parents — a region that is
the union of two components is a class those components name as a parent. A
label value `labels` does not declare is refused, as nnU-Net refuses it.

Each case becomes a subject of its own: an nnU-Net dataset names cases, not
patients, and the report records that as a guess. Two cases of one patient are
then two subjects, so a split by file is subject-safe only if no patient has
two cases.

The parsed `dataset.json` is stashed in `extra["nnunetv2"]`, so an export later
reproduces the original dataset definition instead of inventing one.

Every channel and label volume must share one grid; a volume some other tool
resampled is refused rather than filed onto channel 0's grid.

## Check the result

```bash
medh5 info case.medh5
medh5 validate case.medh5 --level integrity
```

`--level integrity` runs every check, every byte against its digest included,
and fails on errors. `--level strict` also fails on warnings, so it fails every
freshly imported file: the importer writes no de-identification record
(`W903`), and classes minted from mask names carry no ontology code (`W912`).
Strict is the gate for a dataset you publish, once
[de-identification](deidentify.md) and the label set's codes have dealt with
both.

## Related

- **[Converters](../reference/converters.md#nifti)** — options and Python entry points.
- **[What the converters refuse, and why](../explanation/refusals.md)** — the refusals in full.
- **[Export to other formats](export.md)** — going back the other way.
