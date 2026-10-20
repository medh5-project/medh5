# What the converters refuse, and why

Every converter in medh5 would rather fail than guess. This page is the list of
what each one refuses, and the reasoning, because a refusal you do not
understand looks like a bug and a refusal you do understand is usually telling
you something true about your data.

The principle underneath all of it: **an import that silently invents geometry
produces a file that is wrong in a way nothing downstream can detect.** A
crashed import costs an afternoon. A successful import of a fabricated grid
costs a study.

## Most of these refusals carry no diagnostic code

§15.2's [code table](../reference/diagnostic-codes.md) describes conditions found
**in a MEDH5 file**. A DICOM series that disagrees with itself, or two NIfTI
volumes on different grids, are not a MEDH5 file yet — so there is no code for
them, and borrowing one would be worse than having none.

`E202` is an image disagreeing with its grid; neither NIfTI volume is a grid.
`E101` is a reference to a grid that does not exist; nothing here is referenced.
Reporting either would send a reader looking for a defect in a file that was
never written. The converters raise with a message naming the offending input
instead — and a file that is not the format named at all (a text file passed
as DICOM or NIfTI) is refused the same way, naming it.

The exceptions are refusals about a **sample** — what an import into one, or an
export from one, would put in a file — and they carry the code the validator
would report there:

| Code | Refused because |
|---|---|
| `E101` | an import cannot tell which of the sample's grids a SEG or RTSTRUCT belongs to |
| `E401` | the annotation is of a kind the format cannot hold (`to-rtstruct` on a mask, `to-dicom-seg` on boxes or a `mask`) |
| `E402` | a class the sample's label set does not have |
| `E405` | a mask or SEG does not fit its grid: another shape, or frames that are not the grid's slices; a SEG export whose source images are not the annotation grid's slices |
| `E414` | a space or frame of reference the annotation cannot be in: a grid in another known frame, one in pixels or another convention where patient coordinates are needed, or an RTSTRUCT whose ROIs are in several frames |

## No converter writes over what it reads

A `--report`, or an output, naming a file the command reads or writes — through
`..` or a link included — is refused before anything is converted, and so is a
report inside the DICOM tree or nnU-Net dataset the command reads, or the
nnU-Net dataset it writes. `from-dicom-seg --report case.medh5` used to replace
the sample with its own JSON report, and exit 0.

## NIfTI

**Disagreeing grids are refused.** Two NIfTI files with different affines are
not the same grid, and `from_nifti` will not resample one onto the other to make
the import work. Resampling is a decision with a filter, an interpolation order
and a loss of information attached to it; make it deliberately, in your own
code, and say so.

**A 2-D plane off z = 0 is refused.** A 2-D grid (§3.6) lies on the first two
world axes and has no z of its own, so a 2-D NIfTI must lie in the x-y world
plane at z = 0, to 10⁻³ mm. Moved there, a plane at z = 50 shared one grid with
a plane — or a mask — from any other slice. A plane not parallel to x-y is
refused too. Saved as `(x, y, 1)`, the image imports as a one-slice volume that
keeps its position.

**A file that declares no geometry is refused.** `sform_code == qform_code == 0`
is NIfTI stating that it carries no spatial mapping — voxel indices and nothing
else. A reader can still rebuild an affine from `pixdim`, as nibabel does, and
importing that mints a world grid nobody measured. `--assume-geometry` takes the fallback
deliberately, and it is then recorded as a **guess** in the report.

**An sform/qform disagreement is recorded, not resolved in silence.** When both
codes are set and the matrices describe different geometry — the signature of a
file one tool updated and another did not — the sform is used, as is
conventional, and the report carries a guess naming the difference. A reader
that prefers the qform will place that volume somewhere else, and you want to
know which files those are before you train on them. **A file that states no
spatial unit** — NIfTI's `unknown`, the header's default — is read as
millimetres, as NIfTI readers commonly do, and that too is a guess in the report.

**A mask added to a sample must sit on its grid** (`import_seg_nifti`): the
same shape (`E405` otherwise), and the same spacing, origin and direction, with
lengths compared in millimetres. A mask that only matches the shape was drawn on
another grid, and taking its voxels as this one's puts labels where nobody drew
them. `assume_aligned=True` takes them anyway, for a header known to be wrong,
and records the guess.

**Two mask names that make one class key are refused.** A name becomes a key
as a label set allows one — lowercase letters, digits and underscores, so
`GTV-1` is the class `gtv_1`, displayed as `GTV-1` — and `GTV-1` beside
`gtv_1` would merge two masks into one class.

**A one-class export of an annotation with an ignore region is refused**
(`--class`, `to_nifti(class_key=...)`). A 0/1 mask has no value for "nobody
looked here", and its `0` means "examined, absent" (§7.7), so every ignored
voxel would leave as a negative. The whole annotation's labelmap keeps them, as
its ignore id. For the same reason the class must be one the annotation
examined: a class outside `annotated_class_ids` went out as an empty mask, a
verified negative to whoever imported it back, and is refused.

## DICOM

**The stack must agree with itself.** A series is one volume with one geometry
and one modality LUT, so `ImageOrientationPatient`, `PixelSpacing`, the matrix
size (`Rows`, `Columns`) and `RescaleSlope`/`RescaleIntercept` are read once —
but from *every* slice, and checked. A stack whose slices disagree is refused,
naming the offending `SOPInstanceUID`.

Taking the first slice's answer is not a gentler version of this; it is a
different, silent one. A per-slice rescale is ordinary in PET, and collapsing it
to slice 0 reports the wrong activity everywhere that LUT does not apply. One
rotated slice placed on the first slice's direction matrix puts those voxels
where the scanner never did. Neither shows up in the output.

**Irregular slice spacing is refused.** Spacing is measured from the slice
origins, not read from `SliceThickness` — which is the slab, and is regularly
different. Every slice must lie within a tenth of the slice spacing, and at
least 0.01 mm, of where the regular stack puts it: measured per slice against
the whole stack, because drift accumulates and position rounding does not, so
drift that every gap hides is refused and rounding is not. A stack that is not
a regular grid, written as one, silently distorts every distance measured from
it. Each gap used to be held to 1 % of the median gap instead, which passed
4.5 mm of drift and refused positions rounded to two decimals.

**A refused series is left out; the tree is not.** Every series is checked
before its sample is written, and one that is refused — a localizer in three
orientations, a sheared or irregular stack, a transfer syntax no installed
decoder reads — is left out with a warning naming it, which fails the command's
exit status, and the series beside it import. A subject with no series left gets
no file, and only a tree in which nothing can be imported is refused whole. One
such series used to abort the whole import.

**A sheared stack is refused.** The grid is orthonormal: slice *k* sits at the
origin plus *k* steps along the slice normal. A tilted gantry — routine in head
CT — also shifts each slice within its plane, and writing that stack as an
orthonormal grid misplaces the far slices by centimetres (17 mm at the last of
twenty 2.5 mm slices at 20°). The check measures each slice's distance from the
stack's axis against half a pixel, so every voxel the grid places is within half
a voxel of where the scanner put it, and names the tilt angle when it refuses.
It is measured per slice rather than per step, because position rounding in the
files does not accumulate and a tilt does. De-tilting is a resampling, so it is
left to a tool that records it.

**A missing or malformed tag is refused**, rather than defaulted, or left to
surface as a `TypeError` from inside numpy. A series whose slices all omit
`PixelSpacing` used to fall through to 1 mm — which every slice then agreed on,
so the stack was written with an in-plane size the source never stated, and
nothing recorded the assumption.

Cardinality is checked as well as presence, for a reason worth stating: a
`PixelSpacing` of three values reads perfectly well as its first two, and a
stack where *every* slice carries the same wrong length passes an agreement
check that only compares slices to each other.

**Identity is never inferred from filenames, dates or accession numbers.** When
`--group-by subject` cannot establish identity — usually after a
de-identification pass randomised `PatientID` — it falls back to one sample per
study, warns, names the affected inputs and records the fallback. Guessing that
two studies belong to one patient is how a subject ends up in both the training
and the test set.

**Nor is it taken from a key the source contradicts.** Two studies under one
`PatientID` whose birth dates or sexes differ fall back the same way, recorded
as a guess with both values. The opposite error is the quieter one: an
anonymiser that writes a constant `PatientID` merged two patients into one
subject with two visits, which subject-safe splitting then kept together and a
paired model learned "change" between two different people. Splitting is not a
refusal — each study is still a valid sample — and it errs on the side a split
can survive.

**Tags are an explicit allow-list.** Imaging physics — kVp, exposure, kernel,
TR/TE — goes to `acquisition`; everything else stays out (§11.4). A wholesale
tag copy is how a "de-identified" export carries a patient name into a training
set.

**Enhanced frames are placed by their functional groups, or refused.** An
enhanced CT, MR or PET object states each frame's position in its functional
groups, so its frames are held to exactly the checks a classic series'
slices are: ordered by geometry, measured, and refused when they do not form
a grid. `InStackPositionNumber` is checked against the positions rather than
trusted or overruled — a stack whose numbering its geometry contradicts is a
broken object, and re-sorting it by either would be a guess. A number skipped
is a plane missing, and is refused though the positions left are regular.
The same holds of an object's frame count and its parts: an instance whose
`NumberOfFrames` its per-frame groups contradict, or that says it is part of
a concatenation it cannot be joined to, would import a fraction of the
object as all of it.
Objects this importer still does not read — an ultrasound volume, other
multi-frame objects — are skipped and reported as a warning rather than refused, so the
classic series beside them still import; before 1.3.0 such objects were
skipped silently, and before 2.2.0 that included every enhanced CT, MR and
PET object.

## DICOM SEG

**Segments match by label, not by number.** DICOM numbers segments 1..N
positionally. If your label set uses ids 1 and 3, importing by number would
quietly turn "lesion" into whatever class holds id 2. `from_dicom_seg` matches
on `SegmentLabel` and records the mapping as a decision.

**Segments of one class are combined.** A SEG may carry one segment per lesion,
each labelled "Lesion". Each segment used to replace the one before, so every
lesion but the last was lost without a word. A class records where it is, not
which segment drew it, so the segments of one class are combined — their union,
or for `FRACTIONAL` their maximum — and the report records it.

**One label with two modifiers is refused.** Two segments labelled "Kidney",
one modified Left and the other Right, are two structures; combined by their
label, the left and right kidney became one class with one side. Minting
refuses segments of one label whose type modifiers differ, and names both
modifiers, as it refuses segments of one label that code different concepts.

**Frames go only where they were drawn.** Each frame is placed by its own
position, and must be one of the grid's slices, starting at its first voxel,
with the grid's row and column directions and pixel spacing; a SEG drawn on
another reconstruction is refused (`E405`) rather than force-fitted. Frames
turned or spaced against each other are not one volume's slices and are
refused, and so is a frame that states no pixel spacing, rather than read as
1 mm — both without a code, being about the SEG alone.

**Another frame of reference is refused** (`E414`), whether the grid was
matched or named with `--grid`: equal geometry in two frames is no
correspondence, and relating them is a registration the importer does not
apply. A grid in pixels, or in a convention other than LPS or RAS, has no place
for a patient position, and is refused the same way.

**Only `BINARY` and `FRACTIONAL` SEGs are read.** Their frames each hold one
segment and name it. A `LABELMAP` SEG holds every segment in one frame, by
pixel value, and names none; read as binary, every frame became segment 1. It
is refused by name, and so is a frame that names no segment.

**A one-plane SEG read on its own states its thickness, or is refused.** One
plane has no neighbour to measure a step against, so `read_dicom_seg` takes its
`SpacingBetweenSlices`, else its `SliceThickness` (recorded as a decision), and
refuses one that states neither: 1 mm used to be assumed, without a note. A SEG
imported into a sample takes the grid's slices instead.

**Export writes an annotation onto its own grid only.** A SEG's frames take
their geometry from its source images, so the images must be the annotation
grid's slices (`E405` otherwise) in its frame of reference (`E414`; a frame
`medh5 scrub` pseudonymised is compared through `frame_salt=`, `--frame-salt`
on the command line). The export used to compare nothing, and an annotation on
a grid turned in plane, or shifted 10 mm, was written centimetres from where it
was drawn.

**Export writes segments, one per class.** A `labelmap`, `layers`, `bitmask`,
`instances` or `probmap` annotation is exported; any other kind is refused
(`E401`) before anything is read. A geometric annotation has no voxels to
write, and a `mask` no class to name a segment by: the export met the one as
an `AttributeError` and the other as an `IndexError` out of `highdicom`.

**Only the classes the annotation examined are exported.** A segment's 0 says
examined and absent, which a class outside `annotated_class_ids` never was. By
default such a class is left out, recorded as a decision, and naming one is
refused. Every class used to go out, and one nobody looked for became an empty
segment — a verified negative to whoever read it back.

**A FRACTIONAL export takes a `probmap` only.** A fractional SEG
(`segmentation_type="fractional"`) holds probabilities; a `labelmap`, `layers`, `bitmask` or
`instances` annotation holds none, and writing its 0s and 1s as fractions
would claim a certainty nobody measured. It is refused (`E401`) before
anything is read, naming the default binary export, which writes it.

**Export writes the generic concept only where DICOM leaves no choice.** A
segment must carry a property type, a property category and an algorithm
type. Each is taken from the sample where it states one — a code, a category,
the provenance. Where it does not, a more specific concept would be invented,
and a wrong one is undetectable downstream; so the export writes the generic
SCT 91723000 "Anatomical Structure" (or `MANUAL`) and records a guess naming
each class. A code whose system has no DICOM designator is treated as no code:
the designator is never made up.

**Export refuses an ignore region.** A segment has no value for "not
examined", binary or fractional: its 0 says examined and absent. Written from `dense()`, every voxel
the annotation ignores (§7.7) went out as that 0 — unexamined tissue
exported as verified negative — so an annotation with an ignore region is
refused. A NIfTI labelmap keeps the region, as the annotation's ignore id.

**Writing requires `highdicom`** rather than assembling the IOD by hand.
Building per-frame functional groups yourself is how invalid SEGs get published.

## RTSTRUCT

**Contours are patient coordinates**, millimetres in LPS, both ways. A grid in
pixels or another convention relates to none and is refused (`E414`), and so is
a grid in another known frame of reference than the RTSTRUCT's, on import, and
source images in another frame than the contours', on export (`E414`). Stored
as read, contours on an RAS grid or one in metres landed centimetres to metres
from the anatomy.

**Areas import as contours, markers as points; open curves are left out unless asked for.** A
medh5 contour is a closed polygon (§8.6), so an import takes the ROIs drawn as
`CLOSED_PLANAR` contours, and `CLOSEDPLANAR_XOR` ones that are nested or apart
on every slice, whose exclusive-or is then the holes nesting describes. An ROI
of `POINT` contours becomes rows of a `points` annotation (§8.5), and a POINT
contour of more than one point is refused: which point it marks cannot be
told. An ROI of `OPEN_PLANAR` or `OPEN_NONPLANAR` contours is left out with a
warning naming `open_curves="omit"`: filling a line, or recording a class
examined with nothing in it, would invent an area nobody drew.
`open_curves="polylines"` imports it as a `polylines` annotation (format
1.2), on request because that writes the sample at 1.2, which a 1.1 reader
reads without the curves. An ROI
mixing areas with points or lines is refused, since which of the two it is
cannot be told, and so is one mixing exclusive-or contours with plain ones,
which PS3.3 C.8.8.6.1 forbids, and one whose exclusive-or contours cross,
which nesting no longer describes.

**Every contour must lie in a slice of the grid.** One between two slices,
across several (drawn on another orientation, or not planar) or outside the grid
is refused rather than moved onto a slice. Each ROI is in the frame of reference
it states, and ROIs in several frames are refused (`E414`) before the sample is
opened; which frame applied used to depend on the order of the file's frame
list.

**A contour of one ROI inside another's of the same class is refused.** ROIs
whose names make one key are one class, and a class's polygons combine by
nesting depth (§8.6): a contour inside another on its slice is a hole. Two
ROIs named "Bladder", one drawn inside the other, would become a ring — 2.0.2
imported such a pair as 105 voxels where 121 were drawn, its roles nested per
ROI and its rasteriser per class. The refusal names the ROIs and the slice,
and `--roi NUMBER=KEY` imports them as classes of their own. ROIs of one name
that cross or lie apart are combined, with a `merged_rois` note.

**Export refuses a class drawn both ways.** `to-rtstruct --points` writes a
`points` annotation as POINT ROIs beside the contours, and `--curves` a
`polylines` one as open ROIs; a class in two of them is refused, since one ROI
is areas, points or curves, never two, and so is a point or curve of no class
(`E402`): an ROI is a class.

**Export refuses a mask.** `to-rtstruct` on a voxel annotation is an error, not
a marching-squares fallback: the contours it would produce are not the contours
anyone drew, and an RTSTRUCT is a clinical document that asserts they are.

**Rasterisation is opt-in and recorded.** Contours stay contours (§8.6), in
world coordinates. When you do rasterise, the rule used — even-odd fill at voxel
centres, per plane, so a hole is excluded and an island inside it is not — goes
into the provenance graph, because somebody will need to know a year later.

Hole detection groups contours by plane in the grid's **index** space. In world
space the *z* of a contour varies within its own slice under a real oblique
orientation, so grouping there finds no holes at all.

## nnU-Net v2

**Every channel and label volume must share one grid.** A case's channels are
registered to each other by construction — that is what makes them channels — so
a second channel at a different spacing or origin, or a label volume some other
tool resampled, is refused rather than filed onto channel 0's grid (§3.2).

This is the same shared-grid check `from_nifti` applies. Import used to keep
whichever grid it saw first, which wrote the later volumes' voxels intact and
their position silently wrong.

**A label value `dataset.json` does not declare is refused**, as nnU-Net
refuses it, rather than read as background for every class — a verified
negative nobody observed.

**Export refuses a class the sample does not have** (`E402`). `to-nnunet`
matches classes by **id**, not by name — which is precisely why import keeps
nnU-Net's own integers. Matching by name looks equivalent and is not: import
sanitises `dataset.json` names into label-set keys, so a dataset naming a class
`"Tumour Core"` or `"GTV"` resolved nothing — and the export, skipping the miss
silently, came out listing every class, with label files of the right shape in
which every voxel was 0.

**A destination holding other NIfTI files is refused.** nnU-Net trains on every
case file it finds in `imagesTr`, whatever `numTraining` says. An export into a
directory an earlier one wrote left whatever the new one dropped — a removed
case, a dropped channel, a case moved to `imagesTs` and still in `imagesTr`, in
train and test at once — under a `dataset.json` describing none of them. A NIfTI
file there that this export would not write is refused before anything is
written; an identical export rewrites its own files.

**The export is refused as a whole, before a file is written**, when any case
cannot be one of the dataset: a case name that is no file name, or that differs
from another only in case; a case without the annotation (unless
`--unlabeled test`); channels and labels that are not one lattice in one frame,
convention and unit; a class some case never examined; labels that do not come
back from the volume written. These carry no code: the samples are valid, and
what they fail is nnU-Net's layout. The first refused case used to leave the
cases before it written and no `dataset.json`.

## COCO

**Not supported, deliberately.**

COCO is a 2-D polygon/RLE format with no world geometry, no spacing and no frame
of reference. Importing one means inventing a grid; exporting one means
discarding the geometry that makes a medical annotation reproducible. Neither
direction can be done without a silent lie, and every other converter here is
built on not telling one.

A 2-D-native path can be added in a minor version if a concrete need appears —
§3.6 already supports 2-D grids.

## Amending refuses what it cannot attest

Every command that changes a sample amends it: a scrub, a split claim, a
clinical augmentation, `seg`, `medh5.amend`. An amendment refuses, before it
writes anything, a file whose `content_id` no longer recomputes from what it
stores (`E702`) or that holds a digest, or declares a `digest_algo`, it cannot
read (`E703`). Its commit recomputes every digest and the `content_id` from
whatever is stored, so amending such a file would attest an edit nobody
verified, and the only record that the file had changed would be gone.
`medh5 fix --rewrite-digests --reason` is the one way to re-attest it, and it
writes the reason into the file's provenance.

An amendment also refuses to replace a file that changed after it read it:
another writer's commit renamed over it, or an edit in place. Both commits
used to succeed, and the first one's work disappeared. This is detection, not
a lock — see [the design rationale](design-rationale.md#amendments-detect-a-lost-update-rather-than-lock-it-out).

## Related

- **[Converters](../reference/converters.md)** — the commands and their options.
- **[The data model](data-model.md)** — why geometry is stated once and never guessed.
- **[Diagnostic codes](../reference/diagnostic-codes.md)** — the codes that *do* describe a MEDH5 file.
