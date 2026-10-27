# Write and read your first sample

Install the package, write a CT with two labelled structures, read it back, and
look at it from the shell. About twenty minutes.

## Install

```bash
pip install medh5
```

That is all this tutorial needs: it brings NumPy, and HDF5 is built in. The
extras for PyTorch, MONAI, DICOM and NIfTI are on the
[install page](../install.md); [the next tutorial](first-training-run.md) adds
PyTorch.

## Write a sample

```python
import numpy as np
import medh5
from medh5 import LabelClass, LabelSet

labels = LabelSet("demo-v1", version="1.0.0", classes=[
    LabelClass(1, "liver", "Liver", category="organ"),
    LabelClass(2, "spleen", "Spleen", category="organ"),
    LabelClass(3, "lesion", "Lesion", parents=[1], category="lesion"),
])

ct = np.random.default_rng(0).integers(-1000, 1500, (64, 96, 96)).astype(np.int16)
liver = np.zeros(ct.shape, bool); liver[10:40, 20:70, 20:70] = True
lesion = np.zeros(ct.shape, bool); lesion[20:26, 35:45, 35:45] = True

with medh5.create("case_0001.medh5", sample_id="case_0001",
                  subject_id="DEMO-0001") as w:
    w.identity(sex="F", bodypart="abdomen")
    w.label_set(labels)
    w.add_timepoint("tp0", label="baseline", days_from_baseline=0)

    w.add_grid("ct", shape=ct.shape, spacing=(2.0, 0.8, 0.8),
               origin=(-64.0, -38.4, -38.4), timepoint="tp0")

    w.add_image("CT", ct, grid="ct", modality="CT",
                value_type="quantitative", value_units="HU")

    w.add_segmentation("organs", grid="ct",
                       masks={"liver": liver, "lesion": lesion},
                       annotated_classes=["liver", "spleen", "lesion"])
```

Four things happened that are worth naming.

**The grid carries the geometry**, and the image and the segmentation reference
it. They cannot drift apart.

**`annotated_classes` names the spleen** even though there is no spleen mask.
That records "we looked and found none", which is a usable negative example.
Leave it out and the default `"all_given"` records only what you handed over;
pass `annotated_classes="all"` to claim the whole label set, which records every
class in it as examined.

**The encoding was chosen by measurement.** Liver and lesion overlap, so
`add_segmentation` measured the overlap graph and picked an encoding that can
represent it. It returns which one:

<!-- illustrative -->
```python
kind, stats = w.add_segmentation(...)   # ("layers", OverlapStats(...))
```

**The file is written atomically.** It appears complete or not at all; a
crashed writer cannot leave a half-file where a valid one used to be.

## Read it back

```python
with medh5.open("case_0001.medh5") as s:
    s.identity.subject_id                  # "DEMO-0001"
    s.profiles                             # {"core", "seg"}

    ct = s.images["CT"].read(physical=True)          # HU
    patch = s.images["CT"].read((slice(10, 20),) * 3)  # just that block

    organs = s.annotations["organs"]
    organs.kind                            # "layers"
    organs.dense(["liver", "lesion"])      # (2, 64, 96, 96) bool
    organs.labelmap(priority=["lesion"])   # (64, 96, 96) of class ids
    organs.voxel_counts()                  # {1: 75000, 2: 0, 3: 600}
```

`dense()` gives each class its own plane, so the overlap survives. `labelmap()`
flattens to one class per voxel, and where liver and lesion meet one of them has
to go: `priority=` names the classes that win, highest first. Without it,
`labelmap()` paints in class order and warns that it flattened 600 overlapping
voxels.

Reads are lazy. `medh5.open` parses the metadata document and nothing else;
slicing an image reads only the chunks that slice touches.

## Look at it from the shell

```
$ medh5 info case_0001.medh5
$ medh5 tree case_0001.medh5
$ medh5 validate case_0001.medh5
$ medh5 verify case_0001.medh5
```

`validate` checks the file against the specification and reports stable
diagnostic codes; `verify` checks that every object still matches its digest.
This file is valid, with two warnings:

```
case_0001.medh5: OK [semantic] profiles=core,seg (0 errors, 2 warnings)
  WARNING W903 /meta#deidentification: no de-identification record; tooling must treat this file as potentially identifying
  WARNING W912 /meta#label_set: 3 class(es) used by annotations have no ontology binding: [1, 2, 3]
```

A warning is legal and worth knowing: nothing records how the data was
de-identified (`w.deidentification(...)` records it), and the label set's
classes carry no ontology code (`codes=` on each `LabelClass`).
`--level strict` promotes warnings to errors, so it fails this file until both
are addressed — it is the gate for a dataset you publish, not for a first
sample.

On a Mac, [medh5 Viewer](../install.md#a-viewer-and-editor-for-the-mac) opens
the same file and draws the liver and the lesion over the CT, and runs the
same validator from its Tools menu.

## Build a sampling index

One more command before the file goes near a training loop:

```
$ medh5 index build case_0001.medh5
```

It stores, per class, a bounded sample of foreground coordinates, so drawing a
patch centred on the liver becomes a lookup instead of a scan of the labels.
Nothing fails without it — the sampler scans instead, and says so.

## Where to go next

- **[Your first training run](first-training-run.md)** — feed this file to a PyTorch `DataLoader`.
- **[The data model](../explanation/data-model.md)** — the model behind the API.
- **[Import from DICOM](../guides/import-dicom.md)** or **[from NIfTI](../guides/import-nifti.md)** — you probably have one of those, not this.
- **[Specification](../spec/medh5-1.0.md)** — when you need the normative answer.
