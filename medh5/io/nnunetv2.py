"""nnU-Net v2 dataset import and export (spec §5, §7).

An nnU-Net v2 dataset is a directory of NIfTI files plus a ``dataset.json``:

.. code-block:: text

    imagesTr/CASE_0000.nii.gz   channel 0        labelsTr/CASE.nii.gz
    imagesTr/CASE_0001.nii.gz   channel 1        dataset.json
    imagesTs/TEST_0000.nii.gz   a test case      labelsTs/TEST.nii.gz (optional)

Two properties of that layout are worth preserving carefully.

**Label ids are the file's own.**  nnU-Net's ``labels`` maps a name to the
integer written in the label volume, and those integers are meaningful --- a
model trained against them predicts them.  They become MEDH5 class ids
unchanged, so a prediction can be written back without a translation table.
The one id that cannot survive is ``0``: it is nnU-Net's background and MEDH5
reserves it (§5.3), so a class explicitly named for 0 is dropped and reported.

**Regions overlap on purpose.**  A ``labels`` entry whose value is a *list*
(nnU-Net's region-based training) names a union of ids, which is exactly the
overlapping case §7 exists for: the regions are stored as their own classes
alongside the components, and the encoding is chosen by measurement.

**Ignore is not background.**  nnU-Net declares an ``ignore`` label --- the
highest value in ``labels`` --- for voxels nobody examined, which training
and evaluation leave out.  It is read as the annotation's §7.7 ignore region
and written back from it: written as 0 instead, every unexamined voxel became
a verified negative for every class.

``dataset.json`` is stashed verbatim in ``/meta → extra.nnunetv2`` so an export
reproduces the dataset that was imported rather than a reconstruction of it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5._files import write_text
from medh5.errors import MEDH5ValidationError
from medh5.io._common import sanitize_key
from medh5.io.report import ConversionReport

REQUIRED_KEYS = ("channel_names", "labels", "numTraining", "file_ending")
BACKGROUND = 0
IGNORE = "ignore"
"""nnU-Net v2's name for the ignore label, whose value is the highest."""


def read_dataset_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and check an nnU-Net v2 ``dataset.json``."""
    target = Path(os.fspath(path))
    if target.is_dir():
        target = target / "dataset.json"
    if not target.exists():
        raise MEDH5ValidationError(f"dataset.json not found at {target}")
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except ValueError as exc:  # a JSON or UTF-8 error: the input is malformed
        raise MEDH5ValidationError(f"{target} is not JSON: {exc}") from None
    if not isinstance(raw, dict):
        raise MEDH5ValidationError("dataset.json must contain an object")
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise MEDH5ValidationError(f"dataset.json is missing {missing}")
    channels = raw["channel_names"]
    if not isinstance(channels, dict) or not channels:
        raise MEDH5ValidationError(
            "dataset.json `channel_names` must be a non-empty object"
        )
    indices = sorted(int(k) for k in channels)
    if indices != list(range(len(indices))):
        raise MEDH5ValidationError(
            f"channel_names must cover 0..{len(indices) - 1}, got {indices}"
        )
    if not isinstance(raw["labels"], dict) or not raw["labels"]:
        raise MEDH5ValidationError("dataset.json `labels` must be a non-empty object")
    return raw


def _channels(document: Mapping[str, Any]) -> dict[int, str]:
    return {int(k): str(v) for k, v in document["channel_names"].items()}


def _image_ids(channels: Mapping[int, str]) -> dict[int, str]:
    """The image id each channel is stored under: its name, or --- where two
    channels share one --- the name and the channel's index.

    nnU-Net reads a channel name as a normalisation scheme, so two channels
    may well both be ``zscore``.  Keyed by the name alone, the second
    overwrote the first, and the sample held one image of two.
    """
    counts: dict[str, int] = {}
    for name in channels.values():
        counts[name] = counts.get(name, 0) + 1
    ids = {
        index: name if counts[name] == 1 else f"{name}_{index:04d}"
        for index, name in channels.items()
    }
    if len(set(ids.values())) != len(ids):
        raise MEDH5ValidationError(
            f"channel_names {dict(sorted(channels.items()))} name two channels "
            "alike even with their indices; each channel needs an image of its own"
        )
    return ids


def _labels(document: Mapping[str, Any]) -> dict[str, Any]:
    return dict(document["labels"])


def cases(
    root: str | os.PathLike[str],
    document: Mapping[str, Any],
    *,
    include_test: bool = False,
) -> list[str]:
    """Case identifiers present in ``imagesTr`` --- and, with
    *include_test*, in ``imagesTs``."""
    ending = str(document["file_ending"])
    base = Path(os.fspath(root))
    directory = base / "imagesTr"
    if not directory.is_dir():
        raise MEDH5ValidationError(f"{directory} does not exist")
    found = _case_names(directory, ending)
    if include_test:
        found |= _case_names(base / "imagesTs", ending)
    return sorted(found)


def _case_names(directory: Path, ending: str) -> set[str]:
    """The cases whose channel files *directory* holds; none when it is not
    there."""
    found: set[str] = set()
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob(f"*{ending}")):
        stem = path.name[: -len(ending)]
        if "_" in stem and stem.rsplit("_", 1)[1].isdigit():
            found.add(stem.rsplit("_", 1)[0])
    return found


ASSIGNED_BY = "nnU-Net v2 dataset layout"
"""The ``assigned_by`` of the split claim :func:`from_nnunetv2` writes with
``include_test``: the partition is the folder a case was found in."""


def _split_id(source: Path) -> str:
    """The split set an imported dataset's cases claim: ``nnunetv2:<name>``,
    named for the dataset's directory."""
    return f"nnunetv2:{Path(os.path.abspath(source)).name}"


def from_nnunetv2(
    root: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    case_ids: Sequence[str] | None = None,
    coord_system: str = "LPS",
    codec: str = "balanced",
    report: ConversionReport | None = None,
    include_test: bool = False,
) -> ConversionReport:
    """Convert an nnU-Net v2 dataset into one ``.medh5`` per case.

    The training cases are ``imagesTr``'s, labeled from ``labelsTr``.  With
    *include_test*, ``imagesTs``'s test cases are imported too, labeled from
    ``labelsTs`` where it holds their labels, and every case records its
    partition as a §12.3 split claim: set ``nnunetv2:<dataset>``, partition
    ``train`` or ``test``, the dataset layout as its ``assigned_by`` and the
    SHA-256 of ``dataset.json`` as its ``manifest_sha256`` --- so an export
    with ``split=`` puts each case back where it was.  A case in both
    folders is refused.  Without it, the test cases left out are counted in
    a warning.
    """
    import medh5
    from medh5.io.nifti import _same_grid, read_nifti

    log = report or ConversionReport(converter="from-nnunet")
    log.source = os.fspath(root)
    source = Path(os.fspath(root))
    document = read_dataset_json(source)
    channels = _channels(document)
    image_ids = _image_ids(channels)
    restore = _recorded_ids(document)
    label_set, regions, ignore_value = _label_set(document, log)
    label_values = _label_values(label_set, regions)
    declared = _declared_values(document)
    ending = str(document["file_ending"])
    training = set(cases(source, document))
    testing = _case_names(source / "imagesTs", ending)
    if case_ids is not None:
        wanted = list(case_ids)
    else:
        wanted = sorted(training | testing) if include_test else sorted(training)
    # Where each case's files are: a test case is one only in imagesTs.
    test = {c for c in wanted if include_test and c in testing and c not in training}
    if include_test:
        both = sorted(c for c in wanted if c in training and c in testing)
        if both:
            raise MEDH5ValidationError(
                f"case(s) {both} are in both imagesTr and imagesTs: a case is a "
                "training case or a test case, and imported as both its split "
                "claim would say neither; nothing was written"
            )
    elif case_ids is None and testing:
        log.warn(
            "test_cases",
            f"{len(testing)} case(s) in imagesTs were not imported; pass "
            "include_test=True (--include-test) to import them, labeled from "
            "labelsTs where it has their labels",
            {"cases": len(testing), "first": sorted(testing)[0]},
        )
    claim: dict[str, Any] | None = None
    if include_test:
        claim = {
            "set_id": _split_id(source),
            "assigned_by": ASSIGNED_BY,
            "manifest_sha256": hashlib.sha256(
                (source / "dataset.json").read_bytes()
            ).hexdigest(),
        }
        log.decision(
            "splits",
            f"each case claims its folder's partition in split {claim['set_id']!r} "
            "(§12.3): imagesTr's train, imagesTs's test, with dataset.json's "
            "SHA-256 as the manifest; export with split= to write them back",
            {
                "set_id": claim["set_id"],
                "train": len(wanted) - len(test),
                "test": len(test),
            },
        )
    directory = Path(os.fspath(out))
    directory.mkdir(parents=True, exist_ok=True)
    if image_ids != channels:
        log.decision(
            "image_ids",
            "channel_names names more than one channel alike --- nnU-Net reads "
            "the names as normalisation schemes --- so each of those images is "
            "named for its channel index as well",
            {"images": {str(k): v for k, v in sorted(image_ids.items())}},
        )
    log.guess(
        "identity",
        "an nnU-Net dataset names cases, not subjects, so each case was written "
        "as a subject of its own (subject_id = the case id); cases of one "
        "patient are separate subjects here, and a split by file is "
        "subject-safe only if no patient has two cases",
        {"cases": len(wanted)},
    )

    unitless: list[str] = []
    for case in wanted:
        images_folder, labels_folder = (
            ("imagesTs", "labelsTs") if case in test else ("imagesTr", "labelsTr")
        )
        images: dict[str, npt.NDArray[Any]] = {}
        rescales: dict[str, tuple[float, float] | None] = {}
        modalities: dict[str, str] = {}
        geometry: dict[str, Any] | None = None
        for index, name in sorted(channels.items()):
            path = source / images_folder / f"{case}_{index:04d}{ending}"
            if not path.exists():
                raise MEDH5ValidationError(f"case {case!r} has no channel {index}")
            data, geo = read_nifti(path, coord_system=coord_system)
            _replay_geometry(log, f"case {case}: {image_ids[index]}", geo, unitless)
            # Every channel is written onto one grid, so a channel that does not
            # share it must be refused rather than filed under it.  nnU-Net
            # requires co-registered channels and most datasets are, but "the
            # inputs were already correct" is not a check: an unchecked channel
            # at a different spacing lands on the first channel's grid with its
            # voxels intact and its position silently wrong.
            image_id = image_ids[index]
            geometry = _same_grid(geometry, geo, image_id, log) if geometry else geo
            images[image_id] = data
            rescales[image_id] = geo["rescale"]
            modalities[image_id] = _modality(name)
        assert geometry is not None
        label_path = source / labels_folder / f"{case}{ending}"
        masks: dict[int, npt.NDArray[np.bool_]] | None = None
        labelmap: npt.NDArray[Any] | None = None
        marker: int | None = None
        ignored: npt.NDArray[np.bool_] | None = None
        ignored_voxels = 0
        if label_path.exists():
            volume, label_geo = read_nifti(label_path, coord_system=coord_system)
            _replay_geometry(log, f"case {case}: labels", label_geo, unitless)
            # The label volume above all: a label resampled onto a different grid
            # by some other tool is the ordinary way this goes wrong, and the
            # result annotates voxels nobody drew on.
            _same_grid(geometry, label_geo, f"{case} labels", log)
            ids = _label_ids(volume, label_geo["rescale"], case)
            _require_declared(ids, declared, case)
            if ignore_value is not None:
                ignored = np.asarray(ids == ignore_value)
                ignored_voxels = int(np.count_nonzero(ignored))
                if not ignored_voxels:
                    ignored = None
            if label_values is None:
                if ignored is not None:
                    # Out of the values before they are restored: the ignore
                    # label is one above the written values, which can be a
                    # class id once they are restored, and its voxels became
                    # that class.
                    ids = np.where(ignored, BACKGROUND, ids)
                masks = _masks_from(_restored(ids, restore), label_set, regions)
            else:
                labelmap, marker = _marked(
                    ids, ignored, ignore_value, restore, label_values
                )

        target = directory / f"{case}.medh5"
        with medh5.create(
            target, sample_id=case, subject_id=case, codec=codec
        ) as writer:
            tool = writer.software("medh5", medh5.__version__)
            activity = writer.activity(
                "import",
                agent=tool,
                tool="medh5 convert from-nnunet",
                inputs=[f"nnunetv2:{source.name}/{case}"],
            )
            writer.label_set(label_set)
            writer.extra("nnunetv2", dict(document))
            if claim is not None:
                writer.split(
                    set_id=claim["set_id"],
                    partition="test" if case in test else "train",
                    assigned_by=claim["assigned_by"],
                    manifest_sha256=claim["manifest_sha256"],
                )
            writer.add_grid(
                "ref",
                shape=geometry["shape"],
                spacing=geometry["spacing"],
                origin=geometry["origin"],
                direction=geometry["direction"],
                coord_system=geometry["coord_system"],
                units=geometry["units"],
                timepoint="tp0",
            )
            for name, array in images.items():
                # The header's scl_slope/scl_inter, kept as the image's §4.2
                # rescale: dropping them left a CT with intercept -1024 at the
                # stored counts, 1024 HU off what nnU-Net itself reads.
                rescale = rescales[name]
                if rescale is not None:
                    log.decision(
                        "value_scale",
                        f"case {case}: {name}'s scl_slope/scl_inter were stored as "
                        "its rescale; read(physical=True) applies them (§4.2)",
                        {"case": case, "slope": rescale[0], "intercept": rescale[1]},
                    )
                writer.add_image(
                    name,
                    array,
                    grid="ref",
                    modality=modalities[name],
                    rescale_slope=None if rescale is None else rescale[0],
                    rescale_intercept=None if rescale is None else rescale[1],
                    prov=activity,
                )
            if masks or labelmap is not None:
                # The label volume as it is, each value the classes it is in:
                # one mask per class held the volume once per class.
                if labelmap is not None:
                    kind, _ = writer.add_segmentation(
                        "seg",
                        grid="ref",
                        labelmap=labelmap,
                        label_values=label_values,
                        ignore_value=marker,
                        annotated_classes="all",
                        prov=activity,
                    )
                else:
                    kind, _ = writer.add_segmentation(
                        "seg",
                        grid="ref",
                        masks=masks,
                        annotated_classes="all",
                        ignore=ignored,
                        prov=activity,
                    )
                if ignored is not None:
                    log.decision(
                        "ignore",
                        f"case {case}: {ignored_voxels} voxel(s) of nnU-Net's "
                        f"ignore label {ignore_value} became the annotation's "
                        "ignore region (§7.7), unexamined rather than background",
                        {"case": case, "voxels": ignored_voxels},
                    )
                log.decision(
                    "encoding",
                    f"case {case}: labels were stored as {kind!r}",
                    {"case": case, "kind": kind},
                )
        log.outputs.append(str(target))
    if unitless:
        log.guess(
            "units",
            f"{len(unitless)} file(s) state no spatial unit, so their lengths "
            "were read as millimetres, as NIfTI readers commonly do",
            {"files": len(unitless), "first": unitless[0]},
        )
    log.decision(
        "coverage",
        "annotated_class_ids covers the whole label set: an nnU-Net label volume "
        "is exhaustive by construction, so a class absent from it is verified "
        "absent rather than unexamined (§11.3)",
        {"classes": len(label_set)},
    )
    return log


def _replay_geometry(
    log: ConversionReport, name: str, geometry: Mapping[str, Any], unitless: list[str]
) -> None:
    """What the NIfTI reader had to assume, into the report: each file's
    geometry guesses as they are, and a file stating no spatial unit
    counted, for one note over the dataset rather than one per file."""
    for kind, message, detail in geometry.get("notes", ()):
        if kind == "units":
            unitless.append(name)
        else:
            log.guess(kind, f"{name}: {message}", detail)


def _recorded_ids(document: Mapping[str, Any]) -> dict[int, int]:
    """The class id each renumbered label value stands for, as an export
    records it (``medh5_class_ids``)."""
    recorded = document.get("medh5_class_ids") or {}
    if not isinstance(recorded, Mapping):
        raise MEDH5ValidationError(
            "dataset.json `medh5_class_ids` must map label values to class ids"
        )
    out = {int(value): int(class_id) for value, class_id in recorded.items()}
    if len(set(out.values())) != len(out):
        raise MEDH5ValidationError(
            "dataset.json `medh5_class_ids` gives two label values one class id"
        )
    return out


def _declared_values(document: Mapping[str, Any]) -> npt.NDArray[np.int64]:
    """Every value a label volume of the dataset may hold: background, each
    label's value or its region's components, and the ignore label's."""
    values = {BACKGROUND}
    for value in _labels(document).values():
        values.update(int(v) for v in (value if isinstance(value, list) else [value]))
    return np.asarray(sorted(values), dtype=np.int64)


def _require_declared(
    ids: npt.NDArray[Any], declared: npt.NDArray[np.int64], case: str
) -> None:
    """Refuse a label volume holding a value ``labels`` does not declare.

    nnU-Net's own dataset check refuses such a case.  Read as it was, the
    value matched no class, and its voxels became verified negatives for
    every one.
    """
    known = np.isin(ids, declared)
    if not known.all():
        stray = sorted(float(v) for v in np.unique(np.asarray(ids)[~known]))
        raise MEDH5ValidationError(
            f"case {case!r}: the label volume holds "
            f"{[int(v) if v.is_integer() else v for v in stray]}, which "
            f"dataset.json's labels do not declare ({declared.tolist()}); "
            "nnU-Net refuses such a case, and read anyway those voxels would be "
            "verified negatives for every class"
        )


def _restored(ids: npt.NDArray[Any], restore: Mapping[int, int]) -> npt.NDArray[Any]:
    """*ids* with each renumbered value back at its class id."""
    if not restore:
        return ids
    out = np.array(ids, dtype=np.int64, copy=True)
    for value, class_id in restore.items():
        out[ids == value] = class_id
    return out


def _modality(name: str) -> str:
    """nnU-Net channel names are free text; map the common ones, else ``OT``."""
    known = {
        "ct": "CT",
        "t1": "MR",
        "t1ce": "MR",
        "t2": "MR",
        "flair": "MR",
        "pet": "PT",
    }
    return known.get(name.strip().lower(), "OT")


def _label_set(
    document: Mapping[str, Any], log: ConversionReport
) -> tuple[Any, dict[int, list[int]], int | None]:
    """nnU-Net ``labels`` as a MEDH5 label set, keeping its integer ids, and
    the value of its ignore label, which is no class."""
    from medh5.labels import LabelClass, LabelSet

    scalars: dict[str, int] = {}
    region_values: dict[str, list[int]] = {}
    dropped: list[str] = []
    ignore_value: int | None = None
    restore = _recorded_ids(document)
    for name, value in _labels(document).items():
        if name == IGNORE and not isinstance(value, list):
            ignore_value = int(value)
        elif isinstance(value, list):
            region_values[name] = [restore.get(int(v), int(v)) for v in value]
        elif int(value) == BACKGROUND:
            dropped.append(name)
        else:
            scalars[name] = restore.get(int(value), int(value))
    if restore:
        log.decision(
            "label_ids",
            "dataset.json records the class id each renumbered label value stands "
            "for (medh5_class_ids); the class ids were restored",
            {"restored": {str(v): c for v, c in sorted(restore.items())}},
        )

    next_id = max([*scalars.values(), 0]) + 1
    regions: dict[int, list[int]] = {}
    region_ids: dict[str, int] = {}
    for name, components in region_values.items():
        region_ids[name] = next_id
        regions[next_id] = components
        next_id += 1

    # A region is a union of its components, which is exactly a parent/child
    # relation in the §5.1 DAG --- so it is stored as one, rather than as an
    # opaque list only this converter understands.
    parents: dict[int, list[int]] = {}
    for region_id, components in regions.items():
        for component in components:
            parents.setdefault(component, []).append(region_id)

    classes: list[LabelClass] = [
        LabelClass(
            class_id,
            _key(name),
            name,
            parents=tuple(sorted(parents.get(class_id, ()))),
        )
        for name, class_id in sorted(scalars.items(), key=lambda kv: kv[1])
    ]
    classes.extend(
        LabelClass(
            region_ids[name],
            _key(name),
            name,
            properties={"nnunet_region": components},
        )
        for name, components in sorted(region_values.items())
    )
    if dropped:
        log.decision(
            "background",
            f"label(s) {dropped} map to nnU-Net's background 0, which MEDH5 "
            "reserves (§5.3); they were dropped rather than renumbered, so every "
            "other id still matches the label volume",
            {"dropped": dropped},
        )
    if regions:
        log.decision(
            "regions",
            f"{len(regions)} region label(s) name a union of ids; each became its "
            "own class, with its components recorded as children in the label-set "
            "DAG, which is the overlap §7 handles",
            {"regions": {str(k): v for k, v in regions.items()}},
        )
    log.decision(
        "label_ids",
        "nnU-Net's own integer ids were kept, so a model's predictions map back "
        "without a translation table",
        {"ids": {c.key: c.id for c in classes}},
    )
    return LabelSet("nnunetv2", version="1.0.0", classes=classes), regions, ignore_value


def _label_ids(
    volume: npt.NDArray[Any], rescale: tuple[float, float] | None, case: str
) -> npt.NDArray[Any]:
    """The label volume's class ids as every reader of the file sees them.

    nnU-Net loads labels through nibabel or SimpleITK, which apply the header's
    ``scl_slope``/``scl_inter``, so a file storing 1 with slope 32 is class 32
    to the model; matching the stored integers imported it as class 1, or as
    nothing.  A scaling that leaves a voxel between two ids is no label volume,
    and is refused.
    """
    if rescale is None:
        return np.asarray(volume)
    slope, intercept = rescale
    physical = np.asarray(volume, dtype=np.float64) * slope + intercept
    ids = np.round(physical)
    if not np.allclose(physical, ids, atol=1e-6):
        raise MEDH5ValidationError(
            f"case {case!r}: the label volume's scl_slope/scl_inter ({slope}, "
            f"{intercept}) make labels that are not integers; a label volume "
            "holds class ids"
        )
    return ids.astype(np.int64)


def _label_values(
    label_set: Any, regions: Mapping[int, Sequence[int]]
) -> dict[int, list[int]] | None:
    """Each label value's classes: its own, and every region holding it.

    ``None`` when the label set has no class, or a region holds background
    0, which is no value of a labelmap: those cases are written as masks.
    """
    values: dict[int, list[int]] = {}
    for entry in label_set:
        if entry.id not in regions:
            values.setdefault(entry.id, []).append(entry.id)
    for region, components in regions.items():
        for component in components:
            if component == BACKGROUND:
                return None
            values.setdefault(int(component), []).append(region)
    return {value: sorted(set(ids)) for value, ids in values.items()} or None


def _marked(
    ids: npt.NDArray[Any],
    ignored: npt.NDArray[np.bool_] | None,
    ignore_value: int | None,
    restore: Mapping[int, int],
    label_values: Mapping[int, Sequence[int]],
) -> tuple[npt.NDArray[Any], int | None]:
    """The label volume restored to its class ids, and the value that marks
    its ignore region there (``None`` when nothing is ignored).

    The ignore label is one above the written values, so a class id once they
    are restored can be the ignore label: restored, its voxels are marked -1,
    which no label value is.  A volume of floats holding declared integers
    (``_require_declared`` has checked them) is cast to integers, which a
    labelmap is.
    """
    if ids.dtype.kind not in "iu":
        ids = ids.astype(np.int64)
    if ignored is None:
        return _restored(ids, restore), None
    if not restore and ignore_value not in label_values:
        return ids, ignore_value
    # Restored, the volume is already a copy of its own.
    out = _restored(ids, restore) if restore else np.array(ids, dtype=np.int64)
    out[ignored] = -1
    return out, -1


def _masks_from(
    volume: npt.NDArray[Any], label_set: Any, regions: Mapping[int, Sequence[int]]
) -> dict[int, npt.NDArray[np.bool_]]:
    """One boolean mask per class, regions unioned from their components."""
    values = np.asarray(volume)
    masks: dict[int, npt.NDArray[np.bool_]] = {}
    for entry in label_set:
        if entry.id in regions:
            union = np.zeros(values.shape, dtype=bool)
            for component in regions[entry.id]:
                union |= values == component
            masks[entry.id] = union
            continue
        masks[entry.id] = values == entry.id
    return masks


def _key(name: str) -> str:
    return sanitize_key(name)


def to_nnunetv2(
    paths: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    dataset_name: str = "Dataset001_medh5",
    file_ending: str = ".nii.gz",
    annotation: str = "seg",
    classes: Sequence[int | str] | None = None,
    unlabeled: str = "refuse",
    report: ConversionReport | None = None,
    split: str | None = None,
    replace: bool = False,
) -> ConversionReport:
    """Export samples as an nnU-Net v2 dataset.

    When the samples were imported from nnU-Net the stashed ``dataset.json`` is
    reused, so the export reproduces the original dataset rather than a
    reconstruction of it: its channel names and every field this exporter does
    not write are kept as stashed, its labels are kept except where a case
    needs them extended or nnU-Net needs them renumbered (both reported), and
    ``numTraining`` and ``file_ending`` describe the files written.  Without a
    stash, a channel is named for its image, and a CT image ``CT``: nnU-Net
    reads the name as the channel's normalisation, and any name but ``CT``
    z-scores it.

    Everything is checked before anything is written:

    * *file_ending* is ``.nii.gz`` or ``.nii``: every file written is NIfTI;
    * every case's name is a **file name under the dataset**: its
      ``sample_id`` is a sample key (``[A-Za-z0-9_.-]``), not ``.`` or ``..``,
      and no two differ only in case --- an id was a path, and
      ``../../victim`` or an absolute one wrote outside the export;
    * every case is **labeled**: a case without the annotation is refused, or
      with ``unlabeled="test"`` written to ``imagesTs`` and not counted in
      ``numTraining`` --- unless *split* places it in training, which it
      keeps: such a case is refused either way;
    * with *split*, every case is **placed by its claim** in that split set
      (§12.3): a case claiming ``test`` is written to ``imagesTs``, its labels
      to ``labelsTs``, and one claiming ``train`` or ``val`` to ``imagesTr``
      (nnU-Net cross-validates its training cases itself); a case the set does
      not place, places twice, or holds out (``holdout``, ``unassigned``) is
      refused --- a mistyped set id would otherwise train on every test case;
    * the dataset directory holds **no NIfTI file the export would not
      write** in ``imagesTr``, ``labelsTr``, ``imagesTs`` or ``labelsTs`` ---
      an earlier export's dropped case, channel, partition or file ending,
      which nnU-Net would train on --- so an identical re-export rewrites its
      own files and any other is refused (:func:`_require_no_strays`).  With
      *replace*, the directory must be an existing dataset instead --- a
      directory, not a link, holding a ``dataset.json`` --- and the new
      dataset is written over it in place, file by file: NIfTI files of
      those four folders that the export does not write are removed, every
      other file is left where it is, and the report lists both;
    * every channel and the labels are **one lattice in one physical space** ---
      shape, spacing, origin and direction, coordinate system, units, and no two
      known frames of reference --- because nnU-Net pairs their voxels by index;
    * the labels are the **dataset's**, every case's classes, and every case was
      **searched for every one** (``annotated_class_ids``): a label volume is
      exhaustive, so 0 says "examined, absent" for every class, which a case
      not searched for one cannot say.  *classes* exports the subset every case
      examined.  A reused ``dataset.json`` that does not name a class some case
      carries is extended with it, or --- region-based --- refused;
    * label values are **consecutive** (``0..K``), as nnU-Net requires: class
      ids with a gap are written as ``1..K`` in ascending order, and
      ``dataset.json`` records the mapping (``medh5_class_ids``), which
      :func:`from_nnunetv2` reads back;
    * every label volume **gives back every class it exports** --- a region
      painted in ``regions_class_order``, overlapping classes refused rather than
      written over each other --- and every one is a training case's: a
      refused case refuses the export, before any case's files are written.

    Every file is written through the engine's replace (§14.4): to a
    temporary beside it, renamed over the old one once whole, so an
    interrupted export leaves each file whole, the old one or the new.
    ``dataset.json`` is the last file written, and with *replace* the old
    dataset's files that the export no longer writes are removed only after
    it.  An export interrupted part way leaves the dataset a mix of the old
    and the new until it is run again, which completes it.
    """
    from medh5.io.nifti import NIFTI_SUFFIXES

    if unlabeled not in UNLABELED:
        raise MEDH5ValidationError(
            f"unlabeled= is one of {list(UNLABELED)}, not {unlabeled!r}"
        )
    if file_ending not in NIFTI_SUFFIXES:
        raise MEDH5ValidationError(
            f"file_ending= is one of {list(NIFTI_SUFFIXES)}, not {file_ending!r}: "
            "every file the export writes is NIfTI"
        )
    log = report or ConversionReport(converter="to-nnunet")
    root = _dataset_root(out, dataset_name)
    if replace:
        _require_replaceable(root)
    plan = _plan(paths, annotation, classes, log, unlabeled, split)
    _require_under(root, plan, file_ending)
    if not replace:
        _require_no_strays(root, plan, file_ending)
    labels = dict(plan.labels)
    # nnU-Net's ignore label is the highest value (§7.7): ignored voxels
    # written as 0 were verified negatives for every class.
    ignore_value = _ignore_value(labels) if _check_cases(plan, annotation) else None
    if ignore_value is not None:
        labels[IGNORE] = ignore_value
    if not replace:
        _write(plan, root, labels, ignore_value, annotation, file_ending, log)
        return log

    left, owned = _inventory(root)
    written = _write(plan, root, labels, ignore_value, annotation, file_ending, log)
    # After `dataset.json`, so an export interrupted before it has removed
    # nothing.
    removed = sorted(set(owned) - set(written))
    for relative in removed:
        (root / relative).unlink(missing_ok=True)
    log.decision(
        "replace",
        f"the dataset {root} was replaced in place, file by file: "
        f"{len(removed)} NIfTI file(s) of the old one that this export does "
        f"not write were removed, and {len(left)} file(s) that are not the "
        "export's were left where they are",
        {"removed": removed, "carried_over": left},
    )
    return log


def _write(
    plan: _Plan,
    root: Path,
    labels: dict[str, Any],
    ignore_value: int | None,
    annotation: str,
    file_ending: str,
    log: ConversionReport,
) -> list[str]:
    """Write the planned dataset under *root*, ``dataset.json`` last; the
    files written, relative to *root*."""
    import medh5

    written: list[str] = []

    def place(target: Path) -> str:
        written.append(target.relative_to(root).as_posix())
        return str(target)

    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    for case in plan.cases:
        (root / case.images).mkdir(exist_ok=True)
        if case.labeled:
            (root / case.labels).mkdir(exist_ok=True)

    for case in plan.cases:
        with medh5.open(case.path) as sample:
            ann = sample.annotations.get(annotation) if case.labeled else None
            for index, image_id in enumerate(plan.channels):
                target = root / case.images / f"{case.name}_{index:04d}{file_ending}"
                _save(
                    sample.images[image_id].grid,
                    sample.images[image_id].read(),
                    target,
                    rescale=sample.images[image_id].rescale,
                )
                log.outputs.append(place(target))
            if ann is not None:
                volume = _labelmap_for(ann, labels, plan.order, plan.by_label)
                ignored = np.asarray(sample.ignore_region(annotation), dtype=bool)
                if ignored.any():
                    assert ignore_value is not None  # `_check_cases` saw it
                    covered = int(np.count_nonzero(ignored & (volume != BACKGROUND)))
                    volume[ignored] = ignore_value
                    log.decision(
                        "ignore",
                        f"{case.path}: {int(ignored.sum())} ignored voxel(s) were "
                        f"written as nnU-Net's ignore label {ignore_value}"
                        + (
                            f", {covered} of them over a class, which nnU-Net "
                            "neither trains nor scores there"
                            if covered
                            else ""
                        ),
                        {
                            "case": case.name,
                            "voxels": int(ignored.sum()),
                            "over": covered,
                        },
                    )
                target = root / case.labels / f"{case.name}{file_ending}"
                _save(ann.grid, volume, target)
                # Listed only when written.  A sample without the annotation
                # produced no label file and the report named one anyway, so a
                # caller checking `outputs` for what to ship was told about a
                # path that did not exist.
                log.outputs.append(place(target))

    stashed = plan.stashed
    document = dict(stashed) if stashed else {}
    document.update(
        {
            "channel_names": {str(i): n for i, n in enumerate(plan.channel_names)},
            "labels": labels or document.get("labels") or {},
            # The training cases only: a case written to imagesTs is not one.
            "numTraining": sum(1 for case in plan.cases if case.training),
            "file_ending": file_ending,
        }
    )
    if plan.order is None:
        document.pop("regions_class_order", None)
    if plan.class_ids:
        document["medh5_class_ids"] = {
            str(value): class_id for value, class_id in sorted(plan.class_ids.items())
        }
    else:
        document.pop("medh5_class_ids", None)
    # The last file written: a dataset interrupted before it keeps the old
    # description, whole (§14.4).
    write_text(root / "dataset.json", json.dumps(document, indent=2) + "\n")
    log.outputs.append(place(root / "dataset.json"))
    log.decision(
        "dataset_json",
        "the stashed dataset.json was reused: its fields as stashed, its labels "
        "as the notes above leave them, and numTraining and file_ending those "
        "of the files written"
        if plan.reused
        else "a dataset.json was generated from the label set",
        {"reused": plan.reused},
    )
    return written


OWNED = ("imagesTr", "labelsTr", "imagesTs", "labelsTs")
"""The folders whose NIfTI files are an exported dataset's own: nnU-Net
reads every case file there, so :func:`to_nnunetv2` refuses one it would
not write, or with ``replace`` removes it."""


def _require_replaceable(root: Path) -> None:
    """Refuse to replace what is not an nnU-Net dataset directory.

    ``replace`` removes the old dataset's files that the export does not
    write, so it holds itself to what an export could have written: a
    directory of its own --- not a link, which would rewrite the tree it
    names wherever that is --- holding a ``dataset.json``.
    """
    why = None
    if root.is_symlink():
        why = "is a symbolic link; name the dataset directory itself"
    elif not root.exists():
        why = "does not exist; export without replace to write a new dataset"
    elif not root.is_dir():
        why = "is not a directory"
    elif not (root / "dataset.json").is_file():
        why = (
            "holds no dataset.json, so it is no nnU-Net dataset; replace never "
            "deletes a directory that is not one"
        )
    if why is not None:
        raise MEDH5ValidationError(
            f"replace=True (--replace) replaces an nnU-Net dataset, and {root} "
            f"{why}; nothing was written"
        )


def _owned(relative: PurePosixPath, path: Path) -> bool:
    """Whether the file at *relative* in a dataset is the export's: its
    ``dataset.json``, or a NIfTI file in one of the case folders."""
    from medh5.io.nifti import NIFTI_SUFFIXES

    if relative.as_posix() == "dataset.json":
        return True
    return (
        len(relative.parts) == 2
        and relative.parts[0] in OWNED
        and relative.name.endswith(NIFTI_SUFFIXES)
        and not path.is_dir()
    )


def _inventory(root: Path) -> tuple[list[str], list[str]]:
    """The files of *root* that are not the export's, which a replace leaves
    where they are, and the export's own, as sorted relative paths.

    A link is listed as itself, never followed.
    """
    left: list[str] = []
    owned: list[str] = []
    for directory, folders, files in os.walk(root):
        here = Path(directory)
        relative = PurePosixPath(here.relative_to(root).as_posix())
        for name in list(folders):
            if (here / name).is_symlink():
                folders.remove(name)
                files.append(name)
        for name in files:
            entry = relative / name
            (owned if _owned(entry, here / name) else left).append(entry.as_posix())
    return sorted(left), sorted(owned)


UNLABELED = ("refuse", "test")
"""What :func:`to_nnunetv2` does with a case that lacks the annotation."""


def _check_cases(plan: _Plan, annotation: str) -> bool:
    """Every refusal a case's label volume can draw, made before a file is
    written; whether any case has an ignore region to write.

    Made as each case was written, a refusal of the second case left the
    first case's files on disk and no ``dataset.json``.  The volumes are
    built again as they are written rather than held for the whole dataset.
    """
    import medh5

    ignored_anywhere = False
    for case in plan.cases:
        if not case.labeled:
            continue
        with medh5.open(case.path) as sample:
            ann = sample.annotations[annotation]
            volume = _labelmap_for(ann, plan.labels, plan.order, plan.by_label)
            ignored = np.asarray(sample.ignore_region(annotation), dtype=bool)
            lost = _not_given_back(
                ann, volume, plan.labels, ignored, plan.by_label, plan.exported
            )
            if lost:
                raise MEDH5ValidationError(
                    f"{case.path}: the label volume does not give back {lost}: "
                    "nnU-Net holds one value per voxel, so classes that overlap "
                    "must be stated as regions over disjoint components; nothing "
                    "was written"
                )
            ignored_anywhere = ignored_anywhere or bool(ignored.any())
    return ignored_anywhere


def _dataset_root(out: str | os.PathLike[str], dataset_name: str) -> Path:
    """The dataset's directory: a name, never a path."""
    _require_name(dataset_name, "dataset_name")
    return Path(os.fspath(out)) / dataset_name


def _require_name(name: str, what: str) -> None:
    """*name* as one file name: a sample key, and not ``.`` or ``..``.

    Uncoded: ``sample_id`` is free text in a valid sample, so a refusal here
    is the export's, not a defect of the file (§15.2).
    """
    from medh5 import _core

    try:
        _core.validate_sample_key(name)
    except MEDH5ValidationError as e:
        raise MEDH5ValidationError(
            f"{what} {name!r} cannot name a file of the export: {e.message}"
        ) from None
    if name in (".", ".."):
        raise MEDH5ValidationError(f"{what} {name!r} cannot name a file of the export")


def _require_under(root: Path, plan: _Plan, file_ending: str) -> None:
    """Every file the plan writes resolves under *root* --- what the case
    names already ensure, checked where the files are named.

    Each one, every channel's: a file is written through a link at its
    name (§14.4), so a link at a case file naming one outside the dataset
    would have the export write there.
    """
    top = root.resolve()
    for case in plan.cases:
        for target in (
            *(
                root / case.images / f"{case.name}_{index:04d}{file_ending}"
                for index in range(len(plan.channels))
            ),
            root / case.labels / f"{case.name}{file_ending}",
        ):
            if not target.resolve().is_relative_to(top):
                raise MEDH5ValidationError(
                    f"{case.path}: case {case.name!r} would write {target}, "
                    f"outside the dataset {root}"
                )


def _require_no_strays(root: Path, plan: _Plan, file_ending: str) -> None:
    """Refuse a dataset directory holding NIfTI files this export would not
    write, before anything is written.

    nnU-Net trains on every case file it finds in ``imagesTr``, whatever
    ``numTraining`` says.  An export into a directory an earlier one wrote
    left whatever the new plan dropped --- a removed case, a dropped
    channel, a case moved to ``imagesTs`` still in ``imagesTr`` (train and
    test at once), files of the other ending --- under a ``dataset.json``
    describing none of them, and :func:`from_nnunetv2` read the removed
    cases back.  An identical export rewrites its own files, and a file that
    is not NIfTI is not the export's to judge.
    """
    from medh5.io.nifti import NIFTI_SUFFIXES

    planned: set[Path] = set()
    for case in plan.cases:
        planned.update(
            root / case.images / f"{case.name}_{index:04d}{file_ending}"
            for index in range(len(plan.channels))
        )
        if case.labeled:
            planned.add(root / case.labels / f"{case.name}{file_ending}")
    stray = sorted(
        f"{folder}/{path.name}"
        for folder in OWNED
        if (root / folder).is_dir()
        for path in (root / folder).iterdir()
        if path.name.endswith(NIFTI_SUFFIXES)
        and not path.is_dir()
        and path not in planned
    )
    if stray:
        shown = ", ".join(stray[:5]) + (
            f" and {len(stray) - 5} more" if len(stray) > 5 else ""
        )
        raise MEDH5ValidationError(
            f"{root} holds {len(stray)} NIfTI file(s) this export would not "
            f"write ({shown}): nnU-Net trains on every case file it finds, "
            "whatever dataset.json says, so they would join the new dataset; "
            "export to a new directory, pass replace=True (--replace) to "
            "replace the dataset, or remove them first --- nothing was written"
        )


class _Case:
    """One sample of the export: its file, the name its files take, whether
    it carries the annotation, and whether a split claims it for ``test``."""

    __slots__ = ("labeled", "name", "path", "test")

    def __init__(self, path: str, name: str, labeled: bool) -> None:
        self.path = path
        self.name = name
        self.labeled = labeled
        self.test = False

    @property
    def training(self) -> bool:
        """Whether the case is one of ``numTraining``: in ``imagesTr``."""
        return self.labeled and not self.test

    @property
    def images(self) -> str:
        """The folder its channels are written to."""
        return "imagesTr" if self.training else "imagesTs"

    @property
    def labels(self) -> str:
        """The folder its label volume is written to, when it has one."""
        return "labelsTs" if self.test else "labelsTr"


class _Plan:
    """What an export writes, settled before it writes anything."""

    __slots__ = (
        "by_label",
        "cases",
        "channel_names",
        "channels",
        "class_ids",
        "exported",
        "labels",
        "order",
        "reused",
        "stashed",
    )

    def __init__(
        self,
        cases: list[_Case],
        channels: list[str],
        channel_names: list[str],
        labels: dict[str, Any],
        order: list[int] | None,
        stashed: dict[str, Any] | None,
        reused: bool,
        by_label: dict[str, int],
        class_ids: dict[int, int],
        exported: frozenset[int],
    ) -> None:
        self.cases = cases
        # The image each channel is read from, and the name `dataset.json`
        # gives it, which nnU-Net reads as its normalisation.
        self.channels = channels
        self.channel_names = channel_names
        self.labels = labels
        self.order = order
        self.stashed = stashed
        self.reused = reused
        # A scalar label's class, where the plan knows it: what its value
        # cannot say once the values are renumbered.
        self.by_label = by_label
        # Written value -> class id, when the values are not the ids.
        self.class_ids = class_ids
        # The classes every label volume must give back.
        self.exported = exported


def _plan(
    paths: Sequence[str | os.PathLike[str]],
    annotation: str,
    classes: Sequence[int | str] | None,
    log: ConversionReport,
    unlabeled: str = "refuse",
    split: str | None = None,
) -> _Plan:
    """Every case read once, and every refusal but the label volumes' own
    (:func:`_check_cases`) made, before a file is written.

    The labels were the first annotation's and every later case was written
    against them: a class only a later case carried became background there,
    and a case never searched for a class was written as its verified
    absence.  The vocabulary is every case's; a case is held to it, or to
    *classes*; and every channel and the labels are held to the first
    channel's physical space.  Case names are file names, a case without
    labels is no training case, a reused ``dataset.json`` names every class,
    and the label values are consecutive.
    """
    import medh5

    stashed: dict[str, Any] | None = None
    channels: list[str] = []
    channel_names: list[str] = []
    names: dict[int, str] = {}
    searched: list[tuple[str, frozenset[int]]] = []
    cases: list[_Case] = []
    taken: dict[str, str] = {}
    claims: dict[str, list[str]] = {}
    for path in paths:
        with medh5.open(path) as sample:
            name = sample.identity.sample_id
            _require_name(name, f"{os.fspath(path)}: sample_id")
            # Case-insensitive file systems hold one of two names that differ
            # only in case, and the second case overwrote the first.
            folded = name.casefold()
            if folded in taken:
                raise MEDH5ValidationError(
                    f"{os.fspath(path)}: sample_id {name!r} names the same files "
                    f"as {taken[folded]!r} (on a case-insensitive file system); "
                    "each case of a dataset needs a name of its own"
                )
            taken[folded] = name
            stashed = stashed or sample.document.extra.get("nnunetv2")
            if not channels:
                channels, channel_names = _channels_of(sample, stashed, log)
            for image_id in channels:
                if image_id not in sample.images:
                    raise MEDH5ValidationError(
                        f"{path}: no image {image_id!r}; the export needs the same "
                        f"channels in every case ({channels})"
                    )
            reference = sample.images[channels[0]].grid
            for image_id in channels[1:]:
                _require_one_space(
                    reference,
                    sample.images[image_id].grid,
                    f"{path}: channel {image_id!r}",
                    channels[0],
                )
            labeled = annotation in sample.annotations
            case = _Case(os.fspath(path), name, labeled)
            cases.append(case)
            if split is not None:
                claimed = sorted(
                    {c.partition for c in sample.document.splits if c.set_id == split}
                )
                claims[case.path] = claimed
                case.test = claimed == ["test"]
            if not labeled:
                continue
            ann = sample.annotations[annotation]
            # On the annotation's own grid, which nnU-Net needs to be the
            # channels': the reference grid's affine was written whatever grid
            # the labels were on.
            _require_one_space(
                reference, ann.grid, f"{path}: annotation {annotation!r}", channels[0]
            )
            for class_id in ann.class_ids:
                key = ann.class_key(int(class_id))
                if names.setdefault(int(class_id), key) != key:
                    raise MEDH5ValidationError(
                        f"{path}: class {int(class_id)} is {key!r} here and "
                        f"{names[int(class_id)]!r} in an earlier case; one label "
                        "volume value means one class across the dataset"
                    )
            searched.append(
                (os.fspath(path), frozenset(int(c) for c in ann.annotated_class_ids))
            )

    if split is not None:
        _require_placed(claims, split, log)
    bare = [case.path for case in cases if not case.labeled and not case.test]
    if bare and split is not None:
        raise MEDH5ValidationError(
            f"{bare} carry no annotation {annotation!r} and claim train or val "
            f"in split {split!r}: an imagesTr case is an image with its label "
            "volume, and unlabeled='test' would move them out of the partition "
            "their claims place them in; nothing was written"
        )
    if bare and unlabeled == "refuse":
        raise MEDH5ValidationError(
            f"{bare} carry no annotation {annotation!r}: an nnU-Net training case "
            "is an image with its label volume, so these cannot be written as "
            "ones; pass unlabeled='test' (--unlabeled test) to write them to "
            "imagesTs, outside numTraining"
        )
    if bare:
        log.decision(
            "unlabeled",
            f"{len(bare)} case(s) without annotation {annotation!r} were written "
            "to imagesTs, and are not counted in numTraining",
            {"cases": bare},
        )

    wanted = set(names)
    if classes is not None:
        wanted = {_class_id(c, names) for c in classes}
        dropped = sorted(set(names) - wanted)
        if dropped:
            log.decision(
                "classes",
                f"classes= exports {sorted(wanted)}; {[names[c] for c in dropped]} "
                "were left out on request",
                {"exported": sorted(wanted), "left_out": dropped},
            )
    for path, looked in searched:
        unsearched = sorted(wanted - looked)
        if unsearched:
            raise MEDH5ValidationError(
                f"{path}: annotation {annotation!r} was not searched for "
                f"{[names[c] for c in unsearched]}, which the dataset labels. An "
                "nnU-Net label volume is exhaustive --- 0 is a verified negative "
                "for every class --- so this case cannot be written; pass "
                "classes= (--class) naming the classes every case examined"
            )
    if searched:
        log.decision(
            "coverage",
            "every case was searched for every exported class, so 0 in each label "
            "volume is the verified absence it means to nnU-Net (§11.3)",
            {"classes": sorted(wanted), "cases": len(searched)},
        )

    reused = bool(stashed and stashed.get("labels")) and classes is None
    by_label: dict[str, int] = {}
    if reused:
        assert stashed is not None
        labels = dict(stashed["labels"])
        order = stashed.get("regions_class_order")
        order = [int(v) for v in order] if isinstance(order, list) else None
        by_label = _stashed_classes(labels, stashed, names)
        _extend_stash(labels, by_label, order, names, wanted, log)
    else:
        labels = {"background": 0}
        labels.update({names[c]: c for c in sorted(wanted)})
        by_label = {names[c]: c for c in sorted(wanted)}
        order = None
        if stashed and classes is not None:
            log.decision(
                "labels",
                "classes= restricts the labels, so the stashed dataset.json's were "
                "not reused; its other fields were",
                {"classes": sorted(wanted)},
            )
    if order is None:
        labels = _consecutive(labels, log)
    # Each written value whose class is another id, which the import reads
    # back: what a renumbered export, or a reused one, needs recorded.
    class_ids = {
        int(labels[name]): class_id
        for name, class_id in by_label.items()
        if not isinstance(labels.get(name, []), list) and int(labels[name]) != class_id
    }
    return _Plan(
        cases,
        channels,
        channel_names,
        labels,
        order,
        stashed,
        reused,
        by_label,
        class_ids,
        frozenset(wanted),
    )


def _require_placed(
    claims: Mapping[str, Sequence[str]], split: str, log: ConversionReport
) -> None:
    """Refuse a case that split set *split* does not place in ``imagesTr``
    or ``imagesTs``; note where the claims sent the cases.

    ``train`` and ``val`` are ``imagesTr``'s, since nnU-Net cross-validates
    its training cases itself, and ``test`` is ``imagesTs``'s.  A case
    claiming nothing in the set is refused rather than trained on --- a
    mistyped set id would put every test case in training --- and so is one
    whose claims disagree (W906), and one held out of both.
    """
    unplaced = sorted(path for path, found in claims.items() if not found)
    if unplaced:
        raise MEDH5ValidationError(
            f"{unplaced} claim no partition in split {split!r} (§12.3): split= "
            "(--split) sends a case claiming test to imagesTs and one claiming "
            "train or val to imagesTr, so a case the set does not place has no "
            "folder; export the set's cases, or check the set id --- nothing "
            "was written"
        )
    torn = {path: found for path, found in claims.items() if len(found) > 1}
    if torn:
        raise MEDH5ValidationError(
            f"{sorted(torn)} claim more than one partition in split {split!r} "
            f"({sorted(set(p for f in torn.values() for p in f))}), which "
            "conflict (W906); nothing was written"
        )
    held = {
        path: found[0]
        for path, found in claims.items()
        if found[0] not in ("train", "val", "test")
    }
    if held:
        raise MEDH5ValidationError(
            f"{sorted(held)} claim {sorted(set(held.values()))} in split "
            f"{split!r}: a holdout case is kept from both training and testing, "
            "and an unassigned one belongs to neither, so neither has an nnU-Net "
            "folder; leave them out of the export --- nothing was written"
        )
    test = sorted(path for path, found in claims.items() if found == ["test"])
    log.decision(
        "split",
        f"split {split!r} (§12.3): {len(test)} case(s) claiming test were written "
        "to imagesTs, their labels to labelsTs, and "
        f"{len(claims) - len(test)} claiming train or val to imagesTr",
        {"set_id": split, "test": test, "train": len(claims) - len(test)},
    )


def _stashed_classes(
    labels: Mapping[str, Any], stashed: Mapping[str, Any], names: Mapping[int, str]
) -> dict[str, int]:
    """The class each scalar label of a reused ``dataset.json`` writes.

    An export that renumbered recorded the ids (``medh5_class_ids``); an
    imported dataset's values are its class ids; a label naming neither is
    matched by its key.  A label matched by nothing is left to the case,
    which refuses it (E402).
    """
    recorded = {
        int(v): int(c) for v, c in (stashed.get("medh5_class_ids") or {}).items()
    }
    keys = {key: class_id for class_id, key in names.items()}
    out: dict[str, int] = {}
    for name, value in labels.items():
        if name == IGNORE or isinstance(value, list) or int(value) == BACKGROUND:
            continue
        if int(value) in recorded:
            out[name] = recorded[int(value)]
        elif int(value) in names:
            out[name] = int(value)
        elif _key(name) in keys:
            out[name] = keys[_key(name)]
    return out


def _extend_stash(
    labels: dict[str, Any],
    by_label: dict[str, int],
    order: list[int] | None,
    names: Mapping[int, str],
    wanted: set[int],
    log: ConversionReport,
) -> None:
    """Name in a reused ``dataset.json`` every class some case carries.

    A class the stash did not name was written as background: a dataset
    imported with one class and given another later exported the second as 0
    in every case, and ``dataset.json`` without it.
    A scalar stash is extended; a region-based one cannot be without changing
    what ``regions_class_order`` paints, so it is refused.
    """
    keys = {_key(str(name)) for name in labels}
    named = set(by_label.values()) | {c for c in wanted if names[c] in keys}
    missing = sorted(wanted - named)
    if not missing:
        return
    if order is not None or any(isinstance(v, list) for v in labels.values()):
        raise MEDH5ValidationError(
            f"the stashed dataset.json does not name {[names[c] for c in missing]}, "
            "which the cases carry, and it is region-based: a class cannot be "
            "added to its regions without changing what regions_class_order "
            "paints. Pass classes= (--class) to export a new dataset.json"
        )
    used: set[int] = set()
    for value in labels.values():
        used.update(int(v) for v in (value if isinstance(value, list) else [value]))
    for class_id in missing:
        if class_id in used:
            raise MEDH5ValidationError(
                f"the stashed dataset.json writes {class_id} for another label, so "
                f"class {names[class_id]!r} cannot take it; pass classes= "
                "(--class) to export a new dataset.json"
            )
        labels[names[class_id]] = class_id
        by_label[names[class_id]] = class_id
    log.decision(
        "labels",
        f"the stashed dataset.json did not name {[names[c] for c in missing]}, "
        "which the cases carry; they were added to its labels, so no class is "
        "written as background",
        {"added": {names[c]: c for c in missing}},
    )


def _consecutive(labels: dict[str, Any], log: ConversionReport) -> dict[str, Any]:
    """*labels* with values ``0..K``, as nnU-Net requires.

    nnU-Net refuses label values with a gap, and class ids have them --- a
    subset of a dataset's classes, or ids an import kept: ``{1, 3}`` was
    written as given, and nnU-Net's dataset check failed.  The values are
    renumbered ``1..K`` in ascending order; labels already consecutive are
    left as they are.
    """
    scalar = sorted(
        int(v)
        for name, v in labels.items()
        if name != IGNORE and not isinstance(v, list) and int(v) != BACKGROUND
    )
    if scalar == list(range(1, len(scalar) + 1)):
        return labels
    value = {old: new for new, old in enumerate(scalar, start=1)}
    out: dict[str, Any] = {}
    for name, v in labels.items():
        if name == IGNORE:
            continue
        if isinstance(v, list):
            out[name] = [value.get(int(x), int(x)) for x in v]
        else:
            out[name] = value.get(int(v), int(v))
    log.decision(
        "label_values",
        f"nnU-Net needs consecutive label values; the values {scalar} were written "
        f"as 1..{len(scalar)}, and dataset.json records each one's class id "
        "(medh5_class_ids), which the import reads back",
        {"values": {str(new): old for old, new in value.items()}},
    )
    return out


def _class_id(key: int | str, names: Mapping[int, str]) -> int:
    """A class named by id or key, among the dataset's."""
    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        if int(key) in names:
            return int(key)
    else:
        for class_id, name in names.items():
            if name == key:
                return class_id
    raise MEDH5ValidationError(
        f"classes= names {key!r}, which no case's annotation carries; the "
        f"dataset's are {dict(sorted(names.items()))}",
        code="E402",
    )


def _require_one_space(reference: Any, grid: Any, what: str, channel: str) -> None:
    """Refuse a grid that is not *reference*'s physical support.

    nnU-Net pairs channel and label voxels by index, and each NIfTI states its
    own affine: a congruent lattice in LPS against one in RAS, or in metres
    against millimetres, wrote a pair 48 mm --- or a thousandfold --- apart,
    and equal numbers in two known frames of reference are no correspondence.
    Converting between them is resampling, which is not a converter's to do.
    """
    if grid.grid_id == reference.grid_id:
        return
    problems = []
    if not grid.is_congruent(reference):
        problems.append("another lattice")
    if grid.coord_system != reference.coord_system:
        problems.append(f"{grid.coord_system} against {reference.coord_system}")
    if grid.units != reference.units:
        problems.append(f"{grid.units!r} against {reference.units!r}")
    if grid.frame_uid and reference.frame_uid and grid.frame_uid != reference.frame_uid:
        problems.append(
            f"frame of reference {grid.frame_uid!r} against {reference.frame_uid!r}"
        )
    if problems:
        raise MEDH5ValidationError(
            f"{what} is on grid {grid.grid_id!r} and channel {channel!r} on "
            f"{reference.grid_id!r} ({'; '.join(problems)}); nnU-Net needs every "
            "channel and the labels on the channels' grid, and resampling is not "
            "a converter's to do"
        )


def _ignore_value(labels: Mapping[str, Any]) -> int:
    """The value nnU-Net's ignore label takes: its own, or one above every
    other, as nnU-Net requires."""
    if IGNORE in labels and not isinstance(labels[IGNORE], list):
        return int(labels[IGNORE])
    values = [BACKGROUND]
    for value in labels.values():
        values.extend(int(v) for v in (value if isinstance(value, list) else [value]))
    if max(values) + 1 > np.iinfo(np.uint16).max:
        raise MEDH5ValidationError(
            f"labels use {max(values)}, so nnU-Net's ignore label, which must be "
            "the highest, does not fit the uint16 label volume"
        )
    return max(values) + 1


def _stashed_channels(stashed: Mapping[str, Any] | None) -> dict[int, str]:
    if not stashed:
        return {}
    return {int(k): str(v) for k, v in stashed.get("channel_names", {}).items()}


def _channels_of(
    sample: Any, stashed: Mapping[str, Any] | None, log: ConversionReport
) -> tuple[list[str], list[str]]:
    """The image each channel is read from, and the name it is written under.

    A stash names its channels, and they are its images as the import named
    them (:func:`_image_ids`).  Otherwise every image is a channel, named for
    its image id --- except a CT image, named ``CT``: nnU-Net reads a
    channel's name as its normalisation, ``CT`` its CT scheme and any other
    name z-scoring, so a CT imported from DICOM, ``CT_tp0``, was z-scored.
    """
    if stashed:
        named = _stashed_channels(stashed)
        ids = _image_ids(named)
        return [ids[i] for i in sorted(named)], [named[i] for i in sorted(named)]
    channels = sorted(sample.images)
    names = [
        "CT" if str(sample.images[i].modality or "").upper() == "CT" else i
        for i in channels
    ]
    renamed = {i: n for i, n in zip(channels, names, strict=True) if i != n}
    if renamed:
        log.decision(
            "channel_names",
            f"nnU-Net normalises a channel by its name, CT intensities by "
            f"their own scheme and any other name by z-scoring; the CT "
            f"image(s) {sorted(renamed)} were named 'CT'",
            {"channel_names": dict(zip(channels, names, strict=True))},
        )
    return channels, names


def _labelmap_for(
    ann: Any,
    labels: Mapping[str, Any],
    order: Sequence[int] | None = None,
    by_label: Mapping[str, int] | None = None,
) -> npt.NDArray[np.uint16]:
    """A single-value label volume, which is what nnU-Net reads.

    Without ``regions_class_order`` a region is *not* written as its own value:
    nnU-Net derives it from its components, and writing both would
    double-count every voxel.  With it (*order*), the dataset is region-based:
    each label, in ``labels`` order, is painted with its value in *order*, a
    later one over an earlier --- nnU-Net's own conversion back --- so the
    components a region-based dataset names only inside its regions are written
    too.  They were left as background.

    Classes are matched by **id, not by name**.  The import keeps nnU-Net's own
    integers as class ids precisely so no translation table is needed, and the
    name in ``dataset.json`` is free text that ``_key`` sanitises on the way in
    --- so a dataset naming a class ``"Tumour Core"`` stores the key
    ``tumour_core``, and looking the original name back up finds nothing.  That
    lookup used to fail into a bare ``continue``, so every class of any dataset
    whose labels are not already lowercase identifiers was dropped and the
    export wrote an all-background volume with no indication anything was lost.
    A label whose class the plan knows (*by_label*) writes that class, whatever
    its value: renumbered values are no ids.
    """
    known = set(ann.class_ids)
    out = np.zeros(ann.spatial_shape, dtype=np.uint16)
    missing: list[str] = []
    if order is not None:
        entries = _foreground(labels)
        if len(order) != len(entries):
            raise MEDH5ValidationError(
                f"regions_class_order has {len(order)} value(s) for {len(entries)} "
                "label(s); it gives each label, in order, the value it is written as"
            )
        for (name, value), paint in zip(entries, order, strict=True):
            class_id = _class_of(ann, name, value, known)
            if class_id is None:
                missing.append(name)
                continue
            out[ann.dense([class_id])[0]] = int(paint)
    else:
        scalar = {
            name: int(value)
            for name, value in labels.items()
            if not isinstance(value, list) and name != IGNORE
        }
        for name, value in sorted(scalar.items(), key=lambda kv: kv[1]):
            if value == BACKGROUND:
                continue
            class_id = (by_label or {}).get(name)
            if class_id is None:
                class_id = value if value in known else _resolve_or_none(ann, name)
            if class_id is None:
                missing.append(name)
                continue
            out[ann.dense([class_id])[0]] = value
    if missing:
        raise MEDH5ValidationError(
            f"annotation {ann.ann_id!r} carries no class for {missing}, which "
            f"dataset.json names; exporting would write a label volume missing "
            f"those structures without saying so",
            code="E402",
        )
    return out


def _foreground(labels: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The labels a volume value or region states, in ``labels`` order."""
    return [
        (name, value)
        for name, value in labels.items()
        if name != IGNORE and (isinstance(value, list) or int(value) != BACKGROUND)
    ]


def _class_of(ann: Any, name: str, value: Any, known: set[int]) -> int | None:
    """The annotation's class a label names: a value by id, a region by name."""
    if not isinstance(value, list) and int(value) in known:
        return int(value)
    return _resolve_or_none(ann, name)


def _not_given_back(
    ann: Any,
    volume: npt.NDArray[Any],
    labels: Mapping[str, Any],
    ignored: npt.NDArray[np.bool_],
    by_label: Mapping[str, int] | None = None,
    exported: frozenset[int] | None = None,
) -> list[str]:
    """The labels whose stored voxels *volume* does not give back, outside the
    ignore region --- what an export would lose without saying so.

    Each label is read back as nnU-Net reads it: a value as the voxels of that
    value, a region as the voxels of any of its components.  And every
    *exported* class the annotation stores voxels of must be some label's: a
    class the vocabulary did not name was written as background, and checking
    the vocabulary alone could not see it.
    """
    known = set(ann.class_ids)
    out: list[str] = []
    read: set[int] = set()
    for name, value in _foreground(labels):
        class_id = None if isinstance(value, list) else (by_label or {}).get(name)
        if class_id is None:
            class_id = _class_of(ann, name, value, known)
        if class_id is None:
            continue
        read.add(class_id)
        components = (
            [int(v) for v in value] if isinstance(value, list) else [int(value)]
        )
        stored = np.asarray(ann.dense([class_id])[0], dtype=bool) & ~ignored
        written = np.isin(volume, components) & ~ignored
        if not np.array_equal(stored, written):
            lost = int(np.count_nonzero(stored & ~written))
            gained = int(np.count_nonzero(written & ~stored))
            out.append(f"{name!r} ({lost} voxel(s) lost, {gained} gained)")
    for class_id in sorted((exported or frozenset()) & known - read):
        stored = np.asarray(ann.dense([class_id])[0], dtype=bool) & ~ignored
        if stored.any():
            out.append(
                f"class {class_id} ({int(stored.sum())} voxel(s) no label names)"
            )
    return out


def _resolve_or_none(ann: Any, name: str) -> int | None:
    """Fall back to name resolution for a label set that renumbered."""
    for candidate in (name, _key(name)):
        try:
            return int(ann.resolve_class(candidate))
        except Exception:
            continue
    return None


def _save(
    grid: Any,
    array: npt.NDArray[Any],
    path: Path,
    *,
    rescale: tuple[float, float] | None = None,
) -> None:
    """Write one channel or label volume, through the one NIfTI writer.

    *rescale* is the image's §4.2 scale.  ``Image.read()`` returns **stored**
    values, so writing them with no ``scl_slope``/``scl_inter`` handed nnU-Net
    a CT 1024 HU off its own units, in the exporter's default path, with
    nothing in ``dataset.json`` or the report to say so.
    """
    from medh5.io.nifti import for_export, write_nifti

    data, affine = for_export(grid, np.asarray(array))
    write_nifti(
        data,
        affine,
        path,
        rescale=rescale,
        units=grid.units,
        time_values=grid.time_values,
        time_units=grid.time_units,
    )


__all__ = [
    "BACKGROUND",
    "REQUIRED_KEYS",
    "cases",
    "from_nnunetv2",
    "read_dataset_json",
    "to_nnunetv2",
]
