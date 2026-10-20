"""Builders the test modules share.

Samples are built by the public writer, so every test that reads one is also a
test that the writer produces something readable.  ``write_dicom_series`` and
``write_legacy_sample`` write the inputs the converters read, and the h5py
codecs at the end plant what the writer would refuse: the package reads and
writes through the format engine, so a test that breaks a file on purpose
needs a second, independent writer.  Close every h5py handle before the
package reads the file.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.labels import LabelSet

ROOT = Path(__file__).resolve().parents[1]
"""The repository root."""


def _next_minor(version: str) -> str:
    major, _, minor = version.partition(".")
    return f"{major}.{int(minor or 0) + 1}"


LATER_MINOR = _next_minor(medh5.FORMAT_VERSION)
"""A minor this engine does not implement: the one after the newest it does
(the engine's ``version::next_minor``).  What a test needing "a later minor"
declares, so an engine that implements it moves the tests past it."""

SEED = 20260815

SHAPE = (16, 24, 24)


def block(shape: tuple[int, ...], origin: tuple[int, ...], size: int = 6) -> Any:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slice(o, o + size) for o in origin)] = True
    return mask


def write_sample(
    path: Path,
    *,
    label_set: LabelSet | None = None,
    masks: dict[int, Any] | None = None,
    ct: Any = None,
    timepoints: tuple[str, ...] = ("tp0",),
    index: bool = False,
    encoding: str = "auto",
    annotated: Any = "all_given",
    codec: str = "portable",
    sample_id: str | None = None,
    image_name: str = "CT_{tp}",
) -> Path:
    """A complete, valid sample --- the base every reader test starts from.

    ``image_name`` names each timepoint's image; ``{tp}`` is the timepoint.
    """
    rng = np.random.default_rng(SEED)
    image = ct if ct is not None else rng.integers(-1000, 1500, SHAPE).astype(np.int16)
    with medh5.create(
        path, sample_id=sample_id or path.stem, subject_id="subj-A", codec=codec
    ) as w:
        w.identity(sex="F", bodypart="abdomen")
        w.cohort(dataset_id="test", site_id="site-A")
        for i, tp in enumerate(timepoints):
            w.add_timepoint(
                tp, label="baseline" if i == 0 else f"fu{i}", days_from_baseline=90 * i
            )
        if label_set is not None:
            w.label_set(label_set)
        tool = w.software("medh5", medh5.__version__)
        act = w.activity("import", agent=tool, tool="test suite")
        for tp in timepoints:
            w.add_grid(
                f"ct_{tp}",
                shape=SHAPE,
                spacing=(1.5, 0.8, 0.8),
                origin=(-12.0, -9.6, -9.6),
                timepoint=tp,
                frame_uid=f"pseudo:frame-{tp}",
                patch_hint=(8, 8, 8),
            )
            w.add_image(
                image_name.format(tp=tp),
                image,
                grid=f"ct_{tp}",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
                prov=act,
            )
            if masks is not None:
                w.add_segmentation(
                    f"organs_{tp}",
                    grid=f"ct_{tp}",
                    masks=masks,
                    encoding=encoding,
                    annotated_classes=annotated,
                    prov=act,
                    quality={"status": "approved"},
                )
        if index and masks is not None:
            w.build_index(max_coords=64)
        # No amount of shift stated: a file shared beyond whoever holds the
        # key does not state one (§11.4), and a scrub reports one that does.
        w.deidentification(method="dicom-psi-profile")
    return path


def write_dicom_series(
    directory: Path,
    *,
    patient_id: str,
    study_uid: str,
    study_date: str,
    modality: str = "CT",
    shape: tuple[int, int, int] = (6, 16, 20),
    spacing: tuple[float, float, float] = (2.5, 0.8, 0.9),
    origin: tuple[float, float, float] = (-10.0, -20.0, 30.0),
    frame_uid: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """A minimal but valid CT/PT series.

    Two details are deliberate, because the converter's job is to survive them:
    ``SliceThickness`` is twice the slice increment (it is the slab, not the
    step), and ``InstanceNumber`` counts *down*, so a converter that trusts it
    rather than geometry produces a flipped volume.
    """
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    series_uid = generate_uid()
    frame = frame_uid or generate_uid()
    volume = rng.integers(0, 2000, shape).astype(np.uint16)
    for k in range(shape[0]):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.PatientID = patient_id
        ds.PatientName = "ANON^ANON"
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame
        ds.StudyID = "1"
        ds.AccessionNumber = ""
        ds.StudyDate = study_date
        ds.StudyTime = "120000"
        ds.ContentDate = study_date
        ds.ContentTime = "120000"
        ds.SeriesNumber = 1
        ds.Modality = modality
        ds.SeriesDescription = f"{modality} axial"
        ds.Manufacturer = "SYNTH"
        ds.ConvolutionKernel = "B30f"
        ds.SliceThickness = spacing[0] * 2
        ds.PixelSpacing = [spacing[1], spacing[2]]
        ds.ImageOrientationPatient = [0, 0, 1, 0, 1, 0]
        ds.ImagePositionPatient = [origin[0] - k * spacing[0], origin[1], origin[2]]
        ds.InstanceNumber = shape[0] - k
        ds.Rows, ds.Columns = shape[1], shape[2]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.PixelData = volume[k].tobytes()
        ds.save_as(str(directory / f"{modality}_{k:03d}.dcm"), enforce_file_format=True)
    assert pydicom is not None
    return {
        "series_uid": series_uid,
        "frame_uid": frame,
        "volume": volume,
        "paths": sorted(str(p) for p in directory.glob("*.dcm")),
    }


def write_legacy_sample(
    path: Path,
    *,
    images: dict[str, Any],
    seg: dict[str, Any] | None = None,
    bboxes: Any = None,
    bbox_labels: list[str] | None = None,
    bbox_scores: Any = None,
    spacing: list[float] | None = None,
    origin: list[float] | None = None,
    direction: list[list[float]] | None = None,
    coord_system: str | None = None,
    label: int | str | None = None,
    label_name: str | None = None,
    patch_size: list[int] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a 0.x file with plain h5py, to the layout documented in Appendix B.

    Deliberately not written by the 0.x package: 1.0 does not ship one.  The
    migration is therefore tested against the format as specified, not against
    whichever implementation happened to be in the tree.
    """
    import json

    import h5py

    with h5py.File(str(path), "w") as handle:
        group = handle.create_group("images")
        for name, array in images.items():
            group.create_dataset(name, data=np.asarray(array))
        first = np.asarray(next(iter(images.values())))
        group.attrs["shape"] = np.asarray(first.shape, dtype=np.int64)
        if spacing is not None:
            group.attrs["spacing"] = np.asarray(spacing, dtype=np.float64)
        if origin is not None:
            group.attrs["origin"] = np.asarray(origin, dtype=np.float64)
        if direction is not None:
            group.attrs["direction"] = np.asarray(direction, dtype=np.float64).ravel()
        if coord_system is not None:
            group.attrs["coord_system"] = coord_system
        if patch_size is not None:
            group.attrs["patch_size"] = np.asarray(patch_size, dtype=np.int64)

        handle.attrs["schema_version"] = "1"
        handle.attrs["image_names"] = json.dumps(sorted(images))
        handle.attrs["has_seg"] = bool(seg)
        handle.attrs["has_bbox"] = bboxes is not None
        if seg:
            masks = handle.create_group("seg")
            for name, mask in seg.items():
                masks.create_dataset(name, data=np.asarray(mask, dtype=bool))
            handle.attrs["seg_names"] = json.dumps(sorted(seg))
        if label is not None:
            handle.attrs["label"] = label
        if label_name is not None:
            handle.attrs["label_name"] = label_name
        if extra is not None:
            handle.attrs["extra"] = json.dumps(extra)
        if bboxes is not None:
            handle.create_dataset("bboxes", data=np.asarray(bboxes))
        if bbox_scores is not None:
            handle.create_dataset("bbox_scores", data=np.asarray(bbox_scores))
        if bbox_labels is not None:
            handle.create_dataset(
                "bbox_labels",
                data=np.array(bbox_labels, dtype=object),
                dtype=h5py.string_dtype(),
            )
    return path


# -- two visits, two raters (§7.4, §11.2) ----------------------------------


def square_contour(
    grid: Any, plane: float, rows: tuple[int, int], columns: tuple[int, int]
) -> np.ndarray:
    """A square on *plane* whose edges sit on voxel edges, around the voxels
    of *rows* and *columns* (inclusive), in patient coordinates."""
    (r0, r1), (c0, c1) = rows, columns
    corners = [
        (r0 - 0.5, c0 - 0.5),
        (r0 - 0.5, c1 + 0.5),
        (r1 + 0.5, c1 + 0.5),
        (r1 + 0.5, c0 - 0.5),
    ]
    return grid.index_to_world(np.array([[plane, r, c] for r, c in corners], float))


def write_structure_set(
    out: Path,
    rois: dict[int, tuple[str, list[tuple[str | None, np.ndarray]]]],
    frames: list[str],
    *,
    roi_frames: dict[int, str | None] | None = None,
) -> Path:
    """An RTSTRUCT of *rois* --- ``{number: (name, [(type, points), ...])}``,
    a type of ``None`` stating none --- listing *frames*.  Each ROI states the
    frame *roi_frames* gives it (``None``: none), else the first listed."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        RTStructureSetStorage,
        generate_uid,
    )

    structure = Dataset()
    structure.file_meta = FileMetaDataset()
    structure.file_meta.MediaStorageSOPClassUID = RTStructureSetStorage
    structure.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    structure.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    structure.SOPClassUID = RTStructureSetStorage
    structure.SOPInstanceUID = structure.file_meta.MediaStorageSOPInstanceUID
    structure.Modality = "RTSTRUCT"
    structure.StructureSetLabel = out.stem
    structure.ReferencedFrameOfReferenceSequence = []
    for uid in frames:
        item = Dataset()
        item.FrameOfReferenceUID = uid
        structure.ReferencedFrameOfReferenceSequence.append(item)
    structure.StructureSetROISequence = []
    structure.ROIContourSequence = []
    for number, (name, contours) in rois.items():
        roi = Dataset()
        roi.ROINumber = number
        roi.ROIName = name
        frame = (roi_frames or {}).get(number, frames[0] if frames else None)
        if frame:
            roi.ReferencedFrameOfReferenceUID = frame
        structure.StructureSetROISequence.append(roi)
        item = Dataset()
        item.ReferencedROINumber = number
        item.ContourSequence = []
        for kind, points in contours:
            contour = Dataset()
            if kind is not None:
                contour.ContourGeometricType = kind
            contour.NumberOfContourPoints = len(points)
            contour.ContourData = [float(v) for v in np.asarray(points).reshape(-1)]
            item.ContourSequence.append(contour)
        structure.ROIContourSequence.append(item)
    structure.save_as(str(out), enforce_file_format=True)
    return out


def lesion(z: int, y: int, x: int, r: int) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[z - r : z + r, y - r : y + r, x - r : x + r] = True
    return mask


def write_series(
    path: Path,
    label_set,
    *,
    follow_up: list[InstanceInput] | None = None,
    annotated_tp1: list[int] | str = "all_given",
) -> Path:
    """Two visits: lesion 7 grows, lesion 8 vanishes, lesion 9 appears."""
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0", days_from_baseline=0)
        w.add_timepoint("tp1", days_from_baseline=90)
        w.label_set(label_set)
        for tp, frame in (("tp0", "f0"), ("tp1", "f1")):
            w.add_grid(
                f"g_{tp}",
                shape=SHAPE,
                spacing=(2.0, 1.0, 1.0),
                timepoint=tp,
                frame_uid=f"pseudo:{frame}",
            )
            w.add_image(
                f"CT_{tp}",
                np.zeros(SHAPE, dtype=np.int16),
                grid=f"g_{tp}",
                modality="CT",
            )
        w.add_segmentation(
            "les_tp0",
            grid="g_tp0",
            instances=[
                InstanceInput(class_id=3, instance_id=7, mask=lesion(8, 8, 8, 2)),
                InstanceInput(class_id=3, instance_id=8, mask=lesion(8, 16, 16, 1)),
            ],
            annotated_classes=[3],
        )
        w.add_segmentation(
            "les_tp1",
            grid="g_tp1",
            instances=follow_up
            or [
                InstanceInput(class_id=3, instance_id=7, mask=lesion(8, 8, 8, 3)),
                InstanceInput(class_id=3, instance_id=9, mask=lesion(9, 4, 18, 1)),
            ],
            annotated_classes=annotated_tp1,
        )
        w.add_transform(
            "tp0_to_tp1",
            kind="affine",
            matrix=np.eye(4),
            from_frame="pseudo:f0",
            to_frame="pseudo:f1",
        )
    return path


def write_raters(
    path: Path,
    label_set,
    *,
    second: list[InstanceInput] | None = None,
    annotated_second: list[int] | str = "all_given",
) -> Path:
    """Two raters' lesions on one grid: what an agreement is measured between."""
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.label_set(label_set)
        w.add_grid("g", shape=SHAPE, spacing=(2.0, 1.0, 1.0), frame_uid="pseudo:f0")
        w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
        w.add_segmentation(
            "r1",
            grid="g",
            instances=[
                InstanceInput(class_id=3, instance_id=7, mask=lesion(8, 8, 8, 2)),
                InstanceInput(class_id=3, instance_id=8, mask=lesion(8, 16, 16, 1)),
            ],
            annotated_classes=[3],
        )
        w.add_segmentation(
            "r2",
            grid="g",
            instances=second
            or [
                InstanceInput(class_id=3, instance_id=7, mask=lesion(8, 8, 8, 3)),
                InstanceInput(class_id=3, instance_id=9, mask=lesion(9, 4, 18, 1)),
            ],
            annotated_classes=annotated_second,
        )
    return path


# -- h5py, for planting defects --------------------------------------------


def str_dtype() -> Any:
    """The variable-length UTF-8 string dtype used for every string in a file."""
    return h5py.string_dtype(encoding="utf-8")


def as_str(value: Any) -> str:
    """Normalise an HDF5 string attribute to :class:`str`."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return as_str(value[()])
    return str(value)


def encode_attr(value: Any) -> Any:
    """Encode a Python value for ``obj.attrs[...]`` following spec §2.5."""
    if isinstance(value, str):
        return np.array(value, dtype=str_dtype())
    if isinstance(value, (bool, np.bool_)):
        return np.bool_(value)
    if isinstance(value, (int, np.integer)):
        return np.int64(value)
    if isinstance(value, (float, np.floating)):
        return np.float64(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, np.bytes_)):
        return value
    if isinstance(value, Sequence):
        seq = list(value)
        if not seq:
            return np.empty((0,), dtype=np.int64)
        if all(isinstance(v, str) for v in seq):
            return np.array(seq, dtype=str_dtype())
        if all(isinstance(v, (bool, np.bool_)) for v in seq):
            return np.array(seq, dtype=np.bool_)
        if all(isinstance(v, (int, np.integer)) for v in seq):
            return np.array(seq, dtype=np.int64)
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in seq):
            return np.array(seq, dtype=np.float64)
        return np.asarray(seq)
    raise TypeError(f"cannot encode attribute value of type {type(value)!r}")


def group_contents(path: Path, name: str) -> dict[str, Any]:
    """Every attribute, and every dataset's type, chunks and bytes, below
    group *name* of the file at *path*: what two writes of one thing hold
    alike, read by h5py rather than by the engine that wrote them."""
    out: dict[str, Any] = {}

    def attrs(obj: Any) -> dict[str, Any]:
        return {k: np.asarray(v).tolist() for k, v in obj.attrs.items()}

    def visit(key: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            stored = np.asarray(obj[()])
            held = stored.tolist() if stored.dtype.kind == "O" else stored.tobytes()
            out[key] = (obj.dtype.str, obj.shape, obj.chunks, attrs(obj), held)
        else:
            out[key] = attrs(obj)

    with h5py.File(path, "r") as handle:
        group = handle[name]
        out[""] = attrs(group)
        group.visititems(visit)
    return out


def attested_as_written(path: Path) -> None:
    """Write the `content_id` that *path* now recomputes to.

    A fixture planted with h5py stands for a file some writer wrote that way,
    whose root it attested; left stale, it is a file edited after it was
    attested, which no amendment carries forward (§14.4, E702).
    """
    with medh5.open(path) as sample:
        content_id = sample.compute_content_id()
    with h5py.File(path, "r+") as handle:
        handle.attrs["content_id"] = encode_attr(content_id)


def crash_save_as(monkeypatch: Any, before: bytes) -> list[Path]:
    """Make pydicom's ``save_as`` stop half way through the file it is given
    --- half of *before*, then an ``OSError`` --- as a full disk or a killed
    process would; the files it was asked to write, in order."""
    from pydicom.dataset import Dataset

    asked: list[Path] = []

    def crashed(self: Any, filename: Any, *args: Any, **kwargs: Any) -> None:
        asked.append(Path(filename))
        Path(filename).write_bytes(before[: len(before) // 2])
        raise OSError("interrupted")

    monkeypatch.setattr(Dataset, "save_as", crashed)
    return asked


def disk_full_while(monkeypatch, owner: object, name: str) -> None:
    """Make *owner*'s *name* run on a disk with room for 8 bytes a file: the
    write fails part way through, as on a full disk (POSIX only)."""
    import resource
    import signal

    real = getattr(owner, name)

    def limited(*args, **kwargs):
        limit = resource.getrlimit(resource.RLIMIT_FSIZE)
        handler = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (8, limit[1]))
        try:
            return real(*args, **kwargs)
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, limit)
            signal.signal(signal.SIGXFSZ, handler)

    monkeypatch.setattr(owner, name, limited)
