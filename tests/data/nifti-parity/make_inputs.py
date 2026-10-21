"""Build the NIfTI parity corpus's inputs and its case table.

Run once, under the released medh5 2.1.0 (``pip install "medh5[nifti]==2.1.0"
nibabel==5.4.2 numpy==2.5.4 h5py`` in a virtual environment of its own), from
the repository root::

    python tests/data/nifti-parity/make_inputs.py

It writes ``inputs/`` and ``cases.json`` beside itself; ``record.py`` then runs
every case through 2.1.0's converter and writes ``records/``.  The inputs are
committed, so this script documents how they were made rather than being run
by any test.

The NIfTI files are written byte by byte here --- the header from nibabel's
header classes, which only lay out the fields, and the voxels after it ---
because ``nibabel.save`` decides some fields itself (it rewrites
``scl_slope``/``scl_inter`` from its own scaling, and picks the codes), and the
corpus needs every field to be exactly what the case says.  gzip is written
with an mtime of 0, so the bytes are the same on every run.

The samples under ``inputs/samples/`` are written by 2.1.0's writer: they are
what the export and ``import_seg_nifti`` cases start from.
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
SAMPLES = INPUTS / "samples"

# A 3-D volume is (x, y, z) = (4, 3, 2): small, and every axis a different
# extent, so a transposition that goes wrong changes the shape.
SHAPE = (4, 3, 2)
AFFINE = np.array(
    [[0.8, 0.0, 0.0, -10.0], [0.0, 0.9, 0.0, -20.0], [0.0, 0.0, 2.0, 5.0], [0, 0, 0, 1]]
)


def rotation(axis: int, degrees: float) -> np.ndarray:
    """A 3x3 rotation about world *axis*."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    i, j = [a for a in range(3) if a != axis]
    out = np.eye(3)
    out[i, i], out[i, j], out[j, i], out[j, j] = c, -s, s, c
    return out


def oblique(flip: bool = False) -> np.ndarray:
    """An affine rotated about two axes, its third column flipped on *flip*
    (a left-handed index frame: the qform's qfac of -1)."""
    rotated = rotation(2, 30.0) @ rotation(0, 15.0)
    zooms = np.array([0.8, 0.9, -2.0 if flip else 2.0])
    out = np.eye(4)
    out[:3, :3] = rotated * zooms
    out[:3, 3] = [-10.0, -20.0, 5.0]
    return out


def volume(dtype: Any = np.int16, shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    """``arange`` in *dtype*, in NIfTI (x, y, z, ...) order."""
    return np.arange(int(np.prod(shape))).reshape(shape).astype(dtype)


def write(
    name: str,
    data: np.ndarray,
    *,
    affine: np.ndarray | None = AFFINE,
    sform: int = 1,
    qform: int = 1,
    qform_affine: np.ndarray | None = None,
    xyz: str = "mm",
    t: str = "unknown",
    pixdim4: float | None = None,
    toffset: float = 0.0,
    slope: float = float("nan"),
    inter: float = 0.0,
    intent: int = 0,
    descrip: str = "",
    endian: str = "<",
    version: int = 1,
    dtype: Any = None,
    datatype: int | None = None,
    extension: bytes | None = None,
    edit: Any = None,
    tail: bytes = b"",
) -> list[Path]:
    """Write *data* as the NIfTI file ``inputs/<name>``; the files written.

    The suffix chooses the layout: ``.nii`` and ``.nii.gz`` hold the header
    and voxels in one file; ``.hdr`` writes the pair ``.hdr``/``.img``.
    *edit* is given the header after every field is set, to break it.
    """
    pair = name.endswith(".hdr")
    if version == 1:
        cls = nib.nifti1.Nifti1PairHeader if pair else nib.Nifti1Header
    else:
        cls = nib.nifti2.Nifti2PairHeader if pair else nib.Nifti2Header
    header = cls(endianness=endian)
    header.set_data_shape(data.shape)
    header.set_data_dtype(dtype if dtype is not None else data.dtype)
    if affine is not None:
        zooms = list(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))
        if data.ndim > 3:
            zooms += [1.0] * (data.ndim - 3)
        header.set_zooms(zooms[: data.ndim])
        if qform:
            header.set_qform(
                affine if qform_affine is None else qform_affine, code=qform
            )
        if sform:
            header.set_sform(affine, code=sform)
    header.set_xyzt_units(xyz=xyz, t=t)
    if pixdim4 is not None:
        header["pixdim"][4] = pixdim4
    header["toffset"] = toffset
    header["scl_slope"] = slope
    header["scl_inter"] = inter
    header["intent_code"] = intent
    header["descrip"] = descrip.encode()
    extra = b"\0\0\0\0" if extension is None else extension
    if not pair:
        header["vox_offset"] = len(header.binaryblock) + len(extra)
    if edit is not None:
        edit(header)
    voxels = np.asarray(data, dtype=header.get_data_dtype()).tobytes(order="F")
    if datatype is not None:
        header["datatype"] = datatype
    target = INPUTS / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if pair:
        image = target.with_suffix(".img")
        target.write_bytes(header.binaryblock)
        image.write_bytes(voxels + tail)
        return [target, image]
    raw = header.binaryblock + extra + voxels + tail
    if name.endswith(".gz"):
        target.write_bytes(gzip.compress(raw, mtime=0))
    else:
        target.write_bytes(raw)
    return [target]


def text(name: str, content: str) -> None:
    path = INPUTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # Every text file ends in a newline, as the repository's hooks want.
    path.write_text(content if content.endswith("\n") else content + "\n", "utf-8")


def make_niftis() -> None:
    """Every NIfTI input, grouped as the cases use them."""
    # Layouts and byte orders.
    write("le-int16.nii", volume())
    write("le-int16.nii.gz", volume())
    write("be-int16.nii", volume(), endian=">")
    write("be-float32.nii.gz", volume(np.float32) / 4, endian=">")
    write("n2-int16.nii", volume(), version=2)
    write("n2-int16.nii.gz", volume(), version=2)
    write("n2-be-float64.nii", volume(np.float64) / 8, version=2, endian=">")
    write("pair-int16.hdr", volume())
    write("pair-n2-int16.hdr", volume(), version=2)
    write("pair-be-int16.hdr", volume(), endian=">")
    write(
        "extension.nii",
        volume(),
        extension=b"\x01\0\0\0"
        + (16).to_bytes(4, "little")
        + (6).to_bytes(4, "little")
        + b"comment\0",
    )
    write("descrip.nii.gz", volume(), descrip="parity corpus")
    # Element types.
    for kind in (
        "uint8", "int8", "uint16", "int32", "uint32", "int64", "uint64",
        "float32", "float64", "complex64",
    ):  # fmt: skip
        write(f"dtype-{kind}.nii.gz", volume(np.dtype(kind)))
    rgb = np.zeros(SHAPE, dtype=[("R", "u1"), ("G", "u1"), ("B", "u1")])
    rgb["R"] = volume(np.uint8)
    write("dtype-rgb24.nii.gz", rgb)
    # Scaling (§4.2).
    write("scaled-int16.nii.gz", volume(), slope=2.0, inter=-1024.0)
    write("scaled-float32.nii.gz", volume(np.float32), slope=0.5, inter=1.0)
    write("scale-zero.nii.gz", volume(), slope=0.0, inter=5.0)
    write("scale-identity.nii.gz", volume(), slope=1.0, inter=0.0)
    write("scale-inter-only.nii.gz", volume(), slope=1.0, inter=5.0)
    mask = np.zeros(SHAPE, np.int16)
    mask[:2] = 1  # physical 0 under intercept -1
    mask[2:] = 2  # physical 1
    write("mask-scaled.nii.gz", mask, slope=1.0, inter=-1.0)
    # The two mappings.
    write("sform-only.nii.gz", volume(), qform=0)
    write("qform-only.nii.gz", volume(), sform=0)
    write("qform-oblique.nii.gz", volume(), affine=oblique(), sform=0)
    write("qform-flipped.nii.gz", volume(), affine=oblique(flip=True), sform=0)
    write("sform-oblique.nii.gz", volume(), affine=oblique(), qform=0)
    write("both-oblique.nii.gz", volume(), affine=oblique(flip=True))
    moved = AFFINE.copy()
    moved[:3, 3] += [3.0, 0.0, 0.0]
    write("both-disagree.nii.gz", volume(), qform_affine=moved)
    write("sform-code4.nii.gz", volume(), sform=4, qform=0)
    write("no-mapping.nii.gz", volume(), sform=0, qform=0)

    def half_turn(header: Any) -> None:
        """b² + c² + d² = 1 to the last bit: a = 0, a half turn."""
        header["quatern_b"], header["quatern_c"], header["quatern_d"] = 0.0, 0.0, 1.0

    write("qform-half-turn.nii.gz", volume(), sform=0, edit=half_turn)

    def no_rotation(header: Any) -> None:
        header["quatern_b"], header["quatern_c"], header["quatern_d"] = 0.8, 0.8, 0.8

    write("qform-not-rotation.nii.gz", volume(), sform=0, edit=no_rotation)

    def negative_size(header: Any) -> None:
        header["pixdim"][2] = -0.9

    write("qform-negative-pixdim.nii.gz", volume(), sform=0, edit=negative_size)
    # Units (§3.5).
    metres = AFFINE.copy()
    metres[:3, :] /= 1000.0
    metres[3, 3] = 1.0
    write("units-meter.nii.gz", volume(), affine=metres, xyz="meter")
    microns = AFFINE.copy()
    microns[:3, :] *= 1000.0
    microns[3, 3] = 1.0
    write("units-micron.nii.gz", volume(), affine=microns, xyz="micron")
    write("units-unknown.nii.gz", volume(), xyz="unknown")
    # Masks on the le-int16 grid.
    liver = np.zeros(SHAPE, np.uint8)
    liver[:2, :, :] = 1
    lesion = np.zeros(SHAPE, np.uint8)
    lesion[1, 1, :] = 1
    write("mask-liver.nii.gz", liver)
    write("mask-lesion.nii.gz", lesion)
    write("mask-spleen.nii.gz", np.zeros(SHAPE, np.uint8))
    write("mask-float.nii.gz", liver.astype(np.float32) * 0.5)
    write("mask-shape.nii.gz", np.zeros((4, 3, 3), np.uint8))
    shifted = AFFINE.copy()
    shifted[:3, 3] += [0.0, 0.0, 10.0]
    write("mask-origin.nii.gz", liver, affine=shifted)
    write("mask-spacing.nii.gz", liver, affine=np.diag([0.8, 0.9, 2.5, 1.0]) + np.array(
        [[0, 0, 0, -10.0], [0, 0, 0, -20.0], [0, 0, 0, 5.0], [0, 0, 0, 0]]
    ))  # fmt: skip
    mirrored = AFFINE.copy()
    mirrored[0, 0] = -0.8
    write("mask-direction.nii.gz", liver, affine=mirrored)
    write("mask-metres.nii.gz", liver, affine=metres, xyz="meter")
    write("mask-no-mapping.nii.gz", liver, sform=0, qform=0)
    # Grids that disagree with le-int16.
    write("other-shape.nii.gz", volume(shape=(4, 3, 3)))
    write("other-origin.nii.gz", volume(), affine=shifted)
    write("other-direction.nii.gz", volume(), affine=mirrored)
    # Four and five dimensions (§3.6).
    series = volume(shape=(*SHAPE, 3))
    write("cine-sec.nii.gz", series, t="sec", pixdim4=0.5, toffset=2.0)
    write("cine-msec.nii.gz", series, t="msec", pixdim4=500.0, toffset=2000.0)
    write("cine-usec.nii.gz", series, t="usec", pixdim4=500.0, toffset=1000.0)
    write("cine-sec-other.nii.gz", series, t="sec", pixdim4=4.0, toffset=1.0)
    write("cine-unitless.nii.gz", series, pixdim4=2.0)
    write("cine-no-step.nii.gz", series, t="sec", pixdim4=0.0)
    write("intent-time.nii.gz", series, intent=2001)
    write("intent-vector.nii.gz", series, intent=1007)
    write("intent-rgb.nii.gz", series, intent=2003)
    write("dwi.nii.gz", series)
    text("dwi.bval", "0 1000 1000\n")
    write("dwi-other.nii.gz", series)
    text("dwi-other.bval", "0\n500\n2000\n")
    write("multi-echo.nii.gz", series)
    text("multi-echo.json", json.dumps({"EchoTime": [0.01, 0.02, 0.03]}))
    write("echo-numbers.nii.gz", series)
    text("echo-numbers.json", json.dumps({"EchoNumber": [1, 2, 3]}))
    write("echo-scalar.nii.gz", series)
    text("echo-scalar.json", json.dumps({"EchoTime": 0.01}))
    write("echo-short.nii.gz", series)
    text("echo-short.json", json.dumps({"EchoTime": [0.01, 0.02]}))
    write("echo-words.nii.gz", series)
    text("echo-words.json", json.dumps({"EchoTime": ["a", "b", "c"]}))
    write("volume-timing.nii.gz", series, t="sec", pixdim4=1.0)
    text("volume-timing.json", json.dumps({"VolumeTiming": [0.0, 1.5, 4.0]}))
    write("both-kinds.nii.gz", series)
    text(
        "both-kinds.json",
        json.dumps({"EchoTime": [0.01, 0.02, 0.03], "VolumeTiming": [0, 1, 2]}),
    )
    write("broken-sidecar.nii.gz", series)
    text("broken-sidecar.json", "{not json")
    write("squeeze.nii.gz", volume(shape=(*SHAPE, 1)))
    write("vector5.nii.gz", volume(shape=(*SHAPE, 1, 3)), t="sec", pixdim4=2.0)
    write("five.nii.gz", volume(shape=(*SHAPE, 2, 3)))
    # Planes (§3.6).
    plane = AFFINE.copy()
    plane[2, 2], plane[2, 3] = 1.0, 0.0
    write("plane.nii.gz", volume(shape=(4, 3)), affine=plane)
    off = plane.copy()
    off[2, 3] = 50.0
    write("plane-z50.nii.gz", volume(shape=(4, 3)), affine=off)
    tilted = np.eye(4)
    tilted[:3, :3] = rotation(0, 20.0) * [0.8, 0.9, 1.0]
    write("plane-tilted.nii.gz", volume(shape=(4, 3)), affine=tilted)
    flat = plane.copy()
    flat[:3, 2] = 0.0
    write("plane-flat.nii.gz", volume(shape=(4, 3)), affine=flat, qform=0)
    sheared = plane.copy()
    sheared[0, 2] = 0.5
    write("plane-sheared.nii.gz", volume(shape=(4, 3)), affine=sheared, qform=0)
    write("plane-mask.nii.gz", (volume(shape=(4, 3)) % 2).astype(np.uint8), affine=off)
    write("slice.nii.gz", volume(shape=(4, 3, 1)), affine=off)

    # What is not NIfTI, or is broken.
    def bad_magic(header: Any) -> None:
        header["magic"] = b"xx1"

    write("bad-magic.nii", volume(), edit=bad_magic)

    def bad_size(header: Any) -> None:
        header["sizeof_hdr"] = 999

    write("bad-sizeof.nii", volume(), edit=bad_size)

    def too_many_dims(header: Any) -> None:
        header["dim"][0] = 8

    write("dim0-8.nii", volume(), edit=too_many_dims)

    def negative_dim(header: Any) -> None:
        header["dim"][2] = -3

    write("negative-dim.nii", volume(), edit=negative_dim)

    def no_offset(header: Any) -> None:
        header["vox_offset"] = 0

    write("vox-offset-0.nii", volume(), edit=no_offset)

    def far_offset(header: Any) -> None:
        header["vox_offset"] = 4096

    write("vox-offset-far.nii", volume(), edit=far_offset)
    write("bad-datatype.nii", volume(), datatype=77)
    whole = (INPUTS / "le-int16.nii").read_bytes()
    (INPUTS / "short-header.nii").write_bytes(whole[:100])
    (INPUTS / "truncated-data.nii").write_bytes(whole[: 352 + 20])
    (INPUTS / "empty.nii").write_bytes(b"")
    (INPUTS / "random.nii").write_bytes(bytes(range(7, 256, 3)) * 2)
    packed = (INPUTS / "le-int16.nii.gz").read_bytes()
    (INPUTS / "truncated.nii.gz").write_bytes(packed[: len(packed) // 2])
    (INPUTS / "not-gzip.nii.gz").write_bytes(whole)
    write("trailing.nii.gz", volume(), tail=bytes(1 << 20))
    (INPUTS / "pair-missing-img.hdr").write_bytes(
        (INPUTS / "pair-int16.hdr").read_bytes()
    )
    write("pair-short.hdr", volume())
    image = INPUTS / "pair-short.img"
    image.write_bytes(image.read_bytes()[:10])
    text("scan.txt", "not an image\n")
    nib.save(nib.MGHImage(volume(np.float32), AFFINE), str(INPUTS / "not-nifti.mgh"))
    text("labels.json", json.dumps(LABELS))
    text("labels-small.json", json.dumps(LABELS_SMALL))
    make_declined()


def make_declined() -> None:
    """Inputs added in 2.2.1 for the reviews declined in 2.2.0 to keep 2.1.0's
    behaviour (P22-03): each pins what 2.1.0 did, not what a reader might
    expect.  Added later, they leave every earlier input as it was."""
    # An extension whose size runs past `vox_offset`, to the end of the
    # file: nibabel reads it whatever the offset says, and loads the volume.
    past = (
        b"\x01\0\0\0"
        + (64).to_bytes(4, "little")
        + (6).to_bytes(4, "little")
        + b"comment\0"
    )
    for name in ("extension-past-offset.nii", "extension-past-offset.nii.gz"):
        write(name, volume(), extension=past)

    # An infinite `vox_offset` with the extension flag set: nibabel reads
    # extensions to the end of the stream, and its refusal depends on where
    # the stream ends, so nothing could refuse it sooner with 2.1.0's text.
    def infinite(name: str, body: bytes) -> None:
        header = nib.Nifti1Header()
        header.set_data_shape(SHAPE)
        header.set_data_dtype(np.int16)
        header["vox_offset"] = np.inf
        raw = header.binaryblock + b"\x01\0\0\0" + body
        (INPUTS / name).write_bytes(gzip.compress(raw, mtime=0))

    one = (16).to_bytes(4, "little") + (6).to_bytes(4, "little") + b"comment\0"
    infinite("offset-inf-extensions.nii.gz", one * 1000)
    infinite("offset-inf-cut.nii.gz", one * 1000 + one[:12])
    infinite("offset-inf-stray.nii.gz", one * 3 + b"xyz")

    # `xyzt_units` with a reserved bit: nibabel takes the space code as the
    # value modulo 8 and the time code as the rest, so a reserved bit leaves
    # the time unit unknown, and the frames assumed.
    series = volume(shape=(*SHAPE, 3))
    for units in (0x4A, 0x4B, 0x52, 0x8A, 0xCA):

        def reserved(header: Any, units: int = units) -> None:
            header["xyzt_units"] = units

        write(f"cine-units-{units:#04x}.nii.gz", series, pixdim4=2.5, edit=reserved)


LABELS = {
    "id": "organs",
    "version": "1.0.0",
    "classes": [
        {"id": 1, "key": "liver", "name": "Liver"},
        {"id": 2, "key": "lesion", "name": "Lesion"},
        {"id": 3, "key": "spleen", "name": "Spleen"},
        {"id": 5, "key": "gtv_1", "name": "GTV-1"},
    ],
}
LABELS_SMALL = {
    "id": "small",
    "version": "1.0.0",
    "classes": [{"id": 1, "key": "liver", "name": "Liver"}],
}


def make_samples() -> None:
    """The samples the export and ``import_seg_nifti`` cases start from,
    written by 2.1.0's writer."""
    import medh5
    from medh5.labels import LabelClass, LabelSet

    SAMPLES.mkdir(parents=True, exist_ok=True)
    zyx = (2, 3, 4)
    ct = np.arange(24, dtype=np.int16).reshape(zyx) * 10
    liver = np.zeros(zyx, bool)
    liver[:, :, :2] = True
    lesion = np.zeros(zyx, bool)
    lesion[:, 1, 3] = True
    organs = LabelSet(
        "organs",
        [LabelClass(1, "liver", "Liver"), LabelClass(2, "lesion", "Lesion")],
    )

    def grid(w: Any, name: str = "g", **fields: Any) -> None:
        fields.setdefault("shape", zyx)
        fields.setdefault("spacing", (2.0, 0.9, 0.8))
        # le-int16.nii's grid, read into LPS and (z, y, x).
        fields.setdefault("origin", (10.0, 20.0, 5.0))
        fields.setdefault("direction", [[0, 0, -1.0], [0, -1.0, 0], [1.0, 0, 0]])
        fields.setdefault("timepoint", "tp0")
        w.add_grid(name, **fields)

    def create(name: str) -> Any:
        return medh5.create(SAMPLES / f"{name}.medh5", sample_id=name, codec="portable")

    with create("ct") as w:
        grid(w)
        w.label_set(organs)
        w.add_image(
            "CT", ct, grid="g", modality="CT", value_type="quantitative",
            value_units="HU", rescale_slope=1.0, rescale_intercept=-1024.0,
        )  # fmt: skip
        w.add_segmentation(
            "seg", grid="g", masks={1: liver, 2: lesion}, annotated_classes="all"
        )
        w.add_segmentation(
            "partial", grid="g", masks={1: liver, 2: np.zeros(zyx, bool)},
            annotated_classes=[1],
        )  # fmt: skip
        w.add_segmentation(
            "ignored", grid="g", masks={1: liver}, ignore=lesion,
            annotated_classes="all",
        )  # fmt: skip
    with create("plain") as w:
        grid(w)
        w.add_image("CT", ct, grid="g", modality="CT")
    with create("ras") as w:
        grid(w, coord_system="RAS")
        w.add_image("CT", ct, grid="g", modality="CT")
    turned = np.array(
        [[0.9659258262890683, 0.0, 0.25881904510252074],
         [0.0, 1.0, 0.0],
         [-0.25881904510252074, 0.0, 0.9659258262890683]]
    )  # fmt: skip
    with create("oblique") as w:
        grid(w, direction=turned.tolist())
        w.add_image("CT", ct, grid="g", modality="CT")
    for name, units, scale in (("metres", "m", 1e-3), ("microns", "um", 1e3),
                               ("pixels", "px", 1.0)):  # fmt: skip
        with create(name) as w:
            grid(
                w, units=units, spacing=tuple(v * scale for v in (2.0, 0.9, 0.8)),
                origin=tuple(v * scale for v in (10.0, 20.0, 5.0)),
            )  # fmt: skip
            w.add_image("CT", ct, grid="g", modality="CT")
    for name, dtype in (("float32", np.float32), ("float64", np.float64),
                        ("uint8", np.uint8), ("uint16", np.uint16),
                        ("int64", np.int64), ("bool", np.bool_)):  # fmt: skip
        with create(f"dtype-{name}") as w:
            grid(w)
            w.add_image("IM", (ct % 2).astype(dtype) if dtype is np.bool_
                        else ct.astype(dtype), grid="g", modality="OT")  # fmt: skip
    series = np.arange(4 * 24, dtype=np.int16).reshape((4, *zyx))
    for name, times, unit in (("dce-even", [10.0, 12.5, 15.0, 17.5], "s"),
                              ("dce-uneven", [0.0, 1.0, 3.0, 7.0], "s"),
                              ("dce-ms", [0.0, 250.0, 500.0, 750.0], "ms"),
                              ("dce-one", [3.0], "s")):  # fmt: skip
        frames = series[: len(times)]
        with create(name) as w:
            grid(
                w, shape=frames.shape, axis_names=("t", "z", "y", "x"),
                axis_kinds=("time", "spatial", "spatial", "spatial"),
                time_values=times, time_units=unit,
            )  # fmt: skip
            w.add_image("DCE", frames, grid="g", modality="MR")
    channels = series[:2]
    for name, acquisition in (("dwi", {"b_values": [0.0, 1000.0]}),
                              ("echo", {"EchoTime": [0.005, 0.01],
                                        "ImageType": ["ORIGINAL", "PRIMARY"]}),
                              ("echo-all", {"EchoTime": [0.005, 0.01],
                                            "EchoNumbers": [1, 2],
                                            "InversionTime": [0.1, 0.2],
                                            "FlipAngle": [10, 20]}),
                              ("vector", {})):  # fmt: skip
        with create(name) as w:
            grid(
                w, shape=channels.shape, axis_names=("c", "z", "y", "x"),
                axis_kinds=("channel", "spatial", "spatial", "spatial"),
            )  # fmt: skip
            w.add_image("MR", channels, grid="g", modality="MR")
            if acquisition:
                w.acquisition("MR", **acquisition)
    with create("xray") as w:
        w.add_grid(
            "g", shape=(3, 4), spacing=(0.9, 0.8), origin=(20.0, 10.0),
            timepoint="tp0",
        )  # fmt: skip
        w.add_image("XR", np.arange(12, dtype=np.uint16).reshape(3, 4), grid="g",
                    modality="DX")  # fmt: skip
        w.add_segmentation(
            "seg", grid="g", masks={1: np.arange(12).reshape(3, 4) % 3 == 0}
        )
    with create("xray-cine") as w:
        w.add_grid(
            "g", shape=(2, 3, 4), spacing=(0.9, 0.8), origin=(20.0, 10.0),
            axis_names=("t", "y", "x"), axis_kinds=("time", "spatial", "spatial"),
            time_values=[0.0, 1.0], time_units="s", timepoint="tp0",
        )  # fmt: skip
        w.add_image("XR", np.zeros((2, 3, 4), np.uint16), grid="g", modality="XA")
    with create("pet") as w:
        grid(w)
        grid(w, "pet", shape=(1, 2, 2), spacing=(4.0, 1.6, 1.6),
             origin=(30.0, 40.0, 50.0))  # fmt: skip
        w.add_image("CT", ct, grid="g", modality="CT")
        w.add_image("PET", np.ones((1, 2, 2), np.float32), grid="pet", modality="PT")
        w.add_segmentation(
            "petseg", grid="pet", masks={1: np.array([[[True, False], [False, True]]])}
        )
    with create("one-slice") as w:
        # plane-mask.nii.gz read as (x, y, 1).
        grid(w, shape=(1, 3, 4), spacing=(1.0, 0.9, 0.8), origin=(10.0, 20.0, 50.0))
        w.add_image("CT", ct[:1], grid="g", modality="CT")


BESIDE = {
    "object.json": json.dumps({"Manufacturer": "x", "EchoTime": 0.01}),
    "timing.json": json.dumps({"Manufacturer": "x", "VolumeTiming": [0, 1, 2, 3]}),
    "list.json": "[]",
    "broken.json": "{not json",
    "old.bval": "0 1000\n",
}


def make_beside() -> None:
    """Files an export finds beside its target, from an earlier export."""
    for name, content in BESIDE.items():
        text(f"beside/{name}", content)


# --------------------------------------------------------------------------
# The case table.  Paths are relative to this directory; ``out`` names are
# relative to the case's output directory.


def imports() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def case(id: str, images: Any, *, masks: Any = (), out: str | None = None,
             **options: Any) -> None:  # fmt: skip
        if isinstance(images, str):
            images = [["image", images]]
        cases.append({
            "id": id, "call": "from_nifti",
            "images": [[n, f"inputs/{p}"] for n, p in images],
            "masks": [[n, f"inputs/{p}"] for n, p in masks],
            "out": out or f"{id}.medh5", "options": options,
        })  # fmt: skip

    for name in (
        "le-int16.nii", "le-int16.nii.gz", "be-int16.nii", "be-float32.nii.gz",
        "n2-int16.nii", "n2-int16.nii.gz", "n2-be-float64.nii", "pair-int16.hdr",
        "pair-int16.img", "pair-n2-int16.hdr", "pair-be-int16.hdr", "extension.nii",
        "descrip.nii.gz",
        "dtype-uint8.nii.gz", "dtype-int8.nii.gz", "dtype-uint16.nii.gz",
        "dtype-int32.nii.gz", "dtype-uint32.nii.gz", "dtype-int64.nii.gz",
        "dtype-uint64.nii.gz", "dtype-float32.nii.gz", "dtype-float64.nii.gz",
        "dtype-complex64.nii.gz", "dtype-rgb24.nii.gz",
        "scaled-int16.nii.gz", "scaled-float32.nii.gz", "scale-zero.nii.gz",
        "scale-identity.nii.gz", "scale-inter-only.nii.gz",
        "sform-only.nii.gz", "qform-only.nii.gz", "qform-oblique.nii.gz",
        "qform-flipped.nii.gz", "sform-oblique.nii.gz", "both-oblique.nii.gz",
        "both-disagree.nii.gz", "sform-code4.nii.gz", "no-mapping.nii.gz",
        "qform-half-turn.nii.gz", "qform-not-rotation.nii.gz",
        "qform-negative-pixdim.nii.gz",
        "units-meter.nii.gz", "units-micron.nii.gz", "units-unknown.nii.gz",
        "cine-sec.nii.gz", "cine-msec.nii.gz", "cine-usec.nii.gz",
        "cine-unitless.nii.gz", "cine-no-step.nii.gz", "intent-time.nii.gz",
        "intent-vector.nii.gz", "intent-rgb.nii.gz", "dwi.nii.gz",
        "multi-echo.nii.gz", "echo-numbers.nii.gz", "echo-scalar.nii.gz",
        "echo-short.nii.gz", "echo-words.nii.gz", "volume-timing.nii.gz",
        "both-kinds.nii.gz", "broken-sidecar.nii.gz", "squeeze.nii.gz",
        "vector5.nii.gz", "five.nii.gz",
        "plane.nii.gz", "plane-z50.nii.gz", "plane-tilted.nii.gz",
        "plane-flat.nii.gz", "plane-sheared.nii.gz", "slice.nii.gz",
        "bad-magic.nii", "bad-sizeof.nii", "dim0-8.nii", "negative-dim.nii",
        "vox-offset-0.nii", "vox-offset-far.nii", "bad-datatype.nii",
        "short-header.nii", "truncated-data.nii", "empty.nii", "random.nii",
        "truncated.nii.gz", "not-gzip.nii.gz", "trailing.nii.gz",
        "pair-missing-img.hdr", "pair-short.hdr", "scan.txt", "not-nifti.mgh",
        "missing.nii",
        "extension-past-offset.nii", "extension-past-offset.nii.gz",
        "offset-inf-extensions.nii.gz", "offset-inf-cut.nii.gz",
        "offset-inf-stray.nii.gz",
        "cine-units-0x4a.nii.gz", "cine-units-0x4b.nii.gz",
        "cine-units-0x52.nii.gz", "cine-units-0x8a.nii.gz",
        "cine-units-0xca.nii.gz",
    ):  # fmt: skip
        stem = name.replace(".nii.gz", "").replace(".", "-")
        case(f"import-{stem}", [["image", name]])
    # The options, one at a time.
    case("import-no-mapping-assumed", "no-mapping.nii.gz", assume_geometry=True)
    case("import-ras", "le-int16.nii.gz", coord_system="RAS")
    case("import-unknown-system", "le-int16.nii.gz", coord_system="XYZ")
    case("import-untransposed", "le-int16.nii.gz", transpose=False)
    case("import-untransposed-4d", "cine-sec.nii.gz", transpose=False)
    case("import-untransposed-plane", "plane.nii.gz", transpose=False)
    case("import-fourth-time", "dwi.nii.gz", fourth_axis="time")
    case("import-fourth-channel", "cine-sec.nii.gz", fourth_axis="channel")
    case("import-fourth-unknown", "cine-sec.nii.gz", fourth_axis="depth")
    case("import-both-kinds-time", "both-kinds.nii.gz", fourth_axis="time")
    case("import-both-kinds-channel", "both-kinds.nii.gz", fourth_axis="channel")
    case("import-vector5-time", "vector5.nii.gz", fourth_axis="time")
    case("import-plane-z50-volume", "plane-z50.nii.gz", plane_as_volume=True)
    case("import-plane-tilted-volume", "plane-tilted.nii.gz", plane_as_volume=True)
    case("import-plane-flat-volume", "plane-flat.nii.gz", plane_as_volume=True)
    case("import-plane-sheared-volume", "plane-sheared.nii.gz", plane_as_volume=True)
    case("import-plane-volume", "plane.nii.gz", plane_as_volume=True)
    case("import-volume-plane-flag", "le-int16.nii.gz", plane_as_volume=True)
    case("import-codec-portable", "le-int16.nii.gz", codec="portable")
    case("import-codec-fast", "le-int16.nii.gz", codec="fast")
    case("import-codec-training", "le-int16.nii.gz", codec="training")
    case("import-identity", "le-int16.nii.gz", sample_id="case_01",
         subject_id="patient-7")  # fmt: skip
    case("import-sample-only", "le-int16.nii.gz", sample_id="case_02")
    case("import-subject-only", "le-int16.nii.gz", subject_id="patient-7")
    case("import-non-ascii", "le-int16.nii.gz", out="Müller Hans.medh5")
    case("import-modality", [["CT", "le-int16.nii.gz"]], modalities={"CT": "CT"},
         value_units={"CT": "HU"})  # fmt: skip
    case("import-empty", [])
    # Several volumes.
    two = [["CT", "le-int16.nii.gz"], ["PET", "dtype-float32.nii.gz"]]
    case("import-two", two)
    case("import-two-shape", [["CT", "le-int16.nii.gz"], ["B", "other-shape.nii.gz"]])
    case("import-two-origin", [["CT", "le-int16.nii.gz"], ["B", "other-origin.nii.gz"]])
    case("import-two-direction",
         [["CT", "le-int16.nii.gz"], ["B", "other-direction.nii.gz"]])  # fmt: skip
    case("import-two-metres", [["CT", "le-int16.nii.gz"], ["M", "units-meter.nii.gz"]])
    case("import-two-times", [["A", "cine-sec.nii.gz"], ["B", "cine-sec-other.nii.gz"]])
    case("import-two-ms", [["A", "cine-sec.nii.gz"], ["B", "cine-msec.nii.gz"]])
    case("import-two-lend",
         [["A", "cine-unitless.nii.gz"], ["B", "cine-sec.nii.gz"]])  # fmt: skip
    case("import-two-keep",
         [["A", "cine-sec.nii.gz"], ["B", "cine-unitless.nii.gz"]])  # fmt: skip
    case("import-two-kinds", [["DWI", "dwi.nii.gz"], ["CINE", "cine-sec.nii.gz"]])
    case("import-two-dwi", [["A", "dwi.nii.gz"], ["B", "dwi-other.nii.gz"]])
    planes = [["A", "plane-z50.nii.gz"], ["B", "slice.nii.gz"]]
    case("import-two-planes", planes, plane_as_volume=True)
    # Masks.
    ct = [["CT", "le-int16.nii.gz"]]
    organs = [["liver", "mask-liver.nii.gz"], ["lesion", "mask-lesion.nii.gz"]]
    case("masks", ct, masks=organs)
    case("masks-names", ct, masks=[["Liver", "mask-liver.nii.gz"],
                                   ["GTV-1", "mask-lesion.nii.gz"]])  # fmt: skip
    case("masks-collide", ct, masks=[["GTV 1", "mask-liver.nii.gz"],
                                     ["gtv-1", "mask-lesion.nii.gz"]])  # fmt: skip
    case("masks-scaled", ct, masks=[["liver", "mask-scaled.nii.gz"]])
    case("masks-float", ct, masks=[["liver", "mask-float.nii.gz"]])
    case("masks-empty-class", ct, masks=[["spleen", "mask-spleen.nii.gz"]])
    case("masks-annotated", ct, masks=organs, annotated_classes=["liver", "spleen"])
    case("masks-annotated-all", ct, masks=organs, annotated_classes="all")
    case("masks-label-set", ct, masks=organs, label_set="inputs/labels.json")
    case("masks-label-set-names", ct, masks=[["Liver", "mask-liver.nii.gz"],
                                             ["GTV-1", "mask-lesion.nii.gz"]],
         label_set="inputs/labels.json")  # fmt: skip
    case("masks-label-set-missing", ct, masks=organs,
         label_set="inputs/labels-small.json")  # fmt: skip
    case("masks-label-set-annotated", ct, masks=organs, label_set="inputs/labels.json",
         annotated_classes=["liver", "spleen"])  # fmt: skip
    case("masks-label-set-no-masks", ct, label_set="inputs/labels.json")
    case("masks-shape", ct, masks=[["liver", "mask-shape.nii.gz"]])
    case("masks-origin", ct, masks=[["liver", "mask-origin.nii.gz"]])
    case("masks-metres", ct, masks=[["liver", "mask-metres.nii.gz"]])
    case("masks-no-mapping", ct, masks=[["liver", "mask-no-mapping.nii.gz"]])
    case("masks-no-mapping-assumed", ct, masks=[["liver", "mask-no-mapping.nii.gz"]],
         assume_geometry=True)  # fmt: skip
    case("masks-plane", [["XR", "plane-z50.nii.gz"]],
         masks=[["bone", "plane-mask.nii.gz"]], plane_as_volume=True)  # fmt: skip
    case("masks-ras", ct, masks=organs, coord_system="RAS")
    return cases


def reads() -> list[dict[str, Any]]:
    """``read_nifti`` on its own, for the Python facade."""
    cases = []
    for path in sorted(INPUTS.iterdir()):
        if path.is_dir() or path.suffix in (".bval", ".json", ".img", ".txt", ".mgh"):
            continue
        cases.append({
            "id": f"read-{path.name}", "call": "read_nifti",
            "path": f"inputs/{path.name}", "options": {},
        })  # fmt: skip
    for options in ({"transpose": False}, {"coord_system": "RAS"},
                    {"assume_geometry": True}, {"plane_as_volume": True},
                    {"fourth_axis": "channel"}):  # fmt: skip
        tag = "-".join(f"{k}={v}" for k, v in options.items())
        for name in ("le-int16.nii.gz", "cine-sec.nii.gz", "plane-z50.nii.gz",
                     "no-mapping.nii.gz"):  # fmt: skip
            cases.append({
                "id": f"read-{name}-{tag}", "call": "read_nifti",
                "path": f"inputs/{name}", "options": options,
            })  # fmt: skip
    return cases


def exports() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def case(id: str, sample: str, image: str, out: str, *, beside: Any = (),
             **options: Any) -> None:  # fmt: skip
        cases.append({
            "id": id, "call": "to_nifti", "sample": f"inputs/samples/{sample}.medh5",
            "image": image, "out": out,
            "beside": {n: f"inputs/beside/{s}" if s else None for n, s in beside},
            "options": options,
        })  # fmt: skip

    for sample in ("plain", "ras", "oblique", "metres", "microns", "pixels",
                   "dtype-float32", "dtype-float64", "dtype-uint8", "dtype-uint16",
                   "dtype-int64", "dtype-bool", "one-slice"):  # fmt: skip
        image = "IM" if sample.startswith("dtype-") else "CT"
        case(f"export-{sample}", sample, image, f"{sample}.nii.gz")
    case("export-ct", "ct", "CT", "ct.nii.gz")
    case("export-ct-nii", "ct", "CT", "ct.nii")
    case("export-ct-stored", "ct", "CT", "ct.nii.gz", physical=False)
    case("export-labelmap", "ct", "CT", "seg.nii.gz", annotation="seg")
    case("export-class", "ct", "CT", "liver.nii.gz", annotation="seg",
         class_key="liver")  # fmt: skip
    case("export-class-id", "ct", "CT", "lesion.nii.gz", annotation="seg", class_key=2)
    case("export-class-unexamined", "ct", "CT", "x.nii.gz", annotation="partial",
         class_key="lesion")  # fmt: skip
    case("export-class-examined", "ct", "CT", "x.nii.gz", annotation="partial",
         class_key="liver")  # fmt: skip
    case("export-class-ignored", "ct", "CT", "x.nii.gz", annotation="ignored",
         class_key="liver")  # fmt: skip
    case("export-ignored-labelmap", "ct", "CT", "x.nii.gz", annotation="ignored")
    case("export-pet-mask", "pet", "CT", "m.nii.gz", annotation="petseg")
    case("export-pet", "pet", "PET", "pet.nii.gz")
    for sample, image in (("dce-even", "DCE"), ("dce-uneven", "DCE"),
                          ("dce-ms", "DCE"), ("dce-one", "DCE"), ("dwi", "MR"),
                          ("echo", "MR"), ("echo-all", "MR"), ("vector", "MR"),
                          ("xray", "XR"), ("xray-cine", "XR")):  # fmt: skip
        case(f"export-{sample}", sample, image, f"{sample}.nii.gz")
    case("export-xray-seg", "xray", "XR", "seg.nii.gz", annotation="seg")
    case("export-dce-uneven-nii", "dce-uneven", "DCE", "dce.nii")
    # Names that are no NIfTI file, and what is beside the target.
    for name in ("plain", "ct.img", "ct.hdr", "ct.nii.txt"):
        case(f"export-name-{name.replace('.', '-')}", "plain", "CT", name)
    # A beside entry with no source is a directory.
    case("export-directory", "plain", "CT", "folder.nii", beside=[["folder.nii", ""]])
    case("export-missing-image", "plain", "MR", "x.nii.gz")
    case("export-missing-annotation", "ct", "CT", "x.nii.gz", annotation="nope")
    case("export-missing-class", "ct", "CT", "x.nii.gz", annotation="seg",
         class_key="spleen")  # fmt: skip
    case("export-beside-object", "dce-uneven", "DCE", "x.nii.gz",
         beside=[["x.json", "object.json"]])  # fmt: skip
    case("export-beside-timing", "dce-even", "DCE", "x.nii.gz",
         beside=[["x.json", "timing.json"]])  # fmt: skip
    case("export-beside-list", "echo", "MR", "x.nii.gz",
         beside=[["x.json", "list.json"], ["x.bval", "old.bval"]])  # fmt: skip
    case("export-beside-broken", "echo", "MR", "x.nii.gz",
         beside=[["x.json", "broken.json"], ["x.bval", "old.bval"]])  # fmt: skip
    case("export-beside-broken-unneeded", "plain", "CT", "x.nii.gz",
         beside=[["x.json", "broken.json"]])  # fmt: skip
    case("export-beside-bval", "dce-even", "DCE", "x.nii.gz",
         beside=[["x.bval", "old.bval"]])  # fmt: skip
    case("export-beside-bval-dwi", "dwi", "MR", "x.nii",
         beside=[["x.bval", "old.bval"], ["x.json", "object.json"]])  # fmt: skip
    return cases


def seg_imports() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def case(id: str, sample: str, masks: Any, **options: Any) -> None:
        cases.append({
            "id": id, "call": "import_seg_nifti",
            "sample": f"inputs/samples/{sample}.medh5",
            "masks": [[n, f"inputs/{p}"] for n, p in masks], "options": options,
        })  # fmt: skip

    # The `plain` sample's grid is le-int16's, in LPS (z, y, x).
    liver = [["liver", "mask-liver.nii.gz"]]
    organs = [["liver", "mask-liver.nii.gz"], ["lesion", "mask-lesion.nii.gz"]]
    case("seg-plain", "plain", organs)
    case("seg-plain-names", "plain", [["Liver", "mask-liver.nii.gz"],
                                      ["GTV-1", "mask-lesion.nii.gz"]])  # fmt: skip
    case("seg-plain-id", "plain", liver, ann_id="organs")
    case("seg-plain-annotated", "plain", organs, annotated_classes=["liver", "spleen"])
    case("seg-plain-grid", "plain", liver, grid="g")
    case("seg-plain-no-grid", "plain", liver, grid="nope")
    case("seg-plain-system", "plain", liver, coord_system="LPS")
    case("seg-plain-other-system", "plain", liver, coord_system="RAS")
    case("seg-plain-shape", "plain", [["liver", "mask-shape.nii.gz"]])
    case("seg-plain-origin", "plain", [["liver", "mask-origin.nii.gz"]])
    case("seg-plain-spacing", "plain", [["liver", "mask-spacing.nii.gz"]])
    case("seg-plain-direction", "plain", [["liver", "mask-direction.nii.gz"]])
    case("seg-plain-aligned", "plain", [["liver", "mask-origin.nii.gz"]],
         assume_aligned=True)  # fmt: skip
    case("seg-plain-metres", "plain", [["liver", "mask-metres.nii.gz"]])
    case("seg-plain-scaled", "plain", [["liver", "mask-scaled.nii.gz"]])
    case("seg-plain-no-mapping", "plain", [["liver", "mask-no-mapping.nii.gz"]])
    case("seg-plain-no-mapping-assumed", "plain",
         [["liver", "mask-no-mapping.nii.gz"]], assume_geometry=True)  # fmt: skip
    case("seg-plain-untransposed", "plain", liver, transpose=False)
    case("seg-ct-label-set", "ct", liver, ann_id="more")
    case("seg-ct-missing-class", "ct", [["spleen", "mask-spleen.nii.gz"]],
         ann_id="more")  # fmt: skip
    case("seg-ct-taken-id", "ct", liver)
    case("seg-ras", "ras", liver)
    case("seg-one-slice", "one-slice", [["bone", "plane-mask.nii.gz"]],
         plane_as_volume=True)  # fmt: skip
    case("seg-one-slice-plane", "one-slice", [["bone", "plane-mask.nii.gz"]])
    case("seg-plain-not-nifti", "plain", [["liver", "random.nii"]])
    case("seg-plain-empty", "plain", [])
    return cases


def main() -> None:
    if INPUTS.exists():
        shutil.rmtree(INPUTS)
    INPUTS.mkdir()
    make_niftis()
    make_beside()
    make_samples()
    cases = imports() + seg_imports() + exports() + reads()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "case ids are unique"
    table = {"corpus": "medh5 NIfTI parity", "recorded_with": "medh5 2.1.0",
             "cases": cases}  # fmt: skip
    (HERE / "cases.json").write_text(
        json.dumps(table, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(cases)} cases, {sum(1 for _ in INPUTS.rglob('*') if _.is_file())} "
          "input files")  # fmt: skip


if __name__ == "__main__":
    main()
