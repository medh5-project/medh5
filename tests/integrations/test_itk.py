"""ITK transform files in and out (`medh5.io.itk`).

The golden files in `tests/data/itk/` are small enough to check by hand: each
test states the point the file maps and why.  `ants_affine.mat` was written
with `struct` as ANTs writes one --- MATLAB v4 records, little-endian doubles,
a variable named for the ITK type (12 values: the matrix row-major, then the
translation) and `fixed` (the centre).  `TestItkCrossCheck` holds the same
files, and files this module writes, to SimpleITK's own `TransformPoint`; it
runs in CI's `test-itk` job, where SimpleITK is installed.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5Error, MEDH5ValidationError
from medh5.io.itk import (
    ItkTransform,
    from_itk_transform,
    read_itk_transform,
    to_itk_transform,
    write_itk_transform,
)
from medh5.validate import validate_file
from tests.helpers import ROOT, disk_full_while

DATA = ROOT / "tests" / "data" / "itk"

SHAPE = (6, 6, 6)

ANTS = np.array(
    [
        [0.98, -0.17, 0.0, 0.0],
        [0.17, 0.98, 0.0, 0.0],
        [0.0, 0.0, 1.02, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""`ants_affine.mat`'s linear part; its translation is (1.5, -2, 0.5) about
the centre (4, 4, 4)."""


def _ants_matrix() -> np.ndarray:
    out = ANTS.copy()
    centre = np.array([4.0, 4.0, 4.0])
    out[:3, 3] = np.array([1.5, -2.0, 0.5]) + centre - ANTS[:3, :3] @ centre
    return out


def _pair(
    path: Path,
    *,
    coord_system: str = "LPS",
    units: str = "mm",
    scale: float = 1.0,
    visits: bool = True,
) -> Path:
    """Two visits, a fixed grid on frame F and a moving one on frame M, each
    with its axes (z, y, x) as a NIfTI import's are (direction columns per
    array axis); without *visits*, one visit holding both."""
    with medh5.create(path, sample_id="s", codec="portable") as w:
        for index, (role, frame) in enumerate((("fixed", "F"), ("moving", "M"))):
            if visits:
                w.add_timepoint(f"tp{index}", days_from_baseline=index * 30)
            w.add_grid(
                role,
                shape=SHAPE,
                direction=np.eye(3)[::-1],
                spacing=(scale, scale, scale),
                coord_system=coord_system,
                units=units,
                timepoint=f"tp{index}" if visits else None,
                frame_uid=f"pseudo:{frame}",
            )
            w.add_image(
                f"CT_{role}", np.zeros(SHAPE, np.int16), grid=role, modality="CT"
            )
    return path


def _points() -> np.ndarray:
    return np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 7.25], [0.0, 0.0, 0.0]])


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _warp(path: Path, displacement: tuple[float, float, float], *, intent: int = 1007):
    """A constant field on the fixed grid's geometry, as ANTs writes a warp:
    (X, Y, Z, 1, 3), the RAS affine of an LPS grid at the origin."""
    nib = pytest.importorskip("nibabel")
    data = np.zeros((*SHAPE[::-1], 1, 3), np.float32)
    data[...] = displacement
    image = nib.Nifti1Image(data, np.diag([-1.0, -1.0, 1.0, 1.0]))
    image.header.set_intent(intent)
    nib.save(image, str(path))
    return path


class TestRead:
    def test_S10_2_an_itk_affine_maps_fixed_to_moving(self, tmp_path):
        """`affine.tfm`: A (x − c) + t + c with c = (10, 20, 30), so (1, 2, 3)
        maps to (4.2, −4.6, 1.85)."""
        path = _pair(tmp_path / "a.medh5")
        report = from_itk_transform(
            DATA / "affine.tfm",
            path,
            transform_id="reg",
            fixed="fixed",
            moving="moving",
        )
        assert report.ok, report.format(verbose=True)
        with medh5.open(path) as s:
            t = s.transforms["reg"]
            assert (t.kind, t.from_frame, t.to_frame) == (
                "affine",
                "pseudo:F",
                "pseudo:M",
            )
            mapped = t.transform_points([[1.0, 2.0, 3.0]])[0]
            assert np.allclose(mapped, [4.2, -4.6, 1.85], atol=1e-12)
            assert s.transform_between("tp0", "tp1").transform_id == "reg"
            assert s.transforms["reg"].prov is not None
        assert "reg" in validate_file(path).profiles

    def test_S10_5_a_composite_is_flattened_last_first(self, tmp_path):
        """`composite.tfm` lists a translation, then a scale: ITK scales first,
        so (1, 2, 3) maps to (12, 4, 6), and §10.5 lists the scale first."""
        path = _pair(tmp_path / "c.medh5")
        report = from_itk_transform(
            DATA / "composite.tfm",
            path,
            transform_id="reg",
            fixed="fixed",
            moving="moving",
        )
        assert report.of_kind("order")
        with medh5.open(path) as s:
            t = s.transforms["reg"]
            assert t.kind == "composite"
            assert t.component_ids == ("reg.affine1", "reg.affine2") or list(
                t.component_ids
            ) == ["reg.affine1", "reg.affine2"]
            first, second = t.components()
            assert np.allclose(first.matrix, np.diag([2.0, 2.0, 2.0, 1.0]))
            assert np.allclose(second.matrix[:3, 3], [10.0, 0.0, 0.0])
            assert first.to_frame == second.from_frame == "pseudo:reg.mid"
            assert np.allclose(
                t.transform_points([[1.0, 2.0, 3.0]]), [[12.0, 4.0, 6.0]]
            )
            assert s.transform_between("tp0", "tp1").transform_id == "reg"
        assert "reg" in validate_file(path).profiles

    def test_S10_3_the_centre_folds_into_the_translation(self, tmp_path):
        """`euler.tfm`: 90° about z through the centre (10, 0, 0), then
        (1, 2, 3): the centre maps to itself plus the translation, and the
        stored matrix carries t + c − A c."""
        path = _pair(tmp_path / "e.medh5")
        from_itk_transform(
            DATA / "euler.tfm", path, transform_id="reg", fixed="fixed", moving="moving"
        )
        with medh5.open(path) as s:
            matrix = s.transforms["reg"].matrix
        assert np.allclose(matrix[:3, 3], [1.0 + 10.0, 2.0 - 10.0, 3.0], atol=1e-12)
        assert np.allclose(_apply(matrix, np.array([[10.0, 0.0, 0.0]])), [[11, 2, 3]])
        assert np.allclose(_apply(matrix, np.array([[1.0, 2.0, 3.0]])), [[9, -7, 6]])

    def test_the_euler_zyx_flag_changes_the_order(self, tmp_path):
        x, y = 0.3, -0.7
        lines = ["#Insight Transform File V1.0", "#Transform 0"]
        texts = {}
        for flag in (0, 1):
            texts[flag] = "\n".join(
                [
                    *lines,
                    "Transform: Euler3DTransform_double_3_3",
                    f"Parameters: {x} {y} 0 0 0 0",
                    f"FixedParameters: 0 0 0 {flag}",
                ]
            )
        found = {}
        for flag, text in texts.items():
            path = tmp_path / f"e{flag}.tfm"
            path.write_text(text + "\n", encoding="utf-8")
            found[flag] = read_itk_transform(path).matrix[:3, :3]
        cx, sx, cy, sy = np.cos(x), np.sin(x), np.cos(y), np.sin(y)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        assert np.allclose(found[0], rx @ ry)
        assert np.allclose(found[1], ry @ rx)

    def test_S3_1_an_ras_grid_flips_two_axes(self, tmp_path):
        """ITK's (5, 6, 7) mm, in LPS, is (−5, −6, 7) in an RAS grid's world,
        and (−0.005, −0.006, 0.007) in one in metres."""
        text = (
            "#Insight Transform File V1.0\n#Transform 0\n"
            "Transform: TranslationTransform_double_3_3\nParameters: 5 6 7\n"
            "FixedParameters:\n"
        )
        source = tmp_path / "t.tfm"
        source.write_text(text, encoding="utf-8")
        for units, scale, expected in (
            ("mm", 1.0, [-5.0, -6.0, 7.0]),
            ("m", 0.001, [-0.005, -0.006, 0.007]),
        ):
            path = _pair(
                tmp_path / f"{units}.medh5",
                coord_system="RAS",
                units=units,
                scale=scale,
            )
            report = from_itk_transform(
                source, path, transform_id="reg", fixed="fixed", moving="moving"
            )
            assert report.of_kind("coordinates")
            with medh5.open(path) as s:
                mapped = s.transforms["reg"].transform_points([[0.0, 0.0, 0.0]])[0]
            assert np.allclose(mapped, expected, atol=1e-15)

    def test_S10_4_vectors_are_required_for_a_field(self, tmp_path):
        path = _pair(tmp_path / "w.medh5")
        warp = _warp(tmp_path / "warp.nii.gz", (1.0, 0.0, 0.0))
        with pytest.raises(MEDH5ValidationError, match="vectors='lps' or 'ras'"):
            from_itk_transform(
                DATA / "ants_affine.mat",
                path,
                transform_id="reg",
                fixed="fixed",
                moving="moving",
                warp=warp,
            )
        with medh5.open(path) as s:
            assert not s.transforms, "nothing was written"
        with pytest.raises(ValueError, match="'lps' or 'ras'"):
            from_itk_transform(
                DATA / "ants_affine.mat",
                path,
                transform_id="reg",
                fixed="fixed",
                moving="moving",
                warp=warp,
                vectors="xyz",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("vectors", ["lps", "ras"])
    def test_an_ants_pair_earns_reg(self, tmp_path, vectors):
        """The warp is applied first: x_M = A (x_F + u).  With vectors in RAS
        the same numbers mean (−u_x, −u_y, u_z) in LPS."""
        pytest.importorskip("nibabel")
        path = _pair(tmp_path / "p.medh5")
        warp = _warp(tmp_path / "warp.nii.gz", (1.0, 0.5, -0.25))
        report = from_itk_transform(
            DATA / "ants_affine.mat",
            path,
            transform_id="reg",
            fixed="fixed",
            moving="moving",
            warp=warp,
            vectors=vectors,
        )
        assert report.ok, report.format(verbose=True)
        u = np.array([1.0, 0.5, -0.25])
        if vectors == "ras":
            u = u * [-1.0, -1.0, 1.0]
        with medh5.open(path) as s:
            assert set(s.transforms) == {"reg", "reg.warp", "reg.affine"}
            warp_t = s.transforms["reg.warp"]
            assert (warp_t.from_frame, warp_t.to_frame) == (
                "pseudo:F",
                "pseudo:reg.mid",
            )
            assert warp_t.field_grid_id == "fixed", "the field is on the fixed grid"
            composite = s.transforms["reg"]
            assert list(composite.component_ids) == ["reg.warp", "reg.affine"]
            found = s.transform_between("fixed", "moving", by="grid")
            assert found.transform_id == "reg", "the composite is the one shortest"
            inside = np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 0.0]])
            assert np.allclose(
                found.transform_points(inside), _apply(_ants_matrix(), inside + u)
            )
        result = validate_file(path)
        assert not result.errors, result.diagnostics
        assert "reg" in result.profiles

    def test_S3_5_a_warp_in_metres_is_the_same_warp_in_millimetres(self, tmp_path):
        """A NIfTI states its spatial unit: a field in metres, its affine and
        its vectors, imports as the same field written in millimetres."""
        nib = pytest.importorskip("nibabel")
        u = np.array([1.0, 0.5, -0.25])
        imported = {}
        for unit, factor in (("mm", 1.0), ("meter", 1e-3)):
            data = np.zeros((*SHAPE[::-1], 1, 3), np.float64)
            data[...] = u * factor
            affine = np.diag([-1.0, -1.0, 1.0, 1.0])
            affine[:3, :] *= factor
            image = nib.Nifti1Image(data, affine)
            image.header.set_intent(1007)
            image.header.set_xyzt_units(xyz=unit)
            warp = tmp_path / f"warp_{unit}.nii.gz"
            nib.save(image, str(warp))
            path = _pair(tmp_path / f"{unit}.medh5")
            report = from_itk_transform(
                DATA / "ants_affine.mat",
                path,
                transform_id="reg",
                fixed="fixed",
                moving="moving",
                warp=warp,
                vectors="lps",
            )
            assert not report.guesses, report.format(verbose=True)
            with medh5.open(path) as s:
                warp_t = s.transforms["reg.warp"]
                assert warp_t.field_grid_id == "fixed", unit
                imported[unit] = np.asarray(warp_t.read_field())
        assert np.allclose(imported["meter"], imported["mm"])
        assert np.allclose(imported["mm"][:, 0, 0, 0], u)

    def test_an_inverse_warp_is_linked(self, tmp_path):
        pytest.importorskip("nibabel")
        warp = _warp(tmp_path / "warp.nii.gz", (1.0, 0.0, 0.0))
        back = _warp(tmp_path / "back.nii.gz", (-1.0, 0.0, 0.0))
        args: dict[str, Any] = {
            "transform_id": "reg",
            "fixed": "fixed",
            "moving": "moving",
            "warp": warp,
            "vectors": "lps",
            "inverse_warp": back,
        }
        # its field's grid is in the intermediate frame, which would give a
        # visit a second frame
        with pytest.raises(MEDH5ValidationError, match="second frame"):
            from_itk_transform(
                DATA / "ants_affine.mat", _pair(tmp_path / "v.medh5"), **args
            )
        path = _pair(tmp_path / "i.medh5", visits=False)
        from_itk_transform(DATA / "ants_affine.mat", path, **args)
        with medh5.open(path) as s:
            forward = s.transforms["reg.warp"]
            inverse = forward.inverse()
            assert inverse is not None and inverse.transform_id == "reg.inverse_warp"
            assert inverse.field_grid.frame_uid == "pseudo:reg.mid"
            point = np.array([[2.0, 2.0, 2.0]])
            assert np.allclose(
                inverse.transform_points(forward.transform_points(point)), point
            )
        assert not validate_file(path).errors

    @pytest.mark.parametrize("holder", ["grid", "annotation"])
    def test_S10_the_steps_meet_in_a_frame_of_their_own(self, tmp_path, holder):
        """The intermediate frame is named for the transform; a sample that
        already has a frame of that name, on a grid or on an annotation alone,
        would have the steps join its routes, so the import is refused and
        nothing is written."""
        pytest.importorskip("nibabel")
        path = _pair(tmp_path / "p.medh5")
        with medh5.amend(path) as w:
            if holder == "grid":
                w.add_timepoint("tp2", days_from_baseline=60)
                w.add_grid(
                    "other",
                    shape=SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint="tp2",
                    frame_uid="pseudo:reg.mid",
                )
            else:
                w.add_points(
                    "seeds",
                    np.array([[1.0, 2.0, 3.0]]),
                    space="world",
                    frame_uid="pseudo:reg.mid",
                )
        with pytest.raises(MEDH5ValidationError, match="'pseudo:reg.mid'"):
            from_itk_transform(
                DATA / "ants_affine.mat",
                path,
                transform_id="reg",
                fixed="fixed",
                moving="moving",
                warp=_warp(tmp_path / "warp.nii.gz", (1.0, 0.0, 0.0)),
                vectors="lps",
            )
        with medh5.open(path) as s:
            assert not s.transforms, "nothing was written"

    def test_refusals(self, tmp_path):
        path = _pair(tmp_path / "r.medh5")
        args: dict[str, Any] = {
            "transform_id": "reg",
            "fixed": "fixed",
            "moving": "moving",
        }
        # A 4x4x4 lattice takes 3 x 64 coefficients, and the file gives 3.
        with pytest.raises(MEDH5ValidationError, match="takes 192 parameters"):
            from_itk_transform(DATA / "bspline.tfm", path, **args)
        h5 = tmp_path / "t.h5"
        h5.write_bytes(b"\x89HDF\r\n\x1a\n")
        with pytest.raises(MEDH5Error, match="failed to open"):
            read_itk_transform(h5)
        mixed = tmp_path / "m.tfm"
        mixed.write_text(
            "#Insight Transform File V1.0\n#Transform 0\n"
            "Transform: AffineTransform_double_3_2\nParameters: 1 0 0 1 0 0 0 0\n"
            "FixedParameters: 0 0\n",
            encoding="utf-8",
        )
        with pytest.raises(MEDH5ValidationError, match="3-D points to 2-D"):
            read_itk_transform(mixed)
        short = tmp_path / "s.tfm"
        short.write_text(
            "#Insight Transform File V1.0\n#Transform 0\n"
            "Transform: AffineTransform_double_3_3\nParameters: 1 0 0\n"
            "FixedParameters: 0 0 0\n",
            encoding="utf-8",
        )
        with pytest.raises(MEDH5ValidationError, match="takes 12 parameters"):
            read_itk_transform(short)
        pixels = tmp_path / "px.medh5"
        with medh5.create(pixels, sample_id="s", codec="portable") as w:
            for role, frame in (("fixed", "F"), ("moving", "M")):
                w.add_grid(
                    role, shape=SHAPE, spacing=(1, 1, 1), units="px", frame_uid=frame
                )
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="fixed", modality="CT")
        with pytest.raises(MEDH5ValidationError, match="'px'"):
            from_itk_transform(DATA / "affine.tfm", pixels, **args)
        from_itk_transform(DATA / "affine.tfm", path, **args)
        with pytest.raises(MEDH5ValidationError, match="already holds"):
            from_itk_transform(DATA / "affine.tfm", path, **args)


class TestWrite:
    def test_round_trip_tfm_mat_and_h5(self, tmp_path):
        path = _pair(tmp_path / "rt.medh5", coord_system="RAS")
        from_itk_transform(
            DATA / "affine.tfm",
            path,
            transform_id="reg",
            fixed="fixed",
            moving="moving",
        )
        original = read_itk_transform(DATA / "affine.tfm").matrix
        for suffix in (".tfm", ".mat", ".h5"):
            out = tmp_path / f"out{suffix}"
            report = to_itk_transform(path, out, transform_id="reg")
            assert report.outputs == [str(out)]
            assert np.allclose(read_itk_transform(out).matrix, original, atol=1e-12)
        composite = tmp_path / "c.medh5"
        _pair(composite)
        from_itk_transform(
            DATA / "composite.tfm",
            composite,
            transform_id="reg",
            fixed="fixed",
            moving="moving",
        )
        back = read_itk_transform(DATA / "composite.tfm")
        written = to_itk_transform(composite, tmp_path / "c.tfm", transform_id="reg")
        assert written.of_kind("order")
        again = read_itk_transform(tmp_path / "c.tfm")
        assert len(again.components) == 2
        for mine, theirs in zip(again.components, back.components, strict=True):
            assert np.allclose(mine.matrix, theirs.matrix)
        to_itk_transform(composite, tmp_path / "c.mat", transform_id="reg")
        assert np.allclose(read_itk_transform(tmp_path / "c.mat").matrix, back.matrix)

    def test_write_itk_transform_by_suffix(self, tmp_path):
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 2.0, 3.0]
        single = ItkTransform("AffineTransform_double_3_3", matrix)
        write_itk_transform(tmp_path / "a.mat", single)
        data = (tmp_path / "a.mat").read_bytes()
        mopt, rows, cols, imagf, namlen = struct.unpack_from("<5i", data)
        assert (mopt, rows, cols, imagf) == (0, 12, 1, 0)
        assert data[20 : 20 + namlen] == b"AffineTransform_double_3_3\0"
        text = write_itk_transform(tmp_path / "a.tfm", single).read_text("utf-8")
        assert text.startswith("#Insight Transform File V1.0\n")
        assert "Parameters: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 1.0 2.0 3.0" in text
        composite = ItkTransform(
            "CompositeTransform_double_3_3", matrix, (single, single)
        )
        with pytest.raises(MEDH5ValidationError, match="write it as .tfm"):
            write_itk_transform(tmp_path / "c.mat", composite)
        with pytest.raises(MEDH5ValidationError, match="name a .tfm"):
            write_itk_transform(tmp_path / "c.nrrd", single)

    @pytest.mark.parametrize("vectors", ["lps", "ras"])
    def test_a_warp_round_trips(self, tmp_path, vectors):
        nib = pytest.importorskip("nibabel")
        path = _pair(tmp_path / "p.medh5")
        warp = _warp(tmp_path / "warp.nii.gz", (1.0, 0.5, -0.25))
        from_itk_transform(
            DATA / "ants_affine.mat",
            path,
            transform_id="reg",
            fixed="fixed",
            moving="moving",
            warp=warp,
            vectors="lps",
        )
        out, field = tmp_path / "out.mat", tmp_path / "out_warp.nii.gz"
        with pytest.raises(MEDH5ValidationError, match="warp_out="):
            to_itk_transform(path, out, transform_id="reg")
        with pytest.raises(MEDH5ValidationError, match="vectors="):
            to_itk_transform(path, out, transform_id="reg", warp_out=field)
        report = to_itk_transform(
            path, out, transform_id="reg", warp_out=field, vectors=vectors
        )
        assert report.outputs == [str(out), str(field)]
        assert np.allclose(read_itk_transform(out).matrix, _ants_matrix())
        image = nib.load(str(field))
        assert image.shape == (*SHAPE[::-1], 1, 3)
        assert int(image.header["intent_code"]) == 1006
        expected = np.array([1.0, 0.5, -0.25])
        if vectors == "ras":
            expected = expected * [-1.0, -1.0, 1.0]
        assert np.allclose(np.asarray(image.dataobj)[2, 3, 1, 0], expected)
        assert np.allclose(image.affine, np.diag([-1.0, -1.0, 1.0, 1.0]))

    @pytest.mark.skipif(sys.platform == "win32", reason="a file size limit")
    def test_S14_4_an_interrupted_write_leaves_the_previous_files(
        self, tmp_path, monkeypatch
    ):
        """The transform and its field go through the engine's replace: a
        write that fails part way leaves each previous file as it was, and
        no temporary beside it."""
        path = _pair(tmp_path / "p.medh5")
        warp = _warp(tmp_path / "warp.nii.gz", (1.0, 0.5, -0.25))
        from_itk_transform(
            DATA / "ants_affine.mat",
            path,
            transform_id="reg",
            fixed="fixed",
            moving="moving",
            warp=warp,
            vectors="lps",
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out, field = out_dir / "out.mat", out_dir / "out_warp.nii.gz"
        out.write_bytes(b"previous transform")
        field.write_bytes(b"previous field")

        disk_full_while(monkeypatch, medh5._core, "nifti_write_field")
        with pytest.raises(OSError):
            to_itk_transform(
                path, out, transform_id="reg", warp_out=field, vectors="lps"
            )
        assert field.read_bytes() == b"previous field"
        assert sorted(p.name for p in out_dir.iterdir()) == [
            "out.mat",
            "out_warp.nii.gz",
        ]
        monkeypatch.undo()
        # an output refused by name is refused before the field is written
        with pytest.raises(MEDH5ValidationError, match="name a .tfm"):
            to_itk_transform(
                path,
                out_dir / "out.nrrd",
                transform_id="reg",
                warp_out=field,
                vectors="lps",
            )
        assert field.read_bytes() == b"previous field"


class TestCommandLine:
    def test_from_and_to_itk_transform(self, tmp_path, capsys):
        from medh5.cli import main

        path = _pair(tmp_path / "cli.medh5")
        argv = [
            "convert",
            "from-itk-transform",
            str(DATA / "affine.tfm"),
            str(path),
            "--id",
            "reg",
            "--fixed",
            "fixed",
            "--moving",
            "moving",
            "--json",
        ]
        assert main(argv) == 0
        assert '"from-itk-transform"' in capsys.readouterr().out
        out = tmp_path / "back.tfm"
        assert (
            main(["convert", "to-itk-transform", str(path), str(out), "--id", "reg"])
            == 0
        )
        original = read_itk_transform(DATA / "affine.tfm").matrix
        assert np.allclose(read_itk_transform(out).matrix, original)
        assert (
            main(["convert", "to-itk-transform", str(path), str(path), "--id", "reg"])
            != 0
        )


def _itk_h5(path: Path, entries: list[tuple[str, list[float], list[float]]]) -> Path:
    """An ITK HDF5 transform file as ITK writes one, with h5py: each entry's
    type a one-element variable-length ASCII string, a composite's alone."""
    h5py = pytest.importorskip("h5py")
    ascii_ = h5py.string_dtype("ascii")
    with h5py.File(path, "w") as f:
        for name in ("ITKVersion", "HDFVersion", "OSName", "OSVersion"):
            f.create_dataset(name, data=[b"test"], dtype=ascii_)
        group = f.create_group("TransformGroup")
        for index, (name, parameters, fixed) in enumerate(entries):
            member = group.create_group(str(index))
            member.create_dataset("TransformType", data=[name.encode()], dtype=ascii_)
            if not name.startswith("CompositeTransform"):
                member.create_dataset(
                    "TransformFixedParameters", data=np.asarray(fixed, np.float64)
                )
                member.create_dataset(
                    "TransformParameters", data=np.asarray(parameters, np.float64)
                )
    return path


COMPOSITE_H5 = [
    ("CompositeTransform_double_3_3", [], []),
    ("TranslationTransform_double_3_3", [10.0, 0.0, 0.0], []),
    ("ScaleTransform_double_3_3", [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]),
]
"""`composite.tfm`, as ITK writes it to HDF5."""


class TestItkHdf5:
    """ITK's HDF5 transform files, read and written through the engine's
    HDF5.  They were refused before 2.2, whatever they held."""

    def test_an_itk_h5_affine_matches_its_tfm(self, tmp_path):
        text = read_itk_transform(DATA / "affine.tfm")
        lines = dict(
            line.split(":", 1)
            for line in (DATA / "affine.tfm").read_text().splitlines()
            if ":" in line and not line.startswith("#")
        )
        entry = (
            lines["Transform"].strip(),
            [float(v) for v in lines["Parameters"].split()],
            [float(v) for v in lines["FixedParameters"].split()],
        )
        # As ITK writes the same transform: its own type and centre.
        theirs = read_itk_transform(_itk_h5(tmp_path / "theirs.h5", [entry]))
        assert theirs.name == text.name
        assert np.array_equal(theirs.matrix, text.matrix)
        # And as this writes it.
        path = tmp_path / "affine.h5"
        write_itk_transform(path, text)
        assert np.array_equal(read_itk_transform(path).matrix, text.matrix)

    def test_an_itk_h5_composite_is_flattened_in_reverse(self, tmp_path):
        path = _itk_h5(tmp_path / "composite.h5", COMPOSITE_H5)
        found = read_itk_transform(path)
        text = read_itk_transform(DATA / "composite.tfm")
        assert [p.name for p in found.components] == [p.name for p in text.components]
        assert np.array_equal(found.matrix, text.matrix)
        args = {"transform_id": "reg", "fixed": "fixed", "moving": "moving"}
        sample = from_itk_transform(path, _pair(tmp_path / "c.medh5"), **args)
        other = from_itk_transform(
            DATA / "composite.tfm", _pair(tmp_path / "t.medh5"), **args
        )
        assert sample.outputs and other.outputs
        with (
            medh5.open(tmp_path / "c.medh5") as a,
            medh5.open(tmp_path / "t.medh5") as b,
        ):
            assert np.array_equal(
                a.transforms["reg"].transform_points(_points()),
                b.transforms["reg"].transform_points(_points()),
            )
        out = tmp_path / "out.h5"
        to_itk_transform(tmp_path / "t.medh5", out, transform_id="reg")
        back = read_itk_transform(out)
        assert back.is_composite and len(back.components) == 2
        assert np.allclose(back.matrix, text.matrix)

    def test_an_itk_h5_is_refused_as_its_text_would_be(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        bspline = [("BSplineTransform_double_3_3", [0.0] * 3, [1.0] * 3)]
        with pytest.raises(MEDH5ValidationError, match="BSplineTransform"):
            read_itk_transform(_itk_h5(tmp_path / "b.h5", bspline))
        short = [("TranslationTransform_double_3_3", [1.0], [])]
        with pytest.raises(MEDH5ValidationError):
            read_itk_transform(_itk_h5(tmp_path / "s.h5", short))
        with h5py.File(tmp_path / "x.h5", "w") as f:
            f.create_group("images")
        with pytest.raises(MEDH5ValidationError, match="no /TransformGroup"):
            read_itk_transform(tmp_path / "x.h5")
        with h5py.File(tmp_path / "e.h5", "w") as f:
            f.create_group("TransformGroup")
        with pytest.raises(MEDH5ValidationError, match="holds no transform"):
            read_itk_transform(tmp_path / "e.h5")

    def test_an_itk_h5_comes_in_through_the_command_line(self, tmp_path):
        from medh5.cli import main

        path = _itk_h5(tmp_path / "composite.h5", COMPOSITE_H5)
        sample = _pair(tmp_path / "c.medh5")
        grids = ["--id", "reg", "--fixed", "fixed", "--moving", "moving"]
        assert (
            main(["convert", "from-itk-transform", str(path), str(sample), *grids]) == 0
        )
        out = tmp_path / "out.h5"
        command = ["convert", "to-itk-transform", str(sample), str(out), "--id", "reg"]
        assert main(command) == 0
        assert len(read_itk_transform(out).components) == 2


LATTICE = (6, 7, 5)
"""A B-spline lattice's control points per ITK axis, x first."""

OBLIQUE = np.array(
    [
        [np.cos(0.3), -np.sin(0.3), 0.0],
        [np.sin(0.3), np.cos(0.3), 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def _lattice(
    direction: np.ndarray = OBLIQUE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A lattice's origin, spacing and direction in LPS millimetres."""
    return np.array([-20.0, 10.0, 5.0]), np.array([12.0, 9.0, 8.0]), direction


def _nodes(
    origin: np.ndarray, spacing: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """The lattice's control points in LPS, ``(*reversed(LATTICE), 3)``: z, y,
    x, as ITK's parameters order them (x fastest)."""
    zyx = np.stack(
        np.meshgrid(*[np.arange(n) for n in LATTICE[::-1]], indexing="ij"), -1
    )
    return origin + (zyx[..., ::-1] * spacing) @ direction.T


def _bspline_tfm(
    path: Path,
    coefficients: np.ndarray,
    *,
    order: int = 3,
    direction: np.ndarray = OBLIQUE,
    name: str | None = None,
) -> Path:
    """An ITK text file of one ``BSplineTransform`` on :data:`LATTICE`,
    written by hand as ITK writes one: the fixed parameters are the size,
    origin, spacing and direction (row-major), the parameters one block per
    LPS component, x fastest."""
    origin, spacing, direction = _lattice(direction)
    fixed = [*LATTICE, *origin, *spacing, *direction.ravel()]
    suffix = "" if order == 3 else f"_{order}"
    kind = name or f"BSplineTransform_double_3_3{suffix}"
    path.write_text(
        "#Insight Transform File V1.0\n#Transform 0\n"
        f"Transform: {kind}\n"
        "Parameters: " + " ".join(repr(float(v)) for v in coefficients.ravel()) + "\n"
        "FixedParameters: " + " ".join(repr(float(v)) for v in fixed) + "\n",
        encoding="utf-8",
    )
    return path


AFFINE_FIELD = np.array([[0.02, -0.01, 0.0], [0.01, 0.03, -0.02], [0.0, 0.01, -0.01]])
"""``u(x) = A x + b`` in LPS: a B-spline whose coefficients are ``A p + b``
at the control points ``p`` reproduces it exactly wherever every tap is on
the lattice (B-splines of order 1 and 3 reproduce linear functions)."""

AFFINE_OFFSET = np.array([1.5, -2.0, 0.75])


def _affine_coefficients(direction: np.ndarray = OBLIQUE) -> np.ndarray:
    nodes = _nodes(*_lattice(direction))
    values = nodes @ AFFINE_FIELD.T + AFFINE_OFFSET
    return np.moveaxis(values, -1, 0)  # (3, z, y, x)


def _inside(order: int, direction: np.ndarray = OBLIQUE) -> np.ndarray:
    """LPS points where every tap of an order-*order* spline is on the
    lattice: continuous indices in ``[1, n - 2)`` for order 3."""
    origin, spacing, direction = _lattice(direction)
    low, high = (
        (1.0, np.array(LATTICE) - 2.01)
        if order == 3
        else (0.0, np.array(LATTICE) - 1.01)
    )
    rng = np.random.default_rng(3)
    index = low + rng.random((40, 3)) * (high - low)
    return origin + (index * spacing) @ direction.T


class TestBSpline:
    """ITK's ``BSplineTransform`` stored as 1.0 §10.5's ``bspline``; refused
    before 2.2.  The hand-checkable case is a spline that reproduces an
    affine displacement exactly; `TestItkCrossCheck` holds random ones to
    SimpleITK."""

    @pytest.mark.parametrize("order", [1, 3])
    @pytest.mark.parametrize(
        ("coord_system", "units"), [("LPS", "mm"), ("RAS", "mm"), ("RAS", "m")]
    )
    def test_S10_5_a_bspline_imports_as_the_bspline_kind(
        self, tmp_path, order, coord_system, units
    ):
        tfm = _bspline_tfm(tmp_path / "b.tfm", _affine_coefficients(), order=order)
        scale = 0.001 if units == "m" else 1.0
        path = _pair(
            tmp_path / "b.medh5", coord_system=coord_system, units=units, scale=scale
        )
        report = from_itk_transform(
            tfm, path, transform_id="reg", fixed="fixed", moving="moving"
        )
        assert report.ok, report.format(verbose=True)
        [note] = report.of_kind("bspline")
        assert note.detail == {"grid": "reg_cp", "order": order, "size": [*LATTICE]}
        lps = _inside(order)
        expected_lps = lps + lps @ AFFINE_FIELD.T + AFFINE_OFFSET
        h = np.array([-1.0, -1.0, 1.0]) if coord_system == "RAS" else np.ones(3)
        h = h * (1000.0 if units == "m" else 1.0)
        with medh5.open(path) as s:
            t = s.transforms["reg"]
            assert t.kind == "bspline" and t.order == order
            assert t.cp_grid_id == "reg_cp"
            cp = s.grids["reg_cp"]
            assert cp.spatial_shape == LATTICE[::-1]
            assert cp.frame_uid == s.grids["fixed"].frame_uid
            assert (cp.coord_system, cp.units) == (coord_system, units)
            mapped = np.asarray(t.transform_points(lps / h))
        assert np.allclose(mapped * h, expected_lps, atol=1e-9)
        result = validate_file(path)
        assert not result.errors, result.diagnostics
        assert "reg" in result.profiles

    @pytest.mark.parametrize("suffix", [".tfm", ".h5"])
    def test_a_bspline_round_trips(self, tmp_path, suffix):
        coefficients = np.random.default_rng(5).normal(0.0, 3.0, (3, *LATTICE[::-1]))
        tfm = _bspline_tfm(tmp_path / "b.tfm", coefficients)
        path = _pair(tmp_path / "b.medh5", coord_system="RAS")
        from_itk_transform(
            tfm, path, transform_id="reg", fixed="fixed", moving="moving"
        )
        out = tmp_path / f"out{suffix}"
        report = to_itk_transform(path, out, transform_id="reg")
        assert report.outputs == [str(out)]
        back = read_itk_transform(out).bspline
        original = read_itk_transform(tfm).bspline
        assert back is not None and original is not None
        assert back.order == 3 and back.size == LATTICE
        for field in ("origin", "spacing", "direction", "coefficients"):
            assert np.allclose(getattr(back, field), getattr(original, field)), field

    def test_S10_5_a_composite_applies_the_bspline_first(self, tmp_path):
        """ITK lists the B-spline last and applies it first; the affine of
        `affine.tfm` follows it, from an intermediate frame."""
        bspline = _bspline_tfm(tmp_path / "b.tfm", _affine_coefficients())
        affine = (DATA / "affine.tfm").read_text().split("#Transform 0\n", 1)[1]
        spline = bspline.read_text().split("#Transform 0\n", 1)[1]
        composite = tmp_path / "c.tfm"
        composite.write_text(
            "#Insight Transform File V1.0\n#Transform 0\n"
            "Transform: CompositeTransform_double_3_3\n"
            f"#Transform 1\n{affine}#Transform 2\n{spline}",
            encoding="utf-8",
        )
        path = _pair(tmp_path / "c.medh5")
        report = from_itk_transform(
            composite, path, transform_id="reg", fixed="fixed", moving="moving"
        )
        assert report.ok, report.format(verbose=True)
        matrix = read_itk_transform(DATA / "affine.tfm").matrix
        lps = _inside(3)
        warped = lps + lps @ AFFINE_FIELD.T + AFFINE_OFFSET
        with medh5.open(path) as s:
            reg = s.transforms["reg"]
            assert [c.transform_id for c in reg.components()] == [
                "reg.bspline",
                "reg.affine",
            ]
            assert s.transforms["reg.bspline"].to_frame == "pseudo:reg.mid"
            mapped = np.asarray(reg.transform_points(lps))
        assert np.allclose(mapped, _apply(matrix, warped), atol=1e-9)
        assert not validate_file(path).errors
        out = tmp_path / "out.tfm"
        to_itk_transform(path, out, transform_id="reg")
        back = read_itk_transform(out)
        assert [p.name for p in back.components] == [
            "AffineTransform_double_3_3",
            "BSplineTransform_double_3_3",
        ]
        assert not back.is_linear and np.allclose(back.matrix, matrix)

    def test_S10_5_what_a_bspline_cannot_be_imported_as_is_refused(self, tmp_path):
        args: dict[str, Any] = {
            "transform_id": "reg",
            "fixed": "fixed",
            "moving": "moving",
        }
        c = _affine_coefficients()
        with pytest.raises(MEDH5ValidationError, match="order 2"):
            read_itk_transform(_bspline_tfm(tmp_path / "o2.tfm", c, order=2))
        v3 = _bspline_tfm(
            tmp_path / "v3.tfm", c, name="BSplineDeformableTransform_double_3_3"
        )
        with pytest.raises(MEDH5ValidationError, match="ITKv3"):
            read_itk_transform(v3)
        with pytest.raises(MEDH5ValidationError, match="not orthonormal"):
            read_itk_transform(
                _bspline_tfm(tmp_path / "d.tfm", c, direction=2 * np.eye(3))
            )
        with pytest.raises(MEDH5ValidationError, match="not an ITK transform type"):
            read_itk_transform(
                _bspline_tfm(tmp_path / "a.tfm", c, name="AffineTransform_double_3_3_1")
            )
        # Listed first in the composite, so applied last: its lattice would be
        # a grid in an intermediate frame.
        spline = (
            _bspline_tfm(tmp_path / "b.tfm", c)
            .read_text()
            .split("#Transform 0\n", 1)[1]
        )
        affine = (DATA / "affine.tfm").read_text().split("#Transform 0\n", 1)[1]
        late = tmp_path / "late.tfm"
        late.write_text(
            "#Insight Transform File V1.0\n#Transform 0\n"
            "Transform: CompositeTransform_double_3_3\n"
            f"#Transform 1\n{spline}#Transform 2\n{affine}",
            encoding="utf-8",
        )
        path = _pair(tmp_path / "r.medh5")
        with pytest.raises(MEDH5ValidationError, match="step 2 of 2"):
            from_itk_transform(late, path, **args)
        with pytest.raises(MEDH5ValidationError, match="and so is a warp"):
            from_itk_transform(
                tmp_path / "b.tfm",
                path,
                warp=_warp(tmp_path / "w.nii.gz", (1.0, 0.0, 0.0)),
                vectors="lps",
                **args,
            )
        mixed = tmp_path / "m.medh5"
        with medh5.create(mixed, sample_id="s", codec="portable") as w:
            for role, cs in (("fixed", "LPS"), ("moving", "RAS")):
                w.add_grid(
                    role,
                    shape=SHAPE,
                    spacing=(1, 1, 1),
                    coord_system=cs,
                    frame_uid=role,
                )
                w.add_image(role, np.zeros(SHAPE, np.int16), grid=role, modality="CT")
        with pytest.raises(MEDH5ValidationError, match="the fixed grid's convention"):
            from_itk_transform(tmp_path / "b.tfm", mixed, **args)
        with medh5.open(path) as s:
            assert not s.transforms and "reg_cp" not in s.grids, "nothing was written"
        from_itk_transform(tmp_path / "b.tfm", path, **args)
        with pytest.raises(MEDH5ValidationError, match="holds a B-spline"):
            to_itk_transform(path, tmp_path / "out.mat", transform_id="reg")
        assert not (tmp_path / "out.mat").exists()
        with pytest.raises(MEDH5ValidationError, match="already holds"):
            from_itk_transform(
                tmp_path / "b.tfm", path, **{**args, "transform_id": "reg"}
            )
        bspline = read_itk_transform(tmp_path / "b.tfm")
        with pytest.raises(MEDH5ValidationError, match="is a B-spline"):
            write_itk_transform(tmp_path / "b.mat", bspline)


class TestItkCrossCheck:
    """Against SimpleITK's own reading and writing (CI's `test-itk` job)."""

    @pytest.fixture(autouse=True)
    def _sitk(self) -> None:
        self.sitk = pytest.importorskip("SimpleITK")

    @pytest.mark.parametrize(
        "name", ["affine.tfm", "composite.tfm", "euler.tfm", "ants_affine.mat"]
    )
    def test_the_golden_files_map_as_simpleitk_maps_them(self, name):
        theirs = self.sitk.ReadTransform(str(DATA / name))
        mine = read_itk_transform(DATA / name).matrix
        for point in _points():
            expected = theirs.TransformPoint(tuple(float(v) for v in point))
            assert np.allclose(_apply(mine, point[None])[0], expected, atol=1e-6)

    def test_what_simpleitk_writes_is_read(self, tmp_path):
        sitk = self.sitk
        euler = sitk.Euler3DTransform((1.0, -2.0, 3.0), 0.2, -0.4, 0.9, (5.0, 6.0, 7.0))
        euler.SetComputeZYX(True)
        versor = sitk.VersorRigid3DTransform((0.1, 0.2, 0.3, 0.927), (1.0, 2.0, 3.0))
        versor.SetCenter((4.0, 5.0, 6.0))
        similarity = sitk.Similarity3DTransform(
            1.3, (0.0, 0.0, 1.0), 0.5, (1.0, 1.0, 1.0)
        )
        similarity.SetCenter((2.0, 0.0, -2.0))
        scale = sitk.ScaleTransform(3, (1.5, 0.5, 2.0))
        scale.SetCenter((1.0, 2.0, 3.0))
        composite = sitk.CompositeTransform([euler, scale])
        for index, transform in enumerate(
            (euler, versor, similarity, scale, composite)
        ):
            path = tmp_path / f"t{index}.tfm"
            sitk.WriteTransform(transform, str(path))
            mine = read_itk_transform(path).matrix
            for point in _points():
                expected = transform.TransformPoint(tuple(float(v) for v in point))
                assert np.allclose(_apply(mine, point[None])[0], expected, atol=1e-6)

    def test_what_simpleitk_writes_to_hdf5_is_read(self, tmp_path):
        sitk = self.sitk
        euler = sitk.Euler3DTransform((1.0, -2.0, 3.0), 0.2, -0.4, 0.9, (5.0, 6.0, 7.0))
        affine = sitk.AffineTransform(3)
        affine.SetMatrix((1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.2, 0.0, 1.2))
        affine.SetTranslation((1.0, 2.0, 3.0))
        affine.SetCenter((5.0, 6.0, 7.0))
        composite = sitk.CompositeTransform([affine, euler])
        for index, transform in enumerate((euler, affine, composite)):
            path = tmp_path / f"t{index}.h5"
            sitk.WriteTransform(transform, str(path))
            mine = read_itk_transform(path).matrix
            for point in _points():
                expected = transform.TransformPoint(tuple(float(v) for v in point))
                assert np.allclose(_apply(mine, point[None])[0], expected, atol=1e-6)

    def test_what_this_writes_simpleitk_reads(self, tmp_path):
        path = _pair(tmp_path / "w.medh5", coord_system="RAS")
        from_itk_transform(
            DATA / "composite.tfm",
            path,
            transform_id="reg",
            fixed="fixed",
            moving="moving",
        )
        for suffix in (".tfm", ".mat", ".h5"):
            out = tmp_path / f"out{suffix}"
            to_itk_transform(path, out, transform_id="reg")
            theirs = self.sitk.ReadTransform(str(out))
            original = self.sitk.ReadTransform(str(DATA / "composite.tfm"))
            for point in _points():
                p = tuple(float(v) for v in point)
                assert np.allclose(theirs.TransformPoint(p), original.TransformPoint(p))

    @pytest.mark.parametrize("order", [1, 3])
    @pytest.mark.parametrize("coord_system", ["LPS", "RAS"])
    def test_a_bspline_imports_exactly(self, tmp_path, order, coord_system):
        """The plan's bar: the stored ``bspline`` maps every point of the
        transform's domain, its boundary included, within 1e-6 mm of
        SimpleITK --- an oblique lattice, coefficients of a few millimetres.
        It measures about 1e-14 mm."""
        sitk = self.sitk
        rng = np.random.default_rng(order)
        image = sitk.Image([16, 16, 16], sitk.sitkFloat32)
        image.SetOrigin((-20.0, 10.0, 5.0))
        image.SetSpacing((4.0, 50.0 / 15, 2.0))
        image.SetDirection(OBLIQUE.ravel().tolist())
        theirs = sitk.BSplineTransformInitializer(image, [4, 5, 3], order=order)
        theirs.SetParameters(rng.normal(0.0, 4.0, len(theirs.GetParameters())).tolist())
        tfm = tmp_path / "b.h5"
        sitk.WriteTransform(theirs, str(tfm))
        path = _pair(tmp_path / "b.medh5", coord_system=coord_system)
        from_itk_transform(
            tfm, path, transform_id="reg", fixed="fixed", moving="moving"
        )
        lines = [np.linspace(0, 15, 17)] * 3
        index = np.stack(np.meshgrid(*lines, indexing="ij"), -1).reshape(-1, 3)
        lps = np.array(
            [image.TransformContinuousIndexToPhysicalPoint(p.tolist()) for p in index]
        )
        expected = np.array([theirs.TransformPoint(p.tolist()) for p in lps])
        flip = np.array([-1.0, -1.0, 1.0]) if coord_system == "RAS" else np.ones(3)
        with medh5.open(path) as s:
            mapped = np.asarray(s.transforms["reg"].transform_points(lps * flip)) * flip
        assert np.abs(mapped - expected).max() <= 1e-6
        out = tmp_path / "out.tfm"
        to_itk_transform(path, out, transform_id="reg")
        back = sitk.ReadTransform(str(out))
        written = np.array([back.TransformPoint(p.tolist()) for p in lps])
        assert np.abs(written - expected).max() <= 1e-6

    def test_the_vector_convention_simpleitk_writes(self, tmp_path):
        """Pinned for this SimpleITK: a vector image is written as NIfTI's
        `vector` (1007) with its components as they are --- LPS --- while a
        DISPVECT (1006) file is read as RAS and turned into LPS.  So a warp
        this module writes with vectors='ras' is the one ITK reads right, and
        one SimpleITK wrote is read with vectors='lps'."""
        sitk = self.sitk
        nib = pytest.importorskip("nibabel")
        arr = np.zeros((*SHAPE, 3))
        arr[...] = (1.0, 2.0, 3.0)
        image = sitk.GetImageFromArray(arr, isVector=True)
        written = tmp_path / "sitk.nii.gz"
        sitk.WriteImage(image, str(written))
        raw = nib.load(str(written))
        assert int(raw.header["intent_code"]) == 1007
        assert np.allclose(np.asarray(raw.dataobj)[0, 0, 0].ravel(), [1.0, 2.0, 3.0])

        path = _pair(tmp_path / "p.medh5")
        report = from_itk_transform(
            DATA / "affine.tfm",
            path,
            transform_id="reg",
            fixed="fixed",
            moving="moving",
            warp=written,
            vectors="lps",
        )
        assert report.ok, report.format(verbose=True)
        out, field = tmp_path / "out.tfm", tmp_path / "out_warp.nii.gz"
        to_itk_transform(path, out, transform_id="reg", warp_out=field, vectors="ras")
        read_back = sitk.ReadImage(str(field))
        assert np.allclose(read_back.GetPixel(0, 0, 0), (1.0, 2.0, 3.0))
        displacement = sitk.DisplacementFieldTransform(
            sitk.Cast(read_back, sitk.sitkVectorFloat64)
        )
        affine = sitk.ReadTransform(str(out))
        with medh5.open(path) as s:
            mine = s.transforms["reg"]
            for point in ([1.0, 2.0, 3.0], [2.5, 1.0, 4.0]):
                theirs = affine.TransformPoint(displacement.TransformPoint(point))
                assert np.allclose(mine.transform_points([point])[0], theirs, atol=1e-5)
