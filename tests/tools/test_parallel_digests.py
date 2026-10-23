"""§13.1 digests hashed on worker threads: a commit, `verify` and an integrity
pass read the stored chunks on the calling thread and decompress and hash them
on others.  Whatever ``--threads`` says, every digest, refusal, report line and
finding is the one a single thread produces, in the same order."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import h5py
import numpy as np
import pytest

import medh5
from medh5.cli import EXIT_OK, main
from medh5.errors import MEDH5ValidationError
from medh5.validate import validate_file
from tests.helpers import encode_attr, stored_digests, write_legacy_sample
from tests.kits import Numbered

COUNTS = (1, 4)
SHAPE = (16, 64, 64)  # four chunks, each compressed, at every profile


@pytest.fixture
def restore() -> Iterator[None]:
    """Leave the process's count as the test found it."""
    before = medh5.get_threads()
    yield
    medh5.set_threads(before)


def _at(threads: int, call: Callable[[], object]) -> object:
    before = medh5.get_threads()
    medh5.set_threads(threads)
    try:
        return call()
    finally:
        medh5.set_threads(before)


def _sample(path: Path, codec: str = "training") -> Path:
    rng = np.random.default_rng(13)
    masks = {1: np.zeros(SHAPE, bool), 2: np.zeros(SHAPE, bool)}
    masks[1][2:6, 4:20, 3:15] = True
    masks[2][6:10, 10:30, 18:30] = True
    with medh5.create(path, sample_id="s", subject_id="subj-A", codec=codec) as w:
        w.add_timepoint("tp0")
        w.add_grid("g", shape=SHAPE, spacing=(2.0, 0.8, 0.8), patch_hint=(8, 32, 32))
        for name, dtype in (("CT", np.int16), ("PT", np.float32)):
            data = rng.normal(0, 300, SHAPE).astype(dtype)
            w.add_image(name, data, grid="g", modality=name)
        w.add_segmentation("organs", grid="g", masks=masks)
    return path


def _changed(handle: h5py.File, name: str) -> None:
    """Change what *name* stores behind the library's back, whatever filter
    it is stored through: its last chunk's stored bytes overwrite its first."""
    dataset = handle[name]
    mask, stored = dataset.id.read_direct_chunk((0, 32, 32))
    dataset.id.write_direct_chunk((0, 0, 0), stored, mask)


def _attested(path: Path) -> tuple[dict[str, str], str | None]:
    with medh5.open(path) as sample:
        return stored_digests(path), sample.content_id


def _verifies_at_the_other_count(path: Path, threads: int, capsys) -> None:
    other = COUNTS[1] if threads == COUNTS[0] else COUNTS[0]
    assert main(["--threads", str(other), "verify", str(path)]) == EXIT_OK
    capsys.readouterr()


class TestCommits:
    """A file written at any count stamps the digests one thread stamps."""

    @pytest.mark.parametrize("codec", ["training", "balanced", "archive", "portable"])
    def test_S13_1_a_new_file_is_stamped_alike(self, tmp_path, restore, codec):
        stamped = [
            _at(n, lambda n=n: _attested(_sample(tmp_path / f"{n}.medh5", codec)))
            for n in COUNTS
        ]
        assert stamped[0] == stamped[1]
        assert stamped[0][1] is not None

    @pytest.mark.parametrize(
        "command",
        [
            ["recompress", "{f}", "--profile", "archive"],
            [
                "resample",
                "{f}",
                "--image",
                "CT",
                "--spacing",
                "4,2,2",
                "--grid-id",
                "g2",
            ],
            ["seg", "convert", "{f}", "organs", "--to", "layers"],
            ["fix", "{f}", "--rewrite-digests", "--reason", "restamped"],
        ],
        ids=lambda argv: argv[0] if argv[0] != "seg" else "seg-convert",
    )
    def test_S13_1_a_command_commits_the_same_content(self, tmp_path, capsys, command):
        source = _sample(tmp_path / "source.medh5")
        results = []
        for n in COUNTS:
            path = tmp_path / f"{n}.medh5"
            shutil.copyfile(source, path)
            argv = [a.format(f=path) for a in command]
            code = main(["--threads", str(n), *argv])
            assert code == EXIT_OK, capsys.readouterr().err
            results.append(_attested(path))
            _verifies_at_the_other_count(path, n, capsys)
        assert results[0] == results[1]

    def test_S13_1_a_migration_commits_the_same_content(self, tmp_path, capsys):
        old = tmp_path / "old.medh5"
        data = np.random.default_rng(3).integers(0, 900, SHAPE).astype(np.int16)
        write_legacy_sample(old, images={"CT": data})
        results = []
        for n in COUNTS:
            out = tmp_path / f"m{n}"
            argv = ["--threads", str(n), "migrate", str(old), "-o", str(out)]
            assert main(argv) == EXIT_OK, capsys.readouterr().err
            (path,) = out.glob("*.medh5")
            results.append(_attested(path))
            _verifies_at_the_other_count(path, n, capsys)
        assert results[0] == results[1]

    def test_S13_1_a_conversion_commits_the_same_content(self, tmp_path, capsys):
        nib = pytest.importorskip("nibabel")
        volume = np.random.default_rng(5).integers(0, 900, SHAPE).astype(np.int16)
        source = tmp_path / "ct.nii.gz"
        nib.save(nib.Nifti1Image(volume, np.diag([0.8, 0.8, 2.0, 1.0])), str(source))
        results = []
        for n in COUNTS:
            path = tmp_path / str(n) / "case.medh5"
            path.parent.mkdir()
            argv = ["convert", "from-nifti", str(path), "--image", f"CT={source}"]
            assert main(["--threads", str(n), *argv]) == EXIT_OK
            results.append(_attested(path))
            _verifies_at_the_other_count(path, n, capsys)
        assert results[0] == results[1]


def _damaged(path: Path) -> Path:
    """Every way a dataset can fail §13.1, one per dataset: changed content,
    a digest that does not parse, and none at all."""
    _sample(path)
    with h5py.File(path, "r+") as handle:
        _changed(handle, "images/CT")
        handle["images/PT"].attrs["digest"] = encode_attr("sha256:not-hex")
        del handle["annotations/organs/data"].attrs["digest"]
    return path


class TestReports:
    """`verify` and an integrity pass say the same, in the same order."""

    def test_S13_1_verify_reports_line_for_line_alike(self, tmp_path, capsys):
        path = _damaged(tmp_path / "damaged.medh5")
        printed = []
        for n in COUNTS:
            code = main(["--threads", str(n), "verify", str(path)])
            out = capsys.readouterr()
            printed.append((code, out.out, out.err))
        assert printed[0] == printed[1]
        assert printed[0][0] != EXIT_OK

    def test_S15_1_integrity_findings_are_alike_and_in_order(self, tmp_path, restore):
        path = _damaged(tmp_path / "damaged.medh5")

        def findings() -> list[tuple[str, str, str]]:
            report = validate_file(path, level="integrity")
            return [(d.code, d.location, d.message) for d in report.diagnostics]

        found = [_at(n, findings) for n in COUNTS]
        assert found[0] == found[1]
        assert {"E701", "E703"} <= {code for code, _, _ in found[0]}

    @pytest.mark.parametrize("digested", [True, False])
    def test_S7_5_stored_probabilities_are_checked_alike(
        self, tmp_path, restore, digested
    ):
        """The values of a probability map are checked from the bytes its
        digest pass reads, digested or not, on whichever thread hashes it."""
        path = tmp_path / "pm.medh5"
        votes = np.zeros(Numbered.SHAPE)
        votes[2:4] = 0.75
        with Numbered.writer(path) as w:
            for name in ("soft", "softer"):
                w.add_segmentation(
                    name, grid="g", probabilities={1: votes, 2: 1 - votes}
                )
        with h5py.File(path, "r+") as handle:
            for name, value in (("soft", 1.5), ("softer", 0.25)):
                group = handle[f"annotations/{name}"]
                group.attrs["normalized"] = True
                stored = group["data"][...]
                stored[0, 3, 2, 1] = value
                group["data"][...] = stored
                if not digested:
                    del group["data"].attrs["digest"]

        def findings() -> list[tuple[str, str, str]]:
            report = validate_file(path, level="integrity")
            return [(d.code, d.location, d.message) for d in report.diagnostics]

        found = [_at(n, findings) for n in COUNTS]
        assert found[0] == found[1]
        codes = [(code, location) for code, location, _ in found[0]]
        assert ("E411", "/annotations/soft/data") in codes
        assert any(c == "E404" and "softer" in loc for c, loc in codes)

    @pytest.mark.parametrize("codec", ["training", "portable"])
    def test_S13_1_an_undecodable_chunk_is_named_alike(self, tmp_path, restore, codec):
        path = _sample(tmp_path / "broken.medh5", codec)
        with h5py.File(path, "r+") as handle:
            info = handle["images/CT"].id.get_chunk_info(0)
            _changed(handle, "images/PT")
        with path.open("r+b") as raw:  # neither a frame nor a deflate stream
            raw.seek(info.byte_offset)
            raw.write(b"\xde\xad\xbe\xef" * 8)

        def outcome() -> tuple[list[tuple[str, str]], str]:
            report = validate_file(path, level="integrity")
            found = [(d.code, d.location) for d in report.diagnostics]
            with medh5.open(path) as sample, pytest.raises(OSError) as raised:
                sample.verify()
            return found, str(raised.value)

        seen = [_at(n, outcome) for n in COUNTS]
        assert seen[0] == seen[1]
        found, error = seen[0]
        assert ("E001", "/images/CT") in found
        assert ("E701", "/images/PT") in found
        assert "images/CT" in error


class TestRefusals:
    """An amendment refuses what one thread refuses, with the same words, and
    writes nothing."""

    def _refused(self, path: Path) -> tuple[str, str]:
        before = path.read_bytes()
        with pytest.raises(MEDH5ValidationError) as err, medh5.amend(path):
            pass
        assert path.read_bytes() == before
        return err.value.code, str(err.value)

    @pytest.mark.parametrize("codec", ["training", "portable"])
    def test_S13_1_changed_content_refuses_an_amendment_alike(
        self, tmp_path, restore, codec
    ):
        path = _sample(tmp_path / "changed.medh5", codec)
        with h5py.File(path, "r+") as handle:
            _changed(handle, "images/PT")
        refused = [_at(n, lambda: self._refused(path)) for n in COUNTS]
        assert refused[0] == refused[1]
        assert refused[0][0] == "E701" and "images/PT" in refused[0][1]

    def test_S13_1_an_unreadable_digest_refuses_an_amendment_alike(
        self, tmp_path, restore
    ):
        path = _sample(tmp_path / "unparsed.medh5")
        with h5py.File(path, "r+") as handle:
            handle["images/CT"].attrs["digest"] = encode_attr("sha256:not-hex")
        refused = [_at(n, lambda: self._refused(path)) for n in COUNTS]
        assert refused[0] == refused[1]
        assert refused[0][0] == "E703" and "/images/CT" in refused[0][1]

    def test_S13_1_an_amendment_that_drops_the_changed_dataset_commits(
        self, tmp_path, restore
    ):
        """Only what is carried over is checked: dropping the image
        leaves nothing to refuse, at any count."""
        for n in COUNTS:
            path = _sample(tmp_path / f"{n}.medh5")
            with h5py.File(path, "r+") as handle:
                _changed(handle, "images/PT")

            def dropped(path: Path = path) -> None:
                with medh5.amend(path) as w:
                    w.remove_image("PT")

            _at(n, dropped)
            with medh5.open(path) as sample:
                assert "PT" not in sample.images
                assert sample.verify().ok
