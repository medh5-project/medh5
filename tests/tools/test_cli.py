"""The command line."""

from __future__ import annotations

import errno
import io
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, main
from medh5.labels import LabelClass, LabelSet
from tests.helpers import (
    LATER_MINOR,
    block,
    encode_attr,
    write_legacy_sample,
    write_raters,
    write_sample,
    write_series,
)
from tests.kits import Framed, Hierarchy, Numbered, Organs


def run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr()


def _file_state(path: Path) -> tuple[int, int, bytes]:
    """What a rewrite changes even when it writes the same content: the
    inode under every open handle, the modification time, and the bytes."""
    stat = path.stat()
    return stat.st_ino, stat.st_mtime_ns, path.read_bytes()


class TestEntryPoint:
    def test_no_command_prints_help(self, capsys):
        code, out = run(capsys)
        assert code == EXIT_USAGE
        assert "COMMAND" in out.err

    @pytest.mark.parametrize(
        "argv", [[], ["seg"], ["labels"], ["labels", "registry"], ["dataset"]]
    )
    def test_help_asked_for_is_output_and_help_for_a_usage_error_is_not(
        self, capsys, argv
    ):
        """Bare `medh5` printed its help on stdout, and exited 2; a bare
        command group printed its own on stderr, and exited 2 too."""
        code, out = run(capsys, *argv)
        assert (code, out.out) == (EXIT_USAGE, "")
        assert "Usage: medh5" in out.err
        code, out = run(capsys, *argv, "--help")
        assert (code, out.err) == (EXIT_OK, "")
        assert "Usage: medh5" in out.out

    def test_version_and_help_return_their_code(self, capsys):
        assert main(["--version"]) == EXIT_OK
        assert "medh5" in capsys.readouterr().out
        assert main(["--help"]) == EXIT_OK
        assert "COMMAND" in capsys.readouterr().out
        assert main(["--no-such-flag"]) == EXIT_USAGE

    def test_an_argument_that_is_not_utf8_is_a_usage_error(self, capsys):
        """A Latin-1 file name a shell glob expanded reaches ``sys.argv`` as
        surrogate escapes.  It ended the console script in a traceback, and
        the standalone binary in a panic; it is refused as a usage error."""
        code, out = run(capsys, "info", "caf\udce9.medh5")
        assert code == EXIT_USAGE
        assert out.err == "medh5: argument 'caf�.medh5' is not valid UTF-8\n"

    def test_the_grammar_lists_each_option_once(self):
        """``command_tree()`` listed ``-h``, ``--help`` and ``--version``
        twice at every level: once as clap builds them, once by hand."""
        from medh5.cli import command_tree

        def check(node: dict[str, Any], where: str) -> None:
            options = node["options"]
            assert len(options) == len(set(options)), where
            assert {"-h", "--help"} <= set(options), where
            for name, sub in node["commands"].items():
                check(sub, f"{where} {name}")

        tree = command_tree()
        assert "--version" in tree["options"]
        check(tree, "medh5")

    @pytest.mark.skipif(sys.platform == "win32", reason="Ctrl-C is no signal there")
    def test_ctrl_c_ends_the_console_script_at_once(self, tmp_path):
        """The command runs in native code, where Python's handler only marks
        the interrupt: ``conformance build`` ran on to the end after Ctrl-C,
        then printed a ``KeyboardInterrupt`` traceback."""
        import signal
        import subprocess
        import time

        corpus = tmp_path / "corpus"
        child = subprocess.Popen(
            [sys.executable, "-m", "medh5.cli", "conformance", "build", str(corpus)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            # A case on disk: the command is under way, in native code.
            deadline = time.monotonic() + 120
            while not (corpus.is_dir() and any(corpus.iterdir())):
                assert child.poll() is None, child.stderr.read()
                assert time.monotonic() < deadline
                time.sleep(0.005)
            child.send_signal(signal.SIGINT)
            _, err = child.communicate(timeout=60)
        finally:
            child.kill()
            child.wait()
        assert child.returncode == -signal.SIGINT
        assert b"KeyboardInterrupt" not in err

    def test_only_the_console_script_takes_over_ctrl_c(self, monkeypatch):
        """A caller passing *argv* keeps its own handler; the console script
        puts back the one it found."""
        import signal

        from medh5 import cli

        during = []
        real = cli._core.cli_main

        def spy(args, host):
            during.append(signal.getsignal(signal.SIGINT))
            return real(args, host)

        monkeypatch.setattr(cli._core, "cli_main", spy)
        monkeypatch.setattr(sys, "argv", ["medh5", "--version"])
        before = signal.getsignal(signal.SIGINT)
        assert main() == EXIT_OK
        assert main(["--version"]) == EXIT_OK
        assert during == [signal.SIG_DFL, before]
        assert signal.getsignal(signal.SIGINT) is before

    def test_a_missing_name_is_a_handled_cli_error(self, tmp_path: Path, capsys):
        """`convert to-nifti FILE <unknown image>` ended in a KeyError traceback."""
        pytest.importorskip("nibabel")
        from medh5.cli import main

        path = tmp_path / "a.medh5"
        with Framed.writer(path):
            pass
        code = main(
            ["convert", "to-nifti", str(path), "PET", str(tmp_path / "x.nii.gz")]
        )
        assert code == 1
        assert capsys.readouterr().err.startswith("medh5: no image 'PET'")

    def test_a_bare_key_is_named(self):
        from medh5.cli import _what

        assert _what(KeyError("PET")) == "no such entry: 'PET'"
        assert _what(KeyError("no image 'PET'; available: []")).startswith("no image")

    def test_every_command_words_a_failed_lookup_alike(
        self, capsys, tmp_path, sample_path
    ):
        """``seg stats``, ``agree`` and the clinical commands printed the
        engine's sentence in quotes, and ``agree`` put a second "no such
        annotation" before it."""
        from medh5.collection import pack

        for argv in (
            ["seg", "stats", str(sample_path), "nope"],
            ["agree", str(sample_path), "nope", "organs_tp0"],
        ):
            code, out = run(capsys, *argv)
            assert code == EXIT_ERROR
            assert out.err.startswith("medh5: no annotation 'nope'; available: ["), argv
        shard = tmp_path / "s.medh5c"
        pack([sample_path], shard)
        code, expected = run(capsys, "info", str(shard), "--key", "nope")
        assert code == EXIT_ERROR
        assert expected.err.startswith("medh5: collection has no sample 'nope'")
        code, out = run(capsys, "clinical", "show", str(shard), "--key", "nope")
        assert (code, out.err) == (EXIT_ERROR, expected.err)


class TestOutput:
    """What the command line does when its stdout cannot take the output."""

    def test_output_lost_to_a_full_disk_fails_the_command(self, capsys, monkeypatch):
        class Full(io.StringIO):
            def write(self, text: str) -> int:
                raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(sys, "stdout", Full())
        assert main(["conformance", "list"]) == EXIT_ERROR
        assert "medh5: cannot write output: " in capsys.readouterr().err

    def test_a_reader_that_left_early_keeps_the_commands_own_code(
        self, monkeypatch, tmp_path
    ):
        """``medh5 validate ... | head`` exited 0 whatever the files held."""

        class Gone(io.StringIO):
            def write(self, text: str) -> int:
                raise BrokenPipeError(errno.EPIPE, "Broken pipe")

        monkeypatch.setattr(sys, "stdout", Gone())
        assert main(["validate", str(tmp_path / "missing.medh5")]) == EXIT_ERROR
        assert main(["conformance", "list"]) == EXIT_OK

    def test_output_reaches_python_a_block_at_a_time(self, monkeypatch, sample_path):
        """Each fragment the command wrote was a call into Python: two per
        printed line, and one per fragment of a streamed `--json` document."""
        calls = []

        class Counting(io.StringIO):
            def write(self, text: str) -> int:
                calls.append(text)
                return super().write(text)

        stream = Counting()
        monkeypatch.setattr(sys, "stdout", stream)
        assert main(["verify", *[str(sample_path)] * 5]) == EXIT_OK
        assert len(stream.getvalue().splitlines()) == 5
        assert len(calls) == 1

    @pytest.mark.parametrize("like", ["terminal", "line_buffering", "write_through"])
    def test_a_stream_python_writes_by_the_line_is_handed_each_line(
        self, monkeypatch, sample_path, like
    ):
        """Held a block at a time, a batch would show nothing at a terminal
        until it ended.  A line goes as it ends wherever Python's own stream
        goes at least that often: a terminal, a line-buffered stream, and one
        that writes through (`python -u`)."""
        calls = []

        class Recording(io.TextIOWrapper):
            def isatty(self) -> bool:
                return like == "terminal"

            def write(self, text: str) -> int:
                calls.append(text)
                return super().write(text)

        stream = Recording(
            io.BytesIO(),
            encoding="utf-8",
            line_buffering=like == "line_buffering",
            write_through=like == "write_through",
        )
        monkeypatch.setattr(sys, "stdout", stream)
        assert main(["verify", *[str(sample_path)] * 5]) == EXIT_OK
        assert len(calls) == 5
        assert all(text.startswith(f"{sample_path}: OK") for text in calls)
        assert all(text.endswith("\n") and text.count("\n") == 1 for text in calls)

    def test_stdout_goes_out_before_what_stderr_says_after_it(
        self, monkeypatch, tmp_path, sample_path
    ):
        """Buffered, stdout must still reach a log it shares with stderr
        (`2>&1`) ahead of a failure the command reported after it."""
        said: list[str] = []

        class Shared(io.StringIO):
            def write(self, text: str) -> int:
                said.append(text)
                return len(text)

        monkeypatch.setattr(sys, "stdout", Shared())
        monkeypatch.setattr(sys, "stderr", Shared())
        missing = str(tmp_path / "missing.medh5")
        argv = ["verify", str(sample_path), missing, str(sample_path)]
        assert main(argv) == EXIT_ERROR
        text = "".join(said)
        first = text.index(f"{sample_path}: OK")
        failure = text.index("medh5: ")
        assert first < failure < text.index(f"{sample_path}: OK", failure)

    def test_what_the_stdout_encoding_cannot_spell_is_escaped_not_lost(
        self, monkeypatch, longitudinal_path
    ):
        """Windows writes a redirected stdout in its ANSI code page, which has
        no box-drawing characters: ``tree`` dropped every line drawn with
        them, and exited 0."""
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", stream)
        assert main(["tree", str(longitudinal_path)]) == EXIT_OK
        stream.flush()
        printed = buffer.getvalue().decode("cp1252")
        assert "\\u251c\\u2500\\u2500 meta" in printed
        assert "sample document (§2.4)" in printed


class TestUsageErrors:
    """A usage error exits 2, as documented, whichever rule it breaks; the
    files named are never opened."""

    @pytest.mark.parametrize("group", ["clinical", "task", "cache"])
    def test_a_command_of_commands_needs_one(self, capsys, group):
        code, out = run(capsys, group)
        assert code == EXIT_USAGE
        assert f"Usage: medh5 {group}" in out.err

    @pytest.mark.parametrize(
        "cutoffs", [[], ["--cutoff-us", "0", "--cutoff-hours", "1"]]
    )
    def test_a_selection_takes_its_cutoff_once(self, capsys, tmp_path, cutoffs):
        missing = str(tmp_path / "missing.medh5")
        assert run(capsys, "clinical", "select", missing, *cutoffs)[0] == EXIT_USAGE

    @pytest.mark.parametrize("hours", ["nan", "inf", "-inf", "1e300"])
    def test_a_cutoff_in_hours_is_a_time_on_the_subject_clock(
        self, capsys, tmp_path, hours
    ):
        """NaN selected at 0 us and infinity at the clock's ends, certified."""
        missing = str(tmp_path / "missing.medh5")
        code, out = run(
            capsys, "clinical", "select", missing, f"--cutoff-hours={hours}"
        )
        assert code == EXIT_USAGE
        assert "subject clock" in out.err

    @pytest.mark.parametrize(
        "argv",
        [
            ["index", "build", "--max-coords", "0"],
            ["index", "build", "--max-coords", "-7"],
            ["index", "build", "--occupancy", "-3"],
            ["dataset", "stats", "--stride", "0"],
            ["dataset", "stats", "--workers", "-1"],
            ["bench", "--patch", "0"],
            ["bench", "--repeats", "0"],
        ],
    )
    def test_a_count_below_its_least_value_is_refused(self, capsys, tmp_path, argv):
        """``--max-coords -7`` stored no coordinates, ``--occupancy -3`` built
        no map, and a stride of 0 or a negative worker count was taken as
        some other number, all without a word."""
        *command, option, value = argv
        missing = str(tmp_path / "missing")
        code, out = run(capsys, *command, missing, f"{option}={value}")
        assert code == EXIT_USAGE
        assert f"invalid value '{value}' for '{option}" in out.err

    def test_pseudonymising_ids_needs_apply_and_a_salt(self, capsys, tmp_path):
        missing = str(tmp_path / "missing.medh5")
        code, out = run(capsys, "scrub", missing, "--pseudonymise-ids")
        assert code == EXIT_USAGE and "--apply" in out.err
        code, out = run(capsys, "scrub", missing, "--pseudonymise-ids", "--apply")
        assert code == EXIT_USAGE and "needs --salt" in out.err

    @pytest.mark.parametrize(
        ("argv", "partner"),
        [
            (["fix", "{sample}", "--reason", "re-exported"], "--rewrite-digests"),
            (["fix", "{sample}", "--by", "RAD-07"], "--rewrite-digests"),
            (
                ["fix", "{sample}", "--rebuild-index", "--by", "RAD-07"],
                "--rewrite-digests",
            ),
            (
                ["dataset", "split", "{manifest}", "-o", "{split}", "--k-folds", "2"]
                + ["--fold", "0"],
                "--write-claims",
            ),
            (
                ["dataset", "split", "{manifest}", "-o", "{split}"]
                + ["--assigned-by", "RAD-07"],
                "--write-claims",
            ),
        ],
    )
    def test_an_option_another_acts_on_is_refused_without_it(
        self, capsys, tmp_path, sample_path, argv, partner
    ):
        """`fix --reason` without `--rewrite-digests` diagnosed and exited 0,
        and `split --fold` without `--write-claims` wrote the split: each
        option was accepted and did nothing, so the run read as recorded."""
        before = sample_path.read_bytes()
        split = tmp_path / "split.json"
        names = {"sample": sample_path, "manifest": tmp_path / "m.json", "split": split}
        code, out = run(capsys, *(word.format(**names) for word in argv))
        assert code == EXIT_USAGE
        assert partner in out.err
        assert sample_path.read_bytes() == before
        assert not split.exists()

    @pytest.mark.parametrize("threshold", ["nan", "-1", "0", "1.5", "inf"])
    def test_an_iou_threshold_lies_in_0_to_1(self, capsys, sample_path, threshold):
        """NaN matched no object, and 0 or less every pair, objects that do
        not touch among them; both scored as agreement."""
        code, out = run(
            capsys,
            "agree",
            str(sample_path),
            "organs_tp0",
            "organs_tp0",
            f"--threshold={threshold}",
        )
        assert code == EXIT_USAGE
        assert "(0, 1]" in out.err


class TestInfo:
    def test_text_output_names_every_section(self, capsys, longitudinal_path):
        code, out = run(capsys, "info", str(longitudinal_path))
        assert code == EXIT_OK
        for section in ("timepoints", "grids", "images", "annotations", "label set"):
            assert section in out.out

    def test_json_output_is_machine_readable(self, capsys, sample_path):
        code, out = run(capsys, "info", str(sample_path), "--json")
        assert code == EXIT_OK
        payload = json.loads(out.out)
        assert payload["sample_id"] == "case"
        assert payload["grids"][0]["id"] == "ct_tp0"

    def test_a_missing_file_exits_nonzero(self, capsys, tmp_path):
        code, out = run(capsys, "info", str(tmp_path / "nope.medh5"))
        assert code == EXIT_ERROR
        assert "medh5:" in out.err

    def test_tree_labels_spec_roles(self, capsys, longitudinal_path):
        code, out = run(capsys, "tree", str(longitudinal_path))
        assert code == EXIT_OK
        assert "sample document (§2.4)" in out.out
        assert "kind=" in out.out

    @pytest.mark.parametrize(
        ("case", "member", "role"),
        [
            ("clinical-one-visit-history", "clinical", "clinical history (1.1 §3)"),
            ("collection-two-samples", "samples", "collection members (§2.2)"),
        ],
    )
    def test_tree_names_the_clinical_group_and_a_shards_members(
        self, capsys, tmp_path, case, member, role
    ):
        """Both were listed as extension objects (§16)."""
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=[case])
        path = str(next(tmp_path.glob(f"{case}.*")))
        code, out = run(capsys, "tree", path)
        assert code == EXIT_OK
        assert f"├── {member}/" in out.out and f"# {role}" in out.out
        code, out = run(capsys, "tree", path, "--json")
        roles = {o["name"]: o["role"] for o in json.loads(out.out)["objects"]}
        assert roles[member] == role

    def test_the_inspection_commands_take_a_shard_with_a_key(
        self, tmp_path: Path, capsys
    ):
        from medh5.cli import main
        from medh5.collection import pack

        members = []
        for name in ("a", "b"):
            path = tmp_path / f"{name}.medh5"
            with Organs.writer(path) as w:
                w.add_segmentation("seg", grid="g", masks=Organs.blocks())
            members.append(path)
        shard = tmp_path / "s.medh5c"
        pack(members, shard)

        assert main(["info", str(shard)]) == 0
        assert "per-sample detail" in capsys.readouterr().out
        for command in ("info", "tree", "verify", "timeline", "track"):
            assert main([command, str(shard), "--key", "a"]) == 0, command
            capsys.readouterr()
        # A per-sample command with no key says which keys there are.
        assert main(["timeline", str(shard)]) == 1
        assert "--key" in capsys.readouterr().err

    def test_S2_2_a_later_minors_file_lists_its_projection(self, tmp_path, capsys):
        """One grid of an axis kind only a later minor defines failed `info`
        whole.  It lists what the projection reads, names what it leaves
        out, and `--json` says why; a file of 1.1 fails as it did."""
        path = tmp_path / "later.medh5"
        with Organs.writer(path) as w:
            w.add_grid(
                "h", shape=Organs.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image("MR", np.zeros(Organs.SHAPE, np.int16), grid="h", modality="MR")
            w.add_segmentation("seg", grid="g", masks=Organs.blocks())
        with h5py.File(path, "r+") as handle:
            kinds = ["depth", "spatial", "spatial"]
            handle["grids/g"].attrs["axis_kinds"] = encode_attr(kinds)
            handle.attrs["medh5_version"] = encode_attr(LATER_MINOR)
        code, out = run(capsys, "info", str(path))
        assert code == EXIT_OK
        assert "MR" in out.out
        assert "left out     grids/g, images/CT, annotations/seg (W913" in out.out
        code, out = run(capsys, "info", str(path), "--json")
        payload = json.loads(out.out)
        assert [grid["id"] for grid in payload["grids"]] == ["h"]
        assert "axis kind 'depth'" in payload["left_out"]["grids/g"]
        with h5py.File(path, "r+") as handle:
            handle.attrs["medh5_version"] = encode_attr("1.1")
        code, out = run(capsys, "info", str(path))
        assert code == EXIT_ERROR
        assert "E110" in out.err


class TestValidate:
    def test_a_clean_file_exits_zero(self, capsys, sample_path):
        code, out = run(capsys, "validate", str(sample_path))
        assert code == EXIT_OK
        assert "OK" in out.out

    def test_a_broken_file_exits_one(self, capsys, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            del handle.attrs["medh5_version"]
        code, out = run(capsys, "validate", str(sample_path), "--level", "structural")
        assert code == EXIT_ERROR
        assert "E001" in out.out

    def test_json_output(self, capsys, sample_path):
        code, out = run(capsys, "validate", str(sample_path), "--json")
        payload = json.loads(out.out)
        assert payload[0]["ok"] is True

    def test_profile_override(self, capsys, sample_path):
        code, out = run(capsys, "validate", str(sample_path), "--profile", "reg")
        assert code == EXIT_ERROR
        assert "E009" in out.out

    def test_S15_2_verbose_adds_what_each_code_means(self, capsys, longitudinal_path):
        """Every diagnostic is printed either way; ``-v``, which the help
        said printed every diagnostic rather than only the summary, adds
        under each the meaning of its code."""
        _, plain = run(capsys, "validate", str(longitudinal_path))
        _, verbose = run(capsys, "validate", str(longitudinal_path), "-v")
        lines = verbose.out.splitlines()
        meanings = [line for line in lines if line.lstrip().startswith("-> ")]
        assert meanings
        assert [
            line for line in lines if line not in meanings
        ] == plain.out.splitlines()
        assert main(["validate", "--help"]) == EXIT_OK
        assert "also print what each diagnostic's code means" in capsys.readouterr().out


class TestVerify:
    def test_clean(self, capsys, sample_path):
        code, out = run(capsys, "verify", str(sample_path))
        assert code == EXIT_OK
        assert "OK" in out.out

    def test_mismatch_is_named(self, capsys, sample_path):
        with h5py.File(sample_path, "r+") as handle:
            data = handle["annotations/organs_tp0/data"]
            block = np.asarray(data[...])
            block[tuple(0 for _ in block.shape)] = 4
            data[...] = block
        code, out = run(capsys, "verify", str(sample_path))
        assert code == EXIT_ERROR
        assert "MISMATCH" in out.out

    def test_partial(self, capsys, sample_path):
        code, out = run(
            capsys, "verify", str(sample_path), "--partial", "images/CT_tp0", "--json"
        )
        assert code == EXIT_OK
        assert json.loads(out.out)[0]["checked"] == 1

    def test_stale_index_is_reported(self, capsys, longitudinal_path):
        with h5py.File(longitudinal_path, "r+") as handle:
            handle["index/organs_tp0"].attrs["source_digest"] = encode_attr(
                "sha256:" + "0" * 64
            )
        code, out = run(capsys, "verify", str(longitudinal_path))
        assert "STALE" in out.out

    def test_verify_names_the_content_id_state(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        path = tmp_path / "v.medh5"
        with medh5.create(path, sample_id="v") as w:
            w.add_grid(
                "g", shape=Hierarchy.SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0"
            )
            w.add_image("CT", Hierarchy.image(), grid="g", modality="CT")
        assert main(["verify", str(path)]) == 0
        assert "content_id ok" in capsys.readouterr().out
        assert main(["verify", str(path), "--partial", "images/CT"]) == 0
        assert "content_id not verified" in capsys.readouterr().out

    def test_S13_1_a_digest_that_cannot_be_checked_is_named(self, capsys, sample_path):
        """`verify` said FAILED and nothing else for a digest that is not
        ALGO:HEX, and for a `--partial` name with no digest (a group, a typo)."""
        code, out = run(
            capsys, "verify", str(sample_path), "--partial", "annotations/organs_tp0"
        )
        assert code == EXIT_ERROR
        assert "MALFORMED annotations/organs_tp0" in out.out
        with h5py.File(sample_path, "r+") as handle:
            handle["images/CT_tp0"].attrs["digest"] = encode_attr("sha256:nothex")
        code, out = run(capsys, "verify", str(sample_path))
        assert code == EXIT_ERROR
        assert "MALFORMED images/CT_tp0" in out.out

    def test_S13_the_text_names_every_path_the_document_reports(
        self, capsys, longitudinal_path
    ):
        """The text left out a digest its dataset's type cannot have: neither
        checked nor a failure, and `--partial` on one printed `OK 0 objects`
        and nothing else, where `--json` listed it as `unverifiable`."""
        path = str(longitudinal_path)
        with h5py.File(path, "r+") as handle:
            data = handle["annotations/organs_tp0/data"]
            block = np.asarray(data[...])
            block[tuple(np.argwhere(block != 0)[0])] = 0
            data[...] = block
            handle["images/CT_tp1"].attrs["digest"] = encode_attr("sha256:nothex")
            handle["index/organs_tp1"].attrs["source_digest"] = encode_attr(
                "sha256:" + "0" * 64
            )
            handle["annotations/organs_tp1"].create_dataset(
                "instance_ids", data=np.arange(3, dtype=np.int64)
            )
            seq = handle.create_group("x_acme").create_dataset(
                "seq", (1,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
            )
            seq[0] = np.array([1, 2], dtype=np.uint8)
            seq.attrs["digest"] = "sha256:" + "0" * 64
        code, out = run(capsys, "verify", path, "--json")
        reported = json.loads(out.out)[0]
        text_code, text = run(capsys, "verify", path)
        assert code == text_code == EXIT_ERROR
        lines = [line.split()[:2] for line in text.out.splitlines()]
        for field, tag in [
            ("mismatched", "MISMATCH"),
            ("malformed", "MALFORMED"),
            ("unattested", "UNSIGNED"),
            ("stale_index", "STALE"),
            ("unverifiable", "UNCHECKED"),
        ]:
            assert reported[field], field
            for name in reported[field]:
                shown = f"index/{name}" if field == "stale_index" else name
                assert [tag, shown] in lines, (field, name, text.out)
        code, out = run(capsys, "verify", path, "--partial", "x_acme/seq")
        assert code == EXIT_OK
        assert "UNCHECKED x_acme/seq" in out.out


class TestLongitudinal:
    def test_timeline(self, capsys, longitudinal_path):
        code, out = run(capsys, "timeline", str(longitudinal_path))
        assert code == EXIT_OK
        assert "tp0" in out.out and "tp1" in out.out

    def test_timeline_json(self, capsys, longitudinal_path):
        code, out = run(capsys, "timeline", str(longitudinal_path), "--json")
        payload = json.loads(out.out)
        assert payload[1]["images"] == ["CT_tp1"]

    def test_track_without_instances(self, capsys, longitudinal_path):
        code, out = run(capsys, "track", str(longitudinal_path))
        assert code == EXIT_OK
        assert "no instance-carrying annotations in this sample" in out.out

    def test_track_says_why_the_instances_it_found_do_not_join(
        self, capsys, tmp_path, label_set
    ):
        """It said there were no instance-carrying annotations, then listed
        the ones it skipped."""
        from tests.helpers import SHAPE

        path = tmp_path / "kp.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0")
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0), timepoint="tp0")
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="g", modality="CT")
            w.add_keypoints(
                "kp",
                np.zeros((1, 2, 3)),
                keypoint_classes=["lesion", "lesion"],
                class_ids=["lesion"],
                grid="g",
                instance_ids=[7],
            )
        code, out = run(capsys, "track", str(path))
        assert code == EXIT_OK
        assert "no instance-carrying annotation in this sample joins" in out.out
        assert "skipped 'kp'" in out.out

    def test_track_reports_persisted_resolved_and_new(
        self, capsys, tmp_path, label_set
    ):
        import medh5
        from tests.helpers import SHAPE

        def lesion(origin):
            mask = np.zeros(SHAPE, dtype=bool)
            mask[tuple(slice(o, o + 3) for o in origin)] = True
            return mask

        path = tmp_path / "t.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            w.label_set(label_set)
            for tp, frame in (("tp0", "f0"), ("tp1", "f1")):
                w.add_grid(
                    f"g_{tp}",
                    shape=SHAPE,
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=tp,
                    frame_uid=frame,
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
                    InstanceInput(3, 1, mask=lesion((1, 1, 1))),
                    InstanceInput(3, 3, mask=lesion((5, 5, 5))),
                ],
            )
            w.add_segmentation(
                "les_tp1",
                grid="g_tp1",
                instances=[
                    InstanceInput(3, 1, mask=lesion((1, 1, 1))),
                    InstanceInput(3, 8, mask=lesion((9, 9, 9))),
                ],
            )
        code, out = run(capsys, "track", str(path), "--json")
        payload = json.loads(out.out)
        states = {t["instance_id"]: t["states"] for t in payload["tracks"]}
        assert states[1] == {"tp0": "present", "tp1": "present"}
        assert states[3] == {"tp0": "present", "tp1": "resolved"}
        assert states[8] == {"tp0": "resolved", "tp1": "present"}

        code, out = run(capsys, "track", str(path))
        assert code == EXIT_OK
        assert "resolved" in out.out and "new" in out.out


class TestLabels:
    def test_show(self, capsys, sample_path):
        code, out = run(capsys, "labels", "show", str(sample_path))
        assert code == EXIT_OK
        assert "liver" in out.out

    def test_show_json(self, capsys, sample_path):
        code, out = run(capsys, "labels", "show", str(sample_path), "--json")
        assert json.loads(out.out)["id"] == "test-v1"

    def test_show_without_a_label_set(self, capsys, tmp_path):
        import medh5
        from tests.helpers import SHAPE

        path = tmp_path / "bare.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
        code, out = run(capsys, "labels", "show", str(path))
        assert "no label set" in out.out
        # `--json` is a document whatever there is to say: here, `null`.
        code, out = run(capsys, "labels", "show", str(path), "--json")
        assert code == EXIT_OK
        assert json.loads(out.out) is None

    def test_registry_list(self, capsys):
        code, out = run(capsys, "labels", "registry", "list")
        assert code == EXIT_OK
        assert "amos22-organs" in out.out

    def test_check_agrees_with_itself(self, capsys, sample_path):
        code, out = run(capsys, "labels", "check", str(sample_path), str(sample_path))
        assert code == EXIT_OK

    def test_check_detects_drift(self, capsys, tmp_path, label_set, masks):
        from tests.helpers import write_sample

        a = write_sample(tmp_path / "a.medh5", label_set=label_set, masks=masks)
        other = LabelSet("other-v1", classes=[LabelClass(1, "liver", "Liver")])
        b = write_sample(tmp_path / "b.medh5", label_set=other, masks={1: masks[1]})
        code, out = run(capsys, "labels", "check", str(a), str(b))
        assert code == EXIT_ERROR
        assert "distinct vocabularies" in out.out

    def test_usage_errors(self, capsys):
        for argv in (["labels"], ["labels", "registry"]):
            code, out = run(capsys, *argv)
            assert code == EXIT_USAGE, argv
            assert "Usage: medh5 " + " ".join(argv) in out.err


class TestSegAndIndex:
    def test_stats_reports_the_cost_model(self, capsys, sample_path):
        code, out = run(capsys, "seg", "stats", str(sample_path), "organs_tp0")
        assert code == EXIT_OK
        assert "overlap graph" in out.out
        assert "<- stored" in out.out

    def test_stats_json(self, capsys, sample_path):
        code, out = run(
            capsys, "seg", "stats", str(sample_path), "organs_tp0", "--json"
        )
        payload = json.loads(out.out)
        assert payload["kind"] == "layers"
        assert payload["cost_bytes"]["bitmask"] > 0

    def test_stats_on_a_missing_annotation(self, capsys, sample_path):
        code, out = run(capsys, "seg", "stats", str(sample_path), "nope")
        assert code == EXIT_ERROR

    @pytest.mark.parametrize(
        ("case", "annotation", "kept"),
        [("seg-instances", "lesions", "instances"), ("seg-probmap", "soft", "probmap")],
    )
    def test_S7_6_stats_recommends_only_what_keeps_the_annotation(
        self, capsys, tmp_path, case, annotation, kept
    ):
        """`seg stats` recommended a dense encoding for `instances` and called
        the re-encode lossless; the command it printed refuses (E404), as
        dropping every instance_id.  A probability map was recommended
        binary layers, which threshold it."""
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=[case])
        path = str(tmp_path / f"{case}.medh5")
        code, out = run(capsys, "seg", "stats", path, annotation, "--json")
        assert code == EXIT_OK
        assert json.loads(out.out)["recommended"] == kept
        code, out = run(capsys, "seg", "stats", path, annotation)
        assert "seg convert" not in out.out and "losslessly" not in out.out

    def test_stats_names_a_lossless_re_encoding(self, capsys, tmp_path):
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=["seg-bitmask"])
        path = str(tmp_path / "seg-bitmask.medh5")
        code, out = run(capsys, "seg", "stats", path, "organs")
        hint = f"`medh5 seg convert {path} organs --to layers` would re-encode it"
        assert hint in out.out
        assert run(capsys, "seg", "convert", path, "organs", "--to", "layers")[0] == 0

    def test_convert_dry_run_writes_nothing(self, capsys, sample_path):
        before = sample_path.read_bytes()
        code, out = run(
            capsys,
            "seg",
            "convert",
            str(sample_path),
            "organs_tp0",
            "--to",
            "bitmask",
            "--dry-run",
        )
        assert code == EXIT_OK
        assert "would re-encode" in out.out
        assert sample_path.read_bytes() == before

    def test_convert_is_lossless(self, capsys, sample_path, masks):
        import medh5

        code, out = run(
            capsys, "seg", "convert", str(sample_path), "organs_tp0", "--to", "bitmask"
        )
        assert code == EXIT_OK
        with medh5.open(sample_path) as sample:
            seg = sample.annotations["organs_tp0"]
            assert seg.kind == "bitmask"
            for class_id, mask in masks.items():
                assert np.array_equal(seg.dense([class_id])[0], mask)

    def test_S7_6_a_probmap_is_converted_through_its_threshold(self, capsys, tmp_path):
        """§7.6 makes a probmap lossless only under its `threshold` (§7.5):
        another encoding keeps the voxels at or above it, and the help called
        every conversion lossless --- and then that one alone, when instances
        to a dense encoding lose their ids too."""
        path = tmp_path / "soft.medh5"
        votes = np.zeros(Numbered.SHAPE, np.float32)
        votes[1], votes[2], votes[3] = 0.29, 0.3, 0.9
        with Numbered.writer(path) as w:
            w.add_segmentation(
                "soft", grid="g", probabilities={1: votes}, threshold=0.3
            )
        argv = ["seg", "convert", str(path), "soft", "--to", "layers"]
        assert run(capsys, *argv)[0] == EXIT_OK
        with medh5.open(path) as sample:
            soft = sample.annotations["soft"]
            assert soft.kind == "layers"
            assert np.array_equal(soft.dense([1])[0], votes >= 0.3)
        code, out = run(capsys, "seg", "convert", "--help")
        assert code == EXIT_OK
        said = " ".join(out.out.split())
        assert "losslessly but in two cases: a probmap keeps the voxels at or " in said
        assert "instances to a dense encoding lose their ids (--drop-identity)" in said

    def test_index_build(self, capsys, sample_path):
        import medh5

        code, out = run(
            capsys, "index", "build", str(sample_path), "--max-coords", "32"
        )
        assert code == EXIT_OK
        with medh5.open(sample_path) as sample:
            assert "organs_tp0" in sample.index

    def test_S14_3_an_occupancy_block_of_0_builds_no_map(self, capsys, sample_path):
        code, out = run(capsys, "index", "build", str(sample_path), "--occupancy", "0")
        assert code == EXIT_OK
        with medh5.open(sample_path) as sample:
            assert not sample.index["organs_tp0"].has_occupancy

    def test_S14_3_a_sample_with_nothing_to_index_is_done(self, capsys, tmp_path):
        """It exited 1 with `built index for (nothing)` --- having rewritten
        the file --- though an index is optional per annotation (§14.3)."""
        import medh5
        from tests.helpers import SHAPE

        path = tmp_path / "bare.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
        before = _file_state(path)
        code, out = run(capsys, "index", "build", str(path))
        assert code == EXIT_OK
        assert "nothing to index" in out.out
        code, out = run(capsys, "index", "build", str(path), "--json")
        assert (code, json.loads(out.out)) == (EXIT_OK, {str(path): []})
        assert _file_state(path) == before

    def test_S14_3_a_build_that_would_change_nothing_leaves_the_file(
        self, capsys, longitudinal_path
    ):
        """`index build` rewrote the file --- a copy of every image, and a new
        inode under every handle open on it --- when every entry was already
        what it would write.  An entry is rebuilt when any part would change:
        here the occupancy block, which is not stored, and which a block of 10
        cuts into the same coarse shape as the 8 the entries were built with."""
        path = longitudinal_path
        built_with = ["--max-coords", "64"]  # as `write_sample` builds them
        before = _file_state(path)
        code, out = run(capsys, "index", "build", str(path), *built_with)
        assert code == EXIT_OK
        assert out.out == (
            f"{path}: index current for organs_tp0, organs_tp1; the file is unchanged\n"
        )
        assert _file_state(path) == before
        with medh5.open(path) as sample:
            coarse = sample.index["organs_tp0"].occupancy_plane(0)
        code, out = run(
            capsys, "index", "build", str(path), *built_with, "--occupancy", "10"
        )
        assert code == EXIT_OK
        assert out.out == f"{path}: built index for organs_tp0, organs_tp1\n"
        with medh5.open(path) as sample:
            recut = sample.index["organs_tp0"].occupancy_plane(0)
        assert recut.shape == coarse.shape and not np.array_equal(recut, coarse)
        before = _file_state(path)
        code, out = run(
            capsys, "index", "build", str(path), *built_with, "--occupancy", "10"
        )
        assert "the file is unchanged" in out.out
        assert _file_state(path) == before

    def test_S7_6_a_conversion_to_the_stored_encoding_leaves_the_file(
        self, capsys, sample_path
    ):
        """It said `re-encoded organs_tp0: layers -> layers` and rewrote the
        file, transcoding it twice on the way."""
        with medh5.open(sample_path) as sample:
            kind = sample.annotations["organs_tp0"].kind
        before = _file_state(sample_path)
        argv = ["seg", "convert", str(sample_path), "organs_tp0", "--to", kind]
        code, out = run(capsys, *argv)
        assert code == EXIT_OK
        assert "nothing to re-encode" in out.out
        code, out = run(capsys, *argv, "--json")
        payload = json.loads(out.out)
        assert payload["applied"] is False
        assert payload["bytes_after"] == payload["bytes_before"]
        assert _file_state(sample_path) == before

    def test_S7_6_a_dry_run_reports_the_size_the_conversion_writes(
        self, capsys, sample_path
    ):
        """The size is measured once: from the payload on a dry run, and from
        what was written otherwise --- the two must agree."""
        argv = ["seg", "convert", str(sample_path), "organs_tp0", "--to", "bitmask"]
        code, out = run(capsys, *argv, "--dry-run", "--json")
        dry = json.loads(out.out)
        code, out = run(capsys, *argv, "--json")
        done = json.loads(out.out)
        assert (dry["applied"], done["applied"]) == (False, True)
        assert dry["bytes_after"] == done["bytes_after"]
        assert dry["bytes_before"] == done["bytes_before"]

    def test_S7_4_a_conversion_its_kinds_refuse_leaves_the_file(self, capsys, tmp_path):
        """What the kinds alone refuse --- object identity dropped unasked, or
        minted from a dense encoding --- is refused before the file is
        copied for the rewrite, and nothing is left behind."""
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=["seg-instances", "seg-bitmask"])
        for case, annotation, to in [
            ("seg-instances", "lesions", "labelmap"),
            ("seg-bitmask", "organs", "instances"),
        ]:
            path = tmp_path / f"{case}.medh5"
            before = _file_state(path)
            code, out = run(capsys, "seg", "convert", str(path), annotation, "--to", to)
            assert code == EXIT_ERROR
            assert out.err.startswith("medh5: [E404] "), out.err
            assert _file_state(path) == before
            assert not [p for p in tmp_path.iterdir() if ".tmp-" in p.name]

    def test_usage_errors(self, capsys):
        assert run(capsys, "seg")[0] == EXIT_USAGE
        assert run(capsys, "index")[0] == EXIT_USAGE


class TestConformanceCommands:
    def test_list(self, capsys):
        code, out = run(capsys, "conformance", "list")
        assert code == EXIT_OK
        assert "core-minimal" in out.out

    def test_list_json(self, capsys):
        code, out = run(capsys, "conformance", "list", "--json")
        assert any(c["name"] == "seg-layers" for c in json.loads(out.out))

    def test_build_and_run_one_case(self, capsys, tmp_path):
        code, out = run(
            capsys, "conformance", "build", str(tmp_path), "--case", "core-minimal"
        )
        assert code == EXIT_OK
        assert out.out.startswith("wrote 1 cases and ")
        assert (tmp_path / "core-minimal.medh5").exists()
        assert (tmp_path / "expected.json").exists()
        code, out = run(
            capsys, "conformance", "run", str(tmp_path), "--case", "core-minimal"
        )
        assert code == EXIT_OK
        assert "1/1 cases pass" in out.out

    @pytest.mark.parametrize("command", ["build", "run", "publish"])
    def test_a_case_name_the_corpus_lacks_is_refused(self, capsys, tmp_path, command):
        """A misspelt `--case` built, ran or published nothing and reported a
        whole corpus written, or "0/0 cases pass"."""
        code, out = run(
            capsys, "conformance", command, str(tmp_path / "out"), "--case", "nosuch"
        )
        assert code == EXIT_USAGE
        assert "no conformance case 'nosuch'" in out.err
        assert not (tmp_path / "out").exists()

    def test_publish_then_score(self, capsys, tmp_path):
        code, out = run(
            capsys, "conformance", "publish", str(tmp_path), "--case", "core-minimal"
        )
        assert code == EXIT_OK
        assert ": 1 cases, see " in out.out
        assert (tmp_path / "README.md").exists()
        assert (tmp_path / "SHA256SUMS").exists()

        results = tmp_path / "results.json"
        results.write_text(
            json.dumps([{"file": "core-minimal.medh5", "errors": [], "warnings": []}]),
            encoding="utf-8",
        )
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert code == EXIT_OK
        assert "1/1 cases pass" in out.out

    def test_score_reports_a_wrong_answer(self, capsys, tmp_path):
        run(capsys, "conformance", "publish", str(tmp_path), "--case", "core-minimal")
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps([{"file": "core-minimal.medh5", "errors": ["E101"]}]),
            encoding="utf-8",
        )
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert code == EXIT_ERROR
        assert "unexpected" in out.out and "E101" in out.out

    def test_score_fails_when_the_files_have_drifted(self, capsys, tmp_path):
        run(capsys, "conformance", "publish", str(tmp_path), "--case", "core-minimal")
        case = tmp_path / "core-minimal.medh5"
        case.write_bytes(case.read_bytes() + b"\x00")
        results = tmp_path / "results.json"
        results.write_text(
            json.dumps([{"file": "core-minimal.medh5", "errors": [], "warnings": []}]),
            encoding="utf-8",
        )
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert "differ" in out.out and "core-minimal.medh5" in out.out
        # A score over files that are not the published ones is not a score.
        assert code == EXIT_ERROR
        code, out = run(
            capsys, "conformance", "score", str(tmp_path), str(results), "--json"
        )
        assert code == EXIT_ERROR
        payload = json.loads(out.out)
        assert payload["changed"] == ["core-minimal.medh5"]
        assert payload["ok"] is False and payload["passed"] == 1
        assert "differ" in out.err

    def test_score_without_a_suite_fails_cleanly(self, capsys, tmp_path):
        results = tmp_path / "results.json"
        results.write_text("[]", encoding="utf-8")
        code, out = run(capsys, "conformance", "score", str(tmp_path), str(results))
        assert code == EXIT_ERROR

    def test_usage(self, capsys):
        assert run(capsys, "conformance")[0] == EXIT_USAGE


class TestFixAndScrub:
    def test_fix_reports_and_changes_nothing(self, capsys, sample_path):
        code, out = run(capsys, "fix", str(sample_path))
        assert code == EXIT_OK
        assert "nothing to fix" in out.out

    def test_S2_1_fix_names_the_repair_for_an_unknown_algorithm(
        self, capsys, sample_path
    ):
        """A file declaring a `digest_algo` §2.1 does not permit cannot be
        diagnosed; the refusal names the one repair, which restamps it in
        sha256 and records the change."""
        with h5py.File(sample_path, "r+") as handle:
            handle.attrs["digest_algo"] = "blake3"
        before = sample_path.read_bytes()
        code, out = run(capsys, "fix", str(sample_path))
        assert code == EXIT_ERROR
        assert "E703" in out.err and "'blake3'" in out.err
        assert "medh5 fix --rewrite-digests --reason" in out.err
        assert sample_path.read_bytes() == before
        code, out = run(
            capsys,
            "fix",
            str(sample_path),
            "--rewrite-digests",
            "--reason",
            "imported from a blake3 tool",
        )
        assert code == EXIT_OK, out.err
        assert "'blake3'" in out.out and "sha256" in out.out
        assert run(capsys, "validate", str(sample_path), "--level", "integrity")[0] == 0

    def test_fix_refuses_to_restamp_without_a_reason(self, capsys, sample_path):
        before = sample_path.read_bytes()
        code, out = run(capsys, "fix", str(sample_path), "--rewrite-digests")
        assert code == EXIT_USAGE
        assert "--reason" in out.err
        code, out = run(
            capsys, "fix", str(sample_path), "--rewrite-digests", "--reason", ""
        )
        assert code == EXIT_ERROR
        assert "pass a reason" in out.err
        assert sample_path.read_bytes() == before

    def test_fix_restamps_when_told_why(self, capsys, sample_path):
        code, out = run(
            capsys,
            "fix",
            str(sample_path),
            "--rewrite-digests",
            "--reason",
            "rebuilt by an external tool",
        )
        assert code == EXIT_OK
        assert "rewrote digests" in out.out
        assert "asserts nothing" in out.out

    def test_S13_2_fix_names_what_no_line_of_the_content_id_covers(
        self, capsys, sample_path
    ):
        """An undigested dataset in an attested object needs attention, and
        `fix` said only that, without naming it."""
        with h5py.File(sample_path, "r+") as handle:
            handle["annotations/organs_tp0"].create_dataset(
                "instance_ids", data=np.arange(3, dtype=np.int64)
            )
        code, out = run(capsys, "fix", str(sample_path))
        assert code == EXIT_ERROR
        assert "unsigned: annotations/organs_tp0/instance_ids" in out.out

    def test_fix_rebuilds_a_stale_index(self, capsys, longitudinal_path):
        import h5py

        from tests.helpers import encode_attr

        with h5py.File(longitudinal_path, "r+") as handle:
            handle["index/organs_tp0"].attrs["source_digest"] = encode_attr(
                "sha256:" + "0" * 64
            )
        code, out = run(
            capsys, "fix", str(longitudinal_path), "--rebuild-index", "--json"
        )
        assert code == EXIT_OK
        result = json.loads(out.out)[0]
        assert result["changed"]
        assert result["rebuilt_index"] == ["organs_tp0"]

    def test_fix_leaves_a_healthy_file_alone(self, capsys, longitudinal_path):
        """An amend replaces the file, so doing one for nothing is not free."""
        before = longitudinal_path.stat().st_mtime_ns
        code, out = run(
            capsys, "fix", str(longitudinal_path), "--rebuild-index", "--json"
        )
        assert code == EXIT_OK
        result = json.loads(out.out)[0]
        assert not result["changed"]
        assert result["rebuilt_index"] == []
        assert longitudinal_path.stat().st_mtime_ns == before

    @pytest.mark.parametrize(
        "flags",
        [
            [],
            ["--rebuild-index"],
            ["--rewrite-digests", "--reason", "re-exported"],
            ["--rewrite-digests", "--reason", "re-exported", "--rebuild-index"],
        ],
    )
    def test_S13_3_fix_succeeds_exactly_when_nothing_is_left_to_fix(
        self, capsys, longitudinal_path, flags
    ):
        """`--rewrite-digests` restamped an annotation whose index entry pins
        its digest, left the entry stale, said so in a note --- and exited 0,
        where a diagnosis of the file it left exits 1."""
        path = str(longitudinal_path)
        with h5py.File(path, "r+") as handle:
            data = handle["annotations/organs_tp0/data"]
            block = np.asarray(data[...])
            block[tuple(np.argwhere(block != 0)[0])] = 0
            data[...] = block
        code, out = run(capsys, "fix", path, *flags, "--json")
        (entry,) = json.loads(out.out)
        assert entry["ok"] is (code == EXIT_OK)
        left, _ = run(capsys, "fix", path)
        assert code == left, entry
        assert code == (EXIT_OK if len(flags) == 4 else EXIT_ERROR)

    def test_fix_says_what_it_left_and_why_whatever_it_did(
        self, capsys, sample_path, longitudinal_path
    ):
        """The notes were printed only when the file changed, and a digest
        that does not parse was only "needs attention"."""
        code, out = run(capsys, "fix", str(sample_path), "--rebuild-index")
        assert code == EXIT_OK
        assert out.out.startswith(f"{sample_path}: nothing to fix\n  note: ")
        assert "which `medh5 index build` writes" in out.out
        with h5py.File(longitudinal_path, "r+") as handle:
            handle["images/CT_tp0"].attrs["digest"] = encode_attr("sha256:nothex")
        code, out = run(capsys, "fix", str(longitudinal_path))
        assert code == EXIT_ERROR
        assert "  malformed digest: images/CT_tp0 (" in out.out

    def test_the_command_line_names_its_flags_not_keyword_arguments(
        self, capsys, longitudinal_path
    ):
        """The engine words a refusal or a note for the Python API ("pass
        rewrite_digests with a reason", "rebuild them with rebuild_index",
        "`key` applies to collections"); the command line has a flag for each."""
        path = str(longitudinal_path)
        with h5py.File(path, "r+") as handle:
            data = handle["annotations/organs_tp0/data"]
            block = np.asarray(data[...])
            block[tuple(np.argwhere(block != 0)[0])] = 0
            data[...] = block
        said = []
        code, out = run(capsys, "fix", path, "--rebuild-index")
        assert code == EXIT_ERROR
        assert "pass --rewrite-digests with --reason" in out.err
        said.append(out)
        code, out = run(capsys, "fix", path, "--rewrite-digests", "--reason", "x")
        assert "rebuild them with --rebuild-index" in out.out
        said.append(out)
        code, out = run(
            capsys, "fix", path, "--rewrite-digests", "--reason", "x", "--json"
        )
        notes = json.loads(out.out)[0]["notes"]
        assert any("rebuild them with --rebuild-index" in note for note in notes)
        said.append(out)
        code, out = run(capsys, "info", path, "--key", "k")
        assert code == EXIT_ERROR
        assert "--key applies to collections" in out.err
        said.append(out)
        for printed in said:
            for api in ("rewrite_digests", "rebuild_index", "`key`", "open_collection"):
                assert api not in printed.out + printed.err, (api, printed)

    def test_scrub_finds_and_exits_non_zero(self, capsys, tmp_path):
        path = tmp_path / "dirty.medh5"
        with medh5.create(path, sample_id="s", subject_id="subj-A") as w:
            w.add_timepoint("tp0")
            w.add_grid("g", shape=(4, 4, 4), spacing=(1, 1, 1))
            w.add_image("CT", np.zeros((4, 4, 4), np.int16), grid="g", modality="CT")
            w.acquisition("CT", PatientName="Doe^Jane")
        code, out = run(capsys, "scrub", str(path))
        assert code == EXIT_ERROR
        assert "PatientName" in out.out
        assert "re-run with --apply" in out.out
        assert "NOT checked" in out.out

    def test_scrub_applies_and_attests(self, capsys, tmp_path):
        path = tmp_path / "dirty.medh5"
        with medh5.create(path, sample_id="s", subject_id="subj-A") as w:
            w.add_timepoint("tp0", date="2026-02-03")
            w.add_grid("g", shape=(4, 4, 4), spacing=(1, 1, 1))
            w.add_image("CT", np.zeros((4, 4, 4), np.int16), grid="g", modality="CT")
            w.acquisition("CT", PatientName="Doe^Jane")
        code, out = run(
            capsys,
            "scrub",
            str(path),
            "--apply",
            "--date-shift-days",
            "-30",
            "--withhold-shift",
        )
        assert code == EXIT_OK
        with medh5.open(path) as sample:
            assert "PatientName" not in sample.document.acquisition["CT"]
            assert sample.document.deidentification.method == "medh5-scrub"
        assert run(capsys, "scrub", str(path))[0] == EXIT_OK

    def test_scrub_on_a_clean_file_passes(self, capsys, sample_path):
        assert run(capsys, "scrub", str(sample_path))[0] == EXIT_OK


class TestErrorsNameTheFile:
    """An I/O or parse error names the file it is about, as the rest of the
    command line words one (`[Errno 2] No such file or directory: 'x'`);
    these said only "No such file or directory (os error 2)", or "EOF while
    parsing a value"."""

    def test_a_manifest_that_is_missing_or_not_json(self, capsys, tmp_path):
        missing = tmp_path / "missing.json"
        for command in ("split", "stats", "check"):
            code, out = run(capsys, "dataset", command, str(missing))
            assert code == EXIT_ERROR, command
            assert repr(str(missing)) in out.err, command
        truncated = tmp_path / "m.json"
        truncated.write_text("", encoding="utf-8")
        code, out = run(capsys, "dataset", "check", str(truncated))
        assert code == EXIT_ERROR
        assert f"{str(truncated)!r} is not JSON" in out.err

    def test_an_output_that_cannot_be_written(self, capsys, tmp_path, sample_path):
        manifest = tmp_path / "m.json"
        code, _ = run(
            capsys, "dataset", "index", str(sample_path.parent), "-o", str(manifest)
        )
        assert code == EXIT_OK
        target = tmp_path / "nodir" / "x.json"
        for command in ("split", "stats"):
            code, out = run(
                capsys, "dataset", command, str(manifest), "-o", str(target)
            )
            assert code == EXIT_ERROR, command
            assert repr(str(target)) in out.err, command

    def test_a_file_replaced_whole_is_named(self, capsys, tmp_path, sample_path):
        """A manifest and a label set are replaced whole, through a temporary
        beside them; a label set that could not be written said only "No such
        file or directory (os error 2)"."""
        target = tmp_path / "nodir" / "x.json"
        code, out = run(
            capsys, "dataset", "index", str(sample_path.parent), "-o", str(target)
        )
        assert code == EXIT_ERROR
        assert repr(str(target)) in out.err
        old = tmp_path / "old.medh5"
        write_legacy_sample(old, images={"CT": np.zeros((4, 4, 4), np.int16)})
        argv = ["migrate", str(old), "-o", str(tmp_path / "m")]
        code, out = run(capsys, *argv, "--write-labels", str(target))
        assert code == EXIT_ERROR
        assert repr(str(target)) in out.err
        assert not target.parent.exists()

    def test_a_report_that_cannot_be_written(self, capsys, tmp_path):
        old = tmp_path / "old.medh5"
        write_legacy_sample(old, images={"CT": np.zeros((4, 4, 4), np.int16)})
        report = tmp_path / "nodir" / "r.json"
        code, out = run(
            capsys,
            "migrate",
            str(old),
            "-o",
            str(tmp_path / "m"),
            "--report",
            str(report),
        )
        assert code == EXIT_ERROR
        assert repr(str(report)) in out.err

    def test_a_file_where_a_directory_is_wanted(self, capsys, sample_path):
        for argv in (
            ["unpack", str(sample_path), "-o", str(sample_path)],
            ["conformance", "build", str(sample_path), "--case", "core-minimal"],
        ):
            code, out = run(capsys, *argv)
            assert code == EXIT_ERROR, argv
            assert repr(str(sample_path)) in out.err, argv


def _objects(value):
    """Every object in a JSON document, outermost first."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _objects(item)


def _strings(value):
    """Every string in a JSON document, keys included."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


class TestBatches:
    """A command over several files reports on every one: a file it cannot
    process is an entry of its own, the files before it keep their reports,
    the files after it are still done, and the command fails at the end."""

    @pytest.fixture
    def batch(self, tmp_path, sample_path):
        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        after = tmp_path / "after.medh5"
        after.write_bytes(sample_path.read_bytes())
        return [str(sample_path), str(broken), str(after)]

    def test_verify_goes_on_past_a_file_it_cannot_read(self, capsys, batch):
        code, out = run(capsys, "verify", *batch, "--json")
        assert code == EXIT_ERROR
        payload = json.loads(out.out)
        assert [entry["ok"] for entry in payload] == [True, False, True]
        assert payload[1]["path"] == batch[1]
        assert payload[1]["error"] in out.err

    @pytest.mark.parametrize(
        "command",
        [
            ["fix"],
            ["recompress", "--profile", "portable"],
            ["scrub"],
            ["index", "build"],
            ["labels", "check"],
        ],
    )
    def test_every_batch_command_goes_on(self, capsys, batch, command):
        code, out = run(capsys, *command, *batch, "--json")
        assert code == EXIT_ERROR
        payload = json.loads(out.out)
        # Looked for in the parsed document: JSON escapes the backslashes of
        # a Windows path, so the raw text never holds the path as given.
        named = set(_strings(payload))
        assert batch[1] in named and batch[2] in named, payload
        assert "medh5: " in out.err

    @pytest.mark.parametrize(
        "command",
        [
            ["verify"],
            ["fix"],
            ["recompress", "--profile", "portable"],
            ["scrub"],
            ["index", "build"],
            ["labels", "check"],
            ["splits"],
        ],
    )
    def test_a_file_that_failed_has_one_shape_of_entry(self, capsys, batch, command):
        """`fix` gave its reports no `ok`, `index build` a failed file `null`
        and no reason, `labels check` only a map, and `splits` printed
        `UNREADABLE  path: MEDH5FileError: ...` on stdout.  Every batch
        command gives a file it could not process `{path, ok: false, error}`,
        its `error` what stderr said, and every file it did process `ok`."""
        code, out = run(capsys, *command, *batch, "--json")
        assert code == EXIT_ERROR
        entries = [
            entry
            for entry in _objects(json.loads(out.out))
            if {"path", "ok"} <= set(entry)
        ]
        failed = [entry for entry in entries if entry["ok"] is False]
        assert failed == [
            {"path": batch[1], "ok": False, "error": failed[0]["error"]}
        ], entries
        assert f"medh5: {failed[0]['error']}" in out.err.splitlines()
        assert not re.match(r"\w+Error: ", failed[0]["error"])
        if command[0] not in ("index", "splits"):
            assert [entry["path"] for entry in entries] == batch

    def test_a_refused_scrub_keeps_the_reports_of_files_already_changed(
        self, capsys, tmp_path, sample_path
    ):
        """A clinical sample whose text no one attested is refused; the files
        scrubbed before it were changed, and their reports carry the only key to the
        pseudonyms written (`uid_map`)."""
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=["clinical-one-visit-history"])
        clinical = tmp_path / "clinical-one-visit-history.medh5"
        code, out = run(
            capsys,
            "scrub",
            str(sample_path),
            str(clinical),
            "--apply",
            "--salt",
            "s",
            "--json",
        )
        assert code == EXIT_ERROR
        first, refused = json.loads(out.out)
        assert first["path"] == str(sample_path) and first["applied"]
        assert "uid_map" in first
        assert refused["ok"] is False and "clinical text" in refused["error"]


class TestDatasetCommands:
    """The cohort commands: index, split, stats, check."""

    @pytest.fixture
    def cohort(self, tmp_path, label_set):
        from tests.helpers import SHAPE

        root = tmp_path / "cohort"
        root.mkdir()
        masks = {1: block(SHAPE, (2, 2, 2), 8)}
        for index, subject in enumerate(("A", "A", "B", "C", "D", "E")):
            path = root / f"case-{index}.medh5"
            write_sample(
                path, label_set=label_set, masks=masks, sample_id=f"case-{index}"
            )
            with medh5.amend(path) as writer:
                writer.identity(subject_id=f"subj-{subject}")
        return root

    def test_index_writes_a_manifest(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        code, out = run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        assert code == EXIT_OK
        assert "6 sample(s), 5 subject(s)" in out.out
        assert json.loads(manifest.read_text(encoding="utf-8"))["samples"] == 6

    def test_index_json_carries_the_digest(self, capsys, cohort, tmp_path):
        code, out = run(
            capsys,
            "dataset",
            "index",
            str(cohort),
            "-o",
            str(tmp_path / "m.json"),
            "--json",
        )
        payload = json.loads(out.out)
        assert len(payload["sha256"]) == 64
        assert payload["failed"] == []

    def test_index_reports_what_would_not_open(self, capsys, cohort, tmp_path):
        (cohort / "broken.medh5").write_bytes(b"not hdf5")
        code, out = run(
            capsys, "dataset", "index", str(cohort), "-o", str(tmp_path / "m.json")
        )
        assert code == EXIT_OK
        assert "unreadable" in out.out

    def test_index_strict_stops(self, capsys, cohort, tmp_path):
        (cohort / "broken.medh5").write_bytes(b"not hdf5")
        code, out = run(
            capsys,
            "dataset",
            "index",
            str(cohort),
            "-o",
            str(tmp_path / "m.json"),
            "--strict",
        )
        assert code == EXIT_ERROR

    def test_index_of_a_root_that_is_not_there_says_so(self, capsys, tmp_path):
        """It wrote a manifest of nobody, and said nothing of the root."""
        missing = tmp_path / "no-such-cohort"
        manifest = tmp_path / "m.json"
        code, out = run(capsys, "dataset", "index", str(missing), "-o", str(manifest))
        assert code == EXIT_ERROR
        assert repr(str(missing)) in out.err
        assert not manifest.exists()

    def test_split_then_write_claims(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys,
            "dataset",
            "split",
            str(manifest),
            "-o",
            str(tmp_path / "split.json"),
            "--seed",
            "3",
            "--write-claims",
        )
        assert code == EXIT_OK
        assert "wrote split claims into 6 file(s)" in out.out
        payload = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
        assert sum(payload["counts"].values()) == 6
        with medh5.open(sorted(cohort.glob("*.medh5"))[0]) as sample:
            assert sample.document.splits[0].manifest_sha256

    def test_split_warns_when_a_partition_gets_nothing(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys,
            "dataset",
            "split",
            str(manifest),
            "--ratios",
            "train=0.98,val=0.01,test=0.01",
        )
        assert "got no groups" in out.out

    def test_split_k_folds_needs_a_fold_to_write_claims(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys,
            "dataset",
            "split",
            str(manifest),
            "--k-folds",
            "3",
            "--write-claims",
        )
        assert code == EXIT_ERROR
        code, out = run(
            capsys,
            "dataset",
            "split",
            str(manifest),
            "--k-folds",
            "3",
            "--write-claims",
            "--fold",
            "0",
            "--json",
        )
        assert code == EXIT_OK
        assert len(json.loads(out.out)["claims_written"]) == 6

    def test_split_rejects_bad_ratios(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        assert (
            run(capsys, "dataset", "split", str(manifest), "--ratios", "train")[0]
            == EXIT_ERROR
        )

    def test_stats_reports_intensities_and_classes(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys, "dataset", "stats", str(manifest), "-o", str(tmp_path / "s.json")
        )
        assert code == EXIT_OK
        assert "CT_tp0" in out.out
        payload = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
        assert payload["samples"] == 6
        assert payload["images"]["CT_tp0"]["std"] > 0

    def test_stats_can_be_restricted_to_a_partition(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        run(capsys, "dataset", "split", str(manifest), "--write-claims", "--seed", "1")
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys,
            "dataset",
            "stats",
            str(manifest),
            "--partition",
            "train",
            "--json",
        )
        assert code == EXIT_OK
        assert 0 < json.loads(out.out)["samples"] < 6

    @pytest.mark.parametrize(
        ("option", "what"), [("--image", "an image"), ("--annotation", "an annotation")]
    )
    def test_stats_refuses_an_id_no_sample_has(
        self, capsys, cohort, tmp_path, option, what
    ):
        """A misspelt id measured nothing and exited 0 with an empty table."""
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(capsys, "dataset", "stats", str(manifest), option, "CT")
        assert code == EXIT_ERROR
        assert out.err == f"medh5: no sample has {what} 'CT'\n"
        assert out.out == ""

    def test_stats_says_so_when_no_sample_claims_the_partition(
        self, capsys, cohort, tmp_path
    ):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(
            capsys, "dataset", "stats", str(manifest), "--partition", "train"
        )
        assert code == EXIT_ERROR

    def test_check_passes_a_clean_cohort(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        code, out = run(capsys, "dataset", "check", str(manifest))
        assert code == EXIT_OK
        assert "OK" in out.out

    def test_check_reports_a_missing_file(self, capsys, cohort, tmp_path):
        manifest = tmp_path / "m.json"
        run(capsys, "dataset", "index", str(cohort), "-o", str(manifest))
        sorted(cohort.glob("*.medh5"))[0].unlink()
        code, out = run(capsys, "dataset", "check", str(manifest), "--json")
        assert code == EXIT_ERROR
        assert "C402" in {f["code"] for f in json.loads(out.out)["findings"]}

    def test_usage(self, capsys):
        assert run(capsys, "dataset")[0] == EXIT_USAGE


class TestAnnotationKinds:
    """`info` and `validate` must handle the §8/§9 kinds, including grid-less ones."""

    @staticmethod
    def _build(tmp_path, name):
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=[name])
        return tmp_path / f"{name}.medh5"

    def test_info_renders_boxes_and_obb(self, capsys, tmp_path):
        path = self._build(tmp_path, "det-boxes-obb")
        code, out = run(capsys, "info", str(path))
        assert code == EXIT_OK
        assert "boxes" in out.out and "obb" in out.out
        assert "det" in out.out

    def test_info_renders_a_grid_less_classification(self, capsys, tmp_path):
        path = self._build(tmp_path, "cls-staging-and-change")
        code, out = run(capsys, "info", str(path))
        assert code == EXIT_OK
        assert "classification" in out.out
        assert "tp0,tp1" in out.out  # the change label names the visits compared

    def test_info_json_covers_every_kind(self, capsys, tmp_path):
        path = self._build(tmp_path, "shapes-contours-mesh")
        code, out = run(capsys, "info", str(path), "--json")
        kinds = {a["kind"] for a in json.loads(out.out)["annotations"]}
        assert kinds == {"contours", "mesh", "points"}

    def test_validate_is_clean_for_every_phase3_case(self, capsys, tmp_path):
        for name in (
            "det-boxes-obb",
            "det-keypoints",
            "det-boxes-world",
            "shapes-contours-mesh",
            "cls-staging-and-change",
        ):
            path = self._build(tmp_path, name)
            code, out = run(capsys, "validate", str(path), "--level", "integrity")
            assert code == EXIT_OK, out.out

    def test_seg_convert_refuses_a_geometric_annotation(self, capsys, tmp_path):
        path = self._build(tmp_path, "det-boxes-obb")
        code, out = run(
            capsys, "seg", "convert", str(path), "lesions", "--to", "layers"
        )
        assert code == EXIT_ERROR
        assert "not a voxel encoding" in out.err

    def test_tree_names_every_dataset(self, capsys, tmp_path):
        path = self._build(tmp_path, "shapes-contours-mesh")
        code, out = run(capsys, "tree", str(path))
        assert code == EXIT_OK
        for dataset in ("contour_offsets", "contour_role", "faces", "vertices"):
            assert dataset in out.out


class TestTransforms:
    """`info` and `validate` must render and check §10 transforms."""

    @staticmethod
    def _build(tmp_path, name):
        from medh5.conformance import build_corpus

        build_corpus(tmp_path, names=[name])
        return tmp_path / f"{name}.medh5"

    def test_info_renders_the_transform_table(self, capsys, tmp_path):
        path = self._build(tmp_path, "reg-displacement-composite")
        code, out = run(capsys, "info", str(path))
        assert code == EXIT_OK
        assert "transforms" in out.out
        for kind in ("affine", "displacement", "composite"):
            assert kind in out.out
        assert "reg" in out.out

    def test_info_json_carries_frames_and_timepoints(self, capsys, tmp_path):
        path = self._build(tmp_path, "reg-affine-landmarks")
        code, out = run(capsys, "info", str(path), "--json")
        transforms = json.loads(out.out)["transforms"]
        assert transforms[0]["from_frame"] != transforms[0]["to_frame"]
        assert transforms[0]["timepoints"] == ["tp0", "tp1"]

    def test_validate_is_clean_for_every_registration_case(self, capsys, tmp_path):
        for name in (
            "reg-affine-landmarks",
            "reg-inverse-pair",
            "reg-displacement-composite",
            "reg-bspline",
        ):
            path = self._build(tmp_path, name)
            code, out = run(capsys, "validate", str(path), "--level", "strict")
            assert code == EXIT_OK, out.out

    def test_a_registered_pair_no_longer_warns_about_W911(self, capsys, tmp_path):
        """The warning exists to catch exactly the file this case is not."""
        path = self._build(tmp_path, "reg-affine-landmarks")
        code, out = run(capsys, "validate", str(path))
        assert "W911" not in out.out


class TestResample:
    """`resample`: images onto another grid, recorded (§10.2, §11.1)."""

    def test_the_command_and_the_function_write_the_same_file(
        self, capsys, tmp_path, label_set, masks
    ):
        from medh5.transforms import resample

        by_command = write_sample(
            tmp_path / "a.medh5", label_set=label_set, masks=masks
        )
        by_function = tmp_path / "b.medh5"
        by_function.write_bytes(by_command.read_bytes())
        argv = ["--image", "CT_tp0", "--spacing", "3,1.6,1.6", "--grid-id", "half"]
        code, out = run(capsys, "resample", str(by_command), *argv, "--fill", "-1024")
        assert code == EXIT_OK, out.err
        assert "'half'" in out.out and "CT_tp0 -> CT_tp0_on_half" in out.out
        resample(
            by_function, "CT_tp0", spacing=(3, 1.6, 1.6), grid_id="half", fill=-1024
        )
        with medh5.open(by_command) as a, medh5.open(by_function) as b:
            assert a.content_id == b.content_id
            (activity,) = a.document.provenance.activities_by_type("resample")
            assert activity.tool == "medh5 resample"
            # The segmentation came along (W22-10 10.3), in both.
            assert "organs_tp0_on_half" in a.annotations
        assert "organs_tp0 -> organs_tp0_on_half (layers)" in out.out

    def test_no_annotations_moves_the_images_alone(
        self, capsys, tmp_path, label_set, masks
    ):
        path = write_sample(tmp_path / "s.medh5", label_set=label_set, masks=masks)
        argv = ["--image", "CT_tp0", "--spacing", "3,1.6,1.6", "--grid-id", "half"]
        code, out = run(capsys, "resample", str(path), *argv, "--no-annotations")
        assert code == EXIT_OK, out.err
        assert "not carried: organs_tp0" in out.out
        code, out = run(
            capsys,
            "resample",
            str(path),
            *argv[:2],
            "--onto",
            "half",
            "--suffix",
            "_again",
            "--no-annotations",
            "--json",
        )
        assert code == EXIT_OK, out.err
        report = json.loads(out.out)
        assert report["annotations"] == []
        assert report["not_carried"] == ["organs_tp0"]
        with medh5.open(path) as s:
            assert not [a for a in s.annotations if a.endswith(("_on_half", "_again"))]

    def test_json_is_the_report(self, capsys, tmp_path):
        path = write_sample(tmp_path / "s.medh5")
        code, out = run(
            capsys,
            "resample",
            str(path),
            "--image",
            "CT_tp0",
            "--onto",
            "ct_tp0",
            "--interpolation",
            "nearest",
            "--suffix",
            "_copy",
            "--json",
        )
        assert code == EXIT_OK, out.err
        report = json.loads(out.out)
        assert report["grid_id"] == "ct_tp0" and not report["new_grid"]
        assert report["images"] == [
            {
                "source": "CT_tp0",
                "image_id": "CT_tp0_copy",
                "dtype": "int16",
                "valid_mask": None,
                "filled_voxels": 0,
            }
        ]
        assert report["written"] == ["images/CT_tp0_copy"]

    def test_a_dtype_and_a_new_file(self, capsys, tmp_path):
        path = write_sample(tmp_path / "s.medh5")
        out_path = tmp_path / "out.medh5"
        code, out = run(
            capsys,
            "resample",
            str(path),
            "--image",
            "CT_tp0",
            "--onto",
            "ct_tp0",
            "--dtype",
            "int16",
            "-o",
            str(out_path),
        )
        assert code == EXIT_OK, out.err
        with medh5.open(out_path) as s:
            assert s.images["CT_tp0_on_ct_tp0"].dtype == np.int16
        with medh5.open(path) as s:
            assert "CT_tp0_on_ct_tp0" not in s.images

    @pytest.mark.parametrize(
        "argv",
        [
            ["--image", "CT_tp0"],
            ["--image", "CT_tp0", "--onto", "ct_tp0", "--spacing", "1,1,1"],
            ["--image", "CT_tp0", "--spacing", "1,1,1"],
            ["--image", "CT_tp0", "--onto", "ct_tp0", "--grid-id", "n"],
            ["--image", "CT_tp0", "--spacing", "1,0,1", "--grid-id", "n"],
            ["--image", "CT_tp0", "--onto", "ct_tp0", "--interpolation", "lanczos"],
            ["--image", "CT_tp0", "--onto", "ct_tp0", "--dtype", "complex64"],
            ["--onto", "ct_tp0"],
        ],
    )
    def test_a_usage_error_exits_2(self, capsys, tmp_path, argv):
        path = write_sample(tmp_path / "s.medh5")
        code, out = run(capsys, "resample", str(path), *argv)
        assert code == EXIT_USAGE
        assert out.err

    def test_a_refusal_exits_1_with_the_message(self, capsys, tmp_path):
        """Two visits, two frames, and no transform between them."""
        path = write_sample(tmp_path / "s.medh5", timepoints=("tp0", "tp1"))
        before = path.read_bytes()
        code, out = run(
            capsys, "resample", str(path), "--image", "CT_tp1", "--onto", "ct_tp0"
        )
        assert code == EXIT_ERROR
        assert out.err.startswith("medh5: no transform path relates frame")
        assert path.read_bytes() == before


class TestCuration:
    """Shards, provenance, agreement and the split audit (§2.2, §11, §12.3)."""

    def _members(self, tmp_path, label_set, masks, n=2):

        return [
            write_sample(
                tmp_path / f"m{i}.medh5",
                label_set=label_set,
                masks=masks,
                sample_id=f"m{i}",
            )
            for i in range(n)
        ]

    def test_pack_ls_unpack_round_trip(self, capsys, tmp_path, label_set, masks):
        members = self._members(tmp_path, label_set, masks)
        shard = tmp_path / "cohort.medh5c"
        code, out = run(capsys, "pack", *map(str, members), "-o", str(shard))
        assert code == EXIT_OK
        assert "2 samples" in out.out

        code, out = run(capsys, "ls", str(shard))
        assert code == EXIT_OK
        assert "m0" in out.out and "content_id" in out.out

        code, out = run(capsys, "unpack", str(shard), "-o", str(tmp_path / "back"))
        assert code == EXIT_OK
        assert (tmp_path / "back" / "m1.medh5").exists()

    @pytest.mark.parametrize(
        "argv",
        [
            ["prov", "{}"],
            ["labels", "show", "{}"],
            ["labels", "check", "{}"],
            ["seg", "stats", "{}", "organs_tp0"],
            ["seg", "convert", "{}", "organs_tp0", "--to", "bitmask"],
            ["agree", "{}", "organs_tp0", "organs_tp0"],
            ["fix", "{}"],
        ],
    )
    def test_S2_2_a_command_on_one_sample_names_the_way_to_a_member(
        self, capsys, tmp_path, label_set, masks, argv
    ):
        """A collection handed to a command that takes no ``--key`` was refused
        with the Python API's advice, "open it with open_collection()"."""
        shard = tmp_path / "s.medh5c"
        members = self._members(tmp_path, label_set, masks)
        assert run(capsys, "pack", *map(str, members), "-o", str(shard))[0] == EXIT_OK
        code, out = run(capsys, *(a.format(shard) for a in argv))
        assert code == EXIT_ERROR
        assert out.err.startswith(
            f"medh5: {shard} is a collection of 2 sample(s); this command works on one "
            f"sample file --- extract them with `medh5 unpack {shard} -o DIR`\n"
        )

    def test_S2_2_a_command_on_a_collection_names_a_sample_file(
        self, capsys, tmp_path, sample_path
    ):
        """``ls`` and ``unpack`` on a sample file answered with a wrong
        ``medh5_kind`` and "open it with `medh5.open`"."""
        refused = f"medh5: {sample_path} is a sample file, not a collection;"
        code, out = run(capsys, "ls", str(sample_path))
        assert code == EXIT_ERROR
        assert out.err == f"{refused} `medh5 info {sample_path}` describes it\n"
        code, out = run(capsys, "unpack", str(sample_path), "-o", str(tmp_path / "x"))
        assert code == EXIT_ERROR
        assert out.err == f"{refused} there is nothing to unpack\n"

    def test_pack_json_reports_sizes(self, capsys, tmp_path, label_set, masks):
        members = self._members(tmp_path, label_set, masks, n=1)
        shard = tmp_path / "one.medh5c"
        code, out = run(capsys, "pack", str(members[0]), "-o", str(shard), "--json")
        payload = json.loads(out.out)
        assert payload["samples"] == 1
        assert payload["bytes"] > 0

    def test_S2_2_pack_never_writes_over_a_sample_it_packs(
        self, capsys, tmp_path, label_set, masks
    ):
        """`pack a.medh5 -o a.medh5` replaced the sample with a collection
        holding it, and exited 0."""
        first, second = self._members(tmp_path, label_set, masks)
        before = second.read_bytes()
        alias = tmp_path / "." / second.name
        code, out = run(capsys, "pack", str(first), str(second), "-o", str(alias))
        assert code == EXIT_ERROR
        assert "is one of the samples being packed" in out.err
        assert second.read_bytes() == before
        code, out = run(capsys, "pack", str(first), "-o", str(tmp_path))
        assert code == EXIT_ERROR
        assert repr(str(tmp_path)) in out.err

    def test_pack_with_explicit_keys(self, capsys, tmp_path, label_set, masks):
        members = self._members(tmp_path, label_set, masks)
        shard = tmp_path / "k.medh5c"
        code, out = run(
            capsys,
            "pack",
            *map(str, members),
            "-o",
            str(shard),
            "--key",
            "alpha",
            "--key",
            "beta",
        )
        code, out = run(capsys, "ls", str(shard), "--json")
        assert [e["key"] for e in json.loads(out.out)["samples"]] == ["alpha", "beta"]

    def test_ls_on_a_sample_file_fails_clearly(self, capsys, sample_path):
        code, out = run(capsys, "ls", str(sample_path))
        assert code == EXIT_ERROR
        assert "collection" in out.err

    def test_unpack_selecting_a_missing_key(self, capsys, tmp_path, label_set, masks):
        members = self._members(tmp_path, label_set, masks, n=1)
        shard = tmp_path / "s.medh5c"
        run(capsys, "pack", str(members[0]), "-o", str(shard))
        code, out = run(
            capsys, "unpack", str(shard), "-o", str(tmp_path / "x"), "--key", "ghost"
        )
        assert code == EXIT_ERROR

    def test_validate_reads_a_collection(self, capsys, tmp_path, label_set, masks):
        members = self._members(tmp_path, label_set, masks)
        shard = tmp_path / "v.medh5c"
        run(capsys, "pack", *map(str, members), "-o", str(shard))
        code, out = run(capsys, "validate", str(shard), "--level", "integrity")
        assert code == EXIT_OK, out.out

    def test_prov_prints_the_graph(self, capsys, sample_path):
        code, out = run(capsys, "prov", str(sample_path))
        assert code == EXIT_OK
        assert "agents" in out.out and "activities" in out.out
        assert "quality" in out.out
        assert "dicom-psi-profile" in out.out

    def test_prov_json(self, capsys, sample_path):
        code, out = run(capsys, "prov", str(sample_path), "--json")
        payload = json.loads(out.out)
        assert payload["provenance"]["agents"]
        assert payload["deidentification"]["method"] == "dicom-psi-profile"

    def test_prov_without_a_graph(self, capsys, tmp_path):
        import medh5
        from tests.helpers import SHAPE

        path = tmp_path / "bare.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE), grid="g", modality="CT")
        code, out = run(capsys, "prov", str(path))
        assert "no provenance graph" in out.out
        assert "ABSENT (W903)" in out.out

    def test_agree_between_two_voxel_annotations(
        self, capsys, tmp_path, label_set, masks
    ):
        import medh5
        from tests.helpers import SHAPE

        path = tmp_path / "raters.medh5"
        with medh5.create(path, codec="portable") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, dtype=np.int16), grid="g", modality="CT")
            w.add_segmentation("r1", grid="g", masks=masks)
            w.add_segmentation(
                "r2",
                grid="g",
                masks={**masks, 2: block(SHAPE, (8, 8, 8), 3)},
            )
        code, out = run(capsys, "agree", str(path), "r1", "r2")
        assert code == EXIT_OK
        assert "dice =" in out.out and "liver" in out.out

        code, out = run(capsys, "agree", str(path), "r1", "r2", "--json", "--record")
        assert json.loads(out.out)["quality_agreement"]["metric"] == "dice"

        code, out = run(capsys, "agree", str(path), "r1", "nope")
        assert code == EXIT_ERROR

    def test_agree_between_instance_annotations(self, capsys, tmp_path, label_set):

        path = write_raters(tmp_path / "raters.medh5", label_set)
        code, out = run(capsys, "agree", str(path), "r1", "r2")
        assert code == EXIT_OK
        assert "matched" in out.out and "instance_id" in out.out

    def test_agree_refuses_two_visits_grids(self, capsys, tmp_path, label_set):

        path = write_series(tmp_path / "series.medh5", label_set)
        code, out = run(capsys, "agree", str(path), "les_tp0", "les_tp1")
        assert code == EXIT_ERROR
        assert "different grids" in out.err

    def test_splits_audit_reports_leakage(self, capsys, tmp_path, label_set, masks):
        import medh5

        paths = []
        for i in range(2):
            path = tmp_path / f"visit{i}.medh5"
            write_sample(path, label_set=label_set, masks=masks, sample_id=path.stem)
            with medh5.amend(path) as w:
                w.identity(subject_id="subj-shared")
                w.split(set_id="cv5", partition="train" if i == 0 else "test")
            paths.append(path)
        code, out = run(capsys, "splits", *map(str, paths))
        assert code == EXIT_ERROR
        assert "LEAK" in out.out
        assert "re-split rather than re-stamp" in out.out

        code, out = run(capsys, "splits", *map(str, paths), "--json")
        assert len(json.loads(out.out)["leaks"]) == 1

    def test_splits_audit_is_quiet_when_clean(self, capsys, sample_path):
        code, out = run(capsys, "splits", str(sample_path))
        assert code == EXIT_OK
        assert "no split claims" in out.out

    def test_splits_reports_conflicting_manifests(
        self, capsys, tmp_path, label_set, masks
    ):
        import medh5

        paths = []
        for i in range(2):
            path = tmp_path / f"c{i}.medh5"
            write_sample(path, label_set=label_set, masks=masks, sample_id=path.stem)
            with medh5.amend(path) as w:
                w.identity(subject_id=f"subj-{i}")
                w.split(
                    set_id="cv5",
                    partition="train",
                    manifest_sha256=("a" if i == 0 else "b") * 64,
                )
            paths.append(path)
        code, out = run(capsys, "splits", *map(str, paths))
        assert code == EXIT_ERROR
        assert "W906" in out.out

    def test_splits_names_an_unreadable_file_on_stderr(
        self, capsys, tmp_path, sample_path
    ):
        """It printed `UNREADABLE  path: MEDH5FileError: ...` on stdout, the
        Python API's exception name among the audit, and counted the file it
        could not read among those without a split claim."""
        bad = tmp_path / "bad.medh5"
        bad.write_bytes(b"not hdf5")
        code, out = run(capsys, "splits", str(sample_path), str(bad))
        assert code == EXIT_ERROR
        assert out.out == (
            "no split claims in 1 file(s)\n\n1 file(s) carry no split claim\n"
        )
        (said,) = out.err.splitlines()
        assert said.startswith("medh5: ") and repr(str(bad)) in said
        assert "Error:" not in said


class TestRecompressAndBench:
    """``recompress`` and ``bench`` (§14.2)."""

    def test_recompress_preserves_the_content_id(self, capsys, sample_path):
        import medh5

        with medh5.open(sample_path) as sample:
            before = sample.content_id
        code, out = run(capsys, "recompress", str(sample_path), "--profile", "archive")
        assert code == EXIT_OK
        assert "archive" in out.out
        assert "content_id" in out.out
        with medh5.open(sample_path) as sample:
            assert sample.content_id == before

    def test_recompress_json(self, capsys, sample_path):
        code, out = run(
            capsys, "recompress", str(sample_path), "--profile", "portable", "--json"
        )
        payload = json.loads(out.out)
        assert payload[0]["content_id_preserved"] is True
        assert payload[0]["profile"] == "portable"

    def test_recompress_out_takes_one_input(self, capsys, sample_path, tmp_path):
        code, out = run(
            capsys,
            "recompress",
            str(sample_path),
            str(sample_path),
            "--profile",
            "archive",
            "-o",
            str(tmp_path / "x.medh5"),
        )
        assert code == EXIT_USAGE
        assert "single input" in out.err

    def test_recompress_rechunks_only_when_asked(self, capsys, tmp_path):
        import h5py

        import medh5

        path = tmp_path / "chunked.medh5"
        with medh5.create(path, codec="balanced") as w:
            w.add_grid("g", shape=(48, 64, 64), spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT",
                np.zeros((48, 64, 64), dtype=np.int16),
                grid="g",
                modality="CT",
            )
        with h5py.File(path) as handle:
            before = handle["images/CT"].chunks
        run(capsys, "recompress", str(path), "--profile", "archive")
        with h5py.File(path) as handle:
            assert handle["images/CT"].chunks == before

    def test_S14_2_recompress_names_what_it_copied_as_stored(self, capsys, tmp_path):
        """A dataset whose stored type a rebuild would lose --- an enumeration
        and its names --- is copied as it is, and the text output says so on a
        ``KEPT`` line, as ``--json`` lists it under ``kept``."""
        path = write_sample(tmp_path / "typed.medh5", codec="portable")
        colours = h5py.enum_dtype({"RED": 0, "GREEN": 1, "BLUE": 2}, basetype="u1")
        with h5py.File(path, "r+") as f:
            f.create_dataset(
                "x_types/colours", data=np.zeros((48, 64, 64), "u1"), dtype=colours
            )
        code, out = run(capsys, "recompress", str(path), "--profile", "archive")
        assert code == EXIT_OK
        kept = "  KEPT      /x_types/colours (copied as stored: a rebuild would lose"
        assert f"{kept} its type)" in out.out.splitlines()
        code, out = run(
            capsys, "recompress", str(path), "--profile", "portable", "--json"
        )
        assert code == EXIT_OK
        assert json.loads(out.out)[0]["kept"] == ["/x_types/colours"]

    def test_S14_4_recompress_names_what_the_output_check_could_not_read(
        self, capsys, tmp_path
    ):
        """The output has replaced the input when it is checked, so what the
        check cannot read --- here a chunk whose deflate stream is damaged ---
        is reported, not raised.  The text output said ``FAILED (0)`` and
        nothing more; it names it on an ``UNCHECKED`` line, as ``--json`` lists
        it under ``unverified``."""

        def damaged(name: str) -> Path:
            path = write_sample(tmp_path / name, codec="portable")
            pair = np.dtype([("a", "<i4"), ("b", "<f4")])
            with h5py.File(path, "r+") as f:
                table = f.create_dataset(
                    "x_acme/table",
                    shape=(4,),
                    dtype=pair,
                    chunks=(4,),
                    compression="gzip",
                )
                table.id.write_direct_chunk((0,), b"\xa5" * 16)
                table.attrs["digest"] = "sha256:" + "0" * 64
            return path

        path = damaged("text.medh5")
        code, out = run(capsys, "recompress", str(path), "--profile", "archive")
        assert code == EXIT_ERROR
        assert "FAILED (1)" in out.out
        (unchecked,) = [ln for ln in out.out.splitlines() if "UNCHECKED" in ln]
        assert unchecked.startswith("  UNCHECKED /: ")
        assert unchecked.endswith("/x_acme/table")
        path = damaged("json.medh5")
        code, out = run(
            capsys, "recompress", str(path), "--profile", "archive", "--json"
        )
        assert code == EXIT_ERROR
        assert json.loads(out.out)[0]["unverified"] == [unchecked.split(maxsplit=1)[1]]

    def test_bench_runs_against_a_file(self, capsys, longitudinal_path):
        code, out = run(
            capsys,
            "bench",
            str(longitudinal_path),
            "--patch",
            "8",
            "--repeats",
            "2",
            "--no-throughput",
            "--json",
        )
        payload = json.loads(out.out)
        names = {m["name"] for m in payload["measurements"]}
        assert "meta_read_ms" in names
        assert "foreground_sample_ms" in names
        assert all("target" in m for m in payload["measurements"])

    def test_bench_text_output_names_the_targets(self, capsys, sample_path):
        code, out = run(
            capsys,
            "bench",
            str(sample_path),
            "--patch",
            "8",
            "--repeats",
            "2",
            "--no-throughput",
        )
        rows = [line for line in out.out.splitlines() if "(target ≤" in line]
        assert len(rows) >= 4, "each targeted row prints its target"
        missed = [row.split()[1] for row in rows if row.startswith("!")]
        verdict = out.out.rstrip().splitlines()[-1]
        if missed:
            assert verdict.endswith(f"below target: {', '.join(missed)}")
            assert code == EXIT_ERROR
        else:
            assert verdict == "all targets met"
            assert code == EXIT_OK


class TestConvertPathsNeverAlias:
    """A ``--report`` or an output naming a file the command reads or writes
    replaced it with exit status 0 --- ``from-dicom-seg --report case.medh5``
    the only copy of the sample it had just amended.  Each is refused before
    the converter runs, with every file as it was."""

    DICOM = b"\0" * 128 + b"DICM" + b"original"

    # Each command's arguments, and the files it reads or writes (relative to
    # the directory the test builds) that no report may name.
    COMMANDS = {
        "from-nifti": (
            [
                "{d}/out.medh5",
                "--image",
                "CT={d}/ct.nii.gz",
                "--mask",
                "m={d}/m.nii.gz",
                "--label-set",
                "{d}/labels.json",
            ],
            ["ct.nii.gz", "m.nii.gz", "labels.json", "out.medh5"],
        ),
        "import-seg-nifti": (
            ["{d}/case.medh5", "--mask", "m={d}/m.nii.gz"],
            ["case.medh5", "m.nii.gz"],
        ),
        "from-dicom": (["{d}/dcm", "{d}/out.medh5"], ["dcm/CT_000.dcm", "out.medh5"]),
        "from-dicom-seg": (
            ["{d}/seg.dcm", "{d}/case.medh5"],
            ["seg.dcm", "case.medh5"],
        ),
        "to-dicom-seg": (
            [
                "{d}/case.medh5",
                "organs",
                "{d}/out.dcm",
                "--source",
                "{d}/dcm/CT_000.dcm",
            ],
            ["case.medh5", "dcm/CT_000.dcm", "out.dcm"],
        ),
        "from-rtstruct": (["{d}/rt.dcm", "{d}/case.medh5"], ["rt.dcm", "case.medh5"]),
        "to-rtstruct": (
            ["{d}/case.medh5", "rois", "{d}/out.dcm", "--source", "{d}/dcm/CT_000.dcm"],
            ["case.medh5", "dcm/CT_000.dcm", "out.dcm"],
        ),
        "from-nnunet": (
            ["{d}/nn/D1", "{d}/imp"],
            ["nn/D1/dataset.json", "nn/D1/imagesTr/a_0000.nii.gz", "imp/a.medh5"],
        ),
        "to-nnunet": (
            ["{d}/nn2", "{d}/case.medh5", "--dataset-name", "D2"],
            ["case.medh5", "nn2/D2/dataset.json", "nn2/D2/imagesTr/case_0000.nii.gz"],
        ),
    }

    @classmethod
    def _tree(cls, root: Path) -> dict[str, bytes]:
        """Every input the commands name, as placeholders: no converter runs."""
        files = {
            "ct.nii.gz": b"original",
            "m.nii.gz": b"original",
            "labels.json": b"original",
            "seg.dcm": cls.DICOM,
            "rt.dcm": cls.DICOM,
            "case.medh5": b"original",
            "dcm/CT_000.dcm": cls.DICOM,
            "nn/D1/imagesTr/a_0000.nii.gz": b"original",
            "nn/D1/dataset.json": json.dumps(
                {
                    "channel_names": {"0": "CT"},
                    "labels": {"background": 0, "liver": 1},
                    "numTraining": 1,
                    "file_ending": ".nii.gz",
                }
            ).encode(),
        }
        for name, data in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(data)
        (root / "sub").mkdir()
        return files

    @staticmethod
    def _snapshot(root: Path) -> dict[str, bytes]:
        return {
            str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and not p.is_symlink()
        }

    def _refused(self, capsys, monkeypatch, root, command, argv) -> str:
        from medh5 import cli

        before = self._snapshot(root)
        ran: list[str] = []
        monkeypatch.setitem(cli._HANDLERS, command, lambda a: ran.append(command) or 0)
        code, printed = run(capsys, "convert", command, *argv)
        assert code == EXIT_ERROR, (command, argv, printed.err)
        assert not ran, "refused before the converter runs"
        assert self._snapshot(root) == before, "every file as it was"
        return printed.err

    @pytest.mark.parametrize(
        ("command", "target"),
        [(c, t) for c, (_, targets) in COMMANDS.items() for t in targets],
    )
    def test_no_report_names_a_file_the_command_reads_or_writes(
        self, capsys, monkeypatch, tmp_path, command, target
    ):
        self._tree(tmp_path)
        argv = [a.format(d=tmp_path) for a in self.COMMANDS[command][0]]
        err = self._refused(
            capsys,
            monkeypatch,
            tmp_path,
            command,
            [*argv, "--report", str(tmp_path / target)],
        )
        assert err.startswith(f"medh5: --report {tmp_path / target} is ")

    @pytest.mark.parametrize("alias", ["relative", "symlink", "hardlink"])
    def test_an_alias_of_an_input_is_the_input(
        self, capsys, monkeypatch, tmp_path, alias
    ):
        """`sub/../ct.nii.gz`, a link to it, or a second name for its inode:
        the report was written through each of them onto the source."""
        self._tree(tmp_path)
        source = tmp_path / "ct.nii.gz"
        if alias == "relative":
            report = tmp_path / "sub" / ".." / "ct.nii.gz"
        else:
            report = tmp_path / "sub" / "r.json"
            try:
                (report.symlink_to if alias == "symlink" else report.hardlink_to)(
                    source
                )
            except OSError as exc:  # pragma: no cover - no links on this file system
                pytest.skip(str(exc))
        argv = [a.format(d=tmp_path) for a in self.COMMANDS["from-nifti"][0]]
        self._refused(
            capsys,
            monkeypatch,
            tmp_path,
            "from-nifti",
            [*argv, "--report", str(report)],
        )
        assert source.read_bytes() == b"original"

    @pytest.mark.parametrize(
        "argv",
        [
            ["from-nifti", "{d}/ct.nii.gz", "--image", "CT={d}/ct.nii.gz"],
            ["to-nifti", "{d}/m.nii.gz", "CT", "{d}/m.nii.gz"],
            ["from-dicom", "{d}/dcm", "{d}/dcm/CT_000.dcm"],
            [
                "to-dicom-seg",
                "{d}/case.medh5",
                "o",
                "{d}/case.medh5",
                "--source",
                "{d}/seg.dcm",
            ],
            [
                "to-dicom-seg",
                "{d}/case.medh5",
                "o",
                "{d}/seg.dcm",
                "--source",
                "{d}/seg.dcm",
            ],
            [
                "to-rtstruct",
                "{d}/case.medh5",
                "o",
                "{d}/case.medh5",
                "--source",
                "{d}/rt.dcm",
            ],
            ["to-nnunet", "{d}/case.medh5", "{d}/case.medh5"],
        ],
    )
    def test_no_output_names_an_input(self, capsys, monkeypatch, tmp_path, argv):
        self._tree(tmp_path)
        command, *rest = (a.format(d=tmp_path) for a in argv)
        err = self._refused(capsys, monkeypatch, tmp_path, command, rest)
        assert err.startswith("medh5: the output ")

    def test_paths_that_alias_nothing_reach_the_converter(
        self, capsys, monkeypatch, tmp_path
    ):
        """The checks refuse aliases only: a report beside the output, a report
        inside the output directory, and a sample written into the DICOM tree
        over an earlier one --- no DICOM file --- all run."""
        from medh5 import cli

        self._tree(tmp_path)
        (tmp_path / "dcm" / "old.medh5").write_bytes(b"\x89HDF\r\n\x1a\n")
        ran: list[str] = []
        for command in ("from-dicom", "from-nnunet"):
            monkeypatch.setitem(
                cli._HANDLERS, command, lambda a: ran.append(a.out) or 0
            )
        for argv in (
            [
                "from-dicom",
                f"{tmp_path}/dcm",
                f"{tmp_path}/out.medh5",
                "--report",
                f"{tmp_path}/r.json",
            ],
            ["from-dicom", f"{tmp_path}/dcm", f"{tmp_path}/dcm/old.medh5"],
            [
                "from-nnunet",
                f"{tmp_path}/nn/D1",
                f"{tmp_path}/imp",
                "--report",
                f"{tmp_path}/imp/r.json",
            ],
        ):
            assert run(capsys, "convert", *argv)[0] == EXIT_OK, argv
        assert len(ran) == 3

    def test_S2_a_report_naming_an_output_known_only_after_is_not_written(
        self, capsys, monkeypatch, tmp_path
    ):
        """A `from-dicom` directory's samples are named for their subjects,
        which only the import learns: a report naming one is refused there,
        and the sample stays."""
        from medh5 import cli
        from medh5.io.report import ConversionReport

        sample = tmp_path / "out" / "PSEUDO-1.medh5"

        def convert(args: Any) -> int:
            sample.parent.mkdir()
            sample.write_bytes(b"the sample")
            report = ConversionReport(converter="from-dicom", outputs=[str(sample)])
            return cli._finish(report, args)

        monkeypatch.setitem(cli._HANDLERS, "from-dicom", convert)
        (tmp_path / "dcm").mkdir()
        code, printed = run(
            capsys, "convert", "from-dicom", str(tmp_path / "dcm"),
            f"{tmp_path / 'out'}{os.sep}", "--report", str(sample),
        )  # fmt: skip
        assert code == EXIT_ERROR
        assert "the report was not written" in printed.err
        assert sample.read_bytes() == b"the sample"

    def test_S14_4_a_report_keeps_a_link_and_replaces_its_target(
        self, capsys, monkeypatch, tmp_path
    ):
        """Until 2.1 a report replaced a link at its path, where every
        amendment followed it (§14.4); the report now goes through the
        engine's replace, so the link stays and the file it names is
        replaced.  A link planted at the temporary's name is passed over,
        which `s14_4_a_link_at_the_next_temporary_name_is_passed_over`
        holds: the name is the engine's."""
        from medh5 import cli
        from medh5.io.report import ConversionReport

        other = tmp_path / "other.txt"
        other.write_text("someone else's", encoding="utf-8")
        link = tmp_path / "r.json"
        try:
            link.symlink_to(other)
        except OSError as exc:  # pragma: no cover - no links on this file system
            pytest.skip(str(exc))
        monkeypatch.setitem(
            cli._HANDLERS,
            "from-nnunet",
            lambda a: cli._finish(ConversionReport(converter="from-nnunet"), a),
        )
        argv = ["from-nnunet", str(tmp_path / "x"), str(tmp_path / "y")]
        assert run(capsys, "convert", *argv, "--report", str(link))[0] == EXIT_OK
        assert link.is_symlink() and link.resolve() == other.resolve()
        assert (
            json.loads(other.read_text(encoding="utf-8"))["converter"] == "from-nnunet"
        )
        assert sorted(p.name for p in tmp_path.iterdir()) == ["other.txt", "r.json"]

    def test_a_report_naming_the_output_of_a_real_import_is_refused(
        self, capsys, tmp_path
    ):
        nib = pytest.importorskip("nibabel")
        source = tmp_path / "ct.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), np.int16), np.eye(4)), str(source))
        out = tmp_path / "out.medh5"
        code, printed = run(
            capsys, "convert", "from-nifti", str(out), "--image", f"CT={source}",
            "--report", str(out),
        )  # fmt: skip
        assert code == EXIT_ERROR
        assert "is the output this command writes" in printed.err
        assert not out.exists()


class TestConvert:
    """``convert`` and ``migrate``."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["from-dicom-seg", "seg.dcm", "case.medh5"],
            ["to-dicom-seg", "case.medh5", "seg", "seg.dcm", "--source", "1.dcm"],
            ["from-rtstruct", "rt.dcm", "case.medh5"],
            ["to-rtstruct", "case.medh5", "contours", "rt.dcm", "--source", "1.dcm"],
        ],
    )
    def test_S3_4_every_dicom_frame_check_takes_the_scrub_salt(self, monkeypatch, argv):
        """``from-rtstruct`` and ``to-dicom-seg`` compare frames of reference
        as their converters do, but had no ``--frame-salt`` to compare one a
        scrub pseudonymised."""
        from medh5 import cli

        handed: dict[str, Any] = {}
        monkeypatch.setitem(
            cli._HANDLERS, argv[0], lambda a: handed.update(vars(a)) or EXIT_OK
        )
        assert main(["convert", *argv, "--frame-salt", "s3cret"]) == EXIT_OK
        assert handed["frame_salt"] == "s3cret"

    @pytest.mark.parametrize(
        "argv",
        [
            ["to-nifti", "{shard}", "CT_tp0", "{tmp}/x.nii.gz"],
            ["to-dicom-seg", "{shard}", "organs_tp0", "{tmp}/x.dcm", "--source", "1"],
            ["to-rtstruct", "{shard}", "organs_tp0", "{tmp}/x.dcm", "--source", "1"],
            ["from-dicom-seg", "{tmp}/seg.dcm", "{shard}"],
            ["from-rtstruct", "{tmp}/rt.dcm", "{shard}"],
            ["to-nnunet", "{tmp}/nn", "{tmp}/m0.medh5", "{shard}"],
        ],
    )
    def test_S2_2_a_converter_handed_a_collection_names_the_way_to_a_member(
        self, capsys, tmp_path, argv
    ):
        """The engine refuses a collection in the Python API's words, and a
        converter on one told the command line to "open it with
        open_collection()"; it names `medh5 unpack`, before reading anything."""
        member = write_sample(tmp_path / "m0.medh5")
        shard = tmp_path / "s.medh5c"
        assert run(capsys, "pack", str(member), "-o", str(shard))[0] == EXIT_OK
        argv = [a.format(shard=shard, tmp=tmp_path) for a in argv]
        code, out = run(capsys, "convert", *argv)
        assert code == EXIT_ERROR
        assert out.err == (
            f"medh5: {shard} is a collection of 1 sample(s); this command works "
            f"on one sample file --- extract them with `medh5 unpack {shard} -o DIR`\n"
        )
        assert sorted(p.name for p in tmp_path.iterdir()) == ["m0.medh5", "s.medh5c"]
        # The native commands' words, not a copy of them that drifted ("works on
        # sample files"): one refusal, from the same function.
        assert run(capsys, "fix", str(shard))[1].err.startswith(out.err)

    @pytest.fixture
    def nifti(self, tmp_path):
        nib = pytest.importorskip("nibabel")
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        shape = (12, 10, 6)
        ct = tmp_path / "ct.nii.gz"
        mask = tmp_path / "liver.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(shape, np.int16), affine), str(ct))
        volume = np.zeros(shape, np.uint8)
        volume[2:6, 2:6, 1:3] = 1
        nib.save(nib.Nifti1Image(volume, affine), str(mask))
        return {"ct": ct, "mask": mask}

    def test_from_nifti_and_back(self, capsys, tmp_path, nifti):
        out = tmp_path / "case.medh5"
        code, printed = run(
            capsys,
            "convert",
            "from-nifti",
            str(out),
            "--image",
            f"CT={nifti['ct']}",
            "--mask",
            f"liver={nifti['mask']}",
            "--modality",
            "CT=CT",
        )
        assert code == EXIT_OK
        assert "DECISION" in printed.out
        assert out.exists()

        code, printed = run(
            capsys, "convert", "to-nifti", str(out), "CT", str(tmp_path / "b.nii.gz")
        )
        assert code == EXIT_OK
        assert (tmp_path / "b.nii.gz").exists()

    def test_S3_6_the_fourth_axis_can_be_settled_from_the_CLI(self, capsys, tmp_path):
        """`auto` guesses `time` for an unmarked series and says to override it.

        The override existed only in Python, so the one case the argument was
        added for could not be corrected by the interface people convert with.
        """
        nib = pytest.importorskip("nibabel")
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        source = tmp_path / "echo.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 4, 3), np.int16), affine), str(source))
        out = tmp_path / "echo.medh5"

        code, _ = run(
            capsys,
            "convert",
            "from-nifti",
            str(out),
            "--image",
            f"IM={source}",
            "--fourth-axis",
            "channel",
        )
        assert code == EXIT_OK
        with medh5.open(out) as sample:
            assert sample.grids["ref"].axis_kinds[0] == "channel"

    def test_the_report_is_written_and_machine_readable(self, capsys, tmp_path, nifti):
        report = tmp_path / "report.json"
        code, printed = run(
            capsys,
            "convert",
            "from-nifti",
            str(tmp_path / "c.medh5"),
            "--image",
            f"CT={nifti['ct']}",
            "--report",
            str(report),
            "--json",
        )
        assert code == EXIT_OK
        payload = json.loads(printed.out)
        assert payload["converter"] == "from-nifti"
        assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True

    def test_a_malformed_pair_is_reported(self, capsys, tmp_path):
        code, printed = run(
            capsys, "convert", "from-nifti", str(tmp_path / "x.medh5"), "--image", "CT"
        )
        assert code == EXIT_ERROR
        assert "NAME=VALUE" in printed.err

    def test_convert_without_a_subcommand_explains_itself(self, capsys):
        code, printed = run(capsys, "convert")
        assert code == EXIT_USAGE
        assert "Usage: medh5 convert" in printed.err
        assert "from-nifti" in printed.err

    def test_from_nnunet_and_to_nnunet(self, capsys, tmp_path):
        nib = pytest.importorskip("nibabel")
        root = tmp_path / "Dataset001_T"
        (root / "imagesTr").mkdir(parents=True)
        (root / "labelsTr").mkdir()
        shape = (8, 6, 4)
        for channel in range(1):
            nib.save(
                nib.Nifti1Image(np.zeros(shape, np.int16), np.eye(4)),
                str(root / "imagesTr" / f"C1_{channel:04d}.nii.gz"),
            )
        labels = np.zeros(shape, np.uint8)
        labels[1:4, 1:3, 1:2] = 1
        nib.save(
            nib.Nifti1Image(labels, np.eye(4)), str(root / "labelsTr" / "C1.nii.gz")
        )
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "channel_names": {"0": "CT"},
                    "labels": {"background": 0, "liver": 1},
                    "numTraining": 1,
                    "file_ending": ".nii.gz",
                }
            ),
            encoding="utf-8",
        )
        code, printed = run(
            capsys, "convert", "from-nnunet", str(root), str(tmp_path / "out")
        )
        assert code == EXIT_OK
        assert (tmp_path / "out" / "C1.medh5").exists()

        code, printed = run(
            capsys,
            "convert",
            "to-nnunet",
            str(tmp_path / "back"),
            str(tmp_path / "out" / "C1.medh5"),
        )
        assert code == EXIT_OK
        assert (tmp_path / "back" / "Dataset001_medh5" / "dataset.json").exists()

    def test_migrate_writes_labels_then_migrates(self, capsys, tmp_path):
        shape = (6, 8, 10)
        mask = np.zeros(shape, bool)
        mask[1:4, 2:6, 3:8] = True
        old = tmp_path / "old.medh5"
        write_legacy_sample(
            old,
            images={"CT": np.zeros(shape, np.int16)},
            seg={"liver": mask},
            spacing=[1.0, 1.0, 1.0],
        )
        labels = tmp_path / "labels.json"
        code, printed = run(
            capsys,
            "migrate",
            str(old),
            "-o",
            str(tmp_path / "m"),
            "--write-labels",
            str(labels),
        )
        assert code == EXIT_OK
        assert "review before migrating" in printed.out
        assert json.loads(labels.read_text(encoding="utf-8"))["classes"]
        # With `--json` the note goes to stderr and stdout is the report alone.
        code, printed = run(
            capsys,
            "migrate",
            str(old),
            "-o",
            str(tmp_path / "m"),
            "--write-labels",
            str(labels),
            "--json",
        )
        assert code == EXIT_OK
        assert json.loads(printed.out)["converter"] == "migrate"
        assert "review before migrating" in printed.err

        code, printed = run(
            capsys,
            "migrate",
            str(old),
            "-o",
            str(tmp_path / "m"),
            "--label-set",
            str(labels),
        )
        assert code == EXIT_OK
        written = sorted((tmp_path / "m").glob("*.medh5"))
        assert len(written) == 1
        with medh5.open(written[0]) as sample:
            assert sample.annotations["seg"].class_ids

    def test_migrate_reports_an_unreadable_file(self, capsys, tmp_path):
        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not hdf5")
        code, printed = run(capsys, "migrate", str(broken), "-o", str(tmp_path / "m"))
        assert code == EXIT_ERROR
        assert "WARNING" in printed.out


class TestResultTypesAreTheEngines:
    """What a result type says of itself --- its verdict, its counts, its text
    and its JSON --- is the engine's, which the command line prints.  Each was
    computed twice, in Rust and in Python, and the copies drifted: a
    measurement's NaN was ``null`` from one and ``NaN`` from the other."""

    @staticmethod
    def _printed(capsys, *argv: str) -> Any:
        _, out = run(capsys, *argv, "--json")
        return json.loads(out.out)

    def test_S15_1_a_report_says_what_validate_prints(self, capsys, sample_path):
        from medh5.validate import Report, validate_file

        for level in ("semantic", "strict"):
            report = validate_file(sample_path, level=level)
            assert report.diagnostics
            argv = ("validate", str(sample_path), "--level", level)
            assert [report.to_json()] == self._printed(capsys, *argv)
            _, out = run(capsys, *argv, "-v")
            assert out.out.rstrip("\n") == report.format(verbose=True)
            again = Report.from_json(report.to_json())
            assert again.to_json() == report.to_json()
            assert again.format() == report.format()
            assert repr(again) == repr(report)

    def test_S15_1_a_report_built_by_hand_counts_as_the_engine_does(
        self, capsys, sample_path
    ):
        """At ``strict`` a warning is an error in the counts too; a report the
        caller assembles is judged by the same rule as the validator's."""
        from medh5.validate import Diagnostic, Report, merge

        printed = self._printed(
            capsys, "validate", str(sample_path), "--level", "strict"
        )[0]
        assert printed["warnings"] == 0 and printed["promoted"] > 0
        mine = Report(
            path=printed["path"],
            level="strict",
            profiles=tuple(printed["profiles"]),
            diagnostics=[Diagnostic.from_json(d) for d in printed["diagnostics"]],
            checked=printed["checked"],
        )
        assert (len(mine.errors), len(mine.warnings), len(mine.promoted)) == (
            printed["errors"],
            printed["warnings"],
            printed["promoted"],
        )
        assert mine.ok is printed["ok"] is False
        warning = Diagnostic("W903", "/x", "advice", severity="warning")
        lenient = Report("p", level="semantic", diagnostics=[warning])
        assert (lenient.errors, lenient.warnings, lenient.promoted) == (
            (),
            (warning,),
            (),
        )
        assert lenient.ok
        strict = Report("p", level="strict", diagnostics=[warning])
        assert strict.errors == strict.promoted == (warning,)
        assert strict.errors[0] is warning
        merged = merge([lenient, strict], "both")
        assert [d.location for d in merged.diagnostics] == ["p:/x", "p:/x"]
        assert merged.level == "semantic"

    def test_S13_a_verification_says_what_verify_prints(self, capsys, sample_path):
        from medh5.integrity import VerifyResult

        with medh5.open(sample_path) as s:
            result = s.verify()
        (printed,) = self._printed(capsys, "verify", str(sample_path))
        assert printed.pop("path") == str(sample_path)
        assert result.summary() == printed
        assert result.ok is True and result.content_id_ok is True
        broken = VerifyResult(
            content_id_declared="sha256:a", content_id_computed="sha256:b"
        )
        assert broken.content_id_ok is False and broken.ok is False
        assert VerifyResult().content_id_ok is None

    def test_S13_3_a_diagnosis_and_a_repair_say_what_fix_prints(
        self, capsys, sample_path
    ):
        from medh5.integrity import Diagnosis, Repair, diagnose, fix

        diagnosis = diagnose(sample_path)
        (printed,) = self._printed(capsys, "fix", str(sample_path))
        assert printed["diagnosis"] == diagnosis.to_json()
        assert Diagnosis.from_json(diagnosis.to_json()) == diagnosis
        assert diagnosis.clean and not diagnosis.needs_index
        repair = fix(sample_path)
        assert Repair.from_json(repair.to_json()).to_json() == repair.to_json()
        assert repair.changed is False
        stale = Diagnosis("p", stale_index=("organs_tp0",), content_id_ok=False)
        assert (stale.needs_index, stale.needs_digests, stale.clean) == (
            True,
            True,
            False,
        )
        assert Repair("p", stale, rewrote_digests=True).changed

    def test_S14_4_a_recompression_says_what_recompress_prints(
        self, capsys, sample_path, tmp_path
    ):
        import shutil

        from medh5.storage import RecompressResult, recompress

        twin = tmp_path / "twin.medh5"
        shutil.copyfile(sample_path, twin)
        result = recompress(sample_path, "archive")
        (printed,) = self._printed(
            capsys, "recompress", str(twin), "--profile", "archive"
        )
        assert printed.pop("path") == str(twin)
        mine = result.to_json()
        assert mine.pop("path") == str(sample_path)
        assert mine == printed
        again = RecompressResult.from_json(result.to_json())
        assert again == result and str(again) == str(result)
        assert str(result) == (
            f"{result.path}: archive, {result.datasets} datasets, "
            f"{result.bytes_before} -> {result.bytes_after} bytes "
            f"({result.ratio:.2f}×)"
        )
        assert RecompressResult("p", "archive").ratio == 1.0

    def test_a_conformance_summary_is_the_engines(self, capsys, tmp_path):
        from medh5.conformance import CASES, CaseResult, run_corpus, summarize

        results = run_corpus(tmp_path / "py", names=["core-minimal"])
        printed = self._printed(
            capsys,
            "conformance",
            "run",
            str(tmp_path / "cli"),
            "--case",
            "core-minimal",
        )
        assert {k: printed[k] for k in summarize(results)} == summarize(results)
        failing = CaseResult(case=CASES[0], path="p", missing=("E101",))
        assert failing.ok is False
        assert summarize([*results, failing])["failures"] == [failing.to_json()]
        assert [c.to_json()["valid"] for c in CASES] == [c.valid for c in CASES]

    def test_a_measurement_of_nan_is_null_as_the_engine_writes_it(self):
        from medh5.bench import Measurement

        nan = Measurement("draw", float("nan"), target=1.0)
        assert nan.to_json()["value"] is None
        assert nan.ok is False
        assert "draw" in str(nan)
        back = Measurement.from_json(nan.to_json())
        assert math.isnan(back.value) and back.target == 1.0
        met = Measurement("draw", 0.5, target=1.0, detail={"n": 3})
        assert Measurement.from_json(met.to_json()) == met
        assert met.ok and met.to_json()["ok"] is True


def _clinical_export(inputs: Path, target: Path) -> list[str]:
    from tests.kits import History

    history = inputs / "history.medh5"
    History.write(history)
    return ["clinical", "export", str(history), "-o", str(target)]


def _migrate_report(inputs: Path, target: Path) -> list[str]:
    old = inputs / "old.medh5"
    write_legacy_sample(old, images={"CT": np.zeros((4, 4, 4), np.int16)})
    return ["migrate", str(old), "-o", str(inputs / "m"), "--report", str(target)]


def _cohort_manifest(inputs: Path) -> Path:
    """A manifest of three subjects, one sample each."""
    for subject in ("A", "B", "C"):
        path = write_sample(inputs / f"{subject}.medh5", sample_id=f"case-{subject}")
        with medh5.amend(path) as writer:
            writer.identity(subject_id=f"subj-{subject}")
    manifest = inputs / "m.json"
    assert main(["dataset", "index", str(inputs), "-o", str(manifest)]) == EXIT_OK
    return manifest


def _dataset_split(inputs: Path, target: Path) -> list[str]:
    return ["dataset", "split", str(_cohort_manifest(inputs)), "-o", str(target)]


def _dataset_stats(inputs: Path, target: Path) -> list[str]:
    return ["dataset", "stats", str(_cohort_manifest(inputs)), "-o", str(target)]


# Every file the command line writes, each through the engine's replace.
_COMMAND_SITES = {
    "clinical export -o": _clinical_export,
    "migrate --report": _migrate_report,
    "dataset split -o": _dataset_split,
    "dataset stats -o": _dataset_stats,
}


@pytest.mark.parametrize("site", _COMMAND_SITES.values(), ids=list(_COMMAND_SITES))
class TestCommandLineReplaces:
    """Every file the command line writes is replaced as an amendment
    replaces a sample (§14.4, D23): until 2.1 these four wrote in place,
    through a link and not atomically, where `convert --report` replaced
    one."""

    @staticmethod
    def _write(tmp_path: Path, target: Path, site: Any) -> bytes:
        inputs = tmp_path / "in"
        inputs.mkdir()
        assert main(site(inputs, target)) == EXIT_OK
        return target.read_bytes()

    def test_S14_4_a_link_is_kept_and_the_file_it_names_replaced(
        self, tmp_path: Path, site: Any
    ):
        real = tmp_path / "data" / "real.json"
        real.parent.mkdir()
        real.write_bytes(b"before\n")
        link = tmp_path / "given.json"
        try:
            link.symlink_to(Path("data", "real.json"))
        except OSError as exc:  # pragma: no cover - no links on this file system
            pytest.skip(str(exc))
        inode = real.stat().st_ino
        written = self._write(tmp_path, link, site)
        assert link.is_symlink() and os.readlink(link) == str(Path("data", "real.json"))
        assert real.read_bytes() == written and json.loads(written)
        assert real.stat().st_ino != inode  # replaced, not written in place
        assert [p.name for p in real.parent.iterdir()] == ["real.json"]
        beside = sorted(p.name for p in tmp_path.iterdir())
        assert beside == ["data", "given.json", "in"]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
    def test_S14_4_the_file_s_mode_is_kept(self, tmp_path: Path, site: Any):
        target = tmp_path / "private.json"
        target.write_bytes(b"before\n")
        target.chmod(0o600)
        assert json.loads(self._write(tmp_path, target, site))
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_S14_4_a_read_only_file_is_replaced_and_stays_read_only(
        self, tmp_path: Path, site: Any
    ):
        target = tmp_path / "read-only.json"
        target.write_bytes(b"before\n")
        target.chmod(stat.S_IREAD)
        try:
            assert json.loads(self._write(tmp_path, target, site))
            assert not target.stat().st_mode & stat.S_IWRITE
        finally:
            target.chmod(stat.S_IREAD | stat.S_IWRITE)

    def test_S14_4_lines_end_in_a_line_feed_on_every_platform(
        self, tmp_path: Path, site: Any
    ):
        written = self._write(tmp_path, tmp_path / "lines.json", site)
        assert written.endswith(b"\n") and b"\r" not in written


# The commands that take ``--workers``, with options that make each do work.
WORKER_COMMANDS = {
    "validate": ["validate"],
    "verify": ["verify"],
    "fix": ["fix", "--rebuild-index"],
    "index-build": ["index", "build"],
    "recompress": ["recompress", "--profile", "balanced"],
}


class TestWorkers:
    """``--workers N`` runs a batch command's files in up to N processes,
    and prints what one process prints, byte for byte: each file's output in
    input order, the same ``--json`` document, the same exit code."""

    @pytest.fixture
    def batch(self, tmp_path, label_set, masks) -> dict[str, bytes]:
        """Five files as bytes: three clean, one whose digest no longer
        matches its data, and one that is not HDF5 at all."""
        made = tmp_path / "made"
        made.mkdir()
        for name in ("a", "b", "c", "d"):
            write_sample(made / f"{name}.medh5", label_set=label_set, masks=masks)
        with h5py.File(made / "d.medh5", "r+") as handle:
            data = handle["annotations/organs_tp0/data"]
            data[0, 0, 0, 0] = ~data[0, 0, 0, 0]
        (made / "e.medh5").write_bytes(b"not an HDF5 file")
        return {path.name: path.read_bytes() for path in sorted(made.iterdir())}

    @staticmethod
    def fresh(tmp_path: Path, batch: dict[str, bytes], names: Any = None) -> list[str]:
        """The batch written again, at the same paths, for a run of its own."""
        where = tmp_path / "run"
        where.mkdir(exist_ok=True)
        for name, data in batch.items():
            (where / name).write_bytes(data)
        return [str(where / name) for name in (names or batch)]

    def runs(self, capsys, tmp_path, batch, argv, names=None) -> set[Any]:
        seen = set()
        for workers in (None, 1, 2, 3):
            paths = self.fresh(tmp_path, batch, names)
            flags = [] if workers is None else ["--workers", str(workers)]
            code, printed = run(capsys, *argv, *paths, *flags)
            seen.add((code, printed.out, printed.err))
        return seen

    @pytest.mark.parametrize("command", list(WORKER_COMMANDS))
    def test_output_does_not_depend_on_the_worker_count(
        self, capsys, tmp_path, batch, command
    ):
        argv = WORKER_COMMANDS[command]
        for extra in ([], ["--json"]):
            seen = self.runs(capsys, tmp_path, batch, [*argv, *extra])
            assert len(seen) == 1, sorted(seen)
            ((code, out, _),) = seen
            assert code == EXIT_ERROR
            assert out

    def test_the_exit_code_does_not_depend_on_the_worker_count(
        self, capsys, tmp_path, batch
    ):
        clean = ["a.medh5", "b.medh5", "c.medh5"]
        seen = self.runs(capsys, tmp_path, batch, ["verify"], clean)
        assert {code for code, _, _ in seen} == {EXIT_OK}
        assert len(seen) == 1
        seen = self.runs(capsys, tmp_path, batch, ["verify"], [*clean, "d.medh5"])
        assert {code for code, _, _ in seen} == {EXIT_ERROR}
        assert len(seen) == 1

    def test_a_file_named_twice_is_written_by_one_worker(self, capsys, tmp_path, batch):
        """A file named twice, or by a second name (a hard link), goes to one
        worker, in order: two workers writing it at once would race, and the
        second name's report would depend on which won."""
        where = tmp_path / "run"
        names = ["a.medh5", "a.medh5", "b.medh5", "link.medh5", "c.medh5"]
        seen = set()
        for workers in ("1", "3"):
            paths = self.fresh(tmp_path, batch, names[:3])
            link = where / "link.medh5"
            link.unlink(missing_ok=True)
            os.link(where / "b.medh5", link)
            paths += [str(link), str(where / "c.medh5")]
            code, printed = run(capsys, "index", "build", *paths, "--workers", workers)
            seen.add((code, printed.out, printed.err))
            assert code == EXIT_OK
            assert run(capsys, "verify", *paths)[0] == EXIT_OK
        assert len(seen) == 1
        ((_, out, _),) = seen
        lines = out.splitlines()
        assert lines[0].endswith("a.medh5: built index for organs_tp0")
        assert lines[1].endswith(
            "a.medh5: index current for organs_tp0; the file is unchanged"
        )

    def test_a_dying_worker_fails_only_its_file(
        self, capsys, tmp_path, batch, monkeypatch
    ):
        monkeypatch.setenv("MEDH5_TEST_WORKERS", "exit:b.medh5")
        paths = self.fresh(tmp_path, batch, ["a.medh5", "b.medh5", "c.medh5"])
        code, printed = run(capsys, "verify", *paths, "--workers", "2")
        assert code == EXIT_ERROR
        lines = printed.out.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith(f"{paths[0]}: OK")
        assert lines[1].startswith(f"{paths[2]}: OK")
        assert f"{paths[1]}: the worker running it stopped" in printed.err
        assert len(printed.err.splitlines()) == 1

    def test_an_error_stops_where_sequential_stops(
        self, capsys, tmp_path, batch, monkeypatch
    ):
        monkeypatch.setenv("MEDH5_TEST_WORKERS", "error:c.medh5")
        names = ["a.medh5", "b.medh5", "c.medh5", "d.medh5", "e.medh5"]
        seen = self.runs(capsys, tmp_path, batch, ["verify"], names)
        assert len(seen) == 1
        ((code, out, err),) = seen
        assert code == EXIT_ERROR
        lines = out.splitlines()
        assert [Path(line.split(": OK")[0]).name for line in lines] == [
            "a.medh5",
            "b.medh5",
        ]
        assert err.strip().endswith("c.medh5: stopped here, as MEDH5_TEST_WORKERS asks")

    def test_zero_workers_is_one_process(self, capsys, tmp_path, batch):
        paths = self.fresh(tmp_path, batch, ["a.medh5", "b.medh5"])
        assert run(capsys, "verify", *paths, "--workers", "0") == run(
            capsys, "verify", *paths
        )
        code, printed = run(capsys, "verify", *paths, "--workers", "-1")
        assert code == EXIT_USAGE


# -- the converter command line: one typed flag per parameter -----------------------


def _nifti_case(tmp_path: Path, *masks: str) -> dict[str, Path]:
    """A 12×10×6 CT and one mask per name of *masks* on its grid, as NIfTI."""
    nib = pytest.importorskip("nibabel")
    tmp_path.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    shape = (12, 10, 6)
    files = {"ct": tmp_path / "ct.nii.gz"}
    nib.save(nib.Nifti1Image(np.zeros(shape, np.int16), affine), str(files["ct"]))
    for i, name in enumerate(masks):
        volume = np.zeros(shape, np.uint8)
        volume[2 + i : 6, 2:6, 1:3] = 1
        files[name] = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(volume, affine), str(files[name]))
    return files


def _dicom(tmp_path: Path) -> Any:
    """A 6×16×16 CT series of 1 mm voxels and its sample, with no label set
    (``tests.integrations.test_rtstruct``'s case)."""
    pytest.importorskip("pydicom")
    from tests.integrations.test_rtstruct import _dicom_case

    return _dicom_case(tmp_path)


def _with_label_set(path: Path, *classes: tuple[int, str]) -> None:
    with medh5.amend(path) as writer:
        writer.label_set(
            LabelSet(
                "given",
                version="1.0.0",
                classes=[LabelClass(i, key, key.title()) for i, key in classes],
            )
        )


def _liver(grid: Any) -> np.ndarray:
    mask = np.zeros(grid.spatial_shape, bool)
    mask[1:4, 3:9, 4:10] = True
    return mask


def _organs(case: Any) -> None:
    """Annotation ``organs`` on *case*'s sample: liver (1) and lesion (3)."""
    _with_label_set(case.path, (1, "liver"), (3, "lesion"))
    lesion = np.zeros(case.grid.spatial_shape, bool)
    lesion[2, 5:7, 6:8] = True
    with medh5.amend(case.path) as writer:
        writer.add_segmentation(
            "organs",
            grid=case.grid.grid_id,
            masks={1: _liver(case.grid), 3: lesion},
        )


def _liver_seg(case: Any, out: Path) -> Path:
    """A DICOM SEG of one segment, ``liver``, over *case*'s series, written
    from a copy of its sample (which keeps *case*'s without a label set)."""
    pytest.importorskip("highdicom")
    from medh5.io.dicom import from_dicom
    from medh5.io.dicom_seg import to_dicom_seg

    source = out.parent / "seg-source.medh5"
    from_dicom(Path(case.series["paths"][0]).parent, source, group_by="study")
    with medh5.open(source) as sample:
        grid = sample.grids[sorted(sample.grids)[0]]
    _with_label_set(source, (1, "liver"))
    with medh5.amend(source) as writer:
        writer.add_segmentation("organs", grid=grid.grid_id, masks={1: _liver(grid)})
    to_dicom_seg(source, "organs", case.series["paths"], out)
    return out


def _gtv_rtstruct(case: Any, out: Path) -> Path:
    """An RTSTRUCT on *case*'s frame: ROI 1, ``GTV``, a square on plane 2."""
    from tests.integrations.test_rtstruct import (
        CLOSED,
        FRAME_A,
        _square,
        _structure_set,
    )

    square = _square(case.grid, 2, (3, 8), (3, 8))
    return _structure_set(out, {1: ("GTV", [(CLOSED, square)])}, [FRAME_A])


def _nnunet_dataset(tmp_path: Path) -> Path:
    nib = pytest.importorskip("nibabel")
    root = tmp_path / "Dataset001_T"
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    shape = (8, 6, 4)
    nib.save(
        nib.Nifti1Image(np.zeros(shape, np.int16), np.eye(4)),
        str(root / "imagesTr" / "C1_0000.nii.gz"),
    )
    labels = np.zeros(shape, np.uint8)
    labels[1:4, 1:3, 1:2] = 1
    nib.save(nib.Nifti1Image(labels, np.eye(4)), str(root / "labelsTr" / "C1.nii.gz"))
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"0": "CT"},
                "labels": {"background": 0, "liver": 1},
                "numTraining": 1,
                "file_ending": ".nii.gz",
            }
        ),
        encoding="utf-8",
    )
    return root


def _ok(capsys: Any, *argv: str) -> Any:
    code, printed = run(capsys, *argv)
    assert code == EXIT_OK, (argv, printed.out, printed.err)
    return printed


def _sources(case: Any) -> list[str]:
    return [a for p in case.series["paths"] for a in ("--source", p)]


def _examined(path: Path, ann_id: str) -> set[str]:
    with medh5.open(path) as sample:
        ann = sample.annotations[ann_id]
        return {ann.class_key(c) for c in ann.annotated_class_ids}


def _flag_annotated_classes_from_nifti(tmp_path: Path, capsys: Any) -> None:
    files = _nifti_case(tmp_path, "liver")
    out = tmp_path / "case.medh5"
    _ok(
        capsys, "convert", "from-nifti", str(out), "--image", f"CT={files['ct']}",
        "--mask", f"liver={files['liver']}", "--annotated-classes", "liver,tumor",
    )  # fmt: skip
    assert _examined(out, "seg") == {"liver", "tumor"}


def _flag_annotated_classes_import_seg_nifti(tmp_path: Path, capsys: Any) -> None:
    files = _nifti_case(tmp_path, "liver")
    out = tmp_path / "case.medh5"
    _ok(capsys, "convert", "from-nifti", str(out), "--image", f"CT={files['ct']}")
    _ok(
        capsys, "convert", "import-seg-nifti", str(out),
        "--mask", f"liver={files['liver']}", "--annotated-classes", "liver,tumor",
    )  # fmt: skip
    assert _examined(out, "seg") == {"liver", "tumor"}


def _flag_annotated_classes_from_dicom_seg(tmp_path: Path, capsys: Any) -> None:
    case = _dicom(tmp_path)
    seg = _liver_seg(case, tmp_path / "liver.dcm")
    _with_label_set(case.path, (1, "liver"), (3, "lesion"))
    _ok(
        capsys, "convert", "from-dicom-seg", str(seg), str(case.path),
        "--annotated-classes", "all",
    )  # fmt: skip
    assert _examined(case.path, "seg") == {"liver", "lesion"}


def _flag_annotated_classes_from_rtstruct(tmp_path: Path, capsys: Any) -> None:
    case = _dicom(tmp_path)
    rt = _gtv_rtstruct(case, tmp_path / "rt.dcm")
    _with_label_set(case.path, (1, "gtv"), (2, "bladder"))
    _ok(
        capsys, "convert", "from-rtstruct", str(rt), str(case.path),
        "--annotated-classes", "all",
    )  # fmt: skip
    assert _examined(case.path, "contours") == {"gtv", "bladder"}


def _portable(path: Path) -> None:
    from medh5.storage import profile_family

    assert profile_family(path) == "portable"


def _flag_codec_from_nifti(tmp_path: Path, capsys: Any) -> None:
    files = _nifti_case(tmp_path)
    out = tmp_path / "case.medh5"
    _ok(
        capsys, "convert", "from-nifti", str(out), "--image", f"CT={files['ct']}",
        "--codec", "portable",
    )  # fmt: skip
    _portable(out)


def _flag_codec_from_dicom(tmp_path: Path, capsys: Any) -> None:
    case = _dicom(tmp_path)
    out = tmp_path / "portable.medh5"
    root = Path(case.series["paths"][0]).parent
    _ok(capsys, "convert", "from-dicom", str(root), str(out), "--codec", "portable")
    _portable(out)


def _flag_codec_from_nnunet(tmp_path: Path, capsys: Any) -> None:
    root = _nnunet_dataset(tmp_path)
    out = tmp_path / "out"
    _ok(capsys, "convert", "from-nnunet", str(root), str(out), "--codec", "portable")
    _portable(out / "C1.medh5")


def _flag_codec_migrate(tmp_path: Path, capsys: Any) -> None:
    shape = (6, 8, 10)
    mask = np.zeros(shape, bool)
    mask[1:4, 2:6, 3:8] = True
    old = tmp_path / "old.medh5"
    write_legacy_sample(
        old, images={"CT": np.zeros(shape, np.int16)}, seg={"liver": mask},
        spacing=[1.0, 1.0, 1.0],
    )  # fmt: skip
    out = tmp_path / "m"
    _ok(capsys, "migrate", str(old), "-o", str(out), "--codec", "portable")
    (written,) = sorted(out.glob("*.medh5"))
    _portable(written)


def _flag_class_to_dicom_seg(tmp_path: Path, capsys: Any) -> None:
    pytest.importorskip("highdicom")
    import pydicom

    case = _dicom(tmp_path)
    _organs(case)
    out = tmp_path / "seg.dcm"
    _ok(
        capsys, "convert", "to-dicom-seg", str(case.path), "organs", str(out),
        *_sources(case), "--class", "lesion",
    )  # fmt: skip
    written = pydicom.dcmread(out)
    assert [s.SegmentLabel for s in written.SegmentSequence] == ["lesion"]


def _flag_series_description_to_dicom_seg(tmp_path: Path, capsys: Any) -> None:
    pytest.importorskip("highdicom")
    import pydicom

    case = _dicom(tmp_path)
    _organs(case)
    out = tmp_path / "seg.dcm"
    _ok(
        capsys, "convert", "to-dicom-seg", str(case.path), "organs", str(out),
        *_sources(case), "--series-description", "liver, reviewed",
    )  # fmt: skip
    assert pydicom.dcmread(out).SeriesDescription == "liver, reviewed"


def _flag_label_to_rtstruct(tmp_path: Path, capsys: Any) -> None:
    import pydicom

    case = _dicom(tmp_path)
    rt = _gtv_rtstruct(case, tmp_path / "rt.dcm")
    _ok(capsys, "convert", "from-rtstruct", str(rt), str(case.path))
    out = tmp_path / "out.dcm"
    _ok(
        capsys, "convert", "to-rtstruct", str(case.path), "contours", str(out),
        *_sources(case), "--label", "plan B",
    )  # fmt: skip
    assert pydicom.dcmread(out).StructureSetLabel == "plan B"


def _flag_mask_id_from_rtstruct(tmp_path: Path, capsys: Any) -> None:
    case = _dicom(tmp_path)
    rt = _gtv_rtstruct(case, tmp_path / "rt.dcm")
    _ok(
        capsys, "convert", "from-rtstruct", str(rt), str(case.path),
        "--rasterize", "--mask-id", "raster",
    )  # fmt: skip
    with medh5.open(case.path) as sample:
        assert sorted(sample.annotations) == ["contours", "raster"]
        assert sample.annotations["raster"].dense(["gtv"])[0].sum() == 36


def _flag_coord_system_from_nnunet(tmp_path: Path, capsys: Any) -> None:
    root = _nnunet_dataset(tmp_path)
    out = tmp_path / "out"
    _ok(capsys, "convert", "from-nnunet", str(root), str(out), "--coord-system", "RAS")
    with medh5.open(out / "C1.medh5") as sample:
        assert {g.coord_system for g in sample.grids.values()} == {"RAS"}


def _flag_file_ending_to_nnunet(tmp_path: Path, capsys: Any) -> None:
    files = _nifti_case(tmp_path, "liver")
    case = tmp_path / "case.medh5"
    _ok(
        capsys, "convert", "from-nifti", str(case), "--image", f"CT={files['ct']}",
        "--mask", f"liver={files['liver']}",
    )  # fmt: skip
    out = tmp_path / "nn"
    _ok(capsys, "convert", "to-nnunet", str(out), str(case), "--file-ending", ".nii")
    dataset = out / "Dataset001_medh5"
    assert sorted(p.name for p in (dataset / "imagesTr").iterdir()) == ["case_0000.nii"]
    document = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    assert document["file_ending"] == ".nii"


def _flag_value_units_from_nifti(tmp_path: Path, capsys: Any) -> None:
    files = _nifti_case(tmp_path)
    out = tmp_path / "case.medh5"
    _ok(
        capsys, "convert", "from-nifti", str(out), "--image", f"CT={files['ct']}",
        "--value-units", "CT=HU",
    )  # fmt: skip
    with medh5.open(out) as sample:
        assert sample.images["CT"].value_units == "HU"


def _flag_label_set_from_nifti(tmp_path: Path, capsys: Any) -> None:
    files = _nifti_case(tmp_path, "liver")
    labels = tmp_path / "labels.json"
    given = LabelSet(
        "abdomen", version="2.0.0", classes=[LabelClass(7, "liver", "Liver")]
    )
    labels.write_text(json.dumps(given.to_json()), encoding="utf-8")
    out = tmp_path / "case.medh5"
    _ok(
        capsys, "convert", "from-nifti", str(out), "--image", f"CT={files['ct']}",
        "--mask", f"liver={files['liver']}", "--label-set", str(labels),
    )  # fmt: skip
    with medh5.open(out) as sample:
        assert sample.label_set.id == "abdomen"
        assert sample.annotations["seg"].class_ids == (7,)


def _exported(tmp_path: Path, capsys: Any) -> tuple[Path, Path]:
    files = _nifti_case(tmp_path)
    case = tmp_path / "case.medh5"
    _ok(capsys, "convert", "from-nifti", str(case), "--image", f"CT={files['ct']}")
    return case, tmp_path / "back.nii.gz"


def _flag_report_to_nifti(tmp_path: Path, capsys: Any) -> None:
    case, out = _exported(tmp_path, capsys)
    report = tmp_path / "report.json"
    argv = ["convert", "to-nifti", str(case), "CT", str(out), "--report", str(report)]
    printed = _ok(capsys, *argv)
    assert printed.out == f"{out}\n", "stdout is the written path, as before"
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["converter"] == "to-nifti"
    assert written["outputs"] == [str(out)]


def _flag_json_to_nifti(tmp_path: Path, capsys: Any) -> None:
    case, out = _exported(tmp_path, capsys)
    printed = _ok(capsys, "convert", "to-nifti", str(case), "CT", str(out), "--json")
    payload = json.loads(printed.out)
    assert payload["converter"] == "to-nifti"
    assert payload["outputs"] == [str(out)]
    assert out.exists()


_CONVERT_FLAGS = {
    "annotated-classes/from-nifti": _flag_annotated_classes_from_nifti,
    "annotated-classes/from-dicom-seg": _flag_annotated_classes_from_dicom_seg,
    "annotated-classes/from-rtstruct": _flag_annotated_classes_from_rtstruct,
    "annotated-classes/import-seg-nifti": _flag_annotated_classes_import_seg_nifti,
    "codec/from-nifti": _flag_codec_from_nifti,
    "codec/from-dicom": _flag_codec_from_dicom,
    "codec/from-nnunet": _flag_codec_from_nnunet,
    "codec/migrate": _flag_codec_migrate,
    "class/to-dicom-seg": _flag_class_to_dicom_seg,
    "series-description/to-dicom-seg": _flag_series_description_to_dicom_seg,
    "label/to-rtstruct": _flag_label_to_rtstruct,
    "mask-id/from-rtstruct": _flag_mask_id_from_rtstruct,
    "coord-system/from-nnunet": _flag_coord_system_from_nnunet,
    "file-ending/to-nnunet": _flag_file_ending_to_nnunet,
    "value-units/from-nifti": _flag_value_units_from_nifti,
    "label-set/from-nifti": _flag_label_set_from_nifti,
    "report/to-nifti": _flag_report_to_nifti,
    "json/to-nifti": _flag_json_to_nifti,
}


@pytest.mark.parametrize("row", sorted(_CONVERT_FLAGS))
def test_convert_flags_reach_the_converter(tmp_path, capsys, row):
    """Each converter parameter the Python API takes has its own flag, and
    the flag changes what is written: the converters took more than the
    command line could say, so a cohort converted from it had no way to say
    which classes were examined, which codec to write, or what to name a
    SEG's series."""
    _CONVERT_FLAGS[row](tmp_path, capsys)


def test_mask_id_needs_rasterize(capsys, tmp_path):
    code, printed = run(
        capsys, "convert", "from-rtstruct", str(tmp_path / "rt.dcm"),
        str(tmp_path / "c.medh5"), "--mask-id", "m",
    )  # fmt: skip
    assert code == EXIT_USAGE
    assert "--rasterize" in printed.err


def _mint_nifti(tmp_path: Path, annotated: list[str]) -> tuple[Path, str, Any]:
    from medh5.io.nifti import from_nifti

    files = _nifti_case(tmp_path, "liver")
    out = tmp_path / "case.medh5"
    report = from_nifti(
        {"CT": files["ct"]}, out, masks={"liver": files["liver"]},
        annotated_classes=annotated,
    )  # fmt: skip
    return out, "seg", report


def _mint_import_seg_nifti(
    tmp_path: Path, annotated: list[str]
) -> tuple[Path, str, Any]:
    from medh5.io.nifti import from_nifti, import_seg_nifti

    files = _nifti_case(tmp_path, "liver")
    out = tmp_path / "case.medh5"
    from_nifti({"CT": files["ct"]}, out)
    report = import_seg_nifti(
        out, {"liver": files["liver"]}, annotated_classes=annotated
    )
    return out, "seg", report


def _mint_seg(tmp_path: Path, annotated: list[str]) -> tuple[Path, str, Any]:
    from medh5.io.dicom_seg import from_dicom_seg

    case = _dicom(tmp_path)
    seg = _liver_seg(case, tmp_path / "liver.dcm")
    report = from_dicom_seg(seg, case.path, annotated_classes=annotated)
    return case.path, "seg", report


def _mint_rtstruct(tmp_path: Path, annotated: list[str]) -> tuple[Path, str, Any]:
    from medh5.io.rtstruct import from_rtstruct

    case = _dicom(tmp_path)
    rt = _gtv_rtstruct(case, tmp_path / "rt.dcm")
    report = from_rtstruct(rt, case.path, annotated_classes=annotated, rasterize=True)
    return case.path, "contours", report


_MINTING = {
    "nifti": (_mint_nifti, "liver"),
    "import-seg-nifti": (_mint_import_seg_nifti, "liver"),
    "seg": (_mint_seg, "liver"),
    "rtstruct": (_mint_rtstruct, "gtv"),
}


@pytest.mark.parametrize("importer", sorted(_MINTING))
def test_S11_3_annotated_classes_mint_an_examined_absent_class(tmp_path, importer):
    """A class named in ``annotated_classes`` that the input does not supply
    was examined and found absent: an importer that mints its label set
    mints it too, after the classes the input supplies, and says so.  It
    was a bare ``KeyError`` ("label set 'converted' has no class 'tumor'"),
    so "searched for the tumour and found none" could not be imported."""
    build, supplied = _MINTING[importer]
    path, ann_id, report = build(tmp_path, [supplied, "tumor"])
    with medh5.open(path) as sample:
        label_set = sample.label_set
        assert label_set[supplied].id == 1
        assert label_set["tumor"].id == 2, "minted after the supplied ids"
        ann = sample.annotations[ann_id]
        assert set(ann.annotated_class_ids) == {1, 2}
        if ann_id == "seg":
            assert not ann.dense(["tumor"])[0].any()
        else:
            assert {int(p.class_id) for p in ann.polygons()} == {1}
            mask = sample.annotations[f"{ann_id}_mask"]
            assert set(mask.annotated_class_ids) == {1, 2}
            assert not mask.dense(["tumor"])[0].any()
    notes = [n for n in report.of_kind("label_set") if n.detail.get("examined_absent")]
    assert len(notes) == 1 and notes[0].severity == "decision"
    assert notes[0].detail["examined_absent"] == {"tumor": 2}
    assert "tumor" in notes[0].message


def _absent_nifti(tmp_path: Path) -> tuple[Path, Any]:
    from medh5.io.nifti import from_nifti

    files = _nifti_case(tmp_path, "liver")
    given = LabelSet(
        "given", version="1.0.0", classes=[LabelClass(4, "liver", "Liver")]
    )
    out = tmp_path / "case.medh5"
    return out, lambda: from_nifti(
        {"CT": files["ct"]}, out, masks={"liver": files["liver"]},
        label_set=given, annotated_classes=["liver", "tumor"],
    )  # fmt: skip


def _absent_import_seg_nifti(tmp_path: Path) -> tuple[Path, Any]:
    from medh5.io.nifti import from_nifti, import_seg_nifti

    files = _nifti_case(tmp_path, "liver")
    out = tmp_path / "case.medh5"
    from_nifti({"CT": files["ct"]}, out)
    _with_label_set(out, (1, "liver"))
    return out, lambda: import_seg_nifti(
        out, {"liver": files["liver"]}, annotated_classes=["liver", "tumor"]
    )


def _absent_seg(tmp_path: Path) -> tuple[Path, Any]:
    from medh5.io.dicom_seg import from_dicom_seg

    case = _dicom(tmp_path)
    seg = _liver_seg(case, tmp_path / "liver.dcm")
    _with_label_set(case.path, (1, "liver"))
    return case.path, lambda: from_dicom_seg(
        seg, case.path, annotated_classes=["liver", "tumor"]
    )


def _absent_rtstruct(tmp_path: Path) -> tuple[Path, Any]:
    from medh5.io.rtstruct import from_rtstruct

    case = _dicom(tmp_path)
    rt = _gtv_rtstruct(case, tmp_path / "rt.dcm")
    _with_label_set(case.path, (1, "gtv"))
    return case.path, lambda: from_rtstruct(
        rt, case.path, annotated_classes=["gtv", "tumor"]
    )


_ABSENT = {
    "nifti": _absent_nifti,
    "import-seg-nifti": _absent_import_seg_nifti,
    "seg": _absent_seg,
    "rtstruct": _absent_rtstruct,
}


@pytest.mark.parametrize("importer", sorted(_ABSENT))
def test_S11_3_annotated_class_missing_from_label_set_is_E402(tmp_path, importer):
    """Where the label set is given --- the sample's, or ``label_set=`` ---
    an importer mints nothing: a class it is told was examined must be in
    it.  Refused as E402, naming the class and the way out, with nothing
    written."""
    from medh5.errors import MEDH5ValidationError

    path, attempt = _ABSENT[importer](tmp_path)
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(MEDH5ValidationError, match="'tumor'") as caught:
        attempt()
    assert caught.value.code == "E402"
    assert "annotated_classes" in str(caught.value)
    assert (path.read_bytes() if path.exists() else None) == before


def test_label_set_fixes_class_ids_across_imports(capsys, tmp_path):
    """Minted, a class's id is its rank among the masks one case supplies:
    a cohort whose cases segment different organs numbers them differently.
    ``--label-set`` (``label_set=``) gives every case the same ids."""
    pytest.importorskip("nibabel")
    labels = tmp_path / "labels.json"
    vocabulary = LabelSet(
        "abdomen",
        version="1.0.0",
        classes=[LabelClass(1, "liver", "Liver"), LabelClass(2, "spleen", "Spleen")],
    )
    labels.write_text(json.dumps(vocabulary.to_json()), encoding="utf-8")
    ids: dict[str, dict[str, tuple[int, ...]]] = {"minted": {}, "given": {}}
    for case, masks in (("a", ("liver", "spleen")), ("b", ("spleen",))):
        files = _nifti_case(tmp_path / case, *masks)
        argv = ["--image", f"CT={files['ct']}"]
        argv += [a for m in masks for a in ("--mask", f"{m}={files[m]}")]
        for how, extra in (("minted", []), ("given", ["--label-set", str(labels)])):
            out = tmp_path / case / f"{how}.medh5"
            _ok(capsys, "convert", "from-nifti", str(out), *argv, *extra)
            with medh5.open(out) as sample:
                ids[how][case] = sample.annotations["seg"].class_ids
    assert ids["minted"] == {"a": (1, 2), "b": (1,)}, "spleen is 2, then 1"
    assert ids["given"] == {"a": (1, 2), "b": (2,)}, "spleen is 2 in both"


def test_a_mask_the_given_label_set_has_no_class_for_is_E402(tmp_path):
    from medh5.errors import MEDH5ValidationError
    from medh5.io.nifti import from_nifti

    files = _nifti_case(tmp_path, "liver")
    given = LabelSet("given", version="1.0.0", classes=[LabelClass(1, "spleen", "S")])
    out = tmp_path / "case.medh5"
    with pytest.raises(MEDH5ValidationError, match="'liver'") as caught:
        from_nifti(
            {"CT": files["ct"]}, out, masks={"liver": files["liver"]}, label_set=given
        )
    assert caught.value.code == "E402"
    assert not out.exists()


def test_to_rtstruct_refuses_a_label_longer_than_16_characters(capsys, tmp_path):
    """``StructureSetLabel`` is a DICOM SH, at most 16 characters: a longer
    label was written, and the file was invalid DICOM."""
    case = _dicom(tmp_path)
    rt = _gtv_rtstruct(case, tmp_path / "rt.dcm")
    _ok(capsys, "convert", "from-rtstruct", str(rt), str(case.path))
    out = tmp_path / "out.dcm"
    argv = ["convert", "to-rtstruct", str(case.path), "contours", str(out)]
    argv += _sources(case)
    code, printed = run(capsys, *argv, "--label", "a label of 17 ch.")
    assert code == EXIT_ERROR
    assert "16" in printed.err and "--label" in printed.err
    assert not out.exists()
    _ok(capsys, *argv, "--label", "a label of 16 ch")


def test_to_dicom_seg_refuses_a_series_description_longer_than_64(capsys, tmp_path):
    """``SeriesDescription`` is a DICOM LO, at most 64 characters."""
    pytest.importorskip("highdicom")
    case = _dicom(tmp_path)
    _organs(case)
    out = tmp_path / "seg.dcm"
    argv = ["convert", "to-dicom-seg", str(case.path), "organs", str(out)]
    argv += _sources(case)
    code, printed = run(capsys, *argv, "--series-description", "x" * 65)
    assert code == EXIT_ERROR
    assert "64" in printed.err and "--series-description" in printed.err
    assert not out.exists()
    _ok(capsys, *argv, "--series-description", "x" * 64)


def test_import_seg_nifti_adds_masks_to_a_sample(capsys, tmp_path):
    """`import_seg_nifti` had no command."""
    files = _nifti_case(tmp_path, "liver", "spleen")
    case = tmp_path / "case.medh5"
    _ok(capsys, "convert", "from-nifti", str(case), "--image", f"CT={files['ct']}")
    report = tmp_path / "r.json"
    printed = _ok(
        capsys, "convert", "import-seg-nifti", str(case), "--mask",
        f"liver={files['liver']}", "--mask", f"spleen={files['spleen']}",
        "--id", "organs", "--grid", "ref", "--report", str(report),
    )  # fmt: skip
    assert "DECISION" in printed.out
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["converter"] == "import-seg-nifti"
    with medh5.open(case) as sample:
        organs = sample.annotations["organs"]
        assert {organs.class_key(c) for c in organs.class_ids} == {"liver", "spleen"}
    code, printed = run(capsys, "convert", "import-seg-nifti", str(case))
    assert code == EXIT_USAGE and "--mask" in printed.err
