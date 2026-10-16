"""The documentation is checked against the code, not proofread.

Every defect this file guards against was found in review after being written
confidently and shipped: a `--source ct/*.dcm` that argparse rejects before doing
anything, a `--out cold/` that dies with `IsADirectoryError`, a response label
naming one visit when it describes an interval.  Each was a claim about the code
that the code disagreed with, and each was reachable by asking the code.

So the documentation gets the same treatment as anything else here.  A prose page
is not a comment; it is an assertion about behaviour, and an assertion nobody
executes decays.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest

from medh5.cli import command_tree
from tests.helpers import ROOT

DOCS = ROOT / "docs"


README = ROOT / "README.md"


# The normative specification is excluded: its examples are illustrative of the
# format rather than of this package's API, and it is versioned separately.
PAGES = sorted(p for p in DOCS.rglob("*.md") if "spec/medh5-1.0.md" not in str(p))


# A fenced block: its opening fence, its language and its body, up to the
# closing fence --- each found from its own opening, so the prose between two
# blocks is never read as one.
_FENCE = re.compile(r"^[ \t]*```(\w*)[^\n]*\n(.*?)^[ \t]*```", re.S | re.M)


# A shell operator: a bar or an ampersand standing alone, or any redirection.
_SHELL = re.compile(r"(^|\s)(\|\|?|&&?)(\s|$)|[<>]")


def _token(word: str) -> str:
    """A synopsis word without its brackets: `[--level` is the flag `--level`."""
    return word.lstrip("[({").rstrip("])},")


def _shell_lines() -> list[tuple[Path, str]]:
    """Every `medh5 ...` invocation in a fenced block, line continuations joined."""
    out: list[tuple[Path, str]] = []
    for path in [*PAGES, README]:
        for language, block in _FENCE.findall(path.read_text(encoding="utf-8")):
            if language not in ("", "bash", "sh", "console"):
                continue
            for line in re.sub(r"\\\n\s*", " ", block).splitlines():
                line = line.strip().removeprefix("$ ").strip()
                if not line.startswith("medh5 "):
                    continue
                # A trailing `...` elides arguments, and a pipeline or a
                # redirection is more than one invocation.  A bar between two
                # words is a synopsis's alternatives (`[--metric dice|iou]`),
                # whose flags are checked like any others.
                if line.endswith("...") or _SHELL.search(line):
                    continue
                out.append((path, line))
    return out


def _parser_tree() -> dict[tuple[str, ...], dict[str, Any]]:
    """Every command of the CLI's grammar, by its path of subcommand names."""
    tree: dict[tuple[str, ...], dict[str, Any]] = {}

    def walk(node: dict[str, Any], prefix: tuple[str, ...]) -> None:
        tree[prefix] = node
        for name, sub in node["commands"].items():
            walk(sub, (*prefix, name))

    walk(command_tree(), ())
    return tree


# Claims corrected during review, each with the shape that was wrong.  A pattern
# here is cheaper than the round trip that found it the first time.
STALE_CLAIMS: tuple[tuple[str, str, str, str | None], ...] = (
    (
        "physical=True said to return float32",
        r"physical=True\)?`? (returns|gives|reads as)( always)? `?float32",
        "a physical read is float32 only where a rescale applies; pass dtype "
        "for one type across a cohort (D27).",
        None,
    ),
    (
        "a missing file said to escape FileNotFoundError",
        r"`except FileNotFoundError` does not catch it",
        "a missing file is MEDH5FileNotFoundError, a FileNotFoundError with "
        "its errno and filename (2.1).",
        None,
    ),
    (
        "a higher minor read as optional",
        r"\bMAY\b\W*\s*read a higher minor",
        "1.0 §16 and 1.1 §2.2: a reader MUST NOT refuse a file solely for a "
        "higher minor, and reads its supported projection.",
        None,
    ),
    (
        "ignore region read as a class",
        r"dense\(\[?65535",
        "the ignore id is not a class; `layers` gives an all-zero plane. "
        "Use ignore_mask().",
        None,
    ),
    (
        "worker_init_fn called mandatory",
        r"worker_init_fn[^.]{0,80}?(required|mandatory|not optional|must not omit)",
        "the handle cache is PID-keyed, so the callback is an eager reset, "
        "not a requirement.",
        None,
    ),
    (
        "validation described as metadata-only",
        r"touches no voxel|never touches a voxel|costs nothing to run",
        "`structural` and `semantic` do bounded payload scans; "
        "`semantic` measured 287 ms on 12.6 Mvox.",
        None,
    ),
    (
        "glob passed to --source",
        r"--source [^\s]*\*\.dcm",
        "`--source` takes one value per occurrence; a glob is rejected "
        "as extra arguments.",
        None,
    ),
    (
        "recompress --out given a directory",
        r"--out [^\s]*/(\s|$)",
        "`--out` is a single destination filename, not a directory.",
        None,
    ),
    (
        "PairReport credited with resolving transforms",
        r"aligned by a transform",
        "`PairReport` holds files, pairs and cross-sectional skips; "
        "it resolves nothing.",
        None,
    ),
    (
        "change label naming one endpoint",
        r'"response"[^)]*timepoints=\["tp[01]"\]',
        "a label describing an interval names both visits, in acquisition order.",
        None,
    ),
    (
        "probmap export described as keeping every nonzero value",
        r"every nonzero\s+probability becomes 1",
        "`dense()` applies the stored threshold (default 0.5), so sub-threshold "
        "voxels become background.",
        None,
    ),
    (
        "det profile presented as requiring boxes",
        r"(needs boxes can require `det`|require `det`[^.]{0,60}(instead of|rather "
        r"than) a `KeyError`|`det`[^.\n]{0,40}(guarantees|means) boxes)",
        "the check is task='detection'; keypoints and contours satisfy it too, so "
        "requiring `det` does not guarantee a box annotation.",
        None,
    ),
    (
        "ignore_mask() promised on every encoding",
        r"ignore_mask\(\)[^.\n]{0,60}(every encoding|all encodings|any encoding)",
        "only labelmap and layers implement it; bitmask and probmap reference a "
        "separate mask annotation.",
        None,
    ),
    (
        "cohort checked against a pre-scrub manifest",
        r"medh5 dataset check[^\n]*# C501",
        "`deidentified` is captured at index time; re-index after `scrub --apply` "
        "or C501 cannot fire.",
        # Legitimate when the block re-indexes first, which is the fix.
        r"dataset index",
    ),
    (
        "entry-level coverage used as a loss mask",
        r"e\.annotated_class_ids for e in manifest",
        "`Entry.annotated_class_ids` unions every annotation; a loss mask needs "
        "the one being trained on.",
        None,
    ),
    (
        "registration preflight resolved per timepoint",
        r"transform_between\(pair\.first, pair\.second\)",
        "a visit may hold several grids on different frames; resolve between the "
        "grids the images are on.",
        None,
    ),
    (
        "inverse_id offered for graph resolution",
        r"store it under `inverse_id`",
        "`inverse_id` adds a second one-hop path and makes the reverse "
        "direction raise E501.",
        None,
    ),
    (
        "validator payload scans described as capped",
        r"capped at 64\s*M\b|same 64\s*M cap",
        "the ignore-region and layer-overlap scans read the whole payload in "
        "slabs of at most 8 MiB: the memory is bounded, the work is not.",
        None,
    ),
    (
        "a source build promised where PyO3 cannot make one",
        r"free-threaded\s+C?Python(?!\s+3\.1[4-9])|\bPyPy\b(?!\s+3\.1[1-9])",
        "PyO3 0.29 builds for free-threaded CPython from 3.14 and for PyPy from "
        "3.11; on 3.13t or PyPy 3.10 the sdist cannot build.",
        None,
    ),
    (
        "validate --json offered as a submission",
        r"emits a superset",
        "the scorer takes what `validate --json` reports --- `path` and "
        "`diagnostics` in place of `file`, `errors` and `warnings` --- which "
        "is no superset of a submission.",
        None,
    ),
    (
        "strict offered as the check for a new file",
        # Any sample or glob: the 0.x guide's `new/*.medh5` escaped the first
        # form of this, which named `case.medh5` and `case_0001.medh5` only.
        r"medh5 validate \S+\.medh5 --level strict",
        "strict promotes W903 (no de-identification record) and W912 (no "
        "ontology code) to errors, so it fails every freshly written or "
        "imported sample; check one at --level integrity.",
        # Legitimate as the gate for a dataset being published.
        r"publi(sh|cation)",
    ),
    (
        "tracking coverage described without the ignore region",
        r"instance ids nobody looked for\}|class ids examined there\}",
        "an annotation whose ignore region holds any voxel adds nothing to "
        "`coverage`, so an object it does not show is `unexamined`, never "
        "`resolved` (§7.4, §7.7).",
        None,
    ),
    (
        "probmap named as the only lossy transcode",
        r"lossless(ly)?,?\s+(except|but)\s+(for\s+)?(a\s+)?`?probmap",
        "`instances` to a dense encoding drops every instance_id, and is refused "
        "without drop_identity=True (--drop-identity).",
        None,
    ),
    (
        "manifest entries keyed by path alone",
        r"\be\.path:\s*set\(",
        "the members of a collection share its path, so all but one are lost; key "
        "a sample as split.paths() names it, `path::key` for a member.",
        None,
    ),
    (
        "partial-labels' organs said not to have examined its lesion",
        r'is_annotated\("lesion"\)\s*#\s*False',
        "the page writes `organs` with annotated_classes=[liver, spleen, lesion].",
        None,
    ),
    (
        "a batch's used_index read as one item's",
        r'patch = batch\["meta"\]\["patch"\]|batch\["meta"\]\["patch"\]\["used_index"\]'
        r"\s*#\s*True",
        "collate turns the patch metadata into lists, one entry per item; one "
        'item\'s is item["meta"]["patch"]["used_index"].',
        None,
    ),
    (
        "the clinical guide amending the cohort the training guide pins",
        r'amend\("cohort/P-03\.medh5"\)',
        "progression.task.json pins P-03's content_id, so an amendment in place "
        "fails the next guide's preflight (T302); amend a copy.",
        None,
    ),
    (
        "a SEG export called binary only",
        r"export is binary|fractional data is thresholded on the way out",
        'to_dicom_seg(segmentation_type="fractional") writes a probmap as a '
        "FRACTIONAL SEG; only the default, binary export thresholds.",
        None,
    ),
    (
        "Blosc2 decoding said to release HDF5's lock",
        r"Blosc2[^.]{0,120}?(outside (HDF5's |the )?(global )?lock|"
        r"releases? (HDF5's |the )(global )?lock)",
        "only a window read decompresses Blosc2 blocks outside the lock; a whole "
        "read, and every other filter, decompresses under it.",
        r"window",
    ),
    (
        "a writer used by two threads said to raise RuntimeError",
        r"Already borrowed",
        "a call on a writer another thread is using raises MEDH5Error, one "
        "thread at a time (2.2, E4).",
        None,
    ),
    (
        "free-threaded CPython said to turn the GIL back on",
        r"GIL back on|asks for the GIL|turns the GIL on",
        "the cp314t wheel runs without the GIL, and importing medh5 leaves it "
        "off (2.2, E4).",
        None,
    ),
)


# Phrases corrected during review that no surrounding sentence makes true.  They
# sat in a decision table and a list of non-goals, whose other rows say "not"
# for reasons of their own --- which exempts the paragraph from the check above,
# so these are matched wherever they appear.
STALE_PHRASES: tuple[tuple[str, str, str], ...] = (
    (
        "a C-compatible library promised",
        r"C-compatible",
        'the engine has no C interface: no `extern "C"` API and no header; a C '
        "or C++ program runs the `medh5` binary.",
    ),
    (
        "lossless round trips to DICOM promised",
        r"round trips? to DICOM",
        "no exporter writes images as DICOM; annotations go back as a SEG or an "
        "RTSTRUCT against the original series.",
    ),
    (
        "the annotation kinds miscounted",
        r"Thirteen kinds: five voxel encodings",
        "six voxel kinds (five encodings and `mask`), boxes, oriented boxes, "
        "keypoints, points, contours, meshes and classification.",
    ),
    (
        "annotations placed at §5",
        r"annotations`\]? \| §5",
        "§5 is the label space (`labels`); the annotations are §6–§9.",
    ),
    (
        "commit described as os.replace",
        r"`os\.replace`s it into position",
        "the engine fsyncs the temporary file, renames it over the target and "
        "fsyncs the directory.",
    ),
    (
        "every view said to raise after close",
        r"after `s\.close\(\)` using one\s+raises",
        "views and file reads raise; what the sample already read (grids, "
        "annotation headers, a transform's frames) keeps answering.",
    ),
    (
        "DICOM slice spacing held per gap",
        r"gaps disagree by more than a tolerance",
        "each slice is held to where the regular stack puts it, within a tenth "
        "of the spacing and at least 0.01 mm; per-gap bounds let drift "
        "accumulate.",
    ),
    (
        "the L3 read per core",
        r"(?i)detected per-core",
        "the chunk budget is one CPU's share of the L3: its size over the CPUs "
        "sharing it.",
    ),
    (
        "chunks said to be held between 512 KiB and 4 MiB",
        r"(?i)chunks are held between 512 KiB and 4 MiB",
        "the budget is held there; no computed chunk passes 4 MiB, and nothing "
        "raises a smaller one to 512 KiB.",
    ),
)


@pytest.mark.parametrize(("path", "line"), _shell_lines(), ids=lambda v: str(v)[:60])
def test_documented_cli_flags_exist(path: Path, line: str) -> None:
    """A command in the documentation is a command the CLI defines, and so is
    every flag --- a synopsis's bracketed ones included.

    `--source ct/*.dcm` looked right for years and never worked: the option takes
    one value per occurrence, so the glob expanded and argparse exited with
    `unrecognized arguments` before the export began.
    """
    tree = _parser_tree()
    tokens = [_token(word) for word in shlex.split(line)[1:]]
    command: tuple[str, ...] = ()
    for token in tokens:
        if token.startswith("-") or (*command, token) not in tree:
            break
        command = (*command, token)
    parser = tree[command]
    rest = tokens[len(command) :]
    if parser["commands"] and rest and not rest[0].startswith("-"):
        # `medh5 COMMAND [args]` is a placeholder; any other word after a
        # command group must name one of its commands.
        if re.fullmatch(r"[A-Z][A-Z_]*", rest[0]):
            return
        named = " ".join(("medh5", *command))
        pytest.fail(
            f"{path.name}: `{line}`: {rest[0]!r} is not a command of `{named}` "
            f"({', '.join(sorted(parser['commands']))})"
        )
    defined = set(parser["options"])
    used = [
        token
        for token in rest
        # A negative number is a value (`--date-shift-days -117`), not a flag.
        if token.startswith("-") and not re.fullmatch(r"-\d+(\.\d+)?", token)
    ]
    unknown = [f for f in used if f.split("=")[0] not in defined]
    assert not unknown, f"{path.name}: `{line}` uses undefined flag(s) {unknown}"


@pytest.mark.parametrize(
    "line",
    [
        "medh5 valdate case.medh5",
        "medh5 dataset chek cohort/",
        "medh5 conformance rnu /tmp/corpus",
        "medh5 validate PATH [--levle L]",
        "medh5 agree PATH A B [--metirc dice|iou]",
    ],
)
def test_a_misspelled_command_or_flag_is_caught(line: str) -> None:
    """The check above fails what it exists to fail: a command group followed
    by a word that names none of its commands, and a synopsis's bracketed flag
    the CLI does not define."""
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        test_documented_cli_flags_exist(Path("page.md"), line)


def test_a_synopsis_with_alternatives_is_checked() -> None:
    """`[--profile basic|strict]` is a synopsis, not a pipeline, so its line
    is checked; `|| { ... }` is shell, and is not."""
    lines = [line for _, line in _shell_lines()]
    assert any("[--profile basic|strict]" in line for line in lines)
    assert not any("||" in line for line in lines)


def test_the_registration_preflight_resolves_on_frames() -> None:
    """The preflight must ask about frames, and nothing else.

    It resolved by *grid* id, which reads correctly and is still wrong: grid ids
    and timepoint ids are separate namespaces (§2.3), so a grid may be named
    `tp0`, and `Sample._frames_for` matches a timepoint before a grid --- which
    hands the question back to the whole visit and re-answers "registered" for a
    PET pair carrying only a CT registration.  A frame uid is matched last.

    This one is out of reach of `STALE_CLAIMS` below: the recipe contains
    `is not None`, and the paragraph-scoped disclaimer check reads that `not` as
    a correction and exempts the block.
    """
    text = (DOCS / "guides" / "longitudinal.md").read_text(encoding="utf-8")
    block = next(
        b
        for b in re.findall(r"```python\n(.*?)```", text, re.S)
        if "transform_between" in b and "TimepointPairSampler" in b
    )
    returned = re.findall(r"^\s*return\s+(.+)$", block, re.M)
    assert returned, "the preflight names what it resolves on through a helper"
    assert all("frame_uid" in value for value in returned), (
        f"the preflight helper must yield a frame uid, not a grid id: {returned}"
    )


def test_transform_between_over_visits_is_described_as_it_behaves(
    tmp_path: Path,
) -> None:
    """Between two visits of several frames, `transform_between` refuses.

    The guides once said it searched every pair and returned the first path;
    2.0.2 tried the pairs in grid-id order, so renaming a grid changed the
    answer (`apet*` gave `None`, `pet*` a transform). 2.1.0 refuses either
    naming, and the pages say so and point to `by="grid"`. Out of reach of
    `STALE_CLAIMS`: the wrong paragraphs carried a word the disclaimer check
    reads as a correction.
    """
    import numpy as np

    import medh5

    def between_visits(shared: str) -> object:
        path = tmp_path / f"{shared}.medh5"
        with medh5.create(path, sample_id=shared) as w:
            w.add_timepoint("tp0", days_from_baseline=0)
            w.add_timepoint("tp1", days_from_baseline=90)
            for grid, timepoint, frame in (
                (f"{shared}0", "tp0", "pseudo:pet"),
                ("ct0", "tp0", "pseudo:ct0"),
                (f"{shared}1", "tp1", "pseudo:pet"),
                ("ct1", "tp1", "pseudo:ct1"),
            ):
                w.add_grid(
                    grid,
                    shape=(4, 4, 4),
                    spacing=(1.0, 1.0, 1.0),
                    timepoint=timepoint,
                    frame_uid=frame,
                )
            w.add_image("CT", np.zeros((4, 4, 4), np.int16), grid="ct0", modality="CT")
            w.add_transform(
                "ct0_to_ct1",
                kind="affine",
                from_frame="pseudo:ct0",
                to_frame="pseudo:ct1",
                matrix=np.eye(4),
            )
        with medh5.open(path) as s:
            return s.transform_between("tp0", "tp1")

    from medh5.errors import MEDH5ValidationError

    for naming in ("apet", "pet"):
        with pytest.raises(MEDH5ValidationError, match="one frame"):
            between_visits(naming)
    pages = ("guides/longitudinal.md", "guides/registration.md")
    for page in (*pages, "reference/python-api.md"):
        text = (DOCS / page).read_text(encoding="utf-8")
        assert "grid-id order" not in text, page
        assert "returns the first path it finds" not in text, page
        assert "share a frame with the other visit's" not in text, page
        assert "MEDH5ValidationError" in text, page
        assert 'by="grid"' in text, page


def _documented_signatures() -> list[tuple[str, str]]:
    """`(name, parameters)` of each signature row of the Python reference's
    tables: a call standing alone in a row's first cell, elided (`...`) or not."""
    text = (DOCS / "reference" / "python-api.md").read_text(encoding="utf-8")
    return re.findall(r"^\| `((?:medh5\.\w+\.)?[\w.]+)\(([^`]*)\)` \|", text, re.M)


def _plain(func: Any) -> str:
    """A signature as the reference writes one: names, `*`, and defaults."""
    import inspect

    parts: list[str] = []
    star = False
    for p in inspect.signature(func).parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
            star = True
        elif p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
        else:
            if p.kind is p.KEYWORD_ONLY and not star:
                parts.append("*")
                star = True
            parts.append(p.name if p.default is p.empty else f"{p.name}={p.default!r}")
    return ", ".join(parts)


def test_documented_signatures_are_the_real_ones() -> None:
    """A signature in the Python reference is the function's own.

    `Link.between` was documented with its options positional and without
    `target_annotation_id`, and `augment` with a positional `out`: a call
    written from the page raised `TypeError`. A row that elides (`...`) is
    held to the parameters it names, in order.
    """
    import importlib

    import medh5.cache
    import medh5.clinical
    import medh5.task

    modules = (medh5.clinical, medh5.task, medh5.cache)
    rows = _documented_signatures()
    assert any(name == "Link.between" for name, _ in rows)
    for name, documented in rows:
        if name.startswith("medh5."):
            module, _, attr = name.rpartition(".")
            found: Any = getattr(importlib.import_module(module), attr)
        else:
            head, *rest = name.split(".")
            owner = next((m for m in modules if hasattr(m, head)), None)
            if owner is None:
                continue
            found = getattr(owner, head)
            for part in rest:
                found = getattr(found, part)
        real = _plain(found).removeprefix("self, ").removeprefix("self")
        if documented.endswith("..."):
            named = [w.strip() for w in documented.split(",")[:-1]]
            assert real.split(", ")[: len(named)] == named, (name, documented, real)
        elif documented:
            assert documented == real, f"{name}({documented}) is {name}({real})"


@pytest.mark.parametrize(
    ("label", "pattern", "why", "unless"), STALE_CLAIMS, ids=lambda v: str(v)[:40]
)
def test_corrected_claims_stay_corrected(
    label: str, pattern: str, why: str, unless: str | None
) -> None:
    """Each of these was true of the documentation once, and is not true of the code.

    They recur because a correction lands on the page that was reported and not
    on its three siblings.  Instances kept coming back until the class was
    checked, which is what this does.
    """
    # Scoped to the paragraph, not the line: a correction names the wrong form
    # in order to warn against it, and the warning is usually a sentence away
    # from the mention.  Line-scoped matching flags the fix as the defect.
    disclaimed = re.compile(
        r"\bnot\b|instead|rather than|does not|never|finds nothing|all-zero|"
        r"expands to|is rejected|no longer|used to",
        re.I,
    )
    hits: list[str] = []
    for path in [*PAGES, README]:
        text = path.read_text(encoding="utf-8")
        offset = 0
        for para in text.split("\n\n"):
            if unless and re.search(unless, para, re.I):
                offset += len(para) + 2
                continue
            if re.search(pattern, para, re.I) and not disclaimed.search(para):
                line_no = text.count("\n", 0, offset) + 1
                first = next(
                    (ln for ln in para.splitlines() if re.search(pattern, ln, re.I)),
                    para,
                )
                hits.append(f"{path.name}:{line_no}: {first.strip()[:80]}")
            offset += len(para) + 2
    assert not hits, f"{label}: {why}\n  " + "\n  ".join(hits)


@pytest.mark.parametrize(
    ("label", "pattern", "why"), STALE_PHRASES, ids=lambda v: v[:40]
)
def test_corrected_phrases_stay_corrected(label: str, pattern: str, why: str) -> None:
    """Each of these was once on a page, and no context makes it true."""
    hits = [
        f"{path.name}:{text.count(chr(10), 0, match.start()) + 1}"
        for path in [*PAGES, README, *sorted(ROOT.glob("crates/*/README.md"))]
        for text in [path.read_text(encoding="utf-8")]
        for match in re.finditer(pattern, text)
    ]
    assert not hits, f"{label}: {why}\n  " + "\n  ".join(hits)


def test_the_front_pages_link_every_normative_document() -> None:
    """The README and the site's front page linked 1.0 alone, so a reader
    writing an implementation never found 1.1 (the clinical profile, and how
    a later minor version is opened) or the task-and-cache contract."""
    readme = README.read_text(encoding="utf-8")
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    for spec in ("medh5-1.0", "medh5-1.1", "task-cache-1"):
        assert (DOCS / "spec" / f"{spec}.md").is_file()
        assert f"/spec/{spec}/)" in readme, f"README.md does not link {spec}"
        assert f"](spec/{spec}.md)" in index, f"docs/index.md does not link {spec}"


# What is not prose on a Markdown page: fenced and inline code, link targets,
# reference definitions, comments, table separator rows and horizontal rules.
_NOT_PROSE = re.compile(
    r"^[ \t]*```.*?^[ \t]*```"
    r"|(`+)(?:(?!\n\n).)+?\1"
    r"|\]\([^)]*\)|<https?://[^>]+>|^[ \t]*\[[^\]\n]+\]:[^\n]*|<!--.*?-->"
    r"|^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*(?::?-+:?)?[ \t]*$"
    r"|^[ \t]*---[ \t]*$",
    re.S | re.M,
)


def test_a_dash_is_written_as_a_dash() -> None:
    """`---` is three hyphens to GitHub, PyPI, crates.io and the site alike ---
    none of their Markdown makes it a dash --- and every page printed it so,
    over six hundred times, beside the `—` the newer prose wrote."""
    pages = [
        README,
        ROOT / "CHANGELOG.md",
        ROOT / "CONTRIBUTING.md",
        *sorted(ROOT.glob("changes/*.md")),
        *sorted(ROOT.glob("crates/*/README.md")),
        *sorted(DOCS.rglob("*.md")),
    ]
    hits: list[str] = []
    for path in pages:
        text = path.read_text(encoding="utf-8")
        prose = _NOT_PROSE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
        for match in re.finditer(r"(?<!-)---(?!-)", prose):
            line = text.count("\n", 0, match.start()) + 1
            where = path.relative_to(ROOT).as_posix()
            hits.append(f"{where}:{line}: {text.splitlines()[line - 1].strip()[:80]}")
    assert not hits, "write a dash as `—`, not `---`:\n  " + "\n  ".join(hits)


def test_the_rust_example_on_the_docs_page_is_the_tested_one() -> None:
    """`cargo test` compiles and runs the engine crate's README; the docs page
    shows the same block, so it is tested too --- as long as they are one."""
    import re

    block = re.compile(r"```rust\n.*?```\n", re.S)
    readme = (ROOT / "crates/medh5/README.md").read_text(encoding="utf-8")
    page = (ROOT / "docs/reference/rust.md").read_text(encoding="utf-8")
    assert block.findall(page) == block.findall(readme)[:1]
    assert '#![doc = include_str!("../README.md")]' in (
        ROOT / "crates/medh5/src/lib.rs"
    ).read_text(encoding="utf-8")


def _module_column(text: str) -> list[list[str]]:
    """The first column of the "Where things are" table, one list per row."""
    import re

    section = text.split("## Where things are", 1)[1]
    rows = re.findall(r"^\| (.+?) \|", section, re.M)
    return [re.findall(r"`([a-z0-9_]+)`", row) for row in rows if "`" in row]


def test_the_module_tables_name_the_same_modules() -> None:
    """The crate's README and the docs page each carry the module table."""
    readme = (ROOT / "crates/medh5/README.md").read_text(encoding="utf-8")
    page = (ROOT / "docs/reference/rust.md").read_text(encoding="utf-8")
    assert _module_column(readme) == _module_column(page)


def test_every_public_module_is_documented_or_hidden() -> None:
    """A `pub mod` of the engine is in rust.md's table or `#[doc(hidden)]`."""
    import re

    lib = (ROOT / "crates/medh5/src/lib.rs").read_text(encoding="utf-8")
    page = (ROOT / "docs/reference/rust.md").read_text(encoding="utf-8")
    tabled = {name for row in _module_column(page) for name in row}
    public = re.findall(r"^(#\[doc\(hidden\)\]\n)?pub mod (\w+);", lib, re.M)
    assert len(public) > 20
    missing = [name for hidden, name in public if not hidden and name not in tabled]
    assert not missing, f"public modules missing from rust.md's table: {missing}"


def test_the_torch_docstring_agrees_with_the_docs() -> None:
    import medh5.torch as torch_pkg

    assert "not optional" not in (torch_pkg.__doc__ or "")
    assert "recommended" in (torch_pkg.__doc__ or "")


def test_cargo_install_is_told_the_release_profile() -> None:
    """crates.io publishes no profile, so `cargo install medh5-cli` built
    without the workspace's thin LTO and kept the symbol table.  The install
    page and the crate's README give the profile three ways: the environment,
    PowerShell's `$env:` and `--config`; each says what `Cargo.toml` says."""
    workspace = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    section = workspace.split("\n[profile.release]\n", 1)[1].split("\n[", 1)[0]
    profile = {
        key: value.strip('"')
        for key, value in re.findall(r"^([\w-]+) = (\S+)$", section, re.M)
    }
    assert profile == {"lto": "thin", "codegen-units": "4", "strip": "symbols"}
    for path in (DOCS / "install.md", ROOT / "crates/medh5-cli/README.md"):
        text = path.read_text(encoding="utf-8")
        found = {
            "shell": re.findall(r"\bCARGO_PROFILE_RELEASE_(\w+)=(\w+)", text),
            "powershell": re.findall(
                r'^\$env:CARGO_PROFILE_RELEASE_(\w+) = "(\w+)"$', text, re.M
            ),
            "config": re.findall(r"profile\.release\.([\w-]+)=\"?(\w+)", text),
        }
        for form, pairs in found.items():
            named = {key.lower().replace("_", "-"): value for key, value in pairs}
            assert named == profile, f"{path.name}: {form}"


def test_the_cohort_check_page_shows_each_codes_severity(monkeypatch) -> None:
    """The page's Severity column is the engine's `CHECK_SEVERITY`, row by row.

    The hook imports MkDocs, which only the docs job installs, for two types
    the tables never touch: a stand-in module lets it load here.
    """
    import importlib.util
    import sys
    import types

    from medh5.dataset.check import CHECK_CODE_SEVERITY, CHECK_CODES

    files = types.ModuleType("mkdocs.structure.files")
    files.File = files.Files = object  # type: ignore[attr-defined]
    for name in ("mkdocs", "mkdocs.structure"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "mkdocs.structure.files", files)
    spec = importlib.util.spec_from_file_location(
        "medh5_docs_hooks_test", ROOT / "hooks/mkdocs_hooks.py"
    )
    assert spec is not None and spec.loader is not None
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)

    table = hook._cohort_codes_markdown(ROOT)
    rows = re.findall(r"^\| `(C\d{3})`\{ #c\d{3} \} \| (\w+) \| ", table, re.M)
    assert dict(rows) == CHECK_CODE_SEVERITY
    assert [code for code, _ in rows] == sorted(CHECK_CODES)
    assert "`C401`" in table.split("\n\n", 1)[1]


def _published_schemas() -> tuple[tuple[str, str, str], ...]:
    """`SCHEMAS` from the docs hook, read without importing MkDocs."""
    import ast

    tree = ast.parse((ROOT / "hooks/mkdocs_hooks.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == (
            "SCHEMAS"
        ):
            assert node.value is not None
            found: tuple[tuple[str, str, str], ...] = ast.literal_eval(node.value)
            return found
    raise AssertionError("hooks/mkdocs_hooks.py defines no SCHEMAS")


def test_every_engine_schema_is_published() -> None:
    """A schema the engine embeds has a page and a URL on the site.

    At 2.0.2 only the sample-document schema did; the clinical, task and cache
    schemas shipped in the engine and the corpus and nowhere a reader could
    browse them.
    """
    data = ROOT / "crates/medh5/data"
    engine = {p.name.removesuffix(".schema.json") for p in data.glob("*.schema.json")}
    published = _published_schemas()
    assert {name for name, _, _ in published} == engine
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for name, page, _ in published:
        text = (DOCS / page).read_text(encoding="utf-8")
        assert text.count(f"<!--@schema {name}-->") == 1, (name, page)
        assert f": {page}\n" in nav, f"{page} is not in the nav"


def test_a_published_schema_is_the_embedded_bytes() -> None:
    """The file the site serves is the one the engine validates with."""
    import importlib

    for name, _, function in _published_schemas():
        module, attr = function.rsplit(".", 1)
        embedded = getattr(importlib.import_module(module), attr)()
        source = ROOT / "crates/medh5/data" / f"{name}.schema.json"
        # The bytes, not the text: `include_str!` keeps a Windows checkout's
        # CRLF, which reading in text mode would translate away.
        assert embedded == source.read_bytes().decode("utf-8"), name
