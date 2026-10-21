"""The changelog's fragments (`changes/README.md`).

2.0.2's section was drafted at the end of the release, reconciling about twenty
claims against merged code.  Each pull request now writes its own entry, a file
in `changes/`, and the release assembles them; these tests hold the format,
the assembly and the guards that keep an entry from shipping unassembled.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.helpers import ROOT


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves the module by name
    spec.loader.exec_module(module)
    return module


changelog = _load(ROOT / ".github/scripts/changelog.py", "medh5_changelog_test")

CHANGELOG = "# Changelog\n\n## [Unreleased]\n\n## [1.0.0] — 2026-01-01\n\n- old\n"


def _fragment(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    changes = tmp_path / "changes"
    changes.mkdir()
    (changes / "README.md").write_text("# not a fragment\n", encoding="utf-8")
    log = tmp_path / "CHANGELOG.md"
    log.write_text(CHANGELOG, encoding="utf-8")
    return changes, log


def test_every_fragment_in_the_repository_passes_check() -> None:
    changelog.check(ROOT / "changes", ROOT / "CHANGELOG.md")


def test_build_assembles_sections_in_order_and_entries_by_area(tmp_path) -> None:
    changes, log = _tree(tmp_path)
    _fragment(
        changes,
        "b-cli.md",
        "<!-- area: cli -->\n### Fixed\n\n- **A cli fix.** It works\n  now.\n",
    )
    _fragment(
        changes,
        "a-cli.md",
        "<!-- area: cli -->\n### Fixed\n\n- **Another cli fix.**\n",
    )
    _fragment(
        changes,
        "files.md",
        "<!-- area: files -->\n### Added\n\n- **A file thing.**\n\n"
        "### Fixed\n\n- **A file fix.**\n- **A second file fix.**\n",
    )
    _fragment(changes, "ci.md", "<!-- no entry: CI only -->\n")
    _fragment(changes, "lead.md", "**A release.** It has things.\n")
    changelog.build("1.1.0", "2026-02-03", changes, log)
    assert log.read_text(encoding="utf-8") == (
        "# Changelog\n\n## [Unreleased]\n\n"
        "## [1.1.0] — 2026-02-03\n\n"
        "**A release.** It has things.\n\n"
        "### Added\n\n- **A file thing.**\n\n"
        "### Fixed\n\n"
        "- **A file fix.**\n- **A second file fix.**\n"
        "- **Another cli fix.**\n"
        "- **A cli fix.** It works\n  now.\n\n"
        "## [1.0.0] — 2026-01-01\n\n- old\n"
    )
    # The fragments and the lead are consumed; the README stays.
    assert sorted(p.name for p in changes.iterdir()) == ["README.md"]


def test_build_takes_a_pre_release_as_tagged(tmp_path) -> None:
    changes, log = _tree(tmp_path)
    _fragment(changes, "lead.md", "**A candidate.**\n")
    changelog.build("1.1.0-rc.1", "2026-02-03", changes, log)
    assert "## [1.1.0-rc.1] — 2026-02-03\n\n**A candidate.**\n" in log.read_text(
        encoding="utf-8"
    )
    with pytest.raises(changelog.FragmentError, match="X.Y.Z"):
        changelog.build("1.1.0rc1", "2026-02-03", changes, log)


def test_build_needs_a_lead(tmp_path) -> None:
    changes, log = _tree(tmp_path)
    _fragment(changes, "x.md", "<!-- area: cli -->\n### Fixed\n\n- **Fix.**\n")
    with pytest.raises(changelog.FragmentError, match="lead.md"):
        changelog.build("1.1.0", "2026-02-03", changes, log)
    assert log.read_text(encoding="utf-8") == CHANGELOG


def test_build_touches_nothing_without_the_unreleased_line(tmp_path) -> None:
    changes, log = _tree(tmp_path)
    spelt = CHANGELOG.replace("[Unreleased]\n\n", "[Unreleased] \n\n")
    log.write_text(spelt, encoding="utf-8")
    _fragment(changes, "x.md", "<!-- area: cli -->\n### Fixed\n\n- **Fix.**\n")
    _fragment(changes, "lead.md", "**A release.**\n")
    with pytest.raises(changelog.FragmentError, match="Unreleased"):
        changelog.build("1.1.0", "2026-02-03", changes, log)
    assert log.read_text(encoding="utf-8") == spelt
    assert sorted(p.name for p in changes.iterdir()) == ["README.md", "lead.md", "x.md"]


def test_the_release_pull_request_passes_the_fragment_step() -> None:
    """It adds no fragment: `build` deletes them all, and that counts."""
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    step = ci.split("- name: Every pull request records its change", 1)[1]
    step = step.split("\n      - ", 1)[0]
    assert "--diff-filter=AMD HEAD^1 HEAD -- 'changes/*.md'" in step


def test_the_site_shows_the_preview_under_unreleased() -> None:
    """`splice`, which `hooks/mkdocs_hooks.py` applies to the site's changelog.

    The hook itself imports MkDocs, which only the docs job installs; there
    `mkdocs build --strict` runs it, and resolves every fragment's links.
    """
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    preview = "### Fixed\n\n- **A fix.**"
    assert changelog.splice(text, preview) == text.replace(
        "## [Unreleased]\n\n", f"## [Unreleased]\n\n{preview}\n\n", 1
    )
    assert changelog.splice(text, "") == text
    hook = (ROOT / "hooks/mkdocs_hooks.py").read_text(encoding="utf-8")
    assert "script.splice(changelog, preview)" in hook


def test_a_release_refuses_fragments_left_in_changes(tmp_path) -> None:
    notes = _load(ROOT / ".github/scripts/release_notes.py", "medh5_notes_test")
    changes, _ = _tree(tmp_path)
    notes.check_fragments(changes)  # the README alone is fine
    _fragment(changes, "x.md", "<!-- no entry: CI only -->\n")
    with pytest.raises(SystemExit, match="x.md"):
        notes.check_fragments(changes)


def test_check_refuses_entries_under_unreleased(tmp_path) -> None:
    changes, log = _tree(tmp_path)
    log.write_text(
        CHANGELOG.replace("[Unreleased]\n\n", "[Unreleased]\n\n- x\n\n"),
        encoding="utf-8",
    )
    with pytest.raises(changelog.FragmentError, match="Unreleased"):
        changelog.check(changes, log)


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("<!-- area: nowhere -->\n### Fixed\n\n- **x.**\n", "unknown area"),
        ("<!-- area: cli -->\n### Changed\n\n- **x.**\n", "unknown section"),
        ("<!-- area: cli -->\n### Fixed\n\n- **x.** " + "y" * 80 + "\n", "columns"),
        ("<!-- area: cli -->\n### Fixed\n\n- **x.** a --- b\n", "dash"),
        ("<!-- area: cli -->\n### Fixed\n\n- x is fixed.\n", "- \\*\\*"),
        ("### Fixed\n\n- **x.**\n", "area"),
        ("<!-- area: cli -->\n### Fixed\n", "no entry"),
        ("<!-- area: cli -->\n### Fixed\n\n- **x.**\n\n- **y.**\n", "blank line"),
        ("<!-- area: cli -->\n### Fixed\n- **x.**\n", "blank line"),
        ("<!-- area: cli -->\n### Fixed\n\n- **x.**\nmore\n", "indented"),
        ("<!-- area: cli -->\n### Fixed\n\n- **x.**", "newline"),
        ("<!-- no entry: CI only -->\n### Fixed\n", "that line only"),
        ("<!-- no entry:  -->\n", "area"),
        (
            "<!-- area: cli -->\n### Fixed\n\n- **x.**\n\n### Fixed\n\n- **y.**\n",
            "twice",
        ),
    ],
)
def test_check_refuses_a_malformed_fragment(tmp_path, text: str, why: str) -> None:
    changes, log = _tree(tmp_path)
    _fragment(changes, "bad.md", text)
    with pytest.raises(changelog.FragmentError, match=why):
        changelog.check(changes, log)


def test_a_dash_inside_code_is_code(tmp_path) -> None:
    changes, log = _tree(tmp_path)
    _fragment(
        changes, "ok.md", "<!-- area: cli -->\n### Fixed\n\n- **`a---b` reads.**\n"
    )
    assert changelog.check(changes, log) == 1


def test_the_command_line_reports_and_fails(tmp_path, monkeypatch, capsys) -> None:
    changes, log = _tree(tmp_path)
    monkeypatch.setattr(changelog, "CHANGES", changes)
    monkeypatch.setattr(changelog, "CHANGELOG", log)
    _fragment(changes, "x.md", "<!-- area: cli -->\n### Fixed\n\n- **Fix.**\n")
    assert changelog.main(["check"]) == 0
    assert changelog.main(["preview"]) == 0
    assert "- **Fix.**" in capsys.readouterr().out
    _fragment(changes, "y.md", "<!-- area: cli -->\n### Nope\n\n- **Fix.**\n")
    assert changelog.main(["check"]) == 1
    assert "unknown section" in capsys.readouterr().err


def test_check_lists_a_promise_about_a_later_release(tmp_path, monkeypatch, capsys):
    """2.2.0's NIfTI entry said the converters "move onto it next" after the
    pull request that moved them had merged, and 2.1.0's release notes said
    2.2.0 "will require" Python 3.11 before anyone had decided it.  `check`
    lists every such line for the maintainer to confirm, and passes."""
    changes, log = _tree(tmp_path)
    monkeypatch.setattr(changelog, "CHANGES", changes)
    monkeypatch.setattr(changelog, "CHANGELOG", log)
    # The entry as 2.2.0's fragment `w22-02-native-nifti-import.md` held it.
    _fragment(
        changes,
        "w22-02-native-nifti-import.md",
        "<!-- area: converters -->\n### Added\n\n"
        "- **The engine reads and writes NIfTI.** nibabel is\n"
        "  unlinked. The Python converter and the command line move onto it"
        " next.\n",
    )
    _fragment(
        changes,
        "plain.md",
        "<!-- area: cli -->\n### Fixed\n\n- **`--will` is a flag.** It works.\n",
    )
    _fragment(changes, "lead.md", "**2.2.1 will require nothing new.**\n")
    _fragment(changes, "ci.md", "<!-- no entry: a later release will tell -->\n")
    assert changelog.promises(changes) == [
        "changes/w22-02-native-nifti-import.md:5: unlinked. The Python converter"
        " and the command line move onto it next.",
        "changes/lead.md:1: **2.2.1 will require nothing new.**",
    ]
    assert changelog.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "a promise for the maintainer to confirm" in out
    assert "w22-02-native-nifti-import.md:5:" in out and "lead.md:1:" in out
    for line in ("Next, the sampler.", "A later release adds it.", "In future"):
        assert changelog._PROMISE.search(line), line
    for line in ("Nextflow reads it.", "A willing reader.", "The latest release."):
        assert not changelog._PROMISE.search(line), line
