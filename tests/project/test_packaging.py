"""The package's own metadata.

`medh5.__version__` is not decoration: it is stamped into every file's
`generator` (§12) and into every dataset manifest.  A wheel whose declared
version disagrees with the string it writes into user data is a provenance
bug that no amount of format testing catches, because both halves are
internally consistent -- they just describe different releases.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import io
import json
import os
import random
import re
import shutil
import subprocess
import sysconfig
import tarfile
from importlib.metadata import distribution
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import medh5
from tests.helpers import ROOT, write_sample

PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".github/workflows/ci.yml"
BUILD = ROOT / ".github/workflows/build.yml"
FUZZ = ROOT / ".github/workflows/fuzz.yml"
RELEASE = ROOT / ".github/workflows/release.yml"
ADVISORIES = ROOT / ".github/workflows/advisories.yml"
LOCK_REFRESH = ROOT / ".github/workflows/lock-refresh.yml"


WORKSPACE = ROOT / "Cargo.toml"


def _declared() -> str:
    """The version the wheel is built with, read without `tomllib`.

    `tomllib` is 3.11+ and this package supports 3.10.  Guarding the import
    would make this skip on the oldest interpreter it claims to support --
    which is the failure mode the MONAI job exists to prevent -- so it reads
    the lines it needs instead.  Since 2.0 the one source is the Cargo
    workspace: `pyproject.toml` declares the version dynamic, maturin reads it
    from the extension crate, and every crate --- engine, CLI, extension ---
    inherits the workspace's.  What this checks is that the wiring points
    there and nowhere else.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    assert re.search(r'^dynamic\s*=\s*\["version"\]', text, re.M), (
        "pyproject.toml must declare version as dynamic"
    )
    assert not re.search(r"^version\s*=", text, re.M), (
        "a static [project] version would be a second source"
    )
    manifest = re.search(r'^manifest-path\s*=\s*"([^"]+)"', text, re.M)
    assert manifest, "maturin must build the extension crate"
    for crate in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        assert re.search(
            r"^version\.workspace\s*=\s*true", crate.read_text("utf-8"), re.M
        ), f"{crate.parent.name} must inherit the workspace version"
    assert (ROOT / manifest.group(1)).is_file()
    workspace = WORKSPACE.read_text(encoding="utf-8")
    section = workspace.split("[workspace.package]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', section, re.M)
    assert match, "no [workspace.package] version"
    return match.group(1)


def _release_script(name: str) -> ModuleType:
    path = ROOT / ".github" / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jobs(workflow: str) -> dict[str, str]:
    """A workflow's jobs by name, each with the text of its block."""
    body = workflow.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([\w-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _extras() -> dict[str, str]:
    """`pyproject.toml`'s extras by name, each with the text of its list."""
    text = PYPROJECT.read_text(encoding="utf-8")
    section = text.split("[project.optional-dependencies]\n", 1)[1]
    section = section.split("\n[", 1)[0]
    return dict(re.findall(r"^([\w-]+) = \[\n(.*?)^\]", section, re.M | re.S))


def _package_source() -> str:
    """The installed package's Python source (the sdist job tests the wheel
    with the checkout's `medh5/` removed)."""
    package = Path(medh5.__file__).parent
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(package.rglob("*.py"))
    )


def test_the_wheel_version_and_the_stamped_version_agree():
    """The release workflow checks the tag against the workspace version.

    The version is written once, in the Cargo workspace; the wheel, the CLI and
    the engine that stamps `generator` all take it from there, so a wheel that
    reports its version wrongly in the files it writes is no longer a thing a
    bump can produce.
    """
    assert medh5.__version__ == _declared()


def test_the_crates_pin_each_other_at_the_workspace_version():
    """The crates name each other at exactly the workspace version.

    crates.io publishes each crate on its own, so `medh5` depends on the
    `medh5-sys` of its own release through an `=X.Y.Z` pin in
    `[workspace.dependencies]`, and `medh5-cli` on the `medh5` of its own.
    The runbook said the version lived in one place; a bump that set it alone
    would publish an engine depending on the previous release's HDF5 build.
    """
    workspace = WORKSPACE.read_text(encoding="utf-8")
    section = workspace.split("[workspace.dependencies]", 1)[1].split("\n[", 1)[0]
    pins = dict(
        re.findall(
            r'^(medh5[\w-]*)\s*=\s*\{[^}]*version\s*=\s*"=([^"]+)"', section, re.M
        )
    )
    assert set(pins) == {"medh5", "medh5-sys", "medh5-cli"}, pins
    assert set(pins.values()) == {_declared()}, pins


def test_the_format_version_is_not_the_package_version():
    """§1: the *format* is 1.1 (1.0 plus the optional clinical profile). The
    package ships releases against it.

    Tying them together would force a format-version bump for every package
    release, and the format version is what tells a reader whether it can open
    the file at all.

    The assertion used to be ``_declared().startswith("1.0")``, which tied the
    two together in exactly the way the paragraph above forbids -- it only
    looked correct while the package happened to sit on 1.0.x, and the first
    package minor bump against an unchanged format failed it.  1.x then held
    the package MAJOR equal to the format's; 2.0 is the architectural reset
    that broke that on purpose (a Rust engine, one format), so what has to
    hold is that the newest format version is 1.2, the package version is a
    well-formed release of its own, and the package MAJOR never trails the
    format MAJOR it writes.  A file is still written at the lowest version its
    content needs, so an imaging-only sample stays 1.0.
    """
    assert medh5.__format_version__ == "1.2"
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-]?[0-9A-Za-z.]+)?", _declared())
    package_major = int(_declared().split(".")[0])
    assert package_major >= int(medh5.__format_version__.split(".")[0])


def test_release_the_homebrew_formula_names_the_binaries_ci_builds() -> None:
    """The release writes the formula from the checksums of the binaries CI
    built.  A target the formula expects and CI never builds --- or an archive
    named one way by the build and another by the formula --- would publish a
    formula whose download 404s."""
    brew = _release_script("homebrew_formula.py")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    build = (ROOT / ".github/scripts/build-dist.sh").read_text(encoding="utf-8")
    # The targets ci.yml passes to build.yml, one job each.
    built = set(re.findall(r"^      target: (\S+)$", ci, re.M))
    assert len(built) == 6
    assert set(brew.TARGETS.values()) <= built
    assert all(target in release for target in built)
    assert 'name="medh5-$version-$target"' in build
    assert brew.archive("2.0.0", "x") == "medh5-2.0.0-x.tar.gz"

    sums = {brew.archive("2.0.0", t): f"{i:064x}" for i, t in enumerate(sorted(built))}
    text = brew.formula("2.0.0", sums, "medh5-project/medh5")
    base = "https://github.com/medh5-project/medh5/releases/download/v2.0.0"
    for target in brew.TARGETS.values():
        name = brew.archive("2.0.0", target)
        assert f'url "{base}/{name}"' in text
        assert f'sha256 "{sums[name]}"' in text
    assert 'bin.install "medh5"' in text

    # macOS on Intel was dropped in 2.2.0: an Intel Mac is refused by the
    # formula, not sent to an archive no release has.  Its one download sits
    # outside any `on_arm`, since Homebrew reads the URL before it checks the
    # dependency, and a formula with no URL there could not be read at all.
    assert "x86_64-apple-darwin" not in text
    macos = text.split("  on_macos do\n", 1)[1].split("\n  end\n", 1)[0]
    arm = brew.archive("2.0.0", "aarch64-apple-darwin")
    assert macos == (
        f'    depends_on arch: :arm64\n    url "{base}/{arm}"\n    sha256 "{sums[arm]}"'
    )
    linux = text.split("  on_linux do\n", 1)[1].split("\n  end\n", 1)[0]
    assert "depends_on" not in linux and "on_intel" in linux

    del sums[brew.archive("2.0.0", "x86_64-unknown-linux-gnu")]
    with pytest.raises(SystemExit, match="x86_64-unknown-linux-gnu"):
        brew.formula("2.0.0", sums, "medh5-project/medh5")
    listing = f"{'a' * 64}  medh5-2.0.0-x.tar.gz\n\n{'b' * 64} *y.zip\n"
    assert brew.read_sums(listing) == {
        "medh5-2.0.0-x.tar.gz": "a" * 64,
        "y.zip": "b" * 64,
    }


def test_the_2_0_0_notes_describe_the_format_it_writes() -> None:
    """The release publishes its CHANGELOG section, and only that section.

    2.0.0's said "The file format is unchanged: 1.0" while the engine writes
    1.1 for a clinical sample, and the clinical work sat under [Unreleased],
    which no release extracts.  The release now refuses a CHANGELOG with
    entries still under [Unreleased].
    """
    notes = _release_script("release_notes.py")
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    body = notes.notes(text, "2.0.0", "medh5-project/medh5", "v2.0.0")
    assert "**format 1.1**" in body and "**The `clinical` profile**" in body
    assert "format is unchanged" not in body
    pinned = "https://github.com/medh5-project/medh5/blob/v2.0.0/docs/spec/medh5-1.1.md"
    assert f"]({pinned})" in body
    with pytest.raises(SystemExit, match="no section for 0.0.0"):
        notes.notes(text, "0.0.0", "o/r", "v0.0.0")
    notes.check_unreleased("## [Unreleased]\n\n## [9.9.9]\n\n- shipped\n")
    with pytest.raises(SystemExit, match="Unreleased"):
        notes.check_unreleased("## [Unreleased]\n\n- not yet\n\n## [9.9.9]\n\n- x\n")


def test_the_release_notes_fit_a_github_release() -> None:
    """GitHub refuses a release body over its limit, and the release is
    created after the packages are published, so the notes are refused when
    they are written, before anything is published."""
    notes = _release_script("release_notes.py")
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert len(notes.notes(text, "2.0.0", "medh5-project/medh5", "v2.0.0")) <= (
        notes.GITHUB_LIMIT
    )
    # Notes of exactly the limit pass and one character more is refused.  In
    # long lines: the parser's time grows with the blocks, and the thirty
    # thousand one-word items this used to build took most of the file's run.
    line = "- " + "x" * 998
    body = "\n".join([line] * (notes.GITHUB_LIMIT // (len(line) + 1)))
    body += "\n- " + "x" * (notes.GITHUB_LIMIT - len(body) - 3)
    assert len(body) == notes.GITHUB_LIMIT
    assert notes.notes(f"## [9.9.9]\n\n{body}\n", "9.9.9", "o/r", "v9.9.9") == body
    with pytest.raises(SystemExit, match="GitHub takes at most"):
        notes.notes(f"## [9.9.9]\n\n{body}x\n", "9.9.9", "o/r", "v9.9.9")


def test_a_release_page_breaks_no_line_of_the_changelog() -> None:
    """A release page, like a comment, breaks a line at every newline, and the
    changelog wraps at 80 columns: every release page up to 2.0.1 broke each
    paragraph at every line.  The notes render as the changelog does --- the
    same blocks, the same words --- with no line broken."""
    from markdown_it import MarkdownIt

    notes = _release_script("release_notes.py")
    page = MarkdownIt("commonmark", {"breaks": True}).enable("table")
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    flowed = notes.unwrap(text)
    assert notes.shown(flowed) == notes.shown(text)
    assert "<br" not in page.render(flowed)
    versions = re.findall(r"^## \[(\d[^\]]*)\]", text, re.M)
    assert {"0.5.0", "2.0.0"} <= set(versions)
    for version in versions:
        body = notes.notes(text, version, "medh5-project/medh5", f"v{version}")
        assert "<br" not in page.render(body), version


def test_the_release_notes_join_a_paragraphs_lines_and_nothing_else() -> None:
    notes = _release_script("release_notes.py")
    wrapped = [
        "### Fixed",
        "- **A list item** wrapped",
        "  over lines, against a target of",
        "  400.",
        "  - a nested item,",
        "    wrapped",
        "",
        "A line that ends in a hard break\\",
        "keeps it, and",
        "the rest joins.",
        "",
        "```python",
        "x = 1",
        "y = 2",
        "```",
        "",
        "| a | b |",
        "| - | - |",
        "",
        "    indented",
        "    code",
        "",
        "> a quote",
        "> as written",
        "",
        "[ref]: https://example.org",
    ]
    flowed = wrapped.copy()
    flowed[1:6] = [
        "- **A list item** wrapped over lines, against a target of 400.",
        "  - a nested item, wrapped",
    ]
    flowed[5:7] = ["keeps it, and the rest joins."]
    assert notes.unwrap("\n".join(wrapped)) == "\n".join(flowed)


def test_release_notes_that_would_read_otherwise_unwrapped_stay_as_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes = _release_script("release_notes.py")
    assert notes.unwrap("one\ntwo") == "one two"
    monkeypatch.setattr(notes, "shown", lambda source: source)
    assert notes.unwrap("one\ntwo") == "one\ntwo"


def test_a_published_releases_notes_are_written_again_by_its_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`release_notes.py vX.Y.Z` rewrites a published release's notes, for
    `gh release edit`; the [Unreleased] check is the release workflow's."""
    notes = _release_script("release_notes.py")
    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n- next\n\n## [1.0.0] — 2026-01-01\n\nOne\nline.\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    notes.main(["v1.0.0"])
    assert (tmp_path / "release-notes.md").read_text(encoding="utf-8") == "One line.\n"
    monkeypatch.setenv("GITHUB_REF_NAME", "v1.0.0")
    with pytest.raises(SystemExit, match="Unreleased"):
        notes.main([])


class TestPublicNames:
    """Where the public names live."""

    def test_the_writer_lives_beside_the_reader(self):
        import medh5.sample as sample_module

        assert medh5.SampleWriter is sample_module.SampleWriter
        assert medh5.create is sample_module.create
        assert medh5.amend is sample_module.amend
        assert {"SampleWriter", "create", "amend"} <= set(sample_module.__all__)

    def test_optional_dependencies_name_their_extra(self):
        from medh5._optional import require

        with pytest.raises(ImportError, match=r"medh5\[dicomseg\]"):
            require("no_such_module_medh5", extra="dicomseg", purpose="testing")
        assert require("json", extra="x", purpose="y").dumps({}) == "{}"

    def test_the_dead_names_are_gone(self):
        import medh5.geometry as grid
        import medh5.transforms as apply

        assert not hasattr(grid, "iter_spatial_slices")
        assert not hasattr(grid, "KNOWN_COORD_SYSTEMS")
        assert not hasattr(apply, "_refuse_outside")


class TestRepositoryGates:
    """The gates the repository holds itself to."""

    def test_the_lint_gate_refuses_suppressions_that_suppress_nothing(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert '"RUF100"' in text

    def test_actions_are_pinned_by_commit(self):
        uses = []
        workflows = sorted((ROOT / ".github/workflows").glob("*.yml"))
        assert {"ci.yml", "release.yml", "advisories.yml"} <= {
            p.name for p in workflows
        }
        for path in workflows:
            text = path.read_text(encoding="utf-8")
            uses += re.findall(r"uses:\s*(\S+)(.*)", text)
        external = [(u, rest) for u, rest in uses if not u.startswith("./")]
        assert external
        for action, rest in external:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), action
            assert re.search(r"#\s*v\d", rest), f"{action} carries no version comment"
        dependabot = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
        assert "package-ecosystem: github-actions" in dependabot

    def test_the_extension_asks_a_free_threaded_python_for_the_gil(self):
        """PyO3 declares a module GIL-free unless told otherwise, so a build
        for free-threaded CPython ran `_core`, never audited for it, with no
        GIL at all.  An abi3 build cannot show the slot, so the source must."""
        lib = (ROOT / "crates/medh5-python/src/lib.rs").read_text(encoding="utf-8")
        assert "#[pymodule(gil_used = true)]" in lib

    def test_the_minimum_rust_is_a_full_version(self):
        """`rust-version` is published with every crate and cannot be
        corrected there.  It said `1.85` --- 1.85.0 to cargo --- while the HDF5
        crates need 1.85.1, and the job that keeps the promise installed the
        newest 1.85.x for `1.85` and passed."""
        workspace = WORKSPACE.read_text(encoding="utf-8")
        section = workspace.split("[workspace.package]", 1)[1].split("\n[", 1)[0]
        assert re.search(r'^rust-version = "\d+\.\d+\.\d+"$', section, re.M)
        msrv = _jobs(CI.read_text(encoding="utf-8"))["msrv"]
        assert "^[0-9]+\\.[0-9]+\\.[0-9]+$" in msrv

    def test_the_rust_the_project_builds_with_is_pinned(self):
        """CI installed `stable` as it stood that day, and the release re-runs
        CI on its tag: a lint a new stable had added since the pull request
        failed `clippy -D warnings` there, and the artifacts were built by
        whichever compiler that was.  rust-toolchain.toml names one release,
        no older than `rust-version`, with the components the gates run.  The
        unpacked sdist carries no pin: its users build with their own Rust."""
        pin = ROOT / "rust-toolchain.toml"
        if (ROOT / "PKG-INFO").is_file():
            assert not pin.exists()
            return
        text = pin.read_text(encoding="utf-8")
        channel = re.search(r'^channel = "(\d+)\.(\d+)\.(\d+)"$', text, re.M)
        assert channel, "pin a release, X.Y.Z, not a channel"
        workspace = WORKSPACE.read_text(encoding="utf-8")
        msrv = re.search(r'^rust-version = "(\d+)\.(\d+)\.(\d+)"$', workspace, re.M)
        assert msrv
        assert tuple(map(int, channel.groups())) >= tuple(map(int, msrv.groups()))
        components = re.search(r"^components = \[(.*)\]$", text, re.M)
        assert components
        assert {"rustfmt", "clippy"} <= set(re.findall(r'"([\w-]+)"', components[1]))

    def test_every_rust_build_installs_the_pinned_rust_first(self):
        """Each job that compiles Rust installs rust-toolchain.toml's before
        its first cargo or maturin, and nothing installs `stable`: a job that
        did not would build with whatever its runner had.  The manylinux
        containers have no Rust, and installed `stable` with `curl | sh`;
        build-dist.sh now checks a pinned rustup-init against its digest, and
        pins maturin (inside pyproject.toml's range) and CMake."""
        install = "rustup toolchain install --no-self-update"
        build = (ROOT / ".github/scripts/build-dist.sh").read_text(encoding="utf-8")
        assert install in build and "sh.rustup.rs" not in build
        pins = dict(re.findall(r"^(\w+_version)=(\S+)$", build, re.M))
        assert set(pins) == {"maturin_version", "cmake_version", "rustup_version"}
        assert all(re.fullmatch(r"\d+\.\d+\.\d+", v) for v in pins.values()), pins
        assert '"maturin==$maturin_version"' in build
        assert '"cmake==$cmake_version"' in build
        text = PYPROJECT.read_text(encoding="utf-8")
        floor = re.search(r'"maturin>=([\d.]+),<(\d+)"', text)
        assert floor
        version = tuple(map(int, pins["maturin_version"].split(".")))
        assert tuple(map(int, floor[1].split("."))) <= version < (int(floor[2]),)
        for arch in ("x86_64", "aarch64", "x86_64_musl", "aarch64_musl"):
            assert re.search(rf"^rustup_sha256_{arch}=[0-9a-f]{{64}}$", build, re.M)
        assert '[[ "$actual" != "$digest" ]]' in build
        containers = re.findall(r"container: (\S+)", CI.read_text(encoding="utf-8"))
        assert len(containers) == 4
        for image in containers:
            assert re.fullmatch(r"quay\.io/pypa/[\w.-]+@sha256:[0-9a-f]{64}", image)
        built = []
        for workflow in (CI, BUILD, RELEASE):
            for name, job in _jobs(workflow.read_text(encoding="utf-8")).items():
                code = "\n".join(
                    line
                    for line in job.splitlines()
                    if not line.lstrip().startswith("#")
                )
                rust = re.search(r"\bcargo (?!\+)|\bmaturin |build-dist\.sh", code)
                if not rust:
                    continue
                built.append(name)
                assert not re.search(r"(install|default)[^\n]*\bstable\b", code), name
                if "build-dist.sh" not in code:
                    assert install in code[: rust.start()], name
        assert sorted(built) == [
            "build", "crates", "msvc-from-crates-io", "rust", "rust-platforms",
            "sdist", "semver",
        ]  # fmt: skip
        # `build` is build.yml's, which the four platform jobs call.
        callers = [
            name
            for name, job in _jobs(CI.read_text(encoding="utf-8")).items()
            if "uses: ./.github/workflows/build.yml" in job
        ]
        assert sorted(callers) == [
            "build-linux-aarch64",
            "build-linux-musl-aarch64",
            "build-linux-musl-x86_64",
            "build-linux-x86_64",
            "build-macos-arm64",
            "build-windows-x86_64",
        ]

    def test_advisories_are_checked_weekly_and_before_a_release(self):
        """Nothing checked the crates compiled into every wheel and binary, or
        what the extras install, against the published advisories.  Both are
        checked every week, on a pull request that changes what is locked or
        declared, and on the release tag, whose build waits for them."""
        text = (ROOT / ".github/workflows/advisories.yml").read_text(encoding="utf-8")
        for trigger in ("schedule", "workflow_call", "pull_request"):
            assert f"\n  {trigger}:" in text, trigger
        assert "      - Cargo.lock\n" in text and "      - pyproject.toml\n" in text
        assert re.search(r"cargo install cargo-audit --version [\d.]+ --locked", text)
        assert "cargo audit --deny warnings" in text
        assert re.search(r'pip install "pip-audit==[\d.]+"', text)
        assert "pip-audit --requirement requirements.txt --strict" in text
        jobs = _jobs(RELEASE.read_text(encoding="utf-8"))
        assert "uses: ./.github/workflows/advisories.yml" in jobs["advisories"]
        assert re.search(r"^    needs: \[ci, advisories\]$", jobs["build"], re.M)

    def test_the_native_inventory_is_what_the_lock_builds(self):
        """RustSec sees crates, not the C libraries they compile: zlib 1.3.2
        reaches every wheel as `libz-sys` 1.1.29, a version no zlib advisory
        names.  SECURITY.md lists each library with the crate or the vendored
        source that builds it, so a bump that changes a library changes its
        row --- and with it the reading of each advisory against it."""
        security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
        table = security.split("\n## Native inventory\n", 1)[1].split("\n## ", 1)[0]
        rows = {
            library: (version, source)
            for library, version, source in re.findall(
                r"^\| ([\w -]+?) \| ([\d.]+) \| ([^|]+?) \|", table, re.M
            )
        }
        lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
        locked = dict(re.findall(r'^name = "([^"]+)"\nversion = "([^"]+)"', lock, re.M))
        crates = {
            "HDF5": "hdf5-metno-src",
            "zlib": "libz-sys",
            "LZ4": "lz4-sys",
            "Zstandard": "zstd-sys",
        }
        for library, crate in crates.items():
            assert rows[library][1] == f"`{crate}` {locked[crate]}", library
        # The -sys crates that carry their library's version in their own.
        assert locked["lz4-sys"].endswith(f"+lz4-{rows['LZ4'][0]}")
        assert locked["zstd-sys"].endswith(f"+zstd.{rows['Zstandard'][0]}")
        vendor = ROOT / "crates/medh5-sys/vendor"
        blosc2 = (vendor / "c-blosc2/include/blosc2.h").read_text(encoding="utf-8")
        version = re.escape(rows["C-Blosc2"][0])
        assert re.search(rf'#define BLOSC2_VERSION_STRING\s+"{version}"', blosc2)
        patches = (vendor / "hdf5-blosc2/PATCHES.md").read_text(encoding="utf-8")
        assert f"(HDF5-Blosc2 {rows['HDF5-Blosc2 filter'][0]})" in patches
        assert set(rows) == {*crates, "C-Blosc2", "HDF5-Blosc2 filter"}

    def test_a_release_that_bundles_an_affected_zlib_says_why_it_is_not(self):
        """CVE-2026-85091 names zlib 1.3.1.2 through 1.3.2, and a scanner that
        keys on the version flags every wheel, though the code is not linked
        (SECURITY.md).  The VEX statement must name the version being
        released while the bundled zlib is in that range --- and only then."""
        security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
        zlib = re.search(r"^\| zlib \| ([\d.]+) \|", security, re.M)
        assert zlib
        row = f"| [CVE-2026-85091](#cve-2026-85091) | zlib {zlib[1]} | not affected |"
        vex = json.loads((ROOT / "vex/medh5.openvex.json").read_text(encoding="utf-8"))
        (statement,) = [
            s
            for s in vex["statements"]
            if s["vulnerability"]["name"] == "CVE-2026-85091"
        ]
        assert statement["status"] == "not_affected"
        assert statement["justification"] == "vulnerable_code_not_present"
        products = {p["@id"] for p in statement["products"]}
        version = _declared()
        released = {
            f"pkg:pypi/medh5@{version}",
            f"pkg:cargo/medh5-cli@{version}",
            f"pkg:github/XwK-P/medh5@v{version}",
        }
        affected = (1, 3, 1, 2) <= tuple(map(int, zlib[1].split("."))) <= (1, 3, 2)
        if affected:
            assert row in security
            assert released <= products, sorted(released - products)
            lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
            libz = re.search(r'^name = "libz-sys"\nversion = "([^"]+)"', lock, re.M)
            assert libz
            for product in statement["products"]:
                assert {c["@id"] for c in product["subcomponents"]} >= {
                    f"pkg:generic/zlib@{zlib[1]}"
                }
            assert f"pkg:cargo/libz-sys@{libz[1]}" in {
                c["@id"] for p in statement["products"] for c in p["subcomponents"]
            }
        else:
            assert not released & products

    def test_ci_runs_what_ships_where_it_ships(self):
        """The engine's Rust tests ran on Linux alone, `medh5.torch` never on
        Windows, the standalone binary's hand-over to Python nowhere, two of
        the five wheels only through the corpus, and no crate was built from
        its package before the release published it.  The NIfTI parity test
        runs off Linux too: 2.1.0's records hold on every platform (E3)."""
        jobs = _jobs(CI.read_text(encoding="utf-8"))
        assert (
            "cargo package --locked -p medh5-sys -p medh5 -p medh5-cli" in jobs["rust"]
        )
        platforms = jobs["rust-platforms"]
        assert "cargo test -p medh5 --lib --test nifti_parity --locked" in platforms
        assert "[windows-latest, macos-15]" in platforms
        extras = re.search(r"\[(dev,[\w,]+)\]", jobs["test-windows"])
        assert extras and "torch" in extras.group(1).split(",")
        build = _jobs(BUILD.read_text(encoding="utf-8"))["build"]
        for bridged in ("MEDH5_PYTHON=python ", "MEDH5_PYTHON=/nonexistent "):
            assert bridged in build
        assert "name: ci-dist-aarch64-unknown-linux-gnu\n" in jobs["test-linux-aarch64"]
        assert "-rs" in jobs["test-monai"]
        assert "tests/integrations/test_meds.py -v -rs" in jobs["test-meds"]
        # SimpleITK, which the ITK cross-check needs, is installed where it runs
        itk = jobs["test-itk"]
        assert "SimpleITK" in itk and '-k "Itk or itk"' in itk and "-rs" in itk
        assert 'python -c "import SimpleITK, nibabel"' in itk

    def test_an_older_hdf5_reads_what_the_engine_writes(self):
        """The specification promises files HDF5 1.10 reads, and none was ever
        read by an HDF5 older than the 2.0 the test suite's h5py bundles (the
        engine links 2.2).  CI reads the valid cases, and their samples under
        each Blosc2 profile, with each series from the promised one up, and
        compares every value with what the newest h5py reads."""
        spec = (ROOT / "docs/spec/medh5-1.0.md").read_text(encoding="utf-8")
        promised = re.search(r"\*\*Container:\*\* HDF5 ≥ (\d+\.\d+) ", spec)
        assert promised
        job = _jobs(CI.read_text(encoding="utf-8"))["hdf5-readers"]
        series = re.findall(r'^ {10}- hdf5: "(\d+\.\d+)"$', job, re.M)
        assert series == ["1.10", "1.12", "1.14"] and series[0] == promised[1]
        script = "python .github/scripts/hdf5_readers.py"
        for step in (
            "write suite/ cases/",
            "read cases/ --out newest.json",
            "read cases/ --hdf5 ${{ matrix.hdf5 }} --compare newest.json",
        ):
            assert f"{script} {step}" in job, step

    def test_the_older_reader_fails_on_another_series_or_another_value(
        self, tmp_path, capsys
    ):
        """What the job runs fails when h5py runs another HDF5 series than its
        entry names, when a value reads otherwise than it did for the newest
        h5py, and when a file was not read at all."""
        import h5py

        pytest.importorskip("hdf5plugin")  # the script reads Blosc2 through it

        readers = _release_script("hdf5_readers.py")
        cases = tmp_path / "cases"
        cases.mkdir()
        write_sample(cases / "gzip.medh5")
        write_sample(cases / "blosc2.medh5", codec="balanced")
        newest = tmp_path / "newest.json"
        read = ["read", str(cases)]
        assert readers.main([*read, "--out", str(newest)]) == 0
        assert readers.main([*read, "--compare", str(newest)]) == 0
        running = h5py.version.hdf5_version.rsplit(".", 1)[0]
        assert readers.main([*read, "--hdf5", running]) == 0
        capsys.readouterr()
        assert readers.main([*read, "--hdf5", "1.8"]) == 1
        assert "expected HDF5 1.8.x" in capsys.readouterr().err
        found = json.loads(newest.read_text(encoding="utf-8"))
        attrs = found["blosc2.medh5"]["/"]["attrs"]
        attrs[sorted(attrs)[0]] = "0" * 64
        newest.write_text(json.dumps(found), encoding="utf-8")
        (cases / "gzip.medh5").unlink()
        assert readers.main([*read, "--compare", str(newest)]) == 1
        err = capsys.readouterr().err
        assert "gzip.medh5: not in" in err
        assert "blosc2.medh5: reads differently at /\n" in err

    def test_ci_jobs_are_bounded_and_cached_where_they_can_be(self):
        """No job had a timeout (GitHub's default is six hours), one failing
        Python version cancelled the others, every pull request kept eleven
        large artifacts for 90 days, and nothing was cached, so HDF5 compiled
        from source in every Rust job of every push."""
        for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
            for name, job in _jobs(path.read_text(encoding="utf-8")).items():
                if re.search(r"^    uses: ", job, re.M):
                    continue  # a called workflow's jobs carry their own
                timeout = re.search(r"^    timeout-minutes: (\d+)$", job, re.M)
                assert timeout and 0 < int(timeout[1]) <= 60, (path.name, name)
        text = CI.read_text(encoding="utf-8")
        jobs = _jobs(text)
        assert "fail-fast: false" in jobs["test"]
        cache = "uses: Swatinem/rust-cache@"
        for name in ("rust", "rust-platforms", "semver"):
            assert cache in jobs[name], name
            assert "save-if: ${{ github.ref == 'refs/heads/main' }}" in jobs[name]
        # Not where the release's artifacts are built, nor where the job is
        # what a clean build does.
        for name in ("sdist", "msvc-from-crates-io"):
            assert cache not in jobs[name], name
        # `build` saves on `main` and restores anywhere but on the tag, whose
        # artifacts the release publishes, and never in a container.
        built = BUILD.read_text(encoding="utf-8")
        step = built.split(cache, 1)[1].split("\n      - ", 1)[0]
        assert "key: ${{ inputs.target }}" in step
        assert "save-if: ${{ github.ref == 'refs/heads/main' }}" in step
        when = "if: ${{ !inputs.container && !startsWith(github.ref, 'refs/tags/') }}"
        assert built.count(when) == 2  # the toolchain its key reads, and it
        uploads = re.findall(
            r"upload-artifact@\S+ # v[\d.]+\n {8}with:\n((?: {10}.*\n)+)",
            text + built,
        )
        assert len(uploads) == 4
        short = "retention-days: ${{ github.event_name == 'pull_request' && 3 || 90 }}"
        assert all(short in upload for upload in uploads)

    def test_the_local_gate_is_one_chain_and_runs_what_ci_gates(self):
        """CLAUDE.md's chain lacked the native corpus run that CONTRIBUTING's
        had, and neither ran the notices check that fails CI whenever
        Cargo.lock changes without THIRD_PARTY_NOTICES.  The two are one
        chain, and it runs every check CI gates each change on.  (CLAUDE.md
        is not in the sdist, so from there CONTRIBUTING's is checked alone.)"""

        def chain(name: str, heading: str) -> str:
            text = (ROOT / name).read_text(encoding="utf-8")
            section = text.split(f"\n## {heading}\n", 1)[1]
            return section.split("```bash\n", 1)[1].split("```", 1)[0]

        contributing = chain("CONTRIBUTING.md", "The checks")
        if not (ROOT / "PKG-INFO").is_file():
            assert chain("CLAUDE.md", "Pre-commit checks") == contributing
        ci = CI.read_text(encoding="utf-8")
        for gate in (
            "cargo fmt --all -- --check",
            "cargo clippy --workspace --all-targets",
            "cargo test --workspace",
            "python .github/scripts/third_party_notices.py --check",
            "python .github/scripts/changelog.py check",
            "ruff check .",
            "ruff format --check .",
            "mypy medh5",
            "--cov-fail-under=90",
            "mkdocs build --strict",
        ):
            assert gate in ci and gate in contributing, gate
        assert "medh5 conformance run /tmp/corpus " in contributing
        native = "cargo run --release -p medh5-cli -- conformance run"
        assert f"{native} /tmp/corpus-native" in contributing

    def test_every_declared_floor_is_installed_by_ci(self):
        """The extras declared PyTorch 2.0, MONAI 1.3, nibabel 5, pydicom 3.0
        and highdicom 0.23 as their floors, and no job installed any of them;
        only NumPy's, and the test suite's HDF5 tools', were tested.  nibabel
        is the suite's own since 2.2.0, as h5py is."""
        text = PYPROJECT.read_text(encoding="utf-8")
        requirement = r'"([A-Za-z0-9_.-]+)>=([\d.]+)"'
        runtime = text.split("\ndependencies = [", 1)[1].split("]", 1)[0]
        floors = dict(re.findall(requirement, runtime))
        for extra, body in _extras().items():
            found = dict(re.findall(requirement, body))
            if extra == "dev":  # of the tools, the ones the suite reads files with
                tools = ("h5py", "hdf5plugin", "nibabel")
                found = {k: v for k, v in found.items() if k in tools}
            floors.update(found)
        assert set(floors) == {
            "numpy", "torch", "monai", "nibabel", "pydicom", "highdicom", "h5py",
            "hdf5plugin", "pyarrow",
        }  # fmt: skip
        job = _jobs(CI.read_text(encoding="utf-8"))["test-min-deps"]
        for name, floor in floors.items():
            if floor.count(".") == 2:  # a patch floor is installed exactly
                assert f'"{name}=={floor}"' in job, name
                continue
            series = floor if "." in floor else f"{floor}.0"
            assert f'"{name}=={series}.*"' in job, name
        oldest = re.search(r'^requires-python = ">=(3\.\d+)"$', text, re.M)
        assert oldest and f'python-version: "{oldest[1]}"' in job
        assert (
            'python -c "import torch, monai, nibabel, pydicom, highdicom, pyarrow"'
            in job
        )

    def test_the_commit_hooks_run_in_ci_at_the_gates_versions(self):
        """CI never ran the commit hooks, so a commit made without them
        installed reached `main` unchecked; and their `pre-commit-hooks` was a
        2024 release."""
        config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        revs = dict(
            re.findall(
                r"- repo: https://github.com/\S+/(\S+)\n.*?rev: (\S+)", config, re.S
            )
        )
        ruff = next(r for r in _extras()["dev"].split("\n") if '"ruff==' in r)
        assert revs["ruff-pre-commit"] == "v" + ruff.split("==")[1].split('"')[0]
        assert re.fullmatch(r"v\d+\.\d+\.\d+", revs["pre-commit-hooks"])
        assert int(revs["pre-commit-hooks"][1:].split(".")[0]) >= 6
        # Upstream's sources stay as upstream wrote them.
        assert re.search(r"^exclude: \^crates/medh5-sys/vendor/$", config, re.M)
        lint = _jobs(CI.read_text(encoding="utf-8"))["lint"]
        assert re.search(r'pip install "pre-commit==\d+\.\d+\.\d+"', lint)
        assert "pre-commit run --all-files" in lint

    def test_every_commit_on_main_is_tested(self):
        """One concurrency group for `main` keeps one run pending and cancels
        it when the next arrives, whatever `cancel-in-progress` says, so the
        middle of three quick merges went untested.  A pull request's newest
        push still supersedes its older runs."""
        header = CI.read_text(encoding="utf-8").split("\njobs:\n", 1)[0]
        group = re.search(r"^  group: (.+)$", header, re.M)
        assert group and "github.sha" in group[1] and "github.ref" in group[1]
        assert (
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in header
        )

    def test_patch_lines_are_tested_as_main_is(self):
        """CI ran on `main` only, so a pull request to `release/2.0` merged
        untested; the advisory audit's pull-request trigger was `main`'s too."""
        ci = CI.read_text(encoding="utf-8").split("\njobs:\n", 1)[0]
        on = "\n" + ci.split("\non:\n", 1)[1].split("\n  workflow_call:", 1)[0]
        for trigger in ("push", "pull_request"):
            block = on.split(f"\n  {trigger}:\n", 1)[1].split("\n  ", 1)[0]
            assert block.strip() == 'branches: [main, "release/**"]', trigger
        advisories = ADVISORIES.read_text(encoding="utf-8").split("\njobs:\n", 1)[0]
        pull = advisories.split("\n  pull_request:\n", 1)[1]
        assert pull.split("\n", 1)[0].strip() == 'branches: [main, "release/**"]'

    def test_an_advisory_against_an_extra_does_not_hold_a_release(self):
        """`pip-audit --strict` audited NumPy with every extra, and the release
        waited on it: a PyTorch advisory with no fix would have held a medh5
        release that changes nothing PyTorch installs.  What the package
        requires still blocks; the extras block every run but the release's,
        which reports them in its summary."""
        text = ADVISORIES.read_text(encoding="utf-8")
        call = text.split("\n  workflow_call:\n", 1)[1].split("\n  pull_request:", 1)[0]
        release = r"^      release:\n(?:        .*\n)*?        type: boolean$"
        assert re.search(release, call, re.M)
        jobs = _jobs(text)
        assert "continue-on-error" not in jobs["pypi"]
        assert 'print(*sorted(project["dependencies"])' in jobs["pypi"]
        assert "optional-dependencies" not in jobs["pypi"]
        extras = jobs["pypi-extras"]
        assert "    continue-on-error: ${{ inputs.release == true }}\n" in extras
        assert '"optional-dependencies"' in extras
        assert '>> "$GITHUB_STEP_SUMMARY"' in extras
        assert 'exit "$status"' in extras
        caller = _jobs(RELEASE.read_text(encoding="utf-8"))["advisories"]
        assert "    with:\n      release: true\n" in caller

    def test_the_lock_is_refreshed_weekly_but_not_the_native_inventory(self):
        """Cargo.lock moved only by hand: Dependabot cannot regenerate
        THIRD_PARTY_NOTICES with it.  A weekly workflow refreshes it within
        the crates' `rust-version` and opens a pull request, and puts back
        every crate that builds a library of SECURITY.md's native inventory,
        whose row and advisories a person reads again; its bumps are listed."""
        refresh = _release_script("lock_refresh.py")
        security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
        # The crates test_the_native_inventory_is_what_the_lock_builds names.
        native = ["hdf5-metno-src", "libz-sys", "lz4-sys", "zstd-sys"]
        assert refresh.native_crates(security) == native
        lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
        before = refresh.locked(lock)
        assert all(len(before[name]) == 1 for name in native)
        after = {**before, "libz-sys": ["9.9.9"], "serde": ["9.9.9"]}
        moves = refresh.held_back(before, after, native)
        assert moves == [("libz-sys", before["libz-sys"][0], "9.9.9")]
        assert refresh.held_back(before, before, native) == []
        with pytest.raises(SystemExit, match="zstd-sys"):
            refresh.held_back(before, {**before, "zstd-sys": ["1", "2"]}, native)
        body = refresh.summary(moves)
        assert f"| `libz-sys` | {before['libz-sys'][0]} | 9.9.9 |" in body
        assert "serde" not in body
        assert "No crate" in refresh.summary([])
        script = (ROOT / ".github/scripts/lock_refresh.py").read_text(encoding="utf-8")
        assert '"CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS": "fallback"' in script
        assert '"--precise", old' in script and "third_party_notices.py" in script
        text = LOCK_REFRESH.read_text(encoding="utf-8")
        assert '    - cron: "17 6 * * 1"\n' in text
        assert "\n  workflow_dispatch:\n" in text
        (job,) = _jobs(text).values()
        assert "          ref: main\n" in job
        assert "python .github/scripts/lock_refresh.py --summary" in job
        assert "uses: peter-evans/create-pull-request@" in job
        assert "branch: deps/cargo-lock" in job
        # A pull request opened with GITHUB_TOKEN runs no workflows.
        assert "secrets.LOCK_REFRESH_TOKEN" in job and "close and reopen" in job
        dependabot = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
        assert "lock-refresh.yml" in dependabot

    def test_the_weekly_audit_reads_every_line(self):
        """A scheduled workflow runs on the default branch only, so the weekly
        audit saw `main`'s lock and never the patch line's.  Each job audits
        both on the schedule, and a line not cut yet ends its job successfully
        at a step: a job-level `if:` would skip the job, and GitHub counts a
        skipped check as passing."""
        text = ADVISORIES.read_text(encoding="utf-8")
        jobs = _jobs(text)
        assert sorted(jobs) == ["pypi", "pypi-extras", "rustsec"]
        for name, job in jobs.items():
            assert not re.search(r"^    if:", job, re.M), name
            assert 'fromJSON(\'["main", "release/2.0"]\')' in job, name
            assert "ref: ${{ matrix.ref }}" in job, name
            assert 'gh api "repos/$GITHUB_REPOSITORY/git/ref/heads/$REF"' in job, name
            # Only a missing line skips; a failed lookup fails the job.
            assert "grep -q 'HTTP 404'" in job and "exit 1" in job, name
            # The check comes first, and every step after it waits on it.
            steps = job.split("\n      - ")[1:]
            assert "id: line" in steps[0], name
            for step in steps[1:]:
                assert "if: steps.line.outputs.exists == 'true'" in step, name

    def test_fuzzing_runs_on_a_dated_nightly(self):
        """Nothing fuzzed the readers: 2.0.2's denial-of-service bugs were
        found by hand.  The fuzz targets run twice a week on a nightly named
        by its date (one that moves would change the instrumentation under
        the corpus), each job bounded.  A run uploads nothing: a public
        repository's artifacts are public, and the input that broke a run
        works as an exploit until its fix ships."""
        text = FUZZ.read_text(encoding="utf-8")
        assert re.search(r"^  FUZZ_TOOLCHAIN: nightly-\d{4}-\d{2}-\d{2}$", text, re.M)
        assert re.search(r'^    - cron: "[^"]+"$', text, re.M)
        (job,) = _jobs(text).values()
        assert re.search(r"^    timeout-minutes: (\d+)$", job, re.M)[1] == "60"  # type: ignore[index]
        assert "cargo install cargo-fuzz --version 0.13.2 --locked" in job
        assert 'cargo +"$FUZZ_TOOLCHAIN" fuzz run "$TARGET"' in job
        assert "upload-artifact" not in job
        # The C libraries are instrumented as the Rust is, by the clang of the
        # nightly's own LLVM, with HDF5's free lists off --- in the fuzz job,
        # and in nothing that builds what ships.
        cflags = "-fsanitize=address,fuzzer-no-link -DH5_USING_MEMCHECKER=1"
        assert f"CFLAGS: {cflags}" in job
        assert "ASAN_OPTIONS: detect_leaks=0" in job
        assert 'llvm=$(rustc +"$FUZZ_TOOLCHAIN" -vV' in job
        # What the targets exclude is tested in every run.
        assert "rustc --edition 2021 --test src/lib.rs" in job
        # Only the pinned key is trusted, not whatever else the file holds.
        assert 'test "$primaries" = "$LLVM_KEY"' in job
        assert '"clang-$llvm" "libclang-rt-$llvm-dev"' in job
        assert 'echo "CC=clang-$llvm" >> "$GITHUB_ENV"' in job
        workflows = (ROOT / ".github/workflows").glob("*.yml")
        builds = [*workflows, *(ROOT / ".github/scripts").iterdir()]
        for path in builds:
            if path != FUZZ and path.is_file():
                other = path.read_text(encoding="utf-8")
                assert "fsanitize" not in other, path.name
                assert "H5_USING_MEMCHECKER" not in other, path.name
        if (ROOT / "PKG-INFO").is_file():
            return  # the sdist carries no fuzz targets
        # Every target runs, in a workspace of its own: the root's lock and
        # notices never see it.
        fuzz = (ROOT / "fuzz/Cargo.toml").read_text(encoding="utf-8")
        declared = re.findall(r'^path = "fuzz_targets/(\w+)\.rs"$', fuzz, re.M)
        targets = re.search(r"^        target: \[(.*)\]$", job, re.M)
        assert targets and [t.strip() for t in targets[1].split(",")] == declared
        assert "\n[workspace]\n" in fuzz and "publish = false" in fuzz

    def test_the_release_runs_ci_on_the_tagged_commit(self):
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        assert "workflow_call:" in ci and "concurrency:" in ci
        assert "uses: ./.github/workflows/ci.yml" in release
        build = release[release.index("  build:") :]
        assert "needs: [ci, advisories]" in build.split("\n  publish:")[0]
        # The two builds upload in one run when CI is called; names must differ.
        assert "name: ci-dist" in ci and "name: dist" in release


class TestDistribution:
    """What the release ships, and how it ships it."""

    def test_the_required_check_needs_every_job(self):
        """The ruleset required `build` and `test-macos (3.12)`, names no job
        reports --- a matrix reports `build (<target>)` --- so no pull request
        could merge, and a job added later would not have been required.  One
        aggregate job is required instead; it must need every other one, and
        run when one failed (a skipped required check counts as passing)."""
        jobs = _jobs(CI.read_text(encoding="utf-8"))
        gate = jobs.pop("ci-ok")
        needs = re.findall(r"^      - ([\w-]+)$", gate.split("steps:")[0], re.M)
        # A job that cannot fail the run is information, and the gate does not
        # wait for it.
        informational = {
            name
            for name, job in jobs.items()
            if re.search(r"^    continue-on-error: true$", job, re.M)
        }
        assert informational == {"test-python-next"}
        assert sorted(needs) == sorted(set(jobs) - informational)
        assert re.search(r"^    if: always\(\)$", gate, re.M)
        # A cancelled job fails the gate too, named apart in the summary.
        assert '("success", "cancelled")' in gate and "if failed or cancelled:" in gate
        assert "cancelled or superseded: re-run" in gate

    def test_the_api_is_compared_with_the_last_release(self):
        """2.0.2's "no public API change" was checked by hand, and the patch
        still changed Rust structs.  The crates go through cargo-semver-checks
        and the package through public_api.py, each against the newest release
        at or below the tree, and the gate waits for both."""
        jobs = _jobs(CI.read_text(encoding="utf-8"))
        semver, python = jobs["semver"], jobs["python-api"]
        baseline = "python .github/scripts/release.py api-baseline --registry"
        assert f"{baseline} crates" in semver and f"{baseline} pypi" in python
        merge_group = "github.event.merge_group.base_ref"
        assert merge_group in semver and merge_group in python
        assert "cargo install cargo-semver-checks --version 0.51.0 --locked" in semver
        assert "cargo semver-checks -p medh5-sys -p medh5 -p medh5-cli" in semver
        assert '--release-type "${{ steps.base.outputs.release_type }}"' in semver
        # The rustdoc JSON formats 0.51.0 reads: Rust 1.97 to 1.99.
        pin = ROOT / "rust-toolchain.toml"
        if pin.is_file():  # the sdist carries no pin
            toolchain = pin.read_text(encoding="utf-8")
            channel = re.search(r'^channel = "1\.(\d+)\.\d+"$', toolchain, re.M)
            assert channel and 97 <= int(channel[1]) <= 99, toolchain
        # The released wheel and this run's, neither read from the checkout.
        assert re.search(r"^    needs: build-linux-x86_64$", python, re.M)
        assert 'cd "$RUNNER_TEMP"' in python
        assert python.count('/bin/python -I "$script" dump --output') == 2
        assert '--release-type "${{ steps.base.outputs.release_type }}"' in python
        gate = jobs["ci-ok"].split("steps:")[0]
        assert "      - semver\n" in gate and "      - python-api\n" in gate

    def test_each_platform_waits_only_for_its_own_build(self):
        """Every test job needed the one five-target `build`, so no Python test
        started before the slowest build ended.  A job that downloads a
        target's wheel or binary needs that target's build job, and no other."""
        jobs = _jobs(CI.read_text(encoding="utf-8"))
        build_of = {
            re.search(r"^      target: (\S+)$", job, re.M)[1]: name  # type: ignore[index]
            for name, job in jobs.items()
            if "uses: ./.github/workflows/build.yml" in job
        }
        assert len(build_of) == 6
        downloading = 0
        for name, job in jobs.items():
            targets = set(re.findall(r"name: ci-(?:dist|cli)-([\w-]+)$", job, re.M))
            targets.discard("sdist")
            needs = re.search(r"^    needs: (.+)$", job, re.M)
            needed = (set(re.findall(r"[\w-]+", needs[1])) if needs else set()) & set(
                build_of.values()
            )
            if name in build_of.values():
                continue
            if targets:
                downloading += 1
            assert needed == {build_of[t] for t in targets}, name
        assert downloading == 16

    def test_every_shipped_target_is_built_released_and_noticed(self):
        """The targets were listed four times --- the build jobs, the
        release's wheel platforms, its binaries, and the notices' platform
        filter --- and nothing held the lists together: a target added to one
        would ship without its notices, or be built and never released."""
        jobs = _jobs(CI.read_text(encoding="utf-8"))
        built = {
            re.search(r"^      target: (\S+)$", job, re.M)[1]  # type: ignore[index]
            for job in jobs.values()
            if "uses: ./.github/workflows/build.yml" in job
        }
        verify = _jobs(RELEASE.read_text(encoding="utf-8"))["build"]
        table = verify.split("platforms = {", 1)[1].split("}", 1)[0]
        released = dict(re.findall(r'"([\w-]+)": "(\w+)",', table))
        notices = _release_script("third_party_notices.py")
        assert built == set(released) == set(notices.TARGETS)
        for target, platform in released.items():
            # Each wheel tag names the target's architecture as its platform
            # does: aarch64 on Linux, arm64 on macOS, amd64 on Windows.
            arch = target.split("-")[0]
            names = {"aarch64": ("aarch64", "arm64"), "x86_64": ("x86_64", "amd64")}
            assert platform.endswith(names[arch]), (target, platform)
        # Each platform's wheels: abi3, and one for free-threaded 3.14.
        assert '("cp310-abi3", "cp314-cp314t")' in verify
        assert 'f"medh5-{raw}-{target}.{archive}"' in verify

    def test_free_threaded_python_gets_a_wheel_of_its_own(self):
        """Free-threaded CPython cannot load an abi3 extension, so medh5 did
        not install there without a compiler.  Each build makes a second,
        cp314t wheel, and a gating job tests it with the GIL off: the whole
        suite, then the format and tools tests from four threads at once."""
        build = (ROOT / ".github/scripts/build-dist.sh").read_text(encoding="utf-8")
        assert "MEDH5_FT_PYTHON=/opt/python/cp314-cp314t/bin/python" in build
        assert '--interpreter "$MEDH5_FT_PYTHON"' in build
        workflow = _jobs(BUILD.read_text(encoding="utf-8"))["build"]
        assert 'python-version: "3.14t"' in workflow
        assert "MEDH5_FT_PYTHON: ${{ steps.free-threaded.outputs.python-path }}" in (
            workflow
        )
        jobs = _jobs(CI.read_text(encoding="utf-8"))
        job = jobs["test-free-threaded"]
        assert 'python-version: "3.14t"' in job
        assert "dist/*-cp314t-*.whl" in job
        assert 'PYTHON_GIL: "0"' in job
        assert "not sys._is_gil_enabled()" in job
        assert "pytest tests/format tests/tools" in job
        assert "--parallel-threads=4" in job
        assert "test-no-gil" not in jobs, "folded into test-free-threaded"
        # Every other job installs the abi3 wheel by name: two wheels match
        # `dist/*.whl` now.
        for name, job in jobs.items():
            assert "dist/*.whl" not in job, name

    def test_musl_gets_wheels_and_a_static_binary(self):
        """Alpine and the other musl Linux built medh5 from source, which
        takes a compiler, CMake and minutes.  Each architecture has musllinux
        wheels and a binary that links musl statically, so it runs on any
        Linux; the wheels are tested in the image that built them, and the
        binary is checked to need no shared library."""
        jobs = _jobs(CI.read_text(encoding="utf-8"))
        build = (ROOT / ".github/scripts/build-dist.sh").read_text(encoding="utf-8")
        assert "--compatibility musllinux_1_2" in build
        assert 'readelf -d "$exe" | grep -q NEEDED' in build
        for arch in ("x86_64", "aarch64"):
            target = f"{arch}-unknown-linux-musl"
            caller = jobs[f"build-linux-musl-{arch}"]
            assert f"      target: {target}\n" in caller
            image = re.search(r"^      container: (\S+)$", caller, re.M)
            assert image and image[1].startswith(f"quay.io/pypa/musllinux_1_2_{arch}@")
            test = jobs[f"test-linux-musl-{arch}"]
            # Tested in the image it was built in.
            assert f"      IMAGE: {image[1]}\n" in test
            assert f"bash .github/scripts/test-musl.sh {target}" in test
        script = (ROOT / ".github/scripts/test-musl.sh").read_text(encoding="utf-8")
        assert '!= "hdf5plugin"' in script and '"dev"' in script
        assert "--image expects NAME=VALUE" in script
        assert "pytest tests/ -v -rs" in script
        # The checkout's own `medh5/` has no engine, and shadowed the
        # installed one for the first import run from the checkout.
        assert script.index("rm -rf medh5") < script.index("import medh5")
        # The glibc host cannot install a musl wheel: those checks are the
        # test job's.
        steps = _jobs(BUILD.read_text(encoding="utf-8"))["build"].split("\n      - ")
        for name in ("The wheel installs", "The binary hands the converters"):
            (step,) = [s for s in steps if name in s]
            assert "if: ${{ !endsWith(inputs.target, '-musl') }}" in step

    def test_macos_on_intel_is_not_built(self):
        """2.2.0 dropped macOS on Intel.  Its build was the slowest of every
        run on an Intel runner, then cross-built on Apple silicon and tested
        under Rosetta (#70); now nothing builds, tests or ships it, and every
        job runs on every event, pull requests included, on a current image."""
        text = CI.read_text(encoding="utf-8")
        jobs = _jobs(text)
        for name, job in jobs.items():
            assert "github.event_name != 'pull_request'" not in job, name
        runners = set(re.findall(r"(?:runner|runs-on): \[?.*?\b(macos-[\w.-]+)", text))
        assert runners == {"macos-15"}, runners
        assert "x86_64-apple-darwin" not in text
        assert not any("macos-x86_64" in name for name in jobs)
        assert "arch -x86_64" not in text
        # The gate accepts no skip but the one `changes` asks for.
        gate = jobs["ci-ok"]
        assert "EVENT" not in gate
        # Nothing is left of the Rosetta checks.
        workflow = BUILD.read_text(encoding="utf-8")
        assert "rosetta" not in workflow.lower()
        assert "architecture:" not in workflow
        release = RELEASE.read_text(encoding="utf-8")
        assert (
            "x86_64-apple-darwin" not in release and "macosx_11_0_x86_64" not in release
        )

    def test_every_build_names_its_target(self):
        """build-dist.sh built for its host alone: it passed no `--target`, and
        took the binary from `target/release`.  Each build names the target,
        the binary is taken from that target's directory, and on macOS, where
        Rosetta runs either architecture, what was built is checked by
        reading it rather than by running it."""
        build = (ROOT / ".github/scripts/build-dist.sh").read_text(encoding="utf-8")
        builds = re.findall(r"^(?:maturin|cargo) build .*$", build, re.M)
        assert len(builds) == 3
        assert all('--target "$target"' in line for line in builds), builds
        assert 'rustup target add "$target"' in build
        assert 'exe="target/$target/release/medh5"' in build
        assert 'lipo -archs "$built"' in build
        assert '"$unpacked"/*/medh5/_core*.so' in build
        workflow = BUILD.read_text(encoding="utf-8")
        assert "target/release" not in workflow
        assert workflow.count("target/${{ inputs.target }}/release/medh5") == 2

    def test_the_next_python_is_tried_and_not_claimed(self):
        """The classifiers name the interpreters the gating matrix tests, and
        the next one runs the suite as information until it can join them:
        3.15 was out, and nothing had run on it."""
        jobs = _jobs(CI.read_text(encoding="utf-8"))
        matrix = re.search(r"python-version: \[(.*)\]", jobs["test"])
        assert matrix
        tested = [v.strip().strip('"') for v in matrix[1].split(",")]
        text = PYPROJECT.read_text(encoding="utf-8")
        claimed = re.findall(r'"Programming Language :: Python :: (3\.\d+)"', text)
        assert claimed == tested
        newest = max(tested, key=lambda v: int(v.split(".")[1]))
        following = f"3.{int(newest.split('.')[1]) + 1}"
        assert f'python-version: "{following}"' in jobs["test-python-next"]

    def test_the_notices_name_what_is_compiled_in(self):
        """Every crate the notices list is Cargo.lock's, at its version, and
        the C libraries are there; CI regenerates the file and fails on a
        difference (`--check`), which needs the crate sources this cannot."""
        notices = (ROOT / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
        lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
        locked = set(re.findall(r'^name = "([^"]+)"\nversion = "([^"]+)"', lock, re.M))
        table = notices.split("\nRust crates (", 1)[1].split("\nLicence texts\n")[0]
        listed = set(re.findall(r"^  (\S+) (\S+) ", table, re.M))
        assert len(listed) > 100 and listed <= locked, sorted(listed - locked)
        for library in ("HDF5", "C-Blosc2", "HDF5-Blosc2 filter", "zlib", "LZ4"):
            assert re.search(rf"^  {re.escape(library)} ", notices, re.M), library
        # Compiled into every Rust program, and in no lockfile.
        assert re.search(
            r"^  std, core and alloc +MIT OR Apache-2.0 +\[\d+\]$", notices, re.M
        )
        assert "--check" in _jobs(CI.read_text(encoding="utf-8"))["rust"]

    def test_every_artifact_carries_the_licences(self):
        """The wheel (as `license-files`), the sdist (with them), each binary
        archive, and each crate --- whose MIT licence must travel with it."""
        assert (
            'license-files = ["LICENSE", "THIRD_PARTY_NOTICES"]'
            in PYPROJECT.read_text(encoding="utf-8")
        )
        build = (ROOT / ".github/scripts/build-dist.sh").read_text(encoding="utf-8")
        assert 'cp "$exe" LICENSE THIRD_PARTY_NOTICES' in build
        release = RELEASE.read_text(encoding="utf-8")
        assert ".dist-info/licenses/{notice}" in release and 'f"/{notice}"' in release
        licence = (ROOT / "LICENSE").read_bytes()
        for crate in ("medh5", "medh5-cli", "medh5-sys"):
            assert (ROOT / "crates" / crate / "LICENSE").read_bytes() == licence, crate
        sys_manifest = (ROOT / "crates/medh5-sys/Cargo.toml").read_text(
            encoding="utf-8"
        )
        assert '"LICENSE",' in sys_manifest.split("include = [", 1)[1].split("]")[0]

    def test_the_release_profile_strips_the_symbol_table(self):
        """The wheels and binaries shipped their symbol tables: 5.4 MB of the
        30 MB Linux extension, read by nothing that ships.  Stripping keeps
        the dynamic symbols Python imports the module by; nothing in the
        release uploads symbols separately."""
        workspace = WORKSPACE.read_text(encoding="utf-8")
        profile = workspace.split("\n[profile.release]\n", 1)[1].split("\n[", 1)[0]
        assert re.search(r'^strip = "symbols"$', profile, re.M)
        release = RELEASE.read_text(encoding="utf-8")
        build = (ROOT / ".github/scripts/build-dist.sh").read_text(encoding="utf-8")
        for text in (release, build):
            assert not re.search(r"debuginfo|\.pdb\b|dSYM|--strip", text)

    def test_the_installed_package_carries_the_notices(self):
        names = {str(f) for f in distribution("medh5").files or []}
        for notice in ("LICENSE", "THIRD_PARTY_NOTICES"):
            assert any(n.endswith(f".dist-info/licenses/{notice}") for n in names), (
                notice
            )

    def test_the_generator_covers_every_crate_or_refuses(self, tmp_path):
        notices = _release_script("third_party_notices.py")

        def package(name: str, licence: str, files: dict[str, str]) -> dict:
            directory = tmp_path / name
            directory.mkdir()
            for file, text in {"Cargo.toml": "", **files}.items():
                (directory / file).write_bytes(text.encode())
            return {
                "id": name,
                "name": name,
                "version": "1.0.0",
                "license": licence,
                "manifest_path": str(directory / "Cargo.toml"),
            }

        def dep(name: str, kind: str | None = None) -> dict:
            return {"pkg": name, "dep_kinds": [{"kind": kind, "target": None}]}

        apache = "Apache License\r\nVersion 2.0   \r\n\r\n"
        packages = [
            package("medh5-python", "MIT", {}),
            package("medh5-cli", "MIT", {}),
            package("a", "MIT OR Apache-2.0", {"LICENSE-APACHE": apache}),
            package("b", "Apache-2.0", {"LICENSE": "Apache License\nVersion 2.0\n"}),
            package("c", "MIT", {}),
            package("build-only", "GPL-3.0-only", {}),
        ]
        nodes = [
            {
                "id": "medh5-python",
                "deps": [dep("a"), dep("c"), dep("build-only", "build")],
            },
            {"id": "medh5-cli", "deps": [dep("b")]},
            *({"id": n, "deps": []} for n in ("a", "b", "c", "build-only")),
        ]
        metadata = {
            "packages": packages,
            "workspace_members": ["medh5-python", "medh5-cli"],
            "resolve": {"nodes": nodes},
        }
        shipped = notices.shipped(metadata)
        assert [p["name"] for p in shipped] == ["a", "b", "c"]
        texts = notices.Texts()
        for p in shipped:
            for where, text in notices.crate_texts(p):
                texts.cite(text, f"{p['name']}: {where}")
        # One Apache text however its lines end, and the MIT notice for the
        # crate that ships no file.
        assert len(texts.numbers) == 2
        assert notices.crate_texts(shipped[2])[0][1] == notices.MIT
        with pytest.raises(SystemExit, match="is not MIT"):
            notices.crate_texts(package("d", "BSD-3-Clause", {}))
        assert notices.offers_mit("Apache-2.0 OR MIT")
        assert not notices.offers_mit("(MIT OR Apache-2.0) AND Unicode-3.0")

    def test_the_maturin_floor_writes_pep_639_metadata(self):
        """1.9.0 wrote the legacy `License` field and 1.8 could not read the
        table; the floor is 1.9.3, and CI builds the sdist with exactly it."""
        text = PYPROJECT.read_text(encoding="utf-8")
        assert re.findall(r'"maturin>=([\d.]+),<2"', text) == ["1.9.3", "1.9.3"]
        sdist = _jobs(CI.read_text(encoding="utf-8"))["sdist"]
        assert '["build-system"]["requires"]' in sdist
        assert "--no-build-isolation" in sdist
        assert 'f"License-Expression: {expression}"' in sdist
        assert "['License-Expression'] ==" in sdist and "['license']" in sdist

    def test_the_declared_licence_covers_what_is_compiled_in(self):
        """The wheels declared `License-Expression: MIT` while compiling in
        HDF5, C-Blosc2, Zstandard and `subtle` (BSD-3-Clause), LZ4 and
        rust-numpy (BSD-2-Clause), zlib and `foldhash` (Zlib),
        `unicode-general-category` (Apache-2.0) and `borrow-or-share` (MIT-0)
        --- and PyPI cannot correct a release's metadata.  The binary links the
        same libraries, so the Homebrew formula states the same set."""
        (expression,) = re.findall(
            r'^license = "([^"]+)"$', PYPROJECT.read_text(encoding="utf-8"), re.M
        )
        declared = expression.split(" AND ")
        assert declared[0] == "MIT", "the package's own licence leads"
        assert set(declared) == {
            "MIT", "BSD-3-Clause", "BSD-2-Clause", "Zlib", "Apache-2.0", "MIT-0"
        }  # fmt: skip
        notices = (ROOT / "THIRD_PARTY_NOTICES").read_text(encoding="utf-8")
        for licence in ("BSD-3-Clause", "BSD-2-Clause", "Zlib", "MIT-0"):
            assert re.search(rf"  {re.escape(licence)} +\[", notices), licence
        brew = _release_script("homebrew_formula.py")
        assert list(brew.LICENSES) == declared
        sums = {brew.archive("2.0.0", t): "0" * 64 for t in brew.TARGETS.values()}
        formula = brew.formula("2.0.0", sums, "medh5-project/medh5")
        assert f"  license all_of: {json.dumps(declared)}\n" in formula

    def test_no_crate_brings_a_licence_the_package_does_not_declare(self):
        """Nothing checked a new crate's licence against the expression the
        wheels publish: one under the MPL would have shipped under metadata
        that does not name it.  `cargo deny` allows exactly that expression's
        licences, and ci.yml's rust job runs it with sources and bans."""
        (expression,) = re.findall(
            r'^license = "([^"]+)"$', PYPROJECT.read_text(encoding="utf-8"), re.M
        )
        # Read by pattern: Python 3.10, which the suite still runs on, has no
        # tomllib.
        deny = (ROOT / "deny.toml").read_text(encoding="utf-8")
        allow = re.search(r"^allow = \[([^\]]*)\]$", deny, re.M)
        assert allow
        allowed = re.findall(r'"([^"]+)"', allow[1])
        assert sorted(allowed) == sorted(expression.split(" AND "))
        assert len(allowed) == len(set(allowed))
        assert re.search(r'^unknown-registry = "deny"$', deny, re.M)
        assert re.search(r'^unknown-git = "deny"$', deny, re.M)
        clarify = deny.split("[[licenses.clarify]]", 1)[1]
        assert 'crate = "hdf5-metno-src"' in clarify
        assert re.search(r'path = "ext/hdf5/LICENSE", hash = 0x[0-9a-f]{8}', clarify)
        rust = _jobs(CI.read_text(encoding="utf-8"))["rust"]
        assert "cargo install cargo-deny --version 0.20.2 --locked" in rust
        assert "cargo deny --locked check licenses bans sources" in rust

    def test_every_extra_is_one_the_package_asks_for(self):
        """An extra promises that something in the package needs it, so the
        package names each where it does --- `require(..., extra=...)`, or an
        install hint --- and names no other.  `itk` (SimpleITK) was named
        nowhere: CI installed it, and nothing imported it."""
        declared = set(_extras()) - {"dev"}
        source = _package_source()
        named = set(re.findall(r'extra="([\w-]+)"', source))
        for hint in re.findall(r"medh5\[([\w,-]+)\]", source):
            named.update(hint.split(","))
        assert declared - named == set(), "extras nothing in the package needs"
        assert named - declared == set(), "extras the package names but lacks"

    def test_the_dicom_extras_carry_what_the_writers_call(self):
        """The RTSTRUCT writer asks `save_as` to enforce the DICOM file
        format, a keyword pydicom 3.0 added; highdicom supports pydicom 3 from
        0.23.  The floors were pydicom 2.4 and highdicom 0.22, where writing
        an RTSTRUCT is a `TypeError`.  And highdicom 0.23 requires pydicom
        3.0.1, so a 3.0.0 floor was one no resolver could install."""
        rtstruct = Path(medh5.__file__).parent / "io" / "rtstruct.py"
        assert "enforce_file_format=" in rtstruct.read_text(encoding="utf-8")
        extras = _extras()
        for extra in ("dicom", "dicomseg"):
            assert re.findall(r'"pydicom([^"]*)"', extras[extra]) == [">=3.0.1"]
        assert re.findall(r'"highdicom([^"]*)"', extras["dicomseg"]) == [">=0.23"]

    def test_a_published_crate_is_skipped_only_when_identical(self):
        """A re-run after a partial release skips a crate already on crates.io
        only if its contents are the tagged commit's; cargo's generated files,
        whose spelling follows the cargo version, are not compared."""
        release = _release_script("release.py")

        def crate(files: dict[str, bytes]) -> bytes:
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                for name, data in files.items():
                    info = tarfile.TarInfo(f"medh5-2.0.0/{name}")
                    info.size, info.mtime = len(data), random.randrange(2**31)
                    archive.addfile(info, io.BytesIO(data))
            return buffer.getvalue()

        source = {"src/lib.rs": b"fn f() {}", "Cargo.toml.orig": b"[package]"}
        ours = release.crate_files(crate({**source, "Cargo.toml": b"cargo 1.97"}))
        theirs = release.crate_files(crate({**source, "Cargo.toml": b"cargo 1.95"}))
        assert release.differences(ours, theirs) == []
        edited = release.crate_files(crate({**source, "src/lib.rs": b"fn g() {}"}))
        assert release.differences(ours, edited) == ["src/lib.rs: differs"]
        extra = release.crate_files(crate({**source, "build.rs": b""}))
        assert release.differences(ours, extra) == [
            "build.rs: only in the published one"
        ]

    def test_dists_and_assets_are_compared_by_digest(self):
        release = _release_script("release.py")
        local = {"medh5-2.0.0.tar.gz": "a", "medh5-2.0.0-cp310-abi3-x.whl": "b"}
        assert release.dist_version(list(local)) == "2.0.0"
        assert release.dist_problems(local, dict(local)) == []
        assert release.dist_problems(local, {"medh5-2.0.0.tar.gz": "z"}) == [
            "medh5-2.0.0-cp310-abi3-x.whl: not published",
            "medh5-2.0.0.tar.gz: published with different content",
        ]
        upload, download = release.release_plan(
            {"x": "1", "y": "2", "z": "3"}, {"x": "1", "y": None}
        )
        assert (upload, download) == (["z"], ["y"])
        assert release.formula_version(
            'class Medh5 < Formula\n  version "2.0.0"\n'
        ) == ("2.0.0")
        with pytest.raises(SystemExit, match="2 versions"):
            release.dist_version(["medh5-2.0.0.tar.gz", "medh5-2.0.1.tar.gz"])

    def test_a_backport_is_not_latest_and_leaves_the_tap(self, tmp_path, monkeypatch):
        """`gh release create` let GitHub mark every final release "Latest",
        and the tap took every final release's formula: a 2.0.x patch tagged
        after 2.1.0 would have become the repository's Latest and moved
        Homebrew users back to 2.0.  Only the newest final release tagged
        is either."""
        release = _release_script("release.py")
        tags = ["v2.0.0", "v2.0.1", "v2.1.0-rc.1", "v2.1.0", "v1.4.4", "vfoo"]
        assert release.newest("v2.1.0", tags)
        assert release.newest("v2.1.1", [*tags, "v2.1.1"])
        assert release.newest("v2.2.0", tags)
        assert not release.newest("v2.0.2", [*tags, "v2.0.2"])
        assert not release.newest("v1.4.5", tags)
        assert not release.newest("v2.2.0-rc.1", tags)
        # A pre-release tagged after a final one does not make it a backport.
        assert release.newest("v2.1.0", [*tags, "v2.2.0-rc.1"])

        created: list[tuple[str, ...]] = []
        monkeypatch.setattr(release, "run", lambda *c, **_: created.append(c) or "")
        missing = SimpleNamespace(returncode=1, stdout="", stderr="")
        monkeypatch.setattr(
            release, "subprocess", SimpleNamespace(run=lambda *a, **k: missing)
        )
        monkeypatch.setenv("GITHUB_REPOSITORY", "medh5-project/medh5")
        (tmp_path / "notes.md").write_text("notes", encoding="utf-8")
        files = [tmp_path / "notes.md"]
        for argv, flags in (
            ([], set()),
            (["--not-latest"], {"--latest=false"}),
            (["--prerelease"], {"--prerelease"}),
        ):
            created.clear()
            release.main(["github-release", "v2.0.2", *map(str, files), *argv])
            (command,) = created
            assert {"--latest=false", "--prerelease"} & set(command) == flags, argv
        with pytest.raises(SystemExit):
            release.main(["github-release", "v2", "x", "--prerelease", "--not-latest"])

        jobs = _jobs(RELEASE.read_text(encoding="utf-8"))
        assert "latest: ${{ steps.version.outputs.latest }}" in jobs["build"]
        newest = 'release.py newest "$GITHUB_REF_NAME"'
        assert f'{newest} >> "$GITHUB_OUTPUT"' in jobs["build"]
        assert "flags+=(--not-latest)" in jobs["github-release"]
        assert "if: needs.build.outputs.latest == 'true'" in jobs["homebrew"]
        assert re.search(
            r'if \[\[ "\$LATEST" == true \]\]; then\n\s+flags\+=\(--tap "\$TAP"\)',
            jobs["verify"],
        )

    def test_only_the_distributions_are_compared_with_pypi(self, tmp_path, monkeypatch):
        """Under Trusted Publishing the upload signs every distribution and
        writes its attestation beside it, as `<file>.publish.attestation`.
        The check after the upload read those as distributions of a second
        version and failed with PyPI already holding the release, so neither
        the release page nor the formula followed --- and a re-run signed, and
        failed, again."""
        release = _release_script("release.py")
        dist, cli = tmp_path / "dist", tmp_path / "cli"
        dist.mkdir()
        cli.mkdir()
        names = ["medh5-2.0.0-cp310-abi3-win_amd64.whl", "medh5-2.0.0.tar.gz"]
        for name in names:
            (dist / name).write_bytes(name.encode())
            (dist / f"{name}.publish.attestation").write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit, match="2 versions"):
            release.dist_version(list(release.digests(dist)))
        local = release.dist_digests(dist)
        assert sorted(local) == sorted(names)
        assert release.dist_version(list(local)) == "2.0.0"
        published = {name: release.sha256(name.encode()) for name in names}
        monkeypatch.setattr(release, "pypi_digests", lambda version: published)
        assert release.check_pypi(dist, wait=0) == 0
        # And `verify`, which reads the same directory.
        (cli / "medh5-2.0.0-x.tar.gz").write_bytes(b"x")
        assets = {"assets": [{"name": n} for n in [*names, "medh5-2.0.0-x.tar.gz"]]}
        monkeypatch.setattr(
            release, "crates_io_status", lambda name, version: "published"
        )
        monkeypatch.setattr(
            release, "fetch", lambda url, **_: json.dumps(assets).encode()
        )
        monkeypatch.setenv("GITHUB_REPOSITORY", "medh5-project/medh5")
        assert release.verify("v2.0.0", dist, cli, tap=None, token=None) == 0

    def test_a_yanked_crate_is_neither_published_again_nor_called_missing(
        self, tmp_path, monkeypatch, capsys
    ):
        """crates.io keeps a yanked version and never takes it again, and the
        script read one as unpublished: the publishing step ran `cargo
        publish` into crates.io's refusal, and `verify` called the crate
        missing."""
        release = _release_script("release.py")
        records = {
            "medh5-sys": {"version": {"yanked": False}},
            "medh5": {"version": {"yanked": True}},
        }

        def fetch(url: str, **_: object) -> bytes | None:
            name = url.split("/")[-2]
            return json.dumps(records[name]).encode() if name in records else None

        monkeypatch.setattr(release, "fetch", fetch)
        status = {n: release.crates_io_status(n, "2.0.0") for n in release.CRATES}
        assert status == {
            "medh5-sys": "published", "medh5": "yanked", "medh5-cli": "absent"
        }  # fmt: skip
        ran: list[tuple[str, ...]] = []
        monkeypatch.setattr(release, "run", lambda *cmd, **_: ran.append(cmd) or "")
        monkeypatch.setattr(release, "crates_io_status", lambda name, version: "yanked")
        with pytest.raises(SystemExit, match=r"medh5-sys \S+ is on crates.io, yanked"):
            release.publish_crates()
        assert ran == []
        dist, cli = tmp_path / "dist", tmp_path / "cli"
        dist.mkdir()
        cli.mkdir()
        (dist / "medh5-2.0.0.tar.gz").write_bytes(b"sdist")
        monkeypatch.setattr(release, "pypi_digests", lambda v: release.digests(dist))
        assets = {"assets": [{"name": "medh5-2.0.0.tar.gz"}]}
        monkeypatch.setattr(
            release, "fetch", lambda url, **_: json.dumps(assets).encode()
        )
        monkeypatch.setenv("GITHUB_REPOSITORY", "medh5-project/medh5")
        monkeypatch.setattr(release, "crates_io_status", lambda name, v: status[name])
        capsys.readouterr()
        assert release.verify("v2.0.0", dist, cli, tap=None, token=None) == 1
        assert capsys.readouterr().err.splitlines() == [
            "crates.io: medh5 2.0.0 is yanked",
            "crates.io: medh5-cli 2.0.0 is not published",
        ]

    def test_every_crate_is_verified_before_any_is_published(self, monkeypatch):
        """`cargo publish` verified each crate with a long-lived token in the
        environment of every dependency's build script, one crate at a time,
        so a failure verifying `medh5-cli` left `medh5-sys` and `medh5`
        published.  Every crate is verified first, with no token, as CI's
        `rust` job packages them; trusted publishing then issues a token for
        the upload alone, and each crate is in the index before the next."""
        release = _release_script("release.py")
        steps: list[tuple[str, ...]] = []

        def run(*command: str, capture: bool = False, env=None) -> str:
            token = "CARGO_REGISTRY_TOKEN" in (os.environ if env is None else env)
            steps.append((*command, "token" if token else "no token"))
            return ""

        monkeypatch.setenv("CARGO_REGISTRY_TOKEN", "secret")
        monkeypatch.setattr(release, "run", run)
        monkeypatch.setattr(release, "crates_io_status", lambda name, version: "absent")
        monkeypatch.setattr(
            release,
            "wait_for_index",
            lambda name, version: steps.append(("index", name)),
        )
        assert release.main(["crates", "verify"]) == 0
        packages = ("-p", "medh5-sys", "-p", "medh5", "-p", "medh5-cli")
        assert steps == [("cargo", "package", "--locked", *packages, "no token")]
        steps.clear()
        assert release.main(["crates", "publish"]) == 0
        assert steps == [
            step
            for name in ("medh5-sys", "medh5", "medh5-cli")
            for step in (
                ("cargo", "publish", "--locked", "--no-verify", "-p", name, "token"),
                ("index", name),
            )
        ]
        # The verification is CI's own, so the release's is never the first.
        rust = _jobs(CI.read_text(encoding="utf-8"))["rust"]
        assert f"cargo package --locked {' '.join(packages)}" in rust
        jobs = _jobs(RELEASE.read_text(encoding="utf-8"))
        crates = jobs["crates"]
        order = [
            crates.index("release.py crates verify"),
            crates.index("uses: rust-lang/crates-io-auth-action@"),
            crates.index("release.py crates publish"),
        ]
        assert order == sorted(order)
        assert "CARGO_REGISTRY_TOKEN: ${{ steps.auth.outputs.token }}" in crates
        assert "secrets.CARGO_REGISTRY_TOKEN" not in RELEASE.read_text(encoding="utf-8")
        assert "id-token: write" in crates
        # The token reaches the publishing step alone.
        verify_step = crates.split("release.py crates verify", 1)[0].rsplit("- ", 1)[1]
        assert "CARGO_REGISTRY_TOKEN" not in verify_step
        # Where cargo looks a crate up, and what it finds there.
        names = ("medh5-sys", "Serde", "abc", "ab", "a")
        assert [release.index_path(n) for n in names] == [
            "me/dh/medh5-sys", "se/rd/serde", "3/a/abc", "2/ab", "1/a"
        ]  # fmt: skip
        listing = b"".join(
            b'{"name":"medh5-sys","vers":"%s"}\n' % v for v in (b"1.9.0", b"2.0.0")
        )
        monkeypatch.setattr(
            release,
            "fetch",
            lambda url, **_: listing if url.endswith("/me/dh/medh5-sys") else None,
        )
        assert release.in_index("medh5-sys", "2.0.0")
        assert not release.in_index("medh5-sys", "2.0.1")
        assert not release.in_index("medh5", "2.0.0")

    def test_every_artifact_is_attested_before_the_release(self):
        """The binaries' archives on the release page had a checksum file and
        nothing that said who built them.  One job attests every archive,
        wheel and sdist, the only one allowed to, and the release waits for
        it."""
        jobs = _jobs(RELEASE.read_text(encoding="utf-8"))
        attest = jobs["attest"]
        assert re.search(r"^    needs: build$", attest, re.M)
        assert re.search(r"uses: actions/attest@[0-9a-f]{40} # v4\.", attest)
        for pattern in ("cli/*.tar.gz", "cli/*.zip", "dist/*.whl", "dist/*.tar.gz"):
            assert f"            {pattern}\n" in attest, pattern
        writers = {n for n, j in jobs.items() if "attestations: write" in j}
        assert writers == {"attest"}
        needs = re.search(r"^    needs: \[(.*)\]$", jobs["github-release"], re.M)
        assert needs and "attest" in needs[1].split(", ")
        install = (ROOT / "docs/install.md").read_text(encoding="utf-8")
        assert "gh attestation verify" in install
        assert "--repo medh5-project/medh5" in install

    def test_every_publishing_step_can_run_again(self):
        """`cargo publish --workspace` refused a re-run once one crate was up,
        `gh release create` once the release existed, and an unchanged formula
        failed `git commit`; a missing token in this repository only warned."""
        release = RELEASE.read_text(encoding="utf-8")
        jobs = _jobs(release)
        assert "skip-existing: true" in jobs["publish"]
        assert "release.py pypi dist/" in jobs["publish"]
        assert "release.py crates" in jobs["crates"]
        assert "cargo publish --workspace" not in release
        assert "release.py github-release" in jobs["github-release"]
        assert "gh release create" not in release
        assert "git diff --cached --quiet" in jobs["homebrew"]
        assert (
            "release.py verify" in jobs["verify"] and "!cancelled()" in jobs["verify"]
        )
        for job in ("crates", "homebrew"):
            assert (
                "CANONICAL: ${{ github.repository == 'medh5-project/medh5' }}"
                in jobs[job]
            )
            fails = (
                r'"\$CANONICAL" == true \]\]; then\n\s+echo "::error::[^\n]*\n\s+exit 1'
            )
            assert re.search(fails, jobs[job]), job
        contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        assert "### If a release fails partway" in contributing

    def test_the_sdist_carries_what_its_project_tests_read(self):
        """Four of the shipped project tests read files the sdist left out, so
        they failed from the archive; CI now runs them from it."""
        include = re.findall(
            r'\{ path = "([^"]+)", format = "sdist" \}',
            PYPROJECT.read_text(encoding="utf-8"),
        )
        needed = [
            ".github/dependabot.yml",
            ".pre-commit-config.yaml",
            "deny.toml",
            ".github/workflows/ci.yml",
            ".github/workflows/release.yml",
            *(
                p.relative_to(ROOT).as_posix()
                for p in (ROOT / ".github/scripts").iterdir()
                if p.suffix in (".py", ".sh")
            ),
            "mkdocs.yml",
            "hooks/mkdocs_hooks.py",
            "docs/requirements.txt",
            "CHANGELOG.md",
            "SECURITY.md",
            "vex/medh5.openvex.json",
        ]
        for path in needed:
            assert (ROOT / path).is_file(), path
            assert any(fnmatch.fnmatch(path, pattern) for pattern in include), path
        sdist = _jobs(CI.read_text(encoding="utf-8"))["sdist"]
        assert "rm -rf medh5" in sdist and "pytest tests/project" in sdist

    def test_the_sdist_carries_the_tests_data(self):
        """The shipped tests read `tests/data` (the ITK golden files, the NIfTI
        parity corpus); left out, the parity test failed to collect from the
        unpacked sdist."""
        include = re.findall(
            r'\{ path = "([^"]+)", format = "sdist" \}',
            PYPROJECT.read_text(encoding="utf-8"),
        )
        data = [
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "tests" / "data").rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not p.name.startswith(".")
        ]
        assert any("nifti-parity/inputs/" in path for path in data)
        missing = [
            path
            for path in data
            if not any(fnmatch.fnmatch(path, pattern) for pattern in include)
        ]
        assert not missing

    def test_a_damaged_file_is_reported_by_every_binary(self, tmp_path):
        """An HDF5 compiled without NDEBUG aborts on a damaged file; CI damages
        files for every platform's binary, and for a build from a copy of the
        workspace without `.cargo/config.toml` --- a crates.io build --- which
        medh5-sys must stop until HDF5 is rebuilt with the flag."""
        damaged = _release_script("damaged.py")
        data = bytes(range(256)) * 4
        once = damaged.damage(data, random.Random(5))
        assert once != data and len(once) == len(data)
        assert sum(a != b for a, b in zip(data, once, strict=True)) <= 6
        assert damaged.damage(data, random.Random(5)) == once
        # The console script beside this interpreter, whether or not its
        # directory is on PATH.
        binary = shutil.which("medh5", path=sysconfig.get_path("scripts"))
        assert binary, "the package's `medh5` command is not installed"
        (tmp_path / "corpus").mkdir()
        write_sample(tmp_path / "corpus" / "case.medh5")
        assert damaged.main([binary, str(tmp_path / "corpus"), "--runs", "6"]) == 0
        # The shapes of 2.0.2's denial-of-service bugs, as the binary writes
        # them, each reported as an error within bounds; a valid file is not.
        hostile = tmp_path / "robustness"
        subprocess.run(
            [binary, "conformance", "build", "--robustness", str(hostile)], check=True
        )
        assert len(list(hostile.glob("*.medh5"))) == 4
        corpus = [binary, str(tmp_path / "corpus"), "--runs", "1"]
        assert damaged.main([*corpus, "--structural", str(hostile)]) == 0
        # A valid file is not an error.  (What follows the colon is the
        # binary's stderr, which on a free-threaded interpreter holds the
        # warning that the GIL was re-enabled.)
        valid = damaged.structural(binary, tmp_path / "corpus" / "case.medh5")
        assert valid is not None and valid.startswith("exit 0, not 1:")
        jobs = _jobs(CI.read_text(encoding="utf-8"))
        build = _jobs(BUILD.read_text(encoding="utf-8"))["build"]
        msvc = jobs["msvc-from-crates-io"]
        assert "rm -r ws/.cargo" in msvc
        for job in (build, msvc):
            assert "conformance build --robustness" in job
            assert re.search(r"damaged\.py .* --structural ", job)
        assert "cargo clean --release -p hdf5-metno-src" in msvc
        # The job's bash rewrote `/DNDEBUG` into `C:/Program Files/Git/DNDEBUG`.
        assert "CFLAGS_x86_64_pc_windows_msvc: -DNDEBUG" in msvc
