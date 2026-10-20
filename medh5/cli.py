"""The ``medh5`` command line.

One native application over the format engine (the ``medh5-cli`` crate): the
same grammar, output and exit codes --- 0 success, 1 a handled error, 2 a usage
error --- whether it runs as the standalone ``medh5`` binary or as this
package's console script.  NIfTI is converted by the engine itself.  The
commands only Python can run are handed back to the package: the other format
converters (DICOM, DICOM SEG, RTSTRUCT, ITK, nnU-Net), which wrap pydicom and
highdicom or write NIfTI through the engine, and the PyTorch dataloader
benchmark.  The standalone binary reaches them by running this module's
``main`` in a Python interpreter that has the package (``python3`` --- ``python``
on Windows --- or the one ``MEDH5_PYTHON`` names), with the current directory
taken off ``sys.path`` first.

Every converter writes a conversion report, because the interesting part of an
import is not that it succeeded but what it had to decide: which encoding, which
class ids, whether a half-voxel convention changed, whether a timepoint order
was read or guessed.  ``--report FILE`` keeps it; without it the guesses and
warnings still print.
"""

from __future__ import annotations

import contextlib
import json
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from medh5 import _core
from medh5.errors import MEDH5Error, MEDH5ValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.io.report import ConversionReport

EXIT_OK: int = _core.EXIT_OK
EXIT_ERROR: int = _core.EXIT_ERROR
EXIT_USAGE: int = _core.EXIT_USAGE


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``medh5`` on *argv* (``sys.argv[1:]`` by default); the exit code.

    ``--help`` and ``--version`` return 0 and a usage error 2, like every other
    outcome: nothing raises ``SystemExit`` but the console script itself.

    Without *argv* --- as the console script, ``python -m medh5.cli`` and the
    standalone binary's hand-off run it --- Ctrl-C ends the process at once, as
    it ends the standalone binary.  The command runs in native code, and
    Python's handler only marks the interrupt for the interpreter to raise:
    a long command ran on to the end, then printed a ``KeyboardInterrupt``.
    """
    args = [str(a) for a in (sys.argv[1:] if argv is None else argv)]
    if argv is not None:
        return int(_core.cli_main(args, _Host()))
    try:
        previous = signal.signal(signal.SIGINT, signal.SIG_DFL)
    except ValueError:  # pragma: no cover - not the main thread: no handler
        previous = None
    try:
        return int(_core.cli_main(args, _Host()))
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def command_tree() -> dict[str, Any]:
    """The grammar as data: ``{"options", "positionals", "commands"}``,
    recursively --- what documentation is checked against."""
    found: dict[str, Any] = _core.cli_command_tree()
    return found


class _Host:
    """What only the Python package can run, on behalf of the native CLI."""

    def convert(self, command: str, args: dict[str, Any]) -> int:
        """Run converter *command* on the arguments the command line parsed."""
        return _convert(command, args)

    def throughput(
        self, path: str, patch: int, workers: int, annotation: str | None
    ) -> dict[str, Any]:
        """Sustained patches/s through the real dataloader, as a measurement
        record.  Raises ``ImportError`` without PyTorch; the command line
        reports that as a skipped measurement rather than a failure."""
        from medh5.bench import throughput

        found: dict[str, Any] = throughput(
            [path], patch=patch, workers=workers, annotation=annotation
        ).to_json()
        return found


def _what(exc: LookupError) -> str:
    """The message a lookup failed with, or what was looked up.

    ``str(KeyError('x'))`` is ``"'x'"``: the key alone, in quotes.  Most of
    this package's lookups raise with a sentence that names what is available,
    which is printed as it is; a bare key is named as one.  The rule is the
    native CLI's, so a converter's error reads as an engine error does.
    """
    detail = exc.args[0] if len(exc.args) == 1 else exc
    return str(_core.cli_lookup_message(str(detail)))


def _fail(message: str) -> int:
    """Report a handled error on stderr: ``medh5: <message>``, exit code 1."""
    print(f"medh5: {message}", file=sys.stderr)
    return EXIT_ERROR


# -- the converters ``medh5 convert`` hands back to the package -------------------


def _convert(command: str, args: Mapping[str, Any]) -> int:
    """Run converter *command* (``from-dicom``, ``to-dicom-seg``, ...) on its
    parsed arguments, keyed as the 1.x parser named them; the exit code.

    A handled failure --- a format error (an input that is not the format
    named is refused as one by the readers), a missing optional dependency, a
    file that cannot be read or written, a name the file does not have ---
    prints ``medh5: <why>`` and exits 1, as every native command does.  A
    ``--report`` that names a directory is refused before anything is
    converted, not after the output is written, and so is a report or an
    output that would replace a file the command reads or writes
    (:func:`_collision`).
    """
    handler = _HANDLERS.get(command)
    if handler is None:
        return _fail("usage: medh5 convert COMMAND ... (see --help)")
    report = args.get("report")
    if report and Path(report).is_dir():
        return _fail(f"--report {report} is a directory; name the file to write")
    clash = _collision(command, args)
    if clash is not None:
        return _fail(clash)
    try:
        return handler(SimpleNamespace(**args))
    except BrokenPipeError:  # pragma: no cover - `medh5 convert ... | head`
        return EXIT_OK
    except (MEDH5Error, ImportError) as exc:
        return _fail(str(exc))
    except OSError as exc:
        return _fail(_os_error(exc))
    except LookupError as exc:
        return _fail(_what(exc))


def _collision(command: str, args: Mapping[str, Any]) -> str | None:
    """Why *command*'s report or output would replace a file it reads or
    writes, or ``None``.

    Each was written over whatever it named, with exit status 0: a
    ``--report`` naming the sample ``from-dicom-seg`` amends replaced the only
    copy of it with JSON, and an output naming an input replaced the input.
    Paths are compared resolved and, where both exist, as one file, so
    ``sub/../ct.nii.gz``, a link to it and a hard link are the file itself.  A
    report is not written inside a tree the command reads or the nnU-Net
    dataset it writes, whose files are the dataset's; no output is written
    inside a clinical dataset or export the command reads, whose files it
    reads while it writes.
    """
    reads, writes, trees, sealed = _files(command, args)
    report = args.get("report")
    if report:
        for what, path in (*reads, *writes):
            if _same_file(Path(report), path):
                return f"--report {report} is {what}; name another file for the report"
        for what, path in trees:
            if _inside(Path(report), path):
                return (
                    f"--report {report} is inside {what} ({path}); write the "
                    "report outside it"
                )
    for _, out in writes:
        for what, path in reads:
            if _same_file(out, path):
                return f"the output {out} is {what}; name another file to write"
        for what, path in sealed:
            if _inside(out, path):
                return (
                    f"the output {out} is inside {what} ({path}); write it outside it"
                )
    return None


def _files(
    command: str, args: Mapping[str, Any]
) -> tuple[
    list[tuple[str, Path]],
    list[tuple[str, Path]],
    list[tuple[str, Path]],
    list[tuple[str, Path]],
]:
    """What a converter command reads, writes, reads as a whole tree, and
    reads as a tree no output may be written inside, as ``(what, path)``
    pairs --- named from its arguments, before it runs."""
    reads: list[tuple[str, Path]] = []
    writes: list[tuple[str, Path]] = []
    trees: list[tuple[str, Path]] = []
    sealed: list[tuple[str, Path]] = []

    def given(key: str) -> Path | None:
        value = args.get(key)
        return Path(value) if value else None

    def read(what: str, key: str) -> None:
        if (path := given(key)) is not None:
            reads.append((what, path))

    out = given("out")
    if command in ("to-dicom-seg", "to-rtstruct"):
        read("the sample this command reads", "path")
        for source in args.get("source") or []:
            reads.append(("a source image this command reads", Path(source)))
    elif command in ("from-dicom-seg", "from-rtstruct"):
        # The sample is amended in place: an input the report must not name,
        # and no output to compare with it.
        read("the SEG this command imports", "seg")
        read("the RTSTRUCT this command imports", "rtstruct")
        read("the sample this command amends", "sample")
    elif command == "from-itk-transform":
        read("the transform this command imports", "transform")
        read("the warp this command imports", "warp")
        read("the inverse warp this command imports", "inverse_warp")
        read("the sample this command amends", "sample")
    elif command == "to-itk-transform":
        read("the sample this command reads", "sample")
        if (warp_out := given("warp_out")) is not None:
            writes.append(("the warp this command writes", warp_out))
    elif command == "from-dicom" and (root := given("root")) is not None:
        trees.append(("the DICOM tree this command reads", root))
        if out is not None and _inside(out, root) and _is_dicom(out):
            reads.append(("a DICOM file of the tree this command reads", out))
    elif command == "from-nnunet" and (root := given("root")) is not None:
        trees.append(("the nnU-Net dataset this command reads", root))
        if out is not None:
            cases = _nnunet_cases(
                root, args.get("case_ids"), include_test=bool(args.get("include_test"))
            )
            for case in cases:
                writes.append(
                    (
                        f"case {case!r}'s sample, which this command writes",
                        out / f"{case}.medh5",
                    )
                )
    elif command == "to-nnunet":
        for path in args.get("paths") or []:
            reads.append(("a sample this command exports", Path(path)))
        if out is not None and args.get("dataset_name"):
            trees.append(
                (
                    "the nnU-Net dataset this command writes",
                    out / str(args["dataset_name"]),
                )
            )
    elif command == "clinical-augment":
        read("the sample this command amends", "path")
        read("the FHIR Bundle this command imports", "from_fhir")
        if (root := given("from_meds")) is not None:
            trees.append(("the MEDS dataset this command reads", root))
            sealed.append(("the MEDS dataset this command reads", root))
    elif command == "clinical-augment-cohort":
        read("the mapping this command reads", "mapping")
        # Each sample the mapping names is amended in place, or read and
        # written under its name in --out-dir; a mapping that does not
        # parse is refused by the command itself.
        mapping = given("mapping")
        out_dir = given("out_dir")
        with contextlib.suppress(OSError, ValueError, MEDH5ValidationError):
            cohort = (
                {}
                if mapping is None
                else _cohort_mapping(mapping, meds=bool(args.get("from_meds")))
            )
            if out_dir is not None:
                writes.append(
                    ("the directory this command writes the samples into", out_dir)
                )
            for subject, sample in cohort.items():
                whose = f"subject {subject}'s sample, which this command"
                if out_dir is None:
                    reads.append((f"{whose} amends", sample))
                else:
                    reads.append((f"{whose} reads", sample))
                    writes.append((f"{whose} writes", out_dir / sample.name))
        if (root := given("from_meds")) is not None:
            trees.append(("the MEDS dataset this command reads", root))
            sealed.append(("the MEDS dataset this command reads", root))
        if (root := given("from_fhir_bulk")) is not None:
            trees.append(("the FHIR bulk export this command reads", root))
            sealed.append(("the FHIR bulk export this command reads", root))
    if out is not None:
        writes.append(("the output this command writes", out))
    return reads, writes, trees, sealed


def _same_file(first: Path, second: Path) -> bool:
    """Whether two paths name one file: equal once resolved, one file on disk
    by the engine's identity (a hard link, or a name the resolution cannot
    see through), or two outputs not written yet whose names differ only in
    case or normalisation, which a file system blind to them (macOS's,
    Windows') would make one file."""
    from medh5 import _core
    from medh5.io._clinical import file_identity

    with contextlib.suppress(OSError, RuntimeError):  # RuntimeError: a link loop
        if first.resolve() == second.resolve():
            return True
    if _core.same_file(first, second):
        return True
    return bool(file_identity(first) == file_identity(second))


def _inside(path: Path, tree: Path) -> bool:
    """Whether *path* lies inside directory *tree*, once both are resolved."""
    try:
        return path.resolve().is_relative_to(tree.resolve())
    except (OSError, RuntimeError):
        return False


def _is_dicom(path: Path) -> bool:
    """Whether *path* is a file the DICOM importer reads: ``DICM`` at 128."""
    try:
        with open(path, "rb") as handle:
            return handle.read(132)[128:] == b"DICM"
    except OSError:
        return False


def _nnunet_cases(
    root: Path, case_ids: Sequence[str] | None, *, include_test: bool = False
) -> list[str]:
    """The cases ``from-nnunet`` would write, or none where the dataset is
    one the converter refuses itself."""
    if case_ids:
        return list(case_ids)
    from medh5.io.nnunetv2 import cases, read_dataset_json

    try:
        return cases(root, read_dataset_json(root), include_test=include_test)
    except (MEDH5Error, OSError, ValueError, KeyError, TypeError):
        return []


def _os_error(exc: OSError) -> str:
    """``<file>: <why>`` for an error that names its file, as the shell
    says it; the message as it is otherwise."""
    if exc.filename is not None and exc.strerror:
        return f"{exc.filename}: {exc.strerror}"
    return str(exc)


def _pairs(values: list[str] | None, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in values or []:
        if "=" not in entry:
            raise MEDH5Error(f"--{what} expects NAME=VALUE, got {entry!r}")
        name, _, value = entry.partition("=")
        out[name] = value
    return out


def _numbered(values: list[str] | None, what: str) -> dict[int, str] | None:
    """``NUMBER=VALUE`` pairs, keyed by the number; ``None`` for none."""
    out: dict[int, str] = {}
    for name, value in _pairs(values, what).items():
        try:
            out[int(name)] = value
        except ValueError:
            raise MEDH5Error(
                f"--{what} expects NUMBER=VALUE, got {name}={value}"
            ) from None
    return out or None


def _sample_file(path: str) -> str:
    """*path*, as the sample file a converter reads or imports into.

    A collection is refused with the way to its members, as every native
    command that works on samples refuses one: the engine's refusal names the
    Python API's (``open it with open_collection()``), which tells a command
    line user nothing they can type.
    """
    refusal = _core.cli_sample_file_refusal(path)
    if refusal is not None:
        raise MEDH5Error(refusal)
    return path


def _annotated(value: str | None) -> Sequence[str] | str:
    """``--annotated-classes``: ``all_given`` and ``all`` as they are, else
    the comma-separated class keys as a list."""
    if value is None or value in ("all_given", "all"):
        return value or "all_given"
    keys = [key.strip() for key in value.split(",")]
    if not all(keys):
        raise MEDH5Error(
            "--annotated-classes expects all_given, all, or class keys separated "
            f"by commas, got {value!r}"
        )
    return keys


def _finish(report: ConversionReport, args: Any, text: str | None = None) -> int:
    """Keep the report where ``--report`` says, print it, and exit on it.

    An output the arguments could not name in advance --- a sample of a
    ``from-dicom`` directory --- is checked here: a report naming one is not
    written, and the command fails rather than replace the sample.  *text*,
    when given, is printed in place of the report's text (``--json`` still
    prints the report).
    """
    target = getattr(args, "report", None)
    refusal = None
    if target:
        clash = next(
            (o for o in report.outputs if _same_file(Path(target), Path(o))), None
        )
        if clash is None:
            _write_report(Path(target), json.dumps(report.to_json(), indent=2) + "\n")
        else:
            refusal = (
                f"--report {target} is {clash}, which this command wrote; the "
                "report was not written --- name another file for it"
            )
    if getattr(args, "json", False):
        print(json.dumps(report.to_json(), indent=2, default=str))
    else:
        print(report.format(verbose=True) if text is None else text)
    if refusal is not None:
        return _fail(refusal)
    return EXIT_OK if report.ok else EXIT_ERROR


def _write_report(path: Path, text: str) -> None:
    """*text* as *path*, through the engine's replace (§14.4): a link at
    *path* is kept and the file it names replaced, and an error names
    *path*, not the temporary beside it."""
    from medh5 import _core

    _core.write_atomic(path, text)


def _from_dicom(args: Any) -> int:
    from medh5.io.dicom import from_dicom

    report = from_dicom(
        args.root,
        args.out,
        group_by=args.group_by,
        modalities=args.modalities,
        series_uids=args.series_uids,
        per_slice_rescale=args.per_slice_rescale,
        codec=args.codec,
    )
    return _finish(report, args)


def _from_dicom_seg(args: Any) -> int:
    from medh5.io.dicom_seg import from_dicom_seg

    report = from_dicom_seg(
        args.seg,
        _sample_file(args.sample),
        ann_id=args.ann_id,
        grid=args.grid,
        frame_salt=args.frame_salt,
        annotated_classes=_annotated(args.annotated_classes),
    )
    return _finish(report, args)


def _to_dicom_seg(args: Any) -> int:
    from medh5.io.dicom_seg import to_dicom_seg
    from medh5.io.report import ConversionReport

    report = ConversionReport(converter="to-dicom-seg")
    # `--series-description`, where given: the default is the converter's.
    described = (
        {"series_description": args.series_description}
        if args.series_description is not None
        else {}
    )
    to_dicom_seg(
        _sample_file(args.path),
        args.annotation,
        args.source,
        args.out,
        classes=args.classes,
        **described,
        # `--frame-salt`, where the grammar gives this command one.
        frame_salt=getattr(args, "frame_salt", None),
        report=report,
        segmentation_type=getattr(args, "segmentation_type", None) or "binary",
        fractional_type=getattr(args, "fractional_type", None) or "PROBABILITY",
    )
    return _finish(report, args)


def _from_rtstruct(args: Any) -> int:
    from medh5.io.rtstruct import from_rtstruct

    report = from_rtstruct(
        args.rtstruct,
        _sample_file(args.sample),
        ann_id=args.ann_id,
        grid=args.grid,
        rasterize=args.rasterize,
        # `--frame-salt`, where the grammar gives this command one.
        frame_salt=getattr(args, "frame_salt", None),
        roi_classes=_numbered(getattr(args, "roi_classes", None), "roi"),
        points_id=getattr(args, "points_id", None),
        open_curves=getattr(args, "open_curves", None) or "omit",
        curves_id=getattr(args, "curves_id", None),
        mask_id=args.mask_id,
        annotated_classes=_annotated(args.annotated_classes),
    )
    return _finish(report, args)


def _to_rtstruct(args: Any) -> int:
    from medh5.io.report import ConversionReport
    from medh5.io.rtstruct import to_rtstruct

    report = ConversionReport(converter="to-rtstruct")
    # `--label`, where given: the default is the converter's.
    labelled = {"label": args.label} if args.label is not None else {}
    to_rtstruct(
        _sample_file(args.path),
        args.annotation,
        args.source,
        args.out,
        **labelled,
        frame_salt=args.frame_salt,
        report=report,
        points=getattr(args, "points", None),
        curves=getattr(args, "curves", None),
    )
    return _finish(report, args)


def _from_itk_transform(args: Any) -> int:
    from medh5.io.itk import from_itk_transform

    report = from_itk_transform(
        args.transform,
        _sample_file(args.sample),
        transform_id=args.transform_id,
        fixed=args.fixed,
        moving=args.moving,
        warp=args.warp,
        vectors=args.vectors,
        inverse_warp=args.inverse_warp,
    )
    return _finish(report, args)


def _to_itk_transform(args: Any) -> int:
    from medh5.io.itk import to_itk_transform

    report = to_itk_transform(
        _sample_file(args.sample),
        args.out,
        transform_id=args.transform_id,
        warp_out=args.warp_out,
        vectors=args.vectors,
    )
    return _finish(report, args)


def _from_nnunet(args: Any) -> int:
    from medh5.io.nnunetv2 import from_nnunetv2

    report = from_nnunetv2(
        args.root,
        args.out,
        case_ids=args.case_ids,
        include_test=args.include_test,
        coord_system=args.coord_system,
        codec=args.codec,
    )
    return _finish(report, args)


def _to_nnunet(args: Any) -> int:
    from medh5.io.nnunetv2 import to_nnunetv2

    report = to_nnunetv2(
        [_sample_file(p) for p in args.paths],
        args.out,
        dataset_name=args.dataset_name,
        file_ending=args.file_ending,
        annotation=args.annotation,
        classes=args.classes,
        unlabeled=args.unlabeled,
        split=args.split,
        replace=args.replace,
    )
    return _finish(report, args)


_MEDS_ONLY = ("code_system", "kind", "day_precision", "document_code", "both_values")
"""``clinical augment`` options only a MEDS import reads."""


def _clinical_augment(args: Any) -> int:
    """``medh5 clinical augment SAMPLE --from-meds DIR | --from-fhir FILE``."""
    if args.from_fhir:
        given = [f"--{o.replace('_', '-')}" for o in _MEDS_ONLY if getattr(args, o)]
        if given:
            return _fail(
                f"{', '.join(given)}: MEDS options, which --from-fhir does not take"
            )
    sample = _sample_file(args.path)
    common = {
        "clock": _import_clock(args, sample),
        "timezone": args.timezone,
        "origin": args.origin,
        "texts": args.with_text,
        "out": args.out,
    }
    if args.from_meds:
        from medh5.io.meds import augment_from_meds

        result, report = augment_from_meds(
            sample,
            args.from_meds,
            subject_id=args.meds_subject,
            code_systems=_pairs(args.code_system, "code-system"),
            kinds=_pairs(args.kind, "kind"),
            day_precision=args.day_precision or (),
            document_codes=args.document_code or (),
            both_values=args.both_values or "refuse",
            **common,
        )
    else:
        from medh5.io.fhir import augment_from_fhir

        result, report = augment_from_fhir(
            sample, args.from_fhir, subject=args.fhir_subject, **common
        )
    if args.report:
        _write_report(Path(args.report), json.dumps(report.to_json(), indent=2) + "\n")
    if args.json:
        print(json.dumps({"augment": result, "report": report.to_json()}, indent=2))
    else:
        print(
            f"{result['path']}: MEDH5 {result['version_before']} -> "
            f"{result['version_after']}; {result['events']} events, "
            f"{result['documents']} documents, {result['links']} links"
        )
        print(report.format(verbose=True))
    return EXIT_OK if report.ok else EXIT_ERROR


def _clinical_augment_cohort(args: Any) -> int:
    """``medh5 clinical augment-cohort --from-meds DIR | --from-fhir-bulk DIR
    --mapping MAP.csv``."""
    if args.from_fhir_bulk:
        given = [f"--{o.replace('_', '-')}" for o in _MEDS_ONLY if getattr(args, o)]
        if given:
            return _fail(
                f"{', '.join(given)}: MEDS options, which --from-fhir-bulk does "
                "not take"
            )
    mapping = _cohort_mapping(Path(args.mapping), meds=bool(args.from_meds))
    common = {
        "out_dir": args.out_dir,
        "clock": _given_clock(args),
        "timezone": args.timezone,
        "origin": args.origin,
        "texts": args.with_text,
    }
    if args.from_meds:
        from medh5.io.meds import augment_cohort_from_meds

        cohort = augment_cohort_from_meds(
            args.from_meds,
            mapping,
            code_systems=_pairs(args.code_system, "code-system"),
            kinds=_pairs(args.kind, "kind"),
            day_precision=args.day_precision or (),
            document_codes=args.document_code or (),
            both_values=args.both_values or "refuse",
            **common,
        )
    else:
        from medh5.io.fhir import augment_cohort_from_fhir

        cohort = augment_cohort_from_fhir(args.from_fhir_bulk, mapping, **common)
    found = cohort.to_json()
    if args.report:
        _write_report(Path(args.report), json.dumps(found, indent=2) + "\n")
    if args.json:
        print(json.dumps(found, indent=2))
    else:
        print(cohort.format(verbose=True))
    return EXIT_OK if cohort.ok else EXIT_ERROR


def _cohort_mapping(path: Path, *, meds: bool) -> dict[Any, Path]:
    """``--mapping``: a CSV whose header is ``subject,sample``, a row per
    subject; a relative sample path is the CSV's."""
    import csv

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows or [c.strip() for c in rows[0]] != ["subject", "sample"]:
        raise MEDH5ValidationError(
            f"{path}: a mapping's first line is the header subject,sample"
        )
    mapping: dict[Any, Path] = {}
    for number, row in enumerate(rows[1:], 2):
        if not any(c.strip() for c in row):
            continue
        if len(row) != 2:
            raise MEDH5ValidationError(
                f"line {number} of {path} has {len(row)} field(s); a mapping "
                "row is subject,sample"
            )
        subject, sample = (c.strip() for c in row)
        key: Any = subject
        if meds:
            try:
                key = int(subject)
            except ValueError:
                raise MEDH5ValidationError(
                    f"line {number} of {path}: {subject!r} is not a MEDS "
                    "subject_id (an integer)"
                ) from None
        if key in mapping:
            raise MEDH5ValidationError(
                f"line {number} of {path} maps subject {subject} again"
            )
        mapping[key] = path.parent / sample
    return mapping


def _given_clock(args: Any) -> Any:
    """The clock ``--clock-id`` and ``--clock`` name, or ``None``."""
    from medh5.clinical import Clock

    if not args.clock_id:
        return None
    if args.clock == "relative":
        origin = args.origin or "an origin not given"
        return Clock.relative(args.clock_id, f"{origin}, as given at import")
    return Clock(args.clock_id, args.clock)


def _import_clock(args: Any, sample: str) -> Any:
    """The clock ``--clock-id`` and ``--clock`` name, else the sample's."""
    from medh5.sample import open_sample

    given = _given_clock(args)
    if given is not None:
        return given
    with open_sample(sample) as found:
        clinical = found.clinical
        if clinical is None:
            raise MEDH5Error(
                f"{sample} has no clinical clock to import onto: give --clock-id "
                "and --clock"
            )
        return clinical.clock


_HANDLERS: dict[str, Callable[[Any], int]] = {
    "clinical-augment": _clinical_augment,
    "clinical-augment-cohort": _clinical_augment_cohort,
    "from-dicom": _from_dicom,
    "from-dicom-seg": _from_dicom_seg,
    "to-dicom-seg": _to_dicom_seg,
    "from-rtstruct": _from_rtstruct,
    "to-rtstruct": _to_rtstruct,
    "from-itk-transform": _from_itk_transform,
    "to-itk-transform": _to_itk_transform,
    "from-nnunet": _from_nnunet,
    "to-nnunet": _to_nnunet,
}
"""Converter subcommand -> its handler."""


__all__ = ["EXIT_ERROR", "EXIT_OK", "EXIT_USAGE", "command_tree", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
