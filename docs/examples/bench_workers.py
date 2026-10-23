"""Measure the batch commands over 32 files in one process and in four.

A cohort of ``--files`` samples (``--shape``, a CT and a two-class
segmentation, two files per subject, split claims stamped), ``--files`` 0.x
files of the same size (two per subject), and a clinical cohort of ``--files``
subjects (a task manifest over them, and a FHIR bulk export of their
laboratory values), all built here once and reused.  Each command then runs:

- ``1`` --- once, over every file, in one process;
- ``shards`` --- as ``--processes`` invocations at once, each over its share
  of the files in input order (its own output directory, manifest or mapping
  where the command writes or reads one): what the command would take with
  that many workers and nothing else to pay, the ceiling;
- ``workers`` --- once, with ``--workers --processes``, for a command that
  takes it.

Each figure is the median of ``--repeats`` wall-clock runs after one untimed
run, so every run reads from the same page cache.  The commands are every
batch command that reads many files or a cohort: the ten that take
``--workers``, and the four that measured too short to gain from it.  Run it
with the native binary or the console script::

    python docs/examples/bench_workers.py [--medh5 target/release/medh5] [--json out.json]

Synthetic measurements on one machine; nothing here promises a speedup.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import medh5

HERE = Path(__file__).resolve().parent
LOINC = "http://loinc.org"


# -- the cohorts -----------------------------------------------------------------------------


def ct(rng: np.random.Generator, shape: tuple[int, int, int]) -> np.ndarray:
    """A smooth body in noise, so it compresses as a CT does rather than as noise."""
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    c = [s / 2 for s in shape]
    r = (
        ((z - c[0]) / shape[0]) ** 2
        + ((y - c[1]) / shape[1]) ** 2
        + ((x - c[2]) / shape[2]) ** 2
    )
    body = np.where(r < 0.2, 40, -1000).astype(np.int16)
    return (body + rng.normal(0, 20, shape)).astype(np.int16)


def organs(shape: tuple[int, int, int], i: int) -> dict[int, np.ndarray]:
    """Two disjoint balls, so a label volume (nnU-Net's) holds them both."""
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    rad = min(shape) // 6
    out = {}
    for k in (1, 2):
        c = (shape[0] // 2, shape[1] * (2 * k - 1) // 4, shape[2] // 2 + (i % 5) * 2)
        out[k] = (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 < rad**2
    return out


def build_samples(out: Path, files: int, shape: tuple[int, int, int]) -> list[Path]:
    from medh5.labels import LabelClass, LabelSet

    out.mkdir(parents=True, exist_ok=True)
    labels = LabelSet(
        "bench-organs",
        version="1.0.0",
        classes=[LabelClass(1, "liver", "Liver"), LabelClass(2, "spleen", "Spleen")],
    )
    paths = []
    for i in range(files):
        path = out / f"s{i:02d}.medh5"
        paths.append(path)
        rng = np.random.default_rng(i)
        with medh5.create(
            path, sample_id=f"s{i:02d}", subject_id=f"P{i // 2:02d}", codec="balanced"
        ) as w:
            w.cohort(dataset_id="bench", site_id=f"site-{i % 3}")
            w.add_timepoint("tp0", label="baseline", days_from_baseline=0)
            w.label_set(labels)
            w.add_grid(
                "g",
                shape=shape,
                spacing=(2.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                timepoint="tp0",
            )
            w.add_image(
                "CT",
                ct(rng, shape),
                grid="g",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
            )
            w.add_segmentation(
                "seg", grid="g", masks=organs(shape, i), annotated_classes="all_given"
            )
            w.deidentification(method="dicom-psi-profile")
    return paths


def build_legacy(out: Path, files: int, shape: tuple[int, int, int]) -> list[Path]:
    import h5py

    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(files):
        path = out / f"l{i:02d}.h5"
        paths.append(path)
        rng = np.random.default_rng(1000 + i)
        with h5py.File(path, "w") as h:
            group = h.create_group("images")
            group.create_dataset(
                "CT", data=ct(rng, shape), chunks=True, compression="gzip"
            )
            group.attrs["shape"] = np.asarray(shape, dtype=np.int64)
            group.attrs["spacing"] = np.asarray([2.0, 1.0, 1.0])
            group.attrs["origin"] = np.zeros(3)
            seg = h.create_group("seg")
            seg.create_dataset(
                "liver", data=organs(shape, i)[1], chunks=True, compression="gzip"
            )
            h.attrs["schema_version"] = "1"
            h.attrs["image_names"] = json.dumps(["CT"])
            h.attrs["seg_names"] = json.dumps(["liver"])
            h.attrs["has_seg"] = True
            h.attrs["has_bbox"] = False
            h.attrs["extra"] = json.dumps(
                {"patient_id": f"P{i // 2:02d}", "study": f"st{i}"}
            )
    return paths


def preflight_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "bench_preflight", HERE / "bench_preflight.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_clinical(out: Path, subjects: int, shards: int) -> dict[str, Any]:
    """``subjects`` one-fragment clinical subjects, a task over all of them and
    one over each shard; imaging-only samples and a FHIR bulk export of
    laboratory values for ``augment-cohort``, a mapping over all and one per
    shard."""
    from medh5.clinical import DAY
    from medh5.task import Slot, SourceRef, Target, TaskManifest

    bp = preflight_module()
    args = argparse.Namespace(
        fragments=1, events=600, documents=20, note_bytes=1024, visits=3, cutoffs=6
    )
    out.mkdir(parents=True, exist_ok=True)

    def task_over(ids: list[int]) -> TaskManifest:
        task = TaskManifest.new(
            "bench-workers",
            "1",
            identity_namespace="bench",
            slots=[
                Slot("ct", "CT", required=True, patch=(8, 16, 16)),
                Slot("mr", "MR", patch=(8, 16, 16)),
            ],
            target=Target(
                "progression",
                "1",
                bp.RECIST,
                "overall_response",
                positive=("PD",),
                negative=("SD", "PR", "CR"),
                horizon_us=180 * DAY,
                min_follow_up_us=60 * DAY,
            ),
            policy={"context_us": 365 * DAY},
            split=("fold-0", ["train", "val"]),
            base=out,
        )
        for i in ids:
            sid = f"S{i:04d}"
            name = f"{sid}-site0.medh5"
            task.add_subject(
                sid,
                [SourceRef.pin(out / name, uri=name, source_id=f"{sid}-site0")],
                partition="train" if i % 5 else "val",
            )
            for k in range(args.cutoffs):
                task.add_row(f"{sid}@m{k}", sid, k * 30 * DAY + 3 * DAY)
        task.reconcile()
        return task

    for i in range(subjects):
        sid = f"S{i:04d}"
        rng = np.random.default_rng(i)
        (history,) = bp.subject_history(rng, args, 1)
        bp.write_fragment(out / f"{sid}-site0.medh5", sid, 0, args.visits, history, rng)
    ids = list(range(subjects))
    tasks = [task_over(ids).save(out / "task.json")]
    for s, part in enumerate(split(ids, shards)):
        tasks.append(task_over(part).save(out / f"task-{s}.json"))

    fhir = out / "fhir"
    fhir.mkdir(exist_ok=True)
    targets = out / "targets"
    targets.mkdir(exist_ok=True)
    with (fhir / "Observation.ndjson").open("w", encoding="utf-8") as lines:
        for i in ids:
            for j in range(400):
                lines.write(
                    json.dumps(
                        {
                            "resourceType": "Observation",
                            "id": f"o{i}-{j}",
                            "status": "final",
                            "subject": {"reference": f"Patient/p{i}"},
                            "code": {
                                "coding": [
                                    {
                                        "system": LOINC,
                                        "code": bp.LOINC[j % len(bp.LOINC)],
                                    }
                                ]
                            },
                            "effectiveDateTime": f"2020-{1 + j % 12:02d}-{1 + j % 28:02d}T08:15:00Z",
                            "valueQuantity": {"value": float(j % 97), "unit": "mg/dL"},
                        }
                    )
                    + "\n"
                )
    for i in ids:
        with medh5.create(
            targets / f"p{i}.medh5", sample_id=f"p{i}", subject_id=f"p{i}"
        ) as w:
            w.add_timepoint("tp0", label="baseline", days_from_baseline=0)
            w.add_grid(
                "g",
                shape=(4, 8, 8),
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                timepoint="tp0",
            )
            w.add_image(
                "CT",
                np.zeros((4, 8, 8), np.int16),
                grid="g",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
            )
    mappings = []
    for s, part in enumerate([ids, *split(ids, shards)]):
        path = out / ("mapping.csv" if s == 0 else f"mapping-{s - 1}.csv")
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "sample"])
            writer.writerows([f"Patient/p{i}", f"targets/p{i}.medh5"] for i in part)
        mappings.append(path)
    return {"tasks": tasks, "fhir": fhir, "mappings": mappings}


def split(items: list[Any], n: int) -> list[list[Any]]:
    """``items`` in ``n`` runs, in order, as even as they go."""
    k, extra = divmod(len(items), n)
    out, start = [], 0
    for s in range(n):
        end = start + k + (1 if s < extra else 0)
        out.append(items[start:end])
        start = end
    return out


def build(
    out: Path,
    files: int,
    shape: tuple[int, int, int],
    shards: int,
    medh5_cmd: list[str],
) -> dict[str, Any]:
    params = {
        "files": files,
        "shape": list(shape),
        "shards": shards,
        "version": medh5.__version__,
    }
    marker = out / "cohort.json"
    cohort = {
        "samples": sorted((out / "samples").glob("s*.medh5")),
        "legacy": sorted((out / "legacy").glob("l*.h5")),
    }
    if not (marker.exists() and json.loads(marker.read_text()) == params):
        if out.exists():
            shutil.rmtree(out)
        t = time.perf_counter()
        cohort["samples"] = build_samples(out / "samples", files, shape)
        cohort["legacy"] = build_legacy(out / "legacy", files, shape)
        build_clinical(out / "clinical", files, shards)
        run(
            medh5_cmd
            + [
                "dataset",
                "index",
                str(out / "samples"),
                "--out",
                str(out / "manifest.json"),
            ]
        )
        run(
            medh5_cmd
            + [
                "dataset",
                "split",
                str(out / "manifest.json"),
                "--group-by",
                "subject_id",
                "--set-id",
                "main",
                "--write-claims",
            ]
        )
        # The claims changed every file: index again, so the manifest is fresh.
        run(
            medh5_cmd
            + [
                "dataset",
                "index",
                str(out / "samples"),
                "--out",
                str(out / "manifest.json"),
            ]
        )
        for s, part in enumerate(split(cohort["samples"], shards)):
            shard = out / f"shard-{s}"
            shard.mkdir()
            for p in part:
                os.link(p, shard / p.name)
            run(
                medh5_cmd
                + [
                    "dataset",
                    "index",
                    str(shard),
                    "--out",
                    str(out / f"manifest-{s}.json"),
                ]
            )
        marker.write_text(json.dumps(params))
        print(
            f"built the cohorts in {time.perf_counter() - t:.1f} s under {out}",
            file=sys.stderr,
        )
    clinical = out / "clinical"
    cohort["tasks"] = [clinical / "task.json"] + [
        clinical / f"task-{s}.json" for s in range(shards)
    ]
    cohort["mappings"] = [clinical / "mapping.csv"] + [
        clinical / f"mapping-{s}.csv" for s in range(shards)
    ]
    cohort["fhir"] = clinical / "fhir"
    cohort["manifests"] = [out / "manifest.json"] + [
        out / f"manifest-{s}.json" for s in range(shards)
    ]
    return cohort


def run(argv: list[str], ok: tuple[int, ...] = (0,)) -> None:
    found = subprocess.run(argv, capture_output=True, text=True)
    if found.returncode not in ok:
        raise SystemExit(
            f"{' '.join(argv)} exited {found.returncode}:\n{found.stderr[-2000:]}"
        )


# -- the commands ----------------------------------------------------------------------------

# A command's invocation over part `part` of the cohort (None: all of it, in
# one process), with a scratch directory to write into.
Invocation = Callable[[dict[str, Any], int | None, Path], list[str]]


def files(c: dict[str, Any], part: int | None, key: str, shards: int) -> list[str]:
    paths = [str(p) for p in c[key]]
    return paths if part is None else split(paths, shards)[part]


def commands(shards: int) -> dict[str, tuple[Invocation, bool]]:
    """Each command: its invocation, and whether it takes ``--workers``."""

    def f(key: str) -> Callable[[dict[str, Any], int | None], list[str]]:
        return lambda c, part: files(c, part, key, shards)

    def nth(key: str) -> Callable[[dict[str, Any], int | None], str]:
        return lambda c, part: str(c[key][0 if part is None else part + 1])

    samples, legacy = f("samples"), f("legacy")
    manifest, task, mapping = nth("manifests"), nth("tasks"), nth("mappings")
    every_case = [
        line.split()[0]
        for line in subprocess.run(
            [*MEDH5, "conformance", "list"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[2:]
        if line.strip()
    ]

    def cases(part: int | None) -> list[str]:
        chosen = every_case if part is None else split(every_case, shards)[part]
        return [a for name in chosen for a in ("--case", name)]

    return {
        "verify": (lambda c, p, d: ["verify", *samples(c, p)], True),
        "validate": (lambda c, p, d: ["validate", *samples(c, p)], True),
        "index build": (
            lambda c, p, d: ["index", "build", *copies(samples(c, p), d)],
            True,
        ),
        "recompress --profile archive": (
            lambda c, p, d: [
                "recompress",
                "--profile",
                "archive",
                *copies(samples(c, p), d),
            ],
            True,
        ),
        "scrub": (lambda c, p, d: ["scrub", *samples(c, p)], False),
        "splits": (lambda c, p, d: ["splits", *samples(c, p)], False),
        "labels check": (lambda c, p, d: ["labels", "check", *samples(c, p)], False),
        "migrate": (
            lambda c, p, d: [
                "migrate",
                "--out",
                str(d / "migrated"),
                "--group-by",
                "subject",
                "--subject-key",
                "extra.patient_id",
                *legacy(c, p),
            ],
            True,
        ),
        "convert to-nnunet": (
            lambda c, p, d: [
                "convert",
                "to-nnunet",
                str(d / "nnunet"),
                "--annotation",
                "seg",
                "--file-ending",
                ".nii",
                *samples(c, p),
            ],
            True,
        ),
        # A shard's manifest is not the one its files' split claims name (C201).
        "dataset check --deep": (
            lambda c, p, d: ["dataset", "check", "--deep", manifest(c, p)],
            False,
        ),
        "clinical augment-cohort": (
            lambda c, p, d: [
                "clinical",
                "augment-cohort",
                "--from-fhir-bulk",
                str(c["fhir"]),
                "--mapping",
                mapping(c, p),
                "--out-dir",
                str(d / "augmented"),
                "--clock-id",
                "ehr",
                "--clock",
                "utc",
            ],
            True,
        ),
        "task preflight": (lambda c, p, d: ["task", "preflight", task(c, p)], True),
        "conformance run": (
            lambda c, p, d: ["conformance", "run", str(d / "corpus"), *cases(p)],
            True,
        ),
    }


def copies(paths: list[str], scratch: Path) -> list[str]:
    """Fresh copies of ``paths`` for a command that rewrites what it is given."""
    out = scratch / "copies"
    out.mkdir(exist_ok=True)
    made = []
    for p in paths:
        target = out / Path(p).name
        shutil.copyfile(p, target)
        made.append(str(target))
    return made


MEDH5: list[str] = []


def timed(
    c: dict[str, Any], invocation: Invocation, parts: list[int | None], extra: list[str]
) -> float:
    """Wall-clock seconds for ``parts`` run at once, each in a fresh scratch
    directory prepared before the clock starts."""
    with tempfile.TemporaryDirectory() as tmp:
        argvs = []
        for k, part in enumerate(parts):
            scratch = Path(tmp) / str(k)
            scratch.mkdir()
            argvs.append(MEDH5 + invocation(c, part, scratch) + extra)
        t = time.perf_counter()
        procs = [
            subprocess.Popen(a, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for a in argvs
        ]
        for proc, argv in zip(procs, argvs, strict=True):
            out, err = proc.communicate()
            if proc.returncode != 0 and not (
                proc.returncode == 1 and "C201" in out.decode()
            ):
                raise SystemExit(
                    f"{' '.join(argv)} exited {proc.returncode}:\n{err.decode()[-2000:]}"
                )
        return time.perf_counter() - t


def measure(
    c: dict[str, Any],
    name: str,
    invocation: Invocation,
    has_workers: bool,
    shards: int,
    repeats: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {"command": name}
    variants: dict[str, tuple[list[int | None], list[str]]] = {
        "1": ([None], []),
        "shards": (list(range(shards)), []),
    }
    if has_workers:
        variants["workers"] = ([None], ["--workers", str(shards)])
    for label, (parts, extra) in variants.items():
        timed(c, invocation, parts, extra)
        rows[label] = statistics.median(
            timed(c, invocation, parts, extra) for _ in range(repeats)
        )
    rows["ceiling"] = rows["1"] / rows["shards"]
    if has_workers:
        rows["speedup"] = rows["1"] / rows["workers"]
    return rows


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--medh5", default=shutil.which("medh5"), help="the command line to time"
    )
    p.add_argument("--files", type=int, default=32)
    p.add_argument("--shape", type=int, nargs=3, default=(96, 192, 192))
    p.add_argument("--processes", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--only", action="append", help="time only this command; repeatable")
    p.add_argument(
        "--out", type=Path, default=Path(tempfile.gettempdir()) / "medh5-bench-workers"
    )
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    MEDH5[:] = [args.medh5]
    cohort = build(args.out, args.files, tuple(args.shape), args.processes, MEDH5)
    results = []
    print(f"{'command':32} {'1':>8} {'shards':>8} {'workers':>8} {'ceiling':>8}")
    for name, (invocation, has_workers) in commands(args.processes).items():
        if args.only and name not in args.only:
            continue
        row = measure(
            cohort, name, invocation, has_workers, args.processes, args.repeats
        )
        results.append(row)
        workers = f"{row['workers']:8.2f}" if "workers" in row else f"{'-':>8}"
        print(
            f"{name:32} {row['1']:8.2f} {row['shards']:8.2f} {workers} {row['ceiling']:7.2f}x",
            flush=True,
        )
    if args.json:
        doc = {
            "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs",
            "medh5": medh5.__version__,
            "command_line": args.medh5,
            "files": args.files,
            "shape": list(args.shape),
            "processes": args.processes,
            "rows": results,
        }
        args.json.write_text(json.dumps(doc, indent=2) + "\n")
    return results


if __name__ == "__main__":
    main()
