"""Time where a command's time goes: inside HDF5's lock, or outside it.

HDF5 runs one call at a time in a process, and its filters run inside that
call, so a thread can only help with the work done outside an HDF5 call.
This script runs commands of a native ``medh5`` built with phase timings and
reports, for each, the seconds of each phase and the most four threads could
give, by Amdahl's bound: with the phases outside the lock spread over four
threads and everything else as it is now (``now``); with what a chunk
writer's workers would do (``writes``: compressing whole chunks, and the
mapping, sampling and decoding that make them); with what the digests'
workers would do (``digests``: decompressing and hashing, inside the stages
that hash); and with both.

Build the timing CLI first. The ``cfg`` is not a Cargo feature, and no
profile sets it::

    RUSTFLAGS="--cfg medh5_phase_timings" CARGO_TARGET_DIR=target/timings \\
        cargo build --release -p medh5-cli
    python docs/examples/bench_phases.py --medh5 target/timings/release/medh5 \\
        [--plain target/release/medh5] [--json phases.json]

``--plain`` names an ordinary build of the same code, run on the same rows,
to show what the timings themselves cost.

The inputs are written once into ``--out`` (a temporary directory by
default): a 256×512×512 ``int16`` CT, a second CT, a ``float32`` PET and an
eight-label ``uint8`` mask, as ``.nii.gz``, and the sample ``convert
from-nifti`` makes of them, 576 MiB of volumes. Each row runs on a fresh copy
of that sample, ``--repeats`` times, and reports the median:

- ``resample`` of the CT to half its z spacing (512³), nearest without the
  annotation, and linear carrying it;
- ``convert from-nifti`` of the four volumes;
- a no-op amendment, ``dataset split --write-claims`` of a one-sample
  manifest, whose commit hashes every dataset again;
- ``recompress --profile archive``, and ``--profile training`` for the
  throughput of LZ4;
- ``verify`` and ``validate --level integrity``.

The engine's own digest throughput, per algorithm, is timed through
``medh5.integrity.digest_bytes``, with the CPU's SHA extensions named.
Synthetic measurements on one machine; nothing here promises a speedup.
"""

from __future__ import annotations

import argparse
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

#: The phases a timing build reports, in its order, with whether each runs
#: inside an HDF5 call (``crates/medh5/src/timings.rs``).
PHASES = {
    "open": True,
    "metadata": True,
    "copy": True,
    "read": True,
    "write": True,
    "decompress": True,
    "compress": True,
    "window": False,
    "map": False,
    "interpolate": False,
    "carry": False,
    "hash": False,
    "inflate": False,
    "decode": False,
}

#: Phases outside the lock (the filters are inside it).
OUTSIDE = [name for name, locked in PHASES.items() if not locked]
#: What workers would do in a chunk writer (P22-12): compress whole chunks,
#: and the mapping, sampling and decoding that make them. Inflating one gzip
#: stream stays sequential.
WRITES = ["compress", "map", "interpolate", "decode"]
#: What workers would do for the digests (P22-13): decompress and hash, a
#: dataset each, inside the stages that hash.
DIGESTS = ["decompress", "window", "hash"]
DIGEST_STAGES = ["commit.digests", "verify"]

ENV = "MEDH5_PHASE_TIMINGS"


def volumes(shape: tuple[int, int, int], seed: int) -> dict[str, np.ndarray]:
    """A CT-like body with eight organs, a second CT, a PET and the labels."""
    rng = np.random.default_rng(seed)
    z, y, x = (np.linspace(-1, 1, n, dtype=np.float32) for n in shape)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij", sparse=True)
    body = (yy / 0.8) ** 2 + (xx / 0.9) ** 2 < 1.0
    labels = np.zeros(shape, dtype=np.uint8)
    hu = np.where(body, np.float32(40), np.float32(-1000))
    uptake = np.where(body, np.float32(1), np.float32(0))
    centres = rng.uniform(-0.5, 0.5, size=(8, 3)).astype(np.float32)
    radii = rng.uniform(0.12, 0.3, size=(8, 3)).astype(np.float32)
    for k, (c, r) in enumerate(zip(centres, radii, strict=True), start=1):
        inside = ((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + (
            (xx - c[2]) / r[2]
        ) ** 2 < 1.0
        labels[inside] = k
        hu = np.where(inside, np.float32(20 * k - 60), hu)
        uptake = np.where(inside, np.float32(k), uptake)

    def noisy(values: np.ndarray, sigma: float, dtype: type) -> np.ndarray:
        noise = rng.normal(0, sigma, size=shape).astype(np.float32)
        return (values + noise).astype(dtype)

    return {
        "ct": noisy(hu, 20, np.int16),
        "ct2": noisy(hu + 5, 20, np.int16),
        "pet": noisy(uptake, 0.2, np.float32),
        "mask": labels,
    }


def shape_tag(shape: tuple[int, ...]) -> str:
    """What names the inputs of one shape: each cached file carries it, so a
    run with another ``--shape`` in the same ``--out`` builds its own."""
    return "x".join(map(str, shape))


def write_inputs(out: Path, shape: tuple[int, int, int]) -> dict[str, Path]:
    """The four ``.nii.gz`` inputs, written once and reused.  Each is written
    beside its name and renamed into place, so a run stopped halfway leaves
    no file that looks finished."""
    import nibabel as nib

    paths = {name: out / f"{name}-{shape_tag(shape)}.nii.gz" for name in
             ("ct", "ct2", "pet", "mask")}  # fmt: skip
    if all(p.exists() for p in paths.values()):
        return paths
    affine = np.diag([0.8, 0.8, 1.5, 1.0])
    for name, values in volumes(shape, 20261023).items():
        # NIfTI's first axis is x; the sample's grid is z, y, x.
        partial = paths[name].with_name(f"partial-{paths[name].name}")
        nib.save(nib.Nifti1Image(np.ascontiguousarray(values.T), affine), partial)
        os.replace(partial, paths[name])
    return paths


def run(medh5: Path, argv: list[str], cwd: Path) -> tuple[float, dict[str, Any] | None]:
    """Seconds for ``medh5 argv``, and the timings it reported, if any."""
    report = cwd / "phases.jsonl"
    report.unlink(missing_ok=True)
    env = {**os.environ, ENV: str(report)}
    start = time.perf_counter()
    done = subprocess.run(
        [str(medh5), *argv], cwd=cwd, env=env, capture_output=True, text=True
    )
    wall = time.perf_counter() - start
    if done.returncode != 0:
        raise SystemExit(
            f"medh5 {' '.join(argv)} failed ({done.returncode}):\n{done.stderr}"
        )
    if not report.exists():
        return wall, None
    lines = report.read_text().splitlines()
    return wall, json.loads(lines[-1])


Row = tuple[str, Callable[[Path], list[str]]]


def rows(inputs: dict[str, Path]) -> list[Row]:
    """Each command: its name, and its arguments given a fresh copy."""
    fine = f"{0.5 * 1.5},0.8,0.8"
    resample = ["resample", "--image", "CT", "--spacing", fine, "--grid-id", "fine"]
    images = [
        "--image", f"CT={inputs['ct']}", "--image", f"CT2={inputs['ct2']}",
        "--image", f"PET={inputs['pet']}", "--mask", f"organs={inputs['mask']}",
    ]  # fmt: skip
    return [
        (
            "resample nearest",
            lambda c: [*resample, "--interpolation", "nearest", "--no-annotations",
                       "-o", str(c.with_name("out.medh5")), str(c)],
        ),
        (
            "resample linear, with the annotation",
            lambda c: [*resample, "--interpolation", "linear",
                       "-o", str(c.with_name("out.medh5")), str(c)],
        ),
        ("convert from-nifti", lambda c: ["convert", "from-nifti", *images,
                                          str(c.with_name("out.medh5"))]),
        (
            "amendment (split claim)",
            lambda c: ["dataset", "split", str(c.with_name("manifest.json")),
                       "--write-claims", "--ratios", "train=1.0"],
        ),
        ("recompress --profile archive", lambda c: ["recompress", "--profile",
                                                    "archive", str(c)]),
        ("recompress --profile training", lambda c: ["recompress", "--profile",
                                                     "training", str(c)]),
        ("verify", lambda c: ["verify", str(c)]),
        ("validate --level integrity", lambda c: ["validate", "--level",
                                                  "integrity", str(c)]),
    ]  # fmt: skip


def fresh(sample: Path, work: Path, medh5: Path) -> Path:
    """A copy of the sample in an empty directory, with its manifest."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    copy = work / "sample.medh5"
    shutil.copyfile(sample, copy)
    index = ["dataset", "index", str(work), "-o", str(work / "manifest.json")]
    subprocess.run([str(medh5), *index], check=True, capture_output=True,
                   env={k: v for k, v in os.environ.items() if k != ENV})  # fmt: skip
    return copy


def summarise(walls: list[float], reports: list[dict[str, Any]]) -> dict[str, Any]:
    """The median of each phase over the runs, and the ceilings."""
    wall = statistics.median(walls)
    phases = {
        name: {
            "seconds": statistics.median(r["phases"][name]["seconds"] for r in reports),
            "bytes": statistics.median(r["phases"][name]["bytes"] for r in reports),
            "locked": locked,
        }
        for name, locked in PHASES.items()
    }
    stages: dict[str, dict[str, Any]] = {}
    for name in reports[0]["stages"]:
        runs = [r["stages"][name] for r in reports if name in r["stages"]]
        stages[name] = {
            "seconds": statistics.median(s["seconds"] for s in runs),
            "phases": {
                p: statistics.median(s["phases"][p] for s in runs) for p in PHASES
            },
        }
    locked = sum(p["seconds"] for p in phases.values() if p["locked"])
    outside = sum(phases[name]["seconds"] for name in OUTSIDE)
    writes = sum(phases[name]["seconds"] for name in WRITES)
    digests = sum(
        stages[stage]["phases"][name]
        for stage in DIGEST_STAGES
        if stage in stages
        for name in DIGESTS
    )
    return {
        "wall": wall,
        "locked": locked,
        "outside": outside,
        "other": max(0.0, wall - locked - outside),
        "writes": writes,
        "digests": digests,
        "phases": phases,
        "stages": stages,
        "ceiling": {
            "now": ceiling(wall, outside),
            "writes": ceiling(wall, writes),
            "digests": ceiling(wall, digests),
            "both": ceiling(wall, writes + digests),
        },
    }


def ceiling(wall: float, parallel: float, threads: int = 4) -> float:
    """Amdahl's bound: ``parallel`` seconds of ``wall`` spread over ``threads``."""
    if wall <= 0:
        return 1.0
    parallel = min(parallel, wall)
    return wall / (wall - parallel + parallel / threads)


def throughput(name: str, row: dict[str, Any]) -> float | None:
    """MiB/s of one phase of a summarised row, where it moved bytes."""
    phase = row["phases"][name]
    if phase["bytes"] <= 0 or phase["seconds"] <= 0:
        return None
    return phase["bytes"] / phase["seconds"] / (1 << 20)


def sha_extensions() -> str:
    """Whether this CPU has SHA-256 instructions, as the OS reports it."""
    machine = platform.machine().lower()
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.optional.arm.FEAT_SHA256"],
                                 capture_output=True, text=True).stdout.strip()  # fmt: skip
            return "yes" if out == "1" else "no"
        flags = Path("/proc/cpuinfo").read_text()
        if machine in ("x86_64", "amd64"):
            return "yes" if " sha_ni" in flags else "no"
        if machine in ("aarch64", "arm64"):
            return "yes" if " sha2" in flags else "no"
    except OSError:
        pass
    return "unknown"


def digest_rates(megabytes: int, repeats: int) -> dict[str, float]:
    """MiB/s of the engine's hash for each algorithm, on one thread."""
    from medh5.integrity import digest_bytes

    payload = np.random.default_rng(0).integers(0, 256, megabytes << 20, np.uint8)
    data = payload.tobytes()
    rates = {}
    for algo in ("sha256", "sha512", "blake2b"):
        digest_bytes(data, algo)  # untimed: the pages and the code warm
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            digest_bytes(data, algo)
            times.append(time.perf_counter() - t)
        rates[algo] = megabytes / statistics.median(times)
    return rates


def measure(args: argparse.Namespace) -> dict[str, Any]:
    if args.out:
        return measure_in(Path(args.out), args)
    with tempfile.TemporaryDirectory(prefix="bench-phases-") as out:
        return measure_in(Path(out), args)


def measure_in(out: Path, args: argparse.Namespace) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    shape = tuple(args.shape)
    medh5 = Path(args.medh5).resolve()
    inputs = write_inputs(out, shape)  # type: ignore[arg-type]
    # `convert from-nifti` replaces its output whole, so a sample that
    # exists is finished.
    sample = out / f"sample-{shape_tag(shape)}.medh5"
    if not sample.exists():
        argv = ["convert", "from-nifti"]
        for name in ("ct", "ct2", "pet"):
            argv += ["--image", f"{name.upper()}={inputs[name]}"]
        argv += ["--mask", f"organs={inputs['mask']}", str(sample)]
        _, report = run(medh5, argv, out)
        if report is None:
            raise SystemExit(
                f"{medh5} reported no phase timings: build it with "
                '`RUSTFLAGS="--cfg medh5_phase_timings"`'
            )
    results: dict[str, Any] = {}
    for name, argv in rows(inputs):
        walls, reports, plain = [], [], []
        for _ in range(args.repeats):
            copy = fresh(sample, out / "run", medh5)
            wall, report = run(medh5, argv(copy), copy.parent)
            if report is None:
                raise SystemExit(f"{medh5} reported no phase timings for {name}")
            walls.append(wall)
            reports.append(report)
            if args.plain:
                copy = fresh(sample, out / "run", medh5)
                plain.append(run(Path(args.plain).resolve(), argv(copy), copy.parent)[0])
        row = summarise(walls, reports)
        if plain:
            row["plain_wall"] = statistics.median(plain)
        results[name] = row
    return {
        "conditions": {
            "shape": list(shape),
            "repeats": args.repeats,
            "machine": platform.machine(),
            "system": platform.system(),
            "cpus": os.cpu_count(),
            "sha_extensions": sha_extensions(),
        },
        "rows": results,
        "digest_mib_s": digest_rates(args.digest_mib, args.repeats),
    }


def table(results: dict[str, Any]) -> str:
    """The rows as Markdown tables: seconds per phase, then the ceilings."""
    rows = results["rows"].items()
    shown = [p for p in PHASES if any(
        row["phases"][p]["seconds"] >= 0.005 for _, row in rows)]  # fmt: skip
    head = ["Command", "wall s", *shown, "other"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, row in rows:
        wall = f"{row['wall']:.2f}"
        if "plain_wall" in row:
            wall += f" ({row['plain_wall']:.2f})"
        cells = [name, wall, *(f"{row['phases'][p]['seconds']:.2f}" for p in shown)]
        lines.append("| " + " | ".join([*cells, f"{row['other']:.2f}"]) + " |")
    head = ["Command", "locked", "outside", "4× now", "writes", "4× writes",
            "digests", "4× digests", "4× both"]  # fmt: skip
    lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, row in rows:
        share = f"{row['outside']:.2f} ({row['outside'] / row['wall']:.0%})"
        c = row["ceiling"]
        cells = [name, f"{row['locked']:.2f}", share, f"{c['now']:.2f}",
                 f"{row['writes']:.2f}", f"{c['writes']:.2f}",
                 f"{row['digests']:.2f}", f"{c['digests']:.2f}", f"{c['both']:.2f}"]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| Throughput, one core | MiB/s |", "|---|---|"]
    for name, row in rows:
        for phase in ("decompress", "compress", "hash", "inflate"):
            rate = throughput(phase, row)
            if rate is not None:
                lines.append(f"| {phase} in {name} | {rate:.0f} |")
    for algo, rate in results["digest_mib_s"].items():
        lines.append(f"| `digest_bytes`, {algo} | {rate:.0f} |")
    c = results["conditions"]
    lines.append("")
    lines.append(
        f"{c['system']} {c['machine']}, {c['cpus']} CPUs, SHA extensions: "
        f"{c['sha_extensions']}; median of {c['repeats']}."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--medh5", required=True, help="a native CLI built with phase timings")
    parser.add_argument("--plain", help="an ordinary build, to show the timings' own cost")
    parser.add_argument("--shape", type=int, nargs=3, default=(256, 512, 512))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--digest-mib", type=int, default=256)
    parser.add_argument("--out", help="where to keep the inputs (default: a temporary directory)")
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.digest_mib < 1 or min(args.shape) < 4:
        parser.error("--repeats and --digest-mib must be positive, and --shape at least 4")
    results = measure(args)
    print(table(results))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
