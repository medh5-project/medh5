"""Measure reading through medh5 from one, two and four threads, and one call's cost.

One fixed list of work, the same at every thread count, dealt out over
``--threads`` threads, each with its own handles (the pattern
``docs/explanation/concurrency.md`` shows). Unit ``k`` reads file
``k % --files``, so with up to ``--files`` threads each thread keeps to its
own files:

- ``image`` --- a new ``--patch``\\ ³ window of a ``training`` (Blosc2) CT per
  read, the window path that decompresses outside HDF5's lock;
- ``portable`` --- the same windows of the same CT stored ``portable`` (gzip),
  which HDF5 decompresses under its lock;
- ``labels`` --- a new window of the eight-class annotation, ``dense()``;
- ``open`` --- open a sample, read its grid and image metadata, close it.

Each is timed end to end, from the moment the last thread reaches a barrier
to the last thread's finish, as the median of ``--repeats`` runs. A thread
count that takes longer than one thread for the same work is the number to
look at: HDF5 runs one call at a time, so threads are not expected to be
faster, but they should not be slower.

``calls`` is one thread making the smallest calls that reach the engine, a
one-voxel image read and a voxel ``contains()``, which is what the binding's
per-call overhead (taking and releasing the interpreter, the HDF5 lock)
costs when it is not hidden behind a decompression.

Run it on a GIL build and on a free-threaded build (``python3.14t``) to
compare them::

    python docs/examples/bench_threads.py [--files 4] [--reads 512] [--json out.json]

Synthetic measurements on one machine; nothing here promises a speedup.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import medh5
from medh5.bench import synthetic_sample


def build(out: Path, files: int, shape: tuple[int, int, int]) -> dict[str, list[Path]]:
    """``files`` samples per codec, written once and reused."""
    written: dict[str, list[Path]] = {}
    for codec in ("training", "portable"):
        paths = []
        for i in range(files):
            name = f"{codec}-{i}-{'x'.join(map(str, shape))}.medh5"
            if not (out / name).exists():
                synthetic_sample(
                    out,
                    shape=shape,
                    classes=8,
                    codec=codec,
                    index=False,
                    seed=20260815 + i,
                    name=name,
                )
            paths.append(out / name)
        written[codec] = paths
    return written


def windows(
    seed: int, n: int, shape: tuple[int, ...], patch: int
) -> list[tuple[slice, ...]]:
    """``n`` random windows of ``patch`` voxels a side, inside ``shape``."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0, np.maximum(1, np.array(shape) - patch + 1), size=(n, len(shape))
    )
    return [tuple(slice(int(s), int(s) + patch) for s in start) for start in starts]


def run_threads(threads: int, work: Callable[[list[int]], None], total: int) -> float:
    """Seconds for units ``0 … total - 1`` of ``work``, dealt over ``threads``.

    Thread ``i`` calls ``work(units)`` with every unit ``k`` where
    ``k % threads == i``; it opens its own handles. The clock starts as the
    last thread reaches the barrier, before any is released, and stops when
    the last one finishes.
    """
    start: list[float] = []
    barrier = threading.Barrier(
        threads, action=lambda: start.append(time.perf_counter())
    )
    errors: list[BaseException] = []

    def body(i: int) -> None:
        barrier.wait()
        try:
            work(list(range(i, total, threads)))
        except BaseException as e:
            errors.append(e)

    pool = [threading.Thread(target=body, args=(i,)) for i in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    elapsed = time.perf_counter() - start[0]
    if errors:
        raise errors[0]
    return elapsed


def workloads(
    paths: dict[str, list[Path]], args: argparse.Namespace, shape: tuple[int, int, int]
) -> dict[str, Callable[[list[int]], None]]:
    """Each workload's ``work(units)``; unit ``k`` reads file ``k % --files``
    and window ``k`` of one fixed list, whichever thread it falls to."""
    rois = windows(1000, args.reads, shape, args.patch)

    def per_file(
        codec: str, read: Callable[[medh5.Sample, tuple[slice, ...]], object]
    ) -> Callable[[list[int]], None]:
        def work(units: list[int]) -> None:
            files = paths[codec]
            with contextlib.ExitStack() as stack:
                opened: dict[int, medh5.Sample] = {}
                for k in units:
                    f = k % len(files)
                    if f not in opened:
                        opened[f] = stack.enter_context(medh5.open(files[f]))
                    read(opened[f], rois[k])

        return work

    def opening(units: list[int]) -> None:
        files = paths["training"]
        for k in units:
            with medh5.open(files[k % len(files)]) as s:
                s.grids["g"].shape  # noqa: B018 - the metadata is the work
                s.images["CT"].dtype  # noqa: B018

    return {
        "image": per_file("training", lambda s, roi: s.images["CT"].read(roi)),
        "portable": per_file("portable", lambda s, roi: s.images["CT"].read(roi)),
        "labels": per_file(
            "training", lambda s, roi: s.annotations["organs"].dense(roi=roi)
        ),
        "open": opening,
    }


def per_call_us(
    paths: dict[str, list[Path]], calls: int, repeats: int
) -> dict[str, float]:
    """Median microseconds per call, one thread, on an open sample."""
    found: dict[str, list[float]] = {"read_one_voxel": [], "contains": []}
    one = (slice(0, 1),) * 3
    with medh5.open(paths["training"][0]) as s:
        ct, organs = s.images["CT"], s.annotations["organs"]
        for _ in range(repeats):
            t = time.perf_counter()
            for _ in range(calls):
                ct.read(one)
            found["read_one_voxel"].append((time.perf_counter() - t) / calls * 1e6)
            t = time.perf_counter()
            for _ in range(calls):
                organs.contains(1, (0, 0, 0))
            found["contains"].append((time.perf_counter() - t) / calls * 1e6)
    return {name: statistics.median(times) for name, times in found.items()}


def measure(args: argparse.Namespace) -> dict[str, Any]:
    if args.out:
        return measure_in(Path(args.out), args)
    with tempfile.TemporaryDirectory(prefix="bench-threads-") as out:
        return measure_in(Path(out), args)


def measure_in(out: Path, args: argparse.Namespace) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    shape = tuple(args.shape)
    paths = build(out, args.files, shape)  # type: ignore[arg-type]
    results: dict[str, Any] = {}
    for name, work in workloads(paths, args, shape).items():  # type: ignore[arg-type]
        total = args.opens if name == "open" else args.reads
        # warm the page cache, every file at least once
        run_threads(1, work, min(total, max(args.files, total // 8)))
        rows: dict[str, Any] = {}
        for threads in args.threads:
            times = [run_threads(threads, work, total) for _ in range(args.repeats)]
            seconds = statistics.median(times)
            rows[str(threads)] = {
                "seconds": seconds,
                "per_s": total / seconds,
                "runs": times,
            }
        one = rows[str(args.threads[0])]["seconds"]
        for row in rows.values():
            row["vs_first"] = row["seconds"] / one
        results[name] = {"units": total, "threads": rows}
    results["calls_us"] = per_call_us(paths, args.calls, args.repeats)
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    results["conditions"] = {
        "files": args.files,
        "shape": shape,
        "patch": args.patch,
        "reads": args.reads,
        "opens": args.opens,
        "repeats": args.repeats,
        "gil": gil,
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "python": sys.version.split()[0],
        "medh5": medh5.__version__,
        "numpy": np.__version__,
        "out": str(out) if args.out else None,
    }
    return results


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--files", type=int, default=4)
    parser.add_argument("--shape", type=int, nargs=3, default=(192, 256, 256))
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--reads", type=int, default=512, help="windows per workload")
    parser.add_argument(
        "--opens", type=int, default=256, help="opens in the open workload"
    )
    parser.add_argument(
        "--calls", type=int, default=2000, help="calls per timing in calls_us"
    )
    parser.add_argument("--threads", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--out",
        help="where to write the samples, kept for the next run "
        "(default: a temporary directory, removed after)",
    )
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args(argv)
    counts = {
        "--files": [args.files],
        "--shape": args.shape,
        "--patch": [args.patch],
        "--reads": [args.reads],
        "--opens": [args.opens],
        "--calls": [args.calls],
        "--threads": args.threads,
        "--repeats": [args.repeats],
    }
    for flag, values in counts.items():
        if any(v < 1 for v in values):
            parser.error(f"{flag} takes positive integers")
    if args.patch > min(args.shape):
        # A window past the volume is clipped, and would time a smaller one.
        parser.error(f"--patch {args.patch} is larger than --shape {min(args.shape)}")
    results = measure(args)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str) + "\n")
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
