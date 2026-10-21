"""Measure reading through medh5 from one, two and four threads, and one call's cost.

The same amount of work, split evenly over ``--threads`` threads, each with
its own handles (the pattern ``docs/explanation/concurrency.md`` shows):

- ``image`` --- a new ``--patch``\\ ³ window of a ``training`` (Blosc2) CT per
  read, the window path that decompresses outside HDF5's lock;
- ``portable`` --- the same windows of the same CT stored ``portable`` (gzip),
  which HDF5 decompresses under its lock;
- ``labels`` --- a new window of the eight-class annotation, ``dense()``;
- ``open`` --- open a sample, read its grid and image metadata, close it.

Each is timed end to end, from a barrier to the last thread's finish, as the
median of ``--repeats`` runs. A thread count that takes longer than one
thread for the same work is the number to look at: HDF5 runs one call at a
time, so threads are not expected to be faster, but they should not be slower.

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


def run_threads(threads: int, work: Callable[[int, int], None], total: int) -> float:
    """Seconds for ``total`` units of ``work`` spread over ``threads`` threads.

    ``work(index, count)`` does ``count`` units as thread ``index``; it opens
    its own handles. Every thread starts at one barrier, and the clock stops
    when the last one finishes.
    """
    share = [total // threads + (i < total % threads) for i in range(threads)]
    barrier = threading.Barrier(threads + 1)
    errors: list[BaseException] = []

    def body(i: int) -> None:
        barrier.wait()
        try:
            work(i, share[i])
        except BaseException as e:
            errors.append(e)

    pool = [threading.Thread(target=body, args=(i,)) for i in range(threads)]
    for t in pool:
        t.start()
    barrier.wait()
    start = time.perf_counter()
    for t in pool:
        t.join()
    elapsed = time.perf_counter() - start
    if errors:
        raise errors[0]
    return elapsed


def workloads(
    paths: dict[str, list[Path]], args: argparse.Namespace, shape: tuple[int, int, int]
) -> dict[str, Callable[[int, int], None]]:
    patch = args.patch

    def image(codec: str) -> Callable[[int, int], None]:
        def work(i: int, count: int) -> None:
            files = paths[codec]
            with medh5.open(files[i % len(files)]) as s:
                ct = s.images["CT"]
                for roi in windows(1000 + i, count, shape, patch):
                    ct.read(roi)

        return work

    def labels(i: int, count: int) -> None:
        files = paths["training"]
        with medh5.open(files[i % len(files)]) as s:
            organs = s.annotations["organs"]
            for roi in windows(2000 + i, count, shape, patch):
                organs.dense(roi=roi)

    def opening(i: int, count: int) -> None:
        files = paths["training"]
        for k in range(count):
            with medh5.open(files[(i + k) % len(files)]) as s:
                s.grids["g"].shape  # noqa: B018 - the metadata is the work
                s.images["CT"].dtype  # noqa: B018

    return {
        "image": image("training"),
        "portable": image("portable"),
        "labels": labels,
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
    out = Path(args.out or tempfile.mkdtemp(prefix="bench-threads-"))
    out.mkdir(parents=True, exist_ok=True)
    shape = tuple(args.shape)
    paths = build(out, args.files, shape)  # type: ignore[arg-type]
    results: dict[str, Any] = {}
    for name, work in workloads(paths, args, shape).items():  # type: ignore[arg-type]
        total = args.opens if name == "open" else args.reads
        run_threads(1, work, max(1, total // 8))  # warm the page cache
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
        "out": str(out),
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
        "--out", help="where to write the samples (default: a temporary directory)"
    )
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args(argv)
    results = measure(args)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str) + "\n")
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
