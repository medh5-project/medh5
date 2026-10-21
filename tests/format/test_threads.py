"""Engine calls that leave other Python threads running.

A call whose work grows with its data runs with the GIL released, so a
loader thread, a progress bar or a web server keeps going beside it.  Each
test runs a call while another thread stamps the time every half a
millisecond; a gap as long as the call is the GIL held for all of it.  Work
that walks Python objects, and constant-size work, keeps the GIL.
"""

from __future__ import annotations

import itertools
import os
import re
import select
import signal
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.annotations.geometric import (
    Polygon,
    encode_boxes,
    encode_contours,
    encode_points,
)
from medh5.annotations.voxel import normalize_masks
from medh5.curation.agreement import dice, iou
from medh5.transforms import refuse_outside


def watched(call: Callable[[], Any]) -> tuple[float, float, float]:
    """The call's duration, the longest another thread went unscheduled during
    it, and the longest it went unscheduled before it, with nothing running:
    the ticker's own jitter, which a slow timer (macOS) makes milliseconds."""
    stamps: list[float] = []
    done = threading.Event()

    def tick() -> None:
        while not done.is_set():
            stamps.append(time.perf_counter())
            time.sleep(0.0005)

    ticker = threading.Thread(target=tick)
    ticker.start()
    try:
        time.sleep(0.02)
        t0 = time.perf_counter()
        call()
        t1 = time.perf_counter()
    finally:
        done.set()
        ticker.join()
    before = [t for t in stamps if t < t0]
    inside = [t for t in stamps if t0 < t < t1]
    idle = float(np.diff([*before, t0]).max()) if before else 0.0
    return t1 - t0, float(np.diff([t0, *inside, t1]).max()), idle


def released(call: Callable[[], Any], share: float = 0.25) -> None:
    """Assert no other thread waited *share* of the call or longer, beyond the
    ticker's own jitter, taking the best of three runs: one scheduler hiccup
    is not the GIL.  A call holding the GIL throughout still fails, since the
    calls here run many times longer than that jitter."""
    runs = [watched(call) for _ in range(3)]
    took, longest, idle = min(runs, key=lambda run: (run[1] - run[2]) / run[0])
    assert longest < share * took + idle, (
        f"no tick for {longest:.3f} of {took:.3f} s (idle jitter {idle:.3f} s)"
    )


@pytest.fixture(scope="module")
def masks() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    shape = (160, 256, 256)
    return rng.random(shape) < 0.3, rng.random(shape) < 0.3


class TestMasks:
    def test_normalize_masks_does_not_copy_contiguous_bool(self):
        mask = np.zeros((4, 5, 6), dtype=bool)
        resolved, shape = normalize_masks({1: mask})
        assert shape == (4, 5, 6)
        assert np.shares_memory(resolved[1], mask)

    def test_normalize_masks_converts_what_is_not(self):
        fortran = np.asfortranarray(np.ones((4, 5, 6), dtype=bool))
        resolved, _ = normalize_masks({2: fortran, 1: np.ones((4, 5, 6), np.uint8)})
        assert list(resolved) == [1, 2]
        for found in resolved.values():
            assert found.dtype == np.bool_ and found.flags.c_contiguous
            assert found.all()
        assert not np.shares_memory(resolved[2], fortran)

    def test_normalize_masks_still_checks_shapes(self):
        with pytest.raises(medh5.MEDH5ValidationError) as exc:
            normalize_masks({1: np.zeros((4, 5), bool), 2: np.zeros((4, 6), bool)})
        assert exc.value.code == "E405"
        with pytest.raises(medh5.MEDH5ValidationError) as exc:
            normalize_masks({1: np.zeros((4, 5), bool)}, (4, 6))
        assert exc.value.code == "E405"
        with pytest.raises(medh5.MEDH5ValidationError) as exc:
            normalize_masks({})
        assert exc.value.code == "E410"
        with pytest.raises(medh5.MEDH5ValidationError) as exc:
            normalize_masks({0: np.zeros((4, 5), bool)})
        assert exc.value.code == "E303"

    def test_agreement_releases_the_gil(self, masks):
        a, b = masks
        released(lambda: dice(a, b))
        released(lambda: iou(a, b))

    def test_agreement_broadcasts_without_copying(self, masks):
        """A plane against a volume counts as NumPy's broadcast would, with
        no broadcast copy made, and with the GIL released."""
        a, b = masks
        plane = b[:1]
        both = np.broadcast_to(plane, a.shape)
        expect = 2 * (a & both).sum() / (a.sum() + both.sum())
        assert dice(a, plane) == pytest.approx(expect)
        assert iou(plane, a) == pytest.approx((a & both).sum() / (a | both).sum())
        released(lambda: dice(a, plane))

    def test_large_array_arguments_copy_without_the_gil(self, tmp_path: Path, masks):
        """An argument in another layout than C is copied into one, which for
        a large array is a pass over all of it: done with the GIL released."""
        mask = np.asfortranarray(masks[0])
        assert not mask.flags.c_contiguous
        runs = []
        for k in range(3):
            path = tmp_path / f"s{k}.medh5"
            with medh5.create(path, sample_id="s") as w:
                w.add_grid("g", shape=mask.shape, spacing=(1.0, 1.0, 1.0))
                ct = np.zeros(mask.shape, np.int8)
                w.add_image("CT", ct, grid="g", modality="CT")
                # The call alone: the file's creation and commit are other
                # work, and a longer window only catches more scheduler noise,
                # which a slow runner shows at tens of milliseconds.
                runs.append(
                    watched(
                        lambda w=w: w.add_segmentation(
                            "seg", grid="g", masks={1: mask}, encoding="bitmask"
                        )
                    )
                )
        # The best of three: one scheduler hiccup is not the GIL.
        took, longest, idle = min(runs, key=lambda r: r[1] - r[2])
        start = time.perf_counter()
        np.ascontiguousarray(mask)
        copy = time.perf_counter() - start
        assert longest < max(copy / 2, 0.02) + idle, (
            f"no tick for {longest:.3f} of {took:.3f}"
        )
        with medh5.open(path) as s:
            assert np.array_equal(s.annotations["seg"].dense(), mask[np.newaxis])


class TestHandles:
    def test_closing_a_written_dataset_releases_the_gil(self, tmp_path: Path):
        """The last handle to a dataset just written flushes its chunk cache
        when it goes: compression and writes, with the GIL released."""
        shape = (192, 256, 256)
        ct = np.random.default_rng(4).integers(-1000, 1000, shape, dtype=np.int16)
        runs = []
        for k in range(3):
            with medh5.create(tmp_path / f"c{k}.medh5", sample_id="c") as w:
                w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
                held = [w.add_image("CT", ct, grid="g", modality="CT")]
                runs.append(watched(held.clear))
        took, longest, idle = min(runs, key=lambda r: (r[1] - r[2]) / r[0])
        assert longest < 0.25 * took + idle, (
            f"no tick for {longest:.3f} of {took:.3f} s"
        )


class TestGeometric:
    def test_encode_contours_releases_while_packing(self):
        rng = np.random.default_rng(1)
        polygons = [
            Polygon(
                vertices=np.c_[np.full(400_000, z), rng.random((400_000, 2)) * 500],
                class_id=1,
                plane=(0, z),
            )
            for z in range(8)
        ]
        released(lambda: encode_contours(polygons, ndim=3))

    @pytest.mark.parametrize("rows", [1_000_000])
    def test_row_encoders_release_the_gil(self, rows: int):
        rng = np.random.default_rng(2)
        boxes = np.sort(rng.random((rows, 3, 2)) * 100, axis=2)
        class_ids = np.ones(rows, np.int64)
        released(lambda: encode_boxes(boxes, class_ids))
        released(lambda: encode_points(boxes[:, :, 0]))
        # The class ids too: an integer array is not walked as Python ints.
        released(lambda: encode_points(boxes[:, :, 0], class_ids=class_ids))

    def test_refuse_outside_releases_the_gil(self):
        inside = np.ones(40_000_000, dtype=bool)
        released(lambda: refuse_outside(inside))

    def test_to_world_releases_the_gil(self, tmp_path: Path):
        path = tmp_path / "p.medh5"
        with medh5.create(path, sample_id="p") as w:
            w.add_grid("g", shape=(10, 10, 10), spacing=(0.5, 0.7, 1.1))
            w.add_image("CT", np.zeros((10, 10, 10), np.int16), grid="g", modality="CT")
            w.add_points("pts", np.zeros((1, 3)), grid="g")
        coords = np.random.default_rng(3).random((2_000_000, 3)) * 10
        with medh5.open(path) as s:
            ann = s.annotations["pts"]
            # Index space on an axis-aligned grid at the origin.
            assert np.allclose(ann.to_world(coords), coords * (0.5, 0.7, 1.1))
            assert np.allclose(ann.to_index(coords), coords)
            released(lambda: ann.to_world(coords))
            released(lambda: ann.to_index(coords))


# -- one thread at a time (E4) ------------------------------------------------------

BUSY = "is used by one thread at a time"


def refused_while(
    hold: Callable[[int], Any],
    call: Callable[[], Any],
    *,
    attempts: int = 20,
    passed: str | None = None,
    pause: float = 0.0,
) -> medh5.MEDH5Error:
    """The refusal *call* meets from another thread while *hold* runs, tried
    until one call lands inside one of *attempts* runs of *hold*.  Whatever
    else the other thread meets fails the test: a refusal is the only way a
    call on an object in use may end --- but an error naming *passed*, which
    the call raises once it is let through, and which means try again, after
    *pause* (as long before the first call): a call let through holds the
    object too, and one that never pauses keeps *hold* from starting."""
    for attempt in range(attempts):
        started, done = threading.Event(), threading.Event()
        refused: list[medh5.MEDH5Error] = []
        failed: list[BaseException] = []

        def other(
            started: threading.Event,
            done: threading.Event,
            refused: list[medh5.MEDH5Error],
            failed: list[BaseException],
        ) -> None:
            started.set()
            time.sleep(pause)
            while not done.is_set():
                try:
                    call()
                except medh5.MEDH5Error as e:
                    if passed is not None and passed in str(e):
                        time.sleep(pause)
                        continue
                    (refused if BUSY in str(e) else failed).append(e)
                    return
                except BaseException as e:  # pragma: no cover - the failure
                    failed.append(e)
                    return

        thread = threading.Thread(target=other, args=(started, done, refused, failed))
        thread.start()
        started.wait()
        try:
            hold(attempt)
        except medh5.MEDH5Error as e:
            # The other thread's call got there first: this run held nothing.
            if BUSY not in str(e):
                raise
        finally:
            done.set()
            thread.join()
        assert not failed, failed
        if refused:
            return refused[0]
    pytest.fail("no call of the other thread's landed while the first's ran")


def hammered(calls: list[Callable[[], Any]], *, rounds: int = 200) -> int:
    """Run each of *calls* *rounds* times, all at once from as many threads;
    the number refused.  Anything else a call raises fails the test."""
    barrier = threading.Barrier(len(calls))
    refused: list[medh5.MEDH5Error] = []
    failed: list[BaseException] = []

    def run(call: Callable[[], Any]) -> None:
        barrier.wait()
        for _ in range(rounds):
            try:
                call()
            except medh5.MEDH5Error as e:
                if BUSY not in str(e):  # pragma: no cover - the failure
                    failed.append(e)
                refused.append(e)
            except BaseException as e:  # pragma: no cover - the failure
                failed.append(e)

    threads = [threading.Thread(target=run, args=(c,)) for c in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failed, failed[:3]
    return len(refused)


@pytest.fixture(scope="module")
def volume() -> np.ndarray:
    """Enough voxels that writing them takes a while, and compresses slowly."""
    return np.random.default_rng(5).integers(-1000, 1000, (96, 256, 256), np.int16)


class TestOneThreadAtATime:
    """A mutable object of the extension is used by one thread at a time: a
    call that finds it in use is refused with a `MEDH5Error`, never made to
    wait (which could deadlock a free-threaded interpreter) and never raced
    (`RuntimeError: Already borrowed` was the borrow checker's refusal)."""

    def test_two_threads_on_one_writer_get_a_medh5_error(
        self, tmp_path: Path, volume: np.ndarray
    ):
        with medh5.create(tmp_path / "w.medh5", sample_id="w") as w:
            w.add_grid("g", shape=volume.shape, spacing=(1.0, 1.0, 1.0))
            images = itertools.count()

            def hold(_: int) -> None:
                w.add_image(f"CT{next(images)}", volume, grid="g", modality="CT")

            error = refused_while(hold, lambda: w.identity(sex="F"))
            assert f"a SampleWriter {BUSY}" in str(error)
            assert not isinstance(error, RuntimeError)
            # Readers are refused too while the writer is changed under them,
            # its document as much as the writer itself.
            assert BUSY in str(refused_while(hold, lambda: w.closed))
            assert BUSY in str(refused_while(hold, lambda: w.document.identity))
        with medh5.open(tmp_path / "w.medh5") as s:
            # A hold whose call the other thread's got ahead of wrote nothing.
            assert s.images and set(s.images) <= {f"CT{k}" for k in range(next(images))}

    def test_a_refused_call_changes_nothing(self, tmp_path: Path, volume: np.ndarray):
        with medh5.create(tmp_path / "w.medh5", sample_id="w") as w:
            w.add_grid("g", shape=volume.shape, spacing=(1.0, 1.0, 1.0))
            refused_while(
                lambda k: w.add_image(f"CT{k}", volume, grid="g", modality="CT"),
                lambda: w.extra("org.example", {"seen": True}),
            )
            w.extra("org.example", {"seen": False})
        with medh5.open(tmp_path / "w.medh5") as s:
            assert s.document.extra["org.example"] == {"seen": False}

    def test_two_threads_on_one_cache_writer_get_a_medh5_error(self, tmp_path: Path):
        from medh5 import _core
        from medh5.task import SourceRef
        from tests.helpers import write_sample

        source = SourceRef.pin(write_sample(tmp_path / "s.medh5")).to_json()

        handle = _core.cache_create(
            os.fspath(tmp_path / "c.medh5cache"),
            {
                "level": "event",
                "encoder": {"name": "fixture", "revision": "1"},
                "output": {"dtype": "float32", "shape": [4_000_000]},
            },
        )
        values = np.random.default_rng(6).random(4_000_000).astype(np.float32)

        def entry(k: str) -> dict[str, Any]:
            return {"entry_id": k, "sources": [source], "event_id": k}

        error = refused_while(
            lambda k: handle.add(entry(f"e{k}"), values),
            # Let through, the cache refuses its shape (T402): try again.
            lambda: handle.add(entry("other"), values[:4]),
            passed="T402",
            pause=0.01,
        )
        assert f"a CacheWriterHandle {BUSY}" in str(error)
        handle.abort()

    def test_a_write_resolves_its_classes_against_one_label_set(self, tmp_path: Path):
        """A label set another thread swaps in lands before or after a
        write's class names are resolved, never between two of them: every
        polygon of one call has the one class."""
        from medh5.labels import LabelClass, LabelSet
        from tests.helpers import SHAPE

        swapped = [
            LabelSet(
                "t",
                version="1.0.0",
                classes=[
                    LabelClass(liver, "liver", "Liver"),
                    LabelClass(3 - liver, "spleen", "Spleen"),
                ],
            )
            for liver in (1, 2)
        ]
        square = np.array([[4.0, 2, 2], [4, 2, 9], [4, 9, 9], [4, 9, 2]])
        polygons = [Polygon(square, class_id="liver", plane=(0, 4))] * 2_000  # type: ignore[arg-type]
        done = threading.Event()
        failed: list[BaseException] = []
        written: list[str] = []

        with medh5.create(tmp_path / "w.medh5", codec="portable") as w:
            w.label_set(swapped[0])
            w.add_grid("ct", shape=SHAPE, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(SHAPE, np.int16), grid="ct", modality="CT")

            def swap() -> None:
                for k in itertools.count():
                    if done.is_set():
                        return
                    try:
                        w.label_set(swapped[k % 2])
                    except medh5.MEDH5Error as e:
                        if BUSY not in str(e):  # pragma: no cover - the failure
                            failed.append(e)
                            return

            thread = threading.Thread(target=swap)
            thread.start()
            try:
                for k in range(10):
                    try:
                        w.add_contours(f"rt{k}", polygons, grid="ct")
                    except medh5.MEDH5Error as e:
                        assert BUSY in str(e)
                    else:
                        written.append(f"rt{k}")
            finally:
                done.set()
                thread.join()
            assert not failed, failed
        assert written
        with medh5.open(tmp_path / "w.medh5") as s:
            for ann_id in written:
                classes = {p.class_id for p in s.annotations[ann_id].polygons()}
                assert len(classes) == 1, (ann_id, classes)

    def test_two_threads_on_one_patch_draw_get_a_medh5_error(
        self, tmp_path: Path, label_set: Any
    ):
        from medh5.sampling import PatchSampler
        from tests.helpers import SHAPE, block, write_sample

        path = write_sample(
            tmp_path / "s.medh5",
            label_set=label_set,
            masks={1: block(SHAPE, (2, 2, 2))},
        )
        sampler = PatchSampler(patch_size=(16, 24, 24))
        with medh5.open(path) as s:
            draws = sampler._engine().iter_draws(s, None, 10_000, 0)
            assert hammered([lambda: next(draws)] * 2, rounds=2_000) >= 0
            assert draws.__length_hint__() >= 0

    @pytest.mark.parametrize("threads", [8])
    def test_shared_records_are_refused_or_kept_whole(self, threads: int):
        """Short calls on one document, payload or provenance from many
        threads: each ends whole or refused, and what the object holds after
        is every edit that was not refused."""
        from medh5.annotations.base import AnnotationPayload
        from medh5.curation import Agent, Provenance
        from medh5.document import new_document

        doc = new_document("s")
        payload = AnnotationPayload("points")
        prov = Provenance()
        kept: list[str] = []
        lock = threading.Lock()

        def edit(k: int) -> Callable[[], None]:
            count = iter(range(10**9))

            def call() -> None:
                key = f"t{k}.{next(count)}"
                doc.extra[key] = {"k": k}
                with lock:
                    kept.append(key)
                payload.class_ids = (k,)
                assert payload.kind == "points"
                prov.add_agent(Agent(key, "software", key))
                _ = doc.to_json(), len(prov), prov.to_json()

            return call

        hammered([edit(k) for k in range(threads)], rounds=50)
        assert set(kept) <= set(doc.extra)
        assert len(payload.class_ids) == 1
        assert len(prov.agents) <= threads * 50

    @pytest.mark.parametrize("threads", [8])
    def test_a_documents_provenance_agrees_with_its_document(self, threads: int):
        """Threads replacing one agent of a document's provenance: the graph
        and the document end with the same agent, the last one not refused,
        because the provenance is held across both writes."""
        from medh5.curation import Agent
        from medh5.document import new_document

        doc = new_document("s")
        prov = doc.provenance

        def replace(k: int) -> Callable[[], None]:
            count = iter(range(10**9))
            return lambda: prov.add_agent(
                Agent("a", "software", f"t{k}.{next(count)}"), replace=True
            )

        hammered([replace(k) for k in range(threads)], rounds=200)
        assert prov.to_json()["agents"] == doc.provenance.to_json()["agents"]

    @pytest.mark.parametrize("document", [False, True])
    def test_a_provenance_reads_whole_while_its_activities_change(self, document: bool):
        """Threads editing an activity's `params` while others read the
        provenance: every read ends whole, and the last holds every edit.
        A dict is read as it stands in one step, and a document's activity
        under the lock its edits hold."""
        from medh5.curation import Activity, Provenance
        from medh5.document import new_document

        threads = 4
        given = Activity("a", "import", params={"seen": {}})
        if document:
            prov = new_document("s").provenance
            prov.add_activity(given)
        else:
            prov = Provenance(activities=[given])
        act = prov.activities[0]
        rounds = 200

        def edit(k: int) -> Callable[[], None]:
            count = iter(range(10**9))
            return lambda: act.params["seen"].__setitem__(f"{k}.{next(count)}", k)

        def read() -> None:
            seen = prov.to_json()["activities"][0]["params"]["seen"]
            assert all(isinstance(v, int) for v in seen.values())
            assert all(
                isinstance(v, int) for v in act.to_json()["params"]["seen"].values()
            )

        calls = [edit(k) for k in range(threads)] + [read] * threads
        assert hammered(calls, rounds=rounds) == 0
        every = {f"{k}.{n}" for k in range(threads) for n in range(rounds)}
        assert set(prov.to_json()["activities"][0]["params"]["seen"]) == every

    @pytest.mark.parametrize("document", [False, True])
    def test_a_replaced_activity_is_not_read_as_the_graphs(self, document: bool):
        """A thread replacing an activity, then editing the one it replaced,
        while others read the provenance: no read holds the edit, made after
        the activity left the graph.  The activities are read outside the
        provenance's lock, so a read is kept only if the graph still holds
        what was read."""
        from medh5.curation import Activity, Provenance
        from medh5.document import new_document

        def fresh() -> Activity:
            return Activity("a", "import", params={"left": False})

        if document:
            prov = new_document("s").provenance
            prov.add_activity(fresh())
        else:
            prov = Provenance(activities=[fresh()])

        def replace() -> None:
            old = prov.activities[0]
            prov.add_activity(fresh(), replace=True)
            old.params["left"] = True

        def read() -> None:
            assert prov.to_json()["activities"][0]["params"] == {"left": False}

        hammered([replace] + [read] * 3, rounds=2000)

    def test_a_view_read_while_another_thread_edits_it_is_never_empty(self):
        """A thread editing a document's `extra` while others read the same
        view without its lock: every read finds the key nobody touches.  The
        view is brought up to date a key at a time, not cleared and filled
        again."""
        from medh5.document import new_document

        extra = new_document("s").extra
        extra["kept"] = 0
        count = iter(range(10**9))

        def edit() -> None:
            extra["edited"] = next(count)

        def read() -> None:
            assert extra["kept"] == 0
            assert len(extra) >= 1
            assert "kept" in dict(extra)

        hammered([edit] + [read] * 3, rounds=2000)

    def test_a_view_edit_refused_changed_nothing_and_one_kept_is_not_refused(
        self,
    ):
        """A thread setting and deleting keys of a document's `extra` while
        others replace its `acquisition`: an edit that raises left the
        document as it was, and one the document kept does not raise.  The
        edit and the view's reread are one step, under the document's one
        guard."""
        from medh5.document import new_document

        doc = new_document("s")
        extra = doc.extra
        kept: set[str] = set()
        count = iter(range(10**9))
        other = iter(range(10**9))

        def edit() -> None:
            key = f"k{next(count)}"
            extra[key] = 0
            kept.add(key)
            del extra[key]
            kept.discard(key)

        def write() -> None:
            doc.acquisition = {"n": next(other)}

        hammered([edit] + [write] * 3, rounds=2000)
        for _ in range(100):
            try:
                held = set(doc.extra)
                break
            except medh5.MEDH5Error:  # pragma: no cover - only while busy
                continue
        assert held == kept
        assert set(extra) == kept

    def test_a_writers_acquisition_kept_is_not_refused(self, tmp_path: Path):
        """A thread merging acquisition parameters through a writer while
        another adds images to it: a call that raises left the document as
        it was, and one the document kept returns the view it left, not a
        refusal.  The view is read under the guard the edit took."""
        image = np.zeros((16, 32, 32), np.int16)
        with medh5.create(tmp_path / "w.medh5", sample_id="w") as w:
            w.add_grid("g", shape=image.shape, spacing=(1.0, 1.0, 1.0))
            kept: set[str] = set()
            count = iter(range(10**9))
            other = iter(range(10**9))

            def merge() -> None:
                key = f"k{next(count)}"
                params = w.acquisition("CT", **{key: 0})
                kept.add(key)
                assert key in params

            def write() -> None:
                w.add_image(f"CT{next(other)}", image, grid="g", modality="CT")

            hammered([merge, write], rounds=2000)
            assert set(w.document.acquisition.get("CT", {})) == kept
            w.abort()

    @pytest.mark.parametrize("threads", [8])
    def test_threads_setting_one_default_all_get_it(self, threads: int):
        """Threads calling `setdefault` for one missing key, of a document's
        `extra` and of a dict nested in it: every caller gets the one default
        kept, and the document holds it, because the look and the write are
        one step."""
        from medh5.document import new_document

        doc = new_document("s")
        doc.extra["log"] = {}
        extra, member = doc.extra, doc.extra["log"]
        got: dict[str, set[int]] = {}
        lock = threading.Lock()
        rounds = 100

        def call(k: int) -> Callable[[], None]:
            count = iter(range(10**9))

            def run() -> None:
                n = next(count)
                pairs = [
                    (f"m{n}", extra.setdefault(f"m{n}", k)),
                    (f"n{n}", member.setdefault(f"n{n}", k)),
                ]
                with lock:
                    for key, value in pairs:
                        got.setdefault(key, set()).add(value)

            return run

        assert hammered([call(k) for k in range(threads)], rounds=rounds) == 0
        stored = doc.extra
        for n in range(rounds):
            assert got[f"m{n}"] == {stored[f"m{n}"]}
            assert got[f"n{n}"] == {stored["log"][f"n{n}"]}

    @pytest.mark.parametrize("threads", [8])
    def test_threads_editing_one_nested_member_keep_every_edit(self, threads: int):
        """Threads appending to a list nested in one document member, and
        setting keys of a dict beside it: every edit is kept, in the view and
        in the document, because a change and the rewrite it makes are one
        step."""
        from medh5.document import new_document

        doc = new_document("s")
        doc.extra["log"] = {"items": [], "seen": {}}
        member = doc.extra["log"]
        rounds = 100

        def edit(k: int) -> Callable[[], None]:
            count = iter(range(10**9))

            def call() -> None:
                n = next(count)
                member["items"].append(f"{k}.{n}")
                member["seen"][f"{k}.{n}"] = n

            return call

        assert hammered([edit(k) for k in range(threads)], rounds=rounds) == 0
        every = {f"{k}.{n}" for k in range(threads) for n in range(rounds)}
        assert set(member["items"]) == every
        assert set(doc.extra["log"]["items"]) == every
        assert set(doc.extra["log"]["seen"]) == every

    @pytest.mark.parametrize(
        "op",
        [
            "imul",
            "setitem",
            "slice",
            "delitem",
            "insert",
            "pop",
            "remove",
            "key",
            "int key",
            "setdefault",
        ],
    )
    def test_no_callers_code_runs_under_the_live_views_lock(self, op: str):
        """An index's `__index__`, an item's `__eq__` and a key's `__hash__`
        are the caller's code, and may wait on another thread's edit of the
        same document: they run before the live views' lock is taken, so that
        edit lands (under the lock, it waited for the first and the first for
        it)."""
        from medh5.document import new_document

        doc = new_document("s")
        doc.extra["log"] = {"items": ["a", "b", "c"], "seen": {}}
        member = doc.extra["log"]
        items, seen = member["items"], member["seen"]
        landed: list[bool] = []
        waiting: list[threading.Thread] = []

        def hook() -> None:
            if landed or waiting:
                return
            other = threading.Thread(target=seen.__setitem__, args=("other", 1))
            other.start()
            other.join(timeout=5)
            if other.is_alive():
                waiting.append(other)
            landed.append(not other.is_alive())

        class Index:
            def __init__(self, n: int) -> None:
                self.n = n

            def __index__(self) -> int:
                hook()
                return self.n

        class Equal:
            __hash__ = None  # type: ignore[assignment]

            def __eq__(self, other: object) -> bool:
                hook()
                return other == "b"

        class Key(str):
            __slots__ = ()

            def __hash__(self) -> int:
                hook()
                return str.__hash__(self)

            def __eq__(self, other: object) -> bool:
                hook()
                return str.__eq__(self, other)

        class Number(int):
            def __hash__(self) -> int:
                hook()
                return int.__hash__(self)

            def __str__(self) -> str:
                hook()
                return int.__str__(self)

        ops: dict[str, Callable[[], object]] = {
            "imul": lambda: items.__imul__(Index(1)),
            "setitem": lambda: items.__setitem__(Index(0), "x"),
            "slice": lambda: items.__setitem__(slice(Index(0), Index(1)), ["x"]),
            "delitem": lambda: items.__delitem__(Index(0)),
            "insert": lambda: items.insert(Index(0), "x"),
            "pop": lambda: items.pop(Index(0)),
            "remove": lambda: items.remove(Equal()),
            "key": lambda: seen.__setitem__(Key("k"), 2),
            "int key": lambda: seen.__setitem__(Number(7), 2),
            "setdefault": lambda: seen.setdefault(Key("k"), 2),
        }
        try:
            ops[op]()
        finally:
            for other in waiting:  # pragma: no cover - the failure
                other.join()
        # A key is hashed as the built-in value it holds: its hook need not run.
        assert landed in ([], [True])
        assert doc.extra["log"]["seen"].get("other") == (1 if landed else None)
        assert doc.extra["log"]["items"] == list(items)

    @staticmethod
    def _edit_on_another_thread(edit: Callable[[], None]) -> Callable[[], None]:
        """A hook that, the first time it runs, makes *edit* on another thread
        and waits for it; it fails if the edit could not land."""
        landed: list[bool] = []

        def hook() -> None:
            if landed:
                return
            other = threading.Thread(target=edit)
            other.start()
            other.join(timeout=5)
            landed.append(not other.is_alive())
            assert landed == [True], "the edit waited on the live views' lock"

        return hook

    def test_a_sort_is_not_kept_after_an_item_it_sorted_by_changed(self):
        """A dict in a list changed on another thread while the list is
        sorted by it: the order computed from the old value is not kept."""
        from medh5.document import new_document

        doc = new_document("s")
        doc.extra["log"] = {"items": [{"k": 2}, {"k": 1}]}
        items = doc.extra["log"]["items"]
        first = items[0]
        hook = self._edit_on_another_thread(lambda: first.__setitem__("k", 0))

        def by_k(item: dict[str, int]) -> int:
            hook()
            return item["k"]

        with pytest.raises(RuntimeError, match="changed on another thread"):
            items.sort(key=by_k)
        assert doc.extra["log"]["items"] == [{"k": 0}, {"k": 1}] == list(items)

    def test_remove_does_not_take_an_item_that_changed_after_it_matched(self):
        """A dict in a list that matched the value to remove, changed on
        another thread before the removal: it is not removed, as it no
        longer equals the value."""
        from medh5.document import new_document

        doc = new_document("s")
        doc.extra["log"] = {"items": [{"a": 1}, {"a": 2}]}
        items = doc.extra["log"]["items"]
        first = items[0]
        hook = self._edit_on_another_thread(lambda: first.__setitem__("a", 9))

        class Matches:
            __hash__ = None  # type: ignore[assignment]

            def __eq__(self, other: object) -> bool:
                found = other == {"a": 1}
                hook()
                return found

        with pytest.raises(ValueError, match="not in list"):
            items.remove(Matches())
        assert doc.extra["log"]["items"] == [{"a": 9}, {"a": 2}] == list(items)

    @pytest.mark.parametrize(
        "kind", ["activity", "provenance", "label class", "label set", "timeline"]
    )
    def test_a_standalone_records_callers_code_runs_outside_the_lock(self, kind: str):
        """A caller's object in a record no document handed out is converted
        by `to_json` without the live views' lock, which guards nothing of
        such a record: its conversion may wait on an edit of a document."""
        from medh5 import Timeline, Timepoint
        from medh5.curation import Activity, Provenance
        from medh5.document import new_document
        from medh5.labels import LabelClass, LabelSet

        doc = new_document("s")
        doc.extra["log"] = {"seen": {}}
        seen = doc.extra["log"]["seen"]
        hook = self._edit_on_another_thread(lambda: seen.__setitem__("other", 1))

        class Value:
            def to_json(self) -> str:
                hook()
                return "1.2"

        record: Any
        if kind == "activity":
            record = Activity("a", "import")
            record.params["x"] = Value()
        elif kind == "provenance":
            record = Provenance(activities=[Activity("a", "import")])
            record.activities[0].params["x"] = Value()
        elif kind == "label class":
            record = LabelClass(1, "a", "A")
            record.properties["x"] = Value()
        elif kind == "label set":
            record = LabelSet("v", [LabelClass(1, "a", "A")])
            record[1].properties["x"] = Value()
        else:
            record = Timeline([Timepoint("tp0", 0)])
            record[0].series_uids["ct"] = Value()
        assert "1.2" in str(record.to_json())
        assert doc.extra["log"]["seen"] == {"other": 1}

    @pytest.mark.parametrize("given", ["mapping", "record"])
    def test_a_quality_members_callers_code_runs_outside_the_lock(self, given: str):
        """A quality record set on the view is converted before the live
        views' lock is taken, as any other member is: its conversion may wait
        on an edit of a document."""
        from medh5.curation import Agreement, QualityRecord
        from medh5.document import new_document

        doc = new_document("s")
        doc.extra["log"] = {"seen": {}}
        seen = doc.extra["log"]["seen"]
        hook = self._edit_on_another_thread(lambda: seen.__setitem__("other", 1))

        class Dice:
            def to_json(self) -> float:
                hook()
                return 0.5

        value: Any = {
            "status": "reviewed",
            "agreement": [{"metric": "dice", "value": 0.5, "per_class": {"1": Dice()}}],
        }
        if given == "record":
            value = QualityRecord("reviewed", agreement=(Agreement("dice", 0.5),))
            value.agreement[0].per_class["1"] = Dice()
        doc.quality["q"] = value
        assert doc.quality["q"].agreement[0].per_class == {"1": 0.5}
        assert doc.extra["log"]["seen"] == {"other": 1}

    def test_a_key_the_document_cannot_hold_is_refused_before_any_change(self):
        """A nested dict refuses a key JSON has no spelling for as it is
        given, and the member stays editable: refused by the rewrite, it was
        set in the view first, which then refused every further edit."""
        from medh5.document import new_document

        doc = new_document("s")
        doc.extra["log"] = {"seen": {}}
        seen = doc.extra["log"]["seen"]
        with pytest.raises(TypeError, match="keys must be"):
            seen[("a", "b")] = 1
        assert seen.pop(("a", "b"), None) is None
        with pytest.raises(KeyError):
            del seen[("a", "b")]
        seen["b"] = 2
        assert doc.extra["log"]["seen"] == {"b": 2}

    def test_a_label_set_reads_its_classes_in_one_step(self):
        """One thread setting a count in one class's `properties` and then
        another's, while others read the document's label set: a read holds
        the two counts equal or one apart, as no moment had them otherwise,
        because the classes are read under the lock the edits take."""
        from medh5.document import new_document
        from medh5.labels import LabelClass, LabelSet

        doc = new_document("s")
        doc.label_set = LabelSet(
            "v", [LabelClass(1, "a", "A"), LabelClass(2, "b", "B")]
        )
        ls = doc.label_set
        first, second = ls[1].properties, ls[2].properties
        count = iter(range(10**9))

        def edit() -> None:
            n = next(count)
            first["n"] = n
            second["n"] = n

        def read() -> None:
            classes = ls.to_json()["classes"]
            a, b = (c.get("properties", {}).get("n", -1) for c in classes)
            assert a - 1 <= b <= a, (a, b)

        assert hammered([edit] + [read] * 4, rounds=300) == 0

    def test_a_timeline_reads_its_timepoints_in_one_step(self):
        """One thread setting a series in one timepoint and then another,
        while others read the document's timeline: a read holds the two equal
        or one apart, as no moment had them otherwise."""
        from medh5 import Timeline, Timepoint
        from medh5.document import new_document

        doc = new_document("s")
        doc.timepoints = Timeline([Timepoint("tp0", 0), Timepoint("tp1", 1)])
        timeline = doc.timepoints
        first, second = timeline[0].series_uids, timeline[1].series_uids
        count = iter(range(1, 10**9))

        def edit() -> None:
            n = next(count)
            first["ct"] = f"1.{n}"
            second["ct"] = f"1.{n}"

        def read() -> None:
            a, b = (
                int(p.get("series_uids", {}).get("ct", "1.0")[2:])
                for p in timeline.to_json()
            )
            assert a - 1 <= b <= a, (a, b)

        assert hammered([edit] + [read] * 4, rounds=300) == 0

    def test_setdefault_reads_no_default_it_does_not_keep(self):
        """`setdefault` on a key the member holds returns what it holds and
        never converts the default, as a dict never reads one it does not
        keep: a default the document could not hold is no error there."""
        from medh5.document import new_document

        doc = new_document("s")
        doc.extra["kept"] = 1
        doc.extra["log"] = {"kept": 2}
        assert doc.extra.setdefault("kept", object()) == 1
        assert doc.extra["log"].setdefault("kept", object()) == 2
        with pytest.raises(TypeError):
            doc.extra["log"].setdefault("new", object())
        assert "new" not in doc.extra["log"]

    def test_a_payload_reads_whole_while_its_dicts_change(self):
        """Threads adding and removing a payload's datasets and attributes
        while others describe it: each dict is read as it stands in one step,
        so every read ends whole."""
        from medh5.annotations.base import AnnotationPayload

        payload = AnnotationPayload("points")
        datasets, attrs = payload.datasets, payload.attrs
        threads = 4

        def edit(k: int) -> Callable[[], None]:
            count = iter(range(10**9))

            def call() -> None:
                n = next(count)
                datasets[f"{k}.{n}"] = np.zeros(1)
                attrs[f"{k}.{n}"] = n
                if n:
                    del datasets[f"{k}.{n - 1}"], attrs[f"{k}.{n - 1}"]

            return call

        def read() -> None:
            assert payload.describe()["kind"] == "points"

        calls = [edit(k) for k in range(threads)] + [read] * threads
        assert hammered(calls, rounds=500) == 0
        assert len(payload.describe()["datasets"]) == threads

    def test_a_payloads_size_and_repr_read_whole_while_its_datasets_change(self):
        """Declined in 2.2.0's review (#76): `nbytes` and `repr` walk the
        payload's dict through `values()` and `keys()`, which PyO3 takes as
        a list in one call (`PyDict_Values`, `PyDict_Keys`), under the
        dict's own lock on a free-threaded build.  No snapshot is needed:
        threads resizing the dict while others read both never see a dict
        change size under them."""
        from medh5.annotations.base import AnnotationPayload

        payload = AnnotationPayload("points")
        datasets = payload.datasets
        threads = 4

        def edit(k: int) -> Callable[[], None]:
            count = iter(range(10**9))

            def call() -> None:
                n = next(count)
                datasets[f"{k}.{n}"] = np.zeros(1)
                if n:
                    del datasets[f"{k}.{n - 1}"]

            return call

        def read() -> None:
            assert payload.nbytes >= 0
            assert repr(payload).startswith("AnnotationPayload(")

        calls = [edit(k) for k in range(threads)] + [read] * threads
        assert hammered(calls, rounds=500) == 0
        assert payload.nbytes == threads * np.zeros(1).nbytes

    def test_a_replaced_dict_is_not_read_as_the_payloads(self):
        """A thread replacing a payload's datasets, then editing the dict it
        replaced, while others describe the payload: no description holds
        the edit, made after the dict left the payload.  Each dict is taken
        whole while the payload's lock says it is the payload's."""
        from medh5.annotations.base import AnnotationPayload

        payload = AnnotationPayload("points")
        payload.datasets = {"kept": np.zeros(1)}

        def replace() -> None:
            old = payload.datasets
            payload.datasets = {"kept": np.zeros(1)}
            old["left"] = np.zeros(1)

        def read() -> None:
            assert list(payload.describe()["datasets"]) == ["kept"]

        hammered([replace] + [read] * 3, rounds=2000)


class TestShared:
    """What threads may share: an open sample, read from many at once, and the
    process-wide registries."""

    @staticmethod
    def _ends_in_a_forked_child(call: Callable[[], object]) -> bool:
        """Whether *call* ends in a child forked now, within ten seconds."""
        read, write = os.pipe()
        with warnings.catch_warnings():
            # Forking a process with threads is what is under test.
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:  # pragma: no cover - the child reports through the pipe
            ok = False
            try:
                os.close(read)
                call()
                ok = True
            finally:
                os.write(write, b"1" if ok else b"0")
                os._exit(0)
        os.close(write)
        ready, _, _ = select.select([read], [], [], 10)
        if not ready:  # pragma: no cover - the failure: the child hangs
            os.kill(pid, signal.SIGKILL)
        ended = bool(ready) and os.read(read, 1) == b"1"
        os.close(read)
        os.waitpid(pid, 0)
        return ended

    def _runs_in_a_child_forked_while_held(
        self, lock: Any, call: Callable[[], object]
    ) -> bool:
        """Whether *call* ends in a child forked while another thread holds
        *lock*: that thread is not in the child to release it."""
        held, release = threading.Event(), threading.Event()

        def hold() -> None:
            with lock:
                held.set()
                release.wait()

        holder = threading.Thread(target=hold)
        holder.start()
        held.wait()
        try:
            return self._ends_in_a_forked_child(call)
        finally:
            release.set()
            holder.join()

    def _runs_in_children_forked_during(
        self, work: Callable[[], object], call: Callable[[], object], forks: int = 20
    ) -> bool:
        """Whether *call* ends in every one of *forks* children forked while
        another thread does *work* over and over: one forked in the middle of
        it inherits whatever lock the work held, and no thread to release it."""
        stop = threading.Event()

        def busy() -> None:
            while not stop.is_set():
                work()

        worker = threading.Thread(target=busy)
        worker.start()
        try:
            return all(self._ends_in_a_forked_child(call) for _ in range(forks))
        finally:
            stop.set()
            worker.join()

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX: fork")
    def test_a_fork_while_the_registry_is_held_leaves_the_child_a_registry(self):
        """A child forked while another thread loads a vocabulary loads one
        too: the hook gives the child a registry lock of its own."""
        from medh5 import labels

        name = labels.available()[0]
        assert self._runs_in_a_child_forked_while_held(
            labels._REGISTRY, lambda: labels.load(name)
        )

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX: fork")
    def test_a_fork_while_a_live_view_is_edited_leaves_the_child_its_views(self):
        """A child forked while another thread edits a document's live view
        edits views and reads documents too: the hook gives the child a
        views' lock of its own, which the engine looks up on each use."""
        from medh5 import _live
        from medh5.curation import Activity
        from medh5.document import new_document

        doc = new_document("s")
        doc.provenance.add_activity(Activity("a", "import"))
        params = doc.provenance.activities[0].params
        params["x"] = 1
        assert doc.provenance.to_json()["activities"][0]["params"] == {"x": 1}

        def edit() -> None:
            doc.extra["k"] = 1
            params["y"] = 2
            assert doc.provenance.to_json()["activities"][0]["params"] == {
                "x": 1,
                "y": 2,
            }

        assert self._runs_in_a_child_forked_while_held(_live._LOCK, edit)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX: fork")
    def test_a_fork_during_another_threads_reads_leaves_the_child_hdf5(
        self, tmp_path: Path
    ):
        """A child forked while another thread reads a file reads one too:
        the fork waits for HDF5's lock, which a thread in the middle of a read
        holds, and which the child would otherwise inherit held forever."""
        from tests.helpers import write_sample

        other = write_sample(tmp_path / "t.medh5")

        def read() -> None:
            with medh5.open(other) as s:
                s.images["CT_tp0"].read()

        with medh5.open(write_sample(tmp_path / "s.medh5")) as s:
            image = s.images["CT_tp0"]
            assert self._runs_in_children_forked_during(image.read, read)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX: fork")
    def test_a_fork_during_another_threads_loads_leaves_the_child_a_registry(self):
        """A child forked while another thread loads a registered vocabulary
        loads one: the fork waits for the engine's registry lock too, which
        the registry lock the child makes anew in Python does not release."""
        from medh5 import _core, labels
        from medh5.labels import LabelClass, LabelSet

        classes = [LabelClass(i, f"c{i}", f"C{i}") for i in range(1, 20_000)]
        labels.register("test-fork-big", LabelSet("big", classes))
        name = labels.available()[0]
        try:
            assert self._runs_in_children_forked_during(
                lambda: _core.registry_load("test-fork-big"), lambda: labels.load(name)
            )
        finally:
            labels.unregister("test-fork-big")

    def test_the_registry_is_reached_only_from_bindings_that_stay_attached(self):
        """Declined in 2.2.0's review (#76): the engine's registry mutex is
        left out of the fork barrier, because no thread can hold it across a
        fork.  Its bindings run attached (none detaches), so a GIL build's
        fork waits for the GIL and 3.14t's `PyOS_BeforeFork` stops the world
        first; and nothing else in the engine reaches the registry.  A
        binding that detaches, or an engine caller, reopens the finding."""
        from tests.helpers import ROOT

        source = (ROOT / "crates/medh5-python/src/labels.rs").read_text("utf-8")
        bodies = re.findall(
            r"^fn (registry_\w+)[^{]*\{\n(.*?)^\}$", source, re.M | re.S
        )
        assert {name for name, _ in bodies} == {
            "registry_available",
            "registry_load",
            "registry_load_file",
            "registry_register",
            "registry_unregister",
            "registry_describe",
        }
        for name, body in bodies:
            assert "crate::h5(" not in body and ".detach(" not in body, name
        engine = ROOT / "crates/medh5/src"
        callers = [
            path.relative_to(engine).as_posix()
            for path in engine.rglob("*.rs")
            if path.name != "registry.rs"
            and re.search(
                r"\bregistry::(?:load|register|unregister|describe|available)",
                path.read_text("utf-8"),
            )
        ]
        assert callers == []

    @pytest.mark.parametrize("threads", [8])
    def test_threads_naming_outputs_from_one_set_get_distinct_names(self, threads: int):
        """Threads naming groups of one subject into one shared `used` set:
        every name returned is distinct, and the set holds them all, as a
        name is added only if no other thread took it meanwhile."""
        from medh5.io.grouping import Occasion, SubjectGroup, output_name

        group = SubjectGroup("p1", [Occasion("s1", "p1", "20260101")])
        used: set[str] = set()
        names: list[str] = []
        rounds = 100

        def name() -> None:
            names.append(output_name(group, used))

        assert hammered([name] * threads, rounds=rounds) == 0
        assert len(set(names)) == len(names) == threads * rounds
        assert used == set(names)

    @pytest.mark.parametrize("threads", [8])
    def test_one_sample_read_from_eight_threads(
        self, tmp_path: Path, label_set: Any, threads: int
    ):
        """Every thread reads what one thread alone reads: the image whole and
        by region, every annotation's planes, and the digests verified."""
        from tests.helpers import SHAPE, block, write_sample

        path = write_sample(
            tmp_path / "s.medh5",
            label_set=label_set,
            masks={1: block(SHAPE, (2, 2, 2)), 2: block(SHAPE, (1, 1, 1))},
            index=True,
        )
        with medh5.open(path) as s:
            image = s.images["CT_tp0"]
            names = list(s.annotations)
            regions = [
                tuple(slice(i, i + 4) for _ in SHAPE) for i in range(min(SHAPE) - 4)
            ]
            whole = image.read()
            planes = {n: s.annotations[n].dense() for n in names}
            barrier = threading.Barrier(threads)
            failed: list[BaseException] = []

            def read(k: int) -> None:
                barrier.wait()
                try:
                    for i in range(50):
                        roi = regions[(k + i) % len(regions)]
                        assert np.array_equal(image.read(roi), whole[roi])
                        name = names[(k + i) % len(names)]
                        assert np.array_equal(s.annotations[name].dense(), planes[name])
                        if i % 10 == k % 10:
                            assert np.array_equal(s.images["CT_tp0"].read(), whole)
                            assert not s.verify().mismatched
                except BaseException as e:  # pragma: no cover - the failure
                    failed.append(e)

            workers = [threading.Thread(target=read, args=(k,)) for k in range(threads)]
            for t in workers:
                t.start()
            for t in workers:
                t.join()
            assert not failed, failed[:3]

    @pytest.mark.parametrize("threads", [8])
    def test_registries_are_safe_across_threads(self, label_set: Any, threads: int):
        """Threads registering, loading and dropping vocabularies meet only
        what one of them registered under a name, or no name at all."""
        from medh5 import labels

        barrier = threading.Barrier(threads)
        failed: list[BaseException] = []

        def churn(k: int) -> None:
            name = f"org.example.threads-{k % 2}"
            barrier.wait()
            try:
                for _ in range(200):
                    labels.register(name, label_set)
                    try:
                        assert labels.load(name) == label_set
                    except medh5.MEDH5ValidationError as e:
                        # Another thread dropped it between the two (E305).
                        assert "E305" in str(e), e
                    assert labels.load("binary-foreground") is not None
                    labels.unregister(name)
            except BaseException as e:  # pragma: no cover - the failure
                failed.append(e)

        workers = [threading.Thread(target=churn, args=(k,)) for k in range(threads)]
        for t in workers:
            t.start()
        for t in workers:
            t.join()
        assert not failed, failed[:3]
        assert not any(n.startswith("org.example.threads-") for n in labels.available())

    def test_registering_runs_no_callers_code_under_the_registry_lock(self):
        """A vocabulary is read as it stands before the registry's lock is
        taken: a value in a class's `properties` whose conversion loads a
        vocabulary itself (the caller's code) would otherwise wait for the
        lock its own thread holds."""
        from medh5 import labels
        from medh5.labels import LabelClass, LabelSet

        class Loads:
            """A mapping that loads a vocabulary as it is read."""

            def keys(self) -> list[str]:
                labels.load("binary-foreground")
                return ["k"]

            def __getitem__(self, key: str) -> int:
                return 1

        ls = LabelSet("v", [LabelClass(1, "a", "A")])
        ls[1].properties["x"] = Loads()
        done = threading.Thread(
            target=labels.register, args=("org.example.loads", ls), daemon=True
        )
        done.start()
        done.join(timeout=10)
        assert not done.is_alive(), "register waited for its own lock"
        labels.unregister("org.example.loads")
