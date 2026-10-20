"""The storage layer: chunking, codecs, windowed reads, sampling indices and
recompression (spec §14)."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import medh5
from medh5.errors import MEDH5FileError, MEDH5ValidationError
from medh5.storage import (
    COMPRESS_MIN_BYTES,
    MAX_CHUNK_BYTES,
    PROFILES,
    build_index,
    chunk_report,
    dataset_layout,
    is_bulk,
    optimize_chunks,
    read_indices,
    resolve_profile,
    spatial_chunk_for,
)
from medh5.validate import validate_file
from tests.helpers import write_sample
from tests.kits import Framed


class TestChunking:
    def test_S14_1_chunk_starts_from_the_patch_hint(self):
        chunk = spatial_chunk_for((256, 256, 256), (96, 96, 96), itemsize=2)
        assert all(c >= 96 for c in chunk)

    def test_S14_1_chunk_never_exceeds_the_extent(self):
        assert spatial_chunk_for((10, 12, 14), (96, 96, 96)) == (10, 12, 14)

    def test_S14_1_chunk_stays_within_the_cache_budget(self):
        chunk = spatial_chunk_for((512, 512, 512), (64, 64, 64), itemsize=4)
        assert int(np.prod(chunk)) * 4 <= MAX_CHUNK_BYTES

    def test_S14_1_non_spatial_axes_get_extent_one(self):
        chunk = optimize_chunks(
            (4, 32, 64, 64), ("time", "spatial", "spatial", "spatial"), (16, 16, 16)
        )
        assert chunk[0] == 1

    def test_S14_1_stacked_encodings_read_one_plane(self):
        """§14.1: layers/bitmask/probmap MUST chunk as (1, *spatial_chunk)."""
        chunk = optimize_chunks(
            (5, 32, 64, 64), ("spatial", "spatial", "spatial"), (16, 16, 16), leading=1
        )
        assert chunk[0] == 1
        assert len(chunk) == 4

    def test_S14_1_l3_is_one_cpus_share_of_what_sysfs_reports(self):
        """The probe is the engine's: it reads sysfs, and its one process
        (``sysctl``, for macOS) is compiled in only on macOS.  This test
        patched Python's process entry points, which the engine never calls,
        so that half held nothing.  What is observable is the answer: where
        sysfs reports an L3, the share of one of the CPUs sharing it --- the
        whole of it overstated the budget on every multi-core host --- and the
        answer is positive and stable."""
        from medh5.storage import DEFAULT_L3_BYTES, detect_l3_bytes

        found = detect_l3_bytes()
        assert found > 0
        assert detect_l3_bytes() == found
        index3 = Path("/sys/devices/system/cpu/cpu0/cache/index3")
        try:
            size = (index3 / "size").read_text().strip()
            shared = (index3 / "shared_cpu_list").read_text().strip()
        except OSError:
            if sys.platform.startswith("linux"):
                assert found == DEFAULT_L3_BYTES
            return
        unit = {"K": 1024, "M": 1024 * 1024}.get(size[-1], 1)
        total = int(size.rstrip("KM")) * unit
        sharing = 0
        for part in shared.split(","):
            first, _, last = part.partition("-")
            sharing += int(last or first) - int(first) + 1
        assert found == total // sharing

    def test_S14_1_a_chunk_hint_never_exceeds_the_extent(self, tmp_path: Path):
        """A hint larger than the grid was passed to HDF5 as given, which
        refuses a chunk larger than a fixed-size dataset (OSError)."""
        path = tmp_path / "hinted.medh5"
        shape = (64, 64, 64)
        with medh5.create(path, codec="portable") as w:
            w.add_grid(
                "g", shape=shape, spacing=(1.0, 1.0, 1.0), chunk_hint=(128, 16, 96)
            )
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
        with medh5.open(path) as sample:
            assert sample.images["CT"].chunks == (64, 16, 64)

    def test_S14_1_every_array_on_a_hinted_grid_takes_its_spatial_chunk(
        self, tmp_path: Path
    ):
        """§14.1: labels SHOULD chunk as their grid's images do.  The stacked
        encodings ignored the grid's ``chunk_hint`` and fell back to the
        optimiser, so they were chunked apart from the image they label ---
        by the writer, and again by ``recompress --rechunk``."""
        from medh5.storage import recompress

        path = tmp_path / "hinted.medh5"
        shape = (48, 64, 64)
        masks = {1: np.zeros(shape, bool), 2: np.zeros(shape, bool)}
        masks[1][4:20] = True
        masks[2][10:30, 8:40] = True
        with medh5.create(path, codec="portable") as w:
            w.add_grid(
                "g",
                shape=shape,
                spacing=(1.0, 1.0, 1.0),
                chunk_hint=(16, 32, 128),
                frame_uid="F0",
            )
            w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
            w.label_set(Framed.label_set(2))
            w.add_segmentation("layered", grid="g", masks=masks, encoding="layers")
            w.add_segmentation("bits", grid="g", masks=masks, encoding="bitmask")
            w.add_transform(
                "warp",
                kind="displacement",
                from_frame="F0",
                to_frame="F1",
                field=np.zeros((3, *shape), np.float32),
                field_grid="g",
            )
        stacked = ("annotations/layered/data", "annotations/bits/data")
        stacked += ("transforms/warp/field",)
        for _ in range(2):
            with h5py.File(path) as f:
                assert f["images/CT"].chunks == (16, 32, 64)
                for name in stacked:
                    assert f[name].chunks == (1, 16, 32, 64), name
            assert recompress(path, "portable", rechunk=True).ok

    def test_S13_1_a_new_chunk_shape_changes_no_content_id(self, tmp_path: Path):
        """2.0.1 rounded a 96³ float32 patch up to an 8 MiB 128³ chunk; the
        optimiser now keeps it within 4 MiB.  Digests cover decompressed
        content, so the file verifies under one ``content_id`` chunked either
        way, and re-chunking one written the old way keeps it."""
        from medh5.storage import recompress

        shape = (128, 128, 128)
        now, then = tmp_path / "now.medh5", tmp_path / "then.medh5"
        ct = np.indices(shape).sum(axis=0).astype(np.float32)
        with medh5.create(now, codec="training") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0), patch_hint=(96,) * 3)
            w.add_image("CT", ct, grid="g", modality="CT")
        then.write_bytes(now.read_bytes())
        # Chunk the copy as 2.0.1's optimiser did.
        with h5py.File(then, "r+") as f:
            attrs = dict(f["images/CT"].attrs)
            del f["images/CT"]
            node = f["images"].create_dataset("CT", data=ct, chunks=(128, 128, 128))
            node.attrs.update(attrs)
        chunks, ids = [], []
        for path in (now, then):
            with medh5.open(path) as sample:
                assert sample.verify().ok, path.name
                chunks.append(sample.images["CT"].chunks)
                ids.append(sample.content_id)
        assert int(np.prod(chunks[0])) * 4 <= MAX_CHUNK_BYTES
        assert int(np.prod(chunks[1])) * 4 > MAX_CHUNK_BYTES
        result = recompress(then, "training", rechunk=True)
        assert result.ok and result.content_id_preserved
        assert result.content_id == ids[0] == ids[1]
        with medh5.open(then) as sample:
            assert sample.images["CT"].chunks == chunks[0]

    @pytest.mark.parametrize("hint", [(0, 16, 16), (16, -1, 16)])
    def test_S3_2_a_chunk_hint_extent_is_positive(self, tmp_path: Path, hint):
        """A hint of 0 or -1 was written, though no chunk has an extent below
        1: the writer took it for 1, and the validator passed it."""
        with medh5.create(tmp_path / "hinted.medh5", codec="portable") as w:
            with pytest.raises(MEDH5ValidationError) as caught:
                w.add_grid(
                    "g", shape=(64, 64, 64), spacing=(1.0, 1.0, 1.0), chunk_hint=hint
                )
            assert caught.value.code == "E109"
            w.abort()

    def test_axis_kinds_must_describe_the_shape(self):
        with pytest.raises(MEDH5ValidationError):
            optimize_chunks((4, 4), ("spatial",), (2,))

    def test_patch_length_must_match(self):
        with pytest.raises(MEDH5ValidationError):
            spatial_chunk_for((8, 8, 8), (4, 4))

    def test_degenerate_shape_is_refused(self):
        with pytest.raises(MEDH5ValidationError):
            spatial_chunk_for((0, 8, 8))

    def test_chunk_report(self):
        report = chunk_report((64, 64, 64), (32, 32, 32), 2)
        assert report["n_chunks"] == 8
        assert report["chunk_bytes"] == 32 * 32 * 32 * 2


class TestCodecs:
    def test_every_profile_builds_kwargs(self):
        for name in PROFILES:
            kwargs = dataset_layout(
                (256, 256, 8), np.dtype(np.int16), profile=name, role="image"
            )
            assert "chunks" in kwargs

    def test_small_datasets_stay_contiguous(self):
        assert dataset_layout((4,), np.dtype(np.uint16)) == {}
        big = (COMPRESS_MIN_BYTES // 2 + 16,)
        assert dataset_layout(big, np.dtype(np.uint16)) != {}

    def test_empty_datasets_stay_contiguous(self):
        assert dataset_layout((0, 3), np.dtype(np.float32)) == {}

    def test_unknown_profile_is_refused(self):
        with pytest.raises(MEDH5ValidationError):
            resolve_profile("turbo")

    def test_profile_objects_pass_through(self):
        assert resolve_profile(PROFILES["archive"]) is PROFILES["archive"]
        assert resolve_profile(None).name == "balanced"

    def test_S14_2_a_profile_of_ones_own_is_honoured(self):
        """Only a profile's name reached the engine, so a ``CodecProfile`` of
        one's own --- which the signature takes --- was refused as an unknown
        name."""
        from medh5.storage import Codec, CodecProfile

        mine = CodecProfile(
            "mine",
            image=Codec("gzip:9", gzip_level=9),
            label=Codec("gzip:1", gzip_level=1),
            description="ours",
        )
        big, small = (64, 64, 64), (4,)
        named = dataset_layout(big, np.int16, profile="portable")
        assert dataset_layout(big, np.int16, profile=mine) == {
            **named,
            "codec": "gzip:9",
        }
        assert (
            dataset_layout(big, np.int16, profile=mine, role="label")["codec"]
            == "gzip:1"
        )
        assert dataset_layout(small, np.int16, profile=mine) == {}
        for name, profile in PROFILES.items():
            for role in ("image", "label"):
                found = dataset_layout(big, np.int16, profile=profile, role=role)
                assert found == dataset_layout(big, np.int16, profile=name, role=role)
                assert found["codec"] == profile.codec(role).name

    def test_S14_2_dataset_layout_is_dataset_kwargs_renamed(self):
        """2.1 renames ``dataset_kwargs``, whose result stopped being ``h5py``
        keywords in 2.0; the old name is the same function."""
        import medh5.storage

        with pytest.warns(DeprecationWarning):
            old = medh5.storage.dataset_kwargs
        assert old is dataset_layout
        for shape, dtype in (((4,), np.uint16), ((0, 3), np.float32)):
            assert old(shape, dtype) == dataset_layout(shape, dtype) == {}
        for name in PROFILES:
            for role in ("image", "label"):
                assert old((64, 64, 64), np.int16, profile=name, role=role) == (
                    dataset_layout((64, 64, 64), np.int16, profile=name, role=role)
                )

    def test_S14_2_dataset_kwargs_warns(self):
        import medh5.storage

        with pytest.warns(
            DeprecationWarning, match="^medh5 3.0 removes dataset_kwargs"
        ):
            from medh5.storage import dataset_kwargs  # noqa: F401
        assert "dataset_kwargs" not in medh5.storage.__all__
        assert "dataset_layout" in medh5.storage.__all__

    def test_S14_2_portable_needs_no_plugin(self, tmp_path):
        """A `portable` file must open with stock h5py filters only."""
        import h5py

        import medh5

        shape = (64, 64, 64)
        path = tmp_path / "p.medh5"
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", np.zeros(shape, dtype=np.int16), grid="g", modality="CT")
        with h5py.File(path) as handle:
            assert handle["images/CT"].compression == "gzip"
            assert np.asarray(handle["images/CT"][0, 0, :4]).size == 4

    def test_S14_2_every_plugin_filter_is_named(self, tmp_path: Path):
        """``describe_filters`` named gzip, lzf, szip and Blosc, and called
        anything else "chunked" --- Zstandard, which the engine reads, among
        them --- so ``medh5 info`` hid that a reader needs a plugin."""
        hdf5plugin = pytest.importorskip("hdf5plugin")
        from medh5.storage import describe_filters

        path = write_sample(tmp_path / "filters.medh5")
        values = np.arange(4096, dtype="<i4")
        with h5py.File(path, "r+") as f:
            f.create_dataset(
                "x_other/zstd", data=values, chunks=(1024,), **hdf5plugin.Zstd(clevel=5)
            )
            f.create_dataset(
                "x_other/lz4", data=values, chunks=(1024,), **hdf5plugin.LZ4()
            )
            f.create_dataset(
                "x_other/gzip", data=values, chunks=(1024,), compression="gzip"
            )
        with medh5.open(path) as sample:
            found = {
                name: describe_filters(sample.root[f"x_other/{name}"])
                for name in ("zstd", "lz4", "gzip")
            }
        assert found == {"zstd": "zstd:5", "lz4": "filter:32004", "gzip": "gzip:4"}

    def test_is_bulk(self, sample_path):
        with medh5.open(sample_path) as sample:
            stored = sample.root["annotations/organs_tp0/layer_class_ids"]
            assert not is_bulk(stored)


class TestSamplingIndex:
    def test_S14_3_index_answers_foreground_sampling(self, longitudinal_path, masks):
        with medh5.open(longitudinal_path) as sample:
            index = sample.index["organs_tp0"]
            assert index.voxel_counts[1] == int(masks[1].sum())
            centres = index.sample_foreground(1, n=8, rng=np.random.default_rng(0))
            assert centres.shape == (8, 3)
            for centre in centres:
                assert masks[1][tuple(centre)]

    def test_S14_3_coordinates_are_capped(self, tmp_path, label_set):
        big = {1: np.ones((16, 24, 24), dtype=bool)}

        path = write_sample(
            tmp_path / "cap.medh5", label_set=label_set, masks=big, index=True
        )
        with medh5.open(path) as sample:
            index = sample.index["organs_tp0"]
            assert index.coords(1).shape[0] == 64
            assert index.voxel_counts[1] == 16 * 24 * 24

    def test_S14_3_bboxes_are_tight(self, longitudinal_path, masks):
        with medh5.open(longitudinal_path) as sample:
            box = sample.index["organs_tp0"].bbox(1)
            assert box is not None
            from medh5.geometry import box_to_slices

            assert np.array_equal(
                masks[1][box_to_slices(box)],
                masks[1][masks[1].any(axis=(1, 2))][:, masks[1].any(axis=(0, 2))][
                    :, :, masks[1].any(axis=(0, 1))
                ],
            )

    def test_empty_class_has_no_bbox_and_no_samples(self, tmp_path, label_set):

        masks = {1: np.zeros((16, 24, 24), dtype=bool)}
        masks[1][0, 0, 0] = True
        path = write_sample(
            tmp_path / "e.medh5", label_set=label_set, masks=masks, index=True
        )
        with medh5.open(path) as sample:
            index = sample.index["organs_tp0"]
            assert index.voxel_counts[1] == 1
            with pytest.raises(KeyError):
                index.coords(99)

    def test_class_weights(self, longitudinal_path):
        with medh5.open(longitudinal_path) as sample:
            index = sample.index["organs_tp0"]
            weights = index.class_weights()
            assert abs(sum(weights.values()) - 1.0) < 1e-9
            assert set(index.class_weights("uniform").values()) == {1.0}
            with pytest.raises(MEDH5ValidationError):
                index.class_weights("magic")

    def test_occupancy_is_optional(self, sample_path, masks):
        with medh5.open(sample_path) as sample:
            annotation = sample.annotations["organs_tp0"]
            payload = build_index(annotation, occupancy=None, max_coords=16)
            assert payload.occupancy is None
            payload = build_index(annotation, occupancy=8, max_coords=16)
            assert payload.occupancy is not None
            assert payload.occupancy.shape[0] == len(annotation.class_ids)

    def test_summary_and_reading(self, longitudinal_path):
        with medh5.open(longitudinal_path) as sample:
            indices = read_indices(sample.root)
            summary = indices["organs_tp0"].summary()
            assert summary["max_coords"] == 64
            assert summary["source_digest"].startswith("sha256:")

    def test_sampling_an_empty_class_is_an_error(self, tmp_path, label_set):

        masks = {1: np.zeros((16, 24, 24), dtype=bool), 2: np.zeros((16, 24, 24), bool)}
        masks[1][0, 0, 0] = True
        path = write_sample(
            tmp_path / "z.medh5", label_set=label_set, masks=masks, index=True
        )
        with medh5.open(path) as sample, pytest.raises(MEDH5ValidationError):
            sample.index["organs_tp0"].sample_foreground(2, 1)

    def test_S11_3_a_class_found_empty_stays_in_the_contract(self, tmp_path, label_set):
        """A class searched for and not found is `verified absent`, not `unknown`."""

        masks = {1: np.zeros((16, 24, 24), dtype=bool), 2: np.zeros((16, 24, 24), bool)}
        masks[1][0, 0, 0] = True
        path = write_sample(tmp_path / "c.medh5", label_set=label_set, masks=masks)
        with medh5.open(path) as sample:
            seg = sample.annotations["organs_tp0"]
            assert seg.kind == "instances"
            assert set(seg.class_ids) == {1, 2}
            assert seg.is_annotated(2)
            assert not seg.dense([2])[0].any()

    def test_S14_3_the_occupancy_map_equals_the_loop_it_replaced(self):
        from medh5.storage import occupancy as _occupancy

        rng = np.random.default_rng(0)
        for shape in [(9, 7, 5), (16, 16, 16), (33, 17, 8)]:
            mask = rng.random(shape) > 0.9
            coarse = tuple(max(1, -(-n // 8)) for n in shape)
            expected = np.zeros(coarse, dtype=bool)
            for block in np.ndindex(*coarse):
                window = tuple(
                    slice(i * 8, min((i + 1) * 8, n))
                    for i, n in zip(block, shape, strict=True)
                )
                expected[block] = bool(mask[window].any())
            assert np.array_equal(_occupancy(mask, 8), expected), shape

    def test_S14_3_an_occupancy_factor_of_zero_is_refused(self):
        """A factor of 0 divided by zero: a panic, not an error."""
        from medh5.storage import occupancy as _occupancy

        with pytest.raises(ValueError, match="at least 1"):
            _occupancy(np.ones((4, 4, 4), dtype=bool), 0)

    def test_S14_3_the_occupancy_map_is_compressed_one_class_per_chunk(
        self, tmp_path: Path
    ):
        """52 MB of mostly `False` at 200 classes and 512³, stored contiguous."""
        # 40 classes x (64/8, 128/8, 128/8): 80 KiB, over the compression floor.
        path = Framed.many_classes(
            tmp_path / "occ.medh5", classes=40, shape=(64, 128, 128)
        )
        with h5py.File(path, "r") as handle:
            occupancy = handle["index/organs/occupancy"]
            assert occupancy.chunks == (1, *occupancy.shape[1:])
            assert occupancy.compression == "gzip"
            assert occupancy.id.get_storage_size() < occupancy.nbytes / 10
        with medh5.open(path) as s:
            assert s.verify().ok and "organs" in s.fresh_indices


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_S14_2_a_window_reads_what_hdf5plugin_wrote(
    byteorder: str, tmp_path: Path
) -> None:
    """A window decompresses only the Blosc2 blocks it covers, from the stored
    chunk.  What another tool wrote must read the same as HDF5 reads it:
    `hdf5plugin`'s B2ND chunks through that path, a big-endian dataset --- whose
    bytes HDF5 converts on the way out --- through HDF5's."""
    hdf5plugin = pytest.importorskip("hdf5plugin")
    path = tmp_path / "other.medh5"
    w = medh5.create(path, sample_id="o", subject_id="s")
    w.add_grid("g", shape=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    w.add_image("CT", np.zeros((8, 8, 8), np.int16), grid="g", modality="CT")
    w.commit()
    rng = np.random.default_rng(7)
    values = rng.integers(-1000, 1500, (37, 45, 53)).astype(f"{byteorder}i2")
    values[:, :20] = 3  # runs, so blocks compress unevenly
    with h5py.File(path, "r+") as f:
        f.create_dataset(
            "x_other/ct",
            data=values,
            chunks=(16, 16, 32),
            **hdf5plugin.Blosc2(
                cname="zstd", clevel=3, filters=hdf5plugin.Blosc2.SHUFFLE
            ),
        )
    with medh5.open(path) as s:
        ds = s.root["x_other/ct"]
        for _ in range(40):
            lo = [int(rng.integers(0, n)) for n in values.shape]
            hi = [
                int(rng.integers(a + 1, n + 1))
                for a, n in zip(lo, values.shape, strict=True)
            ]
            window = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
            np.testing.assert_array_equal(ds[window], values[window])
        np.testing.assert_array_equal(ds[5, 3:40, 7], values[5, 3:40, 7])


def test_S14_2_blosclz_chunks_read_whole_by_window_and_in_one_dimension(
    tmp_path: Path,
) -> None:
    """The vendored C-Blosc2 is 3.3.5, whose BloscLZ refuses a match length
    that would wrap (the refusal is held by
    `medh5-sys`'s tests).  The controls: what `hdf5plugin` wrote with BloscLZ
    reads whole through HDF5's filter, by window through the block reader, and
    as a plain one-dimensional dataset."""
    hdf5plugin = pytest.importorskip("hdf5plugin")
    path = tmp_path / "blosclz.medh5"
    w = medh5.create(path, sample_id="o", subject_id="s")
    w.add_grid("g", shape=(8, 8, 8), spacing=(1.0, 1.0, 1.0))
    w.add_image("CT", np.zeros((8, 8, 8), np.int16), grid="g", modality="CT")
    w.commit()
    rng = np.random.default_rng(19)
    volume = rng.integers(-1000, 1500, (37, 45, 53)).astype("<i2")
    volume[:, :20] = 3
    line = np.repeat(np.arange(4000, dtype="<i4"), 3)
    options = hdf5plugin.Blosc2(
        cname="blosclz", clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE
    )
    with h5py.File(path, "r+") as f:
        f.create_dataset("x_other/ct", data=volume, chunks=(16, 16, 32), **options)
        f.create_dataset("x_other/line", data=line, chunks=(1000,), **options)
    with medh5.open(path) as s:
        ct, flat = s.root["x_other/ct"], s.root["x_other/line"]
        np.testing.assert_array_equal(ct.read(), volume)
        np.testing.assert_array_equal(ct[3:30, 7:41, 2:50], volume[3:30, 7:41, 2:50])
        np.testing.assert_array_equal(flat.read(), line)
        np.testing.assert_array_equal(flat[1234:5678], line[1234:5678])


class TestFilterValues:
    """A stored filter pipeline is file content: a Blosc2 chunk-size value
    (``cd_values[3]``) smaller than the chunk it describes failed an assertion
    in the vendored filter and aborted the process (SIGABRT at
    ``blosc2_filter.c:534``).  It is the filter's
    ordinary error now, whichever way the dataset is read."""

    # Planted without `hdf5plugin`: with the filter registered, HDF5 runs its
    # `set_local`, which rewrites the value and the defect disappears.
    PLANT = (
        "import sys, h5py\n"
        "with h5py.File(sys.argv[1], 'r+') as f:\n"
        "    src = f['images/CT']\n"
        "    code, flags, values, name = src.id.get_create_plist().get_filter(0)\n"
        "    assert code == 32026, code\n"
        "    values = list(values)\n"
        "    values[3] = 1\n"
        "    dst = f.create_dataset('images/planted', shape=src.shape,"
        " dtype=src.dtype, chunks=src.chunks, compression=code,"
        " compression_opts=tuple(values), allow_unknown_filter=True)\n"
        "    for i in range(src.id.get_num_chunks()):\n"
        "        offset = src.id.get_chunk_info(i).chunk_offset\n"
        "        mask, raw = src.id.read_direct_chunk(offset)\n"
        "        dst.id.write_direct_chunk(offset, raw, mask)\n"
        "    for key in src.attrs:\n"
        "        dst.attrs[key] = src.attrs[key]\n"
        "    del f['images/CT']\n"
        "    f.move('images/planted', 'images/CT')\n"
        "    planted = f['images/CT'].id.get_create_plist().get_filter(0)[2]\n"
        "    assert planted[3] == 1, planted\n"
    )
    READ = (
        "import sys, medh5\n"
        "with medh5.open(sys.argv[1]) as s:\n"
        "    image = s.images['CT']\n"
        "    try:\n"
        "        if sys.argv[2] == 'full':\n"
        "            image.read()\n"
        "        else:\n"
        "            image.read((slice(0, 4), slice(2, 9), slice(1, 5)))\n"
        "    except OSError as exc:\n"
        "        print('refused:', exc)\n"
        "    else:\n"
        "        print('read')\n"
    )

    @staticmethod
    def _run(*argv: str) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "HDF5_PLUGIN_PRELOAD": "::"}
        return subprocess.run(
            [sys.executable, *argv],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

    @pytest.fixture
    def planted(self, tmp_path: Path) -> Path:
        path = tmp_path / "planted.medh5"
        rng = np.random.default_rng(7)
        with medh5.create(path, sample_id="p", codec="balanced") as w:
            w.add_grid("g", shape=(32, 32, 32), spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT",
                rng.integers(0, 1000, (32, 32, 32)).astype(np.int16),
                grid="g",
                modality="CT",
            )
        run = self._run("-c", self.PLANT, str(path))
        assert run.returncode == 0, run.stderr[-2000:]
        return path

    @pytest.mark.parametrize("how", ["full", "window"])
    def test_S14_2_a_chunk_size_the_chunk_exceeds_is_an_error(
        self, planted: Path, how: str
    ) -> None:
        run = self._run("-c", self.READ, str(planted), how)
        assert run.returncode == 0, (run.returncode, run.stderr[-2000:])
        assert run.stdout.startswith("refused:"), run.stdout
        assert "does not match the chunk size" in run.stdout

    def test_S15_the_validator_reports_it(self, planted: Path) -> None:
        run = self._run(
            "-m", "medh5.cli", "validate", "--level", "integrity", str(planted)
        )
        assert run.returncode == 1, (run.returncode, run.stderr[-2000:])
        assert "E001" in run.stdout and "chunk size" in run.stdout


class TestRecompress:
    def test_S14_2_a_type_a_rebuild_would_lose_is_copied_as_stored(
        self, tmp_path: Path
    ) -> None:
        """Recompress rebuilt every bulk dataset from a plain number type:
        an enumeration lost its names and a committed type its identity and
        attributes, while the result reported ``ok``.  Such a dataset is
        copied as stored and reported as kept."""
        from medh5.storage import recompress

        path = write_sample(tmp_path / "typed.medh5", codec="portable")
        rng = np.random.default_rng(12)
        big = rng.integers(0, 3, (48, 64, 64))
        colours = h5py.enum_dtype({"RED": 0, "GREEN": 1, "BLUE": 2}, basetype="u1")
        with h5py.File(path, "r+") as f:
            f["x_types/level"] = np.dtype("<i2")
            f["x_types/level"].attrs["unit"] = "HU"
            level = f["x_types/level"]
            f.create_dataset("x_types/big", data=big.astype("<i2"), dtype=level)
            f.create_dataset(
                "x_types/small", data=np.arange(4, dtype="<i2"), dtype=level
            )
            f.create_dataset("x_types/colours", data=big.astype("u1"), dtype=colours)
            f.create_dataset("x_types/few", data=np.zeros(4, "u1"), dtype=colours)
        with medh5.open(path) as s:
            before = s.content_id
        result = recompress(path, "archive")
        assert result.ok and result.content_id == before
        assert {"/x_types/big", "/x_types/colours"} <= set(result.kept)
        assert result.to_json()["kept"] == result.kept
        with h5py.File(path, "r") as f:
            named = f["x_types/level"]
            assert named.attrs["unit"] == "HU"
            for name in ("big", "small"):
                stored = f[f"x_types/{name}"].id.get_type()
                assert stored.committed(), name
                assert stored == named.id
            for name in ("colours", "few"):
                mapping = h5py.check_enum_dtype(f[f"x_types/{name}"].dtype)
                assert mapping == {"RED": 0, "GREEN": 1, "BLUE": 2}, name
            np.testing.assert_array_equal(f["x_types/big"][...], big)
            np.testing.assert_array_equal(f["x_types/colours"][...], big)
        with medh5.open(path) as s:
            assert s.verify().ok and s.content_id == before

    def test_S14_2_a_portable_recompression_leaves_nothing_needing_a_plugin(
        self, tmp_path: Path
    ):
        """A dataset under the compression threshold was copied as stored, so
        one another writer had compressed with Blosc2 or Zstandard still
        needed the plugin in a file recompressed to be portable."""
        hdf5plugin = pytest.importorskip("hdf5plugin")
        from medh5.storage import profile_family, recompress

        path = tmp_path / "small.medh5"
        image = np.arange(8 * 8 * 8, dtype=np.int16).reshape(8, 8, 8)
        with medh5.create(path, codec="portable") as w:
            w.add_grid("g", shape=image.shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", image, grid="g", modality="CT")
        # Another writer's small datasets: chunked, under a plugin's filter.
        with h5py.File(path, "r+") as f:
            attrs = dict(f["images/CT"].attrs)
            del f["images/CT"]
            f.create_dataset(
                "images/CT", data=image, chunks=(4, 8, 8), **hdf5plugin.Blosc2()
            )
            f["images/CT"].attrs.update(attrs)
            f.create_dataset(
                "x_other/small", data=np.arange(64.0), chunks=(16,), **hdf5plugin.Zstd()
            )
        assert profile_family(path) == "balanced"
        with medh5.open(path) as sample:
            before = sample.content_id
        kept = recompress(path, "balanced", out=tmp_path / "kept.medh5")
        assert kept.ok and kept.changed == []
        assert profile_family(tmp_path / "kept.medh5") == "balanced"
        result = recompress(path, "portable")
        assert result.ok and result.content_id == before
        assert {(p, after) for p, _, after in result.changed} == {
            ("/images/CT", "contiguous"),
            ("/x_other/small", "contiguous"),
        }
        assert profile_family(path) == "portable"
        with medh5.open(path) as sample:
            np.testing.assert_array_equal(sample.images["CT"].read(), image)
            np.testing.assert_array_equal(
                sample.root["x_other/small"].read(), np.arange(64.0)
            )
            assert sample.verify().ok

    def test_S2_2_a_shard_is_recompressed_member_by_member(self, tmp_path: Path):
        """Every member of a collection is re-encoded, and keeps its address."""
        from medh5.collection import open_collection, pack
        from medh5.storage import recompress

        shape = (32, 48, 48)
        sources = []
        for i in range(2):
            path = tmp_path / f"m{i}.medh5"
            image = np.random.default_rng(i).integers(-1000, 1000, shape)
            with medh5.create(path, sample_id=f"m{i}", codec="portable") as w:
                w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
                w.add_image("CT", image.astype(np.int16), grid="g", modality="CT")
            sources.append(path)
        shard = pack(sources, tmp_path / "s.medh5c")
        with open_collection(shard) as collection:
            before = {key: collection[key].content_id for key in collection}
        result = recompress(shard, "archive")
        assert result.ok and result.datasets == 2
        changed = {path for path, _, _ in result.changed}
        assert changed == {"/samples/m0/images/CT", "/samples/m1/images/CT"}
        with open_collection(shard) as collection:
            assert {key: collection[key].content_id for key in collection} == before
            assert all(collection[key].verify().ok for key in collection)

    def test_S14_2_a_member_named_annotations_keeps_the_image_codec(
        self, tmp_path: Path
    ):
        """The codec was chosen by searching the absolute path for
        ``/annotations/``, so every image of a collection member keyed
        ``annotations`` was re-encoded as a label."""
        import shutil

        from medh5.collection import pack
        from medh5.storage import recompress

        shape = (32, 48, 48)
        image = np.random.default_rng(3).integers(-1000, 1000, shape).astype(np.int16)
        source = tmp_path / "m.medh5"
        with medh5.create(source, sample_id="m", codec="portable") as w:
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image("CT", image, grid="g", modality="CT")
        alone = shutil.copy(source, tmp_path / "alone.medh5")
        shard = pack([source], tmp_path / "s.medh5c", keys=["annotations"])
        standalone = {p: c for p, _, c in recompress(alone, "balanced").changed}
        member = {p: c for p, _, c in recompress(shard, "balanced").changed}
        assert member["/samples/annotations/images/CT"] == standalone["/images/CT"]

    def test_S13_1_recompression_preserves_the_content_id(self, tmp_path, label_set):
        from medh5.storage import recompress

        path = tmp_path / "big.medh5"
        shape = (48, 64, 64)
        rng = np.random.default_rng(5)
        mask = np.zeros(shape, dtype=bool)
        mask[4:20, 8:40, 8:40] = True
        with medh5.create(path, codec="training") as w:
            w.label_set(label_set)
            w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
            w.add_image(
                "CT",
                rng.integers(-1000, 1500, shape).astype(np.int16),
                grid="g",
                modality="CT",
            )
            w.add_segmentation("organs", grid="g", masks={1: mask})
        with medh5.open(path) as sample:
            before = sample.content_id
            values = sample.images["CT"].read()

        result = recompress(path, "archive")
        assert result.content_id_preserved
        assert result.content_id == before
        assert result.datasets >= 1
        assert any("zstd" in after for _, _, after in result.changed)

        with medh5.open(path) as sample:
            assert sample.content_id == before
            assert np.array_equal(sample.images["CT"].read(), values)
            assert np.array_equal(sample.annotations["organs"].dense([1])[0], mask)
            assert sample.verify().ok

    def test_out_writes_beside_the_source(self, tmp_path, indexed_cohort):
        from medh5.storage import recompress

        target = tmp_path / "copy.medh5"
        result = recompress(indexed_cohort[0], "portable", out=target)
        assert target.exists()
        assert Path(indexed_cohort[0]).exists()
        assert result.path == str(target)
        assert "portable" in str(result)

    def test_an_unknown_profile_is_refused(self, indexed_cohort):
        from medh5.storage import recompress

        with pytest.raises(MEDH5ValidationError, match="unknown codec profile"):
            recompress(indexed_cohort[0], "maximum-effort")

    def test_recompress_paths_and_json(self, indexed_cohort):
        from medh5.storage import recompress_paths

        results = recompress_paths(indexed_cohort[:2], "portable")
        assert len(results) == 2
        assert set(results[0].to_json()) >= {"path", "profile", "ratio", "content_id"}

    def test_S14_4_recompress_paths_attempts_every_file(self, tmp_path: Path):
        """A file that could not be recompressed stopped the batch: the files
        after it were never tried, and the results of those before it ---
        already replaced --- were lost with the error.  Every file is tried,
        and the error names each failure and counts the rest, as
        ``fix_paths`` does."""
        from medh5.storage import profile_family, recompress_paths

        shape = (64, 64, 64)
        paths = []
        for i in range(2):
            path = tmp_path / f"m{i}.medh5"
            with medh5.create(path, sample_id=f"m{i}", codec="portable") as w:
                w.add_grid("g", shape=shape, spacing=(1.0, 1.0, 1.0))
                w.add_image("CT", np.ones(shape, np.int16), grid="g", modality="CT")
            paths.append(path)
        broken = tmp_path / "broken.medh5"
        broken.write_bytes(b"not an HDF5 file")
        with pytest.raises(MEDH5FileError) as exc:
            recompress_paths([paths[0], broken, paths[1]], "balanced")
        message = str(exc.value)
        assert "1 of 3 files could not be recompressed, and 2 were" in message
        assert "broken.medh5" in message
        assert [profile_family(p) for p in paths] == ["balanced", "balanced"]
        # One file fails as `recompress` does, and a bad profile before any.
        with pytest.raises(MEDH5FileError) as exc:
            recompress_paths([broken], "portable")
        assert "could not be recompressed" not in str(exc.value)
        with pytest.raises(MEDH5ValidationError, match="unknown codec profile"):
            recompress_paths([paths[0], broken], "turbo")
        assert [profile_family(p) for p in paths] == ["balanced", "balanced"]

    def test_S14_1_rechunk_chunks_the_way_the_writer_does(self, tmp_path: Path):
        """A `layers` dataset went to h5py's (2, 16, 24, 48), and drew W902."""
        from medh5.storage import fit_chunks, grid_chunks, recompress

        shape = (32, 48, 48)
        rng = np.random.default_rng(0)
        masks = {}
        for c in range(1, 5):
            mask = np.zeros(shape, bool)
            mask[int(rng.integers(0, 16)) :, int(rng.integers(0, 24)) :, :] = True
            masks[c] = mask
        path = tmp_path / "rc.medh5"
        w = medh5.create(path, sample_id="s", subject_id="s", codec="portable")
        w.add_grid("g", shape=shape, spacing=(1, 1, 1), patch_hint=(16, 16, 16))
        w.add_image(
            "CT", rng.integers(0, 1000, shape).astype(np.int16), grid="g", modality="CT"
        )
        w.label_set(Framed.label_set(4))
        w.add_segmentation("organs", grid="g", masks=masks, encoding="layers")
        w.commit()
        # Re-chunk it the way some other tool might have, then ask for the rule.
        with h5py.File(path, "r+") as handle:
            data = handle["annotations/organs/data"][...]
            attrs = dict(handle["annotations/organs/data"].attrs)
            del handle["annotations/organs/data"]
            node = handle["annotations/organs"].create_dataset(
                "data", data=data, chunks=(2, 8, 12, 24), compression="gzip"
            )
            node.attrs.update(attrs)
        result = recompress(path, "portable", rechunk=True)
        assert result.ok
        with medh5.open(path) as s:
            grid = s.grids["g"]
        with h5py.File(path, "r") as handle:
            data = handle["annotations/organs/data"]
            assert data.chunks == fit_chunks(
                grid_chunks(grid, data.dtype.itemsize, leading=1), data.shape
            )
            assert data.chunks[0] == 1
            image = handle["images/CT"]
            assert image.chunks == grid_chunks(grid, image.dtype.itemsize)
        assert "W902" not in validate_file(path, level="strict").codes

    def test_S14_1_rechunk_finds_each_sample_root_in_a_collection(self, tmp_path: Path):
        from medh5.storage import recompress

        shape = (32, 64, 64)  # 256 KiB of layers: chunked
        paths = []
        for i in range(2):
            path = tmp_path / f"m{i}.medh5"
            first = np.zeros(shape, bool)
            first[2:20] = True
            second = np.zeros(shape, bool)
            second[10:28] = True
            w = medh5.create(
                path, sample_id=f"m{i}", subject_id=f"p{i}", codec="portable"
            )
            w.add_grid("g", shape=shape, spacing=(1, 1, 1), patch_hint=(8, 8, 8))
            w.add_image("CT", np.ones(shape, np.int16), grid="g", modality="CT")
            w.label_set(Framed.label_set(2))
            w.add_segmentation(
                "seg", grid="g", masks={1: first, 2: second}, encoding="layers"
            )
            w.commit()
            paths.append(path)
        shard = tmp_path / "shard.medh5c"
        medh5.pack(paths, shard)
        # Verifying the output read the collection root as a sample, and raised
        # a KeyError for its `/meta` after the file had been replaced.
        result = recompress(shard, "balanced", rechunk=True)
        assert result.ok and result.verified and result.content_id_preserved
        with h5py.File(shard, "r") as handle:
            for key in ("m0", "m1"):
                data = handle[f"samples/{key}/annotations/seg/data"]
                assert data.chunks is not None and data.chunks[0] == 1
        with medh5.open_collection(shard) as collection:
            assert all(collection[key].verify().ok for key in collection)


class TestReferences:
    """A file holding HDF5 references is not rewritten.

    A reference is an address in its own file.  Amend, recompress, pack and
    repack write a new file, where a copied object reference pointed wherever
    its address landed --- and a no-op amend nulled the ones in an extension
    group --- with every digest still verifying."""

    @staticmethod
    def _with_reference(path: Path, where: str) -> Path:
        write_sample(path)
        with h5py.File(path, "r+") as f:
            target = f["images"].ref
            if where == "attribute":
                f["images"].attrs["x_ref"] = target
            elif where == "dataset":
                group = f.require_group("x_ext")
                group.create_dataset("refs", data=[target], dtype=h5py.ref_dtype)
            elif where == "compound":
                compound = np.dtype([("id", "<i4"), ("ref", h5py.ref_dtype)])
                group = f.require_group("x_ext")
                group.create_dataset(
                    "rows", data=np.array([(1, target)], dtype=compound)
                )
            elif where == "named_type":
                # A committed datatype carries attributes too: the walk of
                # groups and datasets passed it by.
                group = f.require_group("x_ext")
                group["t"] = np.dtype("<f4")
                group["t"].attrs["x_ref"] = target
            elif where in ("unlinked_type", "unlinked_attribute_type"):
                # A committed datatype whose name was removed lives on as the
                # type of what uses it, attributes and all: the walk of named
                # types passed it by, and a rewrite nulled the reference.
                group = f.require_group("x_ext")
                group["t"] = np.dtype("<u4")
                group["t"].attrs["x_ref"] = target
                if where == "unlinked_type":
                    group.create_dataset("d", data=[1, 2, 3], dtype=group["t"])
                else:
                    group.attrs.create("a", data=[7], dtype=group["t"])
                del group["t"]
            else:
                compound = np.dtype([("id", "<i4"), ("ref", h5py.ref_dtype)])
                f["x_type"] = np.dtype("<i2")
                f["x_type"].attrs.create(
                    "rows", data=np.array([(1, target)], dtype=compound)
                )
        return path

    @pytest.mark.parametrize(
        "where",
        [
            "attribute",
            "dataset",
            "compound",
            "named_type",
            "root_named_type",
            "unlinked_type",
            "unlinked_attribute_type",
        ],
    )
    def test_S16_a_file_holding_references_is_not_rewritten(self, tmp_path, where):
        from medh5.collection import pack
        from medh5.storage import recompress

        path = self._with_reference(tmp_path / "ref.medh5", where)
        before = path.read_bytes()
        attempts = [
            lambda: medh5.amend(path).__enter__(),
            lambda: recompress(path, profile="portable"),
            lambda: pack([path], tmp_path / "shard.medh5c"),
        ]
        for attempt in attempts:
            with pytest.raises(MEDH5FileError, match="holds HDF5 references"):
                attempt()
        assert path.read_bytes() == before, "the source was not touched"
        assert not (tmp_path / "shard.medh5c").exists()
        # Reading it is unaffected: the file is still a valid sample.
        with medh5.open(path) as sample:
            assert sample.images

    def test_S16_an_unlinked_type_without_references_is_rewritten(self, tmp_path):
        """The control: a committed type with no reference anywhere in it is
        another tool's content like any other, and every rewrite carries it."""
        from medh5.collection import pack
        from medh5.storage import recompress

        path = write_sample(tmp_path / "plain.medh5")
        with h5py.File(path, "r+") as f:
            group = f.require_group("x_ext")
            group["t"] = np.dtype("<u4")
            group["t"].attrs["note"] = "kept"
            group.create_dataset("d", data=[1, 2, 3], dtype=group["t"])
            del group["t"]
        with medh5.amend(path):
            pass
        assert recompress(path, profile="portable").verified
        pack([path], tmp_path / "shard.medh5c")
        with h5py.File(path, "r") as f:
            assert f["x_ext/d"][()].tolist() == [1, 2, 3]
            kept = h5py.Datatype(f["x_ext/d"].id.get_type())
            assert kept.attrs["note"] == "kept"


class TestLinkGraphs:
    """A file's links form a graph, which every tool walks once per object and
    `recompress` copies as a graph.

    The 2.0 walkers followed paths: a hard link back to an ancestor made
    validation and `verify` run without end, and `recompress` --- which also
    followed soft links --- recursed until the stack ran out (SIGSEGV) on a
    13 KiB file.  1.x walked each object once (`H5Ovisit`), so these are also
    the 1.x answers.  Each tool runs in a child process with a time limit: a
    regression hangs or crashes rather than failing.
    """

    @staticmethod
    def _linked(tmp_path: Path, kind: str) -> Path:
        from tests.kits import Numbered

        path = Numbered.plain(tmp_path / f"{kind}.medh5")
        with h5py.File(path, "r+") as handle:
            ext = handle.create_group("x_ext")
            ext.create_dataset("d", data=np.arange(4))
            if kind == "hard":
                ext["loop"] = ext  # to itself
                ext["root"] = handle  # to an ancestor
            elif kind == "soft":
                ext["loop"] = h5py.SoftLink("/x_ext")
            else:
                ext["alias"] = ext["d"]
        return path

    @pytest.mark.parametrize("kind", ["hard", "soft", "alias"])
    def test_every_tool_finishes_on_a_linked_graph(self, tmp_path: Path, kind):
        import subprocess
        import sys

        path = self._linked(tmp_path, kind)
        out = tmp_path / "out.medh5"
        code = (
            "import medh5\n"
            "from medh5.curation import scrub\n"
            "from medh5.storage import recompress\n"
            "from medh5.validate import validate_file\n"
            f"path = {str(path)!r}\n"
            "assert not validate_file(path, level='integrity').errors\n"
            "with medh5.open(path) as sample:\n"
            "    assert sample.verify().ok\n"
            f"assert recompress(path, 'portable', out={str(out)!r}).verified\n"
            "scrub.scan(path)\n"
            "print('finished')\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert done.returncode == 0, done.stderr[-2000:]
        assert done.stdout.strip().endswith("finished")

    def test_S16_recompress_keeps_soft_links_aliases_and_cycles(self, tmp_path: Path):
        from medh5.storage import recompress

        path = self._linked(tmp_path, "alias")
        with h5py.File(path, "r+") as handle:
            ext = handle["x_ext"]
            ext["soft"] = h5py.SoftLink("/x_ext/d")
            ext["loop"] = ext
        with medh5.open(path) as sample:
            before = sample.content_id
        out = tmp_path / "out.medh5"
        result = recompress(path, "portable", out=out)
        assert result.verified and result.content_id == before
        with h5py.File(out, "r") as handle:
            ext = handle["x_ext"]
            link = ext.get("soft", getlink=True)
            assert isinstance(link, h5py.SoftLink) and link.path == "/x_ext/d"
            assert ext["alias"] == ext["d"]  # one object, two names
            assert ext["loop"] == ext  # the cycle, as stored


def _report(path: Path, text: str) -> None:
    from medh5 import cli

    cli._write_report(path, text)


def _files_write_text(path: Path, text: str) -> None:
    from medh5._files import write_text

    write_text(path, text)


def _write_atomic(path: Path, text: str) -> None:
    medh5._core.write_atomic(path, text)


def _dicom_object(path: Path, text: str) -> None:
    """A SEG or an RTSTRUCT export's save, with a dataset that writes *text*."""
    from medh5.io._common import save_dicom

    class Written:
        def save_as(self, filename: str, **options: Any) -> None:
            Path(filename).write_bytes(text.encode())

    save_dicom(Written(), path, enforce_file_format=True)


# Every Python site that writes a file through the engine's replace so far,
# with the suffix its file takes.  Each owner that moves a site
# (tests/project/test_layers.py) adds it here.  nnU-Net's `dataset.json` is
# `files.write_text`'s.  A NIfTI export and its sidecars are the engine's
# own writes since 2.2.0 (its staged file and `replace_text`), so no longer
# Python sites.
_SITES = {
    "write_atomic": (_write_atomic, ".json"),
    "files.write_text": (_files_write_text, ".json"),
    "convert --report": (_report, ".json"),
    "dicom seg, rtstruct": (_dicom_object, ".dcm"),
}


def _another_group() -> int | None:
    """A group this process may give a file besides the one its new files
    get: a supplementary group, or, as root, any other."""
    egid = os.getegid()
    others = [g for g in os.getgroups() if g != egid]
    if others:
        return others[0]
    return (65533 if egid == 65534 else 65534) if os.geteuid() == 0 else None


@pytest.mark.parametrize(("write", "suffix"), _SITES.values(), ids=list(_SITES))
class TestOneReplace:
    """Every file medh5 writes from Python is replaced as an amendment
    replaces a sample (§14.4, D23): at 2.0.2 a report replaced a link, lost
    the file's mode, and wrote CRLF on Windows."""

    def test_S14_4_a_link_is_kept_and_the_file_it_names_replaced(
        self, tmp_path: Path, write: Any, suffix: str
    ):
        real = tmp_path / "data" / f"real{suffix}"
        real.parent.mkdir()
        real.write_bytes(b"before\n")
        link = tmp_path / f"given{suffix}"
        try:
            link.symlink_to(Path("data", real.name))
        except OSError as exc:  # pragma: no cover - no links on this file system
            pytest.skip(str(exc))
        write(link, "after\n")
        assert link.is_symlink() and os.readlink(link) == str(Path("data", real.name))
        assert real.read_bytes() == b"after\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["data", link.name]
        assert [p.name for p in real.parent.iterdir()] == [real.name]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
    def test_S14_4_the_file_s_mode_is_kept(
        self, tmp_path: Path, write: Any, suffix: str
    ):
        target = tmp_path / f"private{suffix}"
        target.write_bytes(b"before\n")
        target.chmod(0o600)
        write(target, "after\n")
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert target.read_bytes() == b"after\n"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX groups")
    def test_S14_4_the_file_s_group_is_kept(
        self, tmp_path: Path, write: Any, suffix: str
    ):
        """The replacement belonged to the writer's primary group, so a file
        of a shared project group left it."""
        group = _another_group()
        if group is None:  # pragma: no cover - GitHub's runner user has one
            pytest.skip("no group but the primary one")
        target = tmp_path / f"shared{suffix}"
        target.write_bytes(b"before\n")
        os.chown(target, -1, group)
        target.chmod(0o640)
        write(target, "after\n")
        assert (target.stat().st_gid, stat.S_IMODE(target.stat().st_mode)) == (
            group,
            0o640,
        )

    def test_S14_4_a_read_only_file_is_replaced_and_stays_read_only(
        self, tmp_path: Path, write: Any, suffix: str
    ):
        """Windows refused both: `os.replace` over a read-only file."""
        target = tmp_path / f"read-only{suffix}"
        target.write_bytes(b"before\n")
        target.chmod(stat.S_IREAD)
        try:
            write(target, "after\n")
            assert not target.stat().st_mode & stat.S_IWRITE
            assert target.read_bytes() == b"after\n"
        finally:
            target.chmod(stat.S_IREAD | stat.S_IWRITE)
        assert [p.name for p in tmp_path.iterdir()] == [target.name]

    def test_S14_4_newlines_are_written_as_given(
        self, tmp_path: Path, write: Any, suffix: str
    ):
        target = tmp_path / f"lines{suffix}"
        write(target, "a\nb\r\nc\n")
        assert target.read_bytes() == b"a\nb\r\nc\n"


class TestStagedFile:
    """A foreign writer's file (nibabel, pydicom) through the same replace."""

    def test_S14_4_write_atomic_and_the_engine_s_own_saves_are_one(
        self, tmp_path: Path
    ):
        """The Python door and the engine's own writer (a cohort manifest)
        write the same bytes with the same mode."""
        from medh5.dataset.manifest import Manifest

        engine, python = tmp_path / "engine.json", tmp_path / "python.json"
        Manifest(entries=[]).save(engine)
        medh5._core.write_atomic(python, engine.read_text(encoding="utf-8"))
        assert python.read_bytes() == engine.read_bytes()
        assert python.stat().st_mode == engine.stat().st_mode
        if sys.platform != "win32":
            for target in (engine, python):
                target.chmod(0o600)
            Manifest(entries=[]).save(engine)
            medh5._core.write_atomic(python, engine.read_bytes())
            assert stat.S_IMODE(engine.stat().st_mode) == 0o600
            assert stat.S_IMODE(python.stat().st_mode) == 0o600

    def test_S14_4_a_staged_file_keeps_the_suffix_and_replaces_on_exit(
        self, tmp_path: Path
    ):
        target = tmp_path / "image.nii.gz"
        target.write_bytes(b"before")
        with medh5._core.stage_file(target) as staged:
            assert staged.path.parent == tmp_path
            assert staged.path.name.endswith("-image.nii.gz")
            assert staged.target == target.resolve()
            staged.path.write_bytes(b"after")
            assert target.read_bytes() == b"before"
        assert target.read_bytes() == b"after"
        assert [p.name for p in tmp_path.iterdir()] == ["image.nii.gz"]

    def test_S14_4_a_staged_file_is_discarded_on_an_exception(self, tmp_path: Path):
        target = tmp_path / "image.nii.gz"
        target.write_bytes(b"before")
        with pytest.raises(KeyError), medh5._core.stage_file(target) as staged:
            staged.path.write_bytes(b"half")
            raise KeyError("the writer failed")
        assert target.read_bytes() == b"before"
        assert [p.name for p in tmp_path.iterdir()] == ["image.nii.gz"]

    def test_S14_4_a_staged_file_commits_or_aborts_once(self, tmp_path: Path):
        committed = medh5._core.stage_file(tmp_path / "a.txt")
        committed.path.write_bytes(b"a")
        committed.commit()
        committed.abort()  # nothing left to discard
        with pytest.raises(MEDH5FileError, match="already"):
            committed.commit()
        assert (tmp_path / "a.txt").read_bytes() == b"a"

        aborted = medh5._core.stage_file(tmp_path / "b.txt")
        aborted.path.write_bytes(b"b")
        aborted.abort()
        with pytest.raises(MEDH5FileError, match="already"):
            aborted.commit()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["a.txt"]

    def test_S14_4_write_atomic_writes_text_or_bytes(self, tmp_path: Path):
        medh5._core.write_atomic(tmp_path / "t", "é\n")
        medh5._core.write_atomic(tmp_path / "b", b"\x00\xff")
        assert (tmp_path / "t").read_bytes() == "é\n".encode()
        assert (tmp_path / "b").read_bytes() == b"\x00\xff"
        with pytest.raises(TypeError, match="str or bytes"):
            medh5._core.write_atomic(tmp_path / "n", 3)  # type: ignore[arg-type]

    def test_S14_4_an_error_names_the_file_not_its_temporary(self, tmp_path: Path):
        missing = tmp_path / "no-such-directory" / "r.json"
        with pytest.raises(OSError, match="no-such-directory") as caught:
            medh5._core.write_atomic(missing, "x")
        assert ".tmp" not in str(caught.value)

    def test_S14_4_same_file_follows_links(self, tmp_path: Path):
        a = tmp_path / "a"
        a.write_bytes(b"")
        (tmp_path / "b").write_bytes(b"")
        assert medh5._core.same_file(a, a)
        assert not medh5._core.same_file(a, tmp_path / "b")
        try:
            (tmp_path / "link").symlink_to(a)
        except OSError:  # pragma: no cover - no links on this file system
            return
        assert medh5._core.same_file(tmp_path / "link", a)
