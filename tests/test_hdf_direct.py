#!/usr/bin/env python
"""Tests for the direct-chunk HDF5 reader and its use by HdfDataset.

The reference everywhere is what h5py returns for the same file: the direct
path must hand out exactly the frames, values and dtype of the h5py path.
"""
import os
import shutil
import struct
import tempfile
import threading
import unittest

import h5py
import hdf5plugin
import numpy as np
import torch

from boost_corr.xpcs_aps_8idi.dataset.hdf_direct import cast_like_hdf5
from boost_corr.xpcs_aps_8idi.dataset.hdf_handler import HdfDataset
from boost_corr.xpcs_aps_8idi.dataset.utils import create_dataset

DATA_PATH = "/entry/data/data"
FRAME_SHAPE = (24, 31)
FRAME_NUM = 37


def make_frames(dtype, frame_num=FRAME_NUM, shape=FRAME_SHAPE):
    rng = np.random.default_rng(0)
    data = rng.poisson(0.5, size=(frame_num,) + shape).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        # detector-style markers and counts beyond the signed range, which
        # HDF5 saturates when reading into a signed buffer
        info = np.iinfo(dtype)
        data[:, 0, :3] = info.max
        data[:, 1, 0] = info.max // 2 + 1
        data[:, 1, 1] = info.max // 2
    return data


def bitshuffle(cname="lz4"):
    return dict(chunks=(1,) + FRAME_SHAPE, **hdf5plugin.Bitshuffle(cname=cname))


def open_dataset(fname, direct_read, **kwargs):
    opts = dict(
        device="cpu",
        batch_size=8,
        avg_frame=1,
        begin_frame=0,
        end_frame=-1,
        stride_frame=1,
        mask_crop=None,
    )
    opts.update(kwargs)
    return HdfDataset(fname, direct_read=direct_read, **opts)


class HdfDirectTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fname = os.path.join(self.tmpdir, "raw.h5")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def write(self, data, **kwargs):
        with h5py.File(self.fname, "w") as f:
            f.create_dataset(DATA_PATH, data=data, **kwargs)

    def assert_same_as_h5py(self, **kwargs):
        """Every batch of the direct path equals the h5py path's."""
        direct = open_dataset(self.fname, True, **kwargs)
        legacy = open_dataset(self.fname, False, **kwargs)
        self.assertTrue(direct.direct_read)
        self.assertFalse(legacy.direct_read)
        self.assertEqual(len(direct), len(legacy))
        self.assertGreater(len(direct), 0)
        for n in range(len(legacy)):
            x, y = direct[n], legacy[n]
            self.assertEqual(x.dtype, y.dtype)
            self.assertEqual(x.shape, y.shape)
            self.assertTrue(torch.equal(x.cpu(), y), f"batch {n} differs")
        direct.close()
        return direct


class TestLayouts(HdfDirectTestCase):
    def test_bitshuffle_lz4(self):
        for dtype in (np.uint8, np.uint16, np.uint32, np.int16, np.int32, np.float32):
            with self.subTest(dtype=dtype):
                self.write(make_frames(dtype), **bitshuffle("lz4"))
                self.assert_same_as_h5py()

    def test_bitshuffle_zstd(self):
        self.write(make_frames(np.uint32), **bitshuffle("zstd"))
        self.assert_same_as_h5py()

    def test_uncompressed_chunked(self):
        self.write(make_frames(np.uint16), chunks=(1,) + FRAME_SHAPE)
        self.assert_same_as_h5py()

    def test_contiguous(self):
        self.write(make_frames(np.uint32))
        self.assert_same_as_h5py()

    def test_chunk_stored_without_filter(self):
        # a writer may skip the filter for single chunks (filter mask bit 0)
        data = make_frames(np.uint32)
        self.write(data, **bitshuffle("lz4"))
        with h5py.File(self.fname, "r+") as f:
            f[DATA_PATH].id.write_direct_chunk((5, 0, 0), data[4].tobytes(), filter_mask=1)
        self.assert_same_as_h5py()

    def test_unsupported_layouts_use_h5py(self):
        data = make_frames(np.uint32)
        one_frame = (1,) + FRAME_SHAPE
        layouts = {
            "gzip": dict(chunks=one_frame, compression="gzip"),
            "multi-frame chunks": dict(chunks=(4,) + FRAME_SHAPE, **hdf5plugin.Bitshuffle()),
            "shuffle + bitshuffle": dict(chunks=one_frame, shuffle=True, **hdf5plugin.Bitshuffle()),
            "lz4": dict(chunks=one_frame, **hdf5plugin.LZ4()),
        }
        for name, kwargs in layouts.items():
            with self.subTest(layout=name):
                self.write(data, **kwargs)
                dset, use_loader = create_dataset(
                    self.fname, device="cpu", avg_frame=1, begin_frame=0,
                    end_frame=-1, stride_frame=1, mask_crop=None,
                )
                self.addCleanup(dset.close)
                self.assertFalse(dset.direct_read)
                self.assertTrue(use_loader)
                # uint32 is read as int32, saturated
                expected = np.minimum(data[:8], 2**31 - 1)
                self.assertTrue(np.array_equal(dset[0].numpy().reshape(8, *FRAME_SHAPE), expected))
                dset.close()

    def test_create_dataset_skips_loader_for_direct_read(self):
        self.write(make_frames(np.uint32), **bitshuffle())
        kwargs = dict(device="cpu", avg_frame=1, begin_frame=0, end_frame=-1,
                      stride_frame=1, mask_crop=None)
        dset, use_loader = create_dataset(self.fname, **kwargs)
        self.assertTrue(dset.direct_read)
        self.assertFalse(use_loader)
        dset.close()

        os.environ["BOOST_CORR_HDF_DIRECT"] = "0"
        try:
            dset, use_loader = create_dataset(self.fname, **kwargs)
        finally:
            del os.environ["BOOST_CORR_HDF_DIRECT"]
        self.assertFalse(dset.direct_read)
        self.assertTrue(use_loader)


class TestFrameSelection(HdfDirectTestCase):
    def setUp(self):
        super().setUp()
        self.write(make_frames(np.uint32), **bitshuffle())

    def test_begin_end(self):
        self.assert_same_as_h5py(begin_frame=5, end_frame=30)

    def test_stride(self):
        self.assert_same_as_h5py(stride_frame=3)

    def test_avg(self):
        dset = self.assert_same_as_h5py(avg_frame=2)
        self.assertEqual(dset[0].dtype, torch.float32)
        dset.close()

    def test_stride_and_avg(self):
        self.assert_same_as_h5py(begin_frame=1, stride_frame=2, avg_frame=3, batch_size=4)

    def test_mask_crop(self):
        # a small crop is applied in the decoding threads, a large one on
        # the device; the index need not be sorted
        rng = np.random.default_rng(1)
        pixel_num = FRAME_SHAPE[0] * FRAME_SHAPE[1]
        for keep, on_device in ((0.1, False), (0.7, True)):
            with self.subTest(keep=keep):
                index = rng.permutation(pixel_num)[: int(keep * pixel_num)]
                dset = self.assert_same_as_h5py(mask_crop=torch.tensor(index))
                self.assertEqual(dset[0].shape[1], index.size)
                self.assertEqual(dset.device_crop is not None, on_device)
                dset.close()

    def test_mask_crop_out_of_range(self):
        dset = open_dataset(self.fname, True, mask_crop=torch.tensor([0, 5, 24 * 31]))
        with self.assertRaises(IndexError):
            dset[0]
        dset.close()

    def test_more_threads_than_frames(self):
        self.assert_same_as_h5py(num_threads=64)

    def test_single_thread(self):
        self.assert_same_as_h5py(num_threads=1)


class TestAccessPattern(HdfDirectTestCase):
    def setUp(self):
        super().setUp()
        self.data = make_frames(np.uint16)
        self.write(self.data, **bitshuffle())
        self.dset = open_dataset(self.fname, True)

    def tearDown(self):
        self.dset.close()
        super().tearDown()

    def expected(self, idx):
        return self.data[idx * 8 : (idx + 1) * 8].reshape(-1, self.dset.pixel_num)

    def check(self, idx):
        self.assertTrue(np.array_equal(self.dset[idx].numpy(), self.expected(idx)))

    def test_random_access(self):
        for idx in (3, 0, 0, 4, 1, 2, 3, 4, 2):
            self.check(idx)

    def test_reset_and_reread(self):
        self.check(0)
        self.dset.__reset__()
        for idx in range(len(self.dset)):
            self.check(idx)
        self.dset.__reset__()
        self.check(0)

    def test_batch_is_not_overwritten_by_later_reads(self):
        # staging buffers are recycled; a batch already handed out must not
        # change when the following ones are read
        first = self.dset[0]
        keep = first.clone()
        for idx in range(1, len(self.dset)):
            self.dset[idx]
        self.assertTrue(torch.equal(first, keep))

    def test_out_of_range(self):
        with self.assertRaises(IndexError):
            self.dset[len(self.dset)]

    def test_sparsity(self):
        sparsity = float(self.dset.get_sparsity())
        self.assertAlmostEqual(sparsity, np.mean(self.data[0] > 0), places=6)
        self.check(0)

    def test_close_stops_threads(self):
        before = threading.active_count()
        dset = open_dataset(self.fname, True)
        dset[0]
        self.assertGreater(threading.active_count(), before)
        dset.close()
        self.assertEqual(threading.active_count(), before)


class TestErrors(HdfDirectTestCase):
    def test_corrupt_chunk_raises_in_reader(self):
        self.write(make_frames(np.uint32), **bitshuffle())
        with h5py.File(self.fname, "r") as f:
            offset = f[DATA_PATH].id.get_chunk_info(20).byte_offset
        with open(self.fname, "r+b") as f:
            f.seek(offset)
            f.write(struct.pack(">Q", 12345))  # bogus uncompressed size
        dset = open_dataset(self.fname, True)
        self.assertTrue(dset.direct_read)
        dset[0]
        dset[1]
        for _ in range(2):
            with self.assertRaises(ValueError):
                dset[2]
        # still usable for the intact batches
        dset[0]
        dset[3]
        dset.close()

    def test_first_frame_mismatch_falls_back(self):
        # the first frame is cross-checked against h5py when the dataset is
        # opened; a reader that disagrees must not be used
        self.write(make_frames(np.uint32), **bitshuffle())
        from boost_corr.xpcs_aps_8idi.dataset import hdf_handler

        class WrongDecoder(hdf_handler.FrameDecoder):
            def decode(self, frame, out):
                super().decode(frame, out)
                out[0] += 1

        original = hdf_handler.FrameDecoder
        hdf_handler.FrameDecoder = WrongDecoder
        try:
            with self.assertLogs(hdf_handler.logger, level="WARNING"):
                dset = open_dataset(self.fname, True)
        finally:
            hdf_handler.FrameDecoder = original
        self.assertFalse(dset.direct_read)


class TestCastLikeHdf5(unittest.TestCase):
    """cast_like_hdf5 must agree with HDF5's own type conversion."""

    cases = [
        (np.uint8, np.int16, torch.uint8),
        (np.uint16, np.int16, torch.int16),
        (np.uint16, np.int32, torch.int16),
        (np.uint16, np.float32, torch.int16),
        (np.uint32, np.int32, torch.int32),
        (np.uint32, np.float32, torch.int32),
        (np.int32, np.float32, torch.int32),
    ]

    def test_matches_hdf5_conversion(self):
        tmpdir = tempfile.mkdtemp()
        try:
            for src, dst, staging in self.cases:
                with self.subTest(src=src, dst=dst):
                    info = np.iinfo(src)
                    values = np.array(
                        [0, 1, 7, 255, info.max // 2, info.max // 2 + 1, info.max - 1, info.max],
                        dtype=np.int64,
                    )
                    values = values[values <= info.max].astype(src)
                    fname = os.path.join(tmpdir, f"{np.dtype(src)}_{np.dtype(dst)}.h5")
                    with h5py.File(fname, "w") as f:
                        f["x"] = values
                    expected = np.zeros(values.shape, dtype=dst)
                    with h5py.File(fname, "r") as f:
                        f["x"].read_direct(expected)

                    staged = torch.from_numpy(values.view(np.dtype(src).str.replace("u", "i"))
                                              if src != np.uint8 else values)
                    self.assertEqual(staged.dtype, staging)
                    kind = np.dtype(src).str[1:]
                    result = cast_like_hdf5(staged, kind, torch.from_numpy(expected).dtype)
                    self.assertTrue(np.array_equal(result.numpy(), expected), f"{result} != {expected}")
        finally:
            shutil.rmtree(tmpdir)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestCuda(HdfDirectTestCase):
    def test_matches_h5py(self):
        self.write(make_frames(np.uint32, frame_num=203), **bitshuffle())
        pixel_num = FRAME_SHAPE[0] * FRAME_SHAPE[1]
        small_crop = torch.arange(0, pixel_num, 7, device="cuda:0")
        large_crop = torch.arange(0, pixel_num, 2, device="cuda:0")
        for kwargs in (dict(), dict(mask_crop=small_crop), dict(mask_crop=large_crop),
                       dict(avg_frame=2, stride_frame=2)):
            with self.subTest(**{k: "set" for k in kwargs}):
                dset = self.assert_same_as_h5py(device="cuda:0", **kwargs)
                self.assertEqual(dset.device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
