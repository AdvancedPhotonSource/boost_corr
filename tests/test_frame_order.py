#!/usr/bin/env python
"""Tests that HDF5 frames reach the correlators in acquisition order.

XPCS correlates every frame with the ones that follow it, so a reader that
hands out frames shuffled, repeated, dropped or mixed gives a wrong g2
without raising anything. Every pixel of a test frame holds the 1-based
number of that frame, so the number of a frame can be read back from the
frame itself, and a frame put together from two different ones shows up as
non-uniform.

Both ways HdfDataset reads a file are covered: the direct-chunk threads, and
h5py wrapped in a DataLoader with worker processes.
"""
import os
import shutil
import tempfile
import unittest

import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import DataLoader

from boost_corr.correlator import MultitauCorrelator, TwotimeCorrelator
from boost_corr.xpcs_aps_8idi.dataset.hdf_handler import HdfDataset
from boost_corr.xpcs_aps_8idi.dataset.utils import create_dataset

DATA_PATH = "/entry/data/data"
FRAME_SHAPE = (128, 192)
# the correlators keep the first level in bfloat16, which is exact for
# integers up to 256
FRAME_NUM = 240

SELECTIONS = (
    dict(),
    dict(begin_frame=7, end_frame=200),
    dict(stride_frame=3),
    dict(avg_frame=2),
    dict(begin_frame=5, stride_frame=2, avg_frame=3),
)

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def write_numbered_frames(fname):
    data = np.empty((FRAME_NUM,) + FRAME_SHAPE, dtype=np.uint16)
    data[:] = np.arange(1, FRAME_NUM + 1).reshape(-1, 1, 1)
    with h5py.File(fname, "w") as f:
        f.create_dataset(DATA_PATH, data=data, chunks=(1,) + FRAME_SHAPE,
                         **hdf5plugin.Bitshuffle())


def expected_numbers(begin_frame=0, end_frame=-1, stride_frame=1, avg_frame=1):
    """the frame numbers a dataset with this selection must hand out"""
    end = FRAME_NUM if end_frame <= 0 else min(end_frame, FRAME_NUM)
    eff_len = stride_frame * avg_frame
    end = begin_frame + (end - begin_frame) // eff_len * eff_len
    picked = np.arange(begin_frame, end, stride_frame) + 1
    return picked.reshape(-1, avg_frame).mean(axis=1)


def dataset_kwargs(**kwargs):
    opts = dict(device="cpu", avg_frame=1, begin_frame=0, end_frame=-1,
                stride_frame=1, mask_crop=None)
    opts.update(kwargs)
    return opts


class FrameOrderTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fname = os.path.join(self.tmpdir, "numbered.h5")
        write_numbered_frames(self.fname)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def frame_numbers(self, x):
        """the number of each frame in x (frames, pixels)"""
        x = x.float().cpu()
        low, high = x.min(dim=1).values, x.max(dim=1).values
        mixed = torch.nonzero(low != high).reshape(-1).tolist()
        self.assertEqual(mixed, [], "frames mixed from different raw frames")
        return low.numpy()

    def assert_in_order(self, numbers, **selection):
        expected = expected_numbers(**selection)
        self.assertEqual(len(numbers), len(expected))
        wrong = np.nonzero(np.asarray(numbers) != expected)[0]
        self.assertEqual(
            wrong.size, 0,
            f"{wrong.size} frames out of order, first at position "
            f"{wrong[:1]}: got {np.asarray(numbers)[wrong[:5]]}, "
            f"expected {expected[wrong[:5]]}")


class TestHdfDatasetFrameOrder(FrameOrderTestCase):
    def open(self, direct_read, **kwargs):
        kwargs.setdefault("batch_size", 8)
        dset = HdfDataset(self.fname, direct_read=direct_read,
                          **dataset_kwargs(**kwargs))
        self.addCleanup(dset.close)
        self.assertEqual(dset.direct_read, direct_read)
        return dset

    def read_all(self, batches):
        return np.concatenate([self.frame_numbers(x) for x in batches])

    def test_direct_read(self):
        for selection in SELECTIONS:
            for num_threads in (1, 4, 32):
                with self.subTest(num_threads=num_threads, **selection):
                    dset = self.open(True, num_threads=num_threads, **selection)
                    for _ in range(2):  # the second pass restarts the prefetch
                        numbers = self.read_all(dset[n] for n in range(len(dset)))
                        self.assert_in_order(numbers, **selection)
                        dset.__reset__()

    def test_direct_read_on_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        for selection in SELECTIONS:
            with self.subTest(**selection):
                dset = self.open(True, device="cuda:0", **selection)
                numbers = self.read_all(dset[n] for n in range(len(dset)))
                self.assert_in_order(numbers, **selection)

    def test_h5py(self):
        for selection in SELECTIONS:
            with self.subTest(**selection):
                dset = self.open(False, **selection)
                numbers = self.read_all(dset[n] for n in range(len(dset)))
                self.assert_in_order(numbers, **selection)

    def test_h5py_with_loader(self):
        # the workers queue a batch and start reading the next one right
        # away; a batch must not change once it has been handed out
        for selection in SELECTIONS:
            for num_workers in (1, 4):
                with self.subTest(num_workers=num_workers, **selection):
                    dset = self.open(False, **selection)
                    loader = DataLoader(dset, batch_size=None,
                                        num_workers=num_workers,
                                        prefetch_factor=4)
                    numbers = self.read_all(loader)
                    self.assert_in_order(numbers, **selection)

    def test_h5py_with_loader_repeated(self):
        # the mix-up is a race; give it several chances to show
        dset = self.open(False, batch_size=2)
        for _ in range(5):
            loader = DataLoader(dset, batch_size=None, num_workers=2,
                                prefetch_factor=8)
            self.assert_in_order(self.read_all(loader))


class TestCorrelatorFrameOrder(FrameOrderTestCase):
    """the same path solve_correlation takes: create_dataset, then
    process_dataset with the loader setting create_dataset returned"""

    def create(self, direct_read, device):
        dset, use_loader = create_dataset(
            self.fname, direct_read=direct_read, **dataset_kwargs(device=device))
        self.addCleanup(dset.close)
        self.assertEqual(dset.direct_read, direct_read)
        self.assertEqual(use_loader, not direct_read)
        self.assertEqual(dset.frame_num, FRAME_NUM)
        return dset, use_loader

    def cases(self):
        for device in DEVICES:
            for direct_read in (True, False):
                with self.subTest(device=device, direct_read=direct_read):
                    yield (device,) + self.create(direct_read, device)

    def test_twotime_cache(self):
        pixel_num = FRAME_SHAPE[0] * FRAME_SHAPE[1]
        qinfo = {
            "dq_idx": np.array([1]),
            "dq_slc": [slice(0, pixel_num)],
            "sq_idx": np.array([1]),
            "sq_slc": [slice(0, pixel_num)],
        }
        for device, dset, use_loader in self.cases():
            correlator = TwotimeCorrelator(
                qinfo=qinfo, frame_num=dset.frame_num, det_size=dset.det_size,
                device=device)
            correlator.process_dataset(dset, verbose=False,
                                       use_loader=use_loader, num_workers=4)
            # row n of the cache is the n-th frame
            self.assert_in_order(self.frame_numbers(correlator.cache))

    def test_multitau_intensity_vs_time(self):
        for device, dset, use_loader in self.cases():
            correlator = MultitauCorrelator(dset.det_size, dset.frame_num,
                                            device=device)
            correlator.process_dataset(dset, verbose=False,
                                       use_loader=use_loader, num_workers=4)
            correlator.post_process()
            scattering, _ = correlator.get_results()
            index, intensity = scattering["intensity_vs_time"].cpu().numpy()
            self.assertTrue(np.array_equal(index, np.arange(FRAME_NUM)))
            # the mean of a frame is its number
            self.assert_in_order(intensity)


if __name__ == "__main__":
    unittest.main()
