#!/usr/bin/env python
"""Tests for RigakuDataset.get_events() and bin-width resolution."""
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from boost_corr.xpcs_aps_8idi.dataset.utils import resolve_bin_width_ns
from boost_corr.xpcs_aps_8idi.dataset.rigaku_handler import RigakuDataset


def pack_rigaku(count, pixel, frame):
    return (
        (np.uint64(frame) << np.uint64(40))
        + (np.uint64(pixel) << np.uint64(16))
        + np.uint64(count)
    )


class TestResolveBinWidthNs(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.raw_fname = os.path.join(self.tmpdir, "run_metatest.bin")
        open(self.raw_fname, "wb").close()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_falls_back_to_bin_time_s_when_no_metadata(self):
        bin_width_ns = resolve_bin_width_ns(self.raw_fname, bin_time_s=1e-6)
        self.assertAlmostEqual(bin_width_ns, 1000.0)

    def test_reads_metadata_frame_time_when_present(self):
        meta_fname = os.path.join(self.tmpdir, "run_metatest_metadata.hdf")
        with h5py.File(meta_fname, "w") as f:
            f["/entry/instrument/detector_1/frame_time"] = 1e-8
        bin_width_ns = resolve_bin_width_ns(self.raw_fname, bin_time_s=1e-6)
        self.assertAlmostEqual(bin_width_ns, 10.0)


class TestRigakuGetEvents(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fname = os.path.join(self.tmpdir, "sample.bin")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_bin(self, events):
        # events: list of (pixel, frame, count), must be frame-sorted
        words = np.array(
            [pack_rigaku(count, pixel, frame) for pixel, frame, count in events],
            dtype=np.uint64,
        )
        words.tofile(self.fname)

    def test_get_events_matches_raw_sparse_triple(self):
        self._write_bin([(1, 0, 3), (3, 0, 1), (0, 1, 2)])
        ds = RigakuDataset(
            self.fname, det_size=(2, 2), batch_size=2, device="cpu",
            signal_spill="none", bin_time_s=1e-6,
        )
        pixel, time_bin, count, bin_width_ns = ds.get_events()
        np.testing.assert_array_equal(pixel, [1, 3, 0])
        np.testing.assert_array_equal(time_bin, [0, 0, 1])
        np.testing.assert_array_equal(count, [3, 1, 2])
        self.assertAlmostEqual(bin_width_ns, 1000.0)  # 1e-6 s -> 1000 ns fallback

    def test_get_events_dtype_is_int64(self):
        self._write_bin([(0, 0, 1)])
        ds = RigakuDataset(
            self.fname, det_size=(2, 2), batch_size=2, device="cpu",
            signal_spill="none",
        )
        pixel, time_bin, count, _ = ds.get_events()
        self.assertEqual(pixel.dtype, np.int64)
        self.assertEqual(time_bin.dtype, np.int64)
        self.assertEqual(count.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
