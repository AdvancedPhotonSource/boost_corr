#!/usr/bin/env python

"""Tests for the RigakuExtend (.bix) sparse data bit layout."""

import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
from boost_corr.xpcs_aps_8idi.dataset.rigaku_utils import convert_sparse_extend
from boost_corr.xpcs_aps_8idi.dataset.rigaku_extend_handler import RigakuExtendDataset


def pack(count, pixel, frame):
    return (
        (np.uint64(frame) << np.uint64(32))
        + (np.uint64(pixel) << np.uint64(12))
        + np.uint64(count)
    )


class TestRigakuExtend(unittest.TestCase):
    def test_000_single_word_roundtrip(self):
        words = np.array([pack(123, 456789, 42)], dtype=np.uint64)
        pixel, frame, count = convert_sparse_extend(words)
        self.assertEqual(pixel[0], 456789)
        self.assertEqual(frame[0], 42)
        self.assertEqual(count[0], 123)

    def test_001_max_values(self):
        # count is stored as uint8 downstream (same convention as the
        # original Rigaku format), so only pixel/frame are exercised at
        # their true bit-width maximum here.
        max_pixel = 2**20 - 1
        max_frame = 2**32 - 1
        words = np.array([pack(0, max_pixel, max_frame)], dtype=np.uint64)
        pixel, frame, count = convert_sparse_extend(words)
        self.assertEqual(pixel[0], max_pixel)
        self.assertEqual(frame[0], max_frame)
        self.assertEqual(count[0], 0)

    def test_002_field_isolation(self):
        # each field must not leak into the neighboring one
        words = np.array(
            [
                pack(0, 0, 0),
                pack(255, 0, 0),  # count field maxed (uint8 range), others zero
                pack(0, 2**20 - 1, 0),  # pixel field maxed, others zero
                pack(0, 0, 2**32 - 1),  # frame field maxed, others zero
            ],
            dtype=np.uint64,
        )
        pixel, frame, count = convert_sparse_extend(words)
        np.testing.assert_array_equal(pixel, [0, 0, 2**20 - 1, 0])
        np.testing.assert_array_equal(frame, [0, 0, 0, 2**32 - 1])
        np.testing.assert_array_equal(count, [0, 255, 0, 0])

    def test_004_count_above_uint8_wraps_like_rigaku(self):
        # documents parity with the original Rigaku decoder: the 12-bit
        # count field is masked correctly, but storage narrows to uint8,
        # so values above 255 wrap modulo 256 rather than raising.
        words = np.array([pack(2**12 - 1, 0, 0)], dtype=np.uint64)
        _, _, count = convert_sparse_extend(words)
        self.assertEqual(count[0], (2**12 - 1) % 256)

    def test_003_multi_word_batch(self):
        words = np.array(
            [pack(1, 10, 0), pack(2, 20, 0), pack(3, 30, 1)], dtype=np.uint64
        )
        pixel, frame, count = convert_sparse_extend(words)
        np.testing.assert_array_equal(pixel, [10, 20, 30])
        np.testing.assert_array_equal(frame, [0, 0, 1])
        np.testing.assert_array_equal(count, [1, 2, 3])


class TestRigakuExtendDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fname = os.path.join(self.tmpdir, "sample.bix")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_bix(self, events):
        # events: list of (pixel, frame, count), must be frame-sorted
        words = np.array(
            [pack(count, pixel, frame) for pixel, frame, count in events],
            dtype=np.uint64,
        )
        words.tofile(self.fname)

    def test_dispatches_from_create_dataset(self):
        from boost_corr.xpcs_aps_8idi.dataset.utils import create_dataset

        self._write_bix([(0, 0, 1)])
        ds, use_loader = create_dataset(self.fname, det_size=(2, 2), device="cpu")
        self.assertIsInstance(ds, RigakuExtendDataset)
        self.assertFalse(use_loader)

    def test_decodes_dense_frames(self):
        self._write_bix([(1, 0, 3), (3, 0, 1), (0, 1, 2)])
        ds = RigakuExtendDataset(
            self.fname,
            det_size=(2, 2),
            batch_size=2,
            device="cpu",
            signal_spill="none",
        )
        self.assertEqual(ds.dataset_type, "RigakuExtend 64bit Binary")

        expected = torch.zeros(2, 4, dtype=torch.uint8)
        expected[0, 1] = 3
        expected[0, 3] = 1
        expected[1, 0] = 2

        torch.testing.assert_close(ds[0], expected)

    def test_signal_spill_disabled_for_now(self):
        # update_signal_spill()'s call sites are currently commented out
        # (superseded by the event correlator's CCF direction mechanism --
        # see EventCorrelator/xpcs_dataset.py's update_signal_spill
        # docstring); the dense-frame correction itself does not run even
        # when signal_spill="h"/"v"/"vh" is requested. This pins that
        # intentional, temporary behavior rather than the correction itself.
        self._write_bix([(2, 0, 1)])
        ds = RigakuExtendDataset(
            self.fname,
            det_size=(1, 5),
            batch_size=1,
            device="cpu",
            signal_spill="h",
        )
        row = ds[0][0]
        expected = torch.tensor([0, 0, 1, 0, 0], dtype=torch.uint8)
        torch.testing.assert_close(row, expected)


if __name__ == "__main__":
    unittest.main()
