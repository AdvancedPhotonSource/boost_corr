#!/usr/bin/env python
"""Tests for TimepixAnalysisDataset.get_events(). Constructs a bare
instance (bypassing __init__, which needs the external timepix_dataset
package and a real file) and sets only the attributes get_events() reads,
so this test has no dependency on real Timepix4 hardware files."""
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

# Mock timepix_dataset module only for this test file, scoped narrowly
# to avoid affecting other tests or future real uses of the module.
sys.modules['timepix_dataset'] = MagicMock()
sys.modules['timepix_dataset.dataset'] = MagicMock()

from boost_corr.xpcs_aps_8idi.dataset.timepix4_handler import TimepixAnalysisDataset
from boost_corr.xpcs_aps_8idi.exceptions import DatasetError


def _bare_instance(num_chips, ifc_list, bin_time_s=1e-6):
    ds = object.__new__(TimepixAnalysisDataset)
    ds.raw_dataset = MagicMock()
    ds.raw_dataset.num_chips = num_chips
    ds.ifc_list = ifc_list
    ds.bin_time_s = bin_time_s
    return ds


class TestTimepix4GetEvents(unittest.TestCase):
    def test_single_chip_returns_events(self):
        ifc = (
            torch.tensor([1, 3, 0], dtype=torch.int64),
            torch.tensor([0, 0, 1], dtype=torch.int64),
            torch.tensor([3, 1, 2], dtype=torch.int64),
        )
        ds = _bare_instance(num_chips=1, ifc_list=[ifc], bin_time_s=1e-6)
        pixel, time_bin, count, bin_width_ns = ds.get_events()
        np.testing.assert_array_equal(pixel, [1, 3, 0])
        np.testing.assert_array_equal(time_bin, [0, 0, 1])
        np.testing.assert_array_equal(count, [3, 1, 2])
        self.assertAlmostEqual(bin_width_ns, 1000.0)
        self.assertEqual(pixel.dtype, np.int64)

    def test_multi_chip_raises_dataset_error(self):
        ds = _bare_instance(num_chips=2, ifc_list=[None, None])
        with self.assertRaises(DatasetError):
            ds.get_events()

    def test_three_chip_remaps_pixel_into_canvas(self):
        # mod_size (2 rows x 3 cols) stitched into a (2, 9) canvas via a
        # layout that mirrors the real 3-chip Timepix4 geometry (module 0
        # lands rightmost): layout = [slice(6, 9), slice(3, 6), slice(0, 3)].
        ifc_list = [
            (
                torch.tensor([0, 4], dtype=torch.int64),  # r0c0, r1c1
                torch.tensor([10, 11], dtype=torch.int64),
                torch.tensor([1, 2], dtype=torch.int64),
            ),
            (
                torch.tensor([0], dtype=torch.int64),  # r0c0
                torch.tensor([20], dtype=torch.int64),
                torch.tensor([3], dtype=torch.int64),
            ),
            (
                torch.tensor([0, 5], dtype=torch.int64),  # r0c0, r1c2
                torch.tensor([30, 31], dtype=torch.int64),
                torch.tensor([4, 5], dtype=torch.int64),
            ),
        ]
        ds = _bare_instance(num_chips=3, ifc_list=ifc_list, bin_time_s=1e-6)
        ds.raw_dataset.mod_size = (2, 3)
        ds.raw_dataset.det_size = (2, 9)
        ds.raw_dataset.layout = [slice(6, 9), slice(3, 6), slice(0, 3)]

        pixel, time_bin, count, bin_width_ns = ds.get_events()

        np.testing.assert_array_equal(pixel, [6, 16, 3, 0, 11])
        np.testing.assert_array_equal(time_bin, [10, 11, 20, 30, 31])
        np.testing.assert_array_equal(count, [1, 2, 3, 4, 5])
        self.assertAlmostEqual(bin_width_ns, 1000.0)
        self.assertEqual(pixel.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
