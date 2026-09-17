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


if __name__ == "__main__":
    unittest.main()
