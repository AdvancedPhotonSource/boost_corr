#!/usr/bin/env python
"""Round-trip test: writing event-correlator payloads through XpcsResult
lands at the expected /xpcs/event/... HDF5 paths."""
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from boost_corr.xpcs_aps_8idi.xpcs_result import XpcsResult


class TestEventOutputSchema(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.raw_fname = os.path.join(self.tmpdir, "run.bin")
        open(self.raw_fname, "wb").close()
        self.meta_fname = os.path.join(self.tmpdir, "run_metadata.hdf")
        with h5py.File(self.meta_fname, "w") as f:
            f["/entry/schema_version"] = "nexus"
        # XpcsResult.__enter__ -> append_metadata_qmap -> copy_qmap always
        # opens qmap_fname (no None-safe path), so a real file with a
        # /qmap group is required even though this test never constructs
        # an XpcsQPartitionMap from it.
        self.qmap_fname = os.path.join(self.tmpdir, "qmap.h5")
        with h5py.File(self.qmap_fname, "w") as f:
            f.create_group("/qmap")
        self.output_dir = os.path.join(self.tmpdir, "out")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_event_payload_lands_at_expected_paths(self):
        payload = {
            "event_run_info": {"n_rings": 2, "directions": ["horizontal"]},
            "event_delay_list_acf": np.array([1, 2, 4]),
            "event_g2_acf": np.ones((2, 3)),
            "event_g2_acf_err": np.zeros((2, 3)),
            "event_delay_list_ccf": np.array([0, 1, 2]),
            "event_g2_ccf_h": np.ones((2, 3)),
            "event_g2_ccf_h_err": np.zeros((2, 3)),
        }
        with XpcsResult(
            raw_fname=self.raw_fname,
            meta_fname=self.meta_fname,
            qmap_fname=self.qmap_fname,
            output_dir=self.output_dir,
            event_config={"signal_spill": "h"},
        ) as result:
            result.append(payload)
            fname = result.fname

        with h5py.File(fname, "r") as f:
            self.assertIn("/xpcs/event/config", f)
            self.assertEqual(f["/xpcs/event/config/signal_spill"][()].decode(), "h")
            self.assertIn("/xpcs/event/run_info", f)
            np.testing.assert_array_equal(f["/xpcs/event/delay_list_acf"][()], [1, 2, 4])
            self.assertEqual(f["/xpcs/event/normalized_g2_acf"][()].shape, (2, 3))
            self.assertEqual(f["/xpcs/event/normalized_g2_acf_err"][()].shape, (2, 3))
            np.testing.assert_array_equal(f["/xpcs/event/delay_list_ccf"][()], [0, 1, 2])
            self.assertEqual(f["/xpcs/event/normalized_g2_ccf_h"][()].shape, (2, 3))
            self.assertEqual(f["/xpcs/event/normalized_g2_ccf_h_err"][()].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
