#!/usr/bin/env python
"""End-to-end test: a synthetic Rigaku .bin file + a synthetic qmap run
through solve_correlation(analysis_type="Multitau", signal_spill="h", ...)
end up as event-correlator output at /xpcs/event/... with correct shapes."""
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from boost_corr.xpcs_aps_8idi.correlation import solve_correlation
from boost_corr.xpcs_aps_8idi.exceptions import ConfigurationError


def pack_rigaku(count, pixel, frame):
    return (
        (np.uint64(frame) << np.uint64(40))
        + (np.uint64(pixel) << np.uint64(16))
        + np.uint64(count)
    )


class TestEventCorrelatorIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmpdir, "out")

        # RigakuDataset never overrides det_size, and solve_correlation has
        # no det_size parameter at all -- XpcsDataset's default (512, 1024)
        # is the only shape a plain .bin file can use here, or
        # qpm.update_rotation() raises AssertionError on a shape mismatch.
        # Use a small active patch within it, matching this project's real
        # run_246 qmap (a few hundred valid pixels out of 524288), rather
        # than a full 512x1024 mask.
        self.det_size = (512, 1024)
        patch_rows = range(10, 14)  # 4 rows
        patch_cols = range(10, 16)  # 6 cols
        patch_pixels = np.array(
            [r * self.det_size[1] + c for r in patch_rows for c in patch_cols],
            dtype=np.int64,
        )

        rng = np.random.default_rng(2)
        n_events = 2000
        n_frames = 500  # "frame" field == 1 us time bin, per bin_time_s below
        pixels = rng.choice(patch_pixels, size=n_events)
        frames = np.sort(rng.integers(0, n_frames, n_events))
        counts = rng.integers(1, 3, n_events)

        self.raw_fname = os.path.join(self.tmpdir, "run.bin")
        words = np.array(
            [pack_rigaku(c, p, f) for p, f, c in zip(pixels, frames, counts)],
            dtype=np.uint64,
        )
        words.tofile(self.raw_fname)

        self.meta_fname = os.path.join(self.tmpdir, "run_metadata.hdf")
        with h5py.File(self.meta_fname, "w") as f:
            f["/entry/schema_version"] = "nexus"

        self.qmap_fname = os.path.join(self.tmpdir, "qmap.h5")
        dqmap = np.zeros(self.det_size, dtype=np.uint8)
        for r in patch_rows:
            for c in patch_cols:
                dqmap[r, c] = 1
        mask = (dqmap > 0).astype(np.uint8)
        with h5py.File(self.qmap_fname, "w") as f:
            f["/qmap/dynamic_roi_map"] = dqmap
            f["/qmap/static_roi_map"] = dqmap
            f["/qmap/mask"] = mask

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_signal_spill_runs_event_correlator_end_to_end(self):
        solve_correlation(
            analysis_type="Multitau",  # default -- must be overridden
            signal_spill="h",
            raw=[self.raw_fname],
            qmap=self.qmap_fname,
            output=self.output_dir,
            meta_fname=self.meta_fname,
            gpu_id=-1,
            bin_time_s=1e-6,
        )
        result_files = [f for f in os.listdir(self.output_dir) if f.endswith("_results.hdf")]
        self.assertEqual(len(result_files), 1)
        with h5py.File(os.path.join(self.output_dir, result_files[0]), "r") as f:
            self.assertIn("/xpcs/event/config", f)
            self.assertIn("/xpcs/event/normalized_g2_acf", f)
            self.assertIn("/xpcs/event/normalized_g2_ccf_h", f)
            self.assertNotIn("/xpcs/event/normalized_g2_ccf_v", f)
            self.assertNotIn("/xpcs/multitau/normalized_g2", f)
            g2_acf = f["/xpcs/event/normalized_g2_acf"][()]
            self.assertEqual(g2_acf.shape[0], 1)  # one dq ring

    def test_signal_spill_with_twotime_raises_before_touching_data(self):
        with self.assertRaises(ConfigurationError):
            solve_correlation(
                analysis_type="Twotime",
                signal_spill="h",
                raw=[self.raw_fname],
                qmap=self.qmap_fname,
                output=self.output_dir,
                meta_fname=self.meta_fname,
                gpu_id=-1,
            )
        self.assertFalse(os.path.isdir(self.output_dir))


if __name__ == "__main__":
    unittest.main()
