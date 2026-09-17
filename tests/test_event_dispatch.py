#!/usr/bin/env python
"""Dispatch-level tests: does solve_correlation route to the event
correlator, and does it reject signal_spill + Twotime/Both correctly?
These test the dispatch logic in isolation (mocking create_qpm/run_jobs)
rather than running a full correlation -- Task 7 covers the full path."""
import unittest
from unittest.mock import patch

import numpy as np

from boost_corr.xpcs_aps_8idi.correlation import common
from boost_corr.xpcs_aps_8idi.exceptions import ConfigurationError


class TestEventDispatch(unittest.TestCase):
    def test_signal_spill_with_twotime_raises(self):
        with self.assertRaises(ConfigurationError):
            common.solve_correlation(
                analysis_type="Twotime", signal_spill="h", raw=["dummy.bin"],
                qmap="dummy.h5",
            )

    def test_signal_spill_with_both_raises(self):
        with self.assertRaises(ConfigurationError):
            common.solve_correlation(
                analysis_type="Both", signal_spill="v", raw=["dummy.bin"],
                qmap="dummy.h5",
            )

    @patch.object(common, "run_jobs")
    @patch.object(common, "build_jobs", return_value=[("dummy.bin", {}, None)])
    @patch.object(common, "load_raw_list", return_value=["dummy.bin"])
    @patch.object(common, "create_qpm")
    @patch.object(common, "get_device", return_value="cpu")
    @patch.object(common.EventCorrelator, "from_config")
    def test_signal_spill_routes_to_event_correlator(
        self, mock_from_config, mock_get_device, mock_create_qpm,
        mock_load_raw_list, mock_build_jobs, mock_run_jobs,
    ):
        common.solve_correlation(
            analysis_type="Multitau",  # default -- signal_spill must override it
            signal_spill="h", raw=["dummy.bin"], qmap="dummy.h5",
        )
        self.assertTrue(mock_run_jobs.called)
        run_args = mock_run_jobs.call_args.kwargs
        self.assertEqual(run_args["config_key"], "event_config")

        # invoke the captured factory to confirm it dispatches to
        # EventCorrelator.from_config, not Multitau's/Twotime's -- avoids
        # relying on closure-cell introspection, which is a CPython
        # implementation detail, not a stable interface to assert on.
        run_args["correlator_factory"](dset=object())
        self.assertTrue(mock_from_config.called)


class TestPrepareCorrelatorNeverReusesEventCorrelator(unittest.TestCase):
    """Regression test: EventCorrelator.frame_num must exist (as None) or
    prepare_correlator raises AttributeError on the second job of any
    multi-file/--num-segments>1 run -- a real bug caught during planning,
    not a hypothetical."""

    def test_prepare_correlator_rebuilds_every_time(self):
        ring_map = np.ones((2, 2), dtype=np.int64) - 1
        ec = common.EventCorrelator(ring_map, ("horizontal",), device="cpu")

        class _FakeDset:
            frame_num = 123
            det_size = (2, 2)

        # must not raise, and must return None (forcing a rebuild) even
        # though det_size matches -- EventCorrelator.frame_num is always None.
        result = common.prepare_correlator(ec, _FakeDset())
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
