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


class TestEventRejectsFrameSegmentation(unittest.TestCase):
    """The event path feeds on dset.get_events(), which always returns the
    whole file: begin/end/avg/stride_frame and num_segments are applied
    nowhere on it. Before this guard, --num-segments 2 produced two
    byte-identical full-file outputs -- silently wrong, worse than an error."""

    NON_DEFAULTS = (
        {"num_segments": 2},
        {"begin_frame": 100},
        {"end_frame": 500},
        {"avg_frame": 2},
        {"stride_frame": 3},
        {"num_segments": 2, "begin_frame": 100},
    )

    def test_non_default_frame_params_raise_before_touching_data(self):
        for extra in self.NON_DEFAULTS:
            with self.subTest(**extra):
                with patch.object(common, "create_qpm") as mock_create_qpm:
                    with self.assertRaises(ConfigurationError) as ctx:
                        common.solve_correlation(
                            signal_spill="h", raw=["dummy.bin"], qmap="dummy.h5",
                            **extra,
                        )
                # must fail on configuration, not after loading the qmap
                self.assertFalse(mock_create_qpm.called)
                msg = str(ctx.exception)
                self.assertIn("does not support", msg)
                for key in extra:
                    self.assertIn(key, msg)

    def test_non_default_frame_params_are_fine_without_signal_spill(self):
        """The guard must only fire on the event path -- multitau still
        supports every one of these flags."""
        with patch.object(common, "get_device", return_value="cpu"), \
                patch.object(common, "create_qpm", side_effect=RuntimeError("reached")):
            with self.assertRaises(RuntimeError):
                common.solve_correlation(
                    analysis_type="Multitau", signal_spill="none",
                    raw=["dummy.bin"], qmap="dummy.h5",
                    num_segments=2, begin_frame=100, end_frame=500,
                    avg_frame=2, stride_frame=3,
                )

    @patch.object(common, "run_jobs")
    @patch.object(common, "build_jobs", return_value=[("dummy.bin", {}, None)])
    @patch.object(common, "load_raw_list", return_value=["dummy.bin"])
    @patch.object(common, "create_qpm")
    @patch.object(common, "get_device", return_value="cpu")
    @patch.object(common.EventCorrelator, "from_config")
    def test_explicit_default_frame_params_do_not_raise(
        self, mock_from_config, mock_get_device, mock_create_qpm,
        mock_load_raw_list, mock_build_jobs, mock_run_jobs,
    ):
        """Passing every frame param explicitly at its documented no-op
        default must still route straight through to the event correlator."""
        common.solve_correlation(
            signal_spill="h", raw=["dummy.bin"], qmap="dummy.h5",
            num_segments=1, begin_frame=0, end_frame=-1,
            avg_frame=1, stride_frame=1,
        )
        self.assertTrue(mock_run_jobs.called)
        self.assertEqual(mock_run_jobs.call_args.kwargs["config_key"], "event_config")

    @patch.object(common, "run_jobs")
    @patch.object(common, "build_jobs", return_value=[("dummy.bin", {}, None)])
    @patch.object(common, "load_raw_list", return_value=["dummy.bin"])
    @patch.object(common, "create_qpm")
    @patch.object(common, "get_device", return_value="cpu")
    @patch.object(common.EventCorrelator, "from_config")
    def test_guard_scratch_state_does_not_leak_into_saved_config(
        self, mock_from_config, mock_get_device, mock_create_qpm,
        mock_load_raw_list, mock_build_jobs, mock_run_jobs,
    ):
        """analysis_kwargs is built from locals() and written verbatim to
        /xpcs/event/config, so the guard's scratch list must not survive."""
        common.solve_correlation(
            signal_spill="h", raw=["dummy.bin"], qmap="dummy.h5",
        )
        analysis_kwargs = mock_run_jobs.call_args.kwargs["analysis_kwargs"]
        self.assertNotIn("_non_default", analysis_kwargs)
        self.assertEqual(analysis_kwargs["analysis_type"], "Event")


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
