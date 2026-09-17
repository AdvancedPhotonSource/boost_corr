#!/usr/bin/env python
"""Tests for EventCorrelator, checked against the brute-force reference
used in test_event_core.py."""
import unittest

import numpy as np
import torch

from boost_corr.correlator.event import EventCorrelator
from boost_corr.correlator._event_core import (
    build_event_table, auto_pairs, neighbour_pairs, log_lag_edges,
    correlate_pairs_torch,
)
from boost_corr.xpcs_aps_8idi.exceptions import DatasetError, ConfigurationError


class _FakeQpm:
    """Minimal stand-in for XpcsQPartitionMap: only .dqmap and .mask are used
    by EventCorrelator.from_config. normalize_scattering here is a recording
    stub (used to inspect the tensors EventCorrelator hands it); the real
    XpcsQPartitionMap.normalize_scattering is exercised end-to-end in
    test_event_integration.py."""
    def __init__(self, dqmap, mask=None):
        self.dqmap = dqmap
        # real qmaps always carry a boolean mask of the same shape; default to
        # "everything valid" so tests that do not care about masking are terse.
        self.mask = (np.ones(np.shape(dqmap), dtype=bool) if mask is None
                     else np.asarray(mask, dtype=bool))
        self.scattering_seen = None

    def normalize_scattering(self, res):
        self.scattering_seen = res
        return {"saxs_2d": res["saxs_2d"]}


class _FakeDataset:
    """Minimal stand-in satisfying the get_events() protocol."""
    def __init__(self, pixel, time_bin, count, bin_width_ns, signal_spill):
        self._events = (pixel, time_bin, count, bin_width_ns)
        self.signal_spill = signal_spill

    def get_events(self):
        return self._events


def _synthetic_ring(rng, ny=3, nx=6, T=200, n_events=400):
    npix = ny * nx
    toa = rng.uniform(0, T, n_events)
    pix = rng.integers(0, npix, n_events)
    dqmap = np.ones((ny, nx), dtype=np.int64)  # single ring, dq label 1
    return pix, toa, dqmap, npix


class TestEventCorrelatorFromConfig(unittest.TestCase):
    def test_rejects_dataset_without_get_events(self):
        class _NoEvents:
            signal_spill = "h"

        with self.assertRaises(DatasetError):
            EventCorrelator.from_config(_FakeQpm(np.ones((2, 2), dtype=np.int64)), _NoEvents(), "cpu")

    def test_rejects_signal_spill_none(self):
        ds = _FakeDataset(np.array([0]), np.array([0]), np.array([1]), 1.0, signal_spill="none")
        with self.assertRaises(ConfigurationError):
            EventCorrelator.from_config(_FakeQpm(np.ones((2, 2), dtype=np.int64)), ds, "cpu")

    def test_direction_selection(self):
        dqmap = np.ones((2, 2), dtype=np.int64)
        for spill, expected in (("h", ("horizontal",)), ("v", ("vertical",)),
                                ("vh", ("horizontal", "vertical"))):
            ds = _FakeDataset(np.array([0]), np.array([0]), np.array([1]), 1.0, signal_spill=spill)
            ec = EventCorrelator.from_config(_FakeQpm(dqmap), ds, "cpu")
            self.assertEqual(ec.directions, expected)

    def test_masked_out_pixels_are_dropped_from_ring_map(self):
        """dq rings must be intersected with qpm.mask, exactly as the dense
        multitau path does (it only ever correlates dqmap[mask_idx_1d]).
        A blemished/dead pixel that still carries a nonzero dq label must not
        enter the event correlator's ACF/CCF pixel-pair lists."""
        ny, nx = 3, 6
        dqmap = np.ones((ny, nx), dtype=np.int64)
        dqmap[2, :] = 2  # second dq ring on the last row
        mask = np.ones((ny, nx), dtype=bool)
        mask[0, 0] = False   # masked-out pixel inside dq ring 1
        mask[2, 3] = False   # masked-out pixel inside dq ring 2

        ds = _FakeDataset(np.array([0]), np.array([0]), np.array([1]), 1.0,
                          signal_spill="h")
        ec = EventCorrelator.from_config(_FakeQpm(dqmap, mask), ds, "cpu")

        # masked-out pixels fall out of every ring (-1 == "no ring")
        self.assertEqual(ec.ring_map[0, 0], -1)
        self.assertEqual(ec.ring_map[2, 3], -1)
        # masked-in pixels keep their (0-based) ring index
        self.assertEqual(ec.ring_map[0, 1], 0)
        self.assertEqual(ec.ring_map[2, 4], 1)
        self.assertEqual(ec.n_rings, 2)

        # ...and therefore out of the pair lists post_process() builds
        flat_masked = [0 * nx + 0, 2 * nx + 3]
        for ring in range(ec.n_rings):
            members = set(auto_pairs(ec.ring_map, ring).ravel().tolist())
            for direction in ("horizontal", "vertical"):
                members |= set(
                    neighbour_pairs(ec.ring_map, ring, direction, True).ravel().tolist()
                )
            for p in flat_masked:
                self.assertNotIn(p, members)

        # without the mask intersection every pixel would still be in a ring
        ec_unmasked = EventCorrelator.from_config(_FakeQpm(dqmap), ds, "cpu")
        self.assertEqual(ec_unmasked.ring_map[0, 0], 0)
        self.assertEqual(ec_unmasked.ring_map[2, 3], 1)


class TestEventCorrelatorAgainstReference(unittest.TestCase):
    def test_matches_direct_correlate_pairs_torch_call(self):
        rng = np.random.default_rng(1)
        pix, toa, dqmap, npix = _synthetic_ring(rng)
        bin_width_ns = 1.0
        ds = _FakeDataset(pix.astype(np.int64), np.floor(toa).astype(np.int64),
                          np.ones_like(pix), bin_width_ns, signal_spill="h")

        qpm = _FakeQpm(dqmap)
        ec = EventCorrelator.from_config(qpm, ds, "cpu")
        ec.process_dataset(ds)
        ec.post_process()
        payloads = ec.get_normalized_payloads(qpm, skip_scattering=True)
        payload = payloads[-1]

        # independently reproduce ring 0's ACF/CCF with the same core
        # functions used by test_event_core.py, and check EventCorrelator's
        # output matches exactly.
        ring_map = dqmap.astype(np.int64) - 1
        table = build_event_table(toa, pix, 1.0, npix, weights=np.ones_like(pix, dtype=np.float64))
        lag_max = table.n_bins // 2
        e_acf = log_lag_edges(1, lag_max, 8)
        e_ccf = log_lag_edges(0, lag_max, 8)
        ref_acf = correlate_pairs_torch(table, auto_pairs(ring_map, 0), e_acf, device="cpu")
        ref_ccf = correlate_pairs_torch(
            table, neighbour_pairs(ring_map, 0, "horizontal", True), e_ccf, device="cpu"
        )

        np.testing.assert_allclose(payload["event_g2_acf"][0], ref_acf.g2, equal_nan=True)
        np.testing.assert_allclose(payload["event_g2_ccf_h"][0], ref_ccf.g2, equal_nan=True)
        self.assertNotIn("event_g2_ccf_v", payload)

    def test_vh_produces_both_directions_matching_reference(self):
        rng = np.random.default_rng(1)
        pix, toa, dqmap, npix = _synthetic_ring(rng)
        bin_width_ns = 1.0
        ds = _FakeDataset(pix.astype(np.int64), np.floor(toa).astype(np.int64),
                          np.ones_like(pix), bin_width_ns, signal_spill="vh")

        qpm = _FakeQpm(dqmap)
        ec = EventCorrelator.from_config(qpm, ds, "cpu")
        ec.process_dataset(ds)
        ec.post_process()
        payloads = ec.get_normalized_payloads(qpm, skip_scattering=True)
        payload = payloads[-1]

        self.assertIn("event_g2_ccf_h", payload)
        self.assertIn("event_g2_ccf_v", payload)

        ring_map = dqmap.astype(np.int64) - 1
        table = build_event_table(toa, pix, 1.0, npix, weights=np.ones_like(pix, dtype=np.float64))
        lag_max = table.n_bins // 2
        e_ccf = log_lag_edges(0, lag_max, 8)
        ref_ccf_h = correlate_pairs_torch(
            table, neighbour_pairs(ring_map, 0, "horizontal", True), e_ccf, device="cpu"
        )
        ref_ccf_v = correlate_pairs_torch(
            table, neighbour_pairs(ring_map, 0, "vertical", True), e_ccf, device="cpu"
        )

        np.testing.assert_allclose(payload["event_g2_ccf_h"][0], ref_ccf_h.g2, equal_nan=True)
        np.testing.assert_allclose(payload["event_g2_ccf_v"][0], ref_ccf_v.g2, equal_nan=True)


class TestEventCorrelatorEmptyRing(unittest.TestCase):
    def test_empty_dq_bin_is_nan_filled_not_crashing(self):
        """A gapped/non-contiguous qmap (dq labels 1 and 3 present, dq=2
        empty) must not crash post_process(); the empty ring's row should
        be NaN-filled while preserving array shape and dq-label alignment
        for the other rings."""
        rng = np.random.default_rng(3)
        ny, nx, T, n_events = 3, 6, 200, 400
        npix = ny * nx
        toa = rng.uniform(0, T, n_events)
        pix = rng.integers(0, npix, n_events)

        # row 0 -> dq label 1 (ring index 0); rows 1-2 -> dq label 3 (ring
        # index 2). dq label 2 (ring index 1) never appears -> empty ring.
        dqmap = np.ones((ny, nx), dtype=np.int64)
        dqmap[ny // 2:, :] = 3
        self.assertEqual(set(np.unique(dqmap).tolist()), {1, 3})

        bin_width_ns = 1.0
        ds = _FakeDataset(pix.astype(np.int64), np.floor(toa).astype(np.int64),
                          np.ones_like(pix), bin_width_ns, signal_spill="h")
        qpm = _FakeQpm(dqmap)
        ec = EventCorrelator.from_config(qpm, ds, "cpu")
        self.assertEqual(ec.n_rings, 3)

        ec.process_dataset(ds)
        ec.post_process()  # must not raise
        payloads = ec.get_normalized_payloads(qpm, skip_scattering=True)
        payload = payloads[-1]

        g2_acf = payload["event_g2_acf"]
        g2_ccf_h = payload["event_g2_ccf_h"]
        self.assertEqual(g2_acf.shape[0], 3)
        self.assertEqual(g2_ccf_h.shape[0], 3)

        self.assertTrue(np.all(np.isnan(g2_acf[1])))
        self.assertTrue(np.all(np.isnan(g2_ccf_h[1])))
        self.assertFalse(np.any(np.isnan(g2_acf[0])))
        self.assertFalse(np.any(np.isnan(g2_acf[2])))

        ring_map = dqmap.astype(np.int64) - 1
        table = build_event_table(toa, pix, 1.0, npix, weights=np.ones_like(pix, dtype=np.float64))
        lag_max = table.n_bins // 2
        e_acf = log_lag_edges(1, lag_max, 8)
        e_ccf = log_lag_edges(0, lag_max, 8)
        for ring in (0, 2):
            ref_acf = correlate_pairs_torch(table, auto_pairs(ring_map, ring), e_acf, device="cpu")
            ref_ccf = correlate_pairs_torch(
                table, neighbour_pairs(ring_map, ring, "horizontal", True), e_ccf, device="cpu"
            )
            np.testing.assert_allclose(g2_acf[ring], ref_acf.g2, equal_nan=True)
            np.testing.assert_allclose(g2_ccf_h[ring], ref_ccf.g2, equal_nan=True)


class TestEventCorrelatorBinWidthInvariance(unittest.TestCase):
    """Regression test: time_bin from get_events() is already an integer bin
    index. Multiplying it into nanoseconds and letting build_event_table
    re-quantize by the same float bin width -- floor(n * b / b) -- is not an
    identity for non-dyadic b (e.g. 1.6 ns), which silently moved ~6% of the
    events one bin early and corrupted exactly the shortest lags."""

    @staticmethod
    def _run(pix, time_bin, count, bin_width_ns, dqmap):
        qpm = _FakeQpm(dqmap)
        ds = _FakeDataset(pix, time_bin, count, bin_width_ns, signal_spill="vh")
        ec = EventCorrelator.from_config(qpm, ds, "cpu")
        ec.process_dataset(ds)
        ec.post_process()
        return ec, ec.get_normalized_payloads(qpm, skip_scattering=True)[-1]

    def test_event_table_bins_are_the_input_time_bins_verbatim(self):
        rng = np.random.default_rng(11)
        pix, toa, dqmap, npix = _synthetic_ring(rng)
        pix = pix.astype(np.int64)
        time_bin = np.floor(toa).astype(np.int64)
        count = np.ones_like(pix)

        for bin_width_ns in (10.0, 1.6, 1e-8 * 1e9, np.float64(np.float32(1e-8)) * 1e9):
            with self.subTest(bin_width_ns=bin_width_ns):
                ec, _ = self._run(pix, time_bin, count, bin_width_ns, dqmap)
                # the table must hold exactly the integer bins handed in --
                # compare as (pixel, time) multisets, since build_event_table
                # re-sorts by (pixel, time).
                got = np.lexsort((ec._table.times, ec._table.pixel))
                want = np.lexsort((time_bin, pix))
                np.testing.assert_array_equal(ec._table.times[got], time_bin[want])
                np.testing.assert_array_equal(ec._table.pixel[got], pix[want])
                # and bin_width must still be carried through for the lag axis
                self.assertEqual(ec._table.bin_width, float(bin_width_ns))

    def test_correlation_is_identical_across_bin_widths_up_to_lag_scale(self):
        rng = np.random.default_rng(11)
        pix, toa, dqmap, npix = _synthetic_ring(rng)
        pix = pix.astype(np.int64)
        time_bin = np.floor(toa).astype(np.int64)
        count = np.ones_like(pix)

        nice_w, odd_w = 10.0, 1.6  # 1.6 ns is deliberately non-dyadic
        _, nice = self._run(pix, time_bin, count, nice_w, dqmap)
        _, odd = self._run(pix, time_bin, count, odd_w, dqmap)

        self.assertEqual(set(nice), set(odd))
        for key in ("event_g2_acf", "event_g2_acf_err",
                    "event_g2_ccf_h", "event_g2_ccf_h_err",
                    "event_g2_ccf_v", "event_g2_ccf_v_err"):
            # bit-identical: the correlation only depends on integer bin
            # indices, never on the physical width of a bin.
            np.testing.assert_array_equal(nice[key], odd[key], err_msg=key)

        # only the lag axis differs, and only by the bin-width ratio
        for key in ("event_delay_list_acf", "event_delay_list_ccf"):
            np.testing.assert_allclose(
                np.asarray(odd[key]),
                np.asarray(nice[key]) * (odd_w / nice_w),
                rtol=1e-12, err_msg=key,
            )
        self.assertEqual(nice["event_run_info"]["bin_width_ns"], nice_w)
        self.assertEqual(odd["event_run_info"]["bin_width_ns"], odd_w)


class TestEventCorrelatorScatteringDevice(unittest.TestCase):
    """Regression test for the GPU crash: the scattering tensors handed to
    qpm.normalize_scattering must live on the correlator's device, or
    average_with_index_map's torch.bincount(qmap, weights=...) raises
    'Expected all tensors to be on the same device' on every GPU run."""

    def _check(self, device):
        rng = np.random.default_rng(5)
        pix, toa, dqmap, npix = _synthetic_ring(rng)
        qpm = _FakeQpm(dqmap)
        ds = _FakeDataset(pix.astype(np.int64), np.floor(toa).astype(np.int64),
                          np.ones_like(pix), 1.0, signal_spill="h")
        ec = EventCorrelator.from_config(qpm, ds, device)
        ec.process_dataset(ds)
        ec.post_process()
        ec.get_normalized_payloads(qpm)  # skip_scattering=False

        seen = qpm.scattering_seen
        self.assertIsNotNone(seen)
        want = torch.device(device)
        for key in ("saxs_2d", "saxs_2d_par", "intensity_vs_time"):
            self.assertEqual(seen[key].device, want, msg=key)

    def test_cpu(self):
        self._check("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA device available")
    def test_cuda(self):
        self._check("cuda:0")


class TestEventCorrelatorNoRings(unittest.TestCase):
    def test_all_zero_dqmap_raises_configuration_error(self):
        """An entirely masked-out/zero dqmap used to reach np.stack([]) and
        die with a bare ValueError; it must report the qmap problem instead."""
        rng = np.random.default_rng(13)
        pix, toa, _dqmap, npix = _synthetic_ring(rng)
        dqmap = np.zeros((3, 6), dtype=np.int64)
        ds = _FakeDataset(pix.astype(np.int64), np.floor(toa).astype(np.int64),
                          np.ones_like(pix), 1.0, signal_spill="h")
        ec = EventCorrelator.from_config(_FakeQpm(dqmap), ds, "cpu")
        self.assertEqual(ec.n_rings, 0)
        ec.process_dataset(ds)
        with self.assertRaises(ConfigurationError):
            ec.post_process()

    def test_fully_masked_dqmap_raises_configuration_error(self):
        """Same, but reached via the mask rather than the dq labels."""
        rng = np.random.default_rng(13)
        pix, toa, dqmap, npix = _synthetic_ring(rng)
        mask = np.zeros(dqmap.shape, dtype=bool)
        ds = _FakeDataset(pix.astype(np.int64), np.floor(toa).astype(np.int64),
                          np.ones_like(pix), 1.0, signal_spill="h")
        ec = EventCorrelator.from_config(_FakeQpm(dqmap, mask), ds, "cpu")
        self.assertEqual(ec.n_rings, 0)
        ec.process_dataset(ds)
        with self.assertRaises(ConfigurationError):
            ec.post_process()


class TestEventCorrelatorDescribe(unittest.TestCase):
    def test_describe_does_not_log_a_bogus_event_count(self):
        """run_jobs calls describe() before process_dataset() builds the
        table, so the old 'events loaded: 0' line was always wrong."""
        dqmap = np.ones((2, 2), dtype=np.int64)
        ds = _FakeDataset(np.array([0]), np.array([0]), np.array([1]), 1.0,
                          signal_spill="h")
        ec = EventCorrelator.from_config(_FakeQpm(dqmap), ds, "cpu")
        with self.assertLogs("boost_corr.correlator.event", level="INFO") as cm:
            ec.describe()
        self.assertEqual(len(cm.output), 1)
        self.assertNotIn("events loaded", "\n".join(cm.output))
        self.assertIn("1 rings", cm.output[0])


if __name__ == "__main__":
    unittest.main()
