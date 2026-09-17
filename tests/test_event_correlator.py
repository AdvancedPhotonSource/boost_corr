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
    """Minimal stand-in for XpcsQPartitionMap: only .dqmap is used by
    EventCorrelator.from_config; normalize_scattering is exercised
    directly against the real XpcsQPartitionMap in test_event_result_schema.py."""
    def __init__(self, dqmap):
        self.dqmap = dqmap


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


if __name__ == "__main__":
    unittest.main()
