#!/usr/bin/env python
"""Brute-force checks of the vendored event-correlator core."""
import unittest

import numpy as np

from boost_corr.correlator._event_core import (
    build_event_table, correlate_pairs_torch, auto_pairs, neighbour_pairs,
    log_lag_edges,
)


def dense_intensity(table):
    I = np.zeros((table.n_bins, table.npix))
    np.add.at(I, (table.times, table.pixel), table.weights)
    return I


def brute_force(I, pairs, edges, normalization):
    T, npix = I.shape
    pairs = np.asarray(pairs)
    M = len(pairs)
    K = len(edges) - 1
    G = np.zeros(K); SL = np.zeros(K); SR = np.zeros(K)
    P = np.zeros(K); W = np.zeros(K)
    mL = np.bincount(pairs[:, 0], minlength=npix)
    mR = np.bincount(pairs[:, 1], minlength=npix)
    IL = (I * mL).sum(axis=1)
    IR = (I * mR).sum(axis=1)
    for k in range(K):
        for tau in range(edges[k], min(edges[k + 1], T)):
            W[k] += T - tau
            for p, q in pairs:
                x, y = I[:T - tau, p], I[tau:, q]
                if p == q and tau == 0:
                    G[k] += (x * (x - 1)).sum()
                else:
                    G[k] += (x * y).sum()
                SL[k] += x.sum()
                SR[k] += y.sum()
            pk = (IL[:T - tau] * IR[tau:]).sum()
            if tau == 0:
                pk -= (I * mL * mR).sum()
            P[k] += pk
    if normalization == "symmetric":
        return (G / (M * W)) / ((SL / (M * W)) * (SR / (M * W)))
    return (G / M) / (P / M ** 2)


def random_table(rng, ny=4, nx=5, T=60, n_events=150):
    npix = ny * nx
    toa = rng.uniform(0, T, n_events)
    pix = rng.integers(0, npix, n_events)
    w = rng.integers(1, 4, n_events)
    return build_event_table(toa, pix, 1.0, npix, weights=w, duration=T)


class TestEventCoreAgainstBruteForce(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        ny, nx = 4, 5
        ring_map = np.zeros((ny, nx), int)
        ring_map[:, 0] = -1
        for trial in range(5):
            table = random_table(rng, ny, nx)
            I = dense_intensity(table)
            edges = log_lag_edges(0, 55, 6)
            for pairs in (
                auto_pairs(ring_map, 0),
                neighbour_pairs(ring_map, 0, "horizontal", True),
                neighbour_pairs(ring_map, 0, "vertical", False),
            ):
                for norm in ("symmetric", "pixel_mean"):
                    got = correlate_pairs_torch(table, pairs, edges, norm, device="cpu")
                    ref = brute_force(I, pairs, edges, norm)
                    ok = np.isfinite(ref)
                    np.testing.assert_allclose(
                        got.g2[ok], ref[ok], rtol=1e-10, atol=1e-10,
                        err_msg=f"trial={trial} norm={norm}",
                    )
                    self.assertTrue(np.array_equal(np.isfinite(got.g2), ok))


if __name__ == "__main__":
    unittest.main()
