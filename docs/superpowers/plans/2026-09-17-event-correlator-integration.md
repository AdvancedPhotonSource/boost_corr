# Event Correlator Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third correlation engine, `EventCorrelator`, that computes g2 directly from photon arrival-time events (ACF + directional CCF) instead of dense per-frame binning, and route to it automatically whenever `--signal-spill` is not `none`/`None` — with two-time correlation explicitly disallowed on that path.

**Architecture:** Vendor the already-validated event-correlator algorithm (torch kernels + pixel-pair/lag-edge geometry) into `boost_corr/correlator/_event_core.py`. Wrap it in an `EventCorrelator` class that implements the same informal protocol as `MultitauCorrelator`/`TwotimeCorrelator` (`from_config`, `reset`, `process_dataset`, `post_process`, `get_normalized_payloads`, `describe`) but internally does a single bulk `dset.get_events()` call instead of iterating dense frame batches. Wire `"Event"` into `xpcs_aps_8idi/correlation/common.py`'s existing `_CORRELATOR_CLASSES`/`_VALID_TYPES`/`_build_run_args` dispatch, gated by a `use_event = signal_spill not in (None, "none")` check in `solve_correlation()`.

**Tech Stack:** Python 3.12, PyTorch (GPU via CUDA/XPU or CPU fallback), NumPy, h5py, pytest/unittest (existing suite uses `unittest.TestCase`).

**Spec:** `docs/superpowers/specs/2026-09-17-event-correlator-integration-design.md`

## Global Constraints

- Scope is **Rigaku-family (RigakuDataset, RigakuExtendDataset) and single-chip Timepix4 only**. `Rigaku3MDataset` and multi-chip `TimepixAnalysisDataset` are explicitly out of scope for this plan (unresolved pixel-remapping geometry + cross-module clock sync, per spec Risks section) — calling the event correlator on them must raise `exc.DatasetError`, not silently produce wrong results.
- No new required CLI flags. `--signal-spill {v,h,vh}` alone triggers the event correlator; its value also selects CCF direction (`h`→horizontal, `v`→vertical, `vh`→both, reported as two separate g2 curves, never pooled).
- `--signal-spill != none` combined with an explicit `--type Twotime` or `--type Both` must raise `exc.ConfigurationError` (exit code 78) — this is a hard error, not a silent override.
- New output HDF5 keys live under `/xpcs/event/...` (`aps_8idi.py`'s `key` dict) — never reuse `/xpcs/multitau/...` paths.
- `EventCorrelator` production code path is torch-only (CPU fallback via `device="cpu"`, same as the rest of boost_corr); no numpy production implementation is vendored — tests validate against a brute-force reference instead (same discipline the sibling scratch project used).
- This is confirmed as a deliberate backward-compatibility change: any existing `--signal-spill {v,h,vh}` + default/explicit `--type Multitau` invocation on Rigaku-family data will now run the event correlator instead of multitau.
- Follow existing test conventions: `unittest.TestCase`, `tempfile.mkdtemp()`/`shutil.rmtree` for file fixtures, a local `pack(...)` helper to synthesize sparse binary words (see `tests/test_rigaku_extend.py`).

---

## Task 1: Vendor the event-correlator core algorithm

**Files:**
- Create: `src/boost_corr/correlator/_event_core.py`
- Test: `tests/test_event_core.py`

**Interfaces:**
- Consumes: nothing from other tasks (this is the foundation).
- Produces (used by Task 4): `EventTable` (dataclass with fields `times`, `weights`, `pixel`, `offsets`, `n_bins`, `bin_width`; properties `npix`, `n_events`; method `pixel_events(p)`), `build_event_table(toa, pixel, bin_width, npix, weights=None, duration=None) -> EventTable`, `log_lag_edges(lag_min, lag_max, per_decade=8) -> np.ndarray`, `auto_pairs(ring_map, ring) -> np.ndarray`, `neighbour_pairs(ring_map, ring, direction="horizontal", symmetric=True) -> np.ndarray`, `CorrelationResult` (dataclass with fields `lag`, `g2`, `err`, `edges`, `pairs`, `n_pairs`, `normalization`; method `valid()` returning a boolean mask of finite `g2` entries), `upload_table(table, device=None) -> (times_t, weights_t, pixel_t)`, `correlate_pairs_torch(table, pairs, edges, normalization="symmetric", device=None, block=50_000, device_table=None) -> CorrelationResult`.

- [ ] **Step 1: Write `_event_core.py`**

```python
"""
Vendored event-driven intensity-correlation core, following

    Chushkin, Correa, Ignatenko, Pennicard, Lange, Fridman, Karl,
    Senfftleben, Lehmkuehler, Westermeier, Graafsma & Cammarata (2025).
    "Achieving nanosecond time resolution with a two-dimensional X-ray
    detector", J. Synchrotron Rad. 32, 1220-1227.

and the event correlator of Chushkin, Caronna & Madsen (2012), J. Appl.
Cryst. 45, 807. This module is a torch-only port validated against a
numpy reference and a brute-force reference in a sibling project; see
docs/superpowers/specs/2026-09-17-event-correlator-integration-design.md.
Only the algorithm lives here -- boost_corr-specific wiring (dataset
access, qmap, output payloads) lives in event.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "EventTable",
    "build_event_table",
    "log_lag_edges",
    "auto_pairs",
    "neighbour_pairs",
    "ring_pixels",
    "CorrelationResult",
    "upload_table",
    "correlate_pairs_torch",
]


@dataclass
class EventTable:
    """Photon events sorted by (pixel, time) with CSR-style pixel offsets."""

    times: np.ndarray
    weights: np.ndarray
    pixel: np.ndarray
    offsets: np.ndarray
    n_bins: int
    bin_width: float = 1.0

    @property
    def npix(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_events(self) -> int:
        return len(self.times)

    def pixel_events(self, p: int):
        sl = slice(self.offsets[p], self.offsets[p + 1])
        return self.times[sl], self.weights[sl]


def build_event_table(toa, pixel, bin_width, npix, weights=None,
                      duration=None) -> EventTable:
    """Quantise arrival times and sort the events by (pixel, time)."""
    toa = np.asarray(toa, dtype=np.float64)
    pixel = np.asarray(pixel, dtype=np.int64)
    t = np.floor(toa / bin_width).astype(np.int64)
    if t.size and t.min() < 0:
        raise ValueError("negative arrival times")
    w = (np.ones(t.size, dtype=np.float64) if weights is None
         else np.asarray(weights, dtype=np.float64))
    order = np.lexsort((t, pixel))
    t, w, pixel = t[order], w[order], pixel[order]
    offsets = np.searchsorted(pixel, np.arange(npix + 1))
    if duration is None:
        n_bins = int(t.max()) + 1 if t.size else 0
    else:
        n_bins = int(np.ceil(duration / bin_width))
        if t.size and t.max() >= n_bins:
            raise ValueError("events later than `duration`")
    return EventTable(t, w, pixel, offsets, n_bins, float(bin_width))


def log_lag_edges(lag_min: int, lag_max: int, per_decade: int = 8) -> np.ndarray:
    """Integer lag-bin edges, quasi-logarithmically spaced (in time bins)."""
    if lag_min < 0:
        raise ValueError("lag_min must be >= 0")
    lo = max(lag_min, 1)
    n = int(np.ceil(np.log10(lag_max / lo) * per_decade)) + 1
    e = np.round(np.logspace(np.log10(lo), np.log10(lag_max), n)).astype(np.int64)
    if lag_min == 0:
        e = np.concatenate(([0], e))
    return np.unique(e)


def _lag_bin_centres(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:] - 1)


def ring_pixels(ring_map, ring) -> np.ndarray:
    return np.flatnonzero(np.asarray(ring_map).ravel() == ring)


def auto_pairs(ring_map, ring) -> np.ndarray:
    """(p, p) pairs for the ACF."""
    p = ring_pixels(ring_map, ring)
    return np.stack([p, p], axis=1)


def neighbour_pairs(ring_map, ring, direction="horizontal", symmetric=True) -> np.ndarray:
    """(p, p+D) pairs for the CCF. direction: "horizontal" or "vertical"."""
    ring_map = np.asarray(ring_map)
    ny, nx = ring_map.shape
    dy, dx = {"horizontal": (0, 1), "vertical": (1, 0)}.get(direction, direction)
    iy, ix = np.nonzero(ring_map == ring)
    jy, jx = iy + dy, ix + dx
    ok = (jy >= 0) & (jy < ny) & (jx >= 0) & (jx < nx)
    iy, ix, jy, jx = iy[ok], ix[ok], jy[ok], jx[ok]
    ok = ring_map[jy, jx] == ring
    p = iy[ok] * nx + ix[ok]
    q = jy[ok] * nx + jx[ok]
    pairs = np.stack([p, q], axis=1)
    if symmetric:
        pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    return pairs


@dataclass
class CorrelationResult:
    lag: np.ndarray
    g2: np.ndarray
    err: np.ndarray
    edges: np.ndarray
    pairs: np.ndarray
    n_pairs: int
    normalization: str

    def valid(self):
        return np.isfinite(self.g2)


def _pair_counts_torch(a, wa, b, wb, edges, block=50_000):
    K = len(edges) - 1
    out = torch.zeros(K, dtype=torch.float64, device=a.device)
    if a.numel() == 0 or b.numel() == 0:
        return out
    cum = torch.cat([wb.new_zeros(1), torch.cumsum(wb, 0)])
    for s in range(0, a.numel(), block):
        aa = a[s:s + block]
        idx = torch.searchsorted(b, aa[:, None] + edges[None, :], right=False)
        out += (wa[s:s + block, None] * torch.diff(cum[idx], dim=1)).sum(dim=0)
    return out


def _window_sums_torch(t, w, edges, T, side, block=50_000):
    K = len(edges) - 1
    out = torch.zeros(K, dtype=torch.float64, device=t.device)
    lo = edges[:-1][None, :]
    hi = edges[1:][None, :]
    for s in range(0, t.numel(), block):
        tt = t[s:s + block]
        lim = (T - tt) if side == "left" else (tt + 1)
        n = torch.clamp(torch.minimum(hi, lim[:, None]) - lo, min=0)
        out += (w[s:s + block, None] * n).sum(dim=0)
    return out


def _lag_weights_torch(edges, T):
    lo = edges[:-1].to(torch.float64)
    hi = torch.clamp(edges[1:], max=T).to(torch.float64)
    n = torch.clamp(hi - lo, min=0)
    return n * (2.0 * T - lo - hi + 1.0) / 2.0


def _pooled_torch(times, weights, pixel, mult):
    m = mult[pixel]
    keep = m > 0
    t = times[keep]
    w = weights[keep] * m[keep].to(torch.float64)
    order = torch.argsort(t, stable=True)
    return t[order], w[order]


def upload_table(table: EventTable, device=None):
    """Upload table.times/weights/pixel to `device` once. Pass the result
    back into correlate_pairs_torch(..., device_table=...) for every
    ring/pair-list computed from the same table to avoid re-uploading the
    (often multi-GB) bulk event arrays on every call."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return (
        torch.as_tensor(table.times, device=device),
        torch.as_tensor(table.weights, dtype=torch.float64, device=device),
        torch.as_tensor(table.pixel, device=device),
    )


def correlate_pairs_torch(table: EventTable, pairs, edges,
                          normalization: str = "symmetric",
                          device=None, block: int = 50_000,
                          device_table=None) -> CorrelationResult:
    """Event correlator for an explicit list of pixel pairs (p, p')."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs_np = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    edges_np = np.asarray(edges, dtype=np.int64)
    if np.any(np.diff(edges_np) <= 0) or edges_np[0] < 0:
        raise ValueError("edges must be strictly increasing and >= 0")
    T = table.n_bins
    M = len(pairs_np)
    K = len(edges_np) - 1
    if M == 0:
        raise ValueError("no pixel pairs")

    edges_t = torch.as_tensor(edges_np, device=device)
    if device_table is None:
        device_table = upload_table(table, device)
    times_t, weights_t, pixel_t = device_table

    G = torch.zeros(K, dtype=torch.float64, device=device)
    for p, q in pairs_np:
        sl_p = slice(int(table.offsets[p]), int(table.offsets[p + 1]))
        sl_q = slice(int(table.offsets[q]), int(table.offsets[q + 1]))
        a, wa = times_t[sl_p], weights_t[sl_p]
        b, wb = times_t[sl_q], weights_t[sl_q]
        if a.numel() == 0 or b.numel() == 0:
            continue
        G += _pair_counts_torch(a, wa, b, wb, edges_t, block)
        if p == q and edges_np[0] == 0:
            G[0] -= wa.sum()

    pairs_t = torch.as_tensor(pairs_np, device=device)
    mL = torch.bincount(pairs_t[:, 0], minlength=table.npix)
    mR = torch.bincount(pairs_t[:, 1], minlength=table.npix)
    tL, wL = _pooled_torch(times_t, weights_t, pixel_t, mL)
    tR, wR = _pooled_torch(times_t, weights_t, pixel_t, mR)
    W = _lag_weights_torch(edges_t, T)

    if normalization == "symmetric":
        SL = _window_sums_torch(tL, wL, edges_t, T, "left", block)
        SR = _window_sums_torch(tR, wR, edges_t, T, "right", block)
        g2 = G * M * W / (SL * SR)
    elif normalization == "pixel_mean":
        P = _pair_counts_torch(tL, wL, tR, wR, edges_t, block)
        if edges_np[0] == 0:
            P[0] -= torch.sum(mL[pixel_t].to(torch.float64)
                              * mR[pixel_t].to(torch.float64) * weights_t)
        g2 = G * M / P
    else:
        raise ValueError("normalization must be 'symmetric' or 'pixel_mean'")
    err = g2 / torch.sqrt(G)

    lag = _lag_bin_centres(edges_np) * table.bin_width
    return CorrelationResult(
        lag, g2.cpu().numpy(), err.cpu().numpy(), edges_np,
        G.cpu().numpy(), M, normalization,
    )
```

- [ ] **Step 2: Write the brute-force parity test**

```python
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
```

- [ ] **Step 3: Run the test**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_event_core.py -v`
Expected: PASS (5 trials x 3 pair-sets x 2 normalizations, all within `rtol=1e-10`).

- [ ] **Step 4: Commit**

```bash
git add src/boost_corr/correlator/_event_core.py tests/test_event_core.py
git commit -m "feat: vendor event-driven correlator core algorithm"
```

---

## Task 2: Bin-width resolution helper + `RigakuDataset.get_events()`

**Files:**
- Modify: `src/boost_corr/xpcs_aps_8idi/dataset/utils.py`
- Modify: `src/boost_corr/xpcs_aps_8idi/dataset/rigaku_handler.py`
- Test: `tests/test_rigaku_get_events.py`

**Interfaces:**
- Consumes: nothing from Task 1 (dataset layer is independent of the algorithm layer).
- Produces (used by Task 4/6/7): `resolve_bin_width_ns(raw_fname: str, bin_time_s: float) -> float` in `dataset/utils.py`. `RigakuDataset.get_events() -> (pixel: np.ndarray[int64], time_bin: np.ndarray[int64], count: np.ndarray[int64], bin_width_ns: float)`, inherited unchanged by `RigakuExtendDataset` (it overrides `read_data()` only, so `self.ifc` is already built from the correct format by the time `get_events()` reads it).

- [ ] **Step 1: Add `resolve_bin_width_ns` to `dataset/utils.py`**

Current top of `src/boost_corr/xpcs_aps_8idi/dataset/utils.py`:
```python
import os
import h5py
from .imm_handler import ImmDataset
```

Change to:
```python
import glob
import logging
import os

import h5py
import numpy as np

from .imm_handler import ImmDataset
```

Add this function after the imports, before `def create_dataset(raw_fname, **kwargs):`:

```python
logger = logging.getLogger(__name__)


def resolve_bin_width_ns(raw_fname: str, bin_time_s: float) -> float:
    """Resolve the physical time-bin width (ns) for a Rigaku-family sparse
    file's "frame" field, which has no self-describing bin width.

    Prefers the sibling `*_metadata.hdf`'s
    `entry/instrument/detector_1/frame_time` (seconds); falls back to
    `bin_time_s` (also seconds, e.g. from --bin-time-s) if no metadata file
    or field is found. Logs a warning if both are present and disagree.
    """
    fallback_ns = bin_time_s * 1e9
    meta_dir = os.path.dirname(os.path.abspath(raw_fname))
    for meta_fname in sorted(glob.glob(os.path.join(meta_dir, "*_metadata.hdf"))):
        try:
            with h5py.File(meta_fname, "r") as f:
                path = "/entry/instrument/detector_1/frame_time"
                if path not in f:
                    continue
                meta_ns = float(f[path][()]) * 1e9
        except OSError:
            continue
        if not np.isclose(meta_ns, fallback_ns, rtol=1e-3):
            logger.warning(
                f"bin width mismatch: metadata {meta_fname} says {meta_ns:.3f} ns, "
                f"--bin-time-s says {fallback_ns:.3f} ns; using metadata value"
            )
        return meta_ns
    return fallback_ns
```

- [ ] **Step 2: Write the failing test for `resolve_bin_width_ns`**

```python
#!/usr/bin/env python
"""Tests for RigakuDataset.get_events() and bin-width resolution."""
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from boost_corr.xpcs_aps_8idi.dataset.utils import resolve_bin_width_ns
from boost_corr.xpcs_aps_8idi.dataset.rigaku_handler import RigakuDataset


def pack_rigaku(count, pixel, frame):
    return (
        (np.uint64(frame) << np.uint64(40))
        + (np.uint64(pixel) << np.uint64(16))
        + np.uint64(count)
    )


class TestResolveBinWidthNs(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.raw_fname = os.path.join(self.tmpdir, "run_metatest.bin")
        open(self.raw_fname, "wb").close()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_falls_back_to_bin_time_s_when_no_metadata(self):
        bin_width_ns = resolve_bin_width_ns(self.raw_fname, bin_time_s=1e-6)
        self.assertAlmostEqual(bin_width_ns, 1000.0)

    def test_reads_metadata_frame_time_when_present(self):
        meta_fname = os.path.join(self.tmpdir, "run_metatest_metadata.hdf")
        with h5py.File(meta_fname, "w") as f:
            f["/entry/instrument/detector_1/frame_time"] = 1e-8
        bin_width_ns = resolve_bin_width_ns(self.raw_fname, bin_time_s=1e-6)
        self.assertAlmostEqual(bin_width_ns, 10.0)


class TestRigakuGetEvents(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fname = os.path.join(self.tmpdir, "sample.bin")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_bin(self, events):
        # events: list of (pixel, frame, count), must be frame-sorted
        words = np.array(
            [pack_rigaku(count, pixel, frame) for pixel, frame, count in events],
            dtype=np.uint64,
        )
        words.tofile(self.fname)

    def test_get_events_matches_raw_sparse_triple(self):
        self._write_bin([(1, 0, 3), (3, 0, 1), (0, 1, 2)])
        ds = RigakuDataset(
            self.fname, det_size=(2, 2), batch_size=2, device="cpu",
            signal_spill="none", bin_time_s=1e-6,
        )
        pixel, time_bin, count, bin_width_ns = ds.get_events()
        np.testing.assert_array_equal(pixel, [1, 3, 0])
        np.testing.assert_array_equal(time_bin, [0, 0, 1])
        np.testing.assert_array_equal(count, [3, 1, 2])
        self.assertAlmostEqual(bin_width_ns, 1000.0)  # 1e-6 s -> 1000 ns fallback

    def test_get_events_dtype_is_int64(self):
        self._write_bin([(0, 0, 1)])
        ds = RigakuDataset(
            self.fname, det_size=(2, 2), batch_size=2, device="cpu",
            signal_spill="none",
        )
        pixel, time_bin, count, _ = ds.get_events()
        self.assertEqual(pixel.dtype, np.int64)
        self.assertEqual(time_bin.dtype, np.int64)
        self.assertEqual(count.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_rigaku_get_events.py -v`
Expected: FAIL — `resolve_bin_width_ns` not defined, `RigakuDataset` has no attribute `get_events`.

- [ ] **Step 4: Add `get_events()` to `RigakuDataset`**

In `src/boost_corr/xpcs_aps_8idi/dataset/rigaku_handler.py`, add this method to the `RigakuDataset` class, immediately after `__getbatch__`:

```python
    def get_events(self):
        """Return (pixel, time_bin, count, bin_width_ns) for the event
        correlator: int64 numpy arrays over the full (uncropped) pixel
        space, sorted by time_bin (as stored in self.ifc). Inherited
        unchanged by RigakuExtendDataset -- it overrides read_data() only,
        so self.ifc is already built from the correct bit layout by the
        time this runs.
        """
        # imported lazily: dataset/utils.py imports RigakuDataset at module
        # level, so a top-level import here would be circular.
        from .utils import resolve_bin_width_ns

        pixel = self.ifc[0].detach().cpu().numpy().astype(np.int64)
        time_bin = self.ifc[1].detach().cpu().numpy().astype(np.int64)
        count = self.ifc[2].detach().cpu().numpy().astype(np.int64)
        bin_width_ns = resolve_bin_width_ns(self.fname, self.bin_time_s)
        return pixel, time_bin, count, bin_width_ns
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_rigaku_get_events.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/boost_corr/xpcs_aps_8idi/dataset/utils.py src/boost_corr/xpcs_aps_8idi/dataset/rigaku_handler.py tests/test_rigaku_get_events.py
git commit -m "feat: add get_events() to RigakuDataset for the event correlator"
```

---

## Task 3: `TimepixAnalysisDataset.get_events()` (single-chip only)

**Files:**
- Modify: `src/boost_corr/xpcs_aps_8idi/dataset/timepix4_handler.py`
- Test: `tests/test_timepix4_get_events.py`

**Interfaces:**
- Consumes: nothing from Tasks 1/2.
- Produces (used by Task 4/6/7): `TimepixAnalysisDataset.get_events() -> (pixel, time_bin, count, bin_width_ns)`, raising `exc.DatasetError` when `self.raw_dataset.num_chips != 1`.

- [ ] **Step 1: Write the failing test**

```python
#!/usr/bin/env python
"""Tests for TimepixAnalysisDataset.get_events(). Constructs a bare
instance (bypassing __init__, which needs the external timepix_dataset
package and a real file) and sets only the attributes get_events() reads,
so this test has no dependency on real Timepix4 hardware files."""
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_timepix4_get_events.py -v`
Expected: FAIL — `TimepixAnalysisDataset` has no attribute `get_events`.

- [ ] **Step 3: Add `get_events()` to `TimepixAnalysisDataset`**

In `src/boost_corr/xpcs_aps_8idi/dataset/timepix4_handler.py`, add this import at the top (after the existing imports):

```python
import boost_corr.xpcs_aps_8idi.exceptions as exc
```

Add this method to the `TimepixAnalysisDataset` class, immediately after `__getbatch__`:

```python
    def get_events(self):
        """Return (pixel, time_bin, count, bin_width_ns) for the event
        correlator. Only single-chip (num_chips == 1) datasets are
        supported in this version: multi-chip datasets need each module's
        pixel index remapped through raw_dataset.layout, which is not yet
        implemented (see docs/superpowers/specs/2026-09-17-event-correlator-
        integration-design.md, Risks section).
        """
        if self.raw_dataset.num_chips != 1:
            raise exc.DatasetError(
                "get_events() is only supported for single-chip Timepix4 "
                f"datasets (this dataset has {self.raw_dataset.num_chips} "
                "chips); use --signal-spill none to run multitau/twotime "
                "instead."
            )
        index, frame, count = self.ifc_list[0]
        pixel = index.detach().cpu().numpy().astype(np.int64)
        time_bin = frame.detach().cpu().numpy().astype(np.int64)
        count = count.detach().cpu().numpy().astype(np.int64)
        bin_width_ns = self.bin_time_s * 1e9
        return pixel, time_bin, count, bin_width_ns
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_timepix4_get_events.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/boost_corr/xpcs_aps_8idi/dataset/timepix4_handler.py tests/test_timepix4_get_events.py
git commit -m "feat: add get_events() to single-chip TimepixAnalysisDataset"
```

---

## Task 4: `EventCorrelator` class

**Files:**
- Create: `src/boost_corr/correlator/event.py`
- Modify: `src/boost_corr/correlator/__init__.py`
- Test: `tests/test_event_correlator.py`

**Interfaces:**
- Consumes: everything from Task 1 (`build_event_table`, `log_lag_edges`, `auto_pairs`, `neighbour_pairs`, `upload_table`, `correlate_pairs_torch` from `boost_corr.correlator._event_core`) and the `get_events()` contract from Tasks 2/3 (any `dset` with a `get_events() -> (pixel, time_bin, count, bin_width_ns)` method and a `signal_spill` attribute in `{"h", "v", "vh"}`).
- Produces (used by Tasks 5/6/7): `EventCorrelator` class with `from_config(cls, qpm, dset, device, **kwargs)`, `reset()`, `describe()`, `process_dataset(dset, verbose=False, use_loader=False, num_workers=0)`, `post_process(**kwargs)`, `get_normalized_payloads(qpm, skip_scattering=False, **kwargs) -> list[dict]`. Payload dict keys: `event_run_info`, `event_delay_list_acf`, `event_g2_acf`, `event_g2_acf_err`, and (depending on selected directions) `event_delay_list_ccf`, `event_g2_ccf_h[_err]`, `event_g2_ccf_v[_err]`. Also exported as `boost_corr.correlator.EventCorrelator`.

- [ ] **Step 1: Write the failing test**

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_event_correlator.py -v`
Expected: FAIL — `boost_corr.correlator.event` module does not exist.

- [ ] **Step 3: Write `event.py`**

```python
"""EventCorrelator: computes g2 directly from photon arrival-time
differences (ACF, same-pixel) and nearest-neighbour-pixel cross-correlation
(CCF), bypassing the dense per-frame binning MultitauCorrelator/
TwotimeCorrelator need. See
docs/superpowers/specs/2026-09-17-event-correlator-integration-design.md.
"""
import logging

import numpy as np
import torch

import boost_corr.xpcs_aps_8idi.exceptions as exc
from ._event_core import (
    build_event_table,
    log_lag_edges,
    auto_pairs,
    neighbour_pairs,
    upload_table,
    correlate_pairs_torch,
)

logger = logging.getLogger(__name__)

_DIRECTIONS_BY_SIGNAL_SPILL = {
    "h": ("horizontal",),
    "v": ("vertical",),
    "vh": ("horizontal", "vertical"),
}


class EventCorrelator:
    def __init__(self, ring_map, directions, device="cpu", per_decade=8):
        self.ring_map = np.asarray(ring_map, dtype=np.int64)
        self.det_size = self.ring_map.shape
        self.n_rings = int(self.ring_map.max()) + 1 if self.ring_map.size else 0
        self.directions = directions
        self.device = device
        self.per_decade = per_decade
        # the event correlator always operates on the full (uncropped)
        # pixel space -- mask_crop is only meaningful for the dense-frame
        # path multitau/twotime use.
        self.mask_crop = None
        # always None: common.py::prepare_correlator() compares
        # existing.frame_num/.det_size to the next job's dataset and only
        # reuses the correlator instance on a match. None != any int, so
        # this always fails the comparison and forces a fresh EventCorrelator
        # per job -- correct, since each job's event table differs even
        # when det_size repeats across jobs. Must stay a real attribute
        # (not omitted) or prepare_correlator raises AttributeError on the
        # second job of any multi-file or --num-segments > 1 run.
        self.frame_num = None

        self._table = None
        self._device_table = None
        self._results = None

    @classmethod
    def from_config(cls, qpm, dset, device, **kwargs):
        if not hasattr(dset, "get_events"):
            raise exc.DatasetError(
                f"{type(dset).__name__} does not support the event correlator "
                "(no get_events() method); use --signal-spill none to run "
                "multitau/twotime instead."
            )
        signal_spill = getattr(dset, "signal_spill", None)
        if signal_spill not in _DIRECTIONS_BY_SIGNAL_SPILL:
            raise exc.ConfigurationError(
                f"signal_spill={signal_spill!r} does not select a CCF "
                "direction; expected 'h', 'v', or 'vh'."
            )
        directions = _DIRECTIONS_BY_SIGNAL_SPILL[signal_spill]
        ring_map = qpm.dqmap.astype(np.int64) - 1
        return cls(ring_map, directions, device=device,
                   per_decade=kwargs.get("per_decade", 8))

    def reset(self):
        self._table = None
        self._device_table = None
        self._results = None

    def describe(self):
        n_events = self._table.n_events if self._table is not None else 0
        logger.info(
            f"event correlator: {self.n_rings} rings, directions={self.directions}"
        )
        logger.info(f"events loaded: {n_events}")

    def process_dataset(self, dset, verbose=False, use_loader=False, num_workers=0):
        pixel, time_bin, count, bin_width_ns = dset.get_events()
        npix = self.det_size[0] * self.det_size[1]
        toa_ns = time_bin.astype(np.float64) * bin_width_ns
        self._table = build_event_table(
            toa_ns, pixel, bin_width_ns, npix, weights=count.astype(np.float64)
        )
        self._device_table = upload_table(self._table, self.device)
        if verbose:
            logger.info(
                f"event table built: {self._table.n_events} events, "
                f"{self._table.n_bins} bins, bin_width={bin_width_ns} ns"
            )

    def post_process(self, **kwargs):
        table = self._table
        lag_max = max(table.n_bins // 2, 1)
        edges_acf = log_lag_edges(1, lag_max, self.per_decade)
        edges_ccf = log_lag_edges(0, lag_max, self.per_decade)

        acf_lag, acf_g2, acf_err = None, [], []
        ccf = {d: {"lag": None, "g2": [], "err": []} for d in self.directions}

        for ring in range(self.n_rings):
            result = correlate_pairs_torch(
                table, auto_pairs(self.ring_map, ring), edges_acf,
                device=self.device, device_table=self._device_table,
            )
            acf_lag = result.lag
            acf_g2.append(result.g2)
            acf_err.append(result.err)

            for direction in self.directions:
                pairs = neighbour_pairs(self.ring_map, ring, direction, symmetric=True)
                result = correlate_pairs_torch(
                    table, pairs, edges_ccf,
                    device=self.device, device_table=self._device_table,
                )
                ccf[direction]["lag"] = result.lag
                ccf[direction]["g2"].append(result.g2)
                ccf[direction]["err"].append(result.err)

        # per-pixel total counts (for the saxs_2d payload) and a coarse
        # intensity-vs-time trace (for QA/beam-stability checks), both
        # computed directly from the event table.
        counts_2d = np.zeros(self.det_size[0] * self.det_size[1], dtype=np.float64)
        np.add.at(counts_2d, table.pixel, table.weights)

        n_time_bins = min(100, max(table.n_bins, 1))
        time_edges = np.linspace(0, table.n_bins, n_time_bins + 1)
        time_idx = np.clip(
            np.searchsorted(time_edges, table.times, side="right") - 1,
            0, n_time_bins - 1,
        )
        it_sum = np.zeros(n_time_bins, dtype=np.float64)
        np.add.at(it_sum, time_idx, table.weights)

        self._results = {
            "acf": {"lag": acf_lag, "g2": np.stack(acf_g2, axis=0), "err": np.stack(acf_err, axis=0)},
            "ccf": {
                d: {
                    "lag": ccf[d]["lag"],
                    "g2": np.stack(ccf[d]["g2"], axis=0),
                    "err": np.stack(ccf[d]["err"], axis=0),
                }
                for d in self.directions
            },
            "saxs_2d": counts_2d,
            "intensity_vs_time": np.vstack([np.arange(n_time_bins), it_sum]),
        }

    def get_normalized_payloads(self, qpm, skip_scattering=False, **kwargs):
        payloads = []
        table = self._table
        results = self._results
        if not skip_scattering:
            n_frames = max(table.n_bins, 1)
            saxs_2d = torch.as_tensor(results["saxs_2d"] / n_frames, dtype=torch.float32)
            output_scattering = {
                "saxs_2d": saxs_2d,
                "saxs_2d_par": saxs_2d.unsqueeze(0),
                "intensity_vs_time": torch.as_tensor(
                    results["intensity_vs_time"], dtype=torch.float32
                ),
                "mask_crop": self.mask_crop,
            }
            payloads.append(qpm.normalize_scattering(output_scattering))

        payload = {
            "event_run_info": {
                "n_rings": self.n_rings,
                "directions": list(self.directions),
                "per_decade": self.per_decade,
                "n_events": table.n_events,
                "bin_width_ns": table.bin_width,
            },
            "event_delay_list_acf": results["acf"]["lag"],
            "event_g2_acf": results["acf"]["g2"],
            "event_g2_acf_err": results["acf"]["err"],
        }
        # the CCF lag axis is identical across directions (same edges_ccf
        # used for all of them), so it's safe for either branch below to
        # set event_delay_list_ccf.
        if "horizontal" in self.directions:
            payload["event_delay_list_ccf"] = results["ccf"]["horizontal"]["lag"]
            payload["event_g2_ccf_h"] = results["ccf"]["horizontal"]["g2"]
            payload["event_g2_ccf_h_err"] = results["ccf"]["horizontal"]["err"]
        if "vertical" in self.directions:
            payload["event_delay_list_ccf"] = results["ccf"]["vertical"]["lag"]
            payload["event_g2_ccf_v"] = results["ccf"]["vertical"]["g2"]
            payload["event_g2_ccf_v_err"] = results["ccf"]["vertical"]["err"]
        payloads.append(payload)
        return payloads
```

- [ ] **Step 4: Export `EventCorrelator` from `correlator/__init__.py`**

Replace the full contents of `src/boost_corr/correlator/__init__.py`:

```python
"""Core correlator implementations."""

from .event import EventCorrelator
from .multitau import MultitauCorrelator
from .twotime import TwotimeCorrelator

__all__ = ["EventCorrelator", "MultitauCorrelator", "TwotimeCorrelator"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_event_correlator.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/boost_corr/correlator/event.py src/boost_corr/correlator/__init__.py tests/test_event_correlator.py
git commit -m "feat: add EventCorrelator"
```

---

## Task 5: Output schema — new `/xpcs/event/...` HDF5 keys

**Files:**
- Modify: `src/boost_corr/xpcs_aps_8idi/aps_8idi.py`
- Modify: `src/boost_corr/xpcs_aps_8idi/xpcs_result.py`
- Test: `tests/test_event_result_schema.py`

**Interfaces:**
- Consumes: the payload dict keys produced by Task 4's `EventCorrelator.get_normalized_payloads` (`event_run_info`, `event_delay_list_acf`, `event_g2_acf[_err]`, `event_delay_list_ccf`, `event_g2_ccf_h[_err]`, `event_g2_ccf_v[_err]`), plus the `event_config` kwarg this task adds to `XpcsResult.__init__`.
- Produces (used by Task 6): `XpcsResult(..., event_config=None)` accepted without error; `hdf_key` (from `aps_8idi.py`) resolves all `event_*` payload keys to `/xpcs/event/...` paths.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_event_result_schema.py -v`
Expected: FAIL — `XpcsResult.__init__() got an unexpected keyword argument 'event_config'`, and/or `KeyError` inside `put_results_in_hdf5` for the unknown `event_*` keys.

- [ ] **Step 3: Add the new keys to `aps_8idi.py`'s `key` dict**

In `src/boost_corr/xpcs_aps_8idi/aps_8idi.py`, add these entries to the `key` dict (anywhere inside the existing `{...}` literal, e.g. right after the `"twotime_config"` line):

```python
    "event_config": "/xpcs/event/config",
    "event_run_info": "/xpcs/event/run_info",
    "event_delay_list_acf": "/xpcs/event/delay_list_acf",
    "event_g2_acf": "/xpcs/event/normalized_g2_acf",
    "event_g2_acf_err": "/xpcs/event/normalized_g2_acf_err",
    "event_delay_list_ccf": "/xpcs/event/delay_list_ccf",
    "event_g2_ccf_h": "/xpcs/event/normalized_g2_ccf_h",
    "event_g2_ccf_h_err": "/xpcs/event/normalized_g2_ccf_h_err",
    "event_g2_ccf_v": "/xpcs/event/normalized_g2_ccf_v",
    "event_g2_ccf_v_err": "/xpcs/event/normalized_g2_ccf_v_err",
```

- [ ] **Step 4: Add `event_config` to `XpcsResult`**

In `src/boost_corr/xpcs_aps_8idi/xpcs_result.py`, change the `XpcsResult.__init__` signature from:

```python
    def __init__(
        self,
        raw_fname=None,
        qmap_fname=None,
        output_dir=None,
        meta_fname=None,
        overwrite=False,
        multitau_config=None,
        twotime_config=None,
        prefix=None,
        suffix=None,
    ) -> None:
```

to:

```python
    def __init__(
        self,
        raw_fname=None,
        qmap_fname=None,
        output_dir=None,
        meta_fname=None,
        overwrite=False,
        multitau_config=None,
        twotime_config=None,
        event_config=None,
        prefix=None,
        suffix=None,
    ) -> None:
```

and change:

```python
        self.analysis_config = {
            "rawdata_path": os.path.abspath(raw_fname),
            "multitau_config": multitau_config or {},
            "twotime_config": twotime_config or {},
        }
```

to:

```python
        self.analysis_config = {
            "rawdata_path": os.path.abspath(raw_fname),
            "multitau_config": multitau_config or {},
            "twotime_config": twotime_config or {},
            "event_config": event_config or {},
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_event_result_schema.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/boost_corr/xpcs_aps_8idi/aps_8idi.py src/boost_corr/xpcs_aps_8idi/xpcs_result.py tests/test_event_result_schema.py
git commit -m "feat: add /xpcs/event/... output schema"
```

---

## Task 6: CLI/dispatch wiring in `solve_correlation`

**Files:**
- Modify: `src/boost_corr/xpcs_aps_8idi/correlation/common.py`
- Test: `tests/test_event_dispatch.py`

**Interfaces:**
- Consumes: `EventCorrelator` (Task 4, imported into `common.py`), the new `event_config` kwarg (Task 5).
- Produces (used by Task 7): `solve_correlation(..., signal_spill="h", analysis_type="Multitau", ...)` resolves internally to `analysis_type="Event"` and runs `EventCorrelator`; `solve_correlation(..., signal_spill="h", analysis_type="Twotime", ...)` raises `exc.ConfigurationError` before any dataset is touched.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_event_dispatch.py -v`
Expected: FAIL — `signal_spill="h", analysis_type="Twotime"` currently runs (no `ConfigurationError`) rather than raising; `EventCorrelator` not referenced in `common.py`.

- [ ] **Step 3: Wire the dispatch in `common.py`**

Add this import alongside the existing correlator imports near the top of `src/boost_corr/xpcs_aps_8idi/correlation/common.py`:

```python
from ...correlator.event import EventCorrelator
```

Change:

```python
_CORRELATOR_CLASSES = {
    "Multitau": MultitauCorrelator,
    "Twotime": TwotimeCorrelator,
}

_VALID_TYPES = ("Multitau", "Twotime", "Both")
```

to:

```python
_CORRELATOR_CLASSES = {
    "Multitau": MultitauCorrelator,
    "Twotime": TwotimeCorrelator,
    "Event": EventCorrelator,
}

_VALID_TYPES = ("Multitau", "Twotime", "Both", "Event")
```

In `_build_run_args`, add an `"Event"` branch alongside the existing `if`/`elif` for `"Multitau"`/`"Twotime"` (after the `elif analysis_type == "Twotime":` block, before `return dict(...)`):

```python
    elif analysis_type == "Event":
        run_opts["normalize_opts"] = {"skip_scattering": skip_scattering}
        run_opts["debug_extra_keys"] = ("num_segments",)
```

At the very top of `solve_correlation`'s body (replacing the current first two lines):

```python
    if analysis_type not in _VALID_TYPES:
        raise ValueError(
            f"Unknown analysis_type {analysis_type!r}. Expected one of {_VALID_TYPES}"
        )
```

with:

```python
    use_event = signal_spill not in (None, "none")
    if use_event and analysis_type in ("Twotime", "Both"):
        raise exc.ConfigurationError(
            f"--signal-spill={signal_spill!r} requires the event correlator, "
            f"which does not support two-time analysis (--type={analysis_type!r})."
        )
    if use_event:
        analysis_type = "Event"

    if analysis_type not in _VALID_TYPES:
        raise ValueError(
            f"Unknown analysis_type {analysis_type!r}. Expected one of {_VALID_TYPES}"
        )
```

Further down, change:

```python
        if analysis_type == "Multitau":
            common_dset_kwargs["bin_time_s"] = bin_time_s
            common_dset_kwargs["run_config_path"] = run_config_path
```

to:

```python
        if analysis_type in ("Multitau", "Event"):
            common_dset_kwargs["bin_time_s"] = bin_time_s
            common_dset_kwargs["run_config_path"] = run_config_path
```

(`run_config_path` is a harmless no-op for Rigaku-family datasets — only `TimepixAnalysisDataset.__init__` reads it — but `bin_time_s` is required by both `RigakuDataset.get_events()`'s fallback and `TimepixAnalysisDataset`'s existing `apply_time_binning` call.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_event_dispatch.py -v`
Expected: PASS (4 tests: 3 in `TestEventDispatch` + 1 in `TestPrepareCorrelatorNeverReusesEventCorrelator`).

- [ ] **Step 5: Run the full existing test suite to check for regressions**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/ -v`
Expected: All tests pass except any pre-existing failures unrelated to this change (e.g. `test_boost_corr_multitau.py` references `boost_corr.help_functions`, which does not exist in the current codebase independent of this work — confirm any failures here are pre-existing by checking they also fail on `git stash`, not newly introduced).

- [ ] **Step 6: Commit**

```bash
git add src/boost_corr/xpcs_aps_8idi/correlation/common.py tests/test_event_dispatch.py
git commit -m "feat: route --signal-spill to EventCorrelator, reject with Twotime/Both"
```

---

## Task 7: End-to-end integration test

**Files:**
- Test: `tests/test_event_integration.py`

**Interfaces:**
- Consumes: the full stack from Tasks 1-6 (`solve_correlation`, `RigakuDataset.get_events`, `EventCorrelator`, output schema).
- Produces: nothing new — this is a regression-guarding integration test exercising the whole path with real (synthetic) files on disk, the same way a user's CLI invocation would.

Note on scope vs. the spec: the spec's testing section describes validating against
"a small slice of real run_246 `.bix` data." This task uses a synthetic `.bin`
file instead, generated in-test with the same `pack_rigaku()` helper as Task 2 —
bundling real run_246 data (or an absolute path into `/scratch/...`) into the
committed test suite would make it non-portable (unavailable in CI or on another
machine) and add a large binary to the git history. Numeric correctness against
the real algorithm is already covered by Task 1's brute-force parity test and
Task 4's reference-function comparison; this task's job is solely to catch
wiring regressions in the full `solve_correlation` -> `EventCorrelator` ->
`XpcsResult` path. Manually re-running `process_run246.py`-equivalent validation
against the actual run_246 `.bix` file remains a good idea before merging, but
outside this automated suite.

- [ ] **Step 1: Write the integration test**

```python
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
```

- [ ] **Step 2: Run the test**

Run: `cd /home/beams/MQICHU/Tools_cloud/xpcs_toolchains/boost_corr && python -m pytest tests/test_event_integration.py -v`
Expected: PASS. If it fails, check first whether the qmap fixture matches `XpcsQPartitionMap.load()`'s expectations exactly (`/qmap/dynamic_roi_map`, `/qmap/static_roi_map`, `/qmap/mask` are required; `/qmap/blemish` is optional and falls back to `mask`) before assuming an `EventCorrelator` bug.

- [ ] **Step 3: Commit**

```bash
git add tests/test_event_integration.py
git commit -m "test: add end-to-end integration test for the event correlator"
```

---

## Explicitly out of scope (do not attempt in this plan)

- `Rigaku3MDataset.get_events()` and multi-chip `TimepixAnalysisDataset.get_events()` — pixel-remapping geometry (`append_data()` / `raw_dataset.layout`) is not yet verified feasible, per the spec's Risks section. Calling `EventCorrelator.from_config` on these datasets correctly raises `exc.DatasetError` today (no `get_events()` defined), which is the safe behavior until a follow-up plan investigates and implements the remapping + cross-module clock-sync safeguard.
- Any new CLI flags (e.g. exposing `per_decade`/`lag_max`) — the spec fixed these at validated defaults (`per_decade=8`, `lag_max=n_bins//2`) to avoid scope creep.
