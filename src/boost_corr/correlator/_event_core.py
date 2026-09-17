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
