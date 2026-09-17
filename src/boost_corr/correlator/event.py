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
            if not (self.ring_map == ring).any():
                # empty dq bin (a gapped/non-contiguous qmap is a supported
                # configuration -- xpcs_qpartitionmap.py's own compute_g2()
                # already skips empty roi_dq bins). auto_pairs/neighbour_pairs
                # would return zero pairs here and correlate_pairs_torch
                # would raise on an empty pair list, so fill this row with
                # NaN instead of crashing the whole job. We must not skip the
                # row entirely -- that would shift every later ring's array
                # index out of alignment with its actual dq label.
                acf_g2.append(np.full(len(edges_acf) - 1, np.nan))
                acf_err.append(np.full(len(edges_acf) - 1, np.nan))
                for direction in self.directions:
                    ccf[direction]["g2"].append(np.full(len(edges_ccf) - 1, np.nan))
                    ccf[direction]["err"].append(np.full(len(edges_ccf) - 1, np.nan))
                continue

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
