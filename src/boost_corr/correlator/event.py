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
        # restrict every dq ring to masked-in pixels, matching what the dense
        # multitau path does (xpcs_qpartitionmap.compute_g2 correlates only
        # dqmap.reshape(-1)[mask_idx_1d]). Nothing zeroes qpm.dqmap outside
        # qpm.mask, so without this a blemished/dead pixel that still carries a
        # nonzero dq label would silently enter the ACF/CCF pixel-pair lists.
        ring_map = np.where(qpm.mask, qpm.dqmap, 0).astype(np.int64) - 1
        return cls(ring_map, directions, device=device,
                   per_decade=kwargs.get("per_decade", 8))

    def reset(self):
        self._table = None
        self._device_table = None
        self._results = None

    def describe(self):
        # no event count here: common.py::run_jobs calls describe() *before*
        # process_dataset() builds the table, so it would always log 0.
        # process_dataset's own verbose branch logs the real count instead.
        logger.info(
            f"event correlator: {self.n_rings} rings, directions={self.directions}"
        )

    def process_dataset(self, dset, verbose=False, use_loader=False, num_workers=0):
        pixel, time_bin, count, bin_width_ns = dset.get_events()
        npix = self.det_size[0] * self.det_size[1]
        # build from the integer time_bin directly (bin_width=1.0) rather than
        # multiplying into ns and re-quantizing by the same float, which is not
        # guaranteed to round-trip exactly for non-dyadic bin_width_ns and can
        # silently misbin events by one time bin. bin_width is attached
        # afterwards: _event_core only consumes it as a multiplicative scale
        # factor on `lag`, never to re-derive bin indices.
        self._table = build_event_table(
            time_bin.astype(np.float64), pixel, 1.0, npix,
            weights=count.astype(np.float64),
        )
        self._table.bin_width = float(bin_width_ns)
        self._device_table = upload_table(self._table, self.device)
        if verbose:
            logger.info(
                f"event table built: {self._table.n_events} events, "
                f"{self._table.n_bins} bins, bin_width={bin_width_ns} ns"
            )

    def post_process(self, **kwargs):
        if self.n_rings == 0:
            # np.stack([]) below would raise a bare ValueError; fail with a
            # message that says what is actually wrong with the qmap.
            raise exc.ConfigurationError(
                "qmap has no valid dq rings (dynamic_roi_map is entirely masked "
                "out or zero); the event correlator needs at least one ring."
            )
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
            # device= is mandatory: qpm.normalize_scattering ->
            # average_with_index_map runs torch.bincount against qmap tensors
            # living on self.device, and mixing a CPU tensor in raises
            # RuntimeError (surfaced as PostProcessingError) on every GPU run.
            saxs_2d = torch.as_tensor(
                results["saxs_2d"] / n_frames, dtype=torch.float32,
                device=self.device,
            )
            output_scattering = {
                "saxs_2d": saxs_2d,
                "saxs_2d_par": saxs_2d.unsqueeze(0),
                "intensity_vs_time": torch.as_tensor(
                    results["intensity_vs_time"], dtype=torch.float32,
                    device=self.device,
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
            # pyXpcsViewer's load_data() also requires
            # /xpcs/multitau/config/{stride_frame,avg_frame} to exist for any
            # file whose analysis type includes "Multitau". Always 1 here by
            # construction: solve_correlation rejects any --stride-frame/
            # --avg-frame != 1 on the event path (get_events() always reads
            # the whole file). The corresponding frame_time (t0) override
            # that makes downstream tau*t0 scaling correct is in
            # common.py's Event branch of _build_run_args.
            "multitau_config": {"stride_frame": 1, "avg_frame": 1},
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

        # Alias the CCF result -- not ACF -- into the plain multitau schema
        # (/xpcs/multitau/delay_list, normalized_g2[_err]) so downstream
        # tools that only recognize Multitau/Twotime output (e.g.
        # pyXpcsViewer's get_analysis_type(), which just checks whether
        # these paths exist) see the correlation --signal-spill actually
        # selected, which is the CCF, not the direction-independent ACF.
        # Multitau's g2/g2_err convention is (n_tau, n_dq); ours is
        # (n_dq, n_tau) -- transpose to match. When both directions are
        # requested (vh), vertical wins the alias, matching the tie-break
        # event_delay_list_ccf already uses above.
        primary_direction = "vertical" if "vertical" in self.directions else "horizontal"
        payload["delay_list"] = results["ccf"][primary_direction]["lag"]
        payload["g2"] = results["ccf"][primary_direction]["g2"].T
        payload["g2_err"] = results["ccf"][primary_direction]["err"].T

        payloads.append(payload)
        return payloads
