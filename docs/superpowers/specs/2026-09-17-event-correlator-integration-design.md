# Event correlator integration — design spec

Status: approved by user (2026-09-17), pending implementation plan.

## Motivation

boost_corr currently offers two correlation engines, selected via `--type`:
`Multitau` (default) and `Twotime` (opt-in, `Twotime`/`Both`). Both consume a
dataset through its dense, per-frame-batch API (`XpcsDataset.__getitem__`),
which requires binning photon events into fixed-width frames before any
correlation happens. For event-timing detectors (Timepix4/Tempus, and
Rigaku-family sparse formats where the "frame" field is configured as a very
fine time bin, e.g. 10 ns) this throws away timing information: the
event-driven correlator algorithm developed and validated separately (see the
sibling `xpcs_event_correlator.py` / `xpcs_event_correlator_torch.py` project)
computes g2 directly from per-photon arrival-time differences, giving an ACF
unaffected by dead-time artifacts down to native timing resolution, and a
CCF (nearest-neighbour-pixel correlation) that has no dead-time floor at all.

This spec adds that event correlator as a third engine, `EventCorrelator`,
selected automatically when `--signal-spill` is not `"none"`/`None`, with
two-time correlation never applicable to this path.

## Baseline (current working tree, uncommitted)

The boost_corr working tree (branch `mc_dev`) already contains, as
uncommitted local changes, all of the following — this spec treats it as the
starting baseline and does not modify it:

- `--signal-spill {none,v,h,vh}` CLI flag (`cli.py`), threaded through
  `solve_correlation(..., signal_spill=None, ...)` (`xpcs_aps_8idi/correlation/common.py`)
  into `common_dset_kwargs["signal_spill"]`, ending up as `XpcsDataset.signal_spill`.
- `XpcsDataset.update_signal_spill()`: charge-sharing correction via
  `torch.roll` on the densified frame, applied in `RigakuDataset.__getbatch__`,
  `Rigaku3MDataset.__getbatch__` (once, on the stitched canvas — sub-modules
  are forced to `signal_spill="none"` to avoid double-counting), and
  `TimepixAnalysisDataset.__getbatch__` for the multi-chip branch only (the
  `num_chips == 1` branch does not call it — see Risks).
- `RigakuExtendDataset` (`dataset/rigaku_extend_handler.py`, new file) and
  `convert_sparse_extend()` (`dataset/rigaku_utils.py`): a Rigaku-format
  variant with 20-bit pixel / 32-bit frame fields, wired into
  `create_dataset()` for the `.bix` extension. This is the exact format used
  to validate `xpcs_event_correlator.py` against real data (run_246) earlier
  in this project.

None of the above needs to change. This spec is purely additive.

## Non-goals

- No change to `Multitau`/`Twotime` behavior when `--signal-spill` is `none`/`None`.
- No new dense-frame code path for the event correlator — it never calls
  `__getitem__`/`__getbatch__`.
- IMM/HDF datasets are out of scope (no sparse/event structure to draw from).

## Architecture overview

```
cli.py --signal-spill h  ──▶  solve_correlation(signal_spill="h", analysis_type=...)
                                      │
                                      ├─ signal_spill not in (None,"none")
                                      │     and analysis_type in ("Twotime","Both")
                                      │     ──▶ raise exc.ConfigurationError
                                      │
                                      └─ else: analysis_type forced to "Event"
                                            │
                                            ▼
                                   EventCorrelator.from_config(qpm, dset, device, ...)
                                            │
                              dset.get_events() ──▶ (pixel, time_bin, count, bin_width_ns)
                                            │
                                   EventTable  ──▶  upload_table (GPU, once)
                                            │
                        per dq ring: ACF (auto_pairs) + CCF (neighbour_pairs, h/v/both)
                                            │
                                   get_normalized_payloads(qpm)
                                            │
                                   XpcsResult.append({...})  ──▶  /xpcs/event/...
```

## 1. CLI & dispatch

No new CLI flags are required for the core routing behavior.

In `solve_correlation()` (`xpcs_aps_8idi/correlation/common.py`):

```python
use_event = signal_spill not in (None, "none")
if use_event and analysis_type in ("Twotime", "Both"):
    raise exc.ConfigurationError(
        f"--signal-spill={signal_spill!r} requires the event correlator, "
        f"which does not support two-time analysis (--type={analysis_type!r})."
    )
if use_event:
    analysis_type = "Event"
```

This check runs regardless of whether `--type` was left at its default
(`"Multitau"`) or passed explicitly — only an explicit `Twotime`/`Both`
conflicts. `_VALID_TYPES` gains `"Event"`; `_CORRELATOR_CLASSES["Event"] =
EventCorrelator`; `_build_run_args` gains an `"Event"` branch (no
`smooth`/`dq_selection` `flag_sort` twotime-only options apply).
`need_twotime` stays `False` for this path — the twotime pipeline is never
reached.

`prepare_correlator`'s reuse-across-jobs cache (keyed by `frame_num`/`det_size`)
is bypassed for `EventCorrelator` — a fresh instance is built per job, since
each job's event table differs regardless of shared `det_size`.

## 2. Dataset event-access layer

New informal-protocol method, implemented per dataset class:

```python
def get_events(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (pixel, time_bin, count, bin_width_ns), pixel/time_bin/count as
    int64 arrays over the dataset's full (uncropped) pixel space."""
```

- `RigakuDataset`, `RigakuExtendDataset`: return `self.ifc` (already sorted
  by frame/time-bin) converted to numpy. `bin_width_ns` from the bin-width
  helper below.
- `TimepixAnalysisDataset`, `num_chips == 1`: return `self.ifc_list[0]`
  (already binned via the existing `apply_time_binning(bin_time_s)` /
  `--bin-time-s` mechanism — no reach into the external `timepix_dataset`
  package's raw ToA is needed). `bin_width_ns = bin_time_s * 1e9`.
- `Rigaku3MDataset`, `TimepixAnalysisDataset` with `num_chips > 1`: each
  module's pixel index must be remapped into the stitched-canvas flat index
  space, mirroring `Rigaku3MDataset.append_data()` / `layout[module_index]`'s
  geometry respectively. **Not yet verified feasible at zero extra
  complexity** — needs a focused read of that geometry before implementation
  (see Risks). May ship as a fast-follow rather than in the initial PR.
- Base `XpcsDataset` (and therefore `HdfDataset`/`ImmDataset`): no
  `get_events()`. `EventCorrelator.from_config` checks
  `hasattr(dset, "get_events")` and raises `exc.DatasetError` otherwise.

**Bin-width resolution** (Rigaku-family only; Timepix4 keeps its existing
mechanism): new helper, e.g. `dataset/utils.py::resolve_bin_width_ns(raw_fname,
bin_time_s)`. Looks for a sibling `<stem>_metadata.hdf` (same stem-stripping
convention `xpcs_result.py::create_unique_file` already uses for
`.bin`/`.bix`/`_metadata`), reads
`entry/instrument/detector_1/frame_time` (seconds) and converts to ns. Falls
back to `--bin-time-s` (already defined, default `1e-6` s) if no metadata
file or field is found. If both are present and disagree, log a warning and
use the metadata value.

## 3. `EventCorrelator` (`boost_corr/correlator/event.py`)

Vendors the core algorithm from the sibling `xpcs_event_correlator.py` /
`xpcs_event_correlator_torch.py` project (the numpy version kept only for
internal test parity-checking, not part of the production path — GPU-first
like the rest of boost_corr, with CPU as a working-but-slow fallback via
`device="cpu"`).

Implements the same informal protocol as `MultitauCorrelator`/`TwotimeCorrelator`:

- `from_config(cls, qpm, dset, device, **kwargs)`: requires
  `hasattr(dset, "get_events")` (else `exc.DatasetError`). Builds
  `ring_map = qpm.dqmap.astype(int) - 1` (after `qpm.update_rotation`, so
  orientation matches `dset`). Derives CCF direction(s) from
  `dset.signal_spill`: `"h"` → `["horizontal"]`, `"v"` → `["vertical"]`,
  `"vh"` → `["horizontal", "vertical"]` (reported as two separate curves, not
  pooled — they're physically distinct pair populations). Builds one shared
  log-spaced lag-edge array per ring set (ACF: `lag_min=1`; CCF: `lag_min=0`;
  `lag_max = table.n_bins // 2`, `per_decade=8` — the defaults already
  validated on run_246 in the sibling project; not exposed as new CLI flags
  in v1, to avoid scope creep) so every dq ring shares one `delay_list`,
  matching the existing multitau convention of a single shared tau axis per
  output.
- `reset()`: no persistent state to clear (each job builds fresh).
- `process_dataset(dset, verbose=False, use_loader=False, num_workers=0)`:
  calls `dset.get_events()` once, builds an `EventTable`, uploads it to
  `device` once via `upload_table()`. Loader/num_workers args accepted for
  signature compatibility with `common.py::run_correlation`, unused.
- `post_process(**kwargs)`: accepts and ignores arbitrary kwargs for
  signature compatibility with `common.py::run_correlation`'s
  `correlation_post_kwargs` (multitau/twotime use this for options like
  `normalize_frame`; the event correlator has no equivalent knobs in v1).
  For each dq ring, runs ACF
  (`correlate_pairs_torch` + `auto_pairs`) and one CCF per selected direction
  (`neighbour_pairs`), reusing the uploaded table (`device_table=`) so the
  event data is transferred to GPU exactly once regardless of ring count.
- `get_normalized_payloads(qpm, skip_scattering=False, **kwargs)`: builds
  `saxs_2d` from per-pixel total counts (`bincount` over the event table) for
  parity with multitau's scattering output, routed through
  `qpm.normalize_scattering` unless `skip_scattering`. Packages g2 results
  (see Output schema).
- `describe()`: logs event count, ring count, and selected CCF direction(s)
  for the `--verbose` path (multitau has this; twotime currently does not —
  `EventCorrelator` should have it to avoid the same parity gap).

## 4. Output schema

New keys added to the `key` dict in `xpcs_aps_8idi/aps_8idi.py`, all under
`/xpcs/event/...`:

| dict key | HDF5 path |
|---|---|
| `event_config` | `/xpcs/event/config` |
| `event_delay_list_acf` | `/xpcs/event/delay_list_acf` |
| `event_g2_acf` | `/xpcs/event/normalized_g2_acf` |
| `event_g2_acf_err` | `/xpcs/event/normalized_g2_acf_err` |
| `event_delay_list_ccf` | `/xpcs/event/delay_list_ccf` |
| `event_g2_ccf_h` / `_err` | `/xpcs/event/normalized_g2_ccf_h[_err]` |
| `event_g2_ccf_v` / `_err` | `/xpcs/event/normalized_g2_ccf_v[_err]` |

Only the CCF direction(s) selected by `signal_spill` are written.
`saxs_2d`/`saxs_1d` reuse the existing `/xpcs/temporal_mean/...` keys
unchanged — same physical meaning as multitau's. No changes to `XpcsResult`
itself: `append()` is already payload-shape-agnostic; only the key-path
lookup table gains entries.

New keys are used instead of reusing `/xpcs/multitau/...` to avoid
mislabeling event-correlator output with multitau provenance/config.

## 5. Known risks / open items

- **Backward-compatibility change**: any existing workflow combining
  `--signal-spill {v,h,vh}` with `--type Multitau` (default or explicit) on
  Rigaku/RigakuExtend data will change behavior once this ships — from
  "multitau on spill-corrected frames" to "event correlator only, multitau
  not run." Confirmed accepted by the user (2026-09-17) as the intended
  effect of this feature, not a regression to guard against.
- **Rigaku3M / multi-chip Timepix4 `get_events()`**: pixel-remapping through
  each module's placement in the stitched canvas is not yet verified at zero
  extra complexity. Needs a dedicated read of `Rigaku3MDataset.append_data()`
  and `TimepixRawDataset.layout` (external `timepix_dataset` package) before
  implementation. May ship as a fast-follow.
- **Cross-module clock synchronization** (Rigaku3M / multi-chip Timepix4
  only): each module's/chip's clock is independently zeroed with no verified
  cross-synchronization. Default: restrict ACF/CCF pixel pairs to within a
  single module/chip; skip (with a warning) any dq ring whose pixels span a
  module boundary, until synchronization is validated. Single-module data
  (plain Rigaku, RigakuExtend, single-chip Timepix4 — the run_246/Tempus case
  used to validate the underlying algorithm) is unaffected.
- **`XpcsDataset.signal_spill` default is `"h"`**, while the CLI default is
  `None` (i.e. `"none"` behavior) — pre-existing inconsistency in the current
  baseline (direct Python API construction without a CLI defaults to
  spill-correction *and* the event correlator; via the CLI, the previous
  default of no correction/multitau is preserved). Not introduced by this
  spec; noted for awareness, not proposed to change here.

## 6. Testing plan

- Unit tests for `EventCorrelator`, mirroring `test_boost_corr_multitau.py`'s
  synthetic-data pattern (small `det_size`, few frames/events), checked
  against a brute-force reference (same discipline as the sibling project's
  `test_xpcs_event_correlator.py`).
- CLI-level test: `--signal-spill h --type Twotime` raises
  `ConfigurationError`; `--signal-spill h` alone (with `--type` left default
  or explicit `Multitau`) resolves to `analysis_type == "Event"`.
- End-to-end numeric-parity test against the sibling project's
  `xpcs_event_correlator.py` (numpy reference) on a small slice of real
  run_246 `.bix` data — the same torch-vs-numpy exact-match discipline used
  throughout this project's earlier validation.
- Targeted tests for `Rigaku3MDataset`/multi-chip `TimepixAnalysisDataset`
  `get_events()` pixel-remapping specifically, once that geometry is
  confirmed (see Risks) — highest-risk pieces get the most direct coverage.
