# Historical experiments

This directory contains research investigations and their reproducibility
inputs. Maintained benchmark protocols live in `benchmarks/specs/`; do not copy
settings from these historical runs into current defaults.

## Local log folders

| Folder | Experiments |
| --- | --- |
| `throughput/` | FROST throughput comparisons with TGL and DyGLib |
| `scaling/` | GPU-count scaling and queue diagnostics |
| `memory/` | Memory footprint measurements |
| `utilization/` | CPU and GPU utilization samples |
| `ablation/` | Stale-state handling and memory/mail ablations |
| `parameter_sweeps/` | Memory width, feature width, and worker queue sweeps |
| `profiling/` | Detailed runtime profiles |

Each folder retains the original experiment and run directory names. Research
scripts, AP investigations, microbenchmarks, and the `lei/` plot bundle remain
at their existing paths. Seed sweeps are grouped by model under
`../seed_sweeps/<model>/`; Hydra logs retain their date/time layout under
`../outputs/`.

Identical nonempty log, result, and plot files share one copy through relative
symbolic links. Local move mappings, duplicate targets, and original SHA-256
checksums are recorded in `log_organization/manifest.json`.

## AP parity investigation, 2026-07-28

`ap_debug_frost_vs_tgl_20260728/` records the FROST/TGL AP investigation across
Wiki, MOOC, Reddit, and LastFM. The main conclusions were:

- the small default-runtime Wiki difference did not establish a FROST accuracy
  bug and was sensitive to TGL dynamic-state ordering;
- multi-dataset screening found no stable dataset-independent advantage;
- an attention projection-order mismatch was corrected;
- historical DyRep results are superseded because DyRep support was removed.

Accepted per-run values remain in `results.csv`, `multidataset_results.csv`, and
`model_results.csv`. Raw logs are archived under
`artifacts/worktree_archive_20260728/`. Relevant correction commits are
`36d9f76` and `e30e9a7`.

## TGL one-GPU versus FROST, 2026-07-29

`tgl_1gpu_vs_frost_2gpu_ap_20260729/` compares matched LastFM/TGN runs. The
archived three-seed result used TGL's former shared-random-negative protocol and
showed no evidence that two FROST workers reduced AP relative to serial one-GPU
TGL. The rerun script now uses the repository-wide fixed-negative Frost
protocol, so new results are not expected to reproduce the archived values
exactly. Both are ten-epoch screens, not equivalence claims.

Reproduce it with:

```bash
uv sync --locked --project third_party/tgl
uv run python experiments/tgl_1gpu_vs_frost_2gpu_ap_20260729/run_comparison.py
```

Outputs go to `artifacts/comparisons/tgl_1gpu_vs_frost_2gpu_ap_20260729/`.

## Shared plot bundle

`lei/` is the self-contained source bundle for the
`gpu_util_lastfm_tgn_dygformer_bs4000` figure. It contains the plotting script,
source logs, utilization samples, curve data, and rendered PDF. Regenerate with:

```bash
uv run python experiments/lei/gpu_util_lastfm_tgn_dygformer_bs4000_plot.py
```
