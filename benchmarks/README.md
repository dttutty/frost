# Benchmark architecture

Benchmark integration is intentionally separate from the FROST runtime. Code
under `src/frost/` must not import `third_party/`; external systems are invoked
through the adapters in `src/frost_bench/adapters/`.

## Layout

- `specs/presets.yaml` owns launch presets and dataset batch-size defaults.
- Memory-model defaults are global batches: WIKIPEDIA 600, MOOC 1200,
  REDDIT 2000, and LASTFM 4000; JODIE uses one fifth of each value. Adapters
  convert the global value to per-GPU sizes where a baseline requires it.
- `specs/dyglib_parity.yaml` is the reviewed snapshot of DyGLib-aligned model
  settings used by FROST.
- `specs/sweeps.yaml` declares model/dataset/seed grids without duplicating
  launch defaults in shell scripts.
- `src/frost_bench/spec.py` defines explicit global/per-GPU batch semantics.
- `specs/throughput_suites.yaml` declares the maintained throughput matrices.
- `src/frost_bench/throughput.py` expands those matrices, runs jobs serially,
  and writes one normalized result schema for FROST, TGL, and DyGLib.
- `src/frost_bench/adapters/` translates an `ExperimentSpec` into a backend
  command without importing baseline implementation modules.
- `src/frost_bench/io.py` owns subprocess logging and tabular output.
- `src/frost_bench/metrics.py` normalizes backend logs into common throughput
  records and owns rank/curve aggregation.
- `src/frost_bench/telemetry.py` owns `nvidia-smi` sampling, parsing, and common
  utilization summaries.
- `src/frost_bench/workload.py` owns epoch and measurement-window arithmetic.

`frost-train` is the canonical single-run preset entry point. `frost-suite`
runs the maintained throughput matrices, and `frost-sweep` is the canonical
seed/grid sweep entry point. Sweeps
run one distributed job at a time; use an external cluster scheduler when jobs
must run concurrently.

```bash
uv run frost-sweep --list
uv run frost-suite --list
uv run frost-suite tgn_memdim_2gpu --dry-run
uv run frost-suite tgn_memdim_2gpu --set mem_dims='[128,256]'
uv run frost-suite memory_models_2gpu --set batch_sizes='[600,1200]'
uv run frost-suite large_tgn_batch_sweep
uv run frost-suite large_dygformer_epoch --set datasets='[MAG]' --set batch_sizes='[4000]'
uv run frost-sweep memory_seeds --dry-run
uv run frost-sweep dyglib_grid --gpus 0,1
uv run frost-compare-tgl-mrr --dataset WIKIPEDIA --epochs 50 --eval-neg-samples 20
```

`frost-compare-tgl-mrr` runs TGN in FROST and distributed TGL with the same
destination-partition candidate pool, seed, global batch size, and per-split
negative-sampler reset. It uses tie-aware MRR and writes both validation and
test curves for every epoch.

## Artifacts

New benchmark runs write beneath `artifacts/` by default. Override the root
without editing a script:

```bash
export FROST_ARTIFACT_ROOT=/path/to/benchmark-artifacts
```

Each benchmark should create a suite/run directory below that root and keep
generated configs, raw logs, normalized metrics, plots, and reports together.
`artifacts/` is ignored by Git.

The `large_*_batch_sweep` suites use bounded train-only windows. The
`large_*_epoch` suites run one complete train split with evaluation disabled;
their result rows include measured throughput, estimated train time, and total
process wall time (including data loading).

The node-feature-dimension profile temporarily replaces the external dataset's
`node_features.npy` and restores the original file or symlink in a `finally`
block after each backend run.

## Adding a backend

1. Add an adapter returning `PreparedRun(command, cwd, env)`.
2. State the backend's `BatchSizeScope` and reject incompatible specs.
3. Add its log parser to `metrics.py` with fixture-based tests.
4. Keep unavoidable baseline changes in the baseline fork; do not import its
   source into FROST or duplicate integration logic across experiment scripts.
