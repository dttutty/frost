# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Build & Run

```bash
# Sync the uv-managed project environment (builds the C++ seq_sampler extension)
uv sync --locked

# Rebuild the editable package after changing C++ seq_sampler sources
uv pip install -e .

# Run training — nproc_per_node = n_gpu_workers + 1 (host process)
uv run torchrun --nproc_per_node=3 run.py \
  model=tgn dataset=LASTFM

# Run tests
uv run pytest tests/ -v
```

## Model Commands

| Model | Command |
|-------|---------|
| TGN | `uv run torchrun --nproc_per_node=3 run.py model=tgn dataset=LASTFM` |
| JODIE | `uv run torchrun --nproc_per_node=3 run.py model=jodie dataset=LASTFM` |
| APAN | `uv run torchrun --nproc_per_node=3 run.py model=apan dataset=LASTFM` |
| TGAT | `uv run torchrun --nproc_per_node=3 run.py model=tgat dataset=LASTFM` |
| DyGFormer | `uv run torchrun --nproc_per_node=3 run.py model=dygformer dataset=LASTFM` |
| GraphMixer | `uv run torchrun --nproc_per_node=3 run.py model=graph_mixer dataset=LASTFM` |
| TCL | `uv run torchrun --nproc_per_node=3 run.py model=tcl dataset=LASTFM` |
| CAWN | `uv run torchrun --nproc_per_node=3 run.py model=cawn dataset=LASTFM` |
| EdgeBank | `uv run python eval_edgebank.py --dataset LASTFM` |

**Config defaults** (`configs/config.yaml`): `model=tgn`, `n_neg=1`, `ring_buf_size=64`, `macro_batch_size=4000`, `d_time=128`, `no_memory_queue_size=4`.

Model recipes own their encoder, optional state updater, semantic contract, and compatible link head. Runtime code must consume the compiled `ExecutionPlan` rather than branch on model target strings.

## Architecture Overview

FROST is a distributed Temporal GNN training framework. Rank assignment: workers = ranks `0..n_gpus-1`, host/master = rank `world_size-1`.

**Entry point:** `run.py` → `frost.launcher.launch(cfg)`. The thin launcher coordinates `frost.launch.config`, `factory`, `distributed`, and `dispatch`, then selects Host vs Worker by rank.

Dataset attachment, per-GPU batch layout, ring-buffer sizing, and dynamic-state shared-memory allocation live in `frost.launch.storage`.

### Host process (`src/frost/runtime/host/`)
- `_engine.py` — `Host` class: drives sampling, batching, and scatter to workers
- `analyzer.py` — `AccessPlanAnalyzer`: resolves contract-declared `NodeSet` scopes into mem/box/nfeat planner inputs
- `_meta_batcher.py` — `MetaBatcher` thread
- `_planner.py` — `Planner` thread (nfeat/mem/box planning)
- `_scatter.py` — `Scatter` thread (dist to workers via `scatter_object_list`)

### Worker process (`src/frost/runtime/worker/`)
- `_engine.py` — `Worker` orchestration and main event loop
- `_pipeline.py` — queue/thread construction and shutdown
- `_trainer.py` — model/optimizer setup and train/eval steps
- `_telemetry.py` — benchmark environment and CPU-process sampling
- `_checkpoint.py` — DDP-safe model save/load
- `_receiver.py` — `Receiver` thread (receives from host)
- `_host_to_device.py` — `H2D` thread (loads features to GPU)
- `_dynamic_assembler.py` — `DynamicAssembler` control loop
- `_dynamic_state.py` — shared-memory and P2P memory/mailbox tensor assembly
- `_static_assembler.py` — `StaticAssembler`
- `p2p_syncer/` — `MemHandler`, `BoxHandler`, `P2PQHandler`, writers

### Models (`src/frost/models/`)
- `temporal_link_model.py` — `TemporalLinkModel`: pure computation (mem_updater → unflatten → encoder → representation adapter → head)
- Runtime state publication is owned by `runtime/worker/_state_coordinator.py`, not by model modules.
- `mem_updater.py` — `MemoryUpdater` (GRU), `TransformerMemoryUpdater`
- `header.py` — `AdditiveLinkHead`, `ConcatLinkHead`
- `encoder/` — `tgn.py`, `tgat.py`, `apan.py`, `jodie.py`, `dygformer.py`, `graph_mixer.py`, `tcl.py`, `cawn.py`

### Shared memory / sampling (`src/frost/shm/`, `src/frost/sampling/`)
- `ring_buf.py` — `RingBuf`: circular shared-memory sampling buffer
- `mem_mgr.py` — `MemBoxStoreManager`: node memory + mailbox
- `dataset_n_feat.py` — `DatasetManager`: loads nfeat/efeat into `/dev/shm`
- `ngh_sampler.py` — `NoSampler` (0-hop), `NghSampler` (1-hop), `MultiHopNghSampler` (multi-hop; `uniform`, `recent`, `time_interval_aware`)

### Data types (`src/frost/datatypes/`)
- `meta.py` — `Meta`: batch metadata carrier
- `sampled_layout.py` — `SampledLayoutView`: semantic `NodeSet` selection over flat sampled layouts
- `graph.py` — `FlatGraph`, `HierGraph`, `NeighborOnHop`
- `plan.py` — runtime memory/mailbox/static transfer plans
- `env.py` — `Env`: distributed rank/device info

## Data Flow

```
Host process
 ├─ ngh_sampler.sample(roots, ts) → RingBuf slot
 ├─ AccessPlanAnalyzer resolves ExecutionPlan node scopes
 └─ scatter_object_list → Workers
                           ├─ Receiver (scatter_object_list)
                           ├─ H2D (load nfeat/efeat/mem/box → GPU)
                           ├─ [DynamicAssembler if mem_updater]
                           └─ DynamicStateCoordinator.batch(meta)
                              └─ TemporalLinkModel.forward(meta)
                                 ├─ mem_updater(flat_graph)
                                 ├─ coordinator hook publishes state tasks
                                 ├─ flat_graph.unflatten(...)
                                 ├─ encoder(graph=meta.graph)
                                 ├─ RepresentationAdapter(...)
                                 └─ head(edge_representations) → logits
```

**FlatGraph layout in RingBuf slot:**
- `[0 : n_src*(2+n_neg)]` → root nodes: `[src | pos_dst | neg_dst×n_neg]`
- `[n_src*(2+n_neg) : ...]` → neighbor tiers per fanout hop
  - `RootExpansion.ALL_ROOTS`: tier_k size = `prev_tier * fanout[k]`
  - `RootExpansion.POSITIVE_ROOTS`: tier_0 size = `n_src * 2 * fanout[0]`

## Key Invariants

### Dimension alignment
- `model.state.updater.d_mem` **must equal** `model.embedding_dim` for JODIE and APAN.
- Sampling fanout is owned by `model.neighborhood.fanout`; encoders receive it only when their computation needs the value.
- APAN uses positive-root one-hop sampling declared by its model contract.
- `d_mail = 2*d_mem + edim` — computed in `launcher.py`; mailbox slot size and `TransformerMemoryUpdater` key/value dim must agree.
- `GraphMixer`: `fanout[0]` is the total neighbors sampled (node encoder uses all, link encoder uses `num_tokens`). `num_tokens` controls the link encoder neighbor count (default 20); its recipe declares concat scoring.
- `TCL`: `num_tokens` should equal `fanout[0]`; `num_depths = fanout[0] + 1`.
- `CAWN`: DyGLib parity selects `ngh_sampler.strategy=time_interval_aware` with `time_scaling_factor=1e-6`; the weight belongs to sampler config, not the encoder.

### Representation and scoring compatibility
- Encoders return `EncoderResult`; `RepresentationAdapter` converts either node-aligned or pair-aligned layouts into `EdgeRepresentations`.
- `DyGFormer`, `CAWN`, and `TCL` declare pair-aligned representations; the adapter preserves their negative-pair source embeddings.
- Heads consume structured edge representations and never infer `n_neg` or layout from tensor length.

### Analyzer ↔ Runtime consistency (APAN)
The `NeighborhoodMailbox` contract must exactly match what `NeighborhoodMailboxRouter.deliver()` writes at runtime:
1. `ngh_start = n_src * (2 + n_neg)` — not `n_src * (2*n_neg + 2)`
2. Filter padding: `pos_ngh = pos_ngh_raw[pos_ngh_raw != -1]`
3. `box_update = TensorSet(pos_ngh)` only — roots are not written by the router

### Sampler invariants
- `NoSampler.fanout` is always `None`; `TemporalLinkModel` treats it as `[]`.
- The C++ `SeqSampler` supports `uniform`, `recent`, and `time_interval_aware`; `time_interval_aware` consumes `time_scaling_factor` from config.
- `time_scaling_factor` is sampler-only config under `ngh_sampler`.
- `MultiHopNghSampler.sample()` receives `ts` of shape `[n_src]`; must expand to `ts.repeat(2 + n_neg)` before calling C++ `get_historical_neighbors()` (positional-only args).

### Memory model detached states
`TransformerMemoryUpdater.forward()` must write cached states to `flat_graph.detached_*`, not `self.detached_*`. `DynamicStateCoordinator` reads `flat_graph.detached_nid/ts/mem` in the updater hook.

## Queue Sizes

| Queue | No memory (TGAT) | With memory (TGN/JODIE/APAN) |
|-------|-----------------|------------------------------|
| `Worker.full_q` | 4 | 1 |
| `Worker.received_q` | 4 | 1 |
| `Host.meta_in_q` | 4 | 2 |
| `Host.scatter_q` | 4 | 64 |
| `Host.mem_plan_out_q` | n/a | 32 |
| `Host.mail_plan_out_q` | n/a | 32 |
| `Host.nfeat_plan_out_q` | 4 | 32 |

For stateless/no-memory models, `no_memory_queue_size` defaults to `4` and currently fans out to `Worker.full_q`, `Worker.received_q`, `Host.scatter_q`, and `Host.nfeat_plan_out_q` when feature planning is enabled.

For memory models, `full_q=1` is intentional: `DynamicAssembler` must wait for P2P completion (`delta_q`) before putting assembled batches in `full_q`. Pipeline depth controlled by `h2d_sem` (released 4× at startup).

For memory models, the host-side queue defaults are fixed in code to `scatter=64` and `mem/mail/nfeat=32`. These values are intentionally not overridable via environment variables.

## Sweeps

## Adding a New Model

1. Create an encoder in `src/frost/models/encoder/my_model.py` that returns a structured result:
   ```python
   class MyEncoder(nn.Module):
       def forward(self, graph: HierGraph) -> EncoderResult:
           # graph.root_mem, root_mem_ts, root_ts, root_nfeat, root_nid
           # graph.neighbors[0].nid/ts/mem/efeat  (if 1-hop)
           return EncoderResult(h)
   ```
2. Add a `ModelRecipe` in `src/frost/models/recipes.py` and declare its complete `ModelContract`: neighborhood hop bounds/root expansion, feature requirements and scope, dynamic-state read/write/mailbox semantics, representation kind, and scoring kind.
3. Create `configs/model/my_model.yaml` as the single composition root:
   ```yaml
   _target_: frost.models.recipes.MyModelRecipe
   embedding_dim: 128
   neighborhood:
     fanout: [10]
   encoder:
     _target_: frost.models.encoder.my_model.MyEncoder
     d_out: ${model.embedding_dim}
   state: null
   head:
     _target_: frost.models.header.AdditiveLinkHead
     d_in: ${model.embedding_dim}
   ```
4. Register the recipe in `RECIPE_TYPES`, add contract/config tests, and run with `run.py model=my_model`.

Do not add a model-specific analyzer or predictor-layout branch. Extend semantic contract/plan types only when the new model genuinely requires a new runtime capability.

## Datasets

Stored in `DATA/<DATASET>/`. Loaded directly by `DatasetManager`. Expected files:
`edges.csv` (columns: `eid`, `src`, `dst`, `time`, `default_split`), exactly one
temporal CSR (`full_graph_no_reverse_edges.npz`,
`full_graph_with_reverse_edges.npz`, or legacy TGL `ext_full.npz`), and optional
`node_features.pt` / `edge_features.pt` artifacts.

`graph_direction=auto` follows TGL's official graph semantics: GDELT/MAG are
directed; WIKI/WIKIPEDIA/REDDIT/MOOC/LASTFM include reverse adjacencies. The
runtime validates adjacency counts against that policy. `frost-data build-graph`
requires an explicit `--directed` or `--bidirectional` choice.

## Environment Variables (Performance)

```bash
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF
export HYDRA_FULL_ERROR=0
```
