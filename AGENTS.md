# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Sync the uv-managed project environment (builds the C++ seq_sampler extension)
uv sync --locked

# Rebuild the editable package after changing C++ seq_sampler sources
uv pip install -e .

# Run training — nproc_per_node = n_gpu_workers + 1 (host process)
uv run torchrun --nproc_per_node=3 run.py \
  access_analyzer=memory_analyzer dataset=LASTFM encoder=tgn \
  mem_updater=gru

# Run tests
uv run pytest tests/ -v
```

## Model Commands

| Model | Command |
|-------|---------|
| TGN | `uv run torchrun --nproc_per_node=3 run.py access_analyzer=memory_analyzer dataset=LASTFM encoder=tgn mem_updater=gru` |
| JODIE | `uv run torchrun --nproc_per_node=3 run.py access_analyzer=memory_analyzer dataset=LASTFM encoder=jodie mem_updater=rnn` |
| APAN | `uv run torchrun --nproc_per_node=3 run.py access_analyzer=apan_memory_analyzer dataset=LASTFM encoder=apan mem_updater=apan` |
| TGAT | `uv run torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=tgat` |
| DyGFormer | `uv run torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=dygformer` |
| GraphMixer | `uv run torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=graph_mixer predictor=graph_mixer` |
| TCL | `uv run torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=tcl predictor=tcl` |
| CAWN | `uv run torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=cawn predictor=dygformer` |
| EdgeBank | `uv run python eval_edgebank.py --dataset LASTFM` |

**Config defaults** (`configs/config.yaml`): `access_analyzer` omitted on non-memory runs, `mem_updater=null`, `n_neg=1`, `ring_buf_size=32`, `macro_batch_size=4000`, `d_time=128`, `no_memory_queue_size=4`.

**Pair-aware predictor auto-switches** (`launcher.py`): if the user leaves the default `predictor=link`, `DyGFormer` and `CAWN` are auto-switched to `DyGFormerLinkPredictor`, and `TCL` is auto-switched to `PairAwareMergeLayerPredictor`.

## Architecture Overview

FROST is a distributed Temporal GNN training framework. Rank assignment: workers = ranks `0..n_gpus-1`, host/master = rank `world_size-1`.

**Entry point:** `run.py` → `frost.launcher.launch(cfg)` — initializes process group (gloo), splits into Host vs Worker based on rank.

### Host process (`src/frost/runtime/host/`)
- `_engine.py` — `Host` class: drives sampling, batching, and scatter to workers
- `analyzer.py` — `StaticAnalyzer`, `MemoryAnalyzer`, `APANMemoryAnalyzer`: compute `Access` (which nodes need mem/box/nfeat fetched)
- `_meta_batcher.py` — `MetaBatcher` thread
- `_planner.py` — `Planner` thread (nfeat/mem/box planning)
- `_scatter.py` — `Scatter` thread (dist to workers via `scatter_object_list`)

### Worker process (`src/frost/runtime/worker/`)
- `_engine.py` — `Worker` class: train/eval loops
- `_receiver.py` — `Receiver` thread (receives from host)
- `_h2d.py` — `H2D` thread (loads features to GPU)
- `_dynamic_assembler.py` — `DynamicAssembler` (merges shared + P2P memory)
- `_static_assembler.py` — `StaticAssembler`
- `_host_to_device.py` — H2D thread (loads features to GPU)
- `p2p_syncer/` — `MemHandler`, `BoxHandler`, `P2PQHandler`, writers

### Models (`src/frost/models/`)
- `_assembled_model.py` — `AssembledModel`: unified forward pass (mem_updater → inflate → encoder → postman → header)
- `mem_updater.py` — `MemoryUpdater` (GRU), `TransformerMemoryUpdater`
- `header.py` — `LinkPredictor`
- `encoder/` — `tgn.py`, `tgat.py`, `apan.py`, `jodie.py`, `dygformer.py`, `graph_mixer.py`, `tcl.py`, `cawn.py`

### Shared memory / sampling (`src/frost/shm/`, `src/frost/sampling/`)
- `ring_buf.py` — `RingBuf`: circular shared-memory sampling buffer
- `mem_mgr.py` — `MemBoxStoreManager`: node memory + mailbox
- `dataset_n_feat.py` — `DatasetManager`: loads nfeat/efeat into `/dev/shm`
- `ngh_sampler.py` — `NoSampler` (0-hop), `NghSampler` (1-hop), `MultiHopNghSampler` (multi-hop; `uniform`, `recent`, `time_interval_aware`)

### Data types (`src/frost/datatypes/`)
- `meta.py` — `Meta`: batch metadata carrier
- `access.py` — `Access`: nfeat/mem/box access patterns per batch
- `graph.py` — `FlatGraph`, `Graph` (inflated), `NeighborOnHop`
- `env.py` — `Env`: distributed rank/device info

## Data Flow

```
Host process
 ├─ ngh_sampler.sample(roots, ts) → RingBuf slot
 ├─ analyzer.count_access(src, dst, all_nid, fanout) → Access
 └─ scatter_object_list → Workers
                           ├─ Receiver (scatter_object_list)
                           ├─ H2D (load nfeat/efeat/mem/box → GPU)
                           ├─ [DynamicAssembler if mem_updater]
                           └─ AssembledModel.forward(meta)
                              ├─ mem_updater(flat_graph)
                              ├─ flat_graph.inflate(fanout, n_neg, sample_4_neg)
                              ├─ encoder(graph=meta.graph)
                              ├─ [postman.deliver() for APAN]
                              └─ header(h) → logits
```

**FlatGraph layout in RingBuf slot:**
- `[0 : n_src*(2+n_neg)]` → root nodes: `[src | pos_dst | neg_dst×n_neg]`
- `[n_src*(2+n_neg) : ...]` → neighbor tiers per fanout hop
  - `sample_4_neg=True`: tier_k size = `prev_tier * fanout[k]` (all roots sampled)
  - `sample_4_neg=False`: tier_0 size = `n_src * 2 * fanout[0]` (src+pos_dst only, APAN)

## Key Invariants

### Dimension alignment
- `d_mem` in `mem_updater/*.yaml` **must equal** `${encoder.out_dim}` for identity-GNN models (JODIE, APAN).
- APAN's effective 1-hop fanout is `mem_updater.mail_recipients`; all other models derive fanout from `encoder.fanout`.
- APAN's neighborhood sampling strategy is `mem_updater.postman_delivery_strategy`; all other models derive it from `encoder.ngh_sampler_strategy`.
- `d_mail = 2*d_mem + edim` — computed in `launcher.py`; mailbox slot size and `TransformerMemoryUpdater` key/value dim must agree.
- `GraphMixer`: `fanout[0]` is the total neighbors sampled (node encoder uses all, link encoder uses `num_tokens`). `num_tokens` controls the link encoder neighbor count (default 20). `predictor=graph_mixer` required for DyGLib-compatible concat predictor.
- `TCL`: `num_tokens` should equal `fanout[0]`; `num_depths = fanout[0] + 1`.
- `CAWN`: default encoder config now uses `ngh_sampler_strategy=time_interval_aware` with `time_scaling_factor=1e-6`; the sampler weight lives in the sampler wrapper, not the encoder module.

### Predictor layout compatibility
- `DyGFormer` and `CAWN` emit pair-aware root layouts `[pos_src | pos_dst | neg_src | neg_dst]`; plain `LinkPredictor` is incompatible, so `launcher.py` auto-switches the default predictor to `DyGFormerLinkPredictor`.
- `TCL` also consumes pair-aware positive/negative pairs, but expects merge-layer scoring; `launcher.py` auto-switches the default predictor to `PairAwareMergeLayerPredictor`.
- `GraphMixer` still uses its explicit concat predictor for DyGLib parity; keep `predictor=graph_mixer`.

### Analyzer ↔ Runtime consistency (APAN)
`APANMemoryAnalyzer.count_access()` must exactly match what `Postman.deliver()` + `merge_1_mail()` writes at runtime:
1. `ngh_start = n_src * (2 + n_neg)` — not `n_src * (2*n_neg + 2)`
2. Filter padding: `pos_ngh = pos_ngh_raw[pos_ngh_raw != -1]`
3. `box_update = TensorSet(pos_ngh)` only — roots are not written by postman

### Sampler invariants
- `NoSampler.fanout` is always `None` regardless of `encoder.fanout`; `_assembled_model.py` treats `None` as `[]`.
- The C++ `SeqSampler` supports `uniform`, `recent`, and `time_interval_aware`; `time_interval_aware` consumes `time_scaling_factor` from config.
- `time_scaling_factor` is sampler-only config. `launcher.py` must strip it from `encoder_cfg` before instantiating the Python encoder.
- `MultiHopNghSampler.sample()` receives `ts` of shape `[n_src]`; must expand to `ts.repeat(2 + n_neg)` before calling C++ `get_historical_neighbors()` (positional-only args).

### Memory model detached states
`TransformerMemoryUpdater.forward()` must write cached states to `flat_graph.detached_*`, not `self.detached_*`. Downstream `_assembled_model.py` reads from `flat_graph.detached_nid/ts/mem`.

## Queue Sizes

| Queue | No memory (TGAT) | With memory (TGN/JODIE/APAN) |
|-------|-----------------|------------------------------|
| `Worker.full_q` | 4 | 1 |
| `Worker.received_q` | 4 | 1 |
| `Host.meta_in_q` | 4 | 2 |
| `Host.scatter_q` | 4 | 1 |

For stateless/no-memory models, `no_memory_queue_size` defaults to `4` and currently fans out to `Worker.full_q`, `Worker.received_q`, `Host.scatter_q`, and `Host.nfeat_plan_out_q` when feature planning is enabled.

For memory models, `full_q=1` is intentional: `DynamicAssembler` must wait for P2P completion (`delta_q`) before putting assembled batches in `full_q`. Pipeline depth controlled by `h2d_sem` (released 4× at startup).

## Sweep Scripts

## Adding a New Encoder

1. Create `src/frost/models/encoder/my_model.py`:
   ```python
   class MyEncoder(nn.Module):
       def forward(self, graph: Graph) -> torch.Tensor:
           # graph.root_mem, root_mem_ts, root_ts, root_nfeat, root_nid
           # graph.neighbors[0].nid/ts/mem/efeat  (if 1-hop)
           return h  # shape [n_root, encoder.out_dim]
   ```
2. Create `configs/encoder/my_model.yaml`:
   ```yaml
   _target_: frost.models.encoder.my_model.MyEncoder
   out_dim: 128
   d_time: ${d_time}
   ndim: ${ndim}
   edim: ${edim}
   ```
3. Run with `run.py encoder=my_model`.

For memory encoders, implement `MemoryUpdaterProto` and set `d_mem: ${encoder.out_dim}`.

## Adding a New Analyzer

```python
class MyAnalyzer:
    def __init__(self, n_neg: int): ...
    def count_access(self, src, dst, all_nid, fanout, valid_all=None) -> Access:
        return Access(
            mem_need=...,    # nodes to read memory for
            mem_update=...,  # nodes whose memory changes
            box_need=...,    # nodes to read mailbox for
            box_update=...,  # nodes whose mailbox changes (must exactly match runtime writes)
            nfeat_need=...,  # nodes to load node features for
        )
```

Config: `configs/access_analyzer/my_analyzer.yaml` with `_target_` and `n_neg: ${n_neg}`.

## Datasets

Stored in `DATA/<DATASET>/`. Loaded directly by `DatasetManager`. Expected files: `edges.csv` (columns: `src`, `dst`, `time`, `default_split`, `Unnamed: 0`), optional `node_features.pt`, `edge_features.pt`.

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
