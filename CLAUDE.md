# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Install package (builds C++ seq_sampler extension via scikit-build-core + nanobind)
pip install -e .

# Run training — nproc_per_node = n_gpu_workers + 1 (host process)
torchrun --nproc_per_node=3 run.py \
  analyzer=tgn dataset=LASTFM encoder=tgn \
  mem_box_store=base mem_updater=tgn \
  ngh_sampler=1_hop mail_combine=last

# Run tests
pytest tests/ -v
```

## Model Commands

| Model | Command |
|-------|---------|
| TGN | `torchrun --nproc_per_node=3 run.py analyzer=tgn dataset=LASTFM encoder=tgn mem_box_store=base mem_updater=tgn ngh_sampler=1_hop mail_combine=last` |
| JODIE | `torchrun --nproc_per_node=3 run.py analyzer=jodie dataset=LASTFM encoder=jodie mem_box_store=base mem_updater=jodie ngh_sampler=0_hop mail_combine=last` |
| APAN | `torchrun --nproc_per_node=3 run.py analyzer=apan dataset=LASTFM encoder=apan mem_box_store=base mem_updater=apan ngh_sampler=1_hop ngh_sampler.sample_4_neg=False postman=apan mail_combine=last` |
| TGAT | `torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=tgat ngh_sampler=2_hop` |
| DyGFormer | `torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=dygformer ngh_sampler=1_hop fanout=[32]` |
| GraphMixer | `torchrun --nproc_per_node=3 run.py dataset=LASTFM encoder=graph_mixer ngh_sampler=1_hop fanout=[20]` |
| EdgeBank | `python eval_edgebank.py --dataset LASTFM` |

**Config defaults** (`configs/config.yaml`): `analyzer=no_mem`, `mem_updater=null`, `mem_box_store=null`, `ngh_sampler=1_hop`, `fanout=[10]`, `n_neg=1`, `batch_size=4000`, `final_emb_dim=128`, `d_time=128`.

## Architecture Overview

FROST is a distributed Temporal GNN training framework. Rank assignment: workers = ranks `0..n_gpus-1`, host/master = rank `world_size-1`.

**Entry point:** `run.py` → `frost.launcher.launch(cfg)` — initializes process group (gloo), splits into Host vs Worker based on rank.

### Host process (`src/frost/runtime/host/`)
- `_engine.py` — `Host` class: drives sampling, batching, and scatter to workers
- `analyzer.py` — `NoMemAnalyzer`, `TGNAnalyzer`, `APANAnalyzer`, `JODIEAnalyzer`: compute `Access` (which nodes need mem/box/nfeat fetched)
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
- `encoder/` — `tgn.py`, `tgat.py`, `apan.py`, `jodie.py`, `dygformer.py`, `graph_mixer.py`, `tcl.py`

### Shared memory / sampling (`src/frost/shm/`, `src/frost/sampling/`)
- `ring_buf.py` — `RingBuf`: circular shared-memory sampling buffer
- `mem_mgr.py` — `MemBoxStoreManager`: node memory + mailbox
- `dataset_n_feat.py` — `DatasetManager`: loads nfeat/efeat into `/dev/shm`
- `ngh_sampler.py` — `NoSampler` (0-hop), `NghSampler` (1/2-hop), `MultiHopNghSampler`

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
                           ├─ [DynamicAssembler if mem_box_store]
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
- `d_mem` in `mem_updater/*.yaml` **must equal** `${final_emb_dim}` for identity-GNN models (JODIE, APAN).
- `d_mail = 2*d_mem + edim` — computed in `launcher.py`; mailbox slot size and `TransformerMemoryUpdater` key/value dim must agree.
- `GraphMixer`/`TCL`: `num_tokens` should equal `fanout[0]`; `TCL num_depths = fanout[0] + 1`.

### Analyzer ↔ Runtime consistency (APAN)
`APANAnalyzer.count_access()` must exactly match what `Postman.deliver()` + `merge_1_mail()` writes at runtime:
1. `ngh_start = n_src * (2 + n_neg)` — not `n_src * (2*n_neg + 2)`
2. Filter padding: `pos_ngh = pos_ngh_raw[pos_ngh_raw != -1]`
3. `box_update = TensorSet(pos_ngh)` only — roots are not written by postman

### Sampler invariants
- `NoSampler.fanout` is always `None` regardless of `cfg.fanout`; `_assembled_model.py` treats `None` as `[]`.
- `MultiHopNghSampler.sample()` receives `ts` of shape `[n_src]`; must expand to `ts.repeat(2 + n_neg)` before calling C++ `get_historical_neighbors()` (positional-only args).

### Memory model detached states
`TransformerMemoryUpdater.forward()` must write cached states to `flat_graph.detached_*`, not `self.detached_*`. Downstream `_assembled_model.py` reads from `flat_graph.detached_nid/ts/mem`.

## Queue Sizes

| Queue | No memory (TGAT) | With memory (TGN/JODIE/APAN) |
|-------|-----------------|------------------------------|
| `Worker.full_q` | 2 | 1 |
| `Worker.received_q` | 2 | 1 |
| `Host.meta_in_q` | 4 | 2 |
| `Host.scatter_q` | 2 | 1 |

For memory models, `full_q=1` is intentional: `DynamicAssembler` must wait for P2P completion (`delta_q`) before putting assembled batches in `full_q`. Pipeline depth controlled by `h2d_sem` (released 4× at startup).

## Adding a New Encoder

1. Create `src/frost/models/encoder/my_model.py`:
   ```python
   class MyEncoder(nn.Module):
       def forward(self, graph: Graph) -> torch.Tensor:
           # graph.root_mem, root_mem_ts, root_ts, root_nfeat, root_nid
           # graph.neighbors[0].nid/ts/mem/efeat  (if 1-hop)
           return h  # shape [n_root, final_emb_dim]
   ```
2. Create `configs/encoder/my_model.yaml`:
   ```yaml
   _target_: frost.models.encoder.my_model.MyEncoder
   d_mem: ${final_emb_dim}
   d_time: ${d_time}
   ndim: ${ndim}
   edim: ${edim}
   ```
3. Run with `run.py encoder=my_model`.

For memory encoders, implement `MemoryUpdaterProto` and set `d_mem: ${final_emb_dim}`.

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

Config: `configs/analyzer/my_analyzer.yaml` with `_target_` and `n_neg: ${n_neg}`.

## Datasets

Stored in `DATA/<DATASET>/`. Loaded by `DatasetManager` from `configs/dataset_mgr/base.yaml`. Expected files: `edges.csv` (columns: `src`, `dst`, `time`, `ext_roll`, `Unnamed: 0`), optional `node_features.pt`, `edge_features.pt`.

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
