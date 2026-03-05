# FROST: Distributed Temporal GNN Training Framework

## Build & Run

```bash
# Install package (builds C++ seq_sampler extension via scikit-build-core + nanobind)
pip install -e .

# Run training (nproc = n_gpu_workers + 1 host process)
torchrun --nproc_per_node=3 run.py \
  analyzer=tgn dataset=LASTFM encoder=tgn \
  mem_box_store=base mem_updater=tgn \
  ngh_sampler=1_hop mail_combine=last

# Run tests
pytest tests/ -v
```

**Rank assignment:** workers = ranks `0..n_gpus-1`, host/master = rank `world_size-1`.

---

## Project Structure

```
lei_pipe/
├── run.py                          # Entry point (Hydra @hydra.main)
├── configs/                        # Hydra YAML configs
│   ├── config.yaml                 # Global defaults
│   ├── analyzer/                   # {no_mem, tgn, apan, jodie}
│   ├── encoder/                    # {tgn, tgat, apan, jodie, dygformer}
│   ├── mem_updater/                # {tgn, apan, jodie} — null for TGAT
│   ├── mem_box_store/              # {base} — null for no-memory models
│   ├── ngh_sampler/                # {0_hop, 1_hop, 2_hop}
│   ├── postman/                    # {apan} — null for non-APAN
│   ├── ring_buf/, neg_sampler/, predictor/, loss_fn/, precision/
│   └── dataset_mgr/
└── src/frost/
    ├── launcher.py                 # launch(cfg): sets up dist, host/worker split
    ├── models/
    │   ├── _assembled_model.py     # AssembledModel: unified forward pass
    │   ├── mem_updater.py          # MemoryUpdater, TransformerMemoryUpdater
    │   ├── header.py               # LinkPredictor (final prediction head)
    │   └── encoder/
    │       ├── tgn.py, tgat.py, apan.py, jodie.py
    │       └── modules/            # TimeEncoder, attention layers
    ├── runtime/
    │   ├── host/
    │   │   ├── _engine.py          # Host class: sampling, batching, scatter
    │   │   ├── analyzer.py         # NoMemAnalyzer, TGNAnalyzer, APANAnalyzer, JODIEAnalyzer
    │   │   ├── _meta_batcher.py    # MetaBatcher thread
    │   │   ├── _planner.py         # Planner thread (nfeat/mem/box planning)
    │   │   └── _scatter.py         # Scatter thread (dist to workers)
    │   └── worker/
    │       ├── _engine.py          # Worker class: train/eval loops
    │       ├── _receiver.py        # Receiver thread (scatter_object_list)
    │       ├── _h2d.py             # H2D thread (features → GPU)
    │       ├── _dy_asm.py          # DyAssembler (merge shared+P2P memory)
    │       ├── postman.py          # Postman.deliver() for APAN mail
    │       └── p2p_syncer/         # MemHandler, BoxHandler, P2PQHandler, writers
    ├── sampling/
    │   ├── ngh_sampler.py          # NoSampler (0-hop), NghSampler (1/2-hop), MultiHopNghSampler
    │   └── neg_sampler.py          # NegLinkSampler
    ├── shm/
    │   ├── ring_buf.py             # RingBuf: circular shared-memory sampling buffer
    │   ├── mem_mgr.py              # MemBoxStoreManager: node memory + mailbox
    │   └── dataset_n_feat.py       # DatasetManager: loads nfeat/efeat into /dev/shm
    ├── datatypes/
    │   ├── graph.py                # FlatGraph, Graph (inflated), NeighborOnHop
    │   ├── meta.py                 # Meta: batch metadata carrier
    │   ├── access.py               # Access: nfeat/mem/box access patterns
    │   ├── env.py                  # Env: distributed rank/device info
    │   └── mem_mail_box/           # MemTsNid, BoxTsNid, mail ops, _data_ops.py
    ├── protocols/                  # Protocol interfaces (analyzer, sampler, mem_updater, postman)
    └── utils/
        └── tensor_set.py           # TensorSet: unique-node sets for planning
```

---

## Model Configurations

| Model  | Command flags | Memory | Sampling | Notes |
|--------|--------------|--------|----------|-------|
| TGN    | `analyzer=tgn encoder=tgn mem_box_store=base mem_updater=tgn ngh_sampler=1_hop mail_combine=last` | GRU | 1-hop | deliver_to=self |
| JODIE  | `analyzer=jodie encoder=jodie mem_box_store=base mem_updater=jodie ngh_sampler=0_hop mail_combine=last` | GRU | 0-hop | time projection |
| APAN   | `analyzer=apan encoder=apan mem_box_store=base mem_updater=apan ngh_sampler=1_hop ngh_sampler.sample_4_neg=False postman=apan mail_combine=last` | Transformer | 1-hop (src+dst only) | deliver_to=neighbors |
| TGAT   | `encoder=tgat ngh_sampler=2_hop` | none | 2-hop | no mem/box/postman |

**Config defaults** (`configs/config.yaml`): `analyzer=no_mem`, `mem_updater=null`, `mem_box_store=null`, `ngh_sampler=1_hop`, `fanout=[10]`, `n_neg=1`, `batch_size=4000`, `final_emb_dim=128`, `d_time=128`.

---

## Data Flow

```
Host process
 ├─ ngh_sampler.sample(roots, ts) → RingBuf slot
 ├─ analyzer.count_access(src, dst, all_nid, fanout) → Access
 └─ scatter_object_list → ──────────────────────►  Worker processes
                                                    ├─ Receiver (scatter_object_list)
                                                    ├─ H2D (load nfeat/efeat/mem/box → GPU)
                                                    ├─ [DyAssembler if mem_box_store]
                                                    ├─ full_q.get()
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

---

## Key Invariants

### Dimension alignment
- `d_mem` in `mem_updater/*.yaml` **must equal** `${final_emb_dim}` for identity-GNN models (JODIE, APAN) so encoder output matches `LinkPredictor` input.
- `d_mail = 2*d_mem + edim` — computed in `launcher.py`; mailbox slot size and `TransformerMemoryUpdater` key/value dim must agree.

### Analyzer ↔ Runtime consistency (for APAN)
`APANAnalyzer.count_access()` builds the P2P `box_update` plan. It must exactly match the set of nodes that `Postman.deliver()` + `merge_1_mail()` writes at runtime. Three rules:
1. `ngh_start = n_src * (2 + n_neg)` — not `n_src * (2*n_neg + 2)`
2. Filter padding: `pos_ngh = pos_ngh_raw[pos_ngh_raw != -1]`
3. `box_update = TensorSet(pos_ngh)` only — roots are not written by postman

### Sampler fanout
- `NoSampler.fanout` is always `None` regardless of global `cfg.fanout` — `_assembled_model.py` treats `None` as `[]` (0-hop, root tier only).
- `MultiHopNghSampler.sample()` receives `ts` of shape `[n_src]`; must expand to `ts.repeat(2 + n_neg)` before calling the C++ `get_historical_neighbors()` (positional-only args, no keyword args).

### Memory model detached states
`TransformerMemoryUpdater.forward()` must write cached states to **`flat_graph.detached_*`**, not `self.detached_*`. The downstream code in `_assembled_model.py` reads from `flat_graph.detached_nid/ts/mem`.

---

## Queue Sizes (conditional on memory model)

Queues are sized larger for no-memory models (no P2P sync overhead):

| Queue | No memory (TGAT) | With memory (TGN/JODIE/APAN) |
|-------|-----------------|------------------------------|
| `Worker.full_q` | 2 | 1 |
| `Worker.received_q` | 2 | 1 |
| `Host.meta_in_q` | 4 | 2 |
| `Host.scatter_q` | 2 | 1 |

For memory models, `full_q=1` is intentional: `DyAssembler` must wait for P2P completion (`delta_q`) before putting assembled batches in `full_q`. Pipeline depth is controlled by `h2d_sem` (released 4× at startup).

---

## Adding a New Encoder

1. **Model file:** `src/frost/models/encoder/my_model.py`
   ```python
   class MyEncoder(nn.Module):
       def forward(self, graph: Graph) -> torch.Tensor:
           # graph.root_mem, root_mem_ts, root_ts, root_nfeat, root_nid
           # graph.neighbors[0].nid/ts/mem/efeat  (if 1-hop)
           return h  # shape [n_root, final_emb_dim]
   ```
2. **Config:** `configs/encoder/my_model.yaml`
   ```yaml
   _target_: frost.models.encoder.my_model.MyEncoder
   d_mem: ${final_emb_dim}
   d_time: ${d_time}
   ndim: ${ndim}
   edim: ${edim}
   ```
3. **Run:** `run.py encoder=my_model`

For no-memory encoders, also create `configs/analyzer/no_mem.yaml` (already exists).
For memory encoders, implement `MemoryUpdaterProto` and set `d_mem: ${final_emb_dim}`.

---

## Adding a New Analyzer

Analyzers tell the host which nodes need mem/box/nfeat fetched and updated each batch.

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

---

## Datasets

Stored in `DATA/<DATASET>/`. Loaded by `DatasetManager` from `configs/dataset_mgr/base.yaml`. Expected files: `edges.csv` (columns: `src`, `dst`, `time`, `ext_roll`, `Unnamed: 0`), optional `node_features.pt`, `edge_features.pt`.

---

## Environment Variables

```bash
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
```
