# Negative-sampling protocols

Negative-sampling results are comparable only when the split, candidate set,
corruption rule, number of negatives, replacement rule, random seed, and metric
are the same. FROST therefore names its protocols by their actual behavior
rather than treating `historical` or `inductive` as universal definitions.

## FROST's destination-corruption layout

For a positive event `(src, dst, time)`, FROST's common link-prediction layout
is:

```text
[src | positive dst | negative dst x n_neg]
```

The predictor repeats the positive source and scores `(src, negative dst)`.
Consequently, the current runtime can express destination corruption but cannot
represent an independently sampled `negative src`. Supporting full-pair
negative edges would require coordinated changes to the sampling buffer, graph
layout, encoders, predictors, and analyzers—not just another sampler class.

## Shipped protocols

| Config | Training | Evaluation | Interpretation |
| --- | --- | --- | --- |
| `rnd_dst_then_random` (default) | Uniform random destination from all node IDs in `[0, num_nodes)` | Runtime random sampling from the same full node range | Sampling with replacement; no positive-destination collision filtering |
| `rnd_dst_then_fixed` | Random destination from the observed destination partition | Fixed `eid,neg_dst` artifact | Reproducible fixed random destination corruption; currently `n_neg=1` |
| `rnd_dst_then_hist` | Random destination from the observed destination partition | Prefer destinations previously connected to the current source, strictly before the batch; exclude that source's current positive destinations; otherwise fall back to the observed destination partition | Per-source historical destination corruption |
| `rnd_dst_then_post_train` | Random destination from the observed destination partition | Prefer destinations connected to the current source after the training cutoff and strictly before the batch; exclude current positives; otherwise fall back to the observed destination partition | Per-source post-training destination corruption |
| `rnd_dst_then_tgb` / `rnd_dst_then_tgb_wiki` | Random destination | Official TGB validation/test candidates | TGB protocol; use its required metric and candidate width |

Historical and post-training destination candidates are sampled with replacement,
so a positive edge may receive the same negative destination more than once
when `n_neg > 1`.

## Historical difference from upstream DyGLib

The original DyGLib snapshot (`3aacc36b`) gave `historical` and `inductive`
different semantics:

- **Historical:** form the global set of historical `(src, dst)` edges, remove
  edges in the current batch, and sample complete negative pairs. If that set
  is too small, fill the remainder with collision-checked random pairs.
- **Inductive:** form the global historical edge set, remove edges observed by
  the training cutoff and edges in the current batch, and sample complete
  negative pairs, again with random-pair fallback when needed.

These are full-pair edge-classification protocols: both `negative src` and
`negative dst` may differ from the positive event. FROST's corresponding
protocols condition on the current source and rank alternative destinations.
Neither task definition is inherently correct or incorrect, but their metrics
must not be compared as if they used the same negatives.

DyGLib's random link-prediction path is the close case: its training and
evaluation call sites discard the random sampler's source output and combine
the current batch source with a sampled destination. With FROST configured for
the observed destination partition, the task semantics match. The concrete
sample sequence can still differ because DyGLib and FROST use different RNG
implementations.

The relevant implementations are:

- FROST: [`historical.py`](../src/frost/neg_sampler/historical.py),
  [`inductive.py`](../src/frost/neg_sampler/inductive.py), and
  [`random.py`](../src/frost/neg_sampler/random.py)
The vendored DyGLib fork no longer carries those unused protocols. Its sole
FROST-aligned entrypoint uses the fixed evaluation artifact and global AP; see
[`PROTOCOL.md`](../third_party/DyGLib/PROTOCOL.md).

## Reporting and comparison

For a result intended for comparison, report at least:

1. The temporal split policy, including how tied timestamps are handled.
2. The exact protocol name from the table above—not merely `historical` or
   `inductive`.
3. The candidate node/edge set and whether current positives are excluded.
4. Whether sampling uses replacement and the number of negatives per positive.
5. The seed, or the fixed-negative artifact and its checksum.
6. The metric and its reduction/tie policy.

All models within one experiment should consume the same split and negative
queries. FROST historical/inductive results should be labeled with their
per-source destination-corruption names and should not be presented as direct
DyGLib historical/inductive comparisons.

Published `frost_dataset` manifests describe the fixed negative artifact only.
Runtime sampler selection and the definitions above are owned by FROST.
