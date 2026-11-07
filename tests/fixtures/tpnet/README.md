# TPNet upstream reference fixtures

These are small synthetic CPU regression fixtures, not experiment results.
They were generated with PyTorch 2.9.0 and one CPU thread from:

- Classic: `lxd99/TPNet@7dd0cf49695a581c5c541baeda47c1b5e98e8748`
- TGB: `lxd99/TGB_TPNet@86d84e26eda708443fb74ea48c4846edd294a296`
- Import dependency for the TGB reference only:
  `shenyangHuang/TGB@3b83e5cfb2010743fd4f7a80a01b0e19dfcf78cb`

To regenerate, check out these exact revisions into sibling `TPNet`,
`TGB_TPNet`, and `TGB` directories outside the Frost repository. Then run:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python tests/test_tpnet_backend.py \
  --upstream /path/to/TPNet --output tests/fixtures/tpnet/classic.json
CUDA_VISIBLE_DEVICES='' uv run --no-sync python tests/test_tpnet_backend.py \
  --upstream /path/to/TGB_TPNet --tgb --output tests/fixtures/tpnet/tgb.json
```

Regenerating the TGB reference also requires its upstream TGB dependencies.
Normal regression tests require neither those dependencies nor any external checkout.
Never regenerate the committed reference using the extracted backend itself.
`--full` captures full gradient/parameter tensors for a one-off detailed
comparison; normal fixtures use sums, squared norms and boundary values.

Each training case uses two epochs of two batches, repeated nodes, equal
interaction timestamps, fixed seeds, dropout, Adam, and state reset/restore.
Classic/TGB output widths differ. Cases cover disabled RP, random projections,
explicit matrices, scaling, encoder/decoder RP ablations, disabled encoder
scores, and TGB's disabled embedding path. Sampling fixtures cover all three
neighbor policies and each variant's negative strategies, including repeated
calls to check RNG progression.

CPU instruction paths can differ in FP32 reductions and Adam updates even with
one thread and fixed seeds. AVX2 versus the original host produced gradient
summary differences up to 3.1e-5 after four updates. Gradient and parameter
summaries therefore use `rtol=1e-4, atol=2e-5`; losses, logits and projection state
retain `rtol=2e-5, atol=2e-6`, and sampling outputs remain exact. The committed
upstream reference values are unchanged. To exercise a different CPU path:

```bash
CUDA_VISIBLE_DEVICES='' ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  uv run --no-sync pytest tests/test_tpnet_backend.py -q
```
