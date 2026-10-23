---
title: Integrated Gradients
slug: xai/ig
category: xai
---

# Integrated Gradients

`ig` explains a classifier-based defense's decision with [Integrated Gradients](https://captum.ai/docs/extension/integrated_gradients) (Sundararajan et al., 2017), computed with `captum.attr.LayerIntegratedGradients` over the classifier's input embeddings. It is **white-box** (needs the classifier's weights, e.g. the gated `meta-llama/Prompt-Guard-86M` after `huggingface-cli login`) but its cost is a fixed `n_steps` forward/backward passes per text — it does not grow with the text's syntactic structure like [SyntaxSHAP](syntaxshap.md), nor with a sampling budget like [Kernel SHAP](kernelshap.md). Supports `promptguard` and `piguard` (`main.py` picks the classifier from `--defense`).

Source: `piarena/xai/ig/`. Design doc: [plans/xai-ig-promptguard.md](../../plans/xai-ig-promptguard.md).

## How To Use

```bash
python main.py --dataset squad_v2 --attack direct --defense promptguard --xai ig --limit 25 --name ig_pilot
```

```yaml
xai: ig
xai_config:
  n_steps: 50
  internal_batch_size: 4
  baseline_token: mask   # mask | pad | unk
  explain_targets: [context]
```

## How it works

- Explained target: `P(malign)` = sum of every class whose label has no "benign" (same binary target as the other methods).
- Path: from a baseline (prefix/suffix special tokens kept, every content token replaced by the `baseline_token`) to the real input, in embedding space.
- Per-token attributions are summed over the embedding dimension and then over each word's subtokens, so `values` is word-aligned with `tokens` and the output contract (`tokens`/`values`/`base_value`/`full_value`/`p_malign`/`injected_span`) matches Kernel SHAP's. Extra fields: `ig_convergence_delta`, `n_steps`, `n_words`.
- `scripts/xai_metrics.py` and the plot scripts work unchanged (`--xai-method ig`, inferred from the `-ig-` filename token); re-scoring uses the same word-level masking as Kernel SHAP.

## Caveats vs. Shapley-based methods

- **Not exactly additive.** `sum(values) ≈ full_value − base_value` only up to `ig_convergence_delta`; `shap.plots.text` shows `base_value + Σ values` as the output value, which then differs from `p_malign` when the delta is large. A warning is logged; raise `n_steps`.
- **`n_steps=50` is a starting point, not a guarantee.** On a randomly initialized toy DeBERTa (sharply non-linear, synthetic only — not Prompt-Guard) the convergence delta was ~1.8 at 50 steps, ~0.5 at 300 and ~1e-5 at 2000. Always check `ig_convergence_delta` on the real model and raise `n_steps` until it is small relative to `full_value − base_value`.
- **Saturation.** The target is close to 1.0 for most blocked injections; a coarse path can miss the sharp transition.
- **Sign/scale semantics.** First-order attribution along a straight path in embedding space (intermediate points are not real tokens); colors are not comparable in magnitude with Kernel SHAP's, only in ranking.
- **Subtoken sums.** Heavily fragmented words (URLs, CJK) accumulate more terms and tend to larger values.
- **Baseline-sensitive.** Compare `mask` and `pad`.
