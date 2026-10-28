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
  target: logit          # logit (default) | prob
  n_steps: 100           # starting steps; doubled while the error > delta_tol
  max_n_steps: 800       # null = fixed n_steps
  delta_tol: 0.05
  internal_batch_size: 4
  baseline_token: mask   # mask | pad | unk
  explain_targets: [context]
```

## How it works

- Explained target: `P(malign)` = sum of every class whose label has no "benign" (same binary target as the other methods). IG itself runs on its **log-odds** (`target: logit`, default) and the values are rescaled by `(full_value − base_value) / (z_full − z_base)` to the probability scale, so Fidelity and the plots are unchanged. Reason: with `target: prob`, P goes ~0.0005 → ~1.0 in a thin slice of the path; real `squad_v2` runs (50 steps) gave `sum(values)` ≈ 0.005 against a gap ≈ 1.0 on `ignore` (99% of samples with >50% error). Ranking is unaffected by the rescale; magnitudes mean "share of the probability gap".
- Path: from a baseline (prefix/suffix special tokens kept, every content token replaced by the `baseline_token`) to the real input, in embedding space.
- Per-token attributions are summed over the embedding dimension and then over each word's subtokens, so `values` is word-aligned with `tokens` and the output contract (`tokens`/`values`/`base_value`/`full_value`/`p_malign`/`injected_span`) matches Kernel SHAP's. Extra fields: `ig_convergence_delta` (`sum(values) − (full_value − base_value)`), `ig_converged`, `ig_target`, `n_steps` (steps actually used), `n_words`.
- `scripts/xai_metrics.py` and the plot scripts work unchanged (`--xai-method ig`, inferred from the `-ig-` filename token); re-scoring uses the same word-level masking as Kernel SHAP.

## Caveats vs. Shapley-based methods

- **Not exactly additive.** `sum(values) ≈ full_value − base_value` only up to `ig_convergence_delta`; `shap.plots.text` shows `base_value + Σ values` as the output value, which then differs from `p_malign` when the delta is large. A warning is logged; raise `n_steps`.
- **Adaptive steps.** Starting at `n_steps`, the run doubles the steps (up to `max_n_steps`) until `|delta| ≤ delta_tol · |full_value − base_value|`; samples that still don't converge are flagged `ig_converged: false` (and logged). Cost per sample grows with the steps used. On synthetic saturated toy models the adaptive logit setting converged in 12/12 (mean 192 steps) versus 1/12 for fixed-50 `prob`; this is synthetic evidence, not a measurement on Prompt-Guard — check `ig_converged` in real results.
- **Saturation.** The target is close to 1.0 for most blocked injections; a coarse path can miss the sharp transition.
- **Sign/scale semantics.** First-order attribution along a straight path in embedding space (intermediate points are not real tokens); colors are not comparable in magnitude with Kernel SHAP's, only in ranking.
- **Subtoken sums.** Heavily fragmented words (URLs, CJK) accumulate more terms and tend to larger values.
- **Baseline-sensitive.** Compare `mask` and `pad`.
