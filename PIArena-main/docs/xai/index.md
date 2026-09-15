---
title: XAI
slug: xai
category: guide
---

# XAI (Explainability)

`--xai` is an optional, opt-in third phase in the evaluation pipeline, run right after the defense phase:

**Attack → Defense → XAI → Evaluation**

An XAI method explains *why* a defense made the decision it made on a given sample — which words in the input most influenced the classification. It never changes the attack, defense, or LLM behavior; it only adds a `xai_result` field to each sample's saved result.

Omit `--xai` and nothing changes from the existing pipeline.

## How To Use

```bash
python main.py --dataset squad_v2 --attack direct --defense promptguard --xai syntaxshap --limit 25 --name xai_pilot
```

`xai_result` is written into the same per-sample result JSON `main.py` already produces (`results/evaluation_results/{name}/...json`), alongside `defense_result`.

## Available Methods

- [SyntaxSHAP](syntaxshap.md) — syntax-tree-restricted Shapley values, adapted from the SyntaxSHAP paper for text-classification defenses (currently `promptguard`).
- [Kernel SHAP](kernelshap.md) — `captum.attr.KernelShap`/`shap.KernelExplainer`-based word-level Shapley values, for text-classification defenses (currently `promptguard`).

Both coexist in the same `XAI_REGISTRY` and share the identical `xai_result` output contract (tokens/values/base_value/full_value/injected_span) and the same `scripts/xai_metrics.py` analysis tooling below — switching between them is a `--xai` flag, not a different pipeline.

## Analysis Tooling

`scripts/xai_metrics.py` computes faithfulness metrics (Fidelity(t), acc@1, rank of the injected span) from a `--xai` result file, and renders `shap`-library saliency maps (`shap.plots.text`/`shap.plots.bar`). It infers which method produced a given result file from its filename (`--xai-method auto`, the default) — see [SyntaxSHAP](syntaxshap.md)/`plans/xai-syntaxshap-promptguard.md` and [Kernel SHAP](kernelshap.md)/`plans/xai-kernelshap-promptguard.md` for the full design of each.
