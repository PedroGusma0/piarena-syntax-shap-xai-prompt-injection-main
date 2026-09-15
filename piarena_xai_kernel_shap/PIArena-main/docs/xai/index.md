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
python main.py --dataset squad_v2 --attack direct --defense promptguard --xai kernelshap --limit 25 --name xai_pilot
```

`xai_result` is written into the same per-sample result JSON `main.py` already produces (`results/evaluation_results/{name}/...json`), alongside `defense_result`.

## Available Methods

- [Kernel SHAP](kernelshap.md) — `captum.attr.KernelShap`-based word-level Shapley values, for text-classification defenses (currently `promptguard`).

## Analysis Tooling

`scripts/xai_metrics.py` computes faithfulness metrics (Fidelity(t), acc@1, rank of the injected span) from a `--xai` result file, and renders `shap`-library saliency maps (`shap.plots.text`/`shap.plots.bar`). See [Kernel SHAP](kernelshap.md) and `plans/xai-kernelshap-promptguard.md` for the full design.
