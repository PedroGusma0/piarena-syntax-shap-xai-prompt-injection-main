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

`scripts/xai_metrics.py` computes faithfulness metrics (Fidelity(t), acc@1, alignment with the injected span) from a `--xai` result file, and renders `shap`-library saliency maps (`shap.plots.text`/`shap.plots.bar`). It infers which method produced a given result file from its filename (`--xai-method auto`, the default) — see [SyntaxSHAP](syntaxshap.md)/`plans/xai-syntaxshap-promptguard.md` and [Kernel SHAP](kernelshap.md)/`plans/xai-kernelshap-promptguard.md` for the full design of each.

## `--attack none` (benign-only run)

With `--attack none` every prompt is benign, so the XAI phase explains **all** samples (blocked and not blocked), not only the ones the defense blocked. `scripts/xai_metrics.py` then:

- skips the injected-span alignment metric (nothing was injected);
- labels each saliency map BLOQUEADO / NÃO BLOQUEADO with P(benign) and P(malign). The explained target is still only P(malign); P(benign) = 1 − P(malign).

Results go in the same `--name` folder as the other attacks (`...-none-promptguard-kernelshap-42.json`).

## Which samples are explained (`explain_scope`)

The XAI method decides (`BaseXAI.should_explain`), via its `explain_scope` setting: `auto` (default — blocked samples, plus every sample with `--attack none`), `blocked`, or `all` (blocked and not blocked, any attack). Switch it in the plugin's `DEFAULT_CONFIG`, in the YAML `xai_config`, or with `--xai_scope`.

## Confusion matrix (every run)

`scripts/xai_metrics.py` writes a confusion matrix with **positive = malign** (predicted malign = blocked) to `_metrics.json`, `_report.md` and `plots/<stem>/confusion_matrix.png`, over the whole raw result. The true label comes from the attack: `--attack none` ⇒ every prompt benign (only FP/TN are filled), any other attack ⇒ every prompt malign (only TP/FN). Also reported: `recall_malign = TP/(TP+FN)` and `fpr = FP/(FP+TN)` (`null` when undefined). Older results used positive = benign (`recall_benign`); regenerate them with `xai_metrics.py`.

## Alignment with the injected span (`injected_top_mass_precision`)

Fraction of the *most important* tokens that lie inside the injected span. "Most important" = the smallest set of tokens, in decreasing |SHAP| order, whose |SHAP| adds up to at least a fraction `m` of the total (`--alignment-mass`, default `0.5`; several values allowed). A single token with a dominant SHAP value (e.g. "Ignore") inside the span gives a high score even if the rest of the span has low values. It only needs `values` + `injected_span` already stored in the result, so it can be recomputed on old results by re-running `xai_metrics.py`. It replaces the former `injected_span_percentile` (mean percentile over the span's tokens). Skipped for `--attack none`.

## Which defense is explained

`main.py` records `defense` and `xai_model` on every explained sample; the saliency maps, bar plots, confusion matrix and report name them (`defense=piguard (leolee99/PIGuard)`). For older results the defense is read from the filename.

## NotInject (over-defense) standalone run

[`scripts/run_notinject_xai.py`](../../scripts/run_notinject_xai.py) explains PromptGuard on the three NotInject subsets from [PIGuard](https://github.com/leolee99/PIGuard) (`datasets/NotInject_{one,two,three}.json`, 113 benign prompts each, one/two/three prompt-injection trigger words) without `main.py`: no attack, no victim LLM, no evaluator. It is the `--attack none` procedure above (every prompt explained; `--defense promptguard|piguard` picks the classifier), writing `main.py`-compatible results (`results/evaluation_results/<name>/notinject_<subset>-novictim-none-<defense>-kernelshap-<seed>.json`, resumable) and then running `scripts/xai_metrics.py` and `scripts/xai_shap_plots_test.py` on each. `word_list` (the trigger words) and `category` are kept in each sample for later analysis.

```bash
python scripts/run_notinject_xai.py --subsets one --limit 5 --name notinject_smoke   # smoke test
python scripts/run_notinject_xai.py                                                  # all 3 subsets
```
