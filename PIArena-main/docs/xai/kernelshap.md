---
title: Kernel SHAP
slug: xai/kernelshap
category: xai
---

# Kernel SHAP

`kernelshap` explains a classifier-based defense's decision using word-level Kernel SHAP values. Two coexisting computation backends are available — [`captum.attr.KernelShap`](https://captum.ai/api/kernel_shap.html) (default) and the official [`shap.KernelExplainer`](https://shap.readthedocs.io/) — both implementing the same Kernel SHAP method (Lundberg & Lee, 2017 — "A Unified Approach to Interpreting Model Predictions"). Currently supports the `promptguard` defense. Only runs on samples the defense actually **blocked** (`detect_flag=True`) — explaining a decision that let the prompt through is out of scope.

Source:
[piarena/xai/kernelshap/](https://github.com/sleeepeer/PIArena/tree/main/piarena/xai/kernelshap)

Design doc:
[plans/xai-kernelshap-promptguard.md](../../plans/xai-kernelshap-promptguard.md)

Model:
[Meta Prompt Guard](https://huggingface.co/meta-llama/Prompt-Guard-86M)

## How To Use

```bash
python main.py --dataset squad_v2 --attack direct --defense promptguard --xai kernelshap --limit 25 --name xai_pilot
```

Works the same way for the other heuristic attacks (`ignore`/`completion`/`combined`/`character`) — the injected-span localization accounts for each one's fixed prefix automatically.

Or via a YAML config's `xai`/`xai_config` keys (same pattern as `attack_config`/`defense_config` — no dedicated CLI flag for the config dict, **except `backend`**, see below):

```yaml
xai: kernelshap
xai_config:
  backend: captum   # or "shap" — see "Backends" below
  n_samples: auto
  baseline_token: mask
  explain_targets: [context]
```

## Backends

`backend` (default `"captum"`) picks which library actually solves the Kernel SHAP regression — `"captum"` (`captum.attr.KernelShap`) or `"shap"` (official `shap.KernelExplainer`, `link="identity"`, a single all-masked background row — the same fixed-baseline semantics as the captum path, so the two are directly comparable). Everything else (tokenization, word-level feature grouping, masking, the output contract below) is identical between the two — switching is a one-line config change, never a code edit. `xai_result["context"]["backend"]` records which one produced a given result.

`--xai_backend captum|shap` is a dedicated CLI shortcut for this one config key (`main.py`'s only per-key CLI override of an `xai_config` value) — it lets you switch backends without writing a YAML `--config` file, and always wins over a `backend` set inside one:

```bash
python main.py --dataset squad_v2 --attack direct --defense promptguard --xai kernelshap --xai_backend captum --name kernelshap_captum_pilot --seed 42
python main.py --dataset squad_v2 --attack direct --defense promptguard --xai kernelshap --xai_backend shap   --name kernelshap_shap_pilot   --seed 42
```

**Note the two runs above use different `--name` values.** The result filename (`{dataset}-{llm}-{attack}-{defense}-{xai}-{seed}.json`) does not encode `backend` — running both backends under the same `--name`/dataset/attack/model/seed makes the second run either no-op (if the first already finished — `main.py` sees a complete file and skips) or, worse, fill in only the missing indices, leaving a single file with a *mix* of `captum`- and `shap`-produced rows. Always give each backend's run its own `--name` (or otherwise vary the output path) if you intend to compare them afterward.

### Comparing backends

`scripts/xai_compare_backends.py` reads two such raw `main.py` result files (one per backend, same dataset/attack/seed) and reports how much their per-word Shapley values agree — Pearson/Spearman correlation, sign agreement, top-k word overlap, and the `injected_span_percentile` gap between backends, per sample and aggregated:

```bash
python scripts/xai_compare_backends.py \
  --captum-result results/evaluation_results/kernelshap_captum_pilot/squad_v2-...-kernelshap-42.json \
  --shap-result   results/evaluation_results/kernelshap_shap_pilot/squad_v2-...-kernelshap-42.json \
  --out-dir results_comparison/direct_backend_agreement
```

Writes `backend_agreement_metrics.json`, `backend_agreement_report.md`, and a `backend_agreement.png` histogram of per-sample Spearman rho under `--out-dir`. See `plans/xai-kernelshap-backend-agreement.md` for the full rationale (why this is a useful cross-check given Kernel SHAP is a sampling-based approximation, and the known `1e6` endpoint-weight difference noted below).

## What It Does

For each sample the defense blocked, it explains **the whole `context`** that `PromptGuardDefense` actually classifies (not just the injected span) — a per-word importance score for `P(malign) = P(INJECTION) + P(JAILBREAK)`, the same aggregate the defense's own `detect_flag` is based on. The `context` is explained as a **single joint coalition game** covering every word at once — Kernel SHAP's cost is `n_samples` (a chosen budget, not derived from the text's structure), so no per-sentence decomposition is needed to keep it tractable (unlike the parallel SyntaxSHAP experiment).

The interpretable feature is a **word**, not a raw subword token: a fast tokenizer's `.word_ids()` groups every subtoken of a word (a URL, a non-Latin script) into one feature automatically, so fragmentation never inflates the number of features to explain.

The exact text that gets injected depends on the attack heuristic — `direct` inserts `injected_task` verbatim, while `ignore`/`completion`/`combined`/`character` prepend a fixed prefix (e.g. `"Ignore previous instructions. "`) first. `injected_span` localization knows about these prefixes (read directly off each attack class's `PREFIX` attribute in `piarena/attacks/heuristic.py` — a single source of truth, not a second copy of the template strings) and falls back to matching bare `injected_task` for any attack it doesn't recognize (e.g. the optimization-based ones).

Output, per sample, under `xai_result["context"]`:

```json
{
  "tokens": ["...", "..."],
  "values": [0.42, 0.01, "..."],
  "base_value": 0.31,
  "full_value": 0.97,
  "p_malign": 0.97,
  "predicted_label": "INJECTION",
  "backend": "captum",
  "injected_span": {"start_token": 42, "end_token": 58}
}
```

`values[i]` corresponds to `tokens[i]` (one entry per word) — a signed Shapley value (positive = pushes toward "malign"/attack, negative = pushes toward "benign"). Note `captum.attr.KernelShap` approximates the coalition-endpoint weights with a large-but-finite value (`1e6`) rather than the paper's literal infinity, so `sum(values)` will not exactly equal `full_value - base_value` — treat as a small expected residual, not a bug.

Samples the defense did **not** block get `xai_result: null` — no explanation is computed for them.

## Parameters

`DEFAULT_CONFIG`:

- `model_name` (default `"meta-llama/Prompt-Guard-86M"`) — the classifier to explain. Loaded directly via `AutoModelForSequenceClassification` (not `PromptGuardDefense`'s own `pipeline()` instance), so every class's logit is available and coalitions can be evaluated as token-id tensors without re-tokenizing text.
- `backend` (default `"captum"`) — `"captum"` or `"shap"`, see "Backends" above.
- `explain_targets` (default `["context"]`) — which span(s) to explain. `"target_inst"` and `"injected_task"` are also accepted (explained in isolation via `explain_row`).
- `n_samples` (default `"auto"`) — number of sampled coalitions per explained span, resolved once M (word count) is known to `min(2*M + 2048, max_n_samples)`. Used by both backends (never left at captum's own default of `25` — far too low for M beyond a handful of words).
- `max_n_samples` (default `20000`) — safety cap on the resolved `n_samples`.
- `perturbations_per_eval` (default `8`) — batch size for scoring sampled coalitions: captum's own native batching knob for that backend, and the chunk size used internally to batch the `shap` backend's `f(X)`.
- `baseline_token` (default `"mask"`) — which special token stands in for a "missing" word (`"mask"` | `"pad"` | `"unk"`, falls back through that order if the preferred one isn't defined on the tokenizer). This is a fixed reference value, not a sampled background distribution.
- `max_length` (default `None`) — truncation length. `None` (the default) means **no truncation**: `context` is scored in full, matching `PromptGuardDefense.execute`'s own untruncated scoring (it calls the HF `pipeline` with no truncation kwarg at all). Set an explicit int only to bound cost for very long contexts — this is an opt-in accuracy/cost tradeoff, since a truncated explanation may no longer cover the same text the defense actually saw if the injected task falls past the cutoff; using it logs a warning. An explicit value is still guarded against the tokenizer's own absurd `model_max_length` sentinel default.

## Analysis

`scripts/xai_metrics.py` computes Fidelity(t)/acc@1, the injected-span rank percentile, an ASR/Utility summary over the whole run, and renders plots from a result file:

```bash
python scripts/xai_metrics.py --result results/evaluation_results/xai_pilot/squad_v2-...-kernelshap-42.json
```

Plots written to `<out-dir>/plots/`:
- `saliency_maps.html` — `shap.plots.text()` per explained sample.
- `fidelity_acc_bars.png` — mean Fidelity(t)/acc@1(t) per threshold, across every explained (blocked) sample.
- `alignment_histogram.png` — distribution of `injected_span_percentile` across the same samples.

The Markdown report ends with an ASR/Utility summary computed over the *entire* raw result (every dataset sample, not just the ones XAI explained) — the benchmark-level numbers the per-sample XAI metrics sit inside, split into overall / blocked / not-blocked — followed by a per-`category` breakdown (Fidelity/acc@1 at the largest requested threshold, plus ASR/Utility over the whole category) when the dataset provides a `category` field (e.g. `squad_v2`'s phishing/access-denial/content-promotion/infrastructure-failure labels); omitted entirely for datasets without one. Each `saliency_maps.html` sample header also names the category alongside the attack.

See `plans/xai-kernelshap-promptguard.md` for what each metric means and why it's defined that way.
