---
title: SyntaxSHAP
slug: xai/syntaxshap
category: xai
---

# SyntaxSHAP

`syntaxshap` explains a classifier-based defense's decision using syntax-tree-restricted Shapley values — an adaptation of the [SyntaxSHAP paper](https://arxiv.org/abs/2402.09259) (originally designed to explain next-token generation by autoregressive LLMs) to text-classification defenses instead. Currently supports the `promptguard` defense. Only runs on samples the defense actually **blocked** (`detect_flag=True`) — explaining a decision that let the prompt through is out of scope.

Source:
[piarena/xai/syntaxshap/](https://github.com/sleeepeer/PIArena/tree/main/piarena/xai/syntaxshap)

Design doc:
[plans/xai-syntaxshap-promptguard.md](../../plans/xai-syntaxshap-promptguard.md) — **nota:** este link está quebrado (arquivo nunca existiu no repo, achado ao portar os ajustes do experimento Kernel SHAP pra este experimento). Este documento (`docs/xai/syntaxshap.md`) é a referência disponível de fato.

Model:
[Meta Prompt Guard](https://huggingface.co/meta-llama/Prompt-Guard-86M)

## How To Use

```bash
python main.py --dataset squad_v2 --attack direct --defense promptguard --xai syntaxshap --limit 25 --name xai_pilot
```

Works the same way for the other heuristic attacks (`ignore`/`completion`/`combined`/`character`) — the injected-span localization accounts for each one's fixed prefix automatically.

Or via a YAML config's `xai`/`xai_config` keys (same pattern as `attack_config`/`defense_config` — no dedicated CLI flag for the config dict):

```yaml
xai: syntaxshap
xai_config:
  algorithm: syntax
  explain_targets: [context]
```

## What It Does

For each sample the defense blocked, it explains **the whole `context`** that `PromptGuardDefense` actually classifies (not just the injected span) — a per-token importance score for `P(malign) = P(INJECTION) + P(JAILBREAK)`, the same aggregate the defense's own `detect_flag` is based on. Long, multi-sentence `context` is handled by segmenting into sentences and explaining each one independently with the rest of the context held fixed (the scaling approach the SyntaxSHAP paper itself suggests for multi-sentence text).

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
  "injected_span": {"start_token": 42, "end_token": 58}
}
```

`values[i]` corresponds to `tokens[i]` — a signed importance score (positive = pushes toward "malign"/attack, negative = pushes toward "benign"). Not an exact additive decomposition of `full_value - base_value` (SyntaxSHAP does not satisfy the Shapley efficiency axiom — see the paper's Appendix B.1) — treat as relative importance/ranking.

Samples the defense did **not** block get `xai_result: null` — no explanation is computed for them.

## Parameters

`DEFAULT_CONFIG`:

- `model_name` (default `"meta-llama/Prompt-Guard-86M"`) — the classifier to explain. Loaded independently from `PromptGuardDefense`'s own pipeline (its own `top_k=None` instance, so every class's score is available).
- `algorithm` (default `"syntax"`) — `"syntax"` (syntax-tree-restricted Shapley) or `"shap"` (exact/unrestricted Shapley — only tractable for very short text, not recommended for whole-`context` explanation). The paper's level-weighted variant (SyntaxSHAP-W) was implemented and then removed — not used by this experiment.
- `explain_targets` (default `["context"]`) — which span(s) to explain. `"target_inst"` and `"injected_task"` are also accepted (explained in isolation via the single-sentence primitive, `explain_row`, without the multi-sentence orchestration).

## Analysis

`scripts/xai_metrics.py` computes Fidelity(t)/acc@1 (adapted from the paper's Eqs. 6 and 8; div@K/Eq. 7 was implemented and then removed — once the explained target is always the binary `[P(BENIGN), P(MALIGN)]` aggregate, div@K collapses algebraically to `2·|Fidelity(t)|`, redundant by construction), the injected-span rank percentile, an ASR/Utility summary over the whole run, and renders plots from a result file:

```bash
python scripts/xai_metrics.py --result results/evaluation_results/xai_pilot/squad_v2-...-syntaxshap-42.json
```

Plots written to `<out-dir>/plots/`:
- `saliency_maps.html` — `shap.plots.text()` per explained sample.
- `fidelity_acc_bars.png` — mean Fidelity(t)/acc@1(t) per threshold, across every explained (blocked) sample.
- `alignment_histogram.png` — distribution of `injected_span_percentile` across the same samples.

The Markdown report ends with an ASR/Utility summary computed over the *entire* raw result (every dataset sample, not just the ones XAI explained) — the benchmark-level numbers the per-sample XAI metrics sit inside, split into overall / blocked / not-blocked.
