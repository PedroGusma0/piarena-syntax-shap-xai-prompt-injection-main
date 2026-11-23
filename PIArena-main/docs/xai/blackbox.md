---
title: Black-box detector XAI
slug: xai-blackbox
category: guide
---

# Black-box XAI (`--xai blackbox`)

Explains a detection defense that is only reachable as a boolean function `text -> detect_flag` — currently **`datasentinel`** (a Mistral-7B + LoRA *known-answer* detector: it is asked to repeat `DGDSGNH` while ignoring the context, and the context is flagged when the key is missing from the generated text). The explained target is the defense's **real decision**, not a surrogate probability. Counterpart of [Kernel SHAP](kernelshap.md)/[Integrated Gradients](ig.md)/[SyntaxSHAP](syntaxshap.md), which need a classifier's logits and are not touched.

```bash
python main.py --dataset squad_v2 --attack direct --defense datasentinel --xai blackbox --limit 3 --name bb_pilot
python scripts/xai_metrics.py --result results/evaluation_results/bb_pilot/<...>-datasentinel-blackbox-42.json
# or: bash scripts/run_blackbox_datasentinel_squad_v2_direct.sh   (LIMIT=3 BASELINE=empty|neutral|clean_context)
```

> Needs a GPU and the DataSentinel checkpoints. Implemented and unit-tested with fake detectors only; not yet run against the real model.

## What it does, per explained sample

1. `full_value = detect(context)`; not flagged → nothing to explain (`skip_reason: not_flagged`).
2. Re-runs the detector `determinism_checks` times and records `deterministic` (4-bit `generate()` can vary).
3. Baseline check: `detect(baseline_text)`. If the baseline is already flagged there is no benign reference: `baseline_valid: false`, the sample is not explained and is left out of the averages (reported separately).
4. **`ddmin`** (delta debugging): a 1-minimal set of sentences, then of words, that still makes the detector fire; each word of the set gets `1/k` (so `sum(values) == 1`).
5. **`sentence_shapley`**: exact Shapley values of the boolean game over sentences when there are at most `max_exact_sentences` (`2^M` detector calls).

Not-kept words are **deleted** (no mask token). The first entry of `methods` fills `values`; the others go to `extra_methods`.

## Config (`xai_config`)

| key | default | meaning |
|---|---|---|
| `methods` | `[ddmin, sentence_shapley]` | ordered; first one fills `values` |
| `baseline` | `empty` | `empty` \| `neutral` (`neutral_text`) \| `clean_context` (context minus the injected span; players = the span's words) |
| `determinism_checks` | `3` | extra detector runs on the full context |
| `max_exact_sentences` | `10` | cap for exact Shapley |
| `max_detector_calls` | `2000` | per-sample budget for the methods (search stops, `truncated: true`) |
| `record_trace` | `false` | keep every ddmin test (kept parts, flag) |
| `explain_scope` | `auto` | blocked samples (`--xai_scope`) |

`blackbox` has no CLI flags beyond `--xai`/`--xai_scope`; the other keys go through a YAML `xai_config` (see the runner script).

## Result fields

`xai_result["context"]`: `tokens`, `values`, `base_value`, `full_value`, `injected_span`, `method`, `granularity`, `explained`, `skip_reason`, `minimal_set`, `baseline`, `baseline_value`, `baseline_valid`, `deterministic`, `n_detector_calls`, `detector_outputs`, `extra_methods` (and `trace`). `p_malign`/`predicted_label` carry the flag.

**Real detector output.** `defense_result.detector_output` is the text DataSentinel generated in the defense's own run, for **every** sample (`detect_flag` is now a bool). `detector_outputs` holds the explanation's re-runs: `full`, `baseline`, `minimal_only` (only the minimal set kept) and `without_minimal` (everything but it), each as `{raw_output, flag}`.

## `scripts/xai_metrics.py` with `blackbox`

Reloads the detector (GPU) to re-score masked variants. Fidelity(t) is 0/1 per sample; `acc@1 = 1 − Fidelity` exactly with a boolean detector, so the report shows Fidelity plus the **deletion rate** (samples no longer blocked after deleting the top-t% words), `minimal_set_precision` (words of the set inside the injected span; replaces `injected_top_mass_precision`, which is biased when all values tie), set size, detector calls per sample and the counts of non-deterministic and invalid-baseline samples. `saliency_maps.html` shows a **"Saída real do detector"** block under each header (defense's original output, re-run, baseline, minimal set only, without the minimal set; model text is HTML-escaped, with a warning when the re-run differs), and a compact block — no map — for samples that were not blocked. Extra plot: `deletion_curve.png`. The confusion matrix and the ASR/Utility summary work as for any defense.

## Adding another boolean defense

Write an adapter (`detect(text) -> Verdict(flag, raw_output)`, heavy imports lazy), and add an entry to `DETECTOR_FOR_DEFENSE` in `piarena/xai/detectors.py`. Nothing in the classifier-based plugins is involved. Design and rationale: `plans/xai-blackbox-datasentinel.md`.
