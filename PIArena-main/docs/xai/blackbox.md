---
title: Black-box detector XAI
slug: xai-blackbox
category: guide
---

# Black-box XAI (`--xai blackbox`)

Explains a detection defense that is only reachable as a boolean function `text -> detect_flag` — currently **`datasentinel`** (a Mistral-7B + LoRA *known-answer* detector: it is asked to repeat `DGDSGNH` while ignoring the context, and the context is flagged when the key is missing from the generated text) — with **Kernel SHAP (`captum.attr.KernelShap`) over the defense's real decision**, every word a player. The values are per word, like the [Kernel SHAP](kernelshap.md) ones for the classifiers, so the saliency maps and metrics are comparable; nothing in the classifier-based plugins is touched.

```bash
python main.py --dataset squad_v2 --attack direct --defense datasentinel --xai blackbox --xai_backend captum --limit 3 --name bb_pilot
python scripts/xai_metrics.py --result results/evaluation_results/bb_pilot/<...>-datasentinel-blackbox-42.json
# or: bash scripts/run_blackbox_datasentinel_squad_v2_direct.sh   (LIMIT=3 BASELINE=empty|neutral|clean_context N_SAMPLES=auto|800)
```

> Needs a GPU, the DataSentinel checkpoints, and `torch` + `captum` + `scikit-learn`. Implemented and unit-tested with fake detectors only (including the real captum solver on synthetic games); not yet run against the real model.

## What it does, per explained sample

1. `full_value = detect(context)`; not flagged → nothing to explain (`skip_reason: not_flagged`).
2. Re-runs the detector `determinism_checks` times and records `deterministic` (4-bit `generate()` can vary).
3. Baseline check: `detect(baseline_text)`. If the baseline is already flagged there is no benign reference: `baseline_valid: false`, the sample is not explained and is left out of the averages (reported separately).
4. Kernel SHAP: `inputs = ones(1, M)`, `baselines = zeros(1, M)`; each coalition captum draws is the text of the kept words (the others are **deleted**, no mask token) and the detector's 0/1 decision is the game's value. Coalition texts are cached.

captum only approximates efficiency, so `sum_values` and `efficiency_error = |Σ values − (full − base)|` are stored, not renormalized.

## Config (`xai_config`)

| key | default | meaning |
|---|---|---|
| `baseline` | `empty` | `empty` \| `neutral` (`neutral_text`) \| `clean_context` (context minus the injected span; the players are then only the span's words) |
| `n_samples` | `auto` | coalitions per sample; `auto` = `clip(16·words, 400, max_detector_calls)` — a starting point to calibrate |
| `batch_size` | `8` | captum `perturbations_per_eval` (every row is still one detector call) |
| `seed` | `0` | seeds captum's coalition sampling |
| `backend` | `captum` | captum only; anything else raises (`--xai_backend captum` sets it) |
| `max_detector_calls` | `3500` | per-sample budget of distinct detector calls; `budget_too_small`/`budget_exceeded` skip the sample |
| `determinism_checks` | `3` | extra detector runs on the full context |
| `explain_scope` | `auto` | blocked samples (`--xai_scope`) |

Only `--xai_backend` and `--xai_scope` have CLI flags; the other keys go through a YAML `xai_config` (see the runner script).

## Cost

One detector call = one DataSentinel generation. The default budget is ≈ 2,100 calls at the median `squad_v2` context (131 words with the injected task) — see `plans/xai-blackbox-kernelshap.md` for the table and a calibration on synthetic boolean games. Measure `n_detector_calls` and `timing.xai_s` on a few samples (`LIMIT=3`) before a larger run.

## Result fields

`xai_result["context"]`: `tokens`, `values`, `base_value`, `full_value`, `injected_span`, `method` (`kernelshap`), `backend`, `granularity` (`word`), `explained`, `skip_reason`, `baseline`, `baseline_value`, `baseline_valid`, `deterministic`, `n_players`, `n_samples`, `seed`, `sum_values`, `efficiency_error`, `n_detector_calls`, `detector_outputs`. `p_malign`/`predicted_label` carry the flag.

**Real detector output.** `defense_result.detector_output` is the text DataSentinel generated in the defense's own run, for **every** sample (`detect_flag` is now a bool). `detector_outputs` holds the explanation's re-runs: `full` and `baseline`, each as `{raw_output, flag}`.

## `scripts/xai_metrics.py` with `blackbox`

Reloads the detector (GPU) to re-score masked variants. Fidelity(t) is 0/1 per sample; `acc@1 = 1 − Fidelity` exactly with a boolean detector, so the report shows Fidelity plus the **deletion rate** (samples no longer blocked after deleting the top-t% words); `injected_top_mass_precision` applies unchanged. It also reports the efficiency error, detector calls per sample and the counts of non-deterministic and invalid-baseline samples. `saliency_maps.html` shows a **"Saída real do detector"** block under each header (the defense's original output, the re-run and the baseline; model text is HTML-escaped, with a warning when the re-run differs), and a compact block — no map — for samples that were not blocked. Extra plot: `deletion_curve.png`. The confusion matrix and the ASR/Utility summary work as for any defense.

## Adding another boolean defense

Write an adapter (`detect(text) -> Verdict(flag, raw_output)`, heavy imports lazy) and add an entry to `DETECTOR_FOR_DEFENSE` in `piarena/xai/detectors.py`. Design and rationale: `plans/xai-blackbox-datasentinel.md`.
