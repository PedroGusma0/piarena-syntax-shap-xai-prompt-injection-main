# Plan: `--xai blackbox` — Kernel SHAP over DataSentinel's real `detect_flag`

Status: implemented; verified only with fake detectors on CPU (including the real captum solver, in a torch+captum venv) — no GPU / real DataSentinel here. Experiment write-up (Portuguese): `../markdowns-do-experimento/datasentinel-blackbox-detect-flag.md`. User-facing doc: [`docs/xai/blackbox.md`](../docs/xai/blackbox.md). Cost and calibration: [`xai-blackbox-kernelshap.md`](xai-blackbox-kernelshap.md).

## Why

The existing XAI methods (`syntaxshap`, `kernelshap`, `ig`) explain an `AutoModelForSequenceClassification` through `P(malign)`. DataSentinel is a Mistral-7B + LoRA (4-bit) *known-answer* detector: `DataSentinelDetector.detect` generates up to 10 tokens and returns 1 (injection) when `"DGDSGNH"` is **not** in the output. There is no classification head and no probability to explain.

Decision: the explained target is the **real binary decision** (`detect_flag`), not a teacher-forcing surrogate (kept as a deferred alternative), and the method is **Kernel SHAP (captum) with every word a player** — the same family as the PromptGuard/PIGuard experiments, so the saliency maps and metrics are comparable. The detector is treated as a boolean black box `text -> flag`.

An earlier iteration of this plan also had delta debugging (`ddmin`) and exact Shapley over sentences; they were removed in favour of Kernel SHAP alone.

## Compatibility rules

- Nothing in `piarena/xai/classifiers.py`, `kernelshap/`, `ig/`, `syntaxshap/` or the PromptGuard/PIGuard defenses changes.
- `BaseXAI.resolve_config(defense_name, config)` (new classmethod) is the identity by default; `main.py` keeps the `kernelshap`/`ig` → `resolve_classifier` branch and calls it for every other plugin.
- `scripts/xai_metrics.py` gained `blackbox`-only branches; on a `kernelshap`-format result its per-sample metrics, aggregate and report are identical to the previous version (checked).
- `import piarena.xai` stays free of torch/transformers/peft/spacy/captum (checked in a fresh interpreter).
- The only defense-side change is DataSentinel's: `detect_verbose` + `detector_output` + `detect_flag` as `bool`.

## Design

```
piarena/xai/detectors.py              Verdict, BooleanDetector (Protocol), DETECTOR_FOR_DEFENSE, resolve_detector, load_detector
piarena/xai/blackbox/
  perturb.py                          whitespace words + offsets; build(text from kept words)
  kernelshap.py                       captum KernelShap over 0/1 coalition rows (lazy torch/captum)
  explainer.py                        BlackBoxExplainer.explain_context / make_rescorer
  xai_blackbox.py                     BlackBoxXAI (registered as "blackbox")
  adapters/datasentinel.py            DataSentinelAdapter -> DataSentinelDetector.detect_verbose
```

Per sample: `full_value = detect(context)` (0 → `skip_reason=not_flagged`); determinism re-runs; baseline check `detect(baseline_text)` (1 → `baseline_valid=False`, not explained, excluded from averages); then Kernel SHAP: `inputs = ones(1, M)`, `baselines = zeros(1, M)`, one feature per word; each 0/1 row captum produces becomes the text of the kept words (the others deleted, no mask token) and the forward function returns the detector's decision. Coalition texts are cached. captum's efficiency is only approximate, so `sum_values` and `efficiency_error` are recorded, not renormalized.

Baselines: `empty`, `neutral`, `clean_context` (context minus the injected span; the players are then only the span's words, everything else stays fixed).

Config (`xai_config`): `baseline`, `neutral_text`, `determinism_checks`, `backend` (`captum` only — anything else raises), `n_samples` (`auto` = `clip(16·M, 400, max_detector_calls)`), `batch_size` (captum `perturbations_per_eval`), `seed`, `max_detector_calls` (3500 distinct calls per sample; `budget_too_small`/`budget_exceeded` skip the sample instead of failing it).

## Output

`xai_result["context"]`: `tokens`, `values` (per word), `base_value`, `full_value`, `injected_span`, `p_malign`/`predicted_label` (the flag), `method: "kernelshap"`, `backend`, `granularity: "word"`, `explained`, `skip_reason`, `baseline*`, `deterministic`, `n_players`, `n_samples`, `seed`, `sum_values`, `efficiency_error`, `n_detector_calls`, `detector_outputs` (`full`, `baseline`: `raw_output` + `flag`). `defense_result.detector_output` holds the text the detector generated in the defense's own run, for every sample.

## `scripts/xai_metrics.py`

Needs the GPU (reloads the detector to re-score). With a 0/1 re-scorer Fidelity(t) is 0/1 per sample and `acc@1 = 1 − Fidelity` exactly, so the report shows Fidelity plus a **deletion** rate (samples no longer blocked after deleting the top-t% words); `injected_top_mass_precision` applies as for the classifiers (the values are continuous). Also reported: Kernel SHAP efficiency error, detector calls per sample, non-deterministic and invalid-baseline counts, and a table of the real detector outputs. `saliency_maps.html` gets a "Saída real do detector" block per sample (HTML-escaped) and a compact block for samples without a map. New plot: `deletion_curve.png`.

## Risks

Boolean outputs make the weighted regression noisy (see the calibration in `xai-blackbox-kernelshap.md`); degenerate baseline (empty text already flagged); non-deterministic 4-bit generation; cost of a 7B generation per coalition; no windows (the detector sees the whole context).

## Verification

`python -m unittest tests.test_xai_blackbox -v` (from `PIArena-main/`; fake detectors, stubbed `piarena.utils`; the captum test is skipped without torch+captum). On a GPU: 3–5 `squad_v2`/`direct` samples with `scripts/run_blackbox_datasentinel_squad_v2_direct.sh` (`LIMIT=3`): check `detect("")`, determinism, `baseline_valid`, that the re-run output equals `defense_result.detector_output`, and the time per sample before any larger run.

## Deferred

Teacher-forcing targets (`1 − P("DGDSGNH")`, or only the first key token); reusing the defense's loaded detector instead of a second 4-bit copy; prefix caching/batching of detector calls.
