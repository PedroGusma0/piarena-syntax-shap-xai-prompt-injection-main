# Plan: `--xai blackbox` — explaining DataSentinel's real `detect_flag`

Status: implemented; verified only with fake detectors on CPU (no GPU / real DataSentinel here). Experiment write-up (Portuguese): `../markdowns-do-experimento/datasentinel-blackbox-detect-flag.md`. User-facing doc: [`docs/xai/blackbox.md`](../docs/xai/blackbox.md).

## Why

The existing XAI methods (`syntaxshap`, `kernelshap`, `ig`) explain an `AutoModelForSequenceClassification` through `P(malign)`. DataSentinel is a Mistral-7B + LoRA (4-bit) *known-answer* detector: `DataSentinelDetector.detect` generates up to 10 tokens and returns 1 (injection) when `"DGDSGNH"` is **not** in the output. There is no classification head and no probability to explain.

Decision: the explained target is the **real binary decision** (`detect_flag`), not a teacher-forcing surrogate (kept as a deferred alternative). That makes the detector a boolean black box `text -> flag`, so the method only needs that interface.

## Compatibility rules

- Nothing in `piarena/xai/classifiers.py`, `kernelshap/`, `ig/`, `syntaxshap/` or the PromptGuard/PIGuard defenses changes.
- `BaseXAI.resolve_config(defense_name, config)` (new classmethod) is the identity by default; `main.py` keeps the `kernelshap`/`ig` → `resolve_classifier` branch and calls it for every other plugin.
- `scripts/xai_metrics.py` gained `blackbox`-only branches; on a `kernelshap`-format result its per-sample metrics, aggregate and report are byte-identical to the previous version (checked).
- `import piarena.xai` stays free of torch/transformers/peft/spacy/captum.
- The only defense-side change is DataSentinel's: `detect_verbose` + `detector_output` + `detect_flag` as `bool`.

## Design

```
piarena/xai/detectors.py              Verdict, BooleanDetector (Protocol), DETECTOR_FOR_DEFENSE, resolve_detector, load_detector
piarena/xai/blackbox/
  perturb.py                          whitespace words + heuristic sentences; build(text from kept words)
  ddmin.py                            delta debugging, injectable oracle, call budget
  sentence_shapley.py                 exact Shapley over sentences
  explainer.py                        BlackBoxExplainer.explain_context / make_rescorer
  xai_blackbox.py                     BlackBoxXAI (registered as "blackbox")
  adapters/datasentinel.py            DataSentinelAdapter -> DataSentinelDetector.detect_verbose
```

Per sample: `full_value = detect(context)` (0 → `skip_reason=not_flagged`); determinism re-runs; baseline check `detect(baseline_text)` (1 → `baseline_valid=False`, not explained, excluded from averages); then `ddmin` (sentences, then words) and, for few sentences, exact `sentence_shapley`. `values` satisfy efficiency (`sum == full - base == 1`): `1/k` per word of the minimal set; the sentence value split over its words.

Baselines: `empty`, `neutral`, `clean_context` (context minus the injected span; players are then only the span's words).

## Output

`xai_result["context"]`: `tokens`, `values`, `base_value`, `full_value`, `injected_span`, `p_malign`/`predicted_label` (flag), plus `method`, `granularity`, `explained`, `skip_reason`, `minimal_set`, `baseline*`, `deterministic`, `n_detector_calls`, `detector_outputs` (`full`, `baseline`, `minimal_only`, `without_minimal`: `raw_output` + `flag`), `extra_methods`, optional `trace`. `defense_result.detector_output` holds the text the detector generated in the defense's own run, for every sample.

## `scripts/xai_metrics.py`

Needs the GPU (reloads the detector to re-score). With a 0/1 re-scorer Fidelity(t) is 0/1 per sample and `acc@1 = 1 − Fidelity` exactly, so the report shows Fidelity plus a **deletion** rate (samples no longer blocked after deleting the top-t% words), `minimal_set_precision` (replaces `injected_top_mass_precision` for ddmin, whose equal values make the mass metric biased), set size, calls per sample, non-deterministic and invalid-baseline counts, and a table of the real detector outputs. `saliency_maps.html` gets a "Saída real do detector" block per sample (HTML-escaped) and a compact block for samples without a map. New plot: `deletion_curve.png`.

## Risks

Degenerate baseline (empty text already flagged); non-deterministic 4-bit generation; non-monotone detector (ddmin is only 1-minimal w.r.t. the tests run; several disjoint triggers possible); cost of a 7B generation per test (`max_detector_calls`, default 2000); no windows (the detector sees the whole context); heuristic sentence splitting.

## Verification

`python -m unittest tests.test_xai_blackbox -v` (from `PIArena-main/`; fake detectors, stubbed `piarena.utils`). On a GPU: 3–5 `squad_v2`/`direct` samples with `scripts/run_blackbox_datasentinel_squad_v2_direct.sh` (`LIMIT=3`): check `detect("")`, determinism, `baseline_valid`, that the re-run output equals `defense_result.detector_output`, and the time per sample before any larger run.

## Deferred

Teacher-forcing targets (`1 − P("DGDSGNH")`, or only the first key token) for Kernel SHAP/IG; Kernel SHAP per word over the boolean game; reusing the defense's loaded detector instead of a second 4-bit copy; prefix caching/batching of detector calls.
