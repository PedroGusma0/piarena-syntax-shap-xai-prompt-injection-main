# Plan: Kernel SHAP (captum) as the `blackbox` method — cost and calibration

Status: **implemented** (`piarena/xai/blackbox/kernelshap.py`, `explainer.py`); see [`xai-blackbox-datasentinel.md`](xai-blackbox-datasentinel.md) for the overall design. Players: **every word** (decision of the user); delta debugging and exact Shapley over sentences were dropped. Cost figures come from the real `squad_v2` contexts; per-call latency is an **assumption** until the first GPU smoke test measures it. The calibration below used fake detectors (synthetic boolean games), not DataSentinel.

## Formulation

`captum.attr.KernelShap` with `inputs = ones(1, M)`, `baselines = zeros(1, M)`, no feature mask (one feature per word). The forward function turns each 0/1 row into the text of the kept words and returns the detector's 0/1 decision. captum only (project rule: never the `shap` backend); `backend` in the config accepts only `captum`. Needs `torch`, `captum` and `scikit-learn` (captum's KernelShap fits its surrogate with sklearn) — imported lazily.

## Cost (real `squad_v2` contexts + the injected task, 200 samples)

| | min | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| words `M` | 56 | 113 | 131 | 148 | 185 | 385 |
| injected-task words | 16 | 23 | 25 | 29 | 34 | 39 |

`n_samples = auto = clip(16·M, 400, 3500)`: ≈ 2,100 detector calls at the median (900–3,500 overall); fewer distinct calls when coalitions repeat (cache). Preconditions add `determinism_checks + 2` calls.

Time per sample = calls × `t_call` (one DataSentinel generation: prefill of ~150–250 tokens + 10 decode steps, 4-bit 7B). **Assumption, to be measured:** 0.3–1.0 s ⇒ roughly 10–35 min per explained sample at the median. Multiply by the number of *blocked* samples (the XAI runs only on those): the whole dataset is a multi-day run unless `t_call` is at the low end, so start with `LIMIT=3` and a curated subset. The smoke test's `n_detector_calls` and `timing.xai_s` give `t_call` directly.

## Calibration (CPU, fake detectors)

captum's KernelShap against games with a known answer:

- **M = 131, game = AND(w40, w41)** (exact Shapley 0.5 for the two triggers, 0 for the other 129), mean over 5 seeds:

| `n_samples` | error on the triggers | max noise on the other words | top-2 = the triggers |
|---:|---:|---:|---|
| 400 | 0.107 | 0.167 | 5/5 |
| 1,000 | 0.036 | 0.101 | 5/5 |
| 2,096 (= 16·M, the `auto` value) | 0.033 | 0.066 | 5/5 |
| 4,000 | 0.016 | 0.043 | 5/5 |

  `Σ values` came out at 1.00 in every run.
- **M = 14 against exact enumeration** (AND, OR, `0 AND (1 OR 2)`, "at least 3 of 6"): max |error| ≈ 0.14–0.18 at 100 samples, 0.06–0.08 at 400, 0.03–0.04 at 1,600 (≈ 1/√n); efficiency error 0.00. Top-1 hit rate is meaningless on games with ties (symmetric players).

Reading: for a sparse trigger the `auto` budget separates the triggers cleanly (error ≈ 7% of the trigger value, noise ≈ 13% of it at worst). It says nothing about the *real* DataSentinel game, which may be denser, non-monotone and non-deterministic — hence the validation below. `16·M` is a starting point, not a result.

## Validation on the GPU machine (not done)

1. Ranking stability: two seeds on the same sample, rank correlation of the per-word values (`markdowns-do-experimento/propriedades-explicacoes-e-metrica-estabilidade.md`).
2. Convergence: the same sample at `n_samples` ∈ {400, 1000, 2000, 4000}; if the top words keep changing, raise `n_samples` (or restrict the game with `baseline: clean_context`).
3. Efficiency error and the deletion rate from `xai_metrics.py`.
4. `deterministic` and `baseline_valid` counts before trusting any value.

## Optional follow-up (separate step)

Batched detector calls. `DataSentinelDefense.execute_batch` already looks for a `batch_detect` hook; `DataSentinelDetector` has none and `QLoraModel.query` handles one prompt. Left-padded batched `generate` could cut `t_call` several-fold, but padding and 4-bit numerics can change the generated text, i.e. **the decision itself**; only worth doing with a check that batched and single-call outputs are identical on a few hundred texts.

## Risks

Boolean outputs make the weighted regression noisy; `clean_context` is a conditional game (players = the injected span only) and must not be compared with the full-context values; 4-bit non-determinism corrupts the game (`determinism_checks` flags the sample but does not cover every coalition); the cost estimate rests on an unmeasured `t_call`.
