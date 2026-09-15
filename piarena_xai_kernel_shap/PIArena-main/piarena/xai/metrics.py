"""Pilot-analysis metrics for the Kernel SHAP x PromptGuard XAI experiment.

Pure numeric functions operating on already-computed `xai_result` records
(tokens + per-token Kernel SHAP values) plus a caller-supplied re-scoring
callback (`rescore_fn`) that re-runs the classifier on a masked variant of
the text — this module doesn't load any model itself; `scripts/xai_metrics.py`
wires it up to a real `ClassifierKernelExplainer` instance.

Algorithm-agnostic by design: it only assumes the `xai_result["context"]`
shape documented in `docs/xai/kernelshap.md` (tokens/values/base_value/
full_value/p_malign/predicted_label/injected_span) — the same contract
used by the parallel SyntaxSHAP experiment (`piarena_xai_syntax_shap/`), so
this file is intentionally kept identical across both experiments rather than
re-derived. See `plans/xai-kernelshap-promptguard.md` ("Métricas") for the
full rationale of each metric (Fidelity(t)/acc@1 port the SyntaxSHAP paper's
Eqs. 6/8 reinterpreted for a binary P(malign) classifier target; div@K
(Eq. 7) is not implemented — it collapses algebraically to `2·|Fidelity(t)|`
once the explained target is always the binary [P(BENIGN), P(MALIGN)]
aggregate, redundant by construction).

No torch/transformers imports here on purpose — keeps this module cheap to
import for anything that just wants the numeric functions (e.g. tests).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def top_t_mask(values: Sequence[float], t: float) -> np.ndarray:
    """Boolean mask, True = keep this token, for the top-t fraction of tokens
    by |value| (t in (0, 1]). Ties broken by original order (stable sort)."""
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k = max(1, round(n * t))
    order = np.argsort(-np.abs(np.asarray(values, dtype=float)), kind="stable")  # descending |value|
    keep = np.zeros(n, dtype=bool)
    keep[order[:k]] = True
    return keep


def fidelity(
    full_value: float,
    values: Sequence[float],
    rescore_fn: Callable[[np.ndarray], float],
    thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
) -> dict:
    """Fid(t) = f_ŷ(full text) - f_ŷ(masked-to-top-t% text). `rescore_fn(keep_mask)`
    masks out everything NOT in keep_mask and returns P(malign) for the
    result. Lower (closer to 0) = more faithful."""
    out = {}
    for t in thresholds:
        keep_mask = top_t_mask(values, t)
        out[str(t)] = full_value - rescore_fn(keep_mask)
    return out


def acc_at_1(
    full_p_malign: float,
    values: Sequence[float],
    rescore_fn: Callable[[np.ndarray], float],
    thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
) -> dict:
    """acc@1 (K=1 — the only non-degenerate K once the target class is binary,
    K>=2 is trivially 100%): does keeping only the top-t% tokens preserve the
    same above/below-0.5 "malign" decision as the full text?"""
    full_decision = full_p_malign >= 0.5
    out = {}
    for t in thresholds:
        keep_mask = top_t_mask(values, t)
        masked_decision = rescore_fn(keep_mask) >= 0.5
        out[str(t)] = bool(masked_decision == full_decision)
    return out


def injected_span_percentile(values: Sequence[float], injected_span: Optional[dict]) -> Optional[float]:
    """Mean percentile rank (0-1, higher = more important) of the tokens
    inside `injected_span` among *all* tokens of the explained text — "does
    the explanation find the truly-injected content within the rest of the
    (legitimate) text?"."""
    if not injected_span or len(values) == 0:
        return None
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.abs(values))  # ascending |value|
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values))
    denom = max(1, len(values) - 1)
    percentiles = ranks / denom
    start, end = injected_span["start_token"], injected_span["end_token"]
    span_percentiles = percentiles[start:end]
    if len(span_percentiles) == 0:
        return None
    return float(np.mean(span_percentiles))
