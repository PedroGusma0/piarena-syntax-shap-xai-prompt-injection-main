"""Pilot-analysis metrics for the SyntaxSHAP × PromptGuard XAI experiment.

Pure numeric functions operating on already-computed `xai_result` records
(tokens + per-token SyntaxSHAP values) plus a caller-supplied re-scoring
callback (`rescore_fn`) that re-runs the classifier on a masked variant of
the text — this module doesn't load any model itself; `scripts/xai_metrics.py`
wires it up to a real `ClassifierSyntaxExplainer` instance.

Metric names/formulas follow the SyntaxSHAP paper (papers-de-referencia/
syntaxshap.md, Section 4), reinterpreted for a binary P(malign) classifier
target instead of next-token generation over a huge vocabulary — see
"Métricas" in plans/xai-syntaxshap-promptguard.md for the full rationale of
each reinterpretation (Fidelity(t)/Fid_rand port directly; acc@K only makes
sense at K=1 once the target is binary).

NOTE: div@K (Eq. 7 of the paper) was implemented and then removed — see the
"Por que agregar em binário" section of the plan. It compared the full
per-class probability vector against the masked one via L1 distance, which
was informative against the raw 3-class PromptGuard output (it could detect
probability mass shifting between INJECTION and JAILBREAK even when their
*sum* didn't move, something Fidelity/acc@1 are blind to since they only see
that sum). Once the experiment settled on always explaining and reporting the
binary aggregate [P(BENIGN), P(MALIGN)] instead of the raw 3-class vector,
div@K became mathematically identical to 2·|Fidelity(t)| for every sample and
threshold (both components of a 2-class distribution move by the same
magnitude in opposite directions, so "sum of |deltas|" and "2×|sum of
deltas|" coincide) — not merely correlated, but redundant by construction.
Kept out rather than kept as a relabeled duplicate of Fidelity.

No torch/transformers/spacy imports here on purpose — keeps this module cheap
to import for anything that just wants the numeric functions (e.g. tests).
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
    """Fid(t) = f_ŷ(full text) - f_ŷ(masked-to-top-t% text) — Eq. 6 of the
    paper. `rescore_fn(keep_mask)` masks out everything NOT in keep_mask and
    returns P(malign) for the result. Lower (closer to 0) = more faithful."""
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
    """acc@1 (Eq. 8, K=1 — the only non-degenerate K once the target class is
    binary, K>=2 is trivially 100%): does keeping only the top-t% tokens
    preserve the same above/below-0.5 "malign" decision as the full text?"""
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
    (legitimate) text?" (métrica 5 do plano — the classification-adapted,
    whole-context version of the paper's Semantic Alignment, Section 5.4)."""
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



# NOTE: a `spearman()` helper used to live here — Spearman rank correlation
# between two equal-length token-importance vectors, computing métrica 7
# ("concordância `syntax` vs `syntax-w`": correlate the per-token ranking
# produced by the plain SyntaxSHAP algorithm against the level-weighted
# variant, per sample). Removed along with `algorithm="syntax-w"` itself
# (piarena/xai/syntaxshap/coalition.py / classifier_explainer.py) — the
# PromptGuard experiment always explains with plain `"syntax"`, so there is
# no second variant left to compare against. `scripts/xai_metrics.py`'s
# `--result-w` flag and the `syntax_vs_syntax_w_spearman` fields were removed
# with it.
