"""Syntax-aware Shapley coalition game — reimplemented core.

This module reimplements the pure numpy/pandas coalition-building and
Shapley-value logic from the original `SyntaxExplainer.compute_shapley_values`
(syntax-shap-main/syntaxshap/explainers/_syntax.py), adapted for classification
models instead of autoregressive text generation.

Why reimplemented instead of imported/subclassed from the vendored code:
see "Correções de arquitetura" in plans/xai-syntaxshap-promptguard.md.
In short: the original `compute_shapley_values`/`get_contribution` are wired
to `captum.attr.LLMAttribution` + `.generate()` (autoregressive-only), and
`feature_exact()` has a critical scaling bug that matters a lot once you
explain a whole (multi-sentence) `context` instead of a single short sentence:

    Bug (present in the original `feature_exact`, both `asymmetric=False` AND
    `asymmetric=True`): it always builds `pd.DataFrame({'id_combination':
    range(2**M)})` plus the FULL powerset of `range(M)`
    (`chain(*[combinations(range(M), i) for i in range(M+1)])`) FIRST, and only
    *afterwards* filters down to the syntax-tree-respecting subset via
    `respects_order`. That means even the tree-restricted ("syntax"/"syntax-w")
    path pays an unconditional O(2^M) enumeration cost before any filtering —
    fine for the paper's tested M ~ 15-20 tokens, but literally impossible for
    M ~ 100-180 tokens (a whole squad_v2 `context`; `range(2**100)` never
    finishes).

    Fix here: `build_allowed_coalitions` constructs the tree-restricted
    coalition set 𝔖 = ⋃_l 𝔖_l *directly*, level by level (matching Eq. 4 of
    the paper: total coalitions ≈ Σ_l 2^(n_l), never touching 2^M). The
    unrestricted/exact powerset path (used only by `algorithm == "shap"`) is
    kept as-is since that mode is inherently O(2^M) by definition — but it is
    no longer computed *unconditionally* for every algorithm the way the
    original `dt_exact = feature_exact(M)` line did.

NOTE: the original paper also defines a level-weighted variant ("SyntaxSHAP-W",
Section 3.4, `algorithm="syntax-w"` — each coalition's marginal contribution
scaled by `1/level`). It was implemented here and then removed: the
PromptGuard experiment always uses plain (unweighted) `"syntax"`, so carrying
a second, unused algorithm mode/parameter (and the `spearman()`-based
concordance metric that only existed to compare the two) was needless surface
area. If a weighted variant is needed again, reintroduce the `weighted` param
below plus the `weight` multiplier in the marginal-contribution loop — the
`level_weight` column `get_token_dependency_tree` already produces was never
removed from `_dependency_tree.py`.

The outer per-coalition marginal-contribution loop in `compute_shapley_values`
mirrors the original algorithm's behavior exactly (same accumulation logic),
just parameterized by a `get_contribution_fn(mask) -> float` callback instead
of a captum-backed method, and operating on scalars throughout (no more
`eval_diff[0, 0].item()` tensor unpacking, since our scoring function already
returns a plain float).
"""
from __future__ import annotations

import math
import time
from itertools import chain, combinations

import numpy as np
import pandas as pd


class XAITimeoutError(RuntimeError):
    """Raised by `compute_shapley_values` when a `deadline` (an absolute
    `time.time()` timestamp, not a relative duration) is reached before every
    coalition's marginal contribution has been evaluated.

    Carries whatever was computed so far, normalized the same way a
    completed call would be (`partial_values`), plus `calls_done`/
    `total_calls` for diagnostics — a timeout is not meant to throw away
    partial work. See `ClassifierSyntaxExplainer.explain_context`'s
    per-sentence timeout handling, which catches this and folds the partial
    result into the overall `context` explanation instead of propagating.
    """

    def __init__(self, partial_values: np.ndarray, calls_done: int, total_calls: int):
        self.partial_values = partial_values
        self.calls_done = calls_done
        self.total_calls = total_calls
        super().__init__(
            f"compute_shapley_values timed out after {calls_done}/{total_calls} forward passes"
        )


class _CoalitionBuildTimeout(Exception):
    """Internal signal, never raised past this module: `deadline` was reached
    while still *enumerating the coalition set itself*
    (`build_allowed_coalitions`/`feature_exact`'s own `O(2^n_l)`/`O(2^M)`
    construction), before a single forward pass could even be scheduled.

    This matters because that construction happens once, upfront, in
    `compute_shapley_values` (`dt = feature_exact(...)`) — *before* its own
    per-coalition loop (where `deadline` is otherwise checked every forward
    pass) ever starts. Without this, a single pathologically wide dependency-
    tree level (Finding 1/2 in CLAUDE.md's cost analysis — e.g. `n_l = 38`)
    would hang inside `feature_exact` materializing ~2^38 rows, never
    reaching the loop where the per-call deadline check lives, no matter how
    short `deadline` is. `compute_shapley_values` catches this internally and
    re-raises the real, public `XAITimeoutError` (with an all-zero
    `partial_values` — nothing was scored yet)."""


def _normalize(dvalues: np.ndarray, count_updates: np.ndarray) -> np.ndarray:
    """Shared by the normal return path and `XAITimeoutError.partial_values`
    — same math either way (see `compute_shapley_values`'s Returns section)."""
    count_updates_safe = np.where(count_updates == 0, 1, count_updates)  # avoid div-by-zero for unreached tokens
    values = dvalues / count_updates_safe
    total = np.sum(values)
    if total != 0:
        values = values / total
    return values


def convert_feat_to_mask(feature, m):
    """Binary mask (length m) with `feature` (a list of indices) set to True."""
    mask = np.zeros(m)
    mask[feature] = 1
    return np.array(mask, dtype=bool)


def respects_order(index, causal_ordering):
    """True iff `index` (a list of feature positions) only contains a feature
    from level l when every feature from every level < l is also in `index`.

    Kept from the original implementation verbatim (ported, not reimplemented)
    — used here only as a reference/testing oracle for `build_allowed_coalitions`
    on small inputs, not in the hot path (see module docstring for why the hot
    path no longer enumerates-then-filters).
    """
    for i in index:
        idx_position = next((pos for pos, sublist in enumerate(causal_ordering) if i in sublist), -1)
        if idx_position == -1:
            raise ValueError("Element not found in causal_ordering")
        if idx_position > 0:
            precedents = [item for sublist in causal_ordering[:idx_position] for item in sublist]
            if not set(precedents).issubset(set(index)):
                return False
    return True


_DEADLINE_CHECK_EVERY = 4096  # batches the time.time() calls during coalition
                               # *construction* (as opposed to the one-per-
                               # forward-pass check in compute_shapley_values'
                               # own loop) — a single deadline check is cheap,
                               # but a pathological level can imply billions of
                               # appends before a natural end, so checking on
                               # literally every one would still add up. This
                               # bounds the overshoot past `deadline` to at
                               # most ~4096 cheap appends, negligible next to
                               # even one real forward pass.


def build_allowed_coalitions(
    causal_ordering: list[list[int]], deadline: float | None = None,
) -> list[list[int]]:
    """Directly construct the syntax-tree-restricted coalition set.

    𝔖 = ⋃_l 𝔖_l where 𝔖_l = { X_<l ∪ σ : σ ∈ 𝒫(X_l) } (paper, Section 3.3) —
    for each level l, every coalition is "all positions from earlier levels"
    unioned with "some subset of this level's own positions". Includes the
    null coalition S_0 = [] (hypothetical level 0).

    Cost: Σ_l 2^(n_l) combinations total, never the full 2^M powerset — this
    is the actual efficient algorithm the paper claims (Eq. 4 / Appendix B.2),
    as opposed to the original `feature_exact(asymmetric=True)` which built
    2^M rows and filtered them down to the same set.

    `deadline` (absolute `time.time()` timestamp, `None` = no check): a
    single pathologically wide level (`n_l` large — Finding 1/2 in CLAUDE.md's
    cost analysis) can make *this construction itself* the actual multi-hour/
    multi-day cost, independent of and before `compute_shapley_values`'s own
    per-forward-pass loop ever runs. Raises `_CoalitionBuildTimeout` (an
    internal signal, not `XAITimeoutError` — this function doesn't know `M`
    for the all-zero `partial_values` shape; its caller does) if exceeded.
    """
    coalitions: list[list[int]] = [[]]
    prefix: list[int] = []
    since_check = 0
    for level_positions in causal_ordering:
        level_positions = list(level_positions)
        # r starts at 1, not 0: the r=0 (empty-subset-of-this-level) case would
        # just reproduce `prefix` as it stood *before* this level — which is
        # either the initial null coalition (first level) or already exactly
        # equal to the previous level's own r=full-subset entry (since
        # `prefix` gets extended by the *entire* previous level right after
        # its loop). Skipping it keeps every coalition a unique set, matching
        # the original `feature_exact`'s behavior (each combination from
        # `itertools.combinations` is inherently unique) instead of emitting
        # literal duplicate rows.
        for r in range(1, len(level_positions) + 1):
            for combo in combinations(level_positions, r):
                if deadline is not None:
                    since_check += 1
                    if since_check >= _DEADLINE_CHECK_EVERY:
                        since_check = 0
                        if time.time() > deadline:
                            raise _CoalitionBuildTimeout()
                coalitions.append(prefix + list(combo))
        prefix = prefix + level_positions
    return coalitions


def feature_exact(
    M: int, asymmetric: bool = False, causal_ordering: list[list[int]] | None = None,
    deadline: float | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with one row per allowed coalition (`features`: list
    of feature indices in that coalition).

    - `asymmetric=False` (used only for `algorithm == "shap"`): exact/unrestricted
      Shapley — the full powerset of `range(M)`, i.e. O(2^M) rows. Only
      tractable for small M (a single short sentence, matching the paper's
      tested range) — this mode is not used for whole-`context` explanation.
    - `asymmetric=True` (used for `algorithm == "syntax"`): the tree-restricted
      set, built via `build_allowed_coalitions` — tractable for much larger M
      since it never enumerates 2^M.

    `deadline`: see `build_allowed_coalitions` — same meaning, also applied to
    the unrestricted-powerset branch below (`asymmetric=False`), which is
    just as unconditionally exponential.
    """
    if asymmetric:
        if causal_ordering is None:
            causal_ordering = [list(range(M))]
        combos = build_allowed_coalitions(causal_ordering, deadline=deadline)
    else:
        combos = []
        since_check = 0
        for combo in chain(*[combinations(range(M), i) for i in range(M + 1)]):
            if deadline is not None:
                since_check += 1
                if since_check >= _DEADLINE_CHECK_EVERY:
                    since_check = 0
                    if time.time() > deadline:
                        raise _CoalitionBuildTimeout()
            combos.append(list(combo))

    dt = pd.DataFrame({'features': combos})
    dt['n_features'] = dt['features'].apply(len)
    dt['mask'] = dt['features'].apply(lambda x: convert_feat_to_mask(x, M))
    return dt


def compute_shapley_values(
    M: int,
    get_contribution_fn,
    algorithm: str = "syntax",
    dependency_dt: pd.DataFrame | None = None,
    progress: bool = False,
    progress_desc: str | None = None,
    deadline: float | None = None,
    stats: dict | None = None,
) -> np.ndarray:
    """Compute per-token SyntaxSHAP values.

    Parameters
    ----------
    M : number of (content) tokens being explained.
    get_contribution_fn : callable, `mask: np.ndarray[bool] (len M) -> float`.
        Scores a masked variant of the input under the model. For the
        classifier adaptation this is `ClassifierSyntaxExplainer._score`
        (a plain forward pass — no captum, no teacher forcing).
    algorithm : "syntax" or "shap".
    dependency_dt : DataFrame from `get_token_dependency_tree` (columns used:
        `level`, `token_position`). Required for "syntax".
    progress : if True, print the exact forward-pass budget for this call
        before running (coalitions × ~1 call each, known upfront from `dt`
        alone — no need to run anything first) and drive a tqdm bar off the
        actual `get_contribution_fn` calls as they happen. This is the only
        place per-call progress is observable at all — the caller (a whole
        `context`, possibly hours long per the cost analysis in the plan) is
        otherwise silent for the entire duration of one sentence/row.
    progress_desc : short label prefixed to the bar, e.g. "sentence 2/6" —
        lets the caller (`ClassifierSyntaxExplainer.explain_context`) show
        *which* sub-unit is running, since cost varies wildly between them
        (Finding 1: driven by each sentence's widest dependency-tree level,
        not by its length).
    deadline : an absolute `time.time()` timestamp (not a relative duration —
        the caller may share one deadline across several `compute_shapley_values`
        calls, e.g. one per sentence of a multi-sentence `context`, so the
        budget has to be a fixed point in time, not "N seconds from now" reset
        on every call). `None` (default) means no timeout, exact original
        behavior. Checked before every single forward pass — negligible
        overhead next to a forward pass — and raises `XAITimeoutError` the
        moment it's exceeded, carrying whatever was computed so far.

    Returns
    -------
    np.ndarray of shape (M,) — one Shapley value per token, normalized to
    sum to 1 in absolute terms (matches the original implementation's final
    `self.values / np.sum(self.values)` step). Note (documented, not a bug):
    SyntaxSHAP does not satisfy the Shapley efficiency axiom (paper, Appendix
    B.1) — these values are relative importances, not an exact additive
    decomposition of the model's output change.

    stats : optional dict, filled IN PLACE (not returned — `values` stays a
        plain `np.ndarray`, so every existing caller/test keeps working
        unchanged) with cost telemetry, for callers that want to log/inspect
        it (e.g. W&B) without re-deriving it: `"widest_level"` (max size of
        any single dependency-tree level — the actual cost driver, Finding 1
        of the cost analysis; only set for `algorithm == "syntax"`),
        `"n_coalitions"`/`"total_calls"` (known upfront, before a single
        forward pass — set even if a timeout happens later), and
        `"calls_done"` (only set on successful completion; on timeout, use
        `XAITimeoutError.calls_done`/`.total_calls` instead — same numbers,
        already carried by the exception).

    Raises
    ------
    XAITimeoutError : if `deadline` is set and reached before completion.
    """
    count_updates = np.zeros(M, dtype=int)
    dvalues = np.zeros(M, dtype=float)

    if algorithm == "syntax":
        if dependency_dt is None:
            raise ValueError(f"algorithm={algorithm!r} requires dependency_dt (from get_token_dependency_tree)")
        causal_ordering = []
        unique_levels = sorted(dependency_dt['level'].unique())
        for level in unique_levels:
            positions = dependency_dt.loc[dependency_dt['level'] == level, 'token_position'].tolist()
            causal_ordering.append(positions)
        if stats is not None:
            # The actual cost driver (2^widest_level, Finding 1 of the cost
            # analysis) — set even if the coalition build or the loop below
            # times out, since it's known from `causal_ordering` alone.
            stats["widest_level"] = max((len(lvl) for lvl in causal_ordering), default=0)
        try:
            dt = feature_exact(M, asymmetric=True, causal_ordering=causal_ordering, deadline=deadline)
        except _CoalitionBuildTimeout:
            # Deadline hit while still enumerating coalitions (e.g. a single
            # pathologically wide dependency-tree level) — before this call's
            # own per-forward-pass loop below ever got a chance to run, let
            # alone check `deadline` itself. Nothing was scored yet.
            raise XAITimeoutError(_normalize(dvalues, count_updates), 0, 0)
    elif algorithm == "shap":
        try:
            dt = feature_exact(M, deadline=deadline)
        except _CoalitionBuildTimeout:
            raise XAITimeoutError(_normalize(dvalues, count_updates), 0, 0)
    else:
        raise ValueError("algorithm must be one of 'syntax', 'shap'")

    dt = dt.reset_index(drop=True)

    # Exact forward-pass budget for this call: for coalition i (of len(dt)-1
    # actually evaluated — the last row is never used as a base coalition),
    # `get_contribution_fn` runs once for the coalition itself (eval_00) plus
    # once per remaining, not-yet-included token (eval_10). Cheap to sum from
    # `dt['n_features']` alone (dt is already the tree-restricted/small set,
    # never the 2^M powerset — see module docstring), so this is knowable
    # *before* issuing a single real forward pass.
    n_coalitions = max(len(dt) - 1, 0)
    total_calls = int(n_coalitions + (M - dt['n_features'].iloc[:n_coalitions]).sum()) if n_coalitions else 0
    if stats is not None:
        # Known before a single forward pass — set even if the loop below
        # times out partway through.
        stats["n_coalitions"] = n_coalitions
        stats["total_calls"] = total_calls

    pbar = None
    if progress:
        label = f" ({progress_desc})" if progress_desc else ""
        print(f"[syntaxshap] M={M} tokens, {n_coalitions} coalitions, "
              f"{total_calls} forward passes to run{label}.")
        if total_calls:
            from tqdm import tqdm
            pbar = tqdm(total=total_calls, desc=f"syntaxshap{label}", leave=False)

    calls_done = 0
    try:
        for i in range(len(dt) - 1):
            if deadline is not None and time.time() > deadline:
                raise XAITimeoutError(_normalize(dvalues, count_updates), calls_done, total_calls)
            combination = dt['features'][i]
            m00 = convert_feat_to_mask(combination, M)
            eval_00 = get_contribution_fn(m00)
            calls_done += 1
            if pbar is not None:
                pbar.update(1)
            remaining_indices = sorted(set(range(M)) - set(combination))
            for ind in remaining_indices:
                if deadline is not None and time.time() > deadline:
                    raise XAITimeoutError(_normalize(dvalues, count_updates), calls_done, total_calls)
                m10 = m00.copy()
                m10[ind] = True
                eval_10 = get_contribution_fn(m10)
                calls_done += 1
                if pbar is not None:
                    pbar.update(1)
                eval_diff = eval_10 - eval_00
                dvalues[ind] += eval_diff
                count_updates[ind] += 1
    finally:
        if pbar is not None:
            pbar.close()

    if stats is not None:
        stats["calls_done"] = calls_done

    return _normalize(dvalues, count_updates)
