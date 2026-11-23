"""Exact Shapley values of a boolean detector game over a small set of players
(sentences). `value(coalition) -> 0.0/1.0` is the detector's decision on the
text built from that coalition; the empty coalition is the baseline.

Cost is `2^M` evaluations (cached by the caller), so it is only used when
`M <= max_exact_sentences`. Efficiency holds by construction:
`sum(phi) == value(all) - value(empty)`.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable, Sequence


def exact_shapley(players: Sequence, value: Callable[[frozenset], float], baseline_value: float) -> list[float]:
    """`value(frozenset_of_players)` is called for every non-empty coalition;
    the empty coalition's value is `baseline_value` (never queried)."""
    m = len(players)
    memo: dict[frozenset, float] = {frozenset(): float(baseline_value)}

    def v(coalition: frozenset) -> float:
        if coalition not in memo:
            memo[coalition] = float(value(coalition))
        return memo[coalition]

    phi = [0.0] * m
    for i, player in enumerate(players):
        others = [p for p in players if p != player]
        for size in range(m):
            weight = factorial(size) * factorial(m - size - 1) / factorial(m)
            for subset in combinations(others, size):
                s = frozenset(subset)
                phi[i] += weight * (v(s | {player}) - v(s))
    return phi
