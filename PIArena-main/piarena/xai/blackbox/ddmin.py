"""Delta debugging (`ddmin`, Zeller & Hildebrandt) for a boolean detector.

Finds a 1-minimal subset of `items` that still makes the detector fire: no
single remaining item can be removed without the detector going quiet. The
oracle is injected (`is_blocked(kept_items) -> bool`), so the algorithm is
testable without any model.

Assumptions worth keeping in mind (see plans/xai-blackbox-datasentinel.md):
  * the full input is blocked (the caller checks it) and the empty input is
    not (the caller's *baseline* check) — `ddmin` never calls the oracle on the
    empty set;
  * the detector is deterministic; it is NOT assumed monotone, but then the
    result is 1-minimal only with respect to the tests actually run, and the
    detector may well have other, disjoint triggering subsets (the one found
    depends on the order tests are run).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence


class BudgetExceeded(RuntimeError):
    """Raised by an oracle wrapper when the per-sample detector-call budget is spent."""


@dataclass
class DDMinResult:
    minimal: list            # surviving items, original order
    n_tests: int             # distinct oracle evaluations performed
    truncated: bool = False  # True if the call budget stopped the search early
    trace: list = field(default_factory=list)  # [(kept_items, flag), ...] when recorded


def _split(items: list, n: int) -> list[list]:
    """Split into `n` contiguous chunks of near-equal size (no empty chunk when n <= len)."""
    size, extra = divmod(len(items), n)
    chunks, start = [], 0
    for i in range(n):
        end = start + size + (1 if i < extra else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


def ddmin(
    items: Sequence,
    is_blocked: Callable[[Sequence], bool],
    record_trace: bool = False,
) -> DDMinResult:
    cache: dict[tuple, bool] = {}
    trace: list = []
    state = {"truncated": False}

    def test(kept: list) -> bool:
        if not kept:  # the empty set is the baseline, assumed benign; never queried
            return False
        key = tuple(kept)
        if key not in cache:
            flag = bool(is_blocked(kept))  # may raise BudgetExceeded
            cache[key] = flag
            if record_trace:
                trace.append((list(kept), flag))
        return cache[key]

    current = list(items)
    n = 2
    try:
        while len(current) >= 2:
            n = min(n, len(current))
            subsets = _split(current, n)
            reduced = False
            for subset in subsets:  # reduce to a single subset
                if test(subset):
                    current, n, reduced = subset, 2, True
                    break
            if not reduced:
                for subset in subsets:  # reduce to a complement
                    drop = set(subset)
                    complement = [x for x in current if x not in drop]
                    if test(complement):
                        current, n, reduced = complement, max(n - 1, 2), True
                        break
            if not reduced:
                if n >= len(current):
                    break
                n = min(len(current), 2 * n)
    except BudgetExceeded:
        state["truncated"] = True

    return DDMinResult(minimal=current, n_tests=len(cache), truncated=state["truncated"], trace=trace)
