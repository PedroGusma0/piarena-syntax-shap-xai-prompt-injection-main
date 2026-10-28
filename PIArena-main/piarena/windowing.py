"""Splits a long text into fixed-size token windows.

Prompt-Guard-86M has a 512-token context window, and Meta recommends
"splitting longer inputs into segments and scanning each" one. This module is
the single place where that split is defined, shared by the windowed defense
(`promptguard_windowed`) and the windowed Kernel SHAP explainer, so a window
that blocked a prompt is exactly the window that gets explained.

No heavy imports here: the tokenizer is passed in by the caller.
"""
from typing import Dict, List, Optional

DEFAULT_WINDOW_TOKENS = 512


def split_into_windows(tokenizer, text: str, window_tokens: int = DEFAULT_WINDOW_TOKENS) -> List[Dict]:
    """Returns the windows of `text`, each `{idx, char_start, char_end, token_start, token_end}`.

    * Each window holds at most `window_tokens - tokenizer.num_special_tokens_to_add()`
      content tokens (510 for a 512 window with `[CLS]`/`[SEP]`).
    * A cut never splits a word: it is moved back to the start of the word the
      cut would have split (unless a single word alone exceeds the budget, e.g.
      a very long URL, in which case it is cut hard).
    * The character ranges tile `text` exactly: no gaps, no overlap
      (`windows[0].char_start == 0`, `windows[-1].char_end == len(text)`), so
      concatenating the window texts gives back `text`.
    * Empty / whitespace-only text yields a single window covering it.
    """
    if window_tokens <= tokenizer.num_special_tokens_to_add(pair=False):
        raise ValueError(f"window_tokens={window_tokens} leaves no room for content tokens.")
    budget = window_tokens - tokenizer.num_special_tokens_to_add(pair=False)

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets = encoded["offset_mapping"]
    word_ids = encoded.word_ids(0)
    n = len(offsets)
    if n == 0:
        return [{"idx": 0, "char_start": 0, "char_end": len(text), "token_start": 0, "token_end": 0}]

    # token index where each window starts
    starts = [0]
    while True:
        cut = starts[-1] + budget
        if cut >= n:
            break
        # move the cut back to a word boundary
        j = cut
        wid = word_ids[j]
        while j > starts[-1] and wid is not None and word_ids[j - 1] == wid:
            j -= 1
        if j == starts[-1]:  # a single word longer than the budget: cut hard
            j = cut
        starts.append(j)

    boundaries = [0]
    for s in starts[1:]:
        b = offsets[s][0]
        while b < len(text) and text[b].isspace():  # keep the window start on a non-space char
            b += 1
        boundaries.append(max(b, boundaries[-1]))
    boundaries.append(len(text))

    windows = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else n
        windows.append({
            "idx": k,
            "char_start": boundaries[k],
            "char_end": boundaries[k + 1],
            "token_start": s,
            "token_end": e,
        })
    return windows


def window_texts(text: str, windows: List[Dict]) -> List[str]:
    return [text[w["char_start"]:w["char_end"]] for w in windows]


def overlaps(window: Dict, char_start: int, char_end: int) -> bool:
    """True if `[char_start, char_end)` intersects the window's character range."""
    return window["char_start"] < char_end and window["char_end"] > char_start


def find_injection_range(context: str, injected_task: Optional[str], prefix: str = "") -> Optional[tuple]:
    """Character range `(start, end)` of the injected task inside `context`, or None.

    Tries `prefix + injected_task` first (heuristic attacks prepend a fixed
    string), then the bare `injected_task`. Same rule `explain_context` uses.
    """
    if not injected_task:
        return None
    if prefix:
        start = context.find(prefix + injected_task)
        if start != -1:
            return start, start + len(prefix) + len(injected_task)
    start = context.find(injected_task)
    if start != -1:
        return start, start + len(injected_task)
    return None


def flatten_windows(xai_result: dict, threshold: float = 0.5, scope: str = "flagged") -> Optional[dict]:
    """Flat `{tokens, values, injected_span, p_malign, ...}` view over the windows of a
    windowed `xai_result`, restricted to a scope:

    * `"all"`      — every window
    * `"flagged"`  — windows with P(malign) >= `threshold` (the ones that blocked)
    * `"injected"` — windows containing the injection

    Used by the global-ranking plots (`bar_global`, `fig9_frequency`) so the
    ~10 benign windows per sample don't dilute the ranking. Returns None if the
    result has no windows or none is selected. `injected_span` is re-based to
    the concatenated words (None if the selected windows hold no injection)."""
    entries = (xai_result or {}).get("windows")
    if not entries:
        return None
    if scope == "all":
        chosen = list(entries)
    elif scope == "flagged":
        chosen = [e for e in entries if e["p_malign"] >= threshold]
    elif scope == "injected":
        chosen = [e for e in entries if e.get("contains_injection")]
    else:
        raise ValueError(f"unknown window scope {scope!r} (expected all|flagged|injected)")
    chosen = [e for e in chosen if e.get("values")]
    if not chosen:
        return None
    tokens, values, span, offset = [], [], None, 0
    for e in chosen:
        tokens.extend(e["tokens"])
        values.extend(e["values"])
        s = e.get("injected_span")
        if s:
            lo, hi = s["start_token"] + offset, s["end_token"] + offset
            span = {"start_token": lo, "end_token": hi} if span is None else {
                "start_token": min(span["start_token"], lo), "end_token": max(span["end_token"], hi)}
        offset += len(e["tokens"])
    top = max(chosen, key=lambda e: e["p_malign"])
    return {
        "tokens": tokens, "values": values, "injected_span": span, "windowed": True,
        "n_windows_used": len(chosen), "base_value": top["base_value"], "full_value": top["full_value"],
        "p_malign": top["p_malign"], "p_benign": top["p_benign"], "predicted_label": top["predicted_label"],
        "backend": top.get("backend"),
    }
