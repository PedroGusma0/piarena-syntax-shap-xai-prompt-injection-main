"""Word segmentation + perturbation for the black-box XAI method.

The detector is a function `text -> bool`, so a perturbation is just a new
string: the words that are not kept are *deleted* (no placeholder token) and
the kept ones are re-joined in their original order. Pure Python + numpy; no
model, tokenizer or spaCy involved.

Features are whitespace-delimited words (`\\S+`), the same granularity the
Kernel SHAP/IG explainers report, so `tokens`, `values` and `injected_span`
(word indices) line up with what `scripts/xai_metrics.py` and `shap.plots.text`
already expect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_WORD_RE = re.compile(r"\S+")


@dataclass
class Segmentation:
    """A context split into words, with the original character offsets so a
    perturbed text can be rebuilt from any subset of words."""
    text: str
    words: list[str]
    spans: list[tuple[int, int]]

    @property
    def n_words(self) -> int:
        return len(self.words)

    def build(self, keep: Sequence[bool] | np.ndarray, empty_text: str = "") -> str:
        """Text made of the kept words only. Consecutive kept words keep their
        original separator (so newlines inside a kept run survive); a gap left
        by deleted words becomes a single space. `empty_text` is returned when
        nothing is kept (e.g. a neutral sentence instead of the empty string)."""
        pieces: list[str] = []
        prev: Optional[int] = None
        for i, kept in enumerate(keep):
            if not kept:
                continue
            if prev is not None:
                pieces.append(self.text[self.spans[prev][1]:self.spans[i][0]] if prev == i - 1 else " ")
            pieces.append(self.words[i])
            prev = i
        return "".join(pieces) if pieces else empty_text

    def keep_mask(self, indices: Sequence[int]) -> np.ndarray:
        mask = np.zeros(self.n_words, dtype=bool)
        if len(indices):
            mask[list(indices)] = True
        return mask

    def word_span(self, char_start: int, char_end: int) -> Optional[dict]:
        """`{"start_token", "end_token"}` (word indices) of the words touching
        the character range, or None if none do. Same shape the Kernel SHAP
        explainer's `injected_span` has."""
        touched = [i for i, (a, b) in enumerate(self.spans) if a < char_end and b > char_start]
        if not touched:
            return None
        return {"start_token": min(touched), "end_token": max(touched) + 1}


def segment(text: str) -> Segmentation:
    """Split `text` into whitespace-delimited words with their offsets."""
    matches = list(_WORD_RE.finditer(text))
    return Segmentation(
        text=text,
        words=[m.group(0) for m in matches],
        spans=[(m.start(), m.end()) for m in matches],
    )
