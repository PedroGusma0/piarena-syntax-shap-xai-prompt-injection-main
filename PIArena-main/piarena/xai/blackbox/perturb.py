"""Text segmentation + perturbation for the black-box XAI methods.

The detector is a function `text -> bool`, so a perturbation is just a new
string: the words/sentences that are not kept are *deleted* (no placeholder
token), and the kept ones are re-joined in their original order. Pure Python +
numpy; no model, tokenizer or spaCy involved.

Features are whitespace-delimited words (`\\S+`), the same granularity the
Kernel SHAP/IG explainers report, so `tokens`, `values` and `injected_span`
(word indices) line up with what `scripts/xai_metrics.py` and `shap.plots.text`
already expect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

_WORD_RE = re.compile(r"\S+")
_SENT_END_RE = re.compile(r"[.!?]+[\"')\]]*$")
# "e.g.", "U.S.", a lone initial ("J.") ...: not a sentence end on their own.
_INITIAL_RE = re.compile(r"^[A-Za-z]\.$")
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.", "e.g.", "i.e.", "cf.", "no.", "inc.", "ltd.",
}


@dataclass
class Segmentation:
    """A context split into words and sentences, with the original character
    offsets so a perturbed text can be rebuilt from any subset of words."""
    text: str
    words: list[str]
    spans: list[tuple[int, int]]
    sentence_ids: list[int]  # sentence index of each word
    n_sentences: int = 0
    sentence_words: list[list[int]] = field(default_factory=list)  # word indices per sentence

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
    """Split `text` into whitespace words and heuristic sentences.

    A sentence ends at a word that ends in `.`/`!`/`?` (optionally followed by
    closing quotes/brackets), unless it is an abbreviation or a lone initial,
    or the next word starts in lowercase. A blank line also ends a sentence.
    Deliberately simple and deterministic — the same function is used when the
    explanation is computed and when `scripts/xai_metrics.py` re-scores it."""
    matches = list(_WORD_RE.finditer(text))
    words = [m.group(0) for m in matches]
    spans = [(m.start(), m.end()) for m in matches]
    sentence_ids: list[int] = []
    sentence = 0
    for i, word in enumerate(words):
        sentence_ids.append(sentence)
        is_last = i == len(words) - 1
        if is_last:
            break
        ends = bool(_SENT_END_RE.search(word)) and word.lower() not in _ABBREVIATIONS and not _INITIAL_RE.match(word)
        next_word = words[i + 1]
        gap = text[spans[i][1]:spans[i + 1][0]]
        if ends and not next_word[0].islower():
            sentence += 1
        elif "\n\n" in gap:
            sentence += 1
    n_sentences = (sentence + 1) if words else 0
    sentence_words: list[list[int]] = [[] for _ in range(n_sentences)]
    for i, sid in enumerate(sentence_ids):
        sentence_words[sid].append(i)
    return Segmentation(
        text=text, words=words, spans=spans, sentence_ids=sentence_ids,
        n_sentences=n_sentences, sentence_words=sentence_words,
    )
