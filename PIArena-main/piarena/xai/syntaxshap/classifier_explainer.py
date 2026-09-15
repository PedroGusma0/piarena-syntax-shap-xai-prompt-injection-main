"""SyntaxSHAP adapted to explain a text CLASSIFIER (a single forward pass),
instead of the original autoregressive-generation target.

See plans/xai-syntaxshap-promptguard.md for the full design rationale. In
short: the syntax-aware coalition game (`coalition.py`) is agnostic to what
kind of model produces the score being explained — the original vendored
`SyntaxExplainer` just happened to hard-wire that scoring step to
`captum.attr.LLMAttribution` + `.generate()`. Here the scoring step is a
plain forward pass through a HuggingFace text-classification pipeline
(wrapped as `thirdparty.models.TransformersPipeline`), and the "target" is a
binary aggregate — P(malign) = sum of every predicted class whose label
does not contain the substring "benign" — rather than an argmax over however
many raw classes the classifier happens to expose. That specific design
choice (and why it matters) is documented in classifier_explainer's
`_compute_malign_indices` and in the plan's "Por que agregar em binário"
section.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from ._thirdparty_path import ensure_thirdparty_on_path

ensure_thirdparty_on_path()

import maskers  # noqa: E402  (thirdparty, flat import — see _thirdparty_path)
from utils import get_token_dependency_tree  # noqa: E402
from utils.transformers import parse_prefix_suffix_for_tokenizer  # noqa: E402

from .coalition import compute_shapley_values, XAITimeoutError  # noqa: E402

log = logging.getLogger(__name__)


def _attack_prefixes() -> dict:
    """`{attack_name: fixed_prefix_string}` for the heuristic attacks
    (`piarena/attacks/heuristic.py`) — read directly off each attack class's
    `PREFIX` attribute, so there is exactly one copy of each template string
    in the codebase (the attack class itself), not a second one duplicated
    here that could drift out of sync.

    Importing `piarena.attacks.heuristic` also runs `piarena/attacks/
    __init__.py`, which pulls in nanogcg/pair/tap/strategy_search (and their
    heavier deps) — harmless in practice because `main.py` already imports
    `piarena.attacks` unconditionally near the top, before XAI is even
    instantiated, so nothing extra gets paid for on the real run path. What
    this buys is keeping that cost out of `piarena.xai`'s own *module-level*
    import: this function is called lazily, from inside `explain_context`,
    not at import time — so `scripts/xai_metrics.py` (which imports
    `classifier_explainer.py` but never calls `explain_context`) never
    triggers it.
    """
    from ...attacks import heuristic as _h

    classes = (_h.DirectAttack, _h.IgnoreAttack, _h.CompletionAttack, _h.CharacterAttack, _h.CombinedAttack)
    return {cls.name: getattr(cls, "PREFIX", "") for cls in classes}


class ClassifierSyntaxExplainer:
    """Explains a `thirdparty.models.TransformersPipeline`-wrapped text
    classifier using syntax-tree-restricted Shapley coalitions.

    Not a subclass of the vendored `explainers.SyntaxExplainer` — see the
    "Correções de arquitetura" section of the plan for why (in short:
    `MaskedModel`/`SyntaxExplainer.__init__` assume an autoregressive
    generation model at every turn, down to requiring `.get_outputs()`,
    `.generate()`, and a captum `LLMAttribution`; none of that applies to a
    classifier, and stripping it all out leaves almost nothing to inherit).
    """

    def __init__(self, pipeline_model, tokenizer, algorithm: str = "syntax", spacy_model: str = "en_core_web_sm"):
        """
        Parameters
        ----------
        pipeline_model : thirdparty.models.TransformersPipeline
            Wraps a HF `pipeline("text-classification", ..., top_k=None)` —
            must expose `.id2label` and return an (n_samples, n_classes) array.
        tokenizer : a raw HF PreTrainedTokenizer(Fast) — the *same* tokenizer
            the classifier itself uses.
        algorithm : "syntax" or "shap".
        """
        if algorithm not in ("syntax", "shap"):
            raise ValueError(f"algorithm must be 'syntax' or 'shap', got {algorithm!r}")
        self.model = pipeline_model
        self.tokenizer = tokenizer
        self.masker = maskers.Text(tokenizer)
        self.algorithm = algorithm
        self.keep_prefix = self.masker.keep_prefix
        self.keep_suffix = self.masker.keep_suffix
        self.malign_indices = self._compute_malign_indices(pipeline_model.id2label)

        self._nlp = None
        self._spacy_model = spacy_model

    @staticmethod
    def _compute_malign_indices(id2label: dict) -> list[int]:
        """Which class indices count as "malign" — i.e. an attack signal.

        Reuses the exact same substring check `PromptGuardDefense.execute`
        already uses (`"benign" not in prediction.lower()`) instead of
        hardcoding label names like "INJECTION"/"JAILBREAK", so this stays
        consistent with whatever the real defense treats as "detected" even
        if the label set differs from what we assumed while planning (see
        Risco #2 do plano).
        """
        indices = [i for i, label in id2label.items() if "benign" not in str(label).lower()]
        if not indices:
            raise ValueError(
                f"Could not find a 'benign' class among id2label={id2label!r} — "
                "cannot compute P(malign). Check the classifier's label set."
            )
        return indices

    def _ensure_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self._spacy_model)
        return self._nlp

    # -- scoring -----------------------------------------------------------

    def _mask_to_string(self, mask: np.ndarray, text: str) -> str:
        """Mask is 1-D, length M (content tokens) — Text.__call__ auto-pads
        prefix/suffix, do NOT reshape to (1, -1) (it expects a 1-D array)."""
        out = self.masker(np.asarray(mask, dtype=bool), text)
        return str(out[0][0])

    def _score(self, text: str) -> float:
        """P(malign) for `text` — a single forward pass through the
        classifier, no MaskedModel/captum involved."""
        scores = self.model([text])[0]
        return float(sum(scores[i] for i in self.malign_indices))

    def _full_scores(self, text: str):
        return self.model([text])[0]

    def _content_token_bounds(self, full_len: int) -> tuple[int, int]:
        end = full_len - self.keep_suffix if self.keep_suffix else full_len
        return self.keep_prefix, end

    def _rebase_dependency_tree(self, dependency_dt, M: int):
        """`get_token_dependency_tree`'s `token_position` column indexes into
        the *raw* tokenization (prefix special tokens included, e.g. [CLS] at
        position 0) — but `coalition.py` expects 0-based indices into the M
        *content* tokens only. Confirmed empirically (test_classifier_explainer):
        without this, `token_position` values off by `keep_prefix` overflow
        the M-length mask array. Also drops any row that (after rebasing)
        still falls outside [0, M) — defensive, in case of the sentence-vs-
        tree token-count mismatches already handled by the callers."""
        dependency_dt = dependency_dt.copy()
        dependency_dt['token_position'] = dependency_dt['token_position'] - self.keep_prefix
        return dependency_dt[(dependency_dt['token_position'] >= 0) & (dependency_dt['token_position'] < M)]

    # -- single-sentence primitive ------------------------------------------

    def explain_row(self, text: str, progress: bool = False, timeout_seconds: float | None = None) -> dict:
        """Explain a single (short, one-sentence-ish) piece of text on its
        own — the primitive the paper itself was tested against. Used
        directly for spans like `injected_task`, and internally per-sentence
        by `explain_context` for longer, multi-sentence text.

        `timeout_seconds` (`None` = no timeout, exact original behavior): a
        wall-clock budget for the Shapley computation itself. On expiry the
        returned dict still has the normal shape/fields, but `values` holds
        whatever was computed before the deadline (see `XAITimeoutError` in
        `coalition.py`) and `"timed_out"` is `True` instead of raising up to
        the caller — a slow row shouldn't crash a whole `main.py` run."""
        deadline = time.time() + timeout_seconds if timeout_seconds is not None else None
        timed_out = False
        row_stats: dict = {}  # cost telemetry (widest_level/total_calls/calls_done) — see coalition.py::compute_shapley_values
        tokens, token_ids = self.masker.token_segments(text)
        full_len = len(token_ids)
        start, end = self._content_token_bounds(full_len)
        M = end - start
        tokens_content = list(tokens[start:end])

        full_scores = self._full_scores(text)
        predicted_label = self.model.id2label[int(np.argmax(full_scores))]
        p_malign = float(sum(full_scores[i] for i in self.malign_indices))

        base_value = self._score(self._mask_to_string(np.zeros(M, dtype=bool), text))
        full_value = p_malign

        if M <= 0:
            values = np.zeros(0)
        elif self.algorithm == "syntax":
            dependency_dt = self._rebase_dependency_tree(get_token_dependency_tree(text, self.tokenizer), M)
            tree_m = len(dependency_dt)
            if tree_m != M:
                log.warning(
                    "get_token_dependency_tree token count (%d) != tokenizer content-token "
                    "count (%d) for text=%r — using the smaller of the two to stay in bounds "
                    "(see Risco #1 do plano, validar na Fase 0).", tree_m, M, text,
                )
                M = min(tree_m, M)
                tokens_content = tokens_content[:M]
                # tree_m counts *rows*, which may not all have token_position < M
                # if positions are non-contiguous — re-filter defensively.
                dependency_dt = dependency_dt[dependency_dt['token_position'] < M]
            try:
                values = compute_shapley_values(
                    M,
                    lambda mask: self._score(self._mask_to_string(mask, text)),
                    algorithm=self.algorithm,
                    dependency_dt=dependency_dt,
                    progress=progress,
                    progress_desc="row",
                    deadline=deadline,
                    stats=row_stats,
                )
            except XAITimeoutError as e:
                values = e.partial_values
                timed_out = True
                row_stats["calls_done"] = e.calls_done
                row_stats.setdefault("total_calls", e.total_calls)
        else:  # "shap" — exact/unrestricted, only viable for small M
            try:
                values = compute_shapley_values(
                    M,
                    lambda mask: self._score(self._mask_to_string(mask, text)),
                    algorithm="shap",
                    progress=progress,
                    progress_desc="row",
                    deadline=deadline,
                    stats=row_stats,
                )
            except XAITimeoutError as e:
                values = e.partial_values
                timed_out = True
                row_stats["calls_done"] = e.calls_done
                row_stats.setdefault("total_calls", e.total_calls)

        return {
            "tokens": tokens_content,
            "values": values.tolist(),
            "base_value": base_value,
            "full_value": full_value,
            "p_malign": p_malign,
            "predicted_label": predicted_label,
            "timed_out": timed_out,
            # Cost telemetry (e.g. for W&B logging in main.py) — `None` when
            # M<=0 (no compute_shapley_values call happened at all).
            "total_forward_passes": row_stats.get("total_calls"),
            "widest_dependency_level": row_stats.get("widest_level"),
        }

    # -- whole-context orchestration (multi-sentence) -----------------------

    def _sentence_char_spans(self, context: str):
        doc = self._ensure_nlp()(context)
        return [(sent.text, sent.start_char, sent.end_char) for sent in doc.sents if sent.text.strip()]

    def _char_span_to_content_token_range(self, offsets, start_char: int, end_char: int, full_len: int):
        """Map a character span of `context` onto content-token indices
        (0-based, excluding prefix/suffix) of `context`'s own tokenization."""
        keep_prefix, keep_suffix_end = self._content_token_bounds(full_len)
        idxs = [
            i - keep_prefix
            for i, (a, b) in enumerate(offsets)
            if b > a and a < end_char and b > start_char and keep_prefix <= i < keep_suffix_end
        ]
        if not idxs:
            return None
        return min(idxs), max(idxs) + 1  # end exclusive

    def explain_context(
        self, context: str, injected_task: Optional[str] = None, attack_name: Optional[str] = None,
        progress: bool = False, timeout_seconds: float | None = None,
    ) -> dict:
        """Explain the *whole* `context` (what PromptGuardDefense actually
        receives), not just an isolated span — see "Como lidar com contexto
        longo/multi-sentença" do plano.

        Strategy (matches the SyntaxSHAP paper's own suggested scaling path
        for multi-sentence text): segment into sentences, run the per-sentence
        Shapley computation independently for each one with the *rest of the
        context held fixed* (every coalition tested is scored against the
        full context text, only that sentence's tokens vary), then stitch the
        per-sentence values into one array covering the whole context.

        `attack_name` (e.g. "direct"/"ignore"/"completion"/"combined") lets
        `injected_span` localization account for the fixed prefix a heuristic
        attack prepends to `injected_task` before injecting it (see
        `_attack_prefixes()`) — without it (or for an attack with no known
        prefix, e.g. an optimization-based one), localization falls back to
        matching bare `injected_task`, same as before.

        `timeout_seconds` (`None` = no timeout, exact original behavior): a
        *single* wall-clock budget for the entire `context`, computed once
        here and shared across every sentence's `compute_shapley_values` call
        below — not reset per sentence, since the point is to bound the cost
        of explaining one sample, not one sentence within it. The moment a
        sentence's computation exceeds the remaining budget, that sentence's
        partial values are kept, no further sentences are started, and the
        returned dict carries `"timed_out": True` plus `sentences_explained`/
        `sentences_total` — the (dict) contract stays the same either way, so
        callers (`xai_syntaxshap.py`, `main.py`, `scripts/xai_metrics.py`)
        never see a raised exception just because a sample was slow."""
        deadline = time.time() + timeout_seconds if timeout_seconds is not None else None
        timed_out = False
        sentences_explained = 0
        n_sentences = 0
        # Cost telemetry aggregated across every sentence actually attempted
        # (e.g. for W&B logging in main.py) — see coalition.py::
        # compute_shapley_values's `stats` param. `total_forward_passes` sums
        # (the true total cost of explaining this whole context);
        # `widest_dependency_level` takes the max (the single worst offender,
        # Finding 1 of the cost analysis — one bad sentence, not the sum,
        # is what actually determines whether the sample times out).
        total_forward_passes = 0
        widest_dependency_level = 0
        tokens, token_ids = self.masker.token_segments(context)
        full_len = len(token_ids)
        start, end = self._content_token_bounds(full_len)
        M_context = end - start
        tokens_content = list(tokens[start:end])

        full_scores = self._full_scores(context)
        predicted_label = self.model.id2label[int(np.argmax(full_scores))]
        p_malign = float(sum(full_scores[i] for i in self.malign_indices))
        full_value = p_malign

        base_value = self._score(self._mask_to_string(np.zeros(M_context, dtype=bool), context))

        values = np.zeros(M_context, dtype=float)
        if M_context > 0:
            encoded = self.tokenizer(context, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]

            # Materialized (not a generator) so `n_sentences` is known upfront —
            # per-sentence progress reporting below needs "i/N" before the
            # first sentence even starts, and per-sentence cost varies wildly
            # (Finding 1: driven by the widest dependency-tree level, not
            # sentence length), so knowing *which* sentence is running is as
            # important as the %-through-this-sentence number from
            # `compute_shapley_values` itself.
            sentences = self._sentence_char_spans(context)
            n_sentences = len(sentences)

            for sent_idx, (sent_text, sent_start_char, sent_end_char) in enumerate(sentences):
                if deadline is not None and time.time() > deadline:
                    # Budget already exhausted by an earlier sentence — don't
                    # even start this one (parsing/tree-building for it would
                    # just be wasted work on top of an already-decided timeout).
                    timed_out = True
                    break
                span = self._char_span_to_content_token_range(offsets, sent_start_char, sent_end_char, full_len)
                if span is None:
                    continue
                tok_start, tok_end = span
                tok_start = max(0, min(tok_start, M_context))
                tok_end = max(tok_start, min(tok_end, M_context))
                sent_M_from_context = tok_end - tok_start
                if sent_M_from_context <= 0:
                    continue

                if self.algorithm == "syntax":
                    dependency_dt = self._rebase_dependency_tree(
                        get_token_dependency_tree(sent_text, self.tokenizer), sent_M_from_context
                    )
                    # Authoritative token count for THIS sentence's coalition
                    # geometry is the tree's own (it defines `causal_ordering`);
                    # the context-offset-derived span only tells us *where* in
                    # the full-context mask this sentence's tokens live. When
                    # the two disagree (tokenizing a sentence standalone can
                    # split slightly differently than as part of the full
                    # context — Risco #5 do plano) we clip to the smaller of
                    # the two rather than raising, and log it for Fase 0 review.
                    sent_M = len(dependency_dt)
                    if sent_M != sent_M_from_context:
                        log.warning(
                            "Sentence token-count mismatch (tree=%d, context-slice=%d) for "
                            "sentence=%r — clipping to the smaller (Risco #5 do plano).",
                            sent_M, sent_M_from_context, sent_text,
                        )
                        sent_M = min(sent_M, sent_M_from_context)
                        # tree row count may not mean all positions < sent_M
                        # if positions are non-contiguous — re-filter defensively
                        # (same reasoning as explain_row).
                        dependency_dt = dependency_dt[dependency_dt['token_position'] < sent_M]
                    tok_end = tok_start + sent_M
                else:
                    dependency_dt = None
                    sent_M = sent_M_from_context

                if sent_M <= 0:
                    continue

                sent_desc = f"sentence {sent_idx + 1}/{n_sentences}"
                if progress:
                    preview = sent_text if len(sent_text) <= 60 else sent_text[:57] + "..."
                    print(f"[syntaxshap] {sent_desc}: M={sent_M} tokens — {preview!r}")

                def get_contribution_fn(local_mask, _start=tok_start, _end=tok_end, _M=M_context):
                    full_mask = np.zeros(_M, dtype=bool)
                    full_mask[_start:_end] = local_mask
                    return self._score(self._mask_to_string(full_mask, context))

                sent_stats: dict = {}
                try:
                    sent_values = compute_shapley_values(
                        sent_M,
                        get_contribution_fn,
                        algorithm=self.algorithm,
                        dependency_dt=dependency_dt,
                        progress=progress,
                        progress_desc=sent_desc,
                        deadline=deadline,
                        stats=sent_stats,
                    )
                except XAITimeoutError as e:
                    values[tok_start:tok_end] = e.partial_values
                    timed_out = True
                    total_forward_passes += e.calls_done
                    widest_dependency_level = max(widest_dependency_level, sent_stats.get("widest_level", 0))
                    if progress:
                        print(f"[syntaxshap] {sent_desc}: timed out "
                              f"({e.calls_done}/{e.total_calls} forward passes done) — "
                              "stopping, remaining sentences of this context left unexplained.")
                    break
                values[tok_start:tok_end] = sent_values
                sentences_explained += 1
                total_forward_passes += sent_stats.get("total_calls", 0)
                widest_dependency_level = max(widest_dependency_level, sent_stats.get("widest_level", 0))

        injected_span = None
        if injected_task:
            prefix = _attack_prefixes().get(attack_name, "") if attack_name else ""
            char_start = context.find(prefix + injected_task) if prefix else -1
            if char_start != -1:
                char_end = char_start + len(prefix) + len(injected_task)
            else:
                # Either no known prefix for this attack, or the prefixed
                # form wasn't found (e.g. a cached injected_context from a
                # different attack_name) — fall back to bare injected_task,
                # the original (never-regressing) behavior.
                char_start = context.find(injected_task)
                char_end = char_start + len(injected_task) if char_start != -1 else -1
            if char_start != -1:
                encoded = self.tokenizer(context, return_offsets_mapping=True)
                span = self._char_span_to_content_token_range(encoded["offset_mapping"], char_start, char_end, full_len)
                if span is not None:
                    injected_span = {"start_token": span[0], "end_token": span[1]}

        return {
            "tokens": tokens_content,
            "values": values.tolist(),
            "base_value": base_value,
            "full_value": full_value,
            "p_malign": p_malign,
            "predicted_label": predicted_label,
            "injected_span": injected_span,
            "timed_out": timed_out,
            "sentences_explained": sentences_explained,
            "sentences_total": n_sentences,
            # Cost telemetry (e.g. for W&B logging in main.py) — see the
            # accumulator comments above. 0 when M_context<=0 (no sentence
            # ever ran).
            "total_forward_passes": total_forward_passes,
            "widest_dependency_level": widest_dependency_level,
        }
