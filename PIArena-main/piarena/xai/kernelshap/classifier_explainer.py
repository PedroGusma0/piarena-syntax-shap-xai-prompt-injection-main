"""Kernel SHAP explainer for a text CLASSIFIER (a single forward pass).

Independent design from the parallel SyntaxSHAP experiment
(`piarena_xai_syntax_shap/piarena/xai/syntaxshap/`) — no dependency tree, no
per-sentence orchestration, no vendored third-party code. See
`plans/xai-kernelshap-promptguard.md` for the full design rationale. In short:

- Two coexisting computation **backends**, selected via `backend=` ("captum"
  = `captum.attr.KernelShap`, the original implementation; "shap" = the
  official `shap.KernelExplainer`) — switch by config, not by editing code.
  Both share every step except the actual Kernel SHAP regression itself: same
  tokenization/word-grouping, same masking mechanism, same output contract.
- Operates on **token-id tensors**, not text — the classifier is loaded
  directly via `AutoModelForSequenceClassification` (no HF `pipeline()`
  wrapper), so a coalition never needs to be re-serialized to a string and
  re-tokenized.
- The interpretable "feature" is a **word**, not a token: a fast tokenizer's
  `.word_ids()` already returns `None` for special tokens (CLS/SEP/PAD) and a
  stable, contiguous, 0-based word index for every real subtoken — this is
  the modern, library-provided equivalent of the manual CLS/SEP-detection fix
  the SyntaxSHAP experiment had to hand-roll for its dependency tree, and it
  also means every subtoken of a fragmented word (a URL, a non-Latin script)
  collapses into a single interpretable feature automatically.
- One `context` is explained as a **single joint coalition game** (one call
  to the backend's Kernel SHAP routine covering every word at once) — this is
  not a simplification of the "real" Kernel SHAP, it *is* Kernel SHAP's
  defining property (a single weighted regression recovers every feature's
  Shapley value jointly; see kernel-shap.md#4.1's contrast with "Shapley
  sampling values", which estimates one feature at a time). There is no
  per-sentence decomposition — that existed in SyntaxSHAP only to contain an
  exponential per-level cost that does not exist here (cost = `n_samples`,
  chosen).
- The explained target is P(malign) = sum of every predicted class whose
  label does not contain the substring "benign" — matching exactly what
  `PromptGuardDefense.execute`'s `detect_flag` checks.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# HF's default `model_max_length` sentinel for tokenizers that never had an
# explicit max length configured is an absurdly large int (commonly
# 1_000_000_000_000_000_019_884_624_838_656 for some fast tokenizers) rather
# than a usable value — falling back on it directly would defeat truncation
# entirely. A well-known HF footgun, not specific to this codebase.
_FALLBACK_MAX_LENGTH = 512
_SENTINEL_MAX_LENGTH_THRESHOLD = 100_000

_BACKENDS = ("captum", "shap")


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


class ClassifierKernelExplainer:
    """Explains an `AutoModelForSequenceClassification` classifier using
    Kernel SHAP, with word-level feature grouping. `backend` selects which
    library actually solves the Shapley regression (`"captum"` or `"shap"`)
    — everything else (tokenization, masking, output shape) is identical
    between the two.
    """

    def __init__(
        self,
        model,
        tokenizer,
        baseline_token: str = "mask",
        n_samples="auto",
        max_n_samples: int = 20000,
        perturbations_per_eval: int = 8,
        max_length: Optional[int] = None,
        backend: str = "captum",
    ):
        """
        Parameters
        ----------
        model : a HuggingFace `AutoModelForSequenceClassification` instance
            (already `.eval()`'d and moved to its target device by the
            caller). Must expose `.config.id2label`.
        tokenizer : the *same* tokenizer the model uses — must be a fast
            tokenizer (`tokenizer.is_fast`) with working `.word_ids()`
            (Risco #1 do plano — verified once, at construction time).
        baseline_token : "mask" | "pad" | "unk" — which special token id
            stands in for a "missing" word. Falls back mask -> pad -> unk if
            the preferred one isn't defined on this tokenizer (Risco #4).
        n_samples : "auto" (resolves per-call to `min(2*M + 2048,
            max_n_samples)`, M = number of words) or a fixed int. NEVER left
            to captum's own default (`25` — dangerous for M in the dozens,
            see Risco #3 do plano). Also used as the `shap` backend's
            `nsamples`, for budget parity between the two.
        perturbations_per_eval : batch size for how many sampled coalitions
            are scored per forward call — captum's own native batching knob
            for that backend, and the chunk size this class uses internally
            when batching the `shap` backend's `f(X)`.
        max_length : truncation length for the tokenizer; `None` resolves to
            `tokenizer.model_max_length`, guarded against the common HF
            sentinel-default footgun (Risco #5).
        backend : "captum" (default, `captum.attr.KernelShap`) or "shap"
            (official `shap.KernelExplainer`) — which library actually solves
            the Shapley regression. The two coexist and produce the same
            output contract; switch via config, not by editing this file.
        """
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError(
                f"ClassifierKernelExplainer requires a fast tokenizer with working "
                f".word_ids() for word-level feature grouping, but {tokenizer!r} is "
                f"not fast (Risco #1 do plano — validar antes da rodada real)."
            )
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}, expected one of {_BACKENDS}.")
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.n_samples_config = n_samples
        self.max_n_samples = max_n_samples
        self.perturbations_per_eval = perturbations_per_eval
        self.backend = backend

        configured_max_length = max_length or getattr(tokenizer, "model_max_length", None)
        if not configured_max_length or configured_max_length > _SENTINEL_MAX_LENGTH_THRESHOLD:
            configured_max_length = _FALLBACK_MAX_LENGTH
        self.max_length = configured_max_length

        self.baseline_token_id = self._resolve_baseline_token_id(baseline_token)
        self.malign_indices = self._compute_malign_indices(model.config.id2label)

    @staticmethod
    def _compute_malign_indices(id2label: dict) -> list:
        """Which class indices count as "malign" — i.e. an attack signal.

        Reuses the exact same substring check `PromptGuardDefense.execute`
        already uses (`"benign" not in prediction.lower()`) instead of
        hardcoding label names like "INJECTION"/"JAILBREAK", so this stays
        consistent with whatever the real defense treats as "detected" even
        if the label set differs from what was assumed while planning
        (Risco #2 do plano)."""
        indices = [i for i, label in id2label.items() if "benign" not in str(label).lower()]
        if not indices:
            raise ValueError(
                f"Could not find a 'benign' class among id2label={id2label!r} — "
                "cannot compute P(malign). Check the classifier's label set."
            )
        return indices

    def _resolve_baseline_token_id(self, preferred: str) -> int:
        candidates = {
            "mask": self.tokenizer.mask_token_id,
            "pad": self.tokenizer.pad_token_id,
            "unk": self.tokenizer.unk_token_id,
        }
        order = [preferred] + [k for k in ("mask", "pad", "unk") if k != preferred]
        for key in order:
            token_id = candidates.get(key)
            if token_id is not None:
                if key != preferred:
                    log.warning(
                        "Tokenizer has no %s token id; falling back to %s (id=%d) "
                        "as the Kernel SHAP baseline (Risco #4 do plano).",
                        preferred, key, token_id,
                    )
                return token_id
        raise RuntimeError(
            f"Tokenizer {self.tokenizer!r} defines none of mask/pad/unk token ids — "
            "cannot pick a Kernel SHAP baseline token."
        )

    def _resolve_n_samples(self, M: int) -> int:
        if self.n_samples_config == "auto":
            return min(2 * M + 2048, self.max_n_samples)
        return int(self.n_samples_config)

    # -- tokenization / word-feature grouping --------------------------------

    def _tokenize_content(self, text: str):
        """Tokenize `text`, split into (prefix special tokens, content
        tokens, suffix special tokens) via `.word_ids()` (`None` = special
        token — the fast-tokenizer-native equivalent of the manual CLS/SEP
        fix the SyntaxSHAP experiment needed for its dependency tree)."""
        import torch

        encoded = self.tokenizer(
            text, return_offsets_mapping=True, truncation=True, max_length=self.max_length,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        word_ids = encoded.word_ids(0)

        content_idx = [i for i, w in enumerate(word_ids) if w is not None]
        if not content_idx:
            empty_ids = torch.zeros(0, dtype=torch.long)
            empty_mask = torch.zeros(0, dtype=torch.long)
            return (
                torch.tensor(input_ids, dtype=torch.long),  # whole (special-tokens-only) sequence as "prefix"
                empty_ids,
                empty_ids,
                empty_mask,
                0,
                [],
            )

        start, end = content_idx[0], content_idx[-1] + 1
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        prefix_ids = input_ids_t[:start]
        suffix_ids = input_ids_t[end:]
        content_ids = input_ids_t[start:end]
        content_word_ids = word_ids[start:end]
        content_offsets = offsets[start:end]

        # HF's word_ids() numbers words 0..N-1 in order of first appearance,
        # so the content span's word ids should already be a contiguous
        # 0..M_words-1 range. Re-map defensively (by first-occurrence order)
        # in case an exotic tokenizer doesn't uphold that contract exactly.
        seen = {}
        remapped = []
        for w in content_word_ids:
            if w not in seen:
                seen[w] = len(seen)
            remapped.append(seen[w])
        feature_mask = torch.tensor(remapped, dtype=torch.long)
        M_words = len(seen)

        return prefix_ids, suffix_ids, content_ids, feature_mask, M_words, content_offsets

    def _word_strings(self, text: str, content_offsets, feature_mask, M_words: int) -> list:
        """Reconstruct one display string per word feature by slicing the
        original text at the min/max character offset of its subtokens —
        avoids detokenization artifacts (extra/missing spaces) that decoding
        subtoken ids back to text can introduce."""
        starts = [None] * M_words
        ends = [None] * M_words
        for (a, b), fid in zip(content_offsets, feature_mask.tolist()):
            if starts[fid] is None or a < starts[fid]:
                starts[fid] = a
            if ends[fid] is None or b > ends[fid]:
                ends[fid] = b
        return [text[s:e] for s, e in zip(starts, ends)]

    # -- scoring --------------------------------------------------------------

    def _forward_func(self, prefix_ids, suffix_ids):
        """Returns a closure `forward(content_ids_batch) -> Tensor[B]` —
        already the scalar P(malign) per example, so captum's
        `KernelShap.attribute()` never needs a `target=` index, and the
        `shap` backend's `f(X)` never needs multi-output handling either."""
        import torch

        def forward(content_ids_batch):
            B = content_ids_batch.shape[0]
            prefix_batch = prefix_ids.unsqueeze(0).expand(B, -1).to(content_ids_batch.device)
            suffix_batch = suffix_ids.unsqueeze(0).expand(B, -1).to(content_ids_batch.device)
            full_ids = torch.cat([prefix_batch, content_ids_batch, suffix_batch], dim=1)
            attention_mask = torch.ones_like(full_ids)
            with torch.no_grad():
                logits = self.model(
                    input_ids=full_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                ).logits
                probs = torch.softmax(logits, dim=-1)
                p_malign = probs[:, self.malign_indices].sum(dim=-1)
            return p_malign.to(content_ids_batch.device)

        return forward

    def _score_with_mask_batch(
        self, keep_mask_words_batch: np.ndarray, prefix_ids, suffix_ids, content_ids, feature_mask,
    ) -> np.ndarray:
        """Batched primitive: `keep_mask_words_batch` is `(B, M_words)` bool
        (True = keep the original word, False = replace with the baseline
        token). Returns `(B,)` P(malign) scores, one per row. Shared by
        `_score_with_mask` (B=1, used by `make_rescorer`/the captum path's
        `base_value`) and the `shap` backend's `f(X)` (B=`nsamples`, chunked
        by the caller into batches of `perturbations_per_eval` rows)."""
        import torch

        keep_words_t = torch.tensor(keep_mask_words_batch, dtype=torch.bool)  # (B, M_words)
        keep_token_mask = keep_words_t[:, feature_mask]  # (B, len(content_ids))
        content_batch = content_ids.unsqueeze(0).expand(keep_token_mask.shape[0], -1)
        masked_content = torch.where(
            keep_token_mask, content_batch, torch.full_like(content_batch, self.baseline_token_id)
        )
        forward = self._forward_func(prefix_ids, suffix_ids)
        return forward(masked_content).detach().cpu().numpy()

    def _score_with_mask(self, keep_mask_words: np.ndarray, prefix_ids, suffix_ids, content_ids, feature_mask) -> float:
        """`rescore_fn(keep_mask)` primitive consumed by `piarena/xai/metrics.py`
        and `scripts/xai_metrics.py` — masks every word NOT in `keep_mask_words`
        to the baseline token and re-scores. Single-row wrapper around
        `_score_with_mask_batch`."""
        scores = self._score_with_mask_batch(
            keep_mask_words[np.newaxis, :], prefix_ids, suffix_ids, content_ids, feature_mask
        )
        return float(scores[0])

    def make_rescorer(self, text: str):
        """Returns a `rescore_fn(keep_mask_words: np.ndarray) -> float` closure
        for `text`, matching exactly the masking mechanism `explain_row`/
        `explain_context` themselves use — the primitive `piarena/xai/metrics.py`
        (`fidelity`/`acc_at_1`) and `scripts/xai_metrics.py` are built around.
        Backend-agnostic: re-scoring always runs the model directly, whether
        the original explanation was computed via captum or shap."""
        prefix_ids, suffix_ids, content_ids, feature_mask, M_words, _ = self._tokenize_content(text)

        def rescore(keep_mask_words: np.ndarray) -> float:
            if M_words == 0:
                # No content words to mask — score is whatever the (special-
                # tokens-only) input already scores.
                return float(self._forward_func(prefix_ids, suffix_ids)(content_ids.unsqueeze(0))[0])
            return self._score_with_mask(keep_mask_words, prefix_ids, suffix_ids, content_ids, feature_mask)

        return rescore

    # -- backend-specific Shapley computation --------------------------------

    def _compute_values_captum(self, content_ids, feature_mask, prefix_ids, suffix_ids, n_samples, progress):
        import torch
        from captum.attr import KernelShap

        forward = self._forward_func(prefix_ids, suffix_ids)
        baselines = torch.full_like(content_ids, self.baseline_token_id)
        attributions = KernelShap(forward).attribute(
            inputs=content_ids.unsqueeze(0).to(self.device),
            baselines=baselines.unsqueeze(0).to(self.device),
            feature_mask=feature_mask.unsqueeze(0).to(self.device),
            n_samples=n_samples,
            perturbations_per_eval=self.perturbations_per_eval,
            return_input_shape=False,
            show_progress=progress,
        )
        return attributions.detach().cpu().numpy().reshape(-1).tolist()

    def _compute_values_shap(self, content_ids, feature_mask, prefix_ids, suffix_ids, M_words, n_samples, progress):
        """Alternative backend: official `shap.KernelExplainer` instead of
        `captum.attr.KernelShap`. Same masking semantics (a fixed baseline —
        the "background" here degenerates to a single all-masked row,
        matching captum's fixed-baseline convention; see
        `markdowns-do-experimento/kernel-shap-implementacoes.md`, secs. 4/11),
        so the two backends' `values` are directly comparable.

        `f(X)` receives shap's whole synthetic coalition matrix
        (`(n_samples, M_words)`, 1=keep word / 0=mask to baseline) and scores
        it in chunks of `perturbations_per_eval` rows via the same
        `_score_with_mask_batch` the captum path's rescoring uses — shap
        itself does no batching of its own (`kernel-shap-implementacoes.md`
        sec. 8: batching is the caller's responsibility for this backend).
        """
        import shap

        def f(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=bool)
            chunks = []
            for i in range(0, len(X), self.perturbations_per_eval):
                chunk = X[i:i + self.perturbations_per_eval]
                chunks.append(
                    self._score_with_mask_batch(chunk, prefix_ids, suffix_ids, content_ids, feature_mask)
                )
            return np.concatenate(chunks) if chunks else np.zeros(0)

        # Single all-masked background row — the degenerate "background
        # dataset" a fixed-baseline explanation collapses to for text (no
        # meaningful "average token" to impute instead).
        background = np.zeros((1, M_words), dtype=bool)
        # link="identity" keeps the output on the same probability scale
        # captum's path uses, so values from the two backends are comparable.
        explainer = shap.KernelExplainer(f, background, link="identity")
        # l1_reg=False: never drop features via AIC-based selection — one
        # value per word, always, matching captum's behavior (which also
        # never discards a feature).
        keep_all = np.ones((1, M_words), dtype=bool)
        values = explainer.shap_values(keep_all, nsamples=n_samples, l1_reg=False, silent=not progress)
        values = np.asarray(values).reshape(-1)
        return values.tolist()

    # -- explanation primitives -------------------------------------------------

    def _explain_text(self, text: str, progress: bool = False) -> dict:
        import torch

        prefix_ids, suffix_ids, content_ids, feature_mask, M_words, content_offsets = self._tokenize_content(text)
        tokens = self._word_strings(text, content_offsets, feature_mask, M_words) if M_words else []

        full_scores = torch.softmax(
            self.model(
                input_ids=torch.cat([prefix_ids, content_ids, suffix_ids]).unsqueeze(0).to(self.device),
                attention_mask=torch.ones(1, len(prefix_ids) + len(content_ids) + len(suffix_ids), device=self.device),
            ).logits,
            dim=-1,
        )[0]
        predicted_label = self.model.config.id2label[int(torch.argmax(full_scores))]
        p_malign = float(full_scores[self.malign_indices].sum())
        full_value = p_malign

        if M_words == 0:
            base_value = p_malign
            values = []
            n_samples = None
        else:
            base_value = self._score_with_mask(
                np.zeros(M_words, dtype=bool), prefix_ids, suffix_ids, content_ids, feature_mask
            )
            n_samples = self._resolve_n_samples(M_words)
            if progress:
                print(f"[kernelshap:{self.backend}] explaining {M_words} word(s), n_samples={n_samples}")

            if self.backend == "captum":
                values = self._compute_values_captum(
                    content_ids, feature_mask, prefix_ids, suffix_ids, n_samples, progress
                )
            else:  # "shap"
                values = self._compute_values_shap(
                    content_ids, feature_mask, prefix_ids, suffix_ids, M_words, n_samples, progress
                )

        return {
            "tokens": tokens,
            "values": values,
            "base_value": base_value,
            "full_value": full_value,
            "p_malign": p_malign,
            "predicted_label": predicted_label,
            "backend": self.backend,
            # Cost telemetry (e.g. for W&B logging in main.py) — the
            # kernelshap equivalent of syntaxshap's total_forward_passes/
            # widest_dependency_level: cost here is bounded by n_samples
            # (min(2*M_words + 2048, max_n_samples), see _resolve_n_samples),
            # not by syntactic structure, but still worth tracking across a
            # sweep (n_samples grows with M_words with no fixed cap besides
            # max_n_samples).
            "n_samples": n_samples,
            "n_words": M_words,
            "_feature_mask": feature_mask,
            "_content_ids": content_ids,
            "_prefix_ids": prefix_ids,
            "_suffix_ids": suffix_ids,
        }

    def explain_row(self, text: str, progress: bool = False) -> dict:
        """Explain a single, isolated span (`target_inst`/`injected_task`)."""
        result = self._explain_text(text, progress=progress)
        for key in ("_feature_mask", "_content_ids", "_prefix_ids", "_suffix_ids"):
            result.pop(key, None)
        return result

    def explain_context(
        self, context: str, injected_task: Optional[str] = None, attack_name: Optional[str] = None,
        progress: bool = False,
    ) -> dict:
        """Explain the *whole* `context` (what `PromptGuardDefense` actually
        receives) as a single Kernel SHAP coalition game over every word.

        `attack_name` (e.g. "direct"/"ignore"/"completion"/"combined") lets
        `injected_span` localization account for the fixed prefix a heuristic
        attack prepends to `injected_task` before injecting it (see
        `_attack_prefixes()`) — without it (or for an attack with no known
        prefix, e.g. an optimization-based one), localization falls back to
        matching bare `injected_task`, same as before.
        """
        result = self._explain_text(context, progress=progress)
        for key in ("_feature_mask", "_content_ids", "_prefix_ids", "_suffix_ids"):
            result.pop(key, None)

        result["injected_span"] = None
        if injected_task and result["tokens"]:
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
                # Re-tokenize (cheap — no model forward pass) to get the
                # content-token offsets/feature ids to map the char span onto.
                _, _, _, feature_mask, _, content_offsets = self._tokenize_content(context)
                touched = [
                    fid for fid, (a, b) in zip(feature_mask.tolist(), content_offsets)
                    if b > a and a < char_end and b > char_start
                ]
                if touched:
                    result["injected_span"] = {"start_token": min(touched), "end_token": max(touched) + 1}

        return result
