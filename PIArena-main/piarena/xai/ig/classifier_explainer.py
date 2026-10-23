"""Integrated Gradients explainer for a text CLASSIFIER (a single forward pass).

Subclasses `ClassifierKernelExplainer` and reuses everything except the
attribution itself: fast-tokenizer word grouping (`_tokenize_content`/
`_word_strings`), the binary target P(malign) (`malign_indices`), the baseline
token, `make_rescorer` (so `piarena/xai/metrics.py` and `scripts/xai_metrics.py`
re-score IG results with the exact same masking), and `explain_context`'s
`injected_span` localization. Only `_explain_text` is overridden, keeping the
same output contract (`tokens`/`values`/`base_value`/`full_value`/`p_malign`/...
plus IG-specific `ig_convergence_delta`/`n_steps`).

How it works: token ids are discrete, so IG is taken on the output of the
model's input-embedding layer (`LayerIntegratedGradients`). The path goes from
a baseline sequence (same prefix/suffix special tokens, every content token
replaced by `baseline_token_id`) to the real sequence. Per-token attributions
are summed over the embedding dimension, then summed over each word's subtokens
(`feature_mask`) so `values` is word-aligned with `tokens`, like KernelSHAP's.
Completeness holds only up to the convergence delta: sum(values) ~=
full_value - base_value (unlike Shapley values, which are exactly additive).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..kernelshap.classifier_explainer import ClassifierKernelExplainer

log = logging.getLogger(__name__)

# |delta| above this (relative to |full_value - base_value|) triggers a warning.
_DELTA_WARN_REL = 0.05


class ClassifierIGExplainer(ClassifierKernelExplainer):
    def __init__(
        self,
        model,
        tokenizer,
        baseline_token: str = "mask",
        max_length: Optional[int] = None,
        n_steps: int = 50,
        internal_batch_size: Optional[int] = 4,
        method: str = "gausslegendre",
    ):
        # `backend="captum"` only satisfies the parent's validation; the
        # result's own `backend` field is overwritten with "ig" below.
        super().__init__(
            model, tokenizer, baseline_token=baseline_token, max_length=max_length, backend="captum",
        )
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        self.method = method

    def _ig_forward(self):
        """`forward(input_ids) -> Tensor[B]` of P(malign). Unlike the parent's
        `_forward_func` this keeps autograd enabled (IG needs gradients)."""
        import torch

        def forward(input_ids):
            attention_mask = torch.ones_like(input_ids)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            return torch.softmax(logits, dim=-1)[:, self.malign_indices].sum(dim=-1)

        return forward

    def _compute_values_ig(self, content_ids, feature_mask, prefix_ids, suffix_ids, M_words, progress):
        """Returns `(values: list[float] per word, convergence_delta: float)`."""
        import torch
        from captum.attr import LayerIntegratedGradients

        lig = LayerIntegratedGradients(self._ig_forward(), self.model.get_input_embeddings())
        full_ids = torch.cat([prefix_ids, content_ids, suffix_ids]).unsqueeze(0).to(self.device)
        baseline_content = torch.full_like(content_ids, self.baseline_token_id)
        baseline_ids = torch.cat([prefix_ids, baseline_content, suffix_ids]).unsqueeze(0).to(self.device)

        with torch.enable_grad():
            attributions, delta = lig.attribute(
                inputs=full_ids,
                baselines=baseline_ids,
                n_steps=self.n_steps,
                method=self.method,
                internal_batch_size=self.internal_batch_size,
                return_convergence_delta=True,
            )
        # (1, L, D) -> (L,): sum over the embedding dimension, keep the content slice.
        token_attr = attributions.sum(dim=-1).squeeze(0).detach()
        start = len(prefix_ids)
        content_attr = token_attr[start:start + len(content_ids)].cpu()
        word_attr = torch.zeros(M_words, dtype=content_attr.dtype)
        word_attr.index_add_(0, feature_mask.cpu(), content_attr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return word_attr.tolist(), float(delta.reshape(-1)[0].detach().cpu())

    def _explain_text(self, text: str, progress: bool = False) -> dict:
        import torch

        prefix_ids, suffix_ids, content_ids, feature_mask, M_words, content_offsets = self._tokenize_content(text)
        tokens = self._word_strings(text, content_offsets, feature_mask, M_words) if M_words else []

        with torch.no_grad():
            full_scores = torch.softmax(
                self.model(
                    input_ids=torch.cat([prefix_ids, content_ids, suffix_ids]).unsqueeze(0).to(self.device),
                    attention_mask=torch.ones(
                        1, len(prefix_ids) + len(content_ids) + len(suffix_ids), device=self.device,
                    ),
                ).logits,
                dim=-1,
            )[0]
        predicted_label = self.model.config.id2label[int(torch.argmax(full_scores))]
        p_malign = float(full_scores[self.malign_indices].sum())
        full_value = p_malign

        delta = None
        if M_words == 0:
            base_value = p_malign
            values = []
        else:
            base_value = self._score_with_mask(
                np.zeros(M_words, dtype=bool), prefix_ids, suffix_ids, content_ids, feature_mask
            )
            if progress:
                print(f"[ig] explaining {M_words} word(s), n_steps={self.n_steps}")
            values, delta = self._compute_values_ig(
                content_ids, feature_mask, prefix_ids, suffix_ids, M_words, progress
            )
            gap = abs(full_value - base_value)
            if gap > 0 and abs(delta) > _DELTA_WARN_REL * gap:
                log.warning(
                    "IG convergence delta %.4g is large relative to full_value - base_value = %.4g; "
                    "sum(values) will not match the output value. Increase n_steps.",
                    delta, full_value - base_value,
                )

        return {
            "tokens": tokens,
            "values": values,
            "base_value": base_value,
            "full_value": full_value,
            "p_malign": p_malign,
            "p_benign": 1.0 - p_malign,
            "predicted_label": predicted_label,
            "backend": "ig",
            "ig_convergence_delta": delta,
            "n_steps": self.n_steps,
            "n_words": M_words,
            "_feature_mask": feature_mask,
            "_content_ids": content_ids,
            "_prefix_ids": prefix_ids,
            "_suffix_ids": suffix_ids,
            "_content_offsets": content_offsets,
        }
