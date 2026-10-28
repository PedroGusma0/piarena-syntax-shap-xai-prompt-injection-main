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

Target (`target=`): IG on P(malign) itself ("prob") fails on saturating
classifiers like Prompt-Guard — P goes ~0.0005 -> ~1.0 in a thin slice of the
path and a few dozen steps all land on the flat parts (observed on real
squad_v2 runs: sum(values) ~0.005 against a gap of ~1.0). The default,
"logit", runs IG on the log-odds z = logsumexp(malign logits) - logsumexp(benign
logits) (P(malign) = sigmoid(z)), which has no such plateau, and then rescales
the values by (full_value - base_value) / (z_full - z_base) so they stay on
the probability scale the rest of the pipeline (Fidelity, plots) expects.
Ranking is unchanged by the rescale; magnitudes become "share of the
probability gap".
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..kernelshap.classifier_explainer import ClassifierKernelExplainer

log = logging.getLogger(__name__)



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
        target: str = "logit",
        max_n_steps: Optional[int] = None,
        delta_tol: float = 0.05,
    ):
        if target not in ("logit", "prob"):
            raise ValueError(f"Unknown target {target!r}, expected 'logit' or 'prob'.")
        # `backend="captum"` only satisfies the parent's validation; the
        # result's own `backend` field is overwritten with "ig" below.
        super().__init__(
            model, tokenizer, baseline_token=baseline_token, max_length=max_length, backend="captum",
        )
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        self.method = method
        self.target = target
        # Adaptive refinement: when |sum(values) - (full_value - base_value)| exceeds
        # `delta_tol` * |full_value - base_value|, double the steps (up to
        # `max_n_steps`) and retry. `None` disables it (fixed `n_steps`).
        self.max_n_steps = max_n_steps
        self.delta_tol = delta_tol
        self.benign_indices = [
            i for i in range(model.config.num_labels) if i not in self.malign_indices
        ]

    def _ig_forward(self):
        """`forward(input_ids) -> Tensor[B]` of the explained scalar: P(malign)
        (`target="prob"`) or its log-odds (`target="logit"`). Unlike the
        parent's `_forward_func` this keeps autograd enabled (IG needs
        gradients)."""
        import torch

        def forward(input_ids):
            attention_mask = torch.ones_like(input_ids)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            if self.target == "prob":
                return torch.softmax(logits, dim=-1)[:, self.malign_indices].sum(dim=-1)
            return (
                torch.logsumexp(logits[:, self.malign_indices], dim=-1)
                - torch.logsumexp(logits[:, self.benign_indices], dim=-1)
            )

        return forward

    def _compute_values_ig(self, content_ids, feature_mask, prefix_ids, suffix_ids, M_words, p_full, p_base, n_steps, progress):
        """Returns `(values: list[float] per word, convergence_delta: float)`,
        both on the probability scale (values sum to ~ full_value - base_value;
        delta = sum(values) - (full_value - base_value))."""
        import torch
        from captum.attr import LayerIntegratedGradients

        lig = LayerIntegratedGradients(self._ig_forward(), self.model.get_input_embeddings())
        full_ids = torch.cat([prefix_ids, content_ids, suffix_ids]).unsqueeze(0).to(self.device)
        baseline_content = torch.full_like(content_ids, self.baseline_token_id)
        baseline_ids = torch.cat([prefix_ids, baseline_content, suffix_ids]).unsqueeze(0).to(self.device)

        with torch.enable_grad():
            attributions, _ = lig.attribute(
                inputs=full_ids,
                baselines=baseline_ids,
                n_steps=n_steps,
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

        scale = 1.0
        if self.target == "logit":
            # Move from log-odds units to the probability gap (see module docstring).
            with torch.no_grad():
                forward = self._ig_forward()
                z_full = float(forward(full_ids)[0])
                z_base = float(forward(baseline_ids)[0])
            dz = z_full - z_base
            if abs(dz) > 1e-6:
                scale = (p_full - p_base) / dz
        word_attr = word_attr * scale
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        delta = float(word_attr.sum()) - (p_full - p_base)
        return word_attr.tolist(), delta

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
        converged = True
        n_steps_used = self.n_steps
        if M_words == 0:
            base_value = p_malign
            values = []
        else:
            base_value = self._score_with_mask(
                np.zeros(M_words, dtype=bool), prefix_ids, suffix_ids, content_ids, feature_mask
            )
            if progress:
                print(f"[ig] explaining {M_words} word(s), n_steps={self.n_steps}" + (f" (adaptive up to {self.max_n_steps})" if self.max_n_steps else ""))
            gap = abs(full_value - base_value)
            while True:
                values, delta = self._compute_values_ig(
                    content_ids, feature_mask, prefix_ids, suffix_ids, M_words,
                    full_value, base_value, n_steps_used, progress,
                )
                converged = gap == 0 or abs(delta) <= self.delta_tol * gap
                if converged or self.max_n_steps is None or n_steps_used >= self.max_n_steps:
                    break
                n_steps_used = min(n_steps_used * 2, self.max_n_steps)
            if not converged:
                log.warning(
                    "IG did not converge after %d steps: delta %.4g vs full_value - base_value = %.4g; "
                    "sum(values) will not match the output value. Raise max_n_steps.",
                    n_steps_used, delta, full_value - base_value,
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
            "ig_target": self.target,
            "ig_converged": converged,
            "n_steps": n_steps_used,
            "n_words": M_words,
            "_feature_mask": feature_mask,
            "_content_ids": content_ids,
            "_prefix_ids": prefix_ids,
            "_suffix_ids": suffix_ids,
            "_content_offsets": content_offsets,
        }
