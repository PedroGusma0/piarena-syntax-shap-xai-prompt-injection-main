"""Kernel SHAP (`captum.attr.KernelShap`) over a boolean detector game.

Black-box formulation: one feature per player (word), `inputs = ones(1, M)`,
`baselines = zeros(1, M)`, so every perturbed row captum hands to the forward
function is a 0/1 coalition vector (1 = keep the word, 0 = delete it). The
forward function maps each row to the detector's decision (0.0/1.0) through
`score_rows`, supplied by `BlackBoxExplainer`; no tokenizer, no logits.

torch/captum are imported lazily, only when a Kernel SHAP game is actually
solved, so `import piarena.xai` stays cheap. `BlackBoxExplainer` takes any
callable with this module's `captum_kernelshap` signature as its `solver`, which
is how the model-free unit tests substitute an exact enumeration.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# score_rows((B, M) bool array) -> sequence of B floats in {0.0, 1.0}
ScoreRows = Callable[[np.ndarray], Sequence[float]]


def captum_kernelshap(
    score_rows: ScoreRows,
    n_players: int,
    n_samples: int,
    batch_size: int = 8,
    seed: int = 0,
    progress: bool = False,
) -> list[float]:
    import torch
    from captum.attr import KernelShap

    def forward(rows):  # (B, M) float tensor of 0.0/1.0
        keep = (rows.detach().cpu().numpy() > 0.5)
        return torch.tensor(np.asarray(score_rows(keep), dtype=np.float32))

    torch.manual_seed(seed)
    attributions = KernelShap(forward).attribute(
        inputs=torch.ones(1, n_players),
        baselines=torch.zeros(1, n_players),
        n_samples=n_samples,
        perturbations_per_eval=batch_size,
        return_input_shape=False,
        show_progress=progress,
    )
    return attributions.detach().cpu().numpy().reshape(-1).tolist()
