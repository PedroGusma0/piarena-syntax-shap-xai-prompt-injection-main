"""Which classifier an XAI method should explain for a given classification
defense. Kept free of heavy imports (no torch/transformers) so `main.py` and
`scripts/xai_metrics.py` can use it at import time.

An XAI method explains a *model*, not a defense object, so without this map
`--defense piguard --xai kernelshap` would silently keep explaining the
XAI plugin's default model (Prompt-Guard-86M) while `detect_flag` came from
PIGuard.
"""
from typing import Optional

CLASSIFIER_FOR_DEFENSE = {
    "promptguard": {"model_name": "meta-llama/Prompt-Guard-86M"},
    # Same classifier, applied window by window (`piarena.windowing`); the
    # windowed explainer is selected by `window_tokens` in `xai_config`.
    "promptguard_windowed": {"model_name": "meta-llama/Prompt-Guard-86M"},
    # PIGuard ships its own model class via `auto_map`, hence trust_remote_code
    # (same flag `PIGuardDefense` itself passes to `from_pretrained`).
    "piguard": {"model_name": "leolee99/PIGuard", "trust_remote_code": True},
}


def resolve_classifier(defense_name: str, overrides: Optional[dict] = None) -> dict:
    """XAI config for `defense_name`'s classifier, merged with the user's own
    `overrides` (which win — e.g. an explicit `model_name` in `xai_config`).

    Raises if `defense_name` is not a supported classifier defense and the
    user did not pass a `model_name`, instead of explaining some other model
    in silence."""
    overrides = dict(overrides or {})
    base = CLASSIFIER_FOR_DEFENSE.get(defense_name)
    if base is None:
        if "model_name" not in overrides:
            raise ValueError(
                f"No classifier known for defense {defense_name!r} "
                f"(supported: {sorted(CLASSIFIER_FOR_DEFENSE)}). Pass `model_name` in xai_config "
                "to explain a specific model."
            )
        return overrides
    return {**base, **overrides}
