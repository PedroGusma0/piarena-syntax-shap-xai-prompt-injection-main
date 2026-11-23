"""Which boolean detector an XAI method should treat as a black box for a given
defense. Counterpart of `piarena/xai/classifiers.py` for defenses that are NOT
`AutoModelForSequenceClassification` classifiers (e.g. DataSentinel, a
generative known-answer detector): all the explainer needs from them is
`detect(text) -> Verdict`.

Kept free of heavy imports (no torch/transformers) so `main.py` and
`scripts/xai_metrics.py` can use it at import time; the adapter module that
actually loads a model is imported lazily by `load_detector`.

Adding a new boolean defense = one adapter class (see
`piarena/xai/blackbox/adapters/datasentinel.py`) + one entry in
`DETECTOR_FOR_DEFENSE`. Nothing in `classifiers.py`, `kernelshap/`, `ig/` or
`syntaxshap/` is involved.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class Verdict:
    """One detector decision on one text.

    `flag` is the defense's own decision (True = blocked / injection
    detected). `raw_output` is whatever the detector produced on the way there
    (for DataSentinel: the text the model generated), or None for detectors
    that don't produce one."""
    flag: bool
    raw_output: Optional[str] = None


@runtime_checkable
class BooleanDetector(Protocol):
    """Black-box view of a detection defense: text in, decision out."""

    name: str

    def detect(self, text: str) -> Verdict:
        ...


DETECTOR_FOR_DEFENSE = {
    "datasentinel": {
        # "module:Class" — imported lazily by `load_detector`.
        "adapter": "piarena.xai.blackbox.adapters.datasentinel:DataSentinelAdapter",
        "model_name": "DataSentinel (mistralai/Mistral-7B-v0.1 + sleeepeer/DataSentinel)",
    },
}


def resolve_detector(defense_name: str, overrides: Optional[dict] = None) -> dict:
    """XAI config for `defense_name`'s detector, merged with the user's own
    `overrides` (which win — e.g. an explicit `adapter` in `xai_config`).

    Raises if `defense_name` has no registered detector and the user did not
    pass an `adapter`, instead of explaining some other model in silence."""
    overrides = dict(overrides or {})
    base = DETECTOR_FOR_DEFENSE.get(defense_name)
    if base is None:
        if "adapter" not in overrides:
            raise ValueError(
                f"No boolean detector known for defense {defense_name!r} "
                f"(supported: {sorted(DETECTOR_FOR_DEFENSE)}). Pass `adapter` ('module:Class') "
                "in xai_config to explain a specific detector."
            )
        return {"model_name": overrides["adapter"], **overrides}
    return {**base, **overrides}


def load_detector(config: dict) -> BooleanDetector:
    """Instantiate the adapter named by `config["adapter"]` ('module:Class').
    `config.get("adapter_config")` is passed to its constructor."""
    spec = config.get("adapter")
    if not spec or ":" not in spec:
        raise ValueError(f"xai_config['adapter'] must be 'module:Class', got {spec!r}")
    module_name, class_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(config.get("adapter_config"))
