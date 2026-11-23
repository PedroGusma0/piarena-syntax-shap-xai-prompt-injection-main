"""`--xai blackbox` — explains a detection defense that is only reachable as a
boolean function `text -> detect_flag` (DataSentinel today) by perturbing its
input and calling it, with the defense's real decision as the explained
target. Counterpart of `--xai kernelshap`/`ig`/`syntaxshap`, which need a
classifier's logits; nothing in those plugins is touched. See
`plans/xai-blackbox-datasentinel.md` for the design rationale and
`piarena/xai/blackbox/explainer.py` for the actual explanation logic.

Heavy imports (torch/transformers/peft) live in the detector adapter and are
only reached on the first `explain()` call, so `import piarena.xai` stays cheap.
"""
from ..base import BaseXAI, register_xai
from ..detectors import load_detector, resolve_detector


@register_xai
class BlackBoxXAI(BaseXAI):
    name = "blackbox"
    DEFAULT_CONFIG = {
        # Filled per defense by `main.py` via `resolve_config` (-> `piarena.xai.detectors.
        # resolve_detector`): "adapter" ('module:Class'), "model_name" (display name).
        "adapter": None,
        "adapter_config": None,
        "model_name": None,
        # Which samples get explained: "auto" (blocked ones; every sample with
        # `--attack none`) | "blocked" | "all". See `BaseXAI.should_explain`.
        "explain_scope": "auto",
        "explain_targets": ["context"],
        # Ordered; the first one fills `values`, the rest go to `extra_methods`.
        "methods": ["ddmin", "sentence_shapley"],
        # What the detector is given when nothing is kept: "empty" (empty text) |
        # "neutral" (`neutral_text`) | "clean_context" (the context minus the
        # injected span; players are then only the injected span's words).
        "baseline": "empty",
        "neutral_text": "The weather is nice today.",
        # Extra detector runs on the full context, to flag non-deterministic generation.
        "determinism_checks": 3,
        # Exact Shapley over sentences needs 2^M detector calls.
        "max_exact_sentences": 10,
        "max_detector_calls": 2000,
        "record_trace": False,
        "progress": True,
    }

    @classmethod
    def resolve_config(cls, defense_name: str, config: dict = None) -> dict:
        return resolve_detector(defense_name, config)

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._explainer = None

    def _ensure_explainer(self):
        if self._explainer is not None:
            return
        from .explainer import BlackBoxExplainer

        self._explainer = BlackBoxExplainer(
            load_detector(self.config),
            methods=self.config["methods"],
            baseline=self.config["baseline"],
            neutral_text=self.config["neutral_text"],
            determinism_checks=self.config["determinism_checks"],
            max_exact_sentences=self.config["max_exact_sentences"],
            max_detector_calls=self.config["max_detector_calls"],
            record_trace=self.config["record_trace"],
        )

    def explain(self, target_inst: str, context: str, injected_task: str, **kwargs) -> dict:
        self._ensure_explainer()
        result = {}
        targets = self.config["explain_targets"]
        progress = self.config.get("progress", True)
        attack_name = kwargs.get("attack_name")

        if "context" in targets:
            result["context"] = self._explainer.explain_context(
                context, injected_task=injected_task, attack_name=attack_name, progress=progress,
            )

        spans = {"target_inst": target_inst, "injected_task": injected_task}
        for key in targets:
            if key == "context":
                continue
            if key not in spans:
                raise ValueError(f"Unknown explain_targets entry {key!r} (expected 'context', 'target_inst', or 'injected_task')")
            result[key] = self._explainer.explain_row(spans[key], progress=progress)

        return result
