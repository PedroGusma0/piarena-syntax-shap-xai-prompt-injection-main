"""`--xai ig` — explains a classifier-based defense's decision using
Integrated Gradients (`captum.attr.LayerIntegratedGradients` over the input
embeddings). White-box counterpart of `--xai kernelshap`/`--xai syntaxshap`:
needs the classifier's weights, but its cost is a fixed `n_steps` forward/
backward passes per text regardless of the text's syntactic structure. See
`plans/xai-ig-promptguard.md` for the design rationale and
`piarena/xai/ig/classifier_explainer.py` for the actual explanation logic.
"""
from ..base import BaseXAI, register_xai


@register_xai
class IntegratedGradientsXAI(BaseXAI):
    name = "ig"
    DEFAULT_CONFIG = {
        # Overridden per defense by `main.py` via `piarena.xai.classifiers.resolve_classifier`
        # (so `--defense piguard` explains PIGuard, not this default).
        "model_name": "meta-llama/Prompt-Guard-86M",
        "trust_remote_code": False,
        # Which samples get explained: "auto" (blocked ones; every sample with
        # `--attack none`) | "blocked" | "all". See `BaseXAI.should_explain`.
        "explain_scope": "auto",
        "explain_targets": ["context"],
        # Starting number of interpolation steps along the baseline -> input
        # path. If the convergence error exceeds `delta_tol` (relative to
        # full_value - base_value) the steps are doubled and IG re-run, up to
        # `max_n_steps` (None = fixed n_steps). Real squad_v2 runs with 50 fixed
        # steps on P(malign) gave sum(values) ~ 0.005 against a gap of ~1.0.
        "n_steps": 100,
        "max_n_steps": 800,
        "delta_tol": 0.05,
        # How many interpolation steps captum runs per forward/backward call
        # (memory knob; does not change the result).
        "internal_batch_size": 4,
        # captum's integral approximation: "gausslegendre" | "riemann_right" |
        # "riemann_left" | "riemann_middle" | "riemann_trapezoid".
        "method": "gausslegendre",
        # Scalar IG explains: "logit" (log-odds of P(malign), rescaled to the
        # probability gap — default, robust to saturation) | "prob" (P(malign)
        # itself; collapses on saturating classifiers, see classifier_explainer.py).
        "target": "logit",
        "baseline_token": "mask",  # "mask" | "pad" | "unk"
        "max_length": None,  # None -> no truncation; matches PromptGuardDefense's own scoring
        "progress": True,
    }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._explainer = None

    def _ensure_explainer(self):
        if self._explainer is not None:
            return
        # Heavy imports (torch/transformers/captum) stay inside these calls.
        from ..kernelshap.xai_kernelshap import _load_classifier
        from .classifier_explainer import ClassifierIGExplainer

        tokenizer, model = _load_classifier(
            self.config["model_name"], self.config["max_length"], self.config["trust_remote_code"],
        )
        self._explainer = ClassifierIGExplainer(
            model,
            tokenizer,
            baseline_token=self.config["baseline_token"],
            max_length=self.config["max_length"],
            n_steps=self.config["n_steps"],
            internal_batch_size=self.config["internal_batch_size"],
            method=self.config["method"],
            target=self.config["target"],
            max_n_steps=self.config["max_n_steps"],
            delta_tol=self.config["delta_tol"],
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
