"""`--xai kernelshap` — explains a classifier-based defense's decision using
Kernel SHAP, computed via either `captum.attr.KernelShap` or the official
`shap.KernelExplainer` (see `backend` in `DEFAULT_CONFIG` below — the two
coexist, pick one via config). See `plans/xai-kernelshap-promptguard.md` for
the full design rationale and `piarena/xai/kernelshap/classifier_explainer.py`
for the actual explanation logic.
"""
from ..base import BaseXAI, register_xai


def _load_classifier(model_name: str, max_length=None):
    """Loads the classifier directly (no HF `pipeline()` wrapper) — Kernel
    SHAP via captum needs tensor-in/tensor-out access, not a text-in/text-out
    convenience API. Heavy imports (torch/transformers) are lazy, only
    reached when an XAI method is actually selected (see `_ensure_explainer`)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model


@register_xai
class KernelShapXAI(BaseXAI):
    name = "kernelshap"
    DEFAULT_CONFIG = {
        "model_name": "meta-llama/Prompt-Guard-86M",
        "explain_targets": ["context"],
        # Never left at captum's own default (25) — see Risco #3 do plano.
        # "auto" resolves per-call to min(2*M + 2048, max_n_samples).
        "n_samples": "auto",
        "max_n_samples": 20000,
        "perturbations_per_eval": 8,
        "baseline_token": "mask",  # "mask" | "pad" | "unk" — see Risco #4 do plano
        "max_length": None,  # None -> tokenizer.model_max_length (guarded, see Risco #5)
        "progress": True,
        # "captum" (default, captum.attr.KernelShap) | "shap" (official
        # shap.KernelExplainer). The two coexist as alternative computation
        # backends behind the identical explain_row/explain_context contract
        # — pick one via xai_config, no code change needed. See
        # plans/xai-kernelshap-promptguard.md, "Duas implementações".
        "backend": "captum",
    }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._explainer = None

    def _ensure_explainer(self):
        if self._explainer is not None:
            return
        from .classifier_explainer import ClassifierKernelExplainer

        tokenizer, model = _load_classifier(self.config["model_name"], self.config["max_length"])
        self._explainer = ClassifierKernelExplainer(
            model,
            tokenizer,
            baseline_token=self.config["baseline_token"],
            n_samples=self.config["n_samples"],
            max_n_samples=self.config["max_n_samples"],
            perturbations_per_eval=self.config["perturbations_per_eval"],
            max_length=self.config["max_length"],
            backend=self.config["backend"],
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
