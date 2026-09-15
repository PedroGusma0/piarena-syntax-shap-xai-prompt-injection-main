from ..base import BaseXAI, register_xai

# NOTE: `ClassifierSyntaxExplainer` is imported lazily, inside `_ensure_explainer`,
# not here at module scope. Its own module (classifier_explainer.py) puts the
# vendored thirdparty/ dir on sys.path and imports it (torch/transformers/
# sklearn/pandas via thirdparty/utils/_general.py) — heavy, and every
# `piarena.xai` import pulls in every registered XAI plugin's module (this one
# included) via `piarena/xai/__init__.py`'s `from . import syntaxshap`. Doing
# that import here at module scope would mean `main.py` importing `piarena.xai`
# — which it must, to make `--xai syntaxshap` selectable at all — would need
# those heavy deps installed even for runs that never pass `--xai`. Deferring
# the import keeps `--xai` genuinely opt-in (same lazy-load pattern
# PromptGuardDefense._ensure_detector() already uses for its own pipeline).


def _load_classifier_pipeline(model_name: str):
    """Load the classifier pipeline for the XAI target model, requesting all
    class scores (needed to compute P(non-benign), not just the top-1 label
    PromptGuardDefense itself uses). Deliberately its own pipeline instance,
    decoupled from PromptGuardDefense._detector — see "Saída por amostra" /
    section 4 do plano for why (no reliance on the defense's private state,
    and we control `top_k` independently of it).

    `top_k=None` is the current `transformers` API for "return every class's
    score"; older versions used `return_all_scores=True` — try both since the
    PIArena requirements.txt does not pin a transformers version (Risco #3).
    """
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        raw_pipeline = pipeline("text-classification", model=model_name, tokenizer=tokenizer, top_k=None)
    except TypeError:
        raw_pipeline = pipeline("text-classification", model=model_name, tokenizer=tokenizer, return_all_scores=True)
    return tokenizer, raw_pipeline


@register_xai
class SyntaxShapXAI(BaseXAI):
    """Explains a text-classification defense's decision using SyntaxSHAP
    (syntax-tree-restricted Shapley coalitions), adapted for classifiers.

    See plans/xai-syntaxshap-promptguard.md for the full design.
    """

    name = "syntaxshap"
    DEFAULT_CONFIG = {
        "model_name": "meta-llama/Prompt-Guard-86M",
        "algorithm": "syntax",  # "syntax" | "shap"
        "explain_targets": ["context"],  # the entry PromptGuard actually receives
        "progress": True,  # print forward-pass budget + tqdm bar per sentence/row —
                            # see coalition.py::compute_shapley_values; a single
                            # `context` can take minutes to hours (see the cost
                            # analysis in plans/xai-syntaxshap-promptguard.md),
                            # so this is on by default rather than opt-in.
        "timeout_seconds": 2400,  # 40 min/sample wall-clock cap (shared across
                                   # every explain_targets entry/sentence of one
                                   # `explain()` call — see ClassifierSyntaxExplainer.
                                   # coalition.py's coalition enumeration is exact/
                                   # uncapped (2^(widest dependency-tree level) per
                                   # sentence), and the real squad_v2 cost
                                   # distribution has a tail into days (see
                                   # "Computational cost bottleneck" in CLAUDE.md) —
                                   # without this, one bad sample can stall an
                                   # unattended multi-day sweep indefinitely.
                                   # `None` disables the timeout entirely.
    }

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._explainer = None

    def _ensure_explainer(self):
        if self._explainer is not None:
            return
        # Deferred on purpose — see the module-level NOTE above.
        from .classifier_explainer import ClassifierSyntaxExplainer

        # `thirdparty/models` cannot be imported as a normal relative
        # subpackage (`from .thirdparty.models import ...`) — its own files
        # use flat absolute imports (`from utils import ...`, `from
        # _serializable import ...`) that only resolve once `thirdparty/`
        # itself sits on sys.path (see _thirdparty_path.py). Importing
        # `.classifier_explainer` just above already called
        # `ensure_thirdparty_on_path()` at its module load time, so a flat
        # `import models` here is safe and resolves to thirdparty/models/.
        import models as _thirdparty_models  # noqa: E402  (flat, sys.path-based — see comment above)
        TransformersPipeline = _thirdparty_models.TransformersPipeline

        tokenizer, raw_pipeline = _load_classifier_pipeline(self.config["model_name"])
        wrapped = TransformersPipeline(raw_pipeline)
        self._explainer = ClassifierSyntaxExplainer(wrapped, tokenizer, algorithm=self.config["algorithm"])

    def explain(self, target_inst: str, context: str, injected_task: str, **kwargs) -> dict:
        self._ensure_explainer()
        result = {}
        targets = self.config["explain_targets"]
        progress = self.config.get("progress", True)
        timeout_seconds = self.config.get("timeout_seconds")
        attack_name = kwargs.get("attack_name")
        if "context" in targets:
            result["context"] = self._explainer.explain_context(
                context, injected_task=injected_task, attack_name=attack_name, progress=progress,
                timeout_seconds=timeout_seconds,
            )
        spans = {"target_inst": target_inst, "injected_task": injected_task}
        for key in targets:
            if key == "context":
                continue
            if key not in spans:
                raise ValueError(f"Unknown explain_targets entry {key!r} (expected 'context', 'target_inst', or 'injected_task')")
            result[key] = self._explainer.explain_row(spans[key], progress=progress, timeout_seconds=timeout_seconds)
        return result
