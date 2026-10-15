from abc import ABC, abstractmethod

from ..registry import XAI_REGISTRY

# Expose decorator at module level for convenience (same pattern as
# piarena/attacks/base.py:register_attack and piarena/defenses/base.py:register_defense)
register_xai = XAI_REGISTRY.register


class BaseXAI(ABC):
    """Base class for all explainability (XAI) methods in PIArena.

    Third plugin type alongside BaseAttack/BaseDefense, added to explain
    defense decisions (starting with `promptguard`). See
    `piarena/xai/syntaxshap/` and `piarena/xai/kernelshap/` for two worked
    examples of the same contract, and `plans/xai-syntaxshap-promptguard.md`/
    `plans/xai-kernelshap-promptguard.md` for their respective design
    rationale.
    """

    name: str
    DEFAULT_CONFIG = {}

    def __init__(self, config: dict = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    EXPLAIN_SCOPES = ("auto", "blocked", "all")

    def should_explain(self, detect_flag, attack_name=None) -> bool:
        """Whether `main.py` should run `explain` on a sample, per the
        `explain_scope` config key:

        - "auto" (default): only blocked samples (`detect_flag=True`), except
          with `--attack none` where every sample is benign and all are explained;
        - "blocked": only blocked samples, always;
        - "all": every sample, blocked or not.
        """
        scope = self.config.get("explain_scope", "auto")
        if scope not in self.EXPLAIN_SCOPES:
            raise ValueError(f"explain_scope must be one of {self.EXPLAIN_SCOPES}, got {scope!r}")
        if scope == "all":
            return True
        if scope == "auto" and attack_name == "none":
            return True
        return bool(detect_flag)

    @abstractmethod
    def explain(self, target_inst: str, context: str, injected_task: str, **kwargs) -> dict:
        """Explain a defense's decision on this sample. Return a dict keyed
        by the span(s) explained (e.g. "context", "injected_task"), each
        value itself a dict of XAI-method-specific fields (tokens, per-token
        importance values, etc.)."""
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"
