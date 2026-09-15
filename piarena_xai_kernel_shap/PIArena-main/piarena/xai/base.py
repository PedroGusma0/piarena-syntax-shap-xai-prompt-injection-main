from abc import ABC, abstractmethod

from ..registry import XAI_REGISTRY

register_xai = XAI_REGISTRY.register


class BaseXAI(ABC):
    """Base class for all explainability (XAI) methods in PIArena.

    Third plugin type alongside BaseAttack/BaseDefense, added to explain
    defense decisions (starting with `promptguard`). See
    `piarena/xai/kernelshap/` for a worked example and
    `plans/xai-kernelshap-promptguard.md` for the design rationale.
    """

    name: str
    DEFAULT_CONFIG = {}

    def __init__(self, config: dict = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    @abstractmethod
    def explain(self, target_inst: str, context: str, injected_task: str, **kwargs) -> dict:
        """Explain a defense's decision on this sample.

        Returns a dict keyed by the span(s) explained (e.g. "context",
        "injected_task"), each value itself a dict of XAI-method-specific
        fields (tokens, per-token importance values, etc.)."""
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"
