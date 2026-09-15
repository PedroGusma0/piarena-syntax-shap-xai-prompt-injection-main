from .base import BaseXAI
from ..registry import XAI_REGISTRY

# Import all XAI plugin modules to trigger @register_xai decorators
from . import kernelshap  # noqa: F401

# Registry is auto-populated by @register_xai decorators
XAI_CLASSES = XAI_REGISTRY


def get_xai(name: str, config: dict = None) -> BaseXAI:
    """Factory: instantiate an XAI method by name."""
    cls = XAI_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown xai method: {name}. Available: {sorted(XAI_REGISTRY)}")
    return cls(config=config)
