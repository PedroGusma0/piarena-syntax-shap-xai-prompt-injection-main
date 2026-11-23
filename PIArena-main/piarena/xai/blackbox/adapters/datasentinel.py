"""`BooleanDetector` adapter for the DataSentinel defense.

Wraps `DataSentinelDetector.detect_verbose`, so the decision is produced by
exactly the code path `DataSentinelDefense.execute` uses (same
`preprocessing()`, same known-answer prompt, same echo-stripping) and the text
the model generated comes back with it. Heavy imports (torch/transformers/peft)
happen lazily on the first `detect()` call.
"""
from __future__ import annotations

from ...detectors import Verdict


class DataSentinelAdapter:
    name = "datasentinel"

    def __init__(self, config: dict | None = None, detector=None):
        """`config`: DataSentinel model config (defaults to
        `DataSentinelDefense.DEFAULT_CONFIG`). `detector`: an already-built
        object with a `detect_verbose(text)` method (tests; or to reuse the
        defense's own instance instead of loading a second copy)."""
        self._config = config
        self._detector = detector

    def _ensure_detector(self):
        if self._detector is not None:
            return
        from piarena.defenses.datasentinel.defense_datasentinel import DataSentinelDefense
        from piarena.defenses.datasentinel.OpenPromptInjection.apps import DataSentinelDetector

        self._detector = DataSentinelDetector(self._config or DataSentinelDefense.DEFAULT_CONFIG)

    def detect(self, text: str) -> Verdict:
        self._ensure_detector()
        verdict = self._detector.detect_verbose(text)
        return Verdict(flag=bool(verdict["flag"]), raw_output=verdict["raw_output"])
