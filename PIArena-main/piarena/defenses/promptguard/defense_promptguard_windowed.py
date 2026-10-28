from typing import List

from ..base import register_defense
from ...windowing import split_into_windows, window_texts
from .defense_promptguard import PromptGuardDefense


@register_defense
class PromptGuardWindowedDefense(PromptGuardDefense):
    """Prompt-Guard-86M applied window by window.

    Prompt-Guard has a 512-token context window; Meta recommends "splitting
    longer inputs into segments and scanning each in parallel to detect the
    presence of violations anywhere in longer prompts"
    (https://huggingface.co/meta-llama/Prompt-Guard-86M). The original
    `promptguard` defense feeds the whole context in one pass instead (no
    truncation, no split), which is both out of the model's trained range and
    quadratic in memory on the `*_long` datasets.

    Here the context is split into `window_tokens`-token windows
    (`piarena.windowing.split_into_windows`, shared with the windowed Kernel
    SHAP explainer) and each window is scored on its own. The prompt is
    blocked if ANY window has P(malign) >= `block_threshold`, where
    P(malign) = P(INJECTION) + P(JAILBREAK) = 1 - P(BENIGN) — the same
    quantity `--xai kernelshap` explains.

    Besides `detect_flag`, `execute` returns the per-window scores so plots
    (`scripts/xai_window_anomaly.py`) work for every sample, blocked or not,
    without running any XAI.
    """

    name = "promptguard_windowed"
    DEFAULT_CONFIG = {
        "window_tokens": 512,
        "block_threshold": 0.5,
        "window_batch_size": 16,
    }

    def _score_windows(self, texts: List[str]) -> List[float]:
        """P(malign) for each text (sum of every non-benign class probability)."""
        preds = self._detector(
            texts,
            top_k=None,
            truncation=True,
            max_length=512,  # safety net: a window re-tokenized by the pipeline can't exceed the model window
            batch_size=self.config["window_batch_size"],
        )
        scores = []
        for pred in preds:
            if isinstance(pred, dict):  # single label returned
                pred = [pred]
            scores.append(float(sum(p["score"] for p in pred if "benign" not in p["label"].lower())))
        return scores

    def _build_result(self, windows, p_malign: List[float]) -> dict:
        thr = self.config["block_threshold"]
        blocking = [i for i, p in enumerate(p_malign) if p >= thr]
        return {
            "detect_flag": bool(blocking),
            "window_p_malign": p_malign,
            "window_spans": [[w["char_start"], w["char_end"]] for w in windows],
            "n_windows": len(windows),
            "max_p_malign": max(p_malign) if p_malign else 0.0,
            "argmax_window": max(range(len(p_malign)), key=p_malign.__getitem__) if p_malign else None,
            "blocking_windows": blocking,
            "block_threshold": thr,
            "window_tokens": self.config["window_tokens"],
        }

    def execute(self, target_inst, context):
        return self.execute_batch([target_inst], [context])[0]

    def execute_batch(self, target_insts: List[str], contexts: List[str]) -> List[dict]:
        self._validate_batch_inputs(target_insts, contexts)
        self._ensure_detector()
        raw_contexts = [context if context is not None else "" for context in contexts]
        tokenizer = self._detector.tokenizer

        per_context_windows = [
            split_into_windows(tokenizer, c, self.config["window_tokens"]) for c in raw_contexts
        ]
        flat_texts = []
        for c, windows in zip(raw_contexts, per_context_windows):
            flat_texts.extend(window_texts(c, windows))
        flat_scores = self._score_windows(flat_texts)

        results, pos = [], 0
        for windows in per_context_windows:
            results.append(self._build_result(windows, flat_scores[pos:pos + len(windows)]))
            pos += len(windows)
        return results
