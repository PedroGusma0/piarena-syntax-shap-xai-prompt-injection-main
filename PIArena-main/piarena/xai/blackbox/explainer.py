"""`BlackBoxExplainer`: explains a boolean detector (`text -> Verdict`) with
Kernel SHAP over its real decision, by perturbing the input and calling the
detector. Works on any `piarena.xai.detectors.BooleanDetector`; nothing here
knows about DataSentinel, classifiers, tokenizers or logits.

Per sample (`explain_context`):
  1. preconditions — `full_value = detect(context)`, determinism re-runs, and
     the baseline check `detect(baseline_text)` (a sample whose baseline is
     already flagged has no benign reference to explain against and is marked
     `baseline_valid=False`, not explained);
  2. Kernel SHAP (`captum.attr.KernelShap`, see `kernelshap.py`) with every word
     a player: a coalition is the text made of the kept words, the detector's
     0/1 decision is the game's value, and the baseline is the text with every
     word deleted.

captum's KernelShap only approximates efficiency (in synthetic boolean games the
sum came out ~exact, but nothing guarantees it), so `sum_values` and
`efficiency_error = |sum(values) - (full_value - base_value)|` are recorded
instead of silently renormalizing. The detector's real output (the text it
generated) at the key points is returned in `detector_outputs`.

See plans/xai-blackbox-datasentinel.md and plans/xai-blackbox-kernelshap.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from ..detectors import BooleanDetector, Verdict
from .perturb import Segmentation, segment

_BASELINES = ("empty", "neutral", "clean_context")
_BACKENDS = ("captum",)
_MIN_SAMPLES_PER_PLAYER = 2  # fewer coalitions than 2*M cannot pin down M values


class BudgetExceeded(RuntimeError):
    """The per-sample detector-call budget is spent."""


@dataclass
class _Game:
    """The perturbation game for one context: `fixed` words are always kept,
    `players` (word indices) are the features Kernel SHAP attributes to."""
    seg: Segmentation
    fixed: list
    players: list
    empty_text: str

    def text(self, kept_players: Iterable[int]) -> str:
        mask = np.zeros(self.seg.n_words, dtype=bool)
        if self.fixed:
            mask[self.fixed] = True
        kept = list(kept_players)
        if kept:
            mask[kept] = True
        return self.seg.build(mask, self.empty_text)


class BlackBoxExplainer:
    def __init__(
        self,
        detector: BooleanDetector,
        baseline: str = "empty",
        neutral_text: str = "The weather is nice today.",
        determinism_checks: int = 3,
        backend: str = "captum",
        n_samples="auto",
        batch_size: int = 8,
        seed: int = 0,
        max_detector_calls: int = 3500,
        solver: Optional[Callable] = None,
    ):
        if baseline not in _BASELINES:
            raise ValueError(f"baseline must be one of {_BASELINES}, got {baseline!r}")
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS} (captum only), got {backend!r}")
        self.detector = detector
        self.baseline = baseline
        self.neutral_text = neutral_text
        self.determinism_checks = int(determinism_checks)
        self.backend = backend
        self.n_samples_config = n_samples
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.max_detector_calls = int(max_detector_calls)
        self._solver = solver
        self._cache: dict[str, Verdict] = {}
        self._calls = 0
        self._budget_active = False

    # -- detector access ----------------------------------------------------

    def _call(self, text: str, use_cache: bool = True) -> Verdict:
        if use_cache and text in self._cache:
            return self._cache[text]
        if self._budget_active and self._calls >= self.max_detector_calls:
            raise BudgetExceeded(f"detector-call budget ({self.max_detector_calls}) spent")
        verdict = self.detector.detect(text)
        self._calls += 1
        self._cache[text] = verdict
        return verdict

    @staticmethod
    def _output(verdict: Verdict) -> dict:
        return {"raw_output": verdict.raw_output, "flag": bool(verdict.flag)}

    # -- game construction --------------------------------------------------

    def _build_game(self, seg: Segmentation, span: Optional[dict]) -> _Game:
        if self.baseline == "clean_context":
            if span is None:
                raise ValueError(
                    "baseline='clean_context' needs the injected span (an `injected_task` that can be "
                    "located in the context); use baseline='empty' or 'neutral' otherwise."
                )
            players = list(range(span["start_token"], span["end_token"]))
            in_span = set(players)
            fixed = [i for i in range(seg.n_words) if i not in in_span]
            return _Game(seg, fixed, players, "")
        empty_text = self.neutral_text if self.baseline == "neutral" else ""
        return _Game(seg, [], list(range(seg.n_words)), empty_text)

    def _resolve_n_samples(self, n_players: int) -> int:
        if self.n_samples_config == "auto":
            # Boolean game, 7B generation per coalition: far fewer than the
            # `2*M + 2048` the classifier explainers use. A starting point to be
            # calibrated against exact Shapley on small games (see the plan).
            wanted = min(max(16 * n_players, 400), self.max_detector_calls)
        else:
            wanted = int(self.n_samples_config)
        return min(wanted, max(self.max_detector_calls - self._calls - 10, 0))

    def _solve(self, score_rows, n_players: int, n_samples: int, progress: bool) -> list:
        solver = self._solver
        if solver is None:
            from .kernelshap import captum_kernelshap as solver
        return solver(score_rows, n_players, n_samples, self.batch_size, self.seed, progress)

    # -- public API ---------------------------------------------------------

    def explain_context(
        self, context: str, injected_task: Optional[str] = None, attack_name: Optional[str] = None,
        progress: bool = False,
    ) -> dict:
        self._cache = {}
        self._calls = 0
        self._budget_active = False

        seg = segment(context)
        span = self._locate_span(seg, context, injected_task, attack_name)
        n = seg.n_words
        result = {
            "tokens": list(seg.words), "values": [0.0] * n, "base_value": 0.0, "full_value": 0.0,
            "p_malign": 0.0, "p_benign": 1.0, "predicted_label": None,
            "method": "kernelshap", "backend": self.backend, "granularity": "word",
            "explained": False, "skip_reason": None, "injected_span": span,
            "baseline": self.baseline, "baseline_value": None, "baseline_valid": None,
            "empty_text": "", "fixed_words": [], "deterministic": None,
            "n_words": n, "n_players": None, "n_samples": None, "seed": self.seed,
            "sum_values": None, "efficiency_error": None,
            "n_detector_calls": 0, "detector_outputs": {},
        }
        if n == 0:
            result["skip_reason"] = "empty_context"
            return result

        # 1. preconditions -------------------------------------------------
        full = self._call(context)
        result["detector_outputs"]["full"] = self._output(full)
        result["full_value"] = float(full.flag)
        result["p_malign"] = float(full.flag)
        result["p_benign"] = 1.0 - float(full.flag)
        result["predicted_label"] = "DETECTED" if full.flag else "NOT_DETECTED"
        if not full.flag:
            result["skip_reason"] = "not_flagged"
            result["n_detector_calls"] = self._calls
            return result

        same = [
            (v.flag == full.flag and v.raw_output == full.raw_output)
            for v in (self._call(context, use_cache=False) for _ in range(self.determinism_checks))
        ]
        result["deterministic"] = all(same)

        game = self._build_game(seg, span)
        result["empty_text"] = game.empty_text
        result["fixed_words"] = list(game.fixed)
        baseline_verdict = self._call(game.text([]))
        result["detector_outputs"]["baseline"] = self._output(baseline_verdict)
        result["baseline_value"] = float(baseline_verdict.flag)
        result["baseline_valid"] = not baseline_verdict.flag
        result["n_detector_calls"] = self._calls
        if baseline_verdict.flag:
            result["skip_reason"] = "baseline_invalid"
            return result
        m = len(game.players)
        if m == 0:
            result["skip_reason"] = "no_players"
            return result
        result["n_players"] = m
        if progress:
            print(f"[blackbox] {n} words ({m} players); deterministic={result['deterministic']}")

        # 2. Kernel SHAP over the detector's decision ----------------------
        n_samples = self._resolve_n_samples(m)
        result["n_samples"] = n_samples
        if n_samples < _MIN_SAMPLES_PER_PLAYER * m:
            result["skip_reason"] = "budget_too_small"
            return result
        if progress:
            print(f"[blackbox:{self.backend}] Kernel SHAP, n_samples={n_samples} (budget {self.max_detector_calls})")

        def score_rows(rows: np.ndarray) -> list:
            return [
                float(self._call(game.text([game.players[j] for j in np.flatnonzero(row)])).flag)
                for row in rows
            ]

        self._budget_active = True
        try:
            phi = self._solve(score_rows, m, n_samples, progress)
        except BudgetExceeded as e:
            result["skip_reason"] = "budget_exceeded"
            result["n_detector_calls"] = self._calls
            if progress:
                print(f"[blackbox] Kernel SHAP stopped: {e}")
            return result
        finally:
            self._budget_active = False

        values = [0.0] * n
        for word, value in zip(game.players, phi):
            values[word] = float(value)
        total = float(sum(values))
        result.update({
            "values": values, "explained": True, "sum_values": total,
            "efficiency_error": abs(total - (result["full_value"] - result["baseline_value"])),
            "n_detector_calls": self._calls,
        })
        return result

    def explain_row(self, text: str, progress: bool = False) -> dict:
        """Explain an isolated span (`target_inst`/`injected_task`): same game, no injection to locate."""
        return self.explain_context(text, injected_task=None, attack_name=None, progress=progress)

    def make_rescorer(self, context: str, xai: Optional[dict] = None):
        """`rescore(keep_mask_words) -> float` (0.0/1.0): the detector's decision on
        `context` with only the kept words (plus the game's always-kept words,
        taken from the stored explanation `xai` when given) — the primitive
        `piarena/xai/metrics.py` and `scripts/xai_metrics.py` are built around."""
        seg = segment(context)
        empty_text = (xai or {}).get("empty_text", "")
        base_mask = seg.keep_mask((xai or {}).get("fixed_words") or [])
        self._budget_active = False

        def rescore(keep_mask: np.ndarray) -> float:
            mask = base_mask | np.asarray(keep_mask, dtype=bool)
            return float(self._call(seg.build(mask, empty_text)).flag)

        return rescore

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _locate_span(seg: Segmentation, context: str, injected_task: Optional[str], attack_name: Optional[str]):
        if not injected_task:
            return None
        # Lazy: reuses the Kernel SHAP explainer's own locator (accounts for the
        # fixed prefix a heuristic attack puts in front of `injected_task`).
        # Its attack-prefix lookup imports `piarena.attacks`, which needs torch;
        # where it can't be imported (no torch: model-free unit tests) fall back
        # to the bare `injected_task`.
        try:
            from ..kernelshap.classifier_explainer import ClassifierKernelExplainer

            located = ClassifierKernelExplainer._locate_injection(context, injected_task, attack_name)
        except ImportError:
            start = context.find(injected_task)
            located = None if start == -1 else (start, start + len(injected_task))
        return None if located is None else seg.word_span(*located)
