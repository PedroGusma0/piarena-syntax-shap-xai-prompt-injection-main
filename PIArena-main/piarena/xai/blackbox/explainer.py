"""`BlackBoxExplainer`: explains a boolean detector (`text -> Verdict`) by
perturbing its input and calling it. Works on any `piarena.xai.detectors.
BooleanDetector`; nothing here knows about DataSentinel, classifiers, tokenizers
or logits.

Per sample (`explain_context`):
  1. preconditions — `full_value = detect(context)`, determinism re-runs, and
     the baseline check `detect(baseline_text)` (a sample whose baseline is
     already flagged has no benign reference to explain against and is marked
     `baseline_valid=False`, not explained);
  2. `ddmin`: a 1-minimal subset of sentences, then of the words inside them,
     that still makes the detector fire;
  3. `sentence_shapley` (when the context has few enough sentences): exact
     Shapley values of the boolean game over sentences.

The first method in `methods` fills `values`; the others go to `extra_methods`.
`values` always satisfy efficiency (`sum(values) == full_value - base_value`,
i.e. 1.0 for a valid sample), so `shap.plots.text` shows a consistent f(x).
Every distinct detector output seen at the key points (full text, baseline,
minimal set only, everything but the minimal set) is returned in
`detector_outputs`, with the raw text the detector generated.

See plans/xai-blackbox-datasentinel.md for the design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from ..detectors import BooleanDetector, Verdict
from .ddmin import BudgetExceeded, ddmin
from .perturb import Segmentation, segment
from .sentence_shapley import exact_shapley

_BASELINES = ("empty", "neutral", "clean_context")
_METHODS = ("ddmin", "sentence_shapley")


@dataclass
class _Game:
    """The perturbation game for one context: `fixed` words are always kept,
    `units` (lists of word indices — sentences, or the injected span's part of
    each sentence for the `clean_context` baseline) are the players."""
    seg: Segmentation
    fixed: list
    units: list
    empty_text: str

    def text(self, kept_words: Iterable[int]) -> str:
        mask = np.zeros(self.seg.n_words, dtype=bool)
        if self.fixed:
            mask[self.fixed] = True
        kept = list(kept_words)
        if kept:
            mask[kept] = True
        return self.seg.build(mask, self.empty_text)

    def unit_words(self, unit_ids: Iterable[int]) -> list:
        return [w for u in unit_ids for w in self.units[u]]


class BlackBoxExplainer:
    def __init__(
        self,
        detector: BooleanDetector,
        methods: Sequence[str] = _METHODS,
        baseline: str = "empty",
        neutral_text: str = "The weather is nice today.",
        determinism_checks: int = 3,
        max_exact_sentences: int = 10,
        max_detector_calls: int = 2000,
        record_trace: bool = False,
    ):
        unknown = [m for m in methods if m not in _METHODS]
        if unknown:
            raise ValueError(f"Unknown blackbox method(s) {unknown!r}; available: {list(_METHODS)}")
        if baseline not in _BASELINES:
            raise ValueError(f"baseline must be one of {_BASELINES}, got {baseline!r}")
        self.detector = detector
        self.methods = list(methods)
        self.baseline = baseline
        self.neutral_text = neutral_text
        self.determinism_checks = int(determinism_checks)
        self.max_exact_sentences = int(max_exact_sentences)
        self.max_detector_calls = int(max_detector_calls)
        self.record_trace = bool(record_trace)
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
            in_span = set(range(span["start_token"], span["end_token"]))
            fixed = [i for i in range(seg.n_words) if i not in in_span]
            units = [[w for w in words if w in in_span] for words in seg.sentence_words]
            units = [u for u in units if u]
            return _Game(seg, fixed, units, "")
        empty_text = self.neutral_text if self.baseline == "neutral" else ""
        return _Game(seg, [], [list(w) for w in seg.sentence_words], empty_text)

    # -- methods ------------------------------------------------------------

    def _run_ddmin(self, game: _Game) -> dict:
        n_units = len(game.units)
        stage1 = ddmin(
            list(range(n_units)),
            lambda kept: self._call(game.text(game.unit_words(kept))).flag,
            record_trace=self.record_trace,
        )
        words = game.unit_words(stage1.minimal)
        truncated = stage1.truncated
        trace = [("units", k, f) for k, f in stage1.trace]
        n_tests = stage1.n_tests
        minimal_words = words
        if not truncated and len(words) > 1:
            stage2 = ddmin(
                words,
                lambda kept: self._call(game.text(kept)).flag,
                record_trace=self.record_trace,
            )
            minimal_words = stage2.minimal
            truncated = stage2.truncated
            n_tests += stage2.n_tests
            trace += [("words", k, f) for k, f in stage2.trace]
        return {
            "minimal_words": sorted(minimal_words),
            "minimal_units": sorted(stage1.minimal),
            "n_tests": n_tests,
            "truncated": truncated,
            "trace": trace,
        }

    def _run_sentence_shapley(self, game: _Game, baseline_value: float) -> dict:
        m = len(game.units)
        if m > self.max_exact_sentences:
            return {"skipped": f"{m} sentences > max_exact_sentences={self.max_exact_sentences}"}
        needed = 2 ** m - 1
        remaining = self.max_detector_calls - self._calls
        if needed > remaining:
            return {"skipped": f"needs up to {needed} detector calls, {max(remaining, 0)} left in the budget"}
        before = self._calls
        phi = exact_shapley(
            list(range(m)),
            lambda coalition: float(self._call(game.text(game.unit_words(sorted(coalition)))).flag),
            baseline_value,
        )
        return {"phi": phi, "n_calls": self._calls - before}

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
        zeros = [0.0] * n
        result = {
            "tokens": list(seg.words), "values": zeros, "base_value": 0.0, "full_value": 0.0,
            "p_malign": 0.0, "p_benign": 1.0, "predicted_label": None,
            "method": None, "granularity": None, "explained": False, "skip_reason": None,
            "injected_span": span, "baseline": self.baseline, "baseline_value": None,
            "baseline_valid": None, "empty_text": "", "fixed_words": [],
            "deterministic": None, "minimal_set": [], "n_words": n, "n_sentences": seg.n_sentences,
            "n_detector_calls": 0, "truncated": False, "detector_outputs": {}, "extra_methods": {},
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
        if progress:
            print(f"[blackbox] {n} words / {seg.n_sentences} sentences; deterministic={result['deterministic']}")

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
        if not game.units:
            result["skip_reason"] = "no_players"
            return result

        # 2. methods ---------------------------------------------------------
        self._budget_active = True
        per_method: dict[str, dict] = {}
        for name in self.methods:
            if name == "ddmin":
                out = self._run_ddmin(game)
                minimal = out["minimal_words"]
                k = len(minimal)
                values = [0.0] * n
                for w in minimal:
                    values[w] = 1.0 / k
                per_method["ddmin"] = {"values": values, "granularity": "word", "detail": out}
                if progress:
                    print(f"[blackbox] ddmin: {k} word(s) in the minimal set, {out['n_tests']} tests")
            else:
                try:
                    out = self._run_sentence_shapley(game, float(baseline_verdict.flag))
                except BudgetExceeded as e:
                    out = {"skipped": str(e)}
                if "skipped" in out:
                    per_method["sentence_shapley"] = {"skipped": out["skipped"]}
                    if progress:
                        print(f"[blackbox] sentence_shapley skipped: {out['skipped']}")
                    continue
                values = [0.0] * n
                for unit, phi in zip(game.units, out["phi"]):
                    for w in unit:
                        values[w] = phi / len(unit)
                per_method["sentence_shapley"] = {
                    "values": values, "granularity": "sentence", "detail": out,
                    "unit_values": list(out["phi"]), "unit_spans": [[u[0], u[-1] + 1] for u in game.units],
                }
        self._budget_active = False

        primary = next((m for m in self.methods if "values" in per_method.get(m, {})), None)
        if primary is None:
            result["skip_reason"] = "no_method_succeeded"
            result["n_detector_calls"] = self._calls
            return result
        result.update({
            "values": per_method[primary]["values"], "method": primary,
            "granularity": per_method[primary]["granularity"], "explained": True,
        })
        for name, data in per_method.items():
            if name == primary:
                continue
            result["extra_methods"][name] = (
                {"skipped": data["skipped"]} if "skipped" in data else
                {k: v for k, v in data.items() if k != "detail"} | {"n_detector_calls": data["detail"].get("n_calls")}
            )

        ddmin_out = per_method.get("ddmin", {}).get("detail")
        if ddmin_out is not None:
            minimal = ddmin_out["minimal_words"]
            result["minimal_set"] = minimal
            result["truncated"] = bool(ddmin_out["truncated"])
            if self.record_trace:
                result["trace"] = [
                    {"stage": stage, "kept": list(kept), "flag": bool(flag)} for stage, kept, flag in ddmin_out["trace"]
                ]
            self._budget_active = False
            result["detector_outputs"]["minimal_only"] = self._output(self._call(game.text(minimal)))
            all_but = [w for w in range(n) if w not in set(minimal)]
            result["detector_outputs"]["without_minimal"] = self._output(self._call(game.text(all_but)))
        result["n_detector_calls"] = self._calls
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
        # where it can't be imported (no torch: model-free unit tests) fall back to the bare `injected_task`.
        try:
            from ..kernelshap.classifier_explainer import ClassifierKernelExplainer

            located = ClassifierKernelExplainer._locate_injection(context, injected_task, attack_name)
        except ImportError:
            start = context.find(injected_task)
            located = None if start == -1 else (start, start + len(injected_task))
        return None if located is None else seg.word_span(*located)
