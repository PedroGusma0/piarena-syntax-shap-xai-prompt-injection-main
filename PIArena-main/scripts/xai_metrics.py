#!/usr/bin/env python3
"""Compute pilot metrics + saliency-map plots for a `--xai` run — works for
both `syntaxshap` and `kernelshap` result files (see `--xai-method` below).

Reads the evaluation-result JSON `main.py` already wrote (it has, per sample,
`xai_result.context` — tokens/values/injected_span — and
`defense_result.detect_flag`, plus `injected_context`, the text that was
actually explained, and `asr`/`utility`, the benchmark's own evaluators),
reloads the classifier + explainer to re-score masked variants for
Fidelity(t)/acc@1, and writes:

  - <out-dir>/<result-stem>_metrics.json     — machine-readable, per-sample + aggregate
  - <out-dir>/<result-stem>_report.md        — human-readable summary table, ending with
                                                an ASR/Utility summary over the whole run
  - <out-dir>/plots/saliency_maps.html       — shap.plots.text() per sample
  - <out-dir>/plots/fidelity_acc_bars.png    — mean Fidelity(t)/acc@1(t) per threshold
  - <out-dir>/plots/alignment_histogram.png  — distribution of injected_span_percentile

Since `main.py` only runs the XAI phase on samples the defense actually
blocked (`detect_flag=True`), `per_sample`/the plots only ever cover that
blocked subset — there is no true-positive/false-negative split to report
here anymore (that framing was dropped along with running XAI unconditionally).
The ASR/Utility summary, by contrast, is computed over the *whole* raw result
(every sample, blocked or not) — it's the benchmark-level context the
per-sample XAI metrics sit inside.

This script used to exist as two near-identical copies (one per `--xai`
method — same CLI, same metrics, same plots, differing only in which
explainer builds `rescore_fn`) before `piarena/xai/syntaxshap/` and
`piarena/xai/kernelshap/` were merged back into one PIArena tree. `--xai-method`
picks the explainer; "auto" (default) infers it from `--result`'s filename,
which `main.py` itself always suffixes with `-{xai}-` (see main.py's
`evaluation_result_path`).

See "Onde e como as métricas aparecem" and "Gráficos" in
plans/xai-syntaxshap-promptguard.md / plans/xai-kernelshap-promptguard.md for
the full design/format (identical between the two methods, method-specific
notes called out inline in each doc).

Usage:
    python scripts/xai_metrics.py \\
        --result results/evaluation_results/xai_pilot/squad_v2-...-syntaxshap-42.json \\
        --model-name meta-llama/Prompt-Guard-86M --algorithm syntax \\
        --thresholds 0.1 0.2 0.3 0.5

    python scripts/xai_metrics.py \\
        --result results/evaluation_results/xai_pilot/squad_v2-...-kernelshap-42.json
"""
import argparse
import json
import os
import time

import numpy as np

from piarena.utils import save_json
from piarena.xai.metrics import (
    acc_at_1,
    fidelity,
    injected_span_percentile,
)

_XAI_METHODS = ("syntaxshap", "kernelshap")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", required=True, help="Path to main.py's evaluation-result JSON.")
    p.add_argument("--xai-method", default="auto", choices=["auto", *_XAI_METHODS],
                    help="Which explainer produced --result. 'auto' (default) infers it from the "
                         "filename's -{xai}- suffix (main.py's own naming convention).")
    p.add_argument("--model-name", default="meta-llama/Prompt-Guard-86M")
    p.add_argument("--algorithm", default="syntax", choices=["syntax", "shap"],
                    help="syntaxshap-only: tree-restricted ('syntax') vs. unrestricted ('shap') coalition "
                         "enumeration. Ignored for --xai-method kernelshap.")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--out-dir", default=None, help="Default: <result dir>/xai_metrics/")
    p.add_argument("--max-plot-samples", type=int, default=25, help="Cap on samples rendered into saliency_maps.html.")
    p.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B logging (on by default). WANDB_MODE=offline unless already set in the environment.")
    return p.parse_args()


def _infer_xai_method(result_path: str) -> str:
    """Parses the `-{xai}-` suffix out of main.py's own filename convention
    (`{dataset}-{llm}-{attack}-{defense}-{xai}-{seed}.json`) — same idea as
    `_guess_attack` below, just for the XAI method instead of the attack."""
    stem = os.path.splitext(os.path.basename(result_path))[0]
    parts = set(stem.split("-"))
    found = [m for m in _XAI_METHODS if m in parts]
    if len(found) == 1:
        return found[0]
    raise ValueError(
        f"Could not infer --xai-method from filename {result_path!r} "
        f"(found {found!r} of {_XAI_METHODS}) — pass --xai-method explicitly."
    )


def _guess_attack(result_path: str | None) -> str:
    """Best-effort parse of the attack name out of main.py's own filename
    convention (`{dataset}-{llm}-{attack}-{defense}-{xai}-{seed}.json` —
    main.py:146-155) — only used to label/group the W&B run. Same logic as
    `xai_compare_fidelity.py::_guess_attack`, duplicated (not imported) to
    keep this script's own import list unchanged."""
    if not result_path:
        return "unknown"
    stem = os.path.splitext(os.path.basename(result_path))[0]
    known_attacks = {"direct", "ignore", "completion", "character", "combined", "none"}
    for part in stem.split("-"):
        if part in known_attacks:
            return part
    return "unknown"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_explainer(xai_method: str, model_name: str, algorithm: str):
    """Returns `(explainer, make_rescorer)` — `make_rescorer(context)` builds
    a `rescore_fn(keep_mask) -> float` closure using exactly the masking
    mechanism the given method's own explainer uses. Deferred, heavy imports
    (torch/transformers/spacy/captum) — only paid once metrics actually run,
    and only for whichever one method this call needs."""
    if xai_method == "syntaxshap":
        from piarena.xai.syntaxshap.classifier_explainer import ClassifierSyntaxExplainer
        from piarena.xai.syntaxshap.xai_syntaxshap import _load_classifier_pipeline
        from piarena.xai.syntaxshap.thirdparty.models import TransformersPipeline  # noqa (path already set up by classifier_explainer import)

        tokenizer, raw_pipeline = _load_classifier_pipeline(model_name)
        explainer = ClassifierSyntaxExplainer(TransformersPipeline(raw_pipeline), tokenizer, algorithm=algorithm)

        def make_rescorer(context):
            def rescore_p_malign(keep_mask: np.ndarray) -> float:
                return explainer._score(explainer._mask_to_string(keep_mask, context))
            return rescore_p_malign

        return explainer, make_rescorer

    # kernelshap
    from piarena.xai.kernelshap.classifier_explainer import ClassifierKernelExplainer
    from piarena.xai.kernelshap.xai_kernelshap import _load_classifier

    tokenizer, model = _load_classifier(model_name)
    # `backend=` doesn't matter here — re-scoring (make_rescorer) never calls
    # captum or shap, it only runs the model directly. Left at the default.
    explainer = ClassifierKernelExplainer(model, tokenizer)
    return explainer, explainer.make_rescorer


def compute_one(sample, make_rescorer, thresholds):
    xai = sample.get("xai_result", {}).get("context")
    defense_result = sample.get("defense_result", {})
    context = sample.get("injected_context")
    if xai is None or context is None:
        return None

    values = xai["values"]
    full_value = xai["full_value"]
    injected_span = xai.get("injected_span")

    rescore_p = make_rescorer(context)

    entry = {
        "predicted_label": xai.get("predicted_label"),
        # "p_malign" is the current field name; "p_non_benign" is kept as a
        # fallback so result files from before the malign rename still parse
        # correctly.
        "p_malign": xai.get("p_malign", xai.get("p_non_benign", full_value)),
        # method-specific metadata: "backend" (kernelshap) comes back None
        # for a syntaxshap result and vice versa for "timed_out" below —
        # simpler (and W&B-safe) than branching on xai_method here.
        "backend": xai.get("backend"),
        "detect_flag": defense_result.get("detect_flag"),
        "fidelity": fidelity(full_value, values, rescore_p, thresholds),
        "acc_at_1": acc_at_1(full_value, values, rescore_p, thresholds),
        # False/absent for every sample computed before the per-sample
        # timeout existed (ClassifierSyntaxExplainer.explain_context) and for
        # every kernelshap sample (that explainer never sets this field at
        # all, its cost is bounded by n_samples, not text structure) — only
        # ever True for a syntaxshap sample that actually hit the timeout.
        "timed_out": bool(xai.get("timed_out", False)),
    }
    if injected_span is not None:
        entry["injected_span"] = injected_span
        entry["injected_span_percentile"] = injected_span_percentile(values, injected_span)
    return entry


def _to_float(value):
    """asr/utility entries are bool (llm_judge/substring_match) or a float
    score (longbench metrics) — both cast cleanly to float for averaging."""
    if value is None:
        return None
    return float(value)


def _asr_utility_summary(result: dict) -> dict:
    """ASR/Utility over the WHOLE raw result (every sample, blocked or not)
    — the benchmark-level numbers the per-sample XAI metrics sit inside.
    `main.py` writes `asr`/`utility` for every sample regardless of `--xai`,
    so this doesn't require re-running any evaluator."""
    def rates(samples):
        asr_vals = [_to_float(s.get("asr")) for s in samples if s.get("asr") is not None]
        utility_vals = [_to_float(s.get("utility")) for s in samples if s.get("utility") is not None]
        return {
            "n": len(samples),
            "asr_rate": float(np.mean(asr_vals)) if asr_vals else None,
            "utility_mean": float(np.mean(utility_vals)) if utility_vals else None,
        }

    all_samples = list(result.values())
    blocked = [s for s in all_samples if s.get("defense_result", {}).get("detect_flag") is True]
    not_blocked = [s for s in all_samples if s.get("defense_result", {}).get("detect_flag") is False]

    return {
        "overall": rates(all_samples),
        "blocked": rates(blocked),
        "not_blocked": rates(not_blocked),
    }


def main():
    args = parse_args()
    result = load_json(args.result)
    xai_method = args.xai_method if args.xai_method != "auto" else _infer_xai_method(args.result)

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.result), "xai_metrics")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    explainer, make_rescorer = _load_explainer(xai_method, args.model_name, args.algorithm)

    per_sample = {}
    t0 = time.time()
    n_done = 0
    for idx, sample in result.items():
        entry = compute_one(sample, make_rescorer, args.thresholds)
        if entry is None:
            continue
        per_sample[idx] = entry
        n_done += 1
    elapsed = time.time() - t0

    aggregate = _aggregate(per_sample, args.thresholds)
    aggregate["runtime"] = {
        "mean_seconds_per_sample": elapsed / n_done if n_done else None,
        "total_seconds": elapsed,
        "n_samples": n_done,
    }
    aggregate["asr_utility"] = _asr_utility_summary(result)

    metrics_out = {
        "meta": {
            "result_path": args.result,
            "xai_method": xai_method,
            "model_name": args.model_name,
            # None for kernelshap (algorithm only applies to syntaxshap's
            # coalition enumeration) rather than omitted, so downstream
            # consumers can rely on the key always being present.
            "algorithm": args.algorithm if xai_method == "syntaxshap" else None,
            "thresholds": args.thresholds,
            "n_samples": n_done,
        },
        "per_sample": per_sample,
        "aggregate": aggregate,
    }

    stem = os.path.splitext(os.path.basename(args.result))[0]
    metrics_path = os.path.join(out_dir, f"{stem}_metrics.json")
    save_json(metrics_out, metrics_path)
    print(f"Wrote {metrics_path}")

    report_path = os.path.join(out_dir, f"{stem}_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_render_report(metrics_out))
    print(f"Wrote {report_path}")

    try:
        _render_plots(result, per_sample, plots_dir, xai_method, max_samples=args.max_plot_samples)
    except ImportError as e:
        print(f"Skipping plots (shap/matplotlib not installed): {e}")

    if not args.no_wandb:
        _log_to_wandb(metrics_out, per_sample, plots_dir, args.thresholds)


def _log_to_wandb(metrics_out, per_sample, plots_dir, thresholds):
    """Logs the already-computed metrics/plots to a W&B run — separate from
    (not resuming) the `wandb.init(job_type="run", ...)` `main.py` itself may
    have done for the same result file (keeps this script runnable
    standalone, without needing to know/pass a run id across processes);
    grouped by attack so runs from both --xai methods show up together in
    the W&B UI."""
    os.environ.setdefault("WANDB_MODE", "offline")
    try:
        import wandb
    except ImportError as e:
        print(f"Skipping W&B logging (wandb not installed): {e}")
        return

    meta = metrics_out["meta"]
    attack = _guess_attack(meta.get("result_path"))
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "piarena-xai-sweep"),
        group=attack,
        job_type="metrics",
        name=f"{attack}-{meta['xai_method']}-metrics",
        config=meta,
    )

    log_payload = {}
    fidelity_png = os.path.join(plots_dir, "fidelity_acc_bars.png")
    alignment_png = os.path.join(plots_dir, "alignment_histogram.png")
    saliency_html = os.path.join(plots_dir, "saliency_maps.html")
    if os.path.exists(fidelity_png):
        log_payload["fidelity_acc_bars"] = wandb.Image(fidelity_png)
    if os.path.exists(alignment_png):
        log_payload["alignment_histogram"] = wandb.Image(alignment_png)
    if os.path.exists(saliency_html):
        with open(saliency_html, "r", encoding="utf-8") as f:
            log_payload["saliency_maps"] = wandb.Html(f.read())

    columns = ["idx", *[f"fidelity@{t}" for t in thresholds], "timed_out", "injected_span_percentile"]
    table = wandb.Table(columns=columns)
    for idx, entry in per_sample.items():
        table.add_data(
            idx,
            *[entry.get("fidelity", {}).get(str(t)) for t in thresholds],
            entry.get("timed_out"),
            entry.get("injected_span_percentile"),
        )
    log_payload["per_sample"] = table

    wandb.log(log_payload)
    # `aggregate` already has exactly the numbers worth seeing on the run's
    # summary tab (fidelity_mean, acc_at_1_rate, injected_span_percentile,
    # timed_out, runtime, asr_utility) — no need to re-derive anything.
    wandb.summary.update(metrics_out["aggregate"])
    wandb.finish()


def _aggregate(per_sample, thresholds):
    def mean_by_t(key):
        out = {}
        for t in thresholds:
            vals = [e[key][str(t)] for e in per_sample.values() if key in e and e[key].get(str(t)) is not None]
            out[str(t)] = float(np.mean(vals)) if vals else None
        return out

    # per_sample only ever contains samples the defense blocked (XAI now only
    # runs on those, see main.py) — no true-positive/false-negative cohort to
    # split by anymore, just one population.
    percentiles = [e["injected_span_percentile"] for e in per_sample.values() if e.get("injected_span_percentile") is not None]
    n_timed_out = sum(1 for e in per_sample.values() if e.get("timed_out"))

    return {
        "fidelity_mean": mean_by_t("fidelity"),
        "acc_at_1_rate": mean_by_t("acc_at_1"),
        "injected_span_percentile": {
            "n_explained": len(percentiles),
            "mean_percentile": float(np.mean(percentiles)) if percentiles else None,
        },
        # How many of the explained (blocked) samples only got a PARTIAL
        # explanation because the per-sample timeout fired mid-way (see
        # classifier_explainer.py's explain_context) — the coverage-bias
        # caveat from plans/xai-syntaxshap-promptguard.md, made a first-class
        # number instead of something only visible by grepping raw results.
        # Always 0 for a run with `timeout_seconds=None` or for methods
        # (kernelshap) that never set this field at all.
        "timed_out": {
            "n": n_timed_out,
            "rate": float(n_timed_out / len(per_sample)) if per_sample else None,
        },
    }


def _render_report(metrics_out) -> str:
    meta = metrics_out["meta"]
    agg = metrics_out["aggregate"]
    title = f"# XAI pilot report — {meta['xai_method']}"
    if meta.get("algorithm"):
        title += f" ({meta['algorithm']})"
    title += f" ({meta['model_name']})"
    lines = [
        title,
        "",
        f"- Result file: `{meta['result_path']}`",
        f"- Samples explained (blocked by the defense): {meta['n_samples']}",
        f"- Runtime: {agg['runtime']['mean_seconds_per_sample']:.2f}s/sample "
        f"({agg['runtime']['total_seconds']:.1f}s total)" if agg['runtime']['mean_seconds_per_sample'] else "- Runtime: n/a",
        f"- Timed out (partial explanation, per-sample timeout hit): "
        f"{agg['timed_out']['n']}/{meta['n_samples']} "
        f"({agg['timed_out']['rate']:.0%})" if agg['timed_out']['rate'] is not None else "- Timed out: n/a",
        "",
        "## Fidelity(t) / acc@1",
        "",
        "| t | Fidelity (lower=better) | acc@1 |",
        "|---|---|---|",
    ]
    for t in meta["thresholds"]:
        key = str(t)
        lines.append(
            f"| {t} | {agg['fidelity_mean'].get(key)} | {agg['acc_at_1_rate'].get(key)} |"
        )
    span = agg["injected_span_percentile"]
    lines += [
        "",
        "## Alinhamento semântico (span injetado)",
        "",
        f"- Amostras explicadas: {span['n_explained']}",
        f"- Percentil médio do span injetado: {span['mean_percentile']}",
        "",
    ]

    au = agg["asr_utility"]
    lines += [
        "## Resumo ASR / Utility",
        "",
        "Calculado sobre o resultado bruto inteiro (todas as amostras do dataset, "
        "não só as explicadas pelo XAI) — o contexto de benchmark em que as métricas "
        "de XAI acima se encaixam.",
        "",
        "| grupo | n | ASR | Utility |",
        "|---|---|---|---|",
        f"| geral | {au['overall']['n']} | {au['overall']['asr_rate']} | {au['overall']['utility_mean']} |",
        f"| bloqueadas pela defesa | {au['blocked']['n']} | {au['blocked']['asr_rate']} | {au['blocked']['utility_mean']} |",
        f"| não bloqueadas | {au['not_blocked']['n']} | {au['not_blocked']['asr_rate']} | {au['not_blocked']['utility_mean']} |",
        "",
    ]
    return "\n".join(lines)


def _render_plots(result, per_sample, plots_dir, xai_method, max_samples=25):
    import shap  # official shap package — plotting only, not computation
    import matplotlib
    matplotlib.use("Agg")

    html_parts = [f"<html><head><meta charset='utf-8'><title>{xai_method} saliency maps</title></head><body>"]
    n = 0
    for idx, sample in result.items():
        xai = sample.get("xai_result", {}).get("context")
        defense_result = sample.get("defense_result", {})
        if xai is None or not xai.get("values"):
            continue
        explanation = shap.Explanation(
            values=np.array(xai["values"]),
            base_values=xai.get("base_value", 0.0),
            data=xai["tokens"],
        )
        p_malign = xai.get("p_malign", xai.get("p_non_benign"))
        header = (
            f"<h3>Sample {idx} — predicted_label={xai.get('predicted_label')}, "
            f"detect_flag={defense_result.get('detect_flag')}, "
            f"P(malign)={p_malign}</h3>"
        )
        try:
            body = shap.plots.text(explanation, display=False)
        except Exception as e:  # pragma: no cover — defensive, don't let one bad sample kill the whole report
            body = f"<p>(failed to render: {e})</p>"
        html_parts.append(header + body)
        n += 1
        if n >= max_samples:
            break
    html_parts.append("</body></html>")
    with open(os.path.join(plots_dir, "saliency_maps.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"Wrote {os.path.join(plots_dir, 'saliency_maps.html')} ({n} sample(s))")

    _render_fidelity_acc_bars(per_sample, plots_dir)
    _render_alignment_histogram(per_sample, plots_dir)


def _render_fidelity_acc_bars(per_sample, plots_dir):
    """Grouped bar chart, one group per threshold t: mean Fidelity(t) vs.
    mean acc@1(t) across every explained (blocked) sample. Replaces the old
    true-positive/false-negative cohort chart, which stopped making sense
    once XAI only ever runs on blocked samples (no FN cohort left to compare
    against)."""
    import matplotlib.pyplot as plt

    thresholds = sorted({t for e in per_sample.values() for t in e.get("fidelity", {})}, key=float)
    if not thresholds:
        return

    fidelity_means = []
    acc_means = []
    for t in thresholds:
        fid_vals = [e["fidelity"][t] for e in per_sample.values() if e.get("fidelity", {}).get(t) is not None]
        acc_vals = [e["acc_at_1"][t] for e in per_sample.values() if e.get("acc_at_1", {}).get(t) is not None]
        fidelity_means.append(float(np.mean(fid_vals)) if fid_vals else 0.0)
        acc_means.append(float(np.mean(acc_vals)) if acc_vals else 0.0)

    x = np.arange(len(thresholds))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, fidelity_means, width, label="Fidelity(t) (lower=better)")
    ax.bar(x + width / 2, acc_means, width, label="acc@1(t)")
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.set_xlabel("t (top-t% tokens/words kept)")
    ax.set_ylabel("mean value")
    ax.set_title("Fidelity(t) / acc@1(t) — blocked samples")
    ax.legend()
    fig.savefig(os.path.join(plots_dir, "fidelity_acc_bars.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {os.path.join(plots_dir, 'fidelity_acc_bars.png')}")


def _render_alignment_histogram(per_sample, plots_dir):
    """Histogram of injected_span_percentile across every explained (blocked)
    sample — the distribution behind the single mean reported in the
    Markdown report's "Alinhamento semântico" section."""
    import matplotlib.pyplot as plt

    percentiles = [e["injected_span_percentile"] for e in per_sample.values() if e.get("injected_span_percentile") is not None]
    if not percentiles:
        return

    fig, ax = plt.subplots()
    ax.hist(percentiles, bins=min(20, max(5, len(percentiles))), range=(0, 1))
    ax.set_xlabel("injected_span_percentile")
    ax.set_ylabel("# samples")
    ax.set_title("Distribuição do alinhamento semântico (span injetado)")
    fig.savefig(os.path.join(plots_dir, "alignment_histogram.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {os.path.join(plots_dir, 'alignment_histogram.png')}")


if __name__ == "__main__":
    main()
