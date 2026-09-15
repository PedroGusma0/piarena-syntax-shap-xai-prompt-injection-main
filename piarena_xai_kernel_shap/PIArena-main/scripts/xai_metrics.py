#!/usr/bin/env python3
"""Compute pilot metrics + saliency-map plots for a `--xai kernelshap` run.

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
here anymore (that framing was dropped along with running XAI unconditionally;
see `plans/xai-kernelshap-promptguard.md`). The ASR/Utility summary, by
contrast, is computed over the *whole* raw result (every sample, blocked or
not) — it's the benchmark-level context the per-sample XAI metrics sit inside.

See `plans/xai-kernelshap-promptguard.md` for the full design/format. This
script is structurally identical to the parallel SyntaxSHAP experiment's
`scripts/xai_metrics.py` (same CLI, same metrics, same plots via the official
`shap` package) — only the explainer used to build the `rescore_fn` differs.

Usage:
    python scripts/xai_metrics.py \\
        --result results/evaluation_results/xai_pilot/squad_v2-...-kernelshap-42.json \\
        --model-name meta-llama/Prompt-Guard-86M \\
        --thresholds 0.1 0.2 0.3 0.5
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", required=True, help="Path to main.py's evaluation-result JSON.")
    p.add_argument("--model-name", default="meta-llama/Prompt-Guard-86M")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--out-dir", default=None, help="Default: <result dir>/xai_metrics/")
    p.add_argument("--max-plot-samples", type=int, default=25, help="Cap on samples rendered into saliency_maps.html.")
    p.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B logging (on by default). WANDB_MODE=offline unless already set in the environment.")
    return p.parse_args()


def _guess_attack(result_path: str | None) -> str:
    """Best-effort parse of the attack name out of main.py's own filename
    convention (`{dataset}-{llm}-{attack}-{defense}-{xai}-{seed}.json`) —
    only used to label/group the W&B run. Same logic as
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


def compute_one(sample, explainer, thresholds):
    xai = sample.get("xai_result", {}).get("context")
    defense_result = sample.get("defense_result", {})
    context = sample.get("injected_context")
    if xai is None or context is None:
        return None

    values = xai["values"]
    full_value = xai["full_value"]
    injected_span = xai.get("injected_span")

    # Matches exactly the masking mechanism ClassifierKernelExplainer itself
    # uses (same baseline token / feature grouping) for this specific context
    # — backend-agnostic: re-scoring runs the model directly, whether the
    # original explanation was computed via captum or shap.
    rescore_p = explainer.make_rescorer(context)

    entry = {
        "predicted_label": xai.get("predicted_label"),
        # "p_malign" is the current field name; "p_non_benign" is kept as a
        # fallback so result files from before the malign rename (e.g. the
        # v1 captum pilot) still parse correctly.
        "p_malign": xai.get("p_malign", xai.get("p_non_benign", full_value)),
        "backend": xai.get("backend"),
        "detect_flag": defense_result.get("detect_flag"),
        "fidelity": fidelity(full_value, values, rescore_p, thresholds),
        "acc_at_1": acc_at_1(full_value, values, rescore_p, thresholds),
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

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.result), "xai_metrics")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Deferred, heavy imports — only needed once we actually run metrics.
    from piarena.xai.kernelshap.classifier_explainer import ClassifierKernelExplainer
    from piarena.xai.kernelshap.xai_kernelshap import _load_classifier

    tokenizer, model = _load_classifier(args.model_name)
    # `backend=` doesn't matter here — re-scoring (make_rescorer) never calls
    # captum or shap, it only runs the model directly. Left at the default.
    explainer = ClassifierKernelExplainer(model, tokenizer)

    per_sample = {}
    t0 = time.time()
    n_done = 0
    for idx, sample in result.items():
        entry = compute_one(sample, explainer, args.thresholds)
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
            "model_name": args.model_name,
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
        _render_plots(result, per_sample, plots_dir, max_samples=args.max_plot_samples)
    except ImportError as e:
        print(f"Skipping plots (shap/matplotlib not installed): {e}")

    if not args.no_wandb:
        _log_to_wandb(metrics_out, per_sample, plots_dir, args.thresholds)


def _log_to_wandb(metrics_out, per_sample, plots_dir, thresholds):
    """Logs the already-computed metrics/plots to a W&B run — separate from
    (not resuming) the `wandb.init(job_type="run", ...)` `main.py` itself may
    have done for the same result file (keeps this script runnable
    standalone, without needing to know/pass a run id across processes);
    grouped by attack so the two show up together in the W&B UI."""
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
        name=f"{attack}-kernelshap-metrics",
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

    columns = ["idx", *[f"fidelity@{t}" for t in thresholds], "injected_span_percentile"]
    table = wandb.Table(columns=columns)
    for idx, entry in per_sample.items():
        table.add_data(
            idx,
            *[entry.get("fidelity", {}).get(str(t)) for t in thresholds],
            entry.get("injected_span_percentile"),
        )
    log_payload["per_sample"] = table

    wandb.log(log_payload)
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

    return {
        "fidelity_mean": mean_by_t("fidelity"),
        "acc_at_1_rate": mean_by_t("acc_at_1"),
        "injected_span_percentile": {
            "n_explained": len(percentiles),
            "mean_percentile": float(np.mean(percentiles)) if percentiles else None,
        },
    }


def _render_report(metrics_out) -> str:
    meta = metrics_out["meta"]
    agg = metrics_out["aggregate"]
    lines = [
        f"# XAI pilot report — kernelshap ({meta['model_name']})",
        "",
        f"- Result file: `{meta['result_path']}`",
        f"- Samples explained (blocked by the defense): {meta['n_samples']}",
        f"- Runtime: {agg['runtime']['mean_seconds_per_sample']:.2f}s/sample "
        f"({agg['runtime']['total_seconds']:.1f}s total)" if agg['runtime']['mean_seconds_per_sample'] else "- Runtime: n/a",
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


def _render_plots(result, per_sample, plots_dir, max_samples=25):
    import shap  # official shap package — plotting only, not computation
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    html_parts = ["<html><head><meta charset='utf-8'><title>Kernel SHAP saliency maps</title></head><body>"]
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
    ax.set_xlabel("t (top-t% words kept)")
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
