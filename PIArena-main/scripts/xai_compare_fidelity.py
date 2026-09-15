#!/usr/bin/env python3
"""Compares Fidelity(t) between two XAI methods that explained the same
attack/dataset/seed — built for `syntaxshap` vs. `kernelshap`, but generic in
either method (labels only).

Reads two already-computed `scripts/xai_metrics.py` outputs (`_metrics.json`
— one per method) and cross-analyzes their `per_sample[idx]["fidelity"]`
entries. No model loading, no torch/transformers/spacy — pure JSON + numpy,
same spirit as `piarena/xai/metrics.py`. See "Parte 2" of the sweep plan for
the full design rationale.

Alignment: only the **intersection** of sample indices both `_metrics.json`
files actually explained is compared — method A (typically `syntaxshap`, if
it hit its own per-sample timeout on some samples — see
`xai_syntaxshap.py`'s `timeout_seconds`) may cover fewer samples than method
B (typically `kernelshap`, effectively uncapped). The comparison numbers
below are therefore only representative of that intersection (the
syntactically "easier" subset), not the full population the defense blocked
— reported explicitly as `n_a_explained`/`n_b_explained`/`n_common` in every
output, and called out again in the report's "Run summary".

Writes, under `--out-dir`:
  - {a_label}_vs_{b_label}_fidelity_comparison_metrics.json  — raw paired data + aggregates
  - {a_label}_vs_{b_label}_fidelity_comparison_report.md     — run summary + comparison table
  - fidelity_comparison.png                                  — grouped bar chart per threshold

Usage:
    python scripts/xai_compare_fidelity.py \\
        --a-metrics results/evaluation_results/full_sweep/xai_metrics/squad_v2-...-syntaxshap-42_metrics.json --a-label syntaxshap \\
        --b-metrics results/evaluation_results/full_sweep/xai_metrics/squad_v2-...-kernelshap-42_metrics.json --b-label kernelshap \\
        --attack direct \\
        --out-dir results_comparison/direct
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a-metrics", required=True, help="Path to method A's _metrics.json (scripts/xai_metrics.py output).")
    p.add_argument("--a-label", default="syntaxshap")
    p.add_argument("--b-metrics", required=True, help="Path to method B's _metrics.json.")
    p.add_argument("--b-label", default="kernelshap")
    p.add_argument("--attack", default=None,
                    help="Attack name, for the report header only — if omitted, parsed "
                         "best-effort from --a-metrics's meta.result_path filename.")
    p.add_argument("--thresholds", type=float, nargs="+", default=None,
                    help="Default: thresholds present in both _metrics.json files.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B logging (on by default). WANDB_MODE=offline unless already set in the environment.")
    return p.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class _NpEncoder(json.JSONEncoder):
    """Local, minimal — deliberately NOT `piarena.utils.NpEncoder` (that
    module imports torch/transformers unconditionally just to define a JSON
    encoder). This script has no reason to need either installed."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)


def _guess_attack(result_path: str | None) -> str:
    """Best-effort parse of the attack name out of main.py's own filename
    convention (`{dataset}-{llm}-{attack}-{defense}-{xai}-{seed}.json` —
    main.py:146-155) — only used when `--attack` wasn't passed. `llm_name`
    itself may contain '-' (e.g. "Qwen-Qwen3-4B-Instruct-2507"), so this is
    inherently ambiguous without knowing the attack list; done as a display
    convenience only, never relied on for correctness."""
    if not result_path:
        return "unknown"
    stem = os.path.splitext(os.path.basename(result_path))[0]
    parts = stem.split("-")
    known_attacks = {"direct", "ignore", "completion", "character", "combined", "none"}
    for part in parts:
        if part in known_attacks:
            return part
    return "unknown"


def _resolve_thresholds(a_meta, b_meta, requested):
    a_available = [str(t) for t in a_meta["thresholds"]]
    b_available = set(str(t) for t in b_meta["thresholds"])
    common = [t for t in a_available if t in b_available]
    if requested is None:
        return common
    requested_str = [str(t) for t in requested]
    missing = [t for t in requested_str if t not in common]
    if missing:
        raise ValueError(
            f"threshold(s) {missing} not present in both _metrics.json files "
            f"(A has {a_available}, B has {sorted(b_available, key=float)})"
        )
    return requested_str


def compare_fidelity(a_metrics: dict, b_metrics: dict, thresholds: list[str]) -> dict:
    a_samples = a_metrics["per_sample"]
    b_samples = b_metrics["per_sample"]
    common_idx = sorted(set(a_samples) & set(b_samples), key=int)

    per_threshold = {}
    for t in thresholds:
        pairs = [
            (a_samples[i]["fidelity"][t], b_samples[i]["fidelity"][t])
            for i in common_idx
            if a_samples[i].get("fidelity", {}).get(t) is not None
            and b_samples[i].get("fidelity", {}).get(t) is not None
        ]
        if not pairs:
            per_threshold[t] = None
            continue
        a_vals = np.array([p[0] for p in pairs], dtype=float)
        b_vals = np.array([p[1] for p in pairs], dtype=float)
        diff = a_vals - b_vals
        a_more_faithful = np.abs(a_vals) < np.abs(b_vals)  # closer to 0 = more faithful (fidelity() docstring)
        per_threshold[t] = {
            "n": len(pairs),
            "a_mean": float(np.mean(a_vals)), "a_std": float(np.std(a_vals)),
            "b_mean": float(np.mean(b_vals)), "b_std": float(np.std(b_vals)),
            "diff_mean_a_minus_b": float(np.mean(diff)), "diff_std_a_minus_b": float(np.std(diff)),
            "a_more_faithful_rate": float(np.mean(a_more_faithful)),
        }

    return {
        "n_a_explained": len(a_samples),
        "n_b_explained": len(b_samples),
        "n_common": len(common_idx),
        "common_indices": common_idx,
        "per_threshold": per_threshold,
    }


def _render_report(a_label, b_label, attack, a_metrics, b_metrics, comparison) -> str:
    a_meta, b_meta = a_metrics["meta"], b_metrics["meta"]
    a_agg, b_agg = a_metrics["aggregate"], b_metrics["aggregate"]

    def _model_names():
        if a_meta.get("model_name") == b_meta.get("model_name"):
            return f"`{a_meta.get('model_name')}`"
        return f"A=`{a_meta.get('model_name')}`, B=`{b_meta.get('model_name')}`"

    def _blocked_n(agg):
        return agg.get("asr_utility", {}).get("blocked", {}).get("n")

    def _rate_line(label, agg):
        au = agg.get("asr_utility", {})
        overall = au.get("overall", {})
        runtime = agg.get("runtime", {})
        timed_out = agg.get("timed_out")  # only present for syntaxshap-side metrics
        timed_out_str = ""
        if timed_out and timed_out.get("rate") is not None:
            timed_out_str = f", timed out {timed_out['n']} ({timed_out['rate']:.0%})"
        rt = runtime.get("mean_seconds_per_sample")
        rt_str = f"{rt:.2f}s/sample" if rt is not None else "n/a"
        # `runtime.n_samples` (not a top-level `agg["n_samples"]`, which
        # doesn't exist) is the same count as meta["n_samples"] — see
        # scripts/xai_metrics.py::main()'s `aggregate["runtime"] = {...,
        # "n_samples": n_done}`.
        n_explained = runtime.get("n_samples", "?")
        return (
            f"- **{label}**: {n_explained} explained "
            f"(of {_blocked_n(agg)} blocked{timed_out_str}) — "
            f"ASR {overall.get('asr_rate')}, Utility {overall.get('utility_mean')}, "
            f"runtime {rt_str}"
        )

    lines = [
        f"# Fidelity comparison — {a_label} vs. {b_label} ({attack})",
        "",
        "## Run summary",
        "",
        f"- Attack: `{attack}`",
        f"- Model: {_model_names()}",
        f"- A result file: `{a_meta.get('result_path')}`",
        f"- B result file: `{b_meta.get('result_path')}`",
        _rate_line(a_label, a_agg),
        _rate_line(b_label, b_agg),
        f"- Samples compared (intersection of what both methods explained): "
        f"{comparison['n_common']} (A explained {comparison['n_a_explained']}, "
        f"B explained {comparison['n_b_explained']})",
    ]
    if comparison["n_common"] < min(comparison["n_a_explained"], comparison["n_b_explained"]):
        lines.append(
            "  - ⚠️ The two methods did not explain the exact same sample set — "
            "the numbers below cover only their intersection, which may not be "
            "representative of either method's full blocked population (see "
            "module docstring / plan's \"Viés de cobertura\")."
        )
    lines += [
        "",
        "## Fidelity(t): lower |value| = more faithful",
        "",
        f"| t | n | {a_label} mean±std | {b_label} mean±std | diff (A-B) mean±std | {a_label} more faithful |",
        "|---|---|---|---|---|---|",
    ]
    for t, row in comparison["per_threshold"].items():
        if row is None:
            lines.append(f"| {t} | 0 | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {t} | {row['n']} | {row['a_mean']:.4f}±{row['a_std']:.4f} | "
            f"{row['b_mean']:.4f}±{row['b_std']:.4f} | "
            f"{row['diff_mean_a_minus_b']:+.4f}±{row['diff_std_a_minus_b']:.4f} | "
            f"{row['a_more_faithful_rate']:.0%} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_plot(a_label, b_label, comparison, out_dir):
    import matplotlib.pyplot as plt

    rows = [(t, row) for t, row in comparison["per_threshold"].items() if row is not None]
    if not rows:
        return
    thresholds = [t for t, _ in rows]
    a_means = [row["a_mean"] for _, row in rows]
    b_means = [row["b_mean"] for _, row in rows]

    x = np.arange(len(thresholds))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, a_means, width, label=f"{a_label} (mean Fidelity(t))")
    ax.bar(x + width / 2, b_means, width, label=f"{b_label} (mean Fidelity(t))")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.set_xlabel("t (top-t% tokens kept)")
    ax.set_ylabel("mean Fidelity(t) (closer to 0 = more faithful)")
    ax.set_title(f"Fidelity(t) — {a_label} vs. {b_label}")
    ax.legend()
    out_path = os.path.join(out_dir, "fidelity_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    args = parse_args()
    a_metrics = load_json(args.a_metrics)
    b_metrics = load_json(args.b_metrics)

    attack = args.attack or _guess_attack(a_metrics["meta"].get("result_path"))
    thresholds = _resolve_thresholds(a_metrics["meta"], b_metrics["meta"], args.thresholds)

    comparison = compare_fidelity(a_metrics, b_metrics, thresholds)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.a_label}_vs_{args.b_label}"

    out = {
        "meta": {
            "attack": attack,
            "a_label": args.a_label, "a_metrics_path": args.a_metrics,
            "b_label": args.b_label, "b_metrics_path": args.b_metrics,
            "thresholds": thresholds,
        },
        "comparison": comparison,
    }
    metrics_path = os.path.join(args.out_dir, f"{stem}_fidelity_comparison_metrics.json")
    save_json(out, metrics_path)
    print(f"Wrote {metrics_path}")

    report = _render_report(args.a_label, args.b_label, attack, a_metrics, b_metrics, comparison)
    report_path = os.path.join(args.out_dir, f"{stem}_fidelity_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {report_path}")

    try:
        _render_plot(args.a_label, args.b_label, comparison, args.out_dir)
    except ImportError as e:
        print(f"Skipping plot (matplotlib not installed): {e}")

    if not args.no_wandb:
        _log_to_wandb(args, attack, out, args.out_dir)


def _log_to_wandb(args, attack, out, out_dir):
    """Logs the comparison table + plot + summary to their own W&B run
    (job_type="compare"), grouped by attack alongside the two methods' own
    "run"/"metrics" runs — makes "who's more faithful" navigable on
    wandb.ai, not just in the `.md` report."""
    os.environ.setdefault("WANDB_MODE", "offline")
    try:
        import wandb
    except ImportError as e:
        print(f"Skipping W&B logging (wandb not installed): {e}")
        return

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "piarena-xai-sweep"),
        group=attack,
        job_type="compare",
        name=f"{attack}-{args.a_label}_vs_{args.b_label}-compare",
        config=out["meta"],
    )

    comparison = out["comparison"]
    log_payload = {}
    plot_path = os.path.join(out_dir, "fidelity_comparison.png")
    if os.path.exists(plot_path):
        log_payload["fidelity_comparison"] = wandb.Image(plot_path)

    table = wandb.Table(columns=[
        "threshold", "n", f"{args.a_label}_mean", f"{args.b_label}_mean",
        "diff_mean_a_minus_b", f"{args.a_label}_more_faithful_rate",
    ])
    for t, row in comparison["per_threshold"].items():
        if row is None:
            continue
        table.add_data(t, row["n"], row["a_mean"], row["b_mean"],
                        row["diff_mean_a_minus_b"], row["a_more_faithful_rate"])
    log_payload["per_threshold"] = table

    wandb.log(log_payload)
    wandb.summary["n_a_explained"] = comparison["n_a_explained"]
    wandb.summary["n_b_explained"] = comparison["n_b_explained"]
    wandb.summary["n_common"] = comparison["n_common"]
    wandb.finish()


if __name__ == "__main__":
    main()
