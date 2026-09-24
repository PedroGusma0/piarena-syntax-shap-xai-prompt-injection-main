#!/usr/bin/env python3
"""Compares Kernel SHAP's two computation backends — `captum.attr.KernelShap`
and the official `shap.KernelExplainer` — on the SAME samples, to check how
much the per-word Shapley values agree between two independent numerical
approximations of the same theoretical method (Lundberg & Lee 2017).

Reads two already-computed `main.py --xai kernelshap` result files (one run
with `--xai_backend captum`, one with `--xai_backend shap` — see
docs/xai/kernelshap.md's "Comparando backends" section for exact commands,
including why they must use different `--name` values to avoid `main.py`'s
result-filename collision, since `backend` is never part of that filename)
and compares each common sample's raw `xai_result["context"]["values"]`
directly. Pure JSON + numpy/scipy, no torch/transformers/captum/shap import
needed here — same "cheap to import" spirit as `piarena/xai/metrics.py` and
`scripts/xai_compare_fidelity.py` (whose structure this script mirrors,
adapted for backend-vs-backend value agreement instead of method-vs-method
Fidelity(t) comparison).

Alignment: only the intersection of sample indices BOTH files actually
explained (`xai_result` non-null) is compared — this should normally be the
full set the defense blocked in either file, since `PromptGuardDefense`'s
`detect_flag` doesn't depend on which XAI backend later explains it; any
mismatch is reported as a warning, not silently dropped. Samples whose
`tokens` differ between the two files (e.g. accidentally compared runs used
different `model_name`/`max_length`) are also excluded and counted.

Metrics computed per common, token-matching sample, over the two backends'
`values` vectors:
  - pearson_r      — linear correlation of the raw signed values.
  - spearman_rho    — rank correlation of |values| (same magnitude-rank
                      convention as `top_t_mask`/`injected_span_percentile`,
                      reused below from `piarena/xai/metrics.py`) — the
                      direct analogue of the removed `syntax` vs. `syntax-w`
                      `spearman()` metric (see that module's trailing
                      comment), this time correlating backend instead of
                      algorithm variant.
  - sign_agreement  — fraction of words where sign(value) agrees between
                      backends (a small |value| can still flip sign; Pearson/
                      Spearman alone don't surface that on their own).
  - topk_overlap[t] — Jaccard overlap between the two backends' top-t% most
                      important words (`top_t_mask`, reused, not
                      reimplemented).
  - injected_span_percentile_diff — difference between the two backends'
                      `injected_span_percentile` (also reused) — do they
                      disagree about whether the *injected* text specifically
                      stood out?

Writes, under `--out-dir`:
  - backend_agreement_metrics.json — raw per-sample + aggregate agreement data
  - backend_agreement_report.md    — run summary + agreement table
  - backend_agreement.png          — histogram of per-sample spearman_rho

Usage:
    python scripts/xai_compare_backends.py \\
        --captum-result results/evaluation_results/kernelshap_captum_pilot/squad_v2-...-kernelshap-42.json \\
        --shap-result   results/evaluation_results/kernelshap_shap_pilot/squad_v2-...-kernelshap-42.json \\
        --out-dir results_comparison/direct_backend_agreement
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

from piarena.xai.metrics import injected_span_percentile, top_t_mask


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captum-result", required=True, help="main.py --xai kernelshap --xai_backend captum output.")
    p.add_argument("--shap-result", required=True, help="main.py --xai kernelshap --xai_backend shap output.")
    p.add_argument("--attack", default=None,
                    help="Attack name, for the report header only — if omitted, parsed "
                         "best-effort from --captum-result's filename.")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5],
                    help="Top-t fractions of words (by |value|) for the topk_overlap metric.")
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
    main.py's `evaluation_result_path`) — only used when `--attack` wasn't
    passed. `llm_name` itself may contain '-', so this is inherently
    ambiguous without knowing the attack list; display convenience only."""
    if not result_path:
        return "unknown"
    stem = os.path.splitext(os.path.basename(result_path))[0]
    parts = stem.split("-")
    known_attacks = {"direct", "ignore", "completion", "character", "combined", "none"}
    for part in parts:
        if part in known_attacks:
            return part
    return "unknown"


def _pearson(a: np.ndarray, b: np.ndarray):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(pearsonr(a, b)[0])


def _spearman(a: np.ndarray, b: np.ndarray):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b)[0])


def _jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = int(np.sum(mask_a | mask_b))
    if union == 0:
        return 1.0
    return float(np.sum(mask_a & mask_b) / union)


def load_and_align(captum_path: str, shap_path: str):
    a_raw = load_json(captum_path)
    b_raw = load_json(shap_path)

    a_explained = {idx: row for idx, row in a_raw.items() if (row.get("xai_result") or {}).get("context")}
    b_explained = {idx: row for idx, row in b_raw.items() if (row.get("xai_result") or {}).get("context")}

    common_idx = sorted(set(a_explained) & set(b_explained), key=int)
    only_a = sorted(set(a_explained) - set(b_explained), key=int)
    only_b = sorted(set(b_explained) - set(a_explained), key=int)

    return a_raw, b_raw, a_explained, b_explained, common_idx, only_a, only_b


def sanity_check_backends(a_explained, b_explained, common_idx):
    a_wrong = [idx for idx in common_idx if a_explained[idx]["xai_result"]["context"].get("backend") != "captum"]
    b_wrong = [idx for idx in common_idx if b_explained[idx]["xai_result"]["context"].get("backend") != "shap"]
    return a_wrong, b_wrong


def sanity_check_tokens(a_explained, b_explained, common_idx):
    return [
        idx for idx in common_idx
        if a_explained[idx]["xai_result"]["context"]["tokens"] != b_explained[idx]["xai_result"]["context"]["tokens"]
    ]


def compare_sample(values_a, values_b, injected_span, thresholds) -> dict:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    pearson_r = _pearson(values_a, values_b)
    spearman_rho = _spearman(np.abs(values_a), np.abs(values_b))
    sign_agreement = float(np.mean(np.sign(values_a) == np.sign(values_b)))

    topk_overlap = {}
    for t in thresholds:
        mask_a = top_t_mask(values_a, t)
        mask_b = top_t_mask(values_b, t)
        topk_overlap[str(t)] = _jaccard(mask_a, mask_b)

    pct_a = injected_span_percentile(values_a, injected_span)
    pct_b = injected_span_percentile(values_b, injected_span)
    pct_diff = (pct_a - pct_b) if (pct_a is not None and pct_b is not None) else None

    return {
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "sign_agreement": sign_agreement,
        "topk_overlap": topk_overlap,
        "injected_span_percentile_a": pct_a,
        "injected_span_percentile_b": pct_b,
        "injected_span_percentile_diff": pct_diff,
    }


def _stats(values: list) -> dict | None:
    arr = np.array([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return None
    return {"n": len(arr), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def aggregate_metrics(per_sample: dict, thresholds) -> dict:
    per_threshold = {}
    for t in thresholds:
        per_threshold[str(t)] = _stats([s["topk_overlap"][str(t)] for s in per_sample.values()])

    return {
        "pearson_r": _stats([s["pearson_r"] for s in per_sample.values()]),
        "spearman_rho": _stats([s["spearman_rho"] for s in per_sample.values()]),
        "sign_agreement": _stats([s["sign_agreement"] for s in per_sample.values()]),
        "injected_span_percentile_diff": _stats([s["injected_span_percentile_diff"] for s in per_sample.values()]),
        "topk_overlap": per_threshold,
    }


def _render_report(attack: str, out: dict) -> str:
    meta, agg = out["meta"], out["aggregate"]

    lines = [
        f"# Backend agreement — captum vs. shap ({attack})",
        "",
        "## Run summary",
        "",
        f"- Attack: `{attack}`",
        f"- Captum result file: `{meta['captum_result_path']}`",
        f"- Shap result file:   `{meta['shap_result_path']}`",
        f"- Samples compared (common indices, minus token mismatches): {meta['n_compared']} "
        f"(captum explained {meta['n_captum_explained']}, shap explained {meta['n_shap_explained']}, "
        f"common {meta['n_common']})",
    ]
    if meta["n_token_mismatches"]:
        lines.append(f"  - ⚠️ {meta['n_token_mismatches']} common samples excluded for mismatched `tokens`.")
    if meta["captum_side_wrong_backend"] or meta["shap_side_wrong_backend"]:
        lines.append(
            f"  - ⚠️ {meta['captum_side_wrong_backend']}/{meta['n_common']} \"captum\"-side samples and "
            f"{meta['shap_side_wrong_backend']}/{meta['n_common']} \"shap\"-side samples recorded an "
            f"unexpected `backend` value — double-check the two input files actually come from different "
            f"backend runs (see sanity warnings printed at run time)."
        )

    lines += [
        "",
        f"## Agreement summary (mean ± std over {meta['n_compared']} samples)",
        "",
        "| metric | mean | std |",
        "|---|---|---|",
    ]
    for key in ("pearson_r", "spearman_rho", "sign_agreement", "injected_span_percentile_diff"):
        row = agg[key]
        lines.append(f"| {key} | n/a | n/a |" if row is None else f"| {key} | {row['mean']:.4f} | {row['std']:.4f} |")

    lines += [
        "",
        "## Top-k overlap (Jaccard) per threshold t",
        "",
        "| t | mean overlap |",
        "|---|---|",
    ]
    for t, row in agg["topk_overlap"].items():
        lines.append(f"| {t} | n/a |" if row is None else f"| {t} | {row['mean']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def _render_plot(per_sample: dict, out_dir: str):
    import matplotlib.pyplot as plt

    values = [s["spearman_rho"] for s in per_sample.values() if s["spearman_rho"] is not None]
    if not values:
        return
    fig, ax = plt.subplots()
    ax.hist(values, bins=20, range=(-1, 1))
    ax.set_xlabel("spearman_rho (per-sample, |value| ranking, captum vs. shap)")
    ax.set_ylabel("# samples")
    ax.set_title("Kernel SHAP backend agreement — per-sample Spearman rho")
    out_path = os.path.join(out_dir, "backend_agreement.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _log_to_wandb(args, attack: str, out: dict):
    """Logs the agreement summary + plot to their own W&B run
    (job_type="compare_backends") — makes "do the two backends agree?"
    navigable on wandb.ai, not just in the `.md` report, following the same
    pattern as `scripts/xai_compare_fidelity.py`'s `_log_to_wandb`."""
    os.environ.setdefault("WANDB_MODE", "offline")
    try:
        import wandb
    except ImportError as e:
        print(f"Skipping W&B logging (wandb not installed): {e}")
        return

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "piarena-xai-sweep"),
        group=attack,
        job_type="compare_backends",
        name=f"{attack}-captum_vs_shap-backend_agreement",
        config=out["meta"],
    )

    log_payload = {}
    plot_path = os.path.join(args.out_dir, "backend_agreement.png")
    if os.path.exists(plot_path):
        log_payload["backend_agreement"] = wandb.Image(plot_path)

    agg = out["aggregate"]
    table = wandb.Table(columns=["metric", "n", "mean", "std"])
    for key in ("pearson_r", "spearman_rho", "sign_agreement", "injected_span_percentile_diff"):
        row = agg[key]
        if row is not None:
            table.add_data(key, row["n"], row["mean"], row["std"])
    log_payload["agreement_summary"] = table

    topk_table = wandb.Table(columns=["t", "n", "mean", "std"])
    for t, row in agg["topk_overlap"].items():
        if row is not None:
            topk_table.add_data(t, row["n"], row["mean"], row["std"])
    log_payload["topk_overlap"] = topk_table

    wandb.log(log_payload)
    wandb.summary["n_compared"] = out["meta"]["n_compared"]
    wandb.finish()


def main():
    args = parse_args()
    a_raw, b_raw, a_explained, b_explained, common_idx, only_a, only_b = load_and_align(
        args.captum_result, args.shap_result,
    )

    print(f"Loaded {len(a_raw)} samples from --captum-result, {len(a_explained)} with a non-null xai_result.")
    print(f"Loaded {len(b_raw)} samples from --shap-result, {len(b_explained)} with a non-null xai_result.")
    print(f"Common indices (both explained): {len(common_idx)}")
    if only_a or only_b:
        print(
            f"⚠️  {len(only_a)} samples explained only in --captum-result, {len(only_b)} only in "
            f"--shap-result — excluded from comparison (unexpected: PromptGuardDefense's detect_flag "
            f"shouldn't depend on the XAI backend that later explains it)."
        )

    a_wrong_backend, b_wrong_backend = sanity_check_backends(a_explained, b_explained, common_idx)
    if a_wrong_backend:
        print(
            f"⚠️  Sanity check: {len(a_wrong_backend)}/{len(common_idx)} samples in --captum-result "
            f"have backend != \"captum\" (expected \"captum\")."
        )
    if b_wrong_backend:
        print(
            f"⚠️  Sanity check: {len(b_wrong_backend)}/{len(common_idx)} samples in --shap-result "
            f"have backend != \"shap\" (expected \"shap\"). This is expected for a self-comparison "
            f"test (same file passed twice) — not an error when intentional, but double-check your "
            f"file paths if this wasn't."
        )

    token_mismatches = sanity_check_tokens(a_explained, b_explained, common_idx)
    usable_idx = [idx for idx in common_idx if idx not in token_mismatches]
    if token_mismatches:
        print(
            f"⚠️  {len(token_mismatches)}/{len(common_idx)} common samples have mismatched `tokens` "
            f"between the two files (different model_name/max_length?) — excluded from comparison."
        )
    else:
        print(f"Sanity check: token sequences match on all {len(usable_idx)}/{len(common_idx)} common samples.")

    attack = args.attack or _guess_attack(args.captum_result)

    per_sample = {}
    for idx in usable_idx:
        ctx_a = a_explained[idx]["xai_result"]["context"]
        ctx_b = b_explained[idx]["xai_result"]["context"]
        per_sample[idx] = compare_sample(ctx_a["values"], ctx_b["values"], ctx_a.get("injected_span"), args.thresholds)

    aggregate = aggregate_metrics(per_sample, args.thresholds) if per_sample else {
        "pearson_r": None, "spearman_rho": None, "sign_agreement": None,
        "injected_span_percentile_diff": None,
        "topk_overlap": {str(t): None for t in args.thresholds},
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "meta": {
            "attack": attack,
            "captum_result_path": args.captum_result,
            "shap_result_path": args.shap_result,
            "thresholds": args.thresholds,
            "n_captum_explained": len(a_explained),
            "n_shap_explained": len(b_explained),
            "n_common": len(common_idx),
            "n_token_mismatches": len(token_mismatches),
            "n_compared": len(usable_idx),
            "captum_side_wrong_backend": len(a_wrong_backend),
            "shap_side_wrong_backend": len(b_wrong_backend),
        },
        "per_sample": per_sample,
        "aggregate": aggregate,
    }

    metrics_path = os.path.join(args.out_dir, "backend_agreement_metrics.json")
    save_json(out, metrics_path)
    print(f"Wrote {metrics_path}")

    report = _render_report(attack, out)
    report_path = os.path.join(args.out_dir, "backend_agreement_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {report_path}")

    try:
        _render_plot(per_sample, args.out_dir)
    except ImportError as e:
        print(f"Skipping plot (matplotlib not installed): {e}")

    if not args.no_wandb:
        _log_to_wandb(args, attack, out)


if __name__ == "__main__":
    main()
