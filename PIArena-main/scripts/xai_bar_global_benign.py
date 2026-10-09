"""Global bar plot of the tokens that most pushed the promptguard decision
towards BENIGN, one PNG per (dataset, attack) result file, in blue.

The explained target is P(malign) = P(INJECTION)+P(JAILBREAK), so P(BENIGN) =
1 - P(malign) and a token's contribution to BENIGN is exactly -SHAP value.
Per token string we average that contribution over every occurrence in the
blocked samples (signed mean, not |.|) and keep the top-k with the largest
positive mean.

Standalone (numpy + matplotlib only; does not import shap).

Usage (from the repo root):
  python PIArena-main/scripts/xai_bar_global_benign.py
  python PIArena-main/scripts/xai_bar_global_benign.py --only dolly_closed_qa --out-dir figs/bar_global_benign
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BLUE = "#008bfb"  # shap's own "negative" blue
STEM_RE = re.compile(r"^(?P<dataset>.+?)-(?P<model>.+)-(?P<attack>[a-z_]+)-promptguard-kernelshap-42$")


def discover(roots, only=None):
    found = []
    for root in roots:
        for f in glob.glob(os.path.join(glob.escape(root), "**", "*-kernelshap-42.json"), recursive=True):
            parts = os.path.relpath(f, REPO_ROOT).replace(os.sep, "/").split("/")
            if {"xai_metrics", "tmp_attack_results", "shap"} & set(parts) or f.endswith("_metrics.json"):
                continue
            if only and only not in "/".join(parts):
                continue
            found.append(f)
    return sorted(found)


def clean(tok):
    t = tok.replace("\n", "\\n").strip()
    return t if t else "·"


def aggregate(path):
    """({token: [benign contributions]}, n_blocked_samples)"""
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    agg, n = defaultdict(list), 0
    for s in result.values():
        ctx = (s.get("xai_result") or {}).get("context")
        if not ctx or not ctx.get("values"):
            continue
        if not (s.get("defense_result") or {}).get("detect_flag"):
            continue
        n += 1
        for t, v in zip(ctx["tokens"], ctx["values"]):
            agg[clean(t).lower()].append(-float(v))
    return agg, n


def plot(agg, n_samples, out_path, top_k=20, min_count=5):
    rows = [(k, float(np.mean(v)), len(v)) for k, v in agg.items() if len(v) >= min_count]
    rows = [r for r in rows if r[1] > 0]
    rows.sort(key=lambda r: -r[1])
    rows = rows[:top_k]
    if not rows:
        return False
    rows = rows[::-1]  # barh draws bottom-up
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.6))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=BLUE, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k} (n={c})" for k, _, c in rows], fontsize=10)
    xmax = max(vals)
    for yi, v in zip(y, vals):
        ax.text(v + xmax * 0.01, yi, f"+{v:.2g}", color=BLUE, va="center", fontsize=10)
    ax.set_xlim(0, xmax * 1.12)
    ax.set_xlabel("mean contribution to BENIGN  (−SHAP of P(malign))")
    ax.set_title(f"tokens contributing most to BENIGN (n>={min_count}, {n_samples} blocked samples)", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", linestyle=":", color="#bbb", linewidth=0.6)
    ax.set_axisbelow(True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=None, help="default: every kernelshap_* dir in the repo root")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "figs", "bar_global_benign"))
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-count", type=int, default=5)
    args = ap.parse_args()

    roots = args.roots or sorted(glob.glob(os.path.join(glob.escape(REPO_ROOT), "kernelshap_*")))
    files = discover(roots, args.only)
    print(f"{len(files)} result file(s)")
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        m = STEM_RE.match(stem)
        name = f"{m['dataset']}_{m['attack']}" if m else stem
        agg, n = aggregate(f)
        if not n:
            print(f"skip {name}: 0 blocked samples with XAI")
            continue
        ok = plot(agg, n, os.path.join(args.out_dir, f"bar_global_benign_{name}.png"), args.top_k, args.min_count)
        print(("wrote " if ok else "skip (no positive-BENIGN tokens) ") + name)


if __name__ == "__main__":
    main()
