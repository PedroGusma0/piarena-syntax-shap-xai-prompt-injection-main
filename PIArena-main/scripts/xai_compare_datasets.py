#!/usr/bin/env python3
"""Compares the *global* token rankings (tokens that push promptguard's
P(malign) up the most) between result files of different datasets/attacks —
built for NotInject one/two/three vs. dolly_closed_qa direct/ignore, but
generic: every (group-A source, group-B source) pair is compared.

Per source: for every *blocked* sample (`detect_flag=True` — the only ones
every source explains; dolly attack runs explain nothing else), each token
(lowercased, surrounding punctuation stripped) collects its SHAP values; the
ranking is the signed mean SHAP per token (needs `--min-count` occurrences),
top-k with positive mean = "pushes towards malign".

Per pair: top-k overlap (shared tokens, Jaccard), and Spearman correlation of
the mean SHAP over every token that passes `--min-count` in both sources.

Writes under `--out-dir`:
  report.md                  — per-source stats + one section per pair
  jaccard_heatmap.png        — Jaccard of the top-k lists, group A x group B
  pair_<a>__<b>.png          — the two top-k rankings side by side, shared tokens highlighted

Reads only the raw result JSONs (numpy + matplotlib; no torch/shap).

Usage (from the repo root):
  python PIArena-main/scripts/xai_compare_datasets.py
  python PIArena-main/scripts/xai_compare_datasets.py --top-k 15 --min-count 3 \\
      --group-a one=path/one.json two=path/two.json --group-b direct=path/d.json
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
RED, GREY, HILITE = "#ff0d57", "#9aa0a6", "#ff9f1c"


def _find(pattern):
    hits = [f for f in glob.glob(os.path.join(glob.escape(REPO_ROOT), "**", pattern), recursive=True)
            if "xai_metrics" not in f and "tmp_attack_results" not in f and not f.endswith("_metrics.json")]
    return sorted(hits)


def default_groups():
    group_a = {}
    for subset in ("one", "two", "three"):
        hits = _find(f"notinject_{subset}-*-kernelshap-42.json")
        if hits:
            group_a[f"notinject_{subset}"] = hits[0]
    group_b = {}
    for attack in ("direct", "ignore"):
        # Prefer the captum run folder that holds both attacks side by side.
        hits = [h for h in _find(f"dolly_closed_qa-*-{attack}-promptguard-kernelshap-42.json")
                if "_none" not in h] or _find(f"dolly_closed_qa-*-{attack}-promptguard-kernelshap-42.json")
        if hits:
            group_b[f"dolly_{attack}"] = hits[0]
    return group_a, group_b


def parse_group(items):
    out = {}
    for it in items:
        label, _, path = it.partition("=")
        out[label] = path
    return out


def clean(tok):
    t = tok.replace("\n", " ").strip().lower()
    stripped = re.sub(r"^\W+|\W+$", "", t)
    return stripped or t or "·"


def aggregate(path):
    """({token: [shap values]}, n_blocked, n_total)"""
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    agg, n_blocked = defaultdict(list), 0
    for s in result.values():
        ctx = (s.get("xai_result") or {}).get("context")
        if not ctx or not ctx.get("values"):
            continue
        if not (s.get("defense_result") or {}).get("detect_flag"):
            continue
        n_blocked += 1
        for t, v in zip(ctx["tokens"], ctx["values"]):
            agg[clean(t)].append(float(v))
    return agg, n_blocked, len(result)


def ranking(agg, min_count):
    """{token: (mean_shap, count)} for tokens with >= min_count occurrences."""
    return {k: (float(np.mean(v)), len(v)) for k, v in agg.items() if len(v) >= min_count}


def top_k(rank, k):
    rows = sorted(((t, m, c) for t, (m, c) in rank.items() if m > 0), key=lambda r: -r[1])
    return rows[:k]


def spearman(rank_a, rank_b):
    shared = sorted(set(rank_a) & set(rank_b))
    if len(shared) < 3:
        return None, len(shared)

    def ranks(x):
        order = np.argsort(np.argsort(-np.asarray(x)))
        return order.astype(float)

    ra = ranks([rank_a[t][0] for t in shared])
    rb = ranks([rank_b[t][0] for t in shared])
    if ra.std() == 0 or rb.std() == 0:
        return None, len(shared)
    return float(np.corrcoef(ra, rb)[0, 1]), len(shared)


def plot_pair(label_a, rows_a, label_b, rows_b, shared, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.4 * max(len(rows_a), len(rows_b), 1) + 1.8))
    for ax, label, rows in ((axes[0], label_a, rows_a), (axes[1], label_b, rows_b)):
        rows = rows[::-1]
        y = np.arange(len(rows))
        colors = [HILITE if t in shared else RED for t, _, _ in rows]
        ax.barh(y, [m for _, m, _ in rows], color=colors, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{t} (n={c})" for t, _, c in rows], fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("mean SHAP of P(malign)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", linestyle=":", color="#bbb", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"{label_a} vs {label_b} — orange = token in both top-k lists ({len(shared)} shared)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(labels_a, labels_b, matrix, out_path):
    fig, ax = plt.subplots(figsize=(1.8 * len(labels_b) + 2.5, 0.9 * len(labels_a) + 1.8))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Reds")
    ax.set_xticks(range(len(labels_b)))
    ax.set_xticklabels(labels_b, rotation=20, ha="right")
    ax.set_yticks(range(len(labels_a)))
    ax.set_yticklabels(labels_a)
    for i in range(len(labels_a)):
        for j in range(len(labels_b)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("Jaccard of top-k lists (tokens pushing towards malign)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group-a", nargs="*", default=None, help="label=path ... (default: NotInject one/two/three)")
    ap.add_argument("--group-b", nargs="*", default=None, help="label=path ... (default: dolly_closed_qa direct/ignore)")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-count", type=int, default=3, help="min occurrences of a token in a source to be ranked")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "figs", "compare_notinject_dolly"))
    args = ap.parse_args()

    def_a, def_b = default_groups()
    group_a = parse_group(args.group_a) if args.group_a else def_a
    group_b = parse_group(args.group_b) if args.group_b else def_b
    if not group_a or not group_b:
        raise SystemExit(f"Missing sources (group A: {list(group_a)}, group B: {list(group_b)}).")
    os.makedirs(args.out_dir, exist_ok=True)

    sources = {}
    for label, path in {**group_a, **group_b}.items():
        agg, n_blocked, n_total = aggregate(path)
        rank = ranking(agg, args.min_count)
        sources[label] = {"path": path, "rank": rank, "top": top_k(rank, args.top_k),
                          "n_blocked": n_blocked, "n_total": n_total}
        print(f"{label}: {n_blocked}/{n_total} blocked, {len(rank)} tokens with n>={args.min_count}")

    lines = [f"# Comparação de rankings globais (top-{args.top_k}, n>={args.min_count})", "",
             "Alvo explicado: P(malign). Ranking: média do valor SHAP por token (minúsculas, sem pontuação nas pontas), "
             "só amostras bloqueadas; entram tokens com valor médio positivo (empurram para maligno).", "",
             "## Fontes", "", "| fonte | bloqueadas / total | tokens ranqueáveis | arquivo |", "|---|---|---|---|"]
    for label, s in sources.items():
        lines.append(f"| {label} | {s['n_blocked']} / {s['n_total']} | {len(s['rank'])} | `{os.path.relpath(s['path'], REPO_ROOT)}` |")

    labels_a, labels_b = list(group_a), list(group_b)
    jaccards = [[0.0] * len(labels_b) for _ in labels_a]
    lines += ["", "## Pares", "", "| A | B | compartilhados (top-k) | Jaccard | Spearman (tokens em ambos) |", "|---|---|---|---|---|"]
    details = []
    for i, la in enumerate(labels_a):
        for j, lb in enumerate(labels_b):
            sa, sb = sources[la], sources[lb]
            ta = {t for t, _, _ in sa["top"]}
            tb = {t for t, _, _ in sb["top"]}
            shared = ta & tb
            union = ta | tb
            jac = len(shared) / len(union) if union else 0.0
            jaccards[i][j] = jac
            rho, n_both = spearman(sa["rank"], sb["rank"])
            rho_s = "n/a" if rho is None else f"{rho:.2f} (n={n_both})"
            lines.append(f"| {la} | {lb} | {len(shared)} | {jac:.2f} | {rho_s} |")
            fig_name = f"pair_{la}__{lb}.png"
            plot_pair(la, sa["top"], lb, sb["top"], shared, os.path.join(args.out_dir, fig_name))
            details.append((la, lb, shared, sa, sb, fig_name))

    for la, lb, shared, sa, sb, fig_name in details:
        pos_a = {t: r + 1 for r, (t, _, _) in enumerate(sa["top"])}
        pos_b = {t: r + 1 for r, (t, _, _) in enumerate(sb["top"])}
        lines += ["", f"### {la} vs {lb}", "", f"![{la} vs {lb}]({fig_name})", ""]
        if shared:
            lines += [f"| token | rank em {la} | rank em {lb} |", "|---|---|---|"]
            for t in sorted(shared, key=lambda t: pos_a[t] + pos_b[t]):
                lines.append(f"| {t} | {pos_a[t]} | {pos_b[t]} |")
        else:
            lines.append("Nenhum token em comum entre os dois top-k.")
        only_a = [t for t, _, _ in sa["top"] if t not in shared][:8]
        only_b = [t for t, _, _ in sb["top"] if t not in shared][:8]
        lines += ["", f"Só em {la}: {', '.join(only_a) or '—'}", "", f"Só em {lb}: {', '.join(only_b) or '—'}"]

    plot_heatmap(labels_a, labels_b, jaccards, os.path.join(args.out_dir, "jaccard_heatmap.png"))
    lines.insert(2, "![jaccard](jaccard_heatmap.png)\n")
    report = os.path.join(args.out_dir, "report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
