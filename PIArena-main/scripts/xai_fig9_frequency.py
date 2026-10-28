"""Frequency plot of the highest-impact words, in the style of Figure 9 of XDAC
(Appendix E.2), one PNG + CSV per (dataset, defense, backend), aggregating every
attack of that dataset.

For each explained sample we take the top-K words by |SHAP| of
xai_result["context"] and split them by sign. The explained target is
P(malign), so SHAP > 0 pushes the defense towards MALIGN and SHAP < 0 towards
BENIGN. Each word is counted once per sample (document frequency); the 20 most
frequent words of each sign are drawn as two curves (x = frequency rank,
y = number of samples), with the word annotated next to each point.

Only samples with a non-null xai_result enter: with explain_scope=auto that is
every sample for attack=none but only the blocked ones otherwise.

Standalone (numpy + matplotlib only; does not import shap).

Usage (from the repo root):
  python PIArena-main/scripts/xai_fig9_frequency.py --roots kernelshap_squad_v2_piguard_direct
  python PIArena-main/scripts/xai_fig9_frequency.py --roots ~/Downloads/kernelshap_squad_v2_4attacks --top-k 10
"""
import argparse
import csv
import glob
import json
import os
import re
import string
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MALIGN_COLOR = "#d62728"
BENIGN_COLOR = "#1f4fd8"
STEM_RE = re.compile(r"^(?P<dataset>.+?)-(?P<model>.+)-(?P<attack>[a-z_]+)-(?P<defense>promptguard|piguard)-(?:kernelshap|ig)-\d+$")
STOPWORDS = frozenset(
    "a an the of to in on at by for with and or but if is are was were be been being it its this that these those "
    "as from into than then so not no do does did has have had i you he she we they them his her their our your "
    "my me us which who whom what when where how can could would should will may might also".split()
)


def discover(roots, only=None, xai_method="kernelshap"):
    found = []
    for root in roots:
        root = os.path.expanduser(root)
        for f in glob.glob(os.path.join(glob.escape(root), "**", f"*-{xai_method}-*.json"), recursive=True):
            parts = set(os.path.relpath(f, root).replace(os.sep, "/").split("/"))
            if {"xai_metrics", "tmp_attack_results"} & parts or f.endswith("_metrics.json"):
                continue
            if only and only not in f:
                continue
            found.append(f)
    return sorted(found)


def normalize(tok):
    t = tok.strip().strip(string.punctuation + "“”‘’…").lower()
    return t


def sample_words(ctx, top_k, drop_stopwords):
    """(set of malign words, set of benign words) among the top-k |SHAP| words."""
    toks, vals = ctx["tokens"], ctx["values"]
    order = sorted(range(len(vals)), key=lambda i: -abs(vals[i]))[:top_k]
    pos, neg = set(), set()
    for i in order:
        w = normalize(toks[i])
        if not w or (drop_stopwords and w in STOPWORDS):
            continue
        if vals[i] > 0:
            pos.add(w)
        elif vals[i] < 0:
            neg.add(w)
    return pos, neg


def aggregate(path, top_k, drop_stopwords, window_scope="flagged"):
    """(malign Counter, benign Counter, n_samples, backend)"""
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    pos_c, neg_c, n, backend = Counter(), Counter(), 0, None
    for s in result.values():
        ctx = (s.get("xai_result") or {}).get("context")
        if ctx and ctx.get("windowed") and window_scope != "all":
            # windowed run: keep only the flagged/injected windows (piarena/windowing.py)
            from piarena.windowing import flatten_windows
            thr = (s.get("defense_result") or {}).get("block_threshold", 0.5)
            ctx = flatten_windows(s["xai_result"], thr, window_scope)
        if not ctx or not ctx.get("values"):
            continue
        n += 1
        backend = backend or ctx.get("backend")
        pos, neg = sample_words(ctx, top_k, drop_stopwords)
        pos_c.update(pos)
        neg_c.update(neg)
    return pos_c, neg_c, n, backend


def plot(pos_c, neg_c, n_samples, out_path, title, n_rank=20, normalize_pct=False):
    series = [
        ("pushes towards MALIGN (SHAP > 0)", MALIGN_COLOR, pos_c.most_common(n_rank), 1),
        ("pushes towards BENIGN (SHAP < 0)", BENIGN_COLOR, neg_c.most_common(n_rank), -1),
    ]
    if not any(rows for _, _, rows, _ in series):
        return False
    scale = 100.0 / n_samples if normalize_pct else 1.0
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, color, rows, side in series:
        xs = list(range(1, len(rows) + 1))
        ys = [c * scale for _, c in rows]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.2, color=color, label=label)
        for i, (x, y, (w, _)) in enumerate(zip(xs, ys, rows)):
            dy = (8 + 9 * (i % 2)) * (1 if side > 0 else -1)
            ax.annotate(w, (x, y), textcoords="offset points", xytext=(0, dy), ha="center",
                        fontsize=7, color=color, rotation=45)
    ax.set_xlim(0, n_rank + 1)
    ax.set_xticks(range(0, n_rank + 1, 5))
    ax.set_xlabel("Order of Frequency")
    ax.set_ylabel("Frequency (% of samples)" if normalize_pct else "Frequency (samples)")
    ax.set_title(title, fontsize=9)
    ax.grid(linestyle=":", color="#bbb", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8)
    ax.margins(y=0.2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def write_csv(pos_c, neg_c, n_samples, out_path, n_rank=20):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["direction", "rank", "word", "n_samples", "pct_samples"])
        for direction, c in (("malign", pos_c), ("benign", neg_c)):
            for rank, (word, cnt) in enumerate(c.most_common(n_rank), 1):
                w.writerow([direction, rank, word, cnt, f"{100 * cnt / n_samples:.1f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xai-method", default="kernelshap", choices=["kernelshap", "ig"])
    ap.add_argument("--roots", nargs="*", default=None, help="default: every {xai-method}_* dir in the repo root")
    ap.add_argument("--only", default=None, help="substring filter on the result file path")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "figs", "fig9_frequency"))
    ap.add_argument("--top-k", type=int, default=10, help="highest-|SHAP| words kept per sample")
    ap.add_argument("--window-scope", choices=["all", "flagged", "injected"], default="flagged",
                    help="Only for windowed runs (promptguard_windowed): which windows feed the ranking: "
                         "flagged (P(malign) >= threshold, default) | injected | all.")
    ap.add_argument("--n-rank", type=int, default=20, help="words per curve")
    ap.add_argument("--drop-stopwords", action="store_true")
    ap.add_argument("--normalize", action="store_true", help="y axis in %% of samples instead of counts")
    args = ap.parse_args()

    roots = args.roots or sorted(glob.glob(os.path.join(glob.escape(REPO_ROOT), f"{args.xai_method}_*")))
    files = discover(roots, args.only, args.xai_method)
    print(f"{len(files)} result file(s)")

    groups = defaultdict(lambda: [Counter(), Counter(), 0, set(), []])  # dataset/defense/backend -> ...
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        m = STEM_RE.match(stem)
        dataset, defense, attack = (m["dataset"], m["defense"], m["attack"]) if m else (stem, "?", "?")
        pos_c, neg_c, n, backend = aggregate(f, args.top_k, args.drop_stopwords, args.window_scope)
        if not n:
            print(f"skip {stem}: 0 samples with XAI")
            continue
        g = groups[(dataset, defense, backend or "?")]
        g[0].update(pos_c)
        g[1].update(neg_c)
        g[2] += n
        g[4].append(f"{attack}={n}")

    for (dataset, defense, backend), (pos_c, neg_c, n, _, per_attack) in sorted(groups.items()):
        name = f"{dataset}_{defense}_{backend}"
        title = (f"{dataset} / {defense} / {backend}: top-{args.n_rank} high-impact words "
                 f"(top-{args.top_k} |SHAP| per sample, {n} samples; {', '.join(per_attack)})")
        png = os.path.join(args.out_dir, f"fig9_frequency_{name}.png")
        ok = plot(pos_c, neg_c, n, png, title, args.n_rank, args.normalize)
        if ok:
            write_csv(pos_c, neg_c, n, png[:-4] + ".csv", args.n_rank)
        print(("wrote " if ok else "skip (no words) ") + name)


if __name__ == "__main__":
    main()
