"""Exploratory test: official `shap.plots.bar` (global + per-sample local) from
a raw PIArena result JSON — no inference, no matplotlib-by-hand plotting.

Input: the raw result file written by main.py (`<...>-kernelshap-42.json`), NOT
the `_metrics.json` (which only has scalars). Needs `xai_result["context"]` with
`tokens`, `values`, `base_value`, `full_value`, `p_malign`, `injected_span`.

Output, all inside `<result dir>/xai_metrics/plots/<result-stem>/` (the same
per-attack folder that holds saliency_maps.html; override base with --out-dir):
  bar_global.png     — mean |SHAP| per token across blocked samples
  saliency_bars.html — one local shap.plots.bar per blocked sample

Usage:
  python scripts/xai_shap_plots_test.py --result <results>/<stem>.json
"""
import argparse
import base64
import html
import io
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402


def _clean(tok: str) -> str:
    t = tok.replace("\n", "\\n").strip()
    return t if t else "·"


def load_blocked(path, include_unblocked=None):
    """{idx: dict} of samples with an explained context that the defense blocked.

    `include_unblocked` also keeps samples the defense let through; defaults to
    True for `--attack none` result files (main.py explains every sample
    there), False otherwise."""
    if include_unblocked is None:
        include_unblocked = "-none-" in os.path.basename(path)
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    out = {}
    for idx, s in result.items():
        # xai_result is explicit None (not absent) for non-blocked samples
        ctx = (s.get("xai_result") or {}).get("context")
        if not ctx or not ctx.get("values"):
            continue
        blocked = bool((s.get("defense_result") or {}).get("detect_flag"))
        if not blocked and not include_unblocked:
            continue
        out[idx] = {
            "values": np.asarray(ctx["values"], dtype=float),
            "tokens": ctx["tokens"],
            "base": float(ctx.get("base_value") or 0.0),
            "full": ctx.get("full_value"),
            "p_malign": ctx.get("p_malign"),
            "p_benign": ctx.get("p_benign", None if ctx.get("p_malign") is None else 1.0 - ctx["p_malign"]),
            "blocked": blocked,
            "label": ctx.get("predicted_label"),
            "span": ctx.get("injected_span"),
            "category": s.get("category"),
        }
    return out


def span_mask(d):
    m = np.zeros(len(d["values"]), dtype=bool)
    if d["span"]:
        m[d["span"]["start_token"]:d["span"]["end_token"]] = True
    return m


def span_percentile(d):
    """Mean |value|-rank percentile of the injected span's tokens (same idea as
    piarena/xai/metrics.py::injected_span_percentile)."""
    m = span_mask(d)
    if not m.any():
        return None
    a = np.abs(d["values"])
    ranks = a.argsort().argsort() / max(len(a) - 1, 1)
    return float(ranks[m].mean())


def explanation(d, mark_span=True):
    m = span_mask(d)
    names = [("▶ " if (mark_span and m[i]) else "") + _clean(t) for i, t in enumerate(d["tokens"])]
    # no `data`: with data==names the labels render as "tok = tok"
    return shap.Explanation(values=d["values"], base_values=d["base"], feature_names=names)


def _fig_bytes(dpi=90):
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close("all")
    return buf.getvalue()


def _save(path, dpi=130):
    plt.gcf().savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print("wrote", path)


def _try(name, fn):
    try:
        fn()
    except Exception as e:  # keep going: one failing plot shouldn't kill the rest
        plt.close("all")
        print(f"[{name}] FAILED: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- local plots
def write_saliency_bars_html(data, local_dir, top_k=15, max_samples=None):
    blocks, n = [], 0
    for i, d in data.items():
        if max_samples is not None and n >= max_samples:
            break
        try:
            plt.figure()
            shap.plots.bar(explanation(d), max_display=top_k, show=False)
            png = base64.b64encode(_fig_bytes()).decode()
            body = f'<img src="data:image/png;base64,{png}">'
        except Exception as e:
            plt.close("all")
            body = f"<p>(failed to render: {html.escape(str(e))})</p>"
        pct = span_percentile(d)
        status = "BLOQUEADO" if d.get("blocked") else "NÃO BLOQUEADO"
        head = (f"<h3>Sample {i} — {status} — category={html.escape(str(d['category']))}, "
                f"predicted_label={d['label']}, P(benign)={d.get('p_benign')}, P(malign)={d['p_malign']}, "
                f"span percentile={'n/a' if pct is None else f'{pct:.2f}'}</h3>")
        blocks.append(head + body)
        n += 1
    page = ("<html><head><meta charset='utf-8'><title>local shap.plots.bar per blocked sample</title></head><body>"
            "<p>▶ = token inside the injected span. Bars: top tokens by |value|; rest grouped.</p>"
            + "\n".join(blocks) + "</body></html>")
    path = os.path.join(local_dir, "saliency_bars.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {path} ({n} blocked sample(s))")


# --------------------------------------------------------------- global plots
def plot_bar_global(data, out_dir, top_k=20, min_count=5):
    agg = defaultdict(list)
    for d in data.values():
        for t, v in zip(d["tokens"], d["values"]):
            agg[_clean(t).lower()].append(abs(v))
    rows = [(k, float(np.mean(v)), len(v)) for k, v in agg.items() if len(v) >= min_count]
    rows.sort(key=lambda r: -r[1])
    rows = rows[:top_k]
    names = [f"{k} (n={c})" for k, _, c in rows]
    e = shap.Explanation(values=np.array([m for _, m, _ in rows]), feature_names=names)
    plt.figure()
    shap.plots.bar(e, max_display=top_k, show=False)
    plt.title(f"mean |SHAP| per token (n>={min_count}, {len(data)} blocked samples)", fontsize=9)
    _save(os.path.join(out_dir, "bar_global.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="raw result JSON (not *_metrics.json)")
    ap.add_argument("--out-dir", default=None, help="default: <result dir>/xai_metrics")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-html-samples", type=int, default=None)
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.result))[0]
    out = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.result)), "xai_metrics")
    plots_dir = os.path.join(out, "plots")
    local_dir = os.path.join(plots_dir, stem)
    os.makedirs(local_dir, exist_ok=True)

    data = load_blocked(args.result)
    print(f"{len(data)} blocked sample(s) with XAI values")
    if not data:
        return

    # additivity sanity check: base + sum(values) ~ full_value
    errs = [abs(d["base"] + d["values"].sum() - d["full"]) for d in data.values() if d["full"] is not None]
    if errs:
        print(f"additivity |base+sum-full|: max={max(errs):.2e} mean={np.mean(errs):.2e}")

    _try("bar_global", lambda: plot_bar_global(data, local_dir))
    _try("html", lambda: write_saliency_bars_html(data, local_dir, args.top_k, args.max_html_samples))


if __name__ == "__main__":
    main()
