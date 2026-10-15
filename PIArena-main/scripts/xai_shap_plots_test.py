"""Exploratory test: official `shap.plots.bar` (global + per-sample local) from
a raw PIArena result JSON — no inference, no matplotlib-by-hand plotting.

Input: the raw result file written by main.py (`<...>-kernelshap-42.json`), NOT
the `_metrics.json` (which only has scalars). Needs `xai_result["context"]` with
`tokens`, `values`, `base_value`, `full_value`, `p_malign`, `injected_span`.

Output, all inside `<result dir>/xai_metrics/plots/<result-stem>/` (the same
per-attack folder that holds saliency_maps.html; override base with --out-dir):
  bar_global.png     — mean |SHAP| per token across explained samples
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
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piarena.xai.metrics import injected_top_mass_precision  # noqa: E402


def _clean(tok: str) -> str:
    t = tok.replace("\n", "\\n").strip()
    return t if t else "·"


_XAI_METHODS = ("syntaxshap", "kernelshap")


def _defense_from_filename(path):
    """`{dataset}-{llm}-{attack}-{defense}-{xai}-{seed}.json` -> the defense part
    (fallback for result files that predate the per-sample `defense` field)."""
    parts = os.path.splitext(os.path.basename(path))[0].split("-")
    for i, part in enumerate(parts):
        if part in _XAI_METHODS and i > 0:
            return parts[i - 1]
    return None


def load_blocked(path, include_unblocked=True):
    """{idx: dict} of samples with an explained context.

    Which samples have one is decided by the XAI method's `explain_scope` (an
    unexplained sample has `xai_result=None`), so by default every explained
    sample is kept, blocked or not; `include_unblocked=False` keeps only the
    ones the defense blocked. Each entry carries the `defense`/`xai_model` the
    explanation is about."""
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    file_defense = _defense_from_filename(path)
    out = {}
    for idx, s in result.items():
        # xai_result is explicit None (not absent) for non-explained samples
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
            "defense": s.get("defense") or file_defense,
            "xai_model": s.get("xai_model"),
        }
    return out


def defense_label(data):
    """'piguard (leolee99/PIGuard)'-style label for titles, from the first sample."""
    first = next(iter(data.values()), {})
    defense, model = first.get("defense"), first.get("xai_model")
    if defense and model:
        return f"{defense} ({model})"
    return defense or "unknown defense"


def span_mask(d):
    m = np.zeros(len(d["values"]), dtype=bool)
    if d["span"]:
        m[d["span"]["start_token"]:d["span"]["end_token"]] = True
    return m


def span_alignment(d, mass=0.5):
    """injected_top_mass_precision of this sample (piarena/xai/metrics.py);
    None when there is no injected span."""
    return injected_top_mass_precision(d["values"], d["span"], mass)


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
        pct = span_alignment(d)
        status = "BLOQUEADO" if d.get("blocked") else "NÃO BLOQUEADO"
        head = (f"<h3>Sample {i} — {status} — defense={html.escape(defense_label({i: d}))} — category={html.escape(str(d['category']))}, "
                f"predicted_label={d['label']}, P(benign)={d.get('p_benign')}, P(malign)={d['p_malign']}, "
                f"alignment (top-mass precision)={'n/a' if pct is None else f'{pct:.2f}'}</h3>")
        blocks.append(head + body)
        n += 1
    page = ("<html><head><meta charset='utf-8'><title>local shap.plots.bar per explained sample</title></head><body>"
            f"<h2>Defense explained: {html.escape(defense_label(data))}</h2>"
            "<p>▶ = token inside the injected span. Bars: top tokens by |value|; rest grouped.</p>"
            + "\n".join(blocks) + "</body></html>")
    path = os.path.join(local_dir, "saliency_bars.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {path} ({n} explained sample(s))")


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
    plt.title(f"{defense_label(data)} — mean |SHAP| per token (n>={min_count}, {len(data)} explained samples)", fontsize=9)
    _save(os.path.join(out_dir, "bar_global.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="raw result JSON (not *_metrics.json)")
    ap.add_argument("--out-dir", default=None, help="default: <result dir>/xai_metrics")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-html-samples", type=int, default=None)
    ap.add_argument("--min-count", type=int, default=5,
                    help="bar_global.png only ranks tokens seen at least this many times. Default 5 (drops one-off "
                         "tokens); use 1 when each word appears few times, e.g. NotInject (every trigger word is "
                         "nearly unique, so 5 hides all of them).")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.result))[0]
    out = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.result)), "xai_metrics")
    plots_dir = os.path.join(out, "plots")
    local_dir = os.path.join(plots_dir, stem)
    os.makedirs(local_dir, exist_ok=True)

    data = load_blocked(args.result)
    print(f"{len(data)} explained sample(s) with XAI values ({defense_label(data)})")
    if not data:
        return

    # additivity sanity check: base + sum(values) ~ full_value
    errs = [abs(d["base"] + d["values"].sum() - d["full"]) for d in data.values() if d["full"] is not None]
    if errs:
        print(f"additivity |base+sum-full|: max={max(errs):.2e} mean={np.mean(errs):.2e}")

    _try("bar_global", lambda: plot_bar_global(data, local_dir, min_count=args.min_count))
    _try("html", lambda: write_saliency_bars_html(data, local_dir, args.top_k, args.max_html_samples))


if __name__ == "__main__":
    main()
