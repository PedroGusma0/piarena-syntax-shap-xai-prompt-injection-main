"""Mapa de saliencia por janela: um grafico por janela (Kernel SHAP em janelas).

Para cada amostra e cada janela selecionada, uma figura com as palavras da
janela em ordem de leitura, cada uma com fundo colorido pelo seu valor SHAP
(φ): vermelho = empurra para malign, azul = empurra para benigno, intensidade
proporcional a |φ| (escala simetrica por figura). Palavras do `injected_span`
vem sublinhadas em vermelho. O cabecalho traz P(malign), base, Σφ e os rotulos
[INJ] (a janela contem a injecao) e [BLOQUEIO] (P(malign) >= limiar).

Le `xai_result["windows"]` (runs com `--defense promptguard_windowed
--window_tokens 512 --xai kernelshap`). Matplotlib puro, sem shap.plots.text.

Uso:
    python scripts/xai_window_saliency.py --result <json> --idx 0 1 2
    python scripts/xai_window_saliency.py --result <json> --max-samples 5 --windows all
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent.parent
_ATTACKS = ("direct", "ignore", "completion", "combined", "character", "none")
FONT = 9
FIG_W_IN = 13.0
DPI = 130
PAD_X, PAD_Y = 4, 3  # px around each word box


def _parts(path):
    return os.path.splitext(os.path.basename(path))[0].split("-")


def layout(words, max_px, renderer, line_h):
    """Greedy line wrapping. Returns [(x, y_line_idx, width)] per word, n_lines."""
    pos, x, line = [], 0.0, 0
    for w in words:
        t = plt.Text(0, 0, w, fontsize=FONT, family="DejaVu Sans")
        t.set_figure(plt.gcf())
        width = t.get_window_extent(renderer).width + 2 * PAD_X + 4
        if x + width > max_px and x > 0:
            x, line = 0.0, line + 1
        pos.append((x, line, width))
        x += width
    return pos, line + 1


def render_window(win, sample_title, thr, out_path):
    words = [t.strip() or "·" for t in win["tokens"]]
    values = np.asarray(win["values"], dtype=float)
    if not len(words) or len(words) != len(values):
        return False
    vmax = float(np.abs(values).max()) or 1e-9
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = matplotlib.colormaps["RdBu_r"]

    span = win.get("injected_span")
    inj = np.zeros(len(words), dtype=bool)
    if span:
        inj[span["start_token"]:span["end_token"]] = True

    width_px = FIG_W_IN * DPI
    tmp = plt.figure(figsize=(FIG_W_IN, 2), dpi=DPI)
    renderer = tmp.canvas.get_renderer()
    line_h = FONT * 1.9 * DPI / 72
    pos, n_lines = layout(words, width_px - 30, renderer, line_h)
    plt.close(tmp)

    head_px, foot_px = 90, 70
    height_px = head_px + n_lines * line_h + foot_px
    fig = plt.figure(figsize=(FIG_W_IN, height_px / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(height_px, 0)
    ax.axis("off")

    tags = ("[INJ]" if win.get("contains_injection") else "") + ("[BLOQUEIO]" if win["p_malign"] >= thr else "")
    ax.text(15, 22, f"{sample_title}", fontsize=10, fontweight="bold", va="center")
    ax.text(15, 50, (f"Janela {win['window_idx']}  ·  P(malign)={win['p_malign']:.3f}  ·  base={win['base_value']:.3f}"
                     f"  ·  Σφ={values.sum():+.3f}  ·  {len(words)} palavras  {tags}"),
            fontsize=10, va="center", color="#b30000" if tags else "#222222")

    for i, (w, v) in enumerate(zip(words, values)):
        x, ln, wd = pos[i]
        x += 15
        y = head_px + ln * line_h
        strength = abs(norm(v) - 0.5) * 2  # 0 (neutral) .. 1
        ax.add_patch(plt.Rectangle((x, y - line_h * 0.42), wd - 2, line_h * 0.84,
                                   facecolor=cmap(norm(v)), edgecolor="none", alpha=0.35 + 0.65 * strength))
        ax.text(x + PAD_X + 2, y, w, fontsize=FONT, va="center", color="white" if strength > 0.75 else "black")
        if inj[i]:
            ax.plot([x + 1, x + wd - 3], [y + line_h * 0.42, y + line_h * 0.42], color="#d62728", lw=2.2)

    cax = fig.add_axes([0.62, 0.45 * foot_px / height_px, 0.33, 0.018])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("φ (vermelho: → malign · azul: → benigno)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if inj.any():
        ax.text(15, height_px - 22, "sublinhado vermelho = palavras da injeção", fontsize=8, color="#d62728", va="center")

    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return True


def select(entries, thr, mode):
    keep = []
    for e in entries:
        flagged = e["p_malign"] >= thr
        injected = bool(e.get("contains_injection"))
        if mode == "all" or (mode == "flagged" and flagged) or (mode == "injected" and injected) \
                or (mode == "flagged+injected" and (flagged or injected)):
            keep.append(e)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--idx", type=int, nargs="*", default=None)
    ap.add_argument("--max-samples", type=int, default=5)
    ap.add_argument("--windows", choices=["flagged", "injected", "flagged+injected", "all"], default="flagged+injected")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <result dir>/xai_metrics/plots/<result stem>/window_saliency")
    args = ap.parse_args()

    result = json.load(open(args.result, encoding="utf-8"))
    parts = _parts(args.result)
    dataset = parts[0]
    attack = next((p for p in parts if p in _ATTACKS), "attack")
    out_dir = args.out_dir or (args.result.parent / "xai_metrics" / "plots" / args.result.stem / "window_saliency")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.idx:
        idxs = [str(i) for i in args.idx]
    else:  # first samples that actually have windowed XAI
        idxs = [i for i, s in result.items() if (s.get("xai_result") or {}).get("windows")][: args.max_samples]

    n_fig = 0
    for idx in idxs:
        s = result.get(idx) or {}
        entries = (s.get("xai_result") or {}).get("windows")
        if not entries:
            print(f"[skip] amostra {idx}: sem xai_result.windows (amostra nao explicada ou run sem janelas)")
            continue
        thr = (s.get("defense_result") or {}).get("block_threshold", 0.5)
        title = f"{dataset} · ataque {attack} · amostra {idx}" + (f" · {s['category']}" if s.get("category") else "")
        for e in select(entries, thr, args.windows):
            out = out_dir / f"idx{idx}_win{e['window_idx']}.png"
            if render_window(e, title, thr, out):
                n_fig += 1
                print(f"[ok] {out}")
    print(f"{n_fig} figura(s) em {out_dir}")


if __name__ == "__main__":
    main()
