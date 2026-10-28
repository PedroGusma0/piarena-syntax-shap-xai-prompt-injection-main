"""Perfil de anomalia por janela (Prompt-Guard em janelas de 512 tokens).

Para cada amostra, um grafico de barras: uma barra por janela, com o P(malign)
que o Prompt-Guard deu aquela janela (`defense_result.window_p_malign`, escrito
por `--defense promptguard_windowed`). Nao precisa de XAI: funciona para todas
as amostras, bloqueadas ou nao.

  * barra vermelha = janela que contem a injecao (verdade-terreno, por offsets);
    barra azul = janela benigna
  * linha tracejada = limiar de bloqueio; linha verde = media basal (janelas sem injecao)
  * rotulos: [INJ] = a janela contem a injecao
             [BLOQUEIO] = P(malign) >= limiar, ou seja, disparou o bloqueio
             (a de maior P(malign) leva [BLOQUEIO *] e a anotacao de score)

Uso:
    python scripts/xai_window_anomaly.py --result results/evaluation_results/<name>/<arquivo>.json --idx 0 1 2
    python scripts/xai_window_anomaly.py --result <atacado>.json --clean-result <none>.json --idx 0   # 2 paineis (A: sem ataque, B: atacado)
    python scripts/xai_window_anomaly.py --result <atacado>.json --max-samples 10 --with-shap
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLUE, RED, GREEN, GRAY = "#1f77b4", "#d62728", "#2ca02c", "#888888"
_ATTACKS = ("direct", "ignore", "completion", "combined", "character", "none")


def _parts(path):
    return os.path.splitext(os.path.basename(path))[0].split("-")


def guess_attack(path):
    return next((p for p in _parts(path) if p in _ATTACKS), None)


def guess_dataset(path):
    return _parts(path)[0]


def attack_prefix(attack):
    """Fixed string a heuristic attack prepends to the injected task ('' if unknown)."""
    if not attack or attack == "none":
        return ""
    try:  # imports piarena.attacks (heavy); fall back to bare-task matching if unavailable
        from piarena.xai.kernelshap.classifier_explainer import _attack_prefixes
        return _attack_prefixes().get(attack, "")
    except Exception:
        return ""


def window_info(sample, prefix, with_injection=True):
    """Per-window dicts {p, start, end, inj, block, text} from defense_result, or None."""
    dr = sample.get("defense_result") or {}
    scores, spans = dr.get("window_p_malign"), dr.get("window_spans")
    if not scores or not spans:
        return None
    thr = dr.get("block_threshold", 0.5)
    context = sample.get("injected_context") or sample.get("context") or ""
    injection = None
    if with_injection and sample.get("injected_task"):
        task = sample["injected_task"]
        start = context.find(prefix + task) if prefix else -1
        if start != -1:
            injection = (start, start + len(prefix) + len(task))
        else:
            start = context.find(task)
            injection = (start, start + len(task)) if start != -1 else None
    out = []
    for p, (a, b) in zip(scores, spans):
        out.append({
            "p": float(p), "start": a, "end": b,
            "inj": bool(injection) and a < injection[1] and b > injection[0],
            "block": p >= thr,
            "text": " ".join(context[a:b].split())[:16],
        })
    return out, thr


def draw(ax, wins, thr, title, ymax=1.32):
    n = len(wins)
    x = np.arange(n)
    colors = [RED if w["inj"] else BLUE for w in wins]
    ax.bar(x, [w["p"] for w in wins], 0.6, color=colors, edgecolor="k", linewidth=0.6)
    top = int(np.argmax([w["p"] for w in wins]))
    for i, w in enumerate(wins):
        ax.text(i, w["p"] + 0.012, f"{w['p']:.2f}", ha="center", va="bottom", fontsize=8,
                color=RED if w["inj"] else "#333333")
    basal = [w["p"] for w in wins if not w["inj"]]
    m = float(np.mean(basal)) if basal else None
    ax.axhline(thr, color="k", ls="--", lw=1.2, label=f"Limiar de bloqueio ({thr:g})")
    if m is not None:
        ax.axhline(m, color=GREEN, ls="-.", lw=1.5, label=f"Média basal ({m:.2f})")
    if wins[top]["block"]:
        ratio = f" ({wins[top]['p'] / m:.1f}x basal)" if m and m >= 1e-3 else ""
        ax.annotate(f"[BLOQUEIO ★]\nScore: {wins[top]['p']:.2f}{ratio}", (top, wins[top]["p"]),
                    xytext=(0, 26), textcoords="offset points", ha="center", fontsize=9, color=RED,
                    fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED),
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fde8e8", ec=RED))
    labels = []
    for i, w in enumerate(wins):
        tags = ("[INJ]" if w["inj"] else "") + ("[BLOQUEIO]" if w["block"] else "")
        labels.append(f"Janela {i}\n({w['text']}…)" + (f"\n{tags}" if tags else ""))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35 if n > 3 else 0, ha="right" if n > 3 else "center", fontsize=7)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, 1.01, 0.2))  # probabilities only; the headroom above 1.0 is for the annotation
    ax.set_ylabel("P(malign) da janela", fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)


def shap_figure(sample, path, k=12):
    """Top-k palavras por |φ| na janela de maior P(malign) (precisa de xai_result.windows)."""
    wins = (sample.get("xai_result") or {}).get("windows")
    if not wins:
        return False
    w = max(wins, key=lambda e: e["p_malign"])
    vals = np.asarray(w["values"], dtype=float)
    if not len(vals):
        return False
    order = np.argsort(-np.abs(vals))[:k][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(order) + 1.6))
    ax.barh([w["tokens"][i].strip()[:24] for i in order], vals[order],
            color=[RED if vals[i] > 0 else BLUE for i in order])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("valor SHAP (φ) — vermelho: empurra para malign")
    ax.set_title(f"Janela {w['window_idx']} (P(malign)={w['p_malign']:.2f}"
                 f"{', [INJ]' if w.get('contains_injection') else ''}) — top {len(order)} palavras", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", type=Path, required=True, help="JSON de resultado (run com promptguard_windowed)")
    ap.add_argument("--clean-result", type=Path, default=None, help="JSON da run --attack none (painel A)")
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="indices das amostras (padrao: as primeiras)")
    ap.add_argument("--max-samples", type=int, default=5)
    ap.add_argument("--with-shap", action="store_true", help="figura extra: top palavras por φ na janela de maior P(malign)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <result dir>/xai_metrics/plots/<result stem>/window_anomaly")
    args = ap.parse_args()

    result = json.load(open(args.result, encoding="utf-8"))
    clean = json.load(open(args.clean_result, encoding="utf-8")) if args.clean_result else None
    attack = guess_attack(args.result)
    dataset = guess_dataset(args.result)
    prefix = attack_prefix(attack)
    idxs = [str(i) for i in args.idx] if args.idx else [i for i in result][: args.max_samples]
    out_dir = args.out_dir or (args.result.parent / "xai_metrics" / "plots" / args.result.stem / "window_anomaly")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in idxs:
        sample = result.get(idx)
        info = window_info(sample, prefix) if sample else None
        if info is None:
            print(f"[skip] amostra {idx}: sem defense_result.window_p_malign (rode com --defense promptguard_windowed)")
            continue
        wins, thr = info
        cat = sample.get("category", "")
        title_b = f"{dataset} · ataque {attack} · amostra {idx}" + (f" · {cat}" if cat else "")
        clean_info = window_info(clean[idx], "", with_injection=False) if clean and idx in clean else None
        if clean_info:
            fig, (axa, axb) = plt.subplots(
                1, 2, figsize=(max(14, 0.9 * (len(clean_info[0]) + len(wins))), 5.6), sharey=True,
                gridspec_kw={"width_ratios": [len(clean_info[0]), len(wins)]},
            )
            draw(axa, clean_info[0], clean_info[1], f"A) Documento original benigno (sem ataque)")
            draw(axb, wins, thr, f"B) Documento injetado ({attack})")
            fig.suptitle(title_b, fontsize=11, fontweight="bold")
        else:
            fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(wins)), 5.6))
            draw(ax, wins, thr, title_b)
        fig.tight_layout()
        out = out_dir / f"{dataset}-{attack}-idx{idx}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[ok] {out}")
        if args.with_shap and shap_figure(sample, out_dir / f"{dataset}-{attack}-idx{idx}-shap.png"):
            print(f"[ok] {out_dir / f'{dataset}-{attack}-idx{idx}-shap.png'}")


if __name__ == "__main__":
    main()
