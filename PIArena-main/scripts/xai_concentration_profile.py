"""Perfil de evidência-malign por janelas de palavras: none x direct x combined.

Para uma mesma amostra, divide o contexto em janelas fixas de N palavras e
plota, por janela, a soma dos valores de Shapley positivos (só tokens que
empurram para malign). Janelas que intersectam o `injected_span` ficam em
vermelho. Sem segmentação em sentenças.

Uso:
    python scripts/xai_concentration_profile.py --idx 0
    python scripts/xai_concentration_profile.py \\
        --none-json ../kernelshap_dolly_closed_qa_captum_none/<...>-none-<...>.json \\
        --attack-dir ../kernelshap_dolly_closed_qa_captum/kernelshap_dolly_closed_qa_captum
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NONE = ROOT / "kernelshap_dolly_closed_qa_captum_none"
DEFAULT_ATTACK = ROOT / "kernelshap_dolly_closed_qa_captum" / "kernelshap_dolly_closed_qa_captum"
SCENARIOS = ["none", "direct", "combined"]
BLUE, RED, GRAY = "#1f77b4", "#d62728", "#999999"


def find_result(directory: Path, scenario: str) -> Path:
    hits = sorted(directory.glob(f"*-{scenario}-*kernelshap-*.json"))
    if not hits:
        raise FileNotFoundError(f"nenhum resultado '{scenario}' em {directory}")
    return hits[0]


def load_context(path: Path, idx: int) -> dict:
    row = json.load(open(path)).get(str(idx)) or {}
    ctx = (row.get("xai_result") or {}).get("context")
    if not ctx:
        raise ValueError(f"amostra {idx} sem xai_result.context em {path.name}")
    return ctx


def windows(ctx: dict, size: int, show_negative: bool):
    """Por janela: (soma φ+, soma φ-, nº de palavras injetadas, 1ª palavra)."""
    values = np.asarray(ctx["values"], dtype=float)
    tokens = ctx["tokens"]
    span = ctx.get("injected_span")
    inj = np.zeros(len(values), dtype=bool)
    if span:
        inj[span["start_token"]:span["end_token"]] = True
    out = []
    for k in range(0, len(values), size):
        v = values[k:k + size]
        out.append({
            "pos": float(np.clip(v, 0, None).sum()),
            "neg": float(np.clip(v, None, 0).sum()) if show_negative else 0.0,
            "n_inj": int(inj[k:k + size].sum()),
            "n": len(v),
            "word": tokens[k].strip()[:10],
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--none-json", type=Path, help="resultado do cenário none (default: pasta dolly none)")
    ap.add_argument("--attack-dir", type=Path, default=DEFAULT_ATTACK, help="pasta com direct/combined")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--show-negative", action="store_true", help="Σmin(φ,0) em cinza abaixo do zero")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = {
        "none": args.none_json or find_result(DEFAULT_NONE, "none"),
        "direct": find_result(args.attack_dir, "direct"),
        "combined": find_result(args.attack_dir, "combined"),
    }
    ctxs = {s: load_context(paths[s], args.idx) for s in SCENARIOS}
    wins = {s: windows(ctxs[s], args.window, args.show_negative) for s in SCENARIOS}

    ymax = max(w["pos"] for s in SCENARIOS for w in wins[s]) * 1.25
    ymin = min((w["neg"] for s in SCENARIOS for w in wins[s]), default=0) * 1.25
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, s in zip(axes, SCENARIOS):
        ctx, ws = ctxs[s], wins[s]
        x = np.arange(len(ws))
        colors = [RED if w["n_inj"] else BLUE for w in ws]
        bars = ax.bar(x, [w["pos"] for w in ws], 0.6, color=colors, edgecolor="k", linewidth=0.6)
        for b, w in zip(bars, ws):
            if 0 < w["n_inj"] < w["n"]:
                b.set_hatch("//")
        if args.show_negative:
            ax.bar(x, [w["neg"] for w in ws], 0.6, color=GRAY, alpha=0.6)
        for i, w in enumerate(ws):
            lab = f"{w['pos']:.3f}"
            if w["n_inj"]:
                lab += f"\n[INJ {w['n_inj']}/{w['n']}]"
            ax.text(i, w["pos"] + ymax * 0.01, lab, ha="center", va="bottom", fontsize=8,
                    color=RED if w["n_inj"] else "k")
        basal = [w["pos"] for w in ws if not w["n_inj"] and w["n"] == args.window]
        title = f"{s}: {ctx.get('predicted_label')} (p_malign={ctx['p_malign']:.4f})"
        if basal:
            m = float(np.mean(basal))
            ax.axhline(m, color="green", ls="-.", lw=1.5, label=f"Média basal ({m:.3f})")
            peak = max(w["pos"] for w in ws if w["n_inj"]) if any(w["n_inj"] for w in ws) else None
            if peak is not None and m > 0:
                title += f"\npico/basal = {peak / m:.1f}x"
            ax.legend(loc="upper left", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"J{i}\n({w['word']}…)" for i, w in enumerate(ws)], fontsize=7, rotation=30)
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Σ φ⁺ por janela (evidência → malign)", fontweight="bold")
    fig.suptitle(f"Perfil de evidência malign por janelas de {args.window} palavras — amostra {args.idx}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = args.out or (Path(__file__).resolve().parent / f"concentration_profile_idx{args.idx}.png")
    fig.savefig(out, dpi=150)
    print(f"salvo em {out}")
    for s in SCENARIOS:
        print(s, "n_janelas=%d" % len(wins[s]), "Σφ⁺=%.4f" % sum(w["pos"] for w in wins[s]))


if __name__ == "__main__":
    main()
