"""Scatter por amostra: cada bolinha = token do prompt; x = valor SHAP (P(malign)).

Adaptado do estilo "Captured / Uncaptured" (ECMLF): lado direito (x>0) = empurra para
MALIGN, lado esquerdo (x<0) = empurra para BENIGN (SHAP de benign = -SHAP de malign).
Vermelho = malign; azul = benign; vermelho com borda preta = trecho injetado.
Uso: python xai_token_scatter.py <result.json> <out_dir>
"""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res, out = sys.argv[1], pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
data = json.load(open(res))
RED, BLUE, SHADE = "#e03c3c", "#3f8fd9", "#e9eff5"

n_ok = 0
for idx, row in sorted(data.items(), key=lambda kv: int(kv[0])):
    xr = (row.get("xai_result") or {}).get("context")
    if not xr:
        continue
    v = np.asarray(xr["values"], float)
    n = len(v)
    inj = np.zeros(n, bool)
    sp = xr.get("injected_span")
    if sp:
        inj[sp["start_token"]:sp["end_token"]] = True
    rng = np.random.default_rng(int(idx))
    y = rng.uniform(0, 1, n)
    mal, ben = v > 0, v <= 0

    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
    ax.axvspan(0, 0.5, color=SHADE, zorder=0)
    ax.axvline(0, color="#999", lw=0.8, ls="--", zorder=1)
    # contexto benigno-lado (vazado) e malign-lado (cheio), injeção por cima
    ax.scatter(v[~inj & ben], y[~inj & ben], s=24, c=BLUE, edgecolors="white", lw=0.3, alpha=.9, zorder=2, label="Benign (context)")
    ax.scatter(v[~inj & mal], y[~inj & mal], s=24, c=RED, edgecolors="white", lw=0.3, alpha=.9, zorder=2, label="Malign (context)")
    ax.scatter(v[inj], y[inj], s=34, c=RED, edgecolors="black", lw=1.2, zorder=3, label="Injected span")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([-.5, -.25, 0, .25, .5]); ax.set_yticks([])
    ax.set_xlabel("SHAP value (← benign | malign →)", fontsize=9)
    ax.set_title(f"KernelSHAP · direct · sample {idx}", fontsize=10, fontweight="bold")
    ax.text(0.02, 0.98, f"Benign\n{int(ben.sum()):,}", transform=ax.transAxes, va="top", fontsize=8)
    ax.text(0.98, 0.98, f"Malign\n{int(mal.sum()):,}", transform=ax.transAxes, va="top", ha="right", fontsize=8)
    fig.legend(loc="lower center", ncol=3, fontsize=7, frameon=False, handletextpad=0.2, columnspacing=1.0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / f"direct_{int(idx):03d}.png")
    plt.close(fig)
    n_ok += 1
print(n_ok, "plots ->", out)
