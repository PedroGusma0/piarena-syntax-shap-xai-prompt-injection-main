"""Relatório XAI v2 (Markdown + LaTeX) das amostras bloqueadas de um resultado --xai kernelshap.

Para cada amostra bloqueada pela defesa:
  1. Acesso bloqueado (V, O, M) — lido do JSON gerado por scripts/xai_vom_classify.py
  2. Tokens que mais contribuíram para a classe malign (top-k por φ > 0)
  3. Ponto de injeção — o prompt é dividido em sentenças; a sentença com maior soma de
     φ > 0 é reportada com suas posições de caractere, um gráfico de barras por sentença
     (estilo scripts/xai_window_anomaly.py) e o prompt com a sentença destacada.

Uso:
    python scripts/xai_relatorio_v2.py --result <direct>.json --vom scripts/xai_vom_squad_v2_direct.json

Saída: <dir do resultado>/xai_metrics/relatorio_v2/{<ataque>.md, <ataque>.tex, figs/<ataque>-idxN.png}
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xai_relatorio import attack_name
from xai_window_anomaly import BLUE, GREEN, RED

_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")


def align_tokens(tokens, context):
    """(start, end) de caractere de cada token no contexto, por busca sequencial."""
    spans, cur = [], 0
    for tok in tokens:
        t = tok.strip()
        a = context.find(t, cur) if t else -1
        if a == -1:
            spans.append((cur, cur))
            continue
        spans.append((a, a + len(t)))
        cur = a + len(t)
    return spans


def split_sentences(context):
    """Lista de (start, end) de sentenças, sem os espaços das bordas."""
    out, a = [], 0
    for m in _SENT_END.finditer(context):
        out.append((a, m.start()))
        a = m.end()
    out.append((a, len(context)))
    res = []
    for a, b in out:
        seg = context[a:b]
        a2 = a + len(seg) - len(seg.lstrip())
        b2 = b - (len(seg) - len(seg.rstrip()))
        if b2 > a2:
            res.append((a2, b2))
    return res


def analyze(sample, top_k):
    x = sample["xai_result"]["context"]
    ctx = sample["injected_context"]
    tokens, vals = x["tokens"], np.asarray(x["values"], dtype=float)
    tspans = align_tokens(tokens, ctx)
    sents = split_sentences(ctx)
    task = sample.get("injected_task") or ""
    i0 = ctx.find(task) if task else -1
    inj = (i0, i0 + len(task)) if i0 != -1 else None

    pos = np.clip(vals, 0, None)
    score = np.zeros(len(sents))
    tok_sent = []
    for (a, b), v in zip(tspans, pos):
        j = next((j for j, (sa, sb) in enumerate(sents) if sa <= a < sb), None)
        tok_sent.append(j)
        if j is not None:
            score[j] += v
    best = int(np.argmax(score))
    total = float(pos.sum()) or 1.0
    in_inj = lambda a, b: bool(inj) and a < inj[1] and b > inj[0]
    order = [i for i in np.argsort(-vals) if vals[i] > 0][:top_k]
    return {
        "x": x, "ctx": ctx, "inj": inj, "sents": sents, "score": score, "best": best,
        "frac": score[best] / total,
        "best_tokens": [i for i, j in enumerate(tok_sent) if j == best],
        "sent_inj": [in_inj(a, b) for a, b in sents],
        "top": [(tokens[i].strip(), float(vals[i]), in_inj(*tspans[i])) for i in order],
    }


def plot(r, title, path):
    sents, score, best = r["sents"], r["score"], r["best"]
    n = len(sents)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * n + 2), 5.2))
    ax.bar(x, score, 0.6, color=[RED if f else BLUE for f in r["sent_inj"]], edgecolor="k", linewidth=0.6)
    ymax = max(score.max(), 1e-9)
    for i, v in enumerate(score):
        ax.text(i, v + 0.012 * ymax, f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                color=RED if r["sent_inj"][i] else "#333333")
    basal = [v for v, f in zip(score, r["sent_inj"]) if not f]
    if basal:
        ax.axhline(np.mean(basal), color=GREEN, ls="-.", lw=1.5, label=f"Média basal ({np.mean(basal):.3f})")
    ax.annotate(f"[PONTO DE INJEÇÃO ★]\nΣφ⁺: {score[best]:.3f} ({r['frac']:.0%} do total)", (best, score[best]),
                xytext=(0, 26), textcoords="offset points", ha="center", fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED), bbox=dict(boxstyle="round,pad=0.3", fc="#fde8e8", ec=RED))
    labels = []
    for i, (a, b) in enumerate(sents):
        txt = " ".join(r["ctx"][a:b].split())[:16]
        labels.append(f"Sentença {i}\n({txt}…)" + ("\n[INJ]" if r["sent_inj"][i] else ""))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35 if n > 3 else 0, ha="right" if n > 3 else "center", fontsize=7)
    ax.set_ylim(0, ymax * 1.35)
    ax.set_ylabel("Σ SHAP malign (φ > 0) da sentença", fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    if basal:
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def prompt_blocks(r):
    """Prompt em 3 blocos (antes, tarefa injetada, depois), cada um como lista de (texto, é_sentença_máxima)."""
    ctx, (a, b) = r["ctx"], r["sents"][r["best"]]
    i0, i1 = r["inj"] or (len(ctx), len(ctx))
    cuts = sorted({0, len(ctx), a, b, i0, i1})
    blocks = ([], [], [])
    for x, y in zip(cuts, cuts[1:]):
        txt = " ".join(ctx[x:y].split())
        if txt:
            blocks[0 if y <= i0 else 1 if x < i1 else 2].append((txt, a <= x and y <= b))
    return blocks


def vom_sentence(v):
    return (f"Acesso bloqueado: detectada tentativa de {v['O']} (Objetivo: {v['O_detalhe']}) "
            f"executada via {' + '.join(v['M'])} (Mecânica), originada por {v['V']} (Vetor).")


# ---------------------------------------------------------------- Markdown

def md_escape(s):
    return s.replace("|", "\\|").replace("*", "\\*").replace("_", "\\_")


def md_section(k, s, v, r, fig_rel, sal_rel=None):
    a, b = r["sents"][r["best"]]
    blocks = prompt_blocks(r)
    hit = r["sent_inj"][r["best"]]
    lines = [
        f"## Amostra {k} ({s.get('category', '-')}, p_malign={r['x']['p_malign']:.3f})", "",
        "**1 — Acesso bloqueado (V, O, M)**", "",
        f"> {vom_sentence(v)}", "",
        f"- Evidência no texto injetado: {'; '.join(v['evidencia'])}", "",
        *([f"![Mapa de saliência — amostra {k}]({sal_rel})", "",
           "_Mapa de saliência (Kernel SHAP): vermelho empurra para malign, azul para benign; borda: tarefa injetada._", ""]
          if sal_rel else []),
        "**2 — Tokens que mais contribuíram para malign (Kernel SHAP)**", "",
        "| # | token | φ (malign) | no trecho injetado? |", "|---|---|---|---|",
    ]
    lines += [f"| {i} | `{t}` | {val:+.4f} | {'sim' if ins else 'não'} |" for i, (t, val, ins) in enumerate(r["top"], 1)]
    lines += [
        "", "**3 — Ponto de injeção**", "",
        f"- Sentença {r['best']} de {len(r['sents'])}, caracteres `[{a}, {b})` do prompt "
        f"(tokens {r['best_tokens'][0]}–{r['best_tokens'][-1]}), Σφ⁺ = {r['score'][r['best']]:.4f} "
        f"({r['frac']:.0%} do φ⁺ total)" if r["best_tokens"] else f"- Sentença {r['best']}, caracteres `[{a}, {b})`",
        f"- Coincide com o trecho realmente injetado: **{'sim' if hit else 'não'}**"
        + (f" (injeção em `[{r['inj'][0]}, {r['inj'][1]})`)" if r["inj"] else ""),
        "", f"![Σφ⁺ por sentença — amostra {k}]({fig_rel})", "",
        "Prompt — _〖 〗: tarefa injetada · **negrito**: sentença de maior concentração de SHAP malign._", "",
        "> " + " ".join(p for p in [
            " ".join(f"**{md_escape(t)}**" if red else md_escape(t) for t, red in blocks[0]),
            ("〖 " + " ".join(f"**{md_escape(t)}**" if red else md_escape(t) for t, red in blocks[1]) + " 〗")
            if blocks[1] else "",
            " ".join(f"**{md_escape(t)}**" if red else md_escape(t) for t, red in blocks[2]),
        ] if p),
        "",
        "---", "",
    ]
    return lines


# ---------------------------------------------------------------- LaTeX

_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''", "…": r"\ldots{}", "φ": r"$\varphi$",
        "Σ": r"$\Sigma$", "⁺": r"$^{+}$"}


def tex(s):
    out = []
    for ch in s:
        if ch in _TEX:
            out.append(_TEX[ch])
        elif ord(ch) < 256:
            out.append(ch)
        else:  # fora do Latin-1: pdflatex+inputenc não compila (Hangul, árabe, ...)
            out.append("?")
    return "".join(out)


def tex_run(segs, boxed=False):
    """Texto do prompt; sentença máxima em vermelho; boxed=True põe borda amarela em cada palavra (tarefa injetada)."""
    out = []
    for t, red in segs:
        for w in t.split():
            w = "\\textcolor{red!70!black}{\\bfseries " + tex(w) + "}" if red else tex(w)
            out.append("\\injbox{" + w + "}" if boxed else w)
    return " ".join(out)


def tex_section(k, s, v, r, fig_rel, sal_rel=None):
    a, b = r["sents"][r["best"]]
    blocks = prompt_blocks(r)
    hit = r["sent_inj"][r["best"]]
    L = [
        f"\\subsection[Amostra {k} --- {tex(s.get('category', '-'))}]"
        f"{{Amostra {k} ({tex(s.get('category', '-'))}, $p_{{malign}}={r['x']['p_malign']:.3f}$)}}", "",
        "\\subsubsection*{1 --- Acesso bloqueado (V, O, M)}", "",
        f"\\begin{{quote}}\\itshape {tex(vom_sentence(v))}\\end{{quote}}", "",
        f"Evidência no texto injetado: {tex('; '.join(v['evidencia']))}.", "",
        *(["\\noindent\\begin{minipage}{\\linewidth}",  # legenda + imagem juntas
           "\\textbf{Mapa de saliência} {\\footnotesize(Kernel SHAP): \\textcolor{red!70!black}{\\textbf{vermelho}} "
            "empurra para malign, \\textcolor{blue!70!black}{\\textbf{azul}} para benign; "
            "\\textcolor[HTML]{C9860F}{\\textbf{borda}}: tarefa injetada.}",
            "\\par\\smallskip\\centering",
            f"\\includegraphics[width=\\linewidth,height=0.5\\textheight,keepaspectratio]{{{sal_rel}}}",
            "\\end{minipage}", "\\medskip", ""] if sal_rel else []),
        "\\subsubsection*{2 --- Tokens que mais contribuíram para malign (Kernel SHAP)}", "",
        "\\begin{table}[H]\\centering\\small",
        "\\begin{tabular}{rlrc}\\toprule",
        "\\# & Token & $\\varphi$ (malign) & No trecho injetado? \\\\\\midrule",
    ]
    L += [f"{i} & \\texttt{{{tex(t)}}} & ${val:+.4f}$ & {'sim' if ins else 'não'} \\\\" for i, (t, val, ins) in enumerate(r["top"], 1)]
    L += ["\\bottomrule\\end{tabular}\\end{table}", "",
          "\\noindent\\begin{minipage}{\\linewidth}",  # item 3 inteiro num bloco: o gráfico não se separa do texto
          "\\subsubsection*{3 --- Ponto de injeção}", "",
          f"Sentença {r['best']} de {len(r['sents'])}, caracteres $[{a}, {b})$ do prompt"
          + (f" (tokens {r['best_tokens'][0]}--{r['best_tokens'][-1]})" if r["best_tokens"] else "")
          + f", $\\Sigma\\varphi^{{+}} = {r['score'][r['best']]:.4f}$ ({r['frac'] * 100:.0f}\\% do $\\varphi^{{+}}$ total)."
          + f" Coincide com o trecho realmente injetado: \\textbf{{{'sim' if hit else 'não'}}}"
          + (f" (injeção em $[{r['inj'][0]}, {r['inj'][1]})$)." if r["inj"] else "."), "",
          "\\smallskip\\centering",
          f"\\includegraphics[width=0.85\\linewidth]{{{fig_rel}}}",
          f"\\captionof{{figure}}{{Soma do SHAP malign ($\\varphi>0$) por sentença --- amostra {k}. Vermelho: sentença com o trecho injetado; $\\bigstar$: ponto de injeção.}}",
          "\\end{minipage}", "\\bigskip", "",
          "\\noindent\\begin{minipage}{\\linewidth}",  # legenda + prompt juntos
          "\\textbf{Prompt:} {\\footnotesize\\textcolor{injborder}{\\textbf{Borda:}} tarefa injetada. "
          "\\textcolor{red!70!black}{\\textbf{Vermelho:}} sentença de maior concentração de SHAP malign.}",
          "\\begin{quote}\\small\\ttfamily",
          " ".join(x for x in (tex_run(blocks[0]), tex_run(blocks[1], boxed=True), tex_run(blocks[2])) if x),
          "\\end{quote}",
          "\\end{minipage}", "",
          "\\FloatBarrier", ""]
    return L


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--vom", type=Path, required=True, help="JSON de scripts/xai_vom_classify.py")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--tex-fig-prefix", default="figs/", help="prefixo dos \\includegraphics no .tex")
    p.add_argument("--saliency-dir", default="saliency",
                   help="pasta (relativa à saída) com os PNGs de scripts/xai_saliency_snapshots.py (idxN.png)")
    a = p.parse_args()

    data = json.loads(a.result.read_text(encoding="utf-8"))
    vom = json.loads(a.vom.read_text(encoding="utf-8"))
    attack = attack_name(a.result)
    out = a.out_dir or a.result.parent / "xai_metrics" / "relatorio_v2"
    (out / "figs").mkdir(parents=True, exist_ok=True)
    blocked = [(k, s) for k, s in data.items()
               if s.get("defense_result", {}).get("detect_flag") and s.get("xai_result", {}).get("context")]
    missing = [k for k, _ in blocked if k not in vom]
    if missing:
        raise SystemExit(f"sem V/O/M para as amostras {missing}")

    md, tx, hits, n_sal = [], [], 0, 0
    for k, s in blocked:
        r = analyze(s, a.top_k)
        hits += r["sent_inj"][r["best"]]
        fig = f"{attack}-idx{k}.png"
        plot(r, f"{attack} · amostra {k} · {s.get('category', '')}", out / "figs" / fig)
        sal = f"{a.saliency_dir}/idx{k}.png" if (out / a.saliency_dir / f"idx{k}.png").exists() else None
        n_sal += sal is not None
        md += md_section(k, s, vom[k], r, f"figs/{fig}", sal)
        tx += tex_section(k, s, vom[k], r, a.tex_fig_prefix + fig, sal)

    n = len(blocked)
    m_count = Counter(m for k, _ in blocked for m in vom[k]["M"])
    p_min = min(s["xai_result"]["context"]["p_malign"] for _, s in blocked)
    criterio = None
    if "-promptguard-" in a.result.name:  # promptguard_windowed usa outra regra (P(malign) >= block_threshold por janela)
        criterio = ("Critério de bloqueio: o prompt é bloqueado quando a classe mais provável do Prompt-Guard é INJECTION "
                    "ou JAILBREAK, ou seja, não é BENIGN"
                    + (f". Todas as amostras bloqueadas têm probabilidade maligna acima de 50% (mínimo: {p_min:.1%})."
                       if p_min > 0.5 else "."))
    summary_md = [
        f"# Relatório XAI v2 — ataque `{attack}`", "",
        f"- Resultado: `{a.result.name}`",
        f"- Amostras bloqueadas e explicadas: {n} / {len(data)}",
        *([f"- {criterio}"] if criterio else []),
        f"- Ponto de injeção = sentença com maior Σφ⁺ (Kernel SHAP, classe malign); coincide com a sentença injetada em "
        f"**{hits}/{n} ({hits / n:.1%})** das amostras.",
        "- V/O/M: taxonomia de `referencia_relatorio.tex` + pôster CrowdStrike + OWASP LLM01 "
        f"(`{a.vom.name}`).", "",
        "| Mecânica | amostras |", "|---|---|", *[f"| {m} | {c} |" for m, c in m_count.most_common()], "", "---", "",
    ]
    summary_tex = [
        f"% Gerado por PIArena-main/scripts/xai_relatorio_v2.py a partir de {tex(a.result.name)}",
        "\\providecolor{injborder}{RGB}{230,180,0}",
        "\\providecommand{\\injbox}[1]{{\\setlength{\\fboxsep}{1pt}\\setlength{\\fboxrule}{0.8pt}"
        "\\fcolorbox{injborder}{white}{\\strut #1}}}",
        f"\\section[Ataque {tex(attack)}]{{Relatório XAI --- squad\\_v2, ataque {tex(attack)}}}", "",
        f"\\noindent Amostras bloqueadas e explicadas: {n} / {len(data)}.", "",
        *([f"\\noindent {tex(criterio)}", ""] if criterio else []),
        f"\\noindent Ponto de injeção = sentença com maior "
        f"$\\Sigma\\varphi^{{+}}$; coincide com a sentença injetada em {hits}/{n} ({hits / n * 100:.1f}\\%) das amostras.", "",
        "\\begin{table}[H]\\centering\\small\\begin{tabular}{lr}\\toprule Mecânica & Amostras \\\\\\midrule",
        *[f"{tex(m)} & {c} \\\\" for m, c in m_count.most_common()],
        "\\bottomrule\\end{tabular}\\caption{Distribuição das mecânicas (uma amostra pode ter várias).}\\end{table}", "",
    ]
    (out / f"{attack}.md").write_text("\n".join(summary_md + md), encoding="utf-8")
    (out / f"{attack}.tex").write_text("\n".join(summary_tex + tx), encoding="utf-8")
    print(out / f"{attack}.md")
    print(out / f"{attack}.tex")
    print(f"ponto de injeção == sentença injetada: {hits}/{n} | mapas de saliência: {n_sal}/{n}")


if __name__ == "__main__":
    main()
