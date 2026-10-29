"""Mapa de saliencia por janela, com o grafico oficial do SHAP (`shap.plots.text`).

Gera UM arquivo HTML por prompt explicado (por padrao, os bloqueados), com TODAS
as janelas (padrao) uma apos a outra: cabecalho da janela + `shap.plots.text()`
das palavras da janela (vermelho = empurra para malign, azul = empurra para benigno). Mesmo
renderizador do `saliency_maps.html` do `xai_metrics.py`, e mesmo contorno
amarelo nas palavras do `injected_span`.

Cabecalho de cada janela: `Janela i/N · P(malign) · base · Σφ` e os rotulos
[INJ] (a janela contem a injecao) e [BLOQUEIO] (P(malign) >= limiar).

Saida: `<out-dir>/idx<i>.html` por prompt + `index.html` com os links.

Le `xai_result["windows"]` (runs com `--defense promptguard_windowed
--window_tokens 512 --xai kernelshap`). So amostras explicadas (bloqueadas, no
escopo padrao) tem janelas com valores SHAP.

Uso:
    python scripts/xai_window_saliency.py --result <json>
    python scripts/xai_window_saliency.py --result <json> --idx 0 3 --windows flagged+injected
    python scripts/xai_window_saliency.py --result <json> --max-samples 10
"""
import argparse
import html
import json
import os
from pathlib import Path

import numpy as np

_ATTACKS = ("direct", "ignore", "completion", "combined", "character", "none")


def _parts(path):
    return os.path.splitext(os.path.basename(path))[0].split("-")


def select(entries, thr, mode):
    keep = []
    for e in entries:
        flagged = e["p_malign"] >= thr
        injected = bool(e.get("contains_injection"))
        if mode == "all" or (mode == "flagged" and flagged) or (mode == "injected" and injected) \
                or (mode == "flagged+injected" and (flagged or injected)):
            keep.append(e)
    return keep


def _helpers():
    """The span-border / numpy-colour fixes `xai_metrics.py` already applies to shap.plots.text()."""
    try:
        from xai_metrics import _fix_numpy_scalar_colors, _mark_injected_span
        return _fix_numpy_scalar_colors, _mark_injected_span
    except Exception as e:  # piarena not importable: render without the span border
        print(f"[aviso] contorno da injecao desativado (nao foi possivel importar xai_metrics: {type(e).__name__}: {e}). "
              "Rode no ambiente `piarena` (pip install -e .) para te-lo.")
        return (lambda h: h), (lambda h, span: h)


def window_block(shap, win, n_windows, thr, fix_colors, mark_span):
    values = np.asarray(win["values"], dtype=float)
    tags = []
    if win.get("contains_injection"):
        tags.append("[INJ]")
    if win["p_malign"] >= thr:
        tags.append("[BLOQUEIO]")
    color = "#b30000" if tags else "#222"
    head = (
        f"<h4 style='margin:18px 0 4px;color:{color}'>Janela {win['window_idx'] + 1}/{n_windows} "
        f"&nbsp;·&nbsp; P(malign)={win['p_malign']:.3f} &nbsp;·&nbsp; base={win['base_value']:.3f} "
        f"&nbsp;·&nbsp; Σφ={values.sum():+.3f} &nbsp;·&nbsp; {len(values)} palavras "
        f"&nbsp; <b>{' '.join(tags)}</b></h4>"
    )
    if not len(values):
        return head + "<p><i>(janela sem palavras explicadas)</i></p>"
    explanation = shap.Explanation(values=values, base_values=win["base_value"], data=win["tokens"])
    try:
        body = shap.plots.text(explanation, display=False)
        body = mark_span(fix_colors(body), win.get("injected_span"))
    except Exception as e:  # one bad window must not kill the report
        body = f"<p>(falha ao renderizar: {html.escape(str(e))})</p>"
    return head + body


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--idx", type=int, nargs="*", default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="limite de amostras no HTML (padrao: todas as explicadas)")
    ap.add_argument("--windows", choices=["all", "flagged", "injected", "flagged+injected"], default="all",
                    help="quais janelas entram (padrao: todas)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <result dir>/xai_metrics/plots/<result stem>/window_saliency/")
    args = ap.parse_args()

    import shap  # official shap package, plotting only

    fix_colors, mark_span = _helpers()
    result = json.load(open(args.result, encoding="utf-8"))
    parts = _parts(args.result)
    dataset = parts[0]
    attack = next((p for p in parts if p in _ATTACKS), "attack")
    out_dir = args.out_dir or (args.result.parent / "xai_metrics" / "plots" / args.result.stem / "window_saliency")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.idx:
        idxs = [str(i) for i in args.idx]
    else:
        idxs = [i for i, s in result.items() if (s.get("xai_result") or {}).get("windows")]
    if args.max_samples:
        idxs = idxs[: args.max_samples]

    style = "<style>body{font-family:sans-serif;margin:20px}</style>"
    n_win, written = 0, []
    for idx in idxs:
        s = result.get(idx) or {}
        entries = (s.get("xai_result") or {}).get("windows")
        if not entries:
            print(f"[skip] amostra {idx}: sem xai_result.windows (amostra nao explicada ou run sem janelas)")
            continue
        dr = s.get("defense_result") or {}
        thr = dr.get("block_threshold", 0.5)
        chosen = select(entries, thr, args.windows)
        status = "BLOQUEADO" if dr.get("detect_flag") else "NÃO BLOQUEADO"
        cat = f" · {html.escape(str(s['category']))}" if s.get("category") else ""
        title = f"Prompt {idx} — {status} — {dataset} · ataque {attack}{cat}"
        doc = [
            f"<html><head><meta charset='utf-8'><title>{title}</title>{style}</head><body>",
            f"<h2>{title}</h2>",
            f"<p>{len(chosen)}/{len(entries)} janelas · P(malign) máx = {max(e['p_malign'] for e in entries):.3f} · "
            "vermelho: empurra a janela para <b>malign</b>; azul: para <b>benigno</b> · "
            "<b>[INJ]</b> = a janela contém a injeção (palavras com contorno amarelo) · "
            "<b>[BLOQUEIO]</b> = P(malign) ≥ limiar.</p>",
        ]
        for w in chosen:
            doc.append(window_block(shap, w, len(entries), thr, fix_colors, mark_span))
            n_win += 1
        doc.append("</body></html>")
        path = out_dir / f"idx{idx}.html"
        path.write_text("\n".join(doc), encoding="utf-8")
        written.append((idx, status, len(chosen), len(entries), max(e["p_malign"] for e in entries), path.name))

    if written:
        rows = "\n".join(
            f"<li><a href='{name}'>Prompt {i}</a> — {st} — {c}/{n} janelas — P(malign) máx {p:.3f}</li>"
            for i, st, c, n, p, name in written
        )
        (out_dir / "index.html").write_text(
            f"<html><head><meta charset='utf-8'><title>Window saliency — {dataset} ({attack})</title>{style}</head>"
            f"<body><h2>Mapas de saliência por janela — {dataset} · ataque {attack}</h2><ul>{rows}</ul></body></html>",
            encoding="utf-8",
        )
    print(f"{len(written)} HTML(s) ({n_win} janela(s)) em {out_dir}")

if __name__ == "__main__":
    main()
