"""Gera um relatório markdown por ataque a partir dos resultados --xai kernelshap.

Para cada amostra detectada pela defesa, forma uma "sentença" com os tokens de
maior valor SHAP (classe malign) e a reporta como origem da prompt injection.

Uso:
    python scripts/xai_relatorio.py --dir ../kernelshap_squad_v2_shap_none
    python scripts/xai_relatorio.py --result <um_resultado.json>

Saída: <dir>/xai_metrics/relatorio/<ataque>.md
"""
import argparse
import json
import math
import re
from pathlib import Path

ATTACKS = ["combined", "completion", "direct", "ignore", "none"]


def attack_name(path: Path) -> str:
    for a in ATTACKS:
        if f"-{a}-" in path.name:
            return a
    return path.stem


def top_sentence(tokens, values, top_frac, max_tokens):
    """Top-K tokens com SHAP>0, na ordem original; lacunas viram ' … '."""
    k = min(max_tokens, max(3, math.ceil(top_frac * len(tokens))))
    ranked = sorted((i for i, v in enumerate(values) if v > 0), key=lambda i: -values[i])[:k]
    idxs = sorted(ranked)
    parts, prev = [], None
    for i in idxs:
        tok = tokens[i]
        if prev is not None and i != prev + 1:
            parts.append(" … ")
            tok = tok.lstrip()
        parts.append(tok)
        prev = i
    return re.sub(r"\s+", " ", "".join(parts)).strip(), idxs


def build_report(result_path: Path, top_frac: float, max_tokens: int) -> str:
    data = json.loads(result_path.read_text())
    attack = attack_name(result_path)
    detected = [(k, s) for k, s in data.items()
                if s.get("defense_result", {}).get("detect_flag") and s.get("xai_result", {}).get("context")]
    lines = [
        f"# Relatório XAI — ataque `{attack}`",
        "",
        f"- Resultado: `{result_path.name}`",
        f"- Amostras detectadas e explicadas: {len(detected)} / {len(data)}",
        f"- Sentença = top-{max_tokens} tokens (≥3, {top_frac:.0%} do contexto) com maior SHAP > 0 para a classe malign, na ordem original; `…` marca lacunas.",
    ]
    if attack == "none":
        lines.append("- **Ataque `none`: não há injeção real; as detecções abaixo são falsos positivos do detector.**")
    lines.append("")
    hits_total = n_total = 0
    for k, s in detected:
        x = s["xai_result"]["context"]
        sentence, idxs = top_sentence(x["tokens"], x["values"], top_frac, max_tokens)
        span = x.get("injected_span")
        lines += [
            f"### Amostra {k} ({s.get('category', '-')}, p_malign={x['p_malign']:.3f})",
            "",
            f'Ação não autorizada: O Agente reconheceu uma tentativa de Prompt Injection originada pela sentença: "{sentence}"',
            "",
        ]
        if span and attack != "none":
            inside = sum(span["start_token"] <= i < span["end_token"] for i in idxs)
            hits_total += inside
            n_total += len(idxs)
            lines.append(f"- Tokens dentro do trecho injetado: {inside}/{len(idxs)}")
            lines.append(f'- Tarefa injetada real: "{s.get("injected_task", "")}"')
            lines.append("")
    if n_total:
        lines += ["---", "", f"Precisão agregada dos tokens no trecho injetado: {hits_total}/{n_total} = {hits_total / n_total:.1%}", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, help="diretório com os JSON de resultado")
    p.add_argument("--result", type=Path, help="um JSON de resultado")
    p.add_argument("--top-frac", type=float, default=0.10)
    p.add_argument("--max-tokens", type=int, default=12)
    a = p.parse_args()
    if a.result:
        files = [a.result]
    elif a.dir:
        files = sorted(a.dir.glob("*-kernelshap-*.json"))
    else:
        p.error("informe --dir ou --result")
    for f in files:
        out_dir = f.parent / "xai_metrics" / "relatorio"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{attack_name(f)}.md"
        out.write_text(build_report(f, a.top_frac, a.max_tokens))
        print(out)


if __name__ == "__main__":
    main()
