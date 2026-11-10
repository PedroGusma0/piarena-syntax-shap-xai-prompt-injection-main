#!/bin/bash
# PIGuard (--defense piguard), squad_v2, ataque direct, SEM XAI,
# registrando o tempo de inferencia do classificador.
# O main.py ja grava por amostra result["timing"]["promptguard_s"] (so o forward do
# classificador) e "defense_total_s" (defesa inteira). Ao final, este script resume
# (media/mediana/p90/p99/max) em ${OUT_ROOT}/latency.json e latency.txt.
#
# Variaveis opcionais: BACKEND_LLM=... SEED=42 bash scripts/<este>.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=squad_v2
ATTACK=direct
DEFENSE=piguard
BACKEND_LLM=${BACKEND_LLM:-Qwen/Qwen3-4B-Instruct-2507}
SEED=${SEED:-42}
NAME=piguard_latency_${DATASET}_${ATTACK}

OUT_ROOT="results/evaluation_results/${NAME}"
RESULT="${OUT_ROOT}/${DATASET}-${BACKEND_LLM//\//-}-${ATTACK}-${DEFENSE}-${SEED}.json"

python main.py \
  --dataset "$DATASET" \
  --backend_llm "$BACKEND_LLM" \
  --attack "$ATTACK" \
  --defense "$DEFENSE" \
  --name "$NAME" \
  --seed "$SEED"

python - "$RESULT" "$OUT_ROOT" <<'PYEOF'
import json, os, sys
import numpy as np

path, out_root = sys.argv[1:3]
samples = list(json.load(open(path, encoding="utf-8")).values())

def stats(key):
    v = np.array([s["timing"][key] for s in samples
                  if (s.get("timing") or {}).get(key) is not None], dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "total_s": float(v.sum()), "mean_s": float(v.mean()),
            "median_s": float(np.median(v)), "p90_s": float(np.percentile(v, 90)),
            "p99_s": float(np.percentile(v, 99)), "min_s": float(v.min()),
            "max_s": float(v.max())}

summary = {"result": path, "promptguard_s": stats("promptguard_s"),
           "defense_total_s": stats("defense_total_s")}
json.dump(summary, open(os.path.join(out_root, "latency.json"), "w"), indent=2)

lines = [f"{'metrica':<17}{'n':>6}{'media':>10}{'mediana':>10}{'p90':>10}{'p99':>10}{'max':>10}{'total':>10}"]
for k in ("promptguard_s", "defense_total_s"):
    s = summary[k]
    if s:
        lines.append(f"{k:<17}{s['n']:>6}{s['mean_s']:>10.4f}{s['median_s']:>10.4f}"
                     f"{s['p90_s']:>10.4f}{s['p99_s']:>10.4f}{s['max_s']:>10.4f}{s['total_s']:>10.1f}")
text = "\n".join(lines)
open(os.path.join(out_root, "latency.txt"), "w").write(text + "\n")
print(text)
PYEOF
