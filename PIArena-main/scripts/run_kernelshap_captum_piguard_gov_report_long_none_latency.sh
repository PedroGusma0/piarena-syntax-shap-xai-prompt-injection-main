#!/bin/bash
# gov_report_long + none + piguard + kernelshap (backend captum, scope all), registrando latência/VRAM
# por fase (attack, piguard, xai) e o perturbations_per_eval usado. Depois roda xai_metrics.py.
# Saídas: <out>/xai_metrics/<stem>_latency.json (com run.perturbations_per_eval) + <out>/latency.{json,txt}.
# Uso: bash scripts/run_kernelshap_captum_piguard_gov_report_long_none_latency.sh
#      PPE=16 LIMIT=3 bash scripts/<este>.sh   (PPE = perturbations_per_eval, default 2; LIMIT = smoke test)
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=gov_report_long
ATTACK=none
DEFENSE=piguard
XAI=kernelshap
BACKEND_LLM=${BACKEND_LLM:-Qwen/Qwen3-4B-Instruct-2507}
SEED=${SEED:-42}
PPE=${PPE:-2}
NAME=kernelshap_captum_piguard_latency_${DATASET}_${ATTACK}_ppe${PPE}
LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")

OUT_ROOT="results/evaluation_results/${NAME}"
RESULT="${OUT_ROOT}/${DATASET}-${BACKEND_LLM//\//-}-${ATTACK}-${DEFENSE}-${XAI}-${SEED}.json"
STEM="$(basename "${RESULT%.json}")"
mkdir -p logs
exec > >(tee -a "logs/${NAME}.log") 2>&1

# Run limpo: o resume do main.py pularia amostras e invalidaria os tempos.
rm -f "$RESULT" "${OUT_ROOT}/tmp_attack_results/${DATASET}-${BACKEND_LLM//\//-}-${ATTACK}-${DEFENSE}-${SEED}.json"

python main.py --dataset "$DATASET" --backend_llm "$BACKEND_LLM" --attack "$ATTACK" \
  --defense "$DEFENSE" --xai "$XAI" --xai_backend captum --xai_scope all \
  --xai_perturbations_per_eval "$PPE" \
  --name "$NAME" --seed "$SEED" --no-wandb "${LIMIT_ARGS[@]}"

python - "$RESULT" "$OUT_ROOT" "$PPE" <<'PYEOF'
import json, os, sys
import numpy as np

path, out_root, ppe = sys.argv[1], sys.argv[2], int(sys.argv[3])
all_samples = list(json.load(open(path, encoding="utf-8")).values())
samples = [s for s in all_samples if not (s.get("timing") or {}).get("cold_start")]
KEYS = ("attack_s", "promptguard_s", "xai_s", "defense_total_s")

def stats(key):
    v = np.array([s["timing"][key] for s in samples
                  if (s.get("timing") or {}).get(key) is not None], dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "total_s": float(v.sum()), "mean_s": float(v.mean()),
            "median_s": float(np.median(v)), "p90_s": float(np.percentile(v, 90)),
            "p99_s": float(np.percentile(v, 99)), "min_s": float(v.min()), "max_s": float(v.max())}

summary = {"result": path, "xai_backend": "captum", "perturbations_per_eval": ppe,
           **{k: stats(k) for k in KEYS}}
json.dump(summary, open(os.path.join(out_root, "latency.json"), "w"), indent=2)
lines = [f"perturbations_per_eval={ppe} (backend captum)",
         f"{'metrica':<17}{'n':>6}{'media':>10}{'mediana':>10}{'p90':>10}{'p99':>10}{'max':>10}{'total':>10}"]
for k in KEYS:
    s = summary[k]
    if s:
        lines.append(f"{k:<17}{s['n']:>6}{s['mean_s']:>10.4f}{s['median_s']:>10.4f}"
                     f"{s['p90_s']:>10.4f}{s['p99_s']:>10.4f}{s['max_s']:>10.4f}{s['total_s']:>10.1f}")
text = "\n".join(lines)
open(os.path.join(out_root, "latency.txt"), "w").write(text + "\n")
print(text)
PYEOF

python scripts/xai_metrics.py --result "$RESULT" --no-wandb

# Grava o perturbations_per_eval também no JSON de latência do xai_metrics.
python3 - "${OUT_ROOT}/xai_metrics/${STEM}_latency.json" "$PPE" <<'PY'
import json, sys
p, ppe = sys.argv[1], int(sys.argv[2])
d = json.load(open(p))
d["run"]["perturbations_per_eval"] = ppe
d["run"]["xai_backend"] = "captum"
json.dump(d, open(p, "w"), indent=2)
PY
echo "Latência: ${OUT_ROOT}/latency.json e ${OUT_ROOT}/xai_metrics/${STEM}_latency.json"
