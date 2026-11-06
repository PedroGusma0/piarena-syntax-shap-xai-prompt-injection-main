#!/bin/bash
# squad_v2 + direct + promptguard + kernelshap (backend shap, scope all), medindo latência e VRAM:
#   tempo total do ataque; tempo médio/total do PromptGuard; tempo médio/total do motor XAI;
#   tempo do xai_metrics. Tudo vai para <out>/xai_metrics/<stem>_latency.json (+ seção no _report.md).
# Gera também: saliency_maps.html, fidelity/alignment/confusion PNGs (xai_metrics), bar_global.png +
# saliency_bars.html (xai_shap_plots_test), xai_token_scatter_direct/, relatorio_v2/ (md/tex/figs).
# Uso: bash scripts/run_kernelshap_shap_promptguard_squad_v2_direct_latency.sh   (LIMIT=3 para smoke test)
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=squad_v2
DEFENSE=promptguard
XAI=kernelshap
ATTACK=direct
BACKEND_LLM=Qwen/Qwen3-4B-Instruct-2507
NAME=kernelshap_squad_v2_promptguard_direct_latency
SEED=42
LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")

RESULT="results/evaluation_results/${NAME}/${DATASET}-${BACKEND_LLM//\//-}-${ATTACK}-${DEFENSE}-${XAI}-${SEED}.json"
STEM="$(basename "${RESULT%.json}")"
OUT="$(dirname "$RESULT")/xai_metrics"
PLOTS="$OUT/plots/$STEM"
mkdir -p logs
LOG="logs/${NAME}.log"
exec > >(tee -a "$LOG") 2>&1

# Run limpo: o resume do main.py pularia amostras e invalidaria os tempos.
rm -f "$RESULT" "results/evaluation_results/${NAME}/tmp_attack_results/${DATASET}-${BACKEND_LLM//\//-}-${ATTACK}-${DEFENSE}-${SEED}.json"

declare -A STEP_S
timed() {  # timed <nome> <cmd...>  (falha de pós-processamento não derruba a run)
  local name="$1"; shift
  local t0 t1; t0=$(date +%s.%N)
  "$@" || echo "[WARN] passo '$name' falhou"
  t1=$(date +%s.%N)
  STEP_S[$name]=$(echo "$t1 - $t0" | bc -l)
}

python main.py --dataset "$DATASET" --backend_llm "$BACKEND_LLM" --attack "$ATTACK" \
  --defense "$DEFENSE" --xai "$XAI" --xai_backend shap --xai_scope all \
  --name "$NAME" --seed "$SEED" --no-wandb "${LIMIT_ARGS[@]}"

python scripts/xai_metrics.py --result "$RESULT" --no-wandb

timed xai_shap_plots_test python scripts/xai_shap_plots_test.py --result "$RESULT" --html
timed xai_token_scatter   python scripts/xai_token_scatter.py "$RESULT" "$OUT/xai_token_scatter_direct"

VOM="$OUT/vom.json"
timed xai_vom_classify python scripts/xai_vom_classify.py "$RESULT" "$VOM"
[ -s "$VOM" ] || cp scripts/xai_vom_squad_v2_direct.json "$VOM"

REL="$OUT/relatorio_v2"
if command -v google-chrome >/dev/null || command -v chromium >/dev/null || command -v chromium-browser >/dev/null; then
  timed saliency_snapshots python scripts/xai_saliency_snapshots.py \
    --html "$PLOTS/saliency_maps.html" --out-dir "$REL/saliency"
else
  echo "[WARN] chrome/chromium ausente: pulando saliency_snapshots"
fi
timed relatorio_v2 python scripts/xai_relatorio_v2.py --result "$RESULT" --vom "$VOM" --out-dir "$REL"
timed xai_bar_global_benign python scripts/xai_bar_global_benign.py

# Anexa os tempos dos passos de shell ao relatório de latência.
STEPS_JSON="{"
for k in "${!STEP_S[@]}"; do STEPS_JSON+="\"$k\": ${STEP_S[$k]},"; done
STEPS_JSON="${STEPS_JSON%,}}"
python3 - "$OUT/${STEM}_latency.json" "$STEPS_JSON" <<'PY'
import json, sys
p, steps = sys.argv[1], json.loads(sys.argv[2])
d = json.load(open(p))
d["latency_s"]["postprocess_shell"] = steps
json.dump(d, open(p, "w"), indent=2)
print("postprocess_shell:", steps)
PY
echo "Latência: $OUT/${STEM}_latency.json"
