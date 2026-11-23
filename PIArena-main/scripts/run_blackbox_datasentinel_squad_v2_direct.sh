#!/bin/bash
# squad_v2 + direct + datasentinel + blackbox: Kernel SHAP (captum) sobre o rótulo binário
# (detect_flag) do DataSentinel, todas as palavras como jogadores. Roda main.py e depois
# xai_metrics.py (mapas de saliência com a saída real do detector, deleção, alinhamento,
# latência). Nada disto foi rodado ainda com o modelo real (precisa de GPU).
# Uso: bash scripts/run_blackbox_datasentinel_squad_v2_direct.sh
#      LIMIT=3 bash scripts/<este>.sh               (smoke test: 3 amostras)
#      BASELINE=neutral bash scripts/<este>.sh      (empty | neutral | clean_context)
#      N_SAMPLES=800 bash scripts/<este>.sh         (coalizões por amostra; default auto = clip(16*palavras, 400, 3500))
# Cada coalizão é uma geração de um Mistral-7B: comece com LIMIT pequeno e meça o tempo por amostra
# (n_detector_calls e timing.xai_s no JSON de resultado).
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-squad_v2}
ATTACK=${ATTACK:-direct}
DEFENSE=datasentinel
XAI=blackbox
BACKEND_LLM=${BACKEND_LLM:-Qwen/Qwen3-4B-Instruct-2507}
SEED=${SEED:-42}
BASELINE=${BASELINE:-empty}
N_SAMPLES=${N_SAMPLES:-auto}
NAME=blackbox_datasentinel_${DATASET}_${ATTACK}_${BASELINE}
LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")

OUT_ROOT="results/evaluation_results/${NAME}"
RESULT="${OUT_ROOT}/${DATASET}-${BACKEND_LLM//\//-}-${ATTACK}-${DEFENSE}-${XAI}-${SEED}.json"
CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$CONFIG"' EXIT
mkdir -p logs
exec > >(tee -a "logs/${NAME}.log") 2>&1

# `blackbox` não tem flags próprias além de --xai_backend/--xai_scope: o resto entra pelo xai_config do YAML.
cat > "$CONFIG" <<YAML
xai_config:
  baseline: ${BASELINE}
  n_samples: ${N_SAMPLES}
YAML

python main.py --config "$CONFIG" --dataset "$DATASET" --backend_llm "$BACKEND_LLM" --attack "$ATTACK" \
  --defense "$DEFENSE" --xai "$XAI" --xai_backend captum \
  --name "$NAME" --seed "$SEED" --no-wandb "${LIMIT_ARGS[@]}"

python scripts/xai_metrics.py --result "$RESULT" --no-wandb
