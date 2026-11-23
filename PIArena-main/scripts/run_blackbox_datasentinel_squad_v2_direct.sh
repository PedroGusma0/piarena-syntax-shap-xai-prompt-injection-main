#!/bin/bash
# squad_v2 + direct + datasentinel + blackbox (alvo = o detect_flag real do DataSentinel).
# Roda main.py e depois xai_metrics.py (mapas de saliência com a saída real do detector,
# deleção, conjunto mínimo). Nada disto foi rodado ainda com o modelo real (precisa de GPU).
# Uso: bash scripts/run_blackbox_datasentinel_squad_v2_direct.sh
#      LIMIT=3 bash scripts/<este>.sh            (smoke test: 3 amostras)
#      BASELINE=neutral bash scripts/<este>.sh   (empty | neutral | clean_context)
# Cada teste é uma geração de um Mistral-7B: comece com LIMIT pequeno e meça o tempo por amostra.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-squad_v2}
ATTACK=${ATTACK:-direct}
DEFENSE=datasentinel
XAI=blackbox
BACKEND_LLM=${BACKEND_LLM:-Qwen/Qwen3-4B-Instruct-2507}
SEED=${SEED:-42}
BASELINE=${BASELINE:-empty}
NAME=blackbox_datasentinel_${DATASET}_${ATTACK}_${BASELINE}
LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")

OUT_ROOT="results/evaluation_results/${NAME}"
RESULT="${OUT_ROOT}/${DATASET}-${BACKEND_LLM//\//-}-${ATTACK}-${DEFENSE}-${XAI}-${SEED}.json"
CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$CONFIG"' EXIT
mkdir -p logs
exec > >(tee -a "logs/${NAME}.log") 2>&1

# `blackbox` não tem flags próprias no main.py: a config entra pelo xai_config do YAML.
cat > "$CONFIG" <<YAML
xai_config:
  baseline: ${BASELINE}
  methods: [ddmin, sentence_shapley]
YAML

python main.py --config "$CONFIG" --dataset "$DATASET" --backend_llm "$BACKEND_LLM" --attack "$ATTACK" \
  --defense "$DEFENSE" --xai "$XAI" \
  --name "$NAME" --seed "$SEED" --no-wandb "${LIMIT_ARGS[@]}"

python scripts/xai_metrics.py --result "$RESULT" --no-wandb
