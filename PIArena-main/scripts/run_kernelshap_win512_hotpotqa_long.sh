#!/bin/bash
# Prompt-Guard em janelas de 512 tokens (--defense promptguard_windowed: bloqueia se
# QUALQUER janela tiver P(malign) >= 0,5) + Kernel SHAP (captum) em TODAS as janelas,
# para os ataques direct, ignore, completion e combined em hotpotqa_long.
# Ao final de cada run: metricas por janela, grafico de anomalia por janela
# (com rotulos [INJ]/[BLOQUEIO]), mapa de saliencia por janela e bar_global_n1; depois
# o fig9_frequency agregando os 4 ataques.
# Plano: plans/xai-promptguard-512-windows.md
set -euo pipefail

DATASET=hotpotqa_long
DEFENSE=promptguard_windowed
XAI=kernelshap
BACKEND_LLM=Qwen/Qwen3-4B-Instruct-2507
NAME=kernelshap_win512_${DATASET}_captum
SEED=42
WINDOW=512
PERTURBATIONS=16   # coalizões por forward; ~0,35-1,4 GB com janelas de 512 tokens

for ATTACK in direct ignore completion combined; do
  python main.py \
    --dataset "$DATASET" \
    --backend_llm "$BACKEND_LLM" \
    --attack "$ATTACK" \
    --defense "$DEFENSE" \
    --window_tokens "$WINDOW" \
    --xai "$XAI" \
    --xai_backend captum \
    --xai_perturbations_per_eval "$PERTURBATIONS" \
    --name "$NAME" \
    --seed "$SEED" \
    --no-wandb

  RESULT="results/evaluation_results/${NAME}/${DATASET}-${BACKEND_LLM//\//-}-${ATTACK}-${DEFENSE}-${XAI}-${SEED}.json"
  python scripts/xai_metrics.py --result "$RESULT" --no-wandb
  python scripts/xai_window_anomaly.py --result "$RESULT" --max-samples 5
  python scripts/xai_window_saliency.py --result "$RESULT" --max-samples 5
  python scripts/xai_shap_plots_test.py --result "$RESULT" --min-count 1
done

python scripts/xai_fig9_frequency.py --roots "results/evaluation_results/${NAME}"
