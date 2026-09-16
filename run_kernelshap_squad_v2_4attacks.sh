#!/usr/bin/env bash
# Script simples e provisório: roda --xai kernelshap nos 4 ataques
# heurísticos, sequencialmente, contra o squad_v2 completo (200 linhas).
#
# Custo aqui é bem mais contido que o syntaxshap: n_samples é um orçamento
# escolhido (min(2*M_palavras + 2048, 20000)), não derivado da estrutura
# sintática do texto -- sem timeout por amostra porque não faz falta (ver
# experimento-kernel-shap/v1-captum-squad_v2-25amostras.md: 25 amostras em
# ~6min de wall-clock total). Estimativa grosseira pra 200 amostras: ordem de
# ~1h por ataque, ~4h pros 4 juntos -- mas isso nunca foi medido em 200
# amostras reais, só extrapolado do pilot de 25; trate como estimativa, não
# garantia.
#
# Cada ataque continua mesmo se o anterior falhar (sem 'set -e').
#
# Uso:
#   bash run_kernelshap_squad_v2_4attacks.sh
#   nohup bash run_kernelshap_squad_v2_4attacks.sh > logs/run_kernelshap_squad_v2_4attacks.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/PIArena-main"

DATASET="datasets/squad_v2.json"   # arquivo local -- NÃO "squad_v2" (esse
                                    # atalho cai no branch de main.py que
                                    # baixa do HF Hub com force_redownload
                                    # toda vez, ver main.py:99-106)
BACKEND_LLM="Qwen/Qwen3-4B-Instruct-2507"
DEFENSE="promptguard"
XAI="kernelshap"
NAME="kernelshap_squad_v2_4attacks"   # sinaliza dataset+método no --name, pra
                                       # não colidir com outro dataset/sweep
SEED=42
ATTACKS=(direct ignore combined completion)
DATASET_NAME="$(basename "$DATASET" .json)"
LLM_NAME="${BACKEND_LLM//\//-}"

export WANDB_MODE="${WANDB_MODE:-offline}"   # não sobrescreve se você já exportou algo

mkdir -p logs
for attack in "${ATTACKS[@]}"; do
    log="logs/${NAME}_${attack}.log"
    echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] iniciando ${XAI} / squad_v2 / ${attack} (log: ${log}) ====="
    python main.py \
        --dataset "$DATASET" \
        --backend_llm "$BACKEND_LLM" \
        --attack "$attack" \
        --defense "$DEFENSE" \
        --xai "$XAI" \
        --name "$NAME" \
        --seed "$SEED" \
        2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    if [ "$status" -eq 0 ]; then
        echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] OK: ${attack} ====="
    else
        echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] FALHOU (exit $status): ${attack} -- continuando pro próximo ataque ====="
    fi

    # scripts/xai_metrics.py logo depois de cada ataque (não só no final do
    # sweep inteiro) -- reproduz o nome de arquivo que main.py monta
    # (main.py:146-155). Roda mesmo se main.py "falhou" acima: o resultado é
    # salvo incrementalmente por amostra, então um erro no meio do dataset
    # ainda deixa um arquivo parcial válido pra tirar métricas.
    result_file="results/evaluation_results/${NAME}/${DATASET_NAME}-${LLM_NAME}-${attack}-${DEFENSE}-${XAI}-${SEED}.json"
    metrics_log="logs/${NAME}_${attack}_metrics.log"
    if [ -f "$result_file" ]; then
        echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] rodando xai_metrics.py (${attack}) (log: ${metrics_log}) ====="
        python scripts/xai_metrics.py --result "$result_file" 2>&1 | tee "$metrics_log"
        metrics_status=${PIPESTATUS[0]}
        if [ "$metrics_status" -eq 0 ]; then
            echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] OK: xai_metrics.py (${attack}) ====="
        else
            echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] FALHOU (exit $metrics_status): xai_metrics.py (${attack}) ====="
        fi
    else
        echo "aviso: resultado não encontrado em ${result_file} -- pulando xai_metrics.py pra ${attack}"
    fi
done

echo "Terminado. Resultados em results/evaluation_results/${NAME}/, logs em logs/${NAME}_*.log"
