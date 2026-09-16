#!/usr/bin/env bash
# Script simples e provisório: roda --xai syntaxshap nos 4 ataques
# heurísticos, sequencialmente, contra o squad_v2 completo (200 linhas).
#
# ATENÇÃO custo: syntaxshap tem timeout de 2400s/40min por amostra
# (SyntaxShapXAI.DEFAULT_CONFIG["timeout_seconds"], piarena/xai/syntaxshap/
# xai_syntaxshap.py) -- não trava pra sempre numa amostra patológica, mas com
# 200 linhas por ataque, se uma fração relevante baterem no teto (o esperado
# para squad_v2 sem curadoria -- ver PIArena-main/CLAUDE.md, "Computational
# cost bottleneck": mediana ~34min/amostra JÁ é quase o próprio timeout, e o
# p90 é 22.5h SEM timeout), um único ataque pode levar de várias horas a
# alguns dias. Os 4 juntos, sem curadoria, são potencialmente uma rodada de
# múltiplos dias. Rode isso dentro de tmux/screen ou com nohup (ver exemplo
# no fim deste arquivo) -- se a conexão SSH cair, o processo continua.
#
# Cada ataque continua mesmo se o anterior falhar (sem 'set -e') -- não
# queremos perder o resto de um sweep de dias por causa de UM erro.
#
# Uso:
#   bash run_syntaxshap_squad_v2_4attacks.sh
#   nohup bash run_syntaxshap_squad_v2_4attacks.sh > logs/run_syntaxshap_squad_v2_4attacks.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/PIArena-main"

DATASET="datasets/squad_v2.json"   # arquivo local -- NÃO "squad_v2" (esse
                                    # atalho cai no branch de main.py que
                                    # baixa do HF Hub com force_redownload
                                    # toda vez, ver main.py:99-106)
BACKEND_LLM="Qwen/Qwen3-4B-Instruct-2507"
DEFENSE="promptguard"
XAI="syntaxshap"
NAME="syntaxshap_squad_v2_4attacks"   # sinaliza dataset+método no --name, pra
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
