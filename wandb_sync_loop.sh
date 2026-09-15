#!/usr/bin/env bash
# Sincroniza as runs offline do W&B (escritas por main.py/scripts/*.py com
# WANDB_MODE=offline) pro wandb.ai remoto, a cada SYNC_INTERVAL_SECONDS
# (default 300s = 5min), rodando em loop até ser interrompido -- assim o
# painel do wandb.ai fica quase ao vivo sem precisar de SSH nem lembrar de
# rodar `wandb sync` manualmente.
#
# `run_full_sweep.py` já sobe este script sozinho em background no início
# (a não ser que --no-wandb-sync seja passado) e o derruba no final -- rodar
# na mão só é necessário se quiser esse comportamento fora do orquestrador:
#   nohup bash wandb_sync_loop.sh > logs/wandb_sync.log 2>&1 &
#
# PIArena-main/ escreve seu próprio wandb/ local (main.py/scripts/*.py rodam
# com cwd=PIArena-main -- ver run_full_sweep.py), então o sync roda dentro
# dela -- um `wandb sync --sync-all` genérico na raiz do workspace não a
# acharia. (Antes do merge de piarena/xai/kernelshap/ de volta pra esta
# árvore, syntaxshap e kernelshap viviam em duas árvores/dois `wandb/`
# separados; este loop sincronizava as duas.)
#
# Uma falha de sync (rede caiu, `wandb login` ainda não foi feito) não pode
# matar o loop nem, principalmente, a run principal -- por isso todo
# `wandb sync` aqui roda com `|| true`. Sem login/API key configurados, o
# sync simplesmente falha e tenta de novo no próximo ciclo -- os dados
# continuam seguros no disco em wandb/offline-run-*/ até você rodar
# `wandb login` e deixar este loop (ou você mesmo) sincronizar depois.

set -uo pipefail   # sem -e: uma falha de sync não pode derrubar o loop
cd "$(dirname "$0")"

INTERVAL="${SYNC_INTERVAL_SECONDS:-300}"
PIARENA_DIR="PIArena-main"

echo "wandb_sync_loop.sh: sincronizando a cada ${INTERVAL}s (Ctrl+C ou SIGTERM pra parar)."

while true; do
    ts="$(date +%H:%M:%S)"
    if [ -d "$PIARENA_DIR/wandb" ]; then
        echo "[$ts] wandb sync ($PIARENA_DIR)..."
        (cd "$PIARENA_DIR" && wandb sync --sync-all) || echo "[$ts] sync falhou em $PIARENA_DIR (ok, tenta de novo no proximo ciclo)"
    fi
    sleep "$INTERVAL"
done
