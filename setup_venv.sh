#!/usr/bin/env bash
# ============================================================================
# Setup do pod (RunPod, ex.: A40 / imagem pytorch:*-cu128*-torch280-*) para
# rodar o sweep completo (run_full_sweep.py: --xai syntaxshap E --xai
# kernelshap, nas duas árvores PIArena-main/ e
# piarena_xai_kernel_shap/PIArena-main/) num único venv compartilhado.
#
# Baseado em `experimento-kernel-shap/setup_pod_kernelshap.sh` -- o processo
# que REALMENTE funcionou rodando o pilot kernelshap num pod de verdade (ver
# run_rodando.txt) -- estendido aqui pra cobrir as duas árvores de uma vez,
# não só a do kernelshap. Os passos 1-3, 5, 7-10 e 13 abaixo são
# essencialmente os mesmos daquele script (mesmos obstáculos reais, mesmas
# soluções); o resto (spacy, captum, requirements.txt combinado, instalação
# editável) é novo, pra cobrir a árvore syntaxshap também.
#
# Uso:
#   bash setup_venv.sh
#   VENV_DIR=/outro/caminho bash setup_venv.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash setup_venv.sh   # só usado se o torch do sistema não estiver disponível (passo 3)
#   HF_TOKEN="hf_..." bash setup_venv.sh             # pula o login interativo
#
# Depois de rodar, ative o venv (as duas árvores usam o mesmo) e rode:
#   source .venv/bin/activate
#   python run_full_sweep.py
#
# --- Por que vllm E fschat são OBRIGATÓRIOS, não opcionais ---
# `piarena/attacks/__init__.py` e `piarena/defenses/__init__.py` importam
# TODO módulo registrado de forma eager (não-lazy, `from . import
# attack_strategy_search`/`from .datafilter import defense_datafilter`/`from
# .datasentinel import defense_datasentinel` no topo do arquivo, incondicional)
# -- confirmado lendo o código das duas árvores. Então só de importar
# `piarena.attacks`/`piarena.defenses` (o que `main.py` já faz sempre), o
# Python tenta carregar `attack_strategy_search` (precisa de `vllm`),
# `defense_datafilter` (precisa de `vllm`) e `defense_datasentinel` (precisa
# de `fastchat`, do pacote `fschat[model_worker,webui]`) -- MESMO rodando
# `--attack direct --defense promptguard`, que não usa nada disso. Isso vale
# pras DUAS árvores igualmente (o código é o mesmo nos dois `piarena/`).
#
# --- Por que --system-site-packages (não reinstalar torch do zero) ---
# Instalar vllm direto no Python do SISTEMA (fora de venv) troca o
# torch/CUDA que já vem pronto na imagem do pod por uma stack CUDA nova
# inteira (vllm puxa nvidia-cuda-runtime/cublas/etc. como dependência) --
# arriscado mexer no sistema todo. Um venv com --system-site-packages
# reaproveita o torch/CUDA que já está no sistema (não baixa de novo, não
# precisa adivinhar o índice CUDA certo pra esse pod) e confina qualquer
# coisa que o pip decidir instalar/trocar por causa do vllm ao venv -- o
# sistema nunca é tocado. Se o torch do sistema não tiver CUDA disponível
# (pod sem GPU ainda, ou imagem sem torch pré-instalado), o passo 3 cai pra
# instalar torch pinado via TORCH_INDEX_URL como fallback.
#
# --- Por que vllm é instalado SEM pin ---
# Dentro do venv (não no sistema), deixar o pip resolver `vllm` livremente é
# seguro -- ele já enxerga o torch do sistema via --system-site-packages
# (satisfaz a dependência de torch do vllm sem o pip precisar procurar uma
# combinação do zero, que é o que causava o backtracking de 100+ releases
# quando torch TAMBÉM estava sem pin). Risco real, visto na prática: o vllm
# resolvido pode pedir uma versão de torch mais nova do que o driver CUDA
# deste pod suporta (ex.: vllm exigindo torch+cu130 num pod com driver
# CUDA 12.8) -- o passo logo abaixo detecta isso e tenta corrigir sozinho
# (reinstalar a versão exata que o vllm pede, mas do índice CUDA certo pra
# este driver). Se mesmo assim não der certo, o jeito manual é pinar uma
# release mais antiga de vllm compatível com o torch que funciona aqui, ex.:
# `pip install "vllm==0.10.0"` (ajuste o número — não existe um valor certo
# universal, depende de qual torch aquele release do vllm pede).
#
# --- Por que as bibliotecas CUDA do vllm precisam de um fix de LD_LIBRARY_PATH ---
# O binário pré-compilado do vllm carrega bibliotecas CUDA em tempo de
# execução via dlopen (libcudart.so.*, libcublas.so.*, etc.) que os pacotes
# pip `nvidia-*` (dependências do próprio vllm/torch) instalam dentro de
# site-packages/nvidia/<nome>/lib/ -- mas esse diretório não entra no
# LD_LIBRARY_PATH automaticamente em todo pod/imagem. Sem isso, `import
# vllm` quebra com "ImportError: libcudart.so.13: cannot open shared object
# file". O passo 8 resolve isso e persiste no `activate` do venv.
#
# --- Por que só UMA árvore fica com `pip install -e .` ---
# PIArena-main/ e piarena_xai_kernel_shap/PIArena-main/ têm cada uma seu
# próprio pacote `piarena/` (a segunda com piarena/xai/kernelshap/ em vez de
# piarena/xai/syntaxshap/) -- MESMO NOME de pacote, então só uma pode ser a
# instalação editável do venv por vez (passo 12 só instala a de
# PIArena-main/). Isso não quebra `run_full_sweep.py`: ele seta PYTHONPATH
# pra árvore certa em cada subprocesso (ver `_env_for()` nele), então
# `main.py`/`scripts/*.py` sempre importam o `piarena` da pasta certa, não
# importa qual delas está "editável" no venv. Rodar `scripts/xai_metrics.py`
# manualmente dentro da árvore kernelshap (fora do orquestrador) precisa do
# mesmo cuidado:
#   cd piarena_xai_kernel_shap/PIArena-main && PYTHONPATH="$(pwd)" python scripts/xai_metrics.py ...
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"   # -> raiz do workspace

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"                                    # só usado no fallback do passo 3
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"  # idem
SYNTAXSHAP_DIR="$(pwd)/PIArena-main"
KERNELSHAP_DIR="$(pwd)/piarena_xai_kernel_shap/PIArena-main"

echo "=============================================="
echo "1. Sanity check da GPU"
echo "=============================================="
nvidia-smi || echo "aviso: nvidia-smi falhou -- ok se este pod ainda não tem GPU anexada; main.py vai exigir uma pra rodar de verdade."

echo ""
echo "=============================================="
echo "2. Criando venv em $VENV_DIR (--system-site-packages: reaproveita torch/CUDA da imagem)"
echo "=============================================="
if [ -d "$VENV_DIR" ]; then
    echo "Venv já existe em $VENV_DIR -- reaproveitando (apague a pasta pra recriar do zero)."
else
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo ""
echo "=============================================="
echo "3. torch/CUDA: reaproveitando do sistema, ou instalando pinado como fallback"
echo "=============================================="
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "torch do sistema já funciona com CUDA (via --system-site-packages) -- não reinstalando."
    python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
else
    echo "torch do sistema não encontrado ou sem CUDA disponível -- instalando pinado."
    echo "--- Instalando torch==$TORCH_VERSION (index: $TORCH_INDEX_URL) ---"
    # --force-reinstall --no-cache-dir: garante torch/torchvision/torchaudio
    # vindos do MESMO índice/build -- misturar builds diferentes quebra a
    # extensão compilada (visto na prática como
    # `OSError: undefined symbol: torch_library_impl`).
    pip install --force-reinstall --no-cache-dir \
        "torch==$TORCH_VERSION" torchvision torchaudio --index-url "$TORCH_INDEX_URL"
fi

echo ""
echo "=============================================="
echo "4. Dependências da árvore syntaxshap (PIArena-main/requirements.txt --"
echo "   cobre a base das duas árvores: transformers, spacy, pandas, shap,"
echo "   matplotlib, fschat, etc., já sem vllm/torch soltos)"
echo "=============================================="
pip install -r "$SYNTAXSHAP_DIR/requirements.txt"

echo ""
echo "=============================================="
echo "5. fschat explícito (import eager de datasentinel -- ver nota no cabeçalho)"
echo "=============================================="
pip install "fschat[model_worker,webui]"

echo ""
echo "=============================================="
echo "6. captum (único pacote extra que a árvore kernelshap precisa além do"
echo "   requirements.txt acima -- kernelshap backend='captum', o default)"
echo "=============================================="
pip install captum

echo ""
echo "=============================================="
echo "7. vllm, sem pin, DENTRO do venv (import eager de datafilter/"
echo "   strategy_search -- ver nota no cabeçalho)"
echo "=============================================="
TORCH_BEFORE_VLLM="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '')"
pip install vllm

# vllm puxa sua própria versão de torch/torchvision como dependência -- se
# isso trocar pra um build que exige um driver CUDA mais novo do que o pod
# realmente tem (visto na prática: driver reportando CUDA 12.8, vllm puxando
# torch+cu130), `torch.cuda.is_available()` vira False silenciosamente, e só
# quebraria lá no final (passo 10) sem dizer o motivo. Detectar e corrigir
# aqui, logo depois do vllm, em vez de deixar pra descobrir quebrado no final.
TORCH_AFTER_VLLM="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '')"
if [ -n "$TORCH_BEFORE_VLLM" ] && [ "$TORCH_AFTER_VLLM" != "$TORCH_BEFORE_VLLM" ]; then
    echo "vllm trocou o torch ($TORCH_BEFORE_VLLM -> $TORCH_AFTER_VLLM) -- conferindo se a GPU ainda funciona com esse build..."
    if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "torch $TORCH_AFTER_VLLM (puxado pelo vllm) não enxerga a GPU neste driver." >&2

        # vllm normalmente embute um binário pré-compilado (_C_stable_libtorch.*.so)
        # que exige EXATAMENTE a versão de torch/torchvision que ele declarou como
        # dependência -- reinstalar um torch mais antigo por cima (o que essa
        # correção fazia antes) deixa a GPU visível mas quebra esse binário com
        # "undefined symbol" (ABI incompatível). Em vez de reverter às cegas,
        # primeiro tenta reinstalar a MESMA versão que o vllm pede, só que vinda
        # do índice CUDA certo pra este driver (o PyTorch costuma publicar builds
        # pra várias versões de CUDA em paralelo do mesmo release).
        VLLM_TORCH_PINS="$(python -c "
import importlib.metadata as m
for r in (m.requires('vllm') or []):
    spec = r.split(';')[0].strip()
    name = spec.split('==')[0].strip()
    if name in ('torch', 'torchvision') and '==' in spec:
        print(spec)
" 2>/dev/null)"
        if [ -n "$VLLM_TORCH_PINS" ]; then
            echo "vllm pede exatamente: $(echo "$VLLM_TORCH_PINS" | tr '\n' ' ') -- tentando essa versão via $TORCH_INDEX_URL." >&2
            # shellcheck disable=SC2086
            pip install --force-reinstall --no-cache-dir $VLLM_TORCH_PINS torchaudio --index-url "$TORCH_INDEX_URL"
        fi

        if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            echo "Ainda sem GPU com a versão exata do vllm (não existe build pra $TORCH_INDEX_URL," >&2
            echo "ou o vllm não declarou uma versão exata) -- revertendo pro torch $TORCH_BEFORE_VLLM" >&2
            echo "que já tinha funcionado (aviso: isso pode quebrar 'import vllm' no passo 9 por" >&2
            echo "incompatibilidade de ABI -- se acontecer, é preciso pinar uma versão mais antiga" >&2
            echo "de vllm compatível com esse torch, ex.: pip install 'vllm<X.Y' e repetir a partir daqui)." >&2
            pip install --force-reinstall --no-cache-dir \
                "torch==${TORCH_BEFORE_VLLM%%+*}" torchvision torchaudio --index-url "$TORCH_INDEX_URL"
        fi

        if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            echo "ERRO: nenhuma combinação de torch tentada conseguiu enxergar a GPU." >&2
            echo "Pare aqui -- rode 'nvidia-smi' e confira a versão real do driver deste pod," >&2
            echo "e escolha um TORCH_INDEX_URL compatível com ela antes de continuar." >&2
            exit 1
        fi
        echo "GPU visível de novo. Confira 'import vllm' no passo 9 logo abaixo --" >&2
        echo "se ainda quebrar, pode ser preciso pinar uma versão de vllm compatível" >&2
        echo "com o torch que ficou instalado e repetir a partir daqui." >&2
    else
        echo "torch $TORCH_AFTER_VLLM (puxado pelo vllm) já enxerga a GPU -- ok, mantendo."
    fi
fi

echo ""
echo "=============================================="
echo "8. Corrigindo o LD_LIBRARY_PATH pras libs CUDA que o vllm carrega em"
echo "   runtime (libcudart.so.*, libcublas.so.*, etc., instaladas pelos"
echo "   pacotes pip nvidia-* dentro de site-packages/nvidia/<nome>/lib/)"
echo "=============================================="
NVIDIA_LIB_DIRS=$(python -c "
import glob, os, sys
dirs = []
for sp in sys.path:
    if sp.endswith('site-packages'):
        dirs.extend(glob.glob(os.path.join(sp, 'nvidia', '*', 'lib')))
print(':'.join(dirs))
")
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    if ! grep -q "NVIDIA_LIB_DIRS" "$VENV_DIR/bin/activate" 2>/dev/null; then
        {
            echo ""
            echo "# --- adicionado por setup_venv.sh (libs CUDA do vllm) ---"
            echo "export LD_LIBRARY_PATH=\"$NVIDIA_LIB_DIRS:\${LD_LIBRARY_PATH:-}\""
        } >> "$VENV_DIR/bin/activate"
    fi
    export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS:${LD_LIBRARY_PATH:-}"
    echo "LD_LIBRARY_PATH atualizado (persistido no activate do venv) com:"
    echo "$NVIDIA_LIB_DIRS" | tr ':' '\n'
else
    echo "Nenhum diretório nvidia/*/lib encontrado em site-packages -- pulando (pode não ser necessário nessa imagem)."
fi

echo ""
echo "=============================================="
echo "9. Confirmando que o vllm importa (é o que mais dá problema)"
echo "=============================================="
python -c "import vllm" && echo "vllm importou OK." || {
    echo "ERRO: vllm ainda não importa mesmo depois do fix do LD_LIBRARY_PATH." >&2
    echo "Rode 'python -c \"import vllm\"' manualmente pra ver o traceback completo --" >&2
    echo "provavelmente falta mais alguma lib nvidia-* específica: 'find / -name" >&2
    echo "\"<lib_faltando>*\" 2>/dev/null', acha o dir, acrescenta ao LD_LIBRARY_PATH" >&2
    echo "manualmente e testa de novo." >&2
    exit 1
}

echo ""
echo "=============================================="
echo "10. Confirmando que a GPU continua visível pro torch depois de tudo"
echo "=============================================="
python -c "
import torch
print('torch:', torch.__version__)
print('CUDA disponível:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    raise SystemExit('ERRO: torch não está enxergando a GPU depois da instalação. Pare aqui e investigue antes de continuar.')
"

echo ""
echo "=============================================="
echo "11. Modelo do spaCy (árvore syntaxshap: get_token_dependency_tree)"
echo "=============================================="
python -m spacy download en_core_web_sm

echo ""
echo "=============================================="
echo "12. pip install -e PIArena-main (conveniência pra rodar scripts/*.py na"
echo "    mão -- só a árvore syntaxshap, ver nota no cabeçalho)"
echo "=============================================="
pip install -e "$SYNTAXSHAP_DIR"

echo ""
echo "=============================================="
echo "13. Login no Hugging Face (meta-llama/Prompt-Guard-86M é gated)"
echo "    Defina HF_TOKEN no ambiente antes de rodar este script pra pular"
echo "    o prompt interativo, ex.: export HF_TOKEN=hf_xxx"
echo "=============================================="
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN"
else
    huggingface-cli login
fi

echo ""
echo "=============================================="
echo "14. Login no W&B (opcional agora -- NÃO é necessário pra gravar as runs"
echo "    offline, só pro wandb_sync_loop.sh conseguir enviar depois pro"
echo "    wandb.ai remoto). Defina WANDB_API_KEY no ambiente pra pular o"
echo "    prompt interativo, ex.: export WANDB_API_KEY=xxxx. Pulando este"
echo "    passo agora é seguro -- rode 'wandb login' manualmente mais tarde"
echo "    quando quiser sincronizar; até lá o sync do loop só vai falhar"
echo "    (com aviso, sem quebrar nada) e tentar de novo no próximo ciclo."
echo "=============================================="
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY"
else
    wandb login || echo "wandb login não concluído agora -- ok, rode 'wandb login' manualmente depois pra habilitar o sync."
fi

echo ""
echo "=============================================="
echo " Setup concluído."
echo ""
echo " Em qualquer terminal novo, ative o venv de novo antes de rodar algo"
echo " (o LD_LIBRARY_PATH das libs CUDA já fica persistido no activate):"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo " Depois, rode o sweep completo (já sobe o wandb_sync_loop.sh sozinho"
echo " em background -- painel do wandb.ai atualiza sem precisar de SSH):"
echo "   python run_full_sweep.py"
echo "=============================================="
