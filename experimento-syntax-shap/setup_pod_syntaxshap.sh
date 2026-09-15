#!/usr/bin/env bash
# ============================================================================
# Setup do pod (RunPod, ex.: A40 / imagem pytorch:*-cu128*-torch280-*) para
# rodar o experimento --xai syntaxshap do PIArena — baseado em
# experimento-kernel-shap/setup_pod_kernelshap.sh (mesma estratégia de venv,
# testada e funcional de verdade num pod real), adaptado para o que o
# SyntaxSHAP precisa a mais (spaCy) e para rodar direto no PROJECT_DIR
# principal (PIArena-main/ na raiz do repo, não um clone aninhado por
# experimento).
#
# Lições herdadas do setup do kernelshap (ver esse script pros detalhes):
#   1. piarena/attacks/__init__.py e piarena/defenses/__init__.py importam
#      TODO módulo registrado de forma eager (não-lazy) — então só de
#      importar piarena.attacks/piarena.defenses no main.py, o Python já
#      tenta carregar `datasentinel` (precisa de `fastchat`, do pacote
#      `fschat[model_worker,webui]`) e `datafilter` (precisa de `vllm`),
#      MESMO você usando --attack direct --defense promptguard
#      --xai syntaxshap, que não usam nada disso. Ou seja: comentar `vllm`/
#      `fschat` do requirements.txt (como foi feito numa correção anterior,
#      só pra destravar `estimate_xai_cost.py` e o resolvedor de pip) NÃO
#      basta pra rodar `main.py` de verdade — ele quebra no import antes de
#      chegar no seu código. Esse script instala os dois explicitamente,
#      igual o do kernelshap.
#   2. Instalar vllm direto no ambiente do SISTEMA (fora de venv) troca o
#      torch/cuda que já vem pronto na imagem por uma stack CUDA nova
#      inteira (vllm puxa nvidia-cuda-runtime/cublas/etc. como dependência)
#      — arriscado, e foi uma fonte de bugs reais de ABI (torch/torchaudio
#      incompatíveis) numa tentativa anterior com conda + torch reinstalado
#      manualmente. A saída: venv com --system-site-packages, que reaproveita
#      o torch/cuda que já está no sistema (não baixa de novo, não troca
#      versão) — qualquer coisa que o pip decidir instalar por causa do vllm
#      fica confinada ao venv.
#   3. Mesmo dentro do venv, fschat e vllm são instalados EXPLICITAMENTE
#      (não só via `-r requirements.txt`), documentando o processo real.
#   4. O binário pré-compilado do vllm carrega libs CUDA em runtime via
#      dlopen (libcudart.so.*, libcublas.so.*, etc.) que os pacotes pip
#      `nvidia-*` instalam dentro de site-packages/nvidia/<nome>/lib/ — esse
#      diretório não entra no LD_LIBRARY_PATH automaticamente. O script
#      resolve isso e persiste no `activate` do venv.
#
# Diferença específica do SyntaxSHAP (vs. kernelshap): precisa de spaCy +
# modelo `en_core_web_sm` (árvore de dependência) — kernelshap não usa isso.
# `pandas`/`cloudpickle`/`shap`/`matplotlib` já estão no requirements.txt.
#
# Este script SÓ faz o setup — igual o do kernelshap. Os comandos do
# cost-check (scripts/estimate_xai_cost.py) e do experimento em si
# (main.py) são pra rodar manualmente depois; o script imprime os dois no
# final.
# ============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/piarena-syntax-shap-xai-prompt-injection/PIArena-main}"
VENV_DIR="${VENV_DIR:-$HOME/piarena-venv}"

echo "=============================================="
echo "1. Sanity check da GPU"
echo "=============================================="
nvidia-smi

echo ""
echo "=============================================="
echo "2. Criando venv em $VENV_DIR (--system-site-packages: reaproveita torch/cuda da imagem)"
echo "=============================================="
if [ -d "$VENV_DIR" ]; then
    echo "Venv já existe em $VENV_DIR, reaproveitando."
else
    python -m venv --system-site-packages "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo ""
echo "=============================================="
echo "3. Entrando na pasta do projeto: $PROJECT_DIR"
echo "=============================================="
cd "$PROJECT_DIR"

echo ""
echo "=============================================="
echo "4a. Instalando o grosso das dependências do requirements.txt"
echo "    (torch NÃO está aqui — vem pronto da imagem via --system-site-packages)"
echo "=============================================="
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo ""
echo "=============================================="
echo "4b. Instalando fschat explicitamente (import eager de datasentinel)"
echo "=============================================="
pip install "fschat[model_worker,webui]"

echo ""
echo "=============================================="
echo "4c. Instalando vllm explicitamente, SEM --no-deps (import eager de"
echo "    datafilter — dentro do venv é seguro deixar resolver por completo)"
echo "=============================================="
pip install vllm

echo ""
echo "=============================================="
echo "4d. Baixando o modelo spaCy (árvore de dependência do SyntaxSHAP,"
echo "    necessário tanto pro estimate_xai_cost.py quanto pro main.py)"
echo "=============================================="
python -m spacy download en_core_web_sm

echo ""
echo "=============================================="
echo "5. Corrigindo o LD_LIBRARY_PATH pras libs CUDA que o vllm carrega em"
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
            echo "# --- adicionado por setup_pod_syntaxshap.sh (libs CUDA do vllm) ---"
            echo "export LD_LIBRARY_PATH=\"$NVIDIA_LIB_DIRS:\${LD_LIBRARY_PATH:-}\""
        } >> "$VENV_DIR/bin/activate"
    fi
    export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS:${LD_LIBRARY_PATH:-}"
    echo "LD_LIBRARY_PATH atualizado (persistido no activate do venv) com:"
    echo "$NVIDIA_LIB_DIRS" | tr ':' '\n'
else
    echo "Nenhum diretório nvidia/*/lib encontrado em site-packages — pulando (pode não ser necessário nessa imagem)."
fi

echo ""
echo "=============================================="
echo "6. Confirmando que vllm e spaCy importam (o que mais dá problema)"
echo "=============================================="
python -c "import vllm" && echo "vllm importou OK." || {
    echo "ERRO: vllm ainda não importa mesmo depois do fix do LD_LIBRARY_PATH."
    echo "Rode 'python -c \"import vllm\"' manualmente pra ver o traceback completo"
    echo "e reporta o erro — provavelmente falta mais alguma lib nvidia-* específica"
    echo "(mesmo processo: 'find / -name \"<lib_faltando>*\" 2>/dev/null', acha o dir,"
    echo "acrescenta ao LD_LIBRARY_PATH manualmente e testa de novo)."
    exit 1
}
python -c "import spacy; spacy.load('en_core_web_sm')" && echo "spaCy + en_core_web_sm OK." || {
    echo "ERRO: spaCy não carregou en_core_web_sm."
    exit 1
}

echo ""
echo "=============================================="
echo "7. Confirmando que a GPU continua visível pro torch depois de tudo"
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
echo "8. Login no Hugging Face (meta-llama/Prompt-Guard-86M é gated)"
echo "   Defina HF_TOKEN no ambiente antes de rodar este script pra pular"
echo "   o prompt interativo, ex.: export HF_TOKEN=hf_xxx"
echo "=============================================="
if [ -n "${HF_TOKEN:-}" ]; then
    if command -v hf >/dev/null 2>&1; then
        hf auth login --token "$HF_TOKEN"
    else
        huggingface-cli login --token "$HF_TOKEN"
    fi
else
    if command -v hf >/dev/null 2>&1; then
        hf auth login
    else
        huggingface-cli login
    fi
fi

echo ""
echo "=============================================="
echo " Setup concluído."
echo ""
echo " Em qualquer terminal novo, ative o venv de novo antes de rodar algo"
echo " (o LD_LIBRARY_PATH das libs CUDA já fica persistido no activate):"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo " Depois, rode manualmente:"
echo ""
echo " (a) cost-check da amostra (CPU-only, sem GPU/modelo gated — confirma"
echo "     que a árvore de dependência não explode antes de gastar tempo de"
echo "     GPU de verdade; ver markdowns-do-experimento/"
echo "     custo-computacional-syntaxshap.md):"
echo ""
echo "   cd $PROJECT_DIR"
echo "   python scripts/estimate_xai_cost.py \\"
echo "     --dataset datasets/squad_v2.json \\"
echo "     --indices 3 \\"
echo "     --attack direct"
echo ""
echo " (b) monta um dataset de 1 linha pro índice escolhido em (a) — --limit"
echo "     pega as N PRIMEIRAS linhas em ordem, não um índice específico:"
echo ""
echo "   python -c \"import json; d=json.load(open('datasets/squad_v2.json')); json.dump([d[3]], open('datasets/squad_v2_pilot_idx3.json','w'), indent=2)\""
echo ""
echo " (c) o experimento em si:"
echo ""
echo "   python main.py \\"
echo "     --dataset datasets/squad_v2_pilot_idx3.json \\"
echo "     --backend_llm Qwen/Qwen3-4B-Instruct-2507 \\"
echo "     --attack direct \\"
echo "     --defense promptguard \\"
echo "     --xai syntaxshap \\"
echo "     --name xai_pilot_syntaxshap \\"
echo "     --seed 42"
echo "=============================================="
