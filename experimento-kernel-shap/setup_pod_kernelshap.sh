#!/usr/bin/env bash
# ============================================================================
# Setup do pod (RunPod, ex.: A40 / imagem pytorch:*-cu128*-torch280-*) para
# rodar o experimento --xai kernelshap do PIArena.
#
# Baseado no processo real que funcionou depois de alguns obstáculos:
#   1. piarena/attacks/__init__.py e piarena/defenses/__init__.py importam
#      TODO módulo registrado de forma eager (não-lazy) — então só de
#      importar piarena.attacks/piarena.defenses no main.py, o Python já
#      tenta carregar `datasentinel` (precisa de `fastchat`, do pacote
#      `fschat[model_worker,webui]`) e `datafilter` (precisa de `vllm`),
#      mesmo você usando --attack direct --defense promptguard
#      --xai kernelshap, que não usam nada disso.
#   2. Instalar vllm direto no ambiente do SISTEMA (fora de venv) troca o
#      torch/cuda que já vem pronto na imagem por uma stack CUDA 13
#      inteiramente nova (o vllm puxa nvidia-cuda-runtime/cublas/etc. cu13
#      como dependência) — arriscado.
#   3. A saída: um venv com --system-site-packages. Ele reaproveita o
#      torch/cuda que já está no sistema (não baixa de novo), mas qualquer
#      coisa que o pip decidir instalar/trocar por causa do vllm fica
#      confinada ao venv — o sistema nunca é tocado.
#   4. Mesmo dentro do venv, fschat e vllm são instalados EXPLICITAMENTE
#      (não só via `-r requirements.txt`) porque foi assim que resolvemos
#      cada ModuleNotFoundError na hora, um de cada vez — deixar isso
#      explícito no script documenta o processo real, e é redundante-mas-
#      inofensivo mesmo que requirements.txt já cubra os dois.
#   5. MESMO instalando vllm por completo, o binário pré-compilado dele
#      (`_C_stable_libtorch`) carrega bibliotecas CUDA em tempo de execução
#      via dlopen (libcudart.so.*, libcublas.so.*, etc.) que os pacotes pip
#      `nvidia-*` instalam dentro de site-packages/nvidia/<nome>/lib/ — mas
#      esse diretório não entra no LD_LIBRARY_PATH automaticamente em todo
#      pod/imagem. Sem isso, o `import vllm` quebra com
#      "ImportError: libcudart.so.13: cannot open shared object file".
#      O script resolve isso automaticamente e persiste no `activate` do
#      venv, pra não precisar repetir toda vez que abrir um terminal novo.
#
# Este script SÓ faz o setup. O comando do experimento em si (main.py) é
# pra você rodar manualmente depois — o script imprime o comando no final.
# ============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/piarena-syntax-shap-xai-prompt-injection/piarena_xai_kernel_shap/PIArena-main}"
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
            echo "# --- adicionado por setup_pod_kernelshap.sh (libs CUDA do vllm) ---"
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
echo "6. Confirmando que o vllm importa (é o que mais deu problema)"
echo "=============================================="
python -c "import vllm" && echo "vllm importou OK." || {
    echo "ERRO: vllm ainda não importa mesmo depois do fix do LD_LIBRARY_PATH."
    echo "Rode 'python -c \"import vllm\"' manualmente pra ver o traceback completo"
    echo "e me manda o erro — provavelmente falta mais alguma lib nvidia-* específica"
    echo "(mesmo processo: 'find / -name \"<lib_faltando>*\" 2>/dev/null', acha o dir,"
    echo "acrescenta ao LD_LIBRARY_PATH manualmente e testa de novo)."
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
    huggingface-cli login --token "$HF_TOKEN"
else
    huggingface-cli login
fi

echo ""
echo "=============================================="
echo " Setup concluído."
echo ""
echo " Em qualquer terminal novo, ative o venv de novo antes de rodar algo"
echo " (o LD_LIBRARY_PATH das libs CUDA já fica persistido no activate):"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo " Depois, rode o experimento manualmente:"
echo ""
echo "   cd $PROJECT_DIR"
echo "   python main.py \\"
echo "     --dataset datasets/squad_v2.json \\"
echo "     --backend_llm Qwen/Qwen3-4B-Instruct-2507 \\"
echo "     --attack direct \\"
echo "     --defense promptguard \\"
echo "     --xai kernelshap \\"
echo "     --limit 25 \\"
echo "     --name xai_pilot_kernelshap \\"
echo "     --seed 42"
echo "=============================================="
