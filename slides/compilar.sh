#!/bin/sh
# Uso: ./compilar.sh   (compila slides_xai.tex com o tectonic do venv .venv-tex)
set -e
cd "$(dirname "$0")"
../.venv-tex/bin/tectonic -X compile slides_xai.tex
