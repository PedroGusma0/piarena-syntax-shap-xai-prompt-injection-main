#!/bin/sh
# Uso: ./compilar.sh arquivo.tex   (compila duas vezes com XeLaTeX)
set -e
f="${1:-exemplo.tex}"
xelatex -interaction=nonstopmode -halt-on-error "$f" >/dev/null
xelatex -interaction=nonstopmode -halt-on-error "$f"  | tail -2
