#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEX_DIR="${ROOT_DIR}/paper/latex"

if [[ -x /Library/TeX/texbin/pdflatex ]]; then
  export PATH="/Library/TeX/texbin:${PATH}"
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found. Run ${ROOT_DIR}/scripts/install_latex_toolchain.sh first." >&2
  exit 1
fi

cd "${LATEX_DIR}"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
fi

echo "Built ${LATEX_DIR}/main.pdf"
