#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ensure_brew() {
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required to install the LaTeX toolchain on macOS." >&2
    exit 1
  fi
}

ensure_tex_path() {
  if [[ -x /Library/TeX/texbin/pdflatex ]]; then
    export PATH="/Library/TeX/texbin:${PATH}"
  elif [[ -x /usr/local/texlive/2026basic/bin/universal-darwin/pdflatex ]]; then
    export PATH="/usr/local/texlive/2026basic/bin/universal-darwin:${PATH}"
  fi
}

install_basictex_macos() {
  ensure_brew
  ensure_tex_path
  if command -v pdflatex >/dev/null 2>&1; then
    return 0
  fi

  echo "Installing BasicTeX via Homebrew cask..."
  brew install --cask basictex
  eval "$(/usr/libexec/path_helper)"
  ensure_tex_path
}

install_tex_packages() {
  ensure_tex_path

  if ! command -v tlmgr >/dev/null 2>&1; then
    echo "tlmgr not found after BasicTeX installation." >&2
    exit 1
  fi

  tlmgr update --self
  tlmgr install \
    latexmk \
    pgf \
    pgfplots \
    natbib \
    parskip \
    enumitem \
    booktabs \
    microtype \
    caption \
    subcaption \
    geometry
}

main() {
  case "$(uname -s)" in
    Darwin)
      install_basictex_macos
      ;;
    *)
      echo "Unsupported platform for automatic LaTeX toolchain installation: $(uname -s)" >&2
      exit 1
      ;;
  esac

  install_tex_packages

  echo "LaTeX toolchain ready."
  echo "pdflatex: $(command -v pdflatex)"
  echo "latexmk: $(command -v latexmk)"
  echo "bibtex: $(command -v bibtex)"
  echo "Project root: ${ROOT_DIR}"
}

main "$@"
