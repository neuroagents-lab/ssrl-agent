#!/usr/bin/env bash
# Use the active Python environment by default. Set CONDA_ENV explicitly to
# activate a named Conda environment before training.
if [[ -n "${CONDA_ENV:-}" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  else
    echo "CONDA_ENV requires Conda; set CONDA_EXE to its bin/conda executable." >&2
    return 1
  fi
  if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "Cannot load Conda from ${CONDA_BASE}." >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi
fi
