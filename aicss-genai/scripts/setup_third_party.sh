#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${ROOT_DIR}/third_party"
PATCHES_DIR="${ROOT_DIR}/patches"

LLAMA_REPO="https://github.com/ggml-org/llama.cpp"
LLAMA_COMMIT="${LLAMA_COMMIT:-45cac7ca7}"

PTI_REPO="https://github.com/intel/pti-gpu.git"
PTI_COMMIT="${PTI_COMMIT:-044440c}"

mkdir -p "${THIRD_PARTY}"

# --- llama.cpp ---
LLAMA_DIR="${THIRD_PARTY}/llama.cpp"
if [[ -d "${LLAMA_DIR}/.git" ]]; then
    CURRENT="$(git -C "${LLAMA_DIR}" rev-parse --short HEAD)"
    echo "llama.cpp already present at ${CURRENT}"
    if [[ "${CURRENT}" != "${LLAMA_COMMIT}" ]]; then
        echo "  warning: expected ${LLAMA_COMMIT}, got ${CURRENT}" >&2
    fi
else
    echo "cloning llama.cpp ..."
    git clone "${LLAMA_REPO}" "${LLAMA_DIR}"
    git -C "${LLAMA_DIR}" checkout "${LLAMA_COMMIT}"
    echo "llama.cpp checked out at ${LLAMA_COMMIT}"
fi

# --- apply patches (skipped for baseline config) ---
BENCH_CONFIG="${LLAMA_BENCH_CONFIG:-optimized}"
if [[ "${BENCH_CONFIG}" == "baseline" ]]; then
    echo "LLAMA_BENCH_CONFIG=baseline: skipping patches (upstream source only)"
else
    PATCH_FILES=("${PATCHES_DIR}"/*.patch)
    if [[ -f "${PATCH_FILES[0]}" ]]; then
        if git -C "${LLAMA_DIR}" diff --quiet -- ggml/src/ggml-sycl/; then
            for PATCH in "${PATCH_FILES[@]}"; do
                echo "applying $(basename "${PATCH}") ..."
                git -C "${LLAMA_DIR}" apply "${PATCH}"
                echo "  applied"
            done
        else
            echo "llama.cpp already has local changes in ggml-sycl/, skipping patches"
        fi
    else
        echo "warning: no patches found in ${PATCHES_DIR}" >&2
    fi
fi

# --- pti-gpu ---
PTI_DIR="${THIRD_PARTY}/pti-gpu"
if [[ -d "${PTI_DIR}/.git" ]]; then
    CURRENT="$(git -C "${PTI_DIR}" rev-parse --short HEAD)"
    echo "pti-gpu already present at ${CURRENT}"
    if [[ "${CURRENT}" != "${PTI_COMMIT}" ]]; then
        echo "  warning: expected ${PTI_COMMIT}, got ${CURRENT}" >&2
    fi
else
    echo "cloning pti-gpu ..."
    git clone "${PTI_REPO}" "${PTI_DIR}"
    git -C "${PTI_DIR}" checkout "${PTI_COMMIT}"
    echo "pti-gpu checked out at ${PTI_COMMIT}"
fi

echo ""
echo "done. third_party layout:"
if [[ "${BENCH_CONFIG}" == "baseline" ]]; then
    echo "  ${LLAMA_DIR}  (${LLAMA_COMMIT})"
    echo "  ${PTI_DIR}  (${PTI_COMMIT})"
else
    echo "  ${LLAMA_DIR}  (${LLAMA_COMMIT} + SYCL patches)"
    echo "  ${PTI_DIR}  (${PTI_COMMIT})"
fi
