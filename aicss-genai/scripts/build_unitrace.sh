#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PTI_DIR="${ROOT_DIR}/third_party/pti-gpu/tools/unitrace"
LLAMA_COMMIT="${LLAMA_COMMIT:-45cac7ca7}"
IMAGE="${UNITRACE_BUILD_IMAGE:-llama-sycl-local:0.14.0-b8.1-${LLAMA_COMMIT}}"
BUILD_DIR="${PTI_DIR}/build"
BUILD_TYPE="${UNITRACE_BUILD_TYPE:-Release}"
CMAKE_GENERATOR="${UNITRACE_CMAKE_GENERATOR:-Ninja}"

if [[ ! -f "${PTI_DIR}/CMakeLists.txt" ]]; then
    echo "missing PTI unitrace source at ${PTI_DIR}" >&2
    exit 1
fi

cleanup() {
    sudo chown -R "$(id -u):$(id -g)" "${BUILD_DIR}" 2>/dev/null || true
}

trap cleanup EXIT

sudo docker run --rm \
    --entrypoint bash \
    -v "${ROOT_DIR}:/workspace" \
    -w "/workspace/third_party/pti-gpu/tools/unitrace" \
    "${IMAGE}" \
    -lc "
        set -euo pipefail
        git config --global --add safe.directory /workspace/third_party/pti-gpu
        source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
        rm -rf build
        cmake -S . -B build -G '${CMAKE_GENERATOR}' \
            -DCMAKE_BUILD_TYPE='${BUILD_TYPE}' \
            -DBUILD_WITH_MPI=0
        cmake --build build -j \$(nproc)
        ./build/unitrace --version
    "
