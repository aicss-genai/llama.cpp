#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT_DIR}/third_party/llama.cpp"
CONTAINER_ROOT="/workspace"
CONTAINER_LLAMA_DIR="${CONTAINER_ROOT}/third_party/llama.cpp"
IMAGE="${LLAMA_SYCL_IMAGE:-intel/llm-scaler-vllm:0.14.0-b8.1}"
TARGETS="${LLAMA_SYCL_TARGETS:-llama-cli llama-bench llama-batched-bench llama-server llama-gguf-split}"
IMAGE_TAG_SAFE="${IMAGE##*:}"
IMAGE_TAG_SAFE="${IMAGE_TAG_SAFE//[^A-Za-z0-9._-]/-}"
BENCH_CONFIG="${LLAMA_BENCH_CONFIG:-optimized}"
case "${BENCH_CONFIG}" in
    optimized|baseline) ;;
    *) echo "ERROR: unknown LLAMA_BENCH_CONFIG '${BENCH_CONFIG}'; expected 'optimized' or 'baseline'" >&2; exit 1 ;;
esac

if [[ ! -d "${LLAMA_DIR}" ]]; then
    echo "missing llama.cpp checkout at ${LLAMA_DIR}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Baseline: cmake build inside the same Intel container, but with only the
# minimal SYCL flags and *no* patches applied.  We build from the clean
# HEAD checkout so the comparison is against upstream code, while still
# using the runtime image whose compute-runtime supports Battlemage.
# ---------------------------------------------------------------------------
if [[ "${BENCH_CONFIG}" == "baseline" ]]; then
    BUILD_DIR="${LLAMA_DIR}/build-sycl-baseline-${IMAGE_TAG_SAFE}"
    CONTAINER_BUILD_DIR="${CONTAINER_LLAMA_DIR}/build-sycl-baseline-${IMAGE_TAG_SAFE}"

    echo "Building llama.cpp SYCL [config=baseline] -> ${BUILD_DIR}"

    cleanup() {
        sudo chown -R "$(id -u):$(id -g)" "${BUILD_DIR}" 2>/dev/null || true
    }
    trap cleanup EXIT

    sudo docker run --rm \
        --entrypoint bash \
        --device /dev/dri:/dev/dri \
        -v "${ROOT_DIR}:/workspace" \
        -w "${CONTAINER_LLAMA_DIR}" \
        "${IMAGE}" \
        -lc "
            set -euo pipefail
            git config --global --add safe.directory '${CONTAINER_LLAMA_DIR}'
            # Ensure we build from clean HEAD (no uncommitted patches)
            git checkout -- .
            source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
            cmake -S . -B '${CONTAINER_BUILD_DIR}' -G Ninja \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_C_COMPILER=icx \
                -DCMAKE_CXX_COMPILER=icpx \
                -DGGML_SYCL=ON \
                -DGGML_SYCL_TARGET=INTEL
            cmake --build '${CONTAINER_BUILD_DIR}' --target ${TARGETS} -j \$(nproc)
        "

    echo "Built baseline -> ${BUILD_DIR}/bin/"
    exit 0
fi

# ---------------------------------------------------------------------------
# Optimized: custom cmake build inside Intel container
# ---------------------------------------------------------------------------
BUILD_DIR="${LLAMA_DIR}/build-sycl-${IMAGE_TAG_SAFE}"
CONTAINER_BUILD_DIR="${CONTAINER_LLAMA_DIR}/build-sycl-${IMAGE_TAG_SAFE}"
MAIN_GPU="${LLAMA_SYCL_MAIN_GPU:-0}"
DEVICE_ARCH="${LLAMA_SYCL_DEVICE_ARCH:-auto}"

if [[ "${DEVICE_ARCH}" == "auto" ]]; then
    PCI_DEVICE_ID="$(sudo xpu-smi discovery -d "${MAIN_GPU}" 2>/dev/null | sed -n 's/.*PCI Device ID: 0x\([0-9A-Fa-f]\+\).*/\1/p' | head -n 1 | tr 'A-Z' 'a-z')"
    if [[ -n "${PCI_DEVICE_ID}" ]]; then
        case "${PCI_DEVICE_ID}" in
            e220|e221|e222|e223)
                DEVICE_ARCH="bmg-g31"
                ;;
            e202|e209|e20b|e20c|e20d|e210|e211|e212|e215|e216)
                DEVICE_ARCH="bmg-g21"
                ;;
        esac
    fi

    if [[ -z "${DEVICE_ARCH}" || "${DEVICE_ARCH}" == "auto" ]]; then
        DEVICE_ARCH="$(sudo docker run --rm --entrypoint bash "${IMAGE}" -lc "
            set -euo pipefail
            tmp=\$(mktemp -d)
            cd \"\$tmp\"
            ocloc query SUPPORTED_DEVICES >/dev/null
            f=\$(ls -1 *.yaml | head -n 1)
            PCI_DEVICE_ID='${PCI_DEVICE_ID}' FNAME=\"\$f\" python3 - <<'PY'
import os
from pathlib import Path

target = os.environ['PCI_DEVICE_ID'].lower()
lines = Path(os.environ['FNAME']).read_text().splitlines()

ip = None
for i, line in enumerate(lines):
    if f'device_id: 0x{target}' in line.lower():
        for j in range(i - 1, -1, -1):
            s = lines[j].strip()
            if s.startswith('ip:'):
                ip = s.split(':', 1)[1].strip().lower()
                break
        if ip:
            break

if not ip:
    raise SystemExit(0)

candidates = []
for line in lines:
    s = line.strip()
    if ':' not in s:
        continue
    name, value = [part.strip() for part in s.split(':', 1)]
    if value.lower() == ip:
        candidates.append(name)

preferred = [name for name in candidates if '-' in name and not name.startswith('xe')]
if preferred:
    print(preferred[0])
elif candidates:
    print(candidates[0])
PY
        " | tr -d '\r' | tail -n 1)"
    fi
fi

ARCH_CMAKE_ARGS=()
if [[ -n "${DEVICE_ARCH}" && "${DEVICE_ARCH}" != "auto" ]]; then
    ARCH_CMAKE_ARGS+=("-DGGML_SYCL_DEVICE_ARCH=${DEVICE_ARCH}")
fi

# Build the SYCL cmake flag list (optimized-only path; baseline exits above)
SYCL_CMAKE_ARGS=(
    -DGGML_SYCL=ON
    -DGGML_SYCL_TARGET=INTEL
    -DGGML_SYCL_F16=ON
    -DGGML_SYCL_DNN=ON
    -DGGML_SYCL_GRAPH=ON
    "${ARCH_CMAKE_ARGS[@]}"
)

echo "Building llama.cpp SYCL [config=optimized] -> ${BUILD_DIR}"

cleanup() {
    sudo chown -R "$(id -u):$(id -g)" "${BUILD_DIR}" 2>/dev/null || true
}

trap cleanup EXIT

sudo docker run --rm \
    --entrypoint bash \
    --device /dev/dri:/dev/dri \
    -v "${ROOT_DIR}:/workspace" \
    -w "${CONTAINER_LLAMA_DIR}" \
    "${IMAGE}" \
    -lc "
        set -euo pipefail
        git config --global --add safe.directory '${CONTAINER_LLAMA_DIR}'
        source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
        cmake -S . -B '${CONTAINER_BUILD_DIR}' -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER=icx \
            -DCMAKE_CXX_COMPILER=icpx \
            ${SYCL_CMAKE_ARGS[*]}
        cmake --build '${CONTAINER_BUILD_DIR}' --target ${TARGETS} -j \$(nproc)
    "
