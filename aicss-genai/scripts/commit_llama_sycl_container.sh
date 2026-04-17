#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT_DIR}/third_party/llama.cpp"
BASE_IMAGE="${LLAMA_SYCL_IMAGE:-intel/llm-scaler-vllm:0.14.0-b8.1}"
IMAGE_TAG_SAFE="${BASE_IMAGE##*:}"
IMAGE_TAG_SAFE="${IMAGE_TAG_SAFE//[^A-Za-z0-9._-]/-}"
BENCH_CONFIG="${LLAMA_BENCH_CONFIG:-optimized}"
case "${BENCH_CONFIG}" in
    optimized|baseline) ;;
    *) echo "ERROR: unknown LLAMA_BENCH_CONFIG '${BENCH_CONFIG}'; expected 'optimized' or 'baseline'" >&2; exit 1 ;;
esac

SOURCE_REV="$(git -C "${LLAMA_DIR}" rev-parse --short=7 HEAD 2>/dev/null || echo unknown)"

# Package both optimized and baseline builds from the host build directory.
# Allow callers to override the build directory explicitly when needed.
if [[ "${BENCH_CONFIG}" == "baseline" ]]; then
    DEFAULT_BUILD_DIR="${LLAMA_DIR}/build-sycl-baseline-${IMAGE_TAG_SAFE}"
    DEFAULT_TARGET_IMAGE="llama-sycl-baseline:${SOURCE_REV}"
else
    DEFAULT_BUILD_DIR="${LLAMA_DIR}/build-sycl-${IMAGE_TAG_SAFE}"
    DEFAULT_TARGET_IMAGE="llama-sycl-local:optimized-${IMAGE_TAG_SAFE}-${SOURCE_REV}"
fi

BUILD_DIR="${LLAMA_SYCL_BUILD_DIR:-${DEFAULT_BUILD_DIR}}"
TARGET_IMAGE="${LLAMA_SYCL_COMMIT_IMAGE:-${DEFAULT_TARGET_IMAGE}}"
CONTAINER_NAME="llama-sycl-pack-${BENCH_CONFIG}-${IMAGE_TAG_SAFE//./-}-$$"

if [[ ! -x "${BUILD_DIR}/bin/llama-bench" ]]; then
    echo "missing built SYCL binaries at ${BUILD_DIR}/bin (config=${BENCH_CONFIG})" >&2
    echo "run: LLAMA_BENCH_CONFIG=${BENCH_CONFIG} ./scripts/build_llama_sycl_container.sh" >&2
    exit 1
fi

cleanup() {
    sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

sudo docker create --name "${CONTAINER_NAME}" --entrypoint bash "${BASE_IMAGE}" -lc "sleep infinity" >/dev/null
sudo docker start "${CONTAINER_NAME}" >/dev/null
sudo docker exec "${CONTAINER_NAME}" bash -lc "
    set -euo pipefail
    mkdir -p /opt/llama.cpp-sycl/bin /opt/src/llama.cpp /opt/llama-sycl-tools /usr/local/bin
"

sudo docker cp "${BUILD_DIR}/bin/." "${CONTAINER_NAME}:/opt/llama.cpp-sycl/bin/"
sudo docker cp "${LLAMA_DIR}/." "${CONTAINER_NAME}:/opt/src/llama.cpp/"
sudo docker cp "${ROOT_DIR}/scripts/build_llama_sycl_container.sh" "${CONTAINER_NAME}:/opt/llama-sycl-tools/"
sudo docker cp "${ROOT_DIR}/scripts/run_qwen9b_sycl_bench.sh" "${CONTAINER_NAME}:/opt/llama-sycl-tools/"

sudo docker exec "${CONTAINER_NAME}" bash -lc "
    set -euo pipefail
    cat > /opt/llama.cpp-sycl/METADATA.txt <<'EOF'
base_image=${BASE_IMAGE}
bench_config=${BENCH_CONFIG}
source_rev=${SOURCE_REV}
source_dir=/opt/src/llama.cpp
build_dir=/opt/llama.cpp-sycl/bin
built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
"

sudo docker exec "${CONTAINER_NAME}" bash -lc '
    set -euo pipefail
    for tool in llama-bench llama-batched-bench llama-cli llama-server llama-gguf-split; do
        cat > "/usr/local/bin/${tool}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/opt/llama.cpp-sycl/bin:\${LD_LIBRARY_PATH:-}
exec /opt/llama.cpp-sycl/bin/${tool} "\$@"
EOF
        chmod +x "/usr/local/bin/${tool}"
    done
'

sudo docker commit \
    --change 'ENV LLAMA_CPP_ROOT=/opt/src/llama.cpp' \
    --change 'ENV LLAMA_CPP_BUILD=/opt/llama.cpp-sycl/bin' \
    --change 'WORKDIR /opt/src/llama.cpp' \
    --change 'ENTRYPOINT ["/bin/bash"]' \
    "${CONTAINER_NAME}" \
    "${TARGET_IMAGE}" >/dev/null

echo "committed ${TARGET_IMAGE}"
echo "metadata: /opt/llama.cpp-sycl/METADATA.txt"
