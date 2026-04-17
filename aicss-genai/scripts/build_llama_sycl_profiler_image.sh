#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_COMMIT="${LLAMA_COMMIT:-45cac7ca7}"
BASE_IMAGE="${LLAMA_PROFILER_BASE_IMAGE:-llama-sycl-local:0.14.0-b8.1-${LLAMA_COMMIT}}"
UNITRACE_BUILD_DIR="${ROOT_DIR}/third_party/pti-gpu/tools/unitrace/build"
TARGET_IMAGE="${LLAMA_PROFILER_IMAGE:-llama-sycl-profiler:0.14.0-b8.1-${LLAMA_COMMIT}}"
CONTAINER_NAME="llama-sycl-profiler-pack-$$"

if [[ ! -x "${UNITRACE_BUILD_DIR}/unitrace" ]]; then
    echo "missing unitrace binary at ${UNITRACE_BUILD_DIR}/unitrace; run build_unitrace.sh first" >&2
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
    mkdir -p /opt/unitrace/bin /opt/unitrace/lib /opt/unitrace/scripts /opt/unitrace/metrics /usr/local/bin
"

sudo docker cp "${UNITRACE_BUILD_DIR}/unitrace" "${CONTAINER_NAME}:/opt/unitrace/bin/"
sudo docker cp "${UNITRACE_BUILD_DIR}/libunitrace_tool.so" "${CONTAINER_NAME}:/opt/unitrace/lib/"
sudo docker cp "${UNITRACE_BUILD_DIR}/scripts/uniview.py" "${CONTAINER_NAME}:/opt/unitrace/scripts/"
sudo docker cp "${UNITRACE_BUILD_DIR}/scripts/metrics/." "${CONTAINER_NAME}:/opt/unitrace/metrics/"

sudo docker exec "${CONTAINER_NAME}" bash -lc "
    set -euo pipefail
    python3 -m pip install --quiet --break-system-packages pandas matplotlib
"

sudo docker exec "${CONTAINER_NAME}" bash -lc '
    set -euo pipefail
    cat > /usr/local/bin/unitrace <<EOF
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/opt/unitrace/lib:${LD_LIBRARY_PATH:-}:/opt/llama.cpp-sycl/bin
exec /opt/unitrace/bin/unitrace "\$@"
EOF
    chmod +x /usr/local/bin/unitrace
'

sudo docker commit \
    --change 'ENV UNITRACE_ROOT=/opt/unitrace' \
    --change 'WORKDIR /opt/src/llama.cpp' \
    --change 'ENTRYPOINT ["/bin/bash"]' \
    "${CONTAINER_NAME}" \
    "${TARGET_IMAGE}" >/dev/null

echo "committed ${TARGET_IMAGE}"
