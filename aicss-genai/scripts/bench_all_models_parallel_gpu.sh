#!/usr/bin/env bash
# bench_all_models_parallel_gpu.sh – run llama-bench for every model in bench_models_manifest.tsv
#
# Multi-GPU parallel execution: auto-detects GPUs via xpu-smi and dispatches
# models to free GPUs, one container per GPU. Override with LLAMA_BENCH_GPU_IDS.
#
# Scored sweep: pp512,1024,2048,4096,8192 × tg128,256,512,1024 × r=5
# Warmup:       pp512 × tg128 × r=1
# Results land in: results/<YYYY-MM-DD-HHMMSS>/<model>-<image-tag>/
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT_DIR}/third_party/llama.cpp"
CONTAINER_ROOT="/workspace"
CONTAINER_LLAMA_DIR="${CONTAINER_ROOT}/third_party/llama.cpp"
MANIFEST_RAW="${LLAMA_BENCH_MANIFEST:-${ROOT_DIR}/scripts/bench_models_manifest.tsv}"
case "${MANIFEST_RAW}" in
    /*) MANIFEST="${MANIFEST_RAW}" ;;
    *)  MANIFEST="${ROOT_DIR}/${MANIFEST_RAW}" ;;
esac
MODEL_FILTER="${LLAMA_BENCH_MODEL_FILTER:-}"
MODELS_DIR="${LLAMA_MODELS_DIR:-${ROOT_DIR}/models}"
CONTAINER_MODELS_DIR="/models"
CONTAINER_RESULTS_ROOT="/results"

IMAGE="${LLAMA_SYCL_IMAGE:-intel/llm-scaler-vllm:0.14.0-b8.1}"
IMAGE_TAG_SAFE="${IMAGE##*:}"
IMAGE_TAG_SAFE="${IMAGE_TAG_SAFE//[^A-Za-z0-9._-]/-}"
BENCH_CONFIG="${LLAMA_BENCH_CONFIG:-optimized}"
case "${BENCH_CONFIG}" in
    optimized) BUILD_SUFFIX="" ;;
    baseline)  BUILD_SUFFIX="-baseline" ;;
    *) echo "ERROR: unknown LLAMA_BENCH_CONFIG '${BENCH_CONFIG}'; expected 'optimized' or 'baseline'" >&2; exit 1 ;;
esac
BUILD_DIR="${LLAMA_DIR}/build-sycl${BUILD_SUFFIX}-${IMAGE_TAG_SAFE}"
CONTAINER_BUILD_DIR="${CONTAINER_LLAMA_DIR}/build-sycl${BUILD_SUFFIX}-${IMAGE_TAG_SAFE}"

RESULTS_BASE="${LLAMA_BENCH_RESULTS_DIR:-${ROOT_DIR}/results/$(date +%F-%H%M%S)}"
THREADS="${LLAMA_BENCH_THREADS:-8}"
WARMUP_PP="${LLAMA_BENCH_WARMUP_PP:-512}"
WARMUP_TG="${LLAMA_BENCH_WARMUP_TG:-128}"
WARMUP_REPS="${LLAMA_BENCH_WARMUP_REPS:-1}"
PROMPT_TOKENS="${LLAMA_BENCH_PROMPT_TOKENS:-512,1024,2048,4096,8192}"
GEN_TOKENS="${LLAMA_BENCH_GEN_TOKENS:-128,256,512,1024}"
REPS="${LLAMA_BENCH_REPS:-5}"
N_BATCH="${LLAMA_BENCH_N_BATCH:-2048}"
N_UBATCH="${LLAMA_BENCH_N_UBATCH:-2048}"
FLASH_ATTN="${LLAMA_BENCH_FLASH_ATTN:-1}"
SPLIT_MODE="${LLAMA_BENCH_SPLIT_MODE:-none}"
MAIN_GPU="${LLAMA_BENCH_MAIN_GPU:-0}"
RUN_BATCHED="${LLAMA_BENCH_RUN_BATCHED:-0}"
BATCHED_NPL="${LLAMA_BENCH_BATCHED_NPL:-1}"

# Override: comma-separated GPU IDs to use (default: auto-detect via xpu-smi)
# Example: LLAMA_BENCH_GPU_IDS=0,1,3
GPU_IDS_OVERRIDE="${LLAMA_BENCH_GPU_IDS:-}"

# Optional SYCL / Level Zero tuning vars forwarded into the container when set
GGML_SYCL_DISABLE_GRAPH="${GGML_SYCL_DISABLE_GRAPH:-}"
GGML_SYCL_DISABLE_OPT="${GGML_SYCL_DISABLE_OPT:-}"
GGML_SYCL_PRIORITIZE_DMMV="${GGML_SYCL_PRIORITIZE_DMMV:-}"
GGML_SYCL_USE_ASYNC_MEM_OP="${GGML_SYCL_USE_ASYNC_MEM_OP:-}"
GGML_SYCL_DEBUG="${GGML_SYCL_DEBUG:-}"
UR_L0_USE_IMMEDIATE_COMMANDLISTS="${UR_L0_USE_IMMEDIATE_COMMANDLISTS:-}"
SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS="${SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS:-}"
UR_L0_DEVICE_SCOPE_EVENTS="${UR_L0_DEVICE_SCOPE_EVENTS:-}"
SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS="${SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS:-}"
UR_L0_BATCH_SIZE="${UR_L0_BATCH_SIZE:-}"
SYCL_PI_LEVEL_ZERO_BATCH_SIZE="${SYCL_PI_LEVEL_ZERO_BATCH_SIZE:-}"

DOCKER_CMD=()

resolve_docker_cmd() {
    if command -v docker &>/dev/null && docker info &>/dev/null; then
        DOCKER_CMD=(docker)
        return
    fi
    if command -v sudo &>/dev/null && sudo docker info &>/dev/null; then
        DOCKER_CMD=(sudo docker)
        return
    fi

    echo "ERROR: Docker daemon not reachable with docker or sudo docker" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
# When the host-side build directory exists (image_strategy=build), mount the
# entire workspace into /workspace so the host-built binary is used.  When it
# does not exist (image_strategy=reuse), the Docker image already contains the
# built binary under its own /workspace — so we skip the host workspace mount
# to avoid shadowing the image's artifacts.
HOST_BUILD_AVAILABLE=false
if [[ -x "${BUILD_DIR}/bin/llama-bench" ]]; then
    HOST_BUILD_AVAILABLE=true
fi

if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: manifest not found: ${MANIFEST}" >&2
    exit 1
fi

resolve_docker_cmd
mkdir -p "${RESULTS_BASE}"
mkdir -p "${MODELS_DIR}"

# ---------------------------------------------------------------------------
# GPU detection & parallel pool
# ---------------------------------------------------------------------------
detect_gpus() {
    if [[ -n "${GPU_IDS_OVERRIDE}" ]]; then
        IFS=',' read -ra GPU_IDS <<< "${GPU_IDS_OVERRIDE}"
    else
        local count=0
        if command -v xpu-smi &>/dev/null; then
            count=$(xpu-smi discovery 2>/dev/null \
                | grep -cP '^\|\s*\d+\s*\|' || true)
        fi
        if (( count == 0 )); then
            count=$(find /dev/dri -name 'renderD*' 2>/dev/null | wc -l)
        fi
        if (( count == 0 )); then
            count=1
        fi
        GPU_IDS=()
        for ((i=0; i<count; i++)); do
            GPU_IDS+=("$i")
        done
    fi
    NUM_GPUS=${#GPU_IDS[@]}
}

detect_gpus
echo "Detected ${NUM_GPUS} GPU(s): ${GPU_IDS[*]}"

# FIFO-based GPU pool – each token is a GPU ID
GPU_POOL=$(mktemp -u /tmp/gpu_pool.XXXXXX)
mkfifo "${GPU_POOL}"
exec 3<>"${GPU_POOL}"   # open rw so writers never block on empty pipe

RESULT_DIR=$(mktemp -d /tmp/bench_results.XXXXXX)

cleanup() {
    exec 3>&- 2>/dev/null || true
    rm -f "${GPU_POOL}" 2>/dev/null || true
    rm -rf "${RESULT_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

# Seed tokens – one byte per GPU (single-digit IDs; 1-byte FIFO ops are atomic)
for _gpu_id in "${GPU_IDS[@]}"; do
    printf '%s' "${_gpu_id}" >&3
done

# ---------------------------------------------------------------------------
# Build the docker --env arg list (constant across all models)
# NOTE: ZE_AFFINITY_MASK is set per-container in bench_one()
# ---------------------------------------------------------------------------
DOCKER_ENV_ARGS=(
    -e ZES_ENABLE_SYSMAN=1
    -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
    -e ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
)
for _env in \
    GGML_SYCL_DISABLE_GRAPH GGML_SYCL_DISABLE_OPT \
    GGML_SYCL_PRIORITIZE_DMMV GGML_SYCL_USE_ASYNC_MEM_OP GGML_SYCL_DEBUG \
    UR_L0_USE_IMMEDIATE_COMMANDLISTS SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS \
    UR_L0_DEVICE_SCOPE_EVENTS SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS \
    UR_L0_BATCH_SIZE SYCL_PI_LEVEL_ZERO_BATCH_SIZE; do
    _val="${!_env}"
    if [[ -n "${_val}" ]]; then
        DOCKER_ENV_ARGS+=(-e "${_env}=${_val}")
    fi
done
# When GGML_SYCL_DISABLE_GRAPH is equal to 0 add ONEAPI_DEVICE_SELECTOR to environment variables
# Graphs are not supported when using multiple devices
if [[ "${GGML_SYCL_DISABLE_GRAPH}" == "0" ]]; then
    DOCKER_ENV_ARGS+=(-e ONEAPI_DEVICE_SELECTOR="level_zero:0")
fi

# ---------------------------------------------------------------------------
# Per-model benchmark function
# ---------------------------------------------------------------------------
bench_one() {
    local name="$1"
    local gpu_id="$2"
    local model_path="${MODELS_DIR}/${name}.gguf"

    if [[ ! -f "${model_path}" ]]; then
        echo "[SKIP] ${name}: model file not found at ${model_path}" >&2
        return 0
    fi

    local results_dir="${RESULTS_BASE}/${name}-${IMAGE_TAG_SAFE}"
    mkdir -p "${results_dir}"

    local container_model_path="${CONTAINER_MODELS_DIR}/${name}.gguf"
    local container_results_dir="${CONTAINER_RESULTS_ROOT}/${name}-${IMAGE_TAG_SAFE}"

    local device_file="${results_dir}/${name}-${IMAGE_TAG_SAFE}-devices.txt"
    local warmup_file="${results_dir}/${name}-${IMAGE_TAG_SAFE}-warmup.md"
    local scored_json="${results_dir}/${name}-${IMAGE_TAG_SAFE}-single-stream.json"
    local batched_jsonl="${results_dir}/${name}-${IMAGE_TAG_SAFE}-batched.jsonl"
    local run_meta="${results_dir}/${name}-${IMAGE_TAG_SAFE}-run.txt"

    local container_device_file="${container_results_dir}/${name}-${IMAGE_TAG_SAFE}-devices.txt"
    local container_warmup_file="${container_results_dir}/${name}-${IMAGE_TAG_SAFE}-warmup.md"
    local container_scored_json="${container_results_dir}/${name}-${IMAGE_TAG_SAFE}-single-stream.json"
    local container_batched_jsonl="${container_results_dir}/${name}-${IMAGE_TAG_SAFE}-batched.jsonl"
    local container_run_meta="${container_results_dir}/${name}-${IMAGE_TAG_SAFE}-run.txt"

    local log_file="${results_dir}/${name}-${IMAGE_TAG_SAFE}-docker.log"
    echo "[RUN ] ${name} on GPU ${gpu_id}"

    # Write run metadata (host-side, before docker so it is always present)
    cat > "${run_meta}" <<EOF
image=${IMAGE}
bench_config=${BENCH_CONFIG}
build_dir=${BUILD_DIR}
model=${model_path}
threads=${THREADS}
warmup_pp=${WARMUP_PP}
warmup_tg=${WARMUP_TG}
warmup_reps=${WARMUP_REPS}
prompt_tokens=${PROMPT_TOKENS}
gen_tokens=${GEN_TOKENS}
reps=${REPS}
n_batch=${N_BATCH}
n_ubatch=${N_UBATCH}
flash_attn=${FLASH_ATTN}
split_mode=${SPLIT_MODE}
main_gpu=${MAIN_GPU}
run_batched=${RUN_BATCHED}
batched_npl=${BATCHED_NPL}
gpu_id=${gpu_id}
ze_affinity_mask=${gpu_id}
ggml_sycl_disable_graph=${GGML_SYCL_DISABLE_GRAPH:-unset}
ggml_sycl_disable_opt=${GGML_SYCL_DISABLE_OPT:-unset}
ggml_sycl_prioritize_dmmv=${GGML_SYCL_PRIORITIZE_DMMV:-unset}
ggml_sycl_use_async_mem_op=${GGML_SYCL_USE_ASYNC_MEM_OP:-unset}
ggml_sycl_debug=${GGML_SYCL_DEBUG:-unset}
ur_l0_use_immediate_commandlists=${UR_L0_USE_IMMEDIATE_COMMANDLISTS:-unset}
sycl_pi_l0_use_immediate_commandlists=${SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS:-unset}
ur_l0_device_scope_events=${UR_L0_DEVICE_SCOPE_EVENTS:-unset}
sycl_pi_l0_device_scope_events=${SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS:-unset}
ur_l0_batch_size=${UR_L0_BATCH_SIZE:-unset}
sycl_pi_l0_batch_size=${SYCL_PI_LEVEL_ZERO_BATCH_SIZE:-unset}
EOF

    local docker_vol_args=()
    local container_workdir="${CONTAINER_LLAMA_DIR}"
    if [[ "${HOST_BUILD_AVAILABLE}" == "true" ]]; then
        docker_vol_args+=(-v "${ROOT_DIR}:/workspace")
    else
        # Without the workspace mount, use a directory that exists in the image
        container_workdir="/opt/src/llama.cpp"
    fi

    "${DOCKER_CMD[@]}" run --rm \
        --entrypoint bash \
        --privileged \
        --device /dev/dri:/dev/dri \
        "${DOCKER_ENV_ARGS[@]}" \
        -e "ZE_AFFINITY_MASK=${gpu_id}" \
        "${docker_vol_args[@]}" \
        -v "${MODELS_DIR}:${CONTAINER_MODELS_DIR}:ro" \
        -v "${RESULTS_BASE}:${CONTAINER_RESULTS_ROOT}" \
        -w "${container_workdir}" \
        "${IMAGE}" \
        -c "
set -euo pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true

# Discover the build directory — use the expected path if it exists,
# otherwise search image-standard locations, then fall back to PATH.
# Set LD_LIBRARY_PATH for the discovered binary's sibling libs.
BENCH_BIN='${CONTAINER_BUILD_DIR}/bin/llama-bench'
if [[ ! -x \"\$BENCH_BIN\" ]]; then
    for d in '${CONTAINER_LLAMA_DIR}'/build-sycl-*/bin/llama-bench \
             /opt/llama.cpp-sycl/bin/llama-bench \
             /opt/src/llama.cpp/build-sycl-*/bin/llama-bench \
             /usr/local/bin/llama-bench; do
        if [[ -x \"\$d\" ]]; then
            BENCH_BIN=\"\$d\"
            break
        fi
    done
fi
if [[ ! -x \"\$BENCH_BIN\" ]]; then
    echo 'ERROR: llama-bench binary not found in image' >&2
    exit 1
fi
BENCH_BIN_DIR=\"\$(dirname \"\$BENCH_BIN\")\"
export LD_LIBRARY_PATH=\"\${BENCH_BIN_DIR}:\${BENCH_BIN_DIR}/../lib:\${LD_LIBRARY_PATH:-}\"
mkdir -p '${container_results_dir}'

\"\$BENCH_BIN\" --list-devices \
    > '${container_device_file}'

\"\$BENCH_BIN\" \
    -m '${container_model_path}' \
    -o md \
    -r '${WARMUP_REPS}' \
    -t '${THREADS}' \
    -p '${WARMUP_PP}' \
    -n '${WARMUP_TG}' \
    -b '${N_BATCH}' \
    -ub '${N_UBATCH}' \
    -ngl 99 \
    -sm '${SPLIT_MODE}' \
    -mg '${MAIN_GPU}' \
    -fa '${FLASH_ATTN}' \
    > '${container_warmup_file}'

\"\$BENCH_BIN\" \
    -m '${container_model_path}' \
    -o json \
    -r '${REPS}' \
    -t '${THREADS}' \
    -p '${PROMPT_TOKENS}' \
    -n '${GEN_TOKENS}' \
    -b '${N_BATCH}' \
    -ub '${N_UBATCH}' \
    -ngl 99 \
    -sm '${SPLIT_MODE}' \
    -mg '${MAIN_GPU}' \
    -fa '${FLASH_ATTN}' \
    > '${container_scored_json}'

if [[ '${RUN_BATCHED}' == '1' ]]; then
    BATCHED_BIN=\"\${BENCH_BIN_DIR}/llama-batched-bench\"
    if [[ ! -x \"\$BATCHED_BIN\" ]]; then
        echo 'WARNING: llama-batched-bench not found at '\"\$BATCHED_BIN\"', skipping batched bench' >&2
    else
    \"\$BATCHED_BIN\" \
        -m '${container_model_path}' \
        -ngl 99 \
        -sm '${SPLIT_MODE}' \
        -mg '${MAIN_GPU}' \
        -c 9216 \
        -b '${N_BATCH}' \
        -ub '${N_UBATCH}' \
        -npp '${PROMPT_TOKENS}' \
        -ntg '${GEN_TOKENS}' \
        -npl '${BATCHED_NPL}' \
        -fa '${FLASH_ATTN}' \
        --output-format jsonl \
        > '${container_batched_jsonl}'
    fi
fi
" > "${log_file}" 2>&1

    chown -R "$(id -u):$(id -g)" "${results_dir}" 2>/dev/null || true
    echo "[DONE] ${name} on GPU ${gpu_id} -> ${results_dir}"
}

# ---------------------------------------------------------------------------
# Parallel dispatch wrapper
# ---------------------------------------------------------------------------
dispatch_model() {
    local name="$1"
    local gpu_id
    read -r -n1 gpu_id <&3  # acquire GPU slot (blocks until available)

    if bench_one "${name}" "${gpu_id}"; then
        if [[ -f "${RESULTS_BASE}/${name}-${IMAGE_TAG_SAFE}/${name}-${IMAGE_TAG_SAFE}-single-stream.json" ]]; then
            echo "pass" > "${RESULT_DIR}/${name}.result"
        else
            echo "skip" > "${RESULT_DIR}/${name}.result"
        fi
    else
        echo "fail" > "${RESULT_DIR}/${name}.result"
        echo "[FAIL] ${name} on GPU ${gpu_id}" >&2
    fi

    printf '%s' "${gpu_id}" >&3  # release GPU slot
}

# ---------------------------------------------------------------------------
# Main loop – dispatch models in parallel (one per GPU)
# ---------------------------------------------------------------------------
PIDS=()

while IFS=$'\t' read -r name _size _url; do
    [[ -z "${name}" ]] && continue
    # If a model filter is set, skip models that don't match (supports regex)
    if [[ -n "${MODEL_FILTER}" ]] && ! echo "${name}" | grep -qE "${MODEL_FILTER}"; then
        continue
    fi
    dispatch_model "${name}" &
    PIDS+=($!)
done < <(tail -n +2 "${MANIFEST}")

# Wait for every background job to finish
for pid in "${PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# Tally results
# ---------------------------------------------------------------------------
pass=0; skip=0; fail=0

for result_file in "${RESULT_DIR}"/*.result; do
    [[ -f "${result_file}" ]] || continue
    status=$(<"${result_file}")
    case "${status}" in
        pass) (( pass++ )) || true ;;
        skip) (( skip++ )) || true ;;
        fail) (( fail++ )) || true ;;
    esac
done

echo ""
echo "===== bench_all_models complete ====="
echo "  GPUs used: ${NUM_GPUS} (${GPU_IDS[*]})"
echo "  passed : ${pass}"
echo "  skipped: ${skip}"
echo "  failed : ${fail}"
echo "  results: ${RESULTS_BASE}"

# ---------------------------------------------------------------------------
# Auto-generate comparison CSV via result_parser.py
# ---------------------------------------------------------------------------
PARSER="${ROOT_DIR}/scripts/result_parser.py"
CSV_OUT="${RESULTS_BASE}/$(basename "${RESULTS_BASE}").csv"

if [[ -x "$(command -v python3)" ]] && [[ -f "${PARSER}" ]]; then
    echo ""
    echo "Generating CSV report: ${CSV_OUT}"
    python3 "${PARSER}" "${RESULTS_BASE}" --format csv --output "${CSV_OUT}" \
        && echo "  CSV written: ${CSV_OUT}" \
        || echo "  WARNING: result_parser.py failed – CSV not generated" >&2
else
    echo "  SKIP: python3 or ${PARSER} not found – CSV not generated" >&2
fi

