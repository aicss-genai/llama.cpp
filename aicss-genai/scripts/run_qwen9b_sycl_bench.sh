#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${ROOT_DIR}/third_party/llama.cpp"
CONTAINER_ROOT="/workspace"
CONTAINER_LLAMA_DIR="${CONTAINER_ROOT}/third_party/llama.cpp"
IMAGE="${LLAMA_SYCL_IMAGE:-intel/llm-scaler-vllm:0.14.0-b8.1}"
IMAGE_TAG_SAFE="${IMAGE##*:}"
IMAGE_TAG_SAFE="${IMAGE_TAG_SAFE//[^A-Za-z0-9._-]/-}"
BENCH_CONFIG="${LLAMA_BENCH_CONFIG:-optimized}"
case "${BENCH_CONFIG}" in
    optimized|baseline) ;;
    *) echo "ERROR: unknown LLAMA_BENCH_CONFIG '${BENCH_CONFIG}'; expected 'optimized' or 'baseline'" >&2; exit 1 ;;
esac

# Config-dependent image, binary path, and init command
if [[ "${BENCH_CONFIG}" == "baseline" ]]; then
    SOURCE_REV="$(git -C "${LLAMA_DIR}" rev-parse --short=7 HEAD 2>/dev/null || echo unknown)"
    BUILD_DIR="${LLAMA_BENCH_BASELINE_BUILD_DIR:-${LLAMA_DIR}/build-sycl-baseline-${SOURCE_REV}}"
    CONTAINER_BUILD_DIR="${LLAMA_BENCH_BASELINE_CONTAINER_BUILD_DIR:-${CONTAINER_LLAMA_DIR}/$(basename "${BUILD_DIR}")}"
    BENCH_IMAGE="${IMAGE}"
    BENCH_BIN="${CONTAINER_BUILD_DIR}/bin"
    BENCH_INIT='source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true'
else
    BUILD_DIR="${LLAMA_DIR}/build-sycl-${IMAGE_TAG_SAFE}"
    CONTAINER_BUILD_DIR="${CONTAINER_LLAMA_DIR}/build-sycl-${IMAGE_TAG_SAFE}"
    BENCH_IMAGE="${IMAGE}"
    BENCH_BIN="${CONTAINER_BUILD_DIR}/bin"
    BENCH_INIT='source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true'
fi
MODELS_DIR="${LLAMA_MODELS_DIR:-${ROOT_DIR}/models}"
MODEL_PATH="${1:-${MODELS_DIR}/Qwen3.5-9B-Q4.gguf}"
RESULTS_DIR="${LLAMA_BENCH_RESULTS_DIR:-${ROOT_DIR}/results/$(date +%F-%H%M%S)}"
THREADS="${LLAMA_BENCH_THREADS:-8}"
WARMUP_PROMPT_TOKENS="${LLAMA_BENCH_WARMUP_PROMPT_TOKENS:-512}"
WARMUP_GEN_TOKENS="${LLAMA_BENCH_WARMUP_GEN_TOKENS:-32}"
WARMUP_REPS="${LLAMA_BENCH_WARMUP_REPS:-1}"
PROMPT_TOKENS="${LLAMA_BENCH_PROMPT_TOKENS:-512,1024,2048,4096,8192}"
GEN_TOKENS="${LLAMA_BENCH_GEN_TOKENS:-128,256,512,1024}"
REPS="${LLAMA_BENCH_REPS:-5}"
N_BATCH="${LLAMA_BENCH_N_BATCH:-2048}"
N_UBATCH="${LLAMA_BENCH_N_UBATCH:-2048}"
FLASH_ATTN="${LLAMA_BENCH_FLASH_ATTN:-1}"
SPLIT_MODE="${LLAMA_BENCH_SPLIT_MODE:-none}"
MAIN_GPU="${LLAMA_BENCH_MAIN_GPU:-0}"
ZE_AFFINITY_MASK="${LLAMA_BENCH_ZE_AFFINITY_MASK:-0}"
RUN_BATCHED="${LLAMA_BENCH_RUN_BATCHED:-0}"
BATCHED_NPL="${LLAMA_BENCH_BATCHED_NPL:-1}"
GGML_SYCL_DISABLE_GRAPH="${GGML_SYCL_DISABLE_GRAPH:-}"
GGML_SYCL_DISABLE_OPT="${GGML_SYCL_DISABLE_OPT:-}"
GGML_SYCL_PRIORITIZE_DMMV="${GGML_SYCL_PRIORITIZE_DMMV:-}"
GGML_SYCL_USE_ASYNC_MEM_OP="${GGML_SYCL_USE_ASYNC_MEM_OP:-}"
GGML_SYCL_DEBUG="${GGML_SYCL_DEBUG:-}"

# Level Zero / SYCL runtime tuning env vars (forwarded into the container if set)
UR_L0_USE_IMMEDIATE_COMMANDLISTS="${UR_L0_USE_IMMEDIATE_COMMANDLISTS:-}"
SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS="${SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS:-}"
UR_L0_DEVICE_SCOPE_EVENTS="${UR_L0_DEVICE_SCOPE_EVENTS:-}"
SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS="${SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS:-}"
UR_L0_BATCH_SIZE="${UR_L0_BATCH_SIZE:-}"
SYCL_PI_LEVEL_ZERO_BATCH_SIZE="${SYCL_PI_LEVEL_ZERO_BATCH_SIZE:-}"

MODEL_TAG="$(basename "${MODEL_PATH}")"
MODEL_TAG="${MODEL_TAG%.gguf}"

if [[ "${BENCH_CONFIG}" == "baseline" ]]; then
    if ! sudo docker image inspect "${BENCH_IMAGE}" &>/dev/null; then
        echo "missing baseline image ${BENCH_IMAGE}" >&2
        echo "run: LLAMA_BENCH_CONFIG=baseline ./scripts/build_llama_sycl_container.sh" >&2
        exit 1
    fi
else
    if [[ ! -x "${BUILD_DIR}/bin/llama-bench" ]]; then
        echo "missing benchmark binary at ${BUILD_DIR}/bin/llama-bench (config=${BENCH_CONFIG})" >&2
        echo "run: LLAMA_BENCH_CONFIG=${BENCH_CONFIG} ./scripts/build_llama_sycl_container.sh" >&2
        exit 1
    fi
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "missing model at ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

cleanup() {
    sudo chown -R "$(id -u):$(id -g)" "${RESULTS_DIR}" 2>/dev/null || true
}

trap cleanup EXIT

DOCKER_ENV_ARGS=(
    -e ZES_ENABLE_SYSMAN=1
    -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
    -e ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK}"
    -e ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
)

for env_name in GGML_SYCL_DISABLE_GRAPH GGML_SYCL_DISABLE_OPT GGML_SYCL_PRIORITIZE_DMMV GGML_SYCL_USE_ASYNC_MEM_OP GGML_SYCL_DEBUG \
    UR_L0_USE_IMMEDIATE_COMMANDLISTS SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS \
    UR_L0_DEVICE_SCOPE_EVENTS SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS \
    UR_L0_BATCH_SIZE SYCL_PI_LEVEL_ZERO_BATCH_SIZE; do
    env_value="${!env_name}"
    if [[ -n "${env_value}" ]]; then
        DOCKER_ENV_ARGS+=(-e "${env_name}=${env_value}")
    fi
done

if [[ "${MODEL_PATH}" == "${ROOT_DIR}"/* ]]; then
    CONTAINER_MODEL_PATH="${CONTAINER_ROOT}${MODEL_PATH#${ROOT_DIR}}"
else
    echo "model path must be inside ${ROOT_DIR} for container access: ${MODEL_PATH}" >&2
    exit 1
fi

CONTAINER_RESULTS_DIR="${CONTAINER_ROOT}${RESULTS_DIR#${ROOT_DIR}}"
DEVICE_LIST_FILE="${RESULTS_DIR}/${MODEL_TAG}-${IMAGE_TAG_SAFE}-devices.txt"
WARMUP_FILE="${RESULTS_DIR}/${MODEL_TAG}-${IMAGE_TAG_SAFE}-warmup.md"
SCORED_JSON="${RESULTS_DIR}/${MODEL_TAG}-${IMAGE_TAG_SAFE}-single-stream.json"
BATCHED_JSONL="${RESULTS_DIR}/${MODEL_TAG}-${IMAGE_TAG_SAFE}-batched.jsonl"
RUN_META_FILE="${RESULTS_DIR}/${MODEL_TAG}-${IMAGE_TAG_SAFE}-run.txt"
CONTAINER_DEVICE_LIST_FILE="${CONTAINER_ROOT}${DEVICE_LIST_FILE#${ROOT_DIR}}"
CONTAINER_WARMUP_FILE="${CONTAINER_ROOT}${WARMUP_FILE#${ROOT_DIR}}"
CONTAINER_SCORED_JSON="${CONTAINER_ROOT}${SCORED_JSON#${ROOT_DIR}}"
CONTAINER_BATCHED_JSONL="${CONTAINER_ROOT}${BATCHED_JSONL#${ROOT_DIR}}"
CONTAINER_RUN_META_FILE="${CONTAINER_ROOT}${RUN_META_FILE#${ROOT_DIR}}"

sudo docker run --rm \
    --entrypoint bash \
    --privileged \
    --device /dev/dri:/dev/dri \
    "${DOCKER_ENV_ARGS[@]}" \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    "${BENCH_IMAGE}" \
    -lc "
        set -euo pipefail
        ${BENCH_INIT}
        mkdir -p '${CONTAINER_RESULTS_DIR}'
        cat > '${CONTAINER_RUN_META_FILE}' <<'EOF'
image=${BENCH_IMAGE}
bench_config=${BENCH_CONFIG}
bench_bin=${BENCH_BIN}
model=${MODEL_PATH}
threads=${THREADS}
warmup_prompt_tokens=${WARMUP_PROMPT_TOKENS}
warmup_gen_tokens=${WARMUP_GEN_TOKENS}
warmup_reps=${WARMUP_REPS}
prompt_tokens=${PROMPT_TOKENS}
gen_tokens=${GEN_TOKENS}
reps=${REPS}
n_batch=${N_BATCH}
n_ubatch=${N_UBATCH}
flash_attn=${FLASH_ATTN}
split_mode=${SPLIT_MODE}
main_gpu=${MAIN_GPU}
ze_affinity_mask=${ZE_AFFINITY_MASK}
run_batched=${RUN_BATCHED}
batched_npl=${BATCHED_NPL}
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
        '${BENCH_BIN}/llama-bench' --list-devices > '${CONTAINER_DEVICE_LIST_FILE}'
        '${BENCH_BIN}/llama-bench' \
            -m '${CONTAINER_MODEL_PATH}' \
            -o md \
            -r '${WARMUP_REPS}' \
            -t '${THREADS}' \
            -p '${WARMUP_PROMPT_TOKENS}' \
            -n '${WARMUP_GEN_TOKENS}' \
            -b '${N_BATCH}' \
            -ub '${N_UBATCH}' \
            -ngl 99 \
            -sm '${SPLIT_MODE}' \
            -mg '${MAIN_GPU}' \
            -fa '${FLASH_ATTN}' \
            > '${CONTAINER_WARMUP_FILE}'
        '${BENCH_BIN}/llama-bench' \
            -m '${CONTAINER_MODEL_PATH}' \
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
            > '${CONTAINER_SCORED_JSON}'
        if [[ '${RUN_BATCHED}' == '1' ]]; then
            '${BENCH_BIN}/llama-batched-bench' \
                -m '${CONTAINER_MODEL_PATH}' \
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
                > '${CONTAINER_BATCHED_JSONL}'
        fi
    "
