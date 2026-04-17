#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${LLAMA_PROFILER_IMAGE:-llama-sycl-profiler:0.14.0-b8.1-45cac7ca7}"
MODELS_DIR="${LLAMA_MODELS_DIR:-${ROOT_DIR}/models}"
MODEL_PATH="${1:-${MODELS_DIR}/Qwen3.5-9B-Q4.gguf}"
TIMESTAMP="$(date +%Y-%m-%dT%H-%M-%S)-$$"
RESULTS_DIR="${LLAMA_PROFILE_RESULTS_DIR:-${ROOT_DIR}/results/profiles/${TIMESTAMP}}"
CONTAINER_ROOT="/workspace"
CONTAINER_RESULTS_DIR="/results"
LLAMA_BENCH_BIN="${LLAMA_PROFILE_LLAMA_BENCH:-llama-bench}"
THREADS="${LLAMA_PROFILE_THREADS:-8}"
PROMPT_TOKENS="${LLAMA_PROFILE_PROMPT_TOKENS:-128}"
GEN_TOKENS="${LLAMA_PROFILE_GEN_TOKENS:-8}"
N_BATCH="${LLAMA_PROFILE_N_BATCH:-512}"
N_UBATCH="${LLAMA_PROFILE_N_UBATCH:-512}"
REPS="${LLAMA_PROFILE_REPS:-1}"
WARMUP_PROMPT_TOKENS="${LLAMA_PROFILE_WARMUP_PROMPT_TOKENS:-0}"
WARMUP_GEN_TOKENS="${LLAMA_PROFILE_WARMUP_GEN_TOKENS:-0}"
WARMUP_REPS="${LLAMA_PROFILE_WARMUP_REPS:-0}"
FLASH_ATTN="${LLAMA_PROFILE_FLASH_ATTN:-1}"
SPLIT_MODE="${LLAMA_PROFILE_SPLIT_MODE:-none}"
MAIN_GPU="${LLAMA_PROFILE_MAIN_GPU:-0}"
ZE_AFFINITY_MASK="${LLAMA_PROFILE_ZE_AFFINITY_MASK:-0}"
METRIC_GROUP="${LLAMA_PROFILE_METRIC_GROUP:-ComputeBasic}"
COLLECT_METRICS="${LLAMA_PROFILE_COLLECT_METRICS:-1}"
TRACE_PREFIX="${LLAMA_PROFILE_TRACE_PREFIX:-decode-trace}"
TRACE_MODE="${LLAMA_PROFILE_TRACE_MODE:-summary}"
GGML_SYCL_DISABLE_GRAPH="${GGML_SYCL_DISABLE_GRAPH:-}"
GGML_SYCL_DISABLE_OPT="${GGML_SYCL_DISABLE_OPT:-}"
GGML_SYCL_PRIORITIZE_DMMV="${GGML_SYCL_PRIORITIZE_DMMV:-}"
GGML_SYCL_USE_ASYNC_MEM_OP="${GGML_SYCL_USE_ASYNC_MEM_OP:-}"
GGML_SYCL_DEBUG="${GGML_SYCL_DEBUG:-}"
OBS_FILE="/proc/sys/dev/xe/observation_paranoid"
ORIG_OBS_VALUE=""

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "missing model at ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -x "${ROOT_DIR}/third_party/pti-gpu/tools/unitrace/build/unitrace" ]]; then
    echo "missing built unitrace; run build_unitrace.sh first" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

cleanup() {
    if [[ -n "${ORIG_OBS_VALUE}" && -w "${OBS_FILE}" ]]; then
        echo "${ORIG_OBS_VALUE}" | sudo tee "${OBS_FILE}" >/dev/null
    fi
    sudo chown -R "$(id -u):$(id -g)" "${RESULTS_DIR}" 2>/dev/null || true
}

trap cleanup EXIT

DOCKER_ENV_ARGS=(
    -e ZES_ENABLE_SYSMAN=1
    -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
    -e ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK}"
    -e ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
)

for env_name in GGML_SYCL_DISABLE_GRAPH GGML_SYCL_DISABLE_OPT GGML_SYCL_PRIORITIZE_DMMV GGML_SYCL_USE_ASYNC_MEM_OP GGML_SYCL_DEBUG; do
    env_value="${!env_name}"
    if [[ -n "${env_value}" ]]; then
        DOCKER_ENV_ARGS+=(-e "${env_name}=${env_value}")
    fi
done

if [[ -r "${OBS_FILE}" ]]; then
    ORIG_OBS_VALUE="$(cat "${OBS_FILE}")"
    if [[ "${COLLECT_METRICS}" == "1" && "${ORIG_OBS_VALUE}" != "0" ]]; then
        echo 0 | sudo tee "${OBS_FILE}" >/dev/null
    fi
fi

MODEL_MOUNT_ARGS=()
if [[ "${MODEL_PATH}" == "${ROOT_DIR}"/* ]]; then
    CONTAINER_MODEL_PATH="${CONTAINER_ROOT}${MODEL_PATH#${ROOT_DIR}}"
else
    MODEL_DIR="$(cd "$(dirname "${MODEL_PATH}")" && pwd)"
    MODEL_FILE="$(basename "${MODEL_PATH}")"
    MODEL_MOUNT_ARGS=(-v "${MODEL_DIR}:/models:ro")
    CONTAINER_MODEL_PATH="/models/${MODEL_FILE}"
fi

CONTAINER_LLAMA_BENCH_BIN="${LLAMA_BENCH_BIN}"
if [[ "${LLAMA_BENCH_BIN}" == "${ROOT_DIR}"/* ]]; then
    CONTAINER_LLAMA_BENCH_BIN="${CONTAINER_ROOT}${LLAMA_BENCH_BIN#${ROOT_DIR}}"
fi
CONTAINER_LLAMA_BENCH_DIR="$(dirname "${CONTAINER_LLAMA_BENCH_BIN}")"
CONTAINER_UNITRACE_BIN="/opt/unitrace/bin/unitrace"
UNITRACE_DIR="${RESULTS_DIR}/unitrace"
mkdir -p "${UNITRACE_DIR}"

{
    echo "image=${IMAGE}"
    echo "model=${MODEL_PATH}"
    echo "llama_bench_bin=${LLAMA_BENCH_BIN}"
    echo "threads=${THREADS}"
    echo "prompt_tokens=${PROMPT_TOKENS}"
    echo "gen_tokens=${GEN_TOKENS}"
    echo "reps=${REPS}"
    echo "warmup_prompt_tokens=${WARMUP_PROMPT_TOKENS}"
    echo "warmup_gen_tokens=${WARMUP_GEN_TOKENS}"
    echo "warmup_reps=${WARMUP_REPS}"
    echo "n_batch=${N_BATCH}"
    echo "n_ubatch=${N_UBATCH}"
    echo "flash_attn=${FLASH_ATTN}"
    echo "split_mode=${SPLIT_MODE}"
    echo "main_gpu=${MAIN_GPU}"
    echo "ze_affinity_mask=${ZE_AFFINITY_MASK}"
    echo "metric_group=${METRIC_GROUP}"
    echo "collect_metrics=${COLLECT_METRICS}"
    echo "trace_mode=${TRACE_MODE}"
    echo "ggml_sycl_disable_graph=${GGML_SYCL_DISABLE_GRAPH:-unset}"
    echo "ggml_sycl_disable_opt=${GGML_SYCL_DISABLE_OPT:-unset}"
    echo "ggml_sycl_prioritize_dmmv=${GGML_SYCL_PRIORITIZE_DMMV:-unset}"
    echo "ggml_sycl_use_async_mem_op=${GGML_SYCL_USE_ASYNC_MEM_OP:-unset}"
    echo "ggml_sycl_debug=${GGML_SYCL_DEBUG:-unset}"
} > "${RESULTS_DIR}/run.txt"

sudo docker image inspect "${IMAGE}" > "${RESULTS_DIR}/image-inspect.json"
sudo xpu-smi topology -m > "${RESULTS_DIR}/xpu-topology.txt" || true
sudo xpu-smi discovery -d "${MAIN_GPU}" > "${RESULTS_DIR}/xpu-discovery-gpu${MAIN_GPU}.txt" || true
sudo xpu-smi stats -d "${MAIN_GPU}" -j > "${RESULTS_DIR}/xpu-stats-pre.json" || true

UNITRACE_TIMING_FLAGS="--host-timing --device-timing"
if [[ "${TRACE_MODE}" == "chrome" ]]; then
    UNITRACE_TIMING_FLAGS="${UNITRACE_TIMING_FLAGS} --chrome-sycl-logging --chrome-call-logging --chrome-kernel-logging --chrome-device-logging"
fi

sudo docker run --rm \
    --entrypoint bash \
    --privileged \
    --device /dev/dri:/dev/dri \
    "${DOCKER_ENV_ARGS[@]}" \
    -v "${ROOT_DIR}:/workspace" \
    "${MODEL_MOUNT_ARGS[@]}" \
    -v "${RESULTS_DIR}:${CONTAINER_RESULTS_DIR}" \
    -w /opt/src/llama.cpp \
    "${IMAGE}" \
    -lc "
        set -euo pipefail
        source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
        export LD_LIBRARY_PATH=/opt/unitrace/lib:'${CONTAINER_LLAMA_BENCH_DIR}':\${LD_LIBRARY_PATH:-}:/opt/llama.cpp-sycl/bin
        mkdir -p '${CONTAINER_RESULTS_DIR}/unitrace'
        '${CONTAINER_UNITRACE_BIN}' --version > '${CONTAINER_RESULTS_DIR}/unitrace-version.txt'
        '${CONTAINER_UNITRACE_BIN}' --device-list > '${CONTAINER_RESULTS_DIR}/unitrace-device-list.txt'
        '${CONTAINER_UNITRACE_BIN}' --metric-list > '${CONTAINER_RESULTS_DIR}/unitrace-metric-list.txt'
        '${CONTAINER_LLAMA_BENCH_BIN}' --list-devices > '${CONTAINER_RESULTS_DIR}/llama-bench-devices.txt'
        if [[ '${WARMUP_REPS}' != '0' ]]; then
            '${CONTAINER_LLAMA_BENCH_BIN}' \
                -m '${CONTAINER_MODEL_PATH}' \
                -o json \
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
                > '${CONTAINER_RESULTS_DIR}/llama-bench-warmup.json'
        fi
        '${CONTAINER_UNITRACE_BIN}' \
            ${UNITRACE_TIMING_FLAGS} \
            --output-dir-path '${CONTAINER_RESULTS_DIR}/unitrace' \
            -o '${CONTAINER_RESULTS_DIR}/unitrace/${TRACE_PREFIX}' \
            '${CONTAINER_LLAMA_BENCH_BIN}' \
                --no-warmup \
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
            > '${CONTAINER_RESULTS_DIR}/llama-bench-profile.json'

        if [[ '${COLLECT_METRICS}' == '1' ]]; then
            '${CONTAINER_UNITRACE_BIN}' \
                --metric-query \
                --group '${METRIC_GROUP}' \
                --chrome-kernel-logging \
                --chrome-device-logging \
                --output-dir-path '${CONTAINER_RESULTS_DIR}/unitrace' \
                -o '${CONTAINER_RESULTS_DIR}/unitrace/${TRACE_PREFIX}-metrics' \
                '${CONTAINER_LLAMA_BENCH_BIN}' \
                    --no-warmup \
                    -m '${CONTAINER_MODEL_PATH}' \
                    -o json \
                    -r 1 \
                    -t '${THREADS}' \
                    -p '${PROMPT_TOKENS}' \
                    -n '${GEN_TOKENS}' \
                    -b '${N_BATCH}' \
                    -ub '${N_UBATCH}' \
                    -ngl 99 \
                    -sm '${SPLIT_MODE}' \
                    -mg '${MAIN_GPU}' \
                    -fa '${FLASH_ATTN}' \
                > '${CONTAINER_RESULTS_DIR}/llama-bench-profile-metrics.json'
        fi
    "

sudo xpu-smi stats -d "${MAIN_GPU}" -j > "${RESULTS_DIR}/xpu-stats-post.json" || true

echo "profile artifacts written to ${RESULTS_DIR}"
