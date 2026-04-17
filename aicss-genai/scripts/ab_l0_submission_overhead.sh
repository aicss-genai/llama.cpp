#!/usr/bin/env bash
#
# A/B test: Level Zero submission overhead tuning for llama.cpp SYCL decode.
#
# Tests immediate command lists, device-scope events, and command list
# batching against the current accepted baseline.
#
# The hypothesis is that CPU-side kernel submission overhead (5.7x CPU/GPU
# ratio in the tail-90% trace) is the dominant decode bottleneck and that
# L0 runtime tuning can reduce it without any kernel code changes.
#
# Quick screen: pp512, tg128, r=1 (one rep) to identify winners.
# Full validation: winners are re-run with r=3 for statistical confidence.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${ROOT_DIR}/models/Qwen3.5-9B-Q4_K_M.gguf"
BENCH="${ROOT_DIR}/scripts/run_qwen9b_sycl_bench.sh"
TIMESTAMP="$(date +%F-%H%M%S)"
BASE_RESULTS="${ROOT_DIR}/results/${TIMESTAMP}-l0-ab"

# Fast screening shape: pp512, tg128, r=1
export LLAMA_BENCH_PROMPT_TOKENS=512
export LLAMA_BENCH_GEN_TOKENS=128
export LLAMA_BENCH_REPS="${LLAMA_BENCH_REPS:-1}"
export LLAMA_BENCH_WARMUP_PROMPT_TOKENS=512
export LLAMA_BENCH_WARMUP_GEN_TOKENS=32
export LLAMA_BENCH_WARMUP_REPS=1
export LLAMA_BENCH_N_BATCH=2048
export LLAMA_BENCH_N_UBATCH=2048

declare -A TESTS
TESTS[baseline]=""
TESTS[imm-cmdlist-ur]="UR_L0_USE_IMMEDIATE_COMMANDLISTS=1"
TESTS[imm-cmdlist-pi]="SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1"
TESTS[imm-cmdlist-devscope]="UR_L0_USE_IMMEDIATE_COMMANDLISTS=1 UR_L0_DEVICE_SCOPE_EVENTS=1"
TESTS[batch-16]="UR_L0_BATCH_SIZE=16"
TESTS[batch-32]="UR_L0_BATCH_SIZE=32"

extract_tps() {
    local json_file="$1"
    if [[ ! -f "${json_file}" ]]; then
        echo "N/A N/A"
        return
    fi
    python3 -c "
import json, sys
with open('${json_file}') as f:
    data = json.load(f)
results = data if isinstance(data, list) else [data]
pp = [r['avg_ts'] for r in results if r.get('n_prompt', 0) > 0]
tg = [r['avg_ts'] for r in results if r.get('n_gen', 0) > 0]
pp_avg = sum(pp)/len(pp) if pp else 0
tg_avg = sum(tg)/len(tg) if tg else 0
print(f'{pp_avg:.2f} {tg_avg:.2f}')
" 2>/dev/null || echo "N/A N/A"
}

echo "=================================================================="
echo " L0 Submission Overhead A/B Test"
echo " Model: $(basename "${MODEL}")"
echo " Shape: pp${LLAMA_BENCH_PROMPT_TOKENS}, tg${LLAMA_BENCH_GEN_TOKENS}, r=${LLAMA_BENCH_REPS}"
echo " Results: ${BASE_RESULTS}"
echo "=================================================================="
echo ""

mkdir -p "${BASE_RESULTS}"

# Run each test variant
for test_name in baseline imm-cmdlist-ur imm-cmdlist-pi imm-cmdlist-devscope batch-16 batch-32; do
    test_envs="${TESTS[$test_name]}"
    results_dir="${BASE_RESULTS}/${test_name}"

    echo "--- ${test_name} ---"
    if [[ -n "${test_envs}" ]]; then
        echo "  env: ${test_envs}"
    else
        echo "  env: (none - baseline)"
    fi

    # Clear L0 env vars
    unset UR_L0_USE_IMMEDIATE_COMMANDLISTS 2>/dev/null || true
    unset SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS 2>/dev/null || true
    unset UR_L0_DEVICE_SCOPE_EVENTS 2>/dev/null || true
    unset SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS 2>/dev/null || true
    unset UR_L0_BATCH_SIZE 2>/dev/null || true
    unset SYCL_PI_LEVEL_ZERO_BATCH_SIZE 2>/dev/null || true

    # Set test-specific env vars
    for kv in ${test_envs}; do
        export "${kv}"
    done

    export LLAMA_BENCH_RESULTS_DIR="${results_dir}"

    if "${BENCH}" "${MODEL}" > "${results_dir}.stdout" 2>&1; then
        echo "  status: OK"
    else
        echo "  status: FAILED (see ${results_dir}.stdout)"
        continue
    fi

    # Extract results
    json_file=$(find "${results_dir}" -name '*-single-stream.json' 2>/dev/null | head -1)
    read pp tg <<< "$(extract_tps "${json_file}")"
    echo "  pp512: ${pp} tok/s"
    echo "  tg128: ${tg} tok/s"
    echo ""

    # Save summary
    echo "${test_name}: pp512=${pp} tg128=${tg} envs=\"${test_envs}\"" >> "${BASE_RESULTS}/summary.txt"
done

echo "=================================================================="
echo " Summary"
echo "=================================================================="
if [[ -f "${BASE_RESULTS}/summary.txt" ]]; then
    cat "${BASE_RESULTS}/summary.txt"
else
    echo "(no results)"
fi
echo ""
echo "Full results: ${BASE_RESULTS}"
