#!/usr/bin/env bash

set -euo pipefail

LLAMA_KERNELS="dequantize_mul_mat_vec_q4_K_sycl,dequantize_mul_mat_vec_q6_K_sycl,launch_kernel,rms_norm_mul_f32_sycl,gemm_kernel"
IPEX_KERNELS="mlp_forward_q4_k_kernel,linear_forward_kernel,ggml_sycl_op_dequantize_mul_mat_vec_q6_k,sdp_fp16_kernel"
KERNELS="${LLAMA_KERNELS},${IPEX_KERNELS}"
GROUP="ComputeBasic"
SAMPLING_INTERVAL_US="50"
UNITRACE_BIN=${UNITRACE_BIN:-"unitrace"}
SKIP_STALL=0
SKIP_METRICS=0

OUTDIR="unitrace_q6k_$(date +%Y%m%d_%H%M%S)"

usage() {
    cat <<'EOF'
Usage:
  ./run_unitrace_q6k_profile.sh [options] -- <application> [args...]

Options:
  --output-dir <path>         Output directory (default: unitrace_q6k_<timestamp>)
  --group <metric-group>      Metric group for query/sampling (default: ComputeBasic)
  --sampling-interval <us>    Sampling interval in microseconds (default: 50)
  --unitrace <path>           Path to unitrace binary (default: unitrace)
  --skip-stall                Skip stall sampling step
  -h, --help                  Show this help

Example:
  ./run_unitrace_q6k_profile.sh --output-dir out/unitrace_q6k -- \
    ./build/bin/llama-bench -m /models/model.gguf -p 256
EOF
}

log() {
    echo "[q6k-profile] $*"
}

die() {
    echo "[q6k-profile] ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ $# -ge 2 ]] || die "--output-dir requires a value"
            OUTDIR="$2"
            shift 2
            ;;
        --group)
            [[ $# -ge 2 ]] || die "--group requires a value"
            GROUP="$2"
            shift 2
            ;;
        --sampling-interval)
            [[ $# -ge 2 ]] || die "--sampling-interval requires a value"
            SAMPLING_INTERVAL_US="$2"
            shift 2
            ;;
        --unitrace)
            [[ $# -ge 2 ]] || die "--unitrace requires a value"
            UNITRACE_BIN="$2"
            shift 2
            ;;
        --skip-stall)
            SKIP_STALL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

[[ $# -ge 1 ]] || die "Missing application command. Use -- <application> [args...]"

APP=("$@")

command -v "$UNITRACE_BIN" >/dev/null 2>&1 || die "unitrace not found: $UNITRACE_BIN"

mkdir -p "$OUTDIR"

log "Output directory: $OUTDIR"
log "Application command: ${APP[*]}"
log "Metric group: $GROUP"
log "Sampling interval (us): $SAMPLING_INTERVAL_US"

run_text_step() {
    local name="$1"
    shift
    log "Running: $name"
    "$@" > "$OUTDIR/$name.txt" 2>&1
}

detect_metric_group() {
    local metric_list_file="$1"
    local requested_group="$2"

    # Parse all group names from lines like: "Group N: <name> (...)"
    mapfile -t all_groups < <(sed -nE 's/^Group [0-9]+: ([^ ]+).*/\1/p' "$metric_list_file")

    if [[ ${#all_groups[@]} -eq 0 ]]; then
        SKIP_METRICS=1
        log "No metric groups reported by unitrace. Query/sampling steps will be skipped."
        return
    fi

    for g in "${all_groups[@]}"; do
        if [[ "$g" == "$requested_group" ]]; then
            GROUP="$requested_group"
            return
        fi
    done

    # Prefer a non-stall group for metric-query / metric-sampling.
    for g in "${all_groups[@]}"; do
        if [[ "$g" != "EuStallSampling" ]]; then
            log "Requested group '$requested_group' unavailable. Falling back to '$g'."
            GROUP="$g"
            return
        fi
    done

    SKIP_METRICS=1
    log "Requested group '$requested_group' unavailable and no compatible fallback found."
    log "Available groups: ${all_groups[*]}"
    log "Skipping metric-query/metric-sampling/reorder_only/dmmv_only steps."
}

run_profile_step() {
    local name="$1"
    shift
    log "Running: $name"
    local unitrace_log="$OUTDIR/${name}.unitrace.log"

    "$UNITRACE_BIN" \
        --output-dir-path "$OUTDIR" \
        "$@" \
        "${APP[@]}" \
        2>&1 | tee "$unitrace_log"

    # Harvest all output artifacts reported by unitrace, including files written
    # outside --output-dir-path (e.g. <step>.<pid>.csv, <step>.metrics.<pid>.csv).
    mapfile -t reported_paths < <(sed -nE 's/^\[INFO\] .* stored in (.+)$/\1/p' "$unitrace_log")
    for reported in "${reported_paths[@]}"; do
        [[ -n "$reported" ]] || continue

        local src="$reported"
        if [[ "$src" != /* ]]; then
            src="$PWD/$src"
        fi

        if [[ -f "$src" ]]; then
            # Keep canonical copy in OUTDIR, regardless of where unitrace wrote it.
            if [[ "$(dirname "$src")" != "$OUTDIR" ]]; then
                cp -f "$src" "$OUTDIR/"
            fi
        fi
    done
}

run_text_step "device_list" "$UNITRACE_BIN" --device-list
run_text_step "metric_list" "$UNITRACE_BIN" --metric-list
detect_metric_group "$OUTDIR/metric_list.txt" "$GROUP"

if [[ "$SKIP_METRICS" -eq 0 ]]; then
    log "Using metric group: $GROUP"
fi

run_profile_step "q6k_timing" \
    --device-timing \
    --chrome-kernel-logging \
    --verbose \
    --demangle \
    --include-kernels "$KERNELS" \
    --output q6k_timing.csv

run_profile_step "q6k_submit" \
    --kernel-submission \
    --chrome-kernel-logging \
    --demangle \
    --include-kernels "$KERNELS" \
    --output q6k_submit.csv

if [[ "$SKIP_METRICS" -eq 0 ]]; then
    run_profile_step "q6k_query" \
        --metric-query \
        --group "$GROUP" \
        --chrome-kernel-logging \
        --demangle \
        --include-kernels "$KERNELS" \
        --output q6k_query.csv

    run_profile_step "q6k_sampling" \
        --metric-sampling \
        --group "$GROUP" \
        --sampling-interval "$SAMPLING_INTERVAL_US" \
        --chrome-kernel-logging \
        --demangle \
        --include-kernels "$KERNELS" \
        --output q6k_sampling.csv
else
    log "Skipping q6k_query and q6k_sampling due to unavailable metric group."
fi

if [[ "$SKIP_STALL" -eq 0 ]]; then
    if ! run_profile_step "q6k_stall" \
        --stall-sampling \
        --chrome-kernel-logging \
        --demangle \
        --include-kernels "$KERNELS" \
        --output q6k_stall.csv; then
        log "Stall sampling failed. Continuing. Use --skip-stall to suppress this step."
    fi
else
    log "Skipping stall sampling (--skip-stall)."
fi

run_profile_step "q6k_timeline" \
    --device-timeline \
    --chrome-device-logging \
    --demangle \
    --include-kernels "$KERNELS" \
    --output q6k_timeline.csv

if [[ "$SKIP_METRICS" -eq 0 ]]; then
    run_profile_step "llama_cpp_only" \
        --metric-query \
        --group "$GROUP" \
        --chrome-kernel-logging \
        --demangle \
        --include-kernels "$LLAMA_KERNELS" \
        --output llama_cpp_only.csv

    run_profile_step "ipex_llm_only" \
        --metric-query \
        --group "$GROUP" \
        --chrome-kernel-logging \
        --demangle \
        --include-kernels "$IPEX_KERNELS" \
        --output ipex_llm_only.csv
else
    log "Skipping llama_cpp_only and ipex_llm_only metric-query steps due to unavailable metric group."
fi

log "Done. Results are under: $OUTDIR"
