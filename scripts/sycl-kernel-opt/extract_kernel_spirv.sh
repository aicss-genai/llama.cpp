#!/usr/bin/env bash
#
# Extract a single kernel's SPIR-V module from a built libggml-sycl.so and
# disassemble it to readable SPIR-V text.
#
# Background:
#   The SYCL device code is stored in an offload bundle section of the shared
#   object. With per-kernel device-code-split, that section is a concatenation
#   of ~hundreds of standalone SPIR-V modules, each beginning with the SPIR-V
#   magic word 0x07230203 (little-endian: 03 02 23 07).
#
#   spirv-dis FAILS on these modules because they use the VectorComputeINTEL
#   capability (5910, emitted for ESIMD / -vc-codegen kernels). The oneAPI
#   llvm-spirv tool disassembles them fine (numeric text form: op names have NO
#   "Op" prefix, e.g. "FMul" not "OpFMul").
#
# Usage:
#   extract_kernel_spirv.sh <libggml-sycl.so> <kernel-name-substr> [out-prefix]
#
#   # from a docker image:
#   cid=$(docker create --entrypoint bash llama-cpp-sycl)
#   docker cp "$cid:/app/libggml-sycl.so" /tmp/libggml-sycl.so
#   docker rm "$cid"
#   extract_kernel_spirv.sh /tmp/libggml-sycl.so q4_K_sycl_reorder_esimd /tmp/our
#
# Output:
#   <out-prefix>.spv       extracted single-kernel SPIR-V module
#   <out-prefix>.spvasm    disassembled SPIR-V text
#
set -euo pipefail

SO="${1:?path to libggml-sycl.so}"
NEEDLE="${2:?kernel name substring}"
OUT="${3:-/tmp/kernel}"

ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
LLVM_SPIRV="${LLVM_SPIRV:-}"
if [[ -z "$LLVM_SPIRV" ]]; then
    LLVM_SPIRV="$(find "$ONEAPI_ROOT/compiler" -name llvm-spirv -type f 2>/dev/null | sort | tail -1 || true)"
fi
[[ -n "$LLVM_SPIRV" && -x "$LLVM_SPIRV" ]] || {
    echo "ERROR: llvm-spirv not found. Set LLVM_SPIRV=/opt/intel/oneapi/compiler/<ver>/bin/compiler/llvm-spirv" >&2
    exit 1
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# 1. Pull the offload bundle out of the .so.
SECTION="__CLANG_OFFLOAD_BUNDLE__sycl-spir64"
objcopy -O binary --only-section="$SECTION" "$SO" "$WORK/bundle.bin"
[[ -s "$WORK/bundle.bin" ]] || { echo "ERROR: section $SECTION empty/missing" >&2; exit 1; }

# 2. Locate the named kernel's byte offset inside the bundle, then find the
#    SPIR-V magic word that starts the module containing it.
NAME_OFF="$(strings -t d "$WORK/bundle.bin" | grep -m1 "$NEEDLE" | awk '{print $1}' || true)"
[[ -n "$NAME_OFF" ]] || { echo "ERROR: kernel '$NEEDLE' not found in bundle" >&2; exit 1; }

# Collect all SPIR-V magic offsets (03 02 23 07).
mapfile -t MAGICS < <(
    python3 - "$WORK/bundle.bin" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
magic = b"\x03\x02\x23\x07"
i = 0
while True:
    j = data.find(magic, i)
    if j < 0:
        break
    # Require 4-byte alignment to avoid false hits inside string tables.
    if j % 4 == 0:
        print(j)
    i = j + 4
PY
)

START=0
END="$(stat -c %s "$WORK/bundle.bin")"
for off in "${MAGICS[@]}"; do
    if (( off <= NAME_OFF )); then
        START="$off"
    else
        END="$off"
        break
    fi
done

echo "kernel '$NEEDLE' name@$NAME_OFF  module [$START, $END)"
dd if="$WORK/bundle.bin" of="$OUT.spv" bs=1 skip="$START" count=$((END - START)) status=none

# 3. Disassemble (VectorComputeINTEL-aware).
"$LLVM_SPIRV" -to-text "$OUT.spv" -o "$OUT.spvasm"
echo "wrote $OUT.spv and $OUT.spvasm"
