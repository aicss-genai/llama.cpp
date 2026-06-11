#!/usr/bin/env bash
#
# Compile a SPIR-V module to GenX machine assembly and vISA for a specific Intel
# GPU, capturing the IGC shader dumps (which include register allocation and
# spill statistics).
#
# This is the highest-value disassembly step: the GenX .asm shows exactly what
# the hardware runs (FMA/mad chains, register reuse, spill fills/stores, send
# count, dpas/XMX usage), none of which is visible at the SPIR-V level.
#
# Device ids (ocloc -device):
#   PVC / Data Center GPU Max 1550 = 12.60.7
#   (run `ocloc ids <name>` or `ocloc compile --help` to list others)
#
# Usage:
#   disasm_genx.sh <input.spv> <out-dir> [device-id]
#
#   disasm_genx.sh /tmp/our.spv  /tmp/igc_ours  12.60.7
#   disasm_genx.sh /tmp/ipex.spv /tmp/igc_ipex  12.60.7
#
# Output (in <out-dir>):
#   *.asm       GenX machine assembly (one per kernel)
#   *.visaasm   vISA (virtual ISA, pre-RA) - useful to see compiler intent
#   plus IGC's other dump artifacts. numGRF / spill info is in the .asm header
#   and in the IGC stat dumps.
#
set -euo pipefail

SPV="${1:?path to .spv}"
OUTDIR="${2:?output directory}"
DEVICE="${3:-12.60.7}"

command -v ocloc >/dev/null 2>&1 || {
    echo "ERROR: ocloc not found (source /opt/intel/oneapi/setvars.sh, or run inside the build image)" >&2
    exit 1
}

mkdir -p "$OUTDIR"

# -vc-codegen routes ESIMD / VectorCompute kernels through the VC backend.
# IGC_ShaderDumpEnable + IGC_DumpToCustomDir capture .asm / .visaasm / stats.
IGC_ShaderDumpEnable=1 \
IGC_DumpToCustomDir="$OUTDIR" \
ocloc compile \
    -file "$SPV" \
    -spirv_input \
    -device "$DEVICE" \
    -options "-vc-codegen" \
    -out_dir "$OUTDIR" || true   # ocloc may warn on multi-kernel modules; dumps still land

echo
echo "=== GenX dumps in $OUTDIR ==="
ls -1 "$OUTDIR"/*.asm 2>/dev/null || echo "(no .asm produced - check ocloc output above)"

echo
echo "=== register / spill summary (grep numGRF / spill in .asm headers) ==="
for f in "$OUTDIR"/*.asm; do
    [[ -e "$f" ]] || continue
    printf '%s: ' "$(basename "$f")"
    grep -hoiE "numGRF=[0-9]+|spill.*size.*[0-9]+|spilled" "$f" | tr '\n' ' '
    echo
done
