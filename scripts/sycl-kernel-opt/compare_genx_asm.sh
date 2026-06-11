#!/usr/bin/env bash
#
# Diff two GenX .asm files (your kernel vs a reference kernel) along the axes
# that actually explain a wall-clock gap:
#
#   1. Register pressure / spill   -> numGRF, spill loads/stores
#   2. Memory messages             -> send count
#   3. XMX / systolic usage        -> dpas count
#   4. Compute opcode mix          -> mad/mul/add/shr/and/... histogram
#   5. Accumulator chain structure -> how often each accumulator GRF is the
#      destination of a mad, and whether two accumulators are INTERLEAVED
#      (latency-hiding) or used in long SERIAL chains (latency-exposed).
#
# (5) is the one that closed the Q4_K gap: a serial 8-deep mad chain on a single
# accumulator stalls on FMA latency; two accumulators updated alternately hide
# each other's latency. STALL% in the profiler is the corresponding symptom.
#
# Usage:
#   compare_genx_asm.sh <ours.asm> <reference.asm>
#
set -euo pipefail

OURS="${1:?ours.asm}"
REF="${2:?reference.asm}"

section() { printf '\n========== %s ==========\n' "$1"; }

count() { grep -hcE "$1" "$2" 2>/dev/null || echo 0; }

report_one() {
    local label="$1" f="$2"
    echo "--- $label ($(basename "$f")) ---"
    echo "  numGRF       : $(grep -hoiE 'numGRF=[0-9]+' "$f" | head -1)"
    echo "  spill        : $(grep -hciE 'spill' "$f") lines mention spill"
    echo "  send (memory): $(count 'send[s]?[ .(]' "$f")"
    echo "  dpas (XMX)   : $(count '\bdpas\b' "$f")"
    echo "  mad          : $(count '\bmad\b' "$f")"
    echo "  mul          : $(count '\bmul\b' "$f")"
    echo "  add          : $(count '\badd\b' "$f")"
    echo "  mov          : $(count '\bmov\b' "$f")"
    echo "  branches     : $(count '\bjmpi\b|\bgoto\b|\bif\b|\bwhile\b' "$f")"
}

section "PER-KERNEL SUMMARY"
report_one "OURS" "$OURS"
echo
report_one "REF " "$REF"

section "OPCODE HISTOGRAM (top 15)"
hist() {
    # Pull the mnemonic: first token after an optional "(label)" predicate.
    grep -hoE '^\s*\(?[A-Za-z0-9_.]*\)?\s*[a-z][a-z0-9_]+' "$1" \
        | awk '{print $NF}' | sort | uniq -c | sort -rn | head -15
}
echo "OURS:"; hist "$OURS"
echo "REF:";  hist "$REF"

section "ACCUMULATOR CHAIN STRUCTURE"
echo "How many times each register is the DESTINATION of a 'mad' (the deeper a"
echo "single register's chain, the more serial / latency-exposed it is)."
echo
acc_dest() {
    grep -hoE '\bmad\b[^,]*' "$1" \
        | grep -oE '(acc[0-9]+|r[0-9]+)' \
        | sort | uniq -c | sort -rn | head -8
}
echo "OURS mad destinations:"; acc_dest "$OURS"
echo "REF  mad destinations:"; acc_dest "$REF"
echo
echo "INTERPRETATION:"
echo "  - One register dominating (e.g. acc0 x8) => serial chain, FMA latency"
echo "    exposed, expect high XVE_STALL%. Restructure so two+ accumulators are"
echo "    updated in the SAME loop iteration (interleaved) to hide latency."
echo "  - Two registers with equal high counts => already interleaved (good)."
echo
echo "To eyeball interleaving, look at the raw mad order:"
echo "  grep -nE '\\bmad\\b' $OURS | grep -oE '(acc[0-9]+|r[0-9]+)' | head -32"
