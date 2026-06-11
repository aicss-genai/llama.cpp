#!/usr/bin/env python3
"""Aggregate unitrace metric-query CSVs per kernel.

unitrace writes one metric row per kernel *launch*. To reason about a kernel's
steady-state behaviour you must aggregate all launches of the same kernel and
look at per-launch averages plus the whole-matmul total GpuTime.

The Kernel column contains commas (it holds the full C++ signature), so the file
MUST be parsed with csv.reader, never positional awk/cut.

Header layout (oneAPI unitrace, ComputeBasic group): the real CSV header is on
the 4th physical line (index 3); the first 3 lines are banner/metadata. Column
names are resolved from that header so this keeps working if the group changes.

Usage:
  analyze_unitrace_metrics.py <metrics.csv> [--filter SUBSTR] [--min-launches N]

Examples:
  # our kernel
  analyze_unitrace_metrics.py out/run/llama_cpp_only.metrics.83.csv --filter q4_K
  # reference kernel
  analyze_unitrace_metrics.py out/run/ipex_llm_only.metrics.93.csv --filter linear_forward
"""

import argparse
import csv
import glob
import sys
from collections import defaultdict

# Columns we care about. Resolved by name against the header so index drift in
# different metric groups does not silently corrupt the analysis.
WANTED = [
    "GpuTime[ns]",
    "XVE_ACTIVE[%]",
    "XVE_STALL[%]",
    "XVE_THREADS_OCCUPANCY_ALL[%]",
    "XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION[%]",
    "XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION[%]",
    "XVE_INST_EXECUTED_SEND_ALL[events]",
    "GPU_MEMORY_BYTE_READ[bytes]",
]

SHORT = {
    "GpuTime[ns]": "GpuTime",
    "XVE_ACTIVE[%]": "ACT%",
    "XVE_STALL[%]": "STALL%",
    "XVE_THREADS_OCCUPANCY_ALL[%]": "OCC%",
    "XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION[%]": "ALU0%",
    "XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION[%]": "ALU1%",
    "XVE_INST_EXECUTED_SEND_ALL[events]": "SEND",
    "GPU_MEMORY_BYTE_READ[bytes]": "bytesRead",
}


def find_header_row(rows):
    """Return the index of the row that starts with the 'Kernel' column."""
    for i, r in enumerate(rows[:10]):
        if r and r[0].strip() == "Kernel":
            return i
    # Fall back to the documented position.
    return 3


def analyze_file(path, name_filter, min_launches):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        print(f"  (empty) {path}")
        return

    h = find_header_row(rows)
    header = rows[h]
    data = rows[h + 1:]

    try:
        idx = {col: header.index(col) for col in WANTED}
    except ValueError as exc:
        print(f"  cannot resolve columns in {path}: {exc}", file=sys.stderr)
        return
    kidx = header.index("Kernel")

    # agg[name] = [sums..., launches]
    agg = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for r in data:
        if len(r) <= max(idx.values()):
            continue
        name = r[kidx]
        if name_filter and name_filter.lower() not in name.lower():
            continue
        try:
            float(r[idx["GpuTime[ns]"]])
        except (ValueError, IndexError):
            continue
        for col, ci in idx.items():
            try:
                agg[name][col] += float(r[ci])
            except (ValueError, IndexError):
                pass
        counts[name] += 1

    if not counts:
        return

    print(f"\n===== {path} =====")
    for name in sorted(counts, key=lambda k: -agg[k]["GpuTime[ns]"]):
        n = counts[name]
        if n < min_launches:
            continue
        total = agg[name]["GpuTime[ns]"]
        print(f"  launches={n:<5d} totalGpuTime={total:,.0f} ns  "
              f"per-launch={total / n:,.0f} ns")
        line = "    "
        for col in WANTED[1:]:
            v = agg[name][col] / n
            line += f"{SHORT[col]}={v:,.1f}  "
        print(line)
        print(f"    {name[:88]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+",
                    help="metric CSV file(s) or glob(s)")
    ap.add_argument("--filter", default="",
                    help="only kernels whose signature contains this substring")
    ap.add_argument("--min-launches", type=int, default=1,
                    help="ignore kernels with fewer than N launches "
                         "(use to isolate the hot matmul shape)")
    args = ap.parse_args()

    paths = []
    for pat in args.csv:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    for p in paths:
        analyze_file(p, args.filter, args.min_launches)


if __name__ == "__main__":
    main()
