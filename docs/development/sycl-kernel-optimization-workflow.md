# SYCL GPU Kernel Optimization Workflow

A reusable, evidence-driven workflow for finding and removing bottlenecks in
llama.cpp SYCL kernels on Intel GPUs (PVC / Data Center GPU Max). It is both a
**summary of how the Q4_K dequantize-matvec (DMMV) ESIMD kernel was optimized
from 62,695 ns to 30,640 ns** (matching the IPEX-LLM reference) and a **guide
for future kernel work**.

Helper scripts live in [scripts/sycl-kernel-opt/](../../scripts/sycl-kernel-opt).

> The golden rule of this workflow: **never optimize on a hunch.** Every change
> was justified by a profiler metric or a machine-code observation, and every
> change was re-measured. Several "obvious" optimizations were measured to be
> neutral or harmful and were reverted. The wins came from the machine code, not
> from guessing.

---

## 0. The loop

```
   pick a metric-backed hypothesis
        |
   make ONE change behind a build flag
        |
   rebuild -> profile -> aggregate metrics  (scripts/analyze_unitrace_metrics.py)
        |
   improved?  --yes-->  keep, update notes
        |
        no
        |
   disassemble (GenX) and find the real cause  (disasm_genx.sh + compare_genx_asm.sh)
        |
   revert the change, form a new hypothesis
```

Keep a running notes file (we used repo memory). Record every experiment,
**including failures**, with the metric numbers. The failures are what stop you
re-trying dead ends.

---

## Techniques, ordered by impact

The ranking below reflects what actually moved the needle on this kernel.
Higher items are cheaper and more often decisive; reach for the lower items only
once the cheap signals are exhausted.

### 1. Profile first and aggregate per kernel (highest impact, do this always)

You cannot optimize what you have not measured, and a single launch's numbers
are noise. unitrace writes **one metric row per kernel launch**; you must
aggregate all launches of the hot kernel.

Run the existing profiler wrapper:

```sh
./run_unitrace_q6k_profile.sh --output-dir out/run-N -- \
    ./build/bin/llama-simple -m /models/model.gguf -p ...
```

Then aggregate the metric CSV per kernel:

```sh
scripts/sycl-kernel-opt/analyze_unitrace_metrics.py \
    out/run-N/llama_cpp_only.metrics.*.csv --filter q4_K --min-launches 100
```

`--min-launches` isolates the hot matmul shape (e.g. the Q4_K {5120} matmul =
900 launches) from warm-up and other shapes.

**CSV gotcha:** the `Kernel` column holds the full C++ signature and contains
commas, and the real header is on the 4th line. Always parse with `csv.reader`
and resolve columns by name -- never with positional `awk`/`cut`. The script
handles both.

**The metrics that matter and what each one tells you:**

| Metric | Reads as | What it points at |
|---|---|---|
| `GpuTime[ns]` (summed over launches) | the only ground-truth score | optimize this, not proxies |
| `XVE_ACTIVE[%]` | fraction of time XVEs issue work | low + low stall => not enough work / occupancy |
| `XVE_STALL[%]` | issue slots lost to stalls | **high => latency-bound** (deps, FMA chains, cache misses) |
| `XVE_THREADS_OCCUPANCY_ALL[%]` | resident threads vs peak | low => too few work-items or too much per-thread register/SLM use |
| `..._ALU0/ALU1_..._UTILIZATION[%]` | FP/INT pipe pressure | which pipe is the real worker; imbalance hints at op-mix issues |
| `..._SEND_ALL[events]` | memory/message instructions | high + high stall => memory-bound; high + low stall => already hidden |
| `GPU_MEMORY_BYTE_READ[bytes]` | DRAM traffic | compare to theoretical bytes; excess => bad layout / re-reads |
| `..._XMX_ACTIVE` / dpas usage | systolic array use | zero on an FP matvec is expected; relevant for GEMM |

**How to read the combination (this is the diagnostic core):**

- **High STALL, ACT moderate, SEND not dominant** -> *latency-bound on
  compute dependencies.* This was Q4_K (STALL 33%). The fix is in the
  instruction schedule (see technique 4), not in memory.
- **High STALL + high SEND + high bytes** -> *memory-bound.* Improve data
  layout, coalescing, or caching. (We tested this hypothesis on Q4_K and
  **disproved** it -- see "fused-meta" failure below.)
- **Low ACT + low STALL + low OCC** -> *not enough parallelism.* Increase
  work-items / rows-per-group or reduce per-thread footprint.
- **OCC capped well below 100%** -> per-thread register or SLM pressure; check
  for spill (technique 3).

### 2. Mirror a known-good reference kernel's *structure* (very high impact)

If a faster implementation of the same math exists (here: IPEX-LLM's
`linear_forward_kernel<float, 2, 4, 16, 12>`), match its **work decomposition**
before micro-tuning: work-group size, rows processed per group, vector widths,
how often the shared operand is loaded. We adopted WG_SIZE=4, 2 output rows per
group, one `block_load<float,256>` activation reused across both rows. This got
us most of the way (to ~40 us) before any disassembly.

Reference its *shape*, not its incidentals. We deliberately did **not** copy
IPEX's scale bit-packing -- measurement showed our `get_scale_min_k4` unpack
does fewer shifts (11 vs 19) and was never the bottleneck.

### 3. Register spill analysis via GenX disassembly (high impact when OCC is low)

Spilling silently destroys performance and is invisible in source. Disassemble
to GenX and check `numGRF` and spill fills/stores.

```sh
# extract your kernel's SPIR-V from the built .so, then compile to GenX
scripts/sycl-kernel-opt/extract_kernel_spirv.sh /tmp/libggml-sycl.so q4_K..._esimd /tmp/our
scripts/sycl-kernel-opt/disasm_genx.sh /tmp/our.spv /tmp/igc_ours 12.60.7
```

The script prints `numGRF` and spill mentions per kernel. On Q4_K, an early
4-accumulator-bank variant spilled (512 B of accumulators -> Function storage,
15 load / 10 store vs the reference's 6/2). It looked like instruction-level
parallelism but the spill negated the win. **Fix: keep the live accumulator
footprint small** -- we collapsed to 2 banks (one per output row). Rule of
thumb: large `simd<>` temporaries that are live across the inner loop are the
usual spill culprits.

### 4. GenX accumulator-chain analysis (the decisive technique here)

When the kernel is **latency-bound (high STALL) but not spilling and not
memory-bound**, the cause is almost always the **dependency chain on the
accumulators**. Inspect the actual `mad` destinations in the GenX asm:

```sh
scripts/sycl-kernel-opt/disasm_genx.sh /tmp/ipex.spv /tmp/igc_ipex 12.60.7
scripts/sycl-kernel-opt/compare_genx_asm.sh /tmp/igc_ours/*.asm /tmp/igc_ipex/*..._2_4_16_12_*.asm
```

What we found on Q4_K (both kernels: 128 GRF, no spill, 3 sends, no dpas -- so
**not** memory- or XMX-bound):

- **Ours:** one accumulator (`acc0`) was the `mad` destination 8 times in a row
  -- an 8-deep **serial** FMA chain. The two output rows were computed in
  **separate** `for (tile = 0; tile < 2)` passes, so within a pass there was no
  independent work to hide FMA latency. -> STALL 33%.
- **Reference:** two accumulators (`r115`, `r116`) each appeared 16 times,
  **interleaved** (`r115, r116, r115, r116, ...`). Two independent chains hide
  each other's latency. -> STALL low, ACT 57%.

**The fix (the change that closed the gap):** restructure the inner loop so both
output rows update in the *same* iteration instead of in serial passes:

```cpp
// before: serial -> latency exposed
for (int tile = 0; tile < 2; ++tile) {
    simd<float,32>& acc = (tile == 0) ? acc0 : acc1;
    for (q2 ...) acc += rhs * deq;     // one chain at a time
}

// after: interleaved -> latency hidden
for (q2 ...) {
    acc0 += rhs * deq_row0;            // two independent chains,
    acc1 += rhs * deq_row1;            // compiler co-schedules them
}
```

Keep the **same number of accumulators** (2). This is co-scheduling, not adding
banks, so it does not reintroduce spill.

Result: 40,700 -> 30,640 ns (-25%), STALL 33.5% -> 22.4% (exactly the targeted
metric), ACT 54.5% -> 61.7%. We now match the reference and exceed its activity.

### 5. SPIR-V disassembly (lower impact -- use to confirm, not to drive)

SPIR-V text is useful to confirm that a *source* change actually changed the
program, or to compare op mix against a reference SPIR-V dump. But it is far
from the metal: `block_load` lowers to a plain `OpLoad`; the LSC messages,
register allocation, and scheduling only appear after the IGC VC backend.

```sh
# llvm-spirv handles VectorComputeINTEL (ESIMD); spirv-dis does NOT.
$ONEAPI/compiler/<ver>/bin/compiler/llvm-spirv -to-text our.spv -o our.spvasm
```

**Key lesson (a recorded failure):** a source-level "narrow the live window"
tweak produced **byte-identical SPIR-V** -- the compiler hoists/sinks the same
way regardless of source order. The +5% wall-clock we first saw was IGC
register-allocation variance from reshuffling source, not a real effect. *Source
micro-tweaks do not move the SPIR-V; only algorithm/footprint changes do.* Use
SPIR-V diffing to *prove* a change is real before you bother profiling it.

---

## Worked timeline (Q4_K {5120} matvec, 900 launches, whole-matmul GpuTime)

| Variant | Time | x base | STALL | ACT | OCC | Verdict |
|---|--:|--:|--:|--:|--:|---|
| best scalar SOA simd16 | 62,695 ns | 1.00x | -- | -- | -- | baseline |
| ESIMD v1 (256-wide deq buffer) | 42,109 ns | 1.49x | 35.0% | -- | 76.3% | mirror reference structure (tech 2) |
| ESIMD 4-bank ILP | 41,727 ns | 1.50x | 32.7% | -- | -- | **spilled** -> reverted (tech 3) |
| ESIMD 2-bank | 40,700 ns | 1.54x | 33.5% | 54.5% | 76.1% | small footprint, no spill |
| ESIMD narrow-window | 42,861 ns | -- | -- | -- | -- | identical SPIR-V, RA noise -> reverted (tech 5) |
| ESIMD fused-meta layout | 41,366 ns | -- | 39.0% | -- | -- | SEND -45% but +1.6% time -> **not memory-bound** -> reverted |
| **ESIMD interleaved rows** | **30,640 ns** | **2.05x** | **22.4%** | **61.7%** | **74.9%** | **GenX acc-chain fix (tech 4)** |
| IPEX reference | ~30,000 ns | -- | -- | 57.5% | 72.1% | target reached |

The two reverts in the middle are as important as the wins: they are the
evidence that ruled out spill-driven ILP and memory layout, leaving the
accumulator dependency chain as the real cause.

---

## Generalizing to any SYCL kernel

1. **Profile and aggregate per kernel** (`analyze_unitrace_metrics.py`). Get
   `GpuTime`, `STALL`, `ACT`, `OCC`, `SEND`, `bytes`.
2. **Classify the bottleneck from the metric combination** (table in tech 1):
   latency-bound, memory-bound, or occupancy-bound.
3. If a faster reference exists, **match its work decomposition first** (tech 2).
4. **Disassemble to GenX** and check, in order: spill (`numGRF`, fills/stores),
   `send` count, `dpas` usage, then the **accumulator/dependency chain**
   (`compare_genx_asm.sh`).
5. **Map metric to machine-code cause:**
   - high STALL + serial mad chain -> interleave independent accumulators.
   - low OCC + spill -> shrink live `simd<>` footprint.
   - high SEND + high bytes -> fix layout/coalescing (verify it is actually the
     bottleneck before investing -- removing already-hidden messages does
     nothing for wall-clock).
6. **Change one thing behind a build flag, re-measure, keep or revert.** Record
   every result, especially failures.

## Environment reference

- Build: `docker build -t llama-cpp-sycl --target full -f .devops/intel.Dockerfile .`
- ESIMD Q4_K path is gated by `-DGGML_SYCL_Q4K_DMMV_ESIMD=1` (default OFF).
- PVC device id for `ocloc`: `12.60.7`.
- `out/` may be root-owned (created by the container); do disassembly work in
  `/tmp`.
- Extract the device image from the image: `docker create --entrypoint bash
  llama-cpp-sycl`, then `docker cp <cid>:/app/libggml-sycl.so /tmp/`.

## Script reference

| Script | Purpose |
|---|---|
| [analyze_unitrace_metrics.py](../../scripts/sycl-kernel-opt/analyze_unitrace_metrics.py) | Aggregate unitrace metric CSVs per kernel (per-launch averages + total GpuTime). |
| [extract_kernel_spirv.sh](../../scripts/sycl-kernel-opt/extract_kernel_spirv.sh) | Pull one kernel's SPIR-V module out of `libggml-sycl.so` and disassemble it (VectorComputeINTEL-aware). |
| [disasm_genx.sh](../../scripts/sycl-kernel-opt/disasm_genx.sh) | Compile a `.spv` to GenX `.asm` + vISA via `ocloc -vc-codegen`, with spill/`numGRF` summary. |
| [compare_genx_asm.sh](../../scripts/sycl-kernel-opt/compare_genx_asm.sh) | Diff two GenX `.asm` files: spill, send, dpas, opcode histogram, and accumulator-chain structure. |
