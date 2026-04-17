# SYCL Optimization Workbook — Intel Battlemage Arc Pro B70

## Hardware Reference

| Parameter | Value |
|---|---|
| GPU | Intel Arc Pro B70 (Xe2-HPG, BMG-G31) |
| Execution Units | 256 |
| Sub-slices | 20 Xe-cores |
| SIMD Width | 16 (native) |
| Peak Memory BW | 542.3 GB/s (GDDR6, 19 Gbps x 256-bit) |
| VRAM | 32 GB GDDR6 |
| XMX (INT8) | ~TBD TOPS |
| PCIe | Gen5 x8 |
| Host CPU | Intel Xeon w9-3475X |

## Model Under Test

| Parameter | Value |
|---|---|
| Model | Qwen3.5-9B Q4_K_M |
| File | `models/Qwen3.5-9B-Q4_K_M.gguf` |
| Model Size | 5.67 GB (5,669,554,176 bytes) |
| Parameters | 8.95B |
| Quant Types in Model | Q4_K (majority), Q5_K, Q6_K, Q8_0 (attention, norm weights) |

## Roofline Analysis

For decode (autoregressive token generation), the workload is **memory-bandwidth bound**.
Each token requires reading the full model weights once from VRAM.

```
Theoretical max decode throughput = Peak BW / Model Size
                                  = 542.3 GB/s / 5.67 GB
                                  = 95.6 tok/s
```

This is the hard ceiling. Real throughput is lower due to:
- KV cache reads (grows with context length)
- Activation memory traffic
- Kernel launch overhead and synchronization
- Memory access pattern inefficiencies (non-coalesced loads, bank conflicts)
- Quantization overhead (scale lookups, bit unpacking)

A practical target of **75-80% roofline** = 71.7 - 76.5 tok/s.

## Optimization Progression — Decode (tg128)

| # | Change | tg128 (tok/s) | vs Prior | vs Initial | Roofline % | Status |
|---|---|---:|---|---|---:|---|
| 0 | Initial broken runtime | 1.17 | baseline | — | 1.2% | superseded |
| 1 | AOT device-arch fix | 50.32 | +4199% | +4199% | 52.6% | accepted |
| 2 | Async mem-op decoupled | 50.27 | -0.1% | +4195% | 52.6% | accepted |
| 3 | Real AOT flag fix (spir64_gen) | 50.01 | -0.5% | +4173% | 52.3% | accepted |
| 4 | Q6_K SWAR byte-sub scalarized | 52.56 | +5.1% | +4391% | 55.0% | accepted |
| 5 | Q5_K reorder MMVQ (validated) | 57.19 | +8.8% | +4787% | 59.8% | accepted |
| 6 | Q8_0 reorder MMVQ + dequant fix | 58.64 | +2.5% | +4911% | 61.3% | accepted |
| 7 | Q5_K reorder dequant fix (correctness) | 58.61 | 0% | +4908% | 61.3% | accepted |
| 8 | **Q5_K interleaved qs/qh layout (tested, reverted)** | **58.51** | **-0.2%** | **+4899%** | **61.2%** | **rejected (neutral)** |

### Roofline Gap Analysis

```
Current:     58.36 tok/s  =  61.0% of 95.6 tok/s roofline  (full sweep, r=5)
Target:      76.5  tok/s  =  80.0% of roofline
Gap:         18.1  tok/s  =  19.0 percentage points

Effective BW: 58.36 * 5.67 GB = 330.9 GB/s of 542.3 GB/s peak
BW gap:       211.4 GB/s wasted
```

### Steady-State Decode Profile (p=1, n=128 with warmup)

Profile: `results/profiles/2026-03-12-v2-steady-decode-128/`
Image: `llama-sycl-profiler:0.14.0-b7.1-4d99d45-q5k-q8_0-reorder-v2`

#### Per-Token Time Budget (17.06 ms/tok = 58.63 tok/s)

| Category | ms/tok | % of Total |
|---|---:|---:|
| **MMVQ kernels** | **10.94** | **64.1%** |
| — Q4_K reorder MMVQ | 6.01 | 35.2% |
| — Q6_K reorder MMVQ | 2.94 | 17.3% |
| — Q5_K reorder MMVQ | 2.03 | 11.9% |
| — Q8_0 reorder MMVQ | 0.59 | 3.5% |
| **Non-MMVQ GPU ops** | **5.17** | **30.3%** |
| — Elementwise ops | 2.51 | 14.7% |
| — Data movement | 1.52 | 8.9% |
| — Quantization (Q8_1) | 0.37 | 2.2% |
| — Normalization | 0.31 | 1.8% |
| — Attention + RoPE | 0.23 | 1.3% |
| — Other (sum_rows, ssm_conv) | 0.22 | 1.3% |
| **Host overhead** | **0.95** | **5.6%** |

#### MMVQ Bandwidth Efficiency

```
MMVQ time:        1399.7 ms  (128 tokens)
Weight data read: 128 × 5.67 GB = 725.8 GB
Effective BW:     725.8 / 1.400 = 518.5 GB/s
Peak BW:          542.3 GB/s
MMVQ BW %:        95.6%
```

**The MMVQ kernels are near peak memory bandwidth.** The entire roofline gap
comes from non-MMVQ overhead (6.12 ms/tok) where the GPU is not reading
model weights.

#### Non-MMVQ Overhead Breakdown

| Kernel | ms/tok | Calls/tok | us/call | Notes |
|---|---:|---:|---:|---|
| op_mul broadcast | 1.34 | 283 | 4.7 | SSM/linear-attention pointwise multiply |
| f32→f32 copy | 0.83 | 81 | 10.3 | Tensor reshaping, KV cache |
| op_add broadcast | 0.44 | 113 | 3.9 | Residual connections + SSM/linear-attention |
| get_rows | 0.41 | 50 | 8.2 | Embedding lookups, SSM state |
| op_repeat broadcast | 0.29 | 73 | 4.0 | Broadcasting for SSM/linear-attention gates |
| quantize q8_1 | 0.37 | 251 | 1.5 | Input quantization for MMVQ (1:1 with MMVQ calls) |
| concat | 0.23 | 24 | 9.5 | KV concat or Mamba state |
| flash_attn | 0.17 | 8 | 20.8 | Flash attention |
| sum_rows | 0.16 | 48 | 3.3 | Softmax / SSM reduction |
| norms + activations | 0.62 | 307 | 2.0 | RMSNorm, L2Norm, SiLU, sigmoid, etc. |
| rope + scale + other | 0.14 | 73 | 2.0 | RoPE, scaling, misc |

**Key observation:** Qwen3.5-9B is a hybrid architecture that interleaves linear-attention/state-space layers (likely DeltaNet or Mamba-variant). The SSM/linear-attention
layers require many elementwise operations (op_mul: 283/tok, op_add: 113/tok,
sigmoid, exp, softplus, ssm_conv) that pure attention models don't have. This
structural overhead accounts for ~2.5 ms/tok.

#### What This Means for Optimization

To reach 75% roofline (71.7 tok/s = 13.95 ms/tok), we need to save 3.11 ms/tok:
- MMVQ kernels are at 95.6% peak BW — minimal gains available
- Must reduce non-MMVQ overhead from 6.12 to 3.01 ms/tok (50% reduction)
- Kernel fusion is the primary lever: reduce ~1500 kernel launches/tok
- Host API overhead (kernel launch, sync) scales linearly with launch count

## Optimization Progression — Prefill (pp512)

| # | Change | pp512 (tok/s) | vs Prior | vs Initial | Status |
|---|---|---:|---|---|---|
| 0 | Initial broken runtime | 15.07 | baseline | — | superseded |
| 1 | AOT device-arch fix | 330.07 | +2090% | +2090% | accepted |
| 2 | Async mem-op decoupled | 349.45 | +5.9% | +2218% | accepted |
| 3 | Real AOT flag fix | 345.49 | -1.1% | +2192% | accepted |
| 4 | Q6_K SWAR byte-sub scalarized | 342.92 | -0.7% | +2175% | accepted |
| 5 | Q5_K reorder MMVQ | 339.50 | -1.0% | +2152% | accepted |
| 6 | Q8_0 reorder MMVQ + dequant fix | 343.65 | +1.2% | +2180% | accepted |
| 7 | Q5_K reorder dequant fix (correctness) | 334.81 | -2.6% | +2121% | accepted |
| 8 | **Q5_K interleaved qs/qh layout (tested, reverted)** | **332.84** | **-0.6%** | **+2108%** | **rejected** |
| 9 | **GGML_SYCL_F16=ON** (Llama-3.1-8B pp512) | **2772.71** | **+155%** | — | **accepted** |

Note: Step 9 measured on Llama-3.1-8B (pure attention), not Qwen3.5-9B (hybrid linear-attention/SSM).

## What Each Change Did and Why

### Step 1: AOT Device-Arch Fix
**Problem:** The SYCL binary was being JIT-compiled at runtime for every kernel launch.
**Fix:** Added auto-detection of BMG-G31 PCI device ID and compiled ahead-of-time with `spir64_gen -device bmg-g31`.
**Why it works:** Eliminates ~30s of `urProgramBuildExp` / `zeModuleCreate` at startup and produces device-optimized ISA.

### Step 2: Async Memory Operations
**Problem:** `GGML_SYCL_USE_ASYNC_MEM_OP` was gated behind graph mode, which is disabled by default.
**Fix:** Decoupled async mem-op from graph enablement so weight reordering and memory copies can overlap with compute.
**Why it works:** The reorder path (`opt_for_reorder`) rearranges quantized weights from struct-of-arrays to separate contiguous qs/d regions. Async copies allow this to pipeline with kernel execution.

### Step 3: Real AOT Flag Fix
**Problem:** The build was using a generic SPIR-V target instead of the real Intel GPU offline compiler path.
**Fix:** Switched to `spir64_gen` with OCLOC offline compilation and removed incompatible flags.
**Why it works:** Produces truly ahead-of-time compiled GPU binaries instead of intermediate SPIR-V that still needs online finalization.

### Step 4: Q6_K SWAR Byte-Sub Scalarization
**Problem:** The reordered Q6_K vec_dot inner loop used DPCT-generated byte-vector subtract operations with array temporaries and a 2-iteration loop.
**Fix:** Replaced with hand-written SWAR 4-byte subtract and explicit 2-slice scalarized code.
**Why it works:** Eliminates array/loop overhead, gives the compiler explicit independent operations to schedule. The Q6_K vec_dot is called for every Q6_K weight block during decode.

### Step 5: Q5_K Reorder MMVQ
**Problem:** Q5_K tensors were dispatched to the DMMV path (16 threads/workgroup) instead of the reorder MMVQ path (256 threads/workgroup with dp4a).
**Fix:** Added `block_q_t<GGML_TYPE_Q5_K>` traits, `reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>` operator, reorder launcher, and reorder function.
**Why it works:** 16x more parallelism per workgroup, dp4a hardware dot-product instructions, and coalesced memory access from the reordered layout.

### Step 6: Q8_0 Reorder MMVQ + Dequantize Fix
**Problem:** Q8_0 tensors (attention/norm weights) were on the DMMV path. Additionally, the initial Q8_0 reorder implementation was missing a reorder-aware dequantize function, causing -4.3% prompt regression when the prefill path (MUL_MAT_SYCL) read reordered data with the standard dequantize that expects block_q8_0 struct layout.
**Fix:** Added full Q8_0 reorder MMVQ path (traits, vec_dot, launcher, reorder function) plus reorder-aware dequantize kernel and dispatch in convert.cpp.
**Why it works:** Same 16x parallelism gain as Q5_K. The dequantize fix ensures the prefill path correctly reads the reordered memory layout (contiguous qs followed by contiguous d values).

### Step 7: Q5_K Reorder Dequantize Fix (Correctness)
**Problem:** Same bug class as Q8_0 Step 6. Q5_K was in `supports_reorder_mmvq` but NOT in `supports_reorder_mul_mat_sycl`, and had no reorder-aware dequantize function. The prefill path read reordered Q5_K data with the standard dequantize (expects block_q5_K struct layout with interleaved qs/qh/scales/dm). This produced corrupted dequantized values.
**Fix:** Added `dequantize_block_q5_K_reorder` kernel (reads contiguous qs, qh, scales, dm regions at computed offsets within the reordered buffer) and `dequantize_row_q5_K_sycl_reorder` launcher. Added Q5_K to `supports_reorder_mul_mat_sycl` dispatch. Updated `ggml_get_to_fp16_sycl` and `ggml_get_to_fp32_sycl` to check reorder flag for Q5_K.
**Why it works:** Ensures prefill (MUL_MAT_SYCL) correctly reads the SoA reordered layout. The -2.6% pp512 regression (343.65→334.81) vs the Q8_0-only build is the cost of correctness — the prior "339.50" was reading partially corrupted data.

### Step 8: Q5_K Interleaved qs/qh Layout (Tested, Reverted)
**Problem:** Q5_K achieves only 79.8% BW efficiency vs Q4_K's 107.7%. Profile shows the qh (high-bit) region is stored ~12.5 MB from qs in the SoA reorder layout, potentially causing cache/TLB pressure.
**Fix attempted:** Changed Q5_K reorder layout from `[all qs][all qh][scales][dm]` to `[qs₀|qh₀][qs₁|qh₁]...[scales][dm]` — interleaving qh immediately after qs per block (160B stride).
**Result:** No measurable change. tg128: 58.51 vs 58.61 (-0.2%, noise). pp512: 332.84 vs 334.81 (-0.6%, noise).
**Root cause:** The Q5_K BW gap is not spatial-locality-driven. It's caused by the extra qh memory load transactions (2 additional cache line fetches per sub-group per iteration) and ALU work for bit manipulation (shift + mask + OR). This is a structural cost of the Q5_K 5-bit format — each sub-group needs ~8 memory transactions vs Q4_K's ~6, regardless of data layout. The 79.8% efficiency matches the expected 6/8 = 75% ratio.
**Decision:** Reverted to maintain consistent SoA layout across quant types. No code changes retained.

### Step 9: GGML_SYCL_F16=ON (Build Flag Fix)
**Problem:** Our build was missing `-DGGML_SYCL_F16=ON`. Without it, the prefill
path (MUL_MAT_SYCL) dequantizes Q4_K weights to FP32 and runs FP32 GEMM via
oneDNN. XMX FP16 throughput is 2x FP32, so this was leaving half the compute
on the table for the compute-bound prefill path.
**Fix:** Added `-DGGML_SYCL_F16=ON` to the cmake configuration in
`scripts/build_llama_sycl_container.sh`.
**Why it works:** With this flag, `ggml-sycl.cpp:2153` sets `use_fp16 = true`,
which routes through `ggml_get_to_fp16_sycl()` for dequantize and calls oneDNN
GEMM with `fpmath_mode::f16` (gemm.hpp:58). This uses XMX FP16 engines which
have 2x the throughput of FP32 ALU.
**Result:** pp512 on Llama-3.1-8B: 1087 → 2773 tok/s (+155%). Decode unchanged
(86 tok/s) because it uses MMVQ kernels, not GEMM.
**Discovery:** Found by comparing our cmake flags against the stock xe_bench
build script (`references/xe_bench-main/llamacpp/01_setup.sh`), which had
`GGML_SYCL_F16=ON` but no `GGML_SYCL_DEVICE_ARCH` (no AOT).

## Known Bugs

All known bugs (Q8_0 dequantize, Q5_K dequantize) have been fixed as of Step 7.

## Remaining Optimization Opportunities

### Tier 0 — Prefill (1.5-1.8x gap vs CUDA, primary target)

Llama-3.1-8B Q4_K_M prefill after F16+AOT: our 2773 tok/s vs CUDA 4906 tok/s (1.77x gap).

**Completed:**
- ~~Enable `GGML_SYCL_F16=ON`~~ — Done (Step 9). 2.55x prefill improvement.
- ~~Fused dequant+GEMM~~ — Attempted (Step 10). **Dead end** — can't out-GEMM oneDNN.

**Current focus:**
1. **Enable MMQ kernel** (Step 11) — CUDA-style quantized GEMM already exists in mmq.cpp.
   Uses dp4a integer dot products on Q4_K×Q8_1, never dequantizes to FP16.
   Eliminates the 3-kernel pipeline overhead (dequant+convert = 50% of MUL_MAT time).
   Currently disabled due to "accuracy issues" — find and fix bugs, then benchmark.

2. **Upgrade MMQ to DPAS INT4×INT8** (after MMQ works) — Replace scalar dp4a with
   `dpas.u4.s8.8.8` systolic instruction. K=32 per DPAS vs 4 per dp4a = 8x more
   compute per instruction. This is the INT4 equivalent of Tensor Cores.

3. **Flash Attention** (30 ms, 15.6%) — secondary priority after MMQ.

### Tier 1 — Decode on hybrid models with linear-attention/SSM layers (Qwen3.5-9B specific)

Decode on dense models (Llama-3.1-8B) is at **72.5% roofline** — near target.
Qwen3.5-9B is at 61% due to non-MMVQ overhead from DeltaNet/Mamba ops.

4. **Kernel fusion** — Reduce ~1500 kernel launches/tok to <500.
   - RMSNorm + Q8_1 quantize, residual add + norm, SSM/linear-attention gate chains
   - This is graph-level work, not individual kernel optimization
   - Estimated impact: 1-2 ms/tok from launch overhead + data movement

5. **Reduce f32 copy overhead** — 0.83 ms/tok (81 calls/tok)

6. **Q8_1 input quantization sharing** — 0.37 ms/tok for 251 launches

### Tier 2 — Marginal MMVQ gains (low priority)

7. **Q4_K/Q6_K** — At peak BW (107.7%/100.3%). No kernel gains available.
8. **Q5_K** — At 79.8% BW. Structural gap from extra qh loads. Not fixable.

### Tier 3 — Architectural changes

9. **Multi-row MMVQ** — First attempt regressed, needs different approach
10. **Graph-level operator scheduling** — Overlap compute with data movement

## Build Configuration Comparison

| Flag | Stock xe_bench (b7973) | Our Build (4d99d45) | Impact |
|---|---|---|---|
| `GGML_SYCL_F16` | **ON** | **ON** (Step 9) | FP16 GEMM via oneDNN XMX |
| `GGML_SYCL_DEVICE_ARCH` | not set | **bmg-g31** | AOT = ~2x decode improvement |
| `GGML_SYCL_DNN` | not set | **ON** | oneDNN GEMM |
| `GGML_SYCL_GRAPH` | not set | **ON** | Graph capture |
| `GGML_BACKEND_DL` | ON | not set | Dynamic loading |

Our build now has the best of both: AOT (decode) + FP16 GEMM (prefill).

## Files Modified (Cumulative)

| File | Changes |
|---|---|
| `ggml/src/ggml-sycl/quants.hpp` | Added Q5_K, Q8_0 reorder block traits |
| `ggml/src/ggml-sycl/vecdotq.hpp` | Added Q5_K, Q8_0 reorder vec_dot operators; Q6_K SWAR scalarization |
| `ggml/src/ggml-sycl/mmvq.cpp` | Added Q5_K, Q8_0 reorder launchers and dispatch |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Added Q5_K, Q8_0 to `supports_reorder_mmvq` and `supports_reorder_mul_mat_sycl`; added Q5_K, Q8_0 reorder functions |
| `ggml/src/ggml-sycl/dequantize.hpp` | Added Q8_0, Q5_K reorder dequantize kernels |
| `ggml/src/ggml-sycl/convert.cpp` | Added Q8_0, Q5_K reorder dequantize launchers and dispatch |

## Full Sweep Benchmark — Final Build (2026-03-12)

Image: `llama-sycl-local:0.14.0-b7.1-4d99d45-final`
Results: `results/2026-03-12-full-sweep/Qwen3.5-9B-Q4_K_M-single-stream.json`
All runs: r=5, b=2048, ub=2048, fa=1, ngl=99, sm=none

### Prefill (prompt processing)

| Prompt Length | tok/s | σ | Notes |
|---:|---:|---:|---|
| 512 | 342.13 | 3.85 | |
| 1024 | 355.12 | 4.02 | |
| 2048 | 358.37 | 1.72 | Peak prefill throughput |
| 4096 | 344.84 | 3.41 | Slight drop at max context |

### Decode (token generation)

| Generation Length | tok/s | σ | Roofline % |
|---:|---:|---:|---:|
| 128 | 58.36 | 0.084 | 61.0% |
| 256 | 58.33 | 0.093 | 61.0% |
| 512 | 58.31 | 0.040 | 61.0% |

### Key Observations

- **Decode throughput is remarkably stable** across generation lengths (58.31-58.36 tok/s, <0.1% variation). KV cache growth has negligible impact at these context lengths.
- **Prefill peaks at pp2048** (358.37 tok/s), with a slight dip at pp4096 likely from increased attention compute cost.
- **Very low variance** across all measurements — σ < 4 tok/s for prefill, σ < 0.1 tok/s for decode.
- **Current standing: 61.0% of roofline.** Gap to 75% target = 13.3 tok/s (22.8% improvement needed).

## Re-validation Sweep (2026-03-16)

After Steps 10-11 (both disabled/reverted), re-validated with MMQ/DPAS code compiled but
dispatch disabled. Confirmed identical performance to Step 7 build via back-to-back comparison.

Build: `build-sycl-0.14.0-b7.1` (MMQ disabled, all Step 1-9 optimizations active)
All runs: r=5, b=2048, ub=2048, fa=1, ngl=99

### Prefill (matches Mar 12 baseline)

| Prompt Length | tok/s | σ | vs Mar 12 |
|---:|---:|---:|---|
| 512 | 342.26 | 10.84 | 0% (same) |
| 1024 | 351.07 | 6.67 | -1.1% (noise) |
| 2048 | 330.58 | 4.87 | -7.7% (thermal) |

### Decode (system-level regression, not code)

| Generation Length | tok/s | σ | vs Mar 12 | Roofline % |
|---:|---:|---:|---|---:|
| 128 | 52.33 | 0.06 | -10.3% | 54.7% |
| 256 | 52.31 | 0.03 | -10.3% | 54.7% |
| 512 | 52.27 | 0.02 | -10.3% | 54.7% |

**Decode regression is environmental.** Verified by running the unmodified Step 7 build
(`build-sycl-0.14.0-b7.1-4d99d45-aot-real-bmg-g31`, dated 2026-03-11) which shows the
same 52.46 tok/s. Likely cause: host driver or kernel update between Mar 12 and Mar 16
(Linux 6.19.6-2-cachyos). Prefill (compute-bound) is unaffected; only decode (BW-bound)
regressed, suggesting a memory subsystem change.

**Action needed:** Investigate host driver/kernel version change to recover ~10% decode.

## Cross-Platform Comparison — Llama-3.1-8B Q4_K_M (2026-03-12)

Apples-to-apples comparison using a model from the reference benchmark spreadsheet.
Same GPU die (BMG-G31, PCI 0xe223). B70 = 32 GB variant, B580 = 12 GB variant.

Baseline: `references/llamacpp_b7973_baseline.json` (Arc Pro B70, stock llama.cpp b7973)
Our build: `llama-sycl-local:0.14.0-b7.1-4d99d45-final` (Arc Pro B70, all optimizations)
Model: `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (4.575 GB, pure attention)

### Decode — 2.85x improvement, 78.5% of CUDA

| Metric | Stock SYCL (b7973) | **Our Optimized** | Vulkan | RTX Pro 4000 CUDA |
|---|---:|---:|---:|---:|
| tg128 (tok/s) | 30.13 | **85.90** | 49.46 | 109.48 |
| tg256 | 29.52 | **85.54** | 49.41 | 109.24 |
| tg512 | 28.85 | **85.05** | 49.06 | 108.23 |
| BW roofline % | 25.4% | **72.5%** | 41.7% | ~92% |
| vs CUDA | 3.63x slower | **1.27x slower** | 2.21x slower | — |
| vs Vulkan | 0.61x | **1.74x** | — | 2.21x |

### Prefill — Step 9: GGML_SYCL_F16=ON (2.55x improvement)

| pp Length | FP32 Build | **F16 Build** | **Improvement** | Stock b7973 | RTX 4000 CUDA | F16/CUDA |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1086.66 | **2772.71** | **2.55x** | 1417.68 | 4906.22 | **56.5%** |
| 1024 | 1157.78 | **3440.98** | **2.97x** | 1100.57 | 4954.42 | **69.5%** |
| 2048 | 1126.83 | **3244.85** | **2.88x** | 766.53 | 4862.82 | **66.7%** |
| 4096 | 1023.38 | **2574.90** | **2.52x** | 476.72 | 4697.60 | **54.8%** |

Image: `llama-sycl-local:0.14.0-b7.1-4d99d45-f16`
Change: Added `-DGGML_SYCL_F16=ON` to cmake flags.
Effect: MUL_MAT_SYCL now dequantizes to FP16 and runs FP16 GEMM via oneDNN
with `fpmath_mode::f16`, utilizing XMX FP16 engines.

Decode unchanged (86.22 tg128) — expected since decode uses MMVQ, not GEMM.
Remaining prefill gap to CUDA (~1.5-1.8x) is GEMM efficiency (cuBLAS vs oneDNN).

### Where the Decode Improvement Came From

**1. AOT compilation fix (~2x, the biggest factor)**
The stock b7973 build uses `llama-sycl-local:b7973 (built from intel/llm-scaler-vllm:1.3)`.
This likely produces generic SPIR-V that does online finalization, NOT true ahead-of-time
compiled GPU binaries for BMG-G31. Our fix:
- Auto-detect BMG-G31 PCI device ID
- Compile with `spir64_gen -device bmg-g31` via OCLOC offline compiler
- Eliminates JIT overhead and produces device-optimized ISA

**Is this an upstream issue?** Yes. The llama.cpp SYCL cmake build does not auto-detect
Battlemage target architecture. Users get generic SPIR-V by default. This is a build system
bug that should be fixed upstream (proper target detection in cmake for Intel discrete GPUs).

**2. Reorder MMVQ kernel additions (~1.4x on top of AOT)**
- Q5_K reorder MMVQ: +8.8% decode (Q5_K is 14.7% of model weights)
- Q8_0 reorder MMVQ: +2.5% decode (Q8_0 is 0.1% of weights, but many small tensors)
- Q6_K SWAR scalarization: +5.1% decode (Q6_K is 26.7% of weights)
- These move Q5_K/Q8_0 from the DMMV path (16 threads/wg) to reorder MMVQ (256 threads/wg
  with dp4a), plus optimize the Q6_K inner loop.

**3. Dequantize correctness fixes (0% perf, critical for correctness)**
- Q8_0 and Q5_K reorder-aware dequantize: ensures prefill (MUL_MAT_SYCL) correctly reads
  the SoA reordered layout. Without these, prefill produces corrupted outputs.

### Prefill Gap Analysis — Profile Results (Step 10)

**Profile setup**: Llama-3.1-8B Q4_K_M, pp512, unitrace with ComputeBasic metrics.
Image: `llama-sycl-profiler:0.14.0-b7.1-4d99d45-f16`

After GGML_SYCL_F16=ON (Step 9), the prefill gap to CUDA is 1.5-1.8x. Profile reveals:

**Inference-only device time breakdown** (excluding model load + one-time reorder):

| Category | Kernel(s) | Time (ms) | % of Inference |
|---|---|---:|---:|
| **GEMM** | `gemm_kernel` (oneDNN) | 64.5 | 33.4% |
| **Dequant Q4_K→FP16** | `dequantize_row_q4_K` | 47.0 | 24.4% |
| **Flash Attention** | `flash_attn_tile<128,128,32,1>` | 30.1 | 15.6% |
| **Convert f32→f16** | `convert_unary_nc_sycl` | 11.9 | 6.2% |
| **Dequant Q6_K→FP16** | `dequantize_row_q6_K` | 6.1 | 3.2% |
| **SwiGLU** | `ggml_sycl_op_swiglu` | 5.7 | 3.0% |
| **Flash Attn init** | `launch_fattn` | 3.1 | 1.6% |
| **Elementwise** | add, mul, rms_norm, rope, set_rows | 11.5 | 6.0% |
| **Other** | quantize_q8_1, cpy, get_rows | 13.1 | 6.8% |
| **Total inference** | | ~193 | 100% |

**Key finding: The 3-kernel MUL_MAT pipeline is the bottleneck (67% of inference).**

Per weight matrix, llama.cpp runs three separate kernels:
1. `dequantize_row_q4_K` → writes FP16 tensor to global memory
2. `gemm_kernel` (oneDNN matmul) → reads FP16, computes, writes FP32
3. `convert_unary_nc f32→f16` → converts output back to FP16

Total MUL_MAT = 129.5 ms, of which dequant+convert overhead = 65 ms (50%).
The GEMM itself (64.5 ms) has **85-88% XMX utilization** — already efficient.

**Why CUDA is faster**: cuBLAS likely fuses dequant into the GEMM prologue — reads
quantized weights, dequantizes in registers, feeds directly to tensor cores. No
intermediate FP16 tensor written to global memory. This eliminates the 65 ms overhead.

**Optimization targets (in priority order)**:

1. **Fused dequant+GEMM** (potential ~50 ms saving, ~25% improvement)
   - Integrate dequant into GEMM prologue using SYCL-TLA mixed-dtype patterns
   - Example 02 in sycl-tla: u4 dequant in MainloopXeL1Staged prologue
   - Eliminates dequant global memory round-trip AND convert_f32→f16

2. **Flash Attention** (30 ms, 15.6%)
   - Current FA uses SLM (21.5 KB per WG), 7.4 KB spill per thread
   - SYCL-TLA Example 06 has FlashAttention V2 with block-2D loads
   - Could be 2-3x more efficient

3. **SwiGLU fusion** (5.7 ms, 3%)
   - SYCL-TLA Example 07: dual GEMM for gated MLP — fuses gate+up GEMM + SwiGLU
   - Small win individually but eliminates kernel launch overhead

**XMX utilization per GEMM shape**:

| GEMM Config | XMX Util | Stall% | Occupancy |
|---|---:|---:|---:|
| SIMD16 {16;2;1}{128;4;1} | 85-88% | 47-49% | 90-95% |
| SIMD16 {32;1;1}{64;8;1} | 73-79% | 55-57% | 95-96% |
| SIMD16 {8;4;1}{64;4;2} | 38-41% | 63-70% | 84-92% |

The smaller GEMM shapes ({8;4;1}) have poor XMX utilization — these are likely the
K/V projection matrices (smaller N dimension). Fused dequant+GEMM would help here too
by hiding the dequant latency in the GEMM pipeline.

### Remaining gap math

Current: 269 ms for pp512 = 1900 tok/s (profiled) / 2773 tok/s (unprofiled)
CUDA:    104 ms for pp512 = 4906 tok/s

Unprofiled inference time ≈ 185 ms (269 ms × 2773/1900 × 193/269).
If fused dequant saves 50 ms: 135 ms → 3793 tok/s (77% of CUDA). Achievable.
If + FA improvement saves 15 ms: 120 ms → 4267 tok/s (87% of CUDA). Ambitious.

### Step 10: Fused Dequant+GEMM DPAS Kernel (Dead End)

**Problem:** The 3-kernel MUL_MAT pipeline (dequant→FP16, convert F32→FP16, oneDNN GEMM)
accounts for 67% of prefill time, with dequant+convert overhead = 50% of that pipeline.

**Attempt 1 — ESIMD DPAS kernel:**
Wrote fused kernel using ESIMD intrinsics. Compiled and ran correctly, but ESIMD doesn't
interop well with standard SYCL (separate compilation model, barriers, SLM).

**Attempt 2 — Inline vISA DPAS kernel (`dequant_gemm.cpp`):**
Rewrote using standard SYCL + inline vISA assembly for DPAS instruction:
```cpp
asm volatile (
    "dpas.hf.hf.8.8 (M1, 16) DST.0 DST.0 SRC1_UD.0 SRC2_UD(0,0)"
    : "+rw"(d) : "rw"(a), "rw"(b)
);
```
Config: TM=32, TN=64, TK=32, 16 subgroups × 16 lanes = 256 WIs/WG.
Three phases per K-step: cooperative dequant→SLM, cooperative B load→SLM, DPAS from SLM.
**Correct results, but SLOWER than baseline:**

| pp length | Baseline (dequant+oneDNN) | Fused DPAS | Delta |
|---:|---:|---:|---:|
| 32 | 108.22 | 113.69 | +5% |
| 128 | 239.69 | 233.12 | -3% |
| 512 | 347.68 | 272.26 | **-22%** |

**Root cause:** Our GEMM achieves ~50-60% XMX utilization vs oneDNN's 85-88%. The small
TK=32 tile (constrained by Q4_K 256-element super-blocks with 8×32-element sub-blocks)
causes excessive barriers and poor pipeline utilization. Element-by-element SLM access
further limits throughput. **Cannot out-GEMM oneDNN — period.**

**Lesson:** Don't replace the GEMM. Eliminate the overhead around it, or use a fundamentally
different approach (quantized integer GEMM like CUDA does).

**Status:** Kernel code retained in `dequant_gemm.cpp` for reference but dispatch DISABLED
in `ggml-sycl.cpp` (fused path removed).

### Step 11: Enable Existing MMQ Kernel + DPAS Upgrade (Attempted, Not Competitive)

**Discovery:** The SYCL backend ALREADY HAS a complete CUDA-style quantized GEMM
implementation in `mmq.cpp` (~3000 lines), ported from CUDA by DPCT. It uses `dpct::dp4a()`
for integer dot products — the same approach CUDA uses (DP4A instructions / Tensor Cores).

**How CUDA handles Q4_K MUL_MAT (the approach MMQ implements):**
1. Quantize activations to Q8_1 (INT8 + scale + sum)
2. Integer dot product: Q4_K weights × Q8_1 activations using DP4A
3. Scale correction: multiply by weight_scale × activation_scale, subtract min correction
4. **Never dequantizes to FP16.** No intermediate tensor. No separate GEMM kernel.

**What we did:**
1. Enabled `ggml_sycl_supports_mmq()` for Q4_K
2. Fixed WARP_SIZE incompatibility (overrode to 32 for MMQ, added `reqd_sub_group_size`)
3. Upgraded dp4a to `dpas.s8.s8.8.8` systolic instructions (inline vISA assembly)
4. Added adaptive NSG_N (TN=16 for batch≤16, TN=32 for larger batches)
5. Implemented sub-block pairing (load Q4_K bytes once, extract both nibbles → 2 DPAS calls)

**Optimization attempts (all failed to close the gap):**
- SLM Q8 activation caching → barriers killed performance (75% of throughput lost at pp8)
- Deferred DPAS batching (both calls before float scaling) → no measurable change
- Wider N-tiles (NSG_N=2) → helped pp32, regressed pp8

**Results (Qwen3.5-9B Q4_K_M, Arc Pro B70):**

| Test | DPAS Kernel | Baseline (oneDNN) | Delta | Path |
|------|---:|---:|---:|---|
| pp8 | 25.97 | ~26 | 0% | MMVQ (not DPAS) |
| pp16 | 55.96 | ~70* | -20% | DPAS |
| pp32 | 98.33 | 108.22 | **-9%** | DPAS |
| pp33 | 108.94 | ~108 | 0% | oneDNN (batch>32) |
| tg128 | 53.09 | 58.36 | **-9%** | MMVQ (not DPAS) |

*pp16 baseline estimated. tg128 regression later confirmed to be environmental (system-level),
not from WARP_SIZE override — verified by running unmodified Step 7 build (same 52.46 tok/s).

**Results (Llama-3.1-8B Q4_K_M, Arc Pro B70, from previous session):**

| Test | DPAS v3 (adaptive) | Baseline (oneDNN) | Delta |
|------|---:|---:|---:|
| pp8 | 72 | — | — |
| pp16 | 185 | — | — |
| pp32 | 263 | 377 | **-30%** |
| pp33 | 376 | ~377 | 0% |
| tg128 | 80.8 | 85.9 | **-6%** |

**Root cause analysis:** The DPAS kernel cannot match the dequant+oneDNN pipeline because:
1. **Non-coalesced weight loads**: Each lane loads from different Q4_K blocks (stride=144B per block × rows)
2. **Float scale overhead**: 8 sub-blocks × per-sub-block scale application between DPAS calls
3. **oneDNN is too good**: 85-88% XMX utilization with optimized memory access patterns

**Decision: DPAS kernel should be DISABLED.** The dequant+oneDNN pipeline remains faster
for all batch sizes. The DPAS code is retained in mmq.cpp for reference but the MMQ dispatch
(`ggml_sycl_supports_mmq()`) should be reverted to return false.

**Lesson learned:** On Xe2, oneDNN's GEMM engine is extremely well-optimized. Neither a
custom FP16 fused kernel (Step 10) nor a custom INT8 quantized kernel (Step 11) can beat
dequant+oneDNN. The dequant overhead (50% of MUL_MAT time) remains the primary gap to CUDA,
but closing it requires oneDNN-level GEMM quality — which we cannot replicate in a custom kernel.

### Code Integration Points for Fused Dequant+GEMM (archived)

**The prefill MUL_MAT call chain** (batch > 1, use_fp16=true):

```
ggml_sycl_mul_mat()                              ggml-sycl.cpp:3618
  ├─ use_mul_mat_q = ggml_sycl_supports_mmq()    ggml-sycl.cpp:3646  ← MMQ PATH
  │   └─ ggml_sycl_op_mul_mat_q()                mmq.cpp:2962        (quantized GEMM)
  └─ else (fallback)
      └─ ggml_sycl_op_mul_mat_sycl()             ggml-sycl.cpp:2126  (dequant+oneDNN)
          ├─ to_fp16_sycl(src0_dd_i, ...)         convert.cpp:625  (dequant Q4_K→FP16)
          ├─ to_fp16_sycl(src1_ddf_i, ...)        convert.cpp:625  (convert F32→FP16)
          ├─ DnnlGemmWrapper::row_gemm()          gemm.hpp         (oneDNN GEMM)
          └─ to_fp32_sycl(dst_f16, ...)           convert.cpp:700  (convert FP16→F32)
```

## Benchmark Environment

- Container: `intel/llm-scaler-vllm:0.14.0-b7.1`
- Build: `llama-sycl-local:0.14.0-b7.1-4d99d45-aot-bmg-g31`
- Benchmark: `llama-bench` with `b=2048`, `ub=2048`, `fa=1`, `ngl=99`, `sm=none`
- All benchmarks use `r=5` repetitions unless noted as screening runs
- Reference baseline: `references/llamacpp_b7973_baseline.json` (Arc Pro B70, stock b7973)
