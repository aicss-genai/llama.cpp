# Comprehensive Benchmark Results

Date: 2026-04-10
Hardware: Intel Arc Pro B70 (Battlemage, Xe2-HPG, 256 EUs, 32 GB GDDR6)
Container: intel/llm-scaler-vllm:0.14.0-b7.1 (oneAPI 2025.2.2)
Base commit: 4d99d45

## Three build configurations compared

1. **JIT stock**: `cmake -DGGML_SYCL=ON -DGGML_NATIVE=OFF` (what a user gets out of the box)
2. **AOT+F16**: `cmake -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DEVICE_ARCH=bmg-g31` (recommended build flags, no code patches)
3. **Optimized**: AOT+F16 + all 14 kernel patches (bugfixes + optimizations + WARP16)

Note: JIT stock cannot build at commit 4d99d45 without patch 0001 (AOT build fix).
The JIT numbers below are from a clean tree that happens to build with JIT mode.
AOT+F16 is the fair baseline for measuring kernel patch impact.

## Llama-3.1-8B Q4_K_M — Context Scaling (prefill)

| Context | JIT stock | AOT+F16 | Optimized | JIT→Opt | AOT→Opt |
|---------|-----------|---------|-----------|---------|---------|
| pp512 | 1112 | 2810 | 3402 | 3.06x | 1.21x |
| pp2048 | 1020 | 2392 | 2843 | 2.79x | 1.19x |
| pp4096 | 932 | 1960 | 2331 | 2.50x | 1.19x |
| pp8192 | 801 | 1462 | 1709 | 2.13x | 1.17x |

## All Models — pp512 + tg128

| Model | JIT pp512 | AOT pp512 | Opt pp512 | AOT→Opt pp | JIT tg128 | AOT tg128 | Opt tg128 | AOT→Opt tg |
|-------|-----------|-----------|-----------|------------|-----------|-----------|-----------|------------|
| Llama-3.2-3B | 2541 | 5586 | 7118 | 1.27x | 141 | 141 | 165 | 1.17x |
| Llama-3.1-8B | 1082 | 2813 | 3400 | 1.21x | 80 | 80 | 88 | 1.10x |
| Qwen3.5-9B | 290 | 353 | 947 | 2.68x | 50 | 50 | 72 | 1.43x |
| Gemma-2-9B | 867 | 2089 | 2680 | 1.28x | 58 | 58 | 68 | 1.18x |
| Mistral-Nemo-12B | 704 | 1843 | 2180 | 1.18x | 53 | 54 | 60 | 1.13x |
| Qwen2.5-14B | 572 | 1462 | 1728 | 1.18x | 43 | 43 | 48 | 1.13x |

## Where the gains come from (honest breakdown)

### Build configuration (JIT → AOT+F16, no code changes)
- AOT compilation: eliminates JIT overhead on first run, enables device-specific optimization
- F16 accumulation: uses half2 vectorized dot products instead of FP32 scalar
- Impact: 1.5-2.5x prefill, ~0% decode (decode is memory-bound, not compute-bound)

### Kernel patches (AOT+F16 → Optimized, code changes)
- Bugfixes (0001-0004, 0007): make the build work, fix correctness
- PAD stride fix (0003): eliminates 360+ CPU fallbacks per forward on Qwen3.5-9B → 2.68x prefill
- New ops (0006): 6 missing SYCL ops ported, eliminates more CPU fallbacks → +20% decode
- Kernel fusion (0005, 0011, 0012): RMSNorm+MUL, UNARY+MUL → +5.5% decode
- oneMKL small matmul (0010): direct dispatch for small GEMMs → +20% prefill
- WARP16 FA tile (0014): native SIMD16 sub-group → +3% prefill, +2% decode
- MMVQ SWAR (in 0001): Q6_K +5.1%, Q5_K +8.8%, Q8_0 +2.5% decode

### What does NOT contribute to these numbers
- XMX flash attention (0013): experimental, env-gated, not active in these benchmarks
- SYCL Graph replay: tested, +0.9%, not included in patches

## Decode improvement is real but smaller

AOT+F16 gives ~0% decode improvement (decode is memory-bound).
Our kernel patches give 10-43% decode improvement depending on model.
The decode gains come from:
- Kernel fusion (fewer kernel launches, less dispatch overhead)
- New ops (eliminate CPU-GPU transfers)
- MMVQ reorder (better memory access patterns in dequant kernels)

## Caveat

These numbers were collected while the host CPU was under partial load from
other work. The absolute numbers may be 5-10% lower than peak. The relative
comparisons (ratios) are valid because all three configs ran back-to-back on
the same machine in the same session.
