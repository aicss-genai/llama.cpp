# SYCL Kernel Patches for Intel Arc Pro (Battlemage)

Base commit: llama.cpp `45cac7ca7`
Target: Intel Arc Pro B70 (Xe2-HPG, BMG-G31, 256 EUs, SIMD16)
Container: `intel/llm-scaler-vllm:0.14.0-b8.1` (oneAPI 2025.2.2)

On the `aicss-genai-fork` branch these patches are already applied as
individual commits on top of `45cac7ca7` — you can build the branch
directly without running `apply.sh`. The patch files in this directory
are kept for reference and for users who want to apply them to a fresh
upstream checkout.

## Quick start

Applying to a fresh upstream llama.cpp checkout:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 45cac7ca7
bash /path/to/aicss-genai/patches/apply.sh .

cmake -B build \
    -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON \
    -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_DEVICE_ARCH=bmg-g31 \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j$(nproc)
```

## Patch series (apply in order)

| # | Patch | Type | Impact (vs AOT+F16 baseline) |
|---|-------|------|------------------------------|
| 0001 | BMG AOT + reorder dequant + async | Bugfix | Required to build on BMG |
| 0002 | Remove dpct capability checks | Cleanup | Code quality |
| 0003 | PAD non-contiguous stride | Bugfix | 2.68x prefill on Qwen3.5-9B |
| 0004 | Relax SUM/MEAN contiguous | Bugfix | Correctness |
| 0005 | RMS_NORM+MUL fusion | Optimization | 1.06x decode |
| 0006 | 6 new ops + oneMKL TRSM | Feature | 1.20x decode, 1.20x prefill |
| 0007 | Barrier local_space fence | Bugfix | Correctness |
| 0008 | Native SYCL shuffles | Cleanup | Code quality |
| 0009 | Scratchpad pool + PVC removal | Optimization | Faster load time |
| 0010 | Small matmul oneMKL direct | Optimization | 1.20x prefill |
| 0011 | Fuse UNARY+MUL | Optimization | 1.01x decode |
| 0012 | RMS_NORM+MUL+ADD 3-op fusion | Optimization | 1.005x decode |

## Requirements

- oneAPI 2025.2+ (icpx compiler, oneDNN, oneMKL)
- Intel Arc Pro dGPU with Xe2 architecture (tested on Arc Pro B70)
- Level Zero runtime

## Build and test

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 45cac7ca7
bash /path/to/aicss-genai/patches/apply.sh .

cmake -B build \
    -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON \
    -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_DEVICE_ARCH=bmg-g31 \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j$(nproc)

# correctness check
build/bin/llama-cli -m model.gguf -c 512 -ngl 99 \
    -p "What is 2+2? Answer in one word." -n 4 -fa 1 \
    --temp 0 --single-turn --simple-io --no-display-prompt

# benchmark
build/bin/llama-bench -m model.gguf -r 3 -ngl 99 -fa 1 -o md
```

## Performance

Measured on 6 models, Arc Pro B70. Kernel patch gains on top of AOT+F16 build:

| Model | Prefill (pp512) | Decode (tg128) |
|-------|-----------------|----------------|
| Qwen3.5-9B | 2.68x | 1.43x |
| Gemma-2-9B | 1.28x | 1.18x |
| Llama-3.2-3B | 1.27x | 1.17x |
| Llama-3.1-8B | 1.21x | 1.10x |
| Mistral-Nemo-12B | 1.18x | 1.13x |
| Qwen2.5-14B | 1.18x | 1.13x |

Build flags alone (AOT + F16, no patches) give 2-2.5x prefill over JIT stock.
Decode is memory-bound and unaffected by build flags.

Full numbers with absolute tok/s values in `../docs/benchmark-results-2026-04-10.md`.
