# aicss-genai-fork — llama.cpp SYCL Optimization for Intel Arc GPUs

A downstream llama.cpp fork that ports the `llamacpp-sycl-bmg` SYCL
optimizations directly on top of upstream commit `45cac7ca7`. The SYCL
kernel patches are already applied as individual commits on this branch —
you can build and run the fork directly.

This work was produced with an agentic engineering approach: agents
surface issues and explore experiments while engineers identify and
reject candidates using domain knowledge. The agent harness is called
**Agentic Auto Optimizer**.

Everything runs inside Docker containers using Intel's oneAPI toolchain,
so no host-side SYCL install is required.

**Current best result:** 2–7x faster than stock SYCL across 8 models,
reaching 76–86% of CUDA (RTX PRO 4000) decode throughput on Q4_K_M
models ≥8B params. See [docs/performance-ledger.md](docs/performance-ledger.md)
for the full comparison table.

## Hardware

| Component | Spec |
|-----------|------|
| GPU | Intel Arc Pro B70 (Xe2-HPG, BMG-G31, 256 EUs, 32 GB GDDR6) |
| Peak BW | 542.3 GB/s |
| CPU | Intel Xeon w9-3475X |
| PCIe | Gen5 x16 |

## Fork layout

This fork adds an `aicss-genai/` directory at the root of the llama.cpp
tree. Everything specific to this downstream fork lives there; the rest
of the repository is unmodified upstream llama.cpp plus the 12 applied
SYCL patches.

```
llama.cpp/                             # upstream llama.cpp + applied SYCL patches
└── aicss-genai/
    ├── README.md                      # this file
    ├── patches/
    │   ├── 0001..0012-*.patch         # individual SYCL patches (also applied as commits)
    │   ├── apply.sh                   # apply patches to a fresh upstream checkout
    │   └── README.md                  # per-patch summary and impact table
    ├── scripts/                       # all automation
    │   ├── setup_third_party.sh              # Clone llama.cpp + pti-gpu (+ apply patches unless baseline)
    │   ├── build_llama_sycl_container.sh     # Build llama.cpp SYCL inside Intel container
    │   ├── commit_llama_sycl_container.sh    # Package build into a Docker image
    │   ├── build_unitrace.sh                 # Build the PTI unitrace profiler
    │   ├── build_llama_sycl_profiler_image.sh # Create profiler image (llama + unitrace)
    │   ├── run_qwen9b_sycl_bench.sh          # Run benchmark sweeps (pp/tg matrix) for a single model
    │   ├── bench_all_models_parallel_gpu.sh  # Parallel multi-GPU sweep for all manifest models
    │   ├── result_parser.py                  # Parse JSON results → text/md/csv reports
    │   ├── compare_configs.py                # Compare two benchmark CSVs and plot
    │   ├── merge_results_dirs.py             # Merge multiple result directories
    │   ├── plot_ab_results.py                # Plot A/B comparison results
    │   ├── profile_llama_sycl_decode.sh      # Profile with unitrace (summary or chrome)
    │   ├── ab_l0_submission_overhead.sh      # A/B test Level Zero runtime tuning
    │   ├── summarize_unitrace_trace.py       # Extract GPU/CPU hotspots from traces
    │   ├── download_bench_models.sh          # Download all benchmark models from HF
    │   ├── verify_bench_models.py            # Validate model file sizes vs manifest
    │   ├── bench_models_manifest.tsv         # 14 models with HF URLs and expected sizes
    │   └── bench_models_manifest_custom.tsv  # Custom subset manifest for quick runs
    └── docs/
        ├── profiling.md                      # Profiling runbook with worked examples
        ├── performance-ledger.md             # Canonical table of every optimization experiment
        ├── optimization-workbook.md          # Roofline analysis, per-token time budgets
        └── benchmark-results-2026-04-10.md   # JIT / AOT+F16 / Optimized comparison across 6 models
```

The scripts use a convention-driven workspace: they expect a sibling
`third_party/llama.cpp` checkout (created by `setup_third_party.sh`) so
that the Docker builds can be rerun in isolation from the fork source
tree. If you are only interested in the fork source itself, you can
ignore the scripts and build the repository root directly with cmake.

### Why `pti-gpu`?

Intel's [PTI-GPU](https://github.com/intel/pti-gpu) provides `unitrace`,
the tool that gives per-kernel GPU timing and hardware metrics on Intel
GPUs. It is how we identify which MMVQ/DMMV kernels are hot and whether
changes actually improve bandwidth utilization. It outputs chrome-format
JSON that can be loaded in `chrome://tracing` or analyzed with
`summarize_unitrace_trace.py`.

## Prerequisites

- Docker with `--device /dev/dri` access (Intel GPU passthrough)
- `xpu-smi` installed on the host (for device arch auto-detection)
- Intel oneAPI container image: `intel/llm-scaler-vllm:0.14.0-b8.1` —
  provides `icx`/`icpx` compilers, oneAPI runtime, and oneDNN. Pulled
  automatically on first build.
- Model files in `models/` (see [Downloading Models](#3-downloading-models))

## Quick Start

### 0. Setup

Clone llama.cpp and pti-gpu at the pinned commits. The scripts live in
`aicss-genai/scripts/` — run them from the repository root:

```bash
sudo rm -r ./aicss-genai/third_party
```

```bash
# Optimized: clones + applies SYCL kernel patches
LLAMA_BENCH_CONFIG=optimized ./aicss-genai/scripts/setup_third_party.sh

# Baseline: clones only, no patches applied
LLAMA_BENCH_CONFIG=baseline ./aicss-genai/scripts/setup_third_party.sh
```

This creates `third_party/llama.cpp` (@ `45cac7ca7`) and
`third_party/pti-gpu` (@ `044440c`). When `LLAMA_BENCH_CONFIG=baseline`,
patches are skipped so the source tree stays identical to upstream. If
the repos already exist, the script skips the clone and warns on commit
mismatch.

Tip: the current fork branch itself already has the patches applied on
top of `45cac7ca7`. The `third_party/` workflow is intended for running
A/B comparisons against an unpatched tree without affecting the fork
checkout.

### 1. Build

Build llama.cpp with SYCL. The build flow depends on `LLAMA_BENCH_CONFIG`:

| Config | How it builds | Output |
|---|---|---|
| `optimized` (default) | Custom cmake inside `intel/llm-scaler-vllm:0.14.0-b8.1` with patches + AOT/F16/DNN/Graph flags | Host dir: `third_party/llama.cpp/build-sycl-<image-tag>/bin/` |
| `baseline` | Custom cmake inside the same Intel container — **no patches, minimal flags** (`GGML_SYCL=ON`, `GGML_SYCL_TARGET=INTEL`) | Host dir: `third_party/llama.cpp/build-sycl-baseline/bin/` |

```bash
# Optimized build (default)
LLAMA_BENCH_CONFIG=optimized ./aicss-genai/scripts/build_llama_sycl_container.sh

# Baseline build — clean upstream source, minimal SYCL flags
LLAMA_BENCH_CONFIG=baseline ./aicss-genai/scripts/build_llama_sycl_container.sh
```

The **baseline** build reverts any local patches (`git checkout -- .`)
and builds with only `-DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL` — no AOT,
no F16, no DNN, no graph support. Both configs produce host-side build
directories under `third_party/llama.cpp/`.

Key cmake flags set by the **optimized** build:
- `-DGGML_SYCL_DEVICE_ARCH=bmg-g31` — AOT offline compilation (eliminates JIT)
- `-DGGML_SYCL_F16=ON` — FP16 accumulation
- `-DGGML_SYCL_DNN=ON` — oneDNN GEMM kernels
- `-DGGML_SYCL_GRAPH=ON` — kernel launch batching

### 2. Package (optional)

Commit a build into a self-contained Docker image with wrapper scripts.
This is optional — the benchmark scripts run directly from the host-side
build directories and don't require a packaged image.

```bash
LLAMA_BENCH_CONFIG=optimized ./aicss-genai/scripts/commit_llama_sycl_container.sh
```
```bash
LLAMA_BENCH_CONFIG=baseline ./aicss-genai/scripts/commit_llama_sycl_container.sh
```

Creates `llama-sycl-local:optimized-<tag>-<commit>` with `llama-bench`,
`llama-cli`, `llama-server` etc. available on `$PATH`.

### 3. Downloading Models

Download all 14 benchmark models from Hugging Face:

```bash
LLAMA_BENCH_MANIFEST=aicss-genai/scripts/bench_models_manifest.tsv ./aicss-genai/scripts/download_bench_models.sh
```

Models are defined in `aicss-genai/scripts/bench_models_manifest.tsv`.
Downloads are resumable and verified against expected file sizes. To
check integrity:

```bash
python aicss-genai/scripts/verify_bench_models.py
```

### 4. Benchmark

### 4a. Single-Model Benchmark (qwen3.5-9b used as the default)

Run a full benchmark sweep across prompt and generation token counts:

```bash
./aicss-genai/scripts/run_qwen9b_sycl_bench.sh [model_path]
```

Defaults to `models/Qwen3.5-9B-Q4_K_M.gguf`. Runs warmup, then scored
runs with `pp512-8192 × tg128-1024 × r=5`. Results land in
`results/<date-time>/`.

Everything is configurable via env vars — see the script header for the
full list (`LLAMA_BENCH_PROMPT_TOKENS`, `LLAMA_BENCH_GEN_TOKENS`,
`LLAMA_BENCH_REPS`, etc.).

### 4b. Full Multi-Model Benchmark (All Models in Manifest)

Run llama-bench across every model in
`aicss-genai/scripts/bench_models_manifest.tsv`, automatically
parallelized across all detected Intel GPUs:

```bash
./aicss-genai/scripts/bench_all_models_parallel_gpu.sh
```

GPU count is auto-detected via `xpu-smi discovery`. To pin specific GPUs:

```bash
LLAMA_BENCH_GPU_IDS=0,1 ./aicss-genai/scripts/bench_all_models_parallel_gpu.sh
```

To benchmark a subset of models, create a custom manifest TSV (same
format as `bench_models_manifest.tsv`) and point `LLAMA_BENCH_MANIFEST`
at it:

```bash
# Example: benchmark only Llama-3.1-8B-Q8 and Qwen3.5-9B-Q4
LLAMA_BENCH_GPU_IDS=0,1 \
LLAMA_BENCH_CONFIG=baseline \
LLAMA_BENCH_MANIFEST=aicss-genai/scripts/bench_models_manifest_custom.tsv \
    ./aicss-genai/scripts/bench_all_models_parallel_gpu.sh
```

Results land in `results/<YYYY-MM-DD-HHMMSS>/` with one subdirectory per
model (`<ModelName>-<image-tag>/`), each containing:

- `*-single-stream.json` — llama-bench JSON output (used by `result_parser.py`)
- `*-warmup.md` — warmup run in Markdown table format
- `*-devices.txt` — GPU device list from `llama-bench --list-devices`
- `*-docker.log` — full container stdout/stderr (error source for skipped models)
- `*-run.txt` — run metadata (image, flags, GPU ID, all env vars)

A comparison CSV is automatically generated at
`results/<YYYY-MM-DD-HHMMSS>/<YYYY-MM-DD-HHMMSS>.csv` when all models
finish.

**Benchmark sweep defaults** (override via env vars):

| Variable | Default | Description |
|---|---|---|
| `LLAMA_BENCH_CONFIG` | `optimized` | Build config to benchmark: `optimized` or `baseline` |
| `LLAMA_BENCH_MANIFEST` | `aicss-genai/scripts/bench_models_manifest.tsv` | Custom manifest TSV for model subset |
| `LLAMA_BENCH_GPU_IDS` | auto-detect | Comma-separated GPU IDs (e.g. `0,1`) |
| `LLAMA_BENCH_RESULTS_DIR` | `results/<date-time>` | Override output directory |
| `LLAMA_BENCH_THREADS` | `8` | CPU threads per container |
| `LLAMA_BENCH_PROMPT_TOKENS` | `512,1024,2048,4096,8192` | Prefill token sizes |
| `LLAMA_BENCH_GEN_TOKENS` | `128,256,512,1024` | Decode token sizes |
| `LLAMA_BENCH_REPS` | `5` | Repetitions per measurement |
| `LLAMA_BENCH_N_BATCH` | `2048` | Logical batch size (`-b`) |
| `LLAMA_BENCH_N_UBATCH` | `2048` | Physical micro-batch size (`-ub`) |
| `LLAMA_BENCH_FLASH_ATTN` | `1` | Flash attention (1=on, 0=off) |
| `LLAMA_BENCH_SPLIT_MODE` | `none` | Tensor split mode (`none`/`row`/`layer`) |
| `LLAMA_BENCH_MAIN_GPU` | `0` | Primary GPU index for multi-GPU split |
| `LLAMA_BENCH_WARMUP_PP` | `512` | Warmup prefill token count |
| `LLAMA_BENCH_WARMUP_TG` | `128` | Warmup decode token count |
| `LLAMA_BENCH_WARMUP_REPS` | `1` | Warmup repetitions |
| `LLAMA_BENCH_RUN_BATCHED` | `0` | Run `llama-batched-bench` after single-stream (1=on) |
| `LLAMA_BENCH_BATCHED_NPL` | `1` | Number of parallel sequences for batched bench |

**SYCL / Level Zero tuning vars** (forwarded into each container when set):

| Variable | Effect |
|---|---|
| `GGML_SYCL_DISABLE_GRAPH` | Disable SYCL kernel graph batching |
| `GGML_SYCL_DISABLE_OPT` | Disable SYCL kernel optimizations |
| `GGML_SYCL_PRIORITIZE_DMMV` | Force DMMV over reorder MMVQ path |
| `GGML_SYCL_USE_ASYNC_MEM_OP` | Enable async memory operations |
| `GGML_SYCL_DEBUG` | Enable SYCL debug output |
| `UR_L0_USE_IMMEDIATE_COMMANDLISTS` | Use Level Zero immediate command lists |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` | Alias for the above (older runtime) |
| `UR_L0_DEVICE_SCOPE_EVENTS` | Use device-scope event timing |
| `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS` | Alias for the above (older runtime) |
| `UR_L0_BATCH_SIZE` | Level Zero command batch size |
| `SYCL_PI_LEVEL_ZERO_BATCH_SIZE` | Alias for the above (older runtime) |

Example — quick 3-rep run on 2 GPUs with immediate command lists enabled:

```bash
LLAMA_BENCH_GPU_IDS=0,1 \
LLAMA_BENCH_REPS=3 \
UR_L0_USE_IMMEDIATE_COMMANDLISTS=1 \
    ./aicss-genai/scripts/bench_all_models_parallel_gpu.sh
```

### 4c. Parse Results

Parse a timestamped results directory into a report. The bench script
auto-generates a CSV on completion; use `result_parser.py` directly for
re-runs or alternate formats.

```bash
# Pretty-printed text to stdout — auto-picks the most recent results/ folder
python3 aicss-genai/scripts/result_parser.py

# Target a specific timestamped directory
python3 aicss-genai/scripts/result_parser.py results/2026-03-23-143052

# Markdown report (format inferred from .md extension)
python3 aicss-genai/scripts/result_parser.py results/2026-03-23-143052 --output results/2026-03-23-143052/report.md

# Comparison CSV (format inferred from .csv extension)
python3 aicss-genai/scripts/result_parser.py results/2026-03-23-143052 --output results/2026-03-23-143052/baseline.csv

# Aggregate summary only — skip per-model tables
python3 aicss-genai/scripts/result_parser.py results/2026-03-23-143052 --no-per-model

# Explicit format flag (overrides file extension inference)
python3 aicss-genai/scripts/result_parser.py results/2026-03-23-143052 --format csv --output baseline.csv
```

**Output formats:**

| Format | Flag | Extension inferred | Contents |
|---|---|---|---|
| `text` | `--format text` | (stdout default) | Plain-text pp/tg throughput tables per model + aggregate |
| `md` | `--format md` | `.md` | Markdown tables, suitable for GitHub or docs |
| `csv` | `--format csv` | `.csv` | 24-column comparison spreadsheet (see below) |

**CSV column layout** (`--format csv`):

| Group | Columns |
|---|---|
| Configuration | Model, Params (B), Quant, Arch, Task Type, Tokens |
| Throughput | SYCL (tok/s), SYCL StdDev, Vulkan (tok/s)\*, Vulkan StdDev\*, CUDA (tok/s)\*, CUDA StdDev\* |
| Speed Ratios | CUDA/SYCL\*, CUDA/Vulkan\*, SYCL/Vulkan\* |
| VRAM | SYCL VRAM, CUDA VRAM\* |
| Analysis | Fastest Backend, Tier (small/medium/large) |
| E2E Latency | SYCL E2E (ms), Vulkan E2E (ms)\*, CUDA E2E (ms)\*, CUDA/SYCL Ratio\* |
| Note | Error reason for skipped/failed models |

\* Columns left empty for future cross-backend benchmark fills.

Row ordering: all prefill (`pp`) rows first across all models, then all
decode (`tg`) rows. Skipped/errored models are appended at the bottom
with their error reason (extracted from `*-docker.log`) in the Note
column.

### 5. Profile

Build the profiler toolchain (one-time):

```bash
./aicss-genai/scripts/build_unitrace.sh
./aicss-genai/scripts/build_llama_sycl_profiler_image.sh
```

Run a profiled decode:

```bash
# Summary mode (host + device timing tables)
./aicss-genai/scripts/profile_llama_sycl_decode.sh [model_path]

# Chrome timeline mode (produces JSON for chrome://tracing)
LLAMA_PROFILE_TRACE_MODE=chrome ./aicss-genai/scripts/profile_llama_sycl_decode.sh [model_path]
```

Profile artifacts land in `results/profiles/<timestamp>/`. The script
collects `ComputeBasic` hardware metrics by default, captures `xpu-smi`
topology/stats snapshots, and handles the `observation_paranoid` sysctl
automatically.

### 6. Analyze Traces

Extract top GPU and CPU hotspots from a chrome-format unitrace JSON:

```bash
# All events
python aicss-genai/scripts/summarize_unitrace_trace.py results/profiles/<run>/unitrace/<file>.json

# GPU-only, tail 90% of trace (skip startup)
python aicss-genai/scripts/summarize_unitrace_trace.py results/profiles/<run>/unitrace/<file>.json gpu_op 25 --start-pct 10

# CPU-only
python aicss-genai/scripts/summarize_unitrace_trace.py results/profiles/<run>/unitrace/<file>.json cpu_op
```

### 7. A/B Testing

Test Level Zero runtime tuning knobs (immediate command lists,
device-scope events, batch sizes) against the baseline:

```bash
./aicss-genai/scripts/ab_l0_submission_overhead.sh
```

Results go to `results/<date-time>-l0-ab/` with a `summary.txt`
comparing all variants.

## Key Optimizations

Two layers of optimization, both required for the full speedup.

### Build flags (cmake)

Set automatically by `build_llama_sycl_container.sh`:

| Flag | What it does | Impact |
|---|---|---|
| `-DGGML_SYCL_DEVICE_ARCH=bmg-g31` | AOT offline compilation for Xe2 (eliminates JIT) | 50x decode from broken baseline |
| `-DGGML_SYCL_F16=ON` | FP16 accumulation (halves BW, native XMX half-precision) | Required for competitive prefill |
| `-DGGML_SYCL_DNN=ON` | oneDNN GEMM kernels (~85% systolic utilization) | Prefill throughput |
| `-DGGML_SYCL_GRAPH=ON` | Kernel launch batching (reduces dispatch overhead) | Latency reduction |

### Kernel patches (source)

Applied to `ggml/src/ggml-sycl/` on top of commit `45cac7ca7`. On this
branch the patches are already applied as individual commits; the
verbatim patch files are preserved in `aicss-genai/patches/` for
reference and for downstream consumers who want to apply them to an
unpatched upstream checkout.

**Patch 0001 — Battlemage kernel optimizations**
(`0001-sycl-battlemage-optimizations.patch`):

| File | Change | Impact |
|---|---|---|
| `CMakeLists.txt` | Fixed AOT to use `spir64_gen` target with proper device flag | AOT actually works |
| `vecdotq.hpp` | Q6_K SWAR byte-subtract scalarization for Xe2 SIMD16 | +5.1% decode |
| `mmvq.cpp` | Q5_K and Q8_0 reorder MMVQ dispatch (256-thread dp4a) | +8.8% / +2.5% decode |
| `quants.hpp` | Reorder-aware quant type traits for Q5_K/Q8_0 | Enables MMVQ path |
| `ggml-sycl.cpp` | `supports_reorder` dispatch for Q5_K/Q8_0, async mem-op plumbing | Enables MMVQ path |
| `dequantize.hpp` | Reorder-aware dequantize kernels for Q5_K/Q8_0 | Fixes prefill regression |
| `convert.cpp` | Dequantize kernel launchers for reordered layouts | Fixes prefill regression |
| `mmq.cpp` | Minor matrix-multiply quant adjustments | — |

**Patches 0002–0012** — bugfixes, correctness fixes, kernel-level
optimizations, and graph fusion. Full per-patch impact table and
per-file summary in [aicss-genai/patches/README.md](patches/README.md).

To apply manually to a separate llama.cpp checkout:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 45cac7ca7
bash /path/to/aicss-genai/patches/apply.sh .
```

See [aicss-genai/patches/README.md](patches/README.md) for build
instructions after applying, and [docs/performance-ledger.md](docs/performance-ledger.md)
for every experiment with exact artifact references.

## Environment Variables

The scripts are configured entirely through env vars. Key ones:

| Variable | Used By | Purpose |
|---|---|---|
| `LLAMA_BENCH_CONFIG` | setup, build, bench | Build config: `optimized` (default) or `baseline` (minimal SYCL flags, no patches) |
| `LLAMA_BENCH_MANIFEST` | bench-parallel | Custom manifest TSV for model subset (default: `aicss-genai/scripts/bench_models_manifest.tsv`) |
| `LLAMA_SYCL_IMAGE` | build, bench (optimized only) | Base Intel container image (default: `intel/llm-scaler-vllm:0.14.0-b8.1`) |
| `LLAMA_SYCL_DEVICE_ARCH` | build | Override GPU arch (default: auto-detect via `xpu-smi`) |
| `LLAMA_BENCH_GPU_IDS` | bench-parallel | Comma-separated GPU IDs to use (default: auto-detect) |
| `LLAMA_BENCH_RESULTS_DIR` | bench-parallel | Override dated output directory |
| `LLAMA_BENCH_THREADS` | bench | CPU threads per container (default: `8`) |
| `LLAMA_BENCH_PROMPT_TOKENS` | bench | Comma-separated prefill token counts (default: `512,1024,2048,4096,8192`) |
| `LLAMA_BENCH_GEN_TOKENS` | bench | Comma-separated decode token counts (default: `128,256,512,1024`) |
| `LLAMA_BENCH_REPS` | bench | Scoring repetitions (default: `5`) |
| `LLAMA_BENCH_N_BATCH` | bench | Logical batch size (default: `2048`) |
| `LLAMA_BENCH_N_UBATCH` | bench | Physical micro-batch size (default: `2048`) |
| `LLAMA_BENCH_FLASH_ATTN` | bench | Flash attention toggle (default: `1`) |
| `LLAMA_BENCH_SPLIT_MODE` | bench | Tensor split mode: `none`/`row`/`layer` (default: `none`) |
| `LLAMA_BENCH_MAIN_GPU` | bench | Primary GPU index for multi-GPU split (default: `0`) |
| `LLAMA_BENCH_WARMUP_PP` | bench | Warmup prefill tokens (default: `512`) |
| `LLAMA_BENCH_WARMUP_TG` | bench | Warmup decode tokens (default: `128`) |
| `LLAMA_BENCH_WARMUP_REPS` | bench | Warmup repetitions (default: `1`) |
| `LLAMA_BENCH_RUN_BATCHED` | bench | Run `llama-batched-bench` after single-stream (default: `0`) |
| `LLAMA_BENCH_BATCHED_NPL` | bench | Number of parallel sequences for batched bench (default: `1`) |
| `LLAMA_PROFILE_TRACE_MODE` | profile | `summary` or `chrome` |
| `LLAMA_PROFILE_METRIC_GROUP` | profile | Hardware metric group (default: `ComputeBasic`) |
| `GGML_SYCL_DISABLE_GRAPH` | bench, profile | Disable SYCL kernel graph batching |
| `GGML_SYCL_DISABLE_OPT` | bench, profile | Disable SYCL kernel optimizations |
| `GGML_SYCL_PRIORITIZE_DMMV` | bench, profile | Force DMMV over reorder MMVQ path |
| `GGML_SYCL_USE_ASYNC_MEM_OP` | bench, profile | Enable async memory operations |
| `GGML_SYCL_DEBUG` | bench, profile | Enable SYCL debug output |
| `UR_L0_USE_IMMEDIATE_COMMANDLISTS` | bench, profile | Use Level Zero immediate command lists |
| `UR_L0_DEVICE_SCOPE_EVENTS` | bench, profile | Use device-scope event timing |
| `UR_L0_BATCH_SIZE` | bench, profile | Level Zero command batch size |

## License

This fork carries the same MIT license as upstream llama.cpp. The SYCL
patches in `aicss-genai/patches/` are contributed under the same terms.
See the top-level [LICENSE](../LICENSE) file.
