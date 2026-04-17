# SYCL Profiling Runbook

## Goal

Reproduce the `llama.cpp` SYCL profiling workflow for Intel Arc/BMG on this machine from a clean session.

This runbook assumes the workspace root is:

- `$REPO_ROOT`

## Prerequisites

- Docker access through `sudo`
- `/dev/dri` passed into containers
- local model inside the workspace, for example:
  - `models/Qwen3.5-9B-Q4_K_M.gguf`
- local `llama.cpp` checkout at:
  - `third_party/llama.cpp`
- local PTI checkout at:
  - `third_party/pti-gpu`

## Images

Current committed images:

- runtime image: `llama-sycl-local:0.14.0-b7.1-4d99d45`
- profiler image: `llama-sycl-profiler:0.14.0-b7.1-4d99d45`
- AOT runtime image: `llama-sycl-local:0.14.0-b7.1-4d99d45-aot-bmg-g31`
- AOT profiler image: `llama-sycl-profiler:0.14.0-b7.1-4d99d45-aot-bmg-g31`
- validated `spir64_gen` AOT runtime image: `llama-sycl-local:0.14.0-b7.1-4d99d45-aot-real-bmg-g31`
- validated `spir64_gen` AOT profiler image: `llama-sycl-profiler:0.14.0-b7.1-4d99d45-aot-real-bmg-g31`

If these images are missing, rebuild them with the scripts below.

## Build Runtime Image

Build `llama.cpp` SYCL inside the known-good Intel container:

```bash
LLAMA_SYCL_IMAGE=intel/llm-scaler-vllm:0.14.0-b7.1 \
  $REPO_ROOT/scripts/build_llama_sycl_container.sh
```

Notes:

- the build script now auto-detects the main GPU PCI device ID with `sudo xpu-smi discovery -d 0`
- on this Battlemage machine, that resolves to `GGML_SYCL_DEVICE_ARCH=bmg-g31`
- verify the configure result in `third_party/llama.cpp/build-sycl-0.14.0-b7.1/CMakeCache.txt`

Commit that build into a reusable image:

```bash
$REPO_ROOT/scripts/commit_llama_sycl_container.sh
```

To preserve the AOT build as a distinct artifact:

```bash
LLAMA_SYCL_COMMIT_IMAGE=llama-sycl-local:0.14.0-b7.1-4d99d45-aot-real-bmg-g31 \
  $REPO_ROOT/scripts/commit_llama_sycl_container.sh
```

## Build Profiling Tooling

Build PTI `unitrace` against the committed runtime image:

```bash
$REPO_ROOT/scripts/build_unitrace.sh
```

Build the profiler image that contains `unitrace`, `libunitrace_tool.so`, `uniview.py`, and metrics configs:

```bash
$REPO_ROOT/scripts/build_llama_sycl_profiler_image.sh
```

For the AOT runtime image:

```bash
LLAMA_PROFILER_BASE_IMAGE=llama-sycl-local:0.14.0-b7.1-4d99d45-aot-real-bmg-g31 \
LLAMA_PROFILER_IMAGE=llama-sycl-profiler:0.14.0-b7.1-4d99d45-aot-real-bmg-g31 \
  $REPO_ROOT/scripts/build_llama_sycl_profiler_image.sh
```

## Metric Permissions

For Battlemage on the `xe` kernel driver, allow observation access:

```bash
echo 0 | sudo tee /proc/sys/dev/xe/observation_paranoid
```

The profiling script will set and restore this automatically when metric collection is enabled.

## Quick Validation

Validate the profiler image:

```bash
sudo docker run --rm --entrypoint bash --privileged --device /dev/dri:/dev/dri \
  llama-sycl-profiler:0.14.0-b7.1-4d99d45-aot-real-bmg-g31 \
  -lc 'unitrace --version && unitrace --device-list && unitrace --metric-list >/tmp/metrics.txt && head -n 20 /tmp/metrics.txt'
```

Validate `llama.cpp` device enumeration:

```bash
sudo docker run --rm --entrypoint bash --privileged --device /dev/dri:/dev/dri \
  -e ZE_AFFINITY_MASK=0 \
  -e ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 \
  llama-sycl-local:0.14.0-b7.1-4d99d45-aot-real-bmg-g31 \
  -lc 'llama-bench --list-devices'
```

## Run A Profile

Summary timing profile:

```bash
LLAMA_PROFILE_TRACE_MODE=summary \
LLAMA_PROFILE_PROMPT_TOKENS=1 \
LLAMA_PROFILE_GEN_TOKENS=1 \
LLAMA_PROFILE_COLLECT_METRICS=0 \
  $REPO_ROOT/scripts/profile_llama_sycl_decode.sh \
  $REPO_ROOT/models/Qwen3.5-9B-Q4_K_M.gguf
```

Chrome timeline profile:

```bash
LLAMA_PROFILE_TRACE_MODE=chrome \
LLAMA_PROFILE_PROMPT_TOKENS=1 \
LLAMA_PROFILE_GEN_TOKENS=1 \
LLAMA_PROFILE_COLLECT_METRICS=0 \
LLAMA_PROFILER_IMAGE=llama-sycl-profiler:0.14.0-b7.1-4d99d45-aot-real-bmg-g31 \
  $REPO_ROOT/scripts/profile_llama_sycl_decode.sh \
  $REPO_ROOT/models/Qwen3.5-9B-Q4_K_M.gguf
```

Optional runtime toggles that are now forwarded into the container and recorded in `run.txt`:

- `GGML_SYCL_DISABLE_GRAPH`
- `GGML_SYCL_DISABLE_OPT`
- `GGML_SYCL_PRIORITIZE_DMMV`
- `GGML_SYCL_USE_ASYNC_MEM_OP`
- `GGML_SYCL_DEBUG`

To trace the current locally rebuilt binary instead of the binary baked into the profiler image:

```bash
LLAMA_PROFILE_LLAMA_BENCH=$REPO_ROOT/third_party/llama.cpp/build-sycl-0.14.0-b7.1/bin/llama-bench \
LLAMA_PROFILER_IMAGE=llama-sycl-profiler:0.14.0-b7.1-4d99d45-aot-real-bmg-g31 \
  $REPO_ROOT/scripts/profile_llama_sycl_decode.sh \
  $REPO_ROOT/models/Qwen3.5-9B-Q4_K_M.gguf
```

Notes:

- this is the preferred path once kernel edits are being tested rapidly
- the script will mount the workspace binary into the profiler container and prepend its directory to `LD_LIBRARY_PATH`
- this avoids rebuilding the profiler image just to trace one kernel iteration
- the profiler flow now calls `/opt/unitrace/bin/unitrace` directly instead of the older `/usr/local/bin/unitrace` wrapper
- this matters because the wrapper used to prepend `/opt/llama.cpp-sycl/bin` ahead of the mounted local build, which could make a local `llama-bench` load the wrong `libggml-sycl.so`
- if you see `No kernel named ... was found` during profiling of a local rebuild, suspect executable/library mismatch first

To avoid collisions when running multiple profiling jobs, the profiling script now creates unique output directories by default using the shell PID. You can also force an explicit location with:

```bash
LLAMA_PROFILE_RESULTS_DIR=$REPO_ROOT/results/profiles/my-run \
  $REPO_ROOT/scripts/profile_llama_sycl_decode.sh \
  $REPO_ROOT/models/Qwen3.5-9B-Q4_K_M.gguf
```

The benchmark runner also accepts:

```bash
LLAMA_BENCH_RESULTS_DIR=$REPO_ROOT/results/my-bench \
  $REPO_ROOT/scripts/run_qwen9b_sycl_bench.sh \
  $REPO_ROOT/models/Qwen3.5-9B-Q4_K_M.gguf
```

The profiling runner also supports an untraced warmup phase:

```bash
LLAMA_PROFILE_WARMUP_PROMPT_TOKENS=512 \
LLAMA_PROFILE_WARMUP_GEN_TOKENS=32 \
LLAMA_PROFILE_WARMUP_REPS=1 \
  $REPO_ROOT/scripts/profile_llama_sycl_decode.sh \
  $REPO_ROOT/models/Qwen3.5-9B-Q4_K_M.gguf
```

Important caveat:

- this warmup runs in a separate `llama-bench` process before the traced `llama-bench` process
- it warms filesystem caches and validates steady-state throughput
- it does not remove per-process JIT or module-creation cost from the traced run
- do not interpret it as a true “same-process steady-state trace”

Metric profile:

```bash
LLAMA_PROFILE_TRACE_MODE=summary \
LLAMA_PROFILE_PROMPT_TOKENS=1 \
LLAMA_PROFILE_GEN_TOKENS=1 \
LLAMA_PROFILE_COLLECT_METRICS=1 \
LLAMA_PROFILE_METRIC_GROUP=ComputeBasic \
  $REPO_ROOT/scripts/profile_llama_sycl_decode.sh \
  $REPO_ROOT/models/Qwen3.5-9B-Q4_K_M.gguf
```

## Outputs

Each run creates:

- `results/profiles/<timestamp>/run.txt`
- `results/profiles/<timestamp>/image-inspect.json`
- `results/profiles/<timestamp>/llama-bench-profile.json`
- `results/profiles/<timestamp>/unitrace-version.txt`
- `results/profiles/<timestamp>/unitrace-device-list.txt`
- `results/profiles/<timestamp>/unitrace-metric-list.txt`
- `results/profiles/<timestamp>/xpu-topology.txt`
- `results/profiles/<timestamp>/xpu-discovery-gpu0.txt`
- `results/profiles/<timestamp>/xpu-stats-pre.json`
- `results/profiles/<timestamp>/xpu-stats-post.json`
- `results/profiles/<timestamp>/unitrace/...`

## Summarize A Completed Trace

GPU hotspots:

```bash
$REPO_ROOT/scripts/summarize_unitrace_trace.py \
  $REPO_ROOT/results/profiles/<timestamp>/unitrace/llama-bench.<pid>.json \
  gpu_op 30
```

CPU hotspots:

```bash
$REPO_ROOT/scripts/summarize_unitrace_trace.py \
  $REPO_ROOT/results/profiles/<timestamp>/unitrace/llama-bench.<pid>.json \
  cpu_op 30
```

## Known Good Baseline Artifact

Completed first chrome trace:

- `results/profiles/2026-03-10T16-33-07/`

Supporting analysis:

- `results/profiles/2026-03-10T16-33-07/unitrace-hotspots-gpu.txt`
- `results/profiles/2026-03-10T16-33-07/unitrace-hotspots-cpu.txt`
- `results/profiles/2026-03-10T16-33-07/llama-bench-profile-summary.txt`

## Known Good Local-Binary Trace Path

The current validated local-binary trace path is:

- profiler image `llama-sycl-profiler:0.14.0-b7.1-4d99d45-aot-real-bmg-g31`
- local binary `third_party/llama.cpp/build-sycl-0.14.0-b7.1/bin/llama-bench`
- artifact `results/profiles/2026-03-11T21-45-00-long-decode-q6k-byte-sub4-scalarized-fix1/`

What was fixed:

- the profiler image wrapper and the profiling script were both changed so the mounted local build directory stays ahead of the baked image binaries in `LD_LIBRARY_PATH`
- this resolved the earlier `RMS_NORM` kernel lookup failure seen when tracing a rebuilt local binary

Important interpretation:

- traced throughput is lower than untraced benchmark throughput because `unitrace` adds heavy timing overhead
- compare traced runs to traced runs, not to the scored untraced benchmark ledger

Key result from the fixed local-binary trace:

- `p=1`, `n=128`, `b=512`, `ub=512`
- `pp1 = 4.60 tok/s`
- `tg128 = 23.51 tok/s`
- top device buckets remained:
  - reordered `q4_K`
  - `q5_K` DMMV
  - reordered `q6_K`
  - M2D copies

## First Conclusions

- SYCL quantized reorder and dequant paths dominate GPU time.
- D2D copies are a real cost.
- Host synchronization and module build/JIT cost are too high.
- The first code and build targets should be:
  - `ggml/src/ggml-sycl/mmq.cpp`
  - `ggml/src/ggml-sycl/mmvq.cpp`
  - `ggml/src/ggml-sycl/dmmv.cpp`
  - reorder staging in `ggml/src/ggml-sycl/ggml-sycl.cpp`
  - AOT build configuration to reduce `zeModuleCreate` / `urProgramBuildExp`

## First Optimization Result

The first concrete fix was not in the kernels. It was in the build path.

- the original build script failed to pass `GGML_SYCL_DEVICE_ARCH` because the `xpu-smi` parser captured trailing table characters
- after fixing the parser, the build now configures with `GGML_SYCL_DEVICE_ARCH=bmg-g31`
- AOT trace artifacts:
  - `results/profiles/2026-03-10T17-04-55/`
  - `results/profiles/2026-03-10T17-05-53/`

Observed impact on the traced `p=1, n=1` case:

- non-AOT baseline `results/profiles/2026-03-10T16-33-07/`
  - `pp1`: `153.1 s`
  - `tg1`: `46.6 s`
  - `urProgramBuildExp`: `39.2 s`
  - `zeModuleCreate`: `39.2 s`
- AOT build `results/profiles/2026-03-10T17-04-55/`
  - `pp1`: `32.5 s`
  - `tg1`: `64.7 ms`
  - `urProgramBuildExp`: `32.3 s`
  - `zeModuleCreate`: `32.3 s`

Observed impact on the warmed benchmark:

- previous warmup on the same model before the AOT fix:
  - `pp512`: `15.07 tok/s`
  - `tg32`: `1.17 tok/s`
- after the AOT fix:
  - `pp512`: `328.42 tok/s`
  - `tg32`: `50.69 tok/s`

Interpretation:

- the broken build pipeline was the first major blocker
- AOT makes the dense SYCL baseline usable enough to continue kernel tuning
- enabling `GGML_SYCL_DISABLE_GRAPH=0` on the `p=1, n=1` AOT trace did not materially improve results, so graphs are not the next best lever

## Second Optimization Result

The next low-risk change was to decouple async USM allocation/free from graph enablement in the SYCL backend:

- file: `ggml/src/ggml-sycl/ggml-sycl.cpp`
- new runtime knob: `GGML_SYCL_USE_ASYNC_MEM_OP`
- default: enabled when the oneAPI async allocation extension is present

Reason:

- reorder staging was still copy-heavy after the AOT fix
- the previous code disabled async allocation/free whenever graphs were disabled, which forced extra host waits in the reorder path even in non-graph execution

Committed images for this variant:

- `llama-sycl-local:0.14.0-b7.1-4d99d45-aot-bmg-g31-asyncmem`
- `llama-sycl-profiler:0.14.0-b7.1-4d99d45-aot-bmg-g31-asyncmem`

Measured dense benchmark impact on Qwen 3.5 9B Q4:

- async on artifact: `results/2026-03-10-async-on/`
  - `pp512`: `349.45 tok/s`
  - `tg128`: `50.27 tok/s`
- async off artifact: `results/2026-03-10-async-off/`
  - `pp512`: `222.39 tok/s`
  - `tg128`: `9.13 tok/s`

Interpretation:

- async memory ops are not a marginal tweak on this machine
- they are required to keep dense decode performance near the improved AOT baseline

## Warm Profiling Finding

The first attempt at a warm profile for the async-mem build is:

- `results/profiles/2026-03-10T17-33-10-warm-asyncmem/`

Run shape:

- untraced warmup: `pp512`, `tg32`
- traced run: `p=1`, `n=1`

Observed warmup throughput:

- `pp512`: `332.39 tok/s`
- `tg32`: `50.69 tok/s`

Observed traced throughput:

- `pp1`: `3.97 tok/s`
- `tg1`: `15.72 tok/s`

Interpretation:

- the untraced warmup confirms the backend stays in the improved steady-state regime
- the traced run still shows `urProgramBuildExp` / `zeModuleCreate` because `unitrace` launches a fresh process
- this workflow is still useful, but it is not a same-process warm trace

## Same-Process Decode Tail Trace

The current best steady-state hotspot view is:

- `results/profiles/2026-03-11T09-40-00-long-decode/`

Run shape:

- traced run only, no separate warmup
- `p=1`
- `n=128`
- `b=512`
- `ub=512`
- trace mode: chrome timeline plus host/device timing

Observed traced throughput:

- `pp1`: `0.03114 tok/s`
- `tg128`: `15.52 tok/s`

Supporting summaries:

- `results/profiles/2026-03-11T09-40-00-long-decode/unitrace-hotspots-cpu.txt`
- `results/profiles/2026-03-11T09-40-00-long-decode/unitrace-hotspots-gpu.txt`
- `results/profiles/2026-03-11T09-40-00-long-decode/unitrace-hotspots-cpu-tail-90pct.txt`
- `results/profiles/2026-03-11T09-40-00-long-decode/unitrace-hotspots-gpu-tail-90pct.txt`

The trace summarizer now supports clipped windows so startup-heavy traces can still be used for steady-state decode analysis:

```bash
$REPO_ROOT/scripts/summarize_unitrace_trace.py \
  $REPO_ROOT/results/profiles/2026-03-11T09-40-00-long-decode/unitrace/llama-bench.<pid>.json \
  gpu_op 20 --start-pct 90
```

```bash
$REPO_ROOT/scripts/summarize_unitrace_trace.py \
  $REPO_ROOT/results/profiles/2026-03-11T09-40-00-long-decode/unitrace/llama-bench.<pid>.json \
  cpu_op 20 --start-pct 90
```

Important interpretation:

- full-trace CPU totals are still dominated by `urProgramBuildExp` / `zeModuleCreate`
- tail-window summaries remove those startup buckets from the top set
- in the last 10% of the trace, CPU time is dominated by `submit`, `urEnqueueKernelLaunch`, `urKernelSetArgValue`, `queue.wait`, `urQueueFinish`, and `zeEventHostSynchronize`
- in the last 10% of the trace, GPU time is dominated by:
  - `reorder_mul_mat_vec_q4_k_q8_1_sycl`
  - `dequantize_mul_mat_vec_q5_K_sycl`
  - `reorder_mul_mat_vec_q6_k_q8_1_sycl`
  - `dequantize_mul_mat_vec_q8_0_sycl`

Conclusion:

- the next kernel/code pass should stay on reorder MMVQ, DMMV dequant paths, and submission overhead
- attention is not the first limiting factor for this dense decode case

## Dispatch A/B Findings

Two direct A/B checks were run on the current async-mem build using the same scored workload:

- workload: `pp512`, `tg128`, `b=2048`, `ub=2048`, `r=3`
- model: `Qwen3.5-9B-Q4_K_M.gguf`

Artifacts:

- DMMV-priority run: `results/2026-03-11-dmmv-priority/`
- optimize-disabled run: `results/2026-03-11-disable-opt/`

Observed throughput:

- baseline async-on:
  - `pp512`: `349.45 tok/s`
  - `tg128`: `50.27 tok/s`
- `GGML_SYCL_PRIORITIZE_DMMV=1`:
  - `pp512`: `337.30 tok/s`
  - `tg128`: `35.46 tok/s`
- `GGML_SYCL_DISABLE_OPT=1`:
  - `pp512`: `345.68 tok/s`
  - `tg128`: `35.42 tok/s`

Interpretation:

- forcing DMMV is a decode regression on this model
- disabling the current optimize/reorder path is also a decode regression
- the correct next target is inside the reorder MMVQ path and the remaining DMMV dequant kernels, not a dispatch flip away from reorder

## AOT Flag Fix

The SYCL AOT path is now fixed and validated.

Root cause:

- the old build only passed `--offload-arch=bmg-g31`
- it did not switch the SYCL device target to `spir64_gen`
- as a result, the compiler accepted the flag into generated commands but treated it as unused during normal object compilation

Fix implemented in `ggml/src/ggml-sycl/CMakeLists.txt`:

- use `-fsycl-targets=spir64_gen`
- use `-Xsycl-target-backend=spir64_gen "-device bmg-g31"`
- skip the old `-ze-intel-greater-than-4GB-buffer-required` link flag in AOT mode because OCLOC rejects it on this toolchain

Validation:

- compile-time `argument unused during compilation` warnings for the old AOT flag path are gone
- `build.ninja` now carries:
  - `-fsycl-targets=spir64_gen`
  - `-Xsycl-target-backend=spir64_gen "-device bmg-g31"`
- the built `libggml-sycl.so.0.9.7` now contains `sycl-spir64_gen-unknown-unknown` bundles
- the AOT link stage now runs real offline compilation through:
  - `llvm-foreach`
  - `ocloc ... -spirv_input -device bmg-g31`

New committed images:

- runtime: `llama-sycl-local:0.14.0-b7.1-4d99d45-aot-real-bmg-g31`
- profiler: `llama-sycl-profiler:0.14.0-b7.1-4d99d45-aot-real-bmg-g31`

Dense benchmark check on the fixed AOT build:

- artifact: `results/2026-03-11-aot-real/`
- scored `pp512`, `tg128`, `b=2048`, `ub=2048`, `r=3`
- result:
  - `pp512`: `345.49 tok/s`
  - `tg128`: `50.01 tok/s`

Interpretation:

- the build is now using the intended offline Intel GPU path
- the flag fix did not regress dense throughput relative to the prior async-mem baseline
- we can now treat future kernel tuning results as coming from a real AOT build instead of an ambiguous mixed path

## Rejected Launch-Geometry Experiment

An exploratory local rebuild changed the reorder MMVQ kernels from the current fixed `16` subgroups to a `128`-thread target intended to mirror CUDA's generic single-vector MMVQ thread budget more closely.

Artifact:

- `results/2026-03-11-reorder-128t/`

Observed throughput:

- `pp512`: `340.00 tok/s`
- `tg128`: `50.10 tok/s`

Interpretation:

- decode was effectively neutral and prompt throughput was slightly worse
- this was not a convincing improvement
- the code change was reverted and should be treated as a negative/neutral experiment, not as current backend state
