# SYCL Disassembled Kernel Workflow

This document captures the workflow used to integrate a disassembled Intel SYCL kernel into `llama.cpp` and turn it into a reusable process for future kernel efforts.

## When To Use This Workflow

Use this workflow when you start from one of the following:
- a disassembled kernel, SPIR-V blob, or vendor-specific kernel dump
- a performance gap that suggests a missing fused path
- a backend op sequence that is semantically correct but too slow when executed as separate ops

The goal is to preserve default behavior, add the kernel as an optional optimization, and keep fallback behavior obvious and safe.

## High-Level Flow

1. Anchor on the concrete execution path.
   - Start from the backend dispatch site, the graph scheduler, a failing benchmark, or the existing hot kernel.
   - Do not begin with broad repo exploration.

2. Form one falsifiable hypothesis.
   - Identify the exact branch, node pattern, or kernel path that likely controls the behavior.
   - Find one cheap check that can disprove it.

3. Make the smallest testable change.
   - Prefer a reversible probe or a narrow refactor first.
   - Keep the first edit local to the controlling code path.

4. Validate immediately after the first substantive edit.
   - Run the cheapest relevant compile, build, or benchmark check.
   - Do not widen scope before that check.

5. Iterate from the result.
   - If the check fails, fix the same slice first.
   - If the hypothesis is falsified, move one hop closer to the real control point.

6. Finish with a focused build and performance validation.
   - Confirm both correctness and throughput.
   - Compare against the unfused baseline with the same workload and flags.

## Recommended Implementation Sequence

### Phase 1: Identify the Kernel Shape

Work out:
- the exact graph pattern being optimized
- the tensor types and layouts involved
- whether the kernel is a linear op, a fork/join subgraph, or a wrapper around another op
- whether the target layout is reordered, contiguous, or requires a local transform

For future efforts, write these findings down before coding. It saves time when the fusion gate or kernel contract changes.

### Phase 2: Add A Feature Flag

Keep the new kernel behind a runtime flag until correctness and performance are proven.

Recommended pattern:
- env flag in backend init
- backend log line showing the flag state
- default off

This makes it easy to compare fused and unfused behavior without rebuilding.

### Phase 3: Add An Internal API

Add a backend-internal entry point for the fused path.

Rules:
- declare it in a backend header only if another translation unit needs it
- keep the implementation in the backend source file unless reuse requires a split
- keep public APIs out of anonymous namespaces

When the launcher does not return a value, make it `void`.

### Phase 4: Implement The Kernel In SYCL C++

Translate the algorithm into ordinary SYCL source, not embedded assembly.

Guidelines:
- reuse existing quantization and reorder helpers where possible
- prefer the same numerics as the unfused path
- keep the first version constrained to the MVP shape set
- do not mix incompatible execution models in the same translation unit unless the toolchain has already proven it safe

### Phase 5: Wire The Graph Hook

For fused subgraphs, add scheduler-time pattern matching in the graph execution loop.

Prefer this over only handling the final op when:
- the fused work spans multiple producer nodes
- the fused output should skip several unfused nodes
- the producer order or node reuse matters

Use the graph scheduler to decide whether to fuse, then call the kernel launcher only when the pattern is accepted.

### Phase 6: Keep Existing Paths As Fallbacks

The fused path should be additive.

Fallback rules:
- if the feature flag is off, use the existing path
- if the shape or layout does not match, use the existing path
- if the device or tensor layout is unsupported, use the existing path

Never let the fused path change behavior for non-target cases.

## Coding Conventions

### Naming

Use names that describe what the code does, not historical implementation details.

Examples:
- `ggml_sycl_op_*_fused` for fused execution helpers
- `ggml_sycl_can_fuse` for fusion eligibility checks
- `ggml_sycl_try_*` only if the function really returns success or failure

If a function only executes and cannot fail in a meaningful way, prefer `void`.

### Linkage And Scope

Use anonymous namespaces for local helpers only.

Do not place functions with external declarations inside an anonymous namespace, or `dlopen`/linker resolution may break.

### Checks And Assertions

Split checks by responsibility:
- graph-shape and eligibility checks in the scheduler gate
- execution-time invariants in the launcher

If the scheduler already guarantees a condition, remove redundant launcher checks only after confirming they are not needed for safety.

### File Layout

Keep backend-specific kernel code in the backend directory.

When renaming files:
- update includes immediately
- update CMake or glob-based source discovery if needed
- update header guards
- verify the old file name is fully gone

### Feature Flags

Keep new paths opt-in until they are stable.

Recommended pattern:
- `GGML_SYCL_ENABLE_<FEATURE>` env var
- boolean or integer backend global
- info log line at init

## Integration Points To Expect

Future kernel efforts usually touch these places:

- `ggml/src/ggml-sycl/ggml-sycl.cpp`
  - backend init and device reporting
  - scheduler-time graph execution
  - fusion gating
  - dispatch hooks

- `ggml/src/ggml-sycl/<kernel>.hpp`
  - launcher declarations
  - backend-internal interfaces

- `ggml/src/ggml-sycl/<kernel>.cpp`
  - kernel implementation
  - helpers for quantization, layout checks, or node matching

- `ggml/src/ggml-sycl/CMakeLists.txt`
  - source discovery if files are not picked up by globbing

- `ggml/src/ggml-sycl/common.hpp` or similar shared headers
  - feature flags or globals shared across translation units

## Mistakes To Avoid

- Using a linear-chain fusion helper for a fork/join subgraph.
- Assuming producer order is fixed when the graph allows either order.
- Keeping a public launcher in an anonymous namespace.
- Leaving stale file names or stale include directives after a rename.
- Adding an ESIMD header to a translation unit that also relies on non-ESIMD subgroup operations if the toolchain cannot mix them.
- Removing a gate from the scheduler without updating the launcher invariants, or vice versa.
- Forgetting that a device-level capability check may be redundant if the tensor metadata already proves the requirement, or the opposite.
- Measuring performance only after a long chain of unrelated edits.

## Validation Checklist

After each meaningful change, verify:
- the code builds
- the old and new file names are consistent
- the fusion gate still matches the intended graph shape
- the launcher still executes the same math as the baseline
- the fallback path still works when the feature flag is off
- the benchmark moves in the expected direction

Suggested order:
1. compile or build the touched backend
2. run the narrow benchmark or test that motivated the change
3. compare against the baseline build with the same inputs and flags

## Practical Notes From This Effort

- Scheduler-time subgraph fusion was the right place to catch the `MUL_MAT + MUL_MAT -> GLU(SWIGLU)` pattern.
- The fused launcher should remain small and focused on execution.
- Performance regressed when fusion eligibility became too restrictive, so be careful when replacing graph checks with narrower helpers.
- A cleanup that is semantically harmless can still change the fusion hit rate.
- Reordered Q4_K support was the correct MVP target; non-reordered support belongs in a later expansion phase if needed.

## Disassembly Recipe

Use this recipe when you need to pull out the generated kernels and inspect the SPIR-V or disassembly.

### 1) Build the latest SPIRV-Tools

```bash
git clone https://github.com/KhronosGroup/SPIRV-Tools.git
cd SPIRV-Tools
python3 utils/git-sync-deps
cmake -B build
cmake --build build -j
```

### 2) Dump all kernels to `.spv` files

```bash
SYCL_DUMP_IMAGES=1 ./llama-bench -m /models/Qwen3-14B-Q4_K_M.gguf -ngl 100 -p 0
```

### 3) Locate the kernel of interest

```bash
strings -f *.spv | c++filt | grep mlp_forward
```

### 4) Disassemble the kernel

```bash
./SPIRV-Tools/build/tools/spirv-dis sycl_spir64_2102.spv -o sycl_spir64_2102.spvasm
```

## Reusable Template For New Kernels

1. Identify the exact op pattern.
2. Decide whether the kernel is linear or a subgraph fusion.
3. Add a runtime feature flag.
4. Add a backend-internal launcher API.
5. Implement the SYCL kernel.
6. Add scheduler-time fusion gating if the pattern spans multiple nodes.
7. Keep fallback behavior intact.
8. Validate build, correctness, and throughput.
9. Clean up naming only after the hot path is stable.
