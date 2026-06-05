//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "fused-q4k-swiglu.hpp"

bool ggml_sycl_can_run_fused_q4k_swiglu(const ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    if (!g_ggml_sycl_enable_fused_q4k_swiglu) {
        return false;
    }

    if (dst == nullptr || dst->op != GGML_OP_GLU || ggml_get_glu_op(dst) != GGML_GLU_OP_SWIGLU) {
        return false;
    }

    // Current implementation is intentionally Intel-only and additive.
    if (!ggml_sycl_info().devices[ctx.device].opt_feature.reorder) {
        return false;
    }

    // Phase A scaffold: accept only common contiguous float/half layouts.
    if (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->src[0] == nullptr || dst->src[0]->type != dst->type) {
        return false;
    }

    return false;
}

bool ggml_sycl_try_fused_q4k_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    if (!ggml_sycl_can_run_fused_q4k_swiglu(ctx, dst)) {
        return false;
    }

    // Phase A scaffold: no fused kernel yet.
    GGML_SYCL_DEBUG("[SYCL] fused_q4k_swiglu: eligible but not implemented, falling back\n");
    return false;
}
