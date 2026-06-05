//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FUSED_Q4K_SWIGLU_HPP
#define GGML_SYCL_FUSED_Q4K_SWIGLU_HPP

#include "common.hpp"

bool ggml_sycl_can_run_fused_q4k_swiglu(const ggml_backend_sycl_context & ctx, const ggml_tensor * dst);
bool ggml_sycl_try_fused_q4k_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FUSED_Q4K_SWIGLU_HPP
