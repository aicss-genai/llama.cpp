//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FFN_HPP
#define GGML_SYCL_FFN_HPP

#include "common.hpp"

void ggml_sycl_op_ffn_fused(
	ggml_backend_sycl_context & ctx,
	const ggml_tensor * gate_mul_mat,
	const ggml_tensor * up_mul_mat,
	ggml_tensor * swiglu_dst);

#endif // GGML_SYCL_FFN_HPP
