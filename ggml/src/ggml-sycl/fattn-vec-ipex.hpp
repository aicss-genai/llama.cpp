#ifndef GGML_SYCL_FATTN_VEC_IPEX_HPP
#define GGML_SYCL_FATTN_VEC_IPEX_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <cmath>
#include <float.h>

#include "common.hpp"
#include "ggml.h"
#include "fattn-common.hpp"

// PROTOTYPE (throwaway) - IPEX-style single-pass flash-attention vec kernel.
//
// Goal: confirm the redesign hypothesis before hand-writing - that giving each
// thread a COMPLETE Q.K dot product (no cross-lane reduction per key) plus a
// single-pass register-resident accumulation removes the 66% DistStall and the
// ~10x L3 over-read of the existing vec kernel, the way IPEX's sdp_fp16_kernel
// does (which wins at only 8.7% occupancy).
//
// Specialized for: D == 128, F16 K/V, decode (one query column), GQA, optional
// non-ALiBi mask. No logit-softcap, no sinks (those keep the existing kernel).
//
// Layout: one workgroup per (head, sequence); `nthreads` threads split the keys.
// Each thread runs an online softmax over its key slice with full in-thread dot
// products, then the workgroup combines the per-thread (m, l, VKQ) partials once.

namespace syclex_ipex = sycl::ext::oneapi::experimental;

template <int D, int nthreads, int warp_size>
static void flash_attn_ext_vec_ipex(const char * __restrict__ Q,
                        const char * __restrict__ K,
                        const char * __restrict__ V,
                        const char * __restrict__ mask,
                        const char * __restrict__ sinks,
                        const int* __restrict__ KV_max,
                        float* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        const float scale,
                        const float max_bias,
                        const float m0,
                        const float m1,
                        const uint32_t n_head_log2,
                        const float logit_softcap,
                        const int32_t ne00,
                        const sycl::uint3 ne01,
                        const int32_t ne02,
                        const int32_t ne03,
                        const int32_t nb01,
                        const int32_t nb02,
                        const int32_t nb03,
                        const int32_t ne10,
                        const int32_t ne11,
                        const int32_t ne12,
                        const int32_t ne13,
                        const int32_t nb11,
                        const int32_t nb12,
                        const int64_t nb13,
                        const int32_t nb21,
                        const int32_t nb22,
                        const int64_t nb23,
                        const int32_t ne31,
                        const int32_t ne32,
                        const int32_t ne33,
                        const int32_t nb31,
                        const int32_t nb32,
                        const int64_t nb33) {
#ifdef SYCL_FLASH_ATTN
    constexpr int D2 = D / 2;

    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid = warp_size * item.get_local_id(1) + item.get_local_id(2);

    const int sequence  = item.get_group(0) / ne02;
    const int head      = item.get_group(0) - sequence * ne02;
    const int gqa_ratio = ne02 / ne12;

    const float * Q_f = (const float *) (Q + nb03 * sequence + nb02 * head);
    const char  * K_base = K + nb13 * sequence + nb12 * (head / gqa_ratio);
    const char  * V_base = V + nb23 * sequence + nb22 * (head / gqa_ratio);
    const sycl::half * maskh = mask ? (const sycl::half *) (mask + nb33 * (sequence % ne33)) : nullptr;

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    const int k_max = KV_max ? KV_max[sequence * item.get_group_range(2) + item.get_group(2)] : ne11;

    // Full Q row resident in registers (scaled), shared shape across all threads.
    sycl::half2 Q_h2[D2];
#pragma unroll
    for (int i = 0; i < D2; ++i) {
        const sycl::float2 q = ((const sycl::float2 *) Q_f)[i];
        Q_h2[i] = sycl::half2(q.x() * scale, q.y() * scale);
    }

    // Per-thread online-softmax state over this thread's key slice.
    float m = -FLT_MAX / 2.0f;
    float l = 0.0f;
    float VKQ[D] = { 0.0f };

    // KV-split: block gy owns a strided slice of the keys so that many workgroups
    // run concurrently (raises HW thread count toward IPEX). Each work-item still
    // computes its own COMPLETE dot products (no cross-lane reduction).
    const int pb = item.get_group_range(1);
    const int gy = item.get_group(1);
    for (int k = gy * nthreads + tid; k < k_max; k += pb * nthreads) {
        const sycl::half2 * K_h2 = (const sycl::half2 *) (K_base + k * nb11);

        // Complete in-thread dot product, 4 partial accumulators for ILP.
        sycl::float2 a0 = { 0.0f, 0.0f }, a1 = { 0.0f, 0.0f }, a2 = { 0.0f, 0.0f }, a3 = { 0.0f, 0.0f };
#pragma unroll
        for (int i = 0; i < D2; i += 4) {
            a0 = a0 + __half22float2(K_h2[i + 0] * Q_h2[i + 0]);
            a1 = a1 + __half22float2(K_h2[i + 1] * Q_h2[i + 1]);
            a2 = a2 + __half22float2(K_h2[i + 2] * Q_h2[i + 2]);
            a3 = a3 + __half22float2(K_h2[i + 3] * Q_h2[i + 3]);
        }
        const sycl::float2 a = (a0 + a1) + (a2 + a3);
        float qk = a.x() + a.y();
        if (maskh) {
            qk += slope * (float) maskh[k];
        }

        const float m_new = sycl::fmax(m, qk);
        const float corr  = sycl::native::exp(m - m_new);
        const float p     = sycl::native::exp(qk - m_new);

        l = l * corr + p;

        const sycl::half2 * V_h2 = (const sycl::half2 *) (V_base + k * nb21);
#pragma unroll
        for (int i = 0; i < D2; ++i) {
            const sycl::float2 v = __half22float2(V_h2[i]);
            VKQ[2 * i + 0] = VKQ[2 * i + 0] * corr + p * v.x();
            VKQ[2 * i + 1] = VKQ[2 * i + 1] * corr + p * v.y();
        }

        m = m_new;
    }

    // V4 single-kernel: one workgroup owns the ENTIRE KV range (parallel_blocks == 1),
    // so this intra-workgroup reduction is the final result - no global partials, no
    // separate combine kernel (the structural cost that capped v2/v3). Multi-warp:
    // reduce across all `nthreads` threads via SLM. m_smem/l_smem are tiny; vkq_smem is
    // nthreads*D and holds each thread's partial numerator for the column reduction.
    syclex_ipex::work_group_static<float[nthreads]>     m_smem;
    syclex_ipex::work_group_static<float[nthreads]>     l_smem;
    syclex_ipex::work_group_static<float[nthreads * D]> vkq_smem;

    m_smem[tid] = m;
    item.barrier(sycl::access::fence_space::local_space);

    float M = -FLT_MAX / 2.0f;
    for (int t = 0; t < nthreads; ++t) {
        M = sycl::fmax(M, m_smem[t]);
    }

    const float corr = sycl::native::exp(m - M);
    l_smem[tid] = l * corr;
#pragma unroll
    for (int i = 0; i < D; ++i) {
        vkq_smem[tid * D + i] = VKQ[i] * corr;
    }
    item.barrier(sycl::access::fence_space::local_space);

    float L = 0.0f;
    for (int t = 0; t < nthreads; ++t) {
        L += l_smem[t];
    }

    // pb == 1 (single-kernel): write the normalized result directly. The pb > 1 path
    // (un-normalized numerator + meta for flash_attn_combine_results) is kept so the
    // kernel still works under KV-split, but the V4 launcher forces pb == 1.
    const int j_dst = ((sequence * int(ne01.z()) + 0) * ne02 + head) * pb + gy;
    float * dst_out = dst + (size_t) j_dst * D;

    for (int d = tid; d < D; d += nthreads) {
        float acc = 0.0f;
        for (int t = 0; t < nthreads; ++t) {
            acc += vkq_smem[t * D + d];
        }
        dst_out[d] = pb > 1 ? acc : acc / L;
    }

    if (pb > 1 && tid == 0) {
        dst_meta[j_dst] = sycl::float2(M, L);
    }

    GGML_UNUSED_VARS(sinks, logit_softcap, ne00, ne03, nb01, ne10, ne13, ne31, ne32, nb31, nb32);
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
#endif // SYCL_FLASH_ATTN
}

template <int D, int type_K, int type_V>
void ggml_sycl_flash_attn_ext_vec_ipex_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    constexpr int warp_size = WARP_16_SIZE;
    constexpr int nthreads  = 8 * WARP_16_SIZE; // V4: 8 warps/wg cover the whole KV range
    constexpr int nwarps    = nthreads / warp_size;

    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;

    // V4 single-kernel: huge nbatch_fa forces ntiles_KQ == 1 -> parallel_blocks == 1.
    // One workgroup per (head, sequence) owns the entire KV range and reduces it
    // intra-workgroup, so there is no separate flash_attn_combine_results pass and no
    // global round-trip of partials (the ~7 us / 23%-of-op cost that capped v2/v3).
    const int nbatch_fa = 1 << 20;

    launch_fattn<D, 1, 1, flash_attn_ext_vec_ipex<D, nthreads, warp_size>, warp_size>(
        ctx, dst, nwarps, /*nbytes_shared*/ 0, nbatch_fa, need_f16_K, need_f16_V, /*stream_k*/ false);
}

#endif // GGML_SYCL_FATTN_VEC_IPEX_HPP
