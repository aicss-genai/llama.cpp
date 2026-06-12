#ifndef GGML_SYCL_FATTN_VEC_ESIMD_HPP
#define GGML_SYCL_FATTN_VEC_ESIMD_HPP

#include <sycl/sycl.hpp>
#include <cmath>
#include <float.h>
#include <cstdint>

#include "common.hpp"
#include "ggml.h"

#if defined(__INTEL_LLVM_COMPILER)
#include <sycl/ext/intel/esimd.hpp>
#define GGML_SYCL_FATTN_ESIMD_ENABLED 1
#endif

// PROTOTYPE (throwaway) - ESIMD flash-attention decode kernel, head dim 128, F16 K/V.
//
// V4 (the SIMT prototype) already gives each work-item a COMPLETE Q.K dot product
// with a register-resident Q, so the remaining structural difference vs IPEX's
// sdp_fp16_kernel is the ESIMD execution model: 1 work-item == 1 HW thread, using
// the full GRF width for one key's dot product and the VKQ accumulator, with wide
// block loads of K/V instead of SIMT 16-lane packing. This kernel isolates that.
//
// v1 mapping: ONE ESIMD thread per (head, sequence) owns the whole KV range and
// keeps Q, the running (m, l) and the 128-wide VKQ accumulator in registers. No
// SLM, no barrier, no cross-thread combine - the purest test of ESIMD efficiency.
// Specialized for D == 128, F16 K/V, decode (one query column), GQA, optional mask;
// no logit-softcap, no sinks (those keep the existing kernel).

#ifdef GGML_SYCL_FATTN_ESIMD_ENABLED
// v2: KV-split. Each (head, sequence) is owned by a workgroup of NSPLIT ESIMD threads;
// thread sp processes the strided key slice k = sp, sp+NSPLIT, ... keeping its partial
// online-softmax state (m, l, acc[D]) in registers, then the workgroup combines the
// NSPLIT partials once via SLM. This mirrors IPEX's per-head KV-blocking (its {32;1;1}
// block over a 1-query decode): it raises the thread count from 40 (NW=1) to 40*NSPLIT
// and cuts each thread's serial loop from kv_len to kv_len/NSPLIT, while preserving the
// register-resident K/V streaming (validated by the L3=2MB read in fattn-15).
template <int D, int NSPLIT>
static void flash_attn_ext_vec_esimd_run(
        const char * __restrict__ Q_data, const char * __restrict__ K_data,
        const char * __restrict__ V_data, const char * __restrict__ mask_data,
        float * __restrict__ dst_data, dpct::queue_ptr stream,
        const int n_head, const int n_seq, const int gqa_ratio, const int kv_len,
        const float scale, const float max_bias, const float m0, const float m1,
        const uint32_t n_head_log2,
        const int64_t q_nb2, const int64_t q_nb3,
        const int64_t k_nb1, const int64_t k_nb2, const int64_t k_nb3,
        const int64_t v_nb1, const int64_t v_nb2, const int64_t v_nb3,
        const int64_t mask_nb3, const int mask_ne3) {
    const size_t n_wg = (size_t) n_head * n_seq;

    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<float, 1> m_smem(sycl::range<1>(NSPLIT), h);
        sycl::local_accessor<float, 1> l_smem(sycl::range<1>(NSPLIT), h);
        sycl::local_accessor<float, 1> acc_smem(sycl::range<1>((size_t) NSPLIT * D), h);

        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(n_wg * NSPLIT), sycl::range<1>(NSPLIT)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                using namespace sycl::ext::intel::esimd;

                const int sp   = (int) it.get_local_id(0);
                const int wg   = (int) it.get_group(0);
                const int head = wg % n_head;
                const int seq  = wg / n_head;

                const int kv_head = head / gqa_ratio;

                const float * Qf   = (const float *) (Q_data + (size_t) seq * q_nb3 + (size_t) head * q_nb2);
                const char  * Kb   = K_data + (size_t) seq * k_nb3 + (size_t) kv_head * k_nb2;
                const char  * Vb   = V_data + (size_t) seq * v_nb3 + (size_t) kv_head * v_nb2;
                const sycl::half * maskh =
                    mask_data ? (const sycl::half *) (mask_data + (size_t) (seq % mask_ne3) * mask_nb3) : nullptr;

                // ALiBi slope (1.0 when max_bias == 0, the decode case we profile).
                float slope = 1.0f;
                if (max_bias > 0.0f) {
                    const float base = head < (int) n_head_log2 ? m0 : m1;
                    const int   e    = head < (int) n_head_log2 ? head + 1 : 2 * (head - (int) n_head_log2) + 1;
                    slope = 1.0f;
                    for (int i = 0; i < e; ++i) {
                        slope *= base;
                    }
                }

                // Q row resident in registers (scaled, fp16), reused across this thread's keys.
                // fp16 throughout matches IPEX: it halves register/instruction width and avoids a
                // half->float convert on every K/V load (the dominant InstrFetch stall in v2).
                simd<sycl::half, D> q = convert<sycl::half>(block_load<float, D>(Qf) * scale);

                float               m   = -FLT_MAX / 2.0f;
                float               l   = 0.0f;
                simd<sycl::half, D> acc = sycl::half(0.0f);

                // Partial online softmax over the strided key slice owned by this thread.
                for (int k = sp; k < kv_len; k += NSPLIT) {
                    const simd<sycl::half, D> kf = block_load<sycl::half, D>((const sycl::half *) (Kb + (size_t) k * k_nb1));

                    float qk = reduce<float>(q * kf, std::plus<>{});
                    if (maskh) {
                        qk += slope * (float) maskh[k];
                    }

                    const float m_new = qk > m ? qk : m;
                    simd<float, 2> earg;
                    earg[0] = m  - m_new;
                    earg[1] = qk - m_new;
                    const simd<float, 2> ex = exp(earg);
                    const float corr = ex[0];
                    const float p    = ex[1];

                    l = l * corr + p;

                    const simd<sycl::half, D> vf = block_load<sycl::half, D>((const sycl::half *) (Vb + (size_t) k * v_nb1));
                    acc = acc * sycl::half(corr) + vf * sycl::half(p);

                    m = m_new;
                }

                // Publish this thread's partial (m, l, acc) to SLM (acc un-normalized, widened once).
                m_smem[sp] = m;
                l_smem[sp] = l;
                const simd<float, D> acc_f = convert<float>(acc);
#pragma unroll
                for (int i = 0; i < D; ++i) {
                    acc_smem[(size_t) sp * D + i] = acc_f[i];
                }
                it.barrier(sycl::access::fence_space::local_space);

                // Combine: global max M, per-thread weight w = exp(m_sp - M), denom L.
                float M = -FLT_MAX / 2.0f;
#pragma unroll
                for (int t = 0; t < NSPLIT; ++t) {
                    const float mt = m_smem[t];
                    M = mt > M ? mt : M;
                }
                simd<float, NSPLIT> marg;
#pragma unroll
                for (int t = 0; t < NSPLIT; ++t) {
                    marg[t] = m_smem[t] - M;
                }
                const simd<float, NSPLIT> w = exp(marg);
                float L = 0.0f;
#pragma unroll
                for (int t = 0; t < NSPLIT; ++t) {
                    L += l_smem[t] * (float) w[t];
                }

                // Each thread writes a strided slice of the D outputs (numerator / L).
                float * out = dst_data + (size_t) (seq * n_head + head) * D;
                for (int d = sp; d < D; d += NSPLIT) {
                    float num = 0.0f;
#pragma unroll
                    for (int t = 0; t < NSPLIT; ++t) {
                        num += acc_smem[(size_t) t * D + d] * (float) w[t];
                    }
                    out[d] = num / L;
                }
            });
    });
}
#endif // GGML_SYCL_FATTN_ESIMD_ENABLED

template <int D, int type_K, int type_V>
void ggml_sycl_flash_attn_ext_vec_esimd_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#ifdef GGML_SYCL_FATTN_ESIMD_ENABLED
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16);
    GGML_ASSERT(Q->ne[0] == D && Q->ne[1] == 1);

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float    m0          = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float    m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // NSPLIT KV-blocks per head: 40 heads x NSPLIT = thread count (IPEX uses 32/head -> 1280).
    constexpr int NSPLIT = 32;

    flash_attn_ext_vec_esimd_run<D, NSPLIT>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        mask ? (const char *) mask->data : nullptr, (float *) dst->data, ctx.stream(),
        (int) Q->ne[2], (int) Q->ne[3], gqa_ratio, (int) K->ne[1],
        scale, max_bias, m0, m1, n_head_log2,
        Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3],
        mask ? mask->nb[3] : 0, mask ? (int) mask->ne[3] : 1);
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("ESIMD flash-attention requires the Intel oneAPI (DPC++) compiler");
#endif // GGML_SYCL_FATTN_ESIMD_ENABLED
}

#endif // GGML_SYCL_FATTN_VEC_ESIMD_HPP
