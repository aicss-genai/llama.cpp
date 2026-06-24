//
// ESIMD implementation of the Q4_K reorder MMVQ base launcher (ncols == 1).
//
// Built only when GGML_SYCL_ESIMD_MMVQ is defined. When the flag is off this
// translation unit contributes nothing and behavior is identical to the scalar
// path. See phase6-esimd-incremental-plan.md for the staged bring-up.
//

#include "mmvq_esimd.hpp"

#if defined(GGML_SYCL_ESIMD_MMVQ) || defined(GGML_SYCL_ESIMD_DPAS)

#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <cstdlib>

#include "ggml.h"
#include "common.hpp"
#include "quants.hpp"

#endif  // GGML_SYCL_ESIMD_MMVQ || GGML_SYCL_ESIMD_DPAS

#ifdef GGML_SYCL_ESIMD_MMVQ

// ----------------------------------------------------------------------------
// Phase A — plumbing foothold. One work-item per row. No reads of vx/vy.
//   A0: store constant 0.0f to dst[row] (Phase 5 "Test A").                [DONE]
//   A1: store (float) row to prove row->work-item mapping.                 [DONE]
//
// Phase B1 — dequant-bootstrap (float MAC). One work-item per row.
//   Reuses the proven-correct Q4_K reorder *weight* decode from the committed
//   kernel-fusion-draft DMMV ESIMD path (a92e9012d:dmmv.cpp): scale/min decode,
//   nibble unpack, dall/dmin. The only change is the activation side: that path
//   consumed float activations, whereas MMVQ feeds q8_1, so we dequantize the
//   q8_1 quants to float (a_j = d8_sub * q8_j) and do a float multiply-add. This
//   is mathematically identical to the scalar dp4a result (modulo float
//   reassociation) and uses zero unproven idioms. All memory access goes through
//   copy_from/copy_to (ESIMD forbids plain pointer deref). The 12-byte scale
//   field is over-read as 16 bytes (the dm region always follows the scales, so
//   the extra 4 bytes are in-bounds), avoiding the invalid block_load<u8,12>.
// ----------------------------------------------------------------------------

// Per-block dp4a dot at FULL SIMD width (simd<int,32>), accumulated into a 32-wide
// float `acc` (reduced once per row by the caller). The earlier simd<int,8> dp4a
// lost to the simd<float,32> float path purely on lane utilization (8 of 32 EU
// lanes used); here dp4a packs 4 sub-blocks/instruction so all 32 lanes are busy,
// giving dp4a's 4-MAC density AT full width. dp4a operand order:
//   dp4a(src0, src1, src2) = src0 + dot4(src1, src2)   (accumulator is FIRST).
// Each lane independently dots its own 4 int8 pairs; lanes [8s..8s+7] hold
// sub-block s's partials, scaled per-sub-block, summed once at the end into
//   Σ_subblocks d8·(dall·sc·dot1 − dmin·m·dot2), matching scalar
// vec_dot_q4_K_q8_1_impl_vmmq. Q4_K nibbles 0..15 so signed≡unsigned.
// Accumulate one block of ONE row into `acc`, using activations PRE-LOADED and
// shared across the row group (u0/u1 = the two 32-dword q8 vectors for sub-block
// groups g=0/1). Multi-row sharing halves activation block_load traffic (the
// activation is identical across output rows) and gives independent ILP chains.
static SYCL_ESIMD_FUNCTION void accumulate_row_block_q4_k(
        int ib, int row, int blocks_per_row,
        const uint8_t * qs_base, const uint8_t * scales_base, const sycl::half * dm_base,
        const sycl::half * y_ds,
        const sycl::ext::intel::esimd::simd<int, 32> & u0,
        const sycl::ext::intel::esimd::simd<int, 32> & u1,
        sycl::ext::intel::esimd::simd<float, 32> & acc) {
    using namespace sycl::ext::intel::esimd;

    const simd<int, 32> ones = 0x01010101;
    const simd<int, 32> z    = 0;

    const size_t bi = (size_t) row * blocks_per_row + ib;

    // (Software prefetch of next-block quants was tried: it raised read BW
    // 269→283 and lowered SBID, but kernel time stayed flat — the single
    // thread's compute/issue rate, not memory latency, is the limiter here.
    // Reverted; kept this note so it isn't re-attempted.)

    // --- weights: proven Q4_K reorder decode (from DMMV ESIMD) ---
    simd<uint8_t, 128> qb = block_load<uint8_t, 128>(qs_base + bi * (QK_K / 2));
    // scale field is 12 bytes; load 16 (power-of-2; dm region follows so in-bounds).
    simd<uint8_t, 16> sc16 = block_load<uint8_t, 16>(scales_base + bi * K_SCALE_SIZE);

    const float dall = (float) dm_base[bi * 2 + 0];
    const float dmin = (float) dm_base[bi * 2 + 1];

    // get_scale_min_k4 (vectorized), 8 sub-block sc and m values.
    simd<uint8_t, 8> s_lo = 0;
    simd<uint8_t, 8> s_hi = 0;
    {
        simd<uint8_t, 4> b0 = sc16.select<4, 1>(0);
        simd<uint8_t, 4> b1 = sc16.select<4, 1>(4);
        simd<uint8_t, 4> b2 = sc16.select<4, 1>(8);
        s_lo.select<4, 1>(0) = b0 & simd<uint8_t, 4>(63);
        s_lo.select<4, 1>(4) = (b2 & simd<uint8_t, 4>(0x0F)) |
                               ((b0 >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
        s_hi.select<4, 1>(0) = b1 & simd<uint8_t, 4>(63);
        s_hi.select<4, 1>(4) = (b2 >> simd<uint8_t, 4>(4)) |
                               ((b1 >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    }
    // Fold the per-sub-block activation scale d8 into the weight scales as VECTORS
    // (was 8 scalar extracts + element-wise builds = the bulk of the scalar ISA).
    // y_ds holds (d,s) half pairs per sub-block; the 8 d values are every-other
    // half starting at ib*16. One vector load + strided select + vector half->float.
    simd<sycl::half, 16> ds16 = block_load<sycl::half, 16>(y_ds + (size_t) ib * 16);
    simd<sycl::half, 8>  d8h  = ds16.select<8, 2>(0);   // every-other = the d values
    simd<float, 8>       d8   = convert<float>(d8h);

    simd<float, 8> sfd_lo = convert<float>(s_lo) * (dall * d8);   // d8 * dall * sc
    simd<float, 8> sfd_hi = convert<float>(s_hi) * (-dmin * d8);  // -d8 * dmin * m

    simd<uint8_t, 128> lo = qb & simd<uint8_t, 128>(0x0f);
    simd<uint8_t, 128> hi = qb >> simd<uint8_t, 128>(4);

    // Sub-block s -> nibbles: even s = lo of weight-bytes [(s/2)*32 .. +32),
    //                          odd  s = hi of the same bytes. Activations (u0/u1)
    // are pre-loaded and shared. Process 4 sub-blocks per dp4a (group g covers
    // sub-blocks 4g..4g+3).
    for (int g = 0; g < 2; ++g) {
        const simd<int, 32> & u = (g == 0) ? u0 : u1;

        // nibbles: interleave lo/hi 32-byte chunks to match the 4 sub-blocks.
        simd<uint8_t, 128> wn;
        wn.select<32, 1>(0)  = lo.select<32, 1>(g * 64 + 0);   // sub-block 4g+0 (lo)
        wn.select<32, 1>(32) = hi.select<32, 1>(g * 64 + 0);   // sub-block 4g+1 (hi)
        wn.select<32, 1>(64) = lo.select<32, 1>(g * 64 + 32);  // sub-block 4g+2 (lo)
        wn.select<32, 1>(96) = hi.select<32, 1>(g * 64 + 32);  // sub-block 4g+3 (hi)
        simd<int, 32> v = wn.bit_cast_view<int>();

        // 32 lane-partials; lanes [8k..8k+7] belong to sub-block 4g+k.
        simd<float, 32> p1 = convert<float>(dp4a<int>(z, v, u));
        simd<float, 32> p2 = convert<float>(dp4a<int>(z, ones, u));

        // Broadcast each of the 4 sub-block scales over its 8 lanes via a single
        // hardware region op (replicate_vs_w_hs<Rep=4,VS=1,W=8,HS=0>): lane 8k+j
        // = scale[4g+k]. Replaces the per-sub-block scalar extract/build loop.
        simd<float, 32> sclo = sfd_lo.replicate_vs_w_hs<4, 1, 8, 0>(g * 4);
        simd<float, 32> schi = sfd_hi.replicate_vs_w_hs<4, 1, 8, 0>(g * 4);

        acc += p1 * sclo + p2 * schi;
    }
}

void reorder_mul_mat_vec_q4_k_q8_1_esimd(const void * vx, const void * vy, float * dst, const int ncols,
                                         const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Best variant: one work-item per row (multi-row sharing saturated BW but cut
    // thread-level latency hiding → net regression; reverted). dp4a at full SIMD
    // width + block_load wide coalesced loads are the wins.
    constexpr int wg_size      = 16;
    const int     global_size  = ceil_div(nrows, wg_size) * wg_size;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(wg_size)),
                         [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                             using namespace sycl::ext::intel::esimd;

                             const int row = it.get_global_id(0);
                             if (row >= nrows) {
                                 return;
                             }

                             const int    blocks_per_row = ncols / QK_K;
                             const size_t nblocks        = (size_t) nrows * blocks_per_row;

                             // Reorder weight layout (matches block_q_t<Q4_K> offsets):
                             //   quants : vx + bi * (QK_K/2)
                             //   scales : vx + nblocks*(QK_K/2) + bi * K_SCALE_SIZE
                             //   dm     : scales_base + nblocks*K_SCALE_SIZE + bi * half2
                             const uint8_t *    qs_base     = static_cast<const uint8_t *>(vx);
                             const uint8_t *    scales_base = qs_base + nblocks * (QK_K / 2);
                             const sycl::half * dm_base     = reinterpret_cast<const sycl::half *>(
                                 scales_base + nblocks * K_SCALE_SIZE);

                             // q8_1 activations: ncols int8 quants, then one half2 (d,s) per
                             // 32-quant block. Only d is needed (matches scalar reorder dot).
                             const int8_t *    y_q  = static_cast<const int8_t *>(vy);
                             const sycl::half * y_ds = reinterpret_cast<const sycl::half *>(
                                 static_cast<const char *>(vy) + ncols);

                             simd<float, 32> acc = 0.0f;
                             for (int ib = 0; ib < blocks_per_row; ++ib) {
                                 simd<int8_t, 256> yq = block_load<int8_t, 256>(y_q + (size_t) ib * QK_K);
                                 simd<int, 32> u0 = yq.select<128, 1>(0).bit_cast_view<int>();
                                 simd<int, 32> u1 = yq.select<128, 1>(128).bit_cast_view<int>();

                                 accumulate_row_block_q4_k(ib, row, blocks_per_row, qs_base, scales_base, dm_base, y_ds, u0, u1, acc);
                             }

                             simd<float, 1> r = reduce<float>(acc, std::plus<>{});
                             r.copy_to(dst + row);
                         });
    });
}

#endif  // GGML_SYCL_ESIMD_MMVQ

#ifdef GGML_SYCL_ESIMD_DPAS

#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

// ----------------------------------------------------------------------------
// Phase E stage 3 - xmx::dpas Q4_K MMVQ kernel.
//
// One work-item per 16-row group: dpas computes 16 output rows per instruction.
// Reads the pre-transposed u4 VNNI layout written by reorder_qw_q4_k_dpas:
//   weights : 256-B u4 VNNI tile per (group, super-block, sub-block)
//   scales  : 12 scale bytes SOA across the 16 rows, per (group, super-block)
//   dm      : 16 dall then 16 dmin halves, per (group, super-block)
// Per sub-block: dpas<8,1,int,int,u4,s8>(0, tile, act) -> int32[16] = dot1 (one
// per row). dot2 = sum of the 32 activations (row-independent). Scales applied
// per row, 16-wide, matching vec_dot_q4_K_q8_1_impl_vmmq:
//   total[n] = dall[n]*Sum_ss d8[ss]*dot1[n,ss]*sc[n,ss]
//            - dmin[n]*Sum_ss d8[ss]*dot2[ss]*m[n,ss].
//
// Requires nrows % 16 == 0 (weight-matrix row counts are always multiples of 16;
// the dominant FFN launch nrows=17408=16*1088). Asserted below.
// ----------------------------------------------------------------------------

// get_scale_min_k4 for sub-block ss, vectorized across the 16 rows. sc16[j] holds
// the 16 rows' scale-byte j (the SOA-by-row layout the reorder writer produced).
static SYCL_ESIMD_FUNCTION void get_scale_min_k4_16(
        int ss,
        const sycl::ext::intel::esimd::simd<uint8_t, 16> sc16[12],
        sycl::ext::intel::esimd::simd<uint8_t, 16> & sc,
        sycl::ext::intel::esimd::simd<uint8_t, 16> & m) {
    using namespace sycl::ext::intel::esimd;
    if (ss < 4) {
        sc = sc16[ss]     & simd<uint8_t, 16>(63);
        m  = sc16[ss + 4] & simd<uint8_t, 16>(63);
    } else {
        sc = (sc16[ss + 4] & simd<uint8_t, 16>(0x0F)) |
             ((sc16[ss - 4] >> simd<uint8_t, 16>(6)) << simd<uint8_t, 16>(4));
        m  = (sc16[ss + 4] >> simd<uint8_t, 16>(4)) |
             ((sc16[ss]     >> simd<uint8_t, 16>(6)) << simd<uint8_t, 16>(4));
    }
}

// K-split cooperation: NSPLIT work-items share one 16-row group, each handling a
// stride of the bpr super-blocks, partials reduced through SLM. dpas alone (1
// work-item/group) starved occupancy (1088 work-items, 32%); this multiplies the
// thread count by NSPLIT to hide DRAM latency. (C1's SLM cooperation regressed at
// 17408 already-oversubscribed threads; here we are undersubscribed, so the same
// mechanism is being re-tested in the regime where it can help.) NSPLIT must be a
// compile-time constant: ESIMD slm_init requires a constexpr size (the runtime
// slm_init overload aborts in IGC), so the env knob selects among instantiations.
template <int NSPLIT>
static void launch_dpas_nsplit(const void * vx, const void * vy, float * dst, const int ncols,
                               const int nrows, const int n_groups, dpct::queue_ptr stream) {
    namespace xmx = sycl::ext::intel::esimd::xmx;
    using dt      = sycl::ext::intel::esimd::xmx::dpas_argument_type;

    const int global_size = n_groups * NSPLIT;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(NSPLIT)),
                         [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                             using namespace sycl::ext::intel::esimd;

                             slm_init<NSPLIT * 16 * sizeof(float)>();

                             const int g = it.get_group(0);   // 16-row group
                             const int s = it.get_local_id(0); // split index within the group

                             const int    bpr     = ncols / QK_K;  // super-blocks per row
                             const size_t nblocks = (size_t) nrows * bpr;

                             // Reorder dpas layout bases (see reorder_qw_q4_k_dpas):
                             //   weights tile (group,sb,ss) at qs_base + (grp*8+ss)*256
                             //   scales  SOA  at scale_base + grp*192 + j*16 + n
                             //   dm           at dm_base    + grp*32  + {n, 16+n}
                             const uint8_t *    qs_base    = static_cast<const uint8_t *>(vx);
                             const uint8_t *    scale_base = qs_base + nblocks * (QK_K / 2);
                             const sycl::half * dm_base    =
                                 reinterpret_cast<const sycl::half *>(scale_base + nblocks * K_SCALE_SIZE);

                             // q8_1 activations: ncols int8 quants, then one (d,s) half2
                             // per 32-quant sub-block. Only d (every-other half) is used.
                             const int8_t *     y_q  = static_cast<const int8_t *>(vy);
                             const sycl::half * y_ds = reinterpret_cast<const sycl::half *>(
                                 static_cast<const char *>(vy) + ncols);

                             simd<float, 16> total = 0.0f;

                             for (int sb = s; sb < bpr; sb += NSPLIT) {
                                 const int grp = g * bpr + sb;

                                 // Per-(group,super-block) scale/dm, 16 rows wide.
                                 simd<uint8_t, 16> sc16[12];
                                 for (int j = 0; j < 12; ++j) {
                                     sc16[j] = block_load<uint8_t, 16>(scale_base + (size_t) grp * 192 + j * 16);
                                 }
                                 simd<sycl::half, 16> dall_h = block_load<sycl::half, 16>(dm_base + (size_t) grp * 32);
                                 simd<sycl::half, 16> dmin_h = block_load<sycl::half, 16>(dm_base + (size_t) grp * 32 + 16);
                                 simd<float, 16>      dall   = convert<float>(dall_h);
                                 simd<float, 16>      dmin   = convert<float>(dmin_h);

                                 // 8 sub-block activation d8 scales (every-other half).
                                 simd<sycl::half, 16> ds16 = block_load<sycl::half, 16>(y_ds + (size_t) sb * 16);
                                 simd<sycl::half, 8>  d8h   = ds16.select<8, 2>(0);  // materialize (convert needs simd, not view)
                                 simd<float, 8>       d8    = convert<float>(d8h);

                                 simd<float, 16> acc_d = 0.0f;
                                 simd<float, 16> acc_m = 0.0f;

                                 for (int ss = 0; ss < 8; ++ss) {
                                     // activation sub-block: 32 int8 at column sb*256 + ss*32.
                                     simd<int8_t, 32> a =
                                         block_load<int8_t, 32>(y_q + (size_t) sb * QK_K + ss * 32);
                                     // weight VNNI tile: 256 bytes = 512 u4.
                                     simd<uint8_t, 256> b =
                                         block_load<uint8_t, 256>(qs_base + ((size_t) (grp * 8 + ss)) * 256);

                                     simd<int, 16> c = 0;
                                     simd<int, 16> dot1 =
                                         xmx::dpas<8, 1, int, int, uint8_t, int8_t, dt::u4, dt::s8>(c, b, a);

                                     // dot2 = sum of the 32 activations (row-independent).
                                     const int dot2 = reduce<int>(simd<int, 32>(convert<int>(a)), std::plus<>{});

                                     simd<uint8_t, 16> sc, m;
                                     get_scale_min_k4_16(ss, sc16, sc, m);

                                     const float d8s = d8[ss];
                                     acc_d += (d8s * convert<float>(dot1)) * convert<float>(sc);
                                     acc_m += (d8s * (float) dot2) * convert<float>(m);
                                 }

                                 total += dall * acc_d - dmin * acc_m;
                             }

                             // Reduce the NSPLIT partials through SLM, then lane 0 stores.
                             slm_block_store<float, 16>(s * 16 * sizeof(float), total);
                             barrier();
                             if (s == 0) {
                                 simd<float, 16> sum = total;
                                 for (int t = 1; t < NSPLIT; ++t) {
                                     sum += slm_block_load<float, 16>(t * 16 * sizeof(float));
                                 }
                                 sum.copy_to(dst + (size_t) g * 16);
                             }
                         });
    });
}

void reorder_mul_mat_vec_q4_k_q8_1_dpas(const void * vx, const void * vy, float * dst, const int ncols,
                                        const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    GGML_ASSERT(nrows % 16 == 0);  // 16-row groups packed by reorder_qw_q4_k_dpas

    const int n_groups = nrows / 16;

    // NSPLIT default 4; env-overridable for sweeping the occupancy optimum. Must
    // map to a compile-time instantiation (see launch_dpas_nsplit).
    // Default NSPLIT=10: a measured sweep optimum on Qwen3-14B (bpr=20). tg128
    // saturates by ~10880 work-items; NSPLIT=20 matches it but doubles SLM/workgroup
    // width for no gain. Ragged divisions of bpr (8, 16) regress (barrier serializes
    // on the longer-loaded work-items). Env-overridable for re-sweeping on other shapes.
    const char * nsplit_env = getenv("GGML_SYCL_DPAS_NSPLIT");
    const int    nsplit     = nsplit_env ? atoi(nsplit_env) : 10;

    switch (nsplit) {
        case 1:  launch_dpas_nsplit<1>(vx, vy, dst, ncols, nrows, n_groups, stream);  break;
        case 2:  launch_dpas_nsplit<2>(vx, vy, dst, ncols, nrows, n_groups, stream);  break;
        case 4:  launch_dpas_nsplit<4>(vx, vy, dst, ncols, nrows, n_groups, stream);  break;
        case 5:  launch_dpas_nsplit<5>(vx, vy, dst, ncols, nrows, n_groups, stream);  break;
        case 8:  launch_dpas_nsplit<8>(vx, vy, dst, ncols, nrows, n_groups, stream);  break;
        case 16: launch_dpas_nsplit<16>(vx, vy, dst, ncols, nrows, n_groups, stream); break;
        case 20: launch_dpas_nsplit<20>(vx, vy, dst, ncols, nrows, n_groups, stream); break;
        case 10:
        default: launch_dpas_nsplit<10>(vx, vy, dst, ncols, nrows, n_groups, stream); break;
    }
}

#endif  // GGML_SYCL_ESIMD_DPAS
