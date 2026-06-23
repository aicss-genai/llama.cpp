//
// ESIMD implementation of the Q4_K reorder MMVQ base launcher (ncols == 1).
//
// Built only when GGML_SYCL_ESIMD_MMVQ is defined. When the flag is off this
// translation unit contributes nothing and behavior is identical to the scalar
// path. See phase6-esimd-incremental-plan.md for the staged bring-up.
//

#include "mmvq_esimd.hpp"

#ifdef GGML_SYCL_ESIMD_MMVQ

#include <sycl/ext/intel/esimd.hpp>

#include "ggml.h"
#include "common.hpp"
#include "quants.hpp"

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
