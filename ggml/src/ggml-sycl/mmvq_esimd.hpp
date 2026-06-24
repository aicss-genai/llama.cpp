//
// ESIMD implementation of the Q4_K reorder MMVQ base launcher (ncols == 1).
// Declaration is always visible; the definition exists only when
// GGML_SYCL_ESIMD_MMVQ is defined (see mmvq_esimd.cpp).
//

#ifndef GGML_SYCL_MMVQ_ESIMD_HPP
#define GGML_SYCL_MMVQ_ESIMD_HPP

#include "common.hpp"

#ifdef GGML_SYCL_ESIMD_MMVQ

void reorder_mul_mat_vec_q4_k_q8_1_esimd(const void * vx, const void * vy, float * dst, const int ncols,
                                         const int nrows, dpct::queue_ptr stream);

#endif  // GGML_SYCL_ESIMD_MMVQ

#ifdef GGML_SYCL_ESIMD_DPAS

// xmx::dpas Q4_K MMVQ path. Reads the dpas VNNI reorder layout written by
// reorder_qw_q4_k_dpas (16 output rows per dpas instruction). Distinct from the
// _esimd dp4a path above and selected by a distinct compile flag.
void reorder_mul_mat_vec_q4_k_q8_1_dpas(const void * vx, const void * vy, float * dst, const int ncols,
                                        const int nrows, dpct::queue_ptr stream);

#endif  // GGML_SYCL_ESIMD_DPAS

#endif  // GGML_SYCL_MMVQ_ESIMD_HPP
