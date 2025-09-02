//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: MIT
//

#include <map>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"
#include "kai/matmul_integer.h"
#include "kai/matmul_integer_to_float.h"
#include "kai/matmul_common.h"
#include "kai/reduce_add_scale_reordered.h"
#include "kai/reduce_common.h"
#include "kai/reorder.h"
#include "kai/reorder_common.h"
#include "kai/reorder_transpose.h"

#include "mlasi_kleidiai.h"

//Matmul with float output of dynamic quantized A and symmetric quantized B.

#pragma pack(push,1)
struct KaiPackedBHeader {
  uint32_t magic;   // 'KAI1' = 0x3149414B
  uint32_t K;
  uint32_t N;
  uint32_t flags;   // reserved
};
#pragma pack(pop)

static constexpr uint32_t kKaiPackedBMagic = 0x3149414B;

inline bool IsKaiPackedB(const void* p, size_t K, size_t N, const std::byte** payload_out) {
  if (!p) return false;
  const auto* h = reinterpret_cast<const KaiPackedBHeader*>(p);
  if (h->magic != kKaiPackedBMagic) return false;
  if (h->K != K || h->N != N) return false;
  *payload_out = reinterpret_cast<const std::byte*>(h + 1);
  return true;
}


size_t
MLASCALL
ArmKleidiAI::MlasDynamicQgemmPackBSize(
    size_t N,
    size_t K
) {
    //Default to sme2_mopa but this may not awalys be the most optimal kernel variant to use
    auto nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    auto kr = kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    auto sr = kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();

    //regardless of kernel variant use neon packing variant
    KLEIDIAI_KERNEL_LOG("kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon Groups=1"
                        << " N="<< N << " K=" << K << " nr=" << nr << " kr=" << kr << " sr=" << sr);
    return kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);
}

void
MLASCALL
ArmKleidiAI::MlasDynamicQgemmPackB(
    size_t N,
    size_t K,
    const int8_t* B,
    const float* Scales,
    const float* Bias,
    void* PackedB
) {
    // Default to sme2_mopa but this may not awalys be the most optimal kernel variant to use
    auto nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    auto kr = kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    auto sr = kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();

    // y - float output
    // scale_factor_lhs - lhs scaling factor
    // scale_factor_rhs - rhs scaling factor
    // lhs_q - lhs quantized (asymmetric, so has zero point)
    // rhs_q - rhs quantized (symmetric so no zero point)
    // lhs_zp - lhs zero point
    // y = (1/(scale_factor_lhs * scale_factor_rhs) * sum( (lhs_q + lhs_zp)*rhs_q )) + bias

    // rhs packing requires lhs_zp because it will perform lhs_zp*rhs_q during rhs packing
    // because lhs quantization is hidden from us, by lhs quant packing, we don't have a value for lhs_zp it is
    // lhs dynamic quantization

    kai_rhs_pack_qsi8cx_params params{
        1,  // lhs_zp - set to 1 so it becomes sum((lhs_q + 1)*rhs_q )),
            // the actual lhs_zp is applied during the matmul
        1.f  // it is not used
    };

    //regardless of kernel variant use neon packing variant
    kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(1, N, K, nr, kr, sr, B,
                                             // N bias values
                                             Bias,
                                             // N scale values
                                             Scales, PackedB, 0, &params);
}

void
MLASCALL
ArmKleidiAI::MlasDynamicQGemmBatch(
    const MLAS_GEMM_DYN_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_DYN_QUANT_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* ThreadPool
) {
    for (auto b = BatchN; b > 0; --b,++DataParams) {
        auto mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
        auto kr = kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
        auto sr = kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();


        //TODO enable multi-threading for lhs packing and matmul
        MLAS_UNREFERENCED_PARAMETER(ThreadPool);

        //Dynamic Quantize A - lhs
        auto lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(Shape.M, Shape.K, mr, kr, sr);
        std::byte* lhs = nullptr;
        std::unique_ptr<std::byte[]> fallback;

        if (DataParams->Workspace && DataParams->WorkspaceSize >= lhs_size) {
            lhs = static_cast<std::byte*>(DataParams->Workspace);
        } else {
            fallback = std::make_unique<std::byte[]>(lhs_size);
            lhs = fallback.get();
        }
        KLEIDIAI_KERNEL_LOG("kai_run_lhs_quant_pack_qai8dxp_f32"
                            << " M="<< Shape.M << " K=" << Shape.K << " mr=" << mr << " kr=" << kr << " sr=" << sr << " m_idx_start=0");

        kai_run_lhs_quant_pack_qai8dxp_f32(Shape.M, Shape.K, mr, kr, sr, 0, DataParams->A,
                                           Shape.K*sizeof(float), lhs);

        KLEIDIAI_KERNEL_LOG("kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa");
        kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa(
            Shape.M, Shape.N, Shape.K, lhs, DataParams->PackedB,
            DataParams->C,
            Shape.N * sizeof(float),
            sizeof(float),
            -std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
        );
    }
}


size_t
MLASCALL
ArmKleidiAI::MlasGemmPackBSize(
    size_t N,
    size_t K,
    bool /*AIsSigned*/,
    bool /*BIsSigned*/
) {
    const size_t payload = kai_get_dst_size_reorder_transpose(K, N);
    return sizeof(KaiPackedBHeader) + payload;
}

bool
MLASCALL
ArmKleidiAI::MlasGemmPackB(
    size_t N,
    size_t K,
    const uint8_t* B,
    size_t ldb,         // elements
    bool /*AIsSigned*/,
    bool /*BIsSigned*/,
    void* PackedB
) {
    // 1) Write header
    auto* hdr = reinterpret_cast<KaiPackedBHeader*>(PackedB);
    hdr->magic = kKaiPackedBMagic;
    hdr->K = static_cast<uint32_t>(K);
    hdr->N = static_cast<uint32_t>(N);
    hdr->flags = 0;

    // 2) Pack payload right after header
    auto* dst_payload = reinterpret_cast<std::byte*>(hdr + 1);

    const kai_reorder_shape_t shape{ .height = K, .width = N };
    const kai_reorder_buffers_t bufs{
        .src = B,
        .src_stride = ldb * sizeof(uint8_t),   // convert elements→bytes
        .dst = dst_payload,
    };
    const kai_reorder_params_t prm{ .flags = 0 };

    kai_run_reorder_transpose(&shape, &bufs, &prm);
    return true;
}

bool
MLASCALL
ArmKleidiAI::MlasGemmBatch(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* /*ThreadPool*/
) {
  const size_t M = Shape.M, N = Shape.N, K = Shape.K;

  for (size_t i = 0; i < BatchN; ++i) {
    const auto& p = DataParams[i];

    // Only accept prepacked RHS in KAI format; no per-column ZP .
    if (!p.BIsPacked || p.PerColumnZeroPoints) return false;

    // Locate packed payload
    const std::byte* rhs_reordered = nullptr;
    if (!IsKaiPackedB(p.B, K, N, &rhs_reordered)) return false; // not ours → let MLAS handle

    // --- LHS: reorder at runtime (A changes every call) ---
    const size_t lhs_reordered_size = kai_get_dst_size_reorder(M, K);
    std::vector<std::byte> lhs_reordered(lhs_reordered_size);
    {
      const kai_reorder_shape_t s{ .height = M, .width = K };
      const kai_reorder_buffers_t b{ .src = p.A, .src_stride = p.lda, .dst = lhs_reordered.data() };
      const kai_reorder_params_t prm{ .flags = 0 };
      kai_run_reorder(&s, &b, &prm);
    }

    // --- Accumulator row/col biases on KAI layouts ---
    const size_t acc_row_bias_size = kai_get_dst_size_reduce_add_scale_reordered(M);
    std::vector<std::byte> acc_row_bias(acc_row_bias_size);
    {
      const kai_reduce_shape_t s{ .height = M, .width = K };
      const int32_t dst_scale = -static_cast<int32_t>(*p.ZeroPointB);
      const int32_t dst_bias  = 0;
      const kai_reduce_buffers_t b{
        .src = lhs_reordered.data(), .src_stride = 0,
        .dst = acc_row_bias.data(),  .dst_stride = 0,
        .dst_scale = &dst_scale,     .dst_scale_stride = 0,
        .dst_bias  = &dst_bias,      .dst_bias_stride  = 0
      };
      const kai_reduce_params_t prm{ .flags = 0 };
      kai_run_reduce_add_scale_reordered(&s, &b, &prm);
    }

    const size_t acc_col_bias_size = kai_get_dst_size_reduce_add_scale_reordered(N);
    std::vector<std::byte> acc_col_bias(acc_col_bias_size);
    {
      const kai_reduce_shape_t s{ .height = N, .width = K };
      const int32_t dst_scale = -static_cast<int32_t>(p.ZeroPointA);
      const int32_t dst_bias  = -static_cast<int32_t>(K) * static_cast<int32_t>(*p.ZeroPointB);
      const kai_reduce_buffers_t b{
        .src = rhs_reordered,        .src_stride = 0,
        .dst = acc_col_bias.data(),  .dst_stride = 0,
        .dst_scale = &dst_scale,     .dst_scale_stride = 0,
        .dst_bias  = &dst_bias,      .dst_bias_stride  = 0
      };
      const kai_reduce_params_t prm{ .flags = 0 };
      kai_run_reduce_add_scale_reordered(&s, &b, &prm);
    }

    // --- GEMM: choose int→float or int32 path based on p.Scale ---
    const bool to_float = (p.Scale != nullptr);

    if (to_float) {
      const size_t lhs_ps = kai_get_lhs_packed_stride_matmul_integer_to_float(K);
      const size_t rhs_ps = kai_get_rhs_packed_stride_matmul_integer_to_float(K);

      static const float kOne = 1.0f;
      const float* scale = p.Scale ? (p.PerColumnScale ? (p.Scale + p.RightScaleOffset) : p.Scale)
                             : &kOne;

      static const float kNegInf = -INFINITY;
      static const float kPosInf =  INFINITY;
      const float* fmin = p.ClampMin ? p.ClampMin : &kNegInf;
      const float* fmax = p.ClampMax ? p.ClampMax : &kPosInf;
      static thread_local std::vector<float> tls_zero_bias;
      const float* col_bias_f32 = p.Bias;

      if (!col_bias_f32) {
        // make sure we have at least N zeros for this thread
        if (tls_zero_bias.size() < N) tls_zero_bias.assign(N, 0.0f);
        col_bias_f32 = tls_zero_bias.data();
      }

      const kai_matmul_shape_t s_mm{ .m = M, .n = N, .k = K };
      const kai_matmul_buffers_t b_mm{
        .lhs = lhs_reordered.data(),
        .lhs_stride = lhs_ps,
        .rhs = rhs_reordered,
        .rhs_stride = rhs_ps,
        .acc_row_bias = acc_row_bias.data(),
        .acc_col_bias = acc_col_bias.data(),
        .scale = scale,
        .col_bias = col_bias_f32,
        .dst = static_cast<void*>(reinterpret_cast<float*>(p.C)),
        .dst_stride = p.ldc * sizeof(float),
        .acc = nullptr,
        .min = fmin, .max = fmax,
      };
      const kai_matmul_params_t prm_mm{ .flags = 0 };
      kai_run_matmul_integer_to_float(&s_mm, &b_mm, &prm_mm);
    } else {
      const size_t lhs_ps = kai_get_lhs_packed_stride_matmul_integer(K);
      const size_t rhs_ps = kai_get_rhs_packed_stride_matmul_integer(K);

      const kai_matmul_shape_t s_mm{ .m = M, .n = N, .k = K };
      const kai_matmul_buffers_t b_mm{
        .lhs = lhs_reordered.data(), .lhs_stride = lhs_ps,
        .rhs = rhs_reordered,        .rhs_stride = rhs_ps,
        .acc_row_bias = acc_row_bias.data(),
        .acc_col_bias = acc_col_bias.data(),
        .scale = nullptr,
        .col_bias = nullptr,
        .dst = static_cast<void*>(p.C),
        .dst_stride = p.ldc * sizeof(int32_t),
        .acc = nullptr,
        .min = nullptr, .max = nullptr,
      };
      const kai_matmul_params_t prm_mm{ .flags = 0 };
      kai_run_matmul_integer(&s_mm, &b_mm, &prm_mm);
    }
  }

  return true;
}
