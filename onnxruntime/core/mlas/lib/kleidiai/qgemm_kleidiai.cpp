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

bool
MLASCALL
ArmKleidiAI::MlasGemmBatch(
    const MLAS_GEMM_QUANT_SHAPE_PARAMS& Shape,
    const MLAS_GEMM_QUANT_DATA_PARAMS* DataParams,
    const size_t BatchN,
    MLAS_THREADPOOL* /*ThreadPool*/
) {
  const size_t m = Shape.M, n = Shape.N, k = Shape.K;

  // Use KAI only for the integer->float path we implement here.
  for (size_t i = 0; i < BatchN; ++i) {
    const auto& p = DataParams[i];
    if (p.BIsPacked)           return false;
    if (p.PerColumnZeroPoints) return false; // Unsupported in the KAI kernel
    if (p.PerColumnScale) return false; // Unsupported in the KAI kernel
    if (p.Scale == nullptr)    return false; // integer output path -> MLAS
  }

  for (size_t i = 0; i < BatchN; ++i) {
    const auto& p = DataParams[i];

    // Inputs/strides
    const void*  lhs_qdata  = static_cast<const void*>(p.A);
    const size_t lhs_stride = p.lda * sizeof(uint8_t);
    const void*  rhs_qdata  = static_cast<const void*>(p.B);
    const size_t rhs_stride = p.ldb * sizeof(uint8_t);

    // True (signed/unsigned) zero-points as int32
    const int32_t zpA = Shape.AIsSigned ? (int32_t)(int8_t)p.ZeroPointA : (int32_t)p.ZeroPointA;
    const uint8_t zpB_u8 = p.ZeroPointB ? *p.ZeroPointB : 0;
    const int32_t zpB = Shape.BIsSigned ? (int32_t)(int8_t)zpB_u8 : (int32_t)zpB_u8;

    // Output (float)
    void*        dst          = static_cast<void*>(reinterpret_cast<float*>(p.C));
    const size_t dst_stride_f = p.ldc * sizeof(float);
    const float* scale        = p.PerColumnScale ? (p.Scale + p.RightScaleOffset) : p.Scale;

    static thread_local std::vector<float> zero_bias;

    // Reorder A (m x k)
    const size_t lhs_reordered_size = kai_get_dst_size_reorder(m, k);
    std::vector<std::byte> lhs_reordered(lhs_reordered_size);
    {
      const kai_reorder_shape_t  s{ .height = m, .width = k };
      const kai_reorder_buffers_t b{ .src = lhs_qdata, .src_stride = lhs_stride, .dst = lhs_reordered.data() };
      const kai_reorder_params_t prm{ .flags = 0 };
      kai_run_reorder(&s, &b, &prm);
    }

    // Reorder/transpose B (k x n)
    const size_t rhs_reordered_size = kai_get_dst_size_reorder_transpose(k, n);
    std::vector<std::byte> rhs_reordered(rhs_reordered_size);
    {
      const kai_reorder_shape_t  s{ .height = k, .width = n };
      const kai_reorder_buffers_t b{ .src = rhs_qdata, .src_stride = rhs_stride, .dst = rhs_reordered.data() };
      const kai_reorder_params_t prm{ .flags = 0 };
      kai_run_reorder_transpose(&s, &b, &prm);
    }

    // Map signed tensors into u8 by adding 128 in-place (so later sums are ΣA' / ΣB')
    auto add128_inplace = [](std::byte* buf, size_t bytes) {
      auto* p8 = reinterpret_cast<uint8_t*>(buf);
      for (size_t t = 0; t < bytes; ++t) p8[t] = static_cast<uint8_t>(p8[t] + 128);
    };
    if (Shape.AIsSigned) add128_inplace(lhs_reordered.data(), lhs_reordered.size());
    if (Shape.BIsSigned) add128_inplace(rhs_reordered.data(), rhs_reordered.size());

    // --- compute row/col sums in u8 domain (after +128 fixups) ---
    const size_t sum_row_size = kai_get_dst_size_reduce_add_scale_reordered(m);
    std::vector<std::byte> sum_row(sum_row_size);
    {
      const kai_reduce_shape_t s{ .height = m, .width = k };
      const int32_t one = 1, zero = 0;
      const kai_reduce_buffers_t b{
        .src = lhs_reordered.data(), .src_stride = 0,
        .dst = sum_row.data(),       .dst_stride = 0,
        .dst_scale = &one,           .dst_scale_stride = 0,
        .dst_bias  = &zero,          .dst_bias_stride  = 0
      };
      const kai_reduce_params_t prm{ .flags = 0 };
      kai_run_reduce_add_scale_reordered(&s, &b, &prm);
    }

    const size_t sum_col_size = kai_get_dst_size_reduce_add_scale_reordered(n);
    std::vector<std::byte> sum_col(sum_col_size);
    {
      const kai_reduce_shape_t s{ .height = n, .width = k };
      const int32_t one = 1, zero = 0;
      const kai_reduce_buffers_t b{
        .src = rhs_reordered.data(), .src_stride = 0,
        .dst = sum_col.data(),       .dst_stride = 0,
        .dst_scale = &one,           .dst_scale_stride = 0,
        .dst_bias  = &zero,          .dst_bias_stride  = 0
      };
      const kai_reduce_params_t prm{ .flags = 0 };
      kai_run_reduce_add_scale_reordered(&s, &b, &prm);
    }

    // --- pre-scale INT32 corrections (what the kernel expects in acc_*_bias) ---
    const int32_t zpA_u = zpA + (Shape.AIsSigned ? 128 : 0);
    const int32_t zpB_u = zpB + (Shape.BIsSigned ? 128 : 0);

    std::vector<int32_t> row_bias_i32(m);
    {
      const int32_t* sr = reinterpret_cast<const int32_t*>(sum_row.data());
      for (size_t i = 0; i < m; ++i) {
        // (-ZPB′) * sum_row(A′)
        row_bias_i32[i] = -zpB_u * sr[i];
      }
    }

    std::vector<int32_t> col_bias_i32(n);
    {
      const int32_t* sc = reinterpret_cast<const int32_t*>(sum_col.data());
      const int64_t k_const = (int64_t)k * (int64_t)zpA_u * (int64_t)zpB_u;
      for (size_t j = 0; j < n; ++j) {
        // (-ZPA′) * sum_col(B′) + K * ZPA′ * ZPB′
        const int64_t t = (int64_t)(-zpA_u) * (int64_t)sc[j] + k_const;
        col_bias_i32[j] = (int32_t)t;
      }
    }

    // --- scalar scale; pass model bias (post-scale float) via col_bias ---
    const float scale_scalar = *scale;          // per-tensor only (you gated per-column)
    static thread_local std::vector<float> zero_bias_f;
    if (!p.Bias) { zero_bias_f.assign(n, 0.0f); }

    const size_t lhs_ps = kai_get_lhs_packed_stride_matmul_integer_to_float(k);
    const size_t rhs_ps = kai_get_rhs_packed_stride_matmul_integer_to_float(k);

    float neg_inf = -std::numeric_limits<float>::infinity();
    float pos_inf =  std::numeric_limits<float>::infinity();
    const float* fmin = p.ClampMin ? p.ClampMin : &neg_inf;
    const float* fmax = p.ClampMax ? p.ClampMax : &pos_inf;

    const kai_matmul_shape_t  s_mm{ .m = m, .n = n, .k = k };
    const kai_matmul_buffers_t b_mm{
      .lhs = lhs_reordered.data(), .lhs_stride = lhs_ps,
      .rhs = rhs_reordered.data(), .rhs_stride = rhs_ps,

      // PRE-SCALE int32 corrections:
      .acc_row_bias = row_bias_i32.data(),
      .acc_col_bias = col_bias_i32.data(),

      // POST-SCALE:
      .scale = &scale_scalar,
      .col_bias = p.Bias ? p.Bias : zero_bias_f.data(),  // add model bias after scaling

      .dst = dst,
      .dst_stride = dst_stride_f,
      .acc = nullptr,
      .min = fmin,
      .max = fmax,
    };
    const kai_matmul_params_t prm_mm{ .flags = 0 };
    kai_run_matmul_integer_to_float(&s_mm, &b_mm, &prm_mm);
  }
  return true;
}
