// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_RECIPES_BF16_BF16_MATMUL_H
#define SUROGATE_SRC_RECIPES_BF16_BF16_MATMUL_H

#include <optional>
#include <stdexcept>
#include <cuda_runtime.h>

#include "kernels/kernels.h"
#include "utilities/tensor.h"

namespace recipes::bf16 {

/**
 * @brief BF16 forward matmul (TN layout)
 *
 * Standard BF16 matmul without quantization. If input dtype differs from weight dtype,
 * quantizes input to match weight dtype before matmul.
 *
 * @tparam RunState Run state type providing cuBLAS handles and temp allocation
 * @param out Output tensor (BF16/FP32)
 * @param inp Input tensor (BF16)
 * @param inp_q Optional quantized input buffer (for matmul_dtype != activation_dtype)
 * @param weight Weight tensor (BF16 or FP8)
 * @param bias Optional bias tensor
 * @param rs Run state with cuBLAS handles
 * @param B Batch size
 * @param T Sequence length
 * @param C Input channels (K dimension)
 * @param OC Output channels (M dimension)
 * @param reuse_inp_quant Whether to reuse existing quantized input
 * @param stream CUDA stream
 */
template <typename RunState>
inline void forward_matmul(Tensor& out,
                           Tensor& inp,
                           Tensor& inp_q,
                           Tensor& weight,
                           std::optional<Tensor> bias,
                           RunState& rs,
                           int B,
                           int T,
                           int C,
                           int OC,
                           bool reuse_inp_quant,
                           cudaStream_t stream) {
    if (weight.DType == inp.DType) {
        // Same dtype - direct matmul
        matmul(out,
               weight,
               inp,
               bias,
               nullptr,
               nullptr,
               rs.CublasLtHandle,
               rs.CuBlasWorkspace,
               OC,
               B * T,
               C,
               EMMTranspose::TN,
               /*accumulate=*/false,
               stream);
        return;
    }

    // Different dtypes - need to quantize input
    if (!inp_q.Data) {
        throw std::runtime_error("bf16::forward_matmul: activation quant buffer not allocated");
    }

    if (!reuse_inp_quant) {
        if (!inp_q.abs_max()) {
            throw std::runtime_error("bf16::forward_matmul: quant buffer missing abs_max/scale Stats");
        }
        quantize_with_abs_max(inp_q, inp_q.scale(), inp, inp_q.abs_max(), (long)B * T * C, rs.DeviceProp, stream);
    }

    if (weight.DType == ETensorDType::BF16) {
        matmul(out,
               weight,
               inp_q,
               bias,
               nullptr,
               nullptr,
               rs.CublasLtHandle,
               rs.CuBlasWorkspace,
               OC,
               B * T,
               C,
               EMMTranspose::TN,
               /*accumulate=*/false,
               stream);
    } else {
        // FP8 weight with scale
        matmul(out,
               weight,
               inp_q,
               bias,
               weight.scale(),
               inp_q.scale(),
               rs.CublasLtHandle,
               rs.CuBlasWorkspace,
               OC,
               B * T,
               C,
               EMMTranspose::TN,
               /*accumulate=*/false,
               stream);
    }
}

/**
 * @brief BF16 backward matmul
 *
 * Standard backward matmul computing:
 * - dinp = dout @ W (gradient w.r.t. input)
 * - dweight = inp^T @ dout (gradient w.r.t. weight)
 * - dbias = sum(dout) (gradient w.r.t. bias)
 *
 * @tparam RunState Run state type providing cuBLAS handles and temp allocation
 * @param dinp Output: gradient w.r.t. input activation
 * @param dweight Output: gradient w.r.t. weight (accumulated if accumulate_gradient=true)
 * @param dbias Optional output: gradient w.r.t. bias
 * @param dout Input: upstream gradient (BF16)
 * @param inp Input activation (BF16, from forward pass)
 * @param weight Weight tensor (BF16)
 * @param bias_buffer Optional buffer for bias gradient computation
 * @param accumulate_gradient Whether to accumulate into dweight
 * @param rs Run state with cuBLAS handles
 * @param B Batch size
 * @param T Sequence length
 * @param C Input channels
 * @param OC Output channels
 * @param stream CUDA stream
 * @param skip_weight_grad Skip weight gradient computation (for LoRA-only mode)
 */
template <typename RunState>
inline void backward_matmul(Tensor& dinp,
                            Tensor& dweight,
                            std::optional<Tensor> dbias,
                            Tensor& dout,
                            Tensor& inp,
                            Tensor& weight,
                            std::optional<Tensor> bias_buffer,
                            bool accumulate_gradient,
                            RunState& rs,
                            int B,
                            int T,
                            int C,
                            int OC,
                            cudaStream_t stream,
                            bool skip_weight_grad = false) {
    const int BT = B * T;

    // Compute dinp = dout @ W
    // Need W transposed: (OC, C) -> (C, OC)
    Tensor weight_tp = rs.temp_alloc(weight.DType, {(long)C, (long)OC}, "weight_tp");
    transpose(weight_tp, weight, OC, C, stream);

    matmul(dinp,
           weight_tp,
           dout,
           std::nullopt,
           nullptr,
           nullptr,
           rs.CublasLtHandle,
           rs.CuBlasWorkspace,
           C,
           BT,
           OC,
           EMMTranspose::TN,
           /*accumulate=*/false,
           stream);

    rs.temp_free(weight_tp);

    // Compute dweight = inp^T @ dout
    if (!skip_weight_grad) {
        // Transpose input: (BT, C) -> (C, BT)
        Tensor inp_tp = rs.temp_alloc(inp.DType, {(long)C, (long)BT}, "inp_tp");
        transpose(inp_tp, inp, BT, C, stream);

        // Transpose dout: (BT, OC) -> (OC, BT)
        Tensor dout_tp = rs.temp_alloc(dout.DType, {(long)OC, (long)BT}, "dout_tp");
        transpose(dout_tp, dout, BT, OC, stream);

        // dweight (OC, C) = dout_tp (OC, BT) @ inp_tp^T (BT, C)
        // Using TN: dweight = dout_tp @ inp_tp with N transposed
        matmul(dweight,
               dout_tp,
               inp_tp,
               std::nullopt,
               nullptr,
               nullptr,
               rs.CublasLtHandle,
               rs.CuBlasWorkspace,
               OC,
               C,
               BT,
               EMMTranspose::TN,
               /*accumulate=*/accumulate_gradient,
               stream);

        rs.temp_free(dout_tp);
        rs.temp_free(inp_tp);

        // Compute dbias if needed
        if (dbias.has_value()) {
            backward_bias(dbias.value(), dout, B, T, OC, rs.DeviceProp, stream);
        }
    }
}

}  // namespace recipes::bf16

#endif  // SUROGATE_SRC_RECIPES_BF16_BF16_MATMUL_H
