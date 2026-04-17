// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_RECIPES_NVFP4_NVFP4_MATMUL_H
#define SUROGATE_SRC_RECIPES_NVFP4_NVFP4_MATMUL_H

#include <optional>
#include <stdexcept>
#include <cuda_runtime.h>

#include "kernels/kernels.h"
#include "utilities/tensor.h"

namespace recipes::nvfp4 {

/**
 * @brief Helper to compute FP4 block scale tensor shape for cuDNN F8_128x4 layout
 */
inline std::pair<long, long> fp4_scale_shape(int rows, int cols) {
    const long scale_rows = ((rows + 127) / 128) * 128;
    const long scale_cols = (((cols + 15) / 16) + 3) / 4 * 4;
    return {scale_rows, scale_cols};
}

/**
 * @brief Helper to compute weight scale tensor shape for cuDNN column-major layout
 */
inline std::pair<long, long> fp4_weight_scale_shape(int N, int K) {
    const long scale_rows = ((K + 15) / 16 + 3) / 4 * 4;
    const long scale_cols = (N + 127) / 128 * 128;
    return {scale_rows, scale_cols};
}

/**
 * @brief NVFP4 forward matmul using cuDNN backend
 *
 * Quantizes input and weight to FP4 E2M1 with two-level block scaling
 * and performs matmul via cuDNN frontend.
 *
 * FP4 E2M1 format:
 * - Values: ±{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
 * - Storage: 2 values packed per byte
 * - Two-level scaling: FP8 E4M3 block scales (per 16 values) + FP32 global amax
 *
 * @tparam RunState Run state type providing handles and temp allocation
 * @param out Output tensor (BF16)
 * @param inp Input tensor (BF16)
 * @param inp_fp4_data FP4 buffer for quantized input data
 * @param inp_fp4_scales FP8 block scales for input
 * @param inp_global_amax Device pointer to global amax for input
 * @param weight Weight tensor (BF16)
 * @param bias Optional bias tensor
 * @param hadamard_ws Hadamard transform workspace (unused per current TE recipe)
 * @param rs Run state containing cuDNN handle and workspace
 * @param B Batch size
 * @param T Sequence length
 * @param C Input channels (K dimension)
 * @param OC Output channels (N dimension)
 * @param use_hadamard Whether to apply RHT before quantization (unused per TE recipe)
 * @param seed Random seed for stochastic operations
 * @param stream CUDA stream
 */
template <typename RunState>
inline void forward_matmul(Tensor& out,
                           Tensor& inp,
                           Tensor& inp_fp4_data,
                           Tensor& inp_fp4_scales,
                           float* inp_global_amax,
                           Tensor& weight,
                           std::optional<Tensor> bias,
                           Tensor& hadamard_ws,
                           RunState& rs,
                           int B,
                           int T,
                           int C,
                           int OC,
                           bool use_hadamard,
                           unsigned int seed,
                           cudaStream_t stream) {
    if (!inp_fp4_data.Data) {
        throw std::runtime_error("nvfp4::forward_matmul: FP4 forward buffer not allocated");
    }

    const int BT = B * T;
    constexpr int FP4_BLOCK_SIZE = 16;

    // NOTE: TransformerEngine's NVFP4 recipe currently applies RHT only for "column-wise usage"
    // (primarily to support wgrad GEMM). For forward fprop GEMMs, keep inputs/weights untransformed.
    (void)use_hadamard;
    (void)hadamard_ws;

    // Step 1: Quantize input to FP4 with two-level block scaling (auto scale)
    quantize_fp4_block_auto_scale(inp_fp4_data.template get<uint8_t>(),
                                  inp_fp4_scales.template get<__nv_fp8_e4m3>(),
                                  inp_global_amax,
                                  inp.template get<nv_bfloat16>(),
                                  BT,
                                  C,
                                  rs.DeviceProp,
                                  stream);

    // Step 2: Quantize weight to FP4 with column-major scale layout
    auto [w_scale_rows, w_scale_cols] = fp4_weight_scale_shape(OC, C);

    Tensor weight_fp4_data = rs.temp_alloc(ETensorDType::BYTE, {(long)OC, (long)(C / 2)}, "weight_fp4_data");
    Tensor weight_fp4_scales = rs.temp_alloc(ETensorDType::FP8_E4M3, {w_scale_rows, w_scale_cols}, "weight_fp4_scales");
    Tensor weight_global_amax = rs.temp_alloc(ETensorDType::FP32, {1}, "weight_global_amax");

    quantize_fp4_weight_2d_auto_scale(weight_fp4_data.template get<uint8_t>(),
                                      weight_fp4_scales.template get<__nv_fp8_e4m3>(),
                                      weight_global_amax.template get<float>(),
                                      weight.template get<nv_bfloat16>(),
                                      OC,
                                      C,
                                      rs.DeviceProp,
                                      stream);

    // Step 3: Execute FP4 matmul via cuDNN with FP32 output for numerical stability
    Tensor out_f32{};
    float* out_f32_ptr = nullptr;
    if (out.DType == ETensorDType::BF16) {
        out_f32 = rs.temp_alloc(ETensorDType::FP32, {(long)BT, (long)OC}, "out_f32");
        out_f32_ptr = out_f32.template get<float>();
    } else if (out.DType == ETensorDType::FP32) {
        out_f32_ptr = out.template get<float>();
    } else {
        throw std::runtime_error("nvfp4::forward_matmul: output must be BF16 or FP32");
    }

    fp4_matmul_f32(out_f32_ptr,
                   inp_fp4_data.template get<uint8_t>(),
                   weight_fp4_data.template get<uint8_t>(),
                   inp_fp4_scales.template get<__nv_fp8_e4m3>(),
                   weight_fp4_scales.template get<__nv_fp8_e4m3>(),
                   1.0f,
                   1.0f,
                   rs.CuBlasWorkspace.template get<std::byte>(),
                   rs.CuBlasWorkspace.bytes(),
                   BT,
                   OC,
                   C,
                   FP4_BLOCK_SIZE,
                   rs.CudnnHandle,
                   stream);

    // Step 4: Apply alpha scaling correction
    fp4_alpha_scale(out_f32_ptr,
                    inp_global_amax,
                    weight_global_amax.template get<float>(),
                    (long)BT * OC,
                    rs.DeviceProp,
                    stream);

    // Step 5: Cast to BF16 if needed
    if (out.DType == ETensorDType::BF16) {
        convert_dtype(out.template get<nv_bfloat16>(), out_f32_ptr, (std::size_t)BT * (std::size_t)OC, stream);
        rs.temp_free(out_f32);
    }

    // Step 6: Add bias if present
    if (bias.has_value()) {
        if (out.DType != ETensorDType::BF16) {
            throw std::runtime_error("nvfp4::forward_matmul: bias add currently requires BF16 output");
        }
        add_bias(out.template get<nv_bfloat16>(), bias->template get<nv_bfloat16>(), B, T, OC, stream);
    }

    rs.temp_free(weight_global_amax);
    rs.temp_free(weight_fp4_scales);
    rs.temp_free(weight_fp4_data);
}

/**
 * @brief NVFP4 backward matmul using cuDNN backend
 *
 * Computes gradients using FP4 E2M1 with stochastic rounding:
 * - dinp = dout @ W
 * - dweight = inp^T @ dout (with RHT for column-wise usage)
 *
 * @tparam RunState Run state type providing handles and temp allocation
 * @param dinp Output: gradient w.r.t. input activation
 * @param dweight Output: gradient w.r.t. weight (accumulated)
 * @param dbias Optional output: gradient w.r.t. bias
 * @param dout Input: upstream gradient (BF16)
 * @param inp Input activation (BF16, from forward pass)
 * @param weight Weight tensor (BF16)
 * @param bias_buffer Optional buffer for bias gradient computation
 * @param accumulate_gradient Whether to accumulate into dweight
 * @param use_hadamard Whether to apply RHT for wgrad GEMM
 * @param rs Run state with cuDNN handles
 * @param B Batch size
 * @param T Sequence length
 * @param C Input channels
 * @param OC Output channels
 * @param seed Random seed for stochastic rounding
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
                            bool use_hadamard,
                            RunState& rs,
                            int B,
                            int T,
                            int C,
                            int OC,
                            unsigned int seed,
                            cudaStream_t stream,
                            bool skip_weight_grad = false) {
    if (inp.DType != ETensorDType::BF16 || weight.DType != ETensorDType::BF16 || dout.DType != ETensorDType::BF16) {
        throw std::runtime_error("nvfp4::backward_matmul: inp/weight/dout must be BF16");
    }

    const int BT = B * T;
    constexpr int FP4_BLOCK_SIZE = 16;

    const unsigned int sr_seed = seed ^ 0x9E3779B9u;
    const unsigned int rht_seed = seed ^ 0xA5A5A5A5u;

    // =========================================================================
    // dinp = dout @ W
    // =========================================================================
    {
        // Transpose weight: (OC, C) -> (C, OC)
        Tensor weight_tp = rs.temp_alloc(ETensorDType::BF16, {(long)C, (long)OC}, "weight_tp");
        transpose(weight_tp, weight, OC, C, stream);

        // Quantize dout (A) with stochastic rounding
        auto [a_scale_rows, a_scale_cols] = fp4_scale_shape(BT, OC);
        Tensor dout_fp4 = rs.temp_alloc(ETensorDType::BYTE, {(long)BT, (long)(OC / 2)}, "dout_fp4");
        Tensor dout_scales = rs.temp_alloc(ETensorDType::FP8_E4M3, {a_scale_rows, a_scale_cols}, "dout_scales");
        Tensor dout_amax = rs.temp_alloc(ETensorDType::FP32, {1}, "dout_amax");
        quantize_fp4_block_stochastic_auto_scale(dout_fp4.template get<uint8_t>(),
                                                 dout_scales.template get<__nv_fp8_e4m3>(),
                                                 dout_amax.template get<float>(),
                                                 dout.template get<nv_bfloat16>(),
                                                 BT,
                                                 OC,
                                                 sr_seed,
                                                 rs.DeviceProp,
                                                 stream);

        // Quantize W^T with 16x16 weight scaling
        auto [w_scale_rows, w_scale_cols] = fp4_weight_scale_shape(C, OC);
        Tensor w_fp4 = rs.temp_alloc(ETensorDType::BYTE, {(long)C, (long)(OC / 2)}, "w_fp4");
        Tensor w_scales = rs.temp_alloc(ETensorDType::FP8_E4M3, {w_scale_rows, w_scale_cols}, "w_scales");
        Tensor w_amax = rs.temp_alloc(ETensorDType::FP32, {1}, "w_amax");
        quantize_fp4_weight_2d_auto_scale(w_fp4.template get<uint8_t>(),
                                          w_scales.template get<__nv_fp8_e4m3>(),
                                          w_amax.template get<float>(),
                                          weight_tp.template get<nv_bfloat16>(),
                                          C,
                                          OC,
                                          rs.DeviceProp,
                                          stream);

        // Matmul output in FP32
        Tensor dinp_f32{};
        float* dinp_f32_ptr = nullptr;
        if (dinp.DType == ETensorDType::BF16) {
            dinp_f32 = rs.temp_alloc(ETensorDType::FP32, {(long)BT, (long)C}, "dinp_f32");
            dinp_f32_ptr = dinp_f32.template get<float>();
        } else if (dinp.DType == ETensorDType::FP32) {
            dinp_f32_ptr = dinp.template get<float>();
        } else {
            throw std::runtime_error("nvfp4::backward_matmul: dinp must be BF16 or FP32");
        }

        fp4_matmul_f32(dinp_f32_ptr,
                       dout_fp4.template get<uint8_t>(),
                       w_fp4.template get<uint8_t>(),
                       dout_scales.template get<__nv_fp8_e4m3>(),
                       w_scales.template get<__nv_fp8_e4m3>(),
                       1.0f,
                       1.0f,
                       rs.CuBlasWorkspace.template get<std::byte>(),
                       rs.CuBlasWorkspace.bytes(),
                       BT,
                       C,
                       OC,
                       FP4_BLOCK_SIZE,
                       rs.CudnnHandle,
                       stream);

        fp4_alpha_scale(dinp_f32_ptr,
                        dout_amax.template get<float>(),
                        w_amax.template get<float>(),
                        (long)BT * C,
                        rs.DeviceProp,
                        stream);

        if (dinp.DType == ETensorDType::BF16) {
            convert_dtype(dinp.template get<nv_bfloat16>(), dinp_f32_ptr, (std::size_t)BT * (std::size_t)C, stream);
            rs.temp_free(dinp_f32);
        }

        rs.temp_free(w_amax);
        rs.temp_free(w_scales);
        rs.temp_free(w_fp4);
        rs.temp_free(dout_amax);
        rs.temp_free(dout_scales);
        rs.temp_free(dout_fp4);
        rs.temp_free(weight_tp);
    }

    // =========================================================================
    // dW += dout^T @ inp
    // =========================================================================
    if (!skip_weight_grad) {
        // Transpose: A = dout^T (OC, BT), B = inp as (C, BT)
        Tensor dout_tp = rs.temp_alloc(ETensorDType::BF16, {(long)OC, (long)BT}, "dout_tp");
        Tensor inp_tp = rs.temp_alloc(ETensorDType::BF16, {(long)C, (long)BT}, "inp_tp");
        transpose(dout_tp, dout, BT, OC, stream);
        transpose(inp_tp, inp, BT, C, stream);

        Tensor dout_tp_r{};
        Tensor inp_tp_r{};
        const Tensor* q_dout_tp = &dout_tp;
        const Tensor* q_inp_tp = &inp_tp;

        if (use_hadamard) {
            if ((BT % 16) != 0) {
                throw std::runtime_error("nvfp4::backward_matmul: BT must be multiple of 16 when hadamard is enabled");
            }
            // Apply RHT only to the column-wise usage (BT is the column dimension)
            dout_tp_r = rs.temp_alloc(ETensorDType::BF16, {(long)OC, (long)BT}, "dout_tp_r");
            inp_tp_r = rs.temp_alloc(ETensorDType::BF16, {(long)C, (long)BT}, "inp_tp_r");
            hadamard_transform_forward(dout_tp_r.template get<nv_bfloat16>(),
                                       dout_tp.template get<nv_bfloat16>(),
                                       nullptr,
                                       OC,
                                       BT,
                                       rht_seed,
                                       stream);
            hadamard_transform_forward(inp_tp_r.template get<nv_bfloat16>(),
                                       inp_tp.template get<nv_bfloat16>(),
                                       nullptr,
                                       C,
                                       BT,
                                       rht_seed,
                                       stream);
            q_dout_tp = &dout_tp_r;
            q_inp_tp = &inp_tp_r;
        }

        // Quantize A (gradient) with stochastic rounding
        auto [a_scale_rows, a_scale_cols] = fp4_scale_shape(OC, BT);
        Tensor a_fp4 = rs.temp_alloc(ETensorDType::BYTE, {(long)OC, (long)(BT / 2)}, "a_fp4");
        Tensor a_scales = rs.temp_alloc(ETensorDType::FP8_E4M3, {a_scale_rows, a_scale_cols}, "a_scales");
        Tensor a_amax = rs.temp_alloc(ETensorDType::FP32, {1}, "a_amax");
        quantize_fp4_block_stochastic_auto_scale(a_fp4.template get<uint8_t>(),
                                                 a_scales.template get<__nv_fp8_e4m3>(),
                                                 a_amax.template get<float>(),
                                                 q_dout_tp->template get<nv_bfloat16>(),
                                                 OC,
                                                 BT,
                                                 sr_seed + 1,
                                                 rs.DeviceProp,
                                                 stream);

        // Quantize B (activation) with deterministic rounding
        auto [b_scale_rows, b_scale_cols] = fp4_weight_scale_shape(C, BT);
        Tensor b_fp4 = rs.temp_alloc(ETensorDType::BYTE, {(long)C, (long)(BT / 2)}, "b_fp4");
        Tensor b_scales = rs.temp_alloc(ETensorDType::FP8_E4M3, {b_scale_rows, b_scale_cols}, "b_scales");
        Tensor b_amax = rs.temp_alloc(ETensorDType::FP32, {1}, "b_amax");
        quantize_fp4_weight_auto_scale(b_fp4.template get<uint8_t>(),
                                       b_scales.template get<__nv_fp8_e4m3>(),
                                       b_amax.template get<float>(),
                                       q_inp_tp->template get<nv_bfloat16>(),
                                       C,
                                       BT,
                                       rs.DeviceProp,
                                       stream);

        // Matmul: (OC, BT) @ (BT, C) -> (OC, C)
        Tensor dw_f32 = rs.temp_alloc(ETensorDType::FP32, {(long)OC, (long)C}, "dw_f32");
        float* dw_f32_ptr = dw_f32.template get<float>();

        fp4_matmul_f32(dw_f32_ptr,
                       a_fp4.template get<uint8_t>(),
                       b_fp4.template get<uint8_t>(),
                       a_scales.template get<__nv_fp8_e4m3>(),
                       b_scales.template get<__nv_fp8_e4m3>(),
                       1.0f,
                       1.0f,
                       rs.CuBlasWorkspace.template get<std::byte>(),
                       rs.CuBlasWorkspace.bytes(),
                       OC,
                       C,
                       BT,
                       FP4_BLOCK_SIZE,
                       rs.CudnnHandle,
                       stream);

        fp4_alpha_scale(dw_f32_ptr,
                        a_amax.template get<float>(),
                        b_amax.template get<float>(),
                        (long)OC * C,
                        rs.DeviceProp,
                        stream);

        // Accumulate or replace dweight
        if (dweight.DType == ETensorDType::BF16) {
            if (!accumulate_gradient) {
                convert_dtype(dweight.template get<nv_bfloat16>(),
                              dw_f32_ptr,
                              (std::size_t)OC * (std::size_t)C,
                              stream);
            } else {
                Tensor dw_bf16 = rs.temp_alloc(ETensorDType::BF16, {(long)OC, (long)C}, "dw_bf16");
                convert_dtype(dw_bf16.template get<nv_bfloat16>(),
                              dw_f32_ptr,
                              (std::size_t)OC * (std::size_t)C,
                              stream);
                vector_add_sr(dweight.template get<nv_bfloat16>(),
                              dweight.template get<nv_bfloat16>(),
                              dw_bf16.template get<nv_bfloat16>(),
                              1.0f,
                              (long)OC * C,
                              seed,
                              stream);
                rs.temp_free(dw_bf16);
            }
        } else if (dweight.DType == ETensorDType::FP32) {
            if (!accumulate_gradient) {
                CUDA_CHECK(cudaMemcpyAsync(dweight.template get<float>(),
                                           dw_f32_ptr,
                                           sizeof(float) * (std::size_t)OC * (std::size_t)C,
                                           cudaMemcpyDeviceToDevice,
                                           stream));
            } else {
                vector_add_sr(dweight.template get<float>(),
                              dweight.template get<float>(),
                              dw_f32_ptr,
                              1.0f,
                              (long)OC * C,
                              seed,
                              stream);
            }
        } else {
            throw std::runtime_error("nvfp4::backward_matmul: dweight must be BF16 or FP32");
        }

        if (dbias.has_value()) {
            if (!bias_buffer.has_value()) {
                throw std::runtime_error("nvfp4::backward_matmul: dbias requested but bias_buffer not provided");
            }
            backward_bias(dbias.value(), dout, nullptr, nullptr, bias_buffer.value(), B, T, OC, rs.DeviceProp, stream);
        }

        rs.temp_free(dw_f32);
        rs.temp_free(b_amax);
        rs.temp_free(b_scales);
        rs.temp_free(b_fp4);
        rs.temp_free(a_amax);
        rs.temp_free(a_scales);
        rs.temp_free(a_fp4);
        if (inp_tp_r.Data) rs.temp_free(inp_tp_r);
        if (dout_tp_r.Data) rs.temp_free(dout_tp_r);
        rs.temp_free(inp_tp);
        rs.temp_free(dout_tp);
    }
}

}  // namespace recipes::nvfp4

#endif  // SUROGATE_SRC_RECIPES_NVFP4_NVFP4_MATMUL_H
