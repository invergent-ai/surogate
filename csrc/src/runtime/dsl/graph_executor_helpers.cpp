// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Helper functions for DSL Graph executor.

#include "runtime/dsl/graph_executor_helpers.h"

#include "kernels/kernels.h"
#include "runtime/core/fp8_scaling_state.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/utils.h"

namespace dsl {

bool allow_quant_layer(const RuntimeOptions& options, const ::modules::ModelConfig& config, int layer_idx) {
    if (layer_idx < 0) return false;
    const int skip_first = options.RecipeOptions.skip_quant_first_layers;
    const int skip_last = options.RecipeOptions.skip_quant_last_layers;
    if (skip_first > 0 && layer_idx < skip_first) return false;
    if (skip_last > 0 && layer_idx >= static_cast<int>(config.NumLayers) - skip_last) return false;
    return true;
}

Tensor* fp8_forward_buffer(DslRunState& rs, ::modules::MatmulOp op) {
    if (!rs.has_fp8_forward()) return nullptr;
    auto& q = rs.fp8_forward_quants();
    switch (op) {
        case ::modules::MatmulOp::QKV:
            return &q.ln1;
        case ::modules::MatmulOp::MLPUp:
            return &q.ln2;
        case ::modules::MatmulOp::AttnOut:
            return &q.att;
        case ::modules::MatmulOp::MLPDown:
            return &q.swiglu;
        default:
            return nullptr;
    }
}

Tensor* fp8_grad_buffer(DslRunState& rs, ::modules::MatmulOp op) {
    if (!rs.has_fp8_hybrid_backward()) return nullptr;
    auto& q = rs.simplified_quant_grads();
    switch (op) {
        case ::modules::MatmulOp::QKV:
            return &q.d_qkv;
        case ::modules::MatmulOp::MLPUp:
            return &q.d_mlp_up;
        case ::modules::MatmulOp::AttnOut:
            return &q.d_res_att;
        case ::modules::MatmulOp::MLPDown:
            return &q.d_res_ffn;
        default:
            return nullptr;
    }
}

int fp8_quantizer_index(const DslRunState& rs, ::modules::MatmulOp op, int layer_idx) {
    if (!rs.has_fp8_delayed_scaling()) return -1;
    switch (op) {
        case ::modules::MatmulOp::QKV:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_LN1);
        case ::modules::MatmulOp::MLPUp:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_LN2);
        case ::modules::MatmulOp::AttnOut:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_ATT);
        case ::modules::MatmulOp::MLPDown:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_SWIGLU);
        default:
            return -1;
    }
}

void add_bias_tensor(Tensor& out, const Tensor& bias, int B, int T, int OC, cudaStream_t stream) {
    if (out.DType != bias.DType) {
        throw std::runtime_error("DSL graph executor: bias_add dtype mismatch");
    }
    if (out.DType == ETensorDType::BF16) {
        add_bias(out.get<nv_bfloat16>(), bias.get<nv_bfloat16>(), B, T, OC, stream);
        return;
    }
    if (out.DType == ETensorDType::FP32) {
        add_bias(out.get<float>(), bias.get<float>(), B, T, OC, stream);
        return;
    }
    throw std::runtime_error("DSL graph executor: bias_add unsupported dtype");
}

void reduce_loss(DslRunState& rs, long B, long T, NCCLCommunicator& comm) {
    deterministic_sum(rs.Losses.template get<float>(), rs.Losses.template get<float>(), B * T, rs.MainStream);
    comm.reduce_loss(rs.Losses.template get<float>(), rs.MainStream);
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    const bool capturing = (cudaStreamIsCapturing(rs.MainStream, &status) == cudaSuccess &&
                            status != cudaStreamCaptureStatusNone);
    if (!capturing) {
        CUDA_CHECK(cudaMemcpyAsync(rs.LossHost, rs.Losses.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost, rs.MainStream));
    }
}

Tensor recompute_lora_rmsnorm(::modules::LoRARunState& lora_rs, const Tensor& residual, const Tensor& weight,
                              float eps, int B, int T, int C, cudaStream_t stream) {
    if (!lora_rs.recompute_ln.Data || !lora_rs.recompute_rstd.Data) {
        throw std::runtime_error("DSL graph executor: LoRA recompute buffers not allocated");
    }
    rmsnorm_forward(lora_rs.recompute_ln, lora_rs.recompute_rstd,
                    residual, weight, nullptr, eps, B, T, C, stream);
    return lora_rs.recompute_ln;
}

}  // namespace dsl
