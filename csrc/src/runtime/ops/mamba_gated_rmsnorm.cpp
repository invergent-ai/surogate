// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mamba gated RMSNorm operation dispatch.

#include "runtime/dsl/compiled_ops.h"

#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_mamba_gated_rmsnorm(const CompiledOp& op) {
    // Inputs: x [..., D], gate [..., D], weight [D]
    // Output: out [..., D]
    // Supports rank-3 [B, T, D] and rank-2 [N, D] (flattened linear-attention path).
    Tensor& x = resolve_tensor(op.inputs[0]);
    Tensor& gate = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);

    if (x.Rank != gate.Rank) {
        throw std::runtime_error("mamba_gated_rmsnorm: x/gate rank mismatch");
    }
    if (x.Rank != 2 && x.Rank != 3) {
        throw std::runtime_error("mamba_gated_rmsnorm: expected x rank 2 or 3");
    }

    const int B = static_cast<int>(x.Sizes[0]);
    const int T = (x.Rank == 3) ? static_cast<int>(x.Sizes[1]) : 1;
    const int D = static_cast<int>(x.Sizes[x.Rank - 1]);
    const long n = static_cast<long>(B) * T * D;
    const std::vector<long> x_shape(x.Sizes.begin(), x.Sizes.begin() + x.Rank);
    if (B <= 0 || T <= 0 || D <= 0 || n <= 0) {
        throw std::runtime_error("mamba_gated_rmsnorm: invalid dimensions B="
                                 + std::to_string(B) + " T=" + std::to_string(T)
                                 + " D=" + std::to_string(D));
    }

    // Get normalization parameters
    const float eps = op.attrs.eps;
    const int groups = op.attrs.n_groups > 0 ? op.attrs.n_groups : 1;
    const bool norm_before_gate = op.attrs.norm_before_gate;

    // 1. silu_gate = silu(gate)
    Tensor silu_gate = mRunState.temp_alloc(x.DType, x_shape);
    mTemps.push_back(silu_gate);
    silu_forward(silu_gate, gate, n, mRunState.MainStream);

    Tensor out_t = mRunState.temp_alloc(x.DType, x_shape);
    Tensor rstd = mRunState.temp_alloc(ETensorDType::FP32, {B * T, groups});
    mTemps.push_back(out_t);
    mTemps.push_back(rstd);

    Tensor normed_or_gated;
    if (norm_before_gate) {
        // Qwen3.5 style: out = RMSNorm(x) * silu(gate)
        Tensor normed = mRunState.temp_alloc(x.DType, x_shape);
        mTemps.push_back(normed);
        mamba_group_rmsnorm_forward(normed, rstd, x, weight, eps, B, T, D, groups, mRunState.MainStream);
        if (x.DType == ETensorDType::BF16) {
            elementwise_mul(out_t.get<nv_bfloat16>(), normed.get<nv_bfloat16>(), silu_gate.get<nv_bfloat16>(),
                            n, mRunState.MainStream);
        } else if (x.DType == ETensorDType::FP16) {
            elementwise_mul(out_t.get<half>(), normed.get<half>(), silu_gate.get<half>(),
                            n, mRunState.MainStream);
        } else {
            elementwise_mul(out_t.get<float>(), normed.get<float>(), silu_gate.get<float>(),
                            n, mRunState.MainStream);
        }
        normed_or_gated = normed;  // Save normalized x for backward.
    } else {
        // Mamba default: out = RMSNorm(x * silu(gate))
        Tensor gated = mRunState.temp_alloc(x.DType, x_shape);
        mTemps.push_back(gated);
        if (x.DType == ETensorDType::BF16) {
            elementwise_mul(gated.get<nv_bfloat16>(), x.get<nv_bfloat16>(), silu_gate.get<nv_bfloat16>(),
                            n, mRunState.MainStream);
        } else if (x.DType == ETensorDType::FP16) {
            elementwise_mul(gated.get<half>(), x.get<half>(), silu_gate.get<half>(),
                            n, mRunState.MainStream);
        } else {
            elementwise_mul(gated.get<float>(), x.get<float>(), silu_gate.get<float>(),
                            n, mRunState.MainStream);
        }
        mamba_group_rmsnorm_forward(out_t, rstd, gated, weight, eps, B, T, D, groups, mRunState.MainStream);
        normed_or_gated = gated;  // Save norm input for backward.
    }

    store_tensor(op.outputs[0], out_t);

    // Save rstd and gated for backward.
    // Must persist via cudaMalloc because temp_alloc'd stack memory is freed at
    // layer boundaries (Stack.restore), leaving dangling pointers in mSaved.
    if (mSaved) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        const bool capturing =
            (cudaStreamIsCapturing(mRunState.MainStream, &capture_status) == cudaSuccess &&
             capture_status != cudaStreamCaptureStatusNone);
        auto persist_save = [&](const std::string& name, const Tensor& src) {
            const size_t bytes = src.bytes();
            if (bytes == 0) return;
            auto buf_it = mMoeSavedBuffers.find(name);
            if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
                if (capturing) {
                    throw std::runtime_error(
                        "mamba_gated_rmsnorm: missing preallocated save buffer for '" + name +
                        "' during CUDA graph capture");
                }
                if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                    CUDA_CHECK(cudaFree(buf_it->second));
                }
                void* new_buffer = nullptr;
                CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                mMoeSavedBuffers[name] = new_buffer;
                mMoeSavedSizes[name] = bytes;
            }
            CUDA_CHECK(cudaMemcpyAsync(mMoeSavedBuffers[name], src.Data, bytes,
                                       cudaMemcpyDeviceToDevice, mRunState.MainStream));
            Tensor saved;
            saved.DType = src.DType;
            saved.Rank = src.Rank;
            for (int d = 0; d < src.Rank; ++d) saved.Sizes[d] = src.Sizes[d];
            saved.Data = static_cast<std::byte*>(mMoeSavedBuffers[name]);
            (*mSaved)[name] = saved;
        };

        persist_save(op.op_id + ".rstd", rstd);
        persist_save(op.op_id + ".normed", normed_or_gated);
    }
}

void CompiledExecutor::dispatch_mamba_gated_rmsnorm_backward(const CompiledOp& op) {
    // Inputs: d_out [..., D], x [..., D], gate [..., D], weight [D], rstd [B*T, G], gated [..., D]
    // Outputs: d_x [..., D], d_gate [..., D], d_weight [D]
    //
    // Saved input[5] is:
    // - norm_before_gate=False: gated = x * silu(gate) (norm input)
    // - norm_before_gate=True : normed = RMSNorm(x) (pre-gate normalized output)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& x = resolve_tensor(op.inputs[1]);
    Tensor& gate = resolve_tensor(op.inputs[2]);
    Tensor& weight = resolve_tensor(op.inputs[3]);
    Tensor& rstd = resolve_tensor(op.inputs[4]);
    Tensor& normed_or_gated = resolve_tensor(op.inputs[5]);

    if (x.Rank != gate.Rank || x.Rank != d_out.Rank) {
        throw std::runtime_error("mamba_gated_rmsnorm_backward: rank mismatch among x/gate/d_out");
    }
    if (x.Rank != 2 && x.Rank != 3) {
        throw std::runtime_error("mamba_gated_rmsnorm_backward: expected x rank 2 or 3");
    }

    const int B = static_cast<int>(x.Sizes[0]);
    const int T = (x.Rank == 3) ? static_cast<int>(x.Sizes[1]) : 1;
    const int D = static_cast<int>(x.Sizes[x.Rank - 1]);
    const long n = static_cast<long>(B) * T * D;
    const std::vector<long> x_shape(x.Sizes.begin(), x.Sizes.begin() + x.Rank);
    const int groups = op.attrs.n_groups > 0 ? op.attrs.n_groups : 1;
    const bool norm_before_gate = op.attrs.norm_before_gate;
    const bool dbg = []() {
        const char* env = std::getenv("SUROGATE_DEBUG_QWEN35_BWD");
        return env && std::string(env) == "1";
    }();

    if (dbg) {
        fprintf(stderr,
                "[QWEN35_BWD][mamba_gated_rmsnorm_backward] out0=%s out1=%s out2=%s\n",
                op.outputs.size() > 0 ? op.outputs[0].name.c_str() : "<none>",
                op.outputs.size() > 1 ? op.outputs[1].name.c_str() : "<none>",
                op.outputs.size() > 2 ? op.outputs[2].name.c_str() : "<none>");
    }

    // Common gate activation for both branches
    Tensor silu_gate = mRunState.temp_alloc(x.DType, x_shape);
    mTemps.push_back(silu_gate);
    silu_forward(silu_gate, gate, n, mRunState.MainStream);

    Tensor d_x = mRunState.temp_alloc(d_out.DType, x_shape);
    Tensor d_gate = mRunState.temp_alloc(d_out.DType, x_shape);
    Tensor d_weight_fp32 = mRunState.temp_alloc(ETensorDType::FP32, {D});
    mTemps.push_back(d_x);
    mTemps.push_back(d_gate);
    mTemps.push_back(d_weight_fp32);

    if (norm_before_gate) {
        // Forward: out = RMSNorm(x) * silu(gate), saved normed_or_gated = RMSNorm(x)
        // d_normed = d_out * silu(gate)
        Tensor d_normed = mRunState.temp_alloc(d_out.DType, x_shape);
        mTemps.push_back(d_normed);
        if (d_out.DType == ETensorDType::BF16) {
            elementwise_mul(d_normed.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(), silu_gate.get<nv_bfloat16>(),
                            n, mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP16) {
            elementwise_mul(d_normed.get<half>(), d_out.get<half>(), silu_gate.get<half>(),
                            n, mRunState.MainStream);
        } else {
            elementwise_mul(d_normed.get<float>(), d_out.get<float>(), silu_gate.get<float>(),
                            n, mRunState.MainStream);
        }
        // d_x, d_weight via RMSNorm backward on x.
        mamba_group_rmsnorm_backward_dx(d_x, d_normed, x, weight, rstd, B, T, D, groups, mRunState.MainStream);
        mamba_group_rmsnorm_backward_dweight_fp32(
            d_weight_fp32, d_normed, x, rstd, B, T, D, groups, mRunState.MainStream);

        // d_gate = silu_backward(d_out * normed, gate)
        Tensor d_out_times_normed = mRunState.temp_alloc(d_out.DType, x_shape);
        mTemps.push_back(d_out_times_normed);
        if (d_out.DType == ETensorDType::BF16) {
            elementwise_mul(d_out_times_normed.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(),
                            normed_or_gated.get<nv_bfloat16>(), n, mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP16) {
            elementwise_mul(d_out_times_normed.get<half>(), d_out.get<half>(),
                            normed_or_gated.get<half>(), n, mRunState.MainStream);
        } else {
            elementwise_mul(d_out_times_normed.get<float>(), d_out.get<float>(),
                            normed_or_gated.get<float>(), n, mRunState.MainStream);
        }
        silu_backward(d_gate, gate, d_out_times_normed, n, mRunState.MainStream);
    } else {
        // Forward: out = RMSNorm(gated), gated = x * silu(gate), saved normed_or_gated = gated
        Tensor d_gated = mRunState.temp_alloc(d_out.DType, x_shape);
        mTemps.push_back(d_gated);
        mamba_group_rmsnorm_backward_dx(
            d_gated, d_out, normed_or_gated, weight, rstd, B, T, D, groups, mRunState.MainStream);
        mamba_group_rmsnorm_backward_dweight_fp32(
            d_weight_fp32, d_out, normed_or_gated, rstd, B, T, D, groups, mRunState.MainStream);

        // d_x = d_gated * silu(gate)
        if (d_out.DType == ETensorDType::BF16) {
            elementwise_mul(d_x.get<nv_bfloat16>(), d_gated.get<nv_bfloat16>(), silu_gate.get<nv_bfloat16>(),
                            n, mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP16) {
            elementwise_mul(d_x.get<half>(), d_gated.get<half>(), silu_gate.get<half>(),
                            n, mRunState.MainStream);
        } else {
            elementwise_mul(d_x.get<float>(), d_gated.get<float>(), silu_gate.get<float>(),
                            n, mRunState.MainStream);
        }

        // d_gate = silu_backward(d_gated * x, gate)
        Tensor d_gated_times_x = mRunState.temp_alloc(d_out.DType, x_shape);
        mTemps.push_back(d_gated_times_x);
        if (d_out.DType == ETensorDType::BF16) {
            elementwise_mul(d_gated_times_x.get<nv_bfloat16>(), d_gated.get<nv_bfloat16>(), x.get<nv_bfloat16>(),
                            n, mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP16) {
            elementwise_mul(d_gated_times_x.get<half>(), d_gated.get<half>(), x.get<half>(),
                            n, mRunState.MainStream);
        } else {
            elementwise_mul(d_gated_times_x.get<float>(), d_gated.get<float>(), x.get<float>(),
                            n, mRunState.MainStream);
        }
        silu_backward(d_gate, gate, d_gated_times_x, n, mRunState.MainStream);
    }

    auto copy_or_accumulate = [&](const TensorRef& out_ref, const Tensor& src, bool allow_accumulate) {
        if (out_ref.name.empty()) {
            return;
        }

        Tensor& dst_ref = ensure_output_tensor(out_ref);
        Tensor dst = dst_ref;
        if (dst.Rank != src.Rank || dst.nelem() != src.nelem()) {
            // Reallocate when shape doesn't match (happens in hybrid models where
            // different block types share tensor_ids but have different dimensions)
            std::vector<long> shape(src.Sizes.begin(), src.Sizes.begin() + src.Rank);
            dst = mRunState.temp_alloc(src.DType, shape);
            mTemps.push_back(dst);
        }

        bool accumulate = false;
        if (allow_accumulate) {
            accumulate = mAccumulateTensors.count(out_ref.name) > 0;
            if (!accumulate) {
                if (auto base = base_param_from_grad(out_ref.name)) {
                    accumulate = mAccumulateTensors.count("d_" + *base) > 0;
                }
            }
        }

        Tensor src_use = src;
        if (dst.DType != src_use.DType) {
            src_use = mRunState.temp_alloc(dst.DType, {static_cast<long>(src.nelem())});
            mTemps.push_back(src_use);
            if (dst.DType == ETensorDType::BF16 && src.DType == ETensorDType::FP32) {
                convert_dtype(src_use.get<nv_bfloat16>(), src.get<float>(), src.nelem(), mRunState.MainStream);
            } else if (dst.DType == ETensorDType::FP32 && src.DType == ETensorDType::BF16) {
                convert_dtype(src_use.get<float>(), src.get<nv_bfloat16>(), src.nelem(), mRunState.MainStream);
            } else {
                throw std::runtime_error("mamba_gated_rmsnorm_backward: unsupported dtype conversion for " + out_ref.name);
            }
            src_use = view_tensor(src_use, std::vector<long>(src.Sizes.begin(), src.Sizes.begin() + src.Rank));
        }

        if (dst.Data != src_use.Data) {
            if (accumulate) {
                vector_add_sr(dst, dst, src_use, 1.0f, static_cast<long>(dst.nelem()), 0, mRunState.MainStream);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dst.Data, src_use.Data, dst.bytes(),
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
            }
        }
        store_tensor(out_ref, dst);
    };

    if (op.outputs.size() > 0) {
        copy_or_accumulate(op.outputs[0], d_x, /*allow_accumulate=*/false);
    }
    if (op.outputs.size() > 1) {
        copy_or_accumulate(op.outputs[1], d_gate, /*allow_accumulate=*/false);
    }
    if (op.outputs.size() > 2) {
        copy_or_accumulate(op.outputs[2], d_weight_fp32, /*allow_accumulate=*/true);
    }

    if (dbg && op.outputs.size() > 2 && !op.outputs[2].name.empty()) {
        Tensor& d_weight_dbg = resolve_tensor(op.outputs[2]);
        fprintf(stderr, "[QWEN35_BWD][mamba_gated_rmsnorm_backward] d_weight_dtype=%d nelem=%zu\n",
                static_cast<int>(d_weight_dbg.DType), d_weight_dbg.nelem());
    }
}

}  // namespace dsl
