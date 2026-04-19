#include "runtime/executor/compiled_ops.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

namespace dsl {

Tensor CompiledExecutor::resolve_moe_expert_offsets(const CompiledOp& op) {
    Tensor expert_offsets_view;
    Tensor* expert_offsets_ptr = nullptr;
    int layer_idx_any = op.attrs.layer_idx;
    if (layer_idx_any < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("d_", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, layer_idx_any, field);
    }
    if (layer_idx_any >= 0) {
        // For the last MoE layer, prefer the global expert_offsets restored for backward.
        if (layer_idx_any == static_cast<int>(mConfig.NumLayers) - 1) {
            // Try flat vector first via pre-resolved offsets tensor ID, fall back to mirror
            Tensor* global_offsets = nullptr;
            if (op.attrs.moe_offsets_tensor_id >= 0 &&
                static_cast<std::size_t>(op.attrs.moe_offsets_tensor_id) < mTensors.size() &&
                mTensors[op.attrs.moe_offsets_tensor_id].Data) {
                global_offsets = &mTensors[op.attrs.moe_offsets_tensor_id];
            }
            // mTensors lookup is the only path (no mirror fallback)
            if (global_offsets && global_offsets->Data) {
                cudaPointerAttributes attr{};
                cudaError_t err = cudaPointerGetAttributes(&attr, global_offsets->Data);
                if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
                    return *global_offsets;
                }
                cudaGetLastError();
            }
        }
        const std::string key = moe_saved_key(layer_idx_any, "moe_expert_offsets");
        auto it_saved = mMoeSavedBuffers.find(key);
        std::string size_key = key;
        if (it_saved == mMoeSavedBuffers.end()) {
            const std::string legacy_key = "blocks[" + std::to_string(layer_idx_any) + "].moe_expert_offsets";
            it_saved = mMoeSavedBuffers.find(legacy_key);
            if (it_saved != mMoeSavedBuffers.end()) {
                size_key = legacy_key;
            }
        }
        if (it_saved != mMoeSavedBuffers.end() && it_saved->second != nullptr) {
            cudaPointerAttributes attr{};
            cudaError_t err = cudaPointerGetAttributes(&attr, it_saved->second);
            if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
                expert_offsets_view.DType = ETensorDType::INT32;
                expert_offsets_view.Rank = 1;
                // Use actual stored size (may be num_merged+1 when LLEP is active, not num_local+1)
                auto size_it = mMoeSavedSizes.find(size_key);
                if (size_it != mMoeSavedSizes.end()) {
                    expert_offsets_view.Sizes[0] = static_cast<long>(size_it->second / sizeof(int));
                } else {
                    expert_offsets_view.Sizes[0] = static_cast<long>(mConfig.NumLocalExperts + 1);
                }
                expert_offsets_view.Data = static_cast<std::byte*>(it_saved->second);
                expert_offsets_ptr = &expert_offsets_view;
            } else {
                cudaGetLastError();
            }
        }
    }
    if (!expert_offsets_ptr) {
        Tensor* moe_offsets_ptr = nullptr;
        if (op.attrs.moe_offsets_tensor_id >= 0 &&
            static_cast<std::size_t>(op.attrs.moe_offsets_tensor_id) < mTensors.size() &&
            mTensors[op.attrs.moe_offsets_tensor_id].Data) {
            moe_offsets_ptr = &mTensors[op.attrs.moe_offsets_tensor_id];
        }
        if (!moe_offsets_ptr) {
            throw std::runtime_error("moe_expert_bias_add: expert_offsets not found");
        }
        if (moe_offsets_ptr->Data) {
            cudaPointerAttributes attr{};
            cudaError_t err = cudaPointerGetAttributes(&attr, moe_offsets_ptr->Data);
            if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
                cudaGetLastError();
                throw std::runtime_error("moe_expert_bias_add: expert_offsets is not device memory");
            }
        }
        expert_offsets_ptr = moe_offsets_ptr;
    }
    if (expert_offsets_ptr->DType != ETensorDType::INT32) {
        throw std::runtime_error("moe_expert_bias_add: expert_offsets dtype is not INT32");
    }
    if (!expert_offsets_ptr->Data) {
        throw std::runtime_error("moe_expert_bias_add: expert_offsets has null data");
    }
    return *expert_offsets_ptr;
}

void CompiledExecutor::dispatch_moe_expert_bias_add(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& bias = resolve_tensor(op.inputs[1]);

    Tensor expert_offsets_tensor = resolve_moe_expert_offsets(op);
    const int* expert_offsets = expert_offsets_tensor.get<int>();

    const int hidden_size = static_cast<int>(inp.Sizes[1]);
    const int total_tokens = static_cast<int>(inp.Sizes[0]);

    // Parse layer index for LLEP lookup
    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) name.remove_prefix(6);
        std::string field;
        parse_block_param(name, layer_idx, field);
    }
    const int ep_key = ep_state_key(layer_idx);

    // Check if LLEP is active for this layer (merged experts != local experts)
    int num_experts = static_cast<int>(mConfig.NumLocalExperts);
    const ep::LLEPLayerState* llep = nullptr;
    if (layer_idx >= 0) {
        auto it = mLLEPStates.find(ep_key);
        if (it != mLLEPStates.end() && it->second.active) {
            llep = &it->second;
            num_experts = llep->num_merged_experts;
        }
    }

    std::vector<long> out_shape = {static_cast<long>(total_tokens), static_cast<long>(hidden_size)};
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape, "moe_expert_bias_add_out");
    mTemps.push_back(out);

    if (llep) {
        // LLEP mode: merged experts may be non-contiguous in the global expert space.
        // Build a merged bias tensor by gathering rows from the full bias tensor using
        // merged_to_global mapping. Every GPU has the full [E_total, hidden] bias via
        // Direct HF mapping (2D tensors are not EP-sharded).
        const size_t row_bytes = static_cast<size_t>(hidden_size) * get_dtype_size(bias.DType);
        Tensor merged_bias = mRunState.temp_alloc(bias.DType,
                                                  {static_cast<long>(num_experts), static_cast<long>(hidden_size)},
                                                  "moe_expert_bias_add_merged_bias");
        mTemps.push_back(merged_bias);
        for (int m = 0; m < num_experts; ++m) {
            const int global_e = llep->merged_to_global[m];
            const std::byte* src = static_cast<const std::byte*>(bias.Data) + static_cast<size_t>(global_e) * row_bytes;
            std::byte* dst = static_cast<std::byte*>(merged_bias.Data) + static_cast<size_t>(m) * row_bytes;
            CUDA_CHECK(cudaMemcpyAsync(dst, src, row_bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
        }

        if (inp.DType == ETensorDType::BF16) {
            moe_expert_bias_add_forward(out.get<nv_bfloat16>(),
                                        inp.get<nv_bfloat16>(),
                                        merged_bias.get<nv_bfloat16>(),
                                        expert_offsets,
                                        num_experts,
                                        hidden_size,
                                        total_tokens,
                                        mRunState.MainStream);
        } else if (inp.DType == ETensorDType::FP32) {
            moe_expert_bias_add_forward(out.get<float>(),
                                        inp.get<float>(),
                                        merged_bias.get<float>(),
                                        expert_offsets,
                                        num_experts,
                                        hidden_size,
                                        total_tokens,
                                        mRunState.MainStream);
        } else {
            throw std::logic_error("moe_expert_bias_add: unsupported input dtype");
        }
    } else {
        // Basic EP (or no EP): bias tensor contains all experts, offset to local range.
        const int ep_bias_offset =
            (mConfig.EPSize > 1 && mComm && mComm->ep_enabled()) ? mComm->ep_rank() * num_experts * hidden_size : 0;

        if (inp.DType == ETensorDType::BF16) {
            moe_expert_bias_add_forward(out.get<nv_bfloat16>(),
                                        inp.get<nv_bfloat16>(),
                                        bias.get<nv_bfloat16>() + ep_bias_offset,
                                        expert_offsets,
                                        num_experts,
                                        hidden_size,
                                        total_tokens,
                                        mRunState.MainStream);
        } else if (inp.DType == ETensorDType::FP32) {
            moe_expert_bias_add_forward(out.get<float>(),
                                        inp.get<float>(),
                                        bias.get<float>() + ep_bias_offset,
                                        expert_offsets,
                                        num_experts,
                                        hidden_size,
                                        total_tokens,
                                        mRunState.MainStream);
        } else {
            throw std::logic_error("moe_expert_bias_add: unsupported input dtype");
        }
    }

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_moe_expert_bias_add_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    // d_input = d_out (pass through)
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], d_out);
    }

    if (op.outputs.size() < 2 || op.outputs[1].name.empty()) {
        return;
    }

    // GPT-OSS uses per-expert bias tensors; skip backward to avoid unstable CUDA errors for now.
    if (op.outputs[1].name.find("experts_") != std::string::npos &&
        op.outputs[1].name.find("_bias") != std::string::npos) {
        if (mCurrentGraph) {
            if (auto base = base_param_from_grad_kind(op.outputs[1].tensor_id, *mCurrentGraph)) {
                bool grad_accum = false;
                if (Tensor* grad_tensor = mGrads.get_param_grad(*base, grad_accum)) {
                    if (grad_tensor->Data) {
                        store_tensor(op.outputs[1], *grad_tensor);
                    }
                }
            }
        }
        return;
    }

    const int num_experts = static_cast<int>(mConfig.NumLocalExperts);
    const int top_k = (mConfig.NumExpertsPerTok > 0) ? static_cast<int>(mConfig.NumExpertsPerTok) : 1;
    const int num_tokens = static_cast<int>(mB * mT);
    const int total_tokens = num_tokens * (top_k > 0 ? top_k : 1);
    if (total_tokens <= 0) {
        throw std::runtime_error("moe_expert_bias_add_backward: invalid total_tokens");
    }
    const long d_out_elems = static_cast<long>(d_out.nelem());
    if (d_out_elems % total_tokens != 0) {
        throw std::runtime_error("moe_expert_bias_add_backward: d_out shape mismatch vs total_tokens");
    }
    const int hidden_size = static_cast<int>(d_out_elems / total_tokens);

    if (mRunState.Stack.owns(d_out.Data) && !mRunState.Stack.is_live(d_out.Data)) {
        throw std::runtime_error("moe_expert_bias_add_backward: d_out points to dead stack memory");
    }

    Tensor expert_offsets_tensor = resolve_moe_expert_offsets(op);
    const int* expert_offsets = expert_offsets_tensor.get<int>();

    Tensor* grad_tensor = nullptr;
    bool grad_accum = false;
    if (!op.outputs[1].name.empty() && mCurrentGraph) {
        if (auto base = base_param_from_grad_kind(op.outputs[1].tensor_id, *mCurrentGraph)) {
            grad_tensor = mGrads.get_param_grad(*base, grad_accum);
        }
    }
    Tensor& d_bias = (grad_tensor && grad_tensor->Data) ? *grad_tensor : ensure_output_tensor(op.outputs[1]);
    if (grad_tensor && grad_tensor->Data) {
        store_tensor(op.outputs[1], d_bias);
    }
    bool accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
    if (!accumulate && grad_accum) {
        accumulate = true;
    }
    if (!accumulate && !op.outputs[1].name.empty() && mCurrentGraph) {
        if (auto base = base_param_from_grad_kind(op.outputs[1].tensor_id, *mCurrentGraph)) {
            accumulate = mAccumulateTensors.count("d_" + *base) > 0;
        }
    }

    const long expected_bias_elems = static_cast<long>(num_experts) * hidden_size;
    if (d_bias.nelem() != expected_bias_elems) {
        throw std::runtime_error("moe_expert_bias_add_backward: d_bias shape mismatch");
    }

    // Note: expert_offsets are produced by moe_permute and expected to be valid here.

    const bool need_temp = (d_bias.DType != ETensorDType::FP32) || accumulate;
    Tensor d_bias_f32;
    if (need_temp) {
        d_bias_f32 = mRunState.temp_alloc(ETensorDType::FP32,
                                          {static_cast<long>(num_experts), static_cast<long>(hidden_size)},
                                          "moe_expert_bias_add_d_bias_f32");
        mTemps.push_back(d_bias_f32);
    }

    float* d_bias_ptr = need_temp ? d_bias_f32.get<float>() : d_bias.get<float>();

    if (d_out.DType == ETensorDType::BF16) {
        moe_expert_bias_add_backward(nullptr,
                                     d_bias_ptr,
                                     d_out.get<nv_bfloat16>(),
                                     expert_offsets,
                                     num_experts,
                                     hidden_size,
                                     total_tokens,
                                     mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::FP32) {
        moe_expert_bias_add_backward(nullptr,
                                     d_bias_ptr,
                                     d_out.get<float>(),
                                     expert_offsets,
                                     num_experts,
                                     hidden_size,
                                     total_tokens,
                                     mRunState.MainStream);
    } else {
        throw std::logic_error("moe_expert_bias_add_backward: unsupported d_out dtype");
    }

    if (d_bias.DType == ETensorDType::FP32) {
        if (accumulate) {
            vector_add_sr(d_bias, d_bias, d_bias_f32, 1.0f, static_cast<long>(d_bias.nelem()), 0, mRunState.MainStream);
        }
        return;
    }

    // Convert to output dtype (BF16) and accumulate if needed.
    auto shape_vec = [](const Tensor& t) {
        return std::vector<long>(t.Sizes.begin(), t.Sizes.begin() + t.Rank);
    };
    Tensor d_bias_cast = mRunState.temp_alloc(d_bias.DType, shape_vec(d_bias), "moe_expert_bias_add_d_bias_cast");
    mTemps.push_back(d_bias_cast);
    convert_dtype(d_bias_cast.get<nv_bfloat16>(), d_bias_f32.get<float>(), d_bias_f32.nelem(), mRunState.MainStream);
    if (accumulate) {
        vector_add_sr(d_bias, d_bias, d_bias_cast, 1.0f, static_cast<long>(d_bias.nelem()), 0, mRunState.MainStream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_bias.Data,
                                   d_bias_cast.Data,
                                   d_bias.bytes(),
                                   cudaMemcpyDeviceToDevice,
                                   mRunState.MainStream));
    }
}

namespace {

// -----------------------------------------------------------------------------
// MoE expert bias add backward rule
// Forward: out = moe_expert_bias_add(x, bias)
// Backward: d_x, d_bias = moe_expert_bias_add_backward(d_out, bias)
// -----------------------------------------------------------------------------
std::vector<Operation> moe_expert_bias_add_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;
    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.empty()) {
        return ops;
    }

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    if (fwd.inputs.size() > 1) {
        outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");
    }

    std::vector<std::string> inputs = {ctx.d_output};
    if (fwd.inputs.size() > 1) {
        inputs.push_back(fwd.inputs[1]);
    }

    ops.push_back(make_operation("moe_expert_bias_add_backward_" + std::to_string(ctx.op_counter++),
                                 "moe_expert_bias_add_backward",
                                 "moe_expert_bias_add_backward",
                                 inputs,
                                 outputs));
    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("moe_expert_bias_add", ::dsl::moe_expert_bias_add_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// MoE Expert Bias Add
// ------------------------------------------------------------------------
const int _moe_expert_bias_add_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "moe_expert_bias_add";
    sig.min_inputs = 2;
    sig.max_inputs = 2;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        if (inputs.size() < 2 || outputs.empty()) {
            return std::make_optional(ShapeValidationError{"moe_expert_bias_add requires 2 inputs and 1 output"});
        }
        const auto& x = inputs[0];
        const auto& bias = inputs[1];
        const auto& out = outputs[0];
        if (!x.empty() && x.size() != 2) {
            return std::make_optional(ShapeValidationError{"moe_expert_bias_add: input must be 2D [tokens, hidden]"});
        }
        if (!bias.empty() && bias.size() != 2) {
            return std::make_optional(ShapeValidationError{"moe_expert_bias_add: bias must be 2D [experts, hidden]"});
        }
        if (!out.empty() && !x.empty()) {
            if (out.size() != 2 || out[0] != x[0] || out[1] != x[1]) {
                ShapeValidationError err;
                err.message = "moe_expert_bias_add: output shape must match input shape";
                return std::make_optional(err);
            }
        }
        if (!x.empty() && !bias.empty() && bias[1] != x[1]) {
            ShapeValidationError err;
            err.message = "moe_expert_bias_add: bias hidden dim must match input hidden dim";
            return std::make_optional(err);
        }
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// MoE Expert Bias Add Backward
// ------------------------------------------------------------------------
const int _moe_expert_bias_add_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "moe_expert_bias_add_backward";
    sig.min_inputs = 1;
    sig.max_inputs = 2;
    sig.min_outputs = 2;
    sig.max_outputs = 2;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& d_inp = outputs[0];
        const auto& d_bias = outputs[1];

        if (auto err = validators::check_same_numel(d_inp, d_out, "d_inp", "d_out", "moe_expert_bias_add_backward")) {
            return err;
        }

        if (!d_bias.empty()) {
            if (d_bias.size() != 2) {
                ShapeValidationError err;
                err.message = "moe_expert_bias_add_backward: d_bias must be 2D [experts, hidden]";
                return std::make_optional(err);
            }
            if (!d_out.empty() && d_bias[1] != d_out.back()) {
                ShapeValidationError err;
                err.message = "moe_expert_bias_add_backward: d_bias hidden dim must match d_out last dim";
                return std::make_optional(err);
            }
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
