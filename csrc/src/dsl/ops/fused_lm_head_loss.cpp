#include "dsl/compiled_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/dsl_weight_manager.h"
#include "dsl/graph_executor_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_fused_lm_head_loss(const CompiledOp& op) {
    Tensor& xF_flat = resolve_tensor(op.inputs[0]);
    Tensor& weight = resolve_tensor(op.inputs[1]);
    Tensor& targets = resolve_tensor(op.inputs[2]);
    Tensor& loss = ensure_output_tensor(op.outputs[0]);

    const long BT = xF_flat.Sizes[0];
    const int V = static_cast<int>(weight.Sizes[0]);
    const int C = static_cast<int>(weight.Sizes[1]);
    const int P = V;

    const int lmhead_chunks = std::max(1, mOptions.LMHeadChunks);
    const long nano_batch_size = BT / static_cast<long>(lmhead_chunks);
    const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;

    const bool need_lm_head =
        mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled());
    if (need_lm_head && mComm) {
        mWeightManager->gather_lm_head(*mComm, mRunState.MainStream);
    }

    const std::size_t xf_stride = get_dtype_size(xF_flat.DType);
    const std::size_t tgt_stride = get_dtype_size(targets.DType);
    const std::size_t loss_stride = get_dtype_size(loss.DType);
    const std::size_t lse_stride = get_dtype_size(ETensorDType::FP32);
    const std::size_t chunk_lse_stride = lse_stride * static_cast<std::size_t>(n_chunks);

    for (int nano_step = 0; nano_step < lmhead_chunks; ++nano_step) {
        const long token_offset = static_cast<long>(nano_step) * nano_batch_size;

        Tensor xF_slice = xF_flat;
        xF_slice.Data = static_cast<std::byte*>(xF_slice.Data) +
                        static_cast<std::size_t>(token_offset) * xf_stride * static_cast<std::size_t>(C);
        xF_slice.Sizes[0] = nano_batch_size;
        xF_slice.Sizes[1] = C;
        xF_slice.Rank = 2;

        Tensor tgt_slice = targets;
        tgt_slice.Data = static_cast<std::byte*>(tgt_slice.Data) +
                         static_cast<std::size_t>(token_offset) * tgt_stride;
        tgt_slice.Sizes[0] = nano_batch_size;
        tgt_slice.Rank = 1;

        Tensor loss_slice = loss;
        loss_slice.Data = static_cast<std::byte*>(loss_slice.Data) +
                          static_cast<std::size_t>(token_offset) * loss_stride;
        loss_slice.Sizes[0] = nano_batch_size;
        loss_slice.Rank = 1;

        Tensor logits = mRunState.non_block_activations().output;
        logits.Sizes[0] = nano_batch_size;
        logits.Sizes[1] = V;
        logits.Rank = 2;

        matmul(logits, weight, xF_slice,
               std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               static_cast<int>(V), static_cast<int>(nano_batch_size), static_cast<int>(C),
               swap_transpose(EMMTranspose::NT), false, mRunState.MainStream);

        Tensor logsumexp_view{};
        Tensor* logsumexp = nullptr;
        if (mRunState.scratch().cross_entropy_logsumexp.Data) {
            logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
            logsumexp_view.Data = static_cast<std::byte*>(logsumexp_view.Data) +
                                  static_cast<std::size_t>(token_offset) * lse_stride;
            logsumexp_view.Sizes[0] = nano_batch_size;
            logsumexp_view.Rank = 1;
            logsumexp = &logsumexp_view;
        }

        if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
            if (!mRunState.scratch().cross_entropy_chunk_logsumexp.Data) {
                throw std::runtime_error("fused_lm_head_loss: chunk logsumexp buffer is not allocated");
            }
            Tensor chunk_lse = mRunState.scratch().cross_entropy_chunk_logsumexp;
            chunk_lse.Data = static_cast<std::byte*>(chunk_lse.Data) +
                             static_cast<std::size_t>(token_offset) * chunk_lse_stride;
            chunk_lse.Sizes[0] = nano_batch_size;
            chunk_lse.Sizes[1] = n_chunks;
            chunk_lse.Rank = 2;

            chunked_cross_entropy_forward(logits, loss_slice, logsumexp, chunk_lse, tgt_slice,
                                          &mRunState.ValidTokenCount,
                                          op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                          static_cast<int>(nano_batch_size), V, P, n_chunks, mRunState.MainStream);
        } else {
            fused_cross_entropy_forward(logits, loss_slice, logsumexp, tgt_slice,
                                        &mRunState.ValidTokenCount,
                                        op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                        static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        }
    }

    if (need_lm_head) {
        mWeightManager->release_lm_head(mRunState.MainStream);
    }
}

void CompiledExecutor::dispatch_fused_lm_head_loss_backward(const CompiledOp& op) {
    Tensor& d_loss = resolve_tensor(op.inputs[0]);
    Tensor& xF_flat = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);
    Tensor& targets = resolve_tensor(op.inputs[3]);

    Tensor* d_xF_ptr = nullptr;
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        d_xF_ptr = &ensure_output_tensor(op.outputs[0]);
    }

    Tensor* d_weight_ptr = nullptr;
    bool d_weight_accumulate = false;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        std::string weight_name = op.outputs[1].name;
        if (auto base = base_param_from_grad(weight_name)) {
            weight_name = *base;
        } else if (weight_name.rfind("d_", 0) == 0) {
            weight_name = weight_name.substr(2);
        }
        bool accum = false;
        Tensor* grad = mGrads.get_param_grad(weight_name, accum);
        d_weight_accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
        if (!d_weight_accumulate) {
            if (auto base = base_param_from_grad(op.outputs[1].name)) {
                d_weight_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }
        if (grad && grad->Data) {
            d_weight_ptr = &ensure_output_tensor(op.outputs[1]);
        }
    }

    // HuggingFace-style normalization: use reduction="sum" semantics.
    // dloss = 1.0 means each valid token contributes equally to the gradient sum.
    // The actual normalization by accumulated valid tokens happens in global_norm_sqrt.
    // Robustly seed d_loss even if the name has SSA suffixes or mapped to loss/losses.
    const std::string d_loss_name = strip_ssa_suffix(op.inputs[0].name);
    bool d_loss_seeded = false;
    if (op.inputs[0].slot == TensorSlot::DLoss ||
        op.inputs[0].slot == TensorSlot::Losses ||
        d_loss_name == "d_loss" || d_loss_name == "loss" || d_loss_name == "losses") {
        fill_constant(d_loss, 1.0f, static_cast<std::size_t>(d_loss.nelem()), mRunState.MainStream);
        d_loss_seeded = true;
    }

    // DEBUG: Trace loss backward input slot/name and d_loss values.
    static int loss_bwd_trace_count = 0;
    if (loss_bwd_trace_count < 5) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> vals(4, 0.0f);
        const std::size_t count = std::min<std::size_t>(4, static_cast<std::size_t>(d_loss.nelem()));
        if (count > 0 && d_loss.Data) {
            cudaMemcpy(vals.data(), d_loss.Data, count * sizeof(float), cudaMemcpyDeviceToHost);
        }
        fprintf(stderr,
                "[LM_HEAD_LOSS_BWD] d_loss input name='%s' base='%s' slot=%d layer_idx=%d seeded=%d ptr=%p "
                "nelem=%ld dtype=%d vals=%.9f,%.9f,%.9f,%.9f\n",
                op.inputs[0].name.c_str(),
                d_loss_name.c_str(),
                static_cast<int>(op.inputs[0].slot),
                op.inputs[0].layer_idx,
                d_loss_seeded ? 1 : 0,
                d_loss.Data,
                static_cast<long>(d_loss.nelem()),
                static_cast<int>(d_loss.DType),
                vals[0], vals[1], vals[2], vals[3]);
        loss_bwd_trace_count++;
    }

    const long BT = xF_flat.Sizes[0];
    const int V = static_cast<int>(weight.Sizes[0]);
    const int C = static_cast<int>(weight.Sizes[1]);
    const int P = V;

    const int lmhead_chunks = std::max(1, mOptions.LMHeadChunks);
    const long nano_batch_size = BT / static_cast<long>(lmhead_chunks);
    const bool need_lm_head =
        mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled()) &&
        (mOptions.LMHeadChunks > 1);
    if (need_lm_head && mComm) {
        mWeightManager->gather_lm_head(*mComm, mRunState.MainStream);
    }

    const std::size_t xf_stride = get_dtype_size(xF_flat.DType);
    const std::size_t tgt_stride = get_dtype_size(targets.DType);
    const std::size_t lse_stride = get_dtype_size(ETensorDType::FP32);
    const std::size_t dloss_stride = get_dtype_size(d_loss.DType);

    // DEBUG: Full targets stats (ignore/oob/valid) and head/tail sample.
    static int targets_full_trace_count = 0;
    if (targets_full_trace_count < 3) {
        cudaStreamSynchronize(mRunState.MainStream);
        if (targets.DType == ETensorDType::INT32) {
            const std::size_t n = static_cast<std::size_t>(targets.nelem());
            std::size_t ignore = 0;
            std::size_t oob = 0;
            std::size_t valid = 0;
            const std::size_t chunk = 1u << 20;  // 1M ints max per copy
            const std::byte* base = static_cast<const std::byte*>(targets.Data);
            std::vector<int> buf;
            for (std::size_t offset = 0; offset < n; offset += chunk) {
                const std::size_t count = std::min(chunk, n - offset);
                buf.resize(count);
                CUDA_CHECK(cudaMemcpy(buf.data(), base + offset * tgt_stride,
                                      count * sizeof(int), cudaMemcpyDeviceToHost));
                for (std::size_t i = 0; i < count; ++i) {
                    const int v = buf[i];
                    if (v == -100) {
                        ignore++;
                    } else if (v < 0 || v >= V) {
                        oob++;
                    } else {
                        valid++;
                    }
                }
            }

            int head[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            int tail[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (n > 0) {
                const std::size_t head_count = std::min<std::size_t>(8, n);
                CUDA_CHECK(cudaMemcpy(head, base, head_count * sizeof(int), cudaMemcpyDeviceToHost));
                const std::size_t tail_offset = (n > 8) ? (n - 8) : 0;
                const std::size_t tail_count = std::min<std::size_t>(8, n);
                CUDA_CHECK(cudaMemcpy(tail, base + tail_offset * tgt_stride,
                                      tail_count * sizeof(int), cudaMemcpyDeviceToHost));
            }

            fprintf(stderr,
                    "[LM_HEAD_TARGETS_FULL] nelem=%zu V=%d ignore=%zu oob=%zu valid=%zu "
                    "head=%d,%d,%d,%d,%d,%d,%d,%d tail=%d,%d,%d,%d,%d,%d,%d,%d\n",
                    n, V, ignore, oob, valid,
                    head[0], head[1], head[2], head[3], head[4], head[5], head[6], head[7],
                    tail[0], tail[1], tail[2], tail[3], tail[4], tail[5], tail[6], tail[7]);
        } else {
            fprintf(stderr, "[LM_HEAD_TARGETS_FULL] dtype=%d nelem=%ld (unsupported)\n",
                    static_cast<int>(targets.DType), static_cast<long>(targets.nelem()));
        }
        targets_full_trace_count++;
    }

    static int d_xf_full_trace_count = 0;
    const bool trace_full_dxf = (d_xF_ptr != nullptr && d_xf_full_trace_count < 3);
    double dxf_full_sum_sq = 0.0;
    long first_nonzero_offset = -1;
    bool dlogits_nonzero_logged = false;

    auto tensor_sum_sq = [&](const Tensor& t) -> double {
        const std::size_t n = static_cast<std::size_t>(t.nelem());
        if (n == 0 || !t.Data) {
            return 0.0;
        }
        const std::size_t chunk = 1u << 20;  // 1M elements per copy
        std::vector<float> buf;
        double sum_sq = 0.0;
        for (std::size_t offset = 0; offset < n; offset += chunk) {
            const std::size_t count = std::min(chunk, n - offset);
            buf.resize(count);
            Tensor tmp = t;
            tmp.Data = static_cast<std::byte*>(tmp.Data) +
                       offset * get_dtype_size(tmp.DType);
            tmp.Sizes[0] = static_cast<long>(count);
            tmp.Rank = 1;
            if (!copy_tensor_sample_as_f32(tmp, count, buf)) {
                break;
            }
            for (std::size_t i = 0; i < count; ++i) {
                const double v = static_cast<double>(buf[i]);
                sum_sq += v * v;
            }
        }
        return sum_sq;
    };

    for (int nano_step = 0; nano_step < lmhead_chunks; ++nano_step) {
        const long token_offset = static_cast<long>(nano_step) * nano_batch_size;

        Tensor xF_slice = xF_flat;
        xF_slice.Data = static_cast<std::byte*>(xF_slice.Data) +
                        static_cast<std::size_t>(token_offset) * xf_stride * static_cast<std::size_t>(C);
        xF_slice.Sizes[0] = nano_batch_size;
        xF_slice.Sizes[1] = C;
        xF_slice.Rank = 2;

        Tensor tgt_slice = targets;
        tgt_slice.Data = static_cast<std::byte*>(tgt_slice.Data) +
                         static_cast<std::size_t>(token_offset) * tgt_stride;
        tgt_slice.Sizes[0] = nano_batch_size;
        tgt_slice.Rank = 1;

        // DEBUG: Trace targets for loss backward (ignore/oob counts + first values).
        static int targets_trace_count = 0;
        if (targets_trace_count < 5 && nano_step == 0) {
            cudaStreamSynchronize(mRunState.MainStream);
            const std::size_t sample = std::min<std::size_t>(
                64, static_cast<std::size_t>(tgt_slice.nelem()));
            if (tgt_slice.DType == ETensorDType::INT32 && sample > 0) {
                std::vector<int> host(sample);
                cudaMemcpy(host.data(), tgt_slice.Data, sample * sizeof(int), cudaMemcpyDeviceToHost);
                int ignore = 0;
                int oob = 0;
                for (std::size_t i = 0; i < sample; ++i) {
                    const int v = host[i];
                    if (v == -100) {
                        ignore++;
                    } else if (v < 0 || v >= V) {
                        oob++;
                    }
                }
                int vals[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                const std::size_t vcount = std::min<std::size_t>(8, sample);
                for (std::size_t i = 0; i < vcount; ++i) {
                    vals[i] = host[i];
                }
                fprintf(stderr,
                        "[LM_HEAD_TARGETS] token_offset=%ld dtype=%d nelem=%ld sample=%zu V=%d ignore=%d oob=%d "
                        "vals=%d,%d,%d,%d,%d,%d,%d,%d\n",
                        token_offset,
                        static_cast<int>(tgt_slice.DType),
                        static_cast<long>(tgt_slice.nelem()),
                        sample,
                        V,
                        ignore,
                        oob,
                        vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]);
            } else {
                fprintf(stderr,
                        "[LM_HEAD_TARGETS] token_offset=%ld dtype=%d nelem=%ld sample=%zu (unsupported dtype)\n",
                        token_offset,
                        static_cast<int>(tgt_slice.DType),
                        static_cast<long>(tgt_slice.nelem()),
                        sample);
            }
            targets_trace_count++;
        }

        Tensor dloss_slice = d_loss;
        dloss_slice.Data = static_cast<std::byte*>(dloss_slice.Data) +
                           static_cast<std::size_t>(token_offset) * dloss_stride;
        dloss_slice.Sizes[0] = nano_batch_size;
        dloss_slice.Rank = 1;

        Tensor logits = mRunState.non_block_activations().output;
        logits.Sizes[0] = nano_batch_size;
        logits.Sizes[1] = V;
        logits.Rank = 2;

        matmul(logits, weight, xF_slice,
               std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               static_cast<int>(V), static_cast<int>(nano_batch_size), static_cast<int>(C),
               swap_transpose(EMMTranspose::NT), false, mRunState.MainStream);

        // DEBUG: Trace logits before CE backward.
        static int logits_pre_trace_count = 0;
        if (logits_pre_trace_count < 5 && nano_step == 0) {
            cudaStreamSynchronize(mRunState.MainStream);
            std::vector<float> vals(4, 0.0f);
            const bool ok = copy_tensor_sample_as_f32(logits, vals.size(), vals);
            fprintf(stderr,
                    "[LM_HEAD_LOGITS_PRE] token_offset=%ld dtype=%d ptr=%p ok=%d vals=%.9f,%.9f,%.9f,%.9f\n",
                    token_offset,
                    static_cast<int>(logits.DType),
                    logits.Data,
                    ok ? 1 : 0,
                    vals[0], vals[1], vals[2], vals[3]);
            logits_pre_trace_count++;
        }

        Tensor logsumexp_view{};
        Tensor* logsumexp = nullptr;
        if (mRunState.scratch().cross_entropy_logsumexp.Data) {
            logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
            logsumexp_view.Data = static_cast<std::byte*>(logsumexp_view.Data) +
                                  static_cast<std::size_t>(token_offset) * lse_stride;
            logsumexp_view.Sizes[0] = nano_batch_size;
            logsumexp_view.Rank = 1;
            logsumexp = &logsumexp_view;
        }

        if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
            chunked_cross_entropy_backward(logits, logits, logsumexp, dloss_slice, tgt_slice,
                                           static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        } else {
            fused_cross_entropy_backward(logits, logits, logsumexp, dloss_slice, tgt_slice,
                                         static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        }

        // DEBUG: Trace d_logits (stored in logits buffer) after CE backward.
        static int dlogits_trace_count = 0;
        if (dlogits_trace_count < 5 && nano_step == 0) {
            cudaStreamSynchronize(mRunState.MainStream);
            std::vector<float> vals(4, 0.0f);
            const bool ok = copy_tensor_sample_as_f32(logits, vals.size(), vals);
            fprintf(stderr,
                    "[LM_HEAD_DLOGITS] token_offset=%ld dtype=%d ptr=%p ok=%d vals=%.9f,%.9f,%.9f,%.9f\n",
                    token_offset,
                    static_cast<int>(logits.DType),
                    logits.Data,
                    ok ? 1 : 0,
                    vals[0], vals[1], vals[2], vals[3]);
            dlogits_trace_count++;
        }

        if (d_weight_ptr) {
            const bool accumulate = d_weight_accumulate || (nano_step != 0);
            matmul(*d_weight_ptr, xF_slice, logits,
                   std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   static_cast<int>(C), static_cast<int>(V), static_cast<int>(nano_batch_size),
                   swap_transpose(EMMTranspose::TN), accumulate, mRunState.MainStream);
        }

        if (d_xF_ptr) {
            Tensor d_xF_slice = *d_xF_ptr;
            const std::size_t dx_stride = get_dtype_size(d_xF_slice.DType);
            d_xF_slice.Data = static_cast<std::byte*>(d_xF_slice.Data) +
                              static_cast<std::size_t>(token_offset) * dx_stride * static_cast<std::size_t>(C);
            d_xF_slice.Sizes[0] = nano_batch_size;
            d_xF_slice.Sizes[1] = C;
            d_xF_slice.Rank = 2;

            matmul(d_xF_slice, weight, logits,
                   std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   static_cast<int>(C), static_cast<int>(nano_batch_size), static_cast<int>(V),
                   swap_transpose(EMMTranspose::NN), false, mRunState.MainStream);

            if (trace_full_dxf) {
                cudaStreamSynchronize(mRunState.MainStream);
                const double sum_sq = tensor_sum_sq(d_xF_slice);
                dxf_full_sum_sq += sum_sq;
                if (sum_sq > 0.0 && first_nonzero_offset < 0) {
                    first_nonzero_offset = token_offset;
                }
                if (!dlogits_nonzero_logged && sum_sq > 0.0) {
                    std::vector<float> vals(4, 0.0f);
                    const bool ok = copy_tensor_sample_as_f32(logits, vals.size(), vals);
                    fprintf(stderr,
                            "[LM_HEAD_DLOGITS_NONZERO] token_offset=%ld dtype=%d ptr=%p ok=%d sumsq=%.9e "
                            "vals=%.9f,%.9f,%.9f,%.9f\n",
                            token_offset,
                            static_cast<int>(logits.DType),
                            logits.Data,
                            ok ? 1 : 0,
                            sum_sq,
                            vals[0], vals[1], vals[2], vals[3]);
                    dlogits_nonzero_logged = true;
                }
            }

            // DEBUG: Trace d_xF output values + L2 norm for loss backward.
            static int d_xf_trace_count = 0;
            if (d_xf_trace_count < 5 && nano_step == 0) {
                cudaStreamSynchronize(mRunState.MainStream);
                std::vector<float> vals(4, 0.0f);
                const bool ok = copy_tensor_sample_as_f32(d_xF_slice, vals.size(), vals);
                double l2_sum = 0.0;
                if (ok) {
                    const std::size_t n = static_cast<std::size_t>(d_xF_slice.nelem());
                    const std::size_t chunk = 1u << 20;  // 1M elements per chunk
                    std::vector<float> buf;
                    for (std::size_t offset = 0; offset < n; offset += chunk) {
                        const std::size_t count = std::min(chunk, n - offset);
                        buf.resize(count);
                        Tensor tmp = d_xF_slice;
                        tmp.Data = static_cast<std::byte*>(tmp.Data) +
                                   offset * get_dtype_size(tmp.DType);
                        tmp.Sizes[0] = static_cast<long>(count);
                        tmp.Rank = 1;
                        if (!copy_tensor_sample_as_f32(tmp, count, buf)) {
                            break;
                        }
                        for (std::size_t i = 0; i < count; ++i) {
                            const double v = static_cast<double>(buf[i]);
                            l2_sum += v * v;
                        }
                    }
                }
                const double l2 = std::sqrt(l2_sum);
                fprintf(stderr,
                        "[LM_HEAD_LOSS_BWD_OUT] d_xF name='%s' slot=%d ptr=%p nelem=%ld dtype=%d ok=%d l2=%.9e "
                        "vals=%.9f,%.9f,%.9f,%.9f\n",
                        op.outputs[0].name.c_str(),
                        static_cast<int>(op.outputs[0].slot),
                        d_xF_slice.Data,
                        static_cast<long>(d_xF_slice.nelem()),
                        static_cast<int>(d_xF_slice.DType),
                        ok ? 1 : 0,
                        l2,
                        vals[0], vals[1], vals[2], vals[3]);
                d_xf_trace_count++;
            }
        }
    }

    if (trace_full_dxf) {
        const double l2 = std::sqrt(dxf_full_sum_sq);
        fprintf(stderr,
                "[LM_HEAD_DXF_FULL] nelem=%ld l2=%.9e nonzero=%d first_nonzero_offset=%ld\n",
                d_xF_ptr ? static_cast<long>(d_xF_ptr->nelem()) : -1,
                l2,
                (dxf_full_sum_sq > 0.0) ? 1 : 0,
                first_nonzero_offset);
        d_xf_full_trace_count++;
    }


    if (need_lm_head) {
        mWeightManager->release_lm_head(mRunState.MainStream);
    }
}


}  // namespace dsl
