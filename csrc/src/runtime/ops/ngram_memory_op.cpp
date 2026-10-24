// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Hashed n-gram memory on the input norm of its layer (runtime/ngram/ngram_memory.h). The memory is
// the engine's single-stream `ngram_ple` layer prologue: it updates the residual entering layer L
// before that layer's input RMSNorm. In the trainer that residual is formed by the layer's fused
// residual + RMSNorm (residual_in + input, the previous layer's output); with the memory the norm
// instead forms residual_out = bf16(residual_in + input + u) in one fp32 sum (bit-identical to the
// plain norm wherever u = 0) and normalises that sum. The backward adds the gate's query-path
// gradient to the residual gradient and accumulates the memory's parameter gradients into its
// adapter gradients. A replayed forward (recompute) reads the stored residual, which already holds
// the memory's output, so the memory never runs in a replay.

#include <cuda_runtime.h>
#include <fmt/format.h>

#include <initializer_list>
#include <stdexcept>

#include "kernels/kernels.h"
#include "kernels/ngram_memory.h"
#include "runtime/executor/compiled_ops.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_grads_manager.h"
#include "runtime/lora/lora_weights_manager.h"
#include "runtime/ngram/ngram_memory.h"
#include "utilities/utils.h"

namespace dsl {

namespace {

bool stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    return cudaStreamIsCapturing(stream, &status) == cudaSuccess && status != cudaStreamCaptureStatusNone;
}

Tensor device_buffer(ETensorDType dtype, std::vector<long> shape, int device) {
    std::size_t bytes = get_dtype_size(dtype);
    for (long d : shape) bytes *= static_cast<std::size_t>(d);
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, std::max<std::size_t>(bytes, 256)));
    CUDA_CHECK(cudaMemset(ptr, 0, std::max<std::size_t>(bytes, 256)));
    Tensor t = Tensor::from_pointer(static_cast<std::byte*>(ptr), device, dtype, shape);
    return t;
}

void free_buffer(Tensor& t) {
    if (t.Data) cudaFree(t.Data);
    t = Tensor{};
}

}  // namespace

bool CompiledExecutor::ngram_memory_active() const {
    return mRunState.runtime_config().ngram_memory_layer >= 0 && mLoRAWeights && mLoRAConfig &&
           mLoRAConfig->enabled() && mLoRAWeights->ngram_memory_layer() >= 0;
}

bool CompiledExecutor::ngram_memory_site(int layer_idx, const std::string& field) const {
    return layer_idx >= 0 && layer_idx == mRunState.runtime_config().ngram_memory_layer && field == "ln1_weight" &&
           ngram_memory_active();
}

const ngram_memory::Spec& CompiledExecutor::ngram_spec() {
    if (!mNgramSpec) {
        mNgramSpec = std::make_shared<const ngram_memory::Spec>(
            ngram_memory::Spec::parse(mRunState.runtime_config().ngram_memory_spec));
    }
    return *mNgramSpec;
}

void CompiledExecutor::free_ngram_buffers() {
    for (Tensor* t : {&mNgram.r, &mNgram.emb, &mNgram.k, &mNgram.v, &mNgram.u, &mNgram.stats, &mNgram.rows,
                      &mNgram.carry, &mNgram.gains, &mNgram.w, &mNgram.partials, &mNgram.d_emb,
                      &mNgram.table_scratch}) {
        free_buffer(*t);
    }
    mNgram.tokens = 0;
}

void CompiledExecutor::ensure_ngram_buffers() {
    const auto& spec = ngram_spec();
    const long tokens = mB * mT;
    const int C = static_cast<int>(mConfig.HiddenSize);
    const int E = spec.embed_dim();
    if (mNgram.tokens == tokens && mNgram.hidden == C && mNgram.embed == E && mNgram.r.Data) return;
    if (mCapturing || stream_capturing(mRunState.MainStream)) {
        throw std::runtime_error("ngram_memory: its buffers must be sized before CUDA graph capture "
                                 "(train the memory without CUDA graphs)");
    }
    CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
    free_ngram_buffers();
    const int dev = mRunState.DeviceId;
    mNgram.r = device_buffer(ETensorDType::BF16, {tokens, C}, dev);
    mNgram.emb = device_buffer(ETensorDType::BF16, {tokens, E}, dev);
    mNgram.k = device_buffer(ETensorDType::BF16, {tokens, C}, dev);
    mNgram.v = device_buffer(ETensorDType::BF16, {tokens, C}, dev);
    mNgram.u = device_buffer(ETensorDType::BF16, {tokens, C}, dev);
    mNgram.stats = device_buffer(ETensorDType::FP32, {tokens, ngram_memory::kTokenStats}, dev);
    mNgram.rows = device_buffer(ETensorDType::INT32, {tokens, spec.hash.heads}, dev);
    mNgram.carry = device_buffer(ETensorDType::INT32, {std::max<long>(mB, 1), 2}, dev);
    mNgram.gains = device_buffer(ETensorDType::FP32, {2, C}, dev);
    mNgram.w = device_buffer(ETensorDType::FP32, {tokens}, dev);
    mNgram.partials =
        device_buffer(ETensorDType::FP32, {ngram_memory::gain_grad_parts(static_cast<int>(tokens)), C}, dev);
    if (spec.trainable) {
        mNgram.d_emb = device_buffer(ETensorDType::FP32, {tokens, E}, dev);
        const auto scratch = static_cast<long>(ngram_memory::table_backward_scratch_bytes(tokens * spec.hash.heads));
        mNgram.table_scratch = device_buffer(ETensorDType::BYTE, {std::max<long>(scratch, 256)}, dev);
    }
    mNgram.tokens = tokens;
    mNgram.hidden = C;
    mNgram.embed = E;
}

void CompiledExecutor::ngram_memory_forward(Tensor& residual_out, Tensor& y, Tensor& rstd, const Tensor& residual_in,
                                            const Tensor& input, const Tensor& weight, float eps) {
    for (const Tensor* t : std::initializer_list<const Tensor*>{&residual_out, &y, &residual_in, &input, &weight}) {
        if (t->DType != ETensorDType::BF16) throw std::runtime_error("ngram_memory: the residual norm must be BF16");
    }
    if (rstd.DType != ETensorDType::FP32) throw std::runtime_error("ngram_memory: rstd must be FP32");
    if (mCapturing || stream_capturing(mRunState.MainStream)) {
        throw std::runtime_error("ngram_memory: not supported under CUDA graph capture (use_cuda_graphs: false)");
    }
    const auto& spec = ngram_spec();
    ensure_ngram_buffers();
    cudaStream_t stream = mRunState.MainStream;
    const int L = spec.layer;
    const long N = mB * mT;
    const int C = static_cast<int>(mConfig.HiddenSize);
    if (!spec.trainable && !mNgramTable) mNgramTable = ngram_memory::acquire_table(spec, mRunState.DeviceId);

    // Token ids and positions of this (chunk of the) micro-batch.
    const Tensor& ids_t = mRunState.Inputs;
    const Tensor& pos_t = mRunState.PositionIDs;
    if (ids_t.DType != ETensorDType::INT32 || static_cast<long>(ids_t.nelem()) != N) {
        throw std::runtime_error(fmt::format("ngram_memory: token ids hold {} values, the forward {}", ids_t.nelem(), N));
    }
    if (pos_t.DType != ETensorDType::INT32 || static_cast<long>(pos_t.nelem()) < N) {
        throw std::runtime_error("ngram_memory: position ids must be INT32 [B, T] or [planes, B, T]");
    }
    const int* ids = ids_t.get<std::int32_t>();
    const int* positions = pos_t.get<std::int32_t>();  // plane 0
    const int* carry = nullptr;
    if (sequence_chunk_active() && mChunkIdx > 0) {
        if (mRunState.NgramCarryChunk != mChunkIdx ||
            static_cast<long>(mRunState.NgramCarry.size()) != 2 * mB) {
            throw std::runtime_error(fmt::format(
                "ngram_memory: chunk {} has no token carry (the chunked step sets it before each chunk; got chunk {})",
                mChunkIdx, mRunState.NgramCarryChunk));
        }
        CUDA_CHECK(cudaMemcpyAsync(mNgram.carry.Data, mRunState.NgramCarry.data(), 2 * mB * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        carry = mNgram.carry.get<std::int32_t>();
    }

    auto& work = mLoRAWeights->get_block(L, stream);
    auto& master = mLoRAWeights->get_master_block(L, stream);
    if (!work.ngram_proj.has_value() || !master.ngram_gain.has_value()) {
        throw std::logic_error("ngram_memory: the adapter holds no memory parameters for layer " + std::to_string(L));
    }

    ngram_memory::rows(mNgram.rows.get<std::int32_t>(), ids, positions, carry, static_cast<int>(mB),
                       static_cast<int>(mT), spec.hash, stream);
    if (spec.trainable) {
        // The trainable table: its FP32 master, each gathered value rounded once to bf16.
        if (!master.ngram_table.has_value() || !master.ngram_table->A.Data ||
            master.ngram_table->A.DType != ETensorDType::FP32 ||
            master.ngram_table->A.nelem() !=
                static_cast<std::size_t>(spec.row_count) * static_cast<std::size_t>(spec.head_dim)) {
            throw std::logic_error("ngram_memory: the adapter holds no FP32 [rows, head_dim] trainable table");
        }
        ngram_memory::gather_fp32(mNgram.emb.get<nv_bfloat16>(), mNgram.rows.get<std::int32_t>(),
                                  master.ngram_table->A.get<float>(), spec.head_dim, spec.hash.heads,
                                  static_cast<int>(N), stream);
    } else {
        ngram_memory::gather_iq4nl(mNgram.emb.get<nv_bfloat16>(), mNgram.rows.get<std::int32_t>(), mNgramTable,
                                   spec.row_bytes, spec.head_dim, spec.hash.heads, static_cast<int>(N), stream,
                                   spec.row_shuffle);
    }
    ngram_memory::residual_add(mNgram.r.get<nv_bfloat16>(), residual_in.get<nv_bfloat16>(), input.get<nv_bfloat16>(),
                               N * C, stream);

    const Tensor& Wk = work.ngram_proj->A;
    const Tensor& Wv = work.ngram_proj->B;
    if (Wk.DType != ETensorDType::BF16 || Wv.DType != ETensorDType::BF16) {
        throw std::runtime_error("ngram_memory: the projection work copies must be BF16 (a BF16 model)");
    }
    float* gains = mNgram.gains.get<float>();
    ngram_memory::to_fp32(gains, master.ngram_gain->A.Data, master.ngram_gain->A.DType == ETensorDType::BF16, C, stream);
    ngram_memory::to_fp32(gains + C, master.ngram_gain->B.Data, master.ngram_gain->B.DType == ETensorDType::BF16, C,
                          stream);

    float* global_stats = nullptr;
    if (!sequence_chunk_kv_sweep() && !mInBackwardPass) {
        if (!mRunState.NgramStats.Data) {
            mRunState.NgramStats = device_buffer(ETensorDType::FP32, {ngram_memory::kGlobalStats}, mRunState.DeviceId);
        }
        global_stats = mRunState.NgramStats.get<float>();
    }
    ngram_memory::forward_core(mNgram.k, mNgram.v, mNgram.stats, mNgram.u, global_stats, mNgram.r, mNgram.emb, Wk, Wv,
                               gains, gains + C, ids, spec.zero_tokens, spec.eps, mRunState.CublasLtHandle,
                               mRunState.CuBlasWorkspace, stream);
    ngram_memory::residual3_rmsnorm_forward(residual_out.get<nv_bfloat16>(), y.get<nv_bfloat16>(), rstd.get<float>(),
                                            residual_in.get<nv_bfloat16>(), input.get<nv_bfloat16>(),
                                            mNgram.u.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), eps,
                                            static_cast<int>(N), C, stream);
}

void CompiledExecutor::ngram_memory_backward(Tensor& d_input, float* abs_max) {
    const auto& spec = ngram_spec();
    const long N = mB * mT;
    const int C = static_cast<int>(mConfig.HiddenSize);
    if (mNgram.tokens != N || !mNgram.r.Data) {
        throw std::logic_error("ngram_memory: backward without the forward of the same micro-batch");
    }
    if (d_input.DType != ETensorDType::BF16 || static_cast<long>(d_input.nelem()) != N * C) {
        throw std::runtime_error("ngram_memory: the residual gradient must be BF16 [B*T, C]");
    }
    if (!mLoRAGrads) return;  // no gradients wanted (a forward-only backward)
    cudaStream_t stream = mRunState.MainStream;
    auto& grads = mLoRAGrads->block_full(spec.layer);
    if (!grads.ngram_proj.has_value() || !grads.ngram_gain.has_value() || !grads.ngram_proj->A.Data) {
        throw std::logic_error("ngram_memory: the adapter gradients are not allocated");
    }
    Tensor dk = mRunState.temp_alloc(ETensorDType::BF16, {N, static_cast<long>(C)}, "ngram_memory_dk");
    mTemps.push_back(dk);
    Tensor dv = mRunState.temp_alloc(ETensorDType::BF16, {N, static_cast<long>(C)}, "ngram_memory_dv");
    mTemps.push_back(dv);
    const float* gains = mNgram.gains.get<float>();
    Tensor d_res = d_input;
    ngram_memory::backward_core(d_res, grads.ngram_proj->A, grads.ngram_proj->B, grads.ngram_gain->A,
                                grads.ngram_gain->B, dk, dv, mNgram.w, mNgram.partials, abs_max, mNgram.r, mNgram.emb,
                                mNgram.k, mNgram.v, mNgram.stats, gains, gains + C,
                                mRunState.Inputs.get<std::int32_t>(), spec.zero_tokens, mRunState.CublasLtHandle,
                                mRunState.CuBlasWorkspace, stream);
    if (spec.trainable) {
        // The rows' gradient (dk W_k + dv W_v, 0 at the zero tokens), scattered into the table gradient in row order.
        if (!grads.ngram_table.has_value() || !grads.ngram_table->A.Data) {
            throw std::logic_error("ngram_memory: the trainable table's gradient is not allocated");
        }
        auto& work = mLoRAWeights->get_block(spec.layer, stream);
        ngram_memory::table_backward(grads.ngram_table->A, mNgram.d_emb, mNgram.table_scratch, dk, dv,
                                     work.ngram_proj->A, work.ngram_proj->B, mNgram.rows.get<std::int32_t>(),
                                     spec.hash.heads, spec.head_dim, mRunState.CublasLtHandle,
                                     mRunState.CuBlasWorkspace, stream);
    }
}

}  // namespace dsl
