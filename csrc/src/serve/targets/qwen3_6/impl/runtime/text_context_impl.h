#include "api/ops/ngram_ple.h"
#include <cuda.h>
#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/prefill_graph.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/nvtx.h"
#include "targets/qwen3_6/impl/runtime/visual_scatter.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include <array>
#include <chrono>
#include <api/targets/qwen3_6/vision_control.h>
#include "api/ops/argmax.h"
#include "ops/linear/bf16/bf16_cublaslt.h"
#include "api/ops/attn_input_proj.h"
#include "api/ops/causal_conv1d_silu.h"
#include "api/ops/embedding.h"
#include "api/ops/gated_delta_net.h"
#include "api/ops/gated_rmsnorm.h"
#include "api/ops/mask_columns.h"
#include "api/ops/gdn_gating.h"
#include "api/ops/gdn_gating_proj.h"
#include "api/ops/gdn_input_proj.h"
#include "api/ops/gqa_attention.h"
#include "api/ops/linear.h"
#include "api/ops/linear_add.h"
#include "api/ops/linear_pair.h"
#include "api/ops/linear_swiglu.h"
#include "api/ops/mtp_pack.h"
#include "api/ops/position.h"
#include "api/ops/residual_add.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/rope.h"
#include "api/ops/scatter.h"
#include "api/ops/scalar.h"
#include "api/ops/sigmoid_mul.h"
#include "api/ops/silu_mul.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::GqaExecutionEnvelope*& slot,
                   const ops::GqaExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::GqaExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                         qwen3_6::PagedKVCacheView kv, LinearAttentionStatePool& state,
                         qwen3_6::RoundState& io, Tensor& prefill_hidden,
                         std::uint32_t prefill_chunk, std::uint32_t text_kv_base,
                         qwen3_6::PagedKVCacheView mtp_kv,
                         const qwen3_6::PagedKVCache* batch_text_kv,
                         const qwen3_6::PagedKVCache* batch_mtp_kv)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(mtp_kv), state_(state), io_(io),
      prefill_hidden_(prefill_hidden), prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base),
      batch_text_kv_(batch_text_kv), batch_mtp_kv_(batch_mtp_kv) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, state_.slot_count() > 1 ? 1 : 0);
    bind();
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t current_slot,
                                         std::int32_t rewrite_checkpoint_slot) {
    // kNoRewriteCheckpointSlot (-1) means the pool holds no checkpoint slots
    // (EngineOptions::rewrite_checkpoints off); a capture request against it
    // is refused where the capture would happen.
    const bool checkpoint_ok = rewrite_checkpoint_slot == kNoRewriteCheckpointSlot ||
                               (rewrite_checkpoint_slot >= 0 &&
                                rewrite_checkpoint_slot < state_.slot_count() &&
                                rewrite_checkpoint_slot != current_slot);
    if (current_slot < 0 || current_slot >= state_.slot_count() || !checkpoint_ok) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_current_slot_            = current_slot;
    linear_state_rewrite_checkpoint_slot_ = rewrite_checkpoint_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::bind() {
    using TargetBindings = LoadedModelData;
    using TargetMlp      = MlpWeights;
    const auto bind_mlp  = [](const TargetMlp& source) { return MlpW{&source}; };

    embed_      = &weights_.token_embedding;
    final_norm_ = &weights_.final_norm;
    lm_head_    = &weights_.output_head;
    if (weights_.optimized_proposal) {
        const auto& proposal = *weights_.optimized_proposal;
        set_proposal_head(&proposal.head, static_cast<const std::int32_t*>(proposal.token_ids.data),
                          proposal.head.n);
    }

    if (mtp_enabled()) {
        if (!weights_.mtp) {
            throw std::invalid_argument("MTP state was enabled without materialized MTP weights");
        }
        const auto& source = *weights_.mtp;
        mtp_               = MtpW{&source,
                    &source.input_projection,
                    &source.embedding_norm,
                    &source.hidden_norm,
                    &source.input_norm,
                    &source.query_norm,
                    &source.key_norm,
                    &source.output,
                    &source.post_attention_norm,
                    &source.final_norm};
    }

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            const auto& source =
                weights_.full_layers[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.post_mixer);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            const auto& source     = weights_.gdn_layers[gidx];
            out.input_norm         = &source.input_norm;
            out.projection         = &source.projection;
            out.conv1d             = &source.convolution;
            out.gdn_norm           = &source.norm;
            out.out_proj           = &source.output;
            out.post_attn_norm     = &source.post_attention_norm;
            out.mlp                = bind_mlp(source.post_mixer);
        }
    }
}

const MtpW& TextContext::mtp_weights() const {
    if (!mtp_enabled()) { throw std::runtime_error("MTP draft weights are not enabled"); }
    return mtp_;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({kCfg.hidden, T});

    auto roots = workspace_recipe::mtp_stem<TextConfig>(work_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 || input_embeddings->ne[0] != kCfg.hidden ||
            input_embeddings->numel() != static_cast<std::int64_t>(kCfg.hidden) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({kCfg.hidden, T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, *mtp_.pre_fc_norm_embedding, kCfg.rms_eps, true, e, s);
    ops::rmsnorm(flat_hidden, *mtp_.pre_fc_norm_hidden, kCfg.rms_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    ops::linear(fc_in, *mtp_.fc, x, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, *mtp_.input_norm, kCfg.rms_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace_recipe::mtp_attention_projection<TextConfig>(work_, T);
    Tensor q              = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k              = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor gate           = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor v              = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat         = q.view({kCfg.q_size, T});
    Tensor gate_flat      = gate.view({kCfg.q_size, T});
    Tensor k_flat         = k.view({kCfg.kv_size, T});
    Tensor v_flat         = v.view({kCfg.kv_size, T});
    Variant::mtp_attention_projection(ah, mtp_.payload->attention, q_flat, gate_flat, k_flat,
                                      v_flat, work_, s);

    const auto results = workspace_recipe::mtp_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
                           *active_backend_kv_table_rows_, kAttnScale,
                           batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, s);
    } else {
        ops::gqa_attention(qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row, kAttnScale,
                           batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace_recipe::mtp_post_attention<TextConfig>(work_, T);
    Tensor o        = post.output;
    ops::linear(a.view({kCfg.q_size, T}), *mtp_.o_proj, o, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x, work_, s);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({kCfg.hidden, T});
    ops::rmsnorm(x, *mtp_.norm, kCfg.rms_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::GqaExecutionEnvelope envelope, bool final_chunk,
                                    Tensor* final_hidden, Tensor* logits, Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {kCfg.hidden, 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ah_last = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Tensor v_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Variant::mtp_kv_projection(ah, mtp_.payload->attention, k_flat, v_flat, work_, s);
        Tensor k  = k_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor v  = v_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
        ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
        ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, kn, s);
        ops::gqa_kv_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(kCfg.hidden) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat    = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Tensor gate_flat = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Variant::mtp_q_gate_projection(ah_last, mtp_.payload->attention, q_flat, gate_flat, work_,
                                       s);
        Tensor q    = q_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor gate = gate_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor qn   = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        ops::rope(last_rope_position, kCfg.rotary_dim, kCfg.rope_theta, qn, s);

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::gqa_attention_cached(qn, last_position, kAttnScale, mtp_kv_.layer_view(0), envelope,
                                  work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::linear(a.view({kCfg.q_size, 1}), *mtp_.o_proj, o, s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::rmsnorm(x_last, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x_last, work_, s);
        }
        ops::rmsnorm(x_last, *mtp_.norm, kCfg.rms_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    const int T = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, kCfg.vocab, T, "proposal logits");
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        ops::linear(hidden, *proposal_head_, proposal_logits, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens, proposal_head_n_, ctx_.stream);
        ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                      ctx_.stream);
    } else {
        Tensor output_logits = matrix_window(logits, T);
        ops::linear(hidden, *lm_head_, output_logits, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens, kCfg.token_domain, ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, ops::GqaExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position, ops::GqaExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}


// --- layer-prologue column staging ---------------------------------------------------------
// Only reached by targets whose Variant declares a layer prologue; the planes come from the
// transient arena of the forward that stages them.
namespace prologue_staging {

inline void fill_i32(Tensor& tensor, std::int32_t value, cudaStream_t stream) {
    const CUresult status = cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(tensor.data),
                                             static_cast<unsigned int>(value),
                                             static_cast<std::size_t>(tensor.numel()), stream);
    if (status != CUDA_SUCCESS) {
        throw std::runtime_error("prologue staging: 32-bit fill failed");
    }
}

// Decode columns: every column is its own segment on its own slot.
inline PrologueColumns decode_columns(WorkspaceArena& work, const Tensor& ids, const Tensor& slots,
                                      std::int32_t columns, cudaStream_t stream) {
    PrologueColumns out;
    out.ids           = ids;
    out.slots         = slots;
    out.segment_begin = work.alloc(DType::I32, {columns});
    out.segment_last  = work.alloc(DType::I32, {columns});
    ops::fill_i32_positions(out.segment_begin, 0, stream);
    fill_i32(out.segment_last, 1, stream);
    return out;
}

// One prompt segment on one slot, `valid` columns long: a host count, or a device scalar for
// the bucket-padded graph bodies where the count is an ingress value.
inline PrologueColumns single_segment_columns(WorkspaceArena& work, const Tensor& ids,
                                              std::int32_t columns, std::int32_t slot,
                                              const Tensor* valid_scalar, std::int32_t valid_host,
                                              cudaStream_t stream) {
    PrologueColumns out;
    out.ids           = ids;
    out.segment_begin = work.alloc(DType::I32, {columns});
    out.slots         = work.alloc(DType::I32, {columns});
    out.segment_last  = work.alloc(DType::I32, {columns});
    fill_i32(out.segment_begin, 0, stream);
    fill_i32(out.slots, slot, stream);
    fill_i32(out.segment_last, 0, stream);
    if (valid_scalar != nullptr) {
        ops::ngram_ple_mark_segment_last(out.segment_last, *valid_scalar, 0, stream);
    } else {
        Tensor last = out.segment_last.slice(0, valid_host - 1, 1);
        ops::set_i32_scalar(last, 1, stream);
    }
    return out;
}

} // namespace prologue_staging

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_slots,
                                        ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                        Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,32]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {kCfg.residual, batch});
        if constexpr (Hooks::prologue) {
            prologue_ =
                prologue_staging::decode_columns(work_, ids, linear_state_slots, batch, stream);
        }
        if (stage_embeds()) { Hooks::embed(weights_, ids, x, work_, stream); } else { stage_import(x, stream); }
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        if (stage_finishes()) {
            Hooks::finish(weights_, x, kCfg.rms_eps, hidden, work_, stream);
            ops::linear(hidden, *lm_head_, logits, stream);
        } else {
            stage_export(x, stream);
        }
    }
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_slots,
                                           ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                           Tensor& logits, Tensor& target_tokens, Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_gqa_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_slots_, &linear_state_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {kCfg.residual, columns});
        Tensor flat_ids = ids.view({columns});
        if constexpr (Hooks::prologue) {
            if (width != 1) {
                throw std::invalid_argument("layer prologue targets serve one column per lane");
            }
            prologue_ = prologue_staging::decode_columns(work_, flat_ids, linear_state_slots,
                                                         columns, stream);
        }
        if (stage_embeds()) { Hooks::embed(weights_, flat_ids, x, work_, stream); } else { stage_import(x, stream); }
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({kCfg.hidden, columns});
        Tensor flat_logits = logits.view({kCfg.vocab, columns});
        Tensor flat_tokens = target_tokens.view({columns});
        if (stage_finishes()) {
            Hooks::finish(weights_, x, kCfg.rms_eps, flat_hidden, work_, stream);
            ops::linear(flat_hidden, *lm_head_, flat_logits, stream);
            ops::argmax(flat_logits, flat_tokens, kCfg.token_domain, stream);
        } else {
            stage_export(x, stream);
        }
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                      Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_slots, envelope, hidden, logits, target_tokens, tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                      ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                      Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_slots, envelope, hidden, logits, target_tokens, sink);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_gqa_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace_recipe::text_attention_projection<TextConfig>(work_, T);
    Tensor h              = projection.hidden;
    Hooks::attention_norm(x, *w.input_norm, kCfg.rms_eps, *w.projection, h, work_, s);

    Tensor q         = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor gate      = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor v         = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({kCfg.q_size, T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    Variant::attention_projection(h, *w.projection, q_flat, gate_flat, k_flat, v_flat, ph, work_,
                                  s);

    const auto results = workspace_recipe::text_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    // QSA indexer: cache this chunk's indexer keys and, past the budget, select the blocks each
    // column may attend to. Empty below the budget, so the dense path is untouched.
    const std::int32_t indexer_columns_per_row =
        active_sequence_batch_ != 0 ? active_sequence_width_ : T;
    const ops::GqaBlockMask selection = text_indexer_selection(
        w, h, T, cache_positions, rope_positions, kv_table_rows, indexer_columns_per_row,
        static_cast<std::int32_t>(active_gqa_envelope_->max_visible_keys),
        batch_text_kv_->batch_layer_view(fidx));
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
                           kAttnScale, batch_text_kv_->batch_layer_view(fidx),
                           *active_gqa_envelope_, work_, a_batch, s, selection);
    } else {
        ops::gqa_attention(qn, kn, v, cache_positions, Tensor{}, kv_table_rows, kAttnScale,
                           batch_text_kv_->batch_layer_view(fidx), *active_gqa_envelope_, work_, a,
                           s, selection);
    }
    ops::sigmoid_mul(gate, a, s);

    Variant::attention_output_projection(a.view({kCfg.q_size, T}), *w.o_proj, x, ph, work_, s);
}

struct PrefillFamilyTimer {
    bool enabled = std::getenv("SUROGATE_SERVE_PREFILL_TIMING") != nullptr;
    cudaEvent_t begin{}, attn{}, mlp_full{}, gdn{}, mlp_gdn{};
    cudaEvent_t g_ctrl{}, g_proj{}, g_conv{}, g_extract{}, g_scan{}, g_norm{}, g_out{}, sub_begin{};
    double t_attn = 0, t_mlp_full = 0, t_gdn = 0, t_mlp_gdn = 0;
    double t_g_ctrl = 0, t_g_proj = 0, t_g_conv = 0, t_g_extract = 0, t_g_scan = 0, t_g_norm = 0,
           t_g_out = 0;
    std::uint64_t chunks = 0, tokens = 0;
    int device = -1;
    PrefillFamilyTimer() = default;
    void ensure_events() {
        if (!enabled || device >= 0) { return; }
        cudaGetDevice(&device);
        for (cudaEvent_t* e : {&begin, &attn, &mlp_full, &gdn, &mlp_gdn, &g_ctrl, &g_proj, &g_conv,
                               &g_extract, &g_scan, &g_norm, &g_out, &sub_begin}) {
            cudaEventCreateWithFlags(e, cudaEventDefault);
        }
    }
};
// One timer per device: events belong to the device they were created on, and a pipeline
// stage's context runs on its own device.
inline PrefillFamilyTimer& prefill_family_timer() {
    static std::array<PrefillFamilyTimer, 16> timers;
    int device = 0;
    cudaGetDevice(&device);
    PrefillFamilyTimer& timer = timers[static_cast<std::size_t>(device) % timers.size()];
    timer.ensure_events();
    return timer;
}
inline void print_gdn_subsplit(const PrefillFamilyTimer& timer, const char* tag) {
    const double g = timer.t_g_ctrl + timer.t_g_proj + timer.t_g_conv + timer.t_g_extract +
                     timer.t_g_scan + timer.t_g_norm + timer.t_g_out;
    if (g <= 0 || timer.chunks == 0) { return; }
    std::fprintf(stderr,
                 "%s gdn sub-split: norm+ctrl %.1f%% in_proj %.1f%% conv %.1f%% extract %.1f%% "
                 "chunked scan %.1f%% gated norm %.1f%% out_proj %.1f%% (%.2f ms/chunk)\n",
                 tag, 100 * timer.t_g_ctrl / g, 100 * timer.t_g_proj / g, 100 * timer.t_g_conv / g,
                 100 * timer.t_g_extract / g, 100 * timer.t_g_scan / g, 100 * timer.t_g_norm / g,
                 100 * timer.t_g_out / g, g / timer.chunks);
}


// Per-window prefill laps (SUROGATE_SERVE_PREFILL_TIMING=1): stream time of the graph replay,
// the embedding/ingress before the layer loop, the eager layer loop, and the lm_head/sample tail,
// plus the host wall time of the window. Eager and replay alike; never under capture.
struct PrefillWindowLaps {
    cudaStream_t stream;
    bool active = false;
    std::int32_t tokens;
    cudaEvent_t begin{}, graph{}, pre{}, layers{}, end{};
    std::chrono::steady_clock::time_point wall_begin;
    bool graph_marked = false, pre_marked = false, layers_marked = false;
    PrefillWindowLaps(cudaStream_t s, std::int32_t window_tokens) : stream(s), tokens(window_tokens) {
        if (!prefill_family_timer().enabled) { return; }
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &status);
        if (status != cudaStreamCaptureStatusNone) { return; }
        for (cudaEvent_t* e : {&begin, &graph, &pre, &layers, &end}) {
            cudaEventCreateWithFlags(e, cudaEventDefault);
        }
        cudaEventRecord(begin, stream);
        wall_begin = std::chrono::steady_clock::now();
        active     = true;
    }
    void mark_graph() { if (active) { cudaEventRecord(graph, stream); graph_marked = true; } }
    void mark_pre() { if (active) { cudaEventRecord(pre, stream); pre_marked = true; } }
    void mark_layers() { if (active) { cudaEventRecord(layers, stream); layers_marked = true; } }
    ~PrefillWindowLaps() {
        if (!active) { return; }
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        const double wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_begin)
                .count();
        static double t_graph = 0, t_pre = 0, t_layers = 0, t_post = 0, t_stream = 0, t_wall = 0;
        static std::uint64_t windows = 0, window_tokens = 0, graph_windows = 0;
        auto lap = [](cudaEvent_t a, cudaEvent_t b) { float ms = 0; cudaEventElapsedTime(&ms, a, b); return static_cast<double>(ms); };
        cudaEvent_t cursor = begin;
        if (graph_marked) { t_graph += lap(cursor, graph); cursor = graph; ++graph_windows; }
        if (pre_marked) { t_pre += lap(cursor, pre); cursor = pre; }
        if (layers_marked) { t_layers += lap(cursor, layers); cursor = layers; }
        t_post += lap(cursor, end);
        t_stream += lap(begin, end);
        t_wall += wall_ms;
        windows += 1; window_tokens += static_cast<std::uint64_t>(tokens);
        if (windows % 32 == 0) {
            std::fprintf(stderr,
                         "prefill-window: %llu windows (%llu graph), %.0f tokens/window: graph %.2f "
                         "ms pre %.2f layers %.2f post %.2f | stream %.2f ms host wall %.2f ms per "
                         "window, %.0f tok/s on the stream\n",
                         static_cast<unsigned long long>(windows),
                         static_cast<unsigned long long>(graph_windows),
                         static_cast<double>(window_tokens) / windows, t_graph / windows,
                         t_pre / windows, t_layers / windows, t_post / windows, t_stream / windows,
                         t_wall / windows, 1000.0 * window_tokens / t_stream);
        }
        for (cudaEvent_t* e : {&begin, &graph, &pre, &layers, &end}) { cudaEventDestroy(*e); }
    }
};

void TextContext::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    // Sub-laps for the prefill family timer; they use their own begin event so the caller's
    // family lap stays intact.
    auto& ftimer                          = prefill_family_timer();
    cudaStreamCaptureStatus sub_capturing = cudaStreamCaptureStatusNone;
    if (ftimer.enabled) { cudaStreamIsCapturing(s, &sub_capturing); }
    const bool sub_timing = ftimer.enabled && ph == Phase::Prefill &&
                            sub_capturing == cudaStreamCaptureStatusNone;
    float sub_ctrl = 0, sub_proj = 0, sub_conv = 0, sub_extract = 0, sub_scan = 0, sub_norm = 0,
          sub_out = 0;
    const auto sub_lap = [&](cudaEvent_t to, float& into) {
        cudaEventRecord(to, s);
        cudaEventSynchronize(to);
        float ms = 0;
        cudaEventElapsedTime(&ms, ftimer.sub_begin, to);
        into += ms;
        cudaEventRecord(ftimer.sub_begin, s);
    };
    if (sub_timing) { cudaEventRecord(ftimer.sub_begin, s); }

    const auto control = workspace_recipe::gdn_control<TextConfig>(work_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    Variant::gdn_norm_control_projection(x, *w.input_norm, kCfg.rms_eps, *w.projection, h, g, beta,
                                         work_, s);
    if (graph_pad_valid_ != nullptr) {
        // Bucket-padded graph body (PATCHES.md #27): zero g/beta past the real
        // token count. g is the log-decay (0 => decay 1) and beta gates the
        // rank-1 update (0 => none), so pad columns leave the recurrent state
        // untouched without any scan-kernel change.
        ops::mask_columns_zero(g, *graph_pad_valid_, s);
        ops::mask_columns_zero(beta, *graph_pad_valid_, s);
    }

    if (sub_timing) { sub_lap(ftimer.g_ctrl, sub_ctrl); }
    const auto projection = workspace_recipe::gdn_projection<TextConfig>(work_, T);
    Tensor z              = projection.output_gate.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor qc             = projection.query;
    Tensor kc             = projection.key;
    Tensor vc             = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        Tensor projection_input = h.view({kCfg.hidden, width, active_sequence_batch_});
        Tensor query_output     = qc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor key_output       = kc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor value_output     = vc.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor gate_output      = z.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor& conv_states     = state_.conv.at(static_cast<std::size_t>(gidx));
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            Variant::gdn_input_projection_record(projection_input, *w.projection, *w.conv1d,
                                                 conv_states, valid, *active_linear_state_slots_,
                                                 records.conv, query_output, key_output,
                                                 value_output, gate_output, ph, work_, s);
        } else {
            Variant::gdn_input_projection_snapshot(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_slots_, *active_linear_state_slots_, query_output, key_output,
                value_output, gate_output, ph, work_, s);
        }
    } else {
        const auto conv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, T);
        Tensor qkv      = conv.projected;
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s);
        if (sub_timing) { sub_lap(ftimer.g_proj, sub_proj); }
        Tensor qkv_c = conv.convolved;
        Tensor conv_state =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
        debug_probe<Variant>("gdn_conv_state_in", conv_state, s);
        if (graph_pad_valid_ != nullptr) {
            ops::causal_conv1d_silu(qkv, *w.conv1d, conv_state, conv_state, qkv_c,
                                    *graph_pad_valid_, s);
        } else {
            ops::causal_conv1d_silu(qkv, *w.conv1d, conv_state, conv_state, qkv_c, s);
        }
        debug_probe<Variant>("gdn_conv", qkv_c, s);
        if (sub_timing) { sub_lap(ftimer.g_conv, sub_conv); }
        ops::extract_bf16_columns(qkv_c, 0, qc, s);
        ops::extract_bf16_columns(qkv_c, kCfg.key_dim, kc, s);
        ops::extract_bf16_columns(qkv_c, 2 * kCfg.key_dim, vc, s);
        if (sub_timing) { sub_lap(ftimer.g_extract, sub_extract); }
    }

    Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});

    Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o  = workspace_recipe::gdn_recurrent_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor& recurrent_states = state_.recurrent.at(static_cast<std::size_t>(gidx));
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch =
            q_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor k_batch =
            k_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor v_batch = vv.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor g_batch = g.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor beta_batch = beta.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor out_batch =
            o.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                               kGdnScale, recurrent_states, valid,
                                               *active_linear_state_slots_, records.key,
                                               records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_snapshot(q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale,
                                          /*normalize_qk=*/true, recurrent_states, valid,
                                          *active_linear_state_slots_, *active_linear_state_slots_,
                                          out_batch, s);
        }
    } else {
        Tensor recurrent_state =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
        debug_probe<Variant>("gdn_recurrent_state_in", recurrent_state, s);
        ops::gated_delta_net(q_recurrent, k_recurrent, vv, g, beta, kGdnScale,
                             /*normalize_qk=*/true, work_, recurrent_state, o, s);
        debug_probe<Variant>("gdn_o", o, s);
    }

    Tensor on = workspace_recipe::gdn_normalized_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (sub_timing) { sub_lap(ftimer.g_scan, sub_scan); }
    ops::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, gdn_output_gate<Variant>(), on, s);
    if (sub_timing) { sub_lap(ftimer.g_norm, sub_norm); }

    Variant::gdn_output_projection(on.view({kCfg.value_dim, T}), *w.out_proj, x, ph, work_, s);
    if (sub_timing) {
        sub_lap(ftimer.g_out, sub_out);
        ftimer.t_g_ctrl += sub_ctrl;       ftimer.t_g_proj += sub_proj;   ftimer.t_g_conv += sub_conv;
        ftimer.t_g_extract += sub_extract; ftimer.t_g_scan += sub_scan;   ftimer.t_g_norm += sub_norm;
        ftimer.t_g_out += sub_out;
    }
}

void TextContext::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    Tensor h       = workspace_recipe::post_mixer_hidden<TextConfig>(work_, T);
    Hooks::post_mixer_norm(x, *post_norm, kCfg.rms_eps, *m.payload, h, work_, s);

    Variant::post_mixer(h, *m.payload, x, ph, work_, s);
}

// Per-family prefill timing behind SUROGATE_SERVE_PREFILL_TIMING: events
// around each mixer and MLP, summarised every 32 prefill chunks. It is the
// instrument for "where does a prefill chunk's time go" when a profiler is
// not available; the synchronize it adds at the end of run_layers only
// exists while the switch is set.

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    PrefillFamilyTimer& timer = prefill_family_timer();
    // Event synchronisation is illegal inside stream capture, and a captured
    // body's replay cannot be timed per family anyway: measure eager prefill
    // only (--enforce-eager), never a body being captured.
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    if (prefill && timer.enabled) { cudaStreamIsCapturing(ctx_.stream, &capturing); }
    const bool timing = prefill && timer.enabled && capturing == cudaStreamCaptureStatusNone;
    float acc_attn = 0, acc_mlp_full = 0, acc_gdn = 0, acc_mlp_gdn = 0;
    const auto lap = [&](cudaEvent_t from, cudaEvent_t to, float& into) {
        cudaEventRecord(to, ctx_.stream);
        cudaEventSynchronize(to);
        float ms = 0;
        cudaEventElapsedTime(&ms, from, to);
        into += ms;
    };
    for (int layer = stage_first_; layer < stage_last_; ++layer) {
        Hooks::layer_prologue(weights_, layer, x, prologue_, ple_state_, work_, ctx_.stream);
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, ctx_.stream); }
                attn_mix(full, x, fidx, ph);
                if (timing) { lap(timer.begin, timer.attn, acc_attn); }
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, ctx_.stream); }
                mlp_tail(full.post_attn_norm, full.mlp, x, ph);
                if (timing) { lap(timer.begin, timer.mlp_full, acc_mlp_full); }
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            nvtx::ScopedRange layer_range(prefill ? nvtx::Name::PrefillLayerGdn
                                                  : nvtx::Name::VerifyLayerGdn,
                                          nvtx::Category::Gdn, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn,
                    static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, ctx_.stream); }
                gdn_mix(gdn, x, gidx, ph);
                if (timing) { lap(timer.begin, timer.gdn, acc_gdn); }
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, ctx_.stream); }
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, ph);
                if (timing) { lap(timer.begin, timer.mlp_gdn, acc_mlp_gdn); }
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        }
    }
    if (timing) {
        timer.t_attn += acc_attn; timer.t_mlp_full += acc_mlp_full;
        timer.t_gdn += acc_gdn;   timer.t_mlp_gdn += acc_mlp_gdn;
        timer.chunks += 1;        timer.tokens += static_cast<std::uint64_t>(x.ne[1]);
        if (timer.chunks % 32 == 0) {
            const double total = timer.t_attn + timer.t_mlp_full + timer.t_gdn + timer.t_mlp_gdn;
            std::fprintf(stderr,
                         "prefill-families: %llu chunks, %.0f tokens/chunk: attn %.1f%% "
                         "mlp(full) %.1f%% gdn %.1f%% mlp(gdn) %.1f%% | %.2f ms/chunk, "
                         "%.0f tok/s inside run_layers\n",
                         static_cast<unsigned long long>(timer.chunks),
                         static_cast<double>(timer.tokens) / timer.chunks,
                         100 * timer.t_attn / total, 100 * timer.t_mlp_full / total,
                         100 * timer.t_gdn / total, 100 * timer.t_mlp_gdn / total,
                         total / timer.chunks, 1000.0 * timer.tokens / total);
            print_gdn_subsplit(timer, "prefill-families");
        }
    }
}

// QSA sparse selection for one full-attention layer. Model-agnostic: driven entirely by the
// Variant's indexer traits and the layer's indexer weights, both absent for every other target.

template <class V>
ops::GqaBlockMask TextContext::text_indexer_selection(const FullLayerW& w, const Tensor& hidden,
                                                      std::int32_t tokens,
                                                      const Tensor& cache_positions,
                                                      const Tensor& rope_positions,
                                                      const Tensor& table_rows,
                                                      std::int32_t columns_per_row,
                                                      std::int32_t keys,
                                                      PagedKVBatchLayerView cache) {
    if constexpr (!requires { V::indexer_head_dim; }) {
        return ops::GqaBlockMask{};
    } else {
        const ops::QsaIndexerGeometry geometry{
            .head_dim   = V::indexer_head_dim,
            .heads      = V::indexer_heads,
            .block      = V::indexer_block,
            .top_k      = V::indexer_top_k,
            .rotary_dim = kCfg.rotary_dim,
            .rope_theta = kCfg.rope_theta,
            .rms_eps    = kCfg.rms_eps,
        };
        // Dependent on V so the discarded branch is never checked against a target whose
        // attention payload has no indexer.
        const auto& payload =
            static_cast<const typename V::FullAttentionProjectionWeights&>(*w.projection);
        const auto& indexer = payload.indexer;
        // SUROGATE_SERVE_NO_QSA_INDEXER=1 turns the whole indexer off — no keys cached, no
        // selection, dense attention over the history: the bisection control.
        static const bool disabled = std::getenv("SUROGATE_SERVE_NO_QSA_INDEXER") != nullptr;
        if (disabled || !indexer.valid() || cache.indexer_pages.data == nullptr || tokens <= 0) {
            return ops::GqaBlockMask{};
        }
        cudaStream_t s = ctx_.stream;
        // The keys are cached raw for every column, whatever the history length: a later query
        // pools them into a block key, so skipping the append below the budget would leave holes.
        Tensor raw_keys = work_.alloc(DType::BF16, {geometry.head_dim, tokens});
        ops::detail::bf16_cublaslt_gemm(indexer.key, hidden, raw_keys, s);
        ops::qsa_indexer_append(raw_keys, cache_positions, table_rows, columns_per_row,
                                indexer.key_norm, geometry, cache, s);
        if (ops::qsa_selection_is_dense(keys, geometry)) { return ops::GqaBlockMask{}; }

        const std::int32_t width = geometry.head_dim * geometry.heads;
        Tensor queries           = work_.alloc(DType::BF16, {width, tokens});
        ops::detail::bf16_cublaslt_gemm(indexer.query, hidden, queries, s);
        Tensor normalized = work_.alloc(DType::BF16, {width, tokens});
        Tensor heads      = queries.view({geometry.head_dim, geometry.heads, tokens});
        Tensor heads_norm = normalized.view({geometry.head_dim, geometry.heads, tokens});
        ops::rmsnorm(heads, indexer.query_norm, geometry.rms_eps, true, heads_norm, s);
        Tensor rope_view = rope_positions.view({tokens});
        ops::rope(rope_view, geometry.rotary_dim, geometry.rope_theta, heads_norm, s);

        const std::int32_t words = ops::qsa_block_mask_words(keys, geometry.block);
        Tensor mask              = work_.alloc(DType::I32, {words, tokens});
        ops::qsa_indexer_select(heads_norm, cache_positions, table_rows, columns_per_row, geometry,
                                cache, keys, work_, mask, s);
        return ops::GqaBlockMask{.words  = static_cast<const std::uint32_t*>(mask.data),
                                 .stride = words,
                                 .block  = geometry.block};
    }
}

void TextContext::set_stage(const StageSpan& stage) {
    const int last = stage.last < 0 ? kCfg.n_layers : stage.last;
    if (stage.first < 0 || last > kCfg.n_layers || stage.first >= last) {
        throw std::invalid_argument("pipeline stage layer range is invalid");
    }
    if ((stage.first > 0 && stage.import_pinned == nullptr) ||
        (last < kCfg.n_layers && stage.export_pinned == nullptr)) {
        throw std::invalid_argument("pipeline stage boundary buffer is missing");
    }
    stage_first_ = stage.first;
    stage_last_  = last;
    stage_       = stage;
}

void TextContext::stage_import(Tensor& x, cudaStream_t stream) {
    const std::int32_t columns = x.ne[1];
    if (columns > stage_.columns) { throw std::logic_error("pipeline stage import wider than its buffer"); }
    CUDA_CHECK(cudaMemcpyAsync(x.data, stage_.import_pinned,
                               static_cast<std::size_t>(kCfg.residual) * columns * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice, stream));
}

void TextContext::stage_export(const Tensor& x, cudaStream_t stream) {
    const std::int32_t columns = x.ne[1];
    if (columns > stage_.columns) { throw std::logic_error("pipeline stage export wider than its buffer"); }
    CUDA_CHECK(cudaMemcpyAsync(stage_.export_pinned, x.data,
                               static_cast<std::size_t>(kCfg.residual) * columns * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}


// ---- Mixed-token round (PATCHES.md #30) -------------------------------------

// One forward over [prefill-chunk | decode-batch] columns: GEMM and fused ops
// run once over all columns; the mixers split per slice onto the existing
// prefill and batch forms. The decode slice's per-lane state/kv indirection
// comes entirely from the caller's device tensors.
PrefillChunkResult TextContext::mixed_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                            std::uint32_t nominal_length, bool finalize_at_end,
                                            const MixedDecodeSlice& decode) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("mixed chunk is outside the prompt");
    }
    const bool is_last = finalize_at_end && begin + nominal_length == full_ids.size();
    const MixedPrefillSegment segment{
        .ids          = full_ids.subspan(begin, nominal_length),
        .kv_base      = static_cast<std::int32_t>(text_kv_base_),
        .kv_table_row = -1, // the program staged io_.text_kv_table_row for this lane
        .state_slot   = static_cast<std::int32_t>(linear_state_current_slot_),
        .finalize     = is_last,
    };
    MixedPrefillFinalize finalize{};
    if (is_last) {
        finalize.hidden    = Tensor{};
        finalize.logits    = matrix_window(io_.logits, 1);
        finalize.positions      = io_.pos;
        finalize.rope_positions = io_.rope_pos;
        finalize.tokens         = io_.token;
        finalize.sampling  = sampling_config_;
    }
    return mixed_chunk_multi(std::span<const MixedPrefillSegment>(&segment, 1), decode, finalize);
}

PrefillChunkResult TextContext::mixed_chunk_multi(std::span<const MixedPrefillSegment> segments,
                                                  const MixedDecodeSlice& decode,
                                                  const MixedPrefillFinalize& finalize) {
    if (segments.empty()) {
        throw std::invalid_argument("mixed chunk needs at least one prefill segment");
    }
    const std::int32_t batch = decode.ids.ne[0]; // 0: a batched prefill round without decode lanes
    if (batch < 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("mixed chunk decode batch is out of range");
    }
    cudaStream_t s   = ctx_.stream;
    int prefill_cols = 0;
    int finalizers   = 0;
    for (const auto& segment : segments) {
        if (segment.ids.empty()) {
            throw std::invalid_argument("mixed chunk segment carries no tokens");
        }
        prefill_cols += static_cast<int>(segment.ids.size());
        finalizers += segment.finalize ? 1 : 0;
    }
    if (finalizers > 0 && (finalize.tokens.data == nullptr || finalize.sampling == nullptr)) {
        throw std::invalid_argument("mixed chunk finalizers need sampler staging");
    }
    const int total  = prefill_cols + batch;
    const int base_i = segments.front().kv_base;
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    work_.reset();
    const auto roots = workspace_recipe::text_prefill_roots<TextConfig>(work_, total, 0, 0);
    Tensor ids_device = roots.ids;
    Tensor ids_decode = ids_device.slice(0, prefill_cols, batch);
    CUDA_CHECK(cudaMemcpyAsync(ids_decode.data, decode.ids.data,
                               static_cast<std::size_t>(batch) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, s));

    Tensor positions = roots.positions;
    // Each segment owns a column range; ids and positions are laid out segment by segment and
    // every mixer below slices the same ranges.
    std::array<int, kMaximumConcurrency> segment_begin{};
    {
        int cursor = 0;
        for (std::size_t i = 0; i < segments.size(); ++i) {
            const int length      = static_cast<int>(segments[i].ids.size());
            segment_begin[i]      = cursor;
            Tensor ids_segment    = ids_device.slice(0, cursor, length);
            Tensor position_range = positions.slice(0, cursor, length);
            copy_i32(segments[i].ids.data(), ids_segment, s);
            ops::fill_i32_positions(position_range, segments[i].kv_base, s);
            cursor += length;
        }
    }
    if constexpr (Hooks::prologue) {
        // Segment columns take their segment's start and slot; the decode lanes behind them
        // are one-column segments on their own slots.
        std::vector<int> begin_host(static_cast<std::size_t>(total));
        std::vector<int> last_host(static_cast<std::size_t>(total), 0);
        std::vector<int> slot_host(static_cast<std::size_t>(prefill_cols));
        for (std::size_t i = 0; i < segments.size(); ++i) {
            const int length = static_cast<int>(segments[i].ids.size());
            for (int c = 0; c < length; ++c) {
                begin_host[static_cast<std::size_t>(segment_begin[i] + c)] = segment_begin[i];
                slot_host[static_cast<std::size_t>(segment_begin[i] + c)]  = segments[i].state_slot;
            }
            last_host[static_cast<std::size_t>(segment_begin[i] + length - 1)] = 1;
        }
        for (int c = prefill_cols; c < total; ++c) {
            begin_host[static_cast<std::size_t>(c)] = c;
            last_host[static_cast<std::size_t>(c)]  = 1;
        }
        prologue_.ids           = ids_device;
        prologue_.segment_begin = work_.alloc(DType::I32, {total});
        prologue_.segment_last  = work_.alloc(DType::I32, {total});
        prologue_.slots         = work_.alloc(DType::I32, {total});
        copy_i32(begin_host.data(), prologue_.segment_begin, s);
        copy_i32(last_host.data(), prologue_.segment_last, s);
        Tensor slots_prefill = prologue_.slots.slice(0, 0, prefill_cols);
        copy_i32(slot_host.data(), slots_prefill, s);
        Tensor slots_decode = prologue_.slots.slice(0, prefill_cols, batch);
        CUDA_CHECK(cudaMemcpyAsync(slots_decode.data, decode.linear_state_slots.data,
                                   static_cast<std::size_t>(batch) * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToDevice, s));
    }
    Tensor positions_decode = positions.slice(0, prefill_cols, batch);
    CUDA_CHECK(cudaMemcpyAsync(positions_decode.data, decode.cache_positions.data,
                               static_cast<std::size_t>(batch) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, s));


    Tensor x = roots.residual;
    if (stage_embeds()) { Hooks::embed(weights_, ids_device, x, work_, s); } else { stage_import(x, s); }

    PrefillFamilyTimer& timer = prefill_family_timer();
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    if (timer.enabled) { cudaStreamIsCapturing(s, &capturing); }
    const bool timing = timer.enabled && capturing == cudaStreamCaptureStatusNone;
    float acc_attn = 0, acc_mlp_full = 0, acc_gdn = 0, acc_mlp_gdn = 0;
    const auto lap = [&](cudaEvent_t from, cudaEvent_t to, float& into) {
        cudaEventRecord(to, s);
        cudaEventSynchronize(to);
        float ms = 0;
        cudaEventElapsedTime(&ms, from, to);
        into += ms;
    };
    for (int layer = stage_first_; layer < stage_last_; ++layer) {
        Hooks::layer_prologue(weights_, layer, x, prologue_, ple_state_, work_, ctx_.stream);
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            {
                auto mixer_scope      = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, s); }
                const auto projection = workspace_recipe::text_attention_projection<TextConfig>(
                    work_, total);
                Tensor h = projection.hidden;
                Hooks::attention_norm(x, *full.input_norm, kCfg.rms_eps, *full.projection, h,
                                      work_, s);
                Tensor q         = projection.query.view({kCfg.head_dim, kCfg.n_q, total});
                Tensor gate      = projection.gate.view({kCfg.head_dim, kCfg.n_q, total});
                Tensor k         = projection.key.view({kCfg.head_dim, kCfg.n_kv, total});
                Tensor v         = projection.value.view({kCfg.head_dim, kCfg.n_kv, total});
                Tensor q_flat    = q.view({kCfg.q_size, total});
                Tensor gate_flat = gate.view({kCfg.q_size, total});
                Tensor k_flat    = k.view({kCfg.kv_size, total});
                Tensor v_flat    = v.view({kCfg.kv_size, total});
                Variant::attention_projection(h, *full.projection, q_flat, gate_flat, k_flat,
                                              v_flat, Phase::Prefill, work_, s);

                const auto results = workspace_recipe::text_attention_results<TextConfig>(work_,
                                                                                         total);
                Tensor qn = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, total});
                Tensor kn = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, total});
                ops::rmsnorm(q, *full.q_norm, kCfg.rms_eps, true, qn, s);
                ops::rmsnorm(k, *full.k_norm, kCfg.rms_eps, true, kn, s);

                Tensor rope_positions = roots.positions;
                if (rope_delta_ != 0) {
                    throw std::logic_error("mixed chunk does not support rope-delta prompts yet");
                }
                Tensor rope_all = rope_positions.view({total});
                if (batch > 0) {
                    Tensor rope_decode = rope_positions.slice(0, prefill_cols, batch);
                    CUDA_CHECK(cudaMemcpyAsync(rope_decode.data, decode.rope_positions.data,
                                               static_cast<std::size_t>(batch) *
                                                   sizeof(std::int32_t),
                                               cudaMemcpyDeviceToDevice, s));
                }
                ops::rope(rope_all, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

                Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, total});
                for (std::size_t sg = 0; sg < segments.size(); ++sg) {
                    const int off = segment_begin[sg];
                    const int len = static_cast<int>(segments[sg].ids.size());
                    // The row scalar is stream-ordered against this segment's launch, so one
                    // scalar serves every segment in turn.
                    if (segments[sg].kv_table_row >= 0) {
                        ops::set_i32_scalar(io_.text_kv_table_row, segments[sg].kv_table_row, s);
                    }
                    const auto seen = static_cast<std::uint32_t>(segments[sg].kv_base + len);
                    const ops::GqaExecutionEnvelope envelope{seen, seen};
                    Tensor qa = qn.slice(2, off, len);
                    Tensor ka = kn.slice(2, off, len);
                    Tensor va = v.slice(2, off, len);
                    Tensor aa = a.slice(2, off, len);
                    ops::gqa_attention(qa, ka, va, positions.slice(0, off, len), Tensor{},
                                       io_.text_kv_table_row, kAttnScale,
                                       batch_text_kv_->batch_layer_view(fidx), envelope, work_, aa,
                                       s);
                }
                if (batch > 0) {
                    Tensor qb = qn.slice(2, prefill_cols, batch)
                                    .view({kCfg.head_dim, kCfg.n_q, 1, batch});
                    Tensor kb = kn.slice(2, prefill_cols, batch)
                                    .view({kCfg.head_dim, kCfg.n_kv, 1, batch});
                    Tensor vb = v.slice(2, prefill_cols, batch)
                                    .view({kCfg.head_dim, kCfg.n_kv, 1, batch});
                    Tensor ab = a.slice(2, prefill_cols, batch)
                                    .view({kCfg.head_dim, kCfg.n_q, 1, batch});
                    Tensor position_batch = decode.cache_positions.view({1, batch});
                    auto decode_scope = work_.scope();
                    const ops::GqaBlockMask decode_selection = text_indexer_selection(
                        full, h.slice(1, prefill_cols, batch), batch, decode.cache_positions,
                        rope_all.slice(0, prefill_cols, batch), decode.kv_table_rows, 1,
                        static_cast<std::int32_t>(decode.envelope.max_visible_keys),
                        batch_text_kv_->batch_layer_view(fidx));
                    ops::gqa_attention(qb, kb, vb, position_batch, Tensor{}, decode.kv_table_rows,
                                       kAttnScale, batch_text_kv_->batch_layer_view(fidx),
                                       decode.envelope, work_, ab, s, decode_selection);
                }
                ops::sigmoid_mul(gate, a, s);
                Variant::attention_output_projection(a.view({kCfg.q_size, total}), *full.o_proj, x,
                                                     Phase::Prefill, work_, s);
                if (timing) { lap(timer.begin, timer.attn, acc_attn); }
            }
            {
                auto mlp_scope = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, s); }
                mlp_tail(full.post_attn_norm, full.mlp, x, Phase::Prefill);
                if (timing) { lap(timer.begin, timer.mlp_full, acc_mlp_full); }
            }
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            {
                auto mixer_scope   = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, s); }
                const auto control = workspace_recipe::gdn_control<TextConfig>(work_, total);
                Tensor h           = control.hidden;
                Tensor g           = control.g;
                Tensor beta        = control.beta;
                Variant::gdn_norm_control_projection(x, *gdn.input_norm, kCfg.rms_eps,
                                                     *gdn.projection, h, g, beta, work_, s);

                float acc_g_ctrl = 0, acc_g_proj = 0, acc_g_conv = 0, acc_g_extract = 0,
                      acc_g_scan = 0, acc_g_norm = 0, acc_g_out = 0;
                if (timing) { lap(timer.begin, timer.g_ctrl, acc_g_ctrl); cudaEventRecord(timer.begin, s); }
                const auto projection = workspace_recipe::gdn_projection<TextConfig>(work_, total);
                Tensor z  = projection.output_gate.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, total});
                Tensor qc = projection.query;
                Tensor kc = projection.key;
                Tensor vc = projection.value;
                const auto conv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, total);
                Tensor qkv      = conv.projected;
                Variant::gdn_input_projection(h, *gdn.projection, qkv, z, Phase::Prefill, work_, s);
                Tensor qkv_c = conv.convolved;
                if (timing) { lap(timer.begin, timer.g_proj, acc_g_proj); cudaEventRecord(timer.begin, s); }
                {
                    for (std::size_t sg = 0; sg < segments.size(); ++sg) {
                        const int off     = segment_begin[sg];
                        const int len     = static_cast<int>(segments[sg].ids.size());
                        Tensor qkv_a      = qkv.slice(1, off, len);
                        Tensor qkv_ca     = qkv_c.slice(1, off, len);
                        Tensor conv_state = state_.conv_slot(
                            static_cast<std::uint32_t>(gidx),
                            static_cast<std::uint32_t>(segments[sg].state_slot));
                        ops::causal_conv1d_silu(qkv_a, *gdn.conv1d, conv_state, conv_state, qkv_ca,
                                                s);
                    }
                }
                if (batch > 0) {
                    Tensor qkv_b  = qkv.slice(1, prefill_cols, batch)
                                       .view({kCfg.conv_dim, 1, batch});
                    Tensor qkv_cb = qkv_c.slice(1, prefill_cols, batch)
                                        .view({kCfg.conv_dim, 1, batch});
                    ops::causal_conv1d_silu_snapshot(qkv_b, *gdn.conv1d,
                                                     state_.conv.at(static_cast<std::size_t>(gidx)),
                                                     Tensor{}, decode.linear_state_slots,
                                                     decode.linear_state_slots, qkv_cb, s);
                }
                if (timing) { lap(timer.begin, timer.g_conv, acc_g_conv); cudaEventRecord(timer.begin, s); }
                ops::extract_bf16_columns(qkv_c, 0, qc, s);
                ops::extract_bf16_columns(qkv_c, kCfg.key_dim, kc, s);
                ops::extract_bf16_columns(qkv_c, 2 * kCfg.key_dim, vc, s);
                if (timing) { lap(timer.begin, timer.g_extract, acc_g_extract); cudaEventRecord(timer.begin, s); }

                Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, total});
                Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, total});
                Tensor vv          = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, total});
                Tensor o = workspace_recipe::gdn_recurrent_output<TextConfig>(work_, total)
                               .view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, total});
                {
                    for (std::size_t sg = 0; sg < segments.size(); ++sg) {
                    const int off = segment_begin[sg];
                    const int len = static_cast<int>(segments[sg].ids.size());
                    Tensor qa = q_recurrent.slice(2, off, len);
                    Tensor ka = k_recurrent.slice(2, off, len);
                    Tensor va = vv.slice(2, off, len);
                    Tensor ga = g.slice(1, off, len);
                    Tensor ba = beta.slice(1, off, len);
                    Tensor oa = o.slice(2, off, len);
                    Tensor recurrent_state = state_.recurrent_slot(
                        static_cast<std::uint32_t>(gidx),
                        static_cast<std::uint32_t>(segments[sg].state_slot));
                    ops::gated_delta_net(qa, ka, va, ga, ba, kGdnScale, true, work_,
                                         recurrent_state, oa, s);
                    }
                }
                if (batch > 0) {
                    Tensor qb = q_recurrent.slice(2, prefill_cols, batch)
                                    .view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1, batch});
                    Tensor kb = k_recurrent.slice(2, prefill_cols, batch)
                                    .view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1, batch});
                    Tensor vb = vv.slice(2, prefill_cols, batch)
                                    .view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1, batch});
                    Tensor gb = g.slice(1, prefill_cols, batch).view({kCfg.gdn_v_heads, 1, batch});
                    Tensor bb =
                        beta.slice(1, prefill_cols, batch).view({kCfg.gdn_v_heads, 1, batch});
                    Tensor ob = o.slice(2, prefill_cols, batch)
                                    .view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1, batch});
                    ops::gated_delta_net_snapshot(
                        qb, kb, vb, gb, bb, kGdnScale, true,
                        state_.recurrent.at(static_cast<std::size_t>(gidx)), Tensor{},
                        decode.linear_state_slots, decode.linear_state_slots, ob, s);
                }
                Tensor on = workspace_recipe::gdn_normalized_output<TextConfig>(work_, total)
                                .view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, total});
                if (timing) { lap(timer.begin, timer.g_scan, acc_g_scan); cudaEventRecord(timer.begin, s); }
                ops::gated_rmsnorm(o, *gdn.gdn_norm, z, kCfg.rms_eps, gdn_output_gate<Variant>(), on, s);
                if (timing) { lap(timer.begin, timer.g_norm, acc_g_norm); cudaEventRecord(timer.begin, s); }
                Variant::gdn_output_projection(on.view({kCfg.value_dim, total}), *gdn.out_proj, x,
                                               Phase::Prefill, work_, s);
                if (timing) { lap(timer.begin, timer.g_out, acc_g_out); timer.t_g_ctrl += acc_g_ctrl; timer.t_g_proj += acc_g_proj; timer.t_g_conv += acc_g_conv; timer.t_g_extract += acc_g_extract; timer.t_g_scan += acc_g_scan; timer.t_g_norm += acc_g_norm; timer.t_g_out += acc_g_out; cudaEventRecord(timer.begin, s); }
                if (timing) { lap(timer.begin, timer.gdn, acc_gdn); }
                if (timing) { acc_gdn += acc_g_proj + acc_g_conv + acc_g_scan + acc_g_out; }
            }
            {
                auto mlp_scope = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, s); }
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, Phase::Prefill);
                if (timing) { lap(timer.begin, timer.mlp_gdn, acc_mlp_gdn); }
            }
        }
    }
    if (timing) {
        timer.t_attn += acc_attn; timer.t_mlp_full += acc_mlp_full;
        timer.t_gdn += acc_gdn;   timer.t_mlp_gdn += acc_mlp_gdn;
        timer.chunks += 1;        timer.tokens += static_cast<std::uint64_t>(prefill_cols);
        if (timer.chunks % 32 == 0) {
            const double tot = timer.t_attn + timer.t_mlp_full + timer.t_gdn + timer.t_mlp_gdn;
            std::fprintf(stderr,
                         "prefill-families(mixed): %llu chunks, %.0f prefill tokens/chunk + %d "
                         "decode cols: attn %.1f%% mlp(full) %.1f%% gdn %.1f%% mlp(gdn) %.1f%% | "
                         "%.2f ms/chunk, %.0f prefill tok/s inside the layer loop\n",
                         static_cast<unsigned long long>(timer.chunks),
                         static_cast<double>(timer.tokens) / timer.chunks, batch,
                         100 * timer.t_attn / tot, 100 * timer.t_mlp_full / tot,
                         100 * timer.t_gdn / tot, 100 * timer.t_mlp_gdn / tot,
                         tot / timer.chunks, 1000.0 * timer.tokens / tot);
            print_gdn_subsplit(timer, "prefill-families(mixed)");
        }
    }

    Tensor xf = prefill_hidden_.data != nullptr
                    ? matrix_window(prefill_hidden_, total)
                    : work_.alloc(DType::BF16, {kCfg.hidden, total});
    if (!stage_finishes()) {
        stage_export(x, s);
    } else {
    Hooks::finish(weights_, x, kCfg.rms_eps, xf, work_, s);

    if (batch > 0) {
        Tensor xf_decode = xf.slice(1, prefill_cols, batch);
        CUDA_CHECK(cudaMemcpyAsync(decode.hidden.data, xf_decode.data,
                                   static_cast<std::size_t>(kCfg.hidden) * batch * 2,
                                   cudaMemcpyDeviceToDevice, s));
        Tensor logits_decode = decode.logits;
        ops::linear(xf_decode, *lm_head_, logits_decode, s);
    }

    // Every segment that finishes samples in this round. Their last hidden columns are
    // gathered into one [hidden, F] window so a single lm_head serves all of them - F separate
    // GEMMs would re-read the whole vocabulary projection F times.
    if (finalizers > 0) {
        int slot = 0;
        for (std::size_t sg = 0; sg < segments.size(); ++sg) {
            if (!segments[sg].finalize) { continue; }
            const int len  = static_cast<int>(segments[sg].ids.size());
            Tensor last_xf = xf.slice(1, segment_begin[sg] + len - 1, 1);
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<char*>(finalize.hidden.data) +
                    static_cast<std::size_t>(slot) * kCfg.hidden * 2,
                last_xf.data, static_cast<std::size_t>(kCfg.hidden) * 2,
                cudaMemcpyDeviceToDevice, s));
            const std::int32_t position = segments[sg].kv_base + len;
            Tensor position_slot        = finalize.positions.slice(0, slot, 1);
            ops::set_i32_scalar(position_slot, position, s);
            if (finalize.rope_positions.data != nullptr) {
                Tensor rope_slot = finalize.rope_positions.slice(0, slot, 1);
                ops::set_i32_scalar(rope_slot, position + rope_delta_, s);
            }
            ++slot;
        }
        Tensor gathered = finalize.hidden.data != nullptr
                              ? finalize.hidden.slice(1, 0, finalizers)
                              : xf.slice(1, segment_begin[segments.size() - 1] +
                                                static_cast<int>(segments.back().ids.size()) - 1,
                                         1);
        Tensor logits   = finalize.logits.slice(1, 0, finalizers);
        ops::linear(gathered, *lm_head_, logits, s);
        Tensor sampled   = finalize.tokens.slice(0, 0, finalizers);
        Tensor positions_out = finalize.positions.slice(0, 0, finalizers);
        ops::sample(logits, sampled, kCfg.token_domain, finalize.sampling, positions_out,
                    ops::kSamplePurposePrefill, work_, s);
    }
    } // stage_finishes
    const bool is_last = finalizers > 0;

    ctx_.synchronize();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(prefill_cols),
                              .finalized        = is_last};
}

// ---- Prefill CUDA graphs (PATCHES.md #27) -----------------------------------

// The capturable chunk body at a padded bucket length. Everything varying per
// replay flows through the family's pinned staging: the ingress {base, valid}
// lands in a device mirror via an in-graph H2D copy, token ids land in the
// arena ids tensor (replay-stable address: deterministic recipe sequence after
// work_.reset()), positions derive on device as iota + base, and rope
// positions always derive as positions + io.rope_delta (the eager body skips
// that op when the delta is zero; here the shape must be static, and a zero
// delta makes it the identity). The attention envelope is a static bound —
// the prefill kernels take per-query visibility from positions on device.
void TextContext::prefill_graph_window(std::int32_t bucket) {
    cudaStream_t s             = ctx_.stream;
    PrefillGraphFamily& family = *prefill_graph_family_;
    work_.reset();
    const auto roots = workspace_recipe::text_prefill_roots<TextConfig>(work_, bucket, 0, 0);

    Tensor ingress = family.ingress_device();
    CUDA_CHECK(cudaMemcpyAsync(ingress.data, family.ingress_staging(),
                               sizeof(PrefillGraphIngress), cudaMemcpyHostToDevice, s));
    Tensor ids_device = roots.ids;
    CUDA_CHECK(cudaMemcpyAsync(ids_device.data, family.ids_staging(),
                               static_cast<std::size_t>(bucket) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, s));

    const Tensor base        = ingress.slice(0, 0, 1);
    graph_pad_valid_storage_ = ingress.slice(0, 1, 1);

    Tensor positions = roots.positions;
    ops::offset_i32_positions(family.iota_window(bucket), base, positions, s);
    Tensor rope_positions = family.rope_positions_window(bucket);
    ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);

    ScopedPositions scoped_cache(active_cache_positions_, positions);
    ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
    const ops::GqaExecutionEnvelope envelope{1, family.kv_capacity()};
    ScopedEnvelope scoped_envelope(active_gqa_envelope_, envelope);
    ScopedValue<const Tensor*> scoped_pad(graph_pad_valid_, &graph_pad_valid_storage_);

    Tensor x = roots.residual;
    if constexpr (Hooks::prologue) {
        prologue_ = prologue_staging::single_segment_columns(
            work_, ids_device, bucket, linear_state_current_slot_, &graph_pad_valid_storage_, 0, s);
    }
    if (stage_embeds()) { Hooks::embed(weights_, ids_device, x, work_, s); } else { stage_import(x, s); }
    NullTap tap;
    run_layers(x, Phase::Prefill, tap);

    if (prefill_hidden_.data == nullptr) {
        throw std::logic_error("prefill graph body requires the persistent prefill hidden store");
    }
    Tensor xf = matrix_window(prefill_hidden_, bucket);
    if (stage_finishes()) {
        Hooks::finish(weights_, x, kCfg.rms_eps, xf, work_, s);
    } else {
        stage_export(x, s);
    }
}

// The capturable mixed-round body (PATCHES.md #30): one forward over
// [prefill-bucket | decode-bucket] columns. The prefill side pads exactly
// like prefill_graph_window: the ingress {base, valid} device mirror drives
// positions and the pad discipline (valid-aware conv, g/beta zeroed over the
// prefill window only, KV garbage confined to positions the next real chunk
// overwrites). The decode side is padded host-side by the caller, which
// duplicates a live lane's ingress row into the pad rows: a pad column then
// recomputes that lane's own update and every state/KV write lands as the
// identical bytes the real column writes. The GDN prefill state runs on the
// shared scratch slot (the graphed-prompt discipline), so the slot baked at
// capture is lane-independent.
void TextContext::mixed_graph_window(std::int32_t chunk_bucket, std::int32_t batch_bucket) {
    cudaStream_t s             = ctx_.stream;
    PrefillGraphFamily& family = *prefill_graph_family_;
    if (rope_delta_ != 0) {
        throw std::logic_error("mixed graph does not support rope-delta prompts");
    }
    const MixedDecodeSlice& decode = mixed_graph_decode_;
    const int prefill_cols         = chunk_bucket;
    const int batch                = batch_bucket;
    const int total                = prefill_cols + batch;
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    work_.reset();
    const auto roots = workspace_recipe::text_prefill_roots<TextConfig>(work_, total, 0, 0);

    Tensor ingress = family.ingress_device();
    CUDA_CHECK(cudaMemcpyAsync(ingress.data, family.ingress_staging(),
                               sizeof(PrefillGraphIngress), cudaMemcpyHostToDevice, s));
    Tensor ids_device  = roots.ids;
    Tensor ids_prefill = ids_device.slice(0, 0, prefill_cols);
    CUDA_CHECK(cudaMemcpyAsync(ids_prefill.data, family.ids_staging(),
                               static_cast<std::size_t>(prefill_cols) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, s));
    Tensor ids_decode = ids_device.slice(0, prefill_cols, batch);
    CUDA_CHECK(cudaMemcpyAsync(ids_decode.data, decode.ids.data,
                               static_cast<std::size_t>(batch) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, s));

    const Tensor base  = ingress.slice(0, 0, 1);
    const Tensor valid = ingress.slice(0, 1, 1);

    Tensor positions         = roots.positions;
    Tensor positions_prefill = positions.slice(0, 0, prefill_cols);
    ops::offset_i32_positions(family.iota_window(chunk_bucket), base, positions_prefill, s);
    Tensor positions_decode = positions.slice(0, prefill_cols, batch);
    CUDA_CHECK(cudaMemcpyAsync(positions_decode.data, decode.cache_positions.data,
                               static_cast<std::size_t>(batch) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, s));

    const ops::GqaExecutionEnvelope prefill_envelope{1, family.kv_capacity()};
    // Banded, matching the ordinary decode graphs (the band is part of the
    // graph key, so a replay never sees a frontier outside it).
    const ops::GqaExecutionEnvelope decode_envelope = mixed_graph_decode_.envelope;

    if constexpr (Hooks::prologue) {
        // Prefill window: one segment on the scratch slot, ending at the ingress valid count;
        // decode columns: one-column segments on the lanes' slots.
        prologue_ = prologue_staging::single_segment_columns(work_, ids_device, total,
                                                             linear_state_current_slot_, &valid,
                                                             0, s);
        if (batch > 0) {
            Tensor begin_decode = prologue_.segment_begin.slice(0, prefill_cols, batch);
            ops::fill_i32_positions(begin_decode, prefill_cols, s);
            Tensor last_decode = prologue_.segment_last.slice(0, prefill_cols, batch);
            prologue_staging::fill_i32(last_decode, 1, s);
            Tensor slots_decode = prologue_.slots.slice(0, prefill_cols, batch);
            CUDA_CHECK(cudaMemcpyAsync(slots_decode.data, decode.linear_state_slots.data,
                                       static_cast<std::size_t>(batch) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToDevice, s));
        }
    }
    Tensor x = roots.residual;
    if (stage_embeds()) { Hooks::embed(weights_, ids_device, x, work_, s); } else { stage_import(x, s); }

    for (int layer = stage_first_; layer < stage_last_; ++layer) {
        Hooks::layer_prologue(weights_, layer, x, prologue_, ple_state_, work_, ctx_.stream);
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            {
                auto mixer_scope      = work_.scope();
                const auto projection = workspace_recipe::text_attention_projection<TextConfig>(
                    work_, total);
                Tensor h = projection.hidden;
                Hooks::attention_norm(x, *full.input_norm, kCfg.rms_eps, *full.projection, h,
                                      work_, s);
                Tensor q         = projection.query.view({kCfg.head_dim, kCfg.n_q, total});
                Tensor gate      = projection.gate.view({kCfg.head_dim, kCfg.n_q, total});
                Tensor k         = projection.key.view({kCfg.head_dim, kCfg.n_kv, total});
                Tensor v         = projection.value.view({kCfg.head_dim, kCfg.n_kv, total});
                Tensor q_flat    = q.view({kCfg.q_size, total});
                Tensor gate_flat = gate.view({kCfg.q_size, total});
                Tensor k_flat    = k.view({kCfg.kv_size, total});
                Tensor v_flat    = v.view({kCfg.kv_size, total});
                Variant::attention_projection(h, *full.projection, q_flat, gate_flat, k_flat,
                                              v_flat, Phase::Prefill, work_, s);

                const auto results = workspace_recipe::text_attention_results<TextConfig>(work_,
                                                                                         total);
                Tensor qn = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, total});
                Tensor kn = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, total});
                ops::rmsnorm(q, *full.q_norm, kCfg.rms_eps, true, qn, s);
                ops::rmsnorm(k, *full.k_norm, kCfg.rms_eps, true, kn, s);

                Tensor rope_positions = roots.positions;
                Tensor rope_all       = rope_positions.view({total});
                if (batch > 0) {
                    Tensor rope_decode = rope_positions.slice(0, prefill_cols, batch);
                    CUDA_CHECK(cudaMemcpyAsync(rope_decode.data, decode.rope_positions.data,
                                               static_cast<std::size_t>(batch) *
                                                   sizeof(std::int32_t),
                                               cudaMemcpyDeviceToDevice, s));
                }
                ops::rope(rope_all, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

                Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, total});
                {
                    Tensor qa = qn.slice(2, 0, prefill_cols);
                    Tensor ka = kn.slice(2, 0, prefill_cols);
                    Tensor va = v.slice(2, 0, prefill_cols);
                    Tensor aa = a.slice(2, 0, prefill_cols);
                    auto segment_scope = work_.scope();
                    const ops::GqaBlockMask segment_selection = text_indexer_selection(
                        full, h.slice(1, 0, prefill_cols), prefill_cols, positions_prefill,
                        rope_all.slice(0, 0, prefill_cols), io_.text_kv_table_row, prefill_cols,
                        static_cast<std::int32_t>(prefill_envelope.max_visible_keys),
                        batch_text_kv_->batch_layer_view(fidx));
                    ops::gqa_attention(qa, ka, va, positions_prefill, Tensor{},
                                       io_.text_kv_table_row, kAttnScale,
                                       batch_text_kv_->batch_layer_view(fidx), prefill_envelope,
                                       work_, aa, s, segment_selection);
                }
                if (batch > 0) {
                    Tensor qb = qn.slice(2, prefill_cols, batch)
                                    .view({kCfg.head_dim, kCfg.n_q, 1, batch});
                    Tensor kb = kn.slice(2, prefill_cols, batch)
                                    .view({kCfg.head_dim, kCfg.n_kv, 1, batch});
                    Tensor vb = v.slice(2, prefill_cols, batch)
                                    .view({kCfg.head_dim, kCfg.n_kv, 1, batch});
                    Tensor ab = a.slice(2, prefill_cols, batch)
                                    .view({kCfg.head_dim, kCfg.n_q, 1, batch});
                    Tensor position_batch = decode.cache_positions.view({1, batch});
                    auto decode_scope = work_.scope();
                    const ops::GqaBlockMask decode_selection = text_indexer_selection(
                        full, h.slice(1, prefill_cols, batch), batch, decode.cache_positions,
                        rope_all.slice(0, prefill_cols, batch), decode.kv_table_rows, 1,
                        static_cast<std::int32_t>(decode_envelope.max_visible_keys),
                        batch_text_kv_->batch_layer_view(fidx));
                    ops::gqa_attention(qb, kb, vb, position_batch, Tensor{}, decode.kv_table_rows,
                                       kAttnScale, batch_text_kv_->batch_layer_view(fidx),
                                       decode_envelope, work_, ab, s, decode_selection);
                }
                ops::sigmoid_mul(gate, a, s);
                Variant::attention_output_projection(a.view({kCfg.q_size, total}), *full.o_proj, x,
                                                     Phase::Prefill, work_, s);
            }
            {
                auto mlp_scope = work_.scope();
                mlp_tail(full.post_attn_norm, full.mlp, x, Phase::Prefill);
            }
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            {
                auto mixer_scope   = work_.scope();
                const auto control = workspace_recipe::gdn_control<TextConfig>(work_, total);
                Tensor h           = control.hidden;
                Tensor g           = control.g;
                Tensor beta        = control.beta;
                Variant::gdn_norm_control_projection(x, *gdn.input_norm, kCfg.rms_eps,
                                                     *gdn.projection, h, g, beta, work_, s);
                {
                    // Zero g/beta over the prefill window's pad columns only
                    // (the decode columns behind the window are live lanes).
                    Tensor g_prefill    = g.slice(1, 0, prefill_cols);
                    Tensor beta_prefill = beta.slice(1, 0, prefill_cols);
                    ops::mask_columns_zero(g_prefill, valid, s);
                    ops::mask_columns_zero(beta_prefill, valid, s);
                }

                const auto projection = workspace_recipe::gdn_projection<TextConfig>(work_, total);
                Tensor z  = projection.output_gate.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, total});
                Tensor qc = projection.query;
                Tensor kc = projection.key;
                Tensor vc = projection.value;
                const auto conv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, total);
                Tensor qkv      = conv.projected;
                Variant::gdn_input_projection(h, *gdn.projection, qkv, z, Phase::Prefill, work_, s);
                Tensor qkv_c = conv.convolved;
                {
                    Tensor qkv_a  = qkv.slice(1, 0, prefill_cols);
                    Tensor qkv_ca = qkv_c.slice(1, 0, prefill_cols);
                    Tensor conv_state =
                        state_.conv_slot(static_cast<std::uint32_t>(gidx),
                                         linear_state_current_slot_);
                    ops::causal_conv1d_silu(qkv_a, *gdn.conv1d, conv_state, conv_state, qkv_ca,
                                            valid, s);
                }
                if (batch > 0) {
                    Tensor qkv_b  = qkv.slice(1, prefill_cols, batch)
                                       .view({kCfg.conv_dim, 1, batch});
                    Tensor qkv_cb = qkv_c.slice(1, prefill_cols, batch)
                                        .view({kCfg.conv_dim, 1, batch});
                    ops::causal_conv1d_silu_snapshot(qkv_b, *gdn.conv1d,
                                                     state_.conv.at(static_cast<std::size_t>(gidx)),
                                                     Tensor{}, decode.linear_state_slots,
                                                     decode.linear_state_slots, qkv_cb, s);
                }
                ops::extract_bf16_columns(qkv_c, 0, qc, s);
                ops::extract_bf16_columns(qkv_c, kCfg.key_dim, kc, s);
                ops::extract_bf16_columns(qkv_c, 2 * kCfg.key_dim, vc, s);

                Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, total});
                Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, total});
                Tensor vv          = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, total});
                Tensor o = workspace_recipe::gdn_recurrent_output<TextConfig>(work_, total)
                               .view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, total});
                {
                    Tensor qa = q_recurrent.slice(2, 0, prefill_cols);
                    Tensor ka = k_recurrent.slice(2, 0, prefill_cols);
                    Tensor va = vv.slice(2, 0, prefill_cols);
                    Tensor ga = g.slice(1, 0, prefill_cols);
                    Tensor ba = beta.slice(1, 0, prefill_cols);
                    Tensor oa = o.slice(2, 0, prefill_cols);
                    Tensor recurrent_state =
                        state_.recurrent_slot(static_cast<std::uint32_t>(gidx),
                                              linear_state_current_slot_);
                    ops::gated_delta_net(qa, ka, va, ga, ba, kGdnScale, true, work_,
                                         recurrent_state, oa, s);
                }
                if (batch > 0) {
                    Tensor qb = q_recurrent.slice(2, prefill_cols, batch)
                                    .view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1, batch});
                    Tensor kb = k_recurrent.slice(2, prefill_cols, batch)
                                    .view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, 1, batch});
                    Tensor vb = vv.slice(2, prefill_cols, batch)
                                    .view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1, batch});
                    Tensor gb = g.slice(1, prefill_cols, batch).view({kCfg.gdn_v_heads, 1, batch});
                    Tensor bb =
                        beta.slice(1, prefill_cols, batch).view({kCfg.gdn_v_heads, 1, batch});
                    Tensor ob = o.slice(2, prefill_cols, batch)
                                    .view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, 1, batch});
                    ops::gated_delta_net_snapshot(
                        qb, kb, vb, gb, bb, kGdnScale, true,
                        state_.recurrent.at(static_cast<std::size_t>(gidx)), Tensor{},
                        decode.linear_state_slots, decode.linear_state_slots, ob, s);
                }
                Tensor on = workspace_recipe::gdn_normalized_output<TextConfig>(work_, total)
                                .view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, total});
                ops::gated_rmsnorm(o, *gdn.gdn_norm, z, kCfg.rms_eps, gdn_output_gate<Variant>(), on, s);
                Variant::gdn_output_projection(on.view({kCfg.value_dim, total}), *gdn.out_proj, x,
                                               Phase::Prefill, work_, s);
            }
            {
                auto mlp_scope = work_.scope();
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, Phase::Prefill);
            }
        }
    }

    if (prefill_hidden_.data == nullptr) {
        throw std::logic_error("mixed graph body requires the persistent prefill hidden store");
    }
    Tensor xf = matrix_window(prefill_hidden_, total);
    if (!stage_finishes()) {
        stage_export(x, s);
        return;
    }
    Hooks::finish(weights_, x, kCfg.rms_eps, xf, work_, s);

    if (batch == 0) { return; }
    Tensor xf_decode = xf.slice(1, prefill_cols, batch);
    CUDA_CHECK(cudaMemcpyAsync(decode.hidden.data, xf_decode.data,
                               static_cast<std::size_t>(kCfg.hidden) * batch * 2,
                               cudaMemcpyDeviceToDevice, s));
    Tensor logits_decode = decode.logits;
    ops::linear(xf_decode, *lm_head_, logits_decode, s);
}

// Stages one mixed round into the family, captures the (chunk bucket, batch
// bucket) pair on first use, and replays it. The caller passes the decode
// slice at the batch bucket with pad rows already duplicated from a live
// row, and runs the eager epilogues (scatter, sample, egress) at the real
// row count afterwards. Returns false when the family is dead or the bucket
// does not fit the workspace window: the caller runs the eager mixed body.
bool TextContext::try_mixed_graph_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                        std::uint32_t nominal, const MixedDecodeSlice& decode,
                                        std::int32_t batch_bucket,
                                        std::int32_t band) {
    if (prefill_graph_family_ == nullptr || rope_delta_ != 0) { return false; }
    PrefillGraphFamily& family = *prefill_graph_family_;
    if (begin >= full_ids.size() || nominal == 0 || nominal > full_ids.size() - begin) {
        return false;
    }
    if (decode.ids.ne[0] != batch_bucket) { return false; }
    const int len                   = static_cast<int>(nominal);
    const std::int32_t chunk_bucket = family.bucket_for(len);
    if (chunk_bucket < len ||
        chunk_bucket + batch_bucket > static_cast<std::int32_t>(prefill_chunk_)) {
        return false;
    }

    std::int32_t* staging = family.ids_staging();
    const auto ids        = full_ids.subspan(begin, nominal);
    for (int i = 0; i < len; ++i) { staging[i] = ids[static_cast<std::size_t>(i)]; }
    for (std::int32_t i = len; i < chunk_bucket; ++i) { staging[i] = 0; }
    PrefillGraphIngress* ingress = family.ingress_staging();
    ingress->base                = static_cast<std::int32_t>(text_kv_base_);
    ingress->valid               = len;
    ingress->batch_valid         = batch_bucket;
    mixed_graph_decode_          = decode;

    DecodeGraphExecutable* executable = family.ensure(
        PrefillGraphFamily::mixed_key(chunk_bucket, batch_bucket, band),
        [this, chunk_bucket, batch_bucket] { mixed_graph_window(chunk_bucket, batch_bucket); });
    if (executable == nullptr) { return false; }
    executable->launch(ctx_.stream);
    return true;
}

void TextContext::precapture_prefill_graphs(std::int32_t effective_chunk) {
    if (prefill_graph_family_ == nullptr || effective_chunk <= 0) { return; }
    PrefillGraphFamily& family = *prefill_graph_family_;
    for (std::int32_t step = 128;; step += 128) {
        const std::int32_t bucket = step < effective_chunk ? step : effective_chunk;
        PrefillGraphIngress* ingress = family.ingress_staging();
        ingress->base                = 0;
        ingress->valid               = bucket;
        std::fill_n(family.ids_staging(), bucket, 0);
        if (family.ensure(bucket, [this, bucket] { prefill_graph_window(bucket); }) == nullptr) {
            return;
        }
        if (bucket == effective_chunk) { return; }
    }
}

// Stages one real chunk into the family, captures the bucket on first use, and
// replays it; then runs the eager epilogues the graph excludes (the final
// sample and the rewrite-checkpoint hidden copy — both need the real length,
// which the graph deliberately does not bake). Returns false when the family
// is dead or capture fails: the caller runs the unchanged eager body.
bool TextContext::try_prefill_graph_chunk(std::span<const int> ids, int t0, int len, int base_i,
                                          bool is_last, int checkpoint_rel) {
    PrefillGraphFamily& family = *prefill_graph_family_;
    if (len <= 0) { return false; }
    const std::int32_t bucket = family.bucket_for(len);
    if (bucket < len) { return false; }

    std::int32_t* staging = family.ids_staging();
    for (int i = 0; i < len; ++i) { staging[i] = ids[static_cast<std::size_t>(t0 + i)]; }
    for (std::int32_t i = len; i < bucket; ++i) { staging[i] = 0; }
    PrefillGraphIngress* ingress = family.ingress_staging();
    ingress->base                = base_i + t0;
    ingress->valid               = len;

    DecodeGraphExecutable* executable =
        family.ensure(bucket, [this, bucket] { prefill_graph_window(bucket); });
    if (executable == nullptr) { return false; }
    executable->launch(ctx_.stream);

    cudaStream_t s = ctx_.stream;
    const int T    = static_cast<int>(ids.size());
    if (is_last) {
        Tensor xf      = matrix_window(prefill_hidden_, len);
        Tensor last_xf = xf.slice(1, len - 1, 1);
        Tensor logits  = matrix_window(io_.logits, 1);
        ops::linear(last_xf, *lm_head_, logits, s);
        ops::set_i32_scalar(io_.pos, base_i + T, s);
        ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
        work_.reset();
        if (sampling_config_ != nullptr) {
            ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_, io_.pos,
                        ops::kSamplePurposePrefill, work_, s);
        } else {
            ops::argmax(logits, io_.token, kCfg.token_domain, s);
        }
    }
    if (checkpoint_rel > 0 && t0 + len == checkpoint_rel &&
        rewrite_checkpoint_hidden_output_ != nullptr) {
        require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16, {kCfg.hidden, 1},
                             "rewrite checkpoint hidden output");
        Tensor xf                      = matrix_window(prefill_hidden_, len);
        const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
        CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data, checkpoint_hidden.data,
                                   checkpoint_hidden.bytes(), cudaMemcpyDeviceToDevice, s));
    }
    return true;
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64         = static_cast<std::int64_t>(base);
    const std::int64_t checkpoint_abs = prefill_rewrite_checkpoint_frontier_;
    const bool has_rewrite_checkpoint =
        checkpoint_abs > base64 && checkpoint_abs <= base64 + static_cast<std::int64_t>(T);
    const int checkpoint_rel =
        has_rewrite_checkpoint ? static_cast<int>(checkpoint_abs - base64) : -1;
    const std::int32_t rewrite_checkpoint_slot = linear_state_rewrite_checkpoint_slot_;
    if (checkpoint_rel > 0 && rewrite_checkpoint_slot == kNoRewriteCheckpointSlot) {
        throw std::logic_error("rewrite checkpoint capture requested with checkpoints disabled");
    }

    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (checkpoint_rel > 0 && t0 < checkpoint_rel && t0 + len > checkpoint_rel) {
            len = checkpoint_rel - t0;
        }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        bool graph_chunk_ran = false;
        PrefillWindowLaps window_laps(s, len);
        if constexpr (!Tap::enabled) {
            if (prefill_graph_family_ != nullptr && multimodal == nullptr &&
                !prepare_mtp_prompt) {
                graph_chunk_ran =
                    try_prefill_graph_chunk(ids, t0, len, base_i, is_last, checkpoint_rel);
                window_laps.mark_graph();
            }
        }
        if (!graph_chunk_ran) {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace_recipe::text_prefill_roots<TextConfig>(
                work_, len, rope_axes, static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::GqaExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_gqa_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            if constexpr (Hooks::prologue) {
                prologue_ = prologue_staging::single_segment_columns(
                    work_, ids_device, len, linear_state_current_slot_, nullptr, len, s);
            }
            if (stage_embeds()) { Hooks::embed(weights_, ids_device, x, work_, s); } else { stage_import(x, s); }
            window_laps.mark_pre();
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            window_laps.mark_layers();
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {kCfg.hidden, len});
            if (stage_finishes()) {
                Hooks::finish(weights_, x, kCfg.rms_eps, xf, work_, s);
            } else {
                stage_export(x, s);
            }

            if (is_last && stage_finishes()) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                ops::linear(last_xf, *lm_head_, logits, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_, io_.pos,
                                ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token, kCfg.token_domain, s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_6::MtpAlignmentWindow mtp_window = qwen3_6::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                std::vector<int> mtp_ids_host(static_cast<std::size_t>(len));
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                for (int j = 0; j < prompt_columns; ++j) {
                    mtp_ids_host[static_cast<std::size_t>(j)] =
                        alignment_ids[static_cast<std::size_t>(mtp_window.shifted_embedding_begin) +
                                      static_cast<std::size_t>(j)];
                }
                if (mtp_window.final_column_uses_generated_token) {
                    int next_token = 0;
                    CUDA_CHECK(cudaStreamSynchronize(s));
                    CUDA_CHECK(cudaMemcpy(&next_token, io_.token.data, sizeof(next_token),
                                          cudaMemcpyDeviceToHost));
                    mtp_ids_host[static_cast<std::size_t>(len - 1)] = next_token;
                }

                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                copy_i32(mtp_ids_host.data(), mtp_ids, s);
                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_6::MtpVisualOverlap overlap = qwen3_6::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace_recipe::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_6::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token     = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token     = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden    = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::GqaExecutionEnvelope ar_envelope{ar_visible, ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (checkpoint_rel > 0 && t0 + len == checkpoint_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {kCfg.hidden, 1}, "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, checkpoint_rel > 0 && t0 + len == checkpoint_rel);
        }

        if (checkpoint_rel > 0 && t0 + len == checkpoint_rel) {
            state_.copy_slot(linear_state_current_slot_, rewrite_checkpoint_slot, s);
            if (ple_state_ != nullptr && !ple_state_->empty()) {
                ple_state_->copy_slot(linear_state_current_slot_, rewrite_checkpoint_slot, s);
            }
        }

        t0 += len;
        break;
    }

    prefill_rewrite_checkpoint_frontier_ = -1;

    ctx_.synchronize();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_6::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
