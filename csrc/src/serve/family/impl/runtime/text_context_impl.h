#include "api/ops/ngram_ple.h"
#include <cuda.h>
#include "family/impl/runtime/instance.h"
#include "family/impl/runtime/text_context.h"
#include "family/impl/runtime/prefill_graph.h"
#include "family/impl/runtime/workspace_recipe.h"

#include "core/nvtx.h"
#include "family/impl/runtime/visual_scatter.h"
#include "family/impl/runtime/vision_context.h"
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <api/family/vision_control.h>
#include "api/ops/argmax.h"
#include "api/ops/next_token_nll.h"
#include "api/ops/sampled_logprob.h"
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
#include "api/ops/logit_softcap.h"
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

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule {
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
    if ((!prefill && !batch) || layers.empty() || layers.size() > 32) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
    if (stage.features) {
        if (active_tokens > stage.features->ne[1]) {
            throw std::logic_error("DFlash pipeline feature storage is too small");
        }
        const auto bytes = static_cast<std::size_t>(stage.features->ne[0]) * sizeof(std::uint16_t);
        if (stage.first > 0) {
            CUDA_CHECK(cudaMemcpy2DAsync(stage.features->data, stage.features->nb[1],
                static_cast<const std::byte*>(stage.import_pinned) + stage.residual_bytes,
                stage.column_bytes, bytes, active_tokens, cudaMemcpyHostToDevice, stream));
            for (std::size_t i = 0; i < layers.size(); ++i) {
                if (layers[i] < stage.first) { captured_mask |= 1U << i; }
            }
        } else {
            CUDA_CHECK(cudaMemsetAsync(stage.features->data, 0, bytes * active_tokens, stream));
        }
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination = stage.features ? stage.features : batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr && stage.features == nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > destination->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(destination->nb[1]);
    auto* target                    = static_cast<std::byte*>(destination->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    std::uint32_t complete_mask = 0;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        if (!stage.features || layers[i] < stage.last) { complete_mask |= 1U << i; }
    }
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (stage.features) {
        const auto bytes = static_cast<std::size_t>(stage.features->ne[0]) * sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(stage.export_pinned) + stage.residual_bytes,
            stage.column_bytes, stage.features->data, stage.features->nb[1], bytes, active_tokens,
            cudaMemcpyDeviceToHost, stream));
        if (batch_features) {
            Tensor compact = stage.features->slice(1, 0, active_tokens).view(
                {stage.features->ne[0], batch_width, batch_size});
            ops::scatter_bf16_batch(compact, *batch_lanes, *batch_valid_columns, *batch_features, stream);
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(stage.export_pinned) +
                stage.residual_bytes + bytes + sizeof(std::int32_t), stage.column_bytes,
                source.data, sizeof(std::int32_t), sizeof(std::int32_t), active_tokens,
                cudaMemcpyDeviceToHost, stream));
        }
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
                         family::PagedKVCacheView kv, LinearAttentionStatePool& state,
                         family::RoundState& io, Tensor& prefill_hidden,
                         std::uint32_t prefill_chunk, std::uint32_t text_kv_base,
                         family::PagedKVCacheView mtp_kv,
                         const family::PagedKVCache* batch_text_kv,
                         const family::PagedKVCache* batch_mtp_kv)
    : ctx_(ctx), weights_(weights), cfg_(weights.geometry), work_(work), kv_(kv),
      mtp_kv_(mtp_kv), state_(state), io_(io),
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

namespace {
[[nodiscard]] bool stage_trace_enabled() {
    static const bool enabled = std::getenv("SUROGATE_SERVE_TRACE_STAGE") != nullptr;
    return enabled;
}
} // namespace

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

    // A trunk-block draft head carries no fixed-tail weights: its block is an ordinary layer,
    // bound where the trunk's are, and the family reads it through the target's hooks.
    // The head's KV and frames exist on every pipeline stage (the verify's inputs are theirs),
    // the head's weights only where it runs; a stage without them leaves `mtp_` empty and
    // `mtp_weights()` refuses any use.
    if (mtp_enabled() && !mtp_block_is_trunk_layer<Variant>() && weights_.mtp) {
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

    full_.resize(static_cast<std::size_t>(cfg_.n_full()));
    gdn_.resize(static_cast<std::size_t>(cfg_.n_gdn()));
    gdn_in_a_.resize(static_cast<std::size_t>(cfg_.n_gdn()));
    gdn_in_b_.resize(static_cast<std::size_t>(cfg_.n_gdn()));
    gdn_conv1d_views_.resize(static_cast<std::size_t>(cfg_.n_gdn()));
    for (int layer = 0; layer < cfg_.n_layers; ++layer) {
        if (cfg_.is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(cfg_.full_idx(layer))];
            const auto& source =
                weights_.full_layers[static_cast<std::size_t>(cfg_.full_idx(layer))];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.post_mixer);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(cfg_.gdn_idx(layer));
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
    if (!mtp_enabled() || mtp_.payload == nullptr) {
        throw std::runtime_error("MTP draft weights are not enabled on this device");
    }
    return mtp_;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({cfg_.hidden, T});

    auto roots = workspace_recipe::mtp_stem(work_, cfg_geometry(), T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 || input_embeddings->ne[0] != cfg_.hidden ||
            input_embeddings->numel() != static_cast<std::int64_t>(cfg_.hidden) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({cfg_.hidden, T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, *mtp_.pre_fc_norm_embedding, cfg_.rms_eps, norm_unit_offset<Variant>(), e,
                 s);
    ops::rmsnorm(flat_hidden, *mtp_.pre_fc_norm_hidden, cfg_.rms_eps, norm_unit_offset<Variant>(),
                 h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    ops::linear(fc_in, *mtp_.fc, x, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, *mtp_.input_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), ah, s);
}

void TextContext::mtp_attention_output(const Tensor& attention, Tensor& out) {
    family::detail::mtp_attention_output_dispatch<Variant>(
        attention, mtp_.payload->attention, *mtp_.o_proj, out, work_, ctx_.stream);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace_recipe::mtp_attention_projection(work_, cfg_geometry(), T);
    Tensor q              = projection.query.view({cfg_.head_dim, cfg_.n_q, T});
    Tensor k              = projection.key.view({cfg_.head_dim, cfg_.n_kv, T});
    Tensor gate           = projection.gate.view({cfg_.head_dim, cfg_.n_q, T});
    Tensor v              = projection.value.view({cfg_.head_dim, cfg_.n_kv, T});
    Tensor q_flat         = q.view({cfg_.q_size, T});
    Tensor gate_flat      = gate.view({cfg_.q_size, T});
    Tensor k_flat         = k.view({cfg_.kv_size, T});
    Tensor v_flat         = v.view({cfg_.kv_size, T});
    Variant::mtp_attention_projection(ah, mtp_.payload->attention, q_flat, gate_flat, k_flat,
                                      v_flat, work_, s);

    const auto results = workspace_recipe::mtp_attention_results(work_, cfg_geometry(), T);
    // As in the trunk: a target without per-head norms attends over the projection's own
    // output, and the normalised planes stay unwritten.
    Tensor qn = attention_qk_norm<Variant>()
                    ? results.normalized_query.view({cfg_.head_dim, cfg_.n_q, T})
                    : q;
    Tensor kn = attention_qk_norm<Variant>()
                    ? results.normalized_key.view({cfg_.head_dim, cfg_.n_kv, T})
                    : k;
    if constexpr (attention_qk_norm<Variant>()) {
        ops::rmsnorm(q, *mtp_.q_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), qn, s);
        ops::rmsnorm(k, *mtp_.k_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), kn, s);
    }
    Tensor rope_for_op = text_rope_positions<Variant>(
        active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions);
    if constexpr (applies_rotary<Variant>()) {
        ops::rope(rope_for_op, cfg_.rotary_dim, cfg_.rope_theta, qn, kn, s);
    }

    Tensor a = results.attention.view({cfg_.head_dim, cfg_.n_q, T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({cfg_.head_dim, cfg_.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({cfg_.head_dim, cfg_.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({cfg_.head_dim, cfg_.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({cfg_.head_dim, cfg_.n_q, width, active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
                           *active_backend_kv_table_rows_, cfg_.attention_scale,
                           batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, s);
    } else {
        ops::gqa_attention(qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row, cfg_.attention_scale,
                           batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
    }
    // A head whose attention writes no gate rows skips the multiply, as the trunk does.
    if constexpr (kAttentionOutputGate) { apply_attention_gate<Variant>(gate, a, s); }

    const auto post = workspace_recipe::mtp_post_attention(work_, cfg_geometry(), T);
    Tensor o        = post.output;
    mtp_attention_output(a.view({cfg_.q_size, T}), o);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, *mtp_.post_attn_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), mh, s);

    {
        auto post_mixer_scope = work_.scope();
        Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x, work_, s);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({cfg_.hidden, T}); // fixed tail only
    ops::rmsnorm(x, *mtp_.norm, cfg_.rms_eps, norm_unit_offset<Variant>(), flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    if constexpr (!mtp_block_is_trunk_layer<Variant>()) {
        // A pipeline stage without the head has the head's KV but not its weights.
        if (mtp_.payload == nullptr) {
            throw std::runtime_error("MTP draft weights are not on this device");
        }
    }
    auto scratch_scope = work_.scope();
    if constexpr (mtp_block_is_trunk_layer<Variant>()) {
        mtp_forward_trunk_block(ids, hidden, input_embeddings, positions, rope_positions, envelope,
                                mtp_hidden);
    } else {
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
        mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
    }
}

template <class V>
void TextContext::mtp_forward_trunk_block(const Tensor& ids, const Tensor& hidden,
                                          const Tensor* input_embeddings, const Tensor& positions,
                                          const Tensor& rope_positions,
                                          ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden) {
    // Only instantiated for a target that says its head is a trunk block; the others never
    // reach here, and their Variants have none of the hooks below.
    if constexpr (!mtp_block_is_trunk_layer<V>()) {
        throw std::logic_error("this target's draft head is not a trunk block");
    } else {
    cudaStream_t s          = ctx_.stream;
    const int T             = ids.ne[0] * ids.ne[1];
    const std::int32_t wide = weights_.geometry.residual;
    // SUROGATE_SERVE_PREFILL_TIMING=1: where a head-block call spends its time, on the stream.
    static const bool timing = std::getenv("SUROGATE_SERVE_PREFILL_TIMING") != nullptr;
    cudaEvent_t lap[5]{};
    const auto mark = [&](int i) {
        if (!timing) { return; }
        CUDA_CHECK(cudaEventCreateWithFlags(&lap[i], cudaEventDefault));
        CUDA_CHECK(cudaEventRecord(lap[i], s));
    };
    mark(0);

    auto roots = workspace_recipe::mtp_trunk_stem(work_, cfg_geometry(), T,
                                                  input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        emb = input_embeddings->view({cfg_.hidden, T});
    } else {
        emb = roots.embedding;
        ops::embedding(ids.view({T}), *embed_, emb, s);
    }

    // The residual the block runs on, seeded from the next token's embedding and the trunk's
    // own wide residual at the previous position.
    Tensor x = roots.residual;
    V::mtp_fold(weights_, emb, hidden.view({wide, T}), x, work_, s);
    mark(1);

    // One trunk block. The head keeps its own KV plane, one layer deep.
    ScopedValue<const Tensor*> position_binding(active_cache_positions_, &positions);
    ScopedValue<const Tensor*> rope_binding(active_rope_positions_, &rope_positions);
    ScopedValue<const ops::GqaExecutionEnvelope*> envelope_binding(active_gqa_envelope_, &envelope);
    // The same pointer view `bind()` builds for a trunk layer, over the head's weights.
    const auto& source = V::mtp_block(weights_);
    FullLayerW block;
    block.input_norm     = &source.input_norm;
    block.projection     = &source.projection;
    block.o_proj         = &source.output;
    block.q_norm         = &source.query_norm;
    block.k_norm         = &source.key_norm;
    block.post_attn_norm = &source.post_attention_norm;
    block.mlp            = MlpW{&source.post_mixer};
    {
        auto mixer_scope = work_.scope();
        attn_mix(block, x, 0, cfg_.n_layers, Phase::Verify, KvPlane::Mtp);
    }
    mark(2);
    {
        auto post_mixer_scope = work_.scope();
        // The draft head is not a stack layer, so it has no layer index of its own: it takes
        // the same one past the end that `attn_mix` above is given. Nothing reads it -- a
        // block epilogue belongs to a family with per-layer inputs, and none of those has a
        // draft head -- but a real index would name a layer this block is not.
        mlp_tail(block.post_attn_norm, block.mlp, x, cfg_.n_layers, Phase::Verify);
    }
    mark(3);

    // The wide residual is the head's output: the next draft step folds into it again. The
    // collapse to the LM head's width happens at proposal time, not here.
    Tensor out = mtp_hidden.view({wide, T});
    CUDA_CHECK(cudaMemcpyAsync(out.data, x.data, out.bytes(), cudaMemcpyDeviceToDevice, s));
    mark(4);
    if (timing) {
        CUDA_CHECK(cudaEventSynchronize(lap[4]));
        float ms[4]{};
        for (int i = 0; i < 4; ++i) { CUDA_CHECK(cudaEventElapsedTime(&ms[i], lap[i], lap[i + 1])); }
        std::fprintf(stderr,
                     "mtp-timing: head block over %d columns: fold %.1f attn %.1f mixture %.1f "
                     "copy %.1f ms (batch %d)\n",
                     T, ms[0], ms[1], ms[2], ms[3], active_sequence_batch_);
        for (auto& e : lap) { CUDA_CHECK(cudaEventDestroy(e)); }
    }
    }
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
    require_tensor_shape(hidden, DType::BF16, {round_hidden_width(), T}, "MTP prefill hidden");
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
        require_tensor_shape(*final_hidden, DType::BF16, {round_hidden_width(), 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {cfg_.vocab, 1}, "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    if constexpr (mtp_block_is_trunk_layer<Variant>()) {
        // The head's block is a trunk block, so the chunk runs through the same forward the
        // decode rounds use. The bulk columns exist to fill the head's KV; the last one also
        // carries the proposal.
        Tensor chunk_hidden = work_.alloc(DType::BF16, {round_hidden_width(), T});
        mtp_forward_core(ids, hidden, positions, rope_positions, envelope, chunk_hidden,
                         input_embeddings);
        if (!final_chunk) { return; }
        Tensor last = chunk_hidden.slice(1, T - 1, 1);
        CUDA_CHECK(cudaMemcpyAsync(final_hidden->data, last.data, final_hidden->bytes(),
                                   cudaMemcpyDeviceToDevice, s));
        proposal_argmax(*final_hidden, *logits, *draft_token);
        return;
    }
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {cfg_.hidden, 1});
        ah_last = work_.alloc(DType::BF16, {cfg_.hidden, 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {cfg_.kv_size, T});
        Tensor v_flat = work_.alloc(DType::BF16, {cfg_.kv_size, T});
        Variant::mtp_kv_projection(ah, mtp_.payload->attention, k_flat, v_flat, work_, s);
        Tensor k  = k_flat.view({cfg_.head_dim, cfg_.n_kv, T});
        Tensor v  = v_flat.view({cfg_.head_dim, cfg_.n_kv, T});
        Tensor kn = k;
        if constexpr (attention_qk_norm<Variant>()) {
            kn = work_.alloc(DType::BF16, {cfg_.head_dim, cfg_.n_kv, T});
            ops::rmsnorm(k, *mtp_.k_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), kn, s);
        }
        if constexpr (applies_rotary<Variant>()) {
            ops::rope(rope_positions, cfg_.rotary_dim, cfg_.rope_theta, kn, s);
        }
        ops::gqa_kv_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(cfg_.hidden) * dtype_size(DType::BF16);
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
        Tensor q_flat    = work_.alloc(DType::BF16, {cfg_.q_size, 1});
        Tensor gate_flat = work_.alloc(DType::BF16, {cfg_.q_size, 1});
        Variant::mtp_q_gate_projection(ah_last, mtp_.payload->attention, q_flat, gate_flat, work_,
                                       s);
        Tensor q    = q_flat.view({cfg_.head_dim, cfg_.n_q, 1});
        Tensor gate = gate_flat.view({cfg_.head_dim, cfg_.n_q, 1});
        Tensor qn   = q;
        if constexpr (attention_qk_norm<Variant>()) {
            qn = work_.alloc(DType::BF16, {cfg_.head_dim, cfg_.n_q, 1});
            ops::rmsnorm(q, *mtp_.q_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), qn, s);
        }
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
        if constexpr (applies_rotary<Variant>()) {
            ops::rope(last_rope_position, cfg_.rotary_dim, cfg_.rope_theta, qn, s);
        }

        Tensor a = work_.alloc(DType::BF16, {cfg_.head_dim, cfg_.n_q, 1});
        ops::gqa_attention_cached(qn, last_position, cfg_.attention_scale, mtp_kv_.layer_view(0), envelope,
                                  work_, a, s);
        if constexpr (kAttentionOutputGate) { apply_attention_gate<Variant>(gate, a, s); }

        Tensor o = work_.alloc(DType::BF16, {cfg_.hidden, 1});
        mtp_attention_output(a.view({cfg_.q_size, 1}), o);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {cfg_.hidden, 1});
        ops::rmsnorm(x_last, *mtp_.post_attn_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), mh,
                     s);
        {
            auto post_mixer_scope = work_.scope();
            Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x_last, work_, s);
        }
        ops::rmsnorm(x_last, *mtp_.norm, cfg_.rms_eps, norm_unit_offset<Variant>(), *final_hidden,
                     s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

/// The head's output, squashed, for a family that caps its logits. A no-op for every other
/// family: `logit_softcap` is zero unless the target declares one or the artifact states one.
///
/// Applied to every plane the head writes rather than once at the sampler, because the logits
/// reach the sampler by several routes -- the batched decode, the finalising gather, the
/// prefill tail -- and a cap missed on one of them is a model that answers differently
/// depending on which route the round took.
void apply_logit_softcap(const ModelConfig& cfg, Tensor& logits, cudaStream_t stream) {
    if (cfg.logit_softcap > 0.0F) { ops::logit_softcap(logits, cfg.logit_softcap, stream); }
}

template <class V>
void TextContext::proposal_argmax(const Tensor& wide, Tensor& logits, Tensor& proposal_tokens) {
    const int T = wide.ne[1];
    // A trunk-block draft head hands back the wide residual, because that is what the next
    // draft step folds into. Its own mixer collapses the streams to the LM head's width here,
    // standing in for the output norm the architecture does not have.
    auto collapse_scope = work_.scope();
    Tensor hidden       = wide;
    if constexpr (mtp_block_is_trunk_layer<V>()) {
        require_tensor_shape(wide, DType::BF16, {weights_.geometry.residual, T}, "proposal hidden");
        hidden = work_.alloc(DType::BF16, {cfg_.hidden, T});
        V::mtp_collapse(weights_, wide, hidden, work_, ctx_.stream);
    }
    require_tensor_shape(hidden, DType::BF16, {cfg_.hidden, T}, "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, cfg_.vocab, T, "proposal logits");
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        ops::linear(hidden, *proposal_head_, proposal_logits, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens, proposal_head_n_, ctx_.stream);
        ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                      ctx_.stream);
    } else {
        Tensor output_logits = matrix_window(logits, T);
        ops::linear(hidden, *lm_head_, output_logits, ctx_.stream);
        apply_logit_softcap(cfg_, output_logits, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens, cfg_.token_domain, ctx_.stream);
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
    require_tensor_shape(hidden, DType::BF16, {round_hidden_width(), T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {round_hidden_width(), T},
                         "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {cfg_.vocab, 1}, "MTP logits");
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
    require_tensor_shape(previous_hidden, DType::BF16, {round_hidden_width(), 1},
                         "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {round_hidden_width(), 1},
                         "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {cfg_.vocab, 1}, "MTP AR logits");
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

// A speculative round: each lane owns `width` consecutive columns, which are one segment.
inline PrologueColumns verify_columns(WorkspaceArena& work, const Tensor& ids, const Tensor& slots,
                                      std::int32_t width, std::int32_t batch,
                                      cudaStream_t stream) {
    PrologueColumns out;
    const std::int32_t columns = width * batch;
    out.ids                    = ids;
    out.slots                  = work.alloc(DType::I32, {columns});
    out.segment_begin          = work.alloc(DType::I32, {columns});
    out.segment_last           = work.alloc(DType::I32, {columns});
    ops::ngram_ple_expand_columns(slots, width, out.slots, out.segment_begin, out.segment_last,
                                  stream);
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

template <class V>
Tensor TextContext::finish_prefill(Tensor& wide, const Tensor& x, cudaStream_t stream) {
    if constexpr (mtp_block_is_trunk_layer<V>()) {
        if (wide.ne[0] == weights_.geometry.residual && weights_.geometry.residual != cfg_.hidden) {
            Tensor collapsed = work_.alloc(DType::BF16, {cfg_.hidden, wide.ne[1]});
            Hooks::finish(weights_, x, cfg_.rms_eps, collapsed, work_, stream);
            CUDA_CHECK(cudaMemcpyAsync(wide.data, x.data, wide.bytes(), cudaMemcpyDeviceToDevice,
                                       stream));
            return collapsed;
        }
    }
    Hooks::finish(weights_, x, cfg_.rms_eps, wide, work_, stream);
    return wide;
}

template <class V>
Tensor TextContext::lm_head_view(const Tensor& stored, cudaStream_t stream) {
    if constexpr (mtp_block_is_trunk_layer<V>()) {
        if (stored.ne[0] == weights_.geometry.residual &&
            weights_.geometry.residual != cfg_.hidden) {
            Tensor collapsed = work_.alloc(DType::BF16, {cfg_.hidden, stored.ne[1]});
            Hooks::finish(weights_, stored, cfg_.rms_eps, collapsed, work_, stream);
            return collapsed;
        }
    }
    return stored;
}

void TextContext::logits_from_hidden(const Tensor& hidden, Tensor& logits) {
    require_tensor_shape(hidden, DType::BF16, {round_hidden_width(), 1}, "cached hidden");
    require_tensor_shape(logits, DType::BF16, {cfg_.vocab, 1}, "cached hidden logits");
    auto scope = work_.scope();
    ops::linear(lm_head_view(hidden, ctx_.stream), *lm_head_, logits, ctx_.stream);
    apply_logit_softcap(cfg_, logits, ctx_.stream);
}

/// Writes the round's hidden output and the logits that follow it.
///
/// With a trunk-block draft head the buffer that crosses the round boundary is the wide
/// residual, because that is what the head folds into next; the LM head still reads the
/// collapsed width, from a scratch. Without one the two are the same tensor, as they always
/// were.
template <class Variant, class Hooks, class Model, class Cfg>
void finish_round_hidden(const Model& weights, const Tensor& x, float eps, Tensor& hidden,
                         Tensor& logits, const Weight& lm_head, const Cfg& cfg,
                         WorkspaceArena& work, cudaStream_t stream) {
    if constexpr (mtp_block_is_trunk_layer<Variant>()) {
        if (hidden.ne[0] == weights.geometry.residual && weights.geometry.residual != cfg.hidden) {
            auto scope       = work.scope();
            Tensor collapsed = work.alloc(DType::BF16, {cfg.hidden, hidden.ne[1]});
            Hooks::finish(weights, x, eps, collapsed, work, stream);
            ops::linear(collapsed, lm_head, logits, stream);
            apply_logit_softcap(cfg, logits, stream);
            CUDA_CHECK(cudaMemcpyAsync(hidden.data, x.data, hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
            return;
        }
    }
    Hooks::finish(weights, x, eps, hidden, work, stream);
    ops::linear(hidden, lm_head, logits, stream);
    apply_logit_softcap(cfg, logits, stream);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_slots,
                                        ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                        Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumBatchColumns)) {
        throw std::invalid_argument("ordinary decode batch size is outside the GPU batch capacity");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {round_hidden_width(), batch},
                         "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {cfg_.vocab, batch}, "ordinary decode logits");

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

        Tensor x = work_.alloc(weights_.geometry.residual_dtype(), {cfg_.residual, batch});
        if constexpr (Hooks::prologue) {
            prologue_ =
                prologue_staging::decode_columns(work_, ids, linear_state_slots, batch, stream);
        }
        active_ids_ = ids;
        if (stage_embeds()) { Hooks::embed(weights_, ids, x, work_, stream); } else { stage_import(x, stream); }
        capture_per_layer_source(x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        if (stage_finishes()) {
            finish_round_hidden<Variant, Hooks>(weights_, x, cfg_.rms_eps, hidden, logits,
                                                *lm_head_, cfg_, work_, stream);
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
        batch > static_cast<std::int32_t>(kMaximumBatchColumns)) {
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
    require_tensor_shape(hidden, DType::BF16, {round_hidden_width(), width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {cfg_.vocab, width, batch},
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

        Tensor x        = work_.alloc(weights_.geometry.residual_dtype(), {cfg_.residual, columns});
        Tensor flat_ids = ids.view({columns});
        if constexpr (Hooks::prologue) {
            // A speculative verify gives each lane several consecutive columns, and they are
            // one segment of that lane's history; an ordinary round's columns are one each.
            prologue_ = width != 1
                            ? prologue_staging::verify_columns(work_, flat_ids, linear_state_slots,
                                                               width, batch, stream)
                            : prologue_staging::decode_columns(work_, flat_ids, linear_state_slots,
                                                               columns, stream);
        }
        active_ids_ = flat_ids;
        if (stage_embeds()) { Hooks::embed(weights_, flat_ids, x, work_, stream); } else { stage_import(x, stream); }
        capture_per_layer_source(x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({round_hidden_width(), columns});
        Tensor flat_logits = logits.view({cfg_.vocab, columns});
        Tensor flat_tokens = target_tokens.view({columns});
        if (stage_finishes()) {
            finish_round_hidden<Variant, Hooks>(weights_, x, cfg_.rms_eps, flat_hidden, flat_logits,
                                                *lm_head_, cfg_, work_, stream);
            ops::argmax(flat_logits, flat_tokens, cfg_.token_domain, stream);
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
        batch > static_cast<std::int32_t>(kMaximumBatchColumns)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {round_hidden_width(), width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {round_hidden_width(), width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {round_hidden_width(), batch},
                         "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {cfg_.vocab, batch}, "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int fidx, int layer, Phase ph,
                           KvPlane plane) {
    // The caller counts attending layers; a model that shares planes needs the *plane* index,
    // which for a sharing layer is the one its source owns.
    //
    // Whether this layer owns planes at all decides three things below: that it projects a key
    // and a value, that it has a key norm to apply, and that its attention appends. A sharing
    // layer does none of the three -- it holds a query and an output, and attends over what an
    // earlier layer wrote.
    const bool owns_kv = plane == KvPlane::Mtp || cfg_.layer_owns_kv(layer);
    if (plane != KvPlane::Mtp) { fidx = cfg_.kv_plane_index(layer); }
    const bool mtp = plane == KvPlane::Mtp;
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_gqa_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace_recipe::text_attention_projection(
        work_, cfg_.hidden, cfg_.layer_q_size(layer), cfg_.layer_kv_size(layer), T);
    Tensor h              = projection.hidden;
    debug_probe<Variant>("residual_in", x, cfg_.n_layers, s);
    Hooks::attention_norm(x, *w.input_norm, cfg_.rms_eps, *w.projection, h, work_, s);
    debug_probe<Variant>("post_input_norm", h, cfg_.n_layers, s);

    const int layer_head_dim = cfg_.layer_head_dim(layer);
    const int layer_n_kv     = cfg_.layer_n_kv(layer);
    const int layer_q_size   = cfg_.layer_q_size(layer);
    const int layer_kv_size  = cfg_.layer_kv_size(layer);
    Tensor q         = projection.query.view({layer_head_dim, cfg_.n_q, T});
    Tensor gate      = projection.gate.view({layer_head_dim, cfg_.n_q, T});
    Tensor k         = projection.key.view({layer_head_dim, layer_n_kv, T});
    Tensor v         = projection.value.view({layer_head_dim, layer_n_kv, T});
    Tensor q_flat    = q.view({layer_q_size, T});
    Tensor gate_flat = gate.view({layer_q_size, T});
    Tensor k_flat    = k.view({layer_kv_size, T});
    Tensor v_flat    = v.view({layer_kv_size, T});
    Variant::attention_projection(h, *w.projection, q_flat, gate_flat, k_flat, v_flat, ph, work_,
                                  s);
    debug_probe<Variant>("q_proj_raw", q_flat, cfg_.n_layers, s);
    debug_probe<Variant>("k_proj_raw", k_flat, cfg_.n_layers, s);
    debug_probe<Variant>("v_proj_raw", v_flat, cfg_.n_layers, s);

    const auto results = workspace_recipe::text_attention_results(
        work_, cfg_.layer_q_size(layer), cfg_.layer_kv_size(layer), T);
    // A target without a per-head query/key norm has nothing to write into the
    // normalised planes, so rope runs in place on the projection's own output.
    Tensor qn = attention_qk_norm<Variant>()
                    ? results.normalized_query.view({layer_head_dim, cfg_.n_q, T})
                    : q;
    Tensor kn = attention_qk_norm<Variant>()
                    ? results.normalized_key.view({layer_head_dim, layer_n_kv, T})
                    : k;
    if constexpr (attention_qk_norm<Variant>()) {
        ops::rmsnorm(q, *w.q_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), qn, s);
        // A layer that owns no key/value planes wrote no key to normalise, and carries no key
        // norm to normalise it with: it attends over what an earlier layer already normalised
        // and cached.
        if (owns_kv) {
            ops::rmsnorm(k, *w.k_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), kn, s);
        }
    }
    debug_probe<Variant>("q_post_headnorm", qn.view({layer_q_size, T}), cfg_.n_layers, s);
    debug_probe<Variant>("k_post_headnorm", kn.view({layer_kv_size, T}), cfg_.n_layers, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = text_rope_positions<Variant>(
        active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions);
    if constexpr (applies_rotary<Variant>()) {
        const auto& g = weights_.geometry;
        if (rope_for_op.ne[1] == 3 && g.mrope_temporal) {
            ops::rope_interleaved(rope_for_op, cfg_.layer_rotary_dim(layer), layer_rope_theta(layer, g),
                                 {g.mrope_temporal, g.mrope_height, g.mrope_width}, qn, kn, s);
        } else {
        ops::rope(rope_for_op, cfg_.layer_rotary_dim(layer), cfg_.layer_rotary_pairs(layer),
                  layer_rope_theta(layer, weights_.geometry), qn, kn, s);
        }
    }
    debug_probe<Variant>("q_post_rope", qn.view({layer_q_size, T}), cfg_.n_layers, s);
    debug_probe<Variant>("k_post_rope", kn.view({layer_kv_size, T}), cfg_.n_layers, s);

    Tensor a = results.attention.view({layer_head_dim, cfg_.n_q, T});
    const Tensor& kv_table_rows =
        mtp ? (active_backend_kv_table_rows_ != nullptr ? *active_backend_kv_table_rows_
                                                        : io_.backend_kv_table_row)
            : (active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row);
    // The planes this layer attends over. Its own where it has any; an earlier layer's where
    // it shares -- `fidx` is already that layer's index, because `kv_plane_index` resolved it.
    const auto kv_view =
        mtp ? batch_mtp_kv_->batch_layer_view(0) : batch_text_kv_->batch_layer_view(fidx);
    // Appending here would corrupt the source layer's history with this one's, which is not an
    // error anywhere -- it is a model that answers slightly wrongly from the layer after the
    // first shared one onwards.
    // QSA indexer: cache this chunk's indexer keys and, past the budget, select the blocks each
    // column may attend to. Empty below the budget, so the dense path is untouched. The draft
    // head skips it and attends densely -- see KvPlane.
    const std::int32_t indexer_columns_per_row =
        active_sequence_batch_ != 0 ? active_sequence_width_ : T;
    const ops::GqaBlockMask selection =
        mtp ? ops::GqaBlockMask{}
            : text_indexer_selection(
                  w, h, T, cache_positions, rope_positions, kv_table_rows, indexer_columns_per_row,
                  static_cast<std::int32_t>(active_gqa_envelope_->max_visible_keys),
                  batch_text_kv_->batch_layer_view(fidx));
    // The round binding says how many keys the launch must be sized for; the window
    // says how many of them this layer may look at. Stamp the layer's window onto a
    // copy -- widening the round binding could not express an alternating stack.
    ops::GqaExecutionEnvelope layer_envelope = *active_gqa_envelope_;
    // The draft head is not a stack layer, so it has no layer index to key a window on; no
    // target with a draft head declares one, and one that grows both must decide it here.
    layer_envelope.sliding_window =
        mtp ? 0 : layer_sliding_window(layer, weights_.geometry);
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({layer_head_dim, cfg_.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({layer_head_dim, layer_n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({layer_head_dim, layer_n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({layer_head_dim, cfg_.n_q, width, active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (owns_kv) {
            ops::gqa_attention(q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
                               cfg_.attention_scale, kv_view, layer_envelope, work_, a_batch, s, selection);
        } else {
            ops::gqa_attention_cached(q_batch, position_batch, valid, kv_table_rows, cfg_.attention_scale,
                                      kv_view, layer_envelope, work_, a_batch, s, selection);
        }
    } else if (owns_kv) {
        ops::gqa_attention(qn, kn, v, cache_positions, Tensor{}, kv_table_rows, cfg_.attention_scale,
                           kv_view, layer_envelope, work_, a, s, selection);
    } else {
        ops::gqa_attention_cached(qn, cache_positions, Tensor{}, kv_table_rows, cfg_.attention_scale,
                                  kv_view, layer_envelope, work_, a, s, selection);
    }
    // A dense stack writes no gate rows; see attention_output_gate<Variant>().
    if constexpr (kAttentionOutputGate) { apply_attention_gate<Variant>(gate, a, s); }

    debug_probe<Variant>("attn_core", a.view({layer_q_size, T}), cfg_.n_layers, s);
    Hooks::attention_output(a.view({layer_q_size, T}), *w.o_proj, *w.projection, x, ph, work_, s);
    debug_probe<Variant>("post_attention_residual", x, cfg_.n_layers, s);
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

/// The short-convolution mixer.
///
/// Four steps and no state beyond the K-1 columns behind the round: normalise the residual,
/// project it to B, C and x at once, convolve the gated input B*x under C, and add the output
/// projection back to the residual. There is no recurrence, so nothing here is sequential in t
/// and a chunk of any width is one pass -- which is why a prefill needs no separate route from
/// a decode, only a different way of reaching the state.
///
/// The state is where the two phases differ. A prefill chunk is one sequence continuing one
/// history, so it reads and rewrites a single slot. A decode round is B lanes that share no
/// history at all, so every row says which slot it starts from and where its new windows go,
/// and the op resolves that indirection itself.
void TextContext::short_conv_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s   = ctx_.stream;
    const int T      = x.ne[1];
    const auto roots = workspace_recipe::short_conv(work_, cfg_geometry(), T);
    Tensor bcx       = roots.projected;
    Tensor convolved = roots.convolved;

    Variant::short_conv_projection(x, *w.input_norm, cfg_.rms_eps, *w.projection, bcx, ph, work_,
                                   s);
    debug_probe<Variant>("short_conv_in", bcx, cfg_.n_layers, s);

    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_slots_ == nullptr) {
            throw std::logic_error(
                "Verify short_conv requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error(
                "short_conv sequence batch binding does not match aggregate columns");
        }
        Tensor rows_in  = bcx.view({3 * cfg_.hidden, width, active_sequence_batch_});
        Tensor rows_out = convolved.view({cfg_.hidden, width, active_sequence_batch_});
        Tensor& conv_states = state_.conv.at(static_cast<std::size_t>(gidx));
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::short_conv_snapshot(rows_in, *w.conv1d, conv_states, *active_linear_state_slots_,
                                 *active_linear_state_slots_, valid, rows_out, cfg_.hidden, s);
    } else {
        Tensor conv_state =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
        debug_probe<Variant>("short_conv_state_in", conv_state, cfg_.n_layers, s);
        if (graph_pad_valid_ != nullptr) {
            // Bucket-padded graph body: the window this round leaves behind must end at the
            // real token count, not at the captured width.
            ops::short_conv(bcx, *w.conv1d, conv_state, convolved, cfg_.hidden, *graph_pad_valid_,
                            s);
        } else {
            ops::short_conv(bcx, *w.conv1d, conv_state, convolved, cfg_.hidden, s);
        }
    }
    debug_probe<Variant>("short_conv_out", convolved, cfg_.n_layers, s);

    // The same linear-add the delta net ends with, and the same leaf: an output projection is an
    // output projection whichever mixer produced the value it reads.
    Variant::gdn_output_projection(convolved, *w.out_proj, x, ph, work_, s);
}

/// The short-convolution mixer over a mixed round: some columns continue prefill sequences,
/// the rest are one column each for decode lanes.
///
/// The split is the same one the delta net makes, and for the same reason -- a prefill segment
/// continues one history in one slot, while the lanes behind it each continue their own -- but
/// it is the whole of the difference here, because a convolution has no scan to run afterwards.
/// `segment_slots` gives one (offset, columns, slot) per prefill segment; `valid` is the padded
/// column count they share, or empty.
void TextContext::short_conv_mix_mixed(const GdnLayerW& w, Tensor& x, int gidx,
                                       std::span<const ShortConvSegment> segments,
                                       std::int32_t prefill_columns, std::int32_t batch,
                                       const Tensor& valid, const Tensor& decode_slots) {
    cudaStream_t s   = ctx_.stream;
    const int total  = x.ne[1];
    const auto roots = workspace_recipe::short_conv(work_, cfg_geometry(), total);
    Tensor bcx       = roots.projected;
    Tensor convolved = roots.convolved;

    Variant::short_conv_projection(x, *w.input_norm, cfg_.rms_eps, *w.projection, bcx,
                                   Phase::Prefill, work_, s);

    for (const ShortConvSegment& segment : segments) {
        Tensor part_in    = bcx.slice(1, segment.offset, segment.columns);
        Tensor part_out   = convolved.slice(1, segment.offset, segment.columns);
        Tensor conv_state = state_.conv_slot(static_cast<std::uint32_t>(gidx), segment.state_slot);
        ops::short_conv(part_in, *w.conv1d, conv_state, part_out, cfg_.hidden, valid, s);
    }
    if (batch > 0) {
        Tensor rows_in  = bcx.slice(1, prefill_columns, batch)
                             .view({3 * cfg_.hidden, 1, batch});
        Tensor rows_out = convolved.slice(1, prefill_columns, batch)
                              .view({cfg_.hidden, 1, batch});
        ops::short_conv_snapshot(rows_in, *w.conv1d, state_.conv.at(static_cast<std::size_t>(gidx)),
                                 decode_slots, decode_slots, Tensor{}, rows_out, cfg_.hidden, s);
    }

    Variant::gdn_output_projection(convolved, *w.out_proj, x, Phase::Prefill, work_, s);
}

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

    const auto control = workspace_recipe::gdn_control(work_, cfg_geometry(), T, kLinearMixer);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    Variant::gdn_norm_control_projection(x, *w.input_norm, cfg_.rms_eps, *w.projection, h, g, beta,
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
    const auto projection = workspace_recipe::gdn_projection(work_, cfg_geometry(), T);
    Tensor z              = projection.output_gate.view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, T});
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
        Tensor projection_input = h.view({cfg_.hidden, width, active_sequence_batch_});
        Tensor query_output     = qc.view({cfg_.key_dim, width, active_sequence_batch_});
        Tensor key_output       = kc.view({cfg_.key_dim, width, active_sequence_batch_});
        Tensor value_output     = vc.view({cfg_.value_dim, width, active_sequence_batch_});
        Tensor gate_output      = z.view({cfg_.value_dim, width, active_sequence_batch_});
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
        const auto conv = workspace_recipe::gdn_prefill_conv(work_, cfg_geometry(), T);
        Tensor qkv      = conv.projected;
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s);
        if (sub_timing) { sub_lap(ftimer.g_proj, sub_proj); }
        Tensor qkv_c = conv.convolved;
        Tensor conv_state =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
        debug_probe<Variant>("gdn_conv_state_in", conv_state, cfg_.n_layers, s);
        if (graph_pad_valid_ != nullptr) {
            ops::causal_conv1d_silu(qkv, *w.conv1d, conv_state, conv_state, qkv_c,
                                    *graph_pad_valid_, s);
        } else {
            ops::causal_conv1d_silu(qkv, *w.conv1d, conv_state, conv_state, qkv_c, s);
        }
        debug_probe<Variant>("gdn_conv", qkv_c, cfg_.n_layers, s);
        if (sub_timing) { sub_lap(ftimer.g_conv, sub_conv); }
        ops::extract_bf16_columns(qkv_c, 0, qc, s);
        ops::extract_bf16_columns(qkv_c, cfg_.key_dim, kc, s);
        ops::extract_bf16_columns(qkv_c, 2 * cfg_.key_dim, vc, s);
        if (sub_timing) { sub_lap(ftimer.g_extract, sub_extract); }
    }

    Tensor q_recurrent = qc.view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, T});

    Tensor vv = vc.view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, T});
    Tensor o  = workspace_recipe::gdn_recurrent_output(work_, cfg_geometry(), T).view(
        {cfg_.gdn_v_dim, cfg_.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor& recurrent_states = state_.recurrent.at(static_cast<std::size_t>(gidx));
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch =
            q_recurrent.view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, width, active_sequence_batch_});
        Tensor k_batch =
            k_recurrent.view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, width, active_sequence_batch_});
        Tensor v_batch = vv.view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, width, active_sequence_batch_});
        Tensor g_batch    = family::detail::linear_gate_view<Variant>(
            g, cfg_.gdn_v_dim, cfg_.gdn_v_heads, width, active_sequence_batch_);
        Tensor beta_batch = beta.view({cfg_.gdn_v_heads, width, active_sequence_batch_});
        Tensor out_batch =
            o.view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            // Replay-record exists so a speculative round can be re-folded from the tokens it
            // accepted; the mixer says which recurrence records, and a mixer without the form
            // refuses there.
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            family::detail::linear_recurrence_record<Variant>(
                q_batch, k_batch, v_batch, g_batch, beta_batch, cfg_.gdn_scale, recurrent_states, valid,
                *active_linear_state_slots_, records, out_batch, s);
        } else {
            family::detail::linear_recurrence_snapshot<Variant>(
                q_batch, k_batch, v_batch, g_batch, beta_batch, cfg_.gdn_scale, recurrent_states, valid,
                *active_linear_state_slots_, *active_linear_state_slots_, out_batch, s);
        }
    } else {
        Tensor recurrent_state =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_current_slot_);
        debug_probe<Variant>("gdn_recurrent_state_in", recurrent_state, cfg_.n_layers, s);
        family::detail::linear_recurrence<Variant>(
            q_recurrent, k_recurrent, vv,
            family::detail::linear_gate_view<Variant>(g, cfg_.gdn_v_dim, cfg_.gdn_v_heads, T), beta,
            cfg_.gdn_scale, work_, recurrent_state, o, s);
        debug_probe<Variant>("gdn_o", o, cfg_.n_layers, s);
    }

    Tensor on = workspace_recipe::gdn_normalized_output(work_, cfg_geometry(), T).view(
        {cfg_.gdn_v_dim, cfg_.gdn_v_heads, T});
    if (sub_timing) { sub_lap(ftimer.g_scan, sub_scan); }
    ops::gated_rmsnorm(o, *w.gdn_norm, z, cfg_.rms_eps, gdn_output_gate<Variant>(), on, s);
    if (sub_timing) { sub_lap(ftimer.g_norm, sub_norm); }

    Variant::gdn_output_projection(on.view({cfg_.value_dim, T}), *w.out_proj, x, ph, work_, s);
    if (sub_timing) {
        sub_lap(ftimer.g_out, sub_out);
        ftimer.t_g_ctrl += sub_ctrl;       ftimer.t_g_proj += sub_proj;   ftimer.t_g_conv += sub_conv;
        ftimer.t_g_extract += sub_extract; ftimer.t_g_scan += sub_scan;   ftimer.t_g_norm += sub_norm;
        ftimer.t_g_out += sub_out;
    }
}

/// `SUROGATE_SERVE_NLL_DUMP=<path>`: after every eager prefill chunk, append one line per prompt
/// position holding the token there, the token that follows it, the negative log-likelihood
/// the model assigned to that following token -- what a perplexity measurement sums -- and the
/// token the model ranked first. It runs
/// the head over every prompt column in slices, so it costs a full lm_head pass and a stream
/// synchronisation per chunk. That is a head pass of its own, so it caps its logits itself:
/// the cap is monotonic, so leaving it out keeps every ranking -- and with it the top-1 column
/// and the greedy text -- exactly right while the *distribution* stays as peaked as the raw
/// head left it. On Gemma 4 that read 93 % on the token it ranked first and 1e-7 on the truth
/// when it was wrong, for a wikitext perplexity 146x llama.cpp's on the same file; it is for measurement, never for serving, and cannot run inside
/// a captured graph (set `SUROGATE_SERVE_PREFILL_GRAPH=0` and `SUROGATE_SERVE_NO_MIXED_GRAPH=1`
/// to keep prefill eager while measuring).
template <class Arena>
void debug_next_token_nll(const Tensor& hidden_all, const Tensor& ids, std::int32_t columns,
                          const Weight& head, std::int32_t vocab, std::int32_t token_domain,
                          float logit_softcap, Arena& work, cudaStream_t stream) {
    static const char* path = std::getenv("SUROGATE_SERVE_NLL_DUMP");
    if (path == nullptr || columns < 2) { return; }
    // The token domain is the family's (the rows `ops::sample` scores); a smaller model of the
    // family has fewer rows than that, and the probe scores what the head actually has.
    token_domain = std::min(token_domain, vocab);
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capturing));
    if (capturing != cudaStreamCaptureStatusNone) { return; } // a graph body: nothing to read
    // `SUROGATE_SERVE_NLL_SLICE` narrows the head's batch (8 keeps it on the GEMV route).
    static const std::int32_t kSlice = [] {
        const char* env = std::getenv("SUROGATE_SERVE_NLL_SLICE");
        const int value = env != nullptr ? std::atoi(env) : 256;
        return value > 0 ? value : 256;
    }();
    auto scope    = work.scope();
    // The slice is bounded by what the request's arena has left: a 152k vocabulary at 256
    // columns is 78 MB of logits, more than a small dense target's whole arena, and the probe
    // is a diagnostic that must fit whatever the plan budgeted, not size the plan.
    const std::size_t spare = work.capacity() > work.used() ? work.capacity() - work.used() : 0;
    const std::size_t fixed = static_cast<std::size_t>(columns) * (sizeof(float) + sizeof(std::int32_t)) + 4096;
    const std::size_t fits  = spare > fixed ? (spare - fixed) / (static_cast<std::size_t>(vocab) * sizeof(std::uint16_t)) : 0;
    const std::int32_t slice = std::max<std::int32_t>(1, std::min<std::int32_t>(kSlice, static_cast<std::int32_t>(fits) & ~7));
    Tensor logits = work.alloc(DType::BF16, {vocab, slice});
    Tensor nll    = work.alloc(DType::FP32, {columns});
    Tensor best   = work.alloc(DType::I32, {columns});
    for (std::int32_t c0 = 0; c0 < columns - 1; c0 += slice) {
        const std::int32_t n = std::min(slice, columns - 1 - c0);
        Tensor hidden        = hidden_all.slice(1, c0, n);
        Tensor slice         = logits.slice(1, 0, n);
        ops::linear(hidden, head, slice, stream);
        if (logit_softcap > 0.0F) { ops::logit_softcap(slice, logit_softcap, stream); }
        Tensor targets = ids.slice(0, c0 + 1, n);
        Tensor out     = nll.slice(0, c0, n);
        Tensor top     = best.slice(0, c0, n);
        ops::next_token_nll(slice, targets, out, &top, token_domain, stream);
    }
    {
        // The last column has no successor to score; its ranking is still worth a line, since
        // it is the very column the engine samples from, so the two must agree.
        Tensor hidden  = hidden_all.slice(1, columns - 1, 1);
        Tensor slice   = logits.slice(1, 0, 1);
        ops::linear(hidden, head, slice, stream);
        if (logit_softcap > 0.0F) { ops::logit_softcap(slice, logit_softcap, stream); }
        Tensor targets = ids.slice(0, columns - 1, 1);
        Tensor out     = nll.slice(0, columns - 1, 1);
        Tensor top     = best.slice(0, columns - 1, 1);
        ops::next_token_nll(slice, targets, out, &top, token_domain, stream);
    }
    std::vector<std::int32_t> host_ids(static_cast<std::size_t>(columns));
    std::vector<float> host_nll(static_cast<std::size_t>(columns));
    std::vector<std::int32_t> host_best(static_cast<std::size_t>(columns));
    CUDA_CHECK(cudaMemcpyAsync(host_ids.data(), ids.data, host_ids.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_best.data(), best.data, host_best.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_nll.data(), nll.data, host_nll.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::FILE* file = std::fopen(path, "a");
    if (file == nullptr) { return; }
    std::fprintf(file, "# chunk %d\n", static_cast<int>(columns));
    for (std::int32_t p = 0; p + 1 < columns; ++p) {
        std::fprintf(file, "%d %d %.6f %d\n", host_ids[static_cast<std::size_t>(p)],
                     host_ids[static_cast<std::size_t>(p) + 1],
                     host_nll[static_cast<std::size_t>(p)],
                     host_best[static_cast<std::size_t>(p)]);
    }
    std::fprintf(file, "%d -1 nan %d\n", host_ids[static_cast<std::size_t>(columns) - 1],
                 host_best[static_cast<std::size_t>(columns) - 1]);
    std::fclose(file);
}

void TextContext::capture_per_layer_source(const Tensor& x, cudaStream_t stream) {
    if constexpr (Hooks::per_layer_inputs) {
        // The layers write the residual in place, so the embedded input has to be kept apart.
        // Allocated from the round's arena, beside `x` itself, so it lives as long as the
        // layer loop that reads it.
        active_embedded_ = work_.alloc(DType::BF16, {x.ne[0], x.ne[1]});
        if (stage_embeds()) {
            CUDA_CHECK(cudaMemcpyAsync(active_embedded_.data, x.data, x.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
        } else {
            // Per-layer inputs use the original token embedding, not the residual received
            // from the preceding stage. Every stage keeps the embedding table for this.
            Hooks::embed(weights_, active_ids_, active_embedded_, work_, stream);
        }
    } else {
        (void)x;
        (void)stream;
    }
}

void TextContext::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, int layer,
                           Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    Tensor h       = workspace_recipe::post_mixer_hidden(work_, cfg_geometry(), T);
    Hooks::post_mixer_norm(x, *post_norm, cfg_.rms_eps, *m.payload, h, work_, s);
    debug_probe<Variant>("post_attention_norm", h, cfg_.n_layers, s);

    Variant::post_mixer(h, *m.payload, x, ph, work_, s);
    // What a block does after its feed-forward, for a family that does anything: Gemma 4's
    // E-series folds in this layer's per-layer input here, between the feed-forward's residual
    // add and the layer scalar. A no-op for every other target.
    Hooks::layer_epilogue(weights_, layer, active_ids_, active_embedded_, x, work_, s);
    debug_probe<Variant>("post_mlp_residual", x, cfg_.n_layers, s);
}

// Per-family prefill timing behind SUROGATE_SERVE_PREFILL_TIMING: events
// around each mixer and MLP, summarised every 32 prefill chunks. It is the
// instrument for "where does a prefill chunk's time go" when a profiler is
// not available; the synchronize it adds at the end of run_layers only
// exists while the switch is set.

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap, const Tensor* deepstack,
                              std::span<const std::int32_t> visual_indices) {
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
    // Pipeline work asks the same question of every failure: which layers did this round
    // actually run. `SUROGATE_SERVE_TRACE_STAGE=1` answers it once per round.
    if (stage_trace_enabled()) {
        std::fprintf(stderr, "stage-trace: layers [%d, %d) of %d, columns %d\n", stage_first_,
                     stage_last_, cfg_.n_layers, x.ne[1]);
    }
    for (int layer = stage_first_; layer < stage_last_; ++layer) {
        Hooks::layer_prologue(weights_, layer, x, prologue_, ple_state_, work_, ctx_.stream);
        if (cfg_.is_full(layer)) {
            const int fidx         = cfg_.full_idx(layer);
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
                attn_mix(full, x, fidx, layer, ph);
                if (timing) { lap(timer.begin, timer.attn, acc_attn); }
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, ctx_.stream); }
                mlp_tail(full.post_attn_norm, full.mlp, x, layer, ph);
                if (timing) { lap(timer.begin, timer.mlp_full, acc_mlp_full); }
            }
        } else {
            const int gidx       = cfg_.gdn_idx(layer);
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
                if constexpr (kLinearMixer == family::LinearMixer::ShortConv) {
                    short_conv_mix(gdn, x, gidx, ph);
                } else {
                    gdn_mix(gdn, x, gidx, ph);
                }
                if (timing) { lap(timer.begin, timer.gdn, acc_gdn); }
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, ctx_.stream); }
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, layer, ph);
                if (timing) { lap(timer.begin, timer.mlp_gdn, acc_mlp_gdn); }
            }
        }
        if (deepstack != nullptr && layer < deepstack->ne[2]) {
            family::detail::add_visual_embeddings(x, deepstack->slice(2, layer, 1),
                                                  visual_indices, ctx_.stream);
        }
        if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
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
    if constexpr (!requires { V::has_qsa_indexer; }) {
        return ops::GqaBlockMask{};
    } else {
        const ops::QsaIndexerGeometry geometry{
            .head_dim   = cfg_geometry().indexer_head_dim,
            .heads      = cfg_geometry().indexer_heads,
            .block      = cfg_geometry().indexer_block,
            .top_k      = cfg_geometry().indexer_top_k,
            .rotary_dim = cfg_.rotary_dim,
            .rope_theta = cfg_.rope_theta,
            .rms_eps    = cfg_.rms_eps,
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
        // A deployment whose whole KV cache is shorter than the budget can never reach a
        // selection, so the keys would never be read: skip the append too. This is what keeps
        // the indexer off the cost of every short-context run (the board's shapes included).
        if (batch_text_kv_ != nullptr &&
            ops::qsa_selection_is_dense(static_cast<std::int32_t>(batch_text_kv_->max_context()),
                                        geometry)) {
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
    const int last = stage.last < 0 ? cfg_.n_layers : stage.last;
    if (stage.first < 0 || last > cfg_.n_layers || stage.first >= last) {
        throw std::invalid_argument("pipeline stage layer range is invalid");
    }
    if ((stage.first > 0 && stage.import_pinned == nullptr) ||
        (last < cfg_.n_layers && stage.export_pinned == nullptr)) {
        throw std::invalid_argument("pipeline stage boundary buffer is missing");
    }
    stage_first_ = stage.first;
    stage_last_  = last;
    stage_       = stage;
}

namespace {
/// SUROGATE_SERVE_STAGE_CHECKSUM=1: the residual as it crosses a boundary, one line per copy,
/// so two runs can be compared stage by stage. Synchronises the stream; diagnostics only.
inline bool stage_checksum_enabled() {
    static const bool enabled = std::getenv("SUROGATE_SERVE_STAGE_CHECKSUM") != nullptr;
    return enabled;
}
inline void stage_checksum(const char* what, int first, int last, const void* device_data,
                           std::int32_t rows, std::int32_t columns, cudaStream_t stream) {
    if (!stage_checksum_enabled()) { return; }
    // Graph capture forbids a synchronise; the captured bodies are the decode rounds, and the
    // prefill's boundary crossings are what this is for.
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capturing) != cudaSuccess ||
        capturing != cudaStreamCaptureStatusNone) {
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
    std::vector<std::uint16_t> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), device_data, count * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    double sum = 0.0, absmax = 0.0, last_column_sum = 0.0;
    std::size_t nans = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(host[i]) << 16;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        if (value != value) { ++nans; continue; }
        sum += value;
        absmax = std::max(absmax, static_cast<double>(std::fabs(value)));
        if (i >= count - static_cast<std::size_t>(rows)) { last_column_sum += value; }
    }
    std::fprintf(stderr,
                 "stage-checksum: %s stage [%d, %d) columns %d sum %.6g absmax %.6g last-column "
                 "%.6g nans %zu\n",
                 what, first, last, columns, sum, absmax, last_column_sum, nans);
}
} // namespace

void TextContext::stage_import(Tensor& x, cudaStream_t stream) {
    const std::int32_t columns = x.ne[1];
    if (columns > stage_.columns) { throw std::logic_error("pipeline stage import wider than its buffer"); }
    const auto width = static_cast<std::size_t>(x.ne[0]) * dtype_size(x.dtype);
    CUDA_CHECK(cudaMemcpy2DAsync(x.data, x.nb[1], stage_.import_pinned,
        stage_.column_bytes ? stage_.column_bytes : width, width, columns,
        cudaMemcpyHostToDevice, stream));
    stage_checksum("import", stage_first_, stage_last_, x.data, cfg_.residual, columns, stream);
}

void TextContext::stage_export(const Tensor& x, cudaStream_t stream) {
    const std::int32_t columns = x.ne[1];
    if (columns > stage_.columns) { throw std::logic_error("pipeline stage export wider than its buffer"); }
    stage_checksum("export", stage_first_, stage_last_, x.data, cfg_.residual, columns, stream);
    const auto width = static_cast<std::size_t>(x.ne[0]) * dtype_size(x.dtype);
    CUDA_CHECK(cudaMemcpy2DAsync(stage_.export_pinned,
        stage_.column_bytes ? stage_.column_bytes : width, x.data, x.nb[1], width, columns,
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
    if (batch < 0 || batch > static_cast<std::int32_t>(kMaximumBatchColumns)) {
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

    // SUROGATE_SERVE_PREFILL_TIMING=1: the whole body's stream time, per call and stage. A
    // diagnostic that waits for the body's end event, so it serialises a pipeline the way the
    // synchronize this body used to end with did; never read a throughput with it on.
    static const bool body_timing = std::getenv("SUROGATE_SERVE_PREFILL_TIMING") != nullptr;
    cudaEvent_t body_begin{}, body_end{};
    const auto body_host_begin = std::chrono::steady_clock::now();
    if (body_timing) {
        CUDA_CHECK(cudaEventCreateWithFlags(&body_begin, cudaEventDefault));
        CUDA_CHECK(cudaEventCreateWithFlags(&body_end, cudaEventDefault));
        CUDA_CHECK(cudaEventRecord(body_begin, s));
    }

    work_.reset();
    const auto roots = workspace_recipe::text_prefill_roots(work_, cfg_geometry(), total, 0, 0);
    Tensor ids_device = roots.ids;
    Tensor ids_decode = ids_device.slice(0, prefill_cols, batch);
    CUDA_CHECK(cudaMemcpyAsync(ids_decode.data, decode.ids.data,
                               static_cast<std::size_t>(batch) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, s));

    Tensor positions = roots.positions;
    // Each segment owns a column range; ids and positions are laid out segment by segment and
    // every mixer below slices the same ranges.
    std::array<int, kMaximumBatchColumns> segment_begin{};
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
    active_ids_ = ids_device;
    if (stage_embeds()) { Hooks::embed(weights_, ids_device, x, work_, s); } else { stage_import(x, s); }
    capture_per_layer_source(x, s);
    debug_probe<Variant>("mixed_embed_out", x, cfg_.n_layers, s);

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
    // Pipeline work asks the same question of every failure: which layers did this round
    // actually run. `SUROGATE_SERVE_TRACE_STAGE=1` answers it once per round.
    if (stage_trace_enabled()) {
        std::fprintf(stderr, "stage-trace: layers [%d, %d) of %d, columns %d\n", stage_first_,
                     stage_last_, cfg_.n_layers, x.ne[1]);
    }
    for (int layer = stage_first_; layer < stage_last_; ++layer) {
        Hooks::layer_prologue(weights_, layer, x, prologue_, ple_state_, work_, ctx_.stream);
        if (cfg_.is_full(layer)) {
            const int fidx         = cfg_.full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            const bool owns_kv = cfg_.layer_owns_kv(layer);
            const auto kv_view = batch_text_kv_->batch_layer_view(cfg_.kv_plane_index(layer));
            {
                auto mixer_scope      = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, s); }
                const auto projection = workspace_recipe::text_attention_projection(
                    work_, cfg_.hidden, cfg_.layer_q_size(layer), cfg_.layer_kv_size(layer),
                    total);
                Tensor h = projection.hidden;
                debug_probe<Variant>("residual_in", x, cfg_.n_layers, s);
                Hooks::attention_norm(x, *full.input_norm, cfg_.rms_eps, *full.projection, h,
                                      work_, s);
                debug_probe<Variant>("post_input_norm", h, cfg_.n_layers, s);
                const int layer_head_dim = cfg_.layer_head_dim(layer);
                const int layer_n_kv     = cfg_.layer_n_kv(layer);
                const int layer_q_size   = cfg_.layer_q_size(layer);
                const int layer_kv_size  = cfg_.layer_kv_size(layer);
                Tensor q    = projection.query.view({layer_head_dim, cfg_.n_q, total});
                Tensor gate = projection.gate.view({layer_head_dim, cfg_.n_q, total});
                Tensor k    = projection.key.view({layer_head_dim, layer_n_kv, total});
                Tensor v    = projection.value.view({layer_head_dim, layer_n_kv, total});
                Tensor q_flat    = q.view({layer_q_size, total});
                Tensor gate_flat = gate.view({layer_q_size, total});
                Tensor k_flat    = k.view({layer_kv_size, total});
                Tensor v_flat    = v.view({layer_kv_size, total});
                Variant::attention_projection(h, *full.projection, q_flat, gate_flat, k_flat,
                                              v_flat, Phase::Prefill, work_, s);
                debug_probe<Variant>("q_proj_raw", q_flat, cfg_.n_layers, s);
                debug_probe<Variant>("k_proj_raw", k_flat, cfg_.n_layers, s);
                debug_probe<Variant>("v_proj_raw", v_flat, cfg_.n_layers, s);

                const auto results = workspace_recipe::text_attention_results(
                    work_, cfg_.layer_q_size(layer), cfg_.layer_kv_size(layer),
                                                                                         total);
                Tensor qn = attention_qk_norm<Variant>()
                                ? results.normalized_query.view({layer_head_dim, cfg_.n_q, total})
                                : q;
                Tensor kn = attention_qk_norm<Variant>()
                                ? results.normalized_key.view({layer_head_dim, layer_n_kv, total})
                                : k;
                if constexpr (attention_qk_norm<Variant>()) {
                    ops::rmsnorm(q, *full.q_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), qn, s);
                    if (owns_kv) {
                        ops::rmsnorm(k, *full.k_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), kn, s);
                    }
                }
                debug_probe<Variant>("q_post_headnorm", qn.view({layer_q_size, total}), cfg_.n_layers, s);
                debug_probe<Variant>("k_post_headnorm", kn.view({layer_kv_size, total}), cfg_.n_layers, s);

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
                if constexpr (applies_rotary<Variant>()) {
                    ops::rope(rope_all, cfg_.layer_rotary_dim(layer),
                              cfg_.layer_rotary_pairs(layer),
                              layer_rope_theta(layer, weights_.geometry), qn, kn, s);
                }
                // A windowed layer looks at fewer keys than the round is sized for;
                // see layer_sliding_window() in residual_policy.h.
                const std::int32_t layer_window = layer_sliding_window(layer, weights_.geometry);
                debug_probe<Variant>("q_post_rope", qn.view({layer_q_size, total}), cfg_.n_layers, s);
                debug_probe<Variant>("k_post_rope", kn.view({layer_kv_size, total}), cfg_.n_layers, s);

                Tensor a = results.attention.view({layer_head_dim, cfg_.n_q, total});
                for (std::size_t sg = 0; sg < segments.size(); ++sg) {
                    const int off = segment_begin[sg];
                    const int len = static_cast<int>(segments[sg].ids.size());
                    // The row scalar is stream-ordered against this segment's launch, so one
                    // scalar serves every segment in turn.
                    if (segments[sg].kv_table_row >= 0) {
                        ops::set_i32_scalar(io_.text_kv_table_row, segments[sg].kv_table_row, s);
                    }
                    const auto seen = static_cast<std::uint32_t>(segments[sg].kv_base + len);
                    const ops::GqaExecutionEnvelope envelope{seen, seen, layer_window};
                    Tensor qa = qn.slice(2, off, len);
                    Tensor ka = kn.slice(2, off, len);
                    Tensor va = v.slice(2, off, len);
                    Tensor aa = a.slice(2, off, len);
                    if (owns_kv) {
                        ops::gqa_attention(qa, ka, va, positions.slice(0, off, len), Tensor{},
                                           io_.text_kv_table_row, cfg_.attention_scale,
                                           kv_view, envelope, work_, aa, s);
                    } else {
                        ops::gqa_attention_cached(qa, positions.slice(0, off, len), Tensor{},
                                                  io_.text_kv_table_row, cfg_.attention_scale,
                                                  kv_view, envelope, work_, aa, s);
                    }
                }
                if (batch > 0) {
                    Tensor qb = qn.slice(2, prefill_cols, batch)
                                    .view({layer_head_dim, cfg_.n_q, 1, batch});
                    Tensor kb = kn.slice(2, prefill_cols, batch)
                                    .view({layer_head_dim, layer_n_kv, 1, batch});
                    Tensor vb = v.slice(2, prefill_cols, batch)
                                    .view({layer_head_dim, layer_n_kv, 1, batch});
                    Tensor ab = a.slice(2, prefill_cols, batch)
                                    .view({layer_head_dim, cfg_.n_q, 1, batch});
                    Tensor position_batch = decode.cache_positions.view({1, batch});
                    auto decode_scope = work_.scope();
                    const ops::GqaBlockMask decode_selection = text_indexer_selection(
                        full, h.slice(1, prefill_cols, batch), batch, decode.cache_positions,
                        rope_all.slice(0, prefill_cols, batch), decode.kv_table_rows, 1,
                        static_cast<std::int32_t>(decode.envelope.max_visible_keys),
                        kv_view);
                    ops::GqaExecutionEnvelope decode_layer_envelope = decode.envelope;
                    decode_layer_envelope.sliding_window                = layer_window;
                    if (owns_kv) {
                        ops::gqa_attention(qb, kb, vb, position_batch, Tensor{}, decode.kv_table_rows,
                                           cfg_.attention_scale, kv_view, decode_layer_envelope,
                                           work_, ab, s, decode_selection);
                    } else {
                        ops::gqa_attention_cached(qb, position_batch, Tensor{}, decode.kv_table_rows,
                                                  cfg_.attention_scale, kv_view,
                                                  decode_layer_envelope, work_, ab, s,
                                                  decode_selection);
                    }
                }
                // A dense stack writes no gate rows; see attention_output_gate<Variant>().
                if constexpr (kAttentionOutputGate) { apply_attention_gate<Variant>(gate, a, s); }
                Hooks::attention_output(a.view({layer_q_size, total}), *full.o_proj,
                                        *full.projection, x,
                                                     Phase::Prefill, work_, s);
                if (timing) { lap(timer.begin, timer.attn, acc_attn); }
            }
            {
                auto mlp_scope = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, s); }
                mlp_tail(full.post_attn_norm, full.mlp, x, layer, Phase::Prefill);
                if (timing) { lap(timer.begin, timer.mlp_full, acc_mlp_full); }
            }
        } else {
            const int gidx       = cfg_.gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            if constexpr (kLinearMixer == family::LinearMixer::ShortConv) {
                auto mixer_scope = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, s); }
                std::vector<ShortConvSegment> parts;
                parts.reserve(segments.size());
                for (std::size_t sg = 0; sg < segments.size(); ++sg) {
                    parts.push_back({segment_begin[sg],
                                     static_cast<std::int32_t>(segments[sg].ids.size()),
                                     static_cast<std::int32_t>(segments[sg].state_slot)});
                }
                short_conv_mix_mixed(gdn, x, gidx, parts, prefill_cols, batch, Tensor{},
                                     decode.linear_state_slots);
                if (timing) { lap(timer.begin, timer.gdn, acc_gdn); }
            } else {
                auto mixer_scope   = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, s); }
                const auto control = workspace_recipe::gdn_control(work_, cfg_geometry(), total, kLinearMixer);
                Tensor h           = control.hidden;
                Tensor g           = control.g;
                Tensor beta        = control.beta;
                Variant::gdn_norm_control_projection(x, *gdn.input_norm, cfg_.rms_eps,
                                                     *gdn.projection, h, g, beta, work_, s);

                float acc_g_ctrl = 0, acc_g_proj = 0, acc_g_conv = 0, acc_g_extract = 0,
                      acc_g_scan = 0, acc_g_norm = 0, acc_g_out = 0;
                if (timing) { lap(timer.begin, timer.g_ctrl, acc_g_ctrl); cudaEventRecord(timer.begin, s); }
                const auto projection = workspace_recipe::gdn_projection(work_, cfg_geometry(), total);
                Tensor z  = projection.output_gate.view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, total});
                Tensor qc = projection.query;
                Tensor kc = projection.key;
                Tensor vc = projection.value;
                const auto conv = workspace_recipe::gdn_prefill_conv(work_, cfg_geometry(), total);
                Tensor qkv      = conv.projected;
                debug_probe<Variant>("mixed_gdn_in", x, cfg_.n_layers, s);
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
                                       .view({cfg_.conv_dim, 1, batch});
                    Tensor qkv_cb = qkv_c.slice(1, prefill_cols, batch)
                                        .view({cfg_.conv_dim, 1, batch});
                    ops::causal_conv1d_silu_snapshot(qkv_b, *gdn.conv1d,
                                                     state_.conv.at(static_cast<std::size_t>(gidx)),
                                                     Tensor{}, decode.linear_state_slots,
                                                     decode.linear_state_slots, qkv_cb, s);
                }
                if (timing) { lap(timer.begin, timer.g_conv, acc_g_conv); cudaEventRecord(timer.begin, s); }
                debug_probe<Variant>("mixed_gdn_conv", qkv_c, cfg_.n_layers, s);
                ops::extract_bf16_columns(qkv_c, 0, qc, s);
                ops::extract_bf16_columns(qkv_c, cfg_.key_dim, kc, s);
                ops::extract_bf16_columns(qkv_c, 2 * cfg_.key_dim, vc, s);
                if (timing) { lap(timer.begin, timer.g_extract, acc_g_extract); cudaEventRecord(timer.begin, s); }

                Tensor q_recurrent = qc.view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, total});
                Tensor k_recurrent = kc.view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, total});
                Tensor vv          = vc.view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, total});
                Tensor o = workspace_recipe::gdn_recurrent_output(work_, cfg_geometry(), total)
                               .view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, total});
                {
                    for (std::size_t sg = 0; sg < segments.size(); ++sg) {
                    const int off = segment_begin[sg];
                    const int len = static_cast<int>(segments[sg].ids.size());
                    Tensor qa = q_recurrent.slice(2, off, len);
                    Tensor ka = k_recurrent.slice(2, off, len);
                    Tensor va = vv.slice(2, off, len);
                    Tensor ga = family::detail::linear_gate_view<Variant>(
                        g.slice(1, off, len), cfg_.gdn_v_dim, cfg_.gdn_v_heads, len);
                    Tensor ba = beta.slice(1, off, len);
                    Tensor oa = o.slice(2, off, len);
                    Tensor recurrent_state = state_.recurrent_slot(
                        static_cast<std::uint32_t>(gidx),
                        static_cast<std::uint32_t>(segments[sg].state_slot));
                    family::detail::linear_recurrence<Variant>(qa, ka, va, ga, ba, cfg_.gdn_scale,
                                                              work_, recurrent_state, oa, s);
                    }
                }
                if (batch > 0) {
                    Tensor qb = q_recurrent.slice(2, prefill_cols, batch)
                                    .view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, 1, batch});
                    Tensor kb = k_recurrent.slice(2, prefill_cols, batch)
                                    .view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, 1, batch});
                    Tensor vb = vv.slice(2, prefill_cols, batch)
                                    .view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, 1, batch});
                    Tensor gb = family::detail::linear_gate_view<Variant>(
                        g.slice(1, prefill_cols, batch), cfg_.gdn_v_dim, cfg_.gdn_v_heads, 1,
                        batch);
                    Tensor bb =
                        beta.slice(1, prefill_cols, batch).view({cfg_.gdn_v_heads, 1, batch});
                    Tensor ob = o.slice(2, prefill_cols, batch)
                                    .view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, 1, batch});
                    family::detail::linear_recurrence_snapshot<Variant>(
                        qb, kb, vb, gb, bb, cfg_.gdn_scale,
                        state_.recurrent.at(static_cast<std::size_t>(gidx)), Tensor{},
                        decode.linear_state_slots, decode.linear_state_slots, ob, s);
                }
                Tensor on = workspace_recipe::gdn_normalized_output(work_, cfg_geometry(), total)
                                .view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, total});
                if (timing) { lap(timer.begin, timer.g_scan, acc_g_scan); cudaEventRecord(timer.begin, s); }
                ops::gated_rmsnorm(o, *gdn.gdn_norm, z, cfg_.rms_eps, gdn_output_gate<Variant>(), on, s);
                if (timing) { lap(timer.begin, timer.g_norm, acc_g_norm); cudaEventRecord(timer.begin, s); }
                Variant::gdn_output_projection(on.view({cfg_.value_dim, total}), *gdn.out_proj, x,
                                               Phase::Prefill, work_, s);
                if (timing) { lap(timer.begin, timer.g_out, acc_g_out); timer.t_g_ctrl += acc_g_ctrl; timer.t_g_proj += acc_g_proj; timer.t_g_conv += acc_g_conv; timer.t_g_extract += acc_g_extract; timer.t_g_scan += acc_g_scan; timer.t_g_norm += acc_g_norm; timer.t_g_out += acc_g_out; cudaEventRecord(timer.begin, s); }
                if (timing) { lap(timer.begin, timer.gdn, acc_gdn); }
                if (timing) { acc_gdn += acc_g_proj + acc_g_conv + acc_g_scan + acc_g_out; }
            }
            {
                auto mlp_scope = work_.scope();
                if (timing) { cudaEventRecord(timer.begin, s); }
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, layer, Phase::Prefill);
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
                    : work_.alloc(DType::BF16, {cfg_.hidden, total});
    if (!stage_finishes()) {
        stage_export(x, s);
    } else {
    Tensor xl = finish_prefill(xf, x, s);
    if (prefill_cols > 0) {
        debug_next_token_nll(xl, ids_device, prefill_cols, *lm_head_, cfg_.vocab, static_cast<std::int32_t>(kTokenDomain), cfg_.logit_softcap, work_, s);
    }

    if (batch > 0) {
        Tensor xf_decode = xf.slice(1, prefill_cols, batch);
        Tensor xl_decode = xl.slice(1, prefill_cols, batch);
        // The decode columns' boundary hidden, as wide as the round's: the residual where a
        // trunk-block draft head widened it. This copied the model width of it, so a lane's
        // tail hidden -- what a zero-suffix reuse samples from, and what the head aligns on
        // below -- was short by the rest under such a head.
        if (decode.hidden.ne[0] != xf.ne[0] || decode.hidden.ne[1] != batch) {
            throw std::logic_error("mixed chunk decode hidden does not match the round's width");
        }
        CUDA_CHECK(cudaMemcpyAsync(decode.hidden.data, xf_decode.data, xf_decode.bytes(),
                                   cudaMemcpyDeviceToDevice, s));
        Tensor logits_decode = decode.logits;
        ops::linear(xl_decode, *lm_head_, logits_decode, s);
        apply_logit_softcap(cfg_, logits_decode, s);
    }

    // The draft head over each segment's columns, so a round under the head keeps the head's
    // KV as current as the trunk's. The columns pair with the prompt's next tokens (the ids
    // shifted by one), never with a sampled one: a mixed round stops a token short of a
    // prompt's end, and the prompt's final chunk pairs its last column with the token it
    // samples and proposes from it. The head's row scalar is stream-ordered per segment, as
    // the trunk's is above.
    for (std::size_t sg = 0; sg < segments.size(); ++sg) {
        const MixedPrefillSegment& segment = segments[sg];
        if (segment.mtp_kv_table_row < 0) { continue; }
        const int len   = static_cast<int>(segment.ids.size());
        const int count = static_cast<int>(segment.mtp_shifted_ids.size());
        if (segment.finalize || count > len) {
            throw std::invalid_argument(
                "mixed chunk head alignment needs at most one shifted id per column and no finalizer");
        }
        // A segment that ends at the prompt's last column aligns one column fewer: that column
        // pairs with the token the zero-suffix step samples. A one-token tail aligns nothing.
        if (count == 0) { continue; }
        if (!segment.mtp_kv.valid()) {
            throw std::logic_error("mixed chunk head alignment without the segment's head KV");
        }
        // The fixed tail appends through the card's per-sequence head view; each segment
        // brings its own, bound for its call (the trunk-block head reads the batch view by
        // the row scalar below and ignores it).
        ScopedValue<family::PagedKVCacheView> segment_view(mtp_kv_, segment.mtp_kv);
        auto alignment_scope = work_.scope();
        // SUROGATE_SERVE_PREFILL_TIMING=1: the alignment's host time and stream time, per segment.
        static const bool timing = std::getenv("SUROGATE_SERVE_PREFILL_TIMING") != nullptr;
        const auto host_begin = std::chrono::steady_clock::now();
        cudaEvent_t begin_event{}, end_event{};
        if (timing) {
            CUDA_CHECK(cudaEventCreateWithFlags(&begin_event, cudaEventDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&end_event, cudaEventDefault));
            CUDA_CHECK(cudaEventRecord(begin_event, s));
        }
        Tensor shifted       = work_.alloc(DType::I32, {count});
        copy_i32(segment.mtp_shifted_ids.data(), shifted, s);
        ops::set_i32_scalar(io_.backend_kv_table_row, segment.mtp_kv_table_row, s);
        Tensor segment_hidden    = xf.slice(1, segment_begin[sg], count);
        Tensor segment_positions = positions.slice(0, segment_begin[sg], count);
        const auto seen          = static_cast<std::uint32_t>(segment.kv_base + count);
        mtp_prefill_chunk(shifted, segment_hidden, nullptr, segment_positions, segment_positions,
                          ops::GqaExecutionEnvelope{seen, seen}, false, nullptr, nullptr, nullptr);
        if (timing) {
            const double host_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - host_begin).count();
            CUDA_CHECK(cudaEventRecord(end_event, s));
            CUDA_CHECK(cudaEventSynchronize(end_event));
            float stream_ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&stream_ms, begin_event, end_event));
            std::fprintf(stderr,
                         "mtp-timing: mixed alignment of %d columns (base %d): host %.1f ms to "
                         "enqueue, %.1f ms on the stream\n",
                         count, segment.kv_base, host_ms, stream_ms);
            CUDA_CHECK(cudaEventDestroy(begin_event));
            CUDA_CHECK(cudaEventDestroy(end_event));
        }
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
                    static_cast<std::size_t>(slot) * round_hidden_width() * 2,
                last_xf.data, static_cast<std::size_t>(round_hidden_width()) * 2,
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
        ops::linear(lm_head_view(gathered, s), *lm_head_, logits, s);
        apply_logit_softcap(cfg_, logits, s);
        Tensor sampled   = finalize.tokens.slice(0, 0, finalizers);
        Tensor positions_out = finalize.positions.slice(0, 0, finalizers);
        ops::sample(logits, sampled, cfg_.token_domain, finalize.sampling, positions_out,
                    ops::kSamplePurposePrefill, work_, s);
    }
    } // stage_finishes
    const bool is_last = finalizers > 0;

    if (body_timing) {
        CUDA_CHECK(cudaEventRecord(body_end, s));
        const double host_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - body_host_begin).count();
        CUDA_CHECK(cudaEventSynchronize(body_end));
        float stream_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&stream_ms, body_begin, body_end));
        std::fprintf(stderr,
                     "mixed-timing: stage [%d,%d) %d prefill columns in %zu segment(s) + %d rows: "
                     "%.1f ms on the stream, %.1f ms host to enqueue\n",
                     stage_first_, stage_last_, prefill_cols, segments.size(), batch, stream_ms,
                     host_ms);
        CUDA_CHECK(cudaEventDestroy(body_begin));
        CUDA_CHECK(cudaEventDestroy(body_end));
    }
    // Enqueued, not finished: the mixed round is a launch/consume pair, and the consume
    // synchronises. This body used to synchronise here, which the graph replay never did, so
    // whenever a round could not replay a graph -- every round under the draft head -- the
    // launch blocked the pipeline driver for the round's whole duration and the stages ran
    // one after another: a wave of eight prompts took 53 % longer than the graph path's on
    // four stages with the same per-stage stream time. The workspace reset is stream-safe:
    // the next body on this stage is launched only after this one has been consumed.
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
    const auto roots = workspace_recipe::text_prefill_roots(work_, cfg_geometry(), bucket, 0, 0);

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
    active_ids_ = ids_device;
    if (stage_embeds()) { Hooks::embed(weights_, ids_device, x, work_, s); } else { stage_import(x, s); }
    capture_per_layer_source(x, s);
    NullTap tap;
    run_layers(x, Phase::Prefill, tap);

    if (prefill_hidden_.data == nullptr) {
        throw std::logic_error("prefill graph body requires the persistent prefill hidden store");
    }
    Tensor xf = matrix_window(prefill_hidden_, bucket);
    if (stage_finishes()) {
        Tensor xl = finish_prefill(xf, x, s);
        debug_next_token_nll(xl, ids_device, bucket, *lm_head_, cfg_.vocab, static_cast<std::int32_t>(kTokenDomain), cfg_.logit_softcap, work_, s);
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
    const auto roots = workspace_recipe::text_prefill_roots(work_, cfg_geometry(), total, 0, 0);

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
    active_ids_ = ids_device;
    if (stage_embeds()) { Hooks::embed(weights_, ids_device, x, work_, s); } else { stage_import(x, s); }
    capture_per_layer_source(x, s);

    // Pipeline work asks the same question of every failure: which layers did this round
    // actually run. `SUROGATE_SERVE_TRACE_STAGE=1` answers it once per round.
    if (stage_trace_enabled()) {
        std::fprintf(stderr, "stage-trace: layers [%d, %d) of %d, columns %d\n", stage_first_,
                     stage_last_, cfg_.n_layers, x.ne[1]);
    }
    for (int layer = stage_first_; layer < stage_last_; ++layer) {
        Hooks::layer_prologue(weights_, layer, x, prologue_, ple_state_, work_, ctx_.stream);
        if (cfg_.is_full(layer)) {
            const int fidx         = cfg_.full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            const bool owns_kv = cfg_.layer_owns_kv(layer);
            const auto kv_view = batch_text_kv_->batch_layer_view(cfg_.kv_plane_index(layer));
            {
                auto mixer_scope      = work_.scope();
                const auto projection = workspace_recipe::text_attention_projection(
                    work_, cfg_.hidden, cfg_.layer_q_size(layer), cfg_.layer_kv_size(layer),
                    total);
                Tensor h = projection.hidden;
                Hooks::attention_norm(x, *full.input_norm, cfg_.rms_eps, *full.projection, h,
                                      work_, s);
                const int layer_head_dim = cfg_.layer_head_dim(layer);
                const int layer_n_kv     = cfg_.layer_n_kv(layer);
                const int layer_q_size   = cfg_.layer_q_size(layer);
                const int layer_kv_size  = cfg_.layer_kv_size(layer);
                Tensor q    = projection.query.view({layer_head_dim, cfg_.n_q, total});
                Tensor gate = projection.gate.view({layer_head_dim, cfg_.n_q, total});
                Tensor k    = projection.key.view({layer_head_dim, layer_n_kv, total});
                Tensor v    = projection.value.view({layer_head_dim, layer_n_kv, total});
                Tensor q_flat    = q.view({layer_q_size, total});
                Tensor gate_flat = gate.view({layer_q_size, total});
                Tensor k_flat    = k.view({layer_kv_size, total});
                Tensor v_flat    = v.view({layer_kv_size, total});
                Variant::attention_projection(h, *full.projection, q_flat, gate_flat, k_flat,
                                              v_flat, Phase::Prefill, work_, s);

                const auto results = workspace_recipe::text_attention_results(
                    work_, cfg_.layer_q_size(layer), cfg_.layer_kv_size(layer),
                                                                                         total);
                Tensor qn = attention_qk_norm<Variant>()
                                ? results.normalized_query.view({layer_head_dim, cfg_.n_q, total})
                                : q;
                Tensor kn = attention_qk_norm<Variant>()
                                ? results.normalized_key.view({layer_head_dim, layer_n_kv, total})
                                : k;
                if constexpr (attention_qk_norm<Variant>()) {
                    ops::rmsnorm(q, *full.q_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), qn, s);
                    if (owns_kv) {
                        ops::rmsnorm(k, *full.k_norm, cfg_.rms_eps, norm_unit_offset<Variant>(), kn, s);
                    }
                }

                Tensor rope_positions = roots.positions;
                Tensor rope_all       = rope_positions.view({total});
                if (batch > 0) {
                    Tensor rope_decode = rope_positions.slice(0, prefill_cols, batch);
                    CUDA_CHECK(cudaMemcpyAsync(rope_decode.data, decode.rope_positions.data,
                                               static_cast<std::size_t>(batch) *
                                                   sizeof(std::int32_t),
                                               cudaMemcpyDeviceToDevice, s));
                }
                if constexpr (applies_rotary<Variant>()) {
                    ops::rope(rope_all, cfg_.layer_rotary_dim(layer),
                              cfg_.layer_rotary_pairs(layer),
                              layer_rope_theta(layer, weights_.geometry), qn, kn, s);
                }
                // A windowed layer looks at fewer keys than the round is sized for;
                // see layer_sliding_window() in residual_policy.h.
                const std::int32_t layer_window = layer_sliding_window(layer, weights_.geometry);
                ops::GqaExecutionEnvelope prefill_layer_envelope = prefill_envelope;
                prefill_layer_envelope.sliding_window                = layer_window;
                ops::GqaExecutionEnvelope decode_layer_envelope = decode_envelope;
                decode_layer_envelope.sliding_window            = layer_window;

                Tensor a = results.attention.view({layer_head_dim, cfg_.n_q, total});
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
                        kv_view);
                    if (owns_kv) {
                        ops::gqa_attention(qa, ka, va, positions_prefill, Tensor{},
                                           io_.text_kv_table_row, cfg_.attention_scale, kv_view,
                                           prefill_layer_envelope, work_, aa, s, segment_selection);
                    } else {
                        ops::gqa_attention_cached(qa, positions_prefill, Tensor{},
                                                  io_.text_kv_table_row, cfg_.attention_scale,
                                                  kv_view, prefill_layer_envelope, work_, aa, s,
                                                  segment_selection);
                    }
                }
                if (batch > 0) {
                    Tensor qb = qn.slice(2, prefill_cols, batch)
                                    .view({layer_head_dim, cfg_.n_q, 1, batch});
                    Tensor kb = kn.slice(2, prefill_cols, batch)
                                    .view({layer_head_dim, layer_n_kv, 1, batch});
                    Tensor vb = v.slice(2, prefill_cols, batch)
                                    .view({layer_head_dim, layer_n_kv, 1, batch});
                    Tensor ab = a.slice(2, prefill_cols, batch)
                                    .view({layer_head_dim, cfg_.n_q, 1, batch});
                    Tensor position_batch = decode.cache_positions.view({1, batch});
                    auto decode_scope = work_.scope();
                    const ops::GqaBlockMask decode_selection = text_indexer_selection(
                        full, h.slice(1, prefill_cols, batch), batch, decode.cache_positions,
                        rope_all.slice(0, prefill_cols, batch), decode.kv_table_rows, 1,
                        static_cast<std::int32_t>(decode_envelope.max_visible_keys),
                        kv_view);
                    if (owns_kv) {
                        ops::gqa_attention(qb, kb, vb, position_batch, Tensor{}, decode.kv_table_rows,
                                           cfg_.attention_scale, kv_view, decode_layer_envelope,
                                           work_, ab, s, decode_selection);
                    } else {
                        ops::gqa_attention_cached(qb, position_batch, Tensor{}, decode.kv_table_rows,
                                                  cfg_.attention_scale, kv_view,
                                                  decode_layer_envelope, work_, ab, s,
                                                  decode_selection);
                    }
                }
                // A dense stack writes no gate rows; see attention_output_gate<Variant>().
                if constexpr (kAttentionOutputGate) { apply_attention_gate<Variant>(gate, a, s); }
                Hooks::attention_output(a.view({layer_q_size, total}), *full.o_proj,
                                        *full.projection, x,
                                                     Phase::Prefill, work_, s);
            }
            {
                auto mlp_scope = work_.scope();
                mlp_tail(full.post_attn_norm, full.mlp, x, layer, Phase::Prefill);
            }
        } else {
            const int gidx       = cfg_.gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            if constexpr (kLinearMixer == family::LinearMixer::ShortConv) {
                auto mixer_scope = work_.scope();
                const ShortConvSegment window{0, prefill_cols,
                                              static_cast<std::int32_t>(linear_state_current_slot_)};
                short_conv_mix_mixed(gdn, x, gidx, std::span<const ShortConvSegment>(&window, 1),
                                     prefill_cols, batch, valid, decode.linear_state_slots);
            } else {
                auto mixer_scope   = work_.scope();
                const auto control = workspace_recipe::gdn_control(work_, cfg_geometry(), total, kLinearMixer);
                Tensor h           = control.hidden;
                Tensor g           = control.g;
                Tensor beta        = control.beta;
                Variant::gdn_norm_control_projection(x, *gdn.input_norm, cfg_.rms_eps,
                                                     *gdn.projection, h, g, beta, work_, s);
                {
                    // Zero g/beta over the prefill window's pad columns only
                    // (the decode columns behind the window are live lanes).
                    Tensor g_prefill    = g.slice(1, 0, prefill_cols);
                    Tensor beta_prefill = beta.slice(1, 0, prefill_cols);
                    ops::mask_columns_zero(g_prefill, valid, s);
                    ops::mask_columns_zero(beta_prefill, valid, s);
                }

                const auto projection = workspace_recipe::gdn_projection(work_, cfg_geometry(), total);
                Tensor z  = projection.output_gate.view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, total});
                Tensor qc = projection.query;
                Tensor kc = projection.key;
                Tensor vc = projection.value;
                const auto conv = workspace_recipe::gdn_prefill_conv(work_, cfg_geometry(), total);
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
                                       .view({cfg_.conv_dim, 1, batch});
                    Tensor qkv_cb = qkv_c.slice(1, prefill_cols, batch)
                                        .view({cfg_.conv_dim, 1, batch});
                    ops::causal_conv1d_silu_snapshot(qkv_b, *gdn.conv1d,
                                                     state_.conv.at(static_cast<std::size_t>(gidx)),
                                                     Tensor{}, decode.linear_state_slots,
                                                     decode.linear_state_slots, qkv_cb, s);
                }
                ops::extract_bf16_columns(qkv_c, 0, qc, s);
                ops::extract_bf16_columns(qkv_c, cfg_.key_dim, kc, s);
                ops::extract_bf16_columns(qkv_c, 2 * cfg_.key_dim, vc, s);

                Tensor q_recurrent = qc.view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, total});
                Tensor k_recurrent = kc.view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, total});
                Tensor vv          = vc.view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, total});
                Tensor o = workspace_recipe::gdn_recurrent_output(work_, cfg_geometry(), total)
                               .view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, total});
                {
                    Tensor qa = q_recurrent.slice(2, 0, prefill_cols);
                    Tensor ka = k_recurrent.slice(2, 0, prefill_cols);
                    Tensor va = vv.slice(2, 0, prefill_cols);
                    Tensor ga = family::detail::linear_gate_view<Variant>(
                        g.slice(1, 0, prefill_cols), cfg_.gdn_v_dim, cfg_.gdn_v_heads,
                        prefill_cols);
                    Tensor ba = beta.slice(1, 0, prefill_cols);
                    Tensor oa = o.slice(2, 0, prefill_cols);
                    Tensor recurrent_state =
                        state_.recurrent_slot(static_cast<std::uint32_t>(gidx),
                                              linear_state_current_slot_);
                    family::detail::linear_recurrence<Variant>(qa, ka, va, ga, ba, cfg_.gdn_scale,
                                                              work_, recurrent_state, oa, s);
                }
                if (batch > 0) {
                    Tensor qb = q_recurrent.slice(2, prefill_cols, batch)
                                    .view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, 1, batch});
                    Tensor kb = k_recurrent.slice(2, prefill_cols, batch)
                                    .view({cfg_.gdn_k_dim, cfg_.gdn_k_heads, 1, batch});
                    Tensor vb = vv.slice(2, prefill_cols, batch)
                                    .view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, 1, batch});
                    Tensor gb = family::detail::linear_gate_view<Variant>(
                        g.slice(1, prefill_cols, batch), cfg_.gdn_v_dim, cfg_.gdn_v_heads, 1,
                        batch);
                    Tensor bb =
                        beta.slice(1, prefill_cols, batch).view({cfg_.gdn_v_heads, 1, batch});
                    Tensor ob = o.slice(2, prefill_cols, batch)
                                    .view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, 1, batch});
                    family::detail::linear_recurrence_snapshot<Variant>(
                        qb, kb, vb, gb, bb, cfg_.gdn_scale,
                        state_.recurrent.at(static_cast<std::size_t>(gidx)), Tensor{},
                        decode.linear_state_slots, decode.linear_state_slots, ob, s);
                }
                Tensor on = workspace_recipe::gdn_normalized_output(work_, cfg_geometry(), total)
                                .view({cfg_.gdn_v_dim, cfg_.gdn_v_heads, total});
                ops::gated_rmsnorm(o, *gdn.gdn_norm, z, cfg_.rms_eps, gdn_output_gate<Variant>(), on, s);
                Variant::gdn_output_projection(on.view({cfg_.value_dim, total}), *gdn.out_proj, x,
                                               Phase::Prefill, work_, s);
            }
            {
                auto mlp_scope = work_.scope();
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, layer, Phase::Prefill);
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
    Tensor xl = finish_prefill(xf, x, s);
    debug_next_token_nll(xl, ids_device, prefill_cols, *lm_head_, cfg_.vocab, static_cast<std::int32_t>(kTokenDomain), cfg_.logit_softcap, work_, s);

    if (batch == 0) { return; }
    Tensor xf_decode = xf.slice(1, prefill_cols, batch);
    Tensor xl_decode = xl.slice(1, prefill_cols, batch);
    if (decode.hidden.ne[0] != xf.ne[0] || decode.hidden.ne[1] != batch) {
        throw std::logic_error("mixed graph decode hidden does not match the round's width");
    }
    CUDA_CHECK(cudaMemcpyAsync(decode.hidden.data, xf_decode.data, xf_decode.bytes(),
                               cudaMemcpyDeviceToDevice, s));
    Tensor logits_decode = decode.logits;
    ops::linear(xl_decode, *lm_head_, logits_decode, s);
    apply_logit_softcap(cfg_, logits_decode, s);
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
    // Bisection switch: run only a prompt's first chunk through the mixed graph and the
    // continuation chunks (begin > 0) through the eager mixed body.
    static const bool kFirstChunkOnly =
        std::getenv("SUROGATE_SERVE_MIXED_GRAPH_FIRST_CHUNK_ONLY") != nullptr;
    if (kFirstChunkOnly && begin > 0) { return false; }
    const int len                   = static_cast<int>(nominal);
    const std::int32_t chunk_bucket = family.bucket_for(len);
    if (chunk_bucket < len ||
        chunk_bucket + batch_bucket > static_cast<std::int32_t>(prefill_chunk_)) {
        return false;
    }
    // Same refusal as try_prefill_graph_chunk: the prefill side of the mixed body
    // writes its whole bucket from text_kv_base_, and must not run past capacity.
    if (static_cast<std::uint32_t>(text_kv_base_) + static_cast<std::uint32_t>(chunk_bucket) >
        family.kv_capacity()) {
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

void TextContext::observe_prompt_logits(const Tensor& hidden, int base, cudaStream_t stream) {
    const int first = std::max(0, score_prompt_start - base - 1);
    const int end = std::min(hidden.ne[1], score_prompt_end - base - 1);
    if (first >= end) return;
    auto scope = work_.scope();
    const std::size_t spare = work_.capacity() - work_.used();
    // Leave room for the head's normalization and projection scratch. Fall back
    // to the persistent one-column logits when a small model has no spare arena.
    const std::size_t bytes_per_column = std::size_t(cfg_.vocab) * 2 + std::size_t(cfg_.hidden) * 16;
    const int fits = spare > 16384 ? int((spare - 16384) / bytes_per_column) : 0;
    const int stripe = std::max(1, std::min({32, fits, end - first}));
    Tensor storage = stripe > 1 ? work_.alloc(DType::BF16, {cfg_.vocab, stripe}) : matrix_window(io_.logits, 1);
    for (int col = first; col < end; col += stripe) {
        auto scratch = work_.scope();
        const int count = std::min(stripe, end - col);
        Tensor logits = storage.slice(1, 0, count);
        ops::linear(lm_head_view(hidden.slice(1, col, count), stream), *lm_head_, logits, stream);
        apply_logit_softcap(cfg_, logits, stream);
        logprob_observer(logits, base + col + 1, false);
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
    // The captured body writes every column of its bucket, pad columns included,
    // through block-table rows that are exactly one capacity wide with no bound
    // check in the append. A chunk whose bucket would run past capacity is not
    // replayed: the eager body writes exactly `len`, which admission mapped.
    if (static_cast<std::uint32_t>(base_i + t0) + static_cast<std::uint32_t>(bucket) >
        family.kv_capacity()) {
        return false;
    }

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
    if (score_prompt && logprob_observer && stage_finishes()) {
        observe_prompt_logits(matrix_window(prefill_hidden_, len), base_i + t0, s);
    }
    if (is_last) {
        Tensor xf      = matrix_window(prefill_hidden_, len);
        Tensor last_xf = xf.slice(1, len - 1, 1);
        Tensor logits  = matrix_window(io_.logits, 1);
        ops::linear(lm_head_view(last_xf, s), *lm_head_, logits, s);
        apply_logit_softcap(cfg_, logits, s);
        ops::set_i32_scalar(io_.pos, base_i + T, s);
        ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
        work_.reset();
        if (sampling_config_ != nullptr) {
            ops::sample(logits, io_.token, cfg_.token_domain, sampling_config_, io_.pos,
                        ops::kSamplePurposePrefill, work_, s);
            if (io_.logprob.data != nullptr) {
                ops::sampled_logprob(logits, io_.token, io_.logprob, cfg_.token_domain, sampling_config_, s);
                if (logprob_observer) logprob_observer(logits, base_i + T, true);
            }
        } else {
            ops::argmax(logits, io_.token, cfg_.token_domain, s);
        }
    }
    if (checkpoint_rel > 0 && t0 + len == checkpoint_rel &&
        rewrite_checkpoint_hidden_output_ != nullptr) {
        require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16, {cfg_.hidden, 1},
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
            const auto roots             = workspace_recipe::text_prefill_roots(
                work_, cfg_geometry(), len, rope_axes, static_cast<std::int32_t>(local_scatter_indices.size()));
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
            active_ids_ = ids_device;
    if (stage_embeds()) { Hooks::embed(weights_, ids_device, x, work_, s); } else { stage_import(x, s); }
    capture_per_layer_source(x, s);
            window_laps.mark_pre();
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            Tensor deepstack;
            if (vision_chunk.deepstack.data != nullptr && !local_scatter_indices.empty()) {
                deepstack = vision_chunk.deepstack.slice(1, visual_begin,
                    static_cast<std::int32_t>(local_scatter_indices.size()));
            }
            run_layers(x, Phase::Prefill, tap, deepstack.data ? &deepstack : nullptr,
                       local_scatter_indices);
            window_laps.mark_layers();
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {cfg_.hidden, len});
            if (stage_finishes()) {
                stage_checksum("finish-input", stage_first_, stage_last_, x.data, cfg_.residual,
                               len, s);
                Tensor xl = finish_prefill(xf, x, s);
                stage_checksum("finish-output", stage_first_, stage_last_, xl.data, xl.ne[0],
                               xl.ne[1], s);
                debug_next_token_nll(xl, ids_device, len, *lm_head_, cfg_.vocab, static_cast<std::int32_t>(kTokenDomain), cfg_.logit_softcap, work_, s);
            } else {
                stage_export(x, s);
            }

            if (score_prompt && logprob_observer && stage_finishes()) observe_prompt_logits(xf, base_i + t0, s);
            if (is_last && stage_finishes()) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                ops::linear(lm_head_view(last_xf, s), *lm_head_, logits, s);
        apply_logit_softcap(cfg_, logits, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token, cfg_.token_domain, sampling_config_, io_.pos,
                                ops::kSamplePurposePrefill, work_, s);
                    if (io_.logprob.data != nullptr) {
                        ops::sampled_logprob(logits, io_.token, io_.logprob, cfg_.token_domain, sampling_config_, s);
                        if (logprob_observer) logprob_observer(logits, base_i + T, true);
                    }
                } else {
                    ops::argmax(logits, io_.token, cfg_.token_domain, s);
                }
            }

            // The head's own prefill -- its KV over the prompt and the first proposal -- runs
            // where the head is; a stage before the last exported its residual above and has
            // nothing to feed it.
            if (prepare_mtp_prompt && stage_finishes()) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const family::MtpAlignmentWindow mtp_window = family::plan_mtp_alignment_window(
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
                    mtp_input_embeddings = work_.alloc(DType::BF16, {cfg_.hidden, len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const family::MtpVisualOverlap overlap = family::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace_recipe::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            family::detail::scatter_shifted_visual_embeddings(
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
                        Tensor next_hidden    = work_.alloc(DType::BF16, {round_hidden_width(), 1});
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
                                     {cfg_.hidden, 1}, "rewrite checkpoint hidden output");
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

PrefillChunkResult TextContext::prefill_chunk(const family::PreparedPromptData& input,
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

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
