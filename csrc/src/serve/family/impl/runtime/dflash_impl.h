#include "family/impl/runtime/instance.h"
#include "family/impl/runtime/schedule.h"
#include "api/ops/sampled_logprob.h"
#include "family/impl/runtime/workspace_recipe.h"

#include "api/ops/argmax.h"
#include "api/ops/attn_input_proj.h"
#include "api/ops/bidirectional_gqa_attention.h"
#include "api/ops/embedding.h"
#include "api/ops/kv_cache_append_prefix.h"
#include "api/ops/linear.h"
#include "api/ops/linear_add.h"
#include "api/ops/linear_pair.h"
#include "api/ops/linear_swiglu.h"
#include "api/ops/prepare_masked_block.h"
#include "api/ops/prepare_ragged_prefix.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/rope.h"
#include "api/ops/scatter.h"
#include "api/ops/scalar.h"
#include "api/ops/speculative_round.h"
#include "api/ops/swa.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule {
namespace {

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.model.dflash.has_value()) {
        throw std::logic_error("DFlash schedule requires DFlash weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

template <class V>
DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        require_dflash_state(state);
        const auto& config = state.execution.model.geometry.dflash;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = config.target_layers(),
            .consume_prefill = std::move(consume_prefill),
            .stage = state.execution.stage,
            .stream = state.execution.device.stream,
        };
    }
}

template <class V>
DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        const auto& config = state.execution.model.geometry.dflash;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = config.target_layers(),
            .stage = state.execution.stage,
            .stream = state.execution.device.stream,
        };
    }
}

template <class V, class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash context append is unavailable for this target");
    } else {
        const auto& config = state.execution.model.geometry.dflash;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != config.feature_rows || features.ne[3] != 1 ||
            positions.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
            commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
            lanes.dtype != DType::I32 || lanes.ne[0] != batch || table_rows.dtype != DType::I32 ||
            table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window = batch == 1 && width > config.local_capacity;
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset = replace_local_window ? width - config.local_capacity : 0;
        const int local_width  = replace_local_window ? config.local_capacity : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(config.local_capacity)
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(config.local_capacity)
                                 : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, config.local_capacity,
                                state.execution.device.stream);
        }

        const auto context_roots =
            workspace_recipe::dflash_context(state.execution.work, columns, config);
        Tensor projected = context_roots.projected;
        ops::linear(features.view({config.feature_rows, columns}),
                    state.execution.model.dflash->feature_projection, projected,
                    state.execution.device.stream);
        Tensor context = context_roots.normalized;
        ops::rmsnorm(projected, state.execution.model.dflash->context_norm, config.rms_epsilon,
                     false, context, state.execution.device.stream);

        for (int layer = 0; layer < config.layers; ++layer) {
            auto layer_scope = state.execution.work.scope();
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            const bool local_layer  = layer < config.local_layers;
            const int layer_width   = local_layer ? local_width : width;
            const int layer_columns = layer_width * batch;
            Tensor layer_context    = local_layer && replace_local_window
                                          ? context.slice(1, local_offset, local_width)
                                          : context;
            Tensor layer_positions  = local_layer && replace_local_window
                                          ? positions.slice(0, local_offset, local_width)
                                          : positions;
            auto layer_roots =
                workspace_recipe::dflash_context_layer(state.execution.work, layer_columns, config);
            Tensor key_raw =
                layer_roots.key_raw.view({config.head_dim, config.kv_heads, layer_columns});
            Tensor value =
                layer_roots.value.view({config.head_dim, config.kv_heads, layer_columns});
            Tensor key_flat   = key_raw.view({config.kv_size(), layer_columns});
            Tensor value_flat = value.view({config.kv_size(), layer_columns});
            if (config.kv_size() == 1024 && (config.hidden == 5120 ||
                (config.hidden == 2048 && config.query_size() == 4096))) {
                ops::linear_pair(layer_context, weight.context_key, weight.context_value, key_flat,
                                 value_flat, state.execution.device.stream);
            } else {
                ops::linear(layer_context, weight.context_key, key_flat, state.execution.device.stream);
                ops::linear(layer_context, weight.context_value, value_flat, state.execution.device.stream);
            }
            Tensor key = layer_roots.key.view({config.head_dim, config.kv_heads, layer_columns});
            ops::rmsnorm(key_raw, weight.key_norm, config.rms_epsilon, false, key,
                         state.execution.device.stream);
            ops::rope(layer_positions.view({layer_columns}), config.head_dim, config.rope_theta,
                      key, state.execution.device.stream);
            Tensor key_batch = key.view({config.head_dim, config.kv_heads, layer_width, batch});
            Tensor value_batch =
                value.view({config.head_dim, config.kv_heads, layer_width, batch});
            Tensor position_batch = layer_positions.view({layer_width, batch});
            if (local_layer) {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                    dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                    state.execution.device.stream);
            } else {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                    dflash_state(state).full_batch_layer(0), state.execution.device.stream);
            }
        }
    }
}

template <class V>
void propose_batch_impl(DFlashBatchContext& state, family::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash proposal is unavailable for this target");
    } else {
        const auto& config = state.execution.model.geometry.dflash;
        if (k == 0 || k >= static_cast<std::uint32_t>(config.block_size)) {
            throw std::invalid_argument("DFlash draft extent exceeds checkpoint block_size");
        }
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        Tensor anchors             = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers           = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor lanes               = frame.lanes.slice(0, 0, batch_size);
        Tensor full_rows           = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor ids                 = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions           = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts              = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, config.mask_token, ids,
                                  positions, state.execution.device.stream);
        Tensor residual = state.execution.work.alloc(DType::BF16, {config.hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                       state.execution.device.stream);

        for (int layer = 0; layer < config.layers; ++layer) {
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            {
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace_recipe::dflash_attention(state.execution.work, columns, config);
                ops::rmsnorm(residual, weight.input_norm, config.rms_epsilon, false, roots.hidden,
                             state.execution.device.stream);
                Tensor query_raw =
                    roots.query_raw.view({config.head_dim, config.query_heads, columns});
                Tensor key_raw = roots.key_raw.view({config.head_dim, config.kv_heads, columns});
                Tensor value   = roots.value.view({config.head_dim, config.kv_heads, columns});
                Tensor query_flat = query_raw.view({config.query_size(), columns});
                Tensor key_flat   = key_raw.view({config.kv_size(), columns});
                Tensor value_flat = value.view({config.kv_size(), columns});
                ops::attn_input_proj(roots.hidden, weight.query_key_value, query_flat, key_flat,
                                     value_flat, state.execution.device.stream);
                Tensor query = roots.query.view({config.head_dim, config.query_heads, columns});
                Tensor key   = roots.key.view({config.head_dim, config.kv_heads, columns});
                ops::rmsnorm(query_raw, weight.query_norm, config.rms_epsilon, false, query,
                             state.execution.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, config.rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(positions.view({columns}), config.head_dim, config.rope_theta, query,
                          key, state.execution.device.stream);
                Tensor query_batch =
                    query.view({config.head_dim, config.query_heads, width, batch_size});
                Tensor key_batch =
                    key.view({config.head_dim, config.kv_heads, width, batch_size});
                Tensor value_batch =
                    value.view({config.head_dim, config.kv_heads, width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {config.head_dim, config.query_heads, width, batch_size});
                if (layer < config.local_layers) {
                    ops::swa(query_batch, key_batch, value_batch, positions, valid_columns, lanes,
                             config.attention_scale,
                             dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                             envelopes.local, state.execution.work, attention_batch,
                             state.execution.device.stream);
                } else {
                    ops::bidirectional_gqa_attention(
                        query_batch, key_batch, value_batch, frontiers, valid_columns, full_rows,
                        config.attention_scale, dflash_state(state).full_batch_layer(0),
                        envelopes.full, state.execution.work, attention_batch,
                        state.execution.device.stream);
                }
                ops::linear_add(roots.attention.view({config.query_size(), columns}),
                                weight.attention_output, residual, state.execution.work,
                                state.execution.device.stream);
            }
            {
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash_mlp(state.execution.work, columns, config);
                ops::rmsnorm(residual, weight.post_attention_norm, config.rms_epsilon, false,
                             roots.hidden, state.execution.device.stream);
                ops::linear_swiglu(roots.hidden, weight.gate_up, roots.intermediate,
                                   state.execution.work, state.execution.device.stream);
                ops::linear_add(roots.intermediate, weight.down, residual, state.execution.work,
                                state.execution.device.stream);
            }
        }

        Tensor packed = state.execution.work.alloc(
            DType::BF16, {config.hidden, static_cast<std::int32_t>(k) * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes =
            static_cast<std::size_t>(config.hidden) * static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(config.hidden) * width * element_bytes;
        const auto* source = static_cast<const std::byte*>(residual.data) +
                             static_cast<std::size_t>(config.hidden) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream));
        Tensor proposal_hidden = state.execution.work.alloc(
            DType::BF16, {config.hidden, static_cast<std::int32_t>(k) * batch_size});
        ops::rmsnorm(packed, state.execution.model.dflash->final_norm, config.rms_epsilon, false,
                     proposal_hidden, state.execution.device.stream);
        Tensor flat_drafts = drafts.view({static_cast<std::int32_t>(k) * batch_size});
        if (state.execution.proposal_head == ProposalHead::Full) {
            Tensor logits = state.execution.work.alloc(
                DType::BF16, {state.execution.model.geometry.output_rows,
                              static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                        state.execution.device.stream);
            ops::argmax(logits, flat_drafts, state.execution.model.geometry.token_domain,
                        state.execution.device.stream);
        } else {
            if (!state.execution.model.optimized_proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *state.execution.model.optimized_proposal;
            Tensor logits        = state.execution.work.alloc(
                DType::BF16, {state.execution.model.geometry.draft_vocab, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, proposal.head, logits, state.execution.device.stream);
            ops::argmax(logits, flat_drafts, state.execution.model.geometry.draft_vocab, state.execution.device.stream);
            ops::proposal_remap_token_ids(flat_drafts,
                                          static_cast<const std::int32_t*>(proposal.token_ids.data),
                                          state.execution.model.geometry.draft_vocab, state.execution.device.stream);
        }
        state.execution.work.reset();
    }
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DFlashEnvelopes envelopes,
                              ops::GqaExecutionEnvelope target_envelope) {
    return [&state, batch_size, k, envelopes, target_envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumBatchColumns) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        family::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(family::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        Tensor anchors          = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers        = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts   = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents          = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns    = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows        = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows      = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes            = frame.lanes.slice(0, 0, batch_size);
        Tensor append_positions = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts    = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts           = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor verify_ids       = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor target_tokens    = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits    = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden    = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden  = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens  = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts  = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted         = frame.accepted_drafts.slice(0, 0, batch_size);

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {state.execution.model.geometry.dflash.feature_rows, width, batch_size});
        ops::prepare_ragged_prefix(dflash_state(state).pending_features, lanes, context_starts,
                                   frontiers, compact_features, append_positions, append_counts,
                                   state.execution.device.stream);
        append_context_impl<Variant>(state, compact_features, append_positions, append_counts,
                                     lanes, dflash_rows, envelopes.append);

        const auto& stage = state.execution.stage;
        const auto metadata = stage.residual_bytes +
            static_cast<std::size_t>(state.execution.model.geometry.dflash.feature_rows) * sizeof(std::uint16_t);
        const auto stream = state.execution.device.stream;
        if (stage.features && stage.first > 0) {
            const auto* packet = static_cast<const std::byte*>(stage.import_pinned);
            CUDA_CHECK(cudaMemcpy2DAsync(verify_ids.data, sizeof(std::int32_t), packet + metadata,
                stage.column_bytes, sizeof(std::int32_t), width * batch_size, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpy2DAsync(target_positions.data, sizeof(std::int32_t),
                packet + metadata + sizeof(std::int32_t), stage.column_bytes, sizeof(std::int32_t),
                width * batch_size, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpy2DAsync(drafts.data, drafts.nb[1],
                static_cast<const std::int32_t*>(verify_ids.data) + 1, verify_ids.nb[1],
                k * sizeof(std::int32_t), batch_size, cudaMemcpyDeviceToDevice, stream));
        } else {
            propose_batch_impl<Variant>(state, frame, batch_size, k, envelopes);
            ops::speculative_prepare_verify_ids(anchors, drafts, extents, verify_ids, stream);
        }
        if (stage.features) {
            auto* packet = static_cast<std::byte*>(stage.export_pinned);
            CUDA_CHECK(cudaMemcpy2DAsync(packet + metadata, stage.column_bytes,
                verify_ids.data, sizeof(std::int32_t), sizeof(std::int32_t), width * batch_size,
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpy2DAsync(packet + metadata + sizeof(std::int32_t), stage.column_bytes,
                target_positions.data, sizeof(std::int32_t), sizeof(std::int32_t), width * batch_size,
                cudaMemcpyDeviceToHost, stream));
        }

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        card.set_ple_state(state.execution.ple);
        card.set_stage(state.execution.stage);
        DFlashFeatureSink sink =
            batch_feature_sink_impl<Variant>(state, lanes, valid_columns, width, batch_size);
        target_verify_accept(state.execution, state.continuation_hidden_store, card,
                             TargetVerifyFrameView{
                                 .ids             = verify_ids,
                                 .cache_positions = target_positions,
                                 .rope_positions  = target_positions,
                                 .valid_columns   = valid_columns,
                                 .kv_table_rows   = text_rows,
                                 .lanes           = lanes,
                                 .target_hidden   = target_hidden,
                                 .target_logits   = target_logits,
                                 .target_tokens   = target_tokens,
                                 .drafts          = drafts,
                                 .current_extents = extents,
                                 .frontiers       = frontiers,
                                 .anchors         = anchors,
                                 .licensed_tokens = licensed_tokens,
                                 .licensed_counts = licensed_counts,
                                 .accepted_drafts = accepted,
                                 .selected_hidden = selected_hidden,
                                 .replay_records  = state.execution.replay_records,
                                 .sampling        = frame.sampling,
                                 .feature_sink    = &sink,
                             },
                             target_envelope);
        auto* scores = reinterpret_cast<RawTokenScores*>(static_cast<std::byte*>(frame.egress.data) +
            offsetof(family::DFlashDecodeEgress, scores));
        ops::score_logprobs_device(target_logits, licensed_tokens, state.execution.model.geometry.token_domain,
            frame.sampling, scores, state.execution.device.stream, width, static_cast<const int*>(licensed_counts.data));
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   offsetof(family::DFlashDecodeEgress, scores) + batch_size * width * sizeof(RawTokenScores), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    capture_graph(state, definition, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope);
    run_prepared(state, executable, body);
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
