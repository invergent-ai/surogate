#include "encoder/gemma_embedding.h"

#include "api/ops/cast.h"
#include "api/ops/embedding.h"
#include "api/ops/encoder_attention.h"
#include "api/ops/gelu.h"
#include "api/ops/gelu_mul.h"
#include "api/ops/l2norm.h"
#include "api/ops/linear.h"
#include "api/ops/mean_pool.h"
#include "api/ops/residual_add.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/rope.h"
#include "api/ops/scale.h"
#include "artifact/binder.h"
#include "encoder/gemma_tokenizer.h"
#include "artifact/typed_binding.h"
#include "ops/linear/bf16/bf16_cublaslt.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::encoder {
namespace {

using artifact::NumericFormat;

std::string layer_object(std::int32_t layer, const char* leaf) {
    return "text/layers/" + std::to_string(layer) + "/" + leaf;
}

struct LayerWeights {
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle post_attention_norm;
    artifact::ObjectHandle pre_feedforward_norm;
    artifact::ObjectHandle post_feedforward_norm;
    artifact::ObjectHandle query;
    artifact::ObjectHandle key;
    artifact::ObjectHandle value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
    artifact::ObjectHandle gate;
    artifact::ObjectHandle up;
    artifact::ObjectHandle down;
};

} // namespace

struct GemmaEmbedding::Impl {
    GemmaEmbeddingConfig config;
    DeviceContext* device = nullptr;

    std::unique_ptr<artifact::Reader> reader;
    artifact::MaterializedArtifact materialized;

    artifact::ObjectHandle token_embedding;
    artifact::ObjectHandle final_norm;
    artifact::ObjectHandle embedding_head;
    std::vector<LayerWeights> layers;

    // Scratch. Sized once for max_tokens and reused; an encoder request is one
    // forward, so nothing has to survive between calls.
    std::unique_ptr<DeviceArena> arena;
    std::unique_ptr<DeviceBuffer> attention_workspace;
    std::size_t attention_workspace_bytes = 0;
    std::unique_ptr<DeviceBuffer> positions;
    std::unique_ptr<GemmaTokenizer> tokenizer;

    [[nodiscard]] Tensor norm(artifact::ObjectHandle handle, std::int32_t width) const {
        return artifact::materialized_tensor(materialized, handle, NumericFormat::BF16, {width});
    }
    [[nodiscard]] Weight matrix(artifact::ObjectHandle handle, std::int32_t rows,
                                std::int32_t columns) const {
        return artifact::materialized_weight(materialized, handle, NumericFormat::W8G32_F16S, rows,
                                             columns);
    }
};

GemmaEmbedding::GemmaEmbedding() : impl_(std::make_unique<Impl>()) {}
GemmaEmbedding::~GemmaEmbedding()                                  = default;
GemmaEmbedding::GemmaEmbedding(GemmaEmbedding&&) noexcept          = default;
GemmaEmbedding& GemmaEmbedding::operator=(GemmaEmbedding&&) noexcept = default;

const GemmaEmbeddingConfig& GemmaEmbedding::config() const noexcept { return impl_->config; }

const GemmaTokenizer& GemmaEmbedding::tokenizer() const noexcept { return *impl_->tokenizer; }

std::uint64_t GemmaEmbedding::weight_bytes() const noexcept {
    return impl_->materialized.stats().device_capacity_bytes;
}

GemmaEmbedding GemmaEmbedding::load(const std::filesystem::path& path, DeviceContext& device) {
    GemmaEmbedding model;
    Impl& impl = *model.impl_;
    impl.device = &device;
    impl.reader = std::make_unique<artifact::Reader>(path);

    const GemmaEmbeddingConfig& config = impl.config;
    const auto hidden       = static_cast<std::uint64_t>(config.hidden);
    const auto head_dim     = static_cast<std::uint64_t>(config.head_dim);
    const auto query_size   = static_cast<std::uint64_t>(config.query_size());
    const auto intermediate = static_cast<std::uint64_t>(config.intermediate);
    const auto vocab        = static_cast<std::uint64_t>(config.vocab);

    artifact::Binder binder(*impl.reader);
    const auto w8 = NumericFormat::W8G32_F16S;
    const auto bf = NumericFormat::BF16;

    impl.token_embedding = artifact::bind_device_tensor(binder, "text/token_embedding", w8,
                                                        {vocab, hidden});
    impl.layers.resize(static_cast<std::size_t>(config.layers));
    for (std::int32_t layer = 0; layer < config.layers; ++layer) {
        LayerWeights& weights = impl.layers[static_cast<std::size_t>(layer)];
        const auto bind_norm  = [&](const char* leaf, std::uint64_t width) {
            return artifact::bind_device_tensor(binder, layer_object(layer, leaf), bf, {width});
        };
        const auto bind_matrix = [&](const char* leaf, std::uint64_t rows, std::uint64_t columns) {
            return artifact::bind_device_tensor(binder, layer_object(layer, leaf), w8,
                                                {rows, columns});
        };
        weights.input_norm            = bind_norm("input_norm", hidden);
        weights.post_attention_norm   = bind_norm("post_attention_norm", hidden);
        weights.pre_feedforward_norm  = bind_norm("pre_feedforward_norm", hidden);
        weights.post_feedforward_norm = bind_norm("post_feedforward_norm", hidden);
        weights.query                 = bind_matrix("attention/query", query_size, hidden);
        weights.key                   = bind_matrix("attention/key", head_dim, hidden);
        weights.value                 = bind_matrix("attention/value", head_dim, hidden);
        weights.query_norm            = bind_norm("attention/query_norm", head_dim);
        weights.key_norm              = bind_norm("attention/key_norm", head_dim);
        weights.output                = bind_matrix("attention/output", hidden, query_size);
        weights.gate                  = bind_matrix("mlp/gate", intermediate, hidden);
        weights.up                    = bind_matrix("mlp/up", intermediate, hidden);
        weights.down                  = bind_matrix("mlp/down", hidden, intermediate);
    }
    impl.final_norm     = artifact::bind_device_tensor(binder, "text/final_norm", bf, {hidden});
    impl.embedding_head = artifact::bind_device_tensor(binder, "text/embedding_head", bf,
                                                       {hidden, hidden});
    const auto tokenizer_model = artifact::bind_raw_resource(binder, "frontend/tokenizer.model");
    (void)artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json");

    impl.materialized = artifact::materialize(*impl.reader, binder.finish(), device);
    impl.tokenizer = std::make_unique<GemmaTokenizer>(GemmaTokenizer::from_serialized_proto(
        impl.materialized.resource_bytes(tokenizer_model)));

    // Scratch for the widest request. Named for what they hold rather than
    // sized by trial: hidden-wide activations, one intermediate-wide pair for
    // the MLP, and q/k/v.
    const auto tokens  = static_cast<std::size_t>(config.max_batch_tokens);
    const std::size_t bf16 = sizeof(std::uint16_t);
    const std::size_t arena_bytes =
        bf16 * tokens *
            (6 * static_cast<std::size_t>(config.hidden) +      // x, h, attn_out, mlp_out, spare
             2 * static_cast<std::size_t>(config.intermediate) + // gate, up
             static_cast<std::size_t>(config.query_size()) + 2 * static_cast<std::size_t>(config.head_dim)) +
        // pooled (FP32) and the head's output
        sizeof(float) * static_cast<std::size_t>(config.hidden) +
        bf16 * 2 * static_cast<std::size_t>(config.hidden) + (1u << 20);
    impl.arena = std::make_unique<DeviceArena>(arena_bytes);

    // Attention is per-sequence, so the score matrix is sized by the longest
    // single sequence rather than by the batch.
    impl.attention_workspace_bytes =
        ops::encoder_attention_workspace_bytes(config.query_heads, config.max_tokens);
    impl.attention_workspace = std::make_unique<DeviceBuffer>(impl.attention_workspace_bytes);
    impl.positions = std::make_unique<DeviceBuffer>(tokens * sizeof(std::int32_t));

    ops::encoder_attention_prewarm();
    ops::detail::bf16_cublaslt_prewarm();
    ops::detail::bf16_cublaslt_prepare(config.hidden, config.hidden, 1);
    return model;
}

std::vector<float> GemmaEmbedding::embed(std::span<const std::int32_t> tokens) {
    std::vector<std::vector<std::int32_t>> one{
        std::vector<std::int32_t>(tokens.begin(), tokens.end())};
    return embed_batch(one).front();
}

std::vector<std::vector<float>> GemmaEmbedding::embed_batch(
    const std::vector<std::vector<std::int32_t>>& sequences) {
    const std::int32_t budget = impl_->config.max_batch_tokens;
    std::vector<std::vector<float>> out;
    out.reserve(sequences.size());

    // Greedy partition: fill a forward until the next sequence would not fit.
    // A single sequence longer than the budget still goes through on its own and
    // is rejected there against max_tokens, which is the limit that means
    // something to the model.
    std::vector<std::vector<std::int32_t>> chunk;
    std::int32_t chunk_tokens = 0;
    const auto flush = [&]() {
        if (chunk.empty()) { return; }
        std::vector<std::vector<float>> vectors = embed_chunk(chunk);
        out.insert(out.end(), std::make_move_iterator(vectors.begin()),
                   std::make_move_iterator(vectors.end()));
        chunk.clear();
        chunk_tokens = 0;
    };
    for (const std::vector<std::int32_t>& sequence : sequences) {
        const auto length = static_cast<std::int32_t>(sequence.size());
        if (!chunk.empty() && chunk_tokens + length > budget) { flush(); }
        chunk.push_back(sequence);
        chunk_tokens += length;
    }
    flush();
    return out;
}

std::vector<std::vector<float>> GemmaEmbedding::embed_chunk(
    const std::vector<std::vector<std::int32_t>>& sequences) {
    Impl& impl                         = *impl_;
    const GemmaEmbeddingConfig& config = impl.config;
    if (sequences.empty()) { return {}; }

    std::vector<std::int32_t> flat;
    std::vector<std::int32_t> offsets;
    std::vector<std::int32_t> lengths;
    std::vector<std::int32_t> positions;
    offsets.reserve(sequences.size());
    lengths.reserve(sequences.size());
    for (const std::vector<std::int32_t>& sequence : sequences) {
        const auto length = static_cast<std::int32_t>(sequence.size());
        if (length <= 0) { throw std::invalid_argument("embed: a sequence is empty"); }
        if (length > config.max_tokens) {
            throw std::invalid_argument("embed: " + std::to_string(length) + " tokens exceeds " +
                                        std::to_string(config.max_tokens));
        }
        offsets.push_back(static_cast<std::int32_t>(flat.size()));
        lengths.push_back(length);
        flat.insert(flat.end(), sequence.begin(), sequence.end());
        // Positions restart per sequence: each is its own document, not a
        // continuation of the one before it.
        for (std::int32_t i = 0; i < length; ++i) { positions.push_back(i); }
    }
    const auto total = static_cast<std::int32_t>(flat.size());
    if (total > config.max_batch_tokens) {
        throw std::invalid_argument("embed: batch of " + std::to_string(total) +
                                    " tokens exceeds max_batch_tokens " +
                                    std::to_string(config.max_batch_tokens));
    }
    const auto batch    = static_cast<std::int32_t>(sequences.size());
    cudaStream_t stream = impl.device->stream;

    DeviceArena& arena = *impl.arena;
    arena.reset();

    CUDA_CHECK(cudaMemcpyAsync(impl.positions->p, positions.data(),
                               positions.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               stream));
    DeviceBuffer ids(static_cast<std::size_t>(total) * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpyAsync(ids.p, flat.data(),
                               static_cast<std::size_t>(total) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    const Tensor id_tensor(ids.p, DType::I32, {total});
    const Tensor position_tensor(impl.positions->p, DType::I32, {total});

    Tensor x        = arena.alloc(DType::BF16, {config.hidden, total});
    Tensor h        = arena.alloc(DType::BF16, {config.hidden, total});
    Tensor attn     = arena.alloc(DType::BF16, {config.hidden, total});
    Tensor query    = arena.alloc(DType::BF16, {config.query_size(), total});
    Tensor key      = arena.alloc(DType::BF16, {config.head_dim, total});
    Tensor value    = arena.alloc(DType::BF16, {config.head_dim, total});
    Tensor attn_out = arena.alloc(DType::BF16, {config.query_size(), total});
    Tensor gate     = arena.alloc(DType::BF16, {config.intermediate, total});
    Tensor up       = arena.alloc(DType::BF16, {config.intermediate, total});

    ops::embedding(id_tensor, impl.matrix(impl.token_embedding, config.vocab, config.hidden), x,
                   stream);
    // Gemma scales the table's output by sqrt(hidden). The norms divide it out
    // again, but the residual stream carries it, so it sets the size of x
    // relative to every block's contribution.
    ops::scale(x, config.embedding_scale, stream);

    for (std::int32_t layer = 0; layer < config.layers; ++layer) {
        const LayerWeights& w     = impl.layers[static_cast<std::size_t>(layer)];
        const bool global         = config.is_global(layer);
        const float theta         = global ? config.rope_theta_global : config.rope_theta_local;
        const std::int32_t window = global ? 0 : config.sliding_window;

        // --- attention, between the sandwich norms -------------------------
        // Every projection runs once over the whole batch: the sequences are
        // adjacent columns and a GEMM does not care where one ends.
        ops::rmsnorm(x, impl.norm(w.input_norm, config.hidden), config.rms_epsilon,
                     /*unit_offset*/ true, h, stream);
        ops::linear(h, impl.matrix(w.query, config.query_size(), config.hidden), query, stream);
        ops::linear(h, impl.matrix(w.key, config.head_dim, config.hidden), key, stream);
        ops::linear(h, impl.matrix(w.value, config.head_dim, config.hidden), value, stream);

        // Per-head QK norm: each head's features are contiguous, so the
        // [head_dim, heads * tokens] view is exactly the rows rmsnorm reduces
        // over -- and it does not care about sequence boundaries either.
        Tensor query_heads(query.data, DType::BF16, {config.head_dim, config.query_heads * total});
        Tensor key_heads(key.data, DType::BF16, {config.head_dim, total});
        ops::rmsnorm(query_heads, impl.norm(w.query_norm, config.head_dim), config.rms_epsilon,
                     true, query_heads, stream);
        ops::rmsnorm(key_heads, impl.norm(w.key_norm, config.head_dim), config.rms_epsilon, true,
                     key_heads, stream);

        // rope reads its position per column, and positions restart per
        // sequence, so this too runs once for the batch.
        Tensor query_rope(query.data, DType::BF16, {config.head_dim, config.query_heads, total});
        Tensor key_rope(key.data, DType::BF16, {config.head_dim, 1, total});
        ops::rope(position_tensor, config.head_dim, theta, query_rope, key_rope, stream);

        // Attention is the one step that must not cross a boundary.
        for (std::int32_t index = 0; index < batch; ++index) {
            const std::int32_t offset = offsets[static_cast<std::size_t>(index)];
            const std::int32_t length = lengths[static_cast<std::size_t>(index)];
            const Tensor q_slice      = query.slice(1, offset, length);
            const Tensor k_slice      = key.slice(1, offset, length);
            const Tensor v_slice      = value.slice(1, offset, length);
            Tensor out_slice          = attn_out.slice(1, offset, length);
            ops::encoder_attention(q_slice, k_slice, v_slice, window, config.attention_scale,
                                   out_slice, impl.attention_workspace->p,
                                   impl.attention_workspace_bytes, stream);
        }

        ops::linear(attn_out, impl.matrix(w.output, config.hidden, config.query_size()), attn,
                    stream);
        ops::rmsnorm(attn, impl.norm(w.post_attention_norm, config.hidden), config.rms_epsilon,
                     true, attn, stream);
        ops::residual_add(attn, x, stream); // x += attn

        // --- MLP, between the other two ------------------------------------
        ops::rmsnorm(x, impl.norm(w.pre_feedforward_norm, config.hidden), config.rms_epsilon, true,
                     h, stream);
        ops::linear(h, impl.matrix(w.gate, config.intermediate, config.hidden), gate, stream);
        ops::linear(h, impl.matrix(w.up, config.intermediate, config.hidden), up, stream);
        // gelu_pytorch_tanh, per the checkpoint's hidden_activation.
        ops::gelu_mul(gate, up, ops::GeluMode::Tanh, gate, stream);
        ops::linear(gate, impl.matrix(w.down, config.hidden, config.intermediate), attn, stream);
        ops::rmsnorm(attn, impl.norm(w.post_feedforward_norm, config.hidden), config.rms_epsilon,
                     true, attn, stream);
        ops::residual_add(attn, x, stream);
    }

    ops::rmsnorm(x, impl.norm(impl.final_norm, config.hidden), config.rms_epsilon, true, h,
                 stream);

    // --- pool, project, normalise -------------------------------------------
    // Pooling is per sequence; the projection that follows is one GEMM over all
    // of them, since by then each sequence is a single column.
    Tensor pooled_f32 = arena.alloc(DType::FP32, {config.hidden, batch});
    Tensor pooled     = arena.alloc(DType::BF16, {config.hidden, batch});
    for (std::int32_t index = 0; index < batch; ++index) {
        const std::int32_t offset = offsets[static_cast<std::size_t>(index)];
        const std::int32_t length = lengths[static_cast<std::size_t>(index)];
        const Tensor columns      = h.slice(1, offset, length);
        Tensor slot               = pooled_f32.slice(1, index, 1);
        Tensor slot_1d(slot.data, DType::FP32, {config.hidden});
        ops::mean_pool(columns, length, /*accumulate*/ false, slot_1d, stream);
    }
    ops::cast_fp32_to_bf16(pooled_f32, pooled, stream);

    // The head is one [hidden, hidden] matrix: the checkpoint's two Dense modules
    // composed at conversion, which they may be because both declare Identity.
    Tensor projected = arena.alloc(DType::BF16, {config.hidden, batch});
    ops::detail::bf16_cublaslt_prepare(config.hidden, config.hidden, batch);
    ops::detail::bf16_cublaslt_gemm(
        artifact::materialized_weight(impl.materialized, impl.embedding_head, NumericFormat::BF16,
                                      config.hidden, config.hidden),
        pooled, projected, stream);
    // l2norm reduces over ne[0], which is the hidden axis: one call for all of them.
    ops::l2norm(projected, 1.0e-12F, projected, stream);

    const auto elements = static_cast<std::size_t>(config.hidden) * batch;
    std::vector<std::uint16_t> host(elements);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), projected.data, host.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::vector<float>> out(static_cast<std::size_t>(batch));
    for (std::int32_t index = 0; index < batch; ++index) {
        std::vector<float>& vector = out[static_cast<std::size_t>(index)];
        vector.resize(static_cast<std::size_t>(config.hidden));
        for (std::int32_t d = 0; d < config.hidden; ++d) {
            const std::uint32_t word =
                static_cast<std::uint32_t>(
                    host[static_cast<std::size_t>(index) * config.hidden + d])
                << 16U;
            float value = 0.0F;
            std::memcpy(&value, &word, sizeof(value));
            vector[static_cast<std::size_t>(d)] = value;
        }
    }
    return out;
}

} // namespace sinfer::encoder
