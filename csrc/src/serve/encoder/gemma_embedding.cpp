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
    // The tokenizer travels with the artifact but is the frontend's business.
    (void)artifact::bind_raw_resource(binder, "frontend/tokenizer.json");
    (void)artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json");

    impl.materialized = artifact::materialize(*impl.reader, binder.finish(), device);

    // Scratch for the widest request. Named for what they hold rather than
    // sized by trial: hidden-wide activations, one intermediate-wide pair for
    // the MLP, and q/k/v.
    const auto tokens  = static_cast<std::size_t>(config.max_tokens);
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
    Impl& impl                        = *impl_;
    const GemmaEmbeddingConfig& config = impl.config;
    const auto count                  = static_cast<std::int32_t>(tokens.size());
    if (count <= 0) { throw std::invalid_argument("embed: needs at least one token"); }
    if (count > config.max_tokens) {
        throw std::invalid_argument("embed: " + std::to_string(count) + " tokens exceeds " +
                                    std::to_string(config.max_tokens));
    }
    cudaStream_t stream = impl.device->stream;

    // Positions are 0..count-1; an encoder never resumes, so they are never
    // anything else.
    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) { host_positions[static_cast<std::size_t>(i)] = i; }
    CUDA_CHECK(cudaMemcpyAsync(impl.positions->p, host_positions.data(),
                               host_positions.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));

    DeviceArena& arena = *impl.arena;
    arena.reset();

    DeviceBuffer ids(static_cast<std::size_t>(count) * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpyAsync(ids.p, tokens.data(),
                               static_cast<std::size_t>(count) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    const Tensor id_tensor(ids.p, DType::I32, {count});
    const Tensor position_tensor(impl.positions->p, DType::I32, {count});

    Tensor x        = arena.alloc(DType::BF16, {config.hidden, count});
    Tensor h        = arena.alloc(DType::BF16, {config.hidden, count});
    Tensor attn     = arena.alloc(DType::BF16, {config.hidden, count});
    Tensor query    = arena.alloc(DType::BF16, {config.query_size(), count});
    Tensor key      = arena.alloc(DType::BF16, {config.head_dim, count});
    Tensor value    = arena.alloc(DType::BF16, {config.head_dim, count});
    Tensor attn_out = arena.alloc(DType::BF16, {config.query_size(), count});
    Tensor gate     = arena.alloc(DType::BF16, {config.intermediate, count});
    Tensor up       = arena.alloc(DType::BF16, {config.intermediate, count});

    ops::embedding(id_tensor, impl.matrix(impl.token_embedding, config.vocab, config.hidden), x,
                   stream);
    // Gemma scales the table's output by sqrt(hidden). The norms divide it out
    // again, but the residual stream carries it, so it sets the size of x
    // relative to every block's contribution.
    ops::scale(x, config.embedding_scale, stream);

    for (std::int32_t layer = 0; layer < config.layers; ++layer) {
        const LayerWeights& w = impl.layers[static_cast<std::size_t>(layer)];
        const bool global     = config.is_global(layer);
        const float theta     = global ? config.rope_theta_global : config.rope_theta_local;
        const std::int32_t window = global ? 0 : config.sliding_window;

        // --- attention, between the sandwich norms -------------------------
        ops::rmsnorm(x, impl.norm(w.input_norm, config.hidden), config.rms_epsilon,
                     /*unit_offset*/ true, h, stream);
        ops::linear(h, impl.matrix(w.query, config.query_size(), config.hidden), query, stream);
        ops::linear(h, impl.matrix(w.key, config.head_dim, config.hidden), key, stream);
        ops::linear(h, impl.matrix(w.value, config.head_dim, config.hidden), value, stream);

        // Per-head QK norm: each head's features are contiguous, so the [head_dim,
        // heads * tokens] view is exactly the rows rmsnorm reduces over.
        Tensor query_heads(query.data, DType::BF16, {config.head_dim, config.query_heads * count});
        Tensor key_heads(key.data, DType::BF16, {config.head_dim, count});
        ops::rmsnorm(query_heads, impl.norm(w.query_norm, config.head_dim), config.rms_epsilon,
                     true, query_heads, stream);
        ops::rmsnorm(key_heads, impl.norm(w.key_norm, config.head_dim), config.rms_epsilon, true,
                     key_heads, stream);

        // rope wants [head_dim, heads, tokens]; Gemma 3 rotates the whole head.
        Tensor query_rope(query.data, DType::BF16, {config.head_dim, config.query_heads, count});
        Tensor key_rope(key.data, DType::BF16, {config.head_dim, 1, count});
        ops::rope(position_tensor, config.head_dim, theta, query_rope, key_rope, stream);

        ops::encoder_attention(query, key, value, window, config.attention_scale, attn_out,
                               impl.attention_workspace->p, impl.attention_workspace_bytes,
                               stream);
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
    Tensor pooled_f32 = arena.alloc(DType::FP32, {config.hidden});
    ops::mean_pool(h, count, /*accumulate*/ false, pooled_f32, stream);

    // The head is one [hidden, hidden] matrix: the checkpoint's two Dense modules
    // composed at conversion, which they may be because both declare Identity.
    Tensor pooled = arena.alloc(DType::BF16, {config.hidden, 1});
    {
        const Tensor source(pooled_f32.data, DType::FP32, {config.hidden, 1});
        ops::cast_fp32_to_bf16(source, pooled, stream);
    }
    Tensor projected = arena.alloc(DType::BF16, {config.hidden, 1});
    ops::detail::bf16_cublaslt_gemm(
        artifact::materialized_weight(impl.materialized, impl.embedding_head, NumericFormat::BF16,
                                      config.hidden, config.hidden),
        pooled, projected, stream);
    // Smallest eps that keeps 1/sqrt(sum + eps) finite for a zero vector without
    // perturbing a real one: the pooled vector has norm order 1.
    ops::l2norm(projected, 1.0e-12F, projected, stream);

    std::vector<std::uint16_t> host(static_cast<std::size_t>(config.hidden));
    CUDA_CHECK(cudaMemcpyAsync(host.data(), projected.data, host.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> out(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
        const std::uint32_t word = static_cast<std::uint32_t>(host[i]) << 16U;
        float value              = 0.0F;
        std::memcpy(&value, &word, sizeof(value));
        out[i] = value;
    }
    return out;
}

} // namespace sinfer::encoder
