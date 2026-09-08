#pragma once

// An encoder, not a decoder.
//
// EmbeddingGemma runs one forward pass and returns a vector. There is no KV
// cache to fill, no token to sample, no second round to make fast -- so this
// does not go through the engine's Program, executor or paged cache, which
// exist for exactly those things. It reads the artifact, binds the weights, and
// runs the graph. llama.cpp reaches the same conclusion from the other side:
// its LLM_ARCH_GEMMA_EMBEDDING allocates no KV cache at all.
//
// What it does share is everything below the round loop: the artifact reader,
// the binder, and the op library.

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include <api/family/text_geometry.h>
#include <algorithm>
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "encoder/gemma_tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace sinfer::encoder {

/// Encoder dimensions and execution settings resolved from the artifact metadata.
struct GemmaEmbeddingConfig {
    std::int32_t layers = 0;
    std::int32_t hidden = 0;
    std::int32_t query_heads = 0;
    std::int32_t head_dim = 0;
    std::int32_t intermediate = 0;
    std::int32_t vocab = 0;
    std::int32_t max_tokens = 0;
    /// A batching policy; increased when a single supported sequence needs more room.
    std::int32_t max_batch_tokens = 8192;
    float rms_epsilon = 0.0F;
    std::int32_t sliding_window = 0;
    float rope_theta_global = 0.0F;
    float rope_theta_local = 0.0F;
    float attention_scale = 0.0F;
    float embedding_scale = 0.0F;
    std::vector<bool> global_layers;

    [[nodiscard]] bool is_global(std::int32_t layer) const {
        return global_layers.at(static_cast<std::size_t>(layer));
    }
    [[nodiscard]] std::int32_t query_size() const noexcept { return query_heads * head_dim; }

    [[nodiscard]] static GemmaEmbeddingConfig from_artifact(const artifact::Reader& reader) {
        if (reader.identity().architecture != "gemma_embedding") {
            throw std::invalid_argument("expected a gemma_embedding artifact; rebuild the serving cache");
        }
        const auto g = family::TextGeometry::resolved(reader.geometry(), reader.layer_types());
        // Both encoder attention implementations currently implement multi-query attention.
        // A backend restriction is checked against the checkpoint, never used as a default.
        if (g.kv_heads != 1) {
            throw std::invalid_argument("the embedding backend requires exactly one key/value head");
        }
        if (g.embedding_scale <= 0.0F || g.sliding_rope_theta <= 0.0F) {
            throw std::invalid_argument("embedding metadata requires embedding_scale and sliding_rope_theta");
        }
        GemmaEmbeddingConfig config;
        config.layers = g.layers;
        config.hidden = g.hidden;
        config.query_heads = g.query_heads;
        config.head_dim = g.head_dim;
        config.intermediate = g.intermediate;
        config.vocab = g.output_rows;
        config.max_tokens = g.max_context;
        config.max_batch_tokens = std::max(config.max_batch_tokens, config.max_tokens);
        config.rms_epsilon = g.rms_epsilon;
        config.sliding_window = g.sliding_window;
        config.rope_theta_global = g.rope_theta;
        config.rope_theta_local = g.sliding_rope_theta;
        config.attention_scale = g.attention_scale;
        config.embedding_scale = g.embedding_scale;
        for (std::int32_t layer = 0; layer < g.layers; ++layer) {
            if (!g.layer_attends(layer)) {
                throw std::invalid_argument("the embedding backend requires attention at every layer");
            }
            config.global_layers.push_back(!g.layer_is_windowed(layer));
        }
        return config;
    }
};

class GemmaEmbedding {
public:
    /// Binds every object the artifact holds and uploads it. Throws if the
    /// artifact carries an object this target does not consume, or omits one it
    /// requires -- the same contract the engine's targets keep.
    static GemmaEmbedding load(const std::filesystem::path& artifact, DeviceContext& device);

    ~GemmaEmbedding();
    GemmaEmbedding(GemmaEmbedding&&) noexcept;
    GemmaEmbedding& operator=(GemmaEmbedding&&) noexcept;
    GemmaEmbedding(const GemmaEmbedding&)            = delete;
    GemmaEmbedding& operator=(const GemmaEmbedding&) = delete;

    /// One sequence in, one L2-normalised vector out. `tokens` must be non-empty
    /// and no longer than `max_tokens`; the task prefix, if any, is already part
    /// of it, because EmbeddingGemma pools the prefix along with the text.
    [[nodiscard]] std::vector<float> embed(std::span<const std::int32_t> tokens);

    /// Several sequences through one forward.
    ///
    /// The sequences are concatenated, so every projection sees one wide matrix
    /// and batches for free -- and the projections are where the work is: at 512
    /// tokens they are 108 GFLOP against attention's 19. Only attention is
    /// per-sequence, because one sequence must not attend to the next; it runs
    /// as a loop over column slices, which costs a few launches and no padding.
    ///
    /// Sequences may differ in length and there is no limit on how many: a
    /// request larger than `max_batch_tokens` is split into forwards that fit,
    /// which keeps the scratch arena a fixed size rather than a function of
    /// whatever arrived.
    [[nodiscard]] std::vector<std::vector<float>> embed_batch(
        const std::vector<std::vector<std::int32_t>>& sequences);

    /// The tokenizer the artifact ships. Text and ids therefore cannot disagree:
    /// they come from the same file the weights did.
    [[nodiscard]] const GemmaTokenizer& tokenizer() const noexcept;

    [[nodiscard]] const GemmaEmbeddingConfig& config() const noexcept;
    /// Bytes of device memory the weights occupy.
    [[nodiscard]] std::uint64_t weight_bytes() const noexcept;

private:
    GemmaEmbedding();
    /// One forward. Its sequences must already fit `max_batch_tokens`.
    [[nodiscard]] std::vector<std::vector<float>> embed_chunk(
        const std::vector<std::vector<std::int32_t>>& sequences);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sinfer::encoder
