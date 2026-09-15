#pragma once

// CPU/GPU text embedding execution shares model metadata and the serving op library.
// EmbeddingGemma uses bidirectional attention, mean pooling and a projection head.
// Harrier's Qwen3/Gemma3 backbones use causal attention and last-token pooling.
// All return one normalized vector per input; there is no autoregressive decode loop.

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include <api/family/text_geometry.h>
#include <algorithm>
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "encoder/embedding_tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace sinfer::encoder {

/// Encoder dimensions and execution settings resolved from the artifact metadata.
struct TextEmbeddingConfig {
    bool gemma = true;
    bool mean_pooling = true;
    float rope_frequency_scale = 1.0F;
    std::int32_t layers = 0;
    std::int32_t hidden = 0;
    std::int32_t query_heads = 0;
    std::int32_t kv_heads = 0;
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
    [[nodiscard]] std::int32_t kv_size() const noexcept { return kv_heads * head_dim; }

    [[nodiscard]] static TextEmbeddingConfig from_artifact(const artifact::Reader& reader) {
        const auto& architecture = reader.identity().architecture;
        const bool gemma = architecture == "gemma_embedding" || architecture == "gemma3_embedding";
        if (!gemma && architecture != "qwen3_embedding") {
            throw std::invalid_argument("expected an embedding artifact; rebuild the serving cache");
        }
        const auto g = family::TextGeometry::resolved(reader.geometry(), reader.layer_types());
        // The original EmbeddingGemma architecture has one KV head. Harrier also uses GQA.
        if (architecture == "gemma_embedding" && g.kv_heads != 1) {
            throw std::invalid_argument("EmbeddingGemma requires exactly one key/value head");
        }
        if (gemma && (g.embedding_scale <= 0.0F || g.sliding_rope_theta <= 0.0F)) {
            throw std::invalid_argument("embedding metadata requires embedding_scale and sliding_rope_theta");
        }
        TextEmbeddingConfig config;
        config.gemma = gemma;
        config.mean_pooling = architecture == "gemma_embedding";
        if (const auto it = reader.geometry().find("rope_frequency_scale"); it != reader.geometry().end()) {
            config.rope_frequency_scale = static_cast<float>(it->second);
            if (!(config.rope_frequency_scale > 0.0F && config.rope_frequency_scale <= 1.0F)) {
                throw std::invalid_argument("invalid embedding RoPE frequency scale");
            }
        }
        config.layers = g.layers;
        config.hidden = g.hidden;
        config.query_heads = g.query_heads;
        config.kv_heads = g.kv_heads;
        config.head_dim = g.head_dim;
        config.intermediate = g.intermediate;
        config.vocab = g.output_rows;
        config.max_tokens = g.max_context;
        config.max_batch_tokens = std::max(config.max_batch_tokens, config.max_tokens);
        config.rms_epsilon = g.rms_epsilon;
        // EmbeddingGemma stores the total bidirectional span in the checkpoint.
        // Encoder kernels take an exclusive distance from the query instead.
        config.sliding_window = config.mean_pooling && g.sliding_window > 0
                                    ? g.sliding_window / 2 + 1 : g.sliding_window;
        config.rope_theta_global = g.rope_theta;
        config.rope_theta_local = gemma ? g.sliding_rope_theta : g.rope_theta;
        config.attention_scale = g.attention_scale;
        config.embedding_scale = gemma ? g.embedding_scale : 1.0F;
        for (std::int32_t layer = 0; layer < g.layers; ++layer) {
            if (!g.layer_attends(layer)) {
                throw std::invalid_argument("the embedding backend requires attention at every layer");
            }
            config.global_layers.push_back(!g.layer_is_windowed(layer));
        }
        return config;
    }
};

class TextEmbedding {
public:
    /// Binds every object the artifact holds and uploads it. Throws if the
    /// artifact carries an object this target does not consume, or omits one it
    /// requires -- the same contract the engine's targets keep.
    static TextEmbedding load(const std::filesystem::path& artifact, DeviceContext& device);

    ~TextEmbedding();
    TextEmbedding(TextEmbedding&&) noexcept;
    TextEmbedding& operator=(TextEmbedding&&) noexcept;
    TextEmbedding(const TextEmbedding&)            = delete;
    TextEmbedding& operator=(const TextEmbedding&) = delete;

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
    [[nodiscard]] const EmbeddingTokenizer& tokenizer() const noexcept;

    [[nodiscard]] const TextEmbeddingConfig& config() const noexcept;
    /// Bytes of device memory the weights occupy.
    [[nodiscard]] std::uint64_t weight_bytes() const noexcept;

private:
    TextEmbedding();
    /// One forward. Its sequences must already fit `max_batch_tokens`.
    [[nodiscard]] std::vector<std::vector<float>> embed_chunk(
        const std::vector<std::vector<std::int32_t>>& sequences);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sinfer::encoder
