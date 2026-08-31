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
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace sinfer::encoder {

/// Geometry the target is built for. Everything here comes from the DSL
/// declaration by way of the artifact; nothing is read from a checkpoint config
/// at serve time.
struct GemmaEmbeddingConfig {
    std::int32_t layers       = 24;
    std::int32_t hidden       = 768;
    std::int32_t query_heads  = 3;
    std::int32_t head_dim     = 256;
    std::int32_t intermediate = 1152;
    std::int32_t vocab        = 262144;
    std::int32_t max_tokens   = 2048;
    /// Total tokens one batched forward may carry, across all its sequences.
    std::int32_t max_batch_tokens = 8192;

    float rms_epsilon = 1.0e-6F;
    /// Local layers mask abs(i - j) >= window; global layers mask nothing.
    std::int32_t sliding_window = 512;
    /// Layer i is global when (i + 1) % period == 0: 5, 11, 17, 23 for 24 layers.
    std::int32_t sliding_period = 6;
    float rope_theta_global     = 1.0e6F;
    float rope_theta_local      = 1.0e4F;
    /// query_pre_attn_scalar ** -0.5, which is *not* always 1/sqrt(head_dim).
    float attention_scale = 0.0625F;
    /// Gemma scales the embedding table by sqrt(hidden) on the way in.
    float embedding_scale = 27.712812921102035F; // sqrt(768)

    [[nodiscard]] bool is_global(std::int32_t layer) const noexcept {
        return (layer + 1) % sliding_period == 0;
    }
    [[nodiscard]] std::int32_t query_size() const noexcept { return query_heads * head_dim; }
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
