#pragma once

// Gemma and Qwen text embedding backbones on host cores.
//
// The same graph as the GPU path in encoder/text_embedding.cpp, over the
// kernels in cpu/cpu_ops.h, reading the same artifact. It is a separate
// implementation rather than a backend behind a virtual interface because the
// two differ in more than their arithmetic: the GPU binds quantised `Weight`
// objects on device and the host decodes them to BF16 once at load. What keeps
// the two honest is not shared code but a shared oracle -- both are checked
// against the same reference embeddings, to the same tolerance.
//
// Weights are decoded to BF16 at load: 0.6 GB for a 300M model, against a host
// that has 504. At 100B parameters this would be the wrong trade and the
// quantised kernels would have to come back.

#include "encoder/cpu/cpu_ops.h"
#include "encoder/text_embedding.h" // TextEmbeddingConfig
#include "encoder/embedding_tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace sinfer::encoder::cpu {

class CpuTextEmbedding {
public:
    /// Reads the artifact and decodes every weight to FP32. `plan` defaults to
    /// the physical cores of one NUMA node; see ThreadPlan.
    static CpuTextEmbedding load(const std::filesystem::path& artifact, ThreadPlan plan = {});

    ~CpuTextEmbedding();
    CpuTextEmbedding(CpuTextEmbedding&&) noexcept;
    CpuTextEmbedding& operator=(CpuTextEmbedding&&) noexcept;
    CpuTextEmbedding(const CpuTextEmbedding&)            = delete;
    CpuTextEmbedding& operator=(const CpuTextEmbedding&) = delete;

    [[nodiscard]] std::vector<float> embed(std::span<const std::int32_t> tokens);
    [[nodiscard]] std::vector<std::vector<float>> embed_batch(
        const std::vector<std::vector<std::int32_t>>& sequences);

    [[nodiscard]] const TextEmbeddingConfig& config() const noexcept;
    [[nodiscard]] const EmbeddingTokenizer& tokenizer() const noexcept;
    /// Bytes of host memory the decoded weights occupy.
    [[nodiscard]] std::uint64_t weight_bytes() const noexcept;
    [[nodiscard]] int threads() const noexcept;

private:
    CpuTextEmbedding();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sinfer::encoder::cpu
