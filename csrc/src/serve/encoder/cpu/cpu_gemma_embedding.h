#pragma once

// The EmbeddingGemma encoder on host cores.
//
// The same graph as the GPU path in encoder/gemma_embedding.cpp, over the
// kernels in cpu/cpu_ops.h, reading the same artifact. It is a separate
// implementation rather than a backend behind a virtual interface because the
// two differ in more than their arithmetic: the GPU binds quantised `Weight`
// objects on device and the host decodes them to FP32 once at load. What keeps
// the two honest is not shared code but a shared oracle -- both are checked
// against the same reference embeddings, to the same tolerance.
//
// Weights are decoded to FP32 at load: 1.2 GB for a 300M model, against a host
// that has 504. At 100B parameters this would be the wrong trade and the
// quantised kernels would have to come back.

#include "encoder/cpu/cpu_ops.h"
#include "encoder/gemma_embedding.h" // GemmaEmbeddingConfig
#include "encoder/gemma_tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace sinfer::encoder::cpu {

class CpuGemmaEmbedding {
public:
    /// Reads the artifact and decodes every weight to FP32. `plan` defaults to
    /// the physical cores of one NUMA node; see ThreadPlan.
    static CpuGemmaEmbedding load(const std::filesystem::path& artifact, ThreadPlan plan = {});

    ~CpuGemmaEmbedding();
    CpuGemmaEmbedding(CpuGemmaEmbedding&&) noexcept;
    CpuGemmaEmbedding& operator=(CpuGemmaEmbedding&&) noexcept;
    CpuGemmaEmbedding(const CpuGemmaEmbedding&)            = delete;
    CpuGemmaEmbedding& operator=(const CpuGemmaEmbedding&) = delete;

    [[nodiscard]] std::vector<float> embed(std::span<const std::int32_t> tokens);
    [[nodiscard]] std::vector<std::vector<float>> embed_batch(
        const std::vector<std::vector<std::int32_t>>& sequences);

    [[nodiscard]] const GemmaEmbeddingConfig& config() const noexcept;
    [[nodiscard]] const GemmaTokenizer& tokenizer() const noexcept;
    /// Bytes of host memory the decoded weights occupy.
    [[nodiscard]] std::uint64_t weight_bytes() const noexcept;
    [[nodiscard]] int threads() const noexcept;

private:
    CpuGemmaEmbedding();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sinfer::encoder::cpu
