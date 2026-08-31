#include "encoder/cpu/cpu_gemma_embedding.h"

#include "artifact/reader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sinfer::encoder::cpu {
namespace {

constexpr std::int32_t kGroup          = 32;
constexpr std::uint64_t kKAlignment    = 128; // row-split-k128-v1
constexpr std::uint64_t kPlaneAlignment = 256;

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

float fp16_to_float(std::uint16_t bits) {
    const std::uint32_t sign     = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1FU;
    const std::uint32_t mantissa = bits & 0x3FFU;
    std::uint32_t word           = 0;
    if (exponent == 0) {
        if (mantissa != 0) { // subnormal: renormalise
            std::uint32_t e = 127 - 15 + 1;
            std::uint32_t m = mantissa;
            while ((m & 0x400U) == 0) {
                m <<= 1U;
                --e;
            }
            word = sign | (e << 23U) | ((m & 0x3FFU) << 13U);
        } else {
            word = sign;
        }
    } else if (exponent == 0x1FU) {
        word = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        word = sign | ((exponent - 15 + 127) << 23U) | (mantissa << 13U);
    }
    float value = 0.0F;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value              = 0.0F;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

/// One W8G32_F16S row-split object to FP32 [n, k], k contiguous.
///
/// The layout is two planes: `n * groups * 32` int8 codes from offset 0, then
/// the binary16 scales at the next 256-byte boundary, one per 32-value group.
/// `k` is padded up to 128 for the codes; Gemma's 768 and 1152 are already
/// multiples of it, but the stride is computed rather than assumed.
std::vector<float> decode_w8(std::span<const std::byte> payload, std::int32_t n, std::int32_t k) {
    const std::uint64_t k_pad         = align_up(static_cast<std::uint64_t>(k), kKAlignment);
    const std::uint64_t groups        = k_pad / kGroup;
    const std::uint64_t code_bytes    = static_cast<std::uint64_t>(n) * groups * kGroup;
    const std::uint64_t scale_offset  = align_up(code_bytes, kPlaneAlignment);
    const std::uint64_t scale_bytes   = static_cast<std::uint64_t>(n) * groups * 2;
    if (payload.size() < scale_offset + scale_bytes) {
        throw std::runtime_error("decode_w8: payload shorter than its geometry implies");
    }

    const auto* codes  = reinterpret_cast<const std::int8_t*>(payload.data());
    const auto* scales = reinterpret_cast<const std::uint16_t*>(payload.data() + scale_offset);

    std::vector<float> out(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        const std::int8_t* code_row    = codes + static_cast<std::uint64_t>(row) * groups * kGroup;
        const std::uint16_t* scale_row = scales + static_cast<std::uint64_t>(row) * groups;
        float* target                  = out.data() + static_cast<std::size_t>(row) * k;
        for (std::uint64_t group = 0; group < groups; ++group) {
            const float scale     = fp16_to_float(scale_row[group]);
            const std::int32_t lo = static_cast<std::int32_t>(group) * kGroup;
            const std::int32_t hi = std::min(k, lo + kGroup);
            for (std::int32_t i = lo; i < hi; ++i) {
                target[i] = static_cast<float>(code_row[i]) * scale;
            }
        }
    }
    return out;
}

std::vector<float> decode_bf16(std::span<const std::byte> payload, std::size_t count) {
    if (payload.size() < count * 2) {
        throw std::runtime_error("decode_bf16: payload shorter than its shape implies");
    }
    const auto* words = reinterpret_cast<const std::uint16_t*>(payload.data());
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) { out[i] = bf16_to_float(words[i]); }
    return out;
}

std::string layer_object(std::int32_t layer, const char* leaf) {
    return "text/layers/" + std::to_string(layer) + "/" + leaf;
}

struct LayerWeights {
    std::vector<float> input_norm, post_attention_norm, pre_feedforward_norm,
        post_feedforward_norm;
    std::vector<float> query, key, value, query_norm, key_norm, output, gate, up, down;
};

} // namespace

struct CpuGemmaEmbedding::Impl {
    GemmaEmbeddingConfig config;
    std::unique_ptr<ThreadPool> pool;
    std::unique_ptr<GemmaTokenizer> tokenizer;

    std::vector<float> token_embedding, final_norm, embedding_head;
    std::vector<LayerWeights> layers;
    std::uint64_t bytes = 0;
};

CpuGemmaEmbedding::CpuGemmaEmbedding() : impl_(std::make_unique<Impl>()) {}
CpuGemmaEmbedding::~CpuGemmaEmbedding()                                          = default;
CpuGemmaEmbedding::CpuGemmaEmbedding(CpuGemmaEmbedding&&) noexcept               = default;
CpuGemmaEmbedding& CpuGemmaEmbedding::operator=(CpuGemmaEmbedding&&) noexcept    = default;

const GemmaEmbeddingConfig& CpuGemmaEmbedding::config() const noexcept { return impl_->config; }
const GemmaTokenizer& CpuGemmaEmbedding::tokenizer() const noexcept { return *impl_->tokenizer; }
std::uint64_t CpuGemmaEmbedding::weight_bytes() const noexcept { return impl_->bytes; }
int CpuGemmaEmbedding::threads() const noexcept { return impl_->pool->threads(); }

CpuGemmaEmbedding CpuGemmaEmbedding::load(const std::filesystem::path& path, ThreadPlan plan) {
    CpuGemmaEmbedding model;
    Impl& impl = *model.impl_;
    if (plan.threads == 0 && plan.cpus.empty()) { plan = ThreadPlan::detect(); }
    impl.pool = std::make_unique<ThreadPool>(std::move(plan));

    const artifact::Reader reader(path);
    const GemmaEmbeddingConfig& config = impl.config;

    const auto quantised = [&](const std::string& name, std::int32_t n, std::int32_t k) {
        const auto span = reader.payload(name);
        return decode_w8(span.data, n, k);
    };
    const auto dense = [&](const std::string& name, std::size_t count) {
        return decode_bf16(reader.payload(name).data, count);
    };

    impl.token_embedding = quantised("text/token_embedding", config.vocab, config.hidden);
    impl.final_norm      = dense("text/final_norm", static_cast<std::size_t>(config.hidden));
    impl.embedding_head  = dense("text/embedding_head",
                                 static_cast<std::size_t>(config.hidden) * config.hidden);

    impl.layers.resize(static_cast<std::size_t>(config.layers));
    for (std::int32_t layer = 0; layer < config.layers; ++layer) {
        LayerWeights& w = impl.layers[static_cast<std::size_t>(layer)];
        const auto norm = [&](const char* leaf, std::size_t width) {
            return dense(layer_object(layer, leaf), width);
        };
        const auto matrix = [&](const char* leaf, std::int32_t n, std::int32_t k) {
            return quantised(layer_object(layer, leaf), n, k);
        };
        const auto hidden = static_cast<std::size_t>(config.hidden);
        const auto head   = static_cast<std::size_t>(config.head_dim);
        w.input_norm            = norm("input_norm", hidden);
        w.post_attention_norm   = norm("post_attention_norm", hidden);
        w.pre_feedforward_norm  = norm("pre_feedforward_norm", hidden);
        w.post_feedforward_norm = norm("post_feedforward_norm", hidden);
        w.query_norm            = norm("attention/query_norm", head);
        w.key_norm              = norm("attention/key_norm", head);
        w.query  = matrix("attention/query", config.query_size(), config.hidden);
        w.key    = matrix("attention/key", config.head_dim, config.hidden);
        w.value  = matrix("attention/value", config.head_dim, config.hidden);
        w.output = matrix("attention/output", config.hidden, config.query_size());
        w.gate   = matrix("mlp/gate", config.intermediate, config.hidden);
        w.up     = matrix("mlp/up", config.intermediate, config.hidden);
        w.down   = matrix("mlp/down", config.hidden, config.intermediate);
    }

    const auto tokenizer_bytes = reader.payload("frontend/tokenizer.model").data;
    impl.tokenizer =
        std::make_unique<GemmaTokenizer>(GemmaTokenizer::from_serialized_proto(tokenizer_bytes));

    impl.bytes = static_cast<std::uint64_t>(impl.token_embedding.size() +
                                            impl.final_norm.size() +
                                            impl.embedding_head.size()) *
                 sizeof(float);
    for (const LayerWeights& w : impl.layers) {
        impl.bytes += static_cast<std::uint64_t>(
                          w.input_norm.size() + w.post_attention_norm.size() +
                          w.pre_feedforward_norm.size() + w.post_feedforward_norm.size() +
                          w.query_norm.size() + w.key_norm.size() + w.query.size() + w.key.size() +
                          w.value.size() + w.output.size() + w.gate.size() + w.up.size() +
                          w.down.size()) *
                      sizeof(float);
    }
    return model;
}

std::vector<float> CpuGemmaEmbedding::embed(std::span<const std::int32_t> tokens) {
    std::vector<std::vector<std::int32_t>> one{
        std::vector<std::int32_t>(tokens.begin(), tokens.end())};
    return embed_batch(one).front();
}

std::vector<std::vector<float>> CpuGemmaEmbedding::embed_batch(
    const std::vector<std::vector<std::int32_t>>& sequences) {
    Impl& impl                         = *impl_;
    const GemmaEmbeddingConfig& config = impl.config;
    std::vector<std::vector<float>> out;
    out.reserve(sequences.size());

    // One sequence per forward here, where the GPU concatenates a batch. On CPU
    // the projections already saturate the cores with a single sequence -- the
    // measured llama.cpp throughput is flat in concurrency for exactly this
    // reason -- so concatenating would buy latency, not throughput.
    for (const std::vector<std::int32_t>& sequence : sequences) {
        const auto tokens = static_cast<std::int32_t>(sequence.size());
        if (tokens <= 0) { throw std::invalid_argument("embed: a sequence is empty"); }
        if (tokens > config.max_tokens) {
            throw std::invalid_argument("embed: " + std::to_string(tokens) + " tokens exceeds " +
                                        std::to_string(config.max_tokens));
        }

        const auto hidden       = static_cast<std::size_t>(config.hidden);
        const auto query_size   = static_cast<std::size_t>(config.query_size());
        const auto head_dim     = static_cast<std::size_t>(config.head_dim);
        const auto intermediate = static_cast<std::size_t>(config.intermediate);
        const auto span         = static_cast<std::size_t>(tokens);

        std::vector<std::int32_t> positions(span);
        for (std::int32_t i = 0; i < tokens; ++i) { positions[static_cast<std::size_t>(i)] = i; }

        std::vector<float> x(hidden * span), h(hidden * span), attn(hidden * span);
        std::vector<float> query(query_size * span), key(head_dim * span), value(head_dim * span);
        std::vector<float> attn_out(query_size * span);
        std::vector<float> gate(intermediate * span), up(intermediate * span);
        std::vector<float> scratch(attention_scratch(config.query_heads, tokens));

        // Qualified: the member `embed` would otherwise shadow the kernel.
        cpu::embed(impl.token_embedding.data(), sequence.data(), x.data(), config.hidden, tokens,
                   config.vocab);
        scale(x.data(), config.embedding_scale, static_cast<std::int64_t>(x.size()));

        for (std::int32_t layer = 0; layer < config.layers; ++layer) {
            const LayerWeights& w     = impl.layers[static_cast<std::size_t>(layer)];
            const bool global         = config.is_global(layer);
            const float theta         = global ? config.rope_theta_global : config.rope_theta_local;
            const std::int32_t window = global ? 0 : config.sliding_window;

            rmsnorm(x.data(), w.input_norm.data(), config.rms_epsilon, true, h.data(),
                    config.hidden, tokens);
            gemm(w.query.data(), h.data(), query.data(), config.query_size(), config.hidden,
                 tokens, *impl.pool);
            gemm(w.key.data(), h.data(), key.data(), config.head_dim, config.hidden, tokens,
                 *impl.pool);
            gemm(w.value.data(), h.data(), value.data(), config.head_dim, config.hidden, tokens,
                 *impl.pool);

            // Per-head QK norm: heads are contiguous within a column, so the
            // whole buffer is heads * tokens rows of head_dim.
            rmsnorm(query.data(), w.query_norm.data(), config.rms_epsilon, true, query.data(),
                    config.head_dim, config.query_heads * tokens);
            rmsnorm(key.data(), w.key_norm.data(), config.rms_epsilon, true, key.data(),
                    config.head_dim, tokens);

            rope(query.data(), positions.data(), config.head_dim, config.query_heads, tokens,
                 theta);
            rope(key.data(), positions.data(), config.head_dim, 1, tokens, theta);

            attention(query.data(), key.data(), value.data(), attn_out.data(), config.query_heads,
                      config.head_dim, tokens, window, config.attention_scale, scratch.data(),
                      *impl.pool);

            gemm(w.output.data(), attn_out.data(), attn.data(), config.hidden,
                 config.query_size(), tokens, *impl.pool);
            rmsnorm(attn.data(), w.post_attention_norm.data(), config.rms_epsilon, true,
                    attn.data(), config.hidden, tokens);
            add(attn.data(), x.data(), static_cast<std::int64_t>(x.size()));

            rmsnorm(x.data(), w.pre_feedforward_norm.data(), config.rms_epsilon, true, h.data(),
                    config.hidden, tokens);
            gemm(w.gate.data(), h.data(), gate.data(), config.intermediate, config.hidden, tokens,
                 *impl.pool);
            gemm(w.up.data(), h.data(), up.data(), config.intermediate, config.hidden, tokens,
                 *impl.pool);
            gelu_mul(gate.data(), up.data(), gate.data(), static_cast<std::int64_t>(gate.size()));
            gemm(w.down.data(), gate.data(), attn.data(), config.hidden, config.intermediate,
                 tokens, *impl.pool);
            rmsnorm(attn.data(), w.post_feedforward_norm.data(), config.rms_epsilon, true,
                    attn.data(), config.hidden, tokens);
            add(attn.data(), x.data(), static_cast<std::int64_t>(x.size()));
        }

        rmsnorm(x.data(), impl.final_norm.data(), config.rms_epsilon, true, h.data(),
                config.hidden, tokens);

        std::vector<float> pooled(hidden);
        mean_pool(h.data(), pooled.data(), config.hidden, tokens);
        std::vector<float> projected(hidden);
        gemm(impl.embedding_head.data(), pooled.data(), projected.data(), config.hidden,
             config.hidden, 1, *impl.pool);
        l2norm(projected.data(), config.hidden, 1, 1.0e-12F);
        out.push_back(std::move(projected));
    }
    return out;
}

} // namespace sinfer::encoder::cpu
