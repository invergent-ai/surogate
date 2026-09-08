#include "encoder/cpu/cpu_gemma_embedding.h"

#include "artifact/reader.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sinfer::encoder::cpu {
namespace {

/// Phase timing, off unless SINFER_CPU_PROFILE is set. Guessing where a forward
/// spends itself has already been wrong once here by a factor of five.
struct Profile {
    bool on = std::getenv("SINFER_CPU_PROFILE") != nullptr;
    std::vector<std::pair<const char*, double>> phases;
    std::chrono::steady_clock::time_point mark;

    void begin() { if (on) { mark = std::chrono::steady_clock::now(); } }
    void end(const char* name) {
        if (!on) { return; }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - mark).count();
        for (auto& [label, total] : phases) {
            if (label == name) { total += seconds; return; }
        }
        phases.emplace_back(name, seconds);
    }
    void report() const {
        if (!on) { return; }
        double total = 0.0;
        for (const auto& [_, seconds] : phases) { total += seconds; }
        std::fprintf(stderr, "  --- forward %.1f ms ---\n", total * 1e3);
        for (const auto& [label, seconds] : phases) {
            std::fprintf(stderr, "  %-14s %7.1f ms  %5.1f%%\n", label, seconds * 1e3,
                         100.0 * seconds / total);
        }
    }
};

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

std::uint16_t float_to_bf16(float value) {
    std::uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    // Round to nearest even, which is what every producer of BF16 here does.
    word += 0x7FFFU + ((word >> 16U) & 1U);
    return static_cast<std::uint16_t>(word >> 16U);
}

/// One W8G32_F16S row-split object to BF16 [n, k], k contiguous.
///
/// BF16 rather than FP32 deliberately: the int8 code times its FP16 scale never
/// carried more than BF16 keeps, and half the bytes means half of what every
/// GEMM reads. Measured against the FP32 decode this changed the final
/// embeddings by nothing visible at the reference tolerance.
///
/// The layout is two planes: `n * groups * 32` int8 codes from offset 0, then
/// the binary16 scales at the next 256-byte boundary, one per 32-value group.
/// `k` is padded up to 128 for the codes; Gemma's 768 and 1152 are already
/// multiples of it, but the stride is computed rather than assumed.
std::vector<std::uint16_t> decode_w8(std::span<const std::byte> payload, std::int32_t n,
                                     std::int32_t k) {
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

    std::vector<std::uint16_t> out(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        const std::int8_t* code_row    = codes + static_cast<std::uint64_t>(row) * groups * kGroup;
        const std::uint16_t* scale_row = scales + static_cast<std::uint64_t>(row) * groups;
        std::uint16_t* target          = out.data() + static_cast<std::size_t>(row) * k;
        for (std::uint64_t group = 0; group < groups; ++group) {
            const float scale     = fp16_to_float(scale_row[group]);
            const std::int32_t lo = static_cast<std::int32_t>(group) * kGroup;
            const std::int32_t hi = std::min(k, lo + kGroup);
            for (std::int32_t i = lo; i < hi; ++i) {
                target[i] = float_to_bf16(static_cast<float>(code_row[i]) * scale);
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

/// A BF16 artifact object kept as BF16 -- the embedding head is stored that way
/// and the GEMM now reads that way.
std::vector<std::uint16_t> raw_bf16(std::span<const std::byte> payload, std::size_t count) {
    if (payload.size() < count * 2) {
        throw std::runtime_error("raw_bf16: payload shorter than its shape implies");
    }
    const auto* words = reinterpret_cast<const std::uint16_t*>(payload.data());
    return {words, words + count};
}

std::string layer_object(std::int32_t layer, const char* leaf) {
    return "text/layers/" + std::to_string(layer) + "/" + leaf;
}

struct LayerWeights {
    // Norms stay FP32: rmsnorm reads them per element and they are tiny.
    std::vector<float> input_norm, post_attention_norm, pre_feedforward_norm,
        post_feedforward_norm, query_norm, key_norm;
    // Matrices are BF16: they are what the GEMMs stream.
    std::vector<std::uint16_t> query, key, value, output, gate, up, down;
};

} // namespace

struct CpuGemmaEmbedding::Impl {
    GemmaEmbeddingConfig config;
    std::unique_ptr<ThreadPool> pool;
    std::unique_ptr<GemmaTokenizer> tokenizer;

    std::vector<std::uint16_t> token_embedding, embedding_head;
    std::vector<float> final_norm;

    // Forward scratch, sized once for max_tokens and reused: allocating these
    // per call meant page-faulting ~15 MB per forward, which measured as most
    // of the ~28 ms the phase profile could not see.
    struct Scratch {
        std::vector<float> x, h, attn, query, key, value, attn_out, gate, up, scores;
        std::vector<std::uint16_t> h16, wide16, q16, k16, v16;
        std::vector<std::int32_t> positions;
    } scratch;
    RopeTable rope_global, rope_local;
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
    impl.config = GemmaEmbeddingConfig::from_artifact(reader);
    const GemmaEmbeddingConfig& config = impl.config;

    const auto checked_tensor = [&](const std::string& name, artifact::NumericFormat format,
                                    std::initializer_list<std::uint64_t> shape) {
        const auto* object = reader.find(name);
        const auto* tensor = object ? std::get_if<artifact::TensorDescriptor>(object) : nullptr;
        const auto layout = format == artifact::NumericFormat::BF16
                                ? artifact::StorageLayout::ContiguousLeV1
                                : artifact::StorageLayout::RowSplitK128V1;
        if (tensor == nullptr || tensor->format != format || tensor->layout != layout ||
            tensor->shape != std::vector<std::uint64_t>(shape)) {
            throw std::invalid_argument("embedding tensor disagrees with checkpoint metadata: " + name);
        }
    };

    const auto quantised = [&](const std::string& name, std::int32_t n, std::int32_t k) {
        checked_tensor(name, artifact::NumericFormat::W8G32_F16S,
                       {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k)});
        const auto span = reader.payload(name);
        return decode_w8(span.data, n, k);
    };
    const auto dense = [&](const std::string& name, std::size_t count) {
        checked_tensor(name, artifact::NumericFormat::BF16, {count});
        return decode_bf16(reader.payload(name).data, count);
    };

    impl.token_embedding = quantised("text/token_embedding", config.vocab, config.hidden);
    impl.final_norm      = dense("text/final_norm", static_cast<std::size_t>(config.hidden));
    checked_tensor("text/embedding_head", artifact::NumericFormat::BF16,
                   {static_cast<std::uint64_t>(config.hidden), static_cast<std::uint64_t>(config.hidden)});
    impl.embedding_head =
        raw_bf16(reader.payload("text/embedding_head").data,
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

    // Two bases, two tables, built once: the angles never depend on the data.
    impl.rope_global = RopeTable(config.head_dim, config.max_tokens, config.rope_theta_global);
    impl.rope_local  = RopeTable(config.head_dim, config.max_tokens, config.rope_theta_local);

    const auto tokenizer_bytes = reader.payload("frontend/tokenizer.model").data;
    impl.tokenizer =
        std::make_unique<GemmaTokenizer>(GemmaTokenizer::from_serialized_proto(tokenizer_bytes));

    {
        // Touch every scratch page now rather than during the first request.
        const auto span         = static_cast<std::size_t>(config.max_tokens);
        const auto hidden       = static_cast<std::size_t>(config.hidden);
        const auto query_size   = static_cast<std::size_t>(config.query_size());
        const auto head_dim     = static_cast<std::size_t>(config.head_dim);
        const auto intermediate = static_cast<std::size_t>(config.intermediate);
        Impl::Scratch& scratch  = impl.scratch;
        scratch.x.assign(hidden * span, 0.0F);
        scratch.h.assign(hidden * span, 0.0F);
        scratch.attn.assign(hidden * span, 0.0F);
        scratch.query.assign(query_size * span, 0.0F);
        scratch.key.assign(head_dim * span, 0.0F);
        scratch.value.assign(head_dim * span, 0.0F);
        scratch.attn_out.assign(query_size * span, 0.0F);
        scratch.gate.assign(intermediate * span, 0.0F);
        scratch.up.assign(intermediate * span, 0.0F);
        scratch.scores.assign(attention_scratch(config.query_heads, config.max_tokens), 0.0F);
        scratch.h16.assign(hidden * span, 0);
        scratch.wide16.assign(std::max(query_size, intermediate) * span, 0);
        scratch.q16.assign(query_size * span, 0);
        scratch.k16.assign(head_dim * span, 0);
        scratch.v16.assign(head_dim * span, 0);
        scratch.positions.assign(span, 0);
    }

    impl.bytes = static_cast<std::uint64_t>(impl.token_embedding.size() +
                                            impl.embedding_head.size()) *
                     sizeof(std::uint16_t) +
                 static_cast<std::uint64_t>(impl.final_norm.size()) * sizeof(float);
    for (const LayerWeights& w : impl.layers) {
        impl.bytes += static_cast<std::uint64_t>(
                          w.input_norm.size() + w.post_attention_norm.size() +
                          w.pre_feedforward_norm.size() + w.post_feedforward_norm.size() +
                          w.query_norm.size() + w.key_norm.size()) *
                          sizeof(float) +
                      static_cast<std::uint64_t>(w.query.size() + w.key.size() + w.value.size() +
                                                 w.output.size() + w.gate.size() + w.up.size() +
                                                 w.down.size()) *
                          sizeof(std::uint16_t);
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

        const auto span   = static_cast<std::size_t>(tokens);
        Impl::Scratch& sc = impl.scratch;
        auto& positions   = sc.positions;
        for (std::int32_t i = 0; i < tokens; ++i) { positions[static_cast<std::size_t>(i)] = i; }
        auto &x = sc.x, &h = sc.h, &attn = sc.attn, &query = sc.query, &key = sc.key,
             &value = sc.value, &attn_out = sc.attn_out, &gate = sc.gate, &up = sc.up;
        auto &h16 = sc.h16, &wide16 = sc.wide16, &q16 = sc.q16, &k16 = sc.k16, &v16 = sc.v16;
        auto& scratch = sc.scores;
        Profile profile;

        // Qualified: the member `embed` would otherwise shadow the kernel.
        cpu::embed(impl.token_embedding.data(), sequence.data(), x.data(), config.hidden, tokens,
                   config.vocab);
        scale(x.data(), config.embedding_scale, static_cast<std::int64_t>(hidden * span));

        for (std::int32_t layer = 0; layer < config.layers; ++layer) {
            const LayerWeights& w     = impl.layers[static_cast<std::size_t>(layer)];
            const bool global         = config.is_global(layer);
            const RopeTable& angles   = global ? impl.rope_global : impl.rope_local;
            const std::int32_t window = global ? 0 : config.sliding_window;

            profile.begin();
            rmsnorm(x.data(), w.input_norm.data(), config.rms_epsilon, true, h.data(),
                    config.hidden, tokens, *impl.pool);
            narrow(h.data(), h16.data(), static_cast<std::int64_t>(hidden * span), *impl.pool);
            profile.end("rmsnorm");
            profile.begin();
            gemm(w.query.data(), h16.data(), query.data(), config.query_size(), config.hidden,
                 tokens, *impl.pool);
            gemm(w.key.data(), h16.data(), key.data(), config.head_dim, config.hidden, tokens,
                 *impl.pool);
            gemm(w.value.data(), h16.data(), value.data(), config.head_dim, config.hidden, tokens,
                 *impl.pool);
            profile.end("gemm");
            profile.begin();

            // Per-head QK norm: heads are contiguous within a column, so the
            // whole buffer is heads * tokens rows of head_dim.
            rmsnorm(query.data(), w.query_norm.data(), config.rms_epsilon, true, query.data(),
                    config.head_dim, config.query_heads * tokens, *impl.pool);
            rmsnorm(key.data(), w.key_norm.data(), config.rms_epsilon, true, key.data(),
                    config.head_dim, tokens, *impl.pool);

            rope(query.data(), positions.data(), config.head_dim, config.query_heads, tokens,
                 angles, *impl.pool);
            rope(key.data(), positions.data(), config.head_dim, 1, tokens, angles, *impl.pool);
            profile.end("qknorm+rope");
            profile.begin();

            narrow(query.data(), q16.data(), static_cast<std::int64_t>(query_size * span),
                   *impl.pool);
            narrow(key.data(), k16.data(), static_cast<std::int64_t>(head_dim * span), *impl.pool);
            narrow(value.data(), v16.data(), static_cast<std::int64_t>(head_dim * span),
                   *impl.pool);
            attention(q16.data(), k16.data(), v16.data(), attn_out.data(), config.query_heads,
                      config.head_dim, tokens, window, config.attention_scale, scratch.data(),
                      *impl.pool);
            profile.end("attention");
            profile.begin();

            narrow(attn_out.data(), wide16.data(),
                   static_cast<std::int64_t>(query_size * span), *impl.pool);
            gemm(w.output.data(), wide16.data(), attn.data(), config.hidden,
                 config.query_size(), tokens, *impl.pool);
            rmsnorm(attn.data(), w.post_attention_norm.data(), config.rms_epsilon, true,
                    attn.data(), config.hidden, tokens, *impl.pool);
            add(attn.data(), x.data(), static_cast<std::int64_t>(hidden * span), *impl.pool);

            rmsnorm(x.data(), w.pre_feedforward_norm.data(), config.rms_epsilon, true, h.data(),
                    config.hidden, tokens, *impl.pool);
            narrow(h.data(), h16.data(), static_cast<std::int64_t>(hidden * span), *impl.pool);
            gemm(w.up.data(), h16.data(), up.data(), config.intermediate, config.hidden, tokens,
                 *impl.pool);
            profile.end("gemm");
            profile.begin();
            // gelu(gate) * up rides the gate projection when a backend can fuse
            // it, which saves two passes over an [intermediate, tokens] buffer.
            if (!gemm_gelu_mul(w.gate.data(), h16.data(), up.data(), gate.data(),
                               config.intermediate, config.hidden, tokens)) {
                gemm(w.gate.data(), h16.data(), gate.data(), config.intermediate, config.hidden,
                     tokens, *impl.pool);
                gelu_mul(gate.data(), up.data(), gate.data(),
                         static_cast<std::int64_t>(intermediate * span), *impl.pool);
            }
            profile.end("mlp_act");
            profile.begin();
            narrow(gate.data(), wide16.data(),
                   static_cast<std::int64_t>(intermediate * span), *impl.pool);
            gemm(w.down.data(), wide16.data(), attn.data(), config.hidden, config.intermediate,
                 tokens, *impl.pool);
            rmsnorm(attn.data(), w.post_feedforward_norm.data(), config.rms_epsilon, true,
                    attn.data(), config.hidden, tokens, *impl.pool);
            add(attn.data(), x.data(), static_cast<std::int64_t>(hidden * span), *impl.pool);
            profile.end("gemm");
        }

        rmsnorm(x.data(), impl.final_norm.data(), config.rms_epsilon, true, h.data(),
                config.hidden, tokens, *impl.pool);

        std::vector<float> pooled(hidden);
        mean_pool(h.data(), pooled.data(), config.hidden, tokens);
        std::vector<std::uint16_t> pooled16(hidden);
        narrow(pooled.data(), pooled16.data(), static_cast<std::int64_t>(hidden), *impl.pool);
        std::vector<float> projected(hidden);
        gemm(impl.embedding_head.data(), pooled16.data(), projected.data(), config.hidden,
             config.hidden, 1, *impl.pool);
        l2norm(projected.data(), config.hidden, 1, 1.0e-12F);
        profile.report();
        out.push_back(std::move(projected));
    }
    return out;
}

} // namespace sinfer::encoder::cpu
