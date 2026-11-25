// Inkling-2-Mini's DSpark drafter: load-time derivations and the append / propose rounds (dspark.h).

#include "targets/inkling2/impl/dspark.h"
#include "targets/inkling2/impl/variant.h"
#include "targets/inkling2/impl/visits.h"
#include "targets/inkling2/impl/loopcd.h"

#include "api/ops/inkling_mix.h"
#include "api/ops/inkling_attention.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>

namespace sinfer::targets::inkling2::detail {
namespace {

using G = ops::DSparkGeometry;

constexpr std::int32_t kAppendChunk = 1024; ///< columns per append pass (a prefill chunk is cut into these)

const void* bf16_weight(const Weight& w, std::int32_t n, std::int32_t k, const char* what) {
    if (w.qtype != QType::BF16_CTRL || w.qdata == nullptr || w.n != n || w.k != k) {
        throw std::logic_error(std::string("inkling2 DSpark: weight ") + what + " is not BF16 [" + std::to_string(n) +
                               ", " + std::to_string(k) + "]");
    }
    return w.qdata;
}

const void* bf16_tensor(const Tensor& t, std::int64_t numel, const char* what) {
    if (t.data == nullptr || t.dtype != DType::BF16 || t.numel() != numel) {
        throw std::logic_error(std::string("inkling2 DSpark: tensor ") + what + " is not BF16 of the expected size");
    }
    return t.data;
}

float bf16_to_float(std::uint16_t h) {
    const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

std::size_t align256(std::size_t x) { return (x + 255) & ~std::size_t{255}; }

void* scratch(WorkspaceArena& work, std::size_t bytes) {
    return work.alloc_bytes(std::max<std::size_t>(bytes, 16)).data;
}

/// out [n, tokens] (ldc = n) = W [n, k] . x on the invariant GEMM.
void gemm(WorkspaceArena& work, const void* w, std::int32_t n, std::int32_t k, const void* x, std::int32_t tokens,
          void* out, DType dtype, cudaStream_t s) {
    const std::size_t bytes = ops::inkling_gemm_inv_workspace_bytes(n, k, tokens, 1);
    ops::inkling_gemm_inv(w, n, k, x, tokens, out, dtype, n, {}, bytes ? scratch(work, bytes) : nullptr, bytes, s);
}

void* bf16_alloc(WorkspaceArena& work, std::int64_t elements) {
    return work.alloc_bytes(static_cast<std::size_t>(elements) * 2).data;
}

} // namespace

family::DFlashGeometry dspark_dflash_geometry(const load::ArtifactConfig& config) {
    if (!config.dspark) { throw std::invalid_argument("inkling2: the artifact carries no dspark block"); }
    const load::DraftConfig& d = *config.dspark;
    const bool ok = d.layers == G::kLayers && d.hidden == G::kHidden && d.query_heads == G::kQHeads &&
                    d.kv_heads == G::kKvHeads && d.head_dim == G::kHeadDim && d.intermediate == G::kIntermediate &&
                    d.block_size == G::kBlock && d.markov_rank == G::kRank && d.sliding_window == G::kWindow &&
                    d.local_layers == std::vector<std::int32_t>{0, 2} &&
                    d.target_visits == std::vector<std::int32_t>{5, 23, 41, 48} &&
                    std::fabs(d.rope_theta - 1.0e6) < 1.0 && std::fabs(d.rms_eps - 1.0e-6) < 1e-12;
    if (!ok) { throw std::invalid_argument("inkling2: the artifact's dspark block is not Inkling-2-Mini's drafter"); }
    family::DFlashGeometry g;
    g.hidden          = G::kHidden;
    g.layers          = G::kLayers;
    g.local_layers    = G::kLayers; // DFlash's rings, unused by DSpark (draft KV = text-pool planes 48..50)
    g.intermediate    = G::kIntermediate;
    g.query_heads     = G::kQHeads;
    g.kv_heads        = G::kKvHeads;
    g.head_dim        = G::kHeadDim;
    g.local_capacity  = 64;
    g.mask_token      = 0; // DSpark's mask rows are the mask_embed vector, not a token
    g.block_size      = G::kBlock;
    g.max_context     = static_cast<std::int32_t>(std::min<std::int64_t>(d.max_position, 1048576));
    g.feature_layers  = G::kFeatures;
    g.feature_rows    = G::kFeatureRows;
    g.rms_epsilon     = G::kEps;
    g.rope_theta      = G::kRopeTheta;
    g.attention_scale = G::kScale;
    // Features named by the visit that produces them (visits.h kFeatureLabels): on one card a label only names its row;
    // on the 2-card split the family's feature sink uses it to tell which stage captures a feature and which imports it.
    static_assert(G::kFeatures == static_cast<int>(kFeatureLabels.size()));
    for (int f = 0; f < G::kFeatures; ++f) {
        g.target_feature_layers[static_cast<std::size_t>(f)] = kFeatureLabels[static_cast<std::size_t>(f)];
    }
    return g;
}

bool dspark_bf16_weights_forced() {
    static const bool forced = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_DSPARK_BF16_WEIGHTS");
        return raw != nullptr && raw[0] == '1';
    }();
    return forced;
}

bool dspark_fp8_weights_requested() {
    // OPT-IN (SUROGATE_SERVE_DSPARK_FP8_WEIGHTS=1) since M10: the BF16 drafter is the default again -- h2's drafter A/B on
    // M7 (perf/h2/m7-55c27326-1121-1757/dab): the FP8 route (W8A16 dspark_gemm_w8 + the FP8 fold) costs ~9.5 % per
    // stream at 8 streams (T0 105.9 vs 117.1, T1 78.5 vs 86.7 tok/s), equal at 1 stream, same tokens / round.
    // SUROGATE_SERVE_DSPARK_BF16_WEIGHTS=1 still forces BF16.
    static const bool fp8 = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_DSPARK_FP8_WEIGHTS");
        return raw != nullptr && raw[0] == '1';
    }();
    return fp8 && !dspark_bf16_weights_forced();
}

namespace {

/// out [n, tokens] (ldc) = W . x: the FP8 route (ops::dspark_gemm_w8) or the BF16 invariant GEMM.
void dense(WorkspaceArena& work, const DSparkDerived& d, const Weight& w, const DSparkW8& w8, std::int32_t n,
           std::int32_t k, const void* x, std::int32_t tokens, void* out, std::int32_t ldc, cudaStream_t s) {
    if (d.fp8) {
        ops::dspark_gemm_w8(w8.codes, w8.scales, n, k, x, tokens, out, ldc, s);
        return;
    }
    const std::size_t bytes = ops::inkling_gemm_inv_workspace_bytes(n, k, tokens, 1);
    ops::inkling_gemm_inv(bf16_weight(w, n, k, "drafter linear"), n, k, x, tokens, out, DType::BF16, ldc, {},
                          bytes ? scratch(work, bytes) : nullptr, bytes, s);
}

} // namespace

bool dspark_fp8_fold_requested() {
    // OPT-IN (SUROGATE_SERVE_DSPARK_FP8_FOLD=1) since M10, only with the FP8 weights (dspark_fp8_weights_requested); CPU
    // r1+r2: no acceptance loss. An FP8 fold over BF16 weights has no proof (review R5 minor 7): the flag alone is ignored,
    // with one line on stderr. SUROGATE_SERVE_DSPARK_BF16_FOLD=1 still forces the BF16 fold.
    static const bool fp8 = [] {
        const char* fp8_raw  = std::getenv("SUROGATE_SERVE_DSPARK_FP8_FOLD");
        const char* bf16_raw = std::getenv("SUROGATE_SERVE_DSPARK_BF16_FOLD");
        const bool asked = fp8_raw != nullptr && fp8_raw[0] == '1' && !(bf16_raw != nullptr && bf16_raw[0] == '1');
        if (asked && !dspark_fp8_weights_requested()) {
            std::fprintf(stderr, "dspark: SUROGATE_SERVE_DSPARK_FP8_FOLD=1 ignored: it needs SUROGATE_SERVE_DSPARK_FP8_WEIGHTS=1\n");
            return false;
        }
        return asked;
    }();
    return fp8;
}

namespace {
// The derived storage block of derive_dspark: alpha, mix norms, rnn_proj split, (BF16 fold), fold bias, inv_freq.
std::size_t dspark_storage_bytes(bool fp8_fold) {
    constexpr std::int64_t D = G::kHidden, V = G::kVocabPadded, R = G::kRank;
    const std::size_t o_norms = align256(16 * sizeof(float));
    const std::size_t o_sp    = align256(o_norms + D * 4 * 2);
    const std::size_t o_hf    = align256(o_sp + G::kRnnRows * G::kRnnStateCols * 2);
    const std::size_t o_fw    = align256(o_hf + G::kRnnRows * G::kRnnContextCols * 2);
    const std::size_t o_fb    = align256(o_fw + (fp8_fold ? 0 : V * R * 2));
    const std::size_t o_freq  = align256(o_fb + V * 2);
    return align256(o_freq + 64 * sizeof(float));
}
std::size_t dspark_fold_storage_bytes() {
    return align256(static_cast<std::size_t>(G::kVocabPadded) * G::kRank) +
           align256(static_cast<std::size_t>(G::kVocabPadded) * 4);
}
// The FP8 linears of one drafter layer, (rows n, columns k): q k v o k_ctx v_ctx gate up down.
constexpr std::int64_t kDraftLinearShapes[9][2] = {
    {G::kQRows, G::kHidden}, {G::kKvRows, G::kHidden}, {G::kKvRows, G::kHidden}, {G::kHidden, G::kQRows},
    {G::kKvRows, G::kHidden}, {G::kKvRows, G::kHidden}, {G::kIntermediate, G::kHidden},
    {G::kIntermediate, G::kHidden}, {G::kHidden, G::kIntermediate}};
} // namespace

std::size_t dspark_derived_device_bytes(bool fp8_weights, bool fp8_fold) {
    std::size_t total = dspark_storage_bytes(fp8_fold) + (fp8_fold ? dspark_fold_storage_bytes() : 0);
    if (fp8_weights) {
        std::size_t w8 = 0;
        for (int l = 0; l < G::kLayers; ++l) {
            for (const auto& nk : kDraftLinearShapes) {
                w8 = align256(w8 + static_cast<std::size_t>(nk[0]) * static_cast<std::size_t>(nk[1]));
                w8 = align256(w8 + static_cast<std::size_t>(nk[0]) * sizeof(float));
            }
        }
        total += w8;
    }
    return total;
}

void derive_dspark(const load::InklingWeights& w, DSparkDerived& out, bool fp8_weights, bool fp8_fold) {
    if (!w.has_draft) { throw std::logic_error("inkling2 DSpark: the artifact's drafter is not bound"); }
    const load::InklingDraftWeights& d = w.draft;
    if (d.layers.size() != static_cast<std::size_t>(G::kLayers)) {
        throw std::logic_error("inkling2 DSpark: the drafter has not 3 layers");
    }
    constexpr std::int64_t D = G::kHidden, V = G::kVocabPadded, R = G::kRank;
    const std::size_t o_alpha = 0;
    const std::size_t o_norms = align256(o_alpha + 16 * sizeof(float));
    const std::size_t o_sp    = align256(o_norms + D * 4 * 2);
    const std::size_t o_hf    = align256(o_sp + G::kRnnRows * G::kRnnStateCols * 2);
    const std::size_t o_fw    = align256(o_hf + G::kRnnRows * G::kRnnContextCols * 2);
    // the BF16 fold stays resident only without the FP8 fold (else it is a load-time temporary)
    const std::size_t o_fb    = align256(o_fw + (fp8_fold ? 0 : V * R * 2));
    const std::size_t o_freq  = align256(o_fb + V * 2);
    const std::size_t total   = align256(o_freq + 64 * sizeof(float));
    out.storage = DeviceBuffer(total);
    auto* base  = static_cast<std::byte*>(out.storage.p);

    // fuse alpha: softmax(fuse_weight.float(), -1) in fp32 (refresh_derived_weights)
    const void* fw_dev = bf16_tensor(d.fuse_weight, 16, "mtp/fuse_weight");
    std::uint16_t fw_host[16];
    CUDA_CHECK(cudaMemcpy(fw_host, fw_dev, sizeof fw_host, cudaMemcpyDeviceToHost));
    float alpha[16];
    for (int r = 0; r < 4; ++r) {
        float m = -INFINITY;
        for (int f = 0; f < 4; ++f) { m = std::max(m, bf16_to_float(fw_host[r * 4 + f])); }
        float e[4], s = 0.0F;
        for (int f = 0; f < 4; ++f) {
            e[f] = std::exp(bf16_to_float(fw_host[r * 4 + f]) - m);
            s += e[f];
        }
        for (int f = 0; f < 4; ++f) { alpha[r * 4 + f] = e[f] / s; }
    }
    CUDA_CHECK(cudaMemcpy(base + o_alpha, alpha, sizeof alpha, cudaMemcpyHostToDevice));
    // mix norms [2304, 4]
    for (int r = 0; r < 4; ++r) {
        const Tensor& n = r < 3 ? d.layers[static_cast<std::size_t>(r)].feature_norm : d.markov_feature_norm;
        CUDA_CHECK(cudaMemcpy(base + o_norms + static_cast<std::size_t>(r) * D * 2, bf16_tensor(n, D, "feature_norm"),
                              D * 2, cudaMemcpyDeviceToDevice));
    }
    // rnn_proj [768, 5120] -> [768, 512] and [768, 4608]
    const auto* rnn = static_cast<const std::byte*>(
        bf16_weight(d.markov_rnn_proj, G::kRnnRows, G::kRnnStateCols + G::kRnnContextCols, "mtp/markov/rnn_proj"));
    const std::size_t pitch = static_cast<std::size_t>(G::kRnnStateCols + G::kRnnContextCols) * 2;
    CUDA_CHECK(cudaMemcpy2D(base + o_sp, G::kRnnStateCols * 2, rnn, pitch, G::kRnnStateCols * 2, G::kRnnRows,
                            cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(base + o_hf, G::kRnnContextCols * 2, rnn + G::kRnnStateCols * 2, pitch,
                            G::kRnnContextCols * 2, G::kRnnRows, cudaMemcpyDeviceToDevice));
    // the fold (fp64 on the device)
    DeviceBuffer fold_temp(fp8_fold ? static_cast<std::size_t>(V * R * 2) : 0);
    std::byte* fold_dst = fp8_fold ? static_cast<std::byte*>(fold_temp.p) : base + o_fw;
    ops::dspark_fold(bf16_weight(w.globals.output_head, G::kVocabPadded, G::kHidden, "unembed"), G::kVocabPadded,
                     nullptr, bf16_weight(d.markov_hidden_out, G::kHidden, G::kRank, "mtp/markov/hidden_out"),
                     bf16_tensor(d.markov_hidden_out_bias, D, "mtp/markov/hidden_out_bias"), fold_dst, base + o_fb,
                     nullptr);
    // RoPE
    float inv_freq[64];
    ops::dspark_rope_inv_freq(G::kRopeTheta, inv_freq);
    CUDA_CHECK(cudaMemcpy(base + o_freq, inv_freq, sizeof inv_freq, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    out.alpha     = reinterpret_cast<const float*>(base + o_alpha);
    out.mix_norms = base + o_norms;
    out.w_sp      = base + o_sp;
    out.w_hf      = base + o_hf;
    out.fold_w    = fp8_fold ? nullptr : base + o_fw;
    if (fp8_fold) {
        // the FP8 fold: per-row e4m3 of the BF16 fold (the fp64 product rounded once, then quantised as the M1 lever)
        out.fold_storage = DeviceBuffer(align256(static_cast<std::size_t>(V) * R) + align256(static_cast<std::size_t>(V) * 4));
        auto* fs         = static_cast<std::byte*>(out.fold_storage.p);
        out.fold_codes   = reinterpret_cast<const std::uint8_t*>(fs);
        out.fold_scales  = reinterpret_cast<const float*>(fs + align256(static_cast<std::size_t>(V) * R));
        ops::dspark_quantize_rows_e4m3(fold_dst, static_cast<std::int32_t>(V), static_cast<std::int32_t>(R),
                                       const_cast<std::uint8_t*>(out.fold_codes), const_cast<float*>(out.fold_scales),
                                       nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        out.fp8_fold = true;
    }
    out.fold_b    = base + o_fb;
    out.inv_freq  = reinterpret_cast<const float*>(base + o_freq);
    if (fp8_weights) {
        // per-row e4m3 + scale of the 9 dense linears of each layer (q k v o k_ctx v_ctx gate up down)
        std::size_t total = 0;
        const auto plan = [&](const Weight& wt) {
            total = align256(total + static_cast<std::size_t>(wt.n) * wt.k);
            total = align256(total + static_cast<std::size_t>(wt.n) * sizeof(float));
        };
        for (const auto& L : d.layers) {
            for (const Weight* wt : {&L.q, &L.k, &L.v, &L.o, &L.k_ctx, &L.v_ctx, &L.gate, &L.up, &L.down}) { plan(*wt); }
        }
        out.w8_storage = DeviceBuffer(total);
        auto* p        = static_cast<std::byte*>(out.w8_storage.p);
        std::size_t at = 0;
        const auto quantize = [&](const Weight& wt) {
            if (wt.qtype != QType::BF16_CTRL || wt.qdata == nullptr) {
                throw std::logic_error("inkling2 DSpark: an FP8 source linear is not BF16");
            }
            DSparkW8 q;
            q.n      = wt.n;
            q.k      = wt.k;
            q.codes  = reinterpret_cast<const std::uint8_t*>(p + at);
            at       = align256(at + static_cast<std::size_t>(wt.n) * wt.k);
            q.scales = reinterpret_cast<const float*>(p + at);
            at       = align256(at + static_cast<std::size_t>(wt.n) * sizeof(float));
            ops::dspark_quantize_rows_e4m3(wt.qdata, wt.n, wt.k, const_cast<std::uint8_t*>(q.codes),
                                           const_cast<float*>(q.scales), nullptr);
            return q;
        };
        for (std::size_t l = 0; l < d.layers.size(); ++l) {
            const auto& L = d.layers[l];
            out.w8[l] = DSparkLayerW8{quantize(L.q),     quantize(L.k),    quantize(L.v),
                                      quantize(L.o),     quantize(L.k_ctx), quantize(L.v_ctx),
                                      quantize(L.gate),  quantize(L.up),   quantize(L.down)};
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        out.fp8 = true;
    }
    // The registry booked dspark_derived_device_bytes before the KV was sized: the allocations must be exactly that.
    const std::size_t allocated = out.storage.bytes + out.fold_storage.bytes + out.w8_storage.bytes;
    if (allocated != dspark_derived_device_bytes(fp8_weights, fp8_fold)) {
        throw std::logic_error("inkling2 DSpark: derived bytes " + std::to_string(allocated) + " != booked " +
                               std::to_string(dspark_derived_device_bytes(fp8_weights, fp8_fold)));
    }
    out.ready     = true;
}

std::size_t dspark_append_workspace_bytes(std::int32_t columns) {
    const std::int64_t c = std::min<std::int32_t>(columns, kAppendChunk);
    return align256(static_cast<std::size_t>(G::kHidden) * c * G::kMixRows * 2) +
           align256(static_cast<std::size_t>(2 * G::kKvRows) * c * 2) + 4096;
}

bool dspark_naive_attention_forced() {
    static const bool forced = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_DSPARK_NAIVE_ATTN");
        return raw != nullptr && raw[0] == '1';
    }();
    return forced;
}

std::size_t dspark_propose_workspace_bytes(std::int32_t batch, std::uint32_t max_visible_keys) {
    const std::int64_t n = static_cast<std::int64_t>(G::kBlock) * batch;
    const std::int64_t k = static_cast<std::int64_t>(G::kMaxDrafts) * batch;
    std::size_t b = 0;
    b += align256(n * 4);                                    // positions
    b += 3 * align256(static_cast<std::size_t>(G::kHidden) * n * 2); // x, resid, h
    b += align256(static_cast<std::size_t>(G::kQkvRows) * n * 2);
    b += align256(static_cast<std::size_t>(G::kQRows) * n * 2) * 2; // q, attn
    b += align256(static_cast<std::size_t>(2 * G::kIntermediate) * n * 2);
    b += align256(static_cast<std::size_t>(G::kIntermediate) * n * 2);
    b += align256(static_cast<std::size_t>(G::kHidden) * n * 2);     // final hidden
    b += align256(static_cast<std::size_t>(G::kVocabPadded) * n * 2); // base logits
    b += align256(static_cast<std::size_t>(G::kRnnContextCols) * k * 2);
    b += align256(static_cast<std::size_t>(G::kRnnRows) * k * 4);
    b += 3 * align256(static_cast<std::size_t>(G::kRank) * batch * 2);
    b += 2 * align256(static_cast<std::size_t>(ops::dspark_vocab_tiles(G::kVocab)) * batch * 4);
    b += align256(n * 4); // query positions (the draft attention)
    // the draft attention (its split partials), the larger of the local and the global plan
    const ops::GqaExecutionEnvelope envelope{0, std::max<std::uint32_t>(max_visible_keys, G::kBlock)};
    b += align256(std::max(ops::inkling_draft_attention_workspace_bytes(G::kBlockWindow, G::kBlock, batch, envelope),
                           ops::inkling_draft_attention_workspace_bytes(0, G::kBlock, batch, envelope)));
    return b + 8192;
}

void dspark_append(const load::InklingWeights& w, const DSparkDerived& d, const DSparkKv& kv, const DSparkAppend& a,
                   WorkspaceArena& work, cudaStream_t s) {
    if (!d.ready || a.features == nullptr || a.positions == nullptr || a.table_rows == nullptr || a.f_rnn == nullptr ||
        a.width <= 0 || a.batch <= 0 || a.feature_stride < G::kFeatureRows) {
        throw std::invalid_argument("inkling2 DSpark append: incomplete call");
    }
    if (a.width > kAppendChunk && (a.batch > 1 || a.counts != nullptr)) {
        throw std::invalid_argument("inkling2 DSpark append: a wider append than 1,024 columns must be one lane's "
                                    "fully committed prefill chunk (counts null)");
    }
    const load::InklingDraftWeights& dw = w.draft;
    // A prefill chunk (B = 1, all columns committed) runs in passes of <= 1,024 columns; f_rnn from the last.
    for (std::int32_t c0 = 0; c0 < a.width; c0 += kAppendChunk) {
        auto scope            = work.scope();
        const std::int32_t W  = a.batch == 1 ? std::min(kAppendChunk, a.width - c0) : a.width;
        const std::int32_t N  = W * a.batch;
        const auto* features  = static_cast<const std::uint16_t*>(a.features) + static_cast<std::int64_t>(c0) * a.feature_stride;
        const std::int32_t* positions = a.positions + c0;
        auto* mix = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(G::kHidden) * N * G::kMixRows));
        ops::dspark_feature_mix(features, a.feature_stride, N, d.alpha, d.mix_norms, G::kMixRows, G::kEps, mix, s);
        auto* kvbuf = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(2 * G::kKvRows) * N));
        for (int l = 0; l < G::kLayers; ++l) {
            const load::InklingDraftLayerWeights& L = dw.layers[static_cast<std::size_t>(l)];
            const void* x = mix + static_cast<std::int64_t>(l) * G::kHidden * N;
            if (d.fp8) {
                const DSparkLayerW8& Q = d.w8[static_cast<std::size_t>(l)];
                dense(work, d, L.k_ctx, Q.k_ctx, G::kKvRows, G::kHidden, x, N, kvbuf, 2 * G::kKvRows, s);
                dense(work, d, L.v_ctx, Q.v_ctx, G::kKvRows, G::kHidden, x, N, kvbuf + G::kKvRows, 2 * G::kKvRows, s);
            } else {
                const ops::InklingGemmProblem problems[2] = {
                    {bf16_weight(L.k_ctx, G::kKvRows, G::kHidden, "k_ctx"), G::kKvRows, x, kvbuf, 2 * G::kKvRows},
                    {bf16_weight(L.v_ctx, G::kKvRows, G::kHidden, "v_ctx"), G::kKvRows, x, kvbuf + G::kKvRows,
                     2 * G::kKvRows},
                };
                ops::inkling_gemm_inv_group(problems, G::kHidden, N, DType::BF16, s);
            }
            ops::dspark_head_norm_rope(kvbuf, 2 * G::kKvRows, G::kKvHeads, N,
                                       bf16_tensor(L.k_ctx_norm, G::kHeadDim, "k_ctx_norm"), positions, d.inv_freq,
                                       G::kEps, kvbuf, 2 * G::kKvRows, s);
            ops::dspark_kv_write(kvbuf, kvbuf + G::kKvRows, 2 * G::kKvRows, positions, W, a.batch, a.counts,
                                 a.table_rows, kv.planes[static_cast<std::size_t>(l)], s);
        }
        if (c0 + W >= a.width) {
            ops::dspark_store_last_column(mix + static_cast<std::int64_t>(3) * G::kHidden * N, W, a.batch, a.counts,
                                          a.lanes, a.f_rnn, s);
        }
        if (a.batch > 1) { break; }
    }
}

void dspark_propose(const load::InklingWeights& w, const DSparkDerived& d, const DSparkKv& kv, const DSparkPropose& p,
                    WorkspaceArena& work, cudaStream_t s) {
    if (!d.ready || p.anchors == nullptr || p.anchor_positions == nullptr || p.table_rows == nullptr ||
        p.f_rnn == nullptr || p.drafts == nullptr || p.batch <= 0 || p.k < 1 || p.k > G::kMaxDrafts) {
        throw std::invalid_argument("inkling2 DSpark propose: incomplete call");
    }
    if (p.batch > 16) { throw std::invalid_argument("inkling2 DSpark propose: at most 16 lanes per call"); }
    auto scope = work.scope();
    const load::InklingDraftWeights& dw = w.draft;
    const std::int32_t B = p.batch, N = G::kBlock * B, K = p.k;
    auto* pos = static_cast<std::int32_t*>(work.alloc_bytes(static_cast<std::size_t>(N) * 4).data);
    ops::dspark_block_positions(p.anchor_positions, B, pos, s);
    const bool naive = kv.naive_attention || dspark_naive_attention_forced();
    auto* qpos       = static_cast<std::int32_t*>(work.alloc_bytes(static_cast<std::size_t>(N) * 4).data);
    if (!naive) { ops::dspark_block_query_positions(p.anchor_positions, B, qpos, s); }
    auto* x     = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(G::kHidden) * N));
    auto* resid = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(G::kHidden) * N));
    auto* h     = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(G::kHidden) * N));
    auto* qkv   = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(G::kQkvRows) * N));
    auto* q     = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(G::kQRows) * N));
    auto* attn  = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(G::kQRows) * N));
    auto* gu    = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(2 * G::kIntermediate) * N));
    auto* act   = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(G::kIntermediate) * N));
    ops::dspark_embed_block(p.anchors, B, w.globals.embedding.qdata, w.globals.embedding.n,
                            bf16_tensor(w.globals.embedding_norm, G::kHidden, "embed_norm"),
                            bf16_tensor(dw.mask_embed, G::kHidden, "mask_embed"), G::kEps, x, s);
    for (int l = 0; l < G::kLayers; ++l) {
        const load::InklingDraftLayerWeights& L = dw.layers[static_cast<std::size_t>(l)];
        const void* in_norm = bf16_tensor(L.input_norm, G::kHidden, "input_norm");
        // pre-norm (layer 0: the block rows are the residual)
        ops::dspark_add_rms_norm(l == 0 ? nullptr : x, l == 0 ? x : resid, in_norm, G::kEps, N, resid, h, s);
        const DSparkLayerW8& Q = d.w8[static_cast<std::size_t>(l)];
        if (d.fp8) {
            dense(work, d, L.q, Q.q, G::kQRows, G::kHidden, h, N, qkv, G::kQkvRows, s);
            dense(work, d, L.k, Q.k, G::kKvRows, G::kHidden, h, N, qkv + G::kQRows, G::kQkvRows, s);
            dense(work, d, L.v, Q.v, G::kKvRows, G::kHidden, h, N, qkv + G::kQRows + G::kKvRows, G::kQkvRows, s);
        } else {
            const ops::InklingGemmProblem qkv_problems[3] = {
                {bf16_weight(L.q, G::kQRows, G::kHidden, "q"), G::kQRows, h, qkv, G::kQkvRows},
                {bf16_weight(L.k, G::kKvRows, G::kHidden, "k"), G::kKvRows, h, qkv + G::kQRows, G::kQkvRows},
                {bf16_weight(L.v, G::kKvRows, G::kHidden, "v"), G::kKvRows, h, qkv + G::kQRows + G::kKvRows,
                 G::kQkvRows},
            };
            ops::inkling_gemm_inv_group(qkv_problems, G::kHidden, N, DType::BF16, s);
        }
        ops::dspark_head_norm_rope(qkv, G::kQkvRows, G::kQHeads, N, bf16_tensor(L.q_norm, G::kHeadDim, "q_norm"), pos,
                                   d.inv_freq, G::kEps, q, G::kQRows, s);
        ops::dspark_head_norm_rope(qkv + G::kQRows, G::kQkvRows, G::kKvHeads, N,
                                   bf16_tensor(L.k_norm, G::kHeadDim, "k_norm"), pos, d.inv_freq, G::kEps,
                                   qkv + G::kQRows, G::kQkvRows, s);
        const ops::DSparkPagedPlane& plane = kv.planes[static_cast<std::size_t>(l)];
        ops::dspark_kv_write(qkv + G::kQRows, qkv + G::kQRows + G::kKvRows, G::kQkvRows, pos, G::kBlock, B, nullptr,
                             p.table_rows, plane, s);
        const std::int32_t window = l == 1 ? 0 : G::kBlockWindow;
        if (naive) {
            ops::dspark_block_attention(q, G::kQRows, p.anchor_positions, p.table_rows, B, window, G::kScale, plane, attn,
                                        s);
        } else {
            // the attention's draft mode: all 8 rows of a lane at P + 7, keys from the plane only
            ops::InklingAttentionRows rows;
            rows.positions     = Tensor(qpos, DType::I32, {G::kBlock, B});
            rows.kv_table_rows = Tensor(const_cast<std::int32_t*>(p.table_rows), DType::I32, {B});
            rows.envelope      = ops::GqaExecutionEnvelope{0, std::max<std::uint32_t>(kv.max_visible_keys, G::kBlock)};
            const Tensor qt(q, DType::BF16, {G::kHeadDim, G::kQHeads, N});
            Tensor ot(attn, DType::BF16, {G::kHeadDim, G::kQHeads, N});
            ops::inkling_draft_attention(qt, window, rows, kv.views[static_cast<std::size_t>(l)], work, ot, s);
        }
        dense(work, d, L.o, Q.o, G::kHidden, G::kQRows, attn, N, x, G::kHidden, s);
        ops::dspark_add_rms_norm(x, resid, bf16_tensor(L.post_attention_norm, G::kHidden, "post_attention_norm"),
                                 G::kEps, N, resid, h, s);
        if (d.fp8) {
            dense(work, d, L.gate, Q.gate, G::kIntermediate, G::kHidden, h, N, gu, 2 * G::kIntermediate, s);
            dense(work, d, L.up, Q.up, G::kIntermediate, G::kHidden, h, N, gu + G::kIntermediate, 2 * G::kIntermediate, s);
        } else {
            const ops::InklingGemmProblem gu_problems[2] = {
                {bf16_weight(L.gate, G::kIntermediate, G::kHidden, "gate"), G::kIntermediate, h, gu,
                 2 * G::kIntermediate},
                {bf16_weight(L.up, G::kIntermediate, G::kHidden, "up"), G::kIntermediate, h, gu + G::kIntermediate,
                 2 * G::kIntermediate},
            };
            ops::inkling_gemm_inv_group(gu_problems, G::kHidden, N, DType::BF16, s);
        }
        ops::dspark_silu_mul(gu, N, act, s);
        dense(work, d, L.down, Q.down, G::kHidden, G::kIntermediate, act, N, x, G::kHidden, s);
    }
    auto* hidden = p.hidden_out != nullptr ? static_cast<std::uint16_t*>(p.hidden_out)
                                           : static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(G::kHidden) * N));
    ops::dspark_add_rms_norm(x, resid, bf16_tensor(dw.final_norm, G::kHidden, "final_norm"), G::kEps, N, nullptr, hidden,
                             s);
    // base logits of all 8 rows (the head GEMM reads the 0.93 GB unembed once whatever the width)
    auto* base = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(G::kVocabPadded) * N));
    gemm(work, bf16_weight(w.globals.output_head, G::kVocabPadded, G::kHidden, "unembed"), G::kVocabPadded, G::kHidden,
         hidden, N, base, DType::BF16, s);
    // the Markov head
    auto* xin = static_cast<std::uint16_t*>(bf16_alloc(work, static_cast<std::int64_t>(G::kRnnContextCols) * K * B));
    ops::dspark_markov_pack(hidden, p.f_rnn, p.lanes, K, B, xin, s);
    auto* ctx = static_cast<float*>(work.alloc_bytes(static_cast<std::size_t>(G::kRnnRows) * K * B * 4).data);
    gemm(work, d.w_hf, G::kRnnRows, G::kRnnContextCols, xin, K * B, ctx, DType::FP32, s);
    void* state[2] = {bf16_alloc(work, static_cast<std::int64_t>(G::kRank) * B),
                      bf16_alloc(work, static_cast<std::int64_t>(G::kRank) * B)}; // ping-pong across steps
    void* latent   = bf16_alloc(work, static_cast<std::int64_t>(G::kRank) * B);
    const std::int32_t tiles = ops::dspark_vocab_tiles(G::kVocab);
    auto* part_v = static_cast<float*>(work.alloc_bytes(static_cast<std::size_t>(tiles) * B * 4).data);
    auto* part_i = static_cast<std::int32_t*>(work.alloc_bytes(static_cast<std::size_t>(tiles) * B * 4).data);
    const void* rnn_bias = bf16_tensor(dw.markov_rnn_proj_bias, G::kRnnRows, "rnn_proj_bias");
    const void* w1       = bf16_weight(dw.markov_w1, G::kVocabPadded, G::kRank, "w1");
    const std::int32_t stride = p.draft_stride > 0 ? p.draft_stride : K;
    if (stride < K) { throw std::invalid_argument("inkling2 DSpark propose: draft stride below k"); }
    for (int j = 0; j < K; ++j) {
        const std::int32_t* prev = j == 0 ? p.anchors : p.drafts + (j - 1);
        ops::dspark_markov_gate(ctx, j, K, B, rnn_bias, d.w_sp, w1, prev, j == 0 ? 1 : stride,
                                j == 0 ? nullptr : state[(j + 1) & 1], state[j & 1], latent, s);
        ops::dspark_markov_vocab(base, G::kVocabPadded, 1 + j, B, d.fp8_fold ? nullptr : d.fold_w, d.fold_b, latent,
                                 G::kVocab, nullptr, part_v, part_i, nullptr, nullptr, s,
                                 d.fp8_fold ? d.fold_codes : nullptr, d.fp8_fold ? d.fold_scales : nullptr,
                                 p.sampling, p.sampling != nullptr ? p.anchor_positions : nullptr, j);
        ops::dspark_markov_pick(part_v, part_i, tiles, B, j, stride, p.drafts, s);
    }
}

// ---- the Variant's DSpark leaves (family/impl/runtime/dspark_round.h) ----

std::vector<std::int32_t> Variant::extra_kv_planes(const family::TextGeometry& geometry,
                                                   const family::StartupFeatures& features) {
    std::vector<std::int32_t> planes;
    if (features.dflash() && geometry.dflash.layers > 0) { planes = {G::kBlockWindow, 0, G::kBlockWindow}; }
    if (loopcd_settings().enabled) {
        // LoopCD's reference visits (loopcd.h): the coda's six layers again, the pool's last six planes (after the
        // drafter's), with the windows of visits 42..47.
        for (int j = 0; j < kLoopcdVisits; ++j) {
            planes.push_back(kVisits[static_cast<std::size_t>(TextConfig::layers - kLoopcdVisits + j)].global
                                 ? 0
                                 : TextConfig::window);
        }
    }
    return planes;
}

namespace {

const InklingModel& dspark_model(const RuntimeModelView& model) {
    if (model.full_layers.empty() || model.full_layers.front().projection.model == nullptr ||
        model.full_layers.front().projection.model->dspark == nullptr) {
        throw std::logic_error("inkling2: the DSpark drafter is not loaded");
    }
    return *model.full_layers.front().projection.model;
}

DSparkKv dspark_kv(const family::DSparkRuntime& rt) {
    // On the 2-card split the last stage's pool holds the planes of visits 30..47 then 48..50 (layers() counts the
    // stage's planes, not the global index): batch_layer_view checks the global plane against the stage's range.
    if (rt.text_kv == nullptr || rt.text_kv->layers() < static_cast<std::uint32_t>(G::kLayers)) {
        throw std::logic_error("inkling2 DSpark: the text pool carries no draft planes (48..50)");
    }
    DSparkKv kv;
    for (int l = 0; l < G::kLayers; ++l) {
        const PagedKVBatchLayerView v = rt.text_kv->batch_layer_view(Variant::dspark_first_plane + static_cast<std::uint32_t>(l));
        if (v.head_dim != G::kHeadDim || v.num_kv_heads != G::kKvHeads ||
            (v.dtype != DType::FP8_E4M3FN && v.dtype != DType::BF16) || v.quant_group != 0) {
            throw std::logic_error("inkling2 DSpark: a draft plane is not [64, 8, 128] FP8 / BF16");
        }
        kv.views[static_cast<std::size_t>(l)] = v;
        kv.planes[static_cast<std::size_t>(l)] =
            ops::DSparkPagedPlane{v.k_pages.data, v.v_pages.data, static_cast<const std::int32_t*>(v.block_tables.data),
                                  v.block_tables.ne[0], v.dtype == DType::FP8_E4M3FN,
                                  static_cast<std::int32_t>(rt.text_kv->max_context())};
    }
    kv.max_visible_keys = rt.text_kv->max_context();
    kv.naive_attention  = kv.views[0].dtype != DType::FP8_E4M3FN; // the draft kernel reads FP8 planes only
    return kv;
}

} // namespace

void Variant::dspark_append(const ModelView& model, const family::DSparkRuntime& rt,
                            const family::DSparkAppendArgs& args) {
    const InklingModel& im = dspark_model(model);
    if (rt.work == nullptr || rt.f_rnn.data == nullptr) { throw std::logic_error("inkling2 DSpark append: no runtime state"); }
    DSparkAppend a;
    a.features       = args.features.data;
    a.feature_stride = args.features.ne[0];
    a.width          = args.features.ne[1];
    a.batch          = args.features.ne[2];
    a.positions      = static_cast<const std::int32_t*>(args.positions.data);
    a.counts         = args.all_committed ? nullptr : static_cast<const std::int32_t*>(args.counts.data);
    a.lanes          = static_cast<const std::int32_t*>(args.lanes.data);
    a.table_rows     = static_cast<const std::int32_t*>(args.table_rows.data);
    a.f_rnn          = rt.f_rnn.data;
    detail::dspark_append(*im.weights, *im.dspark, dspark_kv(rt), a, *rt.work, rt.stream);
}

void Variant::dspark_propose(const ModelView& model, const family::DSparkRuntime& rt,
                             const family::DSparkProposeArgs& args) {
    const InklingModel& im = dspark_model(model);
    if (rt.work == nullptr || rt.f_rnn.data == nullptr) { throw std::logic_error("inkling2 DSpark propose: no runtime state"); }
    DSparkPropose p;
    p.anchors          = static_cast<const std::int32_t*>(args.anchors.data);
    p.anchor_positions = static_cast<const std::int32_t*>(args.anchor_positions.data);
    p.lanes            = static_cast<const std::int32_t*>(args.lanes.data);
    p.table_rows       = static_cast<const std::int32_t*>(args.table_rows.data);
    p.f_rnn            = rt.f_rnn.data;
    p.drafts           = static_cast<std::int32_t*>(args.drafts.data);
    p.draft_stride     = args.drafts.ne[0];
    p.k                = args.k;
    // Coupled sampled drafts (dspark.h) unless SUROGATE_SERVE_DSPARK_SAMPLED_DRAFTS=0 (greedy drafts, A/B).
    static const bool sampled_drafts = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_DSPARK_SAMPLED_DRAFTS");
        return raw == nullptr || raw[0] == '\0' || std::atoi(raw) != 0;
    }();
    p.sampling = sampled_drafts ? args.sampling : nullptr;
    // The vocabulary pass keeps 16 lanes' latents in shared memory: wider rounds run in groups of 16.
    for (std::int32_t b0 = 0; b0 < args.batch; b0 += 16) {
        p.batch = std::min<std::int32_t>(16, args.batch - b0);
        DSparkPropose g = p;
        g.anchors += b0;
        g.anchor_positions += b0;
        if (g.lanes != nullptr) { g.lanes += b0; }
        g.table_rows += b0;
        g.drafts += static_cast<std::int64_t>(b0) * p.draft_stride;
        if (g.sampling != nullptr) { g.sampling += b0; }
        detail::dspark_propose(*im.weights, *im.dspark, dspark_kv(rt), g, *rt.work, rt.stream);
    }
}

std::uint8_t Variant::dspark_content_kind(std::span<const TokenId> ledger) {
    // Inkling-2's content markers (model/tokenizer.json): <|content_thinking|> 200008, <|content_invoke_tool_json|>
    // 200049, <|content_text|> 200004; <|message_model|> 200001 opens a model message (structure until a marker).
    constexpr TokenId kThinking = 200008, kToolJson = 200049, kText = 200004, kMessageModel = 200001;
    constexpr std::size_t kScan = 4096;
    const std::size_t n = ledger.size();
    for (std::size_t i = 0; i < std::min(n, kScan); ++i) {
        const TokenId t = ledger[n - 1 - i];
        if (t == kToolJson) { return 0; }
        if (t == kMessageModel) { return 1; }
        if (t == kThinking) { return 2; }
        if (t == kText) { return 3; }
    }
    return 2; // no marker within reach: a long block of unknown kind -- thinking
}

std::uint32_t Variant::dspark_draft_cap(std::span<const TokenId> ledger, const family::DSparkLaneStats* stats) {
    const std::uint8_t kind = dspark_content_kind(ledger);
    if (stats == nullptr) {
        // static per content kind (a CPU k-policy study): tool-call JSON 7, structure 7, thinking 3, text 2
        constexpr std::uint32_t kStatic[4] = {7, 7, 3, 2};
        return kStatic[kind];
    }
    // Per lane: the k of {0, 1, 2, 3, 7} with the most committed tokens per unit of round time,
    //   E_k = 1 + a + ... + a^k (per-position acceptance a, geometric), cost_k = the 1-stream round time of window k,
    // with a = this request's own acceptance on this content kind (accepted drafts / (accepted drafts + rejecting
    // rounds), decayed), shrunk to the kind's prior by kPriorTrials pseudo-trials. Priors: the engine's real rounds
    // (temperature 0) and a CPU study of the round dynamics: tool JSON ~0.92, structure
    // ~0.85, thinking ~0.70, answer text ~0.52. Costs (ms, 1 x 3k, graphs, T0; regression of decode seconds
    // on rounds per window): k2 9.68, k3 10.44, k7 13.69; plain decode 6.1 (+0.5 for the context append at k 0); k1
    // interpolated. Break-even vs k 0 near a = 0.42; k 7 overtakes k 3 near a = 0.76.
    constexpr float kPrior[4]    = {0.92f, 0.85f, 0.70f, 0.52f};
    constexpr float kPriorTrials = 8.0f;
    constexpr std::uint32_t kWindows[5] = {0, 1, 2, 3, 7};
    constexpr float kCost[5]            = {6.6f, 8.9f, 9.68f, 10.44f, 13.69f};
    const float a = std::clamp((stats->accepted[kind] + kPriorTrials * kPrior[kind]) /
                                   (stats->accepted[kind] + stats->stopped[kind] + kPriorTrials),
                               0.0f, 0.999f);
    std::uint32_t best = kWindows[0];
    float best_rate    = 1.0f / kCost[0];
    for (int w = 1; w < 5; ++w) {
        float e = 1.0f, p = 1.0f;
        for (std::uint32_t i = 0; i < kWindows[w]; ++i) {
            p *= a;
            e += p;
        }
        const float rate = e / kCost[w];
        if (rate > best_rate) {
            best      = kWindows[w];
            best_rate = rate;
        }
    }
    return best;
}

std::size_t Variant::dspark_append_workspace_bytes(std::int32_t columns) {
    return detail::dspark_append_workspace_bytes(columns);
}

std::size_t Variant::dspark_propose_workspace_bytes(std::int32_t batch, std::uint32_t max_visible_keys) {
    return detail::dspark_propose_workspace_bytes(std::min<std::int32_t>(batch, 16), max_visible_keys);
}

} // namespace sinfer::targets::inkling2::detail
