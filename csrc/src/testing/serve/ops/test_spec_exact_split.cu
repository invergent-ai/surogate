// ENGINE-INKLING lane e2 (agent-EN-e2c, 2026-11-21): the split route for untruncated stochastic verify rows
// (speculative_exact_max / partial / finalize kernels) through the whole ops::speculative_accept_greedy_drafts op.
// agent-EN's conditions (17:2x) and what this test checks:
//   1. INVARIANCE: a row's committed tokens, counts, lengths, anchors and token counts do not depend on the batch, the
//      row's place in it, the draft width K of the call (beyond the row's extent) or the other rows; and a call
//      repeated gives the same bits (determinism under per-(request, position) seeds).
//   2. EXACTNESS: over many seeds, the tokens the round commits follow the target distribution: the first token ~ p_0,
//      and, given drafts d_0..d_{j-1} were committed, the token at column j ~ p_j (j = extent: the bonus column).
//      Chi-square per column against p_j (alpha 1e-3) on peaked, medium, flat rows, a low-probability draft, T = 0.7,
//      min_p; a POWER control re-tests every column against p_j with 1 % of the mass moved between two bins (must
//      fail where the samples allow, to show the test resolves an error of that size).
//   3. OLD vs NEW: every row's result must equal the one-CTA wide kernel's (any difference FAILS; agent-EN 17:5x), at
//      Inkling's shape, W4g's real vocabulary (248,320, BF16) and the older Qwen3 shape (151,669 / 151,936).
// `bench`: the op's time per call, old vs split, B = 1 / 8, K = 7, every draft accepted (8 columns drawn).

#include "api/ops/speculative_round.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ops/launcher/speculative_round.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace sinfer;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Logits [physical, cols, B]: smooth noise (or quantised for ties), a peak of height peaks[b * cols + c] at the
// draft drafts[b * K + c] (c < K), padding rows past the domain at 1000 (never drawn).
template <typename T>
__global__ void gen_logits(T* out, int B, int cols, int K, int physical, int domain, unsigned long long seed,
                           const int* drafts, const float* peaks, const int* quantised) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(B) * cols * physical) { return; }
    const int v = static_cast<int>(i % physical);
    const int c = static_cast<int>((i / physical) % cols);
    const int b = static_cast<int>(i / (static_cast<long long>(physical) * cols));
    const unsigned long long h =
        mix64(seed ^ (static_cast<unsigned long long>(b * cols + c) << 40) ^ static_cast<unsigned long long>(v));
    const float u0 = static_cast<float>(h & 0xffffff) / 16777216.0f, u1 = static_cast<float>((h >> 24) & 0xffffff) / 16777216.0f;
    float x = quantised[b] ? rintf((u0 - 0.5f) * 16.0f) * 0.5f
                           : (u0 + u1 - 1.0f) * 6.0f + (((h >> 48) & 0x3ff) == 0 ? 8.0f : 0.0f);
    if (c < K && v == drafts[b * K + c]) { x += peaks[b * cols + c]; }
    if (v >= domain) { x = 1000.0f; }
    if constexpr (sizeof(T) == 2) {
        out[i] = __float2bfloat16_rn(x);
    } else {
        out[i] = x;
    }
}

struct Shape {
    const char* name;
    bool fp32;
    int domain, physical;
};

// Device buffers of one call geometry (B rows, K drafts).
struct Bufs {
    int B = 0, K = 0;
    void* logits = nullptr;
    int *targets = nullptr, *drafts = nullptr, *extents = nullptr, *lengths = nullptr, *anchors = nullptr;
    int *tokens = nullptr, *counts = nullptr, *accepted = nullptr;
    ops::SamplingConfig* cfg = nullptr;
    void alloc(const Shape& s, int b, int k) {
        B = b;
        K = k;
        ck(cudaMalloc(&logits, static_cast<std::size_t>(s.physical) * (k + 1) * b * (s.fp32 ? 4 : 2)), "logits");
        ck(cudaMalloc(&targets, (k + 1) * b * 4), "targets");
        ck(cudaMemset(targets, 0, (k + 1) * b * 4), "targets 0");
        ck(cudaMalloc(&drafts, std::max(1, k) * b * 4), "drafts");
        ck(cudaMalloc(&extents, b * 4), "extents");
        ck(cudaMalloc(&lengths, b * 4), "lengths");
        ck(cudaMalloc(&anchors, b * 4), "anchors");
        ck(cudaMalloc(&tokens, (k + 1) * b * 4), "tokens");
        ck(cudaMalloc(&counts, b * 4), "counts");
        ck(cudaMalloc(&accepted, b * 4), "accepted");
        ck(cudaMalloc(&cfg, b * sizeof(ops::SamplingConfig)), "cfg");
    }
};

struct Out {
    std::vector<int> tokens, counts, accepted, lengths, anchors;
    std::map<int, std::vector<int>> token_counts; // slot -> [domain], for the slots the call's rows use
};

// Rows of one call, host side.
struct Rows {
    std::vector<ops::SamplingConfig> cfg;
    std::vector<int> drafts, extents, lengths, quantised;
    std::vector<float> peaks;
    std::vector<int> count_slot; // -1 or the row's token-count array index
};

struct Env {
    Shape shape;
    WorkspaceArena* work = nullptr;
    float* d_bias = nullptr;
    std::int32_t* d_mask = nullptr;
    int* d_counts = nullptr;     // [kSlots][domain]
    std::vector<int> counts0;    // initial content of every slot
    static constexpr int kSlots = 16;
    explicit Env(Shape s) : shape(s) {
        std::mt19937 g(77);
        std::vector<float> bias(s.domain);
        for (auto& b : bias) { b = std::uniform_real_distribution<float>(-1.0f, 1.0f)(g) * ((g() % 5) == 0 ? 3.0f : 0.0f); }
        ck(cudaMalloc(&d_bias, bias.size() * 4), "bias");
        ck(cudaMemcpy(d_bias, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice), "bias up");
        const int words = (s.domain + 31) / 32;
        std::vector<std::int32_t> mask(words);
        for (auto& w : mask) { w = static_cast<std::int32_t>(g() | g()); }
        ck(cudaMalloc(&d_mask, mask.size() * 4), "mask");
        ck(cudaMemcpy(d_mask, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice), "mask up");
        counts0.assign(static_cast<std::size_t>(kSlots) * s.domain, 0);
        for (int b = 0; b < kSlots; ++b) {
            for (int k = 0; k < 300; ++k) { counts0[static_cast<std::size_t>(b) * s.domain + g() % s.domain] += 1 + g() % 3; }
        }
        ck(cudaMalloc(&d_counts, counts0.size() * 4), "counts");
        work = new WorkspaceArena(ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(s.domain, 1, 7, 1, 64) + 4096);
    }
    void download_slots(Out& o, const std::vector<int>& slots) const {
        for (const int slot : slots) {
            if (slot < 0 || o.token_counts.count(slot)) { continue; }
            auto& v = o.token_counts[slot];
            v.resize(shape.domain);
            ck(cudaMemcpy(v.data(), d_counts + static_cast<std::size_t>(slot) * shape.domain, v.size() * 4,
                          cudaMemcpyDeviceToHost), "tc down");
        }
    }

    // Random rows of a class (identity / invariance part).
    Rows make_rows(int kind, int call, int B, int K) {
        std::mt19937_64 g(1000003ull * static_cast<unsigned long long>(kind) + static_cast<unsigned long long>(call));
        Rows r;
        r.cfg.resize(B);
        r.drafts.resize(static_cast<std::size_t>(std::max(1, K)) * B);
        r.extents.resize(B);
        r.lengths.resize(B);
        r.quantised.assign(B, 0);
        r.peaks.assign(static_cast<std::size_t>(K + 1) * B, 0.0f);
        r.count_slot.assign(B, -1);
        const float heights[5] = {0.0f, 2.0f, 4.0f, 8.0f, 12.0f};
        for (int b = 0; b < B; ++b) {
            ops::SamplingConfig& x = r.cfg[b];
            x.seed        = g();
            x.temperature = 1.0f;
            r.lengths[b]  = static_cast<int>(g() % 262144);
            r.extents[b]  = static_cast<int>(g() % (K + 1));
            for (int j = 0; j < K; ++j) {
                r.drafts[static_cast<std::size_t>(b) * K + j] = static_cast<int>(g() % shape.domain);
                r.peaks[static_cast<std::size_t>(b) * (K + 1) + j] = heights[g() % 5];
            }
            const int k = kind == 7 ? static_cast<int>(g() % 7) : kind;
            switch (k) {
            case 0: break; // temperature 1, untruncated
            case 1: {
                const float t[4] = {0.3f, 0.7f, 1.5f, 2.0f};
                x.temperature = t[g() % 4];
                break;
            }
            case 2: x.min_p = (g() % 3 == 0) ? 0.02f : (g() % 2 ? 0.05f : 0.1f); break;
            case 3:
                x.presence_penalty   = 0.5f;
                x.frequency_penalty  = 0.3f;
                x.repetition_penalty = (g() % 2) ? 1.1f : 1.0f;
                r.count_slot[b]      = b % kSlots;
                if (g() % 2) { x.min_p = 0.05f; }
                break;
            case 4:
                x.logit_bias = d_bias;
                if (g() % 2) { x.token_bitmask = d_mask; }
                x.suppressed[0]    = r.drafts[static_cast<std::size_t>(b) * K]; // a barred draft
                x.suppressed[1]    = 0;
                x.suppressed_count = 2;
                r.count_slot[b]    = b % kSlots; // counts only
                break;
            case 5: r.quantised[b] = 1; break; // ties
            case 6: { // other routes in the same call: greedy, top-k, top-p
                const int o = static_cast<int>(g() % 3);
                if (o == 0) { x.temperature = 0.0f; }
                if (o == 1) { x.top_k = 20; }
                if (o == 2) { x.top_p = 0.9f; }
                break;
            }
            }
        }
        return r;
    }

    // One op call on the current route; returns every output (and the rows' token-count arrays).
    Out run(Bufs& bf, const Rows& r, unsigned long long logit_seed, bool split) {
        const int B = bf.B, K = bf.K, cols = K + 1;
        std::vector<ops::SamplingConfig> cfg = r.cfg;
        for (int b = 0; b < B; ++b) {
            if (r.count_slot[b] >= 0) { cfg[b].token_counts = d_counts + static_cast<std::size_t>(r.count_slot[b]) * shape.domain; }
        }
        ck(cudaMemcpy(bf.cfg, cfg.data(), B * sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice), "cfg up");
        ck(cudaMemcpy(bf.drafts, r.drafts.data(), r.drafts.size() * 4, cudaMemcpyHostToDevice), "drafts up");
        ck(cudaMemcpy(bf.extents, r.extents.data(), B * 4, cudaMemcpyHostToDevice), "extents up");
        ck(cudaMemcpy(bf.lengths, r.lengths.data(), B * 4, cudaMemcpyHostToDevice), "lengths up");
        ck(cudaMemset(bf.anchors, 0, B * 4), "anchors");
        ck(cudaMemcpy(d_counts, counts0.data(), counts0.size() * 4, cudaMemcpyHostToDevice), "counts up");
        int *d_q = nullptr, *d_dr = nullptr;
        float* d_pk = nullptr;
        ck(cudaMalloc(&d_q, B * 4), "q");
        ck(cudaMalloc(&d_pk, r.peaks.size() * 4), "pk");
        ck(cudaMemcpy(d_q, r.quantised.data(), B * 4, cudaMemcpyHostToDevice), "q up");
        ck(cudaMemcpy(d_pk, r.peaks.data(), r.peaks.size() * 4, cudaMemcpyHostToDevice), "pk up");
        d_dr = bf.drafts;
        const long long n = static_cast<long long>(B) * cols * shape.physical;
        if (shape.fp32) {
            gen_logits<float><<<static_cast<unsigned>((n + 255) / 256), 256>>>(static_cast<float*>(bf.logits), B, cols, K,
                shape.physical, shape.domain, logit_seed, d_dr, d_pk, d_q);
        } else {
            gen_logits<__nv_bfloat16><<<static_cast<unsigned>((n + 255) / 256), 256>>>(
                static_cast<__nv_bfloat16*>(bf.logits), B, cols, K, shape.physical, shape.domain, logit_seed, d_dr, d_pk, d_q);
        }
        ck(cudaGetLastError(), "gen");
        ops::detail::speculative_exact_split() = split;
        Tensor lt(bf.logits, shape.fp32 ? DType::FP32 : DType::BF16, {shape.physical, cols, B});
        Tensor tt(bf.targets, DType::I32, {cols, B});
        Tensor dt(bf.drafts, DType::I32, {std::max(1, K), B});
        Tensor et(bf.extents, DType::I32, {B});
        Tensor lnt(bf.lengths, DType::I32, {B});
        Tensor at(bf.anchors, DType::I32, {B});
        Tensor tok(bf.tokens, DType::I32, {cols, B});
        Tensor ct(bf.counts, DType::I32, {B});
        Tensor act(bf.accepted, DType::I32, {B});
        ops::speculative_accept_greedy_drafts(tt, lt, dt, et, lnt, at, tok, ct, act, shape.domain, bf.cfg, *work, nullptr);
        ck(cudaDeviceSynchronize(), "accept");
        cudaFree(d_q);
        cudaFree(d_pk);
        Out o;
        o.tokens.resize(static_cast<std::size_t>(cols) * B);
        o.counts.resize(B);
        o.accepted.resize(B);
        o.lengths.resize(B);
        o.anchors.resize(B);
        ck(cudaMemcpy(o.tokens.data(), bf.tokens, o.tokens.size() * 4, cudaMemcpyDeviceToHost), "tok down");
        ck(cudaMemcpy(o.counts.data(), bf.counts, B * 4, cudaMemcpyDeviceToHost), "cnt down");
        ck(cudaMemcpy(o.accepted.data(), bf.accepted, B * 4, cudaMemcpyDeviceToHost), "acc down");
        ck(cudaMemcpy(o.lengths.data(), bf.lengths, B * 4, cudaMemcpyDeviceToHost), "len down");
        ck(cudaMemcpy(o.anchors.data(), bf.anchors, B * 4, cudaMemcpyDeviceToHost), "anc down");
        download_slots(o, r.count_slot);
        return o;
    }
};

// A row's whole result as one comparable record (tokens up to the call's width are compared up to produced).
std::vector<int> row_record(const Out& o, int b, int cols, const Rows& r, int domain) {
    std::vector<int> rec;
    const int produced = o.counts[b];
    rec.push_back(produced);
    rec.push_back(o.accepted[b]);
    rec.push_back(o.lengths[b]);
    rec.push_back(o.anchors[b]);
    for (int i = 0; i < produced; ++i) { rec.push_back(o.tokens[static_cast<std::size_t>(b) * cols + i]); }
    if (r.count_slot[b] >= 0) {
        // the produced tokens' counts (the slot is shared by no other row in the invariance calls)
        for (int i = 0; i < produced; ++i) {
            const int t = o.tokens[static_cast<std::size_t>(b) * cols + i];
            if (t >= 0 && t < domain) { rec.push_back(o.token_counts.at(r.count_slot[b])[t]); }
        }
    }
    return rec;
}

// Copies row b of a [*, cols, B] logits call into a single-row call with K2 drafts (K2 >= the row's extent).
Rows single_row(const Rows& r, int b, int K, int K2) {
    Rows s;
    s.cfg       = {r.cfg[b]};
    s.extents   = {r.extents[b]};
    s.lengths   = {r.lengths[b]};
    s.quantised = {r.quantised[b]};
    s.count_slot = {r.count_slot[b]};
    s.drafts.assign(std::max(1, K2), 0);
    s.peaks.assign(K2 + 1, 0.0f);
    for (int j = 0; j < std::min(K, K2); ++j) {
        s.drafts[j] = r.drafts[static_cast<std::size_t>(b) * K + j];
        s.peaks[j]  = r.peaks[static_cast<std::size_t>(b) * (K + 1) + j];
    }
    return s;
}

int test_invariance_and_identity(const Shape& s) {
    Env env(s);
    const int K = 7, B = 16;
    Bufs big, one7, one3;
    big.alloc(s, B, K);
    one7.alloc(s, 1, K);
    one3.alloc(s, 1, 3);
    const char* names[8] = {"t1 untruncated", "temperatures 0.3-2", "min_p", "penalties + counts (+min_p)",
                            "bias + barred draft + mask + counts", "ties (quantised)", "other routes mixed in", "mixed"};
    int failures = 0;
    std::printf("| shape | row class | untruncated rows | repeat identical | alone (B=1, K=7) == in batch | alone K=3 == K=7 (extent <= 3) | old one-CTA == split |\n|---|---|---|---|---|---|---|\n");
    for (int kind = 0; kind < 8; ++kind) {
        long rows = 0, rep = 0, alone = 0, k3n = 0, k3 = 0, old_same = 0;
        for (int call = 0; call < 24; ++call) {
            Rows r = env.make_rows(kind, call, B, K);
            const unsigned long long ls = 0xC0FFEEull + 7919ull * static_cast<unsigned long long>(kind * 1000 + call);
            const Out a  = env.run(big, r, ls, true);
            const Out a2 = env.run(big, r, ls, true);
            const Out o  = env.run(big, r, ls, false);
            for (int b = 0; b < B; ++b) {
                const ops::SamplingConfig& c = r.cfg[b];
                const bool untr = c.temperature > 0 && c.top_k <= 0 && c.top_p >= 1.0f;
                const auto ra = row_record(a, b, K + 1, r, s.domain);
                if (!untr) { continue; }
                ++rows;
                rep += ra == row_record(a2, b, K + 1, r, s.domain);
                old_same += ra == row_record(o, b, K + 1, r, s.domain);
                // alone in a B = 1 call: the same logits are regenerated for row 0 only when the generator keys the
                // row index -> regenerate with the row placed at b = 0 by copying the batch call's column bytes.
                Rows s1 = single_row(r, b, K, K);
                const std::size_t col_bytes = static_cast<std::size_t>(s.physical) * (s.fp32 ? 4 : 2);
                ck(cudaMemcpy(one7.logits, static_cast<char*>(big.logits) + static_cast<std::size_t>(b) * (K + 1) * col_bytes,
                              (K + 1) * col_bytes, cudaMemcpyDeviceToDevice), "row copy");
                // the op on row b's logits alone (no regeneration): reset the state and call directly
                auto call_direct = [&](Bufs& bf, const Rows& rr, int KK) {
                    std::vector<ops::SamplingConfig> cfg = rr.cfg;
                    if (rr.count_slot[0] >= 0) { cfg[0].token_counts = env.d_counts + static_cast<std::size_t>(rr.count_slot[0]) * s.domain; }
                    ck(cudaMemcpy(bf.cfg, cfg.data(), sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice), "cfg");
                    ck(cudaMemcpy(bf.drafts, rr.drafts.data(), rr.drafts.size() * 4, cudaMemcpyHostToDevice), "dr");
                    ck(cudaMemcpy(bf.extents, rr.extents.data(), 4, cudaMemcpyHostToDevice), "ex");
                    ck(cudaMemcpy(bf.lengths, rr.lengths.data(), 4, cudaMemcpyHostToDevice), "ln");
                    ck(cudaMemset(bf.anchors, 0, 4), "an");
                    ck(cudaMemcpy(env.d_counts, env.counts0.data(), env.counts0.size() * 4, cudaMemcpyHostToDevice), "counts");
                    ops::detail::speculative_exact_split() = true;
                    Tensor lt(bf.logits, s.fp32 ? DType::FP32 : DType::BF16, {s.physical, KK + 1, 1});
                    Tensor tt(bf.targets, DType::I32, {KK + 1, 1});
                    Tensor dt(bf.drafts, DType::I32, {std::max(1, KK), 1});
                    Tensor et(bf.extents, DType::I32, {1});
                    Tensor lnt(bf.lengths, DType::I32, {1});
                    Tensor at(bf.anchors, DType::I32, {1});
                    Tensor tok(bf.tokens, DType::I32, {KK + 1, 1});
                    Tensor ct(bf.counts, DType::I32, {1});
                    Tensor act(bf.accepted, DType::I32, {1});
                    ops::speculative_accept_greedy_drafts(tt, lt, dt, et, lnt, at, tok, ct, act, s.domain, bf.cfg, *env.work, nullptr);
                    ck(cudaDeviceSynchronize(), "direct");
                    Out oo;
                    oo.tokens.resize(KK + 1);
                    oo.counts.resize(1);
                    oo.accepted.resize(1);
                    oo.lengths.resize(1);
                    oo.anchors.resize(1);
                    ck(cudaMemcpy(oo.tokens.data(), bf.tokens, (KK + 1) * 4, cudaMemcpyDeviceToHost), "t");
                    ck(cudaMemcpy(oo.counts.data(), bf.counts, 4, cudaMemcpyDeviceToHost), "c");
                    ck(cudaMemcpy(oo.accepted.data(), bf.accepted, 4, cudaMemcpyDeviceToHost), "a");
                    ck(cudaMemcpy(oo.lengths.data(), bf.lengths, 4, cudaMemcpyDeviceToHost), "l");
                    ck(cudaMemcpy(oo.anchors.data(), bf.anchors, 4, cudaMemcpyDeviceToHost), "n");
                    env.download_slots(oo, rr.count_slot);
                    return oo;
                };
                const Out ob = call_direct(one7, s1, K);
                alone += ra == row_record(ob, 0, K + 1, s1, s.domain);
                if (r.extents[b] <= 3) {
                    ++k3n;
                    Rows s3 = single_row(r, b, K, 3);
                    ck(cudaMemcpy(one3.logits, static_cast<char*>(big.logits) + static_cast<std::size_t>(b) * (K + 1) * col_bytes,
                                  4 * col_bytes, cudaMemcpyDeviceToDevice), "row copy 3");
                    const Out o3 = call_direct(one3, s3, 3);
                    k3 += ra == row_record(o3, 0, 4, s3, s.domain);
                }
            }
        }
        std::printf("| %s | %s | %ld | %ld | %ld | %ld / %ld | %ld |\n", s.name, names[kind], rows, rep, alone, k3, k3n, old_same);
        std::fflush(stdout);
        // agent-EN 17:5x: any difference from the old one-CTA route fails (same draws by construction; the normalizer is
        // the same double sum in another order)
        if (rep != rows || alone != rows || k3 != k3n || old_same != rows) { ++failures; }
    }
    ops::detail::speculative_exact_split() = true;
    return failures;
}

// ---------------------------------------------------------------------------------------------- exactness
// Inverse of the regularised lower incomplete gamma is overkill: chi-square critical value at alpha = 1e-3 by the
// Wilson-Hilferty approximation (z_{0.999} = 3.0902), accurate to ~1 % for df >= 3.
double chi2_crit(int df) {
    const double z = 3.0902, a = 2.0 / (9.0 * df);
    const double t = 1.0 - a + z * std::sqrt(a);
    return df * t * t * t;
}

struct ExactCase {
    const char* name;
    float temperature, min_p;
    int profile;    // 0 peaked (top ~0.9), 1 medium (top ~0.43), 2 flat (100 equal tokens), 3 medium, low-p draft
};

// Column logits of a case: `ids` = the bin tokens (explicit), the tail at -10 (T 1) or -7 (other T).
// REACH (finding 2026-11-21 18:0x, both routes alike, and ops::sample too): sampling_gumbel draws from a 24-bit uniform,
// so a Gumbel is at most ~17.3 and at least ~-2.85: a token whose scaled logit (x - max) / T lies more than ~20 below the
// column maximum is never drawn by the Gumbel-max (correction / bonus / plain sampling). A T 0.7 case with the tail at
// -10 (gap 20 scaled) failed on both routes with the tail bin empty (expected 59 of 256k: z 7.7). The tail is kept
// inside reach here (gap 11 / 0.7 = 15.7); the reach itself is a property of the shared sampler, reported separately.
void case_logits(const ExactCase& c, int col, int domain, std::vector<float>& x, std::vector<int>& ids, int& draft) {
    x.assign(domain, c.temperature == 1.0f ? -10.0f : -7.0f);
    ids.clear();
    std::mt19937 g(1234 + 17 * col + 1000 * c.profile);
    auto pick = [&]() {
        for (;;) {
            const int v = static_cast<int>(g() % domain);
            if (std::find(ids.begin(), ids.end(), v) == ids.end()) { return v; }
        }
    };
    if (c.profile == 2) {
        for (int i = 0; i < 100; ++i) { ids.push_back(pick()); x[ids.back()] = 0.0f; }
        draft = ids[7];
        return;
    }
    const float peaked[6] = {6.0f, 3.0f, 2.0f, 1.0f, 0.5f, 0.0f};
    const float medium[6] = {4.0f, 3.5f, 3.0f, 2.0f, 1.0f, 0.0f};
    for (int i = 0; i < 6; ++i) {
        ids.push_back(pick());
        x[ids.back()] = c.profile == 0 ? peaked[i] : medium[i];
    }
    draft = c.profile == 3 ? ids[4] : ids[0];
}

int test_exactness(const Shape& s, bool quick) {
    if (!s.fp32) { return 0; } // the distribution test runs at Inkling's shape
    const int K = 3, cols = K + 1, B = 64;
    const int calls = quick ? 1000 : 4000;
    const ExactCase cases[6] = {{"peaked (top 0.9), T 1", 1.0f, 0.0f, 0}, {"medium (top 0.43), T 1", 1.0f, 0.0f, 1},
                                {"flat (100 equal), T 1", 1.0f, 0.0f, 2}, {"medium, low-p draft (5th), T 1", 1.0f, 0.0f, 3},
                                {"medium, T 0.7", 0.7f, 0.0f, 1}, {"medium, T 1, min_p 0.05", 1.0f, 0.05f, 1}};
    Env env(s);
    Bufs bf;
    bf.alloc(s, B, K);
    int failures = 0;
    std::printf("\n| route | case | column | n reaching | bins | chi2 | crit (a=1e-3) | max |z| | power: chi2 vs 1 %% shifted p |\n|---|---|---|---|---|---|---|---|---|\n");
    for (const bool split : {true, false}) {
        for (const ExactCase& c : cases) {
            // columns
            std::vector<std::vector<int>> ids(cols);
            std::vector<int> draft(cols, -1);
            std::vector<std::vector<double>> p(cols); // per column: bin probabilities (ids..., tail)
            const std::size_t col_bytes = static_cast<std::size_t>(s.physical) * 4;
            std::vector<float> host(static_cast<std::size_t>(s.physical) * cols);
            for (int j = 0; j < cols; ++j) {
                std::vector<float> x;
                int d = -1;
                case_logits(c, j, s.domain, x, ids[j], d);
                if (j < K) { draft[j] = d; }
                std::copy(x.begin(), x.end(), host.begin() + static_cast<std::size_t>(j) * s.physical);
                // target probabilities with the min_p floor, as the sampler defines them (double)
                double mx = -1e30;
                for (float v : x) { mx = std::max(mx, static_cast<double>(v)); }
                const double floor = c.min_p > 0 ? mx + c.temperature * std::log(static_cast<double>(c.min_p)) : -1e300;
                double tot = 0, in_bins = 0;
                std::vector<double> w(ids[j].size());
                for (int v = 0; v < s.domain; ++v) {
                    if (x[v] < floor) { continue; }
                    tot += std::exp((x[v] - mx) / c.temperature);
                }
                for (std::size_t i = 0; i < ids[j].size(); ++i) {
                    const float v = x[ids[j][i]];
                    w[i] = v < floor ? 0.0 : std::exp((v - mx) / c.temperature) / tot;
                    in_bins += w[i];
                }
                p[j] = w;
                p[j].push_back(std::max(0.0, 1.0 - in_bins));
            }
            for (int b = 0; b < B; ++b) {
                ck(cudaMemcpy(static_cast<char*>(bf.logits) + static_cast<std::size_t>(b) * cols * col_bytes, host.data(),
                              cols * col_bytes, cudaMemcpyHostToDevice), "exact logits");
            }
            std::vector<std::vector<long>> obs(cols);
            for (int j = 0; j < cols; ++j) { obs[j].assign(p[j].size(), 0); }
            std::vector<long> reach(cols, 0);
            std::mt19937_64 g(99 + c.profile * 7 + static_cast<int>(c.temperature * 10) + (split ? 0 : 1));
            std::vector<ops::SamplingConfig> cfg(B);
            std::vector<int> drafts(static_cast<std::size_t>(K) * B), extents(B, K), lengths(B);
            for (int b = 0; b < B; ++b) {
                for (int j = 0; j < K; ++j) { drafts[static_cast<std::size_t>(b) * K + j] = draft[j]; }
            }
            ck(cudaMemcpy(bf.drafts, drafts.data(), drafts.size() * 4, cudaMemcpyHostToDevice), "dr");
            ck(cudaMemcpy(bf.extents, extents.data(), B * 4, cudaMemcpyHostToDevice), "ex");
            ops::detail::speculative_exact_split() = split;
            Tensor lt(bf.logits, DType::FP32, {s.physical, cols, B});
            Tensor tt(bf.targets, DType::I32, {cols, B});
            Tensor dt(bf.drafts, DType::I32, {K, B});
            Tensor et(bf.extents, DType::I32, {B});
            Tensor lnt(bf.lengths, DType::I32, {B});
            Tensor at(bf.anchors, DType::I32, {B});
            Tensor tok(bf.tokens, DType::I32, {cols, B});
            Tensor ct(bf.counts, DType::I32, {B});
            Tensor act(bf.accepted, DType::I32, {B});
            std::vector<int> tokens(static_cast<std::size_t>(cols) * B), counts(B);
            for (int call = 0; call < calls; ++call) {
                for (int b = 0; b < B; ++b) {
                    cfg[b]             = ops::SamplingConfig{};
                    cfg[b].temperature = c.temperature;
                    cfg[b].min_p       = c.min_p;
                    cfg[b].seed        = g();
                    lengths[b]         = static_cast<int>(g() % 200000);
                }
                ck(cudaMemcpy(bf.cfg, cfg.data(), B * sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice), "cfg");
                ck(cudaMemcpy(bf.lengths, lengths.data(), B * 4, cudaMemcpyHostToDevice), "ln");
                ops::speculative_accept_greedy_drafts(tt, lt, dt, et, lnt, at, tok, ct, act, s.domain, bf.cfg, *env.work, nullptr);
                ck(cudaMemcpy(tokens.data(), bf.tokens, tokens.size() * 4, cudaMemcpyDeviceToHost), "tok");
                ck(cudaMemcpy(counts.data(), bf.counts, B * 4, cudaMemcpyDeviceToHost), "cnt");
                for (int b = 0; b < B; ++b) {
                    for (int j = 0; j < counts[b]; ++j) {
                        // column j is reached iff tokens 0..j-1 are the drafts (the round stops at the first other token)
                        ++reach[j];
                        const int t = tokens[static_cast<std::size_t>(b) * cols + j];
                        const auto it = std::find(ids[j].begin(), ids[j].end(), t);
                        ++obs[j][it == ids[j].end() ? ids[j].size() : static_cast<std::size_t>(it - ids[j].begin())];
                    }
                }
            }
            for (int j = 0; j < cols; ++j) {
                if (reach[j] < 2000) { continue; }
                // merge bins with expected count < 5 into the tail bin
                double chi2 = 0, chi2_shift = 0, maxz = 0;
                int bins = 0;
                double e_small = 0, o_small = 0;
                // power control: move 1 % of the total mass from the largest bin to the second largest
                std::vector<double> q = p[j];
                std::vector<std::size_t> order(q.size());
                for (std::size_t i = 0; i < q.size(); ++i) { order[i] = i; }
                std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return q[a] > q[b]; });
                std::vector<double> qs = q;
                qs[order[0]] -= 0.01;
                qs[order[1]] += 0.01;
                double e_small_s = 0;
                for (std::size_t i = 0; i < q.size(); ++i) {
                    const double e = q[i] * reach[j], es = qs[i] * reach[j], o = static_cast<double>(obs[j][i]);
                    if (e < 5) {
                        e_small += e;
                        e_small_s += es;
                        o_small += o;
                        continue;
                    }
                    ++bins;
                    chi2 += (o - e) * (o - e) / e;
                    chi2_shift += es > 0 ? (o - es) * (o - es) / es : 0;
                    maxz = std::max(maxz, std::fabs(o - e) / std::sqrt(e));
                }
                if (e_small >= 5) {
                    ++bins;
                    chi2 += (o_small - e_small) * (o_small - e_small) / e_small;
                    chi2_shift += (o_small - e_small_s) * (o_small - e_small_s) / std::max(e_small_s, 1e-9);
                }
                const int df = std::max(1, bins - 1);
                const double crit = chi2_crit(df);
                const bool ok = chi2 < crit;
                std::printf("| %s | %s | %d%s | %ld | %d | %.1f | %.1f | %.2f | %.1f |\n", split ? "split" : "one-CTA", c.name, j,
                            j == K ? " (bonus)" : "", reach[j], bins, chi2, crit, maxz, chi2_shift);
                std::fflush(stdout);
                if (!ok) { ++failures; }
            }
        }
    }
    ops::detail::speculative_exact_split() = true;
    return failures;
}

void run_bench(const Shape& s) {
    Env env(s);
    for (const int B : {1, 8}) {
        const int K = 7, cols = K + 1;
        Bufs bf;
        bf.alloc(s, B, K);
        Rows r = env.make_rows(0, 0, B, K);
        for (int b = 0; b < B; ++b) {
            r.extents[b] = K;
            for (int j = 0; j < K; ++j) { r.peaks[static_cast<std::size_t>(b) * cols + j] = 40.0f; } // every draft accepted
        }
        for (const bool split : {false, true}) {
            env.run(bf, r, 0xBEEFull, split); // warm + logits
            Tensor lt(bf.logits, s.fp32 ? DType::FP32 : DType::BF16, {s.physical, cols, B});
            Tensor tt(bf.targets, DType::I32, {cols, B});
            Tensor dt(bf.drafts, DType::I32, {K, B});
            Tensor et(bf.extents, DType::I32, {B});
            Tensor lnt(bf.lengths, DType::I32, {B});
            Tensor at(bf.anchors, DType::I32, {B});
            Tensor tok(bf.tokens, DType::I32, {cols, B});
            Tensor ct(bf.counts, DType::I32, {B});
            Tensor act(bf.accepted, DType::I32, {B});
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            const int reps = 50;
            cudaEventRecord(e0);
            for (int i = 0; i < reps; ++i) {
                ops::speculative_accept_greedy_drafts(tt, lt, dt, et, lnt, at, tok, ct, act, s.domain, bf.cfg, *env.work, nullptr);
            }
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            std::vector<int> counts(B);
            ck(cudaMemcpy(counts.data(), bf.counts, B * 4, cudaMemcpyDeviceToHost), "cnt");
            std::printf("bench accept %-26s B=%d K=7 t1 untruncated, %d tokens committed (row 0)  %-8s %9.1f us / call (eager, whole op)\n",
                        s.name, B, counts[0], split ? "split" : "one-CTA", 1000.0f * ms / reps);
            std::fflush(stdout);
        }
    }
    ops::detail::speculative_exact_split() = true;
}

} // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    // W4g's real vocabulary is 248,320 (agent-EN 17:5x); 151,669 / 151,936 is the older Qwen3 shape S1 tested.
    const Shape shapes[3] = {{"Inkling fp32 201024", true, 201024, 201024},
                             {"W4g bf16 248320", false, 248320, 248320},
                             {"Qwen3 bf16 151669/151936", false, 151669, 151936}};
    const std::string mode = argc > 1 ? argv[1] : "all";
    if (mode == "bench") {
        for (const Shape& s : shapes) { run_bench(s); }
        return 0;
    }
    int failures = 0;
    if (mode == "all" || mode == "invariance") {
        for (const Shape& s : shapes) { failures += test_invariance_and_identity(s); }
    }
    if (mode == "all" || mode == "exact" || mode == "quick") {
        failures += test_exactness(shapes[0], mode == "quick");
    }
    std::printf(failures == 0 ? "SPEC EXACT SPLIT PASS\n" : "SPEC EXACT SPLIT FAIL: %d table rows\n", failures);
    return failures == 0 ? 0 : 1;
}
