// DSpark drafter ops (ENGINE-INKLING lane e2) against the golden vectors of ref/dspark_ref.py (make_vectors),
// op by op with the REFERENCE's inputs (each op isolated), then the Markov chain end to end, then batch
// invariance. Weights come from the checkpoint's mtp.safetensors; the target rows the drafter shares (embed rows
// of the anchors, a slice of the unembed) and the fold from the vectors.
//
// Usage: sinfer_dspark_test <dir> [tag] [round]   (tag bf16 (default), fp8kv or fp8all = FP8 weights + KV; `round`: the target-side
//   append + propose of targets/inkling2/impl/dspark.cpp end to end, needs <dir>/dspark_round/{features,unembed})
//   <dir>/mtp.safetensors, <dir>/dspark_feature_mix/mix64.safetensors, <dir>/dspark_context_kv/<tag>_ctx64,
//   <dir>/dspark_round/{<tag>_block_b3,<tag>_context_kv,test_weights}, <dir>/dspark_head/<tag>_b3,
//   <dir>/dspark_fold/fold.safetensors
// Tolerances (vectors/dspark_*/meta.json): bf16 outputs of GEMM / norm ops <= 1 ulp on <= 1 % of elements and
// rel RMS <= 2e-3; attention <= 2 ulp on <= 2 %; fp32 Markov context rel RMS <= 1e-5; drafts >= 99 % equal.

#include "api/ops/dspark.h"
#include "api/ops/inkling_mix.h"
#include "targets/inkling2/impl/dspark.h"
#include "ops/op_tester.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;
using G = ops::DSparkGeometry;

namespace {

/// DeviceBuffer with data() / size() (the test reads like the other op tests' helpers).
struct DBuf : DeviceBuffer {
    DBuf() = default;
    explicit DBuf(std::size_t n) : DeviceBuffer(n) {}
    DBuf(DeviceBuffer&& o) noexcept : DeviceBuffer(std::move(o)) {}
    [[nodiscard]] void* data() const { return p; }
    [[nodiscard]] std::size_t size() const { return bytes; }
};

// ------------------------------------------------------------------------------------------- safetensors
struct Entry {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::size_t begin = 0, bytes = 0;
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path) : path_(path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) { throw std::runtime_error("cannot open " + path); }
        std::uint64_t header_bytes = 0;
        in.read(reinterpret_cast<char*>(&header_bytes), 8);
        std::string header(header_bytes, '\0');
        in.read(header.data(), static_cast<std::streamsize>(header_bytes));
        data_start_ = 8 + header_bytes;
        const nlohmann::json parsed = nlohmann::json::parse(header); // kept alive: items() refers to it
        for (const auto& [name, value] : parsed.items()) {
            if (name == "__metadata__") { continue; }
            Entry e;
            e.dtype = value["dtype"].get<std::string>();
            for (const auto& d : value["shape"]) { e.shape.push_back(d.get<std::int64_t>()); }
            e.begin = value["data_offsets"][0].get<std::size_t>();
            e.bytes = value["data_offsets"][1].get<std::size_t>() - e.begin;
            entries_[name] = e;
        }
    }
    [[nodiscard]] bool has(const std::string& name) const { return entries_.count(name) != 0; }
    [[nodiscard]] const Entry& entry(const std::string& name) const {
        const auto it = entries_.find(name);
        if (it == entries_.end()) { throw std::runtime_error(path_ + " lacks " + name); }
        return it->second;
    }
    template <class T>
    [[nodiscard]] std::vector<T> read(const std::string& name) const {
        const Entry& e = entry(name);
        std::vector<T> out(e.bytes / sizeof(T));
        std::ifstream in(path_, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(data_start_ + e.begin));
        in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(e.bytes));
        return out;
    }
    /// bf16 tensor as raw bits.
    [[nodiscard]] std::vector<std::uint16_t> bf16(const std::string& name) const {
        if (entry(name).dtype != "BF16") { throw std::runtime_error(name + " is not BF16"); }
        return read<std::uint16_t>(name);
    }
    [[nodiscard]] std::vector<float> f32(const std::string& name) const {
        if (entry(name).dtype != "F32") { throw std::runtime_error(name + " is not F32"); }
        return read<float>(name);
    }
    /// I64 / I32 tensor as int32.
    [[nodiscard]] std::vector<std::int32_t> i32(const std::string& name) const {
        const Entry& e = entry(name);
        if (e.dtype == "I32") { return read<std::int32_t>(name); }
        if (e.dtype != "I64") { throw std::runtime_error(name + " is not an integer tensor"); }
        const auto v = read<std::int64_t>(name);
        return std::vector<std::int32_t>(v.begin(), v.end());
    }

private:
    std::string path_;
    std::size_t data_start_ = 0;
    std::map<std::string, Entry> entries_;
};

// ---------------------------------------------------------------------------------------------- compare
float bf(std::uint16_t h) { return bf16_to_f32(h); }

/// Ordered integer of a bf16 bit pattern (ulp distance = difference).
std::int32_t ordered(std::uint16_t h) {
    const std::int32_t mag = h & 0x7FFF;
    return (h & 0x8000) ? -mag : mag;
}

/// An element is OVER when |got - ref| > ulp_max * 2^-7 * max(|ref|, rms(ref) / 4): bf16 ulps scaled by the value, with a
/// floor at a quarter of the tensor's RMS so a cancellation near zero (a different fp32 summation order of the same
/// GEMM, documented in ref/dspark_ref.py) is judged against the scale of its summands, not its own tiny magnitude.
/// `frac_max` bounds the elements that differ at all; `rel_rms_max` the RMS error relative to the reference's RMS.
struct Tol {
    int ulp_max          = 1;
    double frac_max      = 0.01;
    double rel_rms_max   = 2e-3;
};
constexpr Tol kGemm{2, 0.01, 2e-3};    // GEMM output: one rounding, another accumulation order
constexpr Tol kAttn{4, 0.25, 1e-3};    // P rounded to bf16 under an online (per-tile) max vs the reference's global max
constexpr Tol kTight{1, 0.001, 1e-3};  // norm / rope / elementwise on identical inputs

int failures = 0;

int report(const std::string& label, bool ok, const std::string& info) {
    std::cout << (ok ? "PASS " : "FAIL ") << label << "  " << info << "\n";
    if (!ok) { ++failures; }
    return ok ? 0 : 1;
}

int compare_bf16(const std::string& label, const std::vector<std::uint16_t>& got, const std::vector<std::uint16_t>& ref,
                 Tol tol) {
    if (got.size() != ref.size()) {
        return report(label, false, "size " + std::to_string(got.size()) + " vs " + std::to_string(ref.size()));
    }
    std::size_t differ = 0, over = 0;
    int worst = 0;
    double se = 0.0, sr = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) { sr += static_cast<double>(bf(ref[i])) * bf(ref[i]); }
    const double floor = std::sqrt(sr / static_cast<double>(std::max<std::size_t>(got.size(), 1))) / 4.0;
    double worst_scaled = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const int d = std::abs(ordered(got[i]) - ordered(ref[i]));
        if (d != 0) { ++differ; }
        worst = std::max(worst, d);
        const double e      = static_cast<double>(bf(got[i])) - bf(ref[i]);
        const double scaled = std::fabs(e) / (std::ldexp(1.0, -7) * std::max<double>(std::fabs(bf(ref[i])), floor));
        if (scaled > tol.ulp_max) { ++over; }
        worst_scaled = std::max(worst_scaled, scaled);
        se += e * e;
    }
    const double frac    = static_cast<double>(differ) / static_cast<double>(got.size());
    const double rel_rms = sr > 0 ? std::sqrt(se / sr) : std::sqrt(se);
    const bool ok        = over == 0 && frac <= tol.frac_max && rel_rms <= tol.rel_rms_max && std::isfinite(rel_rms);
    char buf[240];
    std::snprintf(buf, sizeof buf, "n=%zu differ=%.4f%% max_scaled_ulp=%.2f (raw %d) over_tol=%zu rel_rms=%.2e",
                  got.size(), 100.0 * frac, worst_scaled, worst, over, rel_rms);
    return report(label, ok, buf);
}

int compare_f32(const std::string& label, const std::vector<float>& got, const std::vector<float>& ref, double rel_max) {
    double se = 0.0, sr = 0.0;
    for (std::size_t i = 0; i < got.size() && i < ref.size(); ++i) {
        const double e = static_cast<double>(got[i]) - ref[i];
        se += e * e;
        sr += static_cast<double>(ref[i]) * ref[i];
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    char buf[120];
    std::snprintf(buf, sizeof buf, "n=%zu rel_rms=%.2e (max %.0e)", got.size(), rel, rel_max);
    return report(label, got.size() == ref.size() && rel <= rel_max, buf);
}

std::vector<std::uint16_t> dl_bf16(const void* p, std::size_t n) { return from_device<std::uint16_t>(p, n); }

/// rows [a; b; ...] stacked (row-major [rows, cols] each, same cols).
std::vector<std::uint16_t> cat_rows(std::initializer_list<std::vector<std::uint16_t>> parts) {
    std::vector<std::uint16_t> out;
    for (const auto& p : parts) { out.insert(out.end(), p.begin(), p.end()); }
    return out;
}

std::vector<std::uint16_t> slice(const std::vector<std::uint16_t>& v, std::size_t begin, std::size_t n) {
    return std::vector<std::uint16_t>(v.begin() + static_cast<std::ptrdiff_t>(begin),
                                      v.begin() + static_cast<std::ptrdiff_t>(begin + n));
}

/// out BF16 / FP32 [n, tokens] = W [n, k] . x [k, tokens] on lane e1's invariant GEMM.
/// Tag fp8all: the drafter's dense linears run the FP8 route (per-row e4m3 + scale, ops::dspark_gemm_w8) exactly as
/// the engine does after derive_dspark(fp8); the head and rnn_proj GEMMs stay BF16.
bool g_fp8_weights = false;

void gemm_dense(const DBuf& w, int n, int k, const void* x, int tokens, void* out);

void gemm(const DBuf& w, int n, int k, const void* x, int tokens, void* out, DType dtype) {
    const std::size_t ws_bytes = ops::inkling_gemm_inv_workspace_bytes(n, k, tokens, 1);
    DBuf ws(ws_bytes > 0 ? ws_bytes : 16);
    ops::inkling_gemm_inv(w.data(), n, k, x, tokens, out, dtype, n, ops::InklingGemmBatch{}, ws_bytes ? ws.data() : nullptr,
                          ws_bytes, nullptr);
}

void gemm_dense(const DBuf& w, int n, int k, const void* x, int tokens, void* out) {
    if (!g_fp8_weights) {
        gemm(w, n, k, x, tokens, out, DType::BF16);
        return;
    }
    DBuf codes(static_cast<std::size_t>(n) * k), scales(static_cast<std::size_t>(n) * 4);
    ops::dspark_quantize_rows_e4m3(w.data(), n, k, codes.data(), static_cast<float*>(scales.data()), nullptr);
    ops::dspark_gemm_w8(codes.data(), static_cast<const float*>(scales.data()), n, k, x, tokens, out, n, nullptr);
    cuda_synchronize();
}

std::string layer_name(const char* pattern, int l) {
    std::string s(pattern);
    const auto at = s.find("{l}");
    if (at != std::string::npos) { s.replace(at, 3, std::to_string(l)); }
    return s;
}

// ------------------------------------------------------------------------------------------------ the paged plane
/// One draft-KV plane for B lanes: lane b's block table maps its logical pages to its own contiguous physical
/// pages (pages are allocated in reverse order per lane so a wrong table lookup shows).
struct Plane {
    int lanes = 0, pages_per_lane = 0;
    DBuf k, v, tables;
    bool fp8 = false;
    Plane(int lanes_, int max_tokens, bool fp8_) : lanes(lanes_), fp8(fp8_) {
        pages_per_lane = (max_tokens + 63) / 64;
        const std::size_t page_bytes = static_cast<std::size_t>(64) * G::kKvRows * (fp8 ? 1 : 2);
        const std::size_t total      = page_bytes * static_cast<std::size_t>(lanes * pages_per_lane);
        k = DBuf(total);
        v = DBuf(total);
        cuda_check(cudaMemset(k.data(), 0x7F, total), "memset k"); // garbage: unwritten keys must not be read
        cuda_check(cudaMemset(v.data(), 0x7F, total), "memset v");
        std::vector<int> t(static_cast<std::size_t>(lanes * pages_per_lane));
        for (int b = 0; b < lanes; ++b) {
            for (int p = 0; p < pages_per_lane; ++p) { t[b * pages_per_lane + p] = (b + 1) * pages_per_lane - 1 - p; }
        }
        tables = to_device_i32(t);
    }
    [[nodiscard]] ops::DSparkPagedPlane view() const {
        return ops::DSparkPagedPlane{k.data(), v.data(), static_cast<const std::int32_t*>(tables.data()),
                                     pages_per_lane, fp8};
    }
};

} // namespace

// ------------------------------------------------------------------------------------------------ round mode
namespace round_mode {

namespace D = sinfer::targets::inkling2::detail;
namespace LW = sinfer::targets::inkling2::load;

Weight bw(const void* p, int n, int k) {
    Weight w{};
    w.qtype    = QType::BF16_CTRL;
    w.payload  = p;
    w.qdata    = p;
    w.n        = n;
    w.k        = k;
    w.ndim     = 2;
    w.shape[0] = w.padded_shape[0] = n;
    w.shape[1] = w.padded_shape[1] = k;
    return w;
}

struct Store {
    std::vector<DBuf> bufs;
    const void* put(std::vector<std::uint16_t> v) {
        bufs.emplace_back(to_device(v));
        return bufs.back().data();
    }
};

Tensor bt(const void* p, std::int32_t n) { return Tensor(const_cast<void*>(p), DType::BF16, {n}); }

int run(const std::string& dir, const std::string& tag) {
    const bool fp8 = tag == "fp8kv" || tag == "fp8all";
    const SafeTensors mtp(dir + "/mtp.safetensors");
    const SafeTensors feats(dir + "/dspark_round/features.safetensors");
    const SafeTensors unemb(dir + "/dspark_round/unembed.safetensors");
    const SafeTensors tw(dir + "/dspark_round/test_weights.safetensors");
    const SafeTensors rnd(dir + "/dspark_round/" + tag + "_block_b3.safetensors");
    const SafeTensors hd(dir + "/dspark_head/" + tag + "_b3.safetensors");
    Store st;
    LW::InklingWeights w;
    w.has_draft = true;
    // the target rows: the whole unembed; an embed table with only the anchors' rows filled (the drafter reads no other)
    const auto anchors = rnd.i32("anchors");
    const auto apos    = rnd.i32("anchor_pos");
    const int B        = static_cast<int>(apos.size());
    {
        std::vector<std::uint16_t> table(static_cast<std::size_t>(G::kVocabPadded) * G::kHidden, 0);
        const auto rows = tw.bf16("embed_rows");
        for (int b = 0; b < B; ++b) {
            std::memcpy(table.data() + static_cast<std::size_t>(anchors[b]) * G::kHidden,
                        rows.data() + static_cast<std::size_t>(b) * G::kHidden, G::kHidden * 2);
        }
        w.globals.embedding = bw(st.put(std::move(table)), G::kVocabPadded, G::kHidden);
    }
    w.globals.output_head    = bw(st.put(unemb.bf16("unembed")), G::kVocabPadded, G::kHidden);
    w.globals.embedding_norm = bt(st.put(tw.bf16("embed_norm")), G::kHidden);
    LW::InklingDraftWeights& d = w.draft;
    d.fuse_weight = Tensor(const_cast<void*>(st.put(mtp.bf16("fuse_weight"))), DType::BF16, {4, 4});
    d.mask_embed  = bt(st.put(mtp.bf16("mask_embed")), G::kHidden);
    d.final_norm  = bt(st.put(mtp.bf16("norm.weight")), G::kHidden);
    d.layers.resize(3);
    for (int l = 0; l < 3; ++l) {
        auto& L      = d.layers[static_cast<std::size_t>(l)];
        const auto n = [&](const char* pat) { return layer_name(pat, l); };
        L.feature_norm        = bt(st.put(mtp.bf16(n("layers.{l}.dflare_norm.weight"))), G::kHidden);
        L.input_norm          = bt(st.put(mtp.bf16(n("layers.{l}.input_layernorm.weight"))), G::kHidden);
        L.post_attention_norm = bt(st.put(mtp.bf16(n("layers.{l}.post_attention_layernorm.weight"))), G::kHidden);
        L.q     = bw(st.put(mtp.bf16(n("layers.{l}.self_attn.q_proj.weight"))), G::kQRows, G::kHidden);
        L.k     = bw(st.put(mtp.bf16(n("layers.{l}.self_attn.k_proj.weight"))), G::kKvRows, G::kHidden);
        L.v     = bw(st.put(mtp.bf16(n("layers.{l}.self_attn.v_proj.weight"))), G::kKvRows, G::kHidden);
        L.o     = bw(st.put(mtp.bf16(n("layers.{l}.self_attn.o_proj.weight"))), G::kHidden, G::kQRows);
        L.k_ctx = bw(st.put(mtp.bf16(n("layers.{l}.self_attn.k_proj_ctx.weight"))), G::kKvRows, G::kHidden);
        L.v_ctx = bw(st.put(mtp.bf16(n("layers.{l}.self_attn.v_proj_ctx.weight"))), G::kKvRows, G::kHidden);
        L.q_norm     = bt(st.put(mtp.bf16(n("layers.{l}.self_attn.q_norm.weight"))), G::kHeadDim);
        L.k_norm     = bt(st.put(mtp.bf16(n("layers.{l}.self_attn.k_norm.weight"))), G::kHeadDim);
        L.k_ctx_norm = bt(st.put(mtp.bf16(n("layers.{l}.self_attn.k_norm_ctx.weight"))), G::kHeadDim);
        L.gate = bw(st.put(mtp.bf16(n("layers.{l}.mlp.gate_proj.weight"))), G::kIntermediate, G::kHidden);
        L.up   = bw(st.put(mtp.bf16(n("layers.{l}.mlp.up_proj.weight"))), G::kIntermediate, G::kHidden);
        L.down = bw(st.put(mtp.bf16(n("layers.{l}.mlp.down_proj.weight"))), G::kHidden, G::kIntermediate);
    }
    d.markov_feature_norm    = bt(st.put(mtp.bf16("markov_head.dflare_norm.weight")), G::kHidden);
    d.markov_hidden_out      = bw(st.put(mtp.bf16("markov_head.hidden_out.weight")), G::kHidden, G::kRank);
    d.markov_hidden_out_bias = bt(st.put(mtp.bf16("markov_head.hidden_out.bias")), G::kHidden);
    d.markov_w1              = bw(st.put(mtp.bf16("markov_head.markov_w1.weight")), G::kVocabPadded, G::kRank);
    d.markov_rnn_proj        = bw(st.put(mtp.bf16("markov_head.rnn_proj.weight")), G::kRnnRows, 5120);
    d.markov_rnn_proj_bias   = bt(st.put(mtp.bf16("markov_head.rnn_proj.bias")), G::kRnnRows);

    // load-time derivations: alpha and the fold vs the reference
    D::DSparkDerived der;
    D::derive_dspark(w, der, tag == "fp8all");
    {
        const SafeTensors mixv(dir + "/dspark_feature_mix/mix64.safetensors");
        const auto ref = mixv.f32("alpha");
        const auto got = from_device<float>(der.alpha, 16);
        int worst = 0;
        for (int i = 0; i < 16; ++i) {
            std::int32_t a, b;
            std::memcpy(&a, &got[i], 4);
            std::memcpy(&b, &ref[i], 4);
            worst = std::max(worst, std::abs(a - b));
        }
        report("R0 fuse alpha (host softmax) vs reference", worst <= 2, "max fp32 ulp " + std::to_string(worst));
        const SafeTensors fold(dir + "/dspark_fold/fold.safetensors");
        compare_bf16("R0 fold_w (device fp64) vs reference",
                     dl_bf16(der.fold_w, static_cast<std::size_t>(G::kVocabPadded) * G::kRank), fold.bf16("fold_w"),
                     Tol{1, 1e-4, 1e-3});
        compare_bf16("R0 fold_b", dl_bf16(der.fold_b, G::kVocabPadded), fold.bf16("fold_b"), Tol{1, 1e-4, 1e-3});
        const auto ref_freq = tw.f32("inv_freq");
        const auto got_freq = from_device<float>(der.inv_freq, 64);
        int fw = 0;
        for (int i = 0; i < 64; ++i) {
            std::int32_t a, b;
            std::memcpy(&a, &got_freq[i], 4);
            std::memcpy(&b, &ref_freq[i], 4);
            fw = std::max(fw, std::abs(a - b));
        }
        report("R0 RoPE inv_freq vs reference", fw <= 1, "max fp32 ulp " + std::to_string(fw));
    }

    // the draft KV: 3 planes for B lanes; context [0, P_b) appended per lane as a prefill (B = 1, W = P_b)
    const int max_pos = *std::max_element(apos.begin(), apos.end()) + G::kBlock;
    Plane p0(B, max_pos, fp8), p1(B, max_pos, fp8), p2(B, max_pos, fp8);
    D::DSparkKv kv;
    kv.planes = {p0.view(), p1.view(), p2.view()};
    kv.max_visible_keys = static_cast<std::uint32_t>(max_pos);
    if (fp8) {
        // lane c's inkling_draft_attention reads the same pages through the pool's batch-view form (FP8 planes only)
        const Plane* pls[3] = {&p0, &p1, &p2};
        for (int l = 0; l < 3; ++l) {
            const Plane& pl = *pls[l];
            PagedKVBatchLayerView v{};
            v.k_pages      = Tensor(pl.k.data(), DType::FP8_E4M3FN, {static_cast<std::int32_t>(pl.k.size())});
            v.v_pages      = Tensor(pl.v.data(), DType::FP8_E4M3FN, {static_cast<std::int32_t>(pl.v.size())});
            v.block_tables = Tensor(pl.tables.data(), DType::I32, {pl.pages_per_lane, pl.lanes});
            v.head_dim     = G::kHeadDim;
            v.num_kv_heads = G::kKvHeads;
            v.dtype        = DType::FP8_E4M3FN;
            kv.views[static_cast<std::size_t>(l)] = v;
        }
    } else {
        kv.naive_attention = true; // lane c's kernel takes FP8 planes only
    }
    DBuf f_rnn(static_cast<std::size_t>(G::kHidden) * B * 2);
    WorkspaceArena work(std::max(D::dspark_append_workspace_bytes(1024),
                                 D::dspark_propose_workspace_bytes(B, static_cast<std::uint32_t>(max_pos))) +
                        (64u << 20));
    const auto F = feats.bf16("features"); // [pmax, 4, 2304]
    const DBuf dF = to_device(F);
    std::vector<int> all_pos(static_cast<std::size_t>(max_pos));
    for (int i = 0; i < max_pos; ++i) { all_pos[i] = i; }
    const DBuf dpos = to_device_i32(all_pos);
    for (int b = 0; b < B; ++b) {
        const DBuf row = to_device_i32(std::vector<int>{b});
        D::DSparkAppend a;
        a.features       = dF.data();
        a.feature_stride = G::kFeatureRows;
        a.width          = apos[b];
        a.batch          = 1;
        a.positions      = static_cast<const std::int32_t*>(dpos.data());
        a.lanes          = static_cast<const std::int32_t*>(row.data());
        a.table_rows     = static_cast<const std::int32_t*>(row.data());
        a.f_rnn          = f_rnn.data();
        D::dspark_append(w, der, kv, a, work, nullptr);
        work.reset();
    }
    cuda_synchronize();
    compare_bf16("R1 f_rnn after the prefill appends", dl_bf16(f_rnn.data(), static_cast<std::size_t>(G::kHidden) * B),
                 hd.bf16("f_rnn"), kTight);
    // the planes' context K vs the reference context K/V (bf16 tag: values; fp8: codes) at a few positions per lane
    {
        const SafeTensors ckv(dir + "/dspark_round/" + tag + "_context_kv.safetensors");
        std::size_t total = 0, bad = 0;
        for (int l = 0; l < 3; ++l) {
            const Plane& pl = l == 0 ? p0 : l == 1 ? p1 : p2;
            const auto K    = ckv.bf16("L" + std::to_string(l) + ".K");
            const auto pages = from_device<std::uint8_t>(pl.k.data(), pl.k.size());
            for (int b = 0; b < B; ++b) {
                for (int ps = 0; ps < apos[b]; ps += 97) {
                    const int page = (b + 1) * pl.pages_per_lane - 1 - ps / 64;
                    for (int e = 0; e < G::kKvRows; ++e) {
                        const std::size_t off = (static_cast<std::size_t>(page) * 8 + e / 128) * 64 * 128 +
                                                static_cast<std::size_t>(ps % 64) * 128 + e % 128;
                        float got;
                        if (fp8) {
                            __nv_fp8_e4m3 c;
                            std::memcpy(&c, &pages[off], 1);
                            got = static_cast<float>(c);
                        } else {
                            std::uint16_t h;
                            std::memcpy(&h, &pages[off * 2], 2);
                            got = bf(h);
                        }
                        const float ref = bf(K[static_cast<std::size_t>(ps) * G::kKvRows + e]);
                        ++total;
                        bad += std::fabs(got - ref) > 0.0626F * std::max(1.0F, std::fabs(ref)); // <= ~1 e4m3 / few bf16 ulp
                    }
                }
            }
        }
        report("R1 context K in the planes vs reference (sampled positions)", bad <= total / 100,
               std::to_string(bad) + " of " + std::to_string(total) + " beyond tolerance");
    }

    // propose for the B lanes at once, then lane by lane (batch invariance)
    const DBuf d_anch = to_device_i32(std::vector<int>(anchors.begin(), anchors.end()));
    const DBuf d_apos = to_device_i32(std::vector<int>(apos.begin(), apos.end()));
    std::vector<int> rows(static_cast<std::size_t>(B));
    for (int b = 0; b < B; ++b) { rows[b] = b; }
    const DBuf d_rows = to_device_i32(rows);
    DBuf drafts(static_cast<std::size_t>(7) * B * 4), hidden(static_cast<std::size_t>(G::kHidden) * 8 * B * 2);
    D::DSparkPropose p;
    p.anchors          = static_cast<const std::int32_t*>(d_anch.data());
    p.anchor_positions = static_cast<const std::int32_t*>(d_apos.data());
    p.lanes            = static_cast<const std::int32_t*>(d_rows.data());
    p.table_rows       = static_cast<const std::int32_t*>(d_rows.data());
    p.batch            = B;
    p.k                = 7;
    p.f_rnn            = f_rnn.data();
    p.drafts           = static_cast<std::int32_t*>(drafts.data());
    p.hidden_out       = hidden.data();
    D::dspark_propose(w, der, kv, p, work, nullptr);
    work.reset();
    cuda_synchronize();
    const auto hid = dl_bf16(hidden.data(), static_cast<std::size_t>(G::kHidden) * 8 * B);
    compare_bf16("R2 final hidden of the whole round vs reference", hid, rnd.bf16("final_hidden"),
                 Tol{64, 1.0, 2e-2}); // 3 layers of 1-ulp differences compound (through e4m3 codes with fp8kv); the
                                      // drafts below are the gate
    const auto dg   = from_device<std::int32_t>(drafts.data(), static_cast<std::size_t>(7) * B);
    const auto dref = hd.i32("drafts");
    const auto next = hd.i32("transcript_next");
    std::size_t eq = 0, hit_ref = 0, hit_ours = 0;
    std::string gs, rs;
    for (int b = 0; b < B; ++b) {
        bool ok_r = true, ok_o = true;
        for (int j = 0; j < 7; ++j) {
            const std::size_t i = static_cast<std::size_t>(b) * 7 + j;
            eq += dg[i] == dref[i];
            ok_r = ok_r && dref[i] == next[i];
            ok_o = ok_o && dg[i] == next[i];
            hit_ref += ok_r;
            hit_ours += ok_o;
            gs += std::to_string(dg[i]) + " ";
            rs += std::to_string(dref[i]) + " ";
        }
        gs += "| ";
        rs += "| ";
    }
    report(std::string("R2 drafts of the whole round == reference (>= 19 / 21; attention ") +
               (kv.naive_attention ? "naive" : "lane c inkling_draft_attention") + ")",
           eq >= 19,
           std::to_string(eq) + " / 21 [" + gs + "] vs [" + rs + "]; transcript prefix hits ours " +
               std::to_string(hit_ours) + " ref " + std::to_string(hit_ref));
    std::size_t same_h = 0, same_d = 0;
    for (int b = 0; b < B; ++b) {
        const DBuf a1 = to_device_i32(std::vector<int>{anchors[b]}), p1v = to_device_i32(std::vector<int>{apos[b]}),
                   r1 = to_device_i32(std::vector<int>{b});
        DBuf d1(7 * 4), h1(static_cast<std::size_t>(G::kHidden) * 8 * 2);
        D::DSparkPropose q = p;
        q.anchors          = static_cast<const std::int32_t*>(a1.data());
        q.anchor_positions = static_cast<const std::int32_t*>(p1v.data());
        q.lanes            = static_cast<const std::int32_t*>(r1.data());
        q.table_rows       = static_cast<const std::int32_t*>(r1.data());
        q.batch            = 1;
        q.drafts           = static_cast<std::int32_t*>(d1.data());
        q.hidden_out       = h1.data();
        D::dspark_propose(w, der, kv, q, work, nullptr);
        work.reset();
        cuda_synchronize();
        const auto h1v = dl_bf16(h1.data(), static_cast<std::size_t>(G::kHidden) * 8);
        const auto d1v = from_device<std::int32_t>(d1.data(), 7);
        same_h += std::equal(h1v.begin(), h1v.end(), hid.begin() + static_cast<std::ptrdiff_t>(b) * G::kHidden * 8);
        same_d += std::equal(d1v.begin(), d1v.end(), dg.begin() + static_cast<std::ptrdiff_t>(b) * 7);
    }
    report("R3 propose batch-invariant (each lane alone == B = 3, hidden and drafts bitwise)",
           same_h == static_cast<std::size_t>(B) && same_d == static_cast<std::size_t>(B),
           std::to_string(same_h) + " / " + std::to_string(same_d) + " of " + std::to_string(B) + " lanes");
    if (!kv.naive_attention) {
        // A / B: the same round with the naive attention (info: P rounding under another max order may flip ties)
        D::DSparkKv kn = kv;
        kn.naive_attention = true;
        DBuf dn(static_cast<std::size_t>(7) * B * 4);
        D::DSparkPropose q = p;
        q.drafts     = static_cast<std::int32_t*>(dn.data());
        q.hidden_out = nullptr;
        D::dspark_propose(w, der, kn, q, work, nullptr);
        work.reset();
        cuda_synchronize();
        const auto dv = from_device<std::int32_t>(dn.data(), static_cast<std::size_t>(7) * B);
        std::size_t same_c = 0, same_r = 0;
        for (std::size_t i = 0; i < dv.size(); ++i) {
            same_c += dv[i] == dg[i];
            same_r += dv[i] == dref[i];
        }
        report("R4 naive-attention round (info): drafts == lane c's / == reference", same_r >= 19,
               std::to_string(same_c) + " / 21 and " + std::to_string(same_r) + " / 21");
    }
    if (tag == "fp8all") {
        // R5: the FP8 Markov fold (opt-in lever, M1 `fp8fold`) on top: drafts vs the reference (>= 19 / 21)
        D::DSparkDerived der8;
        D::derive_dspark(w, der8, true, true);
        DBuf d8(static_cast<std::size_t>(7) * B * 4);
        D::DSparkPropose q = p;
        q.drafts     = static_cast<std::int32_t*>(d8.data());
        q.hidden_out = nullptr;
        D::dspark_propose(w, der8, kv, q, work, nullptr);
        work.reset();
        cuda_synchronize();
        const auto dv = from_device<std::int32_t>(d8.data(), static_cast<std::size_t>(7) * B);
        std::size_t same_r = 0, same_c = 0;
        for (std::size_t i = 0; i < dv.size(); ++i) {
            same_r += dv[i] == dref[i];
            same_c += dv[i] == dg[i];
        }
        report("R5 FP8 fold round: drafts == reference (>= 19 / 21) / == BF16-fold round", same_r >= 19,
               std::to_string(same_r) + " / 21 and " + std::to_string(same_c) + " / 21");
    }
    std::cout << (failures ? "FAIL" : "ALL PASS") << " dspark round (" << tag << ", " << failures << " failures)\n";
    return failures ? 1 : 0;
}

} // namespace round_mode

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (argc < 2) {
        std::cout << "SKIP: usage: sinfer_dspark_test <dir> [bf16|fp8kv] (golden vectors + mtp.safetensors)\n";
        return 77;
    }
    const std::string dir = argv[1];
    const std::string tag = argc > 2 ? argv[2] : "bf16";
    const bool fp8        = tag == "fp8kv" || tag == "fp8all";
    g_fp8_weights         = tag == "fp8all";
    if (argc > 3 && std::string(argv[3]) == "round") { return round_mode::run(dir, tag); }
    const SafeTensors mtp(dir + "/mtp.safetensors");
    const float eps = G::kEps;
    float inv_freq_h[64];
    ops::dspark_rope_inv_freq(G::kRopeTheta, inv_freq_h);
    const DBuf inv_freq = to_device(std::vector<float>(inv_freq_h, inv_freq_h + 64));

    // ---- T1 feature mix (4 rows) ----
    {
        const SafeTensors v(dir + "/dspark_feature_mix/mix64.safetensors");
        const auto feats = v.bf16("features");
        const int T      = static_cast<int>(v.entry("features").shape[0]);
        std::vector<std::uint16_t> norms;
        for (int l = 0; l < 3; ++l) {
            const auto w = mtp.bf16(layer_name("layers.{l}.dflare_norm.weight", l));
            norms.insert(norms.end(), w.begin(), w.end());
        }
        const auto mw = mtp.bf16("markov_head.dflare_norm.weight");
        norms.insert(norms.end(), mw.begin(), mw.end());
        const DBuf df = to_device(feats), da = to_device(v.f32("alpha")), dn = to_device(norms);
        DBuf out(static_cast<std::size_t>(G::kHidden) * T * 4 * 2);
        ops::dspark_feature_mix(df.data(), G::kFeatureRows, T, static_cast<const float*>(da.data()), dn.data(), 4, eps,
                                out.data(), nullptr);
        cuda_synchronize();
        compare_bf16("T1 feature_mix 4 rows x " + std::to_string(T), dl_bf16(out.data(), v.bf16("out").size()),
                     v.bf16("out"), kGemm);
    }

    // ---- T2 context K / V per draft layer ----
    {
        const SafeTensors v(dir + "/dspark_context_kv/" + tag + "_ctx64.safetensors");
        const auto pos = v.i32("positions");
        const int T    = static_cast<int>(pos.size());
        const DBuf dpos = to_device_i32(std::vector<int>(pos.begin(), pos.end()));
        for (int l = 0; l < 3; ++l) {
            const std::string p = "L" + std::to_string(l) + ".";
            const DBuf w = to_device(cat_rows({mtp.bf16(layer_name("layers.{l}.self_attn.k_proj_ctx.weight", l)),
                                                       mtp.bf16(layer_name("layers.{l}.self_attn.v_proj_ctx.weight", l))}));
            const DBuf c = to_device(v.bf16(p + "c"));
            DBuf kv(static_cast<std::size_t>(2 * G::kKvRows) * T * 2);
            gemm_dense(w, 2 * G::kKvRows, G::kHidden, c.data(), T, kv.data());
            cuda_synchronize();
            compare_bf16("T2 L" + std::to_string(l) + " kv_ctx GEMM", dl_bf16(kv.data(), v.bf16(p + "kv").size()),
                         v.bf16(p + "kv"), kGemm);
            // norm (+ rope) on the reference's GEMM output
            const DBuf kv_ref = to_device(v.bf16(p + "kv"));
            const DBuf kn     = to_device(mtp.bf16(layer_name("layers.{l}.self_attn.k_norm_ctx.weight", l)));
            DBuf normed(static_cast<std::size_t>(G::kKvRows) * T * 2), roped(normed.size());
            ops::dspark_head_norm_rope(kv_ref.data(), 2 * G::kKvRows, G::kKvHeads, T, kn.data(), nullptr, nullptr, eps,
                                       normed.data(), G::kKvRows, nullptr);
            ops::dspark_head_norm_rope(kv_ref.data(), 2 * G::kKvRows, G::kKvHeads, T, kn.data(),
                                       static_cast<const std::int32_t*>(dpos.data()),
                                       static_cast<const float*>(inv_freq.data()), eps, roped.data(), G::kKvRows,
                                       nullptr);
            cuda_synchronize();
            compare_bf16("T2 L" + std::to_string(l) + " k_ctx_norm", dl_bf16(normed.data(), v.bf16(p + "k_normed").size()),
                         v.bf16(p + "k_normed"), kTight);
            if (!fp8) {
                compare_bf16("T2 L" + std::to_string(l) + " k norm + rope", dl_bf16(roped.data(), v.bf16(p + "k").size()),
                             v.bf16(p + "k"), kTight);
            } else {
                // the reference's K / V are the e4m3 roundings: write ours through an FP8 plane and compare codes
                Plane plane(1, T + pos[0], true);
                const std::vector<std::uint16_t> vref = v.bf16(p + "v");
                const DBuf dv = to_device(vref);
                const std::vector<int> rows{0};
                const DBuf drow = to_device_i32(rows);
                ops::dspark_kv_write(roped.data(), dv.data(), G::kKvRows, static_cast<const std::int32_t*>(dpos.data()),
                                     T, 1, nullptr, static_cast<const std::int32_t*>(drow.data()), plane.view(), nullptr);
                cuda_synchronize();
                const auto kpages = from_device<std::uint8_t>(plane.k.data(), plane.k.size());
                const auto codes  = v.read<std::uint8_t>(p + "k_codes");
                std::vector<int> tab(static_cast<std::size_t>(plane.pages_per_lane));
                for (int pg = 0; pg < plane.pages_per_lane; ++pg) { tab[pg] = plane.pages_per_lane - 1 - pg; }
                std::size_t bad = 0;
                for (int t = 0; t < T; ++t) {
                    const int ps = pos[t];
                    for (int e = 0; e < G::kKvRows; ++e) {
                        const std::size_t off = (static_cast<std::size_t>(tab[ps / 64]) * 8 + e / 128) * 64 * 128 +
                                                static_cast<std::size_t>(ps % 64) * 128 + e % 128;
                        bad += kpages[off] != codes[static_cast<std::size_t>(t) * G::kKvRows + e];
                    }
                }
                report("T2 L" + std::to_string(l) + " K e4m3 codes via kv_write", bad <= static_cast<std::size_t>(T) * 10,
                       std::to_string(bad) + " of " + std::to_string(T * G::kKvRows) + " codes differ (1-ulp bf16 rope)");
            }
            if (!fp8) {
                compare_bf16("T2 L" + std::to_string(l) + " v (raw GEMM rows)",
                             [&] {
                                 const auto all = v.bf16(p + "kv");
                                 std::vector<std::uint16_t> out;
                                 for (int t = 0; t < T; ++t) {
                                     const auto* s = all.data() + static_cast<std::size_t>(t) * 2 * G::kKvRows + G::kKvRows;
                                     out.insert(out.end(), s, s + G::kKvRows);
                                 }
                                 return out;
                             }(),
                             v.bf16(p + "v"), kTight);
            }
        }
    }

    // ---- T3 one draft round: embed block, the 3 layers op by op, final norm ----
    const SafeTensors rnd(dir + "/dspark_round/" + tag + "_block_b3.safetensors");
    const SafeTensors ctxkv(dir + "/dspark_round/" + tag + "_context_kv.safetensors");
    const SafeTensors tw(dir + "/dspark_round/test_weights.safetensors");
    const auto anchors_tok = rnd.i32("anchors");
    const auto anchor_pos  = rnd.i32("anchor_pos");
    const int B            = static_cast<int>(anchor_pos.size());
    const int N            = G::kBlock * B;
    const DBuf d_anchor_pos = to_device_i32(std::vector<int>(anchor_pos.begin(), anchor_pos.end()));
    std::vector<int> block_positions(static_cast<std::size_t>(N));
    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < G::kBlock; ++j) { block_positions[b * G::kBlock + j] = anchor_pos[b] + j; }
    }
    const DBuf d_block_pos = to_device_i32(block_positions);
    {
        // embed rows of the anchors only (test_weights): remap the anchors to rows 0..B-1
        const auto embed_rows = tw.bf16("embed_rows");
        std::vector<int> local(static_cast<std::size_t>(B));
        for (int b = 0; b < B; ++b) { local[b] = b; }
        const DBuf de = to_device(embed_rows), dl = to_device_i32(local);
        const DBuf dn = to_device(tw.bf16("embed_norm")), dm = to_device(mtp.bf16("mask_embed"));
        DBuf out(static_cast<std::size_t>(G::kHidden) * N * 2);
        ops::dspark_embed_block(static_cast<const std::int32_t*>(dl.data()), B, de.data(), B, dn.data(), dm.data(), eps,
                                out.data(), nullptr);
        cuda_synchronize();
        compare_bf16("T3 embed_block", dl_bf16(out.data(), rnd.bf16("block_in").size()), rnd.bf16("block_in"), kTight);
    }
    std::vector<int> lane_rows(static_cast<std::size_t>(B));
    for (int b = 0; b < B; ++b) { lane_rows[b] = b; }
    const DBuf d_lane_rows = to_device_i32(lane_rows);
    const int max_pos = *std::max_element(anchor_pos.begin(), anchor_pos.end()) + G::kBlock;
    for (int l = 0; l < 3; ++l) {
        const std::string p  = "L" + std::to_string(l) + ".";
        const std::string tl = "T3 L" + std::to_string(l) + " ";
        const bool local     = l != 1;
        // input norm (fused add for l > 0)
        {
            const DBuf w = to_device(mtp.bf16(layer_name("layers.{l}.input_layernorm.weight", l)));
            DBuf h(static_cast<std::size_t>(G::kHidden) * N * 2), r(h.size());
            if (l == 0) {
                const DBuf x = to_device(rnd.bf16("block_in"));
                ops::dspark_add_rms_norm(nullptr, x.data(), w.data(), eps, N, nullptr, h.data(), nullptr);
            } else {
                const DBuf x = to_device(rnd.bf16(p + "x_in")), res = to_device(rnd.bf16(p + "resid_in"));
                ops::dspark_add_rms_norm(x.data(), res.data(), w.data(), eps, N, r.data(), h.data(), nullptr);
                cuda_synchronize();
                compare_bf16(tl + "add (resid)", dl_bf16(r.data(), rnd.bf16(p + "resid").size()), rnd.bf16(p + "resid"),
                             Tol{0, 0.0, 0.0});
            }
            cuda_synchronize();
            compare_bf16(tl + "input norm", dl_bf16(h.data(), rnd.bf16(p + "h_in").size()), rnd.bf16(p + "h_in"), kTight);
        }
        // qkv GEMM, q / k norm + RoPE
        {
            const DBuf w = to_device(cat_rows({mtp.bf16(layer_name("layers.{l}.self_attn.q_proj.weight", l)),
                                                       mtp.bf16(layer_name("layers.{l}.self_attn.k_proj.weight", l)),
                                                       mtp.bf16(layer_name("layers.{l}.self_attn.v_proj.weight", l))}));
            const DBuf h = to_device(rnd.bf16(p + "h_in"));
            DBuf qkv(static_cast<std::size_t>(G::kQkvRows) * N * 2);
            gemm_dense(w, G::kQkvRows, G::kHidden, h.data(), N, qkv.data());
            cuda_synchronize();
            compare_bf16(tl + "qkv GEMM", dl_bf16(qkv.data(), rnd.bf16(p + "qkv").size()), rnd.bf16(p + "qkv"), kGemm);
            const DBuf ref_qkv = to_device(rnd.bf16(p + "qkv"));
            const DBuf qn = to_device(mtp.bf16(layer_name("layers.{l}.self_attn.q_norm.weight", l)));
            const DBuf kn = to_device(mtp.bf16(layer_name("layers.{l}.self_attn.k_norm.weight", l)));
            DBuf q(static_cast<std::size_t>(G::kQRows) * N * 2), k(static_cast<std::size_t>(G::kKvRows) * N * 2);
            ops::dspark_head_norm_rope(ref_qkv.data(), G::kQkvRows, G::kQHeads, N, qn.data(),
                                       static_cast<const std::int32_t*>(d_block_pos.data()),
                                       static_cast<const float*>(inv_freq.data()), eps, q.data(), G::kQRows, nullptr);
            ops::dspark_head_norm_rope(static_cast<const std::uint16_t*>(ref_qkv.data()) + G::kQRows, G::kQkvRows,
                                       G::kKvHeads, N, kn.data(), static_cast<const std::int32_t*>(d_block_pos.data()),
                                       static_cast<const float*>(inv_freq.data()), eps, k.data(), G::kKvRows, nullptr);
            cuda_synchronize();
            compare_bf16(tl + "q norm + rope", dl_bf16(q.data(), rnd.bf16(p + "q").size()), rnd.bf16(p + "q"), kTight);
            if (!fp8) {
                compare_bf16(tl + "k norm + rope", dl_bf16(k.data(), rnd.bf16(p + "k").size()), rnd.bf16(p + "k"), kTight);
            }
        }
        // attention over the paged plane: context [0, P) from the reference, then the block's own K / V
        {
            Plane plane(B, max_pos, fp8);
            const auto Kc = ctxkv.bf16(p + "K"), Vc = ctxkv.bf16(p + "V");
            const DBuf dK = to_device(Kc), dV = to_device(Vc);
            for (int b = 0; b < B; ++b) {
                std::vector<int> ctx_pos(static_cast<std::size_t>(anchor_pos[b]));
                for (int i = 0; i < anchor_pos[b]; ++i) { ctx_pos[i] = i; }
                const DBuf dp = to_device_i32(ctx_pos), drow = to_device_i32(std::vector<int>{b});
                ops::dspark_kv_write(dK.data(), dV.data(), G::kKvRows, static_cast<const std::int32_t*>(dp.data()),
                                     anchor_pos[b], 1, nullptr, static_cast<const std::int32_t*>(drow.data()), plane.view(),
                                     nullptr);
            }
            const DBuf bk = to_device(rnd.bf16(p + "k")), bv = to_device(rnd.bf16(p + "v"));
            ops::dspark_kv_write(bk.data(), bv.data(), G::kKvRows, static_cast<const std::int32_t*>(d_block_pos.data()),
                                 G::kBlock, B, nullptr, static_cast<const std::int32_t*>(d_lane_rows.data()), plane.view(),
                                 nullptr);
            const DBuf q = to_device(rnd.bf16(p + "q"));
            DBuf out(static_cast<std::size_t>(G::kQRows) * N * 2);
            const int window = local ? G::kBlockWindow : 0;
            ops::dspark_block_attention(q.data(), G::kQRows, static_cast<const std::int32_t*>(d_anchor_pos.data()),
                                        static_cast<const std::int32_t*>(d_lane_rows.data()), B, window, G::kScale,
                                        plane.view(), out.data(), nullptr);
            cuda_synchronize();
            const auto got = dl_bf16(out.data(), rnd.bf16(p + "attn").size());
            compare_bf16(tl + "block attention (" + std::string(local ? "local" : "global") + ")", got,
                         rnd.bf16(p + "attn"), kAttn);
            // batch invariance: each lane alone gives the same bits
            std::size_t differ = 0;
            for (int b = 0; b < B; ++b) {
                DBuf one(static_cast<std::size_t>(G::kQRows) * G::kBlock * 2);
                const DBuf drow = to_device_i32(std::vector<int>{b});
                const DBuf dap  = to_device_i32(std::vector<int>{anchor_pos[b]});
                ops::dspark_block_attention(static_cast<const std::uint16_t*>(q.data()) +
                                                static_cast<std::size_t>(b) * G::kBlock * G::kQRows,
                                            G::kQRows, static_cast<const std::int32_t*>(dap.data()),
                                            static_cast<const std::int32_t*>(drow.data()), 1, window, G::kScale,
                                            plane.view(), one.data(), nullptr);
                cuda_synchronize();
                const auto alone = dl_bf16(one.data(), static_cast<std::size_t>(G::kQRows) * G::kBlock);
                for (std::size_t i = 0; i < alone.size(); ++i) {
                    differ += alone[i] != got[static_cast<std::size_t>(b) * G::kBlock * G::kQRows + i];
                }
            }
            report(tl + "attention batch-invariant (B=3 vs alone)", differ == 0, std::to_string(differ) + " differ");
        }
        // o GEMM, post norm, MLP
        {
            const DBuf wo = to_device(mtp.bf16(layer_name("layers.{l}.self_attn.o_proj.weight", l)));
            const DBuf a  = to_device(rnd.bf16(p + "attn"));
            DBuf o(static_cast<std::size_t>(G::kHidden) * N * 2);
            gemm_dense(wo, G::kHidden, G::kQRows, a.data(), N, o.data());
            cuda_synchronize();
            compare_bf16(tl + "o GEMM", dl_bf16(o.data(), rnd.bf16(p + "o").size()), rnd.bf16(p + "o"), kGemm);
            const DBuf w   = to_device(mtp.bf16(layer_name("layers.{l}.post_attention_layernorm.weight", l)));
            const DBuf ro  = to_device(rnd.bf16(p + "o")), res = to_device(rnd.bf16(p + "resid"));
            DBuf h(static_cast<std::size_t>(G::kHidden) * N * 2), r(h.size());
            ops::dspark_add_rms_norm(ro.data(), res.data(), w.data(), eps, N, r.data(), h.data(), nullptr);
            cuda_synchronize();
            compare_bf16(tl + "add (resid_mid)", dl_bf16(r.data(), rnd.bf16(p + "resid_mid").size()),
                         rnd.bf16(p + "resid_mid"), Tol{0, 0.0, 0.0});
            compare_bf16(tl + "post norm", dl_bf16(h.data(), rnd.bf16(p + "h_post").size()), rnd.bf16(p + "h_post"), kTight);
            const DBuf wgu = to_device(cat_rows({mtp.bf16(layer_name("layers.{l}.mlp.gate_proj.weight", l)),
                                                         mtp.bf16(layer_name("layers.{l}.mlp.up_proj.weight", l))}));
            const DBuf hp  = to_device(rnd.bf16(p + "h_post"));
            DBuf gu(static_cast<std::size_t>(2 * G::kIntermediate) * N * 2);
            gemm_dense(wgu, 2 * G::kIntermediate, G::kHidden, hp.data(), N, gu.data());
            cuda_synchronize();
            compare_bf16(tl + "gate_up GEMM", dl_bf16(gu.data(), rnd.bf16(p + "gate_up").size()), rnd.bf16(p + "gate_up"),
                         kGemm);
            const DBuf rgu = to_device(rnd.bf16(p + "gate_up"));
            DBuf y(static_cast<std::size_t>(G::kIntermediate) * N * 2);
            ops::dspark_silu_mul(rgu.data(), N, y.data(), nullptr);
            cuda_synchronize();
            compare_bf16(tl + "silu * up", dl_bf16(y.data(), rnd.bf16(p + "act").size()), rnd.bf16(p + "act"), kTight);
            const DBuf wd = to_device(mtp.bf16(layer_name("layers.{l}.mlp.down_proj.weight", l)));
            const DBuf ra = to_device(rnd.bf16(p + "act"));
            DBuf mo(static_cast<std::size_t>(G::kHidden) * N * 2);
            gemm_dense(wd, G::kHidden, G::kIntermediate, ra.data(), N, mo.data());
            cuda_synchronize();
            compare_bf16(tl + "down GEMM", dl_bf16(mo.data(), rnd.bf16(p + "mlp").size()), rnd.bf16(p + "mlp"), kGemm);
        }
    }
    {
        const DBuf w = to_device(mtp.bf16("norm.weight"));
        const DBuf x = to_device(rnd.bf16("final_x")), r = to_device(rnd.bf16("final_resid"));
        DBuf h(static_cast<std::size_t>(G::kHidden) * N * 2);
        ops::dspark_add_rms_norm(x.data(), r.data(), w.data(), eps, N, nullptr, h.data(), nullptr);
        cuda_synchronize();
        compare_bf16("T3 final norm", dl_bf16(h.data(), rnd.bf16("final_hidden").size()), rnd.bf16("final_hidden"), kTight);
    }

    // ---- T4 the head: base logits (unembed slice), f_rnn, Markov context, gate / vocab / pick per step, chain ----
    {
        const SafeTensors hd(dir + "/dspark_head/" + tag + "_b3.safetensors");
        const int k      = G::kMaxDrafts;
        const int vpad   = G::kVocabPadded;
        const int domain = G::kVocab;
        const auto base  = hd.bf16("base"); // [B, 7, Vpad]
        // base logits on the unembed slice of test_weights (rows [0, S))
        {
            const auto rows = tw.bf16("unembed_rows");
            const int S     = static_cast<int>(tw.entry("unembed_rows").shape[0]);
            const DBuf w = to_device(rows), x = to_device(hd.bf16("hidden")); // [B, 7, 2304] = [2304, 7B]
            DBuf out(static_cast<std::size_t>(S) * k * B * 2);
            gemm(w, S, G::kHidden, x.data(), k * B, out.data(), DType::BF16);
            cuda_synchronize();
            const auto got = dl_bf16(out.data(), static_cast<std::size_t>(S) * k * B);
            std::vector<std::uint16_t> ref;
            for (int c = 0; c < k * B; ++c) {
                const auto* s = base.data() + static_cast<std::size_t>(c) * vpad;
                ref.insert(ref.end(), s, s + S);
            }
            compare_bf16("T4 base logits GEMM (rows < " + std::to_string(S) + ")", got, ref, kGemm);
        }
        // f_rnn = mix row 3 of the features before the anchor
        {
            std::vector<std::uint16_t> norms;
            for (int l = 0; l < 3; ++l) {
                const auto w = mtp.bf16(layer_name("layers.{l}.dflare_norm.weight", l));
                norms.insert(norms.end(), w.begin(), w.end());
            }
            const auto mw = mtp.bf16("markov_head.dflare_norm.weight");
            norms.insert(norms.end(), mw.begin(), mw.end());
            const SafeTensors mixv(dir + "/dspark_feature_mix/mix64.safetensors");
            const DBuf df = to_device(hd.bf16("features_prev")), da = to_device(mixv.f32("alpha")),
                               dn = to_device(norms);
            DBuf out(static_cast<std::size_t>(G::kHidden) * B * 4 * 2);
            ops::dspark_feature_mix(df.data(), G::kFeatureRows, B, static_cast<const float*>(da.data()), dn.data(), 4,
                                    eps, out.data(), nullptr);
            cuda_synchronize();
            const auto all = dl_bf16(out.data(), static_cast<std::size_t>(G::kHidden) * B * 4);
            compare_bf16("T4 f_rnn (mix row 3)", slice(all, static_cast<std::size_t>(3) * G::kHidden * B,
                                                       static_cast<std::size_t>(G::kHidden) * B),
                         hd.bf16("f_rnn"), kTight);
        }
        // Markov context: pack [hidden rows 1..7 ; f_rnn] and the [768, 4608] GEMM (FP32), + bias on the host
        const auto rnn_w   = mtp.bf16("markov_head.rnn_proj.weight"); // [768, 5120]
        std::vector<std::uint16_t> w_sp, w_hf;
        for (int n = 0; n < G::kRnnRows; ++n) {
            const auto* row = rnn_w.data() + static_cast<std::size_t>(n) * 5120;
            w_sp.insert(w_sp.end(), row, row + 512);
            w_hf.insert(w_hf.end(), row + 512, row + 5120);
        }
        const auto rnn_b_bits = mtp.bf16("markov_head.rnn_proj.bias");
        const DBuf d_wsp = to_device(w_sp), d_whf = to_device(w_hf), d_rb = to_device(rnn_b_bits);
        DBuf ctx(static_cast<std::size_t>(G::kRnnRows) * k * B * 4);
        {
            const DBuf fh = to_device(rnd.bf16("final_hidden")), fr = to_device(hd.bf16("f_rnn"));
            DBuf x(static_cast<std::size_t>(2 * G::kHidden) * k * B * 2);
            ops::dspark_markov_pack(fh.data(), fr.data(), nullptr, k, B, x.data(), nullptr);
            gemm(d_whf, G::kRnnRows, 2 * G::kHidden, x.data(), k * B, ctx.data(), DType::FP32);
            cuda_synchronize();
            auto got = from_device<float>(ctx.data(), static_cast<std::size_t>(G::kRnnRows) * k * B);
            for (std::size_t i = 0; i < got.size(); ++i) { got[i] += bf(rnn_b_bits[i % G::kRnnRows]); }
            compare_f32("T4 Markov context (fp32)", got, hd.f32("markov_ctx"), 1e-5);
        }
        // per step on the reference's inputs
        const DBuf d_w1 = to_device(mtp.bf16("markov_head.markov_w1.weight"));
        const SafeTensors fold(dir + "/dspark_fold/fold.safetensors");
        const DBuf d_fw = to_device(fold.bf16("fold_w")), d_fb = to_device(fold.bf16("fold_b"));
        std::vector<std::uint16_t> base8(static_cast<std::size_t>(vpad) * G::kBlock * B, 0);
        for (int b = 0; b < B; ++b) {
            for (int j = 0; j < k; ++j) {
                std::memcpy(base8.data() + (static_cast<std::size_t>(b) * G::kBlock + 1 + j) * vpad,
                            base.data() + (static_cast<std::size_t>(b) * k + j) * vpad, static_cast<std::size_t>(vpad) * 2);
            }
        }
        const DBuf d_base = to_device(base8);
        const auto prev_ref   = hd.i32("prev");    // [B, 7]
        const auto state_ref  = hd.bf16("state");  // [B, 7, 256]
        const auto latent_ref = hd.bf16("latent");
        const auto bias_ref   = hd.bf16("bias");   // [B, 7, Vpad]
        const auto logit_ref  = hd.bf16("logits"); // [B, 7, domain]
        const auto drafts_ref = hd.i32("drafts");  // [B, 7]
        const int tiles       = ops::dspark_vocab_tiles(domain);
        DBuf part_v(static_cast<std::size_t>(tiles) * B * 4), part_i(part_v.size());
        DBuf state(static_cast<std::size_t>(G::kRank) * B * 2), state2(state.size()), latent(state.size());
        DBuf drafts(static_cast<std::size_t>(k) * B * 4);
        DBuf bias_out(static_cast<std::size_t>(domain) * B * 2), logits_out(bias_out.size());
        std::size_t state_diff = 0, latent_diff = 0, draft_equal = 0;
        std::vector<std::uint16_t> bias_got_all, bias_ref_all, logit_got_all, logit_ref_all;
        for (int j = 0; j < k; ++j) {
            std::vector<int> prev(static_cast<std::size_t>(B));
            std::vector<std::uint16_t> st(static_cast<std::size_t>(G::kRank) * B), lat_ref(st.size()), st_ref(st.size());
            for (int b = 0; b < B; ++b) {
                prev[b] = prev_ref[static_cast<std::size_t>(b) * k + j];
                for (int r = 0; r < G::kRank; ++r) {
                    const std::size_t at = (static_cast<std::size_t>(b) * k + j) * G::kRank + r;
                    st[static_cast<std::size_t>(b) * G::kRank + r] = j > 0 ? state_ref[at - G::kRank] : 0;
                    st_ref[static_cast<std::size_t>(b) * G::kRank + r]  = state_ref[at];
                    lat_ref[static_cast<std::size_t>(b) * G::kRank + r] = latent_ref[at];
                }
            }
            const DBuf dprev = to_device_i32(prev);
            cuda_check(cudaMemcpy(state.data(), st.data(), st.size() * 2, cudaMemcpyHostToDevice), "state");
            ops::dspark_markov_gate(static_cast<const float*>(ctx.data()), j, k, B, d_rb.data(), d_wsp.data(), d_w1.data(),
                                    static_cast<const std::int32_t*>(dprev.data()), 1, j == 0 ? nullptr : state.data(),
                                    state2.data(), latent.data(), nullptr);
            cuda_synchronize();
            const auto sg = dl_bf16(state2.data(), st.size());
            const auto lg = dl_bf16(latent.data(), st.size());
            for (std::size_t i = 0; i < sg.size(); ++i) { // raw from a differently ordered fp32 dot: 1-2 ulp flips
                state_diff += std::abs(ordered(sg[i]) - ordered(st_ref[i])) > 2;
                latent_diff += std::abs(ordered(lg[i]) - ordered(lat_ref[i])) > 2;
            }
            // vocabulary pass on the reference latent
            const DBuf dlat = to_device(lat_ref);
            ops::dspark_markov_vocab(d_base.data(), vpad, 1 + j, B, d_fw.data(), d_fb.data(), dlat.data(), domain, nullptr,
                                     static_cast<float*>(part_v.data()), static_cast<std::int32_t*>(part_i.data()),
                                     bias_out.data(), logits_out.data(), nullptr);
            ops::dspark_markov_pick(static_cast<const float*>(part_v.data()), static_cast<const std::int32_t*>(part_i.data()),
                                    tiles, B, j, k, static_cast<std::int32_t*>(drafts.data()), nullptr);
            cuda_synchronize();
            const auto bg = dl_bf16(bias_out.data(), static_cast<std::size_t>(domain) * B);
            const auto og = dl_bf16(logits_out.data(), static_cast<std::size_t>(domain) * B);
            const auto dg = from_device<std::int32_t>(drafts.data(), static_cast<std::size_t>(k) * B);
            for (int b = 0; b < B; ++b) {
                const auto* bs = bias_ref.data() + (static_cast<std::size_t>(b) * k + j) * vpad;
                const auto* ls = logit_ref.data() + (static_cast<std::size_t>(b) * k + j) * domain;
                bias_got_all.insert(bias_got_all.end(), bg.begin() + static_cast<std::ptrdiff_t>(b) * domain,
                                    bg.begin() + static_cast<std::ptrdiff_t>(b + 1) * domain);
                bias_ref_all.insert(bias_ref_all.end(), bs, bs + domain);
                logit_got_all.insert(logit_got_all.end(), og.begin() + static_cast<std::ptrdiff_t>(b) * domain,
                                     og.begin() + static_cast<std::ptrdiff_t>(b + 1) * domain);
                logit_ref_all.insert(logit_ref_all.end(), ls, ls + domain);
                draft_equal += dg[static_cast<std::size_t>(b) * k + j] == drafts_ref[static_cast<std::size_t>(b) * k + j];
            }
        }
        // A raw that crosses a bf16 boundary under the other fp32 dot order moves g / c / o by one ulp, and sigmoid /
        // tanh then several output ulps: allowed on <= 0.1 % of the 7 x B x 256 values.
        const std::size_t gate_n = static_cast<std::size_t>(k) * B * G::kRank;
        report("T4 Markov gate state (ref inputs, > 2 ulp on <= 0.1 %)", state_diff * 1000 <= gate_n,
               std::to_string(state_diff) + " of " + std::to_string(gate_n) + " over 2 ulp");
        report("T4 Markov gate latent (ref inputs, > 2 ulp on <= 0.1 %)", latent_diff * 1000 <= gate_n,
               std::to_string(latent_diff) + " of " + std::to_string(gate_n) + " over 2 ulp");
        compare_bf16("T4 Markov bias (fold . latent)", bias_got_all, bias_ref_all, kGemm);
        compare_bf16("T4 Markov logits (softcap)", logit_got_all, logit_ref_all, kGemm);
        report("T4 drafts per step on ref inputs", draft_equal == static_cast<std::size_t>(k * B),
               std::to_string(draft_equal) + " / " + std::to_string(k * B) + " equal");
        // the chain end to end on our own outputs (step 0: prev = anchors, zero state)
        {
            const DBuf d_anchor_tok = to_device_i32(std::vector<int>(anchors_tok.begin(), anchors_tok.end()));
            for (int j = 0; j < k; ++j) {
                const std::int32_t* prev = j == 0 ? static_cast<const std::int32_t*>(d_anchor_tok.data())
                                                  : static_cast<const std::int32_t*>(drafts.data()) + (j - 1);
                void* buf[2] = {state.data(), state2.data()};
                ops::dspark_markov_gate(static_cast<const float*>(ctx.data()), j, k, B, d_rb.data(), d_wsp.data(),
                                        d_w1.data(), prev, j == 0 ? 1 : k, j == 0 ? nullptr : buf[(j + 1) & 1],
                                        buf[j & 1], latent.data(), nullptr);
                ops::dspark_markov_vocab(d_base.data(), vpad, 1 + j, B, d_fw.data(), d_fb.data(), latent.data(), domain,
                                         nullptr, static_cast<float*>(part_v.data()),
                                         static_cast<std::int32_t*>(part_i.data()), nullptr, nullptr, nullptr);
                ops::dspark_markov_pick(static_cast<const float*>(part_v.data()),
                                        static_cast<const std::int32_t*>(part_i.data()), tiles, B, j, k,
                                        static_cast<std::int32_t*>(drafts.data()), nullptr);
            }
            cuda_synchronize();
            const auto dg = from_device<std::int32_t>(drafts.data(), static_cast<std::size_t>(k) * B);
            std::size_t eq = 0;
            std::string got_s, ref_s;
            for (std::size_t i = 0; i < dg.size(); ++i) {
                eq += dg[i] == drafts_ref[i];
                got_s += std::to_string(dg[i]) + " ";
                ref_s += std::to_string(drafts_ref[i]) + " ";
            }
            report("T4 Markov chain end to end (drafts == reference)", eq == dg.size(),
                   std::to_string(eq) + " / " + std::to_string(dg.size()) + " [" + got_s + "] vs [" + ref_s + "]");
            // batch invariance of the chain: lane 1 alone
            DBuf st1(static_cast<std::size_t>(G::kRank) * 2), st1b(st1.size()), la1(st1.size()),
                dr1(static_cast<std::size_t>(k) * 4);
            DBuf pv1(static_cast<std::size_t>(tiles) * 4), pi1(pv1.size());
            const DBuf a1 = to_device_i32(std::vector<int>{anchors_tok[1]});
            DBuf ctx1(static_cast<std::size_t>(G::kRnnRows) * k * 4);
            cuda_check(cudaMemcpy(ctx1.data(), static_cast<const float*>(ctx.data()) + static_cast<std::size_t>(G::kRnnRows) * k,
                                  ctx1.size(), cudaMemcpyDeviceToDevice),
                       "ctx1");
            DBuf base1(static_cast<std::size_t>(vpad) * G::kBlock * 2);
            cuda_check(cudaMemcpy(base1.data(), static_cast<const std::uint16_t*>(d_base.data()) +
                                                     static_cast<std::size_t>(vpad) * G::kBlock,
                                  base1.size(), cudaMemcpyDeviceToDevice),
                       "base1");
            for (int j = 0; j < k; ++j) {
                const std::int32_t* prev = j == 0 ? static_cast<const std::int32_t*>(a1.data())
                                                  : static_cast<const std::int32_t*>(dr1.data()) + (j - 1);
                void* buf1[2] = {st1.data(), st1b.data()};
                ops::dspark_markov_gate(static_cast<const float*>(ctx1.data()), j, k, 1, d_rb.data(), d_wsp.data(),
                                        d_w1.data(), prev, 1, j == 0 ? nullptr : buf1[(j + 1) & 1], buf1[j & 1], la1.data(),
                                        nullptr);
                ops::dspark_markov_vocab(base1.data(), vpad, 1 + j, 1, d_fw.data(), d_fb.data(), la1.data(), domain,
                                         nullptr, static_cast<float*>(pv1.data()), static_cast<std::int32_t*>(pi1.data()),
                                         nullptr, nullptr, nullptr);
                ops::dspark_markov_pick(static_cast<const float*>(pv1.data()), static_cast<const std::int32_t*>(pi1.data()),
                                        tiles, 1, j, k, static_cast<std::int32_t*>(dr1.data()), nullptr);
            }
            cuda_synchronize();
            const auto d1 = from_device<std::int32_t>(dr1.data(), static_cast<std::size_t>(k));
            std::size_t same = 0;
            for (int j = 0; j < k; ++j) { same += d1[j] == dg[static_cast<std::size_t>(k) + j]; }
            report("T4 Markov chain batch-invariant (lane 1 alone)", same == static_cast<std::size_t>(k),
                   std::to_string(same) + " / " + std::to_string(k));
        }
    }

    // ---- T5 the fold on the unembed slice ----
    {
        const auto rows = tw.bf16("unembed_rows");
        const int S     = static_cast<int>(tw.entry("unembed_rows").shape[0]);
        const DBuf du = to_device(rows), dh = to_device(mtp.bf16("markov_head.hidden_out.weight")),
                           db = to_device(mtp.bf16("markov_head.hidden_out.bias"));
        DBuf fw(static_cast<std::size_t>(S) * G::kRank * 2), fb(static_cast<std::size_t>(S) * 2);
        ops::dspark_fold(du.data(), S, nullptr, dh.data(), db.data(), fw.data(), fb.data(), nullptr);
        cuda_synchronize();
        const SafeTensors fold(dir + "/dspark_fold/fold.safetensors");
        compare_bf16("T5 fold_w (fp64, rows < " + std::to_string(S) + ")", dl_bf16(fw.data(), static_cast<std::size_t>(S) * G::kRank),
                     slice(fold.bf16("fold_w"), 0, static_cast<std::size_t>(S) * G::kRank), Tol{1, 1e-4, 1e-3});
        compare_bf16("T5 fold_b", dl_bf16(fb.data(), static_cast<std::size_t>(S)), slice(fold.bf16("fold_b"), 0, S),
                     Tol{1, 1e-4, 1e-3});
    }

    std::cout << (failures ? "FAIL" : "ALL PASS") << " dspark ops (" << tag << ", " << failures << " failures)\n";
    return failures ? 1 : 0;
}
