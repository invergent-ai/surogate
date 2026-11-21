// Op test of the Inkling-2-Mini attention (ENGINE-INKLING lane c): api/ops/inkling_attention.h against a
// CPU oracle of the reference numerics (lanes/c/DESIGN-attention.md section 1, test plan section 9).
//
// Oracle policy: the prologue oracle reproduces every rounding point of the reference AND the kernel's
// fp32 reduction order (the sums of squares), so its outputs agree bit for bit except where the
// rsqrt.approx instruction (not reproducible on the host) moves a clip factor by an ulp: those elements
// may differ by one unit of their storage format, and only rarely. The attention oracle takes the
// kernel's bf16 / e4m3 inputs as given, rounds the relative bias to bf16 as the reference does, and
// evaluates scores, softmax and P V in FP64; outputs are judged by the engine's attention criterion.
// Golden vectors from lane a (rl/engine-inkling-2026-11-21/vectors) join once their README exists.

#include "api/ops/inkling_attention.h"
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;
using G = ops::InklingAttentionGeometry;
using ops::InklingLayerKind;

namespace {

constexpr std::int32_t kC = 6656; // e1's per-visit conv slot: [K 1024 | V 1024 | attn 2304 | mlp 2304]

// The engine's BF16 attention criterion (test_gqa_attention.cpp) for fp32 outputs; the gated bf16
// output adds its own storage rounding.
constexpr ReductionCriterion kOutCriterion{2.8e-3, 1.0e-3, 2.7e-3};
constexpr ReductionCriterion kGatedCriterion{4.0e-3, 1.5e-3, 4.0e-3};
constexpr double kLseAbs = 1.0e-3;

int g_failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::cerr << "FAIL " << what << '\n';
    }
}

// ---- host numerics ------------------------------------------------------------------------------------

float bf(float x) { return bf16_to_f32(f32_to_bf16(x)); }

float e4m3_decode(std::uint8_t c) {
    const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    if (e == 15 && m == 7) { return std::numeric_limits<float>::quiet_NaN(); }
    const float v = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(1.0F + m / 8.0F, e - 7);
    return s ? -v : v;
}

const std::vector<float>& e4m3_positive() {
    static const std::vector<float> table = [] {
        std::vector<float> t(127);
        for (int c = 0; c < 127; ++c) { t[c] = e4m3_decode(static_cast<std::uint8_t>(c)); }
        return t;
    }();
    return table;
}

std::uint8_t e4m3_encode(float x) { // RNE, satfinite (cvt.rn.satfinite.e4m3x2.f32)
    if (std::isnan(x)) { return 0x7F; }
    const std::uint8_t sign = std::signbit(x) ? 0x80 : 0;
    const float a           = std::fabs(x);
    const auto& t           = e4m3_positive();
    if (a >= t[126]) { return sign | 126; }
    const int hi = static_cast<int>(std::lower_bound(t.begin(), t.end(), a) - t.begin());
    if (t[hi] == a) { return sign | static_cast<std::uint8_t>(hi); }
    const int lo   = hi - 1;
    const float dl = a - t[lo], dh = t[hi] - a;
    const int code = dl < dh ? lo : dh < dl ? hi : (lo % 2 == 0 ? lo : hi);
    return sign | static_cast<std::uint8_t>(code);
}

float host_rsqrt(float x) { return static_cast<float>(1.0 / std::sqrt(static_cast<double>(x))); }

float clip_factor(float ss, float threshold) {
    const float norm = ss > 0.0F ? ss * host_rsqrt(ss) : 0.0F;
    return std::min(threshold / (norm + G::kClipEps), 1.0F);
}

/// The kernels' sum of squares: `lanes` lanes of `per_lane` consecutive values, each lane left to right,
/// then the xor butterfly (s_l + s_{l ^ o} for o = lanes/2 .. 1).
float lane_sum_squares(const float* x, int lanes, int per_lane) {
    std::vector<float> s(lanes);
    for (int l = 0; l < lanes; ++l) {
        float a = x[l * per_lane] * x[l * per_lane];
        for (int i = 1; i < per_lane; ++i) { a = a + x[l * per_lane + i] * x[l * per_lane + i]; }
        s[l] = a;
    }
    for (int o = lanes / 2; o > 0; o >>= 1) {
        std::vector<float> n(lanes);
        for (int l = 0; l < lanes; ++l) { n[l] = s[l] + s[l ^ o]; }
        s = n;
    }
    return s[0];
}

float host_tau(std::int32_t pos) {
    const float ratio = std::max(static_cast<float>(pos + 1) * (1.0F / 8192.0F), 1.0F);
    return 1.0F + 0.1F * std::log(ratio); // std::log(float) = logf
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::uint16_t ulp_distance_bf16(std::uint16_t a, std::uint16_t b) {
    const auto key = [](std::uint16_t v) { return (v & 0x8000) ? 0x8000 - (v & 0x7fff) : 0x8000 + v; };
    const int d = static_cast<int>(key(a)) - static_cast<int>(key(b));
    return static_cast<std::uint16_t>(std::abs(d));
}
int ulp_distance_e4m3(std::uint8_t a, std::uint8_t b) {
    const auto key = [](std::uint8_t v) { return (v & 0x80) ? 0x80 - (v & 0x7f) : 0x80 + v; };
    return std::abs(static_cast<int>(key(a)) - static_cast<int>(key(b)));
}

struct Rng {
    std::mt19937 g;
    explicit Rng(std::uint32_t seed) : g(seed) {}
    float uni(float a, float b) { return std::uniform_real_distribution<float>(a, b)(g); }
    int pick(int a, int b) { return std::uniform_int_distribution<int>(a, b)(g); }
};

std::vector<std::uint16_t> to_bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> b(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { b[i] = f32_to_bf16(v[i]); }
    return b;
}

// ---- tensors ------------------------------------------------------------------------------------------

Tensor strided_bf16(void* base, std::int64_t elem_offset, std::initializer_list<std::int32_t> inner,
                    std::int32_t columns, std::int64_t column_stride_elems) {
    Tensor t;
    t.data  = static_cast<std::uint16_t*>(base) + elem_offset;
    t.dtype = DType::BF16;
    int d = 0;
    std::int64_t stride = 2;
    for (const std::int32_t e : inner) {
        t.ne[d] = e;
        t.nb[d] = stride;
        stride *= e;
        ++d;
    }
    t.ne[d] = columns;
    t.nb[d] = column_stride_elems * 2;
    for (++d; d < 4; ++d) {
        t.ne[d] = 1;
        t.nb[d] = t.nb[d - 1] * t.ne[d - 1];
    }
    return t;
}

// One visit's KV plane: `rows` sequences, each on its own shuffled physical pages.
struct Plane {
    int rows = 0, logical = 0, physical = 0;
    std::vector<std::int32_t> table; // [rows][logical] = the engine's [logical, rows]
    std::vector<std::uint8_t> hk, hv;
    DeviceBuffer dk, dv, dbt;

    Plane(int rows_in, int max_tokens, std::uint8_t fill, std::uint32_t seed)
        : rows(rows_in), logical((max_tokens + 63) / 64), physical(rows_in * ((max_tokens + 63) / 64) + 1) {
        std::vector<std::int32_t> ids(physical);
        std::iota(ids.begin(), ids.end(), 0);
        std::mt19937 g(seed);
        std::shuffle(ids.begin(), ids.end(), g);
        table.resize(static_cast<std::size_t>(rows) * logical);
        for (std::size_t i = 0; i < table.size(); ++i) { table[i] = ids[i]; }
        const std::size_t bytes = static_cast<std::size_t>(physical) * 8 * 64 * 128;
        hk.assign(bytes, fill);
        hv.assign(bytes, fill);
    }
    std::size_t offset(int row, int pos, int head) const {
        const std::int32_t page = table[static_cast<std::size_t>(row) * logical + (pos >> 6)];
        return ((static_cast<std::size_t>(page) * 8 + head) * 64 + (pos & 63)) * 128;
    }
    void upload() {
        dk  = to_device(hk);
        dv  = to_device(hv);
        dbt = to_device(table);
    }
    void download() {
        hk = from_device<std::uint8_t>(dk, hk.size());
        hv = from_device<std::uint8_t>(dv, hv.size());
    }
    PagedKVBatchLayerView view() const {
        PagedKVBatchLayerView v;
        v.k_pages      = Tensor(dk.p, DType::FP8_E4M3FN, {128, 64, 8, physical});
        v.v_pages      = Tensor(dv.p, DType::FP8_E4M3FN, {128, 64, 8, physical});
        v.block_tables = Tensor(dbt.p, DType::I32, {logical, rows});
        v.head_dim     = 128;
        v.num_kv_heads = 8;
        v.dtype        = DType::FP8_E4M3FN;
        return v;
    }
};

// Per-layer weights (bf16 values), host and device.
struct Layer {
    InklingLayerKind kind;
    std::vector<float> q_scale, r_scale, k_conv, v_conv, rel_proj;
    DeviceBuffer dq, dr, dk, dv, dproj, dtable;

    Layer(InklingLayerKind kind_in, std::uint32_t seed) : kind(kind_in) {
        Rng rng(seed);
        q_scale.resize(24 * 128);
        r_scale.resize(24 * 16);
        for (int h = 0; h < 24; ++h) {
            const float sq = rng.uni(0.4F, 3.0F), sr = rng.uni(0.4F, 3.0F);
            for (int d = 0; d < 128; ++d) { q_scale[h * 128 + d] = bf(sq * rng.uni(0.8F, 1.2F)); }
            for (int d = 0; d < 16; ++d) { r_scale[h * 16 + d] = bf(sr * rng.uni(0.8F, 1.2F)); }
        }
        k_conv.resize(4096);
        v_conv.resize(4096);
        for (auto& w : k_conv) { w = bf(rng.uni(-0.6F, 0.6F)); }
        for (auto& w : v_conv) { w = bf(rng.uni(-0.6F, 0.6F)); }
        rel_proj.resize(16 * 1024);
        for (auto& w : rel_proj) { w = bf(rng.uni(-1.0F, 1.0F)); }
        dq     = to_device_bf16(q_scale);
        dr     = to_device_bf16(r_scale);
        dk     = to_device_bf16(k_conv);
        dv     = to_device_bf16(v_conv);
        dproj  = to_device_bf16(rel_proj);
        dtable = DeviceBuffer(static_cast<std::size_t>(16) * G::kRelTableRows * 2);
        Tensor proj(dproj.p, DType::BF16, {1024, 16});
        Tensor table(dtable.p, DType::BF16, {16, G::kRelTableRows});
        ops::inkling_rel_proj_table(proj, table, nullptr);
        cuda_synchronize();
    }
    bool global() const { return kind == InklingLayerKind::Global; }
    ops::InklingAttentionWeights weights() const {
        ops::InklingAttentionWeights w;
        w.q_scale   = Tensor(dq.p, DType::BF16, {128, 24});
        w.r_scale   = Tensor(dr.p, DType::BF16, {16, 24});
        w.k_conv    = global() ? Tensor(dk.p, DType::BF16, {1024, 4}) : Tensor(dk.p, DType::BF16, {4, 1024});
        w.v_conv    = global() ? Tensor(dv.p, DType::BF16, {1024, 4}) : Tensor(dv.p, DType::BF16, {4, 1024});
        w.rel_table = Tensor(dtable.p, DType::BF16, {16, G::kRelTableRows});
        return w;
    }
    // bias(r, d): the reference's bf16(r . rel_proj[:, d]) (fp32 accumulate; here exact, then rounded).
    float bias(const float* r, int d) const {
        double s = 0.0;
        for (int j = 0; j < 16; ++j) { s += static_cast<double>(r[j]) * rel_proj[j * 1024 + d]; }
        return bf(static_cast<float>(s));
    }
};

// ---- the attention oracle -----------------------------------------------------------------------------

struct Rows {
    int W = 1, B = 1;
    std::vector<int> pos;   // [B][W] (column fastest), consecutive per row
    std::vector<int> valid; // [B]
    bool masked = false;
    DeviceBuffer dpos, dvalid, dtables;
    void upload() {
        dpos = to_device_i32(pos);
        if (masked) { dvalid = to_device_i32(valid); }
        std::vector<int> rows(B);
        std::iota(rows.begin(), rows.end(), 0);
        dtables = to_device_i32(rows);
    }
    ops::InklingAttentionRows view(int max_keys) const {
        ops::InklingAttentionRows r;
        r.positions     = Tensor(dpos.p, DType::I32, {W, B});
        r.valid_columns = masked ? Tensor(dvalid.p, DType::I32, {B}) : Tensor{};
        r.kv_table_rows = Tensor(dtables.p, DType::I32, {B});
        r.envelope.max_visible_keys = static_cast<std::uint32_t>(max_keys);
        return r;
    }
    int max_keys() const {
        int m = 0;
        for (int b = 0; b < B; ++b) {
            if (valid[b] > 0) { m = std::max(m, pos[b * W + valid[b] - 1] + 1); }
        }
        return m;
    }
};

Rows make_rows(int W, int B, const std::vector<int>& first, const std::vector<int>& valid) {
    Rows r;
    r.W = W;
    r.B = B;
    r.pos.resize(static_cast<std::size_t>(W) * B);
    r.valid  = valid.empty() ? std::vector<int>(B, W) : valid;
    r.masked = !valid.empty();
    for (int b = 0; b < B; ++b) {
        for (int j = 0; j < W; ++j) {
            // Past the valid count: repeat the last valid position (any in-range value is legal).
            const int jj = std::min(j, std::max(r.valid[b] - 1, 0));
            r.pos[b * W + j] = first[b] + jj;
        }
    }
    r.upload();
    return r;
}

/// FP64 attention over the plane's codes for every valid column: out [N][24][128], lse [N][24].
void attention_oracle(const Layer& layer, const Rows& rows, const Plane& plane, const std::vector<float>& q,
                      const std::vector<float>& r, std::vector<double>& out, std::vector<double>& lse) {
    const int N = rows.W * rows.B;
    out.assign(static_cast<std::size_t>(N) * 24 * 128, 0.0);
    lse.assign(static_cast<std::size_t>(N) * 24, 0.0);
    const double scale = 1.0 / std::sqrt(128.0);
    std::vector<float> dec(256);
    for (int c = 0; c < 256; ++c) { dec[c] = e4m3_decode(static_cast<std::uint8_t>(c)); }
    for (int b = 0; b < rows.B; ++b) {
        for (int j = 0; j < rows.valid[b]; ++j) {
            const int n = j + rows.W * b, p = rows.pos[b * rows.W + j];
            const int lo = layer.global() ? 0 : std::max(0, p - (G::kWindow - 1));
            for (int h = 0; h < 24; ++h) {
                const float* qh = &q[(static_cast<std::size_t>(n) * 24 + h) * 128];
                const float* rh = &r[(static_cast<std::size_t>(n) * 24 + h) * 16];
                std::vector<double> s(p - lo + 1);
                double m = -std::numeric_limits<double>::infinity();
                for (int x = lo; x <= p; ++x) {
                    const std::uint8_t* kc = &plane.hk[plane.offset(b, x, h / 3)];
                    double dot = 0.0;
                    for (int d = 0; d < 128; ++d) { dot += static_cast<double>(qh[d]) * dec[kc[d]]; }
                    const int dist = p - x;
                    s[x - lo]      = dot * scale + (dist < G::kRelExtent ? layer.bias(rh, dist) : 0.0);
                    m              = std::max(m, s[x - lo]);
                }
                double l = 0.0;
                double* o = &out[(static_cast<std::size_t>(n) * 24 + h) * 128];
                for (int x = lo; x <= p; ++x) {
                    const double w        = std::exp(s[x - lo] - m);
                    const std::uint8_t* vc = &plane.hv[plane.offset(b, x, h / 3)];
                    l += w;
                    for (int d = 0; d < 128; ++d) { o[d] += w * dec[vc[d]]; }
                }
                for (int d = 0; d < 128; ++d) { o[d] /= l; }
                lse[static_cast<std::size_t>(n) * 24 + h] = m + std::log(l);
            }
        }
    }
}

std::vector<double> gated_oracle(const Rows& rows, const std::vector<double>& out, const std::vector<double>& lse,
                                 const std::vector<float>& gate, const std::vector<float>& sink) {
    const int N = rows.W * rows.B;
    std::vector<double> y(static_cast<std::size_t>(N) * 3072, 0.0);
    for (int b = 0; b < rows.B; ++b) {
        for (int j = 0; j < rows.valid[b]; ++j) {
            const int n = j + rows.W * b;
            for (int c = 0; c < 3072; ++c) {
                const int h = c / 128;
                y[static_cast<std::size_t>(n) * 3072 + c] =
                    out[static_cast<std::size_t>(n) * 3072 + c] * sigmoid(gate[static_cast<std::size_t>(n) * 3072 + c]) *
                    sigmoid(lse[static_cast<std::size_t>(n) * 24 + h] - sink[static_cast<std::size_t>(n) * 24 + h]);
            }
        }
    }
    return y;
}

// Random clipped q / r rows (as the prologue produces) and K / V codes for positions [0, L) of every row.
void fill_qr(Rng& rng, const Rows& rows, std::vector<float>& q, std::vector<float>& r) {
    const int N = rows.W * rows.B;
    q.assign(static_cast<std::size_t>(N) * 24 * 128, 0.0F);
    r.assign(static_cast<std::size_t>(N) * 24 * 16, 0.0F);
    for (int b = 0; b < rows.B; ++b) {
        for (int j = 0; j < rows.valid[b]; ++j) {
            const int n = j + rows.W * b;
            for (int h = 0; h < 24; ++h) {
                const float a = rng.uni(0.5F, 3.5F);
                float* qh     = &q[(static_cast<std::size_t>(n) * 24 + h) * 128];
                double ss     = 0.0;
                for (int d = 0; d < 128; ++d) { qh[d] = rng.uni(-a, a); ss += qh[d] * qh[d]; }
                const double f = std::min(1.0, 22.627 / std::sqrt(ss));
                for (int d = 0; d < 128; ++d) { qh[d] = bf(static_cast<float>(qh[d] * f)); }
                float* rh = &r[(static_cast<std::size_t>(n) * 24 + h) * 16];
                ss        = 0.0;
                for (int d = 0; d < 16; ++d) { rh[d] = rng.uni(-a, a); ss += rh[d] * rh[d]; }
                const double fr = std::min(1.0, 8.0 / std::sqrt(ss));
                for (int d = 0; d < 16; ++d) { rh[d] = bf(static_cast<float>(rh[d] * fr)); }
            }
        }
    }
}

void fill_kv(Rng& rng, Plane& plane, int row, int lo_finite, int keys) {
    std::vector<float> k(128);
    for (int x = lo_finite; x < keys; ++x) {
        for (int h = 0; h < 8; ++h) {
            double ss = 0.0;
            for (int d = 0; d < 128; ++d) { k[d] = rng.uni(-3.0F, 3.0F); ss += k[d] * k[d]; }
            const double f = std::min(1.0, 22.627 / std::sqrt(ss));
            std::uint8_t* kc = &plane.hk[plane.offset(row, x, h)];
            std::uint8_t* vc = &plane.hv[plane.offset(row, x, h)];
            for (int d = 0; d < 128; ++d) {
                kc[d] = e4m3_encode(static_cast<float>(k[d] * f));
                vc[d] = e4m3_encode(rng.uni(-1.7F, 1.7F));
            }
        }
    }
}

// ---- attention cases ----------------------------------------------------------------------------------

struct AttnCase {
    std::string name;
    InklingLayerKind kind;
    int W, B;
    std::vector<int> first;
    std::vector<int> valid;
    int force_splits = 0;
};

void run_attention_case(const AttnCase& c, std::uint32_t seed) {
    Rng rng(seed);
    const Layer layer(c.kind, seed + 1);
    const Rows rows  = make_rows(c.W, c.B, c.first, c.valid);
    const int N      = c.W * c.B;
    const int maxk   = rows.max_keys();
    // Codes NaN (0x7F) everywhere they may not be read: whole pages before a local row's window and every
    // position past a row's last key. The kernel must neither read the former nor let the latter reach P V.
    Plane plane(c.B, maxk + 64, 0x7F, seed + 2);
    for (int b = 0; b < c.B; ++b) {
        if (rows.valid[b] == 0) { continue; }
        const int p_first = rows.pos[b * c.W];
        const int keys    = rows.pos[b * c.W + rows.valid[b] - 1] + 1;
        const int lo      = layer.global() ? 0 : std::max(0, p_first - (G::kWindow - 1)) / 64 * 64;
        fill_kv(rng, plane, b, lo, keys);
    }
    plane.upload();
    std::vector<float> q, r;
    fill_qr(rng, rows, q, r);
    // gate / sink inside a [6552, N] query-side buffer, as the projection leaves them.
    std::vector<float> qside(static_cast<std::size_t>(N) * 6552);
    for (auto& x : qside) { x = bf(rng.uni(-3.0F, 3.0F)); }
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < 24; ++h) { qside[static_cast<std::size_t>(n) * 6552 + G::kRowsSink + h] = bf(rng.uni(-2.0F, 6.0F)); }
    }
    std::vector<float> gate(static_cast<std::size_t>(N) * 3072), sink(static_cast<std::size_t>(N) * 24);
    for (int n = 0; n < N; ++n) {
        for (int i = 0; i < 3072; ++i) { gate[static_cast<std::size_t>(n) * 3072 + i] = qside[static_cast<std::size_t>(n) * 6552 + G::kRowsGate + i]; }
        for (int i = 0; i < 24; ++i) { sink[static_cast<std::size_t>(n) * 24 + i] = qside[static_cast<std::size_t>(n) * 6552 + G::kRowsSink + i]; }
    }
    DeviceBuffer dq = to_device_bf16(q), dr = to_device_bf16(r), dqs = to_device_bf16(qside);
    ops::InklingAttentionInputs in;
    in.q         = Tensor(dq.p, DType::BF16, {128, 24, N});
    in.r         = Tensor(dr.p, DType::BF16, {16, 24, N});
    in.gate      = strided_bf16(dqs.p, G::kRowsGate, {3072}, N, 6552);
    in.sink      = strided_bf16(dqs.p, G::kRowsSink, {24}, N, 6552);
    in.rel_table = Tensor(layer.dtable.p, DType::BF16, {16, G::kRelTableRows});

    if (c.force_splits > 0) {
        setenv("SUROGATE_INKLING_ATTN_SPLITS", std::to_string(c.force_splits).c_str(), 1);
    } else {
        unsetenv("SUROGATE_INKLING_ATTN_SPLITS");
    }
    const auto view = rows.view(maxk);
    WorkspaceArena work(std::max<std::size_t>(ops::inkling_rel_attention_workspace_bytes(c.kind, c.W, c.B, view.envelope), 1) + 4096);
    DeviceBuffer dout(static_cast<std::size_t>(N) * 3072 * 2), dof(static_cast<std::size_t>(N) * 3072 * 4),
        dl(static_cast<std::size_t>(N) * 24 * 4);
    cudaMemset(dout.p, 0x7f, dout.bytes);
    Tensor out(dout.p, DType::BF16, {3072, N});
    Tensor of(dof.p, DType::FP32, {128, 24, N}), lse(dl.p, DType::FP32, {24, N});
    ops::inkling_rel_attention(in, c.kind, view, plane.view(), work, out, nullptr);
    ops::inkling_rel_attention_out_lse(in, c.kind, view, plane.view(), work, of, lse, nullptr);
    cuda_synchronize();
    unsetenv("SUROGATE_INKLING_ATTN_SPLITS");

    std::vector<double> ref_o, ref_l;
    attention_oracle(layer, rows, plane, q, r, ref_o, ref_l);
    const std::vector<double> ref_y = gated_oracle(rows, ref_o, ref_l, gate, sink);
    std::vector<double> got_o = from_device_f32(dof, static_cast<std::size_t>(N) * 3072);
    std::vector<double> got_l = from_device_f32(dl, static_cast<std::size_t>(N) * 24);
    std::vector<double> got_y = from_device_bf16(dout, static_cast<std::size_t>(N) * 3072);
    check(verify_reduction(c.name + " out", got_o, ref_o, kOutCriterion) == 0, c.name + " out");
    check(verify_reduction(c.name + " gated", got_y, ref_y, kGatedCriterion) == 0, c.name + " gated");
    double lse_err = 0.0;
    bool invalid_zero = true;
    for (int b = 0; b < c.B; ++b) {
        for (int j = 0; j < c.W; ++j) {
            const int n = j + c.W * b;
            for (int h = 0; h < 24; ++h) {
                const double g = got_l[static_cast<std::size_t>(n) * 24 + h];
                if (j < rows.valid[b]) {
                    lse_err = std::max(lse_err, std::fabs(g - ref_l[static_cast<std::size_t>(n) * 24 + h]));
                } else {
                    invalid_zero = invalid_zero && g == 0.0;
                }
            }
            if (j >= rows.valid[b]) {
                for (int i = 0; i < 3072; ++i) {
                    invalid_zero = invalid_zero && got_y[static_cast<std::size_t>(n) * 3072 + i] == 0.0 &&
                                   got_o[static_cast<std::size_t>(n) * 3072 + i] == 0.0;
                }
            }
        }
    }
    std::printf("%-44s lse max|err| %.2e%s\n", c.name.c_str(), lse_err, invalid_zero ? "" : "  INVALID COLUMNS NOT ZERO");
    check(lse_err <= kLseAbs, c.name + " lse");
    check(invalid_zero, c.name + " invalid columns zero");
}

// ---- the prologue oracle and cases --------------------------------------------------------------------

struct ProCase {
    std::string name;
    InklingLayerKind kind;
    int W, B;
    std::vector<int> first;
    std::vector<int> valid;
    ops::InklingConvMode mode = ops::InklingConvMode::InPlace;
    int capture_column        = -1;
    bool fused                = false; // run inkling_attention (prologue + attention) instead
};

void run_prologue_case(const ProCase& c, std::uint32_t seed) {
    Rng rng(seed);
    const Layer layer(c.kind, seed + 1);
    const Rows rows = make_rows(c.W, c.B, c.first, c.valid);
    const int N     = c.W * c.B;
    const bool glob = layer.global();
    const int S     = c.B + 3; // slots; row b uses slot (b * 2 + 1) % S, capture slot S - 1
    std::vector<int> slots(c.B);
    for (int b = 0; b < c.B; ++b) { slots[b] = (b * 2 + 1) % (S - 1); }
    const int capture_slot = S - 1;

    // Projection buffer [8600, N] and dynamic taps [32, N].
    std::vector<float> proj(static_cast<std::size_t>(N) * 8600);
    for (auto& x : proj) { x = bf(rng.uni(-2.0F, 2.0F)); }
    std::vector<float> dyn_k(static_cast<std::size_t>(N) * 32), dyn_v(static_cast<std::size_t>(N) * 32);
    for (auto& x : dyn_k) { x = bf(rng.uni(-0.3F, 0.3F)); }
    for (auto& x : dyn_v) { x = bf(rng.uni(-0.3F, 0.3F)); }
    std::vector<float> state(static_cast<std::size_t>(kC) * 3 * S);
    for (auto& x : state) { x = bf(rng.uni(-2.0F, 2.0F)); }
    const int Wmax = c.W + 2;
    std::vector<float> records(static_cast<std::size_t>(kC) * Wmax * c.B, 0.0F);

    const int maxk = rows.max_keys();
    // Canary NaN (0x7F): satfinite e4m3 never produces it, so an unwritten slot is unmistakable.
    Plane plane(c.B, maxk + 64, 0x7F, seed + 2);
    // History keys for the fused route's attention (positions before each row's first column).
    if (c.fused) {
        for (int b = 0; b < c.B; ++b) {
            const int p0 = rows.pos[b * c.W];
            const int lo = glob ? 0 : std::max(0, p0 - (G::kWindow - 1)) / 64 * 64;
            fill_kv(rng, plane, b, lo, p0);
        }
    }
    plane.upload();
    const std::vector<std::uint8_t> plane_k0 = plane.hk, plane_v0 = plane.hv;

    DeviceBuffer dproj = to_device_bf16(proj), ddk = to_device_bf16(dyn_k), ddv = to_device_bf16(dyn_v);
    DeviceBuffer dstate = to_device_bf16(state), drec = to_device_bf16(records), dslots = to_device_i32(slots);
    ops::InklingProjections in;
    in.q    = strided_bf16(dproj.p, G::kRowsQ, {128, 24}, N, 8600);
    in.r    = strided_bf16(dproj.p, G::kRowsR, {16, 24}, N, 8600);
    in.gate = strided_bf16(dproj.p, G::kRowsGate, {3072}, N, 8600);
    in.sink = strided_bf16(dproj.p, G::kRowsSink, {24}, N, 8600);
    in.k    = strided_bf16(dproj.p, G::kRowsK, {1024}, N, 8600);
    in.v    = strided_bf16(dproj.p, G::kRowsV, {1024}, N, 8600);
    if (glob) {
        in.k_dyn = Tensor(ddk.p, DType::BF16, {4, 8, N});
        in.v_dyn = Tensor(ddv.p, DType::BF16, {4, 8, N});
    }
    ops::InklingConvStateArgs conv;
    conv.states = Tensor(dstate.p, DType::BF16, {kC, 3, S});
    conv.slots  = Tensor(dslots.p, DType::I32, {c.B});
    conv.mode   = c.mode;
    if (c.mode == ops::InklingConvMode::Record) { conv.records = Tensor(drec.p, DType::BF16, {kC, Wmax, c.B}); }
    conv.capture_column = c.capture_column;
    conv.capture_slot   = c.capture_column >= 0 ? capture_slot : -1;
    const auto view     = rows.view(maxk);

    DeviceBuffer dqo(static_cast<std::size_t>(N) * 3072 * 2), dro(static_cast<std::size_t>(N) * 384 * 2),
        dout(static_cast<std::size_t>(N) * 3072 * 2);
    ops::InklingPrologueOutputs qr;
    qr.q = Tensor(dqo.p, DType::BF16, {128, 24, N});
    qr.r = Tensor(dro.p, DType::BF16, {16, 24, N});
    if (c.fused) {
        WorkspaceArena work(ops::inkling_attention_workspace_capacity_bytes(c.kind, c.B, c.W, c.W, view.envelope) + 4096);
        Tensor out(dout.p, DType::BF16, {3072, N});
        ops::inkling_attention(in, layer.weights(), c.kind, view, conv, plane.view(), work, out, nullptr);
    } else {
        ops::inkling_attention_prologue(in, layer.weights(), c.kind, view, conv, plane.view(), qr, nullptr);
    }
    cuda_synchronize();
    plane.download();

    // ---- oracle ----
    std::vector<float> q_ref(static_cast<std::size_t>(N) * 3072, 0.0F), r_ref(static_cast<std::size_t>(N) * 384, 0.0F);
    std::vector<float> state_ref = state, rec_ref = records;
    std::vector<std::uint8_t> k_ref = plane_k0, v_ref = plane_v0;
    for (int b = 0; b < c.B; ++b) {
        const int valid = rows.valid[b];
        for (int j = 0; j < valid; ++j) {
            const int n = j + c.W * b;
            const int p = rows.pos[b * c.W + j];
            const float tau = glob ? host_tau(p) : 1.0F;
            for (int h = 0; h < 24; ++h) {
                float x[128];
                for (int d = 0; d < 128; ++d) {
                    x[d] = proj[static_cast<std::size_t>(n) * 8600 + G::kRowsQ + h * 128 + d] * layer.q_scale[h * 128 + d];
                }
                const float f = clip_factor(lane_sum_squares(x, 32, 4), G::kQkClip);
                for (int d = 0; d < 128; ++d) {
                    float y = x[d] * f;
                    if (glob) { y = bf(y) * tau; }
                    q_ref[static_cast<std::size_t>(n) * 3072 + h * 128 + d] = bf(y);
                }
                float xr[16];
                for (int d = 0; d < 16; ++d) {
                    xr[d] = proj[static_cast<std::size_t>(n) * 8600 + G::kRowsR + h * 16 + d] * layer.r_scale[h * 16 + d];
                }
                const float fr = clip_factor(lane_sum_squares(xr, 16, 1), G::kRelClip);
                for (int d = 0; d < 16; ++d) {
                    float y = xr[d] * fr;
                    if (glob) { y = bf(y) * tau; }
                    r_ref[static_cast<std::size_t>(n) * 384 + h * 16 + d] = bf(y);
                }
            }
        }
        // K / V conv over [state | columns] with the zero rule by position.
        if (valid == 0) { continue; }
        const int p0 = rows.pos[b * c.W];
        for (int kv = 0; kv < 2; ++kv) {
            const int row_base = kv ? G::kRowsV : G::kRowsK;
            const std::vector<float>& wconv = kv ? layer.v_conv : layer.k_conv;
            const std::vector<float>& dyn   = kv ? dyn_v : dyn_k;
            const int sc0                   = kv * 1024;
            const auto input = [&](int ch, int col) -> float { // column col of this row (col >= -3)
                if (col >= 0) { return proj[static_cast<std::size_t>(col + c.W * b) * 8600 + row_base + ch]; }
                if (p0 + col < 0) { return 0.0F; }
                return state[(static_cast<std::size_t>(slots[b]) * 3 + (3 + col)) * kC + sc0 + ch];
            };
            for (int head = 0; head < 8; ++head) {
                for (int j = 0; j < valid; ++j) {
                    const int n = j + c.W * b;
                    float y[128];
                    for (int i = 0; i < 128; ++i) {
                        const int ch = head * 128 + i;
                        const float cur = input(ch, j);
                        float acc;
                        if (!glob) {
                            acc = 0.0F;
                            for (int k = 0; k < 4; ++k) { acc = std::fma(input(ch, j - 3 + k), wconv[ch * 4 + k], acc); }
                            y[i] = bf((acc + cur) * 0.44721359549995793F);
                        } else {
                            const float* d4 = &dyn[static_cast<std::size_t>(n) * 32 + head * 4];
                            acc = cur * (wconv[0 * 1024 + ch] + d4[0]);
                            for (int lag = 1; lag < 4; ++lag) {
                                acc = std::fma(input(ch, j - lag), wconv[lag * 1024 + ch] + d4[lag], acc);
                            }
                            y[i] = bf((acc + cur) * 0.33333333333333333F);
                        }
                    }
                    const float ss = lane_sum_squares(y, 32, 4);
                    const int p    = rows.pos[b * c.W + j];
                    std::vector<std::uint8_t>& dst = kv ? v_ref : k_ref;
                    const std::size_t off = plane.offset(b, p, head);
                    if (kv == 0) {
                        const float f = clip_factor(ss, G::kQkClip);
                        for (int i = 0; i < 128; ++i) { dst[off + i] = e4m3_encode(y[i] * f); }
                    } else {
                        const float rs = host_rsqrt(ss / 128.0F + G::kNormEps);
                        for (int i = 0; i < 128; ++i) { dst[off + i] = e4m3_encode(y[i] * rs); }
                    }
                }
            }
            // State effects.
            for (int ch = 0; ch < 1024; ++ch) {
                if (c.mode == ops::InklingConvMode::Record) {
                    for (int j = 0; j < valid; ++j) {
                        rec_ref[(static_cast<std::size_t>(b) * Wmax + j) * kC + sc0 + ch] = input(ch, j);
                    }
                } else {
                    float win[3];
                    for (int t = 0; t < 3; ++t) { win[t] = input(ch, valid - 3 + t); }
                    if (c.capture_column >= 0) {
                        for (int t = 0; t < 3; ++t) {
                            state_ref[(static_cast<std::size_t>(capture_slot) * 3 + t) * kC + sc0 + ch] =
                                input(ch, c.capture_column - 3 + t);
                        }
                    }
                    for (int t = 0; t < 3; ++t) {
                        state_ref[(static_cast<std::size_t>(slots[b]) * 3 + t) * kC + sc0 + ch] = win[t];
                    }
                }
            }
        }
    }

    // ---- compare ----
    const auto bits_check = [&](const char* what, const std::vector<std::uint16_t>& got,
                                const std::vector<float>& ref) {
        std::size_t off1 = 0, worse = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            const int d = ulp_distance_bf16(got[i], f32_to_bf16(ref[i]));
            off1 += d == 1;
            worse += d > 1;
        }
        std::printf("%-44s %-8s 1-ulp %zu / %zu, worse %zu\n", c.name.c_str(), what, off1, ref.size(), worse);
        check(worse == 0 && off1 * 200 <= ref.size(), c.name + " " + what);
    };
    if (!c.fused) {
        bits_check("q_out", from_device<std::uint16_t>(dqo, q_ref.size()), q_ref);
        bits_check("r_out", from_device<std::uint16_t>(dro, r_ref.size()), r_ref);
    }
    std::size_t kv_off1 = 0, kv_worse = 0, kv_count = 0;
    for (std::size_t i = 0; i < k_ref.size(); ++i) {
        for (int which = 0; which < 2; ++which) {
            const std::uint8_t g = which ? plane.hv[i] : plane.hk[i];
            const std::uint8_t e = which ? v_ref[i] : k_ref[i];
            const std::uint8_t z = which ? plane_v0[i] : plane_k0[i];
            if (e == z && g == z) { continue; }
            ++kv_count;
            const int d = (e == 0x7F || g == 0x7F) ? 99 : ulp_distance_e4m3(g, e);
            kv_off1 += d == 1;
            kv_worse += d > 1;
        }
    }
    std::printf("%-44s %-8s 1-ulp %zu / %zu, worse %zu\n", c.name.c_str(), "kv", kv_off1, kv_count, kv_worse);
    check(kv_worse == 0 && kv_off1 * 200 <= kv_count, c.name + " kv codes");
    const auto exact = [&](const char* what, const DeviceBuffer& d, const std::vector<float>& ref) {
        const std::vector<std::uint16_t> got = from_device<std::uint16_t>(d, ref.size());
        std::size_t bad = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) { bad += got[i] != f32_to_bf16(ref[i]); }
        std::printf("%-44s %-8s mismatches %zu / %zu\n", c.name.c_str(), what, bad, ref.size());
        check(bad == 0, c.name + " " + what);
    };
    exact("state", dstate, state_ref);
    if (c.mode == ops::InklingConvMode::Record) { exact("records", drec, rec_ref); }

    if (c.fused) {
        // Attention over the oracle's own keys and q / r: tolerance-equal to the fused output.
        plane.hk = k_ref;
        plane.hv = v_ref;
        std::vector<double> ref_o, ref_l;
        attention_oracle(layer, rows, plane, q_ref, r_ref, ref_o, ref_l);
        std::vector<float> gate(static_cast<std::size_t>(N) * 3072), sink(static_cast<std::size_t>(N) * 24);
        for (int n = 0; n < N; ++n) {
            for (int i = 0; i < 3072; ++i) { gate[static_cast<std::size_t>(n) * 3072 + i] = proj[static_cast<std::size_t>(n) * 8600 + G::kRowsGate + i]; }
            for (int i = 0; i < 24; ++i) { sink[static_cast<std::size_t>(n) * 24 + i] = proj[static_cast<std::size_t>(n) * 8600 + G::kRowsSink + i]; }
        }
        const std::vector<double> ref_y = gated_oracle(rows, ref_o, ref_l, gate, sink);
        const std::vector<double> got_y = from_device_bf16(dout, static_cast<std::size_t>(N) * 3072);
        check(verify_reduction(c.name + " fused gated", got_y, ref_y, kGatedCriterion) == 0, c.name + " fused");
    }
}

// ---- small ops ----------------------------------------------------------------------------------------

void test_tau() {
    const int n = 1 << 20;
    std::vector<int> pos(n);
    std::iota(pos.begin(), pos.end(), 0);
    DeviceBuffer dp = to_device_i32(pos), dt(static_cast<std::size_t>(n) * 4);
    Tensor tp(dp.p, DType::I32, {n}), tt(dt.p, DType::FP32, {n});
    ops::inkling_log_tau(tp, tt, nullptr);
    cuda_synchronize();
    const std::vector<float> got = from_device<float>(dt, n);
    std::size_t diff = 0, worse = 0, below = 0;
    for (int i = 0; i < n; ++i) {
        const float ref = host_tau(i);
        if (got[i] != ref) {
            ++diff;
            std::int32_t a, b;
            std::memcpy(&a, &got[i], 4);
            std::memcpy(&b, &ref, 4);
            worse += std::abs(a - b) > 1;
        }
        below += i < 8191 && got[i] != 1.0F;
    }
    std::printf("%-44s differ from host logf %zu / %d (> 1 ulp %zu), tau != 1 below 8191: %zu\n", "log_tau",
                diff, n, worse, below);
    check(worse == 0 && below == 0, "log_tau");
}

void test_rel_table_and_logits() {
    const Layer layer(InklingLayerKind::Global, 77);
    const std::vector<std::uint16_t> table =
        from_device<std::uint16_t>(layer.dtable, static_cast<std::size_t>(16) * G::kRelTableRows);
    std::size_t bad = 0;
    for (int e = 0; e < G::kRelTableRows; ++e) {
        const int d = e - G::kRelTablePad;
        for (int j = 0; j < 16; ++j) {
            const float want = (d >= 0 && d < 1024) ? layer.rel_proj[j * 1024 + d] : 0.0F;
            bad += table[static_cast<std::size_t>(e) * 16 + j] != f32_to_bf16(want);
        }
    }
    std::printf("%-44s mismatches %zu\n", "rel_proj_table", bad);
    check(bad == 0, "rel_proj_table");

    Rng rng(5);
    const int N = 37;
    Rows rows = make_rows(N, 1, {100}, {});
    std::vector<float> q, r;
    fill_qr(rng, rows, q, r);
    DeviceBuffer dr = to_device_bf16(r), drel(static_cast<std::size_t>(N) * 24 * 1024 * 2);
    Tensor tr(dr.p, DType::BF16, {16, 24, N}), trel(drel.p, DType::BF16, {1024, 24, N});
    ops::inkling_rel_logits(tr, Tensor(layer.dtable.p, DType::BF16, {16, G::kRelTableRows}), trel, nullptr);
    cuda_synchronize();
    const std::vector<std::uint16_t> got = from_device<std::uint16_t>(drel, static_cast<std::size_t>(N) * 24 * 1024);
    std::size_t off1 = 0, worse = 0;
    for (int row = 0; row < N * 24; ++row) {
        for (int d = 0; d < 1024; ++d) {
            const int u = ulp_distance_bf16(got[static_cast<std::size_t>(row) * 1024 + d],
                                            f32_to_bf16(layer.bias(&r[static_cast<std::size_t>(row) * 16], d)));
            off1 += u == 1;
            worse += u > 1;
        }
    }
    std::printf("%-44s 1-ulp %zu / %d, worse %zu\n", "rel_logits (k16 MMA vs exact)", off1, N * 24 * 1024, worse);
    check(worse == 0 && off1 * 1000 <= static_cast<std::size_t>(N) * 24 * 1024, "rel_logits");
}

void test_gate_sink() {
    Rng rng(9);
    const int N = 5;
    std::vector<float> out(static_cast<std::size_t>(N) * 3072), lse(static_cast<std::size_t>(N) * 24);
    std::vector<float> qside(static_cast<std::size_t>(N) * 6552);
    for (auto& x : out) { x = rng.uni(-2.0F, 2.0F); }
    for (auto& x : lse) { x = rng.uni(-1.0F, 8.0F); }
    for (auto& x : qside) { x = bf(rng.uni(-5.0F, 5.0F)); }
    DeviceBuffer d_out = to_device_f32(out), d_lse = to_device_f32(lse), dqs = to_device_bf16(qside);
    DeviceBuffer dres(static_cast<std::size_t>(N) * 3072 * 2);
    Tensor t_out(d_out.p, DType::FP32, {128, 24, N}), t_lse(d_lse.p, DType::FP32, {24, N});
    Tensor res(dres.p, DType::BF16, {3072, N});
    ops::inkling_gate_sink(t_out, t_lse, strided_bf16(dqs.p, G::kRowsGate, {3072}, N, 6552),
                           strided_bf16(dqs.p, G::kRowsSink, {24}, N, 6552), res, nullptr);
    cuda_synchronize();
    const std::vector<std::uint16_t> got = from_device<std::uint16_t>(dres, static_cast<std::size_t>(N) * 3072);
    std::size_t worse = 0;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < 3072; ++c) {
            const double g = qside[static_cast<std::size_t>(n) * 6552 + G::kRowsGate + c];
            const double s = qside[static_cast<std::size_t>(n) * 6552 + G::kRowsSink + c / 128];
            const double y = out[static_cast<std::size_t>(n) * 3072 + c] * sigmoid(g) *
                             sigmoid(lse[static_cast<std::size_t>(n) * 24 + c / 128] - s);
            worse += ulp_distance_bf16(got[static_cast<std::size_t>(n) * 3072 + c], f32_to_bf16(static_cast<float>(y))) > 1;
        }
    }
    std::printf("%-44s > 1 ulp %zu / %d\n", "gate_sink", worse, N * 3072);
    check(worse == 0, "gate_sink");
}

// ---- the FP8 attention projections ------------------------------------------------------------------

struct Fp8Weight {
    int n = 0, k = 0, k_per = 0, rows_per = 0;
    std::vector<std::uint8_t> codes;
    std::vector<float> scales; // [n / rows_per][k / k_per]
    DeviceBuffer dcodes, dscales;
    Weight w{};

    Fp8Weight(int n_in, int k_in, int k_per_in, int rows_per_in, std::uint32_t seed)
        : n(n_in), k(k_in), k_per(k_per_in), rows_per(rows_per_in) {
        Rng rng(seed);
        codes.resize(static_cast<std::size_t>(n) * k);
        for (auto& c : codes) {
            std::uint8_t v;
            do { v = static_cast<std::uint8_t>(rng.pick(0, 255)); } while ((v & 0x7F) == 0x7F);
            c = v;
        }
        scales.resize(static_cast<std::size_t>((n + rows_per - 1) / rows_per) * (k / k_per));
        for (auto& x : scales) { x = rng.uni(1e-4F, 4e-3F); }
        dcodes  = to_device(codes);
        dscales = to_device(scales);
        w.qtype       = k_per == k ? QType::FP8_E4M3FN_ROW_F32S : QType::FP8_E4M3FN_BLK128_F32S;
        w.layout      = QuantLayout::Fp8Block128;
        w.scale_dtype = DType::FP32;
        w.qdata       = dcodes.p;
        w.scales      = dscales.p;
        w.n           = n;
        w.k           = k;
        w.scale_ne[0] = k_per;
        w.scale_ne[1] = rows_per;
        w.group_size  = 128;
        w.group       = 128;
    }
    double value(int row, int col) const {
        return static_cast<double>(e4m3_decode(codes[static_cast<std::size_t>(row) * k + col])) *
               scales[static_cast<std::size_t>(row / rows_per) * (k / k_per) + col / k_per];
    }
};

void run_linear_case(const std::string& name, int k, int k_per, int rows_per, int columns,
                     const std::vector<std::pair<int, int>>& ranges, std::uint32_t seed) {
    const int n_rows = k == 2304 ? 8600 : 2304;
    const Fp8Weight fw(n_rows, k, k_per, rows_per, seed);
    Rng rng(seed + 1);
    std::vector<std::vector<float>> xs;
    std::vector<DeviceBuffer> dxs, douts;
    std::vector<Tensor> xt, ot;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        std::vector<float> x(static_cast<std::size_t>(k) * columns);
        for (auto& v : x) { v = bf(rng.uni(-3.0F, 3.0F)); }
        xs.push_back(x);
    }
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        dxs.push_back(to_device_bf16(xs[i]));
        // Outputs as row slices of one [n_rows + 8, N] buffer each (column stride > rows).
        douts.emplace_back(static_cast<std::size_t>(n_rows + 8) * columns * 2);
    }
    std::vector<ops::InklingProjection> pj(ranges.size());
    xt.reserve(ranges.size());
    ot.reserve(ranges.size());
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        xt.emplace_back(dxs[i].p, DType::BF16, std::initializer_list<std::int32_t>{k, columns});
        ot.push_back(strided_bf16(douts[i].p, 0, {ranges[i].second}, columns, n_rows + 8));
        pj[i].x         = &xt[i];
        pj[i].weight    = &fw.w;
        pj[i].row_begin = ranges[i].first;
        pj[i].out       = &ot[i];
    }
    ops::inkling_attn_linear(pj, nullptr, nullptr);
    cuda_synchronize();
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const int rows = ranges[i].second;
        const std::vector<std::uint16_t> got =
            from_device<std::uint16_t>(douts[i], static_cast<std::size_t>(n_rows + 8) * columns);
        std::vector<double> g, ref;
        for (int c = 0; c < columns; ++c) {
            for (int r = 0; r < rows; ++r) {
                double acc = 0.0;
                for (int kk = 0; kk < k; ++kk) {
                    acc += fw.value(ranges[i].first + r, kk) * xs[i][static_cast<std::size_t>(c) * k + kk];
                }
                ref.push_back(acc);
                g.push_back(bf16_to_f32(got[static_cast<std::size_t>(c) * (n_rows + 8) + r]));
            }
        }
        check(verify_reduction(name + " segment " + std::to_string(i), g, ref,
                               ReductionCriterion{2.0e-3, 1.0e-4, 4.0e-3}) == 0,
              name + " segment " + std::to_string(i));
    }
}

void test_attn_linear() {
    const int K = 2304;
    const std::vector<std::pair<int, int>> one   = {{0, 8600}};
    const std::vector<std::pair<int, int>> three = {{0, G::kRowsK}, {G::kRowsK, 1024}, {G::kRowsV, 1024}};
    std::uint32_t seed = 4000;
    for (const int cols : {1, 3, 8, 40, 300}) {
        run_linear_case("attn_linear qkvr 1x128 N=" + std::to_string(cols), K, 128, 1, cols, one, seed++);
        run_linear_case("attn_linear qkvr channel N=" + std::to_string(cols), K, K, 1, cols, one, seed++);
        run_linear_case("attn_linear 3-input 1x128 N=" + std::to_string(cols), K, 128, 1, cols, three, seed++);
    }
    run_linear_case("attn_linear wo 1x128 N=2", 3072, 128, 1, 2, {{0, 2304}}, seed++);
    run_linear_case("attn_linear wo channel N=130", 3072, 3072, 1, 130, {{0, 2304}}, seed++);
    run_linear_case("attn_linear wo 128x128 N=70", 3072, 128, 128, 70, {{0, 2304}}, seed++);
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "no CUDA device: skipped\n";
        return 77;
    }
    const std::string only = argc > 1 ? argv[1] : "";
    const auto want = [&](const std::string& name) { return only.empty() || name.find(only) != std::string::npos; };
    try {
        if (want("small")) {
            test_tau();
            test_rel_table_and_logits();
            test_gate_sink();
        }
        if (want("attn_linear")) { test_attn_linear(); }
        using K = InklingLayerKind;
        using M = ops::InklingConvMode;
        const std::vector<ProCase> pro = {
            {"prologue local decode B3", K::Local, 1, 3, {0, 7, 70000}, {}},
            {"prologue global decode B3 (tau > 1)", K::Global, 1, 3, {0, 9000, 70000}, {}},
            {"prologue local verify W8 Record", K::Local, 8, 2, {2, 5000}, {8, 5}, M::Record},
            {"prologue global verify W3 Record", K::Global, 3, 2, {1, 20000}, {3, 2}, M::Record},
            {"prologue local prefill W300 capture", K::Local, 300, 1, {1500}, {}, M::InPlace, 150},
            {"prologue global prefill W70 from 0", K::Global, 70, 1, {0}, {}, M::InPlace, 2},
            {"fused local decode B2", K::Local, 1, 2, {40, 3000}, {}, M::InPlace, -1, true},
            {"fused global verify W8", K::Global, 8, 1, {9100}, {}, M::Record, -1, true},
            {"fused local prefill W96", K::Local, 96, 1, {1200}, {}, M::InPlace, -1, true},
            {"fused global prefill W40", K::Global, 40, 1, {2000}, {}, M::InPlace, -1, true},
        };
        std::uint32_t seed = 1000;
        for (const auto& c : pro) {
            if (want(c.name)) { run_prologue_case(c, seed); }
            seed += 17;
        }
        const std::vector<AttnCase> att = {
            {"attn global decode B2 (37 / 9000 keys)", K::Global, 1, 2, {36, 8999}, {}},
            {"attn local decode B2 (600 / 5000 keys)", K::Local, 1, 2, {599, 4999}, {}},
            {"attn global verify W3 B2", K::Global, 3, 2, {2100, 500}, {}},
            {"attn local verify W8 B2 valid 8/3", K::Local, 8, 2, {3000, 1100}, {8, 3}},
            {"attn global verify W8 idle row", K::Global, 8, 2, {1000, 0}, {8, 0}},
            {"attn global decode 70k keys", K::Global, 1, 1, {69999}, {}},
            {"attn global prefill W200 from 0", K::Global, 200, 1, {0}, {}},
            {"attn global prefill W130 past 2500", K::Global, 130, 1, {2500}, {}},
            {"attn local prefill W200 past 1900", K::Local, 200, 1, {1900}, {}},
            {"attn local prefill W70 from 0", K::Local, 70, 1, {0}, {}},
            {"attn global prefill W20 past 20000 split", K::Global, 20, 1, {20000}, {}},
            {"attn global prefill W130 forced 4 splits", K::Global, 130, 1, {2500}, {}, 4},
            {"attn local decode forced 1 split", K::Local, 1, 2, {599, 4999}, {}, 1},
            {"attn global decode forced 1 split", K::Global, 1, 2, {36, 8999}, {}, 1},
        };
        for (const auto& c : att) {
            if (want(c.name)) { run_attention_case(c, seed); }
            seed += 17;
        }
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION " << e.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) FAILED\n";
        return 1;
    }
    std::cout << "ALL PASS\n";
    return 0;
}
