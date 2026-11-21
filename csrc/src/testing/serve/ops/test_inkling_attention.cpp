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
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
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
    void set_table_rows(int offset) {
        std::vector<int> rows(B);
        std::iota(rows.begin(), rows.end(), offset);
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

// ---- path / batch invariance (bitwise; agent-EN decision 10:2x) -------------------------------------------------
// One plane of 8 sequences (finite random codes). The target sequence's rows [P0, L0) run as ONE prefill segment (the
// reference) and then through every other route: chunked prefill at several cuts, decode (B = 1, 4, 8 with other
// sequences' rows present), MTP verify W = 8 / 3 (with other rows, with invalid columns), other worker counts and CTA
// targets. Every route's gated bf16 output, fp32 out and lse of the target's columns must equal the reference BIT FOR
// BIT. Inputs per (sequence, position) are generated deterministically, so any route sees the same q / r / gate / sink.

struct InvPlane {
    static constexpr int kSeqs = 8;
    std::vector<int> len;
    std::unique_ptr<Plane> plane;
};

std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

InvPlane make_inv_plane(int target_len, std::uint32_t seed) {
    InvPlane ip;
    std::mt19937 g(seed);
    ip.len.push_back(target_len);
    for (int s = 1; s < InvPlane::kSeqs; ++s) { ip.len.push_back(std::uniform_int_distribution<int>(50, std::min(target_len, 9000))(g)); }
    const int maxlen = *std::max_element(ip.len.begin(), ip.len.end());
    ip.plane = std::make_unique<Plane>(InvPlane::kSeqs, maxlen + 64, 0x7F, seed + 1);
    // Codes with |value| in [2^-6, 3.75] (exponent field 1..8), random sign and mantissa: finite everywhere a key exists.
    for (int s = 0; s < InvPlane::kSeqs; ++s) {
        for (int x = 0; x < ip.len[s]; ++x) {
            for (int h = 0; h < 8; ++h) {
                std::uint8_t* kc = &ip.plane->hk[ip.plane->offset(s, x, h)];
                std::uint8_t* vc = &ip.plane->hv[ip.plane->offset(s, x, h)];
                std::uint32_t st = mix32(seed ^ (static_cast<std::uint32_t>(s) << 24) ^ (static_cast<std::uint32_t>(x) * 9u + h));
                for (int d = 0; d < 128; d += 2) {
                    st                = mix32(st + 0x9e3779b9U);
                    const auto code = [](std::uint32_t r) {
                        return static_cast<std::uint8_t>(((r & 1U) << 7) | ((1U + ((r >> 1) & 7U)) << 3) | ((r >> 4) & 7U));
                    };
                    kc[d]     = code(st);
                    kc[d + 1] = code(st >> 7);
                    vc[d]     = code(st >> 14);
                    vc[d + 1] = code(st >> 21);
                }
            }
        }
    }
    ip.plane->upload();
    return ip;
}

// q / r (clipped like the prologue's outputs), gate, sink of (sequence, position).
void inv_inputs(int seq, int pos, float* q, float* r, float* gate, float* sink) {
    Rng rng(mix32(0x51ed27u ^ (static_cast<std::uint32_t>(seq) << 26) ^ static_cast<std::uint32_t>(pos)));
    for (int h = 0; h < 24; ++h) {
        const float a = rng.uni(0.5F, 3.5F);
        float* qh     = q + h * 128;
        double ss     = 0.0;
        for (int d = 0; d < 128; ++d) { qh[d] = rng.uni(-a, a); ss += qh[d] * qh[d]; }
        const double f = std::min(1.0, 22.627 / std::sqrt(ss));
        for (int d = 0; d < 128; ++d) { qh[d] = bf(static_cast<float>(qh[d] * f)); }
        float* rh = r + h * 16;
        ss        = 0.0;
        for (int d = 0; d < 16; ++d) { rh[d] = rng.uni(-a, a); ss += rh[d] * rh[d]; }
        const double fr = std::min(1.0, 8.0 / std::sqrt(ss));
        for (int d = 0; d < 16; ++d) { rh[d] = bf(static_cast<float>(rh[d] * fr)); }
        sink[h] = bf(rng.uni(-2.0F, 6.0F));
    }
    for (int i = 0; i < 3072; ++i) { gate[i] = bf(rng.uni(-3.0F, 3.0F)); }
}

struct InvRow {
    int seq, first, valid;
};

// Outputs of one route, per column: gated bf16 bits [N][3072], out fp32 bits [N][3072], lse fp32 bits [N][24].
struct InvOut {
    int W = 0;
    bool train_same = false; // the training entry's three outputs == the production calls', bitwise
    std::vector<InvRow> rows;
    std::vector<std::uint16_t> gated;
    std::vector<std::uint32_t> out, lse;
};

InvOut run_inv_route(InklingLayerKind kind, const InvPlane& ip, const Layer& layer, int W, const std::vector<InvRow>& rws,
                     bool meta_static = false) {
    const int B = static_cast<int>(rws.size()), N = W * B;
    Rows rows;
    rows.W = W;
    rows.B = B;
    rows.pos.resize(static_cast<std::size_t>(N));
    rows.valid.resize(B);
    rows.masked = false;
    std::vector<int> table_rows(B);
    std::vector<float> q(static_cast<std::size_t>(N) * 3072, 0.0F), r(static_cast<std::size_t>(N) * 384, 0.0F),
        gate(static_cast<std::size_t>(N) * 3072, 0.0F), sink(static_cast<std::size_t>(N) * 24, 0.0F);
    for (int b = 0; b < B; ++b) {
        rows.valid[b] = rws[b].valid;
        rows.masked   = rows.masked || rws[b].valid != W;
        table_rows[b] = rws[b].seq;
        for (int j = 0; j < W; ++j) {
            const int jj = std::min(j, std::max(rws[b].valid - 1, 0));
            rows.pos[b * W + j] = rws[b].first + jj;
            const std::size_t n = static_cast<std::size_t>(j + W * b);
            if (j < rws[b].valid) {
                inv_inputs(rws[b].seq, rws[b].first + j, &q[n * 3072], &r[n * 384], &gate[n * 3072], &sink[n * 24]);
            }
        }
    }
    rows.upload();
    rows.dtables = to_device_i32(table_rows);
    DeviceBuffer dq = to_device_bf16(q), dr = to_device_bf16(r), dg = to_device_bf16(gate), ds = to_device_bf16(sink);
    ops::InklingAttentionInputs in;
    in.q         = Tensor(dq.p, DType::BF16, {128, 24, N});
    in.r         = Tensor(dr.p, DType::BF16, {16, 24, N});
    in.gate      = Tensor(dg.p, DType::BF16, {3072, N});
    in.sink      = Tensor(ds.p, DType::BF16, {24, N});
    in.rel_table = Tensor(layer.dtable.p, DType::BF16, {16, G::kRelTableRows});
    auto view            = rows.view(rows.max_keys());
    view.metadata_static = meta_static;
    WorkspaceArena work(ops::inkling_rel_attention_workspace_bytes(kind, W, B, view.envelope) + 4096);
    DeviceBuffer dy(static_cast<std::size_t>(N) * 3072 * 2), dof(static_cast<std::size_t>(N) * 3072 * 4),
        dl(static_cast<std::size_t>(N) * 24 * 4);
    cudaMemset(dy.p, 0x7f, dy.bytes);
    cudaMemset(dof.p, 0x7f, dof.bytes);
    Tensor y(dy.p, DType::BF16, {3072, N}), of(dof.p, DType::FP32, {128, 24, N}), lse(dl.p, DType::FP32, {24, N});
    ops::inkling_rel_attention(in, kind, view, ip.plane->view(), work, y, nullptr);
    ops::inkling_rel_attention_out_lse(in, kind, view, ip.plane->view(), work, of, lse, nullptr);
    // The training entry on the same rows: gated, o and lse from one pass, bitwise equal to the two calls above.
    DeviceBuffer ty(dy.bytes), to(dof.bytes), tl(dl.bytes);
    Tensor tyt(ty.p, DType::BF16, {3072, N}), tot(to.p, DType::FP32, {128, 24, N}), tlt(tl.p, DType::FP32, {24, N});
    ops::inkling_rel_attention_train(in, kind, view, ip.plane->view(), work, tyt, tot, tlt, nullptr);
    cuda_synchronize();
    InvOut o;
    o.train_same = from_device<std::uint16_t>(ty, static_cast<std::size_t>(N) * 3072) ==
                       from_device<std::uint16_t>(dy, static_cast<std::size_t>(N) * 3072) &&
                   from_device<std::uint32_t>(to, static_cast<std::size_t>(N) * 3072) ==
                       from_device<std::uint32_t>(dof, static_cast<std::size_t>(N) * 3072) &&
                   from_device<std::uint32_t>(tl, static_cast<std::size_t>(N) * 24) ==
                       from_device<std::uint32_t>(dl, static_cast<std::size_t>(N) * 24);
    o.W     = W;
    o.rows  = rws;
    o.gated = from_device<std::uint16_t>(dy, static_cast<std::size_t>(N) * 3072);
    o.out   = from_device<std::uint32_t>(dof, static_cast<std::size_t>(N) * 3072);
    o.lse   = from_device<std::uint32_t>(dl, static_cast<std::size_t>(N) * 24);
    return o;
}

struct InvCase {
    std::string name;
    InklingLayerKind kind;
    int L0, P0;
    std::vector<int> cuts, decode, verify;
};

void run_invariance_case(const InvCase& c, const char* groups, std::uint32_t seed) {
    setenv("SUROGATE_INKLING_ATTN_GROUPS", groups, 1);
    const std::string tag = "invariance " + c.name + " " + groups;
    const InvPlane ip = make_inv_plane(c.L0, seed);
    const Layer layer(c.kind, seed + 7);
    const InvOut ref = run_inv_route(c.kind, ip, layer, c.L0 - c.P0, {{0, c.P0, c.L0 - c.P0}});
    std::mt19937 g(seed + 3);
    const auto others = [&](int count) {
        std::vector<InvRow> v;
        for (int i = 0; i < count; ++i) {
            const int s = 1 + i % (InvPlane::kSeqs - 1);
            v.push_back({s, std::uniform_int_distribution<int>(0, ip.len[s] - 1)(g), 1});
        }
        return v;
    };
    std::size_t routes = 0, columns = 0, bad_columns = 0, train_bad = 0;
    std::string first_bad;
    const auto compare = [&](const std::string& route, const InvOut& o) {
        ++routes;
        if (!o.train_same) {
            ++train_bad;
            if (first_bad.empty()) { first_bad = route + " (training entry)"; }
        }
        for (std::size_t b = 0; b < o.rows.size(); ++b) {
            if (o.rows[b].seq != 0) { continue; }
            for (int j = 0; j < o.rows[b].valid; ++j) {
                const int p = o.rows[b].first + j;
                if (p < c.P0 || p >= c.L0) { continue; }
                const std::size_t n = static_cast<std::size_t>(j) + static_cast<std::size_t>(o.W) * b;
                const std::size_t m = static_cast<std::size_t>(p - c.P0);
                const bool same =
                    std::memcmp(&o.gated[n * 3072], &ref.gated[m * 3072], 3072 * 2) == 0 &&
                    std::memcmp(&o.out[n * 3072], &ref.out[m * 3072], 3072 * 4) == 0 &&
                    std::memcmp(&o.lse[n * 24], &ref.lse[m * 24], 24 * 4) == 0;
                ++columns;
                if (!same) {
                    ++bad_columns;
                    if (first_bad.empty()) { first_bad = route + " position " + std::to_string(p); }
                }
            }
        }
    };
    // Chunked prefill at the cuts (segments of <= 16 columns run the decode / verify kernel: they must agree too).
    {
        std::vector<int> edges = {c.P0};
        for (const int x : c.cuts) {
            if (x > c.P0 && x < c.L0) { edges.push_back(x); }
        }
        edges.push_back(c.L0);
        for (std::size_t i = 0; i + 1 < edges.size(); ++i) {
            const int w = edges[i + 1] - edges[i];
            compare("chunk [" + std::to_string(edges[i]) + ", " + std::to_string(edges[i + 1]) + ")",
                    run_inv_route(c.kind, ip, layer, w, {{0, edges[i], w}}, (i & 1) != 0));
        }
    }
    for (const int p : c.decode) {
        compare("decode B1", run_inv_route(c.kind, ip, layer, 1, {{0, p, 1}}));
        auto r4 = others(3);
        r4.insert(r4.begin() + 1, InvRow{0, p, 1});
        compare("decode B4", run_inv_route(c.kind, ip, layer, 1, r4));
        compare("decode B4 static metadata", run_inv_route(c.kind, ip, layer, 1, r4, true));
        auto r8 = others(7);
        r8.insert(r8.begin() + 5, InvRow{0, p, 1});
        compare("decode B8", run_inv_route(c.kind, ip, layer, 1, r8));
        r8.back().valid = 0; // an idle padding row
        compare("decode B8 idle row", run_inv_route(c.kind, ip, layer, 1, r8));
        for (const char* splits : {"1", "3", "64"}) {
            setenv("SUROGATE_INKLING_ATTN_SPLITS", splits, 1);
            compare(std::string("decode B1 workers ") + splits, run_inv_route(c.kind, ip, layer, 1, {{0, p, 1}}));
        }
        unsetenv("SUROGATE_INKLING_ATTN_SPLITS");
        setenv("SUROGATE_INKLING_ATTN_CTAS", "20", 1);
        compare("decode B4 cta target 20", run_inv_route(c.kind, ip, layer, 1, r4));
        unsetenv("SUROGATE_INKLING_ATTN_CTAS");
    }
    for (const int v : c.verify) {
        const int w = std::min(8, c.L0 - v);
        compare("verify W8 B1", run_inv_route(c.kind, ip, layer, 8, {{0, v, w}}));
        std::vector<InvRow> r4;
        for (const auto& o : others(3)) { r4.push_back({o.seq, std::max(0, std::min(o.first, ip.len[o.seq] - 8)), std::min(8, ip.len[o.seq])}); }
        r4.insert(r4.begin() + 2, InvRow{0, v, w});
        compare("verify W8 B4", run_inv_route(c.kind, ip, layer, 8, r4));
        compare("verify W8 B4 static metadata", run_inv_route(c.kind, ip, layer, 8, r4, true));
        compare("verify W8 B1 valid 5", run_inv_route(c.kind, ip, layer, 8, {{0, v, std::min(5, w)}}));
        compare("verify W3 B2", run_inv_route(c.kind, ip, layer, 3, {{r4[0].seq, r4[0].first, std::min(3, r4[0].valid)}, {0, v, std::min(3, w)}}));
        compare("verify W16 B1", run_inv_route(c.kind, ip, layer, 16, {{0, std::max(c.P0, std::min(v, c.L0 - 16)), 16}}));
        setenv("SUROGATE_INKLING_ATTN_SPLITS", "1", 1);
        compare("verify W8 B4 workers 1", run_inv_route(c.kind, ip, layer, 8, r4));
        unsetenv("SUROGATE_INKLING_ATTN_SPLITS");
    }
    // The whole segment again with the 4-warp prefill kernel (KS = 1; the default is KS = 2).
    setenv("SUROGATE_INKLING_PREFILL_KS", "1", 1);
    compare("prefill KS1 one segment", run_inv_route(c.kind, ip, layer, c.L0 - c.P0, {{0, c.P0, c.L0 - c.P0}}));
    unsetenv("SUROGATE_INKLING_PREFILL_KS");
    // A short prefill segment forced through the split route (partials + fold kernel) and through 1 worker.
    {
        const int s0 = std::max(c.P0, c.L0 - 40);
        for (const char* splits : {"1", "3"}) {
            setenv("SUROGATE_INKLING_ATTN_SPLITS", splits, 1);
            compare(std::string("prefill W40 workers ") + splits, run_inv_route(c.kind, ip, layer, c.L0 - s0, {{0, s0, c.L0 - s0}}));
        }
        unsetenv("SUROGATE_INKLING_ATTN_SPLITS");
    }
    std::printf("%-44s routes %zu, target columns compared %zu, bit-different %zu, training entry != production on %zu "
                "routes%s%s\n",
                tag.c_str(), routes, columns, bad_columns, train_bad + (ref.train_same ? 0 : 1),
                first_bad.empty() ? "" : "  first: ", first_bad.c_str());
    check(bad_columns == 0 && train_bad == 0 && ref.train_same && columns > 0, tag);
    unsetenv("SUROGATE_INKLING_ATTN_GROUPS");
}

void test_invariance(const std::function<bool(const std::string&)>& want) {
    using K = InklingLayerKind;
    const std::vector<InvCase> cases = {
        {"global short", K::Global, 4700, 0, {1, 17, 64, 100, 511, 512, 513, 1024, 2047, 2048, 2049, 3000},
         {0, 1, 63, 64, 511, 512, 1000, 2047, 2048, 4699}, {0, 505, 2044, 4692}},
        {"global long", K::Global, 70000, 64000, {65535, 65536, 65537, 66000, 68000, 69990},
         {64000, 65535, 65536, 65537, 69999}, {65530, 69992}},
        {"local short", K::Local, 5000, 0, {1, 100, 1023, 1024, 1025, 1100, 2000, 4000, 4990},
         {0, 127, 128, 1023, 1024, 1100, 4999}, {0, 1020, 4992}},
        {"local long", K::Local, 70000, 68000, {68100, 69000, 69990}, {68000, 69999}, {69000}},
        {"global 140k", K::Global, 140000, 130000, {131000, 131071, 131072, 131073, 135000}, {130000, 131071, 131072, 139999},
         {131068}},
    };
    std::uint32_t seed = 7000;
    for (const auto& c : cases) {
        for (const char* groups : {"p1", "p2", "p3"}) {
            if (want("invariance " + c.name + " " + groups)) { run_invariance_case(c, groups, seed); }
        }
        seed += 31;
    }
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
        // The attention part on its own: rerun the prologue op alone on the original state and plane (it must
        // write the same KV and state as the fused call: determinism), then feed the FP64 attention oracle the
        // GPU's own q_out / r_out and KV codes, so the prologue's rare 1-ulp differences (checked above) do not
        // enter the attention check.
        Plane plane2(c.B, maxk + 64, 0x7F, seed + 2);
        plane2.hk = plane_k0;
        plane2.hv = plane_v0;
        plane2.upload();
        DeviceBuffer dstate2 = to_device_bf16(state), drec2 = to_device_bf16(records);
        ops::InklingConvStateArgs conv2 = conv;
        conv2.states = Tensor(dstate2.p, DType::BF16, {kC, 3, S});
        if (c.mode == ops::InklingConvMode::Record) { conv2.records = Tensor(drec2.p, DType::BF16, {kC, Wmax, c.B}); }
        ops::inkling_attention_prologue(in, layer.weights(), c.kind, view, conv2, plane2.view(), qr, nullptr);
        cuda_synchronize();
        plane2.download();
        const bool same_kv    = plane2.hk == plane.hk && plane2.hv == plane.hv;
        const bool same_state = from_device<std::uint16_t>(dstate2, state.size()) ==
                                from_device<std::uint16_t>(dstate, state.size());
        std::printf("%-44s %-8s prologue alone == fused: kv %s, state %s\n", c.name.c_str(), "determ.",
                    same_kv ? "yes" : "NO", same_state ? "yes" : "NO");
        check(same_kv && same_state, c.name + " prologue alone == fused");
        std::vector<float> q_gpu(q_ref.size()), r_gpu(r_ref.size());
        {
            const auto qb = from_device<std::uint16_t>(dqo, q_gpu.size());
            const auto rb = from_device<std::uint16_t>(dro, r_gpu.size());
            for (std::size_t i = 0; i < q_gpu.size(); ++i) { q_gpu[i] = bf16_to_f32(qb[i]); }
            for (std::size_t i = 0; i < r_gpu.size(); ++i) { r_gpu[i] = bf16_to_f32(rb[i]); }
        }
        std::vector<double> ref_o, ref_l;
        attention_oracle(layer, rows, plane, q_gpu, r_gpu, ref_o, ref_l);
        std::vector<float> gate(static_cast<std::size_t>(N) * 3072), sink(static_cast<std::size_t>(N) * 24);
        for (int n = 0; n < N; ++n) {
            for (int i = 0; i < 3072; ++i) { gate[static_cast<std::size_t>(n) * 3072 + i] = proj[static_cast<std::size_t>(n) * 8600 + G::kRowsGate + i]; }
            for (int i = 0; i < 24; ++i) { sink[static_cast<std::size_t>(n) * 24 + i] = proj[static_cast<std::size_t>(n) * 8600 + G::kRowsSink + i]; }
        }
        const std::vector<double> ref_y = gated_oracle(rows, ref_o, ref_l, gate, sink);
        const std::vector<double> got_y = from_device_bf16(dout, static_cast<std::size_t>(N) * 3072);
        check(verify_reduction(c.name + " fused gated", got_y, ref_y, kGatedCriterion) == 0, c.name + " fused");

        // The fused training call on a third copy of the original state and plane: its gated output == the fused
        // call's, its q / r == the prologue's, its o / lse == the out-lse route's on those q / r, bitwise; tau (global)
        // within 1 ulp of the host's logf (the prologue's own device function).
        Plane plane3(c.B, maxk + 64, 0x7F, seed + 2);
        plane3.hk = plane_k0;
        plane3.hv = plane_v0;
        plane3.upload();
        DeviceBuffer dstate3 = to_device_bf16(state), drec3 = to_device_bf16(records);
        ops::InklingConvStateArgs conv3 = conv;
        conv3.states = Tensor(dstate3.p, DType::BF16, {kC, 3, S});
        if (c.mode == ops::InklingConvMode::Record) { conv3.records = Tensor(drec3.p, DType::BF16, {kC, Wmax, c.B}); }
        const std::size_t sn = static_cast<std::size_t>(N);
        DeviceBuffer tq(sn * 3072 * 2), tr(sn * 384 * 2), ttau(sn * 4), to(sn * 3072 * 4), tl(sn * 24 * 4), ty(sn * 3072 * 2),
            ro(sn * 3072 * 4), rl(sn * 24 * 4);
        ops::InklingAttentionTrainOutputs tro;
        tro.q   = Tensor(tq.p, DType::BF16, {128, 24, N});
        tro.r   = Tensor(tr.p, DType::BF16, {16, 24, N});
        tro.tau = glob ? Tensor(ttau.p, DType::FP32, {N}) : Tensor{};
        tro.o   = Tensor(to.p, DType::FP32, {128, 24, N});
        tro.lse = Tensor(tl.p, DType::FP32, {24, N});
        Tensor tyt(ty.p, DType::BF16, {3072, N}), rot(ro.p, DType::FP32, {128, 24, N}), rlt(rl.p, DType::FP32, {24, N});
        WorkspaceArena work3(ops::inkling_attention_workspace_capacity_bytes(c.kind, c.B, c.W, c.W, view.envelope) + 4096);
        ops::inkling_attention_train(in, layer.weights(), c.kind, view, conv3, plane3.view(), work3, tyt, tro, nullptr);
        const ops::InklingAttentionInputs ain{tro.q, tro.r, in.gate, in.sink, layer.weights().rel_table};
        ops::inkling_rel_attention_out_lse(ain, c.kind, view, plane3.view(), work3, rot, rlt, nullptr);
        cuda_synchronize();
        const bool same_y = from_device<std::uint16_t>(ty, sn * 3072) == from_device<std::uint16_t>(dout, sn * 3072);
        const bool same_q = from_device<std::uint16_t>(tq, sn * 3072) == from_device<std::uint16_t>(dqo, sn * 3072);
        const bool same_r = from_device<std::uint16_t>(tr, sn * 384) == from_device<std::uint16_t>(dro, sn * 384);
        const bool same_o = from_device<std::uint32_t>(to, sn * 3072) == from_device<std::uint32_t>(ro, sn * 3072) &&
                            from_device<std::uint32_t>(tl, sn * 24) == from_device<std::uint32_t>(rl, sn * 24);
        int tau_bad = 0;
        if (glob) {
            const std::vector<float> tg = from_device<float>(ttau, sn);
            for (int i = 0; i < N; ++i) {
                const float want = host_tau(rows.pos[i]);
                tau_bad += std::fabs(tg[i] - want) > std::fabs(want) * 2.4e-7F;
            }
        }
        std::printf("%-44s %-8s training call: out %s, q %s, r %s, o+lse %s, tau off > 1 ulp %d\n", c.name.c_str(), "train",
                    same_y ? "==" : "DIFF", same_q ? "==" : "DIFF", same_r ? "==" : "DIFF", same_o ? "==" : "DIFF", tau_bad);
        check(same_y && same_q && same_r && same_o && tau_bad == 0, c.name + " training call");
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
    std::vector<double> dequant; // e4m3(code) * scale, exact in double (lane a's fp8_quant_weight value)
    DeviceBuffer dcodes, dscales;
    Weight w{};

    Fp8Weight(int n_in, int k_in, int k_per_in, int rows_per_in, std::uint32_t seed, bool with_dequant = true)
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
        if (with_dequant) { dequant.resize(codes.size()); }
        for (int r = 0; with_dequant && r < n; ++r) {
            for (int c = 0; c < k; ++c) {
                dequant[static_cast<std::size_t>(r) * k + c] =
                    static_cast<double>(e4m3_decode(codes[static_cast<std::size_t>(r) * k + c])) *
                    scales[static_cast<std::size_t>(r / rows_per) * (k / k_per) + c / k_per];
            }
        }
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
    const double* row(int r) const { return &dequant[static_cast<std::size_t>(r) * k]; }
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
                double acc       = 0.0;
                const double* wr = fw.row(ranges[i].first + r);
                const float* xc  = &xs[i][static_cast<std::size_t>(c) * k];
                for (int kk = 0; kk < k; ++kk) { acc += wr[kk] * xc[kk]; }
                ref.push_back(acc);
                g.push_back(bf16_to_f32(got[static_cast<std::size_t>(c) * (n_rows + 8) + r]));
            }
        }
        check(verify_reduction(name + " segment " + std::to_string(i), g, ref,
                               ReductionCriterion{2.0e-3, 1.0e-4, 4.0e-3}) == 0,
              name + " segment " + std::to_string(i));
    }
}

// Width invariance (bitwise): the first n columns of an N = 8,192 call (GEMM) equal a call with only those n columns,
// n = 1, 2, 8, 9, 16 (narrow kernel), 17, 64 (GEMM), for every projection of a 1- or 3-input launch.
void run_linear_width_case(const std::string& name, int k, int k_per, int rows_per,
                           const std::vector<std::pair<int, int>>& ranges, std::uint32_t seed) {
    const int n_rows = k == 2304 ? 8600 : 2304;
    const int nmax   = 8192;
    const Fp8Weight fw(n_rows, k, k_per, rows_per, seed, false);
    std::vector<DeviceBuffer> dxs;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * nmax);
        std::uint32_t st = mix32(seed * 31u + static_cast<std::uint32_t>(i));
        for (auto& v : x) {
            st = mix32(st + 0x9e3779b9U);
            v  = f32_to_bf16(static_cast<float>(static_cast<int>(st % 60001U) - 30000) * 1e-4F);
        }
        dxs.push_back(to_device(x));
    }
    const auto run = [&](int n) {
        std::vector<DeviceBuffer> douts;
        std::vector<Tensor> xt, ot;
        std::vector<ops::InklingProjection> pj(ranges.size());
        xt.reserve(ranges.size());
        ot.reserve(ranges.size());
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            douts.emplace_back(static_cast<std::size_t>(ranges[i].second) * n * 2);
            xt.emplace_back(dxs[i].p, DType::BF16, std::initializer_list<std::int32_t>{k, n});
            ot.emplace_back(douts[i].p, DType::BF16, std::initializer_list<std::int32_t>{ranges[i].second, n});
            pj[i] = {&xt[i], &fw.w, ranges[i].first, &ot[i]};
        }
        ops::inkling_attn_linear(pj, nullptr, nullptr);
        cuda_synchronize();
        std::vector<std::vector<std::uint16_t>> res;
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            res.push_back(from_device<std::uint16_t>(douts[i], static_cast<std::size_t>(ranges[i].second) * n));
        }
        return res;
    };
    const auto ref = run(nmax);
    std::size_t bad = 0, total = 0;
    std::string first;
    for (const int n : {1, 2, 8, 9, 16, 17, 64}) {
        const auto got = run(n);
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            const std::size_t count = static_cast<std::size_t>(ranges[i].second) * n; // columns are contiguous
            total += count;
            for (std::size_t e = 0; e < count; ++e) {
                if (got[i][e] != ref[i][e]) {
                    ++bad;
                    if (first.empty()) { first = "N=" + std::to_string(n) + " segment " + std::to_string(i); }
                }
            }
        }
    }
    std::printf("%-44s widths 1 2 8 9 16 17 64 vs 8192: %zu / %zu outputs bit-different%s%s\n", name.c_str(), bad, total,
                first.empty() ? "" : "  first: ", first.c_str());
    check(bad == 0, name);
}

void test_attn_linear() {
    {
        const std::vector<std::pair<int, int>> three = {{0, G::kRowsK}, {G::kRowsK, 1024}, {G::kRowsV, 1024}};
        std::uint32_t wseed = 4500;
        run_linear_width_case("attn_linear width qkvr 1x128", 2304, 128, 1, {{0, 8600}}, wseed++);
        run_linear_width_case("attn_linear width qkvr channel", 2304, 2304, 1, {{0, 8600}}, wseed++);
        run_linear_width_case("attn_linear width qkvr 128x128", 2304, 128, 128, {{0, 8600}}, wseed++);
        run_linear_width_case("attn_linear width 3-input 1x128", 2304, 128, 1, three, wseed++);
        run_linear_width_case("attn_linear width wo 1x128", 3072, 128, 1, {{0, 2304}}, wseed++);
        run_linear_width_case("attn_linear width wo channel", 3072, 3072, 1, {{0, 2304}}, wseed++);
    }
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

// ---- replay of vLLM's own kernels (lane a's capture of p000[:256], exported by lanes/c/export_capture.py) ---

struct CapTensor {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;
    std::int64_t numel() const {
        std::int64_t n = 1;
        for (auto d : shape) { n *= d; }
        return n;
    }
    std::vector<float> bf16_values() const {
        std::vector<float> out(numel());
        for (std::int64_t i = 0; i < numel(); ++i) {
            std::uint16_t b;
            std::memcpy(&b, &bytes[2 * i], 2);
            out[i] = bf16_to_f32(b);
        }
        return out;
    }
    std::vector<float> f32_values() const {
        std::vector<float> out(numel());
        std::memcpy(out.data(), bytes.data(), out.size() * 4);
        return out;
    }
};

std::vector<std::uint8_t> read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { throw std::runtime_error("cannot open " + path); }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint8_t> b(static_cast<std::size_t>(n));
    if (n > 0 && std::fread(b.data(), 1, b.size(), f) != b.size()) { throw std::runtime_error("short read " + path); }
    std::fclose(f);
    return b;
}

void run_capture(const std::string& dir) {
    std::map<int, std::map<std::string, CapTensor>> visits;
    std::map<int, bool> global;
    {
        FILE* f = std::fopen((dir + "/MANIFEST.txt").c_str(), "r");
        if (f == nullptr) { throw std::runtime_error("no " + dir + "/MANIFEST.txt"); }
        char line[1024];
        while (std::fgets(line, sizeof line, f) != nullptr) {
            std::istringstream in(line);
            int v;
            std::string name;
            in >> v >> name;
            if (name == "meta") {
                std::string tok;
                while (in >> tok) {
                    if (tok.rfind("global=", 0) == 0) { global[v] = tok.back() == '1'; }
                }
                continue;
            }
            CapTensor t;
            std::string file;
            in >> t.dtype >> file;
            std::int64_t d;
            while (in >> d) { t.shape.push_back(d); }
            t.bytes = read_file(dir + "/" + file);
            visits[v][name] = std::move(t);
        }
        std::fclose(f);
    }
    for (auto& [v, ts] : visits) {
        const bool glob = global[v];
        const InklingLayerKind kind = glob ? InklingLayerKind::Global : InklingLayerKind::Local;
        const int T = static_cast<int>(ts.at("positions").numel());
        const std::string tag = "capture v" + std::to_string(v) + (glob ? " global" : " local");
        const auto dev = [&](const std::string& n) { return to_device(ts.at(n).bytes); };
        std::vector<int> pos(T);
        std::memcpy(pos.data(), ts.at("positions").bytes.data(), static_cast<std::size_t>(T) * 4);
        bool consecutive = true;
        for (int i = 0; i < T; ++i) { consecutive = consecutive && pos[i] == pos[0] + i; }
        check(consecutive, tag + " positions consecutive");
        Rows rows = make_rows(T, 1, {pos[0]}, {});

        DeviceBuffer dq = dev("q"), dr = dev("r"), dk = dev("k"), dv = dev("v"), dg = dev("gate"), ds = dev("sink");
        DeviceBuffer dqs = dev("q_scale"), drs = dev("r_scale"), dkc = dev("k_conv"), dvc = dev("v_conv"),
                     dproj = dev("rel_proj");
        DeviceBuffer dkd, dvd;
        ops::InklingProjections in;
        in.q    = Tensor(dq.p, DType::BF16, {128, 24, T});
        in.r    = Tensor(dr.p, DType::BF16, {16, 24, T});
        in.k    = Tensor(dk.p, DType::BF16, {1024, T});
        in.v    = Tensor(dv.p, DType::BF16, {1024, T});
        in.gate = Tensor(dg.p, DType::BF16, {3072, T});
        in.sink = Tensor(ds.p, DType::BF16, {24, T});
        if (glob) {
            dkd      = dev("k_dyn");
            dvd      = dev("v_dyn");
            in.k_dyn = Tensor(dkd.p, DType::BF16, {4, 8, T});
            in.v_dyn = Tensor(dvd.p, DType::BF16, {4, 8, T});
        }
        DeviceBuffer dtable(static_cast<std::size_t>(16) * G::kRelTableRows * 2);
        ops::InklingAttentionWeights w;
        w.q_scale   = Tensor(dqs.p, DType::BF16, {128, 24});
        w.r_scale   = Tensor(drs.p, DType::BF16, {16, 24});
        w.k_conv    = glob ? Tensor(dkc.p, DType::BF16, {1024, 4}) : Tensor(dkc.p, DType::BF16, {4, 1024});
        w.v_conv    = glob ? Tensor(dvc.p, DType::BF16, {1024, 4}) : Tensor(dvc.p, DType::BF16, {4, 1024});
        w.rel_table = Tensor(dtable.p, DType::BF16, {16, G::kRelTableRows});
        {
            Tensor proj(dproj.p, DType::BF16, {1024, 16});
            ops::inkling_rel_proj_table(proj, w.rel_table, nullptr);
        }
        std::vector<std::uint16_t> zeros(static_cast<std::size_t>(kC) * 3, 0);
        DeviceBuffer dstate = to_device(zeros), dslot = to_device_i32(std::vector<int>{0});
        ops::InklingConvStateArgs conv;
        conv.states = Tensor(dstate.p, DType::BF16, {kC, 3, 1});
        conv.slots  = Tensor(dslot.p, DType::I32, {1});
        Plane plane(1, pos[0] + T + 64, 0x7F, 99);
        plane.upload();
        DeviceBuffer dqo(static_cast<std::size_t>(T) * 3072 * 2), dro(static_cast<std::size_t>(T) * 384 * 2);
        ops::InklingPrologueOutputs qr;
        qr.q = Tensor(dqo.p, DType::BF16, {128, 24, T});
        qr.r = Tensor(dro.p, DType::BF16, {16, 24, T});
        const auto view = rows.view(pos[0] + T);
        ops::inkling_attention_prologue(in, w, kind, view, conv, plane.view(), qr, nullptr);

        WorkspaceArena work(ops::inkling_rel_attention_workspace_bytes(kind, T, 1, view.envelope) + 4096);
        DeviceBuffer dof(static_cast<std::size_t>(T) * 3072 * 4), dl(static_cast<std::size_t>(T) * 24 * 4),
            dy(static_cast<std::size_t>(T) * 3072 * 2), dgs(static_cast<std::size_t>(T) * 3072 * 2);
        Tensor of(dof.p, DType::FP32, {128, 24, T}), lse(dl.p, DType::FP32, {24, T}), y(dy.p, DType::BF16, {3072, T});
        const ops::InklingAttentionInputs ain{qr.q, qr.r, in.gate, in.sink, w.rel_table};
        ops::inkling_rel_attention_out_lse(ain, kind, view, plane.view(), work, of, lse, nullptr);
        ops::inkling_rel_attention(ain, kind, view, plane.view(), work, y, nullptr);
        // gate_sink alone on vLLM's own out / lse: the same instructions, fusion off -> expected bit-exact.
        DeviceBuffer dvo = dev("out"), dvl = dev("lse");
        Tensor vo(dvo.p, DType::FP32, {128, 24, T}), vl(dvl.p, DType::FP32, {24, T}), gres(dgs.p, DType::BF16, {3072, T});
        ops::inkling_gate_sink(vo, vl, in.gate, in.sink, gres, nullptr);
        cuda_synchronize();
        plane.download();

        // q_out bits vs vLLM.
        const auto bits_vs = [&](const char* what, const std::vector<std::uint16_t>& got, const CapTensor& ref,
                                 double max_frac, bool exact) {
            std::size_t off1 = 0, worse = 0;
            for (std::size_t i = 0; i < got.size(); ++i) {
                std::uint16_t r;
                std::memcpy(&r, &ref.bytes[2 * i], 2);
                const int d = ulp_distance_bf16(got[i], r);
                off1 += d == 1;
                worse += d > 1;
            }
            std::printf("%-44s %-10s vs vLLM: 1-ulp %zu / %zu, worse %zu\n", tag.c_str(), what, off1, got.size(), worse);
            check(exact ? off1 + worse == 0 : (worse == 0 && off1 <= max_frac * static_cast<double>(got.size())),
                  tag + " " + what);
        };
        bits_vs("q_out", from_device<std::uint16_t>(dqo, static_cast<std::size_t>(T) * 3072), ts.at("q_out"), 0.005, false);
        bits_vs("gate_sink", from_device<std::uint16_t>(dgs, static_cast<std::size_t>(T) * 3072), ts.at("result"), 0.0, true);
        {
            const std::vector<double> gy = from_device_bf16(dy, static_cast<std::size_t>(T) * 3072);
            std::vector<double> ry(gy.size());
            const std::vector<float> rv = ts.at("result").bf16_values();
            for (std::size_t i = 0; i < ry.size(); ++i) { ry[i] = rv[i]; }
            check(verify_reduction(tag + " gated vs vLLM", gy, ry, kGatedCriterion) == 0, tag + " gated vs vLLM");
        }
        // Rows (token, head) whose q_out differs from vLLM's by an ulp (the L2-clip sum order).
        std::vector<char> q_row_diff(static_cast<std::size_t>(T) * 24, 0);
        {
            const auto qg = from_device<std::uint16_t>(dqo, static_cast<std::size_t>(T) * 3072);
            for (std::size_t i = 0; i < qg.size(); ++i) {
                std::uint16_t r;
                std::memcpy(&r, &ts.at("q_out").bytes[2 * i], 2);
                if (qg[i] != r) { q_row_diff[i / 128] = 1; }
            }
        }
        // KV codes at every written position vs vLLM's cache rows.
        std::size_t kv_off1 = 0, kv_worse = 0;
        for (int which = 0; which < 2; ++which) {
            const CapTensor& ref = ts.at(which ? "v_cache" : "k_cache");
            for (int t = 0; t < T; ++t) {
                for (int h = 0; h < 8; ++h) {
                    const std::size_t off = plane.offset(0, pos[t], h);
                    for (int d = 0; d < 128; ++d) {
                        const std::uint8_t g = which ? plane.hv[off + d] : plane.hk[off + d];
                        const int u = ulp_distance_e4m3(g, ref.bytes[(static_cast<std::size_t>(t) * 8 + h) * 128 + d]);
                        kv_off1 += u == 1;
                        kv_worse += u > 1;
                    }
                }
            }
        }
        std::printf("%-44s %-10s vs vLLM: 1-ulp %zu / %d, worse %zu\n", tag.c_str(), "kv codes", kv_off1, 2 * T * 1024, kv_worse);
        check(kv_worse == 0 && kv_off1 <= 0.005 * 2 * T * 1024, tag + " kv codes");
        // Attention out / lse vs vLLM's prefill kernel (P rounded per 64-key block there, per 32-key block here).
        const std::vector<double> go = from_device_f32(dof, static_cast<std::size_t>(T) * 3072);
        std::vector<double> ro(go.size());
        {
            const std::vector<float> rf = ts.at("out").f32_values();
            for (std::size_t i = 0; i < ro.size(); ++i) { ro[i] = rf[i]; }
        }
        check(verify_reduction(tag + " out vs vLLM", go, ro, kOutCriterion) == 0, tag + " out vs vLLM");
        const std::vector<double> gl = from_device_f32(dl, static_cast<std::size_t>(T) * 24);
        const std::vector<float> rl  = ts.at("lse").f32_values();
        // Rows off by > 1e-4 must be rows whose q differs by an ulp (or, unobservable here, whose r does): count
        // the rest, and bound everything.
        double lmax = 0.0, lmax_clean = 0.0;
        std::size_t off = 0, off_q = 0;
        for (std::size_t i = 0; i < gl.size(); ++i) {
            const double d = std::fabs(gl[i] - rl[i]);
            lmax = std::max(lmax, d);
            if (!q_row_diff[i]) { lmax_clean = std::max(lmax_clean, d); }
            if (d > 1e-4) {
                ++off;
                off_q += q_row_diff[i];
            }
        }
        std::printf("%-44s %-10s vs vLLM: max|d| %.3e (rows with equal q: %.3e); rows > 1e-4: %zu (%zu with a q_out ulp)\n",
                    tag.c_str(), "lse", lmax, lmax_clean, off, off_q);
        check(lmax <= 2e-2 && off * 200 <= gl.size(), tag + " lse vs vLLM");
        // Numerics-variant sweep (the vLLM bit-identity experiment): vLLM's own q_out, every variant of
        // AttnParams::variant, share of out / lse values whose fp32 bits equal vLLM's. Visits 0 and 5 only.
        if (v == 0 || v == 5) {
            DeviceBuffer dvq = dev("q_out");
            const ops::InklingAttentionInputs vin{Tensor(dvq.p, DType::BF16, {128, 24, T}), qr.r, in.gate, in.sink,
                                                  w.rel_table};
            const std::vector<float> ro32 = ts.at("out").f32_values(), rl32 = ts.at("lse").f32_values();
            struct Score { int variant; double out_eq, lse_eq; };
            std::vector<Score> scores;
            for (int variant = 0; variant < 64; ++variant) {
                if ((variant & 6) != 0) { continue; } // bits 2 / 4 (old P V modes) are gone
                setenv("SUROGATE_INKLING_ATTN_VARIANT", std::to_string(variant).c_str(), 1);
                ops::inkling_rel_attention_out_lse(vin, kind, view, plane.view(), work, of, lse, nullptr);
                cuda_synchronize();
                const std::vector<float> go32 = from_device<float>(dof, ro32.size());
                const std::vector<float> gl32 = from_device<float>(dl, rl32.size());
                std::size_t eo = 0, el = 0;
                for (std::size_t i = 0; i < go32.size(); ++i) { eo += std::memcmp(&go32[i], &ro32[i], 4) == 0; }
                for (std::size_t i = 0; i < gl32.size(); ++i) { el += std::memcmp(&gl32[i], &rl32[i], 4) == 0; }
                scores.push_back({variant, 100.0 * eo / go32.size(), 100.0 * el / gl32.size()});
            }
            unsetenv("SUROGATE_INKLING_ATTN_VARIANT");
            std::sort(scores.begin(), scores.end(), [](const Score& a, const Score& b) {
                return a.out_eq != b.out_eq ? a.out_eq > b.out_eq : a.lse_eq > b.lse_eq;
            });
            for (std::size_t k = 0; k < scores.size(); ++k) {
                if (k < 6 || scores[k].variant == 0) {
                    std::printf("%-44s variant %2d: out bits = vLLM %6.2f %%, lse bits = vLLM %6.2f %%\n", tag.c_str(),
                                scores[k].variant, scores[k].out_eq, scores[k].lse_eq);
                }
            }
        }
        if (glob && ts.count("tau") != 0) {
            DeviceBuffer dpos = to_device_i32(pos), dtau(static_cast<std::size_t>(T) * 4);
            Tensor tp(dpos.p, DType::I32, {T}), tt(dtau.p, DType::FP32, {T});
            ops::inkling_log_tau(tp, tt, nullptr);
            cuda_synchronize();
            const std::vector<float> gt = from_device<float>(dtau, T), rt = ts.at("tau").f32_values();
            std::size_t bad = 0;
            for (int i = 0; i < T; ++i) { bad += gt[i] != rt[i]; }
            std::printf("%-44s %-10s vs vLLM: mismatches %zu / %d\n", tag.c_str(), "tau", bad, T);
            check(bad == 0, tag + " tau vs vLLM");
        }
    }
}

// ---- bench: per-kernel time (eager, back-to-back launches on one stream; no CUDA graph) ------------------------

struct BenchPlane {
    int logical = 0;
    DeviceBuffer k, v, bt;
    BenchPlane(int rows, int tokens) : logical((tokens + 63) / 64) {
        const std::size_t pages = static_cast<std::size_t>(rows) * logical;
        k = DeviceBuffer(pages * 8 * 64 * 128);
        v = DeviceBuffer(pages * 8 * 64 * 128);
        cuda_check(cudaMemset(k.p, 0x30, k.bytes), "memset");
        cuda_check(cudaMemset(v.p, 0x31, v.bytes), "memset");
        std::vector<int> t(pages);
        std::iota(t.begin(), t.end(), 0);
        bt = to_device_i32(t);
    }
    PagedKVBatchLayerView view(int rows) const {
        PagedKVBatchLayerView vw;
        const int pages = rows * logical;
        vw.k_pages      = Tensor(k.p, DType::FP8_E4M3FN, {128, 64, 8, pages});
        vw.v_pages      = Tensor(v.p, DType::FP8_E4M3FN, {128, 64, 8, pages});
        vw.block_tables = Tensor(bt.p, DType::I32, {logical, rows});
        vw.head_dim     = 128;
        vw.num_kv_heads = 8;
        vw.dtype        = DType::FP8_E4M3FN;
        return vw;
    }
};

cudaStream_t g_stream = nullptr; // the bench's stream (capturable)

double time_us(const std::function<void()>& f, int reps = 20) {
    for (int i = 0; i < 3; ++i) { f(); }
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, g_stream);
    for (int i = 0; i < reps; ++i) { f(); }
    cudaEventRecord(b, g_stream);
    cudaEventSynchronize(b);
    float ms = 0.0F;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cuda_check(cudaGetLastError(), "bench");
    return 1000.0 * ms / reps;
}

// Time per call when `calls` calls (call(i), i = 0..calls-1) are replayed as one CUDA graph: no host launch gaps,
// PDL edges as captured -- how the engine's decode rounds run.
double time_graph_us(const std::function<void(int)>& call, int calls, int reps = 10) {
    cuda_check(cudaStreamSynchronize(g_stream), "sync");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cuda_check(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal), "capture");
    for (int i = 0; i < calls; ++i) { call(i); }
    cuda_check(cudaStreamEndCapture(g_stream, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    for (int i = 0; i < 2; ++i) { cuda_check(cudaGraphLaunch(exec, g_stream), "graph launch"); }
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, g_stream);
    for (int i = 0; i < reps; ++i) { cudaGraphLaunch(exec, g_stream); }
    cudaEventRecord(b, g_stream);
    cudaEventSynchronize(b);
    float ms = 0.0F;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cuda_check(cudaGetLastError(), "graph bench");
    return 1000.0 * ms / (static_cast<double>(reps) * calls);
}

constexpr double kBwGBs  = 1707.0; // lane h, measured decode DRAM bandwidth
constexpr double kBf16TF = 253.0;  // lane h, measured BF16 MMA peak

void bench_attention(const char* name, InklingLayerKind kind, int W, int B, int first) {
    const Layer layer(kind, 11);
    const int N = W * B, keys = first + W;
    // R copies of every row's KV (> 256 MB in all), one per timed call in turn: the reads come from DRAM, not L2.
    const double row_bytes = (kind == InklingLayerKind::Global ? keys : std::min(keys, 1088)) * 2048.0;
    const int R = W <= G::kMaxSmallColumns ? std::max(1, static_cast<int>(std::ceil(256e6 / (row_bytes * B)))) : 1;
    std::vector<Rows> rot;
    for (int r = 0; r < R; ++r) {
        rot.push_back(make_rows(W, B, std::vector<int>(B, first), {}));
        rot.back().set_table_rows(r * B);
    }
    const Rows& rows = rot[0];
    BenchPlane plane(B * R, keys + 64);
    Rng rng(3);
    std::vector<float> q, r;
    fill_qr(rng, rows, q, r);
    std::vector<float> qside(static_cast<std::size_t>(N) * 6552, 0.25F);
    DeviceBuffer dq = to_device_bf16(q), dr = to_device_bf16(r), dqs = to_device_bf16(qside);
    ops::InklingAttentionInputs in{Tensor(dq.p, DType::BF16, {128, 24, N}), Tensor(dr.p, DType::BF16, {16, 24, N}),
                                   strided_bf16(dqs.p, G::kRowsGate, {3072}, N, 6552),
                                   strided_bf16(dqs.p, G::kRowsSink, {24}, N, 6552),
                                   Tensor(layer.dtable.p, DType::BF16, {16, G::kRelTableRows})};
    std::vector<ops::InklingAttentionRows> views;
    for (const auto& rr : rot) {
        views.push_back(rr.view(keys));
        views.back().metadata_static = std::getenv("BENCH_META_STATIC") != nullptr;
    }
    WorkspaceArena work(ops::inkling_rel_attention_workspace_bytes(kind, W, B, views[0].envelope) + 4096);
    DeviceBuffer dout(static_cast<std::size_t>(N) * 3072 * 2);
    Tensor out(dout.p, DType::BF16, {3072, N});
    const auto pv  = plane.view(B * R);
    const auto call = [&](int i) { ops::inkling_rel_attention(in, kind, views[i % R], pv, work, out, g_stream); };
    int turn = 0;
    const double eager = time_us([&] { call(turn++); }, std::max(20, 2 * R));
    const double us    = time_graph_us(call, std::max(R, W <= G::kMaxSmallColumns ? 16 : 4));
    const bool glob = kind == InklingLayerKind::Global;
    // Bytes: every visible key's K and V once per row (decode / verify); FLOPs: 512 per (query, key, head).
    double keys_read = 0.0, flops = 0.0;
    for (int j = 0; j < W; ++j) {
        const double vis = glob ? first + j + 1.0 : std::min(first + j + 1.0, 1024.0);
        flops += 24.0 * vis * 512.0 * B;
    }
    keys_read       = (glob ? keys : std::min(keys, 1024 + W)) * static_cast<double>(B);
    const double gb = keys_read * 8 * 256 / 1e9;
    if (W <= G::kMaxSmallColumns) {
        std::printf("bench %-40s graph %8.1f us  %7.1f GB/s  %5.1f %% of %.0f GB/s   (eager %.1f us)\n", name, us,
                    gb / (us * 1e-6), 100.0 * gb / (us * 1e-6) / kBwGBs, kBwGBs, eager);
    } else {
        std::printf("bench %-40s graph %8.1f us  %7.1f TFLOP/s  %5.1f %% of %.0f (bf16)   (eager %.1f us)\n", name, us,
                    flops / (us * 1e-6) / 1e12, 100.0 * flops / (us * 1e-6) / 1e12 / kBf16TF, kBf16TF, eager);
    }
}

void bench_prologue(const char* name, InklingLayerKind kind, int W, int B, int first) {
    const Layer layer(kind, 12);
    const Rows rows = make_rows(W, B, std::vector<int>(B, first), {});
    const int N = W * B;
    BenchPlane plane(B, first + W + 64);
    std::vector<float> proj(static_cast<std::size_t>(N) * 8600, 0.5F), dyn(static_cast<std::size_t>(N) * 32, 0.1F);
    DeviceBuffer dproj = to_device_bf16(proj), ddyn = to_device_bf16(dyn);
    std::vector<std::uint16_t> zeros(static_cast<std::size_t>(kC) * 3 * B, 0);
    DeviceBuffer dstate = to_device(zeros);
    std::vector<int> sl(B);
    std::iota(sl.begin(), sl.end(), 0);
    DeviceBuffer dslots = to_device_i32(sl);
    ops::InklingProjections in;
    in.q = strided_bf16(dproj.p, G::kRowsQ, {128, 24}, N, 8600);
    in.r = strided_bf16(dproj.p, G::kRowsR, {16, 24}, N, 8600);
    in.k = strided_bf16(dproj.p, G::kRowsK, {1024}, N, 8600);
    in.v = strided_bf16(dproj.p, G::kRowsV, {1024}, N, 8600);
    if (kind == InklingLayerKind::Global) {
        in.k_dyn = Tensor(ddyn.p, DType::BF16, {4, 8, N});
        in.v_dyn = Tensor(ddyn.p, DType::BF16, {4, 8, N});
    }
    ops::InklingConvStateArgs conv;
    conv.states = Tensor(dstate.p, DType::BF16, {kC, 3, B});
    conv.slots  = Tensor(dslots.p, DType::I32, {B});
    DeviceBuffer dqo(static_cast<std::size_t>(N) * 3072 * 2), dro(static_cast<std::size_t>(N) * 384 * 2);
    ops::InklingPrologueOutputs qr{Tensor(dqo.p, DType::BF16, {128, 24, N}), Tensor(dro.p, DType::BF16, {16, 24, N})};
    const auto view = rows.view(first + W);
    const auto pv   = plane.view(B);
    const auto call   = [&](int) { ops::inkling_attention_prologue(in, layer.weights(), kind, view, conv, pv, qr, g_stream); };
    const double eager = time_us([&] { call(0); });
    const double us    = time_graph_us(call, 16);
    const double gb    = N * 20.5e3 / 1e9;
    std::printf("bench %-40s graph %8.1f us  %7.1f GB/s (~20.5 KB / column)   (eager %.1f us)\n", name, us,
                gb / (us * 1e-6), eager);
}

void bench_linear(const char* name, int k, int k_per, int columns) {
    const int n = k == 2304 ? 8600 : 2304;
    // R weight copies (> 256 MB), one per timed call in turn: the codes come from DRAM, not L2.
    const int R = columns <= 16 ? static_cast<int>(std::ceil(256e6 / (static_cast<double>(n) * k))) : 1;
    std::vector<std::unique_ptr<Fp8Weight>> ws;
    for (int r = 0; r < R; ++r) { ws.push_back(std::make_unique<Fp8Weight>(n, k, k_per, 1, 21 + r, false)); }
    std::vector<float> x(static_cast<std::size_t>(k) * columns, 0.5F);
    DeviceBuffer dx = to_device_bf16(x), dout(static_cast<std::size_t>(n) * columns * 2);
    Tensor xt(dx.p, DType::BF16, {k, columns});
    Tensor ot(dout.p, DType::BF16, {n, columns});
    const auto call = [&](int i) {
        const ops::InklingProjection pj{&xt, &ws[i % R]->w, 0, &ot};
        ops::inkling_attn_linear(std::span<const ops::InklingProjection>(&pj, 1), nullptr, g_stream);
    };
    int turn = 0;
    const double eager = time_us([&] { call(turn++); }, std::max(20, 2 * R));
    const double us    = time_graph_us(call, std::max(R, columns <= 16 ? 16 : 4));
    const double gb = static_cast<double>(n) * k / 1e9, tf = 2.0 * n * k * columns / 1e12;
    if (columns <= 16) {
        std::printf("bench %-40s graph %8.1f us  %7.1f GB/s  %5.1f %% of %.0f GB/s   (eager %.1f us)\n", name, us,
                    gb / (us * 1e-6), 100.0 * gb / (us * 1e-6) / kBwGBs, kBwGBs, eager);
    } else {
        std::printf("bench %-40s graph %8.1f us  %7.1f TFLOP/s  %5.1f %% of %.0f (bf16)   (eager %.1f us)\n", name, us,
                    tf / (us * 1e-6), 100.0 * tf / (us * 1e-6) / kBf16TF, kBf16TF, eager);
    }
}

// Decode attention (1 x 8k global, gated) followed by the wo projection (N = 1, 1x128) in ONE CUDA graph with PDL edges:
// the wo kernel streams its weights while attention runs. Reported per pair, next to the two kernels' separate times.
void bench_attention_wo(const char* name, InklingLayerKind kind, int first) {
    const Layer layer(kind, 13);
    const int keys = first + 1;
    const double row_bytes = (kind == InklingLayerKind::Global ? keys : std::min(keys, 1088)) * 2048.0;
    const int R = std::max(1, static_cast<int>(std::ceil(256e6 / (row_bytes + 2304.0 * 3072))));
    std::vector<Rows> rot;
    for (int r = 0; r < R; ++r) {
        rot.push_back(make_rows(1, 1, {first}, {}));
        rot.back().set_table_rows(r);
    }
    BenchPlane plane(R, keys + 64);
    Rng rng(5);
    std::vector<float> q, r;
    fill_qr(rng, rot[0], q, r);
    std::vector<float> qside(6552, 0.25F);
    DeviceBuffer dq = to_device_bf16(q), dr = to_device_bf16(r), dqs = to_device_bf16(qside);
    ops::InklingAttentionInputs in{Tensor(dq.p, DType::BF16, {128, 24, 1}), Tensor(dr.p, DType::BF16, {16, 24, 1}),
                                   strided_bf16(dqs.p, G::kRowsGate, {3072}, 1, 6552),
                                   strided_bf16(dqs.p, G::kRowsSink, {24}, 1, 6552),
                                   Tensor(layer.dtable.p, DType::BF16, {16, G::kRelTableRows})};
    std::vector<ops::InklingAttentionRows> views;
    for (const auto& rr : rot) { views.push_back(rr.view(keys)); }
    WorkspaceArena work(ops::inkling_rel_attention_workspace_bytes(kind, 1, 1, views[0].envelope) + 4096);
    DeviceBuffer dout(3072 * 2), dy(2304 * 2);
    Tensor out(dout.p, DType::BF16, {3072, 1}), y(dy.p, DType::BF16, {2304, 1});
    std::vector<std::unique_ptr<Fp8Weight>> ws;
    for (int i = 0; i < R; ++i) { ws.push_back(std::make_unique<Fp8Weight>(2304, 3072, 128, 1, 31 + i, false)); }
    const auto pv  = plane.view(R);
    const auto att = [&](int i) { ops::inkling_rel_attention(in, kind, views[i % R], pv, work, out, g_stream); };
    const auto wo  = [&](int i) {
        const ops::InklingProjection pj{&out, &ws[i % R]->w, 0, &y};
        ops::inkling_attn_linear(std::span<const ops::InklingProjection>(&pj, 1), nullptr, g_stream);
    };
    const double t_att  = time_graph_us(att, std::max(R, 16));
    const double t_wo   = time_graph_us(wo, std::max(R, 16));
    const double t_pair = time_graph_us([&](int i) { att(i); wo(i); }, std::max(R, 16));
    std::printf("bench %-40s graph pair %8.1f us  (attention alone %.1f + wo alone %.1f = %.1f)\n", name, t_pair, t_att,
                t_wo, t_att + t_wo);
}

// A/B of the decode kernel's latency knobs (L2 prefetch distance, PDL trigger at entry): same bits, different speed.
void run_bench_ab() {
    using K = InklingLayerKind;
    cuda_check(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking), "stream");
    setenv("SUROGATE_INKLING_ATTN_GROUPS", "p3", 1);
    // Static-metadata pre-wait L2 ask (decode / verify) off vs on; prefill L2 distance 0 vs 4.
    for (const char* meta : {"0", "1"}) {
        if (meta[0] == '1') { setenv("BENCH_META_STATIC", "1", 1); } else { unsetenv("BENCH_META_STATIC"); }
        const std::string tag = std::string(" [p3 meta=") + meta + "]";
        bench_attention(("decode global 1x8k" + tag).c_str(), K::Global, 1, 1, 8191);
        bench_attention(("decode global 1x64k" + tag).c_str(), K::Global, 1, 1, 65535);
        bench_attention(("decode global 1x262k" + tag).c_str(), K::Global, 1, 1, 262143);
        bench_attention(("decode local 1x8k" + tag).c_str(), K::Local, 1, 1, 8191);
        bench_attention(("decode global 8x8k" + tag).c_str(), K::Global, 1, 8, 8191);
        bench_attention(("verify W8 global 1x8k" + tag).c_str(), K::Global, 8, 1, 8184);
        bench_attention(("verify W8 local 1x8k" + tag).c_str(), K::Local, 8, 1, 8184);
    }
    unsetenv("BENCH_META_STATIC");
    for (const char* l2 : {"0", "4"}) {
        setenv("SUROGATE_INKLING_ATTN_L2AHEAD", l2, 1);
        const std::string tag = std::string(" [p3 l2=") + l2 + "]";
        bench_attention(("prefill global W8192 from 0" + tag).c_str(), K::Global, 8192, 1, 0);
        bench_attention(("prefill local W8192 past 8k" + tag).c_str(), K::Local, 8192, 1, 8192);
    }
    unsetenv("SUROGATE_INKLING_ATTN_L2AHEAD");
    unsetenv("SUROGATE_INKLING_ATTN_GROUPS");
    bench_linear("attn_linear qkvr narrow N=1 1x128", 2304, 128, 1);
    bench_linear("attn_linear qkvr narrow N=8 1x128", 2304, 128, 8);
    bench_linear("attn_linear wo narrow N=1 1x128", 3072, 128, 1);
    bench_linear("attn_linear wo narrow N=1 channel", 3072, 3072, 1);
    bench_linear("attn_linear wo narrow N=8 1x128", 3072, 128, 8);
}

// For nsys (per-kernel durations inside the graphs): a few decode probes and the wo projection, p1.
void run_bench_prof() {
    using K = InklingLayerKind;
    cuda_check(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking), "stream");
    setenv("SUROGATE_INKLING_ATTN_GROUPS", "p1", 1);
    bench_attention("prof decode global 1x8k", K::Global, 1, 1, 8191);
    bench_attention("prof decode global 1x262k", K::Global, 1, 1, 262143);
    bench_attention("prof decode local 1x8k", K::Local, 1, 1, 8191);
    bench_attention("prof verify W8 global 1x8k", K::Global, 8, 1, 8184);
    unsetenv("SUROGATE_INKLING_ATTN_GROUPS");
    bench_linear("prof wo narrow N=1 1x128", 3072, 128, 1);
}

void run_bench() {
    using K = InklingLayerKind;
    cuda_check(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking), "stream");
    // Key groups p1 / p2 / p3 (SUROGATE_INKLING_ATTN_GROUPS), same build; BENCH_GROUPS=p1|p2|p3 restricts.
    const char* only_groups = std::getenv("BENCH_GROUPS");
    for (const char* groups : {"p1", "p2", "p3"}) {
        if (only_groups != nullptr && std::string(only_groups) != groups) { continue; }
        setenv("SUROGATE_INKLING_ATTN_GROUPS", groups, 1);
        const auto name = [&](const char* what) { static std::string s; s = std::string(what) + " [" + groups + "]"; return s.c_str(); };
        bench_attention(name("attn decode global 1x8k"), K::Global, 1, 1, 8191);
        bench_attention(name("attn decode global 1x64k"), K::Global, 1, 1, 65535);
        bench_attention(name("attn decode global 1x262k"), K::Global, 1, 1, 262143);
        bench_attention(name("attn decode local 1x8k (window)"), K::Local, 1, 1, 8191);
        bench_attention(name("attn decode global 8x8k"), K::Global, 1, 8, 8191);
        bench_attention(name("attn decode local 8x8k"), K::Local, 1, 8, 8191);
        bench_attention(name("attn verify W8 global 1x8k"), K::Global, 8, 1, 8184);
        bench_attention(name("attn verify W8 global 1x64k"), K::Global, 8, 1, 65528);
        bench_attention(name("attn prefill global W8192 from 0"), K::Global, 8192, 1, 0);
        bench_attention(name("attn prefill global W8192 past 64k"), K::Global, 8192, 1, 65536);
        bench_attention(name("attn prefill local W8192 past 8k"), K::Local, 8192, 1, 8192);
        // Lane h's price-session cases (perf/price/PLAN-P1-P2.md).
        bench_attention(name("attn decode global 4x8k"), K::Global, 1, 4, 8191);
        bench_attention(name("attn decode global 4x64k"), K::Global, 1, 4, 65535);
        bench_attention(name("attn decode global 8x64k"), K::Global, 1, 8, 65535);
        bench_attention(name("attn decode local 4x8k"), K::Local, 1, 4, 8191);
        bench_attention(name("attn verify W8 global 1x262k"), K::Global, 8, 1, 262136);
        bench_attention(name("attn verify W8 global 8x8k"), K::Global, 8, 8, 8184);
        bench_attention(name("attn verify W8 local 1x8k"), K::Local, 8, 1, 8184);
        bench_attention(name("attn verify W8 local 8x8k"), K::Local, 8, 8, 8184);
        bench_attention(name("attn prefill global W8192 past 256k"), K::Global, 8192, 1, 262144);
    }
    // The 4-warp prefill kernel (KS = 1) for comparison, p1.
    setenv("SUROGATE_INKLING_ATTN_GROUPS", "p1", 1);
    setenv("SUROGATE_INKLING_PREFILL_KS", "1", 1);
    bench_attention("attn prefill global W8192 from 0 [p1 KS1]", K::Global, 8192, 1, 0);
    bench_attention("attn prefill global W8192 past 64k [p1 KS1]", K::Global, 8192, 1, 65536);
    bench_attention("attn prefill local W8192 past 8k [p1 KS1]", K::Local, 8192, 1, 8192);
    unsetenv("SUROGATE_INKLING_PREFILL_KS");
    bench_attention_wo("decode global 1x8k + wo N=1 1x128 [p1]", K::Global, 8191);
    bench_attention_wo("decode local 1x8k + wo N=1 1x128 [p1]", K::Local, 8191);
    unsetenv("SUROGATE_INKLING_ATTN_GROUPS");
    if (std::getenv("BENCH_ATTN_ONLY") != nullptr) { return; }
    bench_prologue("prologue local 1x1", K::Local, 1, 1, 8191);
    bench_prologue("prologue global 8x1", K::Global, 1, 8, 8191);
    bench_prologue("prologue local W8192", K::Local, 8192, 1, 0);
    bench_linear("attn_linear qkvr narrow N=1 channel", 2304, 2304, 1);
    bench_linear("attn_linear qkvr narrow N=1 1x128", 2304, 128, 1);
    bench_linear("attn_linear qkvr narrow N=8 1x128", 2304, 128, 8);
    bench_linear("attn_linear qkvr narrow N=16 1x128", 2304, 128, 16);
    bench_linear("attn_linear wo narrow N=1 1x128", 3072, 128, 1);
    bench_linear("attn_linear wo narrow N=1 channel", 3072, 3072, 1);
    bench_linear("attn_linear qkvr GEMM N=8192 1x128", 2304, 128, 8192);
    bench_linear("attn_linear qkvr GEMM N=8192 channel", 2304, 2304, 8192);
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "no CUDA device: skipped\n";
        return 77;
    }
    if (argc > 1 && std::string(argv[1]) == "bench") {
        try {
            if (argc > 2 && std::string(argv[2]) == "ab") {
                run_bench_ab();
                return 0;
            }
            if (argc > 2 && std::string(argv[2]) == "prof") {
                run_bench_prof();
                return 0;
            }
            run_bench();
        } catch (const std::exception& e) {
            std::cerr << "EXCEPTION " << e.what() << '\n';
            return 1;
        }
        return 0;
    }
    if (argc > 2 && std::string(argv[1]) == "capture") {
        try {
            run_capture(argv[2]);
        } catch (const std::exception& e) {
            std::cerr << "EXCEPTION " << e.what() << '\n';
            return 1;
        }
        std::cout << (g_failures == 0 ? "CAPTURE ALL PASS\n" : "CAPTURE FAILED\n");
        return g_failures == 0 ? 0 : 1;
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
            {"fused global decode B2 (tau > 1, 70k keys)", K::Global, 1, 2, {70000, 9000}, {}, M::InPlace, -1, true},
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
            {"attn global prefill W64 past 8000 (8k keys)", K::Global, 64, 1, {8000}, {}},
            {"attn global prefill W20 past 65000 (64k keys)", K::Global, 20, 1, {65000}, {}},
            {"attn local prefill W64 past 65000", K::Local, 64, 1, {65000}, {}},
            {"attn local decode forced 1 split", K::Local, 1, 2, {599, 4999}, {}, 1},
            {"attn global decode forced 1 split", K::Global, 1, 2, {36, 8999}, {}, 1},
        };
        for (const auto& c : att) {
            if (want(c.name)) { run_attention_case(c, seed); }
            seed += 17;
        }
        test_invariance(want);
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
