#include "targets/inkling2/impl/decision/heads.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace sinfer::targets::inkling2::detail::decision {

namespace {

/// torch nn.RMSNorm(eps=None) on FP32 input: eps = finfo(float32).eps.
constexpr float kTorchEps = std::numeric_limits<float>::epsilon();

const Mat& need(const HeadTensors& t, const std::string& name) {
    const auto it = t.find(name);
    if (it == t.end()) { throw std::invalid_argument("decision adapter: no " + name); }
    return it->second;
}

Linear linear(const HeadTensors& t, const std::string& prefix, bool bias) {
    Linear l;
    l.w = need(t, prefix + "/weight");
    if (bias) { l.b = need(t, prefix + "/bias"); }
    return l;
}

RmsNorm norm(const HeadTensors& t, const std::string& prefix) { return RmsNorm{need(t, prefix + "/weight")}; }

MultiheadAttention mha(const HeadTensors& t, const std::string& prefix, int heads) {
    MultiheadAttention m;
    m.in_proj  = Linear{need(t, prefix + "/in_proj_weight"), need(t, prefix + "/in_proj_bias")};
    m.out_proj = linear(t, prefix + "/out_proj", true);
    m.heads    = heads;
    return m;
}

Mat rows_of(std::int64_t rows, std::int64_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.v.assign(static_cast<std::size_t>(rows * cols), 0.0F);
    return m;
}

Mat add(Mat a, const Mat& b) {
    if (a.v.size() != b.v.size()) { throw std::logic_error("decision heads: shape mismatch in add"); }
    for (std::size_t i = 0; i < a.v.size(); ++i) { a.v[i] += b.v[i]; }
    return a;
}

void silu_inplace(Mat& m) {
    for (float& x : m.v) { x = x / (1.0F + std::exp(-x)); }
}

/// Row mean over axis 0: [n, d] -> [1, d] (torch .mean(dim=0)).
Mat mean_rows(const Mat& m) {
    Mat out = rows_of(1, m.cols);
    for (std::int64_t r = 0; r < m.rows; ++r) {
        for (std::int64_t c = 0; c < m.cols; ++c) { out.v[c] += m.row(r)[c]; }
    }
    const float n = static_cast<float>(m.rows);
    for (float& x : out.v) { x /= n; }
    return out;
}

std::vector<float> softmax(std::span<const float> x) {
    std::vector<float> p(x.begin(), x.end());
    const float mx = *std::max_element(p.begin(), p.end());
    float sum = 0.0F;
    for (float& v : p) {
        v = std::exp(v - mx);
        sum += v;
    }
    for (float& v : p) { v /= sum; }
    return p;
}

/// max_p, normalised entropy, top-2 margin of softmax(logits) (the halting / escalation scalars).
struct Spread {
    float max_p, entropy, margin;
};
Spread spread(std::span<const float> logits) {
    const std::vector<float> p = softmax(logits);
    std::vector<float> sorted = p;
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    const std::size_t k = p.size();
    Spread s{};
    s.max_p  = sorted[0];
    s.margin = sorted[0] - (k > 1 ? sorted[1] : 0.0F);
    float ent = 0.0F;
    for (float v : p) { ent += v * std::log(std::max(v, 1e-8F)); }
    s.entropy = -ent / std::max(std::log(static_cast<float>(std::max<std::size_t>(k, 2))), 1e-6F);
    return s;
}

float l2(std::span<const float> x) {
    double s = 0.0;
    for (float v : x) { s += static_cast<double>(v) * v; }
    return static_cast<float>(std::sqrt(s));
}

Mat concat_row(std::initializer_list<std::span<const float>> parts) {
    std::size_t n = 0;
    for (auto p : parts) { n += p.size(); }
    Mat out = rows_of(1, static_cast<std::int64_t>(n));
    std::size_t at = 0;
    for (auto p : parts) {
        std::copy(p.begin(), p.end(), out.v.begin() + static_cast<std::ptrdiff_t>(at));
        at += p.size();
    }
    return out;
}

std::span<const float> span_of(const Mat& m) { return {m.v.data(), m.v.size()}; }

} // namespace

Mat Linear::operator()(const Mat& x) const {
    const std::int64_t out_f = w.rows, in_f = w.cols;
    if (x.cols != in_f) { throw std::logic_error("decision heads: linear input width"); }
    Mat y = rows_of(x.rows, out_f);
    for (std::int64_t n = 0; n < x.rows; ++n) {
        const float* xr = x.row(n);
        float* yr       = y.row(n);
        for (std::int64_t o = 0; o < out_f; ++o) {
            const float* wr = w.row(o);
            float acc       = 0.0F;
            for (std::int64_t i = 0; i < in_f; ++i) { acc += wr[i] * xr[i]; }
            yr[o] = acc + (b ? b->v[static_cast<std::size_t>(o)] : 0.0F);
        }
    }
    return y;
}

Mat RmsNorm::operator()(const Mat& x) const {
    Mat y = x;
    for (std::int64_t r = 0; r < x.rows; ++r) {
        const float* xr = x.row(r);
        double ss       = 0.0;
        for (std::int64_t c = 0; c < x.cols; ++c) { ss += static_cast<double>(xr[c]) * xr[c]; }
        const float inv = 1.0F / std::sqrt(static_cast<float>(ss / static_cast<double>(x.cols)) + kTorchEps);
        float* yr       = y.row(r);
        for (std::int64_t c = 0; c < x.cols; ++c) { yr[c] = xr[c] * inv * w.v[static_cast<std::size_t>(c)]; }
    }
    return y;
}

Mat MultiheadAttention::operator()(const Mat& q_in, const Mat& k_in, const Mat& v_in) const {
    const std::int64_t e = out_proj.w.rows;
    const std::int64_t hd = e / heads;
    // packed in_proj: rows [0, E) q, [E, 2E) k, [2E, 3E) v
    const auto project = [&](const Mat& x, int part) {
        Linear l;
        l.w = rows_of(e, e);
        std::copy(in_proj.w.v.begin() + part * e * e, in_proj.w.v.begin() + (part + 1) * e * e, l.w.v.begin());
        Mat b = rows_of(1, e);
        std::copy(in_proj.b->v.begin() + part * e, in_proj.b->v.begin() + (part + 1) * e, b.v.begin());
        l.b = b;
        return l(x);
    };
    const Mat q = project(q_in, 0), k = project(k_in, 1), v = project(v_in, 2);
    Mat ctx = rows_of(q.rows, e);
    const float scale = 1.0F / std::sqrt(static_cast<float>(hd));
    std::vector<float> score(static_cast<std::size_t>(k.rows));
    for (int h = 0; h < heads; ++h) {
        for (std::int64_t i = 0; i < q.rows; ++i) {
            for (std::int64_t j = 0; j < k.rows; ++j) {
                float s = 0.0F;
                for (std::int64_t d = 0; d < hd; ++d) { s += q.row(i)[h * hd + d] * k.row(j)[h * hd + d]; }
                score[static_cast<std::size_t>(j)] = s * scale;
            }
            const std::vector<float> p = softmax(score);
            for (std::int64_t d = 0; d < hd; ++d) {
                float acc = 0.0F;
                for (std::int64_t j = 0; j < k.rows; ++j) { acc += p[static_cast<std::size_t>(j)] * v.row(j)[h * hd + d]; }
                ctx.row(i)[h * hd + d] = acc;
            }
        }
    }
    return out_proj(ctx);
}

Heads::Heads(const HeadTensors& t, int pooler_max_chunks, int max_loops) : max_loops_(max_loops) {
    const std::string J = "heads/joint_head/";
    j_proj_q_   = linear(t, J + "proj_q", true);
    j_proj_ctx_ = linear(t, J + "proj_ctx", true);
    j_type_emb_ = need(t, J + "type_emb/weight");
    j_norm1_q_  = norm(t, J + "norm1_q");
    j_norm1_kv_ = norm(t, J + "norm1_kv");
    j_cross_    = mha(t, J + "cross_attn", 8);
    j_norm2_    = norm(t, J + "norm2");
    j_self_     = mha(t, J + "self_attn", 8);
    j_norm3_    = norm(t, J + "norm3");
    j_ffn_up_   = linear(t, J + "ffn_up", true);
    j_ffn_down_ = linear(t, J + "ffn_down", true);
    j_norm_out_ = norm(t, J + "norm_out");
    j_ptr_q_    = linear(t, J + "ptr_q", true);
    j_ptr_k_    = linear(t, J + "ptr_k", true);
    j_opt_bias_ = linear(t, J + "opt_bias", true);

    const std::string G = "heads/core_gate/";
    g_norm_ = norm(t, G + "norm");
    g_q_    = linear(t, G + "q", true);
    g_k_    = linear(t, G + "k", true);

    const std::string H = "heads/halt_head/";
    h_norm_d_     = norm(t, H + "norm_d");
    h_norm_o_     = norm(t, H + "norm_o");
    h_norm_delta_ = norm(t, H + "norm_delta");
    h_proj_d_     = linear(t, H + "proj_d", true);
    h_proj_o_     = linear(t, H + "proj_o", true);
    h_proj_delta_ = linear(t, H + "proj_delta", true);
    h_loop_emb_   = need(t, H + "loop_emb/weight");
    h_mlp0_       = linear(t, H + "mlp/0", true);
    h_mlp3_       = linear(t, H + "mlp/3", true);

    has_escalation_ = t.count("heads/escalation_head/proj_d/weight") != 0;
    if (has_escalation_) {
        const std::string E = "heads/escalation_head/";
        e_norm_d_ = norm(t, E + "norm_d");
        e_norm_o_ = norm(t, E + "norm_o");
        e_proj_d_ = linear(t, E + "proj_d", true);
        e_proj_o_ = linear(t, E + "proj_o", true);
        e_mlp0_   = linear(t, E + "mlp/0", true);
        e_mlp3_   = linear(t, E + "mlp/3", true);
    }
    has_aux_ = t.count("heads/aux_head/proj_d/weight") != 0;
    if (has_aux_) {
        const std::string A = "heads/aux_head/";
        a_norm_d_ = norm(t, A + "norm_d");
        a_norm_o_ = norm(t, A + "norm_o");
        a_proj_d_ = linear(t, A + "proj_d", true);
        a_proj_o_ = linear(t, A + "proj_o", true);
        a_mlp0_   = linear(t, A + "mlp/0", true);
        a_mlp3_   = linear(t, A + "mlp/3", true);
    }
    has_pooler_ = t.count("heads/evidence_pooler/norm/weight") != 0;
    if (has_pooler_) {
        const std::string P = "heads/evidence_pooler/";
        p_norm_        = norm(t, P + "norm");
        p_up_          = linear(t, P + "linear_up", false);
        p_down_        = linear(t, P + "linear_down", false);
        pooler_chunks_ = pooler_max_chunks;
    }
}

std::vector<float> Heads::core_gate(const Mat& h_decide, const Mat& h_opts) const {
    const Mat q = g_q_(g_norm_(h_decide));
    const Mat k = g_k_(g_norm_(h_opts));
    const float scale = 1.0F / std::sqrt(static_cast<float>(q.cols));
    std::vector<float> out(static_cast<std::size_t>(k.rows));
    for (std::int64_t i = 0; i < k.rows; ++i) {
        float s = 0.0F;
        for (std::int64_t c = 0; c < k.cols; ++c) { s += k.row(i)[c] * q.v[static_cast<std::size_t>(c)]; }
        out[static_cast<std::size_t>(i)] = s * scale;
    }
    return out;
}

float Heads::halt(const Mat& h_decide, const Mat& h_opts, const Mat* prev, std::span<const float> gate_logits,
                  int loop_index) const {
    const Spread sp = spread(gate_logits);
    Mat delta       = rows_of(1, h_decide.cols);
    float rel_drift = 0.0F;
    if (prev != nullptr) {
        for (std::int64_t c = 0; c < h_decide.cols; ++c) { delta.v[c] = h_decide.v[c] - prev->v[c]; }
        rel_drift = l2(span_of(delta)) / std::max(l2(span_of(h_decide)), 1e-6F);
    }
    Mat feat = add(add(h_proj_d_(h_norm_d_(h_decide)), h_proj_o_(mean_rows(h_norm_o_(h_opts)))),
                   h_proj_delta_(h_norm_delta_(delta)));
    const float k = static_cast<float>(std::max<std::size_t>(gate_logits.size(), 2));
    const float scalars[6] = {sp.max_p, sp.entropy, sp.margin, std::min(rel_drift, 5.0F), std::log(k) / 3.0F,
                              static_cast<float>(loop_index) / 4.0F};
    const std::int64_t li = std::min(loop_index, max_loops_ - 1);
    const std::span<const float> emb(h_loop_emb_.row(li), static_cast<std::size_t>(h_loop_emb_.cols));
    Mat x = concat_row({span_of(feat), std::span<const float>(scalars, 6), emb});
    Mat hidden = h_mlp0_(x);
    silu_inplace(hidden);
    return h_mlp3_(hidden).v[0];
}

Mat Heads::pool(const Mat& h_state) const {
    if (!has_pooler_) { throw std::logic_error("decision heads: no evidence pooler"); }
    const std::int64_t L = h_state.rows, C = pooler_chunks_;
    Mat pooled;
    if (L <= C) {
        pooled = h_state;
    } else {
        pooled = rows_of(C, h_state.cols);
        std::vector<float> counts(static_cast<std::size_t>(C), 0.0F);
        for (std::int64_t i = 0; i < L; ++i) {
            const std::int64_t chunk = (i * C) / L;
            counts[static_cast<std::size_t>(chunk)] += 1.0F;
            for (std::int64_t c = 0; c < h_state.cols; ++c) { pooled.row(chunk)[c] += h_state.row(i)[c]; }
        }
        for (std::int64_t r = 0; r < C; ++r) {
            const float n = std::max(counts[static_cast<std::size_t>(r)], 1.0F);
            for (std::int64_t c = 0; c < pooled.cols; ++c) { pooled.row(r)[c] /= n; }
        }
    }
    Mat up = p_up_(p_norm_(pooled));
    silu_inplace(up);
    return add(pooled, p_down_(up));
}

std::vector<float> Heads::joint(const Mat& h_decide, const Mat& h_opts, const Mat& h_ctx) const {
    const std::int64_t K = h_opts.rows, D = h_decide.cols;
    Mat q_seq = rows_of(K + 1, D);
    std::copy(h_decide.v.begin(), h_decide.v.end(), q_seq.v.begin());
    std::copy(h_opts.v.begin(), h_opts.v.end(), q_seq.v.begin() + D);
    Mat x = j_proj_q_(q_seq);
    for (std::int64_t r = 0; r < x.rows; ++r) {
        const float* te = j_type_emb_.row(r == 0 ? 0 : 1);
        for (std::int64_t c = 0; c < x.cols; ++c) { x.row(r)[c] += te[c]; }
    }
    const Mat ctx = j_proj_ctx_(h_ctx);
    const Mat ctx_n = j_norm1_kv_(ctx);
    x = add(x, j_cross_(j_norm1_q_(x), ctx_n, ctx_n));
    const Mat s_in = j_norm2_(x);
    x = add(x, j_self_(s_in, s_in, s_in));
    Mat up = j_ffn_up_(j_norm3_(x));
    silu_inplace(up);
    x = add(x, j_ffn_down_(up));
    x = j_norm_out_(x);
    Mat d_vec = rows_of(1, x.cols);
    std::copy(x.row(0), x.row(0) + x.cols, d_vec.v.begin());
    Mat o_vecs = rows_of(K, x.cols);
    std::copy(x.v.begin() + x.cols, x.v.end(), o_vecs.v.begin());
    const Mat pq = j_ptr_q_(d_vec), pk = j_ptr_k_(o_vecs), bias = j_opt_bias_(o_vecs);
    const float scale = 1.0F / std::sqrt(static_cast<float>(x.cols));
    std::vector<float> out(static_cast<std::size_t>(K));
    for (std::int64_t i = 0; i < K; ++i) {
        float s = 0.0F;
        for (std::int64_t c = 0; c < pk.cols; ++c) { s += pk.row(i)[c] * pq.v[static_cast<std::size_t>(c)]; }
        out[static_cast<std::size_t>(i)] = s * scale + bias.v[static_cast<std::size_t>(i)];
    }
    return out;
}

float Heads::escalation(const Mat& h_decide, const Mat& h_opts, float total_state_drift,
                        std::span<const float> final_logits, float max_halt_prob) const {
    if (!has_escalation_) { throw std::logic_error("decision heads: no escalation head"); }
    const Spread sp = spread(final_logits);
    const float scalars[4] = {std::clamp(total_state_drift, 0.0F, 10.0F), sp.entropy, sp.margin,
                              std::clamp(max_halt_prob, 0.0F, 1.0F)};
    Mat feat = add(e_proj_d_(e_norm_d_(h_decide)), e_proj_o_(mean_rows(e_norm_o_(h_opts))));
    Mat hidden = e_mlp0_(concat_row({span_of(feat), std::span<const float>(scalars, 4)}));
    silu_inplace(hidden);
    return e_mlp3_(hidden).v[0];
}

std::vector<float> Heads::aux(const Mat& h_decide, const Mat& h_opts) const {
    if (!has_aux_) { return {}; }
    Mat feat   = add(a_proj_d_(a_norm_d_(h_decide)), a_proj_o_(mean_rows(a_norm_o_(h_opts))));
    Mat hidden = a_mlp0_(feat);
    silu_inplace(hidden);
    return a_mlp3_(hidden).v;
}

} // namespace sinfer::targets::inkling2::detail::decision
