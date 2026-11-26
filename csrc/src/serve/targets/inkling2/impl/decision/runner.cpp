#include "targets/inkling2/impl/decision/runner.h"

#include "targets/inkling2/impl/decision/kernels.h"
#include "targets/inkling2/impl/variant.h"
#include "targets/inkling2/impl/visits.h"

#include "api/ops/inkling_mix.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace sinfer::targets::inkling2::detail::decision {

namespace {

using C                        = TextConfig;
constexpr int D                = C::hidden;
constexpr std::int64_t kColumn = C::residual;
constexpr int kCore            = C::core_end - C::core_start;
constexpr float kFp32Eps       = std::numeric_limits<float>::epsilon();

std::uint16_t f32_to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffU) > 0x7f800000U) { return static_cast<std::uint16_t>((u >> 16) | 0x40U); } // NaN
    u += 0x7fffU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

float bf16_to_f32(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

float sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

std::vector<double> softmax_t(const std::vector<float>& x, double t) {
    std::vector<double> p(x.size());
    double mx = -std::numeric_limits<double>::infinity();
    for (float v : x) { mx = std::max(mx, static_cast<double>(v)); }
    double s = 0;
    for (std::size_t i = 0; i < x.size(); ++i) { s += p[i] = std::exp((static_cast<double>(x[i]) - mx) / t); }
    for (double& v : p) { v /= s; }
    return p;
}

Mat rows_mat(const std::vector<float>& host, int first, int count) {
    Mat m;
    m.rows = count;
    m.cols = D;
    m.v.assign(host.begin() + static_cast<std::ptrdiff_t>(first) * D,
               host.begin() + static_cast<std::ptrdiff_t>(first + count) * D);
    return m;
}

float l2(const Mat& a) {
    double s = 0;
    for (float v : a.v) { s += static_cast<double>(v) * v; }
    return static_cast<float>(std::sqrt(s));
}

/// Bump allocator over one device buffer (256-byte aligned pieces).
struct Plan {
    std::size_t bytes = 0;
    std::size_t take(std::size_t n) {
        const std::size_t at = bytes;
        bytes += (n + 255U) & ~std::size_t{255};
        return at;
    }
};

} // namespace

Runner::Runner(const LoadedModelData& base, const std::filesystem::path& adapter_dir, std::int32_t max_tokens)
    : base_(base), file_(adapter_dir), max_tokens_(max_tokens) {
    const auto& meta = file_.meta();
    if (meta.at("base").at("hidden").get<int>() != D || meta.at("base").at("layers").get<int>() != C::physical_layers) {
        throw std::invalid_argument("decision adapter: not an Inkling-2-Mini adapter");
    }
    max_loops_ = meta.at("max_loops").get<int>();
    if (max_loops_ < 1 || max_loops_ > C::max_passes) { throw std::invalid_argument("decision adapter: max_loops out of range"); }
    for (int p = 1; p <= max_loops_; ++p) {
        exit_temps_.push_back(meta.at("exit_temps").value(std::to_string(p), 1.0));
    }
    heads_ = std::make_unique<Heads>(file_.heads(), meta.at("heads").at("evidence_pooler_max_chunks").get<int>(),
                                     max_loops_);
    expert_ff_ = meta.at("scratchpad").at("d_ff").get<int>();
    const int d_ada = meta.at("scratchpad").at("d_ada").get<int>();

    // ---- the adapter's device weights: one buffer, laid out first, filled from host --------------------------------
    Plan plan;
    struct Upload {
        std::size_t at;
        std::vector<std::byte> bytes;
    };
    std::vector<Upload> uploads;
    const auto put = [&](std::vector<std::byte> bytes) {
        const std::size_t at = plan.take(bytes.size());
        uploads.push_back({at, std::move(bytes)});
        return at;
    };
    const auto as_bf16 = [&](const std::string& name) { // a BF16 object as stored, or an F32 one rounded to BF16
        const AdapterObject& o = file_.object(name);
        if (o.dtype == "BF16") { return file_.read(name); }
        const Mat m = file_.read_f32(name);
        std::vector<std::byte> out(m.v.size() * 2);
        auto* p = reinterpret_cast<std::uint16_t*>(out.data());
        for (std::size_t i = 0; i < m.v.size(); ++i) { p[i] = f32_to_bf16(m.v[i]); }
        return out;
    };
    const auto as_f32 = [&](const std::string& name) {
        const AdapterObject& o = file_.object(name);
        if (o.dtype != "F32") { throw std::invalid_argument("decision adapter: " + name + " is not F32"); }
        return file_.read(name);
    };
    struct LayerAt {
        std::size_t qkvr = 0, output = 0, router = 0, down = 0, gain_forget = 0;
    };
    std::vector<LayerAt> at(C::physical_layers);
    for (int L = 0; L < C::physical_layers; ++L) {
        const std::string p = "text/layers/" + std::to_string(L) + "/";
        at[L].qkvr   = put(as_bf16(p + "attn/qkvr"));
        at[L].output = put(as_bf16(p + "attn/output"));
        at[L].router = put(as_bf16(p + "moe/router"));
        if (L >= C::core_start && L < C::core_end) {
            at[L].down        = put(as_bf16(p + "loop/down"));
            at[L].gain_forget = put(as_bf16(p + "loop/gain_forget"));
        }
    }
    std::vector<std::size_t> attn_q(C::physical_layers, 0), mlp_q(C::physical_layers, 0);
    std::vector<int> attn_rows(C::physical_layers, 0);
    for (int L = 1; L < C::physical_layers; ++L) {
        const std::string n = "text/attn_res/attn_query/" + std::to_string(L);
        attn_rows[L]        = static_cast<int>(file_.object(n).shape.at(0));
        attn_q[L]           = put(as_f32(n));
    }
    for (int L = 0; L < C::physical_layers; ++L) { mlp_q[L] = put(as_f32("text/attn_res/mlp_query/" + std::to_string(L))); }
    const std::size_t out_q = put(as_f32("text/attn_res/output_query"));
    const std::size_t cross = put(as_f32("text/loop/cross_pass_queries"));
    const std::size_t cross_kv = put(as_f32("text/loop/cross_pass_kv_queries"));
    struct ExpertAt {
        std::size_t norm = 0, gate_up = 0, down = 0;
        std::vector<std::size_t> ada;
    };
    std::vector<ExpertAt> eat(kCore);
    for (int j = 0; j < kCore; ++j) {
        const std::string p = "scratchpad/" + std::to_string(j) + "/";
        eat[j].norm = put(as_bf16(p + "norm"));
        std::vector<std::byte> gate = as_bf16(p + "gate_proj"), up = as_bf16(p + "up_proj");
        gate.insert(gate.end(), up.begin(), up.end());
        eat[j].gate_up = put(std::move(gate));
        eat[j].down    = put(as_bf16(p + "down_proj"));
        // AdaLN-Zero(k): (scale, shift, gate) = bf16(ada_proj(loop_emb[k])) for every pass k (fp32 accumulation)
        const std::vector<std::byte> emb_raw = as_bf16(p + "loop_emb"), w_raw = as_bf16(p + "ada_proj_weight"),
                                     b_raw = as_bf16(p + "ada_proj_bias");
        const auto* emb = reinterpret_cast<const std::uint16_t*>(emb_raw.data());
        const auto* w   = reinterpret_cast<const std::uint16_t*>(w_raw.data());
        const auto* b   = reinterpret_cast<const std::uint16_t*>(b_raw.data());
        const int loops = static_cast<int>(emb_raw.size() / 2 / static_cast<std::size_t>(d_ada));
        for (int k = 0; k < max_loops_; ++k) {
            const int ki = std::min(k, loops - 1);
            std::vector<std::byte> out(static_cast<std::size_t>(3) * D * 2);
            auto* o = reinterpret_cast<std::uint16_t*>(out.data());
            for (int r = 0; r < 3 * D; ++r) {
                float acc = 0.0F;
                for (int i = 0; i < d_ada; ++i) {
                    acc += bf16_to_f32(w[static_cast<std::size_t>(r) * d_ada + i]) * bf16_to_f32(emb[ki * d_ada + i]);
                }
                o[r] = f32_to_bf16(acc + bf16_to_f32(b[r]));
            }
            eat[j].ada.push_back(put(std::move(out)));
        }
    }
    weights_ = DeviceBuffer(plan.bytes);
    for (const Upload& u : uploads) {
        CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(weights_.p) + u.at, u.bytes.data(), u.bytes.size(),
                              cudaMemcpyHostToDevice));
    }
    std::byte* wb = static_cast<std::byte*>(weights_.p);

    // ---- the layers: the base's weights with the replacements bound over them --------------------------------------
    layers_ = base_.weights.layers;
    for (int L = 0; L < C::physical_layers; ++L) {
        load::InklingLayerWeights& l = layers_.at(static_cast<std::size_t>(L));
        if (l.router.qdata == nullptr) { throw std::logic_error("decision adapter: a base layer is not resident"); }
        const auto bf16_weight = [&](std::size_t offset, std::int32_t n, std::int32_t k) {
            Weight w;
            w.qtype    = QType::BF16_CTRL;
            w.qdata    = wb + offset;
            w.payload  = w.qdata;
            w.payload_bytes = static_cast<std::uint64_t>(n) * k * 2;
            w.n        = n;
            w.k        = k;
            w.shape[0] = k;
            w.shape[1] = n;
            w.ndim     = 2;
            return w;
        };
        l.qkvr          = bf16_weight(at[L].qkvr, C::qkvr_rows, D);
        l.output        = bf16_weight(at[L].output, D, C::attn_width);
        l.fp8_attention = false;
        l.router        = bf16_weight(at[L].router, C::experts, D);
        if (L >= C::core_start && L < C::core_end) {
            l.loop_down        = Tensor(wb + at[L].down, DType::BF16, {2 * D, C::gate_rank, 4});
            l.loop_gain_forget = Tensor(wb + at[L].gain_forget, DType::BF16, {C::gate_rank, 2 * D, 4});
        }
    }
    globals_ = base_.weights.globals;
    for (int L = 1; L < C::physical_layers; ++L) {
        globals_.attn_query.at(static_cast<std::size_t>(L)) = Tensor(wb + attn_q[L], DType::FP32, {D, attn_rows[L]});
    }
    for (int L = 0; L < C::physical_layers; ++L) {
        globals_.mlp_query.at(static_cast<std::size_t>(L)) = Tensor(wb + mlp_q[L], DType::FP32, {D, 1});
    }
    globals_.output_query          = Tensor(wb + out_q, DType::FP32, {D, 1});
    globals_.cross_pass_queries    = Tensor(wb + cross, DType::FP32, {D, 2 * kCore});
    globals_.cross_pass_kv_queries = Tensor(wb + cross_kv, DType::FP32, {D, 2 * kCore});
    derive_attn_res_queries(globals_, queries_);
    queries_.rel_table = base_.queries.rel_table; // rel_proj is the base's (checked by the converter)
    output_query_      = queries_.output;
    for (int j = 0; j < kCore; ++j) {
        SharedExpert e;
        e.norm    = wb + eat[j].norm;
        e.gate_up = wb + eat[j].gate_up;
        e.down    = wb + eat[j].down;
        for (std::size_t a : eat[j].ada) { e.ada.push_back(wb + a); }
        experts_.push_back(std::move(e));
    }

    // ---- one question's state --------------------------------------------------------------------------------------
    const std::int32_t T = max_tokens_;
    const std::int32_t pages = (T + kPagedKVPageSize - 1) / kPagedKVPageSize;
    Plan s;
    const std::size_t x_at = s.take(static_cast<std::size_t>(kColumn) * T * 2);
    const std::size_t ids_at = s.take(static_cast<std::size_t>(T) * 4);
    const std::size_t pos_at = s.take(static_cast<std::size_t>(T) * 4);
    const std::size_t meta_at = s.take(8 * 4);
    const std::size_t kv_bytes = static_cast<std::size_t>(C::head_dim) * kPagedKVPageSize * C::kv_heads * pages * 2;
    const std::size_t k_at = s.take(kv_bytes), v_at = s.take(kv_bytes);
    const std::size_t table_at = s.take(static_cast<std::size_t>(pages) * 4);
    const std::size_t conv_at = s.take(static_cast<std::size_t>(ops::InklingMixGeometry::kConvChannels) * 3 * 2);
    const std::size_t state_at = s.take(Variant::visit_target_state_bytes());
    const std::size_t hidden_at = s.take(static_cast<std::size_t>(D) * T * 2);
    const std::size_t spx_at = s.take(static_cast<std::size_t>(D) * T * 2);
    const std::size_t spgu_at = s.take(static_cast<std::size_t>(2 * expert_ff_) * T * 2);
    const std::size_t spact_at = s.take(static_cast<std::size_t>(expert_ff_) * T * 2);
    const std::size_t spout_at = s.take(static_cast<std::size_t>(D) * T * 2);
    scratch_ = DeviceBuffer(s.bytes);
    CUDA_CHECK(cudaMemset(scratch_.p, 0, s.bytes));
    std::byte* sb = static_cast<std::byte*>(scratch_.p);
    x_         = Tensor(sb + x_at, DType::BF16, {static_cast<std::int32_t>(kColumn), T});
    ids_       = Tensor(sb + ids_at, DType::I32, {T});
    positions_ = Tensor(sb + pos_at, DType::I32, {T, 1});
    meta_      = Tensor(sb + meta_at, DType::I32, {8});
    conv_      = Tensor(sb + conv_at, DType::BF16, {ops::InklingMixGeometry::kConvChannels, 3, 1});
    hidden_    = Tensor(sb + hidden_at, DType::BF16, {D, T});
    target_state_ = sb + state_at;
    sp_x_ = sb + spx_at;
    sp_gate_up_ = sb + spgu_at;
    sp_act_ = sb + spact_at;
    sp_out_ = sb + spout_at;
    std::vector<std::int32_t> iota(static_cast<std::size_t>(std::max(T, pages)));
    std::iota(iota.begin(), iota.end(), 0);
    CUDA_CHECK(cudaMemcpy(positions_.data, iota.data(), static_cast<std::size_t>(T) * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sb + table_at, iota.data(), static_cast<std::size_t>(pages) * 4, cudaMemcpyHostToDevice));
    kv_.k_pages      = Tensor(sb + k_at, DType::BF16, {C::head_dim, kPagedKVPageSize, C::kv_heads, pages});
    kv_.v_pages      = Tensor(sb + v_at, DType::BF16, {C::head_dim, kPagedKVPageSize, C::kv_heads, pages});
    kv_.block_tables = Tensor(sb + table_at, DType::I32, {pages, 1});
    kv_.head_dim     = C::head_dim;
    kv_.num_kv_heads = C::kv_heads;
    kv_.dtype        = DType::BF16;

    workspace_bytes_ = Variant::visit_workspace_capacity_bytes(family::TextGeometry{}, family::TextPhase::Prefill, 0, 0,
                                                               T, static_cast<std::uint32_t>(T), false) +
                       ops::inkling_gemm_inv_workspace_bytes(2 * expert_ff_, D, T, 1) +
                       ops::inkling_gemm_inv_workspace_bytes(D, expert_ff_, T, 1) + (std::size_t{1} << 20);
    workspace_ = DeviceBuffer(workspace_bytes_);
}

Runner::~Runner() = default;

double Runner::mode_threshold(const std::string& mode) const {
    const auto& conformal = file_.meta().at("conformal");
    if (conformal.is_object() && conformal.contains(mode)) {
        const auto& v = conformal.at(mode);
        if (v.is_object() && v.contains("threshold")) { return v.at("threshold").get<double>(); }
        if (v.is_number()) { return v.get<double>(); }
    }
    throw std::invalid_argument("decision adapter: no calibrated mode " + mode);
}

std::vector<std::string> Runner::special_tokens() const {
    return file_.meta().at("special_tokens").get<std::vector<std::string>>();
}

void Runner::shared_expert(int core_index, int pass, int slot, std::int32_t T, cudaStream_t s) {
    const SharedExpert& e = experts_.at(static_cast<std::size_t>(core_index));
    const auto* ada = static_cast<const std::uint16_t*>(e.ada.at(static_cast<std::size_t>(std::min(pass, max_loops_ - 1))));
    void* h = static_cast<std::uint16_t*>(x_.data) + static_cast<std::int64_t>(slot) * D;
    shared_expert_modulate(h, kColumn, e.norm, ada, ada + D, sp_x_, D, T, kFp32Eps, s);
    {
        const std::size_t bytes = ops::inkling_gemm_inv_workspace_bytes(2 * expert_ff_, D, T, 1);
        WorkspaceArena work(DeviceSpan{workspace_.p, workspace_bytes_});
        void* scratch = bytes > 0 ? work.alloc(DType::FP32, {static_cast<std::int32_t>((bytes + 3) / 4)}).data : nullptr;
        ops::inkling_gemm_inv(e.gate_up, 2 * expert_ff_, D, sp_x_, T, sp_gate_up_, DType::BF16, 2 * expert_ff_, {},
                              scratch, bytes, s);
    }
    shared_expert_swiglu(sp_gate_up_, sp_act_, expert_ff_, T, s);
    {
        const std::size_t bytes = ops::inkling_gemm_inv_workspace_bytes(D, expert_ff_, T, 1);
        WorkspaceArena work(DeviceSpan{workspace_.p, workspace_bytes_});
        void* scratch = bytes > 0 ? work.alloc(DType::FP32, {static_cast<std::int32_t>((bytes + 3) / 4)}).data : nullptr;
        ops::inkling_gemm_inv(e.down, D, expert_ff_, sp_act_, T, sp_out_, DType::BF16, D, {}, scratch, bytes, s);
    }
    shared_expert_residual(h, kColumn, ada + 2 * D, sp_out_, D, T, s);
}

void Runner::read_rows(const void* plane, std::int64_t column_stride, std::int32_t T, const std::vector<int>& rows,
                       std::vector<float>& out, cudaStream_t s) {
    // every column [0, T) of the plane (D BF16 rows each), then the requested ones as FP32, in order
    host_rows_.resize(static_cast<std::size_t>(D) * T);
    CUDA_CHECK(cudaMemcpy2DAsync(host_rows_.data(), static_cast<std::size_t>(D) * 2, plane,
                                 static_cast<std::size_t>(column_stride) * 2, static_cast<std::size_t>(D) * 2,
                                 static_cast<std::size_t>(T), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    out.resize(rows.size() * D);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const std::uint16_t* src = host_rows_.data() + static_cast<std::size_t>(rows[r]) * D;
        for (int c = 0; c < D; ++c) { out[r * D + c] = bf16_to_f32(src[c]); }
    }
}

Answer Runner::run(const Question& q, const Policy& policy, cudaStream_t stream, bool trace) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::int32_t T = static_cast<std::int32_t>(q.ids.size());
    if (T < 1 || T > max_tokens_) { throw std::invalid_argument("decision: prompt of " + std::to_string(T) + " tokens"); }
    const int K = static_cast<int>(q.option_idx.size());
    if (K < 1 || q.decide_idx < 0 || q.decide_idx >= T) { throw std::invalid_argument("decision: bad readout positions"); }
    for (int i : q.option_idx) {
        if (i < 0 || i >= T) { throw std::invalid_argument("decision: an option position is outside the prompt"); }
    }
    const int loops = std::clamp(policy.max_loops, 1, max_loops_);
    CUDA_CHECK(cudaMemcpyAsync(ids_.data, q.ids.data(), static_cast<std::size_t>(T) * 4, cudaMemcpyHostToDevice, stream));
    WorkspaceArena work(DeviceSpan{workspace_.p, workspace_bytes_});
    RoundIO io;
    io.model        = &base_.runtime;
    io.im           = &base_.model;
    io.queries      = &queries_;
    io.x            = Tensor(x_.data, DType::BF16, {static_cast<std::int32_t>(kColumn), T});
    io.positions    = Tensor(positions_.data, DType::I32, {T, 1});
    io.meta         = meta_;
    io.kv           = kv_;
    io.conv         = conv_;
    io.target_state = target_state_;
    io.work         = &work;
    io.stream       = stream;
    embed(io, Tensor(ids_.data, DType::I32, {T}));
    const auto& five = kDecisionVisits5;
    for (int v = 0; v < C::core_start; ++v) {
        run_visit(io, five[static_cast<std::size_t>(v)], layers_.at(static_cast<std::size_t>(five[v].layer)), v);
    }
    std::vector<int> core_rows{q.decide_idx};
    core_rows.insert(core_rows.end(), q.option_idx.begin(), q.option_idx.end());
    const int state_end = std::max(1, q.option_idx[0]);
    std::vector<int> exit_rows(static_cast<std::size_t>(state_end));
    std::iota(exit_rows.begin(), exit_rows.end(), 0);
    exit_rows.insert(exit_rows.end(), core_rows.begin(), core_rows.end());

    Answer ans;
    Mat prev;
    bool have_prev = false;
    float total_drift = 0.0F, max_halt = 0.0F;
    std::optional<float> last_halt;
    std::vector<float> host;
    for (int p = 0; p < loops; ++p) {
        const int first = C::core_start + p * kCore;
        int slot = 0;
        for (int v = first; v < first + kCore; ++v) {
            const VisitInfo& vi = five[static_cast<std::size_t>(v)];
            run_visit(io, vi, layers_.at(static_cast<std::size_t>(vi.layer)), v);
            shared_expert(vi.layer - C::core_start, p, vi.next_partial, T, stream);
            slot = vi.next_partial;
        }
        // the layer-17 readout: the decide and option rows of the core's output
        read_rows(static_cast<std::uint16_t*>(x_.data) + static_cast<std::int64_t>(slot) * D, kColumn, T, core_rows,
                  host, stream);
        const Mat hd = rows_mat(host, 0, 1), ho = rows_mat(host, 1, K);
        PassReadout pr;
        if (have_prev) {
            Mat delta = hd;
            for (int c = 0; c < D; ++c) { delta.v[c] -= prev.v[c]; }
            total_drift += l2(delta) / std::max(l2(hd), 1e-6F);
        }
        pr.gate_logits = heads_->core_gate(hd, ho);
        if (p == 0) {
            ans.l1_pred = static_cast<int>(std::max_element(pr.gate_logits.begin(), pr.gate_logits.end()) -
                                           pr.gate_logits.begin());
            if (heads_->has_aux()) {
                for (float a : heads_->aux(hd, ho)) { ans.aux.push_back(sigmoid(a)); }
            }
        }
        bool halt = false;
        if (p < loops - 1 && policy.threshold <= 1.0) {
            const float score = sigmoid(heads_->halt(hd, ho, have_prev ? &prev : nullptr, pr.gate_logits, p));
            pr.halt   = score;
            last_halt = score;
            max_halt  = std::max(max_halt, score);
            halt      = score >= policy.threshold;
        }
        pr.max_halt = max_halt;
        pr.drift    = total_drift;
        prev        = hd;
        have_prev   = true;
        const bool exit_here = (halt || p == loops - 1) && ans.loops_used == 0;
        if (exit_here || trace) {
            const DecisionSchedule sched = decision_schedule(p + 1);
            for (int v = sched.coda_begin(); v < sched.count; ++v) {
                const VisitInfo& vi = sched.visits[v];
                run_visit(io, vi, layers_.at(static_cast<std::size_t>(vi.layer)), 1000 + v);
            }
            Tensor hidden(hidden_.data, DType::BF16, {D, T});
            final_mix(io, sched.final_stack, output_query_, hidden);
            read_rows(hidden_.data, D, T, exit_rows, host, stream);
            const Mat h_state = rows_mat(host, 0, state_end);
            const Mat hd_e = rows_mat(host, state_end, 1), ho_e = rows_mat(host, state_end + 1, K);
            pr.exited      = true;
            pr.exit_logits = heads_->joint(hd_e, ho_e, heads_->pool(h_state));
            pr.probs       = softmax_t(pr.exit_logits, exit_temps_.at(static_cast<std::size_t>(p)));
            if (heads_->has_escalation()) {
                pr.escalate = sigmoid(heads_->escalation(hd_e, ho_e, total_drift, pr.exit_logits, max_halt));
            }
            if (exit_here) {
                ans.loops_used    = p + 1;
                ans.probs         = pr.probs;
                ans.logits        = pr.exit_logits;
                ans.temperature   = exit_temps_.at(static_cast<std::size_t>(p));
                ans.halt_score    = last_halt;
                ans.escalate_prob = pr.escalate;
                ans.state_drift   = total_drift;
            }
        }
        ans.passes.push_back(std::move(pr));
        if (ans.loops_used > 0 && !trace) { break; }
    }
    ans.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return ans;
}

} // namespace sinfer::targets::inkling2::detail::decision
