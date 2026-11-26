// The Inkling-2 decision adapter's host heads (targets/inkling2/impl/decision/heads.cpp) against the System 1.5
// reference (eltokh7/inkling-2-mini-decision modeling.py, BF16 on CUDA): given the reference's own core and exit hidden
// states of a question, every pass's CoreGate logits, halting score, joint (answer) logits and probabilities,
// escalation probability and aux probes must agree with the reference trace.
//
// usage: sinfer_inkling2_decision_heads_test ADAPTER_DIR FIXTURE_DIR
//   FIXTURE_DIR/index.json + <question>.core<p>.f32 / .exit<p>.f32 ([T, 2304] row-major FP32), written from the
//   reference's dumps and trace (scratch tool; skipped when the arguments are absent).

#include "targets/inkling2/impl/decision/adapter_file.h"
#include "targets/inkling2/impl/decision/heads.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace sinfer::targets::inkling2::detail::decision;

namespace {

constexpr int D = 2304;

Mat load_rows(const std::string& path, int T) {
    Mat m;
    m.rows = T;
    m.cols = D;
    m.v.resize(static_cast<std::size_t>(T) * D);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(m.v.data()), static_cast<std::streamsize>(m.v.size() * sizeof(float)));
    if (!f) { throw std::runtime_error("short fixture " + path); }
    return m;
}

Mat gather(const Mat& m, const std::vector<int>& rows) {
    Mat out;
    out.rows = static_cast<std::int64_t>(rows.size());
    out.cols = m.cols;
    for (int r : rows) { out.v.insert(out.v.end(), m.row(r), m.row(r) + m.cols); }
    return out;
}

Mat head_rows(const Mat& m, int n) {
    Mat out;
    out.rows = n;
    out.cols = m.cols;
    out.v.assign(m.v.begin(), m.v.begin() + static_cast<std::ptrdiff_t>(n) * m.cols);
    return out;
}

float sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

std::vector<float> softmax_t(const std::vector<float>& x, double t) {
    std::vector<double> e(x.size());
    double mx = *std::max_element(x.begin(), x.end()), s = 0;
    for (std::size_t i = 0; i < x.size(); ++i) { s += e[i] = std::exp((x[i] - mx) / t); }
    std::vector<float> p(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { p[i] = static_cast<float>(e[i] / s); }
    return p;
}

struct Worst {
    double v = 0;
    std::string where;
    void see(double err, const std::string& w) {
        if (err > v) { v = err; where = w; }
    }
};

double max_abs(const std::vector<float>& a, const nlohmann::json& b) {
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { m = std::max(m, std::abs(a[i] - b[i].get<double>())); }
    return m;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("SKIP: usage %s ADAPTER_DIR FIXTURE_DIR\n", argv[0]);
        return 0;
    }
    const AdapterFile adapter(argv[1]);
    const auto& meta = adapter.meta();
    const Heads heads(adapter.heads(), meta.at("heads").at("evidence_pooler_max_chunks").get<int>(),
                      meta.at("max_loops").get<int>());
    const std::string fixture = argv[2];
    std::ifstream idx(fixture + "/index.json");
    const nlohmann::json index = nlohmann::json::parse(idx);
    Worst gate, halt, logits, probs, esc, aux;
    int checked = 0;
    for (const auto& q : index.at("questions")) {
        const std::string name = q.at("name");
        const int T            = q.at("T");
        const int d_idx        = q.at("decide_idx");
        const auto opt_idx     = q.at("opt_idx").get<std::vector<int>>();
        Mat prev;
        bool have_prev = false;
        const auto& passes = q.at("passes");
        for (std::size_t lp = 0; lp < passes.size(); ++lp) {
            const auto& ref = passes[lp];
            const std::string at = name + " pass " + std::to_string(lp + 1);
            const Mat core = load_rows(fixture + "/" + name + ".core" + std::to_string(lp) + ".f32", T);
            const Mat hd = gather(core, {d_idx}), ho = gather(core, opt_idx);
            const std::vector<float> gl = heads.core_gate(hd, ho);
            gate.see(max_abs(gl, ref.at("gate_logits")), at);
            if (!ref.at("halt").is_null()) {
                const float h = sigmoid(heads.halt(hd, ho, have_prev ? &prev : nullptr, gl, static_cast<int>(lp)));
                halt.see(std::abs(h - ref.at("halt").get<double>()), at);
            }
            if (lp == 0 && heads.has_aux() && !ref.at("aux").is_null()) {
                const std::vector<float> a = heads.aux(hd, ho);
                const char* names[3] = {"needs_multi_hop", "has_ambiguity_or_exception", "needs_calculation"};
                for (int i = 0; i < 3; ++i) {
                    aux.see(std::abs(sigmoid(a[static_cast<std::size_t>(i)]) - ref.at("aux").at(names[i]).get<double>()), at);
                }
            }
            prev      = hd;
            have_prev = true;
            const Mat ex = load_rows(fixture + "/" + name + ".exit" + std::to_string(lp) + ".f32", T);
            const Mat hd_e = gather(ex, {d_idx}), ho_e = gather(ex, opt_idx);
            const Mat ctx = heads.pool(head_rows(ex, std::max(1, opt_idx[0])));
            const std::vector<float> el = heads.joint(hd_e, ho_e, ctx);
            logits.see(max_abs(el, ref.at("exit_logits")), at);
            const double temp = meta.at("exit_temps").at(std::to_string(lp + 1)).get<double>();
            probs.see(max_abs(softmax_t(el, temp), ref.at("probs")), at);
            const float e = sigmoid(heads.escalation(hd_e, ho_e, ref.at("drift").get<float>(), el,
                                                     ref.at("max_halt").get<float>()));
            esc.see(std::abs(e - ref.at("escalate").get<double>()), at);
            ++checked;
        }
    }
    std::printf("%d question-passes\n", checked);
    std::printf("core gate logits  max |err| %.3g  (%s)\n", gate.v, gate.where.c_str());
    std::printf("halting score     max |err| %.3g  (%s)\n", halt.v, halt.where.c_str());
    std::printf("joint logits      max |err| %.3g  (%s)\n", logits.v, logits.where.c_str());
    std::printf("probabilities     max |err| %.3g  (%s)\n", probs.v, probs.where.c_str());
    std::printf("escalate_prob     max |err| %.3g  (%s)\n", esc.v, esc.where.c_str());
    std::printf("aux probes        max |err| %.3g  (%s)\n", aux.v, aux.where.c_str());
    const bool ok = gate.v < 1e-3 && halt.v < 1e-4 && logits.v < 1e-3 && probs.v < 1e-4 && esc.v < 1e-4 && aux.v < 1e-4;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
