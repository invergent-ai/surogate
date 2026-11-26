#pragma once

// The System 1.5 heads of an Inkling-2 decision adapter (System15Engine, eltokh7/inkling-2-mini-decision `modeling.py`),
// on the host in FP32 as the reference runs them: CoreGateHead, DynamicHaltingHead, JointSchemaHead (with the
// FullContextEvidencePooler), EscalationHead and AuxStructureHead. They read a handful of hidden rows per question
// (the decide token, the option tokens, the pooled state), so the host is fast enough (milliseconds) and exact.
//
// Every head is a straight port of its torch module: nn.Linear (x W^T + b), nn.RMSNorm with eps = None (torch then
// uses finfo(float32).eps, not the model's 1e-6), nn.MultiheadAttention (batch_first, packed in_proj, scaled
// dot-product softmax), F.silu, F.softmax. The CrossLoopTrajectoryMixer is not ported: the production path
// (`_run_core_and_exit_batch`, accumulate_exit_trajectory=False) never runs it.

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sinfer::targets::inkling2::detail::decision {

/// A row-major FP32 matrix [rows, cols] (a vector is rows = 1 or cols = 1 as the module stores it).
struct Mat {
    std::vector<float> v;
    std::int64_t rows = 0, cols = 0;
    [[nodiscard]] const float* row(std::int64_t r) const { return v.data() + r * cols; }
    [[nodiscard]] float* row(std::int64_t r) { return v.data() + r * cols; }
};

/// The heads' tensors by name (`heads/<module>/<param>`, '.' of the state dict as '/').
using HeadTensors = std::map<std::string, Mat>;

struct Linear {
    Mat w;                 ///< [out, in]
    std::optional<Mat> b;  ///< [out]
    [[nodiscard]] Mat operator()(const Mat& x) const; ///< x [n, in] -> [n, out]
};

struct RmsNorm {
    Mat w; ///< [d]
    [[nodiscard]] Mat operator()(const Mat& x) const;
};

struct MultiheadAttention {
    Linear in_proj;  ///< [3E, E]
    Linear out_proj; ///< [E, E]
    int heads = 8;
    [[nodiscard]] Mat operator()(const Mat& q, const Mat& k, const Mat& v) const;
};

/// What one pass's core readout gives the halting policy.
struct CoreReadout {
    std::vector<float> gate_logits; ///< CoreGateHead, one per option
    float halt_logit = 0.0F;        ///< DynamicHaltingHead (sigmoid -> halting score); 0 when not computed
};

class Heads {
public:
    explicit Heads(const HeadTensors& t, int pooler_max_chunks, int max_loops);

    /// CoreGateHead on the core rows (h_decide [1, D], h_opts [K, D]).
    [[nodiscard]] std::vector<float> core_gate(const Mat& h_decide, const Mat& h_opts) const;
    /// DynamicHaltingHead's logit; `h_decide_prev` empty on the first pass.
    [[nodiscard]] float halt(const Mat& h_decide, const Mat& h_opts, const Mat* h_decide_prev,
                             std::span<const float> gate_logits, int loop_index) const;
    /// FullContextEvidencePooler on the exit rows [0, state_end) -> [min(L, max_chunks), D].
    [[nodiscard]] Mat pool(const Mat& h_state) const;
    /// JointSchemaHead's option logits.
    [[nodiscard]] std::vector<float> joint(const Mat& h_decide, const Mat& h_opts, const Mat& h_ctx) const;
    /// EscalationHead's logit (sigmoid -> escalate_prob).
    [[nodiscard]] float escalation(const Mat& h_decide, const Mat& h_opts, float total_state_drift,
                                   std::span<const float> final_logits, float max_halt_prob) const;
    /// AuxStructureHead's three probe logits; empty when the adapter has no aux head.
    [[nodiscard]] std::vector<float> aux(const Mat& h_decide, const Mat& h_opts) const;
    [[nodiscard]] bool has_aux() const { return has_aux_; }
    [[nodiscard]] bool has_escalation() const { return has_escalation_; }

private:
    // JointSchemaHead
    Linear j_proj_q_, j_proj_ctx_, j_ffn_up_, j_ffn_down_, j_ptr_q_, j_ptr_k_, j_opt_bias_;
    Mat j_type_emb_;
    RmsNorm j_norm1_q_, j_norm1_kv_, j_norm2_, j_norm3_, j_norm_out_;
    MultiheadAttention j_cross_, j_self_;
    // CoreGateHead
    RmsNorm g_norm_;
    Linear g_q_, g_k_;
    // DynamicHaltingHead
    RmsNorm h_norm_d_, h_norm_o_, h_norm_delta_;
    Linear h_proj_d_, h_proj_o_, h_proj_delta_, h_mlp0_, h_mlp3_;
    Mat h_loop_emb_;
    int max_loops_ = 5;
    // EscalationHead / AuxStructureHead
    bool has_escalation_ = false, has_aux_ = false;
    RmsNorm e_norm_d_, e_norm_o_, a_norm_d_, a_norm_o_;
    Linear e_proj_d_, e_proj_o_, e_mlp0_, e_mlp3_, a_proj_d_, a_proj_o_, a_mlp0_, a_mlp3_;
    // FullContextEvidencePooler
    bool has_pooler_ = false;
    int pooler_chunks_ = 64;
    RmsNorm p_norm_;
    Linear p_up_, p_down_;
};

} // namespace sinfer::targets::inkling2::detail::decision
