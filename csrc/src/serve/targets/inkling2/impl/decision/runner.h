#pragma once

// An Inkling-2 decision adapter served over the base model (System 1.5): the adapter's device weights and the
// question loop. A question is one whole-prompt round: the prelude once, then core passes 1..max_loops -- after each,
// the layer-17 readout (CoreGateHead + DynamicHaltingHead on the host) decides whether to leave -- then the coda of the
// schedule it stopped at, the final mix, and the exit heads (JointSchemaHead with the evidence pooler, EscalationHead).
// This is `System15Engine._run_core_and_exit_batch` for one item (accumulate_exit_trajectory=False, gate="halting").
//
// The adapter's replacements are bound over a copy of the base's layer weights: the attention projections (merged,
// BF16), the routers and the loop controllers (rounded to the engine's BF16: the reference keeps them FP32), and the
// AttnRes / cross-pass queries (FP32, pre-scaled as the base's). Everything else is the base's, shared.

#include "targets/inkling2/impl/decision/adapter_file.h"
#include "targets/inkling2/impl/decision/heads.h"
#include "targets/inkling2/impl/decision/round.h"
#include "targets/inkling2/impl/model.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sinfer::targets::inkling2::detail::decision {

/// The tokens of one (state, question) prompt as the System 1.5 protocol encodes it (`_encode_question`).
struct Question {
    std::vector<std::int32_t> ids;
    std::int32_t decide_idx = 0;            ///< the <decide> marker's position
    std::vector<std::int32_t> option_idx;   ///< each option's closing marker position
};

/// When to stop looping: halt once the halting score reaches `threshold` (> 1: never, the "full" mode), at most
/// `max_loops` passes.
struct Policy {
    double threshold = 0.969;
    int max_loops    = 5;
};

/// One pass's readouts (always the core ones; the exit ones when the pass exited, or every pass with `trace`).
struct PassReadout {
    std::vector<float> gate_logits;
    std::optional<float> halt;          ///< sigmoid(halting logit); absent on the last pass and in "full" mode
    float max_halt = 0.0F, drift = 0.0F;
    bool exited = false;
    std::vector<float> exit_logits;     ///< JointSchemaHead logits (exited passes)
    std::vector<double> probs;          ///< softmax(exit_logits / exit_temps[pass])
    std::optional<float> escalate;      ///< sigmoid(EscalationHead)
    // With `trace`: the hidden rows the heads read on this pass, FP32 row-major (for head training / parity work).
    std::vector<float> cap_core;        ///< [1 + K, D]: the layer-17 decide row, then the option rows
    std::vector<float> cap_exit;        ///< [1 + K, D]: the exit decide row, then the option rows
    std::vector<float> cap_pooled;      ///< [cap_pooled_rows, D]: the evidence pooler's mean-pooled state rows
    std::int64_t cap_pooled_rows = 0;
};

struct Answer {
    int loops_used = 0;
    std::vector<double> probs;          ///< the final distribution over the options
    std::vector<float> logits;
    double temperature = 1.0;
    std::optional<float> halt_score;    ///< the last halting score computed (None before any)
    std::optional<float> escalate_prob;
    float state_drift = 0.0F;
    int l1_pred = -1;                   ///< argmax of pass 1's CoreGate logits
    std::vector<float> aux;             ///< AuxStructureHead probabilities (pass 1), empty without the head
    std::vector<PassReadout> passes;
    double seconds = 0.0;
};

class Runner {
public:
    /// `base`: the served model (its weights stay where they are); `max_tokens`: the longest prompt this runner takes.
    Runner(const LoadedModelData& base, const std::filesystem::path& adapter_dir, std::int32_t max_tokens);
    ~Runner();
    Runner(const Runner&)            = delete;
    Runner& operator=(const Runner&) = delete;

    /// Answers one question on `stream` (synchronises it). `trace`: run every pass to max_loops with an exit readout
    /// after each (parity work; the answer is still the policy's).
    [[nodiscard]] Answer run(const Question& q, const Policy& policy, cudaStream_t stream, bool trace = false);

    [[nodiscard]] const nlohmann::json& meta() const { return file_.meta(); }
    [[nodiscard]] std::int32_t max_tokens() const { return max_tokens_; }
    /// The threshold of a calibrated mode ("fast", "balanced", "strict", "full").
    [[nodiscard]] double mode_threshold(const std::string& mode) const;
    [[nodiscard]] int max_loops() const { return max_loops_; }
    /// The adapter's special-token strings (state, question, option open, option close, decide).
    [[nodiscard]] std::vector<std::string> special_tokens() const;

private:
    void shared_expert(int core_index, int pass, int slot, std::int32_t T, cudaStream_t s);
    void read_rows(const void* plane, std::int64_t column_stride, std::int32_t T, const std::vector<int>& rows,
                   std::vector<float>& out, cudaStream_t s);

    const LoadedModelData& base_;
    AdapterFile file_;
    std::unique_ptr<Heads> heads_;
    std::int32_t max_tokens_ = 0;
    int max_loops_ = 5;
    std::vector<double> exit_temps_; ///< by pass (0-based)

    // device: the adapter's weights
    DeviceBuffer weights_;
    std::vector<load::InklingLayerWeights> layers_; ///< 24 physical layers: the base's with the replacements
    load::InklingGlobals globals_;                  ///< the adapter's raw queries (for the derivation)
    DerivedQueries queries_;
    Tensor output_query_;                           ///< pre-scaled FP32 [D, 1] (queries_.output)
    struct SharedExpert {
        const void* norm = nullptr;    ///< BF16 [D]
        const void* gate_up = nullptr; ///< BF16 [2F, D]: gate rows, then up rows
        const void* down = nullptr;    ///< BF16 [D, F]
        std::vector<const void*> ada;  ///< per pass: BF16 [3, D] = scale, shift, gate
    };
    std::vector<SharedExpert> experts_;
    int expert_ff_ = 2048;

    // device: one question's state
    DeviceBuffer scratch_;
    Tensor x_, ids_, positions_, meta_, conv_, hidden_;
    PagedKVBatchLayerView kv_{};
    Numerics numerics_{};
    void* target_state_ = nullptr;
    DeviceBuffer workspace_;
    std::size_t workspace_bytes_ = 0;
    void* sp_x_ = nullptr; void* sp_gate_up_ = nullptr; void* sp_act_ = nullptr; void* sp_out_ = nullptr;
    std::vector<std::uint16_t> host_rows_;
};

} // namespace sinfer::targets::inkling2::detail::decision
