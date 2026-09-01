#include "targets/qwen3/impl/variant.h"

#include "api/ops/attn_input_proj.h"
#include "api/ops/linear_add.h"
#include "api/ops/linear_swiglu.h"

#include "core/device.h"
#include "family/impl/lora_hook.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::qwen3::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS qwen3_runtime
#include "family/impl/runtime/instantiate.h"

namespace sinfer::targets::qwen3::detail {
namespace {

using family::apply_lora;
using family::apply_lora_qkv;

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

/// This target has one export profile and it is W8. The ungated
/// `attn_input_proj` overload it uses admits A16 only, and keeping the linear
/// leaves on the same policy is what makes the workspace figures below exact
/// rather than optimistic: a capacity sized for A16 does not hold the quantized
/// activation an A8 route would ask for.
constexpr ops::LinearPolicy kTextPolicy = ops::LinearPolicy::A16Only;

[[noreturn]] void no_linear_layers(const char* leaf) {
    throw std::logic_error(
        std::string("qwen3: ") + leaf +
        " was called, but every one of this target's 28 layers is full attention; it has no "
        "linear-attention mixer, no convolution state and no gating projection. Reaching here "
        "means the family runtime resolved a layer to the GDN branch, which its topology "
        "(full_attention_interval == 1, gdn_layers() == 0) cannot produce.");
}

[[noreturn]] void no_speculation(const char* leaf) {
    throw std::logic_error(
        std::string("qwen3: ") + leaf +
        " was called, but this target has no MTP block and no DFlash tower; --spec is refused when "
        "the artifact is bound. Reaching here means a speculative round started without one.");
}

std::size_t post_mixer_workspace_bytes(QType gate_up_qtype, QType down_qtype,
                                       ops::LinearPolicy policy, std::int32_t first,
                                       std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * TextConfig::intermediate, TextConfig::hidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, TextConfig::intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

QType profile_qtype(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return QType::W8G32_F16S;
    }
    throw std::invalid_argument("qwen3: invalid weights profile");
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window; the ranges follow the family's measured
    // split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t, std::uint32_t) {
    return {};
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}

// ---- Attention -------------------------------------------------------------

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, family::TextPhase,
                                   WorkspaceArena&, cudaStream_t stream) {
    // `gate` is deliberately untouched. Qwen3's projection has no output-gate
    // rows, and the family skips the sigmoid multiply for this target
    // (Variant::attention_output_gate == false), so nothing reads the plane.
    (void)gate;
    ops::attn_input_proj(hidden, weights.query_key_value, query, key, value, stream);
    apply_lora_qkv(weights.query_key_value, hidden, query, key, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, kTextPolicy, workspace, stream);
    apply_lora(weight, 3, attention, residual, stream);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    (void)profile_qtype(weights_profile);
    // The ungated three-output `attn_input_proj` overload writes q, k and v
    // directly from the row-split parent; it materializes no packed parent and
    // takes no transient storage.
    return 0;
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, family::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    return ops::linear_add_workspace_capacity_bytes(profile_qtype(weights_profile),
                                                    TextConfig::hidden, TextConfig::query_size,
                                                    kTextPolicy, first, last);
}

// ---- Post-mixer (SwiGLU MLP) ----------------------------------------------

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation, kTextPolicy, workspace, stream);
    ops::linear_add(activation, weights.down, residual, kTextPolicy, workspace, stream);
    // down reads the SwiGLU activation, which is exactly the input its adapter
    // was trained against.
    apply_lora(weights.down, 4, activation, residual, stream);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         family::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    const QType qtype = profile_qtype(weights_profile);
    return post_mixer_workspace_bytes(qtype, qtype, kTextPolicy, first, last);
}

// ---- Leaves this target cannot run -----------------------------------------
//
// The family runtime is a template over this interface, so every leaf below has
// to exist. None of them can be reached by a Qwen3 topology, and each says so
// rather than returning quietly: a silent no-op here would be a layer that
// contributed nothing to the residual, which reads as a model that merely
// answers badly.

void Variant::gdn_input_projection(const Tensor&, const GdnProjectionWeights&, Tensor&, Tensor&,
                                   family::TextPhase, WorkspaceArena&, cudaStream_t) {
    no_linear_layers("gdn_input_projection");
}

void Variant::gdn_input_projection_snapshot(const Tensor&, const GdnProjectionWeights&,
                                            const Tensor&, Tensor&, const Tensor&, const Tensor&,
                                            const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                            family::TextPhase, WorkspaceArena&, cudaStream_t) {
    no_linear_layers("gdn_input_projection_snapshot");
}

void Variant::gdn_input_projection_record(const Tensor&, const GdnProjectionWeights&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                          Tensor&, Tensor&, Tensor&, Tensor&, family::TextPhase,
                                          WorkspaceArena&, cudaStream_t) {
    no_linear_layers("gdn_input_projection_record");
}

void Variant::gdn_output_projection(const Tensor&, const Weight&, Tensor&, family::TextPhase,
                                    WorkspaceArena&, cudaStream_t) {
    no_linear_layers("gdn_output_projection");
}

void Variant::gdn_norm_control_projection(const Tensor&, const Tensor&, float,
                                          const GdnProjectionWeights&, Tensor&, Tensor&, Tensor&,
                                          WorkspaceArena&, cudaStream_t) {
    no_linear_layers("gdn_norm_control_projection");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   family::TextPhase,
                                                                   std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(WeightsProfile,
                                                                            family::TextPhase,
                                                                            std::int32_t,
                                                                            std::int32_t,
                                                                            std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(WeightsProfile,
                                                                          family::TextPhase,
                                                                          std::int32_t,
                                                                          std::int32_t,
                                                                          std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile,
                                                                    family::TextPhase,
                                                                    std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t,
                                                                          std::int32_t) {
    return 0;
}

void Variant::mtp_attention_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                       Tensor&, Tensor&, Tensor&, WorkspaceArena&, cudaStream_t) {
    no_speculation("mtp_attention_projection");
}

void Variant::mtp_kv_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                Tensor&, WorkspaceArena&, cudaStream_t) {
    no_speculation("mtp_kv_projection");
}

void Variant::mtp_q_gate_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                    Tensor&, WorkspaceArena&, cudaStream_t) {
    no_speculation("mtp_q_gate_projection");
}

void Variant::mtp_post_mixer(const Tensor&, const MtpPostMixerWeights&, Tensor&, WorkspaceArena&,
                             cudaStream_t) {
    no_speculation("mtp_post_mixer");
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}


// Parity probe. SUROGATE_SERVE_DUMP_RESIDUAL=<dir> writes each tagged attention
// intermediate of the first forward as raw BF16 behind a 16-byte header
// {magic, rows, columns, occurrence}. Layers run in order, so the occurrence
// count of a tag is its layer index -- which keeps the family's probe signature
// (tag, tensor, stream) unchanged. Synchronises the stream, so it is only ever
// on for parity work.
namespace {

const char* probe_directory() {
    static const char* dir = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_DUMP_RESIDUAL");
        return (raw != nullptr && *raw != '\0') ? raw : nullptr;
    }();
    return dir;
}

std::int32_t probe_columns() {
    static const std::int32_t columns = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_DUMP_COLUMNS");
        return (raw != nullptr && *raw != '\0') ? std::atoi(raw) : 0;
    }();
    return columns;
}

std::map<std::string, int>& probe_counts() {
    static std::map<std::string, int> counts;
    return counts;
}

} // namespace

void Variant::debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream) {
    const char* dir = probe_directory();
    if (dir == nullptr || tensor.data == nullptr) { return; }
    // Prompt-sized rounds only: a long generation would otherwise write a file per
    // decode step per layer.
    if (tensor.ne[1] > 64) { return; }
    // Warmup runs forwards of its own before any request, so a plain "first N
    // occurrences" rule would spend the budget before the prompt under study
    // arrives. Select the round by its width instead: SUROGATE_SERVE_DUMP_COLUMNS
    // names the column count to capture, and the occurrence counter is kept per
    // (tag, width) so the captured round's layers number 0..N-1 whatever ran
    // before it.
    if (probe_columns() > 0 && tensor.ne[1] != probe_columns()) { return; }
    const std::string key = std::string(tag) + "@" + std::to_string(tensor.ne[1]);
    const int occurrence  = probe_counts()[key]++;
    if (occurrence >= TextConfig::layers) { return; }

    const std::size_t bytes = tensor.bytes();
    std::vector<std::byte> host(bytes);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tensor.data, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const std::string path =
        std::string(dir) + "/" + tag + "_" + std::to_string(occurrence) + ".bin";
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { return; }
    const std::int32_t header[4] = {0x51335042, tensor.ne[0], tensor.ne[1], occurrence};
    std::fwrite(header, sizeof(header), 1, file);
    std::fwrite(host.data(), 1, bytes, file);
    std::fclose(file);
}

} // namespace sinfer::targets::qwen3::detail
