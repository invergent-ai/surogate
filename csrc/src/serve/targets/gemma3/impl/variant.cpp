#include "targets/gemma3/impl/variant.h"

#include "api/ops/gelu.h"
#include "api/ops/gelu_mul.h"
#include "api/ops/linear.h"
#include "api/ops/linear_add.h"
#include "api/ops/residual_add.h"
#include "api/ops/rmsnorm.h"

#include "core/device.h"
#include "core/layout.h"
#include "family/impl/lora_hook.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::gemma3_270m::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS gemma3_270m_runtime
#include "family/impl/runtime/instantiate.h"

namespace sinfer::targets::gemma3_270m::detail {
namespace {

using family::apply_lora;

/// Adapter ports. A bank is keyed by (base weight pointer, port), and Gemma 3
/// fuses nothing -- every logical projection is its own matrix, so the pointer
/// alone would already be a sufficient identity. The ports are kept distinct and
/// numbered the way the family's fused targets number them (q/k/v/o/down = 0..4)
/// so that a reader comparing two targets sees the same module in the same slot;
/// gate and up continue the sequence because this target, unlike the fused ones,
/// can actually serve adapters for them.
constexpr std::int32_t kQueryPort  = 0;
constexpr std::int32_t kKeyPort    = 1;
constexpr std::int32_t kValuePort  = 2;
constexpr std::int32_t kOutputPort = 3;
constexpr std::int32_t kDownPort   = 4;
constexpr std::int32_t kGatePort   = 5;
constexpr std::int32_t kUpPort     = 6;

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

/// This target has one export profile and it is W8. Keeping every linear leaf on
/// one policy is what makes the workspace figures below exact rather than
/// optimistic: a capacity sized for A16 does not hold the quantized activation
/// an A8 route would ask for.
constexpr ops::LinearPolicy kTextPolicy = ops::LinearPolicy::A16Only;

/// `hidden_activation: gelu_pytorch_tanh` in every released Gemma 3 config. The
/// mode is not cosmetic -- the tanh approximation and the erf form differ by more
/// than rounding -- which is why `ops::gelu_mul` takes it as a parameter.
constexpr ops::GeluMode kMlpActivation = ops::GeluMode::Tanh;

[[noreturn]] void no_linear_layers(const char* leaf) {
    throw std::logic_error(
        std::string("gemma3: ") + leaf +
        " was called, but every one of this target's 18 layers is full attention; it has no "
        "linear-attention mixer, no convolution state and no gating projection. Reaching here "
        "means the family runtime resolved a layer to the GDN branch, which its topology "
        "(full_attention_interval == 1, gdn_layers() == 0) cannot produce.");
}

[[noreturn]] void no_speculation(const char* leaf) {
    throw std::logic_error(
        std::string("gemma3: ") + leaf +
        " was called, but this target has no MTP block and no DFlash tower; --spec is refused when "
        "the artifact is bound. Reaching here means a speculative round started without one.");
}

QType profile_qtype(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return QType::W8G32_F16S;
    }
    throw std::invalid_argument("gemma3: invalid weights profile");
}

/// Transient bytes of one `ops::linear`, measured in its own scope so that a
/// sequence of them costs the largest rather than the sum. Exact for this
/// target's one profile: the W8 route takes no transient storage at all, so
/// every figure below is zero and the scoping only documents the intent. It is
/// also the call that *validates* the physical problem -- a shape the W8 kernel
/// registry does not admit throws here, at plan time, rather than at the first
/// forward.
void account_linear(WorkspaceLayoutBuilder& layout, QType qtype, std::int32_t output_rows,
                    std::int32_t input_rows, std::int32_t first, std::int32_t last) {
    auto scope = layout.scope();
    (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(qtype, output_rows, input_rows,
                                                                  kTextPolicy, first, last));
}

std::size_t attention_projection_workspace_bytes(QType qtype, std::int32_t first,
                                                 std::int32_t last) {
    // Three GEMMs out of the same hidden state, straight into the family's own
    // query/key/value planes. Nothing is materialized between them, so the whole
    // cost is whichever of the three asks for the most transient storage.
    WorkspaceLayoutBuilder layout;
    account_linear(layout, qtype, TextConfig::query_size, TextConfig::hidden, first, last);
    account_linear(layout, qtype, TextConfig::kv_size, TextConfig::hidden, first, last);
    account_linear(layout, qtype, TextConfig::kv_size, TextConfig::hidden, first, last);
    return layout.peak_bytes(1);
}

std::size_t attention_output_workspace_bytes(QType qtype, std::int32_t first, std::int32_t last) {
    // Sized for the norm-applying tail, which is the model Gemma actually is:
    // o_proj into a plane, `post_attention_norm` over it, then the residual add.
    // The fused `linear_add` the leaf runs today needs strictly less, so the
    // figure does not change when the shared runtime starts routing the
    // projection payload here.
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    account_linear(layout, qtype, TextConfig::hidden, TextConfig::query_size, first, last);
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            qtype, TextConfig::hidden, TextConfig::query_size, kTextPolicy, first, last));
    }
    return layout.peak_bytes(1);
}

std::size_t post_mixer_workspace_bytes(QType qtype, std::int32_t first, std::int32_t last) {
    // gate, up, the gated activation, the down projection, and its normalised
    // copy. Five planes where Llama's SwiGLU MLP needs one, and every one of them
    // is a separate allocation on purpose: `gelu_mul` and `rmsnorm` both require
    // their output not to overlap their inputs.
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    account_linear(layout, qtype, TextConfig::intermediate, TextConfig::hidden, first, last);
    account_linear(layout, qtype, TextConfig::intermediate, TextConfig::hidden, first, last);
    account_linear(layout, qtype, TextConfig::hidden, TextConfig::intermediate, first, last);
    return layout.peak_bytes(1);
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window; the ranges follow the family's measured
    // split-policy transitions until the producer grid reaches its fixed cap. This
    // model's native context stops at 32,768, so the last end is exactly its
    // frontier and the tail beyond it is never reached.
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
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    // `gate` is deliberately untouched. Gemma 3's attention has no output gate,
    // and the family skips the sigmoid multiply for this target
    // (Variant::attention_output_gate == false), so nothing reads the plane.
    (void)gate;
    // Three separate matrices, and three separate LoRA bank keys with them. The
    // family already hands us four independently contiguous destinations, so
    // there is nothing to split afterwards -- which is the reason the declaration
    // keeps q, k and v unfused in the first place.
    ops::linear(hidden, weights.query, query, kTextPolicy, workspace, stream);
    apply_lora(weights.query, kQueryPort, hidden, query, stream);
    ops::linear(hidden, weights.key, key, kTextPolicy, workspace, stream);
    apply_lora(weights.key, kKeyPort, hidden, key, stream);
    ops::linear(hidden, weights.value, value, kTextPolicy, workspace, stream);
    apply_lora(weights.value, kValuePort, hidden, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    // INCOMPLETE, and loudly so. Gemma sandwiches its attention block: the
    // residual takes `post_attention_layernorm(o_proj(attention))`, not
    // `o_proj(attention)`. The norm is bound and materialised
    // (`AttentionProjectionPayload::post_attention_norm`) and the overload below
    // applies it -- but the shared runtime calls this leaf with `*w.o_proj` and
    // nothing else, so from here the weight is unreachable. Applying it needs one
    // change in `family/impl/runtime/`, which this target does not own:
    // route `*w.projection` to this leaf (the three call sites in
    // text_context_impl.h) and the seven-argument overload takes over.
    static bool warned = [] {
        std::fprintf(stderr,
                     "gemma3: post_attention_norm is bound but NOT applied -- the shared runtime "
                     "does not yet hand the attention payload to attention_output_projection. "
                     "Output is not Gemma 3 until it does.\n");
        return true;
    }();
    (void)warned;
    ops::linear_add(attention, weight, residual, kTextPolicy, workspace, stream);
    apply_lora(weight, kOutputPort, attention, residual, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          const FullAttentionProjectionWeights& weights,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                 = workspace.scope();
    const std::int32_t columns = attention.ne[1];
    Tensor projected  = workspace.alloc(DType::BF16, {TextConfig::hidden, columns});
    Tensor normalized = workspace.alloc(DType::BF16, {TextConfig::hidden, columns});
    ops::linear(attention, weight, projected, kTextPolicy, workspace, stream);
    apply_lora(weight, kOutputPort, attention, projected, stream);
    // Zero-centred, like every Gemma norm: the artifact holds `w` and the kernel
    // applies `1 + w`. Same convention as `norm_unit_offset<Variant>()`, which is
    // the family default this target does not override.
    ops::rmsnorm(projected, weights.post_attention_norm, TextConfig::rms_epsilon,
                 /*unit_offset*/ true, normalized, stream);
    ops::residual_add(normalized, residual, stream);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    return attention_projection_workspace_bytes(profile_qtype(weights_profile), first, last);
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, family::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    return attention_output_workspace_bytes(profile_qtype(weights_profile), first, last);
}

// ---- Post-mixer (gated-GELU MLP, between the other two sandwich norms) ------

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                 = workspace.scope();
    const std::int32_t columns = hidden.ne[1];
    Tensor gate       = workspace.alloc(DType::BF16, {TextConfig::intermediate, columns});
    Tensor up         = workspace.alloc(DType::BF16, {TextConfig::intermediate, columns});
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, columns});
    Tensor projected  = workspace.alloc(DType::BF16, {TextConfig::hidden, columns});
    Tensor normalized = workspace.alloc(DType::BF16, {TextConfig::hidden, columns});

    // Gate and up are separate matrices here, where Llama and the Qwen families
    // fuse them and feed one `linear_swiglu`. That costs a launch and buys two
    // adaptable modules: `gate_proj` and `up_proj` have their own weights, so the
    // hook can add a delta to each instead of the fused targets' refusal.
    ops::linear(hidden, weights.gate, gate, kTextPolicy, workspace, stream);
    apply_lora(weights.gate, kGatePort, hidden, gate, stream);
    ops::linear(hidden, weights.up, up, kTextPolicy, workspace, stream);
    apply_lora(weights.up, kUpPort, hidden, up, stream);
    ops::gelu_mul(gate, up, kMlpActivation, activation, stream);
    ops::linear(activation, weights.down, projected, kTextPolicy, workspace, stream);
    // down reads the gated activation, which is exactly the input its adapter was
    // trained against.
    apply_lora(weights.down, kDownPort, activation, projected, stream);
    // The second half of the FFN sandwich: `post_feedforward_layernorm` over the
    // MLP's output, before it reaches the residual.
    ops::rmsnorm(projected, weights.post_feedforward_norm, TextConfig::rms_epsilon,
                 /*unit_offset*/ true, normalized, stream);
    ops::residual_add(normalized, residual, stream);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         family::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    return post_mixer_workspace_bytes(profile_qtype(weights_profile), first, last);
}

// ---- Leaves this target cannot run -----------------------------------------
//
// The family runtime is a template over this interface, so every leaf below has
// to exist. None of them can be reached by a Gemma 3 topology, and each says so
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
    // 'G3PB' -- the header layout is the family's ({magic, rows, columns,
    // occurrence}, little-endian int32); only the magic names the target that
    // wrote the file, so a dump directory cannot be misattributed.
    const std::int32_t header[4] = {0x47335042, tensor.ne[0], tensor.ne[1], occurrence};
    std::fwrite(header, sizeof(header), 1, file);
    std::fwrite(host.data(), 1, bytes, file);
    std::fclose(file);
}

} // namespace sinfer::targets::gemma3_270m::detail
