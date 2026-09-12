#include "targets/gemma3/impl/variant.h"

#include "api/ops/gelu.h"
#include "api/ops/gelu_mul.h"
#include "api/ops/linear.h"
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
#include "family/impl/runtime/target_support.h"
#include "family/impl/runtime/unrunnable_leaves.h"

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

std::size_t attention_projection_workspace_bytes(const family::TextGeometry& g, QType qtype, std::int32_t first,
                                                 std::int32_t last) {
    // Three GEMMs out of the same hidden state, straight into the family's own
    // query/key/value planes. Nothing is materialized between them, so the whole
    // cost is whichever of the three asks for the most transient storage.
    WorkspaceLayoutBuilder layout;
    account_linear(layout, qtype, g.query_size(), g.hidden, first, last);
    account_linear(layout, qtype, g.kv_size(), g.hidden, first, last);
    account_linear(layout, qtype, g.kv_size(), g.hidden, first, last);
    return layout.peak_bytes(1);
}

std::size_t attention_output_workspace_bytes(const family::TextGeometry& g, QType qtype,
                                             std::int32_t first, std::int32_t last) {
    // o_proj into one plane, then `post_attention_norm` accumulating from it onto
    // the residual. One plane, not two: the norm writes where the residual add
    // used to, so nothing holds the normalised copy.
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {g.hidden, last});
    account_linear(layout, qtype, g.hidden, g.query_size(), first, last);
    return layout.peak_bytes(1);
}

std::size_t post_mixer_workspace_bytes(const family::TextGeometry& g, QType qtype, std::int32_t first, std::int32_t last) {
    // gate, up, the gated activation and the down projection. Four planes where
    // Llama's SwiGLU MLP needs one: `gelu_mul` requires its output not to overlap
    // its inputs, so those three stand apart. The normalised copy is gone --
    // the norm accumulates onto the residual instead of through a fifth plane.
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {g.intermediate, last});
    (void)layout.alloc(DType::BF16, {g.intermediate, last});
    (void)layout.alloc(DType::BF16, {g.intermediate, last});
    (void)layout.alloc(DType::BF16, {g.hidden, last});
    account_linear(layout, qtype, g.intermediate, g.hidden, first, last);
    account_linear(layout, qtype, g.intermediate, g.hidden, first, last);
    account_linear(layout, qtype, g.hidden, g.intermediate, first, last);
    return layout.peak_bytes(1);
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window; the ranges follow the family's measured
    // split-policy transitions until the producer grid reaches its fixed cap. This
    // model's native context stops at 32,768, so the last end is exactly its
    // frontier and the tail beyond it is never reached.
    return family::graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
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
    ops::linear_projections(hidden, {{weights.query, query, kTextPolicy},
                                     {weights.key, key, kTextPolicy},
                                     {weights.value, value, kTextPolicy}}, &workspace, stream);
    apply_lora(weights.query, kQueryPort, hidden, query, stream);
    apply_lora(weights.key, kKeyPort, hidden, key, stream);
    apply_lora(weights.value, kValuePort, hidden, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          const FullAttentionProjectionWeights& weights,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                 = workspace.scope();
    const std::int32_t columns = attention.ne[1];
    Tensor projected = workspace.alloc(DType::BF16, {residual.ne[0], columns});
    ops::linear(attention, weight, projected, kTextPolicy, workspace, stream);
    apply_lora(weight, kOutputPort, attention, projected, stream);
    // Zero-centred, like every Gemma norm: the artifact holds `w` and the kernel
    // applies `1 + w`. Same convention as `norm_unit_offset<Variant>()`, which is
    // the family default this target does not override.
    //
    // The norm accumulates straight onto the residual. The plane it used to
    // write and the residual add that read it back existed only because
    // `ops::rmsnorm` forbids aliasing its output with its inputs; the fused form
    // is bit-identical to that pair, which sinfer_rmsnorm_test pins.
    ops::rmsnorm_add(projected, weights.post_attention_norm, weights.rms_epsilon,
                     /*unit_offset*/ true, residual, stream);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    family::validate_token_interval(first, last);
    // The larger of the two routes this artifact may carry: the profile's row-split format,
    // or the K-quants a GGUF served natively keeps. The layout is planned before the weights
    // are read, so it must hold either.
    return std::max(
        attention_projection_workspace_bytes(geometry, profile_qtype(weights_profile), first, last),
        attention_projection_workspace_bytes(geometry, QType::Q4_K, first, last));
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return std::max(
        attention_output_workspace_bytes(geometry, profile_qtype(weights_profile), first, last),
        attention_output_workspace_bytes(geometry, QType::Q4_K, first, last));
}

// ---- Post-mixer (gated-GELU MLP, between the other two sandwich norms) ------

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                 = workspace.scope();
    const std::int32_t columns = hidden.ne[1];
    const std::int32_t intermediate = weights.gate.n;
    Tensor gate       = workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor up         = workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor activation = workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor projected  = workspace.alloc(DType::BF16, {residual.ne[0], columns});
    
    // Gate and up are separate matrices here, where Llama and the Qwen families
    // fuse them and feed one `linear_swiglu`. That costs a launch and buys two
    // adaptable modules: `gate_proj` and `up_proj` have their own weights, so the
    // hook can add a delta to each instead of the fused targets' refusal.
    ops::linear_projections(hidden, {{weights.gate, gate, kTextPolicy},
                                     {weights.up, up, kTextPolicy}}, &workspace, stream);
    apply_lora(weights.gate, kGatePort, hidden, gate, stream);
    apply_lora(weights.up, kUpPort, hidden, up, stream);
    ops::gelu_mul(gate, up, kMlpActivation, activation, stream);
    ops::linear(activation, weights.down, projected, kTextPolicy, workspace, stream);
    // down reads the gated activation, which is exactly the input its adapter was
    // trained against.
    apply_lora(weights.down, kDownPort, activation, projected, stream);
    // The second half of the FFN sandwich: `post_feedforward_layernorm` over the
    // MLP's output, before it reaches the residual.
    ops::rmsnorm_add(projected, weights.post_feedforward_norm, weights.rms_epsilon,
                     /*unit_offset*/ true, residual, stream);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                         family::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    family::validate_token_interval(first, last);
    return std::max(
        post_mixer_workspace_bytes(geometry, profile_qtype(weights_profile), first, last),
        post_mixer_workspace_bytes(geometry, QType::Q4_K, first, last));
}

// ---- Leaves this target cannot run -----------------------------------------
//
// The family runtime is a template over this interface, so every leaf below has
// to exist. None of them can be reached by a Gemma 3 topology, and each says so
// rather than returning quietly: a silent no-op here would be a layer that
// contributed nothing to the residual, which reads as a model that merely
// answers badly.

// Every leaf this target cannot run, defined once in the family: see
// family/impl/runtime/unrunnable_leaves.h. The two arguments are this
// target's own refusal messages.
SINFER_FAMILY_UNRUNNABLE_LEAVES(no_linear_layers, no_speculation)

void Variant::debug_probe(const char* tag, const Tensor& tensor, std::int32_t layer_count, cudaStream_t stream) {
    // Only the magic is this target's: 'G3PB'. Everything else -- which rounds
    // are captured, how the occurrence is counted, the header layout -- is the
    // family's, and lived in nine byte-identical copies before it moved there.
    family::debug_probe_dump(0x47335042, tag, tensor, layer_count, stream);
}

} // namespace sinfer::targets::gemma3_270m::detail
