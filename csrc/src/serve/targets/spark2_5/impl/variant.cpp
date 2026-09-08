#include "targets/spark2_5/impl/variant.h"

#include "api/ops/attn_input_proj.h"
#include "api/ops/residual_add.h"
#include "api/ops/embedding.h"
#include "api/ops/cast.h"
#include "api/ops/linear.h"
#include "api/ops/gelu_mul.h"
#include "api/ops/sigmoid_mul.h"

#include "core/device.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::spark2_5::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS spark2_5_runtime
#include "family/impl/runtime/instantiate.h"
#include "family/impl/runtime/target_support.h"
#include "family/impl/runtime/unrunnable_leaves.h"

namespace sinfer::targets::spark2_5::detail {
namespace {

constexpr ops::LinearPolicy kTextPolicy = ops::LinearPolicy::A16Only;

[[noreturn]] void no_linear_layers(const char* leaf) {
    throw std::logic_error(
        std::string("spark2_5: ") + leaf +
        " was called, but Spark has no linear-attention or convolution layers.");
}

[[noreturn]] void no_speculation(const char* leaf) {
    throw std::logic_error(
        std::string("spark2_5: ") + leaf +
        " was called, but this target has no MTP block and no DFlash tower; --spec is refused when "
        "the artifact is bound. Reaching here means a speculative round started without one.");
}

std::size_t post_mixer_workspace_bytes(const family::TextGeometry& g, QType gate_up_qtype, QType down_qtype,
                                       ops::LinearPolicy policy, std::int32_t first,
                                       std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {g.intermediate, last});
    (void)layout.alloc(DType::BF16, {g.intermediate, last});
    (void)layout.alloc(DType::BF16, {g.intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
            gate_up_qtype, g.intermediate, g.hidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc(DType::BF16, {g.hidden, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
            down_qtype, g.hidden, g.intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

QType profile_qtype(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return QType::W8G32_F16S;
    }
    throw std::invalid_argument("spark2_5: invalid weights profile");
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    return family::graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t, std::uint32_t) {
    return {};
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, family::TextPhase,
                                   WorkspaceArena&, cudaStream_t stream) {
    Tensor head_gate(gate.data, DType::BF16, {weights.output_gate.n, hidden.ne[1]});
    ops::linear(hidden, weights.output_gate, head_gate, stream);
    ops::attn_input_proj(hidden, weights.query_key_value, query, key, value, stream);

}

void Variant::embed_residual(const ModelView& model, const Tensor& ids, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    Tensor embedded = workspace.alloc(DType::BF16, {model.geometry.hidden, residual.ne[1]});
    ops::embedding(ids, model.token_embedding, embedded, stream);
    ops::cast_bf16_to_fp32(embedded, residual, stream);
}

void Variant::apply_attention_gate(const Tensor& gate, Tensor& attention, cudaStream_t stream) {
    Tensor head_gate(gate.data, DType::BF16, {attention.ne[1], attention.ne[2]});
    ops::headwise_sigmoid_mul(head_gate, attention, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    Tensor projected = workspace.alloc(DType::BF16, {weight.n, attention.ne[1]});
    ops::linear(attention, weight, projected, kTextPolicy, workspace, stream);
    ops::residual_add(projected, residual, stream);

}

std::size_t Variant::attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    family::validate_token_interval(first, last);
    (void)profile_qtype(weights_profile);
    return ops::linear_workspace_capacity_bytes(QType::BF16_CTRL, geometry.query_heads,
        geometry.hidden, kTextPolicy, first, last);
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {geometry.hidden, last});
    (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(profile_qtype(weights_profile),
        geometry.hidden, geometry.query_size(), kTextPolicy, first, last));
    return layout.peak_bytes(1);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {weights.gate_up.n / 2, hidden.ne[1]});
    Tensor gate = workspace.alloc(DType::BF16, {weights.gate_up.n / 2, hidden.ne[1]});
    Tensor up = workspace.alloc(DType::BF16, {weights.gate_up.n / 2, hidden.ne[1]});
    ops::linear_rows(hidden, weights.gate_up, 0, gate, &workspace, stream);
    ops::linear_rows(hidden, weights.gate_up, weights.gate_up.n / 2, up, &workspace, stream);
    ops::gelu_mul(gate, up, ops::GeluMode::Exact, activation, stream, true);
    Tensor projected = workspace.alloc(DType::BF16, {weights.down.n, hidden.ne[1]});
    ops::linear(activation, weights.down, projected, kTextPolicy, workspace, stream);
    ops::residual_add(projected, residual, stream);

}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                         family::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    family::validate_token_interval(first, last);
    const QType qtype = profile_qtype(weights_profile);
    return std::max(
        post_mixer_workspace_bytes(geometry, qtype, qtype, kTextPolicy, first, last),
        post_mixer_workspace_bytes(geometry, QType::Q4_K, QType::Q4_K,
                                   ops::LinearPolicy::A16Only, first, last));
}

SINFER_FAMILY_UNRUNNABLE_LEAVES(no_linear_layers, no_speculation)

void Variant::debug_probe(const char* tag, const Tensor& tensor, std::int32_t layer_count, cudaStream_t stream) {
    family::debug_probe_dump(0x53505042, tag, tensor, layer_count, stream);
}

} // namespace sinfer::targets::spark2_5::detail
