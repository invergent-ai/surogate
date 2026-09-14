#include "targets/muse_glimmer/impl/variant.h"
#include "api/ops/embedding.h"
#include "api/ops/linear.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/silu_mul.h"
#include "family/impl/lora_hook.h"

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::muse_glimmer::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS muse_glimmer_runtime
#include "family/impl/runtime/instantiate.h"

namespace sinfer::targets::muse_glimmer::detail {
constexpr auto kPolicy = ops::LinearPolicy::A16Only;

void Variant::embed_residual(const ModelView& model, const Tensor& ids, Tensor& residual,
                             WorkspaceArena& work, cudaStream_t stream) {
    auto scope = work.scope();
    Tensor raw = work.alloc(DType::BF16, {model.geometry.hidden, static_cast<int>(ids.numel())});
    ops::embedding(ids, model.token_embedding, raw, stream);
    ops::rmsnorm_unweighted(raw, model.geometry.rms_epsilon, residual, stream);
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value,
                                   family::TextPhase phase, WorkspaceArena& work,
                                   cudaStream_t stream) {
    gemma3_270m::detail::Variant::attention_projection(hidden, weights, query, gate, key, value,
                                                       phase, work, stream);
    ops::linear(hidden, weights.output_gate, gate, kPolicy, work, stream);
    family::apply_lora(weights.output_gate, 7, hidden, gate, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          const FullAttentionProjectionWeights& weights,
                                          Tensor& residual, family::TextPhase, WorkspaceArena& work,
                                          cudaStream_t stream) {
    auto scope       = work.scope();
    Tensor projected = work.alloc(DType::BF16, {residual.ne[0], attention.ne[1]});
    ops::linear(attention, weight, projected, kPolicy, work, stream);
    family::apply_lora(weight, 3, attention, projected, stream);
    ops::rmsnorm_add(projected, weights.post_attention_norm, weights.rms_epsilon, false, residual,
                     stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& work, cudaStream_t stream) {
    auto scope        = work.scope();
    Tensor gate       = work.alloc(DType::BF16, {weights.gate.n, hidden.ne[1]});
    Tensor up         = work.alloc(DType::BF16, {weights.up.n, hidden.ne[1]});
    Tensor activation = work.alloc(DType::BF16, {weights.gate.n, hidden.ne[1]});
    Tensor projected  = work.alloc(DType::BF16, {residual.ne[0], hidden.ne[1]});
    ops::linear_projections(hidden, {{weights.gate, gate, kPolicy}, {weights.up, up, kPolicy}},
                            &work, stream);
    family::apply_lora(weights.gate, 5, hidden, gate, stream);
    family::apply_lora(weights.up, 6, hidden, up, stream);
    ops::silu_mul(gate, up, activation, stream);
    ops::linear(activation, weights.down, projected, kPolicy, work, stream);
    family::apply_lora(weights.down, 4, activation, projected, stream);
    ops::rmsnorm_add(projected, weights.post_feedforward_norm, weights.rms_epsilon, false, residual,
                     stream);
}
} // namespace sinfer::targets::muse_glimmer::detail
