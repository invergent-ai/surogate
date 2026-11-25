// Inkling-2-Mini Variant: the residual hooks, the LM head, the state / window declarations, and the refusing
// leaves; the instantiation of the family runtime over this Variant (target design notes, 1).

#include "targets/inkling2/impl/variant.h"
#include "family/inkling_vision_ops.h"
#include "targets/inkling2/impl/visits.h"

#include "api/ops/inkling_mix.h"
#include "core/device.h"
#include "family/impl/runtime/target_support.h"

#include <stdexcept>
#include <string>
#include <vector>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::inkling2::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS inkling2_runtime
#include "family/impl/runtime/instantiate.h"
#include "family/impl/runtime/unrunnable_leaves.h"

namespace sinfer::targets::inkling2::detail {
namespace {

using C = TextConfig;

[[noreturn]] void no_family_loop(const char* leaf) {
    throw std::logic_error(std::string("inkling2: ") + leaf +
                           " is a leaf of the family layer loop, which this target never runs (visit_schedule)");
}
[[noreturn]] void no_linear(const char* leaf) {
    throw std::logic_error(std::string("inkling2: no linear-attention layers (") + leaf + ")");
}
[[noreturn]] void no_spec(const char* leaf) {
    throw std::logic_error(std::string("inkling2: no NextN draft head; DSpark runs as DFlash (") + leaf + ")");
}

const InklingModel& model_of(const RuntimeModelView& model) {
    if (model.full_layers.empty() || model.full_layers.front().projection.model == nullptr) {
        throw std::logic_error("inkling2: the model view carries no visits");
    }
    return *model.full_layers.front().projection.model;
}

ops::InklingPlane slot_plane(const Tensor& x, int slot) {
    return {static_cast<std::uint16_t*>(x.data) + static_cast<std::int64_t>(slot) * C::hidden, DType::BF16,
            C::residual};
}

} // namespace

SINFER_FAMILY_UNRUNNABLE_LEAVES(no_linear, no_spec)

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // Only the 8 global visits read past the 1,024-token window; their decode split follows these bands.
    return family::graph_profiles_through(capacity - 1, {1023, 4095, 16383, 65535, 262143});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t, std::uint32_t) { return {}; }

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t capacity, std::uint32_t,
                                                                  std::uint32_t) {
    // DSpark: the verify's split follows the global visits' bands, as an ordinary round's; the drafter's
    // own attention plans from the round shape alone.
    return ordinary_graph_profiles(capacity);
}

LinearAttentionStatePoolSpec Variant::visit_state_spec(const family::TextGeometry& geometry, std::int32_t slots) {
    if (geometry.layers != C::layers) { throw std::logic_error("inkling2: the served geometry is not the visit one"); }
    LinearAttentionStatePoolSpec spec;
    spec.layers        = static_cast<std::uint32_t>(C::layers);
    spec.conv_channels = ops::InklingMixGeometry::kConvChannels;
    spec.conv_width    = ops::InklingMixGeometry::kConvHistory;
    spec.slot_count    = slots;
    spec.conv_dtype    = DType::BF16;
    return spec;
}

std::vector<std::int32_t> Variant::kv_layer_windows(const family::TextGeometry& geometry) {
    if (geometry.layers != C::layers) { throw std::logic_error("inkling2: the served geometry is not the visit one"); }
    std::vector<std::int32_t> out(C::layers);
    for (int v = 0; v < C::layers; ++v) { out[v] = kVisits[v].global ? 0 : C::window; }
    return out;
}

void Variant::embed_residual(const ModelView& model, const Tensor& ids, Tensor& residual, WorkspaceArena&,
                             cudaStream_t stream) {
    const InklingModel& im = model_of(model);
    ops::inkling_embed(ids, model.token_embedding.qdata, model.token_embedding.n, im.weights->globals.embedding_norm,
                       C::rms_epsilon, slot_plane(residual, 0), stream);
}

void Variant::scatter_visual(const ModelView&, const Tensor& visual, const Tensor& indices, Tensor& residual,
                             cudaStream_t stream) {
    if (visual.ne[0] != C::hidden || residual.ne[0] != C::residual || !residual.is_contiguous()) {
        throw std::logic_error("inkling2: visual columns are [2304, V] into the [7 x 2304, T] stack");
    }
    family::inkling_scatter_columns(visual, indices, residual.data, C::residual, stream);
}

void Variant::final_residual_mix(const ModelView& model, const Tensor& residual, Tensor& hidden, WorkspaceArena&,
                                 cudaStream_t stream) {
    const InklingModel& im = model_of(model);
    ops::InklingAttnResSources src;
    src.stack_count = static_cast<std::int32_t>(kFinalStack.size());
    for (std::size_t i = 0; i < kFinalStack.size(); ++i) { src.stack[i] = slot_plane(residual, kFinalStack[i]); }
    src.last = slot_plane(residual, kFinalPartial);
    const Tensor norm(im.weights->globals.final_norm.data, DType::BF16, {C::hidden, 1});
    Tensor out(hidden.data, DType::BF16, {C::hidden, residual.ne[1], 1});
    ops::inkling_attn_res(src, residual.ne[1], im.queries->output, &norm, C::rms_epsilon, out, stream);
}

void Variant::lm_head(const ModelView& model, const Tensor& hidden, Tensor& logits, cudaStream_t stream) {
    if (logits.dtype != DType::FP32 || logits.ne[0] != C::output_rows || hidden.ne[0] != C::hidden ||
        logits.ne[1] * logits.ne[2] != hidden.ne[1] * hidden.ne[2]) {
        throw std::logic_error("inkling2: the LM head writes FP32 [201024, T] logits from a [2304, T] hidden");
    }
    const std::int32_t tokens = hidden.ne[1] * hidden.ne[2];
    // The invariant GEMM: a column's logits do not depend on how many columns share the head.
    ops::inkling_gemm_inv(model.output_head.qdata, C::output_rows, C::hidden, hidden.data, tokens, logits.data,
                          DType::FP32, C::output_rows, {}, nullptr, 0, stream);
    ops::inkling_softcap_f32(static_cast<float*>(logits.data), C::output_rows, tokens, C::output_rows,
                             C::logit_softcap, stream);
}

void Variant::prepare_lm_head(std::int32_t) {
    // The invariant head needs no per-width plan (the cuBLASLt route it replaced did, before graph capture).
}

// ---- leaves of the family layer loop (never reached: visit_schedule replaces the loop) ----

void Variant::attention_projection(const Tensor&, const FullAttentionProjectionWeights&, Tensor&, Tensor&, Tensor&,
                                   Tensor&, family::TextPhase, WorkspaceArena&, cudaStream_t) {
    no_family_loop("attention_projection");
}

void Variant::attention_output_projection(const Tensor&, const Weight&, Tensor&, family::TextPhase, WorkspaceArena&,
                                          cudaStream_t) {
    no_family_loop("attention_output_projection");
}

void Variant::post_mixer(const Tensor&, const PostMixerWeights&, Tensor&, family::TextPhase, WorkspaceArena&,
                         cudaStream_t) {
    no_family_loop("post_mixer");
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(const family::TextGeometry&, WeightsProfile,
                                                                   family::TextPhase, std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(const family::TextGeometry&, WeightsProfile,
                                                                          family::TextPhase, std::int32_t,
                                                                          std::int32_t) {
    return 0;
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry&, WeightsProfile,
                                                         family::TextPhase, std::int32_t, std::int32_t) {
    return 0;
}

} // namespace sinfer::targets::inkling2::detail
