#include "family/vision_tower.h"
#include "family/gemma_vision.h"
#include "family/muse_vision.h"
#include "family/vision_fp8.h"

#include "core/layout.h"
#include "api/ops/add_bias.h"
#include "api/ops/gelu.h"
#include "api/ops/layer_norm.h"
#include "api/ops/linear.h"
#include "api/ops/linear_bias.h"
#include "api/ops/residual_add.h"
#include "api/ops/rope.h"
#include "api/ops/vision_attention.h"
#include "api/ops/vision_pos_embed.h"
#include "api/ops/vision_fp8.h"
#include "ops/linear/mxfp8/mxfp8.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sinfer::family {
namespace {

// The blocks' bias adds and tanh GELU ride their GEMMs' outputs (ops::try_linear_epilogue), and
// the two residual projections accumulate straight into the residual stream
// (ops::try_linear_bias_accumulate), instead of each step taking a pass over the activations --
// about a third of a block at a megapixel image. SUROGATE_SERVE_VISION_FUSED=0 runs them as
// separate ops, as before.
[[nodiscard]] bool vision_fused_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("SUROGATE_SERVE_VISION_FUSED");
        return value == nullptr || std::string_view(value) != "0";
    }();
    return enabled;
}

// --vision-fp8's e4m3 attention runs from this many patches (an image's 16-pixel patches;
// 3072 is about 890x890). With the rotation folded into the e4m3 preparation, rotation and
// attention together run 1.08x the BF16 pair at 3136 patches, 1.12x at 4624, 1.22x at 7056,
// 1.24x at 12544, and break even near 2300 (GB10, 16 heads, sinfer_vision_attention_fp8_test
// bench on a quiet GPU); each FP8 step also moves the outputs a little.
// SUROGATE_SERVE_VISION_FP8_ATTENTION=0 keeps attention BF16;
// SUROGATE_SERVE_VISION_FP8_ATTENTION_MIN_PATCHES moves the threshold.
[[nodiscard]] std::int32_t vision_fp8_attention_min_patches() noexcept {
    static const std::int32_t patches = [] {
        if (const char* off = std::getenv("SUROGATE_SERVE_VISION_FP8_ATTENTION");
            off != nullptr && std::string_view(off) == "0") {
            return std::numeric_limits<std::int32_t>::max();
        }
        if (const char* raw = std::getenv("SUROGATE_SERVE_VISION_FP8_ATTENTION_MIN_PATCHES")) {
            return static_cast<std::int32_t>(std::max(0L, std::strtol(raw, nullptr, 10)));
        }
        return std::int32_t{3072};
    }();
    return patches;
}

[[nodiscard]] bool fused(const Tensor& x, const Weight& weight, const Tensor& bias,
                         ops::LinearEpilogue epilogue, Tensor& out, cudaStream_t stream) {
    return vision_fused_enabled() &&
           ops::try_linear_epilogue(x, weight, bias, epilogue, out, stream);
}

[[nodiscard]] bool fused_residual(const Tensor& x, const Weight& weight, const Tensor& bias,
                                  Tensor& residual, cudaStream_t stream) {
    return vision_fused_enabled() &&
           ops::try_linear_bias_accumulate(x, weight, bias, residual, stream);
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a * b;
}

constexpr std::size_t kWorkspaceAlignment = kVisionWorkspaceAlignment;

struct VisionWorkspaceLayout {
    TensorRegion position_ids;
    TensorRegion cu_seqlens;
    TensorRegion pos_indices;
    TensorRegion pos_weights;
    TensorRegion x;
    TensorRegion patch_bf16;
    TensorRegion attended;
    TensorRegion qkv;
    TensorRegion attention_norm;
    std::optional<LayoutRegion> attention_workspace;
    TensorRegion projected;
    TensorRegion mlp_down;
    TensorRegion mlp_up;
    TensorRegion mlp_norm;
    TensorRegion normalized;
    TensorRegion projector_norm;
    TensorRegion merger_hidden;
    // --vision-fp8: one projection input at a time, quantized (sized for the widest, fc2's).
    std::optional<LayoutRegion> fp8_codes;
    std::optional<LayoutRegion> fp8_scales;
    std::size_t bytes = 0;
};

VisionWorkspaceLayout build_workspace_layout(const VisionGeometry& g,
                                             std::size_t patches64, std::size_t tokens64,
                                             std::size_t segment_count) {
    if (patches64 == 0 || tokens64 == 0 ||
        patches64 != checked_mul(tokens64, static_cast<std::size_t>(g.merge_unit()),
                                 "patch/token relation")) {
        throw std::invalid_argument("Vision workspace requires P=4V>0");
    }
    if (patches64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        tokens64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        segment_count == 0 ||
        segment_count >= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision request dimensions exceed int32");
    }
    const auto patches = static_cast<std::int32_t>(patches64);
    const auto tokens  = static_cast<std::int32_t>(tokens64);

    LayoutBuilder builder;
    VisionWorkspaceLayout out;
    const auto add = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                         const char* label) {
        return builder.add_tensor(dtype, shape, kWorkspaceAlignment, label);
    };
    out.position_ids = add(DType::I32, {patches, 2}, "vision position ids");
    out.cu_seqlens =
        add(DType::I32, {static_cast<std::int32_t>(segment_count + 1)}, "vision segment bounds");
    out.pos_indices = add(DType::I32, {4, patches}, "vision position indices");
    out.pos_weights = add(DType::FP32, {4, patches}, "vision position weights");
    out.x           = add(DType::BF16, {g.hidden, patches}, "vision residual");
    out.patch_bf16  = add(DType::BF16, {g.patch_dim, patches}, "vision BF16 patches");
    if (ops::vision_fp8()) {
        // Outside the scopes below: a block's projection inputs are quantized into it while the
        // scoped activations (h, attended, up) are live, so it must not share their storage.
        const std::int32_t widest = std::max(g.hidden, g.intermediate);
        out.fp8_codes  = builder.add(ops::detail::mxfp8_codes_bytes(patches, widest),
                                     kWorkspaceAlignment, "vision MXFP8 projection input");
        out.fp8_scales = builder.add(ops::detail::mxfp8_scales_bytes(patches, widest),
                                     kWorkspaceAlignment, "vision MXFP8 projection input scales");
    }
    {
        auto attention_scope = builder.scope();
        out.attended         = add(DType::BF16, {g.hidden, patches}, "vision attended");
        {
            auto qkv_scope = builder.scope();
            out.qkv        = add(DType::BF16, {3 * g.hidden, patches}, "vision QKV");
            {
                auto norm_scope    = builder.scope();
                out.attention_norm =
                    add(DType::BF16, {g.hidden, patches}, "vision attention norm");
            }
            const std::size_t attention_bytes = ops::vision_attention_workspace_capacity_bytes(
                patches, patches, static_cast<std::int32_t>(segment_count),
                static_cast<std::int32_t>(segment_count));
            if (attention_bytes != 0) {
                out.attention_workspace =
                    builder.add(attention_bytes, kWorkspaceAlignment, "vision attention workspace");
            }
        }
        out.projected = add(DType::BF16, {g.hidden, patches}, "vision projected");
    }
    {
        auto mlp_scope = builder.scope();
        out.mlp_up     = add(DType::BF16, {g.intermediate, patches}, "vision MLP up");
        {
            auto norm_scope = builder.scope();
            out.mlp_norm    = add(DType::BF16, {g.hidden, patches}, "vision MLP norm");
        }
        out.mlp_down = add(DType::BF16, {g.hidden, patches}, "vision MLP down");
    }
    out.normalized    = add(DType::BF16, {g.hidden, patches}, "vision merger norm");
    if (g.siglip2 && g.projector_norm) {
        out.projector_norm = add(DType::BF16, {g.merger_hidden(), tokens}, "vision projector norm");
    }
    out.merger_hidden = add(DType::BF16, {g.projector_width(), tokens}, "vision merger hidden");
    out.bytes = builder.finish(1, "vision workspace");
    return out;
}

void copy_host(const void* src, Tensor& dst, cudaStream_t stream) {
    if (dst.bytes() == 0) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

} // namespace

VisionContext::VisionContext(DeviceContext& ctx, const VisionWeights& vision,
                             const VisionGeometry& tower, const TextGeometry& text,
                             Probe probe)
    : weights_(&vision), ctx_(ctx), probe_(std::move(probe)) {
    cfg_               = bound_vision_geometry(tower, text);
    if (cfg_.gemma_version || cfg_.muse_glimmer) { return; }
    patch_embed_       = &vision.common.patch_embedding;
    patch_embed_bias_  = &vision.common.patch_embedding_bias;
    position_embed_    = &vision.common.position_embedding;
    post_norm_weight_ = &vision.common.post_norm_weight;
    post_norm_bias_ = &vision.common.post_norm_bias;
    // The materialized tower is the authority on its own depth; reading the count from the
    // geometry a second time is one more way for the two to disagree.
    blocks_.resize(vision.common.layers.size());
    for (std::uint32_t layer = 0; layer < blocks_.size(); ++layer) {
        const auto& source  = vision.common.layers[layer];
        BlockW& out         = blocks_[layer];
        out.norm1_weight    = &source.norm1_weight;
        out.norm1_bias      = &source.norm1_bias;
        out.qkv             = &source.qkv;
        out.qkv_bias        = &source.qkv_bias;
        out.projection      = &source.output;
        out.projection_bias = &source.output_bias;
        out.norm2_weight    = &source.norm2_weight;
        out.norm2_bias      = &source.norm2_bias;
        out.fc1             = &source.fc1;
        out.fc1_bias        = &source.fc1_bias;
        out.fc2             = &source.fc2;
        out.fc2_bias        = &source.fc2_bias;
    }
    for (const auto& source : vision.deepstack) {
        deepstack_.push_back({source.layer, MergerW{
            &source.norm_weight, &source.norm_bias, &source.fc1, &source.fc1_bias,
            &source.fc2, &source.fc2_bias}});
    }
    merger_.norm_weight = &vision.common.merger_norm_weight;
    merger_.norm_bias   = &vision.common.merger_norm_bias;
    merger_.fc1         = &vision.common.merger_fc1;
    merger_.fc1_bias    = &vision.common.merger_fc1_bias;
    merger_.fc2         = &vision.merger_fc2;
    merger_.fc2_bias    = &vision.merger_fc2_bias;
    if (ops::vision_fp8()) {
        fp8_ = find_vision_fp8(&vision);
        if (fp8_ && fp8_->blocks.size() != blocks_.size()) { fp8_.reset(); }
    }
}

const VisionFp8Block* VisionContext::fp8_block(std::size_t layer, std::int32_t patches) const {
    if (!fp8_) { return nullptr; }
    const VisionFp8Block& b = fp8_->blocks[layer];
    const BlockW& w         = blocks_[layer];
    const auto aligned      = [](const Tensor* t) {
        return t != nullptr && t->data != nullptr &&
               (reinterpret_cast<std::uintptr_t>(t->data) & 15U) == 0;
    };
    using ops::detail::mxfp8_gemm_available;
    if (!aligned(w.qkv_bias) || !aligned(w.projection_bias) || !aligned(w.fc1_bias) ||
        !aligned(w.fc2_bias) ||
        !mxfp8_gemm_available(b.qkv.rows, b.qkv.k, patches, true, false) ||
        !mxfp8_gemm_available(b.projection.rows, b.projection.k, patches, true, true) ||
        !mxfp8_gemm_available(b.fc1.rows, b.fc1.k, patches, true, false) ||
        !mxfp8_gemm_available(b.fc2.rows, b.fc2.k, patches, true, true)) {
        return nullptr;
    }
    return &b;
}

std::size_t VisionContext::workspace_bytes(const VisionGeometry& geometry,
                                           const VisionItemControl& item) {
    if (geometry.muse_glimmer) { return muse_vision_workspace_bytes(geometry,item.merged_count); }
    if (geometry.gemma_version) { return gemma_vision_workspace_bytes(geometry,item.merged_count); }
    return build_workspace_layout(geometry, item.patch_count, item.merged_count,
                                  static_cast<std::size_t>(item.segment_count))
        .bytes;
}

std::size_t VisionContext::output_transient_bytes(const VisionGeometry& geometry,
                                                  std::size_t merged_tokens) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision output transient extent must fit positive int32");
    }
    LayoutBuilder layout;
    (void)layout.add_tensor(
        DType::BF16, {geometry.output_hidden, static_cast<std::int32_t>(merged_tokens), 1 + geometry.deepstack_layers},
        kWorkspaceAlignment, "Vision item output transient");
    return layout.finish(kWorkspaceAlignment, "Vision item output transient layout");
}

std::size_t VisionContext::encoding_transient_bytes(const VisionGeometry& geometry,
                                                    std::size_t merged_tokens,
                                                    std::size_t text_residual_bytes) {
    const auto output = output_transient_bytes(geometry, merged_tokens);
    const auto patches = checked_mul(merged_tokens, geometry.merge_unit(), "encoding patches");
    // Once the tower finishes, the same storage holds the text block's residual.
    const auto residual = std::max(text_residual_bytes,
        checked_mul(checked_mul(patches, geometry.hidden, "encoding residual"),
                    sizeof(std::uint16_t), "encoding residual bytes"));
    if (output > std::numeric_limits<std::size_t>::max() - residual) {
        throw std::overflow_error("Vision encoding transient overflows size_t");
    }
    return output + residual;
}

std::size_t VisionContext::workspace_capacity_bytes(const VisionGeometry& geometry,
                                                    std::uint32_t max_merged_tokens,
                                                    std::uint32_t max_segments) {
    if (max_merged_tokens == 0 || max_segments == 0) {
        throw std::invalid_argument("Vision workspace capacity bounds must be positive");
    }
    if (geometry.muse_glimmer) { return muse_vision_workspace_bytes(geometry, std::min(max_merged_tokens, static_cast<std::uint32_t>(geometry.max_image_tokens))); }
    if (geometry.gemma_version) { return gemma_vision_workspace_bytes(geometry, std::min(max_merged_tokens, static_cast<std::uint32_t>(geometry.max_image_tokens))); }
    const std::uint32_t segments = std::min(max_merged_tokens, max_segments);
    return build_workspace_layout(geometry,
                                  checked_mul(max_merged_tokens,
                                              static_cast<std::size_t>(geometry.merge_unit()),
                                              "capacity patch count"),
                                  max_merged_tokens, segments)
        .bytes;
}

void VisionContext::encode(const VisionItemView& item, Tensor& output,
                           WorkspaceArena& workspace) const {
    VisionEncodeState state;
    while (!encode_step(item, output, workspace, state)) {}
}

bool VisionContext::encode_step(const VisionItemView& item, Tensor& output,
                                WorkspaceArena& workspace, VisionEncodeState& state) const {
    if (item.control == nullptr) { throw std::invalid_argument("Vision item control is null"); }
    const VisionItemControl& control = *item.control;
    const VisionGeometry& g          = cfg_;
    const auto patches64                     = control.patch_count;
    const auto tokens64                      = control.merged_count;
    if (item.patches.size() !=
        checked_mul(patches64, static_cast<std::size_t>(g.patch_dim), "patch elements")) {
        throw std::invalid_argument("Vision processor patch buffer has invalid shape");
    }
    if (output.dtype != DType::BF16 || output.ne[0] != g.output_hidden ||
        output.ne[1] != static_cast<std::int32_t>(tokens64) || output.ne[2] != 1 + g.deepstack_layers ||
        output.ne[3] != 1 || !output.is_contiguous() || output.data == nullptr) {
        throw std::invalid_argument("Vision output must be contiguous BF16 [H,V,1+deepstack_layers]");
    }
    if (state.phase == VisionEncodeState::Phase::Complete) { return true; }
    if (state.residual.data && (state.residual.dtype != DType::BF16 ||
        state.residual.ne[0] != g.hidden || state.residual.ne[1] != static_cast<int>(patches64) ||
        state.residual.ne[2] != 1 || state.residual.ne[3] != 1 || !state.residual.is_contiguous())) {
        throw std::invalid_argument("Vision encoding residual has an invalid shape");
    }
    if (g.muse_glimmer) {
        return encode_muse_vision_step(g, *weights_, item, output, workspace, ctx_.stream, probe_, state);
    }
    if (g.gemma_version) {
        return encode_gemma_vision_step(g, *weights_, item, output, workspace, ctx_.stream, probe_, state);
    }
    const VisionWorkspaceLayout layout = build_workspace_layout(
        g, patches64, tokens64, static_cast<std::size_t>(control.segment_count));
    if (workspace.capacity() < layout.bytes) {
        throw std::invalid_argument("Vision workspace capacity is too small for request");
    }
    const auto patches  = static_cast<std::int32_t>(patches64);
    const auto tokens   = static_cast<std::int32_t>(tokens64);
    cudaStream_t stream = ctx_.stream;
    workspace.reset();
    const DeviceSpan backing = workspace.alloc_bytes(layout.bytes, kWorkspaceAlignment);

    Tensor position_ids = layout.position_ids.bind(backing);
    Tensor cu_seqlens   = layout.cu_seqlens.bind(backing);
    Tensor pos_indices  = layout.pos_indices.bind(backing);
    Tensor pos_weights  = layout.pos_weights.bind(backing);
    copy_host(control.position_ids.data(), position_ids, stream);
    copy_host(control.cu_seqlens.data(), cu_seqlens, stream);

    Tensor x = state.residual.data ? state.residual : layout.x.bind(backing);
    if (state.phase == VisionEncodeState::Phase::Embedding) {
        copy_host(control.position_table_indices.data(), pos_indices, stream);
        copy_host(control.position_table_weights.data(), pos_weights, stream);
        Tensor patch_bf16 = layout.patch_bf16.bind(backing);
        copy_host(item.patches.data(), patch_bf16, stream);
        ops::linear(patch_bf16, *patch_embed_, x, stream);
        ops::add_bias(*patch_embed_bias_, x, stream);
        // The artifact records the source table shape [rows,hidden], while Tensor's
        // contiguous matrix convention is [inner,columns]. The payload is already
        // row-major, so this is a zero-copy [hidden,rows] view, not a transpose.
        Tensor position_table = position_embed_->reshape({g.hidden, g.position_embeddings});
        if (g.siglip2) {
            ops::siglip2_pos_embed_add(position_table, control.grid.height, control.grid.width,
                                       g.merge, x, stream);
        } else {
            ops::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
        }
        state.phase = blocks_.empty() ? VisionEncodeState::Phase::Projection : VisionEncodeState::Phase::Blocks;
        return false;
    }
    if (state.phase == VisionEncodeState::Phase::Blocks) {
        const std::size_t layer = state.layer;
        auto& deepstack_index = state.deepstack;
        const BlockW& block = blocks_[layer];
        // --vision-fp8: every projection input is quantized into one MXFP8 buffer just before
        // its GEMM; the residual projections accumulate into x with their bias in the epilogue.
        const VisionFp8Block* fp8 = fp8_block(layer, patches);
        ops::detail::Mxfp8Matrix input;
        if (fp8 != nullptr) {
            input.codes  = layout.fp8_codes->bind(backing).data;
            input.scales = layout.fp8_scales->bind(backing).data;
            input.rows   = patches;
        }
        const auto quantize = [&](const Tensor& t, ops::detail::Mxfp8Transform transform) {
            input.k = t.ne[0];
            ops::detail::mxfp8_quantize(t.data, t.ne[0], transform, input, stream);
        };
        const auto mxfp8 = [&](const ops::detail::Mxfp8Matrix& weight, const Tensor* bias,
                               bool accumulate, Tensor& out) {
            // fp8_block checked every plan at this width, so a refusal here is a bug.
            if (!ops::detail::mxfp8_gemm(weight, input, bias != nullptr ? bias->data : nullptr,
                                         accumulate, out.data, stream)) {
                throw std::logic_error("vision MXFP8 projection lost its plan");
            }
        };
        {
            Tensor attended = layout.attended.bind(backing);
            {
                Tensor qkv = layout.qkv.bind(backing);
                {
                    Tensor h = layout.attention_norm.bind(backing);
                    ops::layer_norm(x, *block.norm1_weight, *block.norm1_bias, g.norm_epsilon, h,
                                    stream);
                    if (fp8 != nullptr) {
                        quantize(h, ops::detail::Mxfp8Transform::None);
                        mxfp8(fp8->qkv, block.qkv_bias, false, qkv);
                    } else if (!fused(h, *block.qkv, *block.qkv_bias, ops::LinearEpilogue::Bias,
                                      qkv, stream)) {
                        ops::linear(h, *block.qkv, qkv, stream);
                        ops::add_bias(*block.qkv_bias, qkv, stream);
                    }
                }
                const std::int32_t plane      = g.hidden;
                const std::size_t plane_bytes = static_cast<std::size_t>(plane) * 2;
                Tensor q(qkv.data, DType::BF16, {g.head_dim(), g.heads, patches});
                Tensor k(static_cast<unsigned char*>(qkv.data) + plane_bytes, DType::BF16,
                         {g.head_dim(), g.heads, patches});
                Tensor v(static_cast<unsigned char*>(qkv.data) + 2 * plane_bytes, DType::BF16,
                         {g.head_dim(), g.heads, patches});
                q.nb[2] = qkv.nb[1];
                k.nb[2] = qkv.nb[1];
                v.nb[2] = qkv.nb[1];
                Tensor attended_heads = attended.view({g.head_dim(), g.heads, patches});
                const DeviceSpan attention_backing = layout.attention_workspace
                                                         ? layout.attention_workspace->bind(backing)
                                                         : backing;
                WorkspaceArena attention_workspace(attention_backing);
                // --vision-fp8 on sm_12x, large images: S = Q K^T and P V on the e4m3 tensor
                // cores, the operands prepared in the projection-input buffer, idle between qkv
                // and the output projection; the rotation is applied on the way into e4m3, so
                // the BF16 Q and K are never rotated in place.
                bool attended_done = false;
                bool rotated       = g.rotary_dim == 0;
                if (fp8 != nullptr && patches >= vision_fp8_attention_min_patches()) {
                    const DeviceSpan scratch = layout.fp8_codes->bind(backing);
                    if (!rotated) {
                        attended_done = ops::vision_attention_fp8_rope(
                            q, k, v, position_ids, g.rotary_dim, g.rope_theta, cu_seqlens,
                            attention_workspace, scratch, attended_heads, stream);
                    }
                    if (!attended_done) {
                        // A rotation the fused route does not reproduce bit for bit: rotate in
                        // place, then the e4m3 attention over the rotated BF16.
                        if (!rotated) {
                            ops::rope(position_ids, g.rotary_dim, g.rope_theta, q, k, stream);
                            rotated = true;
                        }
                        attended_done = ops::vision_attention_fp8(q, k, v, cu_seqlens,
                                                                  attention_workspace, scratch,
                                                                  attended_heads, stream);
                    }
                }
                if (!attended_done) {
                    if (!rotated) { ops::rope(position_ids, g.rotary_dim, g.rope_theta, q, k, stream); }
                    ops::vision_attention(q, k, v, cu_seqlens, attention_workspace, attended_heads,
                                          stream);
                }
            }
            // x += projection(attended) + bias, straight into the residual stream.
            if (fp8 != nullptr) {
                quantize(attended, ops::detail::Mxfp8Transform::None);
                mxfp8(fp8->projection, block.projection_bias, true, x);
            } else if (!fused_residual(attended, *block.projection, *block.projection_bias, x,
                                       stream)) {
                Tensor projected = layout.projected.bind(backing);
                ops::linear(attended, *block.projection, projected, stream);
                ops::add_bias(*block.projection_bias, projected, stream);
                ops::residual_add(projected, x, stream);
            }
        }
        {
            Tensor down = layout.mlp_down.bind(backing);
            Tensor up   = layout.mlp_up.bind(backing);
            {
                Tensor h = layout.mlp_norm.bind(backing);
                ops::layer_norm(x, *block.norm2_weight, *block.norm2_bias, g.norm_epsilon, h,
                                stream);
                if (fp8 != nullptr) {
                    // fc1 + bias only: the tanh GELU rides the quantization of fc2's input below.
                    quantize(h, ops::detail::Mxfp8Transform::None);
                    mxfp8(fp8->fc1, block.fc1_bias, false, up);
                } else if (!fused(h, *block.fc1, *block.fc1_bias, ops::LinearEpilogue::GeluTanhBias,
                                  up, stream)) {
                    ops::linear(h, *block.fc1, up, stream);
                    ops::add_bias(*block.fc1_bias, up, stream);
                    ops::gelu(up, ops::GeluMode::Tanh, stream);
                }
            }
            // x += fc2(up) + bias, straight into the residual stream.
            if (fp8 != nullptr) {
                quantize(up, ops::detail::Mxfp8Transform::GeluTanh);
                mxfp8(fp8->fc2, block.fc2_bias, true, x);
            } else if (!fused_residual(up, *block.fc2, *block.fc2_bias, x, stream)) {
                ops::linear(up, *block.fc2, down, stream);
                ops::add_bias(*block.fc2_bias, down, stream);
                ops::residual_add(down, x, stream);
            }
        }
        if (deepstack_index < deepstack_.size() &&
            deepstack_[deepstack_index].first == static_cast<std::int32_t>(layer)) {
            const auto& merger = deepstack_[deepstack_index].second;
            Tensor merged = x.view({g.merger_hidden(), tokens});
            Tensor norm = layout.normalized.bind(backing).view({g.merger_hidden(), tokens});
            ops::layer_norm(merged, *merger.norm_weight, *merger.norm_bias, g.norm_epsilon, norm, stream);
            Tensor hidden = layout.merger_hidden.bind(backing);
            ops::linear(norm, *merger.fc1, hidden, stream);
            ops::add_bias(*merger.fc1_bias, hidden, stream);
            ops::gelu(hidden, ops::GeluMode::Exact, stream);
            Tensor features = output.slice(2, static_cast<int>(++deepstack_index), 1);
            ops::linear(hidden, *merger.fc2, features, stream);
            ops::add_bias(*merger.fc2_bias, features, stream);
        }
        if (++state.layer == blocks_.size()) { state.phase = VisionEncodeState::Phase::Projection; }
        return false;
    }

    Tensor normalized = layout.normalized.bind(backing);
    ops::layer_norm(x, g.siglip2 ? *post_norm_weight_ : *merger_.norm_weight,
                    g.siglip2 ? *post_norm_bias_ : *merger_.norm_bias, g.norm_epsilon, normalized,
                    stream);
    Tensor merged = normalized.view({g.merger_hidden(), tokens});
    if (g.siglip2 && g.projector_norm) {
        Tensor norm = layout.projector_norm.bind(backing);
        ops::layer_norm(merged, *merger_.norm_weight, *merger_.norm_bias, 1.0e-5F, norm, stream);
        merged = norm;
    }
    Tensor hidden = layout.merger_hidden.bind(backing);
    ops::linear(merged, *merger_.fc1, hidden, stream);
    ops::add_bias(*merger_.fc1_bias, hidden, stream);
    ops::gelu(hidden, ops::GeluMode::Exact, stream);
    Tensor final_output = output.slice(2, 0, 1);
    ops::linear(hidden, *merger_.fc2, final_output, stream);
    ops::add_bias(*merger_.fc2_bias, final_output, stream);
    if (probe_) { probe_("vision_projection", final_output.slice(1, 0, std::min(tokens, 64)),
                        g.layers, stream); }
    state.phase = VisionEncodeState::Phase::Complete;
    return true;
}

} // namespace sinfer::family
