#include "family/gemma_vision.h"
#include "family/gemma_vision_ops.h"
#include "core/layout.h"
#include "api/ops/add_bias.h"
#include "api/ops/gelu.h"
#include "api/ops/gelu_mul.h"
#include "api/ops/layer_norm.h"
#include "api/ops/linear.h"
#include "api/ops/residual_add.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/vision_attention.h"
#include "api/ops/vision_pos_embed.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sinfer::family {
namespace {
struct Layout {
    TensorRegion x, patches, positions, indices, weights, a, b, q, k, v, qkv, gate, up, activated,
        clip, pooled, normalized;
    std::size_t bytes;

    Layout(const VisionGeometry& g, std::size_t tokens) {
        if (!tokens || tokens > static_cast<std::size_t>(g.max_image_tokens) ||
            tokens * g.merge_unit() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("Gemma vision image exceeds its token budget");
        }
        const int p = tokens * g.merge_unit();
        LayoutBuilder builder;
        const auto add = [&](DType type, std::initializer_list<int> shape) {
            return builder.add_tensor(type, shape, kVisionWorkspaceAlignment, "Gemma vision");
        };
        const auto bf = [&](int h, int t) { return add(DType::BF16, {h, t}); };
        x             = bf(g.hidden, p);
        positions     = add(DType::I32, {p, 2});
        if (g.gemma_version == 3) {
            indices = add(DType::I32, {4, p});
            weights = add(DType::FP32, {4, p});
        }
        a = bf(g.hidden, p);
        b = bf(g.hidden, p);
        // Shared clamp scratch also holds the unified encoder's normalized raw patches.
        clip = bf(std::max({g.hidden, g.intermediate, g.patch_dim}), p);
        {
            auto phase = builder.scope();
            patches    = bf(g.patch_dim, p);
        }
        if (!g.encoder_free) {
            {
                auto phase = builder.scope();
                q          = bf(g.hidden, p);
                k          = bf(g.hidden, p);
                v          = bf(g.hidden, p);
                if (g.gemma_version == 3) { qkv = bf(3 * g.hidden, p); }
            }
            {
                auto phase = builder.scope();
                gate       = bf(g.intermediate, p);
                up         = bf(g.intermediate, p);
                activated  = bf(g.intermediate, p);
            }
        }
        pooled     = bf(g.hidden, tokens);
        normalized = bf(g.hidden, tokens);
        bytes      = builder.finish(kVisionWorkspaceAlignment);
    }
};

void copy(const void* source, Tensor& destination, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}
} // namespace

std::size_t gemma_vision_workspace_bytes(const VisionGeometry& g, std::size_t tokens) {
    return Layout(g, tokens).bytes;
}

bool encode_gemma_vision_step(const VisionGeometry& g, const VisionWeights& weights,
                         const VisionItemView& item, Tensor& output, WorkspaceArena& workspace,
                         cudaStream_t stream, const VisionContext::Probe& probe, VisionEncodeState& state) {
    const auto& control = *item.control;
    if (control.segment_count != 1 ||
        control.patch_count != control.merged_count * g.merge_unit()) {
        throw std::invalid_argument(
            "Gemma vision encodes one complete image or video frame at a time");
    }
    const Layout layout(g, control.merged_count);
    workspace.reset();
    const auto storage = workspace.alloc_bytes(layout.bytes, kVisionWorkspaceAlignment);
    const int p = control.patch_count, h = g.hidden;
    const auto tensor = [&](const std::string& name) -> const Tensor& {
        return weights.extra_tensors.at(name);
    };
    Tensor x = state.residual.data ? state.residual : layout.x.bind(storage);
    Tensor a = layout.a.bind(storage), b = layout.b.bind(storage);
    Tensor clip_storage = layout.clip.bind(storage);
    const auto linear   = [&](const std::string& name, const Tensor& input, Tensor& out) {
        const auto bounds = weights.extra_tensors.find(name + "/clip");
        if (bounds == weights.extra_tensors.end()) {
            ops::linear(input, weights.extra_linears.at(name), out, stream);
        } else {
            Tensor clipped(clip_storage.data, DType::BF16, {input.ne[0], input.ne[1]});
            gemma_vision::clamp(input, bounds->second, 0, clipped, stream);
            ops::linear(clipped, weights.extra_linears.at(name), out, stream);
            gemma_vision::clamp(out, bounds->second, 2, out, stream);
        }
    };
    const auto ln = [&](const std::string& name, const Tensor& input, Tensor& out, float eps) {
        ops::layer_norm(input, tensor(name + "/weight"), tensor(name + "/bias"), eps, out, stream);
    };
    const auto rms = [&](const std::string& name, const Tensor& input, Tensor& out) {
        ops::rmsnorm(input, tensor(name), g.norm_epsilon, false, out, stream);
    };
    Tensor positions = layout.positions.bind(storage);
    copy(control.position_ids.data(), positions, stream);
    if (state.phase == VisionEncodeState::Phase::Embedding) {
        Tensor patches = layout.patches.bind(storage);
        copy(item.patches.data(), patches, stream);
        if (g.encoder_free) {
            Tensor normalized(clip_storage.data, DType::BF16, {g.patch_dim, p});
            ln("patch_norm1", patches, normalized, 1.e-5F);
            linear("patch_embedding", normalized, a);
            ops::add_bias(tensor("patch_embedding_bias"), a, stream);
            ln("patch_norm2", a, x, 1.e-5F);
        } else {
            linear("patch_embedding", patches, x);
            if (g.gemma_version == 3) { ops::add_bias(tensor("patch_embedding_bias"), x, stream); }
        }
        if (g.gemma_version == 3) {
            Tensor indices = layout.indices.bind(storage), values = layout.weights.bind(storage);
            copy(control.position_table_indices.data(), indices, stream);
            copy(control.position_table_weights.data(), values, stream);
            ops::vision_pos_embed_add(tensor("position_embedding"), indices, values, x, stream);
        } else {
            gemma_vision::position_add(tensor("position_embedding"), positions, x, stream);
        }
        state.phase = g.encoder_free ? VisionEncodeState::Phase::Projection : VisionEncodeState::Phase::Blocks;
        return false;
    }
    if (state.phase == VisionEncodeState::Phase::Blocks) {
        const int layer = static_cast<int>(state.layer);

        const auto prefix = "layers/" + std::to_string(layer) + "/";
        Tensor q = layout.q.bind(storage), k = layout.k.bind(storage),
               v  = layout.v.bind(storage);
        Tensor qh = q.view({g.head_dim(), g.heads, p}), kh = k.view({g.head_dim(), g.heads, p});
        Tensor vh = v.view({g.head_dim(), g.heads, p});
        if (g.gemma_version == 3) {
            ln(prefix + "norm1", x, a, g.norm_epsilon);
            Tensor qkv = layout.qkv.bind(storage);
            linear(prefix + "attention/qkv", a, qkv);
            ops::add_bias(tensor(prefix + "attention/qkv_bias"), qkv, stream);
            qh       = Tensor(qkv.data, DType::BF16, {g.head_dim(), g.heads, p});
            kh       = Tensor(static_cast<char*>(qkv.data) + h * 2, DType::BF16,
                              {g.head_dim(), g.heads, p});
            vh       = Tensor(static_cast<char*>(qkv.data) + h * 4, DType::BF16,
                              {g.head_dim(), g.heads, p});
            qh.nb[2] = kh.nb[2] = vh.nb[2] = qkv.nb[1];
        } else {
            rms(prefix + "input_norm", x, a);
            linear(prefix + "attention/query", a, b);
            Tensor bh = b.view({g.head_dim(), g.heads * p});
            Tensor qn = q.view({g.head_dim(), g.heads * p});
            rms(prefix + "attention/query_norm", bh, qn);
            linear(prefix + "attention/key", a, b);
            Tensor kn = k.view({g.head_dim(), g.heads * p});
            rms(prefix + "attention/key_norm", bh, kn);
            linear(prefix + "attention/value", a, b);
            Tensor vn = v.view({g.head_dim(), g.heads * p});
            ops::rmsnorm_unweighted(bh, g.norm_epsilon, vn, stream);
            gemma_vision::spatial_rope(positions, g.rope_theta, qh, kh, stream);
        }
        Tensor attended = a.view({g.head_dim(), g.heads, p});
        ops::vision_attention(qh, kh, vh, p, attended, stream,
                              g.gemma_version == 4 ? 1.0F : 0.0F);
        linear(prefix + "attention/output", a, b);
        if (g.gemma_version == 3) {
            ops::add_bias(tensor(prefix + "attention/output_bias"), b, stream);
            ops::residual_add(b, x, stream);
            ln(prefix + "norm2", x, a, g.norm_epsilon);
        } else {
            rms(prefix + "post_attention_norm", b, a);
            ops::residual_add(a, x, stream);
            rms(prefix + "pre_feedforward_norm", x, a);
        }
        Tensor gate = layout.gate.bind(storage), up = layout.up.bind(storage),
               activated = layout.activated.bind(storage);
        if (g.gemma_version == 3) {
            linear(prefix + "mlp/fc1", a, gate);
            ops::add_bias(tensor(prefix + "mlp/fc1_bias"), gate, stream);
            ops::gelu(gate, ops::GeluMode::Tanh, stream);
            linear(prefix + "mlp/fc2", gate, b);
            ops::add_bias(tensor(prefix + "mlp/fc2_bias"), b, stream);
            ops::residual_add(b, x, stream);
        } else {
            linear(prefix + "mlp/gate", a, gate);
            linear(prefix + "mlp/up", a, up);
            ops::gelu_mul(gate, up, ops::GeluMode::Tanh, activated, stream, true);
            linear(prefix + "mlp/down", activated, b);
            rms(prefix + "post_feedforward_norm", b, a);
            ops::residual_add(a, x, stream);
        }
        if (probe) {
            probe("gemma_vision_layer", x.slice(1, 0, std::min(p, 64)), layer, stream);
        }
        if (++state.layer == static_cast<std::size_t>(g.layers)) { state.phase = VisionEncodeState::Phase::Projection; }
        return false;
    }
    if (g.encoder_free) {
        ln("position_norm", x, a, 1.e-5F);
        ops::rmsnorm_unweighted(a, g.norm_epsilon, b, stream);
        Tensor final = output.view({g.output_hidden, p});
        linear("projection", b, final);
    } else {
        Tensor pooled = layout.pooled.bind(storage), normalized = layout.normalized.bind(storage);
        if (g.gemma_version == 3) {
            ln("post_norm", x, a, g.norm_epsilon);
            gemma_vision::pool(a, g.merge_unit(), 1.0F, nullptr, nullptr, pooled, stream);
            ops::rmsnorm(pooled, tensor("projection_norm"), g.norm_epsilon, true, normalized,
                         stream);
        } else {
            gemma_vision::pool(x, g.merge_unit(), std::sqrt(float(h)),
                               g.standardize ? &tensor("std_bias") : nullptr,
                               g.standardize ? &tensor("std_scale") : nullptr, pooled, stream);
            ops::rmsnorm_unweighted(pooled, g.norm_epsilon, normalized, stream);
        }
        Tensor final = output.view({g.output_hidden, static_cast<int>(control.merged_count)});
        linear("projection", normalized, final);
    }
    if (probe) {
        probe("vision_projection", output.slice(1, 0, std::min(output.ne[1], 64)), g.layers,
              stream);
    }
    state.phase = VisionEncodeState::Phase::Complete;
    return true;
}
} // namespace sinfer::family
