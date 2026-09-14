#include "family/muse_vision.h"
#include "core/layout.h"
#include "api/ops/add_bias.h"
#include "api/ops/gelu.h"
#include "api/ops/layer_norm.h"
#include "api/ops/linear.h"
#include "api/ops/linear_bias.h"
#include "api/ops/residual_add.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/rope.h"
#include "api/ops/scatter.h"
#include "api/ops/vision_attention.h"
#include "api/ops/vision_pos_embed.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace sinfer::family {
namespace {
struct Layout {
    TensorRegion x, a, b, q, k, v, ff, patches, positions, indices, values, permutation, cu,
        shuffled, projected_a, projected_b, final;
    std::size_t bytes;

    Layout(const VisionGeometry& g, std::size_t tokens) {
        if (!tokens || tokens > static_cast<std::size_t>(g.max_image_tokens) ||
            tokens > static_cast<std::size_t>(std::numeric_limits<int>::max()) / g.merge_unit()) {
            throw std::invalid_argument("Muse-Glimmer image exceeds its token budget");
        }
        const int p = tokens * g.merge_unit();
        LayoutBuilder builder;
        const auto add = [&](DType type, std::initializer_list<int> shape) {
            return builder.add_tensor(type, shape, kVisionWorkspaceAlignment,
                                      "Muse-Glimmer vision");
        };
        const auto bf = [&](int h, int t) { return add(DType::BF16, {h, t}); };
        x             = bf(g.hidden, p);
        a             = bf(g.hidden, p);
        b             = bf(g.hidden, p);
        positions     = add(DType::I32, {p, 2});
        permutation   = add(DType::I32, {p});
        // P+1 covers any legal window partition, including very narrow images.
        cu = add(DType::I32, {p + 1});
        {
            auto phase = builder.scope();
            patches    = bf(g.patch_dim, p);
            indices    = add(DType::I32, {4, p});
            values     = add(DType::FP32, {4, p});
        }
        {
            auto phase = builder.scope();
            q          = bf(g.hidden, p);
            k          = bf(g.hidden, p);
            v          = bf(g.hidden, p);
        }
        {
            auto phase = builder.scope();
            ff         = bf(g.intermediate, p);
        }
        {
            auto phase  = builder.scope();
            shuffled    = bf(g.merger_hidden(), tokens);
            projected_a = bf(g.projector_hidden, tokens);
            projected_b = bf(g.projector_hidden, tokens);
            final       = bf(g.output_hidden, tokens);
        }
        bytes = builder.finish(kVisionWorkspaceAlignment);
        bytes += ops::vision_attention_workspace_capacity_bytes(p, p, 1, p);
    }
};

void upload(const void* source, Tensor& destination, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}
} // namespace

std::size_t muse_vision_workspace_bytes(const VisionGeometry& g, std::size_t tokens) {
    return Layout(g, tokens).bytes;
}

bool encode_muse_vision_step(const VisionGeometry& g, const VisionWeights& weights,
                             const VisionItemView& item, Tensor& output, WorkspaceArena& workspace,
                             cudaStream_t stream, const VisionContext::Probe& probe,
                             VisionEncodeState& state) {
    const auto& control = *item.control;
    if (control.segment_count != 1 || control.patch_count != control.merged_count * 4) {
        throw std::invalid_argument("Muse-Glimmer encodes one complete image at a time");
    }
    const int p = control.patch_count, h = g.hidden;
    const Layout layout(g, control.merged_count);
    workspace.reset();
    // Leave the packed-attention descriptor tail available to the Op's scoped allocator.
    const auto attention_bytes = ops::vision_attention_workspace_capacity_bytes(p, p, 1, p);
    const auto storage =
        workspace.alloc_bytes(layout.bytes - attention_bytes, kVisionWorkspaceAlignment);
    Tensor x = state.residual.data ? state.residual : layout.x.bind(storage);
    Tensor a = layout.a.bind(storage), b = layout.b.bind(storage);
    const auto tensor = [&](const std::string& name) -> const Tensor& {
        return weights.extra_tensors.at(name);
    };
    const auto linear = [&](const std::string& name, const Tensor& input, Tensor& out,
                            bool bias = false) {
        if (bias) {
            ops::linear_bias(input, weights.extra_linears.at(name), tensor(name + "_bias"), out,
                             stream);
        } else {
            ops::linear(input, weights.extra_linears.at(name), out, stream);
        }
    };
    const auto norm = [&](const std::string& name, const Tensor& input, Tensor& out) {
        ops::layer_norm(input, tensor(name + "/weight"), tensor(name + "/bias"), g.norm_epsilon,
                        out, stream);
    };
    const int gh = control.grid.height, gw = control.grid.width;
    const int side = static_cast<int>(std::sqrt(g.position_embeddings));
    std::vector<int> sparse_to_merged(p), merged_to_sparse(p), positions(p * 2), windows{0};
    int cursor = 0;
    for (int wy = 0; wy < gh; wy += side) {
        for (int wx = 0; wx < gw; wx += side) {
            for (int y = wy; y < std::min(wy + side, gh); ++y) {
                for (int col = wx; col < std::min(wx + side, gw); ++col) {
                    const int merged = ((y / 2) * (gw / 2) + col / 2) * 4 + (y % 2) * 2 + col % 2;
                    sparse_to_merged[cursor] = merged;
                    merged_to_sparse[merged] = cursor;
                    positions[cursor]        = col + 1;
                    positions[p + cursor]    = y + 1;
                    ++cursor;
                }
            }
            windows.push_back(cursor);
        }
    }
    Tensor permutation = layout.permutation.bind(storage);
    if (state.phase == VisionEncodeState::Phase::Embedding) {
        Tensor patches = layout.patches.bind(storage);
        upload(item.patches.data(), patches, stream);
        linear("patch_embedding", patches, a);
        // Half-pixel bilinear sampling with zero padding matches the published position grid.
        std::vector<int> indices(p * 4);
        std::vector<float> values(p * 4);
        for (int i = 0; i < p; ++i) {
            const float yf = (control.position_ids[i] + .5F) * side / gh - .5F;
            const float xf = (control.position_ids[p + i] + .5F) * side / gw - .5F;
            const int y0 = static_cast<int>(std::floor(yf)), x0 = static_cast<int>(std::floor(xf));
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    const int y = y0 + dy, col = x0 + dx, tap = i * 4 + dy * 2 + dx;
                    indices[tap] = std::clamp(y, 0, side - 1) * side + std::clamp(col, 0, side - 1);
                    values[tap] =
                        y < 0 || y >= side || col < 0 || col >= side
                            ? 0.F
                            : (dy ? yf - y0 : 1.F - (yf - y0)) * (dx ? xf - x0 : 1.F - (xf - x0));
                }
        }
        Tensor idx = layout.indices.bind(storage), val = layout.values.bind(storage);
        upload(indices.data(), idx, stream);
        upload(values.data(), val, stream);
        ops::vision_pos_embed_add(tensor("position_embedding"), idx, val, a, stream);
        norm("pre_norm", a, b);
        upload(merged_to_sparse.data(), permutation, stream);
        ops::scatter(b, permutation, x, stream);
        state.phase = VisionEncodeState::Phase::Blocks;
        return false;
    }
    if (state.phase == VisionEncodeState::Phase::Blocks) {
        const auto prefix = "layers/" + std::to_string(state.layer) + "/";
        Tensor q = layout.q.bind(storage), k = layout.k.bind(storage), v = layout.v.bind(storage);
        norm(prefix + "norm1", x, a);
        linear(prefix + "attention/query", a, q, true);
        linear(prefix + "attention/key", a, k, true);
        linear(prefix + "attention/value", a, v, true);
        Tensor qh = q.view({g.head_dim(), g.heads, p}), kh = k.view({g.head_dim(), g.heads, p}),
               vh       = v.view({g.head_dim(), g.heads, p}),
               attended = b.view({g.head_dim(), g.heads, p});
        Tensor pos      = layout.positions.bind(storage);
        upload(positions.data(), pos, stream);
        muse_vision_rope(pos, g.rope_theta, qh, kh, stream);
        if ((state.layer + 1) % 4 == 0 || state.layer + 1 == static_cast<std::size_t>(g.layers)) {
            ops::vision_attention(qh, kh, vh, p, attended, stream);
        } else {
            Tensor cu(layout.cu.bind(storage).data, DType::I32, {static_cast<int>(windows.size())});
            upload(windows.data(), cu, stream);
            ops::vision_attention(qh, kh, vh, cu, workspace, attended, stream);
        }
        linear(prefix + "attention/output", b, a, true);
        ops::residual_add(a, x, stream);
        norm(prefix + "norm2", x, a);
        Tensor ff = layout.ff.bind(storage);
        linear(prefix + "mlp/fc1", a, ff, true);
        ops::gelu(ff, ops::GeluMode::Exact, stream);
        linear(prefix + "mlp/fc2", ff, a, true);
        ops::residual_add(a, x, stream);
        if (++state.layer == static_cast<std::size_t>(g.layers)) {
            state.phase = VisionEncodeState::Phase::Projection;
        }
        return false;
    }
    norm("post_norm", x, a);
    upload(sparse_to_merged.data(), permutation, stream);
    ops::scatter(a, permutation, b, stream);
    Tensor shuffled = layout.shuffled.bind(storage), pa = layout.projected_a.bind(storage),
           pb = layout.projected_b.bind(storage);
    muse_pixel_shuffle(b, shuffled, stream);
    linear("projector/0", shuffled, pa);
    ops::gelu(pa, ops::GeluMode::Exact, stream);
    linear("projector/1", pa, pb);
    ops::gelu(pb, ops::GeluMode::Exact, stream);
    Tensor final = layout.final.bind(storage);
    linear("projector/2", pb, final);
    ops::rmsnorm_unweighted(final, 1.e-5F, output, stream);
    if (probe) { probe("vision.projected", output, -1, stream); }
    state.phase = VisionEncodeState::Phase::Complete;
    return true;
}
} // namespace sinfer::family
