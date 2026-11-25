#include "family/inkling_vision.h"
#include "core/layout.h"
#include "api/ops/gelu.h"
#include "api/ops/layer_norm.h"
#include "api/ops/linear.h"
#include "api/ops/linear_bias.h"
#include "api/ops/vision_attention.h"
#include "api/ops/vision_pos_embed.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sinfer::family {
namespace {
struct Layout {
    TensorRegion x, a, b, q, k, v, ff, patches, positions, indices, values, projected;
    std::size_t bytes;

    Layout(const VisionGeometry& g, std::size_t tokens) {
        if (!tokens || tokens > static_cast<std::size_t>(g.max_image_tokens) ||
            tokens > static_cast<std::size_t>(std::numeric_limits<int>::max() - 1) / g.merge_unit()) {
            throw std::invalid_argument("Inkling-2 image exceeds its token budget");
        }
        const int p = static_cast<int>(tokens) * g.merge_unit(), n = p + 1;
        LayoutBuilder builder;
        const auto add = [&](DType type, std::initializer_list<int> shape) {
            return builder.add_tensor(type, shape, kVisionWorkspaceAlignment, "Inkling-2 vision");
        };
        const auto bf = [&](int h, int t) { return add(DType::BF16, {h, t}); };
        x         = bf(g.hidden, n);  // the standalone tower's residual; the prefill session hands its own
        a         = bf(g.hidden, n);
        b         = bf(g.hidden, n);
        positions = add(DType::I32, {n, 2});
        {
            auto phase = builder.scope();
            patches    = bf(g.patch_dim, p);
            indices    = add(DType::I32, {4, p});
            values     = add(DType::FP32, {4, p});
        }
        {
            auto phase = builder.scope();
            q          = bf(g.hidden, n);
            k          = bf(g.hidden, n);
            v          = bf(g.hidden, n);
        }
        {
            auto phase = builder.scope();
            ff         = bf(g.intermediate, n);
        }
        {
            auto phase = builder.scope();
            projected  = bf(g.projector_hidden, static_cast<int>(tokens));
        }
        bytes = builder.finish(kVisionWorkspaceAlignment);
    }
};

/// SUROGATE_SERVE_INKLING_VISION_DUMP=<dir> (parity work only): each encoded image's input patches and projected
/// output, as <dir>/image_<n>.bin = int32 {grid height, grid width, patches, tokens, patch_dim, output_hidden}, then the
/// BF16 patches [patches][patch_dim] (merge order) and the BF16 output [tokens][output_hidden]. Synchronises the stream.
void dump_image(const VisionGeometry& g, const VisionItemView& item, const Tensor& output, int tokens,
                cudaStream_t stream) {
    static const char* dir = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_INKLING_VISION_DUMP");
        return raw != nullptr && *raw != '\0' ? raw : nullptr;
    }();
    if (dir == nullptr) { return; }
    static std::atomic<int> counter{0};
    const auto& control = *item.control;
    std::vector<std::uint16_t> out(static_cast<std::size_t>(tokens) * g.output_hidden);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), output.data, out.size() * 2, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::string path = std::string(dir) + "/image_" + std::to_string(counter++) + ".bin";
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { return; }
    const std::int32_t header[6] = {control.grid.height, control.grid.width, static_cast<std::int32_t>(control.patch_count),
                                    tokens, g.patch_dim, g.output_hidden};
    std::fwrite(header, sizeof(header), 1, file);
    std::fwrite(item.patches.data(), 2, item.patches.size(), file);
    std::fwrite(out.data(), 2, out.size(), file);
    std::fclose(file);
}

void upload(const void* source, Tensor& destination, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(), cudaMemcpyHostToDevice, stream));
}
} // namespace

std::size_t inkling_vision_workspace_bytes(const VisionGeometry& g, std::size_t tokens) {
    return Layout(g, tokens).bytes;
}

bool encode_inkling_vision_step(const VisionGeometry& g, const VisionWeights& weights, const VisionItemView& item,
                                Tensor& output, WorkspaceArena& workspace, cudaStream_t stream,
                                const VisionContext::Probe& probe, VisionEncodeState& state) {
    const auto& control = *item.control;
    if (control.segment_count != 1 || control.grid.temporal != 1 ||
        control.patch_count != control.merged_count * static_cast<std::size_t>(g.merge_unit()) ||
        control.position_ids.size() != control.patch_count * 2) {
        throw std::invalid_argument("Inkling-2 encodes one still image at a time");
    }
    const int p = static_cast<int>(control.patch_count), n = p + 1, h = g.hidden;
    const int tokens = static_cast<int>(control.merged_count);
    const Layout layout(g, control.merged_count);
    workspace.reset();
    const auto storage = workspace.alloc_bytes(layout.bytes, kVisionWorkspaceAlignment);
    // The residual [hidden, P + 1]: the patches in merge order, then the CLS column.
    Tensor x = state.residual.data ? state.residual : layout.x.bind(storage);
    if (x.dtype != DType::BF16 || x.ne[0] != h || x.ne[1] != n || !x.is_contiguous()) {
        throw std::invalid_argument("Inkling-2 vision residual must be BF16 [hidden, patches + 1]");
    }
    Tensor a = layout.a.bind(storage), b = layout.b.bind(storage);
    const auto tensor = [&](const std::string& name) -> const Tensor& { return weights.extra_tensors.at(name); };
    const auto linear = [&](const std::string& name, const Tensor& input, Tensor& out, bool bias) {
        if (bias) {
            ops::linear_bias(input, weights.extra_linears.at(name), tensor(name + "_bias"), out, stream);
        } else {
            ops::linear(input, weights.extra_linears.at(name), out, stream);
        }
    };
    const auto norm = [&](const std::string& name, const Tensor& input, Tensor& out) {
        ops::layer_norm(input, tensor(name + "/weight"), tensor(name + "/bias"), g.norm_epsilon, out, stream);
    };
    // Native (y, x) of each patch row, in the payload's merge order (build_vision_control).
    const int gh = control.grid.height, gw = control.grid.width;
    const auto row_y = [&](int i) { return control.position_ids[static_cast<std::size_t>(i)]; };
    const auto row_x = [&](int i) { return control.position_ids[static_cast<std::size_t>(p + i)]; };

    if (state.phase == VisionEncodeState::Phase::Embedding) {
        Tensor patches = layout.patches.bind(storage);
        upload(item.patches.data(), patches, stream);
        Tensor patch_columns = a.slice(1, 0, p);
        linear("patch_embedding", patches, patch_columns, false);
        // The 32 x 32 position grid resized to the image's native grid: align_corners=False bilinear with the
        // source coordinate clamped to the grid (vLLM's _source_coords), so every tap is inside it.
        const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(g.position_embeddings))));
        if (side * side != g.position_embeddings) {
            throw std::invalid_argument("Inkling-2 position grid is not square");
        }
        const auto source = [side](int index, int target, int& lo, int& hi, float& w_lo, float& w_hi) {
            float coord = static_cast<float>(side) * (static_cast<float>(index) + 0.5F) / static_cast<float>(target) - 0.5F;
            coord       = std::max(0.0F, std::min(coord, static_cast<float>(side - 1)));
            lo          = static_cast<int>(std::floor(coord));
            hi          = std::min(lo + 1, side - 1);
            w_hi        = coord - static_cast<float>(lo);
            w_lo        = 1.0F - w_hi;
        };
        std::vector<int> indices(static_cast<std::size_t>(p) * 4);
        std::vector<float> values(static_cast<std::size_t>(p) * 4);
        for (int i = 0; i < p; ++i) {
            int y0, y1, x0, x1;
            float wy0, wy1, wx0, wx1;
            source(row_y(i), gh, y0, y1, wy0, wy1);
            source(row_x(i), gw, x0, x1, wx0, wx1);
            const int tap = i * 4;
            indices[tap + 0] = y0 * side + x0;
            indices[tap + 1] = y0 * side + x1;
            indices[tap + 2] = y1 * side + x0;
            indices[tap + 3] = y1 * side + x1;
            values[tap + 0]  = wy0 * wx0;
            values[tap + 1]  = wy0 * wx1;
            values[tap + 2]  = wy1 * wx0;
            values[tap + 3]  = wy1 * wx1;
        }
        Tensor idx = layout.indices.bind(storage), val = layout.values.bind(storage);
        upload(indices.data(), idx, stream);
        upload(values.data(), val, stream);
        ops::vision_pos_embed_add(tensor("position_embedding"), idx, val, patch_columns, stream);
        Tensor cls = a.slice(1, p, 1);
        inkling_vision_cls(tensor("class_embedding"), tensor("cls_position_embedding"), cls, stream);
        norm("pre_norm", a, x);
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
        Tensor qh = q.view({g.head_dim(), g.heads, n}), kh = k.view({g.head_dim(), g.heads, n}),
               vh = v.view({g.head_dim(), g.heads, n}), attended = b.view({g.head_dim(), g.heads, n});
        // RoPE positions: x + 1 on the first half of the pairs, y + 1 on the second; the CLS row turns by 0.
        std::vector<int> positions(static_cast<std::size_t>(n) * 2, 0);
        for (int i = 0; i < p; ++i) {
            positions[static_cast<std::size_t>(i)]     = row_x(i) + 1;
            positions[static_cast<std::size_t>(n + i)] = row_y(i) + 1;
        }
        Tensor pos = layout.positions.bind(storage);
        upload(positions.data(), pos, stream);
        inkling_vision_rope(pos, qh, kh, stream);
        ops::vision_attention(qh, kh, vh, n, attended, stream);
        linear(prefix + "attention/output", b, a, true);
        inkling_vision_scaled_residual(a, tensor(prefix + "ls1"), x, stream);
        norm(prefix + "norm2", x, a);
        Tensor ff = layout.ff.bind(storage);
        linear(prefix + "mlp/fc1", a, ff, true);
        ops::gelu(ff, ops::GeluMode::Exact, stream);
        linear(prefix + "mlp/fc2", ff, a, true);
        inkling_vision_scaled_residual(a, tensor(prefix + "ls2"), x, stream);
        if (++state.layer == static_cast<std::size_t>(g.layers)) { state.phase = VisionEncodeState::Phase::Projection; }
        return false;
    }
    // No final norm: the merge pools the last block's residual, 4 consecutive patch columns per visual token.
    const Tensor pooled(x.data, DType::BF16, {g.merger_hidden(), tokens});
    Tensor projected = layout.projected.bind(storage);
    linear("projector/in", pooled, projected, false);
    ops::gelu(projected, ops::GeluMode::Exact, stream);
    Tensor final_output(output.data, DType::BF16, {g.output_hidden, tokens});
    linear("projector/out", projected, final_output, false);
    if (probe) { probe("vision_projection", final_output.slice(1, 0, std::min(tokens, 64)), g.layers, stream); }
    dump_image(g, item, final_output, tokens, stream);
    state.phase = VisionEncodeState::Phase::Complete;
    return true;
}
} // namespace sinfer::family
