// Inkling-2-Mini loaded model (lane e1): the 48-visit served geometry, the ModelView whose 48 FullLayer entries
// alias the 24 physical layers lane b binds, and the load-time query arithmetic (ARTIFACT.md 5).

#include "targets/inkling2/impl/model.h"
#include "targets/inkling2/impl/visits.h"

#include "artifact/reader.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::targets::inkling2::detail {
namespace {

using C = TextConfig;

std::vector<float> download_f32(const Tensor& t) {
    if (t.data == nullptr || t.dtype != DType::FP32) { throw std::logic_error("inkling2: a query tensor is missing"); }
    std::vector<float> host(static_cast<std::size_t>(t.numel()));
    CUDA_CHECK(cudaMemcpy(host.data(), t.data, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return host;
}

/// fp32(x) * fp32(2304^-0.5), the reference's `copy_(w).mul_(hidden ** -0.5)` on an fp32 parameter.
void scale_queries(std::vector<float>& values) {
    const float scale = static_cast<float>(1.0 / 48.0); // 2304 ** -0.5 = 1/48, rounded to fp32
    for (float& x : values) { x = x * scale; }
}

void derive_queries(const load::InklingGlobals& g, DerivedQueries& out) {
    constexpr int D = C::hidden;
    const int cores = C::core_end - C::core_start;
    std::vector<float> host;
    struct Slice {
        Tensor* target;
        std::size_t offset;
        int rows;
    };
    std::vector<Slice> slices;
    const auto append = [&](Tensor* target, std::vector<float> values, int rows) {
        scale_queries(values);
        slices.push_back({target, host.size(), rows});
        host.insert(host.end(), values.begin(), values.end());
    };
    for (int layer = 1; layer < C::physical_layers; ++layer) {
        append(&out.attn[layer], download_f32(g.attn_query.at(layer)), g.attn_query.at(layer).ne[1]);
    }
    for (int layer = 0; layer < C::physical_layers; ++layer) {
        append(&out.mlp[layer], download_f32(g.mlp_query.at(layer)), 1);
    }
    append(&out.output, download_f32(g.output_query), 1);
    const std::vector<float> cross = download_f32(g.cross_pass_queries);
    const std::vector<float> kv    = download_f32(g.cross_pass_kv_queries);
    if (cross.size() != static_cast<std::size_t>(2 * cores) * D || kv.size() != cross.size()) {
        throw std::logic_error("inkling2: cross-pass queries have an unexpected shape");
    }
    for (int j = 0; j < cores; ++j) {
        // (attn, k, v, mlp) = (cross[2j], kv[2j], kv[2j + 1], cross[2j + 1]) (loop.py process_weights_after_loading)
        std::vector<float> m;
        m.reserve(static_cast<std::size_t>(4) * D);
        const auto row = [&](const std::vector<float>& src, int r) {
            m.insert(m.end(), src.begin() + static_cast<std::ptrdiff_t>(r) * D, src.begin() + static_cast<std::ptrdiff_t>(r + 1) * D);
        };
        row(cross, 2 * j);
        row(kv, 2 * j);
        row(kv, 2 * j + 1);
        row(cross, 2 * j + 1);
        append(&out.message[j], std::move(m), 4);
    }
    out.storage = DeviceBuffer(host.size() * sizeof(float));
    out.storage.copy_from_host(host.data(), host.size() * sizeof(float));
    for (const Slice& s : slices) {
        *s.target = Tensor(static_cast<float*>(out.storage.p) + s.offset, DType::FP32, {D, s.rows});
    }
}

} // namespace

family::TextGeometry visit_geometry(const artifact::Reader& reader) {
    std::map<std::string, double> values = reader.geometry();
    const auto layers = values.find("layers");
    if (layers == values.end() || static_cast<int>(layers->second) != C::physical_layers) {
        throw std::invalid_argument("inkling2: the artifact's geometry does not declare 24 physical layers");
    }
    values["layers"]   = C::layers;
    values["residual"] = C::residual;
    std::vector<std::string> types(C::layers);
    for (int v = 0; v < C::layers; ++v) { types[v] = kVisits[v].global ? "full_attention" : "sliding_attention"; }
    family::TextGeometry g = family::TextGeometry::resolved_moe(values, types);
    if (g.hidden != C::hidden || g.query_heads != C::query_heads || g.kv_heads != C::kv_heads ||
        g.head_dim != C::head_dim || g.sliding_window != C::window || g.output_rows != C::output_rows ||
        g.token_domain != C::token_domain || g.experts != C::experts || g.experts_per_token != C::experts_per_token) {
        throw std::invalid_argument("inkling2: the artifact's geometry is not Inkling-2-Mini's");
    }
    return g;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    // Lane b binds the text-only set (no pixel configs): take exactly that set.
    frontend = family::take_text_only_frontend_resources(backing, plan.plan.frontend);
    weights  = load::load_weights(plan.plan, backing);
    derive_queries(weights.globals, queries);
    model.weights = &weights;
    model.queries = &queries;

    runtime.geometry        = plan.geometry;
    runtime.weights_arena   = &backing.device_arena();
    runtime.features        = plan.features;
    runtime.token_embedding = weights.globals.embedding;
    runtime.final_norm      = weights.globals.final_norm;
    runtime.output_head     = weights.globals.output_head;
    runtime.full_layers.resize(static_cast<std::size_t>(C::layers));
    for (int v = 0; v < C::layers; ++v) {
        const load::InklingLayerWeights& layer = weights.layers.at(static_cast<std::size_t>(kVisits[v].layer));
        auto& f              = runtime.full_layers[static_cast<std::size_t>(v)];
        f.projection         = VisitPayload{&layer, &model, v};
        f.post_mixer         = MoePayload{&layer, v};
        // Aliases of the physical layer (same device pointers): an adapter registered on a weight applies in
        // every pass of a core layer.
        f.input_norm          = layer.attn_input_norm;
        f.output              = layer.output;
        f.post_attention_norm = layer.moe_input_norm;
    }
}

} // namespace sinfer::targets::inkling2::detail
