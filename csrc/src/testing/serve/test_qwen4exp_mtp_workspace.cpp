#include "targets/qwen4exp/impl/variant.h"
#include "core/device.h"
#include "api/ops/sparse_moe.h"

#define SINFER_FAMILY_VARIANT ::sinfer::targets::qwen4exp::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS qwen4exp_runtime
#include "family/impl/runtime/layouts.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using V = targets::qwen4exp::detail::Variant;

// Released checkpoint dimensions, with PLE disabled: this test needs neither weights
// nor its large embedding table to exercise the sequence workspace planner.
family::TextGeometry geometry() {
    family::TextGeometry g;
    g.hidden = 2560; g.residual = 10240; g.layers = 48;
    g.hc_streams = 4; g.hc_low_rank = 320;
    g.intermediate = g.shared_intermediate = 640;
    g.output_rows = 248320; g.token_domain = 248320;
    g.query_heads = 24; g.kv_heads = 2; g.head_dim = 256; g.rotary_dim = 64;
    g.gdn_key_heads = 16; g.gdn_value_heads = 48;
    g.gdn_key_head_dim = g.gdn_value_head_dim = 128; g.gdn_conv_kernel = 4;
    g.experts = 512; g.experts_per_token = 10; g.routed_scale = 1;
    g.mtp_layers = 1; g.max_context = 262144;
    g.indexer_head_dim = 128; g.indexer_heads = 4; g.indexer_block = 4; g.indexer_top_k = 2048;
    g.rms_epsilon = 1e-6F; g.rope_theta = 1e7F;
    g.attention_scale = 0.0625F; g.gdn_scale = 0.0883883476F;
    std::vector<std::string> layers(g.layers, "linear_attention");
    for (int i = 3; i < g.layers; i += 4) { layers[i] = "full_attention"; }
    g.apply_layer_types(layers);
    return g;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    DeviceContext device(0);
    const auto g = geometry();
    for (int chunk : {128, 2048, 4096}) {
        for (int drafts : {1, 3, 5}) {
            EngineOptions options;
            options.max_context = g.max_context;
            options.kv_capacity.explicit_tokens = g.max_context;
            options.prefill_chunk = chunk;
            options.max_concurrency = 4;
            options.use_cuda_graph = false;
            options.speculative = {.backend = SpeculativeBackend::Mtp,
                                   .draft_tokens = static_cast<unsigned>(drafts)};
            auto planner = family::make_sequence_planner<V>(
                device, options, V::WeightsProfile::W8HyperConnection, g, {});
            auto plan = std::move(planner).finalize(g.max_context / kPagedKVPageSize);

            // The input fold still spans the prompt: trunk residual and collapsed
            // view, next-token embeddings, and the head's wide residual remain live.
            const std::size_t live = std::size_t(chunk) * sizeof(std::uint16_t) *
                (2 * g.residual + 2 * g.hidden);
            const auto needed = live + V::mtp_fold_workspace_capacity_bytes(g, 1, chunk);
            const auto planned = plan.impl_->workspace.mtp_prefill;
            std::cout << "chunk=" << chunk << " drafts=" << drafts
                      << " prefill=" << planned << " required>=" << needed << '\n';
            assert(planned >= needed);
            assert(plan.workspace_capacity_bytes() >= planned);
            // No full-prompt draft output or expert workspace remains.
            if (chunk == 2048) { assert(planned < 389648640); }
            // Alignment and autoregressive draft calls need the same block at B*(K+1).
            const int columns = options.max_concurrency * (drafts + 1);
            const auto round_moe = ops::sparse_moe_workspace_capacity_bytes(
                targets::qwen4exp::detail::moe_geometry(g), QType::W8G32_F16S,
                QType::W8G32_F16S, columns, columns);
            assert(plan.impl_->workspace.mtp_round >= round_moe +
                std::size_t(columns) * sizeof(std::uint16_t) * (g.residual + 2 * g.hidden));
        }
    }
}
