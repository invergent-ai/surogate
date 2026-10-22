// The CUDA Graph allowance a sequence plan reserves against the decode graphs the program then
// captures (ProgramImplCore::prepare_graphs), for an engine armed for adapters.
//
// Such an engine runs base rounds (ops::ScopedLoraBaseRound, #262): a round that selects no
// adapter takes the base routes, so every decode graph family is captured twice -- the
// adapter-capable routes and the base routes beside them. The ordinary (non-speculative)
// allowance is a bound per lane for one flavor's graphs, so it is counted once per flavor.
// Counted once, it refused the 35B-A3B NVFP4 with a rank-16 adapter and no draft head
// (2026-10-22): "CUDA Graph preparation consumed 735051776 bytes, exceeding the planned
// allowance of 541065216 bytes" at 43 lanes, 537919488 against 402653184 at 32. An engine
// without adapters and an MTP engine with them (whose per-class bound already held both
// flavors: 289 MiB against 2.56 GiB at 32 lanes) plan exactly what they planned before.
//
// Released Qwen3.6-35B-A3B dimensions (qwen3_5_moe), planned as the groupwise-int profile: this
// target's ordinary bound is the family's 12 MiB a lane for every profile, and that one plans on
// any device. The planner needs no weights; the device is for its DeviceContext only.
#include "targets/qwen3_5_moe/impl/variant.h"
#include "api/ops/batch_invariant.h"
#include "core/device.h"

#define SINFER_FAMILY_VARIANT ::sinfer::targets::qwen3_5_moe::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS qwen3_5_moe_runtime
#include "family/impl/runtime/layouts.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using V = targets::qwen3_5_moe::detail::Variant;

namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;

family::TextGeometry geometry() {
    family::TextGeometry g;
    g.hidden = g.residual = 2048; g.layers = 40;
    g.intermediate = g.shared_intermediate = 512;
    g.experts = 256; g.experts_per_token = 8; g.routed_scale = 1;
    g.output_rows = 248320; g.token_domain = 248077;
    g.query_heads = 16; g.kv_heads = 2; g.head_dim = 256; g.rotary_dim = 64;
    g.gdn_key_heads = 16; g.gdn_value_heads = 32;
    g.gdn_key_head_dim = g.gdn_value_head_dim = 128; g.gdn_conv_kernel = 4;
    g.mtp_layers = 1; g.max_context = 262144;
    g.rms_epsilon = 1e-6F; g.rope_theta = 1e7F;
    g.attention_scale = 0.0625F; g.gdn_scale = 0.0883883476F;
    std::vector<std::string> layers(g.layers, "linear_attention");
    for (int i = 3; i < g.layers; i += 4) { layers[i] = "full_attention"; }
    g.apply_layer_types(layers);
    const auto matrix = [&](const std::string& name, int rows, int columns) {
        g.linear_storage[name] = {rows, columns, {QType::W8G32_F16S}};
    };
    for (int i = 0; i <= g.layers; ++i) {
        const auto prefix = i == g.layers ? std::string("mtp/layer/")
                                          : "text/layers/" + std::to_string(i) + "/";
        if (i == g.layers || g.layer_attends(i)) {
            matrix(prefix + "attention/query_key_gate_value", 2 * g.query_size() + 2 * g.kv_size(), g.hidden);
            matrix(prefix + "attention/output", g.hidden, g.query_size());
        } else {
            matrix(prefix + "gdn/query_key_value_z", g.convolution_dim() + g.value_dim(), g.hidden);
            matrix(prefix + "gdn/output", g.hidden, g.value_dim());
        }
        matrix(prefix + "moe/routed_gate_up", 2 * g.experts * g.intermediate, g.hidden);
        matrix(prefix + "moe/routed_down", g.experts * g.hidden, g.intermediate);
    }
    return g;
}

struct Planned {
    std::size_t graph_allowance = 0;
    std::size_t reservation     = 0;
    bool base_rounds            = false;
};

// The plan of the serving command that refused: 40,960-token context, 4,096-token prompt chunks,
// FP8 KV, an adapter slot of rank 16 when `adapters`, MTP 2 verifying at any width when `mtp`.
Planned plan(DeviceContext& device, const family::TextGeometry& g, std::uint32_t lanes,
             bool adapters, bool mtp, int stage_last = 0) {
    EngineOptions options;
    options.max_context     = 40960;
    options.kv_capacity     = KvCapacityPolicy::explicit_capacity(300000);
    options.prefill_chunk   = 4096;
    options.max_concurrency = lanes;
    options.kv_cache        = KvCacheStorage::Fp8E4M3;
    options.use_cuda_graph  = true;
    options.lora_enable     = adapters;
    options.lora_slots      = adapters ? 1 : 0;
    options.lora_max_rank   = adapters ? 16 : 0;
    if (mtp) {
        options.speculative = {.backend = SpeculativeBackend::Mtp, .draft_tokens = 2,
                               .max_lanes = kSpeculateAtAnyWidth};
    }
    if (stage_last != 0) {
        options.pipeline_stage_first      = 0;
        options.pipeline_stage_last       = stage_last;
        options.pipeline_boundary_columns = options.prefill_chunk + lanes + 128;
    }
    auto planner = family::make_sequence_planner<V>(device, options,
                                                    V::WeightsProfile::GroupwiseInt, g, {});
    const std::uint32_t pages = planner.capacity_curve().minimum_main_page_groups;
    auto finished = std::move(planner).finalize(pages);
    return {finished.impl_->graph_allowance_bytes, finished.device_reservation_bytes(),
            finished.impl_->base_rounds};
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    DeviceContext device(0);
    const auto g = geometry();

    // What the engine measured when it refused, by lane count, and the allowance it refused
    // against: the plan must now hold the graphs it captures.
    struct Refusal {
        std::uint32_t lanes;
        std::size_t consumed;
        std::size_t planned;
    };
    for (const Refusal refusal : {Refusal{43, 735051776, 541065216}, Refusal{32, 537919488, 402653184}}) {
        const std::uint32_t lanes = refusal.lanes;
        const std::size_t one_flavor = 12 * kMiB * lanes;

        const Planned base = plan(device, g, lanes, false, false);
        const Planned armed = plan(device, g, lanes, true, false);
        std::cout << "lanes=" << lanes << " ordinary: without adapters " << base.graph_allowance
                  << ", armed " << armed.graph_allowance << " (measured " << refusal.consumed
                  << ")\n";
        // Without adapters: one flavor, the allowance it always had -- which was all the armed
        // engine was given too.
        assert(!base.base_rounds);
        assert(base.graph_allowance == one_flavor);
        assert(base.graph_allowance == refusal.planned);
        // Armed: base rounds, so both flavors, and the graphs the refusing engine measured fit.
        assert(armed.base_rounds);
        assert(armed.graph_allowance == 2 * one_flavor);
        assert(armed.graph_allowance >= refusal.consumed);
        // The reservation automatic KV sizing reads carries the second flavor.
        assert(armed.reservation >= base.reservation + one_flavor);

        // MTP: its per-class bound holds both flavors already; adapters do not move it.
        const Planned mtp_base  = plan(device, g, lanes, false, true);
        const Planned mtp_armed = plan(device, g, lanes, true, true);
        std::cout << "lanes=" << lanes << " MTP: without adapters " << mtp_base.graph_allowance
                  << ", armed " << mtp_armed.graph_allowance << '\n';
        assert(!mtp_base.base_rounds && mtp_armed.base_rounds);
        assert(mtp_armed.graph_allowance == mtp_base.graph_allowance);
        if (lanes == 32) { assert(mtp_armed.graph_allowance >= 289 * kMiB); } // measured, armed

        // Where the program takes no base rounds it captures one flavor, and the plan says so:
        // a pipeline stage...
        const Planned stage = plan(device, g, lanes, true, false, g.layers / 2);
        assert(!stage.base_rounds);
        assert(stage.graph_allowance == one_flavor);
        // ...and batch-invariant numerics.
        ops::set_batch_invariant(true);
        const Planned invariant = plan(device, g, lanes, true, false);
        ops::set_batch_invariant(false);
        assert(!invariant.base_rounds);
        assert(invariant.graph_allowance == one_flavor);
    }
    std::cout << "ok\n";
    return 0;
}
