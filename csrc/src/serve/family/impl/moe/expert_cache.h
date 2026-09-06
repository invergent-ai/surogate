#pragma once

// The expert cache: a device pool of MoE expert slots fed from the pinned host bank, and a
// host worker pool that computes the misses the split hands it instead of fetching them over
// PCIe (design/INFERENCE.md phase 2, design/serve-engine-flash-next.md D3).
//
// Family machinery. A mixture is a mixture: which of its experts are resident, which cross the
// bus and which the CPU computes are questions about memory and a link, not about the model,
// so every target with routed experts in the host bank gets this by describing one layer as a
// `BankedMixture` and running its round through `ExpertCache::run`. It was Qwen3.8-Flash-Next's
// alone for a week, and GLM-5.3-Flash paid for that at 3.1 tok/s on a card that llama.cpp
// served at 19 by computing the same experts on the host.
//
// One cache per device, keyed by the mixture geometry; pipeline stages of one model share the
// host worker pool (one per process, or one per NUMA node with `--cpu-moe-pool-per-socket`).

#include "api/ops/cpu_expert_compute.h"
#include "api/ops/expert_slot_cache.h"
#include "api/ops/sparse_moe.h"
#include "api/types.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace sinfer::family {

/// One mixture layer whose routed experts live in the host bank, as the cache sees it.
struct BankedMixture {
    /// The layer's index in the model, which keys the cache's directory: every banked layer of
    /// one model must carry a distinct one below `layers`.
    std::int32_t layer  = -1;
    /// How many layers the directory spans (the model's, draft head included).
    std::int32_t layers = 0;
    /// The kernels' view: router and shared expert on the device, the routed pair over the
    /// bank's device-mapped alias. The cache swaps the routed pair for the pool's.
    const ops::SparseMoeWeights* op = nullptr;
    /// Which routed halves the bank holds as Q4G32AM planes (requantised on the way in; the
    /// pool decodes them). A half that is not is W8 planes or the file's blocks, as its
    /// `Weight` says. The halves are independent: a K_XL mixture keeps its 4-bit gate/up as
    /// 4-bit planes and its wider down as W8.
    bool host_gate_up_q4 = false;
    bool host_down_q4    = false;
    /// Host virtual addresses of the routed expert objects (the Weights above hold the mapped
    /// aliases); the CPU expert path reads the planes -- or the file's blocks -- through these.
    const std::byte* host_gate_up = nullptr;
    const std::byte* host_down    = nullptr;

    [[nodiscard]] bool banked() const noexcept {
        return op != nullptr && host_gate_up != nullptr && host_down != nullptr;
    }
};

class ExpertCache {
public:
    /// The cache of the current device, created on the first call for `geometry` over `layers`
    /// layers and returned thereafter. Disabled (and cheap) when no slot count was configured
    /// and none could be derived from free device memory; a second geometry on one device is
    /// refused, since the pool holds experts of one shape.
    [[nodiscard]] static ExpertCache& for_current_device(const ops::SparseMoeGeometry& geometry,
                                                         std::int32_t layers);

    /// Records, for the current device, what the run asked for: the slot count and the runtime
    /// floor an automatic pool must leave, the CPU split's shares and its minimum round width,
    /// and whether pipeline stages get a host pool per NUMA node. `mixture_experts` is the
    /// geometry's expert count, for the pipeline-stage rule that turns the split off when a
    /// stage's pool would hold nearly all of its experts. Call before the cache is created.
    static void configure(const EngineOptions& options, std::size_t runtime_floor_bytes,
                          std::int32_t mixture_experts);
    /// The registry's projection of what the runtime derives from the resident weights, stashed
    /// at plan time so an automatic pool can leave it.
    static void configure_derived_reserve(std::size_t bytes);
    [[nodiscard]] static std::size_t derived_reserve();
    /// What the load holds only while it runs (the materializer's staging), stashed at plan
    /// time: the pool exists before the load and must leave it room, but not on top of the
    /// runtime's own floor -- the two are never resident together, so a caller takes the larger.
    static void configure_load_staging(std::size_t bytes);
    [[nodiscard]] static std::size_t load_staging();

    [[nodiscard]] bool enabled() const noexcept;

    /// One mixture round: the routing is resolved against the directory, the misses are fetched
    /// into the pool or handed to the host, and the expert kernels run over the pool; the
    /// resident parts (router, shared expert) are the layer's own. `destination` is the layer's
    /// BF16 [hidden, tokens] output plane, added into. The host's partial, if the split ran,
    /// is left pending: the caller joins it with `add_pending_partial` (or the wait/finish pair
    /// when it fuses the add into its own combine) before it reads `destination`.
    void run(const BankedMixture& mixture, const Tensor& hidden, Tensor& destination,
             WorkspaceArena& workspace, cudaStream_t stream);

    /// With `--cpu-moe-share auto`, times a PCIe gather and a host round of one layer's experts
    /// outside any capture and sets the share to host / (host + PCIe). Call once after the
    /// weights are bound and before the decode graphs are captured; a no-op otherwise.
    void prepare_split(const BankedMixture& mixture);

    // --- the host partial of the round in flight ---
    [[nodiscard]] bool has_pending_partial() const noexcept;
    /// Makes `stream` wait for the host round (the memop flag or the side stream's event) and
    /// returns the device-mapped FP32 [hidden, tokens] partial for the caller to add.
    [[nodiscard]] const float* wait_pending_partial(cudaStream_t stream);
    /// Closes the round after the partial was consumed; `block_output` feeds the stage check.
    void finish_pending_partial(const Tensor& block_output, cudaStream_t stream);
    /// wait, add into a BF16 [hidden, tokens] plane, finish -- for a caller with no fused add.
    void add_pending_partial(Tensor& destination, cudaStream_t stream);
    /// Every combine calls this, joined or not: the join probe's report needs to be able to say
    /// the split never engaged.
    void tick_combine();

    struct Impl;
    ~ExpertCache();
    ExpertCache(const ExpertCache&)            = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

private:
    ExpertCache();
    std::unique_ptr<Impl> impl_;
};

} // namespace sinfer::family
