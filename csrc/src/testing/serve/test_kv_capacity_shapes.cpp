// Lane f1 (ENGINE-INKLING): every serving shape plans. For each windowed target given on the command line
// (Inkling-2-Mini: an elastic window class charging 175 main pages per allocation; Gemma 3 270M: 95), a
// grid of --max-model-len x --max-num-seqs x --max-num-batched-tokens x CUDA graphs on / off under
// --kv-capacity auto, and for each shape the capacity curve's minimum, minimum + 1, middle, maximum - 1
// and maximum pages:
//   - the plan finalizes and its reservation equals the curve's (finalize_sequence_plan_impl's affinity
//     check). Small auto shapes refused to start in M5 when the curve's maximum passed lanes x logical
//     pages: "physical sequence layout is not affine in Main KV page capacity" (16,384 x 4 eager,
//     32,768 x 4 graphs);
//   - the minimum holds one full-context request with its window charge (the 429 of 2026-11-21 14:35:
//     the curve topped out at lanes x logical pages, so a near-max-model-len prompt plus its charge
//     never fit); printed as minimum - logical pages, which is the charge whenever that term binds;
//   - the explicit-capacity path at the same shapes (a pipeline stage builds with one) plans too.
// Planning only: a device context for DeviceContext, no weights, no KV allocation.
// Usage: sinfer_kv_capacity_shapes_test [--inkling <artifact>] [--gemma3 <artifact>]; 77 without a device or
// without any artifact.
#include <api/targets/gemma3/package.h>
#include <api/targets/inkling2/package.h>

#include "artifact/reader.h"
#include "core/device.h"
#include "runtime/engine/kv_capacity.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <chrono>
#include <filesystem>
#include <utility>
#include <set>
#include <string>
#include <vector>

using namespace sinfer;

namespace {

struct Totals {
    int shapes  = 0;
    int plans   = 0;
    int failed  = 0;
};

template <class Target>
void check_target(const char* name, const std::filesystem::path& artifact, DeviceContext& device,
                  bool windowed, Totals& totals) {
    artifact::Reader reader(artifact);
    const family::TextGeometry geometry = Target::declared_geometry(reader);
    const auto profile                  = Target::resolve_weights(reader.identity());
    const auto native                   = static_cast<std::uint32_t>(geometry.max_context);
    const std::vector<std::uint32_t> contexts{4096, 8192, 16384, 32768, 65536, 98304, 131072, 262144};
    for (const std::uint32_t context : contexts) {
        if (native != 0 && context > native) { continue; }
        for (const std::uint32_t lanes : {1U, 2U, 4U, 8U, 16U}) {
            for (const std::uint32_t chunk : {2048U, 4096U, 8192U}) {
                // graphs on + auto, eager + auto, graphs on + explicit-from-auto (a pipeline stage)
                for (const auto [graphs, automatic] : {std::pair{true, true}, std::pair{false, true},
                                                       std::pair{true, false}}) {
                    {
                        EngineOptions options;
                        options.artifact_path   = artifact;
                        options.max_context     = context;
                        options.max_concurrency = lanes;
                        options.prefill_chunk   = std::min(chunk, context);
                        options.use_cuda_graph  = graphs;
                        options.kv_capacity     = automatic ? KvCapacityPolicy::automatic(0)
                                                            : KvCapacityPolicy::explicit_capacity(context * lanes);
                        // a pipeline stage: an explicit capacity resolved from auto (construct_pipeline)
                        options.kv_capacity_resolved_automatically = !automatic;
                        ++totals.shapes;
                        const std::string shape = std::string(name) + " " + std::to_string(context) + " x " +
                                                  std::to_string(lanes) + " chunk " + std::to_string(chunk) +
                                                  (graphs ? " graphs" : " eager") +
                                                  (automatic ? " auto" : " explicit-from-auto");
                        try {
                            const runtime::SequenceCapacityCurve curve =
                                Target::make_sequence_planner(device, options, profile, geometry).capacity_curve();
                            const std::uint32_t lo = curve.minimum_main_page_groups;
                            const std::uint32_t hi = curve.maximum_main_page_groups;
                            const std::uint32_t logical = (context + 63U) / 64U;
                            if (windowed && automatic && lo <= logical) {
                                std::printf("FAIL %s: curve minimum %u holds no window charge over %u logical pages\n",
                                            shape.c_str(), lo, logical);
                                ++totals.failed;
                            }
                            // The minimum finalizes from the planner's own candidate; the others are built.
                            std::set<std::uint32_t> pages{hi, lo + (hi - lo) / 2};
                            if (lo < hi) { pages.insert(lo + 1); }
                            // An explicit capacity clamps to the curve; take the one this shape asks for.
                            if (!automatic) {
                                const auto explicit_resolution = runtime::resolve_kv_capacity(
                                    options.kv_capacity, curve, ~std::size_t{0});
                                pages.insert(explicit_resolution.resolved_tokens / 64U);
                            }
                            for (const std::uint32_t p : pages) {
                                auto planner = Target::make_sequence_planner(device, options, profile, geometry);
                                const std::size_t expected = planner.capacity_curve().reservation_bytes(p);
                                auto plan = std::move(planner).finalize(p);
                                ++totals.plans;
                                if (plan.device_reservation_bytes() != expected) {
                                    std::printf("FAIL %s: pages %u reservation %zu != curve %zu\n", shape.c_str(), p,
                                                plan.device_reservation_bytes(), expected);
                                    ++totals.failed;
                                }
                            }
                            if (context == 16384 || context == 262144) {
                                std::printf("ok   %s: curve %u..%u pages (minimum - logical = %u)\n", shape.c_str(),
                                            lo, hi, lo > logical ? lo - logical : 0U);
                            }
                        } catch (const std::exception& error) {
                            std::printf("FAIL %s: %s\n", shape.c_str(), error.what());
                            ++totals.failed;
                        }
                    }
                }
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path inkling, gemma3;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--inkling") { inkling = argv[i + 1]; }
        if (flag == "--gemma3") { gemma3 = argv[i + 1]; }
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    if (inkling.empty() && gemma3.empty()) {
        std::printf("SKIP: no artifact (--inkling <path> / --gemma3 <path>)\n");
        return 77;
    }
    DeviceContext device(0);
    Totals totals;
    const auto started = std::chrono::steady_clock::now();
    if (!inkling.empty()) { check_target<targets::inkling2::Package>("inkling2", inkling, device, true, totals); }
    if (!gemma3.empty()) { check_target<targets::gemma3_270m::Package>("gemma3", gemma3, device, true, totals); }
    std::printf("%s: %d shapes, %d plans, %d failures (%.1f s)\n", totals.failed == 0 ? "PASS" : "FAIL",
                totals.shapes, totals.plans, totals.failed,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    return totals.failed == 0 ? 0 : 1;
}
