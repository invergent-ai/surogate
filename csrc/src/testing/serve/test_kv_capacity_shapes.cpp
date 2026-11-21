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
// Plus the 2-card visit split (check_stages): each stage under auto, then built with the shared explicit capacity.
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

// The 2-card visit split under --kv-capacity auto, as construct_pipeline builds it: every stage plans its own
// curve under auto (the preflight), the pool every stage then gets is the smallest stage's resolution, and each
// stage is built with that count as an EXPLICIT capacity marked resolved-from-auto. Before 0da5c31b /
// 5c6a3502 validate_target_options capped that explicit count at lanes x logical pages and refused every
// --max-num-seqs 1 engine (the preflight resolves logical + window charge; e1b 17:27). Budgets: unlimited
// (the curve maximum) and a tight one (minimum + 2 GiB) per stage.
template <class Target>
void check_stages(const char* name, const std::filesystem::path& artifact, DeviceContext& device, Totals& totals) {
    if constexpr (requires { Target::pipeline_stage_bounds(2); }) {
        artifact::Reader reader(artifact);
        const family::TextGeometry geometry = Target::declared_geometry(reader);
        const auto profile                  = Target::resolve_weights(reader.identity());
        const std::vector<int> bounds       = Target::pipeline_stage_bounds(2);
        const auto native                   = static_cast<std::uint32_t>(geometry.max_context);
        for (const std::uint32_t context : {16384U, 131072U, 262144U, 1048576U}) {
            if (native != 0 && context > native) { continue; }
            for (const std::uint32_t lanes : {1U, 2U, 4U, 8U}) {
                for (const std::uint32_t chunk : {2048U, 8192U}) {
                    for (const bool tight : {false, true}) {
                        ++totals.shapes;
                        const std::string shape = std::string(name) + " 2 stages " + std::to_string(context) + " x " +
                                                  std::to_string(lanes) + " chunk " + std::to_string(chunk) +
                                                  (tight ? " tight" : " unlimited");
                        const auto stage_options = [&](int s) {
                            EngineOptions options;
                            options.artifact_path        = artifact;
                            options.max_context          = context;
                            options.max_concurrency      = lanes;
                            options.prefill_chunk        = std::min(chunk, context);
                            options.use_cuda_graph       = true;
                            options.kv_capacity          = KvCapacityPolicy::automatic(0);
                            options.pipeline_stage_first = bounds[static_cast<std::size_t>(s)];
                            options.pipeline_stage_last  = bounds[static_cast<std::size_t>(s) + 1];
                            options.pipeline_boundary_columns = options.prefill_chunk + lanes + 128;
                            return options;
                        };
                        try {
                            // the preflight: each stage under auto, the pool = the smallest stage's
                            std::uint32_t common = 0;
                            for (int s = 0; s < 2; ++s) {
                                const auto curve =
                                    Target::make_sequence_planner(device, stage_options(s), profile, geometry)
                                        .capacity_curve();
                                const std::size_t budget =
                                    tight ? curve.minimum_device_reservation_bytes + (2ULL << 30) : ~std::size_t{0} / 2;
                                const auto resolved =
                                    runtime::resolve_kv_capacity(KvCapacityPolicy::automatic(0), curve, budget)
                                        .resolved_tokens;
                                common = common == 0 ? resolved : std::min(common, resolved);
                            }
                            // the build: the shared count as an explicit capacity resolved from auto
                            for (int s = 0; s < 2; ++s) {
                                EngineOptions options                    = stage_options(s);
                                options.kv_capacity                      = KvCapacityPolicy::explicit_capacity(common);
                                options.kv_capacity_resolved_automatically = true;
                                auto planner = Target::make_sequence_planner(device, options, profile, geometry);
                                const auto curve = planner.capacity_curve();
                                const std::uint32_t pages =
                                    runtime::resolve_kv_capacity(options.kv_capacity, curve, ~std::size_t{0})
                                        .resolved_tokens / 64U;
                                const std::size_t expected = curve.reservation_bytes(pages);
                                auto plan = std::move(planner).finalize(pages);
                                ++totals.plans;
                                if (plan.device_reservation_bytes() != expected) {
                                    std::printf("FAIL %s stage %d: reservation %zu != curve %zu\n", shape.c_str(), s,
                                                plan.device_reservation_bytes(), expected);
                                    ++totals.failed;
                                }
                                // one full-context request with its charge fits the stage's ledger
                                if (pages < curve.minimum_main_page_groups) {
                                    std::printf("FAIL %s stage %d: %u pages under the curve minimum %u\n", shape.c_str(),
                                                s, pages, curve.minimum_main_page_groups);
                                    ++totals.failed;
                                }
                                // every stage holds exactly the shared pool: no stage clamped it (review M8b minor 6)
                                if (static_cast<std::uint64_t>(pages) * 64U != common) {
                                    std::printf("FAIL %s stage %d: %u pages = %llu tokens, not the shared pool %u\n",
                                                shape.c_str(), s, pages,
                                                static_cast<unsigned long long>(pages) * 64ULL, common);
                                    ++totals.failed;
                                }
                                // the upper bound: one page past this stage's curve maximum is refused, not clamped
                                {
                                    EngineOptions over = options;
                                    over.kv_capacity = KvCapacityPolicy::explicit_capacity(
                                        (curve.maximum_main_page_groups + 1U) * 64U);
                                    bool refused = false;
                                    try {
                                        (void)Target::make_sequence_planner(device, over, profile, geometry);
                                    } catch (const std::invalid_argument&) {
                                        refused = true;
                                    }
                                    ++totals.plans;
                                    if (!refused) {
                                        std::printf("FAIL %s stage %d: a shared pool one page past the curve maximum "
                                                    "%u was accepted\n",
                                                    shape.c_str(), s, curve.maximum_main_page_groups);
                                        ++totals.failed;
                                    }
                                }
                                if (lanes == 1 && !tight) {
                                    std::printf("ok   %s stage %d [%d, %d): shared pool %u tokens, stage pages %u "
                                                "(curve %u..%u)\n",
                                                shape.c_str(), s, bounds[static_cast<std::size_t>(s)],
                                                bounds[static_cast<std::size_t>(s) + 1], common, pages,
                                                curve.minimum_main_page_groups, curve.maximum_main_page_groups);
                                }
                            }
                        } catch (const std::exception& error) {
                            std::printf("FAIL %s: %s\n", shape.c_str(), error.what());
                            ++totals.failed;
                        }
                    }
                }
            }
        }
    } else {
        (void)name; (void)artifact; (void)device; (void)totals;
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
    if (!inkling.empty()) {
        check_target<targets::inkling2::Package>("inkling2", inkling, device, true, totals);
        check_stages<targets::inkling2::Package>("inkling2", inkling, device, totals);
    }
    if (!gemma3.empty()) { check_target<targets::gemma3_270m::Package>("gemma3", gemma3, device, true, totals); }
    std::printf("%s: %d shapes, %d plans, %d failures (%.1f s)\n", totals.failed == 0 ? "PASS" : "FAIL",
                totals.shapes, totals.plans, totals.failed,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    return totals.failed == 0 ? 0 : 1;
}
