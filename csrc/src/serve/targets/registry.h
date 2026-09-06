#pragma once

#include "api/types.h"
#include "api/ops/linear.h"
#include "runtime/engine/pipeline_instance.h"
#include "runtime/engine/request_memory.h"
#include <api/targets/gemma3/package.h>
#include <api/targets/glm5_next/package.h>
#include <api/targets/lfm2/package.h>
#include <api/targets/llama/package.h>
#include <api/targets/qwen3/package.h>
#include <api/targets/qwen3_5/package.h>
#include <api/targets/qwen3_moe/package.h>
#include <api/targets/qwen3_5_moe/package.h>
#include <api/targets/qwen4exp/package.h>

#include <memory>
#include <variant>

#include "artifact/binder.h"
#include "artifact/materializer.h"

namespace sinfer {

struct DeviceContext;

namespace targets {

using Gemma3          = gemma3_270m::Package;
using Glm5Next        = glm5_next::Package;
using Lfm2            = lfm2::Package;
using Llama           = llama::Package;
using Qwen3Dense      = qwen3::Package;
using Qwen3Moe        = qwen3_moe::Package;
/// One architecture, every size and generation that shares it: Qwen3.5, 3.6 and 3.8.
using Qwen3_5     = qwen3_5::Package;
using Qwen3_5Moe  = qwen3_5_moe::Package;
using Qwen38FlashNext = qwen4exp::Package;

// One loaded model and one live instance, per target.
//
// These were nine hand-written pairs, identical to the character apart from the
// package they name -- and their constructors in registry.cpp were identical
// too. Two templates say it once; the aliases below keep every existing name,
// because the engine's executor arms and the target variant refer to them.
template <class PackageT>
struct LoadedTarget {
    using Package = PackageT;

    std::unique_ptr<typename Package::LoadedModel> model;
    typename Package::Frontend frontend;

    LoadedTarget(std::unique_ptr<typename Package::LoadedModel> stable_model,
                 const EngineOptions& options)
        : model(std::move(stable_model)), frontend(Package::make_frontend(*model, options)) {}
    ~LoadedTarget() = default;

    LoadedTarget(const LoadedTarget&)            = delete;
    LoadedTarget& operator=(const LoadedTarget&) = delete;
};

template <class PackageT>
struct TargetInstance {
    using Package = PackageT;

    std::unique_ptr<LoadedTarget<Package>> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<typename Package::Program> program;

    TargetInstance(std::unique_ptr<LoadedTarget<Package>> stable_loaded,
                   runtime::KvCapacityResolution resolution,
                   typename Package::SequencePlan sequence_plan, DeviceContext& device)
        : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
          request_memory(device, sequence_plan.request_transient_capacity_bytes()),
          capacity(sequence_plan.capacity()),
          program(Package::create_program(*loaded->model, std::move(sequence_plan), device)) {}
    ~TargetInstance() = default;

    TargetInstance(const TargetInstance&)            = delete;
    TargetInstance& operator=(const TargetInstance&) = delete;
};

using LoadedGemma3 = LoadedTarget<Gemma3>;
using Gemma3Instance = TargetInstance<Gemma3>;
using LoadedGlm5Next = LoadedTarget<Glm5Next>;
using Glm5NextInstance = TargetInstance<Glm5Next>;
using LoadedLfm2 = LoadedTarget<Lfm2>;
using Lfm2Instance = TargetInstance<Lfm2>;
using LoadedLlama = LoadedTarget<Llama>;
using LlamaInstance = TargetInstance<Llama>;
using LoadedQwen3Dense = LoadedTarget<Qwen3Dense>;
using Qwen3DenseInstance = TargetInstance<Qwen3Dense>;
using LoadedQwen3Moe = LoadedTarget<Qwen3Moe>;
using Qwen3MoeInstance = TargetInstance<Qwen3Moe>;
using LoadedQwen3_5 = LoadedTarget<Qwen3_5>;
using Qwen3_5Instance = TargetInstance<Qwen3_5>;
using LoadedQwen3_5Moe = LoadedTarget<Qwen3_5Moe>;
using Qwen3_5MoeInstance = TargetInstance<Qwen3_5Moe>;
using LoadedQwen38FlashNext = LoadedTarget<Qwen38FlashNext>;
using Qwen38FlashNextInstance = TargetInstance<Qwen38FlashNext>;


using Glm5NextPipeline   = runtime::PipelineInstance<Glm5NextInstance>;
using Qwen38FlashNextPipeline = runtime::PipelineInstance<Qwen38FlashNextInstance>;
using Qwen3_5Pipeline    = runtime::PipelineInstance<Qwen3_5Instance>;
using Qwen3_5MoePipeline = runtime::PipelineInstance<Qwen3_5MoeInstance>;

using ActiveTarget =
    std::variant<std::unique_ptr<Gemma3Instance>, std::unique_ptr<Glm5NextInstance>,
                 std::unique_ptr<Lfm2Instance>,
                 std::unique_ptr<LlamaInstance>, std::unique_ptr<Qwen3DenseInstance>,
                 std::unique_ptr<Qwen3MoeInstance>,
                 std::unique_ptr<Qwen3_5Instance>,
                 std::unique_ptr<Qwen3_5MoeInstance>,
                 std::unique_ptr<Qwen38FlashNextInstance>,
                 std::unique_ptr<Glm5NextPipeline>,
                 std::unique_ptr<Qwen38FlashNextPipeline>, std::unique_ptr<Qwen3_5Pipeline>,
                 std::unique_ptr<Qwen3_5MoePipeline>>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    /// The per-request context ceiling this target was planned with — the value the caller
    /// asked for, or, when it asked for 0, the largest the device's free memory allowed.
    std::uint32_t resolved_max_context = 0;
};

/// Pipeline parallelism: one stage instance per device in `options.devices`, layers split
/// evenly, the residual handed over through pinned memory (Flash-Next, Qwen3.8-27B, 35B-A3B).
[[nodiscard]] ConstructedTarget construct_pipeline_target(const EngineOptions& options);

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 DeviceContext& device);

/// Device bytes the runtime will derive from the resident W8 weights (FP8/FP4 planes, Marlin
/// tiles) -- 1.5x their size, an over-estimate on purpose. Subtracted from what is free before
/// the KV capacity is resolved; a target that sizes its own device pools before that point
/// must leave it too, which is why it is declared rather than kept to the registry.
/// `policy` is the target's linear policy: the planes are derived only when an A8 or A4
/// compute profile is admitted, so a target that admits A16 alone projects nothing -- GLM-5.3
/// and Flash-Next both, and the 1.5x of their W8 bytes was starving the expert pool.
std::size_t projected_derived_residency_bytes(
    const artifact::Binder& binder, const artifact::MaterializationPlan& plan,
    ops::LinearPolicy policy = ops::LinearPolicy::AllowA4);

/// Device memory the load holds only while it runs: the materializer stages the source bytes
/// of every object it rearranges rather than copies (a GGUF's F16 into W8 planes, a permuted
/// parent) in one scratch arena, and frees it before serving. It is 13.7 GiB on GLM-5.3-Flash's
/// card against 8.8 GiB of resident weights, and anything created before the load -- the
/// expert pool -- has to leave it room. Computed the way the materializer will.
std::size_t projected_load_staging_bytes(const artifact::Binder& binder,
                                         const artifact::MaterializationPlan& plan);

} // namespace targets
} // namespace sinfer
