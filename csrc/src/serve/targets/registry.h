#pragma once

#include "api/types.h"
#include "runtime/engine/pipeline_instance.h"
#include "runtime/engine/request_memory.h"
#include <api/targets/gemma3/package.h>
#include <api/targets/llama/package.h>
#include <api/targets/qwen3/package.h>
#include <api/targets/qwen3_5/package.h>
#include <api/targets/qwen3_6_27b/package.h>
#include <api/targets/qwen3_6_35b_a3b/package.h>
#include <api/targets/qwen4exp/package.h>

#include <memory>
#include <variant>

namespace sinfer {

struct DeviceContext;

namespace targets {

using Gemma3          = gemma3_270m::Package;
using Llama           = llama::Package;
using Qwen3Dense      = qwen3::Package;
using Qwen3_5    = qwen3_5::Package;
using Qwen3_6_27B    = qwen3_6_27b::Package;
using Qwen3_6_35BA3B = qwen3_6_35b_a3b::Package;
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
using LoadedLlama = LoadedTarget<Llama>;
using LlamaInstance = TargetInstance<Llama>;
using LoadedQwen3Dense = LoadedTarget<Qwen3Dense>;
using Qwen3DenseInstance = TargetInstance<Qwen3Dense>;
using LoadedQwen3_5 = LoadedTarget<Qwen3_5>;
using Qwen3_5Instance = TargetInstance<Qwen3_5>;
using LoadedQwen3_6_27B = LoadedTarget<Qwen3_6_27B>;
using Qwen3_6_27BInstance = TargetInstance<Qwen3_6_27B>;
using LoadedQwen3_6_35BA3B = LoadedTarget<Qwen3_6_35BA3B>;
using Qwen3_6_35BA3BInstance = TargetInstance<Qwen3_6_35BA3B>;
using LoadedQwen38FlashNext = LoadedTarget<Qwen38FlashNext>;
using Qwen38FlashNextInstance = TargetInstance<Qwen38FlashNext>;


using Qwen38FlashNextPipeline = runtime::PipelineInstance<Qwen38FlashNextInstance>;
using Qwen3_6_27BPipeline     = runtime::PipelineInstance<Qwen3_6_27BInstance>;
using Qwen3_6_35BA3BPipeline  = runtime::PipelineInstance<Qwen3_6_35BA3BInstance>;

using ActiveTarget =
    std::variant<std::unique_ptr<Gemma3Instance>, std::unique_ptr<LlamaInstance>, std::unique_ptr<Qwen3DenseInstance>,
                 std::unique_ptr<Qwen3_5Instance>,
                 std::unique_ptr<Qwen3_6_27BInstance>,
                 std::unique_ptr<Qwen3_6_35BA3BInstance>,
                 std::unique_ptr<Qwen38FlashNextInstance>,
                 std::unique_ptr<Qwen38FlashNextPipeline>, std::unique_ptr<Qwen3_6_27BPipeline>,
                 std::unique_ptr<Qwen3_6_35BA3BPipeline>>;

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

} // namespace targets
} // namespace sinfer
