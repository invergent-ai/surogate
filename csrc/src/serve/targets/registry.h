#pragma once

#include "api/types.h"
#include "runtime/engine/pipeline_instance.h"
#include "runtime/engine/request_memory.h"
#include <api/targets/qwen3_5_0_8b/package.h>
#include <api/targets/qwen3_5_2b/package.h>
#include <api/targets/qwen3_5_4b/package.h>
#include <api/targets/qwen3_6_27b/package.h>
#include <api/targets/qwen3_6_35b_a3b/package.h>
#include <api/targets/qwen4exp/package.h>

#include <memory>
#include <variant>

namespace ninfer {

struct DeviceContext;

namespace targets {

using Qwen3_5_0_8B    = qwen3_5_0_8b::Package;
using Qwen3_5_2B      = qwen3_5_2b::Package;
using Qwen3_5_4B      = qwen3_5_4b::Package;
using Qwen3_6_27B    = qwen3_6_27b::Package;
using Qwen3_6_35BA3B = qwen3_6_35b_a3b::Package;
using Qwen38FlashNext = qwen4exp::Package;

struct LoadedQwen3_5_0_8B {
    std::unique_ptr<Qwen3_5_0_8B::LoadedModel> model;
    Qwen3_5_0_8B::Frontend frontend;

    LoadedQwen3_5_0_8B(std::unique_ptr<Qwen3_5_0_8B::LoadedModel> stable_model,
                      const EngineOptions& options);
    ~LoadedQwen3_5_0_8B();

    LoadedQwen3_5_0_8B(const LoadedQwen3_5_0_8B&)            = delete;
    LoadedQwen3_5_0_8B& operator=(const LoadedQwen3_5_0_8B&) = delete;
};

struct LoadedQwen3_5_2B {
    std::unique_ptr<Qwen3_5_2B::LoadedModel> model;
    Qwen3_5_2B::Frontend frontend;

    LoadedQwen3_5_2B(std::unique_ptr<Qwen3_5_2B::LoadedModel> stable_model,
                      const EngineOptions& options);
    ~LoadedQwen3_5_2B();

    LoadedQwen3_5_2B(const LoadedQwen3_5_2B&)            = delete;
    LoadedQwen3_5_2B& operator=(const LoadedQwen3_5_2B&) = delete;
};

struct LoadedQwen3_5_4B {
    std::unique_ptr<Qwen3_5_4B::LoadedModel> model;
    Qwen3_5_4B::Frontend frontend;

    LoadedQwen3_5_4B(std::unique_ptr<Qwen3_5_4B::LoadedModel> stable_model,
                      const EngineOptions& options);
    ~LoadedQwen3_5_4B();

    LoadedQwen3_5_4B(const LoadedQwen3_5_4B&)            = delete;
    LoadedQwen3_5_4B& operator=(const LoadedQwen3_5_4B&) = delete;
};

struct Qwen3_5_0_8BInstance {
    using Package = Qwen3_5_0_8B;

    std::unique_ptr<LoadedQwen3_5_0_8B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_5_0_8B::Program> program;

    Qwen3_5_0_8BInstance(std::unique_ptr<LoadedQwen3_5_0_8B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Qwen3_5_0_8B::SequencePlan sequence_plan, DeviceContext& device);
    ~Qwen3_5_0_8BInstance();

    Qwen3_5_0_8BInstance(const Qwen3_5_0_8BInstance&)            = delete;
    Qwen3_5_0_8BInstance& operator=(const Qwen3_5_0_8BInstance&) = delete;
};

struct Qwen3_5_2BInstance {
    using Package = Qwen3_5_2B;

    std::unique_ptr<LoadedQwen3_5_2B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_5_2B::Program> program;

    Qwen3_5_2BInstance(std::unique_ptr<LoadedQwen3_5_2B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Qwen3_5_2B::SequencePlan sequence_plan, DeviceContext& device);
    ~Qwen3_5_2BInstance();

    Qwen3_5_2BInstance(const Qwen3_5_2BInstance&)            = delete;
    Qwen3_5_2BInstance& operator=(const Qwen3_5_2BInstance&) = delete;
};

struct Qwen3_5_4BInstance {
    using Package = Qwen3_5_4B;

    std::unique_ptr<LoadedQwen3_5_4B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_5_4B::Program> program;

    Qwen3_5_4BInstance(std::unique_ptr<LoadedQwen3_5_4B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Qwen3_5_4B::SequencePlan sequence_plan, DeviceContext& device);
    ~Qwen3_5_4BInstance();

    Qwen3_5_4BInstance(const Qwen3_5_4BInstance&)            = delete;
    Qwen3_5_4BInstance& operator=(const Qwen3_5_4BInstance&) = delete;
};

struct LoadedQwen3_6_27B {
    std::unique_ptr<Qwen3_6_27B::LoadedModel> model;
    Qwen3_6_27B::Frontend frontend;

    LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                      const EngineOptions& options);
    ~LoadedQwen3_6_27B();

    LoadedQwen3_6_27B(const LoadedQwen3_6_27B&)            = delete;
    LoadedQwen3_6_27B& operator=(const LoadedQwen3_6_27B&) = delete;
};

struct Qwen3_6_27BInstance {
    using Package = Qwen3_6_27B;

    std::unique_ptr<LoadedQwen3_6_27B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_27B::Program> program;

    Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Qwen3_6_27B::SequencePlan sequence_plan, DeviceContext& device);
    ~Qwen3_6_27BInstance();

    Qwen3_6_27BInstance(const Qwen3_6_27BInstance&)            = delete;
    Qwen3_6_27BInstance& operator=(const Qwen3_6_27BInstance&) = delete;
};

struct LoadedQwen3_6_35BA3B {
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> model;
    Qwen3_6_35BA3B::Frontend frontend;

    LoadedQwen3_6_35BA3B(std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model,
                         const EngineOptions& options);
    ~LoadedQwen3_6_35BA3B();

    LoadedQwen3_6_35BA3B(const LoadedQwen3_6_35BA3B&)            = delete;
    LoadedQwen3_6_35BA3B& operator=(const LoadedQwen3_6_35BA3B&) = delete;
};

struct Qwen3_6_35BA3BInstance {
    using Package = Qwen3_6_35BA3B;

    std::unique_ptr<LoadedQwen3_6_35BA3B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_35BA3B::Program> program;

    Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                           runtime::KvCapacityResolution resolution,
                           Qwen3_6_35BA3B::SequencePlan sequence_plan, DeviceContext& device);
    ~Qwen3_6_35BA3BInstance();

    Qwen3_6_35BA3BInstance(const Qwen3_6_35BA3BInstance&)            = delete;
    Qwen3_6_35BA3BInstance& operator=(const Qwen3_6_35BA3BInstance&) = delete;
};

struct LoadedQwen38FlashNext {
    std::unique_ptr<Qwen38FlashNext::LoadedModel> model;
    Qwen38FlashNext::Frontend frontend;

    LoadedQwen38FlashNext(std::unique_ptr<Qwen38FlashNext::LoadedModel> stable_model,
                          const EngineOptions& options);
    ~LoadedQwen38FlashNext();

    LoadedQwen38FlashNext(const LoadedQwen38FlashNext&)            = delete;
    LoadedQwen38FlashNext& operator=(const LoadedQwen38FlashNext&) = delete;
};

struct Qwen38FlashNextInstance {
    using Package = Qwen38FlashNext;

    std::unique_ptr<LoadedQwen38FlashNext> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen38FlashNext::Program> program;

    Qwen38FlashNextInstance(std::unique_ptr<LoadedQwen38FlashNext> stable_loaded,
                            runtime::KvCapacityResolution resolution,
                            Qwen38FlashNext::SequencePlan sequence_plan, DeviceContext& device);
    ~Qwen38FlashNextInstance();

    Qwen38FlashNextInstance(const Qwen38FlashNextInstance&)            = delete;
    Qwen38FlashNextInstance& operator=(const Qwen38FlashNextInstance&) = delete;
};

using Qwen38FlashNextPipeline = runtime::PipelineInstance<Qwen38FlashNextInstance>;
using Qwen3_6_27BPipeline     = runtime::PipelineInstance<Qwen3_6_27BInstance>;
using Qwen3_6_35BA3BPipeline  = runtime::PipelineInstance<Qwen3_6_35BA3BInstance>;

using ActiveTarget =
    std::variant<std::unique_ptr<Qwen3_5_0_8BInstance>,
                 std::unique_ptr<Qwen3_5_2BInstance>,
                 std::unique_ptr<Qwen3_5_4BInstance>, std::unique_ptr<Qwen3_6_27BInstance>,
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
} // namespace ninfer
