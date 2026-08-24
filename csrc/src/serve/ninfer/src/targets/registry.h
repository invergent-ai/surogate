#pragma once

#include "ninfer/types.h"
#include "runtime/engine/request_memory.h"
#include <ninfer/targets/qwen3_5_0_8b/package.h>
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6_35b_a3b/package.h>

#include <memory>
#include <variant>

namespace ninfer {

struct DeviceContext;

namespace targets {

using Qwen3_5_0_8B    = qwen3_5_0_8b::Package;
using Qwen3_6_27B    = qwen3_6_27b::Package;
using Qwen3_6_35BA3B = qwen3_6_35b_a3b::Package;

struct LoadedQwen3_5_0_8B {
    std::unique_ptr<Qwen3_5_0_8B::LoadedModel> model;
    Qwen3_5_0_8B::Frontend frontend;

    LoadedQwen3_5_0_8B(std::unique_ptr<Qwen3_5_0_8B::LoadedModel> stable_model,
                      const EngineOptions& options);
    ~LoadedQwen3_5_0_8B();

    LoadedQwen3_5_0_8B(const LoadedQwen3_5_0_8B&)            = delete;
    LoadedQwen3_5_0_8B& operator=(const LoadedQwen3_5_0_8B&) = delete;
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

using ActiveTarget =
    std::variant<std::unique_ptr<Qwen3_5_0_8BInstance>, std::unique_ptr<Qwen3_6_27BInstance>,
                 std::unique_ptr<Qwen3_6_35BA3BInstance>>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
};

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 DeviceContext& device);

} // namespace targets
} // namespace ninfer
