#pragma once

#include "api/types.h"

#include <chrono>
#include <memory>

namespace sinfer {

namespace ops {
class LoraStore;
} // namespace ops

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();

    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] const PromptSummary& summary() const noexcept;
    [[nodiscard]] const PromptPreparationStats& preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    class Impl;
    explicit PreparedPrompt(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class GenerationHandle {
public:
    GenerationHandle() noexcept;
    ~GenerationHandle();

    GenerationHandle(GenerationHandle&&) noexcept;
    GenerationHandle& operator=(GenerationHandle&&) noexcept;

    GenerationHandle(const GenerationHandle&)            = delete;
    GenerationHandle& operator=(const GenerationHandle&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept;

    GenerationResult wait(OutputSink* sink = nullptr, const CancellationView& cancellation = {});

private:
    class Impl;
    explicit GenerationHandle(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

class Engine {
public:
    explicit Engine(EngineOptions options);
    ~Engine();

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;

    // Raw token input is retained for parity tools and repeatable performance measurement.
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously. Destroying an unconsumed handle cancels its
    // request; wait() owns result consumption and may run independently from GPU execution.
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           std::chrono::steady_clock::time_point pending_deadline = {});

    GenerationResult generate(PreparedPrompt prompt, RequestOptions options,
                              OutputSink* sink                     = nullptr,
                              const CancellationView& cancellation = {});

    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] LoadSummary load_summary() const;
    /// Sleep level 1: back the model's device memory up to pinned host and
    /// release the physical VRAM, keeping every virtual address (and therefore
    /// every captured CUDA graph) valid. New submissions are refused while
    /// asleep; in-flight work must be drained by the caller first. Requires the
    /// engine to have been built with sleepable allocations (EngineOptions
    /// sleep_enable). Idempotent.
    /// `allow_active` skips the drained-engine requirement: the worker loop
    /// parks between rounds and in-flight generations resume byte-identically
    /// after the next wake (their whole state rides the offloaded arenas).
    void sleep(bool allow_active = false);
    /// First half of sleep on its own: refuse new submissions while leaving
    /// in-flight requests to finish. The caller drains, then calls sleep().
    /// wake() undoes it if the drain is abandoned.
    void sleep_begin();
    /// Allocate this engine's pinned sleep backup now, so the first sleep is as
    /// fast as every later one. No-op without sleep_enable.
    void prepare_sleep_backup();
    /// Map fresh physical memory at the original addresses and restore the
    /// backup. After this, requests run against byte-identical state -- the
    /// prefix cache survives a sleep. Throws if VRAM was taken by another
    /// process meanwhile; the engine stays asleep and the call can be retried.
    void wake();
    [[nodiscard]] bool is_sleeping() const;

    /// VRAM this engine's sleepable regions occupy while awake -- its cost in
    /// a resident set. Zero when built without sleep_enable.
    [[nodiscard]] std::size_t sleepable_bytes() const;

    /// This engine's adapter store (runtime load/unload operates on it).
    [[nodiscard]] ops::LoraStore& lora_store();

    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    void reset_memory_peaks() noexcept;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace sinfer
