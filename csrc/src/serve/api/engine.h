#pragma once

#include <span>
#include "api/types.h"

#include <chrono>
#include <memory>

namespace sinfer {

namespace ops {
class LoraStore;
class LoraStoreSet;
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
    /// The prompt's token ids, as the model will see them.
    ///
    /// A reinforcement-learning client needs these to score the sequence it was
    /// given, and it must be the engine's own tokenisation rather than the
    /// client's -- re-rendering a chat prompt is not guaranteed to reproduce it.
    /// Copied on request, because a prompt is submitted far more often than it is
    /// asked for.
    [[nodiscard]] std::vector<TokenId> token_ids() const;
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

    /// A prompt served exactly as written, with no chat template: what `/v1/completions` sends,
    /// and the only shape a base model can be asked anything in.
    [[nodiscard]] PreparedPrompt prepare_text(std::string_view text,
                                              bool allow_prefix_identity = true) const;
    /// Whether the loaded artifact carries a chat template. False for a base model, and the
    /// chat-shaped endpoints refuse rather than render nothing.
    [[nodiscard]] bool supports_chat() const;
    /// The stop token ids this model ends on by default. A request asking for a
    /// minimum length bars exactly these until it is reached.
    [[nodiscard]] std::vector<TokenId> default_stop_tokens() const;

    /// The text of each token id, one string per id -- what an OpenAI `logprobs`
    /// entry names beside its number.
    [[nodiscard]] std::vector<std::string> token_texts(std::span<const TokenId> ids) const;

    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    // Establishes queue membership synchronously. Destroying an unconsumed handle cancels its
    // request; wait() owns result consumption and may run independently from GPU execution.
    // lifetime retains external execution resources until GPU work finishes, even
    // when the handle is abandoned before wait().
    [[nodiscard]] GenerationHandle
    submit(PreparedPrompt prompt, RequestOptions options,
           std::chrono::steady_clock::time_point pending_deadline = {},
           std::shared_ptr<void> lifetime = {});

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
    /// Drop the prefix cache and let an elastic KV pool return its idle granules; the
    /// footprint `sleepable_bytes` reports shrinks accordingly. A no-op for arena pools.
    void shrink_kv();

    /// VRAM this engine's sleepable regions occupy while awake -- its cost in
    /// a resident set. Zero when built without sleep_enable.
    [[nodiscard]] std::size_t sleepable_bytes() const;

    /// This engine's adapter store (runtime load/unload operates on it).
    /// The CUDA device this engine's memory lives on. A thread that touches that
    /// memory itself -- an adapter upload from an HTTP handler, say -- must bind
    /// this device first (`ScopedDevice`), or the runtime resolves the pointers
    /// against whatever device that thread happened to be on.
    [[nodiscard]] int device() const;

    /// This engine's adapter stores, one per device it spans. A single-device
    /// engine has one; a pipeline has one per stage's device, and an adapter's
    /// modules are spread across them by layer.
    [[nodiscard]] ops::LoraStoreSet& lora_stores();

    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    void reset_memory_peaks() noexcept;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace sinfer
