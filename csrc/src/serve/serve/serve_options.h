#pragma once

#include "api/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "serve/output_parsers.h"

namespace sinfer::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens                    = 8192;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::vector<BorrowedTensor> borrowed_weights;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact identity.model_id

    /// Scheduler weight class for a model under multi-model overcommit.
    enum class ModelPriority { Low, Normal, High };

    struct LoraModule {
        std::string name;
        std::string path;
    };

    /// Additional models served from the same process (multi-model serving).
    /// Each runs its own Engine -- arenas, worker thread, stream, graphs --
    /// inside the shared CUDA context, which is what lets concurrent rounds
    /// from different models overlap instead of time-slicing.
    struct ExtraModel {
        std::string name;          ///< served model id; requests route by it
        std::string artifact_path;
        std::optional<int> device;
        std::vector<int> devices; ///< devices=A:B:C; empty inherits the primary placement
        std::uint32_t kv_tokens    = 0; ///< required: extras size their KV explicitly
        std::uint32_t max_num_seqs = 0; ///< 0 = inherit the primary's
        std::uint32_t max_context  = 0; ///< 0 = inherit the primary's
        SpeculativeOptions speculative; ///< off unless spec=/draft-tokens= given
        /// This model's own adapters (lora=name:path keys, repeatable). Names
        /// share one flat namespace with every served id and every other
        /// model's adapters, because a request selects by the single `model`
        /// string; collisions are refused at startup.
        std::vector<LoraModule> lora;
        ModelPriority priority = ModelPriority::Normal; ///< priority=high|normal|low
    };
    std::vector<ExtraModel> extra_models;
    /// The primary model's scheduler weight class (--model-priority).
    ModelPriority model_priority = ModelPriority::Normal;
    std::string request_log_jsonl;                // empty => structured request logging disabled
    std::uint32_t max_context              = 8192;
    KvCapacityPolicy kv_capacity           = KvCapacityPolicy::explicit_capacity(8192);
    // --host-moe-layers N|all: experts of this many mixture layers live in pinned host memory,
    // read over PCIe, instead of on the card. What lets a model larger than the cards load.
    std::uint32_t host_moe_layers          = 0;
    // --gpu-layers N: layers kept on the card; every later layer is read from host memory.
    std::uint32_t gpu_layers               = 0;
    bool offload_vision = false;
    bool offload_embeddings = false;
    bool offload_output_head = false;
    std::uint32_t expert_slots             = 0; // --expert-slots N (host-streamed MoE targets)
    EngineOptions::HostExpertBank host_expert_bank               = EngineOptions::HostExpertBank::Auto; // --host-expert-bank w8|q4 (default: q4 when the slot cache is on)
    float cpu_moe_share                    = 0.0F; // --cpu-moe-share F (fraction of misses on the host)
    std::uint32_t cpu_moe_min_tokens       = 0;    // --cpu-moe-min-tokens N (0 = target default)
    float cpu_moe_prefill_share            = -1.0F; // --cpu-moe-prefill-share F (default 0.5 with the split; 0 = off)
    std::uint32_t max_concurrency          = 1;
    std::uint32_t max_pending_requests     = 16;
    std::uint32_t pending_timeout_ms       = 30000;
    // surogate vendor patch (PATCHES.md #13): 2048 measured +16% prefill on
    // qwen3.5-0.8b (K=1024 tiles amortize) and neutral on qwen3.6-27b
    // (1832 vs 1834 tok/s) on RTX 5090.
    std::uint32_t prefill_chunk            = 2048;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    int device                             = 0;
    std::vector<int> devices;                    // --devices a,b,c (pipeline stages, in order)
    KvCacheStorage kv_cache                = KvCacheStorage::Auto;
    std::vector<std::uint32_t> kv_cache_skip_layers;
    bool rewrite_checkpoints = false;
    bool elastic_kv          = true;  // default; --no-elastic-kv puts the Main KV planes in the arena
    bool elastic_kv_overcommit = false; // --elastic-kv-overcommit: floor + device gate
    SpeculativeOptions speculative;
    bool enable_vision      = false;
    bool use_cuda_graph     = true;
    bool allow_prefix_reuse = true;
    bool enable_thinking =
        true; // default thinking mode for the generation prompt (--no-thinking opts out)
    bool preserve_thinking = false;
    int default_max_tokens = kDefaultMaxTokens;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs

    // Output parsing, named the way vLLM names it. Defaults preserve the behaviour
    // that was hard-wired before the flags existed: reasoning fenced with <think>,
    // tool calls in the Qwen block.
    ReasoningFormat reasoning_format = ReasoningFormat::ThinkTags;
    ToolCallFormat tool_call_format  = ToolCallFormat::QwenXml;
    /// --enable-auto-tool-choice: vLLM gates automatic tool selection behind this
    /// flag and requires a tool-call parser with it. Off, a request asking for
    /// `tool_choice: "auto"` with tools attached is refused rather than answered
    /// with prose the caller will try to parse as a call.
    bool enable_auto_tool_choice = false;
    /// --chat-template: a Jinja template read from disk that replaces the one the
    /// artifact carries. Empty keeps the artifact's.
    std::string chat_template;
    std::string chat_template_path;

    /// --enable-lora / --lora-modules name=path[,name=path...]. A request selects an
    /// adapter by naming it in `model`; the base model keeps its own id. vLLM's
    /// --max-loras and --max-lora-rank bound what a deployment will admit, and are
    /// checked at load so a too-large adapter is refused with its rank named rather
    /// than at the first request.
    bool enable_lora = false;
    bool enable_sleep_mode = false;
    std::vector<LoraModule> lora_modules;
    std::uint32_t max_loras     = 1;
    /// Banks are padded to this, so it is the rank an adapter may have and not the
    /// rank it must have. 32 rather than 16 because that is what the trainers
    /// around this engine produce -- a 16 here refused most real adapters with a
    /// flag the caller had no reason to expect. Raising it costs bank memory in
    /// proportion, and only for an engine started with --enable-lora.
    std::uint32_t max_lora_rank = 32;
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy = false; // --greedy: force temperature 0 (exact argmax)

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);
ServeOptions extra_model_options(const ServeOptions& primary, const ServeOptions::ExtraModel& extra);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id);
std::string serve_usage_text(const char* argv0);

} // namespace sinfer::serve
