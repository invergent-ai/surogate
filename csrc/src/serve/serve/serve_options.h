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

// --default-max-tokens when not given: 0, no server default. A chat / Responses request without max_tokens then gets
// the rest of the context, as vLLM 0.31 gives it when the checkpoint's generation_config has no max_new_tokens
// (vllm/entrypoints/serve/utils/api_utils.py get_max_tokens); --default-max-tokens N plays that max_new_tokens.
inline constexpr int kDefaultMaxTokens                    = 0;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;

/// --overlong-max-tokens: `reject` refuses a max_tokens that does not fit the context next to the prompt with
/// vLLM 0.31's HTTP 400 (serve/vllm_max_tokens.h); `cap` caps it to the rest of the context, as the engine did
/// before.
enum class OverlongMaxTokens { Reject, Cap };

/// --thinking-retention off|loop|all (serve/thinking_retention.h): the server keeps the reasoning it generated for a
/// chat completion and puts it back into a later request whose client sent that assistant turn without it -- `loop`
/// only for the turns after the last user message, `all` for every turn. Off by default.
enum class ThinkingRetentionMode : std::uint8_t { Off, Loop, All };
inline constexpr std::size_t kDefaultThinkingRetentionMib           = 512;
inline constexpr std::uint64_t kDefaultThinkingRetentionTtlSeconds = 6 * 3600;

struct ServeOptions {
    bool help_requested = false;
    std::string artifact_path;
    std::vector<BorrowedTensor> borrowed_weights;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::string openrouter_models_file;           // optional provider catalog; separate from /v1/models
    std::optional<std::string> model_id_override; // unset => original model argument

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
    /// --rope-scaling: YaRN over the checkpoint's rotation (EngineOptions::RopeScaling).
    EngineOptions::RopeScaling rope_scaling;
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
    double rate_limit_rps                  = 0; // 0 disables the process-wide token bucket
    std::uint32_t rate_limit_burst         = 1;
    std::uint32_t max_inflight_requests    = 0; // HTTP POSTs, including preprocessing/streaming
    std::uint32_t max_thinking_requests    = 0; // decisions with thinking=true; 0 disables
    std::uint32_t max_image_requests       = 0; // decisions with images; 0 disables
    std::uint32_t pending_timeout_ms       = 30000;
    // --kv-admission reserve|ondemand, --kv-admission-headroom N, --kv-admission-watermark F
    // (api/types.h KvAdmissionOptions, DESIGN-kv-admission.md).
    KvAdmissionOptions kv_admission;
    // --adapter-update-timeout-ms N (0 = inherit pending_timeout_ms, as this
    // shared one number for both jobs before the flag existed). Separate
    // because they bound different things: above is how long ONE request may
    // wait for a lane, this is how long a weight update may wait for every
    // admitted request to release its adapter claim, which scales with the
    // backlog. Sizing the first to a training run silently sized the second
    // too, past the admin client's own timeout, so a weight update that was
    // draining normally failed as an opaque client-side disconnect.
    std::uint32_t adapter_update_timeout_ms = 0;
    /// --lora-swap in-place|drain: what reloading a loaded adapter name does when the request does
    /// not say ("in_place"). drain (the default) lets the adapter's admitted requests finish and
    /// holds new ones; in-place copies the new bytes in at a round boundary and keeps serving --
    /// what an RL rollout fleet wants, whose calls run for minutes.
    bool lora_swap_in_place = false;
    // surogate vendor patch (PATCHES.md #13): 2048 measured +16% prefill on
    // qwen3.5-0.8b (K=1024 tiles amortize) and neutral on qwen3.6-27b
    // (1832 vs 1834 tok/s) on RTX 5090.
    std::uint32_t prefill_chunk            = 2048;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    std::uint32_t gemma_image_tokens = 0;
    int device                             = 0;
    std::vector<int> devices;                    // --devices a,b,c (pipeline stages, in order)
    /// --data-parallel: with --devices, one whole engine per listed GPU instead of a layer
    /// pipeline across them, every one serving the same model id; requests are spread over them
    /// (serve/replica_router.h). For a model that fits one GPU this is the throughput layout: a
    /// pipeline gives each round to one stage at a time and leaves the other cards idle.
    bool data_parallel = false;
    KvCacheStorage kv_cache                = KvCacheStorage::Auto;
    std::vector<std::uint32_t> kv_cache_skip_layers;
    bool elastic_kv          = true;  // default; --no-elastic-kv puts the Main KV planes in the arena
    bool elastic_kv_overcommit = false; // --elastic-kv-overcommit: floor + device gate
    std::size_t gpu_memory_limit_bytes = 0; // --gpu-memory-limit-mib: per-device process budget
    // --prefix-cache-host-gb N: page-locked host memory per GPU for conversations the KV pool
    // takes back (EngineOptions::prefix_cache_host_bytes); unset keeps the 512 MiB archive.
    std::optional<std::size_t> prefix_cache_host_bytes;
    SpeculativeOptions speculative;
    bool enable_vision      = false;
    bool use_cuda_graph     = true;
    bool allow_prefix_reuse = true;
    // --batch-invariant: a request's logits, log-probabilities and greedy tokens do not depend
    // on what else shares its rounds (api/ops/batch_invariant.h). Off by default: it trades
    // throughput for reproducibility.
    bool batch_invariant = false;
    /// --moe-route auto|w4a16|marlin (api/ops/sparse_moe.h SparseMoeRoute): w4a16 keeps the routed
    /// NVFP4 experts on our W4A16 kernels at every width, so a token's log-prob does not depend on
    /// its round's width -- the RL engines' setting (rollouts, KL-anchor prefill); marlin rewrites
    /// them into Marlin tiles at load and serves every width weight-only. auto is the default.
    std::string moe_route = "auto";
    // Default thinking mode for the generation prompt: --thinking or --no-thinking, else what
    // the chat template does when nothing asks (PromptCapabilities::thinking_by_default).
    std::optional<bool> enable_thinking;
    bool preserve_thinking = false;
    /// --thinking-retention, --thinking-retention-mb (the store's byte cap) and --thinking-retention-ttl (how long an
    /// entry is kept without being used).
    ThinkingRetentionMode thinking_retention             = ThinkingRetentionMode::Off;
    std::size_t thinking_retention_bytes                 = kDefaultThinkingRetentionMib << 20;
    std::uint64_t thinking_retention_ttl_seconds         = kDefaultThinkingRetentionTtlSeconds;
    /// --thinking-loop-guard and --thinking-answer-reserve N (StopPolicy::thinking_guard): a thought that goes round
    /// in circles, or that reaches the request's output budget less min(N, budget / 4) tokens, is closed and the
    /// request answers in what is left of its budget. Both off by default.
    bool thinking_loop_guard            = false;
    std::uint32_t thinking_answer_reserve = 0;
    int default_max_tokens = kDefaultMaxTokens;
    // --overlong-max-tokens reject|cap: what a max_tokens that does not fit the context next to the prompt gets.
    // reject (default) answers HTTP 400 as vLLM 0.31 does; cap is the earlier silent cap to the rest of the
    // context (serve/vllm_max_tokens.h).
    OverlongMaxTokens overlong_max_tokens = OverlongMaxTokens::Reject;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs

    // Output parsing, named the way vLLM names it. Defaults preserve the behaviour
    // that was hard-wired before the flags existed: reasoning fenced with <think>,
    // tool calls in the Qwen block.
    ReasoningFormat reasoning_format = ReasoningFormat::ThinkTags;
    ToolCallFormat tool_call_format  = ToolCallFormat::QwenXml;
    /// --tool-call-parser was given. Without it a checkpoint whose vocabulary spells Gemma 4's
    /// tool-call markers is served with the gemma4 parser (GenerationService).
    bool tool_call_format_explicit = false;
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
    /// --capture-routing: Rollout Routing Replay for RL rollouts. Every forward records each MoE
    /// layer's selected experts, and a request with "return_routed_experts": true gets them back
    /// (choice.routed_experts). Under --spec mtp a verify round's accepted columns are stored and its
    /// rejected drafts dropped (the draft head is never recorded); DFlash is refused.
    bool capture_routing = false;
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
    /// --decision-temperature T: the calibration temperature of the decisions endpoint. Every
    /// decisions answer this process gives -- every question type, every route alias, every
    /// served model and adapter -- is read from softmax(logits / T) over the option labels.
    /// Finite and greater than zero; 1 is the model's own distribution. It is not --temperature:
    /// the decisions readout samples nothing, and generation never sees this value.
    double decision_temperature = 1.0;
    /// --decision-attempts N: how many times a decisions request is run before non-finite
    /// option logits are returned as an error. Each attempt after the first recomputes the
    /// whole request from its prompts: a fresh shared-prefix prefill and fresh question rows,
    /// nothing kept from the attempt that failed. 1 turns retrying off.
    std::uint32_t decision_attempts = 3;

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
};

ServeOptions parse_serve_options(int argc, char** argv);
ServeOptions extra_model_options(const ServeOptions& primary, const ServeOptions::ExtraModel& extra);
/// The options of data-parallel replica `replica` of the primary model: the primary's, on
/// `primary.devices[replica]` alone.
ServeOptions replica_options(const ServeOptions& primary, std::size_t replica);
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id);
std::string serve_usage_text(const char* argv0);

} // namespace sinfer::serve
