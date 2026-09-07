#include "serve/serve_options.h"
#include "product/speculative_options.h"

#include <fstream>
#include <sstream>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <string_view>

namespace sinfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    // "auto", also the default when the flag is absent, lets the target choose once
    // its geometry is known: bf16 for a pure-attention stack, e4m3 where
    // linear-attention layers carry it (KvCacheStorage::Auto has the measurements).
    // vLLM reads auto as the model's own dtype; ask for bf16 or fp8 by name to pin it.
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "auto") { return KvCacheStorage::Auto; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    // vLLM spells the e4m3 cache both ways and treats them as one setting.
    if (value == "fp8" || value == "fp8_e4m3") { return KvCacheStorage::Fp8E4M3; }
    // e5m2 trades mantissa for range that KV tensors do not use; the engine
    // stores e4m3 only, so name the difference rather than parse-failing.
    if (value == "fp8_e5m2") {
        throw std::invalid_argument("kv-cache-dtype fp8_e5m2 is not supported; this engine stores "
                                    "e4m3 (use fp8 or fp8_e4m3)");
    }
    throw std::invalid_argument("invalid kv-cache-dtype: " + value +
                                " (expected auto|bf16|fp8|fp8_e4m3|int8)");
}

std::vector<std::uint32_t> parse_kv_skip_layers(const char* text) {
    std::vector<std::uint32_t> layers;
    const std::string value(text);
    std::size_t begin = 0;
    while (begin <= value.size()) {
        const std::size_t comma = value.find(',', begin);
        const std::string item =
            value.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (!item.empty()) {
            layers.push_back(static_cast<std::uint32_t>(
                parse_nonnegative_int(item.c_str(), "kv-cache-dtype-skip-layers")));
        }
        if (comma == std::string::npos) { break; }
        begin = comma + 1;
    }
    if (layers.empty()) {
        throw std::invalid_argument("--kv-cache-dtype-skip-layers needs at least one layer index");
    }
    std::sort(layers.begin(), layers.end());
    layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
    return layers;
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.sinfer> [--host H] [--port N] [--api-key KEY] "
           "[--served-model-name ID] [--max-model-len N|auto] [--kv-capacity N|auto] [--gpu-layers N|all] [--host-moe-layers N|auto|all] [--expert-slots N] [--host-expert-bank w8|q4] [--cpu-moe-share F|auto] [--cpu-moe-prefill-share F] [--cpu-moe-min-tokens N] "
           "[--max-num-seqs N] "
           "[--max-pending-requests N] [--pending-timeout-ms N] "
           "[--max-num-batched-tokens N] [--log-stats-interval-ms N] [--device N] [--devices A,B,...] "
           "[--reasoning-parser NAME] [--tool-call-parser NAME] [--enable-auto-tool-choice] "
           "[--chat-template FILE] [--enable-prefix-caching|--no-enable-prefix-caching] "
           "[--max-request-mib N] [--media-cache-mib N] [--media-live-mib N] "
           "[--media-preprocess-threads N] "
           "[--request-log-jsonl FILE] "
           "[--response-store-max-records N] [--response-store-max-mib N] "
           "[--kv-cache-dtype auto|fp8|int8] [--kv-cache-dtype-skip-layers L,...] "
           "[--spec mtp|dflash --draft-tokens N] [--spec-max-lanes N|all] "
           "[--default-max-tokens N] "
           "[--vision] [--enforce-eager] [--no-prefix-reuse] "
           "[--enable-sleep-mode] [--no-elastic-kv] [--elastic-kv-overcommit] "
           "[--lm-head-draft] [--no-thinking] [--preserve-thinking] [--cors] "
           "[--temperature F] [--top-p F] [--top-k N] [--min-p F] [--presence-penalty F] "
           "[--frequency-penalty F] [--seed N] [--greedy]\n"
           "       serves OpenAI Responses/Chat Completions and Anthropic Messages endpoints\n"
           "       --default-max-tokens defaults to " +
           std::to_string(kDefaultMaxTokens) +
           " when omitted\n"
           "       --max-request-mib defaults to 384 and is enforced before JSON parsing\n"
           "       --media-cache-mib defaults to 1024; 0 disables retained media reuse\n"
           "       --media-live-mib defaults to 2048 and bounds all live BF16 patch payloads\n"
           "       --media-preprocess-threads defaults to 0 (auto, at most 16 workers)\n"
           "       --request-log-jsonl appends full-precision server/request records\n"
           "       --served-model-name overrides the artifact identity.model_id reported by the server\n"
           "       Responses state is process-local and bounded to 1024 records / 256 MiB by "
           "default\n"
           "       --log-stats-interval-ms defaults to 5000; 0 disables periodic throughput logs\n"
           "       --vision enables media and loads the fixed Vision GPU allocations\n"
           "       --kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom\n"
           "       --kv-cache-dtype defaults to fp8 (e4m3), halving the cache; auto means the\n"
           "         same, and bf16 asks for a full-precision cache. Note vLLM reads auto as\n"
           "         the model dtype instead. Only\n"
           "         full-attention layers hold a KV cache, so linear-attention layers are\n"
           "         never quantized. --kv-cache-dtype-skip-layers holds named\n"
           "         full-attention layers at bf16 (comma-separated indices)\n"
           "       --rewrite-checkpoints keeps a per-lane GDN checkpoint so an edited last turn\n"
           "         resumes from its prefix instead of re-prefilling it; one state slot per lane\n"
           "         (72 MiB each on the 27B), off by default\n"
           "       --no-prefix-reuse disables compatible-prefix caching (enabled by default)\n"
           "       --no-elastic-kv keeps the Main KV pool's planes in the arena; by default they are\n"
           "                    mapped on demand and only pages in use (plus a small reserve) hold VRAM\n"
           "       --elastic-kv-overcommit guarantees each model only its\n"
           "                    --kv-capacity (one full-context request when auto) and admits every\n"
           "                    page past that against the device's free memory, shared across models\n"
           "       --enable-sleep-mode adds POST /sleep and /wake_up: sleeping releases VRAM\n"
           "                           (weights and cache parked in host RAM), waking restores in ~a second\n"
           "       --preserve-thinking retains closed-turn assistant reasoning in later prompts\n"
           "       sampler defaults come from the loaded model and resolved thinking mode; "
           "server flags and request fields override individual values.\n"
           "       --reasoning-parser selects how a reasoning span is recognised (default qwen3);\n"
           "         --tool-call-parser how a tool call is (default qwen3_xml). Both accept the\n"
           "         vLLM names, and an unknown one is refused with the supported list.\n"
           "       --enable-auto-tool-choice permits tool_choice 'auto'; it needs a tool-call parser.\n"
           "       --chat-template replaces the artifact's Jinja template with one read from FILE.\n"
           "       --enable-prefix-caching is this engine's default; the flag is accepted for\n"
           "         command-line compatibility, and --no-enable-prefix-caching turns it off.\n"
           "       --greedy forces temperature 0 (exact argmax).\n";
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    bool kv_capacity_explicit        = false;
    bool max_context_explicit        = false;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    options.artifact_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--served-model-name") {
            options.model_id_override = require_value("--served-model-name");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--served-model-name must not be empty");
            }
        } else if (arg == "--max-model-len") {
            const std::string text = require_value("--max-model-len");
            // `auto` (and the absence of the flag) means: fit the largest context the device's
            // free memory allows, up to what the weights were trained for.
            options.max_context =
                text == "auto" ? 0U
                               : static_cast<std::uint32_t>(
                                     parse_nonnegative_int(text.c_str(), "max-model-len"));
            max_context_explicit = true;
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--host-expert-bank") {
            const std::string text = require_value("--host-expert-bank");
            if (text == "q4") {
                options.host_expert_bank = EngineOptions::HostExpertBank::Q4;
            } else if (text == "w8") {
                options.host_expert_bank = EngineOptions::HostExpertBank::W8;
            } else if (text != "auto") {
                throw std::invalid_argument("--host-expert-bank must be w8, q4 or auto");
            }
        } else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers") {
            const std::string spec = require_value(arg.c_str());
            options.gpu_layers =
                spec == "all" ? std::numeric_limits<std::uint32_t>::max()
                              : static_cast<std::uint32_t>(
                                    parse_nonnegative_int(spec.c_str(), "gpu-layers"));
        } else if (arg == "--host-moe-layers") {
            // `all` is the whole stack: a model far larger than the cards then loads with only
            // its routers, norms, attention and shared experts resident.
            const std::string spec = require_value("--host-moe-layers");
            options.host_moe_layers =
                spec == "all"  ? std::numeric_limits<std::uint32_t>::max()
                : spec == "auto" ? EngineOptions::kHostMoeLayersAuto
                                 : static_cast<std::uint32_t>(
                                       parse_nonnegative_int(spec.c_str(), "host-moe-layers"));
        } else if (arg == "--expert-slots") {
            options.expert_slots =
                static_cast<std::uint32_t>(parse_nonnegative_int(require_value("--expert-slots"), "expert-slots"));
        } else if (arg == "--cpu-moe-min-tokens") {
            options.cpu_moe_min_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--cpu-moe-min-tokens"), "cpu-moe-min-tokens"));
        } else if (arg == "--cpu-moe-share") {
            const char* text = require_value("--cpu-moe-share");
            if (std::string(text) == "auto") {
                options.cpu_moe_share = -1.0F; // measured at startup (bandwidth-matched)
            } else {
                options.cpu_moe_share = std::strtof(text, nullptr);
                if (options.cpu_moe_share < 0.0F || options.cpu_moe_share > 1.0F) {
                    throw std::invalid_argument("--cpu-moe-share must be within [0, 1] or auto");
                }
            }
        } else if (arg == "--cpu-moe-prefill-share") {
            options.cpu_moe_prefill_share = std::strtof(require_value("--cpu-moe-prefill-share"), nullptr);
            if (options.cpu_moe_prefill_share < 0.0F || options.cpu_moe_prefill_share > 1.0F) {
                throw std::invalid_argument("--cpu-moe-prefill-share must be within [0, 1]");
            }
        } else if (arg == "--max-num-seqs") {
            // vLLM's name for the same quantity: sequences run per iteration,
            // which here is the lane count. The engine speaks vLLM's option
            // vocabulary so a command line transfers directly and a benchmark
            // comparison is like-for-like without translation.
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-num-seqs"), "max-num-seqs"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--max-num-batched-tokens") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-num-batched-tokens"), "max-num-batched-tokens"));
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--devices") {
            options.devices.clear();
            std::string list = require_value("--devices");
            std::size_t start = 0;
            while (start <= list.size()) {
                const std::size_t comma = list.find(',', start);
                const std::string item  = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!item.empty()) { options.devices.push_back(parse_nonnegative_int(item.c_str(), "devices")); }
                if (comma == std::string::npos) { break; }
                start = comma + 1;
            }
            if (options.devices.empty()) { throw std::invalid_argument("--devices needs at least one device"); }
            options.device = options.devices.front();
        } else if (arg == "--kv-cache-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-cache-dtype"));
        } else if (arg == "--rewrite-checkpoints") {
            options.rewrite_checkpoints = true;
        } else if (arg == "--no-rewrite-checkpoints") {
            options.rewrite_checkpoints = false;
        } else if (arg == "--kv-cache-dtype-skip-layers") {
            options.kv_cache_skip_layers =
                parse_kv_skip_layers(require_value("--kv-cache-dtype-skip-layers"));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
        } else if (arg == "--spec-max-lanes") {
            // The widest round that still verifies drafts; wider rounds run the head's narrow
            // round. "all" verifies at any width; unset takes the engine's default.
            const std::string spec = require_value("--spec-max-lanes");
            options.speculative.max_lanes =
                spec == "all" ? kSpeculateAtAnyWidth
                              : static_cast<std::uint32_t>(
                                    parse_nonnegative_int(spec.c_str(), "spec-max-lanes"));
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--elastic-kv") {
            options.elastic_kv = true; // the default; kept so older launch lines still parse
        } else if (arg == "--no-elastic-kv") {
            options.elastic_kv = false;
        } else if (arg == "--elastic-kv-overcommit") {
            options.elastic_kv            = true;
            options.elastic_kv_overcommit = true;
        } else if (arg == "--enforce-eager") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--enable-prefix-caching") {
            // vLLM's spelling for what this engine has done by default since it
            // shipped; accepted so a command line written for vLLM runs unchanged.
            options.allow_prefix_reuse = true;
        } else if (arg == "--no-enable-prefix-caching") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--reasoning-parser") {
            options.reasoning_format = parse_reasoning_format(require_value("--reasoning-parser"));
        } else if (arg == "--tool-call-parser") {
            options.tool_call_format = parse_tool_call_format(require_value("--tool-call-parser"));
        } else if (arg == "--enable-auto-tool-choice") {
            options.enable_auto_tool_choice = true;
        } else if (arg == "--chat-template") {
            options.chat_template_path = require_value("--chat-template");
        } else if (arg == "--model-priority") {
            const std::string value = require_value("--model-priority");
            if (value == "high") {
                options.model_priority = ServeOptions::ModelPriority::High;
            } else if (value == "normal") {
                options.model_priority = ServeOptions::ModelPriority::Normal;
            } else if (value == "low") {
                options.model_priority = ServeOptions::ModelPriority::Low;
            } else {
                throw std::invalid_argument("--model-priority takes high, normal or low");
            }
        } else if (arg == "--model") {
            // --model name=path[,kv-tokens=N][,max-num-seqs=N][,max-model-len=N]
            const std::string value = require_value("--model");
            ServeOptions::ExtraModel extra;
            std::size_t cursor = 0;
            bool first         = true;
            while (cursor <= value.size()) {
                const std::size_t comma = value.find(',', cursor);
                const std::string part =
                    value.substr(cursor, comma == std::string::npos ? std::string::npos
                                                                    : comma - cursor);
                const std::size_t eq = part.find('=');
                if (eq == std::string::npos) {
                    throw std::invalid_argument("--model expects name=path[,key=value...]");
                }
                const std::string key = part.substr(0, eq);
                const std::string val = part.substr(eq + 1);
                if (first) {
                    extra.name          = key;
                    extra.artifact_path = val;
                    first               = false;
                } else if (key == "kv-tokens") {
                    extra.kv_tokens = static_cast<std::uint32_t>(std::stoul(val));
                } else if (key == "max-num-seqs") {
                    extra.max_num_seqs = static_cast<std::uint32_t>(std::stoul(val));
                } else if (key == "max-model-len") {
                    extra.max_context = static_cast<std::uint32_t>(std::stoul(val));
                } else if (key == "spec") {
                    if (val == "mtp") {
                        extra.speculative.backend = SpeculativeBackend::Mtp;
                    } else if (val == "dflash") {
                        extra.speculative.backend = SpeculativeBackend::DFlash;
                    } else {
                        throw std::invalid_argument("--model: spec= takes mtp or dflash");
                    }
                    if (extra.speculative.draft_tokens == 0) {
                        extra.speculative.draft_tokens = 3;
                    }
                } else if (key == "priority") {
                    if (val == "high") {
                        extra.priority = ServeOptions::ModelPriority::High;
                    } else if (val == "normal") {
                        extra.priority = ServeOptions::ModelPriority::Normal;
                    } else if (val == "low") {
                        extra.priority = ServeOptions::ModelPriority::Low;
                    } else {
                        throw std::invalid_argument("--model: priority= takes high, normal or low");
                    }
                } else if (key == "lora") {
                    const std::size_t colon = val.find(':');
                    if (colon == std::string::npos || colon == 0 || colon + 1 == val.size()) {
                        throw std::invalid_argument("--model: lora= takes name:path");
                    }
                    extra.lora.push_back({val.substr(0, colon), val.substr(colon + 1)});
                } else if (key == "draft-tokens") {
                    extra.speculative.draft_tokens =
                        static_cast<std::uint32_t>(std::stoul(val));
                } else if (key == "spec-max-lanes") {
                    extra.speculative.max_lanes =
                        val == "all" ? kSpeculateAtAnyWidth
                                     : static_cast<std::uint32_t>(std::stoul(val));
                } else {
                    throw std::invalid_argument(
                        "--model: unknown key '" + key +
                        "' (kv-tokens, max-num-seqs, max-model-len, spec, draft-tokens, "
                        "spec-max-lanes, priority, lora)");
                }
                if (comma == std::string::npos) { break; }
                cursor = comma + 1;
            }
            if (extra.name.empty() || extra.artifact_path.empty()) {
                throw std::invalid_argument("--model expects name=path[,key=value...]");
            }
            options.extra_models.push_back(std::move(extra));
        } else if (arg == "--enable-sleep-mode") {
            options.enable_sleep_mode = true;
        } else if (arg == "--enable-lora") {
            options.enable_lora = true;
        } else if (arg == "--lora-modules") {
            // `name=path` entries, comma separated or repeated, as vLLM accepts them.
            std::string spec = require_value("--lora-modules");
            std::size_t begin = 0;
            while (begin <= spec.size()) {
                const std::size_t comma = spec.find(',', begin);
                const std::string entry =
                    spec.substr(begin, comma == std::string::npos ? std::string::npos
                                                                  : comma - begin);
                if (!entry.empty()) {
                    const std::size_t equals = entry.find('=');
                    if (equals == std::string::npos || equals == 0 ||
                        equals + 1 >= entry.size()) {
                        throw std::invalid_argument(
                            "--lora-modules entries are name=path, got '" + entry + "'");
                    }
                    options.lora_modules.push_back(
                        {entry.substr(0, equals), entry.substr(equals + 1)});
                }
                if (comma == std::string::npos) { break; }
                begin = comma + 1;
            }
        } else if (arg == "--max-loras") {
            options.max_loras = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-loras"), "max-loras"));
        } else if (arg == "--max-lora-rank") {
            options.max_lora_rank = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-lora-rank"), "max-lora-rank"));
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            options.sampling_overrides.top_k =
                parse_nonnegative_int(require_value("--top-k"), "top-k");
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (!max_context_explicit) { options.max_context = 0; } // auto by default
    if (!kv_capacity_explicit) {
        options.kv_capacity = options.max_context == 0
                                  ? KvCapacityPolicy::automatic(kDefaultKvCapacityHeadroomBytes)
                                  : KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.elastic_kv_overcommit && !options.elastic_kv) {
        throw std::invalid_argument("--elastic-kv-overcommit needs the elastic pool; drop --no-elastic-kv");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit && options.max_context != 0 &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-model-len");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        // The bound is kMaximumConcurrency; say so rather than restating a
        // number. This message read "[1,8]" long after the ceiling moved to 64,
        // which sends anyone hitting it looking in the wrong place.
        throw std::invalid_argument("--max-num-seqs must be in [1," +
                                    std::to_string(kMaximumConcurrency) + "]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    // vLLM couples these two: automatic tool choice needs a parser to read the
    // model's calls back, and enabling it without one produces a server that
    // accepts `tool_choice: "auto"` and can never satisfy it.
    if (options.enable_auto_tool_choice && options.tool_call_format == ToolCallFormat::None) {
        throw std::invalid_argument(
            "--enable-auto-tool-choice needs --tool-call-parser (supported: " +
            tool_call_parser_names() + ")");
    }
    if (!options.lora_modules.empty() && !options.enable_lora) {
        throw std::invalid_argument("--lora-modules needs --enable-lora");
    }
    if (!options.extra_models.empty()) {
        std::vector<std::string> names;
        for (const auto& extra : options.extra_models) {
            if (extra.kv_tokens == 0 && !options.elastic_kv) {
                // With arena pools the split must be stated: an extra sized from whatever is
                // free would take the primary's headroom. Elastic pools commit only a cap
                // against a shared ledger, so automatic sizing is safe there.
                throw std::invalid_argument(
                    "--model " + extra.name +
                    ": kv-tokens=N is required under --no-elastic-kv -- with arena pools the "
                    "extras state their KV so the deployment's memory split is explicit");
            }
            for (const auto& seen : names) {
                if (seen == extra.name) {
                    throw std::invalid_argument("--model: duplicate name '" + extra.name + "'");
                }
            }
            names.push_back(extra.name);
        }
        if (options.devices.size() > 1) {
            throw std::invalid_argument(
                "--model extras support single-device serving today (the primary may still "
                "pipeline; run extras on their own devices via their own flags later)");
        }
        // One flat namespace: a request selects by the single `model` string, so
        // every served id and every adapter name -- the primary's and each
        // extra's -- must be distinct. Served-id-vs-adapter collisions that
        // involve the primary's artifact identity are enforced at attach, where
        // that identity is known.
        std::vector<std::string> reserved = names;
        for (const auto& module : options.lora_modules) { reserved.push_back(module.name); }
        for (const auto& extra : options.extra_models) {
            for (const auto& module : extra.lora) { reserved.push_back(module.name); }
        }
        std::sort(reserved.begin(), reserved.end());
        for (std::size_t i = 1; i < reserved.size(); ++i) {
            if (reserved[i] == reserved[i - 1]) {
                throw std::invalid_argument(
                    "'" + reserved[i] +
                    "' is used twice across model and adapter names; every name must be unique "
                    "because a request selects by the single `model` field");
            }
        }
    }
    if (options.enable_sleep_mode && options.devices.size() > 1) {
        throw std::invalid_argument(
            "--enable-sleep-mode supports single-device serving today; pipeline stages would "
            "each need their own sleep transition");
    }
    if (options.enable_lora) {
        // Zero modules is a valid start: adapters can arrive later through
        // POST /v1/load_lora_adapter. The capacity flags still bound them.
        if (options.max_loras == 0) { throw std::invalid_argument("--max-loras must be positive"); }
        if (options.max_lora_rank == 0) {
            throw std::invalid_argument("--max-lora-rank must be positive");
        }
        if (options.lora_modules.size() > options.max_loras) {
            throw std::invalid_argument("--lora-modules names " +
                                        std::to_string(options.lora_modules.size()) +
                                        " adapters but --max-loras is " +
                                        std::to_string(options.max_loras));
        }
        // Adapters run under CUDA graphs, the engine's normal serving mode. Three
        // capture defects were found and fixed on the way here, all one lesson:
        // once a round is captured, every choice the host used to make must reach
        // the kernels as data. The prefill graphs are captured with the delta
        // launches in them; a round writes its slot even when the slot is "none",
        // so a base request cannot inherit the previous request's adapter; and the
        // resident-prefix identity carries the slot its values were computed
        // under, so a request cannot resume on another adapter's cache.
        //
        // The bar for "working" is the engine's own graph mode, measured, not
        // assumed: plain base serving with no adapter loaded flips one near-tied
        // token run to run, so exact output hashes are not something graphs
        // provide with or without LoRA. What LoRA is held to: a B=0 adapter
        // reproduces exactly the base model's output set (isolation), a real
        // adapter's outputs sit outside it (application), and single-token
        // requests are as deterministic with an adapter as without one (no added
        // jitter). All three held over repeated runs.
        std::vector<std::string> seen;
        for (const auto& module : options.lora_modules) {
            if (std::find(seen.begin(), seen.end(), module.name) != seen.end()) {
                throw std::invalid_argument("--lora-modules repeats the name '" + module.name +
                                            "'; a request selects an adapter by name");
            }
            seen.push_back(module.name);
        }
    }
    if (!options.chat_template_path.empty()) {
        std::ifstream file(options.chat_template_path, std::ios::binary);
        if (!file) {
            throw std::invalid_argument("--chat-template: cannot read " +
                                        options.chat_template_path);
        }
        std::ostringstream buffer;
        buffer << file.rdbuf();
        options.chat_template = buffer.str();
        if (options.chat_template.empty()) {
            throw std::invalid_argument("--chat-template: " + options.chat_template_path +
                                        " is empty");
        }
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--max-num-batched-tokens must be a positive multiple of 128");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.speculative.backend == SpeculativeBackend::DFlash && options.enable_vision) {
        throw std::invalid_argument("--spec dflash cannot be combined with --vision");
    }
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_id.empty()) {
        throw std::logic_error("loaded artifact model_id must not be empty");
    }
    return std::string(artifact_model_id);
}

} // namespace sinfer::serve
