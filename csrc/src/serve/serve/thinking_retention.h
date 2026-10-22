#pragma once

// Server-side thinking retention (--thinking-retention): the reasoning this server generated for a chat completion,
// kept in RAM and put back into a later request of the same conversation whose client sent that turn back without it.
//
// Agent harnesses resend the whole conversation every turn, and many rebuild the assistant turns from their content and
// tool calls only: tau2 v1.0.1's to_litellm_messages (src/tau2/utils/llm_utils.py) writes an assistant message as
// {"content", "tool_calls"} and nothing else. A template that renders every turn's reasoning then shows the model an
// empty think block for each of its own earlier turns. Surogate3.7 was fine-tuned with all of its earlier reasoning in
// view and uses it as working memory: on replays of its own tau3 dev48 requests at the identity-log step it logged the
// verification without reading the clock 56.3 % of the time with the reasoning dropped and 32.5 % with its own logged
// reasoning put back (n=208), and dropping the reasoning took it from a tie with Occamy to -16 points on the t3style dev
// gate (2026-10-22). `preserve_thinking` cannot help there: it renders the reasoning a client sends, and these send none.
//
// Store: every finished chat completion that produced reasoning is kept under
//   key = SHA-256 chain over (tenant, served model, the request's messages, the visible answer)
// where the tenant is the API key the request presented (hashed into the key; never stored) and the model the name it
// asked for. Restore: an assistant turn of a later request whose reasoning is absent or empty is looked up under the
// same key built from the messages before it and the turn itself, and gets the stored reasoning back.
//
// Both sides hash a normalized form, so a client that re-serializes what it was given still matches:
//   - reasoning fields are left out of every message;
//   - content: the text parts joined, surrounding whitespace trimmed, absent / null == ""; a media part is its URL;
//   - tool calls: the name and the arguments parsed and dumped canonically (sorted keys, compact separators: tau2
//     re-dumps them with json.dumps' ", " / ": ", other clients compact); call ids are left out (some clients make
//     their own), and so is a tool message's tool_call_id; a tool message is its content as text;
//   - the visible answer is what the response carried: content (invalid UTF-8 replaced as the response body replaces
//     it) and the tool calls.
// Entries live in one process-wide store shared by every replica and model of the server: an LRU with a byte cap
// (--thinking-retention-mb) and a TTL counted from an entry's last use (--thinking-retention-ttl). A key covers the
// tenant and the model, and an entry is only ever found under its own key, so nothing crosses tenants or models; a
// message that carries reasoning of its own is never changed. Two answers can share a key -- parallel attempts at one
// task that made the same call with different reasoning -- so an entry keeps the last few, each with its tool calls'
// ids, and a turn sent back with the ids it was given gets its own; a client that made new ids gets the latest.
//
// Modes: `loop` restores only the assistant turns after the last user message (the turns a Qwen template renders with
// their reasoning anyway); `all` restores every turn, and a restored turn before the last user message turns on
// preserve_thinking where the template shows completed turns' reasoning only under it (Qwen3.6's; Surogate3.7's shows
// every turn's anyway) and the request did not set it. A template takes the switch for the whole prompt, so the
// reasoning a client sent itself for a completed turn is then shown too, as the model was trained to see it. `off`
// (the default) neither stores nor restores anything: requests are served byte for byte as before.
//
// Context: restoration never lets a prompt grow past the context less the request's output budget (its max_tokens, else
// --default-max-tokens, else min(32768, context / 4)); GenerationService::prepare drops restored reasoning again,
// oldest turn first, until it fits, and a request whose every restored turn had to go is the request as it was sent.
//
// Scope: /v1/chat/completions (streaming or not) with messages. Token-id prompts (`tokens`), /v1/completions, the
// Responses and Anthropic Messages APIs and /tokenize are neither stored nor restored.

#include "serve/request.h"
#include "serve/serve_options.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sinfer::serve {

using ThinkingDigest = std::array<std::uint8_t, 32>;

[[nodiscard]] std::optional<ThinkingRetentionMode> parse_thinking_retention_mode(std::string_view value) noexcept;
[[nodiscard]] std::string_view thinking_retention_mode_name(ThinkingRetentionMode mode) noexcept;

/// The tenant a request speaks for: the API key it presented, as `Authorization: Bearer KEY` (the scheme matched
/// without regard to case) or else `x-api-key: KEY`. Empty when it presented none.
[[nodiscard]] std::string thinking_tenant(std::string_view authorization, std::string_view x_api_key);

/// Where every key of one tenant and one served model starts: SHA-256 over a version tag, the tenant's own SHA-256
/// and the model name, so the key a conversation is filed under cannot be reached from another tenant or model.
[[nodiscard]] ThinkingDigest thinking_scope(std::string_view tenant, std::string_view model);

/// Tool-call arguments as both sides hash them: parsed as JSON and dumped with sorted keys and no spaces; text that
/// does not parse is kept as written (trimmed). Invalid UTF-8 is replaced first, as the response body replaces it.
[[nodiscard]] std::string canonical_tool_arguments(std::string_view arguments);
/// One message as it is hashed: role, normalized content, and the tool calls' names and canonical arguments. Reasoning,
/// call ids and tool_call_id are left out.
[[nodiscard]] std::string normalized_turn(const ChatTurn& turn);
/// A generated answer as it is hashed: the assistant message a client gets back from it (`content`, `tool_calls`),
/// byte for byte the normalized_turn of that message however the client spells it.
[[nodiscard]] std::string normalized_answer(std::string_view content, const std::vector<ToolCall>& tool_calls);

/// The digest chain of a conversation: d[0] = `scope`, d[i + 1] = SHA-256(d[i] || normalized_turn(messages[i])).
/// d[i + 1] is the key an assistant messages[i] is looked up under, and d.back() is where the request's own answer is
/// filed (answer_key).
[[nodiscard]] std::vector<ThinkingDigest> conversation_digests(const ThinkingDigest& scope,
                                                               const std::vector<ChatTurn>& messages);
[[nodiscard]] ThinkingDigest answer_key(const ThinkingDigest& conversation, std::string_view content,
                                        const std::vector<ToolCall>& tool_calls);

/// The prompt length restoration may grow a request to: the context less the output it asked for (`max_tokens` as
/// parsed: the client's, else --default-max-tokens), or less min(32768, context / 4) when neither says.
[[nodiscard]] std::int64_t thinking_retention_prompt_limit(std::int64_t max_model_len, int max_tokens);
/// How many restored turns to take back, oldest first, so a prompt of `prompt_tokens` fits `limit`: the fewest
/// leading entries of `restored_tokens` (each one's reasoning in tokens) that cover the excess, at least one, and
/// all of them when even that is not enough. 0 when the prompt already fits.
[[nodiscard]] std::size_t restored_turns_to_drop(std::int64_t prompt_tokens, std::int64_t limit,
                                                 const std::vector<std::size_t>& restored_tokens);

struct ThinkingRetentionCounters {
    std::uint64_t stored    = 0; ///< answers whose reasoning was kept
    std::uint64_t restored  = 0; ///< assistant turns whose prompt carried restored reasoning
    std::uint64_t misses    = 0; ///< assistant turns without reasoning that had none stored
    std::uint64_t dropped   = 0; ///< restored turns taken back so the prompt fits the context
    std::uint64_t evictions = 0; ///< entries dropped for room (least recently used first)
    std::uint64_t expired   = 0; ///< entries dropped after --thinking-retention-ttl without use
    std::uint64_t entries   = 0; ///< entries held now
    std::uint64_t bytes     = 0; ///< bytes they are charged (reasoning + a fixed per-entry overhead)
};

/// One lookup: the key of an assistant turn, and the ids of its tool calls as the client sent them back.
struct ThinkingLookup {
    ThinkingDigest key{};
    std::string call_ids;
};

/// The ids of an answer's tool calls, one string (empty without calls). Not part of the key -- some clients make
/// their own ids -- but they tell apart two answers with the same visible content and calls and different reasoning
/// (parallel attempts at one task): an entry keeps up to kVariantsPerKey of them, and a turn whose ids match one gets
/// that one's reasoning; any other turn gets the latest.
[[nodiscard]] std::string thinking_call_ids(const std::vector<ToolCall>& tool_calls);

class ThinkingRetentionStore {
public:
    using Clock = std::chrono::steady_clock;

    /// What one entry is charged beyond its reasoning: the key, the hash and list nodes, the string headers.
    static constexpr std::size_t kEntryOverheadBytes = 160;
    /// Answers kept under one key (same conversation, same visible answer), newest first.
    static constexpr std::size_t kVariantsPerKey = 4;

    /// `now` stands in for the clock in tests.
    ThinkingRetentionStore(std::size_t capacity_bytes, std::chrono::seconds ttl,
                           std::function<Clock::time_point()> now = {});

    /// Keep `reasoning` under `key` as the newest answer with these call ids (replacing one with the same ids), and
    /// evict the least recently used entries until the store is within its capacity. Empty reasoning, or an entry
    /// larger than the whole capacity, is not kept.
    void store(const ThinkingDigest& key, const std::string& call_ids, std::string reasoning);
    /// The reasoning for each lookup, nullopt where there is none; every entry found counts as used now. One lock for
    /// the whole request.
    [[nodiscard]] std::vector<std::optional<std::string>> find(const std::vector<ThinkingLookup>& lookups);
    /// What a request's restoration came to, for the counters: turns that went out restored, turns looked up and not
    /// found, restored turns taken back for the context.
    void count(std::size_t restored, std::size_t misses, std::size_t dropped);
    [[nodiscard]] ThinkingRetentionCounters counters() const;

private:
    struct DigestHash {
        std::size_t operator()(const ThinkingDigest& digest) const noexcept;
    };
    struct Variant {
        std::string call_ids;
        std::string reasoning;
    };
    struct Entry {
        std::vector<Variant> variants; // newest first
        std::size_t charge = 0;
        Clock::time_point used;
        std::list<ThinkingDigest>::iterator lru;
    };
    [[nodiscard]] static std::size_t charge_of(const std::vector<Variant>& variants) noexcept;

    [[nodiscard]] Clock::time_point now() const;
    void erase(std::unordered_map<ThinkingDigest, Entry, DigestHash>::iterator entry);
    /// Drop every entry unused for longer than the TTL: the tail of the LRU list.
    void expire(Clock::time_point now);

    const std::size_t capacity_bytes_;
    const std::chrono::seconds ttl_;
    std::function<Clock::time_point()> now_;
    mutable std::mutex mutex_;
    std::unordered_map<ThinkingDigest, Entry, DigestHash> entries_;
    std::list<ThinkingDigest> lru_; // front: most recently used
    std::size_t bytes_ = 0;
    ThinkingRetentionCounters counters_;
};

/// Look up every assistant turn of `request` the mode covers whose reasoning is absent or empty, and attach what was
/// stored for it (request.thinking_retention records which turns and how many were missed). A request with no messages
/// to render -- token ids, a raw prompt -- is left alone and inactive.
void restore_thinking(GenerationRequest& request, ThinkingRetentionStore& store, ThinkingRetentionMode mode,
                      const ThinkingDigest& scope);

/// File the reasoning of the answer to a looked-up request (`retention.active`) under the key a later request will
/// look it up by, as the response delivered it (invalid UTF-8 replaced). Nothing is kept for an empty reasoning.
void retain_thinking(ThinkingRetentionStore& store, const ThinkingRetentionState& retention, std::string_view content,
                     const std::vector<ToolCall>& tool_calls, std::string reasoning);

} // namespace sinfer::serve
