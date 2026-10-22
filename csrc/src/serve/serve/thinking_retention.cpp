#include "serve/thinking_retention.h"

#include "family/impl/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <utility>

namespace sinfer::serve {
namespace {

using Json = nlohmann::json;

/// What a key starts from. A new version tag files every entry anew, so a change to the normalized form can never
/// match an entry hashed under the old one.
constexpr std::string_view kScopeTag = "surogate.thinking-retention.v1";

/// The restoration budget of a request that names no output limit at all.
constexpr std::int64_t kUnstatedOutputReserve = 32768;

ThinkingDigest digest_of(std::string_view bytes) { return family::frontend_internal::sha256(bytes); }

void put_number(std::string& out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) { out.push_back(static_cast<char>((value >> (8 * i)) & 0xFFU)); }
}

/// Length-prefixed, so ("ab", "c") and ("a", "bc") differ.
void put_text(std::string& out, std::string_view text) {
    put_number(out, text.size());
    out.append(text);
}

std::string_view trim(std::string_view text) {
    const auto space = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
    while (!text.empty() && space(text.front())) { text.remove_prefix(1); }
    while (!text.empty() && space(text.back())) { text.remove_suffix(1); }
    return text;
}

/// Well-formed UTF-8 (RFC 3629: no overlong forms, no surrogates, nothing past U+10FFFF).
bool valid_utf8(std::string_view text) {
    const auto* p   = reinterpret_cast<const unsigned char*>(text.data());
    const auto* end = p + text.size();
    while (p < end) {
        const unsigned char lead = *p;
        if (lead < 0x80U) {
            ++p;
            continue;
        }
        std::size_t length = 0;
        unsigned char low = 0x80U, high = 0xBFU; // the range of the first continuation byte
        if (lead >= 0xC2U && lead <= 0xDFU) {
            length = 2;
        } else if (lead >= 0xE0U && lead <= 0xEFU) {
            length = 3;
            if (lead == 0xE0U) { low = 0xA0U; }
            if (lead == 0xEDU) { high = 0x9FU; }
        } else if (lead >= 0xF0U && lead <= 0xF4U) {
            length = 4;
            if (lead == 0xF0U) { low = 0x90U; }
            if (lead == 0xF4U) { high = 0x8FU; }
        } else {
            return false;
        }
        if (static_cast<std::size_t>(end - p) < length || p[1] < low || p[1] > high) { return false; }
        for (std::size_t i = 2; i < length; ++i) {
            if ((p[i] & 0xC0U) != 0x80U) { return false; }
        }
        p += length;
    }
    return true;
}

/// The text a client is given for `text`: a response body is dumped with invalid UTF-8 replaced by U+FFFD
/// (openai_schema.cpp dump_lossy), so the same replacement is what it can send back. A request's own strings are
/// valid UTF-8 already (the JSON parser refuses anything else) and come back unchanged.
std::string as_delivered(std::string_view text) {
    if (valid_utf8(text)) { return std::string(text); }
    const std::string dumped = Json(std::string(text)).dump(-1, ' ', false, Json::error_handler_t::replace);
    return Json::parse(dumped).get<std::string>();
}

std::string normalized_content(const std::vector<ContentPart>& parts) {
    std::string text;
    for (const ContentPart& part : parts) {
        if (part.kind == ContentKind::Text) {
            text += part.text;
            continue;
        }
        // A media part is what the client named: its type and URL (or data URI).
        text += '\0';
        text += part.type_raw;
        text += '\0';
        text += part.source.value;
        text += '\0';
    }
    return std::string(trim(as_delivered(text)));
}

void put_turn(std::string& out, ChatRole role, std::string_view content, const std::vector<ToolCall>& tool_calls) {
    out.push_back(static_cast<char>(role));
    put_text(out, content);
    put_number(out, tool_calls.size());
    for (const ToolCall& call : tool_calls) {
        put_text(out, as_delivered(call.name));
        put_text(out, canonical_tool_arguments(call.arguments_json));
    }
}

ThinkingDigest chain(const ThinkingDigest& previous, std::string_view normalized) {
    std::string bytes;
    bytes.reserve(previous.size() + normalized.size());
    bytes.append(reinterpret_cast<const char*>(previous.data()), previous.size());
    bytes.append(normalized);
    return digest_of(bytes);
}

} // namespace

std::optional<ThinkingRetentionMode> parse_thinking_retention_mode(std::string_view value) noexcept {
    if (value == "off") { return ThinkingRetentionMode::Off; }
    if (value == "loop") { return ThinkingRetentionMode::Loop; }
    if (value == "all") { return ThinkingRetentionMode::All; }
    return std::nullopt;
}

std::string_view thinking_retention_mode_name(ThinkingRetentionMode mode) noexcept {
    switch (mode) {
    case ThinkingRetentionMode::Off:
        return "off";
    case ThinkingRetentionMode::Loop:
        return "loop";
    case ThinkingRetentionMode::All:
        return "all";
    }
    return "off";
}

std::string thinking_tenant(std::string_view authorization, std::string_view x_api_key) {
    constexpr std::string_view scheme = "bearer ";
    authorization                     = trim(authorization);
    if (authorization.size() > scheme.size() &&
        std::equal(scheme.begin(), scheme.end(), authorization.begin(), [](char expected, char given) {
            return expected == std::tolower(static_cast<unsigned char>(given));
        })) {
        const std::string_view key = trim(authorization.substr(scheme.size()));
        if (!key.empty()) { return std::string(key); }
    }
    if (const std::string_view key = trim(x_api_key); !key.empty()) { return std::string(key); }
    // Any other credential, as written: a gateway in front of an unauthenticated server may use its own scheme.
    return std::string(authorization);
}

ThinkingDigest thinking_scope(std::string_view tenant, std::string_view model) {
    const ThinkingDigest tenant_digest = digest_of(tenant);
    std::string bytes;
    put_text(bytes, kScopeTag);
    put_text(bytes, std::string_view(reinterpret_cast<const char*>(tenant_digest.data()), tenant_digest.size()));
    put_text(bytes, model);
    return digest_of(bytes);
}

std::string canonical_tool_arguments(std::string_view arguments) {
    const std::string delivered = as_delivered(arguments);
    const Json parsed           = Json::parse(delivered, nullptr, /*allow_exceptions=*/false);
    if (parsed.is_discarded()) { return std::string(trim(delivered)); }
    // nlohmann::json keeps object members in a sorted map: dump() is sorted keys and compact separators.
    return parsed.dump(-1, ' ', false, Json::error_handler_t::replace);
}

std::string normalized_turn(const ChatTurn& turn) {
    std::string out;
    put_turn(out, turn.role, normalized_content(turn.content), turn.tool_calls);
    return out;
}

std::string normalized_answer(std::string_view content, const std::vector<ToolCall>& tool_calls) {
    std::string out;
    put_turn(out, ChatRole::Assistant, trim(as_delivered(content)), tool_calls);
    return out;
}

std::vector<ThinkingDigest> conversation_digests(const ThinkingDigest& scope, const std::vector<ChatTurn>& messages) {
    std::vector<ThinkingDigest> digests;
    digests.reserve(messages.size() + 1);
    digests.push_back(scope);
    for (const ChatTurn& turn : messages) { digests.push_back(chain(digests.back(), normalized_turn(turn))); }
    return digests;
}

ThinkingDigest answer_key(const ThinkingDigest& conversation, std::string_view content,
                          const std::vector<ToolCall>& tool_calls) {
    return chain(conversation, normalized_answer(content, tool_calls));
}

std::int64_t thinking_retention_prompt_limit(std::int64_t max_model_len, int max_tokens) {
    const std::int64_t reserve =
        max_tokens > 0 ? static_cast<std::int64_t>(max_tokens) : std::min(kUnstatedOutputReserve, max_model_len / 4);
    return max_model_len - reserve;
}

std::size_t restored_turns_to_drop(std::int64_t prompt_tokens, std::int64_t limit,
                                   const std::vector<std::size_t>& restored_tokens) {
    if (prompt_tokens <= limit || restored_tokens.empty()) { return 0; }
    const std::int64_t excess = prompt_tokens - limit;
    std::int64_t freed        = 0;
    std::size_t count         = 0;
    while (count < restored_tokens.size() && freed < excess) {
        freed += static_cast<std::int64_t>(restored_tokens[count]);
        ++count;
    }
    return std::max<std::size_t>(count, 1);
}

std::string thinking_call_ids(const std::vector<ToolCall>& tool_calls) {
    std::string ids;
    for (const ToolCall& call : tool_calls) { put_text(ids, call.id); }
    return ids;
}

std::size_t ThinkingRetentionStore::DigestHash::operator()(const ThinkingDigest& digest) const noexcept {
    std::size_t value = 0;
    std::memcpy(&value, digest.data(), sizeof value);
    return value;
}

ThinkingRetentionStore::ThinkingRetentionStore(std::size_t capacity_bytes, std::chrono::seconds ttl,
                                               std::function<Clock::time_point()> now)
    : capacity_bytes_(capacity_bytes), ttl_(ttl), now_(std::move(now)) {}

ThinkingRetentionStore::Clock::time_point ThinkingRetentionStore::now() const {
    return now_ ? now_() : Clock::now();
}

std::size_t ThinkingRetentionStore::charge_of(const std::vector<Variant>& variants) noexcept {
    std::size_t charge = kEntryOverheadBytes;
    for (const Variant& variant : variants) { charge += variant.call_ids.size() + variant.reasoning.size(); }
    return charge;
}

void ThinkingRetentionStore::erase(std::unordered_map<ThinkingDigest, Entry, DigestHash>::iterator entry) {
    bytes_ -= entry->second.charge;
    lru_.erase(entry->second.lru);
    entries_.erase(entry);
}

void ThinkingRetentionStore::expire(Clock::time_point now) {
    while (!lru_.empty()) {
        const auto oldest = entries_.find(lru_.back());
        if (now - oldest->second.used <= ttl_) { return; }
        erase(oldest);
        ++counters_.expired;
    }
}

void ThinkingRetentionStore::store(const ThinkingDigest& key, const std::string& call_ids, std::string reasoning) {
    if (reasoning.empty()) { return; }
    const auto at = now();
    const std::lock_guard lock(mutex_);
    expire(at);
    std::vector<Variant> variants;
    if (const auto existing = entries_.find(key); existing != entries_.end()) {
        variants = std::move(existing->second.variants);
        erase(existing);
        std::erase_if(variants, [&](const Variant& variant) { return variant.call_ids == call_ids; });
    }
    variants.insert(variants.begin(), Variant{call_ids, std::move(reasoning)});
    if (variants.size() > kVariantsPerKey) { variants.resize(kVariantsPerKey); }
    // Older answers go before the newest one does: it is what the conversation's next request most likely holds.
    while (variants.size() > 1 && charge_of(variants) > capacity_bytes_) { variants.pop_back(); }
    const std::size_t charge = charge_of(variants);
    if (charge > capacity_bytes_) { return; }
    while (!lru_.empty() && bytes_ + charge > capacity_bytes_) {
        erase(entries_.find(lru_.back()));
        ++counters_.evictions;
    }
    lru_.push_front(key);
    entries_.emplace(key, Entry{std::move(variants), charge, at, lru_.begin()});
    bytes_ += charge;
    ++counters_.stored;
}

std::vector<std::optional<std::string>> ThinkingRetentionStore::find(const std::vector<ThinkingLookup>& lookups) {
    std::vector<std::optional<std::string>> found(lookups.size());
    if (lookups.empty()) { return found; }
    const auto at = now();
    const std::lock_guard lock(mutex_);
    expire(at);
    for (std::size_t i = 0; i < lookups.size(); ++i) {
        const auto entry = entries_.find(lookups[i].key);
        if (entry == entries_.end()) { continue; }
        entry->second.used = at;
        lru_.splice(lru_.begin(), lru_, entry->second.lru);
        const std::vector<Variant>& variants = entry->second.variants;
        const auto same_ids = std::find_if(variants.begin(), variants.end(), [&](const Variant& variant) {
            return variant.call_ids == lookups[i].call_ids;
        });
        found[i] = (same_ids != variants.end() ? *same_ids : variants.front()).reasoning;
    }
    return found;
}

void ThinkingRetentionStore::count(std::size_t restored, std::size_t misses, std::size_t dropped) {
    const std::lock_guard lock(mutex_);
    counters_.restored += restored;
    counters_.misses += misses;
    counters_.dropped += dropped;
}

ThinkingRetentionCounters ThinkingRetentionStore::counters() const {
    const std::lock_guard lock(mutex_);
    ThinkingRetentionCounters result = counters_;
    result.entries                   = entries_.size();
    result.bytes                     = bytes_;
    return result;
}

void restore_thinking(GenerationRequest& request, ThinkingRetentionStore& store, ThinkingRetentionMode mode,
                      const ThinkingDigest& scope) {
    ThinkingRetentionState& state = request.thinking_retention;
    state                         = {};
    if (mode == ThinkingRetentionMode::Off || request.messages.empty() || request.raw_prompt.has_value() ||
        !request.prompt_token_ids.empty()) {
        return;
    }
    const std::vector<ThinkingDigest> digests = conversation_digests(scope, request.messages);
    state.active                              = true;
    state.conversation                        = digests.back();

    std::size_t first_turn = 0;
    std::optional<std::size_t> last_user;
    for (std::size_t i = 0; i < request.messages.size(); ++i) {
        if (request.messages[i].role == ChatRole::User) { last_user = i; }
    }
    if (mode == ThinkingRetentionMode::Loop && last_user) { first_turn = *last_user + 1; }

    std::vector<std::size_t> turns;
    std::vector<ThinkingLookup> lookups;
    for (std::size_t i = first_turn; i < request.messages.size(); ++i) {
        const ChatTurn& turn = request.messages[i];
        // A message that carries reasoning of its own is the client's to keep, also when an empty `reasoning` won
        // over a `reasoning_content` with text.
        if (turn.role != ChatRole::Assistant || !turn.reasoning_content.empty() || turn.sent_reasoning_text) {
            continue;
        }
        turns.push_back(i);
        lookups.push_back(ThinkingLookup{.key = digests[i + 1], .call_ids = thinking_call_ids(turn.tool_calls)});
    }
    std::vector<std::optional<std::string>> found = store.find(lookups);
    for (std::size_t k = 0; k < turns.size(); ++k) {
        if (!found[k]) {
            ++state.missed;
            continue;
        }
        ChatTurn& turn = request.messages[turns[k]];
        state.restored.push_back(ThinkingRetentionState::Restored{
            .message             = turns[k],
            .had_reasoning_field = turn.has_reasoning_content,
            .completed_turn      = last_user && turns[k] < *last_user});
        turn.reasoning_content     = std::move(*found[k]);
        turn.has_reasoning_content = true;
    }
}

void retain_thinking(ThinkingRetentionStore& store, const ThinkingRetentionState& retention, std::string_view content,
                     const std::vector<ToolCall>& tool_calls, std::string reasoning) {
    if (!retention.active || reasoning.empty()) { return; }
    // The reasoning as the client was given it: a generation cut at max_tokens can end inside a character, and the
    // response replaced those bytes; restored as they were they would not even render (a Jinja template's messages
    // are JSON).
    store.store(answer_key(retention.conversation, content, tool_calls), thinking_call_ids(tool_calls),
                as_delivered(reasoning));
}

} // namespace sinfer::serve
