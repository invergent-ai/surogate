// --thinking-retention (serve/thinking_retention.h): the reasoning of an answer is filed under a key over the tenant,
// the model, the request's messages and the visible answer, and an assistant turn sent back without it gets it back.
// Requests go through the engine's own parser (parse_chat_completion_request), as the HTTP handler calls it, so the
// tau2-style resend below -- content null, arguments re-dumped with json.dumps' separators, its own call ids -- is the
// wire form a harness sends. No model or GPU.

#include "serve/openai_schema.h"
#include "serve/request.h"
#include "serve/request_log.h"
#include "serve/serve_options.h"
#include "serve/thinking_retention.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using OJson = nlohmann::ordered_json;
using Json  = nlohmann::json;
using namespace sinfer::serve;
using sinfer::ChatRole;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

GenerationRequest parse(const OJson& body) {
    RequestLimits limits;
    limits.default_max_tokens = 0;
    limits.vllm_max_tokens    = true;
    return parse_chat_completion_request(body, limits);
}

OJson message(const char* role, OJson content) { return OJson{{"role", role}, {"content", std::move(content)}}; }

constexpr const char* kSystem = "You are a banking agent. Verify the customer before acting.";
constexpr const char* kUser   = "Please move 12.50 from my savings account A1 to checking.";
constexpr const char* kModel  = "surogate3.7-flash";

// What the engine answered on turn 1: reasoning, no content, one tool call whose arguments are the parser's text.
const std::string kReasoning1 = "The customer wants a transfer. I must verify identity first, so read the clock.";
const ToolCall kCall1{.id = "call_8f2e", .name = "log_verification",
                      .arguments_json = R"({"customer_id":"A1","time":"10:42","method":"dob"})"};
const std::string kReasoning2 = "Verified at 10:42. Now the transfer of 12.5 from A1.";
const ToolCall kCall2{.id = "call_93ab", .name = "transfer", .arguments_json = R"({"amount": 12.5, "from": "A1", "to": "checking"})"};

OJson turn1() {
    return OJson{{"model", kModel},
                 {"messages", OJson::array({message("system", kSystem), message("user", kUser)})}};
}

/// tau2's to_litellm_messages: assistant = content (None) + tool_calls, each call re-dumped with json.dumps' default
/// ", " / ": " separators and keys in the order Python kept, with a call-level "name"; no reasoning.
OJson tau2_call(const std::string& id, const std::string& name, const OJson& arguments) {
    return OJson{{"id", id},
                 {"name", name},
                 {"function", {{"name", name}, {"arguments", arguments.dump()}}},
                 {"type", "function"}};
}

std::string python_dumps(const OJson& value) {
    // json.dumps(dict): ", " between members, ": " after keys, insertion order.
    std::string out = "{";
    bool first      = true;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (!first) { out += ", "; }
        first = false;
        out += OJson(it.key()).dump() + ": " + it.value().dump();
    }
    return out + "}";
}

OJson tau2_assistant(const std::vector<OJson>& calls, OJson content = nullptr) {
    OJson item = {{"role", "assistant"}, {"content", std::move(content)}};
    item["tool_calls"] = calls.empty() ? OJson(nullptr) : OJson(calls);
    return item;
}

OJson turn2(const std::string& call_id = "call_8f2e") {
    OJson call = tau2_call(call_id, "log_verification", OJson());
    // Python key order as the dict came back from json.loads, re-dumped with spaces.
    call["function"]["arguments"] = python_dumps(OJson{{"customer_id", "A1"}, {"time", "10:42"}, {"method", "dob"}});
    return OJson{{"model", kModel},
                 {"messages", OJson::array({message("system", kSystem), message("user", kUser), tau2_assistant({call}),
                                            OJson{{"role", "tool"}, {"content", "logged: A1 10:42"},
                                                  {"tool_call_id", call_id}}})}};
}

void normalization() {
    check(canonical_tool_arguments(R"({"b": 1, "a": {"y": [1, 2], "x": "é"}})") ==
              canonical_tool_arguments("{\"a\":{\"x\":\"\xC3\xA9\",\"y\":[1,2]},\"b\":1}"),
          "arguments: key order, separators and escapes must not matter");
    check(canonical_tool_arguments(R"({"a":1.5,"b":2})") == R"({"a":1.5,"b":2})", "arguments: canonical form");
    check(canonical_tool_arguments("  not json ") == "not json", "arguments: unparsable text is kept, trimmed");
    check(canonical_tool_arguments(R"({"a":1})") != canonical_tool_arguments(R"({"a":1.0})"),
          "arguments: an integer and a float stay different values");

    ChatTurn with_reasoning;
    with_reasoning.role = ChatRole::Assistant;
    with_reasoning.content.push_back(ContentPart{ContentKind::Text, "  Done. \n", "text"});
    with_reasoning.tool_calls.push_back(ToolCall{"call_1", "f", R"({"x": 1})"});
    with_reasoning.reasoning_content     = "thinking";
    with_reasoning.has_reasoning_content = true;
    ChatTurn without = with_reasoning;
    without.reasoning_content.clear();
    without.has_reasoning_content = false;
    without.tool_calls[0].id      = "call_regenerated";
    without.content[0].text       = "Done.";
    check(normalized_turn(with_reasoning) == normalized_turn(without),
          "reasoning, call ids and surrounding whitespace must not be hashed");
    check(normalized_answer("Done.", {ToolCall{"other_id", "f", R"({"x":1})"}}) == normalized_turn(without),
          "an answer hashes as the assistant message a client sends back for it");

    ChatTurn no_content;
    no_content.role = ChatRole::Assistant;
    no_content.tool_calls.push_back(ToolCall{"c", "f", "{}"});
    ChatTurn empty_content = no_content;
    empty_content.content.push_back(ContentPart{ContentKind::Text, "", "text"});
    check(normalized_turn(no_content) == normalized_turn(empty_content), "content null == \"\"");
    check(normalized_answer("", no_content.tool_calls) == normalized_turn(no_content), "empty answer == null content");

    ChatTurn tool_a;
    tool_a.role         = ChatRole::Tool;
    tool_a.tool_call_id = "call_1";
    tool_a.content.push_back(ContentPart{ContentKind::Text, "result", "text"});
    ChatTurn tool_b     = tool_a;
    tool_b.tool_call_id = "call_2";
    check(normalized_turn(tool_a) == normalized_turn(tool_b), "a tool result's tool_call_id is not hashed");
    ChatTurn user = tool_a;
    user.role     = ChatRole::User;
    check(normalized_turn(user) != normalized_turn(tool_a), "the role is hashed");

    // A response body replaces invalid UTF-8 with U+FFFD (openai_schema.cpp dump_lossy); the client can only send
    // that back.
    const std::string cut       = "cut \xE4\xB8";
    const std::string delivered = Json::parse(Json(cut).dump(-1, ' ', false, Json::error_handler_t::replace))
                                      .get<std::string>();
    check(delivered != cut && normalized_answer(cut, {}) == normalized_answer(delivered, {}),
          "an answer's invalid UTF-8 hashes as the replacement the client was given");

    check(thinking_tenant("Bearer key-1", "") == "key-1" && thinking_tenant("bearer   key-1 ", "") == "key-1" &&
              thinking_tenant("", "key-1") == "key-1",
          "the tenant is the presented key, however it is presented");
    check(thinking_tenant("", "") == "", "no key: the anonymous tenant");
}

/// Turn 1 answered, turn 2 sent back the tau2 way: the tool-call turn gets its reasoning back.
void tau2_resend_restores(ThinkingRetentionMode mode) {
    const std::string name = std::string(thinking_retention_mode_name(mode));
    ThinkingRetentionStore store(1 << 20, std::chrono::hours(6));
    const ThinkingDigest scope = thinking_scope("tenant-key", kModel);

    GenerationRequest first = parse(turn1());
    restore_thinking(first, store, mode, scope);
    check(first.thinking_retention.active && first.thinking_retention.restored.empty() &&
              first.thinking_retention.missed == 0,
          name + ": turn 1 has no assistant turn to restore");
    retain_thinking(store, first.thinking_retention, "", {kCall1}, kReasoning1);
    check(store.counters().stored == 1, name + ": turn 1's reasoning is stored");

    GenerationRequest second = parse(turn2());
    check(second.messages.at(2).reasoning_content.empty(), "tau2 sends no reasoning");
    // The resent prefix is the request turn 1 was: the chain agrees up to it.
    check(conversation_digests(scope, second.messages).at(2) == first.thinking_retention.conversation,
          name + ": the resent messages hash as turn 1's request");
    restore_thinking(second, store, mode, scope);
    check(second.thinking_retention.restored.size() == 1 && second.thinking_retention.missed == 0,
          name + ": the tool-call turn is restored");
    check(second.messages.at(2).reasoning_content == kReasoning1 && second.messages.at(2).has_reasoning_content,
          name + ": with its own reasoning");
    check(!second.thinking_retention.restored.at(0).completed_turn &&
              !second.thinking_retention.restored.at(0).had_reasoning_field,
          name + ": a loop turn the client sent without a reasoning field");

    // Regenerated call ids (some clients make their own) still match.
    GenerationRequest regenerated = parse(turn2("toolu_01XYZ"));
    restore_thinking(regenerated, store, mode, scope);
    check(regenerated.thinking_retention.restored.size() == 1, name + ": regenerated call ids still match");

    // An edited earlier message is another conversation.
    OJson edited                         = turn2();
    edited["messages"][1]["content"]     = "Please move 13.50 from my savings account A1 to checking.";
    GenerationRequest other              = parse(edited);
    restore_thinking(other, store, mode, scope);
    check(other.thinking_retention.restored.empty() && other.thinking_retention.missed == 1,
          name + ": a changed prefix misses");
    check(other.messages.at(2).reasoning_content.empty(), name + ": and leaves the turn alone");

    // Turn 2 answered with a second call; turn 3 resends both, then the user speaks again.
    retain_thinking(store, second.thinking_retention, "Verified.", {kCall2}, kReasoning2);
    OJson third = turn2();
    OJson call2 = tau2_call("call_93ab", "transfer", OJson());
    call2["function"]["arguments"] = python_dumps(OJson{{"amount", 12.5}, {"from", "A1"}, {"to", "checking"}});
    third["messages"].push_back(tau2_assistant({call2}, "Verified."));
    third["messages"].push_back(OJson{{"role", "tool"}, {"content", "ok"}, {"tool_call_id", "call_93ab"}});
    third["messages"].push_back(message("assistant", "The transfer is done."));
    third["messages"].push_back(message("user", "Thanks. What is my balance now?"));
    GenerationRequest after_user = parse(third);
    restore_thinking(after_user, store, mode, scope);
    if (mode == ThinkingRetentionMode::Loop) {
        // Every assistant turn precedes the last user message: none is in the current loop.
        check(after_user.thinking_retention.restored.empty() && after_user.thinking_retention.missed == 0,
              "loop: completed turns are not looked up");
    } else {
        // Both tool-call turns come back; the content-only answer was never generated here.
        check(after_user.thinking_retention.restored.size() == 2 && after_user.thinking_retention.missed == 1,
              "all: every completed turn is looked up");
        check(after_user.messages.at(2).reasoning_content == kReasoning1 &&
                  after_user.messages.at(4).reasoning_content == kReasoning2,
              "all: each with its own reasoning");
        check(after_user.thinking_retention.restored.at(0).completed_turn &&
                  after_user.thinking_retention.restored.at(1).completed_turn,
              "all: turns before the last user message are completed turns");
    }
}

void client_reasoning_is_kept() {
    ThinkingRetentionStore store(1 << 20, std::chrono::hours(6));
    const ThinkingDigest scope = thinking_scope("k", kModel);
    GenerationRequest first    = parse(turn1());
    restore_thinking(first, store, ThinkingRetentionMode::All, scope);
    retain_thinking(store, first.thinking_retention, "", {kCall1}, kReasoning1);

    OJson own                        = turn2();
    own["messages"][2]["reasoning"]  = "the client's own copy";
    GenerationRequest with_own       = parse(own);
    restore_thinking(with_own, store, ThinkingRetentionMode::All, scope);
    check(with_own.messages.at(2).reasoning_content == "the client's own copy" &&
              with_own.thinking_retention.restored.empty() && with_own.thinking_retention.missed == 0,
          "a turn that carries reasoning is never changed or looked up");

    OJson empty                              = turn2();
    empty["messages"][2]["reasoning_content"] = "";
    GenerationRequest with_empty             = parse(empty);
    restore_thinking(with_empty, store, ThinkingRetentionMode::All, scope);
    check(with_empty.thinking_retention.restored.size() == 1 &&
              with_empty.thinking_retention.restored.at(0).had_reasoning_field &&
              with_empty.messages.at(2).reasoning_content == kReasoning1,
          "an empty reasoning string is restored, and remembered as sent");
}

void isolation_and_off() {
    ThinkingRetentionStore store(1 << 20, std::chrono::hours(6));
    GenerationRequest first = parse(turn1());
    restore_thinking(first, store, ThinkingRetentionMode::All, thinking_scope("tenant-a", kModel));
    retain_thinking(store, first.thinking_retention, "", {kCall1}, kReasoning1);

    for (const auto& [tenant, model, what] :
         std::vector<std::tuple<std::string, std::string, std::string>>{{"tenant-b", kModel, "another tenant"},
                                                                       {"", kModel, "an anonymous caller"},
                                                                       {"tenant-a", "other-model", "another model"}}) {
        GenerationRequest second = parse(turn2());
        restore_thinking(second, store, ThinkingRetentionMode::All, thinking_scope(tenant, model));
        check(second.thinking_retention.restored.empty() && second.messages.at(2).reasoning_content.empty(),
              "nothing is restored for " + what);
    }

    GenerationRequest off = parse(turn2());
    restore_thinking(off, store, ThinkingRetentionMode::Off, thinking_scope("tenant-a", kModel));
    check(!off.thinking_retention.active && off.messages.at(2).reasoning_content.empty() &&
              !off.messages.at(2).has_reasoning_content,
          "off: the request is left exactly as parsed");

    OJson tokens            = turn2();
    tokens["tokens"]        = OJson::array({1, 2, 3});
    GenerationRequest given = parse(tokens);
    restore_thinking(given, store, ThinkingRetentionMode::All, thinking_scope("tenant-a", kModel));
    check(!given.thinking_retention.active, "a token-id prompt renders no messages and is left alone");

    // An inactive request files nothing.
    const auto stored = store.counters().stored;
    retain_thinking(store, off.thinking_retention, "x", {}, "reasoning");
    check(store.counters().stored == stored, "an inactive request stores nothing");
}

ThinkingDigest key(int n) {
    ThinkingDigest digest{};
    digest[0] = static_cast<std::uint8_t>(n);
    digest[9] = static_cast<std::uint8_t>(n * 7);
    return digest;
}

std::vector<ThinkingLookup> lookups(std::initializer_list<int> ns) {
    std::vector<ThinkingLookup> out;
    for (const int n : ns) { out.push_back(ThinkingLookup{.key = key(n), .call_ids = {}}); }
    return out;
}

void lru_and_ttl() {
    using Clock = ThinkingRetentionStore::Clock;
    Clock::time_point now{};
    const auto clock          = [&now] { return now; };
    const std::size_t charge  = 1000 + ThinkingRetentionStore::kEntryOverheadBytes;
    ThinkingRetentionStore store(3 * charge, std::chrono::seconds(3600), clock);
    const std::string reasoning(1000, 'r');
    for (int i = 1; i <= 3; ++i) { store.store(key(i), "", reasoning); }
    check(store.counters().entries == 3 && store.counters().bytes == 3 * charge, "three entries fit");
    // Using 1 makes 2 the least recently used.
    check(store.find(lookups({1})).at(0).has_value(), "entry 1 is found");
    store.store(key(4), "", reasoning);
    auto found = store.find(lookups({1, 2, 3, 4}));
    check(found[0] && !found[1] && found[2] && found[3], "the least recently used entry is evicted");
    check(store.counters().evictions == 1 && store.counters().entries == 3, "one eviction");

    // Replacing an answer with the same call ids charges it once.
    store.store(key(3), "", std::string(500, 'n'));
    check(store.counters().entries == 3 && store.find(lookups({3})).at(0) == std::string(500, 'n'),
          "a key stored again holds the newest reasoning");
    check(store.counters().bytes == 2 * charge + 500 + ThinkingRetentionStore::kEntryOverheadBytes,
          "and is charged once");

    // Larger than the whole store, or empty: not kept.
    store.store(key(9), "", std::string(4 * charge, 'x'));
    store.store(key(10), "", "");
    found = store.find(lookups({9, 10}));
    check(!found[0] && !found[1] && store.counters().evictions == 1, "oversized and empty reasoning is not kept");

    // TTL from the last use: an entry used within the TTL stays, one left alone longer is dropped.
    ThinkingRetentionStore aging(1 << 20, std::chrono::seconds(100), clock);
    now = Clock::time_point{};
    aging.store(key(1), "", "a");
    aging.store(key(2), "", "b");
    now += std::chrono::seconds(60);
    check(aging.find(lookups({1})).at(0).has_value(), "found within the TTL");
    now += std::chrono::seconds(60); // 120 s after storing, 60 s after entry 1 was used
    found = aging.find(lookups({1, 2}));
    check(found[0] && !found[1], "the TTL counts from the last use");
    check(aging.counters().expired == 1 && aging.counters().entries == 1, "the stale entry is gone, not just hidden");
    now += std::chrono::seconds(101);
    aging.store(key(3), "", "c"); // a store sweeps too
    check(aging.counters().expired == 2 && aging.counters().entries == 1, "a store drops what expired");
}

/// Two attempts at one task made the same call with different reasoning: same key, told apart by the call ids.
void parallel_attempts() {
    ThinkingRetentionStore store(1 << 20, std::chrono::hours(6));
    const ThinkingDigest scope = thinking_scope("k", kModel);
    GenerationRequest first    = parse(turn1());
    restore_thinking(first, store, ThinkingRetentionMode::All, scope);
    ToolCall attempt_a = kCall1, attempt_b = kCall1;
    attempt_a.id = "call_aaaa";
    attempt_b.id = "call_bbbb";
    retain_thinking(store, first.thinking_retention, "", {attempt_a}, "attempt A's reasoning");
    retain_thinking(store, first.thinking_retention, "", {attempt_b}, "attempt B's reasoning");
    check(store.counters().entries == 1, "the same visible answer is one key");

    for (const auto& [id, expected] : std::vector<std::pair<std::string, std::string>>{
             {"call_aaaa", "attempt A's reasoning"}, {"call_bbbb", "attempt B's reasoning"},
             {"tau2_made_this_id", "attempt B's reasoning"}}) {
        GenerationRequest second = parse(turn2(id));
        restore_thinking(second, store, ThinkingRetentionMode::All, scope);
        check(second.messages.at(2).reasoning_content == expected,
              "call id " + id + " gets " + expected + ", got '" + second.messages.at(2).reasoning_content + "'");
    }

    // Only the last kVariantsPerKey answers are kept.
    for (std::size_t i = 0; i < ThinkingRetentionStore::kVariantsPerKey + 2; ++i) {
        ToolCall call = kCall1;
        call.id       = "call_extra_" + std::to_string(i);
        retain_thinking(store, first.thinking_retention, "", {call}, "extra " + std::to_string(i));
    }
    GenerationRequest oldest = parse(turn2("call_aaaa"));
    restore_thinking(oldest, store, ThinkingRetentionMode::All, scope);
    check(oldest.messages.at(2).reasoning_content ==
              "extra " + std::to_string(ThinkingRetentionStore::kVariantsPerKey + 1),
          "an answer pushed out of its key's variants falls back to the latest");
}

void context_budget() {
    check(thinking_retention_prompt_limit(262144, 16384) == 262144 - 16384, "max_tokens is kept free");
    check(thinking_retention_prompt_limit(262144, 0) == 262144 - 32768, "no output limit: 32768 kept free");
    check(thinking_retention_prompt_limit(8192, 0) == 8192 - 2048, "no output limit, small context: a quarter");
    check(restored_turns_to_drop(1000, 1000, {50, 60}) == 0, "a prompt that fits keeps everything");
    check(restored_turns_to_drop(1000, 900, {50, 60, 70}) == 2, "oldest first, the fewest that cover the excess");
    check(restored_turns_to_drop(1000, 950, {80, 60}) == 1, "one turn can be enough");
    check(restored_turns_to_drop(5000, 900, {50, 60, 70}) == 3, "all of them when even that is not enough");
    check(restored_turns_to_drop(1000, 999, {0, 0}) == 2, "a reasoning measured at nothing still goes");
    check(restored_turns_to_drop(1000, 900, {}) == 0, "nothing restored, nothing to drop");
}

ServeOptions options_from(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), {"surogate-engine", "model.sinfer"});
    std::vector<char*> argv;
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

bool refused(std::vector<std::string> arguments) {
    try {
        (void)options_from(std::move(arguments));
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

void flags_and_log() {
    const ServeOptions defaults = options_from({});
    check(defaults.thinking_retention == ThinkingRetentionMode::Off &&
              defaults.thinking_retention_bytes == (512ULL << 20) && defaults.thinking_retention_ttl_seconds == 21600,
          "off by default, 512 MiB, 6 h");
    const ServeOptions on =
        options_from({"--thinking-retention", "all", "--thinking-retention-mb", "64", "--thinking-retention-ttl", "30m"});
    check(on.thinking_retention == ThinkingRetentionMode::All && on.thinking_retention_bytes == (64ULL << 20) &&
              on.thinking_retention_ttl_seconds == 1800,
          "the flags are read");
    check(options_from({"--thinking-retention", "loop", "--thinking-retention-ttl", "90"}).thinking_retention_ttl_seconds == 90,
          "a TTL in plain seconds");
    check(refused({"--thinking-retention", "some"}) && refused({"--thinking-retention-mb", "0"}) &&
              refused({"--thinking-retention-ttl", "0"}) && refused({"--thinking-retention-ttl", "6w"}),
          "bad values are refused");

    RequestLogContext context;
    context.id                 = 7;
    context.protocol           = "openai_chat_completions";
    context.thinking_retention = true;
    context.thinking_restored  = 3;
    context.thinking_missed    = 1;
    context.thinking_dropped   = 2;
    check(format_request_start(context).find("thinking_retention=[restored=3 missed=1 dropped=2]") != std::string::npos,
          "the console line carries the request's restoration");
    const Json record = Json::parse(format_request_start_json("serve-test", 1, context));
    check(record.at("request").at("thinking_retention") == Json{{"restored", 3}, {"missed", 1}, {"dropped", 2}},
          "the JSON record carries it too");
    context.thinking_retention = false;
    check(format_request_start(context).find("thinking_retention") == std::string::npos &&
              !Json::parse(format_request_start_json("serve-test", 1, context)).at("request").contains("thinking_retention"),
          "nothing is logged when retention is off");
}

} // namespace

int main() {
    normalization();
    tau2_resend_restores(ThinkingRetentionMode::Loop);
    tau2_resend_restores(ThinkingRetentionMode::All);
    client_reasoning_is_kept();
    isolation_and_off();
    lru_and_ttl();
    parallel_attempts();
    context_budget();
    flags_and_log();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
