// Opt-in HTTP regression with two resident models on separate devices.
#include "serve/http_server.h"
#include "serve/anthropic_schema.h"

#include <cassert>
#include <future>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

using namespace sinfer;
using namespace sinfer::serve;
using namespace std::chrono_literals;
using Json = nlohmann::json;

namespace sinfer::serve {
struct HttpServerTestAccess {
    static std::unique_lock<std::mutex> lock_adapters(HttpServer& server) {
        return std::unique_lock(server.adapter_management_mutex_);
    }
};
}

static int unused_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    assert(bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
    socklen_t size = sizeof(address);
    assert(getsockname(fd, reinterpret_cast<sockaddr*>(&address), &size) == 0);
    close(fd);
    return ntohs(address.sin_port);
}

int main() {
    const char* artifact = std::getenv("SUROGATE_MODEL_ROUTING_ARTIFACT");
    const char* adapter = std::getenv("SUROGATE_MODEL_ROUTING_ADAPTER");
    if (!artifact || !adapter) { return 77; }
    ServeOptions options;
    options.artifact_path = artifact;
    options.model_id_override = "primary";
    options.port = unused_port();
    options.max_context = 512;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(512);
    options.max_concurrency = 1;
    options.enable_lora = true;
    options.enable_sleep_mode = true;
    options.max_loras = 2;
    options.max_lora_rank = 8;
    options.log_stats_interval_ms = 0;
    options.use_cuda_graph = false;
    GenerationService primary(options);
    auto extra_options = options;
    extra_options.device = 1;
    extra_options.model_id_override = "extra";
    extra_options.chat_template = "Count from one to ten: 1, 2, 3, 4, 5, 6, 7, 8, 9,";
    GenerationService extra(extra_options);
    HttpServer server(options);
    assert(server.bind());
    server.attach(primary);
    server.attach_extra(extra);
    auto listener = std::async(std::launch::async, [&] { return server.listen(); });
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!server.is_running()) {
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(2ms);
    }
    auto post = [&](const std::string& path, const Json& body) {
        httplib::Client client("127.0.0.1", options.port);
        client.set_read_timeout(10, 0);
        return client.Post(path, body.dump(), "application/json");
    };
    // A load cannot inspect the namespace or publish a name while another
    // administrative operation holds the global gate, even on a different model.
    auto gate = HttpServerTestAccess::lock_adapters(server);
    const Json load{{"lora_name", "shared"}, {"lora_path", adapter}};
    auto first = std::async(std::launch::async, [&] { return post("/v1/load_lora_adapter", load); });
    auto second = std::async(std::launch::async, [&] { return post("/load_lora_adapter?model=extra", load); });
    assert(first.wait_for(200ms) == std::future_status::timeout);
    assert(second.wait_for(200ms) == std::future_status::timeout);
    assert(primary.lora_slot("shared") < 0 && extra.lora_slot("shared") < 0);
    gate.unlock();
    auto a = first.get();
    auto b = second.get();
    assert(a && b);
    if (a->status != 200 && b->status != 200) { std::cerr << a->body << '\n' << b->body << '\n'; }
    assert((a->status == 200 && b->status == 400) || (a->status == 400 && b->status == 200));
    const bool primary_owns = primary.lora_slot("shared") >= 0;
    assert(primary_owns != (extra.lora_slot("shared") >= 0));
    gate.lock();
    auto unload = std::async(std::launch::async, [&] {
        return post(primary_owns ? "/unload_lora_adapter" : "/v1/unload_lora_adapter?model=extra",
                    {{"lora_name", "shared"}});
    });
    assert(unload.wait_for(200ms) == std::future_status::timeout);
    gate.unlock();
    assert(unload.get()->status == 200);
    assert(primary.lora_slot("shared") < 0 && extra.lora_slot("shared") < 0);
    assert(post("/v1/load_lora_adapter", {{"lora_name", "primary-adapter"}, {"lora_path", adapter}})->status == 200);
    assert(post("/v1/load_lora_adapter?model=extra", {{"lora_name", "extra-adapter"}, {"lora_path", adapter}})->status == 200);
    httplib::Client catalog_client("127.0.0.1", options.port);
    auto listing = catalog_client.Get("/v1/models");
    assert(listing && listing->status == 200);
    const auto models = Json::parse(listing->body).at("data");
    assert(models.size() == 4);
    for (const auto& item : models) {
        const auto id = item.at("id").get<std::string>();
        auto result = catalog_client.Get("/v1/models/" + id);
        assert(result && result->status == 200);
        const auto model = Json::parse(result->body);
        assert(model.at("id") == id);
        if (id.ends_with("-adapter")) {
            const auto parent = id == "primary-adapter" ? "primary" : "extra";
            assert(model.at("parent") == parent && item.at("parent") == parent);
        } else { assert(!model.contains("parent") && !item.contains("parent")); }
    }
    auto missing = catalog_client.Get("/v1/models/missing");
    assert(missing && missing->status == 404);
    for (const auto& path : {"/v1/chat/completions", "/v1/completions"}) {
        Json invalid{{"model", "primary"}, {"tokens", Json::array({2147483647})}, {"max_tokens", 1}};
        if (std::string_view(path) == "/v1/completions") { invalid["prompt"] = "Hello"; }
        else { invalid["messages"] = Json::array({{{"role", "user"}, {"content", "Hello"}}}); }
        auto rejected = post(path, invalid);
        assert(rejected && rejected->status == 400);
        const auto error = Json::parse(rejected->body).at("error");
        assert(error.at("param") == "tokens");
        assert(error.at("message").get<std::string>().find("vocabulary") != std::string::npos);
    }
    ModelScheduler scheduler({{"primary", &primary}, {"extra", &extra}},
                             {{0, primary.resident_bytes()}, {1, extra.resident_bytes()}});
    server.attach_scheduler(scheduler);
    Json chat{{"model", "extra"}, {"messages", Json::array({{{"role", "user"}, {"content", "Hello"}}})},
              {"max_tokens", 1}};
    const auto parsed = parse_messages_request(chat, RequestLimits{});
    const int primary_count = primary.count_prompt_tokens(parsed);
    const int extra_count = extra.count_prompt_tokens(parsed);
    assert(primary_count != extra_count);
    for (const auto& id : {"primary", "claude-client-alias", "extra"}) {
        chat["model"] = id;
        const bool routed_extra = std::string_view(id) == "extra";
        if (routed_extra) { extra.sleep(); assert(extra.is_sleeping()); }
        auto count = post("/v1/messages/count_tokens", chat);
        assert(count && count->status == 200);
        const int actual = Json::parse(count->body).at("input_tokens");
        assert(actual == (routed_extra ? extra_count : primary_count));
        if (routed_extra) { assert(!extra.is_sleeping()); }
        auto generated = post("/v1/messages", chat);
        assert(generated && generated->status == 200);
        assert(Json::parse(generated->body).at("usage").at("input_tokens") == actual);
    }
    server.stop();
    assert(listener.wait_for(2s) == std::future_status::ready);
    assert(listener.get());
    std::cout << "Cross-model adapter administration and namespace uniqueness passed\n";
}
