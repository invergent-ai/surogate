// Malformed admin requests are rejected before a model is needed.
#include "serve/http_server.h"

#include <cassert>
#include <future>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace sinfer::serve;
using namespace std::chrono_literals;
using Json = nlohmann::json;

namespace sinfer::serve {
struct HttpServerTestAccess {
    static int bind(HttpServer& server) { return server.server_.bind_to_any_port("127.0.0.1"); }
    static bool listen(HttpServer& server) { return server.server_.listen_after_bind(); }
};
}

int main() {
    ServeOptions options;
    options.max_concurrency = 1;
    options.max_pending_requests = 1;
    options.log_stats_interval_ms = 0;
    HttpServer server(options);
    const int port = HttpServerTestAccess::bind(server);
    assert(port > 0);
    auto listener = std::async(std::launch::async, [&] { return HttpServerTestAccess::listen(server); });
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!server.is_running()) {
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(2ms);
    }
    httplib::Client client("127.0.0.1", port);
    client.set_read_timeout(3, 0);
    for (const auto* route : {"/load_lora_adapter", "/v1/load_lora_adapter",
                              "/unload_lora_adapter", "/v1/unload_lora_adapter"}) {
        const auto reject = [&](const std::string& body) {
            auto response = client.Post(route, body, "application/json");
            assert(response);
            if (response->status != 400) { std::cerr << route << ": " << body << " => " << response->body << '\n'; }
            assert(response->status == 400);
        };
        reject("{");
        for (const auto& value : {Json(nullptr), Json(1), Json(true), Json("value"), Json::array(), Json::array({"a"})}) {
            reject(value.dump());
        }
        reject(Json::object().dump());
        reject(Json{{"lora_name", ""}, {"lora_path", ""}}.dump());
        for (const auto& value : {Json(nullptr), Json(1), Json(true), Json::array(), Json::object()}) {
            reject(Json{{"lora_name", value}, {"lora_path", "/unused"}}.dump());
            if (std::string_view(route).find("unload") == std::string_view::npos) {
                reject(Json{{"lora_name", "adapter"}, {"lora_path", value}}.dump());
            }
        }
    }
    server.stop();
    assert(listener.get());
    std::cout << "LoRA admin input errors return HTTP 400 on all aliases\n";
}
