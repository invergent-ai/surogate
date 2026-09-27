#include "serve/http_server.h"
#include "serve/openrouter_catalog.h"

#include <cassert>
#include <future>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <nlohmann/json.hpp>

using namespace sinfer::serve;
using namespace std::chrono_literals;

namespace sinfer::serve {
struct HttpServerTestAccess {
    static int bind(HttpServer& s) { return s.server_.bind_to_any_port("127.0.0.1"); }
    static bool listen(HttpServer& s) { return s.server_.listen_after_bind(); }
    static void add_stream(HttpServer& s, std::shared_future<void> release,
                           std::promise<void>& started) {
        s.server_.Post("/test/stream", [release, &started](const auto&, auto& res) {
            // Holding another resource must not release the admission permit.
            res.hold_resource(std::make_shared<int>(1));
            res.set_chunked_content_provider("text/plain", [release, &started](size_t, auto& sink) {
                started.set_value();
                release.wait();
                sink.write("done", 4);
                sink.done();
                return true;
            });
        });
    }
};
}

int main() {
    const auto now = AdmissionLimit::Clock::time_point{};
    AdmissionLimit bucket(2, 2, 1, now);
    auto first = bucket.acquire(now);
    assert(first.accepted);
    assert(!bucket.acquire(now).accepted);
    first.permit.reset();
    auto second = bucket.acquire(now);
    assert(second.accepted); // a concurrency rejection did not spend a token
    second.permit.reset();
    assert(!bucket.acquire(now + 499ms).accepted);
    assert(bucket.acquire(now + 500ms).accepted);
    auto rejected = bucket.acquire(now + 500ms);
    assert(!rejected.accepted && rejected.retry_after == 1);
    assert(bucket.acquire(now + 1h).accepted);
    assert(bucket.acquire(now + 1h).accepted);
    assert(!bucket.acquire(now + 1h).accepted); // refill capped at burst

    // Contending threads cannot over-admit; all permits survive the acquisition phase.
    AdmissionLimit concurrent(0, 1, 4);
    std::vector<std::future<AdmissionLimit::Result>> futures;
    for (int i = 0; i < 32; ++i) {
        futures.push_back(std::async(std::launch::async, [&] { return concurrent.acquire(); }));
    }
    std::vector<AdmissionLimit::Result> held;
    int accepted = 0;
    for (auto& f : futures) { held.push_back(f.get()); accepted += held.back().accepted; }
    assert(accepted == 4);
    held.clear();
    assert(concurrent.acquire().accepted);

    const auto run = [](ServeOptions options, auto exercise) {
        options.log_stats_interval_ms = 0;
        HttpServer server(options);
        std::promise<void> release, started;
        HttpServerTestAccess::add_stream(server, release.get_future().share(), started);
        const int port = HttpServerTestAccess::bind(server);
        assert(port > 0);
        auto listener = std::async(std::launch::async, [&] { return HttpServerTestAccess::listen(server); });
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while (!server.is_running()) {
            assert(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(1ms);
        }
        exercise(port, release, started);
        server.stop();
        assert(listener.get());
    };
    const auto catalog_path = std::filesystem::temp_directory_path() /
                              ("sinfer-catalog-" + std::to_string(::getpid()) + ".json");
    const auto catalog = nlohmann::json::parse(R"json({"data":[{
        "schema_version":"2.4", "id":"test-model", "name":"Test model", "is_ready":false,
        "input_modalities":[{"type":"text"}],
        "output_modalities":[{"type":"decisions","supported_parameters":{}}]
    }]})json");
    std::ofstream(catalog_path) << catalog.dump();
    assert(load_openrouter_catalog(catalog_path.string()) == catalog);
    validate_openrouter_catalog_models(catalog, {"test-model"});
    bool rejected_model = false;
    try { validate_openrouter_catalog_models(catalog, {"other"}); }
    catch (const std::invalid_argument&) { rejected_model = true; }
    assert(rejected_model);

    ServeOptions provider;
    provider.api_key = "test-only";
    provider.openrouter_models_file = catalog_path.string();
    run(provider, [&](int port, auto&, auto&) {
        httplib::Client client("127.0.0.1", port);
        assert(client.Get("/openrouter/v1/models")->status == 401);
        const httplib::Headers auth{{"Authorization", "Bearer test-only"}};
        const auto models = client.Get("/openrouter/v1/models", auth);
        assert(models && models->status == 200 && nlohmann::json::parse(models->body) == catalog);
        for (const auto* path : {"/openrouter/v1/decisions", "/openrouter/api/alpha/decisions",
                                 "/v1/systemone", "/api/v1/systemone"}) {
            assert(client.Post(path, "{}", "application/json")->status == 401);
            assert(client.Post(path, auth, "{}", "application/json")->status == 400);
        }
    });
    provider.rate_limit_rps = 0.001;
    provider.rate_limit_burst = 1;
    run(provider, [](int port, auto&, auto&) {
        httplib::Client client("127.0.0.1", port);
        const httplib::Headers auth{{"Authorization", "Bearer test-only"}};
        assert(client.Post("/v1/decisions", auth, "{}", "application/json")->status == 400);
        for (const auto* path : {"/openrouter/v1/decisions", "/openrouter/api/alpha/decisions",
                                 "/v1/systemone", "/api/v1/systemone"}) {
            const auto limited = client.Post(path, auth, "{}", "application/json");
            assert(limited && limited->status == 429 && limited->has_header("Retry-After"));
        }
        assert(client.Get("/openrouter/v1/models", auth)->status == 200);
    });
    // Bad catalogs fail before a server starts; errors must not disclose file contents.
    for (const auto& bad : {std::string("secret-looking-invalid-json"), std::string("{\"data\":[]}"),
                            std::string("{\"data\":[{\"id\":\"model\"}]}"),
                            std::string((1U << 20) + 1, 'x')}) {
        std::ofstream(catalog_path, std::ios::trunc) << bad;
        bool rejected_catalog = false;
        try { (void)load_openrouter_catalog(catalog_path.string()); }
        catch (const std::invalid_argument& e) {
            rejected_catalog = true;
            assert(std::string(e.what()).find("secret-looking") == std::string::npos);
        }
        assert(rejected_catalog);
    }
    std::filesystem::remove(catalog_path);
    ServeOptions rate;
    rate.api_key = "test-only";
    rate.enable_cors = true;
    rate.rate_limit_rps = 0.001;
    rate.rate_limit_burst = 1;
    run(rate, [](int port, auto&, auto&) {
        httplib::Client client("127.0.0.1", port);
        auto unauth = client.Post("/v1/decisions", "{", "application/json");
        assert(unauth && unauth->status == 401);
        httplib::Headers auth{{"Authorization", "Bearer test-only"}};
        auto bad = client.Post("/v1/decisions", auth, "{", "application/json");
        assert(bad && bad->status == 400);
        for (const auto* path : {"/api/alpha/decisions", "/api/v1/decisions", "/v1/messages", "/v1/responses"}) {
            auto r = client.Post(path, auth, "{", "application/json");
            assert(r && r->status == 429 && r->has_header("Retry-After"));
            assert(nlohmann::json::parse(r->body).at("error").at("type") == "rate_limit_error");
        }
        assert(client.Get("/health")->status == 200);
        assert(client.Options("/v1/decisions")->status == 204);
        assert(client.Get("/openrouter/v1/models", auth)->status == 404);
    });
    ServeOptions inflight;
    inflight.max_inflight_requests = 1;
    run(inflight, [](int port, auto& release, auto& started) {
        auto streaming = std::async(std::launch::async, [port] {
            httplib::Client c("127.0.0.1", port);
            return c.Post("/test/stream", "", "application/json");
        });
        assert(started.get_future().wait_for(3s) == std::future_status::ready);
        httplib::Client client("127.0.0.1", port);
        auto busy = client.Post("/v1/decisions", "{", "application/json");
        assert(busy && busy->status == 429);
        assert(client.Get("/health")->status == 200);
        release.set_value();
        assert(streaming.get()->status == 200);
        for (int i = 0; i < 2; ++i) {
            auto malformed = client.Post("/v1/decisions", "{", "application/json");
            assert(malformed && malformed->status == 400);
        }
    });
}
