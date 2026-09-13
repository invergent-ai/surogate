// Opt-in real-engine HTTP coverage with deterministic memory pressure.
// SUROGATE_MODEL_WAKE_TEST_ARTIFACT supplies a prepared text/chat checkpoint.
#include "serve/http_server.h"

#include <cassert>
#include <future>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

using namespace sinfer;
using namespace sinfer::serve;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

// A higher-priority busy resident makes the target wait without allocating a
// second checkpoint or relying on fluctuating free device memory.
struct BusyResident final : ScheduledModel {
    std::atomic<bool> asleep{false};
    std::size_t bytes;
    explicit BusyResident(std::size_t size) : bytes(size) {}
    void prepare_sleep_backup() override {}
    void sleep(bool = false) override { assert(false && "higher-priority busy resident was evicted"); }
    void wake_up() override { assert(false); }
    bool is_sleeping() const override { return asleep; }
    void shrink_kv() override { assert(false); }
    std::size_t active_requests() const override { return asleep ? 0 : 1; }
    std::size_t resumable_requests() const override { return 0; }
    std::size_t resident_bytes(int = -1) const override { return bytes; }
    std::vector<int> devices() const override { return {0}; }
};

int unused_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
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

template<class F> void eventually(F&& predicate) {
    const auto deadline = Clock::now() + 2s;
    while (!predicate() && Clock::now() < deadline) { std::this_thread::sleep_for(2ms); }
    assert(predicate());
}

int main() {
    const char* artifact = std::getenv("SUROGATE_MODEL_WAKE_TEST_ARTIFACT");
    if (!artifact) { return 77; }
    setenv("SUROGATE_MM_KEEPWARM_MS", "0", 1);
    setenv("SUROGATE_MM_PREEMPT_AFTER_MS", "10", 1);
    setenv("SUROGATE_MM_MIN_DWELL_MS", "0", 1);
    ServeOptions options;
    options.artifact_path = artifact;
    options.model_id_override = "target";
    options.port = unused_port();
    options.max_context = 512;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(512);
    options.max_concurrency = 1;
    options.max_pending_requests = 1;
    options.pending_timeout_ms = 300;
    options.enable_sleep_mode = true;
    options.log_stats_interval_ms = 0;
    GenerationService service(options);
    service.warmup();
    HttpServer server(options);
    assert(server.bind());
    server.attach(service);
    BusyResident blocker(service.resident_bytes());
    ModelScheduler scheduler({{"target", &service}, {"busy", &blocker, 2}}, {{0, blocker.bytes}});
    server.attach_scheduler(scheduler);
    service.sleep();
    auto listener = std::async(std::launch::async, [&] { return server.listen(); });
    eventually([&] { return server.is_running(); });
    httplib::Client client("127.0.0.1", options.port);
    client.set_read_timeout(3, 0);
    const Json message{{"role", "user"}, {"content", "Hello"}};
    const Json chat{{"model", "target"}, {"messages", Json::array({message})}, {"max_tokens", 1}};
    const Json completion{{"model", "target"}, {"prompt", "Hello"}, {"max_tokens", 1}};
    const Json response{{"model", "target"}, {"input", "Hello"}, {"max_output_tokens", 16}};
    for (const auto& [path, body] : std::vector<std::pair<std::string, Json>>{
        {"/v1/chat/completions", chat}, {"/v1/completions", completion},
        {"/v1/messages", chat}, {"/v1/responses", response},
        {"/tokenize", completion}, {"/v1/responses/input_tokens", {{"model", "target"}, {"input", "Hello"}}},
        {"/v1/messages/count_tokens", chat}}) {
        const auto start = Clock::now();
        auto result = client.Post(path, body.dump(), "application/json");
        assert(result);
        if (result->status != 503) { std::cerr << path << ": " << result->body << '\n'; }
        assert(result->status == 503);
        const auto error = Json::parse(result->body).at("error");
        if (!path.starts_with("/v1/messages")) { assert(error.at("code") == "request_queue_timeout"); }
        assert(error.at("message").get<std::string>().find("model wake") != std::string::npos);
        assert(Clock::now() - start < 1500ms);
        eventually([&] { return service.active_requests() == 0; });
        assert(service.is_sleeping());
        std::cout << path << " wake timeout passed\n";
    }
    // Pending wake occupies capacity. Disconnect releases it before the 300 ms
    // deadline, and must not trigger a later wake when memory becomes available.
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(options.port);
    assert(connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
    const auto body = chat.dump();
    const auto wire = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: " +
        std::to_string(body.size()) + "\r\n\r\n" + body;
    assert(send(fd, wire.data(), wire.size(), 0) == static_cast<ssize_t>(wire.size()));
    int second = socket(AF_INET, SOCK_STREAM, 0);
    assert(connect(second, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
    assert(send(second, wire.data(), wire.size(), 0) == static_cast<ssize_t>(wire.size()));
    eventually([&] { return service.active_requests() == 2; });
    assert(service.resumable_requests() == 0);
    auto overloaded = client.Post("/v1/completions", completion.dump(), "application/json");
    assert(overloaded && overloaded->status == 429);
    const auto disconnected = Clock::now();
    shutdown(fd, SHUT_RDWR);
    close(fd);
    shutdown(second, SHUT_RDWR);
    close(second);
    eventually([&] { return service.active_requests() == 0; });
    assert(Clock::now() - disconnected < 150ms);
    blocker.asleep = true;
    std::this_thread::sleep_for(600ms);
    assert(service.is_sleeping());
    auto recovered = client.Post("/v1/completions", completion.dump(), "application/json");
    if (recovered) { std::cout << "recovery: " << recovered->body << '\n'; }
    assert(recovered && recovered->status == 200);
    eventually([&] { return service.active_requests() == 0; });
    // Preparation after the wake gate still uses the same deadline.
    GenerationRequest request;
    request.raw_prompt = "Hello";
    request.max_tokens = 1;
    request.max_tokens_set = true;
    try {
        auto prepared = service.prepare(request, {}, [&](const PreparationControl& control) {
            assert(service.active_requests() == 1 && service.resumable_requests() == 0);
            std::this_thread::sleep_until(control.deadline + 5ms);
        });
        assert(false);
    } catch (const ApiException& error) { assert(error.error().code == "request_queue_timeout"); }
    assert(service.active_requests() == 0);
    // Shutdown cancels a pending wake through the HTTP cancellation predicate.
    service.sleep();
    blocker.asleep = false;
    auto pending = std::async(std::launch::async, [&] {
        httplib::Client shutdown_client("127.0.0.1", options.port);
        return shutdown_client.Post("/v1/responses", response.dump(), "application/json");
    });
    eventually([&] { return service.active_requests() == 1; });
    server.stop();
    assert(listener.wait_for(1s) == std::future_status::ready);
    listener.get();
    pending.get();
    assert(service.active_requests() == 0 && service.is_sleeping());
    std::cout << "HTTP wake deadlines, overload, disconnect, recovery and shutdown passed\n";
}
