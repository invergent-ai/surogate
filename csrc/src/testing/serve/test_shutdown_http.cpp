// Stop a live HTTP server during decode while the client remains connected.
#include "serve/http_server.h"

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

template<class F> void eventually(F&& predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!predicate()) {
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(2ms);
    }
}

int main() {
    const char* artifact = std::getenv("SUROGATE_SHUTDOWN_TEST_ARTIFACT");
    if (!artifact) { return 77; }
    ServeOptions options;
    options.artifact_path = artifact;
    options.model_id_override = "target";
    options.max_context = 8192;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(8192);
    options.max_concurrency = 1;
    options.use_cuda_graph = false;
    options.log_stats_interval_ms = 0;
    GenerationService service(options);
    for (const auto& [path, stream] : std::vector<std::pair<std::string, bool>>{
        {"/v1/chat/completions", true}, {"/v1/completions", true}, {"/v1/messages", true},
        {"/v1/responses", true}, {"/v1/responses", false}}) {
        options.port = unused_port();
        HttpServer server(options);
        assert(server.bind());
        server.attach(service);
        auto listener = std::async(std::launch::async, [&] { return server.listen(); });
        eventually([&] { return server.is_running(); });
        Json body{{"model", "target"}, {"stream", stream}, {"temperature", 0}};
        const std::string prompt = "Count the integers from one to ten thousand, one at a time.";
        if (path == "/v1/responses") { body["input"] = prompt; body["max_output_tokens"] = 4096; }
        else {
            body["ignore_eos"] = true;
            body["max_tokens"] = 4096;
            if (path == "/v1/completions") { body["prompt"] = prompt; }
            else { body["messages"] = Json::array({{{"role", "user"}, {"content", prompt}}}); }
        }
        std::atomic<std::size_t> received{0};
        auto client = std::async(std::launch::async, [&] {
            httplib::Client connection("127.0.0.1", options.port);
            connection.set_read_timeout(20, 0);
            httplib::Request request;
            request.method = "POST";
            request.path = path;
            request.body = body.dump();
            request.set_header("Content-Type", "application/json");
            request.response_handler = [](const httplib::Response& response) {
                assert(response.status == 200);
                return true;
            };
            request.content_receiver = [&](const char*, std::size_t size, std::uint64_t, std::uint64_t) {
                received += size;
                return true;
            };
            return connection.send(request);
        });
        eventually([&] {
            return service.runtime_stats().decode_ready_requests != 0 && (!stream || received != 0);
        });
        server.stop();
        assert(listener.wait_for(2s) == std::future_status::ready);
        assert(listener.get());
        client.get();
        eventually([&] { return service.active_requests() == 0; });
        std::cout << path << " stream=" << stream << " cancelled promptly at shutdown\n";
    }
}
