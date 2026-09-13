#include "product/media_acquire/acquire.h"
#include "httplib.h"

#include <iostream>
#include <stdexcept>
#include <thread>

namespace media = sinfer::product::media_acquire;
namespace {
void require(bool value, const char* message) { if (!value) { throw std::runtime_error(message); } }
struct Server {
    httplib::Server http;
    std::thread worker;
    int port = -1;
    explicit Server(const char* host) {
        http.Get("/image", [](const auto&, auto& response) {
            response.set_content("image-test", "application/octet-stream");
        });
        http.Get("/redirect", [](const auto&, auto& response) { response.set_redirect("/image"); });
        port = http.bind_to_any_port(host);
        if (port >= 0) { worker = std::thread([this] { http.listen_after_bind(); }); }
    }
    ~Server() { http.stop(); if (worker.joinable()) { worker.join(); } }
};
}

int main() {
    try {
        for (const char* host : {"[::1]", "[::ffff:127.0.0.1]", "127.0.0.1", "localhost"}) {
            bool blocked = false;
            try { (void)media::acquire_bytes({.kind=media::SourceKind::Url,
                                             .value="http://" + std::string(host) + ":9/image"}); }
            catch (const std::invalid_argument& error) {
                blocked = std::string(error.what()).find("disallowed network addresses") != std::string::npos;
            }
            require(blocked, "private literal escaped network policy or was rejected by the resolver");
        }
        for (const bool ipv6 : {false, true}) {
            Server server(ipv6 ? "::1" : "127.0.0.1");
            if (server.port < 0 && ipv6) {
                std::cout << "SKIP: IPv6 loopback listener unavailable\n";
                continue;
            }
            require(server.port >= 0, "loopback listener unavailable");
            media::Policy policy;
            policy.allow_private_network = true;
            policy.timeout_ms = 3000;
            const std::string url = std::string("http://") + (ipv6 ? "[::1]" : "127.0.0.1") + ":" + std::to_string(server.port);
            for (const char* path : {"/image", "/redirect"}) {
                const auto bytes = media::acquire_bytes({.kind=media::SourceKind::Url, .value=url + path}, policy);
                require(std::string(bytes.begin(), bytes.end()) == "image-test", "literal media URL returned incorrect bytes");
            }
            if (!ipv6) {
                const auto bytes = media::acquire_bytes({.kind=media::SourceKind::Url,
                    .value="http://localhost:" + std::to_string(server.port) + "/image"}, policy);
                require(std::string(bytes.begin(), bytes.end()) == "image-test", "hostname DNS pinning failed");
            }
        }
        std::cout << "IPv4/IPv6 media fetch, redirects and private-address policy: OK\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
