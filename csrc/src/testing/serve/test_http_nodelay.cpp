// TCP_NODELAY on the engine's HTTP servers. Model-free and GPU-free.
//
// cpp-httplib writes a response as two sends (head, then body). With Nagle's algorithm on, the
// body waits for the client's delayed ACK of the head, so every request after the first on a
// keep-alive connection took about 40 ms longer (SUROGATE-CHANGES #12: 64 ms on a new
// connection, 104 ms on each reuse). These checks read the option straight off the sockets and
// time sequential requests over one pooled connection, against a control server left as it was.
#include "serve/audio_http.h"
#include "serve/http_server.h"
#include "serve/http_socket.h"

#include <httplib.h>

#include <algorithm>
#include <arpa/inet.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <future>
#include <iostream>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace sinfer::serve;
using namespace std::chrono_literals;

namespace {

int unused_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    assert(bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
    socklen_t size = sizeof(address);
    assert(getsockname(fd, reinterpret_cast<sockaddr*>(&address), &size) == 0);
    close(fd);
    return ntohs(address.sin_port);
}

/// TCP_NODELAY of this process's sockets whose local port is `port`: the listening socket, or
/// the connections a server accepted on it (the client side of a connection has an ephemeral
/// local port, so it never matches).
std::vector<int> nodelay_on_port(int port, bool listening) {
    std::vector<int> found;
    DIR* dir = opendir("/proc/self/fd");
    assert(dir != nullptr);
    while (const dirent* entry = readdir(dir)) {
        char* end    = nullptr;
        const long n = std::strtol(entry->d_name, &end, 10);
        if (end == entry->d_name || *end != '\0') { continue; }
        const int fd = static_cast<int>(n);
        sockaddr_storage local{};
        socklen_t size = sizeof(local);
        if (getsockname(fd, reinterpret_cast<sockaddr*>(&local), &size) != 0) { continue; }
        if (local.ss_family != AF_INET) { continue; }
        if (ntohs(reinterpret_cast<const sockaddr_in&>(local).sin_port) != port) { continue; }
        int accepting      = 0;
        socklen_t int_size = sizeof(accepting);
        if (getsockopt(fd, SOL_SOCKET, SO_ACCEPTCONN, &accepting, &int_size) != 0) { continue; }
        if ((accepting != 0) != listening) { continue; }
        int nodelay = -1;
        int_size    = sizeof(nodelay);
        assert(getsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, &int_size) == 0);
        found.push_back(nodelay != 0 ? 1 : 0);
    }
    closedir(dir);
    return found;
}

struct Timing {
    double first_ms = 0.0;
    double later_median_ms = 0.0;
    std::vector<int> accepted_nodelay;
};

/// `requests` calls over one keep-alive connection to a plain server that answers with a small
/// body, like a decision. `configure` sets the server up before it binds. A GET, or a POST
/// with a 1 KiB body like a decisions request; cpp-httplib's own client writes a request's head
/// and body separately too, so a POST is only free of the stall when the client also turns
/// Nagle off (most HTTP clients send small requests in one write or do so).
Timing time_keep_alive(void (*configure)(httplib::Server&), bool post = false, bool client_nodelay = false,
                       int requests = 25) {
    httplib::Server server;
    configure(server);
    const std::string body(200, 'x');
    server.Get("/ping", [&](const httplib::Request&, httplib::Response& r) {
        r.set_content(body, "application/json");
    });
    server.Post("/ping", [&](const httplib::Request& q, httplib::Response& r) {
        assert(q.body.size() == 1024);
        r.set_content(body, "application/json");
    });
    // The kernel picks the port, so a parallel test cannot land on the same one.
    const int port = server.bind_to_any_port("127.0.0.1");
    assert(port > 0);
    auto listening = std::async(std::launch::async, [&] { return server.listen_after_bind(); });
    server.wait_until_ready();

    Timing timing;
    httplib::Client client("127.0.0.1", port);
    client.set_keep_alive(true);
    client.set_tcp_nodelay(client_nodelay);
    const std::string payload(1024, 'q');
    std::vector<double> later;
    for (int i = 0; i < requests; ++i) {
        const auto began = std::chrono::steady_clock::now();
        const auto result = post ? client.Post("/ping", payload, "application/json") : client.Get("/ping");
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();
        assert(result && result->status == 200 && result->body == body);
        if (i == 0) {
            timing.first_ms = ms;
            // The connection now exists on the server side: read its option.
            timing.accepted_nodelay = nodelay_on_port(port, false);
        } else {
            later.push_back(ms);
        }
    }
    std::sort(later.begin(), later.end());
    timing.later_median_ms = later[later.size() / 2];
    client.stop();
    server.stop();
    assert(listening.get());
    return timing;
}

} // namespace

int main() {
    // 1. The generation server (HttpServer) sets the option before it binds, so its listening
    //    socket carries it. listen() needs a model, bind() does not.
    {
        ServeOptions options;
        options.host                  = "127.0.0.1";
        options.port                  = unused_port();
        options.log_stats_interval_ms = 0;
        HttpServer server(options);
        assert(server.bind());
        const auto listener = nodelay_on_port(options.port, true);
        assert(listener.size() == 1 && listener.front() == 1);
    }
    // The check itself tells the two apart: a server left alone listens without the option.
    {
        httplib::Server plain;
        const int port = plain.bind_to_any_port("127.0.0.1");
        assert(port > 0);
        const auto listener = nodelay_on_port(port, true);
        assert(listener.size() == 1 && listener.front() == 0);
    }
    // 2. STT and TTS set their servers up through audio::configure, before they bind.
    {
        httplib::Server speech;
        audio::configure(speech, "", 1 << 20);
        const int port = speech.bind_to_any_port("127.0.0.1");
        assert(port > 0);
        const auto listener = nodelay_on_port(port, true);
        assert(listener.size() == 1 && listener.front() == 1);
    }

    // 3. What the option is for: the connections a server accepts carry it, and requests after
    //    the first on a keep-alive connection are no slower than the first. The control, the
    //    server as it was, pays the delayed ACK on every reuse.
    const auto fix     = [](httplib::Server& s) { disable_nagle(s); };
    const auto unfixed = [](httplib::Server&) {};
    const Timing fixed  = time_keep_alive(fix);
    const Timing speech = time_keep_alive([](httplib::Server& s) { audio::configure(s, "", 1 << 20); });
    const Timing posted = time_keep_alive(fix, true, true);
    const Timing before = time_keep_alive(unfixed);
    const Timing before_post = time_keep_alive(unfixed, true, true);
    std::printf("keep-alive, median of requests after the first: GET disable_nagle %.2f ms, GET "
                "audio::configure %.2f ms, POST disable_nagle %.2f ms; unchanged server: GET %.2f ms, "
                "POST %.2f ms\n",
                fixed.later_median_ms, speech.later_median_ms, posted.later_median_ms,
                before.later_median_ms, before_post.later_median_ms);
    // What the servers do to their sockets does not depend on the host: always checked.
    for (const Timing* t : {&fixed, &speech, &posted}) { assert(t->accepted_nodelay == std::vector<int>{1}); }
    for (const Timing* t : {&before, &before_post}) { assert(t->accepted_nodelay == std::vector<int>{0}); }
    // The timing checks need a host whose kernel delays ACKs the usual ~40 ms, which the unchanged
    // server shows; a host tuned otherwise (quickack routes, a short rto_min) cannot show the
    // difference, so there they are reported and skipped rather than failed.
    if (before.later_median_ms > 25.0 && before_post.later_median_ms > 25.0) {
        // Loopback without Nagle answers in well under a millisecond; 10 ms leaves room for a busy
        // host and is still a quarter of the delay being removed.
        assert(fixed.later_median_ms < 10.0);
        assert(speech.later_median_ms < 10.0);
        assert(posted.later_median_ms < 10.0);
    } else {
        std::printf("this host does not delay ACKs; the timing checks are skipped\n");
    }

    std::cout << "TCP_NODELAY checks passed\n";
    return 0;
}
