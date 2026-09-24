// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// The WebSocket framing of serve/websocket.h (SUROGATE-CHANGES #9), against an in-memory stream:
// no socket, no model. Frames are read one byte at a time, as a slow network delivers them.
#include "serve/websocket.h"

#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

namespace ws = sinfer::serve::websocket;

namespace {

class Memory final : public httplib::Stream {
public:
    explicit Memory(std::string input, size_t step = 1) : input_(std::move(input)), step_(step) {}
    bool is_readable() const override { return offset_ < input_.size(); }
    bool is_writable() const override { return true; }
    ssize_t read(char* ptr, size_t size) override {
        const size_t n = std::min({size, step_, input_.size() - offset_});
        if (n == 0) return 0;
        std::memcpy(ptr, input_.data() + offset_, n);
        offset_ += n;
        return static_cast<ssize_t>(n);
    }
    ssize_t write(const char* ptr, size_t size) override {
        const size_t n = std::min<size_t>(size, 7); // short writes, as a full socket gives
        output.append(ptr, n);
        return static_cast<ssize_t>(n);
    }
    void get_remote_ip_and_port(std::string&, int&) const override {}
    void get_local_ip_and_port(std::string&, int&) const override {}
    socket_t socket() const override { return INVALID_SOCKET; }
    std::string output;

private:
    std::string input_;
    size_t offset_ = 0;
    size_t step_;
};

/// One end of a socketpair, for close(), which works on the socket itself.
class Socket final : public httplib::Stream {
public:
    explicit Socket(int fd) : fd_(fd) {}
    bool is_readable() const override { return true; }
    bool is_writable() const override { return true; }
    ssize_t read(char* ptr, size_t size) override { return ::read(fd_, ptr, size); }
    ssize_t write(const char* ptr, size_t size) override { return ::write(fd_, ptr, size); }
    void get_remote_ip_and_port(std::string&, int&) const override {}
    void get_local_ip_and_port(std::string&, int&) const override {}
    socket_t socket() const override { return fd_; }

private:
    int fd_;
};

/// A client frame: masked, with `fin` and `opcode` in the first byte (plus `rsv` bits).
std::string frame(uint8_t opcode, std::string_view payload, bool fin = true, bool mask = true,
                  uint8_t rsv = 0) {
    std::string out;
    out.push_back(static_cast<char>((fin ? 0x80 : 0) | rsv | opcode));
    const uint8_t m = mask ? 0x80 : 0;
    if (payload.size() < 126) {
        out.push_back(static_cast<char>(m | payload.size()));
    } else if (payload.size() <= 0xFFFF) {
        out.push_back(static_cast<char>(m | 126));
        out.push_back(static_cast<char>(payload.size() >> 8));
        out.push_back(static_cast<char>(payload.size() & 0xFF));
    } else {
        out.push_back(static_cast<char>(m | 127));
        for (int i = 7; i >= 0; --i) out.push_back(static_cast<char>((uint64_t(payload.size()) >> (8 * i)) & 0xFF));
    }
    const char key[4] = {0x12, 0x34, 0x56, 0x78};
    if (mask) out.append(key, 4);
    for (size_t i = 0; i < payload.size(); ++i) out.push_back(mask ? payload[i] ^ key[i % 4] : payload[i]);
    return out;
}

uint16_t failure_code(const std::string& input, size_t max_size = 1 << 20) {
    Memory stream(input);
    ws::Failure failure;
    assert(!ws::receive(stream, max_size, &failure));
    return failure.code;
}

} // namespace

int main() {
    // The handshake key from RFC 6455 section 1.3.
    assert(ws::accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");

    // A text message, a binary one of every length encoding, byte by byte.
    for (size_t size : {size_t(0), size_t(125), size_t(126), size_t(65535), size_t(65536), size_t(200000)}) {
        const std::string payload(size, 'x');
        Memory stream(frame(0x2, payload));
        const auto message = ws::receive(stream, 1 << 20);
        assert(message && message->opcode == ws::Opcode::Binary && message->payload == payload);
    }
    // Fragments are joined, and a ping between them is answered with a pong carrying its payload.
    {
        Memory stream(frame(0x1, "Bună ", false) + frame(0x9, "hi") + frame(0x0, "ziua", true));
        const auto message = ws::receive(stream, 1024);
        assert(message && message->opcode == ws::Opcode::Text && message->payload == "Bună ziua");
        assert(stream.output == std::string("\x8A\x02hi", 4));
    }
    // A close frame: its code; no code is 1005.
    {
        Memory stream(frame(0x8, std::string("\x03\xE8", 2) + "bye"));
        const auto message = ws::receive(stream, 1024);
        assert(message && message->opcode == ws::Opcode::Close && message->close_code == 1000);
        Memory empty(frame(0x8, ""));
        assert(ws::receive(empty, 1024)->close_code == 1005);
    }
    // Protocol violations name the close code to answer with.
    assert(failure_code(frame(0x2, "x", true, false)) == 1002);              // unmasked
    assert(failure_code(frame(0x2, "x", true, true, 0x40)) == 1002);         // reserved bit
    assert(failure_code(frame(0x3, "x")) == 1002);                           // reserved opcode
    assert(failure_code(frame(0x0, "x")) == 1002);                           // continuation first
    assert(failure_code(frame(0x1, "a", false) + frame(0x1, "b")) == 1002);  // new message mid-way
    assert(failure_code(frame(0x9, "p", false)) == 1002);                    // fragmented control
    assert(failure_code(frame(0x9, std::string(126, 'p'))) == 1002);         // control too long
    assert(failure_code(frame(0x8, "\x03")) == 1002);                        // one-byte close
    assert(failure_code(frame(0x8, std::string("\x03\xE7", 2))) == 1002);    // close code 999
    assert(failure_code(frame(0x8, std::string("\x03\xED", 2))) == 1002);    // close code 1005
    assert(failure_code(frame(0x2, std::string(1025, 'x')), 1024) == 1009);  // too big
    assert(failure_code(frame(0x2, std::string(600, 'x'), false) + frame(0x0, std::string(600, 'x')), 1024) ==
           1009);                                                            // too big once joined
    assert(failure_code(frame(0x1, "\xC3\x28")) == 1007);                    // invalid UTF-8 text
    assert(failure_code(frame(0x1, "\xED\xA0\x80")) == 1007);                // a surrogate
    assert(failure_code(frame(0x1, "\xC0\xAF")) == 1007);                    // overlong
    assert(failure_code(frame(0x8, std::string("\x03\xE8\xFF", 3))) == 1007); // close reason
    // The 4-byte edges: U+10FFFF passes; above it, F5-FF lead bytes, overlong 3- and 4-byte forms,
    // a truncated sequence and a lone continuation byte do not.
    {
        Memory top(frame(0x1, "\xF4\x8F\xBF\xBF"));
        assert(ws::receive(top, 1024)->payload == "\xF4\x8F\xBF\xBF");
    }
    for (const char* bad : {"\xF4\x90\x80\x80", "\xF5\x80\x80\x80", "\xF8\x88\x80\x80\x80", "\xFF",
                            "\xE0\x80\xAF", "\xF0\x80\x80\xAF", "\xE2\x82", "\x80"}) {
        assert(failure_code(frame(0x1, bad)) == 1007);
    }
    // Registered close codes a peer may send (1012-1014) are accepted; 1015 is never sent.
    for (const uint16_t code : {1012, 1013, 1014}) {
        const std::string payload{char(code >> 8), char(code & 0xFF)};
        Memory stream(frame(0x8, payload));
        assert(ws::receive(stream, 1024)->close_code == code);
    }
    assert(failure_code(frame(0x8, std::string("\x03\xF7", 2))) == 1002);    // close code 1015
    assert(failure_code(std::string("\x82", 1)) == 0);                       // connection broke
    // UTF-8 split across fragments is judged on the whole message.
    {
        Memory stream(frame(0x1, "\xC4", false) + frame(0x0, "\x83"));
        assert(ws::receive(stream, 1024)->payload == "ă");
    }
    // Server frames: unmasked, every length encoding, written whole despite short writes.
    for (size_t size : {size_t(3), size_t(300), size_t(70000)}) {
        Memory stream("");
        assert(ws::send(stream, ws::Opcode::Binary, std::string(size, 'y')));
        const size_t header = size < 126 ? 2 : size <= 0xFFFF ? 4 : 10;
        assert(stream.output.size() == header + size && uint8_t(stream.output[0]) == 0x82);
        assert((uint8_t(stream.output[1]) & 0x80) == 0);
    }
    // A ping or a close in the middle of a message near the size limit does not count toward it.
    {
        Memory stream(frame(0x2, std::string(1000, 'x'), false) + frame(0x9, std::string(100, 'p')) +
                      frame(0x0, std::string(24, 'x')));
        const auto message = ws::receive(stream, 1024);
        assert(message && message->payload.size() == 1024);
        Memory closing(frame(0x2, std::string(1000, 'x'), false) + frame(0x8, std::string("\x03\xE8", 2)));
        assert(ws::receive(closing, 1024)->opcode == ws::Opcode::Close);
    }
    // Pongs are ignored.
    {
        Memory stream(frame(0xA, "unsolicited") + frame(0x1, "ok"));
        assert(ws::receive(stream, 1024)->payload == "ok" && stream.output.empty());
    }
    // A connection that breaks inside a frame's length, mask or payload is a broken connection.
    assert(failure_code(std::string("\x82\xFE\x01", 3)) == 0);
    assert(failure_code(frame(0x2, "abc").substr(0, 4)) == 0);
    assert(failure_code(frame(0x2, "abcdef").substr(0, 8)) == 0);
    // A 64-bit length with the top bit set is too big, not a wrap.
    assert(failure_code(std::string("\x82\xFF\x80\x00\x00\x00\x00\x00\x00\x01", 10) + "mask") == 1009);
    // The handshake: 426 with the version this server speaks, 400 for a missing key.
    {
        httplib::Request request;
        request.method = "GET";
        request.set_header("Upgrade", "WebSocket");
        request.set_header("Connection", "keep-alive, Upgrade");
        request.set_header("Sec-WebSocket-Version", "13");
        request.set_header("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==");
        assert(ws::is_upgrade(request) && !ws::handshake_error(request));
        httplib::Request old_version = request;
        old_version.headers.erase("Sec-WebSocket-Version");
        old_version.set_header("Sec-WebSocket-Version", "8");
        assert(ws::handshake_error(old_version)->status == 426);
        httplib::Response refused;
        ws::refuse_handshake(old_version, refused);
        assert(refused.get_header_value("Sec-WebSocket-Version") == "13");
        httplib::Request no_key = request;
        no_key.headers.erase("Sec-WebSocket-Key");
        assert(ws::handshake_error(no_key)->status == 400);
        httplib::Request plain;
        plain.method = "GET";
        assert(ws::handshake_error(plain)->status == 426);
        httplib::Request posted = request;
        posted.method = "POST";
        assert(ws::handshake_error(posted)->status == 400);
        httplib::Response upgrade;
        ws::refuse_handshake(plain, upgrade);
        assert(upgrade.get_header_value("Upgrade") == "websocket");
    }
    // close() closes the sending side first, so a peer that waits for the server (as RFC 6455
    // asks) is not kept waiting: it reads the close frame, then the end of the stream.
    {
        int fds[2];
        assert(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
        std::string seen;
        std::thread peer([&] {
            char buffer[256];
            for (ssize_t n; (n = ::read(fds[1], buffer, sizeof(buffer))) > 0;) seen.append(buffer, size_t(n));
            ::close(fds[1]);
        });
        Socket server(fds[0]);
        const auto started = std::chrono::steady_clock::now();
        ws::close(server, 1000, "bye");
        const auto took = std::chrono::steady_clock::now() - started;
        peer.join();
        ::close(fds[0]);
        assert(took < std::chrono::milliseconds(300));
        assert(seen == std::string("\x88\x05\x03\xE8" "bye", 7));
    }
    // A close reason that is not UTF-8 is left out rather than sent broken.
    {
        Memory stream("");
        ws::close(stream, 1011, "\xFF\xFE");
        assert(stream.output == std::string("\x88\x02\x03\xF3", 4));
    }
    // A close reason is cut to fit a control frame, at a character boundary.
    {
        Memory stream("");
        std::string reason;
        for (int i = 0; i < 100; ++i) reason += "ș"; // two bytes each
        ws::close(stream, 1011, reason);
        const size_t payload = uint8_t(stream.output[1]);
        assert(payload <= 125 && payload % 2 == 0 && ws::valid_utf8(stream.output.substr(4)));
    }
    std::puts("websocket framing: ok");
    return 0;
}
