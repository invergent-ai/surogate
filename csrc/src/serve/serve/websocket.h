// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
// A minimal server side of RFC 6455 WebSockets on top of httplib's switch_protocols (a surogate
// vendor patch): the opening handshake, and reading and writing messages on the upgraded stream.
// Header-only, for the speech servers' separate builds. No extensions (permessage-deflate is
// declined by not answering it), no subprotocols.

#include <httplib.h>

#include <poll.h>
#include <sys/socket.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace sinfer::serve::websocket {

namespace detail {

inline std::array<uint8_t, 20> sha1(std::string_view data) {
    uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    std::string message(data);
    const uint64_t bits = static_cast<uint64_t>(data.size()) * 8;
    message.push_back(static_cast<char>(0x80));
    while (message.size() % 64 != 56) message.push_back('\0');
    for (int i = 7; i >= 0; --i) message.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
    const auto rotl = [](uint32_t x, int n) { return (x << n) | (x >> (32 - n)); };
    for (size_t chunk = 0; chunk < message.size(); chunk += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(uint8_t(message[chunk + 4 * i])) << 24) |
                   (uint32_t(uint8_t(message[chunk + 4 * i + 1])) << 16) |
                   (uint32_t(uint8_t(message[chunk + 4 * i + 2])) << 8) |
                   uint32_t(uint8_t(message[chunk + 4 * i + 3]));
        }
        for (int i = 16; i < 80; ++i) w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999u; }
            else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1u; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6u; }
            const uint32_t t = rotl(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rotl(b, 30); b = a; a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    std::array<uint8_t, 20> digest{};
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 4; ++j) digest[4 * i + j] = static_cast<uint8_t>((h[i] >> (24 - 8 * j)) & 0xFF);
    return digest;
}

inline std::string base64(const uint8_t* bytes, size_t size) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    size_t i = 0;
    for (; i + 2 < size; i += 3) {
        const uint32_t v = (uint32_t(bytes[i]) << 16) | (uint32_t(bytes[i + 1]) << 8) | bytes[i + 2];
        out += {alphabet[v >> 18], alphabet[(v >> 12) & 63], alphabet[(v >> 6) & 63], alphabet[v & 63]};
    }
    if (i + 1 == size) {
        const uint32_t v = uint32_t(bytes[i]) << 16;
        out += {alphabet[v >> 18], alphabet[(v >> 12) & 63], '=', '='};
    } else if (i + 2 == size) {
        const uint32_t v = (uint32_t(bytes[i]) << 16) | (uint32_t(bytes[i + 1]) << 8);
        out += {alphabet[v >> 18], alphabet[(v >> 12) & 63], alphabet[(v >> 6) & 63], '='};
    }
    return out;
}

inline bool read_exact(httplib::Stream& stream, char* out, size_t size) {
    while (size > 0) {
        const auto n = stream.read(out, size);
        if (n <= 0) return false;
        out += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

inline bool write_all(httplib::Stream& stream, std::string_view data) {
    while (!data.empty()) {
        const auto n = stream.write(data.data(), data.size());
        if (n <= 0) return false;
        data.remove_prefix(static_cast<size_t>(n));
    }
    return true;
}

inline bool header_has_token(const std::string& value, std::string_view token) {
    // Comma-separated, case-insensitive (Connection: keep-alive, Upgrade).
    size_t start = 0;
    while (start <= value.size()) {
        const size_t end  = std::min(value.find(',', start), value.size());
        size_t first      = value.find_first_not_of(" \t", start);
        size_t last       = end;
        while (last > first && (value[last - 1] == ' ' || value[last - 1] == '\t')) --last;
        if (first < last && last - first == token.size()) {
            bool same = true;
            for (size_t i = 0; i < token.size(); ++i)
                same = same && std::tolower(static_cast<unsigned char>(value[first + i])) ==
                                   std::tolower(static_cast<unsigned char>(token[i]));
            if (same) return true;
        }
        start = end + 1;
    }
    return false;
}

/// The length of the longest prefix of `text` that is whole UTF-8 characters, at most `limit`.
inline size_t utf8_prefix(std::string_view text, size_t limit) {
    if (text.size() <= limit) return text.size();
    size_t end = limit;
    while (end > 0 && (static_cast<uint8_t>(text[end]) & 0xC0) == 0x80) --end;
    return end;
}

} // namespace detail

/// True when `text` is well-formed UTF-8 (no overlong forms, surrogates or code points past
/// U+10FFFF), as RFC 6455 requires of text messages and close reasons.
inline bool valid_utf8(std::string_view text) {
    size_t i = 0;
    while (i < text.size()) {
        const uint8_t c = static_cast<uint8_t>(text[i]);
        size_t length;
        uint32_t code;
        if (c < 0x80) { ++i; continue; }
        if ((c & 0xE0) == 0xC0) { length = 2; code = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { length = 3; code = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { length = 4; code = c & 0x07; }
        else return false;
        if (i + length > text.size()) return false;
        for (size_t j = 1; j < length; ++j) {
            const uint8_t next = static_cast<uint8_t>(text[i + j]);
            if ((next & 0xC0) != 0x80) return false;
            code = (code << 6) | (next & 0x3F);
        }
        if ((length == 2 && code < 0x80) || (length == 3 && code < 0x800) ||
            (length == 4 && code < 0x10000) || code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF))
            return false;
        i += length;
    }
    return true;
}

/// True when `request` asks to open a WebSocket (RFC 6455 section 4.2.1).
inline bool is_upgrade(const httplib::Request& request) {
    return detail::header_has_token(request.get_header_value("Upgrade"), "websocket") &&
           detail::header_has_token(request.get_header_value("Connection"), "upgrade");
}

/// The Sec-WebSocket-Accept value for a client's Sec-WebSocket-Key.
inline std::string accept_key(const std::string& key) {
    const auto digest = detail::sha1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    return detail::base64(digest.data(), digest.size());
}

/// Why an opening handshake is refused, and the answer to give.
struct HandshakeError {
    int status;          ///< 426 Upgrade Required, or 400
    std::string message;
};

/// Checks the opening handshake. 426 (with `Upgrade: websocket`, or `Sec-WebSocket-Version: 13`,
/// which refuse_handshake() sets) when the client must switch to what this server speaks.
inline std::optional<HandshakeError> handshake_error(const httplib::Request& request) {
    if (request.method != "GET") return HandshakeError{400, "a WebSocket opens with GET"};
    if (!is_upgrade(request)) return HandshakeError{426, "expected Upgrade: websocket"};
    if (request.get_header_value("Sec-WebSocket-Version") != "13")
        return HandshakeError{426, "Sec-WebSocket-Version must be 13"};
    if (request.get_header_value("Sec-WebSocket-Key").empty())
        return HandshakeError{400, "Sec-WebSocket-Key is missing"};
    return std::nullopt;
}

/// The headers a refused handshake needs (RFC 6455 section 4.2.2 and 4.4).
inline void refuse_handshake(const httplib::Request& request, httplib::Response& response) {
    if (!is_upgrade(request)) response.set_header("Upgrade", "websocket");
    else response.set_header("Sec-WebSocket-Version", "13");
}

/// Answers the handshake and hands the connection to `session`.
inline void accept(const httplib::Request& request, httplib::Response& response,
                   std::function<void(httplib::Stream&)> session) {
    response.set_header("Upgrade", "websocket");
    response.set_header("Connection", "Upgrade");
    response.set_header("Sec-WebSocket-Accept", accept_key(request.get_header_value("Sec-WebSocket-Key")));
    response.switch_protocols(std::move(session));
}

enum class Opcode : uint8_t { Continuation = 0x0, Text = 0x1, Binary = 0x2, Close = 0x8, Ping = 0x9, Pong = 0xA };

/// Writes one unfragmented frame (servers never mask).
inline bool send(httplib::Stream& stream, Opcode opcode, std::string_view payload) {
    std::string frame;
    frame.reserve(payload.size() + 10);
    frame.push_back(static_cast<char>(0x80 | static_cast<uint8_t>(opcode)));
    if (payload.size() < 126) {
        frame.push_back(static_cast<char>(payload.size()));
    } else if (payload.size() <= 0xFFFF) {
        frame.push_back(static_cast<char>(126));
        frame.push_back(static_cast<char>((payload.size() >> 8) & 0xFF));
        frame.push_back(static_cast<char>(payload.size() & 0xFF));
    } else {
        frame.push_back(static_cast<char>(127));
        for (int i = 7; i >= 0; --i) frame.push_back(static_cast<char>((uint64_t(payload.size()) >> (8 * i)) & 0xFF));
    }
    frame.append(payload);
    return detail::write_all(stream, frame);
}

/// Sends a close frame with `code` and a short reason (cut at a character boundary to fit; dropped
/// if it is not valid UTF-8), then closes the sending side (the server closes first, RFC 6455
/// section 7.1.1) and lets what the peer still sends arrive -- the rest of a refused message, its
/// own close frame -- and drops it until the peer closes, for at most a second or a MiB. Closing
/// a socket with unread data resets the connection, and the peer may then lose the close frame
/// and the error before it.
inline void close(httplib::Stream& stream, uint16_t code, std::string_view reason = {}) {
    std::string payload;
    payload.push_back(static_cast<char>(code >> 8));
    payload.push_back(static_cast<char>(code & 0xFF));
    if (valid_utf8(reason)) payload.append(reason.substr(0, detail::utf8_prefix(reason, 123)));
    if (!send(stream, Opcode::Close, payload)) return;
    const auto socket = stream.socket();
    if (socket == INVALID_SOCKET) return;
    ::shutdown(socket, SHUT_WR);
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    char sink[16384];
    for (size_t dropped = 0; dropped < (1u << 20);) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            until - std::chrono::steady_clock::now()).count();
        pollfd ready{socket, POLLIN, 0};
        if (left <= 0 || ::poll(&ready, 1, static_cast<int>(left)) <= 0) return;
        const auto n = ::recv(socket, sink, sizeof(sink), 0);
        if (n <= 0) return; // the peer closed
        dropped += static_cast<size_t>(n);
    }
}

struct Message {
    Opcode opcode = Opcode::Text; ///< Text, Binary or Close
    std::string payload;
    uint16_t close_code = 1005; ///< for Close: the peer's code (1005 when it gave none)
};

/// Why receive() returned no message: the connection broke or timed out (`code` 0, nothing more
/// can be said), or the peer broke the protocol (`code` is the close code to answer with:
/// 1002 protocol error, 1007 invalid UTF-8, 1009 too big).
struct Failure {
    uint16_t code = 0;
    std::string reason;
};

/// True for the close codes a peer may send (RFC 6455 section 7.4).
inline bool valid_close_code(uint16_t code) {
    return (code >= 1000 && code <= 1003) || (code >= 1007 && code <= 1014) ||
           (code >= 3000 && code <= 4999);
}

/// Reads the next data message (fragments joined), answering pings along the way. Returns
/// nullopt when the connection breaks or violates the protocol (`failure` says which), and a
/// Close message when the peer closes. Messages larger than `max_size` violate the protocol.
inline std::optional<Message> receive(httplib::Stream& stream, size_t max_size, Failure* failure = nullptr) {
    const auto fail = [&](uint16_t code, const char* reason) -> std::optional<Message> {
        if (failure != nullptr) *failure = Failure{code, reason};
        return std::nullopt;
    };
    Message message;
    bool fragmented = false;
    for (;;) {
        char head[2];
        if (!detail::read_exact(stream, head, 2)) return fail(0, "");
        const bool fin      = (uint8_t(head[0]) & 0x80) != 0;
        const auto opcode   = static_cast<Opcode>(uint8_t(head[0]) & 0x0F);
        const bool masked   = (uint8_t(head[1]) & 0x80) != 0;
        uint64_t length     = uint8_t(head[1]) & 0x7F;
        if ((uint8_t(head[0]) & 0x70) != 0) return fail(1002, "reserved bits are set; no extension was agreed");
        if (!masked) return fail(1002, "client frames must be masked");
        if (length == 126 || length == 127) {
            char extended[8];
            const int bytes = length == 126 ? 2 : 8;
            if (!detail::read_exact(stream, extended, bytes)) return fail(0, "");
            length = 0;
            for (int i = 0; i < bytes; ++i) length = (length << 8) | uint8_t(extended[i]);
        }
        const bool control = (static_cast<uint8_t>(opcode) & 0x8) != 0;
        if (control && (!fin || length > 125)) return fail(1002, "invalid control frame");
        // A control frame (at most 125 bytes) never counts toward the data message it interrupts.
        if (!control && (length > max_size || length > max_size - message.payload.size()))
            return fail(1009, "message too big");
        char mask[4];
        if (!detail::read_exact(stream, mask, 4)) return fail(0, "");
        std::string payload(static_cast<size_t>(length), '\0');
        if (length > 0 && !detail::read_exact(stream, payload.data(), payload.size())) return fail(0, "");
        for (size_t i = 0; i < payload.size(); ++i) payload[i] ^= mask[i % 4];
        switch (opcode) {
        case Opcode::Ping:
            if (!send(stream, Opcode::Pong, payload)) return fail(0, "");
            continue;
        case Opcode::Pong: continue;
        case Opcode::Close: {
            Message closing;
            closing.opcode = Opcode::Close;
            if (payload.size() == 1) return fail(1002, "invalid close frame");
            if (payload.size() >= 2) {
                closing.close_code = uint16_t(uint8_t(payload[0]) << 8 | uint8_t(payload[1]));
                if (!valid_close_code(closing.close_code)) return fail(1002, "invalid close code");
                if (!valid_utf8(std::string_view(payload).substr(2)))
                    return fail(1007, "close reason is not valid UTF-8");
            }
            return closing;
        }
        case Opcode::Text:
        case Opcode::Binary:
            if (fragmented) return fail(1002, "a new message began before the last one ended");
            message.opcode  = opcode;
            message.payload = std::move(payload);
            fragmented      = !fin;
            break;
        case Opcode::Continuation:
            if (!fragmented) return fail(1002, "continuation without a message");
            message.payload += payload;
            fragmented = !fin;
            break;
        default: return fail(1002, "reserved opcode");
        }
        if (!fragmented) {
            if (message.opcode == Opcode::Text && !valid_utf8(message.payload))
                return fail(1007, "text message is not valid UTF-8");
            return message;
        }
    }
}

} // namespace sinfer::serve::websocket
