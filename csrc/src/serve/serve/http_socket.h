// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
// Socket settings every HTTP server of the engine shares: the generation server, STT, TTS and
// the embedding server.
#include <httplib.h>

namespace sinfer::serve {

/// Turn Nagle's algorithm off on every connection `server` accepts. Call it before the server
/// binds: the option is set on the listening socket, and Linux gives it to each socket
/// `accept` returns.
///
/// cpp-httplib writes a response in two sends, the status line and headers first and the body
/// second, and leaves Nagle on unless told otherwise (`CPPHTTPLIB_TCP_NODELAY` defaults to
/// false). Nagle then holds the body until the client acknowledges the headers, and a client
/// delays that acknowledgement by about 40 ms. A new connection escapes it, because Linux
/// acknowledges at once early in a connection, so the cost fell on every later request of a
/// keep-alive connection -- which is every request of a pooling client: a gateway, the OpenAI
/// and Anthropic SDKs, OpenRouter. Measured on Rune decisions: 64 ms on a new connection,
/// 104 ms on each reuse.
inline void disable_nagle(httplib::Server& server) { server.set_tcp_nodelay(true); }

} // namespace sinfer::serve
