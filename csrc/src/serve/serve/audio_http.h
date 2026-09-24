// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
// Common HTTP contracts for native STT and TTS, independent of the model backend.
#include "http_socket.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>

namespace sinfer::serve::audio {
using json = nlohmann::ordered_json;

class HttpError : public std::runtime_error {
public:
    int status;

    HttpError(int status, const std::string& message)
        : std::runtime_error(message), status(status) {}
};

inline void response(httplib::Response& r, const json& body, int status = 200) {
    r.status = status;
    r.set_content(body.dump(), "application/json");
}

inline void error(httplib::Response& r, const std::string& message, int status) {
    response(r,
             {{"error",
               {{"message", message},
                {"type", status < 500 ? "invalid_request_error" : "server_error"}}}},
             status);
}

inline bool same_key(const std::string& a, const std::string& b) {
    size_t difference = a.size() ^ b.size();
    for (size_t i = 0; i < b.size(); ++i)
        difference |= (i < a.size() ? static_cast<unsigned char>(a[i]) : 0) ^
                      static_cast<unsigned char>(b[i]);
    return difference == 0;
}

inline void configure(httplib::Server& server, const std::string& key, size_t max_body) {
    disable_nagle(server); // both servers call this before they bind
    server.set_payload_max_length(max_body);
    server.set_read_timeout(60);
    server.set_pre_routing_handler([key](const httplib::Request& q, httplib::Response& r) {
        if (!key.empty() && !same_key(q.get_header_value("Authorization"), "Bearer " + key)) {
            error(r, "invalid API key", 401);
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    server.set_error_handler([](const auto&, auto& r) {
        if (r.status == 413 && r.body.empty())
            error(r, "Request body exceeds the configured payload limit", 413);
    });
    server.set_exception_handler([](const auto&, auto& r, std::exception_ptr exception) {
        try {
            std::rethrow_exception(exception);
        } catch (const HttpError& e) {
            error(r, e.what(), e.status);
        } catch (const nlohmann::json::exception&) {
            error(r, "Invalid JSON request or field type", 400);
        } catch (const std::invalid_argument& e) {
            error(r, e.what(), 400);
        } catch (const std::exception& e) {
            std::cerr << "speech request: " << e.what() << '\n';
            error(r, "speech inference failed", 500);
        } catch (...) { error(r, "speech inference failed", 500); }
    });
}
} // namespace sinfer::serve::audio
