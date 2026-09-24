// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
// Prometheus metrics for the native STT and TTS servers, in the Rune server's format
// (serve/http_server.cpp, /metrics): what a supervisor drains a speech server by, and what a
// scraper watches. Header-only, so both servers' separate builds share it.

#include <httplib.h>

#include <array>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace sinfer::serve::audio {

class Metrics {
public:
    struct Options {
        bool streams    = false; ///< the server has realtime streams (STT)
        bool characters = false; ///< the server bills input characters (TTS)
    };

    /// `model` is the served model id, the `model` label of every series.
    Metrics(std::string model, Options options) : model_(escape(model)), options_(options) {}

    /// A request counted in `surogate_requests{state="running"}` for as long as the returned
    /// token lives. The servers take one for every request that is not a GET, before its body is
    /// read, and keep it on the response (`Response::hold_resource`), so a request counts from
    /// arrival -- queued or uploading -- until its response has been written or its client has
    /// gone.
    class Running {
    public:
        explicit Running(Metrics& owner) : owner_(owner) { owner_.running_.fetch_add(1); }
        ~Running() { owner_.running_.fetch_sub(1); }
        Running(const Running&)            = delete;
        Running& operator=(const Running&) = delete;

    private:
        Metrics& owner_;
    };
    [[nodiscard]] std::shared_ptr<Running> start_request() { return std::make_shared<Running>(*this); }

    /// The endpoints a server counts, fixed at startup so counting needs no lock.
    void declare_endpoint(const char* endpoint) {
        for (Endpoint& slot : counters_) {
            if (slot.name == nullptr) {
                slot.name = endpoint;
                return;
            }
        }
        throw std::logic_error("audio metrics: too many endpoints");
    }
    void request_done(std::string_view endpoint, bool ok) {
        (ok ? counter(endpoint).ok : counter(endpoint).error).fetch_add(1);
    }
    void add_audio_seconds(double seconds) {
        audio_microseconds_.fetch_add(static_cast<std::uint64_t>(seconds * 1e6 + 0.5));
    }
    void add_characters(std::uint64_t characters) { characters_.fetch_add(characters); }
    void set_open_streams(std::uint64_t streams) { streams_.store(streams); }

    [[nodiscard]] std::uint64_t running() const { return running_.load(); }

    /// The Prometheus text exposition.
    [[nodiscard]] std::string render() const {
        std::string out;
        out.reserve(1024);
        const auto help = [&out](std::string_view name, std::string_view kind, std::string_view what) {
            out += "# HELP surogate_";
            out += name;
            out += ' ';
            out += what;
            out += "\n# TYPE surogate_";
            out += name;
            out += ' ';
            out += kind;
            out += '\n';
        };
        const std::string model = "model=\"" + model_ + "\"";
        help("up", "gauge", "1 when the server is answering.");
        out += "surogate_up 1\n";
        // Rune's name and label, so one parser drains every Surogate server.
        help("requests", "gauge",
             "Requests in each state. running: every request but a GET, from arrival (queued or "
             "uploading included) until its response has been written or its client has gone.");
        out += "surogate_requests{" + model + ",state=\"running\"} " + std::to_string(running()) + "\n";
        if (options_.streams) {
            help("streams_open", "gauge",
                 "Open realtime transcription streams, from creation until finished, deleted or "
                 "expired.");
            out += "surogate_streams_open{" + model + "} " + std::to_string(streams_.load()) + "\n";
        }
        help("requests_total", "counter", "Finished requests by endpoint and outcome.");
        for (const Endpoint& endpoint : counters_) {
            if (endpoint.name == nullptr) { continue; }
            for (const auto& [outcome, value] :
                 {std::pair<const char*, std::uint64_t>{"ok", endpoint.counts.ok.load()},
                  std::pair<const char*, std::uint64_t>{"error", endpoint.counts.error.load()}}) {
                out += "surogate_requests_total{" + model + ",endpoint=\"" + endpoint.name +
                       "\",outcome=\"" + outcome + "\"} " + std::to_string(value) + "\n";
            }
        }
        help("audio_seconds_total", "counter",
             options_.streams ? "Seconds of audio transcribed." : "Seconds of audio synthesized.");
        out += "surogate_audio_seconds_total{" + model + "} " +
               decimal(static_cast<double>(audio_microseconds_.load()) / 1e6) + "\n";
        if (options_.characters) {
            help("characters_total", "counter", "Input characters synthesized, as billed.");
            out += "surogate_characters_total{" + model + "} " + std::to_string(characters_.load()) +
                   "\n";
        }
        return out;
    }

    /// Registers GET /metrics. `before` runs first, for state to bring up to date (the STT
    /// server prunes expired streams there, when it can without waiting).
    void serve(httplib::Server& server, std::function<void()> before = {}) {
        server.Get("/metrics", [this, before = std::move(before)](const httplib::Request&,
                                                                  httplib::Response& r) {
            if (before) { before(); }
            r.set_content(render(), "text/plain; version=0.0.4; charset=utf-8");
        });
    }

private:
    struct Counts {
        std::atomic<std::uint64_t> ok{0};
        std::atomic<std::uint64_t> error{0};
    };
    struct Endpoint {
        const char* name = nullptr;
        Counts counts;
    };
    Counts& counter(std::string_view endpoint) {
        for (Endpoint& slot : counters_) {
            if (slot.name != nullptr && endpoint == slot.name) { return slot.counts; }
        }
        throw std::logic_error("audio metrics: undeclared endpoint");
    }
    // Independent of the C locale, unlike std::to_string(double).
    static std::string decimal(double value) {
        std::array<char, 64> buffer{};
        const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                          std::chars_format::fixed, 6);
        return std::string(buffer.data(), result.ptr);
    }
    static std::string escape(std::string_view value) {
        std::string escaped;
        for (const char c : value) {
            if (c == '\\' || c == '"') { escaped.push_back('\\'); }
            if (c == '\n') {
                escaped += "\\n";
                continue;
            }
            escaped.push_back(c);
        }
        return escaped;
    }

    std::string model_;
    Options options_;
    std::atomic<std::uint64_t> running_{0};
    std::atomic<std::uint64_t> streams_{0};
    std::atomic<std::uint64_t> audio_microseconds_{0};
    std::atomic<std::uint64_t> characters_{0};
    Endpoint counters_[4];
};

} // namespace sinfer::serve::audio
