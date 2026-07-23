// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "tok_profile.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tokenizer {

bool tok_profile_enabled() {
    static const bool on = [] {
        const char* env = std::getenv("SUROGATE_TOK_PROFILE");
        return env && std::strcmp(env, "1") == 0;
    }();
    return on;
}

TokProfile& tok_profile() {
    static TokProfile prof;
    return prof;
}

namespace {
double ms(uint64_t ns) {
    return static_cast<double>(ns) / 1e6;
}
double pct(uint64_t part, uint64_t whole) {
    return whole ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
}
}  // namespace

std::string tok_profile_report(bool reset) {
    auto& p = tok_profile();
    const uint64_t calls = p.calls.load(std::memory_order_relaxed);
    if (calls == 0) return "";

    const uint64_t render = p.render_ns.load(std::memory_order_relaxed);
    const uint64_t render_walk = p.render_walk_ns.load(std::memory_order_relaxed);
    const uint64_t pretok = p.pretok_ns.load(std::memory_order_relaxed);
    const uint64_t bpe = p.bpe_ns.load(std::memory_order_relaxed);
    const uint64_t total = p.total_ns.load(std::memory_order_relaxed);
    const uint64_t tokens = p.tokens.load(std::memory_order_relaxed);
    const uint64_t renders = p.renders.load(std::memory_order_relaxed);
    const uint64_t render_setup = render > render_walk ? render - render_walk : 0;
    const uint64_t accounted = render + pretok + bpe;
    const uint64_t other = total > accounted ? total - accounted : 0;

    // total is summed per-call across worker threads, so it is aggregate
    // busy-time (not wall-clock); tokens/s below is therefore per-thread.
    const double total_s = static_cast<double>(total) / 1e9;
    const double tok_per_s = total_s > 0 ? static_cast<double>(tokens) / total_s : 0.0;

    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "[tok-profile] encode_for_training breakdown (SUROGATE_TOK_PROFILE=1)\n"
                  "  calls=%llu  renders=%llu  tokens=%llu  busy=%.1fms  (%.2f Mtok/s/thread)\n"
                  "  chat-template render : %9.1f ms  %5.1f%%\n"
                  "    - minja walk       : %9.1f ms  %5.1f%%\n"
                  "    - setup (json/ctx) : %9.1f ms  %5.1f%%\n"
                  "  pretokenize (regex)  : %9.1f ms  %5.1f%%\n"
                  "  BPE merge            : %9.1f ms  %5.1f%%\n"
                  "  other (json/scan/…)  : %9.1f ms  %5.1f%%\n",
                  static_cast<unsigned long long>(calls), static_cast<unsigned long long>(renders),
                  static_cast<unsigned long long>(tokens), ms(total), tok_per_s / 1e6, ms(render), pct(render, total),
                  ms(render_walk), pct(render_walk, total), ms(render_setup), pct(render_setup, total), ms(pretok),
                  pct(pretok, total), ms(bpe), pct(bpe, total), ms(other), pct(other, total));

    if (reset) {
        p.render_ns.store(0, std::memory_order_relaxed);
        p.render_walk_ns.store(0, std::memory_order_relaxed);
        p.pretok_ns.store(0, std::memory_order_relaxed);
        p.bpe_ns.store(0, std::memory_order_relaxed);
        p.total_ns.store(0, std::memory_order_relaxed);
        p.calls.store(0, std::memory_order_relaxed);
        p.renders.store(0, std::memory_order_relaxed);
        p.tokens.store(0, std::memory_order_relaxed);
    }
    return std::string(buf);
}

namespace {
// Print the summary at process exit too, so a plain `surogate tokenize` run
// with SUROGATE_TOK_PROFILE=1 reports even if no caller asks for it explicitly.
struct AtExitReporter {
    AtExitReporter() {
        if (tok_profile_enabled()) {
            std::atexit([] {
                const std::string s = tok_profile_report(false);
                if (!s.empty()) {
                    std::fputs(s.c_str(), stderr);
                    std::fflush(stderr);
                }
            });
        }
    }
} g_atexit_reporter;
}  // namespace

}  // namespace tokenizer
