// LoopCD for Inkling-2-Mini: the engine-wide setting (loopcd.h).

#include "targets/inkling2/impl/loopcd.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sinfer::targets::inkling2::detail {
namespace {

LoopcdSettings parse(const char* raw) {
    LoopcdSettings s;
    if (raw == nullptr) { return s; }
    std::string v(raw);
    if (v.empty() || v == "0" || v == "off") { return s; }
    std::string number = v;
    if (v.rfind("adaptive:", 0) == 0) {
        number = v.substr(9);
    } else if (v.rfind("fixed:", 0) == 0) {
        s.adaptive = false;
        number     = v.substr(6);
    }
    char* end      = nullptr;
    const float w  = std::strtof(number.c_str(), &end);
    if (number.empty() || end == nullptr || *end != '\0' || !(w >= 0.0F) || w > 4.0F) {
        throw std::invalid_argument("SUROGATE_SERVE_INKLING_LOOPCD: expected off, <w>, adaptive:<w> or fixed:<w> with "
                                    "0 <= w <= 4, got '" + v + "'");
    }
    s.enabled  = true;
    s.strength = w;
    return s;
}

std::atomic<int> g_first_plane{-1};

} // namespace

const LoopcdSettings& loopcd_settings() {
    static const LoopcdSettings settings = [] {
        LoopcdSettings s = parse(std::getenv("SUROGATE_SERVE_INKLING_LOOPCD"));
        if (s.enabled) {
            std::fprintf(stderr, "inkling2: LoopCD-Logits ON (%s strength %.3f; reference = pass 0)\n",
                         s.adaptive ? "adaptive, max" : "fixed", static_cast<double>(s.strength));
        }
        return s;
    }();
    return settings;
}

int loopcd_state_layers() { return TextConfig::layers + (loopcd_settings().enabled ? kLoopcdVisits : 0); }

void loopcd_set_first_plane(int plane) { g_first_plane.store(plane); }

int loopcd_first_plane() { return g_first_plane.load(); }

} // namespace sinfer::targets::inkling2::detail
