#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace sinfer::serve {

// Process-wide admission, before parsing/preprocessing. A permit belongs to the HTTP
// response, so streaming, errors and disconnected clients all release it the same way.
class AdmissionLimit {
public:
    using Clock = std::chrono::steady_clock;
    struct Result {
        bool accepted;
        unsigned retry_after;
        std::shared_ptr<void> permit;
    };

    AdmissionLimit(double rate, std::uint32_t burst, std::uint32_t inflight,
                   Clock::time_point now = Clock::now())
        : state_(std::make_shared<State>(rate, burst, inflight, now)) {
        if (!std::isfinite(rate) || rate < 0 || rate > 1000000 ||
            (rate > 0 && (rate < 0.001 || burst == 0))) {
            throw std::invalid_argument("admission rate needs a finite nonnegative rate and positive burst");
        }
    }

    Result acquire(Clock::time_point now = Clock::now()) {
        const auto& s = state_;
        std::lock_guard lock(s->mutex);
        if (s->rate > 0) {
            s->tokens = std::min<double>(s->burst, s->tokens +
                std::max(0.0, std::chrono::duration<double>(now - s->updated).count()) * s->rate);
            s->updated = std::max(now, s->updated);
        }
        if (s->limit && s->active >= s->limit) { return {false, 1, {}}; }
        if (s->rate > 0 && s->tokens < 1) {
            return {false, static_cast<unsigned>(std::max(1.0, std::ceil((1 - s->tokens) / s->rate))), {}};
        }
        // Allocate before spending tokens so allocation failure cannot leak capacity.
        std::shared_ptr<void> permit;
        if (s->limit) { permit = std::make_shared<Permit>(s); }
        if (s->rate > 0) { s->tokens -= 1; }
        if (s->limit) { ++s->active; }
        return {true, 0, std::move(permit)};
    }

private:
    struct State {
        State(double r, std::uint32_t b, std::uint32_t l, Clock::time_point now)
            : rate(r), burst(b), limit(l), tokens(b), updated(now) {}
        std::mutex mutex;
        double rate;
        std::uint32_t burst, limit, active = 0;
        double tokens;
        Clock::time_point updated;
    };
    struct Permit {
        explicit Permit(std::shared_ptr<State> s) : state(std::move(s)) {}
        ~Permit() { std::lock_guard lock(state->mutex); --state->active; }
        std::shared_ptr<State> state;
    };
    std::shared_ptr<State> state_;
};

} // namespace sinfer::serve
