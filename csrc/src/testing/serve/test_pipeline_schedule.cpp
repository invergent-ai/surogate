#include "runtime/engine/pipeline_schedule.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
    struct Flight {
        bool active;
        bool between;
        std::size_t stage;
        std::uint64_t stage_sequence;
    };
    // A long prompt on stage 0 was launched before a request finishing at stage 2.
    // Publishing the latter's first token must not wait for stage 0's GPU/CPU work.
    std::array<Flight, 4> flights{{{true, false, 0, 1}, {true, false, 2, 3},
                                 {true, false, 1, 2}, {true, true, 0, 0}}};
    std::array<bool, 3> done{false, false, true};
    const auto pick = [&] {
        return sinfer::runtime::oldest_ready_pipeline_flight(flights,
            [&](std::size_t stage) { return done[stage]; });
    };
    assert(pick() == 1);
    done[1] = true;
    assert(pick() == 2); // ready work remains fair in issue order
    done[0] = true;
    assert(pick() == 0);
    flights[0].active = false;
    assert(pick() == 2); // inactive and parked flights cannot be consumed
    done.fill(false);
    assert(pick() == -1); // no blocking wait when no GPU has finished
    std::cout << "pipeline advances completed stages without head-of-line blocking\n";
}
