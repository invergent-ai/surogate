#pragma once

#include <cstddef>

namespace sinfer::runtime {

// An unfinished older flight must not block another GPU's completed work. Preserve issue
// order among ready flights so prompt priority cannot indefinitely starve decode work.
template <class Flights, class Ready>
int oldest_ready_pipeline_flight(const Flights& flights, Ready&& ready) {
    int selected = -1;
    for (std::size_t group = 0; group < flights.size(); ++group) {
        const auto& flight = flights[group];
        if (!flight.active || flight.between || !ready(flight.stage)) { continue; }
        if (selected < 0 || flight.stage_sequence < flights[static_cast<std::size_t>(selected)].stage_sequence) {
            selected = static_cast<int>(group);
        }
    }
    return selected;
}

} // namespace sinfer::runtime
