#include "family/impl/runtime/instance.h"
#include "family/impl/runtime/schedule.h"

#include <stdexcept>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule {

template <class Context, class Body>
void run_prepared(Context& state, DecodeGraphExecutable* executable, Body&& body) {
    if (executable != nullptr) {
        if (!executable->ready()) {
            throw std::logic_error("decode graph was not prepared at load time");
        }
        executable->launch(state.execution.device.stream);
    } else {
        body();
    }
}

template <class Context, class Body>
void capture_graph(Context& state, DecodeGraphDefinition& definition, Body&& body) {
    state.execution.work.reset();
    definition.capture(state.execution.device.stream, body);
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
