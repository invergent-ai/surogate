#include "ops/linear/bf16/bf16_cublaslt.h"
#include "ops/linear/fp8/fp8_cublaslt.h"
#include "ops/linear/nvfp4/nvfp4_cublaslt.h"
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
    // Same reason as the prefill bucket's: a graph captured lazily under a live request runs on
    // whichever thread and device the round is on, and every cuBLASLt plane creates its handle
    // and workspace on first use -- an allocation a capture forbids.
    ops::detail::bf16_cublaslt_prewarm();
    ops::detail::fp8_cublaslt_prewarm();
    ops::detail::nvfp4_cublaslt_prewarm();
    definition.capture(state.execution.device.stream, body);
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
