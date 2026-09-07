#include "core/decode_graph.h"
#include <cstdlib>

#include "core/device.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace sinfer {
namespace {

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void discard_capture(cudaStream_t stream) noexcept {
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
}

} // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_) {
    other.graph_ = nullptr;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_ = other.graph_;

    other.graph_ = nullptr;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    reset();

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    try {
        body();
    } catch (...) {
        discard_capture(stream);
        throw;
    }

    cudaGraph_t graph = nullptr;

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        destroy_graph(graph);
        // Throw rather than abort. Both graph families are written to fall
        // back to the eager body when a capture stops fitting, and an
        // aborting check makes those recovery paths unreachable.
        throw std::runtime_error("CUDA Graph capture failed: " +
                                 std::string(cudaGetErrorName(err)) + ": " +
                                 cudaGetErrorString(err));
    }

    graph_ = graph;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_) {
    other.exec_ = nullptr;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    exec_       = other.exec_;
    other.exec_ = nullptr;
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    reset();

    cudaGraphExec_t exec  = nullptr;
    const cudaError_t err = cudaGraphInstantiate(&exec, definition.graph_, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        // Instantiation allocates, so this is where a tight device fails, and
        // it is recoverable: serving the shape eagerly is slower but correct.
        // Aborting here killed a live server mid-round.
        (void)cudaGetLastError();
        throw std::runtime_error("CUDA Graph instantiation failed: " +
                                 std::string(cudaGetErrorName(err)) + ": " +
                                 cudaGetErrorString(err));
    }
    exec_ = exec;
    // The node count is the direct measure of what a captured round costs to
    // dispatch, and nothing else reports it. Off unless asked for.
    if (std::getenv("SUROGATE_SERVE_GRAPH_NODES") != nullptr) {
        std::size_t nodes = 0;
        if (cudaGraphGetNodes(definition.graph_, nullptr, &nodes) == cudaSuccess) {
            std::fprintf(stderr, "graph-nodes: %zu\n", nodes);
            std::fflush(stderr);
        }
    }
}

bool DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = cudaGraphExecUpdate(exec_, definition.graph_, &result);
    if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
        // An update the driver refuses (parameters it cannot patch in place — seen on a
        // pipeline stage's 64-lane profiles) is not fatal: re-instantiate from the new
        // definition instead. Slower to switch, identical to run.
        (void)cudaGetLastError();
        static bool reported = false;
        if (!reported) {
            reported = true;
            std::fprintf(stderr,
                         "decode graph: executable update refused (%s, result %d); re-instantiating instead\n",
                         cudaGetErrorName(err), static_cast<int>(result.result));
        }
        instantiate(definition);
        return true; // the caller must upload the fresh executable before launching it
    }
    return false;
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    // Upload commits the executable's device allocation, so it shares
    // instantiate's failure mode and must stay recoverable for the same reason.
    const cudaError_t err = cudaGraphUpload(exec_, stream);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        throw std::runtime_error("CUDA Graph upload failed: " +
                                 std::string(cudaGetErrorName(err)) + ": " +
                                 cudaGetErrorString(err));
    }
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept { destroy_graph_exec(exec_); }

} // namespace sinfer
