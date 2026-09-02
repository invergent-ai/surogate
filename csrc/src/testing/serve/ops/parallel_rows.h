#pragma once

// Row-parallel evaluation for the op oracles, over one pool for the whole run.
//
// The oracles are called once per case and several times per call, so spawning
// a thread set per call is not a detail: the sparse-MoE oracle's widest cases
// created over a million threads, and the test spent most of its wall time in
// clone3 and allocate_stack rather than in arithmetic (62 s, of which ~52 s was
// thread churn). The pool is process-local and outlives every call, so the cost
// of a call is the work it does.

#include "core/host_worker_pool.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <thread>
#include <vector>

namespace sinfer::test {

/// Threads an oracle pool may hold.
///
/// The suite runs under `ctest -j`, so several test processes are live at once
/// and a pool sized to the whole machine in each of them oversubscribes badly:
/// eight processes holding 64 threads apiece on 64 cores took the suite from
/// 53 s to 132 s and pushed a CPU-bound test past its own tolerance. A modest
/// per-process pool composes with the outer parallelism instead of fighting it;
/// SUROGATE_TEST_ORACLE_THREADS overrides it for a test run on its own.
inline std::int32_t oracle_pool_threads() {
    const std::int32_t hardware =
        std::max(1, static_cast<std::int32_t>(std::thread::hardware_concurrency()));
    if (const char* requested = std::getenv("SUROGATE_TEST_ORACLE_THREADS")) {
        const int parsed = std::atoi(requested);
        if (parsed > 0) { return std::min(hardware, static_cast<std::int32_t>(parsed)); }
    }
    return std::min(hardware, std::int32_t{8});
}

/// Applies `function(begin, end)` to a partition of [0, rows) over a shared
/// pool. Use this when a chunk carries state of its own -- a scratch buffer
/// per thread, say -- so it is built once per chunk rather than once per row.
template <class Function>
void parallel_row_ranges(std::int32_t rows, Function&& function) {
    if (rows <= 0) { return; }
    static const std::int32_t available = oracle_pool_threads();
    static HostWorkerPool pool(static_cast<std::uint32_t>(available), 4096);

    const std::int32_t threads = std::min(rows, available);
    const auto chunk           = [&](std::int32_t index) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * index) / threads);
        const std::int32_t end =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * (index + 1)) / threads);
        function(begin, end);
    };
    if (threads <= 1) {
        function(0, rows);
        return;
    }
    std::vector<std::future<void>> pending;
    pending.reserve(static_cast<std::size_t>(threads) - 1);
    for (std::int32_t index = 1; index < threads; ++index) {
        pending.push_back(pool.submit([&chunk, index] { chunk(index); }));
    }
    chunk(0);
    for (std::future<void>& wait : pending) { wait.get(); }
}

/// Applies `function` to every row in [0, rows), in parallel over a shared pool.
/// The calling thread takes the first chunk, so a small row count costs no
/// handoff and a single-chunk range never touches the pool at all.
template <class Function>
void parallel_rows(std::int32_t rows, Function&& function) {
    if (rows <= 0) { return; }
    static const std::int32_t available = oracle_pool_threads();
    static HostWorkerPool pool(static_cast<std::uint32_t>(available), 4096);

    const std::int32_t threads = std::min(rows, available);
    const auto chunk           = [&](std::int32_t index) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * index) / threads);
        const std::int32_t end =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * (index + 1)) / threads);
        for (std::int32_t row = begin; row < end; ++row) { function(row); }
    };
    if (threads <= 1) {
        for (std::int32_t row = 0; row < rows; ++row) { function(row); }
        return;
    }
    std::vector<std::future<void>> pending;
    pending.reserve(static_cast<std::size_t>(threads) - 1);
    for (std::int32_t index = 1; index < threads; ++index) {
        pending.push_back(pool.submit([&chunk, index] { chunk(index); }));
    }
    chunk(0);
    for (std::future<void>& wait : pending) { wait.get(); }
}

} // namespace sinfer::test
