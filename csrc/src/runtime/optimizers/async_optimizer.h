// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Dispatch-PP async 1-step-stale optimizer (design §"Async optimizer").
//
// A dedicated worker thread drains a depth-1 queue of optimizer-update closures,
// overlapping the CPU-master update with the next iteration's compute. Depth-1 (at
// most one update in flight) yields exactly one-step staleness: iteration N+1
// proceeds on weights as of iteration N-1's update, the accepted v1 default. A
// synchronous mode (overlap=false) runs updates inline for the determinism gate.

#ifndef SUROGATE_SRC_RUNTIME_OPTIMIZERS_ASYNC_OPTIMIZER_H
#define SUROGATE_SRC_RUNTIME_OPTIMIZERS_ASYNC_OPTIMIZER_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace optimizers {

/// Worker that overlaps optimizer updates with the next iteration's compute by
/// draining a depth-1 queue of update closures on a dedicated thread.
///
/// Ordering contract (overlap mode): `submit(u_N)` first fences on the previous
/// update `u_{N-1}` completing, then enqueues `u_N` and returns immediately — so
/// `u_N` runs concurrently with whatever the caller does next, and there is never
/// more than one update outstanding (the one-step-stale bound). A worker exception
/// is captured and re-thrown on the next `submit`/`drain` (no partial commit).
class AsyncOptimizer {
public:
    explicit AsyncOptimizer(bool overlap);
    ~AsyncOptimizer();

    AsyncOptimizer(const AsyncOptimizer&) = delete;
    AsyncOptimizer& operator=(const AsyncOptimizer&) = delete;

    /// Fence on the previous submission, enqueue `update`, return immediately
    /// (overlap mode). In synchronous mode, run `update` inline. Re-throws any
    /// pending worker exception before doing anything.
    void submit(std::function<void()> update);

    /// Block until every submitted update has completed; re-throw worker errors.
    void drain();

    [[nodiscard]] bool overlap_enabled() const {
        return mOverlap;
    }

private:
    void worker_loop();
    void rethrow_if_error_locked(std::unique_lock<std::mutex>& lock);

    const bool mOverlap;
    std::thread mThread;
    mutable std::mutex mMu;
    std::condition_variable mWorkCv;  // worker waits for queued work / stop
    std::condition_variable mDoneCv;  // submit/drain wait for completions
    std::queue<std::function<void()>> mQueue;
    bool mStop = false;
    long mSubmitted = 0;
    long mCompleted = 0;
    std::exception_ptr mError;
};

/// Concrete one-step-stale AdamW over FP32 CPU master parameters — the dispatch-PP
/// CPU-master optimizer path. Each `step()` queues a `cpu_adamw_step` update onto
/// the AsyncOptimizer. The master/moment buffers are touched only by the worker
/// (serialized by the depth-1 queue); read them via `master()` (which drains first).
class AsyncStaleAdamW {
public:
    AsyncStaleAdamW(std::vector<float> master,
                    float lr,
                    float beta1,
                    float beta2,
                    float eps,
                    float weight_decay,
                    bool overlap);

    /// Queue one AdamW update with `grad` (FP32, same length as the master).
    /// `delay_us` optionally stalls the worker so a caller can observe the
    /// in-flight (stale) state. Depth-1: returns once the previous update landed.
    void step(std::vector<float> grad, int delay_us);

    /// Number of updates the worker has finished applying (lock-free probe).
    [[nodiscard]] long applied_count() const {
        return mApplied.load(std::memory_order_acquire);
    }

    void drain() {
        mAsync.drain();
    }

    /// Drain, then return a copy of the FP32 master parameters.
    std::vector<float> master();

    [[nodiscard]] bool overlap_enabled() const {
        return mAsync.overlap_enabled();
    }

private:
    AsyncOptimizer mAsync;
    std::vector<float> mMaster;
    std::vector<float> mM;
    std::vector<float> mV;
    const float mLr;
    const float mBeta1;
    const float mBeta2;
    const float mEps;
    const float mWeightDecay;
    int mStepIdx = 0;
    std::atomic<long> mApplied{0};
};

}  // namespace optimizers

#endif  // SUROGATE_SRC_RUNTIME_OPTIMIZERS_ASYNC_OPTIMIZER_H
