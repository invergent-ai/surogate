// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/optimizers/async_optimizer.h"

#include <chrono>
#include <stdexcept>
#include <utility>

#include "runtime/optimizers/cpu_adamw.h"

namespace optimizers {

AsyncOptimizer::AsyncOptimizer(bool overlap) : mOverlap(overlap) {
    if (mOverlap) {
        mThread = std::thread([this] { worker_loop(); });
    }
}

AsyncOptimizer::~AsyncOptimizer() {
    if (mOverlap) {
        {
            std::lock_guard<std::mutex> lock(mMu);
            mStop = true;
        }
        mWorkCv.notify_all();
        if (mThread.joinable()) {
            mThread.join();
        }
    }
}

void AsyncOptimizer::rethrow_if_error_locked(std::unique_lock<std::mutex>& lock) {
    if (mError) {
        std::exception_ptr err = mError;
        mError = nullptr;  // consume — don't re-throw the same error repeatedly
        lock.unlock();
        std::rethrow_exception(err);
    }
}

void AsyncOptimizer::submit(std::function<void()> update) {
    if (!mOverlap) {
        // Synchronous: run inline so results are bitwise-identical to a plain
        // optimizer step (the determinism gate). Surface exceptions directly.
        update();
        ++mSubmitted;
        ++mCompleted;
        return;
    }

    std::unique_lock<std::mutex> lock(mMu);
    // rethrow_if_error_locked throws (releasing the lock) if a worker update
    // failed; otherwise it returns with the lock still held.
    rethrow_if_error_locked(lock);
    // Depth-1 staleness: wait until every previously submitted update has
    // completed before enqueuing this one. At the enqueue point nothing is in
    // flight; after it, exactly one is — never more.
    mDoneCv.wait(lock, [this] { return mCompleted >= mSubmitted || mError != nullptr; });
    rethrow_if_error_locked(lock);
    mQueue.push(std::move(update));
    ++mSubmitted;
    mWorkCv.notify_one();
    // Return immediately — the worker runs `update` concurrently with the caller.
}

void AsyncOptimizer::drain() {
    if (!mOverlap) {
        return;  // synchronous submits already ran inline
    }
    std::unique_lock<std::mutex> lock(mMu);
    mDoneCv.wait(lock, [this] { return mCompleted >= mSubmitted || mError != nullptr; });
    rethrow_if_error_locked(lock);
}

void AsyncOptimizer::worker_loop() {
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(mMu);
            mWorkCv.wait(lock, [this] { return !mQueue.empty() || mStop; });
            if (mStop && mQueue.empty()) {
                return;
            }
            job = std::move(mQueue.front());
            mQueue.pop();
        }
        try {
            job();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mMu);
            if (!mError) {
                mError = std::current_exception();
            }
            ++mCompleted;  // keep fences from hanging; error surfaces on next call
            mDoneCv.notify_all();
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(mMu);
            ++mCompleted;
        }
        mDoneCv.notify_all();
    }
}

AsyncStaleAdamW::AsyncStaleAdamW(std::vector<float> master,
                                 float lr,
                                 float beta1,
                                 float beta2,
                                 float eps,
                                 float weight_decay,
                                 bool overlap)
    : mAsync(overlap),
      mMaster(std::move(master)),
      mM(mMaster.size(), 0.0f),
      mV(mMaster.size(), 0.0f),
      mLr(lr),
      mBeta1(beta1),
      mBeta2(beta2),
      mEps(eps),
      mWeightDecay(weight_decay) {}

void AsyncStaleAdamW::step(std::vector<float> grad, int delay_us) {
    if (grad.size() != mMaster.size()) {
        throw std::invalid_argument("AsyncStaleAdamW::step: grad size != master size");
    }
    const int step_idx = ++mStepIdx;  // submission is serialized; safe to stamp here
    mAsync.submit([this, g = std::move(grad), step_idx, delay_us]() mutable {
        if (delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        }
        cpu_adamw_step(mMaster.data(),
                       g.data(),
                       mM.data(),
                       mV.data(),
                       mMaster.size(),
                       mLr,
                       mBeta1,
                       mBeta2,
                       step_idx,
                       mEps,
                       mWeightDecay,
                       /*grad_scale=*/1.0f);
        mApplied.fetch_add(1, std::memory_order_release);
    });
}

std::vector<float> AsyncStaleAdamW::master() {
    mAsync.drain();
    return mMaster;
}

}  // namespace optimizers
