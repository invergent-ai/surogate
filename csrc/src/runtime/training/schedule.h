// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_TRAINING_SCHEDULE_H
#define SUROGATE_SRC_TRAINING_SCHEDULE_H

#include <cmath>
#include <numbers>
#include <algorithm> // std::clamp

/**
 * @brief Interface for scalar schedules evaluated per training step.
 *
 * A schedule maps an integer step index (typically starting at 0) to a float
 * value (e.g., learning rate).
 */
class ISchedule {
public:
    /** @brief Virtual destructor. */
    virtual ~ISchedule() = default;

    /**
     * @brief Evaluate the schedule at a given step.
     * @param step Current step index (typically >= 0).
     * @return Scheduled value for the given step.
     */
    virtual float eval(int step) const = 0;
};

/**
 * @brief Cosine decay schedule with optional warmup and base rate floor.
 *
 * Behavior:
 * - For steps in [0, warmup): linear warmup from 0 to peak rate.
 * - For steps >= warmup: cosine interpolation from peak rate down to base rate.
 */
class CosineSchedule : public ISchedule {
public:
    /**
     * @brief Construct a constant schedule at @p peak_rate (no warmup/decay configured).
     * @param peak_rate Peak value returned by the schedule.
     */
    explicit CosineSchedule(float peak_rate) : mPeakRate(peak_rate), mBaseRate(peak_rate) {}

    /**
     * @brief Construct a cosine schedule with warmup and decay.
     * @param peak_rate Value reached after warmup and used as the cosine start.
     * @param steps Number of decay steps (denominator for progress after warmup).
     * @param warmup Number of warmup steps (linear ramp from 0 to @p peak_rate).
     * @param base_rate Minimum value at the end of cosine decay (floor).
     */
    CosineSchedule(float peak_rate, int steps, int warmup, float base_rate)
        : mPeakRate(peak_rate), mDecaySteps(steps), mWarmupSteps(warmup), mBaseRate(base_rate) {}

    /**
     * @brief Evaluate schedule value at @p step.
     * @param step Current step index (typically >= 0).
     * @return Scheduled value at @p step.
     */
    float eval(int step) const override {
        if(step < mWarmupSteps) {
            return mPeakRate * step / mWarmupSteps;
        }
        double pos = (double)(step - mWarmupSteps) / mDecaySteps;
        double frac = 0.5 * std::cos(pos * std::numbers::pi) + 0.5;
        return static_cast<float>(frac * (mPeakRate - mBaseRate) + mBaseRate);
    }

private:
    int mWarmupSteps = 0;
    int mDecaySteps = 1;
    float mPeakRate;
    float mBaseRate;
};

/**
 * @brief Linear interpolation schedule with optional warmup.
 *
 * Behavior:
 * - For steps in [0, warmup): linear warmup from 0 to @p start.
 * - For steps >= warmup: linearly interpolates from @p start to @p end over @p steps,
 *   clamped to [0, 1] progress.
 */
class LinearSchedule : public ISchedule {
public:
    /**
     * @brief Construct a linear schedule.
     * @param start Value at the beginning of decay (after warmup).
     * @param end Value at the end of decay.
     * @param steps Number of decay steps (denominator for progress after warmup).
     * @param warmup Number of warmup steps (linear ramp from 0 to @p start).
     */
    explicit LinearSchedule(float start, float end, int steps, int warmup)
        : mStart(start), mEnd(end), mDecaySteps(steps), mWarmupSteps(warmup) {  }

    /**
     * @brief Evaluate schedule value at @p step.
     * @param step Current step index (typically >= 0).
     * @return Scheduled value at @p step.
     */
    float eval(int step) const override {
        if(step < mWarmupSteps) {
            return mStart * step / mWarmupSteps;
        }
        double pos = (double)(step - mWarmupSteps) / mDecaySteps;
        pos = std::clamp(pos, 0.0, 1.0);
        return mStart + (mEnd-mStart) * pos;
    }

private:
    int mWarmupSteps = 0;
    float mStart;
    float mEnd;
    int mDecaySteps;
};

#endif //SUROGATE_SRC_TRAINING_SCHEDULE_H
