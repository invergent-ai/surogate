// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kernels/decode.h"
#include "utilities/allocator.h"
#include "runtime/executor/paged_decode_cache.h"
#include <vector>

struct DecodeSamplingParams {
    double temperature = 1, top_p = 1, min_p = 0, repetition_penalty = 1;
    double presence_penalty = 0, frequency_penalty = 0, uniform = 0;
    int top_k = -1, top_count = 0, enabled = 1;
};
struct DecodeLogitBias {
    int row, token;
    double value;
};
struct DecodeSamplingRequest {
    DecodeSamplingParams params;
    std::vector<std::pair<int, double>> bias;
    std::vector<int> blocked;
    void validate(int vocabulary) const;
};
struct DecodeSampleResult {
    int token = -1, status = 0;
    double logprob = 0;
    int top_ids[20]{};
    double top_logprobs[20]{};
};

// Workspace shared by all model architectures. Only compact results leave GPU.
class DecodeSampler {
public:
    void set_budget(const std::shared_ptr<dsl::DecodePagePool>& pool) {
        mAllocator.pool = pool;
    }
    std::size_t workspace_bytes() const {
        return mAllocator.total_allocation();
    }
    void prepare(const DecodeSamplingRequest* requests,
                 int B,
                 int V,
                 ETensorDType dtype,
                 cudaStream_t stream,
                 bool upload = true);
    void run(const Tensor& counts, cudaStream_t stream);
    const Tensor& logits() const {
        return mLogits;
    }
    void copy_results(DecodeSampleResult* destination, cudaStream_t stream);
    void release_workspace();

private:
    dsl::DecodeWorkspaceAllocator mAllocator;
    Tensor mLogits, mValues, mSorted, mProbabilities, mIndices, mSortedIndices, mOffsets;
    Tensor mParams, mBias, mResults, mSort, mTileStats, mRowStats;
    std::vector<DecodeSamplingParams> mHostParams;
    std::vector<DecodeLogitBias> mHostBias;
    int mB = 0, mV = 0;
    int mBiasPassSize = 0;
    bool mNeedsSort = false;
    void reserve(Tensor& tensor, ETensorDType dtype, long elements, const char* name);
};

void decode_update_counts(const Tensor& bindings, const Tensor& tokens, int B, int T, int V, cudaStream_t stream);
