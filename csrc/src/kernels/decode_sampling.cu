// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#include "kernels/decode_sampling.h"
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace {
constexpr int Threads = 256;
struct Best {
    double value;
    int token;
};
struct Better {
    __device__ Best operator()(Best a, Best b) const {
        return b.value > a.value || (b.value == a.value && b.token < a.token) ? b : a;
    }
};
struct Maximum {
    __device__ double operator()(double a, double b) const {
        return fmax(a, b);
    }
};
using Sum = cub::BlockReduce<double, Threads>;
using Select = cub::BlockReduce<Best, Threads>;
using Scan = cub::BlockScan<double, Threads>;

__global__ void clear_counts(const DecodeCacheBinding* bindings, int V) {
    const auto state = bindings[blockIdx.y];
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (!state.length && token < V) static_cast<int*>(state.data)[token] = 0;
}
__global__ void add_counts(const DecodeCacheBinding* bindings, const int* tokens, int T) {
    const int row = blockIdx.y, token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token < T) atomicAdd(static_cast<int*>(bindings[row].data) + tokens[row * T + token], 1);
}

template <class Float>
__global__ void transform(const Float* logits,
                          double* values,
                          int* indices,
                          int* offsets,
                          const DecodeSamplingParams* params,
                          const DecodeCacheBinding* bindings,
                          DecodeSampleResult* results,
                          int V) {
    const int row = blockIdx.y, token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token == 0) {
        offsets[row] = row * V;
        if (row + 1 == gridDim.y) offsets[row + 1] = (row + 1) * V;
    }
    if (token >= V) return;
    const auto p = params[row];
    const double raw = static_cast<float>(logits[row * V + token]);
    double value = raw / (p.temperature > 0 ? p.temperature : 1);
    if ((!isfinite(raw) || !isfinite(value)) && p.enabled) atomicExch(&results[row].status, 1);
    const int count = static_cast<const int*>(bindings[row].data)[token];
    if (count) {
        value = value > 0 ? value / p.repetition_penalty : value * p.repetition_penalty;
        value -= p.presence_penalty + count * p.frequency_penalty;
    }
    values[row * V + token] = value;
    indices[row * V + token] = token;
}

// Biases and hard exclusions are uploaded in two passes. A blocked token stays
// blocked even when the request also contains a positive bias for that token.
__global__ void apply_bias(double* values, const DecodeLogitBias* bias, int count, int V) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) values[bias[i].row * V + bias[i].token] += bias[i].value;
}

__global__ void filter(const double* values,
                       const int* indices,
                       double* probabilities,
                       const DecodeSamplingParams* params,
                       int V,
                       bool sorted) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const auto p = params[row];
    values += row * V;
    indices += row * V;
    probabilities += row * V;
    __shared__ union {
        Sum::TempStorage reduce;
        Scan::TempStorage scan;
    } storage;
    __shared__ double maximum, total, prefix;
    double local = -INFINITY;
    for (int i = tid; i < V; i += Threads)
        local = fmax(local, values[i]);
    const double m = Sum(storage.reduce).Reduce(local, Maximum{});
    if (tid == 0) maximum = m;
    __syncthreads();
    const int k = p.top_k > 0 ? min(p.top_k, V) : V;
    local = 0;
    for (int i = tid; i < k; i += Threads)
        local += isfinite(values[i]) ? exp(values[i] - maximum) : 0;
    const double sum = Sum(storage.reduce).Sum(local);
    if (tid == 0) {
        total = sum;
        prefix = 0;
    }
    __syncthreads();
    for (int start = 0; start < V; start += Threads) {
        const int i = start + tid;
        double weight = i < k && isfinite(values[i]) ? exp(values[i] - maximum) : 0;
        if (sorted) {
            double before, tile;
            Scan(storage.scan).ExclusiveSum(weight, before, tile);
            __syncthreads();
            if (prefix + before >= p.top_p * total) weight = 0;
            __syncthreads();
            if (tid == 0) prefix += tile;
            __syncthreads();
        }
        if (weight < p.min_p) weight = 0;  // maximum unnormalized weight is 1
        if (i < V) probabilities[sorted ? indices[i] : i] = weight;
    }
}

template <class Float>
__global__ void sample(const Float* logits,
                       const double* probabilities,
                       const DecodeSamplingParams* params,
                       DecodeSampleResult* results,
                       int V) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const auto p = params[row];
    auto& result = results[row];
    if (!p.enabled || result.status) {
        if (!tid) result.token = -1;
        return;
    }
    logits += row * V;
    probabilities += row * V;
    __shared__ union {
        Sum::TempStorage reduce;
        Select::TempStorage select;
        Scan::TempStorage scan;
    } storage;
    __shared__ double maximum, normalizer, total, prefix;
    __shared__ Best previous;
    const double temperature = p.temperature > 0 ? p.temperature : 1;
    double local_max = -INFINITY, local_sum = 0;
    for (int i = tid; i < V; i += Threads) {
        local_max = fmax(local_max, static_cast<float>(logits[i]) / temperature);
        local_sum += probabilities[i];
    }
    const double m = Sum(storage.reduce).Reduce(local_max, Maximum{});
    if (!tid) maximum = m;
    __syncthreads();
    const double mass = Sum(storage.reduce).Sum(local_sum);
    if (!tid) {
        total = mass;
        prefix = 0;
    }
    __syncthreads();
    if (!(total > 0) || !isfinite(total)) {
        if (!tid) {
            result.token = -1;
            result.status = 2;
        }
        return;
    }
    local_sum = 0;
    for (int i = tid; i < V; i += Threads)
        local_sum += exp(static_cast<float>(logits[i]) / temperature - maximum);
    const double partition = Sum(storage.reduce).Sum(local_sum);
    if (!tid) normalizer = maximum + log(partition);
    __syncthreads();
    Best best{-INFINITY, V};
    if (p.temperature == 0) {
        for (int i = tid; i < V; i += Threads)
            best = Better{}(best, {probabilities[i], i});
    } else {
        // Token-ID order matches inverse-CDF sampling on the host. The host
        // supplies one draw from each request's RNG, independent of batching.
        int selected = V, last = -1;
        for (int start = 0; start < V; start += Threads) {
            const int i = start + tid;
            const double weight = i < V ? probabilities[i] : 0;
            double before, tile;
            Scan(storage.scan).ExclusiveSum(weight, before, tile);
            __syncthreads();
            if (weight > 0) {
                last = i;
                if (prefix + before + weight > p.uniform * total) selected = min(selected, i);
            }
            __syncthreads();
            if (!tid) prefix += tile;
            __syncthreads();
        }
        // A final positive token also covers roundoff at the very end of CDF.
        best = selected < V ? Best{1, selected} : Best{0, -last};
    }
    const Best chosen = Select(storage.select).Reduce(best, Better{});
    if (!tid) {
        result.token = p.temperature == 0 || chosen.value == 1 ? chosen.token : -chosen.token;
        result.logprob = static_cast<float>(logits[result.token]) / temperature - normalizer;
        previous = {INFINITY, -1};
    }
    __syncthreads();
    for (int rank = 0; rank < min(p.top_count, V); ++rank) {
        Best next{-INFINITY, V};
        for (int i = tid; i < V; i += Threads) {
            const double value = static_cast<float>(logits[i]) / temperature;
            if (value < previous.value || (value == previous.value && i > previous.token))
                next = Better{}(next, {value, i});
        }
        const Best top = Select(storage.select).Reduce(next, Better{});
        if (!tid) {
            result.top_ids[rank] = top.token;
            result.top_logprobs[rank] = top.value - normalizer;
            previous = top;
        }
        __syncthreads();
    }
}
}  // namespace

void DecodeSamplingRequest::validate(int vocabulary) const {
    const auto& p = params;
    if (!std::isfinite(p.temperature) || p.temperature < 0 || !std::isfinite(p.top_p) || p.top_p <= 0 || p.top_p > 1 ||
        !std::isfinite(p.min_p) || p.min_p < 0 || p.min_p > 1 || !std::isfinite(p.repetition_penalty) ||
        p.repetition_penalty <= 0 || !std::isfinite(p.presence_penalty) || !std::isfinite(p.frequency_penalty) ||
        !std::isfinite(p.uniform) || p.uniform < 0 || p.uniform >= 1 || p.top_k < -1 || p.top_count < 0 ||
        p.top_count > 20 || (p.enabled != 0 && p.enabled != 1))
        throw std::invalid_argument("Invalid GPU sampling parameters");
    std::unordered_set<int> unique;
    for (const auto& [token, value] : bias)
        if (token < 0 || token >= vocabulary || !std::isfinite(value) || !unique.insert(token).second)
            throw std::invalid_argument("Invalid or duplicate logit bias");
    unique.clear();
    for (int token : blocked)
        if (token < 0 || token >= vocabulary || !unique.insert(token).second)
            throw std::invalid_argument("Invalid or duplicate blocked token");
}

void DecodeSampler::reserve(Tensor& tensor, ETensorDType dtype, long elements, const char* name) {
    if (tensor.Data && tensor.DType == dtype && tensor.nelem() >= elements) return;
    mAllocator.free(tensor);
    tensor = mAllocator.allocate(dtype, name, EAllocationType::ON_DEVICE, {elements});
}

void DecodeSampler::release_workspace() {
    for (auto* tensor : {&mLogits, &mValues, &mSorted, &mProbabilities, &mIndices, &mSortedIndices,
                         &mOffsets, &mParams, &mBias, &mResults, &mSort})
        mAllocator.free(*tensor);
}

void DecodeSampler::prepare(const DecodeSamplingRequest* requests,
                            int B,
                            int V,
                            ETensorDType dtype,
                            cudaStream_t stream,
                            bool upload) {
    if (B <= 0 || V <= 0 || static_cast<long>(B) * V > std::numeric_limits<int>::max())
        throw std::invalid_argument("Invalid GPU sampling dimensions");
    mB = B;
    mV = V;
    mNeedsSort = false;
    mHostParams.clear();
    mHostBias.clear();
    for (int row = 0; row < B; ++row) {
        requests[row].validate(V);
        const auto& p = requests[row].params;
        mNeedsSort |= p.top_p < 1 || (p.top_k > 0 && p.top_k < V);
        mHostParams.push_back(p);
        for (const auto& [token, value] : requests[row].bias)
            mHostBias.push_back({row, token, value});
    }
    const auto biased = mHostBias.size();
    for (int row = 0; row < B; ++row)
        for (int token : requests[row].blocked)
            mHostBias.push_back({row, token, -INFINITY});
    // Bias token IDs and blocked IDs are unique within each request (validated
    // by the binding); each pass writes a disjoint set of vocabulary entries.
    reserve(mLogits, dtype, static_cast<long>(B) * V, "decode_sample_logits");
    reserve(mValues, ETensorDType::BYTE, static_cast<long>(B) * V * sizeof(double), "decode_sample_values");
    reserve(mProbabilities, ETensorDType::BYTE, mValues.nelem(), "decode_sample_probabilities");
    reserve(mIndices, ETensorDType::INT32, static_cast<long>(B) * V, "decode_sample_indices");
    reserve(mOffsets, ETensorDType::INT32, B + 1, "decode_sample_offsets");
    reserve(mParams, ETensorDType::BYTE, B * sizeof(DecodeSamplingParams), "decode_sample_params");
    reserve(mBias,
            ETensorDType::BYTE,
            std::max<std::size_t>(1, mHostBias.size()) * sizeof(DecodeLogitBias),
            "decode_sample_bias");
    reserve(mResults, ETensorDType::BYTE, B * sizeof(DecodeSampleResult), "decode_sample_results");
    if (upload) CUDA_CHECK(cudaMemcpyAsync(mParams.Data,
                               mHostParams.data(),
                               B * sizeof(DecodeSamplingParams),
                               cudaMemcpyHostToDevice,
                               stream));
    if (upload && !mHostBias.empty())
        CUDA_CHECK(cudaMemcpyAsync(mBias.Data,
                                   mHostBias.data(),
                                   mHostBias.size() * sizeof(DecodeLogitBias),
                                   cudaMemcpyHostToDevice,
                                   stream));
    mBiasPassSize = biased;
    if (mNeedsSort) {
        reserve(mSorted, ETensorDType::BYTE, mValues.nelem(), "decode_sample_sorted");
        reserve(mSortedIndices, ETensorDType::INT32, static_cast<long>(B) * V, "decode_sample_sorted_indices");
        std::size_t bytes = 0;
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,
                                                                      bytes,
                                                                      reinterpret_cast<double*>(mValues.Data),
                                                                      reinterpret_cast<double*>(mSorted.Data),
                                                                      mIndices.get<int>(),
                                                                      mSortedIndices.get<int>(),
                                                                      B * V,
                                                                      B,
                                                                      mOffsets.get<int>(),
                                                                      mOffsets.get<int>() + 1,
                                                                      0,
                                                                      64,
                                                                      stream));
        reserve(mSort, ETensorDType::BYTE, bytes, "decode_sample_sort_workspace");
    }
}

void DecodeSampler::run(const Tensor& counts, cudaStream_t stream) {
    auto* params = reinterpret_cast<const DecodeSamplingParams*>(mParams.Data);
    auto* bindings = reinterpret_cast<const DecodeCacheBinding*>(counts.Data);
    auto* results = reinterpret_cast<DecodeSampleResult*>(mResults.Data);
    auto* values = reinterpret_cast<double*>(mValues.Data);
    auto* probabilities = reinterpret_cast<double*>(mProbabilities.Data);
    CUDA_CHECK(cudaMemsetAsync(results, 0, mB * sizeof(DecodeSampleResult), stream));
    auto launch = [&]<class Float>() {
        transform<<<dim3((mV + Threads - 1) / Threads, mB), Threads, 0, stream>>>(mLogits.get<Float>(),
                                                                                  values,
                                                                                  mIndices.get<int>(),
                                                                                  mOffsets.get<int>(),
                                                                                  params,
                                                                                  bindings,
                                                                                  results,
                                                                                  mV);
    };
    if (mLogits.DType == ETensorDType::BF16)
        launch.operator()<nv_bfloat16>();
    else if (mLogits.DType == ETensorDType::FP32)
        launch.operator()<float>();
    else
        throw std::invalid_argument("GPU sampling requires BF16 or FP32 logits");
    auto* bias = reinterpret_cast<const DecodeLogitBias*>(mBias.Data);
    if (mBiasPassSize)
        apply_bias<<<(mBiasPassSize + Threads - 1) / Threads, Threads, 0, stream>>>(values, bias, mBiasPassSize, mV);
    const int blocked = mHostBias.size() - mBiasPassSize;
    if (blocked)
        apply_bias<<<(blocked + Threads - 1) / Threads, Threads, 0, stream>>>(values,
                                                                              bias + mBiasPassSize,
                                                                              blocked,
                                                                              mV);
    auto* indices = mIndices.get<int>();
    if (mNeedsSort) {
        auto bytes = mSort.bytes();
        CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(mSort.Data,
                                                                      bytes,
                                                                      values,
                                                                      reinterpret_cast<double*>(mSorted.Data),
                                                                      indices,
                                                                      mSortedIndices.get<int>(),
                                                                      mB * mV,
                                                                      mB,
                                                                      mOffsets.get<int>(),
                                                                      mOffsets.get<int>() + 1,
                                                                      0,
                                                                      64,
                                                                      stream));
        values = reinterpret_cast<double*>(mSorted.Data);
        indices = mSortedIndices.get<int>();
    }
    filter<<<mB, Threads, 0, stream>>>(values, indices, probabilities, params, mV, mNeedsSort);
    if (mLogits.DType == ETensorDType::BF16)
        sample<<<mB, Threads, 0, stream>>>(mLogits.get<nv_bfloat16>(), probabilities, params, results, mV);
    else
        sample<<<mB, Threads, 0, stream>>>(mLogits.get<float>(), probabilities, params, results, mV);
    CUDA_CHECK(cudaGetLastError());
}

void DecodeSampler::copy_results(DecodeSampleResult* destination, cudaStream_t stream) {
    CUDA_CHECK(
        cudaMemcpyAsync(destination, mResults.Data, mB * sizeof(DecodeSampleResult), cudaMemcpyDeviceToHost, stream));
}

void decode_update_counts(const Tensor& bindings, const Tensor& tokens, int B, int T, int V, cudaStream_t stream) {
    auto* data = reinterpret_cast<const DecodeCacheBinding*>(bindings.Data);
    clear_counts<<<dim3((V + Threads - 1) / Threads, B), Threads, 0, stream>>>(data, V);
    add_counts<<<dim3((T + Threads - 1) / Threads, B), Threads, 0, stream>>>(data, tokens.get<int>(), T);
    CUDA_CHECK(cudaGetLastError());
}
