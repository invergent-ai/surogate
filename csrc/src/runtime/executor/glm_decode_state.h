// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "utilities/allocator.h"
#include "runtime/executor/paged_decode_cache.h"

namespace dsl {
// Operator-derived cache geometry. A zero divisor denotes fixed recurrent
// state; otherwise rows grow as ceil(tokens / row_divisor).
struct DecodeCacheSpec {
    int layer;
    std::string name;
    ETensorDType dtype;
    std::vector<long> shape;
    int row_divisor = 0;
    int window = 0;
    std::int64_t bytes(int tokens, int limit) const {
        long elements = 1;
        for (long d : shape)
            elements *= d;
        if (!row_divisor) return elements * get_dtype_size(dtype);
        const long rows = (tokens + row_divisor - 1) / row_divisor;
        return ((rows + DecodePageRows - 1) / DecodePageRows) * DecodePageRows * elements * get_dtype_size(dtype) +
               ((limit + DecodePageRows - 1) / DecodePageRows) * sizeof(void*);
    }
};
// Request-local history, owned by the model independently of graph recompilation.
// It contains activations only; weights and adapters remain in the trainer.
struct GlmDecodeState {
    bool active = false;
    int length = 0;
    int capacity = 0;
    int limit = 0;
    TensorAllocator allocator;
    std::unordered_map<std::string, Tensor> buffers;
    std::shared_ptr<DecodePagePool> page_pool = std::make_shared<DecodePagePool>();
    std::unordered_map<std::string, std::unique_ptr<PagedDecodeBuffer>> paged;
    ~GlmDecodeState() {
        page_pool->auxiliary_bytes -= allocator.total_allocation();
    }

    std::int64_t storage_bytes() const {
        std::int64_t total = allocator.total_allocation();
        for (const auto& [key, value] : paged)
            if (value) total += value->bytes() + value->table_bytes();
        return total;
    }
    void prepare(const std::vector<DecodeCacheSpec>& specs, int tokens, cudaStream_t stream) {
        reserve(tokens);
        for (const auto& spec : specs) {
            if (spec.row_divisor) {
                auto& buffer = pages(spec.layer, spec.name.c_str(), spec.dtype, spec.shape.at(0));
                const int rows = (tokens + spec.row_divisor - 1) / spec.row_divisor;
                buffer.reserve(rows, stream);
                buffer.make_writable(length / spec.row_divisor, rows, stream);
            } else
                get(spec.layer, spec.name.c_str(), spec.dtype, spec.shape);
        }
    }
    void retire(const std::vector<DecodeCacheSpec>& specs) {
        for (const auto& spec : specs)
            if (spec.window > 0)
                paged.at(std::to_string(spec.layer) + "/" + spec.name)->retire_before(length - spec.window + 1);
    }
    std::shared_ptr<GlmDecodeState> fork(cudaStream_t stream) const {
        auto result = std::make_shared<GlmDecodeState>();
        result->page_pool = page_pool;
        result->active = active;
        result->length = length;
        result->capacity = capacity;
        result->limit = limit;
        for (const auto& [key, buffer] : paged)
            result->paged.emplace(key, buffer->fork(stream));
        for (const auto& [key, tensor] : buffers) {
            const auto split = key.find('/');
            const auto copy = result->get(std::stoi(key.substr(0, split)),
                                          key.substr(split + 1).c_str(),
                                          tensor.DType,
                                          {tensor.Sizes.begin(), tensor.Sizes.begin() + tensor.Rank});
            CUDA_CHECK(cudaMemcpyAsync(copy.Data, tensor.Data, tensor.bytes(), cudaMemcpyDeviceToDevice, stream));
        }
        return result;
    }

    PagedDecodeBuffer& pages(int layer, const char* name, ETensorDType dtype, long width) {
        auto key = std::to_string(layer) + "/" + name;
        auto& buffer = paged[key];
        if (!buffer) buffer = std::make_unique<PagedDecodeBuffer>(page_pool, dtype, width, limit);
        if (!buffer->matches(dtype, width)) throw std::runtime_error("Decode page geometry changed");
        return *buffer;
    }

    void reserve(int required) {
        if (required < 0 || required > limit) throw std::invalid_argument("Decode exceeds maximum context length");
        if (required > capacity)
            capacity = static_cast<int>(std::min<long>(limit, std::max<long>({128, required, 2L * capacity})));
    }

    std::unordered_map<std::string, std::int64_t> stats() const {
        std::unordered_map<std::string, std::int64_t> result{{"length", length},
                                                             {"capacity", capacity},
                                                             {"limit", limit}};
        for (const auto& [key, tensor] : buffers) {
            result["bytes"] += tensor.bytes();
            result[key.substr(key.find('/') + 1) + "_bytes"] += tensor.bytes();
        }
        for (const auto& [key, buffer] : paged) {
            if (!buffer) continue;
            result["bytes"] += buffer->bytes();
            result["pages"] += buffer->pages();
            result[key.substr(key.find('/') + 1) + "_bytes"] += buffer->bytes();
        }
        result["page_size"] = DecodePageRows;
        return result;
    }

    Tensor get(int layer, const char* name, ETensorDType dtype, const std::vector<long>& shape) {
        auto key = std::to_string(layer) + "/" + name;
        auto& tensor = buffers[key];
        auto allocate = [&]() {
            long bytes = get_dtype_size(dtype);
            for (long d : shape)
                bytes *= d;
            page_pool->charge_auxiliary(bytes);
            try {
                return allocator.allocate(dtype, key.c_str(), EAllocationType::ON_DEVICE, shape);
            } catch (...) {
                page_pool->auxiliary_bytes -= bytes;
                throw;
            }
        };
        if (!tensor.Data) tensor = allocate();
        // Decode has one request, so growing the time axis preserves a
        // contiguous prefix. Recurrent states have fixed shapes and bypass it.
        if (tensor.DType == dtype && tensor.Rank == shape.size() && shape.size() >= 2 && shape[0] == 1 &&
            tensor.Sizes[0] == 1 && shape[1] > tensor.Sizes[1] &&
            std::equal(shape.begin() + 2, shape.end(), tensor.Sizes.begin() + 2)) {
            auto grown = allocate();
            CUDA_CHECK(cudaMemcpy(grown.Data, tensor.Data, tensor.bytes(), cudaMemcpyDeviceToDevice));
            page_pool->auxiliary_bytes -= tensor.bytes();
            allocator.free(tensor);
            tensor = grown;
        }
        if (tensor.DType != dtype || tensor.Rank != shape.size() ||
            !std::equal(shape.begin(), shape.end(), tensor.Sizes.begin()))
            throw std::runtime_error("GLM decode cache geometry changed; reset the session");
        return tensor;
    }
};
}  // namespace dsl
