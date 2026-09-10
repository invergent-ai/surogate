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
        if (!tensor.Data) tensor = allocator.allocate(dtype, key.c_str(), EAllocationType::ON_DEVICE, shape);
        // Decode has one request, so growing the time axis preserves a
        // contiguous prefix. Recurrent states have fixed shapes and bypass it.
        if (tensor.DType == dtype && tensor.Rank == shape.size() && shape.size() >= 2 && shape[0] == 1 &&
            tensor.Sizes[0] == 1 && shape[1] > tensor.Sizes[1] &&
            std::equal(shape.begin() + 2, shape.end(), tensor.Sizes.begin() + 2)) {
            auto grown = allocator.allocate(dtype, key.c_str(), EAllocationType::ON_DEVICE, shape);
            CUDA_CHECK(cudaMemcpy(grown.Data, tensor.Data, tensor.bytes(), cudaMemcpyDeviceToDevice));
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
