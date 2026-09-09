// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "utilities/allocator.h"

namespace dsl {
// Request-local history, owned by the model independently of graph recompilation.
// It contains activations only; weights and adapters remain in the trainer.
struct GlmDecodeState {
    bool active = false;
    int length = 0;
    int capacity = 0;
    TensorAllocator allocator;
    std::unordered_map<std::string, Tensor> buffers;

    Tensor get(int layer, const char* name, ETensorDType dtype, const std::vector<long>& shape) {
        auto key = std::to_string(layer) + "/" + name;
        auto& tensor = buffers[key];
        if (!tensor.Data) tensor = allocator.allocate(dtype, key.c_str(), EAllocationType::ON_DEVICE, shape);
        if (tensor.DType != dtype || tensor.Rank != shape.size() ||
            !std::equal(shape.begin(), shape.end(), tensor.Sizes.begin()))
            throw std::runtime_error("GLM decode cache geometry changed; reset the session");
        return tensor;
    }
};
}  // namespace dsl
