// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILS_TENSOR_CONTAINER_H
#define SUROGATE_SRC_UTILS_TENSOR_CONTAINER_H

#include <functional>
#include <string>

#include <nlohmann/json_fwd.hpp>

class TensorShard;

class ITensorContainer {
public:
    virtual void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) = 0;

protected:
    ~ITensorContainer() = default;
};

#endif  //SUROGATE_SRC_UTILS_TENSOR_CONTAINER_H
