// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILITIES_SOL_H
#define SUROGATE_SRC_UTILITIES_SOL_H

#include <utility>
#include <vector>

enum class ETensorDType : int;

// Estimate how many ops of each type a transformer will need
std::vector<std::pair<ETensorDType, long>> get_transformer_ops(long non_embedding_params,
                                                               ETensorDType non_embedding_dtype,
                                                               long embedding_params,
                                                               ETensorDType embedding_dtype,
                                                               long d_att,
                                                               long n_layers,
                                                               long ctx);

// For a given list of operations, how many microseconds would it take the device if it ran at
// peak speed
long estimate_speed_of_light(const char* device, const std::vector<std::pair<ETensorDType, long>>& ops);

float get_peak_rate(const char* device, ETensorDType dtype);

// Runs a large matmul to measure device peak flops in bf16
double measure_real_peak();

// Runs a large matmul to measure device peak flops in fp8
double measure_real_peak_fp8();

#endif  // SUROGATE_SRC_UTILITIES_SOL_H
