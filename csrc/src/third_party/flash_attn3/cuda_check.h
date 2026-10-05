/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdlib.h>

#include <stdexcept>
#include <string>

#include <cutlass/cutlass.h>

#define CHECK_CUDA(call)                        \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            throw std::runtime_error(std::string("flash_attn3: CUDA error: ") +                           \
                                     cudaGetErrorString(status_));                                        \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

#define CHECK_CUTLASS(call)                                                                               \
    do {                                                                                                  \
        cutlass::Status status_ = (call);                                                                 \
        if (status_ != cutlass::Status::kSuccess) {                                                        \
            throw std::runtime_error(std::string("flash_attn3: CUTLASS error: ") +                        \
                                     cutlass::cutlassGetStatusString(status_));                           \
        }                                                                                                 \
    } while(0)
