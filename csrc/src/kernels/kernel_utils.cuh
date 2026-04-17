// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file kernel_utils.cuh
 * @brief Common CUDA device utility functions for reductions and atomic operations.
 *
 * Provides helper functions for warp-level and block-level reductions using
 * shuffle intrinsics and cooperative groups. Used across multiple kernels
 * for sum, max, and absolute max computations.
 */

#ifndef SUROGATE_SRC_KERNELS_KERNEL_UTILS_CUH
#define SUROGATE_SRC_KERNELS_KERNEL_UTILS_CUH

#include <cassert>
#include <cooperative_groups.h>

#ifndef __HIP__
#include <cooperative_groups/reduce.h>
#endif

/**
 * @brief Performs block-wide and global reduction to find maximum absolute value.
 *
 * Uses warp-level intrinsics and atomic operations to efficiently compute
 * the maximum absolute value across all threads in a block, then updates
 * a global maximum. Requires all threads in each warp to be active.
 *
 * @param[in,out] abs_max_ptr Global pointer to accumulate absolute maximum (may be NULL to skip).
 * @param[in,out] block_max Shared memory for block-level reduction (must be pre-allocated).
 * @param thread_max Thread's local maximum value to reduce.
 *
 * @note Requires warp convergence (all 32 threads active) for correctness.
 * @note Uses __float_as_uint for bitwise max comparison of positive floats.
 */
static __forceinline__ __device__ void
handle_absmax_reduction(float* __restrict__ abs_max_ptr, float* __restrict__ block_max, float thread_max) {
    if (abs_max_ptr) {
        auto warp_reduce_max_u32 = [](unsigned v) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                unsigned other = __shfl_xor_sync(0xFFFFFFFFu, v, offset);
                v = (v > other) ? v : other;
            }
            return v;
        };

        // this code is only guaranteed to be correct if it is warp convergent
        // (in theory, ensuring thread 0 hasn't exited would be enough...)
        assert(__activemask() == 0xffffffff);
        unsigned warp_max = warp_reduce_max_u32(__float_as_uint(thread_max));
        if (threadIdx.x % 32 == 0) {
            atomicMax(reinterpret_cast<unsigned*>(block_max), warp_max);
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            atomicMax(reinterpret_cast<unsigned int*>(abs_max_ptr), __float_as_uint(*block_max));
        }
    }
}

/**
 * @brief Performs cooperative group reduction with addition.
 *
 * Wrapper around cooperative_groups::reduce for summing values across
 * a thread group (warp, block, or custom partition).
 *
 * @tparam Group Cooperative group type (e.g., thread_block_tile<32>).
 * @tparam Element Numeric type to reduce (typically float).
 * @param group Reference to the cooperative group.
 * @param value Thread's input value to sum.
 * @return Sum of all values across the group (same on all threads).
 */
template <typename Group, typename Element>
static __forceinline__ __device__ Element reduce_group_add(Group& group, Element value) {
    return cooperative_groups::reduce(group, value, cooperative_groups::plus<Element>());
}

/**
 * @brief Performs cooperative group reduction to find maximum.
 *
 * Wrapper around cooperative_groups::reduce for finding the maximum value
 * across a thread group (warp, block, or custom partition).
 *
 * @tparam Group Cooperative group type (e.g., thread_block_tile<32>).
 * @tparam Element Numeric type to reduce (typically float).
 * @param group Reference to the cooperative group.
 * @param value Thread's input value.
 * @return Maximum of all values across the group (same on all threads).
 */
template <typename Group, typename Element>
static __forceinline__ __device__ Element reduce_group_max(Group& group, Element value) {
    return cooperative_groups::reduce(group, value, cooperative_groups::greater<Element>());
}

/**
 * @brief Warp-level sum reduction using shuffle intrinsics.
 *
 * Performs a butterfly reduction pattern using __shfl_xor_sync to sum
 * values across all 32 threads in a warp. More efficient than shared
 * memory for warp-local reductions.
 *
 * @param val Thread's input value to sum.
 * @return Sum of all 32 warp threads' values (same on all threads).
 *
 * @note Requires all 32 warp threads to be active.
 */
static __forceinline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

/**
 * @brief Warp-level max reduction using shuffle intrinsics.
 *
 * Performs a butterfly reduction pattern using __shfl_xor_sync to find
 * the maximum value across all 32 threads in a warp. Uses fmaxf for
 * proper floating-point max semantics (handles NaN correctly).
 *
 * @param val Thread's input value.
 * @return Maximum of all 32 warp threads' values (same on all threads).
 *
 * @note Requires all 32 warp threads to be active.
 */
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFFu, val, offset));
    }
    return val;
}

#endif  //SUROGATE_SRC_KERNELS_KERNEL_UTILS_CUH
