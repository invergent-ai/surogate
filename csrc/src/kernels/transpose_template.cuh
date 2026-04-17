// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file transpose_template.cuh
 * @brief Template helper for fused apply-and-transpose operations.
 *
 * Provides a reusable device function that applies a transformation
 * (e.g., quantization) while simultaneously transposing a matrix.
 * Used by quantization kernels to fuse type conversion with transpose.
 */

#ifndef SUROGATE_SRC_KERNELS_TRANSPOSE_TEMPLATE_CUH
#define SUROGATE_SRC_KERNELS_TRANSPOSE_TEMPLATE_CUH

#include "utilities/vec.cuh"

/**
 * @brief Device helper that applies an operation and transposes in one pass.
 *
 * Each thread block tile loads a VEC_SIZE x VEC_SIZE block of elements,
 * applies the transformation function to each element, and writes the
 * transposed result. This fuses the conversion and transpose operations
 * to reduce memory traffic.
 *
 * Memory layout:
 * - Input: row-major (rows x cols), read as cache[row][col]
 * - Output: transposed (cols x rows), written as dest[col][row]
 *
 * @tparam VEC_SIZE Tile dimension (elements per thread in each direction).
 * @tparam F Transformation functor type (e.g., lambda for quantization).
 * @tparam TD Destination element type.
 * @tparam TS Source element type.
 * @param op Transformation functor: TD = op(TS).
 * @param[out] dest Destination array of shape (cols, rows) in row-major.
 * @param[in] src Source array of shape (rows, cols) in row-major.
 * @param rows Number of rows in source (columns in destination).
 * @param cols Number of columns in source (rows in destination).
 */
template <int VEC_SIZE, typename F, typename TD, class TS>
__device__ void apply_and_transpose_helper(F&& op, TD* dest, const TS* src, int rows, int cols) {
    long r = VEC_SIZE * (blockIdx.x * blockDim.x + threadIdx.x);
    long c = VEC_SIZE * (blockIdx.y * blockDim.y + threadIdx.y);
    if (c >= cols || r >= rows) {
        return;
    }

    using src_vec_t = GenericVector<TS, VEC_SIZE>;
    src_vec_t cache[VEC_SIZE];
    for (int i = 0; i < VEC_SIZE; i++) {
        cache[i] = src_vec_t::load(src + c + (i + r) * cols);
    }
    using dst_vec_t = GenericVector<TD, VEC_SIZE>;
    dst_vec_t save[VEC_SIZE];
    for (int i = 0; i < VEC_SIZE; i++) {
        for (int j = 0; j < VEC_SIZE; j++) {
            save[i][j] = TD{op(cache[j][i])};
        }
    }

    for (int j = 0; j < VEC_SIZE; j++) {
        save[j].store(dest + (c + j) * rows + r);
    }
}

#endif  //SUROGATE_SRC_KERNELS_TRANSPOSE_TEMPLATE_CUH
