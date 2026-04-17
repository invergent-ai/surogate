// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILS_STRIDED_ITER_CUH
#define SUROGATE_SRC_UTILS_STRIDED_ITER_CUH

#include <tuple>

//! \file strided_iter.cuh
//! \brief Utilities for implementing strided iteration that avoid divisions.
//! \details Assume the following kind of iteration:
//! ```
//!  idx += stride;
//!  a = idx / S;
//!  b = idx % S;
//! ```
//! This requires a division per increment. Instead, we can use additions on both `a` and `b`, and ensure
//! that we handle overflow correctly.

template <typename Int>
class StridedIterator {
public:
    constexpr __host__ __device__ StridedIterator(Int start, Int stride, Int inner_size) {
        mOuterIdx = start / inner_size;
        mInnerIdx = start % inner_size;
        mOuterInc = stride / inner_size;
        mInnerInc = stride % inner_size;
        mInnerSize = inner_size;
    }

    constexpr __host__ __device__ void advance() {
        mOuterIdx += mOuterInc;
        mInnerIdx += mInnerInc;
        if (mInnerIdx >= mInnerSize) {
            mInnerIdx -= mInnerSize;
            ++mOuterIdx;
        }
    }

    template <std::size_t Index>
    constexpr __host__ __device__ Int get() const {
        static_assert(Index < 2, "StridedIterator has only two elements");
        if constexpr (Index == 0) return mOuterIdx;
        if constexpr (Index == 1) return mInnerIdx;
    }

private:
    Int mOuterIdx;
    Int mInnerIdx;
    Int mOuterInc;
    Int mInnerInc;
    Int mInnerSize;
};

namespace std {
template <typename Int>
struct tuple_size<::StridedIterator<Int>> {
    static constexpr std::size_t value = 2;
};

template <size_t Index, typename Int>
struct tuple_element<Index, ::StridedIterator<Int>> {
    static_assert(Index < 2, "StridedIterator has only two elements");
    using type = Int;
};
}  // namespace std

#endif  //SUROGATE_SRC_UTILS_STRIDED_ITER_CUH
