// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILS_VEC_CUH
#define SUROGATE_SRC_UTILS_VEC_CUH

#include <cstring>
#include <type_traits>

#include <vector_types.h>

namespace detail {
/*! \brief Memory transfer mode used by vectorized copy helpers.
 *  \details Selects which load instruction is used on device. `DEFAULT` is portable
 *  (host/device). Other modes are device-only and may have architecture-specific behavior.
 */
enum class TransferMode {
    /*! \brief Plain dereference load/store (`*dst = *src`). Valid on host and device. */
    DEFAULT,
    /*! \brief Read-only cache load via `__ldg`. Device-only. */
    LDG,
    /*! \brief Load via `__ldlu`. Device-only. */
    LU,
    LOAD_CS,
    STORE_CS,
    STORE_CG
};

/*! \brief Primary template for transfer operation dispatch. */
template <TransferMode Mode>
struct Transfer;

template <>
struct Transfer<TransferMode::DEFAULT> {
    template <class T>
    __host__ __device__ static void call(T* dst, const T* src) {
        *dst = *src;
    }
};

template <>
struct Transfer<TransferMode::LDG> {
    template <class T>
    __device__ static void call(T* dst, const T* src) {
        *dst = __ldg(src);
    }
};

template <>
struct Transfer<TransferMode::LU> {
    template <class T>
    __device__ static void call(T* dst, const T* src) {
        *dst = __ldlu(src);
    }
};

template <>
struct Transfer<TransferMode::LOAD_CS> {
    template <class T>
    __device__ static void call(T* dst, const T* src) {
        *dst = __ldcs(src);
    }
};

template <>
struct Transfer<TransferMode::STORE_CG> {
    template <class T>
    __device__ static void call(T* dst, const T* src) {
        __stcg(dst, *src);
    }
};

template <>
struct Transfer<TransferMode::STORE_CS> {
    template <class T>
    __device__ static void call(T* dst, const T* src) {
        __stcs(dst, *src);
    }
};

/*!
 * \brief Copies `NBytes` from `src` to `dst`, using `CopyType` to perform memory access.
 * \details
 * This means that pointers need to be aligned according to `CopyType`'s requirements,
 * and copies are (most likely) be performed using vectorized access according to
 * `CopyType`.
 * The ranges `[src, src+NBytes)` and `[dst, dst + NBytes)` must be non-overlapping.
 *
 * This function is used to implement `memcpy_aligned`, and generally not intended to
 * be used directly.
 *
 *  \tparam CopyType Vector type used for each access (e.g., `int4`).
 *  \tparam NBytes Total number of bytes to copy (must be compile-time constant).
 *  \tparam Mode Load mode used by `Transfer<Mode>`.
 *  \tparam TrueType Underlying element type of the source/destination.
 *
 *  \param[out] dst Destination pointer (must be aligned for `CopyType`).
 *  \param[in]  src Source pointer (must be aligned for `CopyType`).
 *
 *  \note Source and destination ranges must be non-overlapping.
 *  \note `TrueType` must be trivially copyable.
 */
template <class CopyType, int NBytes, TransferMode Mode, class TrueType>
__host__ __device__ void memcpy_as(TrueType* __restrict__ dst, const TrueType* __restrict__ src) {
    static_assert(NBytes % sizeof(TrueType) == 0, "Number of bytes must be a multiple of the true type size");
    static_assert(NBytes % sizeof(CopyType) == 0, "Number of bytes must be a multiple of the copy type size");

    // in order to do simple byte-level copying, the underlying type must be trivially copyable (i.e., compatible
    // with memcpy)
    static_assert(std::is_trivially_copyable_v<TrueType>, "TrueType must be trivially copyable");
    const auto* read_address = reinterpret_cast<const CopyType*>(src);
    auto* write_address = reinterpret_cast<CopyType*>(dst);
#pragma unroll
    for (int i = 0; i < NBytes; i += sizeof(CopyType)) {
        Transfer<Mode>::call(write_address, read_address);
        ++read_address;
        ++write_address;
    }
}

/*! \brief Compute worst-case element alignment for a packed array of fixed-size elements.
 *  \details
 *  Assumes the array base address is 16-byte aligned. Returns the minimum guaranteed
 *  alignment of any element given `size` bytes per element.
 *
 *  \param size Size in bytes of one element.
 *  \return Guaranteed alignment in bytes (power of two, up to 16).
 */
constexpr __host__ __device__ std::size_t alignment_from_size(std::size_t size) {
    for (int i = 2; i <= 16; i *= 2) {
        if ((size % i) != 0) {
            return i / 2;
        }
    }
    return 16;
}
}  // namespace detail

/*!
 * \brief Synchronous copy from `src` to `dst` using the widest vectorized accesses possible.
 *  \details
 *  Chooses the access width based on the *total* number of bytes (`sizeof(T) * Count`),
 *  not on per-element alignment. Intended to encourage vectorized loads/stores.
 *
 *  \tparam Count Number of elements to copy (compile-time constant).
 *  \tparam Mode Transfer mode used for loads (see `detail::TransferMode`).
 *  \tparam T Element type; must be trivially copyable.
 *
 *  \param[out] dst Destination pointer (must be suitably aligned for chosen access width).
 *  \param[in]  src Source pointer (must be suitably aligned for chosen access width).
 *  \param      (unused) Compile-time tag used to pass `Count` explicitly when needed.
 */
template <std::size_t Count, detail::TransferMode Mode, class T>
__host__ __device__ void memcpy_aligned(T* dst, const T* src, std::integral_constant<std::size_t, Count> = {}) {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    constexpr const int NBytes = sizeof(T) * Count;
    using detail::memcpy_as;

    // ideally, we'd just use a simple memcpy, like below, but that does
    // not always generate vectorized loads
    // std::memcpy(values, __builtin_assume_aligned(address, bytes), bytes);

    if constexpr (NBytes % sizeof(int4) == 0) {
        memcpy_as<int4, NBytes, Mode>(dst, src);
    } else if constexpr (NBytes % sizeof(int2) == 0) {
        memcpy_as<int2, NBytes, Mode>(dst, src);
    } else if constexpr (NBytes % sizeof(int1) == 0) {
        memcpy_as<int1, NBytes, Mode>(dst, src);
    } else if constexpr (NBytes % sizeof(short1) == 0) {
        memcpy_as<short1, NBytes, Mode>(dst, src);
    } else {
        memcpy_as<char1, NBytes, Mode>(dst, src);
    }
}

/*! \brief SIMD-like fixed-size vector wrapper with natural indexing.
 *  \details
 *  Provides a trivially-copyable storage array with an alignment chosen to match the
 *  worst-case element alignment for `sizeof(ElementType) * ElementCount`.
 *
 *  \tparam ElementType Scalar element type stored in the vector (must be trivial).
 *  \tparam ElementCount Number of elements stored.
 */
template <class ElementType, std::size_t ElementCount>
class alignas(detail::alignment_from_size(sizeof(ElementType) * ElementCount)) GenericVector {
    static_assert(std::is_trivial_v<ElementType>, "Only trivial types are supported");

public:
    /*! \brief Default constructor. Leaves elements uninitialized. */
    GenericVector() = default;

    /*! \brief Construct a vector with all lanes set to the same value.
     *  \param value Value to broadcast into all elements.
     *  \return A vector where every element equals \p value.
     */
    constexpr static __host__ __device__ GenericVector constant(ElementType value) {
        GenericVector result;
        for (int k = 0; k < size; ++k) {
            result.values[k] = value;
        }
        return result;
    }

    /*! \brief Construct a vector with all lanes set to zero (cast to `ElementType`). */
    constexpr static __host__ __device__ GenericVector zeros() {
        return constant(static_cast<ElementType>(0.f));
    }

    /*! \brief Construct a vector with all lanes set to one (cast to `ElementType`). */
    constexpr static __host__ __device__ GenericVector ones() {
        return constant(1.f);
    }

    /*! \brief Convert from another `GenericVector` with the same element count.
     *  \tparam U Source element type.
     *  \param other Source vector.
     *  \return New vector with each lane converted via `static_cast<ElementType>`.
     */
    template <class U>
    constexpr static __host__ __device__ GenericVector from(GenericVector<U, ElementCount> other) {
        GenericVector<ElementType, ElementCount> result;
        for (int i = 0; i < ElementCount; ++i) {
            result[i] = static_cast<ElementType>(other[i]);
        }
        return result;
    }

    /*! \brief Mutable element access.
     *  \param index Element index in `[0, size)`.
     *  \return Reference to the selected element.
     */
    constexpr __host__ __device__ ElementType& operator[](int index) {
        return values[index];
    }

    /*! \brief Immutable element access.
     *  \param index Element index in `[0, size)`.
     *  \return Const reference to the selected element.
     */
    constexpr __host__ __device__ const ElementType& operator[](int index) const {
        return values[index];
    }

    /*! \brief Number of elements in this vector. */
    static constexpr const std::size_t size = ElementCount;
    /*! \brief Total byte size of this vector (`size * sizeof(ElementType)`). */
    static constexpr const std::size_t bytes = ElementCount * sizeof(ElementType);

    /*! \brief Load a vector from memory using default transfer semantics.
     *  \param address Pointer to `size` contiguous `ElementType` values.
     *  \return Loaded vector.
     *  \note Caller must ensure alignment suitable for the chosen vectorized access.
     */
    static __host__ __device__ GenericVector load(const ElementType* address) {
        GenericVector result;
        memcpy_aligned<size, detail::TransferMode::DEFAULT>(result.values, address);
        return result;
    }

    /*! \brief Load a vector from memory using `__ldg` (device-only).
     *  \param address Pointer to `size` contiguous `ElementType` values.
     *  \return Loaded vector.
     *  \note Intended for read-only data paths.
     */
    static __device__ GenericVector load_ldg(const ElementType* address) {
        GenericVector result;
        memcpy_aligned<size, detail::TransferMode::LDG>(result.values, address);
        return result;
    }

    /*! \brief Load a vector from memory using `__ldlu` (device-only).
     *  \param address Pointer to `size` contiguous `ElementType` values.
     *  \return Loaded vector.
     */
    static __device__ GenericVector load_lu(const ElementType* address) {
        GenericVector result;
        memcpy_aligned<size, detail::TransferMode::LU>(result.values, address);
        return result;
    }

    /*! \brief Load a vector from memory using `__ldcs` (device-only).
     *  \param address Pointer to `size` contiguous `ElementType` values.
     *  \return Loaded vector.
     */
    static __device__ GenericVector load_cs(const ElementType* address) {
        GenericVector result;
        memcpy_aligned<size, detail::TransferMode::LOAD_CS>(result.values, address);
        return result;
    }

    /*! \brief Store this vector to memory using default transfer semantics.
     *  \param[out] dst Destination pointer to `size` contiguous `ElementType` values.
     *  \note Caller must ensure alignment suitable for the chosen vectorized access.
     */
    __host__ __device__ void store(ElementType* dst) {
        memcpy_aligned<size, detail::TransferMode::DEFAULT>(dst, values);
    }

    __host__ __device__ void store_cg(ElementType* dst) {
        memcpy_aligned<size, detail::TransferMode::STORE_CG>(dst, values);
    }

    __host__ __device__ void store_cs(ElementType* dst) {
        memcpy_aligned<size, detail::TransferMode::STORE_CS>(dst, values);
    }

private:
    ElementType values[size];
};

#endif  // SUROGATE_SRC_UTILS_VEC_CUH
