// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// Adapted for Surogate by the Surogate team
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>
#include <type_traits>
#include <vector_types.h>

namespace quartet {

namespace detail {

enum class TransferMode {
    DEFAULT,
    LDG,
    LU,
    LOAD_CS,
    STORE_CS,
    STORE_CG
};

template<TransferMode Mode>
struct Transfer;

template<>
struct Transfer<TransferMode::DEFAULT> {
    template<class T>
    __host__ __device__ static void call(T* dst, const T* src) {
        *dst = *src;
    }
};

template<>
struct Transfer<TransferMode::LDG> {
    template<class T>
    __device__ static void call(T* dst, const T* src) {
        *dst = __ldg(src);
    }
};

template<>
struct Transfer<TransferMode::LU> {
    template<class T>
    __device__ static void call(T* dst, const T* src) {
        *dst = __ldlu(src);
    }
};

template<>
struct Transfer<TransferMode::LOAD_CS> {
    template<class T>
    __device__ static void call(T* dst, const T* src) {
        *dst = __ldcs(src);
    }
};

template<>
struct Transfer<TransferMode::STORE_CG> {
    template<class T>
    __device__ static void call(T* dst, const T* src) {
        __stcg(dst, *src);
    }
};

template<>
struct Transfer<TransferMode::STORE_CS> {
    template<class T>
    __device__ static void call(T* dst, const T* src) {
        __stcs(dst, *src);
    }
};

template<class CopyType, int NBytes, TransferMode Mode, class TrueType>
__host__ __device__ void memcpy_as(TrueType* __restrict__ dst, const TrueType* __restrict__ src) {
    static_assert(NBytes % sizeof(TrueType) == 0, "Number of bytes must be a multiple of the true type size");
    static_assert(NBytes % sizeof(CopyType) == 0, "Number of bytes must be a multiple of the copy type size");
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

constexpr __host__ __device__ std::size_t alignment_from_size(std::size_t size) {
    for (int i = 2; i <= 16; i *= 2) {
        if ((size % i) != 0) {
            return i / 2;
        }
    }
    return 16;
}

}  // namespace detail

template<std::size_t Count, detail::TransferMode Mode, class T>
__host__ __device__ void memcpy_aligned(T* dst, const T* src, std::integral_constant<std::size_t, Count> = {}) {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    constexpr const int NBytes = sizeof(T) * Count;
    using detail::memcpy_as;

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

template<class ElementType, std::size_t ElementCount>
class alignas(detail::alignment_from_size(sizeof(ElementType) * ElementCount)) GenericVector {
    static_assert(std::is_trivial_v<ElementType>, "Only trivial types are supported");

public:
    GenericVector() = default;

    constexpr static __host__ __device__ GenericVector constant(ElementType value) {
        GenericVector result;
        for (int k = 0; k < size; ++k) {
            result.values[k] = value;
        }
        return result;
    }

    constexpr static __host__ __device__ GenericVector zeros() {
        return constant(static_cast<ElementType>(0.f));
    }

    constexpr static __host__ __device__ GenericVector ones() {
        return constant(1.f);
    }

    template<class U>
    constexpr static __host__ __device__ GenericVector from(GenericVector<U, ElementCount> other) {
        GenericVector<ElementType, ElementCount> result;
        for (int i = 0; i < ElementCount; ++i) {
            result[i] = static_cast<ElementType>(other[i]);
        }
        return result;
    }

    constexpr __host__ __device__ ElementType& operator[](int index) {
        return values[index];
    }

    constexpr __host__ __device__ const ElementType& operator[](int index) const {
        return values[index];
    }

    static constexpr const std::size_t size = ElementCount;
    static constexpr const std::size_t bytes = ElementCount * sizeof(ElementType);

    static __host__ __device__ GenericVector load(const ElementType* address) {
        GenericVector result;
        memcpy_aligned<size, detail::TransferMode::DEFAULT>(result.values, address);
        return result;
    }

    static __device__ GenericVector load_ldg(const ElementType* address) {
        GenericVector result;
        memcpy_aligned<size, detail::TransferMode::LDG>(result.values, address);
        return result;
    }

    static __device__ GenericVector load_lu(const ElementType* address) {
        GenericVector result;
        memcpy_aligned<size, detail::TransferMode::LU>(result.values, address);
        return result;
    }

    static __device__ GenericVector load_cs(const ElementType* address) {
        GenericVector result;
        memcpy_aligned<size, detail::TransferMode::LOAD_CS>(result.values, address);
        return result;
    }

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

}  // namespace quartet
