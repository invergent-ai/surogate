// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// Adapted for Surogate by the Surogate team
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector_types.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>
#include <stdexcept>

#include "vec.cuh"

namespace quartet {

// MMA fragment types for 16x16 matrix operations
template<typename T>
struct m16_n16_a_fragment {
    uint4 v;
};

template<typename T>
struct m16_n16_b_fragment {
    uint4 v;
};

template<typename AccDType>
struct m16_n16_k32_c_fragment {
    AccDType v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
};

// PTX type name mapping
template<typename>
constexpr char ptx_type_name[] = "unknown_dtype";

template<>
constexpr char ptx_type_name<float>[4] = "f32";

template<>
constexpr char ptx_type_name<half>[4] = "f16";

template<>
constexpr char ptx_type_name<nv_bfloat16>[5] = "bf16";

template<>
constexpr char ptx_type_name<__nv_fp8_e4m3>[5] = "e4m3";

template<>
constexpr char ptx_type_name<__nv_fp8_e5m2>[5] = "e5m2";

// Fragment load functions
__device__ __forceinline__ m16_n16_a_fragment<nv_bfloat16> load_fragment_a(int lane_id, const nv_bfloat16* base, int ldd) {
    m16_n16_a_fragment<nv_bfloat16> result;
    int g = lane_id / 4;
    int l = lane_id % 4;

    result.v.x = *reinterpret_cast<const uint*>(base + l * 2 + 0 + (g+0) * ldd);
    result.v.y = *reinterpret_cast<const uint*>(base + l * 2 + 0 + (g+8) * ldd);
    result.v.z = *reinterpret_cast<const uint*>(base + l * 2 + 8 + (g+0) * ldd);
    result.v.w = *reinterpret_cast<const uint*>(base + l * 2 + 8 + (g+8) * ldd);
    return result;
}

__device__ __forceinline__ m16_n16_b_fragment<nv_bfloat16> load_fragment_b(int lane_id, const nv_bfloat16* base, int ldd) {
    m16_n16_b_fragment<nv_bfloat16> result;
    int g = lane_id / 4;
    int l = lane_id % 4;

    result.v.x = *reinterpret_cast<const uint*>(base + l * 2 + 0 + (g+0) * ldd);
    result.v.y = *reinterpret_cast<const uint*>(base + l * 2 + 0 + (g+8) * ldd);
    result.v.z = *reinterpret_cast<const uint*>(base + l * 2 + 8 + (g+0) * ldd);
    result.v.w = *reinterpret_cast<const uint*>(base + l * 2 + 8 + (g+8) * ldd);
    return result;
}

__device__ __forceinline__ m16_n16_b_fragment<nv_bfloat16> load_fragment_b_smem(int lane_id, const nv_bfloat16* base, int ldd) {
    m16_n16_b_fragment<nv_bfloat16> result;
    int l16 = lane_id % 16;
    int g2 = lane_id / 16;
    const nv_bfloat16* src = base + l16 * ldd + 8 * g2;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(result.v.x), "=r"(result.v.y), "=r"(result.v.z), "=r"(result.v.w)
        : "l"(__cvta_generic_to_shared(src))
        );
    return result;
}

__device__ __forceinline__ m16_n16_a_fragment<nv_bfloat16> load_fragment_a_smem(int lane_id, const nv_bfloat16* base, int ldd) {
    m16_n16_a_fragment<nv_bfloat16> result;
    int l16 = lane_id % 16;
    int g2 = lane_id / 16;
    const nv_bfloat16* src = base + l16 * ldd + 8 * g2;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(result.v.x), "=r"(result.v.y), "=r"(result.v.z), "=r"(result.v.w)
        : "l"(__cvta_generic_to_shared(src))
        );
    return result;
}

// Shared memory swizzle for bank conflict avoidance
__device__ __forceinline__ int swizzle_smem(int row, int col) {
    int lin = col + 2 * row;
    int g8 = lin / 8;
    int t8 = lin % 8;
    return ((t8 ^ g8) + 8 * g8) * 8;
}

__device__ __forceinline__ void global_to_shared_16_32_swizzle(nv_bfloat16* __restrict__ shared, const nv_bfloat16* __restrict__ global, int stride) {
    int col = threadIdx.x % 2;
    int row = threadIdx.x / 2;
    __pipeline_memcpy_async(
                shared + swizzle_smem(row, col),
                global + col * 8 + row * stride,
                16);
}

__device__ __forceinline__ void global_to_shared_swizzle(__nv_fp4x2_storage_t* __restrict__ shared, const __nv_fp4x2_storage_t* __restrict__ global, int stride) {
    int col = threadIdx.x % 2;
    int row = threadIdx.x / 2;
    __pipeline_memcpy_async(
                shared + swizzle_smem(row, col) / 2,
                global + col * 4 + row * stride / 2,
                4);
}

__device__ __forceinline__ m16_n16_b_fragment<nv_bfloat16> load_fragment_b_swizzle(int lane_id, const nv_bfloat16* base) {
    m16_n16_b_fragment<nv_bfloat16> result;
    int t8 = lane_id % 8;
    int g8 = lane_id / 8;

    const nv_bfloat16* src = base + swizzle_smem(t8 + 8 * (g8/2), g8 % 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(result.v.x), "=r"(result.v.y), "=r"(result.v.z), "=r"(result.v.w)
        : "l"(__cvta_generic_to_shared(src))
        );
    return result;
}

__device__ __forceinline__ m16_n16_a_fragment<nv_bfloat16> load_fragment_a_swizzle(int lane_id, const nv_bfloat16* base) {
    m16_n16_a_fragment<nv_bfloat16> result;
    int t8 = lane_id % 8;
    int g8 = lane_id / 8;

    const nv_bfloat16* src = base + swizzle_smem(t8 + 8 * (g8%2), g8 / 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(result.v.x), "=r"(result.v.y), "=r"(result.v.z), "=r"(result.v.w)
        : "l"(__cvta_generic_to_shared(src))
        );
    return result;
}

__device__ __forceinline__ m16_n16_a_fragment<nv_bfloat16> load_fragment_aT_swizzle(int lane_id, const nv_bfloat16* base) {
    m16_n16_a_fragment<nv_bfloat16> result;
    int t8 = lane_id % 8;
    int g8 = lane_id / 8;

    const nv_bfloat16* src = base + swizzle_smem(t8 + 8 * (g8/2), g8 % 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(result.v.x), "=r"(result.v.y), "=r"(result.v.z), "=r"(result.v.w)
        : "l"(__cvta_generic_to_shared(src))
        );
    return result;
}

__device__ __forceinline__ void ptx_ldmatrix(uint4& dst, const void* src) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w)
        : "l"(__cvta_generic_to_shared(src))
        );
}

template<typename AccDType>
__device__ __forceinline__ void store_fragment_row_major_sync(m16_n16_k32_c_fragment<AccDType>& c, nv_bfloat16* ptr, int row_stride) {
    int g4 = threadIdx.x / 4;
    int c4 = threadIdx.x % 4;
    nv_bfloat162* vptr = reinterpret_cast<nv_bfloat162*>(ptr);
    row_stride /= 2;
    vptr[row_stride * (g4 + 0) + c4 + 0] = make_bfloat162((nv_bfloat16)c.v[0], (nv_bfloat16)c.v[1]);
    vptr[row_stride * (g4 + 8) + c4 + 0] = make_bfloat162((nv_bfloat16)c.v[2], (nv_bfloat16)c.v[3]);

    vptr[row_stride * (g4 + 0) + c4 + 4] = make_bfloat162((nv_bfloat16)c.v[4], (nv_bfloat16)c.v[5]);
    vptr[row_stride * (g4 + 8) + c4 + 4] = make_bfloat162((nv_bfloat16)c.v[6], (nv_bfloat16)c.v[7]);
}

// MMA instruction wrappers
template<typename AType, typename BType>
__device__ __forceinline__ void mma_m16_n16_sync(m16_n16_k32_c_fragment<float>& d,
                                                     m16_n16_a_fragment<AType> a,
                                                     m16_n16_b_fragment<BType> b,
                                                     m16_n16_k32_c_fragment<float> c) {
    static_assert(sizeof(AType) == sizeof(BType), "a and b type must have the same size");

    constexpr int k = 32 / sizeof(AType);
    asm volatile("mma.sync.aligned.m16n8k%26.row.col.f32.%24.%25.f32 "
                 "{%0, %1, %2, %3},"
                 "{%8, %9, %10, %11},"
                 "{%12, %13},"
                 "{%16, %17, %18, %19};\n"
                 "mma.sync.aligned.m16n8k%26.row.col.f32.%24.%25.f32 "
                 "{%4, %5, %6, %7},"
                 "{%8, %9, %10, %11},"
                 "{%14, %15},"
                 "{%20, %21, %22, %23};\n"
        : "=f"(d.v[0]), "=f"(d.v[1]), "=f"(d.v[2]), "=f"(d.v[3]),
          "=f"(d.v[4]), "=f"(d.v[5]), "=f"(d.v[6]), "=f"(d.v[7])
        : "r"(a.v.x), "r"(a.v.y), "r"(a.v.z), "r"(a.v.w),
          "r"(b.v.x), "r"(b.v.y), "r"(b.v.z), "r"(b.v.w),
          "f"(c.v[0]), "f"(c.v[1]), "f"(c.v[2]), "f"(c.v[3]),
          "f"(c.v[4]), "f"(c.v[5]), "f"(c.v[6]), "f"(c.v[7]),
          "C"(ptx_type_name<AType>), "C"(ptx_type_name<BType>), "n"(k));
}

template<typename AType, typename BType>
__device__ __forceinline__ void mma_m16_n16_sync(m16_n16_k32_c_fragment<half>& d,
                                                     m16_n16_a_fragment<AType> a,
                                                     m16_n16_b_fragment<BType> b,
                                                     m16_n16_k32_c_fragment<half> c) {
    auto to_raw = [](half& h) -> unsigned int& {
        return *reinterpret_cast<unsigned int*>(&h);
    };
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.%10.%11.f16 "
                 "{%0, %1},"
                 "{%2, %3, %4, %5},"
                 "{%6, %7},"
                 "{%8, %9};"
        : "=r"(to_raw(d.v[0])), "=r"(to_raw(d.v[2]))
        : "r"(a.v.x), "r"(a.v.y), "r"(a.v.z), "r"(a.v.w),
          "r"(b.v.x), "r"(b.v.y),
          "r"(to_raw(c.v[0])), "r"(to_raw(c.v[2])),
          "C"(ptx_type_name<AType>), "C"(ptx_type_name<BType>));

    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.%10.%11.f16 "
                 "{%0, %1},"
                 "{%2, %3, %4, %5},"
                 "{%6, %7},"
                 "{%8, %9};"
        : "=r"(to_raw(d.v[4])), "=r"(to_raw(d.v[6]))
        : "r"(a.v.x), "r"(a.v.y), "r"(a.v.z), "r"(a.v.w),
          "r"(b.v.z), "r"(b.v.w),
          "r"(to_raw(c.v[4])), "r"(to_raw(c.v[6])),
          "C"(ptx_type_name<AType>), "C"(ptx_type_name<BType>));
}

// FP4 conversion helper
static __device__ __forceinline__ float2 __nv_cvt_fp4x2_to_float2(__nv_fp4x2_storage_t x) {
    __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(x, __nv_fp4_interpretation_t::__NV_E2M1);
    __half_raw r1, r2;
    r1.x = raw.x;
    r2.x = raw.y;
    return make_float2(__half2float(__half(r1)), __half2float(__half(r2)));
}

// Vector abs-max reduction
template<std::size_t Size>
static __forceinline__ __device__ nv_bfloat16 vecReduceAbsMax(GenericVector<nv_bfloat16, Size> val) {
    static_assert(Size == 4 || Size == 8, "Size must be 4 or 8");
#pragma unroll
    for (int k = 0; k < Size; k += 4) {
        nv_bfloat162 m = __hmax2(
            __habs2( make_bfloat162(val[k], val[k+1]) ),
            __habs2( make_bfloat162(val[k + 2], val[k + 3]) )
        );
        val[k/2] = m.x;
        val[k/2 + 1] = m.y;
    }

    for (std::size_t end = Size / 2; end >= 4; end /= 2) {
        for (int k = 0; k < end; k += 4) {
            nv_bfloat162 m = __hmax2(make_bfloat162(val[k], val[k+1]), make_bfloat162(val[k + 2], val[k + 3]));
            val[k/2] = m.x;
            val[k/2 + 1] = m.y;
        }
    }
    return __hmax(val[0], val[1]);
}

// Philox RNG
static __device__ __forceinline__ uint2 mul_hilo(std::uint32_t a, std::uint32_t b) {
    std::uint64_t ab = static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b);
    return {static_cast<std::uint32_t>(ab >> 32ull), static_cast<std::uint32_t>(ab & 0x00000000ffffffffull)};
}

template<int ROUNDS>
static __device__ __forceinline__ uint4 philox(std::uint64_t seed, std::uint32_t x, std::uint32_t y, std::integral_constant<int, ROUNDS> = {}) {
    using std::uint32_t;
    constexpr uint32_t MC[4] = {0xD2511F53, 0x9E3779B9, 0xCD9E8D57, 0xBB67AE85};
    const uint32_t key[2] = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};

    uint32_t R0 = x;
    uint32_t L0 = y;
    uint32_t R1 = 0;
    uint32_t L1 = 0;

    uint32_t K0 = key[0];
    uint32_t K1 = key[1];

    for (int i = 0; i < ROUNDS; ++i) {
        auto [hi0, lo0] = mul_hilo(R0, MC[0]);
        auto [hi1, lo1] = mul_hilo(R1, MC[2]);

        R0 = hi1 ^ L0 ^ K0;
        L0 = lo1;
        R1 = hi0 ^ L1 ^ K1;
        L1 = lo0;

        K0 = (K0 + MC[1]) & 0xFFFFFFFF;
        K1 = (K1 + MC[3]) & 0xFFFFFFFF;
    }
    return make_uint4(R0, L0, R1, L1);
}

// Stochastic rounding for E4M3 scales
static __device__ __forceinline__ __nv_fp8_e4m3 stochastic_rounding(float in, unsigned int noise) {
    __nv_fp8_e4m3 out;
    if (fabs(in) > 448.f || fabs(in) < 0.015625) {
        out.__x = __nv_cvt_float_to_fp8(in, __NV_SATFINITE, __NV_E4M3);
    } else {
        constexpr unsigned SR_MASK = 0x000F'FFFF;
        unsigned int threshold = noise & SR_MASK;
        unsigned int float_bits = __float_as_uint(in);
        unsigned int rounded_bits = float_bits & SR_MASK;
        float_bits = (rounded_bits > threshold) ? (float_bits | SR_MASK) : (float_bits & ~SR_MASK);
        out.__x = __nv_cvt_float_to_fp8(__uint_as_float(float_bits), __NV_SATFINITE, __NV_E4M3);
    }
    return out;
}

// CUDA error checking
class cuda_error : public std::runtime_error {
public:
    cuda_error(cudaError_t err, const std::string& arg) :
            std::runtime_error(arg), code(err) {};
    cudaError_t code;
};

inline void cuda_throw_on_error(cudaError_t status, const char* statement, const char* file, int line) {
    if (status != cudaSuccess) {
        std::string msg = std::string("Cuda Error in ") + file + ":" + std::to_string(line) + " (" + std::string(statement) + "): " + cudaGetErrorName(status) + ": ";
        msg += cudaGetErrorString(status);
        [[maybe_unused]] cudaError_t clear_error = cudaGetLastError();
        throw cuda_error(status, msg);
    }
}

#define QUARTET_CUDA_CHECK(status) quartet::cuda_throw_on_error(status, #status, __FILE__, __LINE__)

}  // namespace quartet
