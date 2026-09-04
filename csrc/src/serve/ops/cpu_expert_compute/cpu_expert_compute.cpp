// CPU expert compute over the planar W8G32 host bank (api/ops/cpu_expert_compute.h).
//
// Layout of one expert: gate/up = 2*intermediate rows (gate rows first) of `hidden` int8 codes,
// scales fp16 per 32-group; down = hidden rows of `intermediate` codes. Per job:
//   xq   = quantise(x)                      (int8 per group, float scale per group)
//   g,u  = dot(gate_row, xq), dot(up_row, xq)
//   h    = silu(g) * u                       (float, intermediate wide)
//   hq   = quantise(h)
//   y    = dot(down_row, hq)                 (hidden wide)
//   out += weight * y
// The AVX-512 path (compiled with -mavx512bw for this file) widens 32 int8 to int16, multiplies
// pairwise into int32 with vpmaddwd and accumulates in float per group; the scalar path is the
// same arithmetic in plain C++ and is what the unit test compares against a double reference.

#include "api/ops/cpu_expert_compute.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#include <sched.h>
#endif

namespace sinfer::ops {
namespace {

constexpr int kGroup = 32;

inline float bf16_to_float(std::uint16_t bits) {
    std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

inline float fp16_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h >> 15) & 1U;
    const std::uint32_t exp  = (h >> 10) & 0x1FU;
    const std::uint32_t mant = h & 0x3FFU;
    std::uint32_t word;
    if (exp == 0) {
        if (mant == 0) {
            word = sign << 31;
        } else { // subnormal
            int e = -1;
            std::uint32_t m = mant;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400U) == 0);
            word = (sign << 31) | static_cast<std::uint32_t>(127 - 15 - e) << 23 | ((m & 0x3FFU) << 13);
        }
    } else if (exp == 31) {
        word = (sign << 31) | 0x7F800000U | (mant << 13);
    } else {
        word = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

inline float silu(float v) { return v / (1.0F + std::exp(-v)); }

/// Quantises `count` floats (a multiple of 32) into int8 groups with one float scale each.
void quantise_groups(const float* values, int count, std::int8_t* q, float* scales);
/// Same quantisation plus, per group, 128 × Σ q (the VNNI kernel's u8×s8 compensation term).
void quantise_groups_sums(const float* values, int count, std::int8_t* q, float* scales, std::int32_t* sums128) {
    quantise_groups(values, count, q, scales);
    for (int g = 0; g < count / kGroup; ++g) {
        std::int32_t sum = 0;
        for (int i = 0; i < kGroup; ++i) { sum += q[g * kGroup + i]; }
        sums128[g] = 128 * sum;
    }
}
void quantise_groups(const float* values, int count, std::int8_t* q, float* scales) {
    for (int g = 0; g < count / kGroup; ++g) {
        float amax = 0.0F;
        for (int i = 0; i < kGroup; ++i) { amax = std::fmax(amax, std::fabs(values[g * kGroup + i])); }
        const float scale = amax / 127.0F;
        const float inv   = scale > 0.0F ? 1.0F / scale : 0.0F;
        scales[g]         = scale;
        for (int i = 0; i < kGroup; ++i) {
            const float r = std::nearbyint(values[g * kGroup + i] * inv);
            q[g * kGroup + i] = static_cast<std::int8_t>(r > 127.0F ? 127.0F : (r < -127.0F ? -127.0F : r));
        }
    }
}

/// Scalar dot of one W8 row against a quantised activation: sum_g (int32 dot) * ws_g * xs_g.
float dot_row_scalar(const std::int8_t* codes, const std::uint16_t* scales, const std::int8_t* xq,
                     const float* xs, int k) {
    float acc = 0.0F;
    for (int g = 0; g < k / kGroup; ++g) {
        std::int32_t dot = 0;
        for (int i = 0; i < kGroup; ++i) {
            dot += static_cast<std::int32_t>(codes[g * kGroup + i]) * static_cast<std::int32_t>(xq[g * kGroup + i]);
        }
        acc += static_cast<float>(dot) * fp16_to_float(scales[g]) * xs[g];
    }
    return acc;
}

/// Software float→FP16 (round-to-nearest-even): the requantiser stores group endpoints with it.
std::uint16_t float_to_fp16(float value) {
    std::uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    const std::uint32_t sign = (f >> 16) & 0x8000U;
    f &= 0x7FFFFFFFU;
    if (f >= 0x47800000U) {                       // overflow (or inf/nan) → inf/nan
        return static_cast<std::uint16_t>(sign | (f > 0x7F800000U ? 0x7E00U : 0x7C00U));
    }
    if (f < 0x38800000U) {                        // subnormal half
        const float scaled = value < 0 ? -value : value;
        const auto bits    = static_cast<std::uint32_t>(scaled * 0x1.0p24F + 0.5F) >> 13;
        return static_cast<std::uint16_t>(sign | bits);
    }
    const std::uint32_t mant = f & 0x00001FFFU;
    std::uint32_t half       = ((f >> 13) - (112U << 10));
    if (mant > 0x1000U || (mant == 0x1000U && (half & 1U))) { ++half; }
    return static_cast<std::uint16_t>(sign | half);
}


/// Scalar dot of one Q4G32AM row: w = step·q + min per group, so
/// Σ w·x = xs·(step·Σ q·xq + min·Σ xq) with both integer sums exact.
float dot_row_q4_scalar(const std::uint8_t* q4, const std::uint16_t* scales,
                        const std::uint16_t* mins, const std::int8_t* xq, const float* xs, int k) {
    float acc = 0.0F;
    for (int g = 0; g < k / kGroup; ++g) {
        std::int32_t dot      = 0;
        std::int32_t sum      = 0;
        const std::uint8_t* c = q4 + static_cast<std::size_t>(g) * (kGroup / 2);
        for (int i = 0; i < kGroup / 2; ++i) {
            const int q0 = c[i] & 0x0F;
            const int q1 = c[i] >> 4;
            const int x0 = xq[g * kGroup + 2 * i];
            const int x1 = xq[g * kGroup + 2 * i + 1];
            dot += q0 * x0 + q1 * x1;
            sum += x0 + x1;
        }
        acc += xs[g] * (fp16_to_float(scales[g]) * static_cast<float>(dot) +
                        fp16_to_float(mins[g]) * static_cast<float>(sum));
    }
    return acc;
}

#if defined(__x86_64__) && defined(__AVX512BW__)
constexpr bool kAvx512Compiled = true;

bool detect_avx512() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}

/// AVX-512 dots of two W8 rows against one quantised activation (gate and up rows share the
/// activation loads): per 32-group, widen int8→int16 (32 lanes), vpmaddwd → 16 int32 pair sums,
/// scale by (w_scale * x_scale) with the row's fp16 scales converted 16 at a time (F16C), and
/// reduce once per row at the end.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c,fma")))
void dot_two_rows_avx512(const std::int8_t* codes0, const std::uint16_t* scales0,
                         const std::int8_t* codes1, const std::uint16_t* scales1,
                         const std::int8_t* xq, const float* xs, int k, float& out0, float& out1) {
    const int groups = k / kGroup;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int g = 0;
    for (; g + 16 <= groups; g += 16) {
        // 16 groups' worth of scales at once.
        const __m512 ws0 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(scales0 + g)));
        const __m512 ws1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(scales1 + g)));
        const __m512 xsv = _mm512_loadu_ps(xs + g);
        const __m512 s0  = _mm512_mul_ps(ws0, xsv);
        const __m512 s1  = _mm512_mul_ps(ws1, xsv);
        alignas(64) float sc0[16];
        alignas(64) float sc1[16];
        _mm512_store_ps(sc0, s0);
        _mm512_store_ps(sc1, s1);
        for (int i = 0; i < 16; ++i) {
            const int gg     = g + i;
            const __m512i xv = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + gg * kGroup)));
            const __m512i w0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes0 + gg * kGroup)));
            const __m512i w1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes1 + gg * kGroup)));
            acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w0, xv)), _mm512_set1_ps(sc0[i]), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w1, xv)), _mm512_set1_ps(sc1[i]), acc1);
        }
    }
    for (; g < groups; ++g) {
        const __m512i xv = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + g * kGroup)));
        const __m512i w0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes0 + g * kGroup)));
        const __m512i w1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes1 + g * kGroup)));
        const float s0   = fp16_to_float(scales0[g]) * xs[g];
        const float s1   = fp16_to_float(scales1[g]) * xs[g];
        acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w0, xv)), _mm512_set1_ps(s0), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w1, xv)), _mm512_set1_ps(s1), acc1);
    }
    out0 = _mm512_reduce_add_ps(acc0);
    out1 = _mm512_reduce_add_ps(acc1);
}

/// Two rows against two tokens: the row loads/widening and the fp16 scale conversion are shared
/// across the tokens, which is the reuse a batched round lives on (a row read from DRAM once
/// serves every token routed to the expert).
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c,fma")))
void dot_two_rows_two_tokens_avx512(const std::int8_t* codes0, const std::uint16_t* scales0,
                                    const std::int8_t* codes1, const std::uint16_t* scales1,
                                    const std::int8_t* xqa, const float* xsa,
                                    const std::int8_t* xqb, const float* xsb, int k,
                                    float& out0a, float& out1a, float& out0b, float& out1b) {
    const int groups = k / kGroup;
    __m512 acc0a = _mm512_setzero_ps(), acc1a = _mm512_setzero_ps();
    __m512 acc0b = _mm512_setzero_ps(), acc1b = _mm512_setzero_ps();
    int g = 0;
    for (; g + 16 <= groups; g += 16) {
        const __m512 ws0 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(scales0 + g)));
        const __m512 ws1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(scales1 + g)));
        const __m512 xsav = _mm512_loadu_ps(xsa + g);
        const __m512 xsbv = _mm512_loadu_ps(xsb + g);
        alignas(64) float s0a[16], s1a[16], s0b[16], s1b[16];
        _mm512_store_ps(s0a, _mm512_mul_ps(ws0, xsav));
        _mm512_store_ps(s1a, _mm512_mul_ps(ws1, xsav));
        _mm512_store_ps(s0b, _mm512_mul_ps(ws0, xsbv));
        _mm512_store_ps(s1b, _mm512_mul_ps(ws1, xsbv));
        for (int i = 0; i < 16; ++i) {
            const int gg     = g + i;
            const __m512i w0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes0 + gg * kGroup)));
            const __m512i w1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes1 + gg * kGroup)));
            const __m512i xa = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(xqa + gg * kGroup)));
            const __m512i xb = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(xqb + gg * kGroup)));
            acc0a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w0, xa)), _mm512_set1_ps(s0a[i]), acc0a);
            acc1a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w1, xa)), _mm512_set1_ps(s1a[i]), acc1a);
            acc0b = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w0, xb)), _mm512_set1_ps(s0b[i]), acc0b);
            acc1b = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w1, xb)), _mm512_set1_ps(s1b[i]), acc1b);
        }
    }
    for (; g < groups; ++g) {
        const __m512i w0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes0 + g * kGroup)));
        const __m512i w1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes1 + g * kGroup)));
        const __m512i xa = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(xqa + g * kGroup)));
        const __m512i xb = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(xqb + g * kGroup)));
        const float w0s = fp16_to_float(scales0[g]), w1s = fp16_to_float(scales1[g]);
        acc0a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w0, xa)), _mm512_set1_ps(w0s * xsa[g]), acc0a);
        acc1a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w1, xa)), _mm512_set1_ps(w1s * xsa[g]), acc1a);
        acc0b = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w0, xb)), _mm512_set1_ps(w0s * xsb[g]), acc0b);
        acc1b = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_madd_epi16(w1, xb)), _mm512_set1_ps(w1s * xsb[g]), acc1b);
    }
    out0a = _mm512_reduce_add_ps(acc0a);
    out1a = _mm512_reduce_add_ps(acc1a);
    out0b = _mm512_reduce_add_ps(acc0b);
    out1b = _mm512_reduce_add_ps(acc1b);
}

/// Expands 32 packed Q4 bytes (two 32-groups) into 64 unsigned byte lanes in element order:
/// after a u8→u16 widen, each 16-bit lane holds one source byte, and (b & 0x0F) | ((b & 0xF0)
/// << 4) leaves the even element in the low byte and the odd element in the high byte.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,f16c,fma")))
inline __m512i expand_q4x64(const std::uint8_t* packed) {
    const __m512i t =
        _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed)));
    return _mm512_or_si512(_mm512_and_si512(t, _mm512_set1_epi16(0x000F)),
                           _mm512_slli_epi16(_mm512_and_si512(t, _mm512_set1_epi16(0x00F0)), 4));
}

/// VNNI Q4G32AM twin of dot_two_rows_two_tokens_vnni. The codes are unsigned nibbles, so there
/// is no sign-flip compensation; instead each group's affine min contributes min·xs·Σxq, added
/// once per 16-group block from the precomputed 128·Σxq sums.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,f16c,fma")))
void dot_two_rows_two_tokens_q4_vnni(
    const std::uint8_t* codes0, const std::uint16_t* scales0, const std::uint16_t* mins0,
    const std::uint8_t* codes1, const std::uint16_t* scales1, const std::uint16_t* mins1,
    const std::int8_t* xqa, const float* xsa, const std::int32_t* xca, const std::int8_t* xqb,
    const float* xsb, const std::int32_t* xcb, int k, float& out0a, float& out1a, float& out0b,
    float& out1b) {
    const int groups    = k / kGroup; // even
    const __m512 inv128 = _mm512_set1_ps(1.0F / 128.0F);
    __m512 acc0a = _mm512_setzero_ps(), acc1a = _mm512_setzero_ps();
    __m512 acc0b = _mm512_setzero_ps(), acc1b = _mm512_setzero_ps();
    for (int g = 0; g < groups; g += 16) {
        const int block    = std::min(16, groups - g);
        const __mmask16 bm = static_cast<__mmask16>((1U << block) - 1U);
        const __m512 ws0  = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(bm, scales0 + g));
        const __m512 ws1  = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(bm, scales1 + g));
        const __m512 wm0  = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(bm, mins0 + g));
        const __m512 wm1  = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(bm, mins1 + g));
        const __m512 xsav = _mm512_maskz_loadu_ps(bm, xsa + g);
        const __m512 xsbv = _mm512_maskz_loadu_ps(bm, xsb + g);
        const __m512 s0a = _mm512_mul_ps(ws0, xsav), s1a = _mm512_mul_ps(ws1, xsav);
        const __m512 s0b = _mm512_mul_ps(ws0, xsbv), s1b = _mm512_mul_ps(ws1, xsbv);
        const __m512 caf = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(bm, xca + g));
        const __m512 cbf = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(bm, xcb + g));
        const __m512 xsa128 = _mm512_mul_ps(xsav, inv128);
        const __m512 xsb128 = _mm512_mul_ps(xsbv, inv128);
        acc0a = _mm512_fmadd_ps(caf, _mm512_mul_ps(wm0, xsa128), acc0a);
        acc1a = _mm512_fmadd_ps(caf, _mm512_mul_ps(wm1, xsa128), acc1a);
        acc0b = _mm512_fmadd_ps(cbf, _mm512_mul_ps(wm0, xsb128), acc0b);
        acc1b = _mm512_fmadd_ps(cbf, _mm512_mul_ps(wm1, xsb128), acc1b);
        for (int i = 0; i < block / 2; ++i) {
            const int gg = g + 2 * i;
            const __m512i idx = _mm512_set_epi32(2 * i + 1, 2 * i + 1, 2 * i + 1, 2 * i + 1,
                                                 2 * i + 1, 2 * i + 1, 2 * i + 1, 2 * i + 1,
                                                 2 * i, 2 * i, 2 * i, 2 * i, 2 * i, 2 * i,
                                                 2 * i, 2 * i);
            const __m512i w0 = expand_q4x64(codes0 + static_cast<std::size_t>(gg) * (kGroup / 2));
            const __m512i w1 = expand_q4x64(codes1 + static_cast<std::size_t>(gg) * (kGroup / 2));
            const __m512i xa = _mm512_loadu_si512(xqa + gg * kGroup);
            const __m512i xb = _mm512_loadu_si512(xqb + gg * kGroup);
            const __m512i d0a = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w0, xa);
            const __m512i d1a = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w1, xa);
            const __m512i d0b = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w0, xb);
            const __m512i d1b = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w1, xb);
            acc0a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d0a), _mm512_permutexvar_ps(idx, s0a), acc0a);
            acc1a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d1a), _mm512_permutexvar_ps(idx, s1a), acc1a);
            acc0b = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d0b), _mm512_permutexvar_ps(idx, s0b), acc0b);
            acc1b = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d1b), _mm512_permutexvar_ps(idx, s1b), acc1b);
        }
    }
    out0a = _mm512_reduce_add_ps(acc0a);
    out1a = _mm512_reduce_add_ps(acc1a);
    out0b = _mm512_reduce_add_ps(acc0b);
    out1b = _mm512_reduce_add_ps(acc1b);
}

/// One-token flavour of the Q4 VNNI dot (the decode rounds' main shape: one job per expert).
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,f16c,fma")))
void dot_two_rows_q4_vnni(const std::uint8_t* codes0, const std::uint16_t* scales0,
                          const std::uint16_t* mins0, const std::uint8_t* codes1,
                          const std::uint16_t* scales1, const std::uint16_t* mins1,
                          const std::int8_t* xq, const float* xs, const std::int32_t* xc, int k,
                          float& out0, float& out1) {
    const int groups    = k / kGroup;
    const __m512 inv128 = _mm512_set1_ps(1.0F / 128.0F);
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    for (int g = 0; g < groups; g += 16) {
        const int block    = std::min(16, groups - g);
        const __mmask16 bm = static_cast<__mmask16>((1U << block) - 1U);
        const __m512 ws0 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(bm, scales0 + g));
        const __m512 ws1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(bm, scales1 + g));
        const __m512 wm0 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(bm, mins0 + g));
        const __m512 wm1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(bm, mins1 + g));
        const __m512 xsv = _mm512_maskz_loadu_ps(bm, xs + g);
        const __m512 s0 = _mm512_mul_ps(ws0, xsv), s1 = _mm512_mul_ps(ws1, xsv);
        const __m512 cf    = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(bm, xc + g));
        const __m512 xs128 = _mm512_mul_ps(xsv, inv128);
        acc0 = _mm512_fmadd_ps(cf, _mm512_mul_ps(wm0, xs128), acc0);
        acc1 = _mm512_fmadd_ps(cf, _mm512_mul_ps(wm1, xs128), acc1);
        for (int i = 0; i < block / 2; ++i) {
            const int gg = g + 2 * i;
            const __m512i idx = _mm512_set_epi32(2 * i + 1, 2 * i + 1, 2 * i + 1, 2 * i + 1,
                                                 2 * i + 1, 2 * i + 1, 2 * i + 1, 2 * i + 1,
                                                 2 * i, 2 * i, 2 * i, 2 * i, 2 * i, 2 * i,
                                                 2 * i, 2 * i);
            const __m512i w0 = expand_q4x64(codes0 + static_cast<std::size_t>(gg) * (kGroup / 2));
            const __m512i w1 = expand_q4x64(codes1 + static_cast<std::size_t>(gg) * (kGroup / 2));
            const __m512i xv = _mm512_loadu_si512(xq + gg * kGroup);
            const __m512i d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w0, xv);
            const __m512i d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w1, xv);
            acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d0), _mm512_permutexvar_ps(idx, s0), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d1), _mm512_permutexvar_ps(idx, s1), acc1);
        }
    }
    out0 = _mm512_reduce_add_ps(acc0);
    out1 = _mm512_reduce_add_ps(acc1);
}


#if defined(__AVX512VNNI__)
constexpr bool kVnniCompiled = true;
bool detect_vnni() { return __builtin_cpu_supports("avx512vnni"); }

/// VNNI variant of the two-rows × two-tokens dot: `vpdpbusd` takes 64 u8×s8 products per
/// instruction (two 32-groups per zmm). The weights are the unsigned operand (w + 128 via a
/// sign-bit flip), so each group's exact int32 dot is D_g − 128·Σx_g with the token's group
/// sums precomputed at quantisation; the numerics stay the exact-int32-then-float-scale of the
/// other paths. Needs k % 64 == 0.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,f16c,fma")))
void dot_two_rows_two_tokens_vnni(const std::int8_t* codes0, const std::uint16_t* scales0,
                                  const std::int8_t* codes1, const std::uint16_t* scales1,
                                  const std::int8_t* xqa, const float* xsa, const std::int32_t* xca,
                                  const std::int8_t* xqb, const float* xsb, const std::int32_t* xcb, int k,
                                  float& out0a, float& out1a, float& out0b, float& out1b) {
    const int groups = k / kGroup; // even
    const __m512i flip = _mm512_set1_epi8(static_cast<char>(0x80));
    __m512 acc0a = _mm512_setzero_ps(), acc1a = _mm512_setzero_ps();
    __m512 acc0b = _mm512_setzero_ps(), acc1b = _mm512_setzero_ps();
    // Lane maps for a group pair (2i, 2i+1) within a 16-group block: scale lanes 0-7 ← group
    // 2i, lanes 8-15 ← group 2i+1; the compensation sits in lane 0 of each group's 8 lanes.
    const __mmask16 comp_mask = 0x0101;
    int g = 0;
    for (; g < groups; g += 16) {
        const int block = std::min(16, groups - g); // 16 or a tail (multiple of 2)
        const __mmask16 bm = static_cast<__mmask16>((1U << block) - 1U);
        const __m512 ws0 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(bm, scales0 + g));
        const __m512 ws1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(bm, scales1 + g));
        const __m512 xsav = _mm512_maskz_loadu_ps(bm, xsa + g);
        const __m512 xsbv = _mm512_maskz_loadu_ps(bm, xsb + g);
        const __m512 s0a = _mm512_mul_ps(ws0, xsav), s1a = _mm512_mul_ps(ws1, xsav);
        const __m512 s0b = _mm512_mul_ps(ws0, xsbv), s1b = _mm512_mul_ps(ws1, xsbv);
        const __m512i ca = _mm512_maskz_loadu_epi32(bm, xca + g);
        const __m512i cb = _mm512_maskz_loadu_epi32(bm, xcb + g);
        for (int i = 0; i < block / 2; ++i) {
            const int gg = g + 2 * i;
            const __m512i idx = _mm512_set_epi32(2 * i + 1, 2 * i + 1, 2 * i + 1, 2 * i + 1, 2 * i + 1, 2 * i + 1,
                                                 2 * i + 1, 2 * i + 1, 2 * i, 2 * i, 2 * i, 2 * i, 2 * i, 2 * i, 2 * i, 2 * i);
            const __m512i w0 = _mm512_xor_si512(_mm512_loadu_si512(codes0 + gg * kGroup), flip);
            const __m512i w1 = _mm512_xor_si512(_mm512_loadu_si512(codes1 + gg * kGroup), flip);
            const __m512i xa = _mm512_loadu_si512(xqa + gg * kGroup);
            const __m512i xb = _mm512_loadu_si512(xqb + gg * kGroup);
            const __m512i compa = _mm512_maskz_permutexvar_epi32(comp_mask, idx, ca);
            const __m512i compb = _mm512_maskz_permutexvar_epi32(comp_mask, idx, cb);
            const __m512i d0a = _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), w0, xa), compa);
            const __m512i d1a = _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), w1, xa), compa);
            const __m512i d0b = _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), w0, xb), compb);
            const __m512i d1b = _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), w1, xb), compb);
            acc0a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d0a), _mm512_permutexvar_ps(idx, s0a), acc0a);
            acc1a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d1a), _mm512_permutexvar_ps(idx, s1a), acc1a);
            acc0b = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d0b), _mm512_permutexvar_ps(idx, s0b), acc0b);
            acc1b = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d1b), _mm512_permutexvar_ps(idx, s1b), acc1b);
        }
    }
    out0a = _mm512_reduce_add_ps(acc0a);
    out1a = _mm512_reduce_add_ps(acc1a);
    out0b = _mm512_reduce_add_ps(acc0b);
    out1b = _mm512_reduce_add_ps(acc1b);
}

/// Interleaved-rows tile ("R16"): a block of 16 rows is stored group by group as 8 zmm of
/// [16 rows][4 k-elements] int8, followed by the rows' 128·Σw compensation (int32[16]) and
/// their fp16 scales converted to float ([16]). One `vpdpbusd` against a broadcast of the
/// token's 4 activation bytes (u8: q ^ 0x80) advances all 16 rows, and the per-group scale is
/// applied once per 16 rows. The tile is built once per expert chunk and reused by every
/// token routed to the expert.
constexpr int kTileRows            = 16;
constexpr int kTileGroupCodeBytes  = kGroup * kTileRows;                       // 512
constexpr int kTileGroupBytes      = kTileGroupCodeBytes + 2 * 64;             // + comp + scales
inline std::size_t tile_block_bytes(int k) { return static_cast<std::size_t>(k / kGroup) * kTileGroupBytes; }

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,f16c,fma")))
void repack_tile_block_vnni(const std::int8_t* codes, const std::uint16_t* scales, int k, std::byte* tile) {
    // codes: 16 rows, row stride k; scales: 16 rows, row stride k/32.
    const int groups     = k / kGroup;
    const __m512i ones   = _mm512_set1_epi8(1);
    for (int g = 0; g < groups; ++g) {
        std::byte* out = tile + static_cast<std::size_t>(g) * kTileGroupBytes;
        __m512i comp   = _mm512_setzero_si512();
        for (int step = 0; step < 8; ++step) {
            // Gather 4 bytes of each of the 16 rows: 16 dword loads.
            alignas(64) std::int32_t lanes[16];
            for (int r = 0; r < kTileRows; ++r) {
                std::memcpy(&lanes[r], codes + static_cast<std::size_t>(r) * k + g * kGroup + step * 4, 4);
            }
            const __m512i w = _mm512_load_si512(lanes);
            _mm512_storeu_si512(out + step * 64, w);
            comp = _mm512_dpbusd_epi32(comp, ones, w); // Σ of the 4 signed bytes per row lane
        }
        _mm512_storeu_si512(out + kTileGroupCodeBytes, _mm512_slli_epi32(comp, 7)); // 128·Σw
        alignas(64) std::uint16_t sc[16];
        for (int r = 0; r < kTileRows; ++r) { sc[r] = scales[static_cast<std::size_t>(r) * groups + g]; }
        _mm512_storeu_ps(reinterpret_cast<float*>(out + kTileGroupCodeBytes + 64),
                         _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i*>(sc))));
    }
}

/// 16 rows of one tile block against two tokens (u8 activations `xua`/`xub` = q ^ 0x80, group
/// scales `xsa`/`xsb`); writes 16 floats per token.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,f16c,fma")))
void dot_tile_block_two_tokens_vnni(const std::byte* tile, int k, const std::uint8_t* xua, const float* xsa,
                                    const std::uint8_t* xub, const float* xsb, float* outa, float* outb) {
    const int groups = k / kGroup;
    __m512 acca = _mm512_setzero_ps(), accb = _mm512_setzero_ps();
    for (int g = 0; g < groups; ++g) {
        const std::byte* blk = tile + static_cast<std::size_t>(g) * kTileGroupBytes;
        __m512i da = _mm512_setzero_si512(), db = _mm512_setzero_si512();
        const std::int32_t* xa32 = reinterpret_cast<const std::int32_t*>(xua + g * kGroup);
        const std::int32_t* xb32 = reinterpret_cast<const std::int32_t*>(xub + g * kGroup);
        for (int step = 0; step < 8; ++step) {
            const __m512i w = _mm512_loadu_si512(blk + step * 64);
            da = _mm512_dpbusd_epi32(da, _mm512_set1_epi32(xa32[step]), w);
            db = _mm512_dpbusd_epi32(db, _mm512_set1_epi32(xb32[step]), w);
        }
        const __m512i comp = _mm512_loadu_si512(blk + kTileGroupCodeBytes);
        const __m512 ws    = _mm512_loadu_ps(reinterpret_cast<const float*>(blk + kTileGroupCodeBytes + 64));
        acca = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(da, comp)), _mm512_mul_ps(ws, _mm512_set1_ps(xsa[g])), acca);
        accb = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(db, comp)), _mm512_mul_ps(ws, _mm512_set1_ps(xsb[g])), accb);
    }
    _mm512_storeu_ps(outa, acca);
    _mm512_storeu_ps(outb, accb);
}
#else
constexpr bool kVnniCompiled = false;
bool detect_vnni() { return false; }
constexpr int kTileRows = 16;
inline std::size_t tile_block_bytes(int) { return 0; }
void repack_tile_block_vnni(const std::int8_t*, const std::uint16_t*, int, std::byte*) {}
void dot_tile_block_two_tokens_vnni(const std::byte*, int, const std::uint8_t*, const float*, const std::uint8_t*,
                                    const float*, float*, float*) {}
void dot_two_rows_two_tokens_vnni(const std::int8_t*, const std::uint16_t*, const std::int8_t*, const std::uint16_t*,
                                  const std::int8_t*, const float*, const std::int32_t*, const std::int8_t*,
                                  const float*, const std::int32_t*, int, float&, float&, float&, float&) {}
#endif

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c,fma")))
float dot_row_avx512(const std::int8_t* codes, const std::uint16_t* scales, const std::int8_t* xq,
                     const float* xs, int k) {
    float a = 0.0F, b = 0.0F;
    dot_two_rows_avx512(codes, scales, codes, scales, xq, xs, k, a, b);
    return a;
}
#else
constexpr bool kAvx512Compiled = false;
constexpr bool kVnniCompiled   = false;
bool detect_avx512() { return false; }
bool detect_vnni() { return false; }
constexpr int kTileRows = 16;
inline std::size_t tile_block_bytes(int) { return 0; }
void repack_tile_block_vnni(const std::int8_t*, const std::uint16_t*, int, std::byte*) {}
void dot_tile_block_two_tokens_vnni(const std::byte*, int, const std::uint8_t*, const float*, const std::uint8_t*,
                                    const float*, float*, float*) {}
void dot_two_rows_two_tokens_vnni(const std::int8_t*, const std::uint16_t*, const std::int8_t*, const std::uint16_t*,
                                  const std::int8_t*, const float*, const std::int32_t*, const std::int8_t*,
                                  const float*, const std::int32_t*, int, float&, float&, float&, float&) {}
float dot_row_avx512(const std::int8_t*, const std::uint16_t*, const std::int8_t*, const float*, int) {
    return 0.0F;
}
void dot_two_rows_avx512(const std::int8_t*, const std::uint16_t*, const std::int8_t*, const std::uint16_t*,
                         const std::int8_t*, const float*, int, float&, float&) {}
void dot_two_rows_two_tokens_avx512(const std::int8_t*, const std::uint16_t*, const std::int8_t*,
                                    const std::uint16_t*, const std::int8_t*, const float*, const std::int8_t*,
                                    const float*, int, float&, float&, float&, float&) {}
#endif

const bool kUseAvx512 = kAvx512Compiled && detect_avx512();
const bool kUseVnni   = kUseAvx512 && kVnniCompiled && detect_vnni() && std::getenv("SUROGATE_CPU_EXPERT_NO_VNNI") == nullptr;
// The tile path pays a repack per expert chunk. As written (scalar 4-byte gathers in the
// repack) it measures level with the pairwise VNNI path at 10-40 tokens per expert, so it is
// opt-in (SUROGATE_CPU_EXPERT_TILE=1) until the repack is vectorised.
const bool kUseTile   = kUseVnni && std::getenv("SUROGATE_CPU_EXPERT_TILE") != nullptr;
const int kTileMinTokens = [] {
    const char* v = std::getenv("SUROGATE_CPU_EXPERT_TILE_MIN");
    return v != nullptr && *v != '\0' ? std::max(2, std::atoi(v)) : 4;
}();

inline float dot_row(const std::int8_t* codes, const std::uint16_t* scales, const std::int8_t* xq,
                     const float* xs, int k) {
    return kUseAvx512 ? dot_row_avx512(codes, scales, xq, xs, k) : dot_row_scalar(codes, scales, xq, xs, k);
}
inline void dot_two_rows(const std::int8_t* codes0, const std::uint16_t* scales0, const std::int8_t* codes1,
                         const std::uint16_t* scales1, const std::int8_t* xq, const float* xs, int k,
                         float& out0, float& out1) {
    if (kUseAvx512) {
        dot_two_rows_avx512(codes0, scales0, codes1, scales1, xq, xs, k, out0, out1);
    } else {
        out0 = dot_row_scalar(codes0, scales0, xq, xs, k);
        out1 = dot_row_scalar(codes1, scales1, xq, xs, k);
    }
}

inline void dot_two_rows_two_tokens(const std::int8_t* codes0, const std::uint16_t* scales0,
                                    const std::int8_t* codes1, const std::uint16_t* scales1,
                                    const std::int8_t* xqa, const float* xsa, const std::int32_t* xca,
                                    const std::int8_t* xqb, const float* xsb, const std::int32_t* xcb, int k,
                                    float& out0a, float& out1a, float& out0b, float& out1b) {
    if (kUseVnni && k % 64 == 0) {
        dot_two_rows_two_tokens_vnni(codes0, scales0, codes1, scales1, xqa, xsa, xca, xqb, xsb, xcb, k, out0a, out1a,
                                     out0b, out1b);
    } else if (kUseAvx512) {
        dot_two_rows_two_tokens_avx512(codes0, scales0, codes1, scales1, xqa, xsa, xqb, xsb, k, out0a, out1a,
                                       out0b, out1b);
    } else {
        out0a = dot_row_scalar(codes0, scales0, xqa, xsa, k);
        out1a = dot_row_scalar(codes1, scales1, xqa, xsa, k);
        out0b = dot_row_scalar(codes0, scales0, xqb, xsb, k);
        out1b = dot_row_scalar(codes1, scales1, xqb, xsb, k);
    }
}

inline void dot_two_rows_two_tokens_q4(
    const std::uint8_t* codes0, const std::uint16_t* scales0, const std::uint16_t* mins0,
    const std::uint8_t* codes1, const std::uint16_t* scales1, const std::uint16_t* mins1,
    const std::int8_t* xqa, const float* xsa, const std::int32_t* xca, const std::int8_t* xqb,
    const float* xsb, const std::int32_t* xcb, int k, float& out0a, float& out1a, float& out0b,
    float& out1b) {
    if (kUseVnni && k % 64 == 0) {
        dot_two_rows_two_tokens_q4_vnni(codes0, scales0, mins0, codes1, scales1, mins1, xqa, xsa,
                                        xca, xqb, xsb, xcb, k, out0a, out1a, out0b, out1b);
    } else {
        out0a = dot_row_q4_scalar(codes0, scales0, mins0, xqa, xsa, k);
        out1a = dot_row_q4_scalar(codes1, scales1, mins1, xqa, xsa, k);
        out0b = dot_row_q4_scalar(codes0, scales0, mins0, xqb, xsb, k);
        out1b = dot_row_q4_scalar(codes1, scales1, mins1, xqb, xsb, k);
    }
}

inline void dot_two_rows_q4(const std::uint8_t* codes0, const std::uint16_t* scales0,
                            const std::uint16_t* mins0, const std::uint8_t* codes1,
                            const std::uint16_t* scales1, const std::uint16_t* mins1,
                            const std::int8_t* xq, const float* xs, const std::int32_t* xc, int k,
                            float& out0, float& out1) {
    if (kUseVnni && k % 64 == 0) {
        dot_two_rows_q4_vnni(codes0, scales0, mins0, codes1, scales1, mins1, xq, xs, xc, k, out0,
                             out1);
    } else {
        out0 = dot_row_q4_scalar(codes0, scales0, mins0, xq, xs, k);
        out1 = dot_row_q4_scalar(codes1, scales1, mins1, xq, xs, k);
    }
}

struct Scratch {
    float* x_float;
    std::int8_t* xq;
    float* xs;
    float* h;
    std::int8_t* hq;
    float* hs;
};

Scratch carve_scratch(const SparseMoeGeometry& geometry, std::byte* base) {
    auto align = [](std::size_t v) { return (v + 63) / 64 * 64; };
    Scratch s{};
    std::size_t offset = 0;
    s.x_float = reinterpret_cast<float*>(base + offset); offset += align(sizeof(float) * geometry.hidden);
    s.xq      = reinterpret_cast<std::int8_t*>(base + offset); offset += align(geometry.hidden);
    s.xs      = reinterpret_cast<float*>(base + offset); offset += align(sizeof(float) * (geometry.hidden / kGroup));
    s.h       = reinterpret_cast<float*>(base + offset); offset += align(sizeof(float) * geometry.intermediate);
    s.hq      = reinterpret_cast<std::int8_t*>(base + offset); offset += align(geometry.intermediate);
    s.hs      = reinterpret_cast<float*>(base + offset);
    return s;
}

void require_geometry(const SparseMoeGeometry& geometry) {
    if (geometry.hidden <= 0 || geometry.hidden % kGroup != 0 || geometry.intermediate <= 0 ||
        geometry.intermediate % kGroup != 0 || geometry.experts <= 0) {
        throw std::invalid_argument("cpu_expert_compute: hidden and intermediate must be positive multiples of 32");
    }
}

} // namespace

void requantise_w8_expert_groups_to_q4(const std::int8_t* codes, const std::uint16_t* scales,
                                       std::int64_t groups, std::uint8_t* q4,
                                       std::uint16_t* q4_scales, std::uint16_t* q4_mins) {
    for (std::int64_t g = 0; g < groups; ++g) {
        const float s        = fp16_to_float(scales[g]);
        const std::int8_t* c = codes + g * kGroup;
        std::int8_t lo8 = c[0], hi8 = c[0];
        for (int i = 1; i < kGroup; ++i) {
            lo8 = std::min(lo8, c[i]);
            hi8 = std::max(hi8, c[i]);
        }
        // Fit the codes against the *rounded* endpoints, so what a reader reconstructs is what
        // was optimised for.
        const std::uint16_t min_bits  = float_to_fp16(static_cast<float>(lo8) * s);
        const std::uint16_t step_bits = float_to_fp16(static_cast<float>(hi8 - lo8) * s / 15.0F);
        const float min_v             = fp16_to_float(min_bits);
        const float step_v            = fp16_to_float(step_bits);
        const float inv               = step_v != 0.0F ? 1.0F / step_v : 0.0F;
        std::uint8_t* out             = q4 + g * (kGroup / 2);
        for (int i = 0; i < kGroup; i += 2) {
            const auto one = [&](int j) {
                const float v = static_cast<float>(c[j]) * s;
                const long q  = std::lround((v - min_v) * inv);
                return static_cast<std::uint8_t>(std::clamp<long>(q, 0, 15));
            };
            out[i / 2] = static_cast<std::uint8_t>(one(i) | (one(i + 1) << 4));
        }
        q4_scales[g] = step_bits;
        q4_mins[g]   = min_bits;
    }
}

bool cpu_expert_compute_has_avx512() noexcept { return kUseAvx512; }
bool cpu_expert_compute_has_vnni() noexcept { return kUseVnni; }
bool cpu_expert_compute_has_tile() noexcept { return kUseTile; }

std::size_t cpu_expert_scratch_bytes(const SparseMoeGeometry& geometry) {
    require_geometry(geometry);
    auto align = [](std::size_t v) { return (v + 63) / 64 * 64; };
    return align(sizeof(float) * geometry.hidden) + align(geometry.hidden) +
           align(sizeof(float) * (geometry.hidden / kGroup)) + align(sizeof(float) * geometry.intermediate) +
           align(geometry.intermediate) + align(sizeof(float) * (geometry.intermediate / kGroup)) + 64;
}

void cpu_expert_compute_job(const SparseMoeGeometry& geometry, const CpuExpertBank& bank,
                            const CpuExpertJob& job, const std::uint16_t* x_column, float* out_column,
                            std::byte* scratch) {
    require_geometry(geometry);
    if (job.expert < 0 || job.expert >= geometry.experts) {
        throw std::invalid_argument("cpu_expert_compute: expert out of range");
    }
    const int hidden       = geometry.hidden;
    const int intermediate = geometry.intermediate;
    const Scratch s        = carve_scratch(geometry, scratch);
    for (int i = 0; i < hidden; ++i) { s.x_float[i] = bf16_to_float(x_column[i]); }
    quantise_groups(s.x_float, hidden, s.xq, s.xs);

    if (bank.format == ExpertBankFormat::GgmlBlocks) {
        // No GGML decoder on this path yet. Reading the blocks as W8 planes would answer with
        // nonsense rather than fail, so it fails.
        throw std::invalid_argument(
            "cpu_expert_compute: a GGML-block expert bank has no CPU path; run the experts on "
            "the device, or convert with --host-expert-bank w8");
    }
    if (bank.format == ExpertBankFormat::Q4G32AM) {
        // Reference path for the Q4 bank: the scalar Q4 dot is the oracle the SIMD kernels are
        // tested against, so this stays scalar on purpose.
        const std::size_t gate_rows = static_cast<std::size_t>(2) * intermediate;
        const std::size_t groups_h  = static_cast<std::size_t>(hidden / kGroup);
        const std::size_t groups_i  = static_cast<std::size_t>(intermediate / kGroup);
        const auto* gate4 = reinterpret_cast<const std::uint8_t*>(bank.gate_up_codes) +
                            static_cast<std::size_t>(job.expert) * gate_rows * (hidden / 2);
        const auto* gsc = reinterpret_cast<const std::uint16_t*>(bank.gate_up_scales) +
                          static_cast<std::size_t>(job.expert) * gate_rows * groups_h;
        const auto* gmn = reinterpret_cast<const std::uint16_t*>(bank.gate_up_mins) +
                          static_cast<std::size_t>(job.expert) * gate_rows * groups_h;
        for (int j = 0; j < intermediate; ++j) {
            const float g = dot_row_q4_scalar(gate4 + static_cast<std::size_t>(j) * (hidden / 2),
                                              gsc + j * groups_h, gmn + j * groups_h, s.xq, s.xs,
                                              hidden);
            const float u = dot_row_q4_scalar(
                gate4 + static_cast<std::size_t>(intermediate + j) * (hidden / 2),
                gsc + (intermediate + j) * groups_h, gmn + (intermediate + j) * groups_h, s.xq,
                s.xs, hidden);
            s.h[j] = silu(g) * u;
        }
        quantise_groups(s.h, intermediate, s.hq, s.hs);
        const auto* down4 = reinterpret_cast<const std::uint8_t*>(bank.down_codes) +
                            static_cast<std::size_t>(job.expert) * hidden * (intermediate / 2);
        const auto* dsc = reinterpret_cast<const std::uint16_t*>(bank.down_scales) +
                          static_cast<std::size_t>(job.expert) * hidden * groups_i;
        const auto* dmn = reinterpret_cast<const std::uint16_t*>(bank.down_mins) +
                          static_cast<std::size_t>(job.expert) * hidden * groups_i;
        for (int r = 0; r < hidden; ++r) {
            out_column[r] += job.weight * dot_row_q4_scalar(
                                              down4 + static_cast<std::size_t>(r) * (intermediate / 2),
                                              dsc + r * groups_i, dmn + r * groups_i, s.hq, s.hs,
                                              intermediate);
        }
        return;
    }
    const std::size_t gate_rows       = static_cast<std::size_t>(2) * intermediate;
    const std::size_t gate_row_codes  = static_cast<std::size_t>(hidden);
    const std::size_t gate_row_scales = static_cast<std::size_t>(hidden / kGroup);
    const auto* gate_codes            = reinterpret_cast<const std::int8_t*>(bank.gate_up_codes) +
                           static_cast<std::size_t>(job.expert) * gate_rows * gate_row_codes;
    const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(bank.gate_up_scales) +
                              static_cast<std::size_t>(job.expert) * gate_rows * gate_row_scales;
    for (int j = 0; j < intermediate; ++j) {
        float g = 0.0F, u = 0.0F;
        dot_two_rows(gate_codes + j * gate_row_codes, gate_scales + j * gate_row_scales,
                     gate_codes + (intermediate + j) * gate_row_codes,
                     gate_scales + (intermediate + j) * gate_row_scales, s.xq, s.xs, hidden, g, u);
        s.h[j] = silu(g) * u;
    }
    quantise_groups(s.h, intermediate, s.hq, s.hs);

    const std::size_t down_row_codes  = static_cast<std::size_t>(intermediate);
    const std::size_t down_row_scales = static_cast<std::size_t>(intermediate / kGroup);
    const auto* down_codes            = reinterpret_cast<const std::int8_t*>(bank.down_codes) +
                           static_cast<std::size_t>(job.expert) * hidden * down_row_codes;
    const auto* down_scales = reinterpret_cast<const std::uint16_t*>(bank.down_scales) +
                              static_cast<std::size_t>(job.expert) * hidden * down_row_scales;
    for (int r = 0; r + 1 < hidden; r += 2) {
        float y0 = 0.0F, y1 = 0.0F;
        dot_two_rows(down_codes + r * down_row_codes, down_scales + r * down_row_scales,
                     down_codes + (r + 1) * down_row_codes, down_scales + (r + 1) * down_row_scales, s.hq, s.hs,
                     intermediate, y0, y1);
        out_column[r] += job.weight * y0;
        out_column[r + 1] += job.weight * y1;
    }
    if (hidden % 2 != 0) {
        const int r = hidden - 1;
        out_column[r] += job.weight * dot_row(down_codes + r * down_row_codes, down_scales + r * down_row_scales,
                                              s.hq, s.hs, intermediate);
    }
}

// ---------------------------------------------------------------------------------------------
// Pool: a round runs in four phases over row-chunked work items so a handful of jobs still
// occupies every core — (0) quantise each token's activation once, (1) gate/up row chunks of
// every expert group (the jobs of one expert, so a weight row read from DRAM serves every
// token routed to it: two rows × two tokens per inner call), (2) quantise each job's
// intermediate, (3) the down row chunks per expert group, accumulating into the token columns
// with atomic adds. Decode rounds (one token per expert) degenerate to the row-chunked GEMV;
// prefill rounds (~10 tokens per expert on a 512-token prompt) run compute-bound instead of
// DRAM-bound. Workers spin briefly before sleeping, which keeps the wake-up latency in the
// microseconds for decode-sized rounds.
// ---------------------------------------------------------------------------------------------

namespace {

constexpr int kPhases      = 4;
constexpr int kPhaseAChunks = 8;  // gate/up: intermediate split into 8 row ranges
constexpr int kPhaseBChunks = 8;  // down: hidden split into 8 row ranges

inline void cpu_relax() {
#if defined(__x86_64__)
    _mm_pause();
#endif
}

} // namespace

struct CpuExpertPool::Impl {
    SparseMoeGeometry geometry;
    std::uint32_t threads = 0;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::mutex run_mutex; // one round at a time: pipeline stages share the pool from driver threads
    std::condition_variable wake;
    std::condition_variable done;
    bool stop = false;
    std::atomic<std::uint64_t> generation{0};
    std::atomic<int> phase{0};
    std::atomic<std::int64_t> next_item{0};
    std::atomic<std::uint32_t> arrived{0};
    std::atomic<std::uint32_t> finished{0};

    // Round state
    const CpuExpertBank* bank   = nullptr;
    const CpuExpertRound* round = nullptr;
    // Per-token quantised activations, per-job intermediates (float, then int8 + scales).
    std::vector<std::int8_t> xq;   // [tokens][hidden]
    std::vector<float> xs;         // [tokens][hidden/32]
    std::vector<float> h;          // [jobs][intermediate]
    std::vector<std::int8_t> hq;   // [jobs][intermediate]
    std::vector<float> hs;         // [jobs][intermediate/32]
    std::vector<std::int32_t> xc;  // [tokens][hidden/32] 128·Σq per group (VNNI compensation)
    std::vector<std::int32_t> hc;  // [jobs][intermediate/32]
    std::vector<std::uint8_t> xu;  // [tokens][hidden] q ^ 0x80 (tile path activations)
    std::vector<std::uint8_t> hu;  // [jobs][intermediate]
    std::vector<std::byte> tiles;  // [threads][tile scratch] repacked expert chunk
    std::size_t tile_bytes = 0;    // per worker
    std::vector<float> x_float;    // [threads][hidden] scratch for phase 0
    // Jobs grouped by expert: `order` lists job indices expert by expert, group g spans
    // order[group_start[g] .. group_start[g+1]).
    std::vector<std::int32_t> order;
    std::vector<std::int32_t> group_start;
    std::vector<std::int32_t> expert_count; // [experts] scratch for the counting sort
    std::int64_t groups = 0;

    void group_jobs() {
        const auto jobs = static_cast<std::int32_t>(round->jobs.size());
        expert_count.assign(static_cast<std::size_t>(geometry.experts), 0);
        for (const CpuExpertJob& job : round->jobs) { ++expert_count[static_cast<std::size_t>(job.expert)]; }
        order.resize(static_cast<std::size_t>(jobs));
        group_start.clear();
        std::int32_t running = 0;
        for (int e = 0; e < geometry.experts; ++e) {
            const std::int32_t n = expert_count[static_cast<std::size_t>(e)];
            if (n == 0) { continue; }
            group_start.push_back(running);
            expert_count[static_cast<std::size_t>(e)] = running; // becomes the scatter cursor
            running += n;
        }
        group_start.push_back(running);
        groups = static_cast<std::int64_t>(group_start.size()) - 1;
        for (std::int32_t j = 0; j < jobs; ++j) {
            const int e = round->jobs[static_cast<std::size_t>(j)].expert;
            order[static_cast<std::size_t>(expert_count[static_cast<std::size_t>(e)]++)] = j;
        }
    }

    std::int64_t items_for_phase(int ph) const {
        const auto jobs = static_cast<std::int64_t>(round->jobs.size());
        if (ph == 0) { return round->tokens; }
        if (ph == 1) { return groups * kPhaseAChunks; }
        if (ph == 2) { return jobs; }
        return groups * kPhaseBChunks;
    }

    void do_item(int ph, std::int64_t item, std::uint32_t worker) {
        const int hidden       = geometry.hidden;
        const int intermediate = geometry.intermediate;
        const int groups_h     = hidden / kGroup;
        const int groups_i     = intermediate / kGroup;
        if (ph == 0) {
            const auto t = static_cast<std::size_t>(item);
            float* xf    = x_float.data() + static_cast<std::size_t>(worker) * hidden;
            const std::uint16_t* x = round->x + t * hidden;
            for (int i = 0; i < hidden; ++i) { xf[i] = bf16_to_float(x[i]); }
            quantise_groups_sums(xf, hidden, xq.data() + t * hidden, xs.data() + t * groups_h,
                                 xc.data() + t * groups_h);
            for (int i = 0; i < hidden; ++i) {
                xu[t * hidden + i] = static_cast<std::uint8_t>(static_cast<std::uint8_t>(xq[t * hidden + i]) ^ 0x80U);
            }
            return;
        }
        if (ph == 1) {
            const auto group = static_cast<std::size_t>(item / kPhaseAChunks);
            const int chunk  = static_cast<int>(item % kPhaseAChunks);
            const std::int32_t g0 = group_start[group], g1 = group_start[group + 1];
            const int expert      = round->jobs[static_cast<std::size_t>(order[static_cast<std::size_t>(g0)])].expert;
            const int per_chunk   = intermediate / kPhaseAChunks;
            const int j0 = chunk * per_chunk;
            const int j1 = chunk == kPhaseAChunks - 1 ? intermediate : j0 + per_chunk;
            const std::size_t gate_rows = static_cast<std::size_t>(2) * intermediate;
            const auto* gate_codes = reinterpret_cast<const std::int8_t*>(bank->gate_up_codes) +
                                     static_cast<std::size_t>(expert) * gate_rows * hidden;
            const auto* gate_scales = reinterpret_cast<const std::uint16_t*>(bank->gate_up_scales) +
                                      static_cast<std::size_t>(expert) * gate_rows * groups_h;
            if (bank->format == ExpertBankFormat::Q4G32AM) {
                const auto* gate4 = reinterpret_cast<const std::uint8_t*>(bank->gate_up_codes) +
                                    static_cast<std::size_t>(expert) * gate_rows *
                                        static_cast<std::size_t>(hidden / 2);
                const auto* gate_mins = reinterpret_cast<const std::uint16_t*>(bank->gate_up_mins) +
                                        static_cast<std::size_t>(expert) * gate_rows * groups_h;
                for (int j = j0; j < j1; ++j) {
                    const std::uint8_t* gc  = gate4 + static_cast<std::size_t>(j) * (hidden / 2);
                    const std::uint16_t* gs = gate_scales + static_cast<std::size_t>(j) * groups_h;
                    const std::uint16_t* gm = gate_mins + static_cast<std::size_t>(j) * groups_h;
                    const std::uint8_t* uc =
                        gate4 + static_cast<std::size_t>(intermediate + j) * (hidden / 2);
                    const std::uint16_t* us =
                        gate_scales + static_cast<std::size_t>(intermediate + j) * groups_h;
                    const std::uint16_t* um =
                        gate_mins + static_cast<std::size_t>(intermediate + j) * groups_h;
                    std::int32_t i = g0;
                    for (; i + 1 < g1; i += 2) {
                        const auto ia = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                        const auto ib =
                            static_cast<std::size_t>(order[static_cast<std::size_t>(i + 1)]);
                        const CpuExpertJob& ja = round->jobs[ia];
                        const CpuExpertJob& jb = round->jobs[ib];
                        float ga = 0.0F, ua = 0.0F, gb = 0.0F, ub = 0.0F;
                        dot_two_rows_two_tokens_q4(
                            gc, gs, gm, uc, us, um,
                            xq.data() + static_cast<std::size_t>(ja.token) * hidden,
                            xs.data() + static_cast<std::size_t>(ja.token) * groups_h,
                            xc.data() + static_cast<std::size_t>(ja.token) * groups_h,
                            xq.data() + static_cast<std::size_t>(jb.token) * hidden,
                            xs.data() + static_cast<std::size_t>(jb.token) * groups_h,
                            xc.data() + static_cast<std::size_t>(jb.token) * groups_h, hidden, ga,
                            ua, gb, ub);
                        h[ia * intermediate + j] = silu(ga) * ua;
                        h[ib * intermediate + j] = silu(gb) * ub;
                    }
                    if (i < g1) {
                        const auto ia = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                        const CpuExpertJob& ja = round->jobs[ia];
                        float ga = 0.0F, ua = 0.0F;
                        dot_two_rows_q4(gc, gs, gm, uc, us, um,
                                        xq.data() + static_cast<std::size_t>(ja.token) * hidden,
                                        xs.data() + static_cast<std::size_t>(ja.token) * groups_h,
                                        xc.data() + static_cast<std::size_t>(ja.token) * groups_h,
                                        hidden, ga, ua);
                        h[ia * intermediate + j] = silu(ga) * ua;
                    }
                }
                return;
            }
            if (kUseTile && g1 - g0 >= kTileMinTokens && (j1 - j0) % kTileRows == 0) {
                // Tile path: repack the chunk's gate rows then up rows (16-row blocks) once,
                // then run every token pair of the group against the tiles.
                std::byte* tile      = tiles.data() + static_cast<std::size_t>(worker) * tile_bytes;
                const int blocks     = (j1 - j0) / kTileRows;
                const std::size_t bb = tile_block_bytes(hidden);
                for (int b = 0; b < blocks; ++b) {
                    repack_tile_block_vnni(gate_codes + static_cast<std::size_t>(j0 + b * kTileRows) * hidden,
                                           gate_scales + static_cast<std::size_t>(j0 + b * kTileRows) * groups_h, hidden,
                                           tile + static_cast<std::size_t>(b) * bb);
                    repack_tile_block_vnni(gate_codes + static_cast<std::size_t>(intermediate + j0 + b * kTileRows) * hidden,
                                           gate_scales + static_cast<std::size_t>(intermediate + j0 + b * kTileRows) * groups_h,
                                           hidden, tile + static_cast<std::size_t>(blocks + b) * bb);
                }
                alignas(64) float ga[kTileRows], gb[kTileRows], ua[kTileRows], ub[kTileRows];
                for (std::int32_t i = g0; i < g1; i += 2) {
                    const auto ia = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                    const auto ib = static_cast<std::size_t>(order[static_cast<std::size_t>(std::min(i + 1, g1 - 1))]);
                    const CpuExpertJob& ja = round->jobs[ia];
                    const CpuExpertJob& jb = round->jobs[ib];
                    const std::uint8_t* xa = xu.data() + static_cast<std::size_t>(ja.token) * hidden;
                    const std::uint8_t* xb = xu.data() + static_cast<std::size_t>(jb.token) * hidden;
                    const float* sa        = xs.data() + static_cast<std::size_t>(ja.token) * groups_h;
                    const float* sb        = xs.data() + static_cast<std::size_t>(jb.token) * groups_h;
                    for (int b = 0; b < blocks; ++b) {
                        dot_tile_block_two_tokens_vnni(tile + static_cast<std::size_t>(b) * bb, hidden, xa, sa, xb, sb, ga, gb);
                        dot_tile_block_two_tokens_vnni(tile + static_cast<std::size_t>(blocks + b) * bb, hidden, xa, sa, xb, sb, ua, ub);
                        for (int r = 0; r < kTileRows; ++r) {
                            h[ia * intermediate + j0 + b * kTileRows + r] = silu(ga[r]) * ua[r];
                            if (i + 1 < g1) { h[ib * intermediate + j0 + b * kTileRows + r] = silu(gb[r]) * ub[r]; }
                        }
                    }
                }
                return;
            }
            for (int j = j0; j < j1; ++j) {
                const std::int8_t* gc = gate_codes + static_cast<std::size_t>(j) * hidden;
                const std::uint16_t* gs = gate_scales + static_cast<std::size_t>(j) * groups_h;
                const std::int8_t* uc = gate_codes + static_cast<std::size_t>(intermediate + j) * hidden;
                const std::uint16_t* us = gate_scales + static_cast<std::size_t>(intermediate + j) * groups_h;
                std::int32_t i = g0;
                for (; i + 1 < g1; i += 2) {
                    const CpuExpertJob& ja = round->jobs[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
                    const CpuExpertJob& jb = round->jobs[static_cast<std::size_t>(order[static_cast<std::size_t>(i + 1)])];
                    float ga = 0.0F, ua = 0.0F, gb = 0.0F, ub = 0.0F;
                    dot_two_rows_two_tokens(gc, gs, uc, us, xq.data() + static_cast<std::size_t>(ja.token) * hidden,
                                            xs.data() + static_cast<std::size_t>(ja.token) * groups_h,
                                            xc.data() + static_cast<std::size_t>(ja.token) * groups_h,
                                            xq.data() + static_cast<std::size_t>(jb.token) * hidden,
                                            xs.data() + static_cast<std::size_t>(jb.token) * groups_h,
                                            xc.data() + static_cast<std::size_t>(jb.token) * groups_h, hidden, ga, ua, gb, ub);
                    h[static_cast<std::size_t>(order[static_cast<std::size_t>(i)]) * intermediate + j]     = silu(ga) * ua;
                    h[static_cast<std::size_t>(order[static_cast<std::size_t>(i + 1)]) * intermediate + j] = silu(gb) * ub;
                }
                if (i < g1) {
                    const CpuExpertJob& ja = round->jobs[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
                    float ga = 0.0F, ua = 0.0F;
                    dot_two_rows(gc, gs, uc, us, xq.data() + static_cast<std::size_t>(ja.token) * hidden,
                                 xs.data() + static_cast<std::size_t>(ja.token) * groups_h, hidden, ga, ua);
                    h[static_cast<std::size_t>(order[static_cast<std::size_t>(i)]) * intermediate + j] = silu(ga) * ua;
                }
            }
            return;
        }
        if (ph == 2) {
            const auto job_index = static_cast<std::size_t>(item);
            quantise_groups_sums(h.data() + job_index * intermediate, intermediate, hq.data() + job_index * intermediate,
                                 hs.data() + job_index * groups_i, hc.data() + job_index * groups_i);
            for (int i = 0; i < intermediate; ++i) {
                hu[job_index * intermediate + i] =
                    static_cast<std::uint8_t>(static_cast<std::uint8_t>(hq[job_index * intermediate + i]) ^ 0x80U);
            }
            return;
        }
        // phase 3: down rows of one expert group against every job of the group.
        const auto group = static_cast<std::size_t>(item / kPhaseBChunks);
        const int chunk  = static_cast<int>(item % kPhaseBChunks);
        const std::int32_t g0 = group_start[group], g1 = group_start[group + 1];
        const int expert      = round->jobs[static_cast<std::size_t>(order[static_cast<std::size_t>(g0)])].expert;
        const int per_chunk   = hidden / kPhaseBChunks;
        const int r0 = chunk * per_chunk;
        const int r1 = chunk == kPhaseBChunks - 1 ? hidden : r0 + per_chunk;
        const auto* down_codes = reinterpret_cast<const std::int8_t*>(bank->down_codes) +
                                 static_cast<std::size_t>(expert) * hidden * intermediate;
        const auto* down_scales = reinterpret_cast<const std::uint16_t*>(bank->down_scales) +
                                  static_cast<std::size_t>(expert) * hidden * groups_i;
        if (bank->format == ExpertBankFormat::Q4G32AM) {
            const auto* down4 = reinterpret_cast<const std::uint8_t*>(bank->down_codes) +
                                static_cast<std::size_t>(expert) * hidden *
                                    static_cast<std::size_t>(intermediate / 2);
            const auto* down_mins = reinterpret_cast<const std::uint16_t*>(bank->down_mins) +
                                    static_cast<std::size_t>(expert) * hidden * groups_i;
            int r = r0;
            for (; r + 1 < r1; r += 2) {
                const std::uint8_t* c0  = down4 + static_cast<std::size_t>(r) * (intermediate / 2);
                const std::uint16_t* s0 = down_scales + static_cast<std::size_t>(r) * groups_i;
                const std::uint16_t* m0 = down_mins + static_cast<std::size_t>(r) * groups_i;
                const std::uint8_t* c1 =
                    down4 + static_cast<std::size_t>(r + 1) * (intermediate / 2);
                const std::uint16_t* s1 = down_scales + static_cast<std::size_t>(r + 1) * groups_i;
                const std::uint16_t* m1 = down_mins + static_cast<std::size_t>(r + 1) * groups_i;
                std::int32_t i = g0;
                for (; i + 1 < g1; i += 2) {
                    const auto ia = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                    const auto ib = static_cast<std::size_t>(order[static_cast<std::size_t>(i + 1)]);
                    const CpuExpertJob& ja = round->jobs[ia];
                    const CpuExpertJob& jb = round->jobs[ib];
                    float y0a = 0.0F, y1a = 0.0F, y0b = 0.0F, y1b = 0.0F;
                    dot_two_rows_two_tokens_q4(c0, s0, m0, c1, s1, m1,
                                               hq.data() + ia * intermediate,
                                               hs.data() + ia * groups_i, hc.data() + ia * groups_i,
                                               hq.data() + ib * intermediate,
                                               hs.data() + ib * groups_i, hc.data() + ib * groups_i,
                                               intermediate, y0a, y1a, y0b, y1b);
                    float* outa = round->out + static_cast<std::size_t>(ja.token) * hidden;
                    float* outb = round->out + static_cast<std::size_t>(jb.token) * hidden;
                    atomic_add(outa + r, ja.weight * y0a);
                    atomic_add(outa + r + 1, ja.weight * y1a);
                    atomic_add(outb + r, jb.weight * y0b);
                    atomic_add(outb + r + 1, jb.weight * y1b);
                }
                if (i < g1) {
                    const auto ia = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                    const CpuExpertJob& ja = round->jobs[ia];
                    float y0 = 0.0F, y1 = 0.0F;
                    dot_two_rows_q4(c0, s0, m0, c1, s1, m1, hq.data() + ia * intermediate,
                                    hs.data() + ia * groups_i, hc.data() + ia * groups_i,
                                    intermediate, y0, y1);
                    float* out = round->out + static_cast<std::size_t>(ja.token) * hidden;
                    atomic_add(out + r, ja.weight * y0);
                    atomic_add(out + r + 1, ja.weight * y1);
                }
            }
            if (r < r1) {
                const std::uint8_t* c0  = down4 + static_cast<std::size_t>(r) * (intermediate / 2);
                const std::uint16_t* s0 = down_scales + static_cast<std::size_t>(r) * groups_i;
                const std::uint16_t* m0 = down_mins + static_cast<std::size_t>(r) * groups_i;
                for (std::int32_t i = g0; i < g1; ++i) {
                    const auto ia = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                    const CpuExpertJob& ja = round->jobs[ia];
                    atomic_add(round->out + static_cast<std::size_t>(ja.token) * hidden + r,
                               ja.weight * dot_row_q4_scalar(c0, s0, m0,
                                                             hq.data() + ia * intermediate,
                                                             hs.data() + ia * groups_i,
                                                             intermediate));
                }
            }
            return;
        }
        if (kUseTile && g1 - g0 >= kTileMinTokens && (r1 - r0) % kTileRows == 0) {
            std::byte* tile      = tiles.data() + static_cast<std::size_t>(worker) * tile_bytes;
            const int blocks     = (r1 - r0) / kTileRows;
            const std::size_t bb = tile_block_bytes(intermediate);
            for (int b = 0; b < blocks; ++b) {
                repack_tile_block_vnni(down_codes + static_cast<std::size_t>(r0 + b * kTileRows) * intermediate,
                                       down_scales + static_cast<std::size_t>(r0 + b * kTileRows) * groups_i, intermediate,
                                       tile + static_cast<std::size_t>(b) * bb);
            }
            alignas(64) float ya[kTileRows], yb[kTileRows];
            for (std::int32_t i = g0; i < g1; i += 2) {
                const auto ia = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                const auto ib = static_cast<std::size_t>(order[static_cast<std::size_t>(std::min(i + 1, g1 - 1))]);
                const CpuExpertJob& ja = round->jobs[ia];
                const CpuExpertJob& jb = round->jobs[ib];
                float* outa = round->out + static_cast<std::size_t>(ja.token) * hidden;
                float* outb = round->out + static_cast<std::size_t>(jb.token) * hidden;
                for (int b = 0; b < blocks; ++b) {
                    dot_tile_block_two_tokens_vnni(tile + static_cast<std::size_t>(b) * bb, intermediate,
                                                   hu.data() + ia * intermediate, hs.data() + ia * groups_i,
                                                   hu.data() + ib * intermediate, hs.data() + ib * groups_i, ya, yb);
                    for (int rr = 0; rr < kTileRows; ++rr) {
                        atomic_add(outa + r0 + b * kTileRows + rr, ja.weight * ya[rr]);
                        if (i + 1 < g1) { atomic_add(outb + r0 + b * kTileRows + rr, jb.weight * yb[rr]); }
                    }
                }
            }
            return;
        }
        // Jobs of different experts share output rows, so accumulate with atomic adds (cheap at
        // this width).
        int r = r0;
        for (; r + 1 < r1; r += 2) {
            const std::int8_t* c0 = down_codes + static_cast<std::size_t>(r) * intermediate;
            const std::uint16_t* s0 = down_scales + static_cast<std::size_t>(r) * groups_i;
            const std::int8_t* c1 = down_codes + static_cast<std::size_t>(r + 1) * intermediate;
            const std::uint16_t* s1 = down_scales + static_cast<std::size_t>(r + 1) * groups_i;
            std::int32_t i = g0;
            for (; i + 1 < g1; i += 2) {
                const auto ia = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                const auto ib = static_cast<std::size_t>(order[static_cast<std::size_t>(i + 1)]);
                const CpuExpertJob& ja = round->jobs[ia];
                const CpuExpertJob& jb = round->jobs[ib];
                float y0a = 0.0F, y1a = 0.0F, y0b = 0.0F, y1b = 0.0F;
                dot_two_rows_two_tokens(c0, s0, c1, s1, hq.data() + ia * intermediate, hs.data() + ia * groups_i,
                                        hc.data() + ia * groups_i, hq.data() + ib * intermediate,
                                        hs.data() + ib * groups_i, hc.data() + ib * groups_i, intermediate, y0a, y1a,
                                        y0b, y1b);
                float* outa = round->out + static_cast<std::size_t>(ja.token) * hidden;
                float* outb = round->out + static_cast<std::size_t>(jb.token) * hidden;
                atomic_add(outa + r, ja.weight * y0a);
                atomic_add(outa + r + 1, ja.weight * y1a);
                atomic_add(outb + r, jb.weight * y0b);
                atomic_add(outb + r + 1, jb.weight * y1b);
            }
            if (i < g1) {
                const auto ia = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                const CpuExpertJob& ja = round->jobs[ia];
                float y0 = 0.0F, y1 = 0.0F;
                dot_two_rows(c0, s0, c1, s1, hq.data() + ia * intermediate, hs.data() + ia * groups_i, intermediate, y0, y1);
                float* out = round->out + static_cast<std::size_t>(ja.token) * hidden;
                atomic_add(out + r, ja.weight * y0);
                atomic_add(out + r + 1, ja.weight * y1);
            }
        }
        if (r < r1) {
            for (std::int32_t i = g0; i < g1; ++i) {
                const auto ia = static_cast<std::size_t>(order[static_cast<std::size_t>(i)]);
                const CpuExpertJob& ja = round->jobs[ia];
                atomic_add(round->out + static_cast<std::size_t>(ja.token) * hidden + r,
                           ja.weight * dot_row(down_codes + static_cast<std::size_t>(r) * intermediate,
                                               down_scales + static_cast<std::size_t>(r) * groups_i,
                                               hq.data() + ia * intermediate, hs.data() + ia * groups_i, intermediate));
            }
        }
    }

    static void atomic_add(float* target, float value) {
        auto* word = reinterpret_cast<std::atomic<std::uint32_t>*>(target);
        std::uint32_t expected = word->load(std::memory_order_relaxed);
        for (;;) {
            float current;
            std::memcpy(&current, &expected, sizeof(current));
            const float next = current + value;
            std::uint32_t desired;
            std::memcpy(&desired, &next, sizeof(desired));
            if (word->compare_exchange_weak(expected, desired, std::memory_order_relaxed)) { return; }
        }
    }

    // Runs one phase: all workers (and the caller) pull items, then barrier.
    void run_phase(int ph, std::uint32_t worker) {
        const std::int64_t items = items_for_phase(ph);
        for (;;) {
            const std::int64_t item = next_item.fetch_add(1, std::memory_order_relaxed);
            if (item >= items) { break; }
            do_item(ph, item, worker);
        }
    }

    std::vector<int> cpus; // explicit pin targets, when given

    void worker_loop(std::uint32_t index, bool pin) {
#if defined(__x86_64__)
        if (pin && index < cpus.size()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus[index], &set);
            sched_setaffinity(0, sizeof(set), &set);
        } else if (pin) {
            // Pin to the index-th CPU the process is allowed to run on (honours a cpuset or
            // numactl --cpunodebind; the first half of a node's allowed set is its physical
            // cores on the usual numbering).
            cpu_set_t allowed;
            CPU_ZERO(&allowed);
            if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
                int seen = -1, chosen = -1;
                for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                    if (CPU_ISSET(cpu, &allowed) && ++seen == static_cast<int>(index)) { chosen = cpu; break; }
                }
                if (chosen >= 0) {
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    CPU_SET(chosen, &set);
                    sched_setaffinity(0, sizeof(set), &set);
                }
            }
        }
#else
        (void)pin;
#endif
        std::uint64_t seen = 0;
        for (;;) {
            // Spin briefly for a new round, then sleep on the condition variable.
            std::uint64_t gen = generation.load(std::memory_order_acquire);
            for (int spin = 0; gen == seen && !stop && spin < 20000; ++spin) {
                cpu_relax();
                gen = generation.load(std::memory_order_acquire);
            }
            if (gen == seen && !stop) {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stop || generation.load(std::memory_order_acquire) != seen; });
                gen = generation.load(std::memory_order_acquire);
            }
            if (stop) { return; }
            seen = gen;
            for (int ph = 0; ph < kPhases; ++ph) {
                // Wait for the phase to open (the coordinator advances `phase` after a barrier).
                while (phase.load(std::memory_order_acquire) != ph + 1) { cpu_relax(); }
                run_phase(ph, index);
                arrived.fetch_add(1, std::memory_order_acq_rel);
                while (phase.load(std::memory_order_acquire) == ph + 1) { cpu_relax(); } // barrier release
            }
            if (finished.fetch_add(1, std::memory_order_acq_rel) + 1 == threads) {
                std::lock_guard<std::mutex> lock(mutex);
                done.notify_all();
            }
        }
    }
};

CpuExpertPool::CpuExpertPool(const SparseMoeGeometry& geometry, Options options)
    : impl_(std::make_unique<Impl>()) {
    require_geometry(geometry);
    impl_->geometry = geometry;
    std::uint32_t threads = options.threads;
    if (!options.cpus.empty()) {
        impl_->cpus = options.cpus;
        threads     = static_cast<std::uint32_t>(options.cpus.size());
    }
    if (threads == 0) {
        // One per physical core on SMT-2 parts, within the CPUs the process may run on.
        unsigned hw = std::thread::hardware_concurrency();
#if defined(__x86_64__)
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) { hw = static_cast<unsigned>(CPU_COUNT(&allowed)); }
#endif
        threads = hw > 1 ? hw / 2 : 1;
    }
    impl_->threads = threads;
    impl_->x_float.resize(static_cast<std::size_t>(threads) * geometry.hidden);
    if (kUseTile) {
        // Per worker: the larger of a gate/up chunk (2 × intermediate/8 rows × hidden) and a
        // down chunk (hidden/8 rows × intermediate), in 16-row tile blocks.
        const std::size_t gate_chunk = static_cast<std::size_t>(2 * (geometry.intermediate / kPhaseAChunks) / kTileRows + 2) *
                                       tile_block_bytes(geometry.hidden);
        const std::size_t down_chunk = static_cast<std::size_t>((geometry.hidden / kPhaseBChunks) / kTileRows + 2) *
                                       tile_block_bytes(geometry.intermediate);
        impl_->tile_bytes = std::max(gate_chunk, down_chunk) + 64;
        impl_->tiles.resize(static_cast<std::size_t>(threads) * impl_->tile_bytes);
    }
    for (std::uint32_t i = 0; i < threads; ++i) {
        impl_->workers.emplace_back([impl = impl_.get(), i, pin = options.pin_threads] { impl->worker_loop(i, pin); });
    }
}

CpuExpertPool::~CpuExpertPool() {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stop = true;
    }
    impl_->wake.notify_all();
    for (auto& t : impl_->workers) { t.join(); }
}

std::uint32_t CpuExpertPool::threads() const noexcept { return impl_->threads; }

void CpuExpertPool::run(const CpuExpertBank& bank, const CpuExpertRound& round) {
    if (round.x == nullptr || round.out == nullptr || round.tokens <= 0) {
        throw std::invalid_argument("cpu_expert_compute: round needs activations, output and tokens");
    }
    for (const CpuExpertJob& job : round.jobs) {
        if (job.token < 0 || job.token >= round.tokens) {
            throw std::invalid_argument("cpu_expert_compute: job token out of range");
        }
        if (job.expert < 0 || job.expert >= impl_->geometry.experts) {
            throw std::invalid_argument("cpu_expert_compute: job expert out of range");
        }
    }
    if (round.jobs.empty()) { return; }
    Impl& impl = *impl_;
    std::lock_guard<std::mutex> run_lock(impl.run_mutex);
    const auto jobs = round.jobs.size();
    impl.xq.resize(static_cast<std::size_t>(round.tokens) * impl.geometry.hidden);
    impl.xs.resize(static_cast<std::size_t>(round.tokens) * (impl.geometry.hidden / kGroup));
    impl.h.resize(jobs * impl.geometry.intermediate);
    impl.hq.resize(jobs * impl.geometry.intermediate);
    impl.hs.resize(jobs * (impl.geometry.intermediate / kGroup));
    impl.xc.resize(static_cast<std::size_t>(round.tokens) * (impl.geometry.hidden / kGroup));
    impl.hc.resize(jobs * (impl.geometry.intermediate / kGroup));
    impl.xu.resize(static_cast<std::size_t>(round.tokens) * impl.geometry.hidden);
    impl.hu.resize(jobs * impl.geometry.intermediate);
    impl.bank  = &bank;
    impl.round = &round;
    impl.group_jobs();
    impl.finished.store(0, std::memory_order_relaxed);
    impl.phase.store(0, std::memory_order_release);
    // Publish the round under the mutex: a worker that has just failed its predicate check
    // cannot miss the notification (it re-checks under the same mutex before blocking).
    {
        std::lock_guard<std::mutex> lock(impl.mutex);
        impl.generation.fetch_add(1, std::memory_order_acq_rel);
    }
    impl.wake.notify_all();
    for (int ph = 0; ph < kPhases; ++ph) {
        impl.next_item.store(0, std::memory_order_relaxed);
        impl.arrived.store(0, std::memory_order_relaxed);
        impl.phase.store(ph + 1, std::memory_order_release);
        std::uint32_t spins = 0;
        while (impl.arrived.load(std::memory_order_acquire) < impl.threads) {
            cpu_relax();
            if ((++spins & 0xFFFFU) == 0U) { impl.wake.notify_all(); } // belt and braces for sleepers
        }
    }
    impl.phase.store(kPhases + 1, std::memory_order_release); // release the last barrier
    std::unique_lock<std::mutex> lock(impl.mutex);
    impl.done.wait(lock, [&] { return impl.finished.load(std::memory_order_acquire) == impl.threads; });
    impl.phase.store(0, std::memory_order_release);
    impl.bank  = nullptr;
    impl.round = nullptr;
}

} // namespace sinfer::ops
