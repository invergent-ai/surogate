// Byte-exact verification of the derived quantization planes (surogate,
// PATCHES.md #20/#21/#25): the FP8-e4m3 plane, the NVFP4 plane (row-major
// SF + codes + row scales), the cutlass ATOM-layout folded SF copy, and the
// registry's derive-once semantics. Host references mirror the device
// encoders exactly; requires an sm_120-class GPU (SKIP 77 otherwise).

#include "core/tensor.h"
#include "ops/linear/w8a8/w4fp4_cutlass_gemm.h"
#include "ops/linear/w8a8/w4fp4_plane.h"
#include "ops/linear/w8a8/w8fp8_plane.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kRows = 256;  // two 128-row SF atoms
constexpr int kK    = 512;  // 16 groups of 32; 32 fp4 16-groups

float half_to_float(std::uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

// --- host mirrors of the device encoders (w4fp4_plane.cu) -------------------

int e2m1_encode(float value) {
    const int sign  = value < 0.0f ? 8 : 0;
    const float mag = std::fabs(value);
    int code;
    if (mag < 0.25f) { code = 0; }
    else if (mag < 0.75f) { code = 1; }
    else if (mag < 1.25f) { code = 2; }
    else if (mag < 1.75f) { code = 3; }
    else if (mag < 2.5f) { code = 4; }
    else if (mag < 3.5f) { code = 5; }
    else if (mag < 5.0f) { code = 6; }
    else { code = 7; }
    return sign | code;
}

float ue4m3_decode(int byte) {
    const int e = (byte >> 3) & 0xF;
    const int m = byte & 7;
    if (e == 0) { return std::ldexp(m / 8.0f, -6); }
    return std::ldexp(1.0f + m / 8.0f, e - 7);
}

int ue4m3_encode_up(float value) {
    if (!(value > 0.0f)) { return 0; }
    if (value >= 448.0f) { return 0x7F; }
    int e;
    const float frac = std::frexp(value, &e);
    int exp_field    = (e - 1) + 7;
    int m            = static_cast<int>(std::ceil((2.0f * frac - 1.0f) * 8.0f));
    if (m == 8) { m = 0; ++exp_field; }
    if (exp_field <= 0) {
        m = static_cast<int>(std::ceil(std::ldexp(value, 6) * 8.0f));
        if (m > 7) { return 1 << 3; }
        return m;
    }
    if (exp_field > 15) { return 0x7F; }
    return (exp_field << 3) | m;
}

int ue4m3_encode_rn(float value) {
    if (!(value > 0.0f)) { return 0; }
    if (value >= 448.0f) { return 0x7F; }
    int e;
    const float frac = std::frexp(value, &e);
    int exp_field    = (e - 1) + 7;
    int m            = static_cast<int>(std::nearbyint((2.0f * frac - 1.0f) * 8.0f));
    if (m == 8) { m = 0; ++exp_field; }
    if (exp_field <= 0) {
        m = static_cast<int>(std::nearbyint(std::ldexp(value, 6) * 8.0f));
        if (m > 7) { return 1 << 3; }
        return m < 0 ? 0 : m;
    }
    if (exp_field > 15) { return 0x7F; }
    return (exp_field << 3) | m;
}

std::int64_t sf_atom_offset(int m, int kg, int kg_total) {
    return (static_cast<std::int64_t>(m / 128) * (kg_total / 4) + kg / 4) * 512 + (m % 32) * 16 +
           ((m / 32) & 3) * 4 + (kg & 3);
}

int failures = 0;

void expect(bool condition, const char* label) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "SKIP: no CUDA device\n");
        return 77;
    }
    if (sinfer::ops::detail::w8_device_compute_capability() < 120) {
        std::fprintf(stderr, "SKIP: quant-plane test needs an sm_120-class GPU\n");
        return 77;
    }

    // Synthetic W8G32 weight.
    std::mt19937 rng(23);
    std::vector<std::int8_t> codes(static_cast<std::size_t>(kRows) * kK);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(kRows) * (kK / 32));
    for (auto& c : codes) { c = static_cast<std::int8_t>(int(rng() % 255) - 127); }
    for (auto& s : scales) {
        const float value = 0.001f + 0.05f * (rng() % 1000) / 1000.0f;
        __half h          = __float2half(value);
        std::memcpy(&s, &h, sizeof(s));
    }

    void* d_codes  = nullptr;
    void* d_scales = nullptr;
    cudaMalloc(&d_codes, codes.size());
    cudaMalloc(&d_scales, scales.size() * 2);
    cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales.data(), scales.size() * 2, cudaMemcpyHostToDevice);

    sinfer::Weight weight;
    weight.qtype           = sinfer::QType::W8G32_F16S;
    weight.layout          = sinfer::QuantLayout::RowSplit;
    weight.scale_dtype     = sinfer::DType::FP16;
    weight.group_size      = 32;
    weight.group           = 32;
    weight.ndim            = 2;
    weight.n               = kRows;
    weight.k               = kK;
    weight.shape[0]        = kRows;
    weight.shape[1]        = kK;
    weight.padded_shape[0] = kRows;
    weight.padded_shape[1] = kK;
    weight.qdata           = d_codes;
    weight.scales          = d_scales;

    sinfer::ops::detail::w8fp8_plane_set_enabled(true);
    sinfer::ops::detail::w8_prefill_quant_set_mode(sinfer::ops::detail::PrefillQuantMode::Fp4);
    cudaStream_t stream = nullptr;

    const auto dequant = [&](int row, int i) {
        return float(codes[static_cast<std::size_t>(row) * kK + i]) *
               half_to_float(scales[static_cast<std::size_t>(row) * (kK / 32) + i / 32]);
    };

    // ---- FP8 plane -----------------------------------------------------------
    {
        const auto plane = sinfer::ops::detail::w8fp8_plane_for(weight, stream);
        expect(plane.codes != nullptr, "fp8 plane derives");
        const auto again = sinfer::ops::detail::w8fp8_plane_for(weight, stream);
        expect(again.codes == plane.codes, "fp8 registry caches");
        cudaStreamSynchronize(stream);

        std::vector<std::uint8_t> fp8(static_cast<std::size_t>(kRows) * kK);
        std::vector<float> row_scales(kRows);
        cudaMemcpy(fp8.data(), plane.codes, fp8.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(row_scales.data(), plane.row_scales, kRows * 4, cudaMemcpyDeviceToHost);
        for (int r = 0; r < kRows; ++r) {
            float rowmax = 0.0f;
            for (int i = 0; i < kK; ++i) { rowmax = std::fmax(rowmax, std::fabs(dequant(r, i))); }
            if (rowmax <= 0.0f) { rowmax = 1.0f; }
            expect(row_scales[r] == rowmax, "fp8 row scale == rowmax");
            for (int i = 0; i < kK; ++i) {
                const std::uint8_t want = __nv_cvt_float_to_fp8(
                    dequant(r, i) / rowmax, __NV_SATFINITE, __NV_E4M3);
                if (fp8[static_cast<std::size_t>(r) * kK + i] != want) {
                    expect(false, "fp8 code byte-exact");
                    r = kRows;
                    break;
                }
            }
        }
    }

    // ---- NVFP4 plane (row-major + atom) --------------------------------------
    {
        const auto plane = sinfer::ops::detail::w4fp4_plane_for(weight, stream);
        expect(plane.codes != nullptr, "fp4 plane derives");
        expect(plane.sf_atom != nullptr, "fp4 atom sf present");
        cudaStreamSynchronize(stream);

        std::vector<std::uint8_t> nibbles(static_cast<std::size_t>(kRows) * kK / 2);
        std::vector<std::uint8_t> sf(static_cast<std::size_t>(kRows) * kK / 16);
        std::vector<float> row_scales(kRows);
        const std::size_t atom_bytes = sinfer::ops::detail::w4fp4_sf_atom_bytes(kRows, kK);
        std::vector<std::uint8_t> atom(atom_bytes);
        cudaMemcpy(nibbles.data(), plane.codes, nibbles.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(sf.data(), plane.sf, sf.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(row_scales.data(), plane.row_scales, kRows * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(atom.data(), plane.sf_atom, atom.size(), cudaMemcpyDeviceToHost);

        bool codes_ok = true, sf_ok = true, atom_ok = true, rs_ok = true;
        for (int r = 0; r < kRows; ++r) {
            float rowmax = 0.0f;
            for (int i = 0; i < kK; ++i) { rowmax = std::fmax(rowmax, std::fabs(dequant(r, i))); }
            if (rowmax <= 0.0f) { rowmax = 1.0f; }
            const float row_scale = rowmax / (448.0f * 6.0f);
            if (row_scales[r] != row_scale) { rs_ok = false; }
            for (int g = 0; g < kK / 16; ++g) {
                float gmax = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    gmax = std::fmax(gmax, std::fabs(dequant(r, g * 16 + i)));
                }
                const int sf_byte  = ue4m3_encode_up(gmax / (6.0f * row_scale));
                const float sf_dec = ue4m3_decode(sf_byte);
                if (sf[static_cast<std::size_t>(r) * (kK / 16) + g] != sf_byte) { sf_ok = false; }
                const int atom_want = ue4m3_encode_rn(sf_dec * row_scale);
                if (atom[sf_atom_offset(r, g, kK / 16)] != atom_want) { atom_ok = false; }
                const float inv = sf_dec > 0.0f ? 1.0f / (row_scale * sf_dec) : 0.0f;
                for (int p = 0; p < 8; ++p) {
                    const int lo   = e2m1_encode(dequant(r, g * 16 + 2 * p) * inv);
                    const int hi   = e2m1_encode(dequant(r, g * 16 + 2 * p + 1) * inv);
                    const auto got = nibbles[static_cast<std::size_t>(r) * (kK / 2) + g * 8 + p];
                    if (got != static_cast<std::uint8_t>(lo | (hi << 4))) { codes_ok = false; }
                }
            }
        }
        expect(rs_ok, "fp4 row scales exact");
        expect(sf_ok, "fp4 row-major sf byte-exact");
        expect(atom_ok, "fp4 atom sf placement + folded value byte-exact");
        expect(codes_ok, "fp4 e2m1 nibbles byte-exact");
    }

    cudaFree(d_codes);
    cudaFree(d_scales);
    if (failures == 0) { std::puts("OK w8 quant planes"); }
    return failures == 0 ? 0 : 1;
}
