// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for BitsAndBytes NF4 quantization kernels

#include <catch2/catch_all.hpp>

#include <cmath>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kernels/kernels.h"
#include "modules/qlora/qlora_config.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace {

// NF4 codebook values (same as in bnb_quant.cu)
constexpr float NF4_CODEBOOK[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

// Helper to get device properties
cudaDeviceProp get_device_props() {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    return props;
}

// Generate random BF16 data
std::vector<nv_bfloat16> generate_random_bf16(int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    std::vector<nv_bfloat16> data(size);
    for (int i = 0; i < size; ++i) {
        data[i] = __float2bfloat16(dist(gen));
    }
    return data;
}

// Compute expected quantized value (CPU reference)
unsigned char quantize_nf4_cpu(float val) {
    // Binary search for closest NF4 value
    int best_idx = 0;
    float best_dist = std::abs(val - NF4_CODEBOOK[0]);

    for (int i = 1; i < 16; ++i) {
        float dist = std::abs(val - NF4_CODEBOOK[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    return static_cast<unsigned char>(best_idx);
}

// Pack two 4-bit values into one byte
unsigned char pack_nf4(unsigned char hi, unsigned char lo) {
    return (hi << 4) | (lo & 0x0F);
}

// Unpack byte to two 4-bit values
void unpack_nf4(unsigned char packed, unsigned char& hi, unsigned char& lo) {
    hi = (packed >> 4) & 0x0F;
    lo = packed & 0x0F;
}

} // anonymous namespace

TEST_CASE("BnB NF4 quantization basic roundtrip", "[quantization][bnb]") {
    const int M = 64;
    const int K = 128;
    const int block_size = 64;
    const int num_elements = M * K;
    const int packed_size = (num_elements + 1) / 2;
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    auto dp = get_device_props();
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Generate random input data
    auto h_input = generate_random_bf16(num_elements);

    // Allocate device memory
    nv_bfloat16* d_input = nullptr;
    unsigned char* d_quantized = nullptr;
    float* d_absmax = nullptr;
    nv_bfloat16* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_quantized, packed_size));
    CUDA_CHECK(cudaMalloc(&d_absmax, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(nv_bfloat16)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(),
                               num_elements * sizeof(nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));

    // Quantize
    quantize_bnb_nf4(d_quantized, d_absmax, d_input, M, K, block_size, dp, stream);

    // Dequantize
    dequantize_bnb_nf4(d_output, d_quantized, d_absmax, M, K, block_size, dp, stream);

    // Copy results back
    std::vector<nv_bfloat16> h_output(num_elements);
    std::vector<float> h_absmax(num_blocks);

    CUDA_CHECK(cudaMemcpyAsync(h_output.data(), d_output,
                               num_elements * sizeof(nv_bfloat16),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_absmax.data(), d_absmax,
                               num_blocks * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify roundtrip accuracy
    // NF4 quantization has limited precision, so we allow some error
    float max_error = 0.0f;
    float mean_error = 0.0f;

    for (int i = 0; i < num_elements; ++i) {
        float original = __bfloat162float(h_input[i]);
        float recovered = __bfloat162float(h_output[i]);
        float error = std::abs(original - recovered);
        max_error = std::max(max_error, error);
        mean_error += error;
    }
    mean_error /= num_elements;

    INFO("Max roundtrip error: " << max_error);
    INFO("Mean roundtrip error: " << mean_error);

    // NF4 with 4 bits can represent 16 values, typical error should be < 0.15 * absmax
    // For normalized values in [-1, 1], max error should be under 0.2
    REQUIRE(max_error < 0.3f);
    REQUIRE(mean_error < 0.1f);

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_quantized));
    CUDA_CHECK(cudaFree(d_absmax));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("BnB NF4 absmax computation", "[quantization][bnb]") {
    const int block_size = 64;
    const int num_blocks = 4;
    const int num_elements = block_size * num_blocks;
    const int packed_size = (num_elements + 1) / 2;

    auto dp = get_device_props();
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create input with known absmax values per block
    std::vector<nv_bfloat16> h_input(num_elements);
    std::vector<float> expected_absmax(num_blocks);

    // Block 0: max = 0.5
    // Block 1: max = 1.0
    // Block 2: max = 0.25
    // Block 3: max = 2.0
    expected_absmax[0] = 0.5f;
    expected_absmax[1] = 1.0f;
    expected_absmax[2] = 0.25f;
    expected_absmax[3] = 2.0f;

    for (int b = 0; b < num_blocks; ++b) {
        for (int i = 0; i < block_size; ++i) {
            int idx = b * block_size + i;
            float val = (i == 0) ? expected_absmax[b] : 0.1f * expected_absmax[b];
            // Alternate signs
            if (i % 2 == 1) val = -val;
            h_input[idx] = __float2bfloat16(val);
        }
    }

    // Allocate device memory
    nv_bfloat16* d_input = nullptr;
    unsigned char* d_quantized = nullptr;
    float* d_absmax = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_quantized, packed_size));
    CUDA_CHECK(cudaMalloc(&d_absmax, num_blocks * sizeof(float)));

    // Copy input
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(),
                               num_elements * sizeof(nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));

    // Quantize (this computes absmax)
    quantize_bnb_nf4(d_quantized, d_absmax, d_input, 1, num_elements, block_size, dp, stream);

    // Get absmax results
    std::vector<float> h_absmax(num_blocks);
    CUDA_CHECK(cudaMemcpyAsync(h_absmax.data(), d_absmax,
                               num_blocks * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify absmax values
    for (int b = 0; b < num_blocks; ++b) {
        INFO("Block " << b << ": expected " << expected_absmax[b] << ", got " << h_absmax[b]);
        // Allow small tolerance due to BF16 precision
        REQUIRE(std::abs(h_absmax[b] - expected_absmax[b]) < 0.01f);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_quantized));
    CUDA_CHECK(cudaFree(d_absmax));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("BnB NF4 double quantization roundtrip", "[quantization][bnb]") {
    const int M = 128;
    const int K = 256;
    const int block_size = 64;
    const int absmax_group_size = 256;
    const int num_elements = M * K;
    const int packed_size = (num_elements + 1) / 2;
    const int num_absmax = (num_elements + block_size - 1) / block_size;
    const int num_groups = (num_absmax + absmax_group_size - 1) / absmax_group_size;

    auto dp = get_device_props();
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Generate random input data
    auto h_input = generate_random_bf16(num_elements);

    // Allocate device memory
    nv_bfloat16* d_input = nullptr;
    unsigned char* d_quantized = nullptr;
    float* d_absmax_fp32 = nullptr;
    unsigned char* d_absmax_quant = nullptr;
    float* d_absmax_scale = nullptr;
    float* d_absmax_offset = nullptr;
    nv_bfloat16* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_quantized, packed_size));
    CUDA_CHECK(cudaMalloc(&d_absmax_fp32, num_absmax * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_absmax_quant, num_absmax));
    CUDA_CHECK(cudaMalloc(&d_absmax_scale, num_groups * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_absmax_offset, num_groups * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(nv_bfloat16)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(),
                               num_elements * sizeof(nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));

    // Step 1: Quantize to NF4 with FP32 absmax
    quantize_bnb_nf4(d_quantized, d_absmax_fp32, d_input, M, K, block_size, dp, stream);

    // Step 2: Apply double quantization to absmax values
    quantize_absmax_double(d_absmax_quant, d_absmax_scale, d_absmax_offset,
                           d_absmax_fp32, num_absmax, absmax_group_size, dp, stream);

    // Step 3: Dequantize with double quantization
    dequantize_bnb_nf4_double(d_output, d_quantized, d_absmax_quant,
                               d_absmax_scale, d_absmax_offset,
                               M, K, block_size, absmax_group_size, dp, stream);

    // Copy results back
    std::vector<nv_bfloat16> h_output(num_elements);
    CUDA_CHECK(cudaMemcpyAsync(h_output.data(), d_output,
                               num_elements * sizeof(nv_bfloat16),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Verify roundtrip accuracy
    // Double quantization adds some additional error from absmax quantization
    float max_error = 0.0f;
    float mean_error = 0.0f;

    for (int i = 0; i < num_elements; ++i) {
        float original = __bfloat162float(h_input[i]);
        float recovered = __bfloat162float(h_output[i]);
        float error = std::abs(original - recovered);
        max_error = std::max(max_error, error);
        mean_error += error;
    }
    mean_error /= num_elements;

    INFO("Max double-quant roundtrip error: " << max_error);
    INFO("Mean double-quant roundtrip error: " << mean_error);

    // Double quantization introduces additional error, so tolerance is higher
    REQUIRE(max_error < 0.4f);
    REQUIRE(mean_error < 0.15f);

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_quantized));
    CUDA_CHECK(cudaFree(d_absmax_fp32));
    CUDA_CHECK(cudaFree(d_absmax_quant));
    CUDA_CHECK(cudaFree(d_absmax_scale));
    CUDA_CHECK(cudaFree(d_absmax_offset));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("BnB NF4 codebook values match expected", "[quantization][bnb]") {
    // Get the codebook from the kernel
    const float* kernel_codebook = get_nf4_codebook();

    // Verify each value
    for (int i = 0; i < 16; ++i) {
        INFO("Codebook index " << i << ": expected " << NF4_CODEBOOK[i]
             << ", got " << kernel_codebook[i]);
        REQUIRE(std::abs(kernel_codebook[i] - NF4_CODEBOOK[i]) < 1e-6f);
    }
}

TEST_CASE("BnB NF4 handles different block sizes", "[quantization][bnb]") {
    const int M = 256;
    const int K = 256;
    const int num_elements = M * K;

    auto dp = get_device_props();
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Generate random input
    auto h_input = generate_random_bf16(num_elements);

    // Allocate input
    nv_bfloat16* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(),
                               num_elements * sizeof(nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));

    // Test different block sizes
    std::vector<int> block_sizes = {64, 128, 256, 512};

    for (int block_size : block_sizes) {
        SECTION("Block size " + std::to_string(block_size)) {
            const int packed_size = (num_elements + 1) / 2;
            const int num_blocks = (num_elements + block_size - 1) / block_size;

            unsigned char* d_quantized = nullptr;
            float* d_absmax = nullptr;
            nv_bfloat16* d_output = nullptr;

            CUDA_CHECK(cudaMalloc(&d_quantized, packed_size));
            CUDA_CHECK(cudaMalloc(&d_absmax, num_blocks * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(nv_bfloat16)));

            // Quantize and dequantize
            REQUIRE_NOTHROW(quantize_bnb_nf4(d_quantized, d_absmax, d_input,
                                              M, K, block_size, dp, stream));
            REQUIRE_NOTHROW(dequantize_bnb_nf4(d_output, d_quantized, d_absmax,
                                                M, K, block_size, dp, stream));

            // Copy results
            std::vector<nv_bfloat16> h_output(num_elements);
            CUDA_CHECK(cudaMemcpyAsync(h_output.data(), d_output,
                                       num_elements * sizeof(nv_bfloat16),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Verify basic roundtrip
            float max_error = 0.0f;
            for (int i = 0; i < num_elements; ++i) {
                float original = __bfloat162float(h_input[i]);
                float recovered = __bfloat162float(h_output[i]);
                max_error = std::max(max_error, std::abs(original - recovered));
            }

            INFO("Block size " << block_size << " max error: " << max_error);
            REQUIRE(max_error < 0.3f);

            CUDA_CHECK(cudaFree(d_quantized));
            CUDA_CHECK(cudaFree(d_absmax));
            CUDA_CHECK(cudaFree(d_output));
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("BnB NF4 preserves zeros", "[quantization][bnb]") {
    const int block_size = 64;
    const int num_elements = block_size * 2;
    const int packed_size = (num_elements + 1) / 2;
    const int num_blocks = 2;

    auto dp = get_device_props();
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create input with all zeros except a few values
    std::vector<nv_bfloat16> h_input(num_elements, __float2bfloat16(0.0f));
    // Add some non-zero values so absmax is not zero
    h_input[0] = __float2bfloat16(1.0f);
    h_input[block_size] = __float2bfloat16(0.5f);

    // Allocate device memory
    nv_bfloat16* d_input = nullptr;
    unsigned char* d_quantized = nullptr;
    float* d_absmax = nullptr;
    nv_bfloat16* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_quantized, packed_size));
    CUDA_CHECK(cudaMalloc(&d_absmax, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(nv_bfloat16)));

    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(),
                               num_elements * sizeof(nv_bfloat16),
                               cudaMemcpyHostToDevice, stream));

    // Quantize and dequantize
    quantize_bnb_nf4(d_quantized, d_absmax, d_input, 1, num_elements, block_size, dp, stream);
    dequantize_bnb_nf4(d_output, d_quantized, d_absmax, 1, num_elements, block_size, dp, stream);

    // Copy results
    std::vector<nv_bfloat16> h_output(num_elements);
    CUDA_CHECK(cudaMemcpyAsync(h_output.data(), d_output,
                               num_elements * sizeof(nv_bfloat16),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Most values should be near zero (NF4 has 0 as codebook value 7)
    int near_zero_count = 0;
    for (int i = 0; i < num_elements; ++i) {
        float val = __bfloat162float(h_output[i]);
        if (std::abs(val) < 0.1f) {
            near_zero_count++;
        }
    }

    INFO("Near-zero outputs: " << near_zero_count << "/" << num_elements);
    // Most values were zero, so most should be near-zero after roundtrip
    REQUIRE(near_zero_count > num_elements - 4);

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_quantized));
    CUDA_CHECK(cudaFree(d_absmax));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("BnB QLoRAConfig factory method", "[quantization][bnb]") {
    // Test the QLoRAConfig::bnb() factory method
    auto cfg = modules::QLoRAConfig::bnb(64, true);

    REQUIRE(cfg.enabled == true);
    REQUIRE(cfg.strategy == modules::QLoRAQuantStrategy::BitsAndBytes);
    REQUIRE(cfg.scale_config.block_size == 64);
    REQUIRE(cfg.bnb_double_quant == true);
    REQUIRE(cfg.base_dtype == ETensorDType::BYTE);
    REQUIRE(cfg.adapter_dtype == ETensorDType::BF16);

    // Test is_bnb() helper
    REQUIRE(cfg.is_bnb() == true);
    REQUIRE(cfg.is_quantized() == true);
    REQUIRE(cfg.is_fp8() == false);
    REQUIRE(cfg.is_fp4() == false);

    // Test with different parameters
    auto cfg2 = modules::QLoRAConfig::bnb(128, false);
    REQUIRE(cfg2.scale_config.block_size == 128);
    REQUIRE(cfg2.bnb_double_quant == false);
}

TEST_CASE("BnB NF4 quantization performance", "[quantization][bnb]") {
    // Test different matrix sizes typical for LLM weight matrices
    struct TestCase {
        int M;
        int K;
        const char* name;
    };

    std::vector<TestCase> test_cases = {
        {4096, 4096, "4K x 4K"},
        {4096, 11008, "gate_up (4K x 11K)"},
        {11008, 4096, "down (11K x 4K)"},
        {4096, 14336, "large MLP (4K x 14K)"},
    };

    auto dp = get_device_props();
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int block_size = 64;
    const int warmup_iterations = 10;
    const int benchmark_iterations = 100;

    for (const auto& tc : test_cases) {
        SECTION(tc.name) {
            const int M = tc.M;
            const int K = tc.K;
            const int num_elements = M * K;
            const int packed_size = (num_elements + 1) / 2;
            const int num_blocks = (num_elements + block_size - 1) / block_size;

            // Generate random input
            auto h_input = generate_random_bf16(num_elements);

            // Allocate device memory
            nv_bfloat16* d_input = nullptr;
            unsigned char* d_quantized = nullptr;
            float* d_absmax = nullptr;

            CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(nv_bfloat16)));
            CUDA_CHECK(cudaMalloc(&d_quantized, packed_size));
            CUDA_CHECK(cudaMalloc(&d_absmax, num_blocks * sizeof(float)));

            CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(),
                                       num_elements * sizeof(nv_bfloat16),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Warmup
            for (int i = 0; i < warmup_iterations; ++i) {
                quantize_bnb_nf4(d_quantized, d_absmax, d_input, M, K, block_size, dp, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Benchmark
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int i = 0; i < benchmark_iterations; ++i) {
                quantize_bnb_nf4(d_quantized, d_absmax, d_input, M, K, block_size, dp, stream);
            }
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

            float avg_time_us = (elapsed_ms * 1000.0f) / benchmark_iterations;
            float throughput_gb_s = (num_elements * sizeof(nv_bfloat16) / 1e9f) / (avg_time_us / 1e6f);

            INFO(tc.name << " quant: " << avg_time_us << " us/iter, " << throughput_gb_s << " GB/s");

            // Just verify it completes without error
            REQUIRE(avg_time_us > 0.0f);

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaFree(d_quantized));
            CUDA_CHECK(cudaFree(d_absmax));
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("BnB NF4 dequantization performance", "[quantization][bnb]") {
    // Test different matrix sizes typical for LLM weight matrices
    struct TestCase {
        int M;
        int K;
        const char* name;
    };

    std::vector<TestCase> test_cases = {
        {4096, 4096, "4K x 4K"},
        {4096, 11008, "gate_up (4K x 11K)"},
        {11008, 4096, "down (11K x 4K)"},
        {4096, 14336, "large MLP (4K x 14K)"},
    };

    auto dp = get_device_props();
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int block_size = 64;
    const int warmup_iterations = 10;
    const int benchmark_iterations = 100;

    for (const auto& tc : test_cases) {
        SECTION(tc.name) {
            const int M = tc.M;
            const int K = tc.K;
            const int num_elements = M * K;
            const int packed_size = (num_elements + 1) / 2;
            const int num_blocks = (num_elements + block_size - 1) / block_size;

            // Generate random input and quantize it first
            auto h_input = generate_random_bf16(num_elements);

            // Allocate device memory
            nv_bfloat16* d_input = nullptr;
            unsigned char* d_quantized = nullptr;
            float* d_absmax = nullptr;
            nv_bfloat16* d_output = nullptr;

            CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(nv_bfloat16)));
            CUDA_CHECK(cudaMalloc(&d_quantized, packed_size));
            CUDA_CHECK(cudaMalloc(&d_absmax, num_blocks * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(nv_bfloat16)));

            CUDA_CHECK(cudaMemcpyAsync(d_input, h_input.data(),
                                       num_elements * sizeof(nv_bfloat16),
                                       cudaMemcpyHostToDevice, stream));

            // Quantize once
            quantize_bnb_nf4(d_quantized, d_absmax, d_input, M, K, block_size, dp, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Warmup dequantization
            for (int i = 0; i < warmup_iterations; ++i) {
                dequantize_bnb_nf4(d_output, d_quantized, d_absmax, M, K, block_size, dp, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Benchmark dequantization
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int i = 0; i < benchmark_iterations; ++i) {
                dequantize_bnb_nf4(d_output, d_quantized, d_absmax, M, K, block_size, dp, stream);
            }
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

            float avg_time_us = (elapsed_ms * 1000.0f) / benchmark_iterations;
            // Output throughput: n BF16 elements written
            float throughput_gb_s = (num_elements * sizeof(nv_bfloat16) / 1e9f) / (avg_time_us / 1e6f);

            INFO(tc.name << " dequant: " << avg_time_us << " us/iter, " << throughput_gb_s << " GB/s");

            // Just verify it completes without error
            REQUIRE(avg_time_us > 0.0f);

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaFree(d_quantized));
            CUDA_CHECK(cudaFree(d_absmax));
            CUDA_CHECK(cudaFree(d_output));
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}
