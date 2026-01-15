// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Higher-level / regression tests for MoE paths that aren't covered well by test-moe.cu:
// - QLoRA weight managers router-gate layout (transpose regression)
// - Selective/compact expert indexing for grouped GEMMs
// - MoE backward kernels (combine + permute)

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "kernels/kernels.h"
#include "modules/lora/lora_config.h"
#include "modules/model_factory.h"
#include "modules/moe/moe_block.h"
#include "modules/qlora/bnb_weights.h"
#include "modules/qlora/fp4_weights.h"
#include "modules/qlora/fp8_weights.h"
#include "modules/qlora/qlora_config.h"
#include "modules/weights/weight_manager.h"
#include "models/qwen3moe/config.h"
#include "training/runtime_options.h"
#include "utilities/allocator.h"
#include "utilities/safetensors.h"
#include "utilities/utils.h"

namespace {

bool cuda_available() {
    int device_count = 0;
    return (cudaGetDeviceCount(&device_count) == cudaSuccess) && (device_count > 0);
}

cudaDeviceProp get_device_props() {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    return props;
}

bool fp8_supported() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return false;
    }
    // FP8 tensor cores are available on Ada (SM 8.9) and newer.
    return (prop.major > 8) || (prop.major == 8 && prop.minor >= 9);
}

bool fp4_supported() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return false;
    }
    // NVFP4 is intended for Blackwell (SM100+) and newer.
    return prop.major >= 10;
}

struct CublasHandle {
    cublasHandle_t handle = nullptr;
    CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle)); }
    ~CublasHandle() { if (handle) cublasDestroy(handle); }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
};

struct CudaStream {
    cudaStream_t stream = nullptr;
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream)); }
    ~CudaStream() { if (stream) cudaStreamDestroy(stream); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

template<typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    std::size_t n = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) : n(count) {
        if (n) {
            CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        }
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), n(other.n) {
        other.ptr = nullptr;
        other.n = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) return *this;
        if (ptr) cudaFree(ptr);
        ptr = other.ptr;
        n = other.n;
        other.ptr = nullptr;
        other.n = 0;
        return *this;
    }

    [[nodiscard]] T* data() { return ptr; }
    [[nodiscard]] const T* data() const { return ptr; }
    [[nodiscard]] std::size_t size() const { return n; }
};

uint16_t bf16_bits(nv_bfloat16 v) {
    uint16_t bits_val = 0;
    std::memcpy(&bits_val, &v, sizeof(bits_val));
    return bits_val;
}

std::string make_temp_path(std::string_view stem) {
    auto dir = std::filesystem::temp_directory_path();
    std::mt19937_64 gen(1234567ULL);
    std::uniform_int_distribution<uint64_t> dist;
    std::string fname = std::string(stem) + "_" + std::to_string(dist(gen)) + ".safetensors";
    return (dir / fname).string();
}

struct TempFile {
    std::string path;
    explicit TempFile(std::string p) : path(std::move(p)) {}
    ~TempFile() {
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
};

Tensor make_tensor(ETensorDType dtype, const std::vector<long>& shape, std::byte* data) {
    Tensor t{};
    t.DType = dtype;
    t.Rank = static_cast<int>(shape.size());
    std::fill(t.Sizes.begin(), t.Sizes.end(), 1);
    for (int i = 0; i < t.Rank; ++i) t.Sizes[i] = shape[static_cast<std::size_t>(i)];
    t.Data = data;
    int dev = 0;
    cudaGetDevice(&dev);
    t.Device = dev;
    return t;
}

struct MinimalMoESafetensorsConfig {
    int vocab_size = 64;
    int hidden_size = 8;
    int intermediate_size = 16;
    int moe_intermediate_size = 6;
    int num_query_heads = 2;
    int num_kv_heads = 1;
    int head_size = 4;
    int num_experts = 4;
};

// Writes a minimal Qwen3-MoE-like safetensors file with exactly 1 layer and the keys needed by
// BnB/FP8/FP4 MoE QLoRA weight managers.
TempFile write_minimal_qwen3_moe_safetensors(const MinimalMoESafetensorsConfig& cfg,
                                             const std::vector<nv_bfloat16>& router_gate_host) {
    const int hidden = cfg.hidden_size;
    const int num_experts = cfg.num_experts;
    const int moe_inter = cfg.moe_intermediate_size;
    const int qkv_out = (cfg.num_query_heads + 2 * cfg.num_kv_heads) * cfg.head_size;

    REQUIRE(static_cast<int>(router_gate_host.size()) == num_experts * hidden);

    // Create deterministic but non-zero weights to keep quantization codepaths happy.
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    auto fill_bf16 = [&](int elems) {
        std::vector<nv_bfloat16> v(static_cast<std::size_t>(elems));
        for (int i = 0; i < elems; ++i) v[static_cast<std::size_t>(i)] = __float2bfloat16(dist(gen));
        return v;
    };

    // Device buffers for all tensors we write.
    DeviceBuffer<nv_bfloat16> d_embed(static_cast<std::size_t>(cfg.vocab_size) * hidden);
    DeviceBuffer<nv_bfloat16> d_norm(hidden);
    DeviceBuffer<nv_bfloat16> d_qkv(static_cast<std::size_t>(qkv_out) * hidden);
    DeviceBuffer<nv_bfloat16> d_o(static_cast<std::size_t>(hidden) * (cfg.num_query_heads * cfg.head_size));
    DeviceBuffer<nv_bfloat16> d_ln1(hidden);
    DeviceBuffer<nv_bfloat16> d_ln2(hidden);
    DeviceBuffer<nv_bfloat16> d_router_gate(static_cast<std::size_t>(num_experts) * hidden);

    CUDA_CHECK(cudaMemcpy(d_embed.data(), fill_bf16(static_cast<int>(d_embed.size())).data(),
                          d_embed.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm.data(), fill_bf16(hidden).data(),
                          hidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qkv.data(), fill_bf16(static_cast<int>(d_qkv.size())).data(),
                          d_qkv.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_o.data(), fill_bf16(static_cast<int>(d_o.size())).data(),
                          d_o.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln1.data(), fill_bf16(hidden).data(),
                          hidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln2.data(), fill_bf16(hidden).data(),
                          hidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_router_gate.data(), router_gate_host.data(),
                          router_gate_host.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    // Experts: store fused gate_up_proj (2*moe_inter, hidden) with [up | gate] row layout.
    // We'll just fill with random BF16 values.
    std::vector<DeviceBuffer<nv_bfloat16>> d_expert_gate_up;
    std::vector<DeviceBuffer<nv_bfloat16>> d_expert_down;
    d_expert_gate_up.reserve(static_cast<std::size_t>(num_experts));
    d_expert_down.reserve(static_cast<std::size_t>(num_experts));

    for (int e = 0; e < num_experts; ++e) {
        d_expert_gate_up.emplace_back(static_cast<std::size_t>(2 * moe_inter) * hidden);
        d_expert_down.emplace_back(static_cast<std::size_t>(hidden) * moe_inter);

        auto h_gate_up = fill_bf16(static_cast<int>(d_expert_gate_up.back().size()));
        auto h_down = fill_bf16(static_cast<int>(d_expert_down.back().size()));

        CUDA_CHECK(cudaMemcpy(d_expert_gate_up.back().data(), h_gate_up.data(),
                              h_gate_up.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_expert_down.back().data(), h_down.data(),
                              h_down.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    }

    const std::string path = make_temp_path("surogate_test_moe");
    TempFile tmp(path);

    SafeTensorWriter writer(path);

    auto reg = [&](const std::string& name, const Tensor& t) {
        writer.register_tensor(name, TensorShard(t));
    };

    // Non-block
    reg("model.embed_tokens.weight",
        make_tensor(ETensorDType::BF16, {cfg.vocab_size, hidden}, reinterpret_cast<std::byte*>(d_embed.data())));
    reg("model.norm.weight",
        make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_norm.data())));

    // Block 0
    reg("model.layers.0.self_attn.qkv_proj.weight",
        make_tensor(ETensorDType::BF16, {qkv_out, hidden}, reinterpret_cast<std::byte*>(d_qkv.data())));
    reg("model.layers.0.self_attn.o_proj.weight",
        make_tensor(ETensorDType::BF16, {hidden, cfg.num_query_heads * cfg.head_size}, reinterpret_cast<std::byte*>(d_o.data())));
    reg("model.layers.0.input_layernorm.weight",
        make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_ln1.data())));
    reg("model.layers.0.post_attention_layernorm.weight",
        make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_ln2.data())));
    reg("model.layers.0.mlp.gate.weight",
        make_tensor(ETensorDType::BF16, {num_experts, hidden}, reinterpret_cast<std::byte*>(d_router_gate.data())));

    for (int e = 0; e < num_experts; ++e) {
        const std::string prefix = "model.layers.0.mlp.experts." + std::to_string(e);
        reg(prefix + ".gate_up_proj.weight",
            make_tensor(ETensorDType::BF16, {2 * moe_inter, hidden}, reinterpret_cast<std::byte*>(d_expert_gate_up[static_cast<std::size_t>(e)].data())));
        reg(prefix + ".down_proj.weight",
            make_tensor(ETensorDType::BF16, {hidden, moe_inter}, reinterpret_cast<std::byte*>(d_expert_down[static_cast<std::size_t>(e)].data())));
    }

    writer.prepare_metadata(nullptr);

    auto write = [&](const std::string& name, const Tensor& t) {
        writer.write_tensor(name, TensorShard(t), nullptr);
    };

    write("model.embed_tokens.weight",
          make_tensor(ETensorDType::BF16, {cfg.vocab_size, hidden}, reinterpret_cast<std::byte*>(d_embed.data())));
    write("model.norm.weight",
          make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_norm.data())));
    write("model.layers.0.self_attn.qkv_proj.weight",
          make_tensor(ETensorDType::BF16, {qkv_out, hidden}, reinterpret_cast<std::byte*>(d_qkv.data())));
    write("model.layers.0.self_attn.o_proj.weight",
          make_tensor(ETensorDType::BF16, {hidden, cfg.num_query_heads * cfg.head_size}, reinterpret_cast<std::byte*>(d_o.data())));
    write("model.layers.0.input_layernorm.weight",
          make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_ln1.data())));
    write("model.layers.0.post_attention_layernorm.weight",
          make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_ln2.data())));
    write("model.layers.0.mlp.gate.weight",
          make_tensor(ETensorDType::BF16, {num_experts, hidden}, reinterpret_cast<std::byte*>(d_router_gate.data())));

    for (int e = 0; e < num_experts; ++e) {
        const std::string prefix = "model.layers.0.mlp.experts." + std::to_string(e);
        write(prefix + ".gate_up_proj.weight",
              make_tensor(ETensorDType::BF16, {2 * moe_inter, hidden}, reinterpret_cast<std::byte*>(d_expert_gate_up[static_cast<std::size_t>(e)].data())));
        write(prefix + ".down_proj.weight",
              make_tensor(ETensorDType::BF16, {hidden, moe_inter}, reinterpret_cast<std::byte*>(d_expert_down[static_cast<std::size_t>(e)].data())));
    }

    writer.finalize(nullptr);

    return tmp;
}

// Writes a 2-layer Qwen3 MoE checkpoint (BF16) to exercise QLoRA+MoE paths where
// selective expert dequantization must be re-applied in backward for each layer.
TempFile write_minimal_qwen3_moe_safetensors_2layers(const MinimalMoESafetensorsConfig& cfg,
                                                     const std::vector<nv_bfloat16>& router_gate_l0,
                                                     const std::vector<nv_bfloat16>& router_gate_l1) {
    const int hidden = cfg.hidden_size;
    const int num_experts = cfg.num_experts;
    const int moe_inter = cfg.moe_intermediate_size;
    const int qkv_out = (cfg.num_query_heads + 2 * cfg.num_kv_heads) * cfg.head_size;

    REQUIRE(static_cast<int>(router_gate_l0.size()) == num_experts * hidden);
    REQUIRE(static_cast<int>(router_gate_l1.size()) == num_experts * hidden);

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    auto fill_bf16 = [&](int elems, float value) {
        std::vector<nv_bfloat16> v(static_cast<std::size_t>(elems));
        for (int i = 0; i < elems; ++i) v[static_cast<std::size_t>(i)] = __float2bfloat16(value);
        return v;
    };
    auto fill_rand_bf16 = [&](int elems) {
        std::vector<nv_bfloat16> v(static_cast<std::size_t>(elems));
        for (int i = 0; i < elems; ++i) v[static_cast<std::size_t>(i)] = __float2bfloat16(dist(gen));
        return v;
    };

    // Non-block: embeddings + final norm.
    DeviceBuffer<nv_bfloat16> d_embed(static_cast<std::size_t>(cfg.vocab_size) * hidden);
    DeviceBuffer<nv_bfloat16> d_norm(hidden);

    // Build embeddings: random, but make token id 1 a deterministic "all ones" vector so routing is stable.
    auto h_embed = fill_rand_bf16(static_cast<int>(d_embed.size()));
    for (int c = 0; c < hidden; ++c) {
        h_embed[static_cast<std::size_t>(1) * hidden + c] = __float2bfloat16(1.0f);
    }
    CUDA_CHECK(cudaMemcpy(d_embed.data(), h_embed.data(),
                          d_embed.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    auto h_norm = fill_bf16(hidden, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_norm.data(), h_norm.data(),
                          hidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    // Shared attention weights across both layers (kept zeroed).
    DeviceBuffer<nv_bfloat16> d_qkv(static_cast<std::size_t>(qkv_out) * hidden);
    DeviceBuffer<nv_bfloat16> d_o(static_cast<std::size_t>(hidden) * (cfg.num_query_heads * cfg.head_size));
    DeviceBuffer<nv_bfloat16> d_expert_gate_up(static_cast<std::size_t>(2 * moe_inter) * hidden);
    DeviceBuffer<nv_bfloat16> d_expert_down(static_cast<std::size_t>(hidden) * moe_inter);
    CUDA_CHECK(cudaMemcpy(d_qkv.data(), fill_bf16(static_cast<int>(d_qkv.size()), 0.0f).data(),
                          d_qkv.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_o.data(), fill_bf16(static_cast<int>(d_o.size()), 0.0f).data(),
                          d_o.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    // NOTE: keep experts non-zero so LoRA gradients don't vanish through SwiGLU at exactly 0.
    // Small positive weights preserve deterministic routing (inputs remain mostly positive).
    constexpr float kExpertInit = 0.05f;
    CUDA_CHECK(cudaMemcpy(d_expert_gate_up.data(), fill_bf16(static_cast<int>(d_expert_gate_up.size()), kExpertInit).data(),
                          d_expert_gate_up.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expert_down.data(), fill_bf16(static_cast<int>(d_expert_down.size()), kExpertInit).data(),
                          d_expert_down.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    // Layernorm weights: ones.
    DeviceBuffer<nv_bfloat16> d_ln1(hidden);
    DeviceBuffer<nv_bfloat16> d_ln2(hidden);
    auto h_ln = fill_bf16(hidden, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_ln1.data(), h_ln.data(), hidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ln2.data(), h_ln.data(), hidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    // Router gates per layer (BF16).
    DeviceBuffer<nv_bfloat16> d_router_gate0(static_cast<std::size_t>(num_experts) * hidden);
    DeviceBuffer<nv_bfloat16> d_router_gate1(static_cast<std::size_t>(num_experts) * hidden);
    CUDA_CHECK(cudaMemcpy(d_router_gate0.data(), router_gate_l0.data(),
                          router_gate_l0.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_router_gate1.data(), router_gate_l1.data(),
                          router_gate_l1.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

    const std::string path = make_temp_path("surogate_test_moe_2layers");
    TempFile tmp(path);
    SafeTensorWriter writer(path);

    auto reg = [&](const std::string& name, const Tensor& t) {
        writer.register_tensor(name, TensorShard(t));
    };
    auto write = [&](const std::string& name, const Tensor& t) {
        writer.write_tensor(name, TensorShard(t), nullptr);
    };

    // Register tensors.
    reg("model.embed_tokens.weight",
        make_tensor(ETensorDType::BF16, {cfg.vocab_size, hidden}, reinterpret_cast<std::byte*>(d_embed.data())));
    reg("model.norm.weight",
        make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_norm.data())));

    for (int layer = 0; layer < 2; ++layer) {
        const std::string lp = "model.layers." + std::to_string(layer);
        reg(lp + ".self_attn.qkv_proj.weight",
            make_tensor(ETensorDType::BF16, {qkv_out, hidden}, reinterpret_cast<std::byte*>(d_qkv.data())));
        reg(lp + ".self_attn.o_proj.weight",
            make_tensor(ETensorDType::BF16, {hidden, cfg.num_query_heads * cfg.head_size}, reinterpret_cast<std::byte*>(d_o.data())));
        reg(lp + ".input_layernorm.weight",
            make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_ln1.data())));
        reg(lp + ".post_attention_layernorm.weight",
            make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_ln2.data())));
        reg(lp + ".mlp.gate.weight",
            make_tensor(ETensorDType::BF16, {num_experts, hidden},
                        reinterpret_cast<std::byte*>((layer == 0) ? d_router_gate0.data() : d_router_gate1.data())));

        for (int e = 0; e < num_experts; ++e) {
            const std::string ep = lp + ".mlp.experts." + std::to_string(e);
            reg(ep + ".gate_up_proj.weight",
                make_tensor(ETensorDType::BF16, {2 * moe_inter, hidden}, reinterpret_cast<std::byte*>(d_expert_gate_up.data())));
            reg(ep + ".down_proj.weight",
                make_tensor(ETensorDType::BF16, {hidden, moe_inter}, reinterpret_cast<std::byte*>(d_expert_down.data())));
        }
    }

    writer.prepare_metadata(nullptr);

    // Write tensors.
    write("model.embed_tokens.weight",
          make_tensor(ETensorDType::BF16, {cfg.vocab_size, hidden}, reinterpret_cast<std::byte*>(d_embed.data())));
    write("model.norm.weight",
          make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_norm.data())));

    for (int layer = 0; layer < 2; ++layer) {
        const std::string lp = "model.layers." + std::to_string(layer);
        write(lp + ".self_attn.qkv_proj.weight",
              make_tensor(ETensorDType::BF16, {qkv_out, hidden}, reinterpret_cast<std::byte*>(d_qkv.data())));
        write(lp + ".self_attn.o_proj.weight",
              make_tensor(ETensorDType::BF16, {hidden, cfg.num_query_heads * cfg.head_size}, reinterpret_cast<std::byte*>(d_o.data())));
        write(lp + ".input_layernorm.weight",
              make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_ln1.data())));
        write(lp + ".post_attention_layernorm.weight",
              make_tensor(ETensorDType::BF16, {hidden}, reinterpret_cast<std::byte*>(d_ln2.data())));
        write(lp + ".mlp.gate.weight",
              make_tensor(ETensorDType::BF16, {num_experts, hidden},
                          reinterpret_cast<std::byte*>((layer == 0) ? d_router_gate0.data() : d_router_gate1.data())));

        for (int e = 0; e < num_experts; ++e) {
            const std::string ep = lp + ".mlp.experts." + std::to_string(e);
            write(ep + ".gate_up_proj.weight",
                  make_tensor(ETensorDType::BF16, {2 * moe_inter, hidden}, reinterpret_cast<std::byte*>(d_expert_gate_up.data())));
            write(ep + ".down_proj.weight",
                  make_tensor(ETensorDType::BF16, {hidden, moe_inter}, reinterpret_cast<std::byte*>(d_expert_down.data())));
        }
    }

    writer.finalize(nullptr);
    return tmp;
}

std::vector<nv_bfloat16> build_router_gate_pattern(int num_experts, int hidden) {
    std::vector<nv_bfloat16> gate(static_cast<std::size_t>(num_experts) * hidden);
    for (int e = 0; e < num_experts; ++e) {
        for (int c = 0; c < hidden; ++c) {
            float v = static_cast<float>(e * 10 + c);
            gate[static_cast<std::size_t>(e) * hidden + c] = __float2bfloat16(v);
        }
    }
    return gate;
}

template<typename T>
std::vector<T> copy_device_to_host(const T* d, std::size_t n) {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// CPU helpers for MoE index building / backward refs.
void compute_expert_counts_cpu(std::vector<int>& counts, const std::vector<int>& expert_indices,
                               int num_tokens, int top_k, int num_experts) {
    counts.assign(static_cast<std::size_t>(num_experts), 0);
    for (int i = 0; i < num_tokens * top_k; ++i) {
        const int e = expert_indices[static_cast<std::size_t>(i)];
        if (0 <= e && e < num_experts) counts[static_cast<std::size_t>(e)]++;
    }
}

void build_indices_cpu(std::vector<int>& gather_indices, std::vector<int>& scatter_indices,
                       const std::vector<int>& expert_indices, const std::vector<int>& expert_counts,
                       int num_tokens, int top_k, int num_experts) {
    std::vector<int> expert_offsets(static_cast<std::size_t>(num_experts + 1), 0);
    for (int e = 0; e < num_experts; ++e) {
        expert_offsets[static_cast<std::size_t>(e + 1)] = expert_offsets[static_cast<std::size_t>(e)] + expert_counts[static_cast<std::size_t>(e)];
    }
    std::vector<int> expert_pos(static_cast<std::size_t>(num_experts), 0);

    const int total = num_tokens * top_k;
    gather_indices.assign(static_cast<std::size_t>(total), 0);
    scatter_indices.assign(static_cast<std::size_t>(total), 0);
    for (int idx = 0; idx < total; ++idx) {
        const int e = expert_indices[static_cast<std::size_t>(idx)];
        const int dst = expert_offsets[static_cast<std::size_t>(e)] + expert_pos[static_cast<std::size_t>(e)];
        gather_indices[static_cast<std::size_t>(dst)] = idx;
        scatter_indices[static_cast<std::size_t>(idx)] = dst;
        expert_pos[static_cast<std::size_t>(e)]++;
    }
}

} // namespace

TEST_CASE("moe_remap_expert_indices maps global->compact indices", "[moe][selective]") {
    if (!cuda_available()) SKIP("CUDA not available");

    CudaStream stream;

    constexpr int num_experts_total = 6;
    constexpr int num_tokens = 4;
    constexpr int top_k = 3;

    std::vector<int> h_expert_indices = {
        1, 4, 5,
        4, 5, 1,
        2, 1, 4,
        0, 5, 3
    };
    REQUIRE(static_cast<int>(h_expert_indices.size()) == num_tokens * top_k);

    std::vector<int> h_expert_to_compact(num_experts_total, -1);
    // Active experts: 1->0, 4->1, 5->2 (others inactive -> -1)
    h_expert_to_compact[1] = 0;
    h_expert_to_compact[4] = 1;
    h_expert_to_compact[5] = 2;

    DeviceBuffer<int> d_expert_indices(h_expert_indices.size());
    DeviceBuffer<int> d_expert_to_compact(h_expert_to_compact.size());
    DeviceBuffer<int> d_remapped(h_expert_indices.size());

    CUDA_CHECK(cudaMemcpyAsync(d_expert_indices.data(), h_expert_indices.data(),
                               h_expert_indices.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_expert_to_compact.data(), h_expert_to_compact.data(),
                               h_expert_to_compact.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream.stream));

    moe_remap_expert_indices(d_remapped.data(),
                             d_expert_indices.data(),
                             d_expert_to_compact.data(),
                             num_tokens, top_k, stream.stream);
    CUDA_CHECK(cudaStreamSynchronize(stream.stream));

    auto h_remapped = copy_device_to_host(d_remapped.data(), h_expert_indices.size());

    for (int i = 0; i < num_tokens * top_k; ++i) {
        int global = h_expert_indices[static_cast<std::size_t>(i)];
        int expected = h_expert_to_compact[static_cast<std::size_t>(global)];
        REQUIRE(h_remapped[static_cast<std::size_t>(i)] == expected);
    }
}

TEST_CASE("moe_grouped_gemm_gate_up supports compact expert weights", "[moe][grouped_gemm]") {
    if (!cuda_available()) SKIP("CUDA not available");

    CudaStream stream;
    CublasHandle cublas;

    const int num_experts = 6;
    const int hidden = 32;      // K
    const int D = 16;
    const int M = 2 * D;

    // Token counts per expert (some zero) -> active experts are those with count>0.
    std::vector<int> counts = {0, 3, 0, 2, 0, 1};
    REQUIRE(static_cast<int>(counts.size()) == num_experts);
    std::vector<int> offsets(num_experts + 1, 0);
    for (int e = 0; e < num_experts; ++e) offsets[e + 1] = offsets[e] + counts[e];
    const int total_tokens = offsets.back();

    std::vector<int> active_experts;
    for (int e = 0; e < num_experts; ++e) if (counts[e] > 0) active_experts.push_back(e);
    const int num_active = static_cast<int>(active_experts.size());

    // Random input (total_tokens, hidden)
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<float> h_input(static_cast<std::size_t>(total_tokens) * hidden);
    for (auto& v : h_input) v = dist(gen);

    // Full weights (num_experts, M, hidden)
    std::vector<float> h_weights_full(static_cast<std::size_t>(num_experts) * M * hidden);
    for (auto& v : h_weights_full) v = dist(gen);

    // Compact weights (num_active, M, hidden) in the order of active_experts.
    std::vector<float> h_weights_compact(static_cast<std::size_t>(num_active) * M * hidden);
    for (int i = 0; i < num_active; ++i) {
        int ge = active_experts[static_cast<std::size_t>(i)];
        const float* src = h_weights_full.data() + static_cast<std::size_t>(ge) * M * hidden;
        float* dst = h_weights_compact.data() + static_cast<std::size_t>(i) * M * hidden;
        std::memcpy(dst, src, static_cast<std::size_t>(M) * hidden * sizeof(float));
    }

    DeviceBuffer<float> d_input(h_input.size());
    DeviceBuffer<float> d_weights_full(h_weights_full.size());
    DeviceBuffer<float> d_weights_compact(h_weights_compact.size());
    DeviceBuffer<int> d_offsets(offsets.size());

    DeviceBuffer<float> d_out_full(static_cast<std::size_t>(total_tokens) * M);
    DeviceBuffer<float> d_out_compact(static_cast<std::size_t>(total_tokens) * M);
    CUDA_CHECK(cudaMemsetAsync(d_out_full.data(), 0, d_out_full.size() * sizeof(float), stream.stream));
    CUDA_CHECK(cudaMemsetAsync(d_out_compact.data(), 0, d_out_compact.size() * sizeof(float), stream.stream));

    CUDA_CHECK(cudaMemcpyAsync(d_input.data(), h_input.data(), h_input.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weights_full.data(), h_weights_full.data(), h_weights_full.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weights_compact.data(), h_weights_compact.data(), h_weights_compact.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_offsets.data(), offsets.data(), offsets.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream.stream));

    // Full run (no active indices)
    moe_grouped_gemm_gate_up(d_out_full.data(), d_input.data(), d_weights_full.data(),
                             d_offsets.data(), num_experts, hidden, D,
                             cublas.handle, stream.stream, offsets.data());

    // Compact run (active experts only, compact weight buffer)
    moe_grouped_gemm_gate_up(d_out_compact.data(), d_input.data(), d_weights_compact.data(),
                             d_offsets.data(), num_experts, hidden, D,
                             cublas.handle, stream.stream, offsets.data(),
                             active_experts.data(), /*weight_is_compact=*/true, /*num_active_experts=*/num_active);

    CUDA_CHECK(cudaStreamSynchronize(stream.stream));

    auto h_out_full = copy_device_to_host(d_out_full.data(), d_out_full.size());
    auto h_out_compact = copy_device_to_host(d_out_compact.data(), d_out_compact.size());

    for (std::size_t i = 0; i < h_out_full.size(); ++i) {
        REQUIRE(h_out_compact[i] == Catch::Approx(h_out_full[i]).margin(1e-4f));
    }
}

TEST_CASE("moe_grouped_gemm_down supports compact expert weights", "[moe][grouped_gemm]") {
    if (!cuda_available()) SKIP("CUDA not available");

    CudaStream stream;
    CublasHandle cublas;

    const int num_experts = 6;
    const int hidden = 32; // C
    const int D = 16;

    // Token counts per expert (some zero)
    std::vector<int> counts = {1, 0, 2, 0, 0, 3};
    REQUIRE(static_cast<int>(counts.size()) == num_experts);
    std::vector<int> offsets(num_experts + 1, 0);
    for (int e = 0; e < num_experts; ++e) offsets[e + 1] = offsets[e] + counts[e];
    const int total_tokens = offsets.back();

    std::vector<int> active_experts;
    for (int e = 0; e < num_experts; ++e) if (counts[e] > 0) active_experts.push_back(e);
    const int num_active = static_cast<int>(active_experts.size());

    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // Input (total_tokens, D)
    std::vector<float> h_input(static_cast<std::size_t>(total_tokens) * D);
    for (auto& v : h_input) v = dist(gen);

    // Full weights (num_experts, hidden, D)
    std::vector<float> h_weights_full(static_cast<std::size_t>(num_experts) * hidden * D);
    for (auto& v : h_weights_full) v = dist(gen);

    // Compact weights (num_active, hidden, D)
    std::vector<float> h_weights_compact(static_cast<std::size_t>(num_active) * hidden * D);
    for (int i = 0; i < num_active; ++i) {
        int ge = active_experts[static_cast<std::size_t>(i)];
        const float* src = h_weights_full.data() + static_cast<std::size_t>(ge) * hidden * D;
        float* dst = h_weights_compact.data() + static_cast<std::size_t>(i) * hidden * D;
        std::memcpy(dst, src, static_cast<std::size_t>(hidden) * D * sizeof(float));
    }

    DeviceBuffer<float> d_input(h_input.size());
    DeviceBuffer<float> d_weights_full(h_weights_full.size());
    DeviceBuffer<float> d_weights_compact(h_weights_compact.size());
    DeviceBuffer<int> d_offsets(offsets.size());

    DeviceBuffer<float> d_out_full(static_cast<std::size_t>(total_tokens) * hidden);
    DeviceBuffer<float> d_out_compact(static_cast<std::size_t>(total_tokens) * hidden);
    CUDA_CHECK(cudaMemsetAsync(d_out_full.data(), 0, d_out_full.size() * sizeof(float), stream.stream));
    CUDA_CHECK(cudaMemsetAsync(d_out_compact.data(), 0, d_out_compact.size() * sizeof(float), stream.stream));

    CUDA_CHECK(cudaMemcpyAsync(d_input.data(), h_input.data(), h_input.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weights_full.data(), h_weights_full.data(), h_weights_full.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_weights_compact.data(), h_weights_compact.data(), h_weights_compact.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_offsets.data(), offsets.data(), offsets.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream.stream));

    moe_grouped_gemm_down(d_out_full.data(), d_input.data(), d_weights_full.data(),
                          d_offsets.data(), num_experts, hidden, D,
                          cublas.handle, stream.stream, offsets.data());

    moe_grouped_gemm_down(d_out_compact.data(), d_input.data(), d_weights_compact.data(),
                          d_offsets.data(), num_experts, hidden, D,
                          cublas.handle, stream.stream, offsets.data(),
                          active_experts.data(), /*weight_is_compact=*/true, /*num_active_experts=*/num_active);

    CUDA_CHECK(cudaStreamSynchronize(stream.stream));

    auto h_out_full = copy_device_to_host(d_out_full.data(), d_out_full.size());
    auto h_out_compact = copy_device_to_host(d_out_compact.data(), d_out_compact.size());

    for (std::size_t i = 0; i < h_out_full.size(); ++i) {
        REQUIRE(h_out_compact[i] == Catch::Approx(h_out_full[i]).margin(1e-4f));
    }
}

TEST_CASE("moe_combine_backward matches CPU reference", "[moe][combine]") {
    if (!cuda_available()) SKIP("CUDA not available");

    CudaStream stream;

    const int num_tokens = 8;
    const int top_k = 2;
    const int num_experts = 4;
    const int hidden = 16;
    const int total_tokens = num_tokens * top_k;

    // Make deterministic expert indices with valid range.
    std::vector<int> h_expert_indices(total_tokens);
    for (int i = 0; i < total_tokens; ++i) h_expert_indices[static_cast<std::size_t>(i)] = i % num_experts;

    std::vector<int> counts;
    compute_expert_counts_cpu(counts, h_expert_indices, num_tokens, top_k, num_experts);

    std::vector<int> h_gather, h_scatter;
    build_indices_cpu(h_gather, h_scatter, h_expert_indices, counts, num_tokens, top_k, num_experts);

    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);

    // Forward saved tensors
    std::vector<float> h_expert_out(static_cast<std::size_t>(total_tokens) * hidden);
    std::vector<float> h_routing_w(static_cast<std::size_t>(num_tokens) * top_k);
    for (auto& v : h_expert_out) v = dist(gen);
    for (auto& v : h_routing_w) v = dist(gen);

    // Upstream grad
    std::vector<float> h_d_out(static_cast<std::size_t>(num_tokens) * hidden);
    for (auto& v : h_d_out) v = dist(gen);

    // CPU ref
    std::vector<float> ref_d_expert_out(h_expert_out.size(), 0.0f);
    std::vector<float> ref_d_routing_w(h_routing_w.size(), 0.0f);

    for (int t = 0; t < num_tokens; ++t) {
        const float* dY = h_d_out.data() + static_cast<std::size_t>(t) * hidden;
        for (int k = 0; k < top_k; ++k) {
            const int assign = t * top_k + k;
            const int pos = h_scatter[static_cast<std::size_t>(assign)];
            const float w = h_routing_w[static_cast<std::size_t>(t) * top_k + k];

            // d_expert_out[pos] += w * dY
            float* dE = ref_d_expert_out.data() + static_cast<std::size_t>(pos) * hidden;
            for (int c = 0; c < hidden; ++c) dE[c] += w * dY[c];

            // d_routing_w[t,k] = dot(dY, expert_out[pos])
            const float* E = h_expert_out.data() + static_cast<std::size_t>(pos) * hidden;
            float dot = 0.0f;
            for (int c = 0; c < hidden; ++c) dot += dY[c] * E[c];
            ref_d_routing_w[static_cast<std::size_t>(t) * top_k + k] = dot;
        }
    }

    DeviceBuffer<float> d_expert_out(h_expert_out.size());
    DeviceBuffer<float> d_routing_w(h_routing_w.size());
    DeviceBuffer<float> d_d_out(h_d_out.size());
    DeviceBuffer<int> d_scatter(h_scatter.size());

    DeviceBuffer<float> d_d_expert_out(h_expert_out.size());
    DeviceBuffer<float> d_d_routing_w(h_routing_w.size());

    CUDA_CHECK(cudaMemcpyAsync(d_expert_out.data(), h_expert_out.data(), h_expert_out.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_routing_w.data(), h_routing_w.data(), h_routing_w.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_d_out.data(), h_d_out.data(), h_d_out.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_scatter.data(), h_scatter.data(), h_scatter.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream.stream));

    CUDA_CHECK(cudaMemsetAsync(d_d_expert_out.data(), 0, d_d_expert_out.size() * sizeof(float), stream.stream));
    CUDA_CHECK(cudaMemsetAsync(d_d_routing_w.data(), 0, d_d_routing_w.size() * sizeof(float), stream.stream));

    moe_combine_backward(d_d_expert_out.data(), d_d_routing_w.data(),
                         d_d_out.data(), d_expert_out.data(), d_routing_w.data(),
                         d_scatter.data(), num_tokens, total_tokens,
                         hidden, top_k, stream.stream);

    CUDA_CHECK(cudaStreamSynchronize(stream.stream));

    auto got_d_expert_out = copy_device_to_host(d_d_expert_out.data(), d_d_expert_out.size());
    auto got_d_routing_w = copy_device_to_host(d_d_routing_w.data(), d_d_routing_w.size());

    for (std::size_t i = 0; i < got_d_expert_out.size(); ++i) {
        REQUIRE(got_d_expert_out[i] == Catch::Approx(ref_d_expert_out[i]).margin(1e-4f));
    }
    for (std::size_t i = 0; i < got_d_routing_w.size(); ++i) {
        REQUIRE(got_d_routing_w[i] == Catch::Approx(ref_d_routing_w[i]).margin(1e-4f));
    }
}

TEST_CASE("moe_permute_backward matches CPU reference", "[moe][permute]") {
    if (!cuda_available()) SKIP("CUDA not available");

    CudaStream stream;

    const int num_tokens = 8;
    const int top_k = 2;
    const int num_experts = 4;
    const int hidden = 16;
    const int total_tokens = num_tokens * top_k;

    // Make deterministic expert indices.
    std::vector<int> h_expert_indices(total_tokens);
    for (int i = 0; i < total_tokens; ++i) h_expert_indices[static_cast<std::size_t>(i)] = i % num_experts;

    std::vector<int> counts;
    compute_expert_counts_cpu(counts, h_expert_indices, num_tokens, top_k, num_experts);

    std::vector<int> h_gather, h_scatter;
    build_indices_cpu(h_gather, h_scatter, h_expert_indices, counts, num_tokens, top_k, num_experts);

    std::mt19937 gen(9);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);

    // Upstream grad in permuted order (total_tokens, hidden)
    std::vector<float> h_d_perm(static_cast<std::size_t>(total_tokens) * hidden);
    for (auto& v : h_d_perm) v = dist(gen);

    // CPU ref: scatter-add into (num_tokens, hidden)
    std::vector<float> ref_d_input(static_cast<std::size_t>(num_tokens) * hidden, 0.0f);
    for (int out_idx = 0; out_idx < total_tokens; ++out_idx) {
        int assign_idx = h_gather[static_cast<std::size_t>(out_idx)];
        int token_idx = assign_idx / top_k;
        for (int c = 0; c < hidden; ++c) {
            ref_d_input[static_cast<std::size_t>(token_idx) * hidden + c] +=
                h_d_perm[static_cast<std::size_t>(out_idx) * hidden + c];
        }
    }

    DeviceBuffer<float> d_d_perm(h_d_perm.size());
    DeviceBuffer<int> d_gather(h_gather.size());
    DeviceBuffer<float> d_d_input(ref_d_input.size());
    CUDA_CHECK(cudaMemsetAsync(d_d_input.data(), 0, d_d_input.size() * sizeof(float), stream.stream));

    CUDA_CHECK(cudaMemcpyAsync(d_d_perm.data(), h_d_perm.data(), h_d_perm.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gather.data(), h_gather.data(), h_gather.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream.stream));

    moe_permute_backward(d_d_input.data(), d_d_perm.data(), d_gather.data(),
                         total_tokens, num_tokens, hidden, top_k, stream.stream);

    CUDA_CHECK(cudaStreamSynchronize(stream.stream));
    auto got = copy_device_to_host(d_d_input.data(), d_d_input.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
        REQUIRE(got[i] == Catch::Approx(ref_d_input[i]).margin(1e-4f));
    }
}

TEST_CASE("BnB/FP8/FP4 MoE router gate loads without transpose", "[moe][qlora][router]") {
    if (!cuda_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        MinimalMoESafetensorsConfig cfg{};
        cfg.vocab_size = 64;
        cfg.hidden_size = 8;
        cfg.intermediate_size = 16;
        cfg.moe_intermediate_size = 6;
        cfg.num_query_heads = 2;
        cfg.num_kv_heads = 1;
        cfg.head_size = 4;
        cfg.num_experts = 4;

        const auto router_gate = build_router_gate_pattern(cfg.num_experts, cfg.hidden_size);
        TempFile tmp = write_minimal_qwen3_moe_safetensors(cfg, router_gate);
        auto dp = get_device_props();
        auto allocator = std::make_shared<TensorAllocator>();

        auto check_gate = [&](const Tensor& gate, const char* tag) {
            REQUIRE(gate.DType == ETensorDType::BF16);
            REQUIRE(gate.Rank == 2);
            REQUIRE(gate.Sizes[0] == cfg.num_experts);
            REQUIRE(gate.Sizes[1] == cfg.hidden_size);
            auto got = copy_device_to_host(reinterpret_cast<const nv_bfloat16*>(gate.Data),
                                           static_cast<std::size_t>(cfg.num_experts) * cfg.hidden_size);
            for (std::size_t i = 0; i < got.size(); ++i) {
                INFO(tag);
                REQUIRE(bf16_bits(got[i]) == bf16_bits(router_gate[i]));
            }
        };

        // --- BnB (NF4) ---
        {
            modules::QLoRAConfig qcfg = modules::QLoRAConfig::bnb(/*block_size=*/64, /*double_quant=*/true);
            qcfg.num_experts = cfg.num_experts;
            qcfg.num_experts_per_tok = 2;
            qcfg.moe_intermediate_size = cfg.moe_intermediate_size;

            modules::BnBWeightsManager::Config wc{
                .num_layers = 1,
                .hidden_size = cfg.hidden_size,
                .intermediate_size = cfg.intermediate_size,
                .num_query_heads = cfg.num_query_heads,
                .num_kv_heads = cfg.num_kv_heads,
                .head_size = cfg.head_size,
                .vocab_size = cfg.vocab_size,
                .qlora_config = qcfg,
                .use_qk_norm = false,
                .tied_embeddings = true,
                .shard_idx = 0,
                .num_shards = 1,
                .offload_experts = false,
            };

            modules::BnBWeightsManager mgr(wc, *allocator, dp);
            mgr.import_and_quantize(tmp.path, comm, /*stream=*/0);
            check_gate(mgr.get_moe_block(0).router_gate, "bnb");
        }

        // --- FP8 ---
        if (fp8_supported()) {
            modules::QLoRAConfig qcfg = modules::QLoRAConfig::fp8(/*block_size=*/128);
            qcfg.num_experts = cfg.num_experts;
            qcfg.num_experts_per_tok = 2;
            qcfg.moe_intermediate_size = cfg.moe_intermediate_size;

            modules::FP8WeightsManager::Config wc{
                .num_layers = 1,
                .hidden_size = cfg.hidden_size,
                .intermediate_size = cfg.intermediate_size,
                .num_query_heads = cfg.num_query_heads,
                .num_kv_heads = cfg.num_kv_heads,
                .head_size = cfg.head_size,
                .vocab_size = cfg.vocab_size,
                .qlora_config = qcfg,
                .use_qk_norm = false,
                .tied_embeddings = true,
                .shard_idx = 0,
                .num_shards = 1,
            };

            modules::FP8WeightsManager mgr(wc, *allocator, dp);
            mgr.import_and_quantize(tmp.path, comm, /*stream=*/0);
            check_gate(mgr.get_moe_block(0).router_gate, "fp8");
        }

        // --- FP4 ---
        if (fp4_supported()) {
            modules::QLoRAConfig qcfg = modules::QLoRAConfig::nvfp4();
            qcfg.num_experts = cfg.num_experts;
            qcfg.num_experts_per_tok = 2;
            qcfg.moe_intermediate_size = cfg.moe_intermediate_size;

            modules::FP4WeightsManager::Config wc{
                .num_layers = 1,
                .hidden_size = cfg.hidden_size,
                .intermediate_size = cfg.intermediate_size,
                .num_query_heads = cfg.num_query_heads,
                .num_kv_heads = cfg.num_kv_heads,
                .head_size = cfg.head_size,
                .vocab_size = cfg.vocab_size,
                .qlora_config = qcfg,
                .use_qk_norm = false,
                .tied_embeddings = true,
                .shard_idx = 0,
                .num_shards = 1,
            };

            modules::FP4WeightsManager mgr(wc, *allocator, dp);
            mgr.import_and_quantize(tmp.path, comm, /*stream=*/0);
            check_gate(mgr.get_moe_block(0).router_gate, "fp4");
        }
    });
}

TEST_CASE("ModularWeightManager allocates MoE router/expert weights in model dtype", "[moe][weights]") {
    if (!cuda_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        const int hidden = 32;
        const int D = 16;
        const int num_experts = 8;
        const int top_k = 2;

        modules::MoEBlockConfigBuilder builder;
        builder.hidden_size(hidden)
            .num_query_heads(2)
            .num_kv_heads(1)
            .head_size(16)
            .num_experts(num_experts)
            .top_k(top_k)
            .intermediate_size(D);

        modules::ModularWeightManager<modules::StandardMoEBlock>::Config cfg{
            .num_layers = 1,
            .block_config = builder.build(),
            .master_dtype = ETensorDType::BF16,
            .model_dtype = ETensorDType::BF16,
            .matmul_dtype = ETensorDType::FP8_E4M3,  // should NOT be used for MoE router/expert weights
            .shard_idx = 0,
            .num_shards = 1,
            .shard_weights = false,
            .offload_master = false,
            .offload_quants = false,
            .use_zero_copy = false,
            .offload_alloc = EAllocationType::PINNED,
            .persistent_quants = false,
            .init_projections_to_zero = false,
            .vocab_size = 128,
            .hidden_size = hidden,
            .tied_embeddings = false,
            .architecture_id = PretrainedConfig::QWEN3_MOE,
            .skip_block_allocation = false,
            .enable_fp8_forward = false,
            .enable_fp4_forward = false,
            .enable_four_over_six = false,
            .four_over_six_metric = recipes::FourOverSixErrorMetric::MSE,
        };

        auto allocator = std::make_shared<TensorAllocator>();
        modules::ModularWeightManager<modules::StandardMoEBlock> wm(cfg, *allocator);
        auto& w = wm.get_block(0, /*stream=*/0);

        REQUIRE(w.router.gate.DType == ETensorDType::BF16);
        REQUIRE(w.router.gate.Rank == 2);
        REQUIRE(w.router.gate.Sizes[0] == num_experts);
        REQUIRE(w.router.gate.Sizes[1] == hidden);

        REQUIRE(w.experts.use_batched);
        REQUIRE(w.experts.gate_up_proj.DType == ETensorDType::BF16);
        REQUIRE(w.experts.gate_up_proj.Rank == 3);
        REQUIRE(w.experts.gate_up_proj.Sizes[0] == num_experts);
        REQUIRE(w.experts.gate_up_proj.Sizes[1] == 2 * D);
        REQUIRE(w.experts.gate_up_proj.Sizes[2] == hidden);

        REQUIRE(w.experts.down_proj.DType == ETensorDType::BF16);
        REQUIRE(w.experts.down_proj.Rank == 3);
        REQUIRE(w.experts.down_proj.Sizes[0] == num_experts);
        REQUIRE(w.experts.down_proj.Sizes[1] == hidden);
        REQUIRE(w.experts.down_proj.Sizes[2] == D);

        wm.release_block(0, /*stream=*/0);

        // Avoid unused parameter warning.
        (void)comm;
    });
}

TEST_CASE("ModularWeightManager imports per-expert MoE weights into batched expert tensors", "[moe][weights]") {
    if (!cuda_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        const int hidden = 8;
        const int D = 4;
        const int num_experts = 2;
        const int top_k = 1;

        // Build a minimal MoE block config and weight manager.
        modules::MoEBlockConfigBuilder builder;
        builder.hidden_size(hidden)
            .num_query_heads(1)
            .num_kv_heads(1)
            .head_size(8)
            .num_experts(num_experts)
            .top_k(top_k)
            .intermediate_size(D);

        modules::ModularWeightManager<modules::StandardMoEBlock>::Config cfg{
            .num_layers = 1,
            .block_config = builder.build(),
            .master_dtype = ETensorDType::BF16,
            .model_dtype = ETensorDType::BF16,
            .matmul_dtype = ETensorDType::BF16,
            .shard_idx = 0,
            .num_shards = 1,
            .shard_weights = false,
            .offload_master = false,
            .offload_quants = false,
            .use_zero_copy = false,
            .offload_alloc = EAllocationType::PINNED,
            .persistent_quants = false,
            .init_projections_to_zero = false,
            .vocab_size = 32,
            .hidden_size = hidden,
            .tied_embeddings = false,
            .architecture_id = PretrainedConfig::QWEN3_MOE,
            .skip_block_allocation = false,
            .enable_fp8_forward = false,
            .enable_fp4_forward = false,
            .enable_four_over_six = false,
            .four_over_six_metric = recipes::FourOverSixErrorMetric::MSE,
        };

        auto allocator = std::make_shared<TensorAllocator>();
        modules::ModularWeightManager<modules::StandardMoEBlock> wm(cfg, *allocator);

        auto fill_const = [](int elems, float value) {
            std::vector<nv_bfloat16> v(static_cast<std::size_t>(elems));
            for (int i = 0; i < elems; ++i) v[static_cast<std::size_t>(i)] = __float2bfloat16(value);
            return v;
        };

        // Create a small safetensors file using HF per-expert naming (gate_proj/up_proj/down_proj).
        const std::string path = make_temp_path("surogate_test_moe_per_expert_import");
        TempFile tmp(path);
        SafeTensorWriter writer(path);

        // Router gate (num_experts, hidden)
        DeviceBuffer<nv_bfloat16> d_router(static_cast<std::size_t>(num_experts) * hidden);
        auto h_router = fill_const(num_experts * hidden, 0.01f);
        CUDA_CHECK(cudaMemcpy(d_router.data(), h_router.data(),
                              h_router.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

        writer.register_tensor("model.layers.0.mlp.gate.weight",
                               TensorShard(make_tensor(ETensorDType::BF16, {num_experts, hidden},
                                                       reinterpret_cast<std::byte*>(d_router.data()))));

        // Experts: gate/up weights are (D, hidden), down weights are (hidden, D).
        std::array<DeviceBuffer<nv_bfloat16>, 6> d_expert{};
        for (auto& b : d_expert) b = DeviceBuffer<nv_bfloat16>(static_cast<std::size_t>(D) * hidden);
        std::array<DeviceBuffer<nv_bfloat16>, 2> d_down{
            DeviceBuffer<nv_bfloat16>(static_cast<std::size_t>(hidden) * D),
            DeviceBuffer<nv_bfloat16>(static_cast<std::size_t>(hidden) * D)
        };

        for (int e = 0; e < num_experts; ++e) {
            // Distinct constants per expert/proj so we can verify placement.
            const float gate_v = 10.0f + static_cast<float>(e);
            const float up_v = 20.0f + static_cast<float>(e);
            const float down_v = 30.0f + static_cast<float>(e);

            auto h_gate = fill_const(D * hidden, gate_v);
            auto h_up = fill_const(D * hidden, up_v);
            auto h_down = fill_const(hidden * D, down_v);

            CUDA_CHECK(cudaMemcpy(d_expert[static_cast<std::size_t>(e) * 2 + 0].data(), h_gate.data(),
                                  h_gate.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_expert[static_cast<std::size_t>(e) * 2 + 1].data(), h_up.data(),
                                  h_up.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_down[static_cast<std::size_t>(e)].data(), h_down.data(),
                                  h_down.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

            const std::string prefix = "model.layers.0.mlp.experts." + std::to_string(e);
            writer.register_tensor(prefix + ".gate_proj.weight",
                                   TensorShard(make_tensor(ETensorDType::BF16, {D, hidden},
                                                           reinterpret_cast<std::byte*>(d_expert[static_cast<std::size_t>(e) * 2 + 0].data()))));
            writer.register_tensor(prefix + ".up_proj.weight",
                                   TensorShard(make_tensor(ETensorDType::BF16, {D, hidden},
                                                           reinterpret_cast<std::byte*>(d_expert[static_cast<std::size_t>(e) * 2 + 1].data()))));
            writer.register_tensor(prefix + ".down_proj.weight",
                                   TensorShard(make_tensor(ETensorDType::BF16, {hidden, D},
                                                           reinterpret_cast<std::byte*>(d_down[static_cast<std::size_t>(e)].data()))));
        }

        writer.prepare_metadata(nullptr);
        writer.write_tensor("model.layers.0.mlp.gate.weight",
                            TensorShard(make_tensor(ETensorDType::BF16, {num_experts, hidden},
                                                    reinterpret_cast<std::byte*>(d_router.data()))),
                            nullptr);
        for (int e = 0; e < num_experts; ++e) {
            const std::string prefix = "model.layers.0.mlp.experts." + std::to_string(e);
            writer.write_tensor(prefix + ".gate_proj.weight",
                                TensorShard(make_tensor(ETensorDType::BF16, {D, hidden},
                                                        reinterpret_cast<std::byte*>(d_expert[static_cast<std::size_t>(e) * 2 + 0].data()))),
                                nullptr);
            writer.write_tensor(prefix + ".up_proj.weight",
                                TensorShard(make_tensor(ETensorDType::BF16, {D, hidden},
                                                        reinterpret_cast<std::byte*>(d_expert[static_cast<std::size_t>(e) * 2 + 1].data()))),
                                nullptr);
            writer.write_tensor(prefix + ".down_proj.weight",
                                TensorShard(make_tensor(ETensorDType::BF16, {hidden, D},
                                                        reinterpret_cast<std::byte*>(d_down[static_cast<std::size_t>(e)].data()))),
                                nullptr);
        }
        writer.finalize(nullptr);

        // Import into batched expert weights.
        REQUIRE_NOTHROW(wm.import_from_file(tmp.path, /*allow_cast=*/true, comm));

        auto& w = wm.get_block(0, /*stream=*/0);
        REQUIRE(w.experts.use_batched);
        REQUIRE(w.experts.gate_up_proj.Rank == 3);
        REQUIRE(w.experts.gate_up_proj.Sizes[0] == num_experts);
        REQUIRE(w.experts.gate_up_proj.Sizes[1] == 2 * D);
        REQUIRE(w.experts.gate_up_proj.Sizes[2] == hidden);

        // Verify placement: [up; gate] in gate_up_proj.
        std::vector<nv_bfloat16> h_gate_up(static_cast<std::size_t>(num_experts) * 2 * D * hidden);
        CUDA_CHECK(cudaMemcpy(h_gate_up.data(), w.experts.gate_up_proj.get<nv_bfloat16>(),
                              h_gate_up.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
        auto at = [&](int e, int row, int col) -> float {
            const std::size_t idx = (static_cast<std::size_t>(e) * 2 * D + static_cast<std::size_t>(row)) * hidden + static_cast<std::size_t>(col);
            return __bfloat162float(h_gate_up[idx]);
        };

        for (int e = 0; e < num_experts; ++e) {
            const float expected_up = 20.0f + static_cast<float>(e);
            const float expected_gate = 10.0f + static_cast<float>(e);
            REQUIRE(at(e, /*row=*/0, /*col=*/0) == Catch::Approx(expected_up));
            REQUIRE(at(e, /*row=*/D - 1, /*col=*/hidden - 1) == Catch::Approx(expected_up));
            REQUIRE(at(e, /*row=*/D, /*col=*/0) == Catch::Approx(expected_gate));
            REQUIRE(at(e, /*row=*/2 * D - 1, /*col=*/hidden - 1) == Catch::Approx(expected_gate));
        }

        // Verify down_proj placement.
        std::vector<nv_bfloat16> h_down(static_cast<std::size_t>(num_experts) * hidden * D);
        CUDA_CHECK(cudaMemcpy(h_down.data(), w.experts.down_proj.get<nv_bfloat16>(),
                              h_down.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
        auto down_at = [&](int e, int row, int col) -> float {
            const std::size_t idx = (static_cast<std::size_t>(e) * hidden + static_cast<std::size_t>(row)) * D + static_cast<std::size_t>(col);
            return __bfloat162float(h_down[idx]);
        };
        for (int e = 0; e < num_experts; ++e) {
            const float expected_down = 30.0f + static_cast<float>(e);
            REQUIRE(down_at(e, /*row=*/0, /*col=*/0) == Catch::Approx(expected_down));
            REQUIRE(down_at(e, /*row=*/hidden - 1, /*col=*/D - 1) == Catch::Approx(expected_down));
        }
    });
}

TEST_CASE("moe_topk_backward matches CPU reference (fp32)", "[moe][topk]") {
    if (!cuda_available()) SKIP("CUDA not available");

    CudaStream stream;

    const int BT = 11;
    const int E = 9;
    const int K = 4;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist_prob(0.001f, 1.0f);
    std::uniform_real_distribution<float> dist_grad(-0.25f, 0.25f);

    std::vector<float> h_probs(static_cast<std::size_t>(BT) * E);
    for (int t = 0; t < BT; ++t) {
        float sum = 0.0f;
        for (int e = 0; e < E; ++e) {
            float v = dist_prob(gen);
            h_probs[static_cast<std::size_t>(t) * E + e] = v;
            sum += v;
        }
        for (int e = 0; e < E; ++e) {
            h_probs[static_cast<std::size_t>(t) * E + e] /= sum;
        }
    }

    DeviceBuffer<float> d_probs(h_probs.size());
    DeviceBuffer<int> d_expert_indices(static_cast<std::size_t>(BT) * K);
    DeviceBuffer<float> d_routing_w(static_cast<std::size_t>(BT) * K);
    DeviceBuffer<float> d_d_routing_w(static_cast<std::size_t>(BT) * K);
    DeviceBuffer<float> d_d_probs(static_cast<std::size_t>(BT) * E);

    CUDA_CHECK(cudaMemcpyAsync(d_probs.data(), h_probs.data(), h_probs.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.stream));

    for (bool normalize : {false, true}) {
        moe_topk_forward(d_expert_indices.data(), d_routing_w.data(), d_probs.data(),
                         BT, E, K, normalize, stream.stream);

        std::vector<float> h_d_routing_w(static_cast<std::size_t>(BT) * K);
        for (auto& v : h_d_routing_w) v = dist_grad(gen);
        CUDA_CHECK(cudaMemcpyAsync(d_d_routing_w.data(), h_d_routing_w.data(), h_d_routing_w.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream.stream));
        CUDA_CHECK(cudaMemsetAsync(d_d_probs.data(), 0, d_d_probs.size() * sizeof(float), stream.stream));

        moe_topk_backward(d_d_probs.data(), d_d_routing_w.data(), d_probs.data(), d_expert_indices.data(),
                          BT, E, K, normalize, stream.stream);

        CUDA_CHECK(cudaStreamSynchronize(stream.stream));

        auto h_expert_indices = copy_device_to_host(d_expert_indices.data(), static_cast<std::size_t>(BT) * K);
        auto h_d_probs = copy_device_to_host(d_d_probs.data(), static_cast<std::size_t>(BT) * E);

        // CPU reference
        std::vector<float> ref(static_cast<std::size_t>(BT) * E, 0.0f);
        for (int t = 0; t < BT; ++t) {
            float S = 0.0f;
            float dot = 0.0f;
            for (int k = 0; k < K; ++k) {
                const int idx = h_expert_indices[static_cast<std::size_t>(t) * K + k];
                const float p = h_probs[static_cast<std::size_t>(t) * E + idx];
                const float g = h_d_routing_w[static_cast<std::size_t>(t) * K + k];
                S += p;
                dot += g * p;
            }

            for (int k = 0; k < K; ++k) {
                const int idx = h_expert_indices[static_cast<std::size_t>(t) * K + k];
                const float p = h_probs[static_cast<std::size_t>(t) * E + idx];
                const float g = h_d_routing_w[static_cast<std::size_t>(t) * K + k];
                if (!normalize) {
                    ref[static_cast<std::size_t>(t) * E + idx] += g;
                } else {
                    ref[static_cast<std::size_t>(t) * E + idx] += (g * S - dot) / (S * S);
                }
            }
        }

        for (std::size_t i = 0; i < ref.size(); ++i) {
            REQUIRE(h_d_probs[i] == Catch::Approx(ref[i]).margin(1e-5f));
        }
    }
}

TEST_CASE("Modular MoE model: 1 step forward/backward/update runs (full finetune)", "[moe][modular]") {
    if (!cuda_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 32;

        Qwen3MoEConfig cfg;
        cfg.HiddenSize = 32;
        cfg.IntermediateSize = 64;   // dense MLP (unused for pure MoE layers)
        cfg.MoeIntermediateSize = 16;
        cfg.NumQueryHeads = 4;
        cfg.NumKeyValHeads = 2;
        cfg.HeadDim = 16;            // != HiddenSize / NumQueryHeads
        cfg.NumLayers = 2;
        cfg.VocabSize = 128;
        cfg.MaxPositionEmbeddings = 128;
        cfg.RopeTheta = 10000.0f;
        cfg.RmsNormEps = 1e-6f;
        cfg.TiedWordEmbeddings = false;
        cfg.UseQKVBias = false;
        cfg.UseQKNorm = true;
        cfg.BosTokenId = 0;
        cfg.EosTokenId = 1;
        cfg.PadTokenId = 2;
        cfg.DType = ETensorDType::BF16;

        cfg.NumExperts = 4;
        cfg.NumExpertsPerTok = 2;
        cfg.DecoderSparseStep = 1;       // all layers MoE
        cfg.MlpOnlyLayers = {};
        cfg.NormTopkProb = true;
        cfg.RouterAuxLossCoef = 0.001f;

        RuntimeOptions opts;
        opts.UseCudaGraphs = false;
        opts.RecomputeBlock = false;
        opts.ModelType = ETensorDType::BF16;
        opts.MatmulType = ETensorDType::BF16;
        opts.MasterDType = ETensorDType::BF16;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_from_pretrained_config(cfg, opts, comm.rank(), comm.world_size(), allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        TensorShard gate_before;
        bool found_gate = false;
        model->weights().iterate_tensors([&](std::string name, const TensorShard& t) {
            if (name == "model.layers.0.mlp.router.gate.weight") {
                gate_before = t;
                found_gate = true;
            }
        });
        REQUIRE(found_gate);
        REQUIRE(gate_before.DType == ETensorDType::BF16);
        REQUIRE(gate_before.Data != nullptr);
        auto h_gate_before = copy_device_to_host(reinterpret_cast<const nv_bfloat16*>(gate_before.Data), 64);

        // Fill inputs/targets/position IDs in pinned host buffers provided by the model.
        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();

        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                pos_ids.get<std::int32_t>()[b * T + t] = t;
            }
        }

        REQUIRE_NOTHROW(model->forward(inputs, pos_ids, comm, /*micro_step=*/0));
        REQUIRE_NOTHROW(model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0));
        REQUIRE_NOTHROW(model->update(comm, /*lr=*/1e-2f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/0.0f));

        CUDA_CHECK(cudaDeviceSynchronize());

        REQUIRE(std::isfinite(model->get_loss()));
        REQUIRE(std::isfinite(model->get_norm()));
        REQUIRE(model->get_norm() > 0.0f);

        TensorShard gate_after;
        bool found_gate_after = false;
        model->weights().iterate_tensors([&](std::string name, const TensorShard& t) {
            if (name == "model.layers.0.mlp.router.gate.weight") {
                gate_after = t;
                found_gate_after = true;
            }
        });
        REQUIRE(found_gate_after);
        auto h_gate_after = copy_device_to_host(reinterpret_cast<const nv_bfloat16*>(gate_after.Data), 64);

        bool changed = false;
        for (std::size_t i = 0; i < h_gate_before.size(); ++i) {
            if (bf16_bits(h_gate_before[i]) != bf16_bits(h_gate_after[i])) {
                changed = true;
                break;
            }
        }
        REQUIRE(changed);
    });
}

TEST_CASE("Modular MoE model: LoRA 1-step grad-norm stays finite", "[moe][lora]") {
    if (!cuda_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 2;
        constexpr int T = 32;

        Qwen3MoEConfig cfg;
        cfg.HiddenSize = 64;
        cfg.IntermediateSize = 128;
        cfg.MoeIntermediateSize = 32;
        cfg.NumQueryHeads = 4;
        cfg.NumKeyValHeads = 2;
        cfg.HeadDim = 16; // Hq*Hd == Hidden
        cfg.NumLayers = 2;
        cfg.VocabSize = 256;
        cfg.MaxPositionEmbeddings = 128;
        cfg.RopeTheta = 10000.0f;
        cfg.RmsNormEps = 1e-6f;
        cfg.TiedWordEmbeddings = false;
        cfg.UseQKVBias = false;
        cfg.UseQKNorm = true;
        cfg.BosTokenId = 0;
        cfg.EosTokenId = 1;
        cfg.PadTokenId = 2;
        cfg.DType = ETensorDType::BF16;

        cfg.NumExperts = 4;
        cfg.NumExpertsPerTok = 2;
        cfg.DecoderSparseStep = 1;
        cfg.MlpOnlyLayers = {};
        cfg.NormTopkProb = true;
        cfg.RouterAuxLossCoef = 0.001f;

        RuntimeOptions opts;
        opts.UseCudaGraphs = false;
        opts.RecomputeBlock = false;
        opts.ModelType = ETensorDType::BF16;
        opts.MatmulType = ETensorDType::BF16;
        opts.MasterDType = ETensorDType::BF16;

        modules::LoRAConfigBuilder lora_builder;
        modules::ModularLoRAConfig lora_cfg = lora_builder.rank(4)
                                                  .alpha(8.0f)
                                                  .dtype(ETensorDType::BF16)
                                                  .clear_targets()
                                                  .attention()
                                                  .mlp()  // MoE MLP targets use grouped per-expert LoRA
                                                  .build();

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_lora_from_pretrained_config(cfg, lora_cfg, opts, comm, allocator);
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);
        model->init_weights(comm);

        // Capture a small checksum of grouped MoE LoRA tensors (exported as per-expert slices)
        // to ensure the optimizer actually updates them.
        float grouped_gate_b_abs_before = 0.0f;
        int found_count = 0;
        model->weights().iterate_tensors([&](std::string name, const TensorShard& t) {
            // PEFT-compatible format: .mlp.experts.{e}.gate_proj.lora_B.weight
            if (name.find(".mlp.experts.") == std::string::npos) return;
            if (name.find(".gate_proj.lora_B.weight") == std::string::npos) return;
            if (t.DType != ETensorDType::BF16) return;
            // First few elements should start at 0 and become non-zero after one update.
            std::array<nv_bfloat16, 8> h{};
            CUDA_CHECK(cudaMemcpy(h.data(), t.get<nv_bfloat16>(), h.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
            for (const auto& v : h) {
                grouped_gate_b_abs_before += std::abs(__bfloat162float(v));
            }
            found_count++;
        });
        REQUIRE(found_count == cfg.NumExperts * cfg.NumLayers);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();

        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                pos_ids.get<std::int32_t>()[b * T + t] = t;
            }
        }

        // Two micro-steps to exercise grad accumulation + MoE scatter-add paths.
        for (int micro = 0; micro < 2; ++micro) {
            REQUIRE_NOTHROW(model->forward(inputs, pos_ids, comm, micro));
            REQUIRE_NOTHROW(model->backward(inputs, targets, comm, /*grad_accum_steps=*/2, micro));
        }

        REQUIRE_NOTHROW(model->update(comm, /*lr=*/1e-2f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/0.0f));

        CUDA_CHECK(cudaDeviceSynchronize());

        float loss = model->get_loss();
        float norm = model->get_norm();
        REQUIRE(std::isfinite(loss));
        REQUIRE(std::isfinite(norm));
        REQUIRE(norm > 0.0f);
        REQUIRE(norm < 1e4f);

        float grouped_gate_b_abs_after = 0.0f;
        model->weights().iterate_tensors([&](std::string name, const TensorShard& t) {
            // PEFT-compatible format: .mlp.experts.{e}.gate_proj.lora_B.weight
            if (name.find(".mlp.experts.") == std::string::npos) return;
            if (name.find(".gate_proj.lora_B.weight") == std::string::npos) return;
            if (t.DType != ETensorDType::BF16) return;
            std::array<nv_bfloat16, 8> h{};
            CUDA_CHECK(cudaMemcpy(h.data(), t.get<nv_bfloat16>(), h.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
            for (const auto& v : h) {
                grouped_gate_b_abs_after += std::abs(__bfloat162float(v));
            }
        });
        REQUIRE(grouped_gate_b_abs_after > grouped_gate_b_abs_before);
    });
}

TEST_CASE("Modular MoE model: QLoRA(BnB) selective expert dequant works in backward across layers", "[moe][qlora][bnb]") {
    if (!cuda_available()) SKIP("CUDA not available");

    NCCLCommunicator::run_communicators(1, false, false, [](NCCLCommunicator& comm) {
        constexpr int B = 1;
        constexpr int T = 16;

        // Tiny 2-layer MoE checkpoint with deterministic, disjoint routing per layer:
        // - layer0 routes to experts {0,1}
        // - layer1 routes to experts {2,3}
        MinimalMoESafetensorsConfig wcfg{};
        wcfg.vocab_size = 64;
        wcfg.hidden_size = 16;
        wcfg.intermediate_size = 16;
        wcfg.moe_intermediate_size = 8;
        wcfg.num_query_heads = 2;
        wcfg.num_kv_heads = 1;
        wcfg.head_size = 8;
        wcfg.num_experts = 4;

        std::vector<nv_bfloat16> gate0(static_cast<std::size_t>(wcfg.num_experts) * wcfg.hidden_size);
        std::vector<nv_bfloat16> gate1(static_cast<std::size_t>(wcfg.num_experts) * wcfg.hidden_size);
        for (int e = 0; e < wcfg.num_experts; ++e) {
            float s0 = (e < 2) ? 1.0f : -1.0f;
            float s1 = (e < 2) ? -1.0f : 1.0f;
            for (int c = 0; c < wcfg.hidden_size; ++c) {
                gate0[static_cast<std::size_t>(e) * wcfg.hidden_size + c] = __float2bfloat16(s0);
                gate1[static_cast<std::size_t>(e) * wcfg.hidden_size + c] = __float2bfloat16(s1);
            }
        }
        TempFile tmp = write_minimal_qwen3_moe_safetensors_2layers(wcfg, gate0, gate1);

        Qwen3MoEConfig cfg;
        cfg.HiddenSize = wcfg.hidden_size;
        cfg.IntermediateSize = wcfg.intermediate_size;
        cfg.MoeIntermediateSize = wcfg.moe_intermediate_size;
        cfg.NumQueryHeads = wcfg.num_query_heads;
        cfg.NumKeyValHeads = wcfg.num_kv_heads;
        cfg.HeadDim = wcfg.head_size;
        cfg.NumLayers = 2;
        cfg.VocabSize = wcfg.vocab_size;
        cfg.MaxPositionEmbeddings = 128;
        cfg.RopeTheta = 10000.0f;
        cfg.RmsNormEps = 1e-6f;
        cfg.TiedWordEmbeddings = true;
        cfg.UseQKVBias = false;
        cfg.UseQKNorm = false;
        cfg.BosTokenId = 0;
        cfg.EosTokenId = 1;
        cfg.PadTokenId = 2;
        cfg.DType = ETensorDType::BF16;

        cfg.NumExperts = wcfg.num_experts;
        cfg.NumExpertsPerTok = 2;
        cfg.DecoderSparseStep = 1;
        cfg.MlpOnlyLayers = {};
        cfg.NormTopkProb = true;
        cfg.RouterAuxLossCoef = 0.0f;

        RuntimeOptions opts;
        opts.UseCudaGraphs = false;
        opts.RecomputeBlock = false;
        opts.ModelType = ETensorDType::BF16;
        opts.MatmulType = ETensorDType::BF16;
        opts.MasterDType = ETensorDType::BF16;
        opts.SelectiveExpertDequant = true;
        opts.OffloadExperts = false;

        modules::LoRAConfigBuilder lora_builder;
        modules::ModularLoRAConfig lora_cfg = lora_builder.rank(2)
                                                  .alpha(4.0f)
                                                  .dtype(ETensorDType::BF16)
                                                  .clear_targets()
                                                  .mlp()  // MoE MLP targets use grouped per-expert LoRA
                                                  .build();

        modules::QLoRAConfig qcfg = modules::QLoRAConfig::bnb(/*block_size=*/64, /*double_quant=*/true);
        qcfg.num_experts = wcfg.num_experts;
        qcfg.num_experts_per_tok = 2;
        qcfg.moe_intermediate_size = wcfg.moe_intermediate_size;

        auto allocator = std::make_shared<TensorAllocator>();
        auto model = modules::ModelFactory::create_lora_from_pretrained_config(cfg, lora_cfg, opts, comm, allocator, qcfg);

        REQUIRE_NOTHROW(model->import_weights(tmp.path, /*allow_cast=*/true, comm));
        model->allocate_run_state(opts, comm, B, T, /*allocate_optimizer=*/true);

        auto get_grouped_gate_b_abs = [&](int layer) {
            // With PEFT-compatible export, grouped weights are exported as per-expert slices
            // We sum over all experts in the layer to get total gradient update
            float abs_sum = 0.0f;
            int found_count = 0;
            const int num_experts = wcfg.num_experts;

            for (int e = 0; e < num_experts; ++e) {
                const std::string needle = "base_model.model.model.layers." + std::to_string(layer) +
                                          ".mlp.experts." + std::to_string(e) + ".gate_proj.lora_B.weight";
                model->weights().iterate_tensors([&](std::string name, const TensorShard& t) {
                    if (name != needle) return;
                    REQUIRE(t.DType == ETensorDType::BF16);
                    std::array<nv_bfloat16, 8> h{};
                    CUDA_CHECK(cudaMemcpy(h.data(), t.get<nv_bfloat16>(), h.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
                    for (const auto& v : h) abs_sum += std::abs(__bfloat162float(v));
                    found_count++;
                });
            }
            REQUIRE(found_count == num_experts);
            return abs_sum;
        };

        const float before_l0 = get_grouped_gate_b_abs(/*layer=*/0);
        const float before_l1 = get_grouped_gate_b_abs(/*layer=*/1);
        REQUIRE(before_l0 == 0.0f);
        REQUIRE(before_l1 == 0.0f);

        auto& inputs = model->get_input_buffer();
        auto& targets = model->get_target_buffer();
        auto& pos_ids = model->get_position_ids_buffer();

        std::fill(inputs.get<std::int32_t>(), inputs.get<std::int32_t>() + B * T, 1);
        std::fill(targets.get<std::int32_t>(), targets.get<std::int32_t>() + B * T, 2);
        for (int t = 0; t < T; ++t) {
            pos_ids.get<std::int32_t>()[t] = t;
        }

        REQUIRE_NOTHROW(model->forward(inputs, pos_ids, comm, /*micro_step=*/0));
        REQUIRE_NOTHROW(model->backward(inputs, targets, comm, /*grad_accum_steps=*/1, /*micro_step=*/0));
        REQUIRE_NOTHROW(model->update(comm, /*lr=*/1e-2f, /*beta1=*/0.9f, /*beta2=*/0.999f, /*t=*/1,
                                      /*epsilon=*/1e-8f, /*weight_decay=*/0.0f, /*grad_clip=*/0.0f));

        CUDA_CHECK(cudaDeviceSynchronize());

        REQUIRE(std::isfinite(model->get_loss()));
        REQUIRE(std::isfinite(model->get_norm()));
        REQUIRE(model->get_norm() > 0.0f);
        REQUIRE(model->get_norm() < 1e4f);

        const float after_l0 = get_grouped_gate_b_abs(/*layer=*/0);
        const float after_l1 = get_grouped_gate_b_abs(/*layer=*/1);
        REQUIRE(after_l0 > 0.0f);
        REQUIRE(after_l1 > 0.0f);
    });
}
