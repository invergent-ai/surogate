// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_bf16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "config/pretrained_config.h"
#include "runtime/dsl/dsl_weight_loader.h"
#include "runtime/dsl/shared_master_store.h"
#include "runtime/qlora/adapter_merger.h"
#include "utilities/allocator.h"
#include "utilities/cu_file.h"
#include "utilities/safetensors.h"

namespace {

using dsl::MappingSpec;
using Kind = MappingSpec::Kind;

void require_gpu() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) SKIP("CUDA device required");
}

struct Weight {
    std::string name;
    std::vector<long> shape;
    std::vector<float> values;
    ETensorDType dtype = ETensorDType::FP32; ///< FP32 or BF16 on disk
};

std::size_t disk_bytes(const Weight& weight) {
    return weight.values.size() * (weight.dtype == ETensorDType::BF16 ? sizeof(nv_bfloat16) : sizeof(float));
}

// Write independent FP32 / BF16 SafeTensors fixtures; no model download or exporter needed.
struct Checkpoint {
    std::filesystem::path directory;
    std::string path;

    explicit Checkpoint(const std::vector<Weight>& weights) {
        std::string pattern = (std::filesystem::temp_directory_path() / "surogate-weight-loader-XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        directory = pattern;
        path = (directory / "model.safetensors").string();
        nlohmann::json metadata = nlohmann::json::object();
        std::size_t offset = 0;
        for (const auto& weight : weights) {
            REQUIRE(std::accumulate(weight.shape.begin(), weight.shape.end(), 1L, std::multiplies<>()) ==
                    weight.values.size());
            const auto end = offset + disk_bytes(weight);
            metadata[weight.name] = {{"dtype", weight.dtype == ETensorDType::BF16 ? "BF16" : "F32"},
                                     {"shape", weight.shape}, {"data_offsets", {offset, end}}};
            offset = end;
        }
        auto header = metadata.dump();
        header.append((8 - header.size() % 8) % 8, ' ');
        const std::uint64_t header_size = header.size();
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        file.write(header.data(), header.size());
        for (const auto& weight : weights) {
            if (weight.dtype == ETensorDType::BF16) {
                std::vector<nv_bfloat16> raw(weight.values.size());
                std::transform(weight.values.begin(), weight.values.end(), raw.begin(),
                               [](float x) { return __float2bfloat16(x); });
                file.write(reinterpret_cast<const char*>(raw.data()), disk_bytes(weight));
            } else {
                file.write(reinterpret_cast<const char*>(weight.values.data()), disk_bytes(weight));
            }
        }
        REQUIRE(file.good());
    }

    ~Checkpoint() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};

Tensor allocate(TensorAllocator& allocator, const std::vector<long>& shape,
                ETensorDType dtype = ETensorDType::FP32, EAllocationType kind = EAllocationType::ON_DEVICE) {
    return allocator.allocate(dtype, "weight-loader-test", kind, shape);
}

std::vector<float> values(const Tensor& tensor) {
    std::vector<float> result(tensor.nelem());
    if (tensor.DType == ETensorDType::BF16) {
        std::vector<nv_bfloat16> raw(tensor.nelem());
        CUDA_CHECK(cudaMemcpy(raw.data(), tensor.Data, tensor.bytes(), cudaMemcpyDefault));
        std::transform(raw.begin(), raw.end(), result.begin(), [](nv_bfloat16 x) { return float(x); });
    } else {
        CUDA_CHECK(cudaMemcpy(result.data(), tensor.Data, tensor.bytes(), cudaMemcpyDefault));
    }
    return result;
}

std::vector<float> sequence(int count, int start = 0) {
    std::vector<float> result(count);
    std::iota(result.begin(), result.end(), float(start));
    return result;
}

// Sets (or, with nullptr, removes) one environment variable for a scope.
struct ScopedEnv {
    std::string name;
    std::optional<std::string> previous;
    ScopedEnv(std::string variable, const char* value) : name(std::move(variable)) {
        if (const char* old = std::getenv(name.c_str())) previous = old;
        if (value) ::setenv(name.c_str(), value, 1);
        else ::unsetenv(name.c_str());
    }
    ~ScopedEnv() {
        if (previous) ::setenv(name.c_str(), previous->c_str(), 1);
        else ::unsetenv(name.c_str());
    }
};

} // namespace

TEST_CASE("DSL direct loads preserve names, singleton squeezing, casts and tied embedding fallback", "[weight-loader]") {
    require_gpu();
    Checkpoint checkpoint({{"plain", {2, 2}, sequence(4)},
                           {"hf.layers.3.conv", {2, 1, 3}, sequence(6, 10)},
                           {"custom.embedding", {2, 2}, sequence(4, 20)}});
    SafeTensorsReader reader(checkpoint.path);
    TensorAllocator allocator;
    PretrainedConfig config;
    config.TiedWordEmbeddings = true;
    dsl::MappingTable mapping{
        {"blocks[{layer}].conv", {.kind = Kind::Direct, .source = "hf.layers.{layer}.conv"}},
        {"embedding", {.kind = Kind::Direct, .source = "custom.embedding"}},
        {"head", {.kind = Kind::Direct, .source = "lm_head.weight"}},
        {"optional", {.kind = Kind::Direct, .source = "absent", .optional = true}}};
    dsl::DslWeightLoader loader(reader, mapping, config, allocator);
    auto plain = allocate(allocator, {2, 2});
    REQUIRE(loader.load_param("plain", plain, false));
    REQUIRE(values(plain) == sequence(4));
    auto conv = allocate(allocator, {2, 3}, ETensorDType::BF16);
    REQUIRE_THROWS(loader.load_param("blocks[3].conv", conv, false));
    REQUIRE(loader.load_param("blocks[3].conv", conv, true));
    REQUIRE(values(conv) == sequence(6, 10));
    REQUIRE(loader.load_param("head", plain, false));
    REQUIRE(values(plain) == sequence(4, 20));
    REQUIRE_FALSE(loader.load_param("optional", plain, false));
    REQUIRE(values(plain) == sequence(4, 20));
    REQUIRE_THROWS(loader.load_param("required", plain, false));
    auto wrong = allocate(allocator, {3, 2});
    REQUIRE_THROWS(loader.load_param("plain", wrong, false));
}

TEST_CASE("DSL direct, fused and split loads select the correct rows for each shard", "[weight-loader]") {
    require_gpu();
    Checkpoint checkpoint({{"q", {3, 2}, sequence(6)}, {"k", {2, 2}, sequence(4, 6)},
                           {"v", {3, 2}, sequence(6, 10)}, {"packed", {8, 2}, sequence(16)}});
    SafeTensorsReader reader(checkpoint.path);
    TensorAllocator allocator;
    PretrainedConfig config; // Global head sizes deliberately differ: infer fuse sizes from the files.
    dsl::MappingTable mapping{
        {"qkv_weight", {.kind = Kind::Fuse, .sources = {"q", "k", "v"}}},
        {"split", {.kind = Kind::Split, .source = "packed", .ranges = {{2, 6}}}}};
    auto global = Tensor::empty(ETensorDType::FP32, {8, 2});
    auto split_global = Tensor::empty(ETensorDType::FP32, {4, 2});
    for (int shards : {1, 2}) {
        for (int shard = 0; shard < shards; ++shard) {
            CAPTURE(shards, shard);
            dsl::DslWeightLoader loader(reader, mapping, config, allocator, {shard, shards});
            auto target = allocate(allocator, {8 / shards, 2});
            REQUIRE(loader.load_param("packed", target, false, shards > 1, &global));
            REQUIRE(values(target) == sequence(16 / shards, shard * 16 / shards));
            REQUIRE(loader.load_param("blocks[0].qkv_weight", target, false, shards > 1, &global));
            REQUIRE(values(target) == sequence(16 / shards, shard * 16 / shards));
            auto split = allocate(allocator, {4 / shards, 2});
            REQUIRE(loader.load_param("split", split, false, shards > 1, &split_global));
            REQUIRE(values(split) == sequence(8 / shards, 4 + shard * 8 / shards));
        }
    }
}

TEST_CASE("DSL transpose loads handle matrices and batched experts with row sharding", "[weight-loader]") {
    require_gpu();
    const auto dtype = GENERATE(ETensorDType::FP32, ETensorDType::BF16);
    Checkpoint checkpoint({{"matrix", {2, 4}, sequence(8)}, {"experts", {4, 2, 3}, sequence(24)}});
    SafeTensorsReader reader(checkpoint.path);
    TensorAllocator allocator;
    PretrainedConfig config;
    dsl::MappingTable mapping{
        {"matrix", {.kind = Kind::Transform, .source = "matrix", .fn = "transpose"}},
        {"experts", {.kind = Kind::Transform, .source = "experts", .fn = "transpose"}}};
    for (bool batched : {false, true}) {
        const std::string name = batched ? "experts" : "matrix";
        const std::vector<long> shape = batched ? std::vector<long>{4, 3, 2} : std::vector<long>{4, 2};
        const auto global = Tensor::empty(dtype, shape);
        std::vector<float> expected = batched
            ? std::vector<float>{0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11,
                                 12, 15, 13, 16, 14, 17, 18, 21, 19, 22, 20, 23}
            : std::vector<float>{0, 4, 1, 5, 2, 6, 3, 7};
        for (int shards : {1, 2}) {
            for (int shard = 0; shard < shards; ++shard) {
                CAPTURE(name, shards, shard);
                dsl::DslWeightLoader loader(reader, mapping, config, allocator, {shard, shards});
                auto local_shape = shape;
                local_shape[0] /= shards;
                auto target = allocate(allocator, local_shape, dtype);
                REQUIRE(loader.load_param(name, target, true, shards > 1, &global));
                const auto count = expected.size() / shards;
                REQUIRE(values(target) == std::vector<float>(expected.begin() + shard * count,
                                                             expected.begin() + (shard + 1) * count));
                if (shards > 1) REQUIRE_THROWS(loader.load_param(name, target, false, true));
            }
        }
    }
    dsl::DslWeightLoader loader(reader, mapping, config, allocator);
    auto wrong_shape = allocate(allocator, {4, 2, 3});
    REQUIRE_THROWS(loader.load_param("experts", wrong_shape, false));
}

TEST_CASE("DSL expert callbacks see complete slices and global expert IDs", "[weight-loader]") {
    require_gpu();
    std::vector<Weight> weights;
    for (int e = 0; e < 4; ++e) {
        const auto prefix = "hf.layers.2.experts." + std::to_string(e);
        weights.push_back({prefix + ".w1", {2, 3}, sequence(6, 100 * e + 10)});
        weights.push_back({prefix + ".w3", {2, 3}, sequence(6, 100 * e)});
    }
    Checkpoint checkpoint(weights);
    SafeTensorsReader reader(checkpoint.path);
    TensorAllocator allocator;
    PretrainedConfig config;
    for (bool fused : {false, true}) {
        dsl::MappingTable mapping{{"blocks[{layer}].experts",
            {.kind = Kind::StackExperts, .source = "hf.layers.{layer}.experts.{expert}.w1",
             .fuse_gate_up = fused, .up_source = "hf.layers.{layer}.experts.{expert}.w3"}}};
        for (int shards : {1, 2}) {
            for (int shard = 0; shard < shards; ++shard) {
                CAPTURE(fused, shards, shard);
                dsl::DslWeightLoader loader(reader, mapping, config, allocator, {shard, shards}, {4, 2});
                auto target = allocate(allocator, {4 / shards, fused ? 4 : 2, 3});
                std::vector<int> seen;
                cudaStream_t stream = nullptr;
                CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                const auto on_expert = [&](int e, Tensor& expert) {
                    auto expected = sequence(6, 100 * e + 10);
                    if (fused) {
                        auto up = sequence(6, 100 * e);
                        up.insert(up.end(), expected.begin(), expected.end());
                        expected = std::move(up);
                    }
                    REQUIRE(expert.Rank == 2);
                    REQUIRE(values(expert) == expected);
                    seen.push_back(e);
                    // Stand-in for an asynchronous adapter update on this exact slice.
                    CUDA_CHECK(cudaMemsetAsync(expert.Data, 0, expert.bytes(), stream));
                };
                REQUIRE(loader.load_param("blocks[2].experts", target, false, shards > 1,
                                           nullptr, stream, on_expert));
                REQUIRE(cudaStreamQuery(stream) == cudaSuccess);
                CUDA_CHECK(cudaStreamDestroy(stream));
                std::vector<int> expected_ids(4 / shards);
                std::iota(expected_ids.begin(), expected_ids.end(), shard * 4 / shards);
                REQUIRE(seen == expected_ids);
                REQUIRE(values(target) == std::vector<float>(target.nelem(), 0.f));
            }
        }
        dsl::DslWeightLoader uneven(reader, mapping, config, allocator, {0, 3}, {4, 2});
        auto target = allocate(allocator, {1, fused ? 4 : 2, 3});
        REQUIRE_THROWS(uneven.load_param("blocks[2].experts", target, false, true));
    }
}

TEST_CASE("DSL ties resolve after source updates and skip external endpoints", "[weight-loader]") {
    require_gpu();
    Checkpoint checkpoint({{"source", {2, 2}, sequence(4)}});
    SafeTensorsReader reader(checkpoint.path);
    TensorAllocator allocator;
    PretrainedConfig config;
    dsl::MappingTable mapping{
        {"destination", {.kind = Kind::TiedTo, .target = "source"}},
        {"external_destination", {.kind = Kind::TiedTo, .target = "source"}},
        {"unavailable_destination", {.kind = Kind::TiedTo, .target = "external_source"}}};
    dsl::DslWeightLoader loader(reader, mapping, config, allocator);
    auto source = allocate(allocator, {2, 2});
    auto destination = allocate(allocator, {2, 2}, ETensorDType::FP32, EAllocationType::PINNED);
    REQUIRE(loader.load_param("destination", destination, false));
    REQUIRE(loader.load_param("external_destination", destination, false));
    REQUIRE(loader.load_param("unavailable_destination", destination, false));
    REQUIRE(loader.load_param("source", source, false));
    const auto updated = sequence(4, 50);
    CUDA_CHECK(cudaMemcpy(source.Data, updated.data(), source.bytes(), cudaMemcpyHostToDevice));
    loader.resolve_tied_params([&](const std::string& name) -> Tensor& {
        REQUIRE((name == "source" || name == "destination"));
        return name == "source" ? source : destination;
    }, [](const std::string& name) { return name.starts_with("external_"); });
    REQUIRE(values(destination) == updated);

    dsl::DslWeightLoader bad_tie(reader, mapping, config, allocator);
    auto small = allocate(allocator, {1, 2});
    REQUIRE(bad_tie.load_param("destination", small, false));
    REQUIRE_THROWS(bad_tie.resolve_tied_params([&](const std::string& name) -> Tensor& {
        return name == "source" ? source : small;
    }));
}

// cpu_training + LoRA: frozen masters are malloc'ed host buffers (dsl::SharedMasterStore) that are
// page-locked only after the read, so the reader must not hand them to CUDA as device memory.
TEST_CASE("DSL loads fill pageable host masters, with and without a cast", "[weight-loader]") {
    require_gpu();
    Checkpoint checkpoint({{"plain", {2, 3}, sequence(6)}, {"cast", {3, 2}, sequence(6, 10)}});
    SafeTensorsReader reader(checkpoint.path);
    TensorAllocator allocator;
    PretrainedConfig config;
    dsl::MappingTable mapping;
    dsl::DslWeightLoader loader(reader, mapping, config, allocator);
    auto host = [](auto& buffer, ETensorDType dtype, const std::vector<long>& shape) {
        return Tensor::from_pointer(reinterpret_cast<std::byte*>(buffer.data()), /*device=*/-1, dtype, shape);
    };
    auto floats = [](const std::vector<nv_bfloat16>& raw) {
        std::vector<float> result(raw.size());
        std::transform(raw.begin(), raw.end(), result.begin(), [](nv_bfloat16 x) { return float(x); });
        return result;
    };

    std::vector<float> plain_data(6, -1.0f);
    auto plain = host(plain_data, ETensorDType::FP32, {2, 3});
    REQUIRE(loader.load_param("plain", plain, false));
    REQUIRE(plain_data == sequence(6));

    std::vector<nv_bfloat16> cast_data(6);
    auto cast = host(cast_data, ETensorDType::BF16, {3, 2});
    REQUIRE_THROWS(loader.load_param("cast", cast, false));
    REQUIRE(loader.load_param("cast", cast, true));
    REQUIRE(floats(cast_data) == sequence(6, 10));

    // Converting read staged in chunks smaller than the tensor (4 + 2 elements).
    std::uint64_t header_size = 0;
    std::ifstream(checkpoint.path, std::ios::binary).read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    const auto begin = static_cast<std::ptrdiff_t>(sizeof(header_size) + header_size);  // "plain" is stored first
    auto staging = allocate(allocator, {4});
    std::vector<nv_bfloat16> chunked(6);
    cuFileRef file(checkpoint.path);
    file.read_and_convert(reinterpret_cast<std::byte*>(chunked.data()), begin, begin + 6 * sizeof(float),
                          checkpoint.path, ETensorDType::BF16, ETensorDType::FP32, staging.Data, staging.bytes(),
                          /*host_target=*/true);
    REQUIRE(floats(chunked) == sequence(6));
}

TEST_CASE("Shared master store releases waiters when the claimer's read fails", "[weight-loader]") {
    dsl::SharedMasterStore store;
    store.reserve("weight", 16);
    REQUIRE(store.try_claim("weight"));
    auto waiter = std::async(std::launch::async, [&] { store.wait_populated("weight"); });
    store.fail("weight");
    REQUIRE_THROWS(waiter.get());
    REQUIRE_FALSE(store.try_claim("weight"));
    REQUIRE_THROWS(store.wait_populated("weight"));
    store.clear();
}

// cpu_training's shared masters are pageable while they are read: a transpose must not be
// written into them by the kernel, and the loader's device temporaries must not outlive the load.
TEST_CASE("DSL transposes into pageable host masters match device loads", "[weight-loader]") {
    require_gpu();
    const auto dtype = GENERATE(ETensorDType::FP32, ETensorDType::BF16);
    Checkpoint checkpoint({{"matrix", {2, 4}, sequence(8)}, {"experts", {4, 2, 3}, sequence(24)}});
    SafeTensorsReader reader(checkpoint.path);
    TensorAllocator allocator;
    PretrainedConfig config;
    dsl::MappingTable mapping{
        {"matrix", {.kind = Kind::Transform, .source = "matrix", .fn = "transpose"}},
        {"experts", {.kind = Kind::Transform, .source = "experts", .fn = "transpose"}}};
    for (bool batched : {false, true}) {
        CAPTURE(batched);
        const std::string name = batched ? "experts" : "matrix";
        const std::vector<long> shape = batched ? std::vector<long>{4, 3, 2} : std::vector<long>{4, 2};
        dsl::DslWeightLoader loader(reader, mapping, config, allocator);
        auto device = allocate(allocator, shape, dtype);
        REQUIRE(loader.load_param(name, device, true));
        std::vector<std::byte> buffer(device.bytes(), std::byte{0xAB});
        auto host = Tensor::from_pointer(buffer.data(), /*device=*/-1, dtype, shape);
        const auto device_bytes = allocator.total_allocation(EAllocationType::ON_DEVICE);
        REQUIRE(loader.load_param(name, host, true));
        REQUIRE(values(host) == values(device));
        REQUIRE(allocator.total_allocation(EAllocationType::ON_DEVICE) == device_bytes);
    }
}

// #270: what a host master holds after the load must be the file's contents, on every element,
// whichever read route filled it -- not only on sampled slices. Sizes straddle the 8 MiB staging
// chunk, and the BF16 tensor takes the converting read that host masters stage on the device.
TEST_CASE("DSL host-master loads equal device loads and the file on full tensors", "[weight-loader]") {
    require_gpu();
    const bool disable_cufile = GENERATE(false, true);
    CAPTURE(disable_cufile);
    std::mt19937 rng(270);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const auto random_values = [&](long count) {
        std::vector<float> result(static_cast<std::size_t>(count));
        for (float& x : result) x = normal(rng);
        return result;
    };
    const long plain_count = 2 * (8L << 20) / long(sizeof(float)) + 12345;
    const long cast_count = 3 * (8L << 20) / long(sizeof(nv_bfloat16)) + 777;
    const Weight plain{"plain", {plain_count}, random_values(plain_count)};
    const Weight cast{"cast", {cast_count}, random_values(cast_count), ETensorDType::BF16};
    Checkpoint checkpoint({plain, cast});
    ScopedEnv cufile("SUROGATE_DISABLE_CUFILE", disable_cufile ? "1" : nullptr);
    SafeTensorsReader reader(checkpoint.path);
    TensorAllocator allocator;
    PretrainedConfig config;
    dsl::MappingTable mapping;
    dsl::DslWeightLoader loader(reader, mapping, config, allocator);

    for (const Weight* weight : {&plain, &cast}) {
        CAPTURE(weight->name);
        // What an independent reader of the file gets: the stored floats, or the stored BF16
        // values widened exactly.
        std::vector<float> expected = weight->values;
        if (weight->dtype == ETensorDType::BF16) {
            for (float& x : expected) x = __bfloat162float(__float2bfloat16(x));
        }
        auto device = allocate(allocator, weight->shape);
        REQUIRE(loader.load_param(weight->name, device, true));
        std::vector<float> from_device(expected.size());
        CUDA_CHECK(cudaMemcpy(from_device.data(), device.Data, device.bytes(), cudaMemcpyDeviceToHost));
        std::vector<float> host_buffer(expected.size(), std::numeric_limits<float>::quiet_NaN());
        auto host = Tensor::from_pointer(reinterpret_cast<std::byte*>(host_buffer.data()), /*device=*/-1,
                                         ETensorDType::FP32, weight->shape);
        REQUIRE(loader.load_param(weight->name, host, true));
        const std::size_t bytes = expected.size() * sizeof(float);
        REQUIRE(std::memcmp(from_device.data(), expected.data(), bytes) == 0);
        REQUIRE(std::memcmp(host_buffer.data(), expected.data(), bytes) == 0);
    }
}

// A stacked adapter is merged into the base weights during import; under cpu_training the base
// weight is a pageable host master that cuBLAS cannot write.
TEST_CASE("Stacked adapters merge into pageable host masters", "[weight-loader]") {
    require_gpu();
    // W [3, 4], lora_A [2, 4], lora_B [3, 2]; small integers keep every BF16 value exact.
    const std::vector<float> w = sequence(12);
    const std::vector<float> a{1, 0, 2, 1, 0, 1, 1, 2};
    const std::vector<float> b{1, 2, 0, 1, 3, 0};
    Checkpoint base({{"layer.proj.weight", {3, 4}, w}});
    Checkpoint adapter({{"base_model.model.layer.proj.lora_A.weight", {2, 4}, a},
                        {"base_model.model.layer.proj.lora_B.weight", {3, 2}, b}});
    std::filesystem::rename(adapter.path, adapter.directory / "adapter_model.safetensors");
    adapter.path = (adapter.directory / "adapter_model.safetensors").string();
    std::ofstream(adapter.directory / "adapter_config.json") << R"({"r": 2, "lora_alpha": 4})";
    SafeTensorsReader base_reader(base.path);
    dsl::MappingTable mapping{{"proj", {.kind = Kind::Direct, .source = "layer.proj.weight"}}};
    qlora::AdapterMerger merger(adapter.directory.string(), mapping, base_reader);

    std::vector<float> expected = w;  // W + (alpha / r) * B @ A
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            for (int k = 0; k < 2; ++k) expected[row * 4 + col] += 2.0f * b[row * 2 + k] * a[k * 4 + col];
        }
    }

    std::vector<nv_bfloat16> host_buffer(12);
    std::transform(w.begin(), w.end(), host_buffer.begin(), [](float x) { return __float2bfloat16(x); });
    TensorAllocator allocator;
    auto device = allocate(allocator, {3, 4}, ETensorDType::BF16);
    CUDA_CHECK(cudaMemcpy(device.Data, host_buffer.data(), device.bytes(), cudaMemcpyHostToDevice));
    auto host = Tensor::from_pointer(reinterpret_cast<std::byte*>(host_buffer.data()), /*device=*/-1,
                                     ETensorDType::BF16, std::vector<long>{3, 4});
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    merger.apply("proj", device, stream);
    merger.apply("proj", host, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    REQUIRE(values(device) == expected);
    REQUIRE(values(host) == expected);
}

// #270's check on a real checkpoint, run by hand:
//   SUROGATE_LOAD_CHECK_PATH=/path/to/model dsl-weight-loader-tests "[load-check]"
// Every tensor of every *.safetensors file is read in full into a pageable host buffer, as a
// cpu_training shared master is, on the buffered route and the cuFile route, and compared byte for
// byte with an independent read of the file (its own header parse and plain reads). BF16 tensors
// are also read with the BF16 -> FP32 conversion and compared with the exact widening.
TEST_CASE("Host-master loads of a real checkpoint equal the file on every tensor", "[.][load-check]") {
    const char* root = std::getenv("SUROGATE_LOAD_CHECK_PATH");
    if (root == nullptr) SKIP("set SUROGATE_LOAD_CHECK_PATH to a checkpoint file or directory");
    require_gpu();
    std::vector<std::filesystem::path> files;
    if (std::filesystem::is_directory(root)) {
        for (const auto& item : std::filesystem::directory_iterator(root)) {
            if (item.path().extension() == ".safetensors") files.push_back(item.path());
        }
        std::sort(files.begin(), files.end());
    } else {
        files.emplace_back(root);
    }
    REQUIRE_FALSE(files.empty());
    const bool disable_cufile = GENERATE(true, false);
    ScopedEnv cufile("SUROGATE_DISABLE_CUFILE", disable_cufile ? "1" : nullptr);
    std::size_t tensors = 0, converted = 0, mismatches = 0;
    std::uint64_t bytes = 0;
    for (const auto& file : files) {
        std::ifstream in(file, std::ios::binary);
        std::uint64_t header_size = 0;
        in.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        std::string header(header_size, '\0');
        in.read(header.data(), static_cast<std::streamsize>(header_size));
        const auto metadata = nlohmann::json::parse(header);
        const std::uint64_t data_start = sizeof(header_size) + header_size;
        SafeTensorsReader reader(file.string());
        for (const auto& [name, info] : metadata.items()) {
            if (name == "__metadata__") continue;
            const auto& entry = reader.find_entry(name);
            const std::uint64_t begin = info.at("data_offsets").at(0).get<std::uint64_t>();
            const std::uint64_t end = info.at("data_offsets").at(1).get<std::uint64_t>();
            std::vector<std::byte> expected(end - begin);
            in.seekg(static_cast<std::streamoff>(data_start + begin));
            in.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(expected.size()));
            REQUIRE(in.good());

            std::vector<std::byte> host(expected.size(), std::byte{0x5A});
            auto target = Tensor::from_pointer(host.data(), /*device=*/-1, entry.dtype(), entry.shape());
            entry.read_tensor(target, false);
            if (host != expected) {
                ++mismatches;
                WARN("mismatch: " << file.filename().string() << " " << name);
            }
            if (entry.dtype() == ETensorDType::BF16) {
                const std::size_t count = expected.size() / sizeof(std::uint16_t);
                std::vector<float> widened(count, std::numeric_limits<float>::quiet_NaN());
                auto wide = Tensor::from_pointer(reinterpret_cast<std::byte*>(widened.data()), /*device=*/-1,
                                                 ETensorDType::FP32, entry.shape());
                entry.read_tensor(wide, true);
                std::vector<std::uint32_t> exact(count);
                for (std::size_t i = 0; i < count; ++i) {
                    std::uint16_t raw;
                    std::memcpy(&raw, expected.data() + i * sizeof(raw), sizeof(raw));
                    exact[i] = std::uint32_t{raw} << 16;
                }
                if (std::memcmp(widened.data(), exact.data(), count * sizeof(float)) != 0) {
                    ++mismatches;
                    WARN("BF16 -> FP32 mismatch: " << file.filename().string() << " " << name);
                }
                ++converted;
            }
            ++tensors;
            bytes += expected.size();
        }
    }
    INFO(tensors << " tensors (" << converted << " also widened), " << bytes << " bytes, route "
                 << (disable_cufile ? "buffered" : "cuFile"));
    REQUIRE(mismatches == 0);
}
