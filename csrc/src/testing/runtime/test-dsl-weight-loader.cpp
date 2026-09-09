// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_bf16.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include "config/pretrained_config.h"
#include "runtime/dsl/dsl_weight_loader.h"
#include "utilities/allocator.h"
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
};

// Write independent, tiny FP32 SafeTensors fixtures; no model download or exporter needed.
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
            const auto end = offset + weight.values.size() * sizeof(float);
            metadata[weight.name] = {{"dtype", "F32"}, {"shape", weight.shape}, {"data_offsets", {offset, end}}};
            offset = end;
        }
        auto header = metadata.dump();
        header.append((8 - header.size() % 8) % 8, ' ');
        const std::uint64_t header_size = header.size();
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        file.write(header.data(), header.size());
        for (const auto& weight : weights) {
            file.write(reinterpret_cast<const char*>(weight.values.data()), weight.values.size() * sizeof(float));
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
