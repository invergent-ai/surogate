#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "artifact_fixture.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::array<std::byte, 3> kResource = {
    std::byte{1},
    std::byte{1},
    std::byte{1},
};
constexpr std::array<std::byte, 4> kTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
constexpr std::array<std::byte, 8> kSecondTensor = {
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
};
constexpr std::size_t kFp8TensorBytes = 260;
constexpr std::size_t kTailReadBytes  = 256 + kFp8TensorBytes;

sinfer::test::artifact_fixture::TemporaryArtifact write_fixture() {
    using Json = sinfer::test::artifact_fixture::Json;
    return sinfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({
                            {{"name", "frontend/test.json"},
                             {"kind", "resource"},
                             {"encoding", "raw-bytes-v1"},
                             {"offset", 0},
                             {"bytes", 3}},
                            {{"name", "weights/test"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 256},
                             {"bytes", 4}},
                            {{"name", "weights/second"},
                             {"kind", "tensor"},
                             {"shape", {4}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 8192},
                             {"bytes", 8}},
                            {{"name", "weights/fp8"},
                             {"kind", "tensor"},
                             {"shape", {2, 4}},
                             {"format", "FP8_E4M3FN_ROW_BF16S"},
                             {"layout", "row-scale-v1"},
                             {"offset", 8448},
                             {"bytes", kFp8TensorBytes}},
                        })},
        },
        "materialization");
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    try {
        auto fixture = write_fixture();
        sinfer::artifact::Reader reader(fixture.path);
        sinfer::artifact::Binder validation_binder(reader);
        const auto validated_resource = validation_binder.require_resource(
            "frontend/test.json", sinfer::artifact::ResourceEncoding::RawBytesV1);
        validation_binder.retain_on_host(validated_resource);
        constexpr std::array<std::uint64_t, 1> validated_shape = {2};
        const auto validated_only                              = validation_binder.require_tensor(
            "weights/test", sinfer::artifact::NumericFormat::BF16,
            sinfer::artifact::StorageLayout::ContiguousLeV1, validated_shape);
        validation_binder.validate_only(validated_only);
        constexpr std::array<std::uint64_t, 1> retained_shape = {4};
        const auto retained_tensor                            = validation_binder.require_tensor(
            "weights/second", sinfer::artifact::NumericFormat::BF16,
            sinfer::artifact::StorageLayout::ContiguousLeV1, retained_shape);
        validation_binder.materialize_on_device(retained_tensor);
        constexpr std::array<std::uint64_t, 2> fp8_shape = {2, 4};
        const auto validated_fp8                         = validation_binder.require_tensor(
            "weights/fp8", sinfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
            sinfer::artifact::StorageLayout::RowScaleV1, fp8_shape);
        validation_binder.validate_only(validated_fp8);
        const auto validation_plan = validation_binder.finish();
        require(validation_plan.object_count == 4 && validation_plan.host_objects.size() == 1 &&
                    validation_plan.device_objects.size() == 1 &&
                    validation_plan.device_capacity_bytes == kSecondTensor.size(),
                "validate-only tensor was included in the materialization plan");

        int device_count              = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(count_error);
        if (device_count == 0) {
            std::cout << "SKIP: no CUDA devices\n";
            return 77;
        }

        sinfer::artifact::Binder binder(reader);

        const auto resource = binder.require_resource(
            "frontend/test.json", sinfer::artifact::ResourceEncoding::RawBytesV1);
        binder.retain_on_host(resource);
        constexpr std::array<std::uint64_t, 1> second_shape = {4};
        const auto second =
            binder.require_tensor("weights/second", sinfer::artifact::NumericFormat::BF16,
                                  sinfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
        binder.materialize_on_device(second);

        // Bind in the opposite order from the artifact. Device placement order and file read order
        // are intentionally independent, exercising the direct-I/O scatter path.
        constexpr std::array<std::uint64_t, 1> tensor_shape = {2};
        const auto tensor =
            binder.require_tensor("weights/test", sinfer::artifact::NumericFormat::BF16,
                                  sinfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
        binder.materialize_on_device(tensor);

        const auto fp8 = binder.require_tensor(
            "weights/fp8", sinfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
            sinfer::artifact::StorageLayout::RowScaleV1, fp8_shape);
        binder.materialize_on_device(fp8);

        const sinfer::artifact::MaterializationPlan plan = binder.finish();
        require(plan.object_count == 4 && plan.host_objects.size() == 1 &&
                    plan.device_objects.size() == 3 && plan.device_capacity_bytes == 772,
                "binder produced the wrong materialization plan");

        sinfer::DeviceContext device(0);
        auto materialized = sinfer::artifact::materialize(reader, plan, device);

        std::array<std::byte, kTensor.size()> copied{};
        CUDA_CHECK(cudaMemcpy(copied.data(), materialized.device_data(tensor), copied.size(),
                              cudaMemcpyDeviceToHost));
        require(copied == kTensor, "device tensor payload differs from the artifact");
        std::array<std::byte, kSecondTensor.size()> second_copied{};
        CUDA_CHECK(cudaMemcpy(second_copied.data(), materialized.device_data(second),
                              second_copied.size(), cudaMemcpyDeviceToHost));
        require(second_copied == kSecondTensor,
                "second device tensor payload differs from the artifact");
        std::array<std::byte, kFp8TensorBytes> fp8_copied{};
        CUDA_CHECK(cudaMemcpy(fp8_copied.data(), materialized.device_data(fp8), fp8_copied.size(),
                              cudaMemcpyDeviceToHost));
        require(std::all_of(fp8_copied.begin(), fp8_copied.end(),
                            [](std::byte value) { return value == std::byte{4}; }),
                "FP8 device tensor payload differs from the artifact");

        const sinfer::Weight fp8_weight = sinfer::artifact::materialized_weight(
            materialized, fp8, sinfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S, 2, 4);
        require(fp8_weight.qtype == sinfer::QType::FP8_E4M3FN_ROW_BF16S &&
                    fp8_weight.layout == sinfer::QuantLayout::RowScale &&
                    fp8_weight.scale_dtype == sinfer::DType::BF16 && fp8_weight.n == 2 &&
                    fp8_weight.k == 4 && fp8_weight.group == 4 && fp8_weight.group_size == 4 &&
                    fp8_weight.qdata == fp8_weight.payload && fp8_weight.qhigh == nullptr &&
                    fp8_weight.scales == static_cast<const std::byte*>(fp8_weight.payload) + 256 &&
                    fp8_weight.payload_bytes == kFp8TensorBytes,
                "materialized FP8 Weight metadata is incomplete");

        const auto retained = materialized.resource_bytes(resource);
        require(std::equal(retained.begin(), retained.end(), kResource.begin(), kResource.end()),
                "retained resource payload differs from the artifact");

        const auto& stats = materialized.stats();
        require(stats.tensor_count == 3 && stats.resource_count == 1 &&
                    stats.h2d_bytes == kTensor.size() + kSecondTensor.size() + kFp8TensorBytes &&
                    stats.retained_resource_bytes == kResource.size() &&
                    stats.file_bytes == kResource.size() +
                                            sinfer::artifact::Reader::direct_io_alignment +
                                            kTailReadBytes,
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == plan.device_capacity_bytes &&
                    materialized.device_arena().used() == plan.device_capacity_bytes,
                "materialized tensor does not own the planned device backing");

        // Shared serving must use the caller's bytes, never the file payload,
        // and destroying the artifact must not release the caller's storage.
        sinfer::DeviceBuffer owner(kSecondTensor.size());
        owner.fill(9);
        const sinfer::BorrowedTensor binding{"weights/second", owner.p, {4}, owner.bytes, 0};
        {
            const std::array borrowed{binding};
            auto shared = sinfer::artifact::materialize(reader, validation_plan, device, nullptr, borrowed);
            require(shared.device_data(retained_tensor) == owner.p, "borrowed pointer was replaced");
            require(shared.stats().h2d_bytes == 0 && shared.stats().device_capacity_bytes == 0 &&
                        shared.device_arena().capacity() == 0,
                    "borrowed weights allocated or uploaded base storage");
            const auto bytes = shared.resource_bytes(validated_resource);
            require(std::equal(bytes.begin(), bytes.end(), kResource.begin(), kResource.end()),
                    "shared materialization lost frontend resources");
            owner.fill(7);
            CUDA_CHECK(cudaMemcpy(second_copied.data(), shared.device_data(retained_tensor),
                                  second_copied.size(), cudaMemcpyDeviceToHost));
            require(std::all_of(second_copied.begin(), second_copied.end(),
                                [](std::byte b) { return b == std::byte{7}; }),
                    "borrowed tensor does not see its owner's storage");
        }
        owner.fill(8);
        owner.copy_to_host(second_copied.data(), second_copied.size());
        require(second_copied[0] == std::byte{8}, "borrowed destruction released owner storage");
        auto rejects = [&](std::vector<sinfer::BorrowedTensor> borrowed) {
            bool rejected = false;
            try {
                auto invalid = sinfer::artifact::materialize(reader, validation_plan, device, nullptr, borrowed);
            } catch (const sinfer::artifact::ArtifactError&) { rejected = true; }
            require(rejected, "invalid borrowed binding was accepted");
        };
        auto invalid = binding;
        invalid.name = "missing";
        rejects({invalid});
        invalid = binding;
        invalid.shape = {2, 2};
        rejects({invalid});
        invalid = binding;
        invalid.bytes = 4;
        rejects({invalid});
        invalid = binding;
        invalid.device = 1;
        rejects({invalid});
        invalid = binding;
        invalid.dtype = sinfer::SharedWeightDType::FP32;
        rejects({invalid});
        rejects({binding, binding});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
