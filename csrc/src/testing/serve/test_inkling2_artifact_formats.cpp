// Inkling-2 artifact formats in the C++ reader (artifact contract I-1): the two formats the
// Inkling artifact adds (MXFP8, FP8_E4M3FN_1X128_F32S) parse, size and route like the Python
// writer says; every existing format keeps its size. With INKLING2_ARTIFACT set to a converted
// Inkling-2-Mini artifact, the reader also opens it and checks the objects the target binds.
// CPU only: no device is touched.

#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using sinfer::artifact::ArtifactError;
using sinfer::artifact::NumericFormat;
using sinfer::artifact::Reader;
using sinfer::artifact::ResourceDescriptor;
using sinfer::artifact::StorageLayout;
using sinfer::artifact::TensorDescriptor;
using sinfer::artifact::tensor_encoded_size;
using Json = nlohmann::json;
using sinfer::test::artifact_fixture::write_fixture;

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error("FAIL: " + what); }
}

template <class F>
void expect_artifact_error(F&& f, const std::string& what) {
    try {
        f();
    } catch (const ArtifactError&) {
        return;
    }
    throw std::runtime_error("FAIL: expected ArtifactError: " + what);
}

void test_sizes() {
    // Python: layouts.mxfp8_geometry("MXFP8", (3, 64)) -> codes 192, scales at 256, 6 bytes: 262.
    const std::array<std::uint64_t, 2> mx = {3, 64};
    require(tensor_encoded_size(StorageLayout::Mxfp8K32V1, NumericFormat::MXFP8, mx) == 262, "MXFP8 [3,64]");
    const auto g = sinfer::artifact::mxfp8_geometry(NumericFormat::MXFP8, mx);
    require(g.groups_per_row == 2 && g.code_plane_bytes == 192 && g.scale_plane_offset == 256 &&
                g.scale_plane_bytes == 6, "MXFP8 geometry");
    // Inkling-2-Mini's w13 bank: [128*2304, 2304] -> 679,477,248 codes + 21,233,664 scales.
    const std::array<std::uint64_t, 2> w13 = {128ull * 2304, 2304};
    require(tensor_encoded_size(StorageLayout::Mxfp8K32V1, NumericFormat::MXFP8, w13) == 700710912ull,
            "MXFP8 w13 bank");
    // Python: fp8_grid_geometry("FP8_E4M3FN_1X128_F32S", (24, 256)) -> 6144 + 24*2*4.
    const std::array<std::uint64_t, 2> grid = {24, 256};
    require(tensor_encoded_size(StorageLayout::Fp8Grid1x128F32V1, NumericFormat::FP8_E4M3FN_1X128_F32S, grid) ==
                6144 + 192, "FP8 1x128 [24,256]");
    expect_artifact_error([] {
        const std::array<std::uint64_t, 2> bad = {2, 48};
        (void)tensor_encoded_size(StorageLayout::Mxfp8K32V1, NumericFormat::MXFP8, bad);
    }, "MXFP8 k not a whole number of 32s");
    expect_artifact_error([] {
        const std::array<std::uint64_t, 2> bad = {2, 64};
        (void)tensor_encoded_size(StorageLayout::Fp8Grid1x128F32V1, NumericFormat::FP8_E4M3FN_1X128_F32S, bad);
    }, "FP8 1x128 k not a whole number of 128s");
    expect_artifact_error([] {
        (void)tensor_encoded_size(StorageLayout::Mxfp8K32V1, NumericFormat::NVFP4,
                                  std::array<std::uint64_t, 2>{128, 64});
    }, "MXFP8 layout with another format");
    // Unchanged neighbours.
    require(tensor_encoded_size(StorageLayout::RowScaleF32V1, NumericFormat::FP8_E4M3FN_ROW_F32S,
                                std::array<std::uint64_t, 2>{2, 128}) == 256 + 8, "ROW_F32S unchanged");
    require(sinfer::artifact::format_name(NumericFormat::MXFP8) == "MXFP8", "MXFP8 name");
    require(sinfer::artifact::layout_name(StorageLayout::Fp8Grid1x128F32V1) == "fp8-1x128-f32-v1", "1x128 layout name");
}

void test_fixture_parses() {
    const Json directory = {
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}, {"architecture", "inkling2"}}},
        {"objects", Json::array({
             {{"name", "mx"}, {"kind", "tensor"}, {"shape", {3, 64}}, {"format", "MXFP8"},
              {"layout", "mxfp8-k32-v1"}, {"offset", 0}, {"bytes", 262}},
             {{"name", "grid"}, {"kind", "tensor"}, {"shape", {24, 256}}, {"format", "FP8_E4M3FN_1X128_F32S"},
              {"layout", "fp8-1x128-f32-v1"}, {"offset", 512}, {"bytes", 6336}},
         })},
    };
    const auto fixture = write_fixture(directory, "inkling2_formats");
    Reader reader(fixture.path);
    const auto* mx   = std::get_if<TensorDescriptor>(reader.find("mx"));
    const auto* grid = std::get_if<TensorDescriptor>(reader.find("grid"));
    require(mx && mx->format == NumericFormat::MXFP8 && mx->layout == StorageLayout::Mxfp8K32V1, "mx parses");
    require(grid && grid->format == NumericFormat::FP8_E4M3FN_1X128_F32S &&
                grid->layout == StorageLayout::Fp8Grid1x128F32V1, "grid parses");
}

void test_real_artifact() {
    const char* path = std::getenv("INKLING2_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout << "INKLING2_ARTIFACT not set: real-artifact checks skipped\n";
        return;
    }
    Reader reader(path);
    require(reader.identity().architecture == "inkling2", "architecture");
    require(reader.geometry().at("hidden") == 2304 && reader.geometry().at("layers") == 24, "geometry");
    require(reader.layer_types().size() == 24 && reader.layer_types()[5] == "full_attention" &&
                reader.layer_types()[0] == "sliding_attention", "layer types");
    std::size_t tensors = 0, resources = 0, mxfp8 = 0, nvfp4 = 0;
    for (const auto& object : reader.objects()) {
        if (const auto* t = std::get_if<TensorDescriptor>(&object)) {
            ++tensors;
            mxfp8 += t->format == NumericFormat::MXFP8;
            nvfp4 += t->format == NumericFormat::NVFP4;
        } else {
            ++resources;
        }
    }
    require(mxfp8 == 8 && nvfp4 == 40, "expert banks: 8 MXFP8, 40 NVFP4");
    const auto* w13 = std::get_if<TensorDescriptor>(reader.find("text/layers/20/moe/w13"));
    require(w13 && w13->shape == std::vector<std::uint64_t>{128ull * 2304, 2304}, "layer 20 w13");
    const auto* qkvr = std::get_if<TensorDescriptor>(reader.find("text/layers/5/attn/qkvr"));
    require(qkvr && qkvr->shape == std::vector<std::uint64_t>{8600, 2304}, "layer 5 qkvr");
    require(std::get_if<ResourceDescriptor>(reader.find("inkling2/config.json")) != nullptr, "config resource");
    const auto config = reader.payload("inkling2/config.json");
    const auto doc = Json::parse(reinterpret_cast<const char*>(config.data.data()),
                                 reinterpret_cast<const char*>(config.data.data()) + config.data.size());
    require(doc.at("visits").at("count") == 48 && doc.at("schema") == "inkling2-artifact/1", "config document");
    std::cout << "real artifact: " << tensors << " tensors, " << resources << " resources\n";
}

} // namespace

int main() {
    try {
        test_sizes();
        test_fixture_parses();
        test_real_artifact();
        std::cout << "PASS inkling2 artifact formats\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
