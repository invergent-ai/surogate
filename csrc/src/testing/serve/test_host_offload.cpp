#include "artifact_fixture.h"
#include "artifact/typed_binding.h"
#include "family/impl/load/host_bank.h"
#include "family/impl/load/host_decode.h"
#include "family/impl/moe/expert_cache.h"
#include "core/engine_context.h"
#include "core/device.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace sinfer;
using artifact::NumericFormat;
using artifact::StorageLayout;

namespace {
struct Advice { void* address; std::size_t bytes; int kind; };
thread_local std::vector<Advice>* recorded_advice = nullptr;
std::atomic<int> live_pinned{0};
}

extern "C" cudaError_t CUDARTAPI cudaHostRegister(void* pointer, std::size_t bytes, unsigned flags) {
    static const auto real = reinterpret_cast<decltype(&cudaHostRegister)>(dlsym(RTLD_NEXT, "cudaHostRegister"));
    const auto result = real(pointer, bytes, flags);
    if (result == cudaSuccess) { ++live_pinned; }
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaHostUnregister(void* pointer) {
    static const auto real = reinterpret_cast<decltype(&cudaHostUnregister)>(dlsym(RTLD_NEXT, "cudaHostUnregister"));
    const auto result = real(pointer);
    if (result == cudaSuccess) { --live_pinned; }
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaHostAlloc(void** pointer, std::size_t bytes, unsigned flags) {
    using Allocate = cudaError_t(CUDARTAPI*)(void**, std::size_t, unsigned);
    static const auto real = reinterpret_cast<Allocate>(dlsym(RTLD_NEXT, "cudaHostAlloc"));
    const auto result = real(pointer, bytes, flags);
    if (result == cudaSuccess) { ++live_pinned; }
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaFreeHost(void* pointer) {
    static const auto real = reinterpret_cast<decltype(&cudaFreeHost)>(dlsym(RTLD_NEXT, "cudaFreeHost"));
    const auto result = real(pointer);
    if (pointer && result == cudaSuccess) { --live_pinned; }
    return result;
}

extern "C" int madvise(void* address, std::size_t bytes, int kind) noexcept {
    static const auto real = reinterpret_cast<decltype(&madvise)>(dlsym(RTLD_NEXT, "madvise"));
    if (recorded_advice && (kind == MADV_SEQUENTIAL || kind == MADV_WILLNEED)) {
        recorded_advice->push_back({address, bytes, kind});
    }
    return real(address, bytes, kind);
}

namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
void put_half(std::byte* out, std::uint16_t bits) { std::memcpy(out, &bits, 2); }

void test_readahead() {
    const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    struct Mapping {
        std::size_t bytes;
        void* data;
        explicit Mapping(std::size_t n) : bytes(n), data(mmap(nullptr, n, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)) {
            require(data != MAP_FAILED, "source mmap failed");
        }
        ~Mapping() { munmap(data, bytes); }
    } mapping(page * 6);
    auto* bytes = static_cast<std::byte*>(mapping.data);
    for (std::size_t i = 0; i < mapping.bytes; ++i) { bytes[i] = std::byte(i % 251); }
    for (bool split : {false, true}) {
        family::HostObjectPlan source;
        source.name = "readahead";
        source.handle = {0};
        if (split) { source.parts = {{bytes + 31, page + 9}, {}, {bytes + page * 3 + 47, page + 101}}; }
        else { source.payload = {bytes + 17, page + 49}; }
        std::vector<Advice> observed;
        observed.reserve(64);
        recorded_advice = &observed;
        struct Reset { ~Reset() { recorded_advice = nullptr; } } reset;
        const family::HostBank bank({.objects={source}});
        recorded_advice = nullptr;
        auto spans = source.parts;
        if (spans.empty()) { spans.push_back(source.payload); }
        std::vector<std::byte> expected;
        for (const auto span : spans) {
            if (span.empty()) { continue; }
            expected.insert(expected.end(), span.begin(), span.end());
            const auto address = reinterpret_cast<std::uintptr_t>(span.data());
            const auto begin = address / page * page;
            const auto size = span.size() + address - begin;
            for (int kind : {MADV_SEQUENTIAL, MADV_WILLNEED}) {
                require(std::any_of(observed.begin(), observed.end(), [&](const Advice& a) {
                    return a.address == reinterpret_cast<void*>(begin) && a.bytes == size && a.kind == kind;
                }), "host bank did not advise every source run");
            }
        }
        require(std::none_of(observed.begin(), observed.end(), [](const Advice& a) {
            return a.address == nullptr || a.bytes == 0;
        }), "host bank advised an empty range");
        const auto& object = bank.object(source.handle);
        require(object.bytes == expected.size() && std::memcmp(object.host, expected.data(), expected.size()) == 0,
                "multi-run host-bank copy changed payload bytes");
    }
}

void test_host_codecs() {
    for (auto format : {NumericFormat::Q4G64_F16S, NumericFormat::Q5G64_F16S,
                        NumericFormat::Q6G64_F16S, NumericFormat::W8G32_F16S}) {
        const std::array<std::uint64_t, 2> shape{2, 704};
        const auto g = artifact::row_split_geometry(format, shape);
        std::vector<std::byte> bytes(g.encoded_bytes);
        std::fill_n(bytes.data(), g.low_plane_bytes,
                    format == NumericFormat::W8G32_F16S ? std::byte{7} : std::byte{0x77});
        std::fill_n(bytes.data() + g.high_plane_offset, g.high_plane_bytes, std::byte{0xff});
        for (std::uint64_t i = 0; i < g.scale_plane_bytes; i += 2) {
            put_half(bytes.data() + g.scale_plane_offset + i, 0x3c00);
        }
        family::HostObjectPlan source;
        source.payload = bytes; source.decode_k = 704;
        source.source_tensor.shape = {2, 704}; source.source_tensor.format = format;
        source.source_tensor.layout = StorageLayout::RowSplitK128V1;
        std::array<float, 704> decoded;
        family::decode_host_row(source, 1, decoded.data());
        const float expected = g.high_plane_bytes == 0 ? 7.F : -9.F;
        for (float value : decoded) { require(value == expected, "padded row-split decode changed a value"); }
    }
    {
        const std::array<std::uint64_t, 2> shape{256, 64};
        const auto g = artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
        std::vector<std::byte> bytes(g.encoded_bytes);
        std::fill_n(bytes.data(), 128 * 32, std::byte{0x22});
        std::fill_n(bytes.data() + 128 * 32, 128 * 32, std::byte{0x44});
        std::fill_n(bytes.data() + g.scale_plane_offset, g.scale_plane_bytes, std::byte{0x38});
        const float divisor = 1.F; std::memcpy(bytes.data() + g.weight_divisor_offset, &divisor, 4);
        family::HostObjectPlan source;
        source.payload = bytes; source.decode_k = 64; source.swap_half_rows = 128;
        source.scale_rows = 128; source.row_scales = {3.F, 5.F};
        source.source_tensor.shape = {256, 64}; source.source_tensor.format = NumericFormat::NVFP4;
        source.source_tensor.layout = StorageLayout::BlockScaleK16M128x4V1;
        std::array<float, 64> decoded;
        for (int row : {0, 31, 32, 127, 128, 255}) {
            family::decode_host_row(source, row, decoded.data());
            for (float value : decoded) {
                require(value == (row < 128 ? 10.F : 3.F), "NVFP4 lost tiling, scales, or gate/up ordering");
            }
        }
    }
    {
        std::vector<std::byte> first(68), second(128);
        for (int group = 0; group < 2; ++group) {
            put_half(first.data() + group * 34, 0x3c00);
            std::fill_n(first.data() + group * 34 + 2, 32, std::byte(group + 1));
            for (int i = 0; i < 32; ++i) { put_half(second.data() + (group * 32 + i) * 2, group == 0 ? 0x4200 : 0x4400); }
        }
        family::HostObjectPlan source;
        source.parts = {first, second}; source.decode_k = 64;
        source.source_tensor.shape = {2, 64}; source.source_tensor.format = NumericFormat::Q8_0;
        source.source_tensor.layout = StorageLayout::GgmlBlocksV1;
        source.source_tensor.segments = {{NumericFormat::Q8_0, 1}, {NumericFormat::F16, 1}};
        source.source_tensor.group_map = {1, 0};
        std::array<float, 64> decoded;
        for (int row = 0; row < 2; ++row) {
            family::decode_host_row(source, row, decoded.data());
            require(decoded[0] == row * 2 + 2 && decoded[32] == row * 2 + 1,
                    "host expert lost a segment format or column permutation");
        }
    }
}

void test_decode_failure_recovery() {
    const int baseline = live_pinned;
    for (bool odd_width : {true, false}) {
        std::vector<std::uint16_t> values(4 * (odd_width ? 33 : 32), 0x3f80);
        family::HostObjectPlan source;
        source.name = "failing decode";
        source.handle = {1};
        source.payload = std::as_bytes(std::span(values));
        source.decode_type = QType::BF16_CTRL;
        source.decode_rows = 4;
        source.decode_k = odd_width ? 33 : 32;
        if (!odd_width) {
            // Force a real row-conversion failure after workers have started.
            source.q5_rows = 4;
            source.q5_k = 32;
        }
        auto first = source;
        first.handle = {0};
        first.name = "already loaded";
        first.decode_rows = first.q5_rows = 0;
        const family::HostBankPlan plan{.objects={first, source}};
        for (int attempt = 0; attempt < 3; ++attempt) {
            bool rejected = false;
            try { (void)family::HostBank::shared(plan); }
            catch (const std::exception& error) {
                rejected = std::string(error.what()).find(source.name) != std::string::npos;
            }
            require(rejected, "invalid host row did not report a recoverable load error");
            require(live_pinned == baseline, "failed host-bank construction leaked pinned allocations");
        }
    }
    std::vector<std::uint16_t> values(128, 0x3f80);
    family::HostObjectPlan source;
    source.name = "valid decode";
    source.handle = {0};
    source.payload = std::as_bytes(std::span(values));
    source.decode_rows = 4; source.decode_k = 32; source.decode_type = QType::BF16_CTRL;
    {
        const auto bank = family::HostBank::shared({.objects={source}});
        const auto& object = bank->object(source.handle);
        const auto* codes = static_cast<const std::int8_t*>(object.host);
        require(object.bytes == 136 && std::all_of(codes, codes + 128, [](auto code) { return code == 127; }),
                "valid host decode failed after a rejected bank");
    }
    require(live_pinned == baseline, "valid decoded bank leaked pinned allocations");
}

void test_slot_limits() {
    const ops::SparseMoeGeometry geometry{128, 256, 8, 128};
    require(family::ExpertCache::pool_floor_bytes(geometry, 2, 0) ==
                family::ExpertCache::pool_floor_bytes(geometry, 2, 64),
            "automatic cache minimum must permit a bounded expert batch");
    require(family::ExpertCache::pool_floor_bytes(geometry, 2, 8) <
                family::ExpertCache::pool_floor_bytes(geometry, 2, 256),
            "one token's selected experts should need less memory than a layer");
    for (std::uint32_t slots : {1U, 7U, 0xffffffffU, 1U << 24}) {
        bool rejected = false;
        try { (void)family::ExpertCache::pool_floor_bytes(geometry, 2, slots); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid expert slot count was silently clamped or overflowed");
    }
}

void test_placement_and_isolation() {
    using test::artifact_fixture::Json;
    Json objects = Json::array();
    const std::vector<std::string> names{
        "text/layers/0/mlp/gate_up", "text/layers/1/moe/routed_gate_up",
        "text/layers/1/moe/routed_down", "text/layers/1/input_norm",
        "text/layers/2/moe/routed_gate_up", "text/layers/2/moe/routed_down"};
    for (std::size_t i = 0; i < names.size(); ++i) {
        objects.push_back({{"name", names[i]}, {"kind", "tensor"}, {"shape", {1}},
            {"format", "BF16"}, {"layout", "contiguous-le-v1"}, {"offset", i * 256}, {"bytes", 2}});
    }
    auto fixture = test::artifact_fixture::write_fixture(
        {{"identity", {{"model_id", "offload-test"}, {"weights_id", "test"}}}, {"objects", objects}}, "host-offload");
    artifact::Reader reader(fixture.path), another_reader(fixture.path);
    const auto plan = [&](artifact::Reader& r) {
        artifact::Binder binder(r); binder.set_offload(2, 1);
        for (const auto& name : names) { (void)artifact::bind_tensor(binder, name, NumericFormat::BF16, {1}, artifact::TensorPlacement::Device); }
        auto materialization = binder.finish();
        require(materialization.bank_objects.size() == 4, "offload counts must count actual MoE layers");
        require(materialization.device_objects.size() == 2, "offload moved a resident tensor");
        return family::collect_host_bank(binder, materialization);
    };
    const auto first_plan = plan(reader), second_plan = plan(another_reader);
    const auto first = family::HostBank::shared(first_plan);
    require(first == family::HostBank::shared(first_plan), "same reader's stages did not share host weights");
    require(first != family::HostBank::shared(second_plan), "separate models shared host weights");

    artifact::Binder components(reader);
    components.set_offload(std::nullopt, 0, 0, true, true, true);
    require(components.offloads("vision/layers/0/attention/qkv") &&
                components.offloads("vision/patch_embedding") &&
                components.offloads("text/token_embedding") &&
                components.offloads("text/output_head") &&
                !components.offloads("text/layers/0/input_norm"),
            "component offload selected the wrong weights");
    artifact::Binder tied(reader);
    tied.set_offload(std::nullopt, 0, 0, false, false, true);
    require(tied.offloads("text/token_embedding"), "tied output head did not offload its embedding storage");

    ops::EngineOpsContext first_context, second_context;
    const ops::SparseMoeGeometry geometry{128, 2, 1, 128};
    EngineOptions options; options.expert_slots = 2;
    ops::bind_ops_context(&first_context);
    family::ExpertCache::configure(options, 0, 2);
    auto* first_cache = &family::ExpertCache::for_current_device(geometry, 2);
    ops::bind_ops_context(&second_context);
    family::ExpertCache::configure(options, 0, 2);
    auto* second_cache = &family::ExpertCache::for_current_device(geometry, 2);
    require(first_cache != second_cache, "separate engines shared their expert cache");
    ops::bind_ops_context(&first_context);
    require(first_cache == &family::ExpertCache::for_current_device(geometry, 2), "an engine lost its own cache");
    ops::bind_ops_context(nullptr);
}
} // namespace

int main() {
    try {
        test_host_codecs();
        test_slot_limits();
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { std::cout << "SKIP: CUDA unavailable\n"; return 77; }
        DeviceContext device(0);
        test_readahead();
        test_decode_failure_recovery();
        test_placement_and_isolation();
        std::cout << "ok\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
