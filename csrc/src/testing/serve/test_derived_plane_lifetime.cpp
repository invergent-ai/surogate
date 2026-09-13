#include "core/arena.h"
#include "core/device.h"
#include "core/engine_context.h"
#include "core/sleep.h"
#include "ops/linear/marlin/marlin_plane.h"
#include "ops/linear/marlin/marlin_repack.h"
#include "ops/linear/w8a8/w8fp8_plane.h"
#include "ops/linear/w8a8/w4fp4_plane.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace sinfer;
namespace detail = sinfer::ops::detail;

namespace {
void require(bool value, const char* message) { if (!value) { throw std::runtime_error(message); } }
struct Context {
    ops::EngineOpsContext ops;
    ~Context() { ops::bind_ops_context(nullptr); }
};
struct Fixture {
    static constexpr int n = 256, k = 512;
    DeviceBuffer codes{n * k}, scales{n * k / 32 * 2};
    Weight weight{};
    Fixture() {
        codes.fill(7);
        std::vector<std::uint16_t> values(n * k / 32, 0x3c00);
        scales.copy_from_host(values.data(), scales.bytes);
        weight.qtype = QType::W8G32_F16S;
        weight.layout = QuantLayout::RowSplit;
        weight.scale_dtype = DType::FP16;
        weight.group = weight.group_size = 32;
        weight.ndim = 2;
        weight.n = weight.shape[0] = weight.padded_shape[0] = n;
        weight.k = weight.shape[1] = weight.padded_shape[1] = k;
        weight.qdata = codes.p;
        weight.scales = scales.p;
    }
};
struct View { const void* pointer; std::size_t bytes; };

std::vector<View> derive(Context& context, const Fixture& fixture, bool sleepable) {
    ops::bind_ops_context(&context.ops);
    set_sleepable_allocations(sleepable);
    detail::w8fp8_plane_set_enabled(true);
    detail::marlin_plane_set_enabled(true);
    detail::w8_prefill_quant_set_mode(detail::PrefillQuantMode::Fp4);
    const auto fp8 = detail::w8fp8_plane_for(fixture.weight, nullptr);
    const auto fp4 = detail::w4fp4_plane_for(fixture.weight, nullptr);
    const auto marlin = detail::marlin_plane_for(fixture.weight, nullptr);
    const auto scratch = detail::marlin_scratch();
    void* fused = detail::marlin_fused_parent(8192, nullptr);
    const float* one = detail::w4fp4_alpha_one();
    require(fp8.codes && fp4.codes && marlin.b_packed && scratch.c_tmp && scratch.locks && fused && one,
            "derived plane or scratch allocation declined");
    CUDA_CHECK(cudaMemset(scratch.c_tmp, 0x52, 256));
    CUDA_CHECK(cudaMemset(fused, 0x2b, 8192));
    CUDA_CHECK(cudaDeviceSynchronize());
    constexpr auto n = Fixture::n, k = Fixture::k;
    return {{fp8.codes, n * k}, {fp8.row_scales, n * sizeof(float)},
            {fp4.codes, n * k / 2}, {fp4.sf, n * k / 16},
            {fp4.sf_atom, detail::w4fp4_sf_atom_bytes(n, k)}, {fp4.row_scales, n * sizeof(float)},
            {marlin.b_packed, detail::marlin_b_out_words(n, k) * sizeof(std::uint32_t)},
            {marlin.scales, n * k / 32 * 2}, {scratch.a_pad, 256}, {scratch.c_tmp, 256},
            {static_cast<const char*>(static_cast<const void*>(scratch.locks)) - 256, 512},
            {fused, 8192}, {one, sizeof(float)}};
}

std::vector<std::byte> read(const std::vector<View>& views) {
    std::size_t bytes = 0;
    for (const auto& view : views) { bytes += view.bytes; }
    std::vector<std::byte> result(bytes);
    std::size_t offset = 0;
    for (const auto& view : views) {
        CUDA_CHECK(cudaMemcpy(result.data() + offset, view.pointer, view.bytes, cudaMemcpyDeviceToHost));
        offset += view.bytes;
    }
    return result;
}

void test_owners(bool sleepable) {
    Fixture fixture;
    auto first = std::make_unique<Context>(), second = std::make_unique<Context>();
    const void* owner = &first->ops;
    const auto a = derive(*first, fixture, sleepable);
    const auto expected = read(a);
    const auto logical = detail::marlin_plane_bytes() + detail::w8_derived_plane_bytes();
    require(logical > 0, "plane accounting is empty");
    ops::bind_ops_context(&second->ops);
    require(detail::marlin_plane_bytes() == 0 && detail::w8_derived_plane_bytes() == 0,
            "a fresh engine inherited another engine's plane accounting");
    const auto b = derive(*second, fixture, sleepable);
    require(a.front().pointer != b.front().pointer, "two engines shared a derived plane keyed by the same weight address");
    const auto second_expected = read(b);
    detail::w8_prefill_quant_set_mode(detail::PrefillQuantMode::Fp8);
    detail::w8fp8_plane_set_enabled(false);
    ops::bind_ops_context(&first->ops);
    require(detail::w8_prefill_quant_mode() == detail::PrefillQuantMode::Fp4 && detail::w8fp8_plane_enabled(),
            "an engine changed another engine's plane configuration");

    DeviceBuffer output(expected.size());
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    std::size_t offset = 0;
    for (const auto& view : a) {
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(output.p) + offset, view.pointer,
                                   view.bytes, cudaMemcpyDeviceToDevice, stream));
        offset += view.bytes;
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    const auto owned = sleep_owned_bytes(owner, 0);
    if (sleepable) {
        require(owned >= logical, "derived planes are missing from sleep accounting");
        const auto other = sleep_owned_bytes(&second->ops, 0);
        for (int round = 0; round < 2; ++round) {
            const auto free_before_sleep = device_free_bytes(0);
            require(sleep_device(0, owner) == owned && device_asleep(0, owner) &&
                    device_free_bytes(0) + (1U << 20) >= free_before_sleep + owned &&
                    sleep_device(0, owner) == 0,
                    "sleep did not release all derived-plane memory");
            require(sleep_owned_bytes(&second->ops, 0) == other && read(b) == second_expected,
                    "sleep released another engine's planes");
            require(wake_device(0, owner) == owned, "wake did not restore all derived-plane memory");
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            require(read({{output.p, output.bytes}}) == expected, "captured derived-plane addresses or contents changed after wake");
        }
    } else {
        require(owned == 0, "ordinary plane allocations entered sleep mode");
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    const auto free_before = device_free_bytes(0);
    first.reset();
    require(sleep_owned_bytes(owner, 0) == 0, "destroyed engine retained registered planes");
    require(device_free_bytes(0) + (1U << 20) >= free_before + (sleepable ? owned : logical),
            "destroyed engine leaked derived planes or Marlin scratch");
    require(read(b) == second_expected, "destroying one engine invalidated another's planes");
    const void* second_owner = &second->ops;
    if (sleepable) { (void)sleep_device(0, second_owner); }
    second.reset();
    require(sleep_owned_bytes(second_owner, 0) == 0 && sleep_backup_bytes(0) == 0,
            "destroying a sleeping engine leaked regions or host backups");

    struct FailedLoad {
        Context context;
        FailedLoad(const Fixture& fixture, bool sleepable, const void*& owner) {
            owner = &context.ops;
            (void)derive(context, fixture, sleepable);
            throw std::runtime_error("injected model construction failure");
        }
    };
    const void* failed_owner = nullptr;
    const auto before_failure = device_free_bytes(0);
    try { FailedLoad load(fixture, sleepable, failed_owner); }
    catch (const std::runtime_error& error) {
        require(std::string(error.what()) == "injected model construction failure", "failure occurred before plane construction");
    }
    require(sleep_owned_bytes(failed_owner, 0) == 0 && device_free_bytes(0) + (1U << 20) >= before_failure,
            "failed engine construction retained derived buffers");
}

void test_pipeline_devices(int count) {
    auto context = std::make_unique<Context>();
    const void* owner = &context->ops;
    std::vector<std::unique_ptr<Fixture>> fixtures;
    std::vector<std::vector<View>> views;
    for (int device = 0; device < count; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        fixtures.push_back(std::make_unique<Fixture>());
        ops::bind_ops_context(&context->ops);
        require(detail::marlin_plane_bytes() == 0 && detail::w8_derived_plane_bytes() == 0,
                "a pipeline device inherited another device's derived-byte counters");
        views.push_back(derive(*context, *fixtures.back(), true));
        for (const auto& view : views.back()) {
            cudaPointerAttributes attributes{};
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, view.pointer));
            require(attributes.device == device, "derived plane belongs to another pipeline device");
        }
    }
    for (int device = 0; device < count; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        const auto expected = read(views[device]);
        const auto bytes = sleep_owned_bytes(owner, device);
        require(sleep_device(device, owner) == bytes && wake_device(device, owner) == bytes,
                "pipeline derived planes did not sleep and wake on their device");
        require(read(views[device]) == expected, "pipeline derived contents changed through sleep");
    }
    context.reset();
    for (int device = 0; device < count; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        require(sleep_owned_bytes(owner, device) == 0, "pipeline teardown leaked derived regions");
        fixtures[device].reset();
    }
}
} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    try {
        DeviceContext device(0);
        if (detail::w8_device_compute_capability() < 120) { return 77; }
        test_owners(false);
        test_owners(true);
        test_pipeline_devices(std::min(count, 2));
        std::cout << "derived-plane ownership, sleep and teardown: OK\n";
    } catch (const std::exception& error) {
        ops::bind_ops_context(nullptr);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
