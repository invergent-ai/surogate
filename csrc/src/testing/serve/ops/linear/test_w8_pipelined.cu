// The pipelined W8G32 kernel (w8_rowsplit_gemm_pipelined.cuh) against the medium-T split-K kernel
// whose arithmetic it keeps: for every row tiling, row grouping and pipeline depth, at every width
// the batch-consistent route can give it, the two must write the same bits. With `bench` it also
// times them all, from CUDA graphs, on a vocabulary head and the usual linear shapes, beside a
// plain streaming read of the same bytes (what HBM gives).
//
//   sinfer_w8_pipelined_test            correctness only (a ctest)
//   sinfer_w8_pipelined_test bench [n k ...]

#include "ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_pipelined.cuh"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;
using namespace sinfer::ops::detail;

namespace {

struct Problem {
    int n, k;
    DeviceBuffer codes, scales;
};

Problem make_problem(int n, int k, std::uint32_t seed) {
    Problem p{n, k, DeviceBuffer(static_cast<std::size_t>(n) * k),
              DeviceBuffer(static_cast<std::size_t>(n) * (k / 32) * 2)};
    std::mt19937 rng(seed);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(n) * k);
    for (std::size_t i = 0; i < codes.size(); i += 4) {
        const std::uint32_t r = rng();
        std::memcpy(&codes[i], &r, std::min<std::size_t>(4, codes.size() - i));
    }
    std::uniform_real_distribution<float> scale(0.0005F, 0.02F);
    std::vector<__half> scales(static_cast<std::size_t>(n) * (k / 32));
    for (auto& s : scales) { s = __float2half(scale(rng)); }
    p.codes.copy_from_host(codes.data(), codes.size());
    p.scales.copy_from_host(scales.data(), scales.size() * 2);
    return p;
}

int sms() {
    int device = 0, count = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device), "SM count");
    return count;
}

// The medium-T route as launch_w8_consistent ran it before the pipelined kernel.
void launch_reference(const Problem& p, const __nv_bfloat16* x, __nv_bfloat16* out, int count,
                      bool row_pairs, cudaStream_t stream) {
    const W8ContiguousOutput output{out, p.n};
    const auto* codes  = static_cast<const std::uint8_t*>(p.codes.p);
    const auto* scales = static_cast<const std::uint8_t*>(p.scales.p);
    const auto launch  = [&]<int Columns>() {
        const unsigned tiles = static_cast<unsigned>((count + Columns - 1) / Columns);
        if (row_pairs) {
            w8_rowsplit_medium_t_splitk_kernel<0, Columns, 4, 1, 2, W8ContiguousOutput, false, 2>
                <<<dim3(p.n / 32, tiles), 128, 0, stream>>>(x, codes, scales, output, count, p.k);
        } else {
            w8_rowsplit_medium_t_splitk_kernel<0, Columns, 4, 1, 2>
                <<<dim3(p.n / 16, tiles), 128, 0, stream>>>(x, codes, scales, output, count, p.k);
        }
    };
    if (count <= 8) { launch.template operator()<8>(); }
    else if (count <= 16) { launch.template operator()<16>(); }
    else if (count <= 32) { launch.template operator()<32>(); }
    else { launch.template operator()<64>(); }
}

int smem_optin() {
    int device = 0, bytes = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
               "shared memory limit");
    return bytes;
}

// One shape of the pipelined kernel: rows a warp (16 x row_tiles), sets of four warps a CTA, and
// K groups in flight (0: the route's own count for the column bucket).
struct Variant {
    int row_tiles, row_groups, stages;
    std::string name() const {
        return "r" + std::to_string(16 * row_tiles * row_groups) + "/g" + std::to_string(row_groups) +
               "/s" + (stages == 0 ? std::string("route") : std::to_string(stages));
    }
};
const Variant kVariants[] = {{1, 1, 0}, {2, 1, 0}, {2, 2, 2}, {2, 2, 3}, {2, 2, 4}, {1, 2, 4}};

template <int Columns>
constexpr int route_stages() { return Columns <= 16 ? 4 : Columns <= 32 ? 3 : 2; }

// False when the variant does not tile the rows or does not fit this device's shared memory.
template <int Columns, int Stages, int RowTiles, int RowGroups>
bool launch_pipelined_one(const Problem& p, const __nv_bfloat16* x, __nv_bfloat16* out, int count,
                          cudaStream_t stream) {
    using Layout = W8PipelinedLayout<Columns, Stages, RowTiles, RowGroups>;
    if (p.n % Layout::kRowsPerCta != 0 || Layout::kBytes > static_cast<std::size_t>(smem_optin())) {
        return false;
    }
    auto* kernel = w8_rowsplit_pipelined_kernel<Columns, Stages, RowTiles, RowGroups>;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(Layout::kBytes)),
               "dynamic shared memory");
    kernel<<<dim3(p.n / Layout::kRowsPerCta, (count + Columns - 1) / Columns), Layout::kThreads,
             Layout::kBytes, stream>>>(x, static_cast<const std::uint8_t*>(p.codes.p),
                                       static_cast<const std::uint8_t*>(p.scales.p),
                                       W8ContiguousOutput{out, p.n}, count, p.k);
    return true;
}

template <int Columns>
bool launch_bucket(const Variant& v, const Problem& p, const __nv_bfloat16* x, __nv_bfloat16* out,
                   int count, cudaStream_t stream) {
    constexpr int S = route_stages<Columns>();
    switch (v.row_tiles * 100 + v.row_groups * 10 + v.stages) {
    case 110: return launch_pipelined_one<Columns, S, 1, 1>(p, x, out, count, stream);
    case 210: return launch_pipelined_one<Columns, S, 2, 1>(p, x, out, count, stream);
    case 222: return launch_pipelined_one<Columns, 2, 2, 2>(p, x, out, count, stream);
    case 223: return launch_pipelined_one<Columns, 3, 2, 2>(p, x, out, count, stream);
    case 224: return launch_pipelined_one<Columns, 4, 2, 2>(p, x, out, count, stream);
    case 124: return launch_pipelined_one<Columns, 4, 1, 2>(p, x, out, count, stream);
    default: return false;
    }
}

// The variant at the column bucket launch_w8_consistent gives `count` columns.
bool launch_variant(const Variant& v, const Problem& p, const __nv_bfloat16* x, __nv_bfloat16* out,
                    int count, cudaStream_t stream) {
    if (count <= 8) { return launch_bucket<8>(v, p, x, out, count, stream); }
    if (count <= 16) { return launch_bucket<16>(v, p, x, out, count, stream); }
    if (count <= 32) { return launch_bucket<32>(v, p, x, out, count, stream); }
    return launch_bucket<64>(v, p, x, out, count, stream);
}

int check(const Problem& p, int count, cudaStream_t stream) {
    std::vector<float> xs(static_cast<std::size_t>(p.k) * count);
    fill_uniform(xs, 31 + count, -3.0F, 3.0F);
    DeviceBuffer x = to_device_bf16(xs);
    const std::size_t elements = static_cast<std::size_t>(p.n) * count;
    DeviceBuffer want(elements * 2), got(elements * 2);
    // fill() is a cudaMemset on the legacy stream, which a non-blocking stream does not wait for.
    want.fill(0x7f);
    cuda_synchronize();
    const auto* xp = static_cast<const __nv_bfloat16*>(x.p);
    launch_reference(p, xp, static_cast<__nv_bfloat16*>(want.p), count, false, stream);
    cuda_check(cudaGetLastError(), "reference launch");
    cuda_synchronize(stream);
    const auto a = from_device<std::uint16_t>(want, elements);
    const std::string label = "n=" + std::to_string(p.n) + " k=" + std::to_string(p.k) + " T=" +
                              std::to_string(count);
    int failures = 0, ran = 0;
    for (const Variant& v : kVariants) {
        got.fill(0x7f);
        cuda_synchronize();
        if (!launch_variant(v, p, xp, static_cast<__nv_bfloat16*>(got.p), count, stream)) { continue; }
        cuda_check(cudaGetLastError(), "launch");
        cuda_synchronize(stream);
        ++ran;
        const auto b = from_device<std::uint16_t>(got, elements);
        std::size_t differ = 0;
        for (std::size_t i = 0; i < elements; ++i) { differ += a[i] != b[i]; }
        if (differ != 0) {
            std::cerr << "FAIL " << label << ' ' << v.name() << ": " << differ << " of " << elements
                      << " outputs differ\n";
            ++failures;
        }
    }
    if (failures == 0) { std::cout << "PASS " << label << " (" << ran << " variants)\n"; }
    return failures;
}

__global__ void stream_read_kernel(const uint4* data, std::size_t vectors, unsigned* sink) {
    unsigned acc = 0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < vectors;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint4 v = data[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) { sink[0] = acc; }
}

template <class Launch>
double graph_us(Launch launch, cudaStream_t stream, int iters) {
    for (int i = 0; i < 3; ++i) { launch(); }
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), "capture");
    for (int i = 0; i < iters; ++i) { launch(); }
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    cuda_check(cudaGraphLaunch(exec, stream), "warm graph");
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "event");
    cuda_check(cudaEventCreate(&stop), "event");
    cuda_check(cudaEventRecord(start, stream), "record");
    cuda_check(cudaGraphLaunch(exec, stream), "graph");
    cuda_check(cudaEventRecord(stop, stream), "record");
    cuda_check(cudaEventSynchronize(stop), "sync");
    float ms = 0;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cuda_check(cudaGraphExecDestroy(exec), "destroy");
    cuda_check(cudaGraphDestroy(graph), "destroy");
    return ms * 1000.0 / iters;
}

void bench(const Problem& p, cudaStream_t stream) {
    const bool row_pairs = p.n % 32 == 0 && p.n / 32 >= 2 * sms();
    const double bytes   = static_cast<double>(p.n) * p.k * (1.0 + 2.0 / 32);
    DeviceBuffer sink(16);
    const double read_us = graph_us(
        [&] {
            stream_read_kernel<<<sms() * 8, 512, 0, stream>>>(static_cast<const uint4*>(p.codes.p),
                                                              p.codes.bytes / 16,
                                                              static_cast<unsigned*>(sink.p));
        },
        stream, 20);
    const double code_bytes = static_cast<double>(p.codes.bytes);
    std::printf("n=%6d k=%6d  streaming read of the codes %7.1f us (%.2f TB/s)\n", p.n, p.k, read_us,
                code_bytes / read_us * 1e-6);
    for (const int count : {1, 2, 4, 8, 16, 32, 64}) {
        std::vector<float> xs(static_cast<std::size_t>(p.k) * count);
        fill_uniform(xs, 7 + count, -3.0F, 3.0F);
        DeviceBuffer x = to_device_bf16(xs);
        DeviceBuffer out(static_cast<std::size_t>(p.n) * count * 2);
        const auto* xp = static_cast<const __nv_bfloat16*>(x.p);
        auto* op       = static_cast<__nv_bfloat16*>(out.p);
        const int iters = p.n * static_cast<double>(p.k) > 1e8 ? 20 : 200;
        const double ref = graph_us([&] { launch_reference(p, xp, op, count, row_pairs, stream); }, stream, iters);
        std::printf("n=%6d k=%6d T=%3d  %-14s %8.1f us (%.2f TB/s)\n", p.n, p.k, count,
                    row_pairs ? "medium r32" : "medium r16", ref, bytes / ref * 1e-6);
        for (const Variant& v : kVariants) {
            if (!launch_variant(v, p, xp, op, count, stream)) { continue; }
            const double us = graph_us([&] { launch_variant(v, p, xp, op, count, stream); }, stream, iters);
            std::printf("n=%6d k=%6d T=%3d  %-14s %8.1f us (%.2f TB/s)  ratio %.3f\n", p.n, p.k, count,
                        v.name().c_str(), us, bytes / us * 1e-6, us / ref);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    if (argc > 1 && std::string(argv[1]) == "bench") {
        std::vector<std::pair<int, int>> shapes;
        for (int i = 2; i + 1 < argc; i += 2) { shapes.emplace_back(std::atoi(argv[i]), std::atoi(argv[i + 1])); }
        if (shapes.empty()) { shapes = {{151936, 4096}, {248320, 5120}, {12288, 4096}, {4096, 12288}}; }
        for (const auto& [n, k] : shapes) { bench(make_problem(n, k, 3), stream); }
        return 0;
    }
    int failures = 0;
    // A head-like shape that every variant tiles; narrower ones, K one group short of the deepest
    // pipeline, and rows only 16-row CTAs tile.
    const Problem wide    = make_problem(128 * (sms() + 3), 4096, 11);
    const Problem narrow  = make_problem(1024, 768, 13);
    const Problem shallow = make_problem(2048, 512, 17);
    const Problem odd     = make_problem(16 * 65, 1024, 19);
    for (const int count : {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 129}) {
        failures += check(wide, count, stream);
        failures += check(narrow, count, stream);
        failures += check(shallow, count, stream);
        failures += check(odd, count, stream);
    }
    if (failures != 0) {
        std::cerr << failures << " pipelined W8 case(s) differ from the medium-T kernel\n";
        return 1;
    }
    std::cout << "pipelined W8: bit-identical to the medium-T kernel\n";
    return 0;
}
