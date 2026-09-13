#include "api/ops/gqa_workspace.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <array>
#include <set>

using namespace sinfer;
using namespace sinfer::test;

namespace {
constexpr int kContext = 1048576;
constexpr int kPages = kContext / kPagedKVPageSize;
constexpr int kPhysicalPages = 257;

void copy(const Tensor& tensor, const void* values) {
    CUDA_CHECK(cudaMemcpy(tensor.data, values, tensor.bytes(), cudaMemcpyHostToDevice));
}

int verify(int dim, int heads, int kv_heads, DType dtype, int batch, int width, bool sparse) {
    DeviceArena arena(96U << 20);
    Tensor q = arena.alloc(DType::BF16, {dim, heads, width, batch});
    Tensor positions = arena.alloc(DType::I32, {width, batch});
    Tensor rows = arena.alloc(DType::I32, {batch});
    Tensor out = arena.alloc(DType::BF16, {dim, heads, width, batch});
    CUDA_CHECK(cudaMemset(q.data, 0, q.bytes()));
    PagedKVBatchLayerView cache;
    cache.head_dim = dim;
    cache.num_kv_heads = kv_heads;
    cache.dtype = dtype;
    cache.k_pages = arena.alloc(dtype, {dim, kPagedKVPageSize, kv_heads, kPhysicalPages});
    cache.v_pages = arena.alloc(dtype, {dim, kPagedKVPageSize, kv_heads, kPhysicalPages});
    cache.block_tables = arena.alloc(DType::I32, {kPages, batch});
    CUDA_CHECK(cudaMemset(cache.k_pages.data, 0, cache.k_pages.bytes()));
    if (dtype == DType::I8) {
        cache.quant_group = 64;
        cache.k_scale_pages = arena.alloc(DType::FP16, {dim / 64, kPagedKVPageSize, kv_heads, kPhysicalPages});
        cache.v_scale_pages = arena.alloc(DType::FP16, {dim / 64, kPagedKVPageSize, kv_heads, kPhysicalPages});
        const std::vector<std::uint16_t> scales(cache.k_scale_pages.numel(), 0x3c00);
        copy(cache.k_scale_pages, scales.data());
        copy(cache.v_scale_pages, scales.data());
    }
    const auto value = [](int physical, int head) { return 1 << ((physical + head) % 4); };
    std::vector<std::uint16_t> bf16(cache.v_pages.numel());
    std::vector<std::uint8_t> codes(cache.v_pages.numel());
    for (int page = 0; page < kPhysicalPages; ++page) {
        for (int head = 0; head < kv_heads; ++head) {
            const auto begin = (page * kv_heads + head) * kPagedKVPageSize * dim;
            const int exponent = (page + head) % 4;
            std::fill_n(bf16.begin() + begin, kPagedKVPageSize * dim, f32_to_bf16(value(page, head)));
            std::fill_n(codes.begin() + begin, kPagedKVPageSize * dim,
                        dtype == DType::I8 ? value(page, head) : 0x38 + 8 * exponent);
        }
    }
    copy(cache.v_pages, dtype == DType::BF16 ? static_cast<void*>(bf16.data()) : codes.data());
    // Repeated read-only pages represent a long history without allocating GBs
    // of identical values. Different table rows and fragmented IDs expose address errors.
    std::vector<int> tables(kPages * batch), row_ids(batch);
    for (int row = 0; row < batch; ++row) {
        row_ids[row] = batch - 1 - row;
        for (int page = 0; page < kPages; ++page) {
            tables[row * kPages + page] = (page * 17 + row * 29) % kPhysicalPages;
        }
    }
    copy(cache.block_tables, tables.data());
    copy(rows, row_ids.data());
    constexpr int mask_stride = kContext / 4 / 32;
    Tensor mask = arena.alloc(DType::I32, {mask_stride, width, batch});
    ops::GqaBlockMask selection;
    if (sparse) { selection = {static_cast<std::uint32_t*>(mask.data), mask_stride, 4}; }
    const ops::GqaExecutionEnvelope envelope{1, kContext};
    WorkspaceArena scratch(std::max<std::size_t>(256,
        ops::gqa_attention_history_workspace_capacity_bytes(dim, heads, kv_heads, dtype,
                                                            envelope, batch, width, width)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    const auto run = [&] {
        ops::gqa_attention_cached(q, positions, Tensor{}, rows, 1.0F / std::sqrt(float(dim)),
                                  cache, envelope, scratch, out, stream, selection);
    };
    int failures = 0;
    for (int context : {262144, 262145, kContext}) {
        std::vector<int> pos(batch * width);
        std::vector<std::uint32_t> bits(mask.numel(), 0);
        std::vector<double> expected(batch * width * kv_heads);
        for (int row = 0; row < batch; ++row) {
            for (int col = 0; col < width; ++col) {
                const int lane = row * width + col;
                const int last = context - width - row * 19 + col;
                pos[lane] = last;
                const std::set<int> selected{1, std::min(65537, last / 4), last / 4};
                for (int block : selected) { bits[lane * mask_stride + block / 32] |= 1U << (block % 32); }
                for (int head = 0; head < kv_heads; ++head) {
                    double sum = 0;
                    int count = 0;
                    if (sparse) {
                        for (int block : selected) {
                            const int n = std::min(4, last + 1 - block * 4);
                            sum += n * value(tables[row_ids[row] * kPages + block * 4 / 64], head);
                            count += n;
                        }
                    } else {
                        for (int page = 0; page <= last / 64; ++page) {
                            const int n = std::min(64, last + 1 - page * 64);
                            sum += n * value(tables[row_ids[row] * kPages + page], head);
                            count += n;
                        }
                    }
                    expected[lane * kv_heads + head] = sum / count;
                }
            }
        }
        copy(positions, pos.data());
        copy(mask, bits.data());
        const auto check = [&] {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto actual = from_device_bf16(out.data, out.numel());
            for (int lane = 0; lane < batch * width; ++lane) {
                for (int head = 0; head < heads; ++head) {
                    const double want = expected[lane * kv_heads + head / (heads / kv_heads)];
                    for (int d = 0; d < dim; ++d) {
                        const auto got = actual[(lane * heads + head) * dim + d];
                        if (!std::isfinite(got) || std::abs(got - want) > 0.02) {
                            std::cerr << "long attention mismatch: context=" << context << " dtype=" << int(dtype)
                                      << " B=" << batch << " W=" << width << " sparse=" << sparse
                                      << " got=" << got << " expected=" << want << '\n';
                            ++failures;
                            return;
                        }
                    }
                }
            }
        };
        run();
        check();
        if (!graph) {
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            run();
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        }
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        check();
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return failures;
}
}

int main() {
    if (cuda_unavailable()) { return 77; }
    try {
        int failures = 0;
        for (auto dtype : {DType::BF16, DType::FP8_E4M3FN}) {
            failures += verify(512, 64, 1, dtype, 1, 1, false);
            failures += verify(512, 64, 1, dtype, 2, 4, true);
            failures += verify(512, 64, 1, dtype, 1, 17, true);
        }
        failures += verify(256, 24, 4, DType::I8, 1, 1, false);
        failures += verify(256, 24, 4, DType::I8, 2, 6, false);
        std::cout << "long-context attention: " << (failures ? "FAIL" : "PASS") << '\n';
        return failures != 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
