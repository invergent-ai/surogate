#include "family/impl/runtime/attention_workspace.h"
#include "core/arena.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <array>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace sinfer;
using namespace sinfer::test;

int main() {
    if (cuda_unavailable()) { return 77; }
    try {
        family::TextGeometry geometry;
        geometry.head_dim = 256;
        geometry.query_heads = 16;
        geometry.kv_heads = 8;
        geometry.global_head_dim = 512;
        geometry.global_kv_heads = 1;
        constexpr int history = 32768, pages = history / kPagedKVPageSize;
        const ops::GqaExecutionEnvelope envelope{1, history};
        for (const auto dtype : {DType::BF16, DType::FP8_E4M3FN}) {
            for (const auto [batch, width] : {std::pair{128, 1}, {8, 6}, {1, 256}}) {
                const auto capacity = family::detail::attention_workspace_capacity_bytes(
                    geometry, dtype, envelope, batch, 1, width);
                DeviceArena arena(128ULL << 20);
                const int dim = geometry.global_head_dim, heads = geometry.query_heads;
                Tensor query = arena.alloc(DType::BF16, {dim, heads, width, batch});
                Tensor output = arena.alloc(DType::BF16, {dim, heads, width, batch});
                Tensor positions = arena.alloc(DType::I32, {width, batch});
                Tensor rows = arena.alloc(DType::I32, {batch});
                PagedKVBatchLayerView cache;
                cache.head_dim = dim;
                cache.num_kv_heads = 1;
                cache.dtype = dtype;
                cache.k_pages = arena.alloc(dtype, {dim, kPagedKVPageSize, 1, pages});
                cache.v_pages = arena.alloc(dtype, {dim, kPagedKVPageSize, 1, pages});
                cache.block_tables = arena.alloc(DType::I32, {pages, batch});
                std::vector<int> row_ids(batch), table(pages * batch), pos(width * batch);
                std::iota(row_ids.begin(), row_ids.end(), 0);
                for (int row = 0; row < batch; ++row) {
                    std::iota(table.begin() + row * pages, table.begin() + (row + 1) * pages, 0);
                    for (int col = 0; col < width; ++col) { pos[row * width + col] = history - width + col; }
                }
                CUDA_CHECK(cudaMemset(query.data, 0, query.bytes()));
                CUDA_CHECK(cudaMemset(cache.k_pages.data, 0, cache.k_pages.bytes()));
                if (dtype == DType::BF16) {
                    const std::vector<std::uint16_t> ones(cache.v_pages.numel(), f32_to_bf16(1.0F));
                    CUDA_CHECK(cudaMemcpy(cache.v_pages.data, ones.data(), cache.v_pages.bytes(), cudaMemcpyHostToDevice));
                } else {
                    CUDA_CHECK(cudaMemset(cache.v_pages.data, 0x38, cache.v_pages.bytes()));
                }
                CUDA_CHECK(cudaMemcpy(rows.data, row_ids.data(), rows.bytes(), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(positions.data, pos.data(), positions.bytes(), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cache.block_tables.data, table.data(), cache.block_tables.bytes(), cudaMemcpyHostToDevice));
                WorkspaceArena workspace(capacity);
                ops::gqa_attention_cached(query, positions, {}, rows, 1.0F / std::sqrt(float(dim)),
                                           cache, envelope, workspace, output, nullptr);
                CUDA_CHECK(cudaDeviceSynchronize());
                const auto actual = from_device<std::uint16_t>(output.data, output.numel());
                for (auto value : actual) {
                    if (std::abs(bf16_to_f32(value) - 1.0F) > 0.01F) {
                        throw std::runtime_error("global attention result differs from constant-value reference");
                    }
                }
                std::cout << "dtype=" << int(dtype) << " batch=" << batch << " width=" << width
                          << " reserved=" << capacity << " OK\n";
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
