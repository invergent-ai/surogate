#include "api/ops/gqa_attention.h"
#include "core/arena.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <iostream>
#include <numeric>

using namespace sinfer;
using namespace sinfer::test;

int main() {
    if (cuda_unavailable()) { return 77; }
    constexpr int dim = 256, heads = 16, kv_heads = 2, pages = 5, mask_stride = 3;
    int failures = 0;
    for (const auto dtype : {DType::BF16, DType::FP8_E4M3FN}) {
      for (const int batch : {1, 2, 4}) {
        for (const int width : {1, 4, 6, 16}) {
            DeviceArena arena(16U << 20);
            Tensor q = arena.alloc(DType::BF16, {dim, heads, width, batch});
            Tensor k = arena.alloc(DType::BF16, {dim, kv_heads, width, batch});
            Tensor v = arena.alloc(DType::BF16, {dim, kv_heads, width, batch});
            Tensor positions = arena.alloc(DType::I32, {width, batch});
            Tensor rows = arena.alloc(DType::I32, {batch});
            Tensor out = arena.alloc(DType::BF16, {dim, heads, width, batch});
            CUDA_CHECK(cudaMemset(q.data, 0, q.bytes()));
            CUDA_CHECK(cudaMemset(k.data, 0, k.bytes()));
            CUDA_CHECK(cudaMemset(v.data, 0, v.bytes()));
            PagedKVBatchLayerView cache;
            cache.head_dim = dim;
            cache.num_kv_heads = kv_heads;
            cache.dtype = dtype;
            cache.k_pages = arena.alloc(cache.dtype, {dim, kPagedKVPageSize, kv_heads, pages * batch});
            cache.v_pages = arena.alloc(cache.dtype, {dim, kPagedKVPageSize, kv_heads, pages * batch});
            cache.block_tables = arena.alloc(DType::I32, {pages, batch});
            CUDA_CHECK(cudaMemset(cache.k_pages.data, 0, cache.k_pages.bytes()));
            std::vector<int> table(pages * batch), row_ids(batch), pos(width * batch);
            std::vector<std::uint16_t> values(cache.v_pages.numel());
            std::vector<std::uint8_t> fp8_values(cache.v_pages.numel());
            std::vector<std::uint32_t> masks(batch * width * mask_stride, 0);
            for (int row = 0; row < batch; ++row) {
                row_ids[row] = batch - row - 1;
                for (int page = 0; page < pages; ++page) {
                    const int physical = row * pages + (page + 2) % pages;
                    table[row * pages + page] = physical;
                    for (int h = 0; h < kv_heads; ++h) {
                        for (int t = 0; t < kPagedKVPageSize; ++t) {
                            const int exponent = (row + (page * 64 + t) / 4) % 7;
                            const auto value = f32_to_bf16(float(1 << exponent));
                            const auto begin = ((physical * kv_heads + h) * kPagedKVPageSize + t) * dim;
                            std::fill_n(values.begin() + begin, dim, value);
                            std::fill_n(fp8_values.begin() + begin, dim, std::uint8_t(0x38 + exponent * 8));
                        }
                    }
                }
                for (int col = 0; col < width; ++col) {
                    pos[row * width + col] = 256 + col;
                    // Each query has a distinct selection, always in its history.
                    const int block = (row * 11 + col * 3) % 60;
                    masks[(row * width + col) * mask_stride + block / 32] = 1U << (block % 32);
                }
            }
            // An exact allocation also makes a mask-row overrun visible to memcheck.
            DeviceBuffer mask(masks.size() * sizeof(std::uint32_t));
            CUDA_CHECK(cudaMemcpy(mask.p, masks.data(), mask.bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(cache.v_pages.data,
                                   dtype == DType::BF16 ? static_cast<const void*>(values.data()) : fp8_values.data(),
                                   cache.v_pages.bytes(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(cache.block_tables.data, table.data(), cache.block_tables.bytes(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(rows.data, row_ids.data(), rows.bytes(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(positions.data, pos.data(), positions.bytes(), cudaMemcpyHostToDevice));
            const ops::GqaExecutionEnvelope envelope{1, pages * kPagedKVPageSize};
            WorkspaceArena scratch(std::max<std::size_t>(256, ops::gqa_attention_workspace_capacity_bytes(
                dim, heads, kv_heads, cache.dtype, envelope, batch, width, width)));
            const ops::GqaBlockMask selection{static_cast<const std::uint32_t*>(mask.p), mask_stride, 4};
            for (const bool cached : {false, true}) {
                if (cached) {
                    ops::gqa_attention_cached(q, positions, Tensor{}, rows, 1.0F / 16.0F,
                                               cache, envelope, scratch, out, nullptr, selection);
                } else {
                    ops::gqa_attention(q, k, v, positions, Tensor{}, rows, 1.0F / 16.0F,
                                       cache, envelope, scratch, out, nullptr, selection);
                }
                CUDA_CHECK(cudaDeviceSynchronize());
                const auto actual = from_device<std::uint16_t>(out.data, out.numel());
                int mismatches = 0;
                for (int row = 0; row < batch; ++row) {
                    for (int col = 0; col < width; ++col) {
                        const int block = (row * 11 + col * 3) % 60;
                        const auto expected = f32_to_bf16(float(1 << ((row_ids[row] + block) % 7)));
                        for (int i = 0; i < heads * dim; ++i) {
                            mismatches += actual[(row * width + col) * heads * dim + i] != expected;
                        }
                    }
                }
                if (mismatches) {
                    ++failures;
                    std::cerr << "sparse attention dtype=" << int(dtype) << " B=" << batch << " W=" << width
                              << " cached=" << cached << ": " << mismatches << " mismatches\n";
                }
            }
        }
      }
    }
    return failures != 0;
}
