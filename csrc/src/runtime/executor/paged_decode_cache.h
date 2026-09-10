// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "utilities/allocator.h"
#include "utilities/utils.h"

namespace dsl {
inline constexpr int DecodePageRows = 128;

// Shared by all requests on one trainer. Pages are homogeneous byte allocations;
// a request owns its logical page table, independently of its row in a batch.
struct DecodePagePool {
    TensorAllocator allocator;
    std::unordered_map<std::size_t, std::vector<Tensor>> free;
    std::int64_t allocated_bytes = 0, used_bytes = 0, reused_pages = 0;

    Tensor acquire(std::size_t bytes) {
        auto& available = free[bytes];
        Tensor result;
        if (available.empty()) {
            result = allocator.allocate(ETensorDType::BYTE,
                                        "decode_page",
                                        EAllocationType::ON_DEVICE,
                                        {static_cast<long>(bytes)});
            allocated_bytes += bytes;
        } else {
            result = available.back();
            available.pop_back();
            ++reused_pages;
        }
        used_bytes += bytes;
        return result;
    }
    void release(Tensor page) {
        used_bytes -= page.bytes();
        free[page.bytes()].push_back(page);
    }
    void trim() {
        for (auto& [bytes, pages] : free)
            for (auto& page : pages) {
                allocator.free(page);
                allocated_bytes -= bytes;
            }
        free.clear();
    }
};

class PagedDecodeBuffer {
public:
    PagedDecodeBuffer(std::shared_ptr<DecodePagePool> pool, ETensorDType dtype, long width, int limit)
        : pool_(std::move(pool)),
          dtype_(dtype),
          width_(width),
          limit_(limit) {
        if (width <= 0 || limit <= 0) throw std::invalid_argument("Invalid decode page geometry");
        addresses_.resize((limit + DecodePageRows - 1) / DecodePageRows);
        table_ = allocator_.allocate(ETensorDType::BYTE,
                                     "decode_page_table",
                                     EAllocationType::ON_DEVICE,
                                     {static_cast<long>(addresses_.size() * sizeof(void*))});
    }
    ~PagedDecodeBuffer() {
        for (auto& page : pages_)
            pool_->release(page);
    }
    PagedDecodeBuffer(const PagedDecodeBuffer&) = delete;
    PagedDecodeBuffer& operator=(const PagedDecodeBuffer&) = delete;

    void reserve(int rows, cudaStream_t stream) {
        if (rows < 0 || rows > limit_) throw std::invalid_argument("Decode page limit exceeded");
        const int needed = (rows + DecodePageRows - 1) / DecodePageRows;
        if (needed <= pages_.size()) return;
        // The host table has stable storage. Fence before modifying entries used
        // by an earlier asynchronous upload, and publish all newly owned pages.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        while (pages_.size() < needed) {
            auto page = pool_->acquire(DecodePageRows * row_bytes());
            addresses_[pages_.size()] = page.Data;
            pages_.push_back(page);
        }
        CUDA_CHECK(
            cudaMemcpyAsync(table_.Data, addresses_.data(), needed * sizeof(void*), cudaMemcpyHostToDevice, stream));
    }
    void append(const void* source, int start, int rows, cudaStream_t stream, std::size_t source_stride = 0) {
        reserve(start + rows, stream);
        const auto stride = source_stride ? source_stride : row_bytes();
        auto* input = static_cast<const std::byte*>(source);
        while (rows > 0) {
            const int count = std::min(rows, DecodePageRows - start % DecodePageRows);
            CUDA_CHECK(cudaMemcpy2DAsync(pages_[start / DecodePageRows].Data + (start % DecodePageRows) * row_bytes(),
                                         row_bytes(),
                                         input,
                                         stride,
                                         row_bytes(),
                                         count,
                                         cudaMemcpyDeviceToDevice,
                                         stream));
            rows -= count;
            start += count;
            input += count * stride;
        }
    }
    [[nodiscard]] const Tensor& table() const {
        return table_;
    }
    [[nodiscard]] std::size_t row_bytes() const {
        return width_ * get_dtype_size(dtype_);
    }
    [[nodiscard]] std::size_t bytes() const {
        return pages_.size() * DecodePageRows * row_bytes();
    }
    [[nodiscard]] std::size_t pages() const {
        return pages_.size();
    }
    [[nodiscard]] bool matches(ETensorDType dtype, long width) const {
        return dtype == dtype_ && width == width_;
    }

private:
    std::shared_ptr<DecodePagePool> pool_;
    TensorAllocator allocator_;
    ETensorDType dtype_;
    long width_;
    int limit_;
    Tensor table_;
    std::vector<Tensor> pages_;
    std::vector<void*> addresses_;
};

inline Tensor decode_batch_row(Tensor tensor, int row) {
    if (tensor.Rank < 1 || row < 0 || row >= tensor.Sizes[0])
        throw std::invalid_argument("Decode row is outside tensor batch");
    tensor.Data += row * tensor.bytes() / tensor.Sizes[0];
    tensor.Sizes[0] = 1;
    return tensor;
}
}  // namespace dsl
