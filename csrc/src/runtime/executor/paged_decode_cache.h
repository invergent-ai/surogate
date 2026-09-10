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

struct DecodeCapacityError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Shared by all requests on one trainer. Pages are homogeneous byte allocations;
// a request owns its logical page table, independently of its row in a batch.
struct DecodePagePool {
    TensorAllocator allocator;
    std::unordered_map<std::size_t, std::vector<Tensor>> free;
    std::int64_t allocated_bytes = 0, used_bytes = 0, reused_pages = 0;
    std::int64_t auxiliary_bytes = 0, limit_bytes = 0;
    std::int64_t workspace_bytes = 0, memory_limit_bytes = 0;
    std::int64_t execution_headroom_bytes = 0;

    std::int64_t memory_bytes() const {
        return allocated_bytes + auxiliary_bytes + workspace_bytes + execution_headroom_bytes;
    }
    void set_memory_limit(std::int64_t bytes) {
        if (bytes < 0 || (bytes && bytes < used_bytes + auxiliary_bytes + workspace_bytes + execution_headroom_bytes))
            throw std::invalid_argument("Decode memory budget is below live cache and workspace storage");
        trim();
        memory_limit_bytes = bytes;
    }
    void make_memory_room(std::size_t bytes) {
        if (memory_limit_bytes && memory_bytes() + bytes > memory_limit_bytes) trim();
        if (memory_limit_bytes && memory_bytes() + bytes > memory_limit_bytes)
            throw DecodeCapacityError("Decode cache and workspace VRAM budget exhausted");
        std::size_t available = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&available, &total));
        if (available < bytes + execution_headroom_bytes) {
            trim();
            CUDA_CHECK(cudaMemGetInfo(&available, &total));
            if (available < bytes + execution_headroom_bytes)
                throw DecodeCapacityError("Insufficient free VRAM for decode execution workspace");
        }
    }
    void charge_workspace(std::size_t bytes) {
        make_memory_room(bytes);
        workspace_bytes += bytes;
    }

    void make_room(std::size_t bytes) {
        if (limit_bytes && allocated_bytes + auxiliary_bytes + bytes > limit_bytes) trim();
        if (limit_bytes && allocated_bytes + auxiliary_bytes + bytes > limit_bytes)
            throw DecodeCapacityError("Decode cache VRAM budget exhausted");
        make_memory_room(bytes);
    }
    void charge_auxiliary(std::size_t bytes) {
        make_room(bytes);
        auxiliary_bytes += bytes;
    }
    void set_limit(std::int64_t bytes) {
        if (bytes < 0 || (bytes && bytes < used_bytes + auxiliary_bytes))
            throw std::invalid_argument("Decode cache budget is below live cache storage");
        trim();
        limit_bytes = bytes;
    }

    Tensor acquire(std::size_t bytes) {
        Tensor result;
        auto it = free.find(bytes);
        if (it == free.end() || it->second.empty()) {
            make_room(bytes);
            result = allocator.allocate(ETensorDType::BYTE,
                                        "decode_page",
                                        EAllocationType::ON_DEVICE,
                                        {static_cast<long>(bytes)});
            allocated_bytes += bytes;
        } else {
            auto& available = it->second;
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

// Allocation accounting shared by metadata and the reusable sampler. Failed
// allocations return their reservation before admission considers another row.
struct DecodeWorkspaceAllocator {
    TensorAllocator allocator;
    std::shared_ptr<DecodePagePool> pool;
    ~DecodeWorkspaceAllocator() {
        if (pool) pool->workspace_bytes -= allocator.total_allocation();
    }
    Tensor allocate(ETensorDType dtype, const char* name, EAllocationType kind, const std::vector<long>& shape) {
        long bytes = get_dtype_size(dtype);
        for (long d : shape)
            bytes *= d;
        if (pool) pool->charge_workspace(bytes);
        try {
            return allocator.allocate(dtype, name, kind, shape);
        } catch (...) {
            if (pool) pool->workspace_bytes -= bytes;
            throw;
        }
    }
    void free(Tensor& tensor) {
        if (!tensor.Data) return;
        const auto bytes = tensor.bytes();
        allocator.free(tensor);
        if (pool) pool->workspace_bytes -= bytes;
    }
    std::size_t total_allocation() const {
        return allocator.total_allocation();
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
        const auto bytes = addresses_.size() * sizeof(void*);
        pool_->charge_auxiliary(bytes);
        try {
            table_ = allocator_.allocate(ETensorDType::BYTE,
                                         "decode_page_table",
                                         EAllocationType::ON_DEVICE,
                                         {static_cast<long>(bytes)});
        } catch (...) {
            pool_->auxiliary_bytes -= bytes;
            throw;
        }
    }
    ~PagedDecodeBuffer() {
        for (auto& page : pages_)
            pool_->release(page);
        pool_->auxiliary_bytes -= table_.bytes();
    }
    PagedDecodeBuffer(const PagedDecodeBuffer&) = delete;
    PagedDecodeBuffer& operator=(const PagedDecodeBuffer&) = delete;

    void reserve(int rows, cudaStream_t stream) {
        if (rows < 0 || rows > limit_) throw std::invalid_argument("Decode page limit exceeded");
        const int needed = (rows + DecodePageRows - 1) / DecodePageRows;
        if (needed <= pages_.size()) return;
        // Existing entries never change, and earlier uploads cover only their
        // old prefix. Appending new entries therefore needs no stream fence.
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
    // Admission rollback only: callers fence the stream before returning pages.
    void truncate_pages(std::size_t count) {
        while (pages_.size() > count) {
            pool_->release(pages_.back());
            pages_.pop_back();
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
    [[nodiscard]] std::size_t table_bytes() const {
        return table_.bytes();
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
