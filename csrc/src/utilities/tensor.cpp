// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor.h"

#include <iostream>
#include <vector>

#include <cuda_fp8.h>
#include <cuda_runtime.h>

/**
 * @brief Create a contiguous view into @p src by slicing the first dimension.
 *
 * Only dimension 0 is supported because slices must remain contiguous with the
 * current Tensor storage/layout assumptions.
 *
 * @param src  Source tensor to slice (view semantics; no copy).
 * @param dim  Dimension to slice; must be 0.
 * @param start  Inclusive start index along @p dim (in elements).
 * @param end    Exclusive end index along @p dim (in elements).
 * @return Tensor view that shares storage with @p src and has Sizes[dim] = end-start.
 *
 * @throws std::logic_error if @p dim != 0 or if indices are out of bounds.
 */
Tensor slice(const Tensor& src, int dim, long start, long end) {
    if (dim != 0)
        throw std::logic_error("Slices must be contiguous, so only the first dimension can be sliced.");

    if (start >= src.Sizes[dim] || end > src.Sizes[dim])
        throw std::logic_error("Slice out of bounds.");

    std::array<long, MAX_TENSOR_DIM> strides{};

    for (int i = src.Rank; i < MAX_TENSOR_DIM; ++i)
        strides[i] = 0;

    strides[src.Rank - 1] = 1;
    for (int i = src.Rank - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * src.Sizes[i + 1];

    Tensor dst = src;
    dst.Sizes[dim] = end - start;
    std::ptrdiff_t offset = start * strides[dim] * get_dtype_size(src.DType);
    dst.Data = src.Data + offset;
    return dst;
}

/**
 * @brief Asynchronously fill a tensor's buffer with zeros.
 *
 * For device memory, uses cudaMemsetAsync with the provided stream.
 * For host/pinned memory (Device == -1), uses memset since cudaMemsetAsync
 * doesn't work on host memory.
 *
 * @param dst     Tensor whose underlying memory will be set to 0.
 * @param stream  CUDA stream used for the async memset (device memory only).
 */
void fill_zero(Tensor& dst, cudaStream_t stream) {
    if (!dst.Data || dst.bytes() == 0) return;

    if (dst.Device == -1) {
        cudaPointerAttributes attr{};
        cudaError_t attr_err = cudaPointerGetAttributes(&attr, dst.Data);
        if (attr_err == cudaSuccess) {
            if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
                CUDA_CHECK(cudaMemsetAsync(dst.Data, 0, dst.bytes(), stream));
                return;
            }
        } else {
            cudaGetLastError(); // clear sticky error
        }
        // Host/pinned memory - use memset
        std::memset(dst.Data, 0, dst.bytes());
        return;
    }

    // Device memory - use async memset
    CUDA_CHECK(cudaMemsetAsync(dst.Data, 0, dst.bytes(), stream));
}

/**
 * @brief Read a single element from device memory into host memory.
 *
 * @tparam TargetType  Host type to read/interpret the element as.
 * @param index        Element index (in elements, not bytes) into the tensor buffer.
 * @return The value copied from device to host.
 *
 * @note This performs a synchronous device-to-host copy of sizeof(TargetType).
 */
template <class TargetType>
TargetType Tensor::at(long index) const {
    TargetType result;
    CUDA_CHECK(cudaMemcpy(&result, get<TargetType>() + index, sizeof(TargetType), cudaMemcpyDeviceToHost));
    return result;
}

namespace {
/**
 * @brief Copy a contiguous range of elements to host and print them to stdout.
 *
 * @tparam TrueType   Element type as stored in device memory (used for memcpy and host buffer).
 * @tparam PrintType  Type used for printing/casting (e.g., print fp16/bf16/fp8 as float).
 * @param tensor  Source tensor (device-resident data).
 * @param offset  Start element offset into @p tensor (in elements).
 * @param count   Number of elements to print.
 *
 * @note For byte tensors, output is printed in hexadecimal.
 */
template <class TrueType, class PrintType>
void do_print(const Tensor& tensor, long offset, long count) {
    std::ios_base::fmtflags old_flags{std::cout.flags()};

    auto sz = get_dtype_size(tensor.DType);
    std::vector<TrueType> host_buffer(count);
    CUDA_CHECK(cudaMemcpy(host_buffer.data(), tensor.Data + offset * sz, count * sz, cudaMemcpyDeviceToHost));
    if constexpr (std::is_same_v<TrueType, std::byte>)
        std::cout << std::hex;
    for (long i = 0; i < count; ++i)
        std::cout << (PrintType)host_buffer[i] << " ";
    std::cout << std::endl;
    std::cout.flags(old_flags);
}
} // namespace

/**
 * @brief Print a sample of tensor elements to stdout.
 *
 * Dispatches based on the tensor's DType and prints @p count elements starting at @p offset.
 *
 * @param offset  Start element offset (in elements).
 * @param count   Number of elements to print.
 */
void Tensor::print_sample(long offset, long count) const {
    switch (DType) {
    case ETensorDType::FP32:
        do_print<float, float>(*this, offset, count);
        break;
    case ETensorDType::BF16:
        do_print<nv_bfloat16, float>(*this, offset, count);
        break;
    case ETensorDType::FP16:
        do_print<half, float>(*this, offset, count);
        break;
    case ETensorDType::FP8_E4M3:
        do_print<__nv_fp8_e4m3, float>(*this, offset, count);
        break;
    case ETensorDType::FP8_E5M2:
        do_print<__nv_fp8_e5m2, float>(*this, offset, count);
        break;
    case ETensorDType::INT32:
        do_print<int, int>(*this, offset, count);
        break;
    case ETensorDType::INT8:
        do_print<int8_t, int>(*this, offset, count);
        break;
    case ETensorDType::BYTE:
        do_print<std::byte, int>(*this, offset, count);
        break;
    }
}

/**
 * @brief Construct a shard wrapper that initially represents the full (unsharded) tensor.
 *
 * @param src Source tensor to wrap; shard metadata is initialized to a single shard.
 *
 * @note This is view-like: it copies the Tensor header and shares the same underlying Data pointer.
 */
TensorShard::TensorShard(const Tensor& src) : Tensor(src), GlobalShape(src.Sizes), ShardIndex(0), NumShards(1) {
}

/**
 * @brief Compute the total number of elements in the global (pre-sharding) shape.
 *
 * @return Product of GlobalShape[0..Rank).
 */
std::size_t TensorShard::global_nelem() const {
    std::size_t sz = 1;
    for (int i = 0; i < Rank; ++i)
        sz *= GlobalShape[i];
    return sz;
}

/**
 * @brief Compute this shard's starting element offset within the global tensor.
 *
 * @return Offset in elements (not bytes) equal to nelem() * ShardIndex.
 */
std::ptrdiff_t TensorShard::shard_offset() const {
    return nelem() * ShardIndex;
}

/**
 * @brief Create a TensorShard view representing one shard of a tensor along dimension 0.
 *
 * The input tensor is split evenly into @p num shards along the first dimension.
 *
 * @param src  Source tensor to shard (view semantics; no copy).
 * @param idx  Shard index in [0, num).
 * @param num  Total number of shards; must evenly divide src.Sizes[0] and src.bytes().
 * @return TensorShard view for shard @p idx with adjusted Sizes[0] and Data pointer.
 */
TensorShard shard_view(const Tensor& src, int idx, int num) {
    Tensor shard{src};
    shard.Sizes[0] = div_exact(src.Sizes[0], static_cast<long>(num));
    shard.Data = src.Data + div_exact(src.bytes(), static_cast<std::size_t>(num)) * idx;
    return TensorShard{shard, idx, num, src.Sizes};
}
