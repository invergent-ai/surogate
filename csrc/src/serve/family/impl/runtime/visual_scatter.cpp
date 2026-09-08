#include "family/impl/runtime/visual_scatter.h"

#include "core/device.h"
#include "api/ops/scatter.h"
#include "api/ops/residual_add.h"

#include <stdexcept>

namespace sinfer::family::detail {
namespace {

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

} // namespace

void add_visual_embeddings(Tensor& residual, const Tensor& features,
                           std::span<const std::int32_t> indices, cudaStream_t stream) {
    if (features.ne[1] != static_cast<std::int32_t>(indices.size())) {
        throw std::invalid_argument("visual residual features and indices disagree");
    }
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= residual.ne[1] || (i && indices[i] <= indices[i - 1])) {
            throw std::invalid_argument("visual residual indices must be ordered, unique and in bounds");
        }
    }
    for (std::size_t begin = 0; begin < indices.size();) {
        auto end = begin + 1;
        while (end < indices.size() && indices[end] == indices[end - 1] + 1) { ++end; }
        const auto count = static_cast<std::int32_t>(end - begin);
        Tensor destination = residual.slice(1, indices[begin], count);
        ops::residual_add(features.slice(1, static_cast<std::int32_t>(begin), count), destination, stream);
        begin = end;
    }
}

void scatter_shifted_visual_embeddings(Tensor& input_embeddings, const Tensor& visual_embeddings,
                                       const family::MtpVisualOverlap& overlap,
                                       Tensor& destination_indices, cudaStream_t stream) {
    if (overlap.empty() || destination_indices.dtype != DType::I32 ||
        destination_indices.ne[0] != static_cast<std::int32_t>(overlap.size())) {
        throw std::invalid_argument("shifted visual scatter has invalid destination indices");
    }
    const auto count = static_cast<std::int32_t>(overlap.size());
    copy_i32(overlap.destination_columns.data(), destination_indices, stream);
    Tensor embeddings =
        visual_embeddings.slice(1, static_cast<std::int32_t>(overlap.source_begin), count);
    ops::scatter(embeddings, destination_indices, input_embeddings, stream);
}

} // namespace sinfer::family::detail
