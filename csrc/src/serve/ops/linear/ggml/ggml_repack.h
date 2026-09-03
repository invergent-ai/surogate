#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::ggml {

/// Bytes a Q8_0 tensor of `rows` x `k` occupies in the GGUF.
std::size_t q8_0_source_bytes(std::int32_t rows, std::int32_t k);

/// Rearrange Q8_0 into the row-split planes `W8G32_F16S` kernels read.
///
/// The two hold the same numbers -- signed int8 with one binary16 scale per 32 -- and differ only
/// in arrangement: the GGUF interleaves each block's scale with its codes, and the row-split
/// layout gathers all codes into one plane and all scales into another. So this is a permutation
/// of bytes with no arithmetic, which is why serving a Q8_0 checkpoint costs no accuracy however
/// it is arranged, and why the rearranging can happen at load rather than in the file.
///
/// `blocks` is the tensor's Q8_0 bytes in the object's own row order, `out` its planes. Both are
/// device pointers, and `out` must have room for the format's full encoded size.
void q8_0_to_w8_rowsplit_launch(const void* blocks, void* out, std::int32_t rows, std::int32_t k,
                                std::size_t out_bytes, cudaStream_t stream);

} // namespace sinfer::ops::detail::ggml
