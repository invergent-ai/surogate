#include "kernels/column_scale.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace {
template<typename T, bool Reduce>
__global__ void columns_kernel(T* out, const T* a, const T* b, long rows, long columns) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if constexpr (Reduce) {
        if (i >= columns) return;
        float sum = 0;
        for (long row = 0; row < rows; ++row) {
            const long offset = row * columns + i;
            sum += static_cast<float>(a[offset]) * static_cast<float>(b[offset]);
        }
        out[i] = static_cast<T>(sum);
    } else {
        if (i >= rows * columns) return;
        out[i] = static_cast<T>(static_cast<float>(a[i]) * static_cast<float>(b[i % columns]));
    }
}

template<bool Reduce>
void dispatch(Tensor& out, const Tensor& a, const Tensor& b, long rows, long columns, cudaStream_t stream) {
    const auto grid = static_cast<unsigned>(((Reduce ? columns : rows * columns) + 255) / 256);
    if (a.DType == ETensorDType::BF16) {
        columns_kernel<nv_bfloat16, Reduce><<<grid, 256, 0, stream>>>(out.get<nv_bfloat16>(), a.get<nv_bfloat16>(), b.get<nv_bfloat16>(), rows, columns);
    } else if (a.DType == ETensorDType::FP16) {
        columns_kernel<half, Reduce><<<grid, 256, 0, stream>>>(out.get<half>(), a.get<half>(), b.get<half>(), rows, columns);
    } else if (a.DType == ETensorDType::FP32) {
        columns_kernel<float, Reduce><<<grid, 256, 0, stream>>>(out.get<float>(), a.get<float>(), b.get<float>(), rows, columns);
    } else {
        throw std::invalid_argument("column scaling requires floating-point tensors");
    }
    CUDA_CHECK(cudaGetLastError());
}
}

void column_scale(Tensor& out, const Tensor& data, const Tensor& scale, long rows, long columns, cudaStream_t stream) {
    dispatch<false>(out, data, scale, rows, columns, stream);
}
void column_scale_gradient(Tensor& out, const Tensor& grad, const Tensor& data, long rows, long columns, cudaStream_t stream) {
    dispatch<true>(out, grad, data, rows, columns, stream);
}
