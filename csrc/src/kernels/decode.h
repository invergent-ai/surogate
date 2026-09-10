#pragma once
#include "utilities/tensor.h"
#include <cuda_runtime.h>

void decode_paged_attention(const Tensor& qkv,
                            const Tensor& out,
                            const Tensor& lse,
                            const Tensor& pages,
                            const Tensor& scratch,
                            int position,
                            int T,
                            int Hq,
                            int Hkv,
                            int D,
                            int window,
                            float scale,
                            cudaStream_t stream);
void decode_gather_pages(const Tensor& pages,
                         const Tensor& indices,
                         const Tensor& out,
                         const Tensor& slots,
                         int count,
                         int width,
                         cudaStream_t stream);
void decode_delta_rule(const Tensor& q,
                       const Tensor& k,
                       const Tensor& v,
                       const Tensor& g,
                       const Tensor& beta,
                       const Tensor& state,
                       const Tensor& out,
                       bool initial,
                       float scale,
                       cudaStream_t stream);
