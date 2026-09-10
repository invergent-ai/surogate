#pragma once

// One row of an operator's request metadata. Pointer tables and state buffers
// stay request-owned; graph bindings can change membership without embedding
// request addresses in captured kernels.
struct DecodeCacheBinding {
    void* data;
    int length;
    int pages;
};
#include "utilities/tensor.h"
#include <cuda_runtime.h>

void rollout_consistent_attention(const Tensor& qkv,
                                  const Tensor& out,
                                  const Tensor& lse,
                                  const Tensor& scratch,
                                  const int* cu_seqlens,
                                  int num_docs,
                                  int B,
                                  int T,
                                  int Hq,
                                  int Hkv,
                                  int D,
                                  int window,
                                  float scale,
                                  int query_tile,
                                  cudaStream_t stream);

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
                       const Tensor& bindings,
                       const Tensor& out,
                       const Tensor& final_state,
                       float scale,
                       cudaStream_t stream);

constexpr int DELTA_RULE_CHECKPOINT = 32;
void rollout_delta_rule(const Tensor& q,
                        const Tensor& k,
                        const Tensor& v,
                        const Tensor& g,
                        const Tensor& beta,
                        const Tensor& out,
                        const Tensor& state,
                        const Tensor& checkpoints,
                        const int* cu_seqlens,
                        int num_docs,
                        float scale,
                        cudaStream_t stream);
void rollout_delta_rule_backward(const Tensor& q,
                                 const Tensor& k,
                                 const Tensor& v,
                                 const Tensor& g,
                                 const Tensor& beta,
                                 const Tensor& dout,
                                 const Tensor& dq,
                                 const Tensor& dk,
                                 const Tensor& dv,
                                 const Tensor& dg,
                                 const Tensor& dbeta,
                                 const Tensor& checkpoints,
                                 const int* cu_seqlens,
                                 int num_docs,
                                 float scale,
                                 cudaStream_t stream);

// Pack request-owned recurrent state into a contiguous batch (zero new rows),
// or scatter a completed batch back to those same request buffers.
void decode_copy_state(const Tensor& bindings, const Tensor& state, int B, bool scatter, cudaStream_t stream);
void decode_conv_input(const Tensor& x, const Tensor& bindings, const Tensor& extended, int tail, cudaStream_t stream);
void decode_conv_output(const Tensor& computed,
                        const Tensor& extended,
                        const Tensor& bindings,
                        const Tensor& output,
                        int tail,
                        cudaStream_t stream);
void decode_gather_rows(const Tensor& input,
                        const Tensor& positions,
                        const Tensor& output,
                        int B,
                        int T,
                        int C,
                        cudaStream_t stream);
void decode_append_pages_batch(const Tensor& input,
                               const Tensor& bindings,
                               int B,
                               int T,
                               int width,
                               cudaStream_t stream);
void decode_gather_pages_batch(const Tensor& bindings,
                               const Tensor& indices,
                               const Tensor& out,
                               const Tensor& slots,
                               int T,
                               int query_start,
                               int queries,
                               int count,
                               int width,
                               cudaStream_t stream);

void decode_append_kv_batch(const Tensor& qkv,
                            const Tensor& bindings,
                            int B,
                            int T,
                            int Hq,
                            int Hkv,
                            int D,
                            cudaStream_t stream);
void decode_paged_attention_batch(const Tensor& qkv,
                                  const Tensor& out,
                                  const Tensor& lse,
                                  const Tensor& bindings,
                                  const Tensor& scratch,
                                  int B,
                                  int T,
                                  int Hq,
                                  int Hkv,
                                  int D,
                                  int pages,
                                  int window,
                                  float scale,
                                  cudaStream_t stream);
