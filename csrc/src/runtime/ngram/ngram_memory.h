// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Runtime side of the hashed n-gram memory (kernels/ngram_memory.h): the spec the DSL records in the
// IR (`ngram_memory_spec`, surogate/dsl/modules/ngram_memory.py), the frozen table's residency, and
// the forward / backward of the memory on a block of tokens (used by the executor on the memory
// layer's input norm and by the op fixtures of the tests).

#ifndef SUROGATE_SRC_RUNTIME_NGRAM_NGRAM_MEMORY_H
#define SUROGATE_SRC_RUNTIME_NGRAM_NGRAM_MEMORY_H

#include <cstdint>
#include <string>

#include "kernels/ngram_memory.h"
#include "utilities/tensor.h"

typedef struct cublasLtContext* cublasLtHandle_t;

namespace ngram_memory {

struct Spec {
    int layer = -1;            ///< decoder layer whose input the memory updates (0-based)
    Hash hash;                 ///< engine hash constants
    int head_dim = 0;          ///< values per table row
    std::int64_t row_count = 0;
    int row_bytes = 0;         ///< head_dim / 32 * 18 (IQ4_NL)
    std::string table_path;    ///< raw IQ4_NL rows (the GGUF tensor payload)
    std::int64_t table_offset = 0;  ///< byte offset of row 0 in the file
    bool host_table = true;    ///< pinned, mapped host memory (else a device copy)
    float eps = 1e-6F;
    float init_std = 0.02F;    ///< W_k init std (W_v starts at 0, gains at 1)
    std::uint64_t seed = 0;
    float lr_scale = 1.0F;     ///< learning-rate multiplier of the memory parameters
    ZeroTokens zero_tokens;
    RowShuffle row_shuffle;    ///< control arm: rows relabelled (never served)

    [[nodiscard]] bool enabled() const { return layer >= 0; }
    [[nodiscard]] int embed_dim() const { return hash.heads * head_dim; }
    /// Parses the IR's JSON spec; throws std::invalid_argument on an inconsistent one.
    static Spec parse(const std::string& json);
};

/// A device-accessible pointer to the spec's table for `device` (current device). Host residency
/// keeps one pinned, mapped copy per process (cudaHostRegisterPortable); device residency one copy
/// per device. Loaded on first use; the file size must equal row_count * row_bytes + offset.
const std::uint8_t* acquire_table(const Spec& spec, int device);

/// Memory forward on N tokens. r [N, C] (the residual the gate reads), emb [N, E], W_k / W_v [C, E]
/// in the dtype of emb (BF16 or FP32); gk, gq FP32 [C]. Writes k, v [N, C], stats [N, kTokenStats]
/// and u [N, C] (dtype of r).
void forward_core(Tensor& k, Tensor& v, Tensor& stats, Tensor& u, float* global_stats, const Tensor& r,
                  const Tensor& emb, const Tensor& Wk, const Tensor& Wv, const float* gk, const float* gq,
                  const int* ids, const ZeroTokens& zero, float eps, cublasLtHandle_t handle, Tensor& workspace,
                  cudaStream_t stream);

/// Memory backward on N tokens. d_res [N, C]: dL/d(r + u) in, dL/dr out. Accumulates into dWk, dWv
/// [C, E] and dgk, dgq [C] (FP32 or BF16). Scratch: dk, dv [N, C] (dtype of r), w FP32 [N],
/// partials FP32 [gain_grad_parts(N), C].
void backward_core(Tensor& d_res, Tensor& dWk, Tensor& dWv, Tensor& dgk, Tensor& dgq, Tensor& dk, Tensor& dv,
                   Tensor& w, Tensor& partials, float* abs_max, const Tensor& r, const Tensor& emb, const Tensor& k,
                   const Tensor& v, const Tensor& stats, const float* gk, const float* gq, const int* ids,
                   const ZeroTokens& zero, cublasLtHandle_t handle, Tensor& workspace, cudaStream_t stream);

}  // namespace ngram_memory

#endif  // SUROGATE_SRC_RUNTIME_NGRAM_NGRAM_MEMORY_H
