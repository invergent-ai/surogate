// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <unordered_map>
#include <vector>
#include "runtime/executor/glm_decode_state.h"
#include "runtime/executor/execution_request.h"
#include "runtime/jit/jit_kernel.h"

class GlmDsaKernels {
public:
    GlmDsaKernels() = default;
    GlmDsaKernels(const GlmDsaKernels&) = delete;
    GlmDsaKernels& operator=(const GlmDsaKernels&) = delete;
    GlmDsaKernels(GlmDsaKernels&&) = default;
    GlmDsaKernels& operator=(GlmDsaKernels&&) = default;
    void load(const std::unordered_map<std::string, std::string>& manifests);
    [[nodiscard]] bool is_ready() const {
        return mKernels.size() == 17;
    }
    [[nodiscard]] std::size_t workspace_bytes(int B, int T, int length = 0) const;
    // Bound live score storage independently of the query sequence length.
    static constexpr int IndexerQueryTile = 128;
    static constexpr int DecodeQueryTile = 8;
    [[nodiscard]] static std::size_t indexer_workspace_bytes(int B, int T, int TK, int D, int P, int select);
    [[nodiscard]] int selection_slots() const;
    void gather_latents(const Tensor& latent,
                        const Tensor& indices,
                        const Tensor& out,
                        const Tensor& selected,
                        int slots,
                        cudaStream_t stream) const;
    void repack_kv(const Tensor& projected, const Tensor& out, cudaStream_t stream) const;
    void attention_selected(const Tensor& qkv,
                            const Tensor& indices,
                            const Tensor& kv,
                            const Tensor& out,
                            const Tensor& lse,
                            int slots,
                            cudaStream_t stream) const;
    void indexer(const std::vector<Tensor>& inputs,
                 const Tensor& indices,
                 const Tensor& workspace,
                 cudaStream_t stream,
                 dsl::GlmDecodeState* cache = nullptr,
                 int layer = 0,
                 const dsl::ExecutionRequest* request = nullptr) const;
    void attention(const Tensor& qkv,
                   const Tensor& indices,
                   const Tensor& out,
                   const Tensor& lse,
                   cudaStream_t stream,
                   dsl::GlmDecodeState* cache = nullptr,
                   int layer = 0) const;
    void backward(const Tensor& dout,
                  const Tensor& qkv,
                  const Tensor& indices,
                  const Tensor& out,
                  const Tensor& lse,
                  const Tensor& dqkv,
                  cudaStream_t stream) const;

private:
    [[nodiscard]] int constant(const char* kernel, const char* name) const;
    template <typename... Args>
    void launch(const char* name, dim3 grid, cudaStream_t stream, Args... args) const {
        std::array<void*, sizeof...(Args)> params{static_cast<void*>(&args)...};
        mKernels.at(name).launch_triton(grid, params.data(), params.size(), stream);
    }
    std::unordered_map<std::string, JitKernel> mKernels;
};
