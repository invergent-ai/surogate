// Backward-direction launcher for the ported mem-efficient attention
// kernel. BF16, SM80 ABI, kMaxK=65536 (unbounded Hs) GMEM variant.
// Mirrors mem_eff_dispatch.cu (forward); same template choice keeps
// the backward parameter layout matched to the forward kernel's saved
// logsumexp.

#include "runtime/attention/mem_eff/mem_eff_dispatch.h"

#include "runtime/attention/mem_eff/kernel_backward.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>

using AttentionBackwardKernelGmem = PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::arch::Sm80,  // ArchTag
                                                                                    cutlass::bfloat16_t,  // scalar_t
                                                                                    true,                 // kIsAligned_
                                                                                    false,   // kApplyDropout_
                                                                                    false,   // kPreload_
                                                                                    64,      // kBlockSizeI
                                                                                    64,      // kBlockSizeJ
                                                                                    65536>;  // kMaxK

// Auto-generated (kernels/cutlassB_bf16_aligned_k65536.cu), global scope.
__global__ void fmha_cutlassB_bf16_aligned_64x64_k65536_sm80(typename AttentionBackwardKernelGmem::Params p);

namespace surogate {
namespace mem_eff {

namespace {

AttentionBackwardKernelGmem::Params build_params(const BackwardArgs& args) {
    AttentionBackwardKernelGmem::Params p;

    p.query_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(args.q_ptr);
    p.key_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(args.k_ptr);
    p.value_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(args.v_ptr);
    p.logsumexp_ptr = args.lse_ptr;
    p.output_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(args.out_ptr);
    p.grad_output_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(args.d_out_ptr);
    p.delta_ptr = args.delta_ptr;
    p.grad_query_ptr = reinterpret_cast<cutlass::bfloat16_t*>(args.d_q_ptr);
    p.grad_key_ptr = reinterpret_cast<cutlass::bfloat16_t*>(args.d_k_ptr);
    p.grad_value_ptr = reinterpret_cast<cutlass::bfloat16_t*>(args.d_v_ptr);

    p.cu_seqlens_q_ptr = args.cu_seqlens_q_ptr;
    p.cu_seqlens_k_ptr = args.cu_seqlens_k_ptr;

    p.head_dim = args.head_dim_qk;
    p.head_dim_value = args.head_dim_v;
    p.num_queries = args.num_queries;
    p.num_keys = args.num_keys;
    p.num_batches = args.num_batches;
    p.num_heads = args.num_heads;
    p.custom_mask_type = args.causal ? static_cast<uint8_t>(AttentionBackwardKernelGmem::CausalFromTopLeft)
                                     : static_cast<uint8_t>(AttentionBackwardKernelGmem::NoCustomMask);
    p.window_size = args.window_size;
    p.scale = args.softmax_scale;

    p.q_strideM = args.q_strideM;
    p.k_strideM = args.k_strideM;
    p.v_strideM = args.v_strideM;
    p.gQKV_strideM_multiplier = args.gQKV_strideM_multiplier;
    p.q_strideH = args.q_strideH;
    p.k_strideH = args.k_strideH;
    p.v_strideH = args.v_strideH;
    p.q_strideB = args.q_strideB;
    p.k_strideB = args.k_strideB;
    p.v_strideB = args.v_strideB;

    p.o_strideB = args.o_strideB;
    p.o_strideH = args.o_strideH;
    p.gO_strideM = args.gO_strideM;
    p.gO_strideH = args.gO_strideH;
    p.gO_strideB = args.gO_strideB;

    p.lse_strideB = args.lse_strideB;
    p.lse_strideH = args.lse_strideH;
    p.delta_strideB = args.delta_strideB;
    p.delta_strideH = args.delta_strideH;

    p.gQ_strideH = args.gQ_strideH;
    p.gK_strideH = args.gK_strideH;
    p.gV_strideH = args.gV_strideH;
    p.gQ_strideB = args.gQ_strideB;
    p.gK_strideB = args.gK_strideB;
    p.gV_strideB = args.gV_strideB;

    p.num_splits_key = args.num_splits_key;
    p.workspace = args.workspace_ptr;
    return p;
}

}  // namespace

std::size_t backward_workspace_bytes(const BackwardArgs& args) {
    auto p = build_params(args);
    return static_cast<std::size_t>(p.workspace_size());
}

bool backward_workspace_needs_zero(const BackwardArgs& args) {
    auto p = build_params(args);
    return p.should_zero_workspace();
}

void backward_bf16_sm80(const BackwardArgs& args) {
    auto p = build_params(args);
    AttentionBackwardKernelGmem::check_supported(p);

    const size_t smem_bytes = sizeof(typename AttentionBackwardKernelGmem::SharedStorage);
    if (smem_bytes > 0xc000) {
        cudaError_t err = cudaFuncSetAttribute(fmha_cutlassB_bf16_aligned_64x64_k65536_sm80,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(smem_bytes));
        if (err == cudaErrorInvalidValue) {
            std::ostringstream oss;
            oss << "mem_eff::backward_bf16_sm80: GPU does not have enough shared-memory "
                << "(kernel requires " << (smem_bytes / 1024) << " KiB)";
            throw std::runtime_error(oss.str());
        }
    }

    fmha_cutlassB_bf16_aligned_64x64_k65536_sm80<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, args.stream>>>(p);
}

}  // namespace mem_eff
}  // namespace surogate
