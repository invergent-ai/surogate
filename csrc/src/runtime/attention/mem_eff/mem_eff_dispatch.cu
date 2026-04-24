// Forward-direction launcher for the ported mem-efficient attention
// kernel. BF16, SM80 ABI, GMEM variant (kMaxK=65536 → unbounded Hs).

#include "runtime/attention/mem_eff/mem_eff_dispatch.h"

#include "runtime/attention/mem_eff/kernel_forward.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>

using AttentionKernelGmem = PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t,  // scalar_t
                                                                    cutlass::arch::Sm80,  // ArchTag
                                                                    true,                 // IsAligned
                                                                    32,                   // kQueriesPerBlock
                                                                    128,                  // kKeysPerBlock
                                                                    65536,                // kMaxK
                                                                    true,                 // kSingleValueIteration_
                                                                    true>;                // kAddMask

// Auto-generated (kernels/cutlassF_bf16_aligned.cu); at global scope.
__global__ void fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(typename AttentionKernelGmem::Params p);

namespace surogate {
namespace mem_eff {

bool forward_supported(int head_dim_qk, int head_dim_v) {
    return head_dim_qk > 0 && head_dim_v > 0 && head_dim_qk == head_dim_v;
}

void forward_bf16_sm80(const ForwardArgs& args) {
    if (!forward_supported(args.head_dim_qk, args.head_dim_v)) {
        std::ostringstream oss;
        oss << "mem_eff::forward_bf16_sm80: unsupported head-dim combo qk=" << args.head_dim_qk
            << " v=" << args.head_dim_v;
        throw std::runtime_error(oss.str());
    }

    typename AttentionKernelGmem::Params p;
    p.query_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(args.q_ptr);
    p.key_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(args.k_ptr);
    p.value_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(args.v_ptr);
    p.output_ptr = reinterpret_cast<cutlass::bfloat16_t*>(args.out_ptr);
    p.logsumexp_ptr = args.lse_ptr;
    p.output_accum_ptr = args.output_accum_ptr;

    p.seqstart_q_ptr = args.seqstart_q_ptr;
    p.seqstart_k_ptr = args.seqstart_k_ptr;
    p.seqlen_k_ptr = args.seqlen_k_ptr;

    p.head_dim = args.head_dim_qk;
    p.head_dim_value = args.head_dim_v;
    p.num_queries = args.num_queries;
    p.num_keys = args.num_keys;
    p.num_keys_absolute = args.num_keys;
    p.num_heads = args.num_heads;
    p.num_batches = args.num_batches;

    p.q_strideM = args.q_strideM;
    p.k_strideM = args.k_strideM;
    p.v_strideM = args.v_strideM;
    p.o_strideM = args.o_strideM;
    p.q_strideH = args.q_strideH;
    p.k_strideH = args.k_strideH;
    p.v_strideH = args.v_strideH;
    p.q_strideB = args.q_strideB;
    p.k_strideB = args.k_strideB;
    p.v_strideB = args.v_strideB;

    p.window_size = args.window_size;
    p.scale = args.softmax_scale;
    p.custom_mask_type = args.causal ? static_cast<uint8_t>(AttentionKernelGmem::CausalFromTopLeft)
                                     : static_cast<uint8_t>(AttentionKernelGmem::NoCustomMask);

    // Let cutlass' check_supported validate alignment / shape
    // preconditions (throws via our TORCH_CHECK compat shim).
    AttentionKernelGmem::check_supported(p);

    const size_t smem_bytes = sizeof(typename AttentionKernelGmem::SharedStorage);
    if (smem_bytes > 0xc000) {
        // Opt into >48KB dynamic shared memory per block.
        cudaError_t err = cudaFuncSetAttribute(fmha_cutlassF_bf16_aligned_32x128_gmem_sm80,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(smem_bytes));
        if (err == cudaErrorInvalidValue) {
            std::ostringstream oss;
            oss << "mem_eff::forward_bf16_sm80: GPU does not have enough shared-memory "
                << "(kernel requires " << (smem_bytes / 1024) << " KiB)";
            throw std::runtime_error(oss.str());
        }
    }

    // Use the kernel's own grid helpers — `num_queries` is the per-doc
    // max length here, and the per-block bounds check inside
    // ``advance_to_block`` early-returns for blocks that fall beyond a
    // doc's actual length, so oversizing the grid is capture-safe.
    fmha_cutlassF_bf16_aligned_32x128_gmem_sm80<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, args.stream>>>(p);
}

}  // namespace mem_eff
}  // namespace surogate
