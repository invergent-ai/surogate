// Forward-direction launcher for the ported mem-efficient attention
// kernel (upstream: PyTorch aten/src/ATen/native/transformers/cuda/
// mem_eff_attention, itself derived from xformers). BF16 / SM80 ABI
// variant only for the prototype.
//
// This is the skinny entry point that:
//   * picks a precompiled kernel variant based on head_dim_qk and
//     head_dim_v,
//   * builds the AttentionKernel::Params struct from raw pointers,
//   * computes the launch grid,
//   * and launches.
//
// Compilation note: this .cu only comes alive once the surrounding
// cutlass glue (see kernels/cutlassF_bf16_aligned.cu) is wired into the
// build. See design/capture-safe-runtime-plan.md, "Phase 4 attempt 1"
// for the integration checklist.

#include "runtime/attention/mem_eff/mem_eff_dispatch.h"

#include "runtime/attention/mem_eff/kernel_forward.h"

#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace surogate {
namespace mem_eff {

namespace {

// The auto-generated kernels/cutlassF_bf16_aligned.cu provides these
// three bf16 forward variants (see upstream generate_kernels.py):
//   - 64x64  block, kMaxK=64      (register-file output, Hs <= 64)
//   - 64x128 block, kMaxK=128     (register-file output, Hs <= 128)
//   - 32x128 block, kMaxK=65536   (global-memory output, unbounded Hs)
// The GMEM variant is the one that lets us support Hs=256 and Hs=512
// without recompiling — at the cost of some bandwidth for writing back
// accumulator partial sums. Prototype wiring targets GMEM only.

using AttentionKernelGmem =
    PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t,  // scalar_t
                                            cutlass::arch::Sm80,  // ArchTag — SM80 ABI covers Ampere/Hopper/Blackwell
                                            true,                 // IsAligned
                                            32,                   // kQueriesPerBlock
                                            128,                  // kKeysPerBlock
                                            65536,                // kMaxK (uint32_max-style "unlimited")
                                            true,   // kSingleValueIteration_ ignored when kMaxK > kKeysPerBlock
                                            true>;  // kAddMask

extern "C" __global__ void fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(typename AttentionKernelGmem::Params p);

}  // anonymous namespace

bool forward_supported(int head_dim_qk, int head_dim_v) {
    // The GMEM variant handles arbitrary head dim. Later phases may add
    // register-file variants for the fast path on Hs<=64/128.
    return head_dim_qk > 0 && head_dim_v > 0 && head_dim_qk == head_dim_v;
}

void forward_bf16_sm80(const ForwardArgs& args) {
    if (!forward_supported(args.head_dim_qk, args.head_dim_v)) {
        std::ostringstream oss;
        oss << "mem_eff::forward_bf16_sm80: unsupported head-dim combo "
            << "qk=" << args.head_dim_qk << " v=" << args.head_dim_v;
        throw std::runtime_error(oss.str());
    }

    typename AttentionKernelGmem::Params p;
    p.query_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(args.q_ptr);
    p.key_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(args.k_ptr);
    p.value_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(args.v_ptr);
    p.output_ptr = reinterpret_cast<cutlass::bfloat16_t*>(args.out_ptr);
    p.logsumexp_ptr = args.lse_ptr;

    p.seqstart_q_ptr = args.seqstart_q_ptr;
    p.seqstart_k_ptr = args.seqstart_k_ptr;
    p.seqlen_k_ptr = args.seqlen_k_ptr;

    p.head_dim = args.head_dim_qk;
    p.head_dim_value = args.head_dim_v;
    p.num_queries = args.total_tokens;
    p.num_keys = args.total_tokens;
    p.num_keys_absolute = args.total_tokens;

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

    p.num_heads = args.num_heads_q;
    p.num_batches = args.num_batches;
    p.window_size = args.window_size;
    p.scale = args.softmax_scale;

    p.custom_mask_type = args.causal ? PyTorchMemEffAttention::AttentionKernelGmemFwd_causal_tag
                                     : PyTorchMemEffAttention::AttentionKernelGmemFwd_nomask_tag;

    // Launch grid follows the kernel's expectation:
    //   blockIdx.x -> query-block index within a doc
    //   blockIdx.y -> head
    //   blockIdx.z -> batch/doc
    // When inputs are packed, query blocks that fall beyond a doc's
    // length early-return via ``advance_to_block``'s bounds check at
    // kernel_forward.h:233. So the grid can be sized to the worst-case
    // per-step document and stay capture-safe.
    dim3 grid((args.total_tokens + AttentionKernelGmem::kQueriesPerBlock - 1) / AttentionKernelGmem::kQueriesPerBlock,
              args.num_heads_q,
              args.num_batches);
    dim3 block(AttentionKernelGmem::kNumThreads);

    // Shared-memory footprint for the GMEM variant.
    const size_t smem_bytes = sizeof(typename AttentionKernelGmem::SharedStorage);
    if (smem_bytes >= 48 * 1024) {
        cudaFuncSetAttribute(fmha_cutlassF_bf16_aligned_32x128_gmem_sm80,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
    }

    fmha_cutlassF_bf16_aligned_32x128_gmem_sm80<<<grid, block, smem_bytes, args.stream>>>(p);
}

}  // namespace mem_eff
}  // namespace surogate
