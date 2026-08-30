#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"
namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels_oss
{


#if defined(ENABLE_FP4) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, SafeFP4, SafeFP4, half,
        void, EpilogueOpDefault, NONE,
        256, 128, 128, 1, 1, 1,
        false, false, false, true);
#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, SafeFP4, SafeFP4, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        256, 128, 128, 1, 1, 1,
        false, false, false, true);
#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, SafeFP4, SafeFP4, half,
        void, EpilogueOpDefault, NONE,
        256, 128, 128, 1, 1, 1,
        false, false, false, false);
#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, SafeFP4, SafeFP4, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        256, 128, 128, 1, 1, 1,
        false, false, false, false);
#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, SafeFP4, SafeFP4, half,
        void, EpilogueOpDefault, FINALIZE,
        256, 128, 128, 1, 1, 1,
        false, false, false, true);
#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, SafeFP4, SafeFP4, __nv_bfloat16,
        void, EpilogueOpDefault, FINALIZE,
        256, 128, 128, 1, 1, 1,
        false, false, false, true);
#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, SafeFP4, SafeFP4, half,
        void, EpilogueOpDefault, FINALIZE,
        256, 128, 128, 1, 1, 1,
        false, false, false, false);
#endif

#if defined(ENABLE_FP4) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, SafeFP4, SafeFP4, __nv_bfloat16,
        void, EpilogueOpDefault, FINALIZE,
        256, 128, 128, 1, 1, 1,
        false, false, false, false);
#endif

} // namespace cutlass_kernels_oss
} // namespace kernels
} // namespace tensorrt_llm
