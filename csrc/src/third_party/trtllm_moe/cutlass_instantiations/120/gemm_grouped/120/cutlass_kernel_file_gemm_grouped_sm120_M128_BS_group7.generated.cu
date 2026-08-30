#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"
namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels_oss
{


#if defined(ENABLE_FP8) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, __nv_fp8_e4m3, SafeFP4, half,
        void, EpilogueOpDefault, NONE,
        128, 32, 128, 1, 1, 1,
        true, false, false, true);
#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, __nv_fp8_e4m3, SafeFP4, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        128, 32, 128, 1, 1, 1,
        true, false, false, true);
#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, __nv_fp8_e4m3, SafeFP4, half,
        void, EpilogueOpDefault, NONE,
        128, 32, 128, 1, 1, 1,
        true, false, false, false);
#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, __nv_fp8_e4m3, SafeFP4, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        128, 32, 128, 1, 1, 1,
        true, false, false, false);
#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, __nv_fp8_e4m3, SafeFP4, half,
        void, EpilogueOpDefault, NONE,
        128, 64, 128, 1, 1, 1,
        true, false, false, true);
#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, __nv_fp8_e4m3, SafeFP4, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        128, 64, 128, 1, 1, 1,
        true, false, false, true);
#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, __nv_fp8_e4m3, SafeFP4, half,
        void, EpilogueOpDefault, NONE,
        128, 64, 128, 1, 1, 1,
        true, false, false, false);
#endif

#if defined(ENABLE_FP8) && defined(ENABLE_FP4)
        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm120, __nv_fp8_e4m3, SafeFP4, __nv_bfloat16,
        void, EpilogueOpDefault, NONE,
        128, 64, 128, 1, 1, 1,
        true, false, false, false);
#endif

} // namespace cutlass_kernels_oss
} // namespace kernels
} // namespace tensorrt_llm
