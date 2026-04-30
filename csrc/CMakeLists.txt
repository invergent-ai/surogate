cmake_minimum_required(VERSION 3.25...4.0)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum CMake policy version for dependencies")

# Minimum supported GPU architecture is SM89 (Ada Lovelace, RTX 40x0 / L40 / L4).
# SM80-87 (Ampere) and older are not supported.
set(SUROGATE_MIN_CUDA_ARCH 89)

# If no GPU arch has been specified, try to pick an explicit architecture (so we
# can select feature variants like `90a`/`120a` that `-arch=native` won't). Fall
# back to `native` if we cannot detect a GPU at configure time.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _SUROGATE_NVIDIA_SMI_CC
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(_SUROGATE_NVIDIA_SMI_CC)
        string(REPLACE "\n" ";" _SUROGATE_NVIDIA_SMI_CC_LIST "${_SUROGATE_NVIDIA_SMI_CC}")
        list(GET _SUROGATE_NVIDIA_SMI_CC_LIST 0 _SUROGATE_NVIDIA_SMI_CC0)
        string(STRIP "${_SUROGATE_NVIDIA_SMI_CC0}" _SUROGATE_NVIDIA_SMI_CC0)

        if(_SUROGATE_NVIDIA_SMI_CC0 MATCHES "^[0-9]+\\.[0-9]+$")
            # Convert like "12.0" -> "120", "8.9" -> "89"
            string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)$" "\\1\\2" _SUROGATE_CUDA_CC_DIGITS "${_SUROGATE_NVIDIA_SMI_CC0}")

            if(_SUROGATE_CUDA_CC_DIGITS LESS SUROGATE_MIN_CUDA_ARCH)
                message(FATAL_ERROR
                    "Detected GPU compute capability ${_SUROGATE_NVIDIA_SMI_CC0} (SM${_SUROGATE_CUDA_CC_DIGITS}) "
                    "is below the minimum supported architecture SM${SUROGATE_MIN_CUDA_ARCH} (Ada Lovelace). "
                    "Surogate requires SM89 or newer (RTX 40x0 / L40 / L4 / Hopper / Blackwell).")
            endif()

            # Some CUTLASS kernels use "arch-conditional" MMA instructions which
            # require targeting the feature-variant architecture (e.g. `90a`, `100a`, `103a`, `120a`).
            # The 'a' suffix enables tcgen05.mma and block-scaled MMA features (FP4).
            if(_SUROGATE_CUDA_CC_DIGITS STREQUAL "90")
                set(CMAKE_CUDA_ARCHITECTURES 90a)
            elseif(_SUROGATE_CUDA_CC_DIGITS STREQUAL "100")
                set(CMAKE_CUDA_ARCHITECTURES 100a)
            elseif(_SUROGATE_CUDA_CC_DIGITS STREQUAL "103")
                set(CMAKE_CUDA_ARCHITECTURES 103a)
            elseif(_SUROGATE_CUDA_CC_DIGITS STREQUAL "120")
                set(CMAKE_CUDA_ARCHITECTURES 120a)
            elseif(_SUROGATE_CUDA_CC_DIGITS STREQUAL "121")
                set(CMAKE_CUDA_ARCHITECTURES 121a)
            else()
                set(CMAKE_CUDA_ARCHITECTURES "${_SUROGATE_CUDA_CC_DIGITS}")
            endif()

            message(STATUS "Auto-detected GPU compute capability ${_SUROGATE_NVIDIA_SMI_CC0}; setting CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
        else()
            set(CMAKE_CUDA_ARCHITECTURES native)
            message(STATUS "Could not parse GPU compute capability '${_SUROGATE_NVIDIA_SMI_CC0}'; using CMAKE_CUDA_ARCHITECTURES=native")
        endif()
    else()
        set(CMAKE_CUDA_ARCHITECTURES native)
        message(STATUS "Could not detect GPU compute capability; using CMAKE_CUDA_ARCHITECTURES=native")
    endif()
else()
    # User-supplied CMAKE_CUDA_ARCHITECTURES: validate every entry meets the SM89 minimum.
    foreach(_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(_ARCH STREQUAL "native" OR _ARCH STREQUAL "all" OR _ARCH STREQUAL "all-major")
            continue()
        endif()
        string(REGEX REPLACE "[^0-9].*$" "" _ARCH_DIGITS "${_ARCH}")
        if(_ARCH_DIGITS AND _ARCH_DIGITS LESS SUROGATE_MIN_CUDA_ARCH)
            message(FATAL_ERROR
                "CMAKE_CUDA_ARCHITECTURES contains '${_ARCH}', which is below the minimum "
                "supported architecture SM${SUROGATE_MIN_CUDA_ARCH} (Ada Lovelace).")
        endif()
    endforeach()
endif()

include(FetchContent)
project(SUROGATE CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# As cuFile might not be supported everywhere (WSL; AMD), make it an optional dependency
# with a fallback implementation
option(USE_CUFILE "Enable cuFile support for optimized file I/O" ON)
option(USE_NVML "Enable nvml support for detailed GPU stats reporting" ON)
option(PYTHON_BINDING "Build python bindings" ON)
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        EXCLUDE_FROM_ALL
)

FetchContent_Declare(
        cudnn-fe
        GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
        GIT_TAG v1.18.0
        EXCLUDE_FROM_ALL
)

FetchContent_Declare(
        cli11
        QUIET
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.6.1
)
FetchContent_Declare(
        nanobind
        QUIET
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.11.0
)

# we could use fmt only if std::format is not available, but (e.g., g++-12), but
# since it is compatible with std::format, its easier to just use it consistently.
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG        12.0.0
)

# minja: Jinja2 template engine for chat templates (header-only).
# SOURCE_SUBDIR _none skips minja's own CMakeLists.txt (which tries to build tests).
FetchContent_Declare(
        minja
        GIT_REPOSITORY https://github.com/ochafik/minja.git
        GIT_TAG main
        SOURCE_SUBDIR _none
        EXCLUDE_FROM_ALL
)

# Catch2 for tests
FetchContent_Declare(
        catch2
        QUIET
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.6.0
)

# backward-cpp for stack traces on crash
FetchContent_Declare(
        backward
        GIT_REPOSITORY https://github.com/bombela/backward-cpp.git
        GIT_TAG v1.6
)

# CUTLASS for optimized GEMM kernels
# Disable all CUTLASS tests, examples, and profiler to speed up configuration
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Enable CUTLASS tests")
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Enable CUTLASS examples")
set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "Enable CUTLASS tools")
set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "Enable CUTLASS profiler")
# set(CUTLASS_ENABLE_LIBRARY OFF CACHE BOOL "Enable CUTLASS library")
# set(CUTLASS_LIBRARY_KERNELS "" CACHE STRING "Comma-separated list of kernel name filters")
# Only build for SM89+ (Ada/Hopper/Blackwell/Next-gen): 89, 90, 90a, 100a, 103a, 120a, 121a
# SM89: Ada (RTX 40x0/L40/L4), SM90/90a: Hopper, SM100a: Blackwell B200, SM103a: Blackwell B300, SM120a: RTX 50x0/DGX Spark
# Note: SM100/103/120 require 'a' suffix for tcgen05.mma and block-scaled MMA features (FP4)
set(CUTLASS_NVCC_ARCHS "89;90;90a;100a;103a;120a;121a" CACHE STRING "CUDA architectures for CUTLASS")
set(CUTLASS_NVCC_ARCHS_SUPPORTED "89;90;90a;100a;103a;120a;121a" CACHE STRING "NVCC archs to build library kernels")

FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v4.4.1
        EXCLUDE_FROM_ALL
)

# Flash Attention — only the source tree is needed; we compile specific .cu files
# directly into our own static library.  SOURCE_SUBDIR points to a non-existent
# directory so MakeAvailable fetches but doesn't run flash-attention's own build.
FetchContent_Declare(
        flash_attn
        GIT_REPOSITORY https://github.com/Dao-AILab/flash-attention.git
        GIT_TAG v2.8.3
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR _none
        EXCLUDE_FROM_ALL
)

find_package(CUDAToolkit COMPONENTS cuBLAS REQUIRED)

FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(cudnn-fe)
FetchContent_MakeAvailable(cli11)
FetchContent_MakeAvailable(fmt)
FetchContent_MakeAvailable(catch2)
FetchContent_MakeAvailable(cutlass)
FetchContent_MakeAvailable(backward)

FetchContent_MakeAvailable(minja)
FetchContent_MakeAvailable(flash_attn)

# Find libdw (elfutils) or libdwarf for stack traces with backward-cpp
# libdw (from elfutils) is preferred, libdwarf is a fallback
find_path(LIBDW_INCLUDE_DIR NAMES elfutils/libdw.h)
find_library(LIBDW_LIBRARY NAMES dw)
find_path(LIBDWARF_INCLUDE_DIR NAMES libdwarf/libdwarf.h libdwarf.h)
find_library(LIBDWARF_LIBRARY NAMES dwarf)
find_library(LIBELF_LIBRARY NAMES elf)

set(BACKWARD_HAS_DW 0)
set(BACKWARD_HAS_DWARF 0)
set(BACKWARD_LIBS "")

if(LIBDW_INCLUDE_DIR AND LIBDW_LIBRARY)
    message(STATUS "Found libdw: ${LIBDW_LIBRARY} (enables detailed stack traces)")
    set(BACKWARD_HAS_DW 1)
    list(APPEND BACKWARD_LIBS ${LIBDW_LIBRARY})
elseif(LIBDWARF_INCLUDE_DIR AND LIBDWARF_LIBRARY AND LIBELF_LIBRARY)
    message(STATUS "Found libdwarf: ${LIBDWARF_LIBRARY} (enables stack traces)")
    set(BACKWARD_HAS_DWARF 1)
    list(APPEND BACKWARD_LIBS ${LIBDWARF_LIBRARY} ${LIBELF_LIBRARY})
else()
    message(STATUS "Neither libdw nor libdwarf found - stack traces will have limited detail")
    message(STATUS "  Install: sudo apt install libdw-dev (recommended) or libdwarf-dev libelf-dev")
endif()

# optional cufile
if(USE_CUFILE)
    find_package(CUDAToolkit COMPONENTS cuFile QUIET)
    if(TARGET CUDA::cuFile)
        set(CUFILE_SOURCE src/utilities/cu_file.cpp)
        set(CUFILE_LIBS CUDA::cuFile)
    else()
        set(USE_CUFILE 0)
        message(WARNING "cuFile requested but not found, disabling cuFile support")
    endif()
endif ()

if(NOT USE_CUFILE)
    set(CUFILE_SOURCE src/utilities/cu_file_fallback.cpp)
    set(CUFILE_LIBS "")
endif ()

if(USE_NVML)
    find_package(CUDAToolkit COMPONENTS nvml)
    set(NVML_SOURCE src/utilities/gpu_info_nvml.cpp)
    set(NVML_LIBS CUDA::nvml)
else()
    set(NVML_SOURCE src/utilities/gpu_info_fallback.cpp)
    set(NVML_LIBS "")
endif ()

    # allow configuring search paths from env
set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "Folder containing NCCL headers")
set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "Folder containing NCCL library")
find_path(NCCL_INCLUDE_DIRS NAMES nccl.h HINTS ${NCCL_INCLUDE_DIR} REQUIRED)
find_library(NCCL_LIBRARIES NAMES nccl HINTS ${NCCL_LIB_DIR} REQUIRED)
message(STATUS "Using nccl: ${NCCL_LIBRARIES}")
add_library(nvidia::nccl SHARED IMPORTED)
set_property(TARGET nvidia::nccl PROPERTY IMPORTED_LOCATION "${NCCL_LIBRARIES}")
target_include_directories(nvidia::nccl INTERFACE "${NCCL_INCLUDE_DIRS}")

set(MULTI_GPU_DEPS "")
set(MULTI_GPU_DEFS "")

# Threads are always required for the unified multi-GPU communicator
find_package(Threads REQUIRED)
list(APPEND MULTI_GPU_DEPS Threads::Threads)

find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS $ENV{CUDNN_INCLUDE_PATH} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)

find_library(CUDNN_LIBRARY cudnn
        HINTS $ENV{CUDNN_LIBRARY_PATH} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib64 lib)

message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "cuBLAS Library: ${CUDA_cublas_LIBRARY}")
message(STATUS "cuBLASLt Library: ${CUDA_cublasLt_LIBRARY}")

add_library(surogate-common SHARED)

target_sources(surogate-common PRIVATE
    src/utilities/safetensors.cpp
    ${CUFILE_SOURCE}
    ${NVML_SOURCE}
    src/utilities/allocator.cpp
    src/utilities/lazy_allocator.cpp
    src/utilities/utils.cpp
    src/utilities/comm.cpp
    src/utilities/tensor.cpp
    src/utilities/sol.cpp
    src/utilities/dtype.cpp
    src/utilities/stack.cpp
    src/utilities/system_info.cpp
    src/utilities/crash_handler.cpp

    src/runtime/training/dataloader.cpp
    src/runtime/training/logging.cpp
    src/runtime/training/checkpoint.cpp
    src/runtime/training/model.cpp

    src/runtime/dsl/ir.cpp
    src/runtime/dsl/dsl_model.cpp
    src/runtime/dsl/dsl_model_execution.cpp
    src/runtime/dsl/dsl_model_helpers.cpp
    src/runtime/dsl/dsl_model_lora.cpp
    src/runtime/dsl/dsl_model_weights.cpp
    src/runtime/dsl/dsl_param_store.cpp
    src/runtime/dsl/dsl_grad_store.cpp
    src/runtime/dsl/dsl_run_state.cpp
    src/runtime/dsl/dsl_debug.cpp
    src/runtime/dsl/buffer_plan.cpp
    src/runtime/dsl/dsl_weight_manager.cpp
    src/runtime/executor/causal_lm_execution_profile.cpp
    src/runtime/executor/graph_executor.cpp
    src/runtime/executor/graph_executor_utils.cpp
    src/runtime/executor/graph_executor_tensors.cpp
    src/runtime/executor/graph_executor_weight_cache.cpp
    src/runtime/executor/graph_executor_helpers.cpp
    src/runtime/executor/compiled_ops.cpp
    src/runtime/executor/compiled_ops_save.cpp
    src/runtime/executor/compiled_ops_tiled_mlp.cpp
    src/runtime/executor/compiled_ops_execute.cpp
    src/runtime/executor/compiled_ops_execute_forward.cpp
    src/runtime/executor/compiled_ops_execute_backward.cpp
    src/runtime/executor/op_registry.cpp
    src/runtime/executor/op_registrations.cpp
    src/runtime/dsl/graph_compiler.cpp
    src/runtime/dsl/weight_mapping.cpp
    src/runtime/dsl/dsl_weight_loader.cpp
    src/runtime/dsl/autodiff.cpp
    src/runtime/executor/autodiff_registrations.cpp
    src/runtime/dsl/op_shape_signatures.cpp
    src/runtime/dsl/tensor_slot_registry.cpp
    src/runtime/dsl/tensor_slot_dispatch.cpp
    src/runtime/dsl/tensor_role.cpp
    src/runtime/dsl/fusion_rule_registry.cpp
    src/runtime/dsl/fusion_rule_registrations.cpp
    src/runtime/dsl/hook_registry.cpp

    src/runtime/ops/embedding.cpp
    src/runtime/ops/zeros.cpp
    src/runtime/ops/ones.cpp
    src/runtime/ops/fused_residual_rmsnorm.cpp
    src/runtime/ops/view.cpp
    src/runtime/ops/transpose.cpp
    src/runtime/ops/concat_split.cpp
    src/runtime/ops/narrow.cpp
    src/runtime/ops/add.cpp
    src/runtime/ops/bias_add.cpp
    src/runtime/ops/matmul.cpp
    src/runtime/ops/swiglu.cpp
    src/runtime/ops/gelu_glu.cpp
    src/runtime/ops/matmul_swiglu.cpp
    src/runtime/ops/qkv_qk_norm.cpp
    src/runtime/ops/qkv_qk_norm_rope.cpp
    src/runtime/ops/mrope.cpp
    src/runtime/ops/mask_scatter.cpp
    src/runtime/ops/deepstack_inject.cpp
    src/runtime/ops/gelu.cpp
    src/runtime/ops/layernorm.cpp
    src/runtime/ops/rope.cpp
    src/runtime/attention/flash_attention.cpp
    src/runtime/attention/backend_registry.cpp
    src/runtime/attention/backend_cudnn.cpp
    src/runtime/attention/backend_flash_varlen.cpp
    src/runtime/attention/backend_custom.cpp
    src/runtime/attention/backend_mem_eff.cpp
    src/runtime/attention/attention_cudnn.cpp
    src/runtime/attention/flash_attn_varlen.cpp
    src/runtime/attention/flash_attn_scatter.cu
    src/runtime/attention/attention_custom.cu
    src/runtime/ops/cross_entropy_loss.cpp
    src/runtime/ops/fused_lm_head_loss.cpp
    src/runtime/ops/silu.cpp
    src/runtime/ops/relu2.cpp
    src/runtime/ops/mul.cpp
    src/runtime/ops/scale.cpp
    src/runtime/ops/rmsnorm.cpp
    src/runtime/ops/moe_softmax.cpp
    src/runtime/ops/moe_sigmoid.cpp
    src/runtime/ops/moe_topk.cpp
    src/runtime/ops/moe_permute.cpp
    src/runtime/ops/moe_grouped_gemm.cpp
    src/runtime/ops/moe_grouped_gemm_gate_up.cpp
    src/runtime/ops/moe_grouped_gemm_down.cpp
    src/runtime/ops/moe_expert_bias_add.cpp
    src/runtime/ops/gpt_oss_moe_act.cpp
    src/runtime/ops/moe_unpermute.cpp
    src/runtime/ops/ep_dispatch.cpp
    src/runtime/ops/ep_combine.cpp
    src/runtime/ep/lpt_planner.cpp
    src/runtime/ep/weight_transfer.cpp
    src/runtime/ep/ep_state.cpp
    src/runtime/ep/ep_buffer_pool.cpp
    src/runtime/ep/ep_strategy.cpp

    src/runtime/ops/mamba_split_proj.cpp
    src/runtime/ops/mamba_conv1d.cpp
    src/runtime/ops/mamba_split_conv_out.cpp
    src/runtime/ops/mamba_ssm_scan.cpp
    src/runtime/ops/mamba_gated_rmsnorm.cpp
    src/runtime/ops/mamba_out_proj.cpp
    src/runtime/ops/gated_delta_rule.cpp
    src/runtime/ops/qwen3_5_decay.cpp
    src/runtime/ops/repeat_interleave_heads.cpp

    src/config/pretrained_config.cpp

    src/runtime/qlora/generic_quantizer.cpp
    src/runtime/qlora/bnb_quantizer.cpp
    src/runtime/qlora/fp8_quantizer.cpp
    src/runtime/qlora/fp4_quantizer.cpp
    src/runtime/qlora/mxfp4_quantizer.cpp
    src/runtime/qlora/offload_manager.cpp
    src/runtime/qlora/generic_weight_manager.cpp
    src/runtime/qlora/dsl_qlora_pipeline.cpp
    src/runtime/qlora/generic_qlora_provider.cpp
    src/runtime/qlora/adapter_merger.cpp

    src/runtime/lora/lora_weights_manager.cpp
    src/runtime/lora/lora_grads_manager.cpp
    src/runtime/lora/lora_optimizer_state.cpp
    src/runtime/lora/lora_utils.cpp

    # Tokenizer (native C++ BPE, loads HuggingFace tokenizer.json)
    src/tokenizer/tokenizer.cpp
    src/tokenizer/unicode.cpp
    src/tokenizer/unicode_data.cpp

    # JIT kernel loader (Triton, CuTe, etc.)
    src/runtime/jit/jit_kernel.cpp
    src/runtime/jit/gated_delta_rule_kernels.cpp

    # Recipe system
    src/recipes/recipe.cpp
    src/recipes/recipe_factory.cpp
    src/recipes/fp8_hybrid/fp8_hybrid_recipe.cpp
    src/recipes/nvfp4/nvfp4_recipe.cpp

    # NVFP4-Quartet recipe (Quartet-II EDEN quantization)
    src/recipes/nvfp4_quartet/nvfp4_quartet_recipe.cpp
    src/recipes/nvfp4_quartet/kernels/hadamard_init.cu
    src/recipes/nvfp4_quartet/kernels/group_transform_128.cu
    src/recipes/nvfp4_quartet/kernels/group_transform_eden.cu
    src/recipes/nvfp4_quartet/kernels/eden_fp4.cu
    src/recipes/nvfp4_quartet/kernels/dequant_transform_quant.cu

    # Shared kernels (used by all recipes)
    src/kernels/kernels.cpp
    src/kernels/encoder.cu
    src/kernels/matmul.cpp
    src/kernels/swiglu.cu
    src/kernels/gelu_glu.cu
    src/kernels/activation.cu
    src/kernels/fast_lora_silu.cu
    src/kernels/moe/moe_backward_dispatch.cu
    src/kernels/moe/moe_diagnostics.cu
    src/kernels/moe/moe_dispatch_permute.cu
    src/kernels/moe/moe_expert_bias.cu
    src/kernels/moe/moe_fp8_grouped_gemm.cu
    src/kernels/moe/moe_gpt_oss_activation.cu
    src/kernels/moe/moe_grouped_gemm_backward.cu
    src/kernels/moe/moe_grouped_gemm_core.cu
    src/kernels/moe/moe_grouped_gemm_expert.cu
    src/kernels/moe/moe_losses.cu
    src/kernels/moe/moe_routing_activation.cu
    src/kernels/moe/moe_topk.cu
    src/kernels/ep_dispatch.cu
    src/kernels/mamba_kernels.cu
    src/kernels/rope.cu
    src/kernels/mrope.cu
    src/kernels/mask_scatter.cu
    src/kernels/rmsnorm.cu
    src/kernels/qk_norm.cu
    src/kernels/moe_cudnn.cpp
    src/kernels/block_quant.cu
    src/kernels/mxfp4_dequant.cu
    src/kernels/bnb_quant.cu
    src/kernels/global_norm.cu
    src/kernels/bias.cu
    src/runtime/optimizers/adamw_8bit.cu
    src/runtime/optimizers/flash_adamw_8bit.cu
    src/runtime/optimizers/adamw.cu
    src/runtime/optimizers/cpu_adamw.cpp
    src/runtime/optimizers/polar_express.cu
    src/runtime/optimizers/normuon.cu
    src/runtime/optimizers/optimizer.cpp
    src/runtime/optimizers/adamw_optimizer.cpp
    src/runtime/optimizers/adamw_8bit_optimizer.cpp
    src/runtime/optimizers/normuon_optimizer.cpp
    src/kernels/fused_classifier.cu
    src/kernels/transpose.cu
    src/kernels/vector_add.cu
    src/kernels/slice_add.cu
    src/kernels/random.cu
    src/kernels/fill.cu
    src/kernels/convert.cu
    src/kernels/lora_dropout.cu
    src/kernels/lora_small_rank.cu

    # Vendored Mamba2 kernels (selective scan + causal conv1d)
    src/third_party/mamba/selective_scan/selective_scan_fwd_bf16.cu
    src/third_party/mamba/selective_scan/selective_scan_fwd_fp16.cu
    src/third_party/mamba/selective_scan/selective_scan_fwd_fp32.cu
    src/third_party/mamba/selective_scan/selective_scan_bwd_bf16_real.cu
    src/third_party/mamba/selective_scan/selective_scan_bwd_fp16_real.cu
    src/third_party/mamba/selective_scan/selective_scan_bwd_fp32_real.cu
    src/third_party/causal_conv1d/causal_conv1d_fwd.cu
    src/third_party/causal_conv1d/causal_conv1d_bwd.cu
    src/third_party/causal_conv1d/causal_conv1d_update.cu

    # FP8 Hybrid recipe kernels
    src/recipes/fp8_hybrid/kernels/quant.cu
    src/recipes/fp8_hybrid/kernels/delayed_scaling.cu

    # NVFP4 recipe kernels (cuDNN and CUTLASS backends)
    src/recipes/nvfp4/kernels/quant_fp4.cu
    src/recipes/nvfp4/kernels/hadamard.cu
    src/recipes/nvfp4/kernels/matmul_fp4_cudnn.cpp
    src/recipes/nvfp4/kernels/quant_nvfp4_cutlass.cu
    src/recipes/nvfp4/kernels/quant_nvfp4_4o6.cu
    src/recipes/nvfp4/kernels/quant_nvfp4_4o6_sm100.cu
    src/recipes/nvfp4/kernels/matmul_cutlass_fp4.cu
    src/recipes/nvfp4/kernels/matmul_cutlass_fp4_sm100.cu
    src/recipes/nvfp4/kernels/matmul_cutlass_fp4_sm103.cu
    src/recipes/nvfp4/kernels/matmul_cutlass_fp4_sm120.cu
)

# Flash Attention varlen kernels (from DAO-AiLab/flash-attention)
# Used for document-level attention masking in GRPO training
set(FLASH_ATTN_DIR ${flash_attn_SOURCE_DIR}/csrc/flash_attn)
set(FLASH_ATTN_SOURCES
    # Is_causal=true instantiations (full causal attention)
    ${FLASH_ATTN_DIR}/src/flash_fwd_hdim64_bf16_causal_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_fwd_hdim96_bf16_causal_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_fwd_hdim128_bf16_causal_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_fwd_hdim256_bf16_causal_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_bwd_hdim64_bf16_causal_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_bwd_hdim96_bf16_causal_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_bwd_hdim128_bf16_causal_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_bwd_hdim256_bf16_causal_sm80.cu
    # Is_causal=false instantiations (non-causal and local/sliding-window path).
    # Flash-attn's Is_local kernel path requires Is_causal=false; sliding-window
    # attention (Gemma4 sliding blocks, Mistral-style SWA) selects this path.
    ${FLASH_ATTN_DIR}/src/flash_fwd_hdim64_bf16_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_fwd_hdim96_bf16_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_fwd_hdim128_bf16_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_fwd_hdim256_bf16_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_bwd_hdim64_bf16_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_bwd_hdim96_bf16_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_bwd_hdim128_bf16_sm80.cu
    ${FLASH_ATTN_DIR}/src/flash_bwd_hdim256_bf16_sm80.cu
)
add_library(flash_attn STATIC ${FLASH_ATTN_SOURCES})
set_target_properties(flash_attn PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(flash_attn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shims        # ATen shim for PhiloxCudaState
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party  # c10/cuda/CUDAException.h shim
    ${FLASH_ATTN_DIR}/src
    ${cutlass_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/tools/util/include
)
target_compile_definitions(flash_attn PRIVATE FLASH_NAMESPACE=surogate_flash)
target_compile_options(flash_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--threads=${SUROGATE_NVCC_THREADS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Mem-efficient attention (ported from PyTorch aten/src/ATen/native/
# transformers/cuda/mem_eff_attention, originally xformers). Capture-safe
# varlen attention with unbounded head_dim (via kMaxK=65536 GMEM path).
# Replaces the SDPA per-doc slicing backend for Hs > 256 layers that
# flash-varlen's 256 head-dim cap rejects.
set(MEM_EFF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/attention/mem_eff)
set(MEM_EFF_SOURCES
    # BF16 forward: the single auto-generated file provides three block
    # variants (64x64_rf for Hs<=64, 64x128_rf for Hs<=128, 32x128_gmem
    # for unlimited). Prototype wiring uses the GMEM variant only; the
    # rf variants link cleanly alongside but remain unused until the
    # dispatcher picks them as a fast path.
    ${MEM_EFF_DIR}/kernels/cutlassF_bf16_aligned.cu
    ${MEM_EFF_DIR}/mem_eff_dispatch.cu
    # BF16 backward: k65536 variant (128x64 and 64x64 blocks, unbounded K).
    ${MEM_EFF_DIR}/kernels/cutlassB_bf16_aligned_k65536.cu
    ${MEM_EFF_DIR}/mem_eff_backward.cu
    ${MEM_EFF_DIR}/mem_eff_delta.cu
    ${MEM_EFF_DIR}/mem_eff_mqa_reduce.cu
    ${MEM_EFF_DIR}/mem_eff_lse_xform.cu
)
add_library(mem_eff STATIC ${MEM_EFF_SOURCES})
set_target_properties(mem_eff PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mem_eff PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
    ${cutlass_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/tools/util/include
)
target_compile_options(mem_eff PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--threads=${SUROGATE_NVCC_THREADS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_link_libraries(surogate-common PRIVATE flash_attn mem_eff ${CUFILE_LIBS} cudnn_frontend ${CUDNN_LIBRARY} nvidia::nccl fmt::fmt-header-only)
target_link_libraries(surogate-common PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cublasLt ${NVML_LIBS} nlohmann_json::nlohmann_json)
target_link_libraries(surogate-common PUBLIC ${MULTI_GPU_DEPS})
target_compile_definitions(surogate-common PUBLIC ${MULTI_GPU_DEFS})
target_include_directories(surogate-common PRIVATE ${CUDNN_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src/shims ${FLASH_ATTN_DIR}/src)
target_include_directories(surogate-common PUBLIC src src/runtime src/third_party ${cutlass_SOURCE_DIR}/include ${cutlass_SOURCE_DIR}/tools/util/include ${backward_SOURCE_DIR} ${minja_SOURCE_DIR}/include)

# backward-cpp for stack traces
if(BACKWARD_HAS_DW)
    target_compile_definitions(surogate-common PRIVATE BACKWARD_HAS_DW=1)
elseif(BACKWARD_HAS_DWARF)
    target_compile_definitions(surogate-common PRIVATE BACKWARD_HAS_DWARF=1)
endif()
if(BACKWARD_LIBS)
    target_link_libraries(surogate-common PRIVATE ${BACKWARD_LIBS})
endif()
# -rdynamic exports symbols to the dynamic symbol table, enabling backtrace_symbols to resolve names
target_link_options(surogate-common PUBLIC "-rdynamic")

set(SUROGATE_NVCC_THREADS $ENV{NVCC_THREADS})
if(NOT SUROGATE_NVCC_THREADS)
    set(SUROGATE_NVCC_THREADS 8)
endif()

# Always include debug info for stack traces (even in Release builds)
# -g adds debug symbols, -lineinfo adds CUDA line info for GPU crashes
# -fno-omit-frame-pointer preserves frame pointers for stack unwinding
target_compile_options(surogate-common PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-g>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-g>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--threads=${SUROGATE_NVCC_THREADS}>)

# AddressSanitizer and UndefinedBehaviorSanitizer (C++ host code only, not CUDA)
if(ENABLE_SANITIZERS)
    message(STATUS "Sanitizers enabled (ASan + UBSan) for C++ files only")
    target_compile_options(surogate-common PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>
        $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=undefined>)
    target_link_options(surogate-common PUBLIC -fsanitize=address -fsanitize=undefined)
endif()

if (NOT SKBUILD)
    install(TARGETS surogate-common
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
    )
endif ()


if (PYTHON_BINDING)
    find_package(Python COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)
    FetchContent_MakeAvailable(nanobind)
    nanobind_add_module(_surogate STABLE_ABI NOSTRIP src/binding/binding.cpp src/binding/py_train.cpp)
    target_link_libraries(_surogate PUBLIC surogate-common Python::Module fmt::fmt-header-only)
    # mem_eff is a static archive; binding.cpp directly references
    # surogate::mem_eff::forward_bf16_sm80, so the module itself must
    # pull the symbol in (surogate-common doesn't call it and thus
    # doesn't pull the archive objects).
    target_link_libraries(_surogate PRIVATE mem_eff)

    # pick up cuda libraries from within the current python env, if available
    set(RPATH_LIST
        "$ORIGIN"
        "$ORIGIN/../nvidia/nccl/lib"
        "$ORIGIN/../nvidia/cudnn/lib"
        "$ORIGIN/../nvidia/cuda_runtime/lib"
        "$ORIGIN/../nvidia/cublas/lib"
    )
    list(JOIN RPATH_LIST ":" WHEEL_RPATH)

    set_target_properties(_surogate surogate-common PROPERTIES
            INSTALL_RPATH "${WHEEL_RPATH}"
            BUILD_WITH_INSTALL_RPATH OFF
            INSTALL_RPATH_USE_LINK_PATH FALSE
    )

    install(TARGETS _surogate surogate-common
            LIBRARY DESTINATION surogate
    )
endif ()


# ----------------------------------------------------------------------------
# Tests
option(BUILD_TESTS "Build all tests (unit + integration)" OFF)
option(BUILD_TESTS_UNIT "Build unit tests (kernels, modules, components)" ON)
option(BUILD_TESTS_INTEGRATION "Build integration tests (training loops, distributed)" OFF)

if (BUILD_TESTS OR BUILD_TESTS_UNIT OR BUILD_TESTS_INTEGRATION)
    enable_testing()

    # Unit tests: Kernels, quantization, optimizers, LoRA, MoE, models
    if (BUILD_TESTS OR BUILD_TESTS_UNIT)
        add_executable(unit-tests
            src/testing/utilities/test_main.cpp

            # Kernels
            src/testing/kernels/test-swiglu.cu
            src/testing/kernels/test-rope.cu
            src/testing/kernels/test-classifier.cu
            src/testing/kernels/test-grad-clip.cu
            src/testing/kernels/test-multi-tensor-norm.cu


            # Quantization
            src/testing/quantization/test-qlora.cu
            src/testing/quantization/test-fp4.cu
            src/testing/quantization/test-bnb-quant.cpp

            # Optimizers
            src/testing/optimizers/test-adamw.cu
            src/testing/optimizers/test-flash-adamw.cu
            src/testing/optimizers/test-normuon.cu

            # LoRA
            src/testing/lora/test-lora-dropout.cu
            src/testing/lora/test-fast-lora-silu.cu

            # MoE
            src/testing/moe/test-moe.cu

            # Expert Parallelism
            src/testing/ep/test-ep-buffer-pool.cpp

            # JIT kernel loader
            src/testing/jit/test-jit-kernel.cpp

            # Tokenizer
            src/testing/tokenizer/test_tokenizer.cpp

            # Utilities
            src/testing/utilities/test-sol.cpp
            src/testing/utilities/test_dsl_ir.cpp
            src/testing/utilities/test_dsl_goldens.cpp
            src/testing/utilities/test_dsl_module_goldens.cpp
            src/testing/utilities/test_fusion_rule_registry.cpp
            src/testing/utilities/test_op_registry.cpp
            src/testing/utilities/test_tensor_role.cpp
        )
        target_link_libraries(unit-tests PRIVATE surogate-common Catch2::Catch2 CLI11::CLI11 fmt::fmt-header-only)
        target_compile_options(unit-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        )
        add_test(NAME unit-tests COMMAND unit-tests)
    endif()

    # Integration tests: Full training loops, multi-GPU communication
    if (BUILD_TESTS OR BUILD_TESTS_INTEGRATION)
        add_executable(integration-tests
            src/testing/utilities/test_main.cpp

            # Training
            # Distributed
            src/testing/distributed/test-nccl.cpp
        )
        target_link_libraries(integration-tests PRIVATE
            surogate-common Catch2::Catch2 CLI11::CLI11 fmt::fmt-header-only
        )
        target_compile_options(integration-tests PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        )
        add_test(NAME integration-tests COMMAND integration-tests)
    endif()

    # Convenience target for all tests
    if (BUILD_TESTS)
        add_custom_target(all-tests DEPENDS unit-tests integration-tests)
    endif()
endif()
