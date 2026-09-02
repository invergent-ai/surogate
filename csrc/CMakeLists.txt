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

# cuFile (GPUDirect Storage) is on by default and a missing cuFile is a build error:
# hosts where GDS does not work (WSL, AMD, FUSE/network checkpoint storage, no nvidia-fs)
# are handled at *runtime* by falling back to buffered reads, so they are not a reason to
# build without it. Turn USE_CUFILE off only for a toolchain that has no cuFile headers
# or library at all.
option(USE_CUFILE "Enable cuFile support for optimized file I/O" ON)
option(REQUIRE_CUFILE "Fail the build if USE_CUFILE is on but cuFile is not found" ON)
option(USE_NVML "Enable nvml support for detailed GPU stats reporting" ON)
option(PYTHON_BINDING "Build python bindings" ON)
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# Serve engine (csrc/src/serve; third-party provenance in src/serve/NOTICE):
# first-class targets in this project, EXCLUDE_FROM_ALL so `make build` cost
# is unchanged. Build with e.g. `cmake --build <dir> --target
# surogate-engine-cli`. Deps (json, cutlass) come from this file's
# FetchContent; serve targets are pinned to sm_120a.
option(SUROGATE_BUILD_SERVE "Expose the serve engine targets" ON)
option(SUROGATE_SERVE_TESTS "Expose the serve engine test targets" OFF)
option(SUROGATE_SERVE_BENCH "Expose the serve engine bench targets" OFF)
set(SUROGATE_SERVE_CUDA_ARCHS "120a" CACHE STRING
    "CUDA architectures for the serve engine targets (RTX consumer set)")

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

# The Gemma tokenizer. The checkpoint ships tokenizer.model beside
# tokenizer.json, and upstream SentencePiece reproduces HF's ids exactly on it
# (verified over 414 cases: whitespace runs, emoji, CJK, Cyrillic, long inputs
# and 400 random mixed-alphabet strings). Hand-writing a 514,906-merge BPE with
# byte fallback to match would be strictly worse and far easier to get subtly
# wrong.
FetchContent_Declare(
        sentencepiece
        QUIET
        GIT_REPOSITORY https://github.com/google/sentencepiece.git
        GIT_TAG v0.2.1
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

# CUTLASS is used by the NVFP4 (block-scaled) GEMMs and MoE grouped GEMMs. The FP8-hybrid
# recipe does NOT use it -- those matmuls go through cuBLASLt -- so a CUTLASS bump cannot
# move FP8 numbers. Measured on the pool-D 27B replay, fp8 with attention-projection
# quantization held OFF: 4.4.1 = 7.413 s/mb, 4.6.1 = 7.382 s/mb, i.e. -0.4% = noise.
# 4.6.1 is kept for correctness, not speed: 4.5.1 "added missing MXFP8MMAOP and
# MXF8F6F4MMAOP for sm120" plus further SM120 block-scaled fixes.
FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v4.6.1
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

set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sentencepiece)

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
    elseif(REQUIRE_CUFILE)
        message(FATAL_ERROR
            "cuFile required but not found. Install the cuFile development package "
            "(e.g. libcufile-dev-13-1) or configure with -DREQUIRE_CUFILE=OFF to build "
            "without GPUDirect Storage.")
    else()
        set(USE_CUFILE 0)
        message(WARNING "cuFile requested but not found, disabling cuFile support")
    endif()
endif ()

if(NOT USE_CUFILE)
    set(CUFILE_SOURCE src/utilities/cu_file_fallback.cpp)
    set(CUFILE_LIBS "")
endif ()

# The buffered read path is compiled either way: the cuFile build falls back to it at
# runtime when GPUDirect Storage is unusable (network/FUSE-backed checkpoint storage).
list(APPEND CUFILE_SOURCE src/utilities/cu_file_common.cpp)

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
# pip's nvidia-nccl ships only the runtime libnccl.so.2 (no dev symlink), so accept it by name.
find_library(NCCL_LIBRARIES NAMES libnccl.so.2 nccl NAMES_PER_DIR HINTS ${NCCL_LIB_DIR} REQUIRED)
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
    src/runtime/dsl/per_layer_dims.cpp
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
    src/runtime/dsl/shared_master_store.cpp
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
    src/runtime/attention/backend_kvprefix.cpp
    src/runtime/attention/backend_custom.cpp
    src/runtime/attention/backend_mem_eff.cpp
    src/runtime/attention/attention_cudnn.cpp
    src/runtime/attention/flash_attn_varlen.cpp
    src/runtime/attention/flash_attn_scatter.cu
    src/runtime/attention/attention_custom.cu
    src/runtime/ops/cross_entropy_loss.cpp
    src/runtime/ops/fused_lm_head_loss.cpp
    src/runtime/ops/silu.cpp
    src/runtime/ops/softplus.cpp
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
    src/tokenizer/tok_profile.cpp
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
    src/runtime/optimizers/async_optimizer.cpp
    src/runtime/optimizers/polar_express.cu
    src/runtime/optimizers/normuon.cu
    src/runtime/optimizers/optimizer.cpp
    src/runtime/optimizers/adamw_optimizer.cpp
    src/runtime/optimizers/adamw_8bit_optimizer.cpp
    src/runtime/optimizers/normuon_optimizer.cpp
    src/kernels/fused_classifier.cu
    src/kernels/lmhead_compact.cu
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
    nanobind_add_module(_surogate STABLE_ABI NOSTRIP src/binding/binding.cpp src/binding/py_train.cpp
                        src/runtime/executor/dispatch_pp_phase0.cpp)
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
option(BUILD_TESTS "Build integration tests" OFF)
option(BUILD_TESTS_INTEGRATION "Build integration tests (training loops, distributed)" OFF)

if (BUILD_TESTS OR BUILD_TESTS_INTEGRATION)
    enable_testing()

    add_executable(integration-tests
        src/testing/utilities/test_main.cpp
        src/testing/distributed/test-nccl.cpp
        src/testing/kernels/test-dpo-dloss.cu
        src/testing/kernels/test-chunked-attention.cu
        src/testing/tokenizer/test-chat-template.cpp
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

if(SUROGATE_BUILD_SERVE)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
        message(STATUS "serve: requires CUDA >= 13.1; targets disabled")
    else()
        enable_language(C)  # utf8proc
        set(SINFER_ROOT ${CMAKE_SOURCE_DIR}/src/serve)
        set(SUROGATE_VENDOR_DIR ${CMAKE_SOURCE_DIR}/src/third_party)
        set(SINFER_JSON_INCLUDE "${json_SOURCE_DIR}/single_include")
        set(SINFER_BUILD_MEDIA_ACQUIRE ON)
        set(SINFER_BUILD_PROMPT_INPUT ON)
        set(SINFER_BUILD_SERVE ON)
        # Host deps are best-effort: a training-only machine without
        # pkg-config/libcurl must still configure — serve targets drop out
        # with a status message instead of failing the build.
        find_package(PkgConfig QUIET)
        option(SINFER_ENABLE_FFMPEG "Serve: build media decoding against FFmpeg" ON)
        if(SINFER_ENABLE_FFMPEG AND PkgConfig_FOUND)
            pkg_check_modules(FFMPEG IMPORTED_TARGET
                libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
            if(NOT FFMPEG_FOUND)
                message(STATUS "serve: FFmpeg dev libraries not found; media-decode stub")
                set(SINFER_ENABLE_FFMPEG OFF)
            endif()
        else()
            set(SINFER_ENABLE_FFMPEG OFF)
        endif()
        if(PkgConfig_FOUND)
            pkg_check_modules(LIBCURL IMPORTED_TARGET libcurl>=7.85)
        endif()
        if(NOT LIBCURL_FOUND)
            message(STATUS "serve: pkg-config/libcurl not found; serve targets disabled")
        endif()
        set(SERVE_ROOT ${CMAKE_SOURCE_DIR}/src/serve)
        if(LIBCURL_FOUND)

        # ---- serve engine libraries (formerly src/serve/src/CMakeLists.txt) ----
function(sinfer_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${SERVE_ROOT}
    ${SUROGATE_VENDOR_DIR}
    ${SUROGATE_VENDOR_DIR}/utf8proc
    ${SINFER_JSON_INCLUDE})
  # Serve targets build for the RTX arch set (default 120a; sm_89 compiles
  # via the upstream port guards, execution pending Ada hardware). Features
  # above a device's capability gate off at runtime (FP8 plane needs sm_89+,
  # the fp4 profile needs sm_120a). PIC everywhere so the archives can link
  # into the _surogate_serve Python module as well as the executables.
  set_target_properties(${target} PROPERTIES
      CUDA_ARCHITECTURES "${SUROGATE_SERVE_CUDA_ARCHS}"
      POSITION_INDEPENDENT_CODE ON)
endfunction()

function(sinfer_cuda_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# Device/tensor/layout/graph primitives and the model-independent request runtime.
add_library(sinfer_core STATIC EXCLUDE_FROM_ALL
  ${SERVE_ROOT}/core/arena.cu
  ${SERVE_ROOT}/core/sleep.cu
  ${SERVE_ROOT}/core/elastic_kv_region.cu
  ${SERVE_ROOT}/core/engine_context.cpp
  ${SERVE_ROOT}/core/device.cu
  ${SERVE_ROOT}/core/device_footprint.cpp
  ${SERVE_ROOT}/core/dtype.cpp
  ${SERVE_ROOT}/core/host_worker_pool.cpp
  ${SERVE_ROOT}/core/numa.cpp
  ${SERVE_ROOT}/core/gdn_replay_records.cpp
  ${SERVE_ROOT}/core/cyclic_kv_cache.cpp
  ${SERVE_ROOT}/core/paged_kv_cache.cpp
  ${SERVE_ROOT}/core/linear_attention_state.cpp
  ${SERVE_ROOT}/core/ngram_ple_state.cpp
  ${SERVE_ROOT}/core/layout.cpp
  ${SERVE_ROOT}/core/tensor.cpp
  ${SERVE_ROOT}/core/decode_graph.cpp
  ${SERVE_ROOT}/runtime/engine/admission_policy.cpp
  ${SERVE_ROOT}/runtime/engine/kv_capacity.cpp
  ${SERVE_ROOT}/runtime/engine/public_types.cpp
  ${SERVE_ROOT}/runtime/engine/request_memory.cpp)
sinfer_internal_includes(sinfer_core)
sinfer_cuda_archive(sinfer_core)
target_link_libraries(sinfer_core PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvtx3 Threads::Threads ${CMAKE_DL_LIBS})
# libnuma places the host buffers of an offloaded model (core/numa.*). Optional: without it the
# engine warns and the operator must run under `numactl --interleave=all` on a multi-socket host.
find_library(NUMA_LIBRARY NAMES numa)
if(NUMA_LIBRARY)
  target_link_libraries(sinfer_core PUBLIC ${NUMA_LIBRARY})
  message(STATUS "libnuma: ${NUMA_LIBRARY} (host buffer placement enabled)")
else()
  message(STATUS "libnuma not found: host buffer placement disabled; run multi-socket hosts under numactl --interleave=all")
endif()

# Generic .sinfer framing, binding, and direct final materialization.
add_library(sinfer_artifact STATIC EXCLUDE_FROM_ALL
  ${SERVE_ROOT}/artifact/reader.cpp
  ${SERVE_ROOT}/artifact/storage_layouts.cpp
  ${SERVE_ROOT}/artifact/binder.cpp
  ${SERVE_ROOT}/artifact/materializer.cpp
  ${SERVE_ROOT}/artifact/typed_binding.cpp)
sinfer_internal_includes(sinfer_artifact)
target_link_libraries(sinfer_artifact PUBLIC sinfer_core)

# The warp-specialized SM120 kernel uses setmaxnreg to transfer registers from its
# producer warpgroup to its consumer warpgroups. Relocatable device code makes ptxas
# discard that contract, so this self-contained kernel is intentionally compiled in a
# non-RDC archive and called through a host launcher from sinfer_ops.
# surogate vendor patch (csrc/src/serve/PATCHES.md): the W4A4 TMA kernels are
# sm_120-only hardware (TMA + cluster + block-scaled FP4 MMA). Port builds for
# other architectures link a loud-failure stub instead.
if("120a" IN_LIST SUROGATE_SERVE_CUDA_ARCHS OR SUROGATE_SERVE_CUDA_ARCHS STREQUAL "120a")
  add_library(sinfer_nvfp4_tma STATIC EXCLUDE_FROM_ALL
    ${SERVE_ROOT}/ops/linear/nvfp4/nvfp4_w4a4_tma.cu
    ${SERVE_ROOT}/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu)
else()
  add_library(sinfer_nvfp4_tma STATIC EXCLUDE_FROM_ALL
    ${SERVE_ROOT}/ops/linear/nvfp4/nvfp4_w4a4_tma_stub.cpp)
endif()
sinfer_internal_includes(sinfer_nvfp4_tma)
set_target_properties(sinfer_nvfp4_tma PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_compile_options(sinfer_nvfp4_tma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(sinfer_nvfp4_tma PRIVATE sinfer_core CUDA::cudart CUDA::cuda_driver)

# TensorRT-LLM's CUTLASS fused-MoE runner, vendored from FlashInfer 0.6.16.post3
# (src/third_party/trtllm_moe, Apache-2.0; NOTICE there records provenance). It is the kernel
# vLLM serves NVFP4 routed experts through on this hardware (design/35B_VLLM_KERNELS.md), built
# with FlashInfer's own flag set so the instantiations it generated for architecture 120 apply
# unchanged. sm_120-only, non-RDC, like the W4A4 TMA archive above; other architectures link a
# loud-failure stub.
set(TRTLLM_MOE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/trtllm_moe)
if("120a" IN_LIST SUROGATE_SERVE_CUDA_ARCHS OR SUROGATE_SERVE_CUDA_ARCHS STREQUAL "120a")
  file(GLOB TRTLLM_MOE_GENERATED ${TRTLLM_MOE_ROOT}/cutlass_instantiations/120/gemm_grouped/120/*.generated.cu)
  add_library(sinfer_trtllm_moe STATIC EXCLUDE_FROM_ALL
    ${TRTLLM_MOE_ROOT}/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp4_fp4.cu
    ${TRTLLM_MOE_ROOT}/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu
    ${TRTLLM_MOE_ROOT}/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_mixed_utils.cu
    ${TRTLLM_MOE_ROOT}/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp
    # TensorRT-LLM's own error, logging and formatting runtime: the kernels' TLLM_CHECK macros
    # raise and log through it, so the library does not link without these four.
    ${TRTLLM_MOE_ROOT}/nv_internal/cpp/common/tllmException.cpp
    ${TRTLLM_MOE_ROOT}/nv_internal/cpp/common/logger.cpp
    ${TRTLLM_MOE_ROOT}/nv_internal/cpp/common/stringUtils.cpp
    ${TRTLLM_MOE_ROOT}/nv_internal/cpp/common/envUtils.cpp
    # The runner's constructor builds a DeepSeek block-scale GEMM runner and its LoRA hook
    # unconditionally, so both are needed at link even though the routed-NVFP4 profile takes
    # neither path.
    ${TRTLLM_MOE_ROOT}/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu
    ${TRTLLM_MOE_ROOT}/nv_internal/tensorrt_llm/kernels/lora/lora.cpp
    ${TRTLLM_MOE_ROOT}/fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu
    ${TRTLLM_MOE_GENERATED}
    ${SERVE_ROOT}/ops/sparse_moe/trtllm/trtllm_moe.cu)
  target_compile_definitions(sinfer_trtllm_moe PRIVATE
    COMPILE_BLACKWELL_TMA_GEMMS COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS
    ENABLE_BF16 ENABLE_FP8 ENABLE_FP4 USING_OSS_CUTLASS_MOE_GEMM CUTLASS_ENABLE_GDC_FOR_SM100=1
    FLASHINFER_ENABLE_BF16 FLASHINFER_ENABLE_F16 FLASHINFER_ENABLE_FP8_E4M3
    FLASHINFER_ENABLE_FP8_E5M2 FLASHINFER_ENABLE_FP8_E8M0 FLASHINFER_ENABLE_FP4_E2M1)
  target_compile_options(sinfer_trtllm_moe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>
    $<$<COMPILE_LANGUAGE:CUDA>:--threads=${SUROGATE_NVCC_THREADS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
else()
  add_library(sinfer_trtllm_moe STATIC EXCLUDE_FROM_ALL
    ${SERVE_ROOT}/ops/sparse_moe/trtllm/trtllm_moe_stub.cpp)
endif()
sinfer_internal_includes(sinfer_trtllm_moe)
target_include_directories(sinfer_trtllm_moe PRIVATE
  ${TRTLLM_MOE_ROOT}
  ${TRTLLM_MOE_ROOT}/nv_internal
  ${TRTLLM_MOE_ROOT}/nv_internal/include
  ${TRTLLM_MOE_ROOT}/nv_internal/tensorrt_llm/cutlass_extensions/include
  ${TRTLLM_MOE_ROOT}/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include
  ${TRTLLM_MOE_ROOT}/nv_internal/tensorrt_llm/kernels/cutlass_kernels
  ${TRTLLM_MOE_ROOT}/cutlass_instantiations/120
  ${TRTLLM_MOE_ROOT}/flashinfer_include
  ${cutlass_SOURCE_DIR}/include
  ${cutlass_SOURCE_DIR}/tools/util/include
  # cutlass's host-compiled headers reach for libcudacxx; nvcc finds CCCL itself, g++ does not.
  ${CUDAToolkit_INCLUDE_DIRS}/cccl)
set_target_properties(sinfer_trtllm_moe PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  CUDA_STANDARD 17
  CXX_STANDARD 17)
target_link_libraries(sinfer_trtllm_moe PRIVATE sinfer_core CUDA::cudart CUDA::cuda_driver)

# Shared mathematical Ops. The list is intentionally explicit: adding a
# source is a build-boundary decision, not an accidental recursive-glob side effect.
add_library(sinfer_ops STATIC EXCLUDE_FROM_ALL
  ${SERVE_ROOT}/ops/launcher/add_bias.cu
  ${SERVE_ROOT}/ops/launcher/argmax.cu
  ${SERVE_ROOT}/ops/launcher/cast.cu
  ${SERVE_ROOT}/ops/launcher/causal_conv1d.cu
  ${SERVE_ROOT}/ops/launcher/embed_gather.cu
  ${SERVE_ROOT}/ops/launcher/encoder_attention.cu
  ${SERVE_ROOT}/ops/launcher/gdn_gating.cu
  # One translation unit per registered GQA shape, named for the shape. The
  # registry in ops/kernel/gqa_attention_geometry.cuh generates the dispatch and
  # the extern declarations; a registered shape missing its file here is an
  # undefined reference at link.
  ${SERVE_ROOT}/ops/launcher/gqa_attention_decode_gqa256_24q4.cu
  ${SERVE_ROOT}/ops/launcher/gqa_attention_decode_gqa256_16q2.cu
  ${SERVE_ROOT}/ops/launcher/gqa_attention_decode_gqa256_8q2.cu
  ${SERVE_ROOT}/ops/launcher/gqa_attention_decode_gqa256_16q4.cu
  ${SERVE_ROOT}/ops/launcher/gqa_attention_decode_gqa256_24q2.cu
  ${SERVE_ROOT}/ops/launcher/gqa_attention_decode_gqa128_16q8.cu
  ${SERVE_ROOT}/ops/launcher/gqa_attention_decode_gqa64_32q4.cu
  ${SERVE_ROOT}/ops/launcher/gqa_attention_decode_gqa256_4q1.cu
  ${SERVE_ROOT}/ops/launcher/gelu.cu
  ${SERVE_ROOT}/ops/launcher/lora_batched.cu
  ${SERVE_ROOT}/ops/launcher/lora_fused.cu
  ${SERVE_ROOT}/ops/launcher/gelu_and_mul.cu
  ${SERVE_ROOT}/ops/launcher/bidirectional_gqa_attention.cu
  ${SERVE_ROOT}/ops/launcher/swa.cu
  ${SERVE_ROOT}/ops/launcher/gqa_attention_decode.cu
  ${SERVE_ROOT}/ops/launcher/gqa_attention_prefill.cu
  ${SERVE_ROOT}/ops/launcher/kv_cache_append_prefix.cu
  ${SERVE_ROOT}/ops/launcher/l2norm.cu
  ${SERVE_ROOT}/ops/launcher/layer_norm.cu
  ${SERVE_ROOT}/ops/launcher/mtp_pack.cu
  ${SERVE_ROOT}/ops/launcher/mtp_round.cu
  ${SERVE_ROOT}/ops/launcher/speculative_round.cu
  ${SERVE_ROOT}/ops/launcher/position.cu
  ${SERVE_ROOT}/ops/launcher/mask_columns.cu
  ${SERVE_ROOT}/ops/launcher/prepare_ragged_prefix.cu
  ${SERVE_ROOT}/ops/launcher/prepare_masked_block.cu
  ${SERVE_ROOT}/ops/launcher/residual_add.cu
  ${SERVE_ROOT}/ops/launcher/rmsnorm.cu
  ${SERVE_ROOT}/ops/launcher/rope.cu
  ${SERVE_ROOT}/ops/launcher/sampling.cu
  ${SERVE_ROOT}/ops/launcher/scalar.cu
  ${SERVE_ROOT}/ops/launcher/scatter.cu
  ${SERVE_ROOT}/ops/launcher/sigmoid_gate_mul.cu
  ${SERVE_ROOT}/ops/launcher/mean_pool.cu
  ${SERVE_ROOT}/ops/launcher/scale.cu
  ${SERVE_ROOT}/ops/launcher/silu_and_mul.cu
  ${SERVE_ROOT}/ops/launcher/vision_attention.cu
  ${SERVE_ROOT}/ops/launcher/vision_pos_embed.cu
  # Gated DeltaNet Linear Attention core.
  ${SERVE_ROOT}/ops/linear_attention/gated_delta_net/gated_delta_net.cpp
  ${SERVE_ROOT}/ops/linear_attention/gated_delta_net/replay.cpp
  ${SERVE_ROOT}/ops/linear_attention/gated_delta_net/recurrent.cu
  ${SERVE_ROOT}/ops/linear_attention/gated_delta_net/chunked/launch.cu
  ${SERVE_ROOT}/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cu
  ${SERVE_ROOT}/ops/linear_attention/gated_delta_net/chunked/state_passing.cu
  ${SERVE_ROOT}/ops/linear_attention/gated_delta_net/chunked/output.cu
  ${SERVE_ROOT}/ops/attn_input_proj/bf16/bf16_attn_input_decode.cu
  ${SERVE_ROOT}/ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu
  ${SERVE_ROOT}/ops/attn_input_proj/bf16/bf16_attn_input_small_t.cu
  ${SERVE_ROOT}/ops/attn_input_proj/bf16/bf16_attn_input_plan.cpp
  ${SERVE_ROOT}/ops/attn_input_proj/fp8/fp8_attn_input_decode.cu
  ${SERVE_ROOT}/ops/attn_input_proj/fp8/fp8_attn_input_small_t.cu
  ${SERVE_ROOT}/ops/attn_input_proj/fp8/fp8_attn_input_a8.cu
  ${SERVE_ROOT}/ops/attn_input_proj/fp8/fp8_attn_input_plan.cpp
  ${SERVE_ROOT}/ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu
  ${SERVE_ROOT}/ops/attn_input_proj/nvfp4/nvfp4_attn_input_small_t.cu
  ${SERVE_ROOT}/ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu
  ${SERVE_ROOT}/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp
  ${SERVE_ROOT}/ops/attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu
  ${SERVE_ROOT}/ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu
  ${SERVE_ROOT}/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp
  ${SERVE_ROOT}/ops/attn_input_proj/w8/w8_attn_input_decode.cu
  ${SERVE_ROOT}/ops/attn_input_proj/w8/w8_attn_input_gemm_mma.cu
  ${SERVE_ROOT}/ops/attn_input_proj/w8/w8_attn_input_gemm_simt.cu
  ${SERVE_ROOT}/ops/attn_input_proj/w8/w8_attn_input_gemm_splitk.cu
  ${SERVE_ROOT}/ops/attn_input_proj/w8/w8_attn_input_plan.cpp
  ${SERVE_ROOT}/ops/gdn_input_proj/fp8/fp8_gdn_input_decode.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/fp8/fp8_gdn_input_small_t.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/fp8/fp8_gdn_input_plan.cpp
  ${SERVE_ROOT}/ops/gdn_input_proj/fp8/fp8_gdn_conv_fused.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp
  ${SERVE_ROOT}/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_small_t.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp
  ${SERVE_ROOT}/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp
  ${SERVE_ROOT}/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_gemm_mma.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp
  ${SERVE_ROOT}/ops/gdn_input_proj/gdn_projected_conv.cu
  ${SERVE_ROOT}/ops/linear/w8a8/w8a8_act_quant.cu
  ${SERVE_ROOT}/ops/linear/w8a8/w8a8_dispatch.cu
  ${SERVE_ROOT}/ops/linear/w8a8/w4fp4_cutlass_gemm.cu
  ${SERVE_ROOT}/ops/linear/w8a8/w4fp4_plane.cu
  ${SERVE_ROOT}/ops/linear/w8a8/w8fp8_plane.cu
  ${SERVE_ROOT}/ops/linear_swiglu/w8a8/w8a8_linear_swiglu.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/w8/w8_gdn_input_decode.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/w8/w8_gdn_input_gemm_mma.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu
  ${SERVE_ROOT}/ops/gdn_input_proj/w8/w8_gdn_input_plan.cpp
  ${SERVE_ROOT}/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu
  ${SERVE_ROOT}/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp
  ${SERVE_ROOT}/ops/linear/bf16/bf16_gemv.cu
  ${SERVE_ROOT}/ops/linear/bf16/bf16_gemm_mma.cu
  ${SERVE_ROOT}/ops/linear/bf16/bf16_small_t.cu
  ${SERVE_ROOT}/ops/linear/bf16/bf16_dispatch.cpp
  ${SERVE_ROOT}/ops/linear/fp8/fp8_format.cpp
  ${SERVE_ROOT}/ops/linear/fp8/fp8_gemv.cu
  ${SERVE_ROOT}/ops/linear/fp8/fp8_small_t.cu
  ${SERVE_ROOT}/ops/linear/fp8/fp8_a16_mma.cu
  ${SERVE_ROOT}/ops/linear/fp8/fp8_a8.cu
  ${SERVE_ROOT}/ops/linear/fp8/fp8_dispatch.cpp
  ${SERVE_ROOT}/ops/linear/fp8/fp8_cublaslt.cpp
  ${SERVE_ROOT}/ops/linear/fp8/fp8_cublaslt_finish.cu
  ${SERVE_ROOT}/ops/linear/nvfp4/nvfp4_format.cpp
  ${SERVE_ROOT}/ops/linear/nvfp4/nvfp4_gemv.cu
  ${SERVE_ROOT}/ops/linear/nvfp4/nvfp4_small_t.cu
  ${SERVE_ROOT}/ops/linear/nvfp4/nvfp4_w4a4.cu
  ${SERVE_ROOT}/ops/linear/nvfp4/nvfp4_dispatch.cpp
  ${SERVE_ROOT}/ops/linear/nvfp4/nvfp4_cublaslt.cpp
  ${SERVE_ROOT}/ops/linear/q4/q4_small_t_mma.cu
  ${SERVE_ROOT}/ops/linear/q4/q4_rowsplit_gemm_mma.cu
  ${SERVE_ROOT}/ops/linear/q4/q4_rowsplit_gemm_simt.cu
  ${SERVE_ROOT}/ops/linear/q4/q4_rowsplit_gemv.cu
  ${SERVE_ROOT}/ops/linear/q4/q4_dispatch.cpp
  ${SERVE_ROOT}/ops/linear/q5/q5_dispatch.cpp
  ${SERVE_ROOT}/ops/linear/q5/q5_rowsplit_gemm_mma.cu
  ${SERVE_ROOT}/ops/linear/q5/q5_rowsplit_gemm_simt.cu
  ${SERVE_ROOT}/ops/linear/q5/q5_rowsplit_gemv.cu
  ${SERVE_ROOT}/ops/linear_add/bf16/bf16_linear_add_decode.cu
  ${SERVE_ROOT}/ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu
  ${SERVE_ROOT}/ops/linear_add/bf16/bf16_linear_add_small_t.cu
  ${SERVE_ROOT}/ops/linear_add/bf16/bf16_linear_add_plan.cpp
  ${SERVE_ROOT}/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu
  ${SERVE_ROOT}/ops/linear_add/nvfp4/nvfp4_linear_add_small_t.cu
  ${SERVE_ROOT}/ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu
  ${SERVE_ROOT}/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp
  ${SERVE_ROOT}/ops/linear_add/fp8/fp8_linear_add_decode.cu
  ${SERVE_ROOT}/ops/linear_add/fp8/fp8_linear_add_small_t.cu
  ${SERVE_ROOT}/ops/linear_add/fp8/fp8_linear_add_a8.cu
  ${SERVE_ROOT}/ops/linear_add/fp8/fp8_linear_add_plan.cpp
  ${SERVE_ROOT}/ops/linear_add/q5/q5_linear_add_gemm_mma.cu
  ${SERVE_ROOT}/ops/linear_add/q5/q5_linear_add_gemm_simt.cu
  ${SERVE_ROOT}/ops/linear_add/q5/q5_linear_add_gemv.cu
  ${SERVE_ROOT}/ops/linear_add/q5/q5_linear_add_plan.cpp
  ${SERVE_ROOT}/ops/linear_add/w8/w8_linear_add_gemm_mma.cu
  ${SERVE_ROOT}/ops/linear_add/w8/w8_linear_add_gemm_simt.cu
  ${SERVE_ROOT}/ops/linear_add/w8/w8_linear_add_gemm_splitk.cu
  ${SERVE_ROOT}/ops/linear_add/w8/w8_linear_add_plan.cpp
  ${SERVE_ROOT}/ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cu
  ${SERVE_ROOT}/ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu
  ${SERVE_ROOT}/ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp
  ${SERVE_ROOT}/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu
  ${SERVE_ROOT}/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu
  ${SERVE_ROOT}/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu
  ${SERVE_ROOT}/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
  ${SERVE_ROOT}/ops/linear_swiglu/fp8/fp8_linear_swiglu_decode.cu
  ${SERVE_ROOT}/ops/linear_swiglu/fp8/fp8_linear_swiglu_small_t.cu
  ${SERVE_ROOT}/ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu
  ${SERVE_ROOT}/ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp
  ${SERVE_ROOT}/ops/linear_swiglu/w8/w8_linear_swiglu_decode.cu
  ${SERVE_ROOT}/ops/linear_swiglu/w8/w8_linear_swiglu_gemm_mma.cu
  ${SERVE_ROOT}/ops/linear_swiglu/w8/w8_linear_swiglu_gemm_splitk.cu
  ${SERVE_ROOT}/ops/linear_swiglu/w8/w8_linear_swiglu_plan.cpp
  ${SERVE_ROOT}/ops/linear/q6/q6_dispatch.cpp
  ${SERVE_ROOT}/ops/linear/q6/q6_rowsplit_gemm_mma.cu
  ${SERVE_ROOT}/ops/linear/q6/q6_rowsplit_gemm_simt.cu
  ${SERVE_ROOT}/ops/linear/w8/w8_rowsplit_gemm_mma.cu
  ${SERVE_ROOT}/ops/linear/w8/w8_rowsplit_gemm_simt.cu
  ${SERVE_ROOT}/ops/linear/w8/w8_rowsplit_gemm_decode.cu
  ${SERVE_ROOT}/ops/linear/marlin/marlin_gemm.cu
  ${SERVE_ROOT}/ops/linear/marlin/marlin_kernels_generated.cu
  ${SERVE_ROOT}/ops/linear/marlin/marlin_repack.cu
  ${SERVE_ROOT}/ops/linear/marlin/marlin_plane.cu
  ${SERVE_ROOT}/ops/linear/w8/w8_small_t.cu
  ${SERVE_ROOT}/ops/linear/w8/w8_rowsplit_gemm_splitk.cu
  ${SERVE_ROOT}/ops/linear/w8/w8_dispatch.cpp
  ${SERVE_ROOT}/ops/linear_pair/w8/w8_pair_decode.cu
  ${SERVE_ROOT}/ops/linear_pair/w8/w8_pair_gemm_concat.cu
  ${SERVE_ROOT}/ops/linear_pair/w8/w8_pair_gemm_mma.cu
  ${SERVE_ROOT}/ops/linear_pair/w8/w8_pair_gemm_splitk.cu
  ${SERVE_ROOT}/ops/linear_pair/w8/w8_pair_plan.cpp
  ${SERVE_ROOT}/ops/sparse_moe/marlin/marlin_moe_gemm.cu
  ${SERVE_ROOT}/ops/sparse_moe/marlin/marlin_moe_kernels_generated.cu
  ${SERVE_ROOT}/ops/sparse_moe/marlin/marlin_moe_repack.cu
  ${SERVE_ROOT}/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu
  ${SERVE_ROOT}/ops/sparse_moe/decode/sparse_moe_decode_plan.cpp
  ${SERVE_ROOT}/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu
  ${SERVE_ROOT}/ops/sparse_moe/prefill/sparse_moe_prefill_plan.cpp
  ${SERVE_ROOT}/ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu
  ${SERVE_ROOT}/ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp
  ${SERVE_ROOT}/ops/linear/bf16/bf16_cublaslt.cpp
  ${SERVE_ROOT}/ops/hyper_connection/hyper_connection.cu
  ${SERVE_ROOT}/ops/ngram_ple/ngram_ple.cu
  ${SERVE_ROOT}/ops/qsa_indexer/qsa_indexer.cu
  ${SERVE_ROOT}/ops/expert_slot_cache/expert_slot_cache.cu
  ${SERVE_ROOT}/ops/cpu_expert_compute/cpu_expert_compute.cpp
  ${SERVE_ROOT}/ops/linear/linear.cpp
  ${SERVE_ROOT}/ops/wrapper/add_bias.cpp
  ${SERVE_ROOT}/ops/wrapper/argmax.cpp
  ${SERVE_ROOT}/ops/wrapper/cast.cpp
  ${SERVE_ROOT}/ops/wrapper/causal_conv1d_silu.cpp
  ${SERVE_ROOT}/ops/wrapper/embedding.cpp
  ${SERVE_ROOT}/ops/wrapper/gdn_gating.cpp
  ${SERVE_ROOT}/ops/wrapper/gdn_gating_proj.cpp
  ${SERVE_ROOT}/ops/wrapper/gdn_input_proj.cpp
  ${SERVE_ROOT}/ops/wrapper/encoder_attention.cpp
  ${SERVE_ROOT}/ops/wrapper/gelu.cpp
  ${SERVE_ROOT}/ops/wrapper/gelu_mul.cpp
  ${SERVE_ROOT}/ops/wrapper/bidirectional_gqa_attention.cpp
  ${SERVE_ROOT}/ops/wrapper/swa.cpp
  ${SERVE_ROOT}/ops/wrapper/gqa_attention.cpp
  ${SERVE_ROOT}/ops/wrapper/kv_cache_append_prefix.cpp
  ${SERVE_ROOT}/ops/wrapper/attn_input_proj.cpp
  ${SERVE_ROOT}/ops/wrapper/l2norm.cpp
  ${SERVE_ROOT}/ops/wrapper/layer_norm.cpp
  ${SERVE_ROOT}/ops/wrapper/linear_add.cpp
  ${SERVE_ROOT}/ops/wrapper/lora.cpp
  ${SERVE_ROOT}/ops/wrapper/lora_store.cpp
    ${SERVE_ROOT}/ops/wrapper/linear_pair.cpp
  ${SERVE_ROOT}/ops/wrapper/linear_swiglu.cpp
  ${SERVE_ROOT}/ops/wrapper/mtp_pack.cpp
  ${SERVE_ROOT}/ops/wrapper/mtp_round.cpp
  ${SERVE_ROOT}/ops/wrapper/speculative_round.cpp
  ${SERVE_ROOT}/ops/wrapper/position.cpp
  ${SERVE_ROOT}/ops/wrapper/mask_columns.cpp
  ${SERVE_ROOT}/ops/wrapper/prepare_ragged_prefix.cpp
  ${SERVE_ROOT}/ops/wrapper/prepare_masked_block.cpp
  ${SERVE_ROOT}/ops/wrapper/residual_add.cpp
  ${SERVE_ROOT}/ops/wrapper/rmsnorm.cpp
  ${SERVE_ROOT}/ops/wrapper/rope.cpp
  ${SERVE_ROOT}/ops/wrapper/sampling.cpp
  ${SERVE_ROOT}/ops/wrapper/scalar.cpp
  ${SERVE_ROOT}/ops/wrapper/scatter.cpp
  ${SERVE_ROOT}/ops/wrapper/sigmoid_mul.cpp
  ${SERVE_ROOT}/ops/wrapper/mean_pool.cpp
  ${SERVE_ROOT}/ops/wrapper/scale.cpp
  ${SERVE_ROOT}/ops/wrapper/silu_mul.cpp
  ${SERVE_ROOT}/ops/wrapper/sparse_moe.cpp
  ${SERVE_ROOT}/ops/wrapper/vision_attention.cpp
  ${SERVE_ROOT}/ops/wrapper/vision_pos_embed.cpp)

set_source_files_properties(
  ${SERVE_ROOT}/ops/linear/marlin/marlin_gemm.cu
  ${SERVE_ROOT}/ops/linear/marlin/marlin_kernels_generated.cu
  ${SERVE_ROOT}/ops/linear/marlin/marlin_repack.cu
  ${SERVE_ROOT}/ops/sparse_moe/marlin/marlin_moe_gemm.cu
  ${SERVE_ROOT}/ops/sparse_moe/marlin/marlin_moe_kernels_generated.cu
  ${SERVE_ROOT}/ops/sparse_moe/marlin/marlin_moe_repack.cu
  PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

set_source_files_properties(
  ${SERVE_ROOT}/ops/sparse_moe/marlin/marlin_moe_gemm.cu
  ${SERVE_ROOT}/ops/sparse_moe/marlin/marlin_moe_kernels_generated.cu
  PROPERTIES COMPILE_OPTIONS
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-static-global-template-stub=false>")
# Vendor-tuned CPU GEMM, optional and injected by path. Both live outside the
# tree as reference checkouts rather than vendored dependencies, so a build with
# neither must still work -- the encoder's own AVX-512 microkernel is always
# there. Which one a *run* uses is decided from the CPU vendor at runtime, not
# here: a binary built on one machine has to choose sensibly on another.
set(SINFER_ZENDNN_DIR "" CACHE PATH "ZenDNN install prefix (AMD Zen)")
set(SINFER_ONEDNN_DIR "" CACHE PATH "oneDNN install prefix (Intel; what OpenVINO's CPU plugin uses)")

add_library(sinfer_encoder STATIC EXCLUDE_FROM_ALL
  ${SERVE_ROOT}/encoder/gemma_embedding.cpp
  ${SERVE_ROOT}/encoder/gemma_tokenizer.cpp
  ${SERVE_ROOT}/encoder/cpu/cpu_ops.cpp
  ${SERVE_ROOT}/encoder/cpu/cpu_gemma_embedding.cpp
)
sinfer_internal_includes(sinfer_encoder)
find_package(OpenMP REQUIRED)
target_link_libraries(sinfer_encoder PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(sinfer_encoder PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
target_include_directories(sinfer_encoder PRIVATE ${sentencepiece_SOURCE_DIR}/src)
target_link_libraries(sinfer_encoder PUBLIC sinfer_core sinfer_ops sinfer_artifact
                                     PRIVATE sentencepiece-static)
if(SINFER_ZENDNN_DIR)
  target_compile_definitions(sinfer_encoder PRIVATE SINFER_WITH_ZENDNN)
  # ZenDNN's public header includes aocl-utils for cache/CPU identification, so
  # that dependency's headers and library come along with it.
  # ZenDNN's public headers reach into its own dependencies -- aocl-utils for
  # CPU identification, AOCL-DLP for the BLAS backend -- so their include
  # directories travel with it.
  target_include_directories(sinfer_encoder PRIVATE
    ${SINFER_ZENDNN_DIR}/include
    ${SINFER_ZENDNN_DIR}/../deps/aoclutils/include
    ${SINFER_ZENDNN_DIR}/../deps/aocldlp/include
    ${SINFER_ZENDNN_DIR}/../deps/onednn/include)
  target_link_libraries(sinfer_encoder PRIVATE ${SINFER_ZENDNN_DIR}/lib/libzendnnl_archive.a)
  foreach(extra aoclutils/lib/libaoclutils.a aocldlp/lib/libaocl-dlp.a onednn/lib/libdnnl.a)
    if(EXISTS ${SINFER_ZENDNN_DIR}/../deps/${extra})
      target_link_libraries(sinfer_encoder PRIVATE ${SINFER_ZENDNN_DIR}/../deps/${extra})
    endif()
  endforeach()
  message(STATUS "CPU encoder: ZenDNN at ${SINFER_ZENDNN_DIR}")
endif()
if(SINFER_ONEDNN_DIR)
  # oneDNN's own threading runtime is OpenMP, so its symbols come with it. That
  # is the whole of its integration: one header directory, one archive and this.
  find_package(OpenMP REQUIRED)
  target_compile_definitions(sinfer_encoder PRIVATE SINFER_WITH_ONEDNN)
  target_include_directories(sinfer_encoder PRIVATE ${SINFER_ONEDNN_DIR}/include)
  target_link_libraries(sinfer_encoder PRIVATE ${SINFER_ONEDNN_DIR}/lib/libdnnl.a
                                               OpenMP::OpenMP_CXX)
  message(STATUS "CPU encoder: oneDNN at ${SINFER_ONEDNN_DIR}")
endif()

# The encoder server is a product binary, not a test fixture: `surogate serve
# --embed` execs it, so it has to exist after `make serve-build` like the engine
# does. Same source as the old serve_tests target, which is now gone.
add_executable(surogate-embed EXCLUDE_FROM_ALL ${SERVE_ROOT}/encoder/embedding_server.cpp)
target_include_directories(surogate-embed PRIVATE
  ${SERVE_ROOT} ${SINFER_JSON_INCLUDE} ${SUROGATE_VENDOR_DIR}/cpp-httplib)
target_link_libraries(surogate-embed PRIVATE sinfer_encoder)

sinfer_internal_includes(sinfer_ops)
# surogate vendor patch (PATCHES.md #25): cutlass headers for the sm_120a
# blockscaled NVFP4 GEMM TU. Points at the flashinfer-shipped cutlass by
# default; vendoring a snapshot is tracked.
# cutlass comes from the main project's FetchContent (v4.6.1).
set(SINFER_CUTLASS_INCLUDE "${cutlass_SOURCE_DIR}/include")
# The CPU expert-compute kernels carry their own AVX-512 code path (runtime-detected); the
# rest of the engine stays generic x86-64.
set_source_files_properties(${SERVE_ROOT}/ops/cpu_expert_compute/cpu_expert_compute.cpp PROPERTIES
  COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mf16c;-mfma")
set_source_files_properties(${SERVE_ROOT}/ops/linear/w8a8/w4fp4_cutlass_gemm.cu PROPERTIES
    INCLUDE_DIRECTORIES "${SINFER_CUTLASS_INCLUDE};${cutlass_SOURCE_DIR}/tools/util/include"
    COMPILE_OPTIONS "--expt-relaxed-constexpr")
sinfer_cuda_archive(sinfer_ops)
target_link_libraries(sinfer_ops PUBLIC sinfer_core PRIVATE sinfer_nvfp4_tma sinfer_trtllm_moe CUDA::cublasLt)

# Product-independent text and media mechanisms.
add_library(sinfer_text STATIC EXCLUDE_FROM_ALL
  ${SERVE_ROOT}/text/unicode.cpp
  ${SUROGATE_VENDOR_DIR}/utf8proc/utf8proc.c)
sinfer_internal_includes(sinfer_text)

# surogate vendor patch (csrc/src/serve/PATCHES.md): stub decode without FFmpeg.
if(SINFER_ENABLE_FFMPEG)
  add_library(sinfer_media_decode STATIC EXCLUDE_FROM_ALL
    ${SERVE_ROOT}/media/decode/decode.cpp)
  sinfer_internal_includes(sinfer_media_decode)
  target_link_libraries(sinfer_media_decode PRIVATE PkgConfig::FFMPEG)
else()
  add_library(sinfer_media_decode STATIC EXCLUDE_FROM_ALL
    ${SERVE_ROOT}/media/decode/decode_stub.cpp)
  sinfer_internal_includes(sinfer_media_decode)
endif()

if(SINFER_BUILD_MEDIA_ACQUIRE)
  # Product-only path/data/HTTP acquisition. No target package links this library.
  add_library(sinfer_media_acquire STATIC EXCLUDE_FROM_ALL
    ${SERVE_ROOT}/product/media_acquire/acquire.cpp)
  sinfer_internal_includes(sinfer_media_acquire)
  target_link_libraries(sinfer_media_acquire PRIVATE PkgConfig::LIBCURL)
endif()

if(SINFER_BUILD_PROMPT_INPUT)
  # Owning command-line message input shared by product entry points and target diagnostics.
  add_library(sinfer_product_prompt_input STATIC EXCLUDE_FROM_ALL
    ${SERVE_ROOT}/product/prompt_input/prompt_input.cpp)
  sinfer_internal_includes(sinfer_product_prompt_input)
  target_link_libraries(sinfer_product_prompt_input PRIVATE sinfer_media_acquire)
endif()

if(SINFER_BUILD_APPS OR BUILD_TESTING)
  add_library(sinfer_product_load_progress STATIC EXCLUDE_FROM_ALL
    ${SERVE_ROOT}/product/load_progress/load_progress.cpp)
  sinfer_internal_includes(sinfer_product_load_progress)
endif()

# One cold closed registry plus the opaque public product Engine. Family and exact-target
# directories contribute their sources directly to this single compile owner below.
add_library(sinfer_engine STATIC EXCLUDE_FROM_ALL
  ${SERVE_ROOT}/targets/registry.cpp
  ${SERVE_ROOT}/runtime/contract/sampling.cpp
  ${SERVE_ROOT}/runtime/engine/engine.cpp)
# The project's tokenizer, shared with training. The serving frontend delegates
# to it for SentencePiece checkpoints (Llama 2, TinyLlama, Gemma), whose scheme
# its own byte-level BPE cannot represent. Pure C++ with no CUDA dependency.
target_sources(sinfer_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/src/tokenizer/tokenizer.cpp
  ${CMAKE_SOURCE_DIR}/src/tokenizer/tok_profile.cpp
  ${CMAKE_SOURCE_DIR}/src/tokenizer/unicode.cpp
  ${CMAKE_SOURCE_DIR}/src/tokenizer/unicode_data.cpp)
target_include_directories(sinfer_engine PRIVATE ${CMAKE_SOURCE_DIR}/src
                                                 ${CMAKE_SOURCE_DIR}/src/tokenizer
                                                 ${minja_SOURCE_DIR}/include
                                                 ${fmt_SOURCE_DIR}/include)
target_compile_definitions(sinfer_engine PRIVATE FMT_HEADER_ONLY=1)

sinfer_internal_includes(sinfer_engine)
target_include_directories(sinfer_engine PUBLIC ${SERVE_ROOT})

add_subdirectory(${SERVE_ROOT}/family serve_family EXCLUDE_FROM_ALL)
add_subdirectory(${SERVE_ROOT}/targets/qwen3 serve_t_qwen3 EXCLUDE_FROM_ALL)
add_subdirectory(${SERVE_ROOT}/targets/llama serve_t_llama EXCLUDE_FROM_ALL)
add_subdirectory(${SERVE_ROOT}/targets/gemma3 serve_t_gemma3 EXCLUDE_FROM_ALL)
add_subdirectory(${SERVE_ROOT}/targets/qwen3_5_0_8b serve_t_qwen3_5_0_8b EXCLUDE_FROM_ALL)
add_subdirectory(${SERVE_ROOT}/targets/qwen3_5_2b serve_t_qwen3_5_2b EXCLUDE_FROM_ALL)
add_subdirectory(${SERVE_ROOT}/targets/qwen3_5_4b serve_t_qwen3_5_4b EXCLUDE_FROM_ALL)
add_subdirectory(${SERVE_ROOT}/targets/qwen3_6_27b serve_t_qwen3_6_27b EXCLUDE_FROM_ALL)
add_subdirectory(${SERVE_ROOT}/targets/qwen3_6_35b_a3b serve_t_qwen3_6_35b_a3b EXCLUDE_FROM_ALL)
add_subdirectory(${SERVE_ROOT}/targets/qwen4exp serve_t_qwen4exp EXCLUDE_FROM_ALL)

target_link_libraries(sinfer_engine
  PRIVATE
    sinfer_artifact
    sinfer_core
    sinfer_ops
    sinfer_text
    sinfer_media_decode)
add_library(sinfer::engine ALIAS sinfer_engine)

if(SINFER_BUILD_SERVE)
  # HTTP protocol/product adapter. No serving header is part of the public API.
  add_library(sinfer_serve STATIC EXCLUDE_FROM_ALL
    ${SERVE_ROOT}/serve/openai_schema.cpp
    ${SERVE_ROOT}/serve/responses_schema.cpp
    ${SERVE_ROOT}/serve/anthropic_schema.cpp
    ${SERVE_ROOT}/serve/response_store.cpp
    ${SERVE_ROOT}/serve/lora_registry.cpp
    ${SERVE_ROOT}/serve/output_parsers.cpp
    ${SERVE_ROOT}/serve/tool_call_parser.cpp
    ${SERVE_ROOT}/serve/translate.cpp
    ${SERVE_ROOT}/serve/console_log.cpp
    ${SERVE_ROOT}/serve/generation_service.cpp
    ${SERVE_ROOT}/serve/model_scheduler.cpp
    ${SERVE_ROOT}/serve/request_log.cpp
    ${SERVE_ROOT}/serve/http_server.cpp
    ${SERVE_ROOT}/serve/responses_http.cpp
    ${SERVE_ROOT}/serve/serve_options.cpp)
  sinfer_internal_includes(sinfer_serve)
  target_include_directories(sinfer_serve PRIVATE
    ${SUROGATE_VENDOR_DIR}/cpp-httplib)
  target_link_libraries(sinfer_serve
    PUBLIC sinfer_engine Threads::Threads
    PRIVATE sinfer_media_acquire CUDA::cudart)
endif()


        # ---- serve executables (formerly src/serve/apps/CMakeLists.txt) ----
add_executable(sinfer EXCLUDE_FROM_ALL   # surogate vendor patch (csrc/src/serve/PATCHES.md): product binary names.
  ${SERVE_ROOT}/cli/main.cpp
  ${SERVE_ROOT}/cli/options.cpp)
target_include_directories(sinfer PRIVATE
  ${SERVE_ROOT}
  ${SERVE_ROOT}
  ${SUROGATE_VENDOR_DIR})
target_link_libraries(sinfer PRIVATE
  sinfer_engine
  sinfer_product_load_progress
  sinfer_product_prompt_input)

add_executable(sinfer-serve EXCLUDE_FROM_ALL ${SERVE_ROOT}/server/main.cpp)
target_include_directories(sinfer-serve PRIVATE
  ${SINFER_JSON_INCLUDE}
  ${SERVE_ROOT}
  ${SERVE_ROOT}
  ${SUROGATE_VENDOR_DIR}
  ${SUROGATE_VENDOR_DIR}/cpp-httplib)
target_link_libraries(sinfer-serve PRIVATE sinfer_serve sinfer_product_load_progress)

# surogate vendor patch (csrc/src/serve/PATCHES.md): the product binaries carry
# surogate's name; SInfer attribution lives in NOTICE, not in process names.
set_target_properties(sinfer PROPERTIES OUTPUT_NAME surogate-engine-cli)
set_target_properties(sinfer-serve PROPERTIES OUTPUT_NAME surogate-engine)


        if(SUROGATE_SERVE_TESTS)
            enable_testing()
            add_subdirectory(src/testing/serve serve_tests EXCLUDE_FROM_ALL)
        endif()
        if(SUROGATE_SERVE_BENCH)
            add_subdirectory(src/testing/serve/bench serve_bench EXCLUDE_FROM_ALL)
        endif()
        # Python binding for the serve engine: a SEPARATE extension module
        # from _surogate (arch gating + payload + lifecycle isolation).
        if(PYTHON_BINDING)
            nanobind_add_module(_surogate_serve NOSTRIP
                ${CMAKE_SOURCE_DIR}/src/binding/py_serve.cpp)
            target_include_directories(_surogate_serve PRIVATE
                ${SERVE_ROOT} ${SINFER_JSON_INCLUDE})
            target_link_libraries(_surogate_serve PRIVATE
                sinfer_engine sinfer_product_prompt_input)
            set_target_properties(_surogate_serve PROPERTIES EXCLUDE_FROM_ALL TRUE)
        endif()

        endif()  # LIBCURL_FOUND
    endif()
endif()
