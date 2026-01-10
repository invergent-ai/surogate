cmake_minimum_required(VERSION 3.25)

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

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        EXCLUDE_FROM_ALL
)

FetchContent_Declare(
        cudnn-fe
        GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
        GIT_TAG v1.17.0
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
        GIT_TAG v2.9.2
)

# we could use fmt only if std::format is not available, but (e.g., g++-12), but
# since it is compatible with std::format, its easier to just use it consistently.
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG        12.0.0
)

# Catch2 for tests
FetchContent_Declare(
        catch2
        QUIET
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.6.0
)

# CUTLASS for optimized GEMM kernels
# Disable all CUTLASS tests, examples, and profiler to speed up configuration
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Enable CUTLASS tests")
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Enable CUTLASS examples")
set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "Enable CUTLASS tools")
set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "Enable CUTLASS profiler")
# set(CUTLASS_ENABLE_LIBRARY OFF CACHE BOOL "Enable CUTLASS library")
# set(CUTLASS_LIBRARY_KERNELS "" CACHE STRING "Comma-separated list of kernel name filters")
# Only build for SM80+ (Ampere/Hopper/Blackwell/Next-gen): 80, 86, 87, 89, 90, 90a, 100a, 103a, 120a
# SM80-87: Ampere, SM89: Ada, SM90/90a: Hopper, SM100a: Blackwell B200, SM103a: Blackwell B300, SM120a: RTX 50x0/DGX Spark
# Note: SM100/103/120 require 'a' suffix for tcgen05.mma and block-scaled MMA features (FP4)
set(CUTLASS_NVCC_ARCHS "89;90;90a;100a;103a;120a;121a" CACHE STRING "CUDA architectures for CUTLASS")
set(CUTLASS_NVCC_ARCHS_SUPPORTED "89;90;90a;100a;103a;120a;121a" CACHE STRING "NVCC archs to build library kernels")

FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v4.3.4
        EXCLUDE_FROM_ALL
)

find_package(CUDAToolkit COMPONENTS cuBLAS REQUIRED)

FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(cudnn-fe)
FetchContent_MakeAvailable(cli11)
FetchContent_MakeAvailable(fmt)
FetchContent_MakeAvailable(catch2)
FetchContent_MakeAvailable(cutlass)

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

    src/training/dataloader.cpp
    src/training/logging.cpp
    src/training/checkpoint.cpp
    src/training/model.cpp

    src/config/pretrained_config.cpp
    src/models/registry.cpp
    src/models/qwen25.cpp
    src/models/qwen3.cpp
    src/models/llama.cpp

    src/modules/qlora/fp8_weights.cpp
    src/modules/qlora/fp4_weights.cpp
    src/modules/qlora/bnb_weights.cpp

    # Recipe system
    src/recipes/recipe.cpp
    src/recipes/recipe_factory.cpp
    src/recipes/fp8_hybrid/fp8_hybrid_recipe.cpp
    src/recipes/nvfp4/nvfp4_recipe.cpp

    # Shared kernels (used by all recipes)
    src/kernels/kernels.cpp
    src/kernels/encoder.cu
    src/kernels/matmul.cpp
    src/kernels/swiglu.cu
    src/kernels/rope.cu
    src/kernels/rmsnorm.cu
    src/kernels/qk_norm.cu
    src/kernels/cudnn_att.cpp
    src/kernels/block_quant.cu
    src/kernels/bnb_quant.cu
    src/kernels/global_norm.cu
    src/kernels/bias.cu
    src/modules/optimizers/adamw_8bit.cu
    src/modules/optimizers/polar_express.cu
    src/modules/optimizers/normuon.cu
    src/kernels/attention.cu
    src/kernels/fused_classifier.cu
    src/kernels/transpose.cu
    src/kernels/vector_add.cu
    src/kernels/slice_add.cu
    src/kernels/random.cu
    src/kernels/fill.cu
    src/kernels/convert.cu

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

target_link_libraries(surogate-common PRIVATE ${CUFILE_LIBS} cudnn_frontend ${CUDNN_LIBRARY} nvidia::nccl fmt::fmt-header-only)
target_link_libraries(surogate-common PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cublasLt ${NVML_LIBS} nlohmann_json::nlohmann_json)
target_link_libraries(surogate-common PUBLIC ${MULTI_GPU_DEPS})
target_compile_definitions(surogate-common PUBLIC ${MULTI_GPU_DEFS})
target_include_directories(surogate-common PRIVATE ${CUDNN_INCLUDE_DIR})
target_include_directories(surogate-common PUBLIC src ${cutlass_SOURCE_DIR}/include ${cutlass_SOURCE_DIR}/tools/util/include)
target_compile_options(surogate-common PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo> $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

if (NOT SKBUILD)
    install(TARGETS surogate-common
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
    )
endif ()


if (PYTHON_BINDING)
    find_package(Python COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)
    FetchContent_MakeAvailable(nanobind)
    nanobind_add_module(_surogate STABLE_ABI src/binding/binding.cpp src/binding/py_train.cpp)
    target_link_libraries(_surogate PUBLIC surogate-common Python::Module fmt::fmt-header-only)

    # make sure we pick up cuda libraries from within the current python env, if available
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
option(BUILD_TESTS "Build unit tests" ON)
if (BUILD_TESTS)
    enable_testing()

	    add_executable(unit-tests
	            src/testing/test_main.cpp
	            src/testing/test-swiglu.cu
	            src/testing/test-rope.cu
	            src/testing/test-nccl.cpp
	            src/testing/test-grad-clip.cu
	            src/testing/test-classifier.cu
	            src/testing/test-modular-training-loop.cpp
	            src/testing/test_pretrained_config.cpp
	            src/testing/test-sol.cpp
	            src/testing/test-qk-norm-rope.cu
	            src/testing/test-qlora.cu
	            src/testing/test-adamw.cu
	            src/testing/test-normuon.cu
	            src/testing/test-fp4.cu
	            src/testing/test-bnb-quant.cpp
	    )
    target_link_libraries(unit-tests PRIVATE surogate-common Catch2::Catch2 CLI11::CLI11 fmt::fmt-header-only)
    target_compile_options(unit-tests PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

    add_test(NAME unit-tests COMMAND unit-tests)
endif ()
