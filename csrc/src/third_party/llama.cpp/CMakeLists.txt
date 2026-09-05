# Our top level for the vendored subset. Upstream's own root builds a server, examples, tests,
# a UI and eighteen tools; we take the library and one program. Writing this rather than
# patching theirs keeps the exclusions visible in one place and survives an update.
cmake_minimum_required(VERSION 3.14)
project(llama_cpp_vendored C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CPU only, and deliberately: `llama-quantize` reads a GGUF and writes a GGUF, and every
# accelerator backend upstream carries would be built for nothing.
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_SYCL   OFF CACHE BOOL "" FORCE)
set(GGML_BLAS   OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_RPC    OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
# We build `common`, which `llama-quantize-impl` links. Upstream gates its vendored
# cpp-httplib on this same flag, so leaving it off links `-lcpp-httplib` against a target
# that was never declared -- a clean build fails at the last step while a warm one succeeds.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)

# The revision this tree was taken from. Upstream reads these out of git; a vendored copy has
# no git, and answering "unknown" would make `llama-quantize --version` unable to say which
# arithmetic produced a file. See PROVENANCE.md, which must be updated with them.
set(LLAMA_BUILD_NUMBER 10797)
set(LLAMA_BUILD_COMMIT "163a4079")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Upstream's own root includes these before descending; the subdirectories use what they
# define (`llama_add_compile_flags`, the build-info target), so we include them too.
include(CheckIncludeFileCXX)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/license.cmake)
add_subdirectory(ggml)
add_subdirectory(src)
add_subdirectory(vendor)
add_subdirectory(common)
add_subdirectory(tools/quantize)
