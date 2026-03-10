// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// JIT kernel loader implementation.

#include "runtime/jit/jit_kernel.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <cuda.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace {

// ---------------------------------------------------------------------------
// CUDA Driver API error handling (mirrors CUDA_CHECK for the runtime API)
// ---------------------------------------------------------------------------

class cu_error : public std::runtime_error {
public:
    cu_error(CUresult err, const std::string& arg)
        : std::runtime_error(arg), code(err) {}
    CUresult code;
};

void cu_throw_on_error(CUresult status, const char* statement,
                       const char* file, int line) {
    if (status == CUDA_SUCCESS) return;

    const char* name = nullptr;
    const char* msg  = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &msg);
    throw cu_error(
        status,
        fmt::format("CUDA Driver Error at {}:{}: {} returned {} ({})",
                     file, line, statement,
                     name ? name : "unknown",
                     msg  ? msg  : "unknown"));
}

#define CU_CHECK(status) cu_throw_on_error(status, #status, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// File I/O helper
// ---------------------------------------------------------------------------

std::vector<char> read_binary_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error(
            fmt::format("JitKernel: cannot open '{}'", path));
    }
    auto size = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    if (!f) {
        throw std::runtime_error(
            fmt::format("JitKernel: failed to read '{}' ({} bytes)",
                        path, size));
    }
    return buf;
}

// Ensure a CUDA primary context is active before calling driver API functions.
// The runtime API lazily creates the context on first use (e.g. cudaMalloc), but
// driver API calls like cuModuleLoadData require an existing context.
// cudaFree(nullptr) is the canonical no-op that triggers runtime initialization.
void ensure_cuda_context() {
    CUcontext ctx = nullptr;
    CU_CHECK(cuCtxGetCurrent(&ctx));
    if (ctx == nullptr) {
        cudaFree(nullptr);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// JitKernel: static factory methods
// ---------------------------------------------------------------------------

JitKernel JitKernel::load_cubin(const std::string& path,
                                const JitKernelMeta& meta) {
    auto data = read_binary_file(path);
    return load_cubin(data.data(), data.size(), meta);
}

JitKernel JitKernel::load_ptx(const std::string& path,
                               const JitKernelMeta& meta) {
    auto data = read_binary_file(path);
    // PTX is text — ensure null-terminated for cuModuleLoadDataEx
    data.push_back('\0');
    return load_ptx(data.data(), data.size(), meta);
}

JitKernel JitKernel::load_cubin(const void* data, size_t /*size*/,
                                const JitKernelMeta& meta) {
    ensure_cuda_context();
    JitKernel k;
    k.meta_ = meta;
    CU_CHECK(cuModuleLoadData(&k.module_, data));
    k.init_function();
    return k;
}

JitKernel JitKernel::load_ptx(const void* data, size_t /*size*/,
                               const JitKernelMeta& meta) {
    ensure_cuda_context();
    JitKernel k;
    k.meta_ = meta;

    // JIT-compile PTX with default options.
    // Add options here if you need to control max registers, opt level, etc.
    CU_CHECK(cuModuleLoadDataEx(&k.module_, data,
                                0, nullptr, nullptr));
    k.init_function();
    return k;
}

JitKernel JitKernel::load_library(const std::string& path,
                                  const JitKernelMeta& meta) {
    auto data = read_binary_file(path);
    return load_library(data.data(), data.size(), meta);
}

JitKernel JitKernel::load_library(const void* data, size_t /*size*/,
                                  const JitKernelMeta& meta) {
    ensure_cuda_context();
    JitKernel k;
    k.meta_ = meta;

    CU_CHECK(cuLibraryLoadData(&k.library_, data,
                                nullptr, nullptr, 0,
                                nullptr, nullptr, 0));

    CUkernel kern = nullptr;
    CU_CHECK(cuLibraryGetKernel(&kern, k.library_, meta.name.c_str()));
    CU_CHECK(cuKernelGetFunction(&k.function_, kern));

    // Allow large dynamic shared memory (>48 KB) if requested
    if (meta.shared_mem_bytes > 48 * 1024) {
        CU_CHECK(cuFuncSetAttribute(
            k.function_,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            meta.shared_mem_bytes));
    }

    return k;
}

JitKernel JitKernel::load_manifest(const std::string& manifest_path) {
    auto data = read_binary_file(manifest_path);
    auto j = nlohmann::json::parse(data.begin(), data.end());

    JitKernelMeta meta;
    meta.name              = j.at("name").get<std::string>();
    meta.num_warps         = j.value("num_warps", 4);
    meta.shared_mem_bytes  = j.value("shared_mem", 0);
    meta.extra_null_params = j.value("extra_null_params", 0);

    // Resolve cubin/ptx path relative to the manifest directory
    auto dir = std::filesystem::path(manifest_path).parent_path();

    // CuTe DSL / Blackwell cubins use the Library API (cuLibraryLoadData)
    auto backend = j.value("backend", "");
    bool use_library_api = (backend == "cute_dsl");

    if (j.contains("cubin")) {
        auto cubin_path = (dir / j.at("cubin").get<std::string>()).string();
        return use_library_api ? load_library(cubin_path, meta)
                               : load_cubin(cubin_path, meta);
    }
    if (j.contains("ptx")) {
        auto ptx_path = (dir / j.at("ptx").get<std::string>()).string();
        return load_ptx(ptx_path, meta);
    }

    throw std::runtime_error(
        fmt::format("JitKernel: manifest '{}' has neither 'cubin' nor 'ptx' key",
                    manifest_path));
}

// ---------------------------------------------------------------------------
// JitKernel: init, move, destroy
// ---------------------------------------------------------------------------

void JitKernel::init_function() {
    CU_CHECK(cuModuleGetFunction(&function_, module_,
                                  meta_.name.c_str()));

    // Allow large dynamic shared memory (>48 KB) if requested
    if (meta_.shared_mem_bytes > 48 * 1024) {
        CU_CHECK(cuFuncSetAttribute(
            function_,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            meta_.shared_mem_bytes));
    }
}

JitKernel::~JitKernel() noexcept {
    if (library_) {
        cuLibraryUnload(library_);
    } else if (module_) {
        cuModuleUnload(module_);
    }
}

JitKernel::JitKernel(JitKernel&& o) noexcept
    : module_(o.module_),
      library_(o.library_),
      function_(o.function_),
      meta_(std::move(o.meta_)) {
    o.module_   = nullptr;
    o.library_  = nullptr;
    o.function_ = nullptr;
}

JitKernel& JitKernel::operator=(JitKernel&& o) noexcept {
    if (this != &o) {
        if (library_) cuLibraryUnload(library_);
        else if (module_) cuModuleUnload(module_);
        module_   = o.module_;
        library_  = o.library_;
        function_ = o.function_;
        meta_     = std::move(o.meta_);
        o.module_   = nullptr;
        o.library_  = nullptr;
        o.function_ = nullptr;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// JitKernel: launch
// ---------------------------------------------------------------------------

void JitKernel::launch(dim3 grid, void** args, cudaStream_t stream) const {
    int block_x = meta_.num_warps * 32;
    launch(grid, dim3(block_x, 1, 1), meta_.shared_mem_bytes, args, stream);
}

void JitKernel::launch(dim3 grid, dim3 block, int shared_mem_bytes,
                        void** args, cudaStream_t stream) const {
    CU_CHECK(cuLaunchKernel(
        function_,
        grid.x,  grid.y,  grid.z,
        block.x, block.y, block.z,
        static_cast<unsigned int>(shared_mem_bytes),
        stream,
        args,
        nullptr));
}

void JitKernel::launch_triton(dim3 grid, void** user_args, int num_user_args,
                               cudaStream_t stream) const {
    int block_x = meta_.num_warps * 32;
    int extra = meta_.extra_null_params;

    if (extra == 0) {
        // No scratch params — launch directly
        launch(grid, dim3(block_x, 1, 1), meta_.shared_mem_bytes,
               user_args, stream);
        return;
    }

    // Build extended args array: user args + null scratch pointers
    int total = num_user_args + extra;
    std::vector<void*> args(total);
    for (int i = 0; i < num_user_args; ++i) {
        args[i] = user_args[i];
    }
    // Triton scratch pointers: pass pointers to null CUdeviceptr values.
    // These params are declared as .u64 .ptr in the cubin but never loaded.
    static CUdeviceptr null_scratch = 0;
    for (int i = num_user_args; i < total; ++i) {
        args[i] = &null_scratch;
    }

    launch(grid, dim3(block_x, 1, 1), meta_.shared_mem_bytes,
           args.data(), stream);
}
