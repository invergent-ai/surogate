// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// JIT kernel loader for externally compiled CUDA kernels (Triton, CuTe, etc.).
// Loads .cubin or .ptx files via the CUDA Driver API and launches them with
// cuLaunchKernel.

#ifndef SUROGATE_SRC_RUNTIME_JIT_JIT_KERNEL_H
#define SUROGATE_SRC_RUNTIME_JIT_JIT_KERNEL_H

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

/// Metadata describing a compiled JIT kernel.
struct JitKernelMeta {
    std::string name;          ///< Kernel function name (mangled, as in the cubin)
    int num_warps = 4;         ///< Warps per block (block_x = num_warps * 32)
    int shared_mem_bytes = 0;  ///< Dynamic shared memory in bytes

    /// Number of extra null-pointer parameters the compiler appends after user params.
    /// Triton 3.x adds 2 (global_scratch, profile_scratch); plain CUDA kernels use 0.
    int extra_null_params = 0;
};

/// RAII wrapper around a CUDA module + function loaded from an external cubin/PTX.
///
/// Supports two loading paths:
///   - cuModuleLoadData (legacy, for Triton and pre-Blackwell cubins)
///   - cuLibraryLoadData (CUDA 12+, required for Blackwell/sm_120+ cubins)
///
/// Usage:
///   auto kernel = JitKernel::load_cubin("path/to/kernel.cubin", {"kernel_name", 4, 0});
///   void* args[] = {&d_x, &d_y, &d_out, &n};
///   kernel.launch(dim3(grid_x), args, stream);
///
class JitKernel {
public:
    /// Load a compiled cubin from a file.
    static JitKernel load_cubin(const std::string& path, const JitKernelMeta& meta);

    /// Load PTX source from a file (JIT-compiled by the driver on first load).
    static JitKernel load_ptx(const std::string& path, const JitKernelMeta& meta);

    /// Load a cubin from an in-memory buffer.
    static JitKernel load_cubin(const void* data, size_t size, const JitKernelMeta& meta);

    /// Load PTX from an in-memory buffer.
    static JitKernel load_ptx(const void* data, size_t size, const JitKernelMeta& meta);

    /// Load a cubin via cuLibraryLoadData (CUDA 12+ Library API).
    /// Required for Blackwell (sm_120+) cubins that use reserved shared memory.
    static JitKernel load_library(const std::string& path, const JitKernelMeta& meta);

    /// Load a cubin via cuLibraryLoadData from an in-memory buffer.
    static JitKernel load_library(const void* data, size_t size, const JitKernelMeta& meta);

    /// Load from a JSON manifest file (resolves cubin/ptx path relative to manifest).
    /// Manifest format: {"name": "...", "num_warps": 4, "shared_mem": 0, "cubin": "file.cubin"}
    /// If "backend" is "cute_dsl", uses the Library API (cuLibraryLoadData).
    static JitKernel load_manifest(const std::string& manifest_path);

    ~JitKernel() noexcept;
    JitKernel(JitKernel&&) noexcept;
    JitKernel& operator=(JitKernel&&) noexcept;
    JitKernel(const JitKernel&) = delete;
    JitKernel& operator=(const JitKernel&) = delete;

    /// Launch the kernel (raw). Args must include ALL parameters (including
    /// any compiler-added scratch pointers). Block/shared from metadata.
    void launch(dim3 grid, void** args, cudaStream_t stream) const;

    /// Launch with explicit block dimensions and shared memory (overrides metadata).
    void launch(dim3 grid, dim3 block, int shared_mem_bytes, void** args, cudaStream_t stream) const;

    /// Launch a Triton-compiled kernel. Appends `extra_null_params` null pointer
    /// args to the user-provided args array before launching.
    /// @param num_user_args  Number of user-supplied args in the array.
    void launch_triton(dim3 grid, void** user_args, int num_user_args, cudaStream_t stream) const;

    [[nodiscard]] const JitKernelMeta& meta() const {
        return meta_;
    }
    [[nodiscard]] CUfunction function() const {
        return function_;
    }

private:
    JitKernel() = default;
    void init_function();

    CUmodule module_ = nullptr;
    CUlibrary library_ = nullptr;  ///< CUDA 12+ library handle (for Blackwell)
    CUfunction function_ = nullptr;
    JitKernelMeta meta_;
};

#endif  // SUROGATE_SRC_RUNTIME_JIT_JIT_KERNEL_H
