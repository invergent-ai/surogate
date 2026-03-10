// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end tests for the JIT kernel loader (runtime/jit/jit_kernel.h).
// Uses an embedded PTX vector_add kernel to validate the full pipeline:
//   PTX text → cuModuleLoadDataEx → cuModuleGetFunction → cuLaunchKernel → verify results

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "runtime/jit/jit_kernel.h"
#include "utilities/utils.h"

namespace {

// ---------------------------------------------------------------------------
// Minimal PTX for: out[i] = x[i] + y[i]
//
// Equivalent CUDA source:
//   __global__ void vector_add(const float* x, const float* y,
//                              float* out, int n) {
//       int i = blockIdx.x * blockDim.x + threadIdx.x;
//       if (i < n) out[i] = x[i] + y[i];
//   }
// ---------------------------------------------------------------------------
constexpr const char* VECTOR_ADD_PTX = R"(
.version 7.0
.target sm_80
.address_size 64

.visible .entry vector_add(
    .param .u64 param_x,
    .param .u64 param_y,
    .param .u64 param_out,
    .param .u32 param_n
) {
    .reg .pred  %p<2>;
    .reg .b32   %r<5>;
    .reg .b64   %rd<8>;
    .reg .f32   %f<4>;

    ld.param.u64    %rd0, [param_x];
    ld.param.u64    %rd1, [param_y];
    ld.param.u64    %rd2, [param_out];
    ld.param.u32    %r0,  [param_n];

    mov.u32         %r1, %ctaid.x;
    mov.u32         %r2, %ntid.x;
    mov.u32         %r3, %tid.x;
    mad.lo.s32      %r4, %r1, %r2, %r3;   // i = blockIdx.x * blockDim.x + threadIdx.x

    setp.ge.u32     %p0, %r4, %r0;         // if (i >= n) goto END
    @%p0 bra        END;

    cvt.u64.u32     %rd3, %r4;
    shl.b64         %rd4, %rd3, 2;          // byte offset = i * 4
    add.u64         %rd5, %rd0, %rd4;       // &x[i]
    add.u64         %rd6, %rd1, %rd4;       // &y[i]
    add.u64         %rd7, %rd2, %rd4;       // &out[i]

    ld.global.f32   %f0, [%rd5];
    ld.global.f32   %f1, [%rd6];
    add.f32         %f2, %f0, %f1;
    st.global.f32   [%rd7], %f2;

END:
    ret;
}
)";

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// RAII guard for device memory
struct DeviceBuffer {
    float* ptr = nullptr;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
};

// Helper: allocate + upload host data to device
DeviceBuffer make_device(const std::vector<float>& h, cudaStream_t stream = nullptr) {
    DeviceBuffer buf;
    CUDA_CHECK(cudaMalloc(&buf.ptr, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(buf.ptr, h.data(), h.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    return buf;
}

// Helper: download device data to host
std::vector<float> download(float* d_ptr, size_t n, cudaStream_t stream = nullptr) {
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpyAsync(h.data(), d_ptr, n * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h;
}

// RAII guard for a temporary directory (cleaned up on destruction)
struct TempDir {
    std::filesystem::path path;
    TempDir() {
        path = std::filesystem::temp_directory_path() / "surogate_test_jit";
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

}  // namespace

// ===========================================================================

TEST_CASE("JitKernel: load PTX from memory and launch vector_add",
          "[jit][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    int N = 2048;
    int num_warps = 4;
    int block_size = num_warps * 32;  // 128
    int grid_x = (N + block_size - 1) / block_size;

    // Load the PTX kernel
    JitKernelMeta meta{"vector_add", num_warps, 0};
    auto kernel = JitKernel::load_ptx(
        VECTOR_ADD_PTX, std::strlen(VECTOR_ADD_PTX) + 1, meta);

    REQUIRE(kernel.meta().name == "vector_add");
    REQUIRE(kernel.meta().num_warps == num_warps);
    REQUIRE(kernel.function() != nullptr);

    // Prepare input data
    std::vector<float> h_x(N), h_y(N);
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(i) * 0.5f;
    }

    auto d_x   = make_device(h_x);
    auto d_y   = make_device(h_y);
    DeviceBuffer d_out;
    CUDA_CHECK(cudaMalloc(&d_out.ptr, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out.ptr, 0, N * sizeof(float)));

    // Launch
    void* args[] = {&d_x.ptr, &d_y.ptr, &d_out.ptr, &N};
    kernel.launch(dim3(grid_x), args, nullptr);

    // Verify
    auto h_out = download(d_out.ptr, N);
    for (int i = 0; i < N; ++i) {
        float expected = h_x[i] + h_y[i];
        REQUIRE(h_out[i] == Catch::Approx(expected).margin(1e-6f));
    }
}

// ===========================================================================

TEST_CASE("JitKernel: launch with explicit block override",
          "[jit][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    int N = 512;
    int block_size = 256;  // override: 8 warps instead of default 4
    const int grid_x = (N + block_size - 1) / block_size;

    JitKernelMeta meta{"vector_add", /*num_warps=*/4, /*shared_mem_bytes=*/0};
    auto kernel = JitKernel::load_ptx(
        VECTOR_ADD_PTX, std::strlen(VECTOR_ADD_PTX) + 1, meta);

    std::vector<float> h_x(N, 1.0f), h_y(N, 2.0f);
    auto d_x = make_device(h_x);
    auto d_y = make_device(h_y);
    DeviceBuffer d_out;
    CUDA_CHECK(cudaMalloc(&d_out.ptr, N * sizeof(float)));

    // Launch with explicit block=256 (overriding meta.num_warps=4 → 128)
    void* args[] = {&d_x.ptr, &d_y.ptr, &d_out.ptr, &N};
    kernel.launch(dim3(grid_x), dim3(block_size), 0, args, nullptr);

    auto h_out = download(d_out.ptr, N);
    for (int i = 0; i < N; ++i) {
        REQUIRE(h_out[i] == Catch::Approx(3.0f).margin(1e-6f));
    }
}

// ===========================================================================

TEST_CASE("JitKernel: load PTX from file",
          "[jit][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    TempDir tmp;
    auto ptx_path = (tmp.path / "vector_add.ptx").string();

    // Write PTX to file
    {
        std::ofstream f(ptx_path);
        REQUIRE(f.is_open());
        f << VECTOR_ADD_PTX;
    }

    JitKernelMeta meta{"vector_add", 4, 0};
    auto kernel = JitKernel::load_ptx(ptx_path, meta);

    int N = 256;
    std::vector<float> h_x(N, 10.0f), h_y(N, 20.0f);
    auto d_x = make_device(h_x);
    auto d_y = make_device(h_y);
    DeviceBuffer d_out;
    CUDA_CHECK(cudaMalloc(&d_out.ptr, N * sizeof(float)));

    int grid_x = (N + 127) / 128;
    void* args[] = {&d_x.ptr, &d_y.ptr, &d_out.ptr, &N};
    kernel.launch(dim3(grid_x), args, nullptr);

    auto h_out = download(d_out.ptr, N);
    for (int i = 0; i < N; ++i) {
        REQUIRE(h_out[i] == Catch::Approx(30.0f).margin(1e-6f));
    }
}

// ===========================================================================

TEST_CASE("JitKernel: load from JSON manifest",
          "[jit][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    TempDir tmp;

    // Write PTX file
    auto ptx_path = (tmp.path / "vector_add.ptx").string();
    {
        std::ofstream f(ptx_path);
        REQUIRE(f.is_open());
        f << VECTOR_ADD_PTX;
    }

    // Write JSON manifest pointing to the PTX file
    auto manifest_path = (tmp.path / "vector_add.json").string();
    {
        std::ofstream f(manifest_path);
        REQUIRE(f.is_open());
        f << R"({
            "name": "vector_add",
            "num_warps": 4,
            "shared_mem": 0,
            "ptx": "vector_add.ptx"
        })";
    }

    auto kernel = JitKernel::load_manifest(manifest_path);
    REQUIRE(kernel.meta().name == "vector_add");
    REQUIRE(kernel.meta().num_warps == 4);
    REQUIRE(kernel.meta().shared_mem_bytes == 0);

    // Quick launch to verify it actually runs
    int N = 128;
    std::vector<float> h_x(N, 7.0f), h_y(N, 3.0f);
    auto d_x = make_device(h_x);
    auto d_y = make_device(h_y);
    DeviceBuffer d_out;
    CUDA_CHECK(cudaMalloc(&d_out.ptr, N * sizeof(float)));

    void* args[] = {&d_x.ptr, &d_y.ptr, &d_out.ptr, &N};
    kernel.launch(dim3(1), args, nullptr);

    auto h_out = download(d_out.ptr, N);
    for (int i = 0; i < N; ++i) {
        REQUIRE(h_out[i] == Catch::Approx(10.0f).margin(1e-6f));
    }
}

// ===========================================================================

TEST_CASE("JitKernel: move semantics",
          "[jit]") {
    if (!cuda_available()) SKIP("CUDA not available");

    JitKernelMeta meta{"vector_add", 4, 0};
    auto k1 = JitKernel::load_ptx(
        VECTOR_ADD_PTX, std::strlen(VECTOR_ADD_PTX) + 1, meta);
    REQUIRE(k1.function() != nullptr);

    // Move construct
    JitKernel k2(std::move(k1));
    REQUIRE(k2.function() != nullptr);
    REQUIRE(k2.meta().name == "vector_add");
    REQUIRE(k1.function() == nullptr);  // NOLINT: testing moved-from state

    // Move assign
    JitKernel k3 = JitKernel::load_ptx(
        VECTOR_ADD_PTX, std::strlen(VECTOR_ADD_PTX) + 1, meta);
    k3 = std::move(k2);
    REQUIRE(k3.function() != nullptr);
    REQUIRE(k2.function() == nullptr);  // NOLINT: testing moved-from state
}

// ===========================================================================
// Triton-compiled RMSNorm kernel test
//
// Loads a pre-compiled Triton cubin via load_manifest + launch_triton and
// verifies correctness against a CPU reference.
//
// Prerequisites: run `python -c "from surogate.kernels.rmsnorm import
// compile_rmsnorm; compile_rmsnorm(C=768, output_dir='/tmp/surogate_rmsnorm_test')"`.
// ===========================================================================

TEST_CASE("JitKernel: launch_triton with Triton-compiled RMSNorm cubin",
          "[jit][triton][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    const std::string manifest_path =
        "/tmp/surogate_rmsnorm_test/rmsnorm_fwd_C768.json";
    if (!std::filesystem::exists(manifest_path)) {
        SKIP("Triton RMSNorm cubin not compiled. Run: python -c "
             "\"from surogate.kernels.rmsnorm import compile_rmsnorm; "
             "compile_rmsnorm(C=768, output_dir='/tmp/surogate_rmsnorm_test')\"");
    }

    auto kernel = JitKernel::load_manifest(manifest_path);
    REQUIRE(kernel.meta().name == "rmsnorm_fwd_kernel");
    REQUIRE(kernel.meta().num_warps == 4);
    REQUIRE(kernel.meta().shared_mem_bytes == 16);
    REQUIRE(kernel.meta().extra_null_params == 2);

    // Parameters
    int C = 768;
    int N = 32;
    float eps = 1e-5f;

    // Allocate host buffers
    std::vector<float> h_x_f32(N * C), h_w_f32(C);

    // Deterministic input: x[i] = sin(i * 0.01), w[j] = cos(j * 0.03)
    for (int i = 0; i < N * C; ++i) h_x_f32[i] = std::sin(i * 0.01f);
    for (int j = 0; j < C; ++j)     h_w_f32[j] = std::cos(j * 0.03f);

    // Convert to bf16 on host
    std::vector<__nv_bfloat16> h_x(N * C), h_w(C);
    for (int i = 0; i < N * C; ++i) h_x[i] = __float2bfloat16(h_x_f32[i]);
    for (int j = 0; j < C; ++j)     h_w[j] = __float2bfloat16(h_w_f32[j]);

    // Allocate device memory
    __nv_bfloat16 *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
    float* d_rstd = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, N * C * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_w, C * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, N * C * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_rstd, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, N * C * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_rstd, 0, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * C * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), C * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));

    // Kernel args (user params only — launch_triton appends scratch pointers)
    int stride_n = C;
    void* args[] = {&d_x, &d_w, &d_out, &d_rstd, &stride_n, &N, &eps};

    kernel.launch_triton(dim3(N), args, 7, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results
    std::vector<__nv_bfloat16> h_out(N * C);
    std::vector<float> h_rstd(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * C * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // CPU reference: out = (x * rrms) * w, rrms = rsqrt(mean(x^2) + eps)
    // Use the actual bf16 input values (after rounding) for the reference
    for (int row = 0; row < N; ++row) {
        float sum_sq = 0.0f;
        for (int j = 0; j < C; ++j) {
            float xv = __bfloat162float(h_x[row * C + j]);
            sum_sq += xv * xv;
        }
        float rrms = 1.0f / std::sqrt(sum_sq / C + eps);

        // Check rstd
        REQUIRE(h_rstd[row] == Catch::Approx(rrms).epsilon(1e-5f));

        // Check output
        for (int j = 0; j < C; ++j) {
            float xv = __bfloat162float(h_x[row * C + j]);
            float wv = __bfloat162float(h_w[j]);
            float ref = xv * rrms * wv;
            float actual = __bfloat162float(h_out[row * C + j]);
            // BF16 has ~0.4% relative error; use absolute margin for small values
            REQUIRE(actual == Catch::Approx(ref).margin(0.02f));
        }
    }

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_out);
    cudaFree(d_rstd);
}

// ===========================================================================
// CuTe DSL RMSNorm kernel test (via quack)
//
// Loads a pre-compiled CuTe DSL cubin via load_manifest (Library API path)
// and launches with structured CuTe tensor descriptor parameters.
//
// Prerequisites: run `python -c "from surogate.kernels.cute_rmsnorm import
// compile_cute_rmsnorm; compile_cute_rmsnorm(C=768,
// output_dir='/tmp/surogate_cute_rmsnorm_test')"`.
// ===========================================================================

namespace {

// CuTe tensor descriptor parameter structs.
// These must exactly match the compiled kernel's parameter layout
// (see cute_rmsnorm.py manifest "params" section).

// param_0: mX — input [M, N] bf16 tensor descriptor (24 bytes)
//   +0: void*  data_ptr
//   +8: int32  M (dynamic row count)
//  +12: int32  (padding)
//  +16: int64  stride_row (in elements, = N for contiguous)
struct CuteTensorDesc2D {
    void*    data_ptr;
    int32_t  dim0;
    int32_t  _pad0;
    int64_t  stride_row;
};
static_assert(sizeof(CuteTensorDesc2D) == 24);

// param_1: mW — weight [N] bf16 tensor descriptor (8 bytes)
//   +0: void*  data_ptr
struct CuteTensorDesc1D_8 {
    void* data_ptr;
};
static_assert(sizeof(CuteTensorDesc1D_8) == 8);

// param_3: mRstd — rstd [M] fp32 tensor descriptor (16 bytes)
//   +0: void*  data_ptr
//   +8: int64  (padding / unused stride)
struct CuteTensorDesc1D_16 {
    void*   data_ptr;
    int64_t _pad0;
};
static_assert(sizeof(CuteTensorDesc1D_16) == 16);

}  // namespace

TEST_CASE("JitKernel: CuTe DSL RMSNorm via Library API (quack)",
          "[jit][cute][gpu]") {
    if (!cuda_available()) SKIP("CUDA not available");

    const std::string manifest_path =
        "/tmp/surogate_cute_rmsnorm_test/cute_rmsnorm_bf16_C768.json";
    if (!std::filesystem::exists(manifest_path)) {
        SKIP("CuTe RMSNorm cubin not compiled. Run: python -c "
             "\"from surogate.kernels.cute_rmsnorm import compile_cute_rmsnorm; "
             "compile_cute_rmsnorm(C=768, output_dir='/tmp/surogate_cute_rmsnorm_test')\"");
    }

    auto kernel = JitKernel::load_manifest(manifest_path);
    REQUIRE(kernel.meta().num_warps == 4);
    REQUIRE(kernel.meta().shared_mem_bytes == 6160);
    REQUIRE(kernel.function() != nullptr);

    // Parameters
    const int C = 768;
    const int M = 32;    // rows (batch)
    const float eps = 1e-5f;

    // Allocate host buffers
    std::vector<float> h_x_f32(M * C), h_w_f32(C);

    // Deterministic input: x[i] = sin(i * 0.01), w[j] = cos(j * 0.03)
    for (int i = 0; i < M * C; ++i) h_x_f32[i] = std::sin(i * 0.01f);
    for (int j = 0; j < C; ++j)     h_w_f32[j] = std::cos(j * 0.03f);

    // Convert to bf16 on host
    std::vector<__nv_bfloat16> h_x(M * C), h_w(C);
    for (int i = 0; i < M * C; ++i) h_x[i] = __float2bfloat16(h_x_f32[i]);
    for (int j = 0; j < C; ++j)     h_w[j] = __float2bfloat16(h_w_f32[j]);

    // Allocate device memory
    __nv_bfloat16 *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
    float* d_rstd = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, M * C * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_w, C * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, M * C * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_rstd, M * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, M * C * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_rstd, 0, M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), M * C * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), C * sizeof(__nv_bfloat16),
                           cudaMemcpyHostToDevice));

    // Build CuTe tensor descriptor parameters
    // These structs are passed by-value through cuLaunchKernel's kernelParams.
    CuteTensorDesc2D param_x{};
    param_x.data_ptr   = d_x;
    param_x.dim0       = M;
    param_x._pad0      = 0;
    param_x.stride_row = C;  // contiguous: stride = hidden_dim

    CuteTensorDesc1D_8 param_w{};
    param_w.data_ptr = d_w;

    CuteTensorDesc2D param_o{};
    param_o.data_ptr   = d_out;
    param_o.dim0       = 0;   // padding (not used by kernel)
    param_o._pad0      = 0;
    param_o.stride_row = C;

    CuteTensorDesc1D_16 param_rstd{};
    param_rstd.data_ptr = d_rstd;
    param_rstd._pad0    = 0;

    float param_eps = eps;

    void* args[] = {&param_x, &param_w, &param_o, &param_rstd, &param_eps};

    // Grid: ceil(M / rows_per_block). For C=768 bf16: rows_per_block = 4.
    int rows_per_block = 4;
    int grid_x = (M + rows_per_block - 1) / rows_per_block;

    // Launch with explicit block (num_threads) and shared memory from manifest
    kernel.launch(dim3(grid_x), dim3(128, 1, 1),
                  kernel.meta().shared_mem_bytes, args, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results
    std::vector<__nv_bfloat16> h_out(M * C);
    std::vector<float> h_rstd(M);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, M * C * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, M * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // CPU reference: out = (x * rrms) * w, rrms = rsqrt(mean(x^2) + eps)
    for (int row = 0; row < M; ++row) {
        float sum_sq = 0.0f;
        for (int j = 0; j < C; ++j) {
            float xv = __bfloat162float(h_x[row * C + j]);
            sum_sq += xv * xv;
        }
        float rrms = 1.0f / std::sqrt(sum_sq / C + eps);

        // Check rstd
        REQUIRE(h_rstd[row] == Catch::Approx(rrms).epsilon(1e-5f));

        // Check output
        for (int j = 0; j < C; ++j) {
            float xv = __bfloat162float(h_x[row * C + j]);
            float wv = __bfloat162float(h_w[j]);
            float ref = xv * rrms * wv;
            float actual = __bfloat162float(h_out[row * C + j]);
            REQUIRE(actual == Catch::Approx(ref).margin(0.02f));
        }
    }

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_out);
    cudaFree(d_rstd);
}
