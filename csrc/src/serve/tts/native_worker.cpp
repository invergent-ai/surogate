// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {
bool have_avx512() {
#if defined(__x86_64__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

void* load(const std::filesystem::path& path, int flags) {
    void* library = dlopen(path.c_str(), flags);
    if (!library) throw std::runtime_error(dlerror());
    return library;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 9)
            throw std::runtime_error(
                "Expected MODEL CODEC JOBS OUTPUT THREADS CODEC_THREADS KERNELS cpu|cuda");
        auto root   = std::filesystem::canonical(argv[1]).parent_path();
        auto binary = std::filesystem::read_symlink("/proc/self/exe").parent_path();
        const std::string device = argv[8];
        if (device != "cpu" && device != "cuda") throw std::runtime_error("Invalid TTS device");
        std::ifstream profile_file(root / "voices.json");
        auto profile = nlohmann::json::parse(profile_file);
        // The model download verifies these files before execution. This ABI
        // fingerprint prevents pairing our adapter with an unrelated runtime: the CPU runtime, or
        // the same runtime built with CUDA for SM120 (a package's GPU variant, whose lib/ also
        // holds libggml-cuda). Both share one GGML base build.
        constexpr const char* cpu_runtime =
            "139c84abd9973d48c26112e51b310680ac13e1df4d4386cd875f40a7893d44a2";
        constexpr const char* cpu_base =
            "5a46b8f5f84dfd5f1e86730ee9fd280e0acf6649e77c6519a3182f44b5b196de";
        // The GPU variant's runtime: nemo-speech-cpp 07003daa with the package's longform patch,
        // built with CUDA 13 for sm_120 (RTX 50-series, compute capability 12.0), without NCCL,
        // with source paths mapped and an $ORIGIN RUNPATH (docs/inference/tts.md).
        constexpr const char* cuda_runtime = "02fa87a8479ccde6b188b204c0086e87624034e5fb177c94b5ab0c2bb666e395";
        constexpr const char* cuda_base    = "5916a79c9ece31dcbef048992230d1edd75c585c5c8534fc4f94100a40639718";
        constexpr const char* cuda_backend = "285f960f40ade7e1ec7361d74ebbbb77809b8bfc2fdc34a355876a564c55dc2f";
        const auto& files   = profile.at("files");
        const auto& runtime = files.at("lib/libnemo_speech_tts.so.1");
        const auto& base    = files.at("lib/libggml-base.so.0");
        const bool cpu_package  = runtime == cpu_runtime && base == cpu_base;
        const bool cuda_package = runtime == cuda_runtime && base == cuda_base;
        if (!cpu_package && !cuda_package)
            throw std::runtime_error("This voice export uses an unsupported native runtime ABI");
        if (cuda_package && (!files.contains("lib/libggml-cuda.so.0") ||
                             files.at("lib/libggml-cuda.so.0") != cuda_backend))
            throw std::runtime_error(
                "This TTS package's CUDA backend (lib/libggml-cuda.so.0) is not the one its runtime "
                "was built with");
        if (device == "cuda" && !cuda_package)
            throw std::runtime_error(
                "This TTS package has only the CPU runtime; serving on a GPU needs its GPU variant");
        std::string mode = argv[7];
        if (mode != "auto" && mode != "optimized" && mode != "reference")
            throw std::runtime_error("Invalid CPU kernel selection");
        auto optimized = binary / "tts-kernels/libggml-cpu.so.0";
        // On a GPU the model runs in CUDA, and the CPU kernels only matter on the CPU; the CUDA
        // runtime is built against its own GGML, so they are never loaded beside it.
        bool available = device == "cpu" && cpu_package && have_avx512() &&
                         std::filesystem::is_regular_file(optimized);
        if (mode == "optimized" && device == "cpu" && !available)
            throw std::runtime_error(
                "Optimized CPU kernels require the AVX-512 build and compatible CPU");
        // Load the optimized SONAME before the model library resolves GGML.
        void* kernels = nullptr;
        if (mode != "reference" && available) kernels = load(optimized, RTLD_NOW | RTLD_GLOBAL);
        if (device == "cpu")
            std::cerr << "TTS CPU kernels: " << (kernels ? "optimized AVX-512" : "reference") << '\n';
        void* engine = load(root / "lib/libnemo_speech_tts.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (device == "cuda") {
            // With no CUDA device visible the runtime would quietly run on the CPU; refuse instead.
            using ByType = void* (*)(int);
            auto by_type = reinterpret_cast<ByType>(dlsym(engine, "ggml_backend_dev_by_type"));
            constexpr int gpu = 1; // GGML_BACKEND_DEVICE_TYPE_GPU
            if (by_type == nullptr || by_type(gpu) == nullptr)
                throw std::runtime_error("No CUDA device is visible to the TTS worker");
        }
        void* bridge = load(binary / "libsinfer_tts_bridge.so", RTLD_NOW | RTLD_LOCAL);
        auto run =
            reinterpret_cast<int (*)(int, char**)>(dlsym(bridge, "surogate_tts_worker_main"));
        if (!run) throw std::runtime_error(dlerror());
        char* bridge_argv[] = {argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[8], nullptr};
        int result = run(8, bridge_argv);
        dlclose(bridge);
        dlclose(engine);
        if (kernels) dlclose(kernels);
        return result;
    } catch (const std::exception& e) {
        std::cerr << "surogate-tts-worker: " << e.what() << '\n';
        return 1;
    }
}
