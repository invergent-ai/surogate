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
        if (argc != 8)
            throw std::runtime_error(
                "Expected MODEL CODEC JOBS OUTPUT THREADS CODEC_THREADS KERNELS");
        auto root   = std::filesystem::canonical(argv[1]).parent_path();
        auto binary = std::filesystem::read_symlink("/proc/self/exe").parent_path();
        std::ifstream profile_file(root / "voices.json");
        auto profile = nlohmann::json::parse(profile_file);
        // The model download verifies these files before execution. This ABI
        // fingerprint prevents pairing our adapter with an unrelated runtime.
        const auto& files = profile.at("files");
        if (files.at("lib/libnemo_speech_tts.so.1") !=
                "139c84abd9973d48c26112e51b310680ac13e1df4d4386cd875f40a7893d44a2" ||
            files.at("lib/libggml-base.so.0") !=
                "5a46b8f5f84dfd5f1e86730ee9fd280e0acf6649e77c6519a3182f44b5b196de")
            throw std::runtime_error("This voice export uses an unsupported native runtime ABI");
        std::string mode = argv[7];
        if (mode != "auto" && mode != "optimized" && mode != "reference")
            throw std::runtime_error("Invalid CPU kernel selection");
        auto optimized = binary / "tts-kernels/libggml-cpu.so.0";
        bool available = have_avx512() && std::filesystem::is_regular_file(optimized);
        if (mode == "optimized" && !available)
            throw std::runtime_error(
                "Optimized CPU kernels require the AVX-512 build and compatible CPU");
        // Load the optimized SONAME before the model library resolves GGML.
        void* kernels = nullptr;
        if (mode != "reference" && available) kernels = load(optimized, RTLD_NOW | RTLD_GLOBAL);
        std::cerr << "TTS CPU kernels: " << (kernels ? "optimized AVX-512" : "reference") << '\n';
        void* engine = load(root / "lib/libnemo_speech_tts.so.1", RTLD_NOW | RTLD_GLOBAL);
        void* bridge = load(binary / "libsinfer_tts_bridge.so", RTLD_NOW | RTLD_LOCAL);
        auto run =
            reinterpret_cast<int (*)(int, char**)>(dlsym(bridge, "surogate_tts_worker_main"));
        if (!run) throw std::runtime_error(dlerror());
        int result = run(7, argv);
        dlclose(bridge);
        dlclose(engine);
        if (kernels) dlclose(kernels);
        return result;
    } catch (const std::exception& e) {
        std::cerr << "surogate-tts-worker: " << e.what() << '\n';
        return 1;
    }
}
