// CPU expert compute throughput: expert bytes per second the host reaches on the Flash-Next
// geometry (2560/512/10/640) from a synthetic planar W8 bank, for decode-shaped rounds
// (few tokens, many distinct experts) with the pool's thread count.
//
//   ninfer_cpu_expert_compute_bench [threads] [experts_in_bank] [jobs_per_round] [rounds] [experts_per_round]
//   (experts_per_round < experts_in_bank draws each round's experts from a subset, i.e. several
//   tokens per expert as in a prefill round; default = the whole bank)
#include "api/ops/cpu_expert_compute.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;

int main(int argc, char** argv) {
    const unsigned threads    = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 0;
    const int bank_experts    = argc > 2 ? std::atoi(argv[2]) : 512;
    const int jobs_per_round  = argc > 3 ? std::atoi(argv[3]) : 160; // ~16 tokens × 10 paths
    const int rounds          = argc > 4 ? std::atoi(argv[4]) : 20;
    const int experts_per_round = argc > 5 ? std::atoi(argv[5]) : bank_experts;
    const ops::SparseMoeGeometry geometry{2560, bank_experts, 10, 640};
    const std::size_t gate_codes  = static_cast<std::size_t>(geometry.expert_rows()) * geometry.hidden;
    const std::size_t gate_scales = static_cast<std::size_t>(geometry.expert_rows()) * (geometry.hidden / 32) * 2;
    const std::size_t down_codes  = static_cast<std::size_t>(geometry.hidden) * geometry.intermediate;
    const std::size_t down_scales = static_cast<std::size_t>(geometry.hidden) * (geometry.intermediate / 32) * 2;
    const std::size_t per_expert  = gate_codes + gate_scales + down_codes + down_scales;
    std::cout << "bank: " << bank_experts << " experts x " << per_expert / 1e6 << " MB = "
              << static_cast<double>(per_expert) * bank_experts / 1e9 << " GB; avx512 "
              << (ops::cpu_expert_compute_has_avx512() ? "yes" : "no") << " vnni " << (ops::cpu_expert_compute_has_vnni() ? "yes" : "no") << " tile " << (ops::cpu_expert_compute_has_tile() ? "yes" : "no") << "\n";

    std::mt19937 rng(7);
    std::vector<std::byte> gc(gate_codes * bank_experts), gs(gate_scales * bank_experts),
        dc(down_codes * bank_experts), ds(down_scales * bank_experts);
    for (auto* v : {&gc, &dc}) {
        for (auto& b : *v) { b = static_cast<std::byte>(rng() & 0xFF); }
    }
    for (auto* v : {&gs, &ds}) { // fp16 scales around 0.01
        for (std::size_t i = 0; i + 1 < v->size(); i += 2) {
            (*v)[i]     = static_cast<std::byte>(0x1F);
            (*v)[i + 1] = static_cast<std::byte>(0x21);
        }
    }
    const ops::CpuExpertBank bank{gc.data(), gs.data(), dc.data(), ds.data()};

    const int tokens = std::max(1, jobs_per_round / geometry.experts_per_token);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(geometry.hidden) * tokens);
    for (auto& v : x) { v = static_cast<std::uint16_t>(0x3F00 + (rng() & 0xFF)); }
    std::vector<float> out(static_cast<std::size_t>(geometry.hidden) * tokens);
    std::vector<ops::CpuExpertJob> jobs(static_cast<std::size_t>(jobs_per_round));

    ops::CpuExpertPool pool(geometry, {.threads = threads, .pin_threads = true});
    std::cout << "threads: " << pool.threads() << ", tokens/round: " << tokens << ", jobs/round: "
              << jobs_per_round << ", distinct experts/round <= " << experts_per_round << "\n";
    double best_gbs = 0.0, total_s = 0.0;
    for (int r = 0; r < rounds; ++r) {
        for (int j = 0; j < jobs_per_round; ++j) {
            jobs[static_cast<std::size_t>(j)] = {j % tokens, static_cast<int>((r * 131 + rng() % experts_per_round) % bank_experts), 0.1F};
        }
        std::fill(out.begin(), out.end(), 0.0F);
        const auto t0 = std::chrono::steady_clock::now();
        ops::CpuExpertRound round{x.data(), out.data(), tokens, jobs};
        pool.run(bank, round);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double gbs = static_cast<double>(per_expert) * jobs_per_round / s / 1e9;
        best_gbs = std::max(best_gbs, gbs);
        total_s += s;
    }
    std::cout << "per job: " << total_s / rounds / jobs_per_round * 1e6 * pool.threads() << " us thread-time\n";
    std::cout << "expert bytes: best " << best_gbs << " GB/s, mean "
              << static_cast<double>(per_expert) * jobs_per_round * rounds / total_s / 1e9
              << " GB/s over " << rounds << " rounds (" << total_s / rounds * 1e3 << " ms/round)\n";
    return 0;
}
