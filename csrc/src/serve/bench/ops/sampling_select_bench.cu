// Qualification benchmark for Qwen3.6-35B G2 sampling and G3 MTP accept.
//
//   ./ninfer_sampling_select_bench --sample --batch 8 --mode stochastic
//   ./ninfer_sampling_select_bench --mtp --mode stochastic --mtp-k 5
//   ./ninfer_sampling_select_bench --matrix
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kPhysicalRows = 248320;
constexpr std::int32_t kTokenDomain  = 248077;

enum class Mode {
    Greedy,
    Stochastic,
};

struct Options {
    bool sample        = false;
    bool mtp           = false;
    bool matrix        = false;
    bool counts_active = true;
    Mode mode          = Mode::Stochastic;
    int batch          = 1;
    int mtp_k          = 3;
    int top_k          = 20;
};

void usage(const char* argv0) {
    std::printf("usage: %s [--sample|--mtp|--matrix] [--mode greedy|stochastic] "
                "[--batch 1..8] [--mtp-k 1..5] [--top-k 1..20] [--no-counts]\n",
                argv0);
}

int parse_int(std::string_view value, const char* name) {
    try {
        std::size_t parsed = 0;
        const int out      = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) { throw std::invalid_argument("trailing"); }
        return out;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " expects an integer");
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto need_value = [&](const char* name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return argv[++i];
        };
        if (arg == "--sample") {
            options.sample = true;
        } else if (arg == "--mtp") {
            options.mtp = true;
        } else if (arg == "--matrix") {
            options.matrix = true;
        } else if (arg == "--mode") {
            const std::string_view mode = need_value("--mode");
            if (mode == "greedy") {
                options.mode = Mode::Greedy;
            } else if (mode == "stochastic") {
                options.mode = Mode::Stochastic;
            } else {
                throw std::invalid_argument("--mode must be greedy or stochastic");
            }
        } else if (arg == "--batch") {
            options.batch = parse_int(need_value("--batch"), "--batch");
        } else if (arg == "--mtp-k") {
            options.mtp_k = parse_int(need_value("--mtp-k"), "--mtp-k");
        } else if (arg == "--top-k") {
            options.top_k = parse_int(need_value("--top-k"), "--top-k");
        } else if (arg == "--no-counts") {
            options.counts_active = false;
        } else if (arg == "-h" || arg == "--help") {
            usage(argc > 0 ? argv[0] : "ninfer_sampling_select_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (!options.sample && !options.mtp && !options.matrix) { options.matrix = true; }
    if (options.matrix && (options.sample || options.mtp)) {
        throw std::invalid_argument("--matrix cannot be combined with --sample or --mtp");
    }
    if (options.mtp_k < 1 || options.mtp_k > 5) {
        throw std::invalid_argument("--mtp-k must be in [1,5]");
    }
    if (options.batch < 1 || options.batch > 8) {
        throw std::invalid_argument("--batch must be in [1,8]");
    }
    if (options.top_k < 1 || options.top_k > 20) {
        throw std::invalid_argument("--top-k must be in [1,20]");
    }
    return options;
}

DeviceBuffer make_logits(int cols) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(kPhysicalRows) * cols);
    for (int col = 0; col < cols; ++col) {
        const int hot = (17 + col * 7919) % kTokenDomain;
        for (int row = 0; row < kPhysicalRows; ++row) {
            float value = -8.0f + static_cast<float>((row * 17 + col * 31) % 4096) / 4096.0f;
            if (row == hot) { value = 8.0f; }
            if (row >= kTokenDomain) { value = 20.0f; }
            host[static_cast<std::size_t>(col) * kPhysicalRows + row] = f32_to_bf16(value);
        }
    }
    DeviceBuffer device(host.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(device.p, host.data(), device.bytes, cudaMemcpyHostToDevice));
    return device;
}

DeviceBuffer make_i32(const std::vector<std::int32_t>& host) {
    DeviceBuffer device(host.size() * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpy(device.p, host.data(), device.bytes, cudaMemcpyHostToDevice));
    return device;
}

DeviceBuffer make_config(DeviceBuffer& counts, Mode mode, bool counts_active, int top_k) {
    ops::SamplingConfig config;
    config.temperature      = mode == Mode::Greedy ? 0.0f : 0.6f;
    config.top_k            = top_k;
    config.top_p            = 0.95f;
    config.presence_penalty = mode == Mode::Stochastic && counts_active ? 1.0f : 0.0f;
    config.seed             = 20260716ull;
    config.token_counts =
        mode == Mode::Stochastic && counts_active ? static_cast<std::int32_t*>(counts.p) : nullptr;

    DeviceBuffer device(sizeof(ops::SamplingConfig));
    CUDA_CHECK(cudaMemcpy(device.p, &config, sizeof(config), cudaMemcpyHostToDevice));
    return device;
}

DeviceBuffer make_batch_configs(DeviceBuffer& counts, int batch, Mode mode, bool counts_active,
                                int top_k) {
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        ops::SamplingConfig& config = configs[static_cast<std::size_t>(row)];
        config.temperature          = mode == Mode::Greedy ? 0.0f : 0.6f;
        config.top_k                = top_k;
        config.top_p                = 0.95f;
        config.presence_penalty     = mode == Mode::Stochastic && counts_active ? 1.0f : 0.0f;
        config.seed                 = 20260716ull + static_cast<unsigned long long>(row);
        config.token_counts         = mode == Mode::Stochastic && counts_active
                                          ? static_cast<std::int32_t*>(counts.p) +
                                        static_cast<std::size_t>(row) * kTokenDomain
                                          : nullptr;
    }
    DeviceBuffer device(configs.size() * sizeof(ops::SamplingConfig));
    CUDA_CHECK(cudaMemcpy(device.p, configs.data(), device.bytes, cudaMemcpyHostToDevice));
    return device;
}

double stochastic_payload_bytes(int cols, bool counts_active) {
    const double per_col = static_cast<double>(kTokenDomain) * 2.0 +
                           (counts_active ? static_cast<double>(kTokenDomain) * 4.0 : 0.0);
    return per_col * static_cast<double>(cols);
}

void run_sample(DeviceBuffer& logits, DeviceBuffer& counts, int batch, Mode mode,
                bool counts_active, int top_k) {
    CUDA_CHECK(cudaMemset(counts.p, 0, counts.bytes));
    DeviceBuffer configs = make_batch_configs(counts, batch, mode, counts_active, top_k);
    DeviceBuffer out(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    std::vector<std::int32_t> positions(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) { positions[static_cast<std::size_t>(row)] = 4096 + row; }
    DeviceBuffer device_positions = make_i32(positions);
    Tensor tlogits(logits.p, DType::BF16, {kPhysicalRows, batch});
    Tensor tout(out.p, DType::I32, {batch});
    Tensor tpositions(device_positions.p, DType::I32, {batch});
    WorkspaceArena workspace(ops::sampling_workspace_capacity_bytes(kTokenDomain, batch, batch));
    const auto* config_ptr = static_cast<const ops::SamplingConfig*>(configs.p);

    const double bytes  = mode == Mode::Greedy ? static_cast<double>(kTokenDomain) * 2.0 * batch
                                               : stochastic_payload_bytes(batch, counts_active);
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            ops::sample(tlogits, tout, kTokenDomain, config_ptr, tpositions,
                        ops::kSamplePurposeDecode, workspace, stream);
        },
        bytes);
    const std::string label = std::string("G2 B=") + std::to_string(batch) + " " +
                              (mode == Mode::Greedy ? "greedy" : "stochastic") +
                              (counts_active && mode == Mode::Stochastic ? " counts" : "") +
                              " top_k=" + std::to_string(top_k);
    print_result(label.c_str(), result);
}

void run_mtp(DeviceBuffer& logits, DeviceBuffer& counts, int k, Mode mode, bool counts_active,
             int top_k) {
    CUDA_CHECK(cudaMemset(counts.p, 0, counts.bytes));
    DeviceBuffer config = make_config(counts, mode, counts_active, top_k);
    std::vector<std::int32_t> target_host(static_cast<std::size_t>(k + 1));
    std::vector<std::int32_t> draft_host(static_cast<std::size_t>(k));
    for (int col = 0; col <= k; ++col) {
        target_host[static_cast<std::size_t>(col)] = (17 + col * 7919) % kTokenDomain;
        if (col < k) {
            draft_host[static_cast<std::size_t>(col)] = target_host[static_cast<std::size_t>(col)];
        }
    }
    DeviceBuffer targets = make_i32(target_host);
    DeviceBuffer drafts  = make_i32(draft_host);
    DeviceBuffer length  = make_i32({128});
    DeviceBuffer token   = make_i32({-1});
    DeviceBuffer sampled(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    DeviceBuffer num(sizeof(std::int32_t));
    DeviceBuffer accepted(sizeof(std::int32_t));
    DeviceBuffer extent = make_i32({k});

    Tensor ttargets(targets.p, DType::I32, {k + 1});
    Tensor tlogits(logits.p, DType::BF16, {kPhysicalRows, k + 1});
    Tensor tdrafts(drafts.p, DType::I32, {k});
    Tensor textent(extent.p, DType::I32, {1});
    Tensor tlength(length.p, DType::I32, {1});
    Tensor ttoken(token.p, DType::I32, {1});
    Tensor tsampled(sampled.p, DType::I32, {k + 1});
    Tensor tnum(num.p, DType::I32, {1});
    Tensor taccepted(accepted.p, DType::I32, {1});
    WorkspaceArena workspace(
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(kTokenDomain, k, k, 1, 1));
    const auto* config_ptr = static_cast<const ops::SamplingConfig*>(config.p);

    const double bytes  = mode == Mode::Greedy ? static_cast<double>((k + 1) * 4 + k * 4)
                                               : stochastic_payload_bytes(k + 1, counts_active);
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            ops::speculative_accept_greedy_drafts(ttargets, tlogits, tdrafts, textent, tlength,
                                                  ttoken, tsampled, tnum, taccepted, kTokenDomain,
                                                  config_ptr, workspace, stream);
        },
        bytes);
    const std::string label = std::string("G3 K=") + std::to_string(k) + " " +
                              (mode == Mode::Greedy ? "greedy" : "stochastic") +
                              (counts_active && mode == Mode::Stochastic ? " counts" : "") +
                              " top_k=" + std::to_string(top_k);
    print_result(label.c_str(), result);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    try {
        const Options options = parse_args(argc, argv);
        const int max_cols    = options.matrix ? 8
                                               : std::max(options.sample ? options.batch : 1,
                                                       options.mtp ? options.mtp_k + 1 : 1);
        DeviceBuffer logits   = make_logits(max_cols);
        const int count_rows  = options.matrix ? 8 : (options.sample ? options.batch : 1);
        DeviceBuffer counts(static_cast<std::size_t>(kTokenDomain) * count_rows *
                            sizeof(std::int32_t));
        CUDA_CHECK(cudaMemset(counts.p, 0, counts.bytes));

        std::printf("payload: physical_rows=%d token_domain=%d logits=%.3f MiB/row "
                    "counts=%.3f MiB/row\n",
                    kPhysicalRows, kTokenDomain, static_cast<double>(kTokenDomain * 2) / 1048576.0,
                    static_cast<double>(kTokenDomain * 4) / 1048576.0);
        std::printf(
            "workspace: lanes=1 %.1f KiB, lanes=8 %.1f KiB\n",
            static_cast<double>(ops::sampling_workspace_capacity_bytes(kTokenDomain, 1, 1)) /
                1024.0,
            static_cast<double>(ops::sampling_workspace_capacity_bytes(kTokenDomain, 8, 8)) /
                1024.0);
        if (options.matrix) {
            for (const int batch : {1, 2, 4, 8}) {
                run_sample(logits, counts, batch, Mode::Greedy, false, 1);
                run_sample(logits, counts, batch, Mode::Stochastic, true, 20);
            }
            run_sample(logits, counts, 1, Mode::Stochastic, false, 20);
            for (int k = 1; k <= 5; ++k) {
                run_mtp(logits, counts, k, Mode::Greedy, false, 1);
                run_mtp(logits, counts, k, Mode::Stochastic, true, 20);
            }
        } else {
            if (options.sample) {
                run_sample(logits, counts, options.batch, options.mode, options.counts_active,
                           options.top_k);
            }
            if (options.mtp) {
                run_mtp(logits, counts, options.mtp_k, options.mode, options.counts_active,
                        options.top_k);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_sampling_select_bench: %s\n", e.what());
        return 2;
    }
    return 0;
}
