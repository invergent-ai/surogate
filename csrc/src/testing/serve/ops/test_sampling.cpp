// Public-contract qualification for sample().
//
// The deterministic branch is checked exactly against an independent CPU
// argmax.  The stochastic branch is checked against one FP64 mathematical
// distribution oracle built from the BF16 values represented at the public
// input.  The test never reproduces the device RNG algorithm or uses another
// production path as a golden.
#include "api/ops/sampling.h"
#include "api/ops/sampled_logprob.h"
#include "ops/common/sampling_workspace.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

struct Candidate {
    double adjusted = 0.0;
    int token       = 0;
};

struct Distribution {
    std::vector<int> tokens;
    std::vector<double> probabilities;
};

struct RunResult {
    std::vector<int> tokens;
    std::vector<std::vector<int>> counts;
    int integrity_failures = 0;
};

bool same_config(const ops::SamplingConfig& a, const ops::SamplingConfig& b) {
    return a.temperature == b.temperature && a.top_k == b.top_k && a.top_p == b.top_p &&
           a.min_p == b.min_p && a.presence_penalty == b.presence_penalty &&
           a.frequency_penalty == b.frequency_penalty && a.seed == b.seed &&
           a.token_counts == b.token_counts && a.logit_bias == b.logit_bias &&
           a.token_bitmask == b.token_bitmask;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<float> repeat_column(const std::vector<float>& column, int columns) {
    std::vector<float> logits(column.size() * static_cast<std::size_t>(columns));
    for (int t = 0; t < columns; ++t) {
        std::copy(column.begin(), column.end(),
                  logits.begin() + static_cast<std::ptrdiff_t>(t) * column.size());
    }
    return logits;
}

std::vector<int> greedy_oracle(const std::vector<float>& logits, int physical_rows,
                               int token_domain, int columns) {
    std::vector<int> expected(static_cast<std::size_t>(columns));
    for (int t = 0; t < columns; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * physical_rows;
        int best               = 0;
        for (int token = 1; token < token_domain; ++token) {
            if (logits[base + token] > logits[base + best]) { best = token; }
        }
        expected[static_cast<std::size_t>(t)] = best;
    }
    return expected;
}

Distribution distribution_oracle(const std::vector<float>& column, int token_domain,
                                 const ops::SamplingConfig& config,
                                 const std::vector<int>* counts = nullptr) {
    std::vector<Candidate> candidates(static_cast<std::size_t>(token_domain));
    for (int token = 0; token < token_domain; ++token) {
        const int count = counts == nullptr ? 0 : (*counts)[static_cast<std::size_t>(token)];
        double adjusted = static_cast<double>(column[static_cast<std::size_t>(token)]);
        if (count > 0) { adjusted -= static_cast<double>(config.presence_penalty); }
        adjusted -= static_cast<double>(config.frequency_penalty) * static_cast<double>(count);
        candidates[static_cast<std::size_t>(token)] = {adjusted, token};
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.adjusted != b.adjusted) { return a.adjusted > b.adjusted; }
        return a.token < b.token;
    });

    // A row that asked for no limit and no nucleus is drawn from the whole
    // vocabulary; every other row keeps the pipeline's candidate cap.
    int cap = config.top_k > 0 ? config.top_k : token_domain;
    cap = std::min(cap, token_domain);
    candidates.resize(static_cast<std::size_t>(cap));

    std::vector<double> weights(static_cast<std::size_t>(cap));
    const double max_scaled = candidates.front().adjusted / config.temperature;
    double total_weight     = 0.0;
    for (int rank = 0; rank < cap; ++rank) {
        const double weight = std::exp(
            candidates[static_cast<std::size_t>(rank)].adjusted / config.temperature - max_scaled);
        weights[static_cast<std::size_t>(rank)] = weight;
        if (config.min_p <= 0.0F || weight >= config.min_p) { total_weight += weight; }
    }

    const bool use_min_p     = config.min_p > 0.0f;
    const bool use_top_p     = config.top_p < 1.0f;
    const double min_weight  = static_cast<double>(config.min_p) * weights.front();
    const double top_p_limit = static_cast<double>(config.top_p) * total_weight;
    double cumulative        = 0.0;
    int support              = 0;
    for (int rank = 0; rank < cap; ++rank) {
        if (use_min_p && weights[static_cast<std::size_t>(rank)] < min_weight) { break; }
        cumulative += weights[static_cast<std::size_t>(rank)];
        support = rank + 1;
        if (use_top_p && cumulative >= top_p_limit) { break; }
    }
    support = std::max(support, 1);

    Distribution out;
    out.tokens.reserve(static_cast<std::size_t>(support));
    out.probabilities.reserve(static_cast<std::size_t>(support));
    double kept_weight = 0.0;
    for (int rank = 0; rank < support; ++rank) {
        kept_weight += weights[static_cast<std::size_t>(rank)];
    }
    for (int rank = 0; rank < support; ++rank) {
        out.tokens.push_back(candidates[static_cast<std::size_t>(rank)].token);
        out.probabilities.push_back(weights[static_cast<std::size_t>(rank)] / kept_weight);
    }
    return out;
}

RunResult run_batch(const std::vector<float>& logits, int physical_rows, int token_domain,
                    std::vector<ops::SamplingConfig> configs,
                    const std::vector<int>& logical_positions, int purpose,
                    const std::vector<std::vector<int>>& initial_counts = {}) {
    const int batch = static_cast<int>(configs.size());
    if (batch <= 0 || logical_positions.size() != configs.size() ||
        logits.size() != static_cast<std::size_t>(physical_rows) * configs.size() ||
        (!initial_counts.empty() && initial_counts.size() != configs.size())) {
        throw std::invalid_argument("invalid sample batch fixture");
    }
    const std::vector<std::uint16_t> input_bits = bf16_bits(logits);
    DeviceBuffer device_logits                  = to_device(input_bits);
    GuardedDeviceBuffer device_out(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    const std::vector<int> output_sentinel(static_cast<std::size_t>(batch), -777777);
    device_out.copy_from_host(output_sentinel.data(),
                              output_sentinel.size() * sizeof(std::int32_t));

    std::vector<std::unique_ptr<GuardedDeviceBuffer>> device_counts;
    if (!initial_counts.empty()) {
        device_counts.reserve(configs.size());
        for (std::size_t row = 0; row < configs.size(); ++row) {
            if (initial_counts[row].size() != static_cast<std::size_t>(token_domain)) {
                throw std::invalid_argument("invalid sample token-count fixture");
            }
            auto counts = std::make_unique<GuardedDeviceBuffer>(initial_counts[row].size() *
                                                                sizeof(std::int32_t));
            counts->copy_from_host(initial_counts[row].data(), counts->bytes());
            configs[row].token_counts = static_cast<std::int32_t*>(counts->data());
            device_counts.push_back(std::move(counts));
        }
    }
    const std::vector<ops::SamplingConfig> expected_configs = configs;
    DeviceBuffer device_configs                             = to_device(configs);
    DeviceBuffer device_positions                           = to_device(logical_positions);

    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, batch});
    Tensor out_tensor(device_out.data(), DType::I32, {batch});
    Tensor positions_tensor(device_positions.p, DType::I32, {batch});
    const std::size_t workspace_bytes =
        ops::sampling_workspace_capacity_bytes(token_domain, batch, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::sample(logits_tensor, out_tensor, token_domain,
                static_cast<const ops::SamplingConfig*>(device_configs.p), positions_tensor,
                purpose, workspace, nullptr);
    cuda_synchronize();

    RunResult result;
    result.tokens = from_device<int>(device_out.data(), static_cast<std::size_t>(batch));
    result.integrity_failures += device_out.verify_guards("sample output");
    result.integrity_failures +=
        verify_exact("sample read-only logits",
                     from_device<std::uint16_t>(device_logits, input_bits.size()), input_bits);

    const std::vector<ops::SamplingConfig> actual_configs =
        from_device<ops::SamplingConfig>(device_configs, configs.size());
    for (std::size_t row = 0; row < configs.size(); ++row) {
        if (!same_config(actual_configs[row], expected_configs[row])) {
            std::cerr << "sample modified SamplingConfig row " << row << '\n';
            ++result.integrity_failures;
        }
    }
    result.integrity_failures += verify_exact(
        "sample read-only logical positions",
        from_device<std::int32_t>(device_positions, configs.size()), logical_positions);

    result.counts.reserve(device_counts.size());
    for (std::size_t row = 0; row < device_counts.size(); ++row) {
        result.counts.push_back(
            from_device<int>(device_counts[row]->data(), static_cast<std::size_t>(token_domain)));
        result.integrity_failures += device_counts[row]->verify_guards("sample token_counts");
    }
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "sample workspace query/execution high-water mismatch\n";
        ++result.integrity_failures;
    }
    return result;
}

RunResult run_homogeneous_batch(const std::vector<float>& logits, int physical_rows,
                                int token_domain, int batch, ops::SamplingConfig config,
                                int first_position, int purpose,
                                const std::vector<int>* initial_counts = nullptr) {
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch), config);
    std::vector<int> positions(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        positions[static_cast<std::size_t>(row)] = first_position + row;
    }
    std::vector<std::vector<int>> row_counts;
    if (initial_counts != nullptr) {
        row_counts.assign(static_cast<std::size_t>(batch), *initial_counts);
    }
    return run_batch(logits, physical_rows, token_domain, std::move(configs), positions, purpose,
                     row_counts);
}

RunResult run_repeated(const std::vector<float>& column, int token_domain, int total, int batch,
                       ops::SamplingConfig config, int position, int purpose) {
    if (batch <= 0 || total <= 0 || total % batch != 0) {
        throw std::invalid_argument("repeated sample count must be a positive batch multiple");
    }
    const int physical_rows                     = static_cast<int>(column.size());
    const std::vector<float> logits             = repeat_column(column, batch);
    const std::vector<std::uint16_t> input_bits = bf16_bits(logits);
    DeviceBuffer device_logits                  = to_device(input_bits);
    GuardedDeviceBuffer collected(static_cast<std::size_t>(total) * sizeof(std::int32_t));
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch), config);
    const std::vector<ops::SamplingConfig> expected_configs = configs;
    DeviceBuffer device_configs                             = to_device(configs);
    std::vector<int> positions(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i) { positions[static_cast<std::size_t>(i)] = position + i; }
    DeviceBuffer device_positions = to_device(positions);

    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, batch});
    const std::size_t workspace_bytes =
        ops::sampling_workspace_capacity_bytes(token_domain, batch, batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    for (int produced = 0; produced < total; produced += batch) {
        auto* out = static_cast<std::int32_t*>(collected.data()) + produced;
        auto* pos = static_cast<std::int32_t*>(device_positions.p) + produced;
        Tensor out_tensor(out, DType::I32, {batch});
        Tensor positions_tensor(pos, DType::I32, {batch});
        ops::sample(logits_tensor, out_tensor, token_domain,
                    static_cast<const ops::SamplingConfig*>(device_configs.p), positions_tensor,
                    purpose, workspace, nullptr);
    }
    cuda_synchronize();

    RunResult result;
    result.tokens = from_device<int>(collected.data(), static_cast<std::size_t>(total));
    result.integrity_failures += collected.verify_guards("sample repeated output");
    result.integrity_failures +=
        verify_exact("sample repeated read-only logits",
                     from_device<std::uint16_t>(device_logits, input_bits.size()), input_bits);
    const std::vector<ops::SamplingConfig> actual_configs =
        from_device<ops::SamplingConfig>(device_configs, configs.size());
    for (std::size_t row = 0; row < configs.size(); ++row) {
        if (!same_config(actual_configs[row], expected_configs[row])) {
            std::cerr << "sample repeated modified SamplingConfig row " << row << '\n';
            ++result.integrity_failures;
        }
    }
    result.integrity_failures +=
        verify_exact("sample repeated read-only logical positions",
                     from_device<std::int32_t>(device_positions, positions.size()), positions);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "sample repeated workspace query/execution high-water mismatch\n";
        ++result.integrity_failures;
    }
    return result;
}

int verify_distribution(const char* label, const std::vector<int>& samples,
                        const Distribution& expected) {
    std::vector<int> observed(expected.tokens.size(), 0);
    for (int token : samples) {
        const auto it = std::find(expected.tokens.begin(), expected.tokens.end(), token);
        if (it == expected.tokens.end()) {
            std::cerr << label << ": sampled token " << token << " outside oracle support\n";
            return 1;
        }
        ++observed[static_cast<std::size_t>(it - expected.tokens.begin())];
    }

    const double n              = static_cast<double>(samples.size());
    double max_standardized_gap = 0.0;
    for (std::size_t i = 0; i < expected.tokens.size(); ++i) {
        const double probability = expected.probabilities[i];
        const double frequency   = static_cast<double>(observed[i]) / n;
        const double sigma       = std::sqrt(probability * (1.0 - probability) / n);
        const double limit       = 7.0 * sigma + 2.0 / n;
        const double gap         = std::abs(frequency - probability);
        if (gap > limit) {
            std::cerr << label << ": token=" << expected.tokens[i] << " frequency=" << frequency
                      << " oracle=" << probability << " gap=" << gap << " limit=" << limit << '\n';
            return 1;
        }
        if (sigma > 0.0) { max_standardized_gap = std::max(max_standardized_gap, gap / sigma); }
    }
    std::cout << "    " << label << " FP64 distribution match (max z=" << max_standardized_gap
              << ")\n";
    return 0;
}

int bias_and_mask_contract() {
    int failures = 0;
    for (const int domain : {32, 257, 32000}) {
        std::vector<float> column(domain + 1, 0.0F), bias(domain, 0.0F);
        column[0] = 10.0F;
        column[domain] = 1000.0F; // padding must never participate
        bias[5] = 25.0F;
        std::vector<std::int32_t> mask((domain + 31) / 32, 0);
        mask[0] = 1 << 7;
        auto device_bias = to_device(bias);
        auto device_mask = to_device(mask);
        std::vector<ops::SamplingConfig> configs(5);
        configs[1].logit_bias = static_cast<const float*>(device_bias.p);
        configs[2] = configs[1];
        configs[2].token_bitmask = static_cast<const std::int32_t*>(device_mask.p);
        configs[3] = configs[2];
        configs[3].temperature = 1.0F; // untruncated stochastic mask
        configs[4] = configs[3];
        configs[4].top_k = 10; // bounded stochastic mask
        const auto result = run_batch(repeat_column(column, 5), domain + 1, domain, configs,
                                      {0, 1, 2, 3, 4}, ops::kSamplePurposeDecode);
        failures += result.integrity_failures;
        failures += verify_exact("bias and mask across sampler routes", result.tokens, {0, 5, 7, 7, 7});

        // The speculative verifier updates greedy rows across every verification column;
        // stochastic rows retain their unused argmax values for the acceptance sampler.
        auto logits = to_device(bf16_bits(repeat_column(column, 10)));
        auto targets = to_device(std::vector<int>(10, 0));
        auto params = to_device(configs);
        Tensor scores(logits.p, DType::BF16, {domain + 1, 2, 5});
        Tensor output(targets.p, DType::I32, {2, 5});
        ops::sampling_update_greedy_targets(scores, output, domain,
            static_cast<const ops::SamplingConfig*>(params.p), nullptr);
        cuda_synchronize();
        failures += verify_exact("biased speculative targets", from_device<int>(targets, 10),
                                 {0, 0, 5, 5, 7, 7, 0, 0, 0, 0});
    }
    return failures;
}

int greedy_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int batch         = 8;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch, -9.0f);
    for (int row = 0; row < batch; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * physical_rows;
        int first              = (17 + 7919 * row) % token_domain;
        int second             = token_domain - 1 - ((31 + 65537 * row) % token_domain);
        if (second == first) { second = (first + 1) % token_domain; }
        if (second < first) { std::swap(first, second); }
        logits[base + first]             = row == 0 ? -0.0f : 16.0f + row;
        logits[base + second]            = row == 0 ? 0.0f : 16.0f + row;
        logits[base + token_domain]      = 100.0f;
        logits[base + physical_rows - 1] = 200.0f;
    }
    round_to_bf16(logits);

    std::vector<int> counts(static_cast<std::size_t>(token_domain), 0);
    counts[17] = 9;
    ops::SamplingConfig config;
    config.temperature       = 0.0f;
    config.top_k             = 1;
    config.top_p             = 0.01f;
    config.min_p             = 0.99f;
    config.presence_penalty  = 100.0f;
    config.frequency_penalty = 100.0f;
    config.seed              = 12345;

    const RunResult result = run_homogeneous_batch(logits, physical_rows, token_domain, batch,
                                                   config, 77, ops::kSamplePurposeDecode, &counts);
    int failures           = result.integrity_failures;
    failures += verify_exact("sample greedy mathematical result", result.tokens,
                             greedy_oracle(logits, physical_rows, token_domain, batch));
    for (const std::vector<int>& row_counts : result.counts) {
        failures += verify_exact("sample greedy skips token_counts updates", row_counts, counts);
    }
    return failures;
}

int deterministic_stochastic_contract() {
    std::vector<float> column = {5.0f, 4.5f, 4.0f, 3.0f, -1.0f};
    round_to_bf16(column);
    int failures = 0;

    struct Case {
        const char* label;
        float presence;
        float frequency;
        std::vector<int> counts;
    };

    const Case cases[] = {
        {"sample positive-temperature presence penalty", 1.0f, 0.0f, {1, 0, 0, 0, 0}},
        {"sample positive-temperature frequency penalty", 0.0f, 0.5f, {2, 0, 0, 0, 0}},
        {"sample adjusted-logit tie break", 0.5f, 0.0f, {1, 0, 0, 0, 0}},
    };
    for (const Case& test_case : cases) {
        ops::SamplingConfig config;
        config.temperature       = 0.8f;
        config.top_k             = 1;
        config.presence_penalty  = test_case.presence;
        config.frequency_penalty = test_case.frequency;
        config.seed              = 9981;
        const Distribution oracle =
            distribution_oracle(column, static_cast<int>(column.size()), config, &test_case.counts);
        RunResult result = run_homogeneous_batch(column, static_cast<int>(column.size()),
                                                 static_cast<int>(column.size()), 1, config, 11,
                                                 ops::kSamplePurposeDecode, &test_case.counts);

        std::vector<int> expected_counts = test_case.counts;
        ++expected_counts[static_cast<std::size_t>(oracle.tokens.front())];
        failures += result.integrity_failures;
        failures += verify_exact(test_case.label, result.tokens, {oracle.tokens.front()});
        failures += verify_exact("sample increments only selected token", result.counts.front(),
                                 expected_counts);
    }
    return failures;
}

int heterogeneous_batch_contract() {
    constexpr int physical_rows = 260;
    constexpr int token_domain  = 257;
    constexpr int batch         = 4;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * batch, -8.0f);
    const auto set = [&](int row, int token, float value) {
        logits[static_cast<std::size_t>(row) * physical_rows + token] = value;
    };
    set(0, 5, 4.0f);
    set(0, 7, 3.0f);
    set(1, 11, 4.0f);
    set(1, 12, 3.0f);
    set(2, 13, 5.0f);
    set(2, 17, 4.5f);
    set(3, 19, 2.0f);
    set(3, 23, 2.0f);
    for (int row = 0; row < batch; ++row) { set(row, physical_rows - 1, 100.0f); }
    round_to_bf16(logits);

    std::vector<ops::SamplingConfig> configs(batch);
    configs[0].temperature      = 0.0f;
    configs[1].temperature      = 0.7f;
    configs[1].top_k            = 1;
    configs[1].seed             = 101;
    configs[2].temperature      = 0.7f;
    configs[2].top_k            = 1;
    configs[2].presence_penalty = 1.0f;
    configs[2].seed             = 202;
    configs[3].temperature      = 0.0f;

    std::vector<std::vector<int>> counts(batch, std::vector<int>(token_domain, 0));
    counts[0][5]           = 9;
    counts[2][13]          = 1;
    const RunResult result = run_batch(logits, physical_rows, token_domain, std::move(configs),
                                       {7, 103, 999, 41}, ops::kSamplePurposeDecode, counts);

    int failures = result.integrity_failures;
    failures += verify_exact("sample heterogeneous batch tokens", result.tokens, {5, 11, 17, 19});
    std::vector<std::vector<int>> expected = counts;
    ++expected[1][11];
    ++expected[2][17];
    for (int row = 0; row < batch; ++row) {
        failures += verify_exact("sample heterogeneous batch isolated token counts",
                                 result.counts[static_cast<std::size_t>(row)],
                                 expected[static_cast<std::size_t>(row)]);
    }
    return failures;
}

int filtered_distribution_contract() {
    std::vector<float> column = {3.0f, 2.7f,  2.7f,  2.1f,  1.5f,  0.7f,
                                 0.1f, -0.4f, -1.0f, -2.0f, -3.0f, -4.0f};
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 0.75f;
    config.top_k       = 6;
    config.top_p       = 0.86f;
    config.min_p       = 0.12f;
    config.seed        = 20260726;

    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), config);
    if (oracle.tokens.size() < 2 || oracle.tokens.size() >= 6) {
        std::cerr << "filtered distribution fixture did not exercise both filters\n";
        return 1;
    }

    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          config, 400, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample top-k/top-p/min-p", result.tokens, oracle);
}

int wide_nucleus_contract() {
    int failures = 0;
    for (int domain : {128, 4099}) {
        const std::vector<float> column(domain, 0.0F);
        ops::SamplingConfig config;
        config.temperature = 1;
        config.top_p = 0.75F;
        const auto oracle = distribution_oracle(column, domain, config);
        if (oracle.tokens.size() != static_cast<size_t>(std::ceil(0.75 * domain))) { return 1; }
        const auto result = run_repeated(column, domain, 8192, 32, config, 0, ops::kSamplePurposeDecode);
        failures += result.integrity_failures + verify_distribution("full-vocabulary nucleus", result.tokens, oracle);
        if (oracle.tokens.size() > ops::kSamplerSortTile) {
            const double expected_tail = double(oracle.tokens.size() - ops::kSamplerSortTile) / oracle.tokens.size();
            const double observed_tail = double(std::count_if(result.tokens.begin(), result.tokens.end(),
                [](int token) { return token >= ops::kSamplerSortTile; })) / result.tokens.size();
            if (std::abs(expected_tail - observed_tail) > 0.04) { ++failures; }
        }
    }
    return failures;
}

int large_vocabulary_filters_contract() {
    constexpr int domain = 32768;
    std::vector<float> column(domain, -20.0F);
    // Spread the likely tokens across the vocabulary to exercise every scan tile.
    for (int i = 0; i < 128; ++i) { column[i * 251] = -0.01F * i; }
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 1;
    config.top_k = 64;
    config.top_p = 0.8F;
    config.min_p = 0.5F;
    const auto oracle = distribution_oracle(column, domain, config);
    if (oracle.tokens.size() <= 20 || oracle.tokens.size() >= 64) { return 1; }
    const auto result = run_repeated(column, domain, 4096, 32, config, 0, ops::kSamplePurposeDecode);
    return result.integrity_failures + verify_distribution("large-vocabulary filters", result.tokens, oracle);
}

int wide_distribution_contract() {
    std::vector<float> column(24, 0.0f);
    for (int token = 0; token < 24; ++token) {
        column[static_cast<std::size_t>(token)] = 2.0f - 0.1f * token;
    }
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 1.1f;
    config.top_k       = 64;
    config.seed        = 884422;

    const Distribution oracle =
        distribution_oracle(column, static_cast<int>(column.size()), config);
    if (oracle.tokens.size() != 24) {
        std::cerr << "wide top-k oracle fixture has unexpected support\n";
        return 1;
    }

    constexpr int samples  = 16384;
    const RunResult result = run_repeated(column, static_cast<int>(column.size()), samples, 8,
                                          config, 900, ops::kSamplePurposePrefill);
    return result.integrity_failures +
           verify_distribution("sample top-k beyond twenty", result.tokens, oracle);
}

// A row with `top_k <= 0` and no nucleus is drawn from every token, not from the
// twenty best.
//
// The tail is what proves it. This column puts about a third of its mass beyond
// rank twenty, spread over hundreds of tokens that the candidate route cannot
// reach at all -- so the test fails on the old behaviour by sampling nothing out
// there, and the FP64 oracle it is checked against is the full-vocabulary softmax
// rather than a renormalised top twenty.
int untruncated_distribution_contract() {
    constexpr int token_domain = 512;
    std::vector<float> column(token_domain, 0.0f);
    // A shallow ramp: the best token is only a little ahead of the worst, so the
    // mass past rank twenty is large enough to measure in a few thousand draws.
    for (int token = 0; token < token_domain; ++token) {
        column[static_cast<std::size_t>(token)] =
            2.0f - 0.004f * static_cast<float>(token);
    }
    round_to_bf16(column);

    ops::SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k       = 0;   // "no limit of the caller's own"
    config.top_p       = 1.0f;
    config.min_p       = 0.0f;
    config.seed        = 24680;
    const Distribution oracle = distribution_oracle(column, token_domain, config);

    constexpr int samples = 8192;
    const RunResult result =
        run_repeated(column, token_domain, samples, 8, config, 500, ops::kSamplePurposeDecode);

    int failures = result.integrity_failures;
    failures += verify_distribution("sample untruncated full vocabulary", result.tokens, oracle);

    // The candidate route could only ever return one of the first twenty tokens.
    int beyond_cap = 0;
    for (const int token : result.tokens) {
        if (token >= ops::kSamplerFastCandidates) { ++beyond_cap; }
    }
    if (beyond_cap == 0) {
        std::cerr << "sample untruncated full vocabulary: no token past rank "
                  << ops::kSamplerFastCandidates << " was ever drawn -- the row was truncated\n";
        ++failures;
    } else {
        std::cout << "    sample untruncated full vocabulary reached the tail (" << beyond_cap
                  << " of " << samples << " draws past rank " << ops::kSamplerFastCandidates << ")\n";
    }
    return failures;
}

// The same route on a real vocabulary with a peaked distribution.
//
// The shallow-ramp case above would pass even if the tail were over-weighted, and
// a served model's first token is often near-certain -- so this checks the shape
// that matters: one token holding almost all the mass over a full-size domain. A
// draw from anywhere else should be rare, and a heavy upper tail in the underlying
// variate shows up here as a flood of them.
int untruncated_peaked_contract() {
    constexpr int token_domain = 151936;
    std::vector<float> column(token_domain, 0.0f);
    column[7] = 20.0f; // exp(20) against 151935 flat tokens: p_peak ~ 0.99969
    round_to_bf16(column);

    ops::SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k       = 0;
    config.top_p       = 1.0f;
    config.seed        = 13579;

    constexpr int samples = 4096;
    const RunResult result =
        run_repeated(column, token_domain, samples, 8, config, 900, ops::kSamplePurposeDecode);

    int off_peak = 0;
    for (const int token : result.tokens) {
        if (token != 7) { ++off_peak; }
    }
    const double peak_weight = std::exp(20.0);
    const double total       = peak_weight + static_cast<double>(token_domain - 1);
    const double expected    = static_cast<double>(samples) * (1.0 - peak_weight / total);
    const double sigma       = std::sqrt(expected * (peak_weight / total));
    std::cout << "    untruncated peaked: off-peak draws " << off_peak << ", oracle expects "
              << expected << " +/- " << (3.0 * sigma) << '\n';
    if (std::fabs(static_cast<double>(off_peak) - expected) > 5.0 * sigma + 5.0) {
        std::cerr << "sample untruncated peaked: off-peak draws " << off_peak
                  << " but the full-vocabulary oracle expects " << expected << '\n';
        return result.integrity_failures + 1;
    }
    return result.integrity_failures;
}

int real_shape_distribution_contract() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    std::vector<float> column(physical_rows, -20.0f);
    const int ids[]      = {17, 7919, 65537, 200003};
    const float logits[] = {3.0f, 2.0f, 1.0f, 0.0f};
    for (int i = 0; i < 4; ++i) { column[ids[i]] = logits[i]; }
    column[token_domain]      = 100.0f;
    column[physical_rows - 1] = 200.0f;
    round_to_bf16(column);

    ops::SamplingConfig config;
    config.temperature        = 1.0f;
    config.top_k              = 4;
    config.seed               = 7654321;
    const Distribution oracle = distribution_oracle(column, token_domain, config);

    constexpr int samples = 4096;
    RunResult result =
        run_repeated(column, token_domain, samples, 8, config, 2000, ops::kSamplePurposeDecode);
    return result.integrity_failures +
           verify_distribution("sample real token-domain B=8", result.tokens, oracle);
}

int rng_key_contract() {
    std::vector<float> column = {0.0f, 0.0f};
    round_to_bf16(column);
    ops::SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k       = 2;
    config.seed        = 424242;

    constexpr int samples = 128;
    const RunResult baseline =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposeDecode);
    const RunResult repeat =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposeDecode);
    const RunResult rechunked =
        run_repeated(column, 2, samples, 1, config, 100, ops::kSamplePurposeDecode);
    const RunResult shifted =
        run_repeated(column, 2, samples, 8, config, 101, ops::kSamplePurposeDecode);
    const RunResult other_purpose =
        run_repeated(column, 2, samples, 8, config, 100, ops::kSamplePurposePrefill);
    ops::SamplingConfig other_seed_config = config;
    ++other_seed_config.seed;
    const RunResult other_seed =
        run_repeated(column, 2, samples, 8, other_seed_config, 100, ops::kSamplePurposeDecode);

    int failures = baseline.integrity_failures + repeat.integrity_failures +
                   rechunked.integrity_failures + shifted.integrity_failures +
                   other_purpose.integrity_failures + other_seed.integrity_failures;
    failures += verify_exact("sample identical counter key is reproducible", repeat.tokens,
                             baseline.tokens);
    failures += verify_exact("sample RNG does not depend on compact row", rechunked.tokens,
                             baseline.tokens);
    failures += verify_exact("sample position selects the corresponding counter",
                             std::vector<int>(shifted.tokens.begin(), shifted.tokens.end() - 1),
                             std::vector<int>(baseline.tokens.begin() + 1, baseline.tokens.end()));
    if (other_purpose.tokens == baseline.tokens) {
        std::cerr << "sample purpose did not separate the counter stream\n";
        ++failures;
    }
    if (other_seed.tokens == baseline.tokens) {
        std::cerr << "sample seed did not separate the counter stream\n";
        ++failures;
    }
    return failures;
}

int workspace_route_boundary_contract() {
    constexpr int token_domain = 257;
    constexpr int batch        = 8;
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * batch, 0.0f);
    const RunResult result =
        run_homogeneous_batch(logits, token_domain, token_domain, batch, ops::SamplingConfig{}, 0,
                              ops::kSamplePurposeDecode);
    int failures = result.integrity_failures;
    failures +=
        verify_exact("sample workspace route boundary", result.tokens, std::vector<int>(batch, 0));
    return failures;
}

int test_logprob_scores() {
    int failures = 0;
    for (int domain : {7, 1031, 151936}) {
        std::vector<float> input(domain + 17, 1000.0f);
        for (int i = 0; i < domain; ++i) input[i] = float((i * 73) % 127 - 63) / 8;
        auto device = to_device_bf16(input);
        Tensor logits(device.p, DType::BF16, {domain + 17, 1});
        std::vector<int> order(domain);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return input[a] > input[b]; });
        double sum = 0;
        const double maximum = input[order[0]];
        for (int i = 0; i < domain; ++i) sum += std::exp(input[i] - maximum);
        const double normalizer = maximum + std::log(sum);
        for (int count : {0, 1, 20}) {
            for (int token : {0, domain / 2, order[0]}) {
                const auto score = ops::score_logprobs(logits, token, domain, count, nullptr);
                const auto rank = std::count_if(input.begin(), input.begin() + domain,
                    [&](float value) { return value >= input[token]; });
                failures += score.selected.token_id != token || score.selected.rank != rank ||
                    std::abs(score.selected.logprob - (input[token] - normalizer)) > 2e-5;
                failures += score.top.size() != std::min(count, domain);
                for (std::size_t k = 0; k < score.top.size(); ++k) {
                    failures += score.top[k].token_id != order[k] || score.top[k].rank != k + 1 ||
                        std::abs(score.top[k].logprob - (input[order[k]] - normalizer)) > 2e-5;
                }
            }
        }
    }
    if (failures) std::cerr << "log-probability oracle failures: " << failures << '\n';
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = test_logprob_scores();
    // The multi-block column cap is read from kSamplerMaxColumns, not restated.
    // It has moved twice -- 16, then 32, then tied to the ops batch bound -- and
    // the literal 16 these boundaries used to carry went stale silently both
    // times. The header defining it says "Never hardcode it again"; that applies
    // on this side of the contract too.
    constexpr std::int32_t kCap = ops::kSamplerMaxColumns;
    const auto capacity = [](std::int32_t domain, std::int32_t lo, std::int32_t hi) {
        return ops::sampling_workspace_capacity_bytes(domain, lo, hi);
    };
    const auto expect = [&](bool ok, const char* what) {
        if (!ok) {
            std::cerr << "sampling workspace route boundary: " << what << "\n";
            ++failures;
        }
    };
    expect(capacity(256, 1, kCap) == 0, "sized a domain that fits one sampler tile");
    expect(capacity(257, kCap, kCap) != 0, "reported no capacity at the column cap");
    expect(capacity(257, kCap + 1, kCap + 1) == 0, "sized a round past the column cap");
    expect(capacity(257, 1, kCap + 1) == capacity(257, 1, kCap),
           "did not saturate an interval that runs past the column cap");
    try {
        (void)ops::sampling_workspace_capacity_bytes(257, 0, 16);
        std::cerr << "sampling workspace accepted an invalid lane interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += bias_and_mask_contract();
    failures += greedy_contract();
    failures += deterministic_stochastic_contract();
    failures += heterogeneous_batch_contract();
    failures += filtered_distribution_contract();
    failures += wide_distribution_contract();
    failures += wide_nucleus_contract();
    failures += large_vocabulary_filters_contract();
    failures += untruncated_distribution_contract();
    failures += untruncated_peaked_contract();
    failures += real_shape_distribution_contract();
    failures += rng_key_contract();
    failures += workspace_route_boundary_contract();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sample public contract\n";
    return failures == 0 ? 0 : 1;
}
