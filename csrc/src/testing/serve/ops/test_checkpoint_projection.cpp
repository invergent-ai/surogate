#include "api/ops/attn_input_proj.h"
#include "api/ops/gdn_input_proj.h"
#include "ops/direct_bf16_weight.h"
#include "ops/input_projection_test_common.h"

using namespace sinfer;
using namespace sinfer::test;
using namespace sinfer::test::input_projection;
namespace direct = sinfer::test::direct_bf16_weight;

namespace {
int check(const GuardedBf16Tensor& output, const direct::HostWeight& parent,
          int begin, int rows, const std::vector<float>& activation, int tokens) {
    int failures = output.verify_guards("checkpoint projection") +
                   output.verify_fully_written("checkpoint projection");
    std::vector<double> expected;
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < rows; ++r) {
            expected.push_back(direct::dot_fp64(parent, begin + r,
                std::span<const float>(activation.data() + t * parent.k, parent.k)));
        }
    }
    return failures + compare("checkpoint projection", output.values(), expected,
                                ReductionCriterion{3e-3, 4e-3, 5e-3});
}

int run(int hidden, int q_rows, int kv_rows, int tokens) {
    const int rows = 2 * (q_rows + kv_rows);
    direct::DeviceWeight parent(direct::make_patterned(rows, hidden, 223));
    auto activation = make_bf16_activation(hidden, tokens, 991);
    const auto bits = bf16_bits(activation);
    auto input = to_device(bits);
    Tensor x(input.p, DType::BF16, {hidden, tokens});
    GuardedBf16Tensor query(q_rows, tokens), key(kv_rows, tokens), gate(q_rows, tokens), value(kv_rows, tokens);
    auto q = query.tensor(), k = key.tensor(), z = gate.tensor(), v = value.tensor();
    ops::attn_input_proj(x, parent.view(), q, z, k, v, nullptr);
    cuda_synchronize();
    int failures = check(query, parent.host, 0, q_rows, activation, tokens) +
                   check(key, parent.host, q_rows, kv_rows, activation, tokens) +
                   check(gate, parent.host, q_rows + kv_rows, q_rows, activation, tokens) +
                   check(value, parent.host, 2 * q_rows + kv_rows, kv_rows, activation, tokens);
    GuardedBf16Tensor qkv(rows - q_rows, tokens), output_gate(q_rows, tokens);
    auto qkv_tensor = qkv.tensor(), gate_tensor = output_gate.tensor();
    ops::gdn_input_proj(x, parent.view(), qkv_tensor, gate_tensor, nullptr);
    cuda_synchronize();
    failures += check(qkv, parent.host, 0, rows - q_rows, activation, tokens);
    failures += check(output_gate, parent.host, rows - q_rows, q_rows, activation, tokens);
    failures += verify_preserved("checkpoint input", input, bits);
    failures += parent.verify_preserved("checkpoint BF16 parent");
    if (ops::attn_input_proj_workspace_capacity_bytes(QType::BF16_CTRL, rows, hidden,
          ops::LinearPolicy::A16Only, 1, tokens) != 0 ||
        ops::gdn_input_proj_workspace_capacity_bytes(QType::BF16_CTRL, rows, hidden,
          ops::LinearPolicy::A16Only, 1, tokens) != 0) { ++failures; }
    return failures;
}

int mixed_projection(int tokens, bool pair, bool quantized_first) {
    constexpr int hidden = 512, value_rows = 128, z_rows = 128;
    const int first_rows = pair ? 128 : 384;
    const int second_rows = pair ? value_rows + z_rows : z_rows;
    const int qkv_rows = pair ? first_rows + value_rows : first_rows;
    direct::DeviceWeight dense(direct::make_patterned(
        quantized_first ? second_rows : first_rows, hidden, 179));
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor = 3.5F;
    DevicePackedWeight quantized(quantized_weight::make_patterned_weight(
        QType::NVFP4, quantized_first ? first_rows : second_rows, hidden, 293, options));
    const Weight a = quantized_first ? quantized.view() : dense.view();
    const Weight b = quantized_first ? dense.view() : quantized.view();
    const auto ap = quantized_first ? ops::LinearPolicy::AllowA4 : ops::LinearPolicy::A16Only;
    const auto bp = quantized_first ? ops::LinearPolicy::A16Only : ops::LinearPolicy::AllowA4;
    const auto activation = make_bf16_activation(hidden, tokens, 791);
    const auto bits = bf16_bits(activation);
    auto input = to_device(bits);
    Tensor x(input.p, DType::BF16, {hidden, tokens});
    GuardedBf16Tensor qkv(qkv_rows, tokens), gate(z_rows, tokens);
    auto q = qkv.tensor(), z = gate.tensor();
    const auto capacity = pair
        ? ops::gdn_input_proj_pair_workspace_capacity_bytes(a.qtype, b.qtype, first_rows,
            value_rows, z_rows, hidden, ap, bp, tokens, tokens)
        : ops::gdn_input_proj_split_workspace_capacity_bytes(a.qtype, b.qtype, first_rows,
            z_rows, hidden, ap, bp, tokens, tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    WorkspaceArena workspace(DeviceSpan{static_cast<std::byte*>(scratch.data()), scratch.bytes()});
    if (pair) { ops::gdn_input_proj_pair(x, a, b, q, z, ap, bp, workspace, nullptr); }
    else { ops::gdn_input_proj_split(x, a, b, q, z, ap, bp, workspace, nullptr); }
    cuda_synchronize();
    const auto dot = [&](bool first, int row, int t) {
        if (first != quantized_first) {
            return direct::dot_fp64(dense.host, row,
                std::span<const float>(activation.data() + t * hidden, hidden));
        }
        double sum = 0;
        for (int k = 0; k < hidden; ++k) {
            sum += quantized_weight::logical_weight_fp64(quantized.host, row, k) * activation[t * hidden + k];
        }
        return sum;
    };
    std::vector<double> qref, zref;
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < qkv_rows; ++r) {
            qref.push_back(r < first_rows ? dot(true, r, t) : dot(false, r - first_rows, t));
        }
        for (int r = 0; r < z_rows; ++r) { zref.push_back(dot(false, (pair ? value_rows : 0) + r, t)); }
    }
    const ReductionCriterion a4{0.16, 4e-3, 0.16};
    return compare("mixed qkv", qkv.values(), qref, a4) + compare("mixed z", gate.values(), zref, a4) +
        qkv.verify_guards("mixed qkv") + gate.verify_guards("mixed z") +
        qkv.verify_fully_written("mixed qkv") + gate.verify_fully_written("mixed z") +
        scratch.verify_guards("mixed workspace") + dense.verify_preserved("mixed BF16") +
        quantized.verify_preserved("mixed NVFP4") + verify_preserved("mixed input", input, bits);
}

int configured_convolution(int batch, int width, bool record) {
    constexpr int hidden = 128, query_rows = 64, key_rows = 64, value_rows = 128;
    constexpr int channels = query_rows + key_rows + value_rows;
    const int tokens = batch * width, slots = tokens + 1;
    direct::DeviceWeight parent(direct::make_patterned(channels, hidden, 151));
    direct::DeviceWeight gate_weight(direct::make_patterned(value_rows, hidden, 353));
    const auto activation = make_bf16_activation(hidden, tokens, 193);
    auto input = to_device(bf16_bits(activation));
    Tensor x(input.p, DType::BF16, {hidden, width, batch});
    auto conv_values = make_bf16_activation(channels, 4, 431);
    auto conv_data = to_device(bf16_bits(conv_values));
    Tensor conv(conv_data.p, DType::BF16, {channels, 4});
    auto state_values = make_bf16_activation(channels, 3 * slots, 541);
    auto state_bits = bf16_bits(state_values);
    auto states = to_device(state_bits);
    Tensor state(states.p, DType::BF16, {channels, 3, slots});
    std::vector<int> valid(batch, width), initial(batch, 0), base(batch);
    if (batch > 1) { --valid.back(); }
    for (int b = 0; b < batch; ++b) { base[b] = 1 + b * width; }
    auto valid_data = to_device(valid), initial_data = to_device(initial), base_data = to_device(base);
    Tensor valid_tensor(valid_data.p, DType::I32, {batch});
    Tensor initial_tensor(initial_data.p, DType::I32, {batch});
    Tensor base_tensor(base_data.p, DType::I32, {batch});
    GuardedBf16Tensor qo(query_rows, tokens), ko(key_rows, tokens), vo(value_rows, tokens), zo(value_rows, tokens);
    auto q = qo.tensor().view({query_rows, width, batch});
    auto k = ko.tensor().view({key_rows, width, batch});
    auto v = vo.tensor().view({value_rows, width, batch});
    auto z = zo.tensor().view({value_rows, width, batch});
    constexpr auto a16 = ops::LinearPolicy::A16Only;
    const auto capacity = record
        ? ops::gdn_input_proj_conv_record_split_workspace_capacity_bytes(QType::BF16_CTRL, QType::BF16_CTRL,
            channels, value_rows, hidden, a16, a16, batch, width, width)
        : ops::gdn_input_proj_conv_snapshot_split_workspace_capacity_bytes(QType::BF16_CTRL, QType::BF16_CTRL,
            channels, value_rows, hidden, a16, a16, batch, width, width);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    WorkspaceArena workspace(DeviceSpan{static_cast<std::byte*>(scratch.data()), scratch.bytes()});
    GuardedBf16Tensor recorded(channels, tokens);
    auto record_tensor = recorded.tensor().view({channels, width, batch});
    if (record) {
        ops::gdn_input_proj_conv_record_split(x, parent.view(), gate_weight.view(), conv, state,
            valid_tensor, initial_tensor, record_tensor, q, k, v, z, a16, a16, workspace, nullptr);
    } else {
        ops::gdn_input_proj_conv_snapshot_split(x, parent.view(), gate_weight.view(), conv, state,
            valid_tensor, initial_tensor, base_tensor, q, k, v, z, a16, a16, workspace, nullptr);
    }
    cuda_synchronize();
    std::vector<double> qref(query_rows * tokens), kref(key_rows * tokens), vref(value_rows * tokens);
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < channels; ++r) {
            double history[3]{state_values[r], state_values[channels + r], state_values[2 * channels + r]};
            for (int t = 0; t < valid[b]; ++t) {
                const int column = b * width + t;
                const double projected = bf16_to_f32(f32_to_bf16(static_cast<float>(direct::dot_fp64(
                    parent.host, r, std::span<const float>(activation.data() + column * hidden, hidden)))));
                double convolution = conv_values[3 * channels + r] * projected;
                for (int h = 0; h < 3; ++h) { convolution += conv_values[h * channels + r] * history[h]; }
                const double result = convolution / (1 + std::exp(-convolution));
                if (r < query_rows) { qref[column * query_rows + r] = result; }
                else if (r < query_rows + key_rows) { kref[column * key_rows + r - query_rows] = result; }
                else { vref[column * value_rows + r - query_rows - key_rows] = result; }
                history[0] = history[1]; history[1] = history[2]; history[2] = projected;
                if (!record) {
                    for (int h = 0; h < 3; ++h) {
                        state_bits[(base[b] + t) * 3 * channels + h * channels + r] = f32_to_bf16(history[h]);
                    }
                }
            }
        }
    }
    const ReductionCriterion criterion{3e-3, 4e-3, 5e-3};
    int failures = compare("configured convolution q", qo.values(), qref, criterion) +
        compare("configured convolution k", ko.values(), kref, criterion) +
        compare("configured convolution v", vo.values(), vref, criterion) +
        check(zo, gate_weight.host, 0, value_rows, activation, tokens) +
        verify_preserved("configured convolution history", states, state_bits) + scratch.verify_guards("convolution workspace");
    for (const auto* output : {&qo, &ko, &vo}) {
        failures += output->verify_guards("convolution output") + output->verify_fully_written("convolution output");
    }
    if (record) { failures += check(recorded, parent.host, 0, channels, activation, tokens); }
    return failures;
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    int failures = 0;
    for (int tokens : {1, 7, 33}) {
        failures += run(128, 256, 64, tokens);
        failures += run(384, 512, 128, tokens);
    }
    for (int tokens : {1, 7, 33}) {
        for (bool pair : {false, true}) {
            for (bool quantized_first : {false, true}) {
                failures += mixed_projection(tokens, pair, quantized_first);
            }
        }
    }
    for (int batch : {1, 2}) {
        for (bool record : {false, true}) { failures += configured_convolution(batch, 3, record); }
    }
    return failures ? 1 : 0;
}
