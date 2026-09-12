#include "api/ops/gated_delta_net.h"
#include "api/ops/l2norm.h"
#include "core/device.h"

#include "ops/gdn_ref.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr int kStateDim = 128;

constexpr ReductionCriterion gated_delta_net_output_bf16_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/5.5e-3};
}

// The recurrent state is stored bf16 (7fc710a5, "store GDN recurrent state in
// bf16, compute unchanged"), so it carries the same storage quantisation as the
// output beside it and is held to the same floor. The tighter numbers this used
// to carry (2.7e-3 / 3.9e-3) were calibrated when the state was fp32-stored: the
// compute did not change, the storage did, and against an fp64 oracle a
// bf16-stored value is off by up to about a ULP -- roughly 1e-3 at the ~0.15
// magnitudes these states reach.
//
// The gross bound is derived, not tuned: 3.9e-3 is the compute allowance this
// criterion carried while the state was fp32-stored, and bf16 keeps 8 mantissa
// bits, so storing an already-computed value costs up to half a ULP more --
// 2^-9, about 2.0e-3 relative. Their sum is 5.9e-3, rounded to 6.0e-3. Nothing
// about the arithmetic changed, so the compute term is unchanged and the
// storage term is the whole of the difference.
constexpr ReductionCriterion gated_delta_net_state_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/6.0e-3};
}

struct Case {
    const char* name;
    int qk_heads;
    int value_heads;
    int tokens;
    bool normalize_qk;
    bool near_zero_qk = false;
};

void fill_uniform(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

void normalize_rows(std::vector<float>& values, int width) {
    const std::size_t rows = values.size() / static_cast<std::size_t>(width);
    for (std::size_t row = 0; row < rows; ++row) {
        float* base  = values.data() + row * static_cast<std::size_t>(width);
        double sumsq = 0.0;
        for (int d = 0; d < width; ++d) {
            const double value = static_cast<double>(base[d]);
            sumsq += value * value;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (int d = 0; d < width; ++d) {
            base[d] = static_cast<float>(static_cast<double>(base[d]) * inv);
        }
    }
}

gdn_ref::Inputs make_inputs(const Case& test_case, std::uint32_t seed) {
    gdn_ref::Inputs in;
    in.head_dim    = kStateDim;
    in.qk_heads    = test_case.qk_heads;
    in.value_heads = test_case.value_heads;
    in.tokens      = test_case.tokens;

    const std::size_t qk_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * test_case.tokens);
    const std::size_t value_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * test_case.tokens);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);
    in.q.resize(qk_size);
    in.k.resize(qk_size);
    in.v.resize(value_size);
    in.g.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.beta.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.state.resize(state_size);

    std::mt19937 generator(seed);
    fill_uniform(in.q, generator, -1.0f, 1.0f);
    fill_uniform(in.k, generator, -1.0f, 1.0f);
    fill_uniform(in.v, generator, -0.5f, 0.5f);
    fill_uniform(in.g, generator, -0.10f, -0.005f);
    fill_uniform(in.beta, generator, 0.05f, 0.95f);
    fill_uniform(in.state, generator, -0.02f, 0.02f);

    if (test_case.near_zero_qk) {
        for (float& value : in.q) { value *= 1.0e-4f; }
        for (float& value : in.k) { value *= 1.0e-4f; }
    } else if (!test_case.normalize_qk) {
        // Raw-Q/K mode still receives a stable, entirely valid public input. This host-side
        // generation choice is not part of the oracle.
        normalize_rows(in.q, kStateDim);
        normalize_rows(in.k, kStateDim);
    }

    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    return in;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> doubles(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

template <typename T>
int verify_exact(const std::string& label, const std::vector<T>& got,
                 const std::vector<T>& expected) {
    return sinfer::test::verify_exact(label.c_str(), got, expected);
}

int verify_recurrence(const std::string& label, const std::vector<double>& got,
                      const std::vector<double>& expected, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), got, expected, criterion);
}

std::vector<double> bf16_doubles(std::vector<std::uint16_t>::const_iterator begin,
                                 std::vector<std::uint16_t>::const_iterator end) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(end - begin));
    for (auto it = begin; it != end; ++it) { out.push_back(bf16_to_f32(*it)); }
    return out;
}

std::vector<double> read_f32(const void* device, std::size_t count) {
    return doubles(from_device<float>(device, count));
}

int verify_common_inputs_unchanged(const std::string& label, const gdn_ref::Inputs& in,
                                   const DeviceBuffer& q, const DeviceBuffer& k,
                                   const DeviceBuffer& v, const DeviceBuffer& g,
                                   const DeviceBuffer& beta) {
    int failures = 0;
    failures += verify_exact(label + " q unchanged", from_device<std::uint16_t>(q, in.q.size()),
                             bf16_bits(in.q));
    failures += verify_exact(label + " k unchanged", from_device<std::uint16_t>(k, in.k.size()),
                             bf16_bits(in.k));
    failures += verify_exact(label + " v unchanged", from_device<std::uint16_t>(v, in.v.size()),
                             bf16_bits(in.v));
    failures += verify_exact(label + " g unchanged", from_device<float>(g, in.g.size()), in.g);
    failures +=
        verify_exact(label + " beta unchanged", from_device<float>(beta, in.beta.size()), in.beta);
    return failures;
}

struct DeviceInputs {
    explicit DeviceInputs(const gdn_ref::Inputs& in)
        : q(to_device_bf16(in.q)), k(to_device_bf16(in.k)), v(to_device_bf16(in.v)),
          g(to_device_f32(in.g)), beta(to_device_f32(in.beta)) {}

    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
};

int inplace_case(const Case& test_case, std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk);
    DeviceInputs device(in);
    const std::vector<std::uint16_t> state_bits = bf16_bits(in.state);
    GuardedDeviceBuffer state(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state.copy_from_host(state_bits.data(), state.bytes());
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_tensor(state.data(), DType::BF16, {kStateDim, kStateDim, test_case.value_heads});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace, state_tensor,
                         out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " inplace";
    int failures            = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " state", from_device_bf16(state.data(), in.state.size()),
                                  ref.final_state, gated_delta_net_state_criterion());
    failures += state.verify_guards((label + " state").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int distinct_state_case(const Case& test_case, std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk);
    DeviceInputs device(in);
    const std::vector<std::uint16_t> state_bits = bf16_bits(in.state);
    GuardedDeviceBuffer state_in(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_out(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_in.copy_from_host(state_bits.data(), state_in.bytes());
    state_out.fill(0xff);
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_in_tensor(state_in.data(), DType::BF16,
                           {kStateDim, kStateDim, test_case.value_heads});
    Tensor state_out_tensor(state_out.data(), DType::BF16,
                            {kStateDim, kStateDim, test_case.value_heads});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace,
                         state_in_tensor, state_out_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " distinct-state";
    int failures            = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " state",
                                  from_device_bf16(state_out.data(), in.state.size()),
                                  ref.final_state, gated_delta_net_state_criterion());
    failures += verify_exact(label + " state-in unchanged",
                             from_device<std::uint16_t>(state_in.data(), in.state.size()),
                             state_bits);
    failures += state_in.verify_guards((label + " state-in").c_str());
    failures += state_out.verify_guards((label + " state-out").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

// Normalization must have the same represented values whether it is fused into
// decode or materialized by chunked prefill. Compare both public paths directly,
// including their final state, independently of the FP64 recurrence oracle.
int normalization_parity_case(int tokens, bool snapshot) {
    const Case test_case{"normalization parity", 4, 8, tokens, true};
    const auto in = make_inputs(test_case, 15000u + tokens);
    DeviceInputs device(in);
    Tensor q(device.q.p, DType::BF16, {kStateDim, 4, tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, 4, tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, 8, tokens});
    Tensor g(device.g.p, DType::FP32, {8, tokens});
    Tensor beta(device.beta.p, DType::FP32, {8, tokens});
    DeviceBuffer normalized_q(q.bytes()), normalized_k(k.bytes());
    Tensor nq(normalized_q.p, DType::BF16, {kStateDim, 4, tokens});
    Tensor nk(normalized_k.p, DType::BF16, {kStateDim, 4, tokens});
    ops::l2norm(q, 1.0e-6f, nq, nullptr);
    ops::l2norm(k, 1.0e-6f, nk, nullptr);

    const int slots = snapshot ? tokens + 1 : 1;
    std::vector<float> initial(in.state.size() * slots, 0.0f);
    std::copy(in.state.begin(), in.state.end(), initial.begin());
    DeviceBuffer fused_state = to_device_bf16(initial), materialized_state = to_device_bf16(initial);
    DeviceBuffer fused_out(v.bytes()), materialized_out(v.bytes());
    Tensor fs(fused_state.p, DType::BF16, {kStateDim, kStateDim, 8, slots});
    Tensor ms(materialized_state.p, DType::BF16, {kStateDim, kStateDim, 8, slots});
    Tensor fo(fused_out.p, DType::BF16, {kStateDim, 8, tokens});
    Tensor mo(materialized_out.p, DType::BF16, {kStateDim, 8, tokens});
    const float scale = 1.0f / std::sqrt(float(kStateDim));
    if (snapshot) {
        auto initial_slot = to_device_i32({0});
        auto destination_slot = to_device_i32({1});
        Tensor source(initial_slot.p, DType::I32, {1});
        Tensor destination(destination_slot.p, DType::I32, {1});
        ops::gated_delta_net_snapshot(q, k, v, g, beta, scale, true, fs,
                                      Tensor{}, source, destination, fo, nullptr);
        ops::gated_delta_net_snapshot(nq, nk, v, g, beta, scale, false, ms,
                                      Tensor{}, source, destination, mo, nullptr);
        cuda_synchronize();
    } else {
        const auto bytes = ops::gated_delta_net_workspace_capacity_bytes(4, 8, true, tokens, tokens);
        WorkspaceArena workspace(std::max<std::size_t>(bytes, 256));
        ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, fs, fo, nullptr);
        ops::gated_delta_net(nq, nk, v, g, beta, scale, false, workspace, ms, mo, nullptr);
        cuda_synchronize();
    }
    const auto label = std::string(snapshot ? "snapshot" : "prefill") + " normalized T=" + std::to_string(tokens);
    int failures = verify_exact(label + " output", from_device<std::uint16_t>(fused_out, in.v.size()),
                                from_device<std::uint16_t>(materialized_out, in.v.size()));
    failures += verify_exact(label + " state", from_device<std::uint16_t>(fused_state, initial.size()),
                              from_device<std::uint16_t>(materialized_state, initial.size()));
    return failures;
}

// Appending neutral tokens must not change a prefix's output or final state.
// This catches a last partial block silently switching to recurrent arithmetic.
int partial_chunk_parity_case(int tokens, bool normalize, bool split = false, int heads = 8, bool capture = false) {
    const int padded = ((tokens + 63) / 64) * 64;
    const Case test_case{"partial chunk parity", heads / 2, heads, padded, normalize};
    auto in = make_inputs(test_case, 16000u + tokens);
    std::fill(in.g.begin() + tokens * heads, in.g.end(), 0.0f);
    std::fill(in.beta.begin() + tokens * heads, in.beta.end(), 0.0f);
    DeviceInputs device(in);
    Tensor q(device.q.p, DType::BF16, {kStateDim, heads / 2, padded});
    Tensor k(device.k.p, DType::BF16, {kStateDim, heads / 2, padded});
    Tensor v(device.v.p, DType::BF16, {kStateDim, heads, padded});
    Tensor g(device.g.p, DType::FP32, {heads, padded});
    Tensor beta(device.beta.p, DType::FP32, {heads, padded});
    auto prefix_state = to_device_bf16(in.state), padded_state = to_device_bf16(in.state);
    GuardedDeviceBuffer prefix_out(std::size_t(kStateDim) * heads * tokens * 2);
    DeviceBuffer padded_out(v.bytes());
    Tensor ps(prefix_state.p, DType::BF16, {kStateDim, kStateDim, heads});
    Tensor fs(padded_state.p, DType::BF16, {kStateDim, kStateDim, heads});
    Tensor po(prefix_out.data(), DType::BF16, {kStateDim, heads, tokens});
    Tensor fo(padded_out.p, DType::BF16, {kStateDim, heads, padded});
    const auto bytes = ops::gated_delta_net_workspace_capacity_bytes(heads / 2, heads, normalize, tokens, padded);
    WorkspaceArena workspace(bytes);
    const float scale = 1.0f / std::sqrt(float(kStateDim));
    auto short_input = in;
    short_input.q.resize(std::size_t(kStateDim) * (heads / 2) * tokens);
    short_input.k.resize(short_input.q.size());
    short_input.v.resize(std::size_t(kStateDim) * heads * tokens);
    short_input.g.resize(heads * tokens); short_input.beta.resize(heads * tokens);
    DeviceInputs short_device(short_input);
    Tensor sq(short_device.q.p, DType::BF16, {kStateDim, heads / 2, tokens});
    Tensor sk(short_device.k.p, DType::BF16, {kStateDim, heads / 2, tokens});
    Tensor sv(short_device.v.p, DType::BF16, {kStateDim, heads, tokens});
    Tensor sg(short_device.g.p, DType::FP32, {heads, tokens}), sb(short_device.beta.p, DType::FP32, {heads, tokens});
    cuda_synchronize();
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    if (capture) { CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); }
    const int step = split ? 64 : tokens;
    for (int begin = 0; begin < tokens; begin += step) {
        const int count = std::min(step, tokens - begin);
        Tensor part = po.slice(2, begin, count);
        ops::gated_delta_net(sq.slice(2, begin, count), sk.slice(2, begin, count), sv.slice(2, begin, count),
                             sg.slice(1, begin, count), sb.slice(1, begin, count), scale, normalize,
                             workspace, ps, part, stream);
    }
    if (capture) {
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            const auto bits = bf16_bits(in.state);
            CUDA_CHECK(cudaMemcpyAsync(prefix_state.p, bits.data(), prefix_state.bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    ops::gated_delta_net(q, k, v, g, beta, scale, normalize, workspace, fs, fo, stream);
    cuda_synchronize();
    const auto label = "partial chunk T=" + std::to_string(tokens) + " normalize=" + std::to_string(normalize) +
                       " split=" + std::to_string(split) + " heads=" + std::to_string(heads) + " graph=" + std::to_string(capture);
    const auto elements = po.numel();
    int failures = verify_exact(label + " output", from_device<std::uint16_t>(prefix_out.data(), elements),
                                from_device<std::uint16_t>(padded_out, elements));
    failures += verify_exact(label + " state", from_device<std::uint16_t>(prefix_state, in.state.size()),
                              from_device<std::uint16_t>(padded_state, in.state.size()));
    failures += prefix_out.verify_guards(label.c_str());
    if (workspace.used() != 0 || workspace.peak_used() > bytes) {
        std::cerr << label << ": workspace interval missed a partial chunk\n";
        ++failures;
    }
    if (capture) { CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph)); }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return failures;
}

int snapshot_case(const Case& test_case, int slots, int initial_slot, int snapshot_base_slot,
                  std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk, true);
    const std::size_t state_size = in.state.size();
    std::vector<float> initial_states(state_size * static_cast<std::size_t>(slots), 17.0f);
    std::copy(in.state.begin(), in.state.end(),
              initial_states.begin() + static_cast<std::size_t>(initial_slot) * state_size);

    DeviceInputs device(in);
    const std::vector<std::uint16_t> initial_bits = bf16_bits(initial_states);
    GuardedDeviceBuffer states(initial_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    states.copy_from_host(initial_bits.data(), states.bytes());
    out.fill(0xff);
    DeviceBuffer device_initial_slot       = to_device_i32({initial_slot});
    DeviceBuffer device_snapshot_base_slot = to_device_i32({snapshot_base_slot});

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor states_tensor(states.data(), DType::BF16,
                         {kStateDim, kStateDim, test_case.value_heads, slots});
    Tensor initial_slot_tensor(device_initial_slot.p, DType::I32, {1});
    Tensor snapshot_base_slot_tensor(device_snapshot_base_slot.p, DType::I32, {1});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    ops::gated_delta_net_snapshot(q, k, v, g, beta, scale, test_case.normalize_qk, states_tensor,
                                  Tensor{}, initial_slot_tensor, snapshot_base_slot_tensor,
                                  out_tensor, nullptr);
    cuda_synchronize();

    const std::string label             = std::string(test_case.name) + " snapshot";
    const std::vector<std::uint16_t> got_states =
        from_device<std::uint16_t>(states.data(), initial_bits.size());
    const auto got_updated_begin =
        got_states.begin() + static_cast<std::size_t>(snapshot_base_slot) * state_size;
    const auto got_updated_end =
        got_updated_begin + static_cast<std::size_t>(test_case.tokens) * state_size;
    int failures = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " updated state slots",
                                  bf16_doubles(got_updated_begin, got_updated_end), ref.snapshots,
                                  gated_delta_net_state_criterion());
    const auto initial_updated_begin =
        initial_bits.begin() + static_cast<std::size_t>(snapshot_base_slot) * state_size;
    const auto initial_updated_end =
        initial_updated_begin + static_cast<std::size_t>(test_case.tokens) * state_size;
    failures += verify_exact(label + " slots before destination unchanged",
                             std::vector<std::uint16_t>(got_states.begin(), got_updated_begin),
                             std::vector<std::uint16_t>(initial_bits.begin(), initial_updated_begin));
    failures += verify_exact(label + " slots after destination unchanged",
                             std::vector<std::uint16_t>(got_updated_end, got_states.end()),
                             std::vector<std::uint16_t>(initial_updated_end, initial_bits.end()));
    failures +=
        verify_exact(label + " initial-slot scalar unchanged",
                     from_device_i32(device_initial_slot, 1), std::vector<int>{initial_slot});
    failures += verify_exact(label + " snapshot-base scalar unchanged",
                             from_device_i32(device_snapshot_base_slot, 1),
                             std::vector<int>{snapshot_base_slot});
    failures += states.verify_guards((label + " states").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    return failures;
}

int batched_snapshot_case(const Case& test_case, const std::vector<int>& initial_slots,
                          const std::vector<int>& snapshot_bases,
                          const std::vector<int>& valid_columns, int slots, std::uint32_t seed) {
    const int batch   = static_cast<int>(initial_slots.size());
    const int width   = test_case.tokens;
    const bool masked = !valid_columns.empty();
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const std::size_t qk_row_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * width);
    const std::size_t value_row_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * width);
    const std::size_t gate_row_size = static_cast<std::size_t>(test_case.value_heads * width);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);

    gdn_ref::Inputs aggregate;
    aggregate.head_dim    = kStateDim;
    aggregate.qk_heads    = test_case.qk_heads;
    aggregate.value_heads = test_case.value_heads;
    aggregate.tokens      = static_cast<std::int64_t>(width) * batch;
    aggregate.q.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.k.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.v.reserve(value_row_size * static_cast<std::size_t>(batch));
    aggregate.g.reserve(gate_row_size * static_cast<std::size_t>(batch));
    aggregate.beta.reserve(gate_row_size * static_cast<std::size_t>(batch));

    std::vector<gdn_ref::Inputs> rows;
    rows.reserve(static_cast<std::size_t>(batch));
    std::vector<float> initial_states(state_size * static_cast<std::size_t>(slots), 0.125f);
    for (int row = 0; row < batch; ++row) {
        gdn_ref::Inputs input =
            make_inputs(test_case, seed + static_cast<std::uint32_t>(row) * 97U);
        aggregate.q.insert(aggregate.q.end(), input.q.begin(), input.q.end());
        aggregate.k.insert(aggregate.k.end(), input.k.begin(), input.k.end());
        aggregate.v.insert(aggregate.v.end(), input.v.begin(), input.v.end());
        aggregate.g.insert(aggregate.g.end(), input.g.begin(), input.g.end());
        aggregate.beta.insert(aggregate.beta.end(), input.beta.begin(), input.beta.end());
        std::copy(input.state.begin(), input.state.end(),
                  initial_states.begin() +
                      static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(row)]) *
                          state_size);
        rows.push_back(std::move(input));
    }

    std::vector<gdn_ref::Result> references;
    references.reserve(static_cast<std::size_t>(batch));
    std::vector<double> expected_output(value_row_size * static_cast<std::size_t>(batch), 0.0);
    std::vector<bool> written_slots(static_cast<std::size_t>(slots), false);
    for (int row = 0; row < batch; ++row) {
        const int valid = masked ? valid_columns[static_cast<std::size_t>(row)] : width;
        gdn_ref::Inputs oracle_input = rows[static_cast<std::size_t>(row)];
        oracle_input.tokens          = valid;
        oracle_input.q.resize(static_cast<std::size_t>(kStateDim * test_case.qk_heads * valid));
        oracle_input.k.resize(static_cast<std::size_t>(kStateDim * test_case.qk_heads * valid));
        oracle_input.v.resize(static_cast<std::size_t>(kStateDim * test_case.value_heads * valid));
        oracle_input.g.resize(static_cast<std::size_t>(test_case.value_heads * valid));
        oracle_input.beta.resize(static_cast<std::size_t>(test_case.value_heads * valid));
        gdn_ref::Result reference = gdn_ref::evaluate(oracle_input, static_cast<double>(scale),
                                                      test_case.normalize_qk, true);
        std::copy(reference.out.begin(), reference.out.end(),
                  expected_output.begin() + static_cast<std::size_t>(row) * value_row_size);
        for (int column = 0; column < valid; ++column) {
            written_slots[static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(row)] +
                                                   column)] = true;
        }
        references.push_back(std::move(reference));
    }

    DeviceInputs device(aggregate);
    const std::vector<std::uint16_t> initial_bits = bf16_bits(initial_states);
    GuardedDeviceBuffer states(initial_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer out(aggregate.v.size() * sizeof(std::uint16_t));
    states.copy_from_host(initial_bits.data(), states.bytes());
    out.fill(0xff);
    DeviceBuffer device_initial_slots  = to_device(initial_slots);
    DeviceBuffer device_snapshot_bases = to_device(snapshot_bases);
    DeviceBuffer device_valid_columns;
    if (masked) { device_valid_columns = to_device(valid_columns); }

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, width, batch});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, width, batch});
    Tensor states_tensor(states.data(), DType::BF16,
                         {kStateDim, kStateDim, test_case.value_heads, slots});
    Tensor valid_tensor;
    if (masked) { valid_tensor = Tensor(device_valid_columns.p, DType::I32, {batch}); }
    Tensor initial_tensor(device_initial_slots.p, DType::I32, {batch});
    Tensor bases_tensor(device_snapshot_bases.p, DType::I32, {batch});
    Tensor out_tensor(out.data(), DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    ops::gated_delta_net_snapshot(q, k, v, g, beta, scale, test_case.normalize_qk, states_tensor,
                                  valid_tensor, initial_tensor, bases_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) +
                              " batched snapshot B=" + std::to_string(batch) +
                              (masked ? " masked" : " dense");
    int failures                         = 0;
    const std::vector<double> got_output = from_device_bf16(out.data(), aggregate.v.size());
    failures += verify_recurrence(label + " out", got_output, expected_output,
                                  gated_delta_net_output_bf16_criterion());
    if (masked) {
        const std::vector<std::uint16_t> output_bits =
            from_device<std::uint16_t>(out.data(), aggregate.v.size());
        for (int row = 0; row < batch; ++row) {
            for (int column = valid_columns[static_cast<std::size_t>(row)]; column < width;
                 ++column) {
                const std::size_t begin =
                    static_cast<std::size_t>(row) * value_row_size +
                    static_cast<std::size_t>(column) * kStateDim * test_case.value_heads;
                const std::size_t end =
                    begin + static_cast<std::size_t>(kStateDim * test_case.value_heads);
                if (!std::all_of(output_bits.begin() + begin, output_bits.begin() + end,
                                 [](std::uint16_t value) { return value == 0; })) {
                    std::cerr << label << ": invalid output tail is not exact zero\n";
                    ++failures;
                    row = batch;
                    break;
                }
            }
        }
    }

    const std::vector<std::uint16_t> got_states =
        from_device<std::uint16_t>(states.data(), initial_bits.size());
    for (int row = 0; row < batch; ++row) {
        const int valid = masked ? valid_columns[static_cast<std::size_t>(row)] : width;
        const std::size_t begin =
            static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(row)]) * state_size;
        failures += verify_recurrence(
            label + " row " + std::to_string(row) + " snapshots",
            bf16_doubles(got_states.begin() + begin,
                         got_states.begin() + begin +
                             static_cast<std::size_t>(valid) * state_size),
            references[static_cast<std::size_t>(row)].snapshots,
            gated_delta_net_state_criterion());
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (written_slots[static_cast<std::size_t>(slot)]) continue;
        const std::size_t begin = static_cast<std::size_t>(slot) * state_size;
        failures += verify_exact(
            label + " untouched slot " + std::to_string(slot),
            std::vector<std::uint16_t>(got_states.begin() + begin,
                                       got_states.begin() + begin + state_size),
            std::vector<std::uint16_t>(initial_bits.begin() + begin,
                                       initial_bits.begin() + begin + state_size));
    }
    failures +=
        verify_exact(label + " initial selectors unchanged",
                     from_device_i32(device_initial_slots, initial_slots.size()), initial_slots);
    failures +=
        verify_exact(label + " snapshot bases unchanged",
                     from_device_i32(device_snapshot_bases, snapshot_bases.size()), snapshot_bases);
    if (masked) {
        failures += verify_exact(label + " valid columns unchanged",
                                 from_device_i32(device_valid_columns, valid_columns.size()),
                                 valid_columns);
    }
    failures += states.verify_guards((label + " states").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, aggregate, device.q, device.k, device.v,
                                               device.g, device.beta);
    return failures;
}

int contract_rejection_cases() {
    DeviceBuffer q_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer k_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer v_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer g_buffer(8 * sizeof(float));
    DeviceBuffer beta_buffer(8 * sizeof(float));
    DeviceBuffer state_buffer(kStateDim * kStateDim * 8 * sizeof(float));
    DeviceBuffer out_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    WorkspaceArena workspace(256);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));

    auto is_rejected = [&](int activation_dim, int state_dim, int qk_heads, int value_heads) {
        Tensor q(q_buffer.p, DType::BF16, {activation_dim, qk_heads, 1});
        Tensor k(k_buffer.p, DType::BF16, {activation_dim, qk_heads, 1});
        Tensor v(v_buffer.p, DType::BF16, {activation_dim, value_heads, 1});
        Tensor g(g_buffer.p, DType::FP32, {value_heads, 1});
        Tensor beta(beta_buffer.p, DType::FP32, {value_heads, 1});
        Tensor state(state_buffer.p, DType::FP32, {state_dim, state_dim, value_heads});
        Tensor out(out_buffer.p, DType::BF16, {activation_dim, value_heads, 1});
        try {
            ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, state, out, nullptr);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures = 0;
    if (!is_rejected(64, kStateDim, 4, 8)) {
        std::cerr << "gated_delta_net accepted Q/K/V head dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, 64, 4, 8)) {
        std::cerr << "gated_delta_net accepted state dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, kStateDim, 4, 6)) {
        std::cerr << "gated_delta_net accepted a non-divisible head map\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    for (const bool normalize_qk : {false, true}) {
        const std::size_t interval =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 63, 65);
        const std::size_t witness =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 65, 65);
        if (interval != witness) {
            std::cerr << "gated_delta_net interval capacity missed the chunk boundary\n";
            ++failures;
        }
    }
    try {
        (void)ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, 0, 65);
        std::cerr << "gated_delta_net accepted an invalid token interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        (void)ops::gated_delta_net_workspace_capacity_bytes(4, 6, true, 1, 65);
        std::cerr << "gated_delta_net workspace accepted a non-divisible head map\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += contract_rejection_cases();
    for (const int tokens : {1, 4, 16, 63, 64, 65, 127, 128, 254, 256}) {
        failures += normalization_parity_case(tokens, false);
        if (tokens <= 16) { failures += normalization_parity_case(tokens, true); }
    }

    for (const int tokens : {65, 127, 129, 191, 254, 257}) {
        failures += partial_chunk_parity_case(tokens, false);
        failures += partial_chunk_parity_case(tokens, true);
    }

    for (const int tokens : {128, 192, 256}) {
        failures += partial_chunk_parity_case(tokens, false, true);
        failures += partial_chunk_parity_case(tokens, true, true);
    }

    for (int heads : {16, 32, 48}) for (bool normalize : {false, true}) {
        for (int tokens : {65, 127, 2047, 2053}) {
            failures += partial_chunk_parity_case(tokens, normalize, false, heads, true);
        }
    }

    // Registered 27B/35B-A3B geometries, public state forms, and the recurrent/chunk/tail route
    // boundary are all qualified directly against the same complete FP64 recurrence.
    failures += inplace_case({"27b decode fused-qk-norm", 16, 48, 1, true}, 12001u);
    failures += distinct_state_case({"27b raw-qk small-T", 16, 48, 7, false}, 12007u);
    failures += distinct_state_case({"35b pre-chunk fused-qk-norm", 16, 32, 63, true}, 12063u);
    failures += distinct_state_case({"27b exact chunk fused-qk-norm", 16, 48, 64, true}, 12064u);
    failures += distinct_state_case({"27b exact chunk raw-qk", 16, 48, 64, false}, 12164u);
    failures += inplace_case({"35b chunk-tail fused-qk-norm", 16, 32, 65, true}, 12065u);
    failures += distinct_state_case({"generic grouped-map chunk-tail", 3, 12, 65, true}, 12365u);
    failures += distinct_state_case({"27b two-chunk fused-qk-norm", 16, 48, 128, true}, 12128u);
    failures += inplace_case({"35b two-chunk raw-qk", 16, 32, 128, false}, 12228u);
    // surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b symmetric 16/16 heads
    // (identity head map), qualified across decode/small-T/chunk/tail routes.
    failures += inplace_case({"08b decode fused-qk-norm", 16, 16, 1, true}, 13001u);
    failures += distinct_state_case({"08b raw-qk small-T", 16, 16, 7, false}, 13007u);
    failures += distinct_state_case({"08b exact chunk fused-qk-norm", 16, 16, 64, true}, 13064u);
    failures += inplace_case({"08b chunk-tail fused-qk-norm", 16, 16, 65, true}, 13065u);
    failures += distinct_state_case({"08b two-chunk raw-qk", 16, 16, 128, false}, 13128u);

    // Snapshot is a separate public state transition. Nonzero source slots also prove that the
    // selected initial state, not slot zero, seeds the complete recurrence.
    failures += snapshot_case({"27b verify fused-qk-norm", 16, 48, 4, true}, 8, 7, 1, 12104u);
    failures += snapshot_case({"35b verify fused-qk-norm near-zero", 16, 32, 4, true, true}, 8, 6,
                              1, 12204u);
    failures +=
        batched_snapshot_case({"35b ordinary", 16, 32, 1, true}, {8, 9, 10, 11, 12, 13, 14, 15},
                              {0, 1, 2, 3, 4, 5, 6, 7}, {}, 16, 13001u);
    failures += batched_snapshot_case({"27b MTP", 16, 48, 6, true}, {18, 19, 20}, {0, 6, 12},
                                      {6, 3, 1}, 21, 13006u);
    // Row 0's initial state is its final destination; every state tile must load it before write.
    failures += batched_snapshot_case({"35b DFlash", 16, 32, 16, true}, {15, 33}, {0, 16}, {16, 7},
                                      34, 13016u);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net correctness\n";
    return failures == 0 ? 0 : 1;
}
