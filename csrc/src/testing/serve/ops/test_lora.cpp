// ops::lora_delta against a double-precision oracle.
//
// The adapter is the one thing in a served model that a deployment supplies and
// the engine cannot validate against a checkpoint hash, so the arithmetic has to
// be checked directly: a delta that is subtly wrong -- transposed, mis-scaled,
// added to the wrong projection -- produces fluent output that is simply worse,
// which no smoke test catches.

#include "api/ops/lora.h"

#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace sinfer;
using namespace sinfer::test;

int failures = 0;

std::uint16_t to_bf16(float value) {
    std::uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    word += 0x7FFFU + ((word >> 16U) & 1U);
    return static_cast<std::uint16_t>(word >> 16U);
}

float from_bf16(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value              = 0.0F;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<std::uint16_t> out(count);
    for (std::uint16_t& value : out) { value = to_bf16(dist(engine)); }
    return out;
}

Weight bf16_weight(const void* data, std::int32_t n, std::int32_t k) {
    Weight w;
    w.qtype           = QType::BF16_CTRL;
    w.payload         = data;
    w.qdata           = data;
    w.payload_bytes   = static_cast<std::uint64_t>(n) * k * sizeof(std::uint16_t);
    w.n               = n;
    w.k               = k;
    w.ndim            = 2;
    w.shape[0]        = n;
    w.shape[1]        = k;
    w.padded_shape[0] = n;
    w.padded_shape[1] = k;
    w.layout          = QuantLayout::Contiguous;
    return w;
}

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

/// out must become `base + B @ (A @ x)`, evaluated in double from the represented
/// BF16 inputs.
void test_delta(std::int32_t n, std::int32_t k, std::int32_t rank, std::int32_t tokens) {
    const auto host_x    = random_bf16(static_cast<std::size_t>(k) * tokens, 1, 1.0F);
    const auto host_a    = random_bf16(static_cast<std::size_t>(rank) * k, 2, 0.08F);
    const auto host_b    = random_bf16(static_cast<std::size_t>(n) * rank, 3, 0.08F);
    const auto host_base = random_bf16(static_cast<std::size_t>(n) * tokens, 4, 1.0F);

    GuardedDeviceBuffer x_dev(host_x.size() * sizeof(std::uint16_t));
    x_dev.copy_from_host(host_x.data(), x_dev.bytes());
    GuardedDeviceBuffer a_dev(host_a.size() * sizeof(std::uint16_t));
    a_dev.copy_from_host(host_a.data(), a_dev.bytes());
    GuardedDeviceBuffer b_dev(host_b.size() * sizeof(std::uint16_t));
    b_dev.copy_from_host(host_b.data(), b_dev.bytes());
    GuardedDeviceBuffer out_dev(host_base.size() * sizeof(std::uint16_t));
    out_dev.copy_from_host(host_base.data(), out_dev.bytes());
    GuardedDeviceBuffer scratch_dev(ops::lora_workspace_elements(rank, n, tokens) *
                                    sizeof(std::uint16_t));

    Tensor x(x_dev.data(), DType::BF16, {k, tokens});
    Tensor out(out_dev.data(), DType::BF16, {n, tokens});
    Tensor scratch(scratch_dev.data(), DType::BF16,
                   {static_cast<std::int32_t>(ops::lora_workspace_elements(rank, n, tokens))});

    ops::LoraWeights lora;
    lora.a    = bf16_weight(a_dev.data(), rank, k);
    lora.b    = bf16_weight(b_dev.data(), n, rank);
    lora.rank = rank;

    ops::lora_prepare(n, k, rank, tokens);
    ops::lora_delta(x, lora, out, scratch, nullptr);
    cuda_synchronize();
    std::vector<std::uint16_t> got(static_cast<std::size_t>(n) * tokens);
    out_dev.copy_to_host(got.data(), out_dev.bytes());

    double worst = 0.0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t row = 0; row < n; ++row) {
            // low[r] = sum_k A[r,k] x[k,t]; delta = sum_r B[row,r] low[r]
            double delta = 0.0;
            for (std::int32_t r = 0; r < rank; ++r) {
                double low = 0.0;
                for (std::int32_t i = 0; i < k; ++i) {
                    low += static_cast<double>(from_bf16(host_a[static_cast<std::size_t>(r) * k + i])) *
                           static_cast<double>(from_bf16(host_x[static_cast<std::size_t>(t) * k + i]));
                }
                // The intermediate is stored BF16 between the two GEMMs, so the
                // oracle rounds there too; comparing against an unrounded chain
                // would be measuring storage, not the op.
                low = static_cast<double>(from_bf16(to_bf16(static_cast<float>(low))));
                delta += static_cast<double>(
                             from_bf16(host_b[static_cast<std::size_t>(row) * rank + r])) *
                         low;
            }
            const double want =
                static_cast<double>(from_bf16(host_base[static_cast<std::size_t>(t) * n + row])) +
                delta;
            const double have =
                static_cast<double>(from_bf16(got[static_cast<std::size_t>(t) * n + row]));
            const double scale = std::max(1.0, std::abs(want));
            worst              = std::max(worst, std::abs(have - want) / scale);
        }
    }
    const std::string label = "lora_delta n=" + std::to_string(n) + " k=" + std::to_string(k) +
                              " r=" + std::to_string(rank) + " T=" + std::to_string(tokens);
    // BF16 has an 8-bit mantissa, so each storage point costs up to ~2e-3 relative,
    // and the delta passes through two of them: the intermediate `A @ x` and the
    // output. The bound is set from that, not tuned to the observed number, and it
    // is still two orders tighter than any structural error would be -- a
    // transposed operand or PEFT's alpha applied without dividing by r moves the
    // result by tens of percent, not by thousandths.
    check(worst < 1e-2, label + " (relative error " + std::to_string(worst) + ")");
    std::printf("  %-44s max relative error %.2e\n", label.c_str(), worst);
}

/// A mis-bound adapter must be refused, not applied.
void test_refusals() {
    const std::int32_t n = 64, k = 128, rank = 8, tokens = 4;
    const auto host_x = random_bf16(static_cast<std::size_t>(k) * tokens, 5, 1.0F);
    const auto host_a = random_bf16(static_cast<std::size_t>(rank) * k, 6, 0.1F);
    const auto host_b = random_bf16(static_cast<std::size_t>(n) * rank, 7, 0.1F);
    GuardedDeviceBuffer x_dev(host_x.size() * sizeof(std::uint16_t));
    x_dev.copy_from_host(host_x.data(), x_dev.bytes());
    GuardedDeviceBuffer a_dev(host_a.size() * sizeof(std::uint16_t));
    a_dev.copy_from_host(host_a.data(), a_dev.bytes());
    GuardedDeviceBuffer b_dev(host_b.size() * sizeof(std::uint16_t));
    b_dev.copy_from_host(host_b.data(), b_dev.bytes());
    GuardedDeviceBuffer out_dev(static_cast<std::size_t>(n) * tokens * sizeof(std::uint16_t));
    GuardedDeviceBuffer scratch_dev(ops::lora_workspace_elements(rank, n, tokens) *
                                    sizeof(std::uint16_t));

    Tensor x(x_dev.data(), DType::BF16, {k, tokens});
    Tensor out(out_dev.data(), DType::BF16, {n, tokens});
    Tensor scratch(scratch_dev.data(), DType::BF16,
                   {static_cast<std::int32_t>(ops::lora_workspace_elements(rank, n, tokens))});

    ops::LoraWeights wrong_k;
    wrong_k.a    = bf16_weight(a_dev.data(), rank, k / 2); // A for a different projection
    wrong_k.b    = bf16_weight(b_dev.data(), n, rank);
    wrong_k.rank = rank;
    bool threw   = false;
    try {
        ops::lora_delta(x, wrong_k, out, scratch, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "an A whose k is not the activation's is refused");

    ops::LoraWeights wrong_n;
    wrong_n.a    = bf16_weight(a_dev.data(), rank, k);
    wrong_n.b    = bf16_weight(b_dev.data(), n / 2, rank); // B for a different projection
    wrong_n.rank = rank;
    threw        = false;
    try {
        ops::lora_delta(x, wrong_n, out, scratch, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a B whose n is not the output's is refused");

    ops::LoraWeights fine;
    fine.a    = bf16_weight(a_dev.data(), rank, k);
    fine.b    = bf16_weight(b_dev.data(), n, rank);
    fine.rank = rank;
    Tensor tiny(scratch_dev.data(), DType::BF16, {rank}); // too small
    threw = false;
    try {
        ops::lora_delta(x, fine, out, tiny, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a scratch smaller than (rank + n) x tokens is refused");

    // rank 0 is "no adapter" and must be a no-op, not an error: it is how a
    // projection with no adapter attached reaches this call.
    ops::LoraWeights none;
    bool no_throw = true;
    try {
        ops::lora_delta(x, none, out, scratch, nullptr);
    } catch (...) { no_throw = false; }
    check(no_throw, "rank 0 applies nothing and raises nothing");
}

/// The batched path: one round, several adapters, some tokens on the base model.
///
/// This is the case the single-adapter op cannot express, so it is the one that
/// matters. The oracle picks each token's slot itself; a kernel that ignored
/// `ids` and used slot 0 for everything would pass a single-adapter test and
/// fail here.
void test_batched(std::int32_t n, std::int32_t k, std::int32_t rank, std::int32_t slots,
                  const std::vector<std::int32_t>& ids) {
    const auto tokens = static_cast<std::int32_t>(ids.size());
    const auto host_x = random_bf16(static_cast<std::size_t>(k) * tokens, 11, 1.0F);
    const auto host_a = random_bf16(static_cast<std::size_t>(slots) * rank * k, 12, 0.06F);
    const auto host_b = random_bf16(static_cast<std::size_t>(slots) * n * rank, 13, 0.06F);
    const auto host_base = random_bf16(static_cast<std::size_t>(n) * tokens, 14, 1.0F);

    GuardedDeviceBuffer x_dev(host_x.size() * sizeof(std::uint16_t));
    x_dev.copy_from_host(host_x.data(), x_dev.bytes());
    GuardedDeviceBuffer a_dev(host_a.size() * sizeof(std::uint16_t));
    a_dev.copy_from_host(host_a.data(), a_dev.bytes());
    GuardedDeviceBuffer b_dev(host_b.size() * sizeof(std::uint16_t));
    b_dev.copy_from_host(host_b.data(), b_dev.bytes());
    GuardedDeviceBuffer out_dev(host_base.size() * sizeof(std::uint16_t));
    out_dev.copy_from_host(host_base.data(), out_dev.bytes());
    GuardedDeviceBuffer ids_dev(ids.size() * sizeof(std::int32_t));
    ids_dev.copy_from_host(ids.data(), ids_dev.bytes());
    GuardedDeviceBuffer scratch_dev(
        ops::lora_batched_workspace_elements(rank, tokens) * sizeof(std::uint16_t));

    Tensor x(x_dev.data(), DType::BF16, {k, tokens});
    Tensor out(out_dev.data(), DType::BF16, {n, tokens});
    Tensor id_tensor(ids_dev.data(), DType::I32, {tokens});
    Tensor scratch(scratch_dev.data(), DType::BF16,
                   {static_cast<std::int32_t>(ops::lora_batched_workspace_elements(rank, tokens))});

    ops::LoraBank bank;
    bank.a        = a_dev.data();
    bank.b        = b_dev.data();
    bank.a_stride = static_cast<std::int64_t>(rank) * k;
    bank.b_stride = static_cast<std::int64_t>(n) * rank;
    bank.rank     = rank;
    bank.n        = n;
    bank.k        = k;

    ops::lora_delta_batched(x, bank, id_tensor, nullptr, out, scratch, nullptr);
    cuda_synchronize();
    std::vector<std::uint16_t> got(static_cast<std::size_t>(n) * tokens);
    out_dev.copy_to_host(got.data(), out_dev.bytes());

    double worst = 0.0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t slot = ids[static_cast<std::size_t>(t)];
        for (std::int32_t row = 0; row < n; ++row) {
            double delta = 0.0;
            if (slot >= 0) {
                for (std::int32_t r = 0; r < rank; ++r) {
                    double low = 0.0;
                    for (std::int32_t i = 0; i < k; ++i) {
                        low += static_cast<double>(from_bf16(
                                   host_a[(static_cast<std::size_t>(slot) * rank + r) * k + i])) *
                               static_cast<double>(
                                   from_bf16(host_x[static_cast<std::size_t>(t) * k + i]));
                    }
                    low = static_cast<double>(from_bf16(to_bf16(static_cast<float>(low))));
                    delta += static_cast<double>(from_bf16(
                                 host_b[(static_cast<std::size_t>(slot) * n + row) * rank + r])) *
                             low;
                }
            }
            const double want =
                static_cast<double>(from_bf16(host_base[static_cast<std::size_t>(t) * n + row])) +
                delta;
            const double have =
                static_cast<double>(from_bf16(got[static_cast<std::size_t>(t) * n + row]));
            worst = std::max(worst, std::abs(have - want) / std::max(1.0, std::abs(want)));
        }
    }
    const std::string label = "batched n=" + std::to_string(n) + " r=" + std::to_string(rank) +
                              " slots=" + std::to_string(slots) + " T=" + std::to_string(tokens);
    check(worst < 1e-2, label + " (relative error " + std::to_string(worst) + ")");
    std::printf("  %-44s max relative error %.2e\n", label.c_str(), worst);
}

} // namespace

int main() {
    try {
        test_delta(64, 128, 8, 1);    // decode width
        test_delta(256, 512, 16, 4);  // speculative verify width
        test_delta(512, 2048, 32, 37); // a prefill slice, rank 32
        test_refusals();
        // Several adapters and base-model tokens in one round, which is the point.
        test_batched(256, 512, 16, 4, {0, 1, 2, 3, -1, 0, 2, -1});
        test_batched(1024, 1024, 8, 2, {1, -1, 0, 0});
        test_batched(64, 128, 4, 1, {-1, -1});        // every token on the base model
        test_batched(512, 2048, 32, 3, {2, 2, 2, 2}); // one adapter, wide rank
    } catch (const std::exception& error) {
        std::cerr << "lora: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "lora: " << failures << " case(s) failed\n";
        return 1;
    }
    std::printf("OK lora\n");
    return 0;
}
