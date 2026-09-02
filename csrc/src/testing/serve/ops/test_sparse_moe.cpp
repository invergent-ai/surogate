#include "ops/parallel_rows.h"

#include "api/ops/sparse_moe.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <future>
#include <thread>
#include <utility>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

// SparseMoe is the exact qwen3_6_35b_a3b post-mixer Op. qwen3_6_27b has a dense SwiGLU
// post-mixer and therefore contributes no second SparseMoe geometry.
constexpr std::int32_t kHidden         = 2048;
constexpr std::int32_t kExperts        = 256;
constexpr std::int32_t kTopK           = 8;
constexpr std::int32_t kIntermediate   = 512;
constexpr std::int32_t kExpertGateRows = 2 * kIntermediate;
constexpr std::int32_t kRoutedGateRows = kExperts * kExpertGateRows;
constexpr std::int32_t kRoutedDownRows = kExperts * kHidden;
constexpr std::int32_t kSharedGateRows = 2 * kIntermediate;

// All registered variants consume represented BF16 activations and expose one BF16 destination.
// The bound belongs to that A16 compute profile, not to a codec, T, private route, or schedule.
// The limits retain modest headroom over the measured maxima from the complete case matrix:
// rel-L2 1.117e-2 and pointwise 3.687e-3.
constexpr ReductionCriterion kSparseMoeA16Tolerance{
    /*relative_l2*/ 1.2e-2,
    /*gross_absolute*/ 4.0e-3,
    /*gross_relative_to_max_reference*/ 0.0,
};

// The NVFP4 routed profile is served by a W4A4 runner, so the oracle rounds the activations to
// E2M1 as well (`round_activation_to_nvfp4`) and this bound covers only what is left: E2M1 ties
// broken differently from the hardware converter, the runner's fast reciprocal, and FP32 against
// the oracle's FP64. Measured maxima over the complete case matrix are rel-L2 2.049e-2 and
// pointwise 1.151e-2; without the activation model the same cases sit at 9.86e-2, which is the
// format's own headroom on this fixture's narrow-range input rather than an error of the kernel.
constexpr ReductionCriterion kSparseMoeA4Tolerance{
    /*relative_l2*/ 2.5e-2,
    /*gross_absolute*/ 1.5e-2,
    /*gross_relative_to_max_reference*/ 0.0,
};

constexpr std::size_t kOutputGuardBytes = 256;
constexpr std::uint8_t kOutputGuardByte = 0xa5;

struct QuantGeometry {
    std::int32_t group;
    std::size_t code_bytes_per_group;
    std::size_t high_bytes_per_group;
    std::size_t scale_bytes_per_group = 2;   // fp16 for the row-split codecs, e4m3 for NVFP4
    bool block_scale                  = false; // NVFP4 stores scales in 128x4 tiles, not per row
};

QuantGeometry quant_geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return {64, 32, 0};
    case QType::Q5G64_F16S:
        return {64, 32, 8};
    case QType::Q6G64_F16S:
        return {64, 32, 16};
    case QType::W8G32_F16S:
        return {32, 32, 0};
    case QType::NVFP4:
        return {16, 8, 0, 1, true};
    default:
        throw std::invalid_argument("sparse_moe test: unsupported codec");
    }
}

// Quantises a float source into NVFP4 for the routed-expert arm: e2m1 codes row-major, two per
// byte, and one e4m3 scale per 16 values written through the BlockScaleK16M128x4 swizzle the
// engine and the artifact use. The shared fixture cannot do this — `pack_row_split_lowbit`
// handles only row-split codecs, and the options-based packer synthesises scale *patterns*
// rather than quantising real values, which an oracle comparison needs.
/// `block_scale` is the format's second level, one entry per equal band of rows — gate and up
/// for a stacked gate/up expert, one for a down expert — exactly as the checkpoint stores it per
/// projection. The packer quantises `source / block_scale` and leaves the multiply to the kernel,
/// so `dequant` (the oracle's view) still holds the full weight and a kernel that forgets the
/// multiply fails by the size of the scale rather than by rounding.
/// An expert's gate/up rows with the two halves exchanged. The vendored runner reads [up; gate];
/// every other codec, and this file's oracle, reads [gate; up].
std::vector<float> swap_halves(const std::vector<float>& source, std::int32_t columns) {
    const std::size_t half = source.size() / 2;
    std::vector<float> out(source.size());
    std::copy(source.begin() + static_cast<std::ptrdiff_t>(half), source.end(), out.begin());
    std::copy(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(half),
              out.begin() + static_cast<std::ptrdiff_t>(half));
    (void)columns;
    return out;
}

/// Undoes `swap_halves` on a packed weight's decoded values, so the oracle keeps reading
/// [gate; up] while the device payload stays in the runner's order.
quantized_weight::PackedWeight swap_halves_of_dequant(quantized_weight::PackedWeight packed,
                                                      std::int32_t columns) {
    packed.dequant = swap_halves(packed.dequant, columns);
    return packed;
}

quantized_weight::PackedWeight pack_nvfp4(const std::vector<float>& source, std::int32_t n,
                                          std::int32_t k,
                                          const std::vector<float>& block_scale) {
    if ((n % 128) != 0 || (k % 64) != 0 || source.size() != static_cast<std::size_t>(n) * k) {
        throw std::invalid_argument("nvfp4 test packer: N must be a multiple of 128, K of 64");
    }
    if (block_scale.empty() || (n % static_cast<std::int32_t>(block_scale.size())) != 0) {
        throw std::invalid_argument("nvfp4 test packer: block_scale must divide the rows");
    }
    const std::int32_t rows_per_scale = n / static_cast<std::int32_t>(block_scale.size());
    // E2M1 magnitudes, index = code & 7, sign in bit 3.
    static constexpr float kMagnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const std::int32_t groups_per_row     = k / 16;
    const std::int32_t k_tiles            = k / 64;

    quantized_weight::PackedWeight packed;
    packed.code_plane_bytes   = static_cast<std::uint64_t>(n) * k / 2;
    packed.scale_plane_offset = (packed.code_plane_bytes + 255U) / 256U * 256U;
    packed.scale_plane_bytes  = static_cast<std::uint64_t>(n) * k / 16;
    packed.weight_divisor_offset = packed.scale_plane_offset + packed.scale_plane_bytes;
    packed.payload.assign(static_cast<std::size_t>(packed.weight_divisor_offset) + 4U, 0);
    packed.dequant.assign(source.size(), 0.0f);

    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t group = 0; group < groups_per_row; ++group) {
            const std::size_t base = static_cast<std::size_t>(row) * k + group * 16;
            const float second       = block_scale[static_cast<std::size_t>(row / rows_per_scale)];
            const float inverse_second = 1.0f / second;
            float amax                 = 0.0f;
            for (int i = 0; i < 16; ++i) {
                amax = std::max(amax, std::fabs(source[base + i]) * inverse_second);
            }
            // Scale so the largest magnitude lands on E2M1's 6, then round it to a real e4m3
            // value — the kernel decodes the stored byte, so the reference must use that same
            // rounded scale rather than the ideal one.
            const float wanted = amax > 0.0f ? amax / 6.0f : 1.0f;
            const __nv_fp8_e4m3 stored_scale(wanted);
            const float scale = static_cast<float>(stored_scale);
            const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;

            for (int i = 0; i < 16; ++i) {
                const float value      = source[base + i] * inverse_second;
                const float normalised = std::fabs(value) * inverse;
                int best = 0;
                float best_error = std::fabs(normalised - kMagnitudes[0]);
                for (int candidate = 1; candidate < 8; ++candidate) {
                    const float error = std::fabs(normalised - kMagnitudes[candidate]);
                    if (error < best_error) { best_error = error; best = candidate; }
                }
                const bool negative = std::signbit(value);
                const std::uint8_t code =
                    static_cast<std::uint8_t>((negative ? 0x8u : 0x0u) | static_cast<unsigned>(best));
                const std::size_t byte = (static_cast<std::size_t>(row) * k + group * 16 + i) / 2;
                if ((i & 1) == 0) {
                    packed.payload[byte] = static_cast<std::uint8_t>((packed.payload[byte] & 0xF0u) | code);
                } else {
                    packed.payload[byte] =
                        static_cast<std::uint8_t>((packed.payload[byte] & 0x0Fu) | (code << 4));
                }
                packed.dequant[base + i] =
                    (negative ? -1.0f : 1.0f) * kMagnitudes[best] * scale * second;
            }

            const std::int32_t row_inner = row % 128;
            const std::size_t offset =
                packed.scale_plane_offset +
                static_cast<std::size_t>((row / 128) * k_tiles + group / 4) * 512U +
                static_cast<std::size_t>(row_inner % 32) * 16U +
                static_cast<std::size_t>(row_inner / 32) * 4U +
                static_cast<std::size_t>(group % 4);
            packed.payload[offset] = stored_scale.__x;
        }
    }

    packed.weight.qtype                = QType::NVFP4;
    packed.weight.layout               = QuantLayout::BlockScaleK16M128x4;
    packed.weight.scale_dtype          = DType::FP8_E4M3FN;
    packed.weight.payload              = packed.payload.data();
    packed.weight.payload_bytes        = packed.payload.size();
    packed.weight.qdata                = packed.payload.data();
    packed.weight.scales               = packed.payload.data() + packed.scale_plane_offset;
    packed.weight.group_size           = 16;
    packed.weight.group                = 16;
    packed.weight.ndim                 = 2;
    packed.weight.shape[0]             = n;
    packed.weight.shape[1]             = k;
    packed.weight.padded_shape[0]      = n;
    packed.weight.padded_shape[1]      = k;
    packed.weight.n                    = n;
    packed.weight.k                    = k;
    packed.weight.weight_scale_divisor = 1.0f;
    packed.weight.input_scale_divisor  = 1.0f;
    return packed;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

int compare_output(const std::string& label, const std::vector<double>& actual,
                   const std::vector<double>& reference, const ReductionCriterion& criterion) {
    return verify_reduction(label, actual, reference, criterion);
}

class GuardedBf16Output {
public:
    explicit GuardedBf16Output(std::size_t words)
        : storage_(words * sizeof(std::uint16_t), kOutputGuardBytes, kOutputGuardByte),
          words_(words) {}

    void* data() noexcept { return storage_.data(); }

    const void* data() const noexcept { return storage_.data(); }

    std::vector<double> values() const { return from_device_bf16(data(), words_); }

    int verify_guards(const std::string& label) const { return storage_.verify_guards(label); }

private:
    GuardedDeviceBuffer storage_;
    std::size_t words_;
};

class DeviceRowSplit {
public:
    DeviceRowSplit(QType qtype, std::int32_t rows, std::int32_t columns)
        : qtype_(qtype), rows_(rows), columns_(columns), geometry_(quant_geometry(qtype)),
          groups_per_row_(columns / geometry_.group),
          code_row_bytes_(static_cast<std::size_t>(groups_per_row_) *
                          geometry_.code_bytes_per_group),
          high_row_bytes_(static_cast<std::size_t>(groups_per_row_) *
                          geometry_.high_bytes_per_group),
          scale_row_bytes_(static_cast<std::size_t>(groups_per_row_) *
                           geometry_.scale_bytes_per_group),
          codes_(static_cast<std::size_t>(rows) * code_row_bytes_),
          scales_(static_cast<std::size_t>(rows) * scale_row_bytes_) {
        codes_.fill(0);
        scales_.fill(0);
        if (high_row_bytes_ != 0) {
            high_ =
                std::make_unique<DeviceBuffer>(static_cast<std::size_t>(rows) * high_row_bytes_);
            high_->fill(0);
        }
    }

    void copy_rows(const quantized_weight::PackedWeight& source, std::int32_t destination_row) {
        const std::int32_t source_rows = source.weight.n;
        if (source.weight.qtype != qtype_ || source.weight.k != columns_ || source_rows < 1 ||
            destination_row < 0 || destination_row > rows_ - source_rows) {
            throw std::invalid_argument("sparse_moe test: invalid packed row copy");
        }
        codes_.copy_from_host(source.payload.data(),
                              static_cast<std::size_t>(source_rows) * code_row_bytes_,
                              static_cast<std::size_t>(destination_row) * code_row_bytes_);
        if (high_row_bytes_ != 0) {
            high_->copy_from_host(source.payload.data() + source.high_plane_offset,
                                  static_cast<std::size_t>(source_rows) * high_row_bytes_,
                                  static_cast<std::size_t>(destination_row) * high_row_bytes_);
        }
        scales_.copy_from_host(source.payload.data() + source.scale_plane_offset,
                               static_cast<std::size_t>(source_rows) * scale_row_bytes_,
                               static_cast<std::size_t>(destination_row) * scale_row_bytes_);
    }

    Weight weight() const {
        Weight result{};
        result.payload          = codes_.p;
        result.payload_bytes    = codes_.bytes + (high_ ? high_->bytes : 0) + scales_.bytes;
        result.high_plane_bytes = high_ ? high_->bytes : 0;
        result.qtype            = qtype_;
        result.group_size       = static_cast<std::uint32_t>(geometry_.group);
        result.qdata            = codes_.p;
        result.qhigh            = high_ ? high_->p : nullptr;
        result.scales           = scales_.p;
        result.n                = rows_;
        result.k                = columns_;
        result.group            = geometry_.group;
        result.layout      = geometry_.block_scale ? QuantLayout::BlockScaleK16M128x4
                                                   : QuantLayout::RowSplit;
        result.scale_dtype = geometry_.block_scale ? DType::FP8_E4M3FN : DType::FP16;
        result.ndim             = 2;
        result.shape[0]         = rows_;
        result.shape[1]         = columns_;
        result.padded_shape[0]  = rows_;
        result.padded_shape[1]  = columns_;
        return result;
    }

    int verify_rows(const std::string& label, const quantized_weight::PackedWeight& source,
                    std::int32_t destination_row) const {
        const std::int32_t source_rows = source.weight.n;
        int failures                   = 0;
        failures += verify_plane(label + " code", codes_, source.payload.data(),
                                 static_cast<std::size_t>(destination_row) * code_row_bytes_,
                                 static_cast<std::size_t>(source_rows) * code_row_bytes_);
        if (high_row_bytes_ != 0) {
            failures += verify_plane(label + " high", *high_,
                                     source.payload.data() + source.high_plane_offset,
                                     static_cast<std::size_t>(destination_row) * high_row_bytes_,
                                     static_cast<std::size_t>(source_rows) * high_row_bytes_);
        }
        failures += verify_plane(label + " scale", scales_,
                                 source.payload.data() + source.scale_plane_offset,
                                 static_cast<std::size_t>(destination_row) * scale_row_bytes_,
                                 static_cast<std::size_t>(source_rows) * scale_row_bytes_);
        return failures;
    }

private:
    static int verify_plane(const std::string& label, const DeviceBuffer& device,
                            const std::uint8_t* expected, std::size_t offset, std::size_t bytes) {
        std::vector<std::uint8_t> actual(bytes);
        device.copy_to_host(actual.data(), bytes, offset);
        if (std::memcmp(actual.data(), expected, bytes) == 0) { return 0; }
        std::cerr << label << ": persistent weight was modified\n";
        return 1;
    }

    QType qtype_;
    std::int32_t rows_;
    std::int32_t columns_;
    QuantGeometry geometry_;
    std::int32_t groups_per_row_;
    std::size_t code_row_bytes_;
    std::size_t high_row_bytes_;
    std::size_t scale_row_bytes_;
    DeviceBuffer codes_;
    std::unique_ptr<DeviceBuffer> high_;
    DeviceBuffer scales_;
};

Weight dense_bf16_weight(void* data, std::int32_t rows, std::int32_t columns) {
    Weight result{};
    result.payload         = data;
    result.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * sizeof(std::uint16_t);
    result.qtype           = QType::BF16_CTRL;
    result.qdata           = data;
    result.n               = rows;
    result.k               = columns;
    result.layout          = QuantLayout::Contiguous;
    result.ndim            = 2;
    result.shape[0]        = rows;
    result.shape[1]        = columns;
    result.padded_shape[0] = rows;
    result.padded_shape[1] = columns;
    return result;
}

struct RoutePattern {
    std::array<int, kTopK> selected;
    int tied_excluded = -1;
};

constexpr std::array<RoutePattern, 3> kRoutePatterns{{
    {{{255, 0, 17, 31, 63, 127, 191, 223}}, -1},
    {{{0, 17, 31, 63, 127, 191, 223, 254}}, 255},
    {{{223, 191, 127, 63, 31, 17, 0, 255}}, -1},
}};

std::vector<float> make_input(int pattern) {
    std::vector<float> input(kHidden);
    input[0] = 1.0f;
    for (int column = 1; column < kHidden; ++column) {
        input[column] = 0.025f + static_cast<float>((column * 7 + pattern * 11) % 19) * 0.002f;
    }
    for (std::size_t marker = 0; marker < kRoutePatterns.size(); ++marker) {
        input[kHidden - static_cast<int>(kRoutePatterns.size()) + static_cast<int>(marker)] = 0.0f;
    }
    input[kHidden - static_cast<int>(kRoutePatterns.size()) + pattern] = 1.0f;
    round_to_bf16(input);
    return input;
}

std::vector<float> make_residual(int pattern) {
    std::vector<float> residual(kHidden);
    for (int row = 0; row < kHidden; ++row) {
        residual[row] = 0.125f + static_cast<float>((row * 5 + pattern * 13) % 23) * 0.003f;
    }
    round_to_bf16(residual);
    return residual;
}

std::vector<float> make_router() {
    std::vector<float> router(static_cast<std::size_t>(kExperts + 1) * kHidden);
    for (int row = 0; row < kExperts + 1; ++row) {
        for (int column = 0; column < kHidden; ++column) {
            const int pattern = (column * 13 + 5) % 17 - 8;
            router[static_cast<std::size_t>(row) * kHidden + column] =
                static_cast<float>(pattern) * 0.001f;
        }
    }
    for (int expert = 0; expert < kExperts; ++expert) {
        router[static_cast<std::size_t>(expert) * kHidden] -= 8.0f;
    }
    for (std::size_t pattern = 0; pattern < kRoutePatterns.size(); ++pattern) {
        const int marker =
            kHidden - static_cast<int>(kRoutePatterns.size()) + static_cast<int>(pattern);
        const RoutePattern& route = kRoutePatterns[pattern];
        for (int rank = 0; rank < kTopK; ++rank) {
            const float score =
                rank == kTopK - 1 && route.tied_excluded >= 0 ? 2.0f : 4.0f - 0.25f * rank;
            router[static_cast<std::size_t>(route.selected[rank]) * kHidden + marker] +=
                score + 8.0f;
        }
        if (route.tied_excluded >= 0) {
            router[static_cast<std::size_t>(route.tied_excluded) * kHidden + marker] += 10.0f;
        }
    }
    router[static_cast<std::size_t>(kExperts) * kHidden] += 0.375f;
    round_to_bf16(router);
    return router;
}

std::vector<float> make_gate_up(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                                float expert_factor) {
    std::vector<float> source(static_cast<std::size_t>(rows) * columns);
    const std::int32_t split = rows / 2;
    for (std::int32_t row = 0; row < rows; ++row) {
        const float bias       = row < split ? 0.75f : 1.15f;
        const float row_factor = 1.0f + static_cast<float>((row + seed) % 7) * 0.025f;
        for (std::int32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((row * 11LL + column * 5LL + seed) % 15) - 7;
            source[static_cast<std::size_t>(row) * columns + column] =
                0.008f * expert_factor * row_factor * (static_cast<float>(pattern) + bias);
        }
    }
    return source;
}

std::vector<float> make_down(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                             float expert_factor) {
    std::vector<float> source(static_cast<std::size_t>(rows) * columns);
    for (std::int32_t row = 0; row < rows; ++row) {
        const float bias = 0.45f + static_cast<float>((row + seed) % 5) * 0.08f;
        for (std::int32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((row * 7LL + column * 13LL + seed) % 17) - 8;
            source[static_cast<std::size_t>(row) * columns + column] =
                0.007f * expert_factor * (static_cast<float>(pattern) + bias);
        }
    }
    return source;
}

struct HostExpert {
    int id;
    quantized_weight::PackedWeight gate_up;
    quantized_weight::PackedWeight down;
};

const HostExpert& find_expert(const std::vector<HostExpert>& experts, int id) {
    const auto found = std::find_if(experts.begin(), experts.end(),
                                    [id](const HostExpert& expert) { return expert.id == id; });
    if (found == experts.end()) {
        throw std::logic_error("sparse_moe oracle selected an unpopulated expert");
    }
    return *found;
}


double dot_fp64(const std::vector<float>& matrix, std::int32_t row, std::int32_t columns,
                const std::vector<double>& input) {
    const float* weights = matrix.data() + static_cast<std::size_t>(row) * columns;
    double result        = 0.0;
    for (std::int32_t column = 0; column < columns; ++column) {
        result += static_cast<double>(weights[column]) * input[column];
    }
    return result;
}

/// The routed experts of the NVFP4 profile are served by a W4A4 runner, so the activations reach
/// the MMA rounded to E2M1 as well. That rounding is part of the arithmetic the profile defines,
/// not an error of the implementation, so the oracle performs it — from the format's own rules,
/// independently of how the kernel spells them.
///
/// A block of 16 shares one E4M3 scale derived from the block's own maximum and the expert's
/// activation global scale: `SF = e4m3(vecMax * scale / 6)`, and the value the MMA sees is
/// `round_e2m1(v * scale / SF) * SF / scale`. The global scale cancels except through that E4M3
/// rounding, which is exactly why it only has to put `SF` inside E4M3's normal range.
struct Nvfp4ActivationModel {
    const std::vector<float>* gate_up_scale = nullptr;
    const std::vector<float>* down_scale    = nullptr;

    [[nodiscard]] bool enabled() const noexcept { return gate_up_scale != nullptr; }
};

void round_activation_to_nvfp4(std::vector<double>& values, float activation_scale) {
    static constexpr double kMagnitudes[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    constexpr std::size_t kBlock           = 16;
    for (std::size_t base = 0; base < values.size(); base += kBlock) {
        double block_max = 0.0;
        for (std::size_t i = base; i < base + kBlock; ++i) {
            block_max = std::max(block_max, std::fabs(values[i]));
        }
        const __nv_fp8_e4m3 stored(static_cast<float>(block_max * activation_scale / 6.0));
        const double block_scale = static_cast<double>(static_cast<float>(stored));
        if (!(block_scale > 0.0)) {
            for (std::size_t i = base; i < base + kBlock; ++i) { values[i] = 0.0; }
            continue;
        }
        const double step = block_scale / static_cast<double>(activation_scale);
        for (std::size_t i = base; i < base + kBlock; ++i) {
            const double normalised = std::fabs(values[i]) / step;
            int best                = 0;
            double best_error       = std::fabs(normalised - kMagnitudes[0]);
            for (int candidate = 1; candidate < 8; ++candidate) {
                const double error = std::fabs(normalised - kMagnitudes[candidate]);
                if (error < best_error) {
                    best_error = error;
                    best       = candidate;
                }
            }
            values[i] = std::copysign(kMagnitudes[best] * step, values[i]);
        }
    }
}

// The one SparseMoe oracle. It directly evaluates the complete public formula from represented
// BF16 values and independently decoded logical weights, with FP64 accumulation throughout.
// It has no production route, staging cast, workspace dtype, reduction tree, or output rounding.
std::vector<double> sparse_moe_oracle(const std::vector<float>& input,
                                      const std::vector<float>& residual,
                                      const std::vector<float>& router,
                                      const std::vector<HostExpert>& experts,
                                      const quantized_weight::PackedWeight& shared_gate_up,
                                      const quantized_weight::PackedWeight& shared_down,
                                      const RoutePattern& intended_route,
                                      std::span<const int> excluded         = {},
                                      const Nvfp4ActivationModel& activation = {}) {
    const std::vector<double> x(input.begin(), input.end());
    std::vector<double> scores(kExperts + 1);
    sinfer::test::parallel_rows(kExperts + 1,
                  [&](std::int32_t row) { scores[row] = dot_fp64(router, row, kHidden, x); });

    std::vector<int> ranked(kExperts);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::sort(ranked.begin(), ranked.end(), [&](int left, int right) {
        return scores[left] > scores[right] || (scores[left] == scores[right] && left < right);
    });

    std::array<int, kTopK> selected{};
    std::array<double, kTopK> route_weight{};
    double route_denominator = 0.0;
    for (int route = 0; route < kTopK; ++route) {
        selected[route]     = ranked[route];
        route_weight[route] = std::exp(scores[selected[route]] - scores[selected[0]]);
        route_denominator += route_weight[route];
    }
    for (double& weight : route_weight) { weight /= route_denominator; }
    if (selected != intended_route.selected) {
        throw std::logic_error("sparse_moe test router did not create its intended ordered top-8");
    }
    const double shared_scale = 1.0 / (1.0 + std::exp(-scores[kExperts]));

    std::array<std::vector<double>, kTopK> routed_activation;
    for (int route = 0; route < kTopK; ++route) {
        routed_activation[route].resize(kIntermediate);
        const HostExpert& expert = find_expert(experts, selected[route]);
        std::vector<double> expert_x = x;
        if (activation.enabled()) {
            round_activation_to_nvfp4(
                expert_x, (*activation.gate_up_scale)[static_cast<std::size_t>(expert.id)]);
        }
        sinfer::test::parallel_rows(kIntermediate, [&](std::int32_t row) {
            double gate = dot_fp64(expert.gate_up.dequant, row, kHidden, expert_x);
            double up   = dot_fp64(expert.gate_up.dequant, kIntermediate + row, kHidden, expert_x);
            if (activation.enabled()) {
                // The runner writes the first GEMM's result as BF16 and reads it back to form the
                // SwiGLU, so the reference rounds there too.
                gate = static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(gate))));
                up   = static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(up))));
            }
            routed_activation[route][row] = (gate / (1.0 + std::exp(-gate))) * up;
        });
        if (activation.enabled()) {
            round_activation_to_nvfp4(
                routed_activation[route],
                (*activation.down_scale)[static_cast<std::size_t>(expert.id)]);
        }
    }

    std::vector<double> shared_activation(kIntermediate);
    sinfer::test::parallel_rows(kIntermediate, [&](std::int32_t row) {
        const double gate      = dot_fp64(shared_gate_up.dequant, row, kHidden, x);
        const double up        = dot_fp64(shared_gate_up.dequant, kIntermediate + row, kHidden, x);
        shared_activation[row] = (gate / (1.0 + std::exp(-gate))) * up;
    });

    std::vector<double> output(kHidden);
    sinfer::test::parallel_rows(kHidden, [&](std::int32_t row) {
        double value = static_cast<double>(residual[row]);
        for (int route = 0; route < kTopK; ++route) {
            // A path served elsewhere (the round hook's host split) keeps its routing weight
            // and contributes nothing here.
            if (std::find(excluded.begin(), excluded.end(), selected[route]) != excluded.end()) {
                continue;
            }
            const HostExpert& expert = find_expert(experts, selected[route]);
            value += route_weight[route] *
                     dot_fp64(expert.down.dequant, row, kIntermediate, routed_activation[route]);
        }
        value +=
            shared_scale * dot_fp64(shared_down.dequant, row, kIntermediate, shared_activation);
        output[row] = value;
    });
    return output;
}

struct CodecProfile {
    const char* name;
    QType routed_gate_up;
    QType routed_down;
    std::span<const std::int32_t> token_cases;
    bool verify_graph_replay;
};

class SparseMoeFixture {
public:
    explicit SparseMoeFixture(const CodecProfile& profile)
        : profile_(profile), router_(make_router()), router_bits_(bf16_bits(router_)),
          device_router_(to_device(router_bits_)),
          routed_gate_(profile.routed_gate_up, kRoutedGateRows, kHidden),
          routed_down_(profile.routed_down, kRoutedDownRows, kIntermediate),
          shared_gate_(QType::W8G32_F16S, kSharedGateRows, kHidden),
          shared_down_device_(QType::W8G32_F16S, kHidden, kIntermediate),
          gate_up_scale_host_(static_cast<std::size_t>(kExperts) * 2, 1.0f),
          down_scale_host_(static_cast<std::size_t>(kExperts), 1.0f) {
        for (int pattern = 0; pattern < static_cast<int>(kRoutePatterns.size()); ++pattern) {
            inputs_.push_back(make_input(pattern));
            residuals_.push_back(make_residual(pattern));
        }

        std::vector<int> expert_ids;
        for (const RoutePattern& route : kRoutePatterns) {
            for (int expert : route.selected) {
                if (std::find(expert_ids.begin(), expert_ids.end(), expert) == expert_ids.end()) {
                    expert_ids.push_back(expert);
                }
            }
        }
        std::sort(expert_ids.begin(), expert_ids.end());
        for (int expert : expert_ids) {
            const float factor = 0.8f + static_cast<float>((expert * 3) % 11) * 0.045f;
            const auto gate_up_source = make_gate_up(
                kExpertGateRows, kHidden, 100U + static_cast<std::uint32_t>(expert), factor);
            const auto down_source = make_down(
                kHidden, kIntermediate, 300U + static_cast<std::uint32_t>(expert), factor);
            // The checkpoint's second-level scales differ per expert and per projection (3-7x
            // across one layer's experts, measured on RedHatAI/Qwen3.6-35B-A3B-NVFP4), so the
            // test gives every expert its own gate, up and down scale rather than a shared one.
            const float gate_scale = 0.55f + static_cast<float>((expert * 7) % 13) * 0.07f;
            // The vendored runner carries one epilogue alpha per expert, so the NVFP4 profile's
            // gate and up share a second-level scale — as they do in the published checkpoint,
            // which the converter refuses to convert when they do not.
            const float up_scale = profile.routed_gate_up == QType::NVFP4
                                       ? gate_scale
                                       : 0.62f + static_cast<float>((expert * 5) % 11) * 0.09f;
            const float down_scale = 0.71f + static_cast<float>((expert * 3) % 17) * 0.05f;
            gate_up_scale_host_[static_cast<std::size_t>(expert) * 2]     = gate_scale;
            gate_up_scale_host_[static_cast<std::size_t>(expert) * 2 + 1] = up_scale;
            down_scale_host_[static_cast<std::size_t>(expert)]            = down_scale;
            auto gate_up =
                profile.routed_gate_up == QType::NVFP4
                    // The runner reads an expert's rows as [up; gate]; the oracle keeps the
                    // [gate; up] view of the same numbers, so the decoded weights are swapped
                    // back after packing.
                    ? swap_halves_of_dequant(pack_nvfp4(swap_halves(gate_up_source, kHidden),
                                                        kExpertGateRows, kHidden,
                                                        {up_scale, gate_scale}),
                                             kHidden)
                    : quantized_weight::pack_row_split_lowbit(gate_up_source, kExpertGateRows,
                                                              kHidden, profile.routed_gate_up);
            auto down = profile.routed_down == QType::NVFP4
                            ? pack_nvfp4(down_source, kHidden, kIntermediate, {down_scale})
                            : quantized_weight::pack_row_split_lowbit(
                                  down_source, kHidden, kIntermediate, profile.routed_down);
            routed_gate_.copy_rows(gate_up, expert * kExpertGateRows);
            routed_down_.copy_rows(down, expert * kHidden);
            experts_.push_back({expert, std::move(gate_up), std::move(down)});
        }

        shared_gate_host_ = quantized_weight::pack_w8g32_row_split(
            make_gate_up(kSharedGateRows, kHidden, 0x512U, 0.93f), kSharedGateRows, kHidden);
        shared_down_host_ = quantized_weight::pack_w8g32_row_split(
            make_down(kHidden, kIntermediate, 0x731U, 0.87f), kHidden, kIntermediate);
        shared_gate_.copy_rows(shared_gate_host_, 0);
        shared_down_device_.copy_rows(shared_down_host_, 0);
        gate_up_scale_device_ = to_device(gate_up_scale_host_);
        down_scale_device_    = to_device(down_scale_host_);
        if (profile.routed_gate_up == QType::NVFP4) { calibrate_activation_scales(); }

        for (int pattern = 0; pattern < static_cast<int>(kRoutePatterns.size()); ++pattern) {
            references_.push_back(sparse_moe_oracle(inputs_[pattern], residuals_[pattern], router_,
                                                    experts_, shared_gate_host_, shared_down_host_,
                                                    kRoutePatterns[pattern]));
            // The NVFP4 profile computes two different functions: our own kernels keep the
            // activations in BF16, and the vendored runner rounds them to E2M1. Which one serves
            // a round is decided by its width, so the suite carries both references and asserts
            // against the one the width selects.
            w4a4_references_.push_back(sparse_moe_oracle(
                inputs_[pattern], residuals_[pattern], router_, experts_, shared_gate_host_,
                shared_down_host_, kRoutePatterns[pattern], {}, activation_model()));
        }
    }

    int run(std::int32_t tokens, int first_pattern, bool graph_replay) {
        const std::string label = std::string(profile_.name) + " T=" + std::to_string(tokens);
        std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
        std::vector<float> residual(static_cast<std::size_t>(kHidden) * tokens);
        std::vector<double> reference(static_cast<std::size_t>(kHidden) * tokens);
        for (std::int32_t token = 0; token < tokens; ++token) {
            const int pattern = (first_pattern + token) % static_cast<int>(kRoutePatterns.size());
            std::copy(inputs_[pattern].begin(), inputs_[pattern].end(),
                      input.begin() + static_cast<std::size_t>(token) * kHidden);
            std::copy(residuals_[pattern].begin(), residuals_[pattern].end(),
                      residual.begin() + static_cast<std::size_t>(token) * kHidden);
            const std::vector<double>& source = reference_for(tokens, pattern);
            std::copy(source.begin(), source.end(),
                      reference.begin() + static_cast<std::size_t>(token) * kHidden);
        }

        const std::vector<std::uint16_t> input_bits    = bf16_bits(input);
        const std::vector<std::uint16_t> residual_bits = bf16_bits(residual);
        DeviceBuffer device_input                      = to_device(input_bits);
        DeviceBuffer residual_seed                     = to_device(residual_bits);
        GuardedBf16Output destination_storage(residual.size());
        cuda_check(cudaMemcpy(destination_storage.data(), residual_seed.p, residual_seed.bytes,
                              cudaMemcpyDeviceToDevice),
                   "seed SparseMoe destination");

        ops::SparseMoeWeights weights{
            dense_bf16_weight(device_router_.p, kExperts + 1, kHidden),
            routed_gate_.weight(),
            routed_down_.weight(),
            shared_gate_.weight(),
            shared_down_device_.weight(),
            kTopK,
        };
        attach_nvfp4_scales(weights);
        Tensor x(device_input.p, DType::BF16, {kHidden, tokens});
        Tensor destination(destination_storage.data(), DType::BF16, {kHidden, tokens});
        const std::size_t workspace_bytes = ops::sparse_moe_workspace_capacity_bytes(ops::kSparseMoeQwen36Geometry, 
            weights.routed_gate_up.qtype, weights.routed_down.qtype, tokens, tokens);
        WorkspaceArena workspace(workspace_bytes);

        if (graph_replay) {
            cudaStream_t stream  = nullptr;
            cudaGraph_t graph    = nullptr;
            cudaGraphExec_t exec = nullptr;
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "create SparseMoe graph stream");
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                       "begin SparseMoe graph capture");
            cuda_check(cudaMemcpyAsync(destination.data, residual_seed.p, destination.bytes(),
                                       cudaMemcpyDeviceToDevice, stream),
                       "capture SparseMoe residual seed");
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                            stream);
            cuda_check(cudaStreamEndCapture(stream, &graph), "end SparseMoe graph capture");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
                       "instantiate SparseMoe graph");
            cuda_check(cudaGraphLaunch(exec, stream), "launch SparseMoe graph");
            cuda_check(cudaGraphLaunch(exec, stream), "replay SparseMoe graph");
            cuda_check(cudaStreamSynchronize(stream), "synchronize SparseMoe graph");
            cuda_check(cudaGraphExecDestroy(exec), "destroy SparseMoe graph executable");
            cuda_check(cudaGraphDestroy(graph), "destroy SparseMoe graph");
            cuda_check(cudaStreamDestroy(stream), "destroy SparseMoe graph stream");
        } else {
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                            nullptr);
            cuda_synchronize();
        }

        int failures = compare_output(label, destination_storage.values(), reference, tolerance(tokens));
        failures += destination_storage.verify_guards(label);
        failures +=
            verify_exact((label + " input preservation").c_str(),
                         from_device<std::uint16_t>(device_input, input_bits.size()), input_bits);
        if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        return failures;
    }

    // Experts the slot table maps to -1 are "computed elsewhere": every kernel of the round
    // must leave their paths out, including the prefill reduce, which once summed a stale
    // grouped column for them (the Flash-Next host-split corruption, 2026-08-29).
    int run_host_bound(std::int32_t tokens, int first_pattern, std::span<const int> excluded) {
        const std::string label = std::string(profile_.name) + " T=" + std::to_string(tokens) +
                                  " host-bound experts";
        std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
        std::vector<float> residual(static_cast<std::size_t>(kHidden) * tokens);
        std::vector<double> reference(static_cast<std::size_t>(kHidden) * tokens);
        for (std::int32_t token = 0; token < tokens; ++token) {
            const int pattern = (first_pattern + token) % static_cast<int>(kRoutePatterns.size());
            std::copy(inputs_[pattern].begin(), inputs_[pattern].end(),
                      input.begin() + static_cast<std::size_t>(token) * kHidden);
            std::copy(residuals_[pattern].begin(), residuals_[pattern].end(),
                      residual.begin() + static_cast<std::size_t>(token) * kHidden);
            const std::vector<double> partial =
                sparse_moe_oracle(inputs_[pattern], residuals_[pattern], router_, experts_,
                                  shared_gate_host_, shared_down_host_, kRoutePatterns[pattern],
                                  excluded, runs_on_runner(tokens) ? activation_model()
                                                                   : Nvfp4ActivationModel{});
            std::copy(partial.begin(), partial.end(),
                      reference.begin() + static_cast<std::size_t>(token) * kHidden);
        }
        std::vector<std::int32_t> table(static_cast<std::size_t>(kExperts));
        std::iota(table.begin(), table.end(), 0);
        for (const int expert : excluded) { table[static_cast<std::size_t>(expert)] = -1; }
        DeviceBuffer device_table = to_device(table);

        const std::vector<std::uint16_t> input_bits    = bf16_bits(input);
        const std::vector<std::uint16_t> residual_bits = bf16_bits(residual);
        DeviceBuffer device_input                      = to_device(input_bits);
        DeviceBuffer residual_seed                     = to_device(residual_bits);
        GuardedBf16Output destination_storage(residual.size());
        cuda_check(cudaMemcpy(destination_storage.data(), residual_seed.p, residual_seed.bytes,
                              cudaMemcpyDeviceToDevice),
                   "seed SparseMoe destination");
        ops::SparseMoeWeights weights{
            dense_bf16_weight(device_router_.p, kExperts + 1, kHidden),
            routed_gate_.weight(),
            routed_down_.weight(),
            shared_gate_.weight(),
            shared_down_device_.weight(),
            kTopK,
        };
        weights.slot_of_expert = static_cast<const std::int32_t*>(device_table.p);
        attach_nvfp4_scales(weights);
        Tensor x(device_input.p, DType::BF16, {kHidden, tokens});
        Tensor destination(destination_storage.data(), DType::BF16, {kHidden, tokens});
        const std::size_t workspace_bytes = ops::sparse_moe_workspace_capacity_bytes(ops::kSparseMoeQwen36Geometry, 
            weights.routed_gate_up.qtype, weights.routed_down.qtype, tokens, tokens);
        WorkspaceArena workspace(workspace_bytes);
        // Twice: the second call runs over a workspace the first one left populated, which
        // is where a stale grouped column would come from.
        for (int pass = 0; pass < 2; ++pass) {
            cuda_check(cudaMemcpy(destination_storage.data(), residual_seed.p, residual_seed.bytes,
                                  cudaMemcpyDeviceToDevice),
                       "reseed SparseMoe destination");
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination,
                            workspace, nullptr);
            cuda_synchronize();
        }
        int failures = compare_output(label, destination_storage.values(), reference, tolerance(tokens));
        failures += destination_storage.verify_guards(label);
        return failures;
    }

    int verify_persistent_inputs() const {
        int failures = 0;
        failures += verify_exact((std::string(profile_.name) + " router preservation").c_str(),
                                 from_device<std::uint16_t>(device_router_, router_bits_.size()),
                                 router_bits_);
        for (const HostExpert& expert : experts_) {
            failures += routed_gate_.verify_rows(
                std::string(profile_.name) + " routed gate expert " + std::to_string(expert.id),
                expert.gate_up, expert.id * kExpertGateRows);
            failures += routed_down_.verify_rows(
                std::string(profile_.name) + " routed down expert " + std::to_string(expert.id),
                expert.down, expert.id * kHidden);
        }
        failures += shared_gate_.verify_rows(std::string(profile_.name) + " shared gate",
                                             shared_gate_host_, 0);
        failures += shared_down_device_.verify_rows(std::string(profile_.name) + " shared down",
                                                    shared_down_host_, 0);
        return failures;
    }

    /// The op requires the second-level arrays for NVFP4 and rejects them for anything else,
    /// so attach exactly where the profile calls for it.
    void attach_nvfp4_scales(ops::SparseMoeWeights& weights) const {
        if (profile_.routed_gate_up == QType::NVFP4) {
            weights.routed_gate_up_scale = static_cast<const float*>(gate_up_scale_device_.p);
            weights.routed_gate_up_act_scale =
                static_cast<const float*>(gate_up_act_device_.p);
            weights.routed_gate_up_alpha = static_cast<const float*>(gate_up_alpha_device_.p);
        }
        if (profile_.routed_down == QType::NVFP4) {
            weights.routed_down_scale     = static_cast<const float*>(down_scale_device_.p);
            weights.routed_down_act_scale = static_cast<const float*>(down_act_device_.p);
            weights.routed_down_alpha     = static_cast<const float*>(down_alpha_device_.p);
        }
    }

    /// Empty for every A16 codec; for the NVFP4 profile it hands the oracle the same per-expert
    /// activation scales the runner receives.
    /// True when a round of this width is served by the vendored runner rather than our own
    /// kernels — the same crossover `ops::sparse_moe` applies.
    [[nodiscard]] bool runs_on_runner(std::int32_t tokens) const {
        return profile_.routed_gate_up == QType::NVFP4 &&
               tokens >= ops::kSparseMoeTrtllmMinTokens;
    }

    [[nodiscard]] Nvfp4ActivationModel activation_model() const {
        if (profile_.routed_gate_up != QType::NVFP4) { return {}; }
        return {&gate_up_act_host_, &down_act_host_};
    }

    [[nodiscard]] const ReductionCriterion& tolerance(std::int32_t tokens) const {
        return runs_on_runner(tokens) ? kSparseMoeA4Tolerance : kSparseMoeA16Tolerance;
    }

    [[nodiscard]] const std::vector<double>& reference_for(std::int32_t tokens,
                                                           int pattern) const {
        return runs_on_runner(tokens) ? w4a4_references_[static_cast<std::size_t>(pattern)]
                                      : references_[static_cast<std::size_t>(pattern)];
    }

    /// The activation global scales the checkpoint would carry: `6 * 448 / amax` of what each
    /// projection sees, per expert. `amax` is measured here the way a calibration pass measures
    /// it — over this fixture's own inputs for gate/up, and over the SwiGLU output the routed
    /// experts produce from them for down — so the E4M3 block scales the runner derives land in
    /// that format's normal range, which is the only thing the choice governs.
    void calibrate_activation_scales() {
        constexpr float kFp4Max = 6.0f;
        constexpr float kFp8Max = 448.0f;
        double input_amax       = 0.0;
        for (const std::vector<float>& input : inputs_) {
            for (float value : input) {
                input_amax = std::max(input_amax, static_cast<double>(std::fabs(value)));
            }
        }
        double activation_amax = 0.0;
        for (const std::vector<float>& input : inputs_) {
            const std::vector<double> x(input.begin(), input.end());
            for (const HostExpert& expert : experts_) {
                std::vector<double> row_max(kIntermediate, 0.0);
                sinfer::test::parallel_rows(kIntermediate, [&](std::int32_t row) {
                    const double gate = dot_fp64(expert.gate_up.dequant, row, kHidden, x);
                    const double up = dot_fp64(expert.gate_up.dequant, kIntermediate + row,
                                               kHidden, x);
                    row_max[static_cast<std::size_t>(row)] =
                        std::fabs((gate / (1.0 + std::exp(-gate))) * up);
                });
                for (double value : row_max) { activation_amax = std::max(activation_amax, value); }
            }
        }
        gate_up_act_host_.assign(static_cast<std::size_t>(kExperts), 1.0f);
        gate_up_alpha_host_.assign(static_cast<std::size_t>(kExperts), 1.0f);
        down_act_host_.assign(static_cast<std::size_t>(kExperts), 1.0f);
        down_alpha_host_.assign(static_cast<std::size_t>(kExperts), 1.0f);
        for (const HostExpert& expert : experts_) {
            const auto slot = static_cast<std::size_t>(expert.id);
            // A per-expert spread, as a real calibration produces: the runner indexes these by
            // expert, and a wrong index would otherwise be invisible.
            const float jitter = 0.85f + static_cast<float>((expert.id * 5) % 7) * 0.05f;
            const float gate_up_act =
                jitter * static_cast<float>(kFp8Max * kFp4Max / std::max(input_amax, 1e-6));
            const float down_act =
                jitter * static_cast<float>(kFp8Max * kFp4Max / std::max(activation_amax, 1e-6));
            gate_up_act_host_[slot] = gate_up_act;
            down_act_host_[slot]    = down_act;
            // The weight's global scale divides in the checkpoint, so its reciprocal is what the
            // second-level array already holds; alpha undoes both global scales at once.
            gate_up_alpha_host_[slot] = gate_up_scale_host_[slot * 2] / gate_up_act;
            down_alpha_host_[slot]    = down_scale_host_[slot] / down_act;
        }
        gate_up_act_device_   = to_device(gate_up_act_host_);
        gate_up_alpha_device_ = to_device(gate_up_alpha_host_);
        down_act_device_      = to_device(down_act_host_);
        down_alpha_device_    = to_device(down_alpha_host_);
    }

private:
    const CodecProfile& profile_;
    std::vector<float> router_;
    std::vector<std::uint16_t> router_bits_;
    DeviceBuffer device_router_;
    DeviceRowSplit routed_gate_;
    DeviceRowSplit routed_down_;
    DeviceRowSplit shared_gate_;
    DeviceRowSplit shared_down_device_;
    // NVFP4's second level: [experts][2] gate/up and [experts] down, uploaded once. Unit for
    // every other codec, where the arrays stay off the weights entirely.
    std::vector<float> gate_up_scale_host_;
    std::vector<float> down_scale_host_;
    DeviceBuffer gate_up_scale_device_;
    DeviceBuffer down_scale_device_;
    // The W4A4 runner's per-expert activation scale and epilogue alpha, one pair per projection.
    std::vector<float> gate_up_act_host_;
    std::vector<float> gate_up_alpha_host_;
    std::vector<float> down_act_host_;
    std::vector<float> down_alpha_host_;
    DeviceBuffer gate_up_act_device_;
    DeviceBuffer gate_up_alpha_device_;
    DeviceBuffer down_act_device_;
    DeviceBuffer down_alpha_device_;
    std::vector<std::vector<float>> inputs_;
    std::vector<std::vector<float>> residuals_;
    std::vector<HostExpert> experts_;
    quantized_weight::PackedWeight shared_gate_host_;
    quantized_weight::PackedWeight shared_down_host_;
    std::vector<std::vector<double>> references_;
    // The same rounds with the activations rounded to E2M1, for the widths the runner serves.
    std::vector<std::vector<double>> w4a4_references_;
};

int run_profile(const CodecProfile& profile) {
    SparseMoeFixture fixture(profile);
    int failures        = 0;
    std::size_t witness = 0;
    for (std::size_t index = 0; index < profile.token_cases.size(); ++index) {
        const std::int32_t tokens = profile.token_cases[index];
        witness =
            std::max(witness, ops::sparse_moe_workspace_capacity_bytes(ops::kSparseMoeQwen36Geometry, 
                                  profile.routed_gate_up, profile.routed_down, tokens, tokens));
        // Decode starts with the exact top-8 boundary tie; multi-token cases cycle the tie,
        // high/low expert ids, and a different ordering of the same experts.
        failures +=
            fixture.run(tokens, index == 0 ? 1 : 0, profile.verify_graph_replay && index == 1);
    }
    const std::size_t interval = ops::sparse_moe_workspace_capacity_bytes(ops::kSparseMoeQwen36Geometry, 
        profile.routed_gate_up, profile.routed_down, 1, profile.token_cases.back());
    if (interval != witness) {
        std::cerr << profile.name << ": interval workspace capacity missed a route witness\n";
        ++failures;
    }
    if (profile.routed_gate_up == QType::W8G32_F16S) {
        constexpr std::array<int, 2> kHostBound{{17, 191}};
        for (const std::int32_t tokens : {std::int32_t{20}, std::int32_t{768}}) {
            failures += fixture.run_host_bound(tokens, 0, kHostBound);
        }
    }
    failures += fixture.verify_persistent_inputs();
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    // These are public-behavior cases, not route assertions. They exercise decode (T=1), the
    // Small-T supported-domain edges, each profile's first prefill T, the wide-prefill boundary,
    // and one call crossing the 4096-token internal slice without observing any private plan.
    constexpr std::array<std::int32_t, 6> kQ4Q5Tokens{{1, 2, 46, 47, 768, 4097}};
    constexpr std::array<std::int32_t, 5> kQ4Q6Tokens{{1, 2, 46, 47, 768}};
    constexpr std::array<std::int32_t, 5> kW8W8Tokens{{1, 2, 19, 20, 768}};
    // The NVFP4 routed profile runs on the vendored TRT-LLM runner at every width, so the cases
    // walk that runner's own tuning ladder: 1 and 2 are buckets of their own, 19/20 and 47 land
    // inside the power-of-two rungs, 139 and 768 inside the linear ones, and 4,097 crosses the
    // 4,096-row slice bound into a second call.
    constexpr std::array<std::int32_t, 8> kNvfp4Tokens{{1, 2, 19, 20, 47, 139, 768, 4097}};
    const std::array<CodecProfile, 4> profiles{{
        {"sparse_moe q4+q5 a16", QType::Q4G64_F16S, QType::Q5G64_F16S, kQ4Q5Tokens, true},
        {"sparse_moe q4+q6 a16", QType::Q4G64_F16S, QType::Q6G64_F16S, kQ4Q6Tokens, false},
        {"sparse_moe w8+w8 a16", QType::W8G32_F16S, QType::W8G32_F16S, kW8W8Tokens, false},
            // NVFP4 routed experts, served by the vendored TRT-LLM runner: e2m1 codes with an
            // e4m3 scale every 16 values in the dense BlockScaleK16M128x4 layout, an expert's
            // gate/up rows stored [up; gate], and the activations rounded to the same format by
            // the runner. Shapes line up — gate/up is 1,024 x 2,048 and down 2,048 x 512, so N
            // is a multiple of 128 and K of 64 for both.
        {"sparse_moe nvfp4 w4a4", QType::NVFP4, QType::NVFP4, kNvfp4Tokens, false},
    }};

    int failures = 0;
    for (const CodecProfile& profile : profiles) { failures += run_profile(profile); }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe correctness\n";
    return failures == 0 ? 0 : 1;
}
