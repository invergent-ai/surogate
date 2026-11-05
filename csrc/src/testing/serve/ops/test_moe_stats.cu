// Calibration statistics for expert pruning (api/ops/calibration.h) on synthetic Flash-Next-shaped
// mixtures (`--geometry 512|192|160|128`, default 128: the teacher's or a student's, each with W8
// routed experts -- the profile a draft head's experts are converted to -- and NVFP4 ones). Every
// round width class runs -- decode (1), small-T, the wide prefill families and, for NVFP4, the
// TRT-LLM runner from two columns -- and four things are checked:
//
//  1. the observing hook changes nothing: the served output is bit-identical with and without it;
//  2. the probe rounds decompose the served round: shared-only + sum_k alpha_k * y_k reproduces the
//     served output (y_k = the slot-k probe's output), which shows every route honours the
//     reweighted router weights and the zeroed shared scale;
//  3. the statistics the module dumps (counts, gate sums, norm sums, tokens) equal the ones built
//     here from the same rounds' captured ids, weights and probe outputs: counts exactly, sums to
//     FP64-reduction-order tolerance -- also through a CUDA graph replay;
//  4. (W8) the probe outputs equal an FP64 evaluation of each selected expert from the decoded
//     weights, and the routing the oracle's top-k: the statistics are of the experts themselves.
//
// `--dump DIR` also writes the W8 student case's inputs (x, router, the active experts' int8
// codes, all of whose groups have the scale 2^-9 exactly) and its statistics for the PyTorch
// reference (surogate/serve/tools/calibration/moe_stats_reference.py).

#include "ops/nvfp4_device.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "api/ops/calibration.h"
#include "api/ops/sparse_moe.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <unistd.h>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr int kActive   = 40; // experts with weights; the router never selects the others
constexpr int kPatterns = 16; // token patterns, each with its own clear top-k among the active
constexpr float kWeightStep = 1.0F / 512.0F; // every logical weight is an integer times 2^-9

std::uint32_t hash32(std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t salt) {
    std::uint32_t h = a * 0x9E3779B1U ^ (b + 0x7F4A7C15U) * 0x85EBCA77U ^ (c + salt) * 0xC2B2AE3DU;
    h ^= h >> 15;
    h *= 0x2C1B3C6DU;
    h ^= h >> 12;
    h *= 0x297A2D39U;
    h ^= h >> 15;
    return h;
}

/// The logical weight as an int8 code: hashed in [-120, 120], with one lane of every 32-column
/// group forced to +-127, so a W8G32 encoder's group scale is exactly 127 * 2^-9 / 127 = 2^-9 and
/// the decoded weight is the code times 2^-9 -- in this test and in the PyTorch reference alike.
std::int8_t weight_code(int expert, int row, int column, std::uint32_t salt) {
    if (column % 32 == static_cast<int>((static_cast<unsigned>(row) * 7U + static_cast<unsigned>(expert)) % 32U)) {
        return (row + expert) % 2 ? 127 : -127;
    }
    return static_cast<std::int8_t>(static_cast<int>(hash32(expert, row, column, salt) % 241U) - 120);
}

std::vector<float> logical_matrix(int expert, int rows, int columns, std::uint32_t salt) {
    std::vector<float> out(static_cast<std::size_t>(rows) * columns);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < columns; ++c) {
            out[static_cast<std::size_t>(r) * columns + c] = weight_code(expert, r, c, salt) * kWeightStep;
        }
    }
    return out;
}

/// The experts the router can select: spread over the id range, both ends included.
std::vector<int> active_experts(int experts) {
    std::vector<int> out;
    for (int i = 0; i < kActive; ++i) { out.push_back(i * (experts - 1) / (kActive - 1)); }
    return out;
}

/// Pattern p's ranked choice of `top_k` active experts (distinct), deterministic.
std::vector<int> pattern_choice(int pattern, int top_k) {
    std::vector<int> order(kActive);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return hash32(pattern, a, 3, 11) < hash32(pattern, b, 3, 11);
    });
    order.resize(static_cast<std::size_t>(top_k));
    return order; // indices into active_experts
}

struct Layer {
    ops::SparseMoeGeometry geometry;
    QType qtype;
    std::vector<int> active;                    // expert ids
    std::vector<float> router;                  // [router_rows][hidden], BF16-exact
    DeviceBuffer router_device;
    DeviceBuffer gate_codes, gate_scales, down_codes, down_scales;
    DeviceBuffer shared_gate_codes, shared_gate_scales, shared_down_codes, shared_down_scales;
    std::map<int, quantized_weight::PackedWeight> gate_up, down; // the oracle's [gate; up] view
    quantized_weight::PackedWeight shared_gate_up, shared_down;
    // NVFP4's second level and the runner's activation scales.
    std::vector<float> gate_up_scale, down_scale, gate_up_act, gate_up_alpha, down_act, down_alpha;
    DeviceBuffer gate_up_scale_d, down_scale_d, gate_up_act_d, gate_up_alpha_d, down_act_d, down_alpha_d;
    ops::SparseMoeWeights weights{};
};

// --- W8G32 row-split planes for a stack of experts ------------------------------------------

struct RowSplitPlanes {
    std::int32_t rows, columns, groups_per_row;
    std::size_t code_row_bytes, scale_row_bytes;
};

RowSplitPlanes w8_planes(std::int32_t rows, std::int32_t columns) {
    const std::int32_t groups = columns / 32;
    return {rows, columns, groups, static_cast<std::size_t>(groups) * 32, static_cast<std::size_t>(groups) * 2};
}

Weight w8_weight(const DeviceBuffer& codes, const DeviceBuffer& scales, std::int32_t rows, std::int32_t columns) {
    Weight w{};
    w.payload         = codes.p;
    w.payload_bytes   = codes.bytes + scales.bytes;
    w.qtype           = QType::W8G32_F16S;
    w.group_size      = 32;
    w.group           = 32;
    w.qdata           = codes.p;
    w.scales          = scales.p;
    w.n               = rows;
    w.k               = columns;
    w.layout          = QuantLayout::RowSplit;
    w.scale_dtype     = DType::FP16;
    w.ndim            = 2;
    w.shape[0] = w.padded_shape[0] = rows;
    w.shape[1] = w.padded_shape[1] = columns;
    return w;
}

void copy_w8_rows(const quantized_weight::PackedWeight& packed, DeviceBuffer& codes, DeviceBuffer& scales,
                  const RowSplitPlanes& planes, std::int32_t first_row) {
    const std::size_t rows = static_cast<std::size_t>(packed.weight.n);
    codes.copy_from_host(packed.payload.data(), rows * planes.code_row_bytes,
                         static_cast<std::size_t>(first_row) * planes.code_row_bytes);
    scales.copy_from_host(packed.payload.data() + packed.scale_plane_offset, rows * planes.scale_row_bytes,
                          static_cast<std::size_t>(first_row) * planes.scale_row_bytes);
}

// --- NVFP4 (as test_sparse_moe_body.inc packs it) -------------------------------------------

quantized_weight::PackedWeight pack_nvfp4(const std::vector<float>& source, std::int32_t n, std::int32_t k,
                                          const std::vector<float>& block_scale) {
    const std::int32_t rows_per_scale = n / static_cast<std::int32_t>(block_scale.size());
    static constexpr float kMagnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const std::int32_t groups_per_row = k / 16;
    const std::int32_t k_tiles        = k / 64;
    quantized_weight::PackedWeight packed;
    packed.code_plane_bytes   = static_cast<std::uint64_t>(n) * k / 2;
    packed.scale_plane_offset = (packed.code_plane_bytes + 255U) / 256U * 256U;
    packed.scale_plane_bytes  = static_cast<std::uint64_t>(n) * k / 16;
    packed.payload.assign(static_cast<std::size_t>(packed.scale_plane_offset + packed.scale_plane_bytes) + 4U, 0);
    packed.dequant.assign(source.size(), 0.0f);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t group = 0; group < groups_per_row; ++group) {
            const std::size_t base = static_cast<std::size_t>(row) * k + group * 16;
            const float second     = block_scale[static_cast<std::size_t>(row / rows_per_scale)];
            float amax             = 0.0f;
            for (int i = 0; i < 16; ++i) { amax = std::max(amax, std::fabs(source[base + i]) / second); }
            const __nv_fp8_e4m3 stored_scale(amax > 0.0f ? amax / 6.0f : 1.0f);
            const float scale   = static_cast<float>(stored_scale);
            const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
            for (int i = 0; i < 16; ++i) {
                const float value      = source[base + i] / second;
                const float normalised = std::fabs(value) * inverse;
                int best               = 0;
                float best_error       = std::fabs(normalised - kMagnitudes[0]);
                for (int candidate = 1; candidate < 8; ++candidate) {
                    const float error = std::fabs(normalised - kMagnitudes[candidate]);
                    if (error < best_error) { best_error = error; best = candidate; }
                }
                const bool negative = std::signbit(value);
                const auto code = static_cast<std::uint8_t>((negative ? 0x8u : 0x0u) | static_cast<unsigned>(best));
                const std::size_t byte = (static_cast<std::size_t>(row) * k + group * 16 + i) / 2;
                packed.payload[byte] = (i & 1) == 0
                    ? static_cast<std::uint8_t>((packed.payload[byte] & 0xF0u) | code)
                    : static_cast<std::uint8_t>((packed.payload[byte] & 0x0Fu) | (code << 4));
                packed.dequant[base + i] = (negative ? -1.0f : 1.0f) * kMagnitudes[best] * scale * second;
            }
            const std::int32_t inner = row % 128;
            const std::size_t offset = packed.scale_plane_offset +
                static_cast<std::size_t>((row / 128) * k_tiles + group / 4) * 512U +
                static_cast<std::size_t>(inner % 32) * 16U + static_cast<std::size_t>(inner / 32) * 4U +
                static_cast<std::size_t>(group % 4);
            packed.payload[offset] = stored_scale.__x;
        }
    }
    packed.weight.n = n;
    packed.weight.k = k;
    return packed;
}

Weight nvfp4_weight(const DeviceBuffer& codes, const DeviceBuffer& scales, std::int32_t rows, std::int32_t columns) {
    Weight w{};
    w.payload        = codes.p;
    w.payload_bytes  = codes.bytes + scales.bytes;
    w.qtype          = QType::NVFP4;
    w.group_size     = 16;
    w.group          = 16;
    w.qdata          = codes.p;
    w.scales         = scales.p;
    w.n              = rows;
    w.k              = columns;
    w.layout         = QuantLayout::BlockScaleK16M128x4;
    w.scale_dtype    = DType::FP8_E4M3FN;
    w.ndim           = 2;
    w.shape[0] = w.padded_shape[0] = rows;
    w.shape[1] = w.padded_shape[1] = columns;
    w.weight_scale_divisor = 1.0f;
    w.input_scale_divisor  = 1.0f;
    return w;
}

std::vector<float> swap_halves(const std::vector<float>& source) {
    const std::size_t half = source.size() / 2;
    std::vector<float> out(source.size());
    std::copy(source.begin() + static_cast<std::ptrdiff_t>(half), source.end(), out.begin());
    std::copy(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(half),
              out.begin() + static_cast<std::ptrdiff_t>(half));
    return out;
}

// --- The synthetic layer ----------------------------------------------------------------------

/// Pattern columns (one per pattern, at the end of the hidden width) carry the routing: pattern p's
/// k-th choice gets 4 - 0.25 k on the router row's marker column, every other active expert -1, and
/// the inactive ones -8 on column 0. The small hashed background shifts no rank.
std::vector<float> make_router(const ops::SparseMoeGeometry& g, const std::vector<int>& active) {
    std::vector<float> router(static_cast<std::size_t>(g.router_rows()) * g.hidden);
    for (int row = 0; row < g.router_rows(); ++row) {
        for (int c = 0; c < g.hidden; ++c) {
            router[static_cast<std::size_t>(row) * g.hidden + c] =
                static_cast<float>(static_cast<int>(hash32(row, c, 9, 21) % 9U) - 4) / 2048.0F;
        }
    }
    std::vector<bool> is_active(static_cast<std::size_t>(g.experts), false);
    for (const int e : active) { is_active[static_cast<std::size_t>(e)] = true; }
    for (int e = 0; e < g.experts; ++e) {
        if (!is_active[static_cast<std::size_t>(e)]) { router[static_cast<std::size_t>(e) * g.hidden] = -8.0F; }
    }
    for (int p = 0; p < kPatterns; ++p) {
        const int marker = g.hidden - kPatterns + p;
        const std::vector<int> choice = pattern_choice(p, g.experts_per_token);
        for (int a = 0; a < kActive; ++a) {
            router[static_cast<std::size_t>(active[static_cast<std::size_t>(a)]) * g.hidden + marker] = -1.0F;
        }
        for (int k = 0; k < g.experts_per_token; ++k) {
            router[static_cast<std::size_t>(active[static_cast<std::size_t>(choice[static_cast<std::size_t>(k)])]) * g.hidden + marker] =
                4.0F - 0.25F * static_cast<float>(k);
        }
    }
    router[static_cast<std::size_t>(g.experts) * g.hidden] += 0.375F; // the shared expert's gate row
    round_to_bf16(router);
    return router;
}

/// Token t of a run: pattern t % kPatterns marked, a hashed background of multiples of 2^-8.
std::vector<float> make_inputs(const ops::SparseMoeGeometry& g, int first_token, int tokens) {
    std::vector<float> x(static_cast<std::size_t>(g.hidden) * tokens);
    for (int t = 0; t < tokens; ++t) {
        const int token = first_token + t;
        float* column   = x.data() + static_cast<std::size_t>(t) * g.hidden;
        for (int c = 0; c < g.hidden - kPatterns; ++c) {
            column[c] = static_cast<float>(static_cast<int>(hash32(token, c, 5, 33) % 25U) - 12) / 256.0F;
        }
        for (int p = 0; p < kPatterns; ++p) { column[g.hidden - kPatterns + p] = p == token % kPatterns ? 1.0F : 0.0F; }
    }
    round_to_bf16(x);
    return x;
}

Layer make_layer(const ops::SparseMoeGeometry& g, QType qtype) {
    Layer layer;
    layer.geometry = g;
    layer.qtype    = qtype;
    layer.active   = active_experts(g.experts);
    layer.router   = make_router(g, layer.active);
    std::vector<std::uint16_t> router_bits(layer.router.size());
    for (std::size_t i = 0; i < router_bits.size(); ++i) { router_bits[i] = f32_to_bf16(layer.router[i]); }
    layer.router_device = to_device(router_bits);
    const std::int32_t gate_rows = 2 * g.intermediate;
    if (qtype == QType::W8G32_F16S) {
        const RowSplitPlanes gp = w8_planes(g.routed_gate_rows(), g.hidden);
        const RowSplitPlanes dp = w8_planes(g.routed_down_rows(), g.intermediate);
        layer.gate_codes  = DeviceBuffer(static_cast<std::size_t>(gp.rows) * gp.code_row_bytes);
        layer.gate_scales = DeviceBuffer(static_cast<std::size_t>(gp.rows) * gp.scale_row_bytes);
        layer.down_codes  = DeviceBuffer(static_cast<std::size_t>(dp.rows) * dp.code_row_bytes);
        layer.down_scales = DeviceBuffer(static_cast<std::size_t>(dp.rows) * dp.scale_row_bytes);
        for (DeviceBuffer* b : {&layer.gate_codes, &layer.gate_scales, &layer.down_codes, &layer.down_scales}) { b->fill(0); }
        for (const int e : layer.active) {
            auto gu = quantized_weight::pack_w8g32_row_split(logical_matrix(e, gate_rows, g.hidden, 101), gate_rows, g.hidden);
            auto dn = quantized_weight::pack_w8g32_row_split(logical_matrix(e, g.hidden, g.intermediate, 202), g.hidden, g.intermediate);
            copy_w8_rows(gu, layer.gate_codes, layer.gate_scales, gp, e * gate_rows);
            copy_w8_rows(dn, layer.down_codes, layer.down_scales, dp, e * g.hidden);
            layer.gate_up.emplace(e, std::move(gu));
            layer.down.emplace(e, std::move(dn));
        }
        layer.weights.routed_gate_up = w8_weight(layer.gate_codes, layer.gate_scales, gp.rows, gp.columns);
        layer.weights.routed_down    = w8_weight(layer.down_codes, layer.down_scales, dp.rows, dp.columns);
    } else {
        const std::size_t gate_code_bytes = static_cast<std::size_t>(g.routed_gate_rows()) * g.hidden / 2;
        const std::size_t down_code_bytes = static_cast<std::size_t>(g.routed_down_rows()) * g.intermediate / 2;
        layer.gate_codes  = DeviceBuffer(gate_code_bytes);
        layer.gate_scales = DeviceBuffer(static_cast<std::size_t>(g.routed_gate_rows()) * g.hidden / 16);
        layer.down_codes  = DeviceBuffer(down_code_bytes);
        layer.down_scales = DeviceBuffer(static_cast<std::size_t>(g.routed_down_rows()) * g.intermediate / 16);
        for (DeviceBuffer* b : {&layer.gate_codes, &layer.gate_scales, &layer.down_codes, &layer.down_scales}) { b->fill(0); }
        layer.gate_up_scale.assign(static_cast<std::size_t>(g.experts) * 2, 1.0F);
        layer.down_scale.assign(static_cast<std::size_t>(g.experts), 1.0F);
        for (const int e : layer.active) {
            const float second = 0.6F + 0.05F * static_cast<float>(e % 7);
            const float down_second = 0.7F + 0.04F * static_cast<float>(e % 5);
            layer.gate_up_scale[static_cast<std::size_t>(e) * 2]     = second;
            layer.gate_up_scale[static_cast<std::size_t>(e) * 2 + 1] = second;
            layer.down_scale[static_cast<std::size_t>(e)]            = down_second;
            // The runner reads an expert's rows as [up; gate]; the oracle keeps [gate; up].
            auto gu = pack_nvfp4(swap_halves(logical_matrix(e, gate_rows, g.hidden, 101)), gate_rows, g.hidden,
                                 {second, second});
            gu.dequant = swap_halves(gu.dequant);
            auto dn = pack_nvfp4(logical_matrix(e, g.hidden, g.intermediate, 202), g.hidden, g.intermediate, {down_second});
            layer.gate_codes.copy_from_host(gu.payload.data(), static_cast<std::size_t>(gate_rows) * g.hidden / 2,
                                            static_cast<std::size_t>(e) * gate_rows * g.hidden / 2);
            layer.gate_scales.copy_from_host(gu.payload.data() + gu.scale_plane_offset,
                                             static_cast<std::size_t>(gate_rows) * g.hidden / 16,
                                             static_cast<std::size_t>(e) * gate_rows * g.hidden / 16);
            layer.down_codes.copy_from_host(dn.payload.data(), static_cast<std::size_t>(g.hidden) * g.intermediate / 2,
                                            static_cast<std::size_t>(e) * g.hidden * g.intermediate / 2);
            layer.down_scales.copy_from_host(dn.payload.data() + dn.scale_plane_offset,
                                             static_cast<std::size_t>(g.hidden) * g.intermediate / 16,
                                             static_cast<std::size_t>(e) * g.hidden * g.intermediate / 16);
            layer.gate_up.emplace(e, std::move(gu));
            layer.down.emplace(e, std::move(dn));
        }
        // The runner's activation scales: 6 * 448 / amax of what each projection sees, as a
        // calibration would set them (the inputs here stay below 1; the SwiGLU output below ~2).
        layer.gate_up_act.assign(static_cast<std::size_t>(g.experts), 6.0F * 448.0F / 1.0F);
        layer.down_act.assign(static_cast<std::size_t>(g.experts), 6.0F * 448.0F / 2.0F);
        layer.gate_up_alpha.resize(static_cast<std::size_t>(g.experts));
        layer.down_alpha.resize(static_cast<std::size_t>(g.experts));
        for (int e = 0; e < g.experts; ++e) {
            layer.gate_up_alpha[static_cast<std::size_t>(e)] = layer.gate_up_scale[static_cast<std::size_t>(e) * 2] /
                                                               layer.gate_up_act[static_cast<std::size_t>(e)];
            layer.down_alpha[static_cast<std::size_t>(e)] = layer.down_scale[static_cast<std::size_t>(e)] /
                                                            layer.down_act[static_cast<std::size_t>(e)];
        }
        layer.gate_up_scale_d = to_device(layer.gate_up_scale);
        layer.down_scale_d    = to_device(layer.down_scale);
        layer.gate_up_act_d   = to_device(layer.gate_up_act);
        layer.gate_up_alpha_d = to_device(layer.gate_up_alpha);
        layer.down_act_d      = to_device(layer.down_act);
        layer.down_alpha_d    = to_device(layer.down_alpha);
        layer.weights.routed_gate_up = nvfp4_weight(layer.gate_codes, layer.gate_scales, g.routed_gate_rows(), g.hidden);
        layer.weights.routed_down    = nvfp4_weight(layer.down_codes, layer.down_scales, g.routed_down_rows(), g.intermediate);
        layer.weights.routed_gate_up_scale     = static_cast<const float*>(layer.gate_up_scale_d.p);
        layer.weights.routed_down_scale        = static_cast<const float*>(layer.down_scale_d.p);
        layer.weights.routed_gate_up_act_scale = static_cast<const float*>(layer.gate_up_act_d.p);
        layer.weights.routed_gate_up_alpha     = static_cast<const float*>(layer.gate_up_alpha_d.p);
        layer.weights.routed_down_act_scale    = static_cast<const float*>(layer.down_act_d.p);
        layer.weights.routed_down_alpha        = static_cast<const float*>(layer.down_alpha_d.p);
    }
    // The shared expert: W8, as every converted Flash-Next artifact stores it.
    layer.shared_gate_up = quantized_weight::pack_w8g32_row_split(
        logical_matrix(9999, 2 * g.shared_intermediate, g.hidden, 303), 2 * g.shared_intermediate, g.hidden);
    layer.shared_down = quantized_weight::pack_w8g32_row_split(
        logical_matrix(9999, g.hidden, g.shared_intermediate, 404), g.hidden, g.shared_intermediate);
    const RowSplitPlanes sgp = w8_planes(2 * g.shared_intermediate, g.hidden);
    const RowSplitPlanes sdp = w8_planes(g.hidden, g.shared_intermediate);
    layer.shared_gate_codes  = DeviceBuffer(static_cast<std::size_t>(sgp.rows) * sgp.code_row_bytes);
    layer.shared_gate_scales = DeviceBuffer(static_cast<std::size_t>(sgp.rows) * sgp.scale_row_bytes);
    layer.shared_down_codes  = DeviceBuffer(static_cast<std::size_t>(sdp.rows) * sdp.code_row_bytes);
    layer.shared_down_scales = DeviceBuffer(static_cast<std::size_t>(sdp.rows) * sdp.scale_row_bytes);
    copy_w8_rows(layer.shared_gate_up, layer.shared_gate_codes, layer.shared_gate_scales, sgp, 0);
    copy_w8_rows(layer.shared_down, layer.shared_down_codes, layer.shared_down_scales, sdp, 0);
    layer.weights.router_shared_gate = [&] {
        Weight w{};
        w.payload = w.qdata = layer.router_device.p;
        w.payload_bytes = layer.router_device.bytes;
        w.qtype = QType::BF16_CTRL;
        w.n = g.router_rows();
        w.k = g.hidden;
        w.layout = QuantLayout::Contiguous;
        w.ndim = 2;
        w.shape[0] = w.padded_shape[0] = g.router_rows();
        w.shape[1] = w.padded_shape[1] = g.hidden;
        return w;
    }();
    layer.weights.routed_scale      = 1.0F;
    layer.weights.shared_gated      = true;
    layer.weights.shared_gate_up    = w8_weight(layer.shared_gate_codes, layer.shared_gate_scales, sgp.rows, sgp.columns);
    layer.weights.shared_down       = w8_weight(layer.shared_down_codes, layer.shared_down_scales, sdp.rows, sdp.columns);
    layer.weights.experts_per_token = g.experts_per_token;
    return layer;
}

// --- Hooks of this test's own: what the module's hooks see, captured independently ----------

struct Capture {
    int top_k = 0;
    std::vector<int> ids;      // [tokens][top_k] of the whole call
    std::vector<float> alpha;  // [tokens][top_k]
    int calls = 0;
};

void capture_round(void* context, const Tensor& ids, float* alpha, float*, std::int32_t first,
                   std::int32_t tokens, cudaStream_t stream) {
    auto& c = *static_cast<Capture*>(context);
    const std::size_t n = static_cast<std::size_t>(tokens) * c.top_k;
    std::vector<int> i(n);
    std::vector<float> a(n);
    cuda_check(cudaMemcpyAsync(i.data(), ids.data, n * sizeof(int), cudaMemcpyDeviceToHost, stream), "capture ids");
    cuda_check(cudaMemcpyAsync(a.data(), alpha, n * sizeof(float), cudaMemcpyDeviceToHost, stream), "capture alpha");
    cuda_check(cudaStreamSynchronize(stream), "capture sync");
    std::copy(i.begin(), i.end(), c.ids.begin() + static_cast<std::ptrdiff_t>(first) * c.top_k);
    std::copy(a.begin(), a.end(), c.alpha.begin() + static_cast<std::ptrdiff_t>(first) * c.top_k);
    ++c.calls;
}

struct Rewrite {
    int top_k = 0;
    int slot  = -1; // -1: every routed weight zero (the shared expert alone)
};

__global__ void rewrite_kernel(float* alpha, float* shared_scale, int tokens, int top_k, int slot) {
    const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= tokens) { return; }
    for (int k = 0; k < top_k; ++k) { alpha[t * top_k + k] = k == slot ? 1.0F : 0.0F; }
    if (slot >= 0 && shared_scale != nullptr) { shared_scale[t] = 0.0F; }
}

void rewrite_round(void* context, const Tensor&, float* alpha, float* shared_scale, std::int32_t,
                   std::int32_t tokens, cudaStream_t stream) {
    const auto& r = *static_cast<const Rewrite*>(context);
    rewrite_kernel<<<(tokens + 127) / 128, 128, 0, stream>>>(alpha, shared_scale, tokens, r.top_k, r.slot);
}

std::vector<double> run_moe(const Layer& layer, const DeviceBuffer& x, int tokens,
                            const ops::SparseMoeRoundHook& hook, ops::SparseMoeRouting routing) {
    const auto& g = layer.geometry;
    DeviceBuffer out(static_cast<std::size_t>(g.hidden) * tokens * 2);
    out.fill(0);
    Tensor input(x.p, DType::BF16, {g.hidden, tokens});
    Tensor destination(out.p, DType::BF16, {g.hidden, tokens});
    WorkspaceArena workspace(ops::sparse_moe_workspace_capacity_bytes(
        g, layer.weights.routed_gate_up.qtype, layer.weights.routed_down.qtype, tokens, tokens, routing));
    ops::sparse_moe(input, layer.weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                    nullptr, hook, routing);
    cuda_synchronize();
    return from_device_bf16(out, static_cast<std::size_t>(g.hidden) * tokens);
}

double silu(double v) { return v / (1.0 + std::exp(-v)); }

/// The expert's own output for one token, in FP64 from its decoded weights (the A16 formula).
std::vector<double> expert_fp64(const Layer& layer, int expert, const float* x) {
    const auto& g = layer.geometry;
    const auto& gu = layer.gate_up.at(expert).dequant;
    const auto& dn = layer.down.at(expert).dequant;
    std::vector<double> act(static_cast<std::size_t>(g.intermediate));
    for (int r = 0; r < g.intermediate; ++r) {
        double gate = 0, up = 0;
        for (int c = 0; c < g.hidden; ++c) {
            gate += static_cast<double>(gu[static_cast<std::size_t>(r) * g.hidden + c]) * x[c];
            up += static_cast<double>(gu[static_cast<std::size_t>(g.intermediate + r) * g.hidden + c]) * x[c];
        }
        act[static_cast<std::size_t>(r)] = silu(gate) * up;
    }
    std::vector<double> y(static_cast<std::size_t>(g.hidden));
    for (int r = 0; r < g.hidden; ++r) {
        double sum = 0;
        for (int c = 0; c < g.intermediate; ++c) { sum += static_cast<double>(dn[static_cast<std::size_t>(r) * g.intermediate + c]) * act[static_cast<std::size_t>(c)]; }
        y[static_cast<std::size_t>(r)] = sum;
    }
    return y;
}

double relative_l2(const std::vector<double>& got, const std::vector<double>& want) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < want.size(); ++i) { num += (got[i] - want[i]) * (got[i] - want[i]); den += want[i] * want[i]; }
    return std::sqrt(num / std::max(den, 1e-30));
}

template <class T>
std::vector<T> read_npy(const std::filesystem::path& path, std::size_t count) {
    std::ifstream in(path, std::ios::binary);
    char head[10];
    in.read(head, 10);
    const std::size_t header = static_cast<unsigned char>(head[8]) | (static_cast<std::size_t>(static_cast<unsigned char>(head[9])) << 8);
    in.seekg(static_cast<std::streamoff>(10 + header));
    std::vector<T> out(count);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) { throw std::runtime_error("short npy " + path.string()); }
    return out;
}

struct Expected {
    std::vector<unsigned long long> counts;
    std::vector<double> gate, gate_norm, norm;
    unsigned long long tokens = 0;
};

/// One case: every round width once, eagerly; then the module's dump against what the rounds
/// showed. Returns failures.
int run_case(const char* label, const ops::SparseMoeGeometry& g, QType qtype, int layer_row,
             const std::vector<int>& widths, bool oracle, const std::string& dump_dir) {
    int failures = 0;
    const Layer layer = make_layer(g, qtype);
    Expected want;
    want.counts.assign(static_cast<std::size_t>(g.experts), 0);
    want.gate.assign(static_cast<std::size_t>(g.experts), 0.0);
    want.gate_norm.assign(static_cast<std::size_t>(g.experts), 0.0);
    want.norm.assign(static_cast<std::size_t>(g.experts), 0.0);
    ops::moe_stats_reset();
    int first_token = 0;
    std::ofstream x_dump;
    if (!dump_dir.empty()) { x_dump.open(std::filesystem::path(dump_dir) / "x.bf16", std::ios::binary); }
    for (const int tokens : widths) {
        // Width-invariant routing (what prefill asks for) sends an admitted W8 pair to the wide
        // route at every width, so the narrow rounds take ByWidth here and reach decode and small-T.
        const auto routing = tokens >= 20 && qtype == QType::W8G32_F16S ? ops::SparseMoeRouting::WidthInvariant
                                                                        : ops::SparseMoeRouting::ByWidth;
        const std::vector<float> x = make_inputs(g, first_token, tokens);
        std::vector<std::uint16_t> bits(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) { bits[i] = f32_to_bf16(x[i]); }
        if (x_dump) { x_dump.write(reinterpret_cast<const char*>(bits.data()), static_cast<std::streamsize>(bits.size() * 2)); }
        DeviceBuffer x_device = to_device(bits);
        const std::string name = std::string(label) + " T=" + std::to_string(tokens);

        // 1. The served round, plain and observed (the module's hook): bit for bit the same.
        const std::vector<double> plain = run_moe(layer, x_device, tokens, ops::SparseMoeRoundHook{}, routing);
        ops::MoeStatsObserver observer;
        const ops::SparseMoeRoundHook observe = ops::moe_stats_observer(layer_row, observer);
        if (observe.reweight == nullptr) { std::cerr << name << ": no observer hook\n"; return failures + 1; }
        const std::vector<double> served = run_moe(layer, x_device, tokens, observe, routing);
        if (served != plain) { std::cerr << name << ": the observing hook changed the served output\n"; ++failures; }
        // The module's probe rounds for this call, on a workspace of the layer call's size.
        {
            Tensor input(x_device.p, DType::BF16, {g.hidden, tokens});
            WorkspaceArena moe_workspace(ops::sparse_moe_workspace_capacity_bytes(
                g, layer.weights.routed_gate_up.qtype, layer.weights.routed_down.qtype, tokens, tokens, routing));
            WorkspaceArena probe_workspace(ops::moe_stats_probe_workspace_bytes(g.hidden, tokens) + (1U << 20));
            ops::moe_stats_probe(layer_row, input, layer.weights, routing, moe_workspace, probe_workspace, nullptr);
            cuda_synchronize();
        }

        // 2. This test's own capture of the routing, and its own probe rounds.
        Capture capture{g.experts_per_token, std::vector<int>(static_cast<std::size_t>(tokens) * g.experts_per_token, -1),
                        std::vector<float>(static_cast<std::size_t>(tokens) * g.experts_per_token, 0.0F)};
        ops::SparseMoeRoundHook capture_hook{};
        capture_hook.reweight = &capture_round;
        capture_hook.reweight_context = &capture;
        (void)run_moe(layer, x_device, tokens, capture_hook, routing);
        Rewrite rewrite{g.experts_per_token, -1};
        ops::SparseMoeRoundHook rewrite_hook{};
        rewrite_hook.reweight = &rewrite_round;
        rewrite_hook.reweight_context = &rewrite;
        const std::vector<double> shared_only = run_moe(layer, x_device, tokens, rewrite_hook, routing);
        std::vector<std::vector<double>> y(static_cast<std::size_t>(g.experts_per_token));
        for (int k = 0; k < g.experts_per_token; ++k) {
            rewrite.slot = k;
            y[static_cast<std::size_t>(k)] = run_moe(layer, x_device, tokens, rewrite_hook, routing);
        }
        std::vector<double> rebuilt(shared_only);
        for (int t = 0; t < tokens; ++t) {
            for (int k = 0; k < g.experts_per_token; ++k) {
                const double a = capture.alpha[static_cast<std::size_t>(t) * g.experts_per_token + k];
                for (int r = 0; r < g.hidden; ++r) {
                    rebuilt[static_cast<std::size_t>(t) * g.hidden + r] += a * y[static_cast<std::size_t>(k)][static_cast<std::size_t>(t) * g.hidden + r];
                }
            }
        }
        const double decomposition = relative_l2(rebuilt, served);
        if (!(decomposition < 1.5e-2)) {
            std::cerr << name << ": shared + sum alpha*y differs from the served output (rel L2 " << decomposition << ")\n";
            ++failures;
        }
        // 3. What the statistics must hold for this call.
        want.tokens += static_cast<unsigned long long>(tokens);
        double worst_oracle = 0.0;
        for (int t = 0; t < tokens; ++t) {
            const int pattern = (first_token + t) % kPatterns;
            const std::vector<int> choice = pattern_choice(pattern, g.experts_per_token);
            for (int k = 0; k < g.experts_per_token; ++k) {
                const std::size_t at = static_cast<std::size_t>(t) * g.experts_per_token + k;
                const int expert = capture.ids[at];
                const int intended = layer.active[static_cast<std::size_t>(choice[static_cast<std::size_t>(k)])];
                if (expert != intended) {
                    if (failures < 20) { std::cerr << name << ": token " << t << " slot " << k << " routed to " << expert << ", intended " << intended << "\n"; }
                    ++failures;
                    continue;
                }
                double sq = 0;
                for (int r = 0; r < g.hidden; ++r) { const double v = y[static_cast<std::size_t>(k)][static_cast<std::size_t>(t) * g.hidden + r]; sq += v * v; }
                const double length = std::sqrt(sq);
                want.counts[static_cast<std::size_t>(expert)] += 1;
                want.gate[static_cast<std::size_t>(expert)] += capture.alpha[at];
                want.norm[static_cast<std::size_t>(expert)] += length;
                want.gate_norm[static_cast<std::size_t>(expert)] += capture.alpha[at] * length;
                // 4. The probe output against the expert itself, for a sample of tokens.
                if (oracle && t < 3) {
                    const std::vector<double> ref = expert_fp64(layer, expert, x.data() + static_cast<std::size_t>(t) * g.hidden);
                    const std::vector<double> got(y[static_cast<std::size_t>(k)].begin() + static_cast<std::ptrdiff_t>(t) * g.hidden,
                                                  y[static_cast<std::size_t>(k)].begin() + static_cast<std::ptrdiff_t>(t + 1) * g.hidden);
                    worst_oracle = std::max(worst_oracle, relative_l2(got, ref));
                }
            }
        }
        if (oracle && !(worst_oracle < 1.5e-2)) {
            std::cerr << name << ": a probe output differs from its expert's FP64 evaluation (rel L2 " << worst_oracle << ")\n";
            ++failures;
        }
        std::printf("%s: decomposition rel L2 %.2e%s\n", name.c_str(), decomposition,
                    oracle ? (", oracle rel L2 " + std::to_string(worst_oracle)).c_str() : "");
        first_token += tokens;
    }
    // The module's dump against the expectation.
    const std::string dir = ops::moe_stats_dump(std::string(label), false);
    const std::size_t rows = static_cast<std::size_t>(layer_row) + 1;
    (void)rows;
    const std::filesystem::path path(dir);
    // The accumulators are [mixtures][experts]; this case observed row `layer_row` only.
    const auto mixtures = read_npy<unsigned long long>(path / "tokens.npy", 2);
    const std::size_t cells = static_cast<std::size_t>(2) * g.experts;
    const auto counts = read_npy<unsigned long long>(path / "counts.npy", cells);
    const auto gate = read_npy<double>(path / "gate_sum.npy", cells);
    const auto gate_norm = read_npy<double>(path / "gate_norm_sum.npy", cells);
    const auto norm = read_npy<double>(path / "norm_sum.npy", cells);
    if (mixtures[static_cast<std::size_t>(layer_row)] != want.tokens) {
        std::cerr << label << ": tokens " << mixtures[static_cast<std::size_t>(layer_row)] << " != " << want.tokens << "\n";
        ++failures;
    }
    double gate_err = 0, norm_err = 0, gate_norm_err = 0;
    for (int e = 0; e < g.experts; ++e) {
        const std::size_t at = static_cast<std::size_t>(layer_row) * g.experts + e;
        if (counts[at] != want.counts[static_cast<std::size_t>(e)]) {
            std::cerr << label << ": expert " << e << " count " << counts[at] << " != " << want.counts[static_cast<std::size_t>(e)] << "\n";
            ++failures;
        }
        const auto rel = [](double a, double b) { return std::fabs(a - b) / std::max(std::fabs(b), 1e-12); };
        if (want.counts[static_cast<std::size_t>(e)] == 0) {
            if (gate[at] != 0 || norm[at] != 0 || gate_norm[at] != 0) { std::cerr << label << ": unrouted expert " << e << " has sums\n"; ++failures; }
            continue;
        }
        gate_err = std::max(gate_err, rel(gate[at], want.gate[static_cast<std::size_t>(e)]));
        norm_err = std::max(norm_err, rel(norm[at], want.norm[static_cast<std::size_t>(e)]));
        gate_norm_err = std::max(gate_norm_err, rel(gate_norm[at], want.gate_norm[static_cast<std::size_t>(e)]));
    }
    if (!(gate_err < 1e-9) || !(norm_err < 1e-5) || !(gate_norm_err < 1e-5)) {
        std::cerr << label << ": dumped sums off (gate " << gate_err << ", norm " << norm_err << ", gate*norm " << gate_norm_err << ")\n";
        ++failures;
    }
    std::printf("%s: %llu tokens, counts exact, max rel err gate %.1e norm %.1e gate*norm %.1e\n", label,
                want.tokens, gate_err, norm_err, gate_norm_err);
    if (!dump_dir.empty()) {
        // The PyTorch reference's inputs: the router (BF16 bits), the active experts' codes.
        std::ofstream(std::filesystem::path(dump_dir) / "router.bf16", std::ios::binary).write(
            reinterpret_cast<const char*>([&] {
                static std::vector<std::uint16_t> bits;
                bits.resize(layer.router.size());
                for (std::size_t i = 0; i < bits.size(); ++i) { bits[i] = f32_to_bf16(layer.router[i]); }
                return bits.data();
            }()), static_cast<std::streamsize>(layer.router.size() * 2));
        std::ofstream codes(std::filesystem::path(dump_dir) / "experts.i8", std::ios::binary);
        for (const int e : layer.active) {
            for (int r = 0; r < 2 * g.intermediate; ++r) { for (int c = 0; c < g.hidden; ++c) { const auto v = weight_code(e, r, c, 101); codes.write(reinterpret_cast<const char*>(&v), 1); } }
            for (int r = 0; r < g.hidden; ++r) { for (int c = 0; c < g.intermediate; ++c) { const auto v = weight_code(e, r, c, 202); codes.write(reinterpret_cast<const char*>(&v), 1); } }
        }
        std::ofstream meta(std::filesystem::path(dump_dir) / "case.json");
        meta << "{\"hidden\": " << g.hidden << ", \"experts\": " << g.experts << ", \"top_k\": " << g.experts_per_token
             << ", \"intermediate\": " << g.intermediate << ", \"layer_row\": " << layer_row << ", \"weight_step\": "
             << kWeightStep << ", \"stats_dir\": \"" << dir << "\", \"active\": [";
        for (std::size_t i = 0; i < layer.active.size(); ++i) { meta << (i ? ", " : "") << layer.active[i]; }
        meta << "], \"widths\": [";
        for (std::size_t i = 0; i < widths.size(); ++i) { meta << (i ? ", " : "") << widths[i]; }
        meta << "], \"layout\": \"x.bf16: every round's columns [tokens][hidden] in order; router.bf16 [experts+1][hidden]; experts.i8: per active expert [2*intermediate][hidden] gate then up, then [hidden][intermediate] down; weight = code * weight_step\"}\n";
    }
    return failures;
}

/// The probe and the observer captured into a CUDA graph and replayed twice: twice the counts of
/// one eager call, as a decode graph replays them round after round.
int run_graph_case(const ops::SparseMoeGeometry& g, int layer_row) {
    const Layer layer = make_layer(g, QType::W8G32_F16S);
    const int tokens  = 1;
    const std::vector<float> x = make_inputs(g, 0, tokens);
    std::vector<std::uint16_t> bits(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { bits[i] = f32_to_bf16(x[i]); }
    DeviceBuffer x_device = to_device(bits);
    DeviceBuffer out(static_cast<std::size_t>(g.hidden) * tokens * 2);
    Tensor input(x_device.p, DType::BF16, {g.hidden, tokens});
    Tensor destination(out.p, DType::BF16, {g.hidden, tokens});
    const auto routing = ops::SparseMoeRouting::ByWidth;
    const std::size_t moe_bytes = ops::sparse_moe_workspace_capacity_bytes(g, QType::W8G32_F16S, QType::W8G32_F16S, 1, 1, routing);
    WorkspaceArena moe_workspace(moe_bytes);
    WorkspaceArena probe_workspace(ops::moe_stats_probe_workspace_bytes(g.hidden, tokens) + (1U << 20));
    ops::moe_stats_reset();
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ops::MoeStatsObserver observer;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    cuda_check(cudaMemsetAsync(out.p, 0, out.bytes, stream), "zero");
    ops::sparse_moe(input, layer.weights, ops::SparseMoeEpilogue::AddResidual, destination, moe_workspace, stream,
                    ops::moe_stats_observer(layer_row, observer), routing);
    ops::moe_stats_probe(layer_row, input, layer.weights, routing, moe_workspace, probe_workspace, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    cuda_check(cudaGraphLaunch(exec, stream), "launch");
    cuda_check(cudaGraphLaunch(exec, stream), "replay");
    cuda_check(cudaStreamSynchronize(stream), "sync");
    const std::string dir = ops::moe_stats_dump("graph", false);
    const auto counts = read_npy<unsigned long long>(std::filesystem::path(dir) / "counts.npy", static_cast<std::size_t>(2) * g.experts);
    const auto tokens_seen = read_npy<unsigned long long>(std::filesystem::path(dir) / "tokens.npy", 2);
    unsigned long long total = 0;
    for (int e = 0; e < g.experts; ++e) { total += counts[static_cast<std::size_t>(layer_row) * g.experts + e]; }
    cuda_check(cudaGraphExecDestroy(exec), "destroy exec");
    cuda_check(cudaGraphDestroy(graph), "destroy graph");
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
    const bool ok = total == static_cast<unsigned long long>(2 * g.experts_per_token) && tokens_seen[static_cast<std::size_t>(layer_row)] == 2;
    std::printf("graph replay: %llu routed pairs over %llu tokens after two replays (%s)\n", total,
                tokens_seen[static_cast<std::size_t>(layer_row)], ok ? "ok" : "WRONG");
    return ok ? 0 : 1;
}

// ---- residual samples (ops/calibration/residual_samples.cpp) --------------------------------

struct SampleRecord {
    std::uint64_t forward = 0;
    int boundary = 0, rows = 0, count = 0, flags = 0;
    long long base = 0;
    std::vector<int> columns, ids;
    std::vector<float> values; // [count][rows]
};

std::vector<SampleRecord> read_samples(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    std::vector<SampleRecord> out;
    char header[64];
    while (in.read(header, 64)) {
        SampleRecord r;
        if (std::memcmp(header, "RSAMPv1", 8) != 0) { throw std::runtime_error("bad residual-sample record"); }
        std::memcpy(&r.forward, header + 8, 8);
        std::memcpy(&r.boundary, header + 16, 4);
        std::memcpy(&r.rows, header + 28, 4);
        std::memcpy(&r.count, header + 32, 4);
        std::memcpy(&r.base, header + 40, 8);
        std::memcpy(&r.flags, header + 48, 4);
        r.columns.resize(static_cast<std::size_t>(r.count));
        r.ids.resize(static_cast<std::size_t>(r.count));
        std::vector<std::uint16_t> half(static_cast<std::size_t>(r.count) * r.rows);
        in.read(reinterpret_cast<char*>(r.columns.data()), 4 * r.count);
        in.read(reinterpret_cast<char*>(r.ids.data()), 4 * r.count);
        in.read(reinterpret_cast<char*>(half.data()), static_cast<std::streamsize>(2 * half.size()));
        for (const std::uint16_t bits : half) {
            _Float16 value;
            std::memcpy(&value, &bits, 2);
            r.values.push_back(static_cast<float>(value));
        }
        out.push_back(std::move(r));
    }
    return out;
}

/// The sampler as a pipeline drives it: forwards in flight together, every boundary seeing them
/// in the order they started, each copy taken without waiting -- the device buffer is rewritten
/// right after every call, in stream order -- and records written later. Then a forward abandoned
/// after boundary 0, which the token-id check must flag, and a reset.
int run_residual_case(const std::filesystem::path& dir) {
    constexpr int kRows = 8, kBoundaries = 3, kEvery = 3;
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    // value(f, b, c, r): exact in BF16 and FP16
    const auto value = [](int f, int b, int c, int r) {
        return static_cast<float>((f * 64 + b * 16 + c) * (r % 2 == 1 ? -1 : 1));
    };
    std::deque<DeviceBuffer> keep; // push_back keeps references to the others
    const auto call = [&](int f, int width, int b) {
        std::vector<float> host(static_cast<std::size_t>(kRows) * width);
        for (int c = 0; c < width; ++c) {
            for (int r = 0; r < kRows; ++r) { host[static_cast<std::size_t>(c) * kRows + r] = value(f, b, c, r); }
        }
        std::vector<int> token_ids(static_cast<std::size_t>(width));
        for (int c = 0; c < width; ++c) { token_ids[static_cast<std::size_t>(c)] = 1000 * (f + 1) + c; }
        keep.push_back(to_device_bf16(host));
        DeviceBuffer& residual = keep.back();
        keep.push_back(to_device_i32(token_ids));
        DeviceBuffer& id_buffer = keep.back();
        const Tensor residual_tensor(residual.p, DType::BF16, {kRows, width});
        const Tensor id_tensor = b == kBoundaries - 1 ? Tensor{} : Tensor(id_buffer.p, DType::I32, {width});
        ops::residual_sample(b, kBoundaries, 4 * b, residual_tensor, id_tensor, stream);
        // the next layer rewrites both buffers in stream order: the samples must predate it
        cuda_check(cudaMemsetAsync(residual.p, 0x7f, residual.bytes, stream), "rewrite residual");
        cuda_check(cudaMemsetAsync(id_buffer.p, 0x7f, id_buffer.bytes, stream), "rewrite ids");
    };
    int failures = 0;
    const auto expect = [&](bool ok, const std::string& what) {
        if (!ok) {
            std::printf("residual samples: %s WRONG\n", what.c_str());
            ++failures;
        }
    };
    // Forwards 0 (7 columns), 1 (5), 2 (4): a pipeline's interleaving.
    const std::vector<std::pair<int, int>> order{{0, 0}, {1, 0}, {0, 1}, {2, 0}, {1, 1}, {0, 2}, {2, 1}, {1, 2}, {2, 2}};
    const int widths[] = {7, 5, 4};
    for (const auto& [f, b] : order) { call(f, widths[f], b); }
    const std::string summary = ops::residual_samples_flush();
    auto records = read_samples(dir / "residual-samples.bin");
    // grid every 3 over 7 + 5 + 4 columns: forward 0 columns 0, 3, 6; forward 1 (base 7) column 2;
    // forward 2 (base 12) columns 0, 3
    const std::vector<std::vector<int>> grid{{0, 3, 6}, {2}, {0, 3}};
    expect(records.size() == 9, "9 records (" + std::to_string(records.size()) + ")");
    for (const SampleRecord& r : records) {
        const int f = static_cast<int>(r.forward);
        bool ok = f >= 0 && f < 3 && r.columns == grid[static_cast<std::size_t>(f)] && r.flags == 0 &&
                  r.rows == kRows && r.base == (f == 0 ? 0 : f == 1 ? 7 : 12);
        for (int i = 0; ok && i < r.count; ++i) {
            const int c = r.columns[static_cast<std::size_t>(i)];
            ok = r.ids[static_cast<std::size_t>(i)] == 1000 * (f + 1) + c;
            for (int row = 0; ok && row < kRows; ++row) {
                ok = r.values[static_cast<std::size_t>(i) * kRows + row] == value(f, r.boundary, c, row);
            }
        }
        expect(ok, "forward " + std::to_string(f) + " boundary " + std::to_string(r.boundary));
    }
    expect(summary.find("\"complete_forwards\": 3") != std::string::npos &&
               summary.find("\"dropped_forwards\": 0") != std::string::npos &&
               summary.find("\"unmatched_calls\": 0") != std::string::npos &&
               summary.find("\"sampled_columns\": 6") != std::string::npos,
           "summary " + summary.substr(0, 400));
    // Forward 3 starts and is lost after boundary 0; forward 4, of the same width, then gets
    // matched to it -- the case only the token ids can tell -- and its boundary-1 record is flagged.
    call(3, 6, 0);
    call(4, 6, 0);
    call(4, 6, 1);
    call(4, 6, 2);
    const std::string lost = ops::residual_samples_flush();
    records = read_samples(dir / "residual-samples.bin");
    int flagged = 0;
    for (const SampleRecord& r : records) { flagged += r.flags != 0 ? 1 : 0; }
    expect(flagged == 1 && lost.find("\"records_ids_differ\": 1") != std::string::npos,
           "abandoned forward flagged (" + std::to_string(flagged) + ")");
    // A reset starts the file over; forward 5 is sampled from a fresh grid.
    (void)ops::residual_samples_reset();
    call(5, 3, 0);
    call(5, 3, 1);
    call(5, 3, 2);
    const std::string fresh = ops::residual_samples_flush();
    records = read_samples(dir / "residual-samples.bin");
    expect(records.size() == 3 && records[0].forward == 0 && records[0].columns == std::vector<int>{0} &&
               records[0].ids == std::vector<int>{6000} && fresh.find("\"complete_forwards\": 1") != std::string::npos,
           "reset");
    cuda_check(cudaStreamSynchronize(stream), "sync");
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
    std::printf("residual samples: interleaved forwards, async copies, abandoned forward, reset (%s)\n",
                failures == 0 ? "ok" : "WRONG");
    return failures;
}

/// The same columns through one wide round and one decode round each (NVFP4 weights): a column's
/// output must not depend on the width it was computed at beyond rounding. Under the W4A16 route
/// both run our own kernels -- the student geometries' decode and prefill bodies included -- so a
/// width-dependent defect in either shows as a large difference; the auto route (the vendored
/// W4A4 runner from two columns) is printed for scale.
int run_cross_width(const ops::SparseMoeGeometry& g, const std::string& name) {
    if (!nvfp4_device("moe cross-width nvfp4")) { return 0; }
    const Layer layer = make_layer(g, QType::NVFP4);
    constexpr int kColumns = 64;
    const std::vector<float> x = make_inputs(g, 3, kColumns);
    std::vector<std::uint16_t> bits(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { bits[i] = f32_to_bf16(x[i]); }
    const DeviceBuffer all = to_device(bits);
    int failures = 0;
    for (const auto route : {ops::SparseMoeRoute::W4A16, ops::SparseMoeRoute::Auto}) {
        ops::set_sparse_moe_route(route);
        const std::vector<double> wide = run_moe(layer, all, kColumns, ops::SparseMoeRoundHook{}, ops::SparseMoeRouting::ByWidth);
        double worst = 0.0, sum = 0.0;
        for (int c = 0; c < kColumns; ++c) {
            const std::vector<std::uint16_t> column(bits.begin() + static_cast<std::ptrdiff_t>(c) * g.hidden,
                                                    bits.begin() + static_cast<std::ptrdiff_t>(c + 1) * g.hidden);
            const DeviceBuffer one_device = to_device(column);
            const std::vector<double> one = run_moe(layer, one_device, 1, ops::SparseMoeRoundHook{}, ops::SparseMoeRouting::ByWidth);
            const std::vector<double> want(wide.begin() + static_cast<std::ptrdiff_t>(c) * g.hidden,
                                           wide.begin() + static_cast<std::ptrdiff_t>(c + 1) * g.hidden);
            const double rel = relative_l2(one, want);
            worst = std::max(worst, rel);
            sum += rel;
        }
        const bool own = route == ops::SparseMoeRoute::W4A16;
        const bool ok = !own || worst < 2e-2;
        std::printf("%s nvfp4 cross-width (%s route): decode vs %d-column round, rel L2 mean %.2e max %.2e (%s)\n",
                    name.c_str(), own ? "w4a16" : "auto", kColumns, sum / kColumns, worst, ok ? "ok" : "WRONG");
        failures += ok ? 0 : 1;
    }
    ops::set_sparse_moe_route(ops::SparseMoeRoute::Auto);
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    std::string dump_dir;
    int experts = 128;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--dump") { dump_dir = argv[i + 1]; }
        if (std::string(argv[i]) == "--geometry") { experts = std::atoi(argv[i + 1]); }
    }
    if (argc >= 2 && std::string(argv[1]) == "--cross-width") {
        int bad = 0;
        for (const auto* geometry : {&ops::kSparseMoeFlashNextGeometry, &ops::kSparseMoeFlashNext192Geometry,
                                     &ops::kSparseMoeFlashNext160Geometry, &ops::kSparseMoeFlashNext128Geometry}) {
            bad += run_cross_width(*geometry, "flash-next-" + std::to_string(geometry->experts));
        }
        std::cout << (bad == 0 ? "OK" : "FAIL") << " moe cross-width\n";
        return bad == 0 ? 0 : 1;
    }
    const ops::SparseMoeGeometry g = experts == 512 ? ops::kSparseMoeFlashNextGeometry
                                   : experts == 192 ? ops::kSparseMoeFlashNext192Geometry
                                   : experts == 160 ? ops::kSparseMoeFlashNext160Geometry
                                                    : ops::kSparseMoeFlashNext128Geometry;
    const std::string name = "flash-next-" + std::to_string(g.experts);
    const std::filesystem::path out = dump_dir.empty()
        ? std::filesystem::temp_directory_path() / ("sinfer_moe_stats_test_" + std::to_string(::getpid()))
        : std::filesystem::path(dump_dir) / "stats";
    if (!dump_dir.empty()) { std::filesystem::create_directories(dump_dir); }
    ops::configure_moe_stats({.directory = out.string(), .norms = true, .mtp_prompt = false, .model = "synthetic"});
    ops::configure_residual_samples({.directory = (out / "residual").string(), .every = 3, .max_tokens = 1000,
                                     .model = "synthetic"});
    int failures = run_residual_case(out / "residual");
    // Widths: decode, small-T, the W8 prefill floor, a wide-plan round, one past a 1024 slice.
    const std::vector<int> w8_widths{1, 2, 5, 19, 20, 47, 300, 1100};
    {
        ops::moe_stats_prepare_device(2, 1, g.experts, g.experts_per_token, g.hidden);
        failures += run_case((name + "-w8").c_str(), g, QType::W8G32_F16S, 0, w8_widths, true, dump_dir);
        failures += run_graph_case(g, 1);
        if (nvfp4_device("moe stats nvfp4")) {
            // Decode on our W4A16 kernels, the TRT-LLM runner from two columns.
            failures += run_case((name + "-nvfp4").c_str(), g, QType::NVFP4, 0, {1, 2, 17, 300, 1100}, false, "");
        }
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " moe stats\n";
    if (dump_dir.empty()) { std::filesystem::remove_all(out); }
    return failures == 0 ? 0 : 1;
}
