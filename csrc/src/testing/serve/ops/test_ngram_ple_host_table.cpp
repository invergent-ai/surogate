// The n-gram table read in place from files (api/ops/ngram_ple_host_table.h) against the resident
// table it replaces. Every check runs the whole n-gram forward:
//
//   - a BF16 host table holding exactly the values a resident IQ4_NL table decodes to: the forward's
//     outputs, token histories and convolution states are bit-identical, over several rounds, with
//     the rows spread over shards of uneven size in two files at unaligned offsets;
//   - an E4M3 host table with a per-tensor scale against a BF16 host table holding the BF16
//     rounding of each code times the scale: bit-identical;
//   - the same forward captured in a CUDA graph (the host gather is a captured host function) and
//     replayed with new token ids: bit-identical to the eager forward of those ids;
//   - NgramPleHostTable::gather itself, row by row against the files, with a wide request that
//     takes the helper threads.

#include "api/ops/ngram_ple.h"
#include "api/ops/ngram_ple_host_table.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/ngram_ple_state.h"
#include "ops/op_tester.h"

#include <cuda_fp8.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr int kRows = 64, kHeadDim = 32, kHeads = 2;
constexpr int kBatch = 3, kWidth = 5, kTokens = kBatch * kWidth, kSlots = 4;
constexpr int kHidden = 32, kStreams = 2, kChannels = kHidden * kStreams;
constexpr int kKernel = 4, kDilation = 3, kHistory = (kKernel - 1) * kDilation;

Weight projection(const Tensor& tensor) {
    Weight weight{};
    weight.qtype = QType::BF16_CTRL;
    weight.layout = QuantLayout::Contiguous;
    weight.payload = weight.qdata = tensor.data;
    weight.payload_bytes = tensor.bytes();
    weight.n = weight.shape[0] = weight.padded_shape[0] = tensor.ne[0];
    weight.k = weight.shape[1] = weight.padded_shape[1] = tensor.ne[1];
    weight.ndim = 2;
    return weight;
}

constexpr std::int8_t kIq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// A resident IQ4_NL table (FP16 scale 1/128 per block) and the BF16 values it decodes to, which
// are exact in BF16.
void iq4_table(std::vector<std::uint8_t>& blocks, std::vector<std::uint16_t>& values) {
    blocks.assign(kRows * 18, 0);
    values.assign(static_cast<std::size_t>(kRows) * kHeadDim, 0);
    for (int row = 0; row < kRows; ++row) {
        blocks[row * 18 + 1] = 0x20;
        for (int d = 0; d < 16; ++d) {
            const int lo = (row + d * 3) % 16, hi = (row * 3 + d) % 16;
            blocks[row * 18 + 2 + d] = static_cast<std::uint8_t>(lo | (hi << 4));
            values[static_cast<std::size_t>(row) * kHeadDim + d]      = f32_to_bf16(kIq4[lo] / 128.0F);
            values[static_cast<std::size_t>(row) * kHeadDim + 16 + d] = f32_to_bf16(kIq4[hi] / 128.0F);
        }
    }
}

// Writes `rows` (row-major, `row_bytes` each) as shards of the given sizes into two files, each
// shard behind a few bytes of unrelated header, and returns their descriptors.
std::vector<ops::NgramPleShard> write_shards(const std::filesystem::path& dir, const std::string& tag,
                                             const std::vector<std::uint8_t>& rows, int row_bytes) {
    const int sizes[] = {25, 25, 14};
    std::vector<ops::NgramPleShard> shards;
    std::ofstream files[2] = {std::ofstream(dir / (tag + "-a.bin"), std::ios::binary),
                              std::ofstream(dir / (tag + "-b.bin"), std::ios::binary)};
    std::uint64_t written[2] = {0, 0};
    int first = 0;
    for (int s = 0; s < 3; ++s) {
        const int f = s < 2 ? 0 : 1;
        const std::string pad(static_cast<std::size_t>(13 + 7 * s), 'h');
        files[f].write(pad.data(), static_cast<std::streamsize>(pad.size()));
        written[f] += pad.size();
        shards.push_back({(dir / (tag + (f == 0 ? "-a.bin" : "-b.bin"))).string(), written[f], sizes[s]});
        const std::size_t bytes = static_cast<std::size_t>(sizes[s]) * row_bytes;
        files[f].write(reinterpret_cast<const char*>(rows.data()) + static_cast<std::size_t>(first) * row_bytes,
                       static_cast<std::streamsize>(bytes));
        written[f] += bytes;
        first += sizes[s];
    }
    return shards;
}

std::vector<std::uint8_t> as_bytes(const std::vector<std::uint16_t>& words) {
    std::vector<std::uint8_t> out(words.size() * 2);
    std::memcpy(out.data(), words.data(), out.size());
    return out;
}

struct Fixture {
    DeviceArena arena{8U << 20};
    Tensor key, value, conv, norm;
    ops::NgramPleWeights weights;
    ops::NgramPleHash hash;
    ops::NgramPleColumns columns;
    WorkspaceArena workspace{ops::ngram_ple_workspace_capacity_bytes(kStreams, kHidden, kHeads * kHeadDim, kHeads, 1, kTokens)};

    Tensor patterned(std::initializer_list<std::int32_t> shape, int salt) {
        auto tensor = arena.alloc(DType::BF16, shape);
        std::vector<std::uint16_t> values(tensor.numel());
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = f32_to_bf16(float(int((i * 7 + salt) % 29) - 14) / 64.0F);
        }
        CUDA_CHECK(cudaMemcpy(tensor.data, values.data(), tensor.bytes(), cudaMemcpyHostToDevice));
        return tensor;
    }

    Fixture() {
        key   = patterned({kChannels, kHeads * kHeadDim}, 1);
        value = patterned({kHidden, kHeads * kHeadDim}, 3);
        conv  = patterned({kChannels, kKernel}, 5);
        norm  = arena.alloc(DType::FP32, {kChannels});
        const std::vector<float> norms(kChannels, 1.0F);
        CUDA_CHECK(cudaMemcpy(norm.data, norms.data(), norm.bytes(), cudaMemcpyHostToDevice));
        weights = ops::NgramPleWeights{projection(key), projection(value), norm, norm, norm, conv};
        hash.ngram = 3;
        hash.heads = kHeads;
        hash.eos_token = 0;
        hash.multipliers[0] = 17; hash.multipliers[1] = 29; hash.multipliers[2] = 43;
        hash.head_vocab_sizes[0] = 31; hash.head_vocab_sizes[1] = 33; hash.head_offsets[1] = 31;
        columns = ops::NgramPleColumns{arena.alloc(DType::I32, {kTokens}), arena.alloc(DType::I32, {kTokens}),
                                       arena.alloc(DType::I32, {kTokens}), arena.alloc(DType::I32, {kTokens})};
        auto slot_storage = to_device(std::vector<int>{2, 0, 3});
        Tensor lane_slots(slot_storage.p, DType::I32, {kBatch});
        ops::ngram_ple_expand_columns(lane_slots, kWidth, columns.slots, columns.segment_begin,
                                      columns.segment_last, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ops::NgramPleState state() {
        ops::NgramPleState out{arena.alloc(DType::I32, {2, kSlots}), patterned({kHistory, kChannels, kSlots}, 7)};
        const std::vector<int> previous{7, 9, 11, 13, 17, 19, 23, 29};
        CUDA_CHECK(cudaMemcpy(out.history.data, previous.data(), out.history.bytes(), cudaMemcpyHostToDevice));
        return out;
    }

    void ids(int round, cudaStream_t stream) {
        std::vector<int> host(kTokens);
        for (int i = 0; i < kTokens; ++i) { host[i] = i + 31 + 17 * round; }
        host[kWidth + 1] = 0; // EOS cuts predecessor history.
        CUDA_CHECK(cudaMemcpyAsync(columns.ids.data, host.data(), columns.ids.bytes(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

struct Outputs {
    std::vector<std::uint16_t> residual;
    std::vector<int> history;
    std::vector<std::uint16_t> conv;
};

// Two rounds of the forward on fresh state; `graph` captures the forward once and replays it.
Outputs run(Fixture& fx, const ops::NgramPleTable& table, bool graph) {
    ops::NgramPleState st = fx.state();
    Outputs out;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int round = 0; round < 2; ++round) {
        auto residual = fx.patterned({kChannels, kTokens}, 11 + round);
        fx.ids(round, stream);
        if (!graph) {
            ops::ngram_ple_forward(residual, fx.columns, fx.hash, table, fx.weights, st, kStreams, kKernel,
                                   kDilation, 1e-6F, fx.workspace, stream);
        } else {
            cudaGraph_t captured = nullptr;
            cudaGraphExec_t exec = nullptr;
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            ops::ngram_ple_forward(residual, fx.columns, fx.hash, table, fx.weights, st, kStreams, kKernel,
                                   kDilation, 1e-6F, fx.workspace, stream);
            CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
            CUDA_CHECK(cudaGraphInstantiate(&exec, captured, 0));
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaGraphExecDestroy(exec));
            CUDA_CHECK(cudaGraphDestroy(captured));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto r = from_device<std::uint16_t>(residual.data, residual.numel());
        out.residual.insert(out.residual.end(), r.begin(), r.end());
    }
    out.history = from_device<int>(st.history.data, st.history.numel());
    out.conv    = from_device<std::uint16_t>(st.conv_state.data, st.conv_state.numel());
    CUDA_CHECK(cudaStreamDestroy(stream));
    return out;
}

int compare(const std::string& label, const Outputs& a, const Outputs& b) {
    int failures = 0;
    failures += verify_exact((label + ": outputs").c_str(), a.residual, b.residual);
    failures += verify_exact((label + ": token histories").c_str(), a.history, b.history);
    failures += verify_exact((label + ": convolution states").c_str(), a.conv, b.conv);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("sinfer-ngram-host-" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    int failures = 0;
    try {
        Fixture fx;
        std::vector<std::uint8_t> blocks;
        std::vector<std::uint16_t> values;
        iq4_table(blocks, values);
        auto resident_storage = to_device(blocks);
        const ops::NgramPleTable resident{resident_storage.p, kRows, 18, kHeadDim};

        ops::NgramPleHostTable bf16(write_shards(dir, "bf16", as_bytes(values), 2 * kHeadDim), kHeadDim,
                                    ops::NgramPleRowType::Bf16, 1.0F);
        bf16.prepare(kTokens * kHeads);
        ops::NgramPleTable hosted{nullptr, kRows, 2 * kHeadDim, kHeadDim, &bf16};
        const Outputs want = run(fx, resident, false);
        failures += compare("BF16 host table against the resident IQ4_NL table", run(fx, hosted, false), want);
        failures += compare("BF16 host table, captured and replayed", run(fx, hosted, true), want);

        // E4M3 codes (finite ones, both signs) with a scale, against BF16 rows of code * scale.
        std::mt19937 rng(20261102);
        std::vector<std::uint8_t> codes(static_cast<std::size_t>(kRows) * kHeadDim);
        std::vector<std::uint16_t> widened(codes.size());
        constexpr float kScale = 0.0625F;
        for (std::size_t i = 0; i < codes.size(); ++i) {
            std::uint8_t code = static_cast<std::uint8_t>(rng() & 0xFF);
            if ((code & 0x7F) == 0x7F) { code ^= 0x01; } // 0x7F / 0xFF are NaN in E4M3FN
            codes[i] = code;
            __nv_fp8_e4m3 v;
            v.__x      = code;
            widened[i] = f32_to_bf16(static_cast<float>(v) * kScale);
        }
        ops::NgramPleHostTable e4m3(write_shards(dir, "e4m3", codes, kHeadDim), kHeadDim,
                                    ops::NgramPleRowType::E4m3, kScale);
        ops::NgramPleHostTable e4m3_bf16(write_shards(dir, "e4m3-bf16", as_bytes(widened), 2 * kHeadDim), kHeadDim,
                                         ops::NgramPleRowType::Bf16, 1.0F);
        e4m3.prepare(kTokens * kHeads);
        e4m3_bf16.prepare(kTokens * kHeads);
        const ops::NgramPleTable fp8_table{nullptr, kRows, kHeadDim, kHeadDim, &e4m3};
        const ops::NgramPleTable fp8_reference{nullptr, kRows, 2 * kHeadDim, kHeadDim, &e4m3_bf16};
        failures += compare("E4M3 host table (scale 1/16) against BF16 rows of code x scale",
                            run(fx, fp8_table, false), run(fx, fp8_reference, false));
        failures += compare("E4M3 host table, captured and replayed", run(fx, fp8_table, true),
                            run(fx, fp8_reference, false));

        // The gather on its own: every row, then a request wide enough for the helper threads.
        std::vector<std::int32_t> ids(20000);
        for (std::size_t i = 0; i < ids.size(); ++i) { ids[i] = static_cast<std::int32_t>((i * 37 + 5) % kRows); }
        std::vector<std::uint8_t> got(ids.size() * 2 * kHeadDim);
        bf16.gather(ids.data(), static_cast<std::int64_t>(ids.size()), got.data());
        const auto bytes = as_bytes(values);
        int wrong = 0;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            wrong += std::memcmp(got.data() + i * 2 * kHeadDim, bytes.data() + static_cast<std::size_t>(ids[i]) * 2 * kHeadDim,
                                 2 * kHeadDim) != 0;
        }
        if (wrong != 0 || bf16.bad_ids() != 0) {
            std::cerr << "gather: " << wrong << " of " << ids.size() << " rows differ from the files\n";
            ++failures;
        }
    } catch (const std::exception& error) {
        std::cerr << "ngram_ple host table: " << error.what() << '\n';
        ++failures;
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " ngram_ple host table\n";
    return failures == 0 ? 0 : 1;
}
