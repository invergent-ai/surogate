#include "family/impl/lora_globals.h"
#include "core/engine_context.h"
#include "api/ops/embedding.h"
#include "api/types.h"
#include "api/ops/lora_router.h"
#include "api/ops/linear.h"
#include "api/ops/add_bias.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "ops/linear/ggml/ggml_blocks.h"
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace sinfer;
namespace test = sinfer::test;

static void check(const std::vector<double>& actual, const std::vector<double>& expected) {
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > .004 + .008 * std::abs(expected[i])) {
            std::cerr << "extended adapter mismatch " << i << ": " << actual[i] << " vs " << expected[i] << '\n';
            std::abort();
        }
    }
}
static Weight weight(void* data, int n, int k) {
    Weight w;
    w.payload = w.qdata = data;
    w.payload_bytes = n * k * 2;
    w.qtype = QType::BF16_CTRL;
    w.layout = QuantLayout::Contiguous;
    w.ndim = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    return w;
}
static void router_bias() {
    const auto g = ops::kSparseMoeLfm2Moe32Geometry;
    const int e = g.experts, top = g.experts_per_token, count = 3;
    std::vector<float> scores(e * count), base(e), delta(2 * e, 0);
    std::vector<int> ids(top * count), expected_ids(top * count);
    std::vector<double> expected(top * count);
    std::vector<int> selected{-1, 0, 1};
    delta[e - 1] = 2;
    delta[2 * e - 2] = 3;
    for (int j = 0; j < e; ++j) {
        base[j] = (j % 3) / 16.F;
    }
    for (int t = 0; t < count; ++t) {
        std::vector<int> order(e);
        for (int j = 0; j < e; ++j) {
            scores[t * e + j] = (j % 4) / 4.F;
            order[j] = j;
        }
        const auto sigmoid = [&](int j) {
            return 1 / (1 + std::exp(-double(scores[t * e + j])));
        };
        const auto ranked = [&](int j) {
            return sigmoid(j) + base[j] + (selected[t] < 0 ? 0 : delta[selected[t] * e + j]);
        };
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return ranked(a) == ranked(b) ? a < b : ranked(a) > ranked(b);
        });
        double total = 0;
        for (int j = 0; j < top; ++j) {
            total += sigmoid(order[j]);
        }
        for (int j = 0; j < top; ++j) {
            expected_ids[t * top + j] = order[j];
            expected[t * top + j] = sigmoid(order[j]) / total;
        }
    }
    std::vector<float> initial(top * count);
    for (int t = 0; t < count; ++t)
        for (int j = 0; j < top; ++j) {
            ids[t * top + j] = expected_ids[j];
            initial[t * top + j] = expected[j];
        }
    assert(ids != expected_ids);
    auto ds = test::to_device_f32(scores), db = test::to_device_f32(base), dd = test::to_device_f32(delta);
    auto di = test::to_device_i32(ids), da = test::to_device_f32(initial), dslots = test::to_device_i32(selected);
    ops::LoraBank banks[2]{};
    banks[1].bias = static_cast<const float*>(dd.p);
    banks[1].n = e;
    DeviceBuffer table(sizeof(banks));
    test::cuda_check(cudaMemcpy(table.p, banks, sizeof(banks), cudaMemcpyHostToDevice), "router table");
    ops::lora_router_bias(Tensor(ds.p, DType::FP32, {e, count}),
                          Tensor(di.p, DType::I32, {top, count}),
                          Tensor(da.p, DType::FP32, {top, count}),
                          static_cast<const float*>(db.p),
                          nullptr,
                          g,
                          static_cast<const ops::LoraBank*>(table.p),
                          static_cast<const int*>(dslots.p),
                          1,
                          nullptr);
    test::cuda_check(cudaMemcpy(ids.data(), di.p, ids.size() * sizeof(int), cudaMemcpyDeviceToHost), "router ids");
    assert(ids == expected_ids);
    check(test::from_device_f32(da, top * count), expected);
}

static void quantized_norms() {
    constexpr int n = 128, k = 256, rank = 2;
    std::vector<std::uint16_t> a(rank * k), b(n * rank);
    for (int i = 0; i < rank * k; ++i) {
        a[i] = test::f32_to_bf16((i % 5 - 2) / 32.F);
    }
    for (int i = 0; i < n * rank; ++i) {
        b[i] = test::f32_to_bf16((i % 3 - 1) / 16.F);
    }
    for (auto type : {QType::W8G32_F16S,
                      QType::Q4G64_F16S,
                      QType::Q5G64_F16S,
                      QType::Q6G64_F16S,
                      QType::FP8_E4M3FN_ROW_BF16S,
                      QType::NVFP4}) {
        test::quantized_weight::PatternedWeightOptions options;
        if (type == QType::NVFP4) {
            options.weight_scale_divisor = 1.75F;
            options.input_scale_divisor = .5F;
        }
        auto packed = test::quantized_weight::make_patterned_weight(type, n, k, 241, options);
        DeviceBuffer storage(packed.payload.size());
        test::cuda_check(cudaMemcpy(storage.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice),
                         "quantized base");
        const auto actual = ops::lora_weight_norms({{packed.device_weight(storage.p), 0, 0, n}}, a, b, rank, k, n, 1);
        for (int r = 0; r < n; ++r) {
            double sum = 0;
            for (int c = 0; c < k; ++c) {
                double value = test::quantized_weight::logical_weight_fp64(packed, r, c);
                for (int q = 0; q < rank; ++q) {
                    value += test::bf16_to_f32(a[q * k + c]) * test::bf16_to_f32(b[r * rank + q]);
                }
                sum += value * value;
            }
            const double expected = std::sqrt(sum);
            if (std::abs(actual[r] - expected) > 2e-5 * expected) {
                std::cerr << "DoRA quantized norm mismatch type " << int(type) << " row " << r << ": " << actual[r]
                          << " vs " << expected << '\n';
                std::abort();
            }
        }
    }
}

static void shared_automatic_projections() {
    ops::EngineOpsContext context;
    ops::bind_ops_context(&context);
    constexpr int h = 64, tokens = 4, rank = 2;
    std::vector<ops::detail::ggml::block_q8_0> blocks(h * h / 32);
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        blocks[i].d = __float2half_rn(.03125f);
        for (int j = 0; j < 32; ++j) { blocks[i].qs[j] = (i + j) % 9 - 4; }
    }
    DeviceBuffer data(blocks.size() * sizeof(blocks[0]));
    data.copy_from_host(blocks.data(), data.bytes);
    Weight w = weight(data.p, h, h);
    w.qtype = QType::Q8_0; w.layout = QuantLayout::GgmlBlocks;
    w.group = w.group_size = 32; w.scale_dtype = DType::FP16; w.payload_bytes = data.bytes;
    auto& store = ops::lora_store_for_current_device();
    store.configure(2, rank, tokens);
    family::bind_lora_auto(store, -1, "shared_head", w, {});
    store.ensure_banks(); store.set_active(true);
    std::vector<std::uint16_t> a(rank * h), b(h * rank);
    for (std::size_t i = 0; i < a.size(); ++i) { a[i] = test::f32_to_bf16((int(i % 7) - 3) / 16.f); }
    for (std::size_t i = 0; i < b.size(); ++i) { b[i] = test::f32_to_bf16((int(i % 5) - 2) / 32.f); }
    store.set_module_slot(-1, "shared_head", 0, a, b, rank, h, h, 2.f, std::vector<float>(h, .75f));
    store.set_module_slot(-1, "shared_head", 1, a, b, rank, h, h, -1.f);
    auto ids = test::to_device_i32(std::vector<int>{-1, 0, 1, 0});
    Tensor slots(ids.p, DType::I32, {tokens});
    ops::lora_set_round({.slots = &slots, .scratch = store.scratch(tokens)});
    auto input = test::to_device_bf16(std::vector<float>(h * tokens, .25f));
    Tensor x(input.p, DType::BF16, {h, tokens});
    std::array<DeviceBuffer, 3> buffers;
    std::array<Tensor, 3> outputs;
    for (int i = 0; i < 3; ++i) {
        buffers[i] = DeviceBuffer(h * tokens * 2);
        outputs[i] = Tensor(buffers[i].p, DType::BF16, {h, tokens});
    }
    cudaStream_t stream;
    test::cuda_check(cudaStreamCreate(&stream), "shared adapter stream");
    const auto grouped = [&] {
        ops::linear_projections(x, {{w, outputs[0]}, {w, outputs[1]},
                                    {w, outputs[2], ops::LinearPolicy::A16Only, 0}}, nullptr, stream);
    };
    grouped(); test::cuda_check(cudaStreamSynchronize(stream), "shared adapter warmup");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    test::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture shared adapters");
    grouped();
    test::cuda_check(cudaStreamEndCapture(stream, &graph), "end shared adapters");
    test::cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate shared adapters");
    for (int round = 0; round < 3; ++round) {
        std::vector<int> selection{round - 1, 0, 1, 1 - round};
        ids.copy_from_host(selection.data(), ids.bytes);
        auto values = test::to_device_bf16(std::vector<float>(h * tokens, .125f * (round + 1)));
        test::cuda_check(cudaMemcpy(input.p, values.p, input.bytes, cudaMemcpyDeviceToDevice), "update shared input");
        ops::linear(x, w, outputs[0], stream); ops::linear(x, w, outputs[1], stream);
        ops::linear_rows(x, w, 0, outputs[2], nullptr, stream);
        test::cuda_check(cudaStreamSynchronize(stream), "scalar adapters");
        std::array<std::vector<std::uint16_t>, 3> expected;
        for (int i = 0; i < 3; ++i) {
            expected[i].resize(h * tokens);
            buffers[i].copy_to_host(expected[i].data(), buffers[i].bytes);
            buffers[i].fill(0xff);
        }
        test::cuda_check(cudaGraphLaunch(exec, stream), "replay shared adapters");
        test::cuda_check(cudaStreamSynchronize(stream), "shared adapter outputs");
        for (int i = 0; i < 3; ++i) {
            std::vector<std::uint16_t> actual(h * tokens);
            buffers[i].copy_to_host(actual.data(), buffers[i].bytes);
            assert(actual == expected[i]);
        }
    }
    test::cuda_check(cudaGraphExecDestroy(exec), "destroy shared adapter graph");
    test::cuda_check(cudaGraphDestroy(graph), "destroy shared adapter source");
    test::cuda_check(cudaStreamDestroy(stream), "destroy shared adapter stream");
    ops::lora_clear_round(); ops::bind_ops_context(nullptr);
}

int main() {
    if (test::cuda_unavailable()) {
        return 77;
    }
    shared_automatic_projections();
    quantized_norms();
    router_bias();
    ops::EngineOpsContext context;
    ops::bind_ops_context(&context);
    constexpr int h = 64, tokens = 4, rank = 2;
    std::vector<float> base(h * h), a(rank * h), b(h * rank), magnitude(h), original(h, .125F), saved(h, .25F),
        lbias(h, .03125F);
    for (int r = 0; r < h; ++r) {
        magnitude[r] = .75F + (r % 4) / 16.F;
        b[r * rank] = (r % 3 - 1) / 32.F;
        b[r * rank + 1] = (r % 5 - 2) / 64.F;
        for (int k = 0; k < h; ++k) {
            base[r * h + k] = (r == k ? .5F : ((r + k) % 3 - 1) / 64.F);
        }
    }
    for (int k = 0; k < h; ++k) {
        a[k] = (k % 3 - 1) / 16.F;
        a[h + k] = (k % 4 - 2) / 32.F;
    }
    auto db = test::to_device_bf16(base), dob = test::to_device_bf16(original);
    auto dense = weight(db.p, h, h);
    Tensor base_bias(dob.p, DType::BF16, {h});
    auto& store = ops::lora_store_for_current_device();
    store.configure(2, 96, 2);  // also exercises ranks above the one-launch limit and chunked heads
    family::bind_lora_auto(store, -1, "lm_head", dense, base_bias);
    store.register_module(-1, "embed_tokens", {dense.qdata, ops::kLoraEmbeddingPort, h, h});
    store.register_base(dense.qdata, ops::kLoraEmbeddingPort, {{dense, 0, 0, h, true}});
    family::bind_lora_bias(store, -2, "vision.norm1", base_bias);
    // A vision Q/K/V projection shares one output but uses three independent adapters.
    std::vector<float> qkv(3 * h * h);
    for (int i = 0; i < 3; ++i) {
        std::copy(base.begin(), base.end(), qkv.begin() + i * h * h);
    }
    auto dqkv = test::to_device_bf16(qkv);
    auto vw = weight(dqkv.p, 3 * h, h);
    for (int i = 0; i < 3; ++i) {
        family::bind_lora_auto(store,
                               -2,
                               "vision.qkv." + std::to_string(i),
                               vw,
                               {},
                               i * h,
                               h,
                               ops::kLoraAutomaticPort + i);
    }
    store.register_replacement(dense.qdata, ops::kLoraAutomaticPort);
    store.register_replacement(dense.qdata, ops::kLoraEmbeddingPort);
    store.ensure_banks();
    store.set_active(true);
    std::vector<std::uint16_t> abits, bbits;
    for (float v : a) {
        abits.push_back(test::f32_to_bf16(v));
    }
    for (float v : b) {
        bbits.push_back(test::f32_to_bf16(v));
    }
    store.set_module_slot(-1, "lm_head", 0, abits, bbits, rank, h, h, 2, magnitude, saved, lbias);
    store.set_module_slot(-1, "lm_head", 1, abits, bbits, rank, h, h, -1);
    store.set_module_slot(-1, "embed_tokens", 0, abits, bbits, rank, h, h, 2, magnitude);
    store.set_module_slot(-2, "vision.qkv.1", 0, abits, bbits, rank, h, h, 2, magnitude);
    store.set_module_slot(-2, "vision.norm1.bias", 1, {0}, std::vector<std::uint16_t>(h), 1, 1, h, 1, {}, saved);
    std::vector<int> selection{-1, 0, 1, 0}, token_ids{1, 3, 5, 7};
    auto ds = test::to_device_i32(selection), dt = test::to_device_i32(token_ids);
    Tensor slots(ds.p, DType::I32, {tokens}), ids(dt.p, DType::I32, {tokens});
    ops::lora_set_round({.slots = &slots, .scratch = store.scratch(2)});
    std::vector<float> input(h * tokens, 0);
    for (int t = 0; t < tokens; ++t) {
        input[t * h + token_ids[t]] = 1;
    }
    auto dx = test::to_device_bf16(input);
    DeviceBuffer dy(h * tokens * 2), de(h * tokens * 2), dv(3 * h * tokens * 2);
    Tensor x(dx.p, DType::BF16, {h, tokens}), y(dy.p, DType::BF16, {h, tokens});
    Tensor embedding(de.p, DType::BF16, {h, tokens}), visual(dv.p, DType::BF16, {3 * h, tokens});
    const auto delta = [&](int row, int col) {
        return b[row * rank] * a[col] + b[row * rank + 1] * a[h + col];
    };
    bool replacement_loaded = false, slot0_loaded = true;
    std::vector<float> saved_head = base, saved_embed = base;
    for (int i = 0; i < h * h; ++i) {
        saved_head[i] += .125F;
        saved_embed[i] -= .25F;
    }
    auto expected = [&](bool embed) {
        std::vector<double> values(h * tokens);
        for (int t = 0; t < tokens; ++t)
            for (int r = 0; r < h; ++r) {
                const int selected = selection[t] == 0 && !slot0_loaded ? -1 : selection[t];
                const auto& matrix = replacement_loaded && selected == 0 ? (embed ? saved_embed : saved_head) : base;
                const auto bw = [&](int k) {
                    return embed ? matrix[k * h + r] : matrix[r * h + k];
                };
                double value = bw(token_ids[t]);
                if (selected == 0) {
                    double norm = 0;
                    for (int k = 0; k < h; ++k) {
                        const double v = bw(k) + 2 * delta(r, k);
                        norm += v * v;
                    }
                    const double gain = magnitude[r] / std::sqrt(norm);
                    value = gain * (value + 2 * delta(r, token_ids[t]) + (embed ? 0 : 2 * lbias[r]));
                } else if (selected == 1 && !embed) {
                    value -= delta(r, token_ids[t]);
                }
                if (!embed) {
                    value += selected == 0 || selected == 1 ? saved[r] : original[r];
                }
                values[t * h + r] = value;
            }
        return values;
    };
    auto run = [&](cudaStream_t stream) {
        ops::linear(x, dense, y, stream);
        ops::add_bias(base_bias, y, stream);
        ops::embedding(ids, dense, embedding, stream);
        ops::linear(x, vw, visual, stream);
    };
    auto verify = [&] {
        test::cuda_synchronize();
        check(test::from_device_bf16(dy, h * tokens), expected(false));
        check(test::from_device_bf16(de, h * tokens), expected(true));
        std::vector<double> values(3 * h * tokens);
        for (int t = 0; t < tokens; ++t)
            for (int part = 0; part < 3; ++part)
                for (int r = 0; r < h; ++r) {
                    double v = base[r * h + token_ids[t]];
                    if (part == 1 && selection[t] == 0 && slot0_loaded) {
                        double norm = 0;
                        for (int k = 0; k < h; ++k) {
                            double w = base[r * h + k] + 2 * delta(r, k);
                            norm += w * w;
                        }
                        v = (v + 2 * delta(r, token_ids[t])) * magnitude[r] / std::sqrt(norm);
                    }
                    values[(t * 3 + part) * h + r] = v;
                }
        check(test::from_device_bf16(dv, 3 * h * tokens), values);
    };
    run(nullptr);
    verify();
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    test::cuda_check(cudaStreamCreate(&stream), "stream");
    test::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
    run(stream);
    test::cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    test::cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    selection = {0, 1, -1, 0};
    test::cuda_check(cudaMemcpy(ds.p, selection.data(), ds.bytes, cudaMemcpyHostToDevice), "change slots");
    test::cuda_check(cudaGraphLaunch(exec, stream), "replay");
    verify();
    // Load full PEFT embedding/head weights after capture. Tied bases remain independent.
    std::vector<EngineOptions::LoraModulePayload> payloads;
    for (bool embed : {false, true}) {
        EngineOptions::LoraModulePayload payload;
        payload.layer = -1;
        payload.module = embed ? "embed_tokens" : "lm_head";
        payload.slot = 0;
        payload.rank = rank;
        payload.in_dim = payload.out_dim = h;
        payload.scale = 2;
        payload.a = abits;
        payload.b = bbits;
        payload.magnitude = magnitude;
        if (!embed) {
            payload.bias = saved;
            payload.lora_bias = lbias;
        }
        for (float v : embed ? saved_embed : saved_head) {
            payload.base_weight.push_back(test::f32_to_bf16(v));
        }
        payloads.push_back(std::move(payload));
    }
    store.validate_payloads(payloads);
    for (const auto& payload : payloads) {
        assert(payload.prepared_replacements.size() == 1);
        store.set_payload(0, payload);
        assert(payload.prepared_replacements.empty());
    }
    replacement_loaded = true;
    test::cuda_check(cudaGraphLaunch(exec, stream), "replay saved weights");
    verify();
    auto invalid = payloads;
    invalid.back().base_weight.pop_back();
    bool rejected = false;
    try {
        store.validate_payloads(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
    test::cuda_check(cudaGraphLaunch(exec, stream), "replay after rejected weights");
    verify();
    store.clear_slot(0);
    slot0_loaded = false;
    test::cuda_check(cudaGraphLaunch(exec, stream), "replay cleared weights");
    verify();
    test::cuda_check(cudaGraphExecDestroy(exec), "destroy graph");
    test::cuda_check(cudaGraphDestroy(graph), "destroy source");
    test::cuda_check(cudaStreamDestroy(stream), "destroy stream");
    ops::lora_clear_round();
    ops::bind_ops_context(nullptr);
    std::cout
        << "DoRA, saved biases, tied embeddings/heads and strided vision projections passed eagerly and under graphs\n";
}
