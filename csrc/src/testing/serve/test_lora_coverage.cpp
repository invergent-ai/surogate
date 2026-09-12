#include "family/impl/lora_bind.h"
#include "core/engine_context.h"
#include "targets/qwen3_5/impl/variant.h"
#include "targets/spark2_5/impl/variant.h"
#include "ops/op_tester.h"
#include "family/impl/moe/expert_cache.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace sinfer;
namespace test = sinfer::test;

struct Context {
    ops::EngineOpsContext context;
    Context() {
        ops::bind_ops_context(&context);
    }
    ~Context() {
        ops::lora_clear_round();
        ops::bind_ops_context(nullptr);
    }
};

struct ZeroWeight {
    DeviceBuffer codes, scales;
    Weight weight;
    ZeroWeight(int n, int k, bool w8 = false)
        : codes(std::size_t(n) * k * (w8 ? 1 : 2)),
          scales(w8 ? std::size_t(n) * (k / 32) * 2 : 0) {
        codes.fill(0);
        if (w8) {
            scales.fill(0);
        }
        weight.payload = weight.qdata = codes.p;
        weight.payload_bytes = codes.bytes + scales.bytes;
        weight.qtype = w8 ? QType::W8G32_F16S : QType::BF16_CTRL;
        weight.n = weight.shape[0] = weight.padded_shape[0] = n;
        weight.k = weight.shape[1] = weight.padded_shape[1] = k;
        weight.ndim = 2;
        weight.layout = w8 ? QuantLayout::RowSplit : QuantLayout::Contiguous;
        if (w8) {
            weight.scales = scales.p;
            weight.group = weight.group_size = 32;
            weight.scale_dtype = DType::FP16;
        }
    }
};

static void
adapter(ops::LoraStore& store, int layer, const std::string& module, int slot, int in, const std::vector<float>& rows) {
    std::vector<std::uint16_t> a(in, 0), b;
    a[0] = test::f32_to_bf16(1);
    for (float value : rows) {
        b.push_back(test::f32_to_bf16(value));
    }
    store.set_module_slot(layer, module, slot, a, b, 1, in, rows.size(), 1);
}

static void close(const std::vector<double>& actual, const std::vector<double>& expected) {
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 0.002 + 0.006 * std::abs(expected[i])) {
            std::cerr << "mismatch at " << i << ": " << actual[i] << " vs " << expected[i] << '\n';
            std::abort();
        }
    }
}

static void gated_attention(int storage) {
    Context context;
    auto& store = ops::lora_store_for_current_device();
    store.configure(2, 4, 4);
    constexpr int h = 64, q = 64, kv = 32, d = 16, tokens = 4;
    family::TextGeometry g;
    g.hidden = h;
    g.query_heads = q / d;
    g.kv_heads = kv / d;
    g.head_dim = d;
    ZeroWeight fused(2 * q + 2 * kv, h), qk(q + kv, h), gv(q + kv, h);
    ZeroWeight qg(2 * q, h), k(kv, h), v(kv, h);
    using namespace targets::qwen3_5::detail;
    FullAttentionProjectionPayload weights;
    if (storage == 0) {
        weights = FusedAttentionProjectionPayload{fused.weight};
    } else if (storage == 1) {
        weights = SplitAttentionProjectionPayload{qk.weight, gv.weight};
    } else {
        weights = NativeAttentionProjectionPayload{qg.weight, k.weight, v.weight, d};
    }
    family::bind_lora_gated_attention(store, 0, weights, g);
    store.ensure_banks();
    store.set_active(true);
    std::vector<float> b(2 * q);
    for (int i = 0; i < 2 * q; ++i) {
        b[i] = float(i + 1) / 128;
    }
    adapter(store, 0, "self_attn.q_proj", 0, h, b);
    for (auto& value : b) {
        value *= -0.5F;
    }
    adapter(store, 0, "self_attn.q_proj", 1, h, b);
    auto dx = test::to_device_bf16(std::vector<float>(h * tokens, 1));
    DeviceBuffer dq(q * tokens * 2), dg(q * tokens * 2), dk(kv * tokens * 2), dv(kv * tokens * 2);
    auto ids = test::to_device_i32(std::vector<int>{-1, 0, 1, 0});
    Tensor x(dx.p, DType::BF16, {h, tokens}), query(dq.p, DType::BF16, {q, tokens});
    Tensor gate(dg.p, DType::BF16, {q, tokens}), key(dk.p, DType::BF16, {kv, tokens});
    Tensor value(dv.p, DType::BF16, {kv, tokens}), slots(ids.p, DType::I32, {tokens});
    ops::lora_set_round({.slots = &slots, .scratch = store.scratch(tokens)});
    DeviceArena arena(1 << 20);
    WorkspaceArena workspace(DeviceSpan{arena.base(), arena.capacity()});
    auto run = [&](cudaStream_t stream) {
        Variant::attention_projection(x,
                                      weights,
                                      query,
                                      gate,
                                      key,
                                      value,
                                      family::TextPhase::Prefill,
                                      workspace,
                                      stream);
    };
    run(nullptr);
    test::cuda_synchronize();
    cudaStream_t stream;
    test::cuda_check(cudaStreamCreate(&stream), "stream");
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    test::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
    run(stream);
    test::cuda_check(cudaStreamEndCapture(stream, &graph), "capture end");
    test::cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
    const std::vector<int> selection{1, 0, -1, 1};
    ids.copy_from_host(selection.data(), selection.size() * sizeof(int));
    test::cuda_check(cudaGraphLaunch(executable, stream), "replay");
    test::cuda_synchronize(stream);
    std::vector<double> query_expected(q * tokens), gate_expected(q * tokens);
    for (int t = 0; t < tokens; ++t) {
        const double scale = selection[t] == -1 ? 0 : selection[t] == 0 ? 1 : -0.5;
        for (int i = 0; i < q; ++i) {
            const int source = (i / d) * 2 * d + i % d;
            query_expected[t * q + i] = scale * (source + 1) / 128;
            gate_expected[t * q + i] = scale * (source + d + 1) / 128;
        }
    }
    close(test::from_device_bf16(dq, q * tokens), query_expected);
    close(test::from_device_bf16(dg, q * tokens), gate_expected);
    test::cuda_check(cudaGraphExecDestroy(executable), "destroy graph");
    test::cuda_check(cudaGraphDestroy(graph), "destroy graph source");
    test::cuda_check(cudaStreamDestroy(stream), "destroy stream");
}

struct PinnedWeight {
    void* host = nullptr;
    Weight weight;
    explicit PinnedWeight(const Weight& source) : weight(source) {
        const auto codes = static_cast<std::size_t>(source.n) * source.k;
        const auto bytes = codes + codes / 32 * 2;
        test::cuda_check(cudaHostAlloc(&host, bytes, cudaHostAllocMapped), "pinned expert bank");
        std::memset(host, 0, bytes);
        void* device = nullptr;
        test::cuda_check(cudaHostGetDevicePointer(&device, host, 0), "bank alias");
        weight.payload = weight.qdata = device;
        weight.scales = static_cast<std::byte*>(device) + codes;
        weight.payload_bytes = bytes;
    }
    ~PinnedWeight() { if (host) { cudaFreeHost(host); } }
};

static void expert_adapters(bool shared, int tokens = 3, bool cpu = false,
                            int cache_slots = 0, bool expect_host_partial = true,
                            bool auto_cpu = false) {
    Context context;
    const auto g = shared ? ops::kSparseMoeQwen36Geometry : ops::kSparseMoeQwen3MoeGeometry;
    const int h = g.hidden, m = g.intermediate, e = g.experts;
    ZeroWeight router(g.router_rows(), h), gu(e * 2 * m, h, true), down(e * h, m, true);
    ZeroWeight sgu(shared ? 2 * m : 1, h, true), sd(h, m, true);
    ops::SparseMoeWeights weights;
    weights.router_shared_gate = router.weight;
    weights.routed_gate_up = gu.weight;
    weights.routed_down = down.weight;
    weights.experts_per_token = g.experts_per_token;
    if (shared) {
        weights.shared_gate_up = sgu.weight;
        weights.shared_down = sd.weight;
    }
    auto& store = ops::lora_store_for_current_device();
    store.configure(2, 4, tokens);
    family::bind_lora_moe(store, 0, weights);
    store.ensure_banks();
    store.set_active(true);
    for (int slot = 0; slot < 2; ++slot) {
        std::vector<float> bias(e, 0);
        bias[slot] = 2 + slot;
        adapter(store, 0, "mlp.gate", slot, h, bias);
        const auto prefix = "mlp.experts." + std::to_string(slot) + ".";
        adapter(store, 0, prefix + "gate_proj", slot, h, std::vector<float>(m, 0.5F + slot * 0.25F));
        adapter(store, 0, prefix + "up_proj", slot, h, std::vector<float>(m, 0.25F + slot * 0.25F));
        adapter(store, 0, prefix + "down_proj", slot, m, std::vector<float>(h, 0.125F + slot * 0.125F));
        if (shared) {
            adapter(store, 0, "mlp.shared_expert_gate", slot, h, {float(-slot)});
            adapter(store, 0, "mlp.shared_expert.gate_proj", slot, h, std::vector<float>(m, 0.25F));
            adapter(store, 0, "mlp.shared_expert.up_proj", slot, h, std::vector<float>(m, 0.5F));
            adapter(store, 0, "mlp.shared_expert.down_proj", slot, m, std::vector<float>(h, 0.25F));
        }
    }
    std::vector<int> selected(tokens);
    for (int t = 0; t < tokens; ++t) { selected[t] = t % 3 - 1; }
    auto dx = test::to_device_bf16(std::vector<float>(h * tokens, 1));
    auto output = test::to_device_bf16(std::vector<float>(h * tokens, 0.125F));
    auto ids = test::to_device_i32(selected);
    Tensor x(dx.p, DType::BF16, {h, tokens}), result(output.p, DType::BF16, {h, tokens});
    Tensor slots(ids.p, DType::I32, {tokens});
    ops::lora_set_round({.slots = &slots, .scratch = store.scratch(tokens)});
    DeviceArena arena(
        ops::sparse_moe_workspace_capacity_bytes(g, QType::W8G32_F16S, QType::W8G32_F16S, tokens, tokens));
    WorkspaceArena workspace(DeviceSpan{arena.base(), arena.capacity()});
    std::unique_ptr<PinnedWeight> host_gu, host_down;
    family::ExpertCache* cache = nullptr;
    family::BankedMixture mixture;
    if (cpu) {
        setenv("SUROGATE_CPU_EXPERT_THREADS", "4", 1);
        host_gu = std::make_unique<PinnedWeight>(weights.routed_gate_up);
        host_down = std::make_unique<PinnedWeight>(weights.routed_down);
        weights.routed_gate_up = host_gu->weight;
        weights.routed_down = host_down->weight;
        EngineOptions options;
        options.expert_slots = cache_slots > 0 ? cache_slots : e;
        options.cpu_moe_share = auto_cpu ? -1.0F : tokens > 3 ? .5F : 1.0F;
        options.cpu_moe_prefill_share = options.cpu_moe_share;
        options.cpu_moe_min_tokens = 1;
        options.prefill_chunk = tokens;
        family::ExpertCache::configure(options, 0, e);
        cache = &family::ExpertCache::for_current_device(g, 1);
        mixture.layer = 0; mixture.layers = 1; mixture.op = &weights;
        mixture.host_gate_up = static_cast<const std::byte*>(host_gu->host);
        mixture.host_down = static_cast<const std::byte*>(host_down->host);
        if (auto_cpu) { cache->prepare_split(mixture); }
    }
    auto run = [&](cudaStream_t stream) {
        if (cache) {
            cache->run(mixture, x, result, workspace, stream);
            assert(cache->has_pending_partial() == expect_host_partial);
            if (cache->has_pending_partial()) { cache->add_pending_partial(result, stream); }
        } else {
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, result, workspace, stream);
        }
    };
    const auto expected = [&] {
        std::vector<double> out(h * tokens, 0.125);
        for (int token = 0; token < tokens; ++token) {
            const int slot = selected[token];
            if (slot < 0) { continue; }
            const auto silu = [](double v) {
                return v / (1 + std::exp(-v));
            };
            const double alpha = std::exp(2 + slot) / (std::exp(2 + slot) + g.experts_per_token - 1);
            double delta = alpha * silu(0.5 + 0.25 * slot) * (0.25 + 0.25 * slot) * (0.125 + 0.125 * slot);
            if (shared) {
                delta += silu(0.25) * 0.5 * 0.25 / (1 + std::exp(slot));
            }
            for (int i = 0; i < h; ++i) {
                out[token * h + i] += delta;
            }
        }
        return out;
    }();
    run(nullptr);
    test::cuda_synchronize();
    close(test::from_device_bf16(output, h * tokens), expected);
    auto reset = test::to_device_bf16(std::vector<float>(h * tokens, 0.125F));
    test::cuda_check(cudaMemcpy(output.p, reset.p, output.bytes, cudaMemcpyDeviceToDevice), "reset");
    cudaStream_t stream;
    test::cuda_check(cudaStreamCreate(&stream), "stream");
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    test::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
    run(stream);
    test::cuda_check(cudaStreamEndCapture(stream, &graph), "capture end");
    test::cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
    test::cuda_check(cudaGraphLaunch(executable, stream), "replay");
    test::cuda_synchronize(stream);
    close(test::from_device_bf16(output, h * tokens), expected);
    if (tokens == 129 && !cpu && std::getenv("SUROGATE_LORA_BENCH")) {
        const auto measure = [&](auto&& body) {
            cudaEvent_t begin, end;
            test::cuda_check(cudaEventCreate(&begin), "timer");
            test::cuda_check(cudaEventCreate(&end), "timer");
            body(); test::cuda_synchronize();
            test::cuda_check(cudaEventRecord(begin, stream), "timer start");
            for (int i = 0; i < 3; ++i) { body(); }
            test::cuda_check(cudaEventRecord(end, stream), "timer stop");
            test::cuda_check(cudaEventSynchronize(end), "timer wait");
            float ms = 0; test::cuda_check(cudaEventElapsedTime(&ms, begin, end), "elapsed");
            cudaEventDestroy(begin); cudaEventDestroy(end);
            return ms / 3;
        };
        DeviceArena serial_arena(ops::detail::sparse_moe_decode_workspace_bytes(g));
        WorkspaceArena serial_workspace(DeviceSpan{serial_arena.base(), serial_arena.capacity()});
        const auto views = ops::detail::allocate_sparse_moe_decode_workspace(serial_workspace, g);
        const auto* banks = store.bank_table(weights.router_shared_gate.qdata);
        const float serial_ms = measure([&] {
            for (int t = 0; t < tokens; ++t) {
                auto input = x.slice(1, t, 1), output = result.slice(1, t, 1);
                ops::detail::sparse_moe_decode_launch(g, input, input, weights, output, views, stream,
                    nullptr, banks, static_cast<const std::int32_t*>(slots.data) + t);
            }
        });
        const float batch_ms = measure([&] { run(stream); });
        std::cout << "129-token expert prefill: serial " << serial_ms << " ms, batched " << batch_ms
                  << " ms (" << serial_ms / batch_ms << "x)" << std::endl;
    }
    test::cuda_check(cudaGraphExecDestroy(executable), "destroy graph");
    test::cuda_check(cudaGraphDestroy(graph), "destroy graph source");
    test::cuda_check(cudaStreamDestroy(stream), "destroy stream");
}

static void gdn_adapters() {
    Context context;
    constexpr int h = 64, k = 128, v = 128, c = 2 * k + v, heads = 8, width = 2, batch = 2, tokens = width * batch;
    using namespace targets::qwen3_5::detail;
    family::TextGeometry geometry;
    geometry.hidden = h;
    geometry.gdn_key_heads = heads;
    geometry.gdn_key_head_dim = k / heads;
    geometry.gdn_value_heads = heads;
    geometry.gdn_value_head_dim = v / heads;
    ZeroWeight input(c + v, h), control(2 * heads, h), out(h, v);
    auto zeros = test::to_device_f32(std::vector<float>(heads, 0));
    auto dt = test::to_device_f32(std::vector<float>(heads, 0));
    RuntimeModelView::GdnLayer linear;
    linear.output = out.weight;
    auto& weights = linear.projection;
    weights.input_projection = FusedGdnInputProjectionPayload{input.weight};
    weights.control_projection = FusedGdnControlProjectionPayload{control.weight};
    weights.a_log = Tensor(zeros.p, DType::FP32, {heads});
    weights.dt_bias = Tensor(dt.p, DType::FP32, {heads});
    auto& store = ops::lora_store_for_current_device();
    store.configure(2, 4, tokens);
    family::bind_lora_gdn(store, 0, linear, geometry);
    store.ensure_banks();
    store.set_active(true);
    for (int slot = 0; slot < 2; ++slot) {
        adapter(store, 0, "linear_attn.in_proj_qkv", slot, h, std::vector<float>(c, 0.5F + slot * 0.25F));
        adapter(store, 0, "linear_attn.in_proj_z", slot, h, std::vector<float>(v, 0.25F + slot * 0.25F));
        adapter(store, 0, "linear_attn.in_proj_a", slot, h, std::vector<float>(heads, 0.5F + slot * 0.25F));
        adapter(store, 0, "linear_attn.in_proj_b", slot, h, std::vector<float>(heads, 0.25F + slot * 0.25F));
    }
    store.set_module_slot(0, "linear_attn.dt_bias.bias", 1, {0}, std::vector<std::uint16_t>(heads),
                          1, 1, heads, 1, {}, std::vector<float>(heads, .25F));
    const std::vector<int> selection{-1, 0, 1, -1};
    auto ids = test::to_device_i32(selection);
    Tensor slots(ids.p, DType::I32, {tokens});
    ops::lora_set_round({.slots = &slots, .scratch = store.scratch(tokens)});
    auto dx = test::to_device_bf16(std::vector<float>(h * tokens, 1));
    auto norm_data = test::to_device_bf16(std::vector<float>(h, 0));
    DeviceBuffer dh(h * tokens * 2), dg(heads * tokens * 4), db(heads * tokens * 4);
    Tensor residual(dx.p, DType::BF16, {h, tokens}), hidden(dh.p, DType::BF16, {h, tokens});
    Tensor norm(norm_data.p, DType::BF16, {h}), decay(dg.p, DType::FP32, {heads, tokens});
    Tensor beta(db.p, DType::FP32, {heads, tokens});
    DeviceArena arena(1 << 20);
    WorkspaceArena workspace(DeviceSpan{arena.base(), arena.capacity()});
    Variant::gdn_norm_control_projection(residual, norm, 1e-6F, weights, hidden, decay, beta, workspace, nullptr);
    test::cuda_synchronize();
    std::vector<double> expected_g(heads * tokens), expected_b(heads * tokens);
    for (int t = 0; t < tokens; ++t) {
        const double a = selection[t] < 0 ? 0 : 0.5 + 0.25 * selection[t] + (selection[t] == 1 ? .25 : 0);
        const double b = selection[t] < 0 ? 0 : 0.25 + 0.25 * selection[t];
        for (int i = 0; i < heads; ++i) {
            expected_g[t * heads + i] = -std::log1p(std::exp(a));
            expected_b[t * heads + i] = 1 / (1 + std::exp(-b));
        }
    }
    close(test::from_device_f32(dg, heads * tokens), expected_g);
    close(test::from_device_f32(db, heads * tokens), expected_b);
    std::vector<float> taps(4 * c, 0);
    std::fill(taps.begin() + 3 * c, taps.end(), 1);
    auto conv_data = test::to_device_bf16(taps);
    auto state_data = test::to_device_bf16(std::vector<float>(c * 3 * 6, 0));
    auto valid_data = test::to_device_i32({width, width});
    auto initial_data = test::to_device_i32({0, 1});
    auto snapshot_data = test::to_device_i32({2, 4});
    Tensor conv(conv_data.p, DType::BF16, {c, 4}), states(state_data.p, DType::BF16, {c, 3, 6});
    Tensor valid(valid_data.p, DType::I32, {batch}), initial(initial_data.p, DType::I32, {batch});
    Tensor snapshot(snapshot_data.p, DType::I32, {batch});
    DeviceBuffer dq(k * tokens * 2), dk(k * tokens * 2), dv(v * tokens * 2), dz(v * tokens * 2), dr(c * tokens * 2);
    Tensor q(dq.p, DType::BF16, {k, width, batch}), key(dk.p, DType::BF16, {k, width, batch});
    Tensor value(dv.p, DType::BF16, {v, width, batch}), z(dz.p, DType::BF16, {v, width, batch});
    Tensor record(dr.p, DType::BF16, {c, width, batch}), shaped(dx.p, DType::BF16, {h, width, batch});
    std::vector<double> expected(k * tokens), expected_z(v * tokens);
    for (int t = 0; t < tokens; ++t) {
        const double x = selection[t] < 0 ? 0 : 0.5 + 0.25 * selection[t];
        for (int i = 0; i < k; ++i) {
            expected[t * k + i] = x / (1 + std::exp(-x));
        }
        for (int i = 0; i < v; ++i) {
            expected_z[t * v + i] = selection[t] < 0 ? 0 : 0.25 + 0.25 * selection[t];
        }
    }
    Variant::gdn_input_projection_snapshot(shaped,
                                           weights,
                                           conv,
                                           states,
                                           valid,
                                           initial,
                                           snapshot,
                                           q,
                                           key,
                                           value,
                                           z,
                                           family::TextPhase::Verify,
                                           workspace,
                                           nullptr);
    test::cuda_synchronize();
    close(test::from_device_bf16(dq, k * tokens), expected);
    close(test::from_device_bf16(dk, k * tokens), expected);
    close(test::from_device_bf16(dv, v * tokens), expected);
    close(test::from_device_bf16(dz, v * tokens), expected_z);
    Variant::gdn_input_projection_record(shaped,
                                         weights,
                                         conv,
                                         states,
                                         valid,
                                         initial,
                                         record,
                                         q,
                                         key,
                                         value,
                                         z,
                                         family::TextPhase::Verify,
                                         workspace,
                                         nullptr);
    test::cuda_synchronize();
    close(test::from_device_bf16(dq, k * tokens), expected);
    close(test::from_device_bf16(dz, v * tokens), expected_z);
}

int main() {
    if (test::cuda_unavailable()) {
        return 77;
    }
    for (int storage = 0; storage < 3; ++storage) {
        gated_attention(storage);
    }
    gdn_adapters();
    expert_adapters(false);
    expert_adapters(true);
    expert_adapters(false, 129);
    expert_adapters(false, 3, true);
    expert_adapters(false, 33, true);
    // Two token columns fit at once. Distinct per-column adapters must retain their
    // original IDs through cache slices, CPU partials and CUDA graph replay.
    expert_adapters(false, 3, true, 17);
    expert_adapters(true, 7, true, 17);
    expert_adapters(false, 33, true, 17, false);
    expert_adapters(false, 129, true, 17, false);
    expert_adapters(false, 3, true, 8, true, true);
    std::cout << "LoRA split/gated attention and routed/shared expert numerical coverage passed\n";
}
