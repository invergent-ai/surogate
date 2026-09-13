#include "api/ops/expert_slot_cache.h"
#include "api/ops/sparse_moe.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"
#include "core/engine_context.h"
#include "family/impl/moe/expert_cache.h"

#include <iostream>
#include <memory>

using namespace sinfer;
using namespace sinfer::test;

namespace {

int compare(const char* label, const std::vector<std::uint16_t>& actual,
            const std::vector<std::uint16_t>& expected, int allowed_ulp) {
    if (actual.size() != expected.size()) { return 1; }
    int max_ulp = 0;
    float max_absolute = 0;
    const auto ordered = [](std::uint16_t value) {
        return value & 0x8000 ? 0x8000 - int(value & 0x7fff) : 0x8000 + int(value);
    };
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(bf16_to_f32(actual[i]))) {
            std::cerr << label << ": nonfinite output\n";
            return 1;
        }
        max_ulp = std::max(max_ulp, std::abs(ordered(actual[i]) - ordered(expected[i])));
        max_absolute = std::max(max_absolute, std::abs(bf16_to_f32(actual[i]) - bf16_to_f32(expected[i])));
    }
    if (max_ulp > allowed_ulp) {
        std::cerr << label << ": max BF16 ULP=" << max_ulp << " max absolute=" << max_absolute << '\n';
        return 1;
    }
    return 0;
}

struct Context {
    ops::EngineOpsContext context;
    ops::EngineOpsContext* previous = const_cast<ops::EngineOpsContext*>(
        static_cast<const ops::EngineOpsContext*>(ops::current_ops_owner()));
    Context() { ops::bind_ops_context(&context); }
    ~Context() { ops::bind_ops_context(previous); }
};

struct DeviceWeight {
    DeviceBuffer storage;
    Weight weight;
    std::vector<std::uint8_t> host;
    explicit DeviceWeight(quantized_weight::PackedWeight packed, bool retain_host = false)
        : storage(to_device(packed.payload)), weight(packed.device_weight(storage.p)) {
        if (retain_host) { host = std::move(packed.payload); }
    }
};

struct Cache {
    int layer = 0;
    int experts;
    GuardedDeviceBuffer pool_memory, directory_memory, miss_memory, active_memory;
    ops::ExpertSlotPool pool;
    ops::ExpertSlotDirectory directory;
    ops::ExpertMissList misses;
    ops::ExpertHostBank bank;
    Tensor active;

    Cache(const ops::SparseMoeGeometry& g, int slots, const ops::SparseMoeWeights& w)
        : experts(g.experts), pool_memory(ops::expert_slot_pool_bytes(g, slots)),
          directory_memory(ops::expert_slot_directory_bytes(2, g.experts, slots)),
          miss_memory(ops::expert_miss_list_bytes(g.experts)), active_memory(g.experts * sizeof(int)),
          pool(ops::create_expert_slot_pool(g, slots, pool_memory.data())),
          directory(ops::create_expert_slot_directory(2, g.experts, slots, 0,
                                                     directory_memory.data(), nullptr)),
          misses(ops::create_expert_miss_list(g.experts, miss_memory.data())),
          bank(ops::expert_host_bank(g, w.routed_gate_up, w.routed_down)),
          active(active_memory.data(), DType::I32, {g.experts}) {}

    static void resolve(void* context, const Tensor& ids, const Tensor&, const Tensor&,
                        Tensor&, cudaStream_t stream) {
        auto& c = *static_cast<Cache*>(context);
        ops::expert_slot_resolve(ids, c.layer, c.directory, c.misses, stream);
        ops::expert_slot_gather(c.bank, c.misses, c.pool, stream);
        const auto table = c.directory.slot_of_expert.slice(0, c.layer * c.experts, c.experts);
        cuda_check(cudaMemcpyAsync(c.active.data, table.data, c.active.bytes(),
                                    cudaMemcpyDeviceToDevice, stream), "active slot table");
    }

    static void range(void* context, const Tensor& ids, int first, int count, cudaStream_t stream) {
        auto& c = *static_cast<Cache*>(context);
        ops::expert_slot_resolve_range(ids, c.layer, c.directory, c.misses,
                                       c.active, first, count, stream);
        ops::expert_slot_gather(c.bank, c.misses, c.pool, stream);
    }
};

int check_cpu_rounds(family::ExpertCache& cache, family::BankedMixture mixture,
                     const ops::SparseMoeWeights& weights, cudaStream_t stream);

int check(const ops::SparseMoeGeometry& g, bool cpu_split = false) {
    DeviceWeight gate(quantized_weight::make_patterned_weight(
        QType::W8G32_F16S, g.routed_gate_rows(), g.hidden, 291), cpu_split);
    DeviceWeight down(quantized_weight::make_patterned_weight(
        QType::W8G32_F16S, g.routed_down_rows(), g.intermediate, 311), cpu_split);
    std::unique_ptr<DeviceWeight> shared_gate, shared_down;
    if (g.has_shared()) {
        shared_gate = std::make_unique<DeviceWeight>(quantized_weight::make_patterned_weight(
            QType::W8G32_F16S, g.shared_rows(), g.hidden, 313));
        shared_down = std::make_unique<DeviceWeight>(quantized_weight::make_patterned_weight(
            QType::W8G32_F16S, g.hidden, g.shared_intermediate, 317));
    }
    std::vector<float> routing(static_cast<std::size_t>(g.router_rows()) * g.hidden);
    fill_uniform(routing, 331, -0.125F, 0.125F);
    DeviceBuffer router = to_device_bf16(routing);
    Weight rw{};
    rw.qtype = QType::BF16_CTRL;
    rw.layout = QuantLayout::Contiguous;
    rw.payload = rw.qdata = router.p;
    rw.payload_bytes = router.bytes;
    rw.n = rw.shape[0] = rw.padded_shape[0] = g.router_rows();
    rw.k = rw.shape[1] = rw.padded_shape[1] = g.hidden;
    rw.ndim = 2;
    std::vector<float> bias_values(g.experts, 0.0F);
    DeviceBuffer bias = to_device_f32(bias_values);
    std::vector<float> scale_values(g.experts);
    for (int expert = 0; expert < g.experts; ++expert) {
        scale_values[expert] = 0.5F + float(expert % 7) * 0.125F;
    }
    DeviceBuffer expert_scales = to_device_f32(scale_values);
    ops::SparseMoeWeights weights{
        .router_shared_gate = rw,
        .router_bias = g.gating == ops::SparseMoeGating::SigmoidBiasTopK
            ? static_cast<const float*>(bias.p) : nullptr,
        .routed_scale = g.routed_scale,
        .shared_gated = g.shared_gated,
        .swiglu_limit = g.swiglu_limit,
        .activation = g.activation,
        .per_expert_scale = g.per_expert_scaled ? static_cast<const float*>(expert_scales.p) : nullptr,
        .routed_gate_up = gate.weight,
        .routed_down = down.weight,
        .shared_gate_up = shared_gate ? shared_gate->weight : Weight{},
        .shared_down = shared_down ? shared_down->weight : Weight{},
        .experts_per_token = g.experts_per_token,
    };
    cudaStream_t stream{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    int failures = 0;
    for (int slots : {g.experts_per_token, g.experts_per_token * 2 + 1, 64}) {
        if (cpu_split && slots != 64) { continue; }
        Context context;
        EngineOptions options;
        options.expert_slots = slots;
        options.cpu_moe_share = cpu_split ? 1.0F : 0.0F;
        options.cpu_moe_min_tokens = 1;
        options.cpu_moe_prefill_share = 0;
        family::ExpertCache::configure(options, 0, g.experts);
        auto& family_cache = family::ExpertCache::for_current_device(g, 2);
        family::BankedMixture mixture{.layer = 0, .layers = 2, .op = &weights};
        if (cpu_split) {
            mixture.host_gate_up = reinterpret_cast<const std::byte*>(gate.host.data());
            mixture.host_down = reinterpret_cast<const std::byte*>(down.host.data());
            failures += check_cpu_rounds(family_cache, mixture, weights, stream);
            continue;
        }
        Cache cache(g, slots, weights);
        auto pooled = ops::expert_slot_weights(cache.pool, cache.directory, 0, weights);
        pooled.slot_of_expert = static_cast<const int*>(cache.active.data);
        ops::SparseMoeRoundHook hook{&Cache::resolve, &cache,
                                     slots < g.experts ? slots / g.experts_per_token : 0,
                                     slots, &Cache::range};
        ops::SparseMoeRoundHook reference_hook{};
        // Compare against the original resident-layer schedules, including unsliced
        // narrow rounds, rather than giving the reference the bounded cache's schedule.
        for (int tokens : {1, 2, 3, 7, 19, 20, 33, 129, 768}) {
            if (tokens == 768 && slots != 64) { continue; }
            const auto elements = static_cast<std::size_t>(g.hidden) * tokens;
            std::vector<float> values(elements);
            fill_uniform(values, 337 + tokens, -0.125F, 0.125F);
            DeviceBuffer input = to_device_bf16(values);
            fill_uniform(values, 347 + tokens, -0.25F, 0.25F);
            DeviceBuffer router_input = to_device_bf16(values);
            std::vector<std::uint16_t> initial(elements, f32_to_bf16(0.25F));
            DeviceBuffer expected = to_device(initial);
            GuardedDeviceBuffer output(elements * 2);
            Tensor x(input.p, DType::BF16, {g.hidden, tokens});
            Tensor rx(router_input.p, DType::BF16, {g.hidden, tokens});
            Tensor reference(expected.p, DType::BF16, {g.hidden, tokens});
            Tensor actual(output.data(), DType::BF16, {g.hidden, tokens});
            WorkspaceArena workspace(ops::sparse_moe_workspace_capacity_bytes(
                g, QType::W8G32_F16S, QType::W8G32_F16S, 1, tokens));
            ops::sparse_moe(x, rx, weights, ops::SparseMoeEpilogue::AddResidual,
                            reference, workspace, stream, reference_hook);
            cuda_check(cudaStreamSynchronize(stream), "resident result");
            const auto want = from_device<std::uint16_t>(expected, elements);
            for (auto value : want) {
                if (!std::isfinite(bf16_to_f32(value))) {
                    throw std::runtime_error("nonfinite resident fixture output");
                }
            }
            output.copy_from_host(initial.data(), initial.size() * 2);
            ops::sparse_moe(x, rx, pooled, ops::SparseMoeEpilogue::AddResidual,
                            actual, workspace, stream, hook);
            cuda_check(cudaStreamSynchronize(stream), "bounded result");
            const auto bounded = from_device<std::uint16_t>(output.data(), elements);
            // Wide rounds preserve the resident reduction order exactly. Narrow slices
            // can switch between the existing scalar and tiled decode reductions; bound
            // that difference to one BF16 step. Graph replay must still match eager exactly.
            failures += compare("bounded experts eager", bounded, want, tokens < 20 ? 1 : 0);

            cudaGraph_t graph{};
            cudaGraphExec_t executable{};
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
            family_cache.run(mixture, x, actual, workspace, stream, &rx);
            cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
            cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
            for (int replay = 0; replay < 2; ++replay) {
                // Evict the recorded layer with another layer before replaying the graph.
                mixture.layer = 1;
                family_cache.run(mixture, x, actual, workspace, stream, &rx);
                cuda_check(cudaStreamSynchronize(stream), "eviction");
                mixture.layer = 0;
                output.copy_from_host(initial.data(), initial.size() * 2);
                cuda_check(cudaGraphLaunch(executable, stream), "replay");
                cuda_check(cudaStreamSynchronize(stream), "replay complete");
                failures += verify_exact("bounded experts graph after eviction",
                    from_device<std::uint16_t>(output.data(), elements), bounded);
            }
            cuda_check(cudaGraphExecDestroy(executable), "destroy executable");
            cuda_check(cudaGraphDestroy(graph), "destroy graph");
            failures += output.verify_guards("bounded output");
            std::cout << "bounded experts=" << g.experts << " slots=" << slots
                      << " tokens=" << tokens << " failures=" << failures << std::endl;
        }
        failures += cache.pool_memory.verify_guards("bounded pool");
        failures += cache.directory_memory.verify_guards("bounded directory");
        failures += cache.active_memory.verify_guards("bounded active map");
        failures += cache.miss_memory.verify_guards("bounded misses");
    }
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
    return failures;
}

int check_cpu_rounds(family::ExpertCache& cache, family::BankedMixture mixture,
                     const ops::SparseMoeWeights& weights, cudaStream_t stream) {
    const auto& g = ops::kSparseMoeQwen3MoeGeometry;
    constexpr int maximum = 19;
    const auto elements = static_cast<std::size_t>(g.hidden) * maximum;
    DeviceBuffer input(elements * 2), output(elements * 2), expected(elements * 2);
    WorkspaceArena workspace(ops::sparse_moe_workspace_capacity_bytes(
        g, QType::W8G32_F16S, QType::W8G32_F16S, 1, maximum));
    std::array<cudaGraphExec_t, maximum + 1> graphs{};
    int failures = 0;
    // 19 columns create 8+8+3 slices; the next 16-column round ends at a
    // context that was previously non-final. Change widths and inputs to expose
    // stale done flags, then repeat the same sequence under captured graphs.
    for (const bool captured : {false, true}) {
        for (int iteration = 0; iteration < 16; ++iteration) {
            const int tokens = std::array{19, 16, 8, 16}[iteration % 4];
            std::vector<float> values(elements);
            fill_uniform(values, 107 + iteration, -0.125F, 0.125F);
            std::vector<std::uint16_t> bits(elements);
            std::transform(values.begin(), values.end(), bits.begin(), f32_to_bf16);
            CUDA_CHECK(cudaMemcpyAsync(input.p, bits.data(), input.bytes, cudaMemcpyHostToDevice, stream));
            Tensor x(input.p, DType::BF16, {g.hidden, tokens});
            Tensor actual(output.p, DType::BF16, {g.hidden, tokens});
            Tensor reference(expected.p, DType::BF16, {g.hidden, tokens});
            CUDA_CHECK(cudaMemsetAsync(reference.data, 0, reference.bytes(), stream));
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, reference, workspace, stream);
            CUDA_CHECK(cudaMemsetAsync(actual.data, 0, actual.bytes(), stream));
            if (captured) {
                if (!graphs[tokens]) {
                    cudaGraph_t graph{};
                    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                    cache.run(mixture, x, actual, workspace, stream);
                    if (!cache.has_pending_partial()) { throw std::runtime_error("CPU split did not engage"); }
                    cache.add_pending_partial(actual, stream);
                    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
                    CUDA_CHECK(cudaGraphInstantiate(&graphs[tokens], graph, nullptr, nullptr, 0));
                    CUDA_CHECK(cudaGraphDestroy(graph));
                }
                CUDA_CHECK(cudaGraphLaunch(graphs[tokens], stream));
            } else {
                cache.run(mixture, x, actual, workspace, stream);
                if (!cache.has_pending_partial()) { throw std::runtime_error("CPU split did not engage"); }
                cache.add_pending_partial(actual, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            // CPU intermediates are FP32 while the GPU reference rounds its
            // expert intermediates to BF16. Use the existing MoE A16 contract;
            // ULP distance is unsuitable for cancellation near zero.
            failures += verify_reduction(captured ? "CPU split graph slices" : "CPU split eager slices",
                                         from_device_bf16(actual.data, actual.numel()),
                                         from_device_bf16(reference.data, reference.numel()),
                                         {1.2e-2, 1.0e-3, 1.5e-2});
        }
    }
    for (auto graph : graphs) { if (graph) { CUDA_CHECK(cudaGraphExecDestroy(graph)); } }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) { return 77; }
    if (std::getenv("SUROGATE_TEST_CPU_SPLIT")) {
        return check(ops::kSparseMoeQwen3MoeGeometry, true) ? 1 : 0;
    }
    if (std::getenv("SUROGATE_TEST_BOUNDED_GLM")) {
        return check(ops::kSparseMoeGlm53Geometry) ? 1 : 0;
    }
    if (std::getenv("SUROGATE_TEST_BOUNDED_EXTRA")) {
        return check(ops::kSparseMoeFlashNextGeometry) + check(ops::kSparseMoeGemma4Geometry) +
               check(ops::kSparseMoeLfm2Moe32Geometry) + check(ops::kSparseMoeLfm2Moe64Geometry) +
               check(ops::kSparseMoeQwen3Moe235BGeometry) ? 1 : 0;
    }
    return check(ops::kSparseMoeQwen3MoeGeometry) + check(ops::kSparseMoeQwen36Geometry) ? 1 : 0;
}
