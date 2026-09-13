#include "family/impl/runtime/cache_layers.h"
#include "core/arena.h"
#include "core/device.h"

#include <array>
#include <cassert>
#include <iostream>

using namespace sinfer;

namespace {
template <class F> void rejects(F&& action) {
    bool refused = false;
    try { action(); } catch (const std::logic_error&) { refused = true; }
    assert(refused);
}

void check_placement() {
    family::TextGeometry glm;
    glm.layers = 45;
    for (int layer = 3; layer < 44; layer += 4) { glm.declare_attention_layer(layer); }
    family::DecoderStateSpec spec{
        .full_attention_layers = 11, .capacity = 1048576,
        .kv_heads = 1, .attention_head_dim = 512, .kv_dtype = DType::FP8_E4M3FN,
        .indexer_head_dim = 384, .indexer_dtype = DType::FP32,
        .text_physical_page_groups = 16384};
    LayoutBuilder whole_builder;
    const auto whole = family::plan_decoder_state(whole_builder, spec);
    assert(whole.kv_payload_bytes() == 28160ULL * spec.capacity);
    const std::array bounds{0, 8, 13, 19, 24, 30, 35, 40, 45};
    const std::array owners{2, 1, 1, 2, 1, 1, 2, 1};
    std::size_t sum = 0;
    unsigned next_owner = 0;
    for (unsigned stage = 0; stage < owners.size(); ++stage) {
        spec.text_kv_layers = family::detail::stage_kv_layers(glm, bounds[stage], bounds[stage + 1]);
        LayoutBuilder builder;
        const auto local = family::plan_decoder_state(builder, spec);
        assert(local.text_kv.first_layer == next_owner);
        assert(local.text_kv.layers == owners[stage]);
        assert(local.text_kv.pool.planes.size() == 3 * owners[stage]);
        assert(local.kv_payload_bytes() == 2560ULL * spec.capacity * owners[stage]);
        next_owner += owners[stage];
        sum += local.kv_payload_bytes();
    }
    assert(sum == whole.kv_payload_bytes());
    const auto all = family::detail::stage_kv_layers(glm, 0, 0);
    assert(all.first == 0 && all.last == 11);
    const auto empty = family::detail::stage_kv_layers(glm, 8, 11);
    assert(empty.first == 2 && empty.last == 2);
    rejects([&] { (void)family::detail::stage_kv_layers(glm, 10, 9); });

    // Shared consumers are not separate owners. The partition keeps these final four
    // layers together, so their two owners are stored only on that stage.
    family::TextGeometry shared;
    shared.layers = 6;
    shared.declare_kv_sharing();
    for (int layer = 0; layer < 6; ++layer) {
        shared.declare_attention_layer(layer);
        if (layer < 4) { shared.declare_kv_owner(layer); }
    }
    const auto range = family::detail::stage_kv_layers(shared, 2, 6);
    assert(range.first == 2 && range.last == 4);
}

void check_views(DType dtype, bool elastic, bool empty) {
    DeviceContext device;
    family::DecoderStateSpec spec{
        .full_attention_layers = 5,
        .text_kv_layers = family::KvLayerRange{2, empty ? 2U : 4U},
        .mtp_layers = 1, .capacity = 128, .kv_heads = 2, .attention_head_dim = 64,
        .global_kv_heads = 1, .global_attention_head_dim = 128,
        .global_geometry_layers = {1, 3},
        .kv_dtype = dtype, .kv_quant_group = dtype == DType::I8 ? 64 : 0,
        .indexer_head_dim = 16, .indexer_dtype = DType::FP32, .mtp_indexer = true,
        .kv_skip_layers = dtype == DType::FP8_E4M3FN ? std::vector<unsigned>{0, 3, 4}
                                                  : std::vector<unsigned>{},
        .enable_mtp = true, .elastic_kv = elastic, .kv_table_rows = 2,
        .text_physical_page_groups = 4, .text_physical_page_cap = 2,
        .mtp_physical_page_groups = 4};
    LayoutBuilder builder;
    const auto layout = family::plan_decoder_state(builder, spec);
    DeviceArena arena(builder.finish(256));
    const PagedKVElasticOptions options{.device = 0, .fence_stream = device.stream,
                                        .reserve_granules = 0};
    family::DecoderState state({arena.base(), arena.capacity()}, layout, &options);
    auto& cache = state.text_kv;
    auto allocation = cache.pool().reserve(2);
    allocation.bind_row(1, device.stream);
    allocation.materialize_pages(2, device.stream);
    const auto single = cache.execution_view(allocation);
    rejects([&] { (void)single.layer_view(1); });
    rejects([&] { (void)cache.batch_layer_view(4); });
    if (empty) {
        assert(layout.text_kv.payload_bytes() == 0);
        assert(cache.pool().plane_count() == 0);
        assert(cache.pool().elastic_region() == nullptr);
        assert(cache.pool().occupancy().page_bytes == 0);
        rejects([&] { (void)cache.batch_layer_view(2); });
    } else {
        const auto first = single.layer_view(2), second = single.layer_view(3);
        const auto batch_first = cache.batch_layer_view(2), batch_second = cache.batch_layer_view(3);
        assert(first.k_pages.data == batch_first.k_pages.data);
        assert(second.k_pages.data == batch_second.k_pages.data);
        assert(first.k_pages.data != second.k_pages.data);
        assert(first.k_pages.data == cache.pool().plane(0).data);
        assert(first.head_dim == 64 && first.num_kv_heads == 2);
        assert(second.head_dim == 128 && second.num_kv_heads == 1);
        assert(first.dtype == dtype);
        assert(second.dtype == (dtype == DType::FP8_E4M3FN ? DType::BF16 : dtype));
        assert(first.indexer_pages.dtype == DType::FP32);
        assert(second.indexer_pages.data == batch_second.indexer_pages.data);
        if (dtype == DType::I8) {
            assert(first.k_scale_pages.data == batch_first.k_scale_pages.data);
            assert(second.k_scale_pages.ne[0] == 2);
        }
    }
    // A pipeline's main-stack range must not change the separate draft cache's IDs.
    assert(state.mtp_kv->layers() == 1);
    assert(state.mtp_kv->batch_layer_view(0).indexer_pages.dtype == DType::FP32);
    cache.pool().zero_pages(allocation.page_ids(), device.stream);
    allocation.trim_pages(1, device.stream);
    allocation.materialize_pages(2, device.stream);
    device.synchronize();
    allocation.release();
    assert(cache.pool().entitled_pages() == 0 && cache.pool().mapped_pages() == 0);
    spec.text_kv_layers = family::KvLayerRange{4, 6};
    LayoutBuilder invalid;
    rejects([&] { (void)family::plan_decoder_state(invalid, spec); });
}
} // namespace

int main() {
    check_placement();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    for (auto dtype : {DType::BF16, DType::FP8_E4M3FN, DType::I8}) {
        for (bool elastic : {false, true}) {
            check_views(dtype, elastic, false);
            check_views(dtype, elastic, true);
        }
    }
    std::cout << "Pipeline KV ownership, precision, indexers, empty stages and admission passed\n";
}
