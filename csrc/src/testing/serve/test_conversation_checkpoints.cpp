#include "api/family/decoder_state.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/elastic_kv_region.h"
#include "core/sleep.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using namespace sinfer;

namespace {
void fill(const Tensor& tensor, int value, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(tensor.data, value, tensor.bytes(), stream));
}

void expect(const Tensor& tensor, unsigned char value) {
    std::vector<unsigned char> bytes(tensor.bytes());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(bytes.data(), tensor.data, bytes.size(), cudaMemcpyDeviceToHost));
    assert(std::all_of(bytes.begin(), bytes.end(), [=](auto byte) { return byte == value; }));
}

void exercise(DType cache_dtype, bool recurrent) {
    DeviceContext device;
    constexpr int lanes = 3, resident_slots = lanes + 1;
    LayoutBuilder builder;
    family::DecoderStateSpec spec{
        .full_attention_layers = 1, .capacity = 64, .kv_heads = 1, .attention_head_dim = 64,
        .kv_table_rows = lanes, .text_physical_page_groups = 4,
        .linear_attention = {.layers = 2, .conv_channels = 16, .conv_width = 3,
            .value_heads = recurrent ? 2 : 0, .value_head_dim = recurrent ? 8 : 0,
            .key_head_dim = recurrent ? 8 : 0, .slot_count = resident_slots},
        .ple = NgramPleStatePoolSpec{.history_tokens = 2, .conv_history = 3, .channels = 16,
                                     .slot_count = resident_slots}};
    const auto layout = family::plan_decoder_state(builder, spec);
    const auto cyclic_layout = plan_cyclic_kv_cache(builder, 2, 32, 2, 64, lanes, cache_dtype);
    DeviceArena arena(builder.finish(256));
    const DeviceSpan backing{arena.base(), arena.capacity()};
    family::DecoderState decoder(backing, layout);
    CyclicKVCache live(backing, cyclic_layout);

    LayoutBuilder snapshot_builder;
    auto linear = spec.linear_attention;
    linear.slot_count = 1;
    auto ple = *spec.ple;
    ple.slot_count = 1;
    family::DecoderCheckpointLayout snapshots;
    snapshots.linear_attention = plan_linear_attention_state_pool(snapshot_builder, linear);
    snapshots.ple = plan_ngram_ple_state_pool(snapshot_builder, ple);
    snapshots.dflash = plan_cyclic_kv_cache(snapshot_builder, 2, 32, 2, 64, 1, cache_dtype);
    snapshots.slot_bytes = snapshot_builder.finish(kElasticKvGranuleBytes);
    snapshots.lanes = lanes;
    snapshots.capacity = 1;
    const auto prior_commitment = elastic_kv_unmapped_commitment(0);
    decoder.configure_checkpoints(snapshots, {.device = 0, .fence_stream = device.stream});
    assert(decoder.checkpoint_mapped_bytes() == 0);
    assert(decoder.checkpoint_reservation_bytes() == snapshots.slot_bytes);
    assert(elastic_kv_unmapped_commitment(0) == prior_commitment + snapshots.slot_bytes);
    assert(decoder.linear_attention.slot_count() == resident_slots + lanes);
    assert(decoder.linear_attention.all_layers_view().spec.slot_count == resident_slots);

    assert(decoder.try_acquire_checkpoint(0));
    assert(decoder.try_acquire_checkpoint(0)); // Idempotent; no extra page or reference.
    assert(!decoder.try_acquire_checkpoint(1));
    assert(decoder.checkpoint_mapped_bytes() == snapshots.slot_bytes);
    assert(elastic_kv_unmapped_commitment(0) == prior_commitment);
    for (unsigned layer = 0; layer < 2; ++layer) {
        fill(decoder.linear_attention.conv_slot(layer, 0), 0x11 + layer, device.stream);
        if (recurrent) { fill(decoder.linear_attention.recurrent_slot(layer, 0), 0x21 + layer, device.stream); }
        fill(live.layer_view(layer).k, 0x31 + layer, device.stream);
        fill(live.layer_view(layer).v, 0x41 + layer, device.stream);
    }
    fill(decoder.ple.history_slot(0), 0x51, device.stream);
    fill(decoder.ple.conv_slot(0), 0x61, device.stream);
    decoder.copy_state_slot(0, resident_slots, device.stream);
    decoder.checkpoint_dflash(0).copy_lane_from(live, 0, 0, device.stream);
    decoder.reset_state_slot(0, device.stream);
    for (unsigned layer = 0; layer < 2; ++layer) {
        fill(live.layer_view(layer).k.slice(3, 0, 1), 0, device.stream);
        fill(live.layer_view(layer).v.slice(3, 0, 1), 0, device.stream);
    }
    device.synchronize();
    assert(sleep_device(0) >= snapshots.slot_bytes);
    assert(decoder.checkpoint_mapped_bytes() == 0);
    assert(wake_device(0) >= snapshots.slot_bytes);
    decoder.copy_state_slot(resident_slots, 0, device.stream);
    live.copy_lane_from(decoder.checkpoint_dflash(0), 0, 0, device.stream);
    for (unsigned layer = 0; layer < 2; ++layer) {
        expect(decoder.linear_attention.conv_slot(layer, 0), 0x11 + layer);
        if (recurrent) { expect(decoder.linear_attention.recurrent_slot(layer, 0), 0x21 + layer); }
        expect(live.layer_view(layer).k, 0x31 + layer);
        expect(live.layer_view(layer).v, 0x41 + layer);
    }
    expect(decoder.ple.history_slot(0), 0x51);
    expect(decoder.ple.conv_slot(0), 0x61);

    decoder.release_checkpoint(0);
    decoder.flush_checkpoint_releases();
    assert(!decoder.has_checkpoint(0));
    assert(decoder.checkpoint_mapped_bytes() == 0);
    bool refused = false;
    try { (void)decoder.linear_attention.conv_slot(0, resident_slots); }
    catch (const std::logic_error&) { refused = true; }
    assert(refused);
    assert(decoder.try_acquire_checkpoint(2));
    decoder.reset_state_slot(1, device.stream);
    fill(decoder.linear_attention.conv_slot(0, 1), 0x71, device.stream);
    decoder.copy_state_slot(1, resident_slots + 2, device.stream);
    decoder.copy_state_slot(resident_slots + 2, 0, device.stream);
    expect(decoder.linear_attention.conv_slot(0, 0), 0x71);
    assert(!decoder.try_acquire_checkpoint(0));
    decoder.release_checkpoint(2);
    decoder.flush_checkpoint_releases();
    assert(decoder.checkpoint_mapped_bytes() == 0);
    decoder.reset_checkpoint_peak();
    assert(decoder.checkpoint_peak_bytes() == 0);
}
} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) { return 77; }
    for (auto dtype : {DType::BF16, DType::FP8_E4M3FN}) {
        exercise(dtype, true);
        exercise(dtype, false);
    }
    std::cout << "Conversation snapshot demand mapping, bounds, state, eviction and sleep/wake passed\n";
}
