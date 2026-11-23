// Inkling-2 weight bindings (M3) against the converted artifact, host only: every object is bound with the
// expected format and shape and `Binder::finish()` accepts the plan (each object consumed and placed exactly once),
// for the oracle (BF16 attention) choice, with and without the drafter, and with every placement validate-only.
// When the artifact carries the FP8 per-channel attention (the production candidate `*.attn-fp8pc.sinfer`), also for
// that choice: the FP8 objects go to the device, the BF16 attention is validated only, and the planned device bytes
// differ from the oracle plan by exactly the attention objects' sizes. A scheme the artifact lacks is refused.
// The 1-card oracle plan places every tensor but the FP8 variants (19,525,154,400 device bytes, the figure of the code
// before the per-stage globals split); the 2-stage plan of the 2-card design (cut at visit 30) holds the
// embedding on stage 0 only, the head + final norm and the drafter on stage 1 only, the small globals on both, and
// each physical layer on the stages that run one of its visits.
// Needs INKLING2_ARTIFACT (skips otherwise). No device is touched: placements are planned, never materialised.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/inkling2/impl/load/bindings.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <iostream>
#include <variant>
#include <stdexcept>
#include <string>

namespace {

namespace load = sinfer::targets::inkling2::load;
using sinfer::artifact::ArtifactError;
using sinfer::artifact::Binder;
using sinfer::artifact::MaterializationPlan;
using sinfer::artifact::NumericFormat;
using sinfer::artifact::Reader;
using sinfer::artifact::TensorDescriptor;
using sinfer::artifact::TensorPlacement;

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error("FAIL: " + what); }
}

load::BindingPlan bind(const Reader& reader, load::BindOptions options, bool expect_finish = true,
                       std::uint64_t* device_bytes = nullptr) {
    Binder binder(reader);
    auto plan = load::bind_artifact(binder, options);
    if (expect_finish) {
        const MaterializationPlan placed = binder.finish();
        if (device_bytes != nullptr) {
            *device_bytes = 0;
            for (const auto& object : placed.device_objects) { *device_bytes += object.bytes; }
        }
    }
    return plan;
}

std::uint64_t stored_bytes(const Reader& reader, const std::string& name) {
    const auto* tensor = std::get_if<TensorDescriptor>(reader.find(name));
    if (tensor == nullptr) { throw std::runtime_error("FAIL: no object " + name); }
    return tensor->bytes;
}

/// The names a finished plan puts on the device, and their stored bytes.
struct Placed {
    std::set<std::string> device;
    std::uint64_t bytes = 0;
};

Placed place(const Reader& reader, const load::BindOptions& options) {
    Binder binder(reader);
    (void)load::bind_artifact(binder, options);
    const MaterializationPlan plan = binder.finish();
    Placed out;
    for (const auto& object : plan.device_objects) {
        out.device.insert(std::string(sinfer::artifact::object_name(reader.objects()[object.object.index])));
        out.bytes += object.bytes;
    }
    return out;
}

bool starts(const std::string& name, const std::string& prefix) { return name.rfind(prefix, 0) == 0; }

/// -1 for a global object, else its physical layer ("text/layers/<L>/...").
int layer_of(const std::string& name) {
    if (!starts(name, "text/layers/")) { return -1; }
    return std::stoi(name.substr(12, name.find('/', 12) - 12));
}

bool small_global(const std::string& name) {
    return starts(name, "text/attn_res/") || starts(name, "text/loop/");
}

void test_one_card_and_two_stages(const Reader& reader, const load::ArtifactConfig& c) {
    // 1 card, oracle attention, drafter on: every tensor except the FP8 variants, as before the split.
    load::BindOptions one;
    one.draft = true;
    const Placed card = place(reader, one);
    std::uint64_t expected = 0;
    std::size_t tensors    = 0;
    for (const auto& object : reader.objects()) {
        const auto* t = std::get_if<TensorDescriptor>(&object);
        if (t == nullptr || t->name.find("/attn/fp8-") != std::string::npos) { continue; }
        require(card.device.count(t->name) == 1, "1-card plan places " + t->name);
        expected += t->bytes;
        ++tensors;
    }
    require(card.device.size() == tensors && card.bytes == expected, "1-card plan = every non-FP8 tensor");
    require(card.bytes == 19525154400ULL, "1-card device bytes unchanged by the split (19,525,154,400)");

    // 2 stages, cut at visit 30 (Package::pipeline_stage_bounds): stage s keeps the layers its visits use.
    const std::int32_t cut = 30;
    std::array<std::set<int>, 2> resident;
    for (std::int32_t v = 0; v < static_cast<std::int32_t>(c.visits.size()); ++v) {
        resident[v < cut ? 0 : 1].insert(c.visits[v][0]);
    }
    std::array<Placed, 2> stage;
    for (int s = 0; s < 2; ++s) {
        load::BindOptions o;
        o.draft             = s == 1;
        const auto keep     = resident[s];
        o.layer_placement   = [keep](std::int32_t layer) {
            return keep.count(layer) ? TensorPlacement::Device : TensorPlacement::ValidateOnly;
        };
        o.globals_placement       = TensorPlacement::Device;
        o.embedding_placement     = s == 0 ? TensorPlacement::Device : TensorPlacement::ValidateOnly;
        o.head_placement          = s == 1 ? TensorPlacement::Device : TensorPlacement::ValidateOnly;
        o.small_globals_placement = TensorPlacement::Device;
        stage[s] = place(reader, o);
    }
    std::uint64_t duplicated = 0;
    for (const std::string& name : card.device) {
        const int layer = layer_of(name);
        const bool on0 = stage[0].device.count(name) == 1, on1 = stage[1].device.count(name) == 1;
        bool want0 = false, want1 = false;
        if (layer >= 0) {
            want0 = resident[0].count(layer) == 1;
            want1 = resident[1].count(layer) == 1;
        } else if (name == "text/embedding" || name == "text/embedding_norm") {
            want0 = true;
        } else if (name == "text/output_head" || name == "text/final_norm" || starts(name, "mtp/")) {
            want1 = true;
        } else if (small_global(name)) {
            want0 = want1 = true;
        } else {
            require(false, "unclassified object " + name);
        }
        require(on0 == want0 && on1 == want1, "2-stage placement of " + name);
        if (want0 && want1) { duplicated += stored_bytes(reader, name); }
    }
    for (int s = 0; s < 2; ++s) {
        for (const std::string& name : stage[s].device) {
            require(card.device.count(name) == 1, "stage " + std::to_string(s) + " places nothing new: " + name);
        }
    }
    require(stage[0].bytes + stage[1].bytes == card.bytes + duplicated, "2-stage bytes = 1 card + duplicates");
    const std::uint64_t tables = stored_bytes(reader, "text/embedding") + stored_bytes(reader, "text/output_head");
    std::cout << "2 stages (cut " << cut << "): " << stage[0].bytes << " + " << stage[1].bytes
              << " device bytes; duplicated (core layers + small globals) " << duplicated
              << "; the two global tables (" << tables << " bytes) held once\n";
}

bool refuses(const Reader& reader, load::BindOptions options) {
    try {
        (void)bind(reader, options, false);
    } catch (const ArtifactError&) {
        return true;
    }
    return false;
}

void test_config(const load::ArtifactConfig& c) {
    require(c.hidden == 2304 && c.layers == 24 && c.vocab_padded == 201024 && c.vocab == 200058, "text dims");
    require(c.qkvr_rows() == 8600 && c.qkvr_query_side_rows() == 6552, "qkvr rows");
    require(c.visits.size() == 48 && c.visits[23][0] == 11 && c.visits[23][1] == 1 && c.visits[47][0] == 23, "visits");
    require(c.global[5] && c.global[23] && !c.global[6], "global layers");
    require(c.expert_format[19] == load::ExpertFormat::Nvfp4 && c.expert_format[20] == load::ExpertFormat::Mxfp8,
            "expert formats");
    require(c.core_start == 6 && c.core_end == 18 && c.passes == 3 && c.gate_rank == 64, "loop");
    require(c.gate_up_interleaved, "w13 rows interleaved");
    require(c.dspark.has_value() && c.dspark->target_visits.back() == 48 && c.dspark->layers == 3, "dspark");
}

} // namespace

int main() {
    try {
        const char* path = std::getenv("INKLING2_ARTIFACT");
        if (path == nullptr || *path == '\0') {
            std::cout << "INKLING2_ARTIFACT not set: skipped\n";
            return 0;
        }
        const Reader reader(path);

        // The oracle path with the drafter, every object on the device (planned only).
        load::BindOptions device;
        device.draft             = true;
        device.layer_placement   = [](std::int32_t) { return TensorPlacement::Device; };
        device.globals_placement = TensorPlacement::Device;
        std::uint64_t oracle_bytes = 0;
        const auto plan = bind(reader, device, true, &oracle_bytes);
        test_config(plan.config);
        test_one_card_and_two_stages(reader, plan.config);
        require(plan.layers.size() == 24 && plan.has_draft && plan.draft.layers.size() == 3, "plan shape");
        require(plan.layers[5].k_conv_proj.has_value() && !plan.layers[4].k_conv_proj.has_value(), "conv kinds");
        require(plan.layers[0].w13.amax.has_value() && !plan.layers[20].w13.amax.has_value(), "bank kinds");
        require(plan.layers[0].w13.outer_scale_denominator_value == 2688.0F, "denominator word");
        require(plan.layers[3].qkvr.format == NumericFormat::BF16 && !plan.layers[3].qkvr_fp8, "BF16 attention");

        // Without the drafter, and with every placement validate-only (a stage that holds nothing).
        load::BindOptions none;
        none.layer_placement   = [](std::int32_t) { return TensorPlacement::ValidateOnly; };
        none.globals_placement = TensorPlacement::ValidateOnly;
        const auto empty = bind(reader, none);
        require(!empty.layers[0].resident, "validate-only layers are not resident");

        // FP8 attention: every scheme the artifact lacks is refused; per-channel, when carried, is served.
        const auto& schemes = plan.config.fp8_schemes;
        for (const std::string scheme : {"tensor", "channel", "g1x128", "b128"}) {
            if (std::find(schemes.begin(), schemes.end(), scheme) != schemes.end()) { continue; }
            load::BindOptions missing = none;
            missing.attention         = {true, scheme};
            require(refuses(reader, missing), "absent FP8 scheme " + scheme + " refused");
        }
        std::string served = "BF16 attention only";
        if (std::find(schemes.begin(), schemes.end(), "channel") != schemes.end()) {
            load::BindOptions fp8 = device;
            fp8.attention         = {true, "channel"};
            std::uint64_t fp8_bytes = 0;
            const auto fp8_plan     = bind(reader, fp8, true, &fp8_bytes);
            std::uint64_t bf16_attention = 0, fp8_attention = 0;
            for (std::int32_t layer = 0; layer < fp8_plan.config.layers; ++layer) {
                const auto& L       = fp8_plan.layers[layer];
                const std::string p = "text/layers/" + std::to_string(layer) + "/attn/";
                require(L.qkvr_fp8.has_value() && L.output_fp8.has_value(), "FP8 objects bound, layer " + p);
                require(L.qkvr_fp8->format == NumericFormat::FP8_E4M3FN_ROW_F32S &&
                            L.output_fp8->format == NumericFormat::FP8_E4M3FN_ROW_F32S, "per-channel format " + p);
                require(L.qkvr.format == NumericFormat::BF16, "the BF16 oracle attention is still bound " + p);
                bf16_attention += stored_bytes(reader, p + "qkvr") + stored_bytes(reader, p + "output");
                fp8_attention += stored_bytes(reader, p + "fp8-channel/qkvr") +
                                 stored_bytes(reader, p + "fp8-channel/output");
            }
            // The FP8 choice places the FP8 objects instead of the BF16 attention, and nothing else changes.
            require(fp8_bytes + bf16_attention == oracle_bytes + fp8_attention, "device bytes of the FP8 choice");
            // The oracle choice validates the FP8 objects only (they are in the artifact, never on the device).
            require(!plan.layers[0].qkvr_fp8.has_value(), "oracle plan leaves FP8 unbound to the device");
            served = "FP8 per-channel served (device bytes " + std::to_string(fp8_bytes) + " vs oracle " +
                     std::to_string(oracle_bytes) + ")";
        }
        std::cout << "PASS inkling2 bindings: " << reader.objects().size() << " objects bound; " << served << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
