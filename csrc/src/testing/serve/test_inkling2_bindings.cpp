// Inkling-2 weight bindings (lane b, M3) against the converted artifact, host only: every object is bound with the
// expected format and shape and `Binder::finish()` accepts the plan (each object consumed and placed exactly once),
// for the oracle (BF16 attention) choice, with and without the drafter, and with every placement validate-only.
// Needs INKLING2_ARTIFACT (skips otherwise). No device is touched: placements are planned, never materialised.

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/inkling2/impl/load/bindings.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace load = sinfer::targets::inkling2::load;
using sinfer::artifact::ArtifactError;
using sinfer::artifact::Binder;
using sinfer::artifact::NumericFormat;
using sinfer::artifact::Reader;
using sinfer::artifact::TensorPlacement;

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error("FAIL: " + what); }
}

load::BindingPlan bind(const Reader& reader, load::BindOptions options, bool expect_finish = true) {
    Binder binder(reader);
    auto plan = load::bind_artifact(binder, options);
    if (expect_finish) { (void)binder.finish(); }
    return plan;
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
        const auto plan = bind(reader, device);
        test_config(plan.config);
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

        // An FP8 scheme the artifact does not carry is refused at bind.
        load::BindOptions fp8 = none;
        fp8.attention = {true, "channel"};
        bool refused  = false;
        try {
            (void)bind(reader, fp8, false);
        } catch (const ArtifactError&) {
            refused = true;
        }
        require(refused || !plan.config.fp8_schemes.empty(), "missing FP8 scheme refused");
        std::cout << "PASS inkling2 bindings: " << reader.objects().size() << " objects bound\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
