#include <api/family/text_geometry.h>
#include <api/family/vision_geometry.h>

#include <cassert>
#include <limits>
#include <stdexcept>

int main() {
    sinfer::family::TextGeometry geometry;
    geometry.override_from({{"hidden", 384}, {"layers", 7}, {"output_rows", 8192},
                            {"gdn_conv_kernel", 5}, {"experts", 32},
                            {"experts_per_token", 4}, {"attention_scale", 0.125}});
    assert(geometry.hidden == 384 && geometry.layers == 7);
    assert(geometry.output_rows == 8192 && geometry.gdn_conv_kernel == 5);
    assert(geometry.experts == 32 && geometry.experts_per_token == 4);
    assert(geometry.attention_scale == 0.125F);

    const auto before = geometry;
    const auto rejected = [&](const std::map<std::string, double>& values) {
        bool threw = false;
        try {
            geometry.override_from(values);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw && geometry == before);
    };
    rejected({{"hidden", 128}, {"vocab", 4096}});
    rejected({{"conv_kernel", 5}});
    rejected({{"hidden", 1.5}});
    rejected({{"layers", -1}});
    rejected({{"layers", 2147483648.0}});
    rejected({{"rope_theta", std::numeric_limits<double>::infinity()}});
    rejected({{"rms_epsilon", std::numeric_limits<double>::quiet_NaN()}});
    rejected({{"rope_theta", 1e100}});

    sinfer::family::VisionGeometry vision;
    vision.override_from({{"heads", 7}, {"hidden", 448}});
    assert(vision.heads == 7 && vision.hidden == 448);

    const std::map<std::string, double> complete{
        {"hidden", 256}, {"residual", 256}, {"layers", 3}, {"intermediate", 512},
        {"output_rows", 512}, {"token_domain", 500}, {"query_heads", 4}, {"kv_heads", 2},
        {"head_dim", 64}, {"rotary_dim", 64}, {"rms_epsilon", 1e-5},
        {"rope_theta", 123456}, {"max_context", 8192}, {"attention_scale", 0.125},
    };
    const std::array<std::string, 3> types{"full_attention", "full_attention", "full_attention"};
    const auto resolved = sinfer::family::TextGeometry::resolved(complete, types);
    assert(resolved.hidden == 256 && resolved.layers == 3 && resolved.max_context == 8192);
    assert(resolved.query_size() == 256 && resolved.kv_size() == 128);
    assert(resolved.attention_scale == 0.125F && resolved.rms_epsilon == 1e-5F);
    for (const auto& [name, unused] : complete) {
        auto incomplete = complete;
        incomplete.erase(name);
        bool threw = false;
        try {
            (void)sinfer::family::TextGeometry::resolved(incomplete, types);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    auto mrope = complete;
    mrope["mrope_temporal"] = 12;
    mrope["mrope_height"] = 10;
    mrope["mrope_width"] = 10;
    assert(sinfer::family::TextGeometry::resolved(mrope, types).mrope_temporal == 12);
    for (const auto& [name, value] : std::map<std::string, double>{
             {"mrope_temporal", 0}, {"mrope_height", 11}, {"mrope_width", 11}}) {
        auto bad = mrope;
        bad[name] = value;
        bool threw = false;
        try { (void)sinfer::family::TextGeometry::resolved(bad, types); }
        catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
    }

    auto hybrid_values = complete;
    hybrid_values["sliding_window"] = 256;
    const std::array<std::string, 3> hybrid_types{"sliding_attention", "linear_attention", "full_attention"};
    const auto hybrid = sinfer::family::TextGeometry::resolved(hybrid_values, hybrid_types);
    assert(hybrid.layer_attends(0) && !hybrid.layer_attends(1) && hybrid.layer_attends(2));
    assert(hybrid.layer_is_windowed(0) && !hybrid.layer_is_windowed(2));
    assert(hybrid.linear_layer_count() == 1);
    bool missing_schedule = false;
    try {
        (void)sinfer::family::TextGeometry::resolved(complete, {});
    } catch (const std::invalid_argument&) {
        missing_schedule = true;
    }
    assert(missing_schedule);

    auto spark = complete;
    spark["rotary_dim"] = 16;
    spark["sliding_rotary_dim"] = 64;
    spark["sliding_window"] = 192;
    spark["sliding_rope_theta"] = 10000;
    spark["residual_fp32"] = 1;
    const auto spark_geometry = sinfer::family::TextGeometry::resolved(spark, types);
    assert(spark_geometry.rotary_dim == 16 && spark_geometry.sliding_rotary_dim == 64);
    assert(spark_geometry.residual_dtype() == sinfer::DType::FP32);
    assert(resolved.residual_dtype() == sinfer::DType::BF16);
    for (const auto& [name, value] : std::map<std::string, double>{
             {"sliding_rotary_dim", 65}, {"sliding_window", 0},
             {"sliding_rope_theta", 0}, {"residual_fp32", 2}}) {
        auto bad = spark;
        bad[name] = value;
        bool threw = false;
        try { (void)sinfer::family::TextGeometry::resolved(bad, types); }
        catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
    }

    auto gemma3 = complete;
    gemma3.insert({{"sliding_window", 192}, {"sliding_rope_theta", 1234}, {"embedding_scale", 16}});
    const auto resolved_gemma3 = sinfer::family::TextGeometry::resolved_gemma3(gemma3, types);
    assert(resolved_gemma3.rms_epsilon == 1e-5F && resolved_gemma3.sliding_window == 192);
    for (const auto name : {"sliding_window", "sliding_rope_theta", "embedding_scale"}) {
        auto missing = gemma3;
        missing.erase(name);
        bool refused = false;
        try { (void)sinfer::family::TextGeometry::resolved_gemma3(missing, types); }
        catch (const std::invalid_argument&) { refused = true; }
        assert(refused);
    }

    auto gemma = complete;
    gemma.insert({{"global_head_dim", 64}, {"global_kv_heads", 2},
                  {"global_rotary_angles", 16}, {"sliding_window", 128},
                  {"sliding_rope_theta", 20000}, {"embedding_scale", 16},
                  {"logit_softcap", 0}, {"attention_k_eq_v", 1}});
    gemma["attention_scale"] = 1.0;
    gemma["rms_epsilon"] = 1e-4;
    const std::array<std::string, 3> gemma_types{
        "sliding_attention", "full_attention", "full_attention"};
    const auto gemma_dense = sinfer::family::TextGeometry::resolved_gemma4(gemma, gemma_types);
    // Equal head widths must not erase the distinction between local and global attention.
    assert(gemma_dense.has_global_attention_geometry());
    assert(gemma_dense.layer_is_windowed(0) && !gemma_dense.layer_is_windowed(1));
    assert(gemma_dense.rms_epsilon == 1e-4F && gemma_dense.attention_scale == 1.0F);
    for (const auto& [name, unused] : gemma) {
        auto incomplete = gemma;
        incomplete.erase(name);
        bool threw = false;
        try {
            (void)sinfer::family::TextGeometry::resolved_gemma4(incomplete, gemma_types);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    auto gemma_e = gemma;
    gemma_e.insert({{"per_layer_input_dim", 32}, {"per_layer_vocab", 512},
                    {"shared_kv_intermediate", 1024}, {"kv_shared_layers", 1}});
    const auto shared = sinfer::family::TextGeometry::resolved_gemma4(gemma_e, gemma_types, true);
    assert(shared.layer_owns_kv(0) && shared.layer_owns_kv(1) && !shared.layer_owns_kv(2));
    assert(shared.linear_layer_count() == 0); // shared KV ownership does not make a layer linear
    assert(shared.intermediate_for(false) == 1024);
    gemma_e["kv_shared_layers"] = 2;
    bool missing_owner = false;
    try {
        (void)sinfer::family::TextGeometry::resolved_gemma4(gemma_e, gemma_types, true);
    } catch (const std::invalid_argument&) {
        missing_owner = true;
    }
    assert(missing_owner);
    auto gemma_moe = gemma;
    gemma_moe.insert({{"experts", 8}, {"experts_per_token", 2}, {"dense_intermediate", 1024}});
    const auto mixture = sinfer::family::TextGeometry::resolved_gemma4(gemma_moe, gemma_types, false, true);
    assert(mixture.experts == 8 && mixture.experts_per_token == 2);
    assert(mixture.intermediate == 512 && mixture.dense_intermediate == 1024);
}
