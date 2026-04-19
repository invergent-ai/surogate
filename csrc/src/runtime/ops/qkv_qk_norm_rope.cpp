#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {
namespace {

void log_tensor_shape(const char* label, const Tensor& t) {
    fprintf(stderr, "  %s rank=%d dtype=%d sizes=[", label, t.Rank, (int)t.DType);
    for (int i = 0; i < t.Rank; ++i) {
        fprintf(stderr, "%ld%s", t.Sizes[i], (i + 1 < t.Rank) ? "," : "");
    }
    fprintf(stderr, "]\n");
}

void log_qkv_mismatch(const char* op_name,
                      int B,
                      int T,
                      int expected_qkv,
                      long actual_qkv,
                      int Hq,
                      int Hkv,
                      int Hs,
                      bool shard_weights,
                      const Tensor& qkv,
                      const Tensor& q_norm,
                      const Tensor& k_norm,
                      const Tensor& q_rstd,
                      const Tensor& k_rstd) {
    fprintf(stderr, "[QKV_DEBUG] %s qkv shape mismatch\n", op_name);
    fprintf(stderr,
            "  B=%d T=%d expected_qkv=%d actual_qkv=%ld Hq=%d Hkv=%d Hs=%d shard_weights=%d\n",
            B,
            T,
            expected_qkv,
            actual_qkv,
            Hq,
            Hkv,
            Hs,
            shard_weights ? 1 : 0);
    log_tensor_shape("qkv", qkv);
    log_tensor_shape("q_norm", q_norm);
    log_tensor_shape("k_norm", k_norm);
    log_tensor_shape("q_rstd", q_rstd);
    log_tensor_shape("k_rstd", k_rstd);
    fprintf(stderr, "[QKV_DEBUG] aborting: qkv size does not match config and sharding is disabled\n");
}

std::string tensor_shape_debug(const Tensor& t) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < t.Rank; ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << t.Sizes[i];
    }
    oss << "]";
    return oss.str();
}

void debug_dump_runtime_tensor(const std::string& name, const Tensor& t, const char* dump_dir) {
    if (!dump_dir || !*dump_dir || !t.Data || t.nelem() <= 0) {
        return;
    }
    std::string safe;
    safe.reserve(name.size());
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
            safe += c;
        } else {
            safe += '_';
        }
    }
    const std::size_t nelem = static_cast<std::size_t>(t.nelem());
    std::vector<float> host_data(nelem);
    if (t.DType == ETensorDType::FP32) {
        CUDA_CHECK(cudaMemcpy(host_data.data(), t.Data, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    } else if (t.DType == ETensorDType::BF16) {
        std::vector<uint16_t> bf16_data(nelem);
        CUDA_CHECK(cudaMemcpy(bf16_data.data(), t.Data, nelem * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < nelem; ++i) {
            uint32_t bits = static_cast<uint32_t>(bf16_data[i]) << 16;
            float val;
            std::memcpy(&val, &bits, sizeof(float));
            host_data[i] = val;
        }
    } else {
        return;
    }
    const std::string bin_path = std::string(dump_dir) + "/" + safe + ".bin";
    FILE* bin_f = std::fopen(bin_path.c_str(), "wb");
    if (bin_f) {
        std::fwrite(host_data.data(), sizeof(float), nelem, bin_f);
        std::fclose(bin_f);
    }
    const std::string json_path = std::string(dump_dir) + "/" + safe + ".json";
    FILE* json_f = std::fopen(json_path.c_str(), "w");
    if (json_f) {
        std::fprintf(json_f, "{\"name\": \"%s\", \"dtype\": \"float32\", \"shape\": [", name.c_str());
        for (int i = 0; i < t.Rank; ++i) {
            std::fprintf(json_f, "%ld%s", t.Sizes[i], (i + 1 < t.Rank) ? ", " : "");
        }
        std::fprintf(json_f, "]}\n");
        std::fclose(json_f);
    }
}

bool should_dump_qkv_layer(int layer_idx) {
    const char* layer_env = std::getenv("SUROGATE_DEBUG_QKV_DUMP_LAYER");
    if (!layer_env || !*layer_env) {
        return false;
    }
    char* end = nullptr;
    const long requested = std::strtol(layer_env, &end, 10);
    if (end == layer_env) {
        return false;
    }
    return layer_idx == static_cast<int>(requested);
}

}  // namespace

void CompiledExecutor::dispatch_qkv_qk_norm_rope(const CompiledOp& op) {
    Tensor& qkv_in = resolve_tensor(op.inputs[0]);
    Tensor& q_norm = resolve_tensor(op.inputs[1]);
    Tensor& k_norm = resolve_tensor(op.inputs[2]);
    Tensor& freqs = resolve_tensor(op.inputs[3]);
    Tensor& pos_ids = resolve_tensor(op.inputs[4]);
    int dump_layer_idx = -1;
    std::string dump_field;
    const bool dump_layer = parse_block_param(op.inputs[0].name, dump_layer_idx, dump_field) && dump_layer_idx >= 0;

    int Hq = static_cast<int>(mConfig.NumQueryHeads);
    int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(qkv_in, Hq, Hkv, static_cast<int>(mConfig.head_size()));
    const Tensor& qkv_out_candidate = ensure_output_tensor(op.outputs[0]);
    const Tensor& q_rstd_candidate = ensure_output_tensor(op.outputs[1]);
    const Tensor& k_rstd_candidate = ensure_output_tensor(op.outputs[2]);
    const std::vector<long> qkv_shape(qkv_in.Sizes.begin(), qkv_in.Sizes.begin() + qkv_in.Rank);
    const std::vector<long> q_rstd_shape =
        !op.outputs[1].shape.empty() ? op.outputs[1].shape : std::vector<long>{mB, mT, Hq};
    const std::vector<long> k_rstd_shape =
        !op.outputs[2].shape.empty() ? op.outputs[2].shape : std::vector<long>{mB, mT, Hkv};
    Tensor qkv_out = ensure_output_tensor_or_persistent(qkv_out_candidate,
                                                        mRunState,
                                                        mMoeSavedBuffers,
                                                        mMoeSavedSizes,
                                                        op.op_id + "." + op.outputs[0].name + ".qkv_out",
                                                        qkv_in.DType,
                                                        qkv_shape,
                                                        "qkv_qk_norm_rope");
    Tensor q_rstd = ensure_output_tensor_or_persistent(q_rstd_candidate,
                                                       mRunState,
                                                       mMoeSavedBuffers,
                                                       mMoeSavedSizes,
                                                       op.op_id + "." + op.outputs[1].name + ".q_rstd",
                                                       ETensorDType::FP32,
                                                       q_rstd_shape,
                                                       "qkv_qk_norm_rope");
    Tensor k_rstd = ensure_output_tensor_or_persistent(k_rstd_candidate,
                                                       mRunState,
                                                       mMoeSavedBuffers,
                                                       mMoeSavedSizes,
                                                       op.op_id + "." + op.outputs[2].name + ".k_rstd",
                                                       ETensorDType::FP32,
                                                       k_rstd_shape,
                                                       "qkv_qk_norm_rope");
    const bool cudnn_gqa_ok = (Hq == Hkv);
    int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int qkv_expected = qkv_channels;

    // If input and output are different buffers, copy input to output first.
    // The kernel operates in-place on the output buffer.
    if (qkv_in.Data != qkv_out.Data) {
        cudaMemcpyAsync(qkv_out.Data, qkv_in.Data, qkv_in.bytes(), cudaMemcpyDeviceToDevice, mRunState.MainStream);
    }

    auto actual_qkv_channels = [](const Tensor& t) -> long {
        if (t.Rank == 4) {
            return t.Sizes[2] * t.Sizes[3];
        }
        if (t.Rank == 3) {
            return t.Sizes[2];
        }
        return 0;
    };
    const long qkv_actual =
        actual_qkv_channels(qkv_in) > 0 ? actual_qkv_channels(qkv_in) : actual_qkv_channels(qkv_out);
    if (qkv_actual > 0 && qkv_actual != qkv_expected && !mOptions.ShardWeights) {
        log_qkv_mismatch("qkv_qk_norm_rope",
                         static_cast<int>(mB),
                         static_cast<int>(mT),
                         qkv_expected,
                         qkv_actual,
                         Hq,
                         Hkv,
                         Hs,
                         false,
                         qkv_in,
                         q_norm,
                         k_norm,
                         q_rstd,
                         k_rstd);
        throw std::runtime_error("qkv_qk_norm_rope: unexpected qkv shape (no sharding enabled)");
    }
    if (qkv_actual > 0 && qkv_actual != qkv_channels) {
        int q_heads = (q_rstd.Rank == 3) ? static_cast<int>(q_rstd.Sizes[2]) : -1;
        int k_heads = (k_rstd.Rank == 3) ? static_cast<int>(k_rstd.Sizes[2]) : -1;
        if (q_heads > 0 && k_heads > 0) {
            const long expected = static_cast<long>(Hs) * (q_heads + 2 * k_heads);
            if (expected == qkv_actual) {
                Hq = q_heads;
                Hkv = k_heads;
                qkv_channels = static_cast<int>(qkv_actual);
            }
        }
        if (qkv_channels != qkv_actual) {
            if (qkv_channels % qkv_actual == 0) {
                const int shard_factor = static_cast<int>(qkv_channels / qkv_actual);
                if (shard_factor > 1 && (Hq % shard_factor) == 0 && (Hkv % shard_factor) == 0) {
                    Hq /= shard_factor;
                    Hkv /= shard_factor;
                    qkv_channels = static_cast<int>(qkv_actual);
                }
            }
        }
    }
    Tensor qkv_view = qkv_out;
    const long qkv_needed = static_cast<long>(mB) * static_cast<long>(mT) * qkv_channels;
    if ((qkv_out.Rank == 4 || (qkv_out.Rank == 3 && qkv_out.Sizes[2] != qkv_channels)) &&
        static_cast<long>(qkv_out.nelem()) >= qkv_needed) {
        qkv_view = view_tensor(qkv_out, {mB, mT, qkv_channels});
    }
    auto view_rstd = [&](Tensor& rstd, int heads) -> Tensor {
        const long needed = static_cast<long>(mB) * static_cast<long>(mT) * heads;
        if (rstd.Rank == 3 && rstd.Sizes[0] == mB && rstd.Sizes[1] == mT && rstd.Sizes[2] == heads) {
            return rstd;
        }
        if (static_cast<long>(rstd.nelem()) >= needed) {
            return view_tensor(rstd, {mB, mT, heads});
        }
        return rstd;
    };
    Tensor q_rstd_view = view_rstd(q_rstd, Hq);
    Tensor k_rstd_view = view_rstd(k_rstd, Hkv);
    int rotary_dim = op.attrs.rotary_dim;
    const int freq_row_width = (freqs.Rank >= 3 && freqs.Sizes[2] == 2)
                                   ? static_cast<int>(freqs.Sizes[1] * freqs.Sizes[2])
                                   : (freqs.Rank >= 2 ? static_cast<int>(freqs.Sizes[1]) : 0);

    const bool rope_fusable = (rotary_dim > 0) && ((Hs % 2) == 0) && (((Hs / 2) % 32) == 0) && (freqs.Rank >= 2) &&
                              (freq_row_width >= rotary_dim) && (qkv_view.Rank == 3);

    if (mForwardPlan) {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.inputs[0].name, layer_idx, field) && layer_idx >= 0 &&
            static_cast<std::size_t>(layer_idx) < mForwardPlan->size()) {
            AttnForwardPlan plan{};
            plan.valid = true;
            plan.use_qk_norm = true;
            plan.rope_fused = rope_fusable;
            plan.use_cudnn = cudnn_gqa_ok;
            plan.rotary_dim = rotary_dim;
            (*mForwardPlan)[static_cast<std::size_t>(layer_idx)].attn = plan;
        }
    }

    if (dump_layer && should_dump_qkv_layer(dump_layer_idx) && std::getenv("SUROGATE_DEBUG_DUMP_DIR")) {
        std::cerr << "[QKV] layer=" << dump_layer_idx << " in_shape=" << tensor_shape_debug(qkv_in)
                  << " out_shape=" << tensor_shape_debug(qkv_view) << " freqs_shape=" << tensor_shape_debug(freqs)
                  << " rotary_dim=" << rotary_dim << std::endl;
        const char* dump_dir = std::getenv("SUROGATE_DEBUG_DUMP_DIR");
        debug_dump_runtime_tensor(fmt::format("qkv_live.layer{}.in", dump_layer_idx), qkv_in, dump_dir);
    }

    if (rope_fusable) {
        qkv_qk_norm_rope_forward(qkv_view,
                                 q_rstd_view,
                                 k_rstd_view,
                                 q_norm,
                                 k_norm,
                                 freqs,
                                 reinterpret_cast<int*>(pos_ids.Data),
                                 op.attrs.eps,
                                 static_cast<int>(mB),
                                 static_cast<int>(mT),
                                 Hq,
                                 Hkv,
                                 Hs,
                                 rotary_dim,
                                 mRunState.MainStream);
    } else {
        const int q_rows = Hq * Hs;
        qkv_head_rmsnorm_forward(qkv_view,
                                 q_rstd_view,
                                 q_norm,
                                 op.attrs.eps,
                                 static_cast<int>(mB),
                                 static_cast<int>(mT),
                                 qkv_channels,
                                 Hq,
                                 Hs,
                                 0,
                                 mRunState.MainStream);
        qkv_head_rmsnorm_forward(qkv_view,
                                 k_rstd_view,
                                 k_norm,
                                 op.attrs.eps,
                                 static_cast<int>(mB),
                                 static_cast<int>(mT),
                                 qkv_channels,
                                 Hkv,
                                 Hs,
                                 q_rows,
                                 mRunState.MainStream);
        rope_forward(qkv_out,
                     qkv_out,
                     freqs,
                     reinterpret_cast<int*>(pos_ids.Data),
                     nullptr,
                     static_cast<int>(mB),
                     static_cast<int>(mT),
                     Hq,
                     Hkv,
                     Hs,
                     rotary_dim,
                     mRunState.MainStream);
    }

    if (dump_layer && should_dump_qkv_layer(dump_layer_idx) && std::getenv("SUROGATE_DEBUG_DUMP_DIR")) {
        const char* dump_dir = std::getenv("SUROGATE_DEBUG_DUMP_DIR");
        debug_dump_runtime_tensor(fmt::format("qkv_live.layer{}.out", dump_layer_idx), qkv_view, dump_dir);
        debug_dump_runtime_tensor(fmt::format("qkv_live.layer{}.q_rstd", dump_layer_idx), q_rstd_view, dump_dir);
        debug_dump_runtime_tensor(fmt::format("qkv_live.layer{}.k_rstd", dump_layer_idx), k_rstd_view, dump_dir);
    }

    store_tensor(op.outputs[0], qkv_out);
    store_tensor(op.outputs[1], q_rstd);
    store_tensor(op.outputs[2], k_rstd);
}

void CompiledExecutor::dispatch_qkv_qk_norm_rope_backward(const CompiledOp& op) {
    // inputs (from autodiff): d_qkv_out, qkv_out (saved), q_norm_weight, k_norm_weight, q_rstd, k_rstd, freqs, pos_ids
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& qkv = resolve_tensor(op.inputs[1]);  // Saved QKV output from forward
    Tensor& q_norm = resolve_tensor(op.inputs[2]);
    Tensor& k_norm = resolve_tensor(op.inputs[3]);
    Tensor& q_rstd = resolve_tensor(op.inputs[4]);  // Saved RSTD (FP32)
    Tensor& k_rstd = resolve_tensor(op.inputs[5]);  // Saved RSTD (FP32)
    Tensor& freqs = resolve_tensor(op.inputs[6]);
    Tensor& pos_ids = resolve_tensor(op.inputs[7]);
    // Resolve optional dweight outputs first. ensure_output_tensor may allocate into mTemps,
    // so we must not take references to mTemps-backed d_qkv before this point.
    Tensor* d_q_norm = nullptr;
    Tensor* d_k_norm = nullptr;
    bool accum_q = false;
    bool accum_k = false;
    const bool skip_norm_dweight = mRunState.is_lora_only_mode();
    if (!skip_norm_dweight && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        d_q_norm = &ensure_output_tensor(op.outputs[1]);
        accum_q = mAccumulateTensors.count(op.outputs[1].name) > 0;
    }
    if (!skip_norm_dweight && op.outputs.size() > 2 && !op.outputs[2].name.empty()) {
        d_k_norm = &ensure_output_tensor(op.outputs[2]);
        accum_k = mAccumulateTensors.count(op.outputs[2].name) > 0;
    }

    const std::vector<long> d_qkv_shape(qkv.Sizes.begin(), qkv.Sizes.begin() + qkv.Rank);
    Tensor d_qkv = ensure_output_tensor_or_persistent(ensure_output_tensor(op.outputs[0]),
                                                      mRunState,
                                                      mMoeSavedBuffers,
                                                      mMoeSavedSizes,
                                                      op.op_id + "." + op.outputs[0].name + ".d_qkv",
                                                      d_out.DType,
                                                      d_qkv_shape,
                                                      "qkv_qk_norm_rope_backward");

    int Hq = static_cast<int>(mConfig.NumQueryHeads);
    int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(qkv, Hq, Hkv, static_cast<int>(mConfig.head_size()));
    int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int qkv_expected = qkv_channels;

    auto actual_qkv_channels = [](const Tensor& t) -> long {
        if (t.Rank == 4) {
            return t.Sizes[2] * t.Sizes[3];
        }
        if (t.Rank == 3) {
            return t.Sizes[2];
        }
        return 0;
    };
    const long qkv_actual = actual_qkv_channels(qkv) > 0 ? actual_qkv_channels(qkv) : actual_qkv_channels(d_out);
    if (qkv_actual > 0 && qkv_actual != qkv_expected && !mOptions.ShardWeights) {
        log_qkv_mismatch("qkv_qk_norm_rope_backward",
                         static_cast<int>(mB),
                         static_cast<int>(mT),
                         qkv_expected,
                         qkv_actual,
                         Hq,
                         Hkv,
                         Hs,
                         false,
                         qkv,
                         q_norm,
                         k_norm,
                         q_rstd,
                         k_rstd);
        throw std::runtime_error("qkv_qk_norm_rope_backward: unexpected qkv shape (no sharding enabled)");
    }
    if (qkv_actual > 0 && qkv_actual != qkv_channels) {
        int q_heads = (q_rstd.Rank == 3) ? static_cast<int>(q_rstd.Sizes[2]) : -1;
        int k_heads = (k_rstd.Rank == 3) ? static_cast<int>(k_rstd.Sizes[2]) : -1;
        if (q_heads > 0 && k_heads > 0) {
            const long expected = static_cast<long>(Hs) * (q_heads + 2 * k_heads);
            if (expected == qkv_actual) {
                Hq = q_heads;
                Hkv = k_heads;
                qkv_channels = static_cast<int>(qkv_actual);
            }
        }
        if (qkv_channels != qkv_actual) {
            if (qkv_channels % qkv_actual == 0) {
                const int shard_factor = static_cast<int>(qkv_channels / qkv_actual);
                if (shard_factor > 1 && (Hq % shard_factor) == 0 && (Hkv % shard_factor) == 0) {
                    Hq /= shard_factor;
                    Hkv /= shard_factor;
                    qkv_channels = static_cast<int>(qkv_actual);
                }
            }
        }
    }
    const int q_rows = Hq * Hs;
    auto view_qkv = [&](Tensor& t) -> Tensor {
        const long needed = static_cast<long>(mB) * static_cast<long>(mT) * qkv_channels;
        if ((t.Rank == 4 || (t.Rank == 3 && t.Sizes[2] != qkv_channels)) && static_cast<long>(t.nelem()) >= needed) {
            return view_tensor(t, {mB, mT, static_cast<long>(qkv_channels)});
        }
        return t;
    };
    auto view_rstd = [&](Tensor& rstd, int heads) -> Tensor {
        const long needed = static_cast<long>(mB) * static_cast<long>(mT) * heads;
        if (rstd.Rank == 3 && rstd.Sizes[0] == mB && rstd.Sizes[1] == mT && rstd.Sizes[2] == heads) {
            return rstd;
        }
        if (static_cast<long>(rstd.nelem()) >= needed) {
            return view_tensor(rstd, {mB, mT, heads});
        }
        return rstd;
    };

    Tensor qkv_view = view_qkv(qkv);
    Tensor d_out_view = view_qkv(d_out);
    Tensor d_qkv_view = view_qkv(d_qkv);
    Tensor q_rstd_view = view_rstd(q_rstd, Hq);
    Tensor k_rstd_view = view_rstd(k_rstd, Hkv);
    int layer_idx = -1;
    std::string layer_field;
    if (!parse_block_param(op.inputs[1].name, layer_idx, layer_field)) {
        parse_block_param(op.outputs[0].name, layer_idx, layer_field);
    }

    // Compute d_weight before overwriting d_out_view.
    if (d_q_norm) {
        if (d_q_norm->DType == ETensorDType::FP32 && q_norm.DType != ETensorDType::FP32) {
            qkv_head_rmsnorm_rope_backward_dweight_fp32(*d_q_norm,
                                                        d_out_view,
                                                        qkv_view,
                                                        q_norm,
                                                        freqs,
                                                        reinterpret_cast<int*>(pos_ids.Data),
                                                        static_cast<int>(mB),
                                                        static_cast<int>(mT),
                                                        qkv_channels,
                                                        Hq,
                                                        Hs,
                                                        op.attrs.rotary_dim,
                                                        0,
                                                        accum_q,
                                                        mRunState.MainStream);
        } else {
            qkv_head_rmsnorm_rope_backward_dweight(*d_q_norm,
                                                   d_out_view,
                                                   qkv_view,
                                                   q_norm,
                                                   freqs,
                                                   reinterpret_cast<int*>(pos_ids.Data),
                                                   static_cast<int>(mB),
                                                   static_cast<int>(mT),
                                                   qkv_channels,
                                                   Hq,
                                                   Hs,
                                                   op.attrs.rotary_dim,
                                                   0,
                                                   accum_q,
                                                   mRunState.MainStream);
        }
    }
    if (d_k_norm) {
        if (d_k_norm->DType == ETensorDType::FP32 && k_norm.DType != ETensorDType::FP32) {
            qkv_head_rmsnorm_rope_backward_dweight_fp32(*d_k_norm,
                                                        d_out_view,
                                                        qkv_view,
                                                        k_norm,
                                                        freqs,
                                                        reinterpret_cast<int*>(pos_ids.Data),
                                                        static_cast<int>(mB),
                                                        static_cast<int>(mT),
                                                        qkv_channels,
                                                        Hkv,
                                                        Hs,
                                                        op.attrs.rotary_dim,
                                                        q_rows,
                                                        accum_k,
                                                        mRunState.MainStream);
        } else {
            qkv_head_rmsnorm_rope_backward_dweight(*d_k_norm,
                                                   d_out_view,
                                                   qkv_view,
                                                   k_norm,
                                                   freqs,
                                                   reinterpret_cast<int*>(pos_ids.Data),
                                                   static_cast<int>(mB),
                                                   static_cast<int>(mT),
                                                   qkv_channels,
                                                   Hkv,
                                                   Hs,
                                                   op.attrs.rotary_dim,
                                                   q_rows,
                                                   accum_k,
                                                   mRunState.MainStream);
        }
    }

    // Initialize d_qkv with upstream gradient (d_out) so V gradients pass through unchanged.
    // The fused or fallback kernels update Q/K channels in-place.
    if (d_qkv_view.Data != d_out_view.Data) {
        const std::size_t bytes = static_cast<std::size_t>(d_out_view.nelem()) * get_dtype_size(d_out_view.DType);
        CUDA_CHECK(
            cudaMemcpyAsync(d_qkv_view.Data, d_out_view.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }
    // Combined backward for Q and K norms with RoPE
    // Q norm backward (with RoPE): channel_offset=0
    qkv_head_rmsnorm_rope_backward_dx(d_qkv_view,
                                      qkv_view,
                                      q_norm,
                                      q_rstd_view,
                                      freqs,
                                      reinterpret_cast<int*>(pos_ids.Data),
                                      static_cast<int>(mB),
                                      static_cast<int>(mT),
                                      qkv_channels,
                                      Hq,
                                      Hs,
                                      op.attrs.rotary_dim,
                                      0,
                                      mRunState.MainStream,
                                      nullptr);
    // K norm backward (with RoPE): channel_offset=q_rows
    qkv_head_rmsnorm_rope_backward_dx(d_qkv_view,
                                      qkv_view,
                                      k_norm,
                                      k_rstd_view,
                                      freqs,
                                      reinterpret_cast<int*>(pos_ids.Data),
                                      static_cast<int>(mB),
                                      static_cast<int>(mT),
                                      qkv_channels,
                                      Hkv,
                                      Hs,
                                      op.attrs.rotary_dim,
                                      q_rows,
                                      mRunState.MainStream,
                                      nullptr);
    // V doesn't have normalization - its gradients pass through unchanged
    // The d_out already contains the V gradients at the correct offset

    // For FP8 hybrid backward, record abs_max of the final d_qkv for subsequent quantization
    if (mRunState.has_fp8_hybrid_backward()) {
        float* abs_max_ptr = mRunState.simplified_quant_grads().d_qkv.abs_max();
        abs_max(abs_max_ptr,
                d_qkv_view,
                static_cast<long>(d_qkv_view.nelem()),
                mRunState.DeviceProp,
                mRunState.MainStream);
    }

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], d_qkv);
    }
    if (d_q_norm && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        store_tensor(op.outputs[1], *d_q_norm);
    }
    if (d_k_norm && op.outputs.size() > 2 && !op.outputs[2].name.empty()) {
        store_tensor(op.outputs[2], *d_k_norm);
    }
}

namespace {

// -----------------------------------------------------------------------------
// QK-Norm + RoPE backward rule
// Forward: qkv_out, q_rstd, k_rstd = qkv_qk_norm_rope(qkv, q_norm_w, k_norm_w, freqs, pos_ids)
// Backward: d_qkv, d_q_norm_w, d_k_norm_w = qkv_qk_norm_rope_backward(...)
// -----------------------------------------------------------------------------
std::vector<Operation> qkv_qk_norm_rope_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.size() < 5 || fwd.outputs.size() < 3) {
        return ops;
    }

    // The backward kernel expects qkv_rope (the OUTPUT after QK-Norm + RoPE), NOT the original input.
    // The kernel internally applies inverse RoPE to recover the pre-RoPE values for gradient computation.
    std::string qkv_out = fwd.outputs[0];  // qkv_rope - output after QK-Norm + RoPE
    std::string q_rstd = fwd.outputs[1];
    std::string k_rstd = fwd.outputs[2];

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");
    outputs.push_back(ctx.needs_grad(2) ? ctx.d_inputs[2] : "");

    ops.push_back(make_operation("qkv_qk_norm_rope_backward_" + std::to_string(ctx.op_counter++),
                                 "qkv_qk_norm_rope_backward",
                                 "qkv_qk_norm_rope_backward",
                                 {ctx.d_output,
                                  saved_ref(qkv_out),  // Use OUTPUT qkv_rope - kernel applies inverse RoPE internally
                                  fwd.inputs[1],
                                  fwd.inputs[2],
                                  saved_ref(q_rstd),
                                  saved_ref(k_rstd),
                                  fwd.inputs[3],
                                  fwd.inputs[4]},
                                 outputs));

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("qkv_qk_norm_rope", ::dsl::qkv_qk_norm_rope_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// QKVQKNormRoPE
// ------------------------------------------------------------------------
const int _qkv_qk_norm_rope_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "qkv_qk_norm_rope";
    sig.min_inputs = 5;
    sig.max_inputs = 5;
    sig.min_outputs = 3;
    sig.max_outputs = 3;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& qkv = inputs[0];
        const auto& q_norm = inputs[1];
        const auto& k_norm = inputs[2];
        const auto& freqs = inputs[3];
        const auto& pos_ids = inputs[4];
        const auto& qkv_out = outputs[0];
        const auto& q_rstd = outputs[1];
        const auto& k_rstd = outputs[2];

        // Check qkv rank >= 2
        if (qkv.size() < 2) {
            ShapeValidationError err;
            err.message = "qkv_qk_norm_rope: qkv must have rank >= 2";
            return std::make_optional(err);
        }

        // Check q_norm and k_norm are 1D
        if (auto err = validators::check_rank(q_norm, 1, "q_norm", "qkv_qk_norm_rope")) {
            return err;
        }
        if (auto err = validators::check_rank(k_norm, 1, "k_norm", "qkv_qk_norm_rope")) {
            return err;
        }

        // Check freqs rank >= 2
        if (freqs.size() < 2) {
            ShapeValidationError err;
            err.message = "qkv_qk_norm_rope: freqs must have rank >= 2";
            return std::make_optional(err);
        }

        // Check output shape matches input
        if (auto err = validators::check_same_numel(qkv_out, qkv, "qkv_out", "qkv", "qkv_qk_norm_rope")) {
            return err;
        }

        // q_rstd/k_rstd can be flattened [B*T*H], [B*T, H], or [B, T, H]
        // Skip validation if shapes weren't inferred (empty means "unknown")
        const auto rstd_rank_ok = [](const std::vector<long>& s) {
            return s.empty() || s.size() == 1 || s.size() == 2 || s.size() == 3;
        };
        if (!rstd_rank_ok(q_rstd)) {
            ShapeValidationError err;
            err.message = "qkv_qk_norm_rope: q_rstd must be rank 1, 2, or 3";
            return std::make_optional(err);
        }
        if (!rstd_rank_ok(k_rstd)) {
            ShapeValidationError err;
            err.message = "qkv_qk_norm_rope: k_rstd must be rank 1, 2, or 3";
            return std::make_optional(err);
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// QKVQKNormRoPEBackward
// ------------------------------------------------------------------------
const int _qkv_qk_norm_rope_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "qkv_qk_norm_rope_backward";
    sig.min_inputs = 8;
    sig.max_inputs = 8;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& qkv = inputs[1];
        // inputs[2-7] are norm weights, rstds, freqs, pos_ids
        const auto& d_qkv = outputs[0];

        // d_qkv should match d_out and qkv
        if (auto err = validators::check_same_numel(d_qkv, d_out, "d_qkv", "d_out", "qkv_qk_norm_rope_backward")) {
            return err;
        }
        if (auto err = validators::check_same_numel(d_qkv, qkv, "d_qkv", "qkv", "qkv_qk_norm_rope_backward")) {
            return err;
        }

        if (outputs.size() > 1) {
            const auto& d_q_norm = outputs[1];
            const auto& q_norm = inputs[2];
            if (auto err =
                    validators::check_same_numel(d_q_norm, q_norm, "d_q_norm", "q_norm", "qkv_qk_norm_rope_backward")) {
                return err;
            }
        }
        if (outputs.size() > 2) {
            const auto& d_k_norm = outputs[2];
            const auto& k_norm = inputs[3];
            if (auto err =
                    validators::check_same_numel(d_k_norm, k_norm, "d_k_norm", "k_norm", "qkv_qk_norm_rope_backward")) {
                return err;
            }
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
