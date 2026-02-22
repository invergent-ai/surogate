#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
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
    fprintf(stderr, "  B=%d T=%d expected_qkv=%d actual_qkv=%ld Hq=%d Hkv=%d Hs=%d shard_weights=%d\n",
            B, T, expected_qkv, actual_qkv, Hq, Hkv, Hs, shard_weights ? 1 : 0);
    log_tensor_shape("qkv", qkv);
    log_tensor_shape("q_norm", q_norm);
    log_tensor_shape("k_norm", k_norm);
    log_tensor_shape("q_rstd", q_rstd);
    log_tensor_shape("k_rstd", k_rstd);
    fprintf(stderr, "[QKV_DEBUG] aborting: qkv size does not match config and sharding is disabled\n");
}

}  // namespace

void CompiledExecutor::dispatch_qkv_qk_norm(const CompiledOp& op) {
    Tensor& qkv_in = resolve_tensor(op.inputs[0]);
    Tensor& q_norm = resolve_tensor(op.inputs[1]);
    Tensor& k_norm = resolve_tensor(op.inputs[2]);

    Tensor& qkv_out = ensure_output_tensor(op.outputs[0]);
    Tensor& q_rstd = ensure_output_tensor(op.outputs[1]);
    Tensor& k_rstd = ensure_output_tensor(op.outputs[2]);

    int Hq = static_cast<int>(mConfig.NumQueryHeads);
    int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int qkv_expected = qkv_channels;

    if (qkv_in.Data != qkv_out.Data) {
        CUDA_CHECK(cudaMemcpyAsync(qkv_out.Data, qkv_in.Data,
                                   qkv_in.bytes(),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
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
    const long qkv_actual = actual_qkv_channels(qkv_in) > 0 ? actual_qkv_channels(qkv_in)
                                                            : actual_qkv_channels(qkv_out);
    if (qkv_actual > 0 && qkv_actual != qkv_expected && !mOptions.ShardWeights) {
        log_qkv_mismatch("qkv_qk_norm",
                         static_cast<int>(mB), static_cast<int>(mT),
                         qkv_expected, qkv_actual,
                         Hq, Hkv, Hs, false,
                         qkv_in, q_norm, k_norm, q_rstd, k_rstd);
        throw std::runtime_error("qkv_qk_norm: unexpected qkv shape (no sharding enabled)");
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
    if (qkv_view.Rank != 3 || qkv_view.Sizes[0] != mB || qkv_view.Sizes[1] != mT ||
        qkv_view.Sizes[2] != qkv_channels) {
        fprintf(stderr, "[QKV_DEBUG] qkv_qk_norm pre-kernel shape mismatch\n");
        fprintf(stderr, "  expected=[%ld,%ld,%d] cfg: Hq=%d Hkv=%d Hs=%d shard_weights=%d\n",
                mB, mT, qkv_channels, Hq, Hkv, Hs, mOptions.ShardWeights ? 1 : 0);
        log_tensor_shape("qkv_in", qkv_in);
        log_tensor_shape("qkv_out", qkv_out);
        log_tensor_shape("qkv_view", qkv_view);
        log_tensor_shape("q_norm", q_norm);
        log_tensor_shape("k_norm", k_norm);
        log_tensor_shape("q_rstd", q_rstd);
        log_tensor_shape("k_rstd", k_rstd);
        throw std::runtime_error("qkv_qk_norm: unexpected qkv shape (pre-kernel)");
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
    const int q_rows = Hq * Hs;

    qkv_head_rmsnorm_forward(qkv_view, q_rstd_view, q_norm,
                             op.attrs.eps,
                             static_cast<int>(mB), static_cast<int>(mT),
                             qkv_channels, Hq, Hs, 0, mRunState.MainStream);
    qkv_head_rmsnorm_forward(qkv_view, k_rstd_view, k_norm,
                             op.attrs.eps,
                             static_cast<int>(mB), static_cast<int>(mT),
                             qkv_channels, Hkv, Hs, q_rows, mRunState.MainStream);

    store_tensor(op.outputs[0], qkv_out);
}

void CompiledExecutor::dispatch_qkv_qk_norm_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& qkv = resolve_tensor(op.inputs[1]);
    Tensor& q_norm = resolve_tensor(op.inputs[2]);
    Tensor& k_norm = resolve_tensor(op.inputs[3]);
    Tensor& q_rstd = resolve_tensor(op.inputs[4]);
    Tensor& k_rstd = resolve_tensor(op.inputs[5]);

    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    int Hq = static_cast<int>(mConfig.NumQueryHeads);
    int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
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
    const long qkv_actual = actual_qkv_channels(qkv) > 0 ? actual_qkv_channels(qkv)
                                                         : actual_qkv_channels(d_out);
    if (qkv_actual > 0 && qkv_actual != qkv_expected && !mOptions.ShardWeights) {
        log_qkv_mismatch("qkv_qk_norm_backward",
                         static_cast<int>(mB), static_cast<int>(mT),
                         qkv_expected, qkv_actual,
                         Hq, Hkv, Hs, false,
                         qkv, q_norm, k_norm, q_rstd, k_rstd);
        throw std::runtime_error("qkv_qk_norm_backward: unexpected qkv shape (no sharding enabled)");
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
        if ((t.Rank == 4 || (t.Rank == 3 && t.Sizes[2] != qkv_channels)) &&
            static_cast<long>(t.nelem()) >= needed) {
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

    // Optional weight gradients
    Tensor* d_q_norm = nullptr;
    Tensor* d_k_norm = nullptr;
    bool accum_q = false;
    bool accum_k = false;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        d_q_norm = &ensure_output_tensor(op.outputs[1]);
        accum_q = mAccumulateTensors.count(op.outputs[1].name) > 0;
    }
    if (op.outputs.size() > 2 && !op.outputs[2].name.empty()) {
        d_k_norm = &ensure_output_tensor(op.outputs[2]);
        accum_k = mAccumulateTensors.count(op.outputs[2].name) > 0;
    }

    // Compute d_weight before overwriting d_out_view.
    if (d_q_norm) {
        if (d_q_norm->DType == ETensorDType::FP32 && q_norm.DType != ETensorDType::FP32) {
            qkv_head_rmsnorm_backward_dweight_fp32(*d_q_norm, d_out_view, qkv_view, q_norm,
                                                   static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                                   Hq, Hs, 0, accum_q, mRunState.MainStream);
        } else {
            qkv_head_rmsnorm_backward_dweight(*d_q_norm, d_out_view, qkv_view, q_norm,
                                              static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                              Hq, Hs, 0, accum_q, mRunState.MainStream);
        }
    }
    if (d_k_norm) {
        if (d_k_norm->DType == ETensorDType::FP32 && k_norm.DType != ETensorDType::FP32) {
            qkv_head_rmsnorm_backward_dweight_fp32(*d_k_norm, d_out_view, qkv_view, k_norm,
                                                   static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                                   Hkv, Hs, q_rows, accum_k, mRunState.MainStream);
        } else {
            qkv_head_rmsnorm_backward_dweight(*d_k_norm, d_out_view, qkv_view, k_norm,
                                              static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                              Hkv, Hs, q_rows, accum_k, mRunState.MainStream);
        }
    }

    if (d_qkv_view.Data != d_out_view.Data) {
        const std::size_t bytes = static_cast<std::size_t>(d_out_view.nelem()) * get_dtype_size(d_out_view.DType);
        CUDA_CHECK(cudaMemcpyAsync(d_qkv_view.Data, d_out_view.Data, bytes,
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    qkv_head_rmsnorm_backward_dx(d_qkv_view, qkv_view, q_norm, q_rstd_view,
                                 static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                 Hq, Hs, 0, mRunState.MainStream);
    qkv_head_rmsnorm_backward_dx(d_qkv_view, qkv_view, k_norm, k_rstd_view,
                                 static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                 Hkv, Hs, q_rows, mRunState.MainStream);
}

}  // namespace dsl
