#include "dsl/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_embedding(const CompiledOp& op) {
    Tensor& token_ids = resolve_tensor(op.inputs[0]);
    Tensor& emb = op.inputs.size() > 1 ? resolve_tensor(op.inputs[1]) : mWeights.get("embedding");
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    // One-time embedding metadata log (shapes/dtypes/pointers).
    static int emb_meta_count = 0;
    if (emb_meta_count < 3) {
        emb_meta_count++;
        fprintf(stderr,
                "[EMBED_META] token_ids: dtype=%s rank=%d sizes={%ld,%ld,%ld,%ld,%ld} ptr=%p bytes=%zu\n",
                dtype_to_str(token_ids.DType), token_ids.Rank,
                token_ids.Sizes[0], token_ids.Sizes[1], token_ids.Sizes[2], token_ids.Sizes[3], token_ids.Sizes[4],
                token_ids.Data, token_ids.bytes());
        fprintf(stderr,
                "[EMBED_META] emb: dtype=%s rank=%d sizes={%ld,%ld,%ld,%ld,%ld} ptr=%p bytes=%zu\n",
                dtype_to_str(emb.DType), emb.Rank,
                emb.Sizes[0], emb.Sizes[1], emb.Sizes[2], emb.Sizes[3], emb.Sizes[4],
                emb.Data, emb.bytes());
        fprintf(stderr,
                "[EMBED_META] out: dtype=%s rank=%d sizes={%ld,%ld,%ld,%ld,%ld} ptr=%p bytes=%zu\n",
                dtype_to_str(out.DType), out.Rank,
                out.Sizes[0], out.Sizes[1], out.Sizes[2], out.Sizes[3], out.Sizes[4],
                out.Data, out.bytes());
    }

    // One-time embedding weight sample scan for NaN/Inf and magnitude stats
    static int emb_weight_scan_count = 0;
    if (emb_weight_scan_count < 1) {
        emb_weight_scan_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        const std::size_t total = static_cast<std::size_t>(emb.nelem());
        const std::size_t sample = std::min<std::size_t>(4096, total);
        log_tensor_sample_stats("EMB_WT_SAMPLE0", emb, 0, sample);
        if (total > sample) {
            log_tensor_sample_stats("EMB_WT_SAMPLE_MID", emb, total / 2, sample);
            log_tensor_sample_stats("EMB_WT_SAMPLE_END", emb, total - sample, sample);
        }
    }

    // Token-ID bounds check and NaN-sample mapping
    static int emb_token_check_count = 0;
    if (emb_token_check_count < 5) {
        emb_token_check_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        const long BT = static_cast<long>(token_ids.nelem());
        if (BT > 0 && token_ids.Data) {
            std::vector<std::int32_t> tokens(static_cast<std::size_t>(BT));
            cudaMemcpy(tokens.data(), token_ids.Data, static_cast<std::size_t>(BT) * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost);
            std::int32_t min_tok = std::numeric_limits<std::int32_t>::max();
            std::int32_t max_tok = std::numeric_limits<std::int32_t>::min();
            int oob_count = 0;
            int first_oob_idx = -1;
            std::int32_t first_oob_val = 0;
            const int vocab = static_cast<int>(mConfig.VocabSize);
            for (long i = 0; i < BT; ++i) {
                const std::int32_t v = tokens[static_cast<std::size_t>(i)];
                if (v < min_tok) min_tok = v;
                if (v > max_tok) max_tok = v;
                if (v < 0 || v >= vocab) {
                    oob_count++;
                    if (first_oob_idx < 0) {
                        first_oob_idx = static_cast<int>(i);
                        first_oob_val = v;
                    }
                }
            }
            const std::int32_t tok3 = (BT > 3) ? tokens[3] : tokens[0];
            fprintf(stderr,
                    "[EMBED_TOKENS] BT=%ld vocab=%d min=%d max=%d oob=%d token3=%d\n",
                    BT, vocab, min_tok, max_tok, oob_count, tok3);
            if (oob_count > 0) {
                fprintf(stderr,
                        "[EMBED_TOKENS_OOB] first_idx=%d first_val=%d\n",
                        first_oob_idx, first_oob_val);
            }
        }
    }

    // One-time prefill of embedding output with NaN pattern to detect unwritten elements
    static int emb_prefill_count = 0;
    if (emb_prefill_count < 1) {
        emb_prefill_count++;
        cudaMemsetAsync(out.Data, 0xFF, out.bytes(), mRunState.MainStream);
        fprintf(stderr, "[EMBED_PREFILL] out.Data=%p bytes=%zu\n", out.Data, out.bytes());
    }

    // DEBUG: Print actual tokens being used by embedding
    static int emb_debug_count = 0;
    if (emb_debug_count < 20) {
        emb_debug_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<std::int32_t> toks(4);
        cudaMemcpy(toks.data(), token_ids.Data, 4 * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[EMBEDDING] token_ids.Data=%p (rs.Inputs=%p) tokens[0..3]=%d,%d,%d,%d\n",
                token_ids.Data, mRunState.Inputs.Data, toks[0], toks[1], toks[2], toks[3]);
    }

    encoder_forward(out, token_ids, emb, std::nullopt,
                    static_cast<int>(mB), static_cast<int>(mT),
                    mConfig.HiddenSize, mConfig.VocabSize, mRunState.MainStream);

    // DEBUG: Print embedding output after encoder_forward
    // Print at offset 3*C to see embedding of token[3] which varies across micro-steps
    static int emb_out_count = 0;
    if (emb_out_count < 20) {
        emb_out_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> vals(4);
        const std::size_t offset = 3 * static_cast<std::size_t>(mConfig.HiddenSize);
        cudaMemcpy(vals.data(), reinterpret_cast<float*>(out.Data) + offset, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[EMBEDDING_OUT] out.Data=%p token[3] embed vals=%.6f,%.6f,%.6f,%.6f\n",
                out.Data, vals[0], vals[1], vals[2], vals[3]);
        const std::size_t row_elems = static_cast<std::size_t>(mConfig.HiddenSize);
        const std::size_t row_sample = std::min<std::size_t>(row_elems, 256);
        log_tensor_sample_stats("EMBED_OUT_ROW3", out, offset, row_sample);
    }

    // NaN detection for embedding output (token 3)
    log_nan_sample("FWD_EMBED_OUT", -1, op.outputs[0].name, out, 3);
    if (tensor_sample_has_nan_or_inf(out, 3)) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<std::int32_t> toks(4);
        cudaMemcpy(toks.data(), token_ids.Data, 4 * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
        const std::int32_t tok3 = toks[3];
        fprintf(stderr, "[EMBED_NAN_TOKEN] token3_id=%d\n", tok3);
        // Log full-row stats for the embedding weight and output row at token3.
        if (tok3 >= 0 && tok3 < static_cast<std::int32_t>(mConfig.VocabSize)) {
            const std::size_t row_elems = static_cast<std::size_t>(mConfig.HiddenSize);
            const std::size_t emb_offset = static_cast<std::size_t>(tok3) * row_elems;
            const std::size_t out_offset = 3 * row_elems;
            log_tensor_sample_stats("EMB_WT_ROW_TOK3", emb, emb_offset, row_elems);
            log_tensor_sample_stats("EMB_OUT_ROW3_FULL", out, out_offset, row_elems);
        }
        throw std::runtime_error("Embedding output contains NaNs; aborting.");
    }
}

void CompiledExecutor::dispatch_embedding_backward(const CompiledOp& op) {
    // Skip embedding backward entirely in LoRA-only mode
    if (mRunState.is_lora_only_mode()) {
        return;
    }

    // inputs: d_encoded, token_ids
    // outputs: d_embedding (sparse update)
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    if (op.outputs.empty() || op.outputs[0].name.empty()) {
        return;  // Skip if no output expected
    }

    // Get the pre-allocated gradient tensor
    auto it = mTensorMap.find(op.outputs[0].name);
    if (it == mTensorMap.end()) {
        // Gradient not allocated (embedding frozen in LoRA mode)
        return;
    }
    Tensor& d_emb = it->second;

    // encoder_backward requires CPU-side inputs for deterministic bucketing
    if (!mLastInputsCpu || !mLastInputsCpu->Data) {
        throw std::runtime_error("CompiledExecutor: embedding_backward requires CPU inputs (set_last_inputs_cpu)");
    }

    static int emb_bwd_log_count = 0;
    const int vocab = mConfig.VocabSize;
    const int total_tokens = static_cast<int>(mB * mT);
    const long hidden = (d_emb.Rank > 1) ? d_emb.Sizes[1] : 0;

    if (emb_bwd_log_count < 16) {
        int cpu_min = std::numeric_limits<int>::max();
        int cpu_max = std::numeric_limits<int>::min();
        int cpu_oob = 0;
        int cpu_oob_samples = 0;
        int token0 = 0;
        int token3 = 0;
        if (mLastInputsCpu->DType == ETensorDType::INT32) {
            const int* cpu = reinterpret_cast<const int*>(mLastInputsCpu->Data);
            for (int i = 0; i < total_tokens; ++i) {
                const int v = cpu[i];
                if (i == 0) token0 = v;
                if (i == 3) token3 = v;
                cpu_min = std::min(cpu_min, v);
                cpu_max = std::max(cpu_max, v);
                if (v < 0 || v >= vocab) {
                    if (cpu_oob_samples < 8) {
                        fprintf(stderr, "[EMB_BWD_CPU_OOB] idx=%d val=%d vocab=%d\n", i, v, vocab);
                        cpu_oob_samples++;
                    }
                    cpu_oob++;
                }
            }
        } else {
            fprintf(stderr, "[EMB_BWD_CPU_INPUTS] unsupported dtype=%s\n", dtype_to_str(mLastInputsCpu->DType));
        }

        fprintf(stderr,
                "[EMB_BWD_CPU_INPUTS] B=%d T=%d vocab=%d min=%d max=%d oob=%d token0=%d token3=%d\n",
                static_cast<int>(mB), static_cast<int>(mT), vocab,
                cpu_min, cpu_max, cpu_oob, token0, token3);

        if (mRunState.Inputs.Data && mRunState.Inputs.DType == ETensorDType::INT32 && total_tokens > 0) {
            const int sample = std::min(total_tokens, 8);
            std::vector<int> gpu(sample, 0);
            CUDA_CHECK(cudaMemcpyAsync(gpu.data(), mRunState.Inputs.Data, sample * sizeof(int),
                                       cudaMemcpyDeviceToHost, mRunState.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            int mismatch = 0;
            if (mLastInputsCpu->DType == ETensorDType::INT32) {
                const int* cpu = reinterpret_cast<const int*>(mLastInputsCpu->Data);
                for (int i = 0; i < sample; ++i) {
                    if (cpu[i] != gpu[i]) {
                        mismatch++;
                    }
                }
            }
            fprintf(stderr, "[EMB_BWD_INPUTS_CMP] sample=%d mismatch=%d\n", sample, mismatch);
        }

        log_nan_sample("BWD_DOUT", op.inputs[0].layer_idx, op.inputs[0].name, d_out, 3);
        if (tensor_sample_has_nan_or_inf(d_out, 3)) {
            log_tensor_stats_ex("BWD_DOUT_NAN", op.inputs[0].layer_idx, op.inputs[0].name, d_out, 4096, true);
        }

        if (hidden > 0) {
            if (tensor_sample_has_nan_or_inf(d_emb, 3)) {
                const std::size_t off = static_cast<std::size_t>(3) * static_cast<std::size_t>(hidden);
                log_tensor_sample_stats("EMB_DGRAD_PRE_ROW3", d_emb, off, static_cast<std::size_t>(hidden));
            }
            if (token3 >= 0 && token3 < vocab && tensor_sample_has_nan_or_inf(d_emb, token3)) {
                const std::size_t off = static_cast<std::size_t>(token3) * static_cast<std::size_t>(hidden);
                log_tensor_sample_stats("EMB_DGRAD_PRE_ROW_TOK3", d_emb, off, static_cast<std::size_t>(hidden));
            }
        }

        emb_bwd_log_count++;
    }

    unsigned int seed = mRngSeedFn ? mRngSeedFn() : 0;

    encoder_backward(d_emb,
                     mRunState.scratch().encoder_bwd_scratch,
                     mRunState.scratch().encoder_bwd_indices,
                     mRunState.scratch().encoder_bwd_info,
                     d_out,
                     mRunState.Inputs,
                     *mLastInputsCpu,
                     static_cast<int>(mB), static_cast<int>(mT), mConfig.HiddenSize,
                     seed,
                     mRunState.MainStream,
                     mRunState.side_stream_event(),
                     mRunState.side_stream());

    if (emb_bwd_log_count < 32 && hidden > 0) {
        int token3 = 0;
        if (mLastInputsCpu->DType == ETensorDType::INT32) {
            const int* cpu = reinterpret_cast<const int*>(mLastInputsCpu->Data);
            if (total_tokens > 3) {
                token3 = cpu[3];
            }
        }
        if (token3 >= 0 && token3 < vocab) {
            if (tensor_sample_has_nan_or_inf(d_emb, token3)) {
                const std::size_t off = static_cast<std::size_t>(token3) * static_cast<std::size_t>(hidden);
                log_tensor_sample_stats("EMB_DGRAD_ROW_TOK3", d_emb, off, static_cast<std::size_t>(hidden));
            }
        }
        if (tensor_sample_has_nan_or_inf(d_emb, 3)) {
            const std::size_t off = static_cast<std::size_t>(3) * static_cast<std::size_t>(hidden);
            log_tensor_sample_stats("EMB_DGRAD_ROW3", d_emb, off, static_cast<std::size_t>(hidden));
        }
        emb_bwd_log_count++;
    }
}

}  // namespace dsl
