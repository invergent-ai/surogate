// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_CONFIG_PRETRAINED_CONFIG_H
#define SUROGATE_SRC_CONFIG_PRETRAINED_CONFIG_H

#include <string_view>

#include "config/rope_config.h"
#include "utilities/dtype.h"

// Configuration describing a pretrained decoder-only transformer model.
// This mirrors the subset of HuggingFace config.json fields that the trainer/runtime needs.
struct PretrainedConfig {
    enum ArchitectureId {
        LLAMA,
        QWEN2,
        QWEN3,
    } Architecture;

    int BosTokenId;
    int EosTokenId;
    int PadTokenId;

    int HiddenSize;
    int IntermediateSize;
    int VocabSize;
    int NumQueryHeads;
    int NumKeyValHeads;
    int NumLayers;

    // Attention head dimension. If 0, defaults to HiddenSize / NumQueryHeads.
    // Some architectures (e.g., Qwen3) use an explicit head_dim that may not match HiddenSize/NumQueryHeads.
    int HeadDim = 0;

    int MaxPositionEmbeddings;
    float RopeTheta;
    RoPEConfig Rope;  // Flexible RoPE configuration (GLM4 partial, Qwen2-VL M-RoPE, etc.)
    float RmsNormEps;
    bool TiedWordEmbeddings;
    bool UseQKVBias;
    bool UseQKNorm = false;

    ETensorDType DType = ETensorDType::BF16;

    [[nodiscard]] int head_size() const { return HeadDim > 0 ? HeadDim : (HiddenSize / NumQueryHeads); }
    [[nodiscard]] int attn_out_channels() const { return head_size() * NumQueryHeads; }
    [[nodiscard]] int qkv_channels() const { return head_size() * (NumQueryHeads + 2 * NumKeyValHeads); }
    [[nodiscard]] std::string_view model_name() const;
};

PretrainedConfig load_pretrained_config(const char* file_name, ETensorDType dtype);
void save_pretrained_config(const PretrainedConfig& config, const char* file_name);
PretrainedConfig create_pretrained_config_from_name(std::string_view name, ETensorDType dtype);

#endif // SUROGATE_SRC_CONFIG_PRETRAINED_CONFIG_H
