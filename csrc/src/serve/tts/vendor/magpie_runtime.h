// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nemo_speech::tts {

enum class MagpieBackendPreference {
    Auto,
    Cpu,
    Cuda,
};

enum class MagpieUmaMode {
    Auto,
    Off,
    On,
};

enum class MagpieLongformMode {
    Auto,
    Off,
    On,
};

struct MagpieRuntimeConfig {
    std::string magpie_model;
    std::string codec_model;
    int speaker = 0;
    int threads = 4;
    int codec_threads = 0;
    int seed = -1;
    int steps = -1;
    int top_k = -1;
    int chunk_frames = 4;
    int codec_queue_depth = 4;
    int codec_history_frames = -1;
    int codec_future_frames = 1;
    int window_ms = 0;
    float temperature = 0.0f;
    float cfg_scale = 0.0f;
    bool override_temperature = false;
    bool override_cfg_scale = false;
    bool use_cfg = true;
    bool use_local_transformer = true;
    bool lt_fp32 = false;
    bool use_kv_cache = true;
    bool use_stateful_codec = true;
    bool magpie_cpu = false;
    bool codec_cpu = false;
    bool flush_partial_chunk = true;
    bool verbose = false;
    MagpieBackendPreference lt_backend = MagpieBackendPreference::Auto;
    MagpieBackendPreference sampling_backend = MagpieBackendPreference::Auto;
    MagpieUmaMode uma_mode = MagpieUmaMode::Auto;
    MagpieLongformMode longform_mode = MagpieLongformMode::Auto;
};

struct MagpieSynthesisOptions {
    int speaker = -1;
    int seed = -1;
    int steps = -1;
    int top_k = -1;
    float temperature = 0.0f;
    float cfg_scale = 0.0f;
    bool override_temperature = false;
    bool override_cfg_scale = false;
};

struct MagpieSynthesisStats {
    int sample_rate = 0;
    int generated_frames = 0;
    int chunks = 0;
    int e2e_chunks = 0;
    uint64_t samples_written = 0;
    double tokenizer_ms = 0.0;
    double encoder_ms = 0.0;
    double audio_s = 0.0;
    double elapsed_s = 0.0;
    double rtf = 0.0;
    double rtfx = 0.0;
    double ttfa_ms = 0.0;
    double icl_avg_ms = 0.0;
    double icl_min_ms = 0.0;
    double icl_max_ms = 0.0;
    double decoder_audio_s = 0.0;
    double decoder_elapsed_s = 0.0;
    double decoder_rtfx = 0.0;
    double decoder_ttft_ms = 0.0;
    double decoder_itl_avg_ms = 0.0;
    double decoder_itl_min_ms = 0.0;
    double decoder_itl_max_ms = 0.0;
    double decoder_itl_p95_ms = 0.0;
    double decoder_itl_p99_ms = 0.0;
    double codec_audio_s = 0.0;
    double codec_elapsed_s = 0.0;
    double codec_rtfx = 0.0;
    double codec_ttfa_ms = 0.0;
    double codec_icl_avg_ms = 0.0;
    double codec_icl_min_ms = 0.0;
    double codec_icl_max_ms = 0.0;
    double codec_icl_p95_ms = 0.0;
    double codec_icl_p99_ms = 0.0;
    double e2e_ttfa_ms = 0.0;
    double e2e_icl_avg_ms = 0.0;
    double e2e_icl_min_ms = 0.0;
    double e2e_icl_max_ms = 0.0;
    double e2e_icl_p95_ms = 0.0;
    double e2e_icl_p99_ms = 0.0;
    double e2e_rtfx = 0.0;
};

class MagpieTtsRuntime {
   public:
    using PcmCallback = std::function<bool(const std::string&)>;

    explicit MagpieTtsRuntime(MagpieRuntimeConfig config);
    ~MagpieTtsRuntime();

    MagpieTtsRuntime(const MagpieTtsRuntime&) = delete;
    MagpieTtsRuntime& operator=(const MagpieTtsRuntime&) = delete;

    int sample_rate() const;
    int speaker_count() const;
    const std::vector<std::string>& speaker_names() const;
    const std::string& model_name() const;
    const std::string& tokenizer_profile() const;
    int text_vocab_size() const;

    MagpieSynthesisStats synthesize(
        const std::vector<int32_t>& tokens, const MagpieSynthesisOptions& options,
        const PcmCallback& pcm_callback);
    MagpieSynthesisStats synthesize(
        const std::vector<std::vector<int32_t>>& token_chunks,
        const MagpieSynthesisOptions& options, const PcmCallback& pcm_callback);

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nemo_speech::tts
