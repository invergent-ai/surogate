#pragma once

#include "family/impl/frontend/chat_template.h"
#include "family/impl/frontend/tokenizer.h"

#include <api/family/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sinfer::family::frontend_internal {

class MediaPreprocessCache;

enum class ProcessorErrorKind {
    BudgetExceeded,
};

class ProcessorError final : public std::runtime_error {
public:
    ProcessorError(ProcessorErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ProcessorErrorKind kind() const noexcept { return kind_; }

private:
    ProcessorErrorKind kind_;
};

enum class Modality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct VisionGrid {
    int t = 0;
    int h = 0;
    int w = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    Modality modality = Modality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    std::array<std::uint8_t, 32> content_digest{};
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

struct PreprocessStats {
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::uint64_t attention_pairs        = 0;
    std::size_t prompt_tokens            = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;

    [[nodiscard]] std::string summary() const;
};

struct GemmaProcessorOptions {
    int version = 0, patch = 0, merge = 0, image_tokens = 0, video_tokens = 70;
    int image_size = 896, position_embeddings = 0, video_token_id = 0, resample = 2;
    bool encoder_free = false, pan_and_scan = false;
    int min_crop_size = 256, max_crops = 4;
    double crop_ratio = 1.2;
    std::string image_token, video_token, boi_token, eoi_token;
};

struct ProcessorOptions {
    GemmaProcessorOptions gemma;
    bool lfm2_vl = false;
    int image_token_id = 0;
    int lfm_min_tokens = 64;
    int lfm_max_tokens = 256;
    int lfm_min_tiles = 2;
    int lfm_max_tiles = 10;
    int lfm_tile_size = 512;
    int lfm_resample = 3;
    double lfm_pixels_tolerance = 2.0;
    bool lfm_thumbnail = true;
    bool lfm_splitting = true;
    bool lfm_special_tokens = true;
    std::uint64_t image_min_pixels = 32ULL * 32ULL;
    std::uint64_t image_max_pixels = 1024ULL * 1024ULL;
    std::uint64_t video_min_pixels = 128ULL * 32ULL * 32ULL;
    std::uint64_t video_max_pixels = 4ULL * 1024ULL * 1024ULL;
    // Encoded bytes are aggregate per prompt. Decode limits are per item; the fixed worker pool
    // bounds concurrently decoded media.
    std::size_t max_encoded_media_bytes    = kMaximumPromptMediaBytes;
    std::uint64_t max_decoded_pixels       = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    int max_video_source_frames            = 100'000;
    double max_video_duration_seconds      = 600.0;
    std::uint64_t max_raw_patches          = kMaximumVisionRawPatches;
    std::uint64_t max_vision_tokens        = kMaximumVisionTokens;
    double video_fps                       = 2.0;
    int video_min_frames                   = 4;
    int video_max_frames                   = 768;
};

struct ProcessedInput {
    std::vector<int> input_ids;
    std::vector<std::uint8_t> token_types;
    // Axis-major [3, input_ids.size()] in temporal, height, width order.
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    std::vector<VisionItem> vision_items;
    // One immutable row-major [raw_patches, checkpoint patch_dim] payload per Vision item.
    std::vector<std::shared_ptr<const family::PreparedMediaPayload>> media_payloads;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    PreprocessStats stats;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;
};

struct EncodedChat {
    std::vector<int> input_ids;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
};

ProcessedInput process_gemma_vl(const Tokenizer& tokenizer, const ProcessorOptions& options,
    MediaPreprocessCache& cache, std::vector<ChatMessage> messages, ChatRenderOptions render_options,
    const PreparationControl& control);

ProcessedInput process_lfm2_vl(const Tokenizer& tokenizer, const ProcessorOptions& options,
    MediaPreprocessCache& cache, std::vector<ChatMessage> messages, ChatRenderOptions render_options,
    const PreparationControl& control);

EncodedChat encode_rendered_chat(const Tokenizer& tokenizer, const RenderedChat& rendered);

class Processor {
public:
    Processor(const Tokenizer& tokenizer, const CompiledChatTemplate& chat_template,
              ProcessorOptions options, std::shared_ptr<MediaPreprocessCache> media_cache);

    ProcessedInput process(std::vector<ChatMessage> messages, ChatRenderOptions render_options = {},
                           const PreparationControl& control = {},
                           std::optional<RenderedChat> prepared_chat = std::nullopt) const;

private:
    int image_token_id_ = -1, video_token_id_ = -1;
    const Tokenizer& tokenizer_;
    const CompiledChatTemplate& chat_template_;
    ProcessorOptions options_;
    std::shared_ptr<MediaPreprocessCache> media_cache_;
};

} // namespace sinfer::family::frontend_internal
