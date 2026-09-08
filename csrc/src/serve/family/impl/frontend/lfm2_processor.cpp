#include "family/impl/frontend/processor.h"
#include "family/impl/frontend/media_cache.h"
#include "family/impl/frontend/digest.h"
#include "media/decode/decode.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace sinfer::family::frontend_internal {

media::decode::Image resize_processor_image(const media::decode::Image&, int height, int width,
                                            bool bilinear, const PreparationControl&);

namespace {
using Clock          = std::chrono::steady_clock;
using Image          = media::decode::Image;
constexpr int kPatch = 16, kMerge = 2, kFeatures = 3 * kPatch * kPatch;

int aligned(int extent, int minimum) {
    return std::max(minimum, static_cast<int>(std::nearbyint(extent / 32.0)) * 32);
}

std::pair<int, int> image_size(int h, int w, const ProcessorOptions& options) {
    const double area    = static_cast<double>(h) * w;
    const double minimum = options.lfm_min_tokens * 1024.0;
    const double maximum = options.lfm_max_tokens * 1024.0;
    int height = aligned(h, 32), width = aligned(w, 32);
    if (static_cast<double>(height) * width > maximum) {
        const double scale = std::sqrt(area / maximum);
        height             = std::max(32, static_cast<int>(std::floor(h / scale / 32)) * 32);
        width              = std::max(32, static_cast<int>(std::floor(w / scale / 32)) * 32);
    } else if (static_cast<double>(height) * width < minimum) {
        const double scale = std::sqrt(minimum / area);
        height             = static_cast<int>(std::ceil(h * scale / 32)) * 32;
        width              = static_cast<int>(std::ceil(w * scale / 32)) * 32;
    }
    return {height, width};
}

std::pair<int, int> tile_grid(int h, int w, const ProcessorOptions& options) {
    if (!options.lfm_splitting ||
        static_cast<double>(aligned(h, 16)) * aligned(w, 16) <=
            options.lfm_max_tokens * 1024.0 * options.lfm_pixels_tolerance) {
        return {1, 1};
    }
    std::vector<std::pair<int, int>> candidates;
    for (int rows = 1; rows <= options.lfm_max_tiles; ++rows) {
        for (int cols = 1; cols <= options.lfm_max_tiles; ++cols) {
            if (rows * cols >= options.lfm_min_tiles && rows * cols <= options.lfm_max_tiles) {
                candidates.emplace_back(rows, cols);
            }
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](auto a, auto b) { return a.first * a.second < b.first * b.second; });
    const double ratio = static_cast<double>(w) / h;
    double best        = std::numeric_limits<double>::infinity();
    std::pair<int, int> grid{1, 1};
    for (auto [rows, cols] : candidates) {
        const double distance = std::abs(ratio - static_cast<double>(cols) / rows);
        const double target_area =
            static_cast<double>(options.lfm_tile_size) * options.lfm_tile_size * rows * cols;
        if (distance < best ||
            (distance == best && static_cast<double>(w) * h > 0.5 * target_area)) {
            best = distance;
            grid = {rows, cols};
        }
    }
    return grid;
}

std::uint16_t bf16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

std::string role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
        return "system";
    case ChatRole::Developer:
        return "developer";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("unknown chat role");
}

} // namespace

ProcessedInput process_lfm2_vl(const Tokenizer& tokenizer, const ProcessorOptions& options,
                               MediaPreprocessCache& cache, std::vector<ChatMessage> messages,
                               ChatRenderOptions render_options,
                               const PreparationControl& control) {
    auto permit      = cache.acquire_request(control);
    const auto start = Clock::now();
    ProcessedInput out;
    auto& stats = out.stats;
    MediaCacheRequestStats cache_stats;
    auto structured = nlohmann::ordered_json::array();
    for (auto& message : messages) {
        std::string content;
        for (auto& part : message.parts) {
            check_preparation_control(control);
            if (part.kind == ChatPartKind::Text) {
                content += part.text;
                continue;
            }
            if (part.kind != ChatPartKind::Image) {
                throw std::invalid_argument("LFM2-VL supports images; video input is unsupported");
            }
            if (part.media.bytes.size() > options.max_encoded_media_bytes - stats.media_bytes) {
                throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                                     "request media bytes exceed processor budget");
            }
            stats.media_bytes += part.media.bytes.size();
            ++stats.media_items;
            const media::decode::Policy policy{
                .max_bytes          = options.max_encoded_media_bytes,
                .max_decoded_pixels = options.max_decoded_pixels,
                .checkpoint         = [&control] { check_preparation_control(control); },
            };
            auto image                 = media::decode::decode_image(part.media.bytes, policy);
            const auto digest          = sha256(part.media.bytes);
            const auto [height, width] = image_size(image.height, image.width, options);
            const auto [rows, cols]    = tile_grid(image.height, image.width, options);
            const bool split           = rows > 1 || cols > 1;
            const bool bilinear        = options.lfm_resample == 2;
            Image grid;
            if (split) {
                const auto resized_pixels = static_cast<std::uint64_t>(rows) * cols *
                                            options.lfm_tile_size * options.lfm_tile_size;
                if (resized_pixels > options.max_decoded_pixels) {
                    throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                                         "LFM2-VL tile grid exceeds pixel budget");
                }
                grid = resize_processor_image(image, rows * options.lfm_tile_size,
                                              cols * options.lfm_tile_size, bilinear, control);
            }
            if (options.lfm_special_tokens) { content += "<|image_start|>"; }
            const auto append = [&](const Image& source, int top, int left, int h, int w,
                                    const std::string& label) {
                const auto patches = static_cast<std::uint64_t>(h / kPatch) * (w / kPatch);
                const auto tokens  = patches / (kMerge * kMerge);
                if (patches > options.max_raw_patches - stats.raw_patches ||
                    tokens > options.max_vision_tokens - stats.vision_tokens) {
                    throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                                         "LFM2-VL images exceed prompt budget");
                }
                std::vector<std::uint8_t> identity(digest.begin(), digest.end());
                const std::string suffix =
                    label + ":" + std::to_string(h) + ":" + std::to_string(w);
                identity.insert(identity.end(), suffix.begin(), suffix.end());
                const auto tile_digest = sha256(identity);
                const MediaCacheKey key{.digest = tile_digest, .modality = Modality::Image};
                auto media = cache.get_or_prepare(
                    key, control,
                    [&] {
                        auto payload       = cache.allocate_payload(patches * kFeatures, control);
                        std::size_t cursor = 0;
                        for (int by = 0; by < h / 32; ++by) {
                            check_preparation_control(control);
                            for (int bx = 0; bx < w / 32; ++bx) {
                                for (int iy = 0; iy < kMerge; ++iy)
                                    for (int ix = 0; ix < kMerge; ++ix) {
                                        for (int py = 0; py < kPatch; ++py)
                                            for (int px = 0; px < kPatch; ++px) {
                                                const auto offset =
                                                    (static_cast<std::size_t>(top + by * 32 +
                                                                              iy * kPatch + py) *
                                                         source.width +
                                                     left + bx * 32 + ix * kPatch + px) *
                                                    3;
                                                for (int channel = 0; channel < 3; ++channel) {
                                                    const float value =
                                                        (source.rgb[offset + channel] *
                                                             (1.0F / 255.0F) -
                                                         0.5F) /
                                                        0.5F;
                                                    payload->patches[cursor++] = bf16(value);
                                                }
                                            }
                                    }
                            }
                        }
                        VisionItem item;
                        item.grid           = {1, h / kPatch, w / kPatch};
                        item.content_digest = tile_digest;
                        item.patch_count    = patches;
                        return PreparedMedia{std::move(item), std::move(payload)};
                    },
                    cache_stats);
                media.item.patch_begin = stats.raw_patches;
                stats.raw_patches += patches;
                stats.vision_tokens += tokens;
                stats.attention_pairs += patches * patches;
                stats.patch_bytes += patches * kFeatures * sizeof(std::uint16_t);
                out.vision_items.push_back(std::move(media.item));
                out.media_payloads.push_back(std::move(media.payload));
                for (std::uint64_t i = 0; i < tokens; ++i) { content += "<image>"; }
            };
            if (split) {
                for (int row = 0; row < rows; ++row)
                    for (int col = 0; col < cols; ++col) {
                        const auto label = "<|img_row_" + std::to_string(row + 1) + "_col_" +
                                           std::to_string(col + 1) + "|>";
                        if (options.lfm_special_tokens) { content += label; }
                        append(grid, row * options.lfm_tile_size, col * options.lfm_tile_size,
                               options.lfm_tile_size, options.lfm_tile_size, label);
                    }
            }
            if (!split || options.lfm_thumbnail) {
                if (split && options.lfm_special_tokens) { content += "<|img_thumbnail|>"; }
                const auto pixels = static_cast<std::uint64_t>(height) * width;
                if (pixels > options.max_decoded_pixels ||
                    pixels / (kPatch * kPatch) > options.max_raw_patches - stats.raw_patches ||
                    pixels / 1024 > options.max_vision_tokens - stats.vision_tokens) {
                    throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                                         "LFM2-VL resized image exceeds processor budget");
                }
                auto thumbnail = resize_processor_image(image, height, width, bilinear, control);
                append(thumbnail, 0, 0, height, width, "thumbnail");
            }
            if (options.lfm_special_tokens) { content += "<|image_end|>"; }
            std::vector<std::uint8_t>().swap(part.media.bytes);
        }
        nlohmann::ordered_json item{{"role", role_name(message.role)}, {"content", content}};
        if (!message.reasoning_content.empty()) {
            item["reasoning_content"] = message.reasoning_content;
        }
        if (!message.tool_call_id.empty()) { item["tool_call_id"] = message.tool_call_id; }
        for (const auto& call : message.tool_calls) {
            item["tool_calls"].push_back(
                {{"id", call.id},
                 {"type", "function"},
                 {"function",
                  {{"name", call.name},
                   {"arguments", nlohmann::ordered_json::parse(call.arguments_json)}}}});
        }
        structured.push_back(std::move(item));
    }
    stats.media_preprocess_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    permit.reset();
    const auto tokenize_start = Clock::now();
    RenderedChat rendered{tokenizer.render_chat_template_json(
        structured.dump(), render_options.tool_jsons, render_options.add_generation_prompt)};
    out.input_ids = encode_rendered_chat(tokenizer, rendered).input_ids;
    out.token_types.resize(out.input_ids.size(), 0);
    std::size_t cursor = 0;
    for (auto& item : out.vision_items) {
        while (cursor < out.input_ids.size() && out.input_ids[cursor] != options.image_token_id) {
            ++cursor;
        }
        const auto count = item.patch_count / (kMerge * kMerge);
        if (count > out.input_ids.size() - cursor) {
            throw std::invalid_argument("LFM2-VL image tokens are missing from the prompt");
        }
        item.token_spans.push_back({cursor, count});
        for (std::size_t i = 0; i < count; ++i) {
            if (out.input_ids[cursor + i] != options.image_token_id) {
                throw std::invalid_argument("LFM2-VL image tokens do not match the image grid");
            }
            out.token_types[cursor + i] = static_cast<std::uint8_t>(Modality::Image);
        }
        cursor += count;
    }
    for (std::size_t i = 0; i < out.input_ids.size(); ++i) {
        if (out.input_ids[i] == options.image_token_id && !out.token_types[i]) {
            throw std::invalid_argument("LFM2-VL prompt has an image token without an image");
        }
    }
    out.positions.resize(3 * out.input_ids.size());
    for (int axis = 0; axis < 3; ++axis) {
        std::iota(out.positions.begin() + axis * out.input_ids.size(),
                  out.positions.begin() + (axis + 1) * out.input_ids.size(), 0);
    }
    stats.tokenize_seconds   = std::chrono::duration<double>(Clock::now() - tokenize_start).count();
    stats.prompt_tokens      = out.input_ids.size();
    stats.media_cache_hits   = cache_stats.hits;
    stats.media_cache_misses = cache_stats.misses;
    stats.media_singleflight_waits      = cache_stats.singleflight_waits;
    stats.built_patch_bytes             = cache_stats.built_patch_bytes;
    stats.reused_patch_bytes            = cache_stats.reused_patch_bytes;
    stats.media_preprocess_work_seconds = cache_stats.build_seconds;
    check_preparation_control(control);
    return out;
}

} // namespace sinfer::family::frontend_internal
