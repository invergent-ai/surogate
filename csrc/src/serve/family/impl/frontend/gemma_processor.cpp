#include "family/impl/frontend/processor.h"
#include "family/impl/frontend/media_cache.h"
#include "family/impl/frontend/digest.h"
#include "media/decode/decode.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace sinfer::family::frontend_internal {
media::decode::Image resize_processor_image(const media::decode::Image&, int, int, bool,
                                            const PreparationControl&);

namespace {
using Image = media::decode::Image;
using Clock = std::chrono::steady_clock;

std::uint16_t bf16(float v) {
    auto b = std::bit_cast<std::uint32_t>(v);
    return (b + 0x7fffU + ((b >> 16) & 1U)) >> 16;
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

std::pair<int, int> image_size(const Image& image, const GemmaProcessorOptions& g, int tokens) {
    if (g.version == 3) { return {g.image_size, g.image_size}; }
    const int side = g.patch * g.merge;
    const double factor =
        std::sqrt(double(tokens) * side * side / (double(image.height) * image.width));
    int h = int(std::floor(factor * image.height / side)) * side;
    int w = int(std::floor(factor * image.width / side)) * side;
    if (!h) {
        h = side;
        w = std::min(image.width / image.height, tokens) * side;
    }
    if (!w) {
        w = side;
        h = std::min(image.height / image.width, tokens) * side;
    }
    return {h, w};
}

std::vector<Image> crops(const Image& image, const GemmaProcessorOptions& g) {
    std::vector<Image> result;
    const bool wide   = image.width >= image.height;
    const int longer  = wide ? image.width : image.height,
              shorter = wide ? image.height : image.width;
    if (!g.pan_and_scan || double(longer) / shorter < g.crop_ratio) { return result; }
    const int count =
        std::min(g.max_crops, std::max(2, std::min(int(std::floor(double(longer) / shorter + .5)),
                                                   longer / g.min_crop_size)));
    const int extent = (longer + count - 1) / count;
    if (std::min(extent, shorter) < g.min_crop_size) { return result; }
    for (int i = 0; i < count; ++i) {
        const int top = wide ? 0 : i * extent, left = wide ? i * extent : 0;
        Image crop;
        crop.height = std::min(wide ? image.height : extent, image.height - top);
        crop.width  = std::min(wide ? extent : image.width, image.width - left);
        crop.rgb.resize(std::size_t(crop.height) * crop.width * 3);
        for (int y = 0; y < crop.height; ++y) {
            std::copy_n(image.rgb.data() + (std::size_t(top + y) * image.width + left) * 3,
                        crop.width * 3, crop.rgb.data() + std::size_t(y) * crop.width * 3);
        }
        result.push_back(std::move(crop));
    }
    return result;
}
} // namespace

ProcessedInput process_gemma_vl(const Tokenizer& tokenizer, const ProcessorOptions& options,
                                MediaPreprocessCache& cache, std::vector<ChatMessage> messages,
                                ChatRenderOptions render_options,
                                const PreparationControl& control) {
    auto permit      = cache.acquire_request(control);
    const auto start = Clock::now();
    const auto& g    = options.gemma;
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
            const bool video = part.kind == ChatPartKind::Video;
            if (part.media.bytes.size() > options.max_encoded_media_bytes - stats.media_bytes) {
                throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                                     "request media bytes exceed processor budget");
            }
            stats.media_bytes += part.media.bytes.size();
            ++stats.media_items;
            const media::decode::Policy policy{
                .max_bytes                  = options.max_encoded_media_bytes,
                .max_decoded_pixels         = options.max_decoded_pixels,
                .max_decoded_video_pixels   = options.max_decoded_video_pixels,
                .max_video_source_frames    = options.max_video_source_frames,
                .max_video_duration_seconds = options.max_video_duration_seconds,
                .checkpoint                 = [&] { check_preparation_control(control); }};
            std::vector<Image> frames;
            std::vector<double> times;
            if (video) {
                auto decoded =
                    media::decode::decode_video(part.media.bytes, policy, options.video_fps,
                                                options.video_min_frames, options.video_max_frames);
                for (int i : decoded.indices) { times.push_back(i / decoded.fps); }
                frames = std::move(decoded.frames);
            } else {
                frames.push_back(media::decode::decode_image(part.media.bytes, policy));
            }
            const auto source_digest = sha256(part.media.bytes);
            for (std::size_t frame = 0; frame < frames.size(); ++frame) {
                if (video) {
                    std::ostringstream timestamp;
                    timestamp << std::setfill('0') << std::setw(2) << int(times[frame] / 60) << ":"
                              << std::setw(2) << int(times[frame]) % 60 << " ";
                    content += timestamp.str();
                }
                const bool video_marker = video && g.version == 4;
                const int budget        = video_marker ? g.video_tokens : g.image_tokens;
                const auto append       = [&](const Image& source, std::size_t crop) {
                    const auto [height, width] = image_size(source, g, budget);
                    const auto patches = std::uint64_t(height / g.patch) * (width / g.patch);
                    const auto tokens  = patches / (g.merge * g.merge);
                    if (!patches || tokens > std::uint64_t(budget) ||
                        std::uint64_t(height) * width > options.max_decoded_pixels ||
                        (g.version == 4 && (height / g.patch > g.position_embeddings ||
                                            width / g.patch > g.position_embeddings)) ||
                        patches > options.max_raw_patches - stats.raw_patches ||
                        tokens > options.max_vision_tokens - stats.vision_tokens) {
                        throw ProcessorError(ProcessorErrorKind::BudgetExceeded,
                                                   "Gemma images exceed processor budget");
                    }
                    std::vector<std::uint8_t> identity(source_digest.begin(), source_digest.end());
                    const auto suffix = std::to_string(frame) + ":" + std::to_string(crop) + ":" +
                                        std::to_string(height) + ":" + std::to_string(width);
                    identity.insert(identity.end(), suffix.begin(), suffix.end());
                    const auto digest   = sha256(identity);
                    const auto modality = video_marker ? Modality::Video : Modality::Image;
                    const MediaCacheKey key{.digest = digest, .modality = modality};
                    auto media = cache.get_or_prepare(
                        key, control,
                        [&] {
                            const auto image = resize_processor_image(source, height, width,
                                                                            g.resample == 2, control);
                            auto payload =
                                cache.allocate_payload(patches * 3 * g.patch * g.patch, control);
                            std::size_t cursor = 0;
                            for (int by = 0; by < height / (g.patch * g.merge); ++by) {
                                check_preparation_control(control);
                                for (int bx = 0; bx < width / (g.patch * g.merge); ++bx)
                                    for (int iy = 0; iy < g.merge; ++iy)
                                        for (int ix = 0; ix < g.merge; ++ix) {
                                            const auto emit = [&](int y, int x, int c) {
                                                const auto offset =
                                                    (std::size_t((by * g.merge + iy) * g.patch +
                                                                       y) *
                                                         width +
                                                     (bx * g.merge + ix) * g.patch + x) *
                                                        3 +
                                                    c;
                                                float value = image.rgb[offset] * (1.0F / 255.0F);
                                                if (!g.encoder_free) {
                                                    value = 2.0F * (value - 0.5F);
                                                }
                                                payload->patches[cursor++] = bf16(value);
                                            };
                                            if (g.version == 3) {
                                                for (int c = 0; c < 3; ++c)
                                                    for (int y = 0; y < g.patch; ++y)
                                                        for (int x = 0; x < g.patch; ++x) {
                                                            emit(y, x, c);
                                                        }
                                            } else {
                                                for (int y = 0; y < g.patch; ++y)
                                                    for (int x = 0; x < g.patch; ++x)
                                                        for (int c = 0; c < 3; ++c) {
                                                            emit(y, x, c);
                                                        }
                                            }
                                        }
                            }
                            VisionItem item;
                            item.modality       = modality;
                            item.grid           = {1, height / g.patch, width / g.patch};
                            item.content_digest = digest;
                            item.patch_count    = patches;
                            if (video_marker) { item.timestamps = {times[frame]}; }
                            return PreparedMedia{std::move(item), std::move(payload)};
                        },
                        cache_stats);
                    media.item.patch_begin = stats.raw_patches;
                    stats.raw_patches += patches;
                    stats.vision_tokens += tokens;
                    stats.attention_pairs += patches * patches;
                    stats.patch_bytes += patches * 3 * g.patch * g.patch * 2;
                    out.vision_items.push_back(std::move(media.item));
                    out.media_payloads.push_back(std::move(media.payload));
                    if (g.version == 3) { content += "\n\n"; }
                    content += g.boi_token;
                    for (std::uint64_t i = 0; i < tokens; ++i) {
                        content += video_marker ? g.video_token : g.image_token;
                    }
                    content += g.eoi_token;
                    if (g.version == 3) { content += "\n\n"; }
                };
                auto extras = g.version == 3 ? crops(frames[frame], g) : std::vector<Image>{};
                if (!extras.empty()) { content += "Here is the original image "; }
                append(frames[frame], 0);
                if (!extras.empty()) {
                    content += " and here are some crops to help you see better ";
                    for (std::size_t i = 0; i < extras.size(); ++i) {
                        if (i) { content += " "; }
                        append(extras[i], i + 1);
                    }
                }
                if (video && frame + 1 < frames.size()) { content += " "; }
            }
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
        structured.dump(), render_options.tool_jsons, render_options.add_generation_prompt,
        render_options.template_variables)};
    out.input_ids = encode_rendered_chat(tokenizer, rendered).input_ids;
    out.token_types.resize(out.input_ids.size(), 0);
    std::size_t cursor = 0;
    for (auto& item : out.vision_items) {
        const int id = item.modality == Modality::Video ? g.video_token_id : options.image_token_id;
        while (cursor < out.input_ids.size() && out.input_ids[cursor] != id) { ++cursor; }
        const auto count = item.patch_count / (g.merge * g.merge);
        if (count > out.input_ids.size() - cursor) {
            throw std::invalid_argument("Gemma prompt is missing image tokens");
        }
        item.token_spans.push_back({cursor, count});
        for (std::size_t i = 0; i < count; ++i) {
            if (out.input_ids[cursor + i] != id) {
                throw std::invalid_argument("Gemma image tokens do not match the patch grid");
            }
            out.token_types[cursor + i] = static_cast<std::uint8_t>(item.modality);
        }
        cursor += count;
    }
    for (std::size_t i = 0; i < out.input_ids.size(); ++i) {
        if ((out.input_ids[i] == options.image_token_id || out.input_ids[i] == g.video_token_id) &&
            !out.token_types[i]) {
            throw std::invalid_argument("Gemma prompt contains an image/video token without media");
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
