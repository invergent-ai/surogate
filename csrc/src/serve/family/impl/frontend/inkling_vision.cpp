#include "family/impl/frontend/inkling_vision.h"

#include <array>
#include <bit>
#include <stdexcept>
#include <string>

namespace sinfer::family::frontend_internal::inkling {
namespace {

// nvidia/vision.py: _CLIP_MEAN, _PROCESSOR_CLIP_STD (gather_patches' clip_std), _CLIP_STD (its std). Python floats
// handed to a Triton kernel as fp32 scalars: double literal -> float, as the kernel launch converts them.
constexpr std::array<double, 3> kMean     = {0.48145466, 0.4578275, 0.40821073};
constexpr std::array<double, 3> kClipStd  = {0.26862954, 0.2613026, 0.2757771};
constexpr std::array<double, 3> kStd      = {0.26862954, 0.26130258, 0.27577711};

std::uint16_t bf16_bits(float value) noexcept {
    // cvt.rn.bf16.f32 / the bf16 store: round to nearest even (no NaN reaches here).
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

float bf16_value(std::uint16_t bits) noexcept { return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U); }

// gather_patches for one value, each operation rounded to fp32 on its own: the kernel runs with
// enable_fp_fusion=False, so no product may be contracted into an FMA here either (volatile stops the compiler).
std::uint16_t normalize(int raw, int channel) noexcept {
    const float mean     = static_cast<float>(kMean[static_cast<std::size_t>(channel)]);
    const float clip_std = static_cast<float>(kClipStd[static_cast<std::size_t>(channel)]);
    const float std_     = static_cast<float>(kStd[static_cast<std::size_t>(channel)]);
    volatile float v = static_cast<float>(raw);
    v                = v / 255.0F;
    v                = v - mean;
    v                = v / clip_std;
    v                = bf16_value(bf16_bits(v));
    v                = v * std_;
    v                = v + mean;
    v                = 2.0F * v;
    v                = v - 1.0F;
    return bf16_bits(v);
}

// [channel][raw + 1] for raw in -1..255.
const std::array<std::array<std::uint16_t, 257>, 3>& normalization_table() {
    static const auto table = [] {
        std::array<std::array<std::uint16_t, 257>, 3> out{};
        for (int c = 0; c < 3; ++c) {
            for (int raw = -1; raw <= 255; ++raw) {
                out[static_cast<std::size_t>(c)][static_cast<std::size_t>(raw + 1)] = normalize(raw, c);
            }
        }
        return out;
    }();
    return table;
}

// The raw value vLLM's padded image holds at (y, x): the pixel inside the image, -1 outside.
int raw_at(const DecodedImage& image, int y, int x, int c) noexcept {
    if (y >= image.height || x >= image.width) { return -1; }
    return image.rgb[(static_cast<std::size_t>(y) * image.width + x) * 3 + static_cast<std::size_t>(c)];
}

} // namespace

DecodedImage composite(const media::decode::PilImage& image) {
    DecodedImage out;
    out.width        = image.width;
    out.height       = image.height;
    out.has_alpha    = image.has_alpha;
    out.pil_exact    = image.pil_exact;
    out.pixel_format = image.pixel_format;
    const std::size_t pixels = static_cast<std::size_t>(image.width) * image.height;
    if (image.rgba.size() != pixels * 4) { throw std::logic_error("decoded RGBA image has the wrong size"); }
    out.rgb.resize(pixels * 3);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * image.width + x;
            const std::uint8_t* s = image.rgba.data() + i * 4;
            std::uint8_t* d       = out.rgb.data() + i * 3;
            const unsigned a      = s[3];
            if (a == 255U) {  // (c*255 + bg*0 + 127) // 255 == c
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
                continue;
            }
            // _composite_on_checkerboard: background 204 where (x // 8 + y // 8) is even, 153 otherwise.
            const unsigned bg = ((x / 8 + y / 8) % 2 == 0) ? 204U : 153U;
            for (int c = 0; c < 3; ++c) { d[c] = static_cast<std::uint8_t>((s[c] * a + bg * (255U - a) + 127U) / 255U); }
        }
    }
    return out;
}

DecodedImage composite_white(const media::decode::PilImage& image) {
    DecodedImage out;
    out.width        = image.width;
    out.height       = image.height;
    out.has_alpha    = image.has_alpha;
    out.pil_exact    = image.pil_exact;
    out.pixel_format = image.pixel_format;
    const std::size_t pixels = static_cast<std::size_t>(image.width) * image.height;
    if (image.rgba.size() != pixels * 4) { throw std::logic_error("decoded RGBA image has the wrong size"); }
    out.rgb.resize(pixels * 3);
    for (std::size_t i = 0; i < pixels; ++i) {
        const std::uint8_t* s = image.rgba.data() + i * 4;
        std::uint8_t* d       = out.rgb.data() + i * 3;
        const unsigned a      = s[3];
        for (int c = 0; c < 3; ++c) {
            // Paste.c: BLEND(mask, in1, in2) = DIV255(in1 * (255 - mask) + in2 * mask), DIV255(v) = (t + (t >> 8)) >> 8
            // with t = v + 128; exact (== c) where a == 255.
            const unsigned t = 255U * (255U - a) + s[c] * a + 128U;
            d[c]             = static_cast<std::uint8_t>(((t >> 8U) + t) >> 8U);
        }
    }
    return out;
}

namespace {

media::decode::PilOptions pil_options(DecodePath path) {
    return path == DecodePath::Served ? media::decode::PilOptions{.append_jpeg_eoi = false, .exif_transpose = true}
                                      : media::decode::PilOptions{.append_jpeg_eoi = true, .exif_transpose = false};
}

// vLLM's server wraps Pillow's failures (OSError, UnidentifiedImageError, SyntaxError, RuntimeError,
// DecompressionBombError) as ValueError(f"Failed to load image: {e}").
[[noreturn]] void failed_to_load(DecodePath path, const std::exception& e) {
    if (path == DecodePath::Served) { throw std::invalid_argument(std::string("Failed to load image: ") + e.what()); }
    throw std::invalid_argument(e.what());
}

} // namespace

ImageProbe probe_image(std::span<const std::uint8_t> bytes, DecodePath path) {
    media::decode::PilHeader header;
    try {
        header = media::decode::probe_image_pil(bytes, pil_options(path));
    } catch (const media::decode::Error&) {
        throw;
    } catch (const std::invalid_argument& e) { failed_to_load(path, e); } catch (const std::runtime_error& e) {
        failed_to_load(path, e);
    }
    const std::uint64_t pixels = static_cast<std::uint64_t>(header.width) * static_cast<std::uint64_t>(header.height);
    if (pixels > kMaxImagePixels) {
        if (path == DecodePath::Served) {
            // ImageMediaIO.load_bytes, before any decode (a ValueError of its own: not wrapped).
            throw std::invalid_argument("Image dimensions " + std::to_string(header.width) + "x" +
                                        std::to_string(header.height) + " (" + std::to_string(pixels) +
                                        " pixels) exceed the maximum of " + std::to_string(kMaxImagePixels) +
                                        " pixels. Set VLLM_MAX_IMAGE_PIXELS to increase this limit.");
        }
        throw std::invalid_argument("Image size (" + std::to_string(pixels) + " pixels) exceeds limit of " +
                                    std::to_string(kMaxImagePixels) +
                                    " pixels, could be decompression bomb DOS attack.");
    }
    const bool swap = header.orientation >= 5;
    return {swap ? header.height : header.width, swap ? header.width : header.height};
}

DecodedImage decode_image(std::span<const std::uint8_t> bytes, const media::decode::Policy& policy, DecodePath path) {
    media::decode::PilImage image;
    try {
        image = media::decode::decode_image_pil(bytes, policy, pil_options(path));
    } catch (const media::decode::Error&) {
        throw;
    } catch (const std::invalid_argument& e) { failed_to_load(path, e); } catch (const std::runtime_error& e) {
        failed_to_load(path, e);
    }
    return path == DecodePath::Served ? composite_white(image) : composite(image);
}

ImageGrid image_grid(int height, int width) {
    if (height <= 0 || width <= 0) { throw std::invalid_argument("image dimensions must be positive"); }
    return {(height + kImageCell - 1) / kImageCell, (width + kImageCell - 1) / kImageCell};
}

std::uint16_t normalized_bf16(int value, int channel) noexcept {
    return normalization_table()[static_cast<std::size_t>(channel)][static_cast<std::size_t>(value + 1)];
}

void write_patches_merge(const DecodedImage& image, std::span<std::uint16_t> out) {
    const ImageGrid grid = image_grid(image.height, image.width);
    const std::size_t rows = static_cast<std::size_t>(grid.tokens()) * kPatchesPerToken;
    if (out.size() != rows * kPatchFeatures) { throw std::logic_error("Inkling patch payload has the wrong size"); }
    const auto& table = normalization_table();
    std::uint16_t* d  = out.data();
    for (int r = 0; r < grid.rows; ++r) {
        for (int c = 0; c < grid.cols; ++c) {
            for (int dy = 0; dy < kMergeSize; ++dy) {
                for (int dx = 0; dx < kMergeSize; ++dx) {
                    const int y0 = r * kImageCell + dy * kNativePatch;
                    const int x0 = c * kImageCell + dx * kNativePatch;
                    for (int py = 0; py < kNativePatch; ++py) {
                        for (int px = 0; px < kNativePatch; ++px) {
                            for (int ch = 0; ch < 3; ++ch) {
                                *d++ = table[static_cast<std::size_t>(ch)]
                                            [static_cast<std::size_t>(raw_at(image, y0 + py, x0 + px, ch) + 1)];
                            }
                        }
                    }
                }
            }
        }
    }
}

std::vector<std::uint16_t> raw_pixels(const DecodedImage& image) {
    const ImageGrid grid = image_grid(image.height, image.width);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(grid.tokens()) * kImageCell * kImageCell * 3);
    std::size_t i = 0;
    for (int r = 0; r < grid.rows; ++r) {
        for (int c = 0; c < grid.cols; ++c) {
            for (int y = 0; y < kImageCell; ++y) {
                for (int x = 0; x < kImageCell; ++x) {
                    for (int ch = 0; ch < 3; ++ch) {
                        out[i++] = bf16_bits(static_cast<float>(raw_at(image, r * kImageCell + y, c * kImageCell + x, ch)));
                    }
                }
            }
        }
    }
    return out;
}

std::string image_too_large_message(std::uint64_t tokens, std::uint64_t cap) {
    return "The decoder prompt contains a(n) image item with " + std::to_string(tokens) +
           " embedding tokens, which exceeds the pre-allocated encoder cache size " + std::to_string(cap) +
           ". Please reduce the input size or increase the encoder cache size by setting --limit-mm-per-prompt at "
           "startup.";
}

} // namespace sinfer::family::frontend_internal::inkling
