// ENGINE-INKLING lane V-b: a WebP decoded the way Pillow 13 decodes it (the reference for Inkling-2's vision processor),
// with the same library: libwebp 1.6.0 (csrc/cmake/serve_pil_decoders.cmake). Pillow (WebPImagePlugin.py + _webp.c)
// opens every WebP through WebPAnimDecoder in colour mode RGBA and loads frame 0 (canvas = the first frame drawn on a
// transparent canvas); its mode is "RGBA" when the bitstream has alpha, else "RGB" (rawmode "RGBX": alpha dropped).

#include "media/decode/decode.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

#include <webp/decode.h>
#include <webp/demux.h>

namespace sinfer::media::decode::detail {

std::pair<int, int> probe_webp_pil(std::span<const std::uint8_t> bytes) {
    // WebPAnimDecoder's canvas: the VP8X canvas size, or the single frame's size.
    WebPData data{bytes.data(), bytes.size()};
    const std::unique_ptr<WebPDemuxer, void (*)(WebPDemuxer*)> demux(WebPDemux(&data), WebPDemuxDelete);
    if (!demux) { throw std::invalid_argument("failed to decode WebP: not a WebP bitstream"); }
    return {static_cast<int>(WebPDemuxGetI(demux.get(), WEBP_FF_CANVAS_WIDTH)),
            static_cast<int>(WebPDemuxGetI(demux.get(), WEBP_FF_CANVAS_HEIGHT))};
}

PilImage decode_webp_pil(std::span<const std::uint8_t> bytes, std::uint64_t max_pixels) {
    WebPBitstreamFeatures features{};
    if (WebPGetFeatures(bytes.data(), bytes.size(), &features) != VP8_STATUS_OK) {
        throw std::invalid_argument("failed to decode WebP: not a WebP bitstream");
    }
    WebPData data{bytes.data(), bytes.size()};
    WebPAnimDecoderOptions options;
    if (WebPAnimDecoderOptionsInit(&options) == 0) { throw std::runtime_error("libwebp version mismatch"); }
    options.color_mode  = MODE_RGBA;
    options.use_threads = 0;
    const std::unique_ptr<WebPAnimDecoder, void (*)(WebPAnimDecoder*)> decoder(WebPAnimDecoderNew(&data, &options),
                                                                            WebPAnimDecoderDelete);
    if (!decoder) { throw std::invalid_argument("failed to decode WebP: could not create the decoder"); }
    WebPAnimInfo info{};
    if (WebPAnimDecoderGetInfo(decoder.get(), &info) == 0) { throw std::invalid_argument("failed to decode WebP"); }
    if (info.canvas_width == 0 || info.canvas_height == 0) { throw std::invalid_argument("decoded media dimensions are invalid"); }
    if (static_cast<std::uint64_t>(info.canvas_width) * info.canvas_height > max_pixels) {
        throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
    }
    std::uint8_t* canvas = nullptr;
    int timestamp        = 0;
    if (WebPAnimDecoderGetNext(decoder.get(), &canvas, &timestamp) == 0 || canvas == nullptr) {
        throw std::invalid_argument("failed to decode WebP: no frame");
    }
    PilImage out;
    out.width        = static_cast<int>(info.canvas_width);
    out.height       = static_cast<int>(info.canvas_height);
    out.has_alpha    = features.has_alpha != 0;
    out.pil_exact    = true;
    out.pixel_format = features.format == 2 ? "webp-lossless" : "webp-lossy";
    out.rgba.resize(static_cast<std::size_t>(out.width) * static_cast<std::size_t>(out.height) * 4);
    std::memcpy(out.rgba.data(), canvas, out.rgba.size());
    if (!out.has_alpha) {
        for (std::size_t i = 3; i < out.rgba.size(); i += 4) { out.rgba[i] = 255; }
    }
    return out;
}

} // namespace sinfer::media::decode::detail
