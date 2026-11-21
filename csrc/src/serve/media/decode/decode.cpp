#include "media/decode/decode.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sinfer::media::decode {
namespace {

std::string av_error(int code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
    av_strerror(code, text.data(), text.size());
    return text.data();
}

void validate_input(std::span<const std::uint8_t> bytes, const Policy& policy) {
    if (policy.max_bytes == 0) { throw std::invalid_argument("media byte limit must be positive"); }
    if (bytes.empty()) { throw std::invalid_argument("media bytes are empty"); }
    if (bytes.size() > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media bytes exceed byte limit");
    }
}

int exif_orientation(std::span<const std::uint8_t> bytes) {
    constexpr std::size_t limit = 1ULL << 20;
    bytes                       = bytes.first(std::min(limit, bytes.size()));
    if (bytes.size() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8) { return 1; }
    auto be16 = [&](std::size_t offset) {
        return static_cast<std::uint16_t>((bytes[offset] << 8U) | bytes[offset + 1]);
    };
    std::size_t marker = 2;
    while (marker + 4 <= bytes.size()) {
        if (bytes[marker] != 0xff) { break; }
        const std::uint8_t kind = bytes[marker + 1];
        if (kind == 0xda || kind == 0xd9) { break; }
        const std::uint16_t length = be16(marker + 2);
        if (length < 2 || marker + 2 + length > bytes.size()) { break; }
        const std::size_t payload      = marker + 4;
        const std::size_t payload_size = length - 2;
        if (kind == 0xe1 && payload_size >= 14 &&
            std::memcmp(bytes.data() + payload, "Exif\0\0", 6) == 0) {
            const std::size_t tiff = payload + 6;
            const bool little      = bytes[tiff] == 'I' && bytes[tiff + 1] == 'I';
            const bool big         = bytes[tiff] == 'M' && bytes[tiff + 1] == 'M';
            if (!little && !big) { return 1; }
            auto u16 = [&](std::size_t offset) -> std::uint16_t {
                if (offset + 2 > bytes.size()) { return 0; }
                return little ? static_cast<std::uint16_t>(bytes[offset] | bytes[offset + 1] << 8U)
                              : static_cast<std::uint16_t>(bytes[offset] << 8U | bytes[offset + 1]);
            };
            auto u32 = [&](std::size_t offset) -> std::uint32_t {
                if (offset + 4 > bytes.size()) { return 0; }
                if (little) {
                    return static_cast<std::uint32_t>(bytes[offset]) |
                           static_cast<std::uint32_t>(bytes[offset + 1]) << 8U |
                           static_cast<std::uint32_t>(bytes[offset + 2]) << 16U |
                           static_cast<std::uint32_t>(bytes[offset + 3]) << 24U;
                }
                return static_cast<std::uint32_t>(bytes[offset]) << 24U |
                       static_cast<std::uint32_t>(bytes[offset + 1]) << 16U |
                       static_cast<std::uint32_t>(bytes[offset + 2]) << 8U |
                       static_cast<std::uint32_t>(bytes[offset + 3]);
            };
            if (u16(tiff + 2) != 42) { return 1; }
            const std::size_t ifd = tiff + u32(tiff + 4);
            if (ifd + 2 > bytes.size()) { return 1; }
            const std::uint16_t count = u16(ifd);
            for (std::uint16_t i = 0; i < count; ++i) {
                const std::size_t entry = ifd + 2 + static_cast<std::size_t>(i) * 12;
                if (entry + 12 > bytes.size()) { return 1; }
                if (u16(entry) == 0x0112 && u16(entry + 2) == 3 && u32(entry + 4) == 1) {
                    const int orientation = u16(entry + 8);
                    return orientation >= 1 && orientation <= 8 ? orientation : 1;
                }
            }
            return 1;
        }
        marker += 2 + length;
    }
    return 1;
}

struct BufferCursor {
    const std::uint8_t* data = nullptr;
    std::size_t size         = 0;
    std::size_t offset       = 0;
};

int read_packet(void* opaque, std::uint8_t* output, int size) {
    auto& cursor = *static_cast<BufferCursor*>(opaque);
    if (cursor.offset == cursor.size) { return AVERROR_EOF; }
    const std::size_t amount =
        std::min<std::size_t>(static_cast<std::size_t>(size), cursor.size - cursor.offset);
    std::memcpy(output, cursor.data + cursor.offset, amount);
    cursor.offset += amount;
    return static_cast<int>(amount);
}

class AvImageBuffer {
public:
    AvImageBuffer(int width, int height, AVPixelFormat format) {
        constexpr int alignment = 64;
        const int rc =
            av_image_alloc(data_.data(), linesize_.data(), width, height, format, alignment);
        if (rc == AVERROR(ENOMEM)) { throw std::bad_alloc(); }
        if (rc < 0) {
            throw std::runtime_error("failed to allocate media conversion buffer: " + av_error(rc));
        }
    }

    ~AvImageBuffer() { av_freep(data_.data()); }

    AvImageBuffer(const AvImageBuffer&)            = delete;
    AvImageBuffer& operator=(const AvImageBuffer&) = delete;

    [[nodiscard]] std::uint8_t* const* data() const noexcept { return data_.data(); }

    [[nodiscard]] const int* linesize() const noexcept { return linesize_.data(); }

private:
    std::array<std::uint8_t*, 4> data_{};
    std::array<int, 4> linesize_{};
};

std::int64_t seek_packet(void* opaque, std::int64_t offset, int whence) {
    auto& cursor = *static_cast<BufferCursor*>(opaque);
    if (whence == AVSEEK_SIZE) { return static_cast<std::int64_t>(cursor.size); }
    const int base_kind = whence & ~AVSEEK_FORCE;
    std::int64_t base   = 0;
    if (base_kind == SEEK_CUR) {
        base = static_cast<std::int64_t>(cursor.offset);
    } else if (base_kind == SEEK_END) {
        base = static_cast<std::int64_t>(cursor.size);
    } else if (base_kind != SEEK_SET) {
        return AVERROR(EINVAL);
    }
    if (offset < -base || base + offset < 0 ||
        static_cast<std::uint64_t>(base + offset) > cursor.size) {
        return AVERROR(EINVAL);
    }
    cursor.offset = static_cast<std::size_t>(base + offset);
    return static_cast<std::int64_t>(cursor.offset);
}

class Decoder {
public:
    Decoder(std::span<const std::uint8_t> input, std::uint64_t max_decoded_pixels)
        : max_decoded_pixels_(max_decoded_pixels) {
        try {
            format_ = avformat_alloc_context();
            if (format_ == nullptr) { throw std::bad_alloc(); }
            cursor_              = BufferCursor{input.data(), input.size(), 0};
            std::uint8_t* buffer = static_cast<std::uint8_t*>(av_malloc(32'768));
            if (buffer == nullptr) { throw std::bad_alloc(); }
            io_ =
                avio_alloc_context(buffer, 32'768, 0, &cursor_, read_packet, nullptr, seek_packet);
            if (io_ == nullptr) {
                av_free(buffer);
                throw std::bad_alloc();
            }
            format_->pb = io_;
            format_->flags |= AVFMT_FLAG_CUSTOM_IO;
            AVFormatContext* raw = format_;
            int rc               = avformat_open_input(&raw, nullptr, nullptr, nullptr);
            format_              = raw;
            if (rc < 0) { throw std::invalid_argument("failed to open media: " + av_error(rc)); }
            if ((rc = avformat_find_stream_info(format_, nullptr)) < 0) {
                throw std::invalid_argument("failed to inspect media: " + av_error(rc));
            }
            stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if (stream_index_ < 0) {
                throw std::invalid_argument("media has no decodable video stream");
            }
            stream_              = format_->streams[stream_index_];
            const AVCodec* codec = avcodec_find_decoder(stream_->codecpar->codec_id);
            if (codec == nullptr) { throw std::invalid_argument("media codec is not supported"); }
            codec_ = avcodec_alloc_context3(codec);
            if (codec_ == nullptr) { throw std::bad_alloc(); }
            if ((rc = avcodec_parameters_to_context(codec_, stream_->codecpar)) < 0 ||
                (rc = avcodec_open2(codec_, codec, nullptr)) < 0) {
                throw std::invalid_argument("failed to open media codec: " + av_error(rc));
            }
            packet_ = av_packet_alloc();
            frame_  = av_frame_alloc();
            if (packet_ == nullptr || frame_ == nullptr) { throw std::bad_alloc(); }
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Decoder() { cleanup(); }

    Decoder(const Decoder&)            = delete;
    Decoder& operator=(const Decoder&) = delete;

    [[nodiscard]] AVStream* stream() const noexcept { return stream_; }

    [[nodiscard]] double duration_seconds() const noexcept {
        if (stream_->duration > 0) {
            return static_cast<double>(stream_->duration) * av_q2d(stream_->time_base);
        }
        if (format_->duration > 0) { return static_cast<double>(format_->duration) / AV_TIME_BASE; }
        return 0.0;
    }

    template <typename Callback>
    void frames(Callback&& callback) {
        int index    = 0;
        auto receive = [&] {
            while (true) {
                const int rc = avcodec_receive_frame(codec_, frame_);
                if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { return true; }
                if (rc < 0) {
                    throw std::runtime_error("failed to decode media frame: " + av_error(rc));
                }
                if (frame_->crop_top != 0 || frame_->crop_bottom != 0 || frame_->crop_left != 0 ||
                    frame_->crop_right != 0) {
                    const int crop = av_frame_apply_cropping(frame_, 0);
                    if (crop < 0) {
                        throw std::runtime_error("failed to apply media display crop: " +
                                                 av_error(crop));
                    }
                }
                const bool keep_decoding = callback(index++, frame_);
                av_frame_unref(frame_);
                if (!keep_decoding) { return false; }
            }
        };
        while (true) {
            const int rc = av_read_frame(format_, packet_);
            if (rc == AVERROR_EOF) { break; }
            if (rc < 0) { throw std::runtime_error("failed while reading media: " + av_error(rc)); }
            if (packet_->stream_index == stream_index_) {
                const int send = avcodec_send_packet(codec_, packet_);
                if (send < 0 && send != AVERROR(EAGAIN)) {
                    av_packet_unref(packet_);
                    throw std::runtime_error("failed to submit media packet: " + av_error(send));
                }
                if (!receive()) {
                    av_packet_unref(packet_);
                    return;
                }
            }
            av_packet_unref(packet_);
        }
        const int flush = avcodec_send_packet(codec_, nullptr);
        if (flush < 0 && flush != AVERROR_EOF) {
            throw std::runtime_error("failed to flush media decoder: " + av_error(flush));
        }
        (void)receive();
    }

    Image rgb(const AVFrame* frame, int orientation, bool composite_alpha = false) {
        const int width  = frame->width;
        const int height = frame->height;
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("decoded media dimensions are invalid");
        }
        if (static_cast<std::uint64_t>(width) * height > max_decoded_pixels_) {
            throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
        }
        Image out;
        out.width  = width;
        out.height = height;
        const AVPixFmtDescriptor* descriptor =
            av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame->format));
        const bool alpha = composite_alpha && descriptor != nullptr &&
                           (descriptor->flags & AV_PIX_FMT_FLAG_ALPHA) != 0;
        const AVPixelFormat destination_format = alpha ? AV_PIX_FMT_RGBA : AV_PIX_FMT_RGB24;
        AvImageBuffer converted(width, height, destination_format);
        sws_ = sws_getCachedContext(sws_, width, height, static_cast<AVPixelFormat>(frame->format),
                                    width, height, destination_format, SWS_POINT, nullptr, nullptr,
                                    nullptr);
        if (sws_ == nullptr) { throw std::runtime_error("failed to create media color converter"); }
        const int rows = sws_scale(sws_, frame->data, frame->linesize, 0, height, converted.data(),
                                   converted.linesize());
        if (rows != height) { throw std::runtime_error("failed to convert media frame to RGB"); }
        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        out.rgb.resize(pixels * 3);
        if (!alpha) {
            const std::size_t row_bytes = static_cast<std::size_t>(width) * 3;
            for (int y = 0; y < height; ++y) {
                std::memcpy(out.rgb.data() + static_cast<std::size_t>(y) * row_bytes,
                            converted.data()[0] +
                                static_cast<std::size_t>(y) * converted.linesize()[0],
                            row_bytes);
            }
        } else {
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* source =
                    converted.data()[0] + static_cast<std::size_t>(y) * converted.linesize()[0];
                std::uint8_t* destination =
                    out.rgb.data() + static_cast<std::size_t>(y) * width * 3;
                for (int x = 0; x < width; ++x) {
                    const int a = source[4 * x + 3];
                    for (int c = 0; c < 3; ++c) {
                        destination[3 * x + c] = static_cast<std::uint8_t>(
                            ((255 - a) * 255 + a * source[4 * x + c] + 127) / 255);
                    }
                }
            }
        }
        if (orientation == 1) { return out; }
        Image rotated;
        const bool swap = orientation >= 5;
        rotated.width   = swap ? height : width;
        rotated.height  = swap ? width : height;
        rotated.rgb.resize(out.rgb.size());
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int rx = 0;
                int ry = 0;
                if (orientation == 2) {
                    rx = width - 1 - x;
                    ry = y;
                } else if (orientation == 3) {
                    rx = width - 1 - x;
                    ry = height - 1 - y;
                } else if (orientation == 4) {
                    rx = x;
                    ry = height - 1 - y;
                } else if (orientation == 5) {
                    rx = y;
                    ry = x;
                } else if (orientation == 6) {
                    rx = height - 1 - y;
                    ry = x;
                } else if (orientation == 7) {
                    rx = height - 1 - y;
                    ry = width - 1 - x;
                } else { // orientation 8
                    rx = y;
                    ry = width - 1 - x;
                }
                const std::size_t src = (static_cast<std::size_t>(y) * width + x) * 3;
                const std::size_t dst_index =
                    (static_cast<std::size_t>(ry) * rotated.width + rx) * 3;
                std::copy_n(out.rgb.data() + src, 3, rotated.rgb.data() + dst_index);
            }
        }
        return rotated;
    }

    // Pillow's 8-bit RGBA view of a decoded frame (decode_image_pil). Formats whose samples FFmpeg hands over
    // unchanged are unpacked here with Pillow's rules; the rest go through swscale (nearest) and are marked inexact.
    PilImage pil(const AVFrame* frame, bool source_has_no_alpha, bool lossless_codec) {
        const int width  = frame->width;
        const int height = frame->height;
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("decoded media dimensions are invalid");
        }
        if (static_cast<std::uint64_t>(width) * height > max_decoded_pixels_) {
            throw Error(ErrorKind::BudgetExceeded, "decoded media pixels exceed processor limit");
        }
        const auto format = static_cast<AVPixelFormat>(frame->format);
        const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(format);
        PilImage out;
        out.width        = width;
        out.height       = height;
        out.pixel_format = descriptor != nullptr ? descriptor->name : "unknown";
        out.rgba.assign(static_cast<std::size_t>(width) * height * 4, 255);
        bool native = true;
        bool alpha  = false;
        const auto row = [&](int plane, int y) {
            return frame->data[plane] + static_cast<std::ptrdiff_t>(y) * frame->linesize[plane];
        };
        const auto put = [&](int x, int y, int r, int g, int b, int a) {
            std::uint8_t* p = out.rgba.data() + (static_cast<std::size_t>(y) * width + x) * 4;
            p[0] = static_cast<std::uint8_t>(r);
            p[1] = static_cast<std::uint8_t>(g);
            p[2] = static_cast<std::uint8_t>(b);
            p[3] = static_cast<std::uint8_t>(a);
        };
        // 16-bit samples: the high byte (Pillow's "RGB;16B" / "RGBA;16B" / "LA;16B" rawmodes).
        const auto hi16 = [](const std::uint8_t* s, bool big) { return big ? s[0] : s[1]; };
        const auto u16  = [](const std::uint8_t* s, bool big) {
            return big ? (s[0] << 8) | s[1] : (s[1] << 8) | s[0];
        };
        switch (format) {
        case AV_PIX_FMT_RGB24:
        case AV_PIX_FMT_BGR24:
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* s = row(0, y);
                for (int x = 0; x < width; ++x, s += 3) {
                    if (format == AV_PIX_FMT_RGB24) { put(x, y, s[0], s[1], s[2], 255); }
                    else { put(x, y, s[2], s[1], s[0], 255); }
                }
            }
            break;
        case AV_PIX_FMT_RGBA:
        case AV_PIX_FMT_BGRA:
        case AV_PIX_FMT_ARGB:
        case AV_PIX_FMT_ABGR:
        case AV_PIX_FMT_RGB0:
        case AV_PIX_FMT_BGR0:
        case AV_PIX_FMT_0RGB:
        case AV_PIX_FMT_0BGR: {
            // Byte positions of R, G, B, A (A < 0: padding, no alpha).
            int ri = 0, gi = 1, bi = 2, ai = 3;
            if (format == AV_PIX_FMT_BGRA || format == AV_PIX_FMT_BGR0) { ri = 2; bi = 0; }
            if (format == AV_PIX_FMT_ARGB || format == AV_PIX_FMT_0RGB) { ai = 0; ri = 1; gi = 2; bi = 3; }
            if (format == AV_PIX_FMT_ABGR || format == AV_PIX_FMT_0BGR) { ai = 0; bi = 1; gi = 2; ri = 3; }
            const bool has_a = format == AV_PIX_FMT_RGBA || format == AV_PIX_FMT_BGRA ||
                               format == AV_PIX_FMT_ARGB || format == AV_PIX_FMT_ABGR;
            alpha = has_a;
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* s = row(0, y);
                for (int x = 0; x < width; ++x, s += 4) { put(x, y, s[ri], s[gi], s[bi], has_a ? s[ai] : 255); }
            }
            break;
        }
        case AV_PIX_FMT_GRAY8:
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* s = row(0, y);
                for (int x = 0; x < width; ++x) { put(x, y, s[x], s[x], s[x], 255); }
            }
            break;
        case AV_PIX_FMT_YA8:
            alpha = true;
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* s = row(0, y);
                for (int x = 0; x < width; ++x, s += 2) { put(x, y, s[0], s[0], s[0], s[1]); }
            }
            break;
        case AV_PIX_FMT_MONOBLACK:
        case AV_PIX_FMT_MONOWHITE:
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* s = row(0, y);
                for (int x = 0; x < width; ++x) {
                    const bool bit = ((s[x >> 3] >> (7 - (x & 7))) & 1) != 0;
                    const int v    = (bit == (format == AV_PIX_FMT_MONOBLACK)) ? 255 : 0;
                    put(x, y, v, v, v, 255);
                }
            }
            break;
        case AV_PIX_FMT_PAL8: {
            // Palette: 256 x uint32 in native endianness, 0xAARRGGBB; a PNG tRNS / GIF transparent index is its alpha.
            const auto* palette = reinterpret_cast<const std::uint32_t*>(frame->data[1]);
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* s = row(0, y);
                for (int x = 0; x < width; ++x) {
                    const std::uint32_t c = palette[s[x]];
                    const int a           = static_cast<int>(c >> 24);
                    alpha                 = alpha || a != 255;
                    put(x, y, (c >> 16) & 0xff, (c >> 8) & 0xff, c & 0xff, a);
                }
            }
            break;
        }
        case AV_PIX_FMT_RGB48BE:
        case AV_PIX_FMT_RGB48LE:
        case AV_PIX_FMT_RGBA64BE:
        case AV_PIX_FMT_RGBA64LE: {
            const bool big   = format == AV_PIX_FMT_RGB48BE || format == AV_PIX_FMT_RGBA64BE;
            const bool has_a = format == AV_PIX_FMT_RGBA64BE || format == AV_PIX_FMT_RGBA64LE;
            const int stride = has_a ? 8 : 6;
            alpha            = has_a;
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* s = row(0, y);
                for (int x = 0; x < width; ++x, s += stride) {
                    put(x, y, hi16(s, big), hi16(s + 2, big), hi16(s + 4, big), has_a ? hi16(s + 6, big) : 255);
                }
            }
            break;
        }
        case AV_PIX_FMT_GRAY16BE:
        case AV_PIX_FMT_GRAY16LE: {
            // Pillow mode "I;16"; convert("RGB"/"RGBA") clips: values above 255 become 255.
            const bool big = format == AV_PIX_FMT_GRAY16BE;
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* s = row(0, y);
                for (int x = 0; x < width; ++x, s += 2) {
                    const int v = std::min(u16(s, big), 255);
                    put(x, y, v, v, v, 255);
                }
            }
            break;
        }
        case AV_PIX_FMT_YA16BE:
        case AV_PIX_FMT_YA16LE: {
            const bool big = format == AV_PIX_FMT_YA16BE;
            alpha          = true;
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* s = row(0, y);
                for (int x = 0; x < width; ++x, s += 4) {
                    const int v = hi16(s, big);
                    put(x, y, v, v, v, hi16(s + 2, big));
                }
            }
            break;
        }
        default: {
            native = false;
            const bool has_a = descriptor != nullptr && (descriptor->flags & AV_PIX_FMT_FLAG_ALPHA) != 0;
            alpha            = has_a;
            AvImageBuffer converted(width, height, AV_PIX_FMT_RGBA);
            sws_ = sws_getCachedContext(sws_, width, height, format, width, height, AV_PIX_FMT_RGBA, SWS_POINT,
                                        nullptr, nullptr, nullptr);
            if (sws_ == nullptr) { throw std::runtime_error("failed to create media color converter"); }
            const int rows = sws_scale(sws_, frame->data, frame->linesize, 0, height, converted.data(),
                                       converted.linesize());
            if (rows != height) { throw std::runtime_error("failed to convert media frame to RGBA"); }
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* s = converted.data()[0] + static_cast<std::size_t>(y) * converted.linesize()[0];
                for (int x = 0; x < width; ++x, s += 4) { put(x, y, s[0], s[1], s[2], has_a ? s[3] : 255); }
            }
            break;
        }
        }
        if (source_has_no_alpha && alpha) {
            // Pillow's JPEG modes (L, RGB, CMYK) carry no alpha: an alpha plane FFmpeg reports there is not one.
            for (std::size_t i = 3; i < out.rgba.size(); i += 4) { out.rgba[i] = 255; }
            alpha = false;
        }
        out.has_alpha = alpha;
        out.pil_exact = native && lossless_codec;
        return out;
    }

private:
    void cleanup() noexcept {
        sws_freeContext(sws_);
        sws_ = nullptr;
        av_frame_free(&frame_);
        av_packet_free(&packet_);
        avcodec_free_context(&codec_);
        if (format_ != nullptr) { avformat_close_input(&format_); }
        if (io_ != nullptr) {
            // Custom IO owns its buffer separately. Probing may replace the
            // original allocation, so free the buffer currently in the context.
            av_freep(&io_->buffer);
            avio_context_free(&io_);
        }
    }

    AVFormatContext* format_ = nullptr;
    AVIOContext* io_         = nullptr;
    AVCodecContext* codec_   = nullptr;
    AVPacket* packet_        = nullptr;
    AVFrame* frame_          = nullptr;
    AVStream* stream_        = nullptr;
    SwsContext* sws_         = nullptr;
    BufferCursor cursor_;
    int stream_index_                 = -1;
    std::uint64_t max_decoded_pixels_ = 0;
};

int rotation_of(const AVStream* stream) {
    int rotation                 = 0;
    const AVDictionaryEntry* tag = av_dict_get(stream->metadata, "rotate", nullptr, 0);
    if (tag != nullptr) {
        rotation = static_cast<int>(std::nearbyint(std::strtod(tag->value, nullptr)));
    }
    const AVPacketSideData* side =
        av_packet_side_data_get(stream->codecpar->coded_side_data,
                                stream->codecpar->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX);
    if (side != nullptr && side->size >= 9 * sizeof(std::int32_t)) {
        rotation = static_cast<int>(std::nearbyint(
            -av_display_rotation_get(reinterpret_cast<const std::int32_t*>(side->data))));
    }
    rotation %= 360;
    if (rotation < 0) { rotation += 360; }
    if (rotation == 89 || rotation == 91) { rotation = 90; }
    if (rotation == 179 || rotation == 181) { rotation = 180; }
    if (rotation == 269 || rotation == 271) { rotation = 270; }
    return rotation == 90 || rotation == 180 || rotation == 270 ? rotation : 0;
}

int orientation_from_rotation(int rotation) {
    if (rotation == 90) { return 6; }
    if (rotation == 180) { return 3; }
    if (rotation == 270) { return 8; }
    return 1;
}

double fps_of(const AVStream* stream) {
    double fps = av_q2d(stream->avg_frame_rate);
    if (!(fps > 0.0) || !std::isfinite(fps)) { fps = av_q2d(stream->r_frame_rate); }
    return fps > 0.0 && std::isfinite(fps) ? fps : 24.0;
}

int count_frames(std::span<const std::uint8_t> input, const Policy& policy) {
    Decoder decoder(input, policy.max_decoded_pixels);
    int count = 0;
    decoder.frames([&](int index, const AVFrame*) {
        if (policy.checkpoint) { policy.checkpoint(); }
        count = index + 1;
        if (count > policy.max_video_source_frames) {
            throw Error(ErrorKind::BudgetExceeded,
                        "video source frame count exceeds processor limit");
        }
        return true;
    });
    return count;
}

std::vector<int> sample_indices(int total, double source_fps, double target_fps, int min_frames,
                                int max_frames, bool temporal_pairs) {
    if (total <= 0 || source_fps <= 0.0 || target_fps <= 0.0 || min_frames <= 0 ||
        max_frames < min_frames) {
        throw std::invalid_argument("invalid video sampling configuration");
    }
    int count = static_cast<int>(static_cast<double>(total) / source_fps * target_fps);
    count     = std::min({std::max(count, min_frames), max_frames, total});
    if (temporal_pairs) { count = std::min(total, std::max(2, (count / 2) * 2)); }
    std::vector<int> indices(static_cast<std::size_t>(count));
    if (count == 1) {
        indices[0] = 0;
        return indices;
    }
    for (int i = 0; i < count; ++i) {
        const double value                   = static_cast<double>(i) * (total - 1) / (count - 1);
        indices[static_cast<std::size_t>(i)] = static_cast<int>(temporal_pairs ? std::floor(value) : std::nearbyint(value));
    }
    return indices;
}

} // namespace

#ifdef SINFER_PIL_JPEG
namespace detail {
// pil_jpeg.cpp: libjpeg-turbo with Pillow's settings.
PilImage decode_jpeg_pil(std::span<const std::uint8_t> bytes, std::uint64_t max_pixels);
std::pair<int, int> probe_jpeg_pil(std::span<const std::uint8_t> bytes);
} // namespace detail
#endif
#ifdef SINFER_PIL_WEBP
namespace detail {
// pil_webp.cpp: libwebp's animation decoder, frame 0, as Pillow's WebP plugin.
PilImage decode_webp_pil(std::span<const std::uint8_t> bytes, std::uint64_t max_pixels);
std::pair<int, int> probe_webp_pil(std::span<const std::uint8_t> bytes);
} // namespace detail
#endif

bool available() noexcept { return true; }

Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    Decoder decoder(bytes, policy.max_decoded_pixels);
    int orientation = exif_orientation(bytes);
    if (orientation == 1) {
        orientation = orientation_from_rotation(rotation_of(decoder.stream()));
    }
    Image result;
    decoder.frames([&](int index, const AVFrame* frame) {
        if (policy.checkpoint) { policy.checkpoint(); }
        if (index == 0) {
            result = decoder.rgb(frame, orientation);
            return false;
        }
        return true;
    });
    if (result.rgb.empty()) { throw std::invalid_argument("image contains no decoded frame"); }
    return result;
}

namespace {

// ---- Pillow's view of orientation (ImageOps.exif_transpose -> Image.getexif) -------------------------------------

// Orientation (tag 0x0112) in IFD0 of the TIFF structure at `tiff`; 0 when absent or unreadable.
int tiff_orientation(std::span<const std::uint8_t> bytes, std::size_t tiff) {
    if (tiff + 8 > bytes.size()) { return 0; }
    const bool little = bytes[tiff] == 'I' && bytes[tiff + 1] == 'I';
    const bool big    = bytes[tiff] == 'M' && bytes[tiff + 1] == 'M';
    if (!little && !big) { return 0; }
    const auto u16 = [&](std::size_t o) -> std::uint32_t {
        if (o + 2 > bytes.size()) { return 0; }
        return little ? (bytes[o] | bytes[o + 1] << 8U) : (bytes[o] << 8U | bytes[o + 1]);
    };
    const auto u32 = [&](std::size_t o) -> std::uint32_t {
        if (o + 4 > bytes.size()) { return 0; }
        return little ? (static_cast<std::uint32_t>(bytes[o]) | bytes[o + 1] << 8U | bytes[o + 2] << 16U |
                         static_cast<std::uint32_t>(bytes[o + 3]) << 24U)
                      : (static_cast<std::uint32_t>(bytes[o]) << 24U | bytes[o + 1] << 16U | bytes[o + 2] << 8U |
                         bytes[o + 3]);
    };
    if (u16(tiff + 2) != 42) { return 0; }
    const std::size_t ifd = tiff + u32(tiff + 4);
    if (ifd + 2 > bytes.size()) { return 0; }
    const std::uint32_t count = u16(ifd);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::size_t entry = ifd + 2 + static_cast<std::size_t>(i) * 12;
        if (entry + 12 > bytes.size()) { return 0; }
        if (u16(entry) != 0x0112 || u32(entry + 4) != 1) { continue; }
        const std::uint32_t type = u16(entry + 2);
        if (type == 3) { return static_cast<int>(u16(entry + 8)); }  // SHORT
        if (type == 4) { return static_cast<int>(u32(entry + 8)); }  // LONG
        return 0;
    }
    return 0;
}

// Pillow's XMP fallback (Image.getexif): re.search(r'tiff:Orientation(="|>)([0-9])', xmp).
int xmp_orientation(std::string_view xmp) {
    for (std::size_t at = xmp.find("tiff:Orientation"); at != std::string_view::npos;
         at = xmp.find("tiff:Orientation", at + 1)) {
        const std::size_t v = at + 16;
        if (v + 1 < xmp.size() && xmp[v] == '=' && xmp[v + 1] == '"' && v + 2 < xmp.size() &&
            xmp[v + 2] >= '0' && xmp[v + 2] <= '9') {
            return xmp[v + 2] - '0';
        }
        if (v < xmp.size() && xmp[v] == '>' && v + 1 < xmp.size() && xmp[v + 1] >= '0' && xmp[v + 1] <= '9') {
            return xmp[v + 1] - '0';
        }
    }
    return 0;
}

std::string_view as_text(std::span<const std::uint8_t> bytes, std::size_t begin, std::size_t size) {
    return {reinterpret_cast<const char*>(bytes.data()) + begin, size};
}

// The orientation Pillow's exif_transpose applies: EXIF first, XMP when EXIF has none; 1 when neither has one or
// the value is outside 1..8 (exif_transpose maps only 2..8 to a transpose).
int pil_orientation(std::span<const std::uint8_t> bytes) {
    int exif = 0;
    int xmp  = 0;
    const auto be32 = [&](std::size_t o) {
        return static_cast<std::uint32_t>(bytes[o]) << 24U | bytes[o + 1] << 16U | bytes[o + 2] << 8U | bytes[o + 3];
    };
    if (bytes.size() >= 4 && bytes[0] == 0xff && bytes[1] == 0xd8) {
        // JPEG: APP1 "Exif\0\0" (the first one, as JpegImagePlugin keeps) and APP1 XMP.
        std::size_t marker = 2;
        bool have_exif     = false;
        while (marker + 4 <= bytes.size() && bytes[marker] == 0xff) {
            const std::uint8_t kind = bytes[marker + 1];
            if (kind == 0xda || kind == 0xd9) { break; }
            const std::size_t length = static_cast<std::size_t>(bytes[marker + 2]) << 8U | bytes[marker + 3];
            if (length < 2 || marker + 2 + length > bytes.size()) { break; }
            const std::size_t payload = marker + 4, size = length - 2;
            if (kind == 0xe1 && size >= 6 && std::memcmp(bytes.data() + payload, "Exif\0\0", 6) == 0 && !have_exif) {
                have_exif = true;
                exif      = tiff_orientation(bytes.first(payload + size), payload + 6);
            } else if (kind == 0xe1 && size >= 29 &&
                       std::memcmp(bytes.data() + payload, "http://ns.adobe.com/xap/1.0/\0", 29) == 0) {
                xmp = xmp_orientation(as_text(bytes, payload + 29, size - 29));
            }
            marker += 2 + length;
        }
    } else if (bytes.size() >= 8 && std::memcmp(bytes.data(), "\x89PNG\r\n\x1a\n", 8) == 0) {
        // PNG: eXIf (Pillow reads it before or after IDAT: exif_transpose loads the image first) and iTXt XMP.
        std::size_t pos = 8;
        while (pos + 8 <= bytes.size()) {
            const std::size_t length = be32(pos);
            if (pos + 12 + length > bytes.size()) { break; }
            const std::size_t data = pos + 8;
            if (std::memcmp(bytes.data() + pos + 4, "eXIf", 4) == 0) {
                exif = tiff_orientation(bytes.first(data + length), data);
            } else if (std::memcmp(bytes.data() + pos + 4, "iTXt", 4) == 0 && length > 18 &&
                       std::memcmp(bytes.data() + data, "XML:com.adobe.xmp\0", 18) == 0) {
                xmp = xmp_orientation(as_text(bytes, data + 18, length - 18));
            } else if (std::memcmp(bytes.data() + pos + 4, "IEND", 4) == 0) {
                break;
            }
            pos += 12 + length;
        }
    } else if (bytes.size() >= 12 && std::memcmp(bytes.data(), "RIFF", 4) == 0 &&
               std::memcmp(bytes.data() + 8, "WEBP", 4) == 0) {
        // WebP: EXIF and "XMP " chunks (RIFF, little-endian sizes, padded to even).
        std::size_t pos = 12;
        while (pos + 8 <= bytes.size()) {
            const std::size_t length = static_cast<std::size_t>(bytes[pos + 4]) | bytes[pos + 5] << 8U |
                                       bytes[pos + 6] << 16U | static_cast<std::size_t>(bytes[pos + 7]) << 24U;
            if (pos + 8 + length > bytes.size()) { break; }
            const std::size_t data = pos + 8;
            if (std::memcmp(bytes.data() + pos, "EXIF", 4) == 0) {
                const std::size_t skip = length >= 6 && std::memcmp(bytes.data() + data, "Exif\0\0", 6) == 0 ? 6 : 0;
                exif = tiff_orientation(bytes.first(data + length), data + skip);
            } else if (std::memcmp(bytes.data() + pos, "XMP ", 4) == 0) {
                xmp = xmp_orientation(as_text(bytes, data, length));
            }
            pos += 8 + length + (length & 1U);
        }
    } else if (bytes.size() >= 8 && (std::memcmp(bytes.data(), "II*\0", 4) == 0 || std::memcmp(bytes.data(), "MM\0*", 4) == 0)) {
        exif = tiff_orientation(bytes, 0);
    }
    const int orientation = exif != 0 ? exif : xmp;
    return orientation >= 2 && orientation <= 8 ? orientation : 1;
}

// Image.transpose for the orientation (2 FLIP_LEFT_RIGHT, 3 ROTATE_180, 4 FLIP_TOP_BOTTOM, 5 TRANSPOSE,
// 6 ROTATE_270, 7 TRANSVERSE, 8 ROTATE_90), on 4-byte pixels.
void transpose_rgba(PilImage& image, int orientation) {
    if (orientation == 1) { return; }
    const int width = image.width, height = image.height;
    const bool swap = orientation >= 5;
    std::vector<std::uint8_t> out(image.rgba.size());
    const int out_width = swap ? height : width;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int rx = x, ry = y;
            switch (orientation) {
            case 2: rx = width - 1 - x; break;
            case 3: rx = width - 1 - x; ry = height - 1 - y; break;
            case 4: ry = height - 1 - y; break;
            case 5: rx = y; ry = x; break;
            case 6: rx = height - 1 - y; ry = x; break;
            case 7: rx = height - 1 - y; ry = width - 1 - x; break;
            default: rx = y; ry = width - 1 - x; break;  // 8
            }
            std::memcpy(out.data() + (static_cast<std::size_t>(ry) * out_width + rx) * 4,
                        image.rgba.data() + (static_cast<std::size_t>(y) * width + x) * 4, 4);
        }
    }
    image.rgba   = std::move(out);
    image.width  = out_width;
    image.height = swap ? width : height;
}

bool is_jpeg(std::span<const std::uint8_t> bytes) {
    return bytes.size() >= 3 && bytes[0] == 0xff && bytes[1] == 0xd8 && bytes[2] == 0xff;
}

bool is_webp(std::span<const std::uint8_t> bytes) {
    return bytes.size() >= 12 && std::memcmp(bytes.data(), "RIFF", 4) == 0 && std::memcmp(bytes.data() + 8, "WEBP", 4) == 0;
}

} // namespace

PilImage decode_image_pil(std::span<const std::uint8_t> bytes, const Policy& policy, const PilOptions& options) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    const int orientation = options.exif_transpose ? pil_orientation(bytes) : 1;
    // vLLM's processor path (processing.py decode_image): a JPEG that does not end in its EOI marker gets one before
    // Pillow opens it.
    std::vector<std::uint8_t> fixed;
    if (options.append_jpeg_eoi && bytes.size() >= 2 && bytes[0] == 0xff && bytes[1] == 0xd8 &&
        !(bytes[bytes.size() - 2] == 0xff && bytes[bytes.size() - 1] == 0xd9)) {
        fixed.reserve(bytes.size() + 2);
        fixed.assign(bytes.begin(), bytes.end());
        fixed.push_back(0xff);
        fixed.push_back(0xd9);
        bytes = fixed;
    }
    PilImage result;
#ifdef SINFER_PIL_JPEG
    // Pillow's JPEG plugin accepts what starts with FF D8 FF and decodes it with libjpeg-turbo: the same library here.
    if (is_jpeg(bytes)) {
        result = detail::decode_jpeg_pil(bytes, policy.max_decoded_pixels);
        transpose_rgba(result, orientation);
        return result;
    }
#endif
#ifdef SINFER_PIL_WEBP
    // Pillow's WebP plugin: "RIFF" .... "WEBP".
    if (is_webp(bytes)) {
        result = detail::decode_webp_pil(bytes, policy.max_decoded_pixels);
        transpose_rgba(result, orientation);
        return result;
    }
#endif
    Decoder decoder(bytes, policy.max_decoded_pixels);
    const AVCodecParameters* codec = decoder.stream()->codecpar;
    const bool jpeg = codec->codec_id == AV_CODEC_ID_MJPEG || codec->codec_id == AV_CODEC_ID_JPEGLS;
    // Codecs whose decoded samples are the file's own (Pillow and FFmpeg agree on them by definition). A lossy WebP
    // decodes to YUV and lands in the swscale branch, which is inexact anyway.
    const bool lossless = codec->codec_id == AV_CODEC_ID_PNG || codec->codec_id == AV_CODEC_ID_APNG ||
                          codec->codec_id == AV_CODEC_ID_GIF || codec->codec_id == AV_CODEC_ID_BMP ||
                          codec->codec_id == AV_CODEC_ID_WEBP || codec->codec_id == AV_CODEC_ID_TIFF ||
                          codec->codec_id == AV_CODEC_ID_PPM || codec->codec_id == AV_CODEC_ID_PGM ||
                          codec->codec_id == AV_CODEC_ID_PAM || codec->codec_id == AV_CODEC_ID_TARGA ||
                          codec->codec_id == AV_CODEC_ID_QOI;
    bool decoded = false;
    decoder.frames([&](int index, const AVFrame* frame) {
        if (policy.checkpoint) { policy.checkpoint(); }
        if (index == 0) {
            result  = decoder.pil(frame, jpeg, lossless);
            decoded = true;
            return false;
        }
        return true;
    });
    if (!decoded) { throw std::invalid_argument("image contains no decoded frame"); }
    transpose_rgba(result, orientation);
    return result;
}

PilHeader probe_image_pil(std::span<const std::uint8_t> bytes, const PilOptions& options) {
    if (bytes.empty()) { throw std::invalid_argument("media bytes are empty"); }
    PilHeader header;
    header.orientation = options.exif_transpose ? pil_orientation(bytes) : 1;
#ifdef SINFER_PIL_JPEG
    if (is_jpeg(bytes)) {
        const auto [width, height] = detail::probe_jpeg_pil(bytes);
        header.width  = width;
        header.height = height;
        return header;
    }
#endif
#ifdef SINFER_PIL_WEBP
    if (is_webp(bytes)) {
        const auto [width, height] = detail::probe_webp_pil(bytes);
        header.width  = width;
        header.height = height;
        return header;
    }
#endif
    // The common lossless formats straight from their headers: opening them through FFmpeg would decode the first frame
    // (avformat_find_stream_info) before any size limit could refuse it.
    const auto le16 = [&](std::size_t o) { return static_cast<int>(bytes[o] | bytes[o + 1] << 8U); };
    const auto le32 = [&](std::size_t o) {
        return static_cast<std::int32_t>(static_cast<std::uint32_t>(bytes[o]) | bytes[o + 1] << 8U | bytes[o + 2] << 16U |
                                         static_cast<std::uint32_t>(bytes[o + 3]) << 24U);
    };
    const auto be32 = [&](std::size_t o) {
        return static_cast<std::uint32_t>(bytes[o]) << 24U | bytes[o + 1] << 16U | bytes[o + 2] << 8U | bytes[o + 3];
    };
    if (bytes.size() >= 24 && std::memcmp(bytes.data(), "\x89PNG\r\n\x1a\n", 8) == 0 &&
        std::memcmp(bytes.data() + 12, "IHDR", 4) == 0) {
        const std::uint32_t width = be32(16), height = be32(20);
        if (width == 0 || height == 0 || width > 0x7fffffffU || height > 0x7fffffffU) {
            throw std::invalid_argument("decoded media dimensions are invalid");
        }
        header.width  = static_cast<int>(width);
        header.height = static_cast<int>(height);
        return header;
    }
    if (bytes.size() >= 10 && (std::memcmp(bytes.data(), "GIF87a", 6) == 0 || std::memcmp(bytes.data(), "GIF89a", 6) == 0)) {
        header.width  = le16(6);  // the logical screen: Pillow's size, FFmpeg's canvas
        header.height = le16(8);
        if (header.width <= 0 || header.height <= 0) { throw std::invalid_argument("decoded media dimensions are invalid"); }
        return header;
    }
    if (bytes.size() >= 26 && bytes[0] == 'B' && bytes[1] == 'M' && le32(14) >= 40) {
        header.width  = le32(18);
        header.height = le32(22) == std::numeric_limits<std::int32_t>::min() ? 0 : std::abs(le32(22));  // < 0: top-down
        if (header.width <= 0 || header.height <= 0) { throw std::invalid_argument("decoded media dimensions are invalid"); }
        return header;
    }
    Decoder decoder(bytes, std::numeric_limits<std::uint64_t>::max());
    header.width  = decoder.stream()->codecpar->width;
    header.height = decoder.stream()->codecpar->height;
    if (header.width <= 0 || header.height <= 0) { throw std::invalid_argument("decoded media dimensions are invalid"); }
    return header;
}

Video decode_video(std::span<const std::uint8_t> bytes, const Policy& policy, double target_fps,
                   int min_frames, int max_frames, bool temporal_pairs) {
    validate_input(bytes, policy);
    if (policy.checkpoint) { policy.checkpoint(); }
    Decoder probe(bytes, policy.max_decoded_pixels);
    const double fps      = fps_of(probe.stream());
    int total             = probe.stream()->nb_frames > 0 &&
                        probe.stream()->nb_frames <= std::numeric_limits<int>::max()
                                ? static_cast<int>(probe.stream()->nb_frames)
                                : 0;
    const double duration = probe.duration_seconds();
    if (total == 0 && duration > 0.0) { total = static_cast<int>(std::nearbyint(duration * fps)); }
    if (duration > policy.max_video_duration_seconds) {
        throw Error(ErrorKind::BudgetExceeded, "video duration exceeds processor limit");
    }
    if (total == 0) { total = count_frames(bytes, policy); }
    if (total > policy.max_video_source_frames) {
        throw Error(ErrorKind::BudgetExceeded, "video source frame count exceeds processor limit");
    }
    const std::vector<int> indices = sample_indices(total, fps, target_fps, min_frames, max_frames, temporal_pairs);
    const std::uint64_t coded_pixels =
        static_cast<std::uint64_t>(std::max(probe.stream()->codecpar->width, 0)) *
        static_cast<std::uint64_t>(std::max(probe.stream()->codecpar->height, 0));
    if (coded_pixels != 0 && indices.size() > policy.max_decoded_video_pixels / coded_pixels) {
        throw Error(ErrorKind::BudgetExceeded,
                    "sampled source video pixels exceed processor limit");
    }

    Decoder decoder(bytes, policy.max_decoded_pixels);
    const int orientation = orientation_from_rotation(rotation_of(decoder.stream()));
    Video out;
    out.total_frames = total;
    out.fps          = fps;
    out.duration     = duration > 0.0 ? duration : static_cast<double>(total) / fps;
    out.indices      = indices;
    out.frames.reserve(indices.size());
    std::size_t wanted            = 0;
    int decoded                   = 0;
    std::uint64_t retained_pixels = 0;
    decoder.frames([&](int index, const AVFrame* frame) {
        if (policy.checkpoint) { policy.checkpoint(); }
        decoded = index + 1;
        if (wanted < indices.size() && index == indices[wanted]) {
            const std::uint64_t pixels = static_cast<std::uint64_t>(std::max(frame->width, 0)) *
                                         static_cast<std::uint64_t>(std::max(frame->height, 0));
            if (retained_pixels > policy.max_decoded_video_pixels ||
                pixels > policy.max_decoded_video_pixels - retained_pixels) {
                throw Error(ErrorKind::BudgetExceeded,
                            "sampled source video pixels exceed processor limit");
            }
            retained_pixels += pixels;
            out.frames.push_back(decoder.rgb(frame, orientation, true));
            ++wanted;
        }
        return wanted != indices.size();
    });
    const std::size_t required = static_cast<std::size_t>(std::min(min_frames, total));
    if (out.frames.size() < required) {
        throw std::runtime_error("video decoder produced fewer than the minimum sampled frames");
    }
    for (std::size_t i = wanted; i < indices.size(); ++i) {
        if (indices[i] < decoded) {
            throw std::runtime_error("video decoder skipped a requested internal frame");
        }
    }
    out.width  = out.frames.front().width;
    out.height = out.frames.front().height;
    return out;
}

} // namespace sinfer::media::decode
