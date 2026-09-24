// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#include "audio.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>
}

namespace sinfer::speech {
std::vector<float> decode_audio(const std::string& bytes) {
    struct Input {
        const std::string& bytes;
        size_t offset = 0;
    } input{bytes};

    auto read = [](void* opaque, uint8_t* data, int size) -> int {
        auto& in = *static_cast<Input*>(opaque);
        size_t n = std::min<size_t>(size, in.bytes.size() - in.offset);
        if (!n) return AVERROR_EOF;
        std::memcpy(data, in.bytes.data() + in.offset, n);
        in.offset += n;
        return n;
    };
    // The whole upload is in memory, so it can seek. Without this FFmpeg cannot learn the file's
    // size: the MP3 demuxer then ignores the encoder's gapless header and counts its padding as
    // audio (billing up to a frame too much), and an M4A whose index comes after the audio (the
    // usual layout of phone recordings) cannot be read at all.
    auto seek = [](void* opaque, int64_t offset, int whence) -> int64_t {
        auto& in         = *static_cast<Input*>(opaque);
        const auto total = static_cast<int64_t>(in.bytes.size());
        if (whence & AVSEEK_SIZE) return total;
        whence &= ~AVSEEK_FORCE;
        int64_t base = 0;
        if (whence == SEEK_CUR) base = static_cast<int64_t>(in.offset);
        else if (whence == SEEK_END) base = total;
        else if (whence != SEEK_SET) return AVERROR(EINVAL);
        const int64_t position = base + offset;
        if (position < 0 || position > total) return AVERROR(EINVAL);
        in.offset = static_cast<size_t>(position);
        return position;
    };

    struct Resources {
        AVFormatContext* format = nullptr;
        AVIOContext* io         = nullptr;
        AVCodecContext* codec   = nullptr;
        SwrContext* resampler   = nullptr;
        AVFrame* frame          = av_frame_alloc();
        AVPacket* packet        = av_packet_alloc();

        ~Resources() {
            av_packet_free(&packet);
            av_frame_free(&frame);
            swr_free(&resampler);
            avcodec_free_context(&codec);
            avformat_close_input(&format);
            if (io) {
                av_freep(&io->buffer);
                avio_context_free(&io);
            }
        }
    } r;

    r.io = avio_alloc_context(static_cast<unsigned char*>(av_malloc(32768)), 32768, 0, &input, read,
                              nullptr, seek);
    r.format = avformat_alloc_context();
    if (!r.io || !r.format || !r.packet || !r.frame)
        throw std::runtime_error("audio allocation failed");
    r.format->pb = r.io;
    r.format->flags |= AVFMT_FLAG_CUSTOM_IO;
    AVDictionary* options = nullptr;
    av_dict_set(&options, "protocol_whitelist", "", 0);
    av_dict_set(&options, "format_whitelist", "wav,flac,mp3,ogg,matroska,webm,mov,aac", 0);
    int opened = avformat_open_input(&r.format, nullptr, nullptr, &options);
    av_dict_free(&options);
    if (opened < 0 || avformat_find_stream_info(r.format, nullptr) < 0)
        throw std::invalid_argument("invalid or unsupported audio file");
    int stream = av_find_best_stream(r.format, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (stream < 0) throw std::invalid_argument("file contains no audio");
    auto params  = r.format->streams[stream]->codecpar;
    auto decoder = avcodec_find_decoder(params->codec_id);
    if (!decoder) throw std::invalid_argument("unsupported audio codec");
    r.codec = avcodec_alloc_context3(decoder);
    if (!r.codec || avcodec_parameters_to_context(r.codec, params) < 0 ||
        avcodec_open2(r.codec, decoder, nullptr) < 0)
        throw std::invalid_argument("cannot open audio decoder");
    AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
    if (swr_alloc_set_opts2(&r.resampler, &mono, AV_SAMPLE_FMT_FLT, 16000, &r.codec->ch_layout,
                            r.codec->sample_fmt, r.codec->sample_rate, 0, nullptr) < 0 ||
        swr_init(r.resampler) < 0)
        throw std::invalid_argument("unsupported audio sample layout");
    std::vector<float> output;
    auto convert = [&](const uint8_t** data, int count) {
        int capacity = std::max(1, swr_get_out_samples(r.resampler, count));
        if (capacity > 9600512 || output.size() + capacity > 9600512)
            throw std::invalid_argument("audio exceeds 10 minutes");
        size_t start = output.size();
        output.resize(start + capacity);
        auto destination = reinterpret_cast<uint8_t*>(output.data() + start);
        int n            = swr_convert(r.resampler, &destination, capacity, data, count);
        if (n < 0) throw std::invalid_argument("audio resampling failed");
        output.resize(start + n);
        if (output.size() > 9600000) throw std::invalid_argument("audio exceeds 10 minutes");
    };
    auto receive = [&]() {
        int status;
        while ((status = avcodec_receive_frame(r.codec, r.frame)) >= 0) {
            if (r.frame->format != r.codec->sample_fmt ||
                r.frame->sample_rate != r.codec->sample_rate ||
                av_channel_layout_compare(&r.frame->ch_layout, &r.codec->ch_layout))
                throw std::invalid_argument("audio layout changes within the file");
            convert(const_cast<const uint8_t**>(r.frame->extended_data), r.frame->nb_samples);
            av_frame_unref(r.frame);
        }
        if (status != AVERROR(EAGAIN) && status != AVERROR_EOF)
            throw std::invalid_argument("invalid encoded audio");
    };
    int status;
    while ((status = av_read_frame(r.format, r.packet)) >= 0) {
        if (r.packet->stream_index == stream) {
            if (avcodec_send_packet(r.codec, r.packet) < 0)
                throw std::invalid_argument("invalid audio packet");
            receive();
        }
        av_packet_unref(r.packet);
    }
    if (status != AVERROR_EOF) throw std::invalid_argument("truncated audio file");
    avcodec_send_packet(r.codec, nullptr);
    receive();
    convert(nullptr, 0);
    if (output.empty()) throw std::invalid_argument("audio is empty");
    for (auto& v : output) v = std::clamp(v, -1.f, 1.f);
    return output;
}
} // namespace sinfer::speech
