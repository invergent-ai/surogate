// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
// What a finished file transcription returns, for every response_format. Model-free, so the
// reply contract is testable without the speech model.
#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sinfer::speech {

/// Decoded audio is mono at this rate (see decode_audio).
inline constexpr int kTranscriptionSampleRate = 16000;

/// The response header that carries the billed audio length, in seconds, on every successful
/// transcription, whatever its format. A gateway can meter it without parsing the audio or
/// buffering the body.
inline constexpr std::string_view kAudioDurationHeader = "X-Audio-Duration-Seconds";

/// The seconds of audio a transcription is billed for: the decoded 16 kHz samples. Exact to one
/// sample; one sample is 62.5 microseconds, so the value has at most seven decimals.
[[nodiscard]] inline double transcription_seconds(std::size_t samples) {
    return static_cast<double>(samples) / kTranscriptionSampleRate;
}

struct TranscriptionReply {
    std::string body;
    std::string content_type;
    std::vector<std::pair<std::string, std::string>> headers;
};

/// The reply for `text`, transcribed from `samples` decoded samples, in `format` (json, text or
/// verbose_json; the caller has validated it).
///
/// Every format reports the duration. The JSON formats carry OpenAI's usage object for models
/// billed by audio length, `{"type": "duration", "seconds": s}`; verbose_json keeps its own
/// `duration` field as well. `text` has no body to carry it, so the header is the only place for
/// it there, and every format sends the header. The seconds are the same number in all places,
/// written in the shortest form that reads back as the same double.
[[nodiscard]] inline TranscriptionReply transcription_reply(std::string_view format, const std::string& text,
                                                            std::size_t samples) {
    using json            = nlohmann::ordered_json;
    const double seconds  = transcription_seconds(samples);
    const json usage      = {{"type", "duration"}, {"seconds", seconds}};
    TranscriptionReply reply;
    reply.headers.emplace_back(std::string(kAudioDurationHeader), json(seconds).dump());
    if (format == "text") {
        reply.body         = text;
        reply.content_type = "text/plain; charset=utf-8";
    } else if (format == "verbose_json") {
        reply.body = json{{"text", text},
                          {"language", "romanian"},
                          {"duration", seconds},
                          {"task", "transcribe"},
                          {"usage", usage}}
                         .dump();
        reply.content_type = "application/json";
    } else {
        reply.body         = json{{"text", text}, {"usage", usage}}.dump();
        reply.content_type = "application/json";
    }
    return reply;
}

} // namespace sinfer::speech
