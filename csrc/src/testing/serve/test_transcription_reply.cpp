// Every successful file transcription reports the billed audio length (SUROGATE-CHANGES #3):
// a usage object in the JSON formats and a header on every format, so a gateway can meter it
// without decoding the audio. Model-free: the reply is built from the text and the sample count.
#include "transcription.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

using sinfer::speech::kAudioDurationHeader;
using sinfer::speech::transcription_reply;
using sinfer::speech::transcription_seconds;
using json = nlohmann::ordered_json;

namespace {

std::string header(const sinfer::speech::TranscriptionReply& reply) {
    std::string value;
    int count = 0;
    for (const auto& [name, content] : reply.headers) {
        if (name == kAudioDurationHeader) {
            value = content;
            ++count;
        }
    }
    assert(count == 1);
    return value;
}

} // namespace

int main() {
    assert(kAudioDurationHeader == "X-Audio-Duration-Seconds");
    const std::string text = "Bună ziua, comanda a ajuns.";

    // The seconds are the decoded 16 kHz samples, exact to one sample, and read back as the same
    // double from the header text.
    struct Case {
        std::size_t samples;
        const char* written;
    };
    for (const Case& c : {Case{0, "0.0"}, Case{1, "6.25e-05"}, Case{16000, "1.0"}, Case{89121, "5.5700625"},
                          Case{160000 * 6 + 8, "60.0005"}, Case{16000ULL * 3600, "3600.0"}}) {
        const double seconds = transcription_seconds(c.samples);
        assert(seconds == static_cast<double>(c.samples) / 16000.0);
        for (const char* format : {"json", "text", "verbose_json"}) {
            const auto reply = transcription_reply(format, text, c.samples);
            const std::string value = header(reply);
            if (value != c.written) { std::cerr << format << ": header " << value << " != " << c.written << '\n'; }
            assert(value == c.written);
            assert(std::strtod(value.c_str(), nullptr) == seconds);
        }
    }

    // json: the text as before, plus OpenAI's usage object for audio billed by duration.
    {
        const auto reply = transcription_reply("json", text, 89121);
        assert(reply.content_type == "application/json");
        const json body = json::parse(reply.body);
        assert(body.size() == 2 && body.begin().key() == "text" && body["text"] == text);
        assert(body["usage"] == json({{"type", "duration"}, {"seconds", 5.5700625}}));
        assert(reply.body.find("Bună") != std::string::npos); // UTF-8 as is, not \u escapes
    }
    // verbose_json: its fields as before, in order, with the same seconds in `duration` and usage.
    {
        const auto reply = transcription_reply("verbose_json", text, 89121);
        assert(reply.content_type == "application/json");
        const json body = json::parse(reply.body);
        std::string keys;
        for (auto it = body.begin(); it != body.end(); ++it) keys += it.key() + ",";
        assert(keys == "text,language,duration,task,usage,");
        assert(body["text"] == text && body["language"] == "romanian" && body["task"] == "transcribe");
        assert(body["duration"].get<double>() == 5.5700625);
        assert(body["usage"]["seconds"].get<double>() == body["duration"].get<double>());
        assert(body["usage"]["type"] == "duration");
    }
    // text: the body is the transcript and nothing else; the header carries the seconds.
    {
        const auto reply = transcription_reply("text", text, 89121);
        assert(reply.body == text && reply.content_type == "text/plain; charset=utf-8");
        assert(header(reply) == "5.5700625");
    }
    // An empty transcript (silence, or audio shorter than a frame) is still billed for its length.
    {
        const auto reply = transcription_reply("json", "", 300);
        assert(json::parse(reply.body)["text"] == "" && header(reply) == "0.01875");
    }

    std::cout << "transcription reply checks passed\n";
    return 0;
}
