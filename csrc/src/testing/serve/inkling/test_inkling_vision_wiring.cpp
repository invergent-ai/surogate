// Inkling-2 image requests through the serving layer and the family Frontend.
//
// What sinfer_inkling_vision_test proves for the functions, this proves for the path a request takes: the
// chat-completions parser in raw-chat mode collects the body's image parts (GenerationRequest::raw_chat_images),
// translate acquires their bytes (PromptInput::raw_chat_media; here from the data: URLs), and Frontend::prepare (vision
// on, image limit 8 as the vector capture, the default SERVED decode) returns vLLM's prompt ids, one VisionItem per
// image (grid in native patches, one token span over its pads, the payload = the normalisation of the pixels vLLM's
// server fed its encoder), image token types on the pads, 1-D positions; count_tokens agrees;
// a second prepare of the same images hits the media cache; refusals carry vLLM's words; a frontend with vision off
// refuses every image request as vLLM at image limit 0. Vectors: vectors/vision (not in this repository).
// CPU only. Skips (77) without INKLING_MODEL_DIR / INKLING_VECTORS.

#include <api/family/frontend.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "family/impl/frontend/inkling_render.h"
#include "family/impl/frontend/inkling_vision.h"
#include "family/impl/frontend/test_access.h"
#include "serve/openai_schema.h"
#include "serve/request.h"
#include "serve/serve_options.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs  = std::filesystem;
namespace ink = sinfer::family::frontend_internal::inkling;
using Json    = nlohmann::ordered_json;  // bodies keep the client's key order
using sinfer::family::FrontendResources;
using sinfer::family::FrontendTestAccess;

namespace {

int g_reported = 0;

void report(const std::string& what) {
    if (g_reported++ < 40) { std::cout << "  MISMATCH " << what << "\n"; }
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + p.string()); }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct Tensor {
    std::vector<std::int64_t> shape;
    std::string bytes;
    template <class T>
    std::vector<T> as() const {
        std::vector<T> out(bytes.size() / sizeof(T));
        std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
        return out;
    }
};

std::map<std::string, Tensor> load_safetensors(const fs::path& p) {
    const std::string raw = read_file(p);
    std::uint64_t n = 0;
    std::memcpy(&n, raw.data(), 8);
    const Json header = Json::parse(raw.substr(8, n));
    std::map<std::string, Tensor> out;
    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") { continue; }
        Tensor t;
        t.shape = it.value().at("shape").get<std::vector<std::int64_t>>();
        const auto offsets = it.value().at("data_offsets").get<std::vector<std::uint64_t>>();
        t.bytes = raw.substr(8 + n + offsets[0], offsets[1] - offsets[0]);
        out.emplace(it.key(), std::move(t));
    }
    return out;
}

std::vector<std::uint8_t> base64_decode(std::string_view in) {
    static const auto table = [] {
        std::array<int, 256> t{};
        t.fill(-1);
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) { t[static_cast<unsigned char>(a[i])] = i; }
        return t;
    }();
    std::vector<std::uint8_t> out;
    unsigned acc = 0;
    int bits     = 0;
    for (const char ch : in) {
        if (ch == '=') { break; }
        const int v = table[static_cast<unsigned char>(ch)];
        if (v < 0) { throw std::runtime_error("bad base64"); }
        acc = (acc << 6U) | static_cast<unsigned>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>((acc >> bits) & 0xffU));
        }
    }
    return out;
}

// The serving layer's acquisition, for data: URLs (the product path runs media_acquire::acquire_bytes here).
sinfer::OwnedMedia acquire(const sinfer::serve::ContentPart& part) {
    const std::string& url = part.source.value;
    const std::size_t comma = url.find(',');
    if (part.source.kind != sinfer::product::media_acquire::SourceKind::Data || comma == std::string::npos) {
        throw std::runtime_error("test acquires data: URLs only");
    }
    sinfer::OwnedMedia media;
    media.kind        = sinfer::MediaKind::Image;
    media.source_name = "inline-data";
    media.bytes       = base64_decode(std::string_view(url).substr(comma + 1));
    return media;
}

// vLLM's recorded refusal without the exception type ("ValueError: ") and VLLMValidationError's " (parameter=...)".
std::string served_text(std::string error) {
    const std::size_t colon = error.find(": ");
    if (colon != std::string::npos && error.find_first_of(" \n") > colon) { error = error.substr(colon + 2); }
    const std::size_t param = error.rfind(" (parameter=");
    if (param != std::string::npos && error.back() == ')') { error = error.substr(0, param); }
    return error;
}

// GenerationService::prepare's order: the Inkling parse with the server's vision flag (999 images on, 0 off) comes
// before translate acquires anything.
sinfer::PromptInput serve_input(const std::string& body, bool server_vision = true) {
    (void)ink::parse_chat_request(body, server_vision ? 999 : 0);
    sinfer::serve::RequestLimits limits;
    limits.raw_chat = true;
    sinfer::serve::GenerationRequest request =
        sinfer::serve::parse_chat_completion_request(nlohmann::ordered_json::parse(body), limits);
    request.raw_chat_body = std::make_shared<const std::string>(body);
    const sinfer::serve::ResolvedPromptSemantics semantics{};
    return sinfer::serve::to_prompt_input(request, semantics, acquire);
}

} // namespace

int main() {
    const char* model_dir   = std::getenv("INKLING_MODEL_DIR");
    const char* vectors_dir = std::getenv("INKLING_VECTORS");
    if (model_dir == nullptr || vectors_dir == nullptr) {
        std::cout << "SKIP: set INKLING_MODEL_DIR and INKLING_VECTORS\n";
        return 77;
    }
    const fs::path model(model_dir);
    const fs::path vision = fs::path(vectors_dir) / "vision";
    FrontendResources resources;
    resources.tokenizer_json         = read_file(model / "tokenizer.json");
    resources.tokenizer_config_json  = read_file(model / "tokenizer_config.json");
    resources.generation_config_json = read_file(model / "generation_config.json");
    sinfer::family::FrontendOptions on;
    on.vision_enabled         = true;
    on.max_context            = 262'144;
    on.image_limit_per_prompt = 8;  // the vector capture: --limit-mm-per-prompt '{"image": 8}'
    const sinfer::family::Frontend frontend = FrontendTestAccess::create_component_with(resources, on);
    sinfer::family::FrontendOptions off = on;
    off.vision_enabled                       = false;
    const sinfer::family::Frontend text_only = FrontendTestAccess::create_component_with(resources, off);

    Json meta = Json::object();
    if (fs::exists(vision / "processor" / "meta.json")) { meta = Json::parse(read_file(vision / "processor" / "meta.json")); }
    int failures = 0, cases = 0, ok_cases = 0, payload_exact = 0, payload_checked = 0, cache_ok = 0, off_ok = 0;
    std::ifstream lines(vision / "cases" / "cases.jsonl");
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty()) { continue; }
        const Json c             = Json::parse(line);
        const std::string id     = c.at("id").get<std::string>();
        const std::string body   = c.at("body").dump();
        std::string expect       = c.at("expect").get<std::string>();
        std::string exact;
        if (meta.contains("cases") && meta["cases"].contains(id) && meta["cases"][id].contains("serve")) {
            const Json& sv = meta["cases"][id]["serve"];
            if (sv.contains("ok") && sv["ok"].is_boolean()) {
                if (sv["ok"].get<bool>()) { expect = "ok"; }
                else if (sv.contains("error") && sv["error"].is_string()) {
                    exact  = served_text(sv["error"].get<std::string>());
                    expect = "error:" + exact;
                }
            }
        }
        const auto names         = c.at("images").get<std::vector<std::string>>();
        ++cases;
        bool ok = true;
        // vision off: vLLM at image limit 0
        try {
            (void)text_only.prepare(serve_input(body, false));
            report(id + ": vision off accepted an image request");
            ok = false;
        } catch (const std::exception& e) {
            if (std::string(e.what()) == "At most 0 image(s) may be provided in one prompt.") { ++off_ok; }
            else {
                report(id + ": vision off refused with '" + e.what() + "'");
                ok = false;
            }
        }
        std::string error;
        try {
            const sinfer::family::PreparedPrompt prompt = frontend.prepare(serve_input(body));
            const sinfer::family::PreparedPromptData& data = FrontendTestAccess::inspect(prompt);
            const std::uint32_t counted = frontend.count_tokens(serve_input(body));
            const auto tensors = load_safetensors(vision / "processor" / (id + ".safetensors"));
            const bool default_effort = !c.at("body").contains("chat_template_kwargs") ||
                                        !c.at("body").at("chat_template_kwargs").contains("reasoning_effort");
            const auto want = tensors.at(default_effort && tensors.count("prompt_token_ids_099") ? "prompt_token_ids_099"
                                                                                                 : "prompt_token_ids")
                                  .as<std::int32_t>();
            if (!std::equal(want.begin(), want.end(), data.token_ids.begin(), data.token_ids.end())) {
                report(id + ": prompt ids differ");
                ok = false;
            }
            if (counted != data.token_ids.size()) {
                report(id + ": count_tokens " + std::to_string(counted));
                ok = false;
            }
            if (data.vision_items.size() != names.size() || data.media_payloads.size() != names.size()) {
                report(id + ": " + std::to_string(data.vision_items.size()) + " vision items");
                ok = false;
            }
            std::size_t pads = 0, patch_begin = 0;
            for (std::size_t k = 0; ok && k < data.vision_items.size(); ++k) {
                const sinfer::family::VisionItem& item = data.vision_items[k];
                const std::string gk = std::to_string(k);
                const auto grid = (tensors.count("grid_served_" + gk) ? tensors.at("grid_served_" + gk) : tensors.at("grid_" + gk))
                                      .as<std::int64_t>();  // the served grid (an EXIF-rotated image's rows / cols swap)
                const std::uint64_t n = static_cast<std::uint64_t>(grid[1] * grid[2]);
                const bool shape = item.modality == sinfer::family::PromptModality::Image && item.grid.temporal == 1 &&
                                   item.grid.height == grid[1] * 2 && item.grid.width == grid[2] * 2 &&
                                   item.patch_count == 4 * n && item.patch_begin == patch_begin &&
                                   item.token_spans.size() == 1 && item.token_spans[0].count == n &&
                                   data.media_payloads[k] &&
                                   data.media_payloads[k]->patch_elements == 4 * n * ink::kPatchFeatures;
                if (!shape) {
                    report(id + ": vision item " + std::to_string(k) + " shape / span / payload size");
                    ok = false;
                    break;
                }
                patch_begin += item.patch_count;
                const std::size_t b = item.token_spans[0].begin;
                if (data.token_ids[b - 1] != 200005 ||
                    !std::all_of(data.token_ids.begin() + static_cast<std::ptrdiff_t>(b),
                                 data.token_ids.begin() + static_cast<std::ptrdiff_t>(b + n),
                                 [](sinfer::TokenId t) { return t == 200054; })) {
                    report(id + ": span " + std::to_string(k) + " is not the pads after its marker");
                    ok = false;
                }
                pads += n;
                // The pixels vLLM's server fed its encoder (vectors: decoded_rgb_served_k where they differ from
                // processing.py's decoded_rgb_k), normalised (sinfer_inkling_vision_test proves the normalisation
                // against gather_patches): the payload must be exactly that.
                const std::string ks = std::to_string(k);
                const Tensor& rgb = tensors.count("decoded_rgb_served_" + ks) ? tensors.at("decoded_rgb_served_" + ks)
                                                                              : tensors.at("decoded_rgb_" + ks);
                ink::DecodedImage pil;
                pil.height = static_cast<int>(rgb.shape.at(0));
                pil.width  = static_cast<int>(rgb.shape.at(1));
                pil.rgb    = rgb.as<std::uint8_t>();
                std::vector<std::uint16_t> want(static_cast<std::size_t>(4 * n * ink::kPatchFeatures));
                ink::write_patches_merge(pil, want);
                ++payload_checked;
                const auto got = data.media_payloads[k]->span();
                if (std::equal(want.begin(), want.end(), got.begin(), got.end())) { ++payload_exact; }
                else {
                    report(id + ": payload " + names[k] + " differs from the normalised server pixels");
                    ok = false;
                }
            }
            std::size_t typed = 0;
            for (std::size_t i = 0; i < data.token_types.size(); ++i) {
                const bool pad = data.token_ids[i] == 200054;
                typed += data.token_types[i] != 0;
                if ((data.token_types[i] == 1) != pad) {
                    report(id + ": token type at " + std::to_string(i));
                    ok = false;
                    break;
                }
            }
            const std::size_t t = data.token_ids.size();
            for (std::size_t i = 0; i < 3 * t && ok; ++i) {
                if (data.positions.size() != 3 * t || data.positions[i] != static_cast<std::int32_t>(i % t)) {
                    report(id + ": positions are not 1-D");
                    ok = false;
                }
            }
            if (typed != pads) { ok = false; }
            // the same images again: every one a media-cache hit
            const sinfer::family::PreparedPrompt again = frontend.prepare(serve_input(body));
            const auto& data2 = FrontendTestAccess::inspect(again);
            if (data2.prepare.media_cache_hits == names.size() && data2.token_ids == data.token_ids) { ++cache_ok; }
            else {
                report(id + ": second prepare: " + std::to_string(data2.prepare.media_cache_hits) + " cache hits");
                ok = false;
            }
        } catch (const std::exception& e) {
            error = e.what();
        }
        if (expect.starts_with("error:")) {
            const bool refused = !error.empty() && error.find(expect.substr(6)) != std::string::npos &&
                                 (exact.empty() || error == exact);
            std::cout << "  refusal " << id << ": " << (error.empty() ? "(accepted)" : error) << "\n";
            if (!refused) {
                report(id + ": expected a refusal");
                ok = false;
            } else {
                ok = true;  // the checks above ran on no prompt
            }
        } else if (!error.empty()) {
            report(id + ": refused: " + error);
            ok = false;
        }
        if (ok) { ++ok_cases; } else { ++failures; }
    }
    // Review fixes: the per-prompt Vision budget is checked from the headers before any payload is held; a URL
    // the chat parser cannot take is refused after the renderer's own checks (vision off -> vLLM's limit-0 text).
    {
        const std::string big = read_file(vision / "cases" / "img" / "grid_2560x2560.png");  // 4,096 tokens each
        static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string b64;
        for (std::size_t i = 0; i < big.size(); i += 3) {
            const unsigned a0 = static_cast<unsigned char>(big[i]);
            const unsigned a1 = i + 1 < big.size() ? static_cast<unsigned char>(big[i + 1]) : 0U;
            const unsigned a2 = i + 2 < big.size() ? static_cast<unsigned char>(big[i + 2]) : 0U;
            b64 += alphabet[a0 >> 2U];
            b64 += alphabet[((a0 & 3U) << 4U) | (a1 >> 4U)];
            b64 += i + 1 < big.size() ? alphabet[((a1 & 15U) << 2U) | (a2 >> 6U)] : '=';
            b64 += i + 2 < big.size() ? alphabet[a2 & 63U] : '=';
        }
        Json parts = Json::array();
        for (int i = 0; i < 9; ++i) { parts.push_back({{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64," + b64}}}}); }
        const std::string body = Json{{"model", "inkling"}, {"messages", Json::array({{{"role", "user"}, {"content", parts}}})},
                                      {"chat_template_kwargs", {{"reasoning_effort", 0.99}}}}.dump();
        sinfer::family::FrontendOptions wide = on;
        wide.image_limit_per_prompt = 16;
        const sinfer::family::Frontend frontend16 = FrontendTestAccess::create_component_with(resources, wide);
        try {
            (void)frontend16.prepare(serve_input(body));
            report("9 x 4,096-token images (36,864 > the 32,768 Vision budget) were accepted");
            ++failures;
        } catch (const sinfer::RequestError& e) {
            const bool ok = e.kind() == sinfer::RequestErrorKind::MediaBudgetExceeded &&
                            std::string(e.what()) == "vision tokens exceed processor budget";
            std::cout << "  budget: 9 x 4,096 image tokens -> " << e.what() << (ok ? "" : "  <- WRONG") << "\n";
            if (!ok) { ++failures; }
        } catch (const std::exception& e) {
            report(std::string("budget refusal of the wrong type: ") + e.what());
            ++failures;
        }
        const std::string file_url =
            R"({"model":"inkling","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///etc/hostname"}}]}]})";
        const auto refusal = [&](const sinfer::family::Frontend& f) -> std::string {
            try {
                (void)f.prepare(serve_input(file_url, &f != &text_only));
                return "(accepted)";
            } catch (const std::exception& e) { return e.what(); }
        };
        const std::string off_text = refusal(text_only), on_text = refusal(frontend);
        const bool ok = off_text == "At most 0 image(s) may be provided in one prompt." &&
                        on_text == "image_url must use HTTP(S) or a data URI";
        std::cout << "  file:// URL: vision off -> '" << off_text << "', on -> '" << on_text << "'" << (ok ? "" : "  <- WRONG") << "\n";
        if (!ok) { ++failures; }
    }
    std::cout << "cases " << cases << ": " << ok_cases << " ok through parse -> translate -> Frontend::prepare / count_tokens\n"
              << "payloads == the normalised pixels vLLM's server fed its encoder: " << payload_exact << " / " << payload_checked << "\n"
              << "second prepare all media-cache hits: " << cache_ok << "\n"
              << "vision off -> 'At most 0 image(s) may be provided in one prompt.': " << off_ok << " / " << cases << "\n"
              << (failures == 0 ? "PASS" : "FAIL") << " (" << failures << " failures)\n";
    return failures == 0 ? 0 : 1;
}
