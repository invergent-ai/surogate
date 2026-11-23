// Inkling-2 image requests -> prompt ids + the vision payload, against vLLM's own code.
//
// Vectors (vectors/vision, not in this repository, made inside vllm/vllm-openai:inkling-2-mini-x86_64-7cb5dae):
//   cases/cases.jsonl                 chat bodies (images as data: URLs of cases/img/*), expected outcome
//   processor/<case>.safetensors      prompt_token_ids (renderer + multimodal prompt update), bare_token_ids,
//                                     *_099 for a body without an effort (the engine's default 0.99),
//                                     decoded_rgb_k (vLLM decode_image: Pillow), pixels_raw_k (vision_pixels), grid_k
//   processor/patches/<image>.safetensors   patches_merge (the engine payload order), positions_merge,
//                                     patches_vllm_raster (vLLM's gather_patches on a card: THE oracle for the payload)
//   processor/meta.json               per case serve_error (vLLM's exact refusal through a running engine), when recorded
//   *_served_k                        what vLLM's SERVER fed the encoder where it differs from processing.py's path
//                                     (measured on an RTX 5090: Pillow, EXIF transpose, alpha on white, no EOI fix)
//   meta.json cases.<id>.serve        a running vLLM's outcome (ok / its refusal text)
// Both decode paths run (inkling_vision.h DecodePath): Served against the server's pixels and refusals, Processor
// against processing.py's. Checks, per case: refusal text, prompt ids; per image: our decode vs Pillow's (decode +
// Pillow's mode rules; JPEG / WebP with Pillow's own libraries), grid, raw pixels, the payload from our decode and from
// Pillow's pixels vs gather_patches. A format decoded outside Pillow's libraries reports its delta (the decoder-
// fidelity measurement) and gates nothing; everything else is bit for bit. CPU only. Exits 77 (skip) without INKLING_MODEL_DIR (tokenizer.json, tokenizer_config.json,
// generation_config.json) and INKLING_VECTORS. INKLING_IMAGE_LIMIT (default 8 = the capture's --limit-mm-per-prompt).

#include "family/impl/frontend/inkling_json.h"
#include "family/impl/frontend/inkling_render.h"
#include "family/impl/frontend/inkling_vision.h"
#include "family/impl/frontend/tokenizer.h"
#include "media/decode/decode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace fi = sinfer::family::frontend_internal;
namespace ink = sinfer::family::frontend_internal::inkling;
// Ordered: a body is re-serialised in the client's key order (Python dicts keep it; reprs and renders depend on it).
using Json = nlohmann::ordered_json;

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
    std::string dtype;
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
        t.dtype = it.value().at("dtype").get<std::string>();
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
    out.reserve(in.size() * 3 / 4);
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

std::vector<std::uint8_t> data_url_bytes(const std::string& url) {
    const std::size_t comma = url.find(',');
    if (!url.starts_with("data:") || comma == std::string::npos || url.substr(0, comma).find(";base64") == std::string::npos) {
        throw std::runtime_error("test expects base64 data: URLs");
    }
    return base64_decode(std::string_view(url).substr(comma + 1));
}

struct Delta {
    std::size_t values = 0, differing = 0, pixels_differing = 0;
    int max_abs = 0;
    double sum_abs = 0.0;
    std::array<std::size_t, 5> hist{};  // |d| = 1, 2, 3, 4-7, >= 8
};

Delta rgb_delta(const ink::DecodedImage& ours, const Tensor& pil) {
    Delta d;
    const auto ref = pil.as<std::uint8_t>();
    d.values = ref.size();
    if (ours.rgb.size() != ref.size()) {
        d.differing = d.values;
        d.max_abs   = 255;
        return d;
    }
    for (std::size_t i = 0; i < ref.size(); i += 3) {
        bool px = false;
        for (int c = 0; c < 3; ++c) {
            const int diff = std::abs(static_cast<int>(ours.rgb[i + c]) - static_cast<int>(ref[i + c]));
            if (diff == 0) { continue; }
            px = true;
            ++d.differing;
            d.max_abs = std::max(d.max_abs, diff);
            d.sum_abs += diff;
            ++d.hist[diff == 1 ? 0 : diff == 2 ? 1 : diff == 3 ? 2 : diff < 8 ? 3 : 4];
        }
        if (px) { ++d.pixels_differing; }
    }
    return d;
}

std::size_t differing(const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
    if (a.size() != b.size()) { return std::max(a.size(), b.size()); }
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { n += a[i] != b[i]; }
    return n;
}

ink::DecodedImage from_pil(const Tensor& rgb) {
    ink::DecodedImage image;
    image.height = static_cast<int>(rgb.shape.at(0));
    image.width  = static_cast<int>(rgb.shape.at(1));
    image.rgb    = rgb.as<std::uint8_t>();
    return image;
}

std::vector<std::uint16_t> payload(const ink::DecodedImage& image) {
    const ink::ImageGrid grid = ink::image_grid(image.height, image.width);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(grid.tokens()) * ink::kPatchesPerToken * ink::kPatchFeatures);
    ink::write_patches_merge(image, out);
    return out;
}

// Merge-order rows -> vLLM's native raster order (row y * 2cols + x).
std::vector<std::uint16_t> merge_to_raster(const std::vector<std::uint16_t>& merge, ink::ImageGrid grid) {
    std::vector<std::uint16_t> out(merge.size());
    const int w = grid.cols * 2;
    for (int y = 0; y < grid.rows * 2; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t m = (static_cast<std::size_t>(y / 2) * grid.cols + x / 2) * 4 + (y % 2) * 2 + x % 2;
            const std::size_t r = static_cast<std::size_t>(y) * w + x;
            std::memcpy(out.data() + r * ink::kPatchFeatures, merge.data() + m * ink::kPatchFeatures,
                        ink::kPatchFeatures * sizeof(std::uint16_t));
        }
    }
    return out;
}

struct ImageResult {
    bool done = false;
    std::string format;
    bool pil_exact = true;
    bool has_alpha = false;
    Delta delta;
    bool decode_bitwise = false;
    std::size_t payload_ours_vs_merge = 0;      // payload from our decode vs patches_merge
    std::size_t payload_pil_vs_merge = 0;       // payload from Pillow's pixels vs patches_merge
    std::size_t payload_pil_vs_gather = 0;      // ... vs vLLM's gather_patches (raster), when captured
    bool gather_captured = false;
    bool positions_ok = false;
    std::size_t values = 0;
};


// vLLM's recorded refusal without the exception type ("ValueError: ") and VLLMValidationError's " (parameter=...)".
std::string served_text(std::string error) {
    const std::size_t colon = error.find(": ");
    if (colon != std::string::npos && error.find_first_of(" \n") > colon) { error = error.substr(colon + 2); }
    const std::size_t param = error.rfind(" (parameter=");
    if (param != std::string::npos && error.back() == ')') { error = error.substr(0, param); }
    return error;
}

struct PathTotals {
    int cases = 0, cases_ok = 0, ids_exact = 0, refusals_ok = 0, failures = 0;
    int decode_exact_expected = 0, decode_exact_got = 0, payload_bitwise = 0, gather_bitwise = 0, gather_total = 0;
    std::size_t images = 0;
};

PathTotals run_path(ink::DecodePath path, const fs::path& vision, const Json& meta, std::uint64_t limit,
                    const ink::TextEncoder& encode, const ink::SpecialTokens& special) {
    const bool served = path == ink::DecodePath::Served;
    const std::string tag = served ? "_served_" : "_";
    std::cout << "\n=== decode path: " << (served ? "SERVED (vLLM's server: Pillow, EXIF, white, no EOI fix)"
                                                   : "PROCESSOR (processing.py decode_image: no EXIF, checkerboard, EOI fix)")
              << " ===\n";
    const std::uint64_t cap = ink::kDefaultMaxImageTokens;
    const sinfer::media::decode::Policy policy{};
    std::map<std::string, ImageResult> images;
    PathTotals t;
    double decode_seconds = 0.0, payload_seconds = 0.0;
    std::ifstream lines(vision / "cases" / "cases.jsonl");
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty()) { continue; }
        const Json c             = Json::parse(line);
        const std::string id     = c.at("id").get<std::string>();
        const std::string body   = c.at("body").dump();
        std::string expect       = c.at("expect").get<std::string>();
        std::string exact;  // vLLM's refusal word for word, when recorded
        if (meta.contains("cases") && meta["cases"].contains(id) && meta["cases"][id].contains("serve")) {
            // A running vLLM's outcome. The processor path differs from it only in the pixels (and the JPEG EOI fix):
            // request-level refusals apply to both.
            const Json& sv = meta["cases"][id]["serve"];
            if (sv.contains("ok") && sv["ok"].is_boolean()) {
                if (sv["ok"].get<bool>()) {
                    if (served) { expect = "ok"; }
                } else if (sv.contains("error") && sv["error"].is_string()) {
                    const std::string text = served_text(sv["error"].get<std::string>());
                    if (served || !text.starts_with("Failed to load image")) {
                        exact  = text;
                        expect = "error:" + exact;
                    }
                }
            }
        }
        ++t.cases;
        const fs::path vec = vision / "processor" / (id + ".safetensors");
        const auto tensors = fs::exists(vec) ? load_safetensors(vec) : std::map<std::string, Tensor>{};
        const auto tensor_k = [&](const std::string& name, std::size_t k) -> const Tensor* {
            for (const std::string& key : {name + tag + std::to_string(k), name + "_" + std::to_string(k)}) {
                if (tensors.count(key) != 0U) { return &tensors.at(key); }
            }
            return nullptr;
        };
        std::string error;
        std::vector<int> ids;
        try {
            const ink::ChatRequest request = ink::parse_chat_request(body, limit);
            std::vector<std::uint64_t> tokens;
            const std::vector<std::string> names = c.at("images").get<std::vector<std::string>>();
            if (names.size() != request.image_urls.size()) {
                report(id + ": " + std::to_string(request.image_urls.size()) + " image parts parsed, case lists " +
                       std::to_string(names.size()));
                ++t.failures;
            }
            for (std::size_t k = 0; k < request.image_urls.size(); ++k) {
                const std::vector<std::uint8_t> bytes = data_url_bytes(request.image_urls[k]);
                // As the Frontend: the header's size and the cap before any pixel is decoded.
                const ink::ImageProbe probe = ink::probe_image(bytes, path);
                const ink::ImageGrid expected = ink::image_grid(probe.height, probe.width);
                if (expected.tokens() > cap) { throw ink::RequestError(ink::image_too_large_message(expected.tokens(), cap)); }
                const auto t0 = std::chrono::steady_clock::now();
                const ink::DecodedImage image = ink::decode_image(bytes, policy, path);
                decode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const ink::ImageGrid grid = ink::image_grid(image.height, image.width);
                if (grid.rows != expected.rows || grid.cols != expected.cols) {
                    report(id + " image " + std::to_string(k) + ": header grid " + std::to_string(expected.rows) + "x" +
                           std::to_string(expected.cols) + " vs decoded " + std::to_string(grid.rows) + "x" + std::to_string(grid.cols));
                    ++t.failures;
                }
                const Tensor* rgb = tensor_k("decoded_rgb", k);
                if (rgb != nullptr) {
                    const auto gv = tensor_k("grid", k)->as<std::int64_t>();
                    if (gv.size() != 3 || gv[0] != 1 || gv[1] != grid.rows || gv[2] != grid.cols) {
                        report(id + " image " + std::to_string(k) + ": grid " + std::to_string(grid.rows) + "x" +
                               std::to_string(grid.cols));
                        ++t.failures;
                    }
                    const std::string& name = names.at(k);
                    ImageResult& r = images[name + (served && tensors.count("decoded_rgb_served_" + std::to_string(k)) ? " (served)" : "")];
                    if (!r.done) {
                        r.done      = true;
                        r.format    = image.pixel_format;
                        r.pil_exact = image.pil_exact;
                        r.has_alpha = image.has_alpha;
                        r.delta     = rgb_delta(image, *rgb);
                        r.decode_bitwise = r.delta.differing == 0 && image.rgb.size() == r.delta.values;
                        const ink::DecodedImage pil = from_pil(*rgb);
                        const Tensor* raw = tensor_k("pixels_raw", k);
                        if (differing(ink::raw_pixels(pil), raw->as<std::uint16_t>()) != 0) {
                            report(name + ": raw pixels from Pillow's decode differ from vision_pixels");
                            ++t.failures;
                        }
                        if (r.decode_bitwise && differing(ink::raw_pixels(image), raw->as<std::uint16_t>()) != 0) {
                            report(name + ": raw pixels differ");
                            ++t.failures;
                        }
                        const auto t1 = std::chrono::steady_clock::now();
                        const std::vector<std::uint16_t> ours = payload(image);
                        payload_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
                        const std::vector<std::uint16_t> from_pillow = payload(pil);
                        r.values = from_pillow.size();
                        const bool own_pixels = served && tensors.count("decoded_rgb_served_" + std::to_string(k)) != 0U;
                        const fs::path pp = vision / "processor" / "patches" / (name + ".safetensors");
                        if (!own_pixels && fs::exists(pp)) {
                            // processing.py's pixels: the vectors' patches_merge and vLLM's gather_patches capture
                            const auto pt = load_safetensors(pp);
                            const auto merge = pt.at("patches_merge").as<std::uint16_t>();
                            r.payload_ours_vs_merge = differing(ours, merge);
                            r.payload_pil_vs_merge  = differing(from_pillow, merge);
                            if (pt.count("patches_vllm_raster") != 0U) {
                                r.gather_captured = true;
                                r.payload_pil_vs_gather =
                                    differing(merge_to_raster(from_pillow, grid), pt.at("patches_vllm_raster").as<std::uint16_t>());
                            }
                            const auto pos = pt.at("positions_merge").as<std::int32_t>();
                            r.positions_ok = pos.size() == merge.size() / ink::kPatchFeatures * 2;
                            for (std::size_t row = 0; r.positions_ok && row < pos.size() / 2; ++row) {
                                const std::size_t tt = row / 4;
                                const int y = static_cast<int>(tt / static_cast<std::size_t>(grid.cols)) * 2 + static_cast<int>(row % 4) / 2;
                                const int x = static_cast<int>(tt % static_cast<std::size_t>(grid.cols)) * 2 + static_cast<int>(row % 2);
                                r.positions_ok = pos[row * 2] == y && pos[row * 2 + 1] == x;
                            }
                        } else {
                            // the server's own pixels: the payload is the (proven) normalisation of them
                            r.payload_ours_vs_merge = differing(ours, from_pillow);
                            r.payload_pil_vs_merge  = 0;
                            r.positions_ok          = true;
                        }
                    }
                }
                tokens.push_back(grid.tokens());
            }
            ids = ink::render_chat(request, encode, special, tokens);
        } catch (const std::invalid_argument& e) {
            error = e.what();
        } catch (const std::exception& e) {
            error = std::string("EXCEPTION ") + e.what();
        }
        bool ok = true;
        if (expect.starts_with("error:")) {
            const std::string want = expect.substr(6);
            ok = !error.empty() && error.find(want) != std::string::npos && (exact.empty() || error == exact);
            if (ok) { ++t.refusals_ok; }
            else { report(id + ": expected refusal '" + want + "', got '" + (error.empty() ? "ok" : error) + "'"); }
            std::cout << "  refusal " << id << ": " << error.substr(0, 200) << (error.size() > 200 ? "..." : "")
                      << (exact.empty() ? "  [substring of the case's expectation]" : "  [== vLLM's server text]") << "\n";
        } else if (!error.empty()) {
            ok = false;
            report(id + ": refused: " + error);
        } else {
            const bool default_effort = !c.at("body").contains("chat_template_kwargs") ||
                                        !c.at("body").at("chat_template_kwargs").contains("reasoning_effort");
            const std::string key = (default_effort && tensors.count("prompt_token_ids_099") != 0U) ? "prompt_token_ids_099" : "prompt_token_ids";
            if (tensors.count(key) == 0U) {
                ok = false;
                report(id + ": no " + key + " in the vector");
            } else {
                const auto want = tensors.at(key).as<std::int32_t>();
                const bool same = want.size() == ids.size() && std::equal(want.begin(), want.end(), ids.begin());
                if (same) { ++t.ids_exact; }
                else {
                    ok = false;
                    std::size_t at = 0;
                    while (at < std::min(want.size(), ids.size()) && want[at] == ids[at]) { ++at; }
                    report(id + ": prompt ids differ (" + std::to_string(ids.size()) + " vs " + std::to_string(want.size()) +
                           " ids, first difference at " + std::to_string(at) + ")");
                }
            }
        }
        if (ok) { ++t.cases_ok; } else { ++t.failures; }
    }

    std::cout << "cases " << t.cases << ": " << t.cases_ok << " ok (" << t.ids_exact << " prompt-id vectors exact, "
              << t.refusals_ok << " refusals as vLLM)\n";
    std::cout << "per image (" << images.size() << "): decode vs Pillow | payload [4N,1200] bf16 vs gather_patches\n";
    std::printf("  %-36s %-13s %5s %5s %9s %9s %7s %8s %10s %10s %10s %4s\n", "image", "format", "exact", "alpha",
                "values", "differ", "max|d|", "mean|d|", "ours-ref", "pil-merge", "pil-gather", "pos");
    t.images = images.size();
    for (const auto& [name, r] : images) {
        const double mean = r.delta.differing ? r.delta.sum_abs / static_cast<double>(r.delta.values) : 0.0;
        std::printf("  %-36s %-13s %5s %5s %9zu %9zu %7d %8.4f %10zu %10zu %10s %4s\n", name.c_str(), r.format.c_str(),
                    r.pil_exact ? "yes" : "no", r.has_alpha ? "yes" : "no", r.delta.values, r.delta.differing, r.delta.max_abs,
                    mean, r.payload_ours_vs_merge, r.payload_pil_vs_merge,
                    r.gather_captured ? std::to_string(r.payload_pil_vs_gather).c_str() : "-", r.positions_ok ? "ok" : "BAD");
        if (r.delta.differing != 0) {
            std::printf("      |d| histogram (values): 1: %zu  2: %zu  3: %zu  4-7: %zu  >=8: %zu; pixels differing %zu of %zu\n",
                        r.delta.hist[0], r.delta.hist[1], r.delta.hist[2], r.delta.hist[3], r.delta.hist[4],
                        r.delta.pixels_differing, r.delta.values / 3);
        }
        if (r.pil_exact) {
            ++t.decode_exact_expected;
            if (r.decode_bitwise && r.payload_ours_vs_merge == 0) { ++t.decode_exact_got; }
            else {
                report(name + ": a Pillow-exact decode differs (" + std::to_string(r.delta.differing) + " values)");
                ++t.failures;
            }
        }
        if (r.payload_pil_vs_merge == 0) { ++t.payload_bitwise; }
        else {
            report(name + ": payload from Pillow's pixels differs from patches_merge");
            ++t.failures;
        }
        if (r.gather_captured) {
            ++t.gather_total;
            if (r.payload_pil_vs_gather == 0) { ++t.gather_bitwise; }
            else {
                report(name + ": payload differs from vLLM's gather_patches");
                ++t.failures;
            }
        }
        if (!r.positions_ok) {
            report(name + ": positions_merge");
            ++t.failures;
        }
    }
    std::cout << "Pillow-exact decodes bit for bit, payload too: " << t.decode_exact_got << " / " << t.decode_exact_expected
              << "\npayload from Pillow's pixels == patches_merge: " << t.payload_bitwise << " / " << images.size()
              << "\npayload == vLLM gather_patches (card capture): " << t.gather_bitwise << " / " << t.gather_total
              << "\ntimes: decode " << decode_seconds << " s, payload " << payload_seconds << " s\n"
              << (t.failures == 0 ? "path PASS" : "path FAIL") << " (" << t.failures << " failures)\n";
    return t;
}

} // namespace

int main() {
    const char* model_dir   = std::getenv("INKLING_MODEL_DIR");
    const char* vectors_dir = std::getenv("INKLING_VECTORS");
    if (model_dir == nullptr || vectors_dir == nullptr) {
        std::cout << "SKIP: set INKLING_MODEL_DIR and INKLING_VECTORS\n";
        return 77;
    }
    const std::uint64_t limit = std::getenv("INKLING_IMAGE_LIMIT") ? std::stoull(std::getenv("INKLING_IMAGE_LIMIT")) : 8;
    const fs::path model(model_dir);
    const fs::path vision = fs::path(vectors_dir) / "vision";
    const fi::Tokenizer tokenizer(fi::TokenizerResources{read_file(model / "tokenizer.json"),
                                                         read_file(model / "tokenizer_config.json"),
                                                         read_file(model / "generation_config.json"), {}, false});
    const ink::TextEncoder encode = [&](std::string_view text) {
        return tokenizer.encode(text, fi::EncodeOptions{.parse_added_tokens = false, .add_bos = false});
    };
    const ink::SpecialTokens special =
        ink::SpecialTokens::resolve([&](std::string_view s) { return tokenizer.token_id(s); });
    std::cout << "image pad <|unused_200054|> = " << special.image_pad << ", <|content_image|> = " << special.content_image
              << "\n";
    int failures = 0;
    if (special.image_pad != 200054 || special.content_image != 200005) {
        std::cout << "FAIL: vLLM's ids are IMAGE_MARKER_ID 200005 / IMAGE_PAD_ID 200054\n";
        ++failures;
    }
    // vision off: the text-only behaviour, vLLM at image limit 0
    try {
        (void)ink::parse_chat_request(R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]})", 0);
        ++failures;
        std::cout << "FAIL: image limit 0 accepted an image\n";
    } catch (const std::invalid_argument& e) {
        if (std::string(e.what()) != "At most 0 image(s) may be provided in one prompt.") {
            ++failures;
            std::cout << "FAIL: image limit 0 refused with '" << e.what() << "'\n";
        }
    }
    // pydantic's input_value cut never splits a UTF-8 character (review fix): the last 24 bytes start inside an 'é'.
    try {
        (void)ink::parse_chat_request(R"({"messages":[{"role":"user","content":[{"type":"input_image","image_url":"data:,\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9\u00e9x"}]}]})", 8);
        ++failures;
        std::cout << "FAIL: input_image without detail accepted\n";
    } catch (const std::invalid_argument& e) {
        const std::string text = e.what();
        bool valid = true;
        for (std::size_t i = 0; i < text.size();) {  // a strict UTF-8 walk
            const auto c = static_cast<unsigned char>(text[i]);
            const std::size_t n = c < 0x80 ? 1 : (c >> 5U) == 6 ? 2 : (c >> 4U) == 14 ? 3 : (c >> 3U) == 30 ? 4 : 0;
            if (n == 0 || i + n > text.size()) { valid = false; break; }
            for (std::size_t j = 1; j < n; ++j) { valid = valid && (static_cast<unsigned char>(text[i + j]) & 0xC0U) == 0x80U; }
            i += n;
        }
        std::cout << "pydantic input_value cut on a non-ASCII URL: " << (valid ? "valid UTF-8" : "INVALID UTF-8") << "\n";
        if (!valid) { ++failures; }
    }
    Json meta = Json::object();
    if (fs::exists(vision / "processor" / "meta.json")) { meta = Json::parse(read_file(vision / "processor" / "meta.json")); }
    // Orientation / alpha vectors made by vLLM's own loader (ImageMediaIO.load_bytes) in the reference image:
    // a vector generator (not in this repository) -- EXIF 1..8 in JPEG, XMP-only orientation, PNG eXIf, WebP EXIF / XMP, RGBA + EXIF.
    if (const char* orient = std::getenv("INKLING_ORIENT_DIR"); orient != nullptr) {
        const fs::path dir(orient);
        const Json index = Json::parse(read_file(dir / "index.json"));
        int exact = 0, total = 0;
        for (auto it = index.begin(); it != index.end(); ++it) {
            ++total;
            const std::string file = it.value().at("file").get<std::string>();
            const std::string raw  = read_file(dir / file);
            const std::vector<std::uint8_t> bytes(raw.begin(), raw.end());
            const std::string want = read_file(dir / (it.key() + ".rgb"));
            try {
                const ink::ImageProbe probe = ink::probe_image(bytes, ink::DecodePath::Served);
                const ink::DecodedImage image = ink::decode_image(bytes, sinfer::media::decode::Policy{}, ink::DecodePath::Served);
                const bool same = image.width == it.value().at("w").get<int>() && image.height == it.value().at("h").get<int>() &&
                                  probe.width == image.width && probe.height == image.height &&
                                  want.size() == image.rgb.size() && std::memcmp(want.data(), image.rgb.data(), want.size()) == 0;
                if (same) { ++exact; }
                else {
                    report("orientation vector " + it.key() + ": " + std::to_string(image.width) + "x" +
                           std::to_string(image.height) + " (probe " + std::to_string(probe.width) + "x" +
                           std::to_string(probe.height) + ")");
                    ++failures;
                }
            } catch (const std::exception& e) {
                report("orientation vector " + it.key() + ": " + e.what());
                ++failures;
            }
        }
        std::cout << "orientation / alpha vectors (vLLM's ImageMediaIO): " << exact << " / " << total << " bit for bit\n";
    }
    const PathTotals served = run_path(ink::DecodePath::Served, vision, meta, limit, encode, special);
    const PathTotals proc   = run_path(ink::DecodePath::Processor, vision, meta, limit, encode, special);
    failures += served.failures + proc.failures;
    std::cout << "\nSUMMARY served: " << served.cases_ok << "/" << served.cases << " cases (" << served.ids_exact << " ids exact, "
              << served.refusals_ok << " refusals), " << served.decode_exact_got << "/" << served.decode_exact_expected
              << " images bit for bit | processor: " << proc.cases_ok << "/" << proc.cases << " cases (" << proc.ids_exact
              << " ids exact, " << proc.refusals_ok << " refusals), " << proc.decode_exact_got << "/" << proc.decode_exact_expected
              << " images bit for bit, payload == gather_patches " << proc.gather_bitwise << "/" << proc.gather_total << "\n";
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " (" << failures << " failures)\n";
    return failures == 0 ? 0 : 1;
}
