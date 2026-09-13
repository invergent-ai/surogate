#include "encoder/embedding_request.h"
#include "encoder/options.h"

#include <cmath>
#include <iostream>
#include <limits>

using namespace sinfer::encoder;
using Json = nlohmann::json;

int main() {
    int failures = 0;
    const auto check = [&](bool ok, const char* label) {
        if (!ok) { std::cerr << "FAIL " << label << '\n'; ++failures; }
    };
    const auto parse = [](const Json& body) {
        return parse_embedding_request(body, "encoder", 16, 4, 3,
            [](const std::string& text) { return std::vector<std::int32_t>(text.size(), 1); });
    };
    const auto rejected = [&](const Json& body, const char* param, int status = 400) {
        try { (void)parse(body); check(false, "invalid request accepted"); }
        catch (const EmbeddingRequestError& error) {
            check(error.param == param && error.status == status, "attributed request rejection");
        }
    };
    for (const Json input : {Json("a"), Json::array({"a", "bb"}), Json::array({0, 15}),
                             Json::array({Json::array({0}), Json::array({1, 15})})}) {
        const auto request = parse({{"input", input}, {"model", "encoder"}});
        check(!request.sequences.empty() && request.dimensions == 3 && !request.base64, "input forms");
    }
    for (const char* body : {
             R"({"input": [-1]})", R"({"input": [16]})", R"({"input": [2147483647]})",
             R"({"input": [4294967296]})", R"({"input": [18446744073709551615]})",
             R"({"input": [1.5]})", R"({"input": [true]})", R"({"input": [0,false]})",
             R"({"input": [[1],[]]})", R"({"input": []})", R"({"input": ""})",
             R"({"input": [0,1,2,3,4]})", R"({"input": "abcde"})",
             R"({"input": ["a",[1]]})", R"({"input": [[1],"a"]})", R"({})"}) {
        rejected(Json::parse(body), "input");
    }
    rejected(Json::array(), "body");
    rejected({{"input", "a"}, {"model", false}}, "model");
    rejected({{"input", "a"}, {"model", "unknown"}}, "model", 404);
    for (Json count : {Json(0), Json(-1), Json(4), Json(1.5), Json(true), Json("2"),
                       Json(std::numeric_limits<std::uint64_t>::max())}) {
        rejected({{"input", "a"}, {"dimensions", count}}, "dimensions");
    }
    for (Json format : {Json(1), Json(true), Json("unsupported")}) {
        rejected({{"input", "a"}, {"encoding_format", format}}, "encoding_format");
    }
    const auto reduced = parse({{"input", "a"}, {"dimensions", 2}, {"encoding_format", "base64"}});
    check(reduced.dimensions == 2 && reduced.base64, "requested representation");
    const auto defaults = parse({{"input", "a"}, {"dimensions", nullptr}, {"encoding_format", nullptr}});
    check(defaults.dimensions == 3 && !defaults.base64, "null defaults");
    check(parse({{"input", "a"}, {"model", nullptr}}).dimensions == 3, "null model uses running encoder");
    for (const std::string name : {"google/embeddinggemma-300m", "./models/my model.sinfer"}) {
        std::vector<std::string> arguments{"embed", name};
        std::vector<char*> argv;
        for (auto& argument : arguments) { argv.push_back(argument.data()); }
        check(parse_options(argv.size(), argv.data()).served_model_name == name,
              "native model ID preserves supplied argument");
    }

    const auto full = encode_embedding({3.0F, 4.0F, 12.0F}, 3, false);
    check(full == Json::array({3.0F, 4.0F, 12.0F}), "full-size output unchanged");
    const auto small = encode_embedding({3.0F, 4.0F, 12.0F}, 2, false);
    check(std::abs(small[0].get<float>() - 0.6F) < 1e-7F &&
          std::abs(small[1].get<float>() - 0.8F) < 1e-7F, "truncate and normalize");
    check(encode_embedding({1.0F, -2.0F, 0.5F}, 3, true) == "AACAPwAAAMAAAAA/", "little-endian float32 base64");
    check(encode_embedding({0.0F, 0.0F, 1.0F}, 2, false) == Json::array({0.0F, 0.0F}), "zero prefix");
    for (const auto ids : {std::vector<std::int32_t>{-1}, {16}, {2147483647}, {}, {0,1,2,3,4}}) {
        try { validate_embedding_input(ids, 16, 4); check(false, "native invalid input accepted"); }
        catch (const std::invalid_argument&) {}
    }
    std::cout << (failures ? "FAIL" : "OK") << " embedding_request\n";
    return failures ? 1 : 0;
}
