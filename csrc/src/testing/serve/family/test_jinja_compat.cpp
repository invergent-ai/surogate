#include <minja/minja.hpp>

#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using Json = nlohmann::ordered_json;

std::string render(const std::string& source, const Json& variables) {
    return minja::Parser::parse(source, {.trim_blocks = true, .lstrip_blocks = true,
                                       .keep_trailing_newline = false})
        ->render(minja::Context::make(variables));
}

void expect(const std::string& source, const Json& variables, const std::string& wanted) {
    const auto actual = render(source, variables);
    if (actual != wanted) {
        throw std::runtime_error("Jinja mismatch: " + source + "\nexpected: " + wanted +
                                 "\nactual: " + actual);
    }
}

void check_string_indices() {
    const Json vars{{"s", "aé中🙂z"}};
    for (const auto& [index, character] : std::vector<std::pair<int, std::string>>{
             {0, "a"}, {1, "é"}, {2, "中"}, {3, "🙂"}, {4, "z"},
             {-1, "z"}, {-2, "🙂"}, {-3, "中"}, {-4, "é"}, {-5, "a"}}) {
        expect("{{ s[" + std::to_string(index) + "] }}", vars, character);
    }
    expect("{{ s.2 }}", vars, "中");
    for (const auto index : {std::int64_t{5}, std::int64_t{-6},
                            std::numeric_limits<std::int64_t>::max(),
                            std::numeric_limits<std::int64_t>::min()}) {
        expect("{% if s[i] is undefined %}missing{% endif %}",
               {{"s", vars["s"]}, {"i", index}}, "missing");
    }
    expect("{% if s[0] is undefined and s[-1] is undefined %}empty{% endif %}",
           {{"s", ""}}, "empty");
    expect("{% if s['0'] is undefined %}missing{% endif %}", vars, "missing");
    expect("{{ a[0].output }}:{{ a[-1].output }}:{{ d['0'] }}",
           {{"a", Json::array({Json{{"output", "first"}}, Json{{"output", "last"}}})},
            {"d", {{"0", "key"}}}}, "first:last:key");
    // Fixing a string lookup must not turn a genuinely missing intermediate into a value.
    bool refused = false;
    try { (void)render("{{ absent.output }}", Json::object()); }
    catch (const std::runtime_error&) { refused = true; }
    if (!refused) { throw std::runtime_error("missing base lookup was silently accepted"); }
}

void check_glm_tool_results() {
    // The guard from GLM-5.3-Flash's checkpoint, followed by its text/list distinction.
    const std::string source = R"({%- macro is_list_of_outputs(m) -%}
    {%- if m.content and m.content[0].output is defined -%}1{%- endif -%}
{%- endmacro -%}
{%- for m in messages -%}
    {%- if is_list_of_outputs(m) -%}
        {%- for entry in m.content -%}<tool_response>{{ entry.output }}</tool_response>{%- endfor -%}
    {%- elif m.content is string -%}<tool_response>{{ m.content }}</tool_response>
    {%- else -%}{%- for entry in m.content -%}<tool_response>{{ entry.text }}</tool_response>{%- endfor -%}
    {%- endif -%}
{%- endfor -%})";
    for (const std::string content : {"sunny", "{\"result\":42}", "", "晴天🙂"}) {
        expect(source, {{"messages", Json::array({Json{{"role", "tool"}, {"content", content}}})}},
               "<tool_response>" + content + "</tool_response>");
    }
    expect(source, {{"messages", Json::array({
        Json{{"content", Json::array({Json{{"id", "call_1"}, {"output", "first"}},
                                      Json{{"id", "call_2"}, {"output", "second"}}})}},
        Json{{"content", Json::array({Json{{"type", "text"}, {"text", "third"}}})}},
        Json{{"content", Json::array()}}})}},
        "<tool_response>first</tool_response><tool_response>second</tool_response>"
        "<tool_response>third</tool_response>");
}
} // namespace

int main(int argc, char** argv) {
    try {
        check_string_indices();
        check_glm_tool_results();
        // Optional full-checkpoint cases, with expected prompts supplied by Jinja2.
        // No weights, CUDA context, or checkpoint copy is needed to compare renderers.
        if (argc == 3) {
            std::ifstream source_file(argv[1]), cases_file(argv[2]);
            if (!source_file || !cases_file) { throw std::runtime_error("cannot open Jinja fixtures"); }
            const std::string source{std::istreambuf_iterator<char>(source_file), {}};
            const auto cases = Json::parse(cases_file);
            for (const auto& example : cases) {
                expect(source, example.at("variables"), example.at("expected").get<std::string>());
            }
            std::cout << "Full checkpoint Jinja2 comparisons passed: " << cases.size() << '\n';
        }
        std::cout << "Jinja string indexing and GLM tool-result rendering passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
