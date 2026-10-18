#pragma once

// Tool-call argument values exactly as vLLM 0.31's Qwen3 XML parser returns them.
//
// `--tool-call-parser qwen3_coder` and `qwen3_xml` are one class in vLLM 0.31 (vllm/parser/qwen3.py on
// vllm/parser/engine/parser_engine.py, conversion in vllm/tool_parsers/utils.py). It reads every
// `<parameter=NAME>VALUE</parameter>` of a call as text, then converts the value of each parameter the tool's
// schema declares. The types come from the property's `type`, the kinds of its `enum` members and its
// `anyOf`/`oneOf`/`allOf` branches -- not from `$ref` or `const` -- and no type information means string. They are
// tried in the order null, integer, number, boolean, object, array, string with Python's own rules: `int()` and
// `float()` (surrounding whitespace, `+`, leading zeros, `_` between digits; a number with no fraction becomes an
// integer, so 5.0 is 5), "true"/"1"/"false"/"0" in any case for a boolean, "null" in any case for null, `json.loads`
// for an object or an array. A value no type accepts goes through `json.loads` and stays text when that fails. A
// parameter the schema does not declare, and every parameter of a tool without properties, stays text. Nested values
// are not converted further.
//
// The engine typed values by parsing them as JSON, which left `True` a string where vLLM returns a boolean, and
// turned an undeclared `5` into a number where vLLM keeps "5". Clients act on the typed value and send it back in the
// next request's history, which the template renders to the model as its own earlier call (BFCL multi-turn on the same
// weights, 2026-10-18: engine 69.4, vLLM 72.2; re-typing the engine's arguments as vLLM does gained 4.7).
//
// The arguments are serialised as Python's `json.dumps(arguments, ensure_ascii=False)`: ", " and ": " separators,
// floats in Python's repr, integers exact at any size, non-ASCII text unescaped.

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sinfer::serve::vllm_args {

/// Parameter name and raw value, in the order vLLM's converter keeps them.
using Parameters = std::vector<std::pair<std::string, std::string>>;

/// The `<parameter=NAME>VALUE</parameter>` pairs of a function body (the text between `<function=NAME>` and
/// `</function>`) as vLLM's `_qwen3_arg_converter` finds them: tags may carry whitespace inside (`< parameter = x >`,
/// `</ parameter >`), a value ends at its closing tag or at the next opening tag, an unterminated last value is
/// dropped, one wrapping newline is trimmed from each side of a value, text between parameters is ignored, and a
/// repeated name keeps its first position with its last value.
[[nodiscard]] Parameters extract_qwen3_parameters(std::string_view body);

/// The arguments JSON vLLM returns for `parameters` of a tool whose `properties` (the schema's `properties` object,
/// or null) declare the parameter types.
[[nodiscard]] std::string qwen3_arguments_json(const Parameters& parameters, const nlohmann::ordered_json& properties);

/// `json.dumps(json.loads(text), ensure_ascii=False)`; nullopt when Python's `json.loads` refuses `text` or the
/// value holds a non-finite number.
[[nodiscard]] std::optional<std::string> python_json_normalize(std::string_view text);

/// `json.dumps(value, ensure_ascii=False)` for a value held by nlohmann.
[[nodiscard]] std::string python_json_dumps(const nlohmann::ordered_json& value);

/// Python's `str.strip()`: leading and trailing Unicode whitespace removed.
[[nodiscard]] std::string python_strip(std::string_view text);

/// Python's `int(text)`, as its decimal digits with a normalised sign; nullopt when `int()` raises.
[[nodiscard]] std::optional<std::string> python_int(std::string_view text);

/// Python's `float(text)`, infinities and NaN included; nullopt when `float()` raises.
[[nodiscard]] std::optional<double> python_float(std::string_view text);

/// Python's `repr()` of a finite float.
[[nodiscard]] std::string python_float_repr(double value);

/// Python's `int(value)` of a finite float with no fractional part, exact at any magnitude.
[[nodiscard]] std::string python_int_of_float(double value);

} // namespace sinfer::serve::vllm_args
