#pragma once

#include "adi/json.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace adi {

struct FunctionTool {
    std::string name;
    std::string description;
    Json parameters;
    Json response_definition;
    Json prompt_definition;
};

struct FunctionCall {
    std::string name;
    Json arguments;
    std::string arguments_json;
};

struct ParsedModelOutput {
    std::string text;
    std::vector<FunctionCall> function_calls;
};

[[nodiscard]] std::vector<FunctionTool> parse_function_tools(
    const Json &tools);

[[nodiscard]] ParsedModelOutput parse_tool_calls(
    std::string_view output,
    std::span<const FunctionTool> tools,
    bool allow_parallel_calls);

} // namespace adi
