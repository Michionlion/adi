#include "tool_call.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>

namespace adi {
namespace {

constexpr std::string_view tool_open = "<tool_call>";
constexpr std::string_view tool_close = "</tool_call>";
constexpr std::string_view function_open = "<function=";
constexpr std::string_view function_close = "</function>";
constexpr std::string_view parameter_open = "<parameter=";
constexpr std::string_view parameter_close = "</parameter>";

bool valid_name(std::string_view name) {
    return !name.empty() && name.size() <= 64 &&
           std::all_of(name.begin(), name.end(), [](unsigned char character) {
               return (character >= 'a' && character <= 'z') ||
                      (character >= 'A' && character <= 'Z') ||
                      (character >= '0' && character <= '9') ||
                      character == '_' || character == '-';
           });
}

void skip_whitespace(std::string_view text, std::size_t &position) {
    while (position < text.size() &&
           (text[position] == ' ' || text[position] == '\t' ||
            text[position] == '\r' || text[position] == '\n')) {
        ++position;
    }
}

std::string parse_tag_name(
    std::string_view text,
    std::size_t &position,
    std::string_view opening) {
    if (!text.substr(position).starts_with(opening)) {
        throw std::runtime_error("malformed model tool call");
    }
    position += opening.size();
    const auto end = text.find('>', position);
    if (end == std::string_view::npos) {
        throw std::runtime_error("incomplete model tool call");
    }
    const auto name = text.substr(position, end - position);
    if (!valid_name(name)) {
        throw std::runtime_error("model tool call has an invalid name");
    }
    position = end + 1;
    return std::string(name);
}

Json empty_parameters() {
    return Json(Json::Object{
        {"type", Json(std::string("object"))},
        {"properties", Json(Json::Object{})},
    });
}

void validate_parameters(const Json &parameters) {
    if (parameters.object() == nullptr) {
        throw std::runtime_error("function tool 'parameters' must be an object");
    }
    if (const auto *type = parameters.find("type");
        type != nullptr &&
        (type->string() == nullptr || *type->string() != "object")) {
        throw std::runtime_error(
            "function tool parameters must describe an object");
    }
    if (const auto *properties = parameters.find("properties");
        properties != nullptr && properties->object() == nullptr) {
        throw std::runtime_error(
            "function tool parameter 'properties' must be an object");
    }
    if (const auto *required = parameters.find("required"); required != nullptr) {
        if (required->array() == nullptr) {
            throw std::runtime_error(
                "function tool parameter 'required' must be an array");
        }
        for (const auto &name : *required->array()) {
            if (name.string() == nullptr || !valid_name(*name.string())) {
                throw std::runtime_error(
                    "function tool has an invalid required parameter name");
            }
        }
    }
}

FunctionTool parse_function_tool(const Json &value) {
    if (value.object() == nullptr) {
        throw std::runtime_error("each tool must be an object");
    }
    const auto *type = value.find("type");
    if (type == nullptr || type->string() == nullptr ||
        *type->string() != "function") {
        throw std::runtime_error("only function tools are supported");
    }
    const auto *name = value.find("name");
    if (name == nullptr || name->string() == nullptr ||
        !valid_name(*name->string())) {
        throw std::runtime_error(
            "function tool 'name' must contain 1-64 letters, digits, '_' or '-'");
    }

    std::string description;
    if (const auto *field = value.find("description"); field != nullptr) {
        if (field->string() == nullptr) {
            throw std::runtime_error(
                "function tool 'description' must be a string");
        }
        description = *field->string();
    }
    bool strict = false;
    if (const auto *field = value.find("strict"); field != nullptr) {
        if (field->boolean() == nullptr) {
            throw std::runtime_error("function tool 'strict' must be a boolean");
        }
        strict = *field->boolean();
    }
    if (strict) {
        throw std::runtime_error(
            "strict function tools are not supported; set 'strict' to false");
    }

    Json parameters = empty_parameters();
    if (const auto *field = value.find("parameters"); field != nullptr) {
        parameters = *field;
    }
    validate_parameters(parameters);

    Json::Object response_function{
        {"type", Json(std::string("function"))},
        {"name", Json(*name->string())},
    };
    Json::Object prompt_function{
        {"name", Json(*name->string())},
    };
    if (!description.empty()) {
        response_function.emplace_back("description", Json(description));
        prompt_function.emplace_back("description", Json(description));
    }
    response_function.emplace_back("parameters", parameters);
    response_function.emplace_back("strict", Json(false));
    prompt_function.emplace_back("parameters", parameters);

    FunctionTool result;
    result.name = *name->string();
    result.description = std::move(description);
    result.parameters = parameters;
    result.response_definition = Json(std::move(response_function));
    result.prompt_definition = Json(Json::Object{
        {"type", Json(std::string("function"))},
        {"function", Json(std::move(prompt_function))},
    });
    return result;
}

const FunctionTool &find_tool(
    std::span<const FunctionTool> tools,
    std::string_view name) {
    const auto match = std::find_if(
        tools.begin(), tools.end(), [name](const FunctionTool &tool) {
            return tool.name == name;
        });
    if (match == tools.end()) {
        throw std::runtime_error(
            "model called unknown function '" + std::string(name) + "'");
    }
    return *match;
}

const Json *property_schema(const FunctionTool &tool, std::string_view name) {
    const auto *properties = tool.parameters.find("properties");
    return properties == nullptr ? nullptr : properties->find(name);
}

Json typed_argument(
    const FunctionTool &tool,
    std::string_view name,
    std::string value) {
    const auto *schema = property_schema(tool, name);
    if (schema == nullptr) {
        if (const auto *additional = tool.parameters.find("additionalProperties");
            additional != nullptr && additional->boolean() != nullptr &&
            !*additional->boolean()) {
            throw std::runtime_error(
                "model supplied unknown parameter '" + std::string(name) +
                "' to function '" + tool.name + "'");
        }
        return Json(std::move(value));
    }
    if (schema->object() == nullptr) {
        throw std::runtime_error(
            "function parameter schemas must be objects");
    }
    const auto *type = schema->find("type");
    if (type == nullptr) {
        try {
            return parse_json(value);
        } catch (const std::runtime_error &) {
            return Json(std::move(value));
        }
    }
    if (type->string() == nullptr) {
        throw std::runtime_error(
            "function parameter schema 'type' must be a string");
    }
    if (*type->string() == "string") {
        return Json(std::move(value));
    }

    Json parsed;
    try {
        parsed = parse_json(value);
    } catch (const std::runtime_error &) {
        throw std::runtime_error(
            "model supplied invalid " + *type->string() + " parameter '" +
            std::string(name) + "' to function '" + tool.name + "'");
    }
    if (*type->string() == "number" && parsed.number() != nullptr) {
        return parsed;
    }
    if (*type->string() == "integer" && parsed.number() != nullptr &&
        std::floor(*parsed.number()) == *parsed.number()) {
        return parsed;
    }
    if (*type->string() == "boolean" && parsed.boolean() != nullptr) {
        return parsed;
    }
    if (*type->string() == "array" && parsed.array() != nullptr) {
        return parsed;
    }
    if (*type->string() == "object" && parsed.object() != nullptr) {
        return parsed;
    }
    if (*type->string() == "null" && parsed.is_null()) {
        return parsed;
    }
    throw std::runtime_error(
        "model supplied invalid " + *type->string() + " parameter '" +
        std::string(name) + "' to function '" + tool.name + "'");
}

void validate_required(
    const FunctionTool &tool,
    const Json::Object &arguments) {
    const auto *required = tool.parameters.find("required");
    if (required == nullptr) {
        return;
    }
    for (const auto &required_name : *required->array()) {
        const auto present = std::any_of(
            arguments.begin(), arguments.end(), [&](const auto &argument) {
                return argument.first == *required_name.string();
            });
        if (!present) {
            throw std::runtime_error(
                "model omitted required parameter '" + *required_name.string() +
                "' for function '" + tool.name + "'");
        }
    }
}

FunctionCall parse_call(
    std::string_view output,
    std::size_t &position,
    std::span<const FunctionTool> tools) {
    position += tool_open.size();
    skip_whitespace(output, position);
    auto function_name = parse_tag_name(output, position, function_open);
    const auto &tool = find_tool(tools, function_name);
    skip_whitespace(output, position);

    Json::Object arguments;
    std::unordered_set<std::string> argument_names;
    while (output.substr(position).starts_with(parameter_open)) {
        auto parameter_name = parse_tag_name(output, position, parameter_open);
        if (!argument_names.emplace(parameter_name).second) {
            throw std::runtime_error(
                "model repeated parameter '" + parameter_name + "'");
        }
        const auto end = output.find(parameter_close, position);
        if (end == std::string_view::npos) {
            throw std::runtime_error("incomplete model tool parameter");
        }
        std::string parameter_value(output.substr(position, end - position));
        if (!parameter_value.empty() && parameter_value.front() == '\n') {
            parameter_value.erase(0, 1);
        }
        if (!parameter_value.empty() && parameter_value.back() == '\n') {
            parameter_value.pop_back();
        }
        arguments.emplace_back(
            parameter_name,
            typed_argument(tool, parameter_name, std::move(parameter_value)));
        position = end + parameter_close.size();
        skip_whitespace(output, position);
    }
    if (!output.substr(position).starts_with(function_close)) {
        throw std::runtime_error("malformed model function call");
    }
    position += function_close.size();
    skip_whitespace(output, position);
    if (!output.substr(position).starts_with(tool_close)) {
        throw std::runtime_error("malformed model tool call wrapper");
    }
    position += tool_close.size();

    validate_required(tool, arguments);
    FunctionCall result;
    result.name = std::move(function_name);
    result.arguments = Json(std::move(arguments));
    result.arguments_json = json_dump(result.arguments);
    return result;
}

} // namespace

std::vector<FunctionTool> parse_function_tools(const Json &tools) {
    const auto *items = tools.array();
    if (items == nullptr) {
        throw std::runtime_error("'tools' must be an array");
    }
    if (items->size() > 128) {
        throw std::runtime_error("at most 128 function tools are supported");
    }
    std::vector<FunctionTool> result;
    result.reserve(items->size());
    std::unordered_set<std::string> names;
    for (const auto &item : *items) {
        auto tool = parse_function_tool(item);
        if (!names.emplace(tool.name).second) {
            throw std::runtime_error(
                "duplicate function tool name '" + tool.name + "'");
        }
        result.push_back(std::move(tool));
    }
    return result;
}

ParsedModelOutput parse_tool_calls(
    std::string_view output,
    std::span<const FunctionTool> tools,
    bool allow_parallel_calls) {
    ParsedModelOutput result;
    const auto first_call = output.find(tool_open);
    if (first_call == std::string_view::npos) {
        result.text = std::string(output);
        return result;
    }

    result.text = std::string(output.substr(0, first_call));
    if (!result.text.empty() &&
        result.text.find("</think>") == std::string::npos) {
        result.text += "\n</think>";
    }

    std::size_t position = first_call;
    for (;;) {
        result.function_calls.push_back(parse_call(output, position, tools));
        skip_whitespace(output, position);
        if (position == output.size()) {
            break;
        }
        if (!output.substr(position).starts_with(tool_open)) {
            throw std::runtime_error(
                "model added content after a function call");
        }
    }
    if (!allow_parallel_calls && result.function_calls.size() > 1) {
        throw std::runtime_error(
            "model produced parallel calls while parallel_tool_calls is false");
    }
    return result;
}

} // namespace adi
