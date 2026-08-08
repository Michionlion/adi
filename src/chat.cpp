#include "chat.hpp"

#include "unicode.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace adi {
namespace {

constexpr std::size_t supported_template_size = 7764;
constexpr std::uint64_t supported_template_fnv1a = 0x8AE3A8BC6E68F114ULL;

std::string trim(std::string_view text) {
    const auto codepoints = utf8_codepoints(text);
    const auto whitespace = [](std::uint32_t codepoint) {
        // Python str.strip(), which implements Jinja's trim filter, also
        // recognizes these four information separators.
        return unicode_properties(codepoint).whitespace ||
               (codepoint >= 0x1CU && codepoint <= 0x1FU);
    };
    std::size_t begin = 0;
    while (begin < codepoints.size() && whitespace(codepoints[begin])) {
        ++begin;
    }
    std::size_t end = codepoints.size();
    while (end > begin && whitespace(codepoints[end - 1])) {
        --end;
    }
    return codepoints_utf8(std::vector<std::uint32_t>(
        codepoints.begin() + static_cast<std::ptrdiff_t>(begin),
        codepoints.begin() + static_cast<std::ptrdiff_t>(end)));
}

void strip_assistant_reasoning(
    std::string &content,
    std::string &reasoning) {
    const auto first_end = content.find("</think>");
    if (first_end == std::string::npos) {
        return;
    }

    auto reasoning_source = content.substr(0, first_end);
    while (!reasoning_source.empty() && reasoning_source.back() == '\n') {
        reasoning_source.pop_back();
    }
    if (const auto last_start = reasoning_source.rfind("<think>");
        last_start != std::string::npos) {
        reasoning_source.erase(0, last_start + std::string_view("<think>").size());
    }
    while (!reasoning_source.empty() && reasoning_source.front() == '\n') {
        reasoning_source.erase(0, 1);
    }
    reasoning = trim(reasoning_source);

    content.erase(0, content.rfind("</think>") +
                         std::string_view("</think>").size());
    while (!content.empty() && content.front() == '\n') {
        content.erase(0, 1);
    }
}

void append_tool_call(std::string &prompt, const ChatToolCall &call) {
    if (call.arguments.object() == nullptr) {
        throw std::runtime_error("function call arguments must be an object");
    }
    prompt += "<tool_call>\n<function=" + call.name + ">\n";
    for (const auto &[name, value] : *call.arguments.object()) {
        prompt += "<parameter=" + name + ">\n";
        if (const auto *string = value.string()) {
            prompt += *string;
        } else {
            prompt += json_dump(value);
        }
        prompt += "\n</parameter>\n";
    }
    prompt += "</function>\n</tool_call>";
}

} // namespace

bool supported_qwen35_chat_template(
    std::string_view chat_template) noexcept {
    if (chat_template.size() != supported_template_size) {
        return false;
    }
    std::uint64_t fingerprint = 14695981039346656037ULL;
    for (const auto character : chat_template) {
        fingerprint ^= static_cast<unsigned char>(character);
        fingerprint *= 1099511628211ULL;
    }
    return fingerprint == supported_template_fnv1a;
}

std::string qwen35_chat_prompt(
    std::span<const ChatMessage> messages,
    std::span<const Json> tools) {
    if (messages.empty()) {
        throw std::runtime_error("no messages provided");
    }
    if (messages.front().role != "system" &&
        std::any_of(
            messages.begin(),
            messages.end(),
            [](const ChatMessage &message) {
                return message.role == "system";
            })) {
        throw std::runtime_error("system message must be at the beginning");
    }

    std::vector<std::string> contents;
    contents.reserve(messages.size());
    for (const auto &message : messages) {
        contents.push_back(trim(message.content));
    }

    std::size_t last_query = messages.size();
    for (std::size_t index = messages.size(); index-- > 0;) {
        if (messages[index].role == "user" &&
            !(contents[index].starts_with("<tool_response>") &&
              contents[index].ends_with("</tool_response>"))) {
            last_query = index;
            break;
        }
    }
    if (last_query == messages.size()) {
        throw std::runtime_error("no user query found in messages");
    }

    std::string prompt;
    if (!tools.empty()) {
        prompt +=
            "<|im_start|>system\n"
            "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto &tool : tools) {
            prompt += "\n" + json_dump(tool);
        }
        prompt +=
            "\n</tools>"
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n<function=example_function_name>\n"
            "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
            "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
            "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
            "<IMPORTANT>\nReminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT>";
        if (messages.front().role == "system" && !contents.front().empty()) {
            prompt += "\n\n" + contents.front();
        }
        prompt += "<|im_end|>\n";
    } else if (messages.front().role == "system") {
        prompt += "<|im_start|>system\n" + contents.front() +
                  "<|im_end|>\n";
    }
    for (std::size_t index = 0; index < messages.size(); ++index) {
        const auto &role = messages[index].role;
        if (role == "system") {
            if (index != 0) {
                throw std::runtime_error(
                    "system message must be at the beginning");
            }
            continue;
        }
        if (role == "user") {
            prompt += "<|im_start|>user\n" + contents[index] +
                      "<|im_end|>\n";
            continue;
        }
        if (role == "tool") {
            if (index == 0 || messages[index - 1].role != "tool") {
                prompt += "<|im_start|>user";
            }
            prompt += "\n<tool_response>\n" + contents[index] +
                      "\n</tool_response>";
            if (index + 1 == messages.size() ||
                messages[index + 1].role != "tool") {
                prompt += "<|im_end|>\n";
            }
            continue;
        }
        if (role != "assistant") {
            throw std::runtime_error("unsupported input message role");
        }

        std::string reasoning;
        strip_assistant_reasoning(contents[index], reasoning);
        prompt += "<|im_start|>assistant\n";
        if (index > last_query) {
            prompt += "<think>\n" + reasoning + "\n</think>\n\n";
        }
        prompt += contents[index];
        for (std::size_t call_index = 0;
             call_index < messages[index].tool_calls.size();
             ++call_index) {
            if (call_index == 0) {
                if (!contents[index].empty()) {
                    prompt += "\n\n";
                }
            } else {
                prompt += "\n";
            }
            append_tool_call(prompt, messages[index].tool_calls[call_index]);
        }
        prompt += "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n<think>\n";
    return prompt;
}

} // namespace adi
