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
    std::span<const ChatMessage> messages) {
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
    if (messages.front().role == "system") {
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
        if (role != "assistant") {
            throw std::runtime_error("unsupported input message role");
        }

        std::string reasoning;
        strip_assistant_reasoning(contents[index], reasoning);
        prompt += "<|im_start|>assistant\n";
        if (index > last_query) {
            prompt += "<think>\n" + reasoning + "\n</think>\n\n";
        }
        prompt += contents[index] + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n<think>\n";
    return prompt;
}

} // namespace adi
