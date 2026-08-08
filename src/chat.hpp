#pragma once

#include "adi/json.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace adi {

struct ChatToolCall {
    std::string name;
    Json arguments;
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::vector<ChatToolCall> tool_calls;
};

[[nodiscard]] bool supported_qwen35_chat_template(
    std::string_view chat_template) noexcept;

// Text and function-tool specialization of the supported Qwen3.5 template.
// Generation uses the template's default thinking-enabled mode.
[[nodiscard]] std::string qwen35_chat_prompt(
    std::span<const ChatMessage> messages,
    std::span<const Json> tools = {});

} // namespace adi
