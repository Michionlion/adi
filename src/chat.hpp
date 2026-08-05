#pragma once

#include <span>
#include <string>
#include <string_view>

namespace adi {

struct ChatMessage {
    std::string role;
    std::string content;
};

[[nodiscard]] bool supported_qwen35_chat_template(
    std::string_view chat_template) noexcept;

// Text-only specialization of the supported Qwen3.5 template. Generation uses
// the template's default thinking-enabled mode.
[[nodiscard]] std::string qwen35_chat_prompt(
    std::span<const ChatMessage> messages);

} // namespace adi
