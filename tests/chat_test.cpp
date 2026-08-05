#include "adi/tokenizer.hpp"
#include "chat.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

int main() {
    assert(
        adi::qwen_user_prompt("  hello  ") ==
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n");

    const std::vector<adi::ChatMessage> history{
        {"system", "  follow the rules  "},
        {"user", " first "},
        {"assistant", "<think>\nprivate\n</think>\n\nvisible "},
        {"user", "\xE2\x80\xA8next\xE2\x80\xA8"},
    };
    assert(
        adi::qwen35_chat_prompt(history) ==
        "<|im_start|>system\nfollow the rules<|im_end|>\n"
        "<|im_start|>user\nfirst<|im_end|>\n"
        "<|im_start|>assistant\nvisible<|im_end|>\n"
        "<|im_start|>user\nnext<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n");

    const std::vector<adi::ChatMessage> assistant_after_query{
        {"user", "question"},
        {"assistant", "<think>\nprivate\n</think>\n\nvisible"},
    };
    assert(
        adi::qwen35_chat_prompt(assistant_after_query) ==
        "<|im_start|>user\nquestion<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nprivate\n</think>\n\n"
        "visible<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n");

    bool rejected = false;
    try {
        (void)adi::qwen35_chat_prompt(std::vector<adi::ChatMessage>{
            {"user", "question"},
            {"system", "late"},
        });
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        (void)adi::qwen35_chat_prompt(std::vector<adi::ChatMessage>{
            {"user", "<tool_response>not a query</tool_response>"},
        });
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected);

    assert(!adi::supported_qwen35_chat_template(""));
}
