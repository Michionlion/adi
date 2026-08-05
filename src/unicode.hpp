#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace adi {

struct UnicodeProperties {
    bool letter;
    bool number;
    bool mark;
    bool whitespace;
};

[[nodiscard]] UnicodeProperties unicode_properties(std::uint32_t codepoint) noexcept;
[[nodiscard]] std::uint32_t unicode_casefold(std::uint32_t codepoint) noexcept;
[[nodiscard]] std::vector<std::uint32_t> utf8_codepoints(std::string_view text);
[[nodiscard]] std::string codepoints_utf8(
    const std::vector<std::uint32_t> &codepoints);
[[nodiscard]] std::string normalize_nfc(std::string_view text);
[[nodiscard]] std::vector<std::string> qwen35_pretokenize(
    std::string_view normalized_text);

} // namespace adi
