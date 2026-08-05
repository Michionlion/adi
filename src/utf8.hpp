#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace adi {

enum class Utf8Status {
    valid,
    incomplete,
    invalid,
};

struct Utf8Scan {
    Utf8Status status;
    std::size_t valid_prefix;
    std::size_t error_length;
};

// Scans complete Unicode scalar values. An incomplete final sequence is distinct
// from malformed UTF-8 so streaming callers can retain it for the next chunk.
[[nodiscard]] Utf8Scan scan_utf8(std::string_view input) noexcept;
[[nodiscard]] bool valid_utf8(std::string_view input) noexcept;
[[nodiscard]] std::string sanitize_utf8(std::string_view input);
void append_utf8(std::string &output, std::uint32_t codepoint);

} // namespace adi
