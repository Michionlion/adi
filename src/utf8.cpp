#include "utf8.hpp"

#include <stdexcept>

namespace adi {
namespace {

constexpr std::string_view replacement = "\xEF\xBF\xBD";

} // namespace

Utf8Scan scan_utf8(std::string_view input) noexcept {
    std::size_t position = 0;
    while (position < input.size()) {
        const auto first = static_cast<unsigned char>(input[position]);
        if (first <= 0x7FU) {
            ++position;
            continue;
        }

        std::size_t length = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum = 0;
        if (first >= 0xC2U && first <= 0xDFU) {
            length = 2;
            codepoint = first & 0x1FU;
            minimum = 0x80U;
        } else if (first >= 0xE0U && first <= 0xEFU) {
            length = 3;
            codepoint = first & 0x0FU;
            minimum = 0x800U;
        } else if (first >= 0xF0U && first <= 0xF4U) {
            length = 4;
            codepoint = first & 0x07U;
            minimum = 0x10000U;
        } else {
            return {Utf8Status::invalid, position, 1};
        }
        if (input.size() - position < length) {
            for (std::size_t index = position + 1; index < input.size(); ++index) {
                if ((static_cast<unsigned char>(input[index]) & 0xC0U) != 0x80U) {
                    return {Utf8Status::invalid, position, 1};
                }
            }
            return {Utf8Status::incomplete, position, input.size() - position};
        }
        for (std::size_t index = 1; index < length; ++index) {
            const auto byte = static_cast<unsigned char>(input[position + index]);
            if ((byte & 0xC0U) != 0x80U) {
                return {Utf8Status::invalid, position, 1};
            }
            codepoint = (codepoint << 6) | (byte & 0x3FU);
        }
        if (codepoint < minimum || codepoint > 0x10FFFFU ||
            (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
            return {Utf8Status::invalid, position, length};
        }
        position += length;
    }
    return {Utf8Status::valid, position, 0};
}

bool valid_utf8(std::string_view input) noexcept {
    return scan_utf8(input).status == Utf8Status::valid;
}

std::string sanitize_utf8(std::string_view input) {
    std::string result;
    result.reserve(input.size());
    while (!input.empty()) {
        const auto scan = scan_utf8(input);
        result.append(input.substr(0, scan.valid_prefix));
        input.remove_prefix(scan.valid_prefix);
        if (scan.status == Utf8Status::valid) {
            break;
        }
        result.append(replacement);
        if (scan.status == Utf8Status::incomplete) {
            break;
        }
        input.remove_prefix(scan.error_length);
    }
    return result;
}

void append_utf8(std::string &output, std::uint32_t codepoint) {
    if (codepoint <= 0x7FU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU &&
               !(codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint >= 0x10000U && codepoint <= 0x10FFFFU) {
        output.push_back(static_cast<char>(0xF0U | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        throw std::invalid_argument("codepoint is not a Unicode scalar value");
    }
}

} // namespace adi
