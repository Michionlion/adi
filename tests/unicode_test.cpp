#include "unicode.hpp"

#include <cassert>
#include <string>
#include <vector>

int main() {
    assert(adi::normalize_nfc("e\xCC\x81") == "\xC3\xA9");
    assert(adi::normalize_nfc("\xEA\xB0\x80") == "\xEA\xB0\x80");
    assert(adi::normalize_nfc("\xE1\x84\x80\xE1\x85\xA1") == "\xEA\xB0\x80");

    const auto pieces = adi::qwen35_pretokenize(
        "foo  bar a\xF0\x9F\x99\x82" "b 123 \xD9\xA1\xD9\xA2!");
    const std::vector<std::string> expected{
        "foo", " ", " bar", " a", "\xF0\x9F\x99\x82" "b", " ",
        "1", "2", "3", " ", "\xD9\xA1", "\xD9\xA2", "!",
    };
    assert(pieces == expected);

    assert(adi::unicode_properties(0x0627).letter);
    assert(adi::unicode_properties(0x0661).number);
    assert(adi::unicode_properties(0x0301).mark);
    assert(adi::unicode_properties(0x2028).whitespace);
    assert(adi::unicode_properties(0x105C0).letter); // Unicode 16 Todhri.
    assert(!adi::unicode_properties(0x10940).letter); // Assigned in Unicode 17.
    const auto unicode_16_composition = adi::codepoints_utf8(
        std::vector<std::uint32_t>{0x105D2, 0x0307});
    assert(adi::normalize_nfc(unicode_16_composition) ==
           unicode_16_composition); // The checkpoint normalizer is Unicode 15.
    const auto tokenizer_nfc_exception = adi::codepoints_utf8(
        std::vector<std::uint32_t>{0x11935, 0x11930});
    assert(adi::normalize_nfc(tokenizer_nfc_exception) ==
           tokenizer_nfc_exception);
}
