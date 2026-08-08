// The specialized non-expert trellis walk must produce exactly the state
// sequence the generic bit-window extractor produced, including the
// transition that wraps the end of the 1024-bit stream back to words[0].
#include "ne_trellis.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace {

// The generic extractor, reproduced verbatim as the reference. It handles any
// fresh-bit width; the specialized walk exploits the fact that the non-expert
// codec always uses eight.
std::uint32_t extract_stream_bits(
    const std::uint16_t *words,
    std::uint32_t word_count,
    std::uint32_t bit_count,
    std::uint32_t position,
    std::uint32_t width) {
    position %= bit_count;
    const auto word = position / 16;
    const auto offset = position % 16;
    const std::uint32_t window =
        (static_cast<std::uint32_t>(words[word]) << 16) |
        words[(word + 1) % word_count];
    return (window >> (32 - offset - width)) & ((1U << width) - 1U);
}

std::vector<std::uint32_t> reference_states(const std::uint16_t *words) {
    constexpr std::uint32_t register_bits = 16;
    constexpr std::uint32_t fresh_bits = 8;
    constexpr std::uint32_t stream_bits = 1024;
    constexpr std::uint32_t word_count = 64;
    constexpr std::uint32_t states = 128;

    std::vector<std::uint32_t> sequence;
    sequence.reserve(states);
    std::uint32_t state = words[0];
    for (std::uint32_t index = 0; index < states; ++index) {
        if (index != 0) {
            const auto position = register_bits + (index - 1) * fresh_bits;
            const auto fresh = extract_stream_bits(
                words, word_count, stream_bits, position, fresh_bits);
            state = ((state << fresh_bits) & 0xFFFFU) | fresh;
        }
        sequence.push_back(state);
    }
    return sequence;
}

std::uint64_t next_random(std::uint64_t &state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

void check_tile(const std::uint16_t *words) {
    const auto expected = reference_states(words);
    std::vector<std::uint32_t> actual;
    actual.reserve(expected.size());
    adi::detail::for_each_ne_state(
        words,
        [&](std::uint32_t index, std::uint32_t state) {
            assert(index == actual.size());
            assert(state <= 0xFFFFU);
            actual.push_back(state);
        });
    assert(actual == expected);

    // The wrapped transition is the one case the byte order is easy to get
    // wrong, so assert its value directly rather than only through the
    // sequence comparison.
    const std::uint32_t wrapped =
        ((expected[126] << 8) & 0xFFFFU) | (words[0] >> 8);
    assert(expected[127] == wrapped);

    // The row-grouped walk must visit the same states in the same order, and
    // must report the local row and step that the flat index decomposes into.
    // The single-vector kernel relies on both: on the order for its exact
    // arithmetic, and on the grouping to hold a row's sum in a register.
    std::vector<std::uint32_t> grouped;
    grouped.reserve(expected.size());
    adi::detail::for_each_ne_row_state(
        words,
        [&](std::uint32_t local_row, std::uint32_t step, std::uint32_t state) {
            const auto index = static_cast<std::uint32_t>(grouped.size());
            assert(local_row == index >> 3);
            assert(step == (index & 7U));
            assert(local_row < adi::detail::ne_rows_per_tile);
            assert(step < adi::detail::ne_states_per_row);
            assert(state <= 0xFFFFU);
            grouped.push_back(state);
        });
    assert(grouped == expected);
}

} // namespace

int main() {
    std::vector<std::uint16_t> words(64);

    // Degenerate streams first: all zeros, all ones, and a single set bit in
    // each word position exercise the shift and mask boundaries.
    std::fill(words.begin(), words.end(), 0);
    check_tile(words.data());
    std::fill(words.begin(), words.end(), 0xFFFFU);
    check_tile(words.data());
    for (std::size_t index = 0; index < words.size(); ++index) {
        std::fill(words.begin(), words.end(), 0);
        words[index] = 0xFF00U;
        check_tile(words.data());
        words[index] = 0x00FFU;
        check_tile(words.data());
    }

    std::uint64_t random = 0x1234ABCDULL;
    for (std::uint32_t trial = 0; trial < 4000; ++trial) {
        for (auto &word : words) {
            word = static_cast<std::uint16_t>(next_random(random) & 0xFFFFU);
        }
        check_tile(words.data());
    }
}
