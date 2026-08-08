// The row-grouped expert walk must produce exactly the state sequence the
// recurrence produces, including the last state, whose 16-bit window is the
// only one that crosses the end of the 384-bit stream.
#include "expert_trellis.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace {

// An independent reference: read the 16-bit window bit by bit straight out of
// the stream. It shares no arithmetic with either walk, so agreeing with it
// pins down the byte and bit order rather than just the two walks agreeing
// with each other.
std::uint32_t window_state(
    const std::uint16_t *words,
    std::uint32_t state_index) {
    using adi::detail::expert_fresh_bits;
    using adi::detail::expert_register_bits;
    using adi::detail::expert_stream_bits;

    std::uint32_t value = 0;
    for (std::uint32_t bit = 0; bit < expert_register_bits; ++bit) {
        const auto position =
            (state_index * expert_fresh_bits + bit) % expert_stream_bits;
        const auto word = position / 16;
        const auto offset = position % 16;
        value = (value << 1) | ((words[word] >> (15 - offset)) & 1U);
    }
    return value;
}

std::vector<std::uint32_t> recurrence_states(const std::uint16_t *words) {
    std::vector<std::uint32_t> sequence;
    sequence.reserve(adi::detail::expert_states_per_tile);
    adi::detail::for_each_expert_state(
        words,
        [&](std::uint32_t index, std::uint32_t state) {
            assert(index == sequence.size());
            sequence.push_back(state);
        });
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
    const auto expected = recurrence_states(words);
    assert(expected.size() == adi::detail::expert_states_per_tile);

    // The recurrence and the bit-by-bit window must agree first, which is what
    // licenses treating the state register as a window at all.
    for (std::uint32_t index = 0; index < expected.size(); ++index) {
        assert(expected[index] == window_state(words, index));
    }

    std::vector<std::uint32_t> grouped;
    grouped.reserve(expected.size());
    adi::detail::for_each_expert_row_state(
        words,
        [&](std::uint32_t local_row, std::uint32_t step, std::uint32_t state) {
            const auto index = static_cast<std::uint32_t>(grouped.size());
            assert(local_row == index >> 1);
            assert(step == (index & 1U));
            assert(local_row < adi::detail::expert_rows_per_tile);
            assert(step < adi::detail::expert_states_per_row);
            assert(state <= 0xFFFFU);
            grouped.push_back(state);
        });
    assert(grouped == expected);

    // The wrapping window is the one case the masking is easy to get wrong, so
    // assert the last state directly rather than only through the sequence.
    const std::uint32_t wrapped =
        ((expected[30] << adi::detail::expert_fresh_bits) & 0xFFFFU) |
        adi::detail::extract_expert_stream_bits(
            words,
            adi::detail::expert_register_bits +
                30 * adi::detail::expert_fresh_bits);
    assert(expected[31] == wrapped);
}

} // namespace

int main() {
    std::vector<std::uint16_t> words(adi::detail::expert_words_per_tile);

    // Degenerate streams first: all zeros, all ones, and a single set bit in
    // every bit position exercise the shift and mask boundaries, including the
    // window that wraps the stream.
    std::fill(words.begin(), words.end(), 0);
    check_tile(words.data());
    std::fill(words.begin(), words.end(), 0xFFFFU);
    check_tile(words.data());
    for (std::size_t index = 0; index < words.size(); ++index) {
        for (int bit = 0; bit < 16; ++bit) {
            std::fill(words.begin(), words.end(), 0);
            words[index] = static_cast<std::uint16_t>(1U << bit);
            check_tile(words.data());
        }
    }

    std::uint64_t random = 0x51ED270BULL;
    for (std::uint32_t trial = 0; trial < 4000; ++trial) {
        for (auto &word : words) {
            word = static_cast<std::uint16_t>(next_random(random) & 0xFFFFU);
        }
        check_tile(words.data());
    }
}
