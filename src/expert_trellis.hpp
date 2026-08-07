#pragma once

#include <cassert>
#include <cstdint>

namespace adi::detail {

constexpr std::uint32_t expert_words_per_tile = 24;
constexpr std::uint32_t expert_states_per_tile = 32;
constexpr std::uint32_t expert_stream_bits = 384;
constexpr std::uint32_t expert_register_bits = 16;
constexpr std::uint32_t expert_fresh_bits = 12;

// Reads `expert_fresh_bits` from the tile's bit stream at `position`, wrapping
// at the end of the stream. The last transition genuinely wraps: the final
// position is 16 + 30*12 = 376, and 376 + 12 exceeds the 384-bit stream, so it
// takes the tail of the last word and the head of the first.
[[nodiscard]] inline std::uint32_t extract_expert_stream_bits(
    const std::uint16_t *words,
    std::uint32_t position) noexcept {
    position %= expert_stream_bits;
    const auto word = position / 16;
    const auto offset = position % 16;
    const std::uint32_t window =
        (static_cast<std::uint32_t>(words[word]) << 16) |
        words[(word + 1) % expert_words_per_tile];
    return (window >> (32 - offset - expert_fresh_bits)) &
           ((1U << expert_fresh_bits) - 1U);
}

// Walks the 32 trellis states of one expert 16x16 tile.
//
// The expert codec is K=1.5/V=8: a 384-bit stream per tile, a 16-bit register,
// and twelve fresh bits per transition. Twelve bits is not a whole byte, so
// unlike the non-expert walk this one cannot avoid the bit-window extractor.
//
// Callers receive the state index and the 16-bit state. The eight decoded
// values of state i cover tile elements 8*i .. 8*i + 7. Sixteen elements to a
// tile row makes that local row i >> 1, local columns (i & 1) * 8 + component:
// each state fills exactly half a row, and the two states of a row are
// adjacent in the walk.
template <typename Consume>
inline void for_each_expert_state(
    const std::uint16_t *words,
    Consume &&consume) {
    std::uint32_t state = words[0];
    consume(0U, state);
    for (std::uint32_t state_index = 1;
         state_index < expert_states_per_tile;
         ++state_index) {
        const auto position =
            expert_register_bits + (state_index - 1) * expert_fresh_bits;
        state = ((state << expert_fresh_bits) & 0xFFFFU) |
                extract_expert_stream_bits(words, position);
        consume(state_index, state);
    }
}

} // namespace adi::detail
