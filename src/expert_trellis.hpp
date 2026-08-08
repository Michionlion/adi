#pragma once

#include <cassert>
#include <cstdint>

namespace adi::detail {

constexpr std::uint32_t expert_words_per_tile = 24;
constexpr std::uint32_t expert_states_per_tile = 32;
constexpr std::uint32_t expert_stream_bits = 384;
constexpr std::uint32_t expert_register_bits = 16;
constexpr std::uint32_t expert_fresh_bits = 12;
constexpr std::uint32_t expert_rows_per_tile = 16;
constexpr std::uint32_t expert_states_per_row = 2;

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

// The same 32 states, in the same order, reported as the local row they land
// in and the step within that row.
//
// State i covers tile elements 8i..8i+7, so it fills half of local row i >> 1
// and the two states of a row are adjacent in the walk. Grouping them lets a
// caller keep the row's partial sum in a register across the sixteen adds it
// receives -- two states of eight components -- instead of accumulating into
// an indexed slot the compiler has to keep in memory.
//
// Each state is also read from the stream rather than derived from its
// predecessor. Shifting in twelve fresh bits and dropping the top twelve
// leaves a 16-bit window that has simply advanced twelve bits, so state i is
// the window at bit 12*i of the 384-bit stream. Unlike the non-expert codec
// the stride is not a whole byte, so a state costs an unaligned extract
// rather than a word load; what it buys is the same thing, independence.
// Written as the recurrence, every state waits for its predecessor and the
// value lookup waits for the state.
//
// The last state is the only one whose window crosses the end of the stream:
// 12*31 = 372, and 372 + 16 exceeds 384.
template <typename Consume>
inline void for_each_expert_row_state(
    const std::uint16_t *words,
    Consume &&consume) {
    for (std::uint32_t local_row = 0;
         local_row < expert_rows_per_tile;
         ++local_row) {
        for (std::uint32_t step = 0; step < expert_states_per_row; ++step) {
            const auto state_index = (local_row << 1) | step;
            const auto position = state_index * expert_fresh_bits;
            const auto word = position / expert_register_bits;
            const auto offset = position % expert_register_bits;
            const auto next =
                word + 1U == expert_words_per_tile ? 0U : word + 1U;
            const std::uint32_t window =
                (static_cast<std::uint32_t>(words[word]) <<
                 expert_register_bits) |
                words[next];
            consume(
                local_row,
                step,
                (window >> (expert_register_bits - offset)) & 0xFFFFU);
        }
    }
}

} // namespace adi::detail
