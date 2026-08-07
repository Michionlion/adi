#pragma once

#include <cassert>
#include <cstdint>

namespace adi::detail {

constexpr std::uint32_t ne_words_per_tile = 64;
constexpr std::uint32_t ne_states_per_tile = 128;

// Walks the 128 trellis states of one non-expert 16x16 tile.
//
// The non-expert codec is fixed at K=4/V=2: a 1024-bit stream per tile, a
// 16-bit register, and exactly eight fresh bits per transition. That makes
// every transition a whole byte, so the generic bit-window extractor is not
// needed. The byte order the generic extractor produces is:
//
//   state 0        words[0]
//   states 1..126  the high byte then the low byte of words[1] .. words[63]
//   state 127      the high byte of words[0], wrapping the stream
//
// Callers receive the state index and the 16-bit state. The two decoded
// values of state i cover tile element 2*i and 2*i + 1, that is local row
// i >> 3 and local columns (i & 7) * 2 and (i & 7) * 2 + 1.
template <typename Consume>
inline void for_each_ne_state(const std::uint16_t *words, Consume &&consume) {
    std::uint32_t state = words[0];
    std::uint32_t state_index = 0;
    consume(state_index++, state);
    for (std::uint32_t word = 1; word < ne_words_per_tile; ++word) {
        state = ((state << 8) & 0xFFFFU) | (words[word] >> 8);
        consume(state_index++, state);
        state = ((state << 8) & 0xFFFFU) | (words[word] & 0xFFU);
        consume(state_index++, state);
    }
    state = ((state << 8) & 0xFFFFU) | (words[0] >> 8);
    consume(state_index++, state);
    assert(state_index == ne_states_per_tile);
}

} // namespace adi::detail
