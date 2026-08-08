#pragma once

#include <cassert>
#include <cstdint>

namespace adi::detail {

constexpr std::uint32_t ne_words_per_tile = 64;
constexpr std::uint32_t ne_states_per_tile = 128;
constexpr std::uint32_t ne_rows_per_tile = 16;
constexpr std::uint32_t ne_states_per_row = 8;

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

// The same 128 states, in the same order, but reported as the local row they
// land in and the step within that row rather than as a flat index.
//
// State i lands in local row i >> 3, so the eight states of a row are already
// adjacent in the walk. Naming that structure lets a caller keep one row's
// running sum in a register for the eight states that touch it. The flat walk
// cannot: the row index changes under it, so the accumulator has to be an
// indexed slot the compiler must spill, and the sixteen adds a row receives
// per tile then chain through store-to-load forwarding instead of through the
// adder. That chain, not the state lookup, is what the single-vector kernel
// spends its time on.
//
// Step s covers tile elements 2*s and 2*s + 1 of the row, that is local
// columns s * 2 and s * 2 + 1.
template <typename Consume>
inline void for_each_ne_row_state(
    const std::uint16_t *words,
    Consume &&consume) {
    std::uint32_t state = words[0];
    std::uint32_t word = 1;
    bool high = true;
    for (std::uint32_t local_row = 0;
         local_row < ne_rows_per_tile;
         ++local_row) {
        for (std::uint32_t step = 0; step < ne_states_per_row; ++step) {
            if (local_row != 0 || step != 0) {
                std::uint32_t fresh;
                if (word == ne_words_per_tile) {
                    // The final transition wraps the stream back to words[0].
                    fresh = words[0] >> 8;
                } else if (high) {
                    fresh = words[word] >> 8;
                } else {
                    fresh = words[word] & 0xFFU;
                }
                state = ((state << 8) & 0xFFFFU) | fresh;
                if (word != ne_words_per_tile) {
                    if (!high) {
                        ++word;
                    }
                    high = !high;
                }
            }
            consume(local_row, step, state);
        }
    }
}

} // namespace adi::detail
