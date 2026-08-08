#pragma once

// Shared body of the batch-oriented expert kernel. Each ISA translation unit
// instantiates it with a traits type and is compiled with FP contraction
// disabled, so the separate multiply and add below are never fused into an
// FMA. That is what keeps every lane bit-identical to the scalar kernel.

#include "adi/kernels.hpp"
#include "batch_pack.hpp"
#include "expert_trellis.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace adi::detail {

constexpr std::uint32_t expert_tile_size = 16;
constexpr std::uint32_t expert_batch_values_per_state = 8;

// Blocks accumulated in one pass over the tile columns. Each block holds two
// SIMD registers per tile row -- the tile partial and the running row sum --
// so keeping this small lets both stay in registers across the sixteen values
// that share a row.
constexpr std::uint32_t expert_batch_max_group = 4;

#if defined(__AVX2__)
template <std::uint32_t RowBegin>
__m256i expert_row_state_pairs_x86_256(const std::uint16_t *words) {
    static_assert(RowBegin == 0 || RowBegin == 8);
    const auto low = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
        words + (RowBegin == 0 ? 0 : 12)));
    const auto high = [&] {
        if constexpr (RowBegin == 0) {
            return _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(words + 6));
        } else {
            // Rows 12-15 need words 18..23 followed by word zero for the
            // final state's wrap. Build that lane from an in-bounds load.
            auto lane = _mm_srli_si128(
                _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(words + 16)),
                4);
            return _mm_insert_epi16(lane, words[0], 6);
        }
    }();
    auto source = _mm256_castsi128_si256(low);
    source = _mm256_inserti128_si256(source, high, 1);

    // Each 128-bit lane covers four rows starting at a six-word boundary.
    // Pull the adjacent words for every row into dword lanes. Odd rows also
    // need the following word to finish their second twelve-bit transition.
    const auto adjacent = _mm256_shuffle_epi8(
        source,
        _mm256_setr_epi8(
            0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11,
            0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11));
    const auto following = _mm256_shuffle_epi8(
        source,
        _mm256_setr_epi8(
            -1, -1, -1, -1, 4, 5, 6, 7,
            -1, -1, -1, -1, 10, 11, 12, 13,
            -1, -1, -1, -1, 4, 5, 6, 7,
            -1, -1, -1, -1, 10, 11, 12, 13));

    const auto even = _mm256_or_si256(
        _mm256_and_si256(adjacent, _mm256_set1_epi32(0xFFFF)),
        _mm256_or_si256(
            _mm256_slli_epi32(
                _mm256_and_si256(adjacent, _mm256_set1_epi32(0xF)),
                28),
            _mm256_srli_epi32(
                _mm256_and_si256(
                    adjacent, _mm256_set1_epi32(0xFFF00000)),
                4)));
    const auto odd = _mm256_or_si256(
        _mm256_or_si256(
            _mm256_and_si256(
                _mm256_srli_epi32(adjacent, 24),
                _mm256_set1_epi32(0xFF)),
            _mm256_slli_epi32(
                _mm256_and_si256(adjacent, _mm256_set1_epi32(0xFF)),
                8)),
        _mm256_or_si256(
            _mm256_and_si256(
                _mm256_srli_epi32(following, 12),
                _mm256_set1_epi32(0xF0000)),
            _mm256_slli_epi32(
                _mm256_and_si256(following, _mm256_set1_epi32(0xFFF)),
                20)));
    return _mm256_blend_epi32(even, odd, 0xAA);
}
#endif

// Accumulates one tile row of the matrix for Blocks batch blocks.
//
// The scalar kernel walks the 32 states of a tile in order and sends state i
// to local row i >> 1, so the two states of a row are adjacent and each row's
// sixteen products are contiguous in the walk. The order preserved here is
// therefore: tile column by tile column, then for each row the first state's
// eight components followed by the second state's eight, accumulated into a
// per-tile partial that is scaled by the tile's gamma and folded into the row
// sum exactly once.
template <typename Traits, std::uint32_t Blocks>
void expert_accumulate_tile_row(
    const std::uint16_t *tile_words,
    std::uint32_t tile_columns,
    const std::uint16_t *wave_index_row,
    const float *wave_gamma,
    const float *packed,
    std::size_t block_stride,
    const float *state_values,
    float *row_output) {
    using Vec = typename Traits::Vec;
    constexpr auto lanes = Traits::lanes;
    constexpr auto accumulators = Blocks * expert_tile_size * lanes;

    alignas(64) float accumulator[accumulators] = {};
    std::uint16_t states[expert_states_per_tile];
    std::uint32_t state_pairs[expert_rows_per_tile];

    for (std::uint32_t tile_column = 0; tile_column < tile_columns;
         ++tile_column) {
        const auto *words =
            tile_words +
            static_cast<std::size_t>(tile_column) * expert_words_per_tile;
        if constexpr (lanes == 8 || lanes == 16) {
            // Derive sixteen row pairs with lane-local shuffles. AVX-512 CPUs
            // support these AVX2 instructions too, and the arithmetic below
            // remains native to each ISA's batch width.
            _mm256_storeu_si256(
                reinterpret_cast<__m256i *>(state_pairs),
                expert_row_state_pairs_x86_256<0>(words));
            _mm256_storeu_si256(
                reinterpret_cast<__m256i *>(state_pairs + 8),
                expert_row_state_pairs_x86_256<8>(words));
        } else {
            for_each_expert_state(
                words,
                [&](std::uint32_t state_index, std::uint32_t state) {
                    states[state_index] = static_cast<std::uint16_t>(state);
                });
        }
        const Vec gamma =
            Traits::broadcast(wave_gamma[wave_index_row[tile_column]]);
        const float *tile =
            packed +
            static_cast<std::size_t>(tile_column) * expert_tile_size * lanes;
        for (std::uint32_t row = 0; row < expert_tile_size; ++row) {
            // A whole tile's contribution is scaled by one gamma, so it
            // accumulates separately and folds in once -- the vector form of
            // the scalar kernel's per-tile `partial` array.
            Vec partial[Blocks];
            for (std::uint32_t block = 0; block < Blocks; ++block) {
                partial[block] = Traits::zero();
            }
            for (std::uint32_t half = 0; half < 2; ++half) {
                const auto state = [&] {
                    if constexpr (lanes == 8 || lanes == 16) {
                        return static_cast<std::uint16_t>(
                            state_pairs[row] >> (half << 4));
                    } else {
                        return states[(row << 1) | half];
                    }
                }();
                const float *values =
                    state_values + static_cast<std::size_t>(state) *
                                       expert_batch_values_per_state;
                const float *column = tile + (half << 3) * lanes;
                for (std::uint32_t component = 0;
                     component < expert_batch_values_per_state;
                     ++component) {
                    const Vec weight = Traits::broadcast(values[component]);
                    const float *source =
                        column + static_cast<std::size_t>(component) * lanes;
                    for (std::uint32_t block = 0; block < Blocks; ++block) {
                        partial[block] = Traits::add(
                            partial[block],
                            Traits::mul(
                                weight,
                                Traits::load(source + block * block_stride)));
                    }
                }
            }
            for (std::uint32_t block = 0; block < Blocks; ++block) {
                float *slot =
                    accumulator + (block * expert_tile_size + row) * lanes;
                Traits::store(
                    slot,
                    Traits::add(
                        Traits::load(slot),
                        Traits::mul(partial[block], gamma)));
            }
        }
    }
    std::copy_n(accumulator, accumulators, row_output);
}

template <typename Traits>
void expert_tiles_batch(
    const MachExpertMatrix &matrix,
    std::span<const float> state_values,
    std::span<const std::uint16_t> wave_indexes,
    std::span<const float> wave_gamma,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> packed) {
    constexpr auto lanes = Traits::lanes;
    const auto rows = matrix.rows;
    const auto columns = matrix.columns;
    const auto tile_rows = rows / expert_tile_size;
    const auto tile_columns = columns / expert_tile_size;
    const auto blocks = (batch + lanes - 1) / lanes;
    const auto block_stride = static_cast<std::size_t>(columns) * lanes;

    pack_batch_inputs<lanes>(inputs.data(), batch, columns, packed.data());

    for (std::uint32_t base = 0; base < blocks;
         base += expert_batch_max_group) {
        const auto group = std::min(expert_batch_max_group, blocks - base);
        const float *group_packed =
            packed.data() + static_cast<std::size_t>(base) * block_stride;
        parallel_ranges(
            tile_rows,
            1,
            [&](std::uint32_t row_begin, std::uint32_t row_end) {
                alignas(64) float row_output[
                    expert_batch_max_group * expert_tile_size * lanes];
                for (std::uint32_t tile_row = row_begin; tile_row < row_end;
                     ++tile_row) {
                    const auto *tile_words =
                        matrix.trellis.data() +
                        static_cast<std::size_t>(tile_row) * tile_columns *
                            expert_words_per_tile;
                    const auto *wave_index_row =
                        wave_indexes.data() +
                        static_cast<std::size_t>(tile_row) * tile_columns;
                    switch (group) {
                    case 1:
                        expert_accumulate_tile_row<Traits, 1>(
                            tile_words,
                            tile_columns,
                            wave_index_row,
                            wave_gamma.data(),
                            group_packed,
                            block_stride,
                            state_values.data(),
                            row_output);
                        break;
                    case 2:
                        expert_accumulate_tile_row<Traits, 2>(
                            tile_words,
                            tile_columns,
                            wave_index_row,
                            wave_gamma.data(),
                            group_packed,
                            block_stride,
                            state_values.data(),
                            row_output);
                        break;
                    case 3:
                        expert_accumulate_tile_row<Traits, 3>(
                            tile_words,
                            tile_columns,
                            wave_index_row,
                            wave_gamma.data(),
                            group_packed,
                            block_stride,
                            state_values.data(),
                            row_output);
                        break;
                    default:
                        expert_accumulate_tile_row<Traits, 4>(
                            tile_words,
                            tile_columns,
                            wave_index_row,
                            wave_gamma.data(),
                            group_packed,
                            block_stride,
                            state_values.data(),
                            row_output);
                        break;
                    }
                    for (std::uint32_t block = 0; block < group; ++block) {
                        for (std::uint32_t lane = 0; lane < lanes; ++lane) {
                            const auto item = (base + block) * lanes + lane;
                            if (item >= batch) {
                                continue;
                            }
                            float *destination =
                                outputs.data() +
                                static_cast<std::size_t>(item) * rows +
                                static_cast<std::size_t>(tile_row) *
                                    expert_tile_size;
                            for (std::uint32_t row = 0;
                                 row < expert_tile_size;
                                 ++row) {
                                destination[row] = row_output
                                    [(block * expert_tile_size + row) * lanes +
                                     lane];
                            }
                        }
                    }
                }
            });
    }
}

} // namespace adi::detail
