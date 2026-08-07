#pragma once

// Shared body of the batch-oriented non-expert kernel. Each ISA translation
// unit instantiates it with a traits type and is compiled with FP contraction
// disabled, so the separate multiply and add below are never fused into an
// FMA. That is what keeps every lane bit-identical to the scalar kernel.

#include "adi/kernels.hpp"
#include "ne_trellis.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace adi::detail {

constexpr std::uint32_t ne_tile_size = 16;
constexpr std::uint32_t ne_values_per_state_batch = 2;

// Blocks accumulated in one pass over the tile columns. Each block holds one
// SIMD register per tile row, so keeping this small lets the accumulators
// stay in registers across the eight states that share a row.
constexpr std::uint32_t ne_batch_max_group = 4;

// Rearranges [batch, columns] into [block, column, lane]. Padding lanes are
// zero: they contribute nothing to any accumulation and their results are
// never written back to the caller's output.
template <std::uint32_t Lanes>
void ne_pack_inputs(
    const float *inputs,
    std::uint32_t batch,
    std::uint32_t columns,
    float *packed) {
    const auto blocks = (batch + Lanes - 1) / Lanes;
    for (std::uint32_t block = 0; block < blocks; ++block) {
        float *destination =
            packed + static_cast<std::size_t>(block) * columns * Lanes;
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto item = block * Lanes + lane;
            if (item >= batch) {
                for (std::uint32_t column = 0; column < columns; ++column) {
                    destination[static_cast<std::size_t>(column) * Lanes +
                                lane] = 0.0F;
                }
                continue;
            }
            const float *source =
                inputs + static_cast<std::size_t>(item) * columns;
            for (std::uint32_t column = 0; column < columns; ++column) {
                destination[static_cast<std::size_t>(column) * Lanes + lane] =
                    source[column];
            }
        }
    }
}

// Accumulates one tile row of the matrix for Blocks batch blocks.
//
// The scalar kernel walks states 0..127 and sends state i to local row i >> 3.
// The eight states of a row are therefore contiguous, which lets the row
// accumulators live in registers across them while preserving the exact
// order: tile column by tile column, state by state, first value then second.
template <typename Traits, std::uint32_t Blocks>
void ne_accumulate_tile_row(
    const std::uint16_t *tile_words,
    std::uint32_t tile_columns,
    const float *packed,
    std::size_t block_stride,
    const float *state_values,
    float weight_scale,
    float *row_output) {
    using Vec = typename Traits::Vec;
    constexpr auto lanes = Traits::lanes;
    constexpr auto accumulators = Blocks * ne_tile_size * lanes;

    alignas(64) float accumulator[accumulators] = {};
    std::uint16_t states[ne_states_per_tile];

    for (std::uint32_t tile_column = 0; tile_column < tile_columns;
         ++tile_column) {
        const auto *words =
            tile_words + static_cast<std::size_t>(tile_column) *
                             ne_words_per_tile;
        for_each_ne_state(
            words,
            [&](std::uint32_t state_index, std::uint32_t state) {
                states[state_index] = static_cast<std::uint16_t>(state);
            });
        const float *tile =
            packed + static_cast<std::size_t>(tile_column) * ne_tile_size *
                         lanes;
        for (std::uint32_t row = 0; row < ne_tile_size; ++row) {
            Vec sums[Blocks];
            for (std::uint32_t block = 0; block < Blocks; ++block) {
                sums[block] = Traits::load(
                    accumulator + (block * ne_tile_size + row) * lanes);
            }
            for (std::uint32_t step = 0; step < 8; ++step) {
                const auto state = states[(row << 3) | step];
                const float weight0 =
                    state_values[static_cast<std::size_t>(state) *
                                 ne_values_per_state_batch] *
                    weight_scale;
                const float weight1 =
                    state_values[static_cast<std::size_t>(state) *
                                     ne_values_per_state_batch +
                                 1] *
                    weight_scale;
                const Vec broadcast0 = Traits::broadcast(weight0);
                const Vec broadcast1 = Traits::broadcast(weight1);
                const float *column = tile + (step << 1) * lanes;
                for (std::uint32_t block = 0; block < Blocks; ++block) {
                    const float *values = column + block * block_stride;
                    sums[block] = Traits::add(
                        sums[block],
                        Traits::mul(broadcast0, Traits::load(values)));
                    sums[block] = Traits::add(
                        sums[block],
                        Traits::mul(
                            broadcast1, Traits::load(values + lanes)));
                }
            }
            for (std::uint32_t block = 0; block < Blocks; ++block) {
                Traits::store(
                    accumulator + (block * ne_tile_size + row) * lanes,
                    sums[block]);
            }
        }
    }
    std::copy_n(accumulator, accumulators, row_output);
}

template <typename Traits>
void ne_tiles_batch(
    const MachNeMatrix &matrix,
    std::span<const float> state_values,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> packed) {
    constexpr auto lanes = Traits::lanes;
    const auto rows = matrix.rows;
    const auto columns = matrix.columns;
    const auto tile_rows = rows / ne_tile_size;
    const auto tile_columns = columns / ne_tile_size;
    const auto blocks = (batch + lanes - 1) / lanes;
    const auto block_stride = static_cast<std::size_t>(columns) * lanes;

    ne_pack_inputs<lanes>(inputs.data(), batch, columns, packed.data());

    for (std::uint32_t base = 0; base < blocks; base += ne_batch_max_group) {
        const auto group = std::min(ne_batch_max_group, blocks - base);
        const float *group_packed =
            packed.data() + static_cast<std::size_t>(base) * block_stride;
        parallel_ranges(
            tile_rows,
            1,
            [&](std::uint32_t row_begin, std::uint32_t row_end) {
                alignas(64) float row_output[
                    ne_batch_max_group * ne_tile_size * lanes];
                for (std::uint32_t tile_row = row_begin; tile_row < row_end;
                     ++tile_row) {
                    const auto *tile_words =
                        matrix.trellis.data() +
                        static_cast<std::size_t>(tile_row) * tile_columns *
                            ne_words_per_tile;
                    switch (group) {
                    case 1:
                        ne_accumulate_tile_row<Traits, 1>(
                            tile_words,
                            tile_columns,
                            group_packed,
                            block_stride,
                            state_values.data(),
                            matrix.weight_scale,
                            row_output);
                        break;
                    case 2:
                        ne_accumulate_tile_row<Traits, 2>(
                            tile_words,
                            tile_columns,
                            group_packed,
                            block_stride,
                            state_values.data(),
                            matrix.weight_scale,
                            row_output);
                        break;
                    case 3:
                        ne_accumulate_tile_row<Traits, 3>(
                            tile_words,
                            tile_columns,
                            group_packed,
                            block_stride,
                            state_values.data(),
                            matrix.weight_scale,
                            row_output);
                        break;
                    default:
                        ne_accumulate_tile_row<Traits, 4>(
                            tile_words,
                            tile_columns,
                            group_packed,
                            block_stride,
                            state_values.data(),
                            matrix.weight_scale,
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
                                    ne_tile_size;
                            for (std::uint32_t row = 0; row < ne_tile_size;
                                 ++row) {
                                destination[row] = row_output
                                    [(block * ne_tile_size + row) * lanes +
                                     lane];
                            }
                        }
                    }
                }
            });
    }
}

} // namespace adi::detail
