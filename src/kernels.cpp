#include "adi/kernels.hpp"
#include "batch_pack.hpp"
#include "codec_cache.hpp"
#include "expert_batch.hpp"
#include "expert_trellis.hpp"
#include "ne_batch.hpp"
#include "ne_trellis.hpp"
#include "parallel.hpp"
#include "profiling_internal.hpp"
#include "simd.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace adi {
namespace {

constexpr std::uint32_t tile_size = 16;
constexpr std::uint32_t values_per_state = detail::expert_values_per_state;
constexpr std::uint32_t words_per_tile = detail::expert_words_per_tile;
using detail::parallel_ranges;

bool is_power_of_two(std::uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

std::span<const float> expert_state_values(
    const MachExpertMatrix &matrix,
    ExpertScratch &scratch) {
    constexpr std::size_t expected =
        static_cast<std::size_t>(detail::codec_state_count) *
        detail::expert_values_per_state;
    if (!matrix.state_values.empty()) {
        if (matrix.state_values.size() != expected) {
            throw std::invalid_argument("Mach expert state cache shape mismatch");
        }
        return matrix.state_values;
    }
    if (scratch.state_values_source != matrix.tlut.data() ||
        scratch.state_value_components != detail::expert_values_per_state) {
        scratch.state_values = detail::build_expert_state_values(matrix.tlut);
        scratch.state_values_source = matrix.tlut.data();
        scratch.state_value_components = detail::expert_values_per_state;
    }
    return scratch.state_values;
}

std::span<const float> ne_state_values(
    const MachNeMatrix &matrix,
    ExpertScratch &scratch) {
    constexpr std::size_t expected =
        static_cast<std::size_t>(detail::codec_state_count) *
        detail::ne_values_per_state;
    if (!matrix.state_values.empty()) {
        if (matrix.state_values.size() != expected) {
            throw std::invalid_argument("Mach NE state cache shape mismatch");
        }
        return matrix.state_values;
    }
    if (scratch.state_values_source != matrix.tlut.data() ||
        scratch.state_value_components != detail::ne_values_per_state) {
        scratch.state_values = detail::build_ne_state_values(matrix.tlut);
        scratch.state_values_source = matrix.tlut.data();
        scratch.state_value_components = detail::ne_values_per_state;
    }
    return scratch.state_values;
}

// Reference accumulation for the batched non-expert stream. Every SIMD batch
// kernel must reproduce this bit for bit: tile column by tile column, state
// by state, first decoded value then second, with the two products added
// separately.
//
//   inputs  [batch, matrix.columns], already sign-applied and Hadamard'd
//   outputs [batch, matrix.rows], before the output Hadamard and signs
void ne_accumulate_tiles_scalar(
    const MachNeMatrix &matrix,
    std::span<const float> state_values,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs) {
    constexpr std::uint32_t ne_values_per_state = 2;
    using detail::ne_words_per_tile;
    const auto tile_rows = matrix.rows / tile_size;
    const auto tile_columns = matrix.columns / tile_size;

    parallel_ranges(
        tile_rows,
        4,
        [&](std::uint32_t row_begin, std::uint32_t row_end) {
            std::vector<float> row_sums(
                static_cast<std::size_t>(batch) * tile_size);
            for (std::uint32_t tile_row = row_begin;
                 tile_row < row_end;
                 ++tile_row) {
                std::fill(row_sums.begin(), row_sums.end(), 0.0F);
                for (std::uint32_t tile_column = 0;
                     tile_column < tile_columns;
                     ++tile_column) {
                    const auto tile_index =
                        static_cast<std::size_t>(tile_row) * tile_columns +
                        tile_column;
                    const auto *words =
                        matrix.trellis.data() +
                        tile_index * ne_words_per_tile;
                    const auto *tile_input =
                        inputs.data() + tile_column * tile_size;
                    detail::for_each_ne_state(
                        words,
                        [&](std::uint32_t state_index, std::uint32_t state) {
                            const auto local_row = state_index >> 3;
                            const auto local_column = (state_index & 7U) << 1;
                            const float weight0 =
                                state_values[
                                    static_cast<std::size_t>(state) *
                                    ne_values_per_state] *
                                matrix.weight_scale;
                            const float weight1 =
                                state_values[
                                    static_cast<std::size_t>(state) *
                                        ne_values_per_state +
                                    1] *
                                matrix.weight_scale;
                            for (std::uint32_t batch_index = 0;
                                 batch_index < batch;
                                 ++batch_index) {
                                const auto *row =
                                    tile_input +
                                    static_cast<std::size_t>(batch_index) *
                                        matrix.columns;
                                auto &sum =
                                    row_sums[
                                        static_cast<std::size_t>(batch_index) *
                                            tile_size +
                                        local_row];
                                sum += weight0 * row[local_column];
                                sum += weight1 * row[local_column + 1];
                            }
                        });
                }
                for (std::uint32_t batch_index = 0;
                     batch_index < batch;
                     ++batch_index) {
                    std::copy_n(
                        row_sums.begin() +
                            static_cast<std::size_t>(batch_index) *
                                tile_size,
                        tile_size,
                        outputs.begin() +
                            static_cast<std::size_t>(batch_index) *
                                matrix.rows +
                            tile_row * tile_size);
                }
            }
        });
}

// Reference accumulation for the batched expert stream. Every SIMD batch
// kernel must reproduce this bit for bit: tile column by tile column, state by
// state, component by component, into a per-tile partial that the tile's gamma
// scales exactly once before it reaches the row sum.
//
//   inputs  [batch, matrix.columns], already scaled and Hadamard'd
//   outputs [batch, matrix.rows], before the output Hadamard and scales
void expert_accumulate_tiles_scalar(
    const MachExpertMatrix &matrix,
    std::span<const float> state_values,
    std::span<const std::uint16_t> wave_indexes,
    std::span<const float> wave_gamma,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs) {
    const auto tile_rows = matrix.rows / tile_size;
    const auto tile_columns = matrix.columns / tile_size;

    std::vector<float> row_sums(
        static_cast<std::size_t>(batch) * tile_size);
    std::vector<float> partial(
        static_cast<std::size_t>(batch) * tile_size);
    for (std::uint32_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
        std::fill(row_sums.begin(), row_sums.end(), 0.0F);
        for (std::uint32_t tile_column = 0;
             tile_column < tile_columns;
             ++tile_column) {
            std::fill(partial.begin(), partial.end(), 0.0F);
            const auto tile_index =
                static_cast<std::size_t>(tile_row) * tile_columns +
                tile_column;
            const auto *words =
                matrix.trellis.data() + tile_index * words_per_tile;
            detail::for_each_expert_state(
                words,
                [&](std::uint32_t state_index, std::uint32_t state) {
                    const auto local_row = state_index >> 1;
                    const auto column = tile_column * tile_size +
                                        ((state_index & 1U) << 3);
                    for (std::uint32_t component = 0;
                         component < values_per_state;
                         ++component) {
                        const float weight =
                            state_values[
                                static_cast<std::size_t>(state) *
                                    values_per_state +
                                component];
                        for (std::uint32_t batch_index = 0;
                             batch_index < batch;
                             ++batch_index) {
                            partial[
                                static_cast<std::size_t>(batch_index) *
                                    tile_size +
                                local_row] +=
                                weight *
                                inputs[
                                    static_cast<std::size_t>(batch_index) *
                                        matrix.columns +
                                    column + component];
                        }
                    }
                });
            const float gamma = wave_gamma[wave_indexes[tile_index]];
            for (std::size_t index = 0; index < row_sums.size(); ++index) {
                row_sums[index] += partial[index] * gamma;
            }
        }
        for (std::uint32_t batch_index = 0;
             batch_index < batch;
             ++batch_index) {
            std::copy_n(
                row_sums.begin() +
                    static_cast<std::size_t>(batch_index) * tile_size,
                tile_size,
                outputs.begin() +
                    static_cast<std::size_t>(batch_index) * matrix.rows +
                    tile_row * tile_size);
        }
    }
}

std::span<const std::uint16_t> expert_wave_indexes(
    const MachExpertMatrix &matrix,
    std::uint32_t tile_rows,
    std::uint32_t tile_columns,
    ExpertScratch &scratch) {
    const auto expected =
        static_cast<std::size_t>(tile_rows) * tile_columns;
    if (!matrix.wave_indexes.empty()) {
        if (matrix.wave_indexes.size() != expected) {
            throw std::invalid_argument("Mach expert wave-index cache shape mismatch");
        }
        return matrix.wave_indexes;
    }
    if (scratch.wave_tile_rows != tile_rows ||
        scratch.wave_tile_columns != tile_columns) {
        scratch.wave_indexes =
            detail::build_wave_indexes(tile_rows, tile_columns);
        scratch.wave_tile_rows = tile_rows;
        scratch.wave_tile_columns = tile_columns;
    }
    return scratch.wave_indexes;
}

std::span<const float> expert_wave_gamma(
    const MachExpertMatrix &matrix,
    ExpertScratch &scratch) {
    if (!matrix.wave_gamma.empty()) {
        if (matrix.wave_gamma.size() != matrix.wave_gamma_f16.size()) {
            throw std::invalid_argument("Mach expert gamma cache shape mismatch");
        }
        return matrix.wave_gamma;
    }
    if (scratch.wave_gamma_source != matrix.wave_gamma_f16.data() ||
        scratch.wave_gamma_count != matrix.wave_gamma_f16.size()) {
        scratch.wave_gamma = detail::convert_f16_values(matrix.wave_gamma_f16);
        scratch.wave_gamma_source = matrix.wave_gamma_f16.data();
        scratch.wave_gamma_count = matrix.wave_gamma_f16.size();
    }
    return scratch.wave_gamma;
}

} // namespace

float f16_to_f32(std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    std::uint32_t exponent = (bits >> 10) & 0x1FU;
    std::uint32_t mantissa = bits & 0x03FFU;
    std::uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            std::int32_t shift = 0;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03FFU;
            const auto adjusted = static_cast<std::uint32_t>(113 - shift);
            result = sign | (adjusted << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1FU) {
        result = sign | 0x7F800000U | (mantissa << 13);
    } else {
        exponent += 112;
        result = sign | (exponent << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(result);
}

std::uint16_t f32_to_f16(float value) noexcept {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
    const auto exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    const auto mantissa = bits & 0x7FFFFFU;

    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        const auto normalized = mantissa | 0x800000U;
        const auto shift = static_cast<std::uint32_t>(14 - exponent);
        auto rounded = normalized >> shift;
        const auto remainder = normalized & ((1U << shift) - 1U);
        const auto halfway = 1U << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0)) {
            ++rounded;
        }
        return static_cast<std::uint16_t>(sign | rounded);
    }
    if (exponent >= 31) {
        if (((bits >> 23) & 0xFFU) == 0xFFU && mantissa != 0) {
            return static_cast<std::uint16_t>(sign | 0x7C00U | (mantissa >> 13) | 1U);
        }
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }

    auto rounded_mantissa = mantissa >> 13;
    const auto remainder = mantissa & 0x1FFFU;
    if (remainder > 0x1000U ||
        (remainder == 0x1000U && (rounded_mantissa & 1U) != 0)) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400U) {
            rounded_mantissa = 0;
            if (exponent + 1 >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7C00U);
            }
            return static_cast<std::uint16_t>(
                sign | (static_cast<std::uint16_t>(exponent + 1) << 10));
        }
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint16_t>(exponent) << 10) | rounded_mantissa);
}

float bf16_to_f32(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

void mach_expert_matvec(
    const MachExpertMatrix &matrix,
    std::span<const float> input,
    std::span<float> output,
    ExpertScratch &scratch) {
    detail::KernelTimer timer(
        KernelKind::expert,
        static_cast<std::uint64_t>(matrix.rows) * matrix.columns);
    if (!is_power_of_two(matrix.rows) || !is_power_of_two(matrix.columns) ||
        matrix.rows % tile_size != 0 || matrix.columns % tile_size != 0) {
        throw std::invalid_argument("Mach expert dimensions must be tiled powers of two");
    }
    if (input.size() != matrix.columns || output.size() != matrix.rows ||
        matrix.su_f16.size() != matrix.columns || matrix.sv_f16.size() != matrix.rows) {
        throw std::invalid_argument("Mach expert vector or scale shape mismatch");
    }
    const auto tile_rows = matrix.rows / tile_size;
    const auto tile_columns = matrix.columns / tile_size;
    const auto tile_count = static_cast<std::size_t>(tile_rows) * tile_columns;
    if (matrix.trellis.size() != tile_count * words_per_tile ||
        matrix.wave_gamma_f16.size() != tile_rows + tile_columns ||
        matrix.tlut.size() != 32768 * values_per_state) {
        throw std::invalid_argument("Mach expert codec tensor shape mismatch");
    }

    scratch.input.resize(matrix.columns);
    scratch.output.assign(matrix.rows, 0.0F);
    for (std::size_t index = 0; index < input.size(); ++index) {
        scratch.input[index] = input[index] * f16_to_f32(matrix.su_f16[index]);
    }
    detail::hadamard_transform(scratch.input);
    const auto state_values = expert_state_values(matrix, scratch);
    const auto wave_indexes =
        expert_wave_indexes(matrix, tile_rows, tile_columns, scratch);
    const auto wave_gamma = expert_wave_gamma(matrix, scratch);

    for (std::uint32_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
        float row_sums[tile_size] = {};
        for (std::uint32_t tile_column = 0; tile_column < tile_columns; ++tile_column) {
            const auto tile_index =
                static_cast<std::size_t>(tile_row) * tile_columns + tile_column;
            const auto *words = matrix.trellis.data() + tile_index * words_per_tile;
            const float gamma = wave_gamma[wave_indexes[tile_index]];
            // A row's partial sum is built from zero by the two states that
            // cover it and then used once, so it never needs to reach memory:
            // it stays in a register and folds straight into row_sums. The
            // adds keep their order, sixteen into the partial and then one
            // scaled add per row per tile column, so the result is unchanged.
            float partial = 0.0F;
            detail::for_each_expert_row_state(
                words,
                [&](std::uint32_t local_row,
                    std::uint32_t step,
                    std::uint32_t state) {
                    // Step 0 covers the low half of the row, step 1 the high
                    // half, eight tile elements each.
                    if (step == 0) {
                        partial = 0.0F;
                    }
                    const auto column =
                        tile_column * tile_size + (step << 3);
                    for (std::uint32_t component = 0;
                         component < values_per_state;
                         ++component) {
                        partial +=
                            state_values[
                                static_cast<std::size_t>(state) *
                                    values_per_state +
                                component] *
                            scratch.input[column + component];
                    }
                    if (step == detail::expert_states_per_row - 1) {
                        row_sums[local_row] += partial * gamma;
                    }
                });
        }
        std::copy_n(row_sums, tile_size, scratch.output.begin() + tile_row * tile_size);
    }

    detail::hadamard_transform(scratch.output);
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] = scratch.output[index] * f16_to_f32(matrix.sv_f16[index]);
    }
}

void mach_expert_matmul(
    const MachExpertMatrix &matrix,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    ExpertScratch &scratch) {
    if (batch == 1) {
        mach_expert_matvec(matrix, inputs, outputs, scratch);
        return;
    }
    detail::KernelTimer timer(
        KernelKind::expert_batch,
        static_cast<std::uint64_t>(matrix.rows) * matrix.columns * batch);
    if (batch == 0 || !is_power_of_two(matrix.rows) ||
        !is_power_of_two(matrix.columns) || matrix.rows % tile_size != 0 ||
        matrix.columns % tile_size != 0 ||
        inputs.size() != static_cast<std::size_t>(batch) * matrix.columns ||
        outputs.size() != static_cast<std::size_t>(batch) * matrix.rows ||
        matrix.su_f16.size() != matrix.columns ||
        matrix.sv_f16.size() != matrix.rows) {
        throw std::invalid_argument("Mach expert batch shape mismatch");
    }
    const auto tile_rows = matrix.rows / tile_size;
    const auto tile_columns = matrix.columns / tile_size;
    const auto tile_count =
        static_cast<std::size_t>(tile_rows) * tile_columns;
    if (matrix.trellis.size() != tile_count * words_per_tile ||
        matrix.wave_gamma_f16.size() != tile_rows + tile_columns ||
        matrix.tlut.size() != 32768 * values_per_state) {
        throw std::invalid_argument("Mach expert codec tensor shape mismatch");
    }

    scratch.input.resize(
        static_cast<std::size_t>(batch) * matrix.columns);
    scratch.output.assign(
        static_cast<std::size_t>(batch) * matrix.rows,
        0.0F);
    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        auto transformed = std::span<float>(scratch.input).subspan(
            static_cast<std::size_t>(batch_index) * matrix.columns,
            matrix.columns);
        const auto source = inputs.subspan(
            static_cast<std::size_t>(batch_index) * matrix.columns,
            matrix.columns);
        for (std::uint32_t column = 0;
             column < matrix.columns;
             ++column) {
            transformed[column] =
                source[column] * f16_to_f32(matrix.su_f16[column]);
        }
        detail::hadamard_transform(transformed);
    }
    const auto state_values = expert_state_values(matrix, scratch);
    const auto wave_indexes =
        expert_wave_indexes(matrix, tile_rows, tile_columns, scratch);
    const auto wave_gamma = expert_wave_gamma(matrix, scratch);

    // Decoding a packed weight once and applying it to a whole SIMD register
    // of independent batch items is the point of the batch kernel. Below the
    // threshold, and on ISAs without one, expert_accumulate_tiles_scalar stays
    // the reference implementation.
    const auto batch_lanes = detail::expert_batch_lanes();
    if (batch_lanes != 0 && batch >= detail::expert_batch_minimum) {
        scratch.batch_packed.resize(
            detail::batch_packed_floats(matrix.columns, batch, batch_lanes));
        detail::expert_matmul_tiles_batch(
            matrix,
            state_values,
            wave_indexes,
            wave_gamma,
            scratch.input,
            batch,
            scratch.output,
            scratch.batch_packed);
    } else {
        expert_accumulate_tiles_scalar(
            matrix,
            state_values,
            wave_indexes,
            wave_gamma,
            scratch.input,
            batch,
            scratch.output);
    }

    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        auto transformed = std::span<float>(scratch.output).subspan(
            static_cast<std::size_t>(batch_index) * matrix.rows,
            matrix.rows);
        detail::hadamard_transform(transformed);
        auto destination = outputs.subspan(
            static_cast<std::size_t>(batch_index) * matrix.rows,
            matrix.rows);
        for (std::uint32_t row = 0; row < matrix.rows; ++row) {
            destination[row] =
                transformed[row] * f16_to_f32(matrix.sv_f16[row]);
        }
    }
}

void mach_ne_matvec(
    const MachNeMatrix &matrix,
    std::span<const float> input,
    std::span<float> output,
    ExpertScratch &scratch) {
    detail::KernelTimer timer(
        KernelKind::non_expert,
        static_cast<std::uint64_t>(matrix.rows) * matrix.columns);
    constexpr std::uint32_t ne_values_per_state = 2;
    using detail::ne_words_per_tile;

    if (!is_power_of_two(matrix.rows) || !is_power_of_two(matrix.columns) ||
        matrix.rows % tile_size != 0 || matrix.columns % tile_size != 0) {
        throw std::invalid_argument("Mach NE dimensions must be tiled powers of two");
    }
    if (input.size() != matrix.columns || output.size() != matrix.rows ||
        matrix.su.size() != matrix.columns || matrix.sv.size() != matrix.rows) {
        throw std::invalid_argument("Mach NE vector or sign shape mismatch");
    }
    const auto tile_rows = matrix.rows / tile_size;
    const auto tile_columns = matrix.columns / tile_size;
    const auto tile_count = static_cast<std::size_t>(tile_rows) * tile_columns;
    if (matrix.trellis.size() != tile_count * ne_words_per_tile ||
        matrix.tlut.size() != 512 * ne_values_per_state) {
        throw std::invalid_argument("Mach NE codec tensor shape mismatch");
    }

    scratch.input.resize(matrix.columns);
    scratch.output.assign(matrix.rows, 0.0F);
    for (std::size_t index = 0; index < input.size(); ++index) {
        scratch.input[index] = input[index] * static_cast<float>(matrix.su[index]);
    }
    detail::hadamard_transform(scratch.input);
    const auto state_values = ne_state_values(matrix, scratch);

    parallel_ranges(tile_rows, 4, [&](std::uint32_t row_begin, std::uint32_t row_end) {
        for (std::uint32_t tile_row = row_begin; tile_row < row_end; ++tile_row) {
            float row_sums[tile_size] = {};
            for (std::uint32_t tile_column = 0; tile_column < tile_columns;
                 ++tile_column) {
                const auto tile_index =
                    static_cast<std::size_t>(tile_row) * tile_columns + tile_column;
                const auto *words =
                    matrix.trellis.data() + tile_index * ne_words_per_tile;
                const auto *tile_input =
                    scratch.input.data() + tile_column * tile_size;
                // The eight states of a local row arrive together, so the
                // row's running sum stays in a register across them and only
                // touches row_sums once on each side. The adds themselves keep
                // the flat walk's order exactly, so the result is unchanged.
                float row_sum = 0.0F;
                detail::for_each_ne_row_state(
                    words,
                    [&](std::uint32_t local_row,
                        std::uint32_t step,
                        std::uint32_t state) {
                        if (step == 0) {
                            row_sum = row_sums[local_row];
                        }
                        const auto local_column = step << 1;
                        // Two separate accumulations in this order: folding
                        // the pair into one expression changes the rounding.
                        const float weight0 =
                            state_values[
                                static_cast<std::size_t>(state) *
                                ne_values_per_state] *
                            matrix.weight_scale;
                        const float weight1 =
                            state_values[
                                static_cast<std::size_t>(state) *
                                    ne_values_per_state +
                                1] *
                            matrix.weight_scale;
                        row_sum += weight0 * tile_input[local_column];
                        row_sum += weight1 * tile_input[local_column + 1];
                        if (step == detail::ne_states_per_row - 1) {
                            row_sums[local_row] = row_sum;
                        }
                    });
            }
            std::copy_n(
                row_sums,
                tile_size,
                scratch.output.begin() + tile_row * tile_size);
        }
    });
    detail::hadamard_transform(scratch.output);
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] = scratch.output[index] * static_cast<float>(matrix.sv[index]);
    }
}

void mach_ne_matmul(
    const MachNeMatrix &matrix,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    ExpertScratch &scratch) {
    if (batch == 1) {
        mach_ne_matvec(matrix, inputs, outputs, scratch);
        return;
    }
    detail::KernelTimer timer(
        KernelKind::non_expert_batch,
        static_cast<std::uint64_t>(matrix.rows) * matrix.columns * batch);
    constexpr std::uint32_t ne_values_per_state = 2;
    using detail::ne_words_per_tile;

    if (batch == 0 || !is_power_of_two(matrix.rows) ||
        !is_power_of_two(matrix.columns) || matrix.rows % tile_size != 0 ||
        matrix.columns % tile_size != 0 ||
        inputs.size() != static_cast<std::size_t>(batch) * matrix.columns ||
        outputs.size() != static_cast<std::size_t>(batch) * matrix.rows ||
        matrix.su.size() != matrix.columns ||
        matrix.sv.size() != matrix.rows) {
        throw std::invalid_argument("Mach NE batch shape mismatch");
    }
    const auto tile_rows = matrix.rows / tile_size;
    const auto tile_columns = matrix.columns / tile_size;
    const auto tile_count =
        static_cast<std::size_t>(tile_rows) * tile_columns;
    if (matrix.trellis.size() != tile_count * ne_words_per_tile ||
        matrix.tlut.size() != 512 * ne_values_per_state) {
        throw std::invalid_argument("Mach NE codec tensor shape mismatch");
    }

    scratch.input.resize(
        static_cast<std::size_t>(batch) * matrix.columns);
    scratch.output.assign(
        static_cast<std::size_t>(batch) * matrix.rows,
        0.0F);
    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        auto transformed = std::span<float>(scratch.input)
                               .subspan(
                                   static_cast<std::size_t>(batch_index) *
                                       matrix.columns,
                                   matrix.columns);
        const auto source = inputs.subspan(
            static_cast<std::size_t>(batch_index) * matrix.columns,
            matrix.columns);
        for (std::uint32_t column = 0;
             column < matrix.columns;
             ++column) {
            transformed[column] =
                source[column] * static_cast<float>(matrix.su[column]);
        }
        detail::hadamard_transform(transformed);
    }
    const auto state_values = ne_state_values(matrix, scratch);

    // Decoding a packed weight once and applying it to a whole SIMD register
    // of independent batch items is the point of the batch kernel. Below the
    // threshold, and on ISAs without one, the scalar loop below stays the
    // reference implementation.
    const auto batch_lanes = detail::ne_batch_lanes();
    if (batch_lanes != 0 && batch >= detail::ne_batch_minimum) {
        scratch.batch_packed.resize(
            detail::batch_packed_floats(matrix.columns, batch, batch_lanes));
        detail::ne_matmul_tiles_batch(
            matrix,
            state_values,
            scratch.input,
            batch,
            scratch.output,
            scratch.batch_packed);
    } else {
        ne_accumulate_tiles_scalar(
            matrix, state_values, scratch.input, batch, scratch.output);
    }

    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        auto transformed = std::span<float>(scratch.output)
                               .subspan(
                                   static_cast<std::size_t>(batch_index) *
                                       matrix.rows,
                                   matrix.rows);
        detail::hadamard_transform(transformed);
        auto destination = outputs.subspan(
            static_cast<std::size_t>(batch_index) * matrix.rows,
            matrix.rows);
        for (std::uint32_t row = 0; row < matrix.rows; ++row) {
            destination[row] =
                transformed[row] * static_cast<float>(matrix.sv[row]);
        }
    }
}

void mach_embedding_row(
    const MachEmbedding &embedding,
    std::uint32_t token,
    std::span<float> output) {
    detail::KernelTimer timer(KernelKind::embedding, embedding.columns);
    constexpr std::uint32_t group = 64;
    if (token >= embedding.rows || output.size() != embedding.columns ||
        embedding.columns % group != 0 ||
        embedding.packed.size() !=
            static_cast<std::size_t>(embedding.rows) * embedding.columns / 2) {
        throw std::invalid_argument("Mach embedding shape mismatch");
    }
    const auto groups = embedding.columns / group;
    if (embedding.minimum_f16.size() != static_cast<std::size_t>(embedding.rows) * groups ||
        embedding.maximum_f16.size() != static_cast<std::size_t>(embedding.rows) * groups ||
        embedding.exception_indexes.size() != embedding.exception_bf16.size()) {
        throw std::invalid_argument("Mach embedding scale or exception shape mismatch");
    }
    const auto packed_row =
        embedding.packed.subspan(static_cast<std::size_t>(token) * embedding.columns / 2,
                                 embedding.columns / 2);
    const auto scale_offset = static_cast<std::size_t>(token) * groups;
    for (std::uint32_t column = 0; column < embedding.columns; ++column) {
        const auto byte = packed_row[column / 2];
        const auto code = column % 2 == 0 ? byte >> 4 : byte & 0x0FU;
        const auto scale_index = scale_offset + column / group;
        const float minimum = f16_to_f32(embedding.minimum_f16[scale_index]);
        const float maximum = f16_to_f32(embedding.maximum_f16[scale_index]);
        const float step = std::max(maximum - minimum, 1.0e-8F) / 15.0F;
        output[column] = minimum + static_cast<float>(code) * step;
    }

    const auto first = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(token) * embedding.columns);
    const auto last = first + embedding.columns;
    auto iterator = std::lower_bound(
        embedding.exception_indexes.begin(), embedding.exception_indexes.end(), first);
    while (iterator != embedding.exception_indexes.end() && *iterator < last) {
        const auto exception = static_cast<std::size_t>(
            iterator - embedding.exception_indexes.begin());
        output[*iterator - first] = bf16_to_f32(embedding.exception_bf16[exception]);
        ++iterator;
    }
}

void mach_head_matvec(
    const MachHeadChunk &head,
    std::span<const float> input,
    std::span<float> output) {
    detail::KernelTimer timer(
        KernelKind::output_head,
        static_cast<std::uint64_t>(head.rows) * head.columns);
    constexpr std::uint32_t group = 64;
    if (head.columns % group != 0 || input.size() != head.columns ||
        output.size() != head.rows ||
        head.packed.size() !=
            static_cast<std::size_t>(head.rows) * head.columns / 8 * 5 ||
        head.group_scale_f16.size() !=
            static_cast<std::size_t>(head.rows) * head.columns / group) {
        throw std::invalid_argument("Mach output head shape mismatch");
    }
    const auto bytes_per_row = head.columns / 8 * 5;
    const auto groups_per_row = head.columns / group;
    parallel_ranges(head.rows, 512, [&](std::uint32_t row_begin, std::uint32_t row_end) {
        for (std::uint32_t row = row_begin; row < row_end; ++row) {
            const auto packed_row =
                head.packed.subspan(static_cast<std::size_t>(row) * bytes_per_row,
                                    bytes_per_row);
            const auto scales = head.group_scale_f16.subspan(
                static_cast<std::size_t>(row) * groups_per_row,
                groups_per_row);
            output[row] =
                detail::int5_scaled_dot(packed_row, scales, input);
        }
    });

    if (head.protected_rows.empty()) {
        return;
    }
    if (head.protected_bf16.size() !=
        static_cast<std::size_t>(head.protected_rows.size()) * head.columns) {
        throw std::invalid_argument("Mach protected output row shape mismatch");
    }
    for (std::size_t protected_index = 0;
         protected_index < head.protected_rows.size(); ++protected_index) {
        const auto row = head.protected_rows[protected_index];
        if (row >= head.rows) {
            throw std::invalid_argument("Mach protected output row is out of range");
        }
        output[row] = detail::bf16_dot(
            head.protected_bf16.subspan(
                protected_index * head.columns,
                head.columns),
            input);
    }
}

void mach_head_matmul(
    const MachHeadChunk &head,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs) {
    if (batch == 1) {
        mach_head_matvec(head, inputs, outputs);
        return;
    }
    detail::KernelTimer timer(
        KernelKind::output_head_batch,
        static_cast<std::uint64_t>(head.rows) * head.columns * batch);
    constexpr std::uint32_t group = 64;
    if (batch == 0 || head.columns % group != 0 ||
        inputs.size() != static_cast<std::size_t>(batch) * head.columns ||
        outputs.size() != static_cast<std::size_t>(batch) * head.rows ||
        head.packed.size() !=
            static_cast<std::size_t>(head.rows) * head.columns / 8 * 5 ||
        head.group_scale_f16.size() !=
            static_cast<std::size_t>(head.rows) * head.columns / group) {
        throw std::invalid_argument("Mach output head batch shape mismatch");
    }
    const auto bytes_per_row = head.columns / 8 * 5;
    const auto groups_per_row = head.columns / group;
    const auto scratch_size =
        detail::int5_scaled_dot_batch_scratch_size(head.columns, batch);
    parallel_ranges(head.rows, 512, [&](std::uint32_t row_begin, std::uint32_t row_end) {
        std::vector<float> weight_scratch(scratch_size);
        std::vector<float> row_outputs(batch);
        for (std::uint32_t row = row_begin; row < row_end; ++row) {
            const auto packed_row =
                head.packed.subspan(static_cast<std::size_t>(row) * bytes_per_row,
                                    bytes_per_row);
            const auto scales = head.group_scale_f16.subspan(
                static_cast<std::size_t>(row) * groups_per_row,
                groups_per_row);
            detail::int5_scaled_dot_batch(
                packed_row,
                scales,
                inputs,
                batch,
                row_outputs,
                weight_scratch);
            for (std::uint32_t batch_index = 0;
                 batch_index < batch;
                 ++batch_index) {
                outputs[
                    static_cast<std::size_t>(batch_index) * head.rows + row] =
                    row_outputs[batch_index];
            }
        }
    });

    if (head.protected_rows.empty()) {
        return;
    }
    if (head.protected_bf16.size() !=
        static_cast<std::size_t>(head.protected_rows.size()) * head.columns) {
        throw std::invalid_argument("Mach protected output row shape mismatch");
    }
    for (std::size_t protected_index = 0;
         protected_index < head.protected_rows.size(); ++protected_index) {
        const auto row = head.protected_rows[protected_index];
        if (row >= head.rows) {
            throw std::invalid_argument("Mach protected output row is out of range");
        }
        for (std::uint32_t batch_index = 0;
             batch_index < batch;
             ++batch_index) {
            outputs[
                static_cast<std::size_t>(batch_index) * head.rows + row] =
                detail::bf16_dot(
                    head.protected_bf16.subspan(
                        protected_index * head.columns,
                        head.columns),
                    inputs.subspan(
                        static_cast<std::size_t>(batch_index) * head.columns,
                        head.columns));
        }
    }
}

void bf16_matvec(
    const Bf16Matrix &matrix,
    std::span<const float> input,
    std::span<float> output) {
    detail::KernelTimer timer(
        KernelKind::bf16_projection,
        static_cast<std::uint64_t>(matrix.rows) * matrix.columns);
    if (input.size() != matrix.columns || output.size() != matrix.rows ||
        matrix.values.size() !=
            static_cast<std::size_t>(matrix.rows) * matrix.columns) {
        throw std::invalid_argument("BF16 matrix-vector shape mismatch");
    }
    for (std::uint32_t row = 0; row < matrix.rows; ++row) {
        const auto offset = static_cast<std::size_t>(row) * matrix.columns;
        output[row] = detail::bf16_dot(
            matrix.values.subspan(offset, matrix.columns),
            input);
    }
}

void bf16_matmul(
    const Bf16Matrix &matrix,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs) {
    if (batch == 1) {
        bf16_matvec(matrix, inputs, outputs);
        return;
    }
    if (batch == 0 ||
        inputs.size() != static_cast<std::size_t>(batch) * matrix.columns ||
        outputs.size() != static_cast<std::size_t>(batch) * matrix.rows ||
        matrix.values.size() !=
            static_cast<std::size_t>(matrix.rows) * matrix.columns) {
        throw std::invalid_argument("BF16 matrix batch shape mismatch");
    }
    detail::KernelTimer timer(
        KernelKind::bf16_projection,
        static_cast<std::uint64_t>(matrix.rows) * matrix.columns * batch);
    const auto multiply_rows = [&](
        std::uint32_t row_begin,
        std::uint32_t row_end) {
            for (std::uint32_t row = row_begin; row < row_end; ++row) {
                const auto weights = matrix.values.subspan(
                    static_cast<std::size_t>(row) * matrix.columns,
                    matrix.columns);
                for (std::uint32_t batch_index = 0;
                     batch_index < batch;
                     ++batch_index) {
                    outputs[
                        static_cast<std::size_t>(batch_index) * matrix.rows +
                        row] =
                        detail::bf16_dot(
                            weights,
                            inputs.subspan(
                                static_cast<std::size_t>(batch_index) *
                                    matrix.columns,
                                matrix.columns));
                }
            }
        };
    constexpr std::uint64_t minimum_parallel_work = 2 * 1024 * 1024;
    const auto work = static_cast<std::uint64_t>(matrix.rows) *
                      matrix.columns * batch;
    if (work < minimum_parallel_work) {
        multiply_rows(0, matrix.rows);
    } else {
        parallel_ranges(matrix.rows, 64, multiply_rows);
    }
}

void rms_norm(
    std::span<const float> input,
    std::span<const std::uint16_t> weight_bf16,
    float weight_offset,
    float epsilon,
    std::span<float> output) {
    detail::KernelTimer timer(KernelKind::rms_norm, input.size());
    if (input.size() != weight_bf16.size() || output.size() != input.size() ||
        input.empty()) {
        throw std::invalid_argument("RMS norm shape mismatch");
    }
    float squares = 0.0F;
    for (const auto value : input) {
        squares += value * value;
    }
    const float scale =
        1.0F / std::sqrt(squares / static_cast<float>(input.size()) + epsilon);
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = input[index] * scale *
                        (weight_offset + bf16_to_f32(weight_bf16[index]));
    }
}

void l2_normalize(std::span<float> values, float epsilon) {
    if (values.empty() || epsilon < 0.0F) {
        throw std::invalid_argument("L2 norm shape or epsilon is invalid");
    }
    float squares = 0.0F;
    for (const auto value : values) {
        squares += value * value;
    }
    const float scale = 1.0F / std::sqrt(squares + epsilon);
    for (auto &value : values) {
        value *= scale;
    }
}

float silu(float value) noexcept {
    return value / (1.0F + std::exp(-value));
}

float sigmoid(float value) noexcept {
    return 1.0F / (1.0F + std::exp(-value));
}

} // namespace adi
