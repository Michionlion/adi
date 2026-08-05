#include "adi/kernels.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace adi {
namespace {

constexpr std::uint32_t tile_size = 16;
constexpr std::uint32_t tile_values = tile_size * tile_size;
constexpr std::uint32_t register_bits = 16;
constexpr std::uint32_t values_per_state = 8;
constexpr std::uint32_t fresh_bits = 12;
constexpr std::uint32_t states_per_tile = tile_values / values_per_state;
constexpr std::uint32_t stream_bits = 384;
constexpr std::uint32_t words_per_tile = stream_bits / 16;

bool is_power_of_two(std::uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

void hadamard(std::span<float> values) {
    for (std::size_t stride = 1; stride < values.size(); stride *= 2) {
        for (std::size_t block = 0; block < values.size(); block += 2 * stride) {
            for (std::size_t index = 0; index < stride; ++index) {
                const float left = values[block + index];
                const float right = values[block + stride + index];
                values[block + index] = left + right;
                values[block + stride + index] = left - right;
            }
        }
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(values.size()));
    for (auto &value : values) {
        value *= scale;
    }
}

std::uint32_t stream_bit(const std::uint16_t *words, std::uint32_t position) {
    if (position >= stream_bits) {
        position -= stream_bits;
    }
    return (words[position / 16] >> (15 - position % 16)) & 1U;
}

std::uint32_t fresh_value(const std::uint16_t *words, std::uint32_t position) {
    std::uint32_t value = 0;
    for (std::uint32_t bit = 0; bit < fresh_bits; ++bit) {
        value = (value << 1) | stream_bit(words, position + bit);
    }
    return value;
}

void build_wave_indexes(
    std::uint32_t tile_rows,
    std::uint32_t tile_columns,
    std::vector<std::uint16_t> &indexes) {
    indexes.assign(static_cast<std::size_t>(tile_rows) * tile_columns, 0);
    std::uint16_t wave = 0;
    for (std::uint32_t index = 0; index < tile_rows; ++index, ++wave) {
        std::int32_t row = static_cast<std::int32_t>(tile_rows - index - 1);
        std::int32_t column = static_cast<std::int32_t>(tile_columns - 1);
        while (row < static_cast<std::int32_t>(tile_rows) && column >= 0) {
            indexes[static_cast<std::size_t>(row) * tile_columns + column] = wave;
            ++row;
            --column;
        }
    }
    for (std::uint32_t index = 0; index < tile_columns; ++index, ++wave) {
        std::int32_t row = 0;
        std::int32_t column = static_cast<std::int32_t>(tile_columns - index - 1);
        while (row < static_cast<std::int32_t>(tile_rows) && column >= 0) {
            indexes[static_cast<std::size_t>(row) * tile_columns + column] = wave;
            ++row;
            --column;
        }
    }
}

float lattice_value(
    std::span<const float> tlut,
    std::uint32_t state,
    std::uint32_t component) {
    const std::uint64_t product =
        static_cast<std::uint64_t>(state) * (static_cast<std::uint64_t>(state) + 1);
    const auto row = static_cast<std::uint32_t>(product) & 0x7FFFU;
    float value = tlut[static_cast<std::size_t>(row) * values_per_state + component];
    if (component == 0 && ((product >> 15) & 1U) != 0) {
        value = -value;
    }
    return f16_to_f32(f32_to_f16(value));
}

float ne_lattice_value(
    std::span<const float> tlut,
    std::uint32_t state,
    std::uint32_t component) {
    const std::uint64_t product =
        static_cast<std::uint64_t>(state) * (static_cast<std::uint64_t>(state) + 1);
    const auto row = static_cast<std::uint32_t>((product >> 6) & 0x1FFU);
    float value = tlut[static_cast<std::size_t>(row) * 2 + component];
    if (component == 0 && ((product >> 15) & 1U) != 0) {
        value = -value;
    }
    return f16_to_f32(f32_to_f16(value));
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
    hadamard(scratch.input);
    build_wave_indexes(tile_rows, tile_columns, scratch.wave_indexes);

    for (std::uint32_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
        float row_sums[tile_size] = {};
        for (std::uint32_t tile_column = 0; tile_column < tile_columns; ++tile_column) {
            float partial[tile_size] = {};
            const auto tile_index =
                static_cast<std::size_t>(tile_row) * tile_columns + tile_column;
            const auto *words = matrix.trellis.data() + tile_index * words_per_tile;
            std::uint32_t state = words[0];
            for (std::uint32_t state_index = 0; state_index < states_per_tile;
                 ++state_index) {
                if (state_index != 0) {
                    const auto position = register_bits + (state_index - 1) * fresh_bits;
                    state = ((state << fresh_bits) & 0xFFFFU) |
                            fresh_value(words, position);
                }
                for (std::uint32_t component = 0; component < values_per_state;
                     ++component) {
                    const auto element = state_index * values_per_state + component;
                    const auto local_row = element / tile_size;
                    const auto local_column = element % tile_size;
                    const auto column = tile_column * tile_size + local_column;
                    partial[local_row] +=
                        lattice_value(matrix.tlut, state, component) *
                        scratch.input[column];
                }
            }
            const float gamma = f16_to_f32(
                matrix.wave_gamma_f16[scratch.wave_indexes[tile_index]]);
            for (std::uint32_t row = 0; row < tile_size; ++row) {
                row_sums[row] += partial[row] * gamma;
            }
        }
        std::copy_n(row_sums, tile_size, scratch.output.begin() + tile_row * tile_size);
    }

    hadamard(scratch.output);
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] = scratch.output[index] * f16_to_f32(matrix.sv_f16[index]);
    }
}

void mach_ne_matvec(
    const MachNeMatrix &matrix,
    std::span<const float> input,
    std::span<float> output,
    ExpertScratch &scratch) {
    constexpr std::uint32_t ne_values_per_state = 2;
    constexpr std::uint32_t ne_fresh_bits = 8;
    constexpr std::uint32_t ne_states_per_tile = tile_values / ne_values_per_state;
    constexpr std::uint32_t ne_stream_bits = 1024;
    constexpr std::uint32_t ne_words_per_tile = ne_stream_bits / 16;

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
    hadamard(scratch.input);

    for (std::uint32_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
        float row_sums[tile_size] = {};
        for (std::uint32_t tile_column = 0; tile_column < tile_columns; ++tile_column) {
            const auto tile_index =
                static_cast<std::size_t>(tile_row) * tile_columns + tile_column;
            const auto *words =
                matrix.trellis.data() + tile_index * ne_words_per_tile;
            std::uint32_t state = words[0];
            for (std::uint32_t state_index = 0; state_index < ne_states_per_tile;
                 ++state_index) {
                if (state_index != 0) {
                    const auto position =
                        register_bits + (state_index - 1) * ne_fresh_bits;
                    std::uint32_t fresh = 0;
                    for (std::uint32_t bit = 0; bit < ne_fresh_bits; ++bit) {
                        auto stream_position = position + bit;
                        if (stream_position >= ne_stream_bits) {
                            stream_position -= ne_stream_bits;
                        }
                        fresh = (fresh << 1) |
                                ((words[stream_position / 16] >>
                                  (15 - stream_position % 16)) & 1U);
                    }
                    state = ((state << ne_fresh_bits) & 0xFFFFU) | fresh;
                }
                for (std::uint32_t component = 0; component < ne_values_per_state;
                     ++component) {
                    const auto element =
                        state_index * ne_values_per_state + component;
                    const auto local_row = element / tile_size;
                    const auto local_column = element % tile_size;
                    const auto column = tile_column * tile_size + local_column;
                    row_sums[local_row] +=
                        ne_lattice_value(matrix.tlut, state, component) *
                        matrix.weight_scale * scratch.input[column];
                }
            }
        }
        std::copy_n(row_sums, tile_size, scratch.output.begin() + tile_row * tile_size);
    }
    hadamard(scratch.output);
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] = scratch.output[index] * static_cast<float>(matrix.sv[index]);
    }
}

void mach_embedding_row(
    const MachEmbedding &embedding,
    std::uint32_t token,
    std::span<float> output) {
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
    for (std::uint32_t row = 0; row < head.rows; ++row) {
        const auto packed_row =
            head.packed.subspan(static_cast<std::size_t>(row) * bytes_per_row,
                                bytes_per_row);
        float sum = 0.0F;
        for (std::uint32_t block = 0; block < head.columns / 8; ++block) {
            std::uint64_t word = 0;
            for (std::uint32_t byte = 0; byte < 5; ++byte) {
                word |= static_cast<std::uint64_t>(packed_row[block * 5 + byte])
                        << (byte * 8);
            }
            for (std::uint32_t index = 0; index < 8; ++index) {
                const auto column = block * 8 + index;
                const auto code = static_cast<std::int32_t>(
                    (word >> (index * 5)) & 0x1FU) - 16;
                const float scale = f16_to_f32(
                    head.group_scale_f16[
                        static_cast<std::size_t>(row) * groups_per_row +
                        column / group]);
                sum += static_cast<float>(code) * scale * input[column];
            }
        }
        output[row] = sum;
    }

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
        float sum = 0.0F;
        for (std::uint32_t column = 0; column < head.columns; ++column) {
            sum += bf16_to_f32(
                       head.protected_bf16[protected_index * head.columns + column]) *
                   input[column];
        }
        output[row] = sum;
    }
}

void bf16_matvec(
    const Bf16Matrix &matrix,
    std::span<const float> input,
    std::span<float> output) {
    if (input.size() != matrix.columns || output.size() != matrix.rows ||
        matrix.values.size() !=
            static_cast<std::size_t>(matrix.rows) * matrix.columns) {
        throw std::invalid_argument("BF16 matrix-vector shape mismatch");
    }
    for (std::uint32_t row = 0; row < matrix.rows; ++row) {
        float sum = 0.0F;
        const auto offset = static_cast<std::size_t>(row) * matrix.columns;
        for (std::uint32_t column = 0; column < matrix.columns; ++column) {
            sum += bf16_to_f32(matrix.values[offset + column]) * input[column];
        }
        output[row] = sum;
    }
}

void rms_norm(
    std::span<const float> input,
    std::span<const std::uint16_t> weight_bf16,
    float epsilon,
    std::span<float> output) {
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
        output[index] = input[index] * scale * bf16_to_f32(weight_bf16[index]);
    }
}

float silu(float value) noexcept {
    return value / (1.0F + std::exp(-value));
}

float sigmoid(float value) noexcept {
    return 1.0F / (1.0F + std::exp(-value));
}

} // namespace adi
