#include "codec_cache.hpp"

#include "adi/kernels.hpp"

#include <cstddef>
#include <stdexcept>

namespace adi::detail {
namespace {

struct StateMetadata {
    std::uint16_t expert_row;
    std::uint16_t ne_row;
    bool negative;
};

StateMetadata state_metadata(std::uint32_t state) noexcept {
    const std::uint64_t product =
        static_cast<std::uint64_t>(state) * (state + 1);
    return {
        static_cast<std::uint16_t>(product & 0x7FFFU),
        static_cast<std::uint16_t>((product >> 6) & 0x1FFU),
        ((product >> 15) & 1U) != 0,
    };
}

float rounded_lattice_value(float value) noexcept {
    return f16_to_f32(f32_to_f16(value));
}

} // namespace

std::vector<float> build_expert_state_values(std::span<const float> tlut) {
    if (tlut.size() != 32768 * expert_values_per_state) {
        throw std::invalid_argument("expert TLUT shape mismatch");
    }
    std::vector<float> values(
        static_cast<std::size_t>(codec_state_count) *
        expert_values_per_state);
    for (std::uint32_t state = 0; state < codec_state_count; ++state) {
        const auto metadata = state_metadata(state);
        for (std::uint32_t component = 0;
             component < expert_values_per_state;
             ++component) {
            float value =
                tlut[static_cast<std::size_t>(metadata.expert_row) *
                         expert_values_per_state +
                     component];
            if (component == 0 && metadata.negative) {
                value = -value;
            }
            values[static_cast<std::size_t>(state) *
                       expert_values_per_state +
                   component] = rounded_lattice_value(value);
        }
    }
    return values;
}

std::vector<float> build_ne_state_values(std::span<const float> tlut) {
    if (tlut.size() != 512 * ne_values_per_state) {
        throw std::invalid_argument("non-expert TLUT shape mismatch");
    }
    std::vector<float> values(
        static_cast<std::size_t>(codec_state_count) * ne_values_per_state);
    for (std::uint32_t state = 0; state < codec_state_count; ++state) {
        const auto metadata = state_metadata(state);
        for (std::uint32_t component = 0;
             component < ne_values_per_state;
             ++component) {
            float value =
                tlut[static_cast<std::size_t>(metadata.ne_row) *
                         ne_values_per_state +
                     component];
            if (component == 0 && metadata.negative) {
                value = -value;
            }
            values[static_cast<std::size_t>(state) * ne_values_per_state +
                   component] = rounded_lattice_value(value);
        }
    }
    return values;
}

std::vector<float> build_ne_signed_tlut(std::span<const float> tlut) {
    if (tlut.size() != 512 * ne_values_per_state) {
        throw std::invalid_argument("non-expert TLUT shape mismatch");
    }
    std::vector<float> values(1024 * ne_values_per_state);
    // Bits 6..14 of state*(state+1) select the 512-row lattice. Bit 15
    // negates only the first component, so retaining that bit in the row
    // index makes the arithmetic decoder a single small-table gather.
    for (std::uint32_t row = 0; row < 512; ++row) {
        for (std::uint32_t negative = 0; negative < 2; ++negative) {
            const auto signed_row = row | (negative << 9);
            for (std::uint32_t component = 0;
                 component < ne_values_per_state;
                 ++component) {
                // Match the released decoder's f16 round trip before adding
                // the sign, exactly as build_ne_state_values does.
                float value = rounded_lattice_value(
                    tlut[static_cast<std::size_t>(row) *
                             ne_values_per_state +
                         component]);
                if (component == 0 && negative != 0) {
                    value = -value;
                }
                values[static_cast<std::size_t>(signed_row) *
                           ne_values_per_state +
                       component] = value;
            }
        }
    }
    return values;
}

std::vector<std::uint16_t> build_wave_indexes(
    std::uint32_t tile_rows,
    std::uint32_t tile_columns) {
    if (tile_rows == 0 || tile_columns == 0 ||
        tile_rows + tile_columns > 65536U) {
        throw std::invalid_argument("wave-index geometry is invalid");
    }
    std::vector<std::uint16_t> indexes(
        static_cast<std::size_t>(tile_rows) * tile_columns);
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
        std::int32_t column =
            static_cast<std::int32_t>(tile_columns - index - 1);
        while (row < static_cast<std::int32_t>(tile_rows) && column >= 0) {
            indexes[static_cast<std::size_t>(row) * tile_columns + column] = wave;
            ++row;
            --column;
        }
    }
    return indexes;
}

std::vector<float> convert_f16_values(
    std::span<const std::uint16_t> values) {
    std::vector<float> converted(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        converted[index] = f16_to_f32(values[index]);
    }
    return converted;
}

} // namespace adi::detail
