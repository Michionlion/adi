#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace adi::detail {

constexpr std::uint32_t codec_state_count = 1U << 16;
constexpr std::uint32_t expert_values_per_state = 8;
constexpr std::uint32_t ne_values_per_state = 2;

[[nodiscard]] std::vector<float> build_expert_state_values(
    std::span<const float> tlut);

[[nodiscard]] std::vector<float> build_ne_state_values(
    std::span<const float> tlut);

[[nodiscard]] std::vector<std::uint16_t> build_wave_indexes(
    std::uint32_t tile_rows,
    std::uint32_t tile_columns);

[[nodiscard]] std::vector<float> convert_f16_values(
    std::span<const std::uint16_t> values);

} // namespace adi::detail
