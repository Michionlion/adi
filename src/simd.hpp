#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace adi::detail {

enum class CpuIsa : std::uint32_t {
    scalar,
    avx2,
    avx512,
    neon,
    sve,
};

[[nodiscard]] CpuIsa selected_cpu_isa() noexcept;
[[nodiscard]] std::string_view cpu_isa_name(CpuIsa isa) noexcept;
void force_cpu_isa_for_testing(CpuIsa isa) noexcept;
void clear_cpu_isa_for_testing() noexcept;

void hadamard_transform(std::span<float> values);

[[nodiscard]] float int5_scaled_dot(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales_f16,
    std::span<const float> input);

void int5_scaled_dot_batch(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales_f16,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> weight_scratch);

[[nodiscard]] float bf16_dot(
    std::span<const std::uint16_t> weights,
    std::span<const float> input);

[[nodiscard]] float f32_dot(
    std::span<const float> left,
    std::span<const float> right);

[[nodiscard]] float gated_delta_update(
    std::span<float> state,
    std::span<const float> query,
    std::span<const float> key,
    float value,
    float beta,
    float decay);

} // namespace adi::detail
