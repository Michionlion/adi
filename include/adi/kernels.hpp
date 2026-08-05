#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace adi {

struct MachExpertMatrix {
    std::uint32_t rows;
    std::uint32_t columns;
    std::span<const std::uint16_t> trellis;
    std::span<const std::uint16_t> su_f16;
    std::span<const std::uint16_t> sv_f16;
    std::span<const std::uint16_t> wave_gamma_f16;
    std::span<const float> tlut;
};

struct ExpertScratch {
    std::vector<float> input;
    std::vector<float> output;
    std::vector<std::uint16_t> wave_indexes;
};

// Computes y = W*x directly from the Mach-1 K=1.5/V=8 additive stream.
// No dense copy of W is constructed.
void mach_expert_matvec(
    const MachExpertMatrix &matrix,
    std::span<const float> input,
    std::span<float> output,
    ExpertScratch &scratch);

[[nodiscard]] float f16_to_f32(std::uint16_t bits) noexcept;
[[nodiscard]] std::uint16_t f32_to_f16(float value) noexcept;

} // namespace adi
