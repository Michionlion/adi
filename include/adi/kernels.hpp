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

struct MachNeMatrix {
    std::uint32_t rows;
    std::uint32_t columns;
    std::span<const std::uint16_t> trellis;
    std::span<const std::int8_t> su;
    std::span<const std::int8_t> sv;
    float weight_scale;
    std::span<const float> tlut;
};

struct MachEmbedding {
    std::uint32_t rows;
    std::uint32_t columns;
    std::span<const std::uint8_t> packed;
    std::span<const std::uint16_t> minimum_f16;
    std::span<const std::uint16_t> maximum_f16;
    std::span<const std::uint32_t> exception_indexes;
    std::span<const std::uint16_t> exception_bf16;
};

struct MachHeadChunk {
    std::uint32_t rows;
    std::uint32_t columns;
    std::span<const std::uint8_t> packed;
    std::span<const std::uint16_t> group_scale_f16;
    std::span<const std::uint32_t> protected_rows;
    std::span<const std::uint16_t> protected_bf16;
};

// Computes y = W*x directly from the Mach-1 K=1.5/V=8 additive stream.
// No dense copy of W is constructed.
void mach_expert_matvec(
    const MachExpertMatrix &matrix,
    std::span<const float> input,
    std::span<float> output,
    ExpertScratch &scratch);

// Computes y = W*x from the canonical K=4/V=2 non-expert stream.
void mach_ne_matvec(
    const MachNeMatrix &matrix,
    std::span<const float> input,
    std::span<float> output,
    ExpertScratch &scratch);

void mach_embedding_row(
    const MachEmbedding &embedding,
    std::uint32_t token,
    std::span<float> output);

void mach_head_matvec(
    const MachHeadChunk &head,
    std::span<const float> input,
    std::span<float> output);

[[nodiscard]] float f16_to_f32(std::uint16_t bits) noexcept;
[[nodiscard]] std::uint16_t f32_to_f16(float value) noexcept;
[[nodiscard]] float bf16_to_f32(std::uint16_t bits) noexcept;

} // namespace adi
