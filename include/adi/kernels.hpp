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
    std::span<const float> state_values{};
    std::span<const std::uint16_t> wave_indexes{};
    std::span<const float> wave_gamma{};
};

struct ExpertScratch {
    std::vector<float> input;
    std::vector<float> output;
    // Batch-major inputs rearranged as [block][column][lane] for whichever
    // batch-oriented kernel this call uses. A call is either an expert or a
    // non-expert matmul, never both, so the two share the buffer.
    std::vector<float> batch_packed;
    std::vector<std::uint16_t> wave_indexes;
    std::vector<float> state_values;
    std::vector<float> ne_signed_tlut;
    std::vector<float> wave_gamma;
    const float *state_values_source = nullptr;
    const float *ne_signed_tlut_source = nullptr;
    const std::uint16_t *wave_gamma_source = nullptr;
    std::size_t wave_gamma_count = 0;
    std::uint32_t state_value_components = 0;
    std::uint32_t wave_tile_rows = 0;
    std::uint32_t wave_tile_columns = 0;
};

struct MachNeMatrix {
    std::uint32_t rows;
    std::uint32_t columns;
    std::span<const std::uint16_t> trellis;
    std::span<const std::int8_t> su;
    std::span<const std::int8_t> sv;
    float weight_scale;
    std::span<const float> tlut;
    std::span<const float> state_values{};
    std::span<const float> signed_tlut{};
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

struct Bf16Matrix {
    std::uint32_t rows;
    std::uint32_t columns;
    std::span<const std::uint16_t> values;
};

// Computes y = W*x directly from the Mach-1 K=1.5/V=8 additive stream.
// No dense copy of W is constructed.
void mach_expert_matvec(
    const MachExpertMatrix &matrix,
    std::span<const float> input,
    std::span<float> output,
    ExpertScratch &scratch);

void mach_expert_matmul(
    const MachExpertMatrix &matrix,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    ExpertScratch &scratch);

// Computes y = W*x from the canonical K=4/V=2 non-expert stream.
void mach_ne_matvec(
    const MachNeMatrix &matrix,
    std::span<const float> input,
    std::span<float> output,
    ExpertScratch &scratch);

// Inputs and outputs are contiguous batch-major matrices with shapes
// [batch, matrix.columns] and [batch, matrix.rows].
void mach_ne_matmul(
    const MachNeMatrix &matrix,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    ExpertScratch &scratch);

void mach_embedding_row(
    const MachEmbedding &embedding,
    std::uint32_t token,
    std::span<float> output);

void mach_head_matvec(
    const MachHeadChunk &head,
    std::span<const float> input,
    std::span<float> output);

// Inputs and outputs are contiguous batch-major matrices with shapes
// [batch, head.columns] and [batch, head.rows].
void mach_head_matmul(
    const MachHeadChunk &head,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs);

void bf16_matvec(
    const Bf16Matrix &matrix,
    std::span<const float> input,
    std::span<float> output);

void bf16_matmul(
    const Bf16Matrix &matrix,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs);

void rms_norm(
    std::span<const float> input,
    std::span<const std::uint16_t> weight_bf16,
    float weight_offset,
    float epsilon,
    std::span<float> output);

void l2_normalize(std::span<float> values, float epsilon);

[[nodiscard]] float silu(float value) noexcept;
[[nodiscard]] float sigmoid(float value) noexcept;

[[nodiscard]] float f16_to_f32(std::uint16_t bits) noexcept;
[[nodiscard]] std::uint16_t f32_to_f16(float value) noexcept;
[[nodiscard]] float bf16_to_f32(std::uint16_t bits) noexcept;

} // namespace adi
