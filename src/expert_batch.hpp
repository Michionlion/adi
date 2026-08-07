#pragma once

#include "adi/kernels.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace adi::detail {

// Batch items the SIMD expert kernel handles per SIMD operation under the
// selected ISA, or zero when that ISA has no batch kernel.
[[nodiscard]] std::uint32_t expert_batch_lanes() noexcept;

// Smallest batch at which the SIMD kernel is used. Below it the packing and
// per-tile setup cost more than they save, and the scalar loop wins. This is
// lower than the non-expert kernel's threshold because a tile here holds 32
// states rather than 128, so there is less per-state work to pay for before a
// lane starts earning.
constexpr std::uint32_t expert_batch_minimum = 2;

// Accumulates every packed trellis tile across the batch dimension.
//
//   inputs   [batch, matrix.columns], already scaled and Hadamard'd
//   outputs  [batch, matrix.rows], before the output Hadamard and scales
//   packed   scratch of batch_packed_floats() floats
//
// wave_indexes and wave_gamma are the caller's resolved caches, one index per
// tile and one gamma per wave.
//
// Vectorization is over the batch dimension only. Every lane performs the same
// scalar accumulation sequence, in the same order, as the scalar reference, so
// the results are bit-identical.
void expert_matmul_tiles_batch(
    const MachExpertMatrix &matrix,
    std::span<const float> state_values,
    std::span<const std::uint16_t> wave_indexes,
    std::span<const float> wave_gamma,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> packed);

// ISA entry points. Never called unless the running CPU supports the ISA.
void x86_expert_tiles_batch_avx2(
    const MachExpertMatrix &matrix,
    std::span<const float> state_values,
    std::span<const std::uint16_t> wave_indexes,
    std::span<const float> wave_gamma,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> packed);

void x86_expert_tiles_batch_avx512(
    const MachExpertMatrix &matrix,
    std::span<const float> state_values,
    std::span<const std::uint16_t> wave_indexes,
    std::span<const float> wave_gamma,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> packed);

} // namespace adi::detail
