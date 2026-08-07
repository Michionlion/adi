#pragma once

#include "adi/kernels.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace adi::detail {

// Batch items the SIMD non-expert kernel handles per SIMD operation under the
// selected ISA, or zero when that ISA has no batch kernel.
[[nodiscard]] std::uint32_t ne_batch_lanes() noexcept;

// Smallest batch at which the SIMD kernel is used. Below it the packing and
// per-tile setup cost more than they save, and the scalar loop wins.
constexpr std::uint32_t ne_batch_minimum = 4;

// Accumulates every packed trellis tile across the batch dimension.
//
//   inputs   [batch, matrix.columns], already sign-applied and Hadamard'd
//   outputs  [batch, matrix.rows], before the output Hadamard and signs
//   packed   scratch of batch_packed_floats() floats
//
// Vectorization is over the batch dimension only. Every lane performs the
// same scalar accumulation sequence, in the same order, as mach_ne_matmul's
// scalar loop, so the results are bit-identical.
void ne_matmul_tiles_batch(
    const MachNeMatrix &matrix,
    std::span<const float> state_values,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> packed);

// ISA entry points. Never called unless the running CPU supports the ISA.
void x86_ne_tiles_batch_avx2(
    const MachNeMatrix &matrix,
    std::span<const float> state_values,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> packed);

void x86_ne_tiles_batch_avx512(
    const MachNeMatrix &matrix,
    std::span<const float> state_values,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> packed);

} // namespace adi::detail
