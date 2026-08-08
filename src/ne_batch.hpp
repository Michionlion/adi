#pragma once

#include "adi/kernels.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace adi::detail {

// Batch items the SIMD non-expert kernel handles per SIMD operation under the
// selected ISA, or zero when that ISA has no batch kernel.
[[nodiscard]] std::uint32_t ne_batch_lanes() noexcept;

// Smallest batch at which the SIMD kernel is used. Batch 1 never reaches it:
// mach_ne_matmul sends a single vector to mach_ne_matvec before this is
// consulted.
//
// This was four, on the assumption that the packing and per-tile setup had to
// be earned back before the kernel could win. They do not: a batch of two or
// three occupies one SIMD block just as a batch of sixteen does, so it costs
// what one pass over the tiles costs and nothing more. Measured serially on
// the 1408x2048 shared-expert gate, one pass is 0.93 ms against the scalar
// loop's 1.60 ms at batch two and 1.77 ms at batch three; with eight workers
// it is 0.22 ms against 0.39 and 0.48.
constexpr std::uint32_t ne_batch_minimum = 2;

using NeMatvecRowsKernel = void (*)(
    const MachNeMatrix &matrix,
    std::span<const float> signed_tlut,
    std::span<const float> input,
    std::span<float> output,
    std::uint32_t row_begin,
    std::uint32_t row_end);

[[nodiscard]] NeMatvecRowsKernel selected_ne_matvec_rows_kernel() noexcept;

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

void x86_ne_matvec_rows_avx2(
    const MachNeMatrix &matrix,
    std::span<const float> signed_tlut,
    std::span<const float> input,
    std::span<float> output,
    std::uint32_t row_begin,
    std::uint32_t row_end);

void x86_ne_tiles_batch_avx512(
    const MachNeMatrix &matrix,
    std::span<const float> state_values,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> packed);

void x86_ne_matvec_rows_avx512(
    const MachNeMatrix &matrix,
    std::span<const float> signed_tlut,
    std::span<const float> input,
    std::span<float> output,
    std::uint32_t row_begin,
    std::uint32_t row_end);

} // namespace adi::detail
