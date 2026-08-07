#pragma once

#include "simd.hpp"

#include <cstdint>
#include <span>

namespace adi::detail {

using HadamardKernel = void (*)(std::span<float> values);
using Int5DotKernel = float (*)(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input);
using Int5BatchDotKernel = void (*)(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> scratch);
using Bf16DotKernel = float (*)(
    std::span<const std::uint16_t> weights,
    std::span<const float> input);
using GatedDeltaKernel = float (*)(
    std::span<float> state,
    std::span<const float> query,
    std::span<const float> key,
    float value,
    float beta,
    float decay);

struct X86Kernels {
    HadamardKernel hadamard = nullptr;
    Int5DotKernel int5_dot = nullptr;
    Bf16DotKernel bf16_dot = nullptr;
    F32DotKernel f32_dot = nullptr;
    GatedDeltaKernel gated_delta = nullptr;
};

[[nodiscard]] CpuFeatures x86_detect_features() noexcept;
[[nodiscard]] const X86Kernels &x86_avx2_kernels() noexcept;
[[nodiscard]] const X86Kernels &x86_avx512_kernels() noexcept;
[[nodiscard]] float x86_int5_dot_vbmi(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input);
void x86_int5_dot_batch_vbmi(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> scratch);

} // namespace adi::detail
