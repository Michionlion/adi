#include "simd.hpp"

#include "adi/kernels.hpp"

#include <cmath>
#include <cstdint>
#include <span>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif

namespace adi::detail {

CpuIsa arm_detect_isa() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__linux__) && defined(HWCAP_SVE)
    if ((getauxval(AT_HWCAP) & HWCAP_SVE) != 0) {
        return CpuIsa::sve;
    }
#endif
    return CpuIsa::neon;
#else
    return CpuIsa::scalar;
#endif
}

void arm_hadamard(std::span<float> values, CpuIsa) {
#if defined(__aarch64__) || defined(_M_ARM64)
    for (std::size_t stride = 1; stride < values.size(); stride *= 2) {
        for (std::size_t block = 0; block < values.size(); block += 2 * stride) {
            std::size_t index = 0;
            if (stride >= 4) {
                for (; index + 4 <= stride; index += 4) {
                    const auto left = vld1q_f32(values.data() + block + index);
                    const auto right =
                        vld1q_f32(values.data() + block + stride + index);
                    vst1q_f32(
                        values.data() + block + index,
                        vaddq_f32(left, right));
                    vst1q_f32(
                        values.data() + block + stride + index,
                        vsubq_f32(left, right));
                }
            }
            for (; index < stride; ++index) {
                const float left = values[block + index];
                const float right = values[block + stride + index];
                values[block + index] = left + right;
                values[block + stride + index] = left - right;
            }
        }
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(values.size()));
    const auto scale_vector = vdupq_n_f32(scale);
    std::size_t index = 0;
    for (; index + 4 <= values.size(); index += 4) {
        vst1q_f32(
            values.data() + index,
            vmulq_f32(vld1q_f32(values.data() + index), scale_vector));
    }
    for (; index < values.size(); ++index) {
        values[index] *= scale;
    }
#else
    (void)values;
#endif
}

float arm_f32_dot(
    std::span<const float> left,
    std::span<const float> right,
    CpuIsa) {
#if defined(__aarch64__) || defined(_M_ARM64)
    auto sum = vdupq_n_f32(0.0F);
    std::size_t index = 0;
    for (; index + 4 <= left.size(); index += 4) {
        sum = vmlaq_f32(
            sum,
            vld1q_f32(left.data() + index),
            vld1q_f32(right.data() + index));
    }
    float result = vaddvq_f32(sum);
    for (; index < left.size(); ++index) {
        result += left[index] * right[index];
    }
    return result;
#else
    (void)left;
    (void)right;
    return 0.0F;
#endif
}

float arm_bf16_dot(
    std::span<const std::uint16_t> weights,
    std::span<const float> input,
    CpuIsa isa) {
#if defined(__aarch64__) || defined(_M_ARM64)
    auto sum = vdupq_n_f32(0.0F);
    std::size_t index = 0;
    alignas(16) float converted[4];
    for (; index + 4 <= weights.size(); index += 4) {
        for (std::size_t lane = 0; lane < 4; ++lane) {
            converted[lane] = bf16_to_f32(weights[index + lane]);
        }
        sum = vmlaq_f32(
            sum,
            vld1q_f32(converted),
            vld1q_f32(input.data() + index));
    }
    float result = vaddvq_f32(sum);
    for (; index < weights.size(); ++index) {
        result += bf16_to_f32(weights[index]) * input[index];
    }
    return result;
#else
    (void)weights;
    (void)input;
    (void)isa;
    return 0.0F;
#endif
}

float arm_int5_scaled_dot(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input,
    CpuIsa isa) {
#if defined(__aarch64__) || defined(_M_ARM64)
    float result = 0.0F;
    alignas(16) float weights[64];
    for (std::size_t group = 0; group < scales.size(); ++group) {
        const float scale = f16_to_f32(scales[group]);
        for (std::uint32_t block = 0; block < 8; ++block) {
            std::uint64_t word = 0;
            for (std::uint32_t byte = 0; byte < 5; ++byte) {
                word |= static_cast<std::uint64_t>(
                            packed[group * 40 + block * 5 + byte])
                        << (byte * 8);
            }
            for (std::uint32_t index = 0; index < 8; ++index) {
                weights[block * 8 + index] =
                    static_cast<float>(
                        static_cast<std::int32_t>(
                            (word >> (index * 5)) & 0x1FU) -
                        16) *
                    scale;
            }
        }
        result += arm_f32_dot(
            std::span<const float>(weights, 64),
            input.subspan(group * 64, 64),
            isa);
    }
    return result;
#else
    (void)packed;
    (void)scales;
    (void)input;
    (void)isa;
    return 0.0F;
#endif
}

} // namespace adi::detail
