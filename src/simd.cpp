#include "simd.hpp"

#include "adi/kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace adi::detail {
[[nodiscard]] CpuIsa x86_detect_isa() noexcept;
[[nodiscard]] CpuIsa arm_detect_isa() noexcept;

void x86_hadamard(std::span<float> values, CpuIsa isa);
float x86_int5_scaled_dot(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input,
    CpuIsa isa);
float x86_bf16_dot(
    std::span<const std::uint16_t> weights,
    std::span<const float> input,
    CpuIsa isa);
float x86_f32_dot(
    std::span<const float> left,
    std::span<const float> right,
    CpuIsa isa);

void arm_hadamard(std::span<float> values, CpuIsa isa);
float arm_int5_scaled_dot(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input,
    CpuIsa isa);
float arm_bf16_dot(
    std::span<const std::uint16_t> weights,
    std::span<const float> input,
    CpuIsa isa);
float arm_f32_dot(
    std::span<const float> left,
    std::span<const float> right,
    CpuIsa isa);

namespace {

constexpr int no_override = -1;
std::atomic<int> isa_override{no_override};

CpuIsa detect_isa() noexcept {
    if (const char *requested = std::getenv("ADI_CPU_ISA"); requested != nullptr) {
        const std::string_view value(requested);
        if (value == "scalar") {
            return CpuIsa::scalar;
        }
        if (value == "avx2") {
            return x86_detect_isa() == CpuIsa::scalar
                       ? CpuIsa::scalar
                       : CpuIsa::avx2;
        }
        if (value == "avx512") {
            return x86_detect_isa() == CpuIsa::avx512
                       ? CpuIsa::avx512
                       : CpuIsa::scalar;
        }
        if (value == "neon") {
            const auto arm = arm_detect_isa();
            return arm == CpuIsa::neon || arm == CpuIsa::sve
                       ? CpuIsa::neon
                       : CpuIsa::scalar;
        }
        if (value == "sve") {
            return arm_detect_isa() == CpuIsa::sve
                       ? CpuIsa::sve
                       : CpuIsa::scalar;
        }
    }
    const auto x86 = x86_detect_isa();
    return x86 != CpuIsa::scalar ? x86 : arm_detect_isa();
}

void scalar_hadamard(std::span<float> values) {
    for (std::size_t stride = 1; stride < values.size(); stride *= 2) {
        for (std::size_t block = 0; block < values.size(); block += 2 * stride) {
            for (std::size_t index = 0; index < stride; ++index) {
                const float left = values[block + index];
                const float right = values[block + stride + index];
                values[block + index] = left + right;
                values[block + stride + index] = left - right;
            }
        }
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(values.size()));
    for (auto &value : values) {
        value *= scale;
    }
}

float scalar_int5_scaled_dot(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input) {
    constexpr std::uint32_t group = 64;
    float sum = 0.0F;
    for (std::uint32_t block = 0; block < input.size() / 8; ++block) {
        std::uint64_t word = 0;
        for (std::uint32_t byte = 0; byte < 5; ++byte) {
            word |= static_cast<std::uint64_t>(packed[block * 5 + byte])
                    << (byte * 8);
        }
        for (std::uint32_t index = 0; index < 8; ++index) {
            const auto column = block * 8 + index;
            const auto code = static_cast<std::int32_t>(
                (word >> (index * 5)) & 0x1FU) - 16;
            const float scale = f16_to_f32(scales[column / group]);
            sum += static_cast<float>(code) * scale * input[column];
        }
    }
    return sum;
}

float scalar_bf16_dot(
    std::span<const std::uint16_t> weights,
    std::span<const float> input) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < input.size(); ++index) {
        sum += bf16_to_f32(weights[index]) * input[index];
    }
    return sum;
}

float scalar_f32_dot(
    std::span<const float> left,
    std::span<const float> right) {
    float sum = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        sum += left[index] * right[index];
    }
    return sum;
}

} // namespace

CpuIsa selected_cpu_isa() noexcept {
    const auto forced = isa_override.load(std::memory_order_acquire);
    if (forced != no_override) {
        return static_cast<CpuIsa>(forced);
    }
    static const CpuIsa detected = detect_isa();
    return detected;
}

std::string_view cpu_isa_name(CpuIsa isa) noexcept {
    switch (isa) {
    case CpuIsa::scalar:
        return "scalar";
    case CpuIsa::avx2:
        return "avx2";
    case CpuIsa::avx512:
        return "avx512";
    case CpuIsa::neon:
        return "neon";
    case CpuIsa::sve:
        return "sve";
    }
    return "unknown";
}

void force_cpu_isa_for_testing(CpuIsa isa) noexcept {
    isa_override.store(static_cast<int>(isa), std::memory_order_release);
}

void clear_cpu_isa_for_testing() noexcept {
    isa_override.store(no_override, std::memory_order_release);
}

void hadamard_transform(std::span<float> values) {
    const auto isa = selected_cpu_isa();
    if (isa == CpuIsa::avx2 || isa == CpuIsa::avx512) {
        x86_hadamard(values, isa);
    } else if (isa == CpuIsa::neon || isa == CpuIsa::sve) {
        arm_hadamard(values, isa);
    } else {
        scalar_hadamard(values);
    }
}

float int5_scaled_dot(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input) {
    constexpr std::size_t group = 64;
    constexpr std::size_t bytes_per_group = 40;
    if (scales.empty() ||
        packed.size() != scales.size() * bytes_per_group ||
        input.size() != scales.size() * group) {
        throw std::invalid_argument("int5 dot shape mismatch");
    }
    const auto isa = selected_cpu_isa();
    if (isa == CpuIsa::avx2 || isa == CpuIsa::avx512) {
        return x86_int5_scaled_dot(packed, scales, input, isa);
    }
    if (isa == CpuIsa::neon || isa == CpuIsa::sve) {
        return arm_int5_scaled_dot(packed, scales, input, isa);
    }
    return scalar_int5_scaled_dot(packed, scales, input);
}

void int5_scaled_dot_batch(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> weight_scratch) {
    constexpr std::size_t group = 64;
    constexpr std::size_t bytes_per_group = 40;
    const auto columns = scales.size() * group;
    if (batch == 0 || scales.empty() ||
        packed.size() != scales.size() * bytes_per_group ||
        inputs.size() != static_cast<std::size_t>(batch) * columns ||
        outputs.size() != batch || weight_scratch.size() != columns) {
        throw std::invalid_argument("int5 batch dot shape mismatch");
    }

    for (std::size_t scale_index = 0;
         scale_index < scales.size();
         ++scale_index) {
        const float scale = f16_to_f32(scales[scale_index]);
        for (std::size_t block = 0; block < group / 8; ++block) {
            std::uint64_t word = 0;
            for (std::size_t byte = 0; byte < 5; ++byte) {
                word |= static_cast<std::uint64_t>(
                            packed[scale_index * bytes_per_group +
                                   block * 5 + byte])
                        << (byte * 8);
            }
            for (std::size_t index = 0; index < 8; ++index) {
                weight_scratch[scale_index * group + block * 8 + index] =
                    static_cast<float>(
                        static_cast<std::int32_t>(
                            (word >> (index * 5)) & 0x1FU) -
                        16) *
                    scale;
            }
        }
    }
    const auto isa = selected_cpu_isa();
    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        const auto input = inputs.subspan(
            static_cast<std::size_t>(batch_index) * columns,
            columns);
        if (isa == CpuIsa::avx2 || isa == CpuIsa::avx512) {
            outputs[batch_index] = x86_f32_dot(weight_scratch, input, isa);
        } else if (isa == CpuIsa::neon || isa == CpuIsa::sve) {
            float result = 0.0F;
            for (std::size_t group_index = 0;
                 group_index < scales.size();
                 ++group_index) {
                result += arm_f32_dot(
                    weight_scratch.subspan(group_index * group, group),
                    input.subspan(group_index * group, group),
                    isa);
            }
            outputs[batch_index] = result;
        } else {
            outputs[batch_index] = scalar_f32_dot(weight_scratch, input);
        }
    }
}

float bf16_dot(
    std::span<const std::uint16_t> weights,
    std::span<const float> input) {
    if (weights.size() != input.size()) {
        throw std::invalid_argument("BF16 dot shape mismatch");
    }
    const auto isa = selected_cpu_isa();
    if (isa == CpuIsa::avx2 || isa == CpuIsa::avx512) {
        return x86_bf16_dot(weights, input, isa);
    }
    if (isa == CpuIsa::neon || isa == CpuIsa::sve) {
        return arm_bf16_dot(weights, input, isa);
    }
    return scalar_bf16_dot(weights, input);
}

float f32_dot(
    std::span<const float> left,
    std::span<const float> right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("float dot shape mismatch");
    }
    const auto isa = selected_cpu_isa();
    if (isa == CpuIsa::avx2 || isa == CpuIsa::avx512) {
        return x86_f32_dot(left, right, isa);
    }
    if (isa == CpuIsa::neon || isa == CpuIsa::sve) {
        return arm_f32_dot(left, right, isa);
    }
    return scalar_f32_dot(left, right);
}

} // namespace adi::detail
