#include "simd.hpp"

#include "adi/kernels.hpp"
#include "simd_x86.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace adi::detail {
[[nodiscard]] CpuIsa arm_detect_isa() noexcept;

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

struct SimdOps {
    CpuIsa isa;
    HadamardKernel hadamard;
    Int5DotKernel int5_dot;
    Bf16DotKernel bf16_dot;
    F32DotKernel f32_dot;
    GatedDeltaKernel gated_delta;
};

constexpr int no_override = -1;
std::atomic<int> isa_override{no_override};

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

float scalar_gated_delta_update(
    std::span<float> state,
    std::span<const float> query,
    std::span<const float> key,
    float value,
    float beta,
    float decay) {
    float prediction = 0.0F;
    for (std::size_t index = 0; index < state.size(); ++index) {
        state[index] *= decay;
        prediction += state[index] * key[index];
    }
    const float delta = (value - prediction) * beta;
    float attended = 0.0F;
    for (std::size_t index = 0; index < state.size(); ++index) {
        state[index] += key[index] * delta;
        attended += state[index] * query[index];
    }
    return attended;
}

void neon_hadamard(std::span<float> values) {
    arm_hadamard(values, CpuIsa::neon);
}

void sve_hadamard(std::span<float> values) {
    arm_hadamard(values, CpuIsa::sve);
}

float neon_int5_dot(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input) {
    return arm_int5_scaled_dot(packed, scales, input, CpuIsa::neon);
}

float sve_int5_dot(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input) {
    return arm_int5_scaled_dot(packed, scales, input, CpuIsa::sve);
}

float neon_bf16_dot(
    std::span<const std::uint16_t> weights,
    std::span<const float> input) {
    return arm_bf16_dot(weights, input, CpuIsa::neon);
}

float sve_bf16_dot(
    std::span<const std::uint16_t> weights,
    std::span<const float> input) {
    return arm_bf16_dot(weights, input, CpuIsa::sve);
}

float neon_f32_dot(
    std::span<const float> left,
    std::span<const float> right) {
    return arm_f32_dot(left, right, CpuIsa::neon);
}

float sve_f32_dot(
    std::span<const float> left,
    std::span<const float> right) {
    return arm_f32_dot(left, right, CpuIsa::sve);
}

const SimdOps &scalar_ops() noexcept {
    static const SimdOps operations{
        CpuIsa::scalar,
        scalar_hadamard,
        scalar_int5_scaled_dot,
        scalar_bf16_dot,
        scalar_f32_dot,
        scalar_gated_delta_update,
    };
    return operations;
}

const SimdOps &avx2_ops() noexcept {
    static const SimdOps operations = [] {
        const auto &kernels = x86_avx2_kernels();
        return SimdOps{
            CpuIsa::avx2,
            kernels.hadamard,
            kernels.int5_dot,
            kernels.bf16_dot,
            kernels.f32_dot,
            kernels.gated_delta,
        };
    }();
    return operations;
}

const SimdOps &avx512_ops() noexcept {
    static const SimdOps operations = [] {
        const auto &kernels = x86_avx512_kernels();
        const auto int5_dot = detected_cpu_features().avx512vbmi
                                  ? x86_int5_dot_vbmi
                                  : kernels.int5_dot;
        return SimdOps{
            CpuIsa::avx512,
            kernels.hadamard,
            int5_dot,
            kernels.bf16_dot,
            kernels.f32_dot,
            kernels.gated_delta,
        };
    }();
    return operations;
}

const SimdOps &neon_ops() noexcept {
    static const SimdOps operations{
        CpuIsa::neon,
        neon_hadamard,
        neon_int5_dot,
        neon_bf16_dot,
        neon_f32_dot,
        scalar_gated_delta_update,
    };
    return operations;
}

const SimdOps &sve_ops() noexcept {
    static const SimdOps operations{
        CpuIsa::sve,
        sve_hadamard,
        sve_int5_dot,
        sve_bf16_dot,
        sve_f32_dot,
        scalar_gated_delta_update,
    };
    return operations;
}

CpuFeatures detect_features() noexcept {
    auto features = x86_detect_features();
    const auto arm = arm_detect_isa();
    features.neon = arm == CpuIsa::neon || arm == CpuIsa::sve;
    features.sve = arm == CpuIsa::sve;
    return features;
}

const CpuFeatures &cpu_features() noexcept {
    static const CpuFeatures features = detect_features();
    return features;
}

bool supports(CpuIsa isa, const CpuFeatures &features) noexcept {
    switch (isa) {
    case CpuIsa::scalar:
        return true;
    case CpuIsa::avx2:
        return features.avx2;
    case CpuIsa::avx512:
        return features.avx512f && features.avx512bw;
    case CpuIsa::neon:
        return features.neon;
    case CpuIsa::sve:
        return features.sve;
    }
    return false;
}

const SimdOps &ops_for_isa(CpuIsa isa) noexcept {
    if (!supports(isa, cpu_features())) {
        return scalar_ops();
    }
    switch (isa) {
    case CpuIsa::scalar:
        return scalar_ops();
    case CpuIsa::avx2:
        return avx2_ops();
    case CpuIsa::avx512:
        return avx512_ops();
    case CpuIsa::neon:
        return neon_ops();
    case CpuIsa::sve:
        return sve_ops();
    }
    return scalar_ops();
}

CpuIsa requested_isa() noexcept {
    const auto &features = cpu_features();
    if (const char *requested = std::getenv("ADI_CPU_ISA");
        requested != nullptr) {
        const std::string_view value(requested);
        if (value == "scalar") {
            return CpuIsa::scalar;
        }
        if (value == "avx2") {
            return supports(CpuIsa::avx2, features)
                       ? CpuIsa::avx2
                       : CpuIsa::scalar;
        }
        if (value == "avx512") {
            return supports(CpuIsa::avx512, features)
                       ? CpuIsa::avx512
                       : CpuIsa::scalar;
        }
        if (value == "neon") {
            return supports(CpuIsa::neon, features)
                       ? CpuIsa::neon
                       : CpuIsa::scalar;
        }
        if (value == "sve") {
            return supports(CpuIsa::sve, features)
                       ? CpuIsa::sve
                       : CpuIsa::scalar;
        }
    }
    if (supports(CpuIsa::avx512, features)) {
        return CpuIsa::avx512;
    }
    if (supports(CpuIsa::avx2, features)) {
        return CpuIsa::avx2;
    }
    if (supports(CpuIsa::sve, features)) {
        return CpuIsa::sve;
    }
    if (supports(CpuIsa::neon, features)) {
        return CpuIsa::neon;
    }
    return CpuIsa::scalar;
}

const SimdOps &native_ops() noexcept {
    static const SimdOps *operations = &ops_for_isa(requested_isa());
    return *operations;
}

const SimdOps &selected_ops() noexcept {
    const auto forced = isa_override.load(std::memory_order_acquire);
    if (forced != no_override) {
        return ops_for_isa(static_cast<CpuIsa>(forced));
    }
    return native_ops();
}

} // namespace

CpuFeatures detected_cpu_features() noexcept {
    return cpu_features();
}

CpuIsa selected_cpu_isa() noexcept {
    return selected_ops().isa;
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
    selected_ops().hadamard(values);
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
    return selected_ops().int5_dot(packed, scales, input);
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
    const auto &operations = selected_ops();
    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        outputs[batch_index] = operations.f32_dot(
            weight_scratch,
            inputs.subspan(
                static_cast<std::size_t>(batch_index) * columns,
                columns));
    }
}

float bf16_dot(
    std::span<const std::uint16_t> weights,
    std::span<const float> input) {
    if (weights.size() != input.size()) {
        throw std::invalid_argument("BF16 dot shape mismatch");
    }
    return selected_ops().bf16_dot(weights, input);
}

float f32_dot(
    std::span<const float> left,
    std::span<const float> right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("float dot shape mismatch");
    }
    return selected_ops().f32_dot(left, right);
}

float gated_delta_update(
    std::span<float> state,
    std::span<const float> query,
    std::span<const float> key,
    float value,
    float beta,
    float decay) {
    if (state.empty() || state.size() != query.size() ||
        state.size() != key.size()) {
        throw std::invalid_argument("Gated DeltaNet row shape mismatch");
    }
    return selected_ops().gated_delta(
        state,
        query,
        key,
        value,
        beta,
        decay);
}

} // namespace adi::detail
