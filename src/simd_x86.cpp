#include "simd.hpp"

#include "adi/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace adi::detail {

#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
namespace {

__attribute__((target("avx2"))) float reduce_avx2(__m256 value) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, value);
    float sum = 0.0F;
    for (const float lane : lanes) {
        sum += lane;
    }
    return sum;
}

__attribute__((target("avx512f,avx512bw"))) float reduce_avx512(__m512 value) {
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, value);
    float sum = 0.0F;
    for (const float lane : lanes) {
        sum += lane;
    }
    return sum;
}

__attribute__((target("avx2"))) void hadamard_avx2(std::span<float> values) {
    for (std::size_t stride = 1; stride < values.size(); stride *= 2) {
        for (std::size_t block = 0; block < values.size(); block += 2 * stride) {
            std::size_t index = 0;
            if (stride >= 8) {
                for (; index + 8 <= stride; index += 8) {
                    const auto left =
                        _mm256_loadu_ps(values.data() + block + index);
                    const auto right = _mm256_loadu_ps(
                        values.data() + block + stride + index);
                    _mm256_storeu_ps(
                        values.data() + block + index,
                        _mm256_add_ps(left, right));
                    _mm256_storeu_ps(
                        values.data() + block + stride + index,
                        _mm256_sub_ps(left, right));
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
    const float scalar_scale =
        1.0F / std::sqrt(static_cast<float>(values.size()));
    const auto scale = _mm256_set1_ps(scalar_scale);
    std::size_t index = 0;
    for (; index + 8 <= values.size(); index += 8) {
        _mm256_storeu_ps(
            values.data() + index,
            _mm256_mul_ps(_mm256_loadu_ps(values.data() + index), scale));
    }
    for (; index < values.size(); ++index) {
        values[index] *= scalar_scale;
    }
}

__attribute__((target("avx512f,avx512bw"))) void hadamard_avx512(
    std::span<float> values) {
    for (std::size_t stride = 1; stride < values.size(); stride *= 2) {
        for (std::size_t block = 0; block < values.size(); block += 2 * stride) {
            std::size_t index = 0;
            if (stride >= 16) {
                for (; index + 16 <= stride; index += 16) {
                    const auto left =
                        _mm512_loadu_ps(values.data() + block + index);
                    const auto right = _mm512_loadu_ps(
                        values.data() + block + stride + index);
                    _mm512_storeu_ps(
                        values.data() + block + index,
                        _mm512_add_ps(left, right));
                    _mm512_storeu_ps(
                        values.data() + block + stride + index,
                        _mm512_sub_ps(left, right));
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
    const float scalar_scale =
        1.0F / std::sqrt(static_cast<float>(values.size()));
    const auto scale = _mm512_set1_ps(scalar_scale);
    std::size_t index = 0;
    for (; index + 16 <= values.size(); index += 16) {
        _mm512_storeu_ps(
            values.data() + index,
            _mm512_mul_ps(_mm512_loadu_ps(values.data() + index), scale));
    }
    for (; index < values.size(); ++index) {
        values[index] *= scalar_scale;
    }
}

void unpack_int5_group(
    const std::uint8_t *packed,
    float scale,
    float *weights) {
    for (std::uint32_t block = 0; block < 8; ++block) {
        std::uint64_t word = 0;
        for (std::uint32_t byte = 0; byte < 5; ++byte) {
            word |= static_cast<std::uint64_t>(packed[block * 5 + byte])
                    << (byte * 8);
        }
        for (std::uint32_t index = 0; index < 8; ++index) {
            const auto code = static_cast<std::int32_t>(
                (word >> (index * 5)) & 0x1FU) - 16;
            weights[block * 8 + index] = static_cast<float>(code) * scale;
        }
    }
}

__attribute__((target("avx2"))) float int5_dot_avx2(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input) {
    __m256 sum = _mm256_setzero_ps();
    alignas(32) float weights[64];
    for (std::size_t group = 0; group < scales.size(); ++group) {
        unpack_int5_group(
            packed.data() + group * 40,
            f16_to_f32(scales[group]),
            weights);
        for (std::size_t index = 0; index < 64; index += 8) {
            sum = _mm256_add_ps(
                sum,
                _mm256_mul_ps(
                    _mm256_load_ps(weights + index),
                    _mm256_loadu_ps(input.data() + group * 64 + index)));
        }
    }
    return reduce_avx2(sum);
}

__attribute__((target("avx512f,avx512bw"))) float int5_dot_avx512(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input) {
    __m512 sum = _mm512_setzero_ps();
    alignas(64) float weights[64];
    for (std::size_t group = 0; group < scales.size(); ++group) {
        unpack_int5_group(
            packed.data() + group * 40,
            f16_to_f32(scales[group]),
            weights);
        for (std::size_t index = 0; index < 64; index += 16) {
            sum = _mm512_add_ps(
                sum,
                _mm512_mul_ps(
                    _mm512_load_ps(weights + index),
                    _mm512_loadu_ps(input.data() + group * 64 + index)));
        }
    }
    return reduce_avx512(sum);
}

__attribute__((target("avx2"))) float bf16_dot_avx2(
    std::span<const std::uint16_t> weights,
    std::span<const float> input) {
    __m256 sum = _mm256_setzero_ps();
    std::size_t index = 0;
    for (; index + 8 <= weights.size(); index += 8) {
        const auto packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(weights.data() + index));
        auto expanded = _mm256_cvtepu16_epi32(packed);
        expanded = _mm256_slli_epi32(expanded, 16);
        sum = _mm256_add_ps(
            sum,
            _mm256_mul_ps(
                _mm256_castsi256_ps(expanded),
                _mm256_loadu_ps(input.data() + index)));
    }
    float result = reduce_avx2(sum);
    for (; index < weights.size(); ++index) {
        result += bf16_to_f32(weights[index]) * input[index];
    }
    return result;
}

__attribute__((target("avx512f,avx512bw"))) float bf16_dot_avx512(
    std::span<const std::uint16_t> weights,
    std::span<const float> input) {
    __m512 sum = _mm512_setzero_ps();
    std::size_t index = 0;
    for (; index + 16 <= weights.size(); index += 16) {
        const auto packed =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights.data() + index));
        auto expanded = _mm512_cvtepu16_epi32(packed);
        expanded = _mm512_slli_epi32(expanded, 16);
        sum = _mm512_add_ps(
            sum,
            _mm512_mul_ps(
                _mm512_castsi512_ps(expanded),
                _mm512_loadu_ps(input.data() + index)));
    }
    float result = reduce_avx512(sum);
    for (; index < weights.size(); ++index) {
        result += bf16_to_f32(weights[index]) * input[index];
    }
    return result;
}

__attribute__((target("avx2"))) float f32_dot_avx2(
    std::span<const float> left,
    std::span<const float> right) {
    __m256 sum = _mm256_setzero_ps();
    std::size_t index = 0;
    for (; index + 8 <= left.size(); index += 8) {
        sum = _mm256_add_ps(
            sum,
            _mm256_mul_ps(
                _mm256_loadu_ps(left.data() + index),
                _mm256_loadu_ps(right.data() + index)));
    }
    float result = reduce_avx2(sum);
    for (; index < left.size(); ++index) {
        result += left[index] * right[index];
    }
    return result;
}

__attribute__((target("avx512f"))) float f32_dot_avx512(
    std::span<const float> left,
    std::span<const float> right) {
    __m512 sum = _mm512_setzero_ps();
    std::size_t index = 0;
    for (; index + 16 <= left.size(); index += 16) {
        sum = _mm512_add_ps(
            sum,
            _mm512_mul_ps(
                _mm512_loadu_ps(left.data() + index),
                _mm512_loadu_ps(right.data() + index)));
    }
    float result = reduce_avx512(sum);
    for (; index < left.size(); ++index) {
        result += left[index] * right[index];
    }
    return result;
}

} // namespace
#endif

CpuIsa x86_detect_isa() noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512bw")) {
        return CpuIsa::avx512;
    }
    if (__builtin_cpu_supports("avx2")) {
        return CpuIsa::avx2;
    }
#endif
    return CpuIsa::scalar;
}

void x86_hadamard(std::span<float> values, CpuIsa isa) {
#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
    if (isa == CpuIsa::avx512) {
        hadamard_avx512(values);
    } else {
        hadamard_avx2(values);
    }
#else
    (void)values;
    (void)isa;
#endif
}

float x86_int5_scaled_dot(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input,
    CpuIsa isa) {
#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
    return isa == CpuIsa::avx512
               ? int5_dot_avx512(packed, scales, input)
               : int5_dot_avx2(packed, scales, input);
#else
    (void)packed;
    (void)scales;
    (void)input;
    (void)isa;
    return 0.0F;
#endif
}

float x86_bf16_dot(
    std::span<const std::uint16_t> weights,
    std::span<const float> input,
    CpuIsa isa) {
#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
    return isa == CpuIsa::avx512
               ? bf16_dot_avx512(weights, input)
               : bf16_dot_avx2(weights, input);
#else
    (void)weights;
    (void)input;
    (void)isa;
    return 0.0F;
#endif
}

float x86_f32_dot(
    std::span<const float> left,
    std::span<const float> right,
    CpuIsa isa) {
#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
    return isa == CpuIsa::avx512
               ? f32_dot_avx512(left, right)
               : f32_dot_avx2(left, right);
#else
    (void)left;
    (void)right;
    (void)isa;
    return 0.0F;
#endif
}

} // namespace adi::detail
