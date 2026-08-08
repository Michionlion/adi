#include "simd_x86.hpp"

#include "adi/kernels.hpp"

#include <cmath>
#include <cstdint>
#include <span>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#include <immintrin.h>
#endif

namespace adi::detail {
namespace {

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
float reduce_avx2(__m256 value) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, value);
    float sum = 0.0F;
    for (const float lane : lanes) {
        sum += lane;
    }
    return sum;
}

void hadamard_avx2(std::span<float> values) {
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

__m256 unpack_int5_block(
    const std::uint8_t *packed,
    __m256 scale) {
    std::uint64_t word = 0;
    for (std::uint32_t byte = 0; byte < 5; ++byte) {
        word |= static_cast<std::uint64_t>(packed[byte]) << (byte * 8);
    }
    const auto codes = _mm256_setr_epi32(
        static_cast<std::int32_t>((word >> 0) & 0x1FU) - 16,
        static_cast<std::int32_t>((word >> 5) & 0x1FU) - 16,
        static_cast<std::int32_t>((word >> 10) & 0x1FU) - 16,
        static_cast<std::int32_t>((word >> 15) & 0x1FU) - 16,
        static_cast<std::int32_t>((word >> 20) & 0x1FU) - 16,
        static_cast<std::int32_t>((word >> 25) & 0x1FU) - 16,
        static_cast<std::int32_t>((word >> 30) & 0x1FU) - 16,
        static_cast<std::int32_t>((word >> 35) & 0x1FU) - 16);
    return _mm256_mul_ps(_mm256_cvtepi32_ps(codes), scale);
}

float int5_dot_avx2(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input) {
    __m256 sum = _mm256_setzero_ps();
    for (std::size_t group = 0; group < scales.size(); ++group) {
        const auto scale = _mm256_set1_ps(f16_to_f32(scales[group]));
        for (std::size_t block = 0; block < 8; ++block) {
            const auto index = group * 64 + block * 8;
            sum = _mm256_add_ps(
                sum,
                _mm256_mul_ps(
                    unpack_int5_block(
                        packed.data() + group * 40 + block * 5,
                        scale),
                    _mm256_loadu_ps(input.data() + index)));
        }
    }
    return reduce_avx2(sum);
}

float bf16_dot_avx2(
    std::span<const std::uint16_t> weights,
    std::span<const float> input) {
    __m256 sum = _mm256_setzero_ps();
    std::size_t index = 0;
    for (; index + 8 <= weights.size(); index += 8) {
        const auto packed = _mm_loadu_si128(
            reinterpret_cast<const __m128i *>(weights.data() + index));
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

float f32_dot_avx2(
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

float gated_delta_update_avx2(
    std::span<float> state,
    std::span<const float> query,
    std::span<const float> key,
    float value,
    float beta,
    float decay) {
    const auto decay_vector = _mm256_set1_ps(decay);
    __m256 prediction_sum = _mm256_setzero_ps();
    std::size_t index = 0;
    for (; index + 8 <= state.size(); index += 8) {
        auto state_vector = _mm256_mul_ps(
            _mm256_loadu_ps(state.data() + index),
            decay_vector);
        _mm256_storeu_ps(state.data() + index, state_vector);
        prediction_sum = _mm256_add_ps(
            prediction_sum,
            _mm256_mul_ps(
                state_vector,
                _mm256_loadu_ps(key.data() + index)));
    }
    float prediction = reduce_avx2(prediction_sum);
    for (; index < state.size(); ++index) {
        state[index] *= decay;
        prediction += state[index] * key[index];
    }

    const float delta = (value - prediction) * beta;
    const auto delta_vector = _mm256_set1_ps(delta);
    __m256 attended_sum = _mm256_setzero_ps();
    index = 0;
    for (; index + 8 <= state.size(); index += 8) {
        auto state_vector = _mm256_add_ps(
            _mm256_loadu_ps(state.data() + index),
            _mm256_mul_ps(
                _mm256_loadu_ps(key.data() + index),
                delta_vector));
        _mm256_storeu_ps(state.data() + index, state_vector);
        attended_sum = _mm256_add_ps(
            attended_sum,
            _mm256_mul_ps(
                state_vector,
                _mm256_loadu_ps(query.data() + index)));
    }
    float attended = reduce_avx2(attended_sum);
    for (; index < state.size(); ++index) {
        state[index] += key[index] * delta;
        attended += state[index] * query[index];
    }
    return attended;
}
#endif

} // namespace

const X86Kernels &x86_avx2_kernels() noexcept {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    static const X86Kernels kernels{
        hadamard_avx2,
        int5_dot_avx2,
        bf16_dot_avx2,
        f32_dot_avx2,
        gated_delta_update_avx2,
    };
#else
    static const X86Kernels kernels;
#endif
    return kernels;
}

} // namespace adi::detail
