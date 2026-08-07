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
float reduce_avx512(__m512 value) {
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, value);
    float sum = 0.0F;
    for (const float lane : lanes) {
        sum += lane;
    }
    return sum;
}

void hadamard_avx512(std::span<float> values) {
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

float int5_dot_avx512(
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

float bf16_dot_avx512(
    std::span<const std::uint16_t> weights,
    std::span<const float> input) {
    __m512 sum = _mm512_setzero_ps();
    std::size_t index = 0;
    for (; index + 16 <= weights.size(); index += 16) {
        const auto packed = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(weights.data() + index));
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

float f32_dot_avx512(
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

float gated_delta_update_avx512(
    std::span<float> state,
    std::span<const float> query,
    std::span<const float> key,
    float value,
    float beta,
    float decay) {
    const auto decay_vector = _mm512_set1_ps(decay);
    __m512 prediction_sum = _mm512_setzero_ps();
    std::size_t index = 0;
    for (; index + 16 <= state.size(); index += 16) {
        auto state_vector = _mm512_mul_ps(
            _mm512_loadu_ps(state.data() + index),
            decay_vector);
        _mm512_storeu_ps(state.data() + index, state_vector);
        prediction_sum = _mm512_add_ps(
            prediction_sum,
            _mm512_mul_ps(
                state_vector,
                _mm512_loadu_ps(key.data() + index)));
    }
    float prediction = reduce_avx512(prediction_sum);
    for (; index < state.size(); ++index) {
        state[index] *= decay;
        prediction += state[index] * key[index];
    }

    const float delta = (value - prediction) * beta;
    const auto delta_vector = _mm512_set1_ps(delta);
    __m512 attended_sum = _mm512_setzero_ps();
    index = 0;
    for (; index + 16 <= state.size(); index += 16) {
        auto state_vector = _mm512_add_ps(
            _mm512_loadu_ps(state.data() + index),
            _mm512_mul_ps(
                _mm512_loadu_ps(key.data() + index),
                delta_vector));
        _mm512_storeu_ps(state.data() + index, state_vector);
        attended_sum = _mm512_add_ps(
            attended_sum,
            _mm512_mul_ps(
                state_vector,
                _mm512_loadu_ps(query.data() + index)));
    }
    float attended = reduce_avx512(attended_sum);
    for (; index < state.size(); ++index) {
        state[index] += key[index] * delta;
        attended += state[index] * query[index];
    }
    return attended;
}
#endif

} // namespace

const X86Kernels &x86_avx512_kernels() noexcept {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    static const X86Kernels kernels{
        hadamard_avx512,
        int5_dot_avx512,
        bf16_dot_avx512,
        f32_dot_avx512,
        gated_delta_update_avx512,
    };
#else
    static const X86Kernels kernels;
#endif
    return kernels;
}

} // namespace adi::detail
