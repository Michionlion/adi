#include "simd_x86.hpp"

#include "adi/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <span>
#include <utility>

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
    std::uint32_t low;
    std::memcpy(&low, packed, sizeof(low));
    const auto word =
        static_cast<std::uint64_t>(low) |
        (static_cast<std::uint64_t>(packed[4]) << 32);
    const auto words = _mm256_set1_epi64x(static_cast<std::int64_t>(word));
    // Build one little-endian 32-bit window per code, then shift each lane by
    // its bit offset inside that window. This is the AVX2 analogue of the
    // VBMI byte-permute plus multishift decoder, without reading past the
    // five-byte block.
    const auto windows = _mm256_shuffle_epi8(
        words,
        _mm256_setr_epi8(
            0, 1, 2, 3, 0, 1, 2, 3,
            9, 10, 11, 12, 9, 10, 11, 12,
            2, 3, 4, 5, 3, 4, 5, 6,
            11, 12, 13, 14, 12, 13, 14, 15));
    auto codes = _mm256_srlv_epi32(
        windows,
        _mm256_setr_epi32(0, 5, 2, 7, 4, 1, 6, 3));
    codes = _mm256_sub_epi32(
        _mm256_and_si256(codes, _mm256_set1_epi32(0x1F)),
        _mm256_set1_epi32(16));
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

inline __m256 accumulate_int5_block(
    __m256 sum,
    __m256 weights,
    const float *input) {
    return _mm256_add_ps(
        sum,
        _mm256_mul_ps(weights, _mm256_loadu_ps(input)));
}

template <std::size_t Batch, std::size_t... Indexes>
void int5_dot_batch_fixed_avx2(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> inputs,
    std::span<float> outputs,
    std::index_sequence<Indexes...>) {
    const auto columns = scales.size() * 64;
    __m256 sums[Batch];
    ((sums[Indexes] = _mm256_setzero_ps()), ...);
    for (std::size_t group = 0; group < scales.size(); ++group) {
        const auto scale = _mm256_set1_ps(f16_to_f32(scales[group]));
        for (std::size_t block = 0; block < 8; ++block) {
            const auto weights = unpack_int5_block(
                packed.data() + group * 40 + block * 5,
                scale);
            const auto input_offset = group * 64 + block * 8;
            ((sums[Indexes] = accumulate_int5_block(
                  sums[Indexes],
                  weights,
                  inputs.data() + Indexes * columns + input_offset)),
             ...);
        }
    }
    ((outputs[Indexes] = reduce_avx2(sums[Indexes])), ...);
}

template <std::size_t Batch>
void int5_dot_batch_fixed_avx2(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> inputs,
    std::span<float> outputs) {
    int5_dot_batch_fixed_avx2<Batch>(
        packed,
        scales,
        inputs,
        outputs,
        std::make_index_sequence<Batch>{});
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

void x86_int5_dot_batch_avx2(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> scratch) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    if (batch == 2) {
        int5_dot_batch_fixed_avx2<2>(packed, scales, inputs, outputs);
        return;
    }
    if (batch == 4) {
        int5_dot_batch_fixed_avx2<4>(packed, scales, inputs, outputs);
        return;
    }
    if (batch == 8) {
        int5_dot_batch_fixed_avx2<8>(packed, scales, inputs, outputs);
        return;
    }
    // The shared batch interface reserves sixteen floats per item for the
    // AVX-512 kernel. AVX2 uses the first eight as lane accumulators and keeps
    // the same stride so dispatch needs no ISA-specific scratch contract.
    constexpr std::size_t scratch_stride = 16;
    std::fill_n(
        scratch.begin(),
        static_cast<std::size_t>(batch) * scratch_stride,
        0.0F);
    const auto columns = scales.size() * 64;
    for (std::size_t group = 0; group < scales.size(); ++group) {
        const auto scale = _mm256_set1_ps(f16_to_f32(scales[group]));
        for (std::size_t block = 0; block < 8; ++block) {
            const auto weights = unpack_int5_block(
                packed.data() + group * 40 + block * 5,
                scale);
            const auto input_offset = group * 64 + block * 8;
            for (std::uint32_t batch_index = 0;
                 batch_index < batch;
                 ++batch_index) {
                auto *lanes =
                    scratch.data() +
                    static_cast<std::size_t>(batch_index) * scratch_stride;
                const auto *input =
                    inputs.data() +
                    static_cast<std::size_t>(batch_index) * columns +
                    input_offset;
                _mm256_storeu_ps(
                    lanes,
                    _mm256_add_ps(
                        _mm256_loadu_ps(lanes),
                        _mm256_mul_ps(
                            weights,
                            _mm256_loadu_ps(input))));
            }
        }
    }
    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        outputs[batch_index] = reduce_avx2(
            _mm256_loadu_ps(
                scratch.data() +
                static_cast<std::size_t>(batch_index) * scratch_stride));
    }
#else
    (void)packed;
    (void)scales;
    (void)inputs;
    (void)batch;
    (void)outputs;
    (void)scratch;
#endif
}

} // namespace adi::detail
