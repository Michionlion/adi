#include "simd_x86.hpp"

#include "adi/kernels.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#include <immintrin.h>
#endif

namespace adi::detail {

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
namespace {

alignas(64) constexpr std::array<std::uint8_t, 64> int5_pack_indexes{
    0, 1, 2, 3, 4, 40, 40, 40,
    5, 6, 7, 8, 9, 40, 40, 40,
    10, 11, 12, 13, 14, 40, 40, 40,
    15, 16, 17, 18, 19, 40, 40, 40,
    20, 21, 22, 23, 24, 40, 40, 40,
    25, 26, 27, 28, 29, 40, 40, 40,
    30, 31, 32, 33, 34, 40, 40, 40,
    35, 36, 37, 38, 39, 40, 40, 40,
};

alignas(64) constexpr std::array<std::uint8_t, 64> int5_shift_counts{
    0, 5, 10, 15, 20, 25, 30, 35,
    0, 5, 10, 15, 20, 25, 30, 35,
    0, 5, 10, 15, 20, 25, 30, 35,
    0, 5, 10, 15, 20, 25, 30, 35,
    0, 5, 10, 15, 20, 25, 30, 35,
    0, 5, 10, 15, 20, 25, 30, 35,
    0, 5, 10, 15, 20, 25, 30, 35,
    0, 5, 10, 15, 20, 25, 30, 35,
};

__m512i unpack_int5_group_vbmi(const std::uint8_t *packed) {
    constexpr __mmask64 packed_mask = (1ULL << 40) - 1;
    const auto bytes = _mm512_maskz_loadu_epi8(packed_mask, packed);
    const auto lanes = _mm512_permutexvar_epi8(
        _mm512_load_si512(int5_pack_indexes.data()),
        bytes);
    auto codes = _mm512_multishift_epi64_epi8(
        _mm512_load_si512(int5_shift_counts.data()),
        lanes);
    codes = _mm512_and_si512(codes, _mm512_set1_epi8(0x1F));
    return _mm512_sub_epi8(codes, _mm512_set1_epi8(16));
}

float reduce_avx512(__m512 value) {
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, value);
    float sum = 0.0F;
    for (const float lane : lanes) {
        sum += lane;
    }
    return sum;
}

template <int Quarter>
__m512 scaled_int5_quarter(__m512i codes, __m512 scale) {
    const auto code_bytes = _mm512_extracti32x4_epi32(codes, Quarter);
    return _mm512_mul_ps(
        _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(code_bytes)),
        scale);
}

} // namespace
#endif

float x86_int5_dot_vbmi(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> input) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    __m512 sum = _mm512_setzero_ps();
    for (std::size_t group = 0; group < scales.size(); ++group) {
        const auto codes = unpack_int5_group_vbmi(packed.data() + group * 40);
        const auto scale = _mm512_set1_ps(f16_to_f32(scales[group]));
        const auto input_offset = group * 64;
        sum = _mm512_add_ps(
            sum,
            _mm512_mul_ps(
                scaled_int5_quarter<0>(codes, scale),
                _mm512_loadu_ps(input.data() + input_offset)));
        sum = _mm512_add_ps(
            sum,
            _mm512_mul_ps(
                scaled_int5_quarter<1>(codes, scale),
                _mm512_loadu_ps(input.data() + input_offset + 16)));
        sum = _mm512_add_ps(
            sum,
            _mm512_mul_ps(
                scaled_int5_quarter<2>(codes, scale),
                _mm512_loadu_ps(input.data() + input_offset + 32)));
        sum = _mm512_add_ps(
            sum,
            _mm512_mul_ps(
                scaled_int5_quarter<3>(codes, scale),
                _mm512_loadu_ps(input.data() + input_offset + 48)));
    }
    return reduce_avx512(sum);
#else
    (void)packed;
    (void)scales;
    (void)input;
    return 0.0F;
#endif
}


void x86_int5_dot_batch_vbmi(
    std::span<const std::uint8_t> packed,
    std::span<const std::uint16_t> scales,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> scratch) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    constexpr std::size_t lanes = 16;
    const auto columns = scales.size() * 64;
    std::fill_n(scratch.begin(), static_cast<std::size_t>(batch) * lanes, 0.0F);
    for (std::size_t group = 0; group < scales.size(); ++group) {
        const auto codes = unpack_int5_group_vbmi(packed.data() + group * 40);
        const auto scale = _mm512_set1_ps(f16_to_f32(scales[group]));
        const auto weights_0 = scaled_int5_quarter<0>(codes, scale);
        const auto weights_1 = scaled_int5_quarter<1>(codes, scale);
        const auto weights_2 = scaled_int5_quarter<2>(codes, scale);
        const auto weights_3 = scaled_int5_quarter<3>(codes, scale);
        const auto input_offset = group * 64;
        for (std::uint32_t batch_index = 0;
             batch_index < batch;
             ++batch_index) {
            auto sum = _mm512_loadu_ps(
                scratch.data() +
                static_cast<std::size_t>(batch_index) * lanes);
            const auto *input =
                inputs.data() +
                static_cast<std::size_t>(batch_index) * columns +
                input_offset;
            sum = _mm512_add_ps(
                sum,
                _mm512_mul_ps(weights_0, _mm512_loadu_ps(input)));
            sum = _mm512_add_ps(
                sum,
                _mm512_mul_ps(weights_1, _mm512_loadu_ps(input + 16)));
            sum = _mm512_add_ps(
                sum,
                _mm512_mul_ps(weights_2, _mm512_loadu_ps(input + 32)));
            sum = _mm512_add_ps(
                sum,
                _mm512_mul_ps(weights_3, _mm512_loadu_ps(input + 48)));
            _mm512_storeu_ps(
                scratch.data() +
                    static_cast<std::size_t>(batch_index) * lanes,
                sum);
        }
    }
    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        outputs[batch_index] = reduce_avx512(
            _mm512_loadu_ps(
                scratch.data() +
                static_cast<std::size_t>(batch_index) * lanes));
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
