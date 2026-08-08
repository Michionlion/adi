#include "expert_batch.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#include "batch_simd_x86.hpp"
#include "expert_batch_impl.hpp"

#include <immintrin.h>
#endif

namespace adi::detail {

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
namespace {

template <std::uint32_t Step>
__m256i expert_row_state_indexes_i32(__m256i pairs) {
    // Each 32-bit lane contains the two consecutive states for one row. Mask
    // or shift the requested half, then scale once to the eight-float table
    // stride instead of rebuilding the vector through eight scalar loads.
    const auto indexes = [&] {
        if constexpr (Step == 0) {
            return _mm256_and_si256(pairs, _mm256_set1_epi32(0xFFFF));
        } else {
            return _mm256_srli_epi32(pairs, 16);
        }
    }();
    return _mm256_slli_epi32(indexes, 3);
}

template <std::uint32_t Step>
void accumulate_expert_row_step(
    __m256i state_pairs,
    const float *state_values,
    const float *input,
    __m256 &partial) {
    const auto indexes = expert_row_state_indexes_i32<Step>(state_pairs);
    const auto *step_input = input + Step * 8;
    for (std::uint32_t component = 0; component < 8; ++component) {
        const auto component_indexes = _mm256_add_epi32(
            indexes,
            _mm256_set1_epi32(static_cast<std::int32_t>(component)));
        const auto weights =
            _mm256_i32gather_ps(state_values, component_indexes, 4);
        partial = _mm256_add_ps(
            partial,
            _mm256_mul_ps(
                weights,
                _mm256_set1_ps(step_input[component])));
    }
}

void accumulate_expert_row_group(
    __m256i state_pairs,
    const float *state_values,
    const float *input,
    __m256 gamma,
    __m256 &sums) {
    auto partial = _mm256_setzero_ps();
    accumulate_expert_row_step<0>(
        state_pairs, state_values, input, partial);
    accumulate_expert_row_step<1>(
        state_pairs, state_values, input, partial);
    sums = _mm256_add_ps(sums, _mm256_mul_ps(partial, gamma));
}

} // namespace
#endif

void x86_expert_tiles_batch_avx2(
    [[maybe_unused]] const MachExpertMatrix &matrix,
    [[maybe_unused]] std::span<const float> state_values,
    [[maybe_unused]] std::span<const std::uint16_t> wave_indexes,
    [[maybe_unused]] std::span<const float> wave_gamma,
    [[maybe_unused]] std::span<const float> inputs,
    [[maybe_unused]] std::uint32_t batch,
    [[maybe_unused]] std::span<float> outputs,
    [[maybe_unused]] std::span<float> packed) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    expert_tiles_batch<Avx2BatchTraits>(
        matrix,
        state_values,
        wave_indexes,
        wave_gamma,
        inputs,
        batch,
        outputs,
        packed);
#endif
}

void x86_expert_matvec_rows_avx2(
    [[maybe_unused]] const MachExpertMatrix &matrix,
    [[maybe_unused]] std::span<const float> state_values,
    [[maybe_unused]] std::span<const std::uint16_t> wave_indexes,
    [[maybe_unused]] std::span<const float> wave_gamma,
    [[maybe_unused]] std::span<const float> input,
    [[maybe_unused]] std::span<float> output,
    [[maybe_unused]] std::uint32_t row_begin,
    [[maybe_unused]] std::uint32_t row_end) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    constexpr std::uint32_t tile_size = 16;
    constexpr std::uint32_t words_per_tile = 24;
    const auto tile_columns = matrix.columns / tile_size;
    for (std::uint32_t tile_row = row_begin;
         tile_row < row_end;
         ++tile_row) {
        auto sums_0 = _mm256_setzero_ps();
        auto sums_1 = _mm256_setzero_ps();
        for (std::uint32_t tile_column = 0;
             tile_column < tile_columns;
             ++tile_column) {
            const auto tile_index =
                static_cast<std::size_t>(tile_row) * tile_columns +
                tile_column;
            const auto *words =
                matrix.trellis.data() + tile_index * words_per_tile;
            const auto state_pairs_0 =
                expert_row_state_pairs_x86_256<0>(words);
            const auto state_pairs_1 =
                expert_row_state_pairs_x86_256<8>(words);
            const auto gamma = _mm256_set1_ps(
                wave_gamma[wave_indexes[tile_index]]);
            const auto *tile_input =
                input.data() +
                static_cast<std::size_t>(tile_column) * tile_size;
            // SIMD lanes are independent rows. Each lane retains the scalar
            // component order: state zero's eight values, then state one's.
            accumulate_expert_row_group(
                state_pairs_0,
                state_values.data(),
                tile_input,
                gamma,
                sums_0);
            accumulate_expert_row_group(
                state_pairs_1,
                state_values.data(),
                tile_input,
                gamma,
                sums_1);
        }
        auto *row_output =
            output.data() + static_cast<std::size_t>(tile_row) * tile_size;
        _mm256_storeu_ps(row_output, sums_0);
        _mm256_storeu_ps(row_output + 8, sums_1);
    }
#endif
}

} // namespace adi::detail
