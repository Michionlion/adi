#include "ne_batch.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#include "batch_simd_x86.hpp"
#include "ne_batch_impl.hpp"

#include <immintrin.h>
#endif

namespace adi::detail {

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
namespace {

template <int Step>
__m512i ne_states_for_step(__m512i chunks, __m512i next_chunks) {
    // Each 64-bit lane holds the four trellis words for one output row. Even
    // states are whole words; odd states join bytes from adjacent words.
    constexpr int word = Step / 2;
    if constexpr ((Step & 1) == 0) {
        return _mm512_and_si512(
            _mm512_srli_epi64(chunks, word * 16),
            _mm512_set1_epi64(0xFFFF));
    } else {
        const auto low = _mm512_and_si512(
            _mm512_srli_epi64(chunks, word * 16),
            _mm512_set1_epi64(0xFF));
        const auto high = [&] {
            if constexpr (word == 3) {
                return _mm512_and_si512(
                    _mm512_srli_epi64(next_chunks, 8),
                    _mm512_set1_epi64(0xFF));
            } else {
                return _mm512_and_si512(
                    _mm512_srli_epi64(chunks, (word + 1) * 16 + 8),
                    _mm512_set1_epi64(0xFF));
            }
        }();
        return _mm512_or_si512(_mm512_slli_epi64(low, 8), high);
    }
}

template <int Step>
void accumulate_ne_step(
    __m512i chunks,
    __m512i next_chunks,
    const float *signed_tlut,
    float input_0,
    float input_1,
    __m256 scale,
    __m256 &sums) {
    const auto states = ne_states_for_step<Step>(chunks, next_chunks);
    const auto products = _mm512_mul_epu32(
        states,
        _mm512_add_epi64(states, _mm512_set1_epi64(1)));
    const auto indexes = _mm512_and_si512(
        _mm512_srli_epi64(products, 6),
        _mm512_set1_epi64(0x3FF));
    const auto weight_0 = _mm256_mul_ps(
        _mm512_i64gather_ps(indexes, signed_tlut, 8),
        scale);
    const auto weight_1 = _mm256_mul_ps(
        _mm512_i64gather_ps(indexes, signed_tlut + 1, 8),
        scale);
    sums = _mm256_add_ps(
        sums,
        _mm256_mul_ps(weight_0, _mm256_set1_ps(input_0)));
    sums = _mm256_add_ps(
        sums,
        _mm256_mul_ps(weight_1, _mm256_set1_ps(input_1)));
}

template <int Step>
void accumulate_ne_rows(
    __m512i chunks_0,
    __m512i chunks_1,
    __m512i next_0,
    __m512i next_1,
    const float *signed_tlut,
    const float *input,
    __m256 scale,
    __m256 &sums_0,
    __m256 &sums_1) {
    accumulate_ne_step<Step>(
        chunks_0,
        next_0,
        signed_tlut,
        input[Step * 2],
        input[Step * 2 + 1],
        scale,
        sums_0);
    accumulate_ne_step<Step>(
        chunks_1,
        next_1,
        signed_tlut,
        input[Step * 2],
        input[Step * 2 + 1],
        scale,
        sums_1);
}

} // namespace
#endif

void x86_ne_tiles_batch_avx512(
    [[maybe_unused]] const MachNeMatrix &matrix,
    [[maybe_unused]] std::span<const float> state_values,
    [[maybe_unused]] std::span<const float> inputs,
    [[maybe_unused]] std::uint32_t batch,
    [[maybe_unused]] std::span<float> outputs,
    [[maybe_unused]] std::span<float> packed) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    ne_tiles_batch<Avx512BatchTraits>(
        matrix, state_values, inputs, batch, outputs, packed);
#endif
}

void x86_ne_matvec_rows_avx512(
    [[maybe_unused]] const MachNeMatrix &matrix,
    [[maybe_unused]] std::span<const float> signed_tlut,
    [[maybe_unused]] std::span<const float> input,
    [[maybe_unused]] std::span<float> output,
    [[maybe_unused]] std::uint32_t row_begin,
    [[maybe_unused]] std::uint32_t row_end) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    constexpr std::uint32_t tile_size = 16;
    constexpr std::uint32_t words_per_tile = 64;
    const auto tile_columns = matrix.columns / tile_size;
    const auto scale = _mm256_set1_ps(matrix.weight_scale);
    const auto next_indexes = _mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 8);
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
            const auto chunks_0 = _mm512_loadu_si512(words);
            const auto chunks_1 = _mm512_loadu_si512(words + 32);
            // Shift the row chunks by one lane for odd state 7, wrapping the
            // last row in each half to the first row in the other half.
            const auto next_0 = _mm512_permutex2var_epi64(
                chunks_0,
                next_indexes,
                chunks_1);
            const auto next_1 = _mm512_permutex2var_epi64(
                chunks_1,
                next_indexes,
                chunks_0);
            const auto *tile_input =
                input.data() +
                static_cast<std::size_t>(tile_column) * tile_size;
            // Keep the scalar decoder's component-0 then component-1 order
            // for every state; the lanes are 16 independent output rows.
            accumulate_ne_rows<0>(
                chunks_0, chunks_1, next_0, next_1,
                signed_tlut.data(), tile_input, scale, sums_0, sums_1);
            accumulate_ne_rows<1>(
                chunks_0, chunks_1, next_0, next_1,
                signed_tlut.data(), tile_input, scale, sums_0, sums_1);
            accumulate_ne_rows<2>(
                chunks_0, chunks_1, next_0, next_1,
                signed_tlut.data(), tile_input, scale, sums_0, sums_1);
            accumulate_ne_rows<3>(
                chunks_0, chunks_1, next_0, next_1,
                signed_tlut.data(), tile_input, scale, sums_0, sums_1);
            accumulate_ne_rows<4>(
                chunks_0, chunks_1, next_0, next_1,
                signed_tlut.data(), tile_input, scale, sums_0, sums_1);
            accumulate_ne_rows<5>(
                chunks_0, chunks_1, next_0, next_1,
                signed_tlut.data(), tile_input, scale, sums_0, sums_1);
            accumulate_ne_rows<6>(
                chunks_0, chunks_1, next_0, next_1,
                signed_tlut.data(), tile_input, scale, sums_0, sums_1);
            accumulate_ne_rows<7>(
                chunks_0, chunks_1, next_0, next_1,
                signed_tlut.data(), tile_input, scale, sums_0, sums_1);
        }
        auto *row_output =
            output.data() + static_cast<std::size_t>(tile_row) * tile_size;
        _mm256_storeu_ps(row_output, sums_0);
        _mm256_storeu_ps(row_output + 8, sums_1);
    }
#endif
}

} // namespace adi::detail
