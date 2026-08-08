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
__m256i ne_states_for_step(__m256i chunks, __m256i next_chunks) {
    // Each 64-bit lane holds the four trellis words for one output row. Even
    // states are whole words; odd states join bytes from adjacent words.
    constexpr int word = Step / 2;
    if constexpr ((Step & 1) == 0) {
        return _mm256_and_si256(
            _mm256_srli_epi64(chunks, word * 16),
            _mm256_set1_epi64x(0xFFFF));
    } else {
        const auto low = _mm256_and_si256(
            _mm256_srli_epi64(chunks, word * 16),
            _mm256_set1_epi64x(0xFF));
        const auto high = [&] {
            if constexpr (word == 3) {
                return _mm256_and_si256(
                    _mm256_srli_epi64(next_chunks, 8),
                    _mm256_set1_epi64x(0xFF));
            } else {
                return _mm256_and_si256(
                    _mm256_srli_epi64(chunks, (word + 1) * 16 + 8),
                    _mm256_set1_epi64x(0xFF));
            }
        }();
        return _mm256_or_si256(_mm256_slli_epi64(low, 8), high);
    }
}

// Shifts four row chunks left by one row. The final lane comes from the next
// group, so odd state seven sees the first byte of the following row. Passing
// the first group as next_group for the final group preserves the tile wrap.
__m256i next_row_chunks(__m256i chunks, __m256i next_group) {
    const auto shifted = _mm256_permute4x64_epi64(
        chunks, _MM_SHUFFLE(0, 3, 2, 1));
    const auto next_first = _mm256_permute4x64_epi64(
        next_group, _MM_SHUFFLE(0, 0, 0, 0));
    return _mm256_blend_epi32(shifted, next_first, 0xC0);
}

template <int Step>
__m256i packed_ne_states(
    __m256i chunks_0,
    __m256i next_chunks_0,
    __m256i chunks_1,
    __m256i next_chunks_1) {
    // State extraction leaves one value in each 64-bit lane. Compress the
    // low dwords from two four-row groups into one eight-row gather vector.
    const auto states_0 = ne_states_for_step<Step>(chunks_0, next_chunks_0);
    const auto states_1 = ne_states_for_step<Step>(chunks_1, next_chunks_1);
    const auto compress = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    const auto packed_0 = _mm256_permutevar8x32_epi32(states_0, compress);
    const auto packed_1 = _mm256_permutevar8x32_epi32(states_1, compress);
    return _mm256_permute2x128_si256(packed_0, packed_1, 0x20);
}

template <int Step>
void accumulate_ne_step(
    __m256i chunks_0,
    __m256i next_chunks_0,
    __m256i chunks_1,
    __m256i next_chunks_1,
    const float *signed_tlut,
    float input_0,
    float input_1,
    __m256 scale,
    __m256 &sums) {
    const auto states = packed_ne_states<Step>(
        chunks_0, next_chunks_0, chunks_1, next_chunks_1);
    const auto products = _mm256_mullo_epi32(
        states,
        _mm256_add_epi32(states, _mm256_set1_epi32(1)));
    const auto indexes = _mm256_and_si256(
        _mm256_srli_epi32(products, 6),
        _mm256_set1_epi32(0x3FF));
    const auto weight_0 = _mm256_mul_ps(
        _mm256_i32gather_ps(signed_tlut, indexes, 8),
        scale);
    const auto weight_1 = _mm256_mul_ps(
        _mm256_i32gather_ps(signed_tlut + 1, indexes, 8),
        scale);
    sums = _mm256_add_ps(
        sums, _mm256_mul_ps(weight_0, _mm256_set1_ps(input_0)));
    sums = _mm256_add_ps(
        sums, _mm256_mul_ps(weight_1, _mm256_set1_ps(input_1)));
}

template <int Step>
void accumulate_ne_rows(
    const __m256i *chunks,
    const __m256i *next_chunks,
    const float *signed_tlut,
    const float *input,
    __m256 scale,
    __m256 *sums) {
    for (std::uint32_t group = 0; group < 2; ++group) {
        const auto first = group * 2;
        accumulate_ne_step<Step>(
            chunks[first],
            next_chunks[first],
            chunks[first + 1],
            next_chunks[first + 1],
            signed_tlut,
            input[Step * 2],
            input[Step * 2 + 1],
            scale,
            sums[group]);
    }
}

} // namespace
#endif

void x86_ne_tiles_batch_avx2(
    [[maybe_unused]] const MachNeMatrix &matrix,
    [[maybe_unused]] std::span<const float> state_values,
    [[maybe_unused]] std::span<const float> inputs,
    [[maybe_unused]] std::uint32_t batch,
    [[maybe_unused]] std::span<float> outputs,
    [[maybe_unused]] std::span<float> packed) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    ne_tiles_batch<Avx2BatchTraits>(
        matrix, state_values, inputs, batch, outputs, packed);
#endif
}

void x86_ne_matvec_rows_avx2(
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
    for (std::uint32_t tile_row = row_begin;
         tile_row < row_end;
         ++tile_row) {
        __m256 sums[2]{
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        };
        for (std::uint32_t tile_column = 0;
             tile_column < tile_columns;
             ++tile_column) {
            const auto tile_index =
                static_cast<std::size_t>(tile_row) * tile_columns +
                tile_column;
            const auto *words =
                matrix.trellis.data() + tile_index * words_per_tile;
            __m256i chunks[4]{
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(words)),
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(words + 16)),
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(words + 32)),
                _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(words + 48)),
            };
            __m256i next_chunks[4]{
                next_row_chunks(chunks[0], chunks[1]),
                next_row_chunks(chunks[1], chunks[2]),
                next_row_chunks(chunks[2], chunks[3]),
                next_row_chunks(chunks[3], chunks[0]),
            };
            const auto *tile_input =
                input.data() +
                static_cast<std::size_t>(tile_column) * tile_size;
            // Keep the scalar decoder's component-0 then component-1 order
            // for every state; the lanes are 16 independent output rows.
            accumulate_ne_rows<0>(
                chunks, next_chunks, signed_tlut.data(), tile_input, scale, sums);
            accumulate_ne_rows<1>(
                chunks, next_chunks, signed_tlut.data(), tile_input, scale, sums);
            accumulate_ne_rows<2>(
                chunks, next_chunks, signed_tlut.data(), tile_input, scale, sums);
            accumulate_ne_rows<3>(
                chunks, next_chunks, signed_tlut.data(), tile_input, scale, sums);
            accumulate_ne_rows<4>(
                chunks, next_chunks, signed_tlut.data(), tile_input, scale, sums);
            accumulate_ne_rows<5>(
                chunks, next_chunks, signed_tlut.data(), tile_input, scale, sums);
            accumulate_ne_rows<6>(
                chunks, next_chunks, signed_tlut.data(), tile_input, scale, sums);
            accumulate_ne_rows<7>(
                chunks, next_chunks, signed_tlut.data(), tile_input, scale, sums);
        }
        auto *row_output =
            output.data() + static_cast<std::size_t>(tile_row) * tile_size;
        _mm256_storeu_ps(row_output, sums[0]);
        _mm256_storeu_ps(row_output + 8, sums[1]);
    }
#endif
}

} // namespace adi::detail
