#pragma once

// SIMD traits for the batch-oriented codec kernels.
//
// Every translation unit that instantiates a batch kernel with these traits is
// compiled with FP contraction disabled, so the separate multiply and add stay
// separate. Both AVX2 and AVX-512 make FMA available and a fused multiply-add
// rounds differently from a multiply followed by an add, which would put the
// SIMD lanes out of step with the scalar reference kernel.

// Each traits type is guarded by the feature macro of the ISA it uses, so an
// AVX2 translation unit never declares a __m512-returning function it has no
// ABI for.

#include <cstdint>
#include <immintrin.h>

namespace adi::detail {

#if defined(__AVX2__)
struct Avx2BatchTraits {
    static constexpr std::uint32_t lanes = 8;
    using Vec = __m256;

    static Vec load(const float *values) noexcept {
        return _mm256_loadu_ps(values);
    }
    static void store(float *values, Vec value) noexcept {
        _mm256_storeu_ps(values, value);
    }
    static Vec broadcast(float value) noexcept {
        return _mm256_set1_ps(value);
    }
    static Vec zero() noexcept {
        return _mm256_setzero_ps();
    }
    static Vec mul(Vec left, Vec right) noexcept {
        return _mm256_mul_ps(left, right);
    }
    static Vec add(Vec left, Vec right) noexcept {
        return _mm256_add_ps(left, right);
    }
};
#endif

#if defined(__AVX512F__)
struct Avx512BatchTraits {
    static constexpr std::uint32_t lanes = 16;
    using Vec = __m512;

    static Vec load(const float *values) noexcept {
        return _mm512_loadu_ps(values);
    }
    static void store(float *values, Vec value) noexcept {
        _mm512_storeu_ps(values, value);
    }
    static Vec broadcast(float value) noexcept {
        return _mm512_set1_ps(value);
    }
    static Vec zero() noexcept {
        return _mm512_setzero_ps();
    }
    static Vec mul(Vec left, Vec right) noexcept {
        return _mm512_mul_ps(left, right);
    }
    static Vec add(Vec left, Vec right) noexcept {
        return _mm512_add_ps(left, right);
    }
};
#endif

} // namespace adi::detail
