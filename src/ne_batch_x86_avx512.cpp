#include "ne_batch.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#include "ne_batch_impl.hpp"

#include <immintrin.h>
#endif

namespace adi::detail {
namespace {

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
struct Avx512Traits {
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
    // Separate multiply and add. The translation unit disables FP
    // contraction so the compiler cannot fuse these into an FMA, which
    // would round differently from the scalar kernel.
    static Vec mul(Vec left, Vec right) noexcept {
        return _mm512_mul_ps(left, right);
    }
    static Vec add(Vec left, Vec right) noexcept {
        return _mm512_add_ps(left, right);
    }
};
#endif

} // namespace

void x86_ne_tiles_batch_avx512(
    [[maybe_unused]] const MachNeMatrix &matrix,
    [[maybe_unused]] std::span<const float> state_values,
    [[maybe_unused]] std::span<const float> inputs,
    [[maybe_unused]] std::uint32_t batch,
    [[maybe_unused]] std::span<float> outputs,
    [[maybe_unused]] std::span<float> packed) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
    ne_tiles_batch<Avx512Traits>(
        matrix, state_values, inputs, batch, outputs, packed);
#endif
}

} // namespace adi::detail
