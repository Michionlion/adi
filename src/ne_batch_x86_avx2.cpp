#include "ne_batch.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#include "batch_simd_x86.hpp"
#include "ne_batch_impl.hpp"
#endif

namespace adi::detail {

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

} // namespace adi::detail
