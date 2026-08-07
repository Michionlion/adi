#include "expert_batch.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#include "batch_simd_x86.hpp"
#include "expert_batch_impl.hpp"
#endif

namespace adi::detail {

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

} // namespace adi::detail
