#include "expert_batch.hpp"

#include "batch_pack.hpp"
#include "simd.hpp"

#include <stdexcept>

namespace adi::detail {
namespace {

using ExpertTilesBatchKernel = void (*)(
    const MachExpertMatrix &,
    std::span<const float>,
    std::span<const std::uint16_t>,
    std::span<const float>,
    std::span<const float>,
    std::uint32_t,
    std::span<float>,
    std::span<float>);

struct BatchKernel {
    std::uint32_t lanes = 0;
    ExpertTilesBatchKernel run = nullptr;
};

// selected_cpu_isa() already reports scalar for any ISA the running CPU does
// not support, so an ISA-specific entry point is never reached on a CPU that
// cannot execute it.
BatchKernel selected_batch_kernel() noexcept {
    switch (selected_cpu_isa()) {
    case CpuIsa::avx512:
        return {16, x86_expert_tiles_batch_avx512};
    case CpuIsa::avx2:
        return {8, x86_expert_tiles_batch_avx2};
    case CpuIsa::scalar:
    case CpuIsa::neon:
    case CpuIsa::sve:
        break;
    }
    return {};
}

} // namespace

std::uint32_t expert_batch_lanes() noexcept {
    return selected_batch_kernel().lanes;
}

void expert_matmul_tiles_batch(
    const MachExpertMatrix &matrix,
    std::span<const float> state_values,
    std::span<const std::uint16_t> wave_indexes,
    std::span<const float> wave_gamma,
    std::span<const float> inputs,
    std::uint32_t batch,
    std::span<float> outputs,
    std::span<float> packed) {
    const auto kernel = selected_batch_kernel();
    if (kernel.run == nullptr || batch == 0 ||
        inputs.size() !=
            static_cast<std::size_t>(batch) * matrix.columns ||
        outputs.size() != static_cast<std::size_t>(batch) * matrix.rows ||
        packed.size() !=
            batch_packed_floats(matrix.columns, batch, kernel.lanes)) {
        throw std::invalid_argument("Mach expert batch kernel shape mismatch");
    }
    kernel.run(
        matrix,
        state_values,
        wave_indexes,
        wave_gamma,
        inputs,
        batch,
        outputs,
        packed);
}

} // namespace adi::detail
