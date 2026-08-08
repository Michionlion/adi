#include "ne_batch.hpp"

#include "batch_pack.hpp"
#include "simd.hpp"

#include <stdexcept>

namespace adi::detail {
namespace {

using NeTilesBatchKernel = void (*)(
    const MachNeMatrix &,
    std::span<const float>,
    std::span<const float>,
    std::uint32_t,
    std::span<float>,
    std::span<float>);

struct BatchKernel {
    std::uint32_t lanes = 0;
    NeTilesBatchKernel run = nullptr;
};

// selected_cpu_isa() already reports scalar for any ISA the running CPU does
// not support, so an ISA-specific entry point is never reached on a CPU that
// cannot execute it.
BatchKernel selected_batch_kernel() noexcept {
    switch (selected_cpu_isa()) {
    case CpuIsa::avx512:
        return {16, x86_ne_tiles_batch_avx512};
    case CpuIsa::avx2:
        return {8, x86_ne_tiles_batch_avx2};
    case CpuIsa::scalar:
    case CpuIsa::neon:
    case CpuIsa::sve:
        break;
    }
    return {};
}

} // namespace

std::uint32_t ne_batch_lanes() noexcept {
    return selected_batch_kernel().lanes;
}

NeMatvecRowsKernel selected_ne_matvec_rows_kernel() noexcept {
    return selected_cpu_isa() == CpuIsa::avx512
               ? x86_ne_matvec_rows_avx512
               : nullptr;
}

void ne_matmul_tiles_batch(
    const MachNeMatrix &matrix,
    std::span<const float> state_values,
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
        throw std::invalid_argument("Mach NE batch kernel shape mismatch");
    }
    kernel.run(matrix, state_values, inputs, batch, outputs, packed);
}

} // namespace adi::detail
