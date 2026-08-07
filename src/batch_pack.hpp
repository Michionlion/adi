#pragma once

// Input packing shared by the batch-oriented codec kernels. Both vectorize
// across the batch dimension only, so both need the caller's row-major
// [batch, columns] matrix rearranged so that one SIMD load gathers the same
// column of `Lanes` independent batch items.

#include <cstddef>
#include <cstdint>

namespace adi::detail {

// Floats of packing scratch a shape needs, or zero when the ISA has no batch
// kernel.
[[nodiscard]] inline std::size_t batch_packed_floats(
    std::uint32_t columns,
    std::uint32_t batch,
    std::uint32_t lanes) noexcept {
    if (lanes == 0) {
        return 0;
    }
    const auto blocks = (batch + lanes - 1) / lanes;
    return static_cast<std::size_t>(blocks) * columns * lanes;
}

// Rearranges [batch, columns] into [block, column, lane]. Padding lanes are
// zero: they contribute nothing to any accumulation and their results are
// never written back to the caller's output.
template <std::uint32_t Lanes>
void pack_batch_inputs(
    const float *inputs,
    std::uint32_t batch,
    std::uint32_t columns,
    float *packed) {
    const auto blocks = (batch + Lanes - 1) / Lanes;
    for (std::uint32_t block = 0; block < blocks; ++block) {
        float *destination =
            packed + static_cast<std::size_t>(block) * columns * Lanes;
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto item = block * Lanes + lane;
            if (item >= batch) {
                for (std::uint32_t column = 0; column < columns; ++column) {
                    destination[static_cast<std::size_t>(column) * Lanes +
                                lane] = 0.0F;
                }
                continue;
            }
            const float *source =
                inputs + static_cast<std::size_t>(item) * columns;
            for (std::uint32_t column = 0; column < columns; ++column) {
                destination[static_cast<std::size_t>(column) * Lanes + lane] =
                    source[column];
            }
        }
    }
}

} // namespace adi::detail
