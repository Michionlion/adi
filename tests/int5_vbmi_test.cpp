#include "adi/kernels.hpp"
#include "simd.hpp"
#include "simd_x86.hpp"

#include <bit>
#include <cassert>
#include <cstdint>
#include <vector>

namespace {

std::uint32_t next_random(std::uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

} // namespace

int main() {
    const auto features = adi::detail::detected_cpu_features();
    if (!features.avx512f || !features.avx512bw ||
        !features.avx512vbmi) {
        return 0;
    }

    const auto fallback = adi::detail::x86_avx512_kernels().int5_dot;
    assert(fallback != nullptr);
    std::uint32_t random = 0xC001D00DU;
    for (const std::uint32_t columns : {64U, 128U, 192U, 2048U}) {
        for (std::uint32_t trial = 0; trial < 64; ++trial) {
            std::vector<std::uint8_t> packed(columns / 8 * 5);
            std::vector<std::uint16_t> scales(columns / 64);
            std::vector<float> input(columns);
            for (auto &value : packed) {
                value = static_cast<std::uint8_t>(next_random(random));
            }
            for (auto &value : scales) {
                const auto magnitude =
                    static_cast<float>(next_random(random) % 64 + 1) /
                    2048.0F;
                const float sign =
                    (next_random(random) & 1U) == 0 ? 1.0F : -1.0F;
                value = adi::f32_to_f16(sign * magnitude);
            }
            for (auto &value : input) {
                value = static_cast<float>(
                            static_cast<std::int32_t>(
                                next_random(random) % 4097) -
                            2048) /
                        512.0F;
            }

            const float expected = fallback(packed, scales, input);
            const float actual =
                adi::detail::x86_int5_dot_vbmi(packed, scales, input);
            assert(std::bit_cast<std::uint32_t>(actual) ==
                   std::bit_cast<std::uint32_t>(expected));

            adi::detail::force_cpu_isa_for_testing(
                adi::detail::CpuIsa::avx512);
            const float dispatched =
                adi::detail::int5_scaled_dot(packed, scales, input);
            assert(std::bit_cast<std::uint32_t>(dispatched) ==
                   std::bit_cast<std::uint32_t>(actual));
        }
    }
    adi::detail::clear_cpu_isa_for_testing();
}
