#include "adi/kernels.hpp"
#include "simd.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

int main() {
    const auto features = adi::detail::detected_cpu_features();
    const auto selected = adi::detail::selected_cpu_isa();
    if (selected == adi::detail::CpuIsa::avx2) {
        assert(features.avx2);
    }
    if (selected == adi::detail::CpuIsa::avx512) {
        assert(features.avx512f);
        assert(features.avx512bw);
    }
    if (selected == adi::detail::CpuIsa::neon) {
        assert(features.neon);
    }
    if (selected == adi::detail::CpuIsa::sve) {
        assert(features.sve);
    }

    constexpr std::uint32_t columns = 128;
    std::vector<float> left(columns);
    std::vector<float> right(columns);
    for (std::uint32_t index = 0; index < columns; ++index) {
        left[index] = static_cast<float>(static_cast<std::int32_t>(index) - 64) /
                      32.0F;
        right[index] = static_cast<float>((index * 17U) % 71U) / 64.0F;
    }

    adi::detail::force_cpu_isa_for_testing(adi::detail::CpuIsa::scalar);
    const auto scalar_dot = adi::detail::selected_f32_dot_kernel();
    const float scalar = scalar_dot(left, right);
    assert(adi::detail::f32_dot(left, right) == scalar);
    adi::detail::force_cpu_isa_for_testing(selected);
    const float native =
        adi::detail::selected_f32_dot_kernel()(left, right);
    assert(std::abs(native - scalar) <=
           1.0e-5F * std::max(1.0F, std::abs(scalar)));

    if (features.avx2) {
        adi::detail::force_cpu_isa_for_testing(adi::detail::CpuIsa::avx2);
        const float avx2 =
            adi::detail::selected_f32_dot_kernel()(left, right);
        assert(std::abs(avx2 - scalar) <=
               1.0e-5F * std::max(1.0F, std::abs(scalar)));

        // Match the AVX2 kernel's eight independent accumulation lanes. This
        // catches packed-code or scale errors while preserving its reduction
        // order, so the comparison can be bit exact.
        constexpr std::uint32_t int5_columns = 192;
        std::vector<std::uint8_t> packed(int5_columns * 5 / 8);
        for (std::uint32_t block = 0; block < int5_columns / 8; ++block) {
            std::uint64_t word = 0;
            for (std::uint32_t lane = 0; lane < 8; ++lane) {
                const auto code = static_cast<std::int32_t>(
                                      (block * 11 + lane * 7) % 32) -
                                  16;
                word |= static_cast<std::uint64_t>(code + 16) << (lane * 5);
            }
            for (std::uint32_t byte = 0; byte < 5; ++byte) {
                packed[block * 5 + byte] = static_cast<std::uint8_t>(
                    (word >> (byte * 8)) & 0xFFU);
            }
        }
        const std::vector<std::uint16_t> scales{
            adi::f32_to_f16(0.03125F),
            adi::f32_to_f16(0.375F),
            adi::f32_to_f16(1.25F),
        };
        std::vector<float> int5_input(int5_columns);
        for (std::uint32_t index = 0; index < int5_columns; ++index) {
            int5_input[index] =
                static_cast<float>(static_cast<std::int32_t>(index % 29) - 14) /
                13.0F;
        }
        float lanes[8]{};
        for (std::uint32_t block = 0; block < int5_columns / 8; ++block) {
            std::uint64_t word = 0;
            for (std::uint32_t byte = 0; byte < 5; ++byte) {
                word |= static_cast<std::uint64_t>(packed[block * 5 + byte])
                        << (byte * 8);
            }
            const float scale = adi::f16_to_f32(scales[block / 8]);
            for (std::uint32_t lane = 0; lane < 8; ++lane) {
                const auto code = static_cast<std::int32_t>(
                                      (word >> (lane * 5)) & 0x1FU) -
                                  16;
                volatile float weight = static_cast<float>(code) * scale;
                volatile float product =
                    weight * int5_input[block * 8 + lane];
                lanes[lane] += product;
            }
        }
        float expected_int5 = 0.0F;
        for (float lane : lanes) {
            expected_int5 += lane;
        }
        const float avx2_int5 =
            adi::detail::int5_scaled_dot(packed, scales, int5_input);
        assert(std::bit_cast<std::uint32_t>(avx2_int5) ==
               std::bit_cast<std::uint32_t>(expected_int5));
    }
    if (features.avx512f && features.avx512bw) {
        adi::detail::force_cpu_isa_for_testing(adi::detail::CpuIsa::avx512);
        const float avx512 =
            adi::detail::selected_f32_dot_kernel()(left, right);
        assert(std::abs(avx512 - scalar) <=
               1.0e-5F * std::max(1.0F, std::abs(scalar)));
    }

    adi::detail::clear_cpu_isa_for_testing();
}
