#include "adi/kernels.hpp"
#include "simd.hpp"

#include <algorithm>
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
