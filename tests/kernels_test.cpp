#include "adi/kernels.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

int main() {
    constexpr std::uint32_t dimension = 16;
    std::vector<std::uint16_t> trellis(24, 0);
    std::vector<std::uint16_t> su(dimension, adi::f32_to_f16(1.0F));
    std::vector<std::uint16_t> sv(dimension, adi::f32_to_f16(1.0F));
    std::vector<std::uint16_t> gamma{
        adi::f32_to_f16(1.0F),
        adi::f32_to_f16(2.0F),
    };
    std::vector<float> tlut(32768 * 8, 0.0F);
    for (std::size_t row = 0; row < 32768; ++row) {
        for (std::size_t component = 0; component < 8; ++component) {
            tlut[row * 8 + component] = static_cast<float>(component + 1);
        }
    }
    std::vector<float> input(dimension, 1.0F);
    std::vector<float> output(dimension);
    adi::ExpertScratch scratch;
    const adi::MachExpertMatrix matrix{
        dimension,
        dimension,
        trellis,
        su,
        sv,
        gamma,
        tlut,
    };

    adi::mach_expert_matvec(matrix, input, output, scratch);
    assert(std::abs(output[0] - 32.0F) < 1e-5F);
    for (std::size_t index = 1; index < output.size(); ++index) {
        assert(std::abs(output[index]) < 1e-5F);
    }

    for (float value : {0.0F, -0.0F, 1.0F, -2.0F, 0.33325F, 65504.0F}) {
        const auto round_trip = adi::f16_to_f32(adi::f32_to_f16(value));
        assert(round_trip == value);
    }
    assert(std::isinf(adi::f16_to_f32(adi::f32_to_f16(INFINITY))));
}
