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

    std::vector<std::uint16_t> ne_trellis(64, 0);
    std::vector<std::int8_t> signs(dimension, 1);
    std::vector<float> ne_tlut(512 * 2);
    for (std::size_t row = 0; row < 512; ++row) {
        ne_tlut[row * 2] = 1.0F;
        ne_tlut[row * 2 + 1] = 2.0F;
    }
    const adi::MachNeMatrix ne_matrix{
        dimension,
        dimension,
        ne_trellis,
        signs,
        signs,
        0.5F,
        ne_tlut,
    };
    adi::mach_ne_matvec(ne_matrix, input, output, scratch);
    assert(std::abs(output[0] - 8.0F) < 1e-5F);
    for (std::size_t index = 1; index < output.size(); ++index) {
        assert(std::abs(output[index]) < 1e-5F);
    }

    constexpr std::uint32_t embedding_columns = 64;
    std::vector<std::uint8_t> packed_embedding(embedding_columns / 2, 0x1FU);
    std::vector<std::uint16_t> minimum{adi::f32_to_f16(-1.0F)};
    std::vector<std::uint16_t> maximum{adi::f32_to_f16(1.0F)};
    std::vector<std::uint32_t> exception_indexes{3};
    std::vector<std::uint16_t> exception_bits{0x4000U};
    std::vector<float> embedding_output(embedding_columns);
    adi::mach_embedding_row(
        {1, embedding_columns, packed_embedding, minimum, maximum,
         exception_indexes, exception_bits},
        0,
        embedding_output);
    assert(std::abs(embedding_output[0] - (-1.0F + 2.0F / 15.0F)) < 1e-6F);
    assert(embedding_output[1] == 1.0F);
    assert(embedding_output[3] == 2.0F);

    std::vector<std::uint8_t> packed_head(40, 0);
    std::vector<std::uint16_t> head_scales{adi::f32_to_f16(0.5F)};
    std::vector<float> head_input(64, 1.0F);
    std::vector<float> head_output(1);
    adi::mach_head_matvec(
        {1, 64, packed_head, head_scales, {}, {}},
        head_input,
        head_output);
    assert(head_output[0] == -512.0F);
}
