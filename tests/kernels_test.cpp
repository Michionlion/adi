#include "adi/kernels.hpp"
#include "adi/profiling.hpp"
#include "parallel.hpp"
#include "simd.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
    const auto native_isa = adi::detail::selected_cpu_isa();
    adi::reset_kernel_profiles();
    adi::set_kernel_profiling_enabled(true);
    bool worker_exception = false;
    try {
        adi::detail::parallel_ranges(
            1024, 64,
            [](std::uint32_t begin, std::uint32_t) {
                if (begin == 0) {
                    throw std::runtime_error("worker failure");
                }
            });
    } catch (const std::runtime_error &) {
        worker_exception = true;
    }
    assert(worker_exception);

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
    adi::set_kernel_profiling_enabled(false);
    const auto profiles = adi::kernel_profiles();
    const auto &expert_profile =
        profiles[static_cast<std::size_t>(adi::KernelKind::expert)];
    assert(expert_profile.calls == 1);
    assert(expert_profile.work_items == dimension * dimension);
    assert(adi::kernel_name(adi::KernelKind::expert) == "expert");
    std::vector<float> expert_scalar(dimension);
    std::vector<float> expert_native(dimension);
    adi::ExpertScratch expert_scalar_scratch;
    adi::ExpertScratch expert_native_scratch;
    adi::detail::force_cpu_isa_for_testing(adi::detail::CpuIsa::scalar);
    adi::mach_expert_matvec(
        matrix, input, expert_scalar, expert_scalar_scratch);
    adi::detail::force_cpu_isa_for_testing(native_isa);
    adi::mach_expert_matvec(
        matrix, input, expert_native, expert_native_scratch);
    assert(expert_scalar == expert_native);
    std::vector<float> expert_inputs(2 * input.size());
    std::copy(input.begin(), input.end(), expert_inputs.begin());
    std::transform(
        input.begin(),
        input.end(),
        expert_inputs.begin() + input.size(),
        [](float value) { return value * -0.5F; });
    std::vector<float> expert_batch_output(2 * output.size());
    std::vector<float> expert_repeated_output(2 * output.size());
    adi::ExpertScratch expert_batch_scratch;
    adi::ExpertScratch expert_repeated_scratch;
    adi::mach_expert_matmul(
        matrix,
        expert_inputs,
        2,
        expert_batch_output,
        expert_batch_scratch);
    adi::mach_expert_matvec(
        matrix,
        std::span<const float>(expert_inputs).first(input.size()),
        std::span<float>(expert_repeated_output).first(output.size()),
        expert_repeated_scratch);
    adi::mach_expert_matvec(
        matrix,
        std::span<const float>(expert_inputs).last(input.size()),
        std::span<float>(expert_repeated_output).last(output.size()),
        expert_repeated_scratch);
    assert(expert_batch_output == expert_repeated_output);
    assert(std::abs(output[0] - 32.0F) < 1e-5F);
    for (std::size_t index = 1; index < output.size(); ++index) {
        assert(std::abs(output[index]) < 1e-5F);
    }

    for (float value : {0.0F, -0.0F, 1.0F, -2.0F, 0.333251953125F, 65504.0F}) {
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
    std::vector<float> ne_scalar(dimension);
    std::vector<float> ne_native(dimension);
    adi::ExpertScratch ne_scalar_scratch;
    adi::ExpertScratch ne_native_scratch;
    adi::detail::force_cpu_isa_for_testing(adi::detail::CpuIsa::scalar);
    adi::mach_ne_matvec(ne_matrix, input, ne_scalar, ne_scalar_scratch);
    adi::detail::force_cpu_isa_for_testing(native_isa);
    adi::mach_ne_matvec(ne_matrix, input, ne_native, ne_native_scratch);
    assert(ne_scalar == ne_native);

    std::vector<float> ne_inputs(2 * input.size());
    std::copy(input.begin(), input.end(), ne_inputs.begin());
    std::transform(
        input.begin(),
        input.end(),
        ne_inputs.begin() + input.size(),
        [](float value) { return value * 2.0F; });
    std::vector<float> ne_outputs(2 * output.size());
    adi::ExpertScratch batch_ne_scratch;
    adi::mach_ne_matmul(
        ne_matrix,
        ne_inputs,
        2,
        ne_outputs,
        batch_ne_scratch);
    std::vector<float> repeated_ne_outputs(ne_outputs.size());
    adi::ExpertScratch repeated_ne_scratch;
    adi::mach_ne_matvec(
        ne_matrix,
        std::span<const float>(ne_inputs).first(input.size()),
        std::span<float>(repeated_ne_outputs).first(output.size()),
        repeated_ne_scratch);
    adi::mach_ne_matvec(
        ne_matrix,
        std::span<const float>(ne_inputs).last(input.size()),
        std::span<float>(repeated_ne_outputs).last(output.size()),
        repeated_ne_scratch);
    assert(ne_outputs == repeated_ne_outputs);

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
    std::vector<float> head_scalar(1);
    std::vector<float> head_native(1);
    const adi::MachHeadChunk comparison_head{
        1, 64, packed_head, head_scales, {}, {}};
    adi::detail::force_cpu_isa_for_testing(adi::detail::CpuIsa::scalar);
    adi::mach_head_matvec(comparison_head, head_input, head_scalar);
    adi::detail::force_cpu_isa_for_testing(native_isa);
    adi::mach_head_matvec(comparison_head, head_input, head_native);
    assert(
        std::abs(head_scalar[0] - head_native[0]) <=
        1.0e-5F * std::max(1.0F, std::abs(head_scalar[0])));

    std::vector<float> head_inputs(2 * 64);
    std::fill_n(head_inputs.begin(), 64, 1.0F);
    std::fill_n(head_inputs.begin() + 64, 64, 2.0F);
    std::vector<float> head_outputs(2);
    const adi::MachHeadChunk head{
        1, 64, packed_head, head_scales, {}, {}};
    adi::mach_head_matmul(head, head_inputs, 2, head_outputs);
    std::vector<float> repeated_outputs(2);
    adi::mach_head_matvec(
        head,
        std::span<const float>(head_inputs).first(64),
        std::span<float>(repeated_outputs).first(1));
    adi::mach_head_matvec(
        head,
        std::span<const float>(head_inputs).last(64),
        std::span<float>(repeated_outputs).last(1));
    assert(head_outputs == repeated_outputs);

    std::vector<std::uint16_t> bf16_values{
        0x3F80U, 0x4000U, 0x4040U,
        0x4080U, 0x40A0U, 0x40C0U,
    };
    std::vector<float> bf16_input{1.0F, 2.0F, 3.0F};
    std::vector<float> bf16_output(2);
    adi::bf16_matvec({2, 3, bf16_values}, bf16_input, bf16_output);
    assert(bf16_output[0] == 14.0F);
    assert(bf16_output[1] == 32.0F);
    std::vector<float> bf16_batch_input{
        1.0F, 2.0F, 3.0F,
        -1.0F, 0.5F, 2.0F,
    };
    std::vector<float> bf16_batch_output(4);
    adi::bf16_matmul(
        {2, 3, bf16_values},
        bf16_batch_input,
        2,
        bf16_batch_output);
    std::vector<float> bf16_repeated_output(4);
    adi::bf16_matvec(
        {2, 3, bf16_values},
        std::span<const float>(bf16_batch_input).first(3),
        std::span<float>(bf16_repeated_output).first(2));
    adi::bf16_matvec(
        {2, 3, bf16_values},
        std::span<const float>(bf16_batch_input).last(3),
        std::span<float>(bf16_repeated_output).last(2));
    assert(bf16_batch_output == bf16_repeated_output);

    std::vector<std::uint16_t> random_bf16(4 * 67);
    std::vector<float> random_input(67);
    std::uint32_t random_state = 0x12345678U;
    auto next_random = [&]() {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 17;
        random_state ^= random_state << 5;
        return random_state;
    };
    for (auto &value : random_bf16) {
        const float source =
            static_cast<float>(static_cast<std::int32_t>(next_random() % 2049) -
                               1024) /
            512.0F;
        value = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(source) >> 16);
    }
    for (auto &value : random_input) {
        value =
            static_cast<float>(static_cast<std::int32_t>(next_random() % 2049) -
                               1024) /
            512.0F;
    }
    std::vector<float> bf16_scalar(4);
    std::vector<float> bf16_native(4);
    adi::detail::force_cpu_isa_for_testing(adi::detail::CpuIsa::scalar);
    adi::bf16_matvec({4, 67, random_bf16}, random_input, bf16_scalar);
    adi::detail::force_cpu_isa_for_testing(native_isa);
    adi::bf16_matvec({4, 67, random_bf16}, random_input, bf16_native);
    for (std::size_t index = 0; index < bf16_scalar.size(); ++index) {
        assert(
            std::abs(bf16_scalar[index] - bf16_native[index]) <=
            2.0e-5F * std::max(1.0F, std::abs(bf16_scalar[index])));
    }

    std::vector<std::uint16_t> norm_weight(2, 0x3F80U);
    std::vector<float> norm_input{3.0F, 4.0F};
    std::vector<float> norm_output(2);
    adi::rms_norm(norm_input, norm_weight, 0.0F, 0.0F, norm_output);
    assert(std::abs(norm_output[0] - 3.0F / std::sqrt(12.5F)) < 1e-6F);
    assert(std::abs(norm_output[1] - 4.0F / std::sqrt(12.5F)) < 1e-6F);

    std::vector<std::uint16_t> zero_centered_weight(2, 0);
    adi::rms_norm(
        norm_input, zero_centered_weight, 1.0F, 0.0F, norm_output);
    assert(std::abs(norm_output[0] - 3.0F / std::sqrt(12.5F)) < 1e-6F);
    assert(std::abs(norm_output[1] - 4.0F / std::sqrt(12.5F)) < 1e-6F);

    std::vector<float> tiny{1.0e-5F};
    adi::l2_normalize(tiny, 1.0e-6F);
    assert(std::abs(tiny[0] - 1.0e-5F / std::sqrt(1.0001e-6F)) < 1e-7F);
    adi::detail::clear_cpu_isa_for_testing();
}
