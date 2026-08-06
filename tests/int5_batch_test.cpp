#include "adi/kernels.hpp"
#include "simd.hpp"

#include <bit>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
    constexpr std::uint32_t columns = 192;
    constexpr std::uint32_t batch = 7;
    std::vector<std::uint8_t> packed(columns / 8 * 5);
    std::vector<std::uint16_t> scales(columns / 64);
    std::vector<float> inputs(static_cast<std::size_t>(batch) * columns);

    std::uint32_t random = 0xC001D00DU;
    const auto next = [&] {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        return random;
    };
    for (auto &value : packed) {
        value = static_cast<std::uint8_t>(next());
    }
    for (std::size_t index = 0; index < scales.size(); ++index) {
        scales[index] = adi::f32_to_f16(
            0.03125F * static_cast<float>(index + 1));
    }
    for (auto &value : inputs) {
        value = static_cast<float>(
                    static_cast<std::int32_t>(next() % 2049) - 1024) /
                256.0F;
    }

    const auto native = adi::detail::selected_cpu_isa();
    const auto verify_available_isas = [&](const auto &verify) {
        verify(adi::detail::CpuIsa::scalar);
        if (native == adi::detail::CpuIsa::avx512) {
            verify(adi::detail::CpuIsa::avx2);
        }
        if (native != adi::detail::CpuIsa::scalar) {
            verify(native);
        }
    };
    const auto verify = [&](adi::detail::CpuIsa isa) {
        adi::detail::force_cpu_isa_for_testing(isa);
        std::vector<float> expected(batch);
        for (std::uint32_t batch_index = 0;
             batch_index < batch;
             ++batch_index) {
            expected[batch_index] = adi::detail::int5_scaled_dot(
                packed,
                scales,
                std::span<const float>(inputs).subspan(
                    static_cast<std::size_t>(batch_index) * columns,
                    columns));
        }
        std::vector<float> actual(batch);
        std::vector<float> weights(columns);
        adi::detail::int5_scaled_dot_batch(
            packed,
            scales,
            inputs,
            batch,
            actual,
            weights);
        assert(actual == expected);
    };
    verify_available_isas(verify);

    constexpr std::uint32_t head_rows = 1024;
    std::vector<std::uint8_t> head_packed(
        static_cast<std::size_t>(head_rows) * columns / 8 * 5);
    std::vector<std::uint16_t> head_scales(
        static_cast<std::size_t>(head_rows) * columns / 64);
    for (auto &value : head_packed) {
        value = static_cast<std::uint8_t>(next());
    }
    for (auto &value : head_scales) {
        value = adi::f32_to_f16(
            0.0078125F * static_cast<float>(next() % 8 + 1));
    }
    const std::vector<std::uint32_t> protected_rows{0, 511, 1023};
    std::vector<std::uint16_t> protected_values(
        protected_rows.size() * columns);
    for (auto &value : protected_values) {
        const float source = static_cast<float>(
                                 static_cast<std::int32_t>(next() % 513) -
                                 256) /
                             128.0F;
        value = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(source) >> 16);
    }
    const adi::MachHeadChunk head{
        head_rows,
        columns,
        head_packed,
        head_scales,
        protected_rows,
        protected_values,
    };
    const auto verify_head = [&](adi::detail::CpuIsa isa) {
        adi::detail::force_cpu_isa_for_testing(isa);
        std::vector<float> expected(
            static_cast<std::size_t>(batch) * head_rows);
        for (std::uint32_t batch_index = 0;
             batch_index < batch;
             ++batch_index) {
            adi::mach_head_matvec(
                head,
                std::span<const float>(inputs).subspan(
                    static_cast<std::size_t>(batch_index) * columns,
                    columns),
                std::span<float>(expected).subspan(
                    static_cast<std::size_t>(batch_index) * head_rows,
                    head_rows));
        }
        std::vector<float> actual(expected.size());
        adi::mach_head_matmul(head, inputs, batch, actual);
        assert(actual == expected);
    };
    verify_available_isas(verify_head);

    constexpr std::uint32_t bf16_rows = 257;
    constexpr std::uint32_t bf16_columns = 67;
    constexpr std::uint32_t bf16_batch = 5;
    std::vector<std::uint16_t> bf16_values(
        static_cast<std::size_t>(bf16_rows) * bf16_columns);
    std::vector<float> bf16_inputs(
        static_cast<std::size_t>(bf16_batch) * bf16_columns);
    for (auto &value : bf16_values) {
        const float source = static_cast<float>(
                                 static_cast<std::int32_t>(next() % 513) -
                                 256) /
                             128.0F;
        value = static_cast<std::uint16_t>(
            std::bit_cast<std::uint32_t>(source) >> 16);
    }
    for (auto &value : bf16_inputs) {
        value = static_cast<float>(
                    static_cast<std::int32_t>(next() % 2049) - 1024) /
                256.0F;
    }
    const adi::Bf16Matrix bf16{
        bf16_rows,
        bf16_columns,
        bf16_values,
    };
    const auto verify_bf16 = [&](adi::detail::CpuIsa isa) {
        adi::detail::force_cpu_isa_for_testing(isa);
        std::vector<float> expected(
            static_cast<std::size_t>(bf16_batch) * bf16_rows);
        for (std::uint32_t batch_index = 0;
             batch_index < bf16_batch;
             ++batch_index) {
            adi::bf16_matvec(
                bf16,
                std::span<const float>(bf16_inputs).subspan(
                    static_cast<std::size_t>(batch_index) * bf16_columns,
                    bf16_columns),
                std::span<float>(expected).subspan(
                    static_cast<std::size_t>(batch_index) * bf16_rows,
                    bf16_rows));
        }
        std::vector<float> actual(expected.size());
        adi::bf16_matmul(
            bf16,
            bf16_inputs,
            bf16_batch,
            actual);
        assert(actual == expected);
    };
    verify_available_isas(verify_bf16);

    bool rejected = false;
    try {
        std::vector<float> outputs(batch);
        std::vector<float> short_weights(columns - 1);
        adi::detail::int5_scaled_dot_batch(
            packed,
            scales,
            inputs,
            batch,
            outputs,
            short_weights);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    adi::detail::clear_cpu_isa_for_testing();
}
