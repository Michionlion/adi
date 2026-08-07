#include "adi/kernels.hpp"
#include "simd.hpp"

#include <bit>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

std::uint32_t next_random(std::uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void assert_equal_bits(
    std::span<const float> left,
    std::span<const float> right) {
    assert(left.size() == right.size());
    for (std::size_t index = 0; index < left.size(); ++index) {
        assert(std::bit_cast<std::uint32_t>(left[index]) ==
               std::bit_cast<std::uint32_t>(right[index]));
    }
}

} // namespace

int main() {
    const auto features = adi::detail::detected_cpu_features();
    if (!features.avx512f || !features.avx512bw ||
        !features.avx512vbmi) {
        return 0;
    }
    adi::detail::force_cpu_isa_for_testing(adi::detail::CpuIsa::avx512);

    std::uint32_t random = 0x243F6A88U;
    for (const std::uint32_t columns : {64U, 192U, 2048U}) {
        for (const std::uint32_t batch : {2U, 7U, 17U}) {
            std::vector<std::uint8_t> packed(columns / 8 * 5);
            std::vector<std::uint16_t> scales(columns / 64);
            std::vector<float> inputs(
                static_cast<std::size_t>(batch) * columns);
            for (auto &value : packed) {
                value = static_cast<std::uint8_t>(next_random(random));
            }
            for (auto &value : scales) {
                value = adi::f32_to_f16(
                    static_cast<float>(next_random(random) % 64 + 1) /
                    2048.0F);
            }
            for (auto &value : inputs) {
                value = static_cast<float>(
                            static_cast<std::int32_t>(
                                next_random(random) % 4097) -
                            2048) /
                        512.0F;
            }

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

            const auto scratch_size =
                adi::detail::int5_scaled_dot_batch_scratch_size(
                    columns,
                    batch);
            assert(scratch_size == static_cast<std::size_t>(batch) * 16);
            std::vector<float> scratch(scratch_size);
            std::vector<float> actual(batch);
            adi::detail::int5_scaled_dot_batch(
                packed,
                scales,
                inputs,
                batch,
                actual,
                scratch);
            assert_equal_bits(actual, expected);

            std::vector<float> legacy_scratch(columns);
            std::vector<float> legacy(batch);
            adi::detail::int5_scaled_dot_batch(
                packed,
                scales,
                inputs,
                batch,
                legacy,
                legacy_scratch);
            assert_equal_bits(legacy, expected);
        }
    }

    constexpr std::uint32_t rows = 32;
    constexpr std::uint32_t columns = 192;
    constexpr std::uint32_t batch = 7;
    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(rows) * columns / 8 * 5);
    std::vector<std::uint16_t> scales(
        static_cast<std::size_t>(rows) * columns / 64);
    std::vector<float> inputs(static_cast<std::size_t>(batch) * columns);
    for (auto &value : packed) {
        value = static_cast<std::uint8_t>(next_random(random));
    }
    for (auto &value : scales) {
        value = adi::f32_to_f16(
            static_cast<float>(next_random(random) % 64 + 1) / 2048.0F);
    }
    for (auto &value : inputs) {
        value = static_cast<float>(
                    static_cast<std::int32_t>(next_random(random) % 4097) -
                    2048) /
                512.0F;
    }
    const adi::MachHeadChunk head{
        rows,
        columns,
        packed,
        scales,
        {},
        {},
    };
    std::vector<float> expected(static_cast<std::size_t>(batch) * rows);
    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        adi::mach_head_matvec(
            head,
            std::span<const float>(inputs).subspan(
                static_cast<std::size_t>(batch_index) * columns,
                columns),
            std::span<float>(expected).subspan(
                static_cast<std::size_t>(batch_index) * rows,
                rows));
    }
    std::vector<float> actual(expected.size());
    adi::mach_head_matmul(head, inputs, batch, actual);
    assert_equal_bits(actual, expected);

    std::vector<float> invalid_scratch(111);
    std::vector<float> outputs(batch);
    bool rejected = false;
    try {
        adi::detail::int5_scaled_dot_batch(
            std::span<const std::uint8_t>(packed).first(columns / 8 * 5),
            std::span<const std::uint16_t>(scales).first(columns / 64),
            inputs,
            batch,
            outputs,
            invalid_scratch);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    adi::detail::clear_cpu_isa_for_testing();
}
