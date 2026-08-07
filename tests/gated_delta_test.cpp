#include "parallel.hpp"
#include "simd.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
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

bool close(float left, float right) {
    return std::abs(left - right) <=
           2.0e-5F * std::max({1.0F, std::abs(left), std::abs(right)});
}

} // namespace

int main() {
    constexpr std::uint32_t width = 128;
    std::uint32_t random = 0xA341316CU;
    std::vector<float> initial(width);
    std::vector<float> query(width);
    std::vector<float> key(width);
    for (auto *values : {&initial, &query, &key}) {
        for (auto &value : *values) {
            value = static_cast<float>(
                        static_cast<std::int32_t>(next_random(random) % 2049) -
                        1024) /
                    4096.0F;
        }
    }

    const auto native = adi::detail::selected_cpu_isa();
    auto scalar_state = initial;
    adi::detail::force_cpu_isa_for_testing(adi::detail::CpuIsa::scalar);
    const float scalar = adi::detail::gated_delta_update(
        scalar_state,
        query,
        key,
        0.375F,
        0.625F,
        0.9375F);

    auto native_state = initial;
    adi::detail::force_cpu_isa_for_testing(native);
    const float native_result = adi::detail::gated_delta_update(
        native_state,
        query,
        key,
        0.375F,
        0.625F,
        0.9375F);
    assert(close(native_result, scalar));
    for (std::size_t index = 0; index < width; ++index) {
        assert(close(native_state[index], scalar_state[index]));
    }

    auto repeated_state = initial;
    const float repeated = adi::detail::gated_delta_update(
        repeated_state,
        query,
        key,
        0.375F,
        0.625F,
        0.9375F);
    assert(repeated == native_result);
    assert(repeated_state == native_state);

    constexpr std::uint32_t rows = 64;
    std::vector<float> serial_state(static_cast<std::size_t>(rows) * width);
    std::vector<float> parallel_state(serial_state.size());
    for (auto &value : serial_state) {
        value = static_cast<float>(
                    static_cast<std::int32_t>(next_random(random) % 2049) -
                    1024) /
                4096.0F;
    }
    parallel_state = serial_state;
    std::vector<float> serial_output(rows);
    std::vector<float> parallel_output(rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
        serial_output[row] = adi::detail::gated_delta_update(
            std::span<float>(serial_state).subspan(
                static_cast<std::size_t>(row) * width,
                width),
            query,
            key,
            static_cast<float>(row + 1) / 128.0F,
            0.5F,
            0.984375F);
    }
    adi::detail::parallel_ranges(
        rows,
        1,
        [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t row = begin; row < end; ++row) {
                parallel_output[row] = adi::detail::gated_delta_update(
                    std::span<float>(parallel_state).subspan(
                        static_cast<std::size_t>(row) * width,
                        width),
                    query,
                    key,
                    static_cast<float>(row + 1) / 128.0F,
                    0.5F,
                    0.984375F);
            }
        });
    assert(parallel_output == serial_output);
    assert(parallel_state == serial_state);

    bool rejected = false;
    try {
        (void)adi::detail::gated_delta_update(
            std::span<float>(parallel_state).first(width - 1),
            query,
            key,
            0.0F,
            0.0F,
            1.0F);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    adi::detail::clear_cpu_isa_for_testing();
}
