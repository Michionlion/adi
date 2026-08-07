#include "adi/kernels.hpp"
#include "codec_cache.hpp"

#include <cassert>
#include <cstdint>
#include <span>
#include <stdexcept>
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
    constexpr std::uint32_t dimension = 16;
    std::uint32_t random = 0x9E3779B9U;

    std::vector<float> expert_tlut(32768 * 8);
    for (auto &value : expert_tlut) {
        value = static_cast<float>(
                    static_cast<std::int32_t>(next_random(random) % 4097) -
                    2048) /
                511.0F;
    }
    const auto expert_states =
        adi::detail::build_expert_state_values(expert_tlut);
    assert(expert_states.size() == 65536 * 8);

    std::vector<std::uint16_t> expert_trellis(24);
    for (auto &value : expert_trellis) {
        value = static_cast<std::uint16_t>(next_random(random));
    }
    std::vector<std::uint16_t> expert_su(dimension);
    std::vector<std::uint16_t> expert_sv(dimension);
    for (auto &value : expert_su) {
        value = adi::f32_to_f16(
            static_cast<float>(next_random(random) % 31 + 1) / 32.0F);
    }
    for (auto &value : expert_sv) {
        value = adi::f32_to_f16(
            static_cast<float>(next_random(random) % 31 + 1) / 32.0F);
    }
    std::vector<std::uint16_t> gamma_f16{
        adi::f32_to_f16(0.75F),
        adi::f32_to_f16(1.25F),
    };
    const auto wave_indexes = adi::detail::build_wave_indexes(1, 1);
    const auto gamma = adi::detail::convert_f16_values(gamma_f16);
    std::vector<float> input(dimension);
    for (auto &value : input) {
        value = static_cast<float>(
                    static_cast<std::int32_t>(next_random(random) % 2049) -
                    1024) /
                256.0F;
    }

    const adi::MachExpertMatrix uncached_expert{
        dimension,
        dimension,
        expert_trellis,
        expert_su,
        expert_sv,
        gamma_f16,
        expert_tlut,
    };
    const adi::MachExpertMatrix cached_expert{
        dimension,
        dimension,
        expert_trellis,
        expert_su,
        expert_sv,
        gamma_f16,
        expert_tlut,
        expert_states,
        wave_indexes,
        gamma,
    };
    std::vector<float> uncached_output(dimension);
    std::vector<float> cached_output(dimension);
    adi::ExpertScratch uncached_scratch;
    adi::ExpertScratch cached_scratch;
    adi::mach_expert_matvec(
        uncached_expert,
        input,
        uncached_output,
        uncached_scratch);
    adi::mach_expert_matvec(
        cached_expert,
        input,
        cached_output,
        cached_scratch);
    assert(cached_output == uncached_output);

    std::vector<float> expert_inputs(3 * dimension);
    for (std::size_t index = 0; index < expert_inputs.size(); ++index) {
        expert_inputs[index] =
            input[index % dimension] * static_cast<float>(index / dimension + 1);
    }
    std::vector<float> uncached_batch(3 * dimension);
    std::vector<float> cached_batch(3 * dimension);
    adi::mach_expert_matmul(
        uncached_expert,
        expert_inputs,
        3,
        uncached_batch,
        uncached_scratch);
    adi::mach_expert_matmul(
        cached_expert,
        expert_inputs,
        3,
        cached_batch,
        cached_scratch);
    assert(cached_batch == uncached_batch);

    std::vector<float> ne_tlut(512 * 2);
    for (auto &value : ne_tlut) {
        value = static_cast<float>(
                    static_cast<std::int32_t>(next_random(random) % 4097) -
                    2048) /
                511.0F;
    }
    const auto ne_states = adi::detail::build_ne_state_values(ne_tlut);
    assert(ne_states.size() == 65536 * 2);
    std::vector<std::uint16_t> ne_trellis(64);
    for (auto &value : ne_trellis) {
        value = static_cast<std::uint16_t>(next_random(random));
    }
    std::vector<std::int8_t> signs(dimension, 1);
    for (std::size_t index = 0; index < signs.size(); index += 3) {
        signs[index] = -1;
    }
    const adi::MachNeMatrix uncached_ne{
        dimension,
        dimension,
        ne_trellis,
        signs,
        signs,
        0.03125F,
        ne_tlut,
    };
    const adi::MachNeMatrix cached_ne{
        dimension,
        dimension,
        ne_trellis,
        signs,
        signs,
        0.03125F,
        ne_tlut,
        ne_states,
    };
    adi::mach_ne_matvec(
        uncached_ne,
        input,
        uncached_output,
        uncached_scratch);
    adi::mach_ne_matvec(
        cached_ne,
        input,
        cached_output,
        cached_scratch);
    assert(cached_output == uncached_output);

    auto invalid_cache = cached_expert;
    invalid_cache.state_values = std::span<const float>(expert_states).first(8);
    bool rejected = false;
    try {
        adi::mach_expert_matvec(
            invalid_cache,
            input,
            cached_output,
            cached_scratch);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}
