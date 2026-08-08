#include "adi/generation.hpp"
#include "generation_internal.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

std::uint32_t reference_sample_token(
    const std::vector<float> &logits,
    float temperature,
    float top_p,
    std::mt19937_64 &random) {
    std::vector<std::uint32_t> indexes(logits.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::sort(
        indexes.begin(),
        indexes.end(),
        [&](std::uint32_t left, std::uint32_t right) {
            if (logits[left] == logits[right]) {
                return left < right;
            }
            return logits[left] > logits[right];
        });
    const double maximum = static_cast<double>(logits[indexes[0]]);
    std::vector<double> probabilities;
    probabilities.reserve(logits.size());
    double denominator = 0.0;
    for (const auto index : indexes) {
        const double scaled =
            (static_cast<double>(logits[index]) - maximum) /
            static_cast<double>(temperature);
        const double probability = std::exp(scaled);
        probabilities.push_back(probability);
        denominator += probability;
    }
    double cumulative = 0.0;
    std::size_t retained = 0;
    for (; retained < probabilities.size(); ++retained) {
        cumulative += probabilities[retained] / denominator;
        if (cumulative >= top_p) {
            ++retained;
            break;
        }
    }
    retained = std::max<std::size_t>(1, retained);
    const double retained_sum = std::accumulate(
        probabilities.begin(), probabilities.begin() + retained, 0.0);
    std::uniform_real_distribution<double> distribution(0.0, retained_sum);
    const double target = distribution(random);
    double running = 0.0;
    for (std::size_t index = 0; index < retained; ++index) {
        running += probabilities[index];
        if (target <= running) {
            return indexes[index];
        }
    }
    return indexes[retained - 1];
}

} // namespace

int main() {
    std::mt19937_64 random(42);
    const std::vector<float> logits{1.0F, 4.0F, 2.0F};
    assert(adi::sample_token(logits, 0.0F, 1.0F, random) == 1);
    for (int index = 0; index < 20; ++index) {
        assert(adi::sample_token(logits, 0.1F, 0.1F, random) == 1);
    }

    bool rejected = false;
    try {
        (void)adi::sample_token(logits, 1.0e-38F, 0.9F, random);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    auto invalid_logits = logits;
    invalid_logits[0] = NAN;
    rejected = false;
    try {
        (void)adi::sample_token(invalid_logits, 0.7F, 0.9F, random);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    std::vector<float> comparison_logits(4097);
    std::mt19937 comparison_values(7);
    std::uniform_int_distribution<int> logit_values(-80, 80);
    for (auto &value : comparison_logits) {
        value = static_cast<float>(logit_values(comparison_values)) * 0.125F;
    }
    comparison_logits[17] = 0.0F;
    comparison_logits[18] = -0.0F;
    for (const float comparison_temperature : {0.1F, 0.7F, 2.0F}) {
        for (const float comparison_top_p : {0.1F, 0.9F, 1.0F}) {
            for (std::uint64_t seed = 0; seed < 64; ++seed) {
                std::mt19937_64 expected_random(seed);
                std::mt19937_64 actual_random(seed);
                const auto expected = reference_sample_token(
                    comparison_logits,
                    comparison_temperature,
                    comparison_top_p,
                    expected_random);
                const auto actual = adi::sample_token(
                    comparison_logits,
                    comparison_temperature,
                    comparison_top_p,
                    actual_random);
                assert(actual == expected);
                assert(actual_random() == expected_random());
            }
        }
    }
    std::vector<float> signed_zero_logits(64, 0.0F);
    for (std::size_t index = 1; index < signed_zero_logits.size(); index += 2) {
        signed_zero_logits[index] = -0.0F;
    }
    for (std::uint64_t seed = 0; seed < 128; ++seed) {
        std::mt19937_64 expected_random(seed);
        std::mt19937_64 actual_random(seed);
        assert(
            adi::sample_token(
                signed_zero_logits, 0.7F, 1.0F, actual_random) ==
            reference_sample_token(
                signed_zero_logits, 0.7F, 1.0F, expected_random));
        assert(actual_random() == expected_random());
    }

    using adi::generation_detail::resolved_output_limit;
    assert(resolved_output_limit(16, 4, std::nullopt) == 12);
    assert(resolved_output_limit(16, 4, std::uint32_t{12}) == 12);
    assert(resolved_output_limit(16, 15, std::nullopt) == 1);
    assert(resolved_output_limit(16, 15, std::uint32_t{1}) == 1);

    const auto rejects_limit = [](
        std::size_t prompt_tokens,
        std::optional<std::uint32_t> limit) {
        try {
            (void)resolved_output_limit(16, prompt_tokens, limit);
            return false;
        } catch (const std::invalid_argument &) {
            return true;
        }
    };
    assert(rejects_limit(16, std::nullopt));
    assert(rejects_limit(4, std::uint32_t{0}));
    assert(rejects_limit(15, std::uint32_t{2}));
}
