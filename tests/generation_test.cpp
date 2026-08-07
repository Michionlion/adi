#include "adi/generation.hpp"
#include "generation_internal.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

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
