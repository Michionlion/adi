#include "adi/generation.hpp"

#include <cassert>
#include <cmath>
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
}
