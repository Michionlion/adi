#include "adi/generation.hpp"

#include <cassert>
#include <random>
#include <vector>

int main() {
    std::mt19937_64 random(42);
    const std::vector<float> logits{1.0F, 4.0F, 2.0F};
    assert(adi::sample_token(logits, 0.0F, 1.0F, random) == 1);
    for (int index = 0; index < 20; ++index) {
        assert(adi::sample_token(logits, 0.1F, 0.1F, random) == 1);
    }
}
