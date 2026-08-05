#include "adi/executor.hpp"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    std::vector<float> logits(16);
    for (std::size_t index = 0; index < logits.size(); ++index) {
        logits[index] = static_cast<float>(index);
    }
    const auto routes = adi::top_experts(logits);
    float sum = 0.0F;
    for (std::size_t index = 0; index < routes.size(); ++index) {
        assert(routes[index].expert == 15 - index);
        assert(routes[index].weight > 0.0F);
        sum += routes[index].weight;
    }
    assert(std::abs(sum - 1.0F) < 1e-6F);
}
