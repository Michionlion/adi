#include "adi/executor.hpp"

#include <cassert>
#include <cmath>
#include <vector>

int main() {
    const adi::Backend &backend = adi::cpu_backend();
    assert(backend.name == "cpu");
    assert(backend.expert_matvec != nullptr);
    assert(backend.ne_matvec != nullptr);
    assert(backend.embedding_row != nullptr);
    assert(backend.head_matvec != nullptr);
    assert(backend.dense_bf16_matvec != nullptr);
    assert(backend.normalize_rms != nullptr);

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
