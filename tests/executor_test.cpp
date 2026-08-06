#include "adi/executor.hpp"
#include "attention.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

int main() {
    const adi::Backend &backend = adi::cpu_backend();
    assert(
        backend.name == "scalar" ||
        backend.name == "avx2" ||
        backend.name == "avx512" ||
        backend.name == "neon" ||
        backend.name == "sve");
    assert(backend.expert_matvec != nullptr);
    assert(backend.expert_matmul != nullptr);
    assert(backend.ne_matvec != nullptr);
    assert(backend.embedding_row != nullptr);
    assert(backend.head_matvec != nullptr);
    assert(backend.dense_bf16_matvec != nullptr);
    assert(backend.dense_bf16_matmul != nullptr);
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
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> all_probabilities(logits.size());
    float all_sum = 0.0F;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        all_probabilities[index] = std::exp(logits[index] - maximum);
        all_sum += all_probabilities[index];
    }
    for (auto &probability : all_probabilities) {
        probability /= all_sum;
    }
    float selected_sum = 0.0F;
    for (const auto route : routes) {
        selected_sum += all_probabilities[route.expert];
    }
    for (const auto route : routes) {
        const float reference =
            all_probabilities[route.expert] / selected_sum;
        assert(std::abs(reference - route.weight) < 2.0e-7F);
    }

    for (std::uint32_t value_head = 0; value_head < 32; ++value_head) {
        assert(adi::gated_delta_key_head(value_head, 32, 16) ==
               value_head / 2);
    }

    constexpr std::uint32_t query_heads = 4;
    constexpr std::uint32_t kv_heads = 2;
    constexpr std::uint32_t head_size = 5;
    constexpr std::uint32_t tokens = 7;
    std::vector<float> queries(query_heads * head_size);
    std::vector<float> keys(tokens * kv_heads * head_size);
    std::vector<float> values(keys.size());
    for (std::size_t index = 0; index < queries.size(); ++index) {
        queries[index] = std::sin(static_cast<float>(index) * 0.17F);
    }
    for (std::size_t index = 0; index < keys.size(); ++index) {
        keys[index] = std::cos(static_cast<float>(index) * 0.11F);
        values[index] = std::sin(static_cast<float>(index) * 0.07F);
    }
    std::vector<float> online(queries.size());
    adi::detail::grouped_query_online_attention(
        queries,
        keys,
        values,
        query_heads,
        kv_heads,
        head_size,
        online);

    std::vector<float> reference(queries.size());
    std::vector<float> scores(tokens);
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));
    for (std::uint32_t query_head = 0;
         query_head < query_heads;
         ++query_head) {
        const auto kv_head = query_head / (query_heads / kv_heads);
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::uint32_t token = 0; token < tokens; ++token) {
            float score = 0.0F;
            for (std::uint32_t index = 0; index < head_size; ++index) {
                score +=
                    queries[query_head * head_size + index] *
                    keys[(token * kv_heads + kv_head) * head_size + index];
            }
            scores[token] = score * scale;
            maximum = std::max(maximum, scores[token]);
        }
        float denominator = 0.0F;
        for (auto &score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (std::uint32_t token = 0; token < tokens; ++token) {
            const float probability = scores[token] / denominator;
            for (std::uint32_t index = 0; index < head_size; ++index) {
                reference[query_head * head_size + index] +=
                    probability *
                    values[(token * kv_heads + kv_head) * head_size + index];
            }
        }
    }
    for (std::size_t index = 0; index < reference.size(); ++index) {
        assert(std::abs(reference[index] - online[index]) < 2.0e-6F);
    }
}
