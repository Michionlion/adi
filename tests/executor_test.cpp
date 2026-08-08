#include "adi/executor.hpp"
#include "attention.hpp"
#include "simd.hpp"

#include <algorithm>
#include <array>
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
    constexpr std::uint32_t tokens = 67;
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
        head_size,
        false,
        online);
    std::vector<float> parallel_online(queries.size());
    adi::detail::grouped_query_online_attention(
        queries,
        keys,
        values,
        query_heads,
        kv_heads,
        head_size,
        head_size,
        true,
        parallel_online);
    assert(parallel_online == online);

    // Cross the paired-head threshold with the production head grouping.
    // The serial grouped loop is the bit-exact reference for each head.
    constexpr std::uint32_t paired_query_heads = 16;
    constexpr std::uint32_t paired_kv_heads = 2;
    constexpr std::uint32_t paired_head_size = 7;
    constexpr std::uint32_t paired_stride = 9;
    constexpr std::uint32_t paired_tokens = 4096;
    std::vector<float> paired_queries(
        paired_query_heads * paired_stride);
    std::vector<float> paired_keys(
        paired_tokens * paired_kv_heads * paired_head_size);
    std::vector<float> paired_values(paired_keys.size());
    for (std::size_t index = 0; index < paired_queries.size(); ++index) {
        paired_queries[index] =
            std::sin(static_cast<float>(index) * 0.03125F);
    }
    for (std::size_t index = 0; index < paired_keys.size(); ++index) {
        paired_keys[index] =
            std::cos(static_cast<float>(index) * 0.0078125F);
        paired_values[index] =
            std::sin(static_cast<float>(index) * 0.00390625F);
    }
    std::vector<float> paired_reference(
        paired_query_heads * paired_head_size);
    adi::detail::grouped_query_online_attention(
        paired_queries,
        paired_keys,
        paired_values,
        paired_query_heads,
        paired_kv_heads,
        paired_head_size,
        paired_stride,
        false,
        paired_reference);
    std::vector<float> paired_output(paired_reference.size());
    adi::detail::grouped_query_online_attention(
        paired_queries,
        paired_keys,
        paired_values,
        paired_query_heads,
        paired_kv_heads,
        paired_head_size,
        paired_stride,
        true,
        paired_output);
    assert(paired_output == paired_reference);

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

    // Compare against the former two-exponential online update exactly. The
    // score sequence makes each head take both the new-maximum and existing-
    // maximum branches several times.
    constexpr std::uint32_t branch_heads = 2;
    constexpr std::uint32_t branch_head_size = 8;
    constexpr std::uint32_t branch_tokens = 6;
    const std::vector<float> branch_queries{
        1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
        -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
    };
    const std::array<float, branch_tokens> key_levels{
        0.0F, 1.0F, -1.0F, 2.0F, 1.0F, 3.0F};
    std::vector<float> branch_keys(branch_tokens * branch_head_size);
    std::vector<float> branch_values(branch_keys.size());
    for (std::uint32_t token = 0; token < branch_tokens; ++token) {
        for (std::uint32_t index = 0; index < branch_head_size; ++index) {
            const auto offset =
                static_cast<std::size_t>(token) * branch_head_size + index;
            branch_keys[offset] = key_levels[token];
            branch_values[offset] =
                static_cast<float>(token * branch_head_size + index) * 0.03125F;
        }
    }
    std::vector<float> one_exp(branch_heads * branch_head_size);
    adi::detail::grouped_query_online_attention(
        branch_queries,
        branch_keys,
        branch_values,
        branch_heads,
        1,
        branch_head_size,
        branch_head_size,
        false,
        one_exp);
    std::vector<float> two_exp(one_exp.size());
    const auto branch_dot = adi::detail::selected_f32_dot_kernel();
    const float branch_scale =
        1.0F / std::sqrt(static_cast<float>(branch_head_size));
    for (std::uint32_t head = 0; head < branch_heads; ++head) {
        const auto query = std::span<const float>(branch_queries).subspan(
            static_cast<std::size_t>(head) * branch_head_size,
            branch_head_size);
        auto attended = std::span<float>(two_exp).subspan(
            static_cast<std::size_t>(head) * branch_head_size,
            branch_head_size);
        float online_maximum = -std::numeric_limits<float>::infinity();
        float online_denominator = 0.0F;
        for (std::uint32_t token = 0; token < branch_tokens; ++token) {
            const auto key = std::span<const float>(branch_keys).subspan(
                static_cast<std::size_t>(token) * branch_head_size,
                branch_head_size);
            const auto value = std::span<const float>(branch_values).subspan(
                static_cast<std::size_t>(token) * branch_head_size,
                branch_head_size);
            const float score = branch_dot(query, key) * branch_scale;
            const float next_maximum = std::max(online_maximum, score);
            const float previous_scale =
                std::exp(online_maximum - next_maximum);
            const float weight = std::exp(score - next_maximum);
            for (std::uint32_t index = 0; index < branch_head_size; ++index) {
                attended[index] = attended[index] * previous_scale +
                                  value[index] * weight;
            }
            online_denominator =
                online_denominator * previous_scale + weight;
            online_maximum = next_maximum;
        }
        const float inverse = 1.0F / online_denominator;
        for (auto &value : attended) {
            value *= inverse;
        }
    }
    assert(one_exp == two_exp);
}
