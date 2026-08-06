#include "attention.hpp"

#include "simd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace adi::detail {

void grouped_query_online_attention(
    std::span<const float> queries,
    std::span<const float> keys,
    std::span<const float> values,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_size,
    std::span<float> output) {
    if (query_heads == 0 || kv_heads == 0 || head_size == 0 ||
        query_heads % kv_heads != 0 ||
        queries.size() != static_cast<std::size_t>(query_heads) * head_size ||
        keys.size() != values.size() ||
        keys.size() % (static_cast<std::size_t>(kv_heads) * head_size) != 0 ||
        output.size() != queries.size()) {
        throw std::invalid_argument("grouped attention shape mismatch");
    }
    const auto tokens =
        keys.size() / (static_cast<std::size_t>(kv_heads) * head_size);
    if (tokens == 0) {
        throw std::invalid_argument("grouped attention history is empty");
    }

    const auto heads_per_group = query_heads / kv_heads;
    const float score_scale =
        1.0F / std::sqrt(static_cast<float>(head_size));
    std::fill(output.begin(), output.end(), 0.0F);
    std::vector<float> maxima(heads_per_group);
    std::vector<float> denominators(heads_per_group);

    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
        std::fill(
            maxima.begin(),
            maxima.end(),
            -std::numeric_limits<float>::infinity());
        std::fill(denominators.begin(), denominators.end(), 0.0F);

        for (std::size_t token = 0; token < tokens; ++token) {
            const auto state_offset =
                (token * kv_heads + kv_head) * head_size;
            const auto key = keys.subspan(state_offset, head_size);
            const auto value = values.subspan(state_offset, head_size);
            for (std::uint32_t group_head = 0;
                 group_head < heads_per_group;
                 ++group_head) {
                const auto query_head =
                    kv_head * heads_per_group + group_head;
                const auto query = queries.subspan(
                    static_cast<std::size_t>(query_head) * head_size,
                    head_size);
                const float score = f32_dot(query, key) * score_scale;
                const float next_maximum =
                    std::max(maxima[group_head], score);
                const float previous_scale =
                    std::exp(maxima[group_head] - next_maximum);
                const float weight = std::exp(score - next_maximum);
                auto attended = output.subspan(
                    static_cast<std::size_t>(query_head) * head_size,
                    head_size);
                for (std::uint32_t index = 0; index < head_size; ++index) {
                    attended[index] =
                        attended[index] * previous_scale +
                        value[index] * weight;
                }
                denominators[group_head] =
                    denominators[group_head] * previous_scale + weight;
                maxima[group_head] = next_maximum;
            }
        }
        for (std::uint32_t group_head = 0;
             group_head < heads_per_group;
             ++group_head) {
            const auto query_head =
                kv_head * heads_per_group + group_head;
            auto attended = output.subspan(
                static_cast<std::size_t>(query_head) * head_size,
                head_size);
            const float inverse = 1.0F / denominators[group_head];
            for (auto &value : attended) {
                value *= inverse;
            }
        }
    }
}

} // namespace adi::detail
