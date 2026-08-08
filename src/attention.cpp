#include "attention.hpp"

#include "parallel.hpp"
#include "simd.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace adi::detail {
namespace {

void update_online_softmax(
    float score,
    std::span<const float> value,
    float &maximum,
    float &denominator,
    std::span<float> attended) {
    if (score <= maximum) {
        const float weight = std::exp(score - maximum);
        denominator += weight;
        for (std::size_t index = 0; index < attended.size(); ++index) {
            attended[index] += value[index] * weight;
        }
        return;
    }

    const float previous_scale = std::exp(maximum - score);
    denominator = denominator * previous_scale + 1.0F;
    for (std::size_t index = 0; index < attended.size(); ++index) {
        attended[index] = attended[index] * previous_scale + value[index];
    }
    maximum = score;
}

} // namespace

void grouped_query_online_attention(
    std::span<const float> queries,
    std::span<const float> keys,
    std::span<const float> values,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_size,
    std::uint32_t query_stride,
    bool parallel_query_heads,
    std::span<float> output) {
    constexpr std::uint32_t maximum_heads_per_group = 8;
    constexpr std::size_t minimum_parallel_tokens = 64;
    if (query_heads == 0 || kv_heads == 0 || head_size == 0 ||
        query_stride < head_size ||
        query_heads % kv_heads != 0 ||
        query_heads / kv_heads > maximum_heads_per_group ||
        queries.size() !=
            static_cast<std::size_t>(query_heads) * query_stride ||
        keys.size() != values.size() ||
        keys.size() % (static_cast<std::size_t>(kv_heads) * head_size) != 0 ||
        output.size() != static_cast<std::size_t>(query_heads) * head_size) {
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
    const auto dot = selected_f32_dot_kernel();
    std::fill(output.begin(), output.end(), 0.0F);
    const auto attend_heads =
        [&](std::uint32_t head_begin, std::uint32_t head_end) {
            std::array<float, maximum_heads_per_group> maxima;
            std::array<float, maximum_heads_per_group> denominators;
            for (std::uint32_t kv_head = head_begin;
                 kv_head < head_end;
                 ++kv_head) {
                std::fill_n(
                    maxima.begin(),
                    heads_per_group,
                    -std::numeric_limits<float>::infinity());
                std::fill_n(
                    denominators.begin(), heads_per_group, 0.0F);

                for (std::size_t token = 0; token < tokens; ++token) {
                    const auto state_offset =
                        (token * kv_heads + kv_head) * head_size;
                    const auto key = keys.subspan(state_offset, head_size);
                    const auto value =
                        values.subspan(state_offset, head_size);
                    for (std::uint32_t group_head = 0;
                         group_head < heads_per_group;
                         ++group_head) {
                        const auto query_head =
                            kv_head * heads_per_group + group_head;
                        const auto query = queries.subspan(
                            static_cast<std::size_t>(query_head) *
                                query_stride,
                            head_size);
                        const float score = dot(query, key) * score_scale;
                        auto attended = output.subspan(
                            static_cast<std::size_t>(query_head) * head_size,
                            head_size);
                        update_online_softmax(
                            score,
                            value,
                            maxima[group_head],
                            denominators[group_head],
                            attended);
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
                    const float inverse =
                        1.0F / denominators[group_head];
                    for (auto &value : attended) {
                        value *= inverse;
                    }
                }
            }
        };
    if (parallel_query_heads && tokens >= minimum_parallel_tokens) {
        parallel_ranges(
            query_heads,
            1,
            [&](std::uint32_t head_begin, std::uint32_t head_end) {
                for (std::uint32_t query_head = head_begin;
                     query_head < head_end;
                     ++query_head) {
                    const auto kv_head = query_head / heads_per_group;
                    const auto query = queries.subspan(
                        static_cast<std::size_t>(query_head) * query_stride,
                        head_size);
                    auto attended = output.subspan(
                        static_cast<std::size_t>(query_head) * head_size,
                        head_size);
                    float maximum =
                        -std::numeric_limits<float>::infinity();
                    float denominator = 0.0F;
                    for (std::size_t token = 0; token < tokens; ++token) {
                        const auto state_offset =
                            (token * kv_heads + kv_head) * head_size;
                        const auto key =
                            keys.subspan(state_offset, head_size);
                        const auto value =
                            values.subspan(state_offset, head_size);
                        const float score = dot(query, key) * score_scale;
                        update_online_softmax(
                            score,
                            value,
                            maximum,
                            denominator,
                            attended);
                    }
                    const float inverse = 1.0F / denominator;
                    for (auto &value : attended) {
                        value *= inverse;
                    }
                }
            });
    } else {
        attend_heads(0, kv_heads);
    }
}

} // namespace adi::detail
