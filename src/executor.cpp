#include "adi/executor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace adi {
namespace {

std::string layer_prefix(std::uint32_t layer) {
    return "model.language_model.layers." + std::to_string(layer) + ".";
}

void normalize_head(
    std::span<float> values,
    std::span<const std::uint16_t> weights,
    float epsilon) {
    float squares = 0.0F;
    for (const auto value : values) {
        squares += value * value;
    }
    const float scale =
        1.0F / std::sqrt(squares / static_cast<float>(values.size()) + epsilon);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] *= scale * bf16_to_f32(weights[index]);
    }
}

void apply_rope(std::span<float> values, std::uint32_t position) {
    constexpr std::uint32_t rotary_dimensions = 64;
    constexpr std::uint32_t half = rotary_dimensions / 2;
    constexpr float theta_base = 10000000.0F;
    for (std::uint32_t index = 0; index < half; ++index) {
        const float theta =
            static_cast<float>(position) /
            std::pow(theta_base, 2.0F * static_cast<float>(index) /
                                     static_cast<float>(rotary_dimensions));
        const float cosine = std::cos(theta);
        const float sine = std::sin(theta);
        const float first = values[index];
        const float second = values[index + half];
        values[index] = first * cosine - second * sine;
        values[index + half] = second * cosine + first * sine;
    }
}

} // namespace

std::array<ExpertRoute, 8> top_experts(std::span<const float> logits) {
    if (logits.size() < 8) {
        throw std::invalid_argument("MoE routing requires at least eight experts");
    }
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities(logits.size());
    float denominator = 0.0F;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = std::exp(logits[index] - maximum);
        denominator += probabilities[index];
    }
    for (auto &probability : probabilities) {
        probability /= denominator;
    }

    std::vector<std::uint32_t> indexes(logits.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::partial_sort(
        indexes.begin(),
        indexes.begin() + 8,
        indexes.end(),
        [&](std::uint32_t left, std::uint32_t right) {
            if (probabilities[left] == probabilities[right]) {
                return left < right;
            }
            return probabilities[left] > probabilities[right];
        });

    float selected_sum = 0.0F;
    for (std::size_t index = 0; index < 8; ++index) {
        selected_sum += probabilities[indexes[index]];
    }
    selected_sum = std::max(selected_sum, 6.103515625e-5F);
    std::array<ExpertRoute, 8> routes;
    for (std::size_t index = 0; index < routes.size(); ++index) {
        routes[index] = {
            indexes[index],
            probabilities[indexes[index]] / selected_sum,
        };
    }
    return routes;
}

std::array<ExpertRoute, 8> moe_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> input,
    std::span<float> output,
    MoeScratch &scratch) {
    const auto &config = model.config();
    if (input.size() != config.hidden || output.size() != config.hidden) {
        throw std::invalid_argument("MoE input shape mismatch");
    }
    const auto prefix = layer_prefix(layer);
    const auto router =
        model.bf16_matrix(prefix + "mlp.gate.weight");
    scratch.router_logits.resize(config.experts);
    bf16_matvec(router, input, scratch.router_logits);
    const auto routes = top_experts(scratch.router_logits);

    std::fill(output.begin(), output.end(), 0.0F);
    scratch.gate.resize(config.expert_hidden);
    scratch.up.resize(config.expert_hidden);
    scratch.activated.resize(config.expert_hidden);
    scratch.projected.resize(config.hidden);
    for (const auto route : routes) {
        mach_expert_matvec(
            model.expert(layer, route.expert, ExpertProjection::gate),
            input,
            scratch.gate,
            scratch.codec);
        mach_expert_matvec(
            model.expert(layer, route.expert, ExpertProjection::up),
            input,
            scratch.up,
            scratch.codec);
        for (std::size_t index = 0; index < scratch.activated.size(); ++index) {
            scratch.activated[index] = silu(scratch.gate[index]) * scratch.up[index];
        }
        mach_expert_matvec(
            model.expert(layer, route.expert, ExpertProjection::down),
            scratch.activated,
            scratch.projected,
            scratch.codec);
        for (std::size_t index = 0; index < output.size(); ++index) {
            output[index] += route.weight * scratch.projected[index];
        }
    }

    scratch.shared_gate.resize(config.expert_hidden);
    scratch.shared_up.resize(config.expert_hidden);
    scratch.shared_down.resize(config.hidden);
    const auto shared_prefix = prefix + "mlp.shared_expert.";
    mach_ne_matvec(
        model.non_expert(layer, shared_prefix + "gate_proj.weight"),
        input,
        scratch.shared_gate,
        scratch.codec);
    mach_ne_matvec(
        model.non_expert(layer, shared_prefix + "up_proj.weight"),
        input,
        scratch.shared_up,
        scratch.codec);
    for (std::size_t index = 0; index < scratch.shared_gate.size(); ++index) {
        scratch.shared_gate[index] =
            silu(scratch.shared_gate[index]) * scratch.shared_up[index];
    }
    mach_ne_matvec(
        model.non_expert(layer, shared_prefix + "down_proj.weight"),
        scratch.shared_gate,
        scratch.shared_down,
        scratch.codec);

    const auto shared_gate =
        model.bf16_matrix(prefix + "mlp.shared_expert_gate.weight");
    float shared_weight = 0.0F;
    bf16_matvec(shared_gate, input, std::span<float>(&shared_weight, 1));
    shared_weight = sigmoid(shared_weight);
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] += shared_weight * scratch.shared_down[index];
    }
    return routes;
}

void full_attention_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::uint32_t position,
    std::span<const float> input,
    std::span<float> output,
    FullAttentionState &state,
    FullAttentionScratch &scratch) {
    constexpr std::uint32_t query_heads = 16;
    constexpr std::uint32_t kv_heads = 2;
    constexpr std::uint32_t head_size = 256;
    constexpr std::uint32_t query_stride = 2 * head_size;
    constexpr float epsilon = 1.0e-6F;
    const auto &config = model.config();
    if (input.size() != config.hidden || output.size() != config.hidden ||
        state.keys.size() != state.values.size() ||
        state.keys.size() != static_cast<std::size_t>(position) * kv_heads * head_size) {
        throw std::invalid_argument("full attention state or input shape mismatch");
    }
    const auto prefix = layer_prefix(layer) + "self_attn.";
    scratch.query_gate.resize(query_heads * query_stride);
    scratch.key.resize(kv_heads * head_size);
    scratch.value.resize(kv_heads * head_size);
    scratch.attended.resize(query_heads * head_size);
    mach_ne_matvec(
        model.non_expert(layer, prefix + "q_proj.weight"),
        input,
        scratch.query_gate,
        scratch.codec);
    mach_ne_matvec(
        model.non_expert(layer, prefix + "k_proj.weight"),
        input,
        scratch.key,
        scratch.codec);
    mach_ne_matvec(
        model.non_expert(layer, prefix + "v_proj.weight"),
        input,
        scratch.value,
        scratch.codec);

    const auto query_norm = model.bf16_vector(prefix + "q_norm.weight");
    const auto key_norm = model.bf16_vector(prefix + "k_norm.weight");
    for (std::uint32_t head = 0; head < query_heads; ++head) {
        auto query = std::span<float>(
            scratch.query_gate.data() + static_cast<std::size_t>(head) * query_stride,
            head_size);
        normalize_head(query, query_norm, epsilon);
        apply_rope(query, position);
    }
    for (std::uint32_t head = 0; head < kv_heads; ++head) {
        auto key = std::span<float>(
            scratch.key.data() + static_cast<std::size_t>(head) * head_size,
            head_size);
        normalize_head(key, key_norm, epsilon);
        apply_rope(key, position);
    }
    state.keys.insert(state.keys.end(), scratch.key.begin(), scratch.key.end());
    state.values.insert(state.values.end(), scratch.value.begin(), scratch.value.end());

    scratch.scores.resize(static_cast<std::size_t>(position) + 1);
    for (std::uint32_t query_head = 0; query_head < query_heads; ++query_head) {
        const auto kv_head = query_head / (query_heads / kv_heads);
        const auto query = std::span<const float>(
            scratch.query_gate.data() +
                static_cast<std::size_t>(query_head) * query_stride,
            head_size);
        float maximum = -INFINITY;
        for (std::uint32_t token = 0; token <= position; ++token) {
            const auto key_offset =
                (static_cast<std::size_t>(token) * kv_heads + kv_head) * head_size;
            float score = 0.0F;
            for (std::uint32_t index = 0; index < head_size; ++index) {
                score += query[index] * state.keys[key_offset + index];
            }
            score /= std::sqrt(static_cast<float>(head_size));
            scratch.scores[token] = score;
            maximum = std::max(maximum, score);
        }
        float denominator = 0.0F;
        for (std::uint32_t token = 0; token <= position; ++token) {
            scratch.scores[token] = std::exp(scratch.scores[token] - maximum);
            denominator += scratch.scores[token];
        }
        auto attended = std::span<float>(
            scratch.attended.data() +
                static_cast<std::size_t>(query_head) * head_size,
            head_size);
        std::fill(attended.begin(), attended.end(), 0.0F);
        for (std::uint32_t token = 0; token <= position; ++token) {
            const float probability = scratch.scores[token] / denominator;
            const auto value_offset =
                (static_cast<std::size_t>(token) * kv_heads + kv_head) * head_size;
            for (std::uint32_t index = 0; index < head_size; ++index) {
                attended[index] += probability * state.values[value_offset + index];
            }
        }
        const auto gate_offset =
            static_cast<std::size_t>(query_head) * query_stride + head_size;
        for (std::uint32_t index = 0; index < head_size; ++index) {
            attended[index] *= sigmoid(scratch.query_gate[gate_offset + index]);
        }
    }
    mach_ne_matvec(
        model.non_expert(layer, prefix + "o_proj.weight"),
        scratch.attended,
        output,
        scratch.codec);
}

} // namespace adi
