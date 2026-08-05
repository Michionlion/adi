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

void linear_attention_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> input,
    std::span<float> output,
    LinearAttentionState &state,
    LinearAttentionScratch &scratch) {
    constexpr std::uint32_t key_heads = 16;
    constexpr std::uint32_t value_heads = 32;
    constexpr std::uint32_t head_size = 128;
    constexpr std::uint32_t qk_size = key_heads * head_size;
    constexpr std::uint32_t value_size = value_heads * head_size;
    constexpr std::uint32_t channels = 2 * qk_size + value_size;
    constexpr std::uint32_t convolution_width = 4;
    constexpr float epsilon = 1.0e-6F;
    if (input.size() != model.config().hidden ||
        output.size() != model.config().hidden) {
        throw std::invalid_argument("linear attention input shape mismatch");
    }
    const auto prefix = layer_prefix(layer) + "linear_attn.";
    scratch.qkv.resize(channels);
    scratch.gate.resize(value_size);
    scratch.convolved.resize(channels);
    scratch.alpha.resize(value_heads);
    scratch.beta.resize(value_heads);
    scratch.recurrent_output.resize(value_size);
    scratch.normalized.resize(value_size);
    mach_ne_matvec(
        model.non_expert(layer, prefix + "in_proj_qkv.weight"),
        input,
        scratch.qkv,
        scratch.codec);
    mach_ne_matvec(
        model.non_expert(layer, prefix + "in_proj_z.weight"),
        input,
        scratch.gate,
        scratch.codec);
    bf16_matvec(
        model.bf16_matrix(prefix + "in_proj_a.weight"),
        input,
        scratch.alpha);
    bf16_matvec(
        model.bf16_matrix(prefix + "in_proj_b.weight"),
        input,
        scratch.beta);

    const auto convolution_weights =
        model.bf16_data(prefix + "conv1d.weight");
    if (convolution_weights.size() !=
        static_cast<std::size_t>(channels) * convolution_width) {
        throw std::runtime_error("linear attention convolution shape mismatch");
    }
    if (state.convolution.empty()) {
        state.convolution.assign(
            static_cast<std::size_t>(channels) * (convolution_width - 1), 0.0F);
    }
    if (state.convolution.size() !=
        static_cast<std::size_t>(channels) * (convolution_width - 1)) {
        throw std::invalid_argument("linear attention convolution state mismatch");
    }
    for (std::uint32_t channel = 0; channel < channels; ++channel) {
        const auto history = static_cast<std::size_t>(channel) * 3;
        const auto weights = static_cast<std::size_t>(channel) * convolution_width;
        float value = 0.0F;
        for (std::uint32_t index = 0; index < 3; ++index) {
            value += state.convolution[history + index] *
                     bf16_to_f32(convolution_weights[weights + index]);
        }
        value += scratch.qkv[channel] *
                 bf16_to_f32(convolution_weights[weights + 3]);
        scratch.convolved[channel] = silu(value);
        state.convolution[history] = state.convolution[history + 1];
        state.convolution[history + 1] = state.convolution[history + 2];
        state.convolution[history + 2] = scratch.qkv[channel];
    }

    auto query = std::span<float>(scratch.convolved.data(), qk_size);
    auto key = std::span<float>(scratch.convolved.data() + qk_size, qk_size);
    auto value =
        std::span<float>(scratch.convolved.data() + 2 * qk_size, value_size);
    for (std::uint32_t head = 0; head < key_heads; ++head) {
        for (auto vector : {
                 query.subspan(static_cast<std::size_t>(head) * head_size, head_size),
                 key.subspan(static_cast<std::size_t>(head) * head_size, head_size)}) {
            float squares = 0.0F;
            for (const auto component : vector) {
                squares += component * component;
            }
            const float scale = 1.0F / std::max(std::sqrt(squares), epsilon);
            for (auto &component : vector) {
                component *= scale;
            }
        }
    }

    const auto alpha_bias = model.bf16_vector(prefix + "dt_bias");
    const auto a_log = model.bf16_vector(prefix + "A_log");
    if (alpha_bias.size() != value_heads || a_log.size() != value_heads) {
        throw std::runtime_error("linear attention decay shape mismatch");
    }
    if (state.recurrent.empty()) {
        state.recurrent.assign(
            static_cast<std::size_t>(value_heads) * head_size * head_size, 0.0F);
    }
    if (state.recurrent.size() !=
        static_cast<std::size_t>(value_heads) * head_size * head_size) {
        throw std::invalid_argument("linear attention recurrent state mismatch");
    }

    const float output_scale = 1.0F / std::sqrt(static_cast<float>(head_size));
    for (std::uint32_t head = 0; head < value_heads; ++head) {
        const auto key_head = head % key_heads;
        const auto q = query.subspan(
            static_cast<std::size_t>(key_head) * head_size, head_size);
        const auto k = key.subspan(
            static_cast<std::size_t>(key_head) * head_size, head_size);
        const auto v = value.subspan(
            static_cast<std::size_t>(head) * head_size, head_size);
        const float beta = sigmoid(scratch.beta[head]);
        const float biased_alpha = scratch.alpha[head] + bf16_to_f32(alpha_bias[head]);
        const float softplus =
            biased_alpha > 20.0F ? biased_alpha : std::log1p(std::exp(biased_alpha));
        const float decay =
            std::exp(-std::exp(bf16_to_f32(a_log[head])) * softplus);
        for (std::uint32_t row = 0; row < head_size; ++row) {
            auto state_row = std::span<float>(
                state.recurrent.data() +
                    (static_cast<std::size_t>(head) * head_size + row) * head_size,
                head_size);
            float prediction = 0.0F;
            for (std::uint32_t column = 0; column < head_size; ++column) {
                state_row[column] *= decay;
                prediction += state_row[column] * k[column];
            }
            const float delta = (v[row] - prediction) * beta;
            float attended = 0.0F;
            for (std::uint32_t column = 0; column < head_size; ++column) {
                state_row[column] += k[column] * delta;
                attended += state_row[column] * q[column];
            }
            scratch.recurrent_output[
                static_cast<std::size_t>(head) * head_size + row] =
                attended * output_scale;
        }
    }

    const auto norm = model.bf16_vector(prefix + "norm.weight");
    for (std::uint32_t head = 0; head < value_heads; ++head) {
        const auto source = std::span<const float>(
            scratch.recurrent_output.data() +
                static_cast<std::size_t>(head) * head_size,
            head_size);
        auto destination = std::span<float>(
            scratch.normalized.data() + static_cast<std::size_t>(head) * head_size,
            head_size);
        rms_norm(source, norm, epsilon, destination);
        for (std::uint32_t index = 0; index < head_size; ++index) {
            destination[index] *= silu(
                scratch.gate[static_cast<std::size_t>(head) * head_size + index]);
        }
    }
    mach_ne_matvec(
        model.non_expert(layer, prefix + "out_proj.weight"),
        scratch.normalized,
        output,
        scratch.codec);
}

} // namespace adi
