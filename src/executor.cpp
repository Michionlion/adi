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

} // namespace adi
