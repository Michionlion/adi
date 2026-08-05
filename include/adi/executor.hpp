#pragma once

#include "adi/model.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace adi {

struct ExpertRoute {
    std::uint32_t expert;
    float weight;
};

struct MoeScratch {
    ExpertScratch codec;
    std::vector<float> router_logits;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> activated;
    std::vector<float> projected;
    std::vector<float> shared_gate;
    std::vector<float> shared_up;
    std::vector<float> shared_down;
};

struct FullAttentionState {
    std::vector<float> keys;
    std::vector<float> values;
};

struct FullAttentionScratch {
    ExpertScratch codec;
    std::vector<float> query_gate;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> attended;
    std::vector<float> scores;
};

struct LinearAttentionState {
    std::vector<float> convolution;
    std::vector<float> recurrent;
};

struct LinearAttentionScratch {
    ExpertScratch codec;
    std::vector<float> qkv;
    std::vector<float> gate;
    std::vector<float> convolved;
    std::vector<float> alpha;
    std::vector<float> beta;
    std::vector<float> recurrent_output;
    std::vector<float> normalized;
};

[[nodiscard]] std::array<ExpertRoute, 8> top_experts(
    std::span<const float> logits);

[[nodiscard]] std::array<ExpertRoute, 8> moe_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> input,
    std::span<float> output,
    MoeScratch &scratch);

void full_attention_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::uint32_t position,
    std::span<const float> input,
    std::span<float> output,
    FullAttentionState &state,
    FullAttentionScratch &scratch);

void linear_attention_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> input,
    std::span<float> output,
    LinearAttentionState &state,
    LinearAttentionScratch &scratch);

} // namespace adi
