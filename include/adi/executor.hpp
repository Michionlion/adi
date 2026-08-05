#pragma once

#include "adi/backend.hpp"
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

struct MoeExpertScratch {
    ExpertScratch codec;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> activated;
    std::vector<float> projected;
};

struct MoeScratch {
    std::array<MoeExpertScratch, 8> experts;
    ExpertScratch shared_codec;
    std::vector<float> router_logits;
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

struct DecoderState {
    std::uint32_t position = 0;
    std::array<FullAttentionState, 40> full_attention;
    std::array<LinearAttentionState, 40> linear_attention;
};

struct DecoderScratch {
    std::vector<float> hidden;
    std::vector<float> normalized;
    std::vector<float> attention;
    std::vector<float> feed_forward;
    FullAttentionScratch full_attention;
    LinearAttentionScratch linear_attention;
    MoeScratch moe;
};

[[nodiscard]] std::array<ExpertRoute, 8> top_experts(
    std::span<const float> logits);

[[nodiscard]] std::uint32_t gated_delta_key_head(
    std::uint32_t value_head,
    std::uint32_t value_heads,
    std::uint32_t key_heads);

[[nodiscard]] std::array<ExpertRoute, 8> moe_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> input,
    std::span<float> output,
    MoeScratch &scratch,
    const Backend &backend = cpu_backend());

void full_attention_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::uint32_t position,
    std::span<const float> input,
    std::span<float> output,
    FullAttentionState &state,
    FullAttentionScratch &scratch,
    const Backend &backend = cpu_backend());

void linear_attention_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> input,
    std::span<float> output,
    LinearAttentionState &state,
    LinearAttentionScratch &scratch,
    const Backend &backend = cpu_backend());

void decode_token(
    const MachModel &model,
    std::uint32_t token,
    DecoderState &state,
    std::span<float> logits,
    DecoderScratch &scratch,
    const Backend &backend = cpu_backend());

} // namespace adi
