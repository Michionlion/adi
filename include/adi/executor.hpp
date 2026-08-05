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

[[nodiscard]] std::array<ExpertRoute, 8> top_experts(
    std::span<const float> logits);

[[nodiscard]] std::array<ExpertRoute, 8> moe_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> input,
    std::span<float> output,
    MoeScratch &scratch);

} // namespace adi
