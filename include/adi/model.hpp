#pragma once

#include "adi/gguf.hpp"
#include "adi/kernels.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace adi {

enum class ExpertProjection {
    gate,
    up,
    down,
};

struct MachConfig {
    std::uint32_t layers;
    std::uint32_t hidden;
    std::uint32_t experts;
    std::uint32_t active_experts;
    std::uint32_t expert_hidden;
    std::uint32_t context;
    std::uint32_t vocabulary;
};

struct MoeDescriptor {
    Bf16Matrix router;
    Bf16Matrix shared_expert_gate;
    MachNeMatrix shared_gate;
    MachNeMatrix shared_up;
    MachNeMatrix shared_down;
    std::array<std::array<MachExpertMatrix, 3>, 256> experts;
};

struct FullAttentionDescriptor {
    MachNeMatrix query;
    MachNeMatrix key;
    MachNeMatrix value;
    MachNeMatrix output;
    std::span<const std::uint16_t> query_norm;
    std::span<const std::uint16_t> key_norm;
};

struct LinearAttentionDescriptor {
    MachNeMatrix qkv;
    MachNeMatrix gate;
    MachNeMatrix output;
    Bf16Matrix alpha;
    Bf16Matrix beta;
    std::span<const std::uint16_t> convolution;
    std::span<const std::uint16_t> alpha_bias;
    std::span<const std::uint16_t> a_log;
    std::span<const std::uint16_t> norm;
};

struct LayerDescriptor {
    std::span<const std::uint16_t> input_norm;
    std::span<const std::uint16_t> post_attention_norm;
    MoeDescriptor moe;
    bool full_attention = false;
    FullAttentionDescriptor full;
    LinearAttentionDescriptor linear;
};

class MachModel {
  public:
    explicit MachModel(const std::filesystem::path &path);

    [[nodiscard]] const MachConfig &config() const noexcept { return config_; }
    [[nodiscard]] std::string_view name() const noexcept { return name_; }
    [[nodiscard]] const GgufFile &gguf() const noexcept { return file_; }
    [[nodiscard]] const LayerDescriptor &layer(std::uint32_t index) const;
    [[nodiscard]] std::span<const float, 32> rope_inverse_frequencies() const noexcept {
        return rope_inverse_frequencies_;
    }
    [[nodiscard]] std::span<const float, 32> rope_theta_divisors() const noexcept {
        return rope_theta_divisors_;
    }
    [[nodiscard]] std::span<const std::uint16_t> final_norm() const noexcept {
        return final_norm_;
    }
    [[nodiscard]] MachExpertMatrix expert(
        std::uint32_t layer,
        std::uint32_t expert,
        ExpertProjection projection) const;
    [[nodiscard]] MachNeMatrix non_expert(
        std::uint32_t layer,
        std::string_view source_name) const;
    [[nodiscard]] MachEmbedding embedding() const;
    [[nodiscard]] MachHeadChunk head_chunk(std::uint32_t chunk) const;
    [[nodiscard]] Bf16Matrix bf16_matrix(std::string_view source_name) const;
    [[nodiscard]] std::span<const std::uint16_t> bf16_vector(
        std::string_view source_name) const;
    [[nodiscard]] std::span<const std::uint16_t> bf16_data(
        std::string_view source_name) const;

  private:
    void validate_manifest() const;
    void cache_descriptors();

    GgufFile file_;
    MachConfig config_;
    std::string_view name_;
    bool descriptors_ready_ = false;
    MachEmbedding embedding_;
    std::array<MachHeadChunk, 8> head_chunks_;
    std::span<const std::uint16_t> final_norm_;
    std::array<float, 32> rope_inverse_frequencies_;
    std::array<float, 32> rope_theta_divisors_;
    std::vector<float> expert_state_values_;
    std::vector<float> ne_state_values_;
    std::vector<float> ne_signed_tlut_;
    std::vector<std::uint16_t> expert_wave_indexes_forward_;
    std::vector<std::uint16_t> expert_wave_indexes_reverse_;
    std::vector<float> expert_wave_gamma_;
    std::vector<LayerDescriptor> layers_;
};

} // namespace adi
