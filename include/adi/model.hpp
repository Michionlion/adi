#pragma once

#include "adi/gguf.hpp"
#include "adi/kernels.hpp"

#include <cstdint>
#include <filesystem>
#include <string_view>

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

class MachModel {
  public:
    explicit MachModel(const std::filesystem::path &path);

    [[nodiscard]] const MachConfig &config() const noexcept { return config_; }
    [[nodiscard]] const GgufFile &gguf() const noexcept { return file_; }
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
    GgufFile file_;
    MachConfig config_;
};

} // namespace adi
