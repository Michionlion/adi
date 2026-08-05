#pragma once

#include "adi/gguf.hpp"
#include "adi/kernels.hpp"

#include <cstdint>
#include <filesystem>

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

  private:
    GgufFile file_;
    MachConfig config_;
};

} // namespace adi
