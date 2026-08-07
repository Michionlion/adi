#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace adi {

enum class KernelKind : std::uint32_t {
    expert,
    expert_batch,
    non_expert,
    non_expert_batch,
    embedding,
    output_head,
    output_head_batch,
    bf16_projection,
    rms_norm,
    full_attention,
    linear_attention,
    moe,
    decoder_layer,
    count,
};

struct KernelProfile {
    std::uint64_t calls = 0;
    std::uint64_t nanoseconds = 0;
    std::uint64_t work_items = 0;
};

using KernelProfiles = std::array<
    KernelProfile,
    static_cast<std::size_t>(KernelKind::count)>;

void set_kernel_profiling_enabled(bool enabled) noexcept;
[[nodiscard]] bool kernel_profiling_enabled() noexcept;
void reset_kernel_profiles() noexcept;
[[nodiscard]] KernelProfiles kernel_profiles() noexcept;
[[nodiscard]] std::string_view kernel_name(KernelKind kind) noexcept;

} // namespace adi
