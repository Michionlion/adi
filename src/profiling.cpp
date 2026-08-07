#include "adi/profiling.hpp"
#include "profiling_internal.hpp"

#include <array>
#include <atomic>
#include <cstddef>

namespace adi {
namespace {

struct AtomicProfile {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> nanoseconds{0};
    std::atomic<std::uint64_t> work_items{0};
};

std::atomic<bool> profiling_enabled{false};
std::array<AtomicProfile, static_cast<std::size_t>(KernelKind::count)> profiles;

} // namespace

void set_kernel_profiling_enabled(bool enabled) noexcept {
    profiling_enabled.store(enabled, std::memory_order_release);
}

bool kernel_profiling_enabled() noexcept {
    return profiling_enabled.load(std::memory_order_acquire);
}

void reset_kernel_profiles() noexcept {
    for (auto &profile : profiles) {
        profile.calls.store(0, std::memory_order_relaxed);
        profile.nanoseconds.store(0, std::memory_order_relaxed);
        profile.work_items.store(0, std::memory_order_relaxed);
    }
}

KernelProfiles kernel_profiles() noexcept {
    KernelProfiles result;
    for (std::size_t index = 0; index < profiles.size(); ++index) {
        result[index] = {
            profiles[index].calls.load(std::memory_order_relaxed),
            profiles[index].nanoseconds.load(std::memory_order_relaxed),
            profiles[index].work_items.load(std::memory_order_relaxed),
        };
    }
    return result;
}

std::string_view kernel_name(KernelKind kind) noexcept {
    constexpr std::array names{
        std::string_view{"expert"},
        std::string_view{"expert_batch"},
        std::string_view{"non_expert"},
        std::string_view{"non_expert_batch"},
        std::string_view{"embedding"},
        std::string_view{"output_head"},
        std::string_view{"output_head_batch"},
        std::string_view{"bf16_projection"},
        std::string_view{"rms_norm"},
        std::string_view{"full_attention"},
        std::string_view{"linear_attention"},
        std::string_view{"moe"},
        std::string_view{"decoder_layer"},
    };
    const auto index = static_cast<std::size_t>(kind);
    return index < names.size() ? names[index] : std::string_view{"unknown"};
}

void detail::record_kernel_profile(
    KernelKind kind,
    std::uint64_t nanoseconds,
    std::uint64_t work_items) noexcept {
    const auto index = static_cast<std::size_t>(kind);
    if (index >= profiles.size()) {
        return;
    }
    profiles[index].calls.fetch_add(1, std::memory_order_relaxed);
    profiles[index].nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
    profiles[index].work_items.fetch_add(work_items, std::memory_order_relaxed);
}

} // namespace adi
