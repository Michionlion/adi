#pragma once

#include "adi/profiling.hpp"

#include <chrono>
#include <cstdint>

namespace adi::detail {

void record_kernel_profile(
    KernelKind kind,
    std::uint64_t nanoseconds,
    std::uint64_t work_items) noexcept;

class KernelTimer {
  public:
    explicit KernelTimer(KernelKind kind, std::uint64_t work_items = 0) noexcept
        : kind_(kind),
          work_items_(work_items),
          enabled_(kernel_profiling_enabled()) {
        if (enabled_) {
            start_ = Clock::now();
        }
    }

    KernelTimer(const KernelTimer &) = delete;
    KernelTimer &operator=(const KernelTimer &) = delete;

    ~KernelTimer() {
        if (enabled_) {
            detail::record_kernel_profile(
                kind_,
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now() - start_)
                        .count()),
                work_items_);
        }
    }

  private:
    using Clock = std::chrono::steady_clock;

    KernelKind kind_;
    std::uint64_t work_items_;
    bool enabled_;
    Clock::time_point start_{};
};

} // namespace adi::detail
