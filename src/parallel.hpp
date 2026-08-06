#pragma once

#include <cstdint>
#include <functional>
#include <utility>

namespace adi::detail {

void parallel_ranges_impl(
    std::uint32_t count,
    std::uint32_t minimum_per_worker,
    const std::function<void(std::uint32_t, std::uint32_t)> &function);

[[nodiscard]] std::uint32_t worker_thread_count() noexcept;

template <typename Function>
void parallel_ranges(
    std::uint32_t count,
    std::uint32_t minimum_per_worker,
    Function &&function) {
    parallel_ranges_impl(
        count,
        minimum_per_worker,
        std::function<void(std::uint32_t, std::uint32_t)>(
            std::forward<Function>(function)));
}

template <typename Function>
void parallel_tasks(std::uint32_t count, Function &&function) {
    parallel_ranges(
        count,
        1,
        [&](std::uint32_t begin, std::uint32_t end) {
            for (auto index = begin; index < end; ++index) {
                function(index);
            }
        });
}

} // namespace adi::detail
