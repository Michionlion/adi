#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <thread>
#include <vector>

namespace adi::detail {

template <typename Function>
void parallel_ranges(
    std::uint32_t count,
    std::uint32_t minimum_per_worker,
    Function &&function) {
    const auto available = std::max(1U, std::thread::hardware_concurrency());
    const auto workers =
        std::min(available, std::max(1U, count / minimum_per_worker));
    if (workers == 1) {
        function(0, count);
        return;
    }

    std::vector<std::exception_ptr> exceptions(workers);
    const auto invoke = [&](std::uint32_t worker,
                            std::uint32_t begin,
                            std::uint32_t end) noexcept {
        try {
            function(begin, end);
        } catch (...) {
            exceptions[worker] = std::current_exception();
        }
    };

    std::vector<std::jthread> threads;
    threads.reserve(workers - 1);
    for (std::uint32_t worker = 0; worker + 1 < workers; ++worker) {
        const auto begin = count * worker / workers;
        const auto end = count * (worker + 1) / workers;
        threads.emplace_back([&, worker, begin, end] {
            invoke(worker, begin, end);
        });
    }
    invoke(
        workers - 1,
        count * (workers - 1) / workers,
        count);
    for (auto &thread : threads) {
        thread.join();
    }
    for (const auto &exception : exceptions) {
        if (exception) {
            std::rethrow_exception(exception);
        }
    }
}

} // namespace adi::detail
