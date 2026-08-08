// parallel_dynamic hands out indexes on demand. A dropped or repeated index
// would silently corrupt whichever kernel used it, so this checks the
// contract directly rather than relying on a downstream result comparison.
#include "parallel.hpp"

#include <atomic>
#include <barrier>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void check_every_index_once(std::uint32_t count) {
    std::vector<std::atomic<std::uint32_t>> visits(count);
    for (auto &visit : visits) {
        visit.store(0, std::memory_order_relaxed);
    }
    adi::detail::parallel_dynamic(count, [&](std::uint32_t index) {
        assert(index < count);
        visits[index].fetch_add(1, std::memory_order_relaxed);
    });
    for (const auto &visit : visits) {
        assert(visit.load(std::memory_order_relaxed) == 1);
    }
}

// Uneven work is the case the primitive exists for: one very heavy index must
// not decide when everything else finishes.
void check_uneven_work() {
    constexpr std::uint32_t count = 64;
    std::atomic<std::uint64_t> total{0};
    adi::detail::parallel_dynamic(count, [&](std::uint32_t index) {
        const std::uint32_t rounds = index == 0 ? 200000 : 100;
        std::uint64_t sum = 0;
        for (std::uint32_t step = 0; step < rounds; ++step) {
            sum += step % 7;
        }
        total.fetch_add(sum, std::memory_order_relaxed);
    });
    assert(total.load() > 0);
}

void check_nested() {
    // A dispatch inside a dispatch must still run every index exactly once,
    // and the inner dispatch must run in place on the calling thread. The
    // barrier makes every pool thread, including the caller, claim one outer
    // index before any can enter the inner dispatch.
    const auto outer = adi::detail::worker_thread_count();
    constexpr std::uint32_t inner = 5;
    std::vector<std::atomic<std::uint32_t>> visits(outer * inner);
    for (auto &visit : visits) {
        visit.store(0, std::memory_order_relaxed);
    }
    std::barrier ready(static_cast<std::ptrdiff_t>(outer));
    adi::detail::parallel_dynamic(outer, [&](std::uint32_t outer_index) {
        const auto outer_thread = std::this_thread::get_id();
        ready.arrive_and_wait();
        adi::detail::parallel_dynamic(inner, [&](std::uint32_t inner_index) {
            assert(std::this_thread::get_id() == outer_thread);
            visits[outer_index * inner + inner_index].fetch_add(
                1, std::memory_order_relaxed);
        });
    });
    for (const auto &visit : visits) {
        assert(visit.load(std::memory_order_relaxed) == 1);
    }
}

void check_exception_propagates() {
    bool thrown = false;
    try {
        adi::detail::parallel_dynamic(32, [](std::uint32_t index) {
            if (index == 17) {
                throw std::runtime_error("index failed");
            }
        });
    } catch (const std::runtime_error &) {
        thrown = true;
    }
    assert(thrown);
}

} // namespace

int main() {
    // Zero is a no-op, one takes the serial path, and the rest cross the
    // worker count in both directions.
    for (const std::uint32_t count :
         {0U, 1U, 2U, 3U, 7U, 8U, 9U, 31U, 64U, 1000U}) {
        check_every_index_once(count);
    }
    check_uneven_work();
    check_nested();
    check_exception_propagates();
}
