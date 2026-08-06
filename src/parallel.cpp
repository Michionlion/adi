#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string_view>
#include <thread>
#include <vector>

namespace adi::detail {
namespace {

thread_local bool worker_thread = false;

std::uint32_t configured_threads() noexcept {
    const auto available = std::max(1U, std::thread::hardware_concurrency());
    const char *value = std::getenv("ADI_THREADS");
    if (value == nullptr) {
        return available;
    }
    std::uint32_t parsed = 0;
    const std::string_view text(value);
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size() || parsed == 0) {
        return available;
    }
    return std::min(parsed, available);
}

struct TaskGroup {
    explicit TaskGroup(std::uint32_t task_count) : remaining(task_count) {}

    void finish(std::exception_ptr failure) noexcept {
        if (failure) {
            std::lock_guard lock(mutex);
            if (!exception) {
                exception = failure;
            }
        }
        if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            condition.notify_one();
        }
    }

    void wait() {
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] {
            return remaining.load(std::memory_order_acquire) == 0;
        });
        if (exception) {
            std::rethrow_exception(exception);
        }
    }

    std::atomic<std::uint32_t> remaining;
    std::mutex mutex;
    std::condition_variable condition;
    std::exception_ptr exception;
};

class WorkerPool {
  public:
    WorkerPool() : threads_(configured_threads()) {
        workers_.reserve(threads_ > 0 ? threads_ - 1 : 0);
        for (std::uint32_t index = 1; index < threads_; ++index) {
            workers_.emplace_back([this](std::stop_token stop) {
                worker_thread = true;
                worker_loop(stop);
            });
        }
    }

    ~WorkerPool() {
        for (auto &worker : workers_) {
            worker.request_stop();
        }
        condition_.notify_all();
        workers_.clear();
    }

    [[nodiscard]] std::uint32_t threads() const noexcept {
        return threads_;
    }

    void run(
        std::uint32_t count,
        std::uint32_t minimum_per_worker,
        const std::function<void(std::uint32_t, std::uint32_t)> &function) {
        if (count == 0) {
            return;
        }
        const auto task_count = std::min(
            threads_,
            std::max(1U, count / std::max(1U, minimum_per_worker)));
        if (task_count == 1 || worker_thread) {
            function(0, count);
            return;
        }

        const auto group = std::make_shared<TaskGroup>(task_count);
        for (std::uint32_t task = 0; task + 1 < task_count; ++task) {
            const auto begin = count * task / task_count;
            const auto end = count * (task + 1) / task_count;
            enqueue([group, &function, begin, end] {
                std::exception_ptr failure;
                try {
                    function(begin, end);
                } catch (...) {
                    failure = std::current_exception();
                }
                group->finish(failure);
            });
        }

        std::exception_ptr failure;
        try {
            function(
                count * (task_count - 1) / task_count,
                count);
        } catch (...) {
            failure = std::current_exception();
        }
        group->finish(failure);
        group->wait();
    }

  private:
    void enqueue(std::function<void()> task) {
        {
            std::lock_guard lock(mutex_);
            tasks_.push_back(std::move(task));
        }
        condition_.notify_one();
    }

    void worker_loop(std::stop_token stop) {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, stop, [&] { return !tasks_.empty(); });
                if (stop.stop_requested() && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::uint32_t threads_;
    std::vector<std::jthread> workers_;
    std::mutex mutex_;
    std::condition_variable_any condition_;
    std::deque<std::function<void()>> tasks_;
};

WorkerPool &worker_pool() {
    static WorkerPool pool;
    return pool;
}

} // namespace

void parallel_ranges_impl(
    std::uint32_t count,
    std::uint32_t minimum_per_worker,
    const std::function<void(std::uint32_t, std::uint32_t)> &function) {
    worker_pool().run(count, minimum_per_worker, function);
}

std::uint32_t worker_thread_count() noexcept {
    return worker_pool().threads();
}

} // namespace adi::detail
