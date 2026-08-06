#include "adi/generation.hpp"
#include "utf8.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <vector>

namespace adi {
namespace {

void validate_generation_options(
    const MachModel &model,
    const GenerationOptions &options) {
    if (options.max_output_tokens == 0 ||
        options.max_output_tokens > model.config().context ||
        !std::isfinite(options.temperature) || !std::isfinite(options.top_p) ||
        options.temperature < 0.0F ||
        (options.temperature > 0.0F && options.temperature < 1.0e-4F) ||
        options.temperature > 2.0F ||
        options.top_p <= 0.0F || options.top_p > 1.0F) {
        throw std::invalid_argument("invalid generation options");
    }
}

constexpr std::size_t prefill_chunk_tokens = 64;

void throw_if_cancelled(const CancelCallback &cancelled) {
    if (cancelled && cancelled()) {
        throw std::runtime_error("generation cancelled");
    }
}

void prefill_prompt(
    const MachModel &model,
    std::span<const std::uint32_t> tokens,
    DecoderState &state,
    std::span<float> logits,
    PrefillScratch &scratch,
    const CancelCallback &cancelled) {
    for (std::size_t offset = 0; offset < tokens.size();
         offset += prefill_chunk_tokens) {
        throw_if_cancelled(cancelled);
        const auto count = std::min(
            prefill_chunk_tokens,
            tokens.size() - offset);
        prefill(
            model,
            tokens.subspan(offset, count),
            state,
            logits,
            scratch);
    }
    throw_if_cancelled(cancelled);
}

} // namespace

std::uint32_t sample_token(
    std::span<const float> logits,
    float temperature,
    float top_p,
    std::mt19937_64 &random) {
    constexpr float minimum_temperature = 1.0e-4F;
    constexpr float maximum_temperature = 2.0F;
    if (logits.empty() || !std::isfinite(temperature) ||
        !std::isfinite(top_p) || temperature < 0.0F ||
        (temperature > 0.0F && temperature < minimum_temperature) ||
        temperature > maximum_temperature || top_p <= 0.0F || top_p > 1.0F ||
        std::any_of(logits.begin(), logits.end(), [](float value) {
            return !std::isfinite(value);
        })) {
        throw std::invalid_argument("invalid sampling parameters");
    }
    if (temperature == 0.0F) {
        return static_cast<std::uint32_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    std::vector<std::uint32_t> indexes(logits.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::sort(indexes.begin(), indexes.end(), [&](std::uint32_t left, std::uint32_t right) {
        if (logits[left] == logits[right]) {
            return left < right;
        }
        return logits[left] > logits[right];
    });
    const double maximum = static_cast<double>(logits[indexes[0]]);
    std::vector<double> probabilities;
    probabilities.reserve(logits.size());
    double denominator = 0.0;
    for (const auto index : indexes) {
        const double scaled =
            (static_cast<double>(logits[index]) - maximum) /
            static_cast<double>(temperature);
        const double probability = std::exp(scaled);
        probabilities.push_back(probability);
        denominator += probability;
    }
    if (!std::isfinite(denominator) || denominator <= 0.0) {
        throw std::runtime_error("sampling probability sum is invalid");
    }
    double cumulative = 0.0;
    std::size_t retained = 0;
    for (; retained < probabilities.size(); ++retained) {
        cumulative += probabilities[retained] / denominator;
        if (cumulative >= top_p) {
            ++retained;
            break;
        }
    }
    retained = std::max<std::size_t>(1, retained);
    double retained_sum =
        std::accumulate(probabilities.begin(), probabilities.begin() + retained, 0.0);
    std::uniform_real_distribution<double> distribution(0.0, retained_sum);
    const double target = distribution(random);
    double running = 0.0;
    for (std::size_t index = 0; index < retained; ++index) {
        running += probabilities[index];
        if (target <= running) {
            return indexes[index];
        }
    }
    return indexes[retained - 1];
}

GenerationResult generate(
    const MachModel &model,
    Tokenizer &tokenizer,
    std::string_view input,
    const GenerationOptions &options) {
    return generate_from_prompt(
        model, tokenizer, qwen_user_prompt(input), options);
}

GenerationResult generate_from_prompt(
    const MachModel &model,
    Tokenizer &tokenizer,
    std::string_view formatted_prompt,
    const GenerationOptions &options,
    const TokenCallback &token_callback,
    const CancelCallback &cancelled) {
    validate_generation_options(model, options);
    const auto prompt_tokens = tokenizer.encode(formatted_prompt, cancelled);
    if (prompt_tokens.empty() ||
        prompt_tokens.size() + options.max_output_tokens > model.config().context) {
        throw std::invalid_argument("prompt exceeds model context");
    }
    const auto im_end_tokens = tokenizer.encode("<|im_end|>");
    if (im_end_tokens.size() != 1) {
        throw std::runtime_error("tokenizer: invalid im_end token");
    }
    const auto im_end = im_end_tokens[0];

    DecoderState state;
    DecoderScratch scratch;
    PrefillScratch prefill_scratch;
    std::vector<float> logits(model.config().vocabulary);
    prefill_prompt(
        model,
        prompt_tokens,
        state,
        logits,
        prefill_scratch,
        cancelled);
    tokenizer.mask_unused_logits(logits);
    std::mt19937_64 random(options.seed);
    std::vector<std::uint32_t> output_tokens;
    output_tokens.reserve(options.max_output_tokens);
    auto finish_reason = FinishReason::length;
    for (std::uint32_t index = 0; index < options.max_output_tokens; ++index) {
        if (cancelled && cancelled()) {
            throw std::runtime_error("generation cancelled");
        }
        const auto token =
            sample_token(logits, options.temperature, options.top_p, random);
        if (token == tokenizer.eos_token() || token == im_end) {
            finish_reason = FinishReason::stop_token;
            break;
        }
        output_tokens.push_back(token);
        if (token_callback) {
            const auto piece = tokenizer.token_text(token);
            token_callback(piece);
        }
        if (index + 1 < options.max_output_tokens) {
            decode_token(model, token, state, logits, scratch);
            tokenizer.mask_unused_logits(logits);
        }
    }
    return {
        sanitize_utf8(tokenizer.decode(output_tokens)),
        static_cast<std::uint32_t>(prompt_tokens.size()),
        static_cast<std::uint32_t>(output_tokens.size()),
        finish_reason,
    };
}

struct ContinuousBatcher::Impl {
    struct Request {
        [[nodiscard]] bool is_cancelled() const {
            if (callback_failed.load(std::memory_order_acquire)) {
                return true;
            }
            return cancelled && cancelled();
        }

        void publish(std::string piece) {
            if (!stream_tokens) {
                return;
            }
            {
                std::lock_guard lock(stream_mutex);
                stream_pieces.push_back(std::move(piece));
            }
            stream_condition.notify_one();
        }

        void complete_stream() noexcept {
            try {
                std::lock_guard lock(stream_mutex);
                stream_finished = true;
            } catch (...) {
            }
            stream_condition.notify_all();
        }

        std::string prompt;
        GenerationOptions options;
        CancelCallback cancelled;
        std::promise<GenerationResult> promise;
        std::mutex stream_mutex;
        std::condition_variable stream_condition;
        std::deque<std::string> stream_pieces;
        std::atomic<bool> callback_failed = false;
        bool stream_tokens = false;
        bool stream_finished = false;
    };

    struct Active {
        std::shared_ptr<Request> request;
        std::uint32_t input_tokens = 0;
        std::vector<std::uint32_t> output_tokens;
        std::mt19937_64 random;
    };

    Impl(const MachModel &model, Tokenizer &tokenizer)
        : model(model),
          tokenizer(tokenizer),
          worker([this](std::stop_token stop) { run(stop); }) {}

    ~Impl() {
        worker.request_stop();
        condition.notify_all();
    }

    GenerationResult submit(
        std::string_view prompt,
        const GenerationOptions &options,
        const TokenCallback &token_callback,
        const CancelCallback &cancelled) {
        validate_generation_options(model, options);
        auto request = std::make_shared<Request>();
        request->prompt = prompt;
        request->options = options;
        request->cancelled = cancelled;
        request->stream_tokens = static_cast<bool>(token_callback);
        auto result = request->promise.get_future();
        {
            std::lock_guard lock(mutex);
            pending.push_back(request);
        }
        condition.notify_one();
        if (!token_callback) {
            return result.get();
        }

        for (;;) {
            std::deque<std::string> pieces;
            bool finished = false;
            {
                std::unique_lock lock(request->stream_mutex);
                request->stream_condition.wait(lock, [&] {
                    return !request->stream_pieces.empty() ||
                           request->stream_finished;
                });
                pieces.swap(request->stream_pieces);
                finished = request->stream_finished;
            }
            try {
                for (const auto &piece : pieces) {
                    token_callback(piece);
                }
            } catch (...) {
                request->callback_failed.store(
                    true,
                    std::memory_order_release);
                throw;
            }
            if (finished) {
                return result.get();
            }
        }
    }

    void fail(
        const std::shared_ptr<Request> &request,
        std::exception_ptr exception) noexcept {
        try {
            request->promise.set_exception(exception);
        } catch (...) {
        }
        request->complete_stream();
    }

    void finish(Active &active, FinishReason reason) {
        try {
            active.request->promise.set_value({
                sanitize_utf8(tokenizer.decode(active.output_tokens)),
                active.input_tokens,
                static_cast<std::uint32_t>(active.output_tokens.size()),
                reason,
            });
        } catch (...) {
            active.request->complete_stream();
            throw;
        }
        active.request->complete_stream();
    }

    void admit(
        const std::shared_ptr<Request> &request,
        std::vector<Active> &active,
        std::vector<DecoderState> &states,
        std::vector<DecoderScratch> &scratches,
        std::vector<float> &logits) {
        const CancelCallback request_cancelled = [request] {
            return request->is_cancelled();
        };
        throw_if_cancelled(request_cancelled);
        auto prompt_tokens =
            tokenizer.encode(request->prompt, request_cancelled);
        if (prompt_tokens.empty() ||
            prompt_tokens.size() + request->options.max_output_tokens >
                model.config().context) {
            throw std::invalid_argument("prompt exceeds model context");
        }
        DecoderState state;
        PrefillScratch prefill_scratch;
        std::vector<float> prompt_logits(model.config().vocabulary);
        prefill_prompt(
            model,
            prompt_tokens,
            state,
            prompt_logits,
            prefill_scratch,
            request_cancelled);
        tokenizer.mask_unused_logits(prompt_logits);
        Active admitted{
            request,
            static_cast<std::uint32_t>(prompt_tokens.size()),
            {},
            std::mt19937_64(request->options.seed),
        };
        admitted.output_tokens.reserve(request->options.max_output_tokens);

        // Reserve every structure before mutating any of them so an allocation
        // failure cannot leave the structure-of-arrays batch out of sync.
        active.reserve(active.size() + 1);
        states.reserve(states.size() + 1);
        scratches.reserve(scratches.size() + 1);
        logits.reserve(logits.size() + model.config().vocabulary);

        active.push_back(std::move(admitted));
        states.push_back(std::move(state));
        scratches.emplace_back();
        logits.insert(
            logits.end(),
            prompt_logits.begin(),
            prompt_logits.end());
    }

    void admit_pending(
        std::vector<Active> &active,
        std::vector<DecoderState> &states,
        std::vector<DecoderScratch> &scratches,
        std::vector<float> &logits) {
        std::shared_ptr<Request> request;
        {
            std::lock_guard lock(mutex);
            if (pending.empty()) {
                return;
            }
            request = std::move(pending.front());
            pending.pop_front();
        }
        try {
            admit(
                request,
                active,
                states,
                scratches,
                logits);
        } catch (...) {
            fail(request, std::current_exception());
        }
    }

    void run(std::stop_token stop) noexcept {
        std::vector<Active> active;
        std::vector<DecoderState> states;
        std::vector<DecoderScratch> scratches;
        std::vector<float> logits;
        DecoderBatchScratch batch_scratch;
        std::exception_ptr terminal_exception;

        try {
            const auto im_end_tokens = tokenizer.encode("<|im_end|>");
            if (im_end_tokens.size() != 1) {
                throw std::runtime_error(
                    "tokenizer did not encode <|im_end|> as one token");
            }
            const auto im_end = im_end_tokens[0];

            while (!stop.stop_requested()) {
                if (active.empty()) {
                    std::unique_lock lock(mutex);
                    condition.wait(lock, stop, [&] { return !pending.empty(); });
                    if (stop.stop_requested()) {
                        break;
                    }
                    // A short window makes simultaneous arrivals deterministic
                    // without imposing meaningful latency on a lone request.
                    condition.wait_for(
                        lock,
                        stop,
                        std::chrono::milliseconds(2),
                        [] { return false; });
                }
                admit_pending(active, states, scratches, logits);
                if (active.empty()) {
                    continue;
                }

                std::vector<Active> next_active;
                std::vector<DecoderState> next_states;
                std::vector<DecoderScratch> next_scratches;
                std::vector<std::uint32_t> next_tokens;
                next_active.reserve(active.size());
                next_states.reserve(active.size());
                next_scratches.reserve(active.size());
                next_tokens.reserve(active.size());

                for (std::size_t index = 0; index < active.size(); ++index) {
                    try {
                        auto &item = active[index];
                        if (item.request->is_cancelled()) {
                            throw std::runtime_error("generation cancelled");
                        }
                        const auto row = std::span<const float>(logits).subspan(
                            index * model.config().vocabulary,
                            model.config().vocabulary);
                        const auto token = sample_token(
                            row,
                            item.request->options.temperature,
                            item.request->options.top_p,
                            item.random);
                        if (token == tokenizer.eos_token() || token == im_end) {
                            finish(item, FinishReason::stop_token);
                            continue;
                        }
                        item.output_tokens.push_back(token);
                        if (item.request->stream_tokens) {
                            item.request->publish(
                                tokenizer.token_text(token));
                        }
                        if (item.output_tokens.size() ==
                            item.request->options.max_output_tokens) {
                            finish(item, FinishReason::length);
                            continue;
                        }
                        next_tokens.push_back(token);
                        next_active.push_back(std::move(item));
                        next_states.push_back(std::move(states[index]));
                        next_scratches.push_back(std::move(scratches[index]));
                    } catch (...) {
                        fail(active[index].request, std::current_exception());
                    }
                }

                active = std::move(next_active);
                states = std::move(next_states);
                scratches = std::move(next_scratches);
                logits.clear();
                if (!active.empty()) {
                    logits.resize(
                        active.size() * model.config().vocabulary);
                    try {
                        decode_batch(
                            model,
                            next_tokens,
                            states,
                            logits,
                            scratches,
                            batch_scratch);
                        for (std::size_t index = 0;
                             index < active.size();
                             ++index) {
                            tokenizer.mask_unused_logits(
                                std::span<float>(logits).subspan(
                                    index * model.config().vocabulary,
                                    model.config().vocabulary));
                        }
                    } catch (...) {
                        const auto exception = std::current_exception();
                        for (const auto &item : active) {
                            fail(item.request, exception);
                        }
                        active.clear();
                        states.clear();
                        scratches.clear();
                        logits.clear();
                    }
                }
            }
            terminal_exception = std::make_exception_ptr(
                std::runtime_error("continuous batcher stopped"));
        } catch (...) {
            terminal_exception = std::current_exception();
        }

        for (const auto &item : active) {
            fail(item.request, terminal_exception);
        }
        std::deque<std::shared_ptr<Request>> remaining;
        {
            std::lock_guard lock(mutex);
            remaining.swap(pending);
        }
        for (const auto &request : remaining) {
            fail(request, terminal_exception);
        }
    }

    const MachModel &model;
    Tokenizer &tokenizer;
    std::mutex mutex;
    std::condition_variable_any condition;
    std::deque<std::shared_ptr<Request>> pending;
    std::jthread worker;
};

ContinuousBatcher::ContinuousBatcher(
    const MachModel &model,
    Tokenizer &tokenizer)
    : impl_(std::make_unique<Impl>(model, tokenizer)) {}

ContinuousBatcher::~ContinuousBatcher() = default;

GenerationResult ContinuousBatcher::generate_from_prompt(
    std::string_view formatted_prompt,
    const GenerationOptions &options,
    const TokenCallback &token_callback,
    const CancelCallback &cancelled) {
    return impl_->submit(
        formatted_prompt,
        options,
        token_callback,
        cancelled);
}

} // namespace adi
