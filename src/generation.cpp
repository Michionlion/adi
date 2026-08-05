#include "adi/generation.hpp"
#include "utf8.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace adi {

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
    if (options.max_output_tokens == 0 ||
        options.max_output_tokens > model.config().context ||
        !std::isfinite(options.temperature) || !std::isfinite(options.top_p) ||
        options.temperature < 0.0F ||
        (options.temperature > 0.0F && options.temperature < 1.0e-4F) ||
        options.temperature > 2.0F ||
        options.top_p <= 0.0F || options.top_p > 1.0F) {
        throw std::invalid_argument("invalid generation options");
    }
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
    std::vector<float> logits(model.config().vocabulary);
    for (std::size_t index = 0; index < prompt_tokens.size(); ++index) {
        if (cancelled && cancelled()) {
            throw std::runtime_error("generation cancelled");
        }
        decode_token(
            model,
            prompt_tokens[index],
            state,
            index + 1 == prompt_tokens.size()
                ? std::span<float>(logits)
                : std::span<float>{},
            scratch);
    }
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

} // namespace adi
