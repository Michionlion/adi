#include "adi/generation.hpp"

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
    if (logits.empty() || temperature < 0.0F || top_p <= 0.0F || top_p > 1.0F) {
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
    const float maximum = logits[indexes[0]] / temperature;
    std::vector<double> probabilities;
    probabilities.reserve(logits.size());
    double denominator = 0.0;
    for (const auto index : indexes) {
        const double probability = std::exp(
            static_cast<double>(logits[index] / temperature - maximum));
        probabilities.push_back(probability);
        denominator += probability;
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
    if (options.max_output_tokens > model.config().context ||
        options.temperature < 0.0F || options.top_p <= 0.0F || options.top_p > 1.0F) {
        throw std::invalid_argument("invalid generation options");
    }
    const auto prompt = qwen_user_prompt(input);
    const auto prompt_tokens = tokenizer.encode(prompt);
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
    for (const auto token : prompt_tokens) {
        decode_token(model, token, state, logits, scratch);
    }
    std::mt19937_64 random(options.seed);
    std::vector<std::uint32_t> output_tokens;
    output_tokens.reserve(options.max_output_tokens);
    for (std::uint32_t index = 0; index < options.max_output_tokens; ++index) {
        const auto token =
            sample_token(logits, options.temperature, options.top_p, random);
        if (token == tokenizer.eos_token() || token == im_end) {
            break;
        }
        output_tokens.push_back(token);
        if (index + 1 < options.max_output_tokens) {
            decode_token(model, token, state, logits, scratch);
        }
    }
    return {
        tokenizer.decode(output_tokens),
        static_cast<std::uint32_t>(prompt_tokens.size()),
        static_cast<std::uint32_t>(output_tokens.size()),
    };
}

} // namespace adi
