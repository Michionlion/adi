#pragma once

#include "adi/executor.hpp"
#include "adi/tokenizer.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>

namespace adi {

struct GenerationOptions {
    std::optional<std::uint32_t> max_output_tokens;
    float temperature = 0.7F;
    float top_p = 0.9F;
    std::uint64_t seed = 0;
};

enum class FinishReason {
    stop_token,
    length,
};

struct GenerationResult {
    std::string text;
    std::uint32_t input_tokens;
    std::uint32_t output_tokens;
    FinishReason finish_reason;
};

using TokenCallback = std::function<void(std::string_view)>;

[[nodiscard]] std::uint32_t sample_token(
    std::span<const float> logits,
    float temperature,
    float top_p,
    std::mt19937_64 &random);

[[nodiscard]] GenerationResult generate(
    const MachModel &model,
    Tokenizer &tokenizer,
    std::string_view input,
    const GenerationOptions &options);

[[nodiscard]] GenerationResult generate_from_prompt(
    const MachModel &model,
    Tokenizer &tokenizer,
    std::string_view formatted_prompt,
    const GenerationOptions &options,
    const TokenCallback &token_callback = {},
    const CancelCallback &cancelled = {});

class ContinuousBatcher {
  public:
    ContinuousBatcher(const MachModel &model, Tokenizer &tokenizer);
    ~ContinuousBatcher();

    ContinuousBatcher(const ContinuousBatcher &) = delete;
    ContinuousBatcher &operator=(const ContinuousBatcher &) = delete;

    [[nodiscard]] GenerationResult generate_from_prompt(
        std::string_view formatted_prompt,
        const GenerationOptions &options,
        const TokenCallback &token_callback = {},
        const CancelCallback &cancelled = {});

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace adi
