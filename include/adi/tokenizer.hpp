#pragma once

#include "adi/model.hpp"

#include <cstdint>
#include <functional>
#include <list>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace adi {

using CancelCallback = std::function<bool()>;

class Tokenizer {
  public:
    explicit Tokenizer(const MachModel &model);

    [[nodiscard]] std::vector<std::uint32_t> encode(
        std::string_view text,
        const CancelCallback &cancelled = {});
    [[nodiscard]] std::string decode(std::span<const std::uint32_t> tokens) const;
    [[nodiscard]] std::string token_text(std::uint32_t token) const;
    void mask_unused_logits(std::span<float> logits) const;
    [[nodiscard]] std::uint32_t bos_token() const noexcept { return bos_token_; }
    [[nodiscard]] std::uint32_t eos_token() const noexcept { return eos_token_; }

  private:
    struct CacheEntry {
        std::vector<std::uint32_t> tokens;
        std::list<std::string>::iterator recency;
    };

    void encode_normal(
        std::string_view text,
        std::vector<std::uint32_t> &output,
        const CancelCallback &cancelled);
    void encode_piece(
        std::string_view piece,
        std::vector<std::uint32_t> &output,
        const CancelCallback &cancelled);

    std::vector<std::string_view> tokens_;
    std::vector<std::uint64_t> token_types_;
    std::unordered_map<std::string_view, std::uint32_t> token_ids_;
    std::unordered_map<
        std::uint64_t,
        std::pair<std::uint32_t, std::uint32_t>> merge_rules_;
    std::vector<std::pair<std::string_view, std::uint32_t>> special_tokens_;
    std::vector<std::uint32_t> unused_tokens_;
    std::unordered_map<unsigned char, std::string> byte_encoder_;
    std::unordered_map<std::uint32_t, unsigned char> byte_decoder_;
    std::list<std::string> cache_recency_;
    std::unordered_map<std::string, CacheEntry> cache_;
    std::uint32_t bos_token_;
    std::uint32_t eos_token_;
};

[[nodiscard]] std::string qwen_user_prompt(std::string_view input);

} // namespace adi
