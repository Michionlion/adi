#pragma once

#include "adi/model.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace adi {

class Tokenizer {
  public:
    explicit Tokenizer(const MachModel &model);

    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text);
    [[nodiscard]] std::string decode(std::span<const std::uint32_t> tokens) const;
    [[nodiscard]] std::string token_text(std::uint32_t token) const;
    [[nodiscard]] std::uint32_t bos_token() const noexcept { return bos_token_; }
    [[nodiscard]] std::uint32_t eos_token() const noexcept { return eos_token_; }

  private:
    void encode_normal(std::string_view text, std::vector<std::uint32_t> &output);
    void encode_piece(std::string_view piece, std::vector<std::uint32_t> &output);

    std::vector<std::string_view> tokens_;
    std::vector<std::uint64_t> token_types_;
    std::unordered_map<std::string_view, std::uint32_t> token_ids_;
    std::unordered_map<std::string, std::uint32_t> merge_ranks_;
    std::vector<std::pair<std::string_view, std::uint32_t>> special_tokens_;
    std::unordered_map<unsigned char, std::string> byte_encoder_;
    std::unordered_map<std::uint32_t, unsigned char> byte_decoder_;
    std::unordered_map<std::string, std::vector<std::uint32_t>> cache_;
    std::uint32_t bos_token_;
    std::uint32_t eos_token_;
};

[[nodiscard]] std::string qwen_user_prompt(std::string_view input);

} // namespace adi
