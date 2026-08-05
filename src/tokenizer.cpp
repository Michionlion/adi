#include "adi/tokenizer.hpp"
#include "chat.hpp"
#include "unicode.hpp"
#include "utf8.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace adi {
namespace {

std::string utf8(std::uint32_t codepoint) {
    std::string result;
    if (codepoint <= 0x7F) {
        result.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        result.push_back(static_cast<char>(0xC0U | (codepoint >> 6)));
        result.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        result.push_back(static_cast<char>(0xE0U | (codepoint >> 12)));
        result.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3FU)));
        result.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
    return result;
}

} // namespace

Tokenizer::Tokenizer(const MachModel &model) {
    const auto &file = model.gguf();
    tokens_ = file.string_array("tokenizer.ggml.tokens");
    token_types_ = file.integer_array("tokenizer.ggml.token_type");
    const auto merges = file.string_array("tokenizer.ggml.merges");
    const auto bos = file.integer("tokenizer.ggml.bos_token_id");
    const auto eos = file.integer("tokenizer.ggml.eos_token_id");
    if (tokens_.size() != model.config().vocabulary ||
        token_types_.size() != tokens_.size() || merges.empty() || !bos || !eos ||
        *bos >= tokens_.size() || *eos >= tokens_.size()) {
        throw std::runtime_error("tokenizer: missing GGUF tokenizer metadata");
    }
    bos_token_ = static_cast<std::uint32_t>(*bos);
    eos_token_ = static_cast<std::uint32_t>(*eos);
    token_ids_.reserve(tokens_.size());
    for (std::uint32_t index = 0; index < tokens_.size(); ++index) {
        if (!token_ids_.emplace(tokens_[index], index).second) {
            throw std::runtime_error("tokenizer: duplicate vocabulary token");
        }
        if (token_types_[index] == 3 || token_types_[index] == 4) {
            special_tokens_.emplace_back(tokens_[index], index);
        } else if (token_types_[index] == 5) {
            unused_tokens_.push_back(index);
        }
    }
    std::sort(
        special_tokens_.begin(), special_tokens_.end(),
        [](const auto &left, const auto &right) {
            return left.first.size() > right.first.size();
        });
    merge_ranks_.reserve(merges.size());
    for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
        if (!merge_ranks_.emplace(std::string(merges[rank]), rank).second) {
            throw std::runtime_error("tokenizer: duplicate BPE merge");
        }
    }

    std::vector<std::uint32_t> bytes;
    for (std::uint32_t value = '!'; value <= '~'; ++value) {
        bytes.push_back(value);
    }
    for (std::uint32_t value = 0xA1; value <= 0xAC; ++value) {
        bytes.push_back(value);
    }
    for (std::uint32_t value = 0xAE; value <= 0xFF; ++value) {
        bytes.push_back(value);
    }
    std::vector<bool> present(256, false);
    for (const auto value : bytes) {
        present[value] = true;
        byte_encoder_[static_cast<unsigned char>(value)] = utf8(value);
        byte_decoder_[value] = static_cast<unsigned char>(value);
    }
    std::uint32_t extra = 0;
    for (std::uint32_t value = 0; value < 256; ++value) {
        if (!present[value]) {
            const auto codepoint = 256 + extra++;
            byte_encoder_[static_cast<unsigned char>(value)] = utf8(codepoint);
            byte_decoder_[codepoint] = static_cast<unsigned char>(value);
        }
    }
}

std::vector<std::uint32_t> Tokenizer::encode(std::string_view text) {
    std::vector<std::uint32_t> result;
    std::size_t position = 0;
    while (position < text.size()) {
        std::size_t special_position = std::string_view::npos;
        std::string_view special;
        std::uint32_t special_id = 0;
        for (const auto &[candidate, id] : special_tokens_) {
            const auto found = text.find(candidate, position);
            if (found < special_position) {
                special_position = found;
                special = candidate;
                special_id = id;
            }
        }
        if (special_position == std::string_view::npos) {
            encode_normal(text.substr(position), result);
            break;
        }
        if (special_position > position) {
            encode_normal(text.substr(position, special_position - position), result);
        }
        result.push_back(special_id);
        position = special_position + special.size();
    }
    return result;
}

void Tokenizer::encode_normal(
    std::string_view text,
    std::vector<std::uint32_t> &output) {
    const auto normalized = normalize_nfc(text);
    for (const auto &piece : qwen35_pretokenize(normalized)) {
        encode_piece(piece, output);
    }
}

void Tokenizer::encode_piece(
    std::string_view piece,
    std::vector<std::uint32_t> &output) {
    std::string encoded;
    for (const auto byte : piece) {
        encoded += byte_encoder_.at(static_cast<unsigned char>(byte));
    }
    if (const auto found = cache_.find(encoded); found != cache_.end()) {
        cache_recency_.splice(
            cache_recency_.begin(), cache_recency_, found->second.recency);
        output.insert(
            output.end(),
            found->second.tokens.begin(),
            found->second.tokens.end());
        return;
    }
    std::vector<std::string> symbols;
    for (const auto codepoint : utf8_codepoints(encoded)) {
        std::string symbol;
        append_utf8(symbol, codepoint);
        symbols.push_back(std::move(symbol));
    }
    while (symbols.size() > 1) {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::string best_left;
        std::string best_right;
        for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
            const auto found =
                merge_ranks_.find(symbols[index] + " " + symbols[index + 1]);
            if (found != merge_ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_left = symbols[index];
                best_right = symbols[index + 1];
            }
        }
        if (best_rank == std::numeric_limits<std::uint32_t>::max()) {
            break;
        }
        std::vector<std::string> merged;
        for (std::size_t index = 0; index < symbols.size();) {
            if (index + 1 < symbols.size() && symbols[index] == best_left &&
                symbols[index + 1] == best_right) {
                merged.push_back(symbols[index] + symbols[index + 1]);
                index += 2;
            } else {
                merged.push_back(std::move(symbols[index]));
                ++index;
            }
        }
        symbols = std::move(merged);
    }
    std::vector<std::uint32_t> ids;
    ids.reserve(symbols.size());
    for (const auto &symbol : symbols) {
        const auto found = token_ids_.find(symbol);
        if (found == token_ids_.end()) {
            throw std::runtime_error("tokenizer: BPE symbol is absent from vocabulary");
        }
        ids.push_back(found->second);
    }
    constexpr std::size_t maximum_cache_entries = 4096;
    constexpr std::size_t maximum_cacheable_piece_bytes = 256;
    if (piece.size() <= maximum_cacheable_piece_bytes) {
        if (cache_.size() == maximum_cache_entries) {
            cache_.erase(cache_recency_.back());
            cache_recency_.pop_back();
        }
        cache_recency_.push_front(encoded);
        cache_.emplace(
            encoded,
            CacheEntry{ids, cache_recency_.begin()});
    }
    output.insert(output.end(), ids.begin(), ids.end());
}

std::string Tokenizer::decode(std::span<const std::uint32_t> tokens) const {
    std::string encoded;
    for (const auto token : tokens) {
        if (token >= tokens_.size()) {
            throw std::out_of_range("tokenizer: token is out of range");
        }
        encoded += tokens_[token];
    }
    std::string result;
    for (const auto codepoint : utf8_codepoints(encoded)) {
        const auto found = byte_decoder_.find(codepoint);
        if (found == byte_decoder_.end()) {
            append_utf8(result, codepoint);
        } else {
            result.push_back(static_cast<char>(found->second));
        }
    }
    return result;
}

std::string Tokenizer::token_text(std::uint32_t token) const {
    const std::uint32_t value[] = {token};
    return decode(value);
}

void Tokenizer::mask_unused_logits(std::span<float> logits) const {
    if (logits.size() != tokens_.size()) {
        throw std::invalid_argument("tokenizer: logits shape mismatch");
    }
    for (const auto token : unused_tokens_) {
        logits[token] = std::numeric_limits<float>::lowest();
    }
}

std::string qwen_user_prompt(std::string_view input) {
    const ChatMessage message{"user", std::string(input)};
    return qwen35_chat_prompt(std::span<const ChatMessage>(&message, 1));
}

} // namespace adi
