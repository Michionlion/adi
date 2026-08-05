#include "adi/tokenizer.hpp"

#include <algorithm>
#include <cctype>
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

std::pair<std::uint32_t, std::size_t> next_codepoint(
    std::string_view text,
    std::size_t position) {
    const auto first = static_cast<unsigned char>(text[position]);
    if (first < 0x80U) {
        return {first, 1};
    }
    std::size_t length = 0;
    std::uint32_t value = 0;
    if ((first & 0xE0U) == 0xC0U) {
        length = 2;
        value = first & 0x1FU;
    } else if ((first & 0xF0U) == 0xE0U) {
        length = 3;
        value = first & 0x0FU;
    } else if ((first & 0xF8U) == 0xF0U) {
        length = 4;
        value = first & 0x07U;
    } else {
        throw std::runtime_error("tokenizer: invalid UTF-8");
    }
    if (position + length > text.size()) {
        throw std::runtime_error("tokenizer: truncated UTF-8");
    }
    for (std::size_t index = 1; index < length; ++index) {
        const auto byte = static_cast<unsigned char>(text[position + index]);
        if ((byte & 0xC0U) != 0x80U) {
            throw std::runtime_error("tokenizer: invalid UTF-8 continuation");
        }
        value = (value << 6) | (byte & 0x3FU);
    }
    return {value, length};
}

bool ascii_letter(unsigned char value) {
    return std::isalpha(value) != 0 || value >= 0x80U;
}

bool ascii_digit(unsigned char value) {
    return std::isdigit(value) != 0;
}

bool whitespace(unsigned char value) {
    return std::isspace(value) != 0;
}

std::vector<std::string_view> pretokenize(std::string_view text) {
    std::vector<std::string_view> pieces;
    std::size_t position = 0;
    while (position < text.size()) {
        const auto begin = position;
        const auto current = static_cast<unsigned char>(text[position]);
        if (current == '\'' && position + 1 < text.size()) {
            constexpr std::string_view contractions[] = {
                "'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
            };
            bool matched = false;
            for (const auto contraction : contractions) {
                if (position + contraction.size() <= text.size()) {
                    std::string candidate(text.substr(position, contraction.size()));
                    std::transform(
                        candidate.begin(), candidate.end(), candidate.begin(),
                        [](unsigned char value) { return std::tolower(value); });
                    if (candidate == contraction) {
                        position += contraction.size();
                        matched = true;
                        break;
                    }
                }
            }
            if (matched) {
                pieces.push_back(text.substr(begin, position - begin));
                continue;
            }
        }

        std::size_t letter_start = position;
        if (!ascii_letter(current) && !ascii_digit(current) &&
            current != '\r' && current != '\n' && !whitespace(current) &&
            position + 1 < text.size() &&
            ascii_letter(static_cast<unsigned char>(text[position + 1]))) {
            ++letter_start;
        } else if (current == ' ' && position + 1 < text.size() &&
                   ascii_letter(static_cast<unsigned char>(text[position + 1]))) {
            ++letter_start;
        }
        if (letter_start != position || ascii_letter(current)) {
            position = letter_start;
            while (position < text.size() &&
                   ascii_letter(static_cast<unsigned char>(text[position]))) {
                ++position;
            }
            pieces.push_back(text.substr(begin, position - begin));
            continue;
        }
        if (ascii_digit(current)) {
            ++position;
            pieces.push_back(text.substr(begin, 1));
            continue;
        }
        const bool space_before_punctuation =
            current == ' ' && position + 1 < text.size() &&
            !whitespace(static_cast<unsigned char>(text[position + 1])) &&
            !ascii_letter(static_cast<unsigned char>(text[position + 1])) &&
            !ascii_digit(static_cast<unsigned char>(text[position + 1]));
        if (!whitespace(current) || space_before_punctuation) {
            if (space_before_punctuation) {
                ++position;
            }
            while (position < text.size()) {
                const auto value = static_cast<unsigned char>(text[position]);
                if (whitespace(value) || ascii_letter(value) || ascii_digit(value)) {
                    break;
                }
                ++position;
            }
            while (position < text.size() &&
                   (text[position] == '\r' || text[position] == '\n')) {
                ++position;
            }
            pieces.push_back(text.substr(begin, position - begin));
            continue;
        }
        while (position < text.size() &&
               whitespace(static_cast<unsigned char>(text[position]))) {
            ++position;
            if (text[position - 1] == '\r' || text[position - 1] == '\n') {
                while (position < text.size() &&
                       (text[position] == '\r' || text[position] == '\n')) {
                    ++position;
                }
                break;
            }
        }
        if (position - begin > 1 && text[position - 1] == ' ' &&
            position < text.size() &&
            !ascii_letter(static_cast<unsigned char>(text[position])) &&
            !ascii_digit(static_cast<unsigned char>(text[position]))) {
            --position;
        }
        pieces.push_back(text.substr(begin, position - begin));
    }
    return pieces;
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
        token_types_.size() != tokens_.size() || merges.empty() || !bos || !eos) {
        throw std::runtime_error("tokenizer: missing GGUF tokenizer metadata");
    }
    bos_token_ = static_cast<std::uint32_t>(*bos);
    eos_token_ = static_cast<std::uint32_t>(*eos);
    token_ids_.reserve(tokens_.size());
    for (std::uint32_t index = 0; index < tokens_.size(); ++index) {
        token_ids_.emplace(tokens_[index], index);
        if (token_types_[index] == 3 || token_types_[index] == 4) {
            special_tokens_.emplace_back(tokens_[index], index);
        }
    }
    std::sort(
        special_tokens_.begin(), special_tokens_.end(),
        [](const auto &left, const auto &right) {
            return left.first.size() > right.first.size();
        });
    merge_ranks_.reserve(merges.size());
    for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
        merge_ranks_.emplace(std::string(merges[rank]), rank);
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
    for (const auto piece : pretokenize(text)) {
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
        output.insert(output.end(), found->second.begin(), found->second.end());
        return;
    }
    std::vector<std::string> symbols;
    for (std::size_t position = 0; position < encoded.size();) {
        const auto [ignored, length] = next_codepoint(encoded, position);
        (void)ignored;
        symbols.emplace_back(encoded.substr(position, length));
        position += length;
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
    cache_.emplace(encoded, ids);
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
    for (std::size_t position = 0; position < encoded.size();) {
        const auto [codepoint, length] = next_codepoint(encoded, position);
        const auto found = byte_decoder_.find(codepoint);
        if (found == byte_decoder_.end()) {
            result.append(encoded.substr(position, length));
        } else {
            result.push_back(static_cast<char>(found->second));
        }
        position += length;
    }
    return result;
}

std::string Tokenizer::token_text(std::uint32_t token) const {
    const std::uint32_t value[] = {token};
    return decode(value);
}

std::string qwen_user_prompt(std::string_view input) {
    return "<|im_start|>user\n" + std::string(input) +
           "<|im_end|>\n<|im_start|>assistant\n<think>\n";
}

} // namespace adi
