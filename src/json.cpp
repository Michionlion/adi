#include "adi/json.hpp"
#include "utf8.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>

namespace adi {
namespace {

class Parser {
  public:
    explicit Parser(std::string_view input) : input_(input) {}

    Json parse() {
        auto value = parse_value(0);
        whitespace();
        if (position_ != input_.size()) {
            fail("trailing data");
        }
        return value;
    }

  private:
    [[noreturn]] void fail(std::string_view message) const {
        throw std::runtime_error(
            "JSON: " + std::string(message) + " at byte " +
            std::to_string(position_));
    }

    void whitespace() {
        while (position_ < input_.size() &&
               (input_[position_] == ' ' || input_[position_] == '\t' ||
                input_[position_] == '\r' || input_[position_] == '\n')) {
            ++position_;
        }
    }

    bool take(char value) {
        whitespace();
        if (position_ < input_.size() && input_[position_] == value) {
            ++position_;
            return true;
        }
        return false;
    }

    Json parse_value(std::size_t depth) {
        constexpr std::size_t maximum_depth = 128;
        whitespace();
        if (position_ >= input_.size()) {
            fail("expected value");
        }
        if (depth > maximum_depth) {
            fail("nesting exceeds 128 levels");
        }
        switch (input_[position_]) {
        case 'n':
            literal("null");
            return Json();
        case 't':
            literal("true");
            return Json(true);
        case 'f':
            literal("false");
            return Json(false);
        case '"':
            return Json(parse_string());
        case '[':
            return Json(parse_array(depth));
        case '{':
            return Json(parse_object(depth));
        default:
            return Json(parse_number());
        }
    }

    void literal(std::string_view value) {
        if (input_.substr(position_, value.size()) != value) {
            fail("invalid literal");
        }
        position_ += value.size();
    }

    std::uint32_t hex4() {
        if (position_ + 4 > input_.size()) {
            fail("truncated Unicode escape");
        }
        std::uint32_t value = 0;
        for (int index = 0; index < 4; ++index) {
            value <<= 4;
            const char digit = input_[position_++];
            if (digit >= '0' && digit <= '9') {
                value |= digit - '0';
            } else if (digit >= 'a' && digit <= 'f') {
                value |= digit - 'a' + 10;
            } else if (digit >= 'A' && digit <= 'F') {
                value |= digit - 'A' + 10;
            } else {
                fail("invalid Unicode escape");
            }
        }
        return value;
    }

    std::string parse_string() {
        if (!take('"')) {
            fail("expected string");
        }
        std::string result;
        while (position_ < input_.size()) {
            const char value = input_[position_++];
            if (value == '"') {
                if (!valid_utf8(result)) {
                    fail("invalid UTF-8 in string");
                }
                return result;
            }
            if (static_cast<unsigned char>(value) < 0x20U) {
                fail("control character in string");
            }
            if (value != '\\') {
                result.push_back(value);
                continue;
            }
            if (position_ >= input_.size()) {
                fail("truncated escape");
            }
            switch (input_[position_++]) {
            case '"':
                result.push_back('"');
                break;
            case '\\':
                result.push_back('\\');
                break;
            case '/':
                result.push_back('/');
                break;
            case 'b':
                result.push_back('\b');
                break;
            case 'f':
                result.push_back('\f');
                break;
            case 'n':
                result.push_back('\n');
                break;
            case 'r':
                result.push_back('\r');
                break;
            case 't':
                result.push_back('\t');
                break;
            case 'u': {
                auto codepoint = hex4();
                if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
                    if (position_ + 2 > input_.size() ||
                        input_[position_] != '\\' || input_[position_ + 1] != 'u') {
                        fail("missing low surrogate");
                    }
                    position_ += 2;
                    const auto low = hex4();
                    if (low < 0xDC00 || low > 0xDFFF) {
                        fail("invalid low surrogate");
                    }
                    codepoint =
                        0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
                } else if (codepoint >= 0xDC00 && codepoint <= 0xDFFF) {
                    fail("unpaired low surrogate");
                }
                append_utf8(result, codepoint);
                break;
            }
            default:
                fail("invalid escape");
            }
        }
        fail("unterminated string");
    }

    double parse_number() {
        whitespace();
        const auto begin = position_;
        if (position_ < input_.size() && input_[position_] == '-') {
            ++position_;
        }
        if (position_ >= input_.size() ||
            (input_[position_] < '0' || input_[position_] > '9')) {
            fail("invalid number");
        }
        if (input_[position_] == '0') {
            ++position_;
        } else {
            while (position_ < input_.size() &&
                   input_[position_] >= '0' && input_[position_] <= '9') {
                ++position_;
            }
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            if (position_ >= input_.size() ||
                input_[position_] < '0' || input_[position_] > '9') {
                fail("invalid fractional number");
            }
            while (position_ < input_.size() &&
                   input_[position_] >= '0' && input_[position_] <= '9') {
                ++position_;
            }
        }
        if (position_ < input_.size() &&
            (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() &&
                (input_[position_] == '+' || input_[position_] == '-')) {
                ++position_;
            }
            if (position_ >= input_.size() ||
                input_[position_] < '0' || input_[position_] > '9') {
                fail("invalid number exponent");
            }
            while (position_ < input_.size() &&
                   input_[position_] >= '0' && input_[position_] <= '9') {
                ++position_;
            }
        }
        double result = 0.0;
        const auto [end, error] = std::from_chars(
            input_.data() + begin, input_.data() + position_, result);
        if (error != std::errc{} || end != input_.data() + position_ ||
            !std::isfinite(result)) {
            fail("invalid number");
        }
        return result;
    }

    Json::Array parse_array(std::size_t depth) {
        (void)take('[');
        Json::Array result;
        if (take(']')) {
            return result;
        }
        do {
            result.push_back(parse_value(depth + 1));
        } while (take(','));
        if (!take(']')) {
            fail("expected ']'");
        }
        return result;
    }

    Json::Object parse_object(std::size_t depth) {
        (void)take('{');
        Json::Object result;
        std::unordered_set<std::string> keys;
        if (take('}')) {
            return result;
        }
        do {
            whitespace();
            if (position_ >= input_.size() || input_[position_] != '"') {
                fail("expected object key");
            }
            auto key = parse_string();
            if (!take(':')) {
                fail("expected ':'");
            }
            if (!keys.emplace(key).second) {
                fail("duplicate object key");
            }
            result.emplace_back(std::move(key), parse_value(depth + 1));
        } while (take(','));
        if (!take('}')) {
            fail("expected '}'");
        }
        return result;
    }

    std::string_view input_;
    std::size_t position_ = 0;
};

} // namespace

bool Json::is_null() const noexcept {
    return std::holds_alternative<std::monostate>(value_);
}

const bool *Json::boolean() const noexcept {
    return std::get_if<bool>(&value_);
}

const double *Json::number() const noexcept {
    return std::get_if<double>(&value_);
}

const std::string *Json::string() const noexcept {
    return std::get_if<std::string>(&value_);
}

const Json::Array *Json::array() const noexcept {
    return std::get_if<Array>(&value_);
}

const Json::Object *Json::object() const noexcept {
    return std::get_if<Object>(&value_);
}

const Json *Json::find(std::string_view key) const noexcept {
    const auto *items = object();
    if (items == nullptr) {
        return nullptr;
    }
    for (const auto &[candidate, value] : *items) {
        if (candidate == key) {
            return &value;
        }
    }
    return nullptr;
}

Json parse_json(std::string_view input) {
    return Parser(input).parse();
}

std::string json_string(std::string_view input) {
    const auto clean = sanitize_utf8(input);
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (const auto raw : clean) {
        const auto value = static_cast<unsigned char>(raw);
        switch (value) {
        case '"':
            result += "\\\"";
            break;
        case '\\':
            result += "\\\\";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (value < 0x20U) {
                result += "\\u00";
                result.push_back(hex[value >> 4]);
                result.push_back(hex[value & 0x0FU]);
            } else {
                result.push_back(static_cast<char>(value));
            }
        }
    }
    result.push_back('"');
    return result;
}

} // namespace adi
