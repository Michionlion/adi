#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace adi {

class Json {
  public:
    using Array = std::vector<Json>;
    using Object = std::vector<std::pair<std::string, Json>>;

    Json() = default;
    explicit Json(bool value) : value_(value) {}
    explicit Json(double value) : value_(value) {}
    explicit Json(std::string value) : value_(std::move(value)) {}
    explicit Json(Array value) : value_(std::move(value)) {}
    explicit Json(Object value) : value_(std::move(value)) {}

    [[nodiscard]] bool is_null() const noexcept;
    [[nodiscard]] const bool *boolean() const noexcept;
    [[nodiscard]] const double *number() const noexcept;
    [[nodiscard]] const std::string *string() const noexcept;
    [[nodiscard]] const Array *array() const noexcept;
    [[nodiscard]] const Object *object() const noexcept;
    [[nodiscard]] const Json *find(std::string_view key) const noexcept;

  private:
    std::variant<std::monostate, bool, double, std::string, Array, Object> value_;
};

[[nodiscard]] Json parse_json(std::string_view input);
[[nodiscard]] std::string json_string(std::string_view input);
[[nodiscard]] std::string json_dump(const Json &value);

} // namespace adi
