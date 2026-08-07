#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace adi::generation_detail {

[[nodiscard]] std::uint32_t resolved_output_limit(
    std::uint32_t context,
    std::size_t prompt_tokens,
    std::optional<std::uint32_t> requested_limit);

} // namespace adi::generation_detail
