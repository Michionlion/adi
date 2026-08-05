#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace adi {

using BpeRuleMap = std::unordered_map<
    std::uint64_t,
    std::pair<std::uint32_t, std::uint32_t>>;

[[nodiscard]] constexpr std::uint64_t bpe_pair_key(
    std::uint32_t left,
    std::uint32_t right) noexcept {
    return (static_cast<std::uint64_t>(left) << 32) | right;
}

[[nodiscard]] std::vector<std::uint32_t> bpe_merge(
    std::span<const std::uint32_t> symbols,
    const BpeRuleMap &rules,
    const std::function<bool()> &cancelled = {});

} // namespace adi
