#pragma once

#include <cstdint>
#include <span>

namespace adi::detail {

// Query heads are grouped by KV head. The output uses the same
// [query_heads, head_size] layout as queries.
void grouped_query_online_attention(
    std::span<const float> queries,
    std::span<const float> keys,
    std::span<const float> values,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_size,
    std::span<float> output);

} // namespace adi::detail
