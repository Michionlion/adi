#pragma once

#include "adi/generation.hpp"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <string>
#include <string_view>

namespace adi::server_detail {

using SocketHandle = std::intptr_t;

struct ResponseIdentity {
    std::string id;
    std::string message_id;
    std::time_t created_at;
};

[[nodiscard]] std::string response_json(
    std::string_view model_name,
    const GenerationResult &result,
    const ResponseIdentity &identity);

[[nodiscard]] bool connection_cancelled(
    SocketHandle socket,
    std::chrono::steady_clock::time_point deadline) noexcept;

} // namespace adi::server_detail
