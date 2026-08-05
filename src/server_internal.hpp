#pragma once

#include "adi/generation.hpp"

#include <chrono>
#include <ctime>
#include <string>
#include <string_view>

namespace adi::server_detail {

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
    int socket,
    std::chrono::steady_clock::time_point deadline) noexcept;

} // namespace adi::server_detail
