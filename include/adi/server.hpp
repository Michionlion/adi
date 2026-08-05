#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace adi {

struct ServerOptions {
    std::filesystem::path model;
    std::string host = "127.0.0.1";
    std::uint16_t port = 8080;
};

[[noreturn]] void serve(const ServerOptions &options);

} // namespace adi
