#pragma once

#include "adi/options.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace adi {

struct ServerOptions {
    std::filesystem::path model;
    std::string host = "127.0.0.1";
    std::uint16_t port = 9932;
    ExecutionOptions execution;
};

[[noreturn]] void serve(const ServerOptions &options);

} // namespace adi
