#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>

namespace adi::detail {

template <std::size_t Size>
const char *environment_variable(
    const char *name,
    std::array<char, Size> &buffer) noexcept {
#if defined(_MSC_VER)
    std::size_t required = 0;
    if (getenv_s(&required, buffer.data(), buffer.size(), name) != 0 ||
        required == 0 || required > buffer.size()) {
        return nullptr;
    }
    return buffer.data();
#else
    (void)buffer;
    return std::getenv(name);
#endif
}

} // namespace adi::detail
