#pragma once

#include <filesystem>
#include <string_view>

namespace adi::test_detail {

#ifdef _WIN32
using CommandCharacter = wchar_t;

inline std::filesystem::path model_path(const wchar_t *value) {
    return value;
}

inline bool full_argument(const wchar_t *value) {
    return std::wstring_view(value) == L"--full";
}
#else
using CommandCharacter = char;

inline std::filesystem::path model_path(const char *value) {
    return value;
}

inline bool full_argument(const char *value) {
    return std::string_view(value) == "--full";
}
#endif

} // namespace adi::test_detail

#ifdef _WIN32
#define ADI_MODEL_TEST_MAIN wmain
#else
#define ADI_MODEL_TEST_MAIN main
#endif
