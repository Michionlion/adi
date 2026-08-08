#include "adi/gguf.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {

template <typename T>
void append(std::vector<std::byte> &bytes, T value) {
    const auto *begin = reinterpret_cast<const std::byte *>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(value));
}

void append_string(std::vector<std::byte> &bytes, std::string_view value) {
    append(bytes, static_cast<std::uint64_t>(value.size()));
    const auto *begin = reinterpret_cast<const std::byte *>(value.data());
    bytes.insert(bytes.end(), begin, begin + value.size());
}

std::filesystem::path write_fixture(
    bool truncate = false,
    bool unicode_path = false) {
    static std::atomic<std::uint64_t> fixture_counter = 0;
    std::vector<std::byte> bytes;
    append(bytes, std::uint32_t{0x46554747});
    append(bytes, std::uint32_t{3});
    append(bytes, std::uint64_t{1});
    append(bytes, std::uint64_t{5});

    append_string(bytes, "general.alignment");
    append(bytes, std::uint32_t{4});
    append(bytes, std::uint32_t{32});
    append_string(bytes, "general.architecture");
    append(bytes, std::uint32_t{8});
    append_string(bytes, "adi");
    append_string(bytes, "adi.additive");
    append(bytes, std::uint32_t{7});
    append(bytes, std::uint8_t{1});
    append_string(bytes, "tokenizer.ggml.tokens");
    append(bytes, std::uint32_t{9});
    append(bytes, std::uint32_t{8});
    append(bytes, std::uint64_t{2});
    append_string(bytes, "hello");
    append_string(bytes, "world");
    append_string(bytes, "tokenizer.ggml.token_type");
    append(bytes, std::uint32_t{9});
    append(bytes, std::uint32_t{4});
    append(bytes, std::uint64_t{2});
    append(bytes, std::uint32_t{1});
    append(bytes, std::uint32_t{3});

    append_string(bytes, "test.weight");
    append(bytes, std::uint32_t{2});
    append(bytes, std::uint64_t{2});
    append(bytes, std::uint64_t{2});
    append(bytes, std::uint32_t{0});
    append(bytes, std::uint64_t{0});

    while (bytes.size() % 32 != 0) {
        bytes.push_back(std::byte{0});
    }
    for (float value : {1.0F, 2.0F, 3.0F, 4.0F}) {
        append(bytes, value);
    }
    if (truncate) {
        bytes.resize(bytes.size() - 1);
    }

    const auto clock_nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const auto process_nonce =
        reinterpret_cast<std::uintptr_t>(&fixture_counter);
    const auto instance_nonce = fixture_counter.fetch_add(1);
    auto file_name = unicode_path
                         ? std::filesystem::path(
                               u8"adi-gguf-\u6a21\u578b-test-")
                         : std::filesystem::path("adi-gguf-test-");
    file_name += std::to_string(clock_nonce) + "-" +
                 std::to_string(process_nonce) + "-" +
                 std::to_string(instance_nonce) +
                 (truncate ? "-bad.gguf" : ".gguf");
    const auto path = std::filesystem::temp_directory_path() / file_name;
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return path;
}

#ifdef _WIN32
std::wstring extended_path(const std::filesystem::path &path) {
    const auto native = std::filesystem::absolute(path).native();
    if (native.starts_with(L"\\\\")) {
        return L"\\\\?\\UNC\\" + native.substr(2);
    }
    return L"\\\\?\\" + native;
}

void check_long_path() {
    const auto source = write_fixture();
    auto root = source;
    root += ".long";
    const auto native_root = extended_path(root);
    assert(::CreateDirectoryW(native_root.c_str(), nullptr) != 0);
    auto directory = root;
    while ((directory / "fixture.gguf").native().size() < 300) {
        directory /= "segment-0123456789-abcdefghijklmnopqrstuvwxyz";
        const auto native = extended_path(directory);
        assert(::CreateDirectoryW(native.c_str(), nullptr) != 0);
    }
    const auto path = directory / "fixture.gguf";
    const auto native_path = extended_path(path);
    assert(::CreateHardLinkW(native_path.c_str(), source.c_str(), nullptr) != 0);
    {
        const adi::GgufFile file(path);
        assert(file.find_tensor("test.weight") != nullptr);
    }
    assert(::DeleteFileW(native_path.c_str()) != 0);
    for (;;) {
        const auto parent = directory.parent_path();
        const auto native_directory = extended_path(directory);
        assert(::RemoveDirectoryW(native_directory.c_str()) != 0);
        if (directory == root) {
            break;
        }
        directory = parent;
    }
    std::filesystem::remove(source);
}
#endif

} // namespace

int main() {
    const auto path = write_fixture();
    {
        const adi::GgufFile file(path);
        assert(file.version() == 3);
        assert(file.alignment() == 32);
        assert(file.metadata().size() == 5);
        assert(file.tensors().size() == 1);
        assert(file.integer("general.alignment") == 32);
        assert(file.string("general.architecture") == "adi");
        assert(file.boolean("adi.additive") == true);
        assert((file.string_array("tokenizer.ggml.tokens") ==
                std::vector<std::string_view>{"hello", "world"}));
        assert((file.integer_array("tokenizer.ggml.token_type") ==
                std::vector<std::uint64_t>{1, 3}));

        const auto *tensor = file.find_tensor("test.weight");
        assert(tensor != nullptr);
        assert((tensor->dimensions == std::vector<std::uint64_t>{2, 2}));
        assert(tensor->type == adi::GgmlType::f32);
        assert(tensor->elements == 4);
        assert(tensor->bytes == 16);
        const auto data = file.tensor_data(*tensor);
        float first = 0.0F;
        float last = 0.0F;
        std::memcpy(&first, data.data(), sizeof(first));
        std::memcpy(&last, data.data() + 3 * sizeof(float), sizeof(last));
        assert(first == 1.0F);
        assert(last == 4.0F);
    }
    std::filesystem::remove(path);

    const auto unicode_path = write_fixture(false, true);
    {
        const adi::GgufFile file(unicode_path);
        assert(file.find_tensor("test.weight") != nullptr);
    }
    std::filesystem::remove(unicode_path);

#ifdef _WIN32
    check_long_path();
#endif

    const auto truncated_path = write_fixture(true);
    bool rejected = false;
    try {
        const adi::GgufFile ignored(truncated_path);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    std::filesystem::remove(truncated_path);
    assert(rejected);
}
