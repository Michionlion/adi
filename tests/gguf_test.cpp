#include "adi/gguf.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <unistd.h>

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

std::filesystem::path write_fixture(bool truncate = false) {
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

    const auto path = std::filesystem::temp_directory_path() /
                      ("adi-gguf-test-" + std::to_string(::getpid()) +
                       (truncate ? "-bad.gguf" : ".gguf"));
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return path;
}

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
