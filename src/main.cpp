#include "adi/adi.hpp"
#include "adi/gguf.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <string_view>

namespace {

int inspect(const char *path) {
    const adi::GgufFile file(path);
    std::cout << "GGUF v" << file.version() << '\n'
              << "alignment: " << file.alignment() << '\n'
              << "metadata: " << file.metadata().size() << '\n'
              << "tensors: " << file.tensors().size() << '\n';

    for (const auto &entry : file.metadata()) {
        std::cout << "  " << entry.key << " ("
                  << static_cast<std::uint32_t>(entry.type) << ')';
        if (const auto value = file.string(entry.key)) {
            std::cout << " = " << *value;
        } else if (const auto value = file.integer(entry.key)) {
            std::cout << " = " << *value;
        } else if (const auto value = file.number(entry.key)) {
            std::cout << " = " << *value;
        } else if (const auto value = file.boolean(entry.key)) {
            std::cout << " = " << (*value ? "true" : "false");
        }
        std::cout << '\n';
    }

    for (const auto &tensor : file.tensors()) {
        std::cout << "  " << tensor.name << " [";
        for (std::size_t index = 0; index < tensor.dimensions.size(); ++index) {
            if (index != 0) {
                std::cout << ", ";
            }
            std::cout << tensor.dimensions[index];
        }
        std::cout << "] " << adi::ggml_type_name(tensor.type)
                  << " (" << tensor.bytes << " bytes)\n";
    }
    return 0;
}

void usage() {
    std::cerr << "usage:\n"
              << "  adi --version\n"
              << "  adi inspect MODEL.gguf\n";
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "adi " << adi::version() << '\n';
        return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "inspect") {
        try {
            return inspect(argv[2]);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }

    usage();
    return 2;
}
