#include "adi/adi.hpp"
#include "adi/gguf.hpp"
#include "adi/model.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

std::uint32_t parse_u32(std::string_view value, std::string_view name) {
    std::uint32_t result = 0;
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid " + std::string(name));
    }
    return result;
}

adi::ExpertProjection parse_projection(std::string_view value) {
    if (value == "gate") {
        return adi::ExpertProjection::gate;
    }
    if (value == "up") {
        return adi::ExpertProjection::up;
    }
    if (value == "down") {
        return adi::ExpertProjection::down;
    }
    throw std::invalid_argument("projection must be gate, up, or down");
}

int validate(const char *path) {
    const adi::MachModel model(path);
    const auto &config = model.config();
    std::cout << "valid Mach additive model\n"
              << "layers: " << config.layers << '\n'
              << "hidden: " << config.hidden << '\n'
              << "experts: " << config.experts << '\n'
              << "active experts: " << config.active_experts << '\n'
              << "expert hidden: " << config.expert_hidden << '\n'
              << "context: " << config.context << '\n';
    return 0;
}

int bench_expert(int argc, char **argv) {
    const adi::MachModel model(argv[2]);
    const auto layer = parse_u32(argv[3], "layer");
    const auto expert = parse_u32(argv[4], "expert");
    const auto projection = parse_projection(argv[5]);
    const auto iterations = argc == 7 ? parse_u32(argv[6], "iterations") : 3;
    if (iterations == 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    const auto matrix = model.expert(layer, expert, projection);
    std::vector<float> input(matrix.columns);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.01F);
    }
    std::vector<float> output(matrix.rows);
    adi::ExpertScratch scratch;

    adi::mach_expert_matvec(matrix, input, output, scratch);
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        adi::mach_expert_matvec(matrix, input, output, scratch);
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    double checksum = 0.0;
    float max_abs = 0.0F;
    for (const auto value : output) {
        checksum += value;
        max_abs = std::max(max_abs, std::abs(value));
    }
    std::cout << "matrix: " << matrix.rows << "x" << matrix.columns << '\n'
              << "iterations: " << iterations << '\n'
              << "milliseconds/matvec: " << elapsed * 1000.0 / iterations << '\n'
              << "checksum: " << checksum << '\n'
              << "max_abs: " << max_abs << '\n';
    return 0;
}

void usage() {
    std::cerr << "usage:\n"
              << "  adi --version\n"
              << "  adi inspect MODEL.gguf\n"
              << "  adi validate MODEL.gguf\n"
              << "  adi bench-expert MODEL.gguf LAYER EXPERT PROJECTION [ITERATIONS]\n";
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
    if (argc == 3 && std::string_view(argv[1]) == "validate") {
        try {
            return validate(argv[2]);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if ((argc == 6 || argc == 7) && std::string_view(argv[1]) == "bench-expert") {
        try {
            return bench_expert(argc, argv);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }

    usage();
    return 2;
}
