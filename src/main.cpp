#include "adi/adi.hpp"
#include "adi/executor.hpp"
#include "adi/generation.hpp"
#include "adi/gguf.hpp"
#include "adi/model.hpp"
#include "adi/server.hpp"
#include "adi/tokenizer.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <numeric>
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
              << "context: " << config.context << '\n'
              << "vocabulary: " << config.vocabulary << '\n';
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

template <typename Function>
void print_benchmark(
    std::uint32_t iterations,
    std::span<const float> output,
    Function &&function) {
    function();
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        function();
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    double checksum = 0.0;
    float max_abs = 0.0F;
    for (const auto value : output) {
        checksum += value;
        max_abs = std::max(max_abs, std::abs(value));
    }
    std::cout << "iterations: " << iterations << '\n'
              << "milliseconds/matvec: " << elapsed * 1000.0 / iterations << '\n'
              << "checksum: " << checksum << '\n'
              << "max_abs: " << max_abs << '\n';
}

int bench_ne(int argc, char **argv) {
    const adi::MachModel model(argv[2]);
    const auto layer = parse_u32(argv[3], "layer");
    const auto matrix = model.non_expert(layer, argv[4]);
    const auto iterations = argc >= 6 ? parse_u32(argv[5], "iterations") : 3;
    const auto batch = argc == 7 ? parse_u32(argv[6], "batch") : 1;
    if (iterations == 0 || batch == 0) {
        throw std::invalid_argument("iterations and batch must be positive");
    }
    std::vector<float> inputs(
        static_cast<std::size_t>(batch) * matrix.columns);
    for (std::uint32_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (std::uint32_t column = 0; column < matrix.columns; ++column) {
            inputs[
                static_cast<std::size_t>(batch_index) * matrix.columns +
                column] =
                std::sin(
                    static_cast<float>(column + batch_index) * 0.01F);
        }
    }
    std::vector<float> outputs(
        static_cast<std::size_t>(batch) * matrix.rows);
    adi::ExpertScratch scratch;
    adi::mach_ne_matmul(matrix, inputs, batch, outputs, scratch);
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        adi::mach_ne_matmul(matrix, inputs, batch, outputs, scratch);
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    double checksum = 0.0;
    for (const auto value : outputs) {
        checksum += value;
    }
    std::cout << "matrix: " << matrix.rows << "x" << matrix.columns << '\n'
              << "batch: " << batch << '\n'
              << "iterations: " << iterations << '\n'
              << "milliseconds/batch: "
              << elapsed * 1000.0 / iterations << '\n'
              << "milliseconds/vector: "
              << elapsed * 1000.0 / iterations / batch << '\n'
              << "checksum: " << checksum << '\n';
    return 0;
}

int bench_head(int argc, char **argv) {
    const adi::MachModel model(argv[2]);
    const auto chunk_index = parse_u32(argv[3], "chunk");
    const auto head = model.head_chunk(chunk_index);
    const auto iterations = argc >= 5 ? parse_u32(argv[4], "iterations") : 1;
    const auto batch = argc == 6 ? parse_u32(argv[5], "batch") : 1;
    if (iterations == 0 || batch == 0) {
        throw std::invalid_argument("iterations and batch must be positive");
    }
    std::vector<float> inputs(
        static_cast<std::size_t>(batch) * head.columns);
    for (std::uint32_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (std::uint32_t column = 0; column < head.columns; ++column) {
            inputs[
                static_cast<std::size_t>(batch_index) * head.columns +
                column] =
                std::sin(
                    static_cast<float>(column + batch_index) * 0.01F);
        }
    }
    std::vector<float> outputs(
        static_cast<std::size_t>(batch) * head.rows);
    adi::mach_head_matmul(head, inputs, batch, outputs);
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        adi::mach_head_matmul(head, inputs, batch, outputs);
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    double checksum = 0.0;
    for (const auto value : outputs) {
        checksum += value;
    }
    std::cout << "matrix: " << head.rows << "x" << head.columns << '\n'
              << "batch: " << batch << '\n'
              << "iterations: " << iterations << '\n'
              << "milliseconds/batch: "
              << elapsed * 1000.0 / iterations << '\n'
              << "milliseconds/vector: "
              << elapsed * 1000.0 / iterations / batch << '\n'
              << "checksum: " << checksum << '\n';
    return 0;
}

int embedding_row(const char *path, std::string_view token_string) {
    const adi::MachModel model(path);
    const auto token = parse_u32(token_string, "token");
    const auto embedding = model.embedding();
    std::vector<float> output(embedding.columns);
    adi::mach_embedding_row(embedding, token, output);
    double checksum = 0.0;
    float max_abs = 0.0F;
    for (const auto value : output) {
        checksum += value;
        max_abs = std::max(max_abs, std::abs(value));
    }
    std::cout << "token: " << token << '\n'
              << "checksum: " << checksum << '\n'
              << "max_abs: " << max_abs << '\n';
    return 0;
}

int bench_moe(int argc, char **argv) {
    const adi::MachModel model(argv[2]);
    const auto layer = parse_u32(argv[3], "layer");
    const auto iterations = argc == 5 ? parse_u32(argv[4], "iterations") : 1;
    if (iterations == 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    std::vector<float> input(model.config().hidden);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.01F);
    }
    std::vector<float> output(model.config().hidden);
    adi::MoeScratch scratch;
    std::array<adi::ExpertRoute, 8> routes;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        routes = adi::moe_forward(model, layer, input, output, scratch);
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    double checksum = 0.0;
    for (const auto value : output) {
        checksum += value;
    }
    std::cout << "routes:";
    for (const auto route : routes) {
        std::cout << ' ' << route.expert << ':' << route.weight;
    }
    std::cout << '\n'
              << "milliseconds/forward: " << elapsed * 1000.0 / iterations << '\n'
              << "checksum: " << checksum << '\n';
    return 0;
}

int bench_attention(int argc, char **argv) {
    const adi::MachModel model(argv[2]);
    const auto layer = parse_u32(argv[3], "layer");
    const auto tokens = argc == 5 ? parse_u32(argv[4], "tokens") : 1;
    if (tokens == 0) {
        throw std::invalid_argument("tokens must be positive");
    }
    std::vector<float> input(model.config().hidden);
    std::vector<float> output(model.config().hidden);
    adi::FullAttentionState state;
    adi::FullAttentionScratch scratch;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t position = 0; position < tokens; ++position) {
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] =
                std::sin(static_cast<float>(index + position) * 0.01F);
        }
        adi::full_attention_forward(
            model, layer, position, input, output, state, scratch);
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    double checksum = 0.0;
    for (const auto value : output) {
        checksum += value;
    }
    std::cout << "tokens: " << tokens << '\n'
              << "milliseconds/token: " << elapsed * 1000.0 / tokens << '\n'
              << "checksum: " << checksum << '\n';
    return 0;
}

int bench_linear_attention(int argc, char **argv) {
    const adi::MachModel model(argv[2]);
    const auto layer = parse_u32(argv[3], "layer");
    const auto tokens = argc == 5 ? parse_u32(argv[4], "tokens") : 1;
    if (tokens == 0) {
        throw std::invalid_argument("tokens must be positive");
    }
    std::vector<float> input(model.config().hidden);
    std::vector<float> output(model.config().hidden);
    adi::LinearAttentionState state;
    adi::LinearAttentionScratch scratch;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t position = 0; position < tokens; ++position) {
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] =
                std::sin(static_cast<float>(index + position) * 0.01F);
        }
        adi::linear_attention_forward(
            model, layer, input, output, state, scratch);
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    double checksum = 0.0;
    for (const auto value : output) {
        checksum += value;
    }
    std::cout << "tokens: " << tokens << '\n'
              << "milliseconds/token: " << elapsed * 1000.0 / tokens << '\n'
              << "checksum: " << checksum << '\n';
    return 0;
}

int decode_one(const char *path, std::string_view token_string) {
    const adi::MachModel model(path);
    const auto token = parse_u32(token_string, "token");
    std::vector<float> logits(model.config().vocabulary);
    adi::DecoderState state;
    adi::DecoderScratch scratch;
    const auto start = std::chrono::steady_clock::now();
    adi::decode_token(model, token, state, logits, scratch);
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::vector<std::uint32_t> indexes(logits.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::partial_sort(
        indexes.begin(),
        indexes.begin() + 5,
        indexes.end(),
        [&](std::uint32_t left, std::uint32_t right) {
            return logits[left] > logits[right];
        });
    std::cout << "seconds: " << elapsed << '\n'
              << "tokens/second: " << 1.0 / elapsed << '\n'
              << "top logits:";
    for (std::size_t index = 0; index < 5; ++index) {
        std::cout << ' ' << indexes[index] << ':' << logits[indexes[index]];
    }
    std::cout << '\n';
    return 0;
}

int decode_batch(
    const char *path,
    std::string_view token_string,
    std::string_view batch_string) {
    const adi::MachModel model(path);
    const auto token = parse_u32(token_string, "token");
    const auto batch = parse_u32(batch_string, "batch");
    if (batch == 0) {
        throw std::invalid_argument("batch must be positive");
    }
    std::vector<std::uint32_t> tokens(batch, token);
    std::vector<adi::DecoderState> states(batch);
    std::vector<adi::DecoderScratch> scratches(batch);
    std::vector<float> logits(
        static_cast<std::size_t>(batch) * model.config().vocabulary);
    adi::DecoderBatchScratch batch_scratch;
    const auto start = std::chrono::steady_clock::now();
    adi::decode_batch(
        model,
        tokens,
        states,
        logits,
        scratches,
        batch_scratch);
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "batch: " << batch << '\n'
              << "seconds: " << elapsed << '\n'
              << "sequences/second: "
              << static_cast<double>(batch) / elapsed << '\n';
    return 0;
}

int tokenize(const char *path, std::string_view text) {
    const adi::MachModel model(path);
    adi::Tokenizer tokenizer(model);
    const auto tokens = tokenizer.encode(text);
    std::cout << "tokens:";
    for (const auto token : tokens) {
        std::cout << ' ' << token;
    }
    std::cout << '\n' << "decoded: " << tokenizer.decode(tokens) << '\n';
    return 0;
}

int generate_text(int argc, char **argv) {
    const adi::MachModel model(argv[2]);
    adi::Tokenizer tokenizer(model);
    adi::GenerationOptions options;
    options.max_output_tokens = argc == 5 ? parse_u32(argv[4], "max tokens") : 16;
    options.temperature = 0.0F;
    const auto start = std::chrono::steady_clock::now();
    const auto result = adi::generate(model, tokenizer, argv[3], options);
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << result.text << '\n'
              << "input_tokens: " << result.input_tokens << '\n'
              << "output_tokens: " << result.output_tokens << '\n'
              << "seconds: " << elapsed << '\n';
    return 0;
}

int serve_command(int argc, char **argv) {
    adi::ServerOptions options;
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--model" && index + 1 < argc) {
            options.model = argv[++index];
        } else if (argument == "--host" && index + 1 < argc) {
            options.host = argv[++index];
        } else if (argument == "--port" && index + 1 < argc) {
            const auto port = parse_u32(argv[++index], "port");
            if (port == 0 || port > 65535) {
                throw std::invalid_argument("port is out of range");
            }
            options.port = static_cast<std::uint16_t>(port);
        } else {
            throw std::invalid_argument("invalid serve argument");
        }
    }
    if (options.model.empty()) {
        throw std::invalid_argument("--model is required");
    }
    adi::serve(options);
}

void usage() {
    std::cerr << "usage:\n"
              << "  adi --version\n"
              << "  adi inspect MODEL.gguf\n"
              << "  adi validate MODEL.gguf\n"
              << "  adi bench-expert MODEL.gguf LAYER EXPERT PROJECTION [ITERATIONS]\n"
              << "  adi bench-ne MODEL.gguf LAYER SOURCE_NAME [ITERATIONS] [BATCH]\n"
              << "  adi bench-head MODEL.gguf CHUNK [ITERATIONS] [BATCH]\n"
              << "  adi bench-moe MODEL.gguf LAYER [ITERATIONS]\n"
              << "  adi bench-attention MODEL.gguf LAYER [TOKENS]\n"
              << "  adi bench-linear MODEL.gguf LAYER [TOKENS]\n"
              << "  adi decode-token MODEL.gguf TOKEN\n"
              << "  adi decode-batch MODEL.gguf TOKEN BATCH\n"
              << "  adi tokenize MODEL.gguf TEXT\n"
              << "  adi generate MODEL.gguf PROMPT [MAX_TOKENS]\n"
              << "  adi serve --model MODEL.gguf [--host ADDRESS] [--port PORT]\n"
              << "  adi embedding-row MODEL.gguf TOKEN\n";
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
    if ((argc == 5 || argc == 6 || argc == 7) &&
        std::string_view(argv[1]) == "bench-ne") {
        try {
            return bench_ne(argc, argv);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if ((argc == 4 || argc == 5 || argc == 6) &&
        std::string_view(argv[1]) == "bench-head") {
        try {
            return bench_head(argc, argv);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc == 4 && std::string_view(argv[1]) == "embedding-row") {
        try {
            return embedding_row(argv[2], argv[3]);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if ((argc == 4 || argc == 5) && std::string_view(argv[1]) == "bench-moe") {
        try {
            return bench_moe(argc, argv);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if ((argc == 4 || argc == 5) &&
        std::string_view(argv[1]) == "bench-attention") {
        try {
            return bench_attention(argc, argv);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if ((argc == 4 || argc == 5) && std::string_view(argv[1]) == "bench-linear") {
        try {
            return bench_linear_attention(argc, argv);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc == 4 && std::string_view(argv[1]) == "decode-token") {
        try {
            return decode_one(argv[2], argv[3]);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc == 5 && std::string_view(argv[1]) == "decode-batch") {
        try {
            return decode_batch(argv[2], argv[3], argv[4]);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc == 4 && std::string_view(argv[1]) == "tokenize") {
        try {
            return tokenize(argv[2], argv[3]);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if ((argc == 4 || argc == 5) && std::string_view(argv[1]) == "generate") {
        try {
            return generate_text(argc, argv);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc >= 2 && std::string_view(argv[1]) == "serve") {
        try {
            return serve_command(argc, argv);
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }

    usage();
    return 2;
}
