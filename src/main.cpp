#include "adi/adi.hpp"
#include "adi/executor.hpp"
#include "adi/generation.hpp"
#include "adi/gguf.hpp"
#include "adi/model.hpp"
#include "adi/options.hpp"
#include "adi/profiling.hpp"
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

float stable_benchmark_input(
    std::uint64_t index,
    std::uint64_t stream = 0) noexcept {
    std::uint64_t value =
        index + 0x9E3779B97F4A7C15ULL * (stream + 1);
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    value ^= value >> 31;
    const auto centered =
        static_cast<std::int32_t>((value >> 40) & 0xFFFFFFU) - 0x800000;
    return static_cast<float>(centered) / 8388608.0F;
}

// FNV-1a over raw bit patterns. Unlike a sum it detects reordered or
// individually perturbed elements, which is what exactness checks care about.
std::uint64_t hash_bytes(std::uint64_t hash, const void *data, std::size_t size) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 0x00000100000001B3ULL;
    }
    return hash;
}

std::uint64_t hash_floats(std::uint64_t hash, std::span<const float> values) {
    return hash_bytes(hash, values.data(), values.size() * sizeof(float));
}

std::uint64_t decoder_state_checksum(
    const adi::MachModel &model,
    const adi::DecoderState &state) {
    std::uint64_t hash = 0xCBF29CE484222325ULL;
    hash = hash_bytes(hash, &state.position, sizeof(state.position));
    for (std::uint32_t layer = 0; layer < model.config().layers; ++layer) {
        hash = hash_floats(hash, state.full_attention[layer].keys);
        hash = hash_floats(hash, state.full_attention[layer].values);
        hash = hash_floats(hash, state.linear_attention[layer].convolution);
        hash = hash_floats(hash, state.linear_attention[layer].recurrent);
    }
    return hash;
}

template <typename T>
std::size_t vector_bytes(const std::vector<T> &values) {
    return values.capacity() * sizeof(T);
}

std::size_t expert_scratch_bytes(const adi::ExpertScratch &scratch) {
    return vector_bytes(scratch.input) + vector_bytes(scratch.output) +
           vector_bytes(scratch.wave_indexes) +
           vector_bytes(scratch.state_values) +
           vector_bytes(scratch.wave_gamma);
}

std::size_t decoder_scratch_bytes(const adi::DecoderScratch &scratch) {
    std::size_t bytes = vector_bytes(scratch.hidden) +
                        vector_bytes(scratch.normalized) +
                        vector_bytes(scratch.attention) +
                        vector_bytes(scratch.feed_forward);
    const auto &full = scratch.full_attention;
    bytes += expert_scratch_bytes(full.codec) + vector_bytes(full.query_gate) +
             vector_bytes(full.contiguous_queries) + vector_bytes(full.key) +
             vector_bytes(full.value) + vector_bytes(full.attended) +
             vector_bytes(full.scores);
    const auto &linear = scratch.linear_attention;
    bytes += expert_scratch_bytes(linear.codec) + vector_bytes(linear.qkv) +
             vector_bytes(linear.gate) + vector_bytes(linear.convolved) +
             vector_bytes(linear.alpha) + vector_bytes(linear.beta) +
             vector_bytes(linear.recurrent_output) +
             vector_bytes(linear.normalized);
    const auto &moe = scratch.moe;
    for (const auto &expert : moe.experts) {
        bytes += expert_scratch_bytes(expert.codec) +
                 vector_bytes(expert.gate) + vector_bytes(expert.up) +
                 vector_bytes(expert.activated) + vector_bytes(expert.projected);
    }
    bytes += expert_scratch_bytes(moe.shared_codec) +
             vector_bytes(moe.router_logits) + vector_bytes(moe.shared_gate) +
             vector_bytes(moe.shared_up) + vector_bytes(moe.shared_down);
    return bytes;
}

std::size_t moe_batch_scratch_bytes(const adi::MoeBatchScratch &scratch) {
    return vector_bytes(scratch.router_logits) + vector_bytes(scratch.routes) +
           vector_bytes(scratch.counts) + vector_bytes(scratch.offsets) +
           vector_bytes(scratch.cursors) + vector_bytes(scratch.active) +
           vector_bytes(scratch.grouped_batch) +
           vector_bytes(scratch.route_to_grouped) +
           vector_bytes(scratch.gathered) + vector_bytes(scratch.gate) +
           vector_bytes(scratch.up) + vector_bytes(scratch.activated) +
           vector_bytes(scratch.projected);
}

std::size_t batch_scratch_bytes(const adi::DecoderBatchScratch &scratch) {
    return expert_scratch_bytes(scratch.codec) +
           moe_batch_scratch_bytes(scratch.moe) +
           vector_bytes(scratch.head_inputs) +
           vector_bytes(scratch.head_outputs) +
           vector_bytes(scratch.projection_0) +
           vector_bytes(scratch.projection_1) +
           vector_bytes(scratch.projection_2) +
           vector_bytes(scratch.projection_3) +
           vector_bytes(scratch.projection_4) +
           vector_bytes(scratch.projection_5);
}

std::size_t linear_prefill_scratch_bytes(
    const adi::LinearPrefillScratch &scratch) {
    return expert_scratch_bytes(scratch.codec) + vector_bytes(scratch.qkv) +
           vector_bytes(scratch.gate) + vector_bytes(scratch.alpha) +
           vector_bytes(scratch.beta) + vector_bytes(scratch.convolved) +
           vector_bytes(scratch.recurrent_output) +
           vector_bytes(scratch.normalized);
}

std::size_t prefill_scratch_bytes(const adi::PrefillScratch &scratch) {
    return vector_bytes(scratch.hidden) + vector_bytes(scratch.rope_cosine) +
           vector_bytes(scratch.rope_sine) +
           decoder_scratch_bytes(scratch.token) +
           batch_scratch_bytes(scratch.batch) +
           linear_prefill_scratch_bytes(scratch.linear);
}

template <typename Function>
int profiled_benchmark(Function &&function) {
    adi::reset_kernel_profiles();
    adi::set_kernel_profiling_enabled(true);
    try {
        const int result = function();
        adi::set_kernel_profiling_enabled(false);
        std::cout << "kernel_profile:\n";
        const auto profiles = adi::kernel_profiles();
        for (std::size_t index = 0; index < profiles.size(); ++index) {
            const auto &profile = profiles[index];
            if (profile.calls == 0) {
                continue;
            }
            const auto kind = static_cast<adi::KernelKind>(index);
            std::cout << "  " << adi::kernel_name(kind)
                      << " calls=" << profile.calls
                      << " milliseconds="
                      << static_cast<double>(profile.nanoseconds) / 1.0e6
                      << " work_items=" << profile.work_items << '\n';
        }
        return result;
    } catch (...) {
        adi::set_kernel_profiling_enabled(false);
        throw;
    }
}

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
              << "cpu backend: " << adi::cpu_backend().name << '\n'
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
        input[index] = stable_benchmark_input(index);
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
                stable_benchmark_input(column, batch_index);
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
                stable_benchmark_input(column, batch_index);
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
        input[index] = stable_benchmark_input(index);
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

// Reports how much of the batched MoE's wall time the expert matmuls actually
// account for. Expert kernels run one task per active expert, so if the wall
// time greatly exceeds the summed kernel time divided by the worker count,
// the workers are idling at the barrier rather than computing.
//
// Read the efficiency here as an upper bound, not as the runtime's. Routing
// depends on the input, and no synthetic input reproduces the distribution a
// real prompt produces at depth: this command sees about 180 active experts
// holding a few rows each, while an actual prefill concentrates 512 route
// assignments into about 85 experts, the largest holding roughly 56. The flat
// distribution here measures 0.87 efficiency; the real one measures 0.47.
// Use this to compare scheduling strategies against each other, and measure
// the runtime in situ before drawing conclusions about it.
int bench_moe_batch(int argc, char **argv) {
    const adi::MachModel model(argv[2]);
    const auto layer = parse_u32(argv[3], "layer");
    const auto batch = parse_u32(argv[4], "batch");
    const auto iterations = argc >= 6 ? parse_u32(argv[5], "iterations") : 3;
    const std::string_view pattern = argc == 7 ? argv[6] : "varied";
    if (batch == 0 || iterations == 0) {
        throw std::invalid_argument("batch and iterations must be positive");
    }
    if (pattern != "varied" && pattern != "identical") {
        throw std::invalid_argument("pattern must be varied or identical");
    }
    const auto hidden = model.config().hidden;

    // Routing is what decides how the work splits, and it is sensitive to the
    // input distribution: white-noise vectors spread routes across far more
    // experts than real hidden states do. So inputs come from real embedding
    // rows, normalized the way a decoder layer normalizes its MoE input.
    // "identical" repeats one token, the perfectly balanced case of eight
    // equal tasks; "varied" uses distinct tokens.
    std::vector<float> inputs(static_cast<std::size_t>(batch) * hidden);
    std::vector<float> row(hidden);
    const auto embedding = model.embedding();
    for (std::uint32_t index = 0; index < batch; ++index) {
        const auto token = pattern == "identical"
                               ? 1000U
                               : 1000U + index * 977U % model.config().vocabulary;
        adi::mach_embedding_row(embedding, token, row);
        adi::rms_norm(
            row,
            model.layer(layer).post_attention_norm,
            1.0F,
            1.0e-6F,
            std::span<float>(inputs).subspan(
                static_cast<std::size_t>(index) * hidden, hidden));
    }
    std::vector<float> outputs(inputs.size());
    adi::DecoderBatchScratch scratch;
    adi::moe_forward_batch(model, layer, inputs, outputs, scratch);

    adi::reset_kernel_profiles();
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        adi::moe_forward_batch(model, layer, inputs, outputs, scratch);
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const auto seconds = elapsed / iterations;

    const auto profiles = adi::kernel_profiles();
    const auto expert_nanoseconds =
        profiles[static_cast<std::size_t>(adi::KernelKind::expert)].nanoseconds +
        profiles[static_cast<std::size_t>(adi::KernelKind::expert_batch)]
            .nanoseconds;
    const auto expert_seconds =
        static_cast<double>(expert_nanoseconds) / 1.0e9 / iterations;
    const auto threads = adi::worker_threads();
    const auto balanced = expert_seconds / threads;

    std::uint32_t active = 0;
    std::uint32_t widest = 0;
    std::uint32_t narrowest = std::numeric_limits<std::uint32_t>::max();
    for (const auto count : scratch.moe.counts) {
        if (count == 0) {
            continue;
        }
        ++active;
        widest = std::max(widest, count);
        narrowest = std::min(narrowest, count);
    }
    double checksum = 0.0;
    for (const auto value : outputs) {
        checksum += value;
    }

    std::cout << "pattern: " << pattern << '\n'
              << "batch: " << batch << '\n'
              << "workers: " << threads << '\n'
              << "active_experts: " << active << '\n'
              << "rows_per_expert: " << narrowest << ".." << widest << '\n'
              << "seconds/forward: " << seconds << '\n'
              << "expert_kernel_seconds: " << expert_seconds << '\n'
              << "balanced_seconds: " << balanced << '\n'
              << "parallel_efficiency: " << balanced / seconds << '\n'
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
            input[index] = stable_benchmark_input(index, position);
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
            input[index] = stable_benchmark_input(index, position);
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

int bench_prefill(int argc, char **argv) {
    const adi::MachModel model(argv[2]);
    const auto prompt_tokens = parse_u32(argv[3], "tokens");
    adi::ExecutionOptions execution;
    execution.prefill_ubatch = parse_u32(argv[4], "ubatch");
    const auto iterations = argc == 6 ? parse_u32(argv[5], "iterations") : 1;
    if (prompt_tokens == 0 || iterations == 0) {
        throw std::invalid_argument("tokens and iterations must be positive");
    }
    if (prompt_tokens >= model.config().context) {
        throw std::invalid_argument("tokens exceeds model context");
    }
    adi::validate_execution_options(execution);

    // The token sequence is generated, not tokenized, so the timed region
    // never contains tokenizer work and repeats bit-identically across runs.
    std::vector<std::uint32_t> tokens(prompt_tokens);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        std::uint64_t value =
            index + 0x9E3779B97F4A7C15ULL;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        value ^= value >> 31;
        tokens[index] =
            static_cast<std::uint32_t>(value % model.config().vocabulary);
    }

    std::vector<float> logits(model.config().vocabulary);
    adi::PrefillScratch scratch;
    // One untimed pass faults in every model page and grows every scratch
    // vector, so the measured iterations see a warm, steady-state runtime.
    {
        adi::DecoderState warm_state;
        adi::prefill_prompt(model, tokens, warm_state, logits, scratch, execution);
    }
    adi::reset_kernel_profiles();

    double seconds = 0.0;
    std::uint64_t state_checksum = 0;
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        adi::DecoderState state;
        const auto start = std::chrono::steady_clock::now();
        adi::prefill_prompt(model, tokens, state, logits, scratch, execution);
        seconds += std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start).count();
        state_checksum = decoder_state_checksum(model, state);
    }
    const auto average = seconds / iterations;

    std::cout << std::hex
              << "final_logits_checksum: 0x"
              << hash_floats(0xCBF29CE484222325ULL, logits) << '\n'
              << "state_checksum: 0x" << state_checksum << '\n'
              << std::dec
              << "prompt_tokens: " << prompt_tokens << '\n'
              << "ubatch: " << execution.prefill_ubatch << '\n'
              << "iterations: " << iterations << '\n'
              << "seconds/prefill: " << average << '\n'
              << "prompt_tokens/second: "
              << static_cast<double>(prompt_tokens) / average << '\n'
              << "peak_scratch_bytes: " << prefill_scratch_bytes(scratch)
              << '\n';
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
    adi::ExecutionOptions execution;
    for (int index = 4; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--ubatch" && index + 1 < argc) {
            execution.prefill_ubatch = parse_u32(argv[++index], "ubatch");
        } else if (index == 4) {
            options.max_output_tokens = parse_u32(argument, "max tokens");
        } else {
            throw std::invalid_argument("invalid generate argument");
        }
    }
    adi::validate_execution_options(execution);
    options.temperature = 0.0F;
    const auto start = std::chrono::steady_clock::now();
    const auto result =
        adi::generate(model, tokenizer, argv[3], options, execution);
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
            if (port > 65535) {
                throw std::invalid_argument("port is out of range");
            }
            options.port = static_cast<std::uint16_t>(port);
        } else if (argument == "--ubatch" && index + 1 < argc) {
            options.execution.prefill_ubatch =
                parse_u32(argv[++index], "ubatch");
        } else {
            throw std::invalid_argument("invalid serve argument");
        }
    }
    if (options.model.empty()) {
        throw std::invalid_argument("--model is required");
    }
    adi::validate_execution_options(options.execution);
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
              << "  adi bench-moe-batch MODEL.gguf LAYER BATCH [ITERATIONS]"
                 " [varied|identical]\n"
              << "  adi bench-attention MODEL.gguf LAYER [TOKENS]\n"
              << "  adi bench-linear MODEL.gguf LAYER [TOKENS]\n"
              << "  adi bench-prefill MODEL.gguf TOKENS UBATCH [ITERATIONS]\n"
              << "  adi decode-token MODEL.gguf TOKEN\n"
              << "  adi decode-batch MODEL.gguf TOKEN BATCH\n"
              << "  adi tokenize MODEL.gguf TEXT\n"
              << "  adi generate MODEL.gguf PROMPT [MAX_TOKENS] [--ubatch TOKENS]\n"
              << "  adi serve --model MODEL.gguf [--host ADDRESS] [--port PORT]"
                 " [--ubatch TOKENS]\n"
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
            return profiled_benchmark([&] { return bench_expert(argc, argv); });
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if ((argc == 5 || argc == 6 || argc == 7) &&
        std::string_view(argv[1]) == "bench-ne") {
        try {
            return profiled_benchmark([&] { return bench_ne(argc, argv); });
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if ((argc == 4 || argc == 5 || argc == 6) &&
        std::string_view(argv[1]) == "bench-head") {
        try {
            return profiled_benchmark([&] { return bench_head(argc, argv); });
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
            return profiled_benchmark([&] { return bench_moe(argc, argv); });
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc >= 5 && argc <= 7 &&
        std::string_view(argv[1]) == "bench-moe-batch") {
        try {
            return profiled_benchmark(
                [&] { return bench_moe_batch(argc, argv); });
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if ((argc == 4 || argc == 5) &&
        std::string_view(argv[1]) == "bench-attention") {
        try {
            return profiled_benchmark(
                [&] { return bench_attention(argc, argv); });
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if ((argc == 4 || argc == 5) && std::string_view(argv[1]) == "bench-linear") {
        try {
            return profiled_benchmark(
                [&] { return bench_linear_attention(argc, argv); });
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if ((argc == 5 || argc == 6) &&
        std::string_view(argv[1]) == "bench-prefill") {
        try {
            return profiled_benchmark([&] { return bench_prefill(argc, argv); });
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc == 4 && std::string_view(argv[1]) == "decode-token") {
        try {
            return profiled_benchmark(
                [&] { return decode_one(argv[2], argv[3]); });
        } catch (const std::exception &error) {
            std::cerr << "adi: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc == 5 && std::string_view(argv[1]) == "decode-batch") {
        try {
            return profiled_benchmark(
                [&] { return decode_batch(argv[2], argv[3], argv[4]); });
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
    if (argc >= 4 && std::string_view(argv[1]) == "generate") {
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
