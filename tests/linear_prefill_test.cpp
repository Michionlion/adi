// The prefill linear-attention kernel dispatches the worker pool once per
// chunk instead of once per token. It must still produce exactly what the
// same number of token-at-a-time calls produce: the same outputs, the same
// convolution history, and the same Gated DeltaNet recurrent state.
#include "adi/executor.hpp"
#include "adi/model.hpp"
#include "model_test_main.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::uint32_t value_heads = 32;
constexpr std::uint32_t key_heads = 16;
constexpr std::uint32_t head_size = 128;
constexpr std::uint32_t channels = 2 * key_heads * head_size +
                                   value_heads * head_size;

std::uint64_t next_random(std::uint64_t &state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

float random_float(std::uint64_t &state) {
    const auto centered =
        static_cast<std::int32_t>((next_random(state) >> 40) & 0xFFFFFFU) -
        0x800000;
    return static_cast<float>(centered) / 8388608.0F;
}

// A zero state hides any mistake in how the existing history is carried into
// the first token of a chunk, so both paths start from the same nonzero one.
adi::LinearAttentionState seeded_state(std::uint64_t seed) {
    adi::LinearAttentionState state;
    state.convolution.resize(static_cast<std::size_t>(channels) * 3);
    for (auto &value : state.convolution) {
        value = random_float(seed) * 0.5F;
    }
    state.recurrent.resize(
        static_cast<std::size_t>(value_heads) * head_size * head_size);
    for (auto &value : state.recurrent) {
        value = random_float(seed) * 0.25F;
    }
    return state;
}

std::uint32_t first_linear_layer(const adi::MachModel &model) {
    for (std::uint32_t layer = 0; layer < model.config().layers; ++layer) {
        if (!model.layer(layer).full_attention) {
            return layer;
        }
    }
    throw std::runtime_error("model has no linear attention layer");
}

void check_tokens(
    const adi::MachModel &model,
    std::uint32_t layer,
    std::uint32_t tokens) {
    const auto hidden = model.config().hidden;
    std::uint64_t random = 0xC0FFEEULL + tokens;
    std::vector<float> inputs(static_cast<std::size_t>(tokens) * hidden);
    for (auto &value : inputs) {
        value = random_float(random);
    }

    auto reference_state = seeded_state(0x5EED1234ULL);
    std::vector<float> reference(inputs.size());
    adi::LinearAttentionScratch reference_scratch;
    for (std::uint32_t token = 0; token < tokens; ++token) {
        adi::linear_attention_forward(
            model,
            layer,
            std::span<const float>(inputs).subspan(
                static_cast<std::size_t>(token) * hidden, hidden),
            std::span<float>(reference).subspan(
                static_cast<std::size_t>(token) * hidden, hidden),
            reference_state,
            reference_scratch);
    }

    auto state = seeded_state(0x5EED1234ULL);
    std::vector<float> outputs(inputs.size());
    adi::LinearPrefillScratch scratch;
    adi::linear_attention_prefill_chunk(
        model, layer, inputs, tokens, state, scratch, outputs);

    assert(outputs == reference);
    assert(state.convolution == reference_state.convolution);
    assert(state.recurrent == reference_state.recurrent);
    std::printf("linear prefill chunk of %u tokens: exact\n", tokens);
    std::fflush(stdout);
}

} // namespace

int ADI_MODEL_TEST_MAIN(
    int argc,
    adi::test_detail::CommandCharacter **argv) {
    assert(argc == 2);
    const adi::MachModel model(adi::test_detail::model_path(argv[1]));
    const auto layer = first_linear_layer(model);

    // 64 is the default microbatch and 65 crosses it; 7 is a chunk that
    // divides neither the head count nor the worker count evenly.
    for (const std::uint32_t tokens : {1U, 2U, 7U, 16U, 64U, 65U}) {
        check_tokens(model, layer, tokens);
    }

    // A chunk must be rejected rather than silently truncated when the shapes
    // disagree, and must leave the state untouched.
    auto state = seeded_state(0x11223344ULL);
    const auto saved = state;
    adi::LinearPrefillScratch scratch;
    std::vector<float> inputs(model.config().hidden);
    std::vector<float> outputs(model.config().hidden * 2);
    bool rejected = false;
    try {
        adi::linear_attention_prefill_chunk(
            model, layer, inputs, 2, state, scratch, outputs);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    assert(state.convolution == saved.convolution);
    assert(state.recurrent == saved.recurrent);
}
