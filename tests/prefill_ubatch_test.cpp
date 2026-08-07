// Prefill must produce bit-identical logits and state no matter how the
// prompt is split into physical microbatches. Chunk-boundary handling is
// independent of prompt size, so the default matrix stays small enough to run
// on every commit; "--full" runs the wide matrix used before merging.
#include "adi/executor.hpp"
#include "adi/generation.hpp"
#include "adi/model.hpp"
#include "adi/options.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

std::vector<std::uint32_t> prompt_tokens(
    const adi::MachModel &model,
    std::size_t count) {
    std::vector<std::uint32_t> tokens(count);
    for (std::size_t index = 0; index < count; ++index) {
        std::uint64_t value = index + 0x9E3779B97F4A7C15ULL;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        value ^= value >> 31;
        tokens[index] =
            static_cast<std::uint32_t>(value % model.config().vocabulary);
    }
    return tokens;
}

void assert_same_state(
    const adi::MachModel &model,
    const adi::DecoderState &left,
    const adi::DecoderState &right) {
    assert(left.position == right.position);
    for (std::uint32_t layer = 0; layer < model.config().layers; ++layer) {
        assert(
            left.full_attention[layer].keys ==
            right.full_attention[layer].keys);
        assert(
            left.full_attention[layer].values ==
            right.full_attention[layer].values);
        assert(
            left.linear_attention[layer].convolution ==
            right.linear_attention[layer].convolution);
        assert(
            left.linear_attention[layer].recurrent ==
            right.linear_attention[layer].recurrent);
    }
}

void check_length(
    const adi::MachModel &model,
    std::size_t length,
    std::span<const std::uint32_t> ubatches) {
    const auto tokens = prompt_tokens(model, length);
    const auto vocabulary = model.config().vocabulary;

    // One chunk covering the whole prompt is the reference: it is the shape
    // the executor was written against before the microbatch became tunable.
    adi::ExecutionOptions single;
    single.prefill_ubatch =
        static_cast<std::uint32_t>(length <= 4096 ? length : 4096);
    adi::DecoderState reference_state;
    adi::PrefillScratch reference_scratch;
    std::vector<float> reference_logits(vocabulary);
    adi::prefill_prompt(
        model,
        tokens,
        reference_state,
        reference_logits,
        reference_scratch,
        single);
    assert(reference_state.position == length);

    for (const auto ubatch : ubatches) {
        adi::ExecutionOptions execution;
        execution.prefill_ubatch = ubatch;
        adi::DecoderState state;
        adi::PrefillScratch scratch;
        std::vector<float> logits(vocabulary);
        adi::prefill_prompt(
            model, tokens, state, logits, scratch, execution);
        assert(logits == reference_logits);
        assert_same_state(model, state, reference_state);
        std::printf(
            "length %zu ubatch %u: exact\n",
            length,
            static_cast<unsigned>(ubatch));
        std::fflush(stdout);
    }

    // Every chunk but the last already runs with an empty logits span above.
    // This covers the remaining case: a prompt whose final chunk is also
    // asked to skip the vocabulary head.
    adi::DecoderState stateless;
    adi::PrefillScratch stateless_scratch;
    adi::prefill_prompt(
        model, tokens, stateless, {}, stateless_scratch, single);
    assert_same_state(model, stateless, reference_state);
}

// Anchors the whole comparison chain to the token-at-a-time decode path, so
// "all microbatches agree" cannot mean "all microbatches are equally wrong".
void check_decode_anchor(const adi::MachModel &model, std::size_t length) {
    const auto tokens = prompt_tokens(model, length);
    adi::DecoderState decoded;
    adi::DecoderScratch decode_scratch;
    std::vector<float> decoded_logits(model.config().vocabulary);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const bool last = index + 1 == tokens.size();
        adi::decode_token(
            model,
            tokens[index],
            decoded,
            last ? std::span<float>(decoded_logits) : std::span<float>{},
            decode_scratch);
    }

    adi::ExecutionOptions execution;
    execution.prefill_ubatch = static_cast<std::uint32_t>(length);
    adi::DecoderState prefilled;
    adi::PrefillScratch scratch;
    std::vector<float> prefilled_logits(model.config().vocabulary);
    adi::prefill_prompt(
        model, tokens, prefilled, prefilled_logits, scratch, execution);
    assert(prefilled_logits == decoded_logits);
    assert_same_state(model, prefilled, decoded);
    std::printf("decode anchor at length %zu: exact\n", length);
    std::fflush(stdout);
}

void check_rejected_ubatch(const adi::MachModel &model) {
    const auto tokens = prompt_tokens(model, 1);
    for (const std::uint32_t ubatch : {0U, 4097U}) {
        adi::ExecutionOptions execution;
        execution.prefill_ubatch = ubatch;
        adi::DecoderState state;
        adi::PrefillScratch scratch;
        std::vector<float> logits(model.config().vocabulary);
        bool rejected = false;
        try {
            adi::prefill_prompt(
                model, tokens, state, logits, scratch, execution);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        assert(rejected);
        assert(state.position == 0);
    }
}

} // namespace

int main(int argc, char **argv) {
    assert(argc == 2 || argc == 3);
    const adi::MachModel model(argv[1]);
    const bool full = argc == 3 && std::string_view(argv[2]) == "--full";

    check_rejected_ubatch(model);
    check_decode_anchor(model, 3);

    if (full) {
        const std::uint32_t ubatches[] = {1, 7, 64, 256};
        for (const std::size_t length : {1, 63, 64, 65, 257}) {
            check_length(model, length, ubatches);
        }
    } else {
        const std::uint32_t ubatches[] = {1, 3, 8};
        for (const std::size_t length : {1, 5, 9}) {
            check_length(model, length, ubatches);
        }
    }
}
