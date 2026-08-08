#include "adi/executor.hpp"
#include "adi/model.hpp"
#include "adi/tokenizer.hpp"
#include "model_test_main.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

int ADI_MODEL_TEST_MAIN(
    int argc,
    adi::test_detail::CommandCharacter **argv) {
    assert(argc == 2);
    const adi::MachModel model(adi::test_detail::model_path(argv[1]));
    const adi::Tokenizer tokenizer(model);
    const std::vector<std::uint32_t> tokens{
        tokenizer.bos_token(),
        tokenizer.eos_token(),
    };

    std::vector<adi::DecoderState> serial_states(tokens.size());
    std::vector<adi::DecoderScratch> serial_scratches(tokens.size());
    std::vector<float> serial_logits(
        tokens.size() * model.config().vocabulary);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        adi::decode_token(
            model,
            tokens[index],
            serial_states[index],
            std::span<float>(serial_logits)
                .subspan(
                    index * model.config().vocabulary,
                    model.config().vocabulary),
            serial_scratches[index]);
    }

    std::vector<adi::DecoderState> batch_states(tokens.size());
    std::vector<adi::DecoderScratch> batch_scratches(tokens.size());
    std::vector<float> batch_logits(serial_logits.size());
    adi::DecoderBatchScratch batch_scratch;
    adi::decode_batch(
        model,
        tokens,
        batch_states,
        batch_logits,
        batch_scratches,
        batch_scratch);

    assert(batch_logits == serial_logits);
    for (const auto &state : batch_states) {
        assert(state.position == 1);
    }

    adi::DecoderState serial_sequence_state;
    adi::DecoderScratch serial_sequence_scratch;
    std::vector<float> serial_sequence_logits(model.config().vocabulary);
    adi::decode_token(
        model,
        tokens[0],
        serial_sequence_state,
        {},
        serial_sequence_scratch);
    adi::decode_token(
        model,
        tokens[1],
        serial_sequence_state,
        serial_sequence_logits,
        serial_sequence_scratch);

    adi::DecoderState prefill_state;
    adi::PrefillScratch prefill_scratch;
    std::vector<float> prefill_logits(model.config().vocabulary);
    adi::prefill(
        model,
        tokens,
        prefill_state,
        prefill_logits,
        prefill_scratch);
    assert(prefill_logits == serial_sequence_logits);
    assert(prefill_state.position == serial_sequence_state.position);

    adi::DecoderState chunked_prefill_state;
    adi::PrefillScratch chunked_prefill_scratch;
    std::vector<float> chunked_prefill_logits(model.config().vocabulary);
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        adi::prefill(
            model,
            std::span<const std::uint32_t>(tokens).subspan(index, 1),
            chunked_prefill_state,
            chunked_prefill_logits,
            chunked_prefill_scratch);
    }
    assert(chunked_prefill_logits == prefill_logits);
    assert(chunked_prefill_state.position == prefill_state.position);

    for (std::size_t layer = 0; layer < model.config().layers; ++layer) {
        assert(
            prefill_state.full_attention[layer].keys ==
            serial_sequence_state.full_attention[layer].keys);
        assert(
            prefill_state.full_attention[layer].values ==
            serial_sequence_state.full_attention[layer].values);
        assert(
            prefill_state.linear_attention[layer].convolution ==
            serial_sequence_state.linear_attention[layer].convolution);
        assert(
            prefill_state.linear_attention[layer].recurrent ==
            serial_sequence_state.linear_attention[layer].recurrent);
        assert(
            chunked_prefill_state.full_attention[layer].keys ==
            prefill_state.full_attention[layer].keys);
        assert(
            chunked_prefill_state.full_attention[layer].values ==
            prefill_state.full_attention[layer].values);
        assert(
            chunked_prefill_state.linear_attention[layer].convolution ==
            prefill_state.linear_attention[layer].convolution);
        assert(
            chunked_prefill_state.linear_attention[layer].recurrent ==
            prefill_state.linear_attention[layer].recurrent);
    }

    adi::DecoderState exhausted;
    exhausted.position = model.config().context;
    adi::DecoderScratch exhausted_scratch;
    bool rejected = false;
    try {
        adi::decode_token(
            model,
            tokenizer.bos_token(),
            exhausted,
            {},
            exhausted_scratch);
    } catch (const std::out_of_range &) {
        rejected = true;
    }
    assert(rejected);
    assert(exhausted.position == model.config().context);

    std::vector<adi::DecoderState> mixed_states(tokens.size());
    mixed_states[1].position = model.config().context;
    std::vector<adi::DecoderScratch> mixed_scratches(tokens.size());
    std::vector<float> mixed_logits(
        tokens.size() * model.config().vocabulary);
    rejected = false;
    try {
        adi::decode_batch(
            model,
            tokens,
            mixed_states,
            mixed_logits,
            mixed_scratches,
            batch_scratch);
    } catch (const std::out_of_range &) {
        rejected = true;
    }
    assert(rejected);
    assert(mixed_states[0].position == 0);
    assert(mixed_states[1].position == model.config().context);
}
