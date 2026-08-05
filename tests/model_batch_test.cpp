#include "adi/executor.hpp"
#include "adi/model.hpp"
#include "adi/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

int main(int argc, char **argv) {
    assert(argc == 2);
    const adi::MachModel model(argv[1]);
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
}
